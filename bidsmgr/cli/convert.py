"""``bidsmgr-convert`` — drive dcm2niix from the inventory TSV into BIDS.

Closes the CLI loop scan → convert. Orchestration is straight-line code
(architecture.md §12 rule 3 — no Pipeline orchestrator).

Pipeline per (dataset, subject, session)::

    Phase 1   parallel per-series dcm2niix into
              <bids_root>/.tmp_bidsmgr/sub-<id>/<datatype>/

    Phase 2   sequential per-subject post-conv:
              - fixups.fieldmaps.apply_fieldmap_renames
              - fixups.scans_tsv.update_scans_tsv (no-op today)
              - fixups.intended_for.populate_intended_for

    Phase 3   sequential per-subject merge commit:
              new subject  -> os.rename(<staging>/sub-<id>, <bids_root>/sub-<id>)
              existing subject -> merge staged files in (add new ses/datatype;
              keep or, with --overwrite, back-up-and-replace colliding files)

Failure handling: per-series failures don't abort the subject; per-subject
post-conv or atomic-commit failures write a JSON error log to
``<bids_root>/.bidsmgr/errors/sub-<id>_<utcstamp>.json`` capturing the
failed task(s), dcm2niix stderr tails, and traceback. **Staging is always
wiped** at the end of the per-subject block, success or fail
(architecture decision: forensic data goes in the error log, not in
``.tmp_bidsmgr/``).

Reference: super_plan.md §14.5.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from joblib import Parallel, delayed

import bidsmgr

from ..converter import (
    ConvertResult,
    ConvertTask,
    default_backends,
    dispatch,
    select_backend,
)
from ..fixups import (
    apply_fieldmap_renames,
    attach_companion_files,
    enrich_recording_sidecars,
    populate_intended_for,
    update_scans_tsv,
)
from ..recording_meta import (
    RecordingMetaSpec,
    default_spec,
    load_spec,
    resolve_effective,
    scaffold_sidecar_path,
)
from ..util.cancel import OperationCancelled, is_cancelled
from ..util.paths import long_path, safe_path_component
from ._scaffold import ensure_bidsignore, ensure_dataset_description

log = logging.getLogger(__name__)

# Datatypes converted by the mne-bids backend. These rows are run
# sequentially in Phase 1 because ``write_raw_bids`` mutates shared
# dataset files (participants.tsv / scans.tsv) on every call.
_MNE_BIDS_DATATYPES: frozenset[str] = frozenset({"eeg", "meg", "ieeg", "nirs"})


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_convert(
    tsv: Path,
    bids_parent: Path,
    *,
    dataset: Optional[str] = None,
    n_jobs: int = 1,
    overwrite: bool = False,
    dry_run: bool = False,
    dcm2niix_bin: Optional[Path] = None,
    recording_meta: Optional[Path] = None,
    raw_root: Optional[Path] = None,
    skip_residuals: bool = True,
    cancel_check=None,
) -> int:
    """Convert every commit-ready row in ``tsv`` to BIDS under ``bids_parent``.

    Returns 0 on success, non-zero if any subject failed.

    ``recording_meta`` points at an optional recording-metadata JSON. Its
    dataset defaults supply EEG/MEG ``line_freq`` / ``montage`` when the
    inventory cell is blank, and its richer fields (reference, ground,
    filters, device, institution, event maps, ...) are folded into the BIDS
    sidecars after the write. When omitted, a default spec is used
    (``PowerLineFrequency = 50``), preserving prior behaviour.

    ``skip_residuals`` (default True) drops the dcm2niix residual/secondary
    outputs -- derived single-volume duplicates split off one input series
    (e.g. ``..._bolda`` next to ``..._bold``). Pass False to keep them.
    """
    tsv = Path(tsv)
    bids_parent = Path(bids_parent)

    # One-line version banner so a user's conversion log unambiguously
    # records which copy of the package executed (matters when the same
    # machine has both a pip-installed and a dev-checkout install, or
    # when a Windows user pulled but is still running a stale process).
    log.info(
        "bidsmgr %s @ %s | platform=%s py=%s",
        bidsmgr.__version__,
        Path(bidsmgr.__file__).resolve().parent,
        sys.platform,
        sys.version.split()[0],
    )

    # Candidates the EEG/MEG row resolver uses when ``source_file`` is
    # stored relative-to-raw-root in the inventory TSV. We try (in order):
    # explicit ``raw_root`` arg, then the TSV's parent (the GUI's default
    # is to write the TSV at ``<raw_root>/.bidsmgr_scan.tsv``), then the
    # current working directory. The MRI path is unaffected — DICOM files
    # come from the ``files_by_uid`` sidecar in absolute form.
    source_search_roots: tuple[Path, ...] = tuple(
        p for p in [
            Path(raw_root) if raw_root is not None else None,
            tsv.parent,
        ] if p is not None
    )

    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    if "dataset" not in df.columns:
        raise ValueError(
            f"{tsv} has no `dataset` column — re-run `bidsmgr-scan` to "
            "regenerate the inventory with the new column."
        )

    # Defensive in-memory rebuild: the user may have hand-edited the
    # entities JSON (or display cells) in a spreadsheet without running
    # `bidsmgr-rebuild`. Reconcile here so the conversion always sees
    # the freshest BIDS basenames. The TSV file on disk is untouched.
    if "entities" in df.columns:
        from ..inventory.rebuild import rebuild_from_entities
        df, rebuild_report = rebuild_from_entities(df)
        if rebuild_report.rows_updated:
            log.info(
                "in-memory rebuild reconciled %d rows from entities JSON",
                rebuild_report.rows_updated,
            )
        for w in rebuild_report.warnings[:5]:
            log.warning("rebuild: %s", w)

    df = _filter_convertible_rows(df)
    if dataset:
        df = df[df["dataset"] == dataset].copy()

    # The files_by_uid sidecar is only required when the inventory
    # contains MRI rows. EEG/MEG-only inventories store the recording
    # path in each row's ``source_file`` column.
    has_mri_rows = (
        "series_uid" in df.columns
        and (df["series_uid"].astype(str).str.len() > 0).any()
    )
    files_by_uid = _load_files_by_uid_sidecar(tsv, required=has_mri_rows)

    if df.empty:
        log.warning("no rows to convert (after filtering)")
        return 0

    # Recording-metadata spec for EEG/MEG enrichment. Precedence: an explicit
    # --recording-meta path, else the scaffold the scan wrote next to the TSV
    # (auto-discovered), else a default spec (keeps PowerLineFrequency=50).
    if recording_meta is not None:
        spec = load_spec(Path(recording_meta))
        log.info("loaded recording metadata from %s", recording_meta)
    else:
        auto_meta = scaffold_sidecar_path(tsv)
        if auto_meta.exists():
            spec = load_spec(auto_meta)
            log.info("auto-loaded recording-metadata scaffold %s", auto_meta)
        else:
            spec = default_spec()

    backends = default_backends(dcm2niix_bin=dcm2niix_bin)
    # Capture dcm2niix version once for provenance — it's still the
    # primary backend for MRI rows.
    primary = select_backend("mri", dcm2niix_bin=dcm2niix_bin)
    dcm2niix_version = _dcm2niix_version_string(primary.binary)

    n_failed = 0
    cancelled = False
    for dataset_name, dataset_df in df.groupby("dataset"):
        if cancelled:
            break
        if not dataset_name:
            log.warning("skipping %d rows with empty dataset name", len(dataset_df))
            continue

        bids_root = bids_parent / str(dataset_name)
        if not dry_run:
            bids_root.mkdir(parents=True, exist_ok=True)
            ensure_dataset_description(bids_root, generated_by={
                "Name": "bidsmgr",
                "Version": bidsmgr.__version__,
                "Description": "dcm2niix-direct backend",
                "Container": {"Type": "binary", "Tag": dcm2niix_version},
            })
            # Keep .bidsmgr/ + .tmp_bidsmgr/ out of the official bids-validator.
            ensure_bidsignore(bids_root)

        for subject, subject_df in dataset_df.groupby("BIDS_name", sort=True):
            # Stop here if the user requested it. Subjects committed before
            # this point stay (the desired "stop now" behaviour); nothing
            # half-written is left because the commit is per-subject atomic.
            if is_cancelled(cancel_check):
                cancelled = True
                break
            if not subject:
                log.warning("skipping rows with empty BIDS_name")
                continue

            tasks = _build_tasks_for_subject(
                subject_df, bids_root, files_by_uid,
                source_search_roots=source_search_roots,
                skip_residuals=skip_residuals,
                spec=spec,
            )
            if not tasks:
                continue

            if dry_run:
                _print_tasks(tasks)
                continue

            try:
                _convert_subject(
                    tasks, bids_root, backends,
                    n_jobs=n_jobs, overwrite=overwrite,
                    dcm2niix_version=dcm2niix_version,
                    cancel_check=cancel_check,
                    spec=spec,
                )
            except OperationCancelled:
                log.warning("conversion stopped by user (sub-%s not committed)", subject)
                cancelled = True
                break
            except Exception:
                log.exception("subject %s failed during conversion", subject)
                n_failed += 1

    if cancelled:
        log.warning("conversion stopped by user before all subjects were processed")
        return 130  # 128 + SIGINT, conventional "interrupted" code
    return 0 if n_failed == 0 else 1


# ---------------------------------------------------------------------------
# Per-subject pipeline (the three phases)
# ---------------------------------------------------------------------------


def _convert_subject(
    tasks: list[ConvertTask],
    bids_root: Path,
    backends: list,
    *,
    n_jobs: int,
    overwrite: bool,
    dcm2niix_version: str,
    cancel_check=None,
    spec: Optional[RecordingMetaSpec] = None,
) -> None:
    """Run Phases 1–3 for a single (dataset, subject, session) group."""
    subject = tasks[0].subject
    # Subject label is inventory-derived (user-editable). Sanitise into
    # a portable path component so a stray ``|``, ``:`` or trailing dot
    # cannot wedge ``mkdir`` on Windows.
    subj_segment = f"sub-{safe_path_component(subject)}"
    staging = bids_root / ".tmp_bidsmgr" / subj_segment
    staging.mkdir(parents=True, exist_ok=True)

    results: list[ConvertResult] = []
    failed_phase: Optional[str] = None
    exception_obj: Optional[BaseException] = None

    try:
        # Phase 1: parallel per-series dispatch + conversion.
        results = _phase1_parallel_dcm2niix(
            tasks, staging, backends, n_jobs=n_jobs, cancel_check=cancel_check,
        )

        # Phase 2: per-subject post-conv (sequential, fast).
        rename_map = apply_fieldmap_renames(staging)
        n_scans_tsv = update_scans_tsv(staging, rename_map)
        n_intended_for = populate_intended_for(
            staging, subject=subject, session=tasks[0].session,
        )
        # EEG/MEG sidecar/channels/events enrichment from the recording-
        # metadata spec (no-op for MRI-only subjects or when spec is None).
        n_enriched = enrich_recording_sidecars(staging, tasks, spec)
        # Copy any per-row curated companion files (events/beh/stim/...) into
        # the staged tree (place + name only; no conversion).
        n_enriched += attach_companion_files(staging, tasks)
        _prune_empty_dirs(staging)

        # Phase 3: atomic commit. Use the same sanitised segment we
        # built for the staging dir so the commit target name is
        # consistent across OSes.
        target = bids_root / subj_segment
        _merge_commit(staging, target, overwrite=overwrite)
        _write_provenance(
            target, results, rename_map, n_intended_for, n_scans_tsv,
            dcm2niix_version=dcm2niix_version, n_enriched=n_enriched,
        )
        log.info("committed sub-%s to %s", subject, target)

    except OperationCancelled:
        # User Stop, not a failure: no forensic error log. The ``finally``
        # wipes staging so this subject leaves nothing half-written; the
        # caller (run_convert) stops the subject loop.
        raise
    except Exception as exc:
        # Determine which phase blew up: results being non-empty means
        # Phase 1 finished before something went wrong.
        if not results:
            failed_phase = "phase1"
        elif not (bids_root / subj_segment).exists():
            failed_phase = "phase2_or_phase3"
        else:
            failed_phase = "phase3_provenance"
        exception_obj = exc
        _write_error_log(
            bids_root, subject, tasks, results, exc, failed_phase,
        )
        raise
    finally:
        # Always wipe staging (success or fail; forensic data is in the error log).
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        # Try to remove the empty .tmp_bidsmgr container too.
        # mne-bids may have written dataset-level files there (README,
        # dataset_description.json, participants.tsv/json) when it
        # treated `.tmp_bidsmgr/` as its BIDS root for this subject —
        # those are stale (the metadata module writes the canonical
        # versions later at the real BIDS root). Strip them along with
        # any macOS-injected ``.DS_Store`` before rmdir.
        tmp_root = bids_root / ".tmp_bidsmgr"
        if tmp_root.is_dir():
            for stray in tmp_root.iterdir():
                # Preserve any in-progress sub-* dirs (other subjects
                # currently mid-conversion; sequential loop today, but
                # be defensive). Remove plain files only.
                if stray.is_file():
                    try:
                        stray.unlink()
                    except OSError:
                        pass
            try:
                tmp_root.rmdir()
            except OSError:
                pass

        # Surface per-series failures via log even if no exception was raised.
        if exception_obj is None:
            for r in results:
                if not r.success:
                    log.warning(
                        "sub-%s task %s failed: %s",
                        subject, r.task.basename, r.error,
                    )
                    # The dcm2niix stderr tail is captured on the result
                    # but useless if never logged — surface it so the
                    # failure is actionable without digging through the
                    # JSON error log. ``rc=2`` is generic; the tail is
                    # what tells you "MAX_PATH overflow" vs "bad header"
                    # vs "no DICOMs found".
                    tail = (r.dcm2niix_stderr_tail or "").strip()
                    if tail:
                        log.warning(
                            "sub-%s task %s dcm2niix stderr (tail):\n%s",
                            subject, r.task.basename, tail,
                        )


def _phase1_parallel_dcm2niix(
    tasks: list[ConvertTask],
    staging: Path,
    backends: list,
    *,
    n_jobs: int,
    cancel_check=None,
) -> list[ConvertResult]:
    """Run per-series conversion in parallel; return per-task results.

    Backend selection is per-task: each task is dispatched to the
    first registered backend whose ``can_handle()`` returns True.

    A Stop request raises :class:`OperationCancelled` once the in-flight
    task(s) return -- new tasks stop dispatching, the subject's staging is
    wiped by the caller's ``finally``, and nothing is committed.
    """
    from ..util.cancel import OperationCancelled, is_cancelled

    if is_cancelled(cancel_check):
        raise OperationCancelled("conversion cancelled by user")

    # mne-bids' ``write_raw_bids`` rewrites shared dataset files
    # (``participants.tsv`` and the subject's ``*_scans.tsv``) on *every*
    # call. Running several EEG/MEG recordings for one subject in parallel
    # therefore races on those files: one worker truncates+rewrites while
    # another reads, and mne-bids blows up ("loadtxt: input contained no
    # data" -> IndexError). dcm2niix (MRI) and physio tasks write only their
    # own independent outputs and are safe to parallelise. So split the work:
    # mne-bids-backed tasks run sequentially, the rest run in the pool.
    serial = [t for t in tasks if t.datatype in _MNE_BIDS_DATATYPES]
    parallel_tasks = [t for t in tasks if t.datatype not in _MNE_BIDS_DATATYPES]

    out: list[ConvertResult] = []

    for t in serial:
        if is_cancelled(cancel_check):
            raise OperationCancelled("conversion cancelled by user")
        out.append(_phase1_one(backends, t, staging))

    if parallel_tasks:
        if n_jobs == 1 or len(parallel_tasks) == 1:
            for t in parallel_tasks:
                if is_cancelled(cancel_check):
                    raise OperationCancelled("conversion cancelled by user")
                out.append(_phase1_one(backends, t, staging))
        else:
            parallel_backend = "threading" if len(parallel_tasks) < 4 else "loky"
            gen = Parallel(
                n_jobs=n_jobs, backend=parallel_backend, return_as="generator",
            )(delayed(_phase1_one)(backends, t, staging) for t in parallel_tasks)
            for res in gen:
                out.append(res)
                if is_cancelled(cancel_check):
                    raise OperationCancelled("conversion cancelled by user")

    return out


def _phase1_one(backends: list, task: ConvertTask, staging: Path) -> ConvertResult:
    """Picklable single-task wrapper.

    Dispatches to the first matching backend per :func:`dispatch`.
    Catches any exception so a single bad task doesn't take down the
    whole subject.

    ``staging`` is the per-subject root; this wrapper picks the session
    subdir (when the task has a session) so the backend writes to
    ``<staging>/ses-<label>/<datatype>/``.
    """
    # Session label is inventory-derived (user-editable). Run it
    # through ``safe_path_component`` so an illegal-on-Windows char
    # (``|``, ``:``, trailing dot, …) cannot wedge ``mkdir`` here.
    if task.session:
        target_root = staging / f"ses-{safe_path_component(task.session)}"
    else:
        target_root = staging
    target_root.mkdir(parents=True, exist_ok=True)
    try:
        backend = dispatch(backends, task)
    except LookupError as exc:
        return ConvertResult(
            task=task, success=False, error=str(exc),
        )
    try:
        return backend.convert(task, target_root)
    except Exception as exc:
        log.exception("backend %s raised for %s", backend.name, task.basename)
        return ConvertResult(
            task=task, success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _prune_empty_dirs(root: Path) -> None:
    """Remove empty subdirectories under ``root`` (bottom-up).

    Avoids committing empty datatype folders (e.g. ``func/`` left behind
    when its only series failed conversion). ``root`` itself is preserved.
    """
    if not root.is_dir():
        return
    # walk bottom-up so deeper dirs are evaluated before their parents.
    for path in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                # not empty; leave it.
                pass


def _merge_commit(staging_subject: Path, target: Path, *, overwrite: bool) -> None:
    """Commit a staged subject into the dataset, merging into an existing one.

    A brand-new subject is moved in wholesale via the atomic ``os.rename`` fast
    path. When the subject already exists, the staged tree is merged FILE BY FILE
    so adding a new session or datatype lands beside the existing data instead of
    replacing the whole subject (the incremental-conversion case):

    * a staged file with no counterpart on disk is always moved in (the add);
    * a staged file that collides with an existing one is left untouched when
      ``overwrite`` is False (existing data is never overwritten), or backed up to
      ``<bids_root>/.bidsmgr/backup/<sub>_<utcstamp>/<relpath>`` and replaced when
      ``overwrite`` is True.

    Atomicity drops from subject-level to per-file on the merge path; any replaced
    file is backed up first, so an interrupted merge can be reconstructed.
    """
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging_subject, target)
        return

    backup_dir = target.parent / ".bidsmgr" / "backup" / f"{target.name}_{_utc_stamp()}"
    added = replaced = kept = 0
    for src in sorted(staging_subject.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(staging_subject)
        dst = target / rel
        if dst.exists():
            if overwrite:
                bdst = backup_dir / rel
                bdst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst), str(bdst))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                replaced += 1
            else:
                kept += 1  # existing file wins; staged copy dropped with staging
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            added += 1
    log.info(
        "merged into existing %s: %d added, %d replaced, %d kept%s",
        target, added, replaced, kept,
        "" if not kept else " (use --overwrite to replace the kept files)",
    )


# ---------------------------------------------------------------------------
# Task construction from TSV rows
# ---------------------------------------------------------------------------


def _filter_convertible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that are include=1 with a non-empty proposed_basename
    and are not flagged ``bids_guess_skip``.
    """
    out = df.copy()
    if "include" in out.columns:
        out = out[out["include"].astype(str) == "1"]
    if "proposed_basename" in out.columns:
        out = out[out["proposed_basename"].astype(str).str.len() > 0]
    if "bids_guess_skip" in out.columns:
        out = out[~out["bids_guess_skip"].astype(str).str.lower().isin({"true", "1"})]
    return out


def _build_tasks_for_subject(
    subject_df: pd.DataFrame,
    bids_root: Path,
    files_by_uid: dict[str, list[str]],
    *,
    source_search_roots: tuple[Path, ...] = (),
    skip_residuals: bool = True,
    spec: Optional[RecordingMetaSpec] = None,
) -> list[ConvertTask]:
    tasks: list[ConvertTask] = []
    for _, row in subject_df.iterrows():
        task = _row_to_task(
            row, bids_root, files_by_uid,
            source_search_roots=source_search_roots,
            skip_residuals=skip_residuals,
            spec=spec,
        )
        if task is not None:
            tasks.append(task)
    return tasks


def _row_to_task(
    row: pd.Series,
    bids_root: Path,
    files_by_uid: dict[str, list[str]],
    *,
    source_search_roots: tuple[Path, ...] = (),
    skip_residuals: bool = True,
    spec: Optional[RecordingMetaSpec] = None,
) -> Optional[ConvertTask]:
    """Detect MRI vs EEG/MEG row shape and dispatch to the right builder.

    MRI rows have a non-empty ``series_uid``; EEG/MEG rows have a
    non-empty ``source_file`` and a datatype in {eeg, meg, ieeg, nirs}.
    Anything else returns None and the caller skips it.
    """
    series_uid = str(row.get("series_uid", "")).strip()
    if series_uid:
        return _row_to_task_mri(
            row, bids_root, files_by_uid, skip_residuals=skip_residuals,
        )

    source_file = str(row.get("source_file", "")).strip()
    if source_file:
        return _row_to_task_eeg_meg(
            row, bids_root, source_search_roots=source_search_roots, spec=spec,
        )

    return None


def _row_to_task_mri(
    row: pd.Series,
    bids_root: Path,
    files_by_uid: dict[str, list[str]],
    *,
    skip_residuals: bool = True,
) -> Optional[ConvertTask]:
    series_uid = str(row.get("series_uid", "")).strip()
    if not series_uid:
        return None

    # v0.2.5 collapses magnitude+phase fmap rows into one row with the
    # two SeriesInstanceUIDs joined by ``|``. Pull files for every UID.
    uids = [u for u in series_uid.split("|") if u]
    files: list[str] = []
    for uid in uids:
        files.extend(files_by_uid.get(uid, []))
    if not files:
        log.warning(
            "no files_by_uid entry for series %s — skipping (rebuild the "
            "scan inventory and its files_by_uid sidecar)", series_uid,
        )
        return None

    bids_name = str(row.get("BIDS_name", "")).strip()
    subject = bids_name[len("sub-"):] if bids_name.startswith("sub-") else bids_name
    if not subject:
        return None

    raw_session = str(row.get("session", "")).strip()
    session = (
        raw_session[len("ses-"):] if raw_session.startswith("ses-") else (raw_session or None)
    )
    if session == "":
        session = None

    datatype = str(row.get("proposed_datatype", "")).strip()
    suffix = (
        str(row.get("bids_guess_suffix", "")).strip()
        or _suffix_from_basename(str(row.get("proposed_basename", "")))
    )
    basename = str(row.get("proposed_basename", "")).strip()
    if not (datatype and suffix and basename):
        return None

    # Derivatives-bound rows go under <bids_root>/derivatives/<pipeline>/...
    # rather than the live raw tree. v1 doesn't ship the derivatives fixup,
    # so for now skip them with a warning.
    if datatype.startswith("derivatives/"):
        log.info(
            "row series=%s suggests derivatives output (%s); skipping until "
            "the derivatives fixup lands", series_uid, datatype,
        )
        return None

    # Expected outputs differ by suffix. Physio rows produce
    # ``.tsv.gz`` + ``.json`` from the bidsphysio backend; everything
    # else dcm2niix-direct handles produces ``.nii.gz`` + ``.json``,
    # plus DWI extras.
    if suffix == "physio":
        expected_outputs = (".tsv.gz", ".json")
    elif suffix == "dwi":
        expected_outputs = (".nii.gz", ".json", ".bval", ".bvec")
    else:
        expected_outputs = (".nii.gz", ".json")

    # Surface the row's canonical entities dict (the source of truth
    # the user edited) — informational for backends that want to inspect
    # individual entity values without re-parsing the basename.
    entities_dict = _parse_entities_json(row)

    return ConvertTask(
        row_id=series_uid,
        series_uid=series_uid,
        source_files=tuple(Path(p) for p in files),
        dataset=str(row.get("dataset", "")).strip(),
        bids_root=bids_root,
        subject=subject,
        session=session,
        datatype=datatype,
        suffix=suffix,
        entities=entities_dict,
        basename=basename,
        expected_outputs=expected_outputs,
        repetition_type=str(row.get("repetition_type", "")).strip(),
        skip_residuals=skip_residuals,
        companion_files=_parse_companion(row),
    )


def _parse_companion(row: pd.Series) -> tuple[tuple[str, str], ...]:
    """Parse the row's ``companion_files`` JSON column into (suffix, path) pairs.

    Format: a JSON list of ``{"suffix": "events", "path": "/abs/file.tsv"}``.
    Returns ``()`` on missing/malformed input.
    """
    raw = str(row.get("companion_files", "") or "").strip()
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return ()
    out: list[tuple[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                suffix = str(item.get("suffix", "")).strip()
                path = str(item.get("path", "")).strip()
                if suffix and path:
                    out.append((suffix, path))
    return tuple(out)


def _parse_entities_json(row: pd.Series) -> dict[str, str]:
    """Parse the row's ``entities`` JSON column. Returns ``{}`` on error."""
    raw = str(row.get("entities", "")).strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    # Normalise values to str (Pydantic ConvertTask wants dict[str, str]).
    return {str(k): str(v) for k, v in data.items()}


def _row_to_task_eeg_meg(
    row: pd.Series,
    bids_root: Path,
    *,
    source_search_roots: tuple[Path, ...] = (),
    spec: Optional[RecordingMetaSpec] = None,
) -> Optional[ConvertTask]:
    """Build a :class:`ConvertTask` for an EEG/MEG/iEEG/NIRS row.

    EEG/MEG rows carry the recording's path directly in ``source_file``
    (relative to the scan input root or absolute). The mne-bids backend
    reads it via ``mne.io.read_raw`` and writes BIDS via
    ``write_raw_bids``.
    """
    source_file = str(row.get("source_file", "")).strip()
    if not source_file:
        return None

    bids_name = str(row.get("BIDS_name", "")).strip()
    subject = bids_name[len("sub-"):] if bids_name.startswith("sub-") else bids_name
    if not subject:
        return None

    raw_session = str(row.get("session", "")).strip()
    session = (
        raw_session[len("ses-"):] if raw_session.startswith("ses-")
        else (raw_session or None)
    )
    if session == "":
        session = None

    datatype = str(row.get("proposed_datatype", "")).strip().lower()
    if datatype not in {"eeg", "meg", "ieeg", "nirs"}:
        # Either not actually an EEG/MEG row, or the user clobbered
        # the column. Skip.
        return None

    suffix = (
        str(row.get("bids_guess_suffix", "")).strip()
        or _suffix_from_basename(str(row.get("proposed_basename", "")))
        or datatype
    )
    basename = str(row.get("proposed_basename", "")).strip()
    if not basename:
        return None

    # Resolve source_file to an absolute path. The scan stores it
    # relative to the scan input root; the convert input is the BIDS
    # parent — different reference. Try in order: each ``source_search_root``
    # the caller provided (the GUI passes the raw_root explicitly; the
    # CLI auto-fills the TSV's parent), then the current working
    # directory, then ``.resolve()`` as a last resort.
    src_path = Path(source_file)
    if not src_path.is_absolute():
        candidates: list[Path] = []
        for root in source_search_roots:
            candidates.append(Path(root) / src_path)
        candidates.append(Path.cwd() / src_path)
        candidates.append(src_path.resolve())
        for c in candidates:
            if c.exists():
                src_path = c
                break

    # The row's ``entities`` column is the source of truth (the in-memory
    # rebuild ran above so it's already reconciled with display cells).
    # Fall back to individual columns / basename parsing for legacy
    # inventories that don't have the ``entities`` column.
    entities = _parse_entities_json(row)
    if not entities.get("task"):
        task_value = str(row.get("task", "")).strip()
        if task_value:
            entities["task"] = task_value
    if not entities.get("run"):
        run_value = str(row.get("run", "")).strip()
        if not run_value:
            m = re.search(r"_run-(\d+)", basename)
            if m:
                run_value = m.group(1)
        if run_value:
            entities["run"] = run_value

    # mne-bids writes the format-native data file (.edf / .vhdr / .fif /
    # …) plus channels.tsv + datatype JSON. We don't validate the exact
    # extension — the backend collects whatever lands in the datatype dir.
    expected_outputs: tuple[str, ...] = (".json",)

    # Resolve EEG/MEG line_freq + montage. Precedence: the inventory cell
    # wins; else the recording-metadata dataset default (resolved per row by
    # source path); line_freq finally falls back to 50 Hz so
    # PowerLineFrequency is always populated (preserves prior behaviour now
    # that the dataset-wide --line-freq flag is gone).
    eff_acq = resolve_effective(spec, source_file).acquisition if spec is not None else None

    line_freq_raw = str(row.get("line_freq", "")).strip()
    try:
        line_freq = float(line_freq_raw) if line_freq_raw else None
    except ValueError:
        line_freq = None
    if line_freq is None and eff_acq is not None:
        line_freq = eff_acq.power_line_freq
    if line_freq is None:
        line_freq = 50.0

    montage = str(row.get("montage", "")).strip() or None
    if montage is None and eff_acq is not None:
        montage = eff_acq.montage

    # Per-row reference/ground overrides (the spec default applies in the
    # enrichment fixup when these are blank).
    eeg_reference = str(row.get("eeg_reference", "")).strip() or None
    eeg_ground = str(row.get("eeg_ground", "")).strip() or None

    return ConvertTask(
        row_id=source_file,  # source path is unique per recording
        series_uid="",       # blank for EEG/MEG rows
        source_files=(src_path,),
        dataset=str(row.get("dataset", "")).strip(),
        bids_root=bids_root,
        subject=subject,
        session=session,
        datatype=datatype,
        suffix=suffix,
        entities=entities,
        basename=basename,
        expected_outputs=expected_outputs,
        repetition_type=str(row.get("repetition_type", "")).strip(),
        line_freq=line_freq,
        montage=montage,
        eeg_reference=eeg_reference,
        eeg_ground=eeg_ground,
        companion_files=_parse_companion(row),
    )


def _suffix_from_basename(basename: str) -> str:
    """Last underscore-delimited token of a BIDS basename = its suffix."""
    if not basename:
        return ""
    return basename.rsplit("_", 1)[-1]


# ---------------------------------------------------------------------------
# Files_by_uid sidecar
# ---------------------------------------------------------------------------


def _load_files_by_uid_sidecar(
    tsv: Path, *, required: bool = True,
) -> dict[str, list[str]]:
    """Load the per-UID DICOM file map.

    Required for inventories with MRI rows (``series_uid`` populated).
    EEG/MEG-only inventories don't need this sidecar — the recording
    paths live in each row's ``source_file`` column. Pass
    ``required=False`` for those cases; missing sidecar returns ``{}``.
    """
    sidecar = tsv.with_suffix(tsv.suffix + ".files_by_uid.json.gz")
    if not sidecar.exists():
        if required:
            raise FileNotFoundError(
                f"missing files_by_uid sidecar: {sidecar} — "
                "re-run `bidsmgr-scan` to produce it next to the inventory TSV"
            )
        return {}
    with gzip.open(sidecar, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Provenance + error log
# ---------------------------------------------------------------------------


def _write_provenance(
    target: Path,
    results: list[ConvertResult],
    rename_map: dict[Path, Path],
    n_intended_for: int,
    n_scans_tsv: int,
    *,
    dcm2niix_version: str,
    n_enriched: int = 0,
) -> None:
    """Per-subject provenance at ``<target>/.bidsmgr/provenance.json``."""
    prov_dir = target / ".bidsmgr"
    prov_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "generated_at": _utc_stamp(),
        "dcm2niix_version": dcm2niix_version,
        "bidsmgr_version": bidsmgr.__version__,
        "tasks": [
            {
                "row_id": r.task.row_id,
                "series_uid": r.task.series_uid,
                "datatype": r.task.datatype,
                "suffix": r.task.suffix,
                "basename": r.task.basename,
                "outputs": [p.name for p in r.staged_files],
                "n_input_dicoms": len(r.task.source_files),
                "success": r.success,
                "error": r.error,
                "dcm2niix_returncode": r.dcm2niix_returncode,
                "duration_s": r.duration_s,
            }
            for r in results
        ],
        "fieldmap_renames": [
            {"from": old.name, "to": new.name} for old, new in rename_map.items()
        ],
        "intended_for_updated": n_intended_for,
        "scans_tsv_rewritten": n_scans_tsv,
        "recording_sidecars_enriched": n_enriched,
    }
    (prov_dir / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


def _write_error_log(
    bids_root: Path,
    subject: str,
    tasks: list[ConvertTask],
    results: list[ConvertResult],
    exc: BaseException,
    failed_phase: str,
) -> None:
    """Forensic log when a subject fails. Never raises."""
    try:
        err_dir = bids_root / ".bidsmgr" / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        path = err_dir / f"sub-{safe_path_component(subject)}_{_utc_stamp()}.json"
        payload = {
            "schema_version": 1,
            "subject": subject,
            "failed_at_phase": failed_phase,
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "tasks": [_task_dump(t) for t in tasks],
            "results_so_far": [_result_dump(r) for r in results],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        log.error("wrote error log to %s", path)
    except Exception:  # pragma: no cover - logging must not raise
        log.exception("could not write error log")


def _task_dump(t: ConvertTask) -> dict:
    return {
        "row_id": t.row_id, "series_uid": t.series_uid,
        "subject": t.subject, "session": t.session,
        "datatype": t.datatype, "suffix": t.suffix, "basename": t.basename,
        "n_source_files": len(t.source_dicom_files),
        "expected_outputs": list(t.expected_outputs),
        "repetition_type": t.repetition_type,
    }


def _result_dump(r: ConvertResult) -> dict:
    return {
        "basename": r.task.basename,
        "success": r.success, "error": r.error,
        "dcm2niix_returncode": r.dcm2niix_returncode,
        "stderr_tail": r.dcm2niix_stderr_tail,
        "duration_s": r.duration_s,
        "outputs": [p.name for p in r.staged_files],
    }


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------


def _dcm2niix_version_string(binary: Path) -> str:
    """Capture ``dcm2niix --version`` output for provenance."""
    try:
        proc = subprocess.run(
            [str(binary), "-h"], capture_output=True, text=True, timeout=10,
        )
        first = (proc.stdout or proc.stderr).splitlines()[0:2]
        return " ".join(line.strip() for line in first if line.strip()) or "unknown"
    except Exception:
        return "unknown"


def _utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _print_tasks(tasks: Iterable[ConvertTask]) -> None:
    for t in tasks:
        print(
            f"DRY: sub-{t.subject}"
            f"{' ses-' + t.session if t.session else ''} "
            f"{t.datatype}/{t.basename} "
            f"({len(t.source_dicom_files)} files)"
        )


# ---------------------------------------------------------------------------
# argparse entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-convert",
        description=(
            "Convert DICOM series listed in an inventory TSV into BIDS using "
            "dcm2niix. Each distinct `dataset` value in the TSV becomes a "
            "sibling BIDS root under <bids_parent>."
        ),
    )
    parser.add_argument("tsv", help="Inventory TSV produced by `bidsmgr-scan`")
    parser.add_argument(
        "bids_parent",
        help=(
            "Parent directory; each `dataset` value becomes a sibling BIDS "
            "root underneath."
        ),
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Limit conversion to rows with this exact `dataset` value.",
    )
    parser.add_argument(
        "--jobs", "-j", type=int,
        default=max(1, round((os.cpu_count() or 1) * 0.8)),
        help="Number of dcm2niix invocations to run in parallel.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "When a subject already exists, replace colliding files (each is "
            "backed up to <bids_root>/.bidsmgr/backup/ first). Without this, an "
            "existing subject is merged into: new sessions/datatypes are added "
            "and colliding files are kept untouched."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the tasks that would be converted; write nothing.",
    )
    parser.add_argument(
        "--dcm2niix", default=None, type=Path,
        help="Path to a specific dcm2niix binary (defaults to bundled).",
    )
    parser.add_argument(
        "--recording-meta", default=None, type=Path,
        help=(
            "Path to a recording-metadata JSON (EEG/MEG enrichment). Its "
            "dataset defaults supply line_freq / montage for blank inventory "
            "cells, and its richer fields fill sidecar reference / ground / "
            "filters / device / institution, retype auxiliary channels, and "
            "map event codes to labels. Optional; omitting it leaves "
            "PowerLineFrequency=50 by default. Set per-row line_freq / montage "
            "in the inventory TSV columns to override per recording."
        ),
    )
    parser.add_argument(
        "--raw-root", default=None, type=Path,
        help=(
            "Folder the scan was run against. Used as the first "
            "candidate when resolving EEG/MEG rows' relative "
            "``source_file`` paths. If omitted, the TSV's parent "
            "directory is tried (the GUI's default puts the TSV "
            "inside the raw root)."
        ),
    )
    parser.add_argument(
        "--keep-residuals", action="store_true",
        help=(
            "Keep dcm2niix residual/secondary outputs. By default the "
            "converter drops the derived single-volume duplicates dcm2niix "
            "splits off one input series (e.g. ..._bolda alongside ..._bold, "
            "or _Eq_ / _ROI markers). These are not real acquired images and "
            "have no valid BIDS suffix. Pass this flag to keep them."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG)",
    )

    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    return run_convert(
        Path(args.tsv),
        Path(args.bids_parent),
        dataset=args.dataset,
        n_jobs=args.jobs,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        dcm2niix_bin=args.dcm2niix,
        recording_meta=args.recording_meta,
        raw_root=args.raw_root,
        skip_residuals=not args.keep_residuals,
    )


if __name__ == "__main__":
    sys.exit(main())
