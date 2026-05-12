#!/usr/bin/env python3
"""Drive dcm2bids 3.x programmatically, per (study, subject).

The dcm2bids analogue of :mod:`bids_manager.run_heudiconv_from_heuristic`.
This module:

1. Reads the inventory TSV (the same one that fed the config builder).
2. Groups rows by ``StudyDescription`` and ``BIDS_name``.
3. Calls :class:`dcm2bids.Dcm2BidsGen` once per (study, subject), passing
   the per-study JSON config and the subject's DICOM source folder.
4. Optionally converts physio series via ``bidsphysio`` after each subject.

After this stage the output directory still needs the existing fieldmap /
``IntendedFor`` / DWI-derivatives fixups in :mod:`post_conv_renamer` plus
the new schema-driven metadata generator in :mod:`bids_metadata_engine`.
We keep those steps separate so the GUI can compose them.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from dcm2bids.dcm2bids_gen import Dcm2BidsGen

try:
    from bids_manager.schema_renamer import normalize_study_name
    from bids_manager.run_heudiconv_from_heuristic import (
        _is_included,
        _resolve_physio_dicom,
        _resolve_physio_source_folder,
        clean_name,
        is_dicom_file,
        safe_stem,
    )
except Exception:  # pragma: no cover - direct script execution
    from schema_renamer import normalize_study_name  # type: ignore
    from run_heudiconv_from_heuristic import (  # type: ignore
        _is_included,
        _resolve_physio_dicom,
        _resolve_physio_source_folder,
        clean_name,
        is_dicom_file,
        safe_stem,
    )


def _normalize_subject(bids_name: str) -> str:
    s = str(bids_name or "").strip()
    if s.lower().startswith("sub-"):
        s = s[4:]
    return re.sub(r"[^0-9A-Za-z]+", "", s)


def _normalize_session(session: str) -> str:
    s = str(session or "").strip()
    if s.lower().startswith("ses-"):
        s = s[4:]
    return re.sub(r"[^0-9A-Za-z]+", "", s)


def _common_dicom_dir(raw_root: Path, source_folders: Iterable[str]) -> Path:
    """Return the deepest folder under ``raw_root`` that contains every subject row.

    When all source_folder values share a common prefix that resolves to a real
    directory we point dcm2bids at the smallest such directory. Otherwise we
    fall back to ``raw_root`` so the recursive walk still finds everything.
    """

    paths: List[Path] = []
    for sf in source_folders:
        sf = str(sf or "").strip()
        if not sf:
            continue
        candidate = _resolve_physio_source_folder(raw_root, sf)
        if candidate.exists():
            paths.append(candidate)
    if not paths:
        return raw_root

    common = paths[0]
    for p in paths[1:]:
        try:
            common = Path(*Path(_common_path(common, p)).parts)
        except Exception:
            return raw_root
    if common.exists() and common.is_dir():
        return common
    return raw_root


def _common_path(a: Path, b: Path) -> Path:
    parts_a = a.parts
    parts_b = b.parts
    out = []
    for x, y in zip(parts_a, parts_b):
        if x != y:
            break
        out.append(x)
    return Path(*out) if out else Path("/")


def _convert_physio_for_subject(raw_root: Path,
                                bids_out: Path,
                                subj_df: pd.DataFrame,
                                study_name: str) -> None:
    """Run bidsphysio on physio rows for a single subject.

    BIDS templates are derived from ``schema_renamer.build_preview_names``
    (the same naming oracle the config builder uses) rather than from a
    heudiconv heuristic module. Mirrors the contract of
    ``run_heudiconv_from_heuristic.convert_physio_series`` so the rest of
    the pipeline behaves identically.
    """

    if "modality" not in subj_df.columns:
        return
    physio_df = subj_df[subj_df["modality"].astype(str).str.lower() == "physio"]
    if physio_df.empty:
        return
    if "include" in physio_df.columns:
        physio_df = physio_df[physio_df["include"].apply(_is_included)]
    if physio_df.empty:
        return

    try:
        from bidsphysio.dcm2bids import dcm2bidsphysio
    except Exception as exc:
        print(f"bidsphysio not available; skipping physio for {study_name}: {exc}")
        return

    try:
        from bids_manager.schema_renamer import (
            DEFAULT_SCHEMA_DIR,
            SeriesInfo,
            build_preview_names,
            load_bids_schema,
        )
    except Exception:
        from schema_renamer import (  # type: ignore
            DEFAULT_SCHEMA_DIR,
            SeriesInfo,
            build_preview_names,
            load_bids_schema,
        )

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)

    converted: set = set()
    for _, row in physio_df.iterrows():
        subj = _normalize_subject(row.get("BIDS_name", ""))
        ses = _normalize_session(row.get("session", "") or "")
        sequence = str(row.get("sequence", ""))
        modality = str(row.get("modality", ""))
        extra: Dict[str, str] = {}
        for key in ("task", "task_hits", "acq", "run", "dir", "echo"):
            val = row.get(key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if str(val).strip():
                extra[key] = str(val)
        series = SeriesInfo(
            subject=subj,
            session=ses or None,
            modality=modality,
            sequence=sequence,
            rep=None,
            extra=extra,
        )
        datatype, base = build_preview_names([series], schema)[0][1:]

        # Build the on-disk template path that bidsphysio expects (no extension).
        path_parts = [f"sub-{subj}"]
        if ses:
            path_parts.append(f"ses-{ses}")
        path_parts.append(datatype)
        prefix_path = bids_out / Path(*path_parts) / base

        source_folder = str(row.get("source_folder", ""))
        dicom_dir = _resolve_physio_source_folder(raw_root, source_folder)
        if not dicom_dir.exists():
            print(f"Physio source folder missing: {dicom_dir}")
            continue
        dicom_path = _resolve_physio_dicom(raw_root, dicom_dir, row)
        if dicom_path is None:
            print(f"No DICOM physiolog files found in {dicom_dir}; skipping.")
            continue

        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        prefix_str = str(prefix_path)
        if prefix_str in converted:
            continue
        print(f"Converting physio {dicom_path} → {prefix_path}")
        physio_data = dcm2bidsphysio.dcm2bids(dicom_path)
        physio_data.save_to_bids_with_trigger(prefix_str)
        converted.add(prefix_str)


def _staged_subject_dir(raw_root: Path, subj_df: pd.DataFrame, stage_root: Path,
                       subj_label: str) -> Path:
    """Create a staging directory of symlinks containing this subject's DICOMs.

    dcm2bids walks recursively and runs dcm2niix against everything it finds.
    To avoid converting other subjects' series (which are wasted work and can
    confuse criteria matching when SeriesInstanceUIDs overlap across studies)
    we materialise a symlink tree that contains only this subject's source
    folders. Symlinks are zero-copy on POSIX and broken-fallback to copy on
    platforms that don't support them.
    """
    subj_root = stage_root / subj_label
    if subj_root.exists():
        shutil.rmtree(subj_root)
    subj_root.mkdir(parents=True)

    folders = sorted({str(f) for f in subj_df.get("source_folder", []) if str(f).strip()})
    for sf in folders:
        src = _resolve_physio_source_folder(raw_root, sf)
        if not src.exists():
            print(f"  ! source folder missing, skipping: {src}")
            continue
        link = subj_root / Path(sf).name
        # Avoid clobbering when two rows share a folder basename.
        n = 1
        while link.exists():
            n += 1
            link = subj_root / f"{Path(sf).name}_{n}"
        try:
            link.symlink_to(src, target_is_directory=True)
        except OSError:
            shutil.copytree(src, link)
    return subj_root


def run(tsv: Path,
        config_dir: Path,
        raw_root: Path,
        bids_out: Path,
        log_level: str = "WARNING") -> List[Path]:
    """Run dcm2bids for every (study, subject) implied by the inventory TSV.

    Parameters
    ----------
    tsv:
        ``subject_summary.tsv`` produced by ``dicom_inventory``.
    config_dir:
        Folder holding the ``dcm2bids_config_<study>.json`` files written by
        :mod:`build_dcm2bids_config`.
    raw_root:
        Root of the original DICOM tree.
    bids_out:
        Root where per-study BIDS datasets will be written
        (``bids_out/<study>/sub-XXX/...``).

    Returns
    -------
    list of paths
        The per-study BIDS roots that were created/updated.
    """

    df = pd.read_csv(tsv, sep="\t", keep_default_na=False)
    if "StudyDescription" in df.columns:
        df["StudyDescription"] = df["StudyDescription"].apply(normalize_study_name)

    df_inc = df[df.get("include", 1).apply(_is_included)] if "include" in df.columns else df

    bids_out.mkdir(parents=True, exist_ok=True)

    stage_root = bids_out / ".dcm2bids_stage"
    stage_root.mkdir(parents=True, exist_ok=True)

    study_roots: List[Path] = []
    try:
        for study, study_df in df_inc.groupby("StudyDescription"):
            fname = safe_stem(study or "unknown")
            cfg = config_dir / f"dcm2bids_config_{fname}.json"
            if not cfg.exists():
                print(f"  ! No config for study {study!r}; expected {cfg}")
                continue
            study_root = bids_out / fname
            study_root.mkdir(parents=True, exist_ok=True)
            study_roots.append(study_root)

            for bids_name, subj_df in study_df.groupby("BIDS_name"):
                subj = _normalize_subject(bids_name)
                if not subj:
                    print(f"  ! Skipping row group with empty BIDS_name for {study}")
                    continue

                # Use a session label only when the inventory has a single
                # well-defined session for this subject. Otherwise let
                # dcm2bids treat all rows as session-less.
                sessions = sorted({_normalize_session(s) for s in subj_df.get("session", []) if str(s).strip()})
                ses_arg = sessions[0] if len(sessions) == 1 else ""

                stage_dir = _staged_subject_dir(raw_root, subj_df, stage_root,
                                                f"{fname}__sub-{subj}")
                print(f"\n→ dcm2bids: study={study} sub-{subj} session={ses_arg or '(none)'}\n"
                      f"   config={cfg}\n   dicom={stage_dir}\n   out={study_root}")
                try:
                    gen = Dcm2BidsGen(
                        dicom_dir=str(stage_dir),
                        participant=subj,
                        config=str(cfg),
                        output_dir=str(study_root),
                        session=ses_arg,
                        clobber=False,
                        force_dcm2bids=False,
                        log_level=log_level,
                    )
                    gen.run()
                except Exception as exc:
                    print(f"  ! dcm2bids failed for sub-{subj}: {exc}")
                    continue

                _convert_physio_for_subject(raw_root, study_root, subj_df, fname)
    finally:
        # Clean up staging trees; ignore failures so a partial run leaves a
        # debuggable trace if the user needs to investigate.
        try:
            shutil.rmtree(stage_root)
        except Exception:
            pass

    return study_roots


def main() -> None:
    """CLI for the ``run-dcm2bids`` entry point."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Run dcm2bids using configs generated from a TSV inventory"
    )
    parser.add_argument("tsv", help="Path to subject_summary.tsv")
    parser.add_argument("config_dir", help="Directory containing dcm2bids_config_*.json")
    parser.add_argument("raw_root", help="Root of the DICOM tree")
    parser.add_argument("bids_out", help="Root for BIDS output")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="dcm2bids log level (default: WARNING)",
    )
    args = parser.parse_args()
    run(
        Path(args.tsv),
        Path(args.config_dir),
        Path(args.raw_root),
        Path(args.bids_out),
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
