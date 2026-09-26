"""dcm2niix ``BidsGuess`` classifier — improvement_plan.md M1.

dcm2niix already classifies DICOM series via its built-in ``BidsGuess``
heuristics, which are kept up to date with the BIDS spec. We harvest that
classification by running dcm2niix in **sidecar-only** mode (``-b o``) once
per DICOM source folder and parsing the JSON sidecars. The result is the
highest-fidelity DICOM classifier available without writing our own rules.

This is the first feature to land in ``bidsmgr``: it validates the keystone
``schema/`` (because every BidsGuess output is schema-checked) and seeds
the per-row pipeline that the GUI inspector will surface later.

Reference: architecture.md §4.2 layer 1, ``../improvement_plan.md`` M1.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .. import schema
from ..inventory.types import InventoryRow
from ..util.paths import long_path
from .types import Classification

log = logging.getLogger(__name__)

_SHORT_TO_LONG_CACHE: Optional[dict[str, str]] = None


def _short_to_long_entity_map() -> dict[str, str]:
    """Map BIDS short entity names (``"sub"``) to canonical keys (``"subject"``)."""
    global _SHORT_TO_LONG_CACHE
    if _SHORT_TO_LONG_CACHE is None:
        s = schema.get_schema()
        _SHORT_TO_LONG_CACHE = {
            str(s.objects.entities[k].get("name", k)): k
            for k in s.objects.entities.keys()
        }
    return _SHORT_TO_LONG_CACHE


def parse_bids_guess(guess: Sequence[str]) -> tuple[str, dict[str, str], str]:
    """Parse the ``BidsGuess`` field into ``(datatype, entities, suffix)``.

    dcm2niix encodes BidsGuess as ``[<datatype>, "_<key>-<value>_..._<suffix>"]``.
    For the ``"discard"`` datatype the second element is still parseable but
    callers should treat the row as skip-by-default.
    """

    if not guess or len(guess) < 2:
        raise ValueError(f"Malformed BidsGuess: {guess!r}")

    datatype = str(guess[0])
    tail = str(guess[1]).lstrip("_")
    parts = tail.split("_")
    if not parts:
        raise ValueError(f"BidsGuess has no suffix component: {guess!r}")

    suffix = parts[-1]
    short_to_long = _short_to_long_entity_map()
    entities: dict[str, str] = {}
    for chunk in parts[:-1]:
        if "-" not in chunk:
            continue
        short, _, value = chunk.partition("-")
        long_name = short_to_long.get(short, short)
        # dcm2niix encodes ``run-N`` from DICOM SeriesNumber, which is NOT the
        # BIDS-semantic run (BIDS run-N is for repeated acquisitions of the
        # same parameters within a session). Drop it here; the planner /
        # cli/scan re-derives runs cross-row after grouping.
        if long_name == "run":
            continue
        entities[long_name] = value

    return datatype, entities, suffix


def find_dcm2niix() -> Path:
    """Locate the dcm2niix executable.

    Prefers the binary shipped with the ``dcm2niix`` Python package (a pinned
    pip dependency), falling back to ``$PATH``.

    The package's own ``bin_path`` is built as ``<pkgdir>/dcm2niix`` with no
    extension on every platform, but its Windows wheel ships ``dcm2niix.exe``.
    So on Windows the packaged-binary branch never matches and the lookup
    depends entirely on the ``$PATH`` fallback finding the console-script shim
    the wheel drops in ``Scripts/``. That works from an activated venv and fails
    without one — a bundled or embedded interpreter, a desktop shortcut, or
    plain ``<venv>/Scripts/python.exe -m bidsmgr...`` — and the failure is a
    hard ``FileNotFoundError`` that skips the BidsGuess classifier and the probe
    conversion both. Hence the suffix sweep: find the binary the wheel actually
    shipped, rather than depending on how the process was launched.
    """

    try:
        import dcm2niix as _pkg  # type: ignore

        bin_path = Path(getattr(_pkg, "bin_path", "") or "")
        if bin_path.name:
            for candidate in _binary_candidates(bin_path):
                if candidate.is_file():
                    return candidate
    except ImportError:
        pass

    found = shutil.which("dcm2niix")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "dcm2niix executable not found. Install the ``dcm2niix`` pip package "
        "or place the binary on PATH."
    )


def _binary_candidates(bin_path: Path) -> list[Path]:
    """``bin_path`` and the executable suffixes Windows spells it with.

    ``PATHEXT`` is what ``shutil.which`` consults, so honouring it here keeps
    the packaged-binary branch and the ``$PATH`` branch looking for the same
    set of names.
    """
    if os.name != "nt":
        return [bin_path]
    suffixes = [
        ext
        for ext in os.environ.get("PATHEXT", ".EXE;.BAT;.CMD;.COM").split(os.pathsep)
        if ext.strip()
    ]
    if ".EXE" not in {s.upper() for s in suffixes}:
        suffixes.insert(0, ".EXE")
    # The wheel ships a lowercase ``dcm2niix.exe`` and PATHEXT is conventionally
    # uppercase. Windows and macOS both match case-insensitively, so ``.EXE``
    # would resolve and we would return a path naming a file that exists under
    # no such spelling: right binary, wrong name in every log and error after
    # it. Try the spelling the wheel actually ships first.
    suffixes = [".exe"] + [s for s in suffixes if s.lower() != ".exe"]
    return [bin_path] + [bin_path.with_name(bin_path.name + s) for s in suffixes]


def _run_dcm2niix_sidecars(
    dicom_dir: Path,
    output_dir: Path,
    *,
    dcm2niix_bin: Optional[Path] = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Invoke dcm2niix to produce JSON sidecars only.

    Flags:

    * ``-b o``  : sidecar only (no NIfTI written)
    * ``-ba n`` : do not anonymize the sidecar (keeps SeriesInstanceUID)
    * ``-z n``  : no compression (irrelevant for sidecar-only)
    * ``-f %s`` : filename = SERIES NUMBER. **Not ``%j``.**

    ``%j`` is the SeriesInstanceUID, which is ~59 characters of unbounded
    identifier in a path component, and CROSS_PLATFORM_RULES 1.2 forbids
    exactly that: Windows caps a path at 260 characters and dcm2niix does not
    fail politely when handed a longer one. It dies with a stack buffer
    overrun and an EMPTY stderr, so every layer above reads "no sidecars" as
    "nothing to say" and the scan reports success having classified nothing.
    ``converter/backends/dcm2niix_direct`` already says "do not use ``%j``
    here" for the same reason; this call site was missed.

    The series number is short, and it is unique per series within a folder.
    It does not need to be unique beyond that: the join back to an inventory
    row reads ``SeriesInstanceUID`` from INSIDE the JSON, never from the
    filename. Where two pooled studies do collide, dcm2niix's default
    ``-w 2`` adds a suffix rather than overwriting, so no sidecar is lost.
    """

    binary = str(dcm2niix_bin or find_dcm2niix())
    cmd = [
        binary,
        "-b", "o",
        "-ba", "n",
        "-z", "n",
        "-o", str(long_path(output_dir)),
        "-f", "%s",
        str(long_path(dicom_dir)),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _collect_sidecars(directory: Path) -> list[dict]:
    """Read every ``*.json`` in ``directory`` into memory."""
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read sidecar %s: %s", path, exc)
            continue
        data["_sidecar_path"] = str(path)
        out.append(data)
    return out


def canonicalise(datatype: str, suffix: str) -> tuple[str, str]:
    """Resolve a dcm2niix datatype / suffix to the schema's own spelling.

    dcm2niix does not always match the schema's case. PET is the case that
    matters: it emits ``["PET", "PET"]`` while the schema spells both ``pet``,
    so a case-sensitive comparison silently rejects every PET series.

    The resolve is deliberately **case-insensitive rather than lowercasing**:
    the schema spells plenty of suffixes in mixed case (``T1w``, ``FLAIR``,
    ``UNIT1``), so ``.lower()`` would break far more than it fixed. Anything
    with no case-insensitive match is returned untouched, and rejected by
    :func:`_validate_classification` as before.
    """
    if datatype == "discard":
        return datatype, suffix

    dt_map = {d.lower(): d for d in schema.list_datatypes()}
    dt = dt_map.get(datatype.lower(), datatype)
    if dt not in dt_map.values():
        return dt, suffix

    sfx_map = {s.lower(): s for s in schema.list_suffixes(dt)}
    return dt, sfx_map.get(suffix.lower(), suffix)


def looks_like_uid(value: object) -> bool:
    """Whether ``value`` is plausibly a DICOM UID.

    A UID is dot-separated numeric components, and in practice at least
    three of them. This exists because dcm2niix does not always write one.

    Measured 2026-09-26 on a Siemens study whose spectroscopy is stored under
    the STANDARD MR Spectroscopy Storage SOP class
    (``1.2.840.10008.5.1.4.1.1.4.2``): the sidecar's ``SeriesInstanceUID``
    came out as ``133347.357000``, which is the series TIME, not the UID. The
    image series in the same folder got a correct UID, and so did a second
    study whose spectroscopy uses the Siemens private CSA class
    (``1.3.12.2.1107.5.9.1``). So this is specific to that SOP class, and it
    is invisible until a join on the UID quietly matches nothing.
    """
    text = str(value or "")
    if not text:
        return False
    parts = text.split(".")
    return len(parts) >= 3 and all(part.isdigit() for part in parts)


def _validate_classification(datatype: str, suffix: str, entities: dict[str, str]) -> bool:
    """Return ``True`` if the schema accepts this (datatype, suffix, entities) tuple."""
    if datatype == "discard":
        # Not a real datatype — but the row should still produce a Classification
        # marked ``skip=True`` so the GUI can surface the recommendation.
        return True
    if datatype not in schema.list_datatypes():
        return False
    if suffix not in schema.list_suffixes(datatype):
        return False
    allowed = set(schema.allowed_entities(datatype, suffix))
    return all(ent in allowed or ent == "subject" for ent in entities)


def _exit_hint(returncode: int) -> str:
    """A human-readable note for an exit code worth recognising.

    ``0xC0000409`` is the Windows stack buffer overrun dcm2niix dies with
    when handed a path longer than ``MAX_PATH``. It writes nothing to stderr
    on the way out, so without naming it here the log says only that a
    number was not zero.
    """
    codes = {
        3221226505: " (0xC0000409, a Windows stack buffer overrun: this is "
                    "what dcm2niix does when a path exceeds MAX_PATH)",
        3221225477: " (0xC0000005, an access violation)",
        -11: " (SIGSEGV)",
        -6: " (SIGABRT)",
    }
    return codes.get(returncode, "")


def classify_dicom_folder(
    dicom_dir: Path,
    rows: Iterable[InventoryRow],
    *,
    dcm2niix_bin: Optional[Path] = None,
    workdir: Optional[Path] = None,
) -> list[Classification]:
    """Run dcm2niix BidsGuess on ``dicom_dir`` and return classifications.

    Each :class:`InventoryRow` whose ``series_uid`` is present in the sidecar
    output produces one :class:`Classification`. Rows with no matching sidecar
    yield no classification (the caller decides whether to fall back to the
    next classifier in the chain).
    """

    rows = list(rows)
    rows_by_uid: dict[str, list[InventoryRow]] = defaultdict(list)
    for r in rows:
        if r.series_uid:
            rows_by_uid[r.series_uid].append(r)

    # A second index, used only when the sidecar's UID is unusable. Keyed on
    # the series description, and ONLY where that description names exactly
    # one series in this folder: two runs of one protocol share a
    # description, and guessing between them would be worse than not
    # classifying either.
    by_description: dict[str, list[InventoryRow]] = defaultdict(list)
    for r in rows:
        if r.series_description:
            by_description[r.series_description.strip()].append(r)
    unique_by_description = {
        desc: found for desc, found in by_description.items()
        if len({x.series_uid for x in found}) == 1
    }

    use_temp = workdir is None
    if use_temp:
        workdir_ctx = tempfile.TemporaryDirectory()
        out_dir = Path(workdir_ctx.name)
    else:
        out_dir = Path(workdir)
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        proc = _run_dcm2niix_sidecars(dicom_dir, out_dir, dcm2niix_bin=dcm2niix_bin)
        sidecars = _collect_sidecars(out_dir)
        # SAY SO when this produces nothing. The whole classifier chain reads
        # an empty result as "this classifier has no opinion", so a dcm2niix
        # that died leaves a scan that reports success with every MRI row
        # unclassified, and the user sees a feature that stopped working with
        # no error anywhere. That is the shape of every Windows defect in
        # CROSS_PLATFORM_RULES, and it is the reason this branch is loud.
        if rows and not sidecars:
            log.warning(
                "dcm2niix produced NO sidecars for %s, so none of its %d "
                "series could be classified. returncode=%s%s stderr=%r",
                dicom_dir, len(rows), proc.returncode,
                _exit_hint(proc.returncode),
                (proc.stderr or "")[-500:] or "(empty)",
            )
        elif proc.returncode != 0:
            log.warning(
                "dcm2niix returncode=%s for %s%s; stderr=%s",
                proc.returncode, dicom_dir, _exit_hint(proc.returncode),
                (proc.stderr or "")[-500:],
            )
    finally:
        if use_temp:
            workdir_ctx.cleanup()  # type: ignore[name-defined]

    out: list[Classification] = []
    for sidecar in sidecars:
        guess = sidecar.get("BidsGuess")
        if not guess:
            continue
        try:
            datatype, entities, suffix = parse_bids_guess(guess)
        except ValueError as exc:
            log.debug("could not parse BidsGuess %r: %s", guess, exc)
            continue
        # dcm2niix's casing is not always the schema's (it emits "PET"/"PET").
        datatype, suffix = canonicalise(datatype, suffix)

        uid = sidecar.get("SeriesInstanceUID")
        matching_rows = rows_by_uid.get(uid, [])
        if not matching_rows and not looks_like_uid(uid):
            # dcm2niix wrote something that is not a UID, so the join was
            # never going to land. Fall back to the series description, which
            # it reports correctly, and say so: a scan that silently
            # classified nothing is the symptom this repairs.
            description = str(sidecar.get("SeriesDescription") or "").strip()
            matching_rows = unique_by_description.get(description, [])
            if matching_rows:
                log.info(
                    "dcm2niix wrote %r as the SeriesInstanceUID for %r, which "
                    "is not a UID; matched on the series description instead",
                    uid, description,
                )
        if not matching_rows:
            log.debug("BidsGuess sidecar with no matching inventory row: uid=%s", uid)
            continue

        skip = (datatype == "discard")
        valid = _validate_classification(datatype, suffix, entities)
        if not valid:
            log.info(
                "BidsGuess output rejected by schema: datatype=%s suffix=%s entities=%s",
                datatype, suffix, entities,
            )
            continue

        rationale = f"dcm2niix BidsGuess: {list(guess)}"
        confidence = 0.0 if skip else 0.85

        for row in matching_rows:
            out.append(
                Classification(
                    row_id=row.row_id,
                    classifier="dcm2niix_bidsguess",
                    datatype=datatype,
                    suffix=suffix,
                    candidate_entities=dict(entities),
                    confidence=confidence,
                    rationale=rationale,
                    skip=skip,
                )
            )
    return out


def classify(
    rows: Iterable[InventoryRow],
    *,
    dcm2niix_bin: Optional[Path] = None,
) -> list[Classification]:
    """Top-level classifier entry point.

    Groups MRI rows by their containing folder (``row.source.parent`` if
    ``source`` is a file, ``row.source`` if it is a directory) and dispatches
    one dcm2niix invocation per folder.
    """

    rows = [r for r in rows if r.modality == "mri" and r.series_uid]
    if not rows:
        return []

    groups: dict[Path, list[InventoryRow]] = defaultdict(list)
    for r in rows:
        folder = r.source if r.source.is_dir() else r.source.parent
        groups[folder].append(r)

    out: list[Classification] = []
    for folder, group_rows in groups.items():
        out.extend(classify_dicom_folder(folder, group_rows, dcm2niix_bin=dcm2niix_bin))
    return out


__all__ = [
    "looks_like_uid",
    "classify",
    "classify_dicom_folder",
    "parse_bids_guess",
    "find_dcm2niix",
]
