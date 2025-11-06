#!/usr/bin/env python3
"""
post_fmap_rename.py — Fieldmap Renamer (PyCharm-friendly)
---------------------------------------------------------
This script renames fieldmap files in a BIDS dataset so that:
  - echo-1 → _magnitude1
  - echo-2 → _magnitude2
  - plain _fmap → _phasediff
It also **removes** the trailing `_fmap` from the filenames and moves any
``_rep-<n>`` suffix to the end (e.g. ``magnitude1_rep-2``).
Both ``.nii``/``.nii.gz`` images and their JSON sidecars are handled.

After renaming, each fieldmap JSON gains an ``IntendedFor`` field listing
all functional runs in the same subject/session. This allows fMRIPrep and
other BIDS apps to correctly associate fieldmaps with their target EPI
images.

Usage in PyCharm:
  1. Open this script in PyCharm.
  2. Set the BIDS_ROOT path below to your dataset directory.
  3. Run this script (e.g., click ▶️ in the editor).

No CLI arguments required.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re
import sys

# -----------------------------------------------------------------------------
# Configuration: EDIT this path to point to your BIDS dataset
# -----------------------------------------------------------------------------
BIDS_ROOT = Path("/path/to/your/BIDS_dataset")

# -----------------------------------------------------------------------------
# Rename rules based on filename patterns
# -----------------------------------------------------------------------------
RENAME_RULES = [
    # echo-1 → magnitude1
    (re.compile(r"echo[-_]?1", re.I), "magnitude1"),
    # echo-2 → magnitude2
    (re.compile(r"echo[-_]?2", re.I), "magnitude2"),
]
# Match plain '_fmap' before .nii, .nii.gz or .json
FMAP_SUFFIX_RE = re.compile(r"_fmap(?=(\.nii(?:\.gz)?|\.json)$)", re.I)


def _move_rep_suffix(name: str) -> str:
    """Ensure ``_rep-N`` appears after magnitude/phase suffix."""
    name = re.sub(r"(_rep-\d+)(_magnitude[12])", r"\2\1", name)
    name = re.sub(r"(_rep-\d+)(_phasediff)", r"\2\1", name)
    return name


def _parse_time_value(value: Any) -> Optional[float]:
    """Convert a DICOM/BIDS acquisition time representation into seconds.

    ``AcquisitionTime`` can appear in several formats (``HH:MM:SS.sss``,
    ``HHMMSS.sss``, ISO date strings, or even raw seconds). Normalising the
    representation here makes downstream chronological sorting reliable.
    """

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    # ``HH:MM:SS(.ffffff)``
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})(\.\d+)?", text)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        frac = float(match.group(4)) if match.group(4) else 0.0
        return hours * 3600 + minutes * 60 + seconds + frac

    # ``HHMMSS(.ffffff)``
    match = re.fullmatch(r"(\d{2})(\d{2})(\d{2})(\.\d+)?", text)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        frac = float(match.group(4)) if match.group(4) else 0.0
        return hours * 3600 + minutes * 60 + seconds + frac

    # ISO 8601 date time (``YYYY-MM-DDTHH:MM:SS(.ffffff)``)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000

    # Fallback: treat as seconds if it can be interpreted as a float
    try:
        return float(text)
    except ValueError:
        return None


def _extract_acquisition_time(meta: Dict[str, Any]) -> Optional[float]:
    """Return the acquisition time (in seconds) from a sidecar metadata dict."""

    for key in ("AcquisitionTime", "AcqTime", "SeriesTime", "AcquisitionDateTime"):
        if key in meta:
            parsed = _parse_time_value(meta.get(key))
            if parsed is not None:
                return parsed
    return None


def _fieldmap_group_key(json_path: Path) -> str:
    """Return a stable grouping key shared by the JSONs of a single fieldmap."""

    stem = json_path.stem
    # Remove common fmap suffixes so magnitude/phase share the same key.
    for suffix in ("_magnitude1", "_magnitude2", "_phasediff", "_phase1", "_phase2", "_fieldmap"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _format_intended_path(root: Path, bids_root: Path, nifti_name: str) -> str:
    """Create the ``bids::`` path expected in ``IntendedFor`` lists."""

    relative_root = root.relative_to(bids_root).as_posix()
    return f"bids::{relative_root}/{nifti_name}"


def _gather_func_runs(func_dir: Path, root: Path, bids_root: Path) -> List[Dict[str, Any]]:
    """Collect functional runs and their acquisition times."""

    runs: List[Dict[str, Any]] = []
    for nifti in sorted(func_dir.glob("*_bold.nii*")):
        if "sbref" in nifti.name.lower():
            # Reference images are not valid IntendedFor targets.
            continue

        json_path: Optional[Path]
        if nifti.name.endswith(".nii.gz"):
            json_path = nifti.with_name(nifti.name[:-7] + ".json")
        elif nifti.suffix == ".nii":
            json_path = nifti.with_suffix(".json")
        else:
            json_path = None

        meta: Dict[str, Any] = {}
        if json_path and json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        runs.append(
            {
                "nifti_name": nifti.name,
                "json_path": json_path,
                "acq_time": _extract_acquisition_time(meta),
                "intended_path": _format_intended_path(root, bids_root, nifti.name),
            }
        )

    return runs


def _gather_fieldmaps(fmap_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Collect fieldmap JSONs grouped by magnitude/phase pairing."""

    groups: Dict[str, Dict[str, Any]] = {}
    for json_path in sorted(fmap_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        key = _fieldmap_group_key(json_path)
        info = groups.setdefault(
            key,
            {
                "json_paths": [],
                "acq_times": [],
            },
        )
        info["json_paths"].append(json_path)

        acq_time = _extract_acquisition_time(meta)
        if acq_time is not None:
            info["acq_times"].append(acq_time)

    # Finalise the canonical acquisition time for each group using the earliest
    # timestamp recorded across its magnitude/phase files.
    for info in groups.values():
        info["acq_time"] = min(info["acq_times"]) if info["acq_times"] else None

    return groups


def _assign_runs_to_fieldmaps(
    fmap_groups: Dict[str, Dict[str, Any]],
    func_runs: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Determine IntendedFor lists based on acquisition order.

    The returned mapping contains the ``IntendedFor`` entries for every fieldmap
    group. If the chronological data are incomplete, all fieldmaps receive the
    full list of functional runs to preserve previous behaviour.
    """

    intended_all = [run["intended_path"] for run in func_runs]
    if not fmap_groups:
        return {}
    if not func_runs:
        return {key: [] for key in fmap_groups}

    has_complete_fmap_times = all(group.get("acq_time") is not None for group in fmap_groups.values())
    has_complete_run_times = all(run.get("acq_time") is not None for run in func_runs)

    if not (has_complete_fmap_times and has_complete_run_times):
        # Without a full timeline we cannot improve on the previous
        # "all runs for all fieldmaps" strategy, so reuse it.
        return {key: intended_all for key in fmap_groups}

    sorted_fmaps: List[Tuple[str, Dict[str, Any]]] = sorted(
        fmap_groups.items(), key=lambda item: item[1]["acq_time"]
    )
    sorted_runs = sorted(func_runs, key=lambda run: run["acq_time"])

    assignments: Dict[str, List[str]] = {key: [] for key in fmap_groups}
    run_index = 0
    for idx, (key, info) in enumerate(sorted_fmaps):
        start_time = info["acq_time"]
        end_time = (
            sorted_fmaps[idx + 1][1]["acq_time"]
            if idx + 1 < len(sorted_fmaps)
            else float("inf")
        )

        # Skip any functional runs that occurred before this fieldmap.
        while run_index < len(sorted_runs) and sorted_runs[run_index]["acq_time"] < start_time:
            run_index += 1

        assign_index = run_index
        assigned: List[str] = []
        while assign_index < len(sorted_runs) and sorted_runs[assign_index]["acq_time"] < end_time:
            assigned.append(sorted_runs[assign_index]["intended_path"])
            assign_index += 1

        assignments[key] = assigned
        run_index = assign_index

    return assignments

# -----------------------------------------------------------------------------
# Process a single fmap directory
# -----------------------------------------------------------------------------
def process_fmap_dir(fmap_dir: Path) -> None:
    """Rename fieldmap files within ``fmap_dir`` according to BIDS rules."""
    for file in sorted(fmap_dir.iterdir()):
        if not file.is_file():
            continue
        name = file.name
        # apply echo rules
        for pattern, replacement in RENAME_RULES:
            if pattern.search(name) and name.lower().endswith(('.nii', '.nii.gz', '.json')):
                # replace echo tag
                interim = pattern.sub(replacement, name)
                # remove trailing _fmap before extension
                new_name = FMAP_SUFFIX_RE.sub('', interim)
                new_name = _move_rep_suffix(new_name)
                file.rename(fmap_dir / new_name)
                print(f"Renamed: {name} → {new_name}")
                break
        else:
            # apply phase rule for plain fmap (no echo)
            if name.lower().endswith(('.nii', '.nii.gz', '.json')) and '_fmap' in name and not any(rep in name.lower() for rep in ['magnitude1', 'magnitude2']):
                # replace _fmap with _phasediff
                new_name = name.replace('_fmap', '_phasediff')
                new_name = _move_rep_suffix(new_name)
                file.rename(fmap_dir / new_name)
                print(f"Renamed: {name} → {new_name}")

# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------
def post_fmap_rename(bids_root: Path) -> None:
    """Walk ``bids_root`` and apply :func:`process_fmap_dir` to each ``fmap`` folder."""
    if not bids_root.is_dir():
        print(f"Error: '{bids_root}' is not a directory", file=sys.stderr)
        return
    fmap_dirs = list(bids_root.rglob('fmap'))
    if not fmap_dirs:
        print(f"No 'fmap' directories found under {bids_root}")
        return
    for fmap_dir in fmap_dirs:
        process_fmap_dir(fmap_dir)

    # After renaming, populate ``IntendedFor`` in the fieldmap sidecars so
    # downstream tools know which functional runs they apply to.
    add_intended_for(bids_root)

    # Finally, refresh filenames recorded in ``*_scans.tsv`` to match the new
    # fieldmap file names.
    update_scans_tsv(bids_root)


def _update_intended_for(root: Path, bids_root: Path) -> None:
    """Add ``IntendedFor`` entries to fieldmap JSONs under ``root``."""
    # ``root`` points to either ``sub-<id>`` or ``sub-<id>/ses-<id>``
    # within the BIDS dataset. The function expects ``fmap`` and ``func``
    # directories side-by-side inside this folder.
    fmap_dir = root / "fmap"
    func_dir = root / "func"

    # Skip if either directory does not exist (e.g. no functional runs).
    if not (fmap_dir.is_dir() and func_dir.is_dir()):
        return

    func_runs = _gather_func_runs(func_dir, root, bids_root)
    if not func_runs:
        return

    fmap_groups = _gather_fieldmaps(fmap_dir)
    if not fmap_groups:
        return

    assignments = _assign_runs_to_fieldmaps(fmap_groups, func_runs)

    # Apply the computed IntendedFor lists to every JSON that belongs to the
    # same physical fieldmap (magnitude1, magnitude2, phasediff, ...).
    for key, info in fmap_groups.items():
        intended_paths = assignments.get(key, [])
        for js in info["json_paths"]:
            with open(js, "r", encoding="utf-8") as f:
                meta = json.load(f)

            meta["IntendedFor"] = intended_paths

            with open(js, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
                f.write("\n")

            print(f"Updated IntendedFor in {js.relative_to(bids_root)}")


def add_intended_for(bids_root: Path) -> None:
    """Populate ``IntendedFor`` in all fieldmap JSONs."""
    # Walk through all subjects and sessions in the dataset. ``_update_intended_for``
    # handles the actual JSON editing for each folder.
    for sub in bids_root.glob("sub-*"):
        if not sub.is_dir():
            continue
        sessions = [s for s in sub.glob("ses-*") if s.is_dir()]
        if sessions:
            for ses in sessions:
                _update_intended_for(ses, bids_root)
        else:
            _update_intended_for(sub, bids_root)


def _rename_in_scans(tsv: Path, bids_root: Path) -> None:
    """Update file names in a single ``*_scans.tsv`` if needed."""
    import pandas as pd

    df = pd.read_csv(tsv, sep="\t")
    if "filename" not in df.columns:
        return

    changed = False
    for idx, fname in enumerate(df["filename"]):
        path = Path(fname)
        if "fmap" not in path.parts:
            continue
        new_name = path.name
        for pattern, replacement in RENAME_RULES:
            if pattern.search(new_name) and new_name.lower().endswith((".nii", ".nii.gz", ".json")):
                interim = pattern.sub(replacement, new_name)
                new_name = FMAP_SUFFIX_RE.sub("", interim)
                new_name = _move_rep_suffix(new_name)
                break
        else:
            if new_name.lower().endswith((".nii", ".nii.gz", ".json")) and "_fmap" in new_name and not any(rep in new_name.lower() for rep in ["magnitude1", "magnitude2"]):
                new_name = new_name.replace("_fmap", "_phasediff")
                new_name = _move_rep_suffix(new_name)

        if new_name != path.name:
            candidate = tsv.parent / path.parent / new_name
            if candidate.exists():
                df.at[idx, "filename"] = (path.parent / new_name).as_posix()
                changed = True

    if changed:
        df.to_csv(tsv, sep="\t", index=False)
        print(f"Updated {tsv.relative_to(bids_root)}")


def update_scans_tsv(bids_root: Path) -> None:
    """Refresh filenames inside all ``*_scans.tsv`` files."""
    for sub in bids_root.glob("sub-*"):
        if not sub.is_dir():
            continue
        sessions = [s for s in sub.glob("ses-*") if s.is_dir()]
        roots = sessions or [sub]
        for root in roots:
            for tsv in root.glob("*_scans.tsv"):
                _rename_in_scans(tsv, bids_root)

# -----------------------------------------------------------------------------
# Run immediately when executed
# -----------------------------------------------------------------------------
def main() -> None:
    """CLI wrapper around :func:`post_fmap_rename`."""

    import argparse

    parser = argparse.ArgumentParser(description="Rename BIDS fieldmap files")
    parser.add_argument('bids_root', help='Path to BIDS dataset root')
    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    print(f"Starting fieldmap rename in: {bids_root}")
    post_fmap_rename(bids_root)
    print("Done.")


if __name__ == '__main__':
    main()

