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
from pathlib import Path
import json
import re
import sys
from typing import Dict, List, Optional, Tuple

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


def _parse_acq_time(value: str) -> Optional[float]:
    """Return a sortable representation of an ``acq_time`` string.

    The GUI stores acquisition times in ``*_scans.tsv`` as strings such as
    ``"12:34:56.789"``.  To keep the parsing lightweight we simply drop the
    colons and let ``float`` handle the remaining numeric portion.  Invalid
    values (including blanks or ``NaN``) fall back to ``None`` so that callers
    can gracefully degrade to the legacy behaviour of assigning every run to
    every fieldmap.
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    # Remove ``:`` characters while keeping any fractional component.
    numeric = text.replace(":", "")
    try:
        return float(numeric)
    except ValueError:
        return None


def _load_acq_time_lookup(root: Path) -> Dict[str, Optional[float]]:
    """Return a lookup from relative file paths to parsed ``acq_time`` values."""

    lookup: Dict[str, Optional[float]] = {}
    scan_tables = sorted(root.glob("*_scans.tsv"))
    if not scan_tables:
        return lookup

    import pandas as pd  # Local import keeps optional dependency lightweight.

    for table in scan_tables:
        try:
            df = pd.read_csv(table, sep="\t")
        except Exception:  # pragma: no cover - defensive against malformed TSVs
            continue

        if "filename" not in df.columns:
            continue

        acq_series = df.get("acq_time")
        for idx, filename in df["filename"].items():
            rel_path = str(filename).strip()
            if not rel_path:
                continue
            acq_value = None
            if acq_series is not None:
                acq_value = _parse_acq_time(acq_series.iloc[idx])
            # Later tables overwrite earlier ones to match the GUI behaviour.
            lookup[rel_path] = acq_value
    return lookup


def _relative_nifti_paths(json_path: Path, root: Path) -> List[str]:
    """Return possible relative image paths for a JSON sidecar."""

    rel = json_path.relative_to(root).as_posix()
    base = rel[:-5]  # Strip the trailing ``.json``.
    return [f"{base}.nii.gz", f"{base}.nii"]


def _format_intended_entry(path: Path) -> str:
    """Convert a functional run path to the ``bids::`` reference format."""

    return f"bids::{path.name}"


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

    # Collect candidate functional images, skipping reference volumes (``*_sbref``)
    # because they are not intended targets for fieldmap correction.
    func_files = [
        f for f in sorted(func_dir.glob("*.nii*")) if "ref" not in f.name.lower()
    ]
    if not func_files:
        return

    # Build a lookup of acquisition times from the ``*_scans.tsv`` table produced
    # by the GUI.  The keys match the relative paths stored in the table.
    acq_lookup = _load_acq_time_lookup(root)

    fieldmaps: List[Tuple[Path, float]] = []
    func_entries: List[Tuple[Path, float]] = []
    missing_acq_time = False

    # Collect acquisition times for each functional run so that we can order
    # them chronologically.  Missing values trigger a graceful fallback later.
    for func in func_files:
        rel = func.relative_to(root).as_posix()
        acq_time = acq_lookup.get(rel)
        if acq_time is None:
            missing_acq_time = True
        else:
            func_entries.append((func, acq_time))

    # Sort functional runs once so that time-based grouping becomes trivial.
    func_entries.sort(key=lambda item: item[1])

    # Gather fieldmap sidecars and their acquisition times (via the matching
    # image entry in the scans table).
    for js in sorted(fmap_dir.glob("*.json")):
        acq_time = None
        for candidate in _relative_nifti_paths(js, root):
            if candidate in acq_lookup and acq_lookup[candidate] is not None:
                acq_time = acq_lookup[candidate]
                break
        if acq_time is None:
            missing_acq_time = True
        else:
            fieldmaps.append((js, acq_time))

    # If we could not obtain acquisition times, fall back to the historical
    # behaviour (all runs assigned to every fieldmap) but still emit the new
    # ``bids::``-style entries requested by the GUI.
    if missing_acq_time or not fieldmaps or not func_entries:
        rel_paths = [_format_intended_entry(f) for f in func_files]
        for js in fmap_dir.glob("*.json"):
            with open(js, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["IntendedFor"] = rel_paths
            with open(js, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
                f.write("\n")
            print(f"Updated IntendedFor in {js.relative_to(bids_root)}")
        return

    # Ensure fieldmaps follow chronological order so that each one claims the
    # functional runs acquired after it and before the next fieldmap.
    fieldmaps.sort(key=lambda item: item[1])

    func_index = 0
    func_count = len(func_entries)
    for idx, (js, fm_time) in enumerate(fieldmaps):
        next_time = fieldmaps[idx + 1][1] if idx + 1 < len(fieldmaps) else float("inf")

        # Advance past runs that were acquired before the current fieldmap.
        while func_index < func_count and func_entries[func_index][1] < fm_time:
            func_index += 1

        intended: List[str] = []
        scan_idx = func_index
        while scan_idx < func_count and func_entries[scan_idx][1] < next_time:
            intended.append(_format_intended_entry(func_entries[scan_idx][0]))
            scan_idx += 1

        # Persist the updated IntendedFor metadata using the desired format.
        with open(js, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["IntendedFor"] = intended
        with open(js, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
            f.write("\n")
        print(f"Updated IntendedFor in {js.relative_to(bids_root)}")

        func_index = scan_idx


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

