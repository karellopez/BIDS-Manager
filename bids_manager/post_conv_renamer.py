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
from datetime import datetime
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


def _parse_acquisition_time(value):
    """Convert a raw ``AcquisitionTime`` value into seconds past midnight.

    The GUI stores acquisition times in several possible string formats
    (``HH:MM:SS``, ``HHMMSS``, or full ISO timestamps).  This helper normalises
    them so that the chronological order of fieldmaps and functional runs can be
    reconstructed reliably.
    """

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, (list, tuple)) and value:
        value = value[0]

    text = str(value).strip()
    if not text:
        return None

    # ``datetime.fromisoformat`` gracefully handles ``YYYY-mm-ddTHH:MM:SS``
    # style strings.  The resulting ``datetime`` instance keeps fractional
    # seconds, which is important when multiple runs happen during the same
    # minute.
    if "T" in text:
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = None
        if dt is not None:
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000

    if ":" in text:
        parts = text.split(":")
        if len(parts) == 3:
            try:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
            except ValueError:
                pass
            else:
                return hours * 3600 + minutes * 60 + seconds

    match = re.match(r"^(?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})(?P<f>\.\d+)?$", text)
    if match:
        hours = int(match.group("h"))
        minutes = int(match.group("m"))
        seconds = int(match.group("s"))
        fraction = float(match.group("f")) if match.group("f") else 0.0
        return hours * 3600 + minutes * 60 + seconds + fraction

    return None


def _collect_functional_runs(func_dir: Path, root: Path):
    """Gather functional runs alongside their acquisition times.

    Only BOLD runs (``*_bold``) are considered for ``IntendedFor`` entries, and
    reference volumes (``*_sbref``) are skipped.  Each record includes the
    associated NIfTI file name, relative path, and parsed acquisition time.
    """

    runs = []
    for js in sorted(func_dir.glob("*.json")):
        name = js.name.lower()
        if "_bold" not in name or "sbref" in name:
            continue

        with open(js, "r", encoding="utf-8") as f:
            meta = json.load(f)

        acq_time = _parse_acquisition_time(meta.get("AcquisitionTime"))
        nifti = js.with_suffix(".nii.gz")
        if not nifti.exists():
            nifti = js.with_suffix(".nii")
        if not nifti.exists():
            continue

        runs.append(
            {
                "json": js,
                "nifti": nifti,
                "rel_path": nifti.relative_to(root).as_posix(),
                "basename": nifti.name,
                "acq_time": acq_time,
            }
        )
    return runs


def _collect_fieldmaps(fmap_dir: Path):
    """Return fieldmap JSON metadata and acquisition times."""

    fieldmaps = []
    for js in sorted(fmap_dir.glob("*.json")):
        with open(js, "r", encoding="utf-8") as f:
            meta = json.load(f)
        acq_time = _parse_acquisition_time(meta.get("AcquisitionTime"))
        fieldmaps.append({"json": js, "meta": meta, "acq_time": acq_time})
    return fieldmaps


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

    runs = _collect_functional_runs(func_dir, root)
    if not runs:
        return

    fieldmaps = _collect_fieldmaps(fmap_dir)
    if not fieldmaps:
        return

    missing_times = any(run["acq_time"] is None for run in runs) or any(
        fmap["acq_time"] is None for fmap in fieldmaps
    )

    # When acquisition times are unavailable for either side, retain the
    # previous behaviour: associate every BOLD run with every fieldmap.  This
    # guarantees backwards compatibility while still switching to the new
    # ``bids::`` target syntax.
    if missing_times:
        intended_targets = [f"bids::{run['basename']}" for run in runs]
        for fmap in fieldmaps:
            fmap["meta"]["IntendedFor"] = intended_targets
            with open(fmap["json"], "w", encoding="utf-8") as f:
                json.dump(fmap["meta"], f, indent=4)
                f.write("\n")
            print(f"Updated IntendedFor in {fmap['json'].relative_to(bids_root)}")
        return

    runs.sort(key=lambda item: item["acq_time"])
    fieldmaps.sort(key=lambda item: item["acq_time"])

    for idx, fmap in enumerate(fieldmaps):
        start_time = fmap["acq_time"]
        end_time = fieldmaps[idx + 1]["acq_time"] if idx + 1 < len(fieldmaps) else None

        # Select functional runs acquired after the current fieldmap and before
        # the next one (if any).  ``>=`` ensures that runs captured at exactly
        # the same timestamp as the fieldmap are still linked to it.
        assigned_runs = [
            run for run in runs if run["acq_time"] is not None and run["acq_time"] >= start_time
            and (end_time is None or run["acq_time"] < end_time)
        ]

        fmap["meta"]["IntendedFor"] = [f"bids::{run['basename']}" for run in assigned_runs]

        with open(fmap["json"], "w", encoding="utf-8") as f:
            json.dump(fmap["meta"], f, indent=4)
            f.write("\n")

        print(f"Updated IntendedFor in {fmap['json'].relative_to(bids_root)}")


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

