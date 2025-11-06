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

    # Collect paths of all functional images relative to the subject/session
    # directory so that ``IntendedFor`` entries omit the ``sub-*`` prefix.
    # Skip reference volumes (e.g. ``*_sbref``) as they should not appear in the
    # ``IntendedFor`` lists.
    func_files = [
        f for f in sorted(func_dir.glob("*.nii*")) if "ref" not in f.name.lower()
    ]
    if not func_files:
        return

    rel_paths = [f.relative_to(root).as_posix() for f in func_files]

    # Update each JSON sidecar under ``fmap`` with the collected paths.
    for js in fmap_dir.glob("*.json"):
        with open(js, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["IntendedFor"] = rel_paths
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

