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
from datetime import datetime
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

    def _parse_time_to_seconds(value: str) -> Optional[float]:
        """Return seconds elapsed since midnight for a DICOM-style time string."""

        if not value:
            return None
        text = value.strip()
        if not text:
            return None

        # ``HH:MM:SS(.ffffff)`` → straightforward parsing.
        if ":" in text:
            parts = text.split(":")
            if len(parts) < 2:
                return None
            hours = parts[0]
            minutes = parts[1]
            seconds = parts[2] if len(parts) > 2 else "0"
        else:
            # ``HHMMSS(.ffffff)`` – optional fractional part is appended.
            if "." in text:
                main, frac = text.split(".", 1)
            else:
                main, frac = text, ""
            main = re.sub(r"\D", "", main)
            if len(main) < 4:
                return None
            # Pad short strings (e.g. HHMM) with zeros for seconds.
            main = main.ljust(6, "0")
            hours, minutes, seconds = main[:2], main[2:4], main[4:6]
            if frac:
                seconds = f"{seconds}.{re.sub(r'\D', '', frac)}"

        try:
            h = int(hours)
            m = int(minutes)
            s = float(seconds)
        except ValueError:
            return None

        return h * 3600 + m * 60 + s

    def _parse_datetime_string(value: str) -> Optional[datetime]:
        """Best-effort parser for the wide variety of DICOM date-time formats."""

        if not value:
            return None
        text = value.strip()
        if not text:
            return None

        try:
            # ``datetime.fromisoformat`` handles ISO strings and fractional seconds.
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            pass

        # Remove common separators so we can parse compact ``YYYYMMDDHHMMSS(.f)`` values.
        compact = re.sub(r"[^0-9.]", "", text)
        if len(compact) < 8:
            return None

        # Split main datetime and fractional seconds (if present).
        if "." in compact:
            main, frac = compact.split(".", 1)
        else:
            main, frac = compact, ""

        if len(main) < 8:
            return None

        # Pad the main section so we always have YYYYMMDDHHMMSS for unpacking.
        main = main.ljust(14, "0")
        year = int(main[0:4])
        month = int(main[4:6])
        day = int(main[6:8])
        hour = int(main[8:10])
        minute = int(main[10:12])
        second = int(main[12:14])
        micro = int((frac + "000000")[:6]) if frac else 0
        try:
            return datetime(year, month, day, hour, minute, second, micro)
        except ValueError:
            return None

    def _combine_date_time(date_value: str, time_value: str) -> Optional[datetime]:
        """Construct a :class:`datetime` from independent DICOM date and time strings."""

        if not date_value or not time_value:
            return None
        digits = re.sub(r"\D", "", date_value)
        if len(digits) != 8:
            return None
        try:
            year = int(digits[0:4])
            month = int(digits[4:6])
            day = int(digits[6:8])
        except ValueError:
            return None

        seconds = _parse_time_to_seconds(time_value)
        if seconds is None:
            return None

        hour = int(seconds // 3600)
        minute = int((seconds % 3600) // 60)
        sec_float = seconds - hour * 3600 - minute * 60
        sec_int = int(sec_float)
        micro = int(round((sec_float - sec_int) * 1_000_000))
        try:
            return datetime(year, month, day, hour, minute, sec_int, micro)
        except ValueError:
            return None

    def _acquisition_order(meta: Dict[str, object], fallback_index: int) -> Tuple[int, float, int]:
        """Return a sortable key representing the acquisition order for metadata."""

        for key in ("AcquisitionDateTime", "SeriesDateTime"):
            dt_val = meta.get(key)
            dt = _parse_datetime_string(dt_val) if dt_val else None
            if dt is not None:
                return (0, dt.timestamp(), fallback_index)

        date_keys = ("AcquisitionDate", "SeriesDate", "StudyDate")
        time_keys = ("AcquisitionTime", "SeriesTime")
        for date_key in date_keys:
            date_val = meta.get(date_key)
            if not date_val:
                continue
            for time_key in time_keys:
                time_val = meta.get(time_key)
                if not time_val:
                    continue
                dt = _combine_date_time(date_val, time_val)
                if dt is not None:
                    return (0, dt.timestamp(), fallback_index)

        for time_key in time_keys:
            time_val = meta.get(time_key)
            if not time_val:
                continue
            seconds = _parse_time_to_seconds(time_val)
            if seconds is not None:
                return (1, seconds, fallback_index)

        # Final fallback keeps stable ordering even when no timing metadata exist.
        return (2, float(fallback_index), fallback_index)

    def _matching_json(image_path: Path) -> Path:
        """Return the expected JSON sidecar path for a NIfTI image."""

        name = image_path.name
        if name.endswith(".nii.gz"):
            return image_path.with_name(name[:-7] + ".json")
        if name.endswith(".nii"):
            return image_path.with_suffix(".json")
        return image_path.with_name(name + ".json")

    # Build a sorted representation of functional runs together with their
    # acquisition order.  SBRefs are excluded because they should never appear
    # in the automatic ``IntendedFor`` entries.
    func_info = []
    for idx, func_file in enumerate(sorted(func_dir.glob("*.nii*"))):
        if "ref" in func_file.name.lower():
            continue
        json_path = _matching_json(func_file)
        if not json_path.exists():
            order = (2, float(idx), idx)
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                func_meta = json.load(f)
            order = _acquisition_order(func_meta, idx)
        func_info.append(
            {
                "order": order,
                "rel_path": func_file.relative_to(root).as_posix(),
            }
        )

    if not func_info:
        return

    func_info.sort(key=lambda item: item["order"])

    fmap_jsons = sorted(fmap_dir.glob("*.json"))
    if not fmap_jsons:
        return

    fmap_info = []
    for idx, fmap_json in enumerate(fmap_jsons):
        with open(fmap_json, "r", encoding="utf-8") as f:
            fmap_meta = json.load(f)
        order = _acquisition_order(fmap_meta, idx)
        fmap_info.append({"path": fmap_json, "meta": fmap_meta, "order": order})

    fmap_info.sort(key=lambda item: item["order"])

    # Walk through fieldmaps in acquisition order and collect all functional runs
    # acquired after the current fmap but before the next one (if any).
    for pos, fmap in enumerate(fmap_info):
        current_key = fmap["order"]
        next_key = fmap_info[pos + 1]["order"] if pos + 1 < len(fmap_info) else None

        intended: List[str] = []
        for func in func_info:
            if func["order"] < current_key:
                continue
            if next_key is not None and func["order"] >= next_key:
                continue
            intended.append(func["rel_path"])

        fmap_meta = fmap["meta"]
        fmap_meta["IntendedFor"] = intended
        with open(fmap["path"], "w", encoding="utf-8") as f:
            json.dump(fmap_meta, f, indent=4)
            f.write("\n")
        print(f"Updated IntendedFor in {fmap['path'].relative_to(bids_root)}")


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

