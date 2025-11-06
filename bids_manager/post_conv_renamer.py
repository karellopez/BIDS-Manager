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
import csv
from collections import defaultdict

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


def _build_summary_timeline(bids_root: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Load ``subject_summary.tsv`` and collect acq_time ordered entries.

    The summary contains the acquisition metadata gathered by the GUI.  We
    extract only the rows relevant for BOLD runs and fieldmaps so we can later
    associate each fieldmap with the functional runs acquired after it.
    """

    summary_path = bids_root / ".bids_manager" / "subject_summary.tsv"
    if not summary_path.exists():
        return {}

    timeline: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    try:
        with open(summary_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                bids = str(row.get("BIDS_name", "")).strip()
                if not bids:
                    continue
                session = str(row.get("session", "")).strip()
                key = (bids, session)
                modality = str(row.get("modality", "")).strip().lower()
                container = str(row.get("modality_bids", "")).strip().lower()
                proposed = str(row.get("Proposed BIDS name", "")).strip()
                if not proposed:
                    continue
                acq_time = str(row.get("acq_time", "")).strip()

                entry_type = None
                if modality == "bold" and proposed.endswith("_bold.nii.gz"):
                    entry_type = "bold"
                elif container == "fmap":
                    entry_type = "fmap"

                if entry_type is None:
                    continue

                timeline[key].append(
                    {
                        "type": entry_type,
                        "acq_time": acq_time,
                        "bids_path": proposed,
                    }
                )
    except Exception as exc:
        print(f"Warning: failed to load summary for IntendedFor ordering: {exc}")
        return {}

    for entries in timeline.values():
        entries.sort(key=lambda item: (_acq_sort_key(item["acq_time"]), item["bids_path"]))

    return timeline


def _acq_sort_key(acq_time: str) -> tuple[int, str]:
    """Return a numeric key for sorting acquisition times.

    ``acq_time`` strings are typically formatted as ``HH:MM:SS[.ffffff]``.
    Removing the punctuation yields a sortable integer.  Empty or malformed
    values are pushed to the end so they do not disturb the intended order.
    """

    cleaned = re.sub(r"[^0-9]", "", acq_time or "")
    if not cleaned:
        return (sys.maxsize, acq_time)
    try:
        return (int(cleaned), acq_time)
    except ValueError:
        return (sys.maxsize, acq_time)


def _json_candidates(js: Path, root: Path) -> list[str]:
    """Return relative image paths that may correspond to ``js``.

    HeuDiConv writes JSON sidecars next to NIfTI images with matching basenames.
    We therefore try the ``.nii.gz`` and ``.nii`` variants.
    """

    rel_json = js.relative_to(root)
    base = rel_json.as_posix()[:-5]  # strip trailing '.json'
    return [f"{base}.nii.gz", f"{base}.nii"]


def _format_intended(path: str) -> str:
    """Convert a relative BIDS path into the required ``bids::`` format."""

    return f"bids::{Path(path).name}"


def _assign_intended_from_timeline(
    root: Path,
    timeline: list[dict[str, str]],
) -> dict[str, list[str]]:
    """Create a mapping from fieldmap image paths to BOLD targets.

    Parameters
    ----------
    root:
        Subject or session directory currently being processed.
    timeline:
        Acquisition-ordered entries for the matching subject/session.
    """

    assignments: dict[str, list[str]] = {}
    current_fmap: str | None = None

    for entry in timeline:
        kind = entry["type"]
        rel_path = entry["bids_path"]
        full_path = root / rel_path

        if kind == "fmap":
            current_fmap = rel_path
            assignments.setdefault(rel_path, [])
        elif kind == "bold":
            if current_fmap is None:
                continue
            if "ref" in Path(rel_path).name.lower():
                continue
            if not full_path.exists():
                continue
            assignments.setdefault(current_fmap, []).append(rel_path)

    return assignments


def _update_intended_for(
    root: Path,
    bids_root: Path,
    timeline_map: dict[tuple[str, str], list[dict[str, str]]],
) -> None:
    """Add ``IntendedFor`` entries to fieldmap JSONs under ``root``."""
    fmap_dir = root / "fmap"
    func_dir = root / "func"

    if not fmap_dir.is_dir():
        return

    func_files = [
        f for f in sorted(func_dir.glob("*.nii*")) if f.is_file() and "ref" not in f.name.lower()
    ]

    rel_root = root.relative_to(bids_root)
    parts = rel_root.parts
    sub_id = next((p for p in parts if p.startswith("sub-")), "")
    ses_id = next((p for p in parts if p.startswith("ses-")), "")
    timeline = timeline_map.get((sub_id, ses_id), [])

    has_timeline = bool(timeline)
    assignments = _assign_intended_from_timeline(root, timeline) if has_timeline else {}

    fallback = [f.relative_to(root).as_posix() for f in func_files]

    for js in fmap_dir.glob("*.json"):
        intended_rel: list[str] = []
        if assignments:
            for candidate in _json_candidates(js, root):
                targets = assignments.get(candidate)
                if targets:
                    intended_rel = targets
                    break
        if not intended_rel and fallback and not has_timeline:
            intended_rel = fallback

        formatted = []
        if intended_rel:
            formatted = list(dict.fromkeys(_format_intended(path) for path in intended_rel))

        with open(js, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["IntendedFor"] = formatted
        with open(js, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=4)
            handle.write("\n")
        print(f"Updated IntendedFor in {js.relative_to(bids_root)}")


def add_intended_for(bids_root: Path) -> None:
    """Populate ``IntendedFor`` in all fieldmap JSONs."""
    timeline_map = _build_summary_timeline(bids_root)

    for sub in bids_root.glob("sub-*"):
        if not sub.is_dir():
            continue
        sessions = [s for s in sub.glob("ses-*") if s.is_dir()]
        if sessions:
            for ses in sessions:
                _update_intended_for(ses, bids_root, timeline_map)
        else:
            _update_intended_for(sub, bids_root, timeline_map)


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

