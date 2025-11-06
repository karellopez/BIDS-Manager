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
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

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


def _safe_stem(text: str) -> str:
    """Return filesystem-friendly identifier used for study folders."""

    value = "" if text is None else str(text)
    value = value.strip()
    if not value:
        return ""
    return re.sub(r"[^0-9A-Za-z_-]+", "_", value).strip("_")


def _parse_acq_time(raw: object) -> Optional[float]:
    """Convert an ``acq_time`` entry into a sortable float value."""

    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    numeric = text.replace(":", "")
    try:
        value = float(numeric)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _register_acq_entry(acq_map: Dict[str, float], name: str, acq_value: float) -> None:
    """Record ``acq_value`` for ``name`` and its common derivatives."""

    if not name:
        return

    base = Path(name).name
    lower = base.lower()
    acq_map[base] = acq_value

    if lower.endswith(".nii.gz"):
        stem = base[:-7]
        acq_map.setdefault(stem, acq_value)
        acq_map.setdefault(f"{stem}.nii", acq_value)
        acq_map.setdefault(f"{stem}.json", acq_value)
    elif lower.endswith(".nii"):
        stem = base[:-4]
        acq_map.setdefault(stem, acq_value)
        acq_map.setdefault(f"{stem}.json", acq_value)
        acq_map.setdefault(f"{stem}.nii.gz", acq_value)
    elif lower.endswith(".json"):
        stem = base[:-5]
        acq_map.setdefault(stem, acq_value)
        acq_map.setdefault(f"{stem}.nii.gz", acq_value)
        acq_map.setdefault(f"{stem}.nii", acq_value)
    else:
        acq_map.setdefault(base, acq_value)


def _load_summary_table(tsv_path: Path) -> Optional["pd.DataFrame"]:
    """Load the cached subject summary generated during scanning."""

    import pandas as pd  # Absolute import required by repository guidelines.

    try:
        return pd.read_csv(tsv_path, sep="\t", keep_default_na=False)
    except Exception:
        return None


def _filter_summary_for_dataset(
    summary: "pd.DataFrame", dataset_name: str
) -> Optional["pd.DataFrame"]:
    """Restrict ``summary`` rows to the dataset represented by ``dataset_name``."""

    if summary is None or summary.empty:
        return None

    target = _safe_stem(dataset_name).lower()
    if not target:
        return None

    if "StudyDescription" not in summary.columns:
        return summary.copy()

    mask = summary["StudyDescription"].apply(_safe_stem).str.lower() == target
    subset = summary[mask]
    if subset.empty:
        return None
    return subset.copy()


def _filter_summary_for_subject(
    summary: "pd.DataFrame", subject: str, session: Optional[str]
) -> Optional["pd.DataFrame"]:
    """Return rows for ``subject``/``session`` from ``summary``."""

    if summary is None or summary.empty:
        return None

    if "BIDS_name" not in summary.columns:
        return None

    lower_sub = subject.lower()
    work = summary.copy()
    subj_mask = work["BIDS_name"].fillna("").astype(str).str.strip().str.lower() == lower_sub
    work = work[subj_mask]
    if work.empty:
        return None

    session_series = work.get("session")
    if session:
        lower_ses = session.lower()
        if session_series is None:
            return None
        ses_mask = session_series.fillna("").astype(str).str.strip().str.lower() == lower_ses
        work = work[ses_mask]
    else:
        if session_series is not None:
            ses_mask = session_series.fillna("").astype(str).str.strip() == ""
            work = work[ses_mask]

    if work.empty:
        return None
    return work.copy()


def _collect_metadata_from_summary(
    table: "pd.DataFrame",
) -> Tuple[Dict[str, float], List[Tuple[float, Path]]]:
    """Extract acquisition times and BOLD ordering from ``table``."""

    acq_map: Dict[str, float] = {}
    bold_lookup: Dict[str, Tuple[float, Path]] = {}

    for _, row in table.iterrows():
        proposed = str(row.get("Proposed BIDS name", "")).strip()
        if not proposed:
            continue

        acq_value = _parse_acq_time(row.get("acq_time"))
        if acq_value is None:
            continue

        _register_acq_entry(acq_map, proposed, acq_value)

        base_name = Path(proposed).name
        lower = base_name.lower()
        if "ref" in lower:
            continue
        if lower.endswith(("_bold.nii", "_bold.nii.gz")):
            if lower.endswith((".nii", ".nii.gz")):
                current = bold_lookup.get(base_name)
                if current is None or acq_value < current[0]:
                    bold_lookup[base_name] = (acq_value, Path(base_name))

    bold_runs = sorted(bold_lookup.values(), key=lambda item: item[0])
    return acq_map, bold_runs


def _collect_metadata_from_scans(
    root: Path,
) -> Optional[Tuple[Dict[str, float], List[Tuple[float, Path]]]]:
    """Fallback loader that inspects ``*_scans.tsv`` files on disk."""

    scans_tsv = _find_scans_tsv(root)
    if scans_tsv is None:
        return None

    import pandas as pd  # Absolute import per repository conventions.

    try:
        df = pd.read_csv(scans_tsv, sep="\t")
    except Exception:
        return None

    if "filename" not in df.columns or "acq_time" not in df.columns:
        return None

    acq_map: Dict[str, float] = {}
    bold_lookup: Dict[str, Tuple[float, Path]] = {}

    for _, row in df.iterrows():
        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue

        acq_value = _parse_acq_time(row.get("acq_time"))
        if acq_value is None:
            continue

        _register_acq_entry(acq_map, filename, acq_value)

        base_name = Path(filename).name
        lower = base_name.lower()
        if "ref" in lower:
            continue
        if lower.endswith(("_bold.nii", "_bold.nii.gz")):
            current = bold_lookup.get(base_name)
            if current is None or acq_value < current[0]:
                bold_lookup[base_name] = (acq_value, Path(base_name))

    bold_runs = sorted(bold_lookup.values(), key=lambda item: item[0])
    return acq_map, bold_runs


def _lookup_acq_time(name: str, acq_map: Dict[str, float]) -> Optional[float]:
    """Find the acquisition time for ``name`` in ``acq_map``."""

    candidates = [name, Path(name).name]
    lower = name.lower()

    if lower.endswith(".json"):
        stem = name[:-5]
        candidates.extend([stem, f"{stem}.nii", f"{stem}.nii.gz"])
    elif lower.endswith(".nii.gz"):
        stem = name[:-7]
        candidates.extend([stem, f"{stem}.json", f"{stem}.nii"])
    elif lower.endswith(".nii"):
        stem = name[:-4]
        candidates.extend([stem, f"{stem}.json", f"{stem}.nii.gz"])

    for candidate in candidates:
        if candidate in acq_map:
            return acq_map[candidate]
    return None


def _build_assignments(
    fmap_dir: Path, acq_map: Dict[str, float], bold_runs: List[Tuple[float, Path]]
) -> Optional[Dict[Path, List[str]]]:
    """Return IntendedFor assignments using the provided acquisition metadata."""

    if not acq_map:
        return None

    fieldmap_groups: Dict[float, List[Path]] = {}
    for json_file in sorted(fmap_dir.glob("*.json")):
        acq_value = _lookup_acq_time(json_file.name, acq_map)
        if acq_value is None:
            return None
        fieldmap_groups.setdefault(acq_value, []).append(json_file)

    if not fieldmap_groups:
        return {}

    sorted_times = sorted(fieldmap_groups)
    assignments: Dict[Path, List[str]] = {}

    for index, acq_time in enumerate(sorted_times):
        next_time = sorted_times[index + 1] if index + 1 < len(sorted_times) else None
        intended: List[str] = []
        seen: set[str] = set()
        for run_time, rel_path in bold_runs:
            if run_time < acq_time:
                continue
            if next_time is not None and run_time >= next_time:
                continue
            tag = rel_path.name
            if tag in seen:
                continue
            intended.append(f"bids::{tag}")
            seen.add(tag)

        for json_file in fieldmap_groups[acq_time]:
            assignments[json_file] = intended

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
def post_fmap_rename(
    bids_root: Path, summary_df: Optional["pd.DataFrame"] = None
) -> None:
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
    dataset_summary = (
        _filter_summary_for_dataset(summary_df, bids_root.name)
        if summary_df is not None
        else None
    )
    add_intended_for(bids_root, dataset_summary)

    # Finally, refresh filenames recorded in ``*_scans.tsv`` to match the new
    # fieldmap file names.
    update_scans_tsv(bids_root)


def _find_scans_tsv(root: Path) -> Optional[Path]:
    """Return the scans table stored alongside ``root`` if it exists."""

    # Users can rename the table exported by HeuDiConv/this application, so we
    # look for any TSV whose header contains the required columns instead of
    # relying on a fixed ``*_scans.tsv`` pattern.  Reading just the first line
    # is enough to inspect the column names without pulling the entire table
    # into memory.
    for candidate in sorted(root.iterdir()):
        if candidate.suffix.lower() != ".tsv" or not candidate.is_file():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                header = handle.readline()
        except OSError:
            continue

        columns = {col.strip().lstrip("\ufeff").lower() for col in header.split("\t") if col}
        if {"filename", "acq_time"}.issubset(columns):
            return candidate

    return None


def _build_intended_from_summary(
    fmap_dir: Path, summary_subset: Optional["pd.DataFrame"]
) -> Optional[Dict[Path, List[str]]]:
    """Try to build IntendedFor assignments using the cached summary table."""

    if summary_subset is None or summary_subset.empty:
        return None

    acq_map, bold_runs = _collect_metadata_from_summary(summary_subset)
    return _build_assignments(fmap_dir, acq_map, bold_runs)


def _build_intended_from_scans(root: Path, fmap_dir: Path) -> Optional[Dict[Path, List[str]]]:
    """Fallback that inspects ``*_scans.tsv`` located next to ``root``."""

    metadata = _collect_metadata_from_scans(root)
    if metadata is None:
        return None

    acq_map, bold_runs = metadata
    return _build_assignments(fmap_dir, acq_map, bold_runs)


def _update_intended_for(
    root: Path, bids_root: Path, summary_df: Optional["pd.DataFrame"] = None
) -> None:
    """Add ``IntendedFor`` entries to fieldmap JSONs under ``root``."""
    # ``root`` points to either ``sub-<id>`` or ``sub-<id>/ses-<id>``
    # within the BIDS dataset. The function expects ``fmap`` and ``func``
    # directories side-by-side inside this folder.
    fmap_dir = root / "fmap"
    func_dir = root / "func"

    # Skip if either directory does not exist (e.g. no functional runs).
    if not (fmap_dir.is_dir() and func_dir.is_dir()):
        return

    subject_label = root.name if root.name.startswith("sub-") else root.parent.name
    session_label: Optional[str]
    if root.name.startswith("ses-"):
        subject_label = root.parent.name
        session_label = root.name
    else:
        session_label = None

    summary_subset = None
    if summary_df is not None:
        summary_subset = _filter_summary_for_subject(
            summary_df, subject_label, session_label
        )

    assignments = _build_intended_from_summary(fmap_dir, summary_subset)
    if assignments is None:
        assignments = _build_intended_from_scans(root, fmap_dir)

    if assignments is None:
        # Legacy fallback: include every functional run (except reference
        # images) and maintain deterministic ordering.  The only change is the
        # new ``bids::`` prefix requested by the user.
        func_files = [
            f for f in sorted(func_dir.glob("*.nii*")) if "ref" not in f.name.lower()
        ]
        if not func_files:
            return
        rel_paths = [f"bids::{f.name}" for f in func_files]
    else:
        rel_paths = []

    # Update each JSON sidecar under ``fmap`` with the collected paths.
    for js in fmap_dir.glob("*.json"):
        with open(js, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if assignments is None:
            meta["IntendedFor"] = rel_paths
        else:
            meta["IntendedFor"] = assignments.get(js, [])
        with open(js, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
            f.write("\n")
        print(f"Updated IntendedFor in {js.relative_to(bids_root)}")


def add_intended_for(
    bids_root: Path, summary_df: Optional["pd.DataFrame"] = None
) -> None:
    """Populate ``IntendedFor`` in all fieldmap JSONs."""
    # Walk through all subjects and sessions in the dataset. ``_update_intended_for``
    # handles the actual JSON editing for each folder.
    for sub in bids_root.glob("sub-*"):
        if not sub.is_dir():
            continue
        sessions = [s for s in sub.glob("ses-*") if s.is_dir()]
        if sessions:
            for ses in sessions:
                _update_intended_for(ses, bids_root, summary_df)
        else:
            _update_intended_for(sub, bids_root, summary_df)


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
    parser.add_argument(
        '--summary-tsv',
        help='Path to subject_summary.tsv with acquisition times',
        default=None,
    )
    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    summary_df = None
    if args.summary_tsv:
        summary_path = Path(args.summary_tsv)
        summary_df = _load_summary_table(summary_path)
        if summary_df is None:
            print(
                f"Warning: could not read summary table at {summary_path}",
                file=sys.stderr,
            )
    print(f"Starting fieldmap rename in: {bids_root}")
    post_fmap_rename(bids_root, summary_df)
    print("Done.")


if __name__ == '__main__':
    main()

