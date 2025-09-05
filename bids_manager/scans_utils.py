from __future__ import annotations

from pathlib import Path
import pandas as pd


def update_scans_with_map(bids_root: Path, rename_map: dict[str, str]) -> None:
    """Update filenames in ``*_scans.tsv`` after renaming files.

    Parameters
    ----------
    bids_root : Path
        Root of the BIDS dataset.
    rename_map : dict[str, str]
        Mapping of old relative paths to new relative paths within ``bids_root``.
    """
    if not rename_map:
        return

    for sub in bids_root.glob("sub-*"):
        if not sub.is_dir():
            continue
        sessions = [s for s in sub.glob("ses-*") if s.is_dir()]
        roots = sessions or [sub]
        for root in roots:
            for tsv in root.glob("*_scans.tsv"):
                _update_single_scans(tsv, rename_map)


def _update_single_scans(tsv: Path, rename_map: dict[str, str]) -> None:
    df = pd.read_csv(tsv, sep="\t")
    if "filename" not in df.columns:
        return

    changed = False
    for idx, fname in enumerate(df["filename"]):
        new = rename_map.get(fname)
        if new:
            df.at[idx, "filename"] = new
            changed = True

    if changed:
        df.to_csv(tsv, sep="\t", index=False)
        print(f"Updated {tsv}")


def refresh_scans_filenames(bids_root: Path) -> None:
    """Refresh ``filename`` columns in all ``*_scans.tsv`` files.

    This walks through the dataset and ensures each entry matches an
    existing file within the corresponding subject or session.
    """

    for sub in bids_root.glob("sub-*"):
        if not sub.is_dir():
            continue
        sessions = [s for s in sub.glob("ses-*") if s.is_dir()]
        roots = sessions or [sub]
        for root in roots:
            files = {f.name: f.relative_to(root).as_posix() for f in root.rglob('*') if f.is_file()}
            for tsv in root.glob("*_scans.tsv"):
                _refresh_single(tsv, files, bids_root)


def _refresh_single(tsv: Path, file_map: dict[str, str], bids_root: Path) -> None:
    df = pd.read_csv(tsv, sep="\t")
    if "filename" not in df.columns:
        return

    changed = False
    for idx, fname in enumerate(df["filename"]):
        base = Path(fname).name
        new = file_map.get(base)
        if new and new != fname:
            df.at[idx, "filename"] = new
            changed = True

    if changed:
        df.to_csv(tsv, sep="\t", index=False)
        print(f"Refreshed {tsv.relative_to(bids_root)}")
