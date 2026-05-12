#!/usr/bin/env python3
"""Convert raw EEG/MEG/iEEG recordings into BIDS using ``mne-bids``.

Reads the inventory TSV produced by :mod:`bids_manager.eeg_meg_inventory`
and calls :func:`mne_bids.write_raw_bids` once per included row. The
contract is intentionally lighter than the DICOM pipeline: mne-bids itself
writes ``*_channels.tsv``, the datatype-specific JSON sidecar,
``*_events.tsv``, and (when channel positions are present) ``electrodes.tsv``
+ ``coordsystem.json``. Dataset-level metadata
(``dataset_description.json``, ``participants.tsv``, ``README``,
``CHANGES``) is filled afterwards by :mod:`bids_metadata_engine`, which is
converter-agnostic.

Per the user-confirmed defaults:

* When a recording has no usable channel positions we emit a *warning* but
  still write the data — useful for EDF/BDF files acquired without a
  digitiser.
* Events are taken from whatever ``raw`` already exposes (annotations and,
  where available, the STIM channel through mne's auto-discovery).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

import mne
from mne_bids import BIDSPath, write_raw_bids


@dataclass
class ConversionReport:
    """What the run produced and what to surface to the user."""
    written_paths: List[Path] = field(default_factory=list)
    skipped_rows: List[str] = field(default_factory=list)
    missing_positions: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _is_included(value: object) -> bool:
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "no"}


def _normalize_token(value: str, *, allow_empty: bool = False) -> str:
    s = str(value or "").strip()
    if s.lower().startswith(("sub-", "ses-", "task-")):
        s = s.split("-", 1)[1]
    s = re.sub(r"[^0-9A-Za-z]+", "", s)
    if not s and not allow_empty:
        return "X"
    return s


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "n/a"}:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _has_positions(raw) -> bool:
    """Mirror of eeg_meg_inventory._has_positions for the conversion side."""
    import math
    try:
        montage = raw.get_montage()
    except Exception:
        return False
    if montage is None:
        return False
    try:
        positions = montage.get_positions().get("ch_pos") or {}
    except Exception:
        return False
    for vec in positions.values():
        if vec is None:
            continue
        try:
            if any(v is not None and not math.isnan(float(v)) for v in vec):
                return True
        except Exception:
            continue
    return False


def _resolve_source(raw_root: Path, source_field: str) -> Path:
    """Return the on-disk path for an inventory ``source_file`` value."""
    candidate = Path(source_field)
    if candidate.is_absolute():
        return candidate
    return (raw_root / candidate).resolve()


def _convert_row(row: pd.Series, raw_root: Path, bids_root: Path,
                 report: ConversionReport, overwrite: bool) -> None:
    source = _resolve_source(raw_root, str(row.get("source_file", "")))
    if not source.exists():
        report.skipped_rows.append(f"missing source: {source}")
        return

    subject = _normalize_token(row.get("BIDS_name") or row.get("subject"))
    session = _normalize_token(row.get("session", ""), allow_empty=True) or None
    task = _normalize_token(row.get("task", "")) or "task"
    run = _coerce_int(row.get("run"))
    datatype = str(row.get("datatype") or "eeg").strip().lower() or "eeg"

    try:
        raw = mne.io.read_raw(str(source), preload=False, verbose="ERROR")
    except Exception as exc:
        report.errors.append(f"read failed for {source}: {exc}")
        return

    if not _has_positions(raw):
        report.missing_positions.append(source)

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype=datatype,
        root=str(bids_root),
    )

    try:
        # ``format='auto'`` keeps the original format where mne-bids supports
        # it (EDF/BDF/BrainVision/EEGLAB/FIF/CTF/KIT/...) and falls back to
        # BrainVision for EEG / FIF for MEG when the original format isn't
        # in the BIDS-allowed set.
        out_path = write_raw_bids(
            raw,
            bids_path,
            overwrite=overwrite,
            format="auto",
            verbose="ERROR",
        )
    except Exception as exc:
        report.errors.append(f"write_raw_bids failed for {source}: {exc}")
        return

    try:
        report.written_paths.append(Path(out_path.fpath))
    except Exception:
        report.written_paths.append(Path(str(out_path)))


def run(tsv: Path, raw_root: Path, bids_root: Path,
        overwrite: bool = False) -> ConversionReport:
    """Drive ``write_raw_bids`` for every included row in ``tsv``."""
    df = pd.read_csv(tsv, sep="\t", keep_default_na=False, dtype=str)
    if "include" in df.columns:
        df = df[df["include"].apply(_is_included)]
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)
    raw_root = Path(raw_root)

    report = ConversionReport()
    for _, row in df.iterrows():
        _convert_row(row, raw_root, bids_root, report, overwrite)
    return report


def main() -> None:
    """CLI for the ``run-mne-bids`` entry point."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Convert raw EEG/MEG/iEEG recordings into BIDS via mne-bids"
    )
    parser.add_argument("tsv", help="Inventory TSV produced by eeg-meg-inventory")
    parser.add_argument("raw_root", help="Root directory of the raw recordings")
    parser.add_argument("bids_root", help="BIDS output root")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing files in the BIDS dataset")
    args = parser.parse_args()

    report = run(Path(args.tsv), Path(args.raw_root), Path(args.bids_root),
                 overwrite=args.overwrite)
    print(f"\nWrote {len(report.written_paths)} recording(s).")
    for p in report.written_paths:
        print(f"  - {p}")
    if report.missing_positions:
        print(f"\n{len(report.missing_positions)} recording(s) had no channel "
              "positions; electrodes.tsv / coordsystem.json were not written:")
        for p in report.missing_positions:
            print(f"  ! {p}")
    if report.skipped_rows:
        print(f"\nSkipped {len(report.skipped_rows)} row(s):")
        for r in report.skipped_rows:
            print(f"  · {r}")
    if report.errors:
        print(f"\n{len(report.errors)} error(s):")
        for e in report.errors:
            print(f"  ! {e}")


if __name__ == "__main__":
    main()
