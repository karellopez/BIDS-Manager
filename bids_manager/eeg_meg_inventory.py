#!/usr/bin/env python3
"""Inventory raw EEG/MEG/iEEG recordings into a TSV.

This is the EEG/MEG analogue of :mod:`bids_manager.dicom_inventory`. It
walks a directory and probes every candidate file with ``mne.io.read_raw``
(no preload). Files that mne can read produce one row in the output TSV;
files mne can't read are silently skipped, so dropping a folder of mixed
content produces a clean inventory of the recordings only.

Output columns mirror the DICOM TSV's shape where it makes sense
(``BIDS_name``, ``session``, ``include``, ``task``) plus EEG/MEG-specific
fields the user typically wants to verify before conversion: ``datatype``
(eeg/meg/ieeg), ``sfreq``, ``n_channels``, ``recording_time``,
``duration_sec``, ``has_positions`` (bool — informs the GUI's
"missing-position" warning), ``format``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

# ``mne`` is heavy; import lazily so this module is cheap to import in
# environments where mne isn't yet installed (e.g. a fresh venv or unit
# tests that mock the inventory output).
try:
    import mne  # noqa: F401
    _HAS_MNE = True
except Exception:
    _HAS_MNE = False


# Extensions mne-bids' ``write_raw_bids`` accepts as raw inputs (and a few
# common formats it transparently converts to BrainVision/FIF on write).
# Lowercase comparison.
_RECOGNISED_EXTS: tuple[str, ...] = (
    # MEG
    ".fif", ".fif.gz", ".con", ".sqd", ".pdf",
    # EEG
    ".vhdr", ".edf", ".bdf", ".gdf", ".set", ".cnt", ".eeg", ".egi", ".mff",
    # iEEG
    ".mef", ".nwb",
    # NIRS (mne-bids supports it; included for completeness)
    ".snirf",
)
# CTF / KIT / 4D etc. are folder-shaped recordings (.ds, .m4d). Treated separately.
_DIR_FORMATS: tuple[str, ...] = (".ds", ".mff")


@dataclass(frozen=True)
class _ProbeResult:
    """What we extract from a single recording during inventory."""
    source: Path
    sfreq: float
    n_channels: int
    n_times: int
    duration_sec: float
    recording_time: str  # ISO-ish string or empty
    datatype: str        # 'eeg' | 'meg' | 'ieeg' | 'nirs' | ''
    has_positions: bool
    fmt: str             # short label e.g. 'EDF', 'FIF'


def _detect_datatype(raw) -> str:
    """Best-effort datatype label from channel kinds present in ``raw``."""
    try:
        ch_types = set(raw.get_channel_types())
    except Exception:
        ch_types = set()
    if "meg" in ch_types or any(t in ch_types for t in ("mag", "grad", "ref_meg")):
        return "meg"
    if any(t in ch_types for t in ("seeg", "ecog", "dbs")):
        return "ieeg"
    if "eeg" in ch_types:
        return "eeg"
    if any(t.startswith("fnirs") for t in ch_types):
        return "nirs"
    return ""


def _has_positions(raw) -> bool:
    """Return True when the recording carries usable channel positions.

    The canonical check: if ``raw.get_montage()`` returns a montage with at
    least one position vector that's not all-NaN, we consider positions
    present. ``raw.info['dig']`` alone is unreliable because some readers
    populate fiducials but no electrodes.
    """
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
    import math
    for vec in positions.values():
        if vec is None:
            continue
        try:
            if any(v is not None and not math.isnan(float(v)) for v in vec):
                return True
        except Exception:
            continue
    return False


def _format_label(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".fif.gz") or name.endswith(".fif"):
        return "FIF"
    if name.endswith(".vhdr"):
        return "BrainVision"
    if name.endswith(".edf"):
        return "EDF"
    if name.endswith(".bdf"):
        return "BDF"
    if name.endswith(".gdf"):
        return "GDF"
    if name.endswith(".set"):
        return "EEGLAB"
    if name.endswith(".cnt"):
        return "CNT"
    if name.endswith(".eeg"):
        return "Nihon Kohden"
    if name.endswith((".sqd", ".con")):
        return "KIT"
    if name.endswith(".pdf"):
        return "4D"
    if name.endswith(".ds"):
        return "CTF"
    if name.endswith(".mef"):
        return "MEF"
    if name.endswith(".nwb"):
        return "NWB"
    if name.endswith(".snirf"):
        return "SNIRF"
    if name.endswith((".mff",)):
        return "EGI MFF"
    return path.suffix.lstrip(".").upper() or "?"


def _probe(path: Path) -> Optional[_ProbeResult]:
    """Attempt to read ``path`` with mne; return a probe result or None."""
    if not _HAS_MNE:
        return None
    try:
        # ``read_raw`` dispatches on extension. We never preload — keeping
        # this cheap so a directory with many files inventories quickly.
        raw = mne.io.read_raw(str(path), preload=False, verbose="ERROR")
    except Exception:
        return None

    sfreq = float(raw.info.get("sfreq", 0.0))
    n_channels = int(len(raw.ch_names))
    n_times = int(raw.n_times)
    duration_sec = float(n_times / sfreq) if sfreq > 0 else 0.0

    meas_date = raw.info.get("meas_date")
    if meas_date is None:
        recording_time = ""
    else:
        try:
            recording_time = meas_date.isoformat()
        except Exception:
            recording_time = str(meas_date)

    return _ProbeResult(
        source=path,
        sfreq=sfreq,
        n_channels=n_channels,
        n_times=n_times,
        duration_sec=duration_sec,
        recording_time=recording_time,
        datatype=_detect_datatype(raw),
        has_positions=_has_positions(raw),
        fmt=_format_label(path),
    )


def _candidate_paths(root: Path) -> List[Path]:
    """Walk ``root`` and yield candidate recording paths.

    Folder-shaped formats (``.ds``, ``.mff``) are yielded as the directory
    itself; everything else as a file. The walk is shallow on a per-folder
    basis: when we hit a ``.ds`` directory we don't descend into it.
    """
    candidates: List[Path] = []
    for cur, dirs, files in os.walk(root):
        cur_path = Path(cur)
        # If this directory itself is a folder-shaped recording, treat it
        # as a single candidate and stop walking inside.
        suff = cur_path.suffix.lower()
        if suff in _DIR_FORMATS:
            candidates.append(cur_path)
            dirs[:] = []
            continue
        # Don't descend into nested folder-shaped recordings either.
        dirs[:] = [d for d in dirs if not d.lower().endswith(_DIR_FORMATS)]
        # Folder-shaped recordings encountered as subdirs: queue them.
        for d in list(dirs):
            full = cur_path / d
            if full.suffix.lower() in _DIR_FORMATS:
                candidates.append(full)
        for fname in files:
            lname = fname.lower()
            for ext in _RECOGNISED_EXTS:
                if lname.endswith(ext):
                    candidates.append(cur_path / fname)
                    break
    # BrainVision: keep only .vhdr; .eeg/.vmrk are paired sidecars.
    pruned: List[Path] = []
    seen_brainvision_stems: set[Path] = set()
    for p in candidates:
        ln = p.name.lower()
        if ln.endswith(".vhdr"):
            seen_brainvision_stems.add(p.with_suffix(""))
            pruned.append(p)
        elif ln.endswith(".eeg") or ln.endswith(".vmrk"):
            # Skip if a sibling .vhdr exists (we'll have it via the .vhdr).
            if p.with_suffix(".vhdr").exists():
                continue
            pruned.append(p)
        else:
            pruned.append(p)
    return sorted(pruned)


def _bids_id_from_filename(name: str) -> str:
    """Return a BIDS-safe subject token derived from ``name``.

    Strips a leading ``sub-`` if present, then keeps alphanumerics. Empty
    results fall back to the original name (so the user can still see what
    needs editing in the TSV)."""
    s = name
    if s.lower().startswith("sub-"):
        s = s[4:]
    s = re.sub(r"[^0-9A-Za-z]+", "", s)
    return s


def _guess_subject_session_task(path: Path, root: Path) -> tuple[str, str, str]:
    """Heuristic: pull subject/session/task hints from the path under ``root``.

    Looks for ``sub-XXX``, ``ses-YYY``, ``task-ZZZ`` tokens anywhere in the
    relative path. Falls back to the topmost folder name as the subject.
    """
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    parts = list(rel.parts)
    sub = ""
    ses = ""
    task = ""
    for part in parts:
        m = re.match(r"sub-([A-Za-z0-9]+)", part, flags=re.IGNORECASE)
        if m and not sub:
            sub = m.group(1)
        m = re.match(r"ses-([A-Za-z0-9]+)", part, flags=re.IGNORECASE)
        if m and not ses:
            ses = m.group(1)
        m = re.search(r"task-([A-Za-z0-9]+)", part, flags=re.IGNORECASE)
        if m and not task:
            task = m.group(1)
    if not sub and parts:
        sub = _bids_id_from_filename(parts[0])
    if not task:
        # Filename without extension as the fallback task hint, sanitised.
        stem = path.stem
        if stem.lower().endswith(".fif"):
            stem = stem[:-4]
        task = re.sub(r"[^0-9A-Za-z]+", "", stem) or "task"
    return sub, ses, task


def scan_eeg_meg(root: Path, output_tsv: Path) -> pd.DataFrame:
    """Walk ``root`` and write an EEG/MEG/iEEG inventory TSV.

    Returns the resulting DataFrame for convenience (mirrors
    :func:`dicom_inventory.scan_dicoms_long`).
    """
    if not _HAS_MNE:
        raise RuntimeError(
            "mne is not installed; install mne and mne-bids to use the "
            "EEG/MEG inventory (pip install mne mne-bids)."
        )

    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    rows: List[dict] = []
    bids_counter = 0
    bids_id_for_subject: dict[str, str] = {}

    for path in _candidate_paths(root):
        probe = _probe(path)
        if probe is None:
            continue

        sub_hint, ses_hint, task_hint = _guess_subject_session_task(path, root)
        sub_token = sub_hint or _bids_id_from_filename(path.parent.name)
        if not sub_token:
            sub_token = path.stem

        if sub_token not in bids_id_for_subject:
            bids_counter += 1
            bids_id_for_subject[sub_token] = f"{bids_counter:03d}"
        bids_name = bids_id_for_subject[sub_token]

        rel = path.relative_to(root).as_posix()
        rows.append({
            "subject": sub_token,
            "BIDS_name": bids_name,
            "session": ses_hint,
            "task": task_hint,
            "include": 1,
            "datatype": probe.datatype or "eeg",
            "format": probe.fmt,
            "source_file": rel,
            "n_channels": probe.n_channels,
            "sfreq": probe.sfreq,
            "duration_sec": round(probe.duration_sec, 3),
            "n_times": probe.n_times,
            "recording_time": probe.recording_time,
            "has_positions": int(probe.has_positions),
        })

    df = pd.DataFrame(rows, columns=[
        "subject", "BIDS_name", "session", "task", "include",
        "datatype", "format", "source_file",
        "n_channels", "sfreq", "duration_sec", "n_times",
        "recording_time", "has_positions",
    ])

    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"EEG/MEG inventory: {len(df)} recording(s) → {output_tsv}")
    return df


def main() -> None:
    """CLI for the ``eeg-meg-inventory`` entry point."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Inventory raw EEG/MEG/iEEG recordings under a directory"
    )
    parser.add_argument("raw_root", help="Root directory of raw EEG/MEG files")
    parser.add_argument("output_tsv", help="Destination TSV file")
    args = parser.parse_args()
    scan_eeg_meg(Path(args.raw_root), Path(args.output_tsv))


if __name__ == "__main__":
    main()
