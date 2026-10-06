"""Which files are recordings, and which viewer a file opens in.

Routing a click must not read the file: a 500 MB BOLD and a 32 KB spectrum
look the same until you do. So the kind of a file is decided from its name,
its folder and (for a table) its sidecar, here, once, for every host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from ..bids import full_ext

#: Recording files the signal viewer opens.
#:
#: BrainVision sidecars (``.eeg`` / ``.vmrk``) are deliberately absent: the
#: user opens the ``.vhdr``. ``mne.io.read_raw`` dispatches every entry,
#: with the vendor-specific fallbacks in :func:`signal.read_recording`.
RECORDING_FILE_EXTS: frozenset[str] = frozenset({
    ".fif", ".fif.gz", ".con", ".sqd", ".kdf",
    ".vhdr", ".edf", ".bdf", ".gdf", ".set", ".cnt", ".egi", ".mef", ".nwb",
})
#: Folder-shaped recordings (CTF, EGI).
RECORDING_DIR_EXTS: frozenset[str] = frozenset({".ds", ".mff"})

#: Volume files the volume viewer opens.
VOLUME_EXTS: frozenset[str] = frozenset({".nii", ".nii.gz", ".mgz", ".mgh"})

Kind = Literal["volume", "signal", "spectrum", "physio", ""]


def is_recording_path(path) -> bool:
    """True when ``path`` is an EEG/MEG/iEEG recording the viewer can open."""
    p = Path(path)
    ext = full_ext(p)
    if p.is_dir():
        return ext in RECORDING_DIR_EXTS
    return ext in RECORDING_FILE_EXTS


def is_mrs_path(path) -> bool:
    """True when ``path`` should open in the spectrum viewer.

    Decided by the DATATYPE folder and the schema's own suffix list, never by
    opening the file.
    """
    p = Path(path)
    if full_ext(p) not in (".nii", ".nii.gz"):
        return False
    if p.parent.name != "mrs":
        return False
    try:
        from ... import schema

        suffixes = set(schema.list_suffixes("mrs"))
    except Exception:  # noqa: BLE001
        suffixes = {"svs", "mrsi", "unloc", "mrsref"}
    stem = p.name.split(".")[0]
    return stem.rsplit("_", 1)[-1] in suffixes


def kind_of(path, *, sidecar_checked: Optional[bool] = None) -> Kind:
    """Which viewer ``path`` opens in ("" for none).

    A ``.tsv``/``.tsv.gz`` is a ``physio`` recording only when its sidecar
    declares a sampling frequency; pass ``sidecar_checked`` when the caller
    already knows, or it is read here (a few hundred bytes).
    """
    p = Path(path)
    if is_mrs_path(p):
        return "spectrum"
    if full_ext(p) in VOLUME_EXTS:
        return "volume"
    if is_recording_path(p):
        return "signal"
    if full_ext(p) in (".tsv", ".tsv.gz"):
        if sidecar_checked is None:
            from .physio import read_timing

            sidecar_checked = read_timing(p) is not None
        return "physio" if sidecar_checked else ""
    return ""


__all__ = [
    "RECORDING_DIR_EXTS", "RECORDING_FILE_EXTS", "VOLUME_EXTS", "is_mrs_path",
    "is_recording_path", "kind_of",
]
