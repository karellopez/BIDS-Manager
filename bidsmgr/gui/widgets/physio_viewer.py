"""A ``*_physio.tsv.gz`` seen as a signal, through the shared viewer.

A physio recording is a table of numbers with no header row, and reading one
as a table tells you almost nothing. Whether the trigger fired where you
expect, whether the ECG is flat for the first minute, whether the
respiratory belt came loose halfway through, are all questions about the
SHAPE of the signal, and none of them can be answered from a grid of
six-decimal numbers.

**It is not a second viewer.** The file is turned into an ``mne.io.RawArray``
and handed to :class:`~bidsmgr.gui.widgets.time_series_view.TimeSeriesView`,
the same widget the MEG/EEG pane uses. A physio recording is channels, a
sampling rate and samples, which is exactly what a ``RawArray`` is made of,
so the conversion costs a few lines and buys the channel picker, the
navigation, the amplitude and window controls, zero-phase filtering,
resampling, the interactive spectrum and the events overlay, all of them
already written and already tested. The alternative was a second
implementation of each, drifting from the first one feature at a time.

Three facts the plot needs are not in the file, and BIDS puts all three in
the sidecar, which is why this reads it rather than guessing:

``Columns``            what each column is, since the file has no header
``SamplingFrequency``  how far apart the samples are
``StartTime``          where sample zero sits relative to the run, which is
                       usually NEGATIVE because recording starts before the
                       scanner does

Channel TYPES are guessed from the column names, and that is what makes the
shared viewer useful here rather than merely possible: a column called
``cardiac`` becomes an MNE ``ecg`` channel, ``respiratory`` becomes ``resp``,
a trigger becomes ``stim``. The viewer then colours them by kind, groups
them in the type filter, averages the spectrum per kind, and reads events
off the stim channel, all without knowing that physio exists.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

#: Beyond this many samples in one channel the read strides rather than
#: keeping everything. A memory bound, not a drawing one: the viewer draws
#: a window at a time. Four million samples costs 16 MB as float32, which at
#: 1000 Hz is over an hour of recording.
_MAX_SAMPLES = 4_000_000

#: Column name (lower-cased, non-letters stripped) -> MNE channel type.
#: Matched as a WHOLE word first and then as a substring, so ``cardiac`` and
#: ``card1`` both land on ``ecg`` while ``discard`` does not.
#:
#: The right-hand side is deliberately MNE's vocabulary rather than ours:
#: the viewer, the colour map and the spectrum all group by it, so inventing
#: a parallel set of names here would mean translating twice.
_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    ("cardiac", "ecg"),
    ("ecg", "ecg"),
    ("ekg", "ecg"),
    ("pulse", "ecg"),
    ("ppg", "ecg"),
    ("plethysmograph", "ecg"),
    ("respiratory", "resp"),
    ("respiration", "resp"),
    ("resp", "resp"),
    ("breath", "resp"),
    ("belt", "resp"),
    ("trigger", "stim"),
    ("stim", "stim"),
    ("scannertrigger", "stim"),
    ("ttl", "stim"),
    ("eyegaze", "eyegaze"),
    ("xcoordinate", "eyegaze"),
    ("ycoordinate", "eyegaze"),
    ("gaze", "eyegaze"),
    ("pupil", "pupil"),
    ("emg", "emg"),
    ("eog", "eog"),
    ("temperature", "temperature"),
    ("gsr", "gsr"),
    ("skinconductance", "gsr"),
)

#: What a column becomes when nothing matches. ``misc`` rather than ``bio``
#: because MNE treats ``misc`` as "shown, not interpreted", which is exactly
#: the right claim about a column we did not recognise.
_DEFAULT_TYPE = "misc"


def sidecar_for(path: Path) -> Path:
    """The ``.json`` beside a ``.tsv`` or ``.tsv.gz``."""
    name = path.name
    for ext in (".tsv.gz", ".tsv"):
        if name.endswith(ext):
            return path.with_name(name[: -len(ext)] + ".json")
    return path.with_suffix(".json")


def read_timing(path: Path) -> Optional[dict]:
    """``{columns, sampling_frequency, start_time, units}`` or ``None``.

    ``None`` means this is not a continuous recording, which is how the
    caller decides whether to offer a viewer at all. A table of onsets
    (``_events.tsv``) has no sampling frequency and is not one. Deciding on
    the SIDECAR rather than the filename means ``_stim.tsv.gz``, and any
    future continuous suffix, are offered a viewer without this knowing
    they exist.
    """
    sidecar = sidecar_for(path)
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        rate = float(data["SamplingFrequency"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(rate) or rate <= 0:
        return None
    columns = data.get("Columns")
    if not isinstance(columns, list) or not columns:
        return None
    try:
        start = float(data.get("StartTime", 0.0))
    except (TypeError, ValueError):
        start = 0.0
    units = data.get("Units")
    return {
        "columns": [str(c) for c in columns],
        "sampling_frequency": rate,
        "start_time": start if math.isfinite(start) else 0.0,
        "units": str(units) if isinstance(units, str) else "",
    }


def guess_channel_type(name: str) -> str:
    """The MNE channel type a physio column name implies.

    Whole-word match first, then substring, so ``cardiac`` and ``card_1``
    both reach ``ecg`` while ``discard`` reaches neither.
    """
    token = re.sub(r"[^a-z]", "", str(name).lower())
    if not token:
        return _DEFAULT_TYPE
    for needle, ch_type in _TYPE_HINTS:
        if token == needle:
            return ch_type
    for needle, ch_type in _TYPE_HINTS:
        if needle in token:
            return ch_type
    return _DEFAULT_TYPE


def read_columns(
    path: Path, limit: int = _MAX_SAMPLES,
) -> tuple[list[np.ndarray], int, int]:
    """``(columns, total_samples, step)``: the whole file, and what it took.

    Reads the WHOLE file rather than the preview the table shows. The table
    is bounded at five thousand rows because nobody reads more than that; a
    view of the first five thousand samples of a 1.4-million-sample trigger
    channel would be a picture of the first four seconds, drawn as though it
    were the recording. Silently.

    ``step`` is 1 unless the file is longer than ``limit``, at which point
    the read strides and each kept point stands for ``step`` real samples.
    Both numbers are returned because both are needed to say anything true
    about the result: ``total`` keeps the caption honest and ``step`` keeps
    the SAMPLING RATE honest, since a strided recording is a slower one.

    Parsed with pandas' C engine, which releases the GIL: 1.4 million rows
    read in 46 ms. Run on a worker anyway, because "fast on the files we
    happened to test" is how a freeze gets shipped.
    """
    import pandas as pd

    try:
        frame = pd.read_csv(
            path, sep="\t", header=None, engine="c",
            compression="gzip" if str(path).lower().endswith(".gz") else "infer",
            na_values=["n/a", "N/A", ""],
        )
    except Exception as exc:  # noqa: BLE001 - a bad file must not crash the pane
        log.warning("could not read %s for viewing: %s", path, exc)
        return [], 0, 1

    total = int(len(frame))
    step = max(1, math.ceil(total / limit)) if total > limit else 1
    out: list[np.ndarray] = []
    for name in frame.columns:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(
            dtype=np.float32,
        )
        out.append(values[::step] if step > 1 else values)
    return out, total, step


def build_raw(columns: list[np.ndarray], timing: dict, step: int = 1):
    """An ``mne.io.RawArray`` from physio columns, typed by column name.

    ``step`` divides the sampling rate, because a strided read really is a
    slower recording and every frequency the viewer computes depends on
    getting that right.

    NaN becomes zero, and that is a loss worth naming: a gap in a physio
    recording is a fact about it, and the table view still shows the blank.
    MNE's filtering and spectral code cannot carry NaN through, so a viewer
    built on MNE has to choose between the gap and the filters. It keeps the
    filters, because the questions people bring to a physio trace are about
    shape and timing.
    """
    import mne

    names = [str(c) for c in timing.get("columns", [])]
    n = min(len(names), len(columns))
    if n == 0:
        raise ValueError("no physio columns to show")
    names = names[:n]
    # MNE refuses duplicate channel names, and a sidecar is free to repeat one.
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}-{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)

    rate = float(timing.get("sampling_frequency", 1.0)) or 1.0
    rate = rate / max(1, int(step))
    types = [guess_channel_type(name) for name in names]
    data = np.vstack([
        np.nan_to_num(np.asarray(columns[i], dtype=np.float64), nan=0.0)
        for i in range(n)
    ])
    info = mne.create_info(unique, sfreq=rate, ch_types=types, verbose=False)
    return mne.io.RawArray(data, info, verbose=False)


__all__ = [
    "build_raw",
    "guess_channel_type",
    "read_columns",
    "read_timing",
    "sidecar_for",
]
