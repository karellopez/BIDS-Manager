"""Draw a continuous recording's columns against time, and let them be read.

A ``*_physio.tsv.gz`` is a table of numbers with no header row, and reading
one as a table tells you almost nothing: whether the trigger fired where you
expect, whether the ECG is flat for the first minute, whether the respiratory
belt came loose halfway through, are all questions about the SHAPE of the
signal. As a grid of six-decimal numbers they are unanswerable.

The plot needs three facts that are not in the file, and BIDS puts all three
in the sidecar, which is why this reads it rather than guessing:

``Columns``            what each column is, since the file has no header
``SamplingFrequency``  how far apart the samples are
``StartTime``          where sample zero sits relative to the run, which is
                       usually NEGATIVE because recording starts before the
                       scanner does

Four decisions worth stating.

**Every channel is normalised by default.** A trigger is 0 or 5 and an ECG is
fractions of a millivolt; on one shared axis the ECG is a flat line at the
bottom. Normalised, each channel is drawn in its own horizontal band and the
shapes are comparable, which is what the reader came for. Raw values are one
tick away for when the numbers themselves matter, and the band height (gain)
and the gap between bands (spacing) are both adjustable, because how much
vertical room a signal wants depends entirely on what it is.

**The plot holds every sample and lets pyqtgraph decide what to draw.**
Handing 1.4 million points to a plot widget is how a viewer freezes; handing
it four thousand points sampled from those 1.4 million is how a viewer lies,
because zooming in then shows the same four thousand stretched out rather
than the detail that is there. pyqtgraph's own peak downsampling, clipped to
the visible range, does the right thing at every zoom level: about two points
per pixel, taken from the real data, recomputed as you zoom. Measured on a
1,449,001-sample trigger channel: 53 ms to draw, 20 ms to re-draw a zoom.

**Filtering is applied to the whole signal, on a worker.** Filtering the
drawn points instead would filter a signal that had already been decimated,
which is a different signal. Whole-signal filtering costs about 50 ms per
channel, which is under the threshold where a person notices on one channel
and well over it on six, so it runs off the GUI thread and the last request
wins.

That worker is a ``QThread``, and it has to be. The first version used
``QThreadPool``, which is lighter and is what the file-system panes use, and
it segfaulted deterministically: filter on a pooled thread, then compute a
spectrum on a pooled thread, and scipy's FFT dies inside ``pocketfft`` with
no Python traceback. Measured across the alternatives: the same work on a
``QThread`` is fine, on a plain ``threading.Thread`` is fine, and on a
pooled thread is fine too IF ``scipy.signal`` was first imported on the GUI
thread. A 32 MB pool stack does not help, so it is not a stack overflow.
Whatever scipy initialises per thread does not survive a pool thread being
retired and its slot reused. Hence ``RecordingComputeWorker``, which is the
``QThread`` the MEG/EEG viewer already computes its spectra on.

**A gap stays a gap.** A blank or non-numeric cell becomes NaN and is drawn
as a break in the trace, never closed up, because a gap in a recording is a
fact about the recording. The filters restore the gaps afterwards: an IIR
filter fed a NaN returns nothing but NaN from there on.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import CUR
from .flow_layout import flow
from .primitives import ElidedLabel

log = logging.getLogger(__name__)

#: Beyond this many samples in one channel the read strides rather than
#: keeping everything. Not a drawing limit (pyqtgraph downsamples what it
#: draws): a memory one. At four million samples a channel costs 16 MB as
#: float32, which at 1000 Hz is over an hour of recording.
_MAX_SAMPLES = 4_000_000

#: Colours cycled across channels. Palette tokens, so a theme swap follows.
_CURVE_TOKENS = ("accent", "teal", "purple", "text", "dim", "warn")

#: Filter order. High enough to have a usable roll-off, low enough that
#: ``sosfiltfilt`` does not need more padding than a short recording has.
_ORDER = 4

#: Notch quality factor: how narrow the rejected band is. 30 at 50 Hz is
#: about 1.7 Hz wide, which removes mains hum without taking the signal
#: either side of it with it.
_NOTCH_Q = 30.0


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
    caller decides whether to offer a plot at all. A table of onsets
    (``_events.tsv``) has no sampling frequency and is not one.

    ``units`` is whatever the sidecar says the numbers are in, and is
    usually absent. It is carried because an axis labelled "value" tells
    the reader less than one labelled "mmHg".
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


def to_series(rows: list[list[str]], n_columns: int) -> list[np.ndarray]:
    """Columns of floats from the table's rows; blanks and text become NaN.

    NaN rather than skipping: a gap in a recording is a fact about the
    recording, and closing it up would draw a signal that never happened.
    """
    out: list[np.ndarray] = []
    for index in range(n_columns):
        values = np.empty(len(rows), dtype=np.float32)
        for position, row in enumerate(rows):
            cell = row[index] if index < len(row) else ""
            try:
                values[position] = float(cell)
            except (TypeError, ValueError):
                values[position] = np.nan
        out.append(values)
    return out


def normalise(values: np.ndarray) -> np.ndarray:
    """Scale to roughly -0.5..0.5 so channels can share one axis.

    A flat channel stays flat at zero rather than being amplified into
    noise, which is what dividing by a near-zero range would do.
    """
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    low = float(finite.min())
    high = float(finite.max())
    span = high - low
    if span <= 0:
        return np.where(np.isfinite(values), np.float32(0.0), values)
    return ((values - (high + low) / 2.0) / span).astype(np.float32)


def read_columns(
    path: Path, limit: int = _MAX_SAMPLES,
) -> tuple[list[np.ndarray], int, int]:
    """``(columns, total_samples, step)``: the whole file, and what it took.

    Reads the WHOLE file rather than the preview the table shows. The table
    is bounded at five thousand rows because nobody reads more than that;
    a plot of the first five thousand samples of a 1.4-million-sample
    trigger channel would be a picture of the first four seconds, drawn as
    though it were the recording. Silently.

    ``step`` is 1 unless the file is longer than ``limit``, at which point
    the read strides and each kept point stands for ``step`` real samples.
    Both numbers are returned because both are needed to say anything true
    about the result: ``total`` keeps the caption honest, and ``step`` keeps
    the TIME AXIS honest. Position ``i`` is sample ``i * step``, and reading
    it as sample ``i`` drew a 2,479-second recording as ten seconds of one.

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
        log.warning("could not read %s for plotting: %s", path, exc)
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


def filter_signal(
    values: np.ndarray,
    rate: float,
    *,
    high_pass: float = 0.0,
    low_pass: float = 0.0,
    notch: float = 0.0,
) -> np.ndarray:
    """``values`` with the requested filters applied. Zero means "off".

    Zero-phase (``sosfiltfilt``), so a trigger edge stays where it is. A
    causal filter would shift every feature later in time, and "where did
    this happen" is most of what a physio trace is read for.

    Anything the signal cannot support is skipped rather than refused: a
    cut-off at or above Nyquist, a high-pass above the low-pass, a
    recording too short for the filter's own padding. A viewer that
    refuses to draw because one control is out of range is worse than one
    that draws the signal unfiltered.
    """
    values = np.asarray(values, dtype=np.float32)
    nyquist = rate / 2.0
    if values.size == 0 or nyquist <= 0:
        return values

    try:
        from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
    except Exception:  # noqa: BLE001 - scipy is a transitive dep, not a given
        log.debug("scipy is unavailable; the signal is drawn unfiltered")
        return values

    high = high_pass if 0.0 < high_pass < nyquist else 0.0
    low = low_pass if 0.0 < low_pass < nyquist else 0.0
    if high and low and high >= low:
        # A band that excludes everything. The high-pass is the one the
        # reader is more likely to have meant (drift removal), so it wins.
        low = 0.0

    sections = []
    if high and low:
        sections.append(butter(_ORDER, [high, low], btype="band",
                               fs=rate, output="sos"))
    elif high:
        sections.append(butter(_ORDER, high, btype="highpass",
                               fs=rate, output="sos"))
    elif low:
        sections.append(butter(_ORDER, low, btype="lowpass",
                               fs=rate, output="sos"))
    if 0.0 < notch < nyquist:
        b, a = iirnotch(notch, _NOTCH_Q, fs=rate)
        sections.append(tf2sos(b, a))
    if not sections:
        return values

    # An IIR filter fed a NaN returns NaN for every sample after it, so the
    # gaps come out first and go back in afterwards. Zero is the neutral
    # filling: the signal is mean-removed by any high-pass anyway, and a
    # low-pass over a short gap barely sees it.
    gaps = ~np.isfinite(values)
    work = np.where(gaps, 0.0, values).astype(np.float64)
    try:
        for sos in sections:
            work = sosfiltfilt(sos, work)
    except ValueError as exc:
        # Almost always "the length of the input vector must be greater
        # than padlen", i.e. a recording shorter than the filter needs.
        log.debug("could not filter %d samples at %g Hz: %s",
                  values.size, rate, exc)
        return values
    out = work.astype(np.float32)
    out[gaps] = np.nan
    return out


def welch_psd(
    columns: list[np.ndarray], rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(freqs, power)`` for each column, by Welch's method.

    The window is four seconds where the recording allows it, which puts
    the lowest resolved frequency at 0.25 Hz. Respiration lives around
    0.2 to 0.4 Hz, so a shorter window would put the one feature a
    respiratory trace is read for in the first bin.
    """
    from scipy.signal import welch

    length = min(int(c.size) for c in columns)
    nperseg = int(min(length, max(256, round(rate * 4))))
    rows = []
    freqs = np.zeros(0)
    for values in columns:
        clean = np.nan_to_num(
            np.asarray(values[:length], dtype=np.float64), nan=0.0,
        )
        freqs, power = welch(clean, fs=rate, nperseg=nperseg)
        rows.append(power)
    return freqs, np.vstack(rows) if rows else np.zeros((0, 0))


class PhysioPlot(QWidget):
    """One curve per column, with a picker, scaling, filters and a PSD.

    Built lazily by its owner: a session that only ever looks at tables
    should not import pyqtgraph.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg

        self._pg = pg
        self._timing: dict = {}
        #: Every sample, as read. The filters never touch it.
        self._raw: list[np.ndarray] = []
        #: What is drawn: ``_raw`` filtered, or ``_raw`` itself.
        self._view: list[np.ndarray] = []
        #: ``_view`` normalised, cached: it does not depend on the height
        #: or the spacing, and recomputing it on every spin-box step is a
        #: full min/max scan of every sample.
        self._normalised: dict[int, np.ndarray] = {}
        self._boxes: list[QCheckBox] = []
        self._items: dict[int, object] = {}
        self._path: Optional[Path] = None
        self._total = 0
        #: How many real samples one point in ``_raw`` stands for. 1 while
        #: the table's own rows are on screen, more once the whole file is
        #: in and only if it was longer than ``_MAX_SAMPLES``.
        self._step = 1
        self._partial = False
        self._times: Optional[np.ndarray] = None
        self._filter_token = 0
        self._psd_token = 0
        self._psd_dialog: Optional[QWidget] = None
        self._read_worker = None
        self._filter_worker = None
        self._psd_worker = None

        # Display settings.
        self._normalise = True
        self._gain = 1.0
        self._spacing = 1.2

        # A spin box steps as it is dragged. Coalescing means one filter
        # run at the end of the drag rather than one per step.
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(250)
        self._filter_timer.timeout.connect(self._refilter)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Two WRAPPING bars: what to draw, then how to draw it. Separate
        # because a twelve-channel recording would otherwise push the
        # controls onto a row of their own anyway, and this way the split
        # is where it means something.
        self._picker_bar = QFrame()
        self._picker_bar.setObjectName("toolbar")
        self._picker = flow(self._picker_bar, h_spacing=10, v_spacing=4)
        self._picker.setContentsMargins(10, 4, 10, 4)
        outer.addWidget(self._picker_bar)

        self._control_bar = QFrame()
        self._control_bar.setObjectName("toolbar")
        self._controls = flow(self._control_bar, h_spacing=10, v_spacing=4)
        self._controls.setContentsMargins(10, 4, 10, 4)
        outer.addWidget(self._control_bar)
        self._build_controls()

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.showGrid(x=True, y=False, alpha=0.15)
        # Both axes: "squeeze it vertically" is a thing people do with the
        # wheel long before they look for a control that does it.
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.setMinimumWidth(60)
        outer.addWidget(self._plot, 1)

        self._caption = ElidedLabel("")
        self._caption.setObjectName("sidecar-footer-summary")
        outer.addWidget(self._caption)

        self.repaint_for_palette(CUR())

    # -- the control bar ---------------------------------------------------

    def _labelled(self, text: str, widget: QWidget, tip: str) -> None:
        """A control with its caption, added as ONE wrapping item.

        One item so a caption never wraps onto a different row from the
        control it names.
        """
        holder = QHBoxLayout()
        holder.setContentsMargins(0, 0, 0, 0)
        holder.setSpacing(4)
        label = QLabel(text)
        label.setToolTip(tip)
        widget.setToolTip(tip)
        holder.addWidget(label)
        holder.addWidget(widget)
        self._controls.addLayout(holder)

    def _spin(
        self, *, low: float, high: float, value: float, step: float,
        decimals: int, suffix: str = "", prefix: str = "",
        off_text: str = "",
    ) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setValue(value)
        if suffix:
            box.setSuffix(suffix)
        if prefix:
            box.setPrefix(prefix)
        if off_text:
            # Qt shows this instead of the number at the MINIMUM, which is
            # why every filter's "off" is its lowest value rather than a
            # separate tick box.
            box.setSpecialValueText(off_text)
        box.setKeyboardTracking(False)
        return box

    def _build_controls(self) -> None:
        self._raw_box = QCheckBox("Raw values")
        self._raw_box.setChecked(not self._normalise)
        self._raw_box.setToolTip(
            "Draw the numbers as they are. Off by default because a "
            "trigger is 0 or 5 and an ECG is fractions of a millivolt, so "
            "on one axis the ECG is a flat line at the bottom."
        )
        self._raw_box.toggled.connect(self._on_raw_toggled)
        self._controls.addWidget(self._raw_box)

        self._gain_box = self._spin(
            low=0.1, high=50.0, value=self._gain, step=0.25, decimals=2,
            prefix="x ",
        )
        self._gain_box.valueChanged.connect(self._on_gain_changed)
        self._labelled(
            "Height", self._gain_box,
            "How tall each channel's band is drawn. Turn it up to see a "
            "small signal, down when neighbouring channels overlap.",
        )

        self._spacing_box = self._spin(
            low=0.0, high=5.0, value=self._spacing, step=0.1, decimals=2,
        )
        self._spacing_box.valueChanged.connect(self._on_spacing_changed)
        self._labelled(
            "Spacing", self._spacing_box,
            "The gap between channels. Squeeze it to fit more on screen; "
            "at zero they are drawn on top of one another, which is how "
            "you compare two signals directly.",
        )

        self._hp_box = self._spin(
            low=0.0, high=1000.0, value=0.0, step=0.1, decimals=2,
            suffix=" Hz", off_text="Off",
        )
        self._hp_box.valueChanged.connect(self._on_filter_changed)
        self._labelled(
            "High-pass", self._hp_box,
            "Remove slow drift below this frequency. A respiratory belt "
            "that sags over a run, or a baseline that wanders, is drift.",
        )

        self._lp_box = self._spin(
            low=0.0, high=1000.0, value=0.0, step=1.0, decimals=2,
            suffix=" Hz", off_text="Off",
        )
        self._lp_box.valueChanged.connect(self._on_filter_changed)
        self._labelled(
            "Low-pass", self._lp_box,
            "Remove fast noise above this frequency, which is how a "
            "cardiac trace stops looking like fur.",
        )

        self._notch_box = self._spin(
            low=0.0, high=1000.0, value=0.0, step=10.0, decimals=1,
            suffix=" Hz", off_text="Off",
        )
        self._notch_box.valueChanged.connect(self._on_filter_changed)
        self._labelled(
            "Notch", self._notch_box,
            "Remove one narrow band. Set it to the mains frequency where "
            "the recording was made: 50 Hz in most of the world, 60 in "
            "the Americas and Japan.",
        )

        self._psd_btn = QPushButton("  Spectrum")
        self._psd_btn.setObjectName("tb-btn")
        self._psd_btn.setToolTip(
            "How much of the signal sits at each frequency, for the ticked "
            "channels, filtered exactly as they are drawn. A respiratory "
            "trace peaks near 0.3 Hz and a cardiac one near 1 Hz, so the "
            "spectrum says whether a channel is what it claims to be."
        )
        self._psd_btn.clicked.connect(self._show_psd)
        self._controls.addWidget(self._psd_btn)

        self._reset_btn = QPushButton("  Reset view")
        self._reset_btn.setObjectName("tb-btn")
        self._reset_btn.setToolTip(
            "Fit the whole recording back into the window, after panning "
            "or zooming with the mouse."
        )
        self._reset_btn.clicked.connect(self._reset_view)
        self._controls.addWidget(self._reset_btn)

        self._sync_control_enabled()

    def _sync_control_enabled(self) -> None:
        """Height and spacing describe bands, and raw mode has none."""
        for box in (self._gain_box, self._spacing_box):
            box.setEnabled(self._normalise)

    # -- background work ---------------------------------------------------

    def _start(self, attribute: str, work, on_result) -> None:
        """Run ``work`` on a fresh ``QThread``, replacing whatever was there.

        The previous worker is cancelled rather than waited for: cancelling
        suppresses its emission, and blocking the GUI thread until a filter
        finishes is the freeze this is here to avoid. The handlers guard on
        a token as well, because cancellation is best effort.
        """
        from ...workers.meeg_recording_loader import RecordingComputeWorker

        previous = getattr(self, attribute, None)
        if previous is not None:
            previous.cancel()
        worker = RecordingComputeWorker(work, parent=self)
        worker.finished_with_result.connect(on_result)
        worker.failed.connect(
            lambda message, name=attribute: log.warning(
                "physio %s failed: %s", name, message,
            )
        )
        worker.finished.connect(worker.deleteLater)
        setattr(self, attribute, worker)
        worker.start()

    # -- binding -----------------------------------------------------------

    def set_recording(
        self, path: Path, timing: dict, rows: list[list[str]],
    ) -> None:
        """Draw ``path``, from the sidecar's timing.

        The table's ``rows`` are shown at once so something is on screen,
        and the whole file is read on a worker and replaces them.
        """
        self._path = Path(path)
        self._timing = dict(timing)
        columns = self._timing.get("columns", [])
        self._raw = to_series(rows, len(columns))
        self._view = list(self._raw)
        self._normalised = {}
        self._total = len(rows)
        self._step = 1
        self._times = None
        self._items = {}
        self._plot.clear()
        self._rebuild_picker(columns)
        self._retune_filter_ranges()
        self._partial = True
        # Through the filters, not straight to the plot: the controls keep
        # their settings across files, so a caption saying "keeping above
        # 0.5 Hz" while an unfiltered preview is on screen would be a lie
        # for as long as the read takes.
        self._refilter()
        self._reset_view()
        source = self._path
        self._start(
            "_read_worker",
            lambda p=source: (p, *read_columns(p)),
            self._on_read,
        )

    def _on_read(self, result) -> None:
        path, columns, total, step = result
        if path != self._path or not columns:
            return
        self._raw = list(columns)
        self._total = int(total)
        self._step = max(1, int(step))
        self._partial = False
        self._times = None
        self._retune_filter_ranges()
        # The filters have to be re-run: what was on screen was the table's
        # first few thousand rows, and this is the recording.
        self._refilter()

    def _retune_filter_ranges(self) -> None:
        """Cap every filter at Nyquist, because past it there is nothing.

        Done from the sidecar's rate, so a 50 Hz respiratory channel offers
        cut-offs up to 25 Hz and a 400 Hz ECG up to 200, rather than both
        offering a thousand.
        """
        rate = float(self._timing.get("sampling_frequency", 0.0)) or 0.0
        nyquist = max(rate / 2.0, 0.0)
        for box in (self._hp_box, self._lp_box, self._notch_box):
            was = box.blockSignals(True)
            box.setMaximum(nyquist if nyquist > 0 else 0.0)
            box.blockSignals(was)

    def _rebuild_picker(self, columns: list[str]) -> None:
        self._picker.clear()
        self._boxes = []
        for index, name in enumerate(columns):
            box = QCheckBox(name)
            box.setChecked(True)
            box.setToolTip(f"Draw {name}.")
            box.toggled.connect(lambda _c: self.redraw())
            self._picker.addWidget(box)
            self._boxes.append(box)
            del index

    # -- controls ----------------------------------------------------------

    def _on_raw_toggled(self, checked: bool) -> None:
        self._normalise = not checked
        self._sync_control_enabled()
        self.redraw()
        self._reset_view()

    def _on_gain_changed(self, value: float) -> None:
        self._gain = float(value)
        self.redraw()

    def _on_spacing_changed(self, value: float) -> None:
        self._spacing = float(value)
        self.redraw()
        self._reset_view()

    def _on_filter_changed(self, _value: float) -> None:
        self._filter_timer.start()
        self._update_caption(self._drawn_count())

    def _filter_params(self) -> dict:
        return {
            "high_pass": float(self._hp_box.value()),
            "low_pass": float(self._lp_box.value()),
            "notch": float(self._notch_box.value()),
        }

    def _filters_are_off(self) -> bool:
        return not any(self._filter_params().values())

    def _refilter(self) -> None:
        """Re-derive what is drawn from what was read."""
        self._filter_timer.stop()
        if not self._raw:
            return
        if self._filters_are_off():
            self._view = list(self._raw)
            self._normalised = {}
            self.redraw()
            return
        self._filter_token += 1
        token = self._filter_token
        rate = float(self._timing.get("sampling_frequency", 1.0)) or 1.0
        # ``step`` only differs from 1 for a recording long enough to have
        # been strided on read, and then the samples ARE further apart.
        rate = rate / self._step
        raw = list(self._raw)
        params = self._filter_params()
        path = self._path

        def work():
            return (path, token, [filter_signal(c, rate, **params) for c in raw])

        self._start("_filter_worker", work, self._on_filtered)
        self._update_caption(self._drawn_count())

    def _on_filtered(self, result) -> None:
        path, token, columns = result
        if path != self._path or token != self._filter_token or not columns:
            return
        self._view = list(columns)
        self._normalised = {}
        self.redraw()

    def _reset_view(self) -> None:
        self._plot.getPlotItem().enableAutoRange()
        self._plot.getPlotItem().autoRange()

    # -- the spectrum ------------------------------------------------------

    def _checked_indices(self) -> list[int]:
        return [
            i for i in range(len(self._view))
            if i >= len(self._boxes) or self._boxes[i].isChecked()
        ]

    def _show_psd(self) -> None:
        indices = self._checked_indices()
        if not indices:
            return
        names = self._timing.get("columns", [])
        rate = float(self._timing.get("sampling_frequency", 1.0)) or 1.0
        rate = rate / self._step
        columns = [self._view[i] for i in indices]
        labels = [names[i] if i < len(names) else f"column {i + 1}"
                  for i in indices]
        self._psd_token += 1
        token = self._psd_token
        path = self._path
        self._psd_btn.setEnabled(False)

        def work():
            freqs, power = welch_psd(columns, rate)
            return path, token, {
                "freqs": freqs, "data": power,
                "ch_names": labels, "ch_types": labels,
            }

        self._start("_psd_worker", work, self._on_psd)

    def _on_psd(self, outcome) -> None:
        path, token, result = outcome
        self._psd_btn.setEnabled(True)
        if path != self._path or token != self._psd_token or not result:
            return
        from .psd_dialog import PsdDialog

        # Each column is its own kind, so the dialog's colours have to come
        # from the same cycle the traces use or the spectrum of the blue
        # trace would come out a different colour from the blue trace.
        names = list(result["ch_names"])
        colours = {
            name: self._curve_color(index)
            for index, name in enumerate(names)
        }
        dialog = PsdDialog(
            result, parent=self,
            color_for=lambda kind: colours.get(kind, CUR().get("text", "#888")),
            title=f"Spectrum - {self._path.name if self._path else ''}",
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._psd_dialog = dialog
        dialog.show()

    # -- drawing -----------------------------------------------------------

    def _banded(self, index: int) -> np.ndarray:
        """``_view[index]`` scaled to a band, from a cached normalisation.

        Normalising is a full scan for the minimum and maximum, and the
        result does not depend on the height or the spacing. Caching it
        takes a spin-box step on a million-sample channel from 170 ms to
        under twenty, which is the difference between a control that feels
        connected to the plot and one that does not.
        """
        values = self._view[index]
        cached = self._normalised.get(index)
        if cached is None or cached.size != values.size:
            cached = normalise(values)
            self._normalised[index] = cached
        return cached * np.float32(self._gain)

    def _curve_color(self, drawn: int) -> str:
        palette = CUR()
        token = _CURVE_TOKENS[drawn % len(_CURVE_TOKENS)]
        return palette.get(token, "#58a6ff")

    def _time_axis(self, length: int) -> np.ndarray:
        """Seconds for each drawn point, cached because it never changes.

        Point ``i`` is sample ``i * step``, which at ``rate`` samples a
        second is ``start + i * step / rate`` seconds into the run.
        """
        if self._times is not None and self._times.size == length:
            return self._times
        rate = float(self._timing.get("sampling_frequency", 1.0)) or 1.0
        start = float(self._timing.get("start_time", 0.0))
        self._times = (
            start + np.arange(length, dtype=np.float64) * (self._step / rate)
        )
        return self._times

    def _drawn_count(self) -> int:
        return len([i for i in self._checked_indices() if i < len(self._view)])

    def redraw(self) -> None:
        pg = self._pg
        plot_item = self._plot.getPlotItem()
        columns = self._timing.get("columns", [])
        if not columns or not self._view:
            for item in self._items.values():
                plot_item.removeItem(item)
            self._items = {}
            self._caption.setText("Nothing to plot.")
            return

        drawn = 0
        offset = 0.0
        ticks: list[tuple[float, str]] = []
        wanted = set(self._checked_indices())

        for index, name in enumerate(columns):
            item = self._items.get(index)
            if index >= len(self._view) or index not in wanted:
                if item is not None:
                    plot_item.removeItem(item)
                    self._items.pop(index, None)
                continue
            values = self._view[index]
            if values.size == 0:
                continue
            if self._normalise:
                values = self._banded(index) - np.float32(offset)
                ticks.append((-offset, name))
                offset += self._spacing
            times = self._time_axis(values.size)

            if item is None:
                # Peak downsampling clipped to the visible range: about two
                # points per pixel taken from the REAL data, recomputed on
                # every zoom, so zooming in shows detail rather than the
                # same few thousand points stretched out.
                #
                # EMPTY first, then added, then filled. Clipping to the view
                # needs a view: an item built with its data resolves its
                # view before it has a parent, gets the PlotWidget instead
                # of the ViewBox, and raises ``AttributeError:
                # autoRangeEnabled`` out of ``itemChange``, which is inside
                # Qt's event loop and therefore uncatchable at the call
                # site. pyqtgraph short-circuits a data-less item, so this
                # ordering never asks the question until there is a real
                # answer.
                item = pg.PlotDataItem(
                    pen=pg.mkPen(self._curve_color(drawn), width=1),
                    autoDownsample=True,
                    downsampleMethod="peak",
                    connect="finite",     # a NaN gap is a gap, not a line
                )
                plot_item.addItem(item)
                item.setClipToView(True)
                item.setData(times, values)
                self._items[index] = item
            else:
                item.setPen(pg.mkPen(self._curve_color(drawn), width=1))
                item.setData(times, values)
            drawn += 1

        axis = plot_item.getAxis("left")
        if self._normalise and ticks:
            # The channel's NAME where its band is, rather than a number
            # that means nothing once the signal has been normalised.
            axis.setTicks([[(pos, label) for pos, label in ticks], []])
            self._plot.setLabel("left", "")
        else:
            axis.setTicks(None)
            units = self._timing.get("units", "")
            self._plot.setLabel("left", units or "value")

        self._update_caption(drawn)

    def _update_caption(self, drawn: int) -> None:
        columns = self._timing.get("columns", [])
        rate = float(self._timing.get("sampling_frequency", 1.0)) or 1.0
        start = float(self._timing.get("start_time", 0.0))
        total = self._total or (self._view[0].size if self._view else 0)
        held = self._view[0].size if self._view else 0

        bits = [
            f"{drawn} of {len(columns)} channel(s)",
            f"{total:,} samples at {rate:g} Hz",
            f"starting at {start:g} s",
        ]
        if held and held < total:
            bits.append(f"held as {held:,} points")

        params = self._filter_params()
        described = []
        if params["high_pass"]:
            described.append(f"above {params['high_pass']:g} Hz")
        if params["low_pass"]:
            described.append(f"below {params['low_pass']:g} Hz")
        if described:
            bits.append("keeping " + " and ".join(described))
        if params["notch"]:
            bits.append(f"notched at {params['notch']:g} Hz")
        if self._partial:
            bits.append("reading the rest...")
        self._caption.setText(", ".join(bits))

    # -- theme -------------------------------------------------------------

    def repaint_for_palette(self, palette: dict) -> None:
        """Re-colour everything pyqtgraph draws.

        pyqtgraph reads no QSS, so a dark/light swap reaches the plot only
        if something tells it to. The background, the axes and every pen
        are re-set from the new palette here; ``redraw`` re-pens the curves
        because their colours are palette TOKENS, not fixed hex.
        """
        self._plot.setBackground(palette.get("bg", "#0d1117"))
        for axis in ("left", "bottom"):
            item = self._plot.getPlotItem().getAxis(axis)
            item.setPen(palette.get("border", "#30363d"))
            item.setTextPen(palette.get("muted", "#8b949e"))
        self.redraw()


__all__ = [
    "PhysioPlot",
    "filter_signal",
    "normalise",
    "read_columns",
    "read_timing",
    "sidecar_for",
    "to_series",
    "welch_psd",
]
