"""Draw a continuous recording's columns against time.

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

Two decisions worth stating.

**Every channel is normalised by default.** A trigger is 0 or 5 and an ECG
is fractions of a millivolt; on one shared axis the ECG is a flat line at the
bottom. Normalised, each channel is drawn in its own horizontal band and the
shapes are comparable, which is what the reader came for. Raw values are one
tick away for when the numbers themselves matter.

**It is drawn from a decimated view, not from every sample.** A twenty-minute
recording at 1000 Hz is 1.2 million points per channel, and asking a plot
widget to draw that is how a viewer freezes. The visible window is reduced to
about two points per horizontal pixel, which is the most a screen can show.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFrame, QVBoxLayout, QWidget

from ..theme_manager import CUR
from .flow_layout import flow
from .primitives import ElidedLabel

log = logging.getLogger(__name__)

#: The most points to hand the plot per channel. A screen cannot show more
#: than about two per pixel and a plot widget slows to a crawl well before
#: a million.
_MAX_POINTS = 4000

#: Colours cycled across channels. Palette tokens, so a theme swap follows.
_CURVE_TOKENS = ("accent", "teal", "purple", "text", "dim", "warn")


def sidecar_for(path: Path) -> Path:
    """The ``.json`` beside a ``.tsv`` or ``.tsv.gz``."""
    name = path.name
    for ext in (".tsv.gz", ".tsv"):
        if name.endswith(ext):
            return path.with_name(name[: -len(ext)] + ".json")
    return path.with_suffix(".json")


def read_timing(path: Path) -> Optional[dict]:
    """``{columns, sampling_frequency, start_time}`` or ``None``.

    ``None`` means this is not a continuous recording, which is how the
    caller decides whether to offer a plot at all. A table of onsets
    (``_events.tsv``) has no sampling frequency and is not one.
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
    return {
        "columns": [str(c) for c in columns],
        "sampling_frequency": rate,
        "start_time": start if math.isfinite(start) else 0.0,
    }


def to_series(rows: list[list[str]], n_columns: int) -> list[list[float]]:
    """Columns of floats from the table's rows; blanks and text become NaN.

    NaN rather than skipping: a gap in a recording is a fact about the
    recording, and closing it up would draw a signal that never happened.
    """
    series: list[list[float]] = [[] for _ in range(n_columns)]
    for row in rows:
        for index in range(n_columns):
            cell = row[index] if index < len(row) else ""
            try:
                series[index].append(float(cell))
            except (TypeError, ValueError):
                series[index].append(float("nan"))
    return series


def decimate(values: list[float], limit: int = _MAX_POINTS) -> tuple[list[int], list[float]]:
    """``(indices, values)`` reduced to at most ``limit`` points.

    Plain striding, not min/max binning. Binning preserves the envelope of
    a noisy signal better, and it also invents a vertical line between two
    samples that were never adjacent, which on a trigger channel reads as
    an extra pulse. For deciding whether a trigger is where it should be,
    not inventing one matters more.
    """
    count = len(values)
    if count <= limit:
        return list(range(count)), list(values)
    step = max(1, count // limit)
    indices = list(range(0, count, step))
    return indices, [values[i] for i in indices]


def normalise(values: list[float]) -> list[float]:
    """Scale to roughly -0.5..0.5 so channels can share one axis.

    A flat channel stays flat at zero rather than being amplified into
    noise, which is what dividing by a near-zero range would do.
    """
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return values
    low, high = min(finite), max(finite)
    span = high - low
    if span <= 0:
        return [0.0 if math.isfinite(v) else v for v in values]
    mid = (high + low) / 2.0
    return [
        (v - mid) / span if math.isfinite(v) else v for v in values
    ]


def read_columns(
    path: Path, limit: int = _MAX_POINTS,
) -> tuple[list[list[float]], int, int]:
    """``(columns, total_samples, step)``: decimated columns and what it took.

    The count and the STEP are returned separately because the columns are
    decimated, and both are needed to say anything true about them. The
    count keeps the caption honest: "4,016 samples" for a 991,826-sample
    ECG would be the plot lying about the recording. The step keeps the
    TIME AXIS honest: position ``i`` in a decimated column is sample
    ``i * step``, and reading it as sample ``i`` drew a 2,479-second
    recording as ten seconds of one.

    Reads the WHOLE file rather than the preview the table shows. The table
    is bounded at five thousand rows because nobody reads more than that;
    a plot of the first five thousand samples of a 1.4-million-sample
    trigger channel would be a picture of the first four seconds, drawn as
    though it were the recording. Silently.

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
    step = max(1, total // limit) if total > limit else 1
    out: list[list[float]] = []
    for name in frame.columns:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy()
        out.append([float(v) for v in values[::step]])
    return out, total, step


class _ReadSignals(QObject):
    """Bridges the read back to the GUI thread.

    Unparented deliberately, for the reason ``output_fs_pane._ScanSignals``
    spells out: destroying a QObject on one thread while another is inside
    its emit is undefined behaviour in Qt, not a Python error.
    """

    done = pyqtSignal(object, object, int, int)   # path, columns, total, step


class _ReadRunnable(QRunnable):
    def __init__(self, path: Path, signals: _ReadSignals) -> None:
        super().__init__()
        self._path = path
        self._signals = signals

    def run(self) -> None:
        try:
            columns, total, step = read_columns(self._path)
        except Exception:  # pragma: no cover - defensive
            log.exception("physio read failed for %s", self._path)
            columns, total, step = None, 0, 1
        self._signals.done.emit(self._path, columns, total, step)


class PhysioPlot(QWidget):
    """One curve per column, stacked, with a picker and a raw-values toggle.

    Built lazily by its owner: a session that only ever looks at tables
    should not import pyqtgraph.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg

        self._pg = pg
        self._timing: dict = {}
        self._series: list[list[float]] = []
        self._boxes: list[QCheckBox] = []
        self._normalise = True
        self._path: Optional[Path] = None
        self._total = 0
        # How many real samples one point in ``_series`` stands for. 1 while
        # the table's own rows are on screen, more once the whole file is in.
        self._step = 1
        self._partial = False
        self._signals = _ReadSignals()
        self._signals.done.connect(
            self._on_read, Qt.ConnectionType.QueuedConnection,
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # The channel picker and the one option, in a WRAPPING bar so a
        # recording with twelve channels does not pin the pane wide.
        self._bar = QFrame()
        self._bar.setObjectName("toolbar")
        self._bar_layout = flow(self._bar, h_spacing=10, v_spacing=4)
        self._bar_layout.setContentsMargins(10, 4, 10, 4)
        outer.addWidget(self._bar)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.showGrid(x=True, y=False, alpha=0.15)
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.setMinimumWidth(60)
        outer.addWidget(self._plot, 1)

        self._caption = ElidedLabel("")
        self._caption.setObjectName("sidecar-footer-summary")
        outer.addWidget(self._caption)

        self.repaint_for_palette(CUR())

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
        self._series = to_series(rows, len(columns))
        self._total = len(rows)
        self._step = 1
        self._rebuild_picker(columns)
        self._partial = True
        self.redraw()
        QThreadPool.globalInstance().start(
            _ReadRunnable(self._path, self._signals)
        )

    def _on_read(self, path, columns, total, step) -> None:
        if path != self._path or not columns:
            return
        self._series = columns
        self._total = int(total)
        self._step = max(1, int(step))
        self._partial = False
        self.redraw()

    def _rebuild_picker(self, columns: list[str]) -> None:
        self._bar_layout.clear()
        self._boxes = []

        for name in columns:
            box = QCheckBox(name)
            box.setChecked(True)
            box.toggled.connect(lambda _c: self.redraw())
            self._bar_layout.addWidget(box)
            self._boxes.append(box)

        raw = QCheckBox("Raw values")
        raw.setChecked(not self._normalise)
        raw.setToolTip(
            "Draw the numbers as they are. Off by default because a "
            "trigger is 0 or 5 and an ECG is fractions of a millivolt, so "
            "on one axis the ECG is a flat line at the bottom."
        )

        def on_raw(checked: bool) -> None:
            self._normalise = not checked
            self.redraw()

        raw.toggled.connect(on_raw)
        self._bar_layout.addWidget(raw)

    # -- drawing -----------------------------------------------------------

    def redraw(self) -> None:
        self._plot.clear()
        columns = self._timing.get("columns", [])
        if not columns or not self._series:
            self._caption.setText("Nothing to plot.")
            return

        rate = float(self._timing.get("sampling_frequency", 1.0)) or 1.0
        start = float(self._timing.get("start_time", 0.0))
        palette = CUR()
        drawn = 0
        offset = 0.0

        for index, name in enumerate(columns):
            if index >= len(self._series):
                break
            if index < len(self._boxes) and not self._boxes[index].isChecked():
                continue
            positions, values = decimate(self._series[index])
            if not values:
                continue
            if self._normalise:
                values = normalise(values)
                # Each channel in its own band, so they do not overlap.
                values = [v - offset if math.isfinite(v) else v
                          for v in values]
                offset += 1.2
            # ``p`` indexes the decimated column, so the sample it stands
            # for is ``p * step``. Reading it as the sample number is what
            # drew a forty-minute recording as ten seconds.
            times = [start + (p * self._step) / rate for p in positions]
            token = _CURVE_TOKENS[drawn % len(_CURVE_TOKENS)]
            self._plot.plot(
                times, values,
                pen=self._pg.mkPen(palette.get(token, "#58a6ff"), width=1),
                name=name,
                connect="finite",     # a NaN gap is a gap, not a line
            )
            drawn += 1

        total = self._total or (len(self._series[0]) if self._series else 0)
        shown = len(self._series[0]) if self._series else 0
        self._plot.setLabel(
            "left", "channels (normalised)" if self._normalise else "value",
        )
        self._caption.setText(
            f"{drawn} of {len(columns)} channel(s), {total:,} samples at "
            f"{rate:g} Hz, starting at {start:g} s"
            + ("" if shown >= total else f", drawn from {shown:,} points")
            + (" - reading the rest..." if getattr(self, "_partial", False)
               else "")
        )

    # -- theme -------------------------------------------------------------

    def repaint_for_palette(self, palette: dict) -> None:
        self._plot.setBackground(palette.get("bg", "#0d1117"))
        for axis in ("left", "bottom"):
            item = self._plot.getPlotItem().getAxis(axis)
            item.setPen(palette.get("border", "#30363d"))
            item.setTextPen(palette.get("muted", "#8b949e"))
        self.redraw()


__all__ = [
    "PhysioPlot",
    "read_columns",
    "decimate",
    "normalise",
    "read_timing",
    "sidecar_for",
    "to_series",
]
