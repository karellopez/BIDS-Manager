"""The 4-D time-course graph: the voxel under the crosshair, across frames.

A ``scope`` x ``scope`` neighbourhood (1x1 up to 7x7) taken in the plane of
the current view. The x axis is real seconds when the frames have a known
time (``RepetitionTime``, ``VolumeTiming``, PET frame times, which are
uneven), the volume number otherwise; clicking a curve jumps to the frame
under the click.

ONE plot, whatever the neighbourhood. The old graph built one pyqtgraph plot,
axes and all, per voxel: a 7x7 neighbourhood was 49 plots and froze the
window for 640 ms (measured in a full MainWindow on a 1224-volume BOLD), and
dragging the crosshair with it open stalled 115 ms per move. Here every
neighbour is drawn by a SINGLE curve item, its series placed into its cell
and separated from the next by a gap; all the frame markers are one scatter
item; the cell grid is one line item. Four items, so building and updating
cost the same at 1x1 and at 7x7.

With one voxel the plot is an ordinary plot, with real axes in real units.
With a neighbourhood it is a grid of small multiples on a shared y range, so
the neighbours can be compared at a glance.

pyqtgraph reads no QSS: colours come from the viewer theme and are re-applied
on every theme change. Antialiasing is set per item, never process-wide.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from ....viz import views
from ....viz.scene import PLANE_AXIS
from ....viz.theme import parse_colour
from ..bridge import connect_while_alive
from ..context import ViewerContext

log = logging.getLogger(__name__)

_SCALINGS = (("raw", "Raw"), ("percent", "Percent change"), ("demean", "Demeaned"))
_X_AXES = (("auto", "Automatic"), ("seconds", "Seconds"), ("frames", "Volumes"))
_Y_LABEL = {"raw": "Value", "percent": "Change (%)", "demean": "Value minus mean"}

#: Inner margin of a small-multiple cell, as a fraction of the cell.
_MARGIN = 0.07
#: Above this many points the curve is drawn 1 px wide. Qt strokes a wider
#: pen through its general path stroker: a 7x7 neighbourhood of 1224 volumes
#: (60,000 points) took 284 ms to paint at 1.5 px and 1.8 ms at 1 px, which
#: Qt draws with its fast cosmetic line code.
_THICK_LINE_MAX_POINTS = 4000
#: Width of the left axis of the graph and of the physio strip under it.
_LEFT_AXIS_PX = 56


def physio_for_graph(path, t0: float, t1: float, points: int = 2000) -> dict:
    """Worker side: the run's physio (every file of the run on one clock),
    cut to ``[t0, t1]`` run seconds, one LANE per channel.

    A waveform lane is its short gaps bridged, robustly scaled to 0..1 (the
    0.5 and 99.5 percentiles, so one spike cannot flatten it) and
    min/max-decimated to about ``points`` points. An event lane (a trigger,
    detected heartbeats: values only where something happened) is its event
    times, drawn as ticks. What the GUI thread receives is a few thousand
    numbers per channel, whatever the recording's length.
    """
    from ....viz.compute.decimate import peak_decimate
    from ....viz.data import physio as P
    from ....viz.data.signal import open_physio

    src = open_physio(path, None, together=True)
    seconds = src.start_time + np.arange(src.n_times, dtype=float) / src.sfreq
    inside = np.flatnonzero((seconds >= t0) & (seconds <= t1))
    if inside.size == 0:
        raise ValueError("the physio does not overlap the run")
    s0, s1 = int(inside[0]), int(inside[-1]) + 1
    indices = list(range(len(src.ch_names)))
    data = src.read(indices, s0, s1)
    mask = src.gap_mask(indices, s0, s1)
    if mask is None or mask.shape != data.shape:
        mask = ~np.isfinite(data)
    xs = seconds[s0:s1]
    bridge = max(2, int(round(P.BRIDGE_SECONDS * src.sfreq)))
    lanes = []
    for i, row in enumerate(data):
        name, ch_type = src.ch_names[i], src.ch_types[i]
        role = P.channel_role(row, mask[i], ch_type)
        lane = {"name": name, "type": ch_type, "role": role, "x": np.empty(0),
                "y": np.empty(0), "ticks": np.empty(0), "note": ""}
        if role == "events":
            times, _values = P.event_onsets(row, mask[i], xs)
            lane["ticks"] = np.asarray(times, dtype=float)
            lane["note"] = P.describe_events(times)
        elif role == "waveform":
            filled, still, bridged = P.bridge_gaps(row, mask[i], bridge)
            finite = filled[np.isfinite(filled)]
            lo, hi = np.percentile(finite, (0.5, 99.5)) if finite.size else (0.0, 1.0)
            if hi > lo:
                scaled = np.clip((filled - lo) / (hi - lo), 0.0, 1.0)
            else:
                # A constant channel sits mid-lane, where it reads as a line,
                # not on the floor where it reads as missing.
                scaled = np.where(np.isfinite(filled), 0.5, np.nan)
            xd, yd = peak_decimate(xs, scaled, points // 2)
            lane["x"] = np.asarray(xd, dtype=float)
            lane["y"] = np.asarray(yd, dtype=float)
            gaps = int(np.count_nonzero(np.diff(np.r_[0, still.astype(np.int8)]) == 1))
            notes = []
            if bridged:
                notes.append(f"{bridged:,} dropped samples bridged")
            if gaps:
                notes.append(f"{gaps:,} gaps")
            lane["note"] = ", ".join(notes)
        else:
            lane["note"] = "no samples in this run"
        lanes.append(lane)
    return {"names": [ln["name"] for ln in lanes], "lanes": lanes}


def qc_for_graph(src) -> dict:
    """Worker side: the series' per-volume quality as lanes (frame-indexed):
    the global signal, DVARS, the volumes whose DVARS jumps above the
    box-plot fence, and the non-steady-state volumes at the start."""
    from ....viz.compute import qc

    skip = qc.non_steady_state(src)
    pv = qc.per_volume(src, skip=skip)
    n = pv["global"].size
    frames = np.arange(n, dtype=float)

    def scaled(values: np.ndarray, robust: bool) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return v
        lo, hi = (np.percentile(finite, (0.5, 99.5)) if robust
                  else (finite.min(), finite.max()))
        return np.clip((v - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.full_like(v, 0.5)

    fence = pv["fence"]
    lanes = [
        {"name": "global signal", "type": "misc", "role": "waveform", "frames": True,
         "x": frames, "y": scaled(pv["global"], True), "ticks": np.empty(0),
         "note": "the mean over the head, per volume"},
        {"name": "DVARS", "type": "ecg", "role": "waveform", "frames": True,
         "x": frames, "y": scaled(pv["dvars"], False), "ticks": np.empty(0),
         "note": (f"root mean square change from the previous volume, % of the mean; "
                  f"median {np.nanmedian(pv['dvars']):.2f} %, fence {fence:.2f} %")},
        {"name": "DVARS jumps", "type": "stim", "role": "events", "frames": True,
         "x": np.empty(0), "y": np.empty(0), "ticks": pv["flagged"].astype(float),
         "note": (f"{pv['flagged'].size} volumes above the upper box-plot fence "
                  "(75th percentile + 1.5 IQR, FSL's rule): look at them")},
    ]
    if skip:
        lanes.append({"name": "non-steady-state", "type": "stim", "role": "events",
                      "frames": True, "x": np.empty(0), "y": np.empty(0),
                      "ticks": np.arange(skip, dtype=float),
                      "note": f"{skip} bright volumes at the start, left out of the maps"})
    return {"names": [ln["name"] for ln in lanes], "lanes": lanes}


class TimecourseGraph(QWidget):
    """The graph panel (controls row + one plot)."""

    #: The panel needs this many pixels to show what it was asked to (the
    #: physio lanes): the host may grow it.
    wants_height = pyqtSignal(int)

    def __init__(self, ctx: ViewerContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setObjectName("pane-dark")
        self._context = None   # BidsContext of the file (frame times)
        # What is drawn, in DATA units, for queries and click mapping.
        self._dim = 0
        self._x: Optional[np.ndarray] = None
        self._x_is_time = False
        self._series: list[list[Optional[np.ndarray]]] = []
        self._voxels: list[list[Optional[tuple[int, int, int]]]] = []
        self._centres: dict[tuple[int, int], float] = {}
        self._ylim = (0.0, 1.0)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        import pyqtgraph as pg

        self._pg = pg
        lay.addLayout(self._build_controls())

        self.plot = pg.PlotWidget()
        pi = self.plot.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        self._vb = pi.getViewBox()
        self._vb.setMouseEnabled(x=False, y=False)
        self._vb.disableAutoRange()
        # Scrolling over the graph must not zoom it into nothing.
        # The plain wheel is the viewer's (it steps slices). Ctrl+wheel zooms
        # TIME, Shift+wheel pans it, double-click shows the whole run again.
        self.plot.wheelEvent = self._on_wheel
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        #: The time window shown, in the graph's x units (None: the run).
        self._time_view: Optional[tuple[float, float]] = None
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(150)
        self._zoom_timer.timeout.connect(self._draw_physio)
        # Seconds stay seconds: pyqtgraph would relabel a 1040 s run as
        # "1.04 ks".
        for name in ("left", "bottom"):
            pi.getAxis(name).enableAutoSIPrefix(False)
        self._grid_item = pg.PlotCurveItem(connect="pairs", antialias=False)
        self._center_item = pg.PlotCurveItem(connect="pairs", antialias=False)
        self._curve = pg.PlotCurveItem(antialias=True)
        self._markers = pg.ScatterPlotItem()
        # Every event of the run is ONE bar item (one band per event), under
        # the curve. Created once and hidden when empty, never removed.
        self._events_item = pg.BarGraphItem(x0=[0.0], x1=[0.0], y0=[0.0], y1=[0.0])
        self._events_item.setZValue(-5)
        self._events_item.setVisible(False)
        for item in (self._events_item, self._grid_item, self._center_item, self._curve,
                     self._markers):
            pi.addItem(item)
        # Both plots keep a left axis of ONE width, so their time axes line
        # up pixel for pixel (labels made the graph's wider than the strip's).
        pi.getAxis("left").setWidth(_LEFT_AXIS_PX)
        self.plot.setMinimumHeight(70)
        # Graph above, physio below, the divider the user's to drag.
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.setHandleWidth(6)
        self.split.addWidget(self.plot)
        lay.addWidget(self.split, 1)

        # The run's physio, on the same time axis, under the graph.
        self.physio_plot = pg.PlotWidget()
        ppi = self.physio_plot.getPlotItem()
        ppi.setMenuEnabled(False)
        ppi.hideButtons()
        ppi.getAxis("bottom").enableAutoSIPrefix(False)
        ppi.getAxis("left").setStyle(showValues=False)
        self.physio_plot.setMouseEnabled(x=False, y=False)
        self.physio_plot.wheelEvent = self._on_wheel
        self.physio_plot.mouseDoubleClickEvent = lambda _e: self.set_time_view(None)
        ppi.getAxis("left").setWidth(_LEFT_AXIS_PX)
        self.physio_plot.setMinimumHeight(60)
        self.physio_plot.setVisible(False)
        self._physio_curves: list = []
        # Lane names, pooled (created as needed, hidden when unused).
        self._physio_names: list = []
        self._physio_ticks = pg.PlotCurveItem(connect="pairs", antialias=False)
        ppi.addItem(self._physio_ticks)
        self._physio_rules = pg.PlotCurveItem(connect="pairs", antialias=False)
        self._physio_rules.setZValue(-10)
        ppi.addItem(self._physio_rules)
        self._physio_src = None
        self._physio_key = None
        self._physio_generation = 0
        self._physio_sized = False
        self._sized_lanes = 0
        # The series' per-volume quality, computed once per series.
        self._qc_src = None
        self._qc_key = None
        self._qc_generation = 0
        self.split.addWidget(self.physio_plot)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)
        self._sizes_timer = QTimer(self)
        self._sizes_timer.setSingleShot(True)
        self._sizes_timer.setInterval(400)
        self._sizes_timer.timeout.connect(self._remember_split)
        self.split.splitterMoved.connect(lambda *_a: self._sizes_timer.start())
        ctx.jobs.done.connect(self._on_job_done)
        ctx.jobs.failed.connect(self._on_job_failed)

        self._sync_controls()
        ctx.qstore.changed.connect(self._on_changed)
        connect_while_alive(ctx.theme_hub.changed, self, lambda w, _t: w.apply_theme())
        connect_while_alive(ctx.settings_hub.changed, self, lambda w, _s: w.apply_theme())
        self.apply_theme()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sidecar-footer-summary")
        return lbl

    def _build_controls(self) -> QHBoxLayout:
        ctx = self.ctx
        row = QHBoxLayout()
        row.setContentsMargins(8, 0, 8, 0)
        row.setSpacing(6)
        # Which series is plotted: a name, or a choice when there are several
        # (the anatomy's own series and an overlaid BOLD, say).
        self.series_label = QLabel("")
        self.series_label.setObjectName("sidecar-footer-summary")
        row.addWidget(self.series_label)
        self.series_combo = QComboBox()
        self.series_combo.setObjectName("ent-input")
        self.series_combo.setToolTip("The 4-D image whose time course is plotted; the "
                                     "volume slider and Play follow it too.")
        self.series_combo.currentIndexChanged.connect(self._on_series_choice)
        self.series_combo.setVisible(False)
        row.addWidget(self.series_combo)
        row.addSpacing(10)
        row.addWidget(self._label("Scope:"))
        self.scope_spin = QSpinBox()
        self.scope_spin.setRange(1, 4)
        self.scope_spin.setToolTip(
            "Neighbourhood around the crosshair voxel: 1 = the voxel, 2 = 3x3, "
            "3 = 5x5, 4 = 7x7, taken in the plane of the current view."
        )
        self.scope_spin.valueChanged.connect(lambda v: ctx.run("graph.set", scope=v))
        row.addWidget(self.scope_spin)
        row.addSpacing(10)
        row.addWidget(self._label("Dot size:"))
        self.dot_spin = QSpinBox()
        self.dot_spin.setRange(1, 20)
        self.dot_spin.setToolTip("Diameter of the marker at the current frame.")
        self.dot_spin.valueChanged.connect(lambda v: ctx.run("graph.set", dot=v))
        row.addWidget(self.dot_spin)
        row.addSpacing(12)
        self.marks_box = QCheckBox("Mark neighbors")
        self.marks_box.setToolTip("Off: only the centre voxel carries the frame marker.")
        self.marks_box.toggled.connect(lambda v: ctx.run("graph.set", mark_neighbors=bool(v)))
        row.addWidget(self.marks_box)
        row.addSpacing(12)
        row.addWidget(self._label("Scale:"))
        self.scaling_combo = QComboBox()
        self.scaling_combo.setObjectName("ent-input")
        for value, text in _SCALINGS:
            self.scaling_combo.addItem(text, value)
        self.scaling_combo.setToolTip(
            "Raw values; percent change from the voxel's mean (the usual way "
            "to read BOLD); or the mean removed."
        )
        self.scaling_combo.currentIndexChanged.connect(
            lambda i: ctx.run("graph.set", scaling=self.scaling_combo.itemData(i))
        )
        row.addWidget(self.scaling_combo)
        row.addSpacing(8)
        row.addWidget(self._label("Time:"))
        self.x_combo = QComboBox()
        self.x_combo.setObjectName("ent-input")
        for value, text in _X_AXES:
            self.x_combo.addItem(text, value)
        self.x_combo.setToolTip(
            "Automatic plots seconds whenever the sidecar times the volumes "
            "(RepetitionTime, VolumeTiming, PET frame times), else the volume "
            "number."
        )
        self.x_combo.currentIndexChanged.connect(
            lambda i: ctx.run("graph.set", x_axis=self.x_combo.itemData(i))
        )
        row.addWidget(self.x_combo)
        row.addSpacing(12)
        self.events_box = QCheckBox("Events")
        self.events_box.setToolTip(
            "The run's events.tsv on the same time axis: one band per event, "
            "coloured by trial type.")
        self.events_box.toggled.connect(lambda v: ctx.run("graph.set", events=bool(v)))
        row.addWidget(self.events_box)
        self.physio_box = QCheckBox("Physio")
        self.physio_box.setToolTip(
            "The run's physiological recordings under the graph, on the same "
            "time axis, each placed by its own StartTime.")
        self.physio_box.toggled.connect(lambda v: ctx.run("graph.set", physio=bool(v)))
        row.addWidget(self.physio_box)
        self.qc_box = QCheckBox("QC traces")
        self.qc_box.setToolTip(
            "Per volume, under the graph: the global signal, DVARS (how much the "
            "head changes from one volume to the next) and the volumes that jump.")
        self.qc_box.toggled.connect(lambda v: ctx.run("graph.set", qc=bool(v)))
        row.addWidget(self.qc_box)
        row.addStretch(1)
        self.export_button = QPushButton("Export...")
        self.export_button.setObjectName("tb-btn")
        self.export_button.setToolTip(
            "Save the plotted time courses as a CSV table: one row per volume, "
            "the time (or volume number) and one column per voxel, in the "
            "scaling shown.")
        self.export_button.clicked.connect(self._ask_export)
        row.addWidget(self.export_button)
        return row

    def _on_series_choice(self, _i: int) -> None:
        if self._series_syncing:
            return
        layer_id = self.series_combo.currentData()
        if layer_id:
            self.ctx.run("graph.set", layer=layer_id)

    _series_syncing = False

    def _sync_series(self, current) -> None:
        """Name the series plotted; offer a choice when there are several."""
        layers = views.series_layers(self.ctx.store)
        self._series_syncing = True
        try:
            if len(layers) > 1:
                ids = [lay.id for lay in layers]
                if [self.series_combo.itemData(i) for i in range(self.series_combo.count())] != ids:
                    self.series_combo.clear()
                    for lay in layers:
                        self.series_combo.addItem(lay.name or lay.id, lay.id)
                i = self.series_combo.findData(current.id)
                if i >= 0 and self.series_combo.currentIndex() != i:
                    self.series_combo.setCurrentIndex(i)
                self.series_combo.setVisible(True)
                self.series_label.setText("Series")
            else:
                self.series_combo.setVisible(False)
                name = current.name or current.id
                self.series_label.setText(name if len(name) <= 48 else name[:45] + "...")
                self.series_label.setToolTip(name)
        finally:
            self._series_syncing = False

    def _sync_controls(self) -> None:
        g = self.ctx.scene.graph
        for w, v in ((self.scope_spin, g.scope), (self.dot_spin, g.dot)):
            if w.value() != v:
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
        layer, src = views.series_layer(self.ctx.store)
        qc_ok = bool(src is not None and src.fully_loaded and src.loaded_frames >= 3)
        self.qc_box.setVisible(qc_ok)
        if self.qc_box.isChecked() != g.qc:
            self.qc_box.blockSignals(True)
            self.qc_box.setChecked(g.qc)
            self.qc_box.blockSignals(False)
        if self.marks_box.isChecked() != g.mark_neighbors:
            self.marks_box.blockSignals(True)
            self.marks_box.setChecked(g.mark_neighbors)
            self.marks_box.blockSignals(False)
        for combo, value in ((self.scaling_combo, g.scaling), (self.x_combo, g.x_axis)):
            i = combo.findData(value)
            if i >= 0 and combo.currentIndex() != i:
                combo.blockSignals(True)
                combo.setCurrentIndex(i)
                combo.blockSignals(False)
        ctx = self._context
        for box, on, available in (
            (self.events_box, g.events, bool(ctx is not None and ctx.events)),
            (self.physio_box, g.physio, bool(ctx is not None and ctx.physio_paths)),
        ):
            # Offered only where the run HAS events or physio.
            box.setVisible(available)
            if box.isChecked() != on:
                box.blockSignals(True)
                box.setChecked(on)
                box.blockSignals(False)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def set_context(self, bids_ctx) -> None:
        self._context = bids_ctx
        self._time_view = None
        self._physio_src = None
        self._physio_key = None
        self._physio_generation += 1
        self.ctx.jobs.cancel("graph-physio")
        self._sync_controls()
        self.refresh()

    def _on_changed(self, paths) -> None:
        if not self.isVisible():
            return
        if any(p in ("graph", "scene") for p in paths):
            self._sync_controls()
            self.refresh()
        elif any(p in ("cursor", "plane", "mode", "layers") or p.startswith("sources")
                 for p in paths):
            if any(p.startswith("sources") or p == "layers" for p in paths):
                # The series finished reading: the QC traces become possible.
                self._sync_controls()
            self.refresh()
        elif any(p.endswith(".frame") for p in paths):
            self.update_markers()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._sync_controls()
        self.refresh()

    def apply_theme(self) -> None:
        theme = self.ctx.theme
        pg = self._pg
        self.plot.setBackground(theme.plot_background)
        self._set_curve_pen()
        self._grid_item.setPen(pg.mkPen(theme.grid, width=1))
        cross = self.ctx.settings.crosshair.color
        self._center_item.setPen(pg.mkPen(cross, width=1.5))
        self._markers.setBrush(pg.mkBrush(cross))
        self._markers.setPen(pg.mkPen(cross))
        pi = self.plot.getPlotItem()
        for name in ("left", "bottom"):
            ax = pi.getAxis(name)
            ax.setPen(theme.plot_foreground)
            ax.setTextPen(theme.plot_foreground)

    def _set_curve_pen(self, n_points: int = 0) -> None:
        width = 1.5 if n_points <= _THICK_LINE_MAX_POINTS else 1.0
        self._curve.setPen(self._pg.mkPen(self.ctx.theme.plot_foreground, width=width))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _x_axis(self, n: int) -> tuple[np.ndarray, bool]:
        mode = self.ctx.scene.graph.x_axis
        times = None
        if mode != "frames" and self._context is not None:
            times = self._context.frame_times(n)
        if times is not None:
            return np.asarray(times, dtype=float), True
        return np.arange(n, dtype=float), False

    def _neighbours(self, center, src) -> list[list[Optional[tuple[int, int, int]]]]:
        """The voxels of the grid, in the plane of the active view."""
        g = self.ctx.scene.graph
        dim = 2 * (g.scope - 1) + 1
        half = dim // 2
        plane = self.ctx.scene.plane if self.ctx.scene.mode == "single" else self.ctx.active_plane
        ornt = views.orientation(src)
        normal_axis = ornt.data_of[PLANE_AXIS[plane]]
        a, b = [ax for ax in range(3) if ax != normal_axis]
        out = []
        for di in range(-half, half + 1):
            row = []
            for dj in range(-half, half + 1):
                v = list(center)
                v[a] += di
                v[b] += dj
                if all(0 <= v[k] < src.spatial[k] for k in range(3)):
                    row.append((v[0], v[1], v[2]))
                else:
                    row.append(None)
            out.append(row)
        return out

    @staticmethod
    def _scaled(ts: np.ndarray, scaling: str) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        if scaling == "percent":
            mean = float(np.nanmean(ts)) if ts.size else 0.0
            return (ts - mean) / mean * 100.0 if mean else ts * 0.0
        if scaling == "demean":
            return ts - float(np.nanmean(ts))
        return ts

    def refresh(self) -> None:
        if not self.isVisible():
            return
        store = self.ctx.store
        layer, src = views.series_layer(store)
        if layer is None or src is None:
            self._clear()
            return
        world = views.cursor_world(store)
        if world is None:
            return
        # The series' OWN voxel under the crosshair: an overlaid BOLD has
        # its own grid, 2-3 mm, wherever the anatomy beneath is.
        center = views.voxel_in(src, world)
        self._sync_series(layer)
        if center is None:
            self._clear()
            self.series_label.setText(f"{layer.name or layer.id}: the crosshair is outside it")
            return
        grid = self._neighbours(center, src)
        scaling = self.ctx.scene.graph.scaling
        # Times are matched against the FILE's frame count (the sidecar
        # describes every frame), then cut to what was read.
        x, is_time = self._x_axis(src.n_frames_total)
        n = src.loaded_frames
        x = x[:n]
        series = [[None if v is None else self._scaled(src.voxel_series(v)[:n], scaling)
                   for v in row] for row in grid]
        finite = [s for row in series for s in row if s is not None and s.size]
        if not finite or not x.size:
            self._clear()
            return
        centres: dict[tuple[int, int], float] = {}
        if len(grid) > 1:
            # Small multiples: each cell keeps its OWN baseline (its median)
            # and all share one vertical SCALE, as AFNI's graph window does.
            # One shared range drew every curve flat whenever neighbours'
            # means differed by more than their fluctuation (always, in
            # BOLD); one range per cell would make a 1 % wobble look as big
            # as a 10 % one. The scale is robust, or the bright
            # non-steady-state volumes at the start of a run would set it.
            half = 0.0
            for r, row in enumerate(series):
                for c, ts in enumerate(row):
                    if ts is None or not ts.size or not np.isfinite(ts).any():
                        continue
                    m = float(np.nanmedian(ts))
                    centres[(r, c)] = m
                    half = max(half, float(np.nanpercentile(np.abs(ts - m), 99.0)))
            half = half or 1.0
            lo, hi = -half, half
        else:
            lo = float(min(np.nanmin(s) for s in finite))
            hi = float(max(np.nanmax(s) for s in finite))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            self._clear()
            return
        if lo == hi:
            pad = 1.0 if lo == 0 else abs(lo) * 0.05
            lo, hi = lo - pad, hi + pad
        self._dim = len(grid)
        self._x = x
        self._x_is_time = is_time
        self._voxels = grid
        self._series = series
        self._centres = centres
        self._ylim = (lo, hi)
        self._draw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _cell_xy(self, r: int, c: int, x: np.ndarray, y: np.ndarray):
        """Data -> plot coordinates of cell (r, c) of the small multiples:
        the cell's own baseline subtracted, the shared scale applied, and
        anything beyond the scale pinned to the cell's edge (a spike stays
        in its own cell)."""
        dim = self._dim
        x0, x1 = float(self._x[0]), float(self._x[-1])
        lo, hi = self._ylim
        rel = np.clip(np.asarray(y, dtype=float) - self._centres.get((r, c), 0.0), lo, hi)
        sx = (1.0 - 2 * _MARGIN) / max(x1 - x0, 1e-12)
        sy = (1.0 - 2 * _MARGIN) / max(hi - lo, 1e-12)
        px = c + _MARGIN + (np.asarray(x, dtype=float) - x0) * sx
        py = (dim - 1 - r) + _MARGIN + (rel - lo) * sy
        return px, py

    def _draw(self) -> None:
        dim = self._dim
        pi = self.plot.getPlotItem()
        x = self._x
        if dim == 1:
            # One voxel: an ordinary plot with real axes in real units.
            ts = self._series[0][0]
            self._set_curve_pen(ts.size)
            self._curve.setData(x[: ts.size], ts, connect="finite")
            self._grid_item.setData([], [])
            self._center_item.setData([], [])
            pi.showAxis("left")
            pi.showAxis("bottom")
            pi.setLabel("bottom", "Time" if self._x_is_time else "Volume",
                        units="s" if self._x_is_time else None)
            pi.setLabel("left", _Y_LABEL.get(self.ctx.scene.graph.scaling, "Value"))
            lo, hi = self._ylim
            x_hi = float(x[-1]) if x[-1] > x[0] else float(x[0]) + 1.0
            xr = self._time_view or (float(x[0]), x_hi)
            self._vb.setRange(xRange=xr, yRange=(lo, hi),
                              padding=0.0 if self._time_view else 0.03)
        else:
            xs, ys = [], []
            gap = np.array([np.nan])
            for r, row in enumerate(self._series):
                for c, ts in enumerate(row):
                    if ts is None or not ts.size:
                        continue
                    px, py = self._cell_xy(r, c, x[: ts.size], ts)
                    xs += [px, gap]
                    ys += [py, gap]
            all_x = np.concatenate(xs)
            self._set_curve_pen(all_x.size)
            self._curve.setData(all_x, np.concatenate(ys), connect="finite")
            lines_x, lines_y = [], []
            for k in range(dim + 1):
                lines_x += [0, dim, k, k]
                lines_y += [k, k, 0, dim]
            self._grid_item.setData(np.asarray(lines_x, float), np.asarray(lines_y, float))
            h = dim // 2
            box_x = [h, h + 1, h + 1, h + 1, h + 1, h, h, h]
            box_y = [h, h, h, h + 1, h + 1, h + 1, h + 1, h]
            self._center_item.setData(np.asarray(box_x, float), np.asarray(box_y, float))
            pi.hideAxis("left")
            pi.hideAxis("bottom")
            self._vb.setRange(xRange=(0, dim), yRange=(0, dim), padding=0.0)
        self.update_markers()
        self._draw_events()
        self._draw_physio()

    # ------------------------------------------------------------------
    # The run around the series: events and physio
    # ------------------------------------------------------------------

    def _to_x(self, seconds: np.ndarray) -> Optional[np.ndarray]:
        """Run seconds into the graph's x units (seconds, or volumes when the
        axis counts volumes and the repetition time is known)."""
        if self._x_is_time:
            return np.asarray(seconds, dtype=float)
        tr = getattr(self._context, "tr", None)
        if tr:
            return np.asarray(seconds, dtype=float) / float(tr)
        return None

    def event_bands(self) -> list[tuple[float, float, str]]:
        """``(start, end, label)`` of every event drawn, in x units (tests)."""
        return list(getattr(self, "_bands", []))

    def _bands_in_x(self):
        """``(starts, ends, labels)`` in x units: the run's events, or for a
        diffusion series its b=0 volumes (where the signal jumps, which
        reads as an artefact until you know why). None: nothing to draw."""
        ctx = self._context
        x = self._x
        if ctx is None or x is None or not x.size:
            return None
        events = list(getattr(ctx, "events", []) or [])
        if events and self.ctx.scene.graph.events:
            onsets = self._to_x(np.asarray([e.onset for e in events], dtype=float))
            ends = self._to_x(np.asarray([e.onset + e.duration for e in events], dtype=float))
            if onsets is None:
                return None
            return onsets, ends, [e.label for e in events]
        bvals = getattr(ctx, "bvals", None)
        if bvals is not None and len(bvals) >= x.size and x.size > 1:
            from ....viz.bids import shells_of

            b0 = np.flatnonzero(shells_of(bvals[: x.size]) == 0)
            if b0.size == 0 or b0.size == x.size:
                return None
            half = float(x[1] - x[0]) / 2.0
            return x[b0] - half, x[b0] + half, ["b=0"] * b0.size
        return None

    def _draw_events(self) -> None:
        item = self._events_item
        self._bands = []
        found = self._bands_in_x() if self._dim else None
        if found is None:
            item.setVisible(False)
            return
        onsets, ends, names = found
        x0, x1 = float(self._x[0]), float(self._x[-1])
        # An instantaneous event is still a visible band: a fixed share of
        # the run, so it reads the same at any length.
        least = max((x1 - x0) * 0.002, 1e-9)
        ends = np.maximum(ends, onsets + least)
        labels = sorted(set(names))
        theme = self.ctx.theme
        colours = {lab: QColor(*parse_colour(theme.series(i))) for i, lab in enumerate(labels)}
        for c in colours.values():
            c.setAlpha(70)
        keep = (ends >= x0) & (onsets <= x1)
        onsets, ends = onsets[keep], ends[keep]
        kept = [n for n, k in zip(names, keep) if k]
        if not kept:
            item.setVisible(False)
            return
        self._bands = [(float(a), float(b), n) for a, b, n in zip(onsets, ends, kept)]
        brushes = [self._pg.mkBrush(colours[n]) for n in kept]
        if self._dim == 1:
            lo, hi = self._vb.viewRange()[1]
            item.setOpts(x0=onsets, x1=ends, y0=np.full(onsets.size, lo),
                         y1=np.full(onsets.size, hi), brushes=brushes,
                         pens=[self._pg.mkPen(None)] * onsets.size)
        else:
            # The same bands in every cell of the neighbourhood.
            dim = self._dim
            sx = (1.0 - 2 * _MARGIN) / max(x1 - x0, 1e-12)
            px0, px1, py0, py1, br = [], [], [], [], []
            for r in range(dim):
                for c in range(dim):
                    px0.append(c + _MARGIN + (np.clip(onsets, x0, x1) - x0) * sx)
                    px1.append(c + _MARGIN + (np.clip(ends, x0, x1) - x0) * sx)
                    base = (dim - 1 - r) + _MARGIN
                    py0.append(np.full(onsets.size, base))
                    py1.append(np.full(onsets.size, base + 1.0 - 2 * _MARGIN))
                    br += brushes
            item.setOpts(x0=np.concatenate(px0), x1=np.concatenate(px1),
                         y0=np.concatenate(py0), y1=np.concatenate(py1), brushes=br,
                         pens=[self._pg.mkPen(None)] * len(br))
        item.setVisible(True)

    def _run_seconds(self) -> Optional[tuple[float, float]]:
        """The span shown, in run seconds (for the physio read): the zoomed
        window when there is one, so a zoom reads the physio at full detail
        for just that stretch."""
        if self._x is None or not self._x.size:
            return None
        x0, x1 = self._time_view or (float(self._x[0]), float(self._x[-1]))
        if self._x_is_time:
            return x0, x1
        tr = getattr(self._context, "tr", None)
        return (x0 * tr, x1 * tr) if tr else None

    # -- zooming in time ---------------------------------------------------

    def _full_x(self) -> Optional[tuple[float, float]]:
        if self._x is None or not self._x.size:
            return None
        return float(self._x[0]), float(self._x[-1])

    def set_time_view(self, window: Optional[tuple[float, float]]) -> None:
        """Show ``window`` (graph x units) of the run, or all of it."""
        full = self._full_x()
        if window is not None and full is not None:
            lo, hi = sorted(window)
            span = full[1] - full[0]
            minimum = span / 2000.0 if span > 0 else 1e-6
            if hi - lo < minimum:
                mid = (lo + hi) / 2.0
                lo, hi = mid - minimum / 2, mid + minimum / 2
            lo, hi = max(lo, full[0]), min(hi, full[1])
            if hi - lo >= 0.98 * span:
                window = None
            else:
                window = (lo, hi)
        if window == self._time_view:
            return
        self._time_view = window
        if self._dim == 1:
            self._draw()
        else:
            self._paint_physio()
        # The physio for the new stretch, once the wheel settles.
        self._zoom_timer.start()

    def time_view(self) -> Optional[tuple[float, float]]:
        return self._time_view

    def _on_wheel(self, event, *_a, **_k) -> None:
        mods = event.modifiers()
        ctrl = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        delta = event.angleDelta().y() or event.angleDelta().x()
        full = self._full_x()
        if not (ctrl or shift) or not delta or full is None:
            event.ignore()
            return
        lo, hi = self._time_view or full
        if ctrl:
            # Around the pointer, read from the strip or the graph under it.
            widget = self.physio_plot if self.physio_plot.underMouse() else self.plot
            vb = widget.getPlotItem().getViewBox()
            at = vb.mapSceneToView(widget.mapToScene(event.position().toPoint())).x()
            if self._dim > 1 and widget is self.plot:
                at = (lo + hi) / 2.0
            at = min(max(at, lo), hi)
            factor = 0.8 if delta > 0 else 1.25
            self.set_time_view((at - (at - lo) * factor, at + (hi - at) * factor))
        else:
            shift_by = (hi - lo) * (0.1 if delta < 0 else -0.1)
            if lo + shift_by < full[0]:
                shift_by = full[0] - lo
            if hi + shift_by > full[1]:
                shift_by = full[1] - hi
            self.set_time_view((lo + shift_by, hi + shift_by))
        event.accept()

    _SPLIT_KEY = "graph-physio"

    def _remember_split(self) -> None:
        sizes = [int(v) for v in self.split.sizes()]
        if len(sizes) == 2 and all(v > 0 for v in sizes):
            self.ctx.settings_hub.update(lambda s: s.layout_sizes.update({self._SPLIT_KEY: sizes}))

    #: Pixels per physio lane, and for the graph above them.
    LANE_PX = 30
    GRAPH_PX = 150

    def _physio_px(self, lanes: int) -> int:
        return self.LANE_PX * lanes + 36

    def _size_split(self, lanes: int) -> None:
        """The physio its remembered share, else room for every lane. Asks
        the host for the height that needs, first."""
        need = self._physio_px(lanes)
        self.physio_plot.setMinimumHeight(min(need, 22 * lanes + 36))
        self.wants_height.emit(self.GRAPH_PX + need + 50)
        total = max(self.split.height(), self.GRAPH_PX + need)
        saved = self.ctx.settings.layout_sizes.get(self._SPLIT_KEY)
        if saved and len(saved) == 2 and sum(saved) > 0:
            below = int(total * saved[1] / sum(saved))
        else:
            below = min(need, total - self.GRAPH_PX)
        self.split.setSizes([max(total - below, 80), max(below, 60)])

    def _lanes(self) -> list[dict]:
        """Every lane the strip shows: the QC traces, then the physio."""
        out = []
        if self.ctx.scene.graph.qc and self._qc_src is not None:
            out += self._qc_src["lanes"]
        if self.ctx.scene.graph.physio and self._physio_src is not None:
            out += self._physio_src["lanes"]
        return out

    def _draw_physio(self) -> None:
        """The strip under the graph: per-volume QC traces and the run's
        physio, each computed once on a worker, never on a crosshair move."""
        ctx = self._context
        span = self._run_seconds()
        g = self.ctx.scene.graph
        # At any neighbourhood size: the strip has its own time axis, the
        # run's, which the graph's cells share in miniature.
        if g.physio and ctx is not None and ctx.physio_paths and span is not None:
            key = (tuple(str(p) for p in ctx.physio_paths), round(span[0], 6),
                   round(span[1], 6))
            if self._physio_key != key:
                # Read and decimated ONCE per run and span: the run's physio
                # is millions of samples.
                self._physio_key = key
                self._physio_src = None
                self._physio_generation += 1
                self.ctx.jobs.start("graph-physio", self._physio_generation,
                                    physio_for_graph, ctx.physio_paths[0], span[0], span[1])
        _layer, src = views.series_layer(self.ctx.store)
        if g.qc and src is not None and src.fully_loaded:
            key = (id(src), src.loaded_frames)
            if self._qc_key != key:
                self._qc_key = key
                self._qc_src = None
                self._qc_generation += 1
                self.ctx.jobs.start("graph-qc", self._qc_generation, qc_for_graph, src)
        lanes = self._lanes()
        if not lanes:
            self.physio_plot.setVisible(False)
            return
        first = not self.physio_plot.isVisible()
        self.physio_plot.setVisible(True)
        if (first and not self._physio_sized) or len(lanes) > self._sized_lanes:
            # Re-sized when lanes are added (QC traces after the physio, or
            # the other way round): eight lanes in four lanes' room is cramped.
            self._physio_sized = True
            self._sized_lanes = len(lanes)
            self._size_split(len(lanes))
        self._paint_physio()

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if tag == "graph-physio" and generation == self._physio_generation:
            self._physio_src = result
        elif tag == "graph-qc" and generation == self._qc_generation:
            self._qc_src = result
            self._physio_sized = False
        else:
            return
        self._draw_physio()

    def _on_job_failed(self, tag: str, generation: int, message: str) -> None:
        if tag == "graph-physio" and generation == self._physio_generation:
            self._physio_src = None
            self.ctx.status.emit(f"The run's physio could not be read: {message}")
        elif tag == "graph-qc" and generation == self._qc_generation:
            self._qc_src = None
            self.ctx.status.emit(f"The QC traces could not be computed: {message}")

    def _lane_x(self, lane: dict, values: np.ndarray) -> np.ndarray:
        """A lane's positions on the graph's axis: physio is in run seconds,
        QC traces are per VOLUME (placed at that volume's x)."""
        if not lane.get("frames"):
            return self._to_x(values)
        x = self._x
        idx = np.clip(np.asarray(values, dtype=int), 0, max(len(x) - 1, 0))
        return np.asarray(x, dtype=float)[idx]

    def physio_channels(self) -> list[str]:
        """The physio channels drawn (tests)."""
        got = self._physio_src
        if got is None or not self.physio_plot.isVisible() or not self.ctx.scene.graph.physio:
            return []
        return list(got["names"])

    def physio_lanes(self) -> list[dict]:
        """The lanes drawn: name, role, note (tests and the tooltip)."""
        if not self.physio_plot.isVisible():
            return []
        return [{k: lane[k] for k in ("name", "role", "note")} | {"ticks": int(lane["ticks"].size)}
                for lane in self._lanes()]

    def _paint_physio(self) -> None:
        """One lane per channel, top to bottom in file order: waveforms as
        lines, event channels as ticks; names in the left gutter."""
        pg = self._pg
        pi = self.physio_plot.getPlotItem()
        theme = self.ctx.theme
        self.physio_plot.setBackground(theme.plot_background)
        for name in ("left", "bottom"):
            ax = pi.getAxis(name)
            ax.setPen(theme.plot_foreground)
            ax.setTextPen(theme.plot_foreground)
        lanes = self._lanes()
        n = len(lanes)
        waves = [i for i, lane in enumerate(lanes) if lane["role"] == "waveform"]
        while len(self._physio_curves) < len(waves):
            curve = pg.PlotCurveItem(antialias=False)
            pi.addItem(curve)
            self._physio_curves.append(curve)
        for k, curve in enumerate(self._physio_curves):
            curve.setVisible(k < len(waves))
        tick_x, tick_y, rule_x, rule_y, labels = [], [], [], [], []
        x0, x1 = float(self._x[0]), float(self._x[-1])
        k = 0
        for i, lane in enumerate(lanes):
            base = n - 1 - i
            colour = theme.type_colour(lane["type"])
            if i < n - 1:
                rule_x += [x0, x1]
                rule_y += [base, base]
            if lane["role"] == "waveform":
                curve = self._physio_curves[k]
                k += 1
                curve.setPen(pg.mkPen(colour, width=1))
                curve.setData(self._lane_x(lane, lane["x"]), lane["y"] * 0.8 + base + 0.1,
                              connect="finite")
            elif lane["role"] == "events" and lane["ticks"].size:
                tx = self._lane_x(lane, lane["ticks"])
                tick_x.append(np.repeat(tx, 2))
                tick_y.append(np.tile([base + 0.12, base + 0.88], tx.size))
            label = lane["name"]
            if lane["role"] == "events":
                label += f"  {lane['ticks'].size} marks"
            labels.append((base + 1.0, label, colour))
        if tick_x:
            self._physio_ticks.setData(np.concatenate(tick_x), np.concatenate(tick_y))
            self._physio_ticks.setPen(pg.mkPen(theme.token("warning", "#d29922"), width=1))
        else:
            self._physio_ticks.setData([], [])
        self._physio_rules.setData(np.asarray(rule_x, float), np.asarray(rule_y, float))
        self._physio_rules.setPen(pg.mkPen(theme.grid, width=1))
        # Names INSIDE their lanes, top left, on the plot's colour: an axis
        # gutter wide enough for "external_trigger" would cost the graph
        # above the same width (the two axes are kept equal to align).
        while len(self._physio_names) < len(labels):
            item = pg.TextItem("", anchor=(0.0, 0.0))
            item.setZValue(20)
            pi.addItem(item)
            self._physio_names.append(item)
        fill = QColor(theme.plot_background)
        fill.setAlpha(200)
        for k, item in enumerate(self._physio_names):
            item.setVisible(k < len(labels))
            if k < len(labels):
                top, text, colour = labels[k]
                item.setText(text, color=colour)
                item.fill = pg.mkBrush(fill)
                # At the left edge of what is SHOWN: zoomed, the run's start
                # is off screen and the names went with it.
                item.setPos(self._time_view[0] if self._time_view else x0, top)
        if self._time_view is not None:
            pi.setXRange(*self._time_view, padding=0.0)
        else:
            pi.setXRange(x0, x1, padding=0.03)
        pi.setYRange(0, n, padding=0.0)
        pi.setLabel("bottom", "Time" if self._x_is_time else "Volume",
                    units="s" if self._x_is_time else None)
        self.physio_plot.setToolTip("\n".join(
            f"{lane['name']}: {lane['note']}" for lane in lanes if lane["note"]))

    def _marked_cells(self) -> list[tuple[int, int]]:
        """(row, col) of the cells that carry a marker, centre first."""
        h = self._dim // 2
        if not self.ctx.scene.graph.mark_neighbors:
            return [(h, h)]
        cells = [(h, h)]
        for r, row in enumerate(self._series):
            for c, ts in enumerate(row):
                if ts is not None and ts.size and (r, c) != (h, h):
                    cells.append((r, c))
        return cells

    def marker_points(self) -> list[tuple[float, float]]:
        """Where the markers are, in DATA units (x, value), centre first."""
        layer, src = views.series_layer(self.ctx.store)
        if layer is None or src is None or self._x is None or not self._series:
            return []
        t = views.frame_of(self.ctx.store, layer, src)
        out = []
        for r, c in self._marked_cells():
            ts = self._series[r][c]
            if ts is None or not ts.size:
                continue
            i = max(0, min(t, ts.size - 1))
            out.append((float(self._x[i]), float(ts[i])))
        return out

    def update_markers(self) -> None:
        points = self.marker_points()
        size = self.ctx.scene.graph.dot
        if not points:
            self._markers.setData([], [])
            return
        if self._dim == 1:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
        else:
            xs, ys = [], []
            for (r, c), (px, py) in zip(self._marked_cells(), points):
                mx, my = self._cell_xy(r, c, np.array([px]), np.array([py]))
                xs.append(float(mx[0]))
                ys.append(float(my[0]))
        self._markers.setData(xs, ys, size=size)

    def _clear(self) -> None:
        self._curve.setData([], [])
        self._grid_item.setData([], [])
        self._center_item.setData([], [])
        self._markers.setData([], [])
        self._events_item.setVisible(False)
        self.physio_plot.setVisible(False)
        self._bands = []
        self._dim = 0
        self._x = None
        self._series = []

    def _on_click(self, event) -> None:
        """Jump to the frame nearest the click; a double-click shows the
        whole run again after a zoom."""
        if self._x is None or not len(self._x):
            return
        if event.double() and self._time_view is not None:
            self.set_time_view(None)
            return
        pos = self._vb.mapSceneToView(event.scenePos())
        x = pos.x()
        if self._dim > 1:
            c = int(np.floor(x))
            if not 0 <= c < self._dim:
                return
            x0, x1 = float(self._x[0]), float(self._x[-1])
            frac = (x - c - _MARGIN) / (1.0 - 2 * _MARGIN)
            x = x0 + float(np.clip(frac, 0.0, 1.0)) * (x1 - x0)
        frame = int(np.argmin(np.abs(self._x - x)))
        self.ctx.run("frame.set", frame=frame)
        event.accept()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _ask_export(self) -> None:
        if self._x is None:
            return
        layer, src = views.series_layer(self.ctx.store)
        base = src.path.name.split(".")[0] if src is not None else "timecourse"
        path, _ = QFileDialog.getSaveFileName(self, "Export time courses",
                                              f"{base}_timecourse.csv", "CSV (*.csv)")
        if path:
            self.export_csv(path)

    def export_csv(self, path) -> None:
        """Write the plotted series: the x column, then one column per voxel
        (the FILE's own indices, centre first), in the scaling shown."""
        import csv
        from pathlib import Path

        if self._x is None or not self._series:
            raise ValueError("nothing is plotted")
        scaling = self.ctx.scene.graph.scaling
        tag = {"raw": "", "percent": " (percent change)", "demean": " (demeaned)"}[scaling]
        h = self._dim // 2
        cells = [(h, h)] + [(r, c) for r in range(self._dim) for c in range(self._dim)
                            if (r, c) != (h, h)]
        columns, names = [], []
        for r, c in cells:
            ts, vox = self._series[r][c], self._voxels[r][c]
            if ts is None or vox is None:
                continue
            columns.append(ts)
            names.append(f"voxel {vox[0]} {vox[1]} {vox[2]}{tag}")
        n = len(self._x)
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s" if self._x_is_time else "volume", *names])
            for i in range(n):
                writer.writerow([f"{self._x[i]:.6g}",
                                 *(f"{col[i]:.6g}" if i < col.size else "" for col in columns)])

    # ------------------------------------------------------------------
    # Queries (tests, screenshots, linked views)
    # ------------------------------------------------------------------

    def cell_count(self) -> int:
        """How many voxels are plotted (neighbours outside the volume are not)."""
        return sum(1 for row in self._series for ts in row if ts is not None)

    def center_series(self) -> Optional[np.ndarray]:
        if not self._series:
            return None
        h = self._dim // 2
        return self._series[h][h]

    def x_values(self) -> Optional[np.ndarray]:
        return None if self._x is None else self._x.copy()

    def x_is_time(self) -> bool:
        return self._x_is_time

    def marker_size(self) -> float:
        pts = self._markers.points()
        return float(pts[0].size()) if len(pts) else 0.0

    def mouse_locked(self) -> bool:
        return self._vb.state["mouseEnabled"] == [False, False]

    def view_to_scene(self, x: float, y: float):
        """Scene position of DATA point (x, y) in the centre cell (clicks)."""
        from PyQt6.QtCore import QPointF

        if self._dim > 1:
            h = self._dim // 2
            px, py = self._cell_xy(h, h, np.array([x]), np.array([y]))
            x, y = float(px[0]), float(py[0])
        return self._vb.mapViewToScene(QPointF(x, y))


__all__ = ["TimecourseGraph"]
