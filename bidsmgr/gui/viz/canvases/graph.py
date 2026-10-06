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
from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMenu, QPushButton, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from ....viz import views
from ....viz.scene import PLANE_AXIS
from ....viz.theme import parse_colour
from ...widgets.primitives import ElidedLabel
from ..bridge import connect_while_alive
from ..context import ViewerContext

log = logging.getLogger(__name__)

_SCALINGS = (("raw", "Raw values"), ("percent", "% signal change"), ("demean", "Mean removed"))
_X_AXES = (("auto", "Automatic"), ("seconds", "Seconds"), ("frames", "Volume index"))
#: The neighbourhood choices: (scope, text). Scope n is a (2n-1)^2 square.
_SCOPES = ((1, "Voxel"), (2, "3 × 3"), (3, "5 × 5"), (4, "7 × 7"))
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


def _scaled(values: np.ndarray, robust: bool = False, *, floor: Optional[float] = None,
            top: Optional[float] = None) -> tuple[np.ndarray, float, float]:
    """Values mapped to 0..1 for a lane, with the range used: robust (0.5
    to 99.5 percentiles), the full range, or from ``floor`` (0 for FD and a
    share) up to at least ``top`` (so a threshold line has its place)."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return v, 0.0, 1.0
    lo, hi = (np.percentile(finite, (0.5, 99.5)) if robust
              else (float(finite.min()), float(finite.max())))
    if floor is not None:
        lo = floor
    if top is not None:
        hi = max(hi, top)
    if hi <= lo:
        return np.where(np.isfinite(v), 0.5, np.nan), float(lo), float(hi)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0), float(lo), float(hi)


def _lane(name: str, kind: str = "misc", **kw) -> dict:
    lane = {"name": name, "type": kind, "role": "waveform", "frames": True,
            "x": np.empty(0), "y": np.empty(0), "ticks": np.empty(0), "note": ""}
    lane.update(kw)
    return lane


def qc_for_graph(src, rows=None, path=None, root=None, slice_axis: int = 2, *,
                 cancel=None, progress=None) -> dict:
    """Worker side: the QC ``rows`` asked for (ids of ``qc.QC_ROWS``) as
    lanes, frame-indexed: ``{"rows": {id: [lane, ...]}, "skip": n}``. Each
    row is computed only when asked for; the expensive one, motion, is read
    from fMRIPrep's confounds when the run has them."""
    from ....viz.compute import motion as M
    from ....viz.compute import qc

    rows = set(rows if rows is not None else qc.QC_ROW_IDS)
    skip = qc.non_steady_state(src, cancel=cancel)
    n = src.loaded_frames
    frames = np.arange(n, dtype=float)
    out: dict[str, list[dict]] = {}
    if rows & {"dvars", "global"}:
        pv = qc.per_volume(src, skip=skip, cancel=cancel)
        if "global" in rows:
            y, _lo, _hi = _scaled(pv["global"], True)
            out["global"] = [_lane("global signal", "misc", x=frames, y=y,
                                   note="the mean over the head, per volume")]
        if "dvars" in rows:
            y, _lo, hi = _scaled(pv["dvars"], floor=0.0)
            fence = pv["fence"]
            out["dvars"] = [_lane(
                "DVARS", "eog", x=frames, y=y, ticks=pv["flagged"].astype(float),
                label=f"DVARS (% of mean), max {hi:.2f}",
                rule=(fence / hi if np.isfinite(fence) and hi > 0 else None),
                note=(f"root mean square change from the previous volume, % of the "
                      f"mean; median {np.nanmedian(pv['dvars']):.2f} %. Marked: "
                      f"{pv['flagged'].size} volumes above the upper box-plot fence "
                      f"({fence:.2f} %: 75th percentile + 1.5 IQR, FSL's rule)"))]
    if "motion" in rows:
        m = M.motion_for(src, path, root, skip=skip, cancel=cancel, progress=progress)
        fd_max = float(np.nanmax(m.fd)) if np.isfinite(m.fd).any() else 0.0
        y, _lo, hi = _scaled(m.fd, floor=0.0, top=M.FD_THRESHOLD_MM * 1.2)
        over = np.flatnonzero(np.nan_to_num(m.fd) > M.FD_THRESHOLD_MM).astype(float)
        trans = m.params[:, :3]
        rot = np.degrees(m.params[:, 3:])
        t_lo, t_hi = float(np.min(trans)), float(np.max(trans))
        r_lo, r_hi = float(np.min(rot)), float(np.max(rot))

        def joint(block: np.ndarray, lo: float, hi: float) -> list[np.ndarray]:
            span = hi - lo
            if span <= 0:
                return [np.full(n, 0.5) for _ in range(block.shape[1])]
            return [(block[:, k] - lo) / span for k in range(block.shape[1])]

        where = m.describe()
        out["motion"] = [
            _lane("framewise displacement", "ecg", x=frames, y=y, ticks=over,
                  label=f"FD (mm), max {fd_max:.2f}",
                  rule=M.FD_THRESHOLD_MM / hi if hi > 0 else None,
                  note=(f"framewise displacement (Power 2012), {where}; mean "
                        f"{np.nanmean(m.fd):.3f} mm. Marked: {over.size} volumes above "
                        f"{M.FD_THRESHOLD_MM:g} mm (the dashed line)")),
            _lane("translation", "misc", x=frames, ys=joint(trans, t_lo, t_hi), series=True,
                  legend=("x", "y", "z"), label=f"translation (mm), {t_lo:.2f} to {t_hi:.2f}",
                  note=f"the three translations, {where}"),
            _lane("rotation", "misc", x=frames, ys=joint(rot, r_lo, r_hi), series=True,
                  legend=("x", "y", "z"), label=f"rotation (degrees), {r_lo:.2f} to {r_hi:.2f}",
                  note=f"the three rotations, {where}"),
        ]
    if rows & {"outliers", "spikes", "carpet"}:
        sample = qc.sample_series(src, skip=skip, slice_axis=slice_axis, cancel=cancel)
        if "outliers" in rows:
            frac = qc.outlier_fraction(sample["series"], skip=skip)
            y, _lo, hi = _scaled(frac, floor=0.0, top=qc.OUTLIER_LIMIT * 1.2)
            over = np.flatnonzero(np.nan_to_num(frac) > qc.OUTLIER_LIMIT).astype(float)
            peak = float(np.nanmax(frac)) if np.isfinite(frac).any() else 0.0
            out["outliers"] = [_lane(
                "outlier voxels", "resp", x=frames, y=y, ticks=over,
                label=f"outlier voxels (%), max {peak * 100:.1f}",
                rule=qc.OUTLIER_LIMIT / hi if hi > 0 else None,
                note=(f"share of {sample['series'].shape[1]:,} sampled head voxels that "
                      f"are outliers (3dToutcount's rule). Marked: {over.size} volumes "
                      f"above {qc.OUTLIER_LIMIT * 100:g} % (the dashed line, "
                      "afni_proc.py's censoring default)"))]
        if "spikes" in rows:
            sp = qc.slice_spikes(sample["slice_means"], skip=skip)
            where = ", ".join(f"volume {int(t)} slice {int(z)}"
                              for t, z in zip(sp["flagged"][:6], sp["slice"][:6]))
            out["spikes"] = [_lane(
                "slice spikes", "stim", role="events", ticks=sp["flagged"].astype(float),
                note=(f"volumes where one slice departs from its own course by more "
                      f"than {qc.SPIKE_Z:g} robust SD" + (f": {where}" if where else
                                                         ": none")))]
        if "carpet" in rows:
            image = qc.carpet(sample["series"], sample["depth"], skip=skip)
            out["carpet"] = [_lane(
                "carpet", "misc", role="image", image=image, height=4,
                note=(f"{sample['series'].shape[1]:,} sampled voxels' signal over time "
                      f"(z-scored after drift removal), in {image.shape[0]} rows from "
                      "the head's edge (top) inwards: a vertical band across many rows "
                      "is motion or a spike (Power 2017)"))]
    out["nss"] = ([_lane("non-steady-state", "stim", role="events",
                         ticks=np.arange(skip, dtype=float),
                         note=f"{skip} bright volumes at the start, left out of the measures")]
                  if skip else [])
    return {"rows": out, "skip": skip}


class _CappedLabel(ElidedLabel):
    """A name that asks for at most ``cap`` pixels and elides beyond them, so
    a long file name never decides how narrow the graph may be."""

    def __init__(self, cap: int) -> None:
        super().__init__("", mode=Qt.TextElideMode.ElideMiddle)
        self._cap = int(cap)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        hint = super().sizeHint()
        return QSize(min(hint.width(), self._cap), hint.height())


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
        lay.addWidget(self._build_controls())

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
        #: QC rows asked of a worker for the current series (computed or not).
        self._qc_asked: set[str] = set()
        #: The carpet, pooled (created the first time, hidden when unused).
        self._carpet_item = None
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

    def _group(self, label: str, widget: QWidget, tip: str) -> QWidget:
        """A label and its control, kept together when the bar wraps."""
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        lbl = self._label(label)
        lbl.setToolTip(tip)
        widget.setToolTip(tip)
        h.addWidget(lbl)
        h.addWidget(widget)
        return box

    def _build_controls(self) -> QWidget:
        """The graph's controls, by purpose: WHAT is plotted (the series,
        the neighbourhood), HOW (values, time axis, marker), WHAT ELSE on the
        same axis (events, physio, quality rows), and export. A wrapping bar:
        a plain row of them was a 1139 px floor under the whole viewer."""
        from ...widgets.flow_layout import FlowBar

        ctx = self.ctx
        bar = FlowBar(h_spacing=10, v_spacing=4)
        bar.setContentsMargins(8, 0, 8, 0)
        # Which series is plotted: a name, or a choice when there are several
        # (the anatomy's own series and an overlaid BOLD, say).
        self.series_label = _CappedLabel(260)
        self.series_label.setObjectName("sidecar-footer-summary")
        bar.addWidget(self.series_label)
        self.series_combo = QComboBox()
        self.series_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.series_combo.setMinimumContentsLength(10)
        self.series_combo.setToolTip("The 4-D image whose time course is plotted; the "
                                     "volume slider and Play follow it too.")
        self.series_combo.currentIndexChanged.connect(self._on_series_choice)
        self.series_combo.setVisible(False)
        bar.addWidget(self.series_combo)
        self.scope_combo = QComboBox()
        for value, text in _SCOPES:
            self.scope_combo.addItem(text, value)
        self.scope_combo.currentIndexChanged.connect(
            lambda i: ctx.run("graph.set", scope=int(self.scope_combo.itemData(i))))
        bar.addWidget(self._group(
            "Neighbourhood", self.scope_combo,
            "Which voxels are plotted: the crosshair voxel alone, or a square "
            "around it in the plane of the active view, one small plot per voxel."))
        self.scaling_combo = QComboBox()
        for value, text in _SCALINGS:
            self.scaling_combo.addItem(text, value)
        self.scaling_combo.currentIndexChanged.connect(
            lambda i: ctx.run("graph.set", scaling=self.scaling_combo.itemData(i)))
        bar.addWidget(self._group(
            "Values", self.scaling_combo,
            "Raw values as stored; percent signal change from each voxel's own "
            "mean (the usual way to read BOLD); or the mean removed."))
        self.x_combo = QComboBox()
        for value, text in _X_AXES:
            self.x_combo.addItem(text, value)
        self.x_combo.currentIndexChanged.connect(
            lambda i: ctx.run("graph.set", x_axis=self.x_combo.itemData(i)))
        bar.addWidget(self._group(
            "Time axis", self.x_combo,
            "Automatic plots seconds whenever the sidecar times the volumes "
            "(RepetitionTime, VolumeTiming, PET FrameTimesStart), else the "
            "volume index."))
        self.dot_spin = QSpinBox()
        self.dot_spin.setRange(1, 20)
        self.dot_spin.setSuffix(" px")
        self.dot_spin.valueChanged.connect(lambda v: ctx.run("graph.set", dot=v))
        bar.addWidget(self._group(
            "Marker", self.dot_spin,
            "Diameter of the marker at the volume on screen."))
        self.marks_box = QCheckBox("On every voxel")
        self.marks_box.setToolTip("Draw the current-volume marker on every voxel of the "
                                  "neighbourhood; off, on the centre voxel only.")
        self.marks_box.toggled.connect(lambda v: ctx.run("graph.set", mark_neighbors=bool(v)))
        bar.addWidget(self.marks_box)
        bar.addSpacing(6)
        self.events_box = QCheckBox("Events")
        self.events_box.setToolTip(
            "The run's _events.tsv on the same time axis: one band per event, "
            "coloured by trial type.")
        self.events_box.toggled.connect(lambda v: ctx.run("graph.set", events=bool(v)))
        bar.addWidget(self.events_box)
        self.physio_box = QCheckBox("Physiology")
        self.physio_box.setToolTip(
            "The run's physiological recordings (_physio.tsv.gz: cardiac, "
            "respiration, triggers) in lanes under the graph, on the same time "
            "axis, each placed by its own StartTime.")
        self.physio_box.toggled.connect(lambda v: ctx.run("graph.set", physio=bool(v)))
        bar.addWidget(self.physio_box)
        self.qc_box = QCheckBox("QC")
        self.qc_box.setToolTip(
            "Quality control, per volume, in rows under the graph: head motion "
            "(framewise displacement and the six parameters), DVARS, outlier "
            "voxels, slice spikes, the global signal and a carpet plot. "
            "Computed when switched on (motion is read from fMRIPrep's "
            "confounds when the run has them); choose the rows with Rows.")
        self.qc_box.toggled.connect(lambda v: ctx.run("graph.set", qc=bool(v)))
        bar.addWidget(self.qc_box)
        self.qc_rows_button = QPushButton("Rows")
        self.qc_rows_button.setObjectName("tb-btn")
        self.qc_rows_button.setToolTip("Which quality-control rows are drawn.")
        rows_menu = QMenu(self.qc_rows_button)
        self._qc_row_actions = {}
        from ....viz.compute.qc import QC_ROWS

        for row_id, title, help_text in QC_ROWS:
            act = rows_menu.addAction(title)
            act.setCheckable(True)
            act.setToolTip(help_text)
            act.toggled.connect(lambda on, r=row_id: self._toggle_qc_row(r, on))
            self._qc_row_actions[row_id] = act
        rows_menu.setToolTipsVisible(True)
        self.qc_rows_button.setMenu(rows_menu)
        bar.addWidget(self.qc_rows_button)
        bar.addStretch(1)
        self.export_button = QPushButton("Export CSV...")
        self.export_button.setObjectName("tb-btn")
        self.export_button.setToolTip(
            "Save the plotted time courses as a CSV table: one row per volume, "
            "the time (or volume index) and one column per voxel, in the values "
            "shown.")
        self.export_button.clicked.connect(self._ask_export)
        bar.addWidget(self.export_button)
        self.controls = bar
        return bar

    def add_panel_buttons(self, buttons) -> None:
        """Buttons that act on the graph's PANEL (where it sits, maximised,
        in its own window), placed last in the controls."""
        for btn in buttons:
            self.controls.addWidget(btn)

    def _toggle_qc_row(self, row_id: str, on: bool) -> None:
        rows = list(self.ctx.scene.graph.qc_rows)
        if on and row_id not in rows:
            rows.append(row_id)
        elif not on and row_id in rows:
            rows.remove(row_id)
        else:
            return
        self.ctx.run("graph.set", qc_rows=rows)

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
                self.series_label.setText(name)
                self.series_label.setToolTip(name)
        finally:
            self._series_syncing = False

    def _sync_controls(self) -> None:
        g = self.ctx.scene.graph
        if self.dot_spin.value() != g.dot:
            self.dot_spin.blockSignals(True)
            self.dot_spin.setValue(g.dot)
            self.dot_spin.blockSignals(False)
        i = self.scope_combo.findData(int(g.scope))
        if i >= 0 and self.scope_combo.currentIndex() != i:
            self.scope_combo.blockSignals(True)
            self.scope_combo.setCurrentIndex(i)
            self.scope_combo.blockSignals(False)
        layer, src = views.series_layer(self.ctx.store)
        qc_ok = bool(src is not None and src.fully_loaded and src.loaded_frames >= 3)
        if self.qc_box.isHidden() == qc_ok:
            self.qc_box.setVisible(qc_ok)
            self.qc_rows_button.setVisible(qc_ok)
        if self.qc_box.isChecked() != g.qc:
            self.qc_box.blockSignals(True)
            self.qc_box.setChecked(g.qc)
            self.qc_box.blockSignals(False)
        self.qc_rows_button.setEnabled(g.qc)
        for row_id, act in self._qc_row_actions.items():
            if act.isChecked() != (row_id in g.qc_rows):
                act.blockSignals(True)
                act.setChecked(row_id in g.qc_rows)
                act.blockSignals(False)
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
        # Asked for, never insisted on: a minimum of 22 px a lane made the
        # whole viewer unable to shrink below ~430 px with physio open.
        self.physio_plot.setMinimumHeight(48)
        self.wants_height.emit(self.GRAPH_PX + need + 50)
        total = max(self.split.height(), self.GRAPH_PX + need)
        saved = self.ctx.settings.layout_sizes.get(self._SPLIT_KEY)
        if saved and len(saved) == 2 and sum(saved) > 0:
            below = int(total * saved[1] / sum(saved))
        else:
            below = min(need, total - self.GRAPH_PX)
        self.split.setSizes([max(total - below, 80), max(below, 60)])

    def _lanes(self) -> list[dict]:
        """Every lane the strip shows: the QC rows chosen, in their fixed
        order, then the physio."""
        from ....viz.compute.qc import QC_ROW_IDS

        g = self.ctx.scene.graph
        out = []
        if g.qc and self._qc_src is not None:
            have = self._qc_src["rows"]
            for row_id in QC_ROW_IDS:
                if row_id in g.qc_rows:
                    out += have.get(row_id, [])
            out += have.get("nss", [])
        if g.physio and self._physio_src is not None:
            out += self._physio_src["lanes"]
        return out

    @staticmethod
    def _units(lanes: list[dict]) -> int:
        """The strip's height in lane units (a carpet is several)."""
        return int(sum(lane.get("height", 1) for lane in lanes))

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
                self._qc_asked = set()
            have = set(self._qc_src["rows"]) if self._qc_src is not None else set()
            # Only the rows not yet computed. A new job replaces a running
            # one, so it asks for everything still missing, not only the
            # row just switched on; a row that failed is not retried until
            # the rows asked for change.
            missing = set(g.qc_rows) - have
            if missing and not missing <= self._qc_asked:
                self._qc_asked = missing
                self._qc_generation += 1
                bids = self._context
                axis = 2
                if bids is not None:
                    axis = {"i": 0, "j": 1, "k": 2}.get(
                        str(bids.sidecar.get("SliceEncodingDirection", "k"))[:1], 2)
                self.ctx.jobs.start(
                    "graph-qc", self._qc_generation, qc_for_graph, src, sorted(missing),
                    bids.path if bids is not None else None,
                    bids.root if bids is not None else None, axis)
        lanes = self._lanes()
        if not lanes:
            self.physio_plot.setVisible(False)
            return
        first = not self.physio_plot.isVisible()
        self.physio_plot.setVisible(True)
        units = self._units(lanes)
        if (first and not self._physio_sized) or units > self._sized_lanes:
            # Re-sized when lanes are added (QC traces after the physio, or
            # the other way round): eight lanes in four lanes' room is cramped.
            self._physio_sized = True
            self._sized_lanes = units
            self._size_split(units)
        self._paint_physio()

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if tag == "graph-physio" and generation == self._physio_generation:
            self._physio_src = result
        elif tag == "graph-qc" and generation == self._qc_generation:
            if self._qc_src is None:
                self._qc_src = result
            else:
                self._qc_src["rows"].update(result["rows"])
            self._physio_sized = False
        else:
            return
        self._draw_physio()

    def _on_job_failed(self, tag: str, generation: int, message: str) -> None:
        if tag == "graph-physio" and generation == self._physio_generation:
            self._physio_src = None
            self.ctx.status.emit(f"The run's physio could not be read: {message}")
        elif tag == "graph-qc" and generation == self._qc_generation:
            self.ctx.status.emit(f"The QC rows could not be computed: {message}")

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
        """One lane per channel or QC row, top to bottom: waveforms as lines
        (several in one lane share its scale: x, y and z), marks as ticks,
        a threshold as a dashed line, a carpet as an image; each lane's name
        and range inside it, top left."""
        pg = self._pg
        pi = self.physio_plot.getPlotItem()
        theme = self.ctx.theme
        self.physio_plot.setBackground(theme.plot_background)
        for name in ("left", "bottom"):
            ax = pi.getAxis(name)
            ax.setPen(theme.plot_foreground)
            ax.setTextPen(theme.plot_foreground)
        lanes = self._lanes()
        total = self._units(lanes)
        curves_needed = sum(len(lane.get("ys", ())) or 1 for lane in lanes
                            if lane["role"] == "waveform")
        while len(self._physio_curves) < curves_needed:
            curve = pg.PlotCurveItem(antialias=False)
            pi.addItem(curve)
            self._physio_curves.append(curve)
        for k, curve in enumerate(self._physio_curves):
            curve.setVisible(k < curves_needed)
        tick_x, tick_y, rule_x, rule_y, dash_x, dash_y, labels = [], [], [], [], [], [], []
        x0, x1 = float(self._x[0]), float(self._x[-1])
        carpet = None
        k = 0
        top = float(total)
        for i, lane in enumerate(lanes):
            h = float(lane.get("height", 1))
            base = top - h
            colour = theme.type_colour(lane["type"])
            if i < len(lanes) - 1:
                rule_x += [x0, x1]
                rule_y += [base, base]
            if lane["role"] == "waveform":
                ys = lane.get("ys") or [lane["y"]]
                xs = self._lane_x(lane, lane["x"])
                for j, y in enumerate(ys):
                    curve = self._physio_curves[k]
                    k += 1
                    pen = (parse_colour(theme.series(j)) if lane.get("series") else colour)
                    curve.setPen(pg.mkPen(pen, width=1))
                    curve.setData(xs, np.asarray(y, dtype=float) * 0.8 * h + base + 0.1 * h,
                                  connect="finite")
                if lane.get("rule") is not None:
                    level = base + 0.1 * h + float(lane["rule"]) * 0.8 * h
                    dash_x += [x0, x1]
                    dash_y += [level, level]
                if lane["ticks"].size:
                    # Marked volumes: short ticks along the lane's top edge.
                    tx = self._lane_x(lane, lane["ticks"])
                    tick_x.append(np.repeat(tx, 2))
                    tick_y.append(np.tile([top - 0.18 * h, top - 0.02 * h], tx.size))
            elif lane["role"] == "events" and lane["ticks"].size:
                tx = self._lane_x(lane, lane["ticks"])
                tick_x.append(np.repeat(tx, 2))
                tick_y.append(np.tile([base + 0.12 * h, base + 0.88 * h], tx.size))
            elif lane["role"] == "image":
                carpet = (lane, base, h)
            label = lane.get("label", lane["name"])
            if lane["ticks"].size or lane["role"] == "events":
                label += f"  {lane['ticks'].size} marked"
            # Several curves in one lane: their names in their own colours.
            legend = [(text, theme.series(j)) for j, text in enumerate(lane.get("legend", ()))]
            labels.append((top, label, colour, legend))
            top = base
        if tick_x:
            self._physio_ticks.setData(np.concatenate(tick_x), np.concatenate(tick_y))
            self._physio_ticks.setPen(pg.mkPen(theme.token("warning", "#d29922"), width=1))
        else:
            self._physio_ticks.setData([], [])
        self._physio_rules.setData(np.asarray(rule_x, float), np.asarray(rule_y, float))
        self._physio_rules.setPen(pg.mkPen(theme.grid, width=1))
        self._paint_thresholds(dash_x, dash_y)
        self._paint_carpet(carpet)
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
                lane_top, text, colour, legend = labels[k]
                if legend:
                    import html

                    item.setHtml(
                        f'<span style="color:{colour}">{html.escape(text)}</span>&nbsp;&nbsp;'
                        + "&nbsp;".join(f'<span style="color:{c}"><b>{html.escape(t)}</b></span>'
                                        for t, c in legend))
                else:
                    item.setText(text, color=colour)
                item.fill = pg.mkBrush(fill)
                # At the left edge of what is SHOWN: zoomed, the run's start
                # is off screen and the names went with it.
                item.setPos(self._time_view[0] if self._time_view else x0, lane_top)
        if self._time_view is not None:
            pi.setXRange(*self._time_view, padding=0.0)
        else:
            pi.setXRange(x0, x1, padding=0.03)
        pi.setYRange(0, total, padding=0.0)
        pi.setLabel("bottom", "Time" if self._x_is_time else "Volume",
                    units="s" if self._x_is_time else None)
        self.physio_plot.setToolTip("\n".join(
            f"{lane['name']}: {lane['note']}" for lane in lanes if lane["note"]))

    def _paint_thresholds(self, xs: list, ys: list) -> None:
        """The dashed threshold lines (FD 0.5 mm, 5 % outliers, the DVARS
        fence), one pooled item."""
        pg = self._pg
        if getattr(self, "_physio_dashes", None) is None:
            self._physio_dashes = pg.PlotCurveItem(connect="pairs", antialias=False)
            self._physio_dashes.setZValue(-5)
            self.physio_plot.getPlotItem().addItem(self._physio_dashes)
        pen = pg.mkPen(self.ctx.theme.token("warning", "#d29922"), width=1,
                       style=Qt.PenStyle.DashLine)
        self._physio_dashes.setPen(pen)
        self._physio_dashes.setData(np.asarray(xs, float), np.asarray(ys, float))

    def _paint_carpet(self, carpet) -> None:
        """The carpet lane as one image, pooled: hidden when not shown."""
        pg = self._pg
        if carpet is None:
            if self._carpet_item is not None:
                self._carpet_item.setVisible(False)
            return
        lane, base, h = carpet
        if self._carpet_item is None:
            self._carpet_item = pg.ImageItem()
            self._carpet_item.setZValue(-8)
            self.physio_plot.getPlotItem().addItem(self._carpet_item)
        image = np.nan_to_num(np.asarray(lane["image"], dtype=np.float32))
        # Rows top to bottom (the head's edge first); pyqtgraph's y runs up.
        self._carpet_item.setImage(image[::-1].T, levels=(-2.0, 2.0), autoLevels=False)
        x = np.asarray(self._x, dtype=float)
        n = image.shape[1]
        step = (x[-1] - x[0]) / (n - 1) if n > 1 and len(x) == n else 1.0
        left = float(x[0]) - step / 2.0
        from PyQt6.QtCore import QRectF

        self._carpet_item.setRect(QRectF(left, base + 0.05 * h, step * n, 0.9 * h))
        self._carpet_item.setVisible(True)

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
