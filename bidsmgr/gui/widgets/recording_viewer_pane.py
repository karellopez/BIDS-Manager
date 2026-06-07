"""EEG / MEG / iEEG recording viewer (Editor center pane, recording kind).

Sister widget to :class:`NiftiViewerPane` / :class:`TsvViewerPane` /
:class:`SidecarFormPane`. When the user clicks a recording file in the
BIDS tree (``.fif``, ``.edf``, ``.set``, ``.vhdr``, ``.cnt``, CTF ``.ds``,
...), :class:`bidsmgr.gui.editor_panel.EditorPanel` swaps its center pane
to this viewer.

Flow
----
1. **Metadata first.** Selecting a recording shows a themed metadata card
   (channels, sampling rate, duration, per-type counts, filters, bad
   channels, measurement date), read on a background thread
   (:class:`bidsmgr.workers.RecordingMetaWorker`, ``preload=False``).
2. **Load signal.** A "Load signal" button reads the full recording on a
   background thread (:class:`bidsmgr.workers.RecordingSignalWorker`,
   ``preload=True``) and switches to the interactive time-series viewer.
   Its **Close** button returns to the metadata card.

The time-series viewer is a restyled, annotation-free port of the MEEGqc
``qc_viewer`` time-series widget (pyqtgraph): channel-type filtering with
CTF axial-gradiometer handling, an individual-channel picker, visible
count / amplitude scale / time-window controls, navigation + channel
scroll + hover tooltips, raw-vs-normalise rendering, HP/LP/notch
filtering, resample (threaded), an interactive in-app PSD view
(pyqtgraph, MNE-like: per-channel + per-type-average tabs), and a
BIDS-native event overlay (sibling ``*_events.tsv`` plus stim-channel
``find_events``).

Theme handling: toolbar / metadata controls are QSS-driven; the
pyqtgraph plot reads the palette explicitly (it does not honour QSS),
following the :class:`NiftiViewerPane` pattern. Everything (reads, PSD,
resample) runs on ``QThread`` workers so the GUI never blocks; stale
results are dropped via a path guard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .. import icons
from ..theme_manager import CUR
from .primitives import PaneHeader
from .spinner import BusySpinner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recognised recording extensions (the Editor dispatch set).
#
# Deliberately excludes BrainVision sidecars (``.eeg`` / ``.vmrk``) and the
# raw partners opened via their header file: the user opens ``.vhdr``, not the
# binary ``.eeg``. ``mne.io.read_raw`` auto-dispatches every entry below (with
# the format-specific fallbacks in ``_read_raw``).
# ---------------------------------------------------------------------------
_VIEWER_FILE_EXTS: frozenset[str] = frozenset({
    ".fif", ".fif.gz", ".con", ".sqd", ".kdf",
    ".vhdr", ".edf", ".bdf", ".gdf", ".set", ".cnt", ".egi", ".mef", ".nwb",
})
_VIEWER_DIR_EXTS: frozenset[str] = frozenset({".ds", ".mff"})

# channel-type -> palette token, so trace colours follow the theme.
_TYPE_TOKENS: dict[str, str] = {
    "mag": "accent", "grad": "success", "eeg": "purple", "seeg": "purple",
    "ecog": "purple", "eog": "teal", "ecg": "error", "emg": "warning",
    "stim": "warning", "ref_meg": "dim", "misc": "dim", "bio": "teal",
    "resp": "teal", "dbs": "purple",
}

# Distinct colours (palette tokens) for event IDs / PSD curves.
_SERIES_TOKENS: tuple[str, ...] = (
    "accent", "success", "purple", "teal", "warning", "error", "dim",
)


def _full_ext(path) -> str:
    """Lower-case extension, preserving the ``.fif.gz`` double suffix."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".fif.gz"):
        return ".fif.gz"
    return p.suffix.lower()


def is_recording_path(path) -> bool:
    """True when *path* is an EEG/MEG/iEEG recording the viewer can open."""
    p = Path(path)
    ext = _full_ext(p)
    if p.is_dir():
        return ext in _VIEWER_DIR_EXTS
    return ext in _VIEWER_FILE_EXTS


# ---------------------------------------------------------------------------
# Qt-free read helpers (imported by the loader workers).
# ---------------------------------------------------------------------------
# Extensions ``mne.io.read_raw`` refuses to auto-dispatch (ambiguous between
# vendors). Try the generic reader first, then these in order. ``read_raw_ant``
# needs the optional ``antio`` package; absent it the error is surfaced cleanly.
_SPECIFIC_READERS: dict[str, tuple[str, ...]] = {
    ".cnt": ("read_raw_cnt", "read_raw_ant"),
    ".egi": ("read_raw_egi",),
    ".mff": ("read_raw_egi",),
}


def _read_raw(path, *, preload: bool):
    """Read a recording with MNE. ``preload`` controls full vs lazy load."""
    import mne

    p = str(path)
    try:
        return mne.io.read_raw(p, preload=preload, verbose=False)
    except Exception as primary:
        readers = _SPECIFIC_READERS.get(_full_ext(path))
        if not readers:
            raise
        last = primary
        for name in readers:
            fn = getattr(mne.io, name, None)
            if fn is None:
                continue
            try:
                return fn(p, preload=preload, verbose=False)
            except Exception as exc:  # noqa: BLE001 - try the next reader
                last = exc
        raise last


def _summarize_raw(raw, path) -> dict:
    """Build a plain display summary from an ``mne.io.Raw`` (no Qt)."""
    import mne

    info = raw.info
    ch_names = list(info["ch_names"])
    ch_types = [mne.channel_type(info, i) for i in range(len(ch_names))]
    counts: dict[str, int] = {}
    for t in ch_types:
        counts[t] = counts.get(t, 0) + 1
    available = sorted(set(ch_types))
    is_ctf = (
        "mag" in available
        and "grad" not in available
        and (
            "ref_meg" in available
            or getattr(raw, "compensation_grade", None) is not None
            or _full_ext(path) == ".ds"
        )
    )
    try:
        duration = float(raw.times[-1]) if raw.n_times else 0.0
    except Exception:
        duration = 0.0
    meas = info.get("meas_date")
    return {
        "filename": str(path),
        "name": Path(path).name,
        "n_channels": len(ch_names),
        "sfreq": float(info["sfreq"]),
        "duration": duration,
        "n_times": int(raw.n_times),
        "highpass": info.get("highpass"),
        "lowpass": info.get("lowpass"),
        "line_freq": info.get("line_freq"),
        "meas_date": str(meas) if meas else None,
        "ch_type_counts": counts,
        "available_ch_types": available,
        "is_ctf": is_ctf,
        "bads": list(info.get("bads") or []),
    }


def _events_sibling(path) -> Optional[Path]:
    """Return the sibling BIDS ``*_events.tsv`` for *path*, if present."""
    p = Path(path)
    name = p.name
    ext = _full_ext(p)
    if ext and name.lower().endswith(ext):
        stem = name[: -len(ext)]
    else:
        stem = p.stem
    base = stem.rsplit("_", 1)[0] if "_" in stem else stem
    for suffix in ("_events.tsv", "_events.tsv.gz"):
        cand = p.parent / f"{base}{suffix}"
        if cand.exists():
            return cand
    return None


def _read_events_tsv(path) -> List[tuple]:
    """Parse a BIDS ``events.tsv`` into ``[(onset_s, label), ...]``."""
    import csv
    import gzip

    out: List[tuple] = []
    is_gz = str(path).lower().endswith(".gz")
    try:
        opener = (
            gzip.open(path, "rt", encoding="utf-8", newline="")
            if is_gz
            else open(path, "r", encoding="utf-8", newline="")
        )
        with opener as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                onset = row.get("onset")
                if onset is None:
                    continue
                try:
                    t = float(onset)
                except (TypeError, ValueError):
                    continue
                label = (
                    row.get("trial_type")
                    or row.get("value")
                    or row.get("event_type")
                    or ""
                )
                out.append((t, str(label)))
    except Exception as exc:  # noqa: BLE001
        log.debug("could not read events.tsv %s: %s", path, exc)
        return []
    return out


def _series_color(index: int) -> str:
    pal = CUR()
    token = _SERIES_TOKENS[index % len(_SERIES_TOKENS)]
    return pal.get(token, pal.get("text", "#888888"))


def _type_color(ch_type: str) -> str:
    pal = CUR()
    return pal.get(_TYPE_TOKENS.get(ch_type, "dim"), pal.get("text", "#888888"))


# ===========================================================================
# In-app PSD dialog (pyqtgraph, MNE-like, interactive, two tabs)
# ===========================================================================
class _PsdDialog(QDialog):
    """Power Spectral Density viewer.

    Two tabs:
    * **Per channel** - every channel of the selected type overlaid (thin),
      plus the type mean (bold); a "Highlight" picker isolates one channel.
    * **Average** - the per-type mean with a +/-1 std shaded band + legend.

    Both plots are fully interactive (pyqtgraph: drag-pan, wheel/right-drag
    zoom, auto-range button) and carry a crosshair with a frequency / power
    readout, mirroring the useful parts of MNE's spectrum plot.
    """

    def __init__(self, result: dict, *, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Power spectral density")
        self.setObjectName("pane-dark")
        self.resize(900, 560)
        import pyqtgraph as pg

        self._pg = pg
        self._freqs = np.asarray(result["freqs"])
        self._data = np.asarray(result["data"])          # (n_ch, n_freqs)
        self._ch_names: List[str] = list(result["ch_names"])
        self._ch_types: List[str] = list(result["ch_types"])
        # Defensive clamp: only index rows that actually exist in ``data``
        # (compute_psd returns a channel subset, so a caller passing the full
        # name/type lists must never drive an out-of-range index).
        n = min(self._data.shape[0], len(self._ch_names), len(self._ch_types))
        self._by_type: dict[str, List[int]] = {}
        for i in range(n):
            self._by_type.setdefault(self._ch_types[i], []).append(i)
        self._db = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Shared top bar.
        top = QHBoxLayout()
        self._db_chk = QCheckBox("dB (10·log10)")
        self._db_chk.setChecked(True)
        self._db_chk.toggled.connect(self._on_db_toggled)
        top.addWidget(self._db_chk)
        top.addStretch(1)
        hint = QLabel("Drag to pan · wheel / right-drag to zoom · click ⟲ to reset")
        hint.setObjectName("sidecar-footer-summary")
        top.addWidget(hint)
        outer.addLayout(top)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)
        self._build_channel_tab()
        self._build_average_tab()

        self._redraw_channel()
        self._redraw_average()

    # ---- tab construction -------------------------------------------------
    def _build_channel_tab(self) -> None:
        pg = self._pg
        page = QWidget()
        page.setObjectName("pane-dark")
        v = QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        for t in sorted(self._by_type):
            self._type_combo.addItem(f"{t}  ({len(self._by_type[t])})", userData=t)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        ctrl.addWidget(self._type_combo)
        ctrl.addWidget(QLabel("Highlight:"))
        self._hi_combo = QComboBox()
        self._hi_combo.currentIndexChanged.connect(self._redraw_channel)
        ctrl.addWidget(self._hi_combo, 1)
        v.addLayout(ctrl)

        self._ch_plot = pg.PlotWidget()
        self._ch_plot.showGrid(x=True, y=True, alpha=0.15)
        self._ch_plot.setLabel("bottom", "Frequency", units="Hz")
        v.addWidget(self._ch_plot, 1)
        self._ch_readout = QLabel("")
        self._ch_readout.setObjectName("sidecar-footer-summary")
        v.addWidget(self._ch_readout)
        self._theme_plot(self._ch_plot)
        self._ch_cross = self._wire_crosshair(self._ch_plot, self._ch_readout)
        self._tabs.addTab(page, "Per channel")
        self._refresh_highlight_combo()

    def _build_average_tab(self) -> None:
        pg = self._pg
        page = QWidget()
        page.setObjectName("pane-dark")
        v = QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        self._avg_plot = pg.PlotWidget()
        self._avg_plot.showGrid(x=True, y=True, alpha=0.15)
        self._avg_plot.setLabel("bottom", "Frequency", units="Hz")
        self._avg_plot.addLegend(offset=(-10, 10))
        v.addWidget(self._avg_plot, 1)
        self._avg_readout = QLabel("")
        self._avg_readout.setObjectName("sidecar-footer-summary")
        v.addWidget(self._avg_readout)
        self._theme_plot(self._avg_plot)
        self._avg_cross = self._wire_crosshair(self._avg_plot, self._avg_readout)
        self._tabs.addTab(page, "Average (per type)")

    # ---- helpers ----------------------------------------------------------
    def _scale(self, arr):
        if self._db:
            return 10.0 * np.log10(np.maximum(arr, 1e-30))
        return arr

    def _ylabel(self) -> str:
        return "Power (dB)" if self._db else "Power"

    def _theme_plot(self, plot) -> None:
        pg = self._pg
        bg = self.palette().color(QPalette.ColorRole.Base)
        fg = self.palette().color(QPalette.ColorRole.Text)
        plot.setBackground(bg)
        for axis in ("left", "bottom"):
            ax = plot.getPlotItem().getAxis(axis)
            ax.setPen(pg.mkPen(color=fg))
            ax.setTextPen(pg.mkPen(color=fg))

    def _wire_crosshair(self, plot, readout: QLabel):
        pg = self._pg
        pen = pg.mkPen(color=self.palette().color(QPalette.ColorRole.Text), width=1,
                       style=Qt.PenStyle.DashLine)
        vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        vline.setZValue(10)
        hline.setZValue(10)
        plot.addItem(vline, ignoreBounds=True)
        plot.addItem(hline, ignoreBounds=True)

        def _moved(evt):
            pos = evt[0]
            vb = plot.getPlotItem().vb
            if not plot.sceneBoundingRect().contains(pos):
                return
            pt = vb.mapSceneToView(pos)
            vline.setPos(pt.x())
            hline.setPos(pt.y())
            readout.setText(
                f"{pt.x():.2f} Hz   ·   {pt.y():.2f} {self._ylabel()}"
            )

        proxy = pg.SignalProxy(plot.scene().sigMouseMoved, rateLimit=30, slot=_moved)
        return {"vline": vline, "hline": hline, "proxy": proxy}

    def _refresh_highlight_combo(self) -> None:
        t = self._type_combo.currentData()
        self._hi_combo.blockSignals(True)
        self._hi_combo.clear()
        self._hi_combo.addItem("(none)", userData=-1)
        for idx in self._by_type.get(t, []):
            self._hi_combo.addItem(self._ch_names[idx], userData=idx)
        self._hi_combo.blockSignals(False)

    # ---- draw -------------------------------------------------------------
    def _on_db_toggled(self, checked: bool) -> None:
        self._db = checked
        self._redraw_channel()
        self._redraw_average()

    def _on_type_changed(self) -> None:
        self._refresh_highlight_combo()
        self._redraw_channel()

    def _redraw_channel(self) -> None:
        pg = self._pg
        plot = self._ch_plot
        plot.clear()
        plot.addItem(self._ch_cross["vline"], ignoreBounds=True)
        plot.addItem(self._ch_cross["hline"], ignoreBounds=True)
        t = self._type_combo.currentData()
        idxs = self._by_type.get(t, [])
        if not idxs:
            return
        base = QColor(_type_color(t))
        thin = QColor(base)
        thin.setAlpha(60)
        thin_pen = pg.mkPen(color=thin, width=1)
        for i in idxs:
            plot.plot(self._freqs, self._scale(self._data[i]), pen=thin_pen)
        # Mean (bold).
        mean = np.mean(self._data[idxs], axis=0)
        plot.plot(self._freqs, self._scale(mean),
                  pen=pg.mkPen(color=base, width=2), name="mean")
        # Highlighted channel.
        hi = self._hi_combo.currentData()
        if hi is not None and hi >= 0:
            hi_color = self.palette().color(QPalette.ColorRole.Text)
            plot.plot(self._freqs, self._scale(self._data[hi]),
                      pen=pg.mkPen(color=hi_color, width=2))
        plot.setLabel("left", self._ylabel())

    def _redraw_average(self) -> None:
        pg = self._pg
        plot = self._avg_plot
        plot.clear()
        legend = plot.getPlotItem().legend
        if legend is not None:
            legend.clear()
        plot.addItem(self._avg_cross["vline"], ignoreBounds=True)
        plot.addItem(self._avg_cross["hline"], ignoreBounds=True)
        for t in sorted(self._by_type):
            idxs = self._by_type[t]
            arr = self._data[idxs]
            mean = self._scale(np.mean(arr, axis=0))
            color = QColor(_type_color(t))
            if arr.shape[0] > 1:
                std = np.std(self._scale(arr), axis=0)
                band = QColor(color)
                band.setAlpha(40)
                top = plot.plot(self._freqs, mean + std,
                                pen=pg.mkPen(color=band, width=1))
                bot = plot.plot(self._freqs, mean - std,
                                pen=pg.mkPen(color=band, width=1))
                fill = pg.FillBetweenItem(top, bot, brush=pg.mkBrush(band))
                plot.addItem(fill)
            plot.plot(self._freqs, mean, pen=pg.mkPen(color=color, width=2), name=t)
        plot.setLabel("left", self._ylabel())


# ===========================================================================
# Time-series view (the interactive viewer)
# ===========================================================================
class _TimeSeriesView(QWidget):
    """Interactive multi-channel EEG/MEG trace viewer (pyqtgraph)."""

    status_message = pyqtSignal(str)
    loading_changed = pyqtSignal(bool, str)
    close_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")

        self._raw = None
        self._sfreq = 1000.0
        self._ch_names: List[str] = []
        self._ch_types: List[str] = []
        self._available_ch_types: List[str] = []
        self._is_ctf = False
        self._duration = 0.0
        self._n_samples = 0

        self._visible_channels = 20
        self._channel_offset = 0
        self._time_start = 0.0
        self._time_window = 10.0
        self._scale_factor = 1.0
        self._active_ch_type = "all"
        self._selected_channels: Optional[Set[str]] = None
        self._normalize = False
        self._dark_plot = False

        self._display_indices: List[int] = []
        self._shown_ch_info: list = []
        self._overlay_items: list = []

        self._current_filter = None
        self._notch_freq = None
        self._current_filepath: Optional[str] = None
        self._current_root: Optional[Path] = None

        # Events (BIDS-native).
        self._show_events = False
        self._event_source = "auto"
        self._event_line_width = 2
        self._event_color_override: Optional[QColor] = None
        self._tsv_events: List[tuple] = []
        self._stim_events: List[tuple] = []
        self._stim_channels: List[str] = []

        self._resample_worker = None
        self._psd_worker = None

        import pyqtgraph as pg

        self._pg = pg
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(2)

        self._toolbar_widget = self._build_toolbar()
        self._toolbar_toggle = self._make_section_toggle(
            "Display controls", self._toolbar_widget, expanded=True
        )
        main.addWidget(self._toolbar_toggle)
        main.addWidget(self._toolbar_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self._label_area = QWidget()
        self._label_layout = QVBoxLayout(self._label_area)
        self._label_layout.setContentsMargins(2, 0, 2, 0)
        self._label_layout.setSpacing(0)
        label_scroll = QScrollArea()
        label_scroll.setWidget(self._label_area)
        label_scroll.setWidgetResizable(True)
        label_scroll.setFixedWidth(68)
        label_scroll.setObjectName("viewer-label-strip")
        label_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._plot_widget = self._pg.PlotWidget()
        self._plot_widget.setMinimumWidth(60)
        self._plot_widget.showGrid(x=True, y=False, alpha=0.15)
        self._plot_widget.setLabel("bottom", "Time", units="s")
        self._plot_widget.setMouseEnabled(x=True, y=False)
        self._plot_widget.getPlotItem().getAxis("left").setWidth(0)
        self._plot_widget.getPlotItem().getAxis("left").setStyle(
            showValues=False
        )
        self._plot_widget.wheelEvent = self._on_plot_wheel
        self._hover_proxy = self._pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._on_mouse_moved,
        )

        splitter.addWidget(label_scroll)
        splitter.addWidget(self._plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._chan_scroll = QScrollBar(Qt.Orientation.Vertical)
        self._chan_scroll.setToolTip("Scroll channels")
        self._chan_scroll.valueChanged.connect(self._on_channel_scroll)

        plot_row = QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.setSpacing(0)
        plot_row.addWidget(splitter, 1)
        plot_row.addWidget(self._chan_scroll)
        plot_container = QWidget()
        plot_container.setLayout(plot_row)
        main.addWidget(plot_container, 1)

        main.addWidget(self._build_navigation())

        self._events_widget = self._build_events_row()
        self._events_toggle = self._make_section_toggle(
            "Events", self._events_widget, expanded=False
        )
        main.addWidget(self._events_toggle)
        main.addWidget(self._events_widget)

        self._apply_plot_theme()

    def _make_section_toggle(
        self, title: str, widget: QWidget, *, expanded: bool
    ) -> QPushButton:
        btn = QPushButton(("▼ " if expanded else "▶ ") + title)
        btn.setCheckable(True)
        btn.setChecked(expanded)
        btn.setObjectName("viewer-section-toggle")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        widget.setVisible(expanded)

        def _toggle(checked: bool, w=widget, b=btn, t=title) -> None:
            w.setVisible(checked)
            b.setText(("▼ " if checked else "▶ ") + t)

        btn.clicked.connect(_toggle)
        return btn

    def _build_toolbar(self) -> QWidget:
        """Two rows of controls inside a horizontal scroll area.

        The scroll area means the toolbar never forces a wide minimum on the
        pane: when the Editor splitter narrows it, the toolbar scrolls
        horizontally instead of refusing to shrink.
        """
        container = QWidget()
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        # Row 1 - display
        row1 = QFrame()
        row1.setObjectName("toolbar")
        l1 = QHBoxLayout(row1)
        l1.setContentsMargins(10, 4, 10, 4)
        l1.setSpacing(6)

        l1.addWidget(QLabel("Type:"))
        self.cmb_ch_type = QComboBox()
        self.cmb_ch_type.addItem("all")
        self.cmb_ch_type.currentTextChanged.connect(self._on_ch_type_changed)
        l1.addWidget(self.cmb_ch_type)

        self.btn_select = QPushButton("  Channels")
        self.btn_select.setObjectName("tb-btn")
        icons.apply_button(self.btn_select, "channels")
        self.btn_select.setToolTip("Pick specific channels to display")
        self.btn_select.clicked.connect(self._open_channel_selector)
        l1.addWidget(self.btn_select)

        l1.addWidget(QLabel("Count:"))
        self.spn_n = QSpinBox()
        self.spn_n.setRange(1, 500)
        self.spn_n.setValue(self._visible_channels)
        self.spn_n.valueChanged.connect(self._on_n_changed)
        l1.addWidget(self.spn_n)

        l1.addWidget(QLabel("Scale:"))
        self.spn_scale = QDoubleSpinBox()
        self.spn_scale.setRange(0.01, 1000.0)
        self.spn_scale.setValue(1.0)
        self.spn_scale.setSingleStep(0.1)
        self.spn_scale.setDecimals(2)
        self.spn_scale.valueChanged.connect(self._on_scale_changed)
        l1.addWidget(self.spn_scale)

        l1.addWidget(QLabel("Window (s):"))
        self.spn_window = QDoubleSpinBox()
        self.spn_window.setRange(0.1, 3600.0)
        self.spn_window.setValue(self._time_window)
        self.spn_window.setSingleStep(1.0)
        self.spn_window.valueChanged.connect(self._on_window_changed)
        l1.addWidget(self.spn_window)

        self.chk_dark = QCheckBox("Dark plot")
        self.chk_dark.toggled.connect(self._on_dark_toggled)
        l1.addWidget(self.chk_dark)

        self.chk_norm = QCheckBox("Normalize")
        self.chk_norm.setToolTip(
            "Normalise each channel independently to prevent overlap.\n"
            "When off, raw signals share a per-type scale (may overlap)."
        )
        self.chk_norm.toggled.connect(self._on_norm_toggled)
        l1.addWidget(self.chk_norm)

        self.btn_reset = QPushButton("  Reset view")
        self.btn_reset.setObjectName("tb-btn")
        icons.apply_button(self.btn_reset, "reset_view")
        self.btn_reset.clicked.connect(self._reset_view)
        l1.addWidget(self.btn_reset)

        self.btn_close = QPushButton("  Close")
        self.btn_close.setObjectName("tb-btn")
        icons.apply_button(self.btn_close, "close_data")
        self.btn_close.setToolTip("Close the signal and return to the metadata view")
        self.btn_close.clicked.connect(self.close_requested.emit)
        l1.addWidget(self.btn_close)

        l1.addStretch(1)
        cl.addWidget(row1)

        # Row 2 - processing
        row2 = QFrame()
        row2.setObjectName("toolbar")
        l2 = QHBoxLayout(row2)
        l2.setContentsMargins(10, 4, 10, 4)
        l2.setSpacing(6)

        l2.addWidget(QLabel("HP (Hz):"))
        self.spn_hp = QDoubleSpinBox()
        self.spn_hp.setRange(0.0, 500.0)
        self.spn_hp.setDecimals(1)
        self.spn_hp.setSpecialValueText("Off")
        l2.addWidget(self.spn_hp)

        l2.addWidget(QLabel("LP (Hz):"))
        self.spn_lp = QDoubleSpinBox()
        self.spn_lp.setRange(0.0, 5000.0)
        self.spn_lp.setDecimals(1)
        self.spn_lp.setSpecialValueText("Off")
        l2.addWidget(self.spn_lp)

        l2.addWidget(QLabel("Notch (Hz):"))
        self.spn_notch = QDoubleSpinBox()
        self.spn_notch.setRange(0.0, 1000.0)
        self.spn_notch.setDecimals(1)
        self.spn_notch.setSpecialValueText("Off")
        l2.addWidget(self.spn_notch)

        self.btn_filter = QPushButton("  Apply filter")
        self.btn_filter.setObjectName("tb-btn")
        icons.apply_button(self.btn_filter, "filter")
        self.btn_filter.clicked.connect(self._apply_filters)
        l2.addWidget(self.btn_filter)

        self.btn_filter_reset = QPushButton("Reset filters")
        self.btn_filter_reset.setObjectName("tb-btn")
        self.btn_filter_reset.clicked.connect(self._reset_filters)
        l2.addWidget(self.btn_filter_reset)

        l2.addWidget(QLabel("Resample (Hz):"))
        self.spn_resample = QDoubleSpinBox()
        self.spn_resample.setRange(0.0, 10000.0)
        self.spn_resample.setDecimals(0)
        self.spn_resample.setSpecialValueText("Off")
        l2.addWidget(self.spn_resample)

        self.btn_resample = QPushButton("  Resample")
        self.btn_resample.setObjectName("tb-btn")
        icons.apply_button(self.btn_resample, "resample")
        self.btn_resample.clicked.connect(self._apply_resample)
        l2.addWidget(self.btn_resample)

        self.btn_psd = QPushButton("  PSD")
        self.btn_psd.setObjectName("tb-btn")
        icons.apply_button(self.btn_psd, "psd")
        self.btn_psd.setToolTip("Power spectral density (interactive, in-app)")
        self.btn_psd.clicked.connect(self._show_psd)
        l2.addWidget(self.btn_psd)

        l2.addStretch(1)
        cl.addWidget(row2)

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("viewer-toolbar-scroll")
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        scroll.setMaximumHeight(container.sizeHint().height() + 16)
        return scroll

    def _build_navigation(self) -> QWidget:
        nav = QFrame()
        nav.setObjectName("toolbar")
        lay = QHBoxLayout(nav)
        lay.setContentsMargins(10, 2, 10, 2)
        lay.setSpacing(4)

        self.btn_start = QPushButton("|<")
        self.btn_start.setObjectName("tb-btn")
        self.btn_start.setFixedWidth(40)
        self.btn_start.clicked.connect(lambda: self._navigate("start"))
        self.btn_prev = QPushButton("<")
        self.btn_prev.setObjectName("tb-btn")
        self.btn_prev.setFixedWidth(34)
        self.btn_prev.clicked.connect(lambda: self._navigate("prev"))
        self.btn_next = QPushButton(">")
        self.btn_next.setObjectName("tb-btn")
        self.btn_next.setFixedWidth(34)
        self.btn_next.clicked.connect(lambda: self._navigate("next"))
        self.btn_end = QPushButton(">|")
        self.btn_end.setObjectName("tb-btn")
        self.btn_end.setFixedWidth(40)
        self.btn_end.clicked.connect(lambda: self._navigate("end"))

        self.sld_time = QSlider(Qt.Orientation.Horizontal)
        self.sld_time.setRange(0, 1000)
        self.sld_time.valueChanged.connect(self._on_time_slider)
        self.lbl_time = QLabel("0.0 / 0.0 s")
        self.lbl_time.setObjectName("sidecar-footer-summary")

        lay.addWidget(self.btn_start)
        lay.addWidget(self.btn_prev)
        lay.addWidget(self.sld_time, 1)
        lay.addWidget(self.btn_next)
        lay.addWidget(self.btn_end)
        lay.addWidget(self.lbl_time)
        return nav

    def _build_events_row(self) -> QWidget:
        box = QFrame()
        box.setObjectName("toolbar")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(6)

        self.chk_events = QCheckBox("Show events")
        self.chk_events.setEnabled(False)
        self.chk_events.toggled.connect(self._on_events_toggled)
        lay.addWidget(self.chk_events)

        lay.addWidget(QLabel("Source:"))
        self.cmb_event_src = QComboBox()
        self.cmb_event_src.addItem("auto")
        self.cmb_event_src.setEnabled(False)
        self.cmb_event_src.currentTextChanged.connect(self._on_event_src_changed)
        lay.addWidget(self.cmb_event_src)

        lay.addWidget(QLabel("Width:"))
        self.spn_event_w = QSpinBox()
        self.spn_event_w.setRange(1, 10)
        self.spn_event_w.setValue(self._event_line_width)
        self.spn_event_w.setFixedWidth(54)
        self.spn_event_w.valueChanged.connect(self._on_event_width_changed)
        lay.addWidget(self.spn_event_w)

        self.btn_event_color = QPushButton("Color…")
        self.btn_event_color.setObjectName("tb-btn")
        self.btn_event_color.clicked.connect(self._pick_event_color)
        lay.addWidget(self.btn_event_color)
        self.btn_event_color_reset = QPushButton("Reset")
        self.btn_event_color_reset.setObjectName("tb-btn")
        self.btn_event_color_reset.clicked.connect(self._reset_event_color)
        lay.addWidget(self.btn_event_color_reset)

        lay.addStretch(1)
        return box

    # -------------------------------------------------------------- loading
    def set_current_filepath(self, path, root) -> None:
        self._current_filepath = str(path) if path else None
        self._current_root = root

    def load_raw(self, raw) -> None:
        """Accept a preloaded ``mne.io.Raw`` and render it."""
        import mne

        self._raw = raw
        info = raw.info
        self._sfreq = float(info["sfreq"])
        self._ch_names = list(raw.ch_names)
        self._n_samples = int(raw.n_times)
        self._duration = float(raw.times[-1]) if raw.n_times else 0.0
        self._ch_types = [
            mne.channel_type(info, i) for i in range(len(self._ch_names))
        ]
        self._time_start = 0.0
        self._channel_offset = 0
        self._current_filter = None
        self._notch_freq = None
        self._selected_channels = None
        self._available_ch_types = sorted(set(self._ch_types))
        self._is_ctf = (
            "mag" in self._available_ch_types
            and "grad" not in self._available_ch_types
            and (
                "ref_meg" in self._available_ch_types
                or getattr(raw, "compensation_grade", None) is not None
                or (
                    self._current_filepath
                    and _full_ext(self._current_filepath) == ".ds"
                )
            )
        )

        self.spn_window.blockSignals(True)
        self.spn_window.setMaximum(max(0.1, self._duration))
        self.spn_window.setValue(min(self._time_window, max(0.1, self._duration)))
        self._time_window = self.spn_window.value()
        self.spn_window.blockSignals(False)

        self._rebuild_ch_type_combo()
        self._update_display_indices()
        self._update_channel_scrollbar()
        self._extract_events(raw)
        self._refresh_event_controls()
        self._apply_plot_theme()
        self._redraw()

        display = []
        for ct in self._available_ch_types:
            display.append("mag (axial grad)" if self._is_ctf and ct == "mag" else ct)
        self.status_message.emit(
            f"Loaded {len(self._ch_names)} channels, "
            f"{self._duration:.1f}s @ {self._sfreq:.0f} Hz | "
            f"Types: {', '.join(display)}"
        )

    def _rebuild_ch_type_combo(self) -> None:
        self.cmb_ch_type.blockSignals(True)
        self.cmb_ch_type.clear()
        self.cmb_ch_type.addItem("all")
        has_mag = "mag" in self._available_ch_types
        has_grad = "grad" in self._available_ch_types
        if has_mag and has_grad:
            self.cmb_ch_type.addItem("mag+grad")
        for ct in self._available_ch_types:
            label = "mag (axial grad)" if self._is_ctf and ct == "mag" else ct
            self.cmb_ch_type.addItem(label, userData=ct)
        self.cmb_ch_type.setCurrentText("all")
        self._active_ch_type = "all"
        self.cmb_ch_type.blockSignals(False)

    # --------------------------------------------------------------- events
    def _extract_events(self, raw) -> None:
        import mne

        self._stim_events = []
        self._stim_channels = []
        try:
            stim = [
                ch
                for ch, t in zip(raw.ch_names, self._ch_types)
                if t == "stim"
            ]
            self._stim_channels = stim
            if stim:
                try:
                    events = mne.find_events(
                        raw, stim_channel=stim, shortest_event=1, verbose=False
                    )
                except Exception:
                    events = np.empty((0, 3), dtype=int)
                for ev in events:
                    eid = int(ev[2])
                    if eid != 0:
                        self._stim_events.append((float(ev[0]) / self._sfreq, eid))
        except Exception:
            pass

        self._tsv_events = []
        if self._current_filepath:
            sib = _events_sibling(self._current_filepath)
            if sib is not None:
                self._tsv_events = _read_events_tsv(sib)

    def _refresh_event_controls(self) -> None:
        sources = []
        if self._tsv_events:
            sources.append("events.tsv")
        if self._stim_events:
            sources.append("stim")
        has_any = bool(sources)
        self.chk_events.blockSignals(True)
        self.chk_events.setChecked(False)
        self.chk_events.setEnabled(has_any)
        self.chk_events.blockSignals(False)
        self._show_events = False
        self.cmb_event_src.blockSignals(True)
        self.cmb_event_src.clear()
        self.cmb_event_src.addItem("auto")
        for s in sources:
            self.cmb_event_src.addItem(s)
        self.cmb_event_src.setEnabled(has_any)
        self.cmb_event_src.blockSignals(False)
        self._event_source = "auto"
        if has_any:
            n = len(self._tsv_events) + len(self._stim_events)
            self.chk_events.setToolTip(
                f"Overlay {n} event marker(s) from "
                f"{' + '.join(sources)}"
            )
        else:
            self.chk_events.setToolTip(
                "No events found (no sibling events.tsv and no stim channel)"
            )

    def _active_events(self) -> List[tuple]:
        """Return ``[(time, label), ...]`` for the chosen source."""
        src = self._event_source
        out: List[tuple] = []
        if src in ("auto", "events.tsv") and self._tsv_events:
            out.extend((float(t), str(lbl)) for t, lbl in self._tsv_events)
            if src == "auto":
                return out
        if src in ("auto", "stim") and self._stim_events:
            out.extend((float(t), str(eid)) for t, eid in self._stim_events)
        return out

    # ------------------------------------------------------------- display
    def _update_display_indices(self) -> None:
        if self._active_ch_type == "all":
            indices = list(range(len(self._ch_names)))
        elif self._active_ch_type == "mag+grad":
            indices = [
                i for i, t in enumerate(self._ch_types) if t in ("mag", "grad")
            ]
        else:
            indices = [
                i
                for i, t in enumerate(self._ch_types)
                if t == self._active_ch_type
            ]
        if self._selected_channels is not None:
            indices = [
                i for i in indices if self._ch_names[i] in self._selected_channels
            ]
        self._display_indices = indices
        self._update_channel_scrollbar()

    def _update_channel_scrollbar(self) -> None:
        n = len(self._display_indices)
        visible = min(self._visible_channels, n)
        self._chan_scroll.setRange(0, max(0, n - visible))
        self._chan_scroll.setValue(self._channel_offset)

    def _get_data_segment(self, ch_indices, tmin, tmax):
        if self._raw is None:
            return None, None
        smin = max(0, int(tmin * self._sfreq))
        smax = min(self._n_samples, int(tmax * self._sfreq))
        if smax <= smin:
            return None, None
        try:
            data, times = self._raw[ch_indices, smin:smax]
        except Exception:
            return None, None
        if self._current_filter or self._notch_freq:
            import mne

            l_freq = (
                self._current_filter[0]
                if self._current_filter and self._current_filter[0]
                else None
            )
            h_freq = (
                self._current_filter[1]
                if self._current_filter and self._current_filter[1]
                else None
            )
            if l_freq or h_freq:
                try:
                    data = mne.filter.filter_data(
                        data, self._sfreq, l_freq, h_freq, verbose=False
                    )
                except Exception:
                    pass
            if self._notch_freq and self._notch_freq > 0:
                try:
                    data = mne.filter.notch_filter(
                        data, self._sfreq, self._notch_freq, verbose=False
                    )
                except Exception:
                    pass
        return data, times

    def _redraw(self) -> None:
        self._plot_widget.clear()
        self._clear_labels()
        self._overlay_items = []
        self._shown_ch_info = []
        if self._raw is None:
            return
        n_disp = len(self._display_indices)
        if n_disp == 0:
            return
        vis = min(self._visible_channels, n_disp)
        start = min(self._channel_offset, max(0, n_disp - vis))
        shown = self._display_indices[start:start + vis]
        n_shown = len(shown)
        if n_shown == 0:
            return
        tmin = self._time_start
        tmax = min(self._time_start + self._time_window, self._duration)
        data, times = self._get_data_segment(shown, tmin, tmax)
        if data is None:
            return
        scale = self._scale_factor
        plot_item = self._plot_widget.getPlotItem()
        # Channel labels sit on the themed strip, so they always use the
        # theme text colour (NOT the canvas colour) - fixes wash-out when
        # "Dark plot" is on in light theme.
        label_color = self.palette().color(QPalette.ColorRole.Text).name()

        if self._normalize:
            for i in range(n_shown):
                trace = data[i]
                rng = np.ptp(trace) if np.ptp(trace) > 0 else 1.0
                offset = n_shown - 1 - i
                y = ((trace - np.mean(trace)) / rng) * scale + offset
                self._plot_one(plot_item, times, y, shown[i], offset, label_color,
                               n_shown)
        else:
            type_ranges: dict = {}
            for i in range(n_shown):
                ct = self._ch_types[shown[i]]
                type_ranges.setdefault(ct, []).append(np.ptp(data[i]))
            type_scale = {}
            for ct, ranges in type_ranges.items():
                valid = [r for r in ranges if r > 0]
                type_scale[ct] = float(np.median(valid)) if valid else 1.0
            for i in range(n_shown):
                ct = self._ch_types[shown[i]]
                ref = type_scale.get(ct, 1.0) or 1.0
                trace = data[i]
                offset = n_shown - 1 - i
                y = ((trace - np.mean(trace)) / ref) * scale + offset
                self._plot_one(plot_item, times, y, shown[i], offset, label_color,
                               n_shown)

        plot_item.setXRange(tmin, tmax, padding=0)
        plot_item.setYRange(-0.5, n_shown - 0.5, padding=0.02)
        if self._show_events:
            self._draw_events(tmin, tmax, n_shown)
        self.lbl_time.setText(
            f"{tmin:.1f} - {tmax:.1f} / {self._duration:.1f} s"
        )
        if self._duration > 0:
            denom = max(0.01, self._duration - self._time_window)
            self.sld_time.blockSignals(True)
            self.sld_time.setValue(max(0, min(1000, int((tmin / denom) * 1000))))
            self.sld_time.blockSignals(False)

    def _plot_one(self, plot_item, times, y, ch_idx, offset, label_color,
                  n_shown) -> None:
        ch_name = self._ch_names[ch_idx]
        ch_type = self._ch_types[ch_idx]
        pen = self._pg.mkPen(color=_type_color(ch_type), width=1)
        plot_item.plot(times, y, pen=pen)
        self._add_channel_label(ch_name, n_shown, label_color)
        self._shown_ch_info.append((offset, ch_name, ch_type))

    def _clear_labels(self) -> None:
        while self._label_layout.count() > 0:
            item = self._label_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _add_channel_label(self, ch_name, n_shown, color) -> None:
        lbl = QLabel(ch_name)
        lbl.setFixedHeight(
            max(1, int(self._plot_widget.height() / max(n_shown, 1)))
        )
        lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        lbl.setStyleSheet(f"QLabel {{ color: {color}; font-size: 9px; }}")
        self._label_layout.addWidget(lbl)

    def _draw_events(self, tmin, tmax, n_shown) -> None:
        events = self._active_events()
        if not events:
            return
        visible = [e for e in events if tmin <= e[0] <= tmax]
        if len(visible) > 500:
            step = len(visible) // 500
            visible = visible[::step]
        labels = sorted({lbl for _, lbl in visible})
        color_map = {lbl: _series_color(i) for i, lbl in enumerate(labels)}
        override = self._event_color_override
        width = self._event_line_width
        for t, lbl in visible:
            color = (
                override.name() if override and override.isValid()
                else color_map.get(lbl, _series_color(0))
            )
            pen = self._pg.mkPen(color=color, width=width)
            line = self._pg.InfiniteLine(pos=t, angle=90, pen=pen, movable=False)
            self._plot_widget.addItem(line)
            self._overlay_items.append(line)
            if lbl:
                txt = self._pg.TextItem(str(lbl), color=color, anchor=(0.5, 1.0))
                txt.setPos(t, n_shown - 0.3)
                font = txt.textItem.font()
                font.setPointSize(7)
                txt.setFont(font)
                self._plot_widget.addItem(txt)
                self._overlay_items.append(txt)

    # ------------------------------------------------------------- handlers
    def _on_ch_type_changed(self, text) -> None:
        idx = self.cmb_ch_type.currentIndex()
        user_data = self.cmb_ch_type.itemData(idx)
        self._active_ch_type = user_data if user_data else text
        self._channel_offset = 0
        self._update_display_indices()
        self._redraw()

    def _on_n_changed(self, val) -> None:
        self._visible_channels = val
        self._update_channel_scrollbar()
        self._redraw()

    def _on_scale_changed(self, val) -> None:
        self._scale_factor = val
        self._redraw()

    def _on_window_changed(self, val) -> None:
        self._time_window = val
        self._redraw()

    def _on_time_slider(self, val) -> None:
        if self._duration <= 0:
            return
        max_start = max(0.0, self._duration - self._time_window)
        self._time_start = (val / 1000.0) * max_start
        self._redraw()

    def _on_channel_scroll(self, val) -> None:
        self._channel_offset = val
        self._redraw()

    def _on_plot_wheel(self, event) -> None:
        delta = event.angleDelta().y()
        if delta != 0:
            step = -1 if delta > 0 else 1
            new = max(
                self._chan_scroll.minimum(),
                min(self._chan_scroll.maximum(), self._chan_scroll.value() + step),
            )
            self._chan_scroll.setValue(new)
        event.accept()

    def _navigate(self, direction) -> None:
        step = self._time_window * 0.8
        if direction == "start":
            self._time_start = 0.0
        elif direction == "prev":
            self._time_start = max(0.0, self._time_start - step)
        elif direction == "next":
            self._time_start = min(
                max(0.0, self._duration - self._time_window),
                self._time_start + step,
            )
        elif direction == "end":
            self._time_start = max(0.0, self._duration - self._time_window)
        self._time_start = max(0.0, self._time_start)
        self._redraw()

    def _on_norm_toggled(self, checked) -> None:
        self._normalize = checked
        if self._raw is not None:
            self._redraw()

    def _on_dark_toggled(self, checked) -> None:
        self._dark_plot = checked
        self._apply_plot_theme()
        if self._raw is not None:
            self._redraw()

    def _on_mouse_moved(self, evt) -> None:
        pos = evt[0]
        if not self._plot_widget.sceneBoundingRect().contains(pos):
            return
        if not self._shown_ch_info:
            return
        view_pt = self._plot_widget.getPlotItem().vb.mapSceneToView(pos)
        y = view_pt.y()
        nearest = ""
        min_dist = float("inf")
        for y_off, ch_name, _ in self._shown_ch_info:
            d = abs(y - y_off)
            if d < min_dist:
                min_dist = d
                nearest = ch_name
        if min_dist < 0.6:
            QToolTip.showText(QCursor.pos(), nearest, self._plot_widget)
        else:
            QToolTip.hideText()

    def _on_events_toggled(self, checked) -> None:
        self._show_events = checked
        if checked and self._raw is not None:
            self._jump_to_first_event_if_needed()
        if self._raw is not None:
            self._redraw()

    def _on_event_src_changed(self, text) -> None:
        self._event_source = text
        if self._show_events:
            self._jump_to_first_event_if_needed()
            self._redraw()

    def _jump_to_first_event_if_needed(self) -> None:
        """If events are enabled but none fall in the current window, scroll
        to the first one. Events often start well into a recording (the MEG
        sample's first trigger is at ~102 s), so without this "Show events"
        looks like it does nothing at t=0.
        """
        events = self._active_events()
        if not events:
            return
        tmin = self._time_start
        tmax = min(self._time_start + self._time_window, self._duration)
        if any(tmin <= t <= tmax for t, _ in events):
            return
        first = min(t for t, _ in events)
        max_start = max(0.0, self._duration - self._time_window)
        self._time_start = max(0.0, min(first - self._time_window * 0.1, max_start))
        self.status_message.emit(
            f"{len(events)} events; jumped to the first at {first:.1f}s"
        )

    def _on_event_width_changed(self, val) -> None:
        self._event_line_width = val
        if self._show_events:
            self._redraw()

    def _pick_event_color(self) -> None:
        color = QColorDialog.getColor(
            self._event_color_override or QColor("#ffcc00"),
            self,
            "Pick event line colour",
        )
        if color.isValid():
            self._event_color_override = color
            if self._show_events:
                self._redraw()

    def _reset_event_color(self) -> None:
        self._event_color_override = None
        if self._show_events:
            self._redraw()

    # --------------------------------------------------------------- filters
    def _apply_filters(self) -> None:
        hp = self.spn_hp.value()
        lp = self.spn_lp.value()
        notch = self.spn_notch.value()
        self._current_filter = (hp if hp > 0 else None, lp if lp > 0 else None)
        self._notch_freq = notch if notch > 0 else None
        self._redraw()
        self.status_message.emit(
            f"Filter applied: HP={hp:g}Hz LP={lp:g}Hz Notch={notch:g}Hz"
        )

    def _reset_filters(self) -> None:
        for w in (self.spn_hp, self.spn_lp, self.spn_notch):
            w.blockSignals(True)
            w.setValue(0.0)
            w.blockSignals(False)
        self._current_filter = None
        self._notch_freq = None
        self._redraw()
        self.status_message.emit("Filters reset")

    def _apply_resample(self) -> None:
        freq = self.spn_resample.value()
        if freq <= 0 or self._raw is None:
            return
        from ...workers import RecordingResampleWorker

        self.loading_changed.emit(True, f"Resampling to {freq:.0f} Hz…")
        self.btn_resample.setEnabled(False)
        worker = RecordingResampleWorker(self._raw, freq, parent=self)
        worker.finished_with_raw.connect(self._on_resampled)
        worker.failed.connect(self._on_resample_failed)
        worker.finished.connect(worker.deleteLater)
        self._resample_worker = worker
        worker.start()

    def _on_resampled(self, raw, freq) -> None:
        self._resample_worker = None
        self.loading_changed.emit(False, "")
        self.btn_resample.setEnabled(True)
        self._raw = raw
        self._sfreq = float(raw.info["sfreq"])
        self._n_samples = int(raw.n_times)
        self._duration = float(raw.times[-1]) if raw.n_times else 0.0
        self.spn_window.setMaximum(max(0.1, self._duration))
        self._redraw()
        self.status_message.emit(
            f"Resampled to {freq:.0f} Hz ({self._n_samples} samples)"
        )

    def _on_resample_failed(self, msg) -> None:
        self._resample_worker = None
        self.loading_changed.emit(False, "")
        self.btn_resample.setEnabled(True)
        QMessageBox.warning(self, "Resample error", msg)

    # ------------------------------------------------------------------ PSD
    def _show_psd(self) -> None:
        if self._raw is None:
            return
        from ...workers import RecordingComputeWorker

        self.loading_changed.emit(True, "Computing PSD…")
        self.btn_psd.setEnabled(False)
        raw = self._raw
        sfreq = self._sfreq

        def _compute():
            import mne

            fmax = min(sfreq / 2.0, 150.0)
            psd = raw.compute_psd(fmin=0.1, fmax=fmax, verbose=False)
            data = np.asarray(psd.get_data())
            freqs = np.asarray(psd.freqs)
            # compute_psd returns only data channels, in its own order - the
            # rows align with psd.ch_names, NOT the raw channel list. Derive
            # types from the raw info by name so labels/types stay correct.
            names = list(psd.ch_names)
            raw_names = list(raw.ch_names)
            types = []
            for ch in names:
                try:
                    types.append(mne.channel_type(raw.info, raw_names.index(ch)))
                except ValueError:
                    types.append("misc")
            n = min(data.shape[0], len(names), len(types))
            return {
                "freqs": freqs,
                "data": data[:n],
                "ch_names": names[:n],
                "ch_types": types[:n],
            }

        worker = RecordingComputeWorker(_compute, parent=self)
        worker.finished_with_result.connect(self._on_psd_ready)
        worker.failed.connect(self._on_psd_failed)
        worker.finished.connect(worker.deleteLater)
        self._psd_worker = worker
        worker.start()

    def _on_psd_ready(self, result) -> None:
        self._psd_worker = None
        self.loading_changed.emit(False, "")
        self.btn_psd.setEnabled(True)
        dlg = _PsdDialog(result, parent=self)
        dlg.show()
        self.status_message.emit("PSD computed")

    def _on_psd_failed(self, msg) -> None:
        self._psd_worker = None
        self.loading_changed.emit(False, "")
        self.btn_psd.setEnabled(True)
        QMessageBox.warning(self, "PSD error", msg)

    # ----------------------------------------------------------- selection
    def _open_channel_selector(self) -> None:
        if not self._ch_names:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Select channels")
        dlg.resize(360, 520)
        lay = QVBoxLayout(dlg)

        search = QLineEdit()
        search.setPlaceholderText("Search channels…")
        lay.addWidget(search)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        cmb = QComboBox()
        display = []
        for ct in self._available_ch_types:
            display.append("mag (axial grad)" if self._is_ctf and ct == "mag" else ct)
        cmb.addItems(["all"] + display)
        type_row.addWidget(cmb, 1)
        lay.addLayout(type_row)

        lst = QListWidget()
        lst.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for i, name in enumerate(self._ch_names):
            ct = self._ch_types[i]
            disp = "mag (axial grad)" if self._is_ctf and ct == "mag" else ct
            item = QListWidgetItem(f"{name}  [{disp}]")
            item.setData(Qt.ItemDataRole.UserRole, name)
            if self._selected_channels is None or name in self._selected_channels:
                item.setSelected(True)
            lst.addItem(item)
        lay.addWidget(lst, 1)

        qrow = QHBoxLayout()
        b_all = QPushButton("Select all")
        b_none = QPushButton("Select none")
        b_all.clicked.connect(lst.selectAll)
        b_none.clicked.connect(lst.clearSelection)
        qrow.addWidget(b_all)
        qrow.addWidget(b_none)
        lay.addLayout(qrow)

        def _filter():
            text = search.text().lower()
            ct = cmb.currentText()
            if ct == "mag (axial grad)":
                ct = "mag"
            for idx in range(lst.count()):
                item = lst.item(idx)
                name = item.data(Qt.ItemDataRole.UserRole)
                ch_type = self._ch_types[self._ch_names.index(name)]
                visible = True
                if text and text not in name.lower():
                    visible = False
                if ct != "all" and ch_type != ct:
                    visible = False
                item.setHidden(not visible)

        search.textChanged.connect(_filter)
        cmb.currentTextChanged.connect(lambda _=None: _filter())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected = {
                lst.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(lst.count())
                if lst.item(i).isSelected()
            }
            if len(selected) == len(self._ch_names):
                self._selected_channels = None
            else:
                self._selected_channels = selected
            self._channel_offset = 0
            self._update_display_indices()
            self._redraw()
            n = len(selected) if self._selected_channels else len(self._ch_names)
            self.status_message.emit(f"Displaying {n} channels")

    # ----------------------------------------------------------------- reset
    def _reset_view(self) -> None:
        if self._raw is None:
            return
        self._time_start = 0.0
        self._channel_offset = 0
        self._scale_factor = 1.0
        self._time_window = min(10.0, max(0.1, self._duration))
        self._active_ch_type = "all"
        self._current_filter = None
        self._notch_freq = None
        self._selected_channels = None
        self._visible_channels = 20
        self._normalize = False
        for w in (self.spn_scale, self.spn_window, self.spn_n, self.cmb_ch_type,
                  self.spn_hp, self.spn_lp, self.spn_notch, self.chk_norm):
            w.blockSignals(True)
        self.spn_scale.setValue(1.0)
        self.spn_window.setValue(self._time_window)
        self.spn_n.setValue(20)
        self.cmb_ch_type.setCurrentText("all")
        self.spn_hp.setValue(0.0)
        self.spn_lp.setValue(0.0)
        self.spn_notch.setValue(0.0)
        self.chk_norm.setChecked(False)
        for w in (self.spn_scale, self.spn_window, self.spn_n, self.cmb_ch_type,
                  self.spn_hp, self.spn_lp, self.spn_notch, self.chk_norm):
            w.blockSignals(False)
        self._update_display_indices()
        self._update_channel_scrollbar()
        self._redraw()
        self.status_message.emit("View reset")

    # ---------------------------------------------------------------- unload
    def unload(self) -> None:
        for w in (self._resample_worker, self._psd_worker):
            if w is not None:
                w.cancel()
        self._resample_worker = None
        self._psd_worker = None
        self._plot_widget.clear()
        self._clear_labels()
        self._raw = None
        self._ch_names = []
        self._ch_types = []
        self._available_ch_types = []
        self._display_indices = []
        self._shown_ch_info = []
        self._overlay_items = []
        self._current_filter = None
        self._notch_freq = None
        self._selected_channels = None
        self._is_ctf = False
        self._duration = 0.0
        self._n_samples = 0
        self._tsv_events = []
        self._stim_events = []
        self.chk_events.setChecked(False)
        self.chk_events.setEnabled(False)
        self.lbl_time.setText("0.0 / 0.0 s")

    # ----------------------------------------------------------------- theme
    def _app_is_dark(self) -> bool:
        return self.palette().color(QPalette.ColorRole.Base).lightness() < 128

    def _apply_plot_theme(self) -> None:
        pg = self._pg
        dark_theme = self._app_is_dark()
        # In dark theme the canvas is already dark, so the "Dark plot" override
        # is redundant - disable + force it off.
        self.chk_dark.blockSignals(True)
        if dark_theme:
            self._dark_plot = False
            self.chk_dark.setChecked(False)
            self.chk_dark.setEnabled(False)
            self.chk_dark.setToolTip("Dark plot is automatic in dark theme")
        else:
            self.chk_dark.setEnabled(True)
            self.chk_dark.setToolTip(
                "Force a dark plot canvas (the canvas otherwise follows the "
                "app theme)."
            )
        self.chk_dark.blockSignals(False)

        if self._dark_plot and not dark_theme:
            bg = QColor("#11161d")
            x_fg = QColor("#cccccc")
        else:
            bg = self.palette().color(QPalette.ColorRole.Base)
            x_fg = self.palette().color(QPalette.ColorRole.Text)
        y_fg = self.palette().color(QPalette.ColorRole.Text)
        self._plot_widget.setBackground(bg)
        x_axis = self._plot_widget.getPlotItem().getAxis("bottom")
        x_axis.setPen(pg.mkPen(color=x_fg))
        x_axis.setTextPen(pg.mkPen(color=x_fg))
        y_axis = self._plot_widget.getPlotItem().getAxis("left")
        y_axis.setPen(pg.mkPen(color=y_fg))
        y_axis.setTextPen(pg.mkPen(color=y_fg))

    def repaint_for_palette(self, pal: dict) -> None:
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()
        for cmb in self.findChildren(QComboBox):
            view = cmb.view()
            targets = [cmb, view] + (
                [view.viewport()] if view and view.viewport() else []
            )
            for t in targets:
                style.unpolish(t)
                style.polish(t)
                t.update()
        for name, btn in (
            ("channels", self.btn_select),
            ("reset_view", self.btn_reset),
            ("close_data", self.btn_close),
            ("filter", self.btn_filter),
            ("resample", self.btn_resample),
            ("psd", self.btn_psd),
        ):
            icons.apply_button(btn, name)
        self._apply_plot_theme()
        if self._raw is not None:
            self._redraw()


# ===========================================================================
# The pane (metadata-first + time-series, with threaded loading)
# ===========================================================================
class RecordingViewerPane(QWidget):
    """Center pane for EEG/MEG/iEEG recordings.

    Bound to a single recording via :meth:`set_file`; pass ``None`` to
    clear and unload. Metadata loads on a worker thread first; a "Load
    signal" button loads the full recording (also threaded) and reveals
    the interactive time-series view. The view's Close returns here.
    """

    status_message = pyqtSignal(str)
    loading_changed = pyqtSignal(bool, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")
        self.setMinimumWidth(0)

        self._current_file: Optional[Path] = None
        self._current_root: Optional[Path] = None
        self._meta: Optional[dict] = None
        self._meta_worker = None
        self._signal_worker = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Recording"))

        self._stack = QStackedWidget()
        v.addWidget(self._stack, 1)

        self._hint = QLabel(
            "Select an EEG / MEG recording in the BIDS tree to view it."
        )
        self._hint.setObjectName("pane-hint")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)
        self._stack.addWidget(self._hint)

        self._loading_page = self._build_loading_page()
        self._stack.addWidget(self._loading_page)

        self._meta_page = self._build_meta_page()
        self._stack.addWidget(self._meta_page)

        # The pyqtgraph time-series view is built lazily, only when a signal
        # is actually loaded. Editor sessions that just browse JSON/TSV never
        # construct a pyqtgraph plot (cheaper + avoids offscreen-teardown
        # flakiness in the test suite), mirroring the NIfTI viewer.
        self._view: Optional[_TimeSeriesView] = None

        self._stack.setCurrentWidget(self._hint)

    def _ensure_view(self) -> "_TimeSeriesView":
        if self._view is None:
            self._view = _TimeSeriesView()
            self._view.status_message.connect(self.status_message)
            self._view.loading_changed.connect(self.loading_changed)
            self._view.close_requested.connect(self._on_view_close)
            self._stack.addWidget(self._view)
        return self._view

    # ------------------------------------------------------------------ UI
    def _build_loading_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("pane-dark")
        lay = QVBoxLayout(page)
        lay.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self._spinner = BusySpinner()
        row.addWidget(self._spinner)
        row.addStretch(1)
        lay.addLayout(row)
        self._loading_label = QLabel("")
        self._loading_label.setObjectName("pane-hint")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._loading_label)
        lay.addStretch(1)
        return page

    def _build_meta_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("pane-dark")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("viewer-meta-scroll")
        self._meta_body = QWidget()
        self._meta_body.setObjectName("pane-dark")
        self._meta_layout = QVBoxLayout(self._meta_body)
        self._meta_layout.setContentsMargins(18, 16, 18, 16)
        self._meta_layout.setSpacing(6)
        self._meta_layout.addStretch(1)
        scroll.setWidget(self._meta_body)
        outer.addWidget(scroll, 1)

        bar = QFrame()
        bar.setObjectName("toolbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 8, 14, 8)
        bl.setSpacing(8)
        bl.addStretch(1)
        self._load_btn = QPushButton("Load signal")
        self._load_btn.setObjectName("load-signal-btn")
        self._load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._load_btn.clicked.connect(self._load_signal)
        bl.addWidget(self._load_btn)
        bl.addStretch(1)
        outer.addWidget(bar)
        return page

    def _meta_section(self, title: str) -> None:
        lbl = QLabel(title)
        lbl.setObjectName("viewer-meta-section")
        self._meta_layout.insertWidget(self._meta_layout.count() - 1, lbl)

    def _meta_row(self, label: str, value: str) -> None:
        row = QWidget()
        row.setObjectName("viewer-meta-row")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)
        k = QLabel(label)
        k.setObjectName("viewer-meta-key")
        k.setMinimumWidth(96)
        k.setMaximumWidth(140)
        val = QLabel(value)
        val.setObjectName("viewer-meta-val")
        val.setWordWrap(True)
        val.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        rl.addWidget(k)
        rl.addWidget(val, 1)
        self._meta_layout.insertWidget(self._meta_layout.count() - 1, row)

    def _clear_meta(self) -> None:
        while self._meta_layout.count() > 1:
            item = self._meta_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _populate_meta(self, meta: dict) -> None:
        self._clear_meta()

        title = QLabel(meta.get("name", ""))
        title.setObjectName("viewer-meta-title")
        title.setWordWrap(True)
        self._meta_layout.insertWidget(self._meta_layout.count() - 1, title)

        self._meta_section("Recording")
        self._meta_row("Channels", str(meta.get("n_channels", 0)))
        self._meta_row("Sampling rate", f"{meta.get('sfreq', 0):.1f} Hz")
        dur = meta.get("duration", 0.0)
        self._meta_row("Duration", f"{dur:.1f} s ({dur / 60:.1f} min)")
        self._meta_row("Time points", str(meta.get("n_times", 0)))
        if meta.get("meas_date"):
            self._meta_row("Measurement date", str(meta["meas_date"]))

        self._meta_section("Filters")
        hp = meta.get("highpass")
        lp = meta.get("lowpass")
        lf = meta.get("line_freq")
        self._meta_row("High-pass", f"{hp:g} Hz" if hp not in (None, "") else "n/a")
        self._meta_row("Low-pass", f"{lp:g} Hz" if lp not in (None, "") else "n/a")
        self._meta_row("Line frequency", f"{lf:g} Hz" if lf not in (None, "") else "n/a")

        self._meta_section("Channel types")
        counts = meta.get("ch_type_counts", {})
        for ct, n in sorted(counts.items()):
            label = "mag (axial grad)" if meta.get("is_ctf") and ct == "mag" else ct
            self._meta_row(label, str(n))
        if meta.get("is_ctf"):
            note = QLabel(
                "CTF dataset: MEG sensors are axial gradiometers "
                "(MNE labels them “mag”)."
            )
            note.setObjectName("viewer-meta-note")
            note.setWordWrap(True)
            self._meta_layout.insertWidget(self._meta_layout.count() - 1, note)

        bads = meta.get("bads") or []
        if bads:
            self._meta_section(f"Bad channels ({len(bads)})")
            self._meta_row("", ", ".join(bads))

    # -------------------------------------------------------------- public
    def current_file(self) -> Optional[Path]:
        return self._current_file

    def set_file(self, path: Optional[Path], root: Optional[Path]) -> None:
        """Bind to a recording (or ``None`` to clear + unload)."""
        self._cancel_workers()
        self._current_file = path
        self._current_root = root
        if path is None:
            if self._view is not None:
                self._view.unload()
            self._spinner.set_busy(False)
            self._stack.setCurrentWidget(self._hint)
            self._hint.setText(
                "Select an EEG / MEG recording in the BIDS tree to view it."
            )
            self.loading_changed.emit(False, "")
            return

        from ...workers import RecordingMetaWorker

        self._show_loading(f"Loading metadata: {path.name}…")
        worker = RecordingMetaWorker(path, parent=self)
        worker.finished_with_meta.connect(self._on_meta)
        worker.failed.connect(self._on_meta_failed)
        worker.finished.connect(worker.deleteLater)
        self._meta_worker = worker
        worker.start()

    def _show_loading(self, message: str) -> None:
        self._loading_label.setText(message)
        self._spinner.set_busy(True, message="")
        self._stack.setCurrentWidget(self._loading_page)
        self.loading_changed.emit(True, message)

    def _cancel_workers(self) -> None:
        for w in (self._meta_worker, self._signal_worker):
            if w is not None:
                w.cancel()
        self._meta_worker = None
        self._signal_worker = None

    # ----------------------------------------------------------- callbacks
    def _on_meta(self, meta: dict, path: Path) -> None:
        if path != self._current_file:
            return
        self._meta_worker = None
        self._meta = meta
        self._spinner.set_busy(False)
        self.loading_changed.emit(False, "")
        self._populate_meta(meta)
        self._stack.setCurrentWidget(self._meta_page)
        self.status_message.emit(
            f"{meta.get('name', '')}: {meta.get('n_channels', 0)} ch, "
            f"{meta.get('sfreq', 0):.0f} Hz, {meta.get('duration', 0):.1f}s"
        )

    def _on_meta_failed(self, path: Path, error: str) -> None:
        if path != self._current_file:
            return
        self._meta_worker = None
        self._spinner.set_busy(False)
        self.loading_changed.emit(False, "")
        self._stack.setCurrentWidget(self._hint)
        self._hint.setText(f"Could not read {path.name}:\n{error}")
        self.status_message.emit(f"Error: {error.splitlines()[0] if error else ''}")

    def _load_signal(self) -> None:
        if self._current_file is None:
            return
        from ...workers import RecordingSignalWorker

        path = self._current_file
        self._show_loading(f"Loading signal: {path.name}…")
        worker = RecordingSignalWorker(path, parent=self)
        worker.finished_with_raw.connect(self._on_raw)
        worker.failed.connect(self._on_raw_failed)
        worker.finished.connect(worker.deleteLater)
        self._signal_worker = worker
        worker.start()

    def _on_raw(self, raw, path: Path) -> None:
        if path != self._current_file:
            return
        self._signal_worker = None
        self._spinner.set_busy(False)
        self.loading_changed.emit(False, "")
        view = self._ensure_view()
        view.set_current_filepath(path, self._current_root)
        view.load_raw(raw)
        self._stack.setCurrentWidget(view)

    def _on_raw_failed(self, path: Path, error: str) -> None:
        if path != self._current_file:
            return
        self._signal_worker = None
        self._spinner.set_busy(False)
        self.loading_changed.emit(False, "")
        self._stack.setCurrentWidget(self._meta_page)
        QMessageBox.warning(
            self, "Load error", f"Could not load {path.name}:\n{error}"
        )

    def _on_view_close(self) -> None:
        """Close button in the viewer: drop the signal, show metadata."""
        if self._view is not None:
            self._view.unload()
        if self._meta is not None:
            self._stack.setCurrentWidget(self._meta_page)
        else:
            self._stack.setCurrentWidget(self._hint)
        self.status_message.emit("Signal closed")

    # -------------------------------------------------------------- theme
    def repaint_for_palette(self, pal: dict) -> None:
        style = self.style()
        for w in [self, *self._meta_page.findChildren(QWidget),
                  self._loading_page, self._hint]:
            style.unpolish(w)
            style.polish(w)
            w.update()
        if self._view is not None:
            self._view.repaint_for_palette(pal)


__all__ = ["RecordingViewerPane", "is_recording_path"]
