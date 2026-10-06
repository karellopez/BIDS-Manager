"""Signals (MEG, EEG, iEEG, physio) in the viewer shell.

**MEG and EEG open on their metadata.** Channels, rate, duration, types,
filters, bad channels, read with ``preload=False`` on a worker; "Load signal"
then reads the recording and shows the traces, and Close drops it again.
A recording can be gigabytes, and most visits only need the card.

**Physio opens straight into the traces**, decided by its sidecar, with Fit
all (a channel or four fits whole, which three hundred MEG channels do not)
and, when the run has several physio files, "All of this run" on one clock.

Every control runs a command (``viz.commands.signal``), so a key, a button,
the palette and a linked viewer all do the same thing. Reading, resampling
and the spectrum run as jobs (QThread, never a pool: they end in scipy).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from ....viz.bids import full_ext, suffix_of
from ....viz.commands import signal as sigcmd
from ....viz.compute.filters import PRESETS, preset_of
from ....viz.data import formats
from ....viz.data.signal import SignalSource
from ....viz.scene import Scene, SignalLayer, SourceRef, TracesState
from ...widgets.primitives import ElidedLabel
from ..context import ViewerContext
from ..controls import NumberControl
from ..menus import popup_menu

log = logging.getLogger(__name__)

#: Decimals on every frequency field: three, because 0.01 Hz is where a
#: respiratory drift lives ("impossible to put a filter of 0.01").
_DECIMALS = 3

#: Three rows, by purpose, each under a thousand pixels so a narrow pane
#: wraps a row rather than scattering it: WHICH channels and how they are
#: drawn; the SCALE and the stretch of TIME, with the events; and what is
#: DONE to the signal (filters, resampling, the spectrum).
TOOLBAR_ROWS = (
    ("widget:channels", "|", "traces.butterfly", "traces.normalize", "traces.page_scale",
     "traces.clip", "traces.dc", "|", "channels.write_bads", "stretch",
     "view.zen", "signal.close"),
    ("widget:scale", "|", "time.fit", "traces.together", "|", "events.toggle",
     "widget:events", "|", "traces.reset", "stretch", "help.shortcuts"),
    ("widget:filters", "|", "widget:resample", "|", "widget:psd", "traces.line"),
)

#: The width of a slider in the toolbar: enough to aim, not so much that
#: three of them push the toggles onto another line.
_TOOLBAR_SLIDER_PX = 84


def _read_meta(path) -> dict:
    """Worker side: the metadata card, never the samples."""
    from ....viz.data.signal import read_recording, summarize

    return summarize(read_recording(path, preload=False), path)


def _open(path, root, kind: str, together: bool) -> SignalSource:
    from ....viz.data.signal import open_meeg, open_physio

    if kind == "physio":
        return open_physio(path, root, together=together)
    return open_meeg(path, root)


class SignalPresenter:
    """Signal content for a :class:`~bidsmgr.gui.viz.viewer.Viewer`."""

    kind = "signal"
    title = "Recording"
    empty_hint = "Select an EEG or MEG recording, or a physio file, in the BIDS tree."

    def __init__(self, viewer, ctx: ViewerContext) -> None:
        self.viewer = viewer
        self.ctx = ctx
        self.source: Optional[SignalSource] = None
        self.meta: Optional[dict] = None
        self._generation = 0
        self._path: Optional[Path] = None
        self._root: Optional[Path] = None
        self._file_kind = "meeg"
        self.together = False
        #: Zen mode: the traces alone, no toolbar and no overview.
        self.zen = False
        self._relatives: list[Path] = []
        self._traces = None
        self.content = self._build_content()
        ctx.jobs.done.connect(self._on_job_done)
        ctx.jobs.failed.connect(self._on_job_failed)
        ctx.qstore.changed.connect(self._on_changed)
        ctx.status.connect(viewer.status_message)

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def _build_content(self) -> QWidget:
        self.pages = QStackedWidget()
        self.pages.setObjectName("pane-dark")
        self.meta_page = self._build_meta_page()
        self.pages.addWidget(self.meta_page)
        self.traces_page = QWidget()
        self.traces_page.setObjectName("pane-dark")
        v = QVBoxLayout(self.traces_page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._traces_slot = v
        self.navigation = self._build_navigation()
        v.addWidget(self.navigation)
        self.pages.addWidget(self.traces_page)
        return self.pages

    def _ensure_traces(self):
        """The pyqtgraph canvas is built when a signal first arrives: an
        Editor session that only browses tables never builds a plot."""
        if self._traces is None:
            from ..canvases.traces import TracesCanvas

            self._traces = TracesCanvas(self.ctx)
            self._traces_slot.insertWidget(0, self._traces, 1)
        return self._traces

    @property
    def traces(self):
        return self._traces

    def _build_meta_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("pane-dark")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("viewer-meta-scroll")
        body = QWidget()
        body.setObjectName("pane-dark")
        self._meta_layout = QVBoxLayout(body)
        self._meta_layout.setContentsMargins(18, 16, 18, 16)
        self._meta_layout.setSpacing(6)
        self._meta_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        bar = QFrame()
        bar.setObjectName("toolbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 8, 14, 8)
        bl.addStretch(1)
        self.load_button = QPushButton("Load signal")
        self.load_button.setObjectName("load-signal-btn")
        self.load_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_button.clicked.connect(self.load_signal)
        bl.addWidget(self.load_button)
        bl.addStretch(1)
        outer.addWidget(bar)
        return page

    def _build_navigation(self) -> QWidget:
        """|<  <  [the whole recording]  >  >|  and where the window is."""
        from ..canvases.overview import OverviewBar

        nav = QFrame()
        nav.setObjectName("toolbar")
        lay = QHBoxLayout(nav)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(4)
        manager = self.viewer.action_manager
        lay.addWidget(manager.button("time.start"))
        prev = QPushButton("<")
        prev.setObjectName("tb-btn")
        prev.setToolTip(manager.actions["time.prev"].toolTip())
        prev.clicked.connect(lambda: self.viewer.trigger("time.prev"))
        lay.addWidget(prev)
        self.overview = OverviewBar(self.ctx)
        self.overview.setToolTip("The whole recording: activity, events, bad spans and "
                                 "gaps. Click to go there, drag the window, wheel to page.")
        lay.addWidget(self.overview, 1)
        nxt = QPushButton(">")
        nxt.setObjectName("tb-btn")
        nxt.setToolTip(manager.actions["time.next"].toolTip())
        nxt.clicked.connect(lambda: self.viewer.trigger("time.next"))
        lay.addWidget(nxt)
        end = manager.button("time.end")
        lay.addWidget(end)
        # Sized to their text: a push button's default floor is 80 px, and
        # four of them made this row a 540 px floor under the whole pane.
        for btn in (nav.findChildren(QPushButton)):
            fm = btn.fontMetrics()
            btn.setFixedWidth(max(28, fm.horizontalAdvance(btn.text()) + 22))
        self.time_label = ElidedLabel("")
        self.time_label.setObjectName("sidecar-footer-summary")
        lay.addWidget(self.time_label)
        return nav

    # ------------------------------------------------------------------
    # Toolbar widgets
    # ------------------------------------------------------------------

    def toolbar_rows(self):
        return TOOLBAR_ROWS

    def make_widget(self, name: str):
        return {
            "channels": self._widget_channels, "filters": self._widget_filters,
            "resample": self._widget_resample, "psd": self._widget_psd,
            "events": self._widget_events, "scale": self._widget_scale,
        }.get(name, lambda: None)()

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sidecar-footer-summary")
        return lbl

    def _group(self, *widgets) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        for w in widgets:
            h.addWidget(w)
        return box

    @staticmethod
    def _fit(spin: QDoubleSpinBox, widest: str) -> None:
        """As wide as its widest text in ITS font, whatever the platform,
        DPI or app font scale: a fixed pixel width cut "3600.0 s" off on a
        larger font."""
        spin.ensurePolished()
        spin.setFixedWidth(spin.fontMetrics().horizontalAdvance(widest) + 22)

    def _number(self, lo: float, hi: float, *, step: float, unit: str = "",
                default: Optional[float] = None, log: bool = False,
                decimals: Optional[int] = None, tip: str = "",
                widest: str = "0000.00") -> NumberControl:
        ctl = NumberControl(lo, hi, step=step, unit=unit, default=default, log=log,
                            decimals=decimals)
        ctl.slider.setMinimumWidth(_TOOLBAR_SLIDER_PX)
        ctl.slider.setFixedWidth(_TOOLBAR_SLIDER_PX)
        self._fit(ctl.spin, widest)
        if tip:
            ctl.setToolTip(tip + (f" Double-click the slider for {default:g}."
                                  if default is not None else ""))
        return ctl

    def _widget_channels(self) -> list[QWidget]:
        self.type_combo = QComboBox()
        self.type_combo.setToolTip("Which channel type to show")
        self.type_combo.currentIndexChanged.connect(
            lambda _i: self._run_ui("traces.type", ch_type=self.type_combo.currentData()))
        self.channels_button = QPushButton("Channels...")
        self.channels_button.setObjectName("tb-btn")
        self.channels_button.setToolTip("Pick the channels to show")
        self.channels_button.clicked.connect(self.open_channel_picker)
        self.count_control = self._number(
            1, 500, step=1, decimals=0, default=sigcmd.DEFAULT_COUNT, widest="500",
            tip="Traces on screen at once (PgUp and PgDown page through the rest).")
        self.count_control.value_changed.connect(
            lambda v: self._run_ui("traces.count", n=int(round(v))))
        self._multi_channel_widgets = [
            self._group(self._label("Type"), self.type_combo), self.channels_button,
            self._group(self._label("Count"), self.count_control),
        ]
        return list(self._multi_channel_widgets)

    def _widget_scale(self) -> list[QWidget]:
        # Logarithmic, every step the same ratio, so the slider is as fine at
        # 0.1 as at 10.
        self.scale_control = self._number(
            0.01, 1000.0, step=0.1, decimals=2, default=1.0, log=True,
            tip="Amplitude, as a factor ([ and ] change it from the keyboard).")
        self.scale_control.value_changed.connect(
            lambda v: self._run_ui("traces.scale", value=float(v)))
        self.width_control = self._number(
            0.1, 3600.0, step=0.5, unit="s", decimals=1, default=sigcmd.DEFAULT_WIDTH_S,
            log=True, widest="3600.0 s", tip="Seconds on screen (= and - change it).")
        self.width_control.value_changed.connect(
            lambda v: self._run_ui("time.width", seconds=float(v)))
        return [self._group(self._label("Amplitude"), self.scale_control),
                self._group(self._label("Window"), self.width_control)]

    def _widget_events(self) -> QWidget:
        self.event_source = QComboBox()
        self.event_source.setToolTip("Where the events come from: the run's events.tsv, "
                                     "the trigger channel, or the recording's annotations")
        self.event_source.currentIndexChanged.connect(
            lambda _i: self._run_ui("events.source", which=self.event_source.currentData()))
        return self.event_source

    def _widget_filters(self) -> list[QWidget]:
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("A common band, applied at once (the notch is kept)")
        self.preset_combo.activated.connect(self._on_preset)
        self.hp_spin, self.lp_spin, self.notch_spin = (QDoubleSpinBox() for _ in range(3))
        for spin, hi, tip in ((self.hp_spin, 500.0, "High-pass (Hz)"),
                              (self.lp_spin, 5000.0, "Low-pass (Hz)"),
                              (self.notch_spin, 1000.0, "Notch (Hz)")):
            spin.setRange(0.0, hi)
            spin.setDecimals(_DECIMALS)
            spin.setSingleStep(0.1)
            spin.setSpecialValueText("Off")
            self._compact(spin)
            spin.setToolTip(f"{tip}. Zero-phase. A cut-off at or above the Nyquist "
                            "frequency is refused, with the reason. Enter applies.")
            spin.lineEdit().returnPressed.connect(self.apply_filters)
        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("tb-btn")
        apply_btn.setToolTip("Apply the three cut-offs")
        apply_btn.clicked.connect(self.apply_filters)
        reset = self.viewer.action_manager.button("traces.reset_filters")
        return [self._group(self._label("Filter"), self.preset_combo),
                self._group(self._label("HP"), self.hp_spin),
                self._group(self._label("LP"), self.lp_spin),
                self._group(self._label("Notch"), self.notch_spin), apply_btn, reset]

    def _compact(self, spin: QDoubleSpinBox, widest: str = "5000.000") -> None:
        """A number field like the sliders' own: no arrows, a decimal point
        whatever the system locale, as wide as its numbers."""
        spin.setObjectName("viz-number")
        spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        spin.setLocale(self._c_locale())
        self._fit(spin, widest)

    @staticmethod
    def _c_locale():
        # A decimal point whatever the system says: "0,010" reads as ten to
        # an English reader (the same reason every number field here is C).
        from PyQt6.QtCore import QLocale

        loc = QLocale(QLocale.Language.C)
        loc.setNumberOptions(QLocale.NumberOption.OmitGroupSeparator)
        return loc

    def _presets(self) -> tuple:
        return PRESETS.get("physio" if self._file_kind == "physio" else "meeg", ())

    def _on_preset(self, index: int) -> None:
        data = self.preset_combo.itemData(index)
        if data is None or data == "custom":
            return
        hp, lp = data
        tr = self.ctx.scene.traces
        try:
            self.ctx.run("traces.filter", hp=hp, lp=lp, notch=tr.notch)
        except ValueError as exc:
            self.viewer.status_message.emit(str(exc))
            QMessageBox.information(self.viewer, "Filter not applied", str(exc))
            self.sync_widgets()
            return
        from ....viz.compute.filters import FilterSpec

        tr = self.ctx.scene.traces
        self.viewer.status_message.emit(
            f"Filter: {FilterSpec(tr.hp, tr.lp, tr.notch).describe()}")

    def _widget_resample(self) -> list[QWidget]:
        self.resample_spin = QDoubleSpinBox()
        self.resample_spin.setRange(0.0, 10000.0)
        self.resample_spin.setDecimals(0)
        self.resample_spin.setSpecialValueText("Off")
        self.resample_spin.setSuffix(" Hz")
        self.resample_spin.setToolTip("A new sampling rate: the recording is resampled "
                                      "on a worker, and the view kept where it is")
        self._compact(self.resample_spin, "10000 Hz")
        self.resample_button = QPushButton("Resample")
        self.resample_button.setObjectName("tb-btn")
        self.resample_button.clicked.connect(self.resample)
        return [self._group(self._label("Rate"), self.resample_spin), self.resample_button]

    def _widget_psd(self) -> QWidget:
        btn = QPushButton("PSD")
        btn.setObjectName("tb-btn")
        btn.setToolTip("Welch power spectrum of the channels shown")
        menu = popup_menu(btn)
        self._psd_raw = menu.addAction("Spectrum of the raw signal")
        self._psd_raw.triggered.connect(lambda: self.show_psd(filtered=False))
        self._psd_filtered = menu.addAction("Spectrum of the filtered signal")
        self._psd_filtered.triggered.connect(lambda: self.show_psd(filtered=True))
        menu.aboutToShow.connect(
            lambda: self._psd_filtered.setEnabled(self._filter_active()))
        btn.setMenu(menu)
        self.psd_button = btn
        return btn

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: Path, root: Optional[Path]) -> None:
        self._generation += 1
        self._release()
        self._path = Path(path)
        self._root = root
        self._file_kind = "meeg" if formats.is_recording_path(path) else "physio"
        self._relatives = []
        if self._file_kind == "physio":
            from ....viz.data.physio import related_recordings

            try:
                self._relatives = related_recordings(self._path)
            except OSError:
                self._relatives = []
            self.load_signal()
        else:
            self.ctx.jobs.start("meta", self._generation, _read_meta, self._path)

    def load_signal(self) -> None:
        if self._path is None:
            return
        self.viewer.show_loading(f"Loading signal: {self._path.name}")
        self.ctx.jobs.start("signal", self._generation, _open, self._path, self._root,
                            self._file_kind, self.together and len(self._relatives) > 1)

    def _release(self) -> None:
        for tag in ("meta", "signal", "resample", "psd", "overview"):
            self.ctx.jobs.cancel(tag)
        self.source = None
        self.meta = None
        store = self.ctx.store
        store.sources.clear()

    def clear(self) -> None:
        self._generation += 1
        self._release()
        self._path = None
        self.ctx.store.replace_scene(Scene())
        self.ctx.qstore.flush()

    def stop(self) -> None:
        pass

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if generation != self._generation:
            return
        if tag == "meta":
            self.meta = result
            self._fill_meta(result)
            self.pages.setCurrentWidget(self.meta_page)
            self.viewer.on_first_frame()
            self.viewer.on_loaded(self._path)
            self.viewer.status_message.emit(
                f"{result.get('name', '')}: {result.get('n_channels', 0)} ch, "
                f"{result.get('sfreq', 0):.0f} Hz, {result.get('duration', 0):.1f} s")
        elif tag == "signal":
            self._adopt(result)
        elif tag == "resample":
            self._adopt(result, keep_view=True)
            self.viewer.loading_changed.emit(False, "")
            self.resample_button.setEnabled(True)
            self.viewer.status_message.emit(
                f"Resampled to {result.sfreq:.0f} Hz ({result.n_times:,} samples)")
        elif tag == "psd":
            self.viewer.loading_changed.emit(False, "")
            self.psd_button.setEnabled(True)
            from ..canvases.psd import PsdWindow

            win = PsdWindow(result, parent=self.viewer)
            win.show()
            self.psd_window = win
            for message in result.get("messages", []):
                self.viewer.status_message.emit(message)

    def _on_job_failed(self, tag: str, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        if tag in ("meta",):
            self.viewer.on_load_failed(self._path, message)
        elif tag == "signal":
            if self._file_kind == "meeg" and self.meta is not None:
                self.pages.setCurrentWidget(self.meta_page)
                self.viewer.show_content()
                QMessageBox.warning(self.viewer, "Load error",
                                    f"Could not load {self._path.name}:\n{message}")
            else:
                self.viewer.on_load_failed(self._path, message)
        elif tag in ("resample", "psd"):
            self.viewer.loading_changed.emit(False, "")
            for btn in (getattr(self, "resample_button", None), getattr(self, "psd_button", None)):
                if btn is not None:
                    btn.setEnabled(True)
            QMessageBox.warning(self.viewer, "Signal", message)

    def _adopt(self, src: SignalSource, *, keep_view: bool = False) -> None:
        """Put a read recording on screen. Everything measured in its own
        terms resets; standing preferences (dark plot, event colour) stay."""
        self.source = src
        store = self.ctx.store
        sid = "sig0"
        store.sources = {sid: src}
        if keep_view:
            scene = store.scene
            scene.traces.t0 = min(scene.traces.t0, max(0.0, src.duration - scene.traces.width))
            store.changed({"sources:sig0", "traces.time"})
        else:
            scene = Scene()
            scene.sources = {sid: SourceRef(id=sid, path=str(src.path), kind="signal")}
            scene.layers = [SignalLayer(id="sig", source=sid, name=src.path.name)]
            scene.traces = TracesState(**sigcmd.opening_state(src))
            store.replace_scene(scene)
        self._ensure_traces()
        self.pages.setCurrentWidget(self.traces_page)
        self.viewer.on_first_frame()
        self.viewer.on_loaded(src.path)
        if src.note:
            self.viewer.status_message.emit(src.note)
        if not keep_view:
            types = ", ".join(src.type_label(t) for t in src.available_types)
            self.viewer.status_message.emit(
                f"Loaded {len(src.ch_names)} channels, {src.duration:.1f} s at "
                f"{src.sfreq:.0f} Hz | Types: {types}")
        self.ctx.qstore.flush()

    # ------------------------------------------------------------------
    # The metadata card
    # ------------------------------------------------------------------

    def _meta_clear(self) -> None:
        while self._meta_layout.count() > 1:
            item = self._meta_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _meta_add(self, widget: QWidget) -> None:
        self._meta_layout.insertWidget(self._meta_layout.count() - 1, widget)

    def _meta_section(self, title: str) -> None:
        lbl = QLabel(title)
        lbl.setObjectName("viewer-meta-section")
        self._meta_add(lbl)

    def _meta_row(self, key: str, value: str) -> None:
        row = QWidget()
        row.setObjectName("viewer-meta-row")
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        k = QLabel(key)
        k.setObjectName("viewer-meta-key")
        k.setMinimumWidth(96)
        k.setMaximumWidth(140)
        v = QLabel(value)
        v.setObjectName("viewer-meta-val")
        v.setWordWrap(True)
        v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        h.addWidget(k)
        h.addWidget(v, 1)
        self._meta_add(row)

    def _fill_meta(self, meta: dict) -> None:
        self._meta_clear()
        title = QLabel(meta.get("name", ""))
        title.setObjectName("viewer-meta-title")
        title.setWordWrap(True)
        self._meta_add(title)
        self._meta_section("Recording")
        self._meta_row("Channels", str(meta.get("n_channels", 0)))
        self._meta_row("Sampling rate", f"{meta.get('sfreq', 0):.1f} Hz")
        dur = meta.get("duration", 0.0)
        self._meta_row("Duration", f"{dur:.1f} s ({dur / 60:.1f} min)")
        self._meta_row("Time points", str(meta.get("n_times", 0)))
        if meta.get("meas_date"):
            self._meta_row("Measurement date", str(meta["meas_date"]))
        self._meta_section("Filters")
        for key, label in (("highpass", "High-pass"), ("lowpass", "Low-pass"),
                           ("line_freq", "Line frequency")):
            value = meta.get(key)
            self._meta_row(label, f"{value:g} Hz" if value not in (None, "") else "n/a")
        self._meta_section("Channel types")
        for ct, n in sorted(meta.get("ch_type_counts", {}).items()):
            label = "mag (axial grad)" if meta.get("is_ctf") and ct == "mag" else ct
            self._meta_row(label, str(n))
        if meta.get("is_ctf"):
            note = QLabel("CTF dataset: MEG sensors are axial gradiometers "
                          "(MNE labels them “mag”).")
            note.setObjectName("viewer-meta-note")
            note.setWordWrap(True)
            self._meta_add(note)
        bads = meta.get("bads") or []
        if bads:
            self._meta_section(f"Bad channels ({len(bads)})")
            self._meta_row("", ", ".join(bads))

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    _syncing = False

    def _run_ui(self, command_id: str, **params) -> None:
        if self._syncing:
            return
        try:
            self.ctx.run(command_id, **params)
        except ValueError as exc:
            self.viewer.status_message.emit(str(exc))

    def _filter_active(self) -> bool:
        tr = self.ctx.scene.traces
        return bool(tr.hp or tr.lp or tr.notch)

    def apply_filters(self) -> None:
        """Apply the three fields at once; a refused cut-off is said, and
        nothing changes."""
        try:
            changed = self.ctx.run("traces.filter", hp=self.hp_spin.value() or None,
                                   lp=self.lp_spin.value() or None,
                                   notch=self.notch_spin.value() or None)
        except ValueError as exc:
            self.viewer.status_message.emit(str(exc))
            QMessageBox.information(self.viewer, "Filter not applied", str(exc))
            return
        from ....viz.compute.filters import FilterSpec

        tr = self.ctx.scene.traces
        if changed:
            self.viewer.status_message.emit(
                f"Filter: {FilterSpec(tr.hp, tr.lp, tr.notch).describe()}")

    def resample(self) -> None:
        freq = self.resample_spin.value()
        if freq <= 0 or self.source is None:
            return
        from ....viz.data.signal import resampled

        self.viewer.loading_changed.emit(True, f"Resampling to {freq:.0f} Hz")
        self.resample_button.setEnabled(False)
        self.ctx.jobs.start("resample", self._generation, resampled, self.source, freq)

    def show_psd(self, *, filtered: bool = False) -> None:
        """The spectrum of the channels SHOWN, raw or filtered by choice."""
        src = self.source
        if src is None:
            return
        from ....viz.compute.filters import FilterSpec
        from ....viz.compute.spectral import psd

        tr = self.ctx.scene.traces
        indices = src.picks_for(tr.ch_type, tr.picks) or list(range(len(src.ch_names)))
        spec = FilterSpec(tr.hp, tr.lp, tr.notch) if filtered else None
        self.viewer.loading_changed.emit(True, "Computing the spectrum")
        self.psd_button.setEnabled(False)
        self.ctx.jobs.start("psd", self._generation, psd, src, indices, spec=spec)

    def open_channel_picker(self) -> None:
        src = self.source
        if src is None:
            return
        dlg = ChannelPicker(src, self.ctx.scene.traces.picks, parent=self.viewer)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._run_ui("traces.pick", names=dlg.selected())
            n = len(src.picks_for("all", self.ctx.scene.traces.picks))
            self.viewer.status_message.emit(f"Showing {n} channels")

    # ------------------------------------------------------------------
    # Scene -> widgets
    # ------------------------------------------------------------------

    def _on_changed(self, paths) -> None:
        self.sync_widgets()
        self.viewer.update_footer()

    def sync_widgets(self) -> None:
        src = self.source
        tr = self.ctx.scene.traces
        self._syncing = True
        try:
            if not hasattr(self, "type_combo"):
                return
            if src is not None:
                wanted = [("all", "all")]
                if "mag" in src.available_types and "grad" in src.available_types:
                    wanted.append(("mag+grad", "mag+grad"))
                wanted += [(t, src.type_label(t)) for t in src.available_types]
                have = [(self.type_combo.itemData(i), self.type_combo.itemText(i))
                        for i in range(self.type_combo.count())]
                if have != wanted:
                    self.type_combo.clear()
                    for data, text in wanted:
                        self.type_combo.addItem(text, data)
                i = self.type_combo.findData(tr.ch_type)
                self.type_combo.setCurrentIndex(max(0, i))
                multi = len(src.ch_names) > 1
                # A type filter, a picker and a count can only say the same
                # thing about one channel: noise in a full toolbar.
                for w in self._multi_channel_widgets:
                    w.setVisible(multi)
                pool = len(src.picks_for(tr.ch_type, tr.picks)) or 1
                self.count_control.set_range(1, max(1, min(500, pool)))
                self.width_control.set_range(0.1, max(0.2, src.duration))
            self.count_control.set_value(tr.count)
            self.scale_control.set_value(tr.scale)
            self.width_control.set_value(tr.width)
            for spin, value in ((self.hp_spin, tr.hp), (self.lp_spin, tr.lp),
                                (self.notch_spin, tr.notch)):
                spin.setValue(value or 0.0)
            self._sync_presets(tr)
            if src is not None:
                sources = [("auto", "Events: auto")] + [(x, x) for x in src.event_sources()]
                have = [self.event_source.itemData(i) for i in range(self.event_source.count())]
                if have != [d for d, _t in sources]:
                    self.event_source.clear()
                    for data, text in sources:
                        self.event_source.addItem(text, data)
                self.event_source.setCurrentIndex(
                    max(0, self.event_source.findData(tr.event_source)))
                self.event_source.setVisible(bool(src.event_sources()))
                t1 = min(tr.t0 + tr.width, src.duration)
                self.time_label.setText(f"{tr.t0 + src.start_time:.1f} - "
                                        f"{t1 + src.start_time:.1f} / "
                                        f"{src.duration:.1f} s")
        finally:
            self._syncing = False

    def _sync_presets(self, tr) -> None:
        presets = self._presets()
        wanted = [("No filter", (None, None))] + [(label, (hp, lp)) for label, hp, lp in presets]
        have = [self.preset_combo.itemText(i) for i in range(self.preset_combo.count())]
        if have != [t for t, _d in wanted] + ["Custom"]:
            self.preset_combo.clear()
            for text, data in wanted:
                self.preset_combo.addItem(text, data)
            self.preset_combo.addItem("Custom", "custom")
        kind = "physio" if self._file_kind == "physio" else "meeg"
        label = "No filter" if tr.hp is None and tr.lp is None else preset_of(kind, tr.hp, tr.lp)
        index = self.preset_combo.findText(label) if label else -1
        self.preset_combo.setCurrentIndex(index if index >= 0 else self.preset_combo.count() - 1)

    def action_context(self) -> dict[str, Any]:
        src = self.source
        showing = src is not None and self.pages.currentWidget() is self.traces_page
        tr = self.ctx.scene.traces
        return {
            "traces": showing,
            "meeg": self._file_kind == "meeg",
            "physio": self._file_kind == "physio",
            "fit": showing and self._file_kind == "physio",
            "filtered": bool(tr.hp or tr.lp or tr.notch),
            "events": bool(src and src.event_sources()),
            "events.on": tr.events,
            "normalize": tr.normalize,
            "relatives": len(self._relatives) > 1,
            "together": self.together,
            "traces.butterfly": tr.butterfly,
            "traces.clip": tr.clip,
            "traces.remove_dc": tr.remove_dc,
            "traces.page_scale": tr.page_scale,
            "zen": self.zen,
            # Something to write: the session's bads differ from the file's.
            "bads.changed": bool(src is not None and tr.bads is not None
                                 and set(tr.bads) != set(src.bads)),
        }

    def toolbar_wanted(self) -> bool:
        """The controls act on traces; the metadata card has its own button.
        Zen mode hides them."""
        return (self.source is not None and self.pages.currentWidget() is self.traces_page
                and not self.zen)

    #: Which mouse tables the help shows.
    mouse_canvases = ("traces",)

    def apply_mode(self) -> None:
        pass

    def apply_graph(self) -> None:
        pass

    def restore_panels(self) -> None:
        pass

    def canvases(self, kind: str) -> list:
        if kind == "traces" and self._traces is not None and self._traces.isVisible():
            return [self._traces]
        return []

    def figure_widget(self) -> QWidget:
        if self._traces is not None and self.pages.currentWidget() is self.traces_page:
            return self._traces
        return self.pages

    # ------------------------------------------------------------------
    # GUI actions
    # ------------------------------------------------------------------

    def gui_action(self, name: str, params) -> bool:
        if name == "psd":
            self.show_psd(filtered=False)
            return True
        if name == "line":
            self.open_line_style()
            return True
        if name == "together":
            self.together = not self.together
            self.load_signal()
            return True
        if name == "close_signal":
            self.close_signal()
            return True
        if name == "write_bads":
            self.write_bads()
            return True
        if name == "zen":
            self.set_zen(not self.zen)
            return True
        return False

    def set_zen(self, on: bool) -> None:
        """The traces alone: no toolbar, no overview. Z again, or closing
        the signal, brings them back."""
        self.zen = bool(on)
        self.navigation.setVisible(not self.zen)
        self.viewer._sync_toolbar()
        self.viewer.refresh_actions()
        if self.zen:
            key = ", ".join(self.viewer.action_manager.keys_for("view.zen")) or "Z"
            self.viewer.status_message.emit(f"Zen mode: {key} brings the controls back")

    def write_bads(self) -> bool:
        """The channels marked bad in this session, into the recording's
        ``_channels.tsv`` (one undoable operation of the dataset)."""
        from ....editor.channels import set_bad_channels
        from ....viz.bids import dataset_root
        from ....viz.commands.signal import bad_channels
        from ....viz.data.signal import channels_sibling

        src = self.source
        if src is None:
            return False
        tsv = channels_sibling(src.path)
        if tsv is None:
            self.viewer.status_message.emit(
                f"{src.path.name} has no _channels.tsv beside it to write the bad "
                "channels into.")
            return False
        root = self.viewer.current_root() or dataset_root(src.path) or src.path.parent
        bads = bad_channels(self.ctx.store)
        try:
            changed = set_bad_channels(root, tsv, bads)
        except Exception as exc:  # noqa: BLE001 - said, never swallowed
            self.viewer.status_message.emit(f"The bad channels were not written: {exc}")
            return False
        src.bads = set(bads)
        src.bads_from = tsv.name
        self.ctx.store.changed({"traces.look"})
        self.viewer.refresh_actions()
        self.viewer.status_message.emit(
            f"{len(bads)} bad channel{'s' if len(bads) != 1 else ''} written to {tsv.name}"
            f" ({changed} row{'s' if changed != 1 else ''} changed); undo it from the "
            "Editor's history" if changed else f"{tsv.name} already says so")
        return True

    def close_signal(self) -> None:
        """Drop the samples, back to the metadata card (MEG/EEG) or out."""
        self._generation += 1
        for tag in ("signal", "resample", "psd"):
            self.ctx.jobs.cancel(tag)
        self.source = None
        self.ctx.store.sources.clear()
        if self.zen:
            self.set_zen(False)
        if self._file_kind == "meeg" and self.meta is not None:
            self.pages.setCurrentWidget(self.meta_page)
            self.viewer.refresh_actions()
            self.viewer._sync_toolbar()
            self.viewer.status_message.emit("Signal closed")
        self.viewer.close_requested.emit()

    def open_line_style(self) -> None:
        from ..panels.line_style import MAX_WIDTH, LineStyleDialog

        src = self.source
        traces = self._traces
        shown = len(traces.shown_channels()) if traces is not None else 1
        cap = MAX_WIDTH if (traces is None or traces.max_pen_width(shown) > 1) else 1
        ts = self.ctx.settings.traces
        types = list(dict.fromkeys(src.ch_types)) if src is not None else []
        dlg = LineStyleDialog(ts.line_width, ts.line_color or None,
                              allow_by_type=len(set(types)) > 1, channel_types=types,
                              max_width=cap, traces_shown=shown, parent=self.viewer)
        self.line_dialog = dlg
        dlg.exec()

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    def summary(self) -> str:
        src = self.source
        if src is not None:
            # A table names its BIDS suffix (physio, stim, motion); a
            # recording its format.
            kind = (suffix_of(src.path.name) or "physio") if self._file_kind == "physio" \
                else full_ext(src.path).lstrip(".")
            return (f"{len(src.ch_names)} ch · {src.sfreq:g} Hz · "
                    f"{src.duration:.1f} s · {kind}")
        if self.meta:
            m = self.meta
            return f"{m.get('n_channels', 0)} ch · {m.get('sfreq', 0):g} Hz"
        return ""

    def readout(self) -> str:
        tr = self.ctx.scene.traces
        src = self.source
        if src is None:
            return ""
        from ....viz.compute.filters import FilterSpec

        spec = FilterSpec(tr.hp, tr.lp, tr.notch)
        return spec.describe() if spec.active else ""


class ChannelPicker(QDialog):
    """Pick channels: a searchable list, filterable by type."""

    def __init__(self, src: SignalSource, picks: Optional[list[str]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select channels")
        self.resize(380, 540)
        self._src = src
        lay = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search channels")
        lay.addWidget(self.search)
        row = QHBoxLayout()
        row.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.setObjectName("ent-input")
        self.type_combo.addItem("all", "all")
        for t in src.available_types:
            self.type_combo.addItem(src.type_label(t), t)
        row.addWidget(self.type_combo, 1)
        lay.addLayout(row)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        chosen = set(picks) if picks is not None else None
        for name, kind in zip(src.ch_names, src.ch_types):
            item = QListWidgetItem(f"{name}  [{src.type_label(kind)}]")
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setData(Qt.ItemDataRole.UserRole + 1, kind)
            self.list.addItem(item)
            item.setSelected(chosen is None or name in chosen)
        lay.addWidget(self.list, 1)
        quick = QHBoxLayout()
        all_btn = QPushButton("Select all")
        none_btn = QPushButton("Select none")
        all_btn.clicked.connect(self.list.selectAll)
        none_btn.clicked.connect(self.list.clearSelection)
        quick.addWidget(all_btn)
        quick.addWidget(none_btn)
        lay.addLayout(quick)
        self.search.textChanged.connect(self._filter)
        self.type_combo.currentIndexChanged.connect(lambda _i: self._filter())
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _filter(self) -> None:
        text = self.search.text().lower()
        kind = self.type_combo.currentData()
        for i in range(self.list.count()):
            item = self.list.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            ch_kind = item.data(Qt.ItemDataRole.UserRole + 1)
            item.setHidden(bool(text and text not in name.lower())
                           or (kind != "all" and ch_kind != kind))

    def selected(self) -> Optional[list[str]]:
        names = [self.list.item(i).data(Qt.ItemDataRole.UserRole)
                 for i in range(self.list.count()) if self.list.item(i).isSelected()]
        return None if len(names) == len(self._src.ch_names) else names


__all__ = ["ChannelPicker", "SignalPresenter"]
