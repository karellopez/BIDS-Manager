"""Signals in the viewer: MEG/EEG on their metadata card, physio in the TSV
pane, one traces canvas for both.

The Qt-free halves (reading, events, filters, the spectrum, decimation) are
tested in ``tests/unit/test_viz_signal_data.py`` and
``tests/unit/test_viz_physio.py``. These are what would be wrong in a way
somebody notices by looking: controls that do not suit the recording, a pen
that makes a drag crawl, a gap drawn as a spike, an event in the wrong place.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from bidsmgr.gui.viz import Viewer  # noqa: E402
from bidsmgr.gui.viz.bridge import SettingsHub  # noqa: E402
from tests.fixtures.signals import write_fif, write_physio  # noqa: E402

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def rec(tmp_path: Path) -> Path:
    """Five seconds of EEG with two stim pulses and a curated events table."""
    path = write_fif(tmp_path, "sub-01_task-rest_eeg.fif", sfreq=200.0, seconds=5.0,
                     stim_at=(40, 300))
    (tmp_path / "sub-01_task-rest_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0.5\t0\tgo\n2.5\t1.0\tstop\n")
    return path


@pytest.fixture()
def recording(tmp_path: Path) -> Path:
    """Two physio channels at 100 Hz, starting before the scanner did."""
    rows = [[math.sin(i / 10.0), 5.0 if i % 25 == 0 else 0.0] for i in range(1000)]
    return write_physio(
        tmp_path, "sub-001_task-x_recording-both_physio.tsv.gz", rows,
        {"Columns": ["cardiac", "trigger"], "SamplingFrequency": 100.0,
         "StartTime": -3.5})


@pytest.fixture()
def one_channel(tmp_path: Path) -> Path:
    rows = [[math.sin(i / 8.0)] for i in range(600)]
    return write_physio(
        tmp_path, "sub-001_task-x_run-01_recording-cardiac_physio.tsv.gz", rows,
        {"Columns": ["cardiac"], "SamplingFrequency": 100.0, "StartTime": 0.0})


@pytest.fixture()
def one_runs_relatives(tmp_path: Path, one_channel: Path) -> Path:
    write_physio(
        tmp_path, "sub-001_task-x_run-01_recording-respiratory_physio.tsv.gz",
        [[math.cos(i / 40.0)] for i in range(300)],
        {"Columns": ["respiratory"], "SamplingFrequency": 50.0, "StartTime": -1.0})
    write_physio(
        tmp_path, "sub-001_task-x_run-01_recording-trigger_physio.tsv.gz",
        [[5.0 if i % 20 == 0 else 0.0] for i in range(600)],
        {"Columns": ["trigger"], "SamplingFrequency": 100.0, "StartTime": 0.0})
    return one_channel


def _viewer(qtbot, size=(1000, 640)) -> Viewer:
    viewer = Viewer(kind="signal")
    qtbot.addWidget(viewer)
    viewer.resize(*size)
    viewer.show()
    qtbot.waitExposed(viewer)
    return viewer


def _open(qtbot, viewer: Viewer, path: Path, root=None) -> Viewer:
    with qtbot.waitSignal(viewer.loaded, timeout=30_000):
        viewer.set_file(path, root if root is not None else path.parent)
    viewer.qstore.flush()
    return viewer


def _load_signal(qtbot, viewer: Viewer) -> Viewer:
    with qtbot.waitSignal(viewer.loaded, timeout=30_000):
        viewer.presenter.load_button.click()
    viewer.qstore.flush()
    return viewer


def _traces(viewer: Viewer):
    canvas = viewer.presenter.traces
    assert canvas is not None
    return canvas


def _pane(qtbot):
    from bidsmgr.gui.widgets.tsv_viewer_pane import TsvViewerPane

    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    pane.resize(900, 600)
    pane.show()
    qtbot.waitExposed(pane)
    return pane


def _load_table(qtbot, pane, path, root) -> None:
    pane.set_file(path, root)
    qtbot.waitUntil(lambda: pane._loader is None, timeout=20_000)


def _visualize(qtbot, pane, path, root) -> Viewer:
    """The physio file, opened in the TSV pane's embedded signal viewer."""
    _load_table(qtbot, pane, path, root)
    pane._visualize_btn.setChecked(True)
    viewer = pane.visualizer()
    assert viewer is not None
    qtbot.waitUntil(lambda: viewer.source() is not None and viewer.page() == "content",
                    timeout=30_000)
    viewer.qstore.flush()
    return viewer


def _set_traces(**values) -> None:
    def apply(s):
        for key, value in values.items():
            setattr(s.traces, key, value)

    SettingsHub.instance().update(apply)


# ---------------------------------------------------------------------------
# MEG and EEG: the card, then the signal
# ---------------------------------------------------------------------------


class TestMetadataFirst:
    def test_the_viewer_starts_on_its_hint(self, qtbot):
        viewer = _viewer(qtbot)
        assert viewer.page() == "hint"
        assert "EEG or MEG" in viewer.hint_text()

    def test_a_recording_opens_on_its_card_not_its_samples(self, qtbot, rec):
        viewer = _open(qtbot, _viewer(qtbot), rec)
        p = viewer.presenter
        assert viewer.page() == "content"
        assert p.pages.currentWidget() is p.meta_page
        assert p.meta["n_channels"] == 5
        assert p.source is None, "nothing preloaded until asked"
        assert p.traces is None, "and no plot built for a card"
        assert not viewer.toolbar_visible(), "the controls act on traces"

    def test_load_signal_shows_the_traces(self, qtbot, rec):
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        p = viewer.presenter
        assert p.pages.currentWidget() is p.traces_page
        assert viewer.toolbar_visible()
        types = [p.type_combo.itemData(i) for i in range(p.type_combo.count())]
        assert types == ["all", "eeg", "eog", "stim"]
        assert _traces(viewer).label_texts() == ["Fz", "Cz", "Pz", "EOG", "STI"]
        assert viewer.action("events.toggle").isEnabled(), "events.tsv and stim"

    def test_every_control_runs(self, qtbot, rec):
        """Each control is a command; together they must not raise and must
        leave a drawable state."""
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        viewer.run("traces.type", ch_type="eeg")
        viewer.run("traces.count", n=2)
        viewer.run("traces.scale", value=2.0)
        viewer.run("time.width", seconds=2.0)
        for action in ("time.next", "time.end", "time.start", "traces.normalize",
                       "traces.normalize", "channels.next", "traces.bigger",
                       "events.toggle"):
            viewer.trigger(action)
        viewer.qstore.flush()
        tr = viewer.scene.traces
        assert tr.ch_type == "eeg" and tr.count == 2 and tr.events
        assert _traces(viewer).label_texts() == ["Cz", "Pz"]
        viewer.trigger("traces.reset")
        viewer.qstore.flush()
        assert viewer.scene.traces.ch_type == "all"
        assert viewer.scene.traces.width == pytest.approx(5.0, abs=0.01)

    def test_a_filter_is_applied_and_removed(self, qtbot, rec):
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        p = viewer.presenter
        p.hp_spin.setValue(1.0)
        p.lp_spin.setValue(40.0)
        p.apply_filters()
        viewer.qstore.flush()
        assert (viewer.scene.traces.hp, viewer.scene.traces.lp) == (1.0, 40.0)
        assert viewer.action("traces.reset_filters").isEnabled()
        assert "high-pass 1 Hz" in viewer.readout_text()
        viewer.trigger("traces.reset_filters")
        viewer.qstore.flush()
        assert viewer.scene.traces.hp is None and viewer.scene.traces.lp is None

    def test_a_cut_off_above_nyquist_is_refused_with_the_reason(
        self, qtbot, rec, monkeypatch,
    ):
        """The old view swallowed MNE's error and said "Filter applied"."""
        from PyQt6.QtWidgets import QMessageBox

        shown = []
        monkeypatch.setattr(QMessageBox, "information",
                            lambda *a, **k: shown.append(a[2]))
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        p = viewer.presenter
        p.lp_spin.setValue(150.0)          # Nyquist is 100 Hz
        with qtbot.waitSignal(viewer.status_message, timeout=2000) as said:
            p.apply_filters()
        assert "Nyquist" in said.args[0]
        assert shown and "Nyquist" in shown[0]
        assert viewer.scene.traces.lp is None, "and nothing changed"

    def test_close_goes_back_to_the_card(self, qtbot, rec):
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        with qtbot.waitSignal(viewer.close_requested, timeout=2000):
            viewer.trigger("signal.close")
        p = viewer.presenter
        assert p.source is None
        assert p.pages.currentWidget() is p.meta_page
        assert not viewer.toolbar_visible()

    def test_none_clears(self, qtbot, rec):
        viewer = _open(qtbot, _viewer(qtbot), rec)
        viewer.set_file(None, None)
        assert viewer.page() == "hint"
        assert viewer.presenter.meta is None

    def test_an_unreadable_recording_says_so(self, qtbot, tmp_path):
        bad = tmp_path / "sub-01_task-x_eeg.edf"
        bad.write_bytes(b"not an edf at all")
        viewer = _viewer(qtbot)
        with qtbot.waitSignal(viewer.load_failed, timeout=20_000):
            viewer.set_file(bad, tmp_path)

    def test_a_theme_swap_repaints_the_plot(self, qtbot, rec):
        """pyqtgraph reads no QSS, so the palette has to be handed down."""
        from bidsmgr.gui import theme_manager
        from bidsmgr.gui.viz.bridge import ThemeHub

        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        plot = _traces(viewer).plot
        try:
            viewer.repaint_for_palette(theme_manager.LIGHT)
            assert plot.backgroundBrush().color().name().lower() == \
                theme_manager.LIGHT["bg"].lower()
        finally:
            ThemeHub.instance().publish(theme_manager.DARK)
        assert plot.backgroundBrush().color().name().lower() == \
            theme_manager.DARK["bg"].lower()


class TestEvents:
    def test_turning_them_on_jumps_to_the_first(self, qtbot, tmp_path):
        """Triggers often start well into a recording (the lab's MEG sample:
        102 s); turning events on over an empty window looked broken."""
        path = write_fif(tmp_path, "sub-01_task-x_eeg.fif", sfreq=100.0, seconds=60.0)
        (tmp_path / "sub-01_task-x_events.tsv").write_text("onset\tduration\n50.0\t0\n")
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), path))
        assert viewer.scene.traces.t0 == 0.0
        viewer.trigger("events.toggle")
        viewer.qstore.flush()
        tr = viewer.scene.traces
        assert tr.t0 < 50.0 < tr.t0 + tr.width

    def test_a_trigger_is_drawn_where_it_happened(self, qtbot, tmp_path):
        """Not ``first_samp`` late: the lab's own MEG put every trigger 96 s
        after it happened."""
        path = write_fif(tmp_path, "sub-01_task-x_eeg.fif", sfreq=100.0, seconds=10.0,
                         stim_at=(250,), first_samp=5000)
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), path))
        viewer.trigger("events.toggle")
        viewer.qstore.flush()
        lines = [item for item in _traces(viewer).overlay_items()
                 if type(item).__name__ == "InfiniteLine"]
        assert [round(line.value(), 3) for line in lines] == [2.5]

    def test_an_event_that_lasts_is_a_span(self, qtbot, rec):
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        viewer.trigger("events.toggle")
        viewer.qstore.flush()
        kinds = [type(item).__name__ for item in _traces(viewer).overlay_items()]
        assert kinds.count("LinearRegionItem") == 1, "stop lasts a second"
        assert kinds.count("InfiniteLine") == 2


class TestThePowerSpectrum:
    def test_it_opens_for_the_channels_shown(self, qtbot, rec):
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        viewer.run("traces.type", ch_type="eeg")
        p = viewer.presenter
        p.show_psd(filtered=False)
        qtbot.waitUntil(lambda: getattr(p, "psd_window", None) is not None, timeout=30_000)
        win = p.psd_window
        qtbot.addWidget(win)
        assert win._names == ["Fz", "Cz", "Pz"]
        assert win.tabs.count() == 2
        win.db_box.setChecked(False)
        win.db_box.setChecked(True)
        assert p.psd_button.isEnabled()

    def test_the_filtered_spectrum_says_it_is_filtered(self, qtbot, rec):
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        viewer.run("traces.filter", hp=None, lp=30.0, notch=None)
        p = viewer.presenter
        p.show_psd(filtered=True)
        qtbot.waitUntil(lambda: getattr(p, "psd_window", None) is not None, timeout=30_000)
        qtbot.addWidget(p.psd_window)
        assert "low-pass 30 Hz" in p.psd_window.what.text()

    def test_more_names_than_rows_does_not_break_it(self, qtbot):
        from bidsmgr.gui.viz.canvases.psd import PsdWindow

        win = PsdWindow({
            "freqs": np.linspace(1.0, 40.0, 40),
            "data": np.random.default_rng(0).random((3, 40)) * 1e-10,
            "ch_names": ["a", "b", "c", "d", "e"],
            "ch_types": ["eeg", "eeg", "eeg", "mag", "mag"],
        })
        qtbot.addWidget(win)
        assert win.tabs.count() == 2


class TestResampling:
    def test_it_keeps_the_view_and_changes_the_rate(self, qtbot, rec):
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        p = viewer.presenter
        viewer.run("traces.type", ch_type="eeg")
        p.resample_spin.setValue(100.0)
        p.resample()
        qtbot.waitUntil(lambda: p.source is not None and p.source.sfreq == 100.0,
                        timeout=30_000)
        viewer.qstore.flush()
        assert viewer.scene.traces.ch_type == "eeg", "the view is kept"
        assert p.resample_button.isEnabled()


# ---------------------------------------------------------------------------
# Physio in the TSV pane
# ---------------------------------------------------------------------------


class TestTheTablePaneOffersIt:
    def test_a_recording_gets_a_visualize_button(self, qtbot, recording, tmp_path):
        pane = _pane(qtbot)
        _load_table(qtbot, pane, recording, tmp_path)
        assert pane._visualize_btn.isVisibleTo(pane)
        assert pane._visualize_btn.text().strip() == "Visualize"

    def test_an_events_table_does_not(self, qtbot, tmp_path):
        folder = tmp_path / "sub-001" / "func"
        folder.mkdir(parents=True)
        events = folder / "sub-001_task-x_events.tsv"
        events.write_text("onset\tduration\n0.0\t1.0\n")
        pane = _pane(qtbot)
        _load_table(qtbot, pane, events, tmp_path)
        assert not pane._visualize_btn.isVisibleTo(pane)

    def test_pressing_it_opens_the_shared_viewer(self, qtbot, recording, tmp_path):
        pane = _pane(qtbot)
        viewer = _visualize(qtbot, pane, recording, tmp_path)
        assert isinstance(viewer, Viewer) and viewer.kind == "signal"
        assert pane._stack.currentWidget() is viewer
        assert viewer.source().ch_names == ["cardiac", "trigger"]
        assert viewer.source().n_times == 1000, "the first sample is not a header"

    def test_pressing_it_again_goes_back_to_the_table(self, qtbot, recording, tmp_path):
        pane = _pane(qtbot)
        _visualize(qtbot, pane, recording, tmp_path)
        pane._visualize_btn.setChecked(False)
        assert pane._stack.currentIndex() == 1
        assert pane.visualizer().source() is None, "the samples are dropped"

    def test_the_viewer_does_not_pin_the_pane_wide(self, qtbot, recording, tmp_path):
        pane = _pane(qtbot)
        _visualize(qtbot, pane, recording, tmp_path)
        assert pane.minimumSizeHint().width() <= 400

    def test_the_viewer_speaks_through_the_pane(self, qtbot, recording, tmp_path):
        pane = _pane(qtbot)
        seen = []
        pane.status_message.connect(seen.append)
        _visualize(qtbot, pane, recording, tmp_path)
        assert any("Loaded 2 channels" in s for s in seen)

    def test_physio_is_drawn_in_run_time(self, qtbot, recording, tmp_path):
        """StartTime -3.5: sample zero is three and a half seconds before the
        scanner started, so the axis starts there, and the run's events land
        where they happened."""
        viewer = _visualize(qtbot, _pane(qtbot), recording, tmp_path)
        x, _y = _traces(viewer).drawn()[0]
        assert x[0] == pytest.approx(-3.5)


class TestControlsThatSuitTheChannelCount:
    def test_one_channel_hides_the_channel_controls(self, qtbot, one_channel, tmp_path):
        viewer = _visualize(qtbot, _pane(qtbot), one_channel, tmp_path)
        p = viewer.presenter
        assert viewer.source().ch_names == ["cardiac"]
        assert not p.type_combo.isVisibleTo(viewer)
        assert not p.count_control.isVisibleTo(viewer)
        assert not p.channels_button.isVisibleTo(viewer)

    def test_several_channels_show_them(self, qtbot, recording, tmp_path):
        viewer = _visualize(qtbot, _pane(qtbot), recording, tmp_path)
        p = viewer.presenter
        assert p.type_combo.isVisibleTo(viewer)
        assert p.count_control.isVisibleTo(viewer)
        assert p.channels_button.isVisibleTo(viewer)

    def test_together_is_offered_only_with_relatives(
        self, qtbot, one_runs_relatives, tmp_path,
    ):
        viewer = _visualize(qtbot, _pane(qtbot), one_runs_relatives, tmp_path)
        assert viewer.action("traces.together").isEnabled()

    def test_a_lone_recording_does_not_offer_it(self, qtbot, one_channel, tmp_path):
        viewer = _visualize(qtbot, _pane(qtbot), one_channel, tmp_path)
        assert not viewer.action("traces.together").isEnabled()

    def test_together_brings_the_controls_back(self, qtbot, one_runs_relatives, tmp_path):
        viewer = _visualize(qtbot, _pane(qtbot), one_runs_relatives, tmp_path)
        viewer.trigger("traces.together")
        qtbot.waitUntil(lambda: viewer.source() is not None
                        and len(viewer.source().ch_names) == 3, timeout=30_000)
        viewer.qstore.flush()
        p = viewer.presenter
        assert viewer.action("traces.together").isChecked()
        assert viewer.source().start_time == -1.0, "the belt started first"
        assert p.type_combo.isVisibleTo(viewer)
        assert p.count_control.isVisibleTo(viewer)

    def test_physio_has_no_close_button_and_meg_has(self, qtbot, recording, tmp_path, rec):
        """Physio is closed by its pane's Visualize toggle; MEG goes back to
        its card."""
        physio = _visualize(qtbot, _pane(qtbot), recording, tmp_path)
        assert not physio.action("signal.close").isEnabled()
        assert not physio.button("signal.close").isVisibleTo(physio), "not even greyed out"
        meg = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        assert meg.action("signal.close").isEnabled()
        assert meg.button("signal.close").isVisibleTo(meg)


# ---------------------------------------------------------------------------
# How a trace is drawn
# ---------------------------------------------------------------------------


class TestHowTheTraceIsDrawn:
    def test_a_few_traces_draw_thick(self, qtbot, recording, tmp_path):
        """Two physio channels can afford the wider pen; a hairline reads as
        a scratch."""
        viewer = _visualize(qtbot, _pane(qtbot), recording, tmp_path)
        canvas = _traces(viewer)
        assert viewer.presenter.ctx.settings.traces.line_width == 0, "automatic"
        assert canvas.pen_width() >= 2
        assert canvas._curves[0].opts["pen"].width() >= 2

    def test_many_traces_draw_thin(self, qtbot, recording, tmp_path):
        """Not taste: Qt strokes a wider pen through its general path code
        (79.8 ms against 11.3 ms for eight traces) and a drag pays it on
        every frame."""
        viewer = _visualize(qtbot, _pane(qtbot), recording, tmp_path)
        assert _traces(viewer).pen_width(n_shown=40) == 1

    def test_a_chosen_width_is_honoured_where_affordable(self, qtbot, recording, tmp_path):
        viewer = _visualize(qtbot, _pane(qtbot), recording, tmp_path)
        _set_traces(line_width=6)
        canvas = _traces(viewer)
        assert canvas.pen_width(n_shown=1) == 6
        assert canvas.pen_width(n_shown=2) == 6

    def test_a_chosen_width_is_CAPPED_where_it_is_not(self, qtbot, recording, tmp_path):
        viewer = _visualize(qtbot, _pane(qtbot), recording, tmp_path)
        _set_traces(line_width=6)
        canvas = _traces(viewer)
        assert canvas.pen_width(n_shown=20) == 1
        assert canvas.pen_width(n_shown=300) == 1

    def test_the_line_settings_change_the_trace(self, qtbot, recording, tmp_path):
        """Every open viewer follows a change to the settings at once."""
        viewer = _visualize(qtbot, _pane(qtbot), recording, tmp_path)
        _set_traces(line_width=6, line_color="#ff7b72")
        pen = _traces(viewer)._curves[0].opts["pen"]
        assert pen.width() == 6
        assert pen.color().name() == "#ff7b72"

    def test_a_gap_is_a_break_not_a_spike(self, qtbot, tmp_path):
        """A dropped sample is not a value. Drawn as the zero it is stored
        as, it reads as an artefact on a trace centred anywhere else."""
        rows = [[float("nan") if 100 <= i < 200 else 50.0 + math.sin(i / 5.0)]
                for i in range(600)]
        path = write_physio(tmp_path, "sub-001_task-x_recording-cardiac_physio.tsv.gz",
                            rows, {"Columns": ["cardiac"], "SamplingFrequency": 100.0})
        viewer = _visualize(qtbot, _pane(qtbot), path, tmp_path)
        viewer.trigger("time.fit")
        viewer.qstore.flush()
        _x, y = _traces(viewer).drawn()[0]
        assert np.isnan(y).any(), "the gap must not be drawn as a value"
        assert np.isfinite(y).any()

    def test_physio_offers_fit_all_and_meg_does_not(self, qtbot, recording, tmp_path, rec):
        """Physio is a channel or four and fits whole; a MEG recording is
        hundreds of millions of samples."""
        physio = _visualize(qtbot, _pane(qtbot), recording, tmp_path)
        assert physio.action("time.fit").isEnabled()
        assert physio.button("time.fit").isVisibleTo(physio)
        meg = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        assert not meg.action("time.fit").isEnabled()
        for action in ("time.fit", "traces.together"):
            assert not meg.button(action).isVisibleTo(meg), (
                f"{action} can never apply to a MEG recording, so it is not shown")

    def test_fit_all_shows_the_whole_recording(self, qtbot, recording, tmp_path):
        viewer = _visualize(qtbot, _pane(qtbot), recording, tmp_path)
        viewer.run("time.width", seconds=1.0)
        viewer.trigger("time.fit")
        viewer.qstore.flush()
        tr = viewer.scene.traces
        assert tr.width == pytest.approx(viewer.source().duration)
        assert tr.t0 == 0.0

    def _quiet_then_loud(self, qtbot, tmp_path):
        rows = [[(1.0 if i < 300 else 100.0) * math.sin(i / 5.0)] for i in range(600)]
        path = write_physio(tmp_path, "sub-001_task-x_recording-cardiac_physio.tsv.gz",
                            rows, {"Columns": ["cardiac"], "SamplingFrequency": 100.0})
        viewer = _visualize(qtbot, _pane(qtbot), path, tmp_path)
        viewer.run("time.width", seconds=2.0)
        viewer.run("traces.option", field="clip", value=False)

        def drawn_height(start):
            viewer.run("time.set", t0=start)
            viewer.qstore.flush()
            _x, y = _traces(viewer).drawn()[0]
            return float(np.nanmax(y) - np.nanmin(y))

        return viewer, drawn_height

    def test_pages_are_drawn_at_one_scale_so_they_compare(self, qtbot, tmp_path):
        """A quiet stretch looks quiet beside a loud one: one amplitude per
        type for the whole recording (the per-page scale it replaces drew
        both the same size)."""
        _viewer, drawn_height = self._quiet_then_loud(qtbot, tmp_path)
        assert drawn_height(4.0) == pytest.approx(100 * drawn_height(0.5), rel=0.1)

    def test_the_page_can_be_scaled_to_itself(self, qtbot, tmp_path):
        """Zoomed into a quiet stretch, a quiet stretch fills the pane."""
        viewer, drawn_height = self._quiet_then_loud(qtbot, tmp_path)
        viewer.run("traces.option", field="page_scale", value=True)
        assert drawn_height(0.5) == pytest.approx(drawn_height(4.0), rel=0.05)

    def test_a_one_sample_trigger_survives_the_decimation(self, qtbot, tmp_path):
        rows = [[0.0] for _ in range(200_000)]
        rows[137_421] = [9.0]
        path = write_physio(tmp_path, "sub-001_task-x_recording-trigger_physio.tsv.gz",
                            rows, {"Columns": ["trigger"], "SamplingFrequency": 1000.0})
        viewer = _visualize(qtbot, _pane(qtbot), path, tmp_path)
        viewer.trigger("time.fit")
        viewer.qstore.flush()
        x, y = _traces(viewer).drawn()[0]
        assert y.size < 5000, "decimated to about two points per pixel"
        assert x[int(np.nanargmax(y))] == pytest.approx(137.421, abs=0.5)


class TestWhatARedrawCosts:
    """Measured lessons, kept as tests so they stay learned."""

    def test_a_change_of_scale_reads_nothing(self, qtbot, rec, monkeypatch):
        """Scale, normalisation, colour and theme redraw from the decimated
        envelopes of the window on screen. Re-reading and re-decimating a
        five-channel, four-million-sample physio run cost 179 ms a key press."""
        from bidsmgr.gui.viz.canvases import traces as T

        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        reads = []
        real = T.segment
        monkeypatch.setattr(T, "segment", lambda *a, **k: (reads.append(1), real(*a, **k))[1])
        for action in ("traces.bigger", "traces.normalize", "traces.smaller"):
            viewer.trigger(action)
            viewer.qstore.flush()
        assert reads == []
        viewer.run("time.width", seconds=2.0)
        viewer.trigger("time.next")
        viewer.qstore.flush()
        assert reads, "a new window is read"

    def test_a_new_recording_is_never_drawn_from_the_old_ones_cache(self, qtbot, tmp_path):
        a = write_fif(tmp_path, "sub-01_task-a_eeg.fif", seconds=5.0)
        b = write_fif(tmp_path, "sub-01_task-b_eeg.fif", seconds=5.0)
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), a))
        first = [y.copy() for _x, y in _traces(viewer).drawn()]
        viewer = _load_signal(qtbot, _open(qtbot, viewer, b))
        assert _traces(viewer)._env_key is not None
        # Same shape, same noise seed: only the cache would make them equal
        # if it were keyed on the window alone. Different files must still
        # be redrawn from their own samples.
        assert _traces(viewer).source.path == b
        assert len(_traces(viewer).drawn()) == len(first)

    def test_the_names_are_one_widget_whatever_the_count(self, qtbot, rec):
        """A QLabel per channel cost 0.8 s for a hundred channels, each one
        polished against the app stylesheet as it appeared."""
        from PyQt6.QtWidgets import QWidget

        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        canvas = _traces(viewer)
        viewer.run("traces.count", n=2)
        viewer.qstore.flush()
        few = len(canvas.findChildren(QWidget))
        viewer.run("traces.count", n=5)
        viewer.qstore.flush()
        assert len(canvas.findChildren(QWidget)) == few
        assert canvas.label_texts() == ["Fz", "Cz", "Pz", "EOG", "STI"]

    def test_each_name_sits_level_with_its_trace(self, qtbot, rec):
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        canvas = _traces(viewer)
        strip = canvas._strip
        n = len(strip.names)
        for i in range(n):
            y = canvas._label_y(n - 1 - i)          # the trace's own baseline
            assert strip.band_at(y) == i


class TestTheLineDialog:
    def _dialog(self, qtbot, *args, **kwargs):
        from bidsmgr.gui.viz.panels.line_style import LineStyleDialog

        dlg = LineStyleDialog(*args, **kwargs)
        qtbot.addWidget(dlg)
        return dlg

    def test_it_does_not_offer_a_width_that_would_be_capped(self, qtbot):
        """Offering a number and then ignoring it is how an app looks broken."""
        wide = self._dialog(qtbot, 0, None, max_width=1, traces_shown=20)
        assert wide._slider.maximum() == 1
        assert not wide._slider.isEnabled()
        narrow = self._dialog(qtbot, 0, None, max_width=8, traces_shown=1)
        assert narrow._slider.maximum() == 8
        assert narrow._slider.isEnabled()

    def test_the_viewer_opens_it_capped_when_many_traces_show(self, qtbot, rec, monkeypatch):
        from bidsmgr.gui.viz.panels.line_style import LineStyleDialog

        monkeypatch.setattr(LineStyleDialog, "exec", lambda self: 0)
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec))
        viewer.trigger("traces.line")
        dlg = viewer.presenter.line_dialog
        qtbot.addWidget(dlg)
        assert dlg._slider.maximum() == 1, "five traces on screen: a hairline only"
        assert list(dlg._swatches) == ["eeg", "eog", "stim"]

    def test_it_lists_only_the_types_the_viewer_has(self, qtbot):
        dlg = self._dialog(qtbot, 0, None, channel_types=["mag", "grad", "mag"])
        assert list(dlg._swatches) == ["mag", "grad"], "deduped, order kept"
        assert dlg._reset_btn is not None

    def test_a_single_channel_view_is_not_asked_about_types(self, qtbot):
        dlg = self._dialog(qtbot, 2, "#58a6ff", allow_by_type=False, channel_types=())
        assert dlg._swatches == {}
        assert dlg._reset_btn is None
        assert not dlg._by_type.isEnabled()

    def test_a_type_colour_is_stored_and_reset_deletes_it(self, qtbot):
        dlg = self._dialog(qtbot, 0, None, channel_types=["mag", "grad"])
        seen = []
        dlg.type_colors_changed.connect(seen.append)
        dlg.set_type_colour("grad", "#ff7b72")
        assert SettingsHub.instance().settings.traces.type_colors == {"grad": "#ff7b72"}
        dlg._reset_type_colours()
        assert seen[-1] == {}
        assert SettingsHub.instance().settings.traces.type_colors == {}

    def test_automatic_width_reads_as_automatic(self, qtbot):
        dlg = self._dialog(qtbot, 0, None, channel_types=["mag"])
        assert dlg._width_label.text() == "automatic"
        dlg._slider.setValue(4)
        assert dlg._width_label.text() == "4 px"
        assert SettingsHub.instance().settings.traces.line_width == 4, (
            "written as the user moves, judged on the plot behind")


# ---------------------------------------------------------------------------
# The Editor
# ---------------------------------------------------------------------------


class TestTheEditor:
    def test_a_recording_routes_to_the_signal_viewer(self, qtbot, rec):
        from bidsmgr.gui.editor_panel import EditorPanel

        ep = EditorPanel()
        qtbot.addWidget(ep)
        ep._set_root(rec.parent, persist=False)
        with qtbot.waitSignal(ep._signal_viewer.loaded, timeout=30_000):
            ep._on_file_selected(rec)
        assert ep._center_stack.currentWidget() is ep._signal_viewer
        assert ep._signal_viewer.presenter.meta["n_channels"] == 5

    def test_moving_on_clears_the_viewer_left_behind(self, qtbot, rec):
        from bidsmgr.gui.editor_panel import EditorPanel

        ep = EditorPanel()
        qtbot.addWidget(ep)
        ep._set_root(rec.parent, persist=False)
        with qtbot.waitSignal(ep._signal_viewer.loaded, timeout=30_000):
            ep._on_file_selected(rec)
        ep._on_file_selected(rec.parent / "sub-01_task-rest_events.tsv")
        qtbot.waitUntil(lambda: ep._tsv_viewer._loader is None, timeout=20_000)
        assert ep._signal_viewer.current_file() is None
        assert ep._signal_viewer.page() == "hint"

    def test_recordings_get_their_own_tree_icon(self, qapp):
        """File and folder-shaped recordings get the brain-and-signal icon;
        a plain folder keeps the folder icon."""
        from bidsmgr.gui import icons

        assert not icons.icon_for_path("sub-01_task-x_eeg.edf").isNull()
        ds = icons.icon_for_path("sub-01_task-x_meg.ds", is_dir=True)
        mff = icons.icon_for_path("sub-01_eeg.mff", is_dir=True)
        folder = icons.icon_for_path("anat", is_dir=True)
        assert ds.cacheKey() != folder.cacheKey()
        assert mff.cacheKey() != folder.cacheKey()


# ---------------------------------------------------------------------------
# What a reviewer of MEG and EEG needs: bads, butterfly, scale bars, filters
# off the GUI thread
# ---------------------------------------------------------------------------


class TestReviewing:
    def _meg(self, qtbot, tmp_path, rec):
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), rec, tmp_path))
        return viewer, _traces(viewer)

    def test_a_click_on_a_name_marks_the_channel_bad(self, qtbot, tmp_path, rec):
        viewer, traces = self._meg(qtbot, tmp_path, rec)
        traces._on_name_clicked(1)               # Cz
        viewer.qstore.flush()
        assert viewer.scene.traces.bads == ["Cz"]
        assert 1 in traces._strip.bad_rows, "its name in the error colour"
        assert viewer.action("channels.write_bads").isEnabled()
        viewer.store.undo()
        assert viewer.scene.traces.bads is None

    def test_the_bads_are_written_to_channels_tsv(self, qtbot, tmp_path, rec):
        tsv = tmp_path / "sub-01_task-rest_channels.tsv"
        tsv.write_text("name\ttype\tunits\nFz\tEEG\tV\nCz\tEEG\tV\nPz\tEEG\tV\n"
                       "EOG\tEOG\tV\nSTI\tTRIG\tV\n")
        viewer, traces = self._meg(qtbot, tmp_path, rec)
        viewer.run("channels.toggle_bad", name="Pz")
        viewer.qstore.flush()
        viewer.trigger("channels.write_bads")
        assert "Pz\tEEG\tV\tbad" in tsv.read_text()
        assert not viewer.action("channels.write_bads").isEnabled(), "nothing left to write"

    def test_butterfly_overlays_each_type(self, qtbot, tmp_path, rec):
        viewer, traces = self._meg(qtbot, tmp_path, rec)
        viewer.trigger("traces.butterfly")
        viewer.qstore.flush()
        assert traces.label_texts() == ["eeg", "eog", "stim"]
        assert set(traces._offsets) == {0, 1, 2}

    def test_a_scale_bar_per_type_in_its_unit(self, qtbot, tmp_path, rec):
        viewer, traces = self._meg(qtbot, tmp_path, rec)
        shown = [t.toPlainText() for t in traces._bar_texts if t.isVisible()]
        assert any(text.endswith("µV") for text in shown)

    def test_a_filter_never_runs_on_the_gui_thread(self, qtbot, tmp_path, rec, monkeypatch):
        import threading

        from bidsmgr.viz.compute import filters

        viewer, traces = self._meg(qtbot, tmp_path, rec)
        main = threading.main_thread()
        threads = []
        real = filters.apply

        def spy(*a, **k):
            threads.append(threading.current_thread())
            return real(*a, **k)

        monkeypatch.setattr(filters, "apply", spy)
        viewer.run("traces.filter", hp=1.0, lp=30.0)
        viewer.qstore.flush()
        assert traces.is_preview(), "drawn unfiltered until the filter is done"
        qtbot.waitUntil(lambda: not traces.is_preview(), timeout=20_000)
        assert threads and all(t is not main for t in threads)
        viewer.run("time.page", n=1)
        viewer.qstore.flush()
        assert not traces.is_preview(), "after that, a page is a slice"

    def test_housekeeping_channels_are_not_in_all(self, qtbot, tmp_path):
        import mne

        n = 1000
        info = mne.create_info(["MEG0111", "IAS_X", "SYS201"], 500.0, ["mag", "ias", "syst"])
        raw = mne.io.RawArray(np.random.default_rng(0).normal(0, 1e-12, (3, n)), info,
                              verbose=False)
        path = tmp_path / "sub-01_task-x_meg.fif"
        raw.save(path, overwrite=True, verbose=False)
        viewer = _load_signal(qtbot, _open(qtbot, _viewer(qtbot), path, tmp_path))
        assert _traces(viewer).label_texts() == ["MEG0111"]
        viewer.run("traces.type", ch_type="ias")
        viewer.qstore.flush()
        assert _traces(viewer).label_texts() == ["IAS_X"]


# ---------------------------------------------------------------------------
# Navigating a recording: the overview, the time cursor, zen mode, the
# controls of the toolbar
# ---------------------------------------------------------------------------


@pytest.fixture()
def long_rec(tmp_path: Path) -> Path:
    """A minute of EEG with two stim pulses."""
    return write_fif(tmp_path, "sub-01_task-rest_eeg.fif", sfreq=200.0, seconds=60.0,
                     stim_at=(2000, 6000))


def _loaded(qtbot, path: Path) -> Viewer:
    viewer = _open(qtbot, _viewer(qtbot), path)
    return _load_signal(qtbot, viewer)


def _overview(qtbot, viewer: Viewer):
    bar = viewer.presenter.overview
    qtbot.waitUntil(lambda: bar.profile() is not None, timeout=15_000)
    return bar


class TestTheOverview:
    def test_it_draws_the_whole_recording_and_the_window_on_it(self, qtbot, long_rec):
        viewer = _loaded(qtbot, long_rec)
        bar = _overview(qtbot, viewer)
        assert bar.profile().size >= 100
        rect = bar.view_rect()
        inner = bar._inner()
        assert rect.left() == pytest.approx(inner.left(), abs=1)
        assert rect.width() == pytest.approx(inner.width() * 10.0 / 60.0, abs=1.5)
        assert bar.grab().width() > 0

    def test_a_click_outside_the_window_centres_it_there(self, qtbot, long_rec):
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest

        viewer = _loaded(qtbot, long_rec)
        bar = _overview(qtbot, viewer)
        inner = bar._inner()
        x = int(inner.left() + 0.75 * inner.width())
        QTest.mouseClick(bar, Qt.MouseButton.LeftButton, pos=QPoint(x, bar.height() // 2))
        viewer.qstore.flush()
        assert viewer.scene.traces.t0 == pytest.approx(45.0 - 5.0, abs=0.6)

    def test_dragging_the_window_moves_it(self, qtbot, long_rec):
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest

        viewer = _loaded(qtbot, long_rec)
        bar = _overview(qtbot, viewer)
        inner = bar._inner()
        rect = bar.view_rect()
        start = QPoint(int(rect.center().x()), bar.height() // 2)
        QTest.mousePress(bar, Qt.MouseButton.LeftButton, pos=start)
        assert viewer.scene.traces.t0 == 0.0, "a press inside the window does not jump"
        QTest.mouseMove(bar, start + QPoint(100, 0))
        QTest.mouseRelease(bar, Qt.MouseButton.LeftButton, pos=start + QPoint(100, 0))
        viewer.qstore.flush()
        assert viewer.scene.traces.t0 == pytest.approx(100 / inner.width() * 60.0, abs=0.6)

    def test_the_wheel_pages(self, qtbot, long_rec):
        from PyQt6.QtCore import QPoint, QPointF, Qt
        from PyQt6.QtGui import QWheelEvent
        from PyQt6.QtWidgets import QApplication

        viewer = _loaded(qtbot, long_rec)
        bar = _overview(qtbot, viewer)
        pos = QPointF(bar.width() / 2, bar.height() / 2)
        ev = QWheelEvent(pos, bar.mapToGlobal(pos), QPoint(0, 0), QPoint(0, -120),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(bar, ev)
        viewer.qstore.flush()
        assert viewer.scene.traces.t0 == pytest.approx(8.0), "one page is 80 % of the window"

    def test_it_follows_a_resampled_recording(self, qtbot, long_rec):
        viewer = _loaded(qtbot, long_rec)
        bar = _overview(qtbot, viewer)
        first = bar.profile()
        viewer.presenter.resample_spin.setValue(100.0)
        viewer.presenter.resample()
        qtbot.waitUntil(lambda: viewer.source().sfreq == 100.0, timeout=30_000)
        qtbot.waitUntil(lambda: bar.profile() is not None and bar.profile() is not first,
                        timeout=15_000)


class TestTheTimeCursor:
    def _click(self, canvas, t: float, row_y: float = 0.0) -> None:
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtTest import QTest

        scene_pt = canvas._vb.mapViewToScene(QPointF(t, row_y))
        QTest.mouseClick(canvas.plot.viewport(), Qt.MouseButton.LeftButton,
                         pos=canvas.plot.mapFromScene(scene_pt))

    def test_a_click_places_it_and_a_second_removes_it(self, qtbot, long_rec):
        viewer = _loaded(qtbot, long_rec)
        canvas = _traces(viewer)
        seen = []
        viewer.status_message.connect(seen.append)
        self._click(canvas, 3.0, row_y=canvas._offsets[0])
        viewer.qstore.flush()
        assert viewer.scene.cursor.time == pytest.approx(3.0, abs=0.05)
        assert canvas.cursor_visible()
        assert any(s.startswith("Cursor at 3.0") and "µV" in s for s in seen), seen
        self._click(canvas, 3.0)
        viewer.qstore.flush()
        assert viewer.scene.cursor.time is None
        assert not canvas.cursor_visible()

    def test_a_drag_does_not_place_it(self, qtbot, long_rec):
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtTest import QTest

        viewer = _loaded(qtbot, long_rec)
        canvas = _traces(viewer)
        a = canvas.plot.mapFromScene(canvas._vb.mapViewToScene(QPointF(5.0, 0.0)))
        b = canvas.plot.mapFromScene(canvas._vb.mapViewToScene(QPointF(3.0, 0.0)))
        vp = canvas.plot.viewport()
        QTest.mousePress(vp, Qt.MouseButton.LeftButton, pos=a)
        for k in range(1, 6):
            QTest.mouseMove(vp, a + (b - a) * k / 5)
        QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=b)
        viewer.qstore.flush()
        assert viewer.scene.cursor.time is None
        assert viewer.scene.traces.t0 > 1.0, "the drag scrubbed instead"

    def test_reset_view_removes_it(self, qtbot, long_rec):
        viewer = _loaded(qtbot, long_rec)
        viewer.run("cursor.time", t=2.0)
        viewer.qstore.flush()
        assert _traces(viewer).cursor_visible()
        viewer.trigger("traces.reset")
        viewer.qstore.flush()
        assert not _traces(viewer).cursor_visible()


class TestZenMode:
    def test_z_leaves_the_traces_alone_and_brings_the_controls_back(self, qtbot, long_rec):
        viewer = _loaded(qtbot, long_rec)
        p = viewer.presenter
        assert viewer._toolbar.isVisible() and p.navigation.isVisible()
        viewer.trigger("view.zen")
        assert not viewer._toolbar.isVisible()
        assert not p.navigation.isVisible()
        assert viewer.action("view.zen").isChecked()
        assert _traces(viewer).isVisible()
        viewer.trigger("view.zen")
        assert viewer._toolbar.isVisible() and p.navigation.isVisible()

    def test_closing_the_signal_ends_it(self, qtbot, long_rec):
        viewer = _loaded(qtbot, long_rec)
        viewer.trigger("view.zen")
        viewer.trigger("signal.close")
        assert not viewer.presenter.zen
        assert viewer.presenter.navigation.isVisibleTo(viewer.presenter.traces_page)


class TestTheToolbar:
    def test_amplitude_window_and_count_are_sliders_with_numbers(self, qtbot, long_rec):
        viewer = _loaded(qtbot, long_rec)
        p = viewer.presenter
        p.scale_control.type_value(2.5)
        p.width_control.type_value(20.0)
        p.count_control.type_value(2)
        viewer.qstore.flush()
        tr = viewer.scene.traces
        assert (tr.scale, tr.width, tr.count) == (2.5, 20.0, 2)
        assert p.width_control.maximum() == pytest.approx(60.0, abs=0.01), \
            "no longer than the recording"
        viewer.trigger("traces.bigger")
        viewer.qstore.flush()
        assert p.scale_control.value() == pytest.approx(2.5 * 1.25, abs=0.01)

    def test_a_preset_filters_at_once_and_a_typed_band_is_custom(self, qtbot, long_rec):
        viewer = _loaded(qtbot, long_rec)
        p = viewer.presenter
        combo = p.preset_combo
        assert combo.currentText() == "No filter"
        index = combo.findText("1 to 40 Hz")
        combo.setCurrentIndex(index)
        combo.activated.emit(index)
        viewer.qstore.flush()
        tr = viewer.scene.traces
        assert (tr.hp, tr.lp) == (1.0, 40.0)
        assert p.hp_spin.value() == 1.0 and p.lp_spin.value() == 40.0
        p.lp_spin.setValue(33.0)
        p.apply_filters()
        viewer.qstore.flush()
        assert combo.currentText() == "Custom"

    def test_physio_offers_physio_bands(self, qtbot, recording, tmp_path):
        viewer = _visualize(qtbot, _pane(qtbot), recording, tmp_path)
        texts = [viewer.presenter.preset_combo.itemText(i)
                 for i in range(viewer.presenter.preset_combo.count())]
        assert any(t.startswith("Breathing") for t in texts)
        assert "1 to 40 Hz" not in texts

    def test_the_event_styling_moved_to_settings(self, qtbot, long_rec):
        """The events row (width, colour, by-label) duplicated the Settings
        page; the toolbar keeps the toggle and the source."""
        viewer = _loaded(qtbot, long_rec)
        assert not hasattr(viewer.presenter, "events_row")
        assert viewer.presenter.event_source.isVisibleTo(viewer)


def test_the_help_names_the_clicks():
    from bidsmgr.gui.viz.help import shortcut_sections

    class _Manager:
        def keys_for(self, _id):
            return ()

    sections = dict(shortcut_sections(_Manager(), {}, "signal", canvases=("traces",)))
    rows = dict(sections["Mouse on the traces"])
    assert rows["Mark a channel bad, or good again"] == ["Click its name"]
    assert "Place the time cursor (click it again to remove it)" in rows


def test_two_events_a_moment_apart_print_one_label(qtbot, long_rec):
    """Both lines drawn, one name: two triggers 5 ms apart printed one
    label over the other."""
    (long_rec.parent / "sub-01_task-rest_events.tsv").write_text(
        "onset\tduration\ttrial_type\n1.000\t0\texternal_trigger\n"
        "1.005\t0\texternal_trigger\n6.0\t0\texternal_trigger\n")
    viewer = _loaded(qtbot, long_rec)
    viewer.run("events.source", which="events.tsv")
    viewer.run("events.show", value=True)
    viewer.qstore.flush()
    canvas = _traces(viewer)
    lines = [i for i in canvas._event_lines if i.isVisible()]
    texts = [i for i in canvas._event_texts if i.isVisible()]
    assert len(lines) == 3
    assert len(texts) == 2
