"""A continuous recording drawn as a signal, not read as a grid of numbers.

Whether the trigger fired where you expect, whether the ECG is flat for the
first minute, whether the respiratory belt came loose: all questions about
the SHAPE of the signal, and all unanswerable from six-decimal numbers in a
table.

The three facts a plot needs are not in the file. BIDS puts them in the
sidecar (``Columns``, ``SamplingFrequency``, ``StartTime``), which is also
what tells a continuous recording apart from a table of onsets, so that is
what decides whether a plot is offered at all.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.gui.widgets.physio_plot import (
    filter_signal,
    normalise,
    read_columns,
    read_timing,
    to_series,
    welch_psd,
)

pytestmark = pytest.mark.gui


def _write(root: Path, name: str, rows: list[list[float]], meta: dict) -> Path:
    folder = root / "sub-001" / "func"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write("\t".join(f"{v:g}" for v in row) + "\n")
    sidecar = folder / name.replace(".tsv.gz", ".json")
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    return path


@pytest.fixture()
def recording(tmp_path: Path) -> Path:
    """Two channels at 100 Hz, starting before the scanner did."""
    rows = [[math.sin(i / 10.0), 5.0 if i % 25 == 0 else 0.0]
            for i in range(1000)]
    return _write(
        tmp_path, "sub-001_task-x_recording-both_physio.tsv.gz", rows,
        {"Columns": ["cardiac", "trigger"], "SamplingFrequency": 100.0,
         "StartTime": -3.5},
    )


class TestWhatCountsAsPlottable:
    def test_a_recording_is(self, recording):
        timing = read_timing(recording)
        assert timing == {
            "columns": ["cardiac", "trigger"],
            "sampling_frequency": 100.0,
            "start_time": -3.5,
            "units": "",
        }

    def test_units_are_carried_when_the_sidecar_says(self, tmp_path):
        """An axis labelled "value" tells the reader less than "mmHg"."""
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0]],
            {"Columns": ["a"], "SamplingFrequency": 50, "Units": "mmHg"},
        )
        assert read_timing(path)["units"] == "mmHg"

    def test_a_table_of_onsets_is_not(self, tmp_path):
        """An events table has no sampling frequency, and that is exactly
        how the standard distinguishes the two."""
        folder = tmp_path / "sub-001" / "func"
        folder.mkdir(parents=True)
        events = folder / "sub-001_task-x_events.tsv"
        events.write_text("onset\tduration\ttrial_type\n0.0\t1.0\tgo\n")
        (folder / "sub-001_task-x_events.json").write_text(
            json.dumps({"trial_type": {"Description": "what happened"}})
        )
        assert read_timing(events) is None

    def test_a_file_with_no_sidecar_is_not(self, tmp_path):
        folder = tmp_path / "sub-001" / "func"
        folder.mkdir(parents=True)
        lonely = folder / "sub-001_task-x_physio.tsv.gz"
        with gzip.open(lonely, "wt") as handle:
            handle.write("1\n2\n")
        assert read_timing(lonely) is None

    @pytest.mark.parametrize("meta", [
        {"Columns": ["a"], "SamplingFrequency": 0},
        {"Columns": ["a"], "SamplingFrequency": -5},
        {"Columns": ["a"], "SamplingFrequency": "fast"},
        {"Columns": [], "SamplingFrequency": 100},
        {"SamplingFrequency": 100},
    ])
    def test_a_sidecar_that_does_not_say_is_not(self, tmp_path, meta):
        path = _write(tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0]], meta)
        assert read_timing(path) is None

    def test_a_missing_start_time_is_zero_not_a_refusal(self, tmp_path):
        """StartTime is required by the standard, but a file without one is
        still plottable: the shape is what the reader came for."""
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0]],
            {"Columns": ["a"], "SamplingFrequency": 50},
        )
        assert read_timing(path)["start_time"] == 0.0


class TestReadingTheWholeFile:
    def test_it_reads_past_the_table_preview(self, tmp_path):
        """The table stops at five thousand rows because nobody reads more
        than that. A plot of the first five thousand samples of a long
        recording would be a picture of the first few seconds, drawn as
        though it were the whole thing, and silently."""
        rows = [[float(i)] for i in range(20000)]
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", rows,
            {"Columns": ["x"], "SamplingFrequency": 100.0, "StartTime": 0.0},
        )
        columns, total, step = read_columns(path)
        assert total == 20000
        # Every sample is KEPT. pyqtgraph decides what to draw at the
        # current zoom; throwing samples away here would mean zooming in
        # shows the same points stretched out rather than the detail.
        assert step == 1
        assert columns[0].size == 20000

    def test_an_enormous_file_strides_and_says_by_how_much(self, tmp_path):
        """Past the cap it is a memory decision, not a drawing one."""
        rows = [[float(i)] for i in range(20000)]
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", rows,
            {"Columns": ["x"], "SamplingFrequency": 100.0, "StartTime": 0.0},
        )
        columns, total, step = read_columns(path, limit=1000)
        assert total == 20000
        assert step > 1
        assert columns[0].size <= 1000
        # The step is what lets a position be turned back into a time:
        # point i of the strided column IS sample i * step.
        assert columns[0][3] == pytest.approx(3 * step)

    def test_a_short_file_is_kept_whole(self, recording):
        columns, total, step = read_columns(recording)
        assert step == 1
        assert total == 1000
        assert columns[0].size == 1000

    def test_an_unreadable_file_is_empty_not_an_exception(self, tmp_path):
        path = tmp_path / "nope.tsv.gz"
        assert read_columns(path) == ([], 0, 1)


class TestTheMaths:
    def test_blanks_and_text_become_gaps(self):
        """A gap in a recording is a fact about the recording, and closing
        it up would draw a signal that never happened."""
        series = to_series([["1", "a"], ["", "2"]], 2)
        assert series[0][0] == 1.0
        assert math.isnan(series[0][1])
        assert math.isnan(series[1][0])
        assert series[1][1] == 2.0

    def test_normalising_puts_a_channel_in_a_band(self):
        assert normalise(np.array([0.0, 10.0])).tolist() == [-0.5, 0.5]

    def test_a_flat_channel_stays_flat(self):
        """Dividing by a near-zero range would amplify it into noise."""
        assert normalise(np.array([3.0, 3.0, 3.0])).tolist() == [0.0, 0.0, 0.0]

    def test_normalising_keeps_the_gaps(self):
        out = normalise(np.array([0.0, float("nan"), 10.0]))
        assert math.isnan(float(out[1]))


class TestFiltering:
    def _two_tones(self, rate=200.0, seconds=8.0):
        t = np.arange(int(rate * seconds)) / rate
        slow = np.sin(2 * np.pi * 0.2 * t)
        fast = np.sin(2 * np.pi * 40.0 * t)
        return (slow + fast).astype(np.float32), rate

    def _power_at(self, values, rate, freq):
        freqs, power = welch_psd([np.asarray(values)], rate)
        return float(power[0][int(np.argmin(np.abs(freqs - freq)))])

    def test_no_filter_returns_the_signal_untouched(self):
        values, rate = self._two_tones()
        assert np.array_equal(filter_signal(values, rate), values)

    def test_a_high_pass_removes_the_slow_component(self):
        values, rate = self._two_tones()
        out = filter_signal(values, rate, high_pass=1.0)
        assert self._power_at(out, rate, 0.2) < self._power_at(values, rate, 0.2) / 100
        # and leaves the fast one where it was
        assert self._power_at(out, rate, 40.0) == pytest.approx(
            self._power_at(values, rate, 40.0), rel=0.3,
        )

    def test_a_low_pass_removes_the_fast_component(self):
        values, rate = self._two_tones()
        out = filter_signal(values, rate, low_pass=5.0)
        assert self._power_at(out, rate, 40.0) < self._power_at(values, rate, 40.0) / 100

    def test_a_notch_removes_one_band_and_leaves_its_neighbours(self):
        rate = 500.0
        t = np.arange(int(rate * 8)) / rate
        values = (np.sin(2 * np.pi * 50.0 * t)
                  + np.sin(2 * np.pi * 10.0 * t)).astype(np.float32)
        out = filter_signal(values, rate, notch=50.0)
        assert self._power_at(out, rate, 50.0) < self._power_at(values, rate, 50.0) / 100
        assert self._power_at(out, rate, 10.0) == pytest.approx(
            self._power_at(values, rate, 10.0), rel=0.2,
        )

    def test_a_cut_off_past_nyquist_is_skipped_not_refused(self):
        """A viewer that will not draw because one control is out of range
        is worse than one that draws the signal unfiltered."""
        values, rate = self._two_tones()
        out = filter_signal(values, rate, low_pass=rate)          # == 2x nyq
        assert np.array_equal(out, values)

    def test_a_high_pass_above_the_low_pass_keeps_the_high_pass(self):
        """A band that excludes everything is a slip, and drift removal is
        the likelier intent."""
        values, rate = self._two_tones()
        both = filter_signal(values, rate, high_pass=20.0, low_pass=1.0)
        alone = filter_signal(values, rate, high_pass=20.0)
        assert np.allclose(both, alone, equal_nan=True)

    def test_a_recording_too_short_to_filter_comes_back_whole(self):
        """Rather than an exception out of scipy's padding check."""
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.array_equal(filter_signal(values, 100.0, low_pass=10.0), values)

    def test_the_gaps_survive_the_filter(self):
        """An IIR filter fed a NaN returns NaN for every sample after it,
        so the gaps come out first and go back in afterwards."""
        values, rate = self._two_tones()
        values = values.copy()
        values[500:510] = np.nan
        out = filter_signal(values, rate, high_pass=1.0)
        assert np.all(np.isnan(out[500:510]))
        assert np.isfinite(out[:500]).all()
        assert np.isfinite(out[510:]).all()


class TestTheSpectrum:
    def test_a_tone_shows_up_at_its_frequency(self):
        rate = 200.0
        t = np.arange(int(rate * 10)) / rate
        values = np.sin(2 * np.pi * 7.0 * t).astype(np.float32)
        freqs, power = welch_psd([values], rate)
        assert freqs[int(np.argmax(power[0]))] == pytest.approx(7.0, abs=0.5)

    def test_gaps_do_not_make_it_nan(self):
        rate = 200.0
        values = np.sin(np.arange(2000) / 5.0).astype(np.float32)
        values[100:120] = np.nan
        _freqs, power = welch_psd([values], rate)
        assert np.isfinite(power).all()


class TestTheViewerOffersIt:
    def _pane(self, qtbot):
        from bidsmgr.gui.widgets.tsv_viewer_pane import TsvViewerPane

        pane = TsvViewerPane()
        qtbot.addWidget(pane)
        pane.resize(800, 600)
        return pane

    def _load(self, qtbot, pane, path, root):
        pane.set_file(path, root)
        qtbot.waitUntil(lambda: pane._loader is None, timeout=20_000)

    def _plotting(self, qtbot, pane):
        pane._plot_btn.setChecked(True)
        qtbot.waitUntil(lambda: not pane._plot_page._partial, timeout=20_000)
        return pane._plot_page

    def test_a_recording_gets_a_plot_button(self, qtbot, recording, tmp_path):
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        assert pane._plot_btn.isVisibleTo(pane)

    def test_an_events_table_does_not(self, qtbot, tmp_path):
        folder = tmp_path / "sub-001" / "func"
        folder.mkdir(parents=True)
        events = folder / "sub-001_task-x_events.tsv"
        events.write_text("onset\tduration\n0.0\t1.0\n")
        pane = self._pane(qtbot)
        self._load(qtbot, pane, events, tmp_path)
        assert not pane._plot_btn.isVisibleTo(pane)

    def test_pressing_it_shows_the_plot(self, qtbot, recording, tmp_path):
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        plot = self._plotting(qtbot, pane)
        assert pane._stack.currentWidget() is plot
        assert "2 of 2 channel(s)" in plot._caption.text()
        assert "1,000 samples" in plot._caption.text()

    def test_pressing_it_again_goes_back_to_the_table(
        self, qtbot, recording, tmp_path,
    ):
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        pane._plot_btn.setChecked(True)
        pane._plot_btn.setChecked(False)
        assert pane._stack.currentIndex() == 1

    def test_the_first_sample_is_not_eaten_by_a_header(
        self, qtbot, recording, tmp_path,
    ):
        """A physio TSV has no column names in it: they are in the sidecar.
        pandas reads the first SAMPLE as the header, and it belongs back in
        the data."""
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        assert len(pane._plot_rows) == 1000

    def test_unticking_a_channel_drops_its_curve(
        self, qtbot, recording, tmp_path,
    ):
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        plot = self._plotting(qtbot, pane)
        plot._boxes[0].setChecked(False)
        assert "1 of 2 channel(s)" in plot._caption.text()
        assert 0 not in plot._items

    def test_the_plot_does_not_pin_the_pane_wide(
        self, qtbot, recording, tmp_path,
    ):
        """It is a viewer in a splitter, like every other one."""
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        pane._plot_btn.setChecked(True)
        assert pane.minimumSizeHint().width() <= 260


class TestTheControls:
    def _plot(self, qtbot, recording, tmp_path):
        from bidsmgr.gui.widgets.tsv_viewer_pane import TsvViewerPane

        pane = TsvViewerPane()
        qtbot.addWidget(pane)
        pane.resize(800, 600)
        pane.set_file(recording, tmp_path)
        qtbot.waitUntil(lambda: pane._loader is None, timeout=20_000)
        pane._plot_btn.setChecked(True)
        qtbot.waitUntil(lambda: not pane._plot_page._partial, timeout=20_000)
        return pane, pane._plot_page

    def test_the_channel_names_label_the_axis(self, qtbot, recording, tmp_path):
        """A number means nothing once the signal has been normalised."""
        _pane, plot = self._plot(qtbot, recording, tmp_path)
        ticks = plot._plot.getPlotItem().getAxis("left")._tickLevels
        labels = [label for _pos, label in ticks[0]]
        assert labels == ["cardiac", "trigger"]

    def test_raw_values_drop_the_band_controls(self, qtbot, recording, tmp_path):
        """Height and spacing describe bands, and raw mode has none."""
        _pane, plot = self._plot(qtbot, recording, tmp_path)
        assert plot._gain_box.isEnabled()
        plot._raw_box.setChecked(True)
        assert not plot._gain_box.isEnabled()
        assert not plot._spacing_box.isEnabled()

    def test_spacing_moves_the_bands(self, qtbot, recording, tmp_path):
        _pane, plot = self._plot(qtbot, recording, tmp_path)
        before = plot._plot.getPlotItem().getAxis("left")._tickLevels[0]
        plot._spacing_box.setValue(3.0)
        after = plot._plot.getPlotItem().getAxis("left")._tickLevels[0]
        assert after[1][0] < before[1][0]      # the second band moved down

    def test_height_scales_the_trace(self, qtbot, recording, tmp_path):
        _pane, plot = self._plot(qtbot, recording, tmp_path)
        _x, before = plot._items[0].getData()
        plot._gain_box.setValue(4.0)
        _x, after = plot._items[0].getData()
        assert np.nanmax(np.abs(after)) > np.nanmax(np.abs(before)) * 3

    def test_the_filters_are_capped_at_nyquist(self, qtbot, recording, tmp_path):
        """Past it there is nothing to filter."""
        _pane, plot = self._plot(qtbot, recording, tmp_path)
        assert plot._hp_box.maximum() == pytest.approx(50.0)
        assert plot._lp_box.maximum() == pytest.approx(50.0)
        assert plot._notch_box.maximum() == pytest.approx(50.0)

    def test_filtering_changes_what_is_drawn(self, qtbot, recording, tmp_path):
        _pane, plot = self._plot(qtbot, recording, tmp_path)
        before = plot._view[0].copy()
        plot._hp_box.setValue(5.0)
        plot._refilter()                       # the debounce, skipped
        qtbot.waitUntil(
            lambda: not np.array_equal(plot._view[0], before), timeout=10_000,
        )
        assert "keeping above 5 Hz" in plot._caption.text()

    def test_turning_the_filters_off_restores_the_signal(
        self, qtbot, recording, tmp_path,
    ):
        _pane, plot = self._plot(qtbot, recording, tmp_path)
        original = plot._raw[0].copy()
        plot._hp_box.setValue(5.0)
        plot._refilter()
        qtbot.waitUntil(
            lambda: not np.array_equal(plot._view[0], original), timeout=10_000,
        )
        plot._hp_box.setValue(0.0)
        plot._refilter()
        assert np.array_equal(plot._view[0], original)

    def test_a_stale_filter_result_is_discarded(self, qtbot, recording, tmp_path):
        """The last request wins: the controls move faster than the work."""
        _pane, plot = self._plot(qtbot, recording, tmp_path)
        current = plot._view[0].copy()
        stale = [np.zeros_like(current)]
        plot._on_filtered((plot._path, plot._filter_token - 1, stale))
        assert np.array_equal(plot._view[0], current)

    def test_the_spectrum_opens_on_what_is_drawn(
        self, qtbot, recording, tmp_path,
    ):
        _pane, plot = self._plot(qtbot, recording, tmp_path)
        plot._boxes[1].setChecked(False)
        plot._show_psd()
        qtbot.waitUntil(lambda: plot._psd_dialog is not None, timeout=10_000)
        dialog = plot._psd_dialog
        qtbot.addWidget(dialog)
        assert dialog._ch_names == ["cardiac"]

    def test_a_theme_swap_reaches_the_plot(self, qtbot, recording, tmp_path):
        """pyqtgraph reads no QSS, so the palette has to be handed down.

        Without this the plot kept the old theme's background until the
        app was restarted and it happened to be built under the new one.
        """
        pane, plot = self._plot(qtbot, recording, tmp_path)
        pane.repaint_for_palette({"bg": "#ffffff", "border": "#000000",
                                  "muted": "#333333", "accent": "#ff0000"})
        brush = plot._plot.backgroundBrush().color().name()
        assert brush == "#ffffff"
