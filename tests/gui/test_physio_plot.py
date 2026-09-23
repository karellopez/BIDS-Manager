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

import pytest

from bidsmgr.gui.widgets.physio_plot import (
    decimate,
    normalise,
    read_columns,
    read_timing,
    to_series,
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
        }

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
        _columns, total, _step = read_columns(path)
        assert total == 20000

    def test_it_decimates_and_says_by_how_much(self, tmp_path):
        rows = [[float(i)] for i in range(20000)]
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", rows,
            {"Columns": ["x"], "SamplingFrequency": 100.0, "StartTime": 0.0},
        )
        columns, total, step = read_columns(path, limit=1000)
        assert len(columns[0]) <= 1001
        assert step > 1
        # The step is what lets a position be turned back into a time:
        # point i of the decimated column IS sample i * step.
        assert columns[0][3] == pytest.approx(3 * step)
        assert total == 20000

    def test_a_short_file_is_not_decimated(self, recording):
        columns, total, step = read_columns(recording)
        assert step == 1
        assert total == 1000
        assert len(columns[0]) == 1000

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
        out = normalise([0.0, 10.0])
        assert out == [-0.5, 0.5]

    def test_a_flat_channel_stays_flat(self):
        """Dividing by a near-zero range would amplify it into noise."""
        assert normalise([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]

    def test_normalising_keeps_the_gaps(self):
        out = normalise([0.0, float("nan"), 10.0])
        assert math.isnan(out[1])

    def test_decimate_keeps_the_positions(self):
        positions, values = decimate(list(range(100)), limit=10)
        assert values == [float(p) if isinstance(p, float) else p
                          for p in positions]
        assert len(values) <= 11


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
        pane._plot_btn.setChecked(True)
        assert pane._stack.currentWidget() is pane._plot_page
        qtbot.waitUntil(
            lambda: not pane._plot_page._partial, timeout=20_000
        )
        assert "2 of 2 channel(s)" in pane._plot_page._caption.text()
        assert "1,000 samples" in pane._plot_page._caption.text()

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
        pane._plot_btn.setChecked(True)
        plot = pane._plot_page
        plot._boxes[0].setChecked(False)
        assert "1 of 2 channel(s)" in plot._caption.text()

    def test_the_plot_does_not_pin_the_pane_wide(
        self, qtbot, recording, tmp_path,
    ):
        """It is a viewer in a splitter, like every other one."""
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        pane._plot_btn.setChecked(True)
        assert pane.minimumSizeHint().width() <= 260
