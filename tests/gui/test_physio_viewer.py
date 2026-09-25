"""A physio recording opened as a signal, in the viewer every signal uses.

A ``*_physio.tsv.gz`` is a table of numbers with no header row, and reading
one as a table answers nothing about the shape of the trace. The three facts
a viewer needs are in the sidecar, which is also what distinguishes a
continuous recording from a table of onsets, so that is what decides whether
a viewer is offered at all.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.gui.widgets.physio_viewer import (
    build_raw,
    guess_channel_type,
    read_columns,
    read_timing,
    sidecar_for,
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


class TestWhatCountsAsARecording:
    def test_a_recording_is(self, recording):
        assert read_timing(recording) == {
            "columns": ["cardiac", "trigger"],
            "sampling_frequency": 100.0,
            "start_time": -3.5,
            "units": "",
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
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0]],
            {"Columns": ["a"], "SamplingFrequency": 50},
        )
        assert read_timing(path)["start_time"] == 0.0

    def test_the_sidecar_is_found_for_both_extensions(self, tmp_path):
        assert sidecar_for(Path("a/b_physio.tsv.gz")).name == "b_physio.json"
        assert sidecar_for(Path("a/b_physio.tsv")).name == "b_physio.json"


class TestReadingTheWholeFile:
    def test_it_reads_past_the_table_preview(self, tmp_path):
        """The table stops at five thousand rows. A view of the first five
        thousand samples of a long recording would be a picture of its first
        few seconds, drawn as though it were the whole thing."""
        rows = [[float(i)] for i in range(20000)]
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", rows,
            {"Columns": ["x"], "SamplingFrequency": 100.0, "StartTime": 0.0},
        )
        columns, total, step = read_columns(path)
        assert (total, step, columns[0].size) == (20000, 1, 20000)

    def test_an_enormous_file_strides_and_says_by_how_much(self, tmp_path):
        rows = [[float(i)] for i in range(20000)]
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", rows,
            {"Columns": ["x"], "SamplingFrequency": 100.0, "StartTime": 0.0},
        )
        columns, total, step = read_columns(path, limit=1000)
        assert total == 20000 and step > 1 and columns[0].size <= 1000
        # Point i of the strided column IS sample i * step.
        assert columns[0][3] == pytest.approx(3 * step)

    def test_an_unreadable_file_is_empty_not_an_exception(self, tmp_path):
        assert read_columns(tmp_path / "nope.tsv.gz") == ([], 0, 1)


class TestChannelTyping:
    """What makes the shared viewer USEFUL here rather than merely possible.

    The viewer colours by kind, groups the type filter by kind, averages the
    spectrum per kind and reads events off a stim channel. All of that comes
    free once the columns carry MNE's own channel types.
    """

    @pytest.mark.parametrize("name,want", [
        ("cardiac", "ecg"), ("ECG", "ecg"), ("pulse", "ecg"),
        ("respiratory", "resp"), ("resp", "resp"), ("breath_belt", "resp"),
        ("trigger", "stim"), ("external_trigger", "stim"), ("TTL", "stim"),
        ("x_coordinate", "eyegaze"), ("pupil_size", "pupil"),
        ("temperature", "temperature"), ("gsr", "gsr"),
    ])
    def test_a_column_name_implies_its_kind(self, name, want):
        assert guess_channel_type(name) == want

    def test_an_unrecognised_column_is_misc_not_a_guess(self):
        """``misc`` is MNE for "shown, not interpreted", which is exactly
        the right claim about a column we did not recognise."""
        assert guess_channel_type("weird_column") == "misc"

    def test_a_substring_match_does_not_fire_on_the_wrong_word(self):
        """``discard`` contains ``card``, and is not a cardiac trace."""
        assert guess_channel_type("discard") == "misc"


class TestBuildingTheRaw:
    def test_channels_rate_and_types_survive(self, recording):
        timing = read_timing(recording)
        columns, _total, step = read_columns(recording)
        raw = build_raw(columns, timing, step)
        assert raw.ch_names == ["cardiac", "trigger"]
        assert raw.get_channel_types() == ["ecg", "stim"]
        assert raw.info["sfreq"] == 100.0
        assert raw.n_times == 1000

    def test_a_strided_read_slows_the_rate_to_match(self, recording):
        """A strided recording really IS a slower one, and every frequency
        the viewer computes depends on getting that right."""
        timing = read_timing(recording)
        columns, _total, _step = read_columns(recording)
        raw = build_raw(columns, timing, step=4)
        assert raw.info["sfreq"] == 25.0

    def test_duplicate_column_names_are_disambiguated(self, tmp_path):
        """MNE refuses duplicates and a sidecar is free to repeat a name."""
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0, 2.0]],
            {"Columns": ["ecg", "ecg"], "SamplingFrequency": 50},
        )
        columns, _t, step = read_columns(path)
        raw = build_raw(columns, read_timing(path), step)
        assert raw.ch_names == ["ecg", "ecg-1"]

    def test_gaps_become_zero_so_the_filters_work(self, tmp_path):
        """A loss worth naming. MNE cannot carry NaN through a filter or an
        FFT, so a viewer built on MNE must choose between the gap and the
        filters; the table view still shows the blank."""
        path = tmp_path / "sub-001" / "func" / "sub-001_physio.tsv.gz"
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt") as handle:
            handle.write("1\nn/a\n3\n")
        (path.parent / "sub-001_physio.json").write_text(
            json.dumps({"Columns": ["ecg"], "SamplingFrequency": 10})
        )
        columns, _t, step = read_columns(path)
        assert math.isnan(float(columns[0][1]))
        raw = build_raw(columns, read_timing(path), step)
        assert np.isfinite(raw.get_data()).all()

    def test_no_columns_is_refused_not_guessed(self):
        with pytest.raises(ValueError):
            build_raw([], {"columns": [], "sampling_frequency": 100.0})


class TestTheViewerOffersIt:
    def _pane(self, qtbot):
        from bidsmgr.gui.widgets.tsv_viewer_pane import TsvViewerPane

        pane = TsvViewerPane()
        qtbot.addWidget(pane)
        pane.resize(900, 600)
        return pane

    def _load(self, qtbot, pane, path, root):
        pane.set_file(path, root)
        qtbot.waitUntil(lambda: pane._loader is None, timeout=20_000)

    def test_a_recording_gets_a_visualize_button(self, qtbot, recording, tmp_path):
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        assert pane._visualize_btn.isVisibleTo(pane)
        assert pane._visualize_btn.text().strip() == "Visualize"

    def test_an_events_table_does_not(self, qtbot, tmp_path):
        folder = tmp_path / "sub-001" / "func"
        folder.mkdir(parents=True)
        events = folder / "sub-001_task-x_events.tsv"
        events.write_text("onset\tduration\n0.0\t1.0\n")
        pane = self._pane(qtbot)
        self._load(qtbot, pane, events, tmp_path)
        assert not pane._visualize_btn.isVisibleTo(pane)

    def test_pressing_it_opens_the_shared_viewer(self, qtbot, recording, tmp_path):
        from bidsmgr.gui.widgets.time_series_view import TimeSeriesView

        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        pane._visualize_btn.setChecked(True)
        qtbot.waitUntil(lambda: pane._signal_worker is None, timeout=20_000)
        assert isinstance(pane._viewer_page, TimeSeriesView)
        assert pane._stack.currentWidget() is pane._viewer_page
        assert pane._viewer_page._ch_names == ["cardiac", "trigger"]

    def test_pressing_it_again_goes_back_to_the_table(
        self, qtbot, recording, tmp_path,
    ):
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        pane._visualize_btn.setChecked(True)
        qtbot.waitUntil(lambda: pane._signal_worker is None, timeout=20_000)
        pane._visualize_btn.setChecked(False)
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

    def test_the_viewer_does_not_pin_the_pane_wide(
        self, qtbot, recording, tmp_path,
    ):
        """It is a viewer in a splitter, like every other one."""
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        pane._visualize_btn.setChecked(True)
        qtbot.waitUntil(lambda: pane._signal_worker is None, timeout=20_000)
        assert pane.minimumSizeHint().width() <= 400
