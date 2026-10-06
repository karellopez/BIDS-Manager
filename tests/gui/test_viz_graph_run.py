"""The BOLD graph with the run around it: events and physio on its time axis."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("pyqtgraph")

from bidsmgr.gui.viz import Viewer  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """A 20-volume run at TR 2 s (40 s), two events and a cardiac trace
    that starts 3 s before the first volume."""
    root = tmp_path / "Study"
    func = root / "sub-01" / "func"
    func.mkdir(parents=True)
    rng = np.random.default_rng(0)
    data = (1000 + rng.normal(0, 5, (6, 6, 4, 20))).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    img.header.set_zooms((1.0, 1.0, 1.0, 2.0))
    nib.save(img, str(func / "sub-01_task-x_run-1_bold.nii.gz"))
    (func / "sub-01_task-x_run-1_bold.json").write_text(json.dumps({"RepetitionTime": 2.0}))
    (func / "sub-01_task-x_run-1_events.tsv").write_text(
        "onset\tduration\ttrial_type\n4\t6\tfaces\n20\t0\tbutton\n")
    with gzip.open(func / "sub-01_task-x_run-1_recording-cardiac_physio.tsv.gz", "wt") as fh:
        for i in range(4600):           # 46 s at 100 Hz
            fh.write(f"{math.sin(i / 7.0):g}\n")
    (func / "sub-01_task-x_run-1_recording-cardiac_physio.json").write_text(json.dumps(
        {"Columns": ["cardiac"], "SamplingFrequency": 100.0, "StartTime": -3.0}))
    return root


def _bold(root: Path) -> Path:
    return root / "sub-01" / "func" / "sub-01_task-x_run-1_bold.nii.gz"


def _graph(qtbot, root: Path, path: Path):
    v = Viewer(kind="volume")
    qtbot.addWidget(v)
    v.resize(1000, 700)
    v.show()
    qtbot.waitExposed(v)
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(path, root)
    v.run("view.graph", value=True)
    v.run("graph.set", scope=1, x_axis="seconds")
    v.qstore.flush()
    qtbot.waitUntil(lambda: v.presenter.graph is not None, timeout=5000)
    return v, v.presenter.graph


class TestEvents:
    def test_they_are_drawn_on_the_time_axis(self, qtbot, run):
        _v, g = _graph(qtbot, run, _bold(run))
        bands = g.event_bands()
        assert [(round(a, 3), lab) for a, _b, lab in bands] == [(4.0, "faces"), (20.0, "button")]
        assert bands[0][1] == pytest.approx(10.0), "a span for an event that lasts"
        assert bands[1][1] > 20.0, "an instant still a visible band"

    def test_against_volumes_they_are_placed_by_the_repetition_time(self, qtbot, run):
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", x_axis="frames")
        v.qstore.flush()
        assert g.event_bands()[0][0] == pytest.approx(2.0), "4 s at TR 2 s is volume 2"

    def test_they_can_be_turned_off(self, qtbot, run):
        v, g = _graph(qtbot, run, _bold(run))
        assert g.events_box.isVisibleTo(g)
        v.run("graph.set", events=False)
        v.qstore.flush()
        assert g.event_bands() == []
        assert not g.events_box.isChecked()

    def test_a_neighbourhood_carries_them_in_every_cell(self, qtbot, run):
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", scope=2)
        v.qstore.flush()
        assert g.cell_count() == 9
        assert len(g.event_bands()) == 2

    def test_a_run_without_events_offers_none(self, qtbot, run):
        (run / "sub-01" / "func" / "sub-01_task-x_run-1_events.tsv").unlink()
        _v, g = _graph(qtbot, run, _bold(run))
        assert not g.events_box.isVisibleTo(g)
        assert g.event_bands() == []


class TestPhysio:
    def test_it_is_drawn_under_the_graph_in_run_time(self, qtbot, run):
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", physio=True)
        v.qstore.flush()
        qtbot.waitUntil(lambda: g.physio_channels() == ["cardiac"], timeout=20_000)
        x, _y = g._physio_curves[0].getData()
        assert x.min() >= 0.0 - 1e-6, "cut to the run (it started 3 s before it)"
        assert x.max() <= 38.0 + 0.02

    def test_it_is_shown_for_a_neighbourhood_too(self, qtbot, run):
        """It used to vanish at any scope but one voxel, with its box still
        ticked: the strip keeps the run's own time axis."""
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", physio=True, scope=2)
        v.qstore.flush()
        qtbot.waitUntil(lambda: g.physio_channels() == ["cardiac"], timeout=20_000)

    def test_a_trigger_log_is_drawn_as_ticks_with_its_count(self, qtbot, run):
        """A scanner log writes the trigger's value where it fired and
        nothing in between: a lane of ticks, not a line of dots."""
        import gzip
        import json

        func = run / "sub-01" / "func"
        rows = ["n/a"] * 400
        for k in range(0, 400, 40):
            rows[k] = "5"
        with gzip.open(func / "sub-01_task-x_run-1_recording-trigger_physio.tsv.gz", "wt") as fh:
            fh.write("\n".join(rows) + "\n")
        (func / "sub-01_task-x_run-1_recording-trigger_physio.json").write_text(json.dumps(
            {"Columns": ["trigger"], "SamplingFrequency": 10.0, "StartTime": 0.0}))
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", physio=True)
        v.qstore.flush()
        qtbot.waitUntil(lambda: len(g.physio_lanes()) == 2, timeout=20_000)
        lanes = {lane["name"]: lane for lane in g.physio_lanes()}
        assert lanes["cardiac"]["role"] == "waveform"
        assert lanes["trigger"]["role"] == "events"
        assert lanes["trigger"]["ticks"] == 10
        assert "10 marks" in lanes["trigger"]["note"]

    def test_moving_the_crosshair_does_not_read_it_again(self, qtbot, run, monkeypatch):
        from bidsmgr.gui.viz.canvases import graph as G

        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", physio=True)
        v.qstore.flush()
        qtbot.waitUntil(lambda: g.physio_channels() == ["cardiac"], timeout=20_000)
        reads = []
        real = G.physio_for_graph
        monkeypatch.setattr(G, "physio_for_graph", lambda *a, **k: (reads.append(1), real(*a, **k))[1])
        for _ in range(5):
            v.run("cursor.step", plane="axial", n=1)
            v.qstore.flush()
        assert reads == []


class TestDiffusion:
    @pytest.fixture
    def dwi(self, tmp_path: Path) -> Path:
        root = tmp_path / "Study"
        d = root / "sub-01" / "dwi"
        d.mkdir(parents=True)
        data = np.random.default_rng(1).random((4, 4, 3, 7)).astype(np.float32) * 100
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(d / "sub-01_dwi.nii.gz"))
        (d / "sub-01_dwi.bval").write_text("5 995 1000 1005 0 2000 1995\n")
        return root

    def _open(self, qtbot, root):
        path = root / "sub-01" / "dwi" / "sub-01_dwi.nii.gz"
        v = Viewer(kind="volume")
        qtbot.addWidget(v)
        v.resize(1000, 700)
        v.show()
        qtbot.waitExposed(v)
        with qtbot.waitSignal(v.loaded, timeout=20_000):
            v.set_file(path, root)
        v.qstore.flush()
        return v

    def test_the_volume_label_says_its_b_value(self, qtbot, dwi):
        v = self._open(qtbot, dwi)
        v.run("frame.set", frame=2)
        v.qstore.flush()
        assert v.presenter.frame_control.value() == 2
        assert v.presenter.frame_control.spin.suffix() == " / 6"
        assert v.presenter._frame_hdr.text() == "Volume  b=1000"

    def test_the_shell_keys_step_within_a_shell(self, qtbot, dwi):
        v = self._open(qtbot, dwi)
        assert v.action("frame.same_shell").isEnabled()
        assert v.action("frame.same_shell").shortcut().toString() == "]"
        v.run("frame.set", frame=1)
        v.trigger("frame.same_shell")
        v.qstore.flush()
        assert v.scene.layers[0].frame == 2
        v.trigger("frame.shell")
        v.qstore.flush()
        assert v.scene.layers[0].frame == 5

    def test_the_graph_marks_the_b0_volumes(self, qtbot, dwi):
        v = self._open(qtbot, dwi)
        v.run("view.graph", value=True)
        v.run("graph.set", scope=1, x_axis="frames")
        v.qstore.flush()
        qtbot.waitUntil(lambda: v.presenter.graph is not None, timeout=5000)
        bands = v.presenter.graph.event_bands()
        assert [(round(a + 0.5), lab) for a, _b, lab in bands] == [(0, "b=0"), (4, "b=0")]

    def test_a_series_without_b_values_has_no_shell_actions(self, qtbot, run):
        v, _g = _graph(qtbot, run, _bold(run))
        assert not v.action("frame.same_shell").isEnabled()


class TestQcTracesAndZoom:
    def test_qc_traces_flag_the_volume_that_jumps(self, qtbot, run):
        import numpy as _np

        bold = _bold(run)
        img = nib.load(str(bold))
        data = _np.asarray(img.dataobj).astype("float32")
        data[..., 11] += 400.0
        nib.save(nib.Nifti1Image(data, img.affine), str(bold))
        v, g = _graph(qtbot, run, bold)
        qtbot.waitUntil(lambda: g.qc_box.isVisibleTo(g), timeout=20_000)
        v.run("graph.set", qc=True)
        v.qstore.flush()
        qtbot.waitUntil(lambda: any(lane["name"] == "DVARS jumps" for lane in g.physio_lanes()),
                        timeout=20_000)
        jumps = [lane for lane in g.physio_lanes() if lane["name"] == "DVARS jumps"][0]
        assert jumps["ticks"] >= 1 and "box-plot" in jumps["note"]
        assert {lane["name"] for lane in g.physio_lanes()} >= {"global signal", "DVARS"}

    def test_ctrl_wheel_zooms_time_and_reads_the_physio_for_it(self, qtbot, run):
        from PyQt6.QtCore import QPoint, QPointF, Qt
        from PyQt6.QtGui import QWheelEvent

        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", physio=True)
        v.qstore.flush()
        qtbot.waitUntil(lambda: g.physio_channels() == ["cardiac"], timeout=20_000)
        g.set_time_view((10.0, 20.0))
        assert g.time_view() == (10.0, 20.0)
        assert g.plot.getPlotItem().getViewBox().viewRange()[0] == pytest.approx([10.0, 20.0])
        qtbot.waitUntil(lambda: g._physio_src is not None
                        and float(g._physio_src["lanes"][0]["x"].min()) >= 9.9, timeout=20_000)
        ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.ShiftModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        g._on_wheel(ev)
        lo, hi = g.time_view()
        assert hi - lo == pytest.approx(10.0) and lo < 10.0, "Shift+wheel pans"
        g.set_time_view(None)
        assert g.time_view() is None
