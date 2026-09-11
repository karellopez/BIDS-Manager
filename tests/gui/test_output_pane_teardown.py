"""A scan that comes back after the pane is gone must not raise.

`OutputFsPane` walks the output directory on the global thread pool and hands
the result back through a queued connection. The emit is therefore already
sitting in the event loop when the pane is torn down, and delivering it reaches
a Python wrapper whose C++ tree has been deleted.

This was showing up as a teardown error on an unrelated converter test, which
is the shape this class of bug always takes: it lands on whichever test happens
to run when the garbage collector gets round to the pane. The generation guard
already in the slot does not cover it, because the generation is still current.
The widget is just no longer there.

In normal use the pane outlives the window, so this is mostly a test-suite
symptom. It is still a real crash path: close a project while its output scan
is in flight and the callback lands on a destroyed tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6 import sip

from bidsmgr.gui.output_fs_pane import OutputFsPane

pytestmark = pytest.mark.gui


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    root = tmp_path / "out"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"\0" * 16)
    return root


def test_a_scan_landing_after_teardown_is_dropped(qtbot, output_dir) -> None:
    """The exact sequence, forced rather than waited for."""
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.set_root(output_dir)

    # Capture what the worker would hand back, then destroy the pane and
    # deliver it, which is what the queued connection does.
    generation = pane._scan_generation
    pane.deleteLater()
    pane.setParent(None)
    sip.delete(pane._tree)

    # Must return quietly rather than raising RuntimeError.
    pane._on_scan_done(generation, None)


def test_a_tree_that_dies_MID_render_is_survived(qtbot, output_dir) -> None:
    """The check at the top of the slot is necessary and not sufficient.

    The C++ tree is destroyed with its parent panel, and the parent can be
    collected by Python's cyclic collector, which runs on allocation, which
    means between any two statements in the slot. So the tree can be alive at
    the guard and gone three lines later. Forced here by deleting it from
    inside a call the slot makes.
    """
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.set_root(output_dir)
    qtbot.waitUntil(lambda: pane._tree.topLevelItemCount() > 0, timeout=4000)

    tree = pane._tree
    original = pane._clear_watcher

    def kill_it():
        original()
        sip.delete(tree)

    pane._clear_watcher = kill_it
    # Passes the top guard, then loses the tree partway through.
    pane._on_scan_done(pane._scan_generation, None)


def test_a_real_error_is_still_raised(qtbot, output_dir) -> None:
    """The catch must not turn a genuine bug into a silent no-op."""
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.set_root(output_dir)

    def boom(*_a, **_k):
        raise RuntimeError("something actually wrong")

    pane._render_scan = boom
    with pytest.raises(RuntimeError, match="something actually wrong"):
        pane._on_scan_done(pane._scan_generation, None)


def test_the_worker_does_not_emit_into_a_dead_pane(qtbot, output_dir) -> None:
    """The other half, one frame earlier.

    The slot guard only helps if the emit itself succeeds. When the pane dies
    mid-walk the worker emits on a deleted QObject, which raises on the WORKER
    thread where nothing catches it.
    """
    from bidsmgr.gui.output_fs_pane import _ScanRunnable, _ScanSignals

    holder = OutputFsPane()
    qtbot.addWidget(holder)
    signals = _ScanSignals(holder)
    runnable = _ScanRunnable(1, output_dir, signals)
    sip.delete(signals)

    runnable.run()   # must not raise


def test_a_real_scan_still_renders(qtbot, output_dir) -> None:
    """The guards must not swallow the ordinary case."""
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.set_root(output_dir)
    qtbot.waitUntil(lambda: pane._tree.topLevelItemCount() > 0, timeout=4000)
    root_item = pane._tree.topLevelItem(0)
    assert root_item.text(0).startswith("out")
