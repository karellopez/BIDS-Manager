"""Tests for the collapsible / detachable ``PanelFrame`` wrapper."""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QLabel, QWidget

from bidsmgr.gui.widgets import PanelFrame
from bidsmgr.gui.widgets.primitives import PaneHeader

pytestmark = pytest.mark.gui


def _pane_with_header() -> QWidget:
    from PyQt6.QtWidgets import QVBoxLayout
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.addWidget(PaneHeader("Inner title"))
    lay.addWidget(QLabel("body"))
    return w


def test_collapse_toggles_body_visibility(qtbot) -> None:
    inner = QLabel("content")
    frame = PanelFrame(inner, "Demo", edge="top")
    qtbot.addWidget(frame)
    frame.show()
    assert not frame.is_collapsed()
    assert inner.isVisible()

    frame.toggle_collapsed()
    assert frame.is_collapsed()
    assert not inner.isVisible()

    frame.toggle_collapsed()
    assert not frame.is_collapsed()
    assert inner.isVisible()


def test_non_collapsible_frame_ignores_collapse(qtbot) -> None:
    frame = PanelFrame(QLabel("x"), "Inspection", collapsible=False)
    qtbot.addWidget(frame)
    frame.set_collapsed(True)
    assert not frame.is_collapsed()  # no-op


def test_inner_pane_header_is_hidden(qtbot) -> None:
    pane = _pane_with_header()
    frame = PanelFrame(pane, "Outer title")
    qtbot.addWidget(frame)
    hdr = pane.findChild(PaneHeader)
    assert hdr is not None
    assert not hdr.isVisible()  # PanelFrame shows the single title instead


def test_detach_then_reattach_restores_parent(qtbot) -> None:
    inner = QLabel("content")
    frame = PanelFrame(inner, "Demo")
    qtbot.addWidget(frame)
    frame.show()

    assert inner.parent() is frame
    frame.detach()
    assert frame.is_detached()
    # Body moved into the floating window, not the frame.
    assert inner.parent() is not frame

    frame.reattach()
    assert not frame.is_detached()
    assert inner.parent() is frame
    assert inner.isVisible()


def test_detach_expands_a_collapsed_frame(qtbot) -> None:
    inner = QLabel("content")
    frame = PanelFrame(inner, "Demo", edge="left")
    qtbot.addWidget(frame)
    frame.show()
    frame.set_collapsed(True)
    assert frame.is_collapsed()
    frame.detach()
    # Detaching a collapsed frame expands it first so the body is usable.
    assert not frame.is_collapsed()
    assert frame.is_detached()
    frame.reattach()
