"""The guide: a switch left of Settings, and one ringed control at a time.

Pinned here, in a full main window:

- the switch sits in the header just left of Settings, is off at first, and
  is remembered;
- the steps follow the window: where to save, name, create; the raw data,
  scan, check the table (only once the USER has clicked a row, not the row
  selected when the table loads), convert, then the Editor; validate, fix;
- the ring is around the step's control and the caption never covers it,
  stays inside the window, and keeps clear of what its text points to;
- the overlay is a child of the window, never a window of its own, never
  takes a click, and hides while a dialog is open;
- a theme and a font-size change re-measure it.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest
from PyQt6.QtCore import QEvent, QObject, QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog, QFrame, QPushButton

pytestmark = pytest.mark.gui


def _rows(n: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "participant_id": "sub-001", "session": "", "include": 1,
            "modality": "mri", "datatype": "func", "bids_guess_suffix": "bold",
            "bids_name": f"sub-001_task-t{i}_run-1_bold",
            "bids_guess_confidence": "0.9", "bids_guess_skip": False, "issues": "",
            "sequence": f"bold_{i}", "series_uid": f"1.{i}", "dataset": "ds",
            "source_file": "", "source_folder": "s", "acq_time": f"10:{i:02d}:00",
            "task": f"t{i}", "run": "1",
            "entities": json.dumps({"subject": "001", "task": f"t{i}", "run": "1"}),
        })
    return pd.DataFrame(rows)


class _WindowSpy(QObject):
    """Records every window shown."""

    def __init__(self) -> None:
        super().__init__()
        self.shown: list[str] = []

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        if event.type() == QEvent.Type.Show and obj.isWidgetType() and obj.isWindow():
            self.shown.append(f"{type(obj).__name__}#{obj.objectName()}")
        return False


def _settle(qtbot, ms: int = 150) -> None:
    end = time.time() + ms / 1000
    while time.time() < end:
        qtbot.wait(10)


@pytest.fixture
def window(qtbot, qapp, monkeypatch, tmp_path):
    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager

    picks = {"dir": str(tmp_path)}
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: picks["dir"]))
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qtbot.addWidget(win)
    win.resize(1400, 900)
    win.show()
    qtbot.waitExposed(win)
    win.picks = picks
    win.theme_manager = theme
    return win


def _guide_on(qtbot, win) -> None:
    win._header.guide_switch.click()
    _settle(qtbot)


def _step(win):
    win.guide.refresh()
    step = win.guide.overlay.step
    return step.key if step is not None else None


def _rect_of(win, widget):
    return win.guide._rect_in_host(widget)


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


def test_the_switch_sits_left_of_settings_and_is_remembered(qtbot, window) -> None:
    from bidsmgr.gui.app_settings import AppSettings

    header = window._header
    switch = header.guide_switch
    assert switch.isVisible() and not switch.isChecked()
    assert not window.guide.enabled and not window.guide.overlay.isVisible()
    gap = header._settings_btn.geometry().left() - switch.geometry().right()
    assert 0 < gap < 20, "just left of Settings"
    _guide_on(qtbot, window)
    assert window.guide.enabled and AppSettings.load().guided_mode
    switch.click()
    assert not window.guide.enabled and not window.guide.overlay.isVisible()
    assert AppSettings.load().guided_mode is False


def test_a_window_opens_with_the_guide_on_when_it_was_left_on(qtbot, qapp) -> None:
    from bidsmgr.gui.app_settings import AppSettings
    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager

    AppSettings.remember_guided_mode(True)
    win = MainWindow(ThemeManager(qapp))
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    assert win._header.guide_switch.isChecked() and win.guide.enabled
    _settle(qtbot)
    assert win.guide.overlay.isVisible() and win.guide.overlay.step.key == "home.location"


# ---------------------------------------------------------------------------
# The paths
# ---------------------------------------------------------------------------


def test_home_where_name_create(qtbot, window) -> None:
    _guide_on(qtbot, window)
    welcome = window.welcome
    assert _step(window) == "home.location"
    assert window.guide.overlay.target_rect == _rect_of(window, welcome._browse_btn)
    welcome._browse_btn.click()
    assert _step(window) == "home.name"
    QTest.keyClicks(welcome._name_edit, "Study_A")
    assert _step(window) == "home.create"
    assert window.guide.overlay.target_rect == _rect_of(window, welcome._create_btn)


def test_the_converter_path(qtbot, window, tmp_path) -> None:
    _guide_on(qtbot, window)
    window._apply_active_view("converter", persist=False)
    assert _step(window) == "convert.no_project"
    window.welcome.create_project(tmp_path, "Study_A")
    _settle(qtbot)
    conv = window.converter
    assert _step(window) == "convert.raw"
    raw = tmp_path / "raw"
    raw.mkdir()
    window.picks["dir"] = str(raw)
    conv._raw_pathbar.change_button.click()
    assert _step(window) == "convert.scan"
    conv.load_inventory(_rows())
    _settle(qtbot)
    # A row is selected for the user when the table loads; that is not them
    # checking it.
    assert _step(window) == "convert.review"
    target = window.guide.overlay.target_rect
    assert target == _rect_of(window, conv._table)
    table = conv._table
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                     pos=table.visualRect(table.model().index(1, 2)).center())
    assert _step(window) == "convert.run"
    conv.convert_finished.emit(0, tmp_path)
    _settle(qtbot)
    assert _step(window) == "convert.done"
    assert window.guide.overlay.target_rect == _rect_of(window, window._header._editor_btn)
    # A new scan starts the path again from its table.
    conv.scan_finished.emit(_rows(), None)
    assert _step(window) != "convert.done"


def test_the_editor_path(qtbot, window, tmp_path) -> None:
    from bidsmgr.editor.types import ValidationReport

    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps(
        {"Name": "ds", "BIDSVersion": "1.10.0"}))
    window.welcome.open_project(root)
    _guide_on(qtbot, window)
    window._apply_active_view("editor", persist=False)
    _settle(qtbot)
    editor = window.editor
    editor._report_worker = None
    # A converted dataset offers several things to do: the guide asks first.
    window.guide.refresh()
    assert window.guide.question.isVisible()
    window.guide.question.dismissed.emit()
    _settle(qtbot, 60)
    assert _step(window) == "editor.validate"
    editor._report = ValidationReport(bids_root=root, counts={"ok": 1, "warn": 0, "err": 2})
    assert _step(window) == "editor.fix"
    assert window.guide.overlay.target_rect == _rect_of(window, editor._validation_frame)
    # The whole Validation card, not its first row of controls.
    assert window.guide.overlay.target_rect.height() > 300
    editor._report = ValidationReport(bids_root=root, counts={"ok": 3, "warn": 0, "err": 0})
    assert _step(window) == "editor.valid" and window.guide.overlay.step.done


# ---------------------------------------------------------------------------
# The overlay
# ---------------------------------------------------------------------------


def test_the_caption_never_covers_the_control_and_stays_in_the_window(
        qtbot, window, tmp_path) -> None:
    _guide_on(qtbot, window)
    overlay = window.guide.overlay
    welcome = window.welcome
    for act in (lambda: None, welcome._browse_btn.click,
                lambda: QTest.keyClicks(welcome._name_edit, "S")):
        act()
        window.guide.refresh()
        bubble, target = overlay.bubble_rect, overlay.target_rect
        assert not bubble.intersects(target)
        assert overlay.rect().contains(bubble.toAlignedRect())
    window.welcome.create_project(tmp_path, "Study_B")
    window._apply_active_view("converter", persist=False)
    (tmp_path / "raw").mkdir()
    window.picks["dir"] = str(tmp_path / "raw")
    window.converter._raw_pathbar.change_button.click()
    window.converter.load_inventory(_rows())
    _settle(qtbot)
    assert _step(window) == "convert.review"
    properties = _rect_of(window, window.converter._properties)
    assert not overlay.bubble_rect.toAlignedRect().intersects(properties), \
        "the caption sends the user to Properties, so it does not cover it"


def test_the_overlay_is_never_a_window_and_never_takes_a_click(qtbot, window) -> None:
    spy = _WindowSpy()
    QApplication.instance().installEventFilter(spy)
    try:
        _guide_on(qtbot, window)
        window.welcome._browse_btn.click()
        _settle(qtbot)
    finally:
        QApplication.instance().removeEventFilter(spy)
    overlay = window.guide.overlay
    assert not overlay.isWindow() and overlay.parentWidget() is window.centralWidget()
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert not [w for w in spy.shown if "Guide" in w], spy.shown
    # A click on the ringed control reaches it through the overlay.
    name = window.welcome._name_edit
    centre = name.mapTo(window, QPoint(name.width() // 2, name.height() // 2))
    assert window.childAt(centre) is not overlay


def test_hidden_while_a_dialog_is_open(qtbot, window) -> None:
    _guide_on(qtbot, window)
    assert window.guide.overlay.isVisible()
    dialog = QDialog(window)
    qtbot.addWidget(dialog)
    dialog.setModal(True)
    dialog.show()
    _settle(qtbot)
    window.guide.refresh()
    assert not window.guide.overlay.isVisible()
    dialog.close()
    _settle(qtbot)
    window.guide.refresh()
    assert window.guide.overlay.isVisible()


def test_a_theme_and_font_size_change_re_measure_it(qtbot, window) -> None:
    _guide_on(qtbot, window)
    overlay = window.guide.overlay
    before = overlay.bubble_rect
    switch_before = window._header.guide_switch.sizeHint()
    window.theme_manager.set_font_scale(1.3)
    _settle(qtbot)
    window.guide.refresh()
    assert overlay.bubble_rect.height() > before.height()
    assert window._header.guide_switch.sizeHint().width() > switch_before.width()
    for theme in ("light", "hc-dark"):
        window.theme_manager.apply(theme)
        _settle(qtbot, 60)
        window.guide.refresh()
        assert overlay.isVisible()
    window.theme_manager.set_font_scale(1.0)


def test_a_finished_path_fades(qtbot, window, monkeypatch, tmp_path) -> None:
    from bidsmgr.gui import guide

    monkeypatch.setattr(guide, "DONE_LINGER_MS", 50)
    _guide_on(qtbot, window)
    window.welcome.create_project(tmp_path, "Study_C")
    window._apply_active_view("converter", persist=False)
    overlay = window.guide.overlay
    overlay._linger.setInterval(50)
    window.converter.convert_finished.emit(0, tmp_path)
    _settle(qtbot, 900)
    window.guide.refresh()
    assert overlay.step.done and overlay._opacity == 0.0


# ---------------------------------------------------------------------------
# A new raw folder; the question when coming back to a view
# ---------------------------------------------------------------------------


def _with_a_table(qtbot, window, tmp_path):
    window.welcome.create_project(tmp_path, "Study_Q")
    window._apply_active_view("converter", persist=False)
    raw = tmp_path / "raw"
    raw.mkdir()
    window.picks["dir"] = str(raw)
    window.converter._raw_pathbar.change_button.click()
    window.converter.load_inventory(_rows())
    _settle(qtbot)


def _come_back_to_the_converter(qtbot, window) -> None:
    window._apply_active_view("welcome", persist=False)
    window.guide.refresh()
    window.guide.question.dismissed.emit()       # Home asks too, with a project open
    _settle(qtbot, 60)
    window._apply_active_view("converter", persist=False)
    window.guide.refresh()
    _settle(qtbot, 60)


def test_a_new_raw_folder_leads_back_to_scan(qtbot, window, tmp_path) -> None:
    _guide_on(qtbot, window)
    _with_a_table(qtbot, window, tmp_path)
    assert _step(window) == "convert.review"
    other = tmp_path / "raw2"
    other.mkdir()
    window.picks["dir"] = str(other)
    window.converter._raw_pathbar.change_button.click()
    assert _step(window) == "convert.scan"
    # The new scan's table starts the path again from checking it.
    window.converter.load_inventory(_rows())
    assert _step(window) == "convert.review"


def test_coming_back_asks_what_to_do(qtbot, window, tmp_path) -> None:
    spy = _WindowSpy()
    QApplication.instance().installEventFilter(spy)
    try:
        _guide_on(qtbot, window)
        _with_a_table(qtbot, window, tmp_path)
        _come_back_to_the_converter(qtbot, window)
    finally:
        QApplication.instance().removeEventFilter(spy)
    question = window.guide.question
    assert question.isVisible() and not question.isWindow()
    assert question.parentWidget() is window.centralWidget()
    assert not [w for w in spy.shown if "uide" in w or "hoice" in w], spy.shown
    assert question.choices() == ["scan", "review", "convert"]
    assert not window.guide.overlay.isVisible(), "one thing at a time"
    # The path's own next step has the focus, so Return takes it.
    focused = QApplication.focusWidget()
    assert getattr(focused, "key", None) == "review"
    # Inside the views, clear of the header.
    assert window.guide._anchor().contains(question.geometry())


def test_an_answer_steers_the_hints_until_it_is_reached(qtbot, window, tmp_path) -> None:
    _guide_on(qtbot, window)
    _with_a_table(qtbot, window, tmp_path)
    _come_back_to_the_converter(qtbot, window)
    rows = {r.key: r for r in window.guide.question.findChildren(QFrame, "guide-choice")}
    QTest.mouseClick(rows["scan"], Qt.MouseButton.LeftButton)
    _settle(qtbot, 60)
    assert not window.guide.question.isVisible()
    assert window.guide.goal("converter") == "scan"
    assert _step(window) == "convert.raw_new"
    assert window.guide.overlay.step.heading == "Scan new data · Step 1 of 2"
    other = tmp_path / "raw3"
    other.mkdir()
    window.picks["dir"] = str(other)
    window.converter._raw_pathbar.change_button.click()
    assert _step(window) == "convert.scan_new"
    window.converter.load_inventory(_rows())     # the scan's table arrives
    assert _step(window) == "convert.review"
    assert window.guide.goal("converter") is None, "reached: the path continues"


def test_not_now_escape_or_a_click_elsewhere_closes_it(qtbot, window, tmp_path) -> None:
    _guide_on(qtbot, window)
    _with_a_table(qtbot, window, tmp_path)
    question = window.guide.question
    for close in ("not now", "escape", "elsewhere"):
        _come_back_to_the_converter(qtbot, window)
        assert question.isVisible(), close
        if close == "not now":
            QTest.mouseClick(question.findChild(QPushButton, "guide-dismiss"),
                             Qt.MouseButton.LeftButton)
        elif close == "escape":
            QTest.keyClick(question, Qt.Key.Key_Escape)
        else:
            table = window.converter._table
            QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                             pos=table.visualRect(table.model().index(2, 2)).center())
        _settle(qtbot, 80)
        assert not question.isVisible(), close
        assert window.guide.goal("converter") is None
        assert window.guide.overlay.isVisible(), close
    # The click elsewhere still did what it does: the user checked a row.
    assert _step(window) == "convert.run"


def test_switching_the_guide_off_closes_the_question(qtbot, window, tmp_path) -> None:
    _guide_on(qtbot, window)
    _with_a_table(qtbot, window, tmp_path)
    _come_back_to_the_converter(qtbot, window)
    assert window.guide.question.isVisible()
    window.guide.set_enabled(False)
    assert not window.guide.question.isVisible()
    assert not window.guide.overlay.isVisible()
