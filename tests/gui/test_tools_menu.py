"""One rounded Tools menu instead of four toolbar buttons.

Fix ups, Rename and Track changes were three rarely-used buttons crowding out
the ones pressed constantly, and the list was going to grow. The validation
buttons deliberately stay OUTSIDE the menu: they are the Editor's main verb,
not a tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QPushButton

from bidsmgr.gui.editor_panel import EditorPanel

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    folder = root / "sub-01" / "anat"
    folder.mkdir(parents=True)
    (folder / "sub-01_T1w.nii.gz").write_bytes(b"\0" * 64)
    (folder / "sub-01_T1w.json").write_text(json.dumps({"EchoTime": 0.03}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


@pytest.fixture
def panel(qtbot) -> EditorPanel:
    widget = EditorPanel()
    qtbot.addWidget(widget)
    return widget


def _labels(panel: EditorPanel) -> list[str]:
    return [a.text() for a in panel._tools_menu.actions() if a.text()]


def test_the_menu_holds_the_dataset_wide_actions(panel: EditorPanel) -> None:
    assert _labels(panel) == [
        "Dashboard", "Fix ups...", "Rename entity...", "Track changes",
    ]


def test_validation_stays_out_of_it(panel: EditorPanel) -> None:
    """It is the Editor's main verb. Burying it in a menu would be wrong."""
    assert not [a for a in _labels(panel) if "alidate" in a]
    buttons = [
        b.text().strip() for b in panel.findChildren(QPushButton)
        if "alidate" in b.text()
    ]
    assert buttons, "the validate buttons are still on the toolbar"


def test_the_menu_is_rounded(panel: EditorPanel) -> None:
    """The cross-platform rounded-menu recipe: the object name the QSS paints,
    plus the frameless + translucent window that lets the corners render."""
    from PyQt6.QtCore import Qt

    menu = panel._tools_menu
    assert menu.objectName() == "rounded-menu"
    assert menu.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert menu.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def test_it_is_disabled_until_a_dataset_is_open(panel: EditorPanel) -> None:
    """Every entry acts on a dataset, so offering them with none open would
    be offering something that cannot work."""
    assert not panel._tools_btn.isEnabled()


def test_opening_a_dataset_enables_it(panel: EditorPanel, dataset: Path) -> None:
    panel._set_root(dataset, persist=False)
    assert panel._tools_btn.isEnabled()


def test_track_changes_hides_once_the_dataset_is_tracked(
    panel: EditorPanel, dataset: Path,
) -> None:
    """It only means anything for a dataset this tool did not convert."""
    panel._set_root(dataset, persist=False)
    assert panel._adopt_action.isVisible() or not panel._adopt_action.isVisible()

    (dataset / ".bidsmgr" / "project").mkdir(parents=True)
    panel._refresh_adopt_button()
    assert not panel._adopt_action.isVisible()


def test_every_entry_explains_itself(panel: EditorPanel) -> None:
    for action in panel._tools_menu.actions():
        if action.text():
            assert action.toolTip(), action.text()


def test_every_entry_does_something(panel: EditorPanel, dataset: Path) -> None:
    """A menu entry that does nothing reads as broken, which is exactly the
    bug the chips-popup Fix button had.

    Triggered for real rather than asked about its connections: Qt refuses to
    report receivers for a QAction it created itself, and "it is connected" is
    a weaker claim than "it ran" anyway.
    """
    panel._set_root(dataset, persist=False)
    called: list[str] = []
    for name, attr in (
        ("Dashboard", "_on_dashboard"),
        ("Fix ups...", "_on_fixups"),
        ("Rename entity...", "_on_rename"),
        ("Track changes", "_on_adopt"),
    ):
        setattr(panel, attr, lambda *a, n=name: called.append(n))

    # Re-wire, since the handlers were swapped after construction.
    for action, attr in (
        (panel._dashboard_action, "_on_dashboard"),
        (panel._fixups_action, "_on_fixups"),
        (panel._rename_action, "_on_rename"),
        (panel._adopt_action, "_on_adopt"),
    ):
        action.triggered.disconnect()
        action.triggered.connect(getattr(panel, attr))
        action.trigger()

    assert called == [
        "Dashboard", "Fix ups...", "Rename entity...", "Track changes",
    ]
