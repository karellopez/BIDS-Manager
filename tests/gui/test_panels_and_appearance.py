"""Folding panels from anywhere on their handle, the filter tree following
the inspection table, glyph buttons that show their glyph, validate icons
in the app's icon colour, and the user's appearance (accent, tint, icon
style, file-tree colours) on top of any theme."""

from __future__ import annotations

import pandas as pd
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QLabel

from bidsmgr.gui import appearance as ap
from bidsmgr.gui.theme_manager import PALETTES, THEMES, ThemeManager, theme_info
from bidsmgr.gui.widgets import PanelFrame

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------


def test_a_side_panel_folds_from_anywhere_on_its_strip(qtbot) -> None:
    frame = PanelFrame(QLabel("body"), "Filter / structure", edge="left")
    qtbot.addWidget(frame)
    frame.resize(300, 400)
    frame.show()
    qtbot.waitExposed(frame)
    strip = frame._strip
    for y in (5, strip.height() // 2, strip.height() - 6):
        before = frame.is_collapsed()
        qtbot.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(strip.width() // 2, y))
        assert frame.is_collapsed() != before, f"a click at y={y} did not fold"
    # Folded, the strip carries the panel's name.
    frame.set_collapsed(True)
    assert strip.collapsed and "Open" in strip.toolTip()


def test_a_top_panel_folds_from_its_title_bar_but_not_its_buttons(qtbot) -> None:
    frame = PanelFrame(QLabel("body"), "Raw data tree", edge="top")
    qtbot.addWidget(frame)
    frame.resize(300, 300)
    frame.show()
    qtbot.waitExposed(frame)
    bar = frame._bar
    qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=QPoint(bar.width() // 2, 10))
    assert frame.is_collapsed()
    qtbot.mouseClick(frame._title_lbl, Qt.MouseButton.LeftButton)
    assert not frame.is_collapsed(), "a click on the title is a click on the bar"
    qtbot.mouseClick(frame._detach_btn, Qt.MouseButton.LeftButton)
    assert not frame.is_collapsed(), "the detach button kept its own click"
    frame.reattach()


# ---------------------------------------------------------------------------
# The filter tree and the inspection table
# ---------------------------------------------------------------------------


def _rows() -> pd.DataFrame:
    from tests.gui.test_side_panes import _func_row

    return pd.DataFrame([
        _func_row(),
        _func_row(participant_id="sub-002", bids_name="sub-002_ses-pre_task-rest_bold",
                  bids_path="sub-002_ses-pre_task-rest_bold", series_uid="9.9.9"),
    ])


def test_picking_a_sequence_in_the_filter_selects_it_everywhere(qtbot, tmp_path) -> None:
    from bidsmgr.gui.converter_panel import ConverterPanel

    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.load_inventory(_rows(), output_tsv=tmp_path / "inv.tsv")
    tree = panel._filter_pane._tree
    leaf = panel._filter_pane._leaf_for_row(1)
    tree.setCurrentItem(leaf)
    assert panel._table.currentIndex().row() == 1
    assert panel._properties._row == 1
    # And back: a row picked in the table is the current leaf.
    panel._jump_to_row(0)
    assert tree.currentItem() is panel._filter_pane._leaf_for_row(0)


# ---------------------------------------------------------------------------
# Buttons
# ---------------------------------------------------------------------------


def test_a_glyph_button_shows_its_glyph(qtbot, qapp) -> None:
    """The minus that removes a template entry rendered as an empty box once
    the generic button padding left it no room in its fixed width."""
    from PyQt6.QtWidgets import QLineEdit, QPushButton

    from bidsmgr.gui.theme_manager import ThemeManager
    from bidsmgr.gui.widgets.template_form import _RowList

    ThemeManager(qapp).apply("dark")          # the stylesheet whose padding hid it

    def make_row():
        edit = QLineEdit()
        return edit, edit.text, edit.setText

    rows = _RowList(make_row)
    qtbot.addWidget(rows)
    rows.show()
    qtbot.waitExposed(rows)
    minus = [b for b in rows.findChildren(QPushButton) if b.text() == "−"]
    assert minus, "no remove button"
    for b in minus:
        assert b.objectName() == "glyph-btn"
        style = b.style()
        from PyQt6.QtWidgets import QStyle, QStyleOptionButton

        opt = QStyleOptionButton()
        b.initStyleOption(opt)
        room = style.subElementRect(QStyle.SubElement.SE_PushButtonContents, opt, b).width()
        assert b.fontMetrics().horizontalAdvance(b.text()) <= room, "the glyph has no room"


def test_the_validate_icons_are_not_recoloured(qtbot, monkeypatch, tmp_path) -> None:
    from bidsmgr.gui import icons
    from bidsmgr.gui.editor_panel import EditorPanel

    seen: list = []
    real = icons.apply_button

    def spy(btn, name, *, color=None, **kw):
        if name in ("file_check", "folder_check"):
            seen.append(color)
        return real(btn, name, color=color, **kw)

    monkeypatch.setattr(icons, "apply_button", spy)
    panel = EditorPanel()
    qtbot.addWidget(panel)
    (tmp_path / "dataset_description.json").write_text("{}", encoding="utf-8")
    panel._sync_validate_buttons_from_selection(tmp_path / "dataset_description.json")
    assert seen and all(c is None for c in seen), "a validate icon was given its own colour"


# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------


def test_an_accent_preset_reads_on_every_theme_of_its_kind() -> None:
    for t in THEMES:
        if t.id.startswith("hc-"):
            continue
        p = PALETTES[t.id]
        for key, (_label, on_dark, on_light) in ap.ACCENTS.items():
            accent = on_dark if t.dark else on_light
            for s in ("bg", "surface", "surface2"):
                assert ap.contrast(accent, p[s]) >= 4.5, (t.id, key, s)
            assert ap.contrast(accent, p["surface3"]) >= 4.0, (t.id, key)


def test_the_appearance_is_laid_over_the_theme() -> None:
    base = PALETTES["graphite"]
    look = ap.Appearance(accent="pink", tint=8, icons="accent", tree={"sidecar": "#123456"})
    pal = ap.apply(base, look, dark=True)
    assert pal["accent"] == ap.ACCENTS["pink"][1]
    assert pal["accent_bg"].startswith("rgba(") and pal["accent_bg"] != base["accent_bg"]
    assert pal["surface"] != base["surface"], "the tint reached the surfaces"
    assert ap.contrast(pal["text"], pal["surface"]) >= 7, "and kept the text readable"
    assert pal["tree_sidecar"] == "#123456"
    assert pal["tree_folder"] == pal["accent"], "folders follow the accent"
    assert ap.contrast(pal["primary_btn_text"], pal["accent"]) >= 4.5
    # Nothing chosen changes nothing.
    assert ap.apply(base, ap.Appearance(), dark=True) == {**base}


def test_a_bad_stored_appearance_is_made_valid() -> None:
    look = ap.Appearance(accent="chartreuse", tint=99, icons="neon",
                         tree={"folder": "#zzz", "nope": "#ffffff", "table": "#00aa00"})
    clean = look.normalised()
    assert (clean.accent, clean.tint, clean.icons) == ("", ap.MAX_TINT, "monochrome")
    assert clean.tree == {"table": "#00aa00"}


def test_the_icon_style_colours_action_icons_only() -> None:
    from bidsmgr.gui import icons

    try:
        icons.set_style("monochrome")
        assert icons.styled_key("scan", "accent") == "text"
        assert icons.styled_key("err", "error") == "error", "a status keeps its meaning"
        assert icons.styled_key("tree_json", "tree_sidecar") == "tree_sidecar"
        icons.set_style("accent")
        assert icons.styled_key("undo", "text") == "accent"
        icons.set_style("colourful")
        assert icons.styled_key("scan", "accent") == "accent"
    finally:
        icons.set_style("monochrome")


def test_the_theme_manager_applies_the_appearance(qapp) -> None:
    from bidsmgr.gui.theme_manager import CUR

    m = ThemeManager(qapp, appearance=ap.Appearance(accent="#cc5500"))
    try:
        m.apply("porcelain")
        assert CUR()["accent"] == "#cc5500" and m.palette["accent"] == "#cc5500"
        assert "#cc5500" in qapp.styleSheet()
        m.set_appearance(ap.Appearance())
        assert CUR()["accent"] == PALETTES["porcelain"]["accent"]
    finally:
        m.set_appearance(ap.Appearance(), apply=False)
        m.apply("dark")


def test_settings_edits_and_saves_the_appearance(qtbot) -> None:
    from bidsmgr.gui.app_settings import AppSettings
    from bidsmgr.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    editor = dlg._appearance_editor
    editor.accent.setCurrentIndex(editor.accent.findData("teal"))
    editor.tint.setValue(5)
    editor.icons.setCurrentIndex(editor.icons.findData("colourful"))
    editor.set_tree_colour("table", "#336699")
    assert editor.tree_buttons["table"].text() == "#336699"
    assert editor.tree_resets["table"].isEnabled()
    dlg._on_save()
    saved = AppSettings.load()
    assert (saved.accent, saved.tint, saved.icon_style) == ("teal", 5, "colourful")
    assert saved.tree_colours == {"table": "#336699"}
    # Restore defaults covers it.
    dlg._on_restore_defaults()
    assert editor.appearance() == ap.Appearance()


def test_a_hard_to_read_accent_says_so(qtbot) -> None:
    from bidsmgr.gui.appearance_editor import AppearanceEditor

    editor = AppearanceEditor()
    qtbot.addWidget(editor)
    editor.set_theme("porcelain")
    editor.load(ap.Appearance(accent="#f0f0a0"))
    assert editor.warning.isVisibleTo(editor) and ":1" in editor.warning.text()
    editor.load(ap.Appearance(accent="blue"))
    assert not editor.warning.isVisibleTo(editor)
    assert theme_info("porcelain").dark is False
