"""The seven themes and the two bundled typefaces.

Every theme must give the stylesheet every token it names (a missing one
leaves ``$name`` in the text and Qt drops the whole stylesheet), read
clearly (contrast measured, not eyeballed), say whether it is dark, and be
pickable from the header and from Settings. The interface is drawn in
Inter and code in JetBrains Mono, bundled, so no rule may name a font only
macOS has.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bidsmgr.gui import theme_assets, typefaces
from bidsmgr.gui.theme_manager import PALETTES, THEMES, ThemeManager, theme_info
from bidsmgr.viz.theme import VizTheme, luminance

pytestmark = pytest.mark.gui

QSS = (Path(typefaces.__file__).parent / "theme.qss").read_text(encoding="utf-8")
GUI = Path(typefaces.__file__).parent


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.fixture
def manager(qapp):
    m = ThemeManager(qapp)
    yield m
    m.apply("dark")


def test_there_are_seven_themes_and_each_is_whole() -> None:
    assert [t.id for t in THEMES] == ["dark", "dim", "nord", "hc-dark", "light", "paper",
                                      "hc-light"]
    keys = set(PALETTES["dark"])
    for t in THEMES:
        assert set(PALETTES[t.id]) == keys, t.id
        assert theme_info(t.partner).dark != t.dark, t.id


def test_every_token_the_stylesheet_names_is_given() -> None:
    used = set(re.findall(r"\$\{?([A-Za-z_]\w*)\}?", QSS))
    images = set(theme_assets.images(PALETTES["dark"]))
    for t in THEMES:
        assert not used - set(PALETTES[t.id]) - images, t.id


@pytest.mark.parametrize("theme_id", [t.id for t in THEMES])
def test_text_reads_clearly_in_every_theme(theme_id) -> None:
    """WCAG contrast: text at 7:1 (AAA) on every surface, secondary text and
    every status colour at 4.5:1 (4.0 on the button surface), the outline
    of an input at 3:1; stricter for the high-contrast pair."""
    p = PALETTES[theme_id]
    strong = theme_id.startswith("hc-")
    for surface in ("bg", "surface", "surface2", "surface3"):
        assert _contrast(p["text"], p[surface]) >= (15 if strong else 7), surface
        assert _contrast(p["dim"], p[surface]) >= (7 if strong else 4.5), surface
        for colour in ("accent", "success", "warning", "error", "purple", "teal"):
            floor = 7 if strong else (4.0 if surface == "surface3" else 4.5)
            assert _contrast(p[colour], p[surface]) >= floor, (colour, surface)
    assert _contrast(p["primary_btn_text"], p["accent"]) >= (7 if strong else 4.5)
    assert _contrast(p["input_border"], p["bg"]) >= (4.5 if strong else 3.0)


def test_each_theme_says_whether_it_is_dark() -> None:
    for t in THEMES:
        assert VizTheme.from_palette(PALETTES[t.id], t.id).dark == t.dark, t.id


def test_applying_a_theme_leaves_no_token_unfilled(manager, qapp) -> None:
    for t in THEMES:
        manager.apply(t.id)
        assert manager.name == t.id and manager.is_dark == t.dark
        sheet = qapp.styleSheet()
        assert not re.search(r"\$[A-Za-z_]", sheet), t.id
        assert PALETTES[t.id]["accent"] in sheet


def test_toggle_goes_to_the_partner(manager) -> None:
    manager.apply("dim")
    assert manager.toggle() == "paper"
    manager.apply("hc-light")
    assert manager.toggle() == "hc-dark"


def test_the_stylesheet_images_are_drawn_in_the_theme(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(theme_assets, "_cache_dir", lambda: tmp_path / "with space")
    tokens = theme_assets.write(PALETTES["paper"], "paper")
    assert set(tokens) == set(theme_assets.images(PALETTES["paper"]))
    for token in tokens.values():
        assert token.startswith('url("') and "\\" not in token
        path = Path(token[5:-2])
        assert path.is_file() and path.read_text(encoding="utf-8").startswith("<svg")
    tick = Path(tokens["icon_check"][5:-2]).read_text(encoding="utf-8")
    assert PALETTES["paper"]["primary_btn_text"] in tick


def test_the_typefaces_are_bundled_and_used(qapp) -> None:
    from PyQt6.QtGui import QFontInfo

    assert typefaces.load()
    assert QFontInfo(typefaces.ui_font(13)).family() == typefaces.UI_FAMILY
    from bidsmgr.gui.viz import fonts

    assert QFontInfo(fonts.font(12, mono=True)).family() == typefaces.MONO_FAMILY
    for name in ("Inter-OFL.txt", "JetBrainsMono-OFL.txt"):
        assert (typefaces.FONT_DIR / name).is_file(), "a licence travels with its fonts"


def test_no_rule_or_widget_names_a_font_only_macos_has() -> None:
    mac = re.compile(r"SF Mono|Menlo|Monaco|AppleSystemUIFont|Helvetica Neue|SFMono")
    assert not mac.search(QSS)
    families = set(re.findall(r'font-family:\s*([^;]+);', QSS))
    assert families == {'"JetBrains Mono"'}, families
    for path in GUI.rglob("*.py"):
        if path.name == "typefaces.py":
            continue
        assert not mac.search(path.read_text(encoding="utf-8")), path


def test_the_header_lists_every_theme_and_applies_the_pick(manager, qtbot) -> None:
    from bidsmgr.gui.app_settings import AppSettings
    from bidsmgr.gui.main_window import MainWindow

    manager.apply("dark")
    win = MainWindow(manager)
    qtbot.addWidget(win)
    header = win._header
    header._open_theme_menu()
    menu = header._theme_menu
    acts = [a for a in menu.actions() if a.data()]
    assert [a.data() for a in acts] == [t.id for t in THEMES]
    current = next(a for a in acts if a.isChecked())
    assert current.data() == "dark" and current.font().weight() > 500
    next(a for a in acts if a.data() == "nord").trigger()
    menu.close()
    assert manager.name == "nord"
    assert AppSettings.load().theme == "nord"
    assert "Nord" in header._theme_btn.toolTip()


def test_settings_lists_the_themes_by_name(qtbot) -> None:
    from bidsmgr.gui.app_settings import AppSettings
    from bidsmgr.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    combo = dlg._theme_combo
    assert [combo.itemData(i) for i in range(combo.count())] == [t.id for t in THEMES]
    assert combo.itemText(combo.findData("hc-light")) == "High contrast light"
    assert not combo.itemIcon(0).isNull(), "each theme shows its swatch"


def test_welcome_links_follow_a_theme_change(manager, qtbot) -> None:
    from PyQt6.QtWidgets import QLabel

    from bidsmgr.gui.welcome_panel import WelcomePanel

    manager.apply("nord")
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    link = next(lb for lb in panel.findChildren(QLabel) if lb.objectName() == "welcome-link")
    assert PALETTES["nord"]["accent"] in link.text()
    manager.apply("paper")
    panel.repaint_for_palette(PALETTES["paper"])
    assert PALETTES["paper"]["accent"] in link.text()
    assert PALETTES["nord"]["accent"] not in link.text()


def test_the_cli_accepts_every_theme() -> None:
    from bidsmgr.main import _theme_ids

    assert _theme_ids() == [t.id for t in THEMES]
