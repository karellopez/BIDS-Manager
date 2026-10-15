"""The request for a GitHub star: when it asks, what it says, how to stop it."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from bidsmgr.gui import star_request as sr

DAY0 = date(2026, 10, 1)


# ---------------------------------------------------------------------------
# When it asks
# ---------------------------------------------------------------------------


def test_once_a_day_until_the_user_has_starred() -> None:
    never = sr.StarSchedule()
    assert sr.due(never, DAY0), "the first launch after What's new asks"
    asked = sr.StarSchedule(last_asked=DAY0.isoformat())
    assert not sr.due(asked, DAY0), "twice on one day"
    assert sr.due(asked, DAY0 + timedelta(days=1))
    assert not sr.due(asked, DAY0 + timedelta(days=1), enabled=False), "the switch is off"
    asked.opened = True
    assert not sr.due(asked, DAY0 + timedelta(days=30)), "asked again after going to GitHub"


def test_the_request_comes_once_a_day(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    dialog = sr.maybe_show(host, DAY0)
    assert isinstance(dialog, sr.StarDialog)
    assert dialog.parent() is host, "a parentless dialog would be a stray window"
    assert dialog._buttons["later"].text() == "Not now"
    dialog.reject()
    assert sr.maybe_show(host, DAY0) is None, "asked twice on one day"
    tomorrow = sr.maybe_show(host, DAY0 + timedelta(days=1))
    assert isinstance(tomorrow, sr.StarDialog)
    tomorrow.reject()


def test_the_settings_switch_stops_it(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QWidget

    from bidsmgr.gui.app_settings import AppSettings

    host = QWidget()
    qtbot.addWidget(host)
    AppSettings.remember_ask_for_star(False)
    assert sr.maybe_show(host, DAY0) is None


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------


def test_star_on_github_opens_the_project_and_ends_the_requests(
        qtbot, isolated_settings, monkeypatch) -> None:
    from PyQt6.QtWidgets import QWidget

    opened = []
    monkeypatch.setattr(sr.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    dialog = sr.StarDialog(asked=True)
    qtbot.addWidget(dialog)
    assert dialog._buttons["star"].isDefault()
    dialog._buttons["star"].click()
    assert opened == [sr.REPO_URL]
    assert sr.REPO_URL == "https://github.com/ANCPLabOldenburg/BIDS-Manager"
    assert dialog.result() == dialog.DialogCode.Accepted
    host = QWidget()
    qtbot.addWidget(host)
    assert sr.maybe_show(host, DAY0 + timedelta(days=3)) is None


def test_dont_ask_again_is_the_settings_switch(qtbot, isolated_settings) -> None:
    from bidsmgr.gui.app_settings import AppSettings

    dialog = sr.StarDialog()
    qtbot.addWidget(dialog)
    assert dialog._buttons["later"].text() == "Close", "opened by hand: nothing to postpone"
    assert not dialog.never.isChecked()
    dialog.never.setChecked(True)
    assert AppSettings.load().ask_for_star is False
    dialog.never.setChecked(False)
    assert AppSettings.load().ask_for_star is True
    AppSettings.remember_ask_for_star(False)
    reopened = sr.StarDialog()
    qtbot.addWidget(reopened)
    assert reopened.never.isChecked()


def test_settings_display_page_carries_the_switch(qtbot, isolated_settings) -> None:
    from bidsmgr.gui.app_settings import AppSettings
    from bidsmgr.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    assert dlg._ask_for_star.isChecked()
    dlg._ask_for_star.setChecked(False)
    dlg._on_save()
    assert AppSettings.load().ask_for_star is False


def test_it_says_why_how_and_that_nothing_is_tracked(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QLabel

    dialog = sr.StarDialog()
    qtbot.addWidget(dialog)
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    for needed in ("helpful and valuable", "click Star", "more visible", "following its updates",
                   "does not track you", "no usage data", "only way we get an idea"):
        assert needed in text, needed
    assert "—" not in text, "no em dashes in what the user reads"


@pytest.mark.parametrize("theme_id", ["dark", "light", "hc-dark"])
def test_the_dialog_paints_and_stops_when_hidden(qtbot, isolated_settings, theme_id) -> None:
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui.theme_manager import ThemeManager

    ThemeManager(QApplication.instance()).apply(theme_id)
    dialog = sr.StarDialog(asked=True)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    assert dialog.canvas.is_running()
    for t in (0.0, 1.0, sr.GATHER_AT + 0.7, sr.MORPH_AT + 0.5, sr.WALK_AT + 1.0,
              sr.PRESS_AT + 0.1, sr.CHEER_AT + 0.4, sr.SETTLED + 3.0):
        dialog.canvas.set_time(t)
        assert not dialog.canvas.grab().isNull()
    dialog.hide()
    assert not dialog.canvas.is_running(), "the animation keeps running behind a closed dialog"


# ---------------------------------------------------------------------------
# The animation's timeline
# ---------------------------------------------------------------------------


def test_each_kind_has_its_own_form() -> None:
    parts = sr.StarParticles()
    assert parts.n == sum(sr.COUNTS)
    figures = parts.figures_at(1.6)
    assert np.isfinite(figures).all()
    for i, centre in enumerate(parts.anchor):
        mine = figures[parts.members[i]]
        assert np.abs(mine.mean(axis=0) - centre).max() < 0.12, sr.KINDS[i]
        spread = mine.max(axis=0) - mine.min(axis=0)
        assert spread.max() > 0.2, f"{sr.KINDS[i]} is a dot, not a form"
    # The brain is longer than it is high (seen from the side).
    brain = sr.form_points(0, parts.base[parts.members[0]], 0.0)
    width, height = brain.max(axis=0) - brain.min(axis=0)
    assert width > 1.1 * height
    # The traces scroll; the brain only rocks.
    eeg = parts.base[parts.members[3]]
    assert not np.allclose(sr.form_points(3, eeg, 0.0), sr.form_points(3, eeg, 0.5))


def test_the_forms_become_the_logo_then_a_star() -> None:
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QPolygonF

    parts = sr.StarParticles()
    hold = (sr.GATHER_AT + sr.GATHER + sr.GATHER_STAGGER + sr.MORPH_AT) / 2
    assert np.abs(parts.at(hold) - parts.logo).max() < 1e-9, "the logo's lines"
    star = QPolygonF([QPointF(x, y) for x, y in
                      sr.star_outline(sr.STAR_RADIUS * 1.001) + sr.STAR_CENTRE])
    inside = [star.containsPoint(QPointF(x, y), Qt.FillRule.WindingFill) for x, y in parts.star]
    assert sum(inside) >= 0.97 * parts.n
    dim = sr.MORPH_AT + sr.MORPH + sr.MORPH_STAGGER + 0.1
    assert np.abs(parts.at(dim) - parts.star).max() < 1e-9
    late = parts.at(sr.SETTLED + 30.0)
    assert np.abs(late - parts.star).max() < 0.02, "the star drifts instead of breathing"


def test_the_star_forms_dim_and_the_button_lights_it() -> None:
    parts = sr.StarParticles(counts=(20,) * 7)
    kinds = np.tile(np.array([[1.0, 0.2, 0.2]]), (7, 1))
    grey = np.array([0.5, 0.5, 0.5])
    dim = sr.MORPH_AT + sr.MORPH + 0.2
    assert sr.dimness(dim) == pytest.approx(1.0) and sr.lit(dim) == 0.0
    dark = parts.colours(dim, kinds, grey)
    assert np.allclose(dark, kinds[0] + (grey - kinds[0]) * 0.72), "dim means greyed"
    bright = sr.PRESS_AT + sr.LIGHT + 0.01
    assert sr.lit(bright) == pytest.approx(1.0) and sr.dimness(bright) == pytest.approx(0.0)
    calm = sr.SETTLED + sr.GLINT_EVERY * 10 + sr.GLINT + 0.1        # between two glints
    assert np.allclose(parts.colours(calm, kinds, grey), kinds[0])


def test_the_scientist_walks_up_presses_and_cheers() -> None:
    start = sr.pose_at(sr.WALK_AT)
    assert start.x < -1.15, "walks in from outside the canvas"
    arrived = sr.pose_at(sr.WALK_AT + sr.WALK)
    assert arrived.x == pytest.approx(sr.STOP_X)
    sx, sy = sr.shoulder(arrived)
    assert np.hypot(sr.PUSH_AT[0] - sx, sr.PUSH_AT[1] - sy) == pytest.approx(sr.ARM, abs=1e-6), \
        "one arm's length from the button"
    pressing = sr.pose_at(sr.PRESS_AT)
    assert np.allclose(pressing.hand, sr.PUSH_AT), "the hand is on the button when it lights"
    assert sr.cap_offset(sr.PRESS_AT + 0.12) == pytest.approx(sr.PRESS_DEPTH)
    cheering = sr.pose_at(sr.CHEER_AT + 0.5)
    assert cheering.hand[1] < sr.shoulder(cheering)[1], "arms up"
    after = sr.pose_at(sr.SETTLED + 5)
    assert after.looks_up and after.hand[1] > sr.shoulder(after)[1]
    # Never inside the console.
    assert sr.STOP_X + 0.062 < sr.BUTTON_X - sr.BUTTON_HALF


# ---------------------------------------------------------------------------
# Where else it opens
# ---------------------------------------------------------------------------


def test_home_help_us_sits_in_the_cards_bottom_right_corner(
        qtbot, isolated_settings, monkeypatch) -> None:
    from PyQt6.QtWidgets import QSpacerItem

    from bidsmgr.gui import welcome_panel as wp

    shown = []
    monkeypatch.setattr(sr, "show_dialog", lambda parent, **kw: shown.append(parent))
    panel = wp.WelcomePanel()
    qtbot.addWidget(panel)
    button = panel._star_button
    assert button.text() == "Help us" and not button.icon().isNull()
    card = button.parentWidget().layout()
    corner = card.itemAt(card.count() - 1).layout()
    assert corner is not None and corner.itemAt(corner.count() - 1).widget() is button
    assert isinstance(corner.itemAt(0), QSpacerItem), "pushed to the right"
    assert isinstance(card.itemAt(card.count() - 2), QSpacerItem), "pushed to the bottom"
    button.click()
    assert len(shown) == 1


def test_the_star_icon_takes_the_accent(qtbot, isolated_settings) -> None:
    from bidsmgr.gui import icons

    assert icons.NAMES["star"][1] == "accent"
    for style in ("monochrome", "accent", "colourful"):
        icons.set_style(style)
        assert icons.styled_key("star", "accent") == "accent", style
    icons.set_style("monochrome")


def test_the_whats_new_dialog_opens_it(qtbot, isolated_settings, monkeypatch) -> None:
    from bidsmgr.gui import whats_new

    shown = []
    monkeypatch.setattr(sr, "show_dialog", lambda parent, **kw: shown.append(parent))
    dialog = whats_new.WhatsNewDialog()
    qtbot.addWidget(dialog)
    dialog._buttons["star"].click()
    assert shown == [dialog]
    assert dialog._buttons["continue"].isDefault()


def test_building_the_window_never_asks(qtbot, isolated_settings) -> None:
    """The request is scheduled by main(), so tests that build a window get none."""
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager

    theme = ThemeManager(QApplication.instance())
    theme.apply("dark")
    win = MainWindow(theme)
    qtbot.addWidget(win)
    qtbot.wait(700)
    assert not win.findChildren(sr.StarDialog)
