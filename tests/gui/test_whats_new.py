"""The "What's new" dialog: shown once per release, a scene per feature,
painted in the theme's colours."""

from __future__ import annotations

import math
import time

import pytest

from bidsmgr.gui import news_scenes as ns
from bidsmgr.gui import whats_new as wn


def test_shown_until_this_releases_news_has_been_seen() -> None:
    assert wn.NEWS_FOR == "1.5.2"
    assert wn.should_show("")             # a first install
    assert wn.should_show("1.5")          # an update from the release before
    assert not wn.should_show(wn.NEWS_FOR)
    assert not wn.should_show(f" {wn.NEWS_FOR} ")


def test_maybe_show_opens_it_once(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QWidget

    from bidsmgr.gui.app_settings import AppSettings

    host = QWidget()
    qtbot.addWidget(host)
    first = wn.maybe_show(host)
    assert isinstance(first, wn.WhatsNewDialog)
    assert first.parent() is host, "a parentless dialog would be a stray window"
    assert AppSettings.whats_new_seen() == wn.NEWS_FOR
    first.reject()
    assert wn.maybe_show(host) is None, "shown again after it was seen"


def test_building_the_window_never_opens_it(qtbot, isolated_settings) -> None:
    """The dialog is scheduled by main(), so tests that build a window get none."""
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager

    theme = ThemeManager(QApplication.instance())
    theme.apply("dark")
    win = MainWindow(theme)
    qtbot.addWidget(win)
    qtbot.wait(700)
    assert not win.findChildren(wn.WhatsNewDialog)


# ---------------------------------------------------------------------------
# What it says
# ---------------------------------------------------------------------------


def test_it_lists_what_151_and_152_brought() -> None:
    keys = [s.key for s in ns.SCENES]
    assert keys == ["guide", "physio", "dummy", "names", "sidecars", "derivatives",
                    "undo", "anonymise", "ignore"]
    versions = [s.version for s in ns.SCENES]
    assert versions == sorted(versions, reverse=True), "newest release first, grouped"
    assert set(versions) == {"1.5.1", "1.5.2"}
    for scene in ns.SCENES:
        for words in (scene.title, scene.text, scene.shows):
            assert words and "—" not in words and "–" not in words, scene.key
        assert 6.0 <= scene.duration <= 12.0, scene.key
    assert "—" not in ns.ALSO and "Windows" in ns.ALSO


def test_the_words_shown_are_the_apps_own() -> None:
    """A scene shows what the app does, in its words: the guide's real first
    question, the menu entries and the anonymiser's real output."""
    import inspect

    from bidsmgr.anonymise.policy import CATEGORIES, PURPOSES
    from bidsmgr.editor.sidecars import ORIGINS
    from bidsmgr.gui.guide_steps import find_task

    source = inspect.getsource(ns)
    facts = {"project": True, "rows": 5, "tables": 1, "row_clicks": 0, "raw_picks": 1}
    review = find_task("converter", "review", facts)
    assert review.label in source and len(review.steps) == 4
    for step in review.steps[:3]:
        assert step.title in source
    for origin in ORIGINS.values():
        if origin != "Anonymisation":
            assert origin in source, origin
    titles = {c.key: c.title for c in CATEGORIES}
    for key in ("identity", "dates", "site", "devices"):
        assert titles[key] in source, key
    for _key, heading in PURPOSES[:3]:
        assert heading.upper() in source, heading
    for entry in ("Remove faces...", "Remove the skull...", "Ignore in validation",
                  "Every .txt file", "Write its sidecar"):
        assert entry in source, entry


def test_the_anonymised_values_are_what_the_anonymiser_writes(tmp_path) -> None:
    """The scene's sidecar, anonymised with the Recommended preset: the same
    fields go, the same kinds of values come."""
    import json
    import re

    from bidsmgr.anonymise.plan import plan_anonymise
    from bidsmgr.anonymise.policy import policy_for

    (tmp_path / "dataset_description.json").write_text('{"Name": "t", "BIDSVersion": "1.10.0"}')
    folder = tmp_path / "sub-01" / "anat"
    folder.mkdir(parents=True)
    (folder / "sub-01_T1w.json").write_text(json.dumps({
        "PatientName": "Anna Müller", "PatientID": "OL_0007",
        "AcquisitionDateTime": "2026-03-14T09:12:00", "InstitutionName": "University of Oldenburg",
        "InstitutionAddress": "Ammerländer Heerstr. 114", "DeviceSerialNumber": "167023",
        "StationName": "AWP167023", "RepetitionTime": 2.3}))
    plan = plan_anonymise(tmp_path, policy_for("recommended"))
    changes = {c.what: c for f in plan.files for c in f.changes}
    assert changes["PatientName"].after == "" and changes["PatientID"].after == ""
    assert re.fullmatch(r'"19(0[2-9]|1\d|2[0-5])-\d\d-\d\dT09:12:00"',
                        changes["AcquisitionDateTime"].after)
    assert changes["InstitutionAddress"].after == '"n/a"'
    assert "InstitutionName" not in changes, "Recommended keeps the institution's name"
    assert changes["DeviceSerialNumber"].after.startswith("device-")
    assert changes["StationName"].after.startswith("station-")
    assert "RepetitionTime" not in changes


def test_the_shifted_date_keeps_the_weekday() -> None:
    from datetime import date

    assert date(2026, 3, 14).weekday() == date(1905, 3, 11).weekday()
    assert (date(2026, 3, 14) - date(1905, 3, 11)).days % 7 == 0


def test_the_subject_labels_and_entities_shown_are_what_the_scan_makes() -> None:
    from bidsmgr.inventory import name_entities, subject_label

    assert subject_label.curate("OL_0007") == "OL0007"
    assert subject_label.curate("Müller") == "Muller"
    assert name_entities.from_name("t1_mprage_acq-highres") == {"acq": "highres"}
    assert subject_label.SOURCES[subject_label.PATIENT_ID] == "Patient ID"
    assert subject_label.SOURCES[subject_label.FAMILY_NAME] == "Family name"


def test_the_stripped_derivative_is_named_as_the_tool_names_it(tmp_path) -> None:
    from bidsmgr.deface.derivatives import PIPELINE, derivative_path

    image = tmp_path / "derivatives" / "fmriprep" / "sub-01" / "anat" / \
        "sub-01_desc-preproc_T1w.nii.gz"
    out = derivative_path(tmp_path, image)
    assert out is not None and PIPELINE == "bidsmgr-skullstrip"
    assert out.relative_to(tmp_path).as_posix() == \
        "derivatives/bidsmgr-skullstrip/sub-01/anat/sub-01_desc-preprocbrain_T1w.nii.gz"


# ---------------------------------------------------------------------------
# How it plays
# ---------------------------------------------------------------------------


def test_a_scene_fades_in_holds_and_fades_out() -> None:
    assert wn.scene_alpha(0.0, 8.0) == 0.0
    assert wn.scene_alpha(wn.FADE, 8.0) == 1.0
    assert wn.scene_alpha(4.0, 8.0) == 1.0
    assert wn.scene_alpha(8.0, 8.0) == 0.0
    assert ns.swap(0.0) == (1.0, 0.0) and ns.swap(1.0) == (0.0, 1.0)
    old, new = ns.swap(0.5)
    assert old == new == 0.0, "the old text has gone before the new one comes"


@pytest.mark.parametrize("theme_id", ["dark", "light", "hc-dark"])
def test_every_scene_paints_at_every_moment(qtbot, isolated_settings, theme_id) -> None:
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui.theme_manager import ThemeManager

    ThemeManager(QApplication.instance()).apply(theme_id)
    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    dialog.resize(1000, 600)
    dialog.show()
    qtbot.waitExposed(dialog)
    assert dialog.stage.is_running()
    for i, scene in enumerate(ns.SCENES):
        dialog.stage.set_scene(i, by_user=True)
        for k in range(9):
            dialog.stage.set_time(scene.duration * k / 8)
            assert not dialog.stage.grab().isNull()
    dialog.hide()
    assert not dialog.stage.is_running(), "the animation keeps running behind a closed dialog"


def test_a_frame_is_cheap(qtbot, isolated_settings) -> None:
    """At 60 frames a second a frame has 16 ms; every scene stays far below."""
    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    dialog.resize(1000, 600)
    dialog.show()
    qtbot.waitExposed(dialog)
    for i, scene in enumerate(ns.SCENES):
        dialog.stage.set_scene(i, by_user=True)
        dialog.stage.set_time(0.5)
        dialog.stage.repaint()           # warm the fonts and the template
        start = time.perf_counter()
        for k in range(20):
            dialog.stage.set_time(scene.duration * k / 20)
            dialog.stage.repaint()
        per_frame = (time.perf_counter() - start) / 20 * 1000
        assert per_frame < 12.0, f"{scene.key}: {per_frame:.1f} ms a frame"


def test_the_stage_keeps_the_scenes_proportions() -> None:
    from PyQt6.QtWidgets import QApplication  # noqa: F401  (a QApplication exists)

    stage = wn.FeatureStage()
    try:
        assert stage.hasHeightForWidth()
        for width in (400, 600, 900):
            height = stage.heightForWidth(width)
            pad, below = stage._pads()
            assert math.isclose((width - 2 * pad) / (height - 2 * pad - below), ns.W / ns.H,
                                rel_tol=0.01)
    finally:
        stage.deleteLater()


def test_scenes_hand_over_until_the_user_picks_one(qtbot, isolated_settings, monkeypatch) -> None:
    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    stage = dialog.stage
    assert stage.index() == 0 and stage.is_auto()
    monkeypatch.setattr(stage, "time", lambda: 99.0)
    stage._tick()
    assert stage.index() == 1, "a finished scene hands over to the next"
    assert dialog.feature_buttons[1].isChecked()
    assert dialog.scene_title.text() == ns.SCENES[1].title
    assert dialog.scene_text.text() == ns.SCENES[1].text
    dialog.feature_buttons[4].click()
    assert stage.index() == 4 and not stage.is_auto()
    stage._tick()
    assert stage.index() == 4, "a scene the user picked plays again"
    monkeypatch.undo()
    stage.set_scene(len(ns.SCENES) - 1)
    stage.set_scene(stage.index() + 1)
    assert stage.index() == 0, "after the last, the first"


def test_arrow_keys_pick_the_previous_or_next(qtbot, isolated_settings) -> None:
    from PyQt6.QtCore import Qt

    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    dialog._buttons["continue"].setFocus()
    qtbot.keyClick(dialog, Qt.Key.Key_Right)
    assert dialog.stage.index() == 1 and not dialog.stage.is_auto()
    qtbot.keyClick(dialog, Qt.Key.Key_Left)
    qtbot.keyClick(dialog, Qt.Key.Key_Left)
    assert dialog.stage.index() == len(ns.SCENES) - 1
    assert dialog.feature_buttons[-1].isChecked()


def test_the_stage_says_what_it_shows(qtbot, isolated_settings) -> None:
    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    for i, scene in enumerate(ns.SCENES):
        dialog.stage.set_scene(i)
        assert scene.title in dialog.stage.accessibleName()
        assert dialog.stage.accessibleDescription() == scene.shows


def test_the_features_are_grouped_by_release(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QLabel

    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    groups = [label.text() for label in dialog.findChildren(QLabel)
              if label.objectName() == "whatsnew-group"]
    assert groups == ["New in 1.5.2", "New in 1.5.1"]
    assert [b.text() for b in dialog.feature_buttons] == [s.title for s in ns.SCENES]
    assert all(b.toolTip() for b in dialog.feature_buttons)


def test_the_text_fits_at_a_larger_font(qtbot, isolated_settings) -> None:
    """At 130 percent the feature names still fit their column."""
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui.theme_manager import ThemeManager

    theme = ThemeManager(QApplication.instance())
    theme.set_font_scale(1.3)
    theme.apply("dark")
    try:
        dialog = wn.WhatsNewDialog()
        qtbot.addWidget(dialog)
        dialog.resize(1300, 780)
        dialog.show()
        qtbot.waitExposed(dialog)
        for button in dialog.feature_buttons:
            assert button.width() >= button.sizeHint().width() - 1, button.text()
            assert button.geometry().right() <= dialog.width()
    finally:
        theme.set_font_scale(1.0)
        theme.apply("dark")


def test_the_buttons_open_the_documentation(qtbot, isolated_settings, monkeypatch) -> None:
    opened = []
    monkeypatch.setattr(wn.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    dialog._buttons["docs"].click()
    dialog._buttons["notes"].click()
    assert opened == [wn.DOCS_URL, wn.NOTES_URL]
    assert wn.DOCS_URL.startswith("https://ancplaboldenburg.github.io/bids_manager_documentation/")
    assert wn.NOTES_URL.endswith("/updates/#v152")
    assert dialog._buttons["continue"].isDefault()
    dialog._buttons["continue"].click()
    assert dialog.result() == dialog.DialogCode.Accepted


def test_show_dialog_deletes_the_dialog_when_closed(qtbot, isolated_settings) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    dialog = wn.show_dialog(host)
    assert dialog.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.reject()


def test_seven_distinct_colours_in_every_theme() -> None:
    """The star request's seven kinds of data: seven colours a reader can
    tell apart, each standing out from the theme's background."""
    from bidsmgr.gui import theme_manager as tm
    from bidsmgr.gui.tones import contrast, oklch
    from bidsmgr.viz.theme import VizTheme

    def lab(c):
        light, chroma, hue = oklch(c)
        return light, chroma * math.cos(math.radians(hue)), chroma * math.sin(math.radians(hue))

    for theme_id, palette in tm.PALETTES.items():
        theme = VizTheme.from_palette(palette, theme_id)
        tokens = {**theme.tokens, **{f"series{i + 1}": theme.series(i) for i in range(6)}}
        colours = wn.seven_colours(tokens)
        assert len(colours) == 7 and len(set(colours)) == 7, theme_id
        closest = min(math.dist(lab(a), lab(b))
                      for i, a in enumerate(colours) for b in colours[i + 1:])
        assert closest >= 0.06, f"{theme_id}: two kinds share a colour ({closest:.3f})"
        assert min(contrast(c, palette["bg"]) for c in colours) >= 3.0, theme_id


def test_the_classifier_settings_credit_the_swiftcat_team(qtbot, isolated_settings) -> None:
    """Settings > Classification says the parameters are for MRI, and credits
    the SWIFTCAT team, behind an info icon at the top of the page."""
    from bidsmgr.classifier.acquisition.config import ClassifierConfig
    from bidsmgr.gui import classifier_rules
    from bidsmgr.gui.viz.panels import explain_popup

    page = classifier_rules.ClassifierSettingsPage(("general",), root=ClassifierConfig,
                                                   hint="A hint.")
    qtbot.addWidget(page)
    page.about_button.click()
    _menu, label = explain_popup.last_shown()
    text = label.text()
    for needed in CREDIT_WORDS:
        assert needed in text, needed
    _menu.close()


#: What the note must say.
CREDIT_WORDS = ("For MRI only", "SWIFTCAT", "Marta Loureiro", "Rita G. Nunes",
                "Rita Reis Nunes", "LaSEEB", "University of Lisbon", "Patricia Clement",
                "UZ Gent", "dcm2niix", "early implementation", "works well overall",
                "clinical and heterogeneous data", "stay tuned")
