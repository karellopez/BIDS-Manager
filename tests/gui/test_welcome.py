"""Tests for the project-first Welcome tab + lifecycle wiring (Phase D)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.cli._scaffold import project_bundle_dir
from bidsmgr.cli.create import open_or_create_workspace
from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.main_window import MainWindow
from bidsmgr.gui.theme_manager import ThemeManager
from bidsmgr.gui.welcome_panel import WelcomePanel
from bidsmgr.project import Project

pytestmark = pytest.mark.gui


def test_create_project_scaffolds_and_emits(qtbot, isolated_settings, tmp_path) -> None:
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    captured: list = []
    panel.project_opened.connect(lambda proj, root: captured.append((proj, root)))

    root = panel.create_project(tmp_path, "My Study")

    # Slugified folder under the chosen parent.
    assert root == tmp_path / "My-Study"
    assert (root / "dataset_description.json").exists()
    assert project_bundle_dir(root).exists()
    dd = json.loads((root / "dataset_description.json").read_text())
    assert dd["Name"] == "My Study"  # human name preserved
    # Signal fired with a Project + the dataset root; recent list updated.
    assert len(captured) == 1 and isinstance(captured[0][0], Project)
    assert str(root) in AppSettings.load().recent_projects


def test_open_project_adopts_existing(qtbot, isolated_settings, tmp_path) -> None:
    # A dataset created out of band, no .bidsmgr bundle yet.
    root = tmp_path / "external"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "External", "BIDSVersion": "1.11.1"})
    )
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    captured: list = []
    panel.project_opened.connect(lambda proj, r: captured.append((proj, r)))

    panel.open_project(root)

    assert project_bundle_dir(root).exists()  # adopted: bundle added
    assert json.loads((root / "dataset_description.json").read_text())["Name"] == "External"
    assert len(captured) == 1


def test_delete_project_from_disk(qtbot, isolated_settings, tmp_path) -> None:
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    root = panel.create_project(tmp_path, "Doomed")  # also adds to recent
    assert root.exists() and str(root) in AppSettings.load().recent_projects

    ok = panel.delete_project(root, from_disk=True)
    assert ok
    assert not root.exists()                                   # folder gone
    assert str(root) not in AppSettings.load().recent_projects  # forgotten


def test_remove_from_list_keeps_folder(qtbot, isolated_settings, tmp_path) -> None:
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    root = panel.create_project(tmp_path, "Keep")

    panel.delete_project(root, from_disk=False)
    assert root.exists()                                        # folder kept
    assert str(root) not in AppSettings.load().recent_projects  # just forgotten


def test_delete_refuses_non_dataset_folder(qtbot, isolated_settings, tmp_path) -> None:
    # A plain folder (not a BIDS dataset / BM project) must never be rmtree'd.
    plain = tmp_path / "not_a_dataset"
    plain.mkdir()
    (plain / "important.txt").write_text("keep me")
    AppSettings.remember_recent_project(plain)

    panel = WelcomePanel()
    qtbot.addWidget(panel)
    ok = panel.delete_project(plain, from_disk=True)
    assert ok is False
    assert plain.exists() and (plain / "important.txt").exists()  # untouched


def test_mainwindow_lands_on_welcome_and_binds_on_open(
    qapp, isolated_settings, tmp_path,
) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    # Lands on Welcome (index 2) with no project.
    assert win.stack.currentIndex() == 2

    # Simulate the user creating a project from the Welcome tab.
    root = tmp_path / "study"
    proj = open_or_create_workspace(root, name="Study")
    win._on_project_opened(proj, root)
    qapp.processEvents()

    # Switches to the Converter, binds the project, and soft-locks the output.
    assert win.stack.currentIndex() == 0
    assert win.converter._project is proj
    assert win.converter._bids_root == root
    assert win.converter._bids_pathbar.change_button.isEnabled() is False


def test_theme_swap_repaints_welcome(qapp, isolated_settings, monkeypatch) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    calls: list = []
    original = win.welcome.repaint_for_palette

    def _spy(pal: dict) -> None:
        calls.append(pal)
        return original(pal)

    monkeypatch.setattr(win.welcome, "repaint_for_palette", _spy)

    theme.toggle()  # dark -> light
    theme.toggle()  # light -> dark

    # The Welcome panel is in the palette cascade (runs the unpolish/polish
    # dance) so it never keeps stale white/black surfaces.
    assert len(calls) == 2


def test_home_pill_returns_to_welcome(qapp, isolated_settings) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    # Go to Converter, then Home pill back to Welcome.
    win._header._converter_btn.setChecked(True)
    win._header._pill_group.idClicked.emit(0)
    qapp.processEvents()
    assert win.stack.currentIndex() == 0

    win._header._welcome_btn.setChecked(True)
    win._header._pill_group.idClicked.emit(2)
    qapp.processEvents()
    assert win.stack.currentIndex() == 2
