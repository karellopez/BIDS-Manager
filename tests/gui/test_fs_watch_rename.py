"""Renaming a folder the GUI is watching, which Windows refuses.

``QFileSystemWatcher`` is ``FindFirstChangeNotification`` on Windows, which
holds an open handle on every directory it watches, and an open handle makes
``MoveFile`` fail with ``ERROR_ACCESS_DENIED``. A subject rename then moved
every file correctly, failed only on the call that would have moved the folder,
and left the emptied original behind — the reported symptom.

The behavioural assertions here are Windows-only because the behaviour is: on
POSIX a watched directory renames fine and there is nothing to prove. What is
checked everywhere is the contract of the context manager itself, so a change
that broke the restore would be caught on any CI runner.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PyQt6.QtCore import QFileSystemWatcher
from PyQt6.QtWidgets import QWidget

from bidsmgr.gui.fs_watch import live_watchers, watchers_released

pytestmark = pytest.mark.gui

windows_only = pytest.mark.skipif(
    os.name != "nt", reason="only Windows holds a handle on a watched directory",
)


def _tree(tmp_path: Path) -> Path:
    """A miniature of the shape that broke: sub-X/ses-pre/<datatype>."""
    leaf = tmp_path / "sub-old" / "ses-pre" / "eeg"
    leaf.mkdir(parents=True)
    (leaf / "sub-old_ses-pre_task-rest_eeg.edf").write_bytes(b"")
    return leaf


@windows_only
def test_a_watched_directory_cannot_be_renamed(qtbot, tmp_path):
    """The defect itself. If this ever stops failing, the fix is unnecessary."""
    leaf = _tree(tmp_path)
    subject = leaf.parent.parent

    host = QWidget()
    qtbot.addWidget(host)
    watcher = QFileSystemWatcher(host)
    watcher.addPaths([str(leaf), str(leaf.parent), str(subject)])

    with pytest.raises(PermissionError) as excinfo:
        subject.rename(subject.with_name("sub-new"))
    assert getattr(excinfo.value, "winerror", None) == 5
    assert subject.exists(), "the rename failed, so the old folder is still there"


@windows_only
def test_a_watch_on_a_descendant_alone_also_blocks_the_parent(qtbot, tmp_path):
    """Why the dataset-folder rename needs the same treatment.

    Nothing watches the project root itself; the panes watch the datatype
    folders under it. That is still enough to pin the root in place.
    """
    leaf = _tree(tmp_path)
    subject = leaf.parent.parent

    host = QWidget()
    qtbot.addWidget(host)
    watcher = QFileSystemWatcher(host)
    watcher.addPaths([str(leaf)])          # the leaf only

    with pytest.raises(PermissionError):
        subject.rename(subject.with_name("sub-new"))


@windows_only
def test_released_watches_let_the_folder_move(qtbot, tmp_path):
    """The fix, through the real context manager."""
    leaf = _tree(tmp_path)
    subject = leaf.parent.parent

    host = QWidget()
    host.show()                            # top-level, so live_watchers finds it
    qtbot.addWidget(host)
    watcher = QFileSystemWatcher(host)
    watcher.addPaths([str(leaf), str(leaf.parent), str(subject)])
    assert watcher in live_watchers()

    target = subject.with_name("sub-new")
    with watchers_released():
        subject.rename(target)

    assert target.is_dir()
    assert not subject.exists(), "no empty original left behind"


def test_the_block_restores_what_still_exists(qtbot, tmp_path):
    """Contract, checked on every platform.

    Paths that survived the block come back; paths that were renamed away do
    not, because re-adding a path that no longer exists is how a watcher
    accumulates dead entries and starts logging about them.
    """
    going = tmp_path / "sub-old" / "ses-pre" / "eeg"
    going.mkdir(parents=True)
    keeper = tmp_path / "keep-me"
    keeper.mkdir()

    host = QWidget()
    host.show()
    qtbot.addWidget(host)
    watcher = QFileSystemWatcher(host)
    watcher.addPaths([str(going), str(keeper)])

    with watchers_released():
        if os.name == "nt":
            assert watcher.directories() == [], "watches must be off inside"
        going.rmdir()                      # this one goes away

    restored = set(watcher.directories())
    assert str(keeper) in restored
    assert str(going) not in restored


def test_it_is_a_no_op_without_an_application(monkeypatch):
    """Never raises, whatever the GUI state — a rename must not depend on it."""
    monkeypatch.setattr(
        "bidsmgr.gui.fs_watch.QApplication.instance", staticmethod(lambda: None),
    )
    assert live_watchers() == []
    with watchers_released():
        pass
