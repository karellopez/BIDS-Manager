"""Comparing any two NIfTI images.

The side-by-side viewer was built for defacing and is not about defacing: raw
against preprocessed, two echoes, a derivative against the scan it came from,
one subject against another. What is tested here is that it is genuinely
general, that two images which do NOT match still open, and that the sync
covers everything the defacing view covers, because it is the same widget.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from bidsmgr.deface.engines import TEMPLATE  # noqa: E402
from bidsmgr.gui.compare_dialog import (  # noqa: E402
    CompareDialog,
    is_nifti,
    open_compare,
)

pytestmark = pytest.mark.gui


@pytest.fixture
def images(tmp_path: Path):
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"})
    )
    a = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    b = root / "sub-01" / "anat" / "sub-01_T2w.nii.gz"
    shutil.copyfile(TEMPLATE, a)
    shutil.copyfile(TEMPLATE, b)
    return root, a, b


@pytest.fixture
def opened(qtbot):
    made = []

    def _open(*args, **kwargs):
        dlg = CompareDialog(*args, **kwargs)
        qtbot.addWidget(dlg)
        made.append(dlg)
        return dlg

    yield _open
    for dlg in made:
        dlg.close()


def _wait(qtbot, dlg):
    qtbot.waitUntil(
        lambda: dlg._panes.left.crosshair_voxel() is not None
        and dlg._panes.right.crosshair_voxel() is not None,
        timeout=20_000,
    )


@pytest.mark.parametrize("name,expected", [
    ("a.nii", True), ("a.nii.gz", True), ("A.NII.GZ", True),
    ("a.json", False), ("a.tsv", False), ("a.niftii", False),
])
def test_what_counts_as_an_image(name, expected):
    assert is_nifti(Path(name)) is expected


def test_two_images_open_side_by_side(qtbot, opened, images):
    root, a, b = images
    dlg = opened(a, b, root=root)
    _wait(qtbot, dlg)

    assert dlg._panes.left.current_file() == a
    assert dlg._panes.right.current_file() == b


def test_it_opens_with_nothing_chosen(qtbot, opened, images):
    """A Tools entry has no selection to work from, so it must still open."""
    root, _a, _b = images
    dlg = opened(root=root)

    assert dlg._panes.left.current_file() is None
    assert "nothing chosen" in dlg._left_label.text()
    assert "nothing chosen" in dlg._right_label.text()


def test_one_selected_image_lands_on_the_left(qtbot, opened, images):
    root, a, _b = images
    dlg = opened(a, root=root)
    assert dlg._panes.left.current_file() is None, (
        "it should wait for the second image rather than show half a pair"
    )
    assert a.name in dlg._left_label.text()


def test_captions_are_dataset_relative_when_there_is_a_dataset(
    qtbot, opened, images,
):
    root, a, b = images
    dlg = opened(a, b, root=root)
    assert dlg._left_label.text() == "sub-01/anat/sub-01_T1w.nii.gz"


def test_captions_fall_back_to_the_name_outside_a_dataset(
    qtbot, opened, images, tmp_path,
):
    _root, a, b = images
    stray = tmp_path / "elsewhere.nii.gz"
    shutil.copyfile(TEMPLATE, stray)
    dlg = opened(a, stray, root=None)
    assert dlg._right_label.text() == "elsewhere.nii.gz"


def test_images_of_different_shapes_still_open(qtbot, opened, images, tmp_path):
    """Only the crosshair link is meaningless across shapes, not the view."""
    nib = pytest.importorskip("nibabel")
    root, a, _b = images
    cropped = root / "sub-01" / "anat" / "sub-01_desc-crop_T1w.nii.gz"
    nib.save(nib.load(str(TEMPLATE)).slicer[:, :, 5:], str(cropped))

    dlg = opened(a, cropped, root=root)
    _wait(qtbot, dlg)
    qtbot.waitUntil(lambda: dlg._panes.note.text() != "Loading…", timeout=20_000)

    assert dlg._panes.link.isChecked(), "sync was switched off entirely"
    assert not dlg._panes._linkable
    assert "different sizes" in dlg._panes.note.text()

    # The plane still follows.
    dlg._panes.left._shortcut_orientation(0)
    assert dlg._panes.right.view_state()["orientation"] == 0


def test_the_crosshair_is_linked_when_the_shapes_match(qtbot, opened, images):
    root, a, b = images
    dlg = opened(a, b, root=root)
    _wait(qtbot, dlg)
    qtbot.waitUntil(lambda: dlg._panes.link.isEnabled(), timeout=20_000)

    moved = list(dlg._panes.left.crosshair_voxel())
    moved[0] = max(0, moved[0] - 5)
    dlg._panes.left.set_crosshair_voxel(moved)
    dlg._panes.left._broadcast_crosshair()

    assert dlg._panes.right.crosshair_voxel() == moved


def test_only_one_toolbar_while_synced(qtbot, opened, images):
    root, a, b = images
    dlg = opened(a, b, root=root)

    assert not dlg._panes.right._toolbar.isVisibleTo(dlg)
    dlg._panes.link.setChecked(False)
    assert dlg._panes.right._toolbar.isVisibleTo(dlg)


def test_closing_mid_load_stops_both_reads(qtbot, opened, images):
    """A running loader QThread destroyed with its pane aborts the process."""
    root, a, b = images
    dlg = opened(a, b, root=root)
    assert dlg._panes.left._loader is not None or dlg._panes.right._loader is not None

    dlg.close()

    assert dlg._panes.left._loader is None
    assert dlg._panes.right._loader is None


def test_open_compare_takes_whatever_was_selected(qtbot, images):
    root, a, b = images
    dlg = open_compare(None, [a, b], root=root)
    qtbot.addWidget(dlg)
    assert (dlg._left, dlg._right) == (a, b)
    dlg.close()


def test_open_compare_ignores_things_that_are_not_images(qtbot, images):
    root, a, _b = images
    sidecar = root / "dataset_description.json"
    dlg = open_compare(None, [sidecar, a], root=root)
    qtbot.addWidget(dlg)
    assert dlg._left == a and dlg._right is None
    dlg.close()


def test_the_window_fits_the_screen(qtbot, opened, images):
    from PyQt6.QtWidgets import QApplication

    root, a, b = images
    dlg = opened(a, b, root=root)
    screen = dlg.screen() or QApplication.primaryScreen()
    if screen is None:
        pytest.skip("no screen")
    assert dlg.width() <= screen.availableGeometry().width()
