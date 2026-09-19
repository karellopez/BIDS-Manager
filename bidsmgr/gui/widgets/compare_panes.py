"""Two NIfTI viewers side by side, driven as one.

Built for the defacing before-and-after view and then wanted for everything
else, which is the usual sign that it belonged in its own widget. Comparing
two images is not a defacing question: a preprocessed volume against its raw
input, two echoes, this run against last week's, a derivative against the scan
it came from. The only thing defacing contributed was knowing which two files
to open.

What "driven as one" covers, because a partial sync is worse than none: the
crosshair, the slice, the plane, the volume index, RAS and radiological, the
orientation labels, and on the 3-D side the camera, every effect parameter,
and the cut plane INCLUDING which side of it is kept. Two renders cut from
opposite sides look like a difference in the data.

Two things it deliberately does not do.

**It does not link the crosshair between images of different shapes.** A
crosshair is a voxel index, and the same index is a different place in a
cropped or resampled image. Everything else still follows, and the reason is
on screen rather than left to be discovered.

**It shows ONE toolbar while synced.** Two identical toolbars driving one
shared state is not a choice the user has, it is the same control drawn twice.
Unsyncing gives the right-hand image its own.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .nifti_viewer_pane import NiftiViewerPane
from .primitives import ElidedLabel

log = logging.getLogger(__name__)


class ComparePanes(QWidget):
    """Two viewers, one set of controls, and a switch to separate them."""

    #: Both images have finished loading and their shapes are known.
    both_loaded = pyqtSignal()

    def __init__(
        self,
        *,
        left_title: str = "Left",
        right_title: str = "Right",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._linkable = False
        self._syncing = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self._left_caption, self.left = self._column(split, left_title)
        self._right_caption, self.right = self._column(split, right_title)
        split.setSizes([560, 560])
        outer.addWidget(split, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.link = QCheckBox("Sync the two views")
        self.link.setChecked(True)
        self.link.setEnabled(False)
        self.link.setToolTip(
            "On: one set of controls drives both images, and the crosshair, "
            "slice, plane, volume, 3-D camera, effects and cut plane stay "
            "together, which is the only way a comparison means anything.\n\n"
            "Off: each image gets its own controls, for when you want to look "
            "at one of them on its own."
        )
        self.link.toggled.connect(self._on_link_toggled)
        row.addWidget(self.link)
        self.note = QLabel("Loading…")
        self.note.setObjectName("muted-note")
        self.note.setWordWrap(True)
        row.addWidget(self.note, 1)
        self.footer = row
        outer.addLayout(row)

        for pane in (self.left, self.right):
            pane.loaded.connect(self._on_one_loaded)
        self.left.crosshair_moved.connect(
            lambda voxel: self._mirror_crosshair(self.right, voxel)
        )
        self.right.crosshair_moved.connect(
            lambda voxel: self._mirror_crosshair(self.left, voxel)
        )
        self.left.view_changed.connect(
            lambda state: self._mirror_view(self.right, state)
        )
        self.right.view_changed.connect(
            lambda state: self._mirror_view(self.left, state)
        )
        self.right.set_toolbar_visible(False)

    # -- building ---------------------------------------------------------

    def _column(self, split: QSplitter, title: str):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        caption = ElidedLabel(title)
        caption.setObjectName("section-caption")
        lay.addWidget(caption)
        pane = NiftiViewerPane()
        # Explicitly shrinkable. A viewer pane's natural minimum is its widest
        # control row, which is fine for the single pane in the Editor and not
        # fine for two: the window could not be dragged narrower than the sum,
        # and it grew itself to that width when the images loaded.
        pane.setMinimumWidth(260)
        lay.addWidget(pane, 1)
        split.addWidget(box)
        return caption, pane

    # -- loading ----------------------------------------------------------

    def show_images(
        self,
        left: Path,
        right: Path,
        *,
        root: Optional[Path] = None,
        left_title: str = "",
        right_title: str = "",
    ) -> None:
        """Bind both viewers. Reading happens on their own threads."""
        if left_title:
            self._left_caption.setText(left_title)
        if right_title:
            self._right_caption.setText(right_title)
        self._linkable = False
        self.link.setEnabled(False)
        self.note.setText("Loading…")

        # Warm nibabel on THIS thread before starting either load. Both panes
        # read on their own QThread and import nibabel lazily, so this is the
        # only place in the application where two worker threads race the same
        # COLD import. Python's import lock does not survive that: one thread
        # blocks in `importlib._bootstrap.acquire` while the other is still
        # executing the module, and the process aborts.
        import nibabel  # noqa: F401

        self.left.set_file(Path(left), root)
        self.right.set_file(Path(right), root)

    def _on_one_loaded(self, _path: Path) -> None:
        left, right = self._shape(self.left), self._shape(self.right)
        if left is None or right is None:
            return

        self.link.setEnabled(True)
        self._linkable = left[:3] == right[:3]
        if self._linkable:
            self.note.setText("")
            # Start both at the same place, so the first thing on screen is a
            # comparison rather than two unrelated slices.
            self._mirror_crosshair(self.right, self.left.crosshair_voxel())
        else:
            # Sync stays ON. The plane, the 3-D camera and the effects are
            # still worth sharing; only the crosshair is meaningless across
            # shapes, so only the crosshair is dropped.
            self.note.setText(
                f"The images are different sizes ({left[:3]} and {right[:3]}), "
                "so the crosshair is not linked: the same voxel is not the "
                "same place. Everything else still follows."
            )
        self.both_loaded.emit()

    @staticmethod
    def _shape(pane: NiftiViewerPane):
        data = getattr(pane, "_data", None)
        return None if data is None else tuple(data.shape)

    # -- linking ----------------------------------------------------------

    def _mirror_crosshair(self, target: NiftiViewerPane, voxel) -> None:
        if self._syncing or not self._linkable or not self.link.isChecked():
            return
        self._syncing = True
        try:
            target.set_crosshair_voxel(voxel)
        finally:
            self._syncing = False

    def _mirror_view(self, target: NiftiViewerPane, state: dict) -> None:
        if self._syncing or not self.link.isChecked():
            return
        self._syncing = True
        try:
            target.apply_view_state(state, with_crosshair=self._linkable)
        finally:
            self._syncing = False

    def _on_link_toggled(self, on: bool) -> None:
        """One toolbar when linked, one per image when not."""
        self.right.set_toolbar_visible(not on)
        if on:
            # Re-linking adopts the left view, so the two are comparable again
            # rather than staying however they drifted apart.
            self._mirror_view(self.right, self.left.view_state())

    # -- closing ----------------------------------------------------------

    def stop(self) -> None:
        """Abandon any in-flight read, before anything is destroyed.

        Each pane's loader is a QThread parented to it, so a pane destroyed
        mid-read destroys a RUNNING QThread, and Qt answers that by aborting
        the process rather than raising.
        """
        for pane in (self.left, self.right):
            try:
                pane.stop_loading()
            except RuntimeError:
                pass


__all__ = ["ComparePanes"]
