"""Two viewers side by side, driven as one.

Built for the defacing before-and-after view and wanted for everything else:
a preprocessed volume against its raw input, two echoes, this run against
last week's, a derivative against its scan.

"Driven as one" covers the crosshair, the slice, the plane, the frame, the
display options, and on the 3-D side the camera, every effect parameter and
the cut plane INCLUDING which side is kept (two renders cut from opposite
sides look like a difference in the data). Contrast is linked too when the
two images are versions of the same data (defacing), and optional otherwise:
two different contrasts of one head want their own windows.

The crosshair travels as MILLIMETRES, out through one image's affine and in
through the other's, so a cropped image, another resolution, another storage
order and another modality all follow to the same anatomy; the note still
says when the grids differ.

While synced there is ONE toolbar: two identical toolbars driving one state
is the same control drawn twice. Unsyncing gives the right image its own.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget,
)

from ..widgets.primitives import ElidedLabel
from .link import ViewerLink
from .viewer import Viewer

log = logging.getLogger(__name__)


class ComparePanes(QWidget):
    """Two viewers, one set of controls, and a switch to separate them."""

    #: Both images are read and their grids are known.
    both_loaded = pyqtSignal()

    def __init__(self, *, left_title: str = "Left", right_title: str = "Right",
                 link_contrast: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        #: Whether the two share a voxel grid. Not whether they can be linked
        #: (they always can, through the world); only what the note says.
        self._same_grid = False
        self._loaded: set[str] = set()

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
            "together.\n\nOff: each image gets its own controls."
        )
        self.link.toggled.connect(self._on_link_toggled)
        row.addWidget(self.link)
        self.contrast = QCheckBox("Same contrast")
        self.contrast.setChecked(link_contrast)
        self.contrast.setToolTip(
            "Give both images the same window, gamma and colour map. Right for "
            "two versions of the same image (before and after defacing); "
            "leave off for two different contrasts."
        )
        self.contrast.toggled.connect(self._on_contrast_toggled)
        row.addWidget(self.contrast)
        self.note = QLabel("Loading...")
        self.note.setObjectName("dlg-hint")
        self.note.setWordWrap(True)
        row.addWidget(self.note, 1)
        self.footer = row
        outer.addLayout(row)

        self._link = ViewerLink(self.left, self.right, window=link_contrast)
        self._link.enabled = False
        self.left.loaded.connect(lambda _p: self._on_one_loaded("left"))
        self.right.loaded.connect(lambda _p: self._on_one_loaded("right"))
        self.right.set_toolbar_visible(False)

    def _column(self, split: QSplitter, title: str):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        caption = ElidedLabel(title)
        caption.setObjectName("section-caption")
        lay.addWidget(caption)
        viewer = Viewer()
        # Explicitly shrinkable: the window must not be held wider than the
        # sum of two toolbars, nor grow itself when the images load.
        viewer.setMinimumWidth(260)
        lay.addWidget(viewer, 1)
        split.addWidget(box)
        return caption, viewer

    # -- loading -----------------------------------------------------------

    def show_images(self, left: Path, right: Path, *, root: Optional[Path] = None,
                    left_title: str = "", right_title: str = "") -> None:
        """Open both. Reading happens on each viewer's own worker."""
        if left_title:
            self._left_caption.setText(left_title)
        if right_title:
            self._right_caption.setText(right_title)
        self._same_grid = False
        self._loaded.clear()
        self._link.enabled = False
        self.link.setEnabled(False)
        self.note.setText("Loading...")
        # Warm nibabel on THIS thread first. Both viewers read on their own
        # QThread and import nibabel lazily; two workers racing one COLD
        # import can deadlock the import lock and abort the process.
        import nibabel  # noqa: F401

        self.left.set_file(Path(left), root)
        self.right.set_file(Path(right), root)

    def _on_one_loaded(self, side: str) -> None:
        self._loaded.add(side)
        if self._loaded != {"left", "right"}:
            return
        a = self.left.presenter.source
        b = self.right.presenter.source
        if a is None or b is None:
            return
        self.link.setEnabled(True)
        self._same_grid = tuple(a.spatial) == tuple(b.spatial)
        self.note.setText("" if self._same_grid else (
            f"Different sizes ({tuple(a.spatial)} and {tuple(b.spatial)}). The "
            "crosshair is matched by position in the scanner rather than by "
            "voxel, so it still points at the same place in both."
        ))
        self._link.enabled = self.link.isChecked()
        if self._link.enabled:
            # Start both at the same place and the same view, so the first
            # thing on screen is a comparison.
            self._link.sync_now(self.left)
        self.both_loaded.emit()

    # -- linking -----------------------------------------------------------

    def _on_link_toggled(self, on: bool) -> None:
        self.right.set_toolbar_visible(not on)
        self._link.enabled = bool(on) and self._loaded == {"left", "right"}
        if self._link.enabled:
            self._link.sync_now(self.left)

    def _on_contrast_toggled(self, on: bool) -> None:
        self._link.window = bool(on)
        if self._link.enabled and on:
            self._link.sync_now(self.left)

    @property
    def viewer_link(self) -> ViewerLink:
        return self._link

    # -- closing -----------------------------------------------------------

    def stop(self) -> None:
        """Abandon in-flight reads before anything is destroyed: destroying a
        RUNNING QThread aborts the process."""
        for viewer in (self.left, self.right):
            try:
                viewer.stop_loading()
            except RuntimeError:
                pass

    def repaint_for_palette(self, pal: dict) -> None:
        for viewer in (self.left, self.right):
            viewer.repaint_for_palette(pal)


__all__ = ["ComparePanes"]
