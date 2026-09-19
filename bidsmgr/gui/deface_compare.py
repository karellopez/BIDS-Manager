"""Look at the image before defacing and after it, side by side.

Every defacing tool ends its documentation by telling you to inspect the
result, and then leaves you to find your own viewer. That advice is not
decoration: the two ways defacing goes wrong are opposite, and both are plain
to see and impossible to reason about. Too little removed and the participant
is still identifiable, which is the failure the user was trying to avoid. Too
much removed and the cerebellum or the front of the brain is gone, which
quietly ruins the analysis and survives every validator.

So this is two :class:`NiftiViewerPane` s, the undefaced copy on the left and
what is in the dataset on the right, with one crosshair between them. Moving it
in one pane moves it in the other, because comparing two images at different
slices tells you nothing.

The crosshair is a VOXEL INDEX, so linking is only honest when the two images
have the same shape. ``allineate-robust`` crops the neck, so its output does
not, and the link is switched off and said so rather than silently showing two
different places.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..deface import compare, status
from .widgets.nifti_viewer_pane import NiftiViewerPane
from .widgets.primitives import ElidedLabel

log = logging.getLogger(__name__)


class DefaceCompareDialog(QDialog):
    """Before and after, for one image, with a linked crosshair."""

    def __init__(self, root: Path, rel: str, parent=None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._rel = str(rel).replace("\\", "/")
        self._original = compare.original_for(self._root, self._rel)
        # Set once both panes have loaded and their shapes are known.
        self._linkable = False
        self._syncing = False

        self.setWindowTitle(f"Before and after: {Path(self._rel).name}")
        self.setSizeGripEnabled(True)
        self._size_to_screen(1180, 720)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        self._header = QLabel()
        self._header.setObjectName("dialog-title")
        self._header.setWordWrap(True)
        outer.addWidget(self._header)

        self._subhead = QLabel()
        self._subhead.setObjectName("dialog-subtitle")
        self._subhead.setWordWrap(True)
        outer.addWidget(self._subhead)

        if self._original is None:
            self._build_nothing_to_compare(outer)
            return

        self._build_panes(outer)

    # -- the ordinary case ------------------------------------------------

    def _build_panes(self, outer: QVBoxLayout) -> None:
        eng = status.defaced_by_us(
            status.read_sidecar(status.sidecar_for(self._root / self._rel))
        )
        self._header.setText(self._rel)
        self._subhead.setText(
            f"Left: the image before defacing, from {self._original.description}. "
            f"Right: what is in the dataset now"
            + (f", defaced with {eng.label}." if eng else ".")
        )

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self._before = self._titled(split, "Before", self._original.path)
        self._after = self._titled(split, "After", self._root / self._rel)
        split.setSizes([560, 560])
        outer.addWidget(split, 1)

        row = QHBoxLayout()
        self._link = QCheckBox("Sync the two views")
        self._link.setChecked(True)
        self._link.setEnabled(False)
        self._link.setToolTip(
            "On: one set of controls drives both images, and the crosshair, "
            "slice, orientation, 3-D mode and volume stay together, which is "
            "the only way a comparison means anything.\n\n"
            "Off: each image gets its own controls, for when you want to look "
            "at one of them on its own."
        )
        self._link.toggled.connect(self._on_link_toggled)
        row.addWidget(self._link)
        self._link_note = QLabel("Loading…")
        self._link_note.setObjectName("muted-note")
        self._link_note.setWordWrap(True)
        row.addWidget(self._link_note, 1)
        self._restore = QPushButton("Put the face back")
        self._restore.setObjectName("tb-btn")
        self._restore.setEnabled(False)
        self._restore.setToolTip(
            "Restore this image from the undefaced copy on the left. The copy "
            "is kept, so this can be defaced again afterwards."
        )
        self._restore.clicked.connect(self._on_restore)
        row.addWidget(self._restore)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        outer.addLayout(row)

        self._before.crosshair_moved.connect(self._from_before)
        self._after.crosshair_moved.connect(self._from_after)
        self._before.view_changed.connect(self._view_from_before)
        self._after.view_changed.connect(self._view_from_after)
        self._before.loaded.connect(self._on_one_loaded)
        self._after.loaded.connect(self._on_one_loaded)
        # Linked to start with, so only ONE toolbar is on screen. Two identical
        # toolbars driving one shared state is not a choice, it is the same
        # control drawn twice.
        self._after.set_toolbar_visible(False)

        # Warm nibabel on THIS thread before starting either load. Both panes
        # read on their own QThread, and `_load_nifti` imports nibabel lazily,
        # so this is the only place in the app where two worker threads race
        # the same cold import. Python's import lock does not survive that:
        # one thread blocks in `importlib._bootstrap.acquire` while the other
        # is still executing the module, and the process aborts. Nothing is
        # lost by importing it here, since a dialog that exists to show two
        # NIfTIs is going to need it either way.
        import nibabel  # noqa: F401

        self._before.set_file(self._original.path, self._root)
        self._after.set_file(self._root / self._rel, self._root)

    def _size_to_screen(self, want_w: int, want_h: int) -> None:
        """Open at the requested size, or at what the screen actually has.

        A fixed 1180x720 is bigger than the work area on a 13-inch laptop
        once the dock and menu bar are taken out, and a dialog that opens
        larger than the screen cannot be resized back by dragging an edge
        that is off the display.
        """
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(want_w, want_h)
            return
        available = screen.availableGeometry()
        self.resize(
            min(want_w, int(available.width() * 0.92)),
            min(want_h, int(available.height() * 0.92)),
        )

    def _titled(self, split: QSplitter, title: str, path: Path) -> NiftiViewerPane:
        """One labelled column of the splitter, returning its viewer."""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        cap = ElidedLabel(f"{title}: {path.name}")
        cap.setObjectName("section-caption")
        lay.addWidget(cap)
        pane = NiftiViewerPane()
        # Explicitly shrinkable. A viewer pane's natural minimum is its
        # widest control row, which is fine for the one pane in the Editor
        # and not fine for two of them side by side: the dialog could not be
        # made narrower than the sum, and it grew itself to that width as
        # soon as the images loaded. Setting a real minimum overrides the
        # hint, and the toolbar scrolls sideways instead.
        pane.setMinimumWidth(260)
        lay.addWidget(pane, 1)
        split.addWidget(box)
        return pane

    # -- linking ----------------------------------------------------------

    def _on_one_loaded(self, _path: Path) -> None:
        """Decide whether linking is meaningful, once both shapes are known."""
        before, after = self._shape(self._before), self._shape(self._after)
        if before is None or after is None:
            return
        self._restore.setEnabled(True)
        self._link.setEnabled(True)
        self._linkable = before[:3] == after[:3]
        if self._linkable:
            self._link_note.setText("")
            # Start both at the same place, so the first thing on screen is a
            # comparison rather than two unrelated slices.
            self._from_before(self._before.crosshair_voxel())
            return
        # Sync stays ON: the view mode, orientation and 3-D controls are still
        # worth sharing. Only the crosshair is dropped, because a voxel index
        # is a different place in a cropped image.
        self._link_note.setText(
            f"The images are different sizes ({before[:3]} and {after[:3]}), "
            "so the crosshair is not linked: the same voxel is not the same "
            "place. The neck-cropping engine does this by design. Everything "
            "else still follows."
        )

    @staticmethod
    def _shape(pane: NiftiViewerPane) -> Optional[tuple]:
        data = getattr(pane, "_data", None)
        return None if data is None else tuple(data.shape)

    def _from_before(self, voxel) -> None:
        self._mirror(self._after, voxel)

    def _from_after(self, voxel) -> None:
        self._mirror(self._before, voxel)

    def _mirror(self, target: NiftiViewerPane, voxel) -> None:
        # ``set_crosshair_voxel`` does not emit, but the repaint it triggers
        # can still re-enter through a queued signal, so the guard stays.
        if self._syncing or not self._linkable or not self._link.isChecked():
            return
        self._syncing = True
        try:
            target.set_crosshair_voxel(voxel)
        finally:
            self._syncing = False

    def _view_from_before(self, state: dict) -> None:
        self._mirror_view(self._after, state)

    def _view_from_after(self, state: dict) -> None:
        self._mirror_view(self._before, state)

    def _mirror_view(self, target: NiftiViewerPane, state: dict) -> None:
        """Make the other pane show things the same way.

        Sync covers HOW, not WHERE, when the images are different sizes: the
        view mode and the orientation are meaningful for a cropped image, a
        voxel index is not. So the crosshair is dropped from the state and
        everything else still follows.
        """
        if self._syncing or not self._link.isChecked():
            return
        self._syncing = True
        try:
            target.apply_view_state(state, with_crosshair=self._linkable)
        finally:
            self._syncing = False

    def _on_link_toggled(self, on: bool) -> None:
        """One toolbar when linked, one per image when not."""
        self._after.set_toolbar_visible(not on)
        if not on:
            return
        # Re-linking adopts the left pane's view, so the two are immediately
        # comparable again rather than staying however they drifted apart.
        self._mirror_view(self._after, self._before.view_state())

    # -- restoring --------------------------------------------------------

    def _on_restore(self) -> None:
        """Put this one image back, from the copy being shown on the left."""
        from .deface_revert_dialog import DefaceRevertDialog

        dlg = DefaceRevertDialog(self._root, [self._root / self._rel], self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        # The image on the right is now the one on the left. Re-read it rather
        # than leaving the pane showing bytes that are no longer on disk.
        self._after.set_file(None, None)
        self._after.set_file(self._root / self._rel, self._root)
        self._restore.setEnabled(False)
        self._subhead.setText(
            f"{self._rel} has been restored. Both sides now show the same "
            "image."
        )

    # -- closing -----------------------------------------------------------

    def done(self, result: int) -> None:  # noqa: D102 - Qt signature
        # Both panes may still be reading. Their loader threads are parented
        # to them, so letting the dialog be destroyed first destroys a running
        # QThread, which aborts the process. A comparison window is closed
        # quickly by definition, so this is the normal path, not the edge.
        self._stop_panes()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        self._stop_panes()
        super().closeEvent(event)

    def _stop_panes(self) -> None:
        for name in ("_before", "_after"):
            pane = getattr(self, name, None)
            if pane is not None:
                try:
                    pane.stop_loading()
                except RuntimeError:
                    pass

    # -- the case with nothing to compare ---------------------------------

    def _build_nothing_to_compare(self, outer: QVBoxLayout) -> None:
        self._header.setText("No undefaced copy of this image")
        self._subhead.setText(compare.explain_missing(self._root, self._rel))
        outer.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        outer.addLayout(row)


__all__ = ["DefaceCompareDialog"]
