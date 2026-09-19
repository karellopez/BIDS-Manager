"""Compare any two NIfTI images, driven as one.

The side-by-side viewer was built for defacing, where the two images are
chosen for you: the copy before, the file now. It turned out to be the thing
people wanted for everything else, and the only part that was about defacing
was knowing which two files to open.

So this is the same viewer with a pair of file pickers. Raw against
preprocessed, two echoes, a derivative against the scan it came from, this
week's run against last week's, one subject against another. Nothing here
knows or cares which.

It does not require the images to match. Different shapes, different
resolutions and different orientations all open; the crosshair link is the
only thing that switches off, because a voxel index is a different place in a
differently shaped image, and the reason says so on screen.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .deface_compare import size_to_screen
from .widgets.compare_panes import ComparePanes
from .widgets.primitives import ElidedLabel

log = logging.getLogger(__name__)

NIFTI_FILTER = "NIfTI images (*.nii *.nii.gz);;All files (*)"


def is_nifti(path) -> bool:
    name = Path(path).name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


class CompareDialog(QDialog):
    """Two NIfTI images side by side, with everything synchronised."""

    def __init__(
        self,
        left: Optional[Path] = None,
        right: Optional[Path] = None,
        *,
        root: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root) if root else None
        self._left: Optional[Path] = Path(left) if left else None
        self._right: Optional[Path] = Path(right) if right else None

        self.setWindowTitle("Compare images")
        self.setSizeGripEnabled(True)
        size_to_screen(self, 1180, 720)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        header = QLabel("Compare two images")
        header.setObjectName("dialog-title")
        outer.addWidget(header)

        subtitle = QLabel(
            "One set of controls drives both: crosshair, slice, plane, "
            "volume, 3-D camera, effects and the cut plane. Images of "
            "different sizes still open; only the crosshair stops being "
            "linked, because the same voxel is then a different place."
        )
        subtitle.setObjectName("dialog-subtitle")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        pickers = QHBoxLayout()
        pickers.setSpacing(8)
        self._left_label, left_btn = self._picker(pickers, "Left", self._pick_left)
        self._right_label, right_btn = self._picker(
            pickers, "Right", self._pick_right
        )
        outer.addLayout(pickers)
        self._left_btn, self._right_btn = left_btn, right_btn

        self._panes = ComparePanes()
        outer.addWidget(self._panes, 1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        self._panes.footer.addWidget(close)

        self._refresh_labels()
        if self._left and self._right:
            self._show()

    # -- picking ----------------------------------------------------------

    def _picker(self, row: QHBoxLayout, side: str, slot):
        button = QPushButton(f"Choose {side.lower()}…")
        button.setObjectName("tb-btn")
        button.clicked.connect(slot)
        row.addWidget(button)
        label = ElidedLabel("")
        label.setObjectName("dlg-hint")
        row.addWidget(label, 1)
        return label, button

    def _ask(self, side: str) -> Optional[Path]:
        start = ""
        for candidate in (self._left, self._right, self._root):
            if candidate:
                start = str(Path(candidate).parent if Path(candidate).is_file()
                            else candidate)
                break
        chosen, _ = QFileDialog.getOpenFileName(
            self, f"Choose the {side} image", start, NIFTI_FILTER,
        )
        return Path(chosen) if chosen else None

    def _pick_left(self) -> None:
        picked = self._ask("left")
        if picked:
            self._left = picked
            self._after_pick()

    def _pick_right(self) -> None:
        picked = self._ask("right")
        if picked:
            self._right = picked
            self._after_pick()

    def _after_pick(self) -> None:
        self._refresh_labels()
        if self._left and self._right:
            self._show()

    def _refresh_labels(self) -> None:
        for label, path in (
            (self._left_label, self._left), (self._right_label, self._right),
        ):
            label.setText(self._describe(path) if path else "nothing chosen yet")

    def _describe(self, path: Path) -> str:
        """The dataset-relative path when there is one, else the name."""
        if self._root:
            try:
                return Path(path).resolve().relative_to(
                    Path(self._root).resolve()
                ).as_posix()
            except ValueError:
                pass
        return Path(path).name

    # -- showing ----------------------------------------------------------

    def _show(self) -> None:
        self._panes.show_images(
            self._left, self._right, root=self._root,
            left_title=f"Left: {self._describe(self._left)}",
            right_title=f"Right: {self._describe(self._right)}",
        )

    # -- closing ----------------------------------------------------------

    def done(self, result: int) -> None:  # noqa: D102 - Qt signature
        self._panes.stop()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        self._panes.stop()
        super().closeEvent(event)


def open_compare(
    parent, targets: Sequence[Path], *, root: Optional[Path] = None,
) -> CompareDialog:
    """Open the dialog on whatever the caller had selected.

    Two images picked in the tree open straight away. One opens on the left
    with the right still to choose, which is the common case: you are looking
    at something and want to put another image beside it.
    """
    images = [Path(t) for t in targets if is_nifti(t) and Path(t).is_file()]
    dlg = CompareDialog(
        images[0] if images else None,
        images[1] if len(images) > 1 else None,
        root=root, parent=parent,
    )
    return dlg


__all__ = ["CompareDialog", "is_nifti", "open_compare"]
