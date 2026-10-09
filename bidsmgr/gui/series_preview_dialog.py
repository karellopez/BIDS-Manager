"""A DICOM series, converted on the spot and shown before the real conversion.

From the Converter's Properties panel: one row's series goes through the
same dcm2niix call the probe makes, into a temporary folder, and opens in the
volume viewer. Whether the scout really is a scout, whether the "T1" is the
T1, whether a series that splits into echoes or magnitude and phase splits
the way the plan says, is a look rather than a guess. The folder is removed
when the window closes.

The window opens on the CLICK, pending, with a spinner while the series is
converted on a thread (``set_images`` fills it, ``set_failed`` says why
not). Opened when the conversion finished instead, it came up seconds after
the click, and the window system left it behind the main window.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget,
)

from .viz import Viewer


class SeriesPreviewDialog(QDialog):
    def __init__(self, images: Optional[list[Path]], work_dir: Optional[Path], title: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from .widgets import BusySpinner

        self.setWindowTitle(f"Preview: {title}")
        self.resize(1000, 680)
        self._work_dir = Path(work_dir) if work_dir else None
        self.images: list[Path] = []
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QWidget()
        bar.setObjectName("toolbar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(12, 6, 12, 6)
        note = QLabel("Converted now for a look, the way the conversion will. "
                      "Nothing is written to the dataset.")
        note.setObjectName("sidecar-footer-summary")
        note.setWordWrap(True)
        h.addWidget(note, 1)
        self.choice = QComboBox()
        self.choice.setObjectName("ent-input")
        self.choice.setVisible(False)
        self.choice.setToolTip("This series made several images (echoes, magnitude and "
                               "phase, ...): pick one")
        self.choice.currentIndexChanged.connect(self._show)
        h.addWidget(self.choice)
        v.addWidget(bar)

        # Pending: a spinner and what is happening, centred where the image
        # will be.
        self._stack = QStackedWidget()
        pending = QWidget()
        pending.setObjectName("pane")
        pl = QVBoxLayout(pending)
        pl.addStretch(1)
        self.spinner = BusySpinner()
        self.spinner.set_busy(True, message="Converting the series for a look...")
        pl.addWidget(self.spinner, 0, Qt.AlignmentFlag.AlignHCenter)
        self.status = QLabel("")
        self.status.setObjectName("pane-hint")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setVisible(False)
        pl.addWidget(self.status)
        pl.addStretch(1)
        self._stack.addWidget(pending)
        self.viewer = Viewer(kind="volume", parent=self)
        self._stack.addWidget(self.viewer)
        v.addWidget(self._stack, 1)
        if images:
            self.set_images(images, work_dir)

    def is_pending(self) -> bool:
        return self._stack.currentIndex() == 0

    def set_images(self, images: list[Path], work_dir: Optional[Path]) -> None:
        """The converted images: show the first, offer the others."""
        self._work_dir = Path(work_dir) if work_dir else self._work_dir
        self.images = [Path(p) for p in images]
        self.choice.blockSignals(True)
        self.choice.clear()
        for path in self.images:
            self.choice.addItem(path.name, str(path))
        self.choice.blockSignals(False)
        self.choice.setVisible(len(self.images) > 1)
        self.spinner.set_busy(False)
        self._stack.setCurrentWidget(self.viewer)
        self._show(0)

    def set_failed(self, message: str) -> None:
        """The series could not be converted: say why, in the window."""
        self.spinner.set_busy(False)
        self.spinner.setVisible(False)
        self.status.setText(f"The series could not be converted: {message}")
        self.status.setVisible(True)

    def _show(self, index: int) -> None:
        if 0 <= index < len(self.images):
            self.viewer.set_file(self.images[index], None)

    def done(self, result: int) -> None:  # noqa: D401 - Qt override
        self.viewer.stop_loading()
        if self._work_dir is not None:
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._work_dir = None
        super().done(result)


__all__ = ["SeriesPreviewDialog"]
