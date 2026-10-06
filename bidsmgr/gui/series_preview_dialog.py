"""A DICOM series, converted on the spot and shown before the real conversion.

From the Converter's Properties panel: one row's series goes through the
same dcm2niix call the probe makes, into a temporary folder, and opens in the
volume viewer. Whether the scout really is a scout, whether the "T1" is the
T1, whether a series that splits into echoes or magnitude and phase splits
the way the plan says, is a look rather than a guess. The folder is removed
when the window closes.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .viz import Viewer


class SeriesPreviewDialog(QDialog):
    def __init__(self, images: list[Path], work_dir: Optional[Path], title: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Preview: {title}")
        self.resize(1000, 680)
        self._work_dir = Path(work_dir) if work_dir else None
        self.images = [Path(p) for p in images]
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
        for path in self.images:
            self.choice.addItem(path.name, str(path))
        self.choice.setVisible(len(self.images) > 1)
        self.choice.setToolTip("This series made several images (echoes, magnitude and "
                               "phase, ...): pick one")
        self.choice.currentIndexChanged.connect(self._show)
        h.addWidget(self.choice)
        v.addWidget(bar)
        self.viewer = Viewer(kind="volume", parent=self)
        v.addWidget(self.viewer, 1)
        self._show(0)

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
