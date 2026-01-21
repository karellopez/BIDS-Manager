"""Shared PyQt widgets for visualization-related layouts."""

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QScrollArea


class ShrinkableScrollArea(QScrollArea):
    """``QScrollArea`` variant that allows the parent splitter to shrink."""

    def minimumSizeHint(self) -> QSize:  # noqa: D401 - Qt override
        return QSize(0, 0)

    def sizeHint(self) -> QSize:  # noqa: D401 - Qt override
        hint = super().sizeHint()
        return QSize(max(0, hint.width()), max(0, hint.height()))
