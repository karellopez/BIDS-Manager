"""Common GUI helpers and shared resources for BIDS Manager.

This module centralises paths to bundled assets and lightweight widgets that
are reused by both the converter and editor modules. Keeping these items in a
single place avoids circular imports between the dedicated converter/editor
modules introduced during the GUI refactor.
"""

from pathlib import Path

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QScrollArea

# Base directory for the package, used to resolve bundled resources regardless
# of where the calling module lives.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Paths to images bundled with the application
LOGO_FILE = PACKAGE_ROOT / "miscellaneous" / "images" / "Logo.png"
ICON_FILE = PACKAGE_ROOT / "miscellaneous" / "images" / "Icon.png"
ANCP_LAB_FILE = PACKAGE_ROOT / "miscellaneous" / "images" / "ANCP_lab.png"
KAREL_IMG_FILE = PACKAGE_ROOT / "miscellaneous" / "images" / "Karel.jpeg"
JOCHEM_IMG_FILE = PACKAGE_ROOT / "miscellaneous" / "images" / "Jochem.jpg"

# Directory used to store persistent user preferences
PREF_DIR = PACKAGE_ROOT / "user_preferences"


class ShrinkableScrollArea(QScrollArea):
    """``QScrollArea`` variant that allows the parent splitter to shrink."""

    def minimumSizeHint(self) -> QSize:  # noqa: D401 - Qt override
        return QSize(0, 0)

    def sizeHint(self) -> QSize:  # noqa: D401 - Qt override
        hint = super().sizeHint()
        return QSize(max(0, hint.width()), max(0, hint.height()))

