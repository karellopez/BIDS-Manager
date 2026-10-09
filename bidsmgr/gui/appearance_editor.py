"""Settings > Display > Appearance: the accent, a tint of the surfaces, how
icons are coloured and the file trees' colours, on top of any theme.

The values are an :class:`~bidsmgr.gui.appearance.Appearance`; this widget
only edits one. Colours shown are always the EFFECTIVE ones for the theme
chosen in the dialog (a tree kind left alone shows that theme's colour), and
an accent hard to read on that theme says so beside it.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QColorDialog, QComboBox, QFormLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QWidget,
)

from . import appearance as ap
from .theme_manager import PALETTES, theme_info

_CUSTOM = "__custom__"


def colour_dot(colour: str, side: int) -> QIcon:
    """A round dot of ``colour`` (a legend mark, never a square)."""
    app = QApplication.instance()
    ratio = app.devicePixelRatio() if app is not None else 1.0
    pm = QPixmap(max(1, int(side * ratio)), max(1, int(side * ratio)))
    pm.setDevicePixelRatio(ratio)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    # A hairline in a mid grey, so a white or black dot still shows on a
    # dialog of the same colour.
    p.setPen(QPen(QColor(128, 128, 128, 160), 1.0))
    p.setBrush(QColor(colour))
    p.drawEllipse(QPointF(side / 2, side / 2), side * 0.38, side * 0.38)
    p.end()
    return QIcon(pm)


class AppearanceEditor(QWidget):
    """Edits an :class:`Appearance`; ``changed`` on every edit."""

    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from .viz import fonts

        self._theme = "dark"
        self._custom: str = ""
        self._tree: dict[str, str] = {}
        side = fonts.px(14)
        self._side = side
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self.accent = QComboBox()
        self.accent.setToolTip("The colour of the main actions, links, focus and selection. "
                               "\"Theme's own\" keeps the theme's; a preset has a shade for "
                               "dark themes and one for light themes.")
        self.accent.currentIndexChanged.connect(self._on_accent)
        accent_row = QWidget()
        ar = QHBoxLayout(accent_row)
        ar.setContentsMargins(0, 0, 0, 0)
        ar.addWidget(self.accent, 1)
        form.addRow("Accent:", accent_row)
        self.warning = QLabel("")
        self.warning.setObjectName("dlg-hint")
        self.warning.setWordWrap(True)
        self.warning.setVisible(False)
        form.addRow("", self.warning)

        self.tint = QSlider(Qt.Orientation.Horizontal)
        self.tint.setRange(0, ap.MAX_TINT)
        self.tint.setToolTip("Mix a little of the accent into the surfaces (percent). "
                             "0 keeps the theme's own greys.")
        self.tint_value = QLabel("0 %")
        self.tint_value.setObjectName("dlg-hint")
        self.tint_value.setMinimumWidth(fonts.px(36))
        self.tint.valueChanged.connect(self._on_tint)
        tint_row = QWidget()
        tr = QHBoxLayout(tint_row)
        tr.setContentsMargins(0, 0, 0, 0)
        tr.addWidget(self.tint, 1)
        tr.addWidget(self.tint_value)
        form.addRow("Surface tint:", tint_row)

        self.icons = QComboBox()
        for key, label in ap.ICON_STYLES.items():
            self.icons.addItem(label, key)
        self.icons.setToolTip("How toolbar and action icons are coloured. Status icons "
                              "(ok, warning, error) and the file trees' icons keep their "
                              "own colours.")
        self.icons.currentIndexChanged.connect(self.changed)
        form.addRow("Icons:", self.icons)

        self.tree_buttons: dict[str, QPushButton] = {}
        self.tree_resets: dict[str, QPushButton] = {}
        for kind, (label, _token) in ap.TREE_KINDS.items():
            btn = QPushButton()
            btn.setObjectName("tb-btn")
            btn.setToolTip(f"The colour of {label.lower()} in the file trees. Click to choose.")
            btn.clicked.connect(lambda _c=False, k=kind: self._pick_tree(k))
            reset = QPushButton("Reset")
            reset.setObjectName("tb-btn-ghost")
            reset.setToolTip("Back to the theme's colour")
            reset.clicked.connect(lambda _c=False, k=kind: self._reset_tree(k))
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(btn)
            rl.addWidget(reset)
            rl.addStretch(1)
            form.addRow(f"{label}:", row)
            self.tree_buttons[kind] = btn
            self.tree_resets[kind] = reset
        self._fill_accents()

    # -- values ----------------------------------------------------------------

    def set_theme(self, theme_id: str) -> None:
        """The theme the dialog shows: defaults and warnings follow it."""
        self._theme = theme_id if theme_id in PALETTES else "dark"
        keep = self.accent.currentData()
        self._fill_accents()
        index = self.accent.findData(keep)
        self.accent.setCurrentIndex(max(0, index))
        self._sync()

    def load(self, look: ap.Appearance) -> None:
        look = look.normalised()
        self._custom = look.accent if look.accent.startswith("#") else ""
        self._tree = dict(look.tree)
        self._fill_accents()
        key = _CUSTOM if self._custom else look.accent
        self.accent.blockSignals(True)
        self.accent.setCurrentIndex(max(0, self.accent.findData(key)))
        self.accent.blockSignals(False)
        self.tint.blockSignals(True)
        self.tint.setValue(look.tint)
        self.tint.blockSignals(False)
        self.icons.blockSignals(True)
        self.icons.setCurrentIndex(max(0, self.icons.findData(look.icons)))
        self.icons.blockSignals(False)
        self._sync()

    def appearance(self) -> ap.Appearance:
        key = self.accent.currentData() or ""
        accent = self._custom if key == _CUSTOM else key
        return ap.Appearance(accent=accent, tint=self.tint.value(),
                             icons=self.icons.currentData() or "monochrome",
                             tree=dict(self._tree)).normalised()

    def effective(self) -> dict:
        """The palette these choices make on the dialog's theme."""
        info = theme_info(self._theme)
        return ap.apply(PALETTES[self._theme], self.appearance(), dark=info.dark,
                        strong=self._theme.startswith("hc-"))

    # -- building ----------------------------------------------------------------

    def _fill_accents(self) -> None:
        dark = theme_info(self._theme).dark
        own = PALETTES[self._theme]["accent"]
        self.accent.blockSignals(True)
        current = self.accent.currentData()
        self.accent.clear()
        self.accent.addItem(colour_dot(own, self._side), "Theme's own", "")
        for key, (label, on_dark, on_light) in ap.ACCENTS.items():
            self.accent.addItem(colour_dot(on_dark if dark else on_light, self._side), label, key)
        custom = self._custom or own
        self.accent.addItem(colour_dot(custom, self._side),
                            f"Custom ({custom})" if self._custom else "Custom...", _CUSTOM)
        if current is not None:
            self.accent.setCurrentIndex(max(0, self.accent.findData(current)))
        self.accent.blockSignals(False)

    def _sync(self) -> None:
        pal = self.effective()
        message = ap.accent_warning(pal)
        self.warning.setText(message)
        self.warning.setVisible(bool(message))
        self.tint_value.setText(f"{self.tint.value()} %")
        for kind, btn in self.tree_buttons.items():
            colour = pal[f"tree_{kind}"]
            btn.setIcon(colour_dot(colour, self._side))
            btn.setText(colour if kind in self._tree else "Theme's own")
            btn.setToolTip(f"{colour}. Click to choose another colour for "
                           f"{ap.TREE_KINDS[kind][0].lower()} in the file trees.")
            self.tree_resets[kind].setEnabled(kind in self._tree)

    # -- edits ---------------------------------------------------------------------

    def _on_accent(self, _index: int) -> None:
        if self.accent.currentData() == _CUSTOM:
            start = QColor(self._custom or PALETTES[self._theme]["accent"])
            chosen = QColorDialog.getColor(start, self, "Accent colour")
            if chosen.isValid():
                self._custom = chosen.name()
                self._fill_accents()
                self.accent.blockSignals(True)
                self.accent.setCurrentIndex(self.accent.findData(_CUSTOM))
                self.accent.blockSignals(False)
            elif not self._custom:
                self.accent.blockSignals(True)
                self.accent.setCurrentIndex(0)
                self.accent.blockSignals(False)
        self._sync()
        self.changed.emit()

    def _on_tint(self, _value: int) -> None:
        self._sync()
        self.changed.emit()

    def _pick_tree(self, kind: str) -> None:
        start = QColor(self.effective()[f"tree_{kind}"])
        label = ap.TREE_KINDS[kind][0]
        chosen = QColorDialog.getColor(start, self, f"Colour of {label.lower()}")
        if chosen.isValid():
            self.set_tree_colour(kind, chosen.name())

    def set_tree_colour(self, kind: str, colour: Optional[str]) -> None:
        if colour:
            self._tree[kind] = colour
        else:
            self._tree.pop(kind, None)
        self._sync()
        self.changed.emit()

    def _reset_tree(self, kind: str) -> None:
        self.set_tree_colour(kind, None)


__all__ = ["AppearanceEditor", "colour_dot"]
