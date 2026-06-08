"""Small reusable widgets used by both Converter and Editor views.

Lift-and-shift from ``inspector_proto/proto.py`` (lines 181-227 of the
prototype). Same QSS object names, same paint behaviour. No logic
change here — these are pure presentation primitives the higher-level
views compose.

Object-name conventions (consumed by ``theme.qss``):

* ``Chip``          → ``chipKind`` property selects color (``default``,
  ``success``, ``warn``, ``err``, ``purple``, ``teal``).
* ``VSep``          → ``vsep`` — 1px vertical divider used in toolbars.
* ``PaneHeader``    → ``pane-h5`` — 28px uppercase pane title.
* ``PathBar``       → ``pathbar`` (root), ``path-label`` (left text),
  ``path-field`` (readonly QLineEdit), ``tb-btn-ghost`` (change button).

All four respect palette swaps through the QSS alone; no per-widget
``repaint_for_palette`` is needed.
"""

from __future__ import annotations

from typing import Sequence

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
)


def _chip_qcolor(token: str) -> QColor:
    """Parse a palette token (hex or ``rgba(r,g,b,a)``) into a QColor.

    ``QColor`` cannot parse the CSS ``rgba(...)`` strings the palette uses
    for translucent tints, so handle those explicitly.
    """
    s = (token or "").strip()
    if s.startswith(("rgba(", "rgb(")):
        nums = s[s.index("(") + 1: s.rindex(")")].split(",")
        try:
            r, g, b = (int(float(nums[i])) for i in range(3))
            a = int(float(nums[3]) * 255) if len(nums) > 3 else 255
            return QColor(r, g, b, a)
        except (ValueError, IndexError):
            return QColor("#000000")
    return QColor(s or "#000000")


class Chip(QLabel):
    """A small pill label that **paints its own rounded background**.

    ``kind`` selects the colour set: ``default``, ``success``, ``warn``,
    ``err``, ``accent``, ``purple``, ``teal``.

    The pill is drawn in :meth:`paintEvent` with a corner radius of
    ``height / 2``, so it is ALWAYS a perfect pill at any font size /
    DPI / platform - QSS ``border-radius`` proved unreliable across
    macOS/Linux/Windows + font scales. Colours are read live from the
    active palette (:func:`bidsmgr.gui.theme_manager.CUR`) at paint time,
    so a theme swap (which triggers ``update()``) recolours automatically.

    Emits :pyattr:`clicked` on a left-mouse release **only if**
    :meth:`set_clickable` is called with ``True``.
    """

    clicked = pyqtSignal()

    # kind -> (background token, foreground token, border token).
    _KIND_TOKENS = {
        "default": ("surface3", "dim", "border"),
        "success": ("success_bg", "success", "success_border"),
        "warn":    ("warning_bg", "warning", "warning_border"),
        "err":     ("error_bg", "error", "error_border"),
        "accent":  ("accent_bg", "accent", "accent_border"),
        "purple":  ("purple_bg", "purple", "purple_border"),
        "teal":    ("teal_bg", "teal", "teal_border"),
    }

    def __init__(self, text: str, kind: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self._kind = kind or "default"
        # Keep the property so any QSS targeting chipKind still matches.
        self.setProperty("chipKind", self._kind)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        # Transparent widget background so ONLY the painted pill shows - the
        # parent (toolbar / panel) shows through the rounded corners. Without
        # this Qt fills the widget rect first, leaving a square behind the pill.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Horizontal breathing room + a hair of vertical so the pill reads.
        self.setContentsMargins(9, 1, 9, 1)
        self._clickable = False

    def setText(self, text: str) -> None:  # noqa: N802 — Qt naming
        super().setText(text)
        self.updateGeometry()
        self.update()

    def set_clickable(self, clickable: bool) -> None:
        """Toggle whether mouse presses on the chip emit ``clicked``."""
        self._clickable = clickable
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if clickable else
            Qt.CursorShape.ArrowCursor
        )

    def mouseReleaseEvent(self, event):  # noqa: N802 — Qt naming
        if self._clickable and event.button() == Qt.MouseButton.LeftButton \
                and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):  # noqa: N802 — Qt naming
        from ..theme_manager import CUR

        pal = CUR()
        bg_t, fg_t, br_t = self._KIND_TOKENS.get(
            self._kind, self._KIND_TOKENS["default"]
        )
        bg = _chip_qcolor(pal.get(bg_t, "#1c2128"))
        fg = _chip_qcolor(pal.get(fg_t, "#8b949e"))
        border = _chip_qcolor(pal.get(br_t, "#21262d"))

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2.0
        p.setBrush(QBrush(bg))
        p.setPen(QPen(border, 1))
        p.drawRoundedRect(rect, radius, radius)
        p.setPen(QPen(fg))
        p.setFont(self.font())
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


class VSep(QFrame):
    """1px-wide vertical separator. Used between toolbar groups."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("vsep")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedWidth(1)


class PaneHeader(QLabel):
    """28px uppercase header used at the top of every splitter pane."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text.upper(), parent)
        self.setObjectName("pane-h5")
        self.setFixedHeight(28)


class PathBar(QFrame):
    """One-row "label · value · trailing chips · change… button" strip.

    Used for the Converter view's raw-input and BIDS-output paths, and
    for the Editor view's BIDS-root path. The value field is read-only
    by design — path edits happen via the ``change…`` button (file
    dialog wiring lives in the view that owns the bar).

    ``ok=True`` paints a ✔ prefix, ``ok=False`` a ○ prefix (these are
    inline characters today; the architecture doc allows replacing with
    bundled SVGs later).
    """

    def __init__(
        self,
        label: str,
        value: str,
        ok: bool = False,
        trailing_chips: Sequence[tuple[str, str]] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("pathbar")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 9, 14, 9)
        lay.setSpacing(10)

        lbl = QLabel(label)
        lbl.setObjectName("path-label")
        lbl.setMinimumWidth(80)
        lay.addWidget(lbl)

        ico = "✔  " if ok else "○  "
        field = QLineEdit(f"{ico}{value}")
        field.setObjectName("path-field")
        field.setReadOnly(True)
        lay.addWidget(field, 1)

        for chip_kind, chip_text in trailing_chips or ():
            lay.addWidget(Chip(chip_text, chip_kind))

        btn = QPushButton("change…")
        btn.setObjectName("tb-btn-ghost")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(btn)

        # Expose the change-button + value field so views can
        # ``.clicked.connect(...)`` / ``set_value(...)`` without
        # fishing through the layout.
        self.change_button = btn
        self._field = field

    def set_value(self, value: str, *, ok: bool = False) -> None:
        """Update the displayed path. Re-renders the ✔/○ prefix as well."""
        ico = "✔  " if ok else "○  "
        self._field.setText(f"{ico}{value}")

    def value(self) -> str:
        """Return the current value without the ✔/○ prefix."""
        raw = self._field.text()
        return raw[3:] if raw[:1] in ("✔", "○") else raw


__all__ = ["Chip", "PaneHeader", "PathBar", "VSep"]
