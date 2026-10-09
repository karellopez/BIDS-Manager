"""A collapsible / detachable wrapper around a pane (or a sub-splitter).

Each side region in the Converter / Editor is wrapped in a ``PanelFrame``
to get two affordances:

* a **collapse caret** that folds the body toward an ``edge``:
    - ``"top"`` / ``"bottom"``  fold by height; caret + title + detach share
      one thin horizontal bar on that edge.
    - ``"left"`` / ``"right"``  fold by width; the title + detach stay in a
      horizontal bar across the top, while the collapse caret sits centred in
      a thin VERTICAL strip on the panel's outer edge.
  When collapsed inside a splitter the freed space is handed to a designated
  ``grow_target`` (the inspection table / the editor viewer) so the work
  surface expands automatically, no manual drag needed.
* a **detach button** (always in the horizontal bar) that pops the body out
  into a floating window; closing it docks it back. ``inner`` may be a whole
  ``QSplitter``, so two panes (inspection + properties) detach as one unit.

The frame hides the pane's own ``PaneHeader`` so there is a single header.
A pane that has buttons belonging NEXT to its title can name a child widget
:data:`HEADER_EXTRAS` and the frame lifts it into that one header; it is
handed back when the pane detaches, so the buttons travel with the pane.
State is not persisted across restarts.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import icons
from .primitives import PaneHeader

_QWIDGETSIZE_MAX = (1 << 24) - 1
_STRIP_PX = 24  # collapsed vertical-strip thickness
_BAR_PX = 30    # horizontal title-bar height

#: The gap between two cards (the handle of a ``card-split`` splitter).
CARD_GAP_PX = 8


def card_canvas(content: QWidget) -> QWidget:
    """``content`` (cards in a splitter) with the canvas showing around it."""
    holder = QWidget()
    holder.setObjectName("card-canvas")
    lay = QVBoxLayout(holder)
    lay.setContentsMargins(CARD_GAP_PX, CARD_GAP_PX // 2, CARD_GAP_PX, CARD_GAP_PX)
    lay.setSpacing(0)
    lay.addWidget(content)
    return holder


# Object name a pane gives a child widget to have it rendered beside the
# frame's title instead of inside the pane. One name rather than a per-pane
# argument, so a pane stays buildable on its own and the frame stays generic.
HEADER_EXTRAS = "pane-header-extras"


class FoldStrip(QWidget):
    """The handle of a panel that folds sideways, clickable ANYWHERE on it
    (a caret button in a strip was a target the size of the caret). Painted
    like the viewer's controls tab: open, a slim strip with a chevron;
    folded, the chevron and the panel's name written down the strip."""

    clicked = pyqtSignal()

    def __init__(self, title: str, edge: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.edge = edge
        self.collapsed = False
        self.setObjectName("panel-frame-strip")
        self.setFixedWidth(_STRIP_PX)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._sync_tip()

    def set_collapsed(self, on: bool) -> None:
        self.collapsed = bool(on)
        self._sync_tip()
        self.update()

    def _sync_tip(self) -> None:
        self.setToolTip(f"{'Open' if self.collapsed else 'Fold away'} "
                        f"{self.title or 'this panel'}")

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
                event.position().toPoint()):
            self.clicked.emit()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().leaveEvent(event)
        self.update()

    def _points_left(self) -> bool:
        # Open, the chevron points the way the panel folds (toward its
        # edge); folded, the way it opens.
        toward_edge = self.edge == "left"
        return toward_edge != self.collapsed

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        from ...viz.theme import parse_colour
        from ..viz import fonts
        from ..viz.bridge import ThemeHub

        theme = ThemeHub.instance().theme
        hover = self.underMouse()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(2.0, 3.0, -2.0, -3.0)
        if hover or self.collapsed:
            path = QPainterPath()
            path.addRoundedRect(r, 6.0, 6.0)
            p.fillPath(path, QColor(*parse_colour(theme.token(
                "surface3" if hover else "surface2", "#161b22"))))
        ink = QColor(*parse_colour(theme.text if hover else theme.dim))
        cx = r.center().x()
        name = (self.title or "").upper()
        font = fonts.font(10, bold=True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        text_len = QFontMetricsF(font).horizontalAdvance(name) if self.collapsed else 0.0
        group = 10.0 + (12.0 + text_len if text_len else 0.0)
        top = max(r.top() + 8.0, r.center().y() - group / 2.0)
        y = top + 5.0
        half = fonts.px(4)
        pen = QPen(ink, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        if self._points_left():
            pts = [QPointF(cx + half * 0.5, y - half), QPointF(cx - half * 0.5, y),
                   QPointF(cx + half * 0.5, y + half)]
        else:
            pts = [QPointF(cx - half * 0.5, y - half), QPointF(cx + half * 0.5, y),
                   QPointF(cx - half * 0.5, y + half)]
        p.drawPolyline(pts)
        if text_len:
            p.setFont(font)
            p.save()
            p.translate(cx, y + 12.0)
            p.rotate(90)
            p.drawText(QRectF(0, -9, text_len + 4.0, 18),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)
            p.restore()
        p.end()


class PanelFrame(QFrame):
    """Collapsible + detachable container for one pane (or sub-splitter)."""

    state_changed = pyqtSignal()

    def __init__(
        self,
        inner: QWidget,
        title: str = "",
        *,
        edge: str = "top",            # "top" | "bottom" | "left" | "right"
        collapsible: bool = True,
        detachable: bool = True,
        hide_inner_header: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("panel-frame")
        self._inner = inner
        self._title = title
        self._edge = edge
        self._vertical_fold = edge in ("left", "right")
        self._collapsible = collapsible
        self._detachable = detachable
        self._collapsed = False
        self._detached: Optional[QDialog] = None
        self._splitter: Optional[QSplitter] = None
        self._grow_target: Optional[QWidget] = None
        self._saved_extent: Optional[int] = None

        # Buttons the pane wants beside the title rather than under it. Kept
        # with the layout they came from, so detaching can hand them back.
        self._extras: Optional[QWidget] = None
        self._extras_home = None

        if hide_inner_header:
            hdr = inner.findChild(PaneHeader)
            if hdr is not None:
                hdr.setVisible(False)
            extras = inner.findChild(QWidget, HEADER_EXTRAS)
            if extras is not None:
                self._extras = extras
                self._extras_home = extras.parentWidget().layout()

        # Placeholder shown in place of the body while detached.
        self._placeholder = QLabel("Detached - close the window to dock it back.")
        self._placeholder.setObjectName("pane-hint")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setVisible(False)

        self._build_controls()
        self._assemble()
        self._refresh_icons()

    # ------------------------------------------------------------------
    def make_card(self) -> "PanelFrame":
        """Draw this panel as a card: a rounded, outlined surface on the
        window's canvas (the stylesheet's ``QFrame[card="true"]``). Only a
        panel at the top of a layout is a card; panels nested in it stay
        flat, so there are no cards inside cards."""
        self.setProperty("card", True)
        # One pixel in from the outline, so the content does not paint over it.
        self.layout().setContentsMargins(1, 1, 1, 1)
        self.style().unpolish(self)
        self.style().polish(self)
        return self

    def _build_controls(self) -> None:
        self._caret = QToolButton()
        self._caret.setObjectName("panel-frame-caret")
        self._caret.setAutoRaise(True)
        self._caret.setCursor(Qt.CursorShape.PointingHandCursor)
        self._caret.setToolTip("Collapse / expand")
        self._caret.clicked.connect(self.toggle_collapsed)

        self._detach_btn = QToolButton()
        self._detach_btn.setObjectName("panel-frame-detach")
        self._detach_btn.setAutoRaise(True)
        self._detach_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._detach_btn.setToolTip("Detach into a floating window")
        self._detach_btn.clicked.connect(self.toggle_detached)

        self._title_lbl = QLabel(self._title.upper())
        self._title_lbl.setObjectName("pane-h5")

        # Horizontal title bar. For top/bottom panels it also carries the
        # caret; for left/right panels the caret lives in the side strip and
        # the bar holds just the title + detach.
        self._bar = QFrame()
        self._bar.setObjectName("panel-frame-header")
        self._bar.setFixedHeight(_BAR_PX)
        bl = QHBoxLayout(self._bar)
        bl.setContentsMargins(6, 0, 4, 0)
        bl.setSpacing(4)
        if not self._vertical_fold:
            bl.addWidget(self._caret)
        bl.addWidget(self._title_lbl, 1)
        if self._extras is not None:
            bl.addWidget(self._extras)
        bl.addWidget(self._detach_btn)

        # Side strip (left/right only): the whole strip is the handle.
        self._strip: Optional[FoldStrip] = None
        if self._vertical_fold:
            self._strip = FoldStrip(self._title, self._edge)
            self._strip.clicked.connect(self.toggle_collapsed)
            self._strip.setVisible(self._collapsible)
            self._caret.setVisible(False)
        else:
            self._caret.setVisible(self._collapsible)
            if self._collapsible:
                # A top or bottom panel folds from anywhere on its title bar
                # (its buttons keep their own clicks).
                self._bar.setProperty("foldable", True)
                self._bar.setCursor(Qt.CursorShape.PointingHandCursor)
                self._bar.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
                self._bar.setToolTip(f"Fold or open {self._title or 'this panel'}")
                self._bar.installEventFilter(self)
        self._detach_btn.setVisible(self._detachable)

    def _assemble(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if self._vertical_fold:
            # [ title bar              ]
            # [ strip | content ]   (strip on the panel's outer edge)
            outer.addWidget(self._bar)
            self._body_row = QWidget()
            brow = QHBoxLayout(self._body_row)
            brow.setContentsMargins(0, 0, 0, 0)
            brow.setSpacing(0)
            if self._edge == "left":
                brow.addWidget(self._strip)
                brow.addWidget(self._inner, 1)
                brow.addWidget(self._placeholder, 1)
                self._content_index = 1
            else:  # right
                brow.addWidget(self._inner, 1)
                brow.addWidget(self._placeholder, 1)
                brow.addWidget(self._strip)
                self._content_index = 0
            outer.addWidget(self._body_row, 1)
        else:
            self._body_row = None
            if self._edge == "bottom":
                outer.addWidget(self._inner, 1)
                outer.addWidget(self._placeholder, 1)
                outer.addWidget(self._bar)
                self._content_index = 0
            else:  # top
                outer.addWidget(self._bar)
                outer.addWidget(self._inner, 1)
                outer.addWidget(self._placeholder, 1)
                self._content_index = 1

    # ------------------------------------------------------------------
    # Splitter wiring
    # ------------------------------------------------------------------

    def attach_splitter(self, splitter: QSplitter, grow_target: Optional[QWidget] = None) -> None:
        self._splitter = splitter
        self._grow_target = grow_target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inner(self) -> QWidget:
        return self._inner

    def is_collapsed(self) -> bool:
        return self._collapsed

    def is_detached(self) -> bool:
        return self._detached is not None

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt override
        from PyQt6.QtCore import QEvent

        if (obj is self._bar and event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
                and self._bar.rect().contains(event.position().toPoint())):
            self.toggle_collapsed()
            return True
        return super().eventFilter(obj, event)

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        if not self._collapsible or collapsed == self._collapsed or self.is_detached():
            return
        self._collapsed = collapsed
        self._inner.setVisible(not collapsed)
        if self._vertical_fold:
            # Fold to just the side strip: hide the top title bar too.
            self._bar.setVisible(not collapsed)
        self._apply_fold(collapsed)
        self._refresh_icons()
        self.state_changed.emit()

    def _grow_index(self, sizes_len: int, idx: int) -> Optional[int]:
        if self._grow_target is not None and self._splitter is not None:
            gi = self._splitter.indexOf(self._grow_target)
            if gi != -1 and gi != idx:
                return gi
        others = [i for i in range(sizes_len) if i != idx]
        return max(others, key=lambda i: self._splitter.sizes()[i]) if others else None

    def _apply_fold(self, collapsed: bool) -> None:
        sp = self._splitter
        if self._vertical_fold:
            if collapsed:
                self.setFixedWidth(_STRIP_PX)
            else:
                self.setMinimumWidth(0)
                self.setMaximumWidth(_QWIDGETSIZE_MAX)
        else:
            self.setMaximumHeight(_BAR_PX + 4 if collapsed else _QWIDGETSIZE_MAX)

        if sp is None or sp.indexOf(self) == -1:
            return
        sizes = sp.sizes()
        idx = sp.indexOf(self)
        grow = self._grow_index(len(sizes), idx)
        if grow is None:
            return
        strip = _STRIP_PX if self._vertical_fold else _BAR_PX + 4
        if collapsed:
            self._saved_extent = sizes[idx]
            delta = sizes[idx] - strip
            sizes[idx] = strip
            sizes[grow] += delta
        else:
            restore = self._saved_extent or 280
            delta = restore - sizes[idx]
            sizes[idx] = restore
            sizes[grow] = max(strip, sizes[grow] - delta)
        sp.setSizes(sizes)

    def toggle_detached(self) -> None:
        if self.is_detached():
            self.reattach()
        else:
            self.detach()

    def detach(self) -> None:
        if self.is_detached():
            return
        if self._collapsed:
            self.set_collapsed(False)

        win = QDialog(self.window())
        win.setWindowTitle(f"BIDS-Manager - {self._title}" if self._title else "BIDS-Manager")
        win.setObjectName("panel-frame-float")
        # A bare QDialog gets only a close button on Linux / Windows. Promote
        # it to a normal top-level window so it carries minimize + maximize
        # buttons and can be maximised / tiled like any other window.
        win.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        lay = QVBoxLayout(win)
        lay.setContentsMargins(0, 0, 0, 0)
        # Give the pane its buttons back before it leaves, or the floating
        # window arrives without the controls it owns while they sit uselessly
        # on a title bar whose body is gone.
        self._return_extras()
        self._inner.setParent(win)
        lay.addWidget(self._inner)
        self._inner.setVisible(True)
        win.resize(max(self._inner.width(), 520), max(self._inner.height(), 380))
        win.finished.connect(lambda _=0: self.reattach())
        self._detached = win

        self._placeholder.setVisible(True)
        self._caret.setVisible(False)
        self._detach_btn.setToolTip("Re-dock the floating window")
        self._refresh_icons()
        win.show()
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # Header extras: buttons a pane owns but that belong beside the title
    # ------------------------------------------------------------------

    def _return_extras(self) -> None:
        """Put the pane's buttons back where the pane built them."""
        if self._extras is None or self._extras_home is None:
            return
        self._extras_home.addWidget(self._extras)
        self._extras.setVisible(True)

    def _take_extras(self) -> None:
        """Lift them back into the title bar."""
        if self._extras is None:
            return
        layout = self._bar.layout()
        layout.insertWidget(layout.count() - 1, self._extras)
        self._extras.setVisible(True)

    def reattach(self) -> None:
        if not self.is_detached():
            return
        win = self._detached
        self._detached = None
        # Move the body back next to its control, at its original position.
        parent_layout = self._body_row.layout() if self._vertical_fold else self.layout()
        self._inner.setParent(self._body_row if self._vertical_fold else self)
        parent_layout.insertWidget(self._content_index, self._inner, 1)
        self._inner.setVisible(True)
        self._take_extras()
        self._placeholder.setVisible(False)
        self._caret.setVisible(self._collapsible)
        self._detach_btn.setVisible(self._detachable)
        self._detach_btn.setToolTip("Detach into a floating window")
        self._refresh_icons()
        try:
            win.finished.disconnect()
        except TypeError:
            pass
        win.close()
        win.deleteLater()
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------

    def repaint_for_palette(self, pal: dict) -> None:
        self._refresh_icons()
        inner_repaint = getattr(self._inner, "repaint_for_palette", None)
        if callable(inner_repaint):
            inner_repaint(pal)

    def _refresh_icons(self) -> None:
        if self._strip is not None:
            self._strip.set_collapsed(self._collapsed)
        glyph = "panel_expand" if self._collapsed else "panel_collapse"
        self._caret.setIcon(icons.icon(glyph))
        self._detach_btn.setIcon(
            icons.icon("reattach" if self.is_detached() else "detach")
        )


__all__ = ["CARD_GAP_PX", "FoldStrip", "PanelFrame", "card_canvas"]
