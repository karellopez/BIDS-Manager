"""The guide: one highlighted control, the next thing to do.

Switched on and off in the header, left of Settings (:class:`GuideSwitch`,
remembered under ``ui/guided_mode``). While it is on, a ring is drawn around
the control to use next and a short caption beside it says what it does and
why; both move on as the work does (choose where to save, name, create;
choose the raw data, scan, check the table, convert; validate, fix).

Three pieces:

- :mod:`guide_steps` decides the step from facts each view reports
  (``guide_facts``) and names its control (``guide_targets``); no Qt.
- :class:`GuideOverlay` paints the ring and the caption over the window. It
  is a CHILD of the central widget, never a window (GUI_LOOK_AND_FEEL 4.26),
  transparent to the mouse, so it never stands between the user and a click.
- :class:`Guide` reads the window four times a second while it is on and
  repaints only when the step or the control's place changed, and only the
  region of the ring and caption.

Hidden while a dialog is open: the user is busy with it, and a ring around a
control they cannot reach would be noise.
"""

from __future__ import annotations

from typing import Optional, Sequence

from PyQt6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .guide_steps import Choice, Step, choices, goal_done, next_step
from .viz import fonts

#: How often the window is read while the guide is on.
POLL_MS = 250
#: How long a finished path stays before it fades.
DONE_LINGER_MS = 8000


def _palette() -> dict:
    from .theme_manager import CUR

    return CUR()


def _colour(token: str, alpha: float = 1.0) -> QColor:
    from .widgets.primitives import qcolor

    c = QColor(qcolor(_palette().get(token, "#888888")))
    c.setAlphaF(max(0.0, min(1.0, c.alphaF() * alpha)))
    return c


def _area(r: QRectF) -> float:
    return max(0.0, r.width()) * max(0.0, r.height())


def _blend(a: QColor, b: QColor, t: float) -> QColor:
    return QColor.fromRgbF(
        a.redF() + (b.redF() - a.redF()) * t,
        a.greenF() + (b.greenF() - a.greenF()) * t,
        a.blueF() + (b.blueF() - a.blueF()) * t,
        a.alphaF() + (b.alphaF() - a.alphaF()) * t,
    )


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


class GuideSwitch(QAbstractButton):
    """A labelled on/off switch, painted: a track with a knob, then "Guide".

    Painted rather than styled so the knob is round and the track a pill at
    every font size and on every platform (the reason ``Chip`` paints its
    pill too). Colours are read at paint time, so a theme change needs only
    an ``update``; sizes come from ``fonts.px``, so a font-size change needs
    only ``updateGeometry``.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("guide-switch")
        self.setCheckable(True)
        self.setText("Guide")
        self.setAccessibleName("Guide")
        self.setToolTip("Guide: highlight the next step to take, one at a time")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # Reached with Tab and toggled with Space; a click leaves no outline.
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._knob = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_knob)
        self.toggled.connect(self._slide)

    def set_on(self, on: bool) -> None:
        """Set without the slide and without ``toggled``'s side effects."""
        self.blockSignals(True)
        self.setChecked(bool(on))
        self.blockSignals(False)
        self._anim.stop()
        self._knob = 1.0 if on else 0.0
        self.update()

    def _slide(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._knob)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def _on_knob(self, value) -> None:
        self._knob = float(value)
        self.update()

    def _font(self) -> QFont:
        f = fonts.font(12)
        f.setWeight(QFont.Weight.DemiBold if self.isChecked() else QFont.Weight.Medium)
        return f

    def _geometry(self) -> tuple[int, int, int, int]:
        """``(padding, track width, track height, gap)`` at the font scale."""
        return fonts.px(8), fonts.px(28), fonts.px(16), fonts.px(7)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        pad, track_w, track_h, gap = self._geometry()
        f = fonts.font(12)
        f.setWeight(QFont.Weight.DemiBold)    # room for the bolder "on" label
        fm = QFontMetrics(f)
        return QSize(pad + track_w + gap + fm.horizontalAdvance(self.text()) + pad,
                     max(fonts.px(28), fm.height() + fonts.px(10)))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return self.sizeHint()

    def repaint_for_palette(self) -> None:
        self.updateGeometry()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        pad, track_w, track_h, gap = self._geometry()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        hovered = self.underMouse()
        if hovered or self.isDown():
            p.setPen(QPen(_colour("border"), 1))
            p.setBrush(_colour("surface2" if self.isDown() else "surface3"))
            p.drawRoundedRect(r, 6, 6)
        if self.hasFocus() and not hovered:
            p.setPen(QPen(_colour("accent"), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r, 6, 6)

        t = self._knob
        track = QRectF(pad, (self.height() - track_h) / 2.0, track_w, track_h)
        p.setPen(QPen(_blend(_colour("input_border"), _colour("accent"), t), 1))
        p.setBrush(_blend(_colour("surface3"), _colour("accent"), t))
        p.drawRoundedRect(track, track_h / 2.0, track_h / 2.0)
        inset = max(2.0, track_h * 0.15)
        d = track_h - 2 * inset
        x = track.left() + inset + (track_w - 2 * inset - d) * t
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_blend(_colour("dim"), _colour("primary_btn_text"), t))
        p.drawEllipse(QRectF(x, track.top() + inset, d, d))

        p.setFont(self._font())
        p.setPen(_colour("text") if (self.isChecked() or hovered) else _colour("dim"))
        text = QRectF(track.right() + gap, 0, self.width() - track.right() - gap, self.height())
        p.drawText(text, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                   self.text())


# ---------------------------------------------------------------------------
# The overlay
# ---------------------------------------------------------------------------


class GuideOverlay(QWidget):
    """The ring around the control and the caption beside it, painted.

    Repaints only the ring's and the caption's region: the overlay covers
    the whole window, and an ``update()`` of all of it repaints every table
    and tree under it.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("guide-overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.hide()
        self._step: Optional[Step] = None
        self._target: Optional[QRectF] = None
        self._anchor = QRectF()
        self._keep_clear: tuple = ()
        self._bubble = QRectF()
        self._side: Optional[str] = None
        self._tip = QPointF()
        self._lines: tuple = ()
        self._entrance = 1.0
        self._opacity = 1.0
        self._dirty = QRect()
        self._enter = QVariantAnimation(self)
        self._enter.setDuration(600)
        self._enter.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._enter.setStartValue(0.0)
        self._enter.setEndValue(1.0)
        self._enter.valueChanged.connect(self._on_entrance)
        self._fade = QVariantAnimation(self)
        self._fade.setDuration(500)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.valueChanged.connect(self._on_fade)
        self._linger = QTimer(self)
        self._linger.setSingleShot(True)
        self._linger.setInterval(DONE_LINGER_MS)
        self._linger.timeout.connect(self._fade.start)

    # -- what is shown ------------------------------------------------------

    @property
    def step(self) -> Optional[Step]:
        return self._step

    @property
    def target_rect(self) -> Optional[QRectF]:
        return self._target

    @property
    def bubble_rect(self) -> QRectF:
        return QRectF(self._bubble)

    def clear(self) -> None:
        self._enter.stop()
        self._fade.stop()
        self._linger.stop()
        self._step = None
        self._target = None
        self.hide()

    def show_step(self, step: Step, target: Optional[QRect], anchor: QRect,
                  keep_clear: tuple = ()) -> None:
        """Point at ``target`` (overlay coordinates; ``None`` when the
        control is not on screen, then the caption sits at the top of
        ``anchor``), keeping the caption off ``keep_clear``."""
        new_target = QRectF(target) if target is not None else None
        clear = tuple(QRectF(r) for r in keep_clear)
        new_step = self._step is None or step.key != self._step.key
        if not new_step and step == self._step and new_target == self._target \
                and QRectF(anchor) == self._anchor and clear == self._keep_clear \
                and self.isVisible():
            return
        self._step = step
        self._target = new_target
        self._anchor = QRectF(anchor)
        self._keep_clear = clear
        self._layout()
        if new_step:
            self._fade.stop()
            self._linger.stop()
            self._opacity = 1.0
            self._enter.stop()
            if step.busy:
                # The user just pressed this control and the app is starting
                # its work: shown at once, so no frame is painted while that
                # work begins (measured: an entrance there added a 200 ms
                # pause to the start of a conversion).
                self._entrance = 1.0
            else:
                self._entrance = 0.0
                self._enter.start()
            if step.done:
                self._linger.start()
        if not self.isVisible():
            self.show()
        self._repaint()

    def relayout(self) -> None:
        """The font size or theme changed: measure the caption again."""
        if self._step is not None:
            self._layout()
            self._repaint()

    # -- layout -------------------------------------------------------------

    def _fonts(self) -> tuple[QFont, QFont, QFont]:
        counter = fonts.font(10, bold=True)
        counter.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 106)
        return counter, fonts.font(13, bold=True), fonts.font(12)

    def _layout(self) -> None:
        step = self._step
        counter_font, title_font, body_font = self._fonts()
        pad = fonts.px(12)
        margin = fonts.px(8)
        width = min(fonts.px(300), max(fonts.px(160), self.width() - 2 * margin))
        inner = width - 2 * pad
        lines = []
        y = pad
        flags = int(Qt.TextFlag.TextWordWrap)
        label = step.heading.upper()
        if label:
            h = QFontMetrics(counter_font).boundingRect(
                QRect(0, 0, inner, 100000), flags, label).height()
            lines.append((counter_font, label, QRectF(pad, y, inner, h), "counter"))
            y += h + fonts.px(4)
        for font, text, role, after in ((title_font, step.title, "title", fonts.px(4)),
                                        (body_font, step.text, "body", 0)):
            h = QFontMetrics(font).boundingRect(QRect(0, 0, inner, 100000), flags, text).height()
            lines.append((font, text, QRectF(pad, y, inner, h), role))
            y += h + after
        height = y + pad
        self._lines = tuple(lines)
        self._bubble, self._side, self._tip = self._place(width, height)

    #: What the caption should not cover, by kind, with a weight: a button or
    #: a field the user may need next counts fully, a text a quarter, a list
    #: or table (large, mostly content) a little.
    _WEIGHTS = ((QAbstractButton, 1.0), (QLineEdit, 1.0), (QComboBox, 1.0),
                (QAbstractSpinBox, 1.0), (QAbstractItemView, 0.15), (QLabel, 0.25))

    def _obstacles(self, roi: QRectF) -> list[tuple[QRectF, float]]:
        """What lies in ``roi`` that the caption should not cover, with
        weights, plus what the step's text sends the user to (four times).

        Walks only the VISIBLE widgets that reach into ``roi``, pruning hidden
        branches (the other views of the window) and branches outside it, and
        stops at a control. Measured: ``findChildren`` over the whole window
        took up to 450 ms while a conversion filled the panels.
        """
        found: list[tuple[QRectF, float]] = [(r, 4.0) for r in self._keep_clear]
        host = self.parentWidget()
        if host is None:
            return found
        stack = [(w, QRectF(w.geometry()), QRectF(host.rect()))
                 for w in host.children()
                 if isinstance(w, QWidget) and w is not self and not w.isHidden()]
        while stack:
            widget, rect, clip = stack.pop()
            seen = rect.intersected(clip)
            if seen.isEmpty() or not seen.intersects(roi):
                continue
            weight = next((wt for kind, wt in self._WEIGHTS if isinstance(widget, kind)), 0.0)
            if weight:
                found.append((seen, weight))
                continue                      # a control: nothing inside it counts
            for child in widget.children():
                if isinstance(child, QWidget) and not child.isHidden() \
                        and not child.isWindow():
                    stack.append((child, QRectF(child.geometry()).translated(rect.topLeft()),
                                  seen))
        return found

    def _place(self, w: float, h: float) -> tuple[QRectF, Optional[str], QPointF]:
        """Beside the control, where the caption covers the least the user
        may need (preferring below, then above, right, left); inside it, with
        no arrow, when it fills the room; at the top of the view when it is
        off screen."""
        gap = fonts.px(12)
        m = fonts.px(8)
        area = QRectF(self.rect()).adjusted(m, m, -m, -m)

        def clamp(v: float, lo: float, hi: float) -> float:
            return max(lo, min(v, hi)) if hi >= lo else lo

        t = self._target
        if t is None:
            a = self._anchor if not self._anchor.isEmpty() else area
            x = clamp(a.center().x() - w / 2, area.left(), area.right() - w)
            return QRectF(x, clamp(a.top() + fonts.px(16), area.top(), area.bottom() - h),
                          w, h), None, QPointF()
        xs = (t.center().x() - w / 2, t.left(), t.right() - w)
        ys = (t.center().y() - h / 2, t.top())
        candidates = []
        for side, order in (("below", 0), ("above", 1), ("right", 2), ("left", 3)):
            for i, along in enumerate(xs if side in ("below", "above") else ys):
                if side == "below":
                    rect = QRectF(clamp(along, area.left(), area.right() - w),
                                  t.bottom() + gap, w, h)
                elif side == "above":
                    rect = QRectF(clamp(along, area.left(), area.right() - w),
                                  t.top() - gap - h, w, h)
                elif side == "right":
                    rect = QRectF(t.right() + gap,
                                  clamp(along, area.top(), area.bottom() - h), w, h)
                else:
                    rect = QRectF(t.left() - gap - w,
                                  clamp(along, area.top(), area.bottom() - h), w, h)
                if area.contains(rect):
                    candidates.append((side, rect, order * 3 + i))
        if candidates:
            roi = QRectF(candidates[0][1])
            for _side, rect, _rank in candidates[1:]:
                roi = roi.united(rect)
            obstacles = [(r, wt) for r, wt in self._obstacles(roi)
                         if not t.contains(r) and not r.contains(t)]

            def cost(candidate) -> float:
                _side, rect, rank = candidate
                covered = sum(wt * _area(rect.intersected(r)) for r, wt in obstacles)
                return covered + rank      # equal cover: the preferred side
            side, rect, _rank = min(candidates, key=cost)
            tip = {
                "below": QPointF(t.center().x(), t.bottom() + fonts.px(4)),
                "above": QPointF(t.center().x(), t.top() - fonts.px(4)),
                "right": QPointF(t.right() + fonts.px(4), t.center().y()),
                "left": QPointF(t.left() - fonts.px(4), t.center().y()),
            }[side]
            return rect, side, tip
        inside = t.intersected(area)
        x = clamp(inside.center().x() - w / 2, area.left(), area.right() - w)
        y = clamp(inside.top() + fonts.px(24), area.top(), area.bottom() - h)
        return QRectF(x, y, w, h), None, QPointF()

    # -- painting -----------------------------------------------------------

    def _region(self) -> QRect:
        grow = fonts.px(24)
        r = self._bubble.adjusted(-fonts.px(10), -fonts.px(10), fonts.px(10), fonts.px(14))
        if self._target is not None:
            r = r.united(self._target.adjusted(-grow, -grow, grow, grow))
        return r.toAlignedRect()

    def _repaint(self) -> None:
        region = self._region()
        self.update(region.united(self._dirty) if not self._dirty.isNull() else region)
        self._dirty = region

    def _on_entrance(self, value) -> None:
        # About twenty frames, not the animation clock's sixty: each one
        # repaints whatever lies under the ring and the caption.
        value = float(value)
        if value < 1.0 and value - self._entrance < 0.05:
            return
        self._entrance = value
        self._repaint()

    def _on_fade(self, value) -> None:
        self._opacity = float(value)
        self._repaint()

    def _accent(self) -> str:
        return "success" if self._step is not None and self._step.done else "accent"

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._step is None or self._opacity <= 0.0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        e = self._entrance
        if self._target is not None:
            self._paint_ring(p, e)
        self._paint_bubble(p, e)

    def _paint_ring(self, p: QPainter, e: float) -> None:
        a = self._opacity * e
        grow = fonts.px(4) + (1.0 - e) * fonts.px(12)
        # Kept inside the window: a card at its edge keeps its whole ring.
        edge = fonts.px(10) / 2.0 + 1
        r = self._target.adjusted(-grow, -grow, grow, grow).intersected(
            QRectF(self.rect()).adjusted(edge, edge, -edge, -edge))
        radius = min(float(fonts.px(10)), r.height() / 2.0)
        accent = self._accent()
        p.setBrush(Qt.BrushStyle.NoBrush)
        # A soft glow, then a band in the canvas colour that keeps the line
        # off whatever is under it, then the line (GUI_LOOK_AND_FEEL 4.6).
        p.setPen(QPen(_colour(accent, 0.22 * a), fonts.px(10)))
        p.drawRoundedRect(r, radius, radius)
        p.setPen(QPen(_colour("bg", 0.85 * a), fonts.px(4)))
        p.drawRoundedRect(r, radius, radius)
        line = QPen(_colour(accent, a), max(2, fonts.px(2)))
        if self._step.busy:
            line.setStyle(Qt.PenStyle.DashLine)
        p.setPen(line)
        p.drawRoundedRect(r, radius, radius)

    def _bubble_path(self) -> QPainterPath:
        b = self._bubble
        path = QPainterPath()
        path.addRoundedRect(b, fonts.px(10), fonts.px(10))
        if self._side is None:
            return path
        half = fonts.px(7)
        inset = fonts.px(10) + half
        tip = self._tip
        if self._side in ("below", "above"):
            x = max(b.left() + inset, min(tip.x(), b.right() - inset))
            edge = b.top() if self._side == "below" else b.bottom()
            pts = [QPointF(x - half, edge), QPointF(x, tip.y()), QPointF(x + half, edge)]
        else:
            y = max(b.top() + inset, min(tip.y(), b.bottom() - inset))
            edge = b.left() if self._side == "right" else b.right()
            pts = [QPointF(edge, y - half), QPointF(tip.x(), y), QPointF(edge, y + half)]
        arrow = QPainterPath()
        arrow.addPolygon(QPolygonF(pts))
        arrow.closeSubpath()
        return path.united(arrow)

    def _paint_bubble(self, p: QPainter, e: float) -> None:
        a = self._opacity * e
        p.save()
        p.setOpacity(a)
        # Lifted by a few pixels at the start of the entrance.
        p.translate(0, (1.0 - e) * fonts.px(6))
        path = self._bubble_path()
        for dy, alpha in ((4, 0.10), (2, 0.14)):
            p.fillPath(path.translated(0, dy), QColor(0, 0, 0, int(255 * alpha)))
        p.setPen(QPen(_colour(self._accent()), 1.2))
        p.setBrush(_colour("surface"))
        p.drawPath(path)
        left = self._bubble.topLeft()
        flags = int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft
                    | Qt.AlignmentFlag.AlignTop)
        colour = {"counter": _colour(self._accent()), "title": _colour("text"),
                  "body": _colour("dim")}
        for font, text, rect, role in self._lines:
            p.setFont(font)
            p.setPen(colour[role])
            p.drawText(rect.translated(left), flags, text)
        p.restore()


# ---------------------------------------------------------------------------
# The question
# ---------------------------------------------------------------------------


class _ChoiceRow(QFrame):
    """One answer: its label, a line on what it leads to, "Next step" on the
    path's own. A frame rather than a button so the two lines wrap and the
    card's layout measures them; clicked, or Return/Space with the focus."""

    clicked = pyqtSignal()

    def __init__(self, choice: Choice, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("guide-choice")
        self.setProperty("recommended", bool(choice.recommended))
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName(choice.label)
        self.setToolTip(choice.detail)
        self.key = choice.key
        lay = QVBoxLayout(self)
        lay.setContentsMargins(fonts.px(12), fonts.px(8), fonts.px(12), fonts.px(8))
        lay.setSpacing(fonts.px(2))
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(fonts.px(8))
        title = QLabel(choice.label, self)
        title.setObjectName("guide-choice-title")
        title.setWordWrap(True)
        top.addWidget(title, 1)
        labels = [title]
        if choice.recommended:
            tag = QLabel("Next step", self)
            tag.setObjectName("guide-choice-tag")
            top.addWidget(tag, 0, Qt.AlignmentFlag.AlignTop)
            labels.append(tag)
        lay.addLayout(top)
        detail = QLabel(choice.detail, self)
        detail.setObjectName("guide-choice-detail")
        detail.setWordWrap(True)
        lay.addWidget(detail)
        labels.append(detail)
        for label in labels:
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.MouseButton.LeftButton \
                and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


class GuideQuestion(QFrame):
    """The question "What would you like to do now?", asked when the user
    comes to a view that offers more than one thing to do. A card in the window (a child of
    the central widget, never a window: GUI_LOOK_AND_FEEL 4.26), styled by the
    stylesheet so it follows the theme and the font size. An answer emits
    ``chosen``; Not now, Escape or a click anywhere else emits ``dismissed``,
    and the click still does what it does."""

    chosen = pyqtSignal(str)
    dismissed = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("guide-question")
        self.hide()
        self._anchor = QRect()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(fonts.px(16), fonts.px(14), fonts.px(16), fonts.px(12))
        lay.setSpacing(fonts.px(8))
        kicker = QLabel("GUIDE", self)
        kicker.setObjectName("guide-question-kicker")
        title = QLabel("What would you like to do now?", self)
        title.setObjectName("guide-question-title")
        title.setWordWrap(True)
        lay.addWidget(kicker)
        lay.addWidget(title)
        self._rows = QVBoxLayout()
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(fonts.px(6))
        lay.addLayout(self._rows)
        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        foot.addStretch(1)
        later = QPushButton("Not now", self)
        later.setObjectName("guide-dismiss")
        later.setCursor(Qt.CursorShape.PointingHandCursor)
        later.clicked.connect(self.dismissed.emit)
        foot.addWidget(later)
        lay.addLayout(foot)

    def choices(self) -> list[str]:
        return [row.key for row in self.findChildren(_ChoiceRow) if not row.isHidden()]

    def ask(self, choices: Sequence[Choice], anchor: QRect) -> None:
        """Show ``choices`` over the top of ``anchor`` (the views' area)."""
        while self._rows.count():
            row = self._rows.takeAt(0).widget()
            if row is not None:
                row.hide()                    # before it loses its parent (4.26)
                row.setParent(None)
                row.deleteLater()
        for choice in choices:
            row = _ChoiceRow(choice, self)
            row.clicked.connect(lambda key=choice.key: self.chosen.emit(key))
            self._rows.addWidget(row)
        self.place(anchor)
        self.raise_()
        if not self.isVisible():
            self.show()
            QApplication.instance().installEventFilter(self)
        # The focus on the path's own next step, so Return takes it and the
        # focus outline never marks another answer as the one to pick.
        rows = [r for r in self.findChildren(_ChoiceRow) if not r.isHidden()]
        pick = next((r for r in rows if r.property("recommended")), rows[0] if rows else None)
        if pick is not None:
            pick.setFocus(Qt.FocusReason.OtherFocusReason)

    def place(self, anchor: QRect) -> None:
        """Centred over the views, a little below their top."""
        self._anchor = QRect(anchor)
        width = max(fonts.px(260), min(fonts.px(420), anchor.width() - 2 * fonts.px(16)))
        self.setFixedWidth(width)
        lay = self.layout()
        lay.activate()
        height = lay.totalHeightForWidth(width) if lay.hasHeightForWidth() \
            else lay.totalSizeHint().height()
        x = anchor.left() + (anchor.width() - width) // 2
        y = anchor.top() + max(fonts.px(24), int((anchor.height() - height) * 0.2))
        self.setGeometry(x, y, width, height)

    def close_question(self) -> None:
        if self.isVisible():
            QApplication.instance().removeEventFilter(self)
            self.hide()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key.Key_Escape:
            self.dismissed.emit()
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt naming
        if event.type() == QEvent.Type.MouseButtonPress and self.isVisible() \
                and isinstance(obj, QWidget) and obj is not self \
                and not self.isAncestorOf(obj):
            QTimer.singleShot(0, self.dismissed.emit)
        return False


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------


class Guide(QObject):
    """Reads the window, decides the step, places the overlay.

    ``window`` provides ``guide_view()``, ``guide_facts()``, ``guide_target(
    name)`` and a ``stack`` (the views' area), and its ``converter`` emits
    ``scan_finished`` and ``convert_finished``.
    """

    def __init__(self, window) -> None:
        super().__init__(window)
        self._window = window
        self._host = window.centralWidget()
        self._overlay = GuideOverlay(self._host)
        self._question = GuideQuestion(self._host)
        self._question.chosen.connect(self._on_chosen)
        self._question.dismissed.connect(self._on_dismissed)
        self._host.installEventFilter(self)
        self._on = False
        self._converted_root: Optional[str] = None
        self._conversions = 0
        self._subjects: dict[str, bool] = {}
        #: The view the guide last saw; entering another one may ask.
        self._view: Optional[str] = None
        #: Per view, the goal the user chose and the facts when they chose it.
        self._goal: dict[str, str] = {}
        self._before: dict[str, dict] = {}
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self.refresh)
        converter = window.converter
        converter.convert_finished.connect(self._on_convert_finished)
        converter.scan_finished.connect(self._on_scan_finished)
        from .viz.bridge import connect_while_alive

        connect_while_alive(QApplication.instance().focusChanged, self,
                            lambda guide, *_: guide._soon())

    @property
    def overlay(self) -> GuideOverlay:
        return self._overlay

    @property
    def question(self) -> GuideQuestion:
        return self._question

    @property
    def enabled(self) -> bool:
        return self._on

    def goal(self, view: str) -> Optional[str]:
        return self._goal.get(view)

    def set_enabled(self, on: bool) -> None:
        self._on = bool(on)
        self._view = None                 # switched on: the view is entered afresh
        self._goal.clear()
        if self._on:
            self._timer.start()
            self.refresh()
        else:
            self._timer.stop()
            self._question.close_question()
            self._overlay.clear()

    def repaint_for_palette(self) -> None:
        self._overlay.relayout()
        if self._question.isVisible():
            self._question.place(self._anchor())

    def _anchor(self) -> QRect:
        stack = self._window.stack
        return QRect(stack.mapTo(self._host, QPoint(0, 0)), stack.size())

    # -- the question ---------------------------------------------------------

    def _on_chosen(self, key: str) -> None:
        view = self._view
        if view is not None:
            self._goal[view] = key
            self._before[view] = self._facts(view)
        self._question.close_question()
        self.refresh()

    def _on_dismissed(self) -> None:
        self._question.close_question()
        self.refresh()

    # -- what the guide tracks itself ----------------------------------------

    def _on_convert_finished(self, rc: int, _parent) -> None:
        self._conversions += 1
        if int(rc) == 0:
            root = self._window.converter.guide_facts().get("_root")
            self._converted_root = str(root) if root is not None else None
        self._subjects.clear()
        self._soon()

    def _on_scan_finished(self, *_args) -> None:
        self._converted_root = None
        self._soon()

    def _has_subjects(self, root) -> bool:
        key = str(root)
        if key not in self._subjects:
            try:
                from pathlib import Path

                self._subjects[key] = any(
                    p.name.startswith("sub-") and p.is_dir() for p in Path(root).iterdir())
            except OSError:
                self._subjects[key] = False
        return self._subjects[key]

    def _facts(self, view: str) -> dict:
        facts = dict(self._window.guide_facts())
        root = facts.pop("_root", None)
        if view == "converter":
            facts["converted"] = root is not None and str(root) == self._converted_root
            facts["conversions"] = self._conversions
        if view in ("converter", "editor") and root is not None:
            facts["subjects"] = self._has_subjects(root)
        return facts

    # -- placing ---------------------------------------------------------------

    def _soon(self) -> None:
        if self._on:
            QTimer.singleShot(0, self.refresh)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt naming
        if obj is self._host and event.type() == QEvent.Type.Resize and self._on:
            self._overlay.setGeometry(self._host.rect())
            self._overlay.relayout()
            if self._question.isVisible():
                self._question.place(self._anchor())
            self._soon()
        return False

    def _rect_in_host(self, widget: Optional[QWidget]) -> Optional[QRect]:
        """Where ``widget`` shows in the host, clipped by every parent (a
        scrolled list shows part of what it holds); ``None`` when it is not
        on screen in this window.

        Not ``visibleRegion``: that leaves out what an opaque child paints
        (a scroll area's viewport), and ringed only the top of a panel."""
        host = self._host
        if widget is None or not widget.isVisible() or widget.window() is not self._window:
            return None
        rect = QRect(widget.mapTo(host, QPoint(0, 0)), widget.size())
        parent = widget.parentWidget()
        while parent is not None and parent is not host:
            rect = rect.intersected(QRect(parent.mapTo(host, QPoint(0, 0)), parent.size()))
            parent = parent.parentWidget()
        rect = rect.intersected(host.rect())
        return None if rect.isEmpty() else rect

    def refresh(self) -> None:
        if not self._on:
            return
        window = self._window
        if not window.isVisible() or window.isMinimized() \
                or QApplication.activeModalWidget() is not None:
            self._overlay.clear()
            return
        view = window.guide_view()
        facts = self._facts(view)
        if view != self._view:
            # Come to a view: where it offers more than one thing to do, ask.
            self._view = view
            self._goal.pop(view, None)
            asked = choices(view, facts)
            if len(asked) >= 2:
                self._question.ask(asked, self._anchor())
            else:
                self._question.close_question()
        if self._question.isVisible():
            self._overlay.clear()
            return
        goal = self._goal.get(view)
        if goal is not None and goal_done(view, goal, self._before.get(view, {}), facts):
            goal = None
            self._goal.pop(view, None)
        step = next_step(view, facts, goal)
        if step is None:
            self._overlay.clear()
            return
        if self._overlay.geometry() != self._host.rect():
            self._overlay.setGeometry(self._host.rect())
        target = self._rect_in_host(window.guide_target(step.target))
        clear = [r for r in (self._rect_in_host(window.guide_target(name))
                             for name in step.keep_clear) if r is not None]
        self._overlay.raise_()
        self._overlay.show_step(step, target, self._anchor(), tuple(clear))


__all__ = ["DONE_LINGER_MS", "Guide", "GuideOverlay", "GuideQuestion", "GuideSwitch",
           "POLL_MS"]
