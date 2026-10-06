"""The quality check of a recording, lane by lane, over its whole length.

What the 4-D volume's QC rows are for a BOLD run, for MEG and EEG:
``viz.compute.meeg_qc`` measures every few seconds, TYPE BY TYPE, how many
of a type's channels are off and how much muscle it shows, and this strip
draws those measures end to end, one lane per type and measure in the
type's own colour, with the segments the check flags in the warning colour
and the window on screen as a rectangle. Click or drag to go there, as on the
overview bar above it; hover for the values and why a segment was flagged.

ONE painted widget (guard 8e). Computed only when the user asks.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QFont, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QToolTip, QWidget

from ....viz.commands import signal as sigcmd
from ..bridge import connect_while_alive
from ..context import ViewerContext
from .overview import _qcolor



class QualityStrip(QWidget):
    """Lanes of the quality check across the recording; click to go."""

    LANE_PX = 22
    LABEL_PX = 112
    RADIUS = 5.0

    def __init__(self, ctx: ViewerContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setObjectName("viz-quality-strip")
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(LANE_MIN_WIDTH)
        self._result: Optional[dict] = None
        self._lanes: list = []
        self._grab: Optional[float] = None
        self._last_go = 0.0
        self._wheel_acc = 0.0
        ctx.qstore.changed.connect(self._on_changed)
        connect_while_alive(ctx.theme_hub.changed, self, lambda w, _t: w.update())

    # ------------------------------------------------------------------
    def set_result(self, result: Optional[dict]) -> None:
        self._result = result
        # The lanes the check measured, each within ONE channel type: how
        # many of the type's channels are off, and its muscle.
        self._lanes = [] if result is None else [
            dict(lane, values=np.asarray(lane["values"], dtype=float))
            for lane in result.get("lanes", [])]
        self.setFixedHeight(len(self._lanes) * self.LANE_PX + 8 if self._lanes else 0)
        self.update()

    def result(self) -> Optional[dict]:
        return self._result

    def _on_changed(self, paths) -> None:
        if any(p.startswith(("traces", "cursor")) for p in paths):
            self.update()

    @property
    def source(self):
        return sigcmd.source(self.ctx.store)

    # -- geometry ----------------------------------------------------------
    def _plot_rect(self) -> QRectF:
        return QRectF(self.LABEL_PX, 4.0, max(1.0, self.width() - self.LABEL_PX - 4.0),
                      self.height() - 8.0)

    def _x_of(self, t: float, dur: float) -> float:
        r = self._plot_rect()
        return r.left() + (t / dur if dur > 0 else 0.0) * r.width()

    def _t_of(self, x: float, dur: float) -> float:
        r = self._plot_rect()
        return min(max((x - r.left()) / r.width(), 0.0), 1.0) * dur

    # -- painting ------------------------------------------------------------
    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self._lanes:
            return
        theme = self.ctx.theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        outline = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        shape = QPainterPath()
        shape.addRoundedRect(outline, self.RADIUS, self.RADIUS)
        p.fillPath(shape, _qcolor(theme.token("surface2", "#161b22")))
        p.save()
        p.setClipPath(shape)
        src = self.source
        res = self._result
        r = self._plot_rect()
        font = QFont(self.font())
        font.setPixelSize(10)
        p.setFont(font)
        dim = _qcolor(theme.dim)
        warn = _qcolor(theme.token("warning", "#d29922"), 110)
        if src is not None and res is not None and src.duration > 0:
            dur = src.duration
            offset = src.start_time
            step = float(res["segment_s"])
            times = np.asarray(res["times"], dtype=float) - offset
            # Flagged segments: warning bands across every lane.
            p.setPen(Qt.PenStyle.NoPen)
            for t, bad in zip(times, res["flagged"]):
                if bad:
                    x0, x1 = self._x_of(t, dur), self._x_of(t + step, dur)
                    p.fillRect(QRectF(x0, r.top(), max(x1 - x0, 1.5), r.height()), warn)
            for k, lane_info in enumerate(self._lanes):
                name, typical = lane_info["name"], lane_info["typical"]
                reach, line, values = lane_info["reach"], lane_info["line"], lane_info["values"]
                colour = theme.type_colour(lane_info["type"])
                top = r.top() + k * self.LANE_PX
                lane = QRectF(r.left(), top + 1.0, r.width(), self.LANE_PX - 2.0)
                p.setPen(dim)
                p.drawText(QRectF(6.0, top, self.LABEL_PX - 10.0, self.LANE_PX),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, name)
                finite = values[np.isfinite(values)]
                hi = max(reach, float(np.percentile(finite, 99.5)) if finite.size else reach)
                lo = min(0.0, float(finite.min())) if finite.size else 0.0
                span = max(hi - lo, 1e-9)

                def y_of(v: float) -> float:
                    return lane.bottom() - (min(max(v, lo), hi) - lo) / span * lane.height()

                xs = [self._x_of(t + step / 2.0, dur) for t in times]
                pts = [QPointF(lane.left(), lane.bottom())]
                pts += [QPointF(x, y_of(v if np.isfinite(v) else lo)) for x, v in zip(xs, values)]
                pts.append(QPointF(lane.right(), lane.bottom()))
                p.setPen(QPen(_qcolor(colour, 200), 1.0))
                p.setBrush(_qcolor(colour, 55))
                p.drawPolygon(QPolygonF(pts))
                ref = line if line is not None else typical
                if lo <= ref <= hi:
                    pen = QPen(dim, 1.0, Qt.PenStyle.DashLine)
                    p.setPen(pen)
                    y = y_of(ref)
                    p.drawLine(QPointF(lane.left(), y), QPointF(lane.right(), y))
            # The window on screen, and the time cursor.
            tr = self.ctx.scene.traces
            x0 = self._x_of(tr.t0, dur)
            x1 = self._x_of(min(tr.t0 + tr.width, dur), dur)
            if x1 - x0 < 6.0:
                mid = (x0 + x1) / 2.0
                x0, x1 = mid - 3.0, mid + 3.0
            p.setPen(QPen(_qcolor(theme.accent), 1.5))
            p.setBrush(_qcolor(theme.accent, 35))
            p.drawRoundedRect(QRectF(x0, r.top(), x1 - x0, r.height()), 3.0, 3.0)
            cursor = self.ctx.scene.cursor.time
            if cursor is not None and 0.0 <= cursor - offset <= dur:
                x = self._x_of(cursor - offset, dur)
                p.setPen(QPen(_qcolor(theme.text), 1.0))
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        p.restore()
        p.setPen(QPen(_qcolor(theme.token("border", "#21262d")), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(shape)
        p.end()

    # -- input ---------------------------------------------------------------
    def _go(self, t0: float, *, finish: bool = False) -> None:
        now = time.perf_counter()
        if not finish and now - self._last_go < 1.0 / 30.0:
            return
        self._last_go = now
        self.ctx.run("time.set", t0=float(t0))
        self.ctx.qstore.flush()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        src = self.source
        if src is None or event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        tr = self.ctx.scene.traces
        t = self._t_of(event.position().x(), src.duration)
        if tr.t0 <= t <= tr.t0 + tr.width:
            self._grab = t - tr.t0
        else:
            self._grab = tr.width / 2.0
            self._go(t - self._grab, finish=True)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        src = self.source
        if src is None:
            return
        t = self._t_of(event.position().x(), src.duration)
        if self._grab is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._go(t - self._grab)
            return
        if event.position().x() < self.LABEL_PX:
            # Over a lane's name: what the lane measures and how to read it.
            from ....viz.compute.meeg_qc import HELP

            k = int((event.position().y() - self._plot_rect().top()) // self.LANE_PX)
            if 0 <= k < len(self._lanes):
                lane = self._lanes[k]
                QToolTip.showText(event.globalPosition().toPoint(),
                                  f"{lane['name']}: {HELP.get(lane['key'], '')}", self)
            return
        QToolTip.showText(event.globalPosition().toPoint(), self.describe_at(t), self)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        src = self.source
        if self._grab is not None and src is not None:
            self._go(self._t_of(event.position().x(), src.duration) - self._grab, finish=True)
        self._grab = None

    def wheelEvent(self, event) -> None:  # noqa: N802
        ad, pd = event.angleDelta(), event.pixelDelta()
        use_pixel = not pd.isNull()
        delta = (pd.y() or pd.x()) if use_pixel else (ad.y() or ad.x())
        self._wheel_acc += delta
        thresh = 40.0 if use_pixel else 120.0
        steps = int(self._wheel_acc / thresh)
        self._wheel_acc -= steps * thresh
        if steps:
            self.ctx.run("time.page", n=-float(steps))
        event.accept()

    def describe_at(self, t: float) -> str:
        """The values of the segment at recording time ``t`` (tooltip)."""
        res, src = self._result, self.source
        if res is None or src is None:
            return ""
        step = float(res["segment_s"])
        i = int((t + src.start_time - float(res["times"][0])) // step)
        if not 0 <= i < len(res["times"]):
            return ""
        parts = []
        for lane in self._lanes:
            v = float(lane["values"][i])
            if np.isfinite(v):
                parts.append(f"{lane['name']} {v * 100:.0f} %" if lane["percent"]
                             else f"{lane['name']} z {v:.1f}")
        text = f"{res['times'][i]:.0f} to {res['times'][i] + step:.0f} s: " + ", ".join(parts)
        if res["flagged"][i]:
            text += "\nFlagged: " + ", ".join(res["segment_reasons"][i])
        return text + "\nClick to go there"


#: The strip is never narrower than its label column and a few pixels of lane.
LANE_MIN_WIDTH = QualityStrip.LABEL_PX + 40

__all__ = ["QualityStrip"]
