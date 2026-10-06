"""The mosaic page: many slices in one figure, from a line of text.

Each tile is a plane at a millimetre position, coloured exactly as the slice
views colour it (same layers, windows and colour maps), so a mosaic is a
figure of what the viewer shows, not a separate rendering of it. Exported
with the screenshot action like any canvas.
"""

from __future__ import annotations

import logging
import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from ....viz import views
from ....viz.compute import geometry, intensity, mosaic
from ....viz.scene import PLANE_AXIS
from ..bridge import connect_while_alive
from ..context import ViewerContext

log = logging.getLogger(__name__)


def _font(px: int) -> QFont:
    from ...theme_manager import scaled_px

    f = QFont()
    f.setPixelSize(scaled_px(px))
    return f


class MosaicCanvas(QWidget):
    """Draws the scene's mosaic line."""

    def __init__(self, ctx: ViewerContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setMinimumSize(60, 60)
        self._cache: dict = {}
        ctx.qstore.changed.connect(self._on_changed)
        connect_while_alive(ctx.theme_hub.changed, self, lambda w, _t: w.update())

    def _on_changed(self, paths) -> None:
        if self.isVisible() and any(
            not p.startswith(("render.", "clips", "graph", "cursor")) for p in paths
        ):
            self._cache.clear()
            self.update()

    def _tile_image(self, tile: mosaic.Tile):
        key = (tile.plane, round(tile.mm, 3))
        if key in self._cache:
            return self._cache[key]
        store = self.ctx.store
        layer, src = views.base(store)
        if src is None:
            return None
        world = np.asarray(views.cursor_world(store), dtype=float)
        world[PLANE_AXIS[tile.plane]] = tile.mm
        d = self.ctx.scene.display
        grid = geometry.grid_for(
            tile.plane, affine=src.affine, spatial=src.spatial, zooms=src.zooms3,
            orientation=views.orientation(src), cursor_world=world, space=d.space,
            ras=d.ras, radiological=d.radiological,
        )
        layers = []
        first = True
        for lay in self.ctx.scene.layers:
            if lay.kind != "volume" or not lay.visible:
                continue
            values = views.layer_values(store, lay, grid)
            if values is None:
                if first:
                    return None
                continue
            lsrc = views.source_of(store, lay)
            window = lay.display.window or lsrc.robust_range(views.frame_of(store, lay, lsrc))
            layers.append(intensity.colorize(values, lay.display, is_base=first, window=window))
            first = False
        if not layers:
            return None
        buf = np.ascontiguousarray(intensity.composite(layers))
        img = QImage(buf.data, buf.shape[1], buf.shape[0], buf.shape[1] * 4,
                     QImage.Format.Format_RGBA8888).copy()
        self._cache[key] = (img, grid)
        return self._cache[key]

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        theme = self.ctx.theme
        if not self.ctx.transparent:
            p.fillRect(self.rect(), QColor(theme.background))
        spec = mosaic.parse(self.ctx.scene.mosaic)
        rows = []
        for row in spec.rows:
            items = []
            for tile in row:
                got = self._tile_image(tile)
                if got is not None:
                    items.append((tile, got[0], got[1]))
            if items:
                rows.append(items)
        if not rows:
            p.setPen(QColor(theme.dim))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Type a mosaic line, for example:  A -20 0 20 40 ; C 0 S 0")
            p.end()
            return
        overlap = spec.overlap
        # Size in millimetres, then fit the whole figure to the widget.
        row_sizes = []
        for items in rows:
            widths = [g.extent_mm[0] for _t, _i, g in items]
            height = max(g.extent_mm[1] for _t, _i, g in items)
            total = sum(widths) - overlap * sum(widths[1:])
            row_sizes.append((total, height, widths))
        fig_w = max(w for w, _h, _ws in row_sizes)
        fig_h = sum(h for _w, h, _ws in row_sizes)
        label_h = 14 if spec.labels else 0
        avail_w = max(1.0, self.width() - 8.0)
        avail_h = max(1.0, self.height() - 8.0 - label_h * len(rows))
        ppm = min(avail_w / max(fig_w, 1e-6), avail_h / max(fig_h, 1e-6))
        y = 4.0
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        for items, (row_w, row_h, widths) in zip(rows, row_sizes):
            x = 4.0 + (avail_w - row_w * ppm) / 2.0
            for k, (tile, img, grid) in enumerate(items):
                w_mm, h_mm = grid.extent_mm
                rect = QRectF(x, y + (row_h - h_mm) * ppm / 2.0, w_mm * ppm, h_mm * ppm)
                p.drawImage(rect, img)
                if tile.cross:
                    self._paint_cross(p, rect, grid, tile, items)
                if spec.labels:
                    p.setPen(QColor(theme.dim))
                    p.setFont(_font(10))
                    p.drawText(QRectF(rect.left(), rect.bottom() + 1, rect.width(), label_h),
                               Qt.AlignmentFlag.AlignCenter,
                               f"{tile.plane[0].upper()} {tile.mm:g}")
                x += widths[k] * ppm * (1.0 - overlap)
            y += row_h * ppm + label_h
        p.end()

    def _paint_cross(self, p: QPainter, rect: QRectF, grid, tile, items) -> None:
        pen = QPen(QColor(self.ctx.settings.crosshair.color))
        pen.setCosmetic(True)
        p.setPen(pen)
        rows, cols = grid.shape
        for other, _img, _g in items:
            if other.plane == tile.plane:
                continue
            world = np.asarray(grid.origin, dtype=float).copy()
            world[PLANE_AXIS[other.plane]] = other.mm
            c, r, _d = grid.world_to_pixel(world)
            axis = PLANE_AXIS[other.plane]
            if abs(grid.col_vec[axis]) > abs(grid.row_vec[axis]) and np.isfinite(c):
                x = rect.left() + (c + 0.5) / cols * rect.width()
                p.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            elif np.isfinite(r):
                y = rect.top() + (r + 0.5) / rows * rect.height()
                p.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))


class MosaicPage(QWidget):
    """The mosaic canvas with its line editor above it."""

    def __init__(self, ctx: ViewerContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(6, 4, 6, 0)
        lbl = QLabel("Mosaic")
        lbl.setObjectName("sidecar-footer-summary")
        row.addWidget(lbl)
        self.edit = QLineEdit(ctx.scene.mosaic)
        self.edit.setToolTip(
            "A, C, S choose the plane; numbers are slice positions in mm; "
            "';' starts a row; X draws where the other tiles cut; L- hides "
            "labels; H 0.3 overlaps tiles by 30 percent."
        )
        self.edit.editingFinished.connect(
            lambda: ctx.run("view.mosaic_text", text=self.edit.text())
        )
        row.addWidget(self.edit, 1)
        lay.addLayout(row)
        self.canvas = MosaicCanvas(ctx)
        lay.addWidget(self.canvas, 1)
        ctx.qstore.changed.connect(self._on_changed)

    def _on_changed(self, paths) -> None:
        if "mosaic" in paths or "scene" in paths:
            if self.edit.text() != self.ctx.scene.mosaic:
                self.edit.setText(self.ctx.scene.mosaic)


__all__ = ["MosaicCanvas", "MosaicPage"]
