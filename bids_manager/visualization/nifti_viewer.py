"""NIfTI slice viewer and graph controls."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPen, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
import matplotlib.pyplot as plt


class _AutoUpdateLabel(QLabel):
    """QLabel that triggers a callback whenever it is resized."""

    def __init__(self, update_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_fn = update_fn

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if callable(self._update_fn):
            self._update_fn()


class _ImageLabel(_AutoUpdateLabel):
    """Label that notifies on resize and mouse clicks."""

    def __init__(self, update_fn, click_fn, *args, **kwargs):
        super().__init__(update_fn, *args, **kwargs)
        self._click_fn = click_fn

    def mousePressEvent(self, event):
        if callable(self._click_fn):
            self._click_fn(event)
        super().mousePressEvent(event)


class NiftiViewer:
    """Encapsulates NIfTI slice and graph UI logic for ``MetadataViewer``."""

    def __init__(self, host, nib_module) -> None:
        self._host = host
        self._nib = nib_module

    def setup_toolbar(self) -> None:
        """Toolbar for NIfTI viewer with orientation buttons and sliders."""
        host = self._host
        host.orientation = 2  # 0=sagittal, 1=coronal, 2=axial (default)
        host.ax_btn = QPushButton("Axial")
        host.co_btn = QPushButton("Coronal")
        host.sa_btn = QPushButton("Sagittal")
        for b in (host.ax_btn, host.co_btn, host.sa_btn):
            b.setCheckable(True)
        host.ax_btn.setChecked(True)
        host.ax_btn.clicked.connect(lambda: self.set_orientation(2))
        host.co_btn.clicked.connect(lambda: self.set_orientation(1))
        host.sa_btn.clicked.connect(lambda: self.set_orientation(0))
        host.toolbar.addWidget(host.sa_btn)
        host.toolbar.addWidget(host.co_btn)
        host.toolbar.addWidget(host.ax_btn)

        host.view3d_btn = QPushButton("3D View")
        host.view3d_btn.clicked.connect(host._show_3d_view)
        host.toolbar.addWidget(host.view3d_btn)

        host.graph_btn = QPushButton("Graph")
        host.graph_btn.setCheckable(True)
        host.graph_btn.clicked.connect(self.toggle_graph)
        host.toolbar.addWidget(host.graph_btn)

        def add_slider(title, slider, val_label=None):
            box = QVBoxLayout()
            lab = QLabel(title)
            lab.setAlignment(Qt.AlignCenter)
            box.addWidget(lab)
            row = QHBoxLayout()
            row.addWidget(slider)
            if val_label is not None:
                row.addWidget(val_label)
            box.addLayout(row)
            host.toolbar.addLayout(box)

        host.slice_slider = QSlider(Qt.Horizontal)
        host.slice_slider.valueChanged.connect(self.update_slice)
        host.slice_val = QLabel("0")
        add_slider("Slice", host.slice_slider, host.slice_val)

        host.vol_slider = QSlider(Qt.Horizontal)
        host.vol_slider.valueChanged.connect(self.update_slice)
        host.vol_val = QLabel("0")
        add_slider("Volume", host.vol_slider, host.vol_val)

        host.bright_slider = QSlider(Qt.Horizontal)
        host.bright_slider.setRange(-100, 100)
        host.bright_slider.setValue(0)
        host.bright_slider.valueChanged.connect(self.update_slice)
        add_slider("Brightness", host.bright_slider)

        host.contrast_slider = QSlider(Qt.Horizontal)
        host.contrast_slider.setRange(0, 200)
        host.contrast_slider.setValue(100)
        host.contrast_slider.valueChanged.connect(self.update_slice)
        add_slider("Contrast", host.contrast_slider)

        host.voxel_val_label = QLabel("N/A")
        host.value_row.addWidget(QLabel("Voxel value:"))
        host.value_row.addWidget(host.voxel_val_label)
        host.value_row.addStretch()
        host.toolbar.addStretch()

    def create_view(self, path: Path, img_data=None) -> QWidget:
        """Create a viewer widget for NIfTI images with slice/volume controls."""
        host = self._host
        meta = {}
        if img_data is None:
            host.nifti_img = self._nib.load(str(path))
            data, meta = host._get_nifti_data(host.nifti_img)
        else:
            if isinstance(img_data, tuple) and len(img_data) >= 2:
                host.nifti_img = img_data[0]
                data = img_data[1]
                if len(img_data) >= 3 and isinstance(img_data[2], dict):
                    meta = img_data[2] or {}
            else:
                host.nifti_img = img_data
                data, meta = host._get_nifti_data(host.nifti_img)

        if data is None:
            data, meta = host._get_nifti_data(host.nifti_img)

        host.data = data
        host._nifti_meta = meta or {}
        host._nifti_is_color = bool(host._nifti_meta.get("is_rgb"))
        widget = QWidget()
        vlay = QVBoxLayout(widget)

        host.cross_voxel = [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2]
        host._img_scale = (1.0, 1.0)

        host.img_label = _ImageLabel(self.update_slice, self._on_image_clicked)
        host.img_label.setAlignment(Qt.AlignCenter)
        host.img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        host.img_label.setMinimumSize(1, 1)

        img_container = QWidget()
        ic_layout = QVBoxLayout(img_container)
        ic_layout.setContentsMargins(0, 0, 0, 0)
        ic_layout.addWidget(host.img_label)

        host.splitter = QSplitter(Qt.Vertical)
        host.splitter.addWidget(img_container)

        host.graph_panel = QWidget()
        g_lay = QVBoxLayout(host.graph_panel)
        g_lay.setContentsMargins(0, 0, 0, 0)
        g_lay.setSpacing(2)

        host.graph_canvas = FigureCanvas(plt.Figure(figsize=(4, 2)))
        host.graph_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        host.ax = host.graph_canvas.figure.subplots()
        g_lay.addWidget(host.graph_canvas)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        host.scope_spin = QSpinBox()
        host.scope_spin.setRange(1, 4)
        host.scope_spin.setValue(1)
        host.scope_spin.valueChanged.connect(self.update_graph)
        scope_row.addWidget(host.scope_spin)
        scope_row.addSpacing(10)
        scope_row.addWidget(QLabel("Dot size:"))
        host.dot_size_spin = QSpinBox()
        host.dot_size_spin.setRange(1, 20)
        host.dot_size_spin.setValue(6)
        host.dot_size_spin.valueChanged.connect(self.update_graph)
        scope_row.addWidget(host.dot_size_spin)
        scope_row.addSpacing(15)
        host.mark_neighbors_box = QCheckBox("Mark neighbors")
        host.mark_neighbors_box.setChecked(True)
        host.mark_neighbors_box.stateChanged.connect(self.update_graph)
        scope_row.addWidget(host.mark_neighbors_box)
        scope_row.addStretch()
        g_lay.addLayout(scope_row)

        host.graph_panel.setVisible(False)
        host.splitter.addWidget(host.graph_panel)
        host.splitter.setStretchFactor(0, 1)
        host.splitter.setStretchFactor(1, 1)

        vlay.addWidget(host.splitter)

        if data.ndim == 4 and not host._nifti_is_color:
            n_vols = data.shape[3]
        else:
            n_vols = 1
        host.vol_slider.setMaximum(max(n_vols - 1, 0))
        host.vol_slider.setEnabled(n_vols > 1)
        host.vol_slider.setValue(0)
        host.vol_val.setText("0")
        host.graph_btn.setVisible(n_vols > 1)

        self.set_orientation(host.orientation)
        self.update_slice()
        return widget

    def _get_volume_data(self, vol_idx: Optional[int] = None):
        """Return the 3-D volume used for display."""
        host = self._host
        if vol_idx is None:
            slider = getattr(host, "vol_slider", None)
            vol_idx = slider.value() if slider is not None else 0

        if host.data.ndim == 4 and not getattr(host, "_nifti_is_color", False):
            vol_idx = max(0, min(vol_idx, host.data.shape[3] - 1))
            return host.data[..., vol_idx]
        return host.data

    def set_orientation(self, axis: int) -> None:
        """Set viewing orientation and update slice slider."""
        host = self._host
        host.orientation = axis
        slider = getattr(host, "vol_slider", None)
        vol_idx = slider.value() if slider is not None else 0
        vol = self._get_volume_data(vol_idx)
        axis_len = vol.shape[axis]
        host.slice_slider.setMaximum(max(axis_len - 1, 0))
        host.slice_slider.setEnabled(axis_len > 1)
        host.slice_slider.setValue(axis_len // 2)
        host.slice_val.setText(str(axis_len // 2))
        self.update_slice()

    def update_slice(self) -> None:
        """Update displayed slice when slider moves."""
        host = self._host
        slider = getattr(host, "vol_slider", None)
        vol_idx = slider.value() if slider is not None else 0
        vol = self._get_volume_data(vol_idx)
        axis = getattr(host, "orientation", 2)
        slice_idx = (
            host.slice_slider.value()
            if hasattr(host, "slice_slider")
            else vol.shape[axis] // 2
        )
        host.slice_val.setText(str(slice_idx))
        host.vol_val.setText(str(vol_idx))
        if axis == 0:
            slice_img = vol[slice_idx, :, :]
        elif axis == 1:
            slice_img = vol[:, slice_idx, :]
        else:
            slice_img = vol[:, :, slice_idx]

        arr = slice_img.astype(np.float32)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()

        bright = getattr(host, "bright_slider", None)
        contrast = getattr(host, "contrast_slider", None)
        b_val = bright.value() / 100.0 if bright else 0.0
        c_factor = (contrast.value() / 100.0) if contrast else 1.0
        arr = (arr - 0.5) * c_factor + 0.5 + b_val
        arr = np.clip(arr, 0, 1)

        arr = (arr * 255).astype(np.uint8)
        arr = np.rot90(arr)
        if arr.ndim == 2:
            h, w = arr.shape
            img = QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, c = arr.shape
            fmt = QImage.Format_RGB888 if c == 3 else QImage.Format_RGBA8888
            bytes_per_line = w * c
            img = QImage(arr.tobytes(), w, h, bytes_per_line, fmt)
        pix = QPixmap.fromImage(img)

        scaled = pix.scaled(host.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        host._img_scale = (scaled.width() / w, scaled.height() / h)

        if host.cross_voxel is not None:
            x_rot, y_rot = self._voxel_to_arr(host.cross_voxel)
            scale_x, scale_y = host._img_scale
            x_s = int(x_rot * scale_x)
            y_s = int(y_rot * scale_y)
            painter = QPainter(scaled)
            theme_color = host.palette().color(QPalette.Highlight)
            pen = QPen(theme_color)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(x_s, 0, x_s, scaled.height())
            painter.drawLine(0, y_s, scaled.width(), y_s)
            square = max(2, int(min(scaled.width(), scaled.height()) * 0.02))
            half = square // 2
            painter.drawRect(x_s - half, y_s - half, square, square)
            painter.end()

        host.img_label.setPixmap(scaled)

        self._update_value()
        if host.graph_panel.isVisible():
            self.update_graph_marker()

    def _label_pos_to_img_coords(self, pos):
        host = self._host
        pix = host.img_label.pixmap()
        if pix is None:
            return None
        pw, ph = pix.width(), pix.height()
        lw, lh = host.img_label.width(), host.img_label.height()
        off_x, off_y = (lw - pw) / 2, (lh - ph) / 2
        x = pos.x() - off_x
        y = pos.y() - off_y
        if 0 <= x < pw and 0 <= y < ph:
            scale_x, scale_y = host._img_scale
            return int(x / scale_x), int(y / scale_y)
        return None

    def _arr_to_voxel(self, x, y):
        host = self._host
        vol_idx = host.vol_slider.value()
        vol = self._get_volume_data(vol_idx)
        axis = host.orientation
        slice_idx = host.slice_slider.value()
        if axis == 0:
            j = x
            k = vol.shape[2] - 1 - y
            return slice_idx, j, k
        if axis == 1:
            i = x
            k = vol.shape[2] - 1 - y
            return i, slice_idx, k
        i = x
        j = vol.shape[1] - 1 - y
        return i, j, slice_idx

    def _voxel_to_arr(self, voxel):
        host = self._host
        i, j, k = voxel
        vol_idx = host.vol_slider.value()
        vol = self._get_volume_data(vol_idx)
        axis = host.orientation
        if axis == 0:
            x = j
            y = vol.shape[2] - 1 - k
        elif axis == 1:
            x = i
            y = vol.shape[2] - 1 - k
        else:
            x = i
            y = vol.shape[1] - 1 - j
        return x, y

    def _on_image_clicked(self, event) -> None:
        coords = self._label_pos_to_img_coords(event.pos())
        if coords:
            voxel = self._arr_to_voxel(*coords)
            self._host.cross_voxel = list(voxel)
            self.update_slice()
            if self._host.graph_panel.isVisible():
                self.update_graph()

    def _update_value(self) -> None:
        host = self._host
        if host.cross_voxel is None:
            host.voxel_val_label.setText("N/A")
            return
        vol_idx = host.vol_slider.value()
        i, j, k = host.cross_voxel
        if getattr(host, "_nifti_is_color", False):
            vec = np.asarray(host.data[i, j, k, :], dtype=float)
            if vec.size == 0:
                host.voxel_val_label.setText("N/A")
            else:
                components = ", ".join(f"{v:.3g}" for v in vec)
                host.voxel_val_label.setText(f"[{components}]")
            return

        if host.data.ndim == 4:
            val = host.data[i, j, k, vol_idx]
        else:
            val = host.data[i, j, k]
        host.voxel_val_label.setText(f"{float(val):.3g}")

    def toggle_graph(self) -> None:
        host = self._host
        visible = host.graph_btn.isChecked()
        host.graph_panel.setVisible(visible)
        total = host.splitter.size().height()
        if visible:
            host.splitter.setSizes([total // 2, total // 2])
            self.update_graph()
        else:
            host.splitter.setSizes([total, 0])

    def update_graph(self) -> None:
        host = self._host
        if (
            host.data.ndim != 4
            or host.cross_voxel is None
            or getattr(host, "_nifti_is_color", False)
        ):
            return

        level = host.scope_spin.value()
        dim = 2 * (level - 1) + 1
        half = dim // 2
        i0, j0, k0 = host.cross_voxel
        orient = host.orientation

        host.graph_canvas.figure.clf()
        axes = host.graph_canvas.figure.subplots(
            dim, dim, squeeze=False, sharex=True, sharey=True
        )

        line_color = "#000000" if not host._is_dark_theme() else "#ffffff"
        marker_color = host.palette().color(QPalette.Highlight).name()
        bg_color = host.palette().color(QPalette.Base).name()
        dot_size = host.dot_size_spin.value()
        host.graph_canvas.figure.set_facecolor(bg_color)
        host.markers = []
        host.marker_ts = []
        global_min = float("inf")
        global_max = float("-inf")

        for r, di in enumerate(range(-half, half + 1)):
            for c, dj in enumerate(range(-half, half + 1)):
                ax = axes[r][c]
                i, j, k = i0, j0, k0
                if orient == 0:
                    j = j0 + di
                    k = k0 + dj
                elif orient == 1:
                    i = i0 + di
                    k = k0 + dj
                else:
                    i = i0 + di
                    j = j0 + dj

                if not (
                    0 <= i < host.data.shape[0]
                    and 0 <= j < host.data.shape[1]
                    and 0 <= k < host.data.shape[2]
                ):
                    ax.axis("off")
                    continue

                ts_orig = host.data[i, j, k, :]
                ts = ts_orig
                global_min = min(global_min, ts_orig.min())
                global_max = max(global_max, ts_orig.max())
                ax.set_facecolor(bg_color)
                ax.plot(ts, color=line_color, linewidth=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False)
                if host.mark_neighbors_box.isChecked() or (r == half and c == half):
                    host.marker_ts.append(ts)
                    idx = host.vol_slider.value()
                    marker, = ax.plot(
                        [idx],
                        [ts[idx]],
                        "o",
                        color=marker_color,
                        markersize=dot_size,
                    )
                    host.markers.append(marker)

        if global_min < global_max:
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_ylim(global_min, global_max)

        host.graph_canvas.figure.tight_layout(pad=0.1)
        host.graph_canvas.draw()

    def update_graph_marker(self) -> None:
        host = self._host
        if (
            not getattr(host, "markers", None)
            or not getattr(host, "marker_ts", None)
            or getattr(host, "_nifti_is_color", False)
        ):
            return
        marker_color = host.palette().color(QPalette.Highlight).name()
        idx = host.vol_slider.value()
        for marker, ts in zip(host.markers, host.marker_ts):
            i = max(0, min(idx, len(ts) - 1))
            marker.set_data([i], [ts[i]])
            marker.set_color(marker_color)
            marker.set_markersize(host.dot_size_spin.value())
        host.graph_canvas.draw_idle()
