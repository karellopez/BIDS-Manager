"""3-D surface viewer tailored for FreeSurfer meshes."""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib import colors as mcolors
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QVBoxLayout,
)

from .scene_utils import (
    HAS_PYQTGRAPH,
    _AdjustableAxisItem,
    _create_directional_light_shader,
    _create_flat_color_shader,
    gl,
    pg,
)


class FreeSurferSurfaceDialog(QDialog):
    """Simplified viewer tailored for FreeSurfer ``*.pial``/``*.white`` meshes."""

    def __init__(
        self,
        parent,
        vertices: np.ndarray,
        faces: np.ndarray,
        title: Optional[str] = None,
        dark_theme: bool = False,
    ) -> None:
        super().__init__(parent)
        if not HAS_PYQTGRAPH:
            raise RuntimeError(
                "Surface rendering requires the optional 'pyqtgraph' dependency."
            )

        self.setWindowTitle(title or "FreeSurfer Surface")
        self.resize(900, 720)
        self.setMinimumSize(620, 480)
        self.setSizeGripEnabled(True)

        verts = np.asarray(vertices, dtype=np.float32)
        faces_arr = np.asarray(faces, dtype=np.int32)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("Surface vertices must be an (N, 3) array")
        if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
            raise ValueError("Surface faces must be an (M, 3) array")
        if faces_arr.size and int(faces_arr.max(initial=-1)) >= verts.shape[0]:
            raise ValueError("Face indices exceed available vertices")

        self._vertices = verts
        self._faces = faces_arr
        self._fg_color = "#f0f0f0" if dark_theme else "#202020"
        self._canvas_bg = "#202020" if dark_theme else "#ffffff"

        self._mesh_item: Optional[gl.GLMeshItem] = None
        self._axis_item: Optional[gl.GLAxisItem] = None
        self._current_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._light_shader = _create_directional_light_shader()
        self._flat_shader = _create_flat_color_shader()
        self._lighting_enabled = self._light_shader is not None

        layout = QVBoxLayout(self)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(self._canvas_bg)
        self.view.opts["distance"] = 200
        self.view.opts["elevation"] = 20
        self.view.opts["azimuth"] = -60
        layout.addWidget(self.view, 1)

        controls = QGroupBox("Display settings")
        controls_layout = QGridLayout(controls)
        controls_layout.setColumnStretch(1, 1)

        # Provide a handful of constant colours so users can quickly switch the
        # appearance without diving into advanced scalar options.
        self._colour_presets: list[tuple[str, str]] = [
            ("Clay", "#d87c5a"),
            ("Slate", "#4f6d7a"),
            ("Bone", "#e9d8a6"),
            ("Forest", "#2a9d8f"),
            ("Carbon", "#5c5f66"),
        ]

        row = 0
        controls_layout.addWidget(QLabel("Surface colour:"), row, 0)
        self.color_combo = QComboBox()
        for name, colour in self._colour_presets:
            self.color_combo.addItem(name, userData=colour)
        controls_layout.addWidget(self.color_combo, row, 1, 1, 2)

        row += 1
        controls_layout.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(85)
        controls_layout.addWidget(self.opacity_slider, row, 1)
        self.opacity_label = QLabel("0.85")
        controls_layout.addWidget(self.opacity_label, row, 2)

        row += 1
        self.axes_checkbox = QCheckBox("Show axes")
        controls_layout.addWidget(self.axes_checkbox, row, 0, 1, 3)

        row += 1
        self.light_checkbox = QCheckBox("Enable lighting")
        self.light_checkbox.setChecked(self._lighting_enabled)
        controls_layout.addWidget(self.light_checkbox, row, 0, 1, 3)
        if self._light_shader is None and self._flat_shader is None:
            self._lighting_enabled = False
            self.light_checkbox.setEnabled(False)
            self.light_checkbox.setChecked(False)

        layout.addWidget(controls)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._initialising = True

        self.color_combo.currentIndexChanged.connect(self._on_color_change)
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        self.axes_checkbox.toggled.connect(self._on_axes_toggle)
        self.light_checkbox.toggled.connect(self._on_light_toggle)

        if self._vertices.size:
            mins = np.min(self._vertices, axis=0).astype(np.float32, copy=False)
            maxs = np.max(self._vertices, axis=0).astype(np.float32, copy=False)
        else:
            mins = np.zeros(3, dtype=np.float32)
            maxs = np.zeros(3, dtype=np.float32)
        self._update_scene_bounds(mins, maxs)
        self._update_axis_item()
        self._update_light_shader()
        self._update_mesh()

        self._initialising = False
        self._update_status()

    def _current_colour_rgba(self) -> np.ndarray:
        colour_hex = self.color_combo.currentData()
        if not colour_hex:
            colour_hex = "#d87c5a"
        rgb = np.array(mcolors.to_rgb(str(colour_hex)), dtype=np.float32)
        alpha = np.clip(self.opacity_slider.value() / 100.0, 0.1, 1.0)
        colours = np.empty((self._vertices.shape[0], 4), dtype=np.float32)
        colours[:, :3] = rgb
        colours[:, 3] = alpha
        return colours

    def _update_mesh(self) -> None:
        if self._vertices.size == 0 or self._faces.size == 0:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self.status_label.setText("Surface mesh has no vertices or faces to render.")
            return

        meshdata = gl.MeshData(vertexes=self._vertices, faces=self._faces)
        colours = self._current_colour_rgba()
        meshdata.setVertexColors(colours)

        if self._mesh_item is None:
            self._mesh_item = gl.GLMeshItem(
                meshdata=meshdata,
                smooth=False,
                shader="shaded",
                drawEdges=False,
            )
            self.view.addItem(self._mesh_item)
        else:
            self._mesh_item.setMeshData(meshdata=meshdata)

        self._apply_mesh_shader()
        alpha = self.opacity_slider.value() / 100.0
        self._mesh_item.setGLOptions("translucent" if alpha < 0.999 else "opaque")
        if not self._initialising:
            self.view.update()

    def _update_status(self) -> None:
        if self._vertices.size == 0 or self._faces.size == 0:
            summary = "Surface mesh has no vertices or faces to render."
        else:
            mins, maxs = self._current_bounds if self._current_bounds else (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            )
            summary = (
                f"Surface mesh: {self._vertices.shape[0]:,} vertices / "
                f"{self._faces.shape[0]:,} faces. Opacity {self.opacity_slider.value() / 100.0:.2f}. "
                f"Colour preset: {self.color_combo.currentText()}. "
                f"Bounds X {mins[0]:.1f}–{maxs[0]:.1f} mm, "
                f"Y {mins[1]:.1f}–{maxs[1]:.1f} mm, Z {mins[2]:.1f}–{maxs[2]:.1f} mm. "
            )
        if self._light_shader is None and self._flat_shader is None:
            summary += " Lighting controls unavailable."
        else:
            summary += " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        self.status_label.setText(summary)

    def _apply_mesh_shader(self) -> None:
        if self._mesh_item is None:
            return
        if self._light_shader is not None and self._lighting_enabled:
            self._mesh_item.setShader(self._light_shader)
        elif self._flat_shader is not None:
            self._mesh_item.setShader(self._flat_shader)
        else:
            self._mesh_item.setShader("shaded")

    def _update_light_shader(self) -> None:
        if self._light_shader is None:
            return
        self._light_shader["lightDir"] = [0.5, 0.3, 0.8]
        self._light_shader["lightParams"] = [1.0, 0.35]
        self._apply_mesh_shader()

    def _update_scene_bounds(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        spans = np.maximum(maxs - mins, 1e-3)
        center = (mins + maxs) / 2.0
        self._current_bounds = (mins, maxs)
        self.view.opts["center"] = pg.Vector(center[0], center[1], center[2])
        self.view.opts["distance"] = float(np.linalg.norm(spans) * 1.2)

    def _update_axis_item(self) -> None:
        if not self.axes_checkbox.isChecked():
            if self._axis_item is not None:
                self.view.removeItem(self._axis_item)
                self._axis_item = None
            return
        if self._axis_item is None:
            axis = _AdjustableAxisItem() if '_AdjustableAxisItem' in globals() else gl.GLAxisItem()
            axis.setSize(1, 1, 1)
            self.view.addItem(axis)
            self._axis_item = axis
        mins, maxs = self._current_bounds if self._current_bounds else (
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )
        spans = np.maximum(maxs - mins, 1e-3)
        self._axis_item.setSize(spans[0], spans[1], spans[2])
        if hasattr(self._axis_item, "setLineWidth"):
            self._axis_item.setLineWidth(2.0)

    def _on_color_change(self, _index: int) -> None:
        self._update_mesh()
        if not self._initialising:
            self._update_status()

    def _on_opacity_change(self, value: int) -> None:
        self.opacity_label.setText(f"{value / 100.0:.2f}")
        self._update_mesh()
        if not self._initialising:
            self._update_status()

    def _on_axes_toggle(self, _checked: bool) -> None:
        self._update_axis_item()
        if not self._initialising:
            self.view.update()

    def _on_light_toggle(self, checked: bool) -> None:
        self._lighting_enabled = bool(checked)
        self._apply_mesh_shader()
        if not self._initialising and self._mesh_item is not None:
            self._mesh_item.update()
        self._update_status()


