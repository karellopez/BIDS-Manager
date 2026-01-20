"""3-D surface rendering dialog for GIFTI meshes."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
import matplotlib.pyplot as plt

from .scene_utils import (
    HAS_PYQTGRAPH,
    _AdjustableAxisItem,
    _SLICE_ORIENTATIONS,
    _SliceControl,
    _create_directional_light_shader,
    _create_flat_color_shader,
    gl,
    pg,
)
from .widgets import ShrinkableScrollArea


class Surface3DDialog(QDialog):
    """Interactive renderer for surface meshes from GIFTI or FreeSurfer."""

    def __init__(
        self,
        parent,
        vertices: np.ndarray,
        faces: np.ndarray,
        scalars: Optional[dict[str, np.ndarray]] = None,
        meta: Optional[dict[str, Any]] = None,
        title: Optional[str] = None,
        dark_theme: bool = False,
    ) -> None:
        super().__init__(parent)
        if not HAS_PYQTGRAPH:
            raise RuntimeError(
                "Surface rendering requires the optional 'pyqtgraph' dependency."
            )
        self._initialising = True
        self.setWindowTitle(title or "Surface Viewer")
        self.resize(1200, 900)
        self.setMinimumSize(780, 560)
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
        self._meta = meta or {}
        self._dark_theme = bool(dark_theme)
        self._fg_color = "#f0f0f0" if self._dark_theme else "#202020"
        self._canvas_bg = "#202020" if self._dark_theme else "#ffffff"
        self._axis_label_items: list[gl.GLTextItem] = []
        self._data_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._slice_controls: dict[str, _SliceControl] = {}

        self._scalar_fields: dict[str, np.ndarray] = {}
        if scalars:
            for name, values in scalars.items():
                arr = np.asarray(values, dtype=np.float32).reshape(-1)
                if arr.shape[0] != verts.shape[0]:
                    continue
                safe_name = str(name).strip() or f"Field {len(self._scalar_fields) + 1}"
                self._scalar_fields[safe_name] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        self._mesh_item: Optional[gl.GLMeshItem] = None
        self._axis_item: Optional[gl.GLAxisItem] = None
        self._current_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._colormap_cache: dict[tuple[str, float], mcolors.Colormap] = {}
        self._light_shader = _create_directional_light_shader()
        self._flat_shader = _create_flat_color_shader()
        self._lighting_enabled = True

        if self._vertices.size:
            mins = np.min(self._vertices, axis=0).astype(np.float32, copy=False)
            maxs = np.max(self._vertices, axis=0).astype(np.float32, copy=False)
            self._data_bounds = (mins.copy(), maxs.copy())

        layout = QVBoxLayout(self)

        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(12)
        layout.addWidget(self._splitter)

        self._view_container = QWidget()
        view_layout = QVBoxLayout(self._view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        # ``GLViewWidget`` renders using OpenGL so panning/zooming the scene does
        # not require recomputing the mesh when interacting with the viewport.
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setBackgroundColor(self._canvas_bg)
        self.view.opts["distance"] = 200
        self.view.opts["elevation"] = 20
        self.view.opts["azimuth"] = -60
        view_layout.addWidget(self.view)

        self._splitter.addWidget(self._view_container)

        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(6, 6, 6, 6)
        settings_layout.setSpacing(10)

        # Allow the combined colour bar + control pane to be docked on the
        # side or bottom of the viewport.
        placement_layout = QHBoxLayout()
        placement_layout.setSpacing(6)
        placement_layout.addWidget(QLabel("Panel placement:"))
        self.panel_location_combo = QComboBox()
        self.panel_location_combo.addItems(["Bottom", "Left", "Right"])
        placement_layout.addWidget(self.panel_location_combo)
        placement_layout.addStretch()
        settings_layout.addLayout(placement_layout)

        scalar_row = QHBoxLayout()
        scalar_row.setContentsMargins(0, 0, 0, 0)
        scalar_row.setSpacing(8)
        scalar_row.addWidget(QLabel("Scalar:"))
        self.scalar_combo = QComboBox()
        self.scalar_combo.addItem("Constant colour", userData=None)
        for name in sorted(self._scalar_fields):
            self.scalar_combo.addItem(name, userData=name)
        self.scalar_combo.setEnabled(bool(self._scalar_fields))
        scalar_row.addWidget(self.scalar_combo, 1)
        scalar_row.addStretch()
        settings_layout.addLayout(scalar_row)

        appearance_group = QGroupBox("Appearance")
        appearance_layout = QGridLayout(appearance_group)
        appearance_layout.setColumnStretch(1, 1)

        row = 0
        appearance_layout.addWidget(QLabel("Colormap:"), row, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "twilight",
            "cubehelix",
            "Spectral",
            "coolwarm",
            "YlGnBu",
            "Greys",
            "bone",
        ])
        self.colormap_combo.setToolTip(
            "Select the matplotlib colormap for vertex intensities.",
        )
        appearance_layout.addWidget(self.colormap_combo, row, 1)

        row += 1
        appearance_layout.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.setToolTip(
            "Global alpha applied to rendered surfaces.",
        )
        appearance_layout.addWidget(self.opacity_slider, row, 1)
        self.opacity_label = QLabel("0.80")
        appearance_layout.addWidget(self.opacity_label, row, 2)

        row += 1
        appearance_layout.addWidget(QLabel("Colour intensity:"), row, 0)
        self.color_intensity_slider = QSlider(Qt.Horizontal)
        self.color_intensity_slider.setRange(10, 200)
        self.color_intensity_slider.setValue(100)
        self.color_intensity_slider.setToolTip(
            "Scale factor applied to RGB values after colormap lookup (0.1–2.0×).",
        )
        appearance_layout.addWidget(self.color_intensity_slider, row, 1)
        self.color_intensity_label = QLabel("1.00×")
        appearance_layout.addWidget(self.color_intensity_label, row, 2)
        settings_layout.addWidget(appearance_group)

        # Keep the colour bar within the scroll area so it collapses together
        # with the rest of the surface controls.
        colorbar_group = QGroupBox("Colour bar")
        colorbar_layout = QVBoxLayout(colorbar_group)
        colorbar_layout.setContentsMargins(8, 6, 8, 6)
        colorbar_layout.setSpacing(4)
        self._colorbar_canvas = FigureCanvas(plt.Figure(figsize=(2.6, 1.4)))
        self._colorbar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._colorbar_canvas.setFixedHeight(180)
        self._colorbar_canvas.figure.patch.set_facecolor(self._canvas_bg)
        colorbar_layout.addWidget(self._colorbar_canvas)
        settings_layout.addWidget(colorbar_group)

        axes_group = QGroupBox("Axes")
        axes_layout = QGridLayout(axes_group)
        axes_layout.setColumnStretch(1, 1)

        row = 0
        self.axes_checkbox = QCheckBox("Show axes")
        self.axes_checkbox.setChecked(False)
        axes_layout.addWidget(self.axes_checkbox, row, 0, 1, 3)

        row += 1
        axes_layout.addWidget(QLabel("Axis thickness:"), row, 0)
        self.axis_thickness_slider = QSlider(Qt.Horizontal)
        self.axis_thickness_slider.setRange(1, 12)
        self.axis_thickness_slider.setValue(3)
        self.axis_thickness_slider.setToolTip(
            "Line width of the anatomical axes (pixels).",
        )
        self.axis_thickness_slider.setEnabled(False)
        axes_layout.addWidget(self.axis_thickness_slider, row, 1)
        self.axis_thickness_value = QLabel("3 px")
        axes_layout.addWidget(self.axis_thickness_value, row, 2)

        row += 1
        self.axis_labels_checkbox = QCheckBox("Show axis labels")
        self.axis_labels_checkbox.setChecked(False)
        self.axis_labels_checkbox.setEnabled(False)
        axes_layout.addWidget(self.axis_labels_checkbox, row, 0, 1, 3)
        settings_layout.addWidget(axes_group)

        slice_group = QGroupBox("Slice planes")
        slice_layout = QVBoxLayout(slice_group)
        slice_layout.setContentsMargins(8, 4, 8, 4)
        slice_layout.setSpacing(6)
        self._slice_controls.clear()
        for key, axis_idx, neg_name, pos_name in _SLICE_ORIENTATIONS:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(6)
            checkbox = QCheckBox(key.capitalize())
            checkbox.setChecked(False)
            row_layout.addWidget(checkbox)
            row_layout.addSpacing(4)
            row_layout.addWidget(QLabel(f"{neg_name}:"))
            min_slider = QSlider(Qt.Horizontal)
            min_slider.setRange(0, 100)
            min_slider.setValue(0)
            min_slider.setEnabled(False)
            row_layout.addWidget(min_slider, 1)
            min_value = QLabel("0.0 mm")
            row_layout.addWidget(min_value)
            row_layout.addSpacing(6)
            row_layout.addWidget(QLabel(f"{pos_name}:"))
            max_slider = QSlider(Qt.Horizontal)
            max_slider.setRange(0, 100)
            max_slider.setValue(100)
            max_slider.setEnabled(False)
            row_layout.addWidget(max_slider, 1)
            max_value = QLabel("0.0 mm")
            row_layout.addWidget(max_value)
            row_layout.addStretch()
            slice_layout.addLayout(row_layout)
            control = _SliceControl(
                checkbox=checkbox,
                min_slider=min_slider,
                max_slider=max_slider,
                min_value=min_value,
                max_value=max_value,
                axis=axis_idx,
                negative_name=neg_name,
                positive_name=pos_name,
            )
            self._slice_controls[key] = control
            checkbox.toggled.connect(lambda checked, name=key: self._on_slice_toggle(name, checked))
            min_slider.valueChanged.connect(
                lambda value, name=key: self._on_slice_slider_change(name, "min", value)
            )
            max_slider.valueChanged.connect(
                lambda value, name=key: self._on_slice_slider_change(name, "max", value)
            )
        slice_layout.addStretch()
        settings_layout.addWidget(slice_group)

        self.lighting_group = QGroupBox("Lighting")
        light_layout = QGridLayout(self.lighting_group)
        light_row = 0
        self.light_enable_checkbox = QCheckBox("Enable lighting")
        self.light_enable_checkbox.setChecked(True)
        light_layout.addWidget(self.light_enable_checkbox, light_row, 0, 1, 3)
        light_row += 1
        light_layout.addWidget(QLabel("Azimuth:"), light_row, 0)
        self.light_azimuth_slider = QSlider(Qt.Horizontal)
        self.light_azimuth_slider.setRange(-180, 180)
        self.light_azimuth_slider.setValue(-45)
        self.light_azimuth_slider.setToolTip(
            "Horizontal direction of the light source relative to the mesh (°).",
        )
        light_layout.addWidget(self.light_azimuth_slider, light_row, 1)
        self.light_azimuth_label = QLabel("-45°")
        light_layout.addWidget(self.light_azimuth_label, light_row, 2)

        light_row += 1
        light_layout.addWidget(QLabel("Elevation:"), light_row, 0)
        self.light_elevation_slider = QSlider(Qt.Horizontal)
        self.light_elevation_slider.setRange(-90, 90)
        self.light_elevation_slider.setValue(30)
        self.light_elevation_slider.setToolTip(
            "Vertical angle of the light source relative to the mesh (°).",
        )
        light_layout.addWidget(self.light_elevation_slider, light_row, 1)
        self.light_elevation_label = QLabel("30°")
        light_layout.addWidget(self.light_elevation_label, light_row, 2)

        light_row += 1
        light_layout.addWidget(QLabel("Intensity:"), light_row, 0)
        self.light_intensity_slider = QSlider(Qt.Horizontal)
        self.light_intensity_slider.setRange(10, 300)
        self.light_intensity_slider.setValue(130)
        self.light_intensity_slider.setToolTip(
            "Diffuse lighting strength (0.1–3.0×). Ambient light stays constant.",
        )
        light_layout.addWidget(self.light_intensity_slider, light_row, 1)
        self.light_intensity_label = QLabel("1.30×")
        light_layout.addWidget(self.light_intensity_label, light_row, 2)
        if self._light_shader is None and self._flat_shader is None:
            self.lighting_group.setEnabled(False)
        settings_layout.addWidget(self.lighting_group)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        settings_layout.addWidget(self.status_label)
        settings_layout.addStretch()

        self._panel_scroll = ShrinkableScrollArea()
        self._panel_scroll.setWidget(settings_container)
        self._panel_scroll.setWidgetResizable(True)
        self._panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._splitter.addWidget(self._panel_scroll)
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 1)

        self.panel_location_combo.currentTextChanged.connect(self._on_panel_location_change)
        self.scalar_combo.currentIndexChanged.connect(self._on_scalar_change)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_change)
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        self.color_intensity_slider.valueChanged.connect(self._on_color_intensity_change)
        self.axes_checkbox.toggled.connect(self._on_axes_toggle)
        self.axis_thickness_slider.valueChanged.connect(self._on_axis_thickness_change)
        self.axis_labels_checkbox.toggled.connect(self._on_axis_labels_toggle)
        self.light_enable_checkbox.toggled.connect(self._on_light_enabled_toggle)
        self.light_azimuth_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_elevation_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_intensity_slider.valueChanged.connect(self._on_light_setting_change)

        self._apply_panel_layout(self.panel_location_combo.currentText())
        self._update_light_controls_enabled()

        # Prepare labels and shader uniforms before the first draw call.
        self._on_axis_thickness_change(self.axis_thickness_slider.value())
        self._update_slice_labels()
        self._on_color_intensity_change(self.color_intensity_slider.value())
        self._update_light_labels()
        self._update_light_shader()

        self._initialising = False
        self._update_plot()

    def _apply_panel_layout(self, location: str) -> None:
        """Reorder the settings splitter when users move the control pane."""

        if not hasattr(self, "_splitter"):
            return

        normalised = (location or "bottom").strip().lower()
        if normalised not in {"bottom", "left", "right"}:
            normalised = "bottom"

        view_first = True
        if normalised == "bottom":
            self._splitter.setOrientation(Qt.Vertical)
            view_first = True
        else:
            self._splitter.setOrientation(Qt.Horizontal)
            view_first = normalised != "left"

        widgets = [self._view_container, self._panel_scroll]
        if not view_first:
            widgets.reverse()
        for index, widget in enumerate(widgets):
            if self._splitter.indexOf(widget) != index:
                self._splitter.insertWidget(index, widget)

        if view_first:
            self._splitter.setStretchFactor(0, 4)
            self._splitter.setStretchFactor(1, 1)
        else:
            self._splitter.setStretchFactor(0, 1)
            self._splitter.setStretchFactor(1, 4)

    def _on_panel_location_change(self, location: str) -> None:
        self._apply_panel_layout(location)

    def _update_light_controls_enabled(self) -> None:
        enabled = bool(self.lighting_group.isEnabled() and self._lighting_enabled)
        for widget in (
            self.light_azimuth_slider,
            self.light_elevation_slider,
            self.light_intensity_slider,
        ):
            widget.setEnabled(enabled)
        for label in (
            self.light_azimuth_label,
            self.light_elevation_label,
            self.light_intensity_label,
        ):
            label.setEnabled(enabled)

    def _apply_mesh_shader(self) -> None:
        if self._mesh_item is None:
            return
        if self._light_shader is not None and self._lighting_enabled:
            self._mesh_item.setShader(self._light_shader)
        elif self._flat_shader is not None:
            self._mesh_item.setShader(self._flat_shader)
        else:
            self._mesh_item.setShader("shaded")

    def _on_light_enabled_toggle(self, checked: bool) -> None:
        self._lighting_enabled = bool(checked)
        self._update_light_controls_enabled()
        if self._lighting_enabled:
            self._update_light_shader()
        else:
            self._apply_mesh_shader()
        if not self._initialising and self._mesh_item is not None:
            self._mesh_item.update()
            self.view.update()

    def _current_color_intensity(self) -> float:
        return max(0.1, self.color_intensity_slider.value() / 100.0)

    def _get_adjusted_colormap(self, cmap_name: str) -> mcolors.Colormap:
        intensity = round(self._current_color_intensity(), 3)
        key = (cmap_name, intensity)
        cached = self._colormap_cache.get(key)
        if cached is not None:
            return cached
        base = plt.get_cmap(cmap_name)
        samples = base(np.linspace(0.0, 1.0, 512))
        samples[:, :3] = np.clip(samples[:, :3] * intensity, 0.0, 1.0)
        cmap = mcolors.ListedColormap(samples, name=f"{cmap_name}_x{intensity:.3f}")
        self._colormap_cache[key] = cmap
        return cmap

    def _map_colors(
        self, values: np.ndarray, cmap: mcolors.Colormap, alpha: float
    ) -> np.ndarray:
        # ``values`` are pre-normalised to [0, 1] so we can reuse matplotlib
        # colour maps and simply blend in the requested opacity.
        colours = np.asarray(cmap(np.clip(values, 0.0, 1.0)), dtype=np.float32)
        colours[:, 3] = np.clip(colours[:, 3] * alpha, 0.0, 1.0)
        return colours

    def _update_slice_labels(self) -> None:
        if not self._slice_controls:
            return
        bounds = self._data_bounds
        has_bounds = bounds is not None
        mins: Optional[np.ndarray]
        maxs: Optional[np.ndarray]
        if has_bounds:
            mins, maxs = bounds  # type: ignore[assignment]
        else:
            mins = maxs = None
        for control in self._slice_controls.values():
            slider_enabled = bool(has_bounds and control.checkbox.isChecked())
            control.min_slider.setEnabled(slider_enabled)
            control.max_slider.setEnabled(slider_enabled)
            if mins is None or maxs is None:
                control.min_value.setText("--")
                control.max_value.setText("--")
                continue
            axis = control.axis
            axis_min = float(mins[axis])
            axis_max = float(maxs[axis])
            span = axis_max - axis_min
            min_frac = control.min_slider.value() / 100.0
            max_frac = control.max_slider.value() / 100.0
            min_mm = axis_min + span * min_frac
            max_mm = axis_min + span * max_frac
            control.min_value.setText(f"{min_mm:.1f} mm")
            control.max_value.setText(f"{max_mm:.1f} mm")

    def _active_slice_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not self._slice_controls or self._data_bounds is None:
            return None
        mins, maxs = self._data_bounds
        min_bounds = np.asarray(mins, dtype=np.float32).copy()
        max_bounds = np.asarray(maxs, dtype=np.float32).copy()
        active = False
        for control in self._slice_controls.values():
            if not control.checkbox.isChecked():
                continue
            axis = control.axis
            axis_min = float(mins[axis])
            axis_max = float(maxs[axis])
            span = axis_max - axis_min
            min_frac = control.min_slider.value() / 100.0
            max_frac = control.max_slider.value() / 100.0
            min_bounds[axis] = float(axis_min + span * min_frac)
            max_bounds[axis] = float(axis_min + span * max_frac)
            active = True
        if not active:
            return None
        return min_bounds, max_bounds

    def _slice_status_suffix(self) -> str:
        bounds = self._active_slice_bounds()
        if bounds is None:
            return ""
        mins, maxs = bounds
        parts: list[str] = []
        for control in self._slice_controls.values():
            if not control.checkbox.isChecked():
                continue
            axis = control.axis
            parts.append(
                f"{control.negative_name}–{control.positive_name} "
                f"{mins[axis]:.1f}–{maxs[axis]:.1f} mm"
            )
        if not parts:
            return ""
        return " Slice windows: " + ", ".join(parts) + "."

    def _mask_coords(
        self, coords: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        mins, maxs = bounds
        if coords.size == 0:
            return np.zeros(0, dtype=bool)
        mask = np.ones(coords.shape[0], dtype=bool)
        for axis in range(3):
            mask &= (coords[:, axis] >= mins[axis]) & (coords[:, axis] <= maxs[axis])
        return mask

    def _apply_slice_to_points(
        self, coords: np.ndarray, values: Optional[np.ndarray]
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[tuple[np.ndarray, np.ndarray]]]:
        bounds = self._active_slice_bounds()
        if bounds is None or coords.size == 0:
            return coords, values, None
        mask = self._mask_coords(coords, bounds)
        if not np.any(mask):
            empty_coords = coords[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_coords, empty_vals, bounds
        filtered_coords = coords[mask]
        filtered_values = values[mask] if values is not None else None
        return filtered_coords, filtered_values, bounds

    def _clip_mesh_to_slices(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        values: Optional[np.ndarray],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[tuple[np.ndarray, np.ndarray]],
    ]:
        bounds = self._active_slice_bounds()
        if bounds is None or verts.size == 0:
            return verts, faces, values, None
        mask = self._mask_coords(verts, bounds)
        if not np.any(mask):
            empty_verts = verts[:0].reshape(0, 3)
            empty_faces = faces[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_verts, empty_faces, empty_vals, bounds
        selected = np.where(mask)[0]
        index_map = -np.ones(mask.shape[0], dtype=np.int64)
        index_map[selected] = np.arange(selected.size, dtype=np.int64)
        mapped_faces = index_map[faces]
        valid = (mapped_faces >= 0).all(axis=1)
        mapped_faces = mapped_faces[valid]
        if mapped_faces.size == 0:
            empty_verts = verts[:0].reshape(0, 3)
            empty_faces = faces[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_verts, empty_faces, empty_vals, bounds
        new_verts = verts[selected]
        new_values = values[selected] if values is not None else None
        return new_verts, mapped_faces.astype(np.int32, copy=False), new_values, bounds

    def _on_slice_toggle(self, name: str, _checked: bool) -> None:
        if name not in self._slice_controls:
            return
        self._update_slice_labels()
        if not self._initialising:
            self._update_plot()

    def _on_slice_slider_change(self, name: str, which: str, value: int) -> None:
        control = self._slice_controls.get(name)
        if control is None:
            return
        if which == "min":
            other = control.max_slider
            if value > other.value():
                other.blockSignals(True)
                other.setValue(value)
                other.blockSignals(False)
        else:
            other = control.min_slider
            if value < other.value():
                other.blockSignals(True)
                other.setValue(value)
                other.blockSignals(False)
        self._update_slice_labels()
        if not self._initialising and control.checkbox.isChecked():
            self._update_plot()
    def _clear_colorbar(self) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        fig.patch.set_facecolor(self._canvas_bg)
        self._colorbar_canvas.draw_idle()

    def _update_colorbar(
        self, cmap: mcolors.Colormap, vmin: float, vmax: float, label: str = "Scalar value"
    ) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        # Render the colour bar horizontally so it mirrors the slice legend above
        # while preserving vertical space for the rest of the controls.
        ax = fig.add_axes([0.12, 0.35, 0.76, 0.32])
        norm = plt.Normalize(vmin, vmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(mappable, cax=ax, orientation="horizontal")
        colorbar.set_label(label, color=self._fg_color, labelpad=6)
        colorbar.ax.xaxis.set_ticks_position("bottom")
        colorbar.ax.xaxis.set_label_position("bottom")
        colorbar.ax.tick_params(colors=self._fg_color, which="both")
        for spine in colorbar.ax.spines.values():
            spine.set_color(self._fg_color)
        colorbar.ax.set_facecolor(self._canvas_bg)
        fig.patch.set_facecolor(self._canvas_bg)
        self._colorbar_canvas.draw_idle()

    def _on_color_intensity_change(self, value: int) -> None:
        self.color_intensity_label.setText(f"{value / 100.0:.2f}×")
        if not self._initialising:
            self._update_plot()

    def _update_light_labels(self) -> None:
        self.light_azimuth_label.setText(f"{self.light_azimuth_slider.value():+d}°")
        self.light_elevation_label.setText(
            f"{self.light_elevation_slider.value():+d}°"
        )
        self.light_intensity_label.setText(
            f"{self.light_intensity_slider.value() / 100.0:.2f}×"
        )

    def _update_light_shader(self) -> None:
        shader = self._light_shader
        if shader is None or not self._lighting_enabled:
            self._apply_mesh_shader()
            return
        azimuth = math.radians(self.light_azimuth_slider.value())
        elevation = math.radians(self.light_elevation_slider.value())
        cos_el = math.cos(elevation)
        direction = [
            float(cos_el * math.cos(azimuth)),
            float(cos_el * math.sin(azimuth)),
            float(math.sin(elevation)),
        ]
        shader["lightDir"] = direction
        shader["lightParams"] = [
            max(0.0, self.light_intensity_slider.value() / 100.0),
            0.35,
        ]
        self._apply_mesh_shader()

    def _on_light_setting_change(self, _value: int) -> None:
        self._update_light_labels()
        self._update_light_shader()
        if self._initialising:
            return
        if self._mesh_item is not None:
            self._mesh_item.update()
            self.view.update()

    def _update_scene_bounds(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        spans = np.maximum(maxs - mins, 1e-3)
        # Keep the camera centred on the mesh to avoid jarring jumps when users
        # switch between scalar fields or adjust opacity.
        center = (mins + maxs) / 2.0
        self._current_bounds = (mins, maxs)
        self.view.opts["center"] = pg.Vector(center[0], center[1], center[2])
        self.view.opts["distance"] = float(np.linalg.norm(spans) * 1.2)
        self._update_axis_item()

    def _on_scalar_change(self, _index: int) -> None:
        self._update_plot()

    def _on_colormap_change(self, _name: str) -> None:
        self._update_plot()

    def _on_opacity_change(self, value: int) -> None:
        self.opacity_label.setText(f"{value / 100.0:.2f}")
        self._update_plot()

    def _update_plot(self) -> None:
        self._clear_colorbar()
        if self._faces.size == 0 or self._vertices.size == 0:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self.status_label.setText("Surface mesh has no vertices or faces to render.")
            return

        alpha = self.opacity_slider.value() / 100.0
        cmap_name = self.colormap_combo.currentText() or "viridis"
        scalar_key = self.scalar_combo.currentData()
        cmap = self._get_adjusted_colormap(cmap_name)

        summary = (
            f"Surface mesh: {self._vertices.shape[0]:,} vertices / {self._faces.shape[0]:,} faces. "
        )

        normalised_values: Optional[np.ndarray] = None
        vmin: Optional[float] = None
        vmax: Optional[float] = None
        if scalar_key and scalar_key in self._scalar_fields:
            raw_values = self._scalar_fields[scalar_key]
            vmin = float(np.min(raw_values))
            vmax = float(np.max(raw_values))
            if math.isclose(vmin, vmax):
                normalised_values = np.zeros_like(raw_values, dtype=np.float32)
                vmin, vmax = vmin - 0.5, vmax + 0.5
            else:
                norm = (raw_values - vmin) / (vmax - vmin)
                normalised_values = np.clip(norm.astype(np.float32, copy=False), 0.0, 1.0)

        clipped_verts, clipped_faces, clipped_values, slice_bounds = self._clip_mesh_to_slices(
            self._vertices, self._faces, normalised_values
        )

        if clipped_verts.size == 0 or clipped_faces.size == 0:
            if slice_bounds is not None:
                if self._mesh_item is not None:
                    self.view.removeItem(self._mesh_item)
                    self._mesh_item = None
                self._clear_colorbar()
                mins, maxs = slice_bounds
                self._update_scene_bounds(mins, maxs)
                self.status_label.setText(
                    "Slice planes removed all surface triangles. "
                    "Adjust the slicer sliders or disable slicing."
                )
                return
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self.status_label.setText("Surface mesh has no vertices or faces to render.")
            return

        meshdata = gl.MeshData(vertexes=clipped_verts, faces=clipped_faces)
        if clipped_values is not None and scalar_key and scalar_key in self._scalar_fields:
            colours = self._map_colors(clipped_values, cmap, alpha)
            meshdata.setVertexColors(colours)
            assert vmin is not None and vmax is not None
            self._update_colorbar(cmap, vmin, vmax)
            summary += f"Scalar '{scalar_key}' range {vmin:.4g} – {vmax:.4g}. "
        else:
            base_color = cmap(0.6)
            colour = np.array(
                [[base_color[0], base_color[1], base_color[2], base_color[3] * alpha]],
                dtype=np.float32,
            )
            meshdata.setVertexColors(np.repeat(colour, clipped_verts.shape[0], axis=0))
            self._update_colorbar(cmap, 0.0, 1.0, label="Colormap preview")
            summary += "Constant colouring applied. "

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
        self._update_light_shader()
        self._mesh_item.setGLOptions("translucent" if alpha < 0.999 else "opaque")

        mins = np.min(clipped_verts, axis=0)
        maxs = np.max(clipped_verts, axis=0)
        self._update_scene_bounds(mins, maxs)

        summary += f"Opacity {alpha:.2f}."
        if self._light_shader is not None or self._flat_shader is not None:
            summary += " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        if self._scalar_fields and (not scalar_key or scalar_key not in self._scalar_fields):
            summary += " Select a scalar field to colour the surface."
        summary += self._slice_status_suffix()
        self.status_label.setText(summary)


