"""3-D volume rendering dialog for NIfTI data."""

from __future__ import annotations

import importlib.util
import math
from typing import Any, Optional, Sequence

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
    QSpinBox,
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


_HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None
if _HAS_SKIMAGE:
    from skimage import measure as sk_measure
else:
    sk_measure = None

HAS_SKIMAGE = _HAS_SKIMAGE


class Volume3DDialog(QDialog):
    """Interactive 3-D renderer for NIfTI volumes."""

    def __init__(
        self,
        parent,
        data: np.ndarray,
        meta: Optional[dict] = None,
        voxel_sizes: Optional[Sequence[float]] = None,
        default_mode: Optional[str] = None,
        title: Optional[str] = None,
        dark_theme: bool = False,
    ) -> None:
        super().__init__(parent)
        if not HAS_PYQTGRAPH:
            raise RuntimeError(
                "3-D volume rendering requires the optional 'pyqtgraph' dependency."
            )
        self.setWindowTitle(title or "3D Volume Viewer")
        self.resize(1800, 1000)
        self.setMinimumSize(780, 560)
        self.setSizeGripEnabled(True)

        self._raw = np.asarray(data)
        if self._raw.ndim < 3:
            raise ValueError("3-D rendering requires data with at least three axes")

        self._meta = meta or {}
        self._voxel_sizes = self._normalise_voxel_sizes(voxel_sizes)
        self._voxel_sizes_vec = np.asarray(self._voxel_sizes, dtype=np.float32)
        self._dark_theme = bool(dark_theme)
        self._max_points = 120_000
        self._surface_step = 1
        self._initialising = True
        self._scalar_min = 0.0
        self._scalar_max = 0.0
        self._scalar_volume: Optional[np.ndarray] = None
        self._normalised_volume: Optional[np.ndarray] = None
        self._downsampled: Optional[np.ndarray] = None
        self._downsample_step = 1
        self._point_sorted_values: Optional[np.ndarray] = None
        self._point_sorted_indices: Optional[np.ndarray] = None
        self._point_shape: Optional[tuple[int, int, int]] = None
        self._scalar_hist_upper_edges: Optional[np.ndarray] = None
        self._scalar_hist_cumulative: Optional[np.ndarray] = None
        self._scatter_item: Optional[gl.GLScatterPlotItem] = None
        self._mesh_item: Optional[gl.GLMeshItem] = None
        self._axis_item: Optional[gl.GLAxisItem] = None
        self._current_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._colormap_cache: dict[tuple[str, float], mcolors.Colormap] = {}
        self._light_shader = _create_directional_light_shader()
        self._flat_shader = _create_flat_color_shader()
        self._lighting_enabled = True

        self._fg_color = "#f0f0f0" if self._dark_theme else "#202020"
        self._canvas_bg = "#202020" if self._dark_theme else "#ffffff"
        self._axis_label_items: list[gl.GLTextItem] = []
        self._data_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._slice_controls: dict[str, _SliceControl] = {}

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
        # not require recomputing the voxel subset on every interaction.
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

        # Allow users to dock the settings pane to whichever edge feels most
        # comfortable when working on small screens.
        placement_layout = QHBoxLayout()
        placement_layout.setSpacing(6)
        placement_label = QLabel("Panel placement:")
        placement_layout.addWidget(placement_label)
        self.panel_location_combo = QComboBox()
        self.panel_location_combo.addItems(["Bottom", "Left", "Right"])
        # Default to a side-by-side layout so the viewport opens on the left.
        self.panel_location_combo.setCurrentText("Right")
        placement_layout.addWidget(self.panel_location_combo)
        placement_layout.addStretch()
        settings_layout.addLayout(placement_layout)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        agg_label = QLabel("Aggregation:")
        header_row.addWidget(agg_label)
        self.agg_combo = QComboBox()
        self._aggregators = {}
        self._init_aggregator_options()
        header_row.addWidget(self.agg_combo)
        header_row.addSpacing(12)

        render_label = QLabel("Render mode:")
        header_row.addWidget(render_label)
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["Point cloud", "Surface mesh"])
        self.render_mode_combo.setToolTip(
            "Switch between scattered voxels and iso-surface rendering.",
        )
        header_row.addWidget(self.render_mode_combo)
        header_row.addStretch()
        controls_layout.addLayout(header_row)

        # Stack the render-mode sliders beneath the mode selector so they are
        # visually associated with the chosen representation.
        slider_grid = QGridLayout()
        slider_grid.setContentsMargins(0, 0, 0, 0)
        slider_grid.setHorizontalSpacing(8)
        slider_grid.setVerticalSpacing(6)
        slider_grid.setColumnStretch(1, 1)

        self.thresh_text = QLabel("Threshold:")
        slider_grid.addWidget(self.thresh_text, 0, 0)
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(60)
        self.thresh_slider.setPageStep(5)
        self.thresh_slider.setToolTip(
            "Minimum normalised intensity included in the rendering.",
        )
        self.thresh_slider.valueChanged.connect(self._update_plot)
        slider_grid.addWidget(self.thresh_slider, 0, 1)
        self.thresh_label = QLabel("0.60")
        slider_grid.addWidget(self.thresh_label, 0, 2)

        self.point_label = QLabel("Point size:")
        slider_grid.addWidget(self.point_label, 1, 0)
        self.point_slider = QSlider(Qt.Horizontal)
        self.point_slider.setRange(1, 12)
        self.point_slider.setValue(4)
        self.point_slider.setToolTip("Marker diameter for point-cloud rendering.")
        self.point_slider.valueChanged.connect(self._update_plot)
        slider_grid.addWidget(self.point_slider, 1, 1)

        controls_layout.addLayout(slider_grid)
        controls_layout.addStretch()
        settings_layout.addWidget(controls_widget)

        options_group = QGroupBox("Rendering options")
        options = QGridLayout(options_group)
        options.setColumnStretch(1, 1)

        row = 0
        options.addWidget(QLabel("Colormap:"), row, 0)
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
            "Select the matplotlib colormap for voxel intensities.",
        )
        self.colormap_combo.currentTextChanged.connect(lambda _: self._update_plot())
        options.addWidget(self.colormap_combo, row, 1)

        row += 1
        options.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.setToolTip(
            "Global alpha applied to rendered points or the surface mesh.",
        )
        options.addWidget(self.opacity_slider, row, 1)
        self.opacity_label = QLabel("0.80")
        options.addWidget(self.opacity_label, row, 2)

        row += 1
        options.addWidget(QLabel("Colour intensity:"), row, 0)
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(10, 200)
        self.intensity_slider.setValue(100)
        self.intensity_slider.setToolTip(
            "Scale factor applied to RGB values after colormap lookup (0.1–2.0×).",
        )
        options.addWidget(self.intensity_slider, row, 1)
        self.intensity_label = QLabel("1.00×")
        options.addWidget(self.intensity_label, row, 2)

        row += 1
        options.addWidget(QLabel("Maximum points:"), row, 0)
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(1_000, 20_000_000)
        self.max_points_spin.setSingleStep(5_000)
        self.max_points_spin.setValue(self._max_points)
        self.max_points_spin.setToolTip(
            "Upper bound on voxels displayed in point-cloud mode.",
        )
        options.addWidget(self.max_points_spin, row, 1)

        row += 1
        options.addWidget(QLabel("Downsample step:"), row, 0)
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(0, 8)
        self.downsample_spin.setValue(0)
        self.downsample_spin.setToolTip(
            "Manual voxel stride applied before rendering (0 = automatic).",
        )
        options.addWidget(self.downsample_spin, row, 1)

        row += 1
        self.surface_step_label = QLabel("Marching cubes step:")
        options.addWidget(self.surface_step_label, row, 0)
        self.surface_step_spin = QSpinBox()
        self.surface_step_spin.setRange(1, 6)
        self.surface_step_spin.setValue(self._surface_step)
        self.surface_step_spin.setToolTip(
            "Sampling stride for marching cubes when computing iso-surfaces.",
        )
        options.addWidget(self.surface_step_spin, row, 1)

        row += 1
        self.axes_checkbox = QCheckBox("Show axes")
        self.axes_checkbox.setChecked(False)
        options.addWidget(self.axes_checkbox, row, 0, 1, 2)

        row += 1
        options.addWidget(QLabel("Axis thickness:"), row, 0)
        self.axis_thickness_slider = QSlider(Qt.Horizontal)
        self.axis_thickness_slider.setRange(1, 12)
        self.axis_thickness_slider.setValue(3)
        self.axis_thickness_slider.setToolTip(
            "Line width of the anatomical axes (pixels).",
        )
        options.addWidget(self.axis_thickness_slider, row, 1)
        self.axis_thickness_value = QLabel("3 px")
        options.addWidget(self.axis_thickness_value, row, 2)

        row += 1
        self.axis_labels_checkbox = QCheckBox("Show axis labels")
        self.axis_labels_checkbox.setChecked(False)
        options.addWidget(self.axis_labels_checkbox, row, 0, 1, 2)
        self.axis_thickness_slider.setEnabled(False)
        self.axis_labels_checkbox.setEnabled(False)
        settings_layout.addWidget(options_group)

        # Embed the colour bar alongside the rest of the controls so it folds
        # away with the settings pane.
        colorbar_group = QGroupBox("Colour bar")
        colorbar_layout = QVBoxLayout(colorbar_group)
        colorbar_layout.setContentsMargins(8, 6, 8, 6)
        colorbar_layout.setSpacing(4)
        self._colorbar_canvas = FigureCanvas(plt.Figure(figsize=(2.6, 1.4)))
        self._colorbar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._colorbar_canvas.setFixedHeight(180)
        self._colorbar_canvas.figure.patch.set_facecolor(self._canvas_bg)
        colorbar_layout.addWidget(self._colorbar_canvas)

        # Host the colour bar in its own lightweight container so it can share
        # the retractable pane while remaining independent from the parameter list.
        self._colorbar_panel = QWidget()
        colorbar_panel_layout = QVBoxLayout(self._colorbar_panel)
        colorbar_panel_layout.setContentsMargins(6, 6, 6, 6)
        colorbar_panel_layout.setSpacing(6)
        colorbar_panel_layout.addWidget(colorbar_group)

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
            min_slider.valueChanged.connect(lambda value, name=key: self._on_slice_slider_change(name, "min", value))
            max_slider.valueChanged.connect(lambda value, name=key: self._on_slice_slider_change(name, "max", value))
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

        # A nested splitter keeps the scrollable controls and the colour bar in
        # separate panes so each can be resized without affecting the other.
        self._panel_splitter = QSplitter(Qt.Vertical)
        self._panel_splitter.setChildrenCollapsible(False)
        self._panel_splitter.addWidget(self._panel_scroll)
        self._panel_splitter.addWidget(self._colorbar_panel)
        self._panel_splitter.setStretchFactor(0, 3)
        self._panel_splitter.setStretchFactor(1, 1)

        self._panel_container = QWidget()
        panel_container_layout = QVBoxLayout(self._panel_container)
        panel_container_layout.setContentsMargins(0, 0, 0, 0)
        panel_container_layout.setSpacing(0)
        panel_container_layout.addWidget(self._panel_splitter)

        self._splitter.addWidget(self._panel_container)

        default_name = None
        if default_mode and default_mode in self._aggregators:
            default_name = default_mode
        elif self._raw.ndim > 3:
            if self._meta.get("is_rgb"):
                default_name = "Mean |value|"
            else:
                default_name = "RMS (DTI)"

        if default_name:
            self.agg_combo.setCurrentText(default_name)

        self.panel_location_combo.currentTextChanged.connect(self._on_panel_location_change)
        self.agg_combo.currentTextChanged.connect(self._on_agg_change)
        self.render_mode_combo.currentTextChanged.connect(self._on_render_mode_change)
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        self.intensity_slider.valueChanged.connect(self._on_intensity_change)
        self.max_points_spin.valueChanged.connect(self._on_max_points_change)
        self.downsample_spin.valueChanged.connect(self._on_downsample_change)
        self.surface_step_spin.valueChanged.connect(self._on_surface_step_change)
        self.axes_checkbox.toggled.connect(self._on_axes_toggle)
        self.axis_thickness_slider.valueChanged.connect(self._on_axis_thickness_change)
        self.axis_labels_checkbox.toggled.connect(self._on_axis_labels_toggle)
        self.light_enable_checkbox.toggled.connect(self._on_light_enabled_toggle)
        self.light_azimuth_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_elevation_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_intensity_slider.valueChanged.connect(self._on_light_setting_change)

        self._apply_panel_layout(self.panel_location_combo.currentText())
        self._update_light_controls_enabled()

        # Prime the UI labels without triggering expensive redraws while we are
        # still constructing the dialog.
        self._on_intensity_change(self.intensity_slider.value())
        self._on_axis_thickness_change(self.axis_thickness_slider.value())
        self._update_slice_labels()
        self._update_light_labels()
        self._update_light_shader()

        self._initialising = False
        self._update_mode_dependent_controls()
        self._compute_scalar_volume()
        self._update_plot()

    def _apply_panel_layout(self, location: str) -> None:
        """Reposition the settings pane relative to the 3-D viewport."""

        if not hasattr(self, "_splitter"):
            return

        normalised = (location or "bottom").strip().lower()
        if normalised not in {"bottom", "left", "right"}:
            normalised = "bottom"

        orientation = Qt.Vertical if normalised == "bottom" else Qt.Horizontal
        self._splitter.setOrientation(orientation)

        view_first = True
        if orientation == Qt.Horizontal:
            view_first = normalised != "left"

        panel_widget = getattr(self, "_panel_container", None)
        widgets = [self._view_container, panel_widget]
        if panel_widget is None:
            widgets = [self._view_container]
        if not view_first and len(widgets) == 2:
            widgets.reverse()
        for index, widget in enumerate(widgets):
            if widget is None:
                continue
            if self._splitter.indexOf(widget) != index:
                self._splitter.insertWidget(index, widget)

        # Bias the main splitter so the viewport consumes roughly three quarters
        # of the available space in the default layout.
        stretch = (3, 1) if view_first else (1, 3)
        self._splitter.setStretchFactor(0, stretch[0])
        if len(widgets) > 1:
            self._splitter.setStretchFactor(1, stretch[1])
            scale = 120
            self._splitter.setSizes([stretch[0] * scale, stretch[1] * scale])

        panel_splitter = getattr(self, "_panel_splitter", None)
        if isinstance(panel_splitter, QSplitter):
            # Rotate the internal splitter so the colour bar always occupies its
            # own pane next to the parameter controls regardless of docking side.
            if orientation == Qt.Vertical:
                panel_splitter.setOrientation(Qt.Horizontal)
                panel_splitter.setSizes([220, 160])
            else:
                panel_splitter.setOrientation(Qt.Vertical)
                panel_splitter.setSizes([300, 160])

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

    def _normalise_voxel_sizes(self, voxel_sizes):
        if not voxel_sizes:
            return (1.0, 1.0, 1.0)
        values = tuple(float(v) for v in voxel_sizes[:3])
        if len(values) < 3:
            values = values + (1.0,) * (3 - len(values))
        return tuple(max(1e-6, abs(v)) for v in values)

    def _init_aggregator_options(self) -> None:
        if self._raw.ndim <= 3:
            self._aggregators["Intensity"] = "Intensity"
            self.agg_combo.addItem("Intensity")
            self.agg_combo.setEnabled(False)
        else:
            for name in ("RMS (DTI)", "Mean |value|", "Max |value|", "First volume"):
                self._aggregators[name] = name
                self.agg_combo.addItem(name)

    def _on_agg_change(self):
        if self._initialising:
            return
        self._compute_scalar_volume()
        self._update_plot()

    def _compute_scalar_volume(self) -> None:
        arr = np.asarray(self._raw, dtype=np.float32)
        if arr.ndim <= 3:
            scalar = arr
        else:
            axis = arr.ndim - 1
            mode = self.agg_combo.currentText()
            if mode == "RMS (DTI)":
                scalar = np.sqrt(np.nanmean(np.square(arr), axis=axis))
            elif mode == "Max |value|":
                scalar = np.nanmax(np.abs(arr), axis=axis)
            elif mode == "First volume":
                scalar = np.abs(np.take(arr, 0, axis=axis))
            else:
                scalar = np.nanmean(np.abs(arr), axis=axis)
        scalar = np.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0)
        self._scalar_volume = scalar
        finite = np.isfinite(scalar)
        if not finite.any():
            self._scalar_min = 0.0
            self._scalar_max = 0.0
            self._normalised_volume = np.zeros_like(scalar, dtype=np.float32)
        else:
            min_val = float(np.min(scalar[finite]))
            max_val = float(np.max(scalar[finite]))
            self._scalar_min = min_val
            self._scalar_max = max_val
            if abs(max_val - min_val) < 1e-6:
                self._normalised_volume = np.zeros_like(scalar, dtype=np.float32)
            else:
                norm = (scalar - min_val) / (max_val - min_val)
                self._normalised_volume = norm.astype(np.float32, copy=False)
        self._build_scalar_histogram()
        self._prepare_downsampled()

    def _build_scalar_histogram(self) -> None:
        self._scalar_hist_upper_edges = None
        self._scalar_hist_cumulative = None
        volume = self._normalised_volume
        if volume is None:
            return

        flattened = np.asarray(volume, dtype=np.float32).ravel()
        if flattened.size == 0:
            self._scalar_hist_upper_edges = np.empty(0, dtype=np.float32)
            self._scalar_hist_cumulative = np.empty(0, dtype=np.int64)
            return

        edges = np.linspace(0.0, 1.0, 513, dtype=np.float32)
        counts, _ = np.histogram(flattened, bins=edges)
        cumulative = np.cumsum(counts[::-1], dtype=np.int64)[::-1]
        self._scalar_hist_upper_edges = edges[1:]
        self._scalar_hist_cumulative = cumulative

    def _prepare_downsampled(self) -> None:
        vol = self._normalised_volume
        if vol is None:
            self._downsampled = None
            self._downsample_step = 1
            self._data_bounds = None
            self._update_slice_labels()
            return
        manual_step = 0
        if hasattr(self, "downsample_spin"):
            manual_step = int(self.downsample_spin.value())
        if manual_step > 0:
            step = manual_step
        else:
            step = 1
            total = float(vol.size)
            if total > self._max_points:
                step = int(np.ceil(np.cbrt(total / self._max_points)))
        step = max(1, step)
        slices = tuple(slice(None, None, step) for _ in range(3))
        self._downsampled = vol[slices]
        self._downsample_step = step
        if self._downsampled.size == 0:
            self._data_bounds = None
        else:
            ds_shape = np.ones(3, dtype=np.float32)
            dims = min(self._downsampled.ndim, 3)
            ds_shape[:dims] = np.asarray(self._downsampled.shape[:dims], dtype=np.float32)
            scale = self._voxel_sizes_vec * float(step)
            spans = np.maximum((ds_shape - 1.0) * scale, 0.0)
            mins = np.zeros(3, dtype=np.float32)
            self._data_bounds = (mins, mins + spans)
        self._update_slice_labels()
        self._rebuild_point_cache()

    def _rebuild_point_cache(self) -> None:
        self._point_sorted_values = None
        self._point_sorted_indices = None
        self._point_shape = None
        downsampled = self._downsampled
        if downsampled is None:
            return

        flat = np.asarray(downsampled, dtype=np.float32).ravel()
        self._point_shape = downsampled.shape
        if flat.size == 0:
            self._point_sorted_values = np.empty(0, dtype=np.float32)
            self._point_sorted_indices = np.empty(0, dtype=np.int64)
            return

        order = np.argsort(flat, kind="mergesort")
        self._point_sorted_indices = order.astype(np.int64, copy=False)
        self._point_sorted_values = flat.take(order)

    def _indices_to_world(
        self, indices: np.ndarray, shape: tuple[int, int, int], scale: np.ndarray
    ) -> np.ndarray:
        coords_mm = np.empty((indices.size, 3), dtype=np.float32)
        if indices.size == 0:
            return coords_mm

        plane = int(shape[1]) * int(shape[2])
        axis0 = indices // plane
        remainder = indices - axis0 * plane
        axis1 = remainder // int(shape[2])
        axis2 = remainder - axis1 * int(shape[2])

        coords_mm[:, 0] = axis0.astype(np.float32, copy=False)
        coords_mm[:, 1] = axis1.astype(np.float32, copy=False)
        coords_mm[:, 2] = axis2.astype(np.float32, copy=False)
        coords_mm *= scale
        return coords_mm

    def _estimate_voxels_above_threshold(self, thr: float) -> int:
        if self._scalar_hist_upper_edges is None or self._scalar_hist_cumulative is None:
            if self._normalised_volume is None:
                return 0
            return int(np.count_nonzero(self._normalised_volume >= thr))

        idx = int(np.searchsorted(self._scalar_hist_upper_edges, thr, side="left"))
        if idx >= self._scalar_hist_cumulative.size:
            return 0
        return int(self._scalar_hist_cumulative[idx])

    def _current_color_intensity(self) -> float:
        slider = getattr(self, "intensity_slider", None)
        if slider is None:
            return 1.0
        return max(0.1, slider.value() / 100.0)

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
        # Clamp to the normalised range and map to RGBA so OpenGL can upload a
        # single colour array for all voxels.
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

    def _remove_axis_labels(self) -> None:
        if not self._axis_label_items:
            return
        for item in self._axis_label_items:
            try:
                self.view.removeItem(item)
            except Exception:  # pragma: no cover - defensive cleanup
                continue
        self._axis_label_items.clear()

    def _update_axis_labels(self) -> None:
        self._remove_axis_labels()
        if (
            getattr(self, "axis_labels_checkbox", None) is None
            or not self.axes_checkbox.isChecked()
            or not self.axis_labels_checkbox.isChecked()
            or self._current_bounds is None
        ):
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-6)
        safe_spans = np.where(spans > 0, spans, 1.0)
        centre = (mins + maxs) / 2.0
        offsets = safe_spans * 0.05
        colour = QColor(self._fg_color)
        font = QFont("Helvetica", 13)

        for _name, axis_idx, neg_name, pos_name in _SLICE_ORIENTATIONS:
            if axis_idx == 0:
                neg_pos = [
                    float(mins[0] - offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(maxs[0] + offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
            elif axis_idx == 1:
                neg_pos = [
                    float(centre[0]),
                    float(mins[1] - offsets[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(maxs[1] + offsets[1]),
                    float(centre[2]),
                ]
            else:
                neg_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(mins[2] - offsets[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(maxs[2] + offsets[2]),
                ]
            for text, position in ((neg_name, neg_pos), (pos_name, pos_pos)):
                label = gl.GLTextItem()
                label.setData(text=text, pos=position, color=colour, font=font)
                self.view.addItem(label)
                self._axis_label_items.append(label)

    def _update_axis_item(self) -> None:
        if self._axis_item is not None:
            self.view.removeItem(self._axis_item)
            self._axis_item = None
        if not self.axes_checkbox.isChecked() or self._current_bounds is None:
            self._remove_axis_labels()
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-3)
        axis = _AdjustableAxisItem() if _AdjustableAxisItem is not None else gl.GLAxisItem()
        axis.setSize(spans[0], spans[1], spans[2])
        axis.translate(float(mins[0]), float(mins[1]), float(mins[2]))
        thickness = getattr(self, "axis_thickness_slider", None)
        if thickness is not None and hasattr(axis, "setLineWidth"):
            axis.setLineWidth(float(thickness.value()))
        self.view.addItem(axis)
        self._axis_item = axis
        self._update_axis_labels()

    def _on_axes_toggle(self, checked: bool) -> None:
        slider = getattr(self, "axis_thickness_slider", None)
        if slider is not None:
            slider.setEnabled(checked)
        label_cb = getattr(self, "axis_labels_checkbox", None)
        if label_cb is not None:
            label_cb.setEnabled(checked)
            if not checked:
                self._remove_axis_labels()
        self._update_axis_item()

    def _on_axis_thickness_change(self, value: int) -> None:
        if hasattr(self, "axis_thickness_value"):
            self.axis_thickness_value.setText(f"{value} px")
        if self._axis_item is not None and hasattr(self._axis_item, "setLineWidth"):
            self._axis_item.setLineWidth(float(value))
            self.view.update()

    def _on_axis_labels_toggle(self, checked: bool) -> None:
        if not checked:
            self._remove_axis_labels()
        else:
            self._update_axis_labels()

    def _on_intensity_change(self, value: int) -> None:
        self.intensity_label.setText(f"{value / 100.0:.2f}×")
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
        if self.render_mode_combo.currentText() == "Surface mesh" and self._mesh_item is not None:
            self._mesh_item.update()
            self.view.update()

    def _clear_colorbar(self) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        fig.patch.set_facecolor(self._canvas_bg)
        self._colorbar_canvas.draw_idle()

    def _update_colorbar(
        self,
        cmap: mcolors.Colormap,
        vmin: float = 0.0,
        vmax: float = 1.0,
        label: str = "Normalised intensity",
    ) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        # ``ScalarMappable`` draws a classic matplotlib colour bar giving users a
        # persistent reference for the current normalisation range. We render it
        # horizontally so it reads naturally underneath the legend label while
        # leaving more vertical room for the surrounding controls.
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

    def _remove_axis_labels(self) -> None:
        if not self._axis_label_items:
            return
        for item in self._axis_label_items:
            try:
                self.view.removeItem(item)
            except Exception:  # pragma: no cover - defensive cleanup
                continue
        self._axis_label_items.clear()

    def _update_axis_labels(self) -> None:
        self._remove_axis_labels()
        if (
            getattr(self, "axis_labels_checkbox", None) is None
            or not self.axes_checkbox.isChecked()
            or not self.axis_labels_checkbox.isChecked()
            or self._current_bounds is None
        ):
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-6)
        safe_spans = np.where(spans > 0, spans, 1.0)
        centre = (mins + maxs) / 2.0
        offsets = safe_spans * 0.05
        colour = QColor(self._fg_color)
        font = QFont("Helvetica", 13)

        for _name, axis_idx, neg_name, pos_name in _SLICE_ORIENTATIONS:
            if axis_idx == 0:
                neg_pos = [
                    float(mins[0] - offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(maxs[0] + offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
            elif axis_idx == 1:
                neg_pos = [
                    float(centre[0]),
                    float(mins[1] - offsets[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(maxs[1] + offsets[1]),
                    float(centre[2]),
                ]
            else:
                neg_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(mins[2] - offsets[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(maxs[2] + offsets[2]),
                ]
            for text, position in ((neg_name, neg_pos), (pos_name, pos_pos)):
                label = gl.GLTextItem()
                label.setData(text=text, pos=position, color=colour, font=font)
                self.view.addItem(label)
                self._axis_label_items.append(label)

    def _update_scene_bounds(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        spans = np.maximum(maxs - mins, 1e-3)
        # Update the camera centre/distance so the current geometry fits snugly
        # inside the viewport regardless of the downsampling level.
        center = (mins + maxs) / 2.0
        self._current_bounds = (mins, maxs)
        self.view.opts["center"] = pg.Vector(center[0], center[1], center[2])
        self.view.opts["distance"] = float(np.linalg.norm(spans) * 1.2)
        self._update_axis_item()

    def _update_axis_item(self) -> None:
        if self._axis_item is not None:
            self.view.removeItem(self._axis_item)
            self._axis_item = None
        if not self.axes_checkbox.isChecked() or self._current_bounds is None:
            self._remove_axis_labels()
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-3)
        axis = _AdjustableAxisItem() if _AdjustableAxisItem is not None else gl.GLAxisItem()
        axis.setSize(spans[0], spans[1], spans[2])
        axis.translate(float(mins[0]), float(mins[1]), float(mins[2]))
        thickness = getattr(self, "axis_thickness_slider", None)
        if thickness is not None and hasattr(axis, "setLineWidth"):
            axis.setLineWidth(float(thickness.value()))
        self.view.addItem(axis)
        self._axis_item = axis
        self._update_axis_labels()

    def _on_axes_toggle(self, checked: bool) -> None:
        slider = getattr(self, "axis_thickness_slider", None)
        if slider is not None:
            slider.setEnabled(checked)
        label_cb = getattr(self, "axis_labels_checkbox", None)
        if label_cb is not None:
            label_cb.setEnabled(checked)
            if not checked:
                self._remove_axis_labels()
        self._update_axis_item()

    def _on_axis_thickness_change(self, value: int) -> None:
        if hasattr(self, "axis_thickness_value"):
            self.axis_thickness_value.setText(f"{value} px")
        if self._axis_item is not None and hasattr(self._axis_item, "setLineWidth"):
            self._axis_item.setLineWidth(float(value))
            self.view.update()

    def _on_axis_labels_toggle(self, checked: bool) -> None:
        if not checked:
            self._remove_axis_labels()
        else:
            self._update_axis_labels()

    def _update_plot(self):
        if self._downsampled is None or self._normalised_volume is None:
            self.status_label.setText("No scalar volume available for rendering.")
            self._clear_colorbar()
            return

        thr = self.thresh_slider.value() / 100.0
        self.thresh_label.setText(f"{thr:.2f}")
        cmap = self.colormap_combo.currentText() or "viridis"
        alpha = self.opacity_slider.value() / 100.0
        mode = self.render_mode_combo.currentText()

        if mode == "Surface mesh":
            self._draw_surface_mesh(thr, cmap, alpha)
        else:
            self._draw_point_cloud(thr, cmap, alpha)

    def _handle_empty_point_cloud(self, shape: Sequence[int], scale: np.ndarray, thr: float) -> None:
        if self._scatter_item is not None:
            self.view.removeItem(self._scatter_item)
            self._scatter_item = None
        mins = np.zeros(3, dtype=np.float32)
        spans = np.maximum((np.asarray(shape, dtype=np.float32) - 1.0) * scale, 1e-3)
        maxs = mins + spans
        self._update_scene_bounds(mins, maxs)
        self._clear_colorbar()
        self.status_label.setText(
            f"No voxels above threshold {thr:.2f}. Lower the threshold to reveal data."
        )

    def _draw_point_cloud(self, thr: float, cmap_name: str, alpha: float) -> None:
        downsampled = self._downsampled
        if downsampled is None:
            self.status_label.setText("No data available for point-cloud rendering.")
            self._clear_colorbar()
            return

        if self._mesh_item is not None:
            # Switching from the iso-surface to point-cloud view discards the
            # cached ``GLMeshItem`` so only the scatter is redrawn.
            self.view.removeItem(self._mesh_item)
            self._mesh_item = None

        step = self._downsample_step
        scale = self._voxel_sizes_vec * float(step)
        shape = self._point_shape or downsampled.shape
        sorted_values = self._point_sorted_values
        sorted_indices = self._point_sorted_indices

        coords_mm: Optional[np.ndarray] = None
        values: Optional[np.ndarray] = None

        if (
            sorted_values is None
            or sorted_indices is None
            or self._point_shape is None
            or sorted_values.size == 0
        ):
            mask = downsampled >= thr
            coords = np.argwhere(mask)
            if coords.size == 0:
                self._handle_empty_point_cloud(shape, scale, thr)
                return
            coords_mm = coords.astype(np.float32, copy=False)
            coords_mm *= scale
            values = downsampled[mask].astype(np.float32, copy=False)
        else:
            start = int(np.searchsorted(sorted_values, thr, side="left"))
            if start >= sorted_values.size:
                self._handle_empty_point_cloud(shape, scale, thr)
                return

            selected_indices = sorted_indices[start:]
            selected_values = sorted_values[start:]
            if selected_indices.size > self._max_points:
                sample_idx = np.linspace(
                    0, selected_indices.size - 1, self._max_points, dtype=np.int64
                )
                selected_indices = selected_indices[sample_idx]
                selected_values = selected_values[sample_idx]

            if selected_indices.size == 0:
                self._handle_empty_point_cloud(shape, scale, thr)
                return

            if selected_indices.size > 1:
                order = np.argsort(selected_indices, kind="mergesort")
                selected_indices = selected_indices[order]
                selected_values = selected_values[order]

            coords_mm = self._indices_to_world(selected_indices, shape, scale)
            values = selected_values.astype(np.float32, copy=False)

        if coords_mm is None or values is None:
            self._handle_empty_point_cloud(shape, scale, thr)
            return

        coords_mm, values, slice_bounds = self._apply_slice_to_points(coords_mm, values)
        if coords_mm.size == 0:
            if slice_bounds is not None:
                if self._scatter_item is not None:
                    self.view.removeItem(self._scatter_item)
                    self._scatter_item = None
                self._clear_colorbar()
                mins, maxs = slice_bounds
                self._update_scene_bounds(mins, maxs)
                self.status_label.setText(
                    "Slice planes removed all voxels at threshold "
                    f"{thr:.2f}. Adjust the slicer sliders or disable slicing."
                )
                return
            self._handle_empty_point_cloud(shape, scale, thr)
            return

        if values is None:
            values = np.empty(coords_mm.shape[0], dtype=np.float32)
        cmap = self._get_adjusted_colormap(cmap_name)
        colors = self._map_colors(values, cmap, alpha)

        if self._scatter_item is None:
            # ``pxMode`` keeps the slider-controlled marker size in screen
            # pixels, mirroring the behaviour of the original matplotlib view.
            self._scatter_item = gl.GLScatterPlotItem(pxMode=True)
            self.view.addItem(self._scatter_item)
        self._scatter_item.setData(pos=coords_mm, color=colors, size=float(self.point_slider.value()))
        self._scatter_item.setGLOptions("translucent" if alpha < 0.999 else "opaque")

        mins = coords_mm.min(axis=0)
        maxs = coords_mm.max(axis=0)
        self._update_scene_bounds(mins, maxs)
        self._update_colorbar(cmap)

        total_voxels = self._estimate_voxels_above_threshold(thr)
        spin = getattr(self, "downsample_spin", None)
        downsample_source = "manual" if spin and spin.value() > 0 else "auto"
        lighting_suffix = ""
        if self._light_shader is not None or self._flat_shader is not None:
            lighting_suffix = " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        self.status_label.setText(
            "Point cloud: "
            f"{coords_mm.shape[0]:,} voxels (threshold {thr:.2f}, opacity {alpha:.2f}, "
            f"point size {self.point_slider.value()}). "
            f"Downsample step {step} ({downsample_source}); "
            f"≈ total voxels ≥ threshold {total_voxels:,}.{lighting_suffix}"
            f"{self._slice_status_suffix()}"
        )

    def _draw_surface_mesh(self, thr: float, cmap_name: str, alpha: float) -> None:
        if not HAS_SKIMAGE:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                "Surface rendering requires the optional 'scikit-image' dependency."
            )
            return

        volume = self._normalised_volume
        if volume is None:
            self.status_label.setText("No scalar volume available for rendering.")
            self._clear_colorbar()
            return

        if self._scatter_item is not None:
            self.view.removeItem(self._scatter_item)
            self._scatter_item = None

        step = self._downsample_step
        slices = tuple(slice(None, None, step) for _ in range(3))
        reduced = volume[slices]

        if min(reduced.shape) < 2:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                "Volume is too small after downsampling to compute a surface mesh."
            )
            return

        vol_min = float(np.min(reduced))
        vol_max = float(np.max(reduced))
        if not (vol_min < thr < vol_max):
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                f"Iso level {thr:.2f} is outside the volume range [{vol_min:.2f}, {vol_max:.2f}]."
            )
            return

        try:
            verts, faces, _normals, values = sk_measure.marching_cubes(
                reduced,
                level=thr,
                step_size=max(1, self._surface_step),
                spacing=(
                    step * self._voxel_sizes[0],
                    step * self._voxel_sizes[1],
                    step * self._voxel_sizes[2],
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(f"Marching cubes failed: {exc}")
            return

        vertex_values = np.clip(values, 0.0, 1.0)
        verts, faces, vertex_values, slice_bounds = self._clip_mesh_to_slices(
            verts, faces, vertex_values
        )
        if verts.size == 0 or faces.size == 0:
            if slice_bounds is not None:
                if self._mesh_item is not None:
                    self.view.removeItem(self._mesh_item)
                    self._mesh_item = None
                self._clear_colorbar()
                mins, maxs = slice_bounds
                self._update_scene_bounds(mins, maxs)
                self.status_label.setText(
                    f"Slice planes removed all surface triangles at iso level {thr:.2f}. "
                    "Adjust the slicer sliders or disable slicing."
                )
                return
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                f"No closed surface found at iso level {thr:.2f}; adjust the slider."
            )
            return

        if vertex_values is None:
            vertex_values = np.zeros(verts.shape[0], dtype=np.float32)

        meshdata = gl.MeshData(vertexes=verts, faces=faces)
        cmap = self._get_adjusted_colormap(cmap_name)
        vertex_colors = self._map_colors(vertex_values, cmap, alpha)
        meshdata.setVertexColors(vertex_colors)

        if self._mesh_item is None:
            # ``GLMeshItem`` retains the uploaded vertex buffers so subsequent
            # slider tweaks only update colours instead of reallocating the mesh.
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

        mins = np.min(verts, axis=0)
        maxs = np.max(verts, axis=0)
        self._update_scene_bounds(mins, maxs)
        self._update_colorbar(cmap, vmin=thr, vmax=1.0, label="Iso level (normalised)")

        total_voxels = int(np.count_nonzero(self._normalised_volume >= thr))
        spin = getattr(self, "downsample_spin", None)
        downsample_source = "manual" if spin and spin.value() > 0 else "auto"
        lighting_suffix = ""
        if self._light_shader is not None or self._flat_shader is not None:
            lighting_suffix = " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        self.status_label.setText(
            "Surface mesh: "
            f"{verts.shape[0]:,} vertices / {faces.shape[0]:,} faces (iso {thr:.2f}, "
            f"opacity {alpha:.2f}). Downsample step {step} ({downsample_source}); "
            f"marching step {self._surface_step}. Total voxels ≥ level {total_voxels:,}.{lighting_suffix}"
            f"{self._slice_status_suffix()}"
        )

    def _on_render_mode_change(self, _mode: str) -> None:
        if self._initialising:
            return
        self._update_mode_dependent_controls()
        self._update_plot()

    def _update_mode_dependent_controls(self) -> None:
        mode = self.render_mode_combo.currentText()
        is_surface = mode == "Surface mesh"
        self.point_label.setVisible(not is_surface)
        self.point_slider.setVisible(not is_surface)
        self.point_slider.setEnabled(not is_surface)
        self.surface_step_label.setEnabled(is_surface)
        self.surface_step_spin.setEnabled(is_surface)
        if hasattr(self, "lighting_group"):
            self.lighting_group.setEnabled(
                is_surface
                and (self._light_shader is not None or self._flat_shader is not None)
            )
            self._update_light_controls_enabled()
        if is_surface:
            self.thresh_text.setText("Iso level:")
            self.thresh_slider.setToolTip(
                "Iso-surface level within the normalised volume (0–1)."
            )
        else:
            self.thresh_text.setText("Threshold:")
            self.thresh_slider.setToolTip(
                "Minimum normalised intensity included in the rendering."
            )

    def _on_opacity_change(self, value: int) -> None:
        self.opacity_label.setText(f"{value / 100.0:.2f}")
        if not self._initialising:
            self._update_plot()

    def _on_max_points_change(self, value: int) -> None:
        self._max_points = int(value)
        self._prepare_downsampled()
        if not self._initialising:
            self._update_plot()

    def _on_downsample_change(self, _value: int) -> None:
        self._prepare_downsampled()
        if not self._initialising:
            self._update_plot()

    def _on_surface_step_change(self, value: int) -> None:
        self._surface_step = max(1, int(value))
        if not self._initialising and self.render_mode_combo.currentText() == "Surface mesh":
            self._update_plot()
