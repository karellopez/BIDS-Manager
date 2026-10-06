"""Volumes (MRI, PET, any NIfTI) in the viewer shell.

The presenter owns what is specific to volumes: the view pages (one plane,
three planes, 3-D, three planes with 3-D, hero, mosaic), the time-course
graph, the toolbar's sliders and contrast controls, loading, and the readout.
The shell (:class:`~bidsmgr.gui.viz.viewer.Viewer`) owns the frame around it:
header, toolbar host, loading page, footer, keys, links.

Loading is two jobs. ``open`` reads the header and the BIDS context; then
``stream`` reads the voxels front to back. The views appear as soon as the
FIRST frame is in, so a 1224-volume run shows within a tenth of a second and
the rest arrives while the user is already looking.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout,
    QLabel, QMenu, QPushButton, QScrollArea, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget,
)

from ....viz import layouts, views
from ....viz.bids import BidsContext, bids_context
from ....viz.compute import colormaps
from ....viz.data.volume import VolumeSource, memory_budget_bytes, open_volume
from ....viz.scene import (
    Cursor, GraphState, Scene, SourceRef, VolumeDisplay, VolumeLayer,
)
from ..context import ViewerContext
from ..menus import popup_menu, submenu

log = logging.getLogger(__name__)

#: How often streaming progress may repaint (seconds).
_PROGRESS_INTERVAL = 0.1

#: ONE row: what you switch while looking (views, slice, volume) and the
#: menus. Everything you set (the look, the view conventions, the layout, the
#: 3-D) is in the controls column, by purpose.
TOOLBAR_ROWS = (
    ("view.sagittal", "view.coronal", "view.axial", "|", "view.multi",
     "view.3d", "view.combo", "view.hero", "view.graph", "|", "widget:slice",
     "widget:frame", "frame.play", "stretch", "widget:tools", "widget:views",
     "view.inspector", "help.shortcuts"),
)


#: The base layer's look a scene restores.
_SCENE_DISPLAY_KEYS = {"colormap", "colormap_negative", "window", "window_negative", "gamma",
                       "invert", "opacity", "threshold_mode", "outline_px", "interpolation"}


def open_overlay_with_context(path: Path, root: Optional[Path], *, budget_bytes=None,
                              cancel=None):
    """Worker-side: an overlay read whole, and, when it is a series, its own
    run context (repetition time, events, physio) for the graph."""
    from ....viz.overlays import open_overlay

    overlay = open_overlay(path, budget_bytes=budget_bytes, cancel=cancel)
    src = overlay.source
    ctx = None
    if src.is_4d and not src.is_rgb and not src.in_memory:
        try:
            ctx = bids_context(Path(path), root)
        except Exception:  # noqa: BLE001 - the graph works without it
            ctx = None
    return overlay, ctx


def open_with_context(path: Path, root: Optional[Path]) -> tuple[VolumeSource, BidsContext]:
    """Worker-side: the header and the dataset context, nothing heavier."""
    src = open_volume(path)
    ctx = bids_context(path, root)
    return src, ctx


def _caption(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sidecar-footer-summary")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lbl


class VolumePresenter:
    """Volume content for a :class:`Viewer`."""

    kind = "volume"
    title = "NIfTI"
    #: Which mouse tables the help shows.
    mouse_canvases = ("slice", "render")
    empty_hint = "Select a NIfTI (.nii / .nii.gz) file in the BIDS tree to view it."

    def __init__(self, viewer, ctx: ViewerContext) -> None:
        self.viewer = viewer
        self.ctx = ctx
        self.source: Optional[VolumeSource] = None
        self.bids: Optional[BidsContext] = None
        self._generation = 0
        self._last_progress = 0.0
        self._first_frame_shown = False
        #: The window the viewer chose by itself (None once the user picks).
        self._auto_window: Optional[tuple[float, float]] = None
        #: The kind of file on screen (a layout preset id); survives a clear,
        #: so looking at a sidecar and back keeps the arrangement.
        self._preset_id = ""
        self._pending_sizes: dict[str, list[int]] = {}
        self._sizes_timer = QTimer(viewer)
        self._sizes_timer.setSingleShot(True)
        self._sizes_timer.setInterval(400)
        self._sizes_timer.timeout.connect(self._flush_sizes)
        self._mosaic_page = None
        self._render = None
        self._graph = None
        #: Each series' run context (repetition time, events, physio), by
        #: source id: the base's, and every 4-D overlay's.
        self._contexts: dict[str, Any] = {}
        self._graph_context_source: Optional[str] = None
        self._play_timer = QTimer(viewer)
        self._play_timer.timeout.connect(self._play_tick)
        self._remembering = True
        # Overlays read on their own jobs, one tag each, so several can load
        # at once and all are cancelled when the base image changes.
        self._overlay_seq = 0
        self._overlay_tags: list[str] = []
        #: A saved scene's look for each overlay job (display, name, ...).
        self._overlay_looks: dict[str, dict] = {}
        #: Overlays asked for while the base image was still opening.
        self._pending_overlays: list[tuple[Path, dict]] = []
        #: A saved scene being opened: applied once its image is open.
        self._pending_scene: Optional[dict] = None
        #: Its quality maps, computed once the series is read.
        self._pending_qc: list[dict] = []
        #: The crosshair to open the next file at (set by ``carry_cursor``).
        self._carried_cursor = None
        #: The header inspector, asked for while the image was opening (the
        #: rule to select, "" for none).
        self._pending_header: Optional[str] = None
        self.header_dialog = None
        self.content = self._build_content()
        ctx.jobs.done.connect(self._on_job_done)
        ctx.jobs.failed.connect(self._on_job_failed)
        ctx.jobs.progressed.connect(self._on_job_progress)
        ctx.qstore.changed.connect(self._on_changed)
        # Build the 3-D canvas NOW when a GPU exists, before the window is
        # shown: adding the first QOpenGLWidget to a visible window makes Qt
        # recreate the native window on Windows and Linux.
        if ctx.gpu_ok:
            try:
                self._ensure_render()
            except Exception as exc:  # noqa: BLE001 - never block the 2-D viewer
                log.warning("Could not pre-create the 3-D view: %s", exc)
                ctx.gpu_ok = False

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def _build_content(self) -> QWidget:
        self.vsplit = QSplitter(Qt.Orientation.Vertical)
        self.vsplit.setHandleWidth(2)
        self.vsplit.setChildrenCollapsible(False)
        self.pages = QStackedWidget()
        self.pages.setObjectName("nifti-canvas")
        from .tiles import TiledPage

        self.tiles = TiledPage(self)
        self.pages.addWidget(self.tiles)
        self.vsplit.addWidget(self.pages)
        self.graph_host = QWidget()
        lay = QVBoxLayout(self.graph_host)
        lay.setContentsMargins(0, 0, 0, 0)
        self.graph_host.setVisible(False)
        self.vsplit.addWidget(self.graph_host)
        self.vsplit.setStretchFactor(0, 3)
        self.vsplit.setStretchFactor(1, 1)
        self.vsplit.splitterMoved.connect(
            lambda *_a: self.remember_sizes("volume.graph", self.vsplit))
        # The controls column beside images and graph. Built empty; the
        # inspector is created the first time it is opened.
        self.hsplit = QSplitter(Qt.Orientation.Horizontal)
        self.hsplit.setHandleWidth(2)
        self.hsplit.setChildrenCollapsible(False)
        self.hsplit.addWidget(self.vsplit)
        self.side = QScrollArea()
        self.side.setObjectName("viz-side-scroll")
        self.side.viewport().setObjectName("viz-side-viewport")
        self.side.setWidgetResizable(True)
        self.side.setFrameShape(QFrame.Shape.NoFrame)
        self.side.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.side.setMinimumWidth(300)
        self.side.setVisible(False)
        self.hsplit.addWidget(self.side)
        self.hsplit.setStretchFactor(0, 1)
        self.hsplit.setStretchFactor(1, 0)
        self.hsplit.splitterMoved.connect(
            lambda *_a: self.remember_sizes("volume.side", self.hsplit))
        self._inspector = None
        return self.hsplit

    def _page(self, mode: str) -> QWidget:
        """The page a mode shows: the tiled page, or the mosaic."""
        if mode != "mosaic":
            return self.tiles
        if self._mosaic_page is None:
            from ..canvases.mosaic import MosaicPage

            self._mosaic_page = MosaicPage(self.ctx)
            self.pages.addWidget(self._mosaic_page)
        return self._mosaic_page

    def _ensure_render(self):
        if self._render is not None:
            return self._render
        from ..canvases.render import RenderCanvas

        self._render = RenderCanvas(self.ctx)
        self.tiles.set_render(self._render)
        return self._render

    def _ensure_graph(self):
        if self._graph is None:
            from ..canvases.graph import TimecourseGraph

            self._graph = TimecourseGraph(self.ctx)
            self._graph.wants_height.connect(self._grow_graph)
            self.graph_host.layout().addWidget(self._graph)
            self._graph_context_source = None
            self._sync_graph_context()
        return self._graph

    def _sync_graph_context(self) -> None:
        """Give the graph the run context of the series it plots (the base's,
        or an overlaid BOLD's own), once per change of series."""
        if self._graph is None:
            return
        layer, _src = views.series_layer(self.ctx.store)
        source = layer.source if layer is not None else None
        if source == self._graph_context_source:
            return
        self._graph_context_source = source
        self._graph.set_context(self._contexts.get(source) if source else None)

    def _grow_graph(self, pixels: int) -> None:
        """Give the graph pane ``pixels`` if it has less, never more than
        three fifths of the viewer (the images stay the point)."""
        sizes = self.vsplit.sizes()
        if len(sizes) < 2 or not self.graph_host.isVisible():
            return
        total = sum(sizes)
        want = min(int(pixels), int(total * 0.6))
        if sizes[-1] >= want:
            return
        self.vsplit.setSizes([total - want] + [0] * (len(sizes) - 2) + [want])

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def three_d_capable(self) -> bool:
        src = self.source
        if src is None:
            return False
        if any(d < 2 for d in src.spatial):
            return False
        return not src.is_rgb or src.channels in (3, 4)

    def render_allowed(self) -> bool:
        """Whether the 3-D view can show this image here."""
        return bool(self._render is not None and self.ctx.gpu_ok and self.three_d_capable())

    def allowed_mode(self, mode: str) -> str:
        if mode in ("3d", "combo") and not (self.ctx.gpu_ok and self.three_d_capable()):
            return "multi"
        return mode

    def apply_mode(self) -> None:
        scene = self.ctx.scene
        mode = self.allowed_mode(scene.mode)
        if mode != scene.mode:
            # A state asking for 3-D arrives on machines without a GPU as a
            # matter of course (a scene saved on a workstation): fall back to
            # the three planes, never to a single slice.
            scene.mode = mode
        if mode in ("3d", "combo"):
            try:
                self._ensure_render()
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not open the 3-D view: %s", exc)
                scene.mode = mode = "multi"
        page = self._page(mode)
        if page is self.tiles:
            self.tiles.apply()
        self.pages.setCurrentWidget(page)
        # The graph below or beside the views.
        want = (Qt.Orientation.Horizontal if scene.layout.graph == "right"
                else Qt.Orientation.Vertical)
        if self.vsplit.orientation() != want:
            self.vsplit.setOrientation(want)
        if self._render is not None and self._render.isVisible():
            self._render.refresh_volume()

    def apply_graph(self) -> None:
        series, _src = views.series_layer(self.ctx.store)
        show = bool(self.ctx.scene.graph_visible and series is not None
                    and self.ctx.scene.mode != "3d")
        if show:
            self._ensure_graph()
            if not self.graph_host.isVisible():
                self.graph_host.setVisible(True)
                self.restore_sizes("volume.graph", self.vsplit, [0.7, 0.3])
        else:
            self.graph_host.setVisible(False)

    # ------------------------------------------------------------------
    # Toolbar widgets
    # ------------------------------------------------------------------

    def toolbar_rows(self):
        return TOOLBAR_ROWS

    def make_widget(self, name: str):
        """A toolbar item by name: one widget, a list of widgets (each its
        own wrap point), or None."""
        return {
            "tools": self._widget_tools,
            "slice": self._widget_slice, "frame": self._widget_frame,
            "views": self._widget_views,
        }.get(name, lambda: None)()

    def _number_group(self, title: str, tip: str):
        """A titled slider-and-number (slice, volume): typable, and wheel
        only when focused."""
        from ..controls import NumberControl

        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        hdr = QLabel(title)
        hdr.setObjectName("sidecar-footer-summary")
        h.addWidget(hdr)
        nc = NumberControl(0, 1, step=1, decimals=0)
        nc.slider.setMinimumWidth(90)
        nc.spin.setFixedWidth(96)
        nc.setToolTip(tip)
        h.addWidget(nc)
        return box, hdr, nc

    def _widget_slice(self) -> QWidget:
        box, self._slice_hdr, self.slice_control = self._number_group(
            "Slice", "The slice shown; type a number, or drag")
        self.slice_control.value_changed.connect(self._on_slice_control)
        return box

    def _widget_frame(self) -> QWidget:
        box, self._frame_hdr, self.frame_control = self._number_group(
            "Volume", "The volume of the 4-D series shown; type a number, or drag")
        self.frame_control.value_changed.connect(
            lambda v: self._run_ui("frame.set", frame=int(v)))
        self.frame_control.pressed.connect(self.ctx.store.begin_gesture)
        self.frame_control.released.connect(self.ctx.store.end_gesture)
        return box

    #: The Tools menu, in order ("" is a separator). Each entry is an action,
    #: so it also has a key, a palette entry and a help line.
    TOOLS = ("header.show", "", "overlay.add", "", "qc.mean", "qc.sd", "qc.tsnr", "",
             "frame.same_shell", "frame.shell", "frame.shell_prev", "", "deface.preview")

    def _widget_tools(self) -> QWidget:
        """What can be done with the open image beyond looking at it."""
        btn = QPushButton("Tools")
        btn.setObjectName("tb-btn-toggle")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setToolTip("Overlays, quality maps and the dataset around this image")
        menu = popup_menu(btn)
        self._fill_tools_menu(menu)
        # Rebuilt as it opens: the fieldmap entries depend on the image.
        menu.aboutToShow.connect(lambda m=menu: self._fill_tools_menu(m))
        btn.setMenu(menu)
        self.tools_button = btn
        return btn

    def _fill_tools_menu(self, menu: QMenu) -> None:
        menu.clear()
        for action_id in self.TOOLS:
            if action_id:
                menu.addAction(self.viewer.action(action_id))
            else:
                menu.addSeparator()
        menu.addSeparator()
        go = submenu(menu, "Go to")
        for action_id in self.NAVIGATION:
            go.addAction(self.viewer.action(action_id))
        bids = self.bids
        if bids is not None and bids.fieldmaps:
            maps = submenu(menu, "Fieldmaps for this image")
            for fmap in bids.fieldmaps:
                act = maps.addAction(f"Draw {fmap.name} over this")
                act.triggered.connect(lambda _c=False, f=fmap: self.add_overlay(f))
        if bids is not None and bids.datatype == "fmap" and bids.intended_for:
            meant = submenu(menu, "Images this fieldmap is for")
            for target in bids.intended_for:
                act = meant.addAction(f"Open {target.name} with this over it")
                act.setEnabled(target.is_file())
                act.triggered.connect(lambda _c=False, t=target: self.open_under(t))

    #: The "Go to" submenu.
    NAVIGATION = ("nav.run_next", "nav.run_prev", "nav.echo_next", "nav.echo_prev",
                  "nav.ses_next", "nav.ses_prev", "nav.sub_next", "nav.sub_prev")

    def open_under(self, target: Path) -> None:
        """Open ``target`` and draw the image on screen over it (a fieldmap
        over the run it corrects)."""
        if self.source is None:
            return
        overlay = self.source.path
        self.viewer.navigate(Path(target))
        self.viewer.add_overlay(overlay)

    def navigate(self, entity: str, step: int) -> bool:
        """The previous (``step`` < 0) or next file along ``entity``."""
        found = (self.bids.neighbours.get(entity) if self.bids is not None else None)
        if not found:
            return False
        target = found[0] if step < 0 else found[1]
        if target is None:
            return False
        self.viewer.navigate(target)
        return True

    def carry_cursor(self) -> None:
        """Keep the crosshair's world position for the next file opened."""
        world = self.ctx.scene.cursor.world
        self._carried_cursor = tuple(world) if world is not None else None

    def _widget_views(self) -> QWidget:
        """Saved views: apply one, save the current one, delete one."""
        btn = QPushButton("Views")
        btn.setObjectName("tb-btn-toggle")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setToolTip("Views you saved: a layout, plane, display options, graph "
                       "and 3-D look, applied to whatever image is open.")
        menu = popup_menu(btn)
        menu.aboutToShow.connect(lambda m=menu: self._fill_views_menu(m))
        btn.setMenu(menu)
        self.views_button = btn
        return btn

    def _fill_views_menu(self, menu: QMenu) -> None:
        from ....viz import scenes

        menu.clear()
        menu.addAction(self.viewer.action("views.save"))
        menu.addAction(self.viewer.action("scene.save"))
        menu.addAction(self.viewer.action("view.command_line"))
        found = scenes.list_scenes(self.scene_root())
        if found:
            sub = submenu(menu, "Scenes of this dataset")
            for name, path in found:
                act = sub.addAction(name)
                act.triggered.connect(lambda _c=False, p=path: self.open_scene(p))
        names = sorted(self.ctx.settings.view_presets, key=str.lower)
        if names:
            menu.addSeparator()
            for name in names:
                act = menu.addAction(name)
                act.triggered.connect(lambda _c=False, n=name: self.apply_view(n))
            delete = submenu(menu, "Delete a saved view")
            for name in names:
                act = delete.addAction(name)
                act.triggered.connect(lambda _c=False, n=name: self.delete_view(n))

    # -- saved views (also the scripting surface) ------------------------

    #: What a saved view leaves out: where you were (cursor, zoom and pan
    #: belong to one image), and what is not a view at all.
    _VIEW_EXCLUDE = {"sources", "layers", "cursor", "views", "measurements",
                     "schema_version"}

    def save_view(self, name: str) -> None:
        name = name.strip()
        if not name:
            return
        state = self.ctx.scene.model_dump(mode="json", exclude=self._VIEW_EXCLUDE)
        self.ctx.settings_hub.update(lambda s: s.view_presets.__setitem__(name, state))
        self.viewer.status_message.emit(f"Saved the view {name!r}")

    def apply_view(self, name: str) -> bool:
        state = self.ctx.settings.view_presets.get(name)
        if state is None:
            return False
        self.viewer.apply_state(state)
        self.ctx.qstore.flush()
        return True

    def delete_view(self, name: str) -> None:
        if name in self.ctx.settings.view_presets:
            self.ctx.settings_hub.update(lambda s: s.view_presets.pop(name, None))

    # -- scenes: saved WITH the dataset ------------------------------------

    def scene_root(self) -> Optional[Path]:
        from ....viz.bids import dataset_root

        root = self.viewer.current_root()
        if root is None and self.source is not None:
            root = dataset_root(self.source.path)
        return root

    def save_scene(self, name: str) -> Optional[Path]:
        """Save what is on screen as a scene of the dataset."""
        from ....viz import scenes

        name = name.strip()
        root = self.scene_root()
        if not name or root is None or self.source is None:
            return None
        data = scenes.snapshot(self.ctx.scene, self.ctx.store.sources, root, name)
        path = scenes.save(root, data)
        self.viewer.status_message.emit(
            f"Saved the scene {name!r} in the dataset ({scenes.SCENES.as_posix()})")
        return path

    def open_scene(self, path: Path) -> bool:
        """Open a saved scene: its image, its overlays in their looks, its
        crosshair and layout."""
        from ....viz import scenes

        try:
            data = scenes.load(Path(path))
        except (OSError, ValueError) as exc:
            self.viewer.status_message.emit(f"Could not open the scene: {exc}")
            return False
        root = self.scene_root() or Path(path).parents[3]
        base = scenes.from_rel(root, data["base"])
        if not base.is_file():
            self.viewer.status_message.emit(f"The scene's image is gone: {data['base']}")
            return False
        data["_root"] = str(root)
        self.viewer.navigate(base, root)
        # Set AFTER navigating: opening the image clears what was pending.
        self._pending_scene = data
        return True

    def _apply_scene(self, data: dict) -> None:
        from ....viz import scenes

        root = Path(data.get("_root") or ".")
        view = dict(data.get("view") or {})
        cursor = (view.pop("cursor", None) or {}).get("world")
        self.viewer.apply_state(view)
        base = data.get("base_display")
        if base:
            self.ctx.run("layer.set", **{k: v for k, v in base.items()
                                         if k in _SCENE_DISPLAY_KEYS and v is not None})
        if data.get("frame"):
            self.ctx.run("frame.set", frame=int(data["frame"]))
        if cursor is not None:
            self.ctx.run("cursor.set_world", x=float(cursor[0]), y=float(cursor[1]),
                         z=float(cursor[2]), snap=False)
        for item in data.get("overlays", []):
            look = {"display": item.get("display"), "name": item.get("name", ""),
                    "visible": bool(item.get("visible", True)),
                    "in_3d": bool(item.get("in_3d", True))}
            origin = str(item.get("origin") or "")
            if origin.startswith("qc:"):
                self._pending_qc.append({"which": origin[3:], "look": look})
            elif origin.startswith("deface:"):
                self.preview_defacing(origin[len("deface:"):], **look)
            else:
                self.add_overlay(scenes.from_rel(root, item["path"]), **look)
        self.ctx.qstore.flush()
        self.viewer.status_message.emit(f"Opened the scene {data.get('name', '')!r}")

    def _ask_scene_name(self) -> None:
        from PyQt6.QtWidgets import QInputDialog, QMessageBox

        from ....viz import scenes

        root = self.scene_root()
        if root is None or self.source is None:
            self.viewer.status_message.emit(
                "A scene is saved in a dataset: open an image from one first.")
            return
        name, ok = QInputDialog.getText(self.viewer, "Save as a scene of this dataset",
                                        "Name for this scene:")
        if not ok or not name.strip():
            return
        if scenes.scene_path(root, name.strip()).is_file():
            answer = QMessageBox.question(
                self.viewer, "Replace the scene",
                f"This dataset has a scene called {name.strip()!r}. Replace it?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.save_scene(name)

    def _ask_view_name(self) -> None:
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self.viewer, "Save this view",
                                        "Name for this view:")
        if ok and name.strip():
            if name.strip() in self.ctx.settings.view_presets:
                from PyQt6.QtWidgets import QMessageBox

                answer = QMessageBox.question(
                    self.viewer, "Replace the saved view",
                    f"A view called {name.strip()!r} exists. Replace it?")
                if answer != QMessageBox.StandardButton.Yes:
                    return
            self.save_view(name)

    _syncing_widgets = False

    def _run_ui(self, command_id: str, **params) -> None:
        if not self._syncing_widgets:
            self.ctx.run(command_id, **params)

    def _active_plane(self) -> str:
        scene = self.ctx.scene
        return scene.plane if scene.mode == "single" else self.ctx.active_plane

    def _on_slice_control(self, value: float) -> None:
        if self._syncing_widgets:
            return
        self.ctx.run("cursor.set_slice", plane=self._active_plane(), index=int(value))

    def sync_widgets(self) -> None:
        """Read the scene into the toolbar widgets (no command runs)."""
        store = self.ctx.store
        layer, src = views.base(store)
        self._syncing_widgets = True
        try:
            if hasattr(self, "slice_control"):
                plane = self._active_plane()
                n = views.slice_count(store, plane)
                self._slice_hdr.setText(f"Slice ({plane})" if self.ctx.scene.mode != "single"
                                        else "Slice")
                self.slice_control.set_range(0, max(n - 1, 0))
                self.slice_control.spin.setSuffix(f" / {max(n - 1, 0)}")
                self.slice_control.setEnabled(n > 1)
                self.slice_control.set_value(views.slice_index(store, plane))
            if hasattr(self, "frame_control"):
                s_layer, s_src = views.series_layer(store)
                if s_layer is not None:
                    layer, src = s_layer, s_src
                n = src.n_frames if src is not None else 1
                t = views.frame_of(store, layer, src) if src is not None and layer else 0
                self.frame_control.set_range(0, max(n - 1, 0))
                self.frame_control.spin.setSuffix(f" / {max(n - 1, 0)}")
                self.frame_control.setEnabled(n > 1)
                self.frame_control.set_value(t)
                title = "Volume"
                bvals = getattr(src, "bvals", None) if src is not None else None
                if bvals is not None and t < len(bvals):
                    title += f"  b={float(bvals[t]):g}"
                self._frame_hdr.setText(title)
        finally:
            self._syncing_widgets = False

    # ------------------------------------------------------------------
    # Context for actions
    # ------------------------------------------------------------------

    def action_context(self) -> dict[str, Any]:
        scene = self.ctx.scene
        src = self.source
        layer = scene.base_layer()
        has = src is not None and layer is not None
        ci = self.ctx.active_clip
        clip = scene.clips[ci] if ci < len(scene.clips) else None
        return {
            "volume": has,
            "volume.4d": views.series_layer(self.ctx.store)[0] is not None,
            "volume.3d": bool(has and self.three_d_capable()),
            "gpu": bool(self.ctx.gpu_ok),
            "slices": bool(has and scene.mode != "3d"),
            "render": bool(has and self.ctx.gpu_ok and scene.mode in ("3d", "combo", "hero")),
            "mode": scene.mode,
            "plane": scene.plane,
            "display.labels": scene.display.labels,
            "display.ras": scene.display.ras,
            "display.radiological": scene.display.radiological,
            "display.colorbar": scene.display.colorbar,
            "display.crosshair": scene.display.crosshair,
            "space": scene.display.space,
            "graph": scene.graph_visible,
            "clip.active": bool(clip and clip.active),
            "clip.at_cursor": bool(scene.render.cut_at_cursor),
            "layer.invert": bool(layer.display.invert) if layer else False,
            "layer.nearest": bool(layer and layer.display.interpolation == "nearest"),
            "playing": self.ctx.playing,
            "panel.inspector": self.inspector_open(),
            "volume.loaded": bool(has and src.fully_loaded),
            "volume.dwi": bool(has and src.bvals is not None),
            "volume.deface": bool(has and not src.is_4d and not src.is_rgb
                                  and self._deface_available()),
            **self._navigation_context(),
        }

    def _navigation_context(self) -> dict[str, bool]:
        out = {}
        found = self.bids.neighbours if self.bids is not None else {}
        for entity in ("run", "echo", "ses", "sub"):
            prev, nxt = found.get(entity, (None, None))
            out[f"nav.{entity}.prev"] = prev is not None
            out[f"nav.{entity}.next"] = nxt is not None
        return out

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: Path, root: Optional[Path]) -> None:
        self._generation += 1
        self._pending_overlays = []
        self._pending_header = None
        self._pending_scene = None
        self._pending_qc = []
        self.stop_play()
        self._release()
        self._first_frame_shown = False
        self._auto_window = None
        self.ctx.jobs.start("open", self._generation, open_with_context, path, root)

    def _release(self) -> None:
        for tag in self._overlay_tags:
            self.ctx.jobs.cancel(tag)
        self._overlay_tags = []
        self._overlay_looks = {}
        self.ctx.jobs.cancel("stream")
        self.ctx.jobs.cancel("open")
        self.ctx.jobs.cancel("render-volume")
        self.ctx.jobs.cancel("series-window")
        if self.source is not None:
            self.source.release()
        self.source = None
        self.bids = None
        if self._render is not None:
            self._render.clear()

    def clear(self) -> None:
        self._generation += 1
        self._pending_overlays = []
        self._pending_header = None
        self._pending_scene = None
        self._pending_qc = []
        self.stop_play()
        self._release()
        store = self.ctx.store
        store.sources.clear()
        keep = self._persistent_scene()
        store.replace_scene(keep)
        self.ctx.qstore.flush()

    def _persistent_scene(self) -> Scene:
        """A fresh scene that keeps the user's view preferences."""
        old = self.ctx.scene
        s = Scene()
        s.mode = old.mode
        s.plane = old.plane
        s.display = old.display.model_copy(deep=True)
        s.graph_visible = old.graph_visible
        s.graph = old.graph.model_copy(deep=True)
        s.render = old.render.model_copy(deep=True)
        s.clips = [c.model_copy() for c in old.clips]
        return s

    def initial_scene(self, src: VolumeSource, path: Path) -> Scene:
        """The scene a newly opened file starts in.

        The ARRANGEMENT (mode, plane, graph) follows the kind of file: a
        file of the kind already on screen keeps the current view (you are
        driving); a different kind opens in the arrangement that kind last
        had, else its preset (a BOLD run with its graph, a T1 without). The
        LOOK (gamma, colour map) is the user's default for new images, and
        the display conventions (RAS, radiological, labels) are kept from
        whatever was on screen, or read from the settings for the first
        file a viewer opens.
        """
        settings = self.ctx.settings
        vs = settings.volume
        prefs = self._persistent_scene()
        sid = "vol0"
        prefs.sources = {sid: SourceRef(id=sid, path=str(path), kind="volume")}
        display = VolumeDisplay(gamma=vs.gamma, interpolation=vs.interpolation,
                                colormap=vs.colormap if colormaps.exists(vs.colormap) else "gray")
        prefs.layers = [VolumeLayer(id="base", source=sid, name=path.name, display=display)]
        prefs.cursor = Cursor(world=tuple(float(v) for v in src.center_world()))
        carried, self._carried_cursor = self._carried_cursor, None
        if carried is not None and views.voxel_in(src, np.asarray(carried, dtype=float)):
            # Moved here from another run or subject: the same spot.
            prefs.cursor = Cursor(world=tuple(float(v) for v in carried))
        if self.viewer.first_open:
            prefs.display.labels = vs.labels
            prefs.display.colorbar = vs.colorbar
            prefs.display.ras = vs.ras
            prefs.display.radiological = vs.radiological
            prefs.display.space = vs.space
            prefs.plane = vs.plane
        bids = self.bids
        preset = layouts.match(bids.datatype if bids else "", bids.suffix if bids else "",
                               bool(src.is_4d and not src.is_rgb))
        if self.viewer.first_open or preset.id != self._preset_id:
            default_mode = vs.mode or ("combo" if self.ctx.gpu_ok else "multi")
            view = layouts.opening_view(preset, settings.layout_state.get(preset.id),
                                        default_mode)
            prefs.mode = view["mode"]
            prefs.plane = view.get("plane", prefs.plane)
            prefs.graph_visible = bool(view.get("graph_visible", False))
            if view.get("graph"):
                prefs.graph = GraphState.model_validate(
                    {**prefs.graph.model_dump(), **view["graph"]})
        self._preset_id = preset.id
        return prefs

    @property
    def layout_id(self) -> str:
        """The kind of file on screen, as a layout preset id."""
        return self._preset_id

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if generation != self._generation:
            return
        if tag in self._overlay_tags:
            self._overlay_tags.remove(tag)
            look = self._overlay_looks.pop(tag, {})
            overlay, run_ctx = result if isinstance(result, tuple) else (result, None)
            n = tag.split(":", 1)[1]
            if run_ctx is not None:
                self._contexts[f"ovl{n}"] = run_ctx
            self._adopt_overlay(n, overlay, **look)
            return
        if tag == "open":
            src, bids = result
            self.source = src
            self.bids = bids
            self._contexts = {"vol0": bids}
            bvals = getattr(bids, "bvals", None)
            if bvals is not None and len(bvals) == src.n_frames_total and src.n_frames > 1:
                # Only when they describe THESE volumes: the inspector says
                # when they do not.
                src.bvals = bvals
            store = self.ctx.store
            store.sources = {"vol0": src}
            self._remembering = False
            try:
                scene = self.initial_scene(src, src.path)
                store.replace_scene(scene)
                self.viewer.first_open = False
                # Unconditionally re-validate the layout for this file.
                scene.mode = self.allowed_mode(scene.mode)
            finally:
                self._remembering = True
            self._graph_context_source = None
            self._sync_graph_context()
            budget = memory_budget_bytes(self.ctx.settings.volume.memory_mb)
            self.ctx.jobs.start("stream", self._generation, src.stream,
                                budget_bytes=budget)
            self.ctx.qstore.flush()
            pending, self._pending_overlays = self._pending_overlays, []
            for path, look in pending:
                self.add_overlay(path, **look)
            if self._pending_header is not None:
                rule, self._pending_header = self._pending_header, None
                self.show_header(rule)
            if self._pending_scene is not None:
                data, self._pending_scene = self._pending_scene, None
                self._apply_scene(data)
        elif tag == "stream":
            pending_qc, self._pending_qc = self._pending_qc, []
            self._auto_window_from_frame()
            self.ctx.store.changed({"sources:vol0"})
            self.ctx.qstore.flush()
            self.viewer.on_loaded(self.source.path if self.source else None)
            src = self.source
            if src is not None and src.is_4d and not src.is_rgb:
                # The whole series, sampled, on a worker: PET's first frames
                # hold almost no counts, so a window taken from frame 0 alone
                # saturates every later one.
                self.ctx.jobs.start("series-window", self._generation, src.series_range)
            for item in pending_qc:
                self.add_quality_map(item["which"], **item["look"])
        elif tag == "series-window":
            self._apply_series_window(result)

    # ------------------------------------------------------------------
    # The window a volume opens with
    # ------------------------------------------------------------------

    def _auto_window_from_frame(self) -> None:
        """Fill the window ONCE, from the frame on screen.

        ``None`` in the scene means "not chosen yet". Left as ``None``, every
        canvas computed a robust range per frame, so playback recomputed a
        percentile on every step and the brightness flickered from frame to
        frame. Written into the scene, the contrast stays put while frames
        change, and it is a value a linked viewer and a saved scene can copy.
        """
        store = self.ctx.store
        layer, src = views.base(store)
        if layer is None or src is None or src.is_rgb or layer.display.window is not None:
            return
        rng = src.robust_range(views.frame_of(store, layer, src))
        if rng is None:
            return
        layer.display.window = rng
        self._auto_window = rng
        store.changed({f"layer:{layer.id}.display"})

    def _apply_series_window(self, rng) -> None:
        """Widen the opening window to the series, unless the user already
        chose one (any change away from the automatic value counts)."""
        store = self.ctx.store
        layer, _src = views.base(store)
        if layer is None or rng is None or self._auto_window is None:
            return
        if layer.display.window != self._auto_window:
            return
        rng = (float(rng[0]), float(rng[1]))
        if rng == self._auto_window:
            return
        layer.display.window = rng
        self._auto_window = rng
        store.changed({f"layer:{layer.id}.display"})
        self.ctx.qstore.flush()

    def _on_job_failed(self, tag: str, generation: int, message: str) -> None:
        if generation == self._generation and tag in self._overlay_tags:
            self._overlay_tags.remove(tag)
            self._overlay_looks.pop(tag, None)
            self.viewer.loading_changed.emit(False, "")
            self.viewer.status_message.emit(f"Could not add the overlay: {message}")
            return
        if generation != self._generation or tag not in ("open", "stream"):
            return
        path = self.viewer.current_file()
        self._release()
        self.viewer.on_load_failed(path, message)

    def _on_job_progress(self, tag: str, generation: int, done: int, total: int) -> None:
        if tag != "stream" or generation != self._generation or self.source is None:
            return
        now = time.monotonic()
        first = not self._first_frame_shown and self.source.loaded_frames > 0
        if first:
            self._first_frame_shown = True
            self._auto_window_from_frame()
            self.viewer.on_first_frame()
        if first or done >= total or now - self._last_progress >= _PROGRESS_INTERVAL:
            self._last_progress = now
            self.ctx.store.changed({"sources:vol0"})

    # ------------------------------------------------------------------
    # Scene changes
    # ------------------------------------------------------------------

    def _on_changed(self, paths) -> None:
        if paths & {"mode", "scene", "layout", "plane"}:
            self.apply_mode()
        if self._remembering and self.source is not None and (
            paths & {"mode", "plane", "graph", "graph_visible"}
        ):
            self._remember_arrangement()
        if any(p.startswith("display.") for p in paths) and self._remembering:
            d = self.ctx.scene.display

            def keep(s):
                s.volume.labels = d.labels
                s.volume.colorbar = d.colorbar
                s.volume.ras = d.ras
                s.volume.radiological = d.radiological
                s.volume.space = d.space

            self.ctx.settings_hub.update(keep)
        if any(p in ("graph_visible", "mode", "scene", "layers", "graph") or p.startswith("sources")
               for p in paths):
            self.apply_graph()
            self._sync_graph_context()
        self.sync_widgets()
        self.viewer.update_footer()

    def _remember_arrangement(self) -> None:
        """Written as the user switches (the Editor is not always closed
        cleanly): the arrangement of THIS kind of file, and the mode and
        plane as the default for kinds never arranged."""
        scene = self.ctx.scene
        state = scene.model_dump(mode="json", include=set(layouts.LAYOUT_KEYS))
        kind = self._preset_id

        def keep(s):
            if kind:
                s.layout_state[kind] = layouts.arrangement(state)
            s.volume.mode = scene.mode
            s.volume.plane = scene.plane

        self.ctx.settings_hub.update(keep)

    # ------------------------------------------------------------------
    # Splitter sizes
    # ------------------------------------------------------------------

    def remember_sizes(self, key: str, splitter) -> None:
        """Keep a splitter's sizes (written after the drag settles)."""
        sizes = [int(v) for v in splitter.sizes()]
        if not sizes or sum(sizes) <= 0:
            return
        self._pending_sizes[key] = sizes
        self._sizes_timer.start()

    def restore_sizes(self, key: str, splitter, fallback: list[float]) -> None:
        """Apply remembered sizes, scaled to the splitter's size today;
        else the ``fallback`` fractions."""
        total = (splitter.height() if splitter.orientation() == Qt.Orientation.Vertical
                 else splitter.width()) or 600
        saved = self.ctx.settings.layout_sizes.get(key)
        fractions = ([v / sum(saved) for v in saved]
                     if saved and len(saved) == splitter.count() and sum(saved) > 0
                     else fallback)
        splitter.setSizes([max(1, int(total * f)) for f in fractions])

    def _flush_sizes(self) -> None:
        pending, self._pending_sizes = self._pending_sizes, {}
        if pending:
            self.ctx.settings_hub.update(lambda s: s.layout_sizes.update(pending))

    # ------------------------------------------------------------------
    # GUI-only actions
    # ------------------------------------------------------------------

    def gui_action(self, name: str, params) -> bool:
        if name == "slice":
            self.ctx.run("cursor.step", plane=self._active_plane(), n=int(params.get("n", 1)))
            return True
        if name == "play":
            self.toggle_play()
            return True
        if name == "save_view":
            self._ask_view_name()
            return True
        if name == "inspector":
            self.set_inspector(not self.inspector_open())
            return True
        if name == "add_overlay":
            self.choose_overlay()
            return True
        if name == "qc":
            self.add_quality_map(str(params.get("map", "tsnr")))
            return True
        if name == "header":
            self.show_header()
            return True
        if name == "deface_preview":
            self.preview_defacing()
            return True
        if name == "save_scene":
            self._ask_scene_name()
            return True
        if name == "command_line":
            self.show_command_line()
            return True
        if name == "navigate":
            self.navigate(str(params.get("entity", "run")), int(params.get("step", 1)))
            return True
        return False

    def show_command_line(self):
        """The view on screen as a command line (``viz.reproduce``)."""
        from ..panels.command_line import CommandLineDialog

        if self.source is None:
            return None
        dlg = CommandLineDialog(self.ctx.store, parent=self.viewer)
        self.command_line_dialog = dlg
        dlg.show()
        return dlg

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def add_overlay(self, path: Path, **look) -> bool:
        """Read ``path`` on a worker and draw it over the open image, in the
        look its contents call for (``viz.overlays``), or in ``look``
        (``display``, ``name``, ``visible``: what a saved scene recorded).
        False when there is no image to put it over."""
        if self.source is None:
            if self.viewer.current_file() is not None:
                # The base is still opening: the overlay follows it.
                self._pending_overlays.append((Path(path), look))
                return True
            self.viewer.status_message.emit("Open an image first: an overlay goes over one.")
            return False
        tag = self._overlay_job(look)
        self.viewer.loading_changed.emit(True, f"Reading {Path(path).name}")
        budget = memory_budget_bytes(self.ctx.settings.volume.memory_mb)
        self.ctx.jobs.start(tag, self._generation, open_overlay_with_context, Path(path),
                            self.viewer.current_root(), budget_bytes=budget)
        return True

    def _overlay_job(self, look: dict) -> str:
        self._overlay_seq += 1
        tag = f"overlay:{self._overlay_seq}"
        self._overlay_tags.append(tag)
        self._overlay_looks[tag] = dict(look)
        return tag

    def show_header(self, rule: str = ""):
        """The header beside the sidecar (one window per viewer, refilled for
        the image on screen); ``rule`` selects the row a validation finding
        is about. Returns the dialog, or None with no image open."""
        from ....viz.inspect import inspect
        from ..panels.header import HeaderDialog

        src = self.source
        if src is None:
            if self.viewer.current_file() is not None:
                # Still opening: shown once the header is read.
                self._pending_header = rule or ""
            return None
        rows = inspect(src, self.bids)
        old = getattr(self, "header_dialog", None)
        # Checked, not tracked through ``destroyed`` (CLAUDE.md guard 8d).
        if old is not None and not sip.isdeleted(old):
            old.close()
        dlg = HeaderDialog(src.path.name, rows, parent=self.viewer)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.header_dialog = dlg
        if rule:
            dlg.select_rule(rule)
        dlg.show()
        return dlg

    def add_quality_map(self, which: str, **look) -> bool:
        """Mean, standard deviation or temporal SNR of the open series,
        computed on a worker and drawn over it."""
        _layer, src = views.series_layer(self.ctx.store)
        if src is None or not src.fully_loaded:
            self.viewer.status_message.emit(
                "A quality map needs a 4-D series read whole: wait until it is read.")
            return False
        from ....viz.compute.qc import TITLES, quality_overlay

        tag = self._overlay_job({**look, "origin": f"qc:{which}"})
        self.viewer.loading_changed.emit(True, f"Computing the {TITLES[which].lower()}")
        self.ctx.jobs.start(tag, self._generation, quality_overlay, src, which)
        return True

    _deface_ok: Optional[bool] = None

    def _deface_available(self) -> bool:
        """Whether a defacing engine is installed (looked up once)."""
        if self._deface_ok is None:
            try:
                from ....deface.run import available

                type(self)._deface_ok = bool(available())
            except Exception:  # noqa: BLE001 - no engine is an answer, not an error
                type(self)._deface_ok = False
        return bool(self._deface_ok)

    def preview_defacing(self, engine_id: str = "", **look) -> bool:
        """What defacing would remove from the image on screen, drawn over
        it; nothing is changed. The engine runs on a worker."""
        src = self.source
        if src is None or src.is_4d or src.in_memory:
            return False
        from ....viz.overlays import deface_preview

        tag = self._overlay_job({**look, "origin": f"deface:{engine_id}"})
        self.viewer.loading_changed.emit(True, "Running the defacing engine for a preview")
        self.ctx.jobs.start(tag, self._generation, deface_preview, src.path, engine_id)
        return True

    def _adopt_overlay(self, n: str, overlay, *, name: str = "", display=None,
                       visible: bool = True, in_3d: bool = True, origin: str = "") -> None:
        sid, lid = f"ovl{n}", f"overlay{n}"
        src = overlay.source
        self.ctx.store.sources[sid] = src
        label = name or overlay.name or src.path.name
        look = display if display is not None else overlay.display.model_dump()
        self.ctx.run("layer.add", id=lid, source=sid, name=label, display=look,
                     path="" if src.in_memory else str(src.path), origin=origin,
                     visible=bool(visible), in_3d=bool(in_3d))
        self.ctx.qstore.flush()
        base = self.source
        if (src.is_4d and not src.is_rgb and base is not None and not base.is_4d
                and not self.ctx.scene.graph_visible):
            # A series over an anatomy: its time course is why it is here.
            self.ctx.run("view.graph", value=True)
        self.viewer.loading_changed.emit(False, "")
        self.viewer.status_message.emit(f"Overlay {label}: {overlay.note}")
        self.viewer.overlay_added.emit(lid)

    def choose_overlay(self) -> None:
        """Pick a file to put over the open image."""
        from PyQt6.QtWidgets import QFileDialog

        if self.source is None:
            return
        start = str(self.source.path.parent)
        path, _f = QFileDialog.getOpenFileName(
            self.viewer, "Add an overlay", start,
            "Images (*.nii *.nii.gz *.mgz *.mgh);;All files (*)")
        if path:
            self.add_overlay(Path(path))

    def toggle_play(self) -> None:
        if self._play_timer.isActive():
            self.stop_play()
            return
        if views.series_layer(self.ctx.store)[0] is None:
            return
        fps = self.ctx.settings.volume.fps
        self._play_timer.start(max(16, int(1000 / fps)))
        self.ctx.playing = True
        self.ctx.playing_changed.emit(True)
        self.viewer.refresh_actions()

    def stop_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self.ctx.playing:
            self.ctx.playing = False
            self.ctx.playing_changed.emit(False)
            self.viewer.refresh_actions()

    def _play_tick(self) -> None:
        layer, src = views.series_layer(self.ctx.store)
        if layer is None or src is None:
            self.stop_play()
            return
        nxt = (layer.frame + 1) % max(src.n_frames, 1)
        if not src.frame_ready(nxt):
            return   # wait for the stream to catch up rather than skip
        self.ctx.run("frame.set", frame=nxt, layer=layer.id)

    # ------------------------------------------------------------------
    # What is on screen
    # ------------------------------------------------------------------

    @property
    def graph(self):
        """The time-course graph, once it has been shown."""
        return self._graph

    @property
    def render_canvas(self):
        """The 3-D canvas (None without a GPU)."""
        return self._render

    def figure_widget(self) -> QWidget:
        """What a screenshot shows: the views (and the graph, when it is
        open), never a panel or an editor. The mosaic's figure is its canvas."""
        page = self.pages.currentWidget()
        if self.ctx.scene.mode == "mosaic" and page is not None and hasattr(page, "canvas"):
            return page.canvas
        return self.vsplit

    # -- the side column ----------------------------------------------------

    def inspector_open(self) -> bool:
        return self.side.isVisible()

    def set_inspector(self, on: bool, *, remember: bool = True) -> None:
        if on and self._inspector is None:
            from ..panels.inspector import Inspector

            self._inspector = Inspector(self.ctx, self)
            self._inspector.add_requested.connect(self.choose_overlay)
            self.side.setWidget(self._inspector)
            # Never narrower than its controls: with no horizontal scroll bar
            # a narrower column cut them off at its edge.
            self.side.setMinimumWidth(
                self._inspector.minimumSizeHint().width()
                + self.side.verticalScrollBar().sizeHint().width() + 2)
        was = self.side.isVisible()
        self.side.setVisible(bool(on))
        if on and not was:
            self.restore_sizes("volume.side", self.hsplit, [0.76, 0.24])
        if remember and self.ctx.settings.volume.inspector != bool(on):
            self.ctx.settings_hub.update(lambda s: setattr(s.volume, "inspector", bool(on)))
        self.viewer.refresh_actions()

    @property
    def inspector(self):
        return self._inspector

    def restore_panels(self) -> None:
        """Open the controls column if the user left it open."""
        if self.ctx.settings.volume.inspector and not self.inspector_open():
            self.set_inspector(True, remember=False)

    def canvases(self, kind: str) -> list:
        if kind == "render":
            return [self._render] if self._render is not None and self._render.isVisible() else []
        if kind == "graph":
            return [self._graph] if self._graph is not None and self._graph.isVisible() else []
        page = self.pages.currentWidget()
        if page is None:
            return []
        if kind == "mosaic":
            from ..canvases.mosaic import MosaicCanvas

            return [w for w in page.findChildren(MosaicCanvas) if w.isVisible()]
        if page is self.tiles:
            return [w for t in self.tiles.tiles
                    if t != "render" and (w := self.tiles.widget_for(t)) is not None
                    and w.isVisible()]
        return []

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    def summary(self) -> str:
        src = self.source
        if src is None:
            return ""
        shape = "x".join(str(s) for s in src.facts.shape)
        text = f"{shape} · {src.facts.disk_dtype}"
        if src.is_rgb:
            text += " · RGB"
        if src.is_4d and not src.fully_loaded:
            text += f" · reading {src.loaded_frames}/{src.n_frames}"
        if src.truncated:
            text += (f" · first {src.n_frames} of {src.n_frames_total} volumes "
                     "(memory limit, Settings > Viewer)")
        return text

    def readout(self) -> str:
        return views.readout(self.ctx.store)

    def stop(self) -> None:
        self.stop_play()


__all__ = ["VolumePresenter", "open_with_context"]
