"""Every viewer preference, as one typed model.

One schema instead of scattered ``AppSettings`` keys and module globals: the
crosshair, the defaults a newly opened image gets, the 3-D quality, how traces
are drawn, the user's keyboard and mouse bindings, and the layouts used.
Stored as JSON under the ``viz/`` namespace (``gui/viz/settings_store.py``)
and broadcast on change, so an open viewer follows a preference at once.

Each field carries what the Settings page needs to draw it, as data: a
title, a help text, and in ``json_schema_extra`` the widget hints (``range``,
``step``, ``unit``, ``kind`` for a colour or a colour map, ``labels`` for a
choice). The Viewer page is GENERATED from this module, so a preference added
here appears in Settings without anyone building a control for it.

Pure data; validation clamps rather than rejects, so a hand-edited or older
settings file still loads.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .render3d import QUALITY_DEFAULT, QUALITY_MAX, QUALITY_MIN


class _Section(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


def _hint(**extra: Any) -> dict:
    return {"json_schema_extra": extra}


class CrosshairSettings(_Section):
    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Crosshair")

    color: str = Field("#4FC3F7", title="Colour", **_hint(kind="colour"),
                       description="The crosshair on every slice and in the graph.")
    thickness: int = Field(1, title="Thickness", **_hint(range=(1, 5), unit="px"),
                           description="Line width of the crosshair.")
    gap: int = Field(0, title="Gap", **_hint(range=(0, 40), unit="px"),
                     description="Pixels left empty around the centre, so the "
                                 "voxel under the crosshair stays visible.")

    @field_validator("thickness")
    @classmethod
    def _thick(cls, v: int) -> int:
        return max(1, min(int(v), 5))

    @field_validator("gap")
    @classmethod
    def _gap(cls, v: int) -> int:
        return max(0, min(int(v), 40))


class VolumeSettings(_Section):
    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Images")

    #: The layout a scan opens in: "" = the best this machine can show.
    mode: Literal["", "single", "multi", "3d", "combo", "hero", "mosaic"] = Field(
        "", title="Opens in",
        description="The layout a volume opens in. Automatic: the three "
                    "planes, with the 3-D view where there is a GPU. "
                    "Switching layout in the viewer changes this.",
        **_hint(labels={"": "Automatic", "single": "One plane",
                        "multi": "Three planes", "3d": "3-D",
                        "combo": "Three planes and 3-D", "hero": "Hero",
                        "mosaic": "Mosaic"}))
    plane: Literal["sagittal", "coronal", "axial"] = Field(
        "axial", title="Plane", description="The plane of the one-plane and hero layouts.")
    gamma: float = Field(1.25, title="Gamma", **_hint(range=(0.1, 10.0), step=0.05),
                         description="Display gamma of a newly opened image: above 1 "
                                     "lifts the mid-tones.")
    interpolation: Literal["linear", "nearest"] = Field(
        "linear", title="Pixels",
        description="Linear smooths between voxels; nearest shows each voxel "
                    "as a square, as the data is.",
        **_hint(labels={"linear": "Smooth (linear)", "nearest": "Blocky (nearest)"}))
    colormap: str = Field("gray", title="Colour map", **_hint(kind="colormap"),
                          description="The colour map a newly opened image gets.")
    labels: bool = Field(True, title="Orientation letters and cube")
    colorbar: bool = Field(False, title="Colour bar")
    ras: bool = Field(True, title="RAS orientation",
                      description="Off: each axis in the file's own storage order.")
    radiological: bool = Field(False, title="Radiological convention",
                               description="Mirror left and right: the patient's left "
                                           "on the image's right.")
    space: Literal["voxel", "world"] = Field(
        "voxel", title="Slices",
        description="Voxel: the file's own grid (exact, fast). World: the "
                    "scanner's axes, resampled, so an oblique scan is upright.",
        **_hint(labels={"voxel": "Voxel grid", "world": "World space"}))
    #: Frames kept in memory, MB. 0 = automatic (half the free memory).
    memory_mb: int = Field(0, title="Memory for 4-D series",
                           **_hint(range=(0, 262144), step=256, unit="MB", zero="Automatic"),
                           description="The most a 4-D series may hold in memory. "
                                       "Automatic is half of the free memory.")
    #: Playback speed, frames per second.
    fps: float = Field(8.0, title="Playback speed", **_hint(range=(0.5, 60.0), step=0.5,
                                                            unit="volumes/s"))
    inspector: bool = Field(True, title="Controls column open",
                            description="The column beside the images with the layers, "
                                        "their look, the view, the layout and the 3-D "
                                        "controls.")

    @field_validator("gamma")
    @classmethod
    def _gamma(cls, v: float) -> float:
        return max(0.1, min(float(v), 10.0))

    @field_validator("fps")
    @classmethod
    def _fps(cls, v: float) -> float:
        return max(0.5, min(float(v), 60.0))

    @field_validator("memory_mb")
    @classmethod
    def _mem(cls, v: int) -> int:
        return max(0, int(v))


class RenderSettings(_Section):
    model_config = ConfigDict(extra="ignore", validate_assignment=True, title="3-D")

    quality: int = Field(QUALITY_DEFAULT, title="Quality",
                         **_hint(range=(QUALITY_MIN, QUALITY_MAX), step=64, unit="steps"),
                         description="Ray-march steps of a newly opened render. "
                                     "Higher is finer and slower.")

    @field_validator("quality")
    @classmethod
    def _q(cls, v: int) -> int:
        return max(QUALITY_MIN, min(int(v), QUALITY_MAX))


class TraceSettings(_Section):
    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Signals")

    #: 0 = automatic (thin for many traces, 2 px for one or two).
    line_width: int = Field(0, title="Line width", **_hint(range=(0, 8), unit="px",
                                                            zero="Automatic"),
                            description="Automatic: thin for many traces, 2 px for "
                                        "one or two.")
    line_color: str = Field("", title="Line colour", **_hint(kind="colour", empty="By type"),
                            description="One colour for every trace; empty colours "
                                        "each by its channel type.")
    type_colors: dict[str, str] = Field(default_factory=dict, title="Channel type colours")
    dark_plot: bool = Field(False, title="Dark plot in the light theme",
                            description="Draw signals on a dark canvas even when the "
                                        "app is light. In the dark theme the canvas is "
                                        "dark already.")
    event_width: int = Field(2, title="Event line width", **_hint(range=(1, 10), unit="px"))
    event_color: str = Field("", title="Event colour",
                             **_hint(kind="colour", empty="By label"),
                             description="One colour for every event; empty colours "
                                         "each label differently.")

    @field_validator("line_width")
    @classmethod
    def _w(cls, v: int) -> int:
        v = int(v)
        return 0 if v <= 0 else min(v, 8)

    @field_validator("event_width")
    @classmethod
    def _ew(cls, v: int) -> int:
        return max(1, min(int(v), 10))


class VizSettings(_Section):
    crosshair: CrosshairSettings = Field(default_factory=CrosshairSettings)
    volume: VolumeSettings = Field(default_factory=VolumeSettings)
    render: RenderSettings = Field(default_factory=RenderSettings)
    traces: TraceSettings = Field(default_factory=TraceSettings)
    #: action id -> key sequences, overriding the defaults (empty list = unbound).
    keymap: dict[str, list[str]] = Field(default_factory=dict)
    #: "<canvas>:<gesture>" -> tool id, overriding the default mouse map.
    mousemap: dict[str, str] = Field(default_factory=dict)
    #: "<layout>.<splitter>" -> splitter sizes, restored the next time.
    layout_sizes: dict[str, list[int]] = Field(default_factory=dict)
    #: Named views the user saved ("Save view as..."): JSON-able view state
    #: (layout, plane, display, graph, 3-D look), never a file or a cursor.
    view_presets: dict[str, dict] = Field(default_factory=dict)
    #: Per kind of file (a layout preset id such as ``mri.func``), the view
    #: it last had, so a BOLD run opens with its graph and a T1 without.
    layout_state: dict[str, dict] = Field(default_factory=dict)
    #: Which sections of the controls column are open (section key -> open).
    inspector_sections: dict[str, bool] = Field(default_factory=dict)


#: The sections the generated Viewer page shows, in order.
PAGE_SECTIONS = ("crosshair", "volume", "render", "traces")


__all__ = [
    "CrosshairSettings", "PAGE_SECTIONS", "RenderSettings", "TraceSettings",
    "VizSettings", "VolumeSettings",
]
