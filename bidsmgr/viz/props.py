"""Every editable property of a layer, described once, as data.

A property is a field of :class:`~bidsmgr.viz.scene.VolumeDisplay` (or of
the layer itself) with what a control needs to edit it: a kind, a range and
step, a unit, the choices of a choice, a help text, and WHEN it applies
(``when`` in the action-expression language: ``scalar``, ``overlay``,
``two_tailed``). The Layers panel is generated from this table, so adding a
property to the display model and a line here gives it a control, a tooltip
and a command, with no widget code.

Every property is changed through ``layer.set``, so it is undoable, linkable
between two viewers, and scriptable.

Pure data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Prop:
    key: str
    title: str
    kind: Literal["colormap", "window", "float", "bool", "choice"]
    #: Where the value lives: the layer's ``display`` or the layer itself.
    target: Literal["display", "layer"] = "display"
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.01
    unit: str = ""
    choices: tuple[tuple[Any, str], ...] = field(default_factory=tuple)
    help: str = ""
    #: When the control is offered (expression over the layer context).
    when: str = ""
    #: Which section shows it: ``display`` (how values become colours) or
    #: ``overlay`` (what only a layer drawn over another needs).
    group: str = "display"


LAYER_PROPS: tuple[Prop, ...] = (
    Prop("visible", "Shown", "bool", target="layer",
         help="Hide the layer without removing it."),
    Prop("in_3d", "In 3-D", "bool", group="overlay", target="layer", when="overlay",
         help="Draw this overlay in the 3-D render as well as on the slices."),
    Prop("colormap", "Colour map", "colormap", when="scalar && !labels",
         help="How values turn into colours."),
    Prop("window", "Window", "window", when="scalar && !labels",
         help="The lowest and highest value shown, in the data's own units."),
    Prop("gamma", "Gamma", "float", lo=0.1, hi=10.0, step=0.05, when="scalar && !labels",
         help="Above 1 lifts the mid-tones; below 1 darkens them."),
    Prop("invert", "Invert", "bool", when="scalar && !labels",
         help="Run the colour map backwards."),
    Prop("opacity", "Opacity", "float", lo=0.0, hi=1.0, step=0.05, unit="",
         help="How much of the layers below shows through."),
    Prop("interpolation", "Pixels", "choice",
         choices=(("linear", "Smooth (linear)"), ("nearest", "Blocky (nearest)")),
         help="Nearest shows each voxel as a square, as the data is."),
    Prop("threshold_mode", "Below the window", "choice", when="scalar && !labels",
         choices=(("range", "Shown"), ("hide_below", "Hidden"),
                  ("translucent_below", "Faint")),
         help="How values below the window's low end are drawn: shown like "
              "any other (an anatomical image), hidden (a thresholded "
              "overlay), or faint, so a cluster that nearly reached the "
              "threshold still shows."),
    Prop("colormap_negative", "Negative colour map", "colormap", when="scalar && !labels",
         help="Draw values below zero with their own colour map (a "
              "two-tailed statistic). Empty: no separate negative tail."),
    Prop("window_negative", "Negative window", "window", when="two_tailed && !labels",
         help="Threshold and saturation of the negative tail, as magnitudes."),
    Prop("outline_px", "Outline", "float", group="overlay", lo=0.0, hi=5.0, step=1.0, when="overlay",
         help="Draw only the edge of each region, this many pixels wide: its "
              "extent without hiding the image under it. 0 fills it."),
)

PROP_BY_KEY = {p.key: p for p in LAYER_PROPS}


def layer_context(layer, src) -> dict[str, Any]:
    """The facts a property's ``when`` is evaluated against."""
    is_rgb = bool(getattr(src, "is_rgb", False))
    return {
        "scalar": not is_rgb,
        "rgb": is_rgb,
        "overlay": bool(getattr(layer, "id", "base") != "base"),
        "two_tailed": bool(getattr(getattr(layer, "display", None), "colormap_negative", "")),
        # An atlas is coloured by its table: no colour map, window or gamma.
        "labels": getattr(getattr(layer, "display", None), "label_table", None) is not None,
    }


__all__ = ["LAYER_PROPS", "PROP_BY_KEY", "Prop", "layer_context"]
