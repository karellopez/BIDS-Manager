"""User-facing actions: what a button, a menu entry or a key does.

An action names a command and its parameters, a title, default keys, WHEN it
applies and, for a toggle, WHEN it is on. Toolbars, right-click menus, the
help overlay, the command palette and the keyboard are all built from this
one table, so a shortcut and the button it duplicates can never drift apart,
and the help can never list a key that does nothing (both happened with the
hand-written lists this replaces).

``when`` and ``checked`` are tiny expressions over a context dict the viewer
fills: atoms are flags (``gpu``) or ``key=value`` (``mode=multi``), joined by
``&&`` and ``||`` with ``!`` for negation. Kept deliberately small: anything
cleverer belongs in the context, where it can be tested.

Default keys are data. A user's bindings are stored as overrides
(``VizSettings.keymap``) and merged by :func:`effective_keymap`; conflicts are
reported per context by :func:`conflicts`, never silently resolved.

Pure data, Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ActionDef:
    id: str
    title: str
    #: Command id to run, or ``gui:<name>`` for something only the GUI does
    #: (play, open the help, take a screenshot).
    command: str
    params: Mapping[str, Any] = field(default_factory=dict)
    category: str = "General"
    keys: tuple[str, ...] = ()
    when: str = ""
    checked: str = ""
    #: Whether its button appears at all (empty: always). ``when`` greys an
    #: action out where it does not apply NOW; ``shown`` hides one that can
    #: never apply to this kind of file (Fit all on a MEG recording).
    shown: str = ""
    help: str = ""
    #: Where it shows: "toolbar" actions get a button when a layout lists them.
    short: str = ""

    @property
    def checkable(self) -> bool:
        return bool(self.checked)

    @property
    def label(self) -> str:
        return self.short or self.title


def _a(id_, title, command, params=None, **kw) -> ActionDef:
    return ActionDef(id_, title, command, dict(params or {}), **kw)


#: Every action, in help / palette order.
ACTIONS: tuple[ActionDef, ...] = (
    # -- layout -------------------------------------------------------------
    _a("view.axial", "Axial view", "view.plane", {"plane": "axial", "single": True},
       category="Views", keys=("A",), when="volume", short="Axial",
       checked="mode=single && plane=axial"),
    _a("view.sagittal", "Sagittal view", "view.plane", {"plane": "sagittal", "single": True},
       category="Views", keys=("S",), when="volume", short="Sagittal",
       checked="mode=single && plane=sagittal"),
    _a("view.coronal", "Coronal view", "view.plane", {"plane": "coronal", "single": True},
       category="Views", keys=("C",), when="volume", short="Coronal",
       checked="mode=single && plane=coronal"),
    _a("view.multi", "Multi-planar (three planes)", "view.toggle_mode", {"mode": "multi"},
       category="Views", keys=("M",), when="volume", short="Multi-Planar",
       checked="mode=multi"),
    _a("view.3d", "3-D volume render", "view.toggle_mode", {"mode": "3d"},
       category="Views", keys=("D",), when="volume.3d && gpu", short="3D",
       checked="mode=3d"),
    _a("view.combo", "Multi-planar with 3-D", "view.toggle_mode", {"mode": "combo"},
       category="Views", keys=("P",), when="volume.3d && gpu", short="Multi-Planar 3D",
       checked="mode=combo"),
    _a("view.hero", "Hero: one large view and the others beside it",
       "view.toggle_mode", {"mode": "hero"}, category="Views", when="volume",
       short="Hero", checked="mode=hero"),
    _a("view.mosaic", "Mosaic of many slices", "view.toggle_mode", {"mode": "mosaic"},
       category="Views", when="volume", short="Mosaic", checked="mode=mosaic"),
    _a("view.graph", "Time-course graph", "view.graph", category="Views",
       keys=("G",), when="volume.4d && !mode=3d", short="Graph", checked="graph"),
    # -- display -----------------------------------------------------------
    _a("view.labels", "Orientation labels and cube", "view.flag", {"flag": "labels"},
       category="Display", keys=("O",), when="volume", short="Orientation labels",
       checked="display.labels"),
    _a("view.ras", "RAS orientation (off: the file's own storage order)",
       "view.flag", {"flag": "ras"}, category="Display", when="volume",
       short="RAS", checked="display.ras"),
    _a("view.radiological", "Radiological convention (mirror left and right)",
       "view.flag", {"flag": "radiological"}, category="Display", keys=("L",),
       when="volume", short="Radiological", checked="display.radiological"),
    _a("view.world", "World space (draw oblique scans upright)", "view.space",
       category="Display", keys=("W",), when="volume", short="World space",
       checked="space=world"),
    _a("view.colorbar", "Colour bar", "view.flag", {"flag": "colorbar"},
       category="Display", keys=("B",), when="volume", short="Colour bar",
       checked="display.colorbar"),
    _a("view.crosshair", "Show the crosshair", "view.flag", {"flag": "crosshair"},
       category="Display", keys=("X",), when="volume", short="Crosshair",
       checked="display.crosshair"),
    _a("view.reset", "Reset zoom and pan", "view.reset", category="Display",
       keys=("R",), when="volume", short="Reset zoom"),
    # -- navigation --------------------------------------------------------
    _a("slice.next", "Next slice", "gui:slice", {"n": 1}, category="Navigate",
       keys=("Up",), when="volume && slices"),
    _a("slice.prev", "Previous slice", "gui:slice", {"n": -1}, category="Navigate",
       keys=("Down",), when="volume && slices"),
    _a("slice.next10", "Ten slices forward", "gui:slice", {"n": 10},
       category="Navigate", keys=("PgUp",), when="volume && slices"),
    _a("slice.prev10", "Ten slices back", "gui:slice", {"n": -10},
       category="Navigate", keys=("PgDown",), when="volume && slices"),
    _a("frame.next", "Next volume", "frame.step", {"n": 1}, category="Navigate",
       keys=("Right",), when="volume.4d"),
    _a("frame.prev", "Previous volume", "frame.step", {"n": -1},
       category="Navigate", keys=("Left",), when="volume.4d"),
    _a("frame.first", "First volume", "frame.set", {"frame": 0},
       category="Navigate", keys=("Shift+Left",), when="volume.4d"),
    _a("frame.last", "Last volume", "frame.set", {"frame": 10 ** 9},
       category="Navigate", keys=("Shift+Right",), when="volume.4d"),
    _a("frame.same_shell", "Next volume of this shell", "frame.same_shell", {"n": 1},
       category="Diffusion", keys=("]",), when="volume.dwi",
       help="Step through every volume of the shell on screen: how a bad "
            "direction is found"),
    _a("frame.same_shell_prev", "Previous volume of this shell", "frame.same_shell",
       {"n": -1}, category="Diffusion", keys=("[",), when="volume.dwi"),
    _a("frame.shell", "Next shell", "frame.shell", {"n": 1}, category="Diffusion",
       when="volume.dwi", help="The first volume of the next b-value"),
    _a("frame.shell_prev", "Previous shell", "frame.shell", {"n": -1},
       category="Diffusion", when="volume.dwi"),
    # -- the dataset around the image ---------------------------------------
    _a("nav.run_next", "Next run", "gui:navigate", {"entity": "run", "step": 1},
       category="Dataset", keys=("Alt+Right",), when="nav.run.next",
       help="The same image of the next run, crosshair where it was"),
    _a("nav.run_prev", "Previous run", "gui:navigate", {"entity": "run", "step": -1},
       category="Dataset", keys=("Alt+Left",), when="nav.run.prev"),
    _a("nav.echo_next", "Next echo", "gui:navigate", {"entity": "echo", "step": 1},
       category="Dataset", when="nav.echo.next"),
    _a("nav.echo_prev", "Previous echo", "gui:navigate", {"entity": "echo", "step": -1},
       category="Dataset", when="nav.echo.prev"),
    _a("nav.ses_next", "Next session", "gui:navigate", {"entity": "ses", "step": 1},
       category="Dataset", keys=("Alt+PgDown",), when="nav.ses.next"),
    _a("nav.ses_prev", "Previous session", "gui:navigate", {"entity": "ses", "step": -1},
       category="Dataset", keys=("Alt+PgUp",), when="nav.ses.prev"),
    _a("nav.sub_next", "Next subject", "gui:navigate", {"entity": "sub", "step": 1},
       category="Dataset", keys=("Alt+Down",), when="nav.sub.next",
       help="The same image of the next subject"),
    _a("nav.sub_prev", "Previous subject", "gui:navigate", {"entity": "sub", "step": -1},
       category="Dataset", keys=("Alt+Up",), when="nav.sub.prev"),
    _a("deface.preview", "Preview defacing", "gui:deface_preview", category="Layers",
       when="volume.deface",
       help="What the defacing engine would blank, drawn in red over the image. "
            "Nothing is changed"),
    _a("frame.play", "Play the series", "gui:play", category="Navigate",
       keys=("Space",), when="volume.4d", short="Play", checked="playing"),
    _a("cursor.center", "Centre the crosshair", "cursor.center", category="Navigate",
       keys=("Home",), when="volume"),
    # -- window -------------------------------------------------------------
    _a("window.robust", "Window to the robust range", "window.robust",
       category="Contrast", keys=("F",), when="volume", short="Auto"),
    _a("window.series", "Window for the whole series", "window.robust",
       {"series": True}, category="Contrast", keys=("Shift+F",), when="volume.4d"),
    _a("window.full", "Window to the full range", "window.full",
       category="Contrast", when="volume", short="Full"),
    _a("window.invert", "Invert the colour map", "layer.toggle",
       {"field": "invert"}, category="Contrast", keys=("I",), when="volume",
       checked="layer.invert"),
    _a("view.nearest", "Blocky pixels (nearest neighbour)", "layer.toggle",
       {"field": "nearest"}, category="Display", keys=("N",), when="volume",
       short="Blocky", checked="layer.nearest",
       help="Draw each voxel as a square, as the data is, instead of smoothing "
            "between voxels"),
    # -- 3-D ----------------------------------------------------------------
    _a("clip.toggle", "Clip plane on or off", "clip.toggle", category="3-D slicer",
       keys=("Shift+Z",), when="render", checked="clip.active"),
    _a("clip.axial", "Axial cut", "clip.axis", {"plane": "axial"},
       category="3-D slicer", keys=("Shift+A",), when="render"),
    _a("clip.sagittal", "Sagittal cut", "clip.axis", {"plane": "sagittal"},
       category="3-D slicer", keys=("Shift+S",), when="render"),
    _a("clip.coronal", "Coronal cut", "clip.axis", {"plane": "coronal"},
       category="3-D slicer", keys=("Shift+C",), when="render"),
    _a("clip.invert", "Cut from the other side", "clip.invert",
       category="3-D slicer", keys=("Shift+X",), when="render"),
    _a("clip.at_cursor", "Cut through the crosshair", "clip.toggle_at_cursor",
       category="3-D slicer", keys=("Shift+T",), when="render", checked="clip.at_cursor",
       help="Every clip plane passes through the crosshair: the cut faces are "
            "the slices the 2-D views show, and follow them"),
    _a("clip.corner", "Corner cut at the crosshair", "clip.preset", {"preset": "corner"},
       category="3-D slicer", keys=("Shift+O",), when="render",
       help="Cut out the corner facing you, at the crosshair: the three slices "
            "in 3-D"),
    _a("render.reset_view", "Reset the 3-D camera", "render.reset_view",
       category="3-D", keys=("Shift+R",), when="render"),
    _a("render.left", "Look from the left", "render.preset_view", {"side": "left"},
       category="3-D", when="render"),
    _a("render.right", "Look from the right", "render.preset_view", {"side": "right"},
       category="3-D", when="render"),
    _a("render.front", "Look from the front", "render.preset_view", {"side": "front"},
       category="3-D", when="render"),
    _a("render.top", "Look from the top", "render.preset_view", {"side": "top"},
       category="3-D", when="render"),
    _a("header.show", "Header and sidecar...", "gui:header", category="Layers",
       keys=("H",), when="volume", short="Header",
       help="The image's header beside its sidecar, every disagreement said: "
            "repetition time, slice timing, b-values, units, orientation"),
    _a("overlay.add", "Add an overlay...", "gui:add_overlay", category="Layers",
       when="volume", short="Add overlay",
       help="Draw another image over this one: a statistical map, an atlas, a "
            "mask, another contrast. Its look follows from what it holds."),
    _a("qc.mean", "Mean of the series", "gui:qc", {"map": "mean"}, category="Layers",
       when="volume.4d && volume.loaded", short="Mean image",
       help="The mean over every volume, as an overlay"),
    _a("qc.sd", "Standard deviation of the series", "gui:qc", {"map": "sd"},
       category="Layers", when="volume.4d && volume.loaded", short="Standard deviation",
       help="Where the signal moves: motion at the edges, vessels, ghosts"),
    _a("qc.tsnr", "Temporal SNR of the series", "gui:qc", {"map": "tsnr"},
       category="Layers", when="volume.4d && volume.loaded", short="Temporal SNR",
       help="Mean over standard deviation per voxel: how much of the signal is signal"),
    _a("view.inspector", "Controls column", "gui:inspector", category="Views",
       keys=("Ctrl+I",), when="volume", short="Controls", checked="panel.inspector",
       help="The layers and their look, the view, the layout and the 3-D "
            "controls, grouped by purpose"),
    # -- signals (MEG, EEG, physio) -------------------------------------------
    _a("time.next", "Next page of time", "time.page", {"n": 1}, category="Signals",
       keys=("Right",), when="traces"),
    _a("time.prev", "Previous page of time", "time.page", {"n": -1}, category="Signals",
       keys=("Left",), when="traces"),
    _a("time.start", "Start of the recording", "time.start", category="Signals",
       keys=("Shift+Left",), when="traces", short="|<"),
    _a("time.end", "End of the recording", "time.end", category="Signals",
       keys=("Shift+Right",), when="traces", short=">|"),
    _a("time.zoom_in", "Shorter time window", "time.zoom", {"factor": 0.5},
       category="Signals", keys=("=",), when="traces"),
    _a("time.zoom_out", "Longer time window", "time.zoom", {"factor": 2.0},
       category="Signals", keys=("-",), when="traces"),
    _a("time.fit", "Fit the whole recording", "time.fit", category="Signals",
       keys=("F",), when="traces && fit", shown="physio", short="Fit all",
       help="Put the whole recording in the window at once"),
    _a("channels.next", "Next channel", "traces.scroll", {"n": 1}, category="Signals",
       keys=("Down",), when="traces"),
    _a("channels.prev", "Previous channel", "traces.scroll", {"n": -1}, category="Signals",
       keys=("Up",), when="traces"),
    _a("channels.page_next", "Next page of channels", "traces.page_channels", {"n": 1},
       category="Signals", keys=("PgDown",), when="traces"),
    _a("channels.page_prev", "Previous page of channels", "traces.page_channels",
       {"n": -1}, category="Signals", keys=("PgUp",), when="traces"),
    _a("traces.bigger", "Larger amplitude", "traces.scale", {"factor": 1.25},
       category="Signals", keys=("]",), when="traces"),
    _a("traces.smaller", "Smaller amplitude", "traces.scale", {"factor": 0.8},
       category="Signals", keys=("[",), when="traces"),
    _a("traces.normalize", "Normalise each channel", "traces.normalize",
       category="Signals", keys=("N",), when="traces", short="Normalize",
       checked="normalize",
       help="Each channel scaled to its own range, so none overlaps; off, "
            "channels of one type share a scale"),
    _a("traces.butterfly", "Butterfly: every channel of a type in one band",
       "traces.option", {"field": "butterfly"}, category="Signals", keys=("B",),
       when="traces", short="Butterfly", checked="traces.butterfly",
       help="Overlay the channels of each type, to see what they share"),
    _a("traces.clip", "Clip traces to their band", "traces.option", {"field": "clip"},
       category="Signals", when="traces", short="Clip", checked="traces.clip",
       help="Cut a trace off at one and a half bands, so an artefact cannot "
            "paint over its neighbours"),
    _a("traces.dc", "Remove each channel's mean", "traces.option", {"field": "remove_dc"},
       category="Signals", keys=("D",), when="traces", short="DC",
       checked="traces.remove_dc",
       help="Centre each trace in its band (the mean over the window taken out)"),
    _a("traces.page_scale", "Scale each page to itself", "traces.option",
       {"field": "page_scale"}, category="Signals", when="traces", short="Page scale",
       checked="traces.page_scale",
       help="Off: one amplitude per channel type for the whole recording, so a "
            "quiet page and a noisy one compare. On: each page fills its bands."),
    _a("channels.write_bads", "Write the bad channels to channels.tsv", "gui:write_bads",
       category="Signals", when="traces && meeg && bads.changed", shown="meeg",
       short="Save bads",
       help="Mark the channels you clicked as bad (status) in the recording's "
            "_channels.tsv; undoable in the Editor's history"),
    _a("traces.reset", "Reset the view", "traces.reset", category="Signals",
       keys=("R",), when="traces", short="Reset view",
       help="Back to the opening window, scale, channels and no filter"),
    _a("traces.reset_filters", "Remove the filters", "traces.reset_filters",
       category="Signals", keys=("Shift+R",), when="traces && filtered",
       short="Reset filters"),
    _a("events.toggle", "Show events", "events.show", category="Signals",
       keys=("E",), when="traces && events", short="Events", checked="events.on",
       help="Event markers from the run's events.tsv, or the stim channel's "
            "triggers; turning them on jumps to the first when none is in view"),
    _a("traces.psd", "Power spectrum", "gui:psd", category="Signals", keys=("P",),
       when="traces", short="PSD",
       help="Welch spectrum of the channels shown (raw, or filtered)"),
    _a("traces.line", "Line thickness and colours", "gui:line", category="Signals",
       when="traces || spectrum", short="Line"),
    _a("traces.together", "Every physio file of this run", "gui:together",
       category="Signals", when="physio && relatives", shown="physio",
       short="All of this run",
       checked="together",
       help="The cardiac trace, the belt and the trigger of one run on one "
            "clock: resampled onto the fastest grid, each at its own start time"),
    _a("view.zen", "Zen mode: only the traces", "gui:zen", category="Signals",
       keys=("Z",), when="traces", short="Zen", checked="zen",
       help="Hide the toolbars and the overview and leave the traces the whole "
            "pane; Z again brings them back"),
    _a("signal.close", "Close the signal", "gui:close_signal", category="Signals",
       when="traces && meeg", shown="meeg", short="Close",
       help="Drop the signal and go back to the metadata"),
    # -- spectra (MRS) ------------------------------------------------------
    _a("spectrum.fid", "Show the FID", "spectrum.toggle", {"field": "fid"},
       category="Spectrum", keys=("T",),
       when="spectrum", short="FID", checked="spectrum.fid",
       help="The time-domain signal the scanner measured, against seconds"),
    _a("spectrum.metabolites", "Metabolite positions", "spectrum.toggle",
       {"field": "metabolites"}, category="Spectrum", keys=("L",),
       when="spectrum && proton", short="Metabolites", checked="spectrum.metabolites"),
    _a("spectrum.window", "Standard window (0.2 to 4.2 ppm)", "spectrum.toggle",
       {"field": "standard_window"}, category="Spectrum", keys=("W",), when="spectrum && proton",
       short="Standard window", checked="spectrum.window"),
    _a("spectrum.reference", "Overlay the water reference", "spectrum.toggle",
       {"field": "reference"}, category="Spectrum", keys=("O",), when="spectrum && reference",
       short="Reference", checked="spectrum.reference"),
    _a("spectrum.anatomy", "Show the voxel on the anatomy", "gui:anatomy",
       category="Spectrum", when="spectrum && voxel", short="On anatomy",
       help="Where the spectrum was measured: the voxel outlined on this "
            "subject's anatomical image"),
    _a("spectrum.fit", "Fit the whole band", "gui:spectrum_fit", category="Spectrum",
       keys=("F",), when="spectrum", short="Fit all"),
    _a("spectrum.reset", "Reset the view", "gui:spectrum_reset", category="Spectrum",
       keys=("R",), when="spectrum", short="Reset view"),
    _a("spectrum.reset_processing", "Reset the processing", "spectrum.reset_processing",
       category="Spectrum", keys=("Shift+R",), when="spectrum"),
    _a("spectrum.auto_phase", "Phase automatically", "spectrum.auto_phase",
       category="Spectrum", keys=("P",), when="spectrum && !spectrum.fid", short="Auto phase",
       help="The zero-order phase that makes NAA, creatine and choline upright"),
    _a("spectrum.water", "Fit the height without the water", "spectrum.toggle",
       {"field": "exclude_water"}, category="Spectrum", when="spectrum && proton",
       short="Ignore water", checked="spectrum.exclude_water",
       help="Leave the residual water band (4.4 to 5.0 ppm) out of the height, so a "
            "water peak cannot flatten the metabolites; it then runs off the top"),
    _a("spectrum.gain_up", "Taller (look under the peaks)", "spectrum.gain", {"factor": 1.25},
       category="Spectrum", keys=("]",), when="spectrum"),
    _a("spectrum.gain_down", "Shorter", "spectrum.gain", {"factor": 0.8},
       category="Spectrum", keys=("[",), when="spectrum"),
    # -- saved views ----------------------------------------------------------
    _a("scene.save", "Save as a scene of this dataset...", "gui:save_scene",
       category="Views", when="volume",
       help="The image, its overlays in their looks, the crosshair and the layout, "
            "saved in the dataset (.bidsmgr/viz/scenes) to open again or share"),
    _a("view.command_line", "Show the command line...", "gui:command_line",
       category="Views", when="volume && volume.loaded",
       help="The bidsmgr-view call (or Python) that reproduces this view: to open "
            "it again, or render it to a PNG with no window"),
    _a("views.save", "Save this view as...", "gui:save_view", category="Views",
       when="volume", short="Save view",
       help="Keep the layout, plane, display options, graph and 3-D look under "
            "a name, to apply to any image later"),
    # -- general ------------------------------------------------------------
    _a("edit.undo", "Undo a display change", "gui:undo", category="General",
       keys=("Ctrl+Z",), when="undo"),
    _a("edit.redo", "Redo a display change", "gui:redo", category="General",
       keys=("Ctrl+Shift+Z",), when="redo"),
    _a("export.screenshot", "Save a screenshot", "gui:screenshot",
       category="General", keys=("Ctrl+Shift+S",), when="volume || traces || spectrum",
       short="Screenshot"),
    _a("help.shortcuts", "Keyboard and mouse shortcuts", "gui:help",
       category="General", keys=("F1",), short="Shortcuts"),
    _a("help.palette", "Find an action", "gui:palette", category="General",
       keys=("Ctrl+Shift+P",)),
)

ACTION_BY_ID: dict[str, ActionDef] = {a.id: a for a in ACTIONS}


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


def _atom(atom: str, ctx: Mapping[str, Any]) -> bool:
    atom = atom.strip()
    if not atom:
        return True
    negate = atom.startswith("!")
    if negate:
        atom = atom[1:].strip()
    if "=" in atom:
        key, value = (s.strip() for s in atom.split("=", 1))
        result = str(ctx.get(key, "")) == value
    else:
        result = bool(ctx.get(atom, False))
    return (not result) if negate else result


def evaluate(expr: str, ctx: Mapping[str, Any]) -> bool:
    """Evaluate ``a && b || !c``-style expressions (no parentheses)."""
    expr = (expr or "").strip()
    if not expr:
        return True
    return any(
        all(_atom(term, ctx) for term in clause.split("&&"))
        for clause in expr.split("||")
    )


# ---------------------------------------------------------------------------
# Keymap
# ---------------------------------------------------------------------------


def normalise_key(key: str) -> str:
    """One spelling per key sequence (``ctrl+shift+z`` -> ``Ctrl+Shift+Z``)."""
    parts = [p.strip() for p in key.replace(" ", "").split("+") if p.strip()]
    if not parts:
        return ""
    order = {"Ctrl": 0, "Meta": 1, "Alt": 2, "Shift": 3}
    mods = []
    base = parts[-1]
    for p in parts[:-1]:
        cap = p.capitalize()
        if cap == "Control":
            cap = "Ctrl"
        if cap == "Cmd" or cap == "Command":
            cap = "Ctrl"   # Qt maps Ctrl to Command on macOS
        mods.append(cap)
    mods = sorted(set(mods), key=lambda m: order.get(m, 9))
    base = base if len(base) > 1 else base.upper()
    if base.lower() in ("pgup", "pageup"):
        base = "PgUp"
    if base.lower() in ("pgdown", "pagedown", "pgdn"):
        base = "PgDown"
    if len(base) > 1 and base not in ("PgUp", "PgDown"):
        base = base[0].upper() + base[1:]
    return "+".join(mods + [base])


def effective_keymap(overrides: Optional[Mapping[str, list[str]]] = None) -> dict[str, list[str]]:
    """Every action's keys: the defaults, replaced where the user rebound."""
    out: dict[str, list[str]] = {}
    for action in ACTIONS:
        keys = list(action.keys)
        if overrides and action.id in overrides:
            keys = list(overrides[action.id])
        out[action.id] = [normalise_key(k) for k in keys if normalise_key(k)]
    return out


#: Condition words only one kind of viewer ever sets. An action whose
#: conditions use one of them belongs to that viewer; anything else with a
#: condition is a volume action; an action with none belongs to every viewer.
_KIND_ATOMS = {
    "signal": frozenset({"traces", "meeg", "physio", "fit", "filtered", "events",
                         "events.on", "relatives", "normalize", "together",
                         "traces.butterfly", "traces.clip", "traces.remove_dc",
                         "traces.page_scale", "bads.changed", "zen"}),
    "spectrum": frozenset({"spectrum", "spectrum.fid", "spectrum.metabolites",
                           "spectrum.reference", "spectrum.window", "spectrum.exclude_water",
                           "proton", "reference", "voxel"}),
}
_EVERY_VIEWER = frozenset({"undo", "redo"})
VIEWER_KINDS = ("volume", "signal", "spectrum")


def kinds_of(action: ActionDef) -> frozenset[str]:
    """The viewers (``volume``, ``signal``, ``spectrum``) an action belongs to."""
    import re

    atoms: set[str] = set()
    for expr in (action.when, action.shown, action.checked):
        atoms |= set(re.findall(r"[A-Za-z_][A-Za-z_0-9.]*", expr or ""))
    atoms -= _EVERY_VIEWER
    if not atoms:
        return frozenset(VIEWER_KINDS)
    for kind, vocab in _KIND_ATOMS.items():
        if atoms & vocab:
            return frozenset({kind})
    return frozenset({"volume"})


def _contexts_overlap(a: ActionDef, b: ActionDef) -> bool:
    # Two actions whose WHEN can never both hold may share a key (Shift+A cuts
    # in 3-D only, A switches planes only where slices show). Anything else
    # sharing a key is a conflict the user must see.
    probe_sets = (
        {"traces": True, "fit": True, "filtered": True, "events": True, "physio": True,
         "relatives": True, "meeg": True, "undo": True, "redo": True},
        {"spectrum": True, "proton": True, "reference": True, "undo": True, "redo": True},
        {"volume": True, "volume.4d": True, "volume.3d": True, "gpu": True,
         "slices": True, "render": False, "mode": "single", "undo": True, "redo": True},
        {"volume": True, "volume.4d": True, "volume.3d": True, "gpu": True,
         "slices": False, "render": True, "mode": "3d", "undo": True, "redo": True},
        {"volume": True, "volume.4d": True, "volume.3d": True, "gpu": True,
         "slices": True, "render": True, "mode": "combo", "undo": True, "redo": True},
    )
    return any(evaluate(a.when, ctx) and evaluate(b.when, ctx) for ctx in probe_sets)


def conflicts(keymap: Mapping[str, list[str]]) -> list[tuple[str, str, str]]:
    """``(key, action, other action)`` for every key bound twice where both
    actions can apply at the same time."""
    owners: dict[str, list[str]] = {}
    for action_id, keys in keymap.items():
        for key in keys:
            owners.setdefault(key, []).append(action_id)
    out = []
    for key, ids in owners.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ACTION_BY_ID.get(ids[i]), ACTION_BY_ID.get(ids[j])
                if a is None or b is None or _contexts_overlap(a, b):
                    out.append((key, ids[i], ids[j]))
    return out


__all__ = [
    "ACTIONS", "ACTION_BY_ID", "ActionDef", "VIEWER_KINDS", "conflicts",
    "effective_keymap", "evaluate", "kinds_of", "normalise_key",
]
