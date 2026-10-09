"""The user's own touches on top of a theme: the accent, a tint of the
surfaces, how icons are coloured, and the colours of the file trees.

A theme (``theme_manager.PALETTES``) is a complete palette; an
:class:`Appearance` is laid over it by :func:`apply`, so every theme, the
ones to come included, takes the same choices. Nothing here is a new
colour role: the accent override replaces ``accent`` (and the washes and
button text that follow from it), the tint mixes the accent into the
surfaces, and the file-tree colours replace the ``tree_*`` tokens every
palette carries.

Pure data and arithmetic: no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

#: Accent presets: (label, colour on dark themes, colour on light themes).
#: Each pair reads at 4.5:1 or better on the surfaces of its kind of theme.
ACCENTS: dict[str, tuple[str, str, str]] = {
    "blue": ("Blue", "#7db4ff", "#1f5fc2"),
    "indigo": ("Indigo", "#a9abff", "#4a4ec6"),
    "violet": ("Violet", "#c8a3ff", "#7a3fd1"),
    "pink": ("Pink", "#f79ccf", "#ad2c78"),
    "red": ("Red", "#ff9189", "#b52a26"),
    "orange": ("Orange", "#f8ac6a", "#9c4c08"),
    "amber": ("Amber", "#f2c14e", "#7a5700"),
    "green": ("Green", "#6dd796", "#186d38"),
    "teal": ("Teal", "#5ad7cc", "#0f6c66"),
    "graphite": ("Graphite", "#c4c4cc", "#4a4a53"),
}

#: How action icons are coloured. Status icons (ok, warning, error) and the
#: file trees' icons keep their own colours in every style.
ICON_STYLES: dict[str, str] = {
    "monochrome": "Monochrome (the text colour)",
    "accent": "Accent",
    "colourful": "Colourful (by purpose)",
}

#: The file trees' kinds of entry, by label. Every theme gives each its own
#: colour (the palette's ``tree_<kind>``): folders are the accent, other
#: files the secondary text, and images, sidecars, tables and recordings
#: the theme's own shades.
TREE_KINDS: dict[str, str] = {
    "folder": "Folders",
    "image": "Images (NIfTI)",
    "sidecar": "Sidecars (JSON)",
    "table": "Tables (TSV)",
    "recording": "Recordings (EEG, MEG, physiology)",
    "other": "Other files",
}

#: The strongest tint offered, in percent of the accent mixed into a surface.
MAX_TINT = 12


@dataclass
class Appearance:
    """The user's touches; the defaults change nothing."""

    #: "" (the theme's own), a key of :data:`ACCENTS`, or "#rrggbb".
    accent: str = ""
    #: Percent of the accent mixed into the surfaces, 0 to :data:`MAX_TINT`.
    tint: int = 0
    #: A key of :data:`ICON_STYLES`.
    icons: str = "monochrome"
    #: Kind (a key of :data:`TREE_KINDS`) -> "#rrggbb"; a kind not listed
    #: keeps the theme's colour.
    tree: dict[str, str] = field(default_factory=dict)

    def normalised(self) -> "Appearance":
        """A copy with every value valid (a hand-edited setting cannot stop
        the app opening)."""
        accent = self.accent if (self.accent in ACCENTS or _is_hex(self.accent)) else ""
        icons = self.icons if self.icons in ICON_STYLES else "monochrome"
        tint = max(0, min(MAX_TINT, int(self.tint or 0)))
        tree = {k: v for k, v in (self.tree or {}).items() if k in TREE_KINDS and _is_hex(v)}
        return Appearance(accent=accent, tint=tint, icons=icons, tree=tree)


def _is_hex(value: str) -> bool:
    v = (value or "").strip()
    return len(v) == 7 and v.startswith("#") and all(c in "0123456789abcdefABCDEF"
                                                      for c in v[1:])


def _rgb(hex6: str) -> tuple[int, int, int]:
    h = hex6.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, round(c))) for c in rgb))


def mix(base: str, other: str, amount: float) -> str:
    """``base`` moved ``amount`` (0 to 1) of the way toward ``other``."""
    a, b = _rgb(base), _rgb(other)
    return _hex(tuple(x + (y - x) * amount for x, y in zip(a, b)))


def luminance(hex6: str) -> float:
    def channel(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = _rgb(hex6)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _rgba(hex6: str, alpha: float) -> str:
    r, g, b = _rgb(hex6)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def accent_colour(appearance: Appearance, dark: bool, theme_accent: str) -> str:
    """The accent ``appearance`` asks for on a theme of this kind."""
    if appearance.accent in ACCENTS:
        _label, on_dark, on_light = ACCENTS[appearance.accent]
        return on_dark if dark else on_light
    if _is_hex(appearance.accent):
        return appearance.accent.lower()
    return theme_accent


def apply(palette: Mapping[str, str], appearance: Appearance, *, dark: bool,
          strong: bool = False) -> dict[str, str]:
    """``palette`` with ``appearance`` laid over it."""
    look = appearance.normalised()
    pal = dict(palette)
    accent = accent_colour(look, dark, pal["accent"])
    if accent != pal["accent"]:
        tint, edge = (0.22, 0.75) if strong else ((0.12, 0.40) if dark else (0.10, 0.32))
        pal["accent"] = accent
        pal["accent_bg"] = _rgba(accent, tint)
        pal["accent_border"] = _rgba(accent, edge)
        # The text on an accent button: whichever of the canvas and white
        # reads better on it.
        light_text, dark_text = "#ffffff", ("#0b0b0c" if dark else pal["text"])
        pal["primary_btn_text"] = (light_text if contrast(light_text, accent)
                                   >= contrast(dark_text, accent) else dark_text)
    if look.tint:
        amount = look.tint / 100.0
        for key, share in (("bg", 0.6), ("surface", 1.0), ("surface2", 1.0),
                           ("surface3", 1.0), ("border", 1.2), ("subtle", 1.0)):
            pal[key] = mix(pal[key], accent, min(1.0, amount * share))
    for kind, colour in look.tree.items():
        pal[f"tree_{kind}"] = colour
    if look.accent and not look.tree.get("folder") and palette.get("tree_folder") == palette[
            "accent"]:
        # Folders follow the accent unless the user gave them a colour.
        pal["tree_folder"] = accent
    return pal


def accent_warning(palette: Mapping[str, str]) -> str:
    """A sentence when the accent is hard to read on this theme, else ""."""
    worst = min(contrast(palette["accent"], palette[s]) for s in ("bg", "surface", "surface2"))
    if worst < 3.0:
        return (f"This accent reads at {worst:.1f}:1 on this theme's surfaces; links and "
                "selected items will be hard to see. 4.5:1 or more is comfortable.")
    if worst < 4.5:
        return (f"This accent reads at {worst:.1f}:1 on this theme's surfaces: fine for "
                "buttons, a little light for small text.")
    return ""


__all__ = ["ACCENTS", "Appearance", "ICON_STYLES", "MAX_TINT", "TREE_KINDS", "accent_colour",
           "accent_warning", "apply", "contrast", "mix"]
