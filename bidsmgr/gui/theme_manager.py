"""Theme manager — owns the palette + token-based QSS template.

The QSS at ``theme.qss`` (sibling of this module) is a ``string.Template``
with named tokens (``$bg``, ``$accent``, ``$success_bg``, …). On theme
change we ``safe_substitute(**palette)`` and call
``QApplication.setStyleSheet`` with the result.

This file is the proven implementation from the prototype at
``../../inspector_proto/proto.py``. The prototype validated visual
fidelity end-to-end including a working dark↔light toggle on
real Siemens Prisma 3T data.

Usage:

    from bidsmgr.gui.theme_manager import ThemeManager
    theme = ThemeManager(app)
    theme.apply('dark')   # or any id in THEMES ('light', 'dim', 'nord', ...)
    theme.toggle()        # to the partner of the other kind

Other GUI modules subscribe to palette changes:

    theme.add_listener(lambda pal: my_widget.repaint_for_palette(pal))

The current palette is always available via ``theme.palette`` or via
the module-level ``CUR()`` accessor used by paint code that runs outside
the listener flow (e.g. ``QStyledItemDelegate.paint``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Callable

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication


# Baseline app-default font size in logical pixels. Multiplied by the
# current ``FONT_SCALE()`` before being applied to ``QApplication.font``
# in ``ThemeManager.apply`` so the user's chosen scale propagates to
# every widget that doesn't have an explicit QSS ``font-size`` rule.
BASE_FONT_PIXEL_SIZE = 12

# Matches every ``font-size: Npx`` declaration in the QSS template. Used
# by ``_scale_qss_font_sizes`` to multiply each by the active scale.
_FONT_SIZE_RE = re.compile(r"font-size:\s*(\d+)px")


def _scale_qss_font_sizes(qss: str, scale: float) -> str:
    """Multiply every ``font-size: Npx`` in *qss* by *scale*.

    Sizes round to the nearest int and clamp at 1 px so a very small
    scale can't produce ``font-size: 0px`` (which Qt silently rejects).
    """
    if scale == 1.0:
        return qss

    def _replace(m: re.Match[str]) -> str:
        base = int(m.group(1))
        scaled = max(1, round(base * scale))
        return f"font-size: {scaled}px"

    return _FONT_SIZE_RE.sub(_replace, qss)


# =====================================================================
#  PALETTES
# =====================================================================
DARK: dict[str, str] = {
    'bg':         '#0a0e13',
    'surface':    '#11161d',
    'surface2':   '#161b22',
    'surface3':   '#1c2128',
    'border':     '#21262d',
    # The outline of a control a user types into. Deliberately brighter than
    # 'border': that one separates panels quietly, while an input has to be
    # findable at a glance, and on this near-black surface the panel border is
    # all but invisible.
    'input_border': '#586069',
    'subtle':     '#1a1f26',
    'text':       '#e6edf3',
    'dim':        '#8b949e',
    'muted':      '#656d76',
    'accent':     '#58a6ff',
    'success':    '#3fb950',
    'warning':    '#d29922',
    'error':      '#f85149',
    'purple':     '#d2a8ff',
    'teal':       '#39c5cf',

    'muted_40':        'rgba(101,109,118,0.40)',

    'accent_bg':       'rgba(88,166,255,0.12)',
    'accent_border':   'rgba(88,166,255,0.40)',
    'success_bg':      'rgba(63,185,80,0.12)',
    'success_border':  'rgba(63,185,80,0.30)',
    'warning_bg':      'rgba(210,153,34,0.12)',
    'warning_border':  'rgba(210,153,34,0.30)',
    'error_bg':        'rgba(248,81,73,0.12)',
    'error_border':    'rgba(248,81,73,0.30)',
    'purple_bg':       'rgba(210,168,255,0.12)',
    'purple_border':   'rgba(210,168,255,0.30)',
    'teal_bg':         'rgba(57,197,207,0.12)',
    'teal_border':     'rgba(57,197,207,0.30)',

    'primary_btn_text': '#0a0e13',
    'pressed_alpha':    'rgba(255,255,255,0.04)',
}

LIGHT: dict[str, str] = {
    'bg':         '#ffffff',
    'surface':    '#f6f8fa',
    'surface2':   '#ffffff',
    'surface3':   '#eef1f4',
    'border':     '#d0d7de',
    'input_border': '#8c959f',
    'subtle':     '#e5e7ea',
    'text':       '#1f2328',
    'dim':        '#656d76',
    'muted':      '#8c959f',
    'accent':     '#0969da',
    'success':    '#1a7f37',
    'warning':    '#9a6700',
    'error':      '#cf222e',
    'purple':     '#8250df',
    'teal':       '#1d7a8c',

    'muted_40':        'rgba(140,149,159,0.40)',

    'accent_bg':       'rgba(9,105,218,0.08)',
    'accent_border':   'rgba(9,105,218,0.32)',
    'success_bg':      'rgba(26,127,55,0.10)',
    'success_border':  'rgba(26,127,55,0.30)',
    'warning_bg':      'rgba(154,103,0,0.10)',
    'warning_border':  'rgba(154,103,0,0.30)',
    'error_bg':        'rgba(207,34,46,0.10)',
    'error_border':    'rgba(207,34,46,0.30)',
    'purple_bg':       'rgba(130,80,223,0.10)',
    'purple_border':   'rgba(130,80,223,0.30)',
    'teal_bg':         'rgba(29,122,140,0.10)',
    'teal_border':     'rgba(29,122,140,0.30)',

    'primary_btn_text': '#ffffff',
    'pressed_alpha':    'rgba(0,0,0,0.04)',
}



def _rgba(hex6: str, alpha: float) -> str:
    h = hex6.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha:.2f})'


def _derive(base: dict[str, str], *, dark: bool, strong: bool = False) -> dict[str, str]:
    """A whole palette from its base colours: the tinted backgrounds and
    borders of each status colour, the muted wash and the pressed shade,
    with the alphas Dark and Light use (higher for the high-contrast pair)."""
    tint, edge = (0.22, 0.75) if strong else ((0.12, 0.30) if dark else (0.10, 0.30))
    pal = dict(base)
    for key in ('accent', 'success', 'warning', 'error', 'purple', 'teal'):
        pal[f'{key}_bg'] = _rgba(base[key], tint)
        pal[f'{key}_border'] = _rgba(base[key], edge if key != 'accent' or strong
                                     else (0.40 if dark else 0.32))
    pal['muted_40'] = _rgba(base['muted'], 0.40)
    pal['pressed_alpha'] = 'rgba(255,255,255,0.06)' if dark else 'rgba(0,0,0,0.05)'
    return pal


#: A softer dark: mid greys, lower contrast for long sessions.
DIM = _derive({
    'bg': '#1c2128', 'surface': '#22272e', 'surface2': '#2a3038', 'surface3': '#323942',
    'border': '#3d444d', 'input_border': '#6e7681', 'subtle': '#2a3038',
    'text': '#cdd9e5', 'dim': '#9aa6b2', 'muted': '#768390',
    'accent': '#6cb6ff', 'success': '#6bc46d', 'warning': '#daaa3f', 'error': '#f47067',
    'purple': '#dcbdfb', 'teal': '#56d4dd', 'primary_btn_text': '#1c2128',
}, dark=True)

#: Cool blue-grey surfaces with the Nord palette's muted accents.
NORD = _derive({
    'bg': '#272c36', 'surface': '#2e3440', 'surface2': '#353c4a', 'surface3': '#3b4252',
    'border': '#434c5e', 'input_border': '#7b88a1', 'subtle': '#353c4a',
    'text': '#eceff4', 'dim': '#b9c1cf', 'muted': '#8892a6',
    'accent': '#88c0d0', 'success': '#a3be8c', 'warning': '#ebcb8b', 'error': '#ec959c',
    'purple': '#c8a2c4', 'teal': '#8fbcbb', 'primary_btn_text': '#2e3440',
}, dark=True)

#: A warm light: cream surfaces and warm greys, softer than white.
PAPER = _derive({
    'bg': '#fcfaf5', 'surface': '#f4efe4', 'surface2': '#fffdf8', 'surface3': '#ebe4d4',
    'border': '#d9cfbb', 'input_border': '#8f8470', 'subtle': '#e6dece',
    'text': '#33291f', 'dim': '#5f5444', 'muted': '#8d806b',
    'accent': '#1d5f86', 'success': '#3d6b1f', 'warning': '#875400', 'error': '#a8321f',
    'purple': '#6f4697', 'teal': '#1e6b66', 'primary_btn_text': '#ffffff',
}, dark=False)

#: Black, white and strong borders: every text at 7:1 or better (AAA).
HC_DARK = _derive({
    'bg': '#000000', 'surface': '#0b0b0b', 'surface2': '#141414', 'surface3': '#202020',
    'border': '#7a7a7a', 'input_border': '#cfcfcf', 'subtle': '#1a1a1a',
    'text': '#ffffff', 'dim': '#e0e0e0', 'muted': '#b8b8b8',
    'accent': '#5cc8ff', 'success': '#5ef07a', 'warning': '#ffd84d', 'error': '#ff8f8f',
    'purple': '#e7b6ff', 'teal': '#5cf0f0', 'primary_btn_text': '#000000',
}, dark=True, strong=True)

HC_LIGHT = _derive({
    'bg': '#ffffff', 'surface': '#ffffff', 'surface2': '#ffffff', 'surface3': '#ececec',
    'border': '#6e6e6e', 'input_border': '#1a1a1a', 'subtle': '#e3e3e3',
    'text': '#000000', 'dim': '#1f1f1f', 'muted': '#4d4d4d',
    'accent': '#0039a6', 'success': '#08521a', 'warning': '#6b3f00', 'error': '#94000d',
    'purple': '#4f1a99', 'teal': '#00545a', 'primary_btn_text': '#ffffff',
}, dark=False, strong=True)


@dataclass(frozen=True)
class ThemeInfo:
    """A theme the user can pick: its palette, and what it is."""

    id: str
    label: str
    dark: bool
    #: The theme of the other kind ``toggle`` switches to.
    partner: str
    description: str


#: Every theme, in the order the menus list them (dark ones first).
THEMES: tuple[ThemeInfo, ...] = (
    ThemeInfo('dark', 'Dark', True, 'light', 'Near-black surfaces, the default.'),
    ThemeInfo('dim', 'Dim', True, 'paper', 'A softer dark in mid greys, for long sessions.'),
    ThemeInfo('nord', 'Nord', True, 'light', 'Cool blue-grey surfaces with muted accents.'),
    ThemeInfo('hc-dark', 'High contrast dark', True, 'hc-light',
              'Black and white with strong borders.'),
    ThemeInfo('light', 'Light', False, 'dark', 'White surfaces.'),
    ThemeInfo('paper', 'Paper', False, 'dim', 'Warm cream surfaces, softer than white.'),
    ThemeInfo('hc-light', 'High contrast light', False, 'hc-dark',
              'White and black with strong borders.'),
)

PALETTES: dict[str, dict[str, str]] = {
    'dark': DARK, 'dim': DIM, 'nord': NORD, 'hc-dark': HC_DARK,
    'light': LIGHT, 'paper': PAPER, 'hc-light': HC_LIGHT,
}


def theme_info(theme_id: str) -> ThemeInfo:
    """The theme ``theme_id`` (Dark for an id no longer known)."""
    return next((t for t in THEMES if t.id == theme_id), THEMES[0])


def theme_ids() -> list[str]:
    return [t.id for t in THEMES]


# =====================================================================
#  Module-level "current palette" + "current font scale" accessors.
# =====================================================================
_CURRENT: dict[str, str] = DARK
_FONT_SCALE: float = 1.0


def CUR() -> dict[str, str]:
    """Return the active palette dict.

    Used by ``QStyledItemDelegate`` paint methods (which are not GUI
    widgets and don't subscribe to listeners). Updated whenever
    ``ThemeManager.apply`` is called.
    """
    return _CURRENT


def FONT_SCALE() -> float:
    """Return the active UI font-size multiplier (1.0 = baseline).

    Paint code and inline ``setStyleSheet`` snippets use this so their
    hard-coded pixel sizes scale with the user's preference set via
    ``Settings → Display → Font scale``. Updated whenever
    ``ThemeManager.apply`` is called.
    """
    return _FONT_SCALE


def scaled_px(base: int) -> int:
    """Return *base* (px) multiplied by the active font scale, rounded
    to the nearest int and clamped at 1.

    Convenience for call sites that paint or build inline stylesheets:
    ``f.setPixelSize(scaled_px(11))``.
    """
    return max(1, round(base * _FONT_SCALE))


def rgba(hex6: str, alpha: float) -> QColor:
    """Convenience: ``hex6`` color with the given ``alpha`` (0–1)."""
    c = QColor(hex6)
    c.setAlphaF(alpha)
    return c


# =====================================================================
#  ThemeManager
# =====================================================================
class ThemeManager:
    """Owns the QSS template + active palette. Re-applies on toggle.

    Listeners are called *after* the QSS is applied with the new palette
    dict, so they can repaint anything that doesn't pick up automatically.
    """

    def __init__(
        self,
        app: QApplication,
        qss_path: Path | None = None,
        font_scale: float = 1.0,
    ):
        self._app = app
        self._raw_template_text = (
            qss_path or Path(__file__).parent / 'theme.qss'
        ).read_text(encoding='utf-8')
        self._theme = 'dark'
        self._listeners: list[Callable[[dict], None]] = []
        # The font scale is applied to the QSS template + QApplication
        # default font at every ``apply`` call. Stored on the manager so
        # callers can swap it without re-creating the manager.
        self._font_scale = max(0.5, min(float(font_scale), 2.0))

    # ------------------------------------------------------------- listeners
    def add_listener(self, fn: Callable[[dict], None]) -> None:
        """``fn(palette: dict)`` is called after every theme change."""
        self._listeners.append(fn)

    # ---------------------------------------------------------------- state
    @property
    def palette(self) -> dict[str, str]:
        return PALETTES[self._theme]

    @property
    def name(self) -> str:
        return self._theme

    @property
    def font_scale(self) -> float:
        return self._font_scale

    def set_font_scale(self, scale: float) -> None:
        """Update the UI font scale and re-apply the active theme.

        Clamps to ``[0.5, 2.0]`` so a bad value (corrupted setting,
        out-of-range from a future Settings UI) can't render the GUI
        unusable. No-op when the value would not change anything.
        """
        scale = max(0.5, min(float(scale), 2.0))
        if scale == self._font_scale:
            return
        self._font_scale = scale
        # Re-apply the current theme so the QSS gets re-scaled and every
        # listener (panels' ``repaint_for_palette``) refreshes any
        # inline-stylesheet font sizes that were baked at construction.
        self.apply(self._theme)

    # --------------------------------------------------------------- actions
    def apply(self, theme: str) -> None:
        if theme not in PALETTES:
            return
        global _CURRENT, _FONT_SCALE
        self._theme = theme
        pal = PALETTES[theme]
        _CURRENT = pal
        _FONT_SCALE = self._font_scale

        # Apply the active scale to (a) the QSS template every ``font-size:
        # Npx`` declaration, then (b) the QApplication default font's
        # pixel size so widgets without an explicit QSS font-size rule
        # scale too. Custom paint code reads the scale via ``FONT_SCALE()``.
        scaled_template = Template(
            _scale_qss_font_sizes(self._raw_template_text, self._font_scale)
        )
        # The stylesheet's images (tick, chevrons) are drawn in this
        # palette's colours and named by token beside its colours.
        from .theme_assets import write as write_theme_images

        tokens = {**pal, **write_theme_images(pal, theme)}
        self._app.setStyleSheet(scaled_template.safe_substitute(**tokens))
        self._update_qpalette(pal)
        try:
            app_font = QFont(self._app.font())
            app_font.setPixelSize(
                max(1, round(BASE_FONT_PIXEL_SIZE * self._font_scale))
            )
            self._app.setFont(app_font)
        except Exception:
            pass

        for fn in self._listeners:
            try:
                fn(pal)
            except Exception as exc:  # pragma: no cover — never let a listener crash the app
                print(f'[theme listener] {exc}')

    @property
    def is_dark(self) -> bool:
        return theme_info(self._theme).dark

    def toggle(self) -> str:
        """Switch to the current theme's partner of the other kind (Dark and
        Light, Dim and Paper, the two high-contrast themes)."""
        self.apply(theme_info(self._theme).partner)
        return self._theme

    # ------------------------------------------------------------- internals
    def _update_qpalette(self, pal: dict[str, str]) -> None:
        """Set ``QPalette`` baseline so widgets QSS doesn't fully cover behave."""
        p = QPalette()
        p.setColor(QPalette.ColorRole.Window,          QColor(pal['bg']))
        p.setColor(QPalette.ColorRole.WindowText,      QColor(pal['text']))
        p.setColor(QPalette.ColorRole.Base,            QColor(pal['bg']))
        p.setColor(QPalette.ColorRole.AlternateBase,   QColor(pal['surface']))
        p.setColor(QPalette.ColorRole.Text,            QColor(pal['text']))
        p.setColor(QPalette.ColorRole.Button,          QColor(pal['surface3']))
        p.setColor(QPalette.ColorRole.ButtonText,      QColor(pal['text']))
        p.setColor(QPalette.ColorRole.Highlight,       QColor(pal['accent']))
        p.setColor(QPalette.ColorRole.HighlightedText, QColor(pal['primary_btn_text']))
        # Rich-text ``<a href>`` links (welcome-panel resources) read this role,
        # so theme swaps recolour them automatically.
        p.setColor(QPalette.ColorRole.Link,            QColor(pal['accent']))
        p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(pal['surface2']))
        p.setColor(QPalette.ColorRole.ToolTipText,     QColor(pal['text']))
        self._app.setPalette(p)
