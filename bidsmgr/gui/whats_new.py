"""What is new in this version: shown once, the first time it opens.

A user who updates, or installs for the first time, sees one dialog that says
what the release brings: a list of the new features beside a stage that plays
a short animation of each (``gui/news_scenes.py``), one after the other. A
click on a feature, or an arrow key, plays that one, again and again, until
another is picked. Under the stage, the feature's name and what it does.

Shown once per release whose notes this is (``NEWS_FOR``): this one tells
what 1.5.1 and 1.5.2 brought together, so whoever saw the 1.5 news sees it
once more. Called from :func:`bidsmgr.main.main` after the window is up,
never from the window itself, so a test that builds a window gets no dialog.

The animation runs only while the dialog is visible, at 60 frames a second
(30 once a frame costs more than 12 ms), in the theme's colours and the app's
fonts. The particle helpers below (``logo_points``, ``seven_colours``) are
the request for a star's (``gui/star_request.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QElapsedTimer, QRectF, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .news_scenes import ALSO, H, SCENES, W, Scene, paint_scene

#: The release this news describes. Bump it with new text when a release
#: deserves a new dialog; a release without news keeps it, so it is not
#: shown again.
NEWS_FOR = "1.5.2"

DOCS_URL = "https://ancplaboldenburg.github.io/bids_manager_documentation/"
NOTES_URL = DOCS_URL + "updates/#v152"

_LOGO = Path(__file__).parent / "assets" / "logo.png"

#: The seven kinds of data BIDS Manager reads (the star request's figures).
KINDS = ("MRI", "MR spectroscopy", "PET", "EEG", "MEG", "Physiology", "Eye tracking")

#: Seconds a scene takes to appear and to fade at its end.
FADE = 0.35


# ---------------------------------------------------------------------------
# When to show it
# ---------------------------------------------------------------------------


def should_show(seen: str) -> bool:
    """Whether the news of ``NEWS_FOR`` has not been shown yet."""
    return (seen or "").strip() != NEWS_FOR


def maybe_show(window: QWidget) -> Optional["WhatsNewDialog"]:
    """Open the dialog over ``window`` if this release's news is unseen.

    Remembered as soon as it opens, so a crash or a closed laptop does not
    bring it back on every start.
    """
    from .app_settings import AppSettings

    if not should_show(AppSettings.whats_new_seen()):
        return None
    AppSettings.remember_whats_new_seen(NEWS_FOR)
    return show_dialog(window)


def show_dialog(parent: QWidget) -> "WhatsNewDialog":
    """Open the dialog over ``parent`` (the first launch, or Home's What's
    new link); it is deleted when closed, so opening it again adds none."""
    dialog = WhatsNewDialog(parent)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.open()
    return dialog


# ---------------------------------------------------------------------------
# Particles of the logo and the seven kinds' colours (the star request's)
# ---------------------------------------------------------------------------


def logo_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points along the edges of the logo (the folder and the brain's
    folds), centred, the longer side 1."""
    img = QImage(str(_LOGO))
    if img.isNull():  # a missing asset still gives a figure, a ring
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([0.45 * np.cos(a), 0.45 * np.sin(a)], axis=1)
    img = img.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    pixels = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w]
    solid = pixels[..., 3] > 128          # ARGB32 is B, G, R, A in memory
    inner = solid.copy()
    inner[1:-1, 1:-1] &= solid[:-2, 1:-1] & solid[2:, 1:-1] & solid[1:-1, :-2] & solid[1:-1, 2:]
    ys, xs = np.nonzero(solid & ~inner)
    pick = rng.choice(len(xs), size=n, replace=len(xs) < n)
    scale = float(max(w, h))
    return np.stack([(xs[pick] - w / 2) / scale, (ys[pick] - h / 2) / scale], axis=1)


def ease(s: np.ndarray) -> np.ndarray:
    """Ease in and out (cubic), clipped to 0..1."""
    s = np.clip(s, 0.0, 1.0)
    return np.where(s < 0.5, 4 * s ** 3, 1 - (-2 * s + 2) ** 3 / 2)


#: Theme colours a kind of data may take, in order of preference: the plot
#: series first (chosen to stay apart, colour-blind readers included), then
#: other named colours. Which seven are distinct differs per theme (the light
#: themes' fifth series is a violet beside their purple), so they are picked.
_CANDIDATES = ("series1", "series2", "series3", "series4", "series5",
               "purple", "teal", "warning", "error", "success", "accent")


def seven_colours(tokens: dict) -> list[str]:
    """Seven colourful, mutually distinct theme colours for the seven kinds.

    Greedy in OKLab: take the preferred candidates in order, but each next one
    is the candidate farthest from those already taken, among the colourful
    ones (a grey says nothing about a kind of data).
    """
    import math

    from .tones import is_hex, oklch

    pool = []
    for name in _CANDIDATES:
        value = tokens.get(name, "")
        if is_hex(value) and value[:7].lower() not in (c.lower() for c in pool):
            _light, chroma, _hue = oklch(value[:7])
            if chroma >= 0.06:
                pool.append(value[:7])

    def lab(colour: str) -> tuple[float, float, float]:
        light, chroma, hue = oklch(colour)
        return light, chroma * math.cos(math.radians(hue)), chroma * math.sin(math.radians(hue))

    chosen = pool[:1]
    while len(chosen) < len(KINDS) and len(chosen) < len(pool):
        rest = [c for c in pool if c not in chosen]
        chosen.append(max(rest, key=lambda c: min(math.dist(lab(c), lab(x)) for x in chosen)))
    while len(chosen) < len(KINDS):          # a palette with too few colours repeats
        chosen.append(chosen[len(chosen) % max(1, len(pool))] if pool else "#888888")
    return chosen


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


def scene_alpha(t: float, duration: float) -> float:
    """How visible a scene is ``t`` seconds in: it appears, holds, and fades
    at its end, so one scene hands over to the next."""
    return max(0.0, min(1.0, t / FADE, (duration - t) / FADE))


class FeatureStage(QWidget):
    """The scenes, one after the other. A scene the user picked plays again
    until another is picked; until then each hands over to the next."""

    scene_changed = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from .viz import fonts
        from .viz.bridge import ThemeHub, connect_while_alive

        self.setObjectName("whatsnew-stage")
        self.setMinimumWidth(fonts.px(360))
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self._index = 0
        self._auto = True
        self._clock = QElapsedTimer()
        self._frozen: Optional[float] = None
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._paint_ms = 0.0           # a running average of the paint's cost
        self._describe()
        connect_while_alive(ThemeHub.instance().changed, self, lambda w, _t: w.update())

    # -- which scene, and when --------------------------------------------

    def index(self) -> int:
        return self._index

    def scene(self) -> Scene:
        return SCENES[self._index]

    def is_auto(self) -> bool:
        """Whether the scenes still hand over to each other by themselves."""
        return self._auto

    def set_scene(self, index: int, *, by_user: bool = False) -> None:
        """Play scene ``index`` from its start; one the user picked plays
        again until another is picked."""
        self._index = index % len(SCENES)
        if by_user:
            self._auto = False
        if self._clock.isValid():
            self._clock.restart()
        self._describe()
        self.scene_changed.emit(self._index)
        self.update()

    def set_time(self, t: Optional[float]) -> None:
        """Freeze the scene at ``t`` seconds (``None`` lets it run)."""
        self._frozen = t
        self.update()

    def time(self) -> float:
        if self._frozen is not None:
            return self._frozen
        return self._clock.elapsed() / 1000.0 if self._clock.isValid() else 0.0

    def is_running(self) -> bool:
        return self._timer.isActive()

    def _tick(self) -> None:
        if self._frozen is None and self.time() >= self.scene().duration:
            if self._auto:
                self.set_scene(self._index + 1)
            else:
                self._clock.restart()
        self.update()

    def _describe(self) -> None:
        scene = self.scene()
        self.setAccessibleName(f"Animation: {scene.title}")
        self.setAccessibleDescription(scene.shows)

    # -- Qt -----------------------------------------------------------------

    @staticmethod
    def _pads() -> tuple[int, int]:
        """The card's padding around the scene, and the room under it for
        the progress line."""
        from .viz import fonts

        return fonts.px(10), fonts.px(4)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt
        """As tall as the scenes' own proportions (16 by 10) ask."""
        pad, below = self._pads()
        return round((width - 2 * pad) * H / W) + 2 * pad + below

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt
        from .viz import fonts

        width = fonts.px(600)
        return QSize(width, self.heightForWidth(width))

    def showEvent(self, event) -> None:  # noqa: N802 - Qt
        super().showEvent(event)
        if not self._clock.isValid():
            self._clock.start()
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt
        from .theme_manager import CUR
        from .viz import fonts

        clock = QElapsedTimer()
        clock.start()
        palette = CUR()
        scene = self.scene()
        t = self.time()
        if self._frozen is None and self._paint_ms > 12.0 and self._timer.interval() == 16:
            self._timer.setInterval(33)     # a slow machine draws half as often

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        card = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = fonts.px(10)
        p.setPen(QPen(QColor(palette.get("border", "#444444")), 1.0))
        p.setBrush(QColor(palette.get("surface", "#222222")))
        p.drawRoundedRect(card, radius, radius)
        p.setBrush(Qt.BrushStyle.NoBrush)

        pad, below = self._pads()
        stage = card.adjusted(pad, pad, -pad, -pad - below)
        p.save()
        p.setClipRect(card.adjusted(1, 1, -1, -1))
        p.setOpacity(scene_alpha(t, scene.duration) if self._frozen is None else 1.0)
        paint_scene(p, stage, scene, t)
        p.restore()

        # How far the scene has played, along the card's bottom edge.
        done = max(0.0, min(1.0, t / scene.duration))
        y = card.bottom() - fonts.px(6)
        left, right = card.left() + radius, card.right() - radius
        track = QColor(palette.get("subtle", "#333333"))
        p.setPen(QPen(track, max(2.0, fonts.px(2)), Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(int(left), int(y), int(right), int(y))
        if done > 0:
            p.setPen(QPen(QColor(palette.get("dim", "#999999")), max(2.0, fonts.px(2)),
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(int(left), int(y), int(left + (right - left) * done), int(y))
        p.end()
        self._paint_ms = 0.9 * self._paint_ms + 0.1 * clock.nsecsElapsed() / 1e6


class WhatsNewDialog(QDialog):
    """What this release brings: a stage of animations beside the list."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from .viz import fonts

        self.setObjectName("whatsnew-dialog")
        self.setWindowTitle("What's new in BIDS Manager")
        self.setModal(True)
        width, height = fonts.px(1000), fonts.px(600)
        screen = (parent.screen() if parent is not None else None) or self.screen()
        if screen is not None:
            room = screen.availableGeometry()
            width, height = min(width, int(room.width() * 0.94)), min(height,
                                                                       int(room.height() * 0.9))
        self.resize(width, height)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(fonts.px(18), fonts.px(18), fonts.px(22), fonts.px(10))
        body.setSpacing(fonts.px(22))

        show = QVBoxLayout()
        show.setSpacing(fonts.px(4))
        self.stage = FeatureStage(self)
        show.addWidget(self.stage)
        show.addSpacing(fonts.px(8))
        self.scene_title = QLabel(self)
        self.scene_title.setObjectName("whatsnew-scene-title")
        self.scene_text = QLabel(self)
        self.scene_text.setObjectName("whatsnew-scene-text")
        self.scene_text.setWordWrap(True)
        self.scene_text.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        # Room for three lines, so the stage does not jump as the text changes.
        self.scene_text.setMinimumHeight(round(fonts.px(12) * 1.45 * 3))
        show.addWidget(self.scene_title)
        show.addWidget(self.scene_text)
        show.addStretch(1)
        body.addLayout(show, 17)

        side = QVBoxLayout()
        side.setSpacing(fonts.px(2))
        kicker = QLabel("What's new")
        kicker.setObjectName("whatsnew-kicker")
        title = QLabel(f"BIDS Manager {NEWS_FOR}")
        title.setObjectName("whatsnew-title")
        lede = QLabel("What versions 1.5.1 and 1.5.2 bring, together. Pick one to see it "
                      "again.")
        lede.setObjectName("whatsnew-lede")
        lede.setWordWrap(True)
        for widget in (kicker, title, lede):
            side.addWidget(widget)
        side.addSpacing(fonts.px(8))
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.feature_buttons: list[QPushButton] = []
        version = None
        for i, scene in enumerate(SCENES):
            if scene.version != version:
                version = scene.version
                heading = QLabel(f"New in {version}")
                heading.setObjectName("whatsnew-group")
                side.addSpacing(fonts.px(6))
                side.addWidget(heading)
            button = QPushButton(scene.title, self)
            button.setObjectName("whatsnew-feature")
            button.setCheckable(True)
            button.setAutoDefault(False)
            button.setToolTip(scene.text)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _c=False, k=i: self.stage.set_scene(k, by_user=True))
            self._group.addButton(button, i)
            self.feature_buttons.append(button)
            side.addWidget(button)
        side.addSpacing(fonts.px(10))
        also = QLabel(ALSO)
        also.setObjectName("whatsnew-item")
        also.setWordWrap(True)
        side.addWidget(also)
        side.addStretch(1)
        body.addLayout(side, 9)
        outer.addLayout(body, 1)

        footer = QFrame(self)
        footer.setObjectName("whatsnew-footer")
        row = QHBoxLayout(footer)
        row.setContentsMargins(fonts.px(16), fonts.px(10), fonts.px(16), fonts.px(10))
        notes = QPushButton("Release notes", footer)
        notes.setAutoDefault(False)
        notes.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(NOTES_URL)))
        star = QPushButton("Give us a star", footer)
        star.setAutoDefault(False)
        star.setToolTip("Why a star on GitHub helps, and how to give one")
        star.clicked.connect(self._show_star_request)
        docs = QPushButton("Open the documentation", footer)
        docs.setAutoDefault(False)
        docs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        start = QPushButton("Continue", footer)
        start.setDefault(True)
        start.clicked.connect(self.accept)
        row.addWidget(notes)
        row.addWidget(star)
        row.addStretch(1)
        row.addWidget(docs)
        row.addWidget(start)
        outer.addWidget(footer)
        self._buttons = {"notes": notes, "star": star, "docs": docs, "continue": start}

        self.stage.scene_changed.connect(self._show_scene)
        self._show_scene(self.stage.index())
        start.setFocus()

    def _show_scene(self, index: int) -> None:
        scene = SCENES[index]
        self.feature_buttons[index].setChecked(True)
        self.scene_title.setText(scene.title)
        self.scene_text.setText(scene.text)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt
        """Left and right (or up and down) pick the previous or next feature."""
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Up: -1,
                Qt.Key.Key_Right: 1, Qt.Key.Key_Down: 1}.get(Qt.Key(event.key()))
        if step is None:
            super().keyPressEvent(event)
            return
        self.stage.set_scene(self.stage.index() + step, by_user=True)

    def _show_star_request(self) -> None:
        """The request for a GitHub star (``gui/star_request.py``), over this."""
        from .star_request import show_dialog

        show_dialog(self)


__all__ = ["DOCS_URL", "FADE", "FeatureStage", "KINDS", "NEWS_FOR", "NOTES_URL",
           "WhatsNewDialog", "ease", "logo_points", "maybe_show", "scene_alpha", "seven_colours",
           "should_show", "show_dialog"]
