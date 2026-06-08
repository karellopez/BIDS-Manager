"""BIDS dataset tree (Editor view, left pane).

Walks ``<bids_root>/sub-*/ses-*/<datatype>/`` and renders the result
as a ``QTreeWidget`` with BIDS-aware coloring:

* directories  → accent
* ``.nii(.gz)`` → text
* ``.json``    → purple
* ``.tsv(.gz)``→ teal
* other files  → dim

Folder-shaped recordings (``.ds`` for CTF MEG, ``.mff`` for EGI EEG)
collapse to a single leaf so the tree mirrors how BIDS treats them
(one recording = one node).

The :class:`BidsTreeDelegate` reads a per-row severity badge published
at ``BADGE_ROLE``. Step 2 leaves that empty; Step 3 fills it once the
validator runs.

Selection emits :pyattr:`file_selected` with the absolute path so
later steps (sidecar form, validation panel) can react.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QFileSystemWatcher, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QLabel,
    QStackedLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import icons
from ..delegates.bids_tree import BADGE_ROLE, BidsTreeDelegate
from ..theme_manager import CUR
from .primitives import PaneHeader

# Severity ordering for folder rollup — pick the worst of any descendant.
_SEVERITY_RANK: dict[str, int] = {"ok": 0, "warn": 1, "err": 2}

log = logging.getLogger(__name__)


# Walk depth cap. A BIDS path is at most
# ``<root>/sub-X/ses-Y/<datatype>/<file>`` (4 levels deep from root),
# plus a few extra for derivatives subtrees. Eight is a generous cap.
_MAX_DEPTH = 8

# Junk / scratch dirs we never want to expose in the tree.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".svn", ".hg", "__pycache__",
    ".tmp", ".tmp_bidsmgr", ".bidsmgr",
    "node_modules", ".idea", ".vscode",
})

# Directories whose contents are *one* BIDS recording — collapse them
# to a leaf node and stop recursing.
_FOLDER_RECORDING_SUFFIXES: tuple[str, ...] = (".ds", ".mff")

# Item data roles.
PATH_ROLE = Qt.ItemDataRole.UserRole          # absolute path string
COLOR_TOKEN_ROLE = Qt.ItemDataRole.UserRole + 1  # palette token for foreground


def _color_token_for(entry_name: str, is_dir: bool) -> str:
    """Pick the palette token used to color a row.

    Folder-recordings (``.ds`` / ``.mff``) are colored like data files
    (``text``) even though they're directories on disk, because the
    user thinks of them as recordings.
    """
    lower = entry_name.lower()
    if is_dir and not lower.endswith(_FOLDER_RECORDING_SUFFIXES):
        return "accent"
    if lower.endswith(".nii.gz") or lower.endswith(".nii"):
        return "text"
    if lower.endswith(".json"):
        return "purple"
    if lower.endswith(".tsv") or lower.endswith(".tsv.gz"):
        return "teal"
    if lower.endswith(_FOLDER_RECORDING_SUFFIXES):
        # CTF .ds / EGI .mff — folder-shaped recordings.
        return "text"
    return "dim"


def _is_folder_recording(name: str) -> bool:
    return name.lower().endswith(_FOLDER_RECORDING_SUFFIXES)


def _norm_path(p) -> str:
    """Resolve a path to a comparable string.

    Resolving both sides means macOS ``/private/var/...`` vs ``/var/...``
    symlink differences between the tree paths and the validation report paths
    don't cause spurious badge misses.
    """
    try:
        return str(Path(p).resolve())
    except OSError:
        return str(Path(p))


def _walk(
    folder: Path,
    parent_item: QTreeWidgetItem,
    *,
    depth: int,
    dirs: Optional[list[str]] = None,
) -> None:
    """Populate ``parent_item`` with the contents of ``folder``.

    When ``dirs`` is supplied, every directory recursed into is appended to it
    so the caller can register them with a ``QFileSystemWatcher`` for live
    refresh (mirrors the Converter's output tree).
    """
    if depth >= _MAX_DEPTH:
        return
    try:
        entries = sorted(
            os.scandir(folder),
            # Directories before files; within each group, case-insensitive.
            key=lambda e: (not e.is_dir(), e.name.lower()),
        )
    except (PermissionError, FileNotFoundError) as exc:
        log.debug("scandir failed for %s: %s", folder, exc)
        return

    pal = CUR()
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.name in _SKIP_DIRS:
            continue
        is_dir = entry.is_dir()
        item = QTreeWidgetItem([entry.name])
        token = _color_token_for(entry.name, is_dir)
        item.setData(0, PATH_ROLE, entry.path)
        item.setData(0, COLOR_TOKEN_ROLE, token)
        item.setForeground(0, QColor(pal[token]))
        item.setIcon(0, icons.icon_for_path(entry.name, is_dir=is_dir))
        parent_item.addChild(item)
        # Recurse only into real directories that are not folder-recordings.
        if is_dir and not _is_folder_recording(entry.name):
            if dirs is not None:
                dirs.append(entry.path)
            _walk(Path(entry.path), item, depth=depth + 1, dirs=dirs)


class BidsTreePane(QWidget):
    """Left pane of the Editor view — BIDS dataset tree.

    Construct, then call :meth:`set_root` once a directory is chosen.
    Emits :pyattr:`file_selected` (absolute :class:`Path`) when the
    user picks a leaf row.
    """

    file_selected = pyqtSignal(Path)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._root: Optional[Path] = None
        # Remember the last severity badges so a live (watcher-driven) refresh
        # re-applies them onto the freshly-walked items instead of dropping
        # them. Keyed by normalised absolute path string.
        self._last_badges: dict[str, str] = {}

        # Live refresh: every visible directory is registered with a
        # ``QFileSystemWatcher`` so files created / deleted / renamed under the
        # BIDS root update the tree without a manual reopen (parity with the
        # Converter's output tree). Bursts of events (e.g. a conversion writing
        # into the open dataset) are coalesced through a 500 ms debounce.
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_fs_changed)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self.refresh)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("BIDS dataset"))

        # Stack the tree on top of an empty-state hint; we flip between
        # them as the user opens / clears a root.
        self._stack = QStackedLayout()
        self._stack.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self._stack, 1)

        self._tree = QTreeWidget()
        self._tree.setObjectName("raw-tree")
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(False)
        self._tree.setIndentation(14)
        from ..theme_manager import scaled_px
        _tree_ico = scaled_px(icons.DEFAULT_TREE_ICON_SIZE)
        self._tree.setIconSize(QSize(_tree_ico, _tree_ico))
        self._tree.setItemDelegate(BidsTreeDelegate(self._tree))
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)

        self._hint = QLabel(
            "No BIDS dataset opened.\n\n"
            "Use “Open BIDS root…” in the toolbar."
        )
        self._hint.setObjectName("pane-hint")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)

        self._stack.addWidget(self._hint)   # index 0
        self._stack.addWidget(self._tree)   # index 1
        self._stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def root(self) -> Optional[Path]:
        return self._root

    def set_root(self, path: Optional[Path]) -> None:
        """Switch the tree to a new BIDS root (or clear it with ``None``).

        Re-setting the SAME root (e.g. the editor re-opening the dataset it is
        already showing) preserves the user's expansion / selection by routing
        through :meth:`refresh`, exactly like the Converter's output tree. Only
        a genuinely new root forgets badges + expansion and opens to depth 2.

        Repopulates synchronously. Datasets are small enough that the walk
        completes well under a frame on any modern disk; if that ever stops
        being true we can swap to the threadpool pattern ``OutputFsPane`` uses.
        """
        if (
            path is not None
            and self._root is not None
            and Path(path) == self._root
            and self._tree.topLevelItemCount() > 0
        ):
            # Same root, already rendered -> in-place refresh (keep the view).
            self.refresh()
            return

        self._clear_watcher()
        self._tree.clear()
        if path is None or not path.exists() or not path.is_dir():
            self._root = None
            self._last_badges = {}
            self._stack.setCurrentIndex(0)
            return

        self._root = path
        self._last_badges = {}  # new root -> stale badges no longer apply
        dirs = self._populate(path)
        self._watch(dirs)
        self._tree.expandToDepth(2)
        self._stack.setCurrentIndex(1)

    def refresh(self) -> None:
        """Re-walk the current root, preserving the user's view.

        Unlike :meth:`set_root`, this keeps whatever the user had expanded /
        selected (and the scroll position) and re-applies the last severity
        badges, so a live filesystem change does not collapse the tree or wipe
        validation markers. No-op if no root is set.
        """
        if self._root is None:
            return
        snap = self._snapshot_state()
        self._clear_watcher()
        # Block selection signals so re-selecting the same row after the rebuild
        # does not spuriously reload the center viewer on every disk tick.
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            if not self._root.exists() or not self._root.is_dir():
                self._root = None
                self._stack.setCurrentIndex(0)
                return
            dirs = self._populate(self._root)
            self._watch(dirs)
            self._restore_state(snap)
        finally:
            self._tree.blockSignals(False)
        # Re-apply badges from the cache (validation has not re-run, but the
        # markers are still valid for files that survived the change).
        if self._last_badges:
            self._apply_badge_map(self._last_badges)

    def _populate(self, path: Path) -> list[str]:
        """Build the tree under ``path`` and return the dirs to watch."""
        pal = CUR()
        # Top-level item carries the dataset name. We do NOT colour it
        # as a directory — the dataset root is the user's anchor, so
        # we paint it in the default text token.
        top = QTreeWidgetItem([path.name or str(path)])
        top.setData(0, PATH_ROLE, str(path))
        top.setData(0, COLOR_TOKEN_ROLE, "text")
        top.setForeground(0, QColor(pal["text"]))
        top.setIcon(0, icons.icon_for_path(path.name or str(path), is_dir=True))
        self._tree.addTopLevelItem(top)
        dirs: list[str] = [str(path)]
        _walk(path, top, depth=0, dirs=dirs)
        self._stack.setCurrentIndex(1)
        return dirs

    # ------------------------------------------------------------------
    # Live refresh (QFileSystemWatcher)
    # ------------------------------------------------------------------

    def _watch(self, dirs: list[str]) -> None:
        if dirs:
            self._watcher.addPaths(dirs)

    def _clear_watcher(self) -> None:
        existing = self._watcher.directories()
        if existing:
            self._watcher.removePaths(existing)

    def _on_fs_changed(self, _path: str) -> None:
        """A watched directory changed — schedule one debounced refresh."""
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    # ------------------------------------------------------------------
    # View-state preservation across an in-place refresh
    # ------------------------------------------------------------------

    @staticmethod
    def _item_key(item: QTreeWidgetItem) -> str:
        """Stable identity for a row across rebuilds (its absolute path)."""
        return str(item.data(0, PATH_ROLE) or "")

    def _snapshot_state(self) -> dict:
        """Capture expanded paths + current selection + scroll position."""
        snap: dict = {
            "expanded": set(),
            "selected": None,
            "scroll": self._tree.verticalScrollBar().value(),
        }
        cur = self._tree.currentItem()
        if cur is not None:
            snap["selected"] = self._item_key(cur)

        def _walk_items(item: QTreeWidgetItem) -> None:
            if item.isExpanded():
                snap["expanded"].add(self._item_key(item))
            for i in range(item.childCount()):
                _walk_items(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            _walk_items(self._tree.topLevelItem(i))
        return snap

    def _restore_state(self, snap: dict) -> None:
        """Re-apply expansion / selection / scroll captured by a snapshot."""
        expanded: set = snap["expanded"]
        selected = snap["selected"]

        def _walk_items(item: QTreeWidgetItem) -> None:
            key = self._item_key(item)
            if key in expanded:
                item.setExpanded(True)
            if selected is not None and key == selected:
                self._tree.setCurrentItem(item)
            for i in range(item.childCount()):
                _walk_items(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            _walk_items(self._tree.topLevelItem(i))
        self._tree.verticalScrollBar().setValue(snap["scroll"])

    def set_badges(self, severities: dict[Path, str]) -> None:
        """Stamp per-row severity badges from a path → severity map.

        ``severities`` keys must be absolute paths matching the paths
        stored at :data:`PATH_ROLE`. Files not in the map get no badge.
        Directories receive the **rollup** (worst) severity across
        their visible descendants.
        """
        # Build a normalised lookup (string form) so we don't have to
        # construct ``Path`` for every tree item.
        leaf_map = {_norm_path(p): s for p, s in severities.items()}
        # Remember so a live (watcher-driven) refresh can re-stamp the rebuilt
        # tree without re-running validation.
        self._last_badges = dict(leaf_map)
        self._apply_badge_map(leaf_map)

    def _apply_badge_map(self, leaf_map: dict[str, str]) -> None:
        """Stamp a normalised path -> severity map onto the current tree."""

        def visit(item: QTreeWidgetItem) -> str | None:
            """Set this item's badge; return the rolled-up severity for
            propagation to the parent."""
            children_worst: str | None = None
            for i in range(item.childCount()):
                child_sev = visit(item.child(i))
                if child_sev is not None:
                    if children_worst is None or \
                            _SEVERITY_RANK[child_sev] > _SEVERITY_RANK[children_worst]:
                        children_worst = child_sev
            if item.childCount() > 0:
                # Directory: rollup-only badge (file-level wins over
                # implicit "ok" because we have no leaf severity to
                # represent the folder itself).
                badge = children_worst
            else:
                # Leaf: look up by absolute path (normalised).
                path_str = item.data(0, PATH_ROLE)
                badge = leaf_map.get(_norm_path(path_str)) if path_str else None
            if badge:
                item.setData(0, BADGE_ROLE, badge)
            else:
                item.setData(0, BADGE_ROLE, None)
            return badge

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i))
        # Force the delegate to repaint with the new badge data.
        self._tree.viewport().update()

    def clear_badges(self) -> None:
        """Remove every badge from the tree (and forget the cached map)."""
        self._last_badges = {}
        def visit(item: QTreeWidgetItem) -> None:
            item.setData(0, BADGE_ROLE, None)
            for i in range(item.childCount()):
                visit(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i))
        self._tree.viewport().update()

    # ------------------------------------------------------------------
    # Theme + selection
    # ------------------------------------------------------------------

    def repaint_for_palette(self, pal: dict) -> None:
        """Re-colour every row in place using the new palette.

        Iterates the tree without re-walking disk — each item carries
        its palette-token name at :data:`COLOR_TOKEN_ROLE` so we can
        just look the new color up.
        """
        def visit(item: QTreeWidgetItem) -> None:
            token = item.data(0, COLOR_TOKEN_ROLE)
            if token and token in pal:
                item.setForeground(0, QColor(pal[token]))
            # Re-tint the type icon from the cleared cache.
            path_str = item.data(0, PATH_ROLE) or item.text(0)
            try:
                is_dir = Path(path_str).is_dir() if path_str else False
            except OSError:
                is_dir = False
            item.setIcon(0, icons.icon_for_path(item.text(0), is_dir=is_dir))
            for i in range(item.childCount()):
                visit(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i))

    def _on_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        path_str = items[0].data(0, PATH_ROLE)
        if path_str:
            self.file_selected.emit(Path(path_str))


__all__ = ["BidsTreePane", "PATH_ROLE"]
