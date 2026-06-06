"""Tiny snapshot-based undo/redo stack for the Editor panes.

Qt-free and content-agnostic: a *snapshot* is whatever opaque value a pane
captures for its editable state (a ``(header, rows)`` tuple for the TSV table,
a deep-copied dict for the JSON sidecar). The pane records the pre-edit
snapshot after each edit; ``undo`` / ``redo`` swap snapshots, pushing the
current state onto the opposite stack so the chain is reversible.

The pane owns the snapshot/restore semantics; this class only sequences them.
"""

from __future__ import annotations

from typing import Any


class SnapshotHistory:
    """Undo/redo stacks of opaque snapshots for one editable document."""

    def __init__(self) -> None:
        self._undo: list[Any] = []
        self._redo: list[Any] = []

    def clear(self) -> None:
        """Drop all history (e.g. when a new file is bound)."""
        self._undo.clear()
        self._redo.clear()

    def record(self, pre_edit_snapshot: Any) -> None:
        """Record the state as it was *before* the edit that just happened.

        A fresh edit diverges the timeline, so the redo stack is cleared.
        """
        self._undo.append(pre_edit_snapshot)
        self._redo.clear()

    def undo(self, current_snapshot: Any) -> Any:
        """Return the snapshot to restore, or ``None`` if nothing to undo.

        ``current_snapshot`` is pushed onto the redo stack so a later redo
        returns here.
        """
        if not self._undo:
            return None
        self._redo.append(current_snapshot)
        return self._undo.pop()

    def redo(self, current_snapshot: Any) -> Any:
        """Return the snapshot to re-apply, or ``None`` if nothing to redo."""
        if not self._redo:
            return None
        self._undo.append(current_snapshot)
        return self._redo.pop()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)


__all__ = ["SnapshotHistory"]
