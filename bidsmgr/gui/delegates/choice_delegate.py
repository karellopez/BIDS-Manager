"""Combobox cell delegate for constrained-choice inventory columns.

``montage`` and ``line_freq`` must never be free-typed: a montage name has to
be one MNE recognises, and a power-line frequency is one of the two grid
conventions. This delegate paints exactly like :class:`CellTextDelegate` (so the
column looks identical) but opens a non-editable ``QComboBox`` for editing.

A leading "blank" entry maps to the empty string, so a user can clear the cell
back to "use the dataset default".
"""

from __future__ import annotations

import functools
from typing import Callable, Sequence, Union

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox

from .inspection_cells import CellTextDelegate

_FALLBACK_MONTAGES = [
    "standard_1005", "standard_1020", "biosemi16", "biosemi32", "biosemi64",
    "biosemi128", "easycap-M1", "easycap-M10", "GSN-HydroCel-64_1.0",
    "GSN-HydroCel-128", "GSN-HydroCel-256",
]


@functools.lru_cache(maxsize=1)
def builtin_montages() -> list[str]:
    """The MNE built-in montage names (lazy + cached; fallback if MNE absent)."""
    try:
        import mne

        names = list(mne.channels.get_builtin_montages())
        return names or list(_FALLBACK_MONTAGES)
    except Exception:
        return list(_FALLBACK_MONTAGES)


class ChoiceDelegate(CellTextDelegate):
    """Dropdown editor for a constrained column; paints like a text cell.

    Parameters
    ----------
    choices
        Either a list of option strings or a zero-arg callable returning one
        (callable lets the montage list load lazily on first edit).
    role
        Paint role forwarded to :class:`CellTextDelegate`.
    blank_label
        The label shown for the empty-string value (always the first item).
    """

    def __init__(
        self,
        choices: Union[Sequence[str], Callable[[], Sequence[str]]],
        *,
        role: str = "plain",
        blank_label: str = "(none)",
        parent=None,
    ) -> None:
        super().__init__(role, parent)
        self._choices = choices
        self._blank_label = blank_label

    def _options(self) -> list[str]:
        raw = self._choices() if callable(self._choices) else self._choices
        return [self._blank_label] + [str(x) for x in raw]

    def createEditor(self, parent, option, index):  # noqa: N802 (Qt signature)
        cb = QComboBox(parent)
        cb.setEditable(False)  # dropdown-only: never hand-typed
        cb.addItems(self._options())
        return cb

    def setEditorData(self, editor, index):  # noqa: N802
        value = str(
            index.data(Qt.ItemDataRole.EditRole)
            or index.data(Qt.ItemDataRole.DisplayRole)
            or ""
        ).strip()
        if not value:
            editor.setCurrentIndex(0)
            return
        pos = editor.findText(value)
        if pos < 0:
            # Preserve an existing-but-unlisted value (e.g. a custom montage
            # name set via the CLI) rather than silently dropping it.
            editor.addItem(value)
            pos = editor.findText(value)
        editor.setCurrentIndex(pos)

    def setModelData(self, editor, model, index):  # noqa: N802
        text = editor.currentText()
        if text == self._blank_label:
            text = ""
        model.setData(index, text, Qt.ItemDataRole.EditRole)


__all__ = ["ChoiceDelegate", "builtin_montages"]
