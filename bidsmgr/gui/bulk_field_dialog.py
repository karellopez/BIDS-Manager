"""Pick the files an edit applies to, before it is applied.

One dialog, three callers: stating a field for more than one recording, fixing
a finding in every file it fired on, and stamping TODO placeholders. All three
ask the same question, so they ask it the same way.

The design rule is that a count is not enough. "12 files" tells the user how
many, not which, and not what each one says now. Every candidate is a row
showing its current value and what it would become, ticked by default only
when it would actually change. A file the standard does not declare the field
for is shown greyed with the reason, rather than hidden, because a user who
expected it there deserves to know why it is not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..editor import bulk_edit as be
from .theme_manager import CUR

_COL_FILE = 0
_COL_NOW = 1
_COL_NEXT = 2


class BulkFieldDialog(QDialog):
    """Choose which files receive ``field = value``.

    Construct with the candidates already computed, or pass ``anchor`` and a
    scope and let the dialog compute them. Call :meth:`exec` and, when it
    returns accepted, read :meth:`selected` and :meth:`value`.
    """

    def __init__(
        self,
        root: Path,
        field: str,
        *,
        candidates: Optional[list[be.FileCandidate]] = None,
        anchor: Optional[Path] = None,
        initial_value: Any = None,
        allow_scope: bool = True,
        title: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._field = field
        self._anchor = anchor
        self._fixed_candidates = candidates
        self._candidates: list[be.FileCandidate] = []
        self.setWindowTitle(title or f"Apply {field} to other files")
        self.setModal(True)
        self.resize(760, 520)

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        lede = QLabel(
            f"Write <b>{field}</b> into the files you tick. Nothing is "
            "written until you press Apply, and the whole batch can be "
            "undone as one step."
        )
        lede.setWordWrap(True)
        v.addWidget(lede)

        # Value.
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Value:"))
        self._value_edit = QLineEdit()
        self._value_edit.setObjectName("ent-input")
        self._value_edit.setPlaceholderText(
            "text, or JSON for a number, list or object"
        )
        if initial_value is not None:
            self._value_edit.setText(_as_text(initial_value))
        self._value_edit.textChanged.connect(self._refresh_preview)
        row.addWidget(self._value_edit, 1)
        v.addLayout(row)

        # Scope.
        self._scope_combo = QComboBox()
        self._scope_combo.setObjectName("ent-input")
        if allow_scope and candidates is None:
            for key in be.SCOPES:
                self._scope_combo.addItem(be.SCOPE_LABELS[key], key)
            self._scope_combo.currentIndexChanged.connect(self._reload)
            srow = QHBoxLayout()
            srow.setSpacing(8)
            srow.addWidget(QLabel("Look in:"))
            srow.addWidget(self._scope_combo, 1)
            v.addLayout(srow)
        else:
            self._scope_combo.setVisible(False)

        # Table.
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["File", "Now", "Becomes"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(_COL_FILE, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(
            _COL_NOW, QHeaderView.ResizeMode.ResizeToContents
        )
        hh.setSectionResizeMode(
            _COL_NEXT, QHeaderView.ResizeMode.ResizeToContents
        )
        v.addWidget(self._table, 1)

        # Selection helpers.
        tools = QHBoxLayout()
        tools.setSpacing(8)
        for label, slot in (
            ("Select all", self._select_all),
            ("Select none", self._select_none),
            ("Only those that change", self._select_changing),
        ):
            btn = QPushButton(label)
            btn.setObjectName("tb-btn")
            btn.clicked.connect(slot)
            tools.addWidget(btn)
        tools.addStretch(1)
        self._summary = QLabel("")
        tools.addWidget(self._summary)
        v.addLayout(tools)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        self._apply_btn = buttons.button(QDialogButtonBox.StandardButton.Apply)
        self._apply_btn.setDefault(True)
        self._apply_btn.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

        self._reload()

    # -- data ----------------------------------------------------------

    def _reload(self) -> None:
        if self._fixed_candidates is not None:
            self._candidates = list(self._fixed_candidates)
        else:
            scope = self._scope_combo.currentData() or be.SCOPE_SAME_KIND
            self._candidates = be.candidates(
                self._root, self._field, anchor=self._anchor, scope=scope,
            )
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._candidates))
        for r, cand in enumerate(self._candidates):
            box = QCheckBox(cand.rel)
            box.setEnabled(cand.applicable)
            if not cand.applicable:
                box.setToolTip(cand.reason)
                box.setObjectName("candidate-off")
            box.stateChanged.connect(self._refresh_summary)
            self._table.setCellWidget(r, _COL_FILE, box)
            now = QTableWidgetItem(cand.current_text())
            now.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(r, _COL_NOW, now)
            nxt = QTableWidgetItem("")
            nxt.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(r, _COL_NEXT, nxt)
        self._refresh_preview()
        self._select_changing()

    # -- value ---------------------------------------------------------

    def value(self) -> Any:
        """The typed value, parsed as JSON when it parses, else as text."""
        text = self._value_edit.text().strip()
        if text == "":
            return ""
        try:
            return json.loads(text)
        except ValueError:
            return text

    def _refresh_preview(self) -> None:
        new = self.value()
        shown = _as_text(new)
        pal = CUR()
        for r, cand in enumerate(self._candidates):
            item = self._table.item(r, _COL_NEXT)
            if item is None:
                continue
            if not cand.applicable:
                item.setText("not applicable")
                item.setForeground(_qcolor(pal["muted"]))
                continue
            item.setText(shown)
            item.setForeground(
                _qcolor(pal["accent"] if cand.would_change(new)
                        else pal["muted"])
            )
        self._refresh_summary()

    # -- selection -----------------------------------------------------

    def _boxes(self) -> list[QCheckBox]:
        out = []
        for r in range(self._table.rowCount()):
            w = self._table.cellWidget(r, _COL_FILE)
            if isinstance(w, QCheckBox):
                out.append(w)
        return out

    def _select_all(self) -> None:
        for b in self._boxes():
            if b.isEnabled():
                b.setChecked(True)

    def _select_none(self) -> None:
        for b in self._boxes():
            b.setChecked(False)

    def _select_changing(self) -> None:
        new = self.value()
        for b, cand in zip(self._boxes(), self._candidates):
            b.setChecked(bool(cand.would_change(new)))

    def _refresh_summary(self) -> None:
        n = sum(1 for b in self._boxes() if b.isChecked())
        total = len(self._candidates)
        self._summary.setText(f"{n} of {total} selected")
        if hasattr(self, "_apply_btn"):
            self._apply_btn.setEnabled(n > 0)

    def selected(self) -> list[be.FileCandidate]:
        """The candidates the user ticked."""
        return [
            cand for b, cand in zip(self._boxes(), self._candidates)
            if b.isChecked()
        ]


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _qcolor(spec: str):
    from PyQt6.QtGui import QColor

    return QColor(spec)


__all__ = ["BulkFieldDialog"]
