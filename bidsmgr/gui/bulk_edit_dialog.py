"""Modal dialog: apply one value to one column across every selected row.

Reached from the **✎ Bulk edit…** toolbar button (enabled when ≥ 2
rows are selected in the inspection table). Dispatches through
:meth:`InventoryTableModel.bulk_set` so entity rebuilds, mirror cells
and the basename column all stay in sync.

For columns that have a schema-bounded set of values (``datatype``,
``suffix``), the dialog offers a combo box populated from the schema
engine; everything else uses a free-form ``QLineEdit``.
"""

from __future__ import annotations

from typing import Iterable, Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from .. import schema as schema_mod
from .delegates import builtin_montages
from .models import COLUMNS, InventoryTableModel


# Human-readable header per column key. Keeps the dropdown clear about
# what each option actually does.
_COLUMN_DESCRIPTION: dict[str, str] = {
    "id":        "Subject identifier — updates BIDS_name AND the subject entity.",
    "dataset":   "Dataset slug — the convert verb groups rows by this column.",
    "ses":       "Session label (no ``ses-`` prefix; the converter adds it).",
    "task":      "Task entity.",
    "run":       "Run number.",
    "datatype":  "BIDS datatype (anat, func, dwi, …). Triggers a basename rebuild.",
    "suffix":    "BIDS suffix (T1w, bold, …). Triggers a basename rebuild.",
    "line_freq": "EEG/MEG power-line frequency (Hz). Choose 50 or 60.",
    "montage":   "EEG/MEG montage. Choose from the MNE built-in montages.",
    "eeg_reference": "EEG/iEEG reference electrode (e.g. Cz, average).",
    "eeg_ground":    "EEG/iEEG ground electrode.",
    "PatientSex":    "Participant sex. Choose M / F / O.",
    "PatientAge":    "Participant age in years.",
    "Handedness":    "Participant handedness. Choose R / L / A.",
}

# Columns whose value must come from a fixed, non-editable dropdown (never
# free-typed). datatype / suffix stay editable combos (schema-bounded but
# large + interdependent); these are short, closed sets.
_FIXED_CHOICES: dict[str, list[str]] = {
    "line_freq": ["50", "60"],
    "PatientSex": ["M", "F", "O"],
    "Handedness": ["R", "L", "A"],
}


class BulkEditDialog(QDialog):
    """One-shot apply-value-to-column dialog.

    Pass the live model + the list of source DataFrame row indices the
    user has selected. On Apply, calls ``model.bulk_set(rows, key, value)``
    and reports how many rows changed back via the return value of
    :meth:`changed_count` (after ``exec()``).
    """

    def __init__(
        self,
        model: InventoryTableModel,
        rows: Iterable[int],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bulk edit")
        self.setModal(True)
        self.resize(440, 280)
        self._model = model
        self._rows: list[int] = list(rows)
        self._changed: int = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---------- header ----------
        header = QFrame()
        header.setObjectName("issue-dialog-header")
        hl = QVBoxLayout(header)
        hl.setContentsMargins(18, 14, 18, 14)
        hl.setSpacing(2)
        title = QLabel(f"Bulk edit · {len(self._rows)} row"
                       f"{'s' if len(self._rows) != 1 else ''} selected")
        title.setObjectName("issue-dialog-title")
        sub = QLabel(
            "Pick a column and the new value to write into every "
            "selected row. Entity rebuilds + basenames update "
            "automatically."
        )
        sub.setObjectName("issue-dialog-subtitle")
        sub.setWordWrap(True)
        hl.addWidget(title)
        hl.addWidget(sub)
        outer.addWidget(header)

        # ---------- form ----------
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 14)
        bl.setSpacing(8)

        form = QFormLayout()
        form.setSpacing(8)
        form.setContentsMargins(0, 0, 0, 0)

        self._col_combo = QComboBox()
        self._col_combo.setObjectName("ent-input")
        for key in InventoryTableModel.BULK_EDITABLE_KEYS:
            spec = next((c for c in COLUMNS if c.key == key), None)
            label = spec.header if (spec and spec.header) else key
            self._col_combo.addItem(label, userData=key)
        # Then every entity the schema allows on all the selected rows and
        # that has no column of its own. Without these the dialog could not
        # set ``acq``, which is the entity that tells two otherwise
        # identical acquisitions apart and therefore the one most often
        # wanted on a multi-row selection.
        self._entity_descriptions: dict[str, str] = {}
        for entity in self._model.bulk_editable_entities(self._rows):
            try:
                info = schema_mod.entity_info(entity)
            except KeyError:
                continue
            key = f"{InventoryTableModel.ENTITY_KEY_PREFIX}{entity}"
            self._col_combo.addItem(f"{info.name} ({entity})", userData=key)
            self._entity_descriptions[key] = (
                f"{info.description.strip()} "
                f"Written as ``{info.name}-<{info.format.name}>``."
            ).strip()
        self._col_combo.currentIndexChanged.connect(self._on_column_changed)
        form.addRow("Column:", self._col_combo)

        # Value editor — swapped in/out depending on the column kind.
        # For ``datatype`` / ``suffix`` we offer a schema-bounded combo;
        # everything else uses a free-form line edit.
        self._value_edit = QLineEdit()
        self._value_edit.setObjectName("tb-input")
        self._value_combo = QComboBox()
        self._value_combo.setObjectName("ent-input")
        self._value_combo.setEditable(True)  # users can still free-type
        form.addRow("New value:", self._value_edit)
        form.addRow("", self._value_combo)
        self._value_combo.setVisible(False)

        # Clearing an entity is a different instruction from setting it to
        # nothing, and there is no way to type the difference, so it is a
        # tick rather than an empty box. Only entities can be removed: a
        # column is part of the table's shape and an empty one is a blank
        # cell, not an absent field.
        self._remove_check = QCheckBox("Remove this entity from every selected row")
        self._remove_check.setToolTip(
            "Takes the entity off the selected rows entirely, so it stops "
            "appearing in their BIDS names. The value box is ignored."
        )
        self._remove_check.toggled.connect(self._on_remove_toggled)
        form.addRow("", self._remove_check)
        self._value_row_index = 1  # the row we toggle (line edit vs combo)
        # Which editor is active. Tracked explicitly rather than via
        # ``isVisible()`` (unreliable before the dialog is shown / in tests).
        self._value_is_combo = False

        # Per-column description so the user knows what the apply will do.
        self._description = QLabel("")
        self._description.setObjectName("pane-hint")
        self._description.setWordWrap(True)
        self._description.setContentsMargins(0, 0, 0, 0)
        # Reset padding from the global ``pane-hint`` rule for this
        # compact placement.
        self._description.setStyleSheet("padding: 6px 0;")

        bl.addLayout(form)
        bl.addWidget(self._description)
        bl.addStretch(1)

        outer.addWidget(body, 1)

        # ---------- footer ----------
        footer = QFrame()
        footer.setObjectName("issue-dialog-footer")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(14, 10, 14, 10)
        fl.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel,
        )
        self._apply_btn = buttons.button(QDialogButtonBox.StandardButton.Apply)
        self._apply_btn.clicked.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        fl.addWidget(buttons)
        outer.addWidget(footer)

        # Initialise the value editor for whatever column is selected.
        self._on_column_changed(self._col_combo.currentIndex())

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def changed_count(self) -> int:
        """Rows actually modified by the last :meth:`exec` call."""
        return self._changed

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_column_changed(self, _idx: int) -> None:
        """Swap the value editor + refresh the description text."""
        key = self._col_combo.currentData()
        if key is None:
            return
        self._description.setText(
            self._entity_descriptions.get(key)
            or _COLUMN_DESCRIPTION.get(key, "")
        )

        is_entity = key.startswith(InventoryTableModel.ENTITY_KEY_PREFIX)
        self._remove_check.setVisible(is_entity)
        if not is_entity:
            self._remove_check.setChecked(False)

        # Decide which editor to show.
        if key.startswith(InventoryTableModel.ENTITY_KEY_PREFIX):
            # An entity value is free text bounded by the entity's own
            # format, which ``set_entity`` and the basename rebuild check.
            # Offering the values already in use saves retyping one that
            # some rows carry and others do not.
            entity = key[len(InventoryTableModel.ENTITY_KEY_PREFIX):]
            in_use = self._model.entity_values_in_use(self._rows, entity)
            if in_use:
                self._show_value_combo(in_use, editable=True)
            else:
                self._show_value_lineedit()
        elif key == "datatype":
            # Schema-bounded but editable (large set, lets the user type).
            self._show_value_combo(sorted(schema_mod.list_datatypes()), editable=True)
        elif key == "suffix":
            # ``suffix`` depends on datatype, but bulk-applying spans
            # rows that may differ. Offer the union of all datatypes'
            # suffixes so the user can pick something reasonable.
            options = sorted({
                s for dt in schema_mod.list_datatypes()
                for s in schema_mod.list_suffixes(dt)
            })
            self._show_value_combo(options, editable=True)
        elif key == "montage":
            # Dropdown-only: MNE built-in montages, never hand-typed.
            self._show_value_combo(builtin_montages(), editable=False)
        elif key in _FIXED_CHOICES:
            # Short closed sets (line_freq / sex / handedness): dropdown-only.
            self._show_value_combo(_FIXED_CHOICES[key], editable=False)
        else:
            self._show_value_lineedit()

    def _on_remove_toggled(self, removing: bool) -> None:
        """Grey the value editor out: with Remove ticked it is not read."""
        self._value_edit.setEnabled(not removing)
        self._value_combo.setEnabled(not removing)

    def _show_value_combo(self, options: list[str], *, editable: bool = False) -> None:
        self._value_combo.blockSignals(True)
        self._value_combo.clear()
        self._value_combo.setEditable(editable)
        self._value_combo.addItems(options)
        self._value_combo.blockSignals(False)
        self._value_edit.setVisible(False)
        self._value_combo.setVisible(True)
        self._value_is_combo = True

    def _show_value_lineedit(self) -> None:
        self._value_combo.setVisible(False)
        self._value_edit.setVisible(True)
        self._value_is_combo = False

    def _read_value(self) -> str:
        if self._value_is_combo:
            return self._value_combo.currentText().strip()
        return self._value_edit.text().strip()

    def _on_apply(self) -> None:
        key = self._col_combo.currentData()
        if key is None:
            return
        value = self._read_value()

        if not value:
            # An ENTITY can be cleared, and clearing it is the only way to
            # take one off a group of files: an entity the scan proposed
            # but the dataset should not carry. Every other column keeps
            # the old rule, because an empty value there is a user who has
            # not finished typing rather than an instruction.
            if not self._remove_check.isChecked():
                return
        elif self._remove_check.isChecked():
            # Removing wins over a value left in the box, but say so rather
            # than silently discarding what was typed.
            value = ""

        self._changed = self._model.bulk_set(self._rows, key, value)
        self.accept()


__all__ = ["BulkEditDialog"]
