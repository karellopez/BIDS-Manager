"""The repairs a dataset can be given, listed with what each would do.

Three actions live here, and they share a shape: work out what is missing,
show it, change nothing until the user says so, and record the change so it
can be taken back.

The companion-file list is a checklist rather than a button, because
generating files is the one action here that adds content to a dataset. A
reader cannot tell a generated stub from an authored file by looking at the
tree, so the user has to see the list before it is written, and every stub
carries the TODO convention that keeps the validator reporting it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ..fixups import associations as assoc
from ..fixups.citation import citation_path, write_citation


class FixupsDialog(QDialog):
    """Dataset-wide repairs: companion files, citation file, TODO stamps."""

    def __init__(self, root: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._missing: list[assoc.MissingAssociation] = []
        self.setWindowTitle("Fix up this dataset")
        self.setModal(True)
        self.resize(780, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(12)

        v.addWidget(_lede(
            "Each of these changes files on disk, and each can be undone as a "
            "single step from the Editor's history."
        ))

        # --- Companion files ------------------------------------------
        v.addWidget(_section_title("Missing companion files"))
        self._assoc_hint = _wrapped("", hint=True)
        v.addWidget(self._assoc_hint)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["File", "What would be written"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self._table, 1)

        tools = QHBoxLayout()
        tools.setSpacing(8)
        for label, slot in (
            ("Select all", lambda: self._set_all(True)),
            ("Select none", lambda: self._set_all(False)),
        ):
            b = QPushButton(label)
            b.setObjectName("tb-btn")
            b.clicked.connect(slot)
            tools.addWidget(b)
        tools.addStretch(1)
        self._generate_btn = QPushButton("Generate selected")
        self._generate_btn.setObjectName("tb-btn")
        self._generate_btn.clicked.connect(self._on_generate)
        tools.addWidget(self._generate_btn)
        v.addLayout(tools)

        # --- Citation file --------------------------------------------
        v.addWidget(_section_title("Citation file"))
        self._citation_hint = _wrapped("", hint=True)
        v.addWidget(self._citation_hint)
        crow = QHBoxLayout()
        crow.addStretch(1)
        self._citation_btn = QPushButton("Write CITATION.cff")
        self._citation_btn.setObjectName("tb-btn")
        self._citation_btn.clicked.connect(self._on_citation)
        crow.addWidget(self._citation_btn)
        v.addLayout(crow)

        # --- TODO placeholders -----------------------------------------
        v.addWidget(_section_title("Unanswered metadata"))
        todo_hint = _wrapped(
            "Writes the literal TODO into every missing required and "
            "recommended sidecar field, so what is unanswered is visible in "
            "the file and reported by validation instead of being an absence "
            "nobody notices. Existing values are never overwritten.\n\n"
            "This runs the metadata engine, which also refreshes "
            "participants.tsv and the scans tables.",
            hint=True,
        )
        v.addWidget(todo_hint)
        trow = QHBoxLayout()
        trow.addStretch(1)
        self._todo_btn = QPushButton("Stamp TODO placeholders")
        self._todo_btn.setObjectName("tb-btn")
        self._todo_btn.clicked.connect(self._on_stamp_todos)
        trow.addWidget(self._todo_btn)
        v.addLayout(trow)

        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.setContentsMargins(16, 8, 16, 12)
        outer.addWidget(buttons)

        self.refresh()

    # -- state ---------------------------------------------------------

    def refresh(self) -> None:
        """Re-scan the dataset and redraw both sections."""
        self._missing = assoc.find_missing(self._root)
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._missing))
        for r, item in enumerate(self._missing):
            box = QCheckBox(item.rel)
            box.setChecked(True)
            self._table.setCellWidget(r, 0, box)
            what = QLabel(item.label)
            what.setToolTip(f"Belongs to {item.source.name}")
            self._table.setCellWidget(r, 1, what)
        self._table.resizeColumnToContents(0)
        if self._missing:
            self._assoc_hint.setText(
                f"{len(self._missing)} file(s) the standard associates with a "
                "recording are not there. What can be read from the recording "
                "is read from it; the rest is written as a stub that stays "
                "invalid, and visible to validation, until you fill it in."
            )
        else:
            self._assoc_hint.setText(
                "Every recording has the companion files the standard "
                "associates with it."
            )
        self._generate_btn.setEnabled(bool(self._missing))

        exists = citation_path(self._root).exists()
        self._citation_hint.setText(
            "CITATION.cff is already here. Writing it again would overwrite "
            "it, so the button is disabled."
            if exists else
            "Generated from dataset_description.json. BIDS treats the "
            "citation file as the single source for Authors, License, "
            "HowToAcknowledge and ReferencesAndLinks, so those move out of "
            "the description rather than being duplicated."
        )
        self._citation_btn.setEnabled(not exists)

    def _boxes(self) -> list[QCheckBox]:
        out = []
        for r in range(self._table.rowCount()):
            w = self._table.cellWidget(r, 0)
            if isinstance(w, QCheckBox):
                out.append(w)
        return out

    def _set_all(self, state: bool) -> None:
        for b in self._boxes():
            b.setChecked(state)

    # -- actions -------------------------------------------------------

    def _on_generate(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        chosen = [
            item for b, item in zip(self._boxes(), self._missing)
            if b.isChecked()
        ]
        if not chosen:
            return
        created, failed = assoc.generate(self._root, chosen)
        if failed:
            QMessageBox.warning(
                self, "Some files were not written",
                "\n".join(f"{p}: {why}" for p, why in failed[:6]),
            )
        else:
            QMessageBox.information(
                self, "Companion files written",
                f"{len(created)} file(s) created.\n\nStubs carry TODO "
                "rows and stay invalid until you fill them in: an events "
                "table will now report an error, because TODO is not a valid "
                "onset. That is deliberate. A valid but empty table would "
                "look exactly like a recording that had no events.",
            )
        self.refresh()

    def _on_stamp_todos(self) -> None:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        from ..metadata.engine import run_metadata

        answer = QMessageBox.question(
            self, "Stamp TODO placeholders",
            "Every missing required and recommended field gets the literal "
            "TODO written into it, and participants.tsv and the scans tables "
            "are refreshed.\n\nValidation will then report each TODO, which "
            "is the point: an unanswered field becomes visible rather than "
            "absent.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            report = run_metadata(
                self._root, fill_todos=True, write_report=False,
            )
        except Exception as exc:  # noqa: BLE001 - report rather than crash
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Could not stamp placeholders", str(exc))
            return
        QApplication.restoreOverrideCursor()
        filled = len(getattr(report, "todo_fills", []) or [])
        QMessageBox.information(
            self, "Placeholders written",
            f"{filled} field(s) now carry TODO. Validation will report each "
            "one until it is answered.",
        )
        self.refresh()

    def _on_citation(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        path = write_citation(self._root)
        if path is None:
            return
        QMessageBox.information(
            self, "CITATION.cff written",
            "Written from dataset_description.json, and the fields the "
            "citation file now owns were removed from the description so "
            "the two cannot disagree.",
        )
        self.refresh()


def _lede(text: str) -> QLabel:
    return _wrapped(text)


def _wrapped(text: str, *, hint: bool = False) -> QLabel:
    """A word-wrapped label that is given the height its text needs.

    A wrapped QLabel reports a one-line height hint, so a vertical layout
    hands it one line and clips the rest. Minimum-vertical policy plus the
    label's own heightForWidth is the fix.
    """
    label = QLabel(text)
    label.setWordWrap(True)
    if hint:
        label.setObjectName("pane-hint")
    label.setSizePolicy(
        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding,
    )
    return label


def _section_title(text: str) -> QWidget:
    frame = QFrame()
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(0, 6, 0, 0)
    label = QLabel(text)
    label.setObjectName("dlg-section-title")
    layout.addWidget(label)
    layout.addStretch(1)
    return frame


__all__ = ["FixupsDialog"]
