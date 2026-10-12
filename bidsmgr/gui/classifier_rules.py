"""Settings > Classification: every threshold the classifier tests (generated
from ``classifier/acquisition/config.py``) and the evidence rules.

The rules are the shipped ``rulebook.json`` (which feature counts for which
name, and how much) with the user's changes laid over it. They are edited
as a file: exported here, changed in any text editor, loaded back. Only
what differs from the shipped rules is kept (``ClassifierConfig.rulebook``),
so a new release's rules still reach every name the user did not touch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..classifier.acquisition import engine
from .viz.settings_pages import GeneratedSettingsPage


class ClassifierSettingsPage(GeneratedSettingsPage):
    """The generated groups, then the evidence rules."""

    def __init__(self, sections: tuple[str, ...], *, hint: str = "", parent=None,
                 root=None) -> None:
        self._delta: dict = {}
        self.rules_state: Optional[QLabel] = None
        self.reset_button: Optional[QPushButton] = None
        super().__init__(sections, hint=hint, parent=parent, root=root)

    def extend(self, lay: QVBoxLayout) -> None:
        box = QGroupBox("Evidence rules")
        col = QVBoxLayout(box)
        intro = QLabel(
            "Which acquisition parameter or word in the series name counts for which "
            "name, and how much. Export the rules, change a weight or add a piece of "
            "evidence in any text editor, and load the file back. Only what differs "
            "from the rules shipped with BIDS Manager is kept.")
        intro.setObjectName("dlg-hint")
        intro.setWordWrap(True)
        col.addWidget(intro)
        self.rules_state = QLabel()
        self.rules_state.setObjectName("rules-state")
        self.rules_state.setWordWrap(True)
        col.addWidget(self.rules_state)
        row = QHBoxLayout()
        row.addStretch(1)
        self.export_button = QPushButton("Export the rules...")
        self.export_button.setObjectName("tb-btn")
        self.export_button.setToolTip("Write the rules in use (shipped and yours) to a JSON "
                                      "file to read or edit")
        self.export_button.clicked.connect(self._on_export)
        self.load_button = QPushButton("Load rules...")
        self.load_button.setObjectName("tb-btn")
        self.load_button.setToolTip("Use the rules in a JSON file (a whole exported "
                                    "rulebook, or only the names you changed). Saved with "
                                    "Save")
        self.load_button.clicked.connect(self._on_load)
        self.reset_button = QPushButton("Use the shipped rules")
        self.reset_button.setObjectName("tb-btn")
        self.reset_button.setToolTip("Forget your changes to the rules (saved with Save)")
        self.reset_button.clicked.connect(lambda: self.set_rule_changes({}))
        for button in (self.export_button, self.load_button, self.reset_button):
            row.addWidget(button)
        col.addLayout(row)
        lay.addWidget(box)
        self._show_state()

    # -- the changes ---------------------------------------------------------

    def rule_changes(self) -> dict:
        return dict(self._delta)

    def set_rule_changes(self, delta: dict) -> None:
        self._delta = dict(delta or {})
        self._show_state()

    def _show_state(self) -> None:
        if self.rules_state is None:
            return
        book = engine.rulebook(self._delta)
        names = len(book["candidates"])
        lines = sum(len(c.get("evidence", [])) for c in book["candidates"].values())
        changed = self._delta.get("candidates") or {}
        removed = sum(1 for v in changed.values() if v is None)
        altered = len(changed) - removed
        if not changed:
            mine = "As shipped."
        else:
            parts = []
            if altered:
                parts.append(f"{altered} name{'s' if altered != 1 else ''} changed or added")
            if removed:
                parts.append(f"{removed} removed")
            mine = "Your changes: " + ", ".join(parts) + "."
        self.rules_state.setText(f"{names} names, {lines} pieces of evidence. {mine}")
        if self.reset_button is not None:
            self.reset_button.setEnabled(bool(changed))

    # -- the page's settings ---------------------------------------------------

    def load(self, settings) -> None:
        super().load(settings)
        self.set_rule_changes(getattr(settings, "rulebook", {}) or {})

    def apply_to(self, settings) -> None:
        super().apply_to(settings)
        settings.rulebook = self.rule_changes()

    # -- files -------------------------------------------------------------------

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export the evidence rules", str(Path.home() / "bidsmgr_rules.json"),
            "JSON (*.json)")
        if path:
            self.export_rules(Path(path))

    def export_rules(self, path: Path) -> None:
        """The rules in use, whole, as a file to read or edit."""
        Path(path).write_text(json.dumps(engine.rulebook(self._delta), indent=2,
                                         ensure_ascii=False) + "\n", encoding="utf-8")

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load evidence rules", str(Path.home()), "JSON (*.json)")
        if not path:
            return
        problems = self.load_rules(Path(path))
        if problems:
            shown = "\n".join(f"- {p}" for p in problems[:8])
            more = f"\n... and {len(problems) - 8} more" if len(problems) > 8 else ""
            QMessageBox.warning(self, "Rules not loaded",
                                f"{Path(path).name} cannot be used:\n\n{shown}{more}")

    def load_rules(self, path: Path) -> list[str]:
        """Take the rules in ``path`` (a whole rulebook, as exported, or only
        the names that change); returns what is wrong with them, and loads
        nothing then."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return [f"it is not readable JSON ({exc})"]
        if not isinstance(data, dict):
            return ["it is not a JSON object"]
        full = "about" in data
        delta = engine.rulebook_delta(data, full=full)
        problems = engine.rulebook_problems(engine.rulebook(delta))
        if not problems:
            self.set_rule_changes(delta)
        return problems


__all__ = ["ClassifierSettingsPage"]
