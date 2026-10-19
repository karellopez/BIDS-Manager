"""Anonymise the open dataset's metadata, after showing every change.

Over :mod:`bidsmgr.anonymise`. A preset (Recommended or Strict), what it
does in words, and then the plan: each file and what changes in it (before,
after, why), what needs a person to read it (free text, labels), and what a
format does not let BIDS Manager rewrite. Nothing is written until Anonymise
is pressed; the change is one step of the Editor's history.

The plan and the writing both run on a thread: planning reads every sidecar
and table, and writing a FIF goes through MNE, which takes seconds per
recording.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..anonymise import PRESET_LABELS, PRESETS, RECOMMENDED, apply_anonymise, plan_anonymise
from .dialog_chrome import build_footer_with, build_header, card, hint
from .widgets.spinner import BusySpinner
from .widgets.splitter import Splitter
from .widgets.tree_click import toggle_on_click

log = logging.getLogger(__name__)

#: Above this many files the plan opens folded, so it stays readable.
_UNFOLDED = 15


def _apply_job(root: Path, plan, label: str, forget: bool) -> tuple[int, int, list]:
    """On the thread: write the plan as one operation; then, if asked,
    forget the history's copies. ``(files written, bytes freed, failed)``."""
    from ..project.operations import begin_operation, forget_history

    with begin_operation(root, label) as op:
        written = apply_anonymise(plan, op=op)
    freed = forget_history(root) if forget else 0
    return len(written), freed, list(plan.failed)


class AnonymiseDialog(QDialog):
    """Choose a preset, read the plan, anonymise."""

    def __init__(self, root: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._plan = None
        self._written = 0
        self._generation = 0
        self._busy_state = False
        self._released = None
        from .viz.bridge import JobRunner

        self._jobs = JobRunner(self)
        self._jobs.done.connect(self._on_done)
        self._jobs.failed.connect(self._on_failed)

        self.setWindowTitle("Anonymise metadata")
        self.setModal(True)
        self.resize(1000, 800)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Anonymise metadata",
            "Removes what identifies a participant from the dataset's metadata, in "
            "every kind of file: sidecars, tables, recording headers, photographs. "
            "Dates are <b>shifted</b>, not removed, by the same number of whole weeks "
            "for all of one participant's dates, so intervals and weekdays survive. "
            "The recorded data are never changed. <b>Nothing is written until you "
            "press Anonymise</b>; it is one step of the Editor's history.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        how, hl = card("How")
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(QLabel("Preset:"))
        self._preset = QComboBox()
        for value in PRESETS:
            self._preset.addItem(PRESET_LABELS[value], userData=value)
        self._preset.setCurrentIndex(max(0, self._preset.findData(RECOMMENDED)))
        self._preset.currentIndexChanged.connect(lambda _i: self._replan())
        row.addWidget(self._preset)
        row.addStretch(1)
        hl.addLayout(row)
        self._what = hint("")
        hl.addWidget(self._what)
        bl.addWidget(how)

        split = Splitter(Qt.Orientation.Vertical)
        plan_card, pl = card("What would change")
        busy = QHBoxLayout()
        self._spinner = BusySpinner()
        busy.addWidget(self._spinner)
        self._busy_label = QLabel("")
        self._busy_label.setObjectName("dlg-hint")
        busy.addWidget(self._busy_label, 1)
        self._busy = QWidget()
        self._busy.setLayout(busy)
        pl.addWidget(self._busy)
        self._tree = QTreeWidget()
        self._tree.setObjectName("check-tree")
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["File, and what changes", "Becomes", "Why"])
        self._tree.setUniformRowHeights(True)
        self._tree.itemChanged.connect(lambda *_a: self._update_status())
        toggle_on_click(self._tree)
        pl.addWidget(self._tree, 1)
        split.addWidget(plan_card)

        other, ol = card("Not changed")
        self._other = QTreeWidget()
        self._other.setObjectName("check-tree")
        self._other.setColumnCount(2)
        self._other.setHeaderLabels(["What", "Why"])
        self._other.setUniformRowHeights(True)
        toggle_on_click(self._other)
        ol.addWidget(self._other, 1)
        split.addWidget(other)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        bl.addWidget(split, 1)

        self._forget = QCheckBox(
            "Then delete the copies the Editor keeps for Undo (they hold the originals)")
        self._forget.setToolTip(
            "To undo, the Editor keeps a copy of every file it changes under "
            ".bidsmgr/. After anonymising, those copies are the identifiable "
            "originals: a dataset shared or backed up with .bidsmgr/ in it still "
            "holds them.\n\nTicked, they are deleted once the anonymisation is "
            "written. Nothing done before can be undone afterwards.")
        bl.addWidget(self._forget)
        outer.addWidget(body, 1)

        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        self._status.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.Ok)
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setObjectName("tb-btn-primary")
        self._ok.setText("Anonymise")
        buttons.accepted.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._status, buttons))

        self._replan()

    # -- planning ---------------------------------------------------------

    def busy(self) -> bool:
        return self._busy_state

    def _policy(self):
        return PRESETS[self._preset.currentData() or RECOMMENDED]

    def _set_busy(self, on: bool, text: str = "") -> None:
        self._busy_state = on
        self._busy.setVisible(on)
        self._spinner.set_busy(on, message="")
        self._busy_label.setText(text)
        for w in (self._tree, self._preset, self._forget):
            w.setEnabled(not on)
        self._ok.setEnabled(not on and bool(self._plan and self._plan.files))

    def _replan(self) -> None:
        policy = self._policy()
        # Two columns: eight lines down one would leave the plan four rows.
        lines = [f"&bull; {line}." for line in policy.describe()]
        half = (len(lines) + 1) // 2
        cells = "".join(
            f"<tr><td style='padding-right:18px'>{a}</td><td>{b}</td></tr>"
            for a, b in zip(lines[:half], lines[half:] + [""]))
        self._what.setText(f"<table cellspacing='0' cellpadding='1'>{cells}</table>")
        self._generation += 1
        self._plan = None
        self._tree.clear()
        self._other.clear()
        self._set_busy(True, "Reading the dataset…")
        self._status.setText("")
        self._jobs.start("plan", self._generation, plan_anonymise, self._root, policy)

    def _on_done(self, tag: str, generation: int, result) -> None:
        if generation != self._generation:
            return
        if tag == "plan":
            self._plan = result
            self._set_busy(False)
            self._fill()
        else:
            self._finish_apply(result)

    def _on_failed(self, tag: str, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        if tag == "apply" and self._released is not None:
            self._released.__exit__(None, None, None)
            self._released = None
        self._set_busy(False)
        log.warning("anonymisation %s failed: %s", tag, message)
        last = message.strip().splitlines()[-1] if message.strip() else message
        if tag == "apply":
            QMessageBox.warning(self, "Anonymisation failed",
                                f"Nothing was changed.\n\n{last}")
        self._status.setText(f"Could not {'read' if tag == 'plan' else 'write'}: {last}")

    def _fill(self) -> None:
        plan = self._plan
        self._tree.blockSignals(True)
        self._tree.clear()
        unfold = len(plan.files) <= _UNFOLDED
        for planned in plan.files:
            head = QTreeWidgetItem([plan.rel(planned.path),
                                    f"{len(planned.changes)} change"
                                    + ("s" if len(planned.changes) != 1 else ""), ""])
            head.setData(0, Qt.ItemDataRole.UserRole, plan.rel(planned.path))
            head.setFlags(head.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            head.setCheckState(0, Qt.CheckState.Checked)
            head.setToolTip(0, str(planned.path))
            for change in planned.changes:
                child = QTreeWidgetItem(head, [f"{change.what}: {change.before}",
                                               change.after or "(removed)", change.reason])
                child.setToolTip(0, f"{change.what}: {change.before}")
                child.setToolTip(1, change.after or "(removed)")
            self._tree.addTopLevelItem(head)
            head.setExpanded(unfold)
        # The path column fits its names up to half the width; what a value
        # becomes gets a third, so a shifted date is read whole.
        from .viz import fonts

        width = max(self._tree.viewport().width(), fonts.px(700))
        self._tree.resizeColumnToContents(0)
        self._tree.setColumnWidth(0, min(self._tree.columnWidth(0), width // 2))
        self._tree.setColumnWidth(1, width // 3)
        self._tree.blockSignals(False)

        self._other.clear()
        grouped = plan.grouped_reviews()
        if grouped:
            review = QTreeWidgetItem([f"To read by hand ({len(grouped)})",
                                      "not changed: only a person can judge it"])
            for what, value, why, paths in grouped:
                where = (plan.rel(paths[0]) if len(paths) == 1
                         else f"{len(paths)} files")
                item = QTreeWidgetItem(review, [f"{what} = {value!r}", f"{why} ({where})"])
                item.setToolTip(0, "\n".join(plan.rel(p) for p in paths[:20])
                                + (f"\nand {len(paths) - 20} more" if len(paths) > 20 else ""))
            self._other.addTopLevelItem(review)
            review.setExpanded(len(grouped) <= _UNFOLDED)
        if plan.cannot:
            cannot = QTreeWidgetItem([f"Cannot be changed here ({len(plan.cannot)})", ""])
            for path, why in plan.cannot:
                QTreeWidgetItem(cannot, [path, why])
            self._other.addTopLevelItem(cannot)
            cannot.setExpanded(len(plan.cannot) <= _UNFOLDED)
        if not plan.reviews and not plan.cannot:
            QTreeWidgetItem(self._other, ["Nothing: every file can be cleaned here", ""])
        self._other.resizeColumnToContents(0)
        self._update_status()

    # -- choosing and applying ----------------------------------------------

    def _chosen(self) -> set[str]:
        return {self._tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
                for i in range(self._tree.topLevelItemCount())
                if self._tree.topLevelItem(i).checkState(0) == Qt.CheckState.Checked}

    def _update_status(self) -> None:
        if self._plan is None:
            return
        if not self._plan.files:
            self._status.setText(self._plan.summary())
            self._ok.setEnabled(False)
            self._ok.setText("Anonymise")
            return
        chosen = self._chosen()
        changes = sum(len(f.changes) for f in self._plan.files
                      if self._plan.rel(f.path) in chosen)
        self._status.setText(f"{len(chosen):,} of {len(self._plan.files):,} files ticked, "
                             f"{changes:,} changes.")
        self._ok.setEnabled(bool(chosen))
        self._ok.setText(f"Anonymise {len(chosen):,} file{'s' if len(chosen) != 1 else ''}")

    def _on_apply(self) -> None:
        if self._plan is None or not self._plan.files:
            return
        chosen = self._chosen()
        forget = self._forget.isChecked()
        text = (f"Anonymise {len(chosen):,} file(s) with the {self._preset.currentText()} "
                "preset?\n\nIt is one step of the Editor's history: Undo takes it back.")
        if forget:
            text += ("\n\nThe copies kept for Undo are then DELETED: nothing done "
                     "before, this included, can be undone afterwards.")
        answer = QMessageBox.question(
            self, "Anonymise metadata", text,
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Cancel if forget else QMessageBox.StandardButton.Ok)
        if answer != QMessageBox.StandardButton.Ok:
            return
        plan = self._plan
        plan.files = [f for f in plan.files if plan.rel(f.path) in chosen]
        from .fs_watch import watchers_released

        # Windows cannot replace a watched file: the watches go for the write.
        self._released = watchers_released()
        self._released.__enter__()
        self._generation += 1
        self._set_busy(True, "Anonymising…")
        label = f"Anonymise metadata ({self._preset.currentText()})"
        self._jobs.start("apply", self._generation, _apply_job, self._root, plan, label, forget)

    def _finish_apply(self, result) -> None:
        if self._released is not None:
            self._released.__exit__(None, None, None)
            self._released = None
        self._busy_state = False
        self._written, _freed, failed = result
        if failed:
            listed = "\n".join(f"{p}: {why}" for p, why in failed[:10])
            more = f"\n... and {len(failed) - 10} more" if len(failed) > 10 else ""
            QMessageBox.warning(
                self, "Some files were not anonymised",
                f"{len(failed)} file(s) could not be rewritten and were left as they "
                f"were; the rest is anonymised.\n\n{listed}{more}")
        self.accept()

    def written(self) -> int:
        """How many files were anonymised, once accepted."""
        return self._written


__all__ = ["AnonymiseDialog"]
