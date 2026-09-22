"""Which files point at which, shown as the tree the dataset actually is.

Two rewrites got this wrong in the same way: they showed a FLAT list of
sidecars and a field name, and left the reader to work out what pointed at
what. "sub-001_run-2_phasediff.json | IntendedFor | 2 file(s)" tells you a
relationship exists and nothing about what it relates.

A reference is a relationship between two files, so it is drawn as one:

    sub-001 / ses-pre
      fmap  sub-001_acq-fm2_run-1_phasediff.nii.gz
            IntendedFor -> func/sub-001_task-x_run-1_bold.nii.gz      ok
            IntendedFor -> func/sub-001_task-x_run-2_bold.nii.gz      missing
      func  sub-001_task-x_run-1_bold.nii.gz
            used by <- fmap/sub-001_acq-fm2_run-1_phasediff.nii.gz

Both directions, because half the questions people have are the reverse one:
not "what does this fieldmap correct" but "is this run corrected by anything".
Nothing else in the tool can answer that.

Editing is where you are looking. Select any row and the panel underneath
holds that file's targets as a ticked list; there is no separate mode and no
second dialog. The suggestion button fills in what the acquisition times
imply, and says so in a sentence.

Opens on the whole dataset. A file may be passed in to start with it
selected, which is what the tree's right-click does, but nothing requires
one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..editor import linkage
from .dialog_chrome import build_footer_with, build_header, card, hint

log = logging.getLogger(__name__)

# What a row's status says, and how it is coloured. Kept short: the column
# is read at a glance and a sentence there is a sentence nobody reads.
_OK = "ok"
_MISSING = "missing"
_EXTRA = "not implied by the times"
_ABSENT = "implied, not set"
_NONE_YET = "nothing set"

_COLOURS = {
    _MISSING: "#f85149",
    _EXTRA: "#d29922",
    _ABSENT: "#d29922",
    _NONE_YET: "#8b949e",
}


class LinkageDialog(QDialog):
    """Every reference in the dataset, both ways round, and an editor."""

    def __init__(
        self,
        root: Path,
        target: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._current: Optional[Path] = None
        self._changed = 0

        self.setWindowTitle("References between files")
        self.setModal(True)
        self.resize(1120, 800)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Which files point at which",
            "A few BIDS fields hold a pointer at another file rather than a "
            "value. A fieldmap's <b>IntendedFor</b> names the runs it can "
            "correct; a derivative's <b>Sources</b> names what it was made "
            "from. Every one in this dataset is below, drawn as the "
            "relationship it is, in both directions.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        split = QSplitter(Qt.Orientation.Vertical)

        tree_card, tl = card("The dataset, and what points at what")
        tl.addWidget(hint(
            "An arrow out (→) is what this file points at. An arrow in "
            "(←) is what points at this file, which is the question "
            "nothing else here answers: whether a run has a fieldmap at all. "
            "Select any file to edit its references below."
        ))
        self._tree = QTreeWidget()
        self._tree.setObjectName("check-tree")
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["File and its references", "Field", "Status"])
        self._tree.setUniformRowHeights(True)
        self._tree.currentItemChanged.connect(self._on_row_selected)
        tl.addWidget(self._tree, 1)

        tools = QHBoxLayout()
        tools.setSpacing(8)
        for text, fn in (
            ("Expand all", self._tree.expandAll),
            ("Collapse all", self._tree.collapseAll),
        ):
            btn = QPushButton(text)
            btn.setObjectName("tb-btn")
            btn.clicked.connect(fn)
            tools.addWidget(btn)
        tools.addStretch(1)
        self._summary = QLabel("")
        self._summary.setObjectName("dlg-hint")
        tools.addWidget(self._summary)
        tl.addLayout(tools)
        split.addWidget(tree_card)

        edit_card, el = card("Edit the selected file's references")
        self._what = QLabel("Select a file above.")
        self._what.setObjectName("dlg-hint")
        self._what.setWordWrap(True)
        el.addWidget(self._what)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Field:"))
        self._field = QComboBox()
        self._field.setObjectName("ent-input")
        self._field.currentIndexChanged.connect(lambda _i: self._fill_targets())
        row.addWidget(self._field)
        row.addStretch(1)
        self._propose_btn = QPushButton("Use what the times imply")
        self._propose_btn.setObjectName("tb-btn")
        self._propose_btn.clicked.connect(self._on_propose)
        row.addWidget(self._propose_btn)
        self._clear_btn = QPushButton("Point at nothing")
        self._clear_btn.setObjectName("tb-btn")
        self._clear_btn.setToolTip(
            "Untick everything and save, which removes the field rather "
            "than writing an empty list. An empty list claims the file "
            "points at nothing, which is a different and wronger statement "
            "than not saying."
        )
        self._clear_btn.clicked.connect(lambda: self._set_all(False))
        row.addWidget(self._clear_btn)
        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("tb-btn-primary")
        self._save_btn.clicked.connect(self._on_save)
        row.addWidget(self._save_btn)
        el.addLayout(row)

        self._targets = QTreeWidget()
        self._targets.setObjectName("check-tree")
        self._targets.setColumnCount(2)
        self._targets.setHeaderLabels(["Tick what this file should point at", "State"])
        self._targets.setRootIsDecorated(False)
        self._targets.setUniformRowHeights(True)
        self._targets.itemChanged.connect(lambda *_a: self._update_status())
        el.addWidget(self._targets, 1)

        self._reason = hint("")
        el.addWidget(self._reason)
        split.addWidget(edit_card)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        bl.addWidget(split, 1)
        outer.addWidget(body, 1)

        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        self._status.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._status, buttons))

        self._set_editor_enabled(False)
        self._reload()
        if target is not None:
            self._select(linkage.sidecar_for(Path(target)))

    # -- helpers -----------------------------------------------------------

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self._root.resolve()).as_posix()
        except ValueError:
            return path.name

    def _set_editor_enabled(self, on: bool) -> None:
        for widget in (self._field, self._propose_btn, self._clear_btn,
                       self._save_btn, self._targets):
            widget.setEnabled(on)

    # -- building the tree -------------------------------------------------

    def _reload(self) -> None:
        """One branch per subject and session, one row per file that takes
        part in a reference, and one child per relationship."""
        from ..editor.rename import walk_dataset

        self._tree.clear()
        counts: dict[str, int] = {}

        # Collect first, so the reverse direction can be drawn: a file has
        # to know what points AT it, which only the whole sweep can say.
        outgoing: dict[Path, list[tuple[str, str, Optional[Path]]]] = {}
        incoming: dict[Path, list[tuple[Path, str]]] = {}
        proposals: dict[Path, set[str]] = {}

        for path in walk_dataset(self._root):
            if not path.name.endswith(".json"):
                continue
            for link in linkage.read_links(self._root, path):
                for written, resolved in zip(link.targets, link.resolved):
                    outgoing.setdefault(path, []).append(
                        (link.field, written, resolved)
                    )
                    if resolved is not None:
                        incoming.setdefault(resolved, []).append(
                            (path, link.field)
                        )
            if "IntendedFor" in linkage.fields_for(self._root, path):
                proposal = linkage.propose(self._root, path, "IntendedFor")
                if proposal:
                    proposals[path] = {
                        linkage.to_uri(self._root, p) for p in proposal.targets
                    }
                    outgoing.setdefault(path, [])

        groups: dict[str, QTreeWidgetItem] = {}

        def group_for(path: Path) -> QTreeWidgetItem:
            rel = Path(self._rel(path))
            # DIRECTORIES only. A filename starts with ``sub-`` too, so
            # including it gave every file a group of its own and the tree
            # read as a flat list with extra indentation.
            parts = [
                p for p in rel.parts[:-1] if p.startswith(("sub-", "ses-"))
            ]
            key = "/".join(parts) or "the dataset"
            node = groups.get(key)
            if node is None:
                node = QTreeWidgetItem(self._tree, [key, "", ""])
                node.setFirstColumnSpanned(True)
                node.setExpanded(True)
                groups[key] = node
            return node

        for path in sorted(set(outgoing) | set(incoming)):
            parent = QTreeWidgetItem(group_for(path), [
                self._rel(path).rsplit("/", 1)[-1], "", "",
            ])
            parent.setData(0, Qt.ItemDataRole.UserRole, str(path))
            parent.setExpanded(True)
            parent.setToolTip(0, self._rel(path))

            wanted = proposals.get(path, set())
            written = {w for _f, w, _r in outgoing.get(path, [])}

            for field, target, resolved in outgoing.get(path, []):
                if resolved is None:
                    state = _MISSING
                elif wanted and target not in wanted:
                    state = _EXTRA
                else:
                    state = _OK
                self._child(parent, f"→ {self._short(target)}", field, state)
                counts[state] = counts.get(state, 0) + 1

            for missing in sorted(wanted - written):
                self._child(
                    parent, f"→ {self._short(missing)}",
                    "IntendedFor", _ABSENT,
                )
                counts[_ABSENT] = counts.get(_ABSENT, 0) + 1

            if not outgoing.get(path) and not wanted and path not in incoming:
                self._child(parent, "→ nothing", "", _NONE_YET)

            for source, field in sorted(incoming.get(path, [])):
                self._child(
                    parent, f"← {self._rel(source).rsplit('/', 1)[-1]}",
                    f"{field} (incoming)", "",
                )

        for column in range(3):
            self._tree.resizeColumnToContents(column)

        if not groups:
            self._summary.setText(
                "No file in this dataset points at another. That is normal "
                "with no fieldmaps and no derivatives."
            )
            self._status.setText("")
            return
        self._summary.setText(
            ", ".join(f"{n} {state}" for state, n in sorted(counts.items()))
            or "nothing to report"
        )
        self._status.setText("Select a file to edit what it points at.")

    def _short(self, target: str) -> str:
        """A written target as ``datatype/filename``, which is what reads."""
        body = target.split(":", 2)[-1] if target.startswith("bids:") else target
        parts = [p for p in body.split("/") if p]
        return "/".join(parts[-2:]) if len(parts) >= 2 else body

    def _child(
        self, parent: QTreeWidgetItem, text: str, field: str, state: str,
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [text, field, state])
        colour = _COLOURS.get(state)
        if colour:
            item.setForeground(2, QBrush(QColor(colour)))
        return item

    def _select(self, path: Path) -> None:
        wanted = str(path)
        stack = [self._tree.topLevelItem(i)
                 for i in range(self._tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item is None:
                continue
            if item.data(0, Qt.ItemDataRole.UserRole) == wanted:
                self._tree.setCurrentItem(item)
                self._tree.scrollToItem(item)
                return
            stack.extend(item.child(i) for i in range(item.childCount()))

    # -- the editor --------------------------------------------------------

    def _on_row_selected(self, current, _previous) -> None:
        # A child row is a relationship; editing it means editing its parent.
        while current is not None and not current.data(0, Qt.ItemDataRole.UserRole):
            current = current.parent()
        if current is None:
            self._current = None
            self._what.setText("Select a file above.")
            self._targets.clear()
            self._field.clear()
            self._set_editor_enabled(False)
            return

        self._current = Path(current.data(0, Qt.ItemDataRole.UserRole))
        self._set_editor_enabled(True)
        self._field.blockSignals(True)
        self._field.clear()
        for name in linkage.fields_for(self._root, self._current):
            self._field.addItem(name, userData=name)
        self._field.blockSignals(False)
        self._fill_targets()

    def _current_field(self) -> str:
        return self._field.currentData() or ""

    def _fill_targets(self) -> None:
        self._targets.clear()
        if self._current is None:
            return
        field = self._current_field()
        if not field:
            self._what.setText(
                f"<b>{self._rel(self._current)}</b><br>"
                "This file carries no field that points at another."
            )
            return

        self._what.setText(
            f"<b>{self._rel(self._current)}</b><br>"
            f"<b>{field}</b>: {linkage.FIELD_HELP.get(field, '')}"
        )

        current = {
            t for link in linkage.read_links(self._root, self._current)
            if link.field == field for t in link.targets
        }
        resolved_now = {linkage.resolve(self._root, t) for t in current}

        self._targets.blockSignals(True)
        for candidate in linkage.candidates(self._root, self._current, field):
            item = QTreeWidgetItem([self._rel(candidate), ""])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                0,
                Qt.CheckState.Checked if candidate in resolved_now
                else Qt.CheckState.Unchecked,
            )
            item.setData(0, Qt.ItemDataRole.UserRole, str(candidate))
            self._targets.addTopLevelItem(item)

        for target in sorted(current):
            if linkage.resolve(self._root, target) is None:
                item = QTreeWidgetItem([target, _MISSING])
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Unchecked)
                item.setForeground(1, QBrush(QColor(_COLOURS[_MISSING])))
                self._targets.addTopLevelItem(item)

        self._targets.resizeColumnToContents(0)
        self._targets.blockSignals(False)

        proposal = linkage.propose(self._root, self._current, field)
        self._propose_btn.setEnabled(proposal is not None)
        self._reason.setText(
            f"What the acquisition times imply: {proposal.reason}"
            if proposal else
            "No rule proposes this field, so tick what it should point at."
        )
        self._update_status()

    def _rows(self) -> list[QTreeWidgetItem]:
        return [
            self._targets.topLevelItem(i)
            for i in range(self._targets.topLevelItemCount())
        ]

    def _checked(self) -> list[Path]:
        return [
            Path(r.data(0, Qt.ItemDataRole.UserRole)) for r in self._rows()
            if r.checkState(0) == Qt.CheckState.Checked
            and r.data(0, Qt.ItemDataRole.UserRole)
        ]

    def _set_all(self, on: bool) -> None:
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        self._targets.blockSignals(True)
        for row in self._rows():
            if row.data(0, Qt.ItemDataRole.UserRole):
                row.setCheckState(0, state)
        self._targets.blockSignals(False)
        self._update_status()

    def _on_propose(self) -> None:
        proposal = linkage.propose(
            self._root, self._current, self._current_field()
        )
        if proposal is None:
            return
        wanted = {str(p) for p in proposal.targets}
        self._targets.blockSignals(True)
        for row in self._rows():
            key = row.data(0, Qt.ItemDataRole.UserRole)
            if key:
                row.setCheckState(
                    0,
                    Qt.CheckState.Checked if key in wanted
                    else Qt.CheckState.Unchecked,
                )
        self._targets.blockSignals(False)
        self._update_status()

    def _update_status(self) -> None:
        if self._current is None:
            return
        n = len(self._checked())
        missing = sum(1 for r in self._rows() if r.text(1) == _MISSING)
        parts = [f"{n} file(s) ticked."]
        if missing:
            parts.append(
                f"{missing} current target(s) point at nothing and go on save."
            )
        parts.append("Nothing is written until you press Save.")
        self._status.setText(" ".join(parts))

    def _on_save(self) -> None:
        if self._current is None:
            return
        field = self._current_field()
        if not field:
            return
        sidecar, uris = linkage.plan_write(
            self._root, self._current, field, self._checked()
        )
        try:
            self._changed += linkage.apply_links(
                self._root, [(sidecar, field, uris)],
                label=f"Set {field} on {sidecar.name}",
            )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            QMessageBox.warning(
                self, "Could not save",
                f"{sidecar.name} was not changed.\n\n{exc}",
            )
            return
        keep = self._current
        self._reload()
        self._select(keep)

    def changed_count(self) -> int:
        return self._changed


__all__ = ["LinkageDialog"]
