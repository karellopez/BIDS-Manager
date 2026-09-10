"""Right pane of the Editor — validation findings, grouped.

Visual reference: ``inspector_proto/proto.py`` ``EditorView._right_pane``
(lines 942-980).

Three stacked sections, each with a title + a count chip + zero-or-more
:class:`ValMessage` rows:

1. **Dataset** — :pyattr:`ValidationReport.dataset_issues`. Findings
   that aren't tied to any single file (e.g. missing
   ``dataset_description.json`` at the root, dangling
   ``IntendedFor`` URIs).
2. **Folder** — issues for the parent folder of the file currently
   selected in the BIDS tree. Sourced from
   :pyattr:`ValidationReport.folder_issues`. Empty when no file is
   selected or the folder has no issues.
3. **File** — :pyattr:`FileVerdict.issues` for the file currently
   selected. Empty when no file is selected, the file has no
   ``FileVerdict`` (e.g. user hasn't validated yet), or the
   ``FileVerdict`` has zero issues.

Section headers stay visible even when empty (with a muted "no
issues" line) so the layout doesn't jump as the user clicks around.

Like every other Editor pane, this widget is **QSS-driven** — every
palette colour lives in ``theme.qss`` under the ``val-*`` object
names; the theme manager's global re-apply (followed by the
unpolish/polish dance from :meth:`repaint_for_palette`) handles
dark↔light swaps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QSizePolicy,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...editor.types import (
    FieldLevel,
    FileVerdict,
    Issue,
    Severity,
    SidecarField,
    ValidationReport,
)
from ...editor.grouping import FindingGroup, group_report, summarise
from .primitives import Chip, PaneHeader
from .val_message import ValMessage

log = logging.getLogger(__name__)


def _count_chip_kind(issues: list[Issue]) -> str:
    """Pick the :class:`Chip` ``kind`` for a section count chip.

    Worst severity wins — one ``err`` in a section flips its chip red.
    """
    if any(i.severity is Severity.ERR for i in issues):
        return "err"
    if any(i.severity is Severity.WARN for i in issues):
        return "warn"
    return ""  # default (neutral) chip


def _find_verdict(
    report: Optional[ValidationReport],
    root: Optional[Path],
    path: Optional[Path],
) -> Optional[FileVerdict]:
    """Same path-resolution logic as the sidecar form's lookup."""
    if report is None or root is None or path is None:
        return None
    try:
        target_abs = str(path.resolve())
    except OSError:
        target_abs = str(path)
    try:
        root_resolved = root.resolve()
    except OSError:
        root_resolved = root
    for fv in report.files:
        fp = fv.path
        candidate = fp if fp.is_absolute() else root_resolved / fp
        try:
            candidate_abs = str(candidate.resolve())
        except OSError:
            candidate_abs = str(candidate)
        if candidate_abs == target_abs:
            return fv
    return None


class _ClipRow(QFrame):
    """A row that renders at its natural size but claims no minimum width.

    A horizontal row of buttons otherwise reports its full width as the
    minimum of everything above it, and the Editor's splitter must be able to
    squeeze this pane down to nothing. Overriding the minimum rather than
    giving the buttons an Ignored size policy keeps them from being crushed
    into each other when there IS room.
    """

    def minimumSizeHint(self):  # noqa: N802 - Qt override
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint


def _folder_key_for(root: Optional[Path], path: Optional[Path]) -> Optional[str]:
    """Compute the relative-folder key the validator uses in
    :pyattr:`ValidationReport.folder_issues`.

    Returns ``None`` if we can't form a meaningful key (no root, file
    isn't under the root, etc.). For a file at
    ``<root>/sub-01/ses-01/anat/foo.json`` the key is
    ``"sub-01/ses-01/anat"``.
    """
    if root is None or path is None:
        return None
    try:
        rel = path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    parent = rel.parent
    if str(parent) in ("", "."):
        return ""
    return str(parent)


class ValidationPane(QWidget):
    """Read-only validation summary (Editor right pane).

    Emits :pyattr:`fix_requested` with ``(file_path, field_name)`` when
    the user clicks a ValMessage's fix button. For dataset / folder
    issues the file path is the currently-bound file (if any); for
    file issues it's the file the section is showing.
    """

    fix_requested = pyqtSignal(object, str)  # (Path | None, field_name)
    # A grouped finding's fix-in-all-files button. Carries the
    # FindingGroup so the host can open the candidate picker.
    fix_group_requested = pyqtSignal(object)
    # (file, rule_id, field) for a warning somebody wants to accept.
    accept_requested = pyqtSignal(object, str, str)
    # Emitted by the File section's "Highlight in editor" button: highlight
    # every shown error/warning field (JSON) or column (TSV) for this file.
    highlight_all_requested = pyqtSignal(object)  # (Path | None)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        # Keep a small floor so the user can squeeze it down when they want
        # more room for the viewer / tree (the splitter respects this).
        # The pane itself asks for very little; what used to stop it
        # narrowing was the text inside it reporting its full width as a
        # minimum. Those labels elide now.
        self.setMinimumWidth(48)

        self._report: Optional[ValidationReport] = None
        self._current_file: Optional[Path] = None
        self._current_root: Optional[Path] = None
        # Read lazily and cached per render pass, so accepting one
        # finding does not re-read the file for every other row.
        self._accepted = None

        # Per-section state: lazily replaced on every render.
        self._section_widgets: list[QWidget] = []

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Validation"))

        # Which findings the pane lists. Per-file is the original behaviour.
        # Grouped answers a different question: a rule that fires on 200 files
        # is one problem, and reading it 200 times teaches nothing the first
        # reading did not.
        self._grouped: bool = False
        mode = _ClipRow()
        mode.setObjectName("val-mode-row")
        ml = QHBoxLayout(mode)
        ml.setContentsMargins(14, 8, 14, 0)
        ml.setSpacing(8)
        self._file_mode_btn = QPushButton("This file")
        self._file_mode_btn.setObjectName("view-pill")
        self._file_mode_btn.setCheckable(True)
        self._file_mode_btn.setChecked(True)
        self._group_mode_btn = QPushButton("Whole dataset")
        self._group_mode_btn.setObjectName("view-pill")
        self._group_mode_btn.setCheckable(True)
        self._group_mode_btn.setToolTip(
            "Every finding in the dataset, collapsed to one row per kind "
            "with the number of files it fired on."
        )
        grp = QButtonGroup(mode)
        grp.setExclusive(True)
        grp.addButton(self._file_mode_btn, 0)
        grp.addButton(self._group_mode_btn, 1)
        grp.idClicked.connect(self._on_mode_clicked)
        self._mode_group = grp
        ml.addWidget(self._file_mode_btn)
        ml.addWidget(self._group_mode_btn)
        ml.addStretch(1)
        self._mode_summary = QLabel("")
        self._mode_summary.setObjectName("pane-hint")
        ml.addWidget(self._mode_summary)
        # Only the summary label gives way; the two pills keep their natural
        # size or they draw on top of each other. The row itself claims no
        # minimum (see :class:`_ClipRow`), so the splitter can still squeeze
        # this pane to nothing and the buttons simply clip.
        self._mode_summary.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred,
        )
        v.addWidget(mode)

        # Scrollable body. ``val-panel`` carries the QSS background.
        self._body = QWidget()
        self._body.setObjectName("val-panel")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(14, 12, 14, 12)
        self._body_layout.setSpacing(10)
        self._body_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # No early horizontal scrollbar - let the cards wrap/clip so the pane
        # can be squeezed narrow without a scrollbar popping in.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._body)
        # A scroll area otherwise reports its contents' minimum as its own, so
        # the pane could not be squeezed past whatever the widest card wanted.
        scroll.setMinimumWidth(0)
        scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        v.addWidget(scroll, 1)

        # Initial empty render.
        self._render()

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def set_report(self, report: Optional[ValidationReport]) -> None:
        """Bind the panel to a fresh :class:`ValidationReport`."""
        self._report = report
        self._render()

    def set_current_file(
        self,
        path: Optional[Path],
        root: Optional[Path],
    ) -> None:
        """Tell the panel which file (and root) the user is focused on.

        Drives the "folder" and "file" sections — they re-render to
        match the new context. Dataset section is unaffected.
        """
        self._current_file = path
        self._current_root = root
        self._render()

    def repaint_for_palette(self, pal: dict) -> None:
        """Same QSS-only refresh pattern as :class:`SidecarFormPane`.

        Forces Qt's unpolish/polish cycle so every descendant widget
        re-evaluates the freshly-applied global QSS — without this
        Qt's per-widget style cache holds stale colours for custom
        widgets like our ``QFrame#val-msg-*`` rows.
        """
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    @staticmethod
    def _allowed_severities() -> set:
        """Which severities to list, from the ``validate_show`` setting.

        The tree badges + chips always show the full picture; this only
        narrows the findings list so the user can focus on errors (or
        warnings). Read fresh each render so a Settings change takes effect
        on the next validation / file selection.
        """
        try:
            from ...gui.app_settings import AppSettings
            show = AppSettings.load().validate_show
        except Exception:
            show = "error_warning"
        if show == "error":
            return {Severity.ERR}
        if show == "warning":
            return {Severity.WARN}
        return {Severity.ERR, Severity.WARN}

    def _render(self) -> None:
        """Tear down + rebuild the three sections from current state."""
        self._allowed = self._allowed_severities()
        # Drop existing section widgets. ``setParent(None)`` is the
        # critical detach — otherwise the deleteLater is deferred and
        # the old sections paint over the new ones briefly.
        for w in self._section_widgets:
            self._body_layout.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self._section_widgets.clear()

        # Pre-validation empty state.
        if self._report is None:
            hint = QLabel(
                "Run “Validate dataset” to populate this panel."
            )
            hint.setObjectName("pane-hint")
            hint.setAlignment(Qt.AlignmentFlag.AlignTop)
            hint.setWordWrap(True)
            self._insert_section_widget(hint)
            return

        if self._grouped:
            self._render_grouped()
            return

        self._mode_summary.setText('')
        # Section 1: dataset issues.
        # Fix buttons on dataset issues land on the currently-selected
        # file if any (matches what the user expects when they're
        # already viewing ``dataset_description.json``).
        self._insert_section(
            "Dataset",
            self._report.dataset_issues,
            empty_text="No dataset-level issues.",
            target_file=self._current_file,
        )

        # Section 2: folder issues (parent of current file).
        folder_key = _folder_key_for(self._current_root, self._current_file)
        folder_issues = []
        folder_label = "Folder"
        if folder_key is not None:
            folder_issues = list(
                self._report.folder_issues.get(folder_key, [])
            )
            label = folder_key if folder_key else "(dataset root)"
            folder_label = f"Folder · {label}"
        self._insert_section(
            folder_label,
            folder_issues,
            empty_text=(
                "No folder-level issues."
                if self._current_file is not None
                else "Select a file to see folder-level findings."
            ),
            target_file=self._current_file,
        )

        # Section 3: file issues (FileVerdict for current file).
        verdict = _find_verdict(
            self._report, self._current_root, self._current_file
        )
        file_issues = list(verdict.issues) if verdict else []
        file_label = "File"
        if self._current_file is not None:
            file_label = f"File · {self._current_file.name}"
        # Empty-text rules:
        # - No file selected → guide the user to pick one.
        # - File selected (whether or not the validator emitted a
        #   FileVerdict for it) → "No file-level issues." Some file
        #   types (``.nii.gz`` etc.) never get a FileVerdict; showing
        #   "select a file" would be confusing.
        if self._current_file is None:
            empty_text = "Select a file in the BIDS tree to see its findings."
        else:
            empty_text = "No file-level issues."
        self._insert_section(
            file_label,
            file_issues,
            empty_text=empty_text,
            target_file=self._current_file,
            highlight_button=True,
        )

        # Section 4: schema audit (only when the current file has one).
        # The JSON validation_report carries every SidecarField — that's
        # info the form pane uses but the user can't easily eyeball.
        # Surface a compact summary here.
        if verdict is not None and verdict.sidecar_fields:
            self._insert_schema_audit_section(verdict)

    def _insert_section(
        self,
        title: str,
        issues: list[Issue],
        *,
        empty_text: str,
        target_file: Optional[Path] = None,
        highlight_button: bool = False,
    ) -> None:
        # Apply the "Show findings" severity filter (errors only / warnings
        # only / both). The count chip + empty state reflect the filtered list.
        allowed = getattr(self, "_allowed", {Severity.ERR, Severity.WARN})
        issues = [i for i in issues if i.severity in allowed]

        section = QFrame()
        section.setObjectName("val-section")
        sl = QVBoxLayout(section)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # Header row: title + (optional) highlight button + count chip.
        head = QHBoxLayout()
        head.setSpacing(10)
        head.setContentsMargins(0, 0, 0, 0)
        title_l = QLabel(title)
        title_l.setObjectName("val-section-title")
        head.addWidget(title_l)
        head.addStretch(1)
        # "Highlight in editor" — only on the File section, only when there are
        # field/column findings to point at.
        if highlight_button and target_file is not None and any(i.field for i in issues):
            hl_btn = QPushButton("Highlight in editor")
            # Distinct object name so it is not mistaken for a per-finding fix
            # button (it carries the same styling token plus its own).
            hl_btn.setObjectName("val-highlight-all")
            hl_btn.setToolTip(
                "Highlight every shown error / warning field (JSON) or column "
                "(TSV) for this file in the editor."
            )
            hl_btn.clicked.connect(
                lambda _=False, p=target_file: self.highlight_all_requested.emit(p)
            )
            head.addWidget(hl_btn)
        count_l = Chip(str(len(issues)), _count_chip_kind(issues))
        head.addWidget(count_l)
        sl.addLayout(head)

        # Messages (or an empty-state hint).
        if not issues:
            empty = QLabel(empty_text)
            empty.setObjectName("pane-hint")
            empty.setWordWrap(True)
            sl.addWidget(empty)
        else:
            for issue in issues:
                msg = ValMessage(
                    severity=(
                        issue.severity.value
                        if isinstance(issue.severity, Severity)
                        else str(issue.severity)
                    ),
                    rule=issue.rule_id,
                    body_html=issue.message,
                    fix_label=issue.fix_label,
                    field=issue.field,
                    schema_rule=issue.schema_rule,
                )
                # Re-emit fix clicks with the file context so the host
                # panel can jump to the right place.
                msg.fix_requested.connect(
                    lambda field, p=target_file:
                        self.fix_requested.emit(p, field)
                )
                msg.accept_requested.connect(
                    lambda rule, field, p=target_file:
                        self.accept_requested.emit(p, rule, field)
                )
                decision = self._is_accepted(target_file, issue)
                if decision is not None:
                    # Shown, not hidden: a decision somebody made is part of
                    # the record, and hiding it would make the next reviewer
                    # rediscover the same finding.
                    msg.setEnabled(False)
                    msg.setToolTip(
                        "Accepted by {who} on {at}.\n\n{note}".format(
                            who=decision.who or "somebody",
                            at=decision.at.replace("T", " "),
                            note=decision.note or "No reason was given.",
                        )
                    )
                sl.addWidget(msg)

        self._insert_section_widget(section)

    # ------------------------------------------------------------------
    # Grouped (whole-dataset) mode
    # ------------------------------------------------------------------

    def _on_mode_clicked(self, idx: int) -> None:
        grouped = idx == 1
        if grouped == self._grouped:
            return
        self._grouped = grouped
        self._render()

    def _render_grouped(self) -> None:
        """One row per kind of finding, with the file count beside it."""
        groups = group_report(self._report, allowed=self._allowed)
        self._mode_summary.setText(summarise(groups))
        if not groups:
            hint = QLabel("No findings at the current severity filter.")
            hint.setObjectName("pane-hint")
            hint.setWordWrap(True)
            self._insert_section_widget(hint)
            return
        for grp in groups:
            self._insert_section_widget(self._build_group_row(grp))

    def _build_group_row(self, grp: FindingGroup) -> QWidget:
        """A finding, its count, the files under it, and a fix-all action."""
        sev = (
            grp.severity.value
            if isinstance(grp.severity, Severity) else str(grp.severity)
        )
        card = QFrame()
        card.setObjectName("val-section")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(10)
        head.setContentsMargins(0, 0, 0, 0)
        title = QLabel(grp.title())
        title.setObjectName("val-section-title")
        title.setToolTip(grp.message)
        # A rule id plus a field name is long, and the count and the fix
        # button are the parts that must survive a narrow pane. Ignored
        # horizontal policy lets the label give way rather than push them out.
        title.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred,
        )
        head.addWidget(title, 1)
        if grp.field:
            fix = QPushButton("Fix in all files")
            fix.setObjectName("val-highlight-all")
            fix.setToolTip(
                "Review the {n} files this fired on and write a value into "
                "the ones you tick.".format(n=grp.count)
            )
            fix.clicked.connect(
                lambda _=False, g=grp: self.fix_group_requested.emit(g)
            )
            head.addWidget(fix)
        head.addWidget(Chip(
            "{n} file{s}".format(n=grp.count, s="" if grp.count == 1 else "s"),
            "err" if sev == "error" else "warn",
        ))
        cl.addLayout(head)

        msg = ValMessage(
            severity=sev,
            rule=grp.rule_id,
            body_html=grp.message,
            fix_label=None,
            field=grp.field,
            schema_rule=grp.schema_rule,
        )
        cl.addWidget(msg)

        # The files, listed rather than summarised: the point of grouping is
        # to stop repeating the message, not to hide where it landed.
        shown = [str(x) for x in grp.files[:8]]
        extra = len(grp.files) - len(shown)
        text = "\n".join(shown)
        if extra > 0:
            text += "\nand {n} more".format(n=extra)
        listing = QLabel(text)
        listing.setObjectName("pane-hint")
        listing.setWordWrap(True)
        listing.setToolTip("\n".join(str(x) for x in grp.files))
        cl.addWidget(listing)
        return card

    def _is_accepted(self, target_file, issue):
        """The decision covering this finding, or ``None``."""
        if target_file is None or self._current_root is None:
            return None
        if issue.severity is not Severity.WARN:
            return None
        from ...editor.review import is_accepted, load

        if self._accepted is None:
            self._accepted = load(self._current_root)
        try:
            rel = str(
                Path(target_file).resolve().relative_to(
                    Path(self._current_root).resolve()
                )
            )
        except (ValueError, OSError):
            return None
        return is_accepted(
            self._accepted, file=rel, rule_id=issue.rule_id,
            field=issue.field or "",
        )

    def reload_acceptances(self) -> None:
        """Forget the cached decisions; the next render re-reads them."""
        self._accepted = None
        self._render()

    def _insert_section_widget(self, widget: QWidget) -> None:
        """Insert ``widget`` before the trailing stretch and remember it."""
        insert_idx = self._body_layout.count() - 1
        self._body_layout.insertWidget(insert_idx, widget)
        self._section_widgets.append(widget)

    def _insert_schema_audit_section(
        self, verdict: FileVerdict,
    ) -> None:
        """Render a compact schema-audit summary for the current file.

        Counts per level + the names of any missing required /
        recommended fields. Optional / deprecated counts are shown but
        their member lists are folded — they're noise for daily review.
        """
        section = QFrame()
        section.setObjectName("val-section")
        sl = QVBoxLayout(section)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # Header row.
        head = QHBoxLayout()
        head.setSpacing(10)
        head.setContentsMargins(0, 0, 0, 0)
        title_l = QLabel("Schema audit")
        title_l.setObjectName("val-section-title")
        head.addWidget(title_l)
        head.addStretch(1)

        # Per-level breakdown.
        by_level: dict[FieldLevel, list[SidecarField]] = {
            FieldLevel.REQUIRED:    [],
            FieldLevel.RECOMMENDED: [],
            FieldLevel.OPTIONAL:    [],
            FieldLevel.DEPRECATED:  [],
        }
        for f in verdict.sidecar_fields:
            by_level.setdefault(f.level, []).append(f)
        missing_req = [f for f in by_level[FieldLevel.REQUIRED] if not f.present]
        missing_rec = [f for f in by_level[FieldLevel.RECOMMENDED] if not f.present]

        # Header chip — severity reflects the worst missing level.
        total_fields = sum(len(v) for v in by_level.values())
        if missing_req:
            chip_kind = "err"
            chip_text = f"{len(missing_req)} missing required"
        elif missing_rec:
            chip_kind = "warn"
            chip_text = f"{len(missing_rec)} missing recommended"
        else:
            chip_kind = ""
            chip_text = f"{total_fields} fields"
        chip = Chip(chip_text, chip_kind)
        head.addWidget(chip)
        sl.addLayout(head)

        # Per-level lines.
        for level, fields in by_level.items():
            if not fields:
                continue
            present = sum(1 for f in fields if f.present)
            row = self._build_audit_row(level, fields, present)
            sl.addWidget(row)

        self._insert_section_widget(section)

    def _build_audit_row(
        self,
        level: FieldLevel,
        fields: list[SidecarField],
        present: int,
    ) -> QFrame:
        row = QFrame()
        row.setObjectName("val-audit-row")
        rl = QVBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        total = len(fields)
        missing = [f for f in fields if not f.present]
        summary = QLabel(
            f"{level.value.capitalize()}: {present}/{total} present"
        )
        summary.setObjectName("val-audit-summary")
        rl.addWidget(summary)
        # Only required / recommended get a missing-list expanded inline
        # — optional / deprecated are noisy. Long lists are truncated.
        if missing and level in (FieldLevel.REQUIRED, FieldLevel.RECOMMENDED):
            names = ", ".join(f.name for f in missing[:8])
            extra = "" if len(missing) <= 8 else f", … (+{len(missing) - 8})"
            miss_lbl = QLabel(f"missing: {names}{extra}")
            miss_lbl.setObjectName("val-audit-missing")
            miss_lbl.setWordWrap(True)
            rl.addWidget(miss_lbl)
        return row


__all__ = ["ValidationPane"]
