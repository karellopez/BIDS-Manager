"""Post-conversion BIDS editor view (M6).

Visual reference: ``inspector_proto/proto.py`` ``EditorView`` (3-pane
horizontal splitter — BIDS tree | sidecar form | validation panel).

Through Step 3 the panel hosts:

* a toolbar with ``Open BIDS root…``, ``Validate dataset``, status
  chips, a busy spinner, and (disabled) file / folder validate stubs;
* a :class:`PathBar` showing the active dataset;
* the 3-pane horizontal splitter with the
  :class:`BidsTreePane` on the left (badges stamped from the in-memory
  :class:`bidsmgr.editor.types.ValidationReport`).

Center and right panes remain placeholders for Steps 4-6.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..editor.types import FileVerdict, Severity, ValidationReport
from ..workers import FileReportWorker, FolderReportWorker, ReportWorker
from . import icons
from .widgets import (
    BidsTreePane,
    BusySpinner,
    Chip,
    NiftiViewerPane,
    PanelFrame,
    PathBar,
    RecordingViewerPane,
    SidecarFormPane,
    TsvViewerPane,
    ValidationPane,
    VSep,
    is_recording_path,
)

log = logging.getLogger(__name__)


class EditorPanel(QWidget):
    """Post-convert BIDS editor (Editor pill in the top header).

    Emits :pyattr:`log_message` for every progress line so
    :class:`MainWindow` can mirror Editor activity into the status bar
    just like it already does for the Converter.
    """

    log_message = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("editor-panel")

        self._report: Optional[ValidationReport] = None
        self._report_worker: Optional[ReportWorker] = None
        # Partial-validate workers (toolbar's Validate file / folder
        # buttons). Kept on self so they're not garbage-collected
        # mid-flight.
        self._partial_worker: Optional[QObject] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self._toolbar = self._build_toolbar()
        v.addWidget(self._toolbar)

        self._path_bar = PathBar("Dataset", "(none)", ok=False)
        self._path_bar.change_button.clicked.connect(self._on_change_root)
        v.addWidget(self._path_bar)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(1)
        self._splitter.setChildrenCollapsible(False)

        self._tree_pane = BidsTreePane()
        self._tree_pane.file_selected.connect(self._on_file_selected)
        # Drive the Validate file/folder button enable-state from the
        # tree selection — file → file button, folder → folder button.
        self._tree_pane.file_selected.connect(
            self._sync_validate_buttons_from_selection
        )
        # Center pane is itself a stack — different viewer per file
        # kind. Index 0 = JSON sidecar (default for unsupported kinds
        # too — its empty-state hint guides the user to the JSON peer).
        # Index 1 = TSV table. Index 2 = NIfTI 2-D slice viewer.
        self._sidecar_form = SidecarFormPane()
        self._tsv_viewer = TsvViewerPane()
        self._nifti_viewer = NiftiViewerPane()
        self._recording_viewer = RecordingViewerPane()
        self._center_stack = QStackedWidget()
        self._center_stack.addWidget(self._sidecar_form)
        self._center_stack.addWidget(self._tsv_viewer)
        self._center_stack.addWidget(self._nifti_viewer)
        self._center_stack.addWidget(self._recording_viewer)
        # Threaded panes drive the toolbar busy spinner + status bar.
        self._tsv_viewer.loading_changed.connect(self._on_pane_loading)
        self._recording_viewer.loading_changed.connect(self._on_pane_loading)
        self._recording_viewer.status_message.connect(self.log_message)
        # Undo/redo toolbar buttons follow the active editable pane.
        self._sidecar_form.history_changed.connect(self._sync_undo_redo)
        self._tsv_viewer.history_changed.connect(self._sync_undo_redo)
        self._center_stack.currentChanged.connect(self._sync_undo_redo)
        self._validation_pane = ValidationPane()
        self._validation_pane.fix_requested.connect(self._on_fix_requested)
        self._validation_pane.highlight_all_requested.connect(
            self._on_highlight_all_requested
        )
        # The BIDS tree folds to the left, the validation pane to the right,
        # and the center viewer grows into the freed space. The viewer is
        # not collapsible (it's the main surface) but IS detachable so the
        # JSON / TSV / NIfTI view can pop out into its own window.
        self._tree_frame = PanelFrame(self._tree_pane, "BIDS tree", edge="left")
        self._center_frame = PanelFrame(
            self._center_stack, "Viewer", collapsible=False, detachable=True,
            hide_inner_header=False,
        )
        self._validation_frame = PanelFrame(
            self._validation_pane, "Validation", edge="right",
        )
        self._panel_frames = [
            self._tree_frame, self._center_frame, self._validation_frame,
        ]
        self._splitter.addWidget(self._tree_frame)
        self._splitter.addWidget(self._center_frame)
        self._splitter.addWidget(self._validation_frame)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setSizes([320, 700, 380])
        # Collapsing the tree / validation hands width to the center viewer.
        self._tree_frame.attach_splitter(self._splitter, grow_target=self._center_frame)
        self._validation_frame.attach_splitter(self._splitter, grow_target=self._center_frame)
        v.addWidget(self._splitter, 1)

        # Restore last-opened root if available.
        from .app_settings import AppSettings
        settings = AppSettings.load()
        if settings.editor_bids_root:
            root = Path(settings.editor_bids_root)
            if root.exists() and root.is_dir():
                self._set_root(root, persist=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tree_pane(self) -> BidsTreePane:
        return self._tree_pane

    def current_root(self) -> Optional[Path]:
        return self._tree_pane.root()

    def current_report(self) -> Optional[ValidationReport]:
        return self._report

    def start_dataset_validation(self) -> None:
        """Kick a :class:`ReportWorker` against the current root.

        Public so tests (and a future keyboard shortcut) can trigger
        the same flow without synthesising a button click.
        """
        root = self.current_root()
        if root is None:
            return
        if self._report_worker is not None and self._report_worker.isRunning():
            return  # debounce concurrent runs
        self._set_busy(True, "Validating dataset…")
        schema, max_rows, flag_todos = self._validation_engine_opts()
        worker = ReportWorker(
            root, strict=self._strict_btn.isChecked(),
            schema=schema, max_rows=max_rows, flag_todos=flag_todos, parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_with_report.connect(self._on_report_ready)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(self._on_worker_finished)
        self._report_worker = worker
        worker.start()

    def start_file_validation(self) -> None:
        """Re-validate the currently-selected file (layer 1 only)."""
        root = self.current_root()
        if root is None:
            return
        target = self._tree_pane_current_path()
        if target is None or not target.is_file():
            return
        if self._partial_worker is not None and getattr(
            self._partial_worker, "isRunning", lambda: False,
        )():
            return
        self._set_busy(True, f"Validating {target.name}…")
        schema, max_rows, flag_todos = self._validation_engine_opts()
        worker = FileReportWorker(
            root, target, schema=schema, max_rows=max_rows,
            flag_todos=flag_todos, parent=self,
        )
        self._wire_partial_worker(worker)
        self._partial_worker = worker
        worker.start()

    def start_folder_validation(self) -> None:
        """Re-validate the currently-selected folder (layer 1 only)."""
        root = self.current_root()
        if root is None:
            return
        target = self._tree_pane_current_path()
        if target is None or not target.is_dir():
            return
        if self._partial_worker is not None and getattr(
            self._partial_worker, "isRunning", lambda: False,
        )():
            return
        self._set_busy(True, f"Validating {target.name}/…")
        schema, max_rows, flag_todos = self._validation_engine_opts()
        worker = FolderReportWorker(
            root, target, schema=schema, max_rows=max_rows,
            flag_todos=flag_todos, parent=self,
        )
        self._wire_partial_worker(worker)
        self._partial_worker = worker
        worker.start()

    def _wire_partial_worker(self, worker) -> None:
        worker.progress.connect(self._on_progress)
        worker.finished_with_verdicts.connect(self._on_partial_ready)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(self._on_partial_worker_finished)

    # ------------------------------------------------------------------
    # Toolbar / panes
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("toolbar")
        bar.setFixedHeight(44)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 6, 14, 6)
        lay.setSpacing(8)

        self._open_btn = QPushButton("  Open BIDS root…")
        self._open_btn.setObjectName("tb-btn")
        icons.apply_button(self._open_btn, "open_folder")
        self._open_btn.clicked.connect(self._on_change_root)
        lay.addWidget(self._open_btn)

        lay.addWidget(VSep())

        # Undo / Redo for in-memory edits to the active center pane (JSON
        # sidecar + TSV table). They delegate to whichever editable pane is
        # showing; disabled when nothing is undoable (or a NIfTI is shown).
        self._undo_btn = QPushButton("  Undo")
        self._undo_btn.setObjectName("tb-btn-ghost")
        icons.apply_button(self._undo_btn, "undo")
        self._undo_btn.setToolTip("Undo the last edit (JSON / TSV)")
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._on_editor_undo)
        lay.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("  Redo")
        self._redo_btn.setObjectName("tb-btn-ghost")
        icons.apply_button(self._redo_btn, "redo")
        self._redo_btn.setToolTip("Redo the last undone edit")
        self._redo_btn.setEnabled(False)
        self._redo_btn.clicked.connect(self._on_editor_redo)
        lay.addWidget(self._redo_btn)

        lay.addWidget(VSep())

        # Validate file / folder reuse the dataset report — but run
        # only layer 1 (per-file checks). Layer 2 is dataset-wide and
        # is reserved for the dataset-level button.
        self._validate_file_btn = QPushButton("  Validate file")
        self._validate_file_btn.setObjectName("tb-btn")
        icons.apply_button(self._validate_file_btn, "file_check")
        self._validate_file_btn.setEnabled(False)
        self._validate_file_btn.setToolTip(
            "Re-validate the file currently selected in the BIDS "
            "tree. Updates only that file's badge + Validation pane "
            "section; the dataset-wide report stays intact."
        )
        self._validate_file_btn.clicked.connect(self.start_file_validation)
        lay.addWidget(self._validate_file_btn)

        self._validate_folder_btn = QPushButton("  Validate folder")
        self._validate_folder_btn.setObjectName("tb-btn")
        icons.apply_button(self._validate_folder_btn, "folder_check")
        self._validate_folder_btn.setEnabled(False)
        self._validate_folder_btn.setToolTip(
            "Re-validate every file under the folder currently "
            "selected in the BIDS tree. Skips folder/dataset-level "
            "checks — use “Validate dataset” for the full pass."
        )
        self._validate_folder_btn.clicked.connect(self.start_folder_validation)
        lay.addWidget(self._validate_folder_btn)

        self._validate_dataset_btn = QPushButton("  Validate dataset")
        self._validate_dataset_btn.setObjectName("tb-btn")
        icons.apply_button(self._validate_dataset_btn, "dataset")
        self._validate_dataset_btn.setEnabled(False)
        self._validate_dataset_btn.clicked.connect(self.start_dataset_validation)
        lay.addWidget(self._validate_dataset_btn)

        # Deep-checks toggle — when on, "Validate dataset" reads NIfTI
        # headers and file contents (slower, more thorough); when off it
        # runs the fast structural pass used for live revalidation. Maps
        # to the validator's read-headers mode. State persists via
        # AppSettings (the ``editor_strict_validate`` key, kept for
        # back-compat).
        from .app_settings import AppSettings
        self._strict_btn = QPushButton("  Deep checks")
        self._strict_btn.setObjectName("tb-btn-toggle")
        icons.apply_button(self._strict_btn, "strict")
        self._strict_btn.setCheckable(True)
        self._strict_btn.setChecked(AppSettings.load().editor_strict_validate)
        self._strict_btn.setToolTip(
            "Deep validation checks\n\n"
            "When ON, “Validate dataset” also opens NIfTI image headers to "
            "catch truncated or corrupt .nii / .nii.gz files. This is the "
            "only extra file read, and the only thing this toggle changes.\n\n"
            "When OFF, the validator runs the fast structural pass — file "
            "naming, entities, sidecar fields, TSV columns, associations. "
            "This is the mode used for live revalidation as you edit.\n\n"
            "Note: EEG / MEG / iEEG recordings are validated from their "
            "sidecars + channels.tsv and are never opened, so this toggle "
            "has no effect on a recordings-only dataset.\n\n"
            "Tip: leave this OFF while editing and flip it ON once before "
            "a final review."
        )
        self._strict_btn.toggled.connect(self._on_strict_toggled)
        lay.addWidget(self._strict_btn)

        lay.addWidget(VSep())

        # Status chips — kept hidden until a report lands. Each chip
        # opens the file-issues dialog filtered by the matching
        # severity (same "jump to" pattern as the Converter's toolbar).
        self._chip_ok = Chip("0 valid", "success")
        self._chip_warn = Chip("0 warnings", "warn")
        self._chip_err = Chip("0 errors", "err")
        for chip, sev in (
            (self._chip_ok, "ok"),
            (self._chip_warn, "warn"),
            (self._chip_err, "err"),
        ):
            chip.setVisible(False)
            chip.set_clickable(True)
            chip.clicked.connect(
                lambda s=sev: self._open_issues_dialog(s)
            )
            lay.addWidget(chip)

        lay.addStretch(1)

        # Busy spinner + activity label, mirroring the Converter toolbar.
        self._spinner = BusySpinner()
        lay.addWidget(self._spinner)

        return bar

    # ------------------------------------------------------------------
    # Strict-validation toggle
    # ------------------------------------------------------------------

    def _on_strict_toggled(self, checked: bool) -> None:
        from .app_settings import AppSettings
        AppSettings.remember_editor_strict_validate(checked)

    @staticmethod
    def _allowed_severities() -> set:
        """Severities the Editor displays, from the ``validate_show`` setting.

        Applied uniformly to the tree dots, the status chips, and the
        Validation pane so "errors only" / "warnings only" hides the rest
        everywhere (not just in the pane list).
        """
        from .app_settings import AppSettings
        show = AppSettings.load().validate_show
        if show == "error":
            return {Severity.ERR}
        if show == "warning":
            return {Severity.WARN}
        return {Severity.ERR, Severity.WARN}

    @staticmethod
    def _validation_engine_opts() -> tuple[Optional[str], int, bool]:
        """Resolve the bidsval schema + max-rows + flag-todos knobs from settings.

        Threaded into every validation worker so the engine never reads
        settings itself (it stays Qt-free). ``schema`` is ``None`` for
        bidsval's bundled default.
        """
        from .app_settings import AppSettings
        s = AppSettings.load()
        return (
            s.validate_schema_version or None,
            int(s.validate_max_rows),
            bool(s.validate_flag_todos),
        )

    # ------------------------------------------------------------------
    # Open-root flow
    # ------------------------------------------------------------------

    def _on_change_root(self) -> None:
        from .app_settings import AppSettings
        current = self.current_root()
        start = str(current) if current else (
            AppSettings.load().editor_bids_root or str(Path.home())
        )
        chosen = QFileDialog.getExistingDirectory(
            self, "Open BIDS root", start,
        )
        if not chosen:
            return
        self._set_root(Path(chosen), persist=True)

    def _set_root(self, path: Path, *, persist: bool) -> None:
        self._tree_pane.set_root(path)
        self._tree_pane.clear_badges()
        self._path_bar.set_value(str(path), ok=True)
        # Switching roots invalidates any stored report + form state.
        self._report = None
        self._sidecar_form.set_file(None, None, None)
        self._tsv_viewer.set_file(None, None)
        self._nifti_viewer.set_file(None, None)
        self._recording_viewer.set_file(None, None)
        self._center_stack.setCurrentWidget(self._sidecar_form)
        self._validation_pane.set_report(None)
        self._validation_pane.set_current_file(None, None)
        self._hide_chips()
        # Enable dataset-level validation now that we have a root.
        self._validate_dataset_btn.setEnabled(True)
        if persist:
            from .app_settings import AppSettings
            AppSettings.remember_editor_bids_root(path)

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, message: str) -> None:
        self.log_message.emit(message)

    def _on_report_ready(
        self, report: ValidationReport, root: Path,
    ) -> None:
        self._report = report
        self._stamp_tree_badges(report)
        self._update_chips(report)
        # Refresh the form for whatever file the user already has
        # selected — the FileVerdict for it just became available.
        current = self._sidecar_form.current_file()
        if current is not None:
            self._sidecar_form.set_file(current, self.current_root(), report)
        # The validation panel binds against the in-memory report.
        self._validation_pane.set_report(report)
        self._validation_pane.set_current_file(current, self.current_root())
        self.log_message.emit(
            f"Validation done — {report.counts.get('ok', 0)} ok, "
            f"{report.counts.get('warn', 0)} warn, "
            f"{report.counts.get('err', 0)} err"
        )
        del root  # only kept on the signal for caller correlation

    def _on_worker_failed(self, traceback_text: str) -> None:
        self.log_message.emit(f"Validation failed:\n{traceback_text}")

    def _on_worker_finished(self) -> None:
        self._set_busy(False)
        # Re-enable the button only if a root is still loaded.
        self._validate_dataset_btn.setEnabled(self.current_root() is not None)
        self._report_worker = None

    def _on_partial_ready(
        self,
        verdicts: list[FileVerdict],
        bids_root: Path,
        target_path: Path,
    ) -> None:
        """Merge per-file or per-folder verdicts into the in-memory report.

        Existing FileVerdicts for the same paths are replaced; new ones
        append. Dataset / folder issues stay untouched (they belong to
        the dataset-wide pass).
        """
        del bids_root  # we already cache it as self._current_root
        if not verdicts:
            return
        # Ensure we have a report to merge into.
        if self._report is None:
            self._report = ValidationReport(bids_root=self.current_root())
        existing_by_path = {fv.path: i for i, fv in enumerate(self._report.files)}
        for fv in verdicts:
            if fv.path in existing_by_path:
                self._report.files[existing_by_path[fv.path]] = fv
            else:
                existing_by_path[fv.path] = len(self._report.files)
                self._report.files.append(fv)
        # Re-rollup severity + counts (cheap to recompute).
        self._recompute_report_summary(self._report)
        # Refresh the UI.
        self._stamp_tree_badges(self._report)
        self._update_chips(self._report)
        self._validation_pane.set_report(self._report)
        self._validation_pane.set_current_file(
            self._sidecar_form.current_file(), self.current_root(),
        )
        # Sidecar pane re-binds against the now-fresh verdict.
        current = self._sidecar_form.current_file()
        if current is not None:
            self._sidecar_form.set_file(
                current, self.current_root(), self._report,
            )
        self.log_message.emit(
            f"Validation done — {len(verdicts)} file"
            f"{'s' if len(verdicts) != 1 else ''} re-checked"
        )
        del target_path

    def _on_partial_worker_finished(self) -> None:
        self._set_busy(False)
        self._partial_worker = None
        self._sync_validate_buttons_from_selection(
            self._tree_pane_current_path(),
        )

    def _tree_pane_current_path(self) -> Optional[Path]:
        """Return the absolute path of the BIDS tree's current selection."""
        from .widgets.bids_tree_pane import PATH_ROLE
        item = self._tree_pane._tree.currentItem()
        if item is None:
            return None
        raw = item.data(0, PATH_ROLE)
        return Path(raw) if raw else None

    def _sync_validate_buttons_from_selection(
        self, path: Optional[Path],
    ) -> None:
        """Enable Validate file when a file is selected, Validate
        folder when a folder is selected.

        Both stay disabled until a BIDS root is open. A partial-validate
        already in flight also disables them (debounce).
        """
        has_root = self.current_root() is not None
        running = self._partial_worker is not None and getattr(
            self._partial_worker, "isRunning", lambda: False,
        )()
        from .theme_manager import CUR
        pal = CUR()
        if path is None or not has_root or running:
            self._validate_file_btn.setEnabled(False)
            self._validate_folder_btn.setEnabled(False)
            # No selection: both icons stay accent (default).
            icons.apply_button(self._validate_file_btn, "file_check",
                               color=pal.get("accent"))
            icons.apply_button(self._validate_folder_btn, "folder_check",
                               color=pal.get("accent"))
            return
        file_selected = path.is_file()
        folder_selected = path.is_dir()
        self._validate_file_btn.setEnabled(file_selected)
        self._validate_folder_btn.setEnabled(folder_selected)
        # Recolor icons: the button matching the selected kind goes
        # green (success); the other stays accent blue.
        icons.apply_button(
            self._validate_file_btn, "file_check",
            color=pal.get("success" if file_selected else "accent"),
        )
        icons.apply_button(
            self._validate_folder_btn, "folder_check",
            color=pal.get("success" if folder_selected else "accent"),
        )

    @staticmethod
    def _recompute_report_summary(report: ValidationReport) -> None:
        """Reset ``severity`` and ``counts`` after files were mutated.

        Counts are PER-FINDING (every error/warning issue), matching the
        adapter + bidsval; ``ok`` is the number of clean files.
        """
        all_issues = [i for i in report.dataset_issues if not i.mirrored]
        for issues in report.folder_issues.values():
            all_issues.extend(i for i in issues if not i.mirrored)
        for f in report.files:
            all_issues.extend(i for i in f.issues if not i.mirrored)
        report.counts = {
            "ok": sum(1 for f in report.files if f.severity is Severity.OK),
            "warn": sum(1 for i in all_issues if i.severity is Severity.WARN),
            "err": sum(1 for i in all_issues if i.severity is Severity.ERR),
        }
        if any(i.severity is Severity.ERR for i in all_issues):
            report.severity = Severity.ERR
        elif any(i.severity is Severity.WARN for i in all_issues):
            report.severity = Severity.WARN
        else:
            report.severity = Severity.OK

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stamp_tree_badges(self, report: ValidationReport) -> None:
        """Map :class:`FileVerdict` severities onto tree rows.

        ``FileVerdict.path`` is **relative to bids_root**; the tree
        items carry absolute paths, so we resolve here before handing
        off to the pane.
        """
        root = self.current_root()
        if root is None:
            return
        allowed = self._allowed_severities()
        severities: dict[Path, str] = {}
        for fv in report.files:
            # Badge = worst ALLOWED issue on the file; clean / filtered-out
            # files get the green "ok" dot. So "errors only" leaves only error
            # files red and everything else green.
            issues = [i for i in fv.issues if i.severity in allowed]
            if any(i.severity is Severity.ERR for i in issues):
                value = "err"
            elif any(i.severity is Severity.WARN for i in issues):
                value = "warn"
            else:
                value = "ok"
            absolute = (root / fv.path).resolve() if not fv.path.is_absolute() \
                else fv.path
            severities[absolute] = value
        self._tree_pane.set_badges(severities)

    def _update_chips(self, report: ValidationReport) -> None:
        counts = report.counts
        allowed = self._allowed_severities()
        self._chip_ok.setText(f"{counts.get('ok', 0)} valid")
        self._chip_warn.setText(f"{counts.get('warn', 0)} warnings")
        self._chip_err.setText(f"{counts.get('err', 0)} errors")
        # The "Show findings" filter hides the chips for severities it excludes
        # (so "errors only" doesn't surface a warnings counter). "valid" stays.
        self._chip_ok.setVisible(True)
        self._chip_warn.setVisible(Severity.WARN in allowed)
        self._chip_err.setVisible(Severity.ERR in allowed)

    def _hide_chips(self) -> None:
        for chip in (self._chip_ok, self._chip_warn, self._chip_err):
            chip.setVisible(False)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._spinner.set_busy(busy, message=message)
        # While a validation run is in flight we lock the trigger so
        # double-clicks don't pile up workers.
        self._validate_dataset_btn.setEnabled(
            (not busy) and (self.current_root() is not None)
        )

    # ------------------------------------------------------------------
    # Tree ↔ sidecar form wiring
    # ------------------------------------------------------------------

    def _on_file_selected(self, path: Path) -> None:
        """User picked a row in the BIDS tree.

        Routes to a viewer based on the file extension:

        * ``.tsv`` / ``.tsv.gz`` → :class:`TsvViewerPane` (table).
        * ``.nii`` / ``.nii.gz`` → :class:`NiftiViewerPane` (2-D slice
          viewer with orientation buttons + brightness/contrast).
        * EEG / MEG / iEEG recordings (``.fif``, ``.edf``, ``.set``,
          ``.vhdr``, ``.cnt``, CTF ``.ds``, …) →
          :class:`RecordingViewerPane` (metadata card + threaded
          time-series viewer).
        * everything else (JSON sidecars, …) → :class:`SidecarFormPane`.
        """
        if path.is_dir():
            # CTF .ds / EGI .mff are directory-shaped recordings — route
            # them to the recording viewer instead of the dir-clear path.
            if is_recording_path(path):
                self._show_recording(path)
                self._validation_pane.set_current_file(path, self.current_root())
                return
            # Plain directories don't carry sidecars. We still push the
            # folder down to the validation panel so its "Folder"
            # section can reflect the user's focus.
            self._sidecar_form.set_file(None, None, None)
            self._tsv_viewer.set_file(None, None)
            self._nifti_viewer.set_file(None, None)
            self._recording_viewer.set_file(None, None)
            self._center_stack.setCurrentWidget(self._sidecar_form)
            self._validation_pane.set_current_file(path, self.current_root())
            return
        name = path.name.lower()
        root = self.current_root()
        if name.endswith(".tsv") or name.endswith(".tsv.gz"):
            self._tsv_viewer.set_file(path, root)
            # Other panes get cleared so a future toggle back doesn't
            # show stale state for a different file.
            self._sidecar_form.set_file(None, None, None)
            self._nifti_viewer.set_file(None, None)
            self._recording_viewer.set_file(None, None)
            self._center_stack.setCurrentWidget(self._tsv_viewer)
        elif name.endswith(".nii") or name.endswith(".nii.gz"):
            self._nifti_viewer.set_file(path, root)
            self._sidecar_form.set_file(None, None, None)
            self._tsv_viewer.set_file(None, None)
            self._recording_viewer.set_file(None, None)
            self._center_stack.setCurrentWidget(self._nifti_viewer)
        elif is_recording_path(path):
            self._show_recording(path)
        else:
            self._sidecar_form.set_file(path, root, self._report)
            self._tsv_viewer.set_file(None, None)
            self._nifti_viewer.set_file(None, None)
            self._recording_viewer.set_file(None, None)
            self._center_stack.setCurrentWidget(self._sidecar_form)
        self._validation_pane.set_current_file(path, root)

    def _show_recording(self, path: Path) -> None:
        """Route an EEG/MEG/iEEG recording to the recording viewer."""
        root = self.current_root()
        self._recording_viewer.set_file(path, root)
        self._sidecar_form.set_file(None, None, None)
        self._tsv_viewer.set_file(None, None)
        self._nifti_viewer.set_file(None, None)
        self._center_stack.setCurrentWidget(self._recording_viewer)

    def _on_pane_loading(self, busy: bool, message: str) -> None:
        """Mirror a center pane's threaded load onto the toolbar spinner.

        Only drives the spinner; it does not touch the validate buttons
        (those are owned by :meth:`_set_busy`). When a dataset validation
        is in flight we leave the spinner under its control.
        """
        if self._report_worker is not None and self._report_worker.isRunning():
            return
        self._spinner.set_busy(busy, message=message)

    # ------------------------------------------------------------------
    # Theme cascade (called by MainWindow._on_palette_changed)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Jump-to-file + jump-to-field interactivity
    # ------------------------------------------------------------------

    def _open_issues_dialog(self, severity: str) -> None:
        """Open a modal :class:`EditorIssuesDialog` filtered by severity.

        The dialog lists every file in the current report whose
        severity matches. Activating a card selects the file in the
        BIDS tree (which cascades through the panes).
        """
        if self._report is None or self.current_root() is None:
            return
        from .editor_issues_dialog import EditorIssuesDialog
        dlg = EditorIssuesDialog(
            self._report, severity, self.current_root(), parent=self,
        )
        dlg.file_selected.connect(self.select_file_in_tree)
        dlg.show()

    def select_file_in_tree(self, path: Path) -> None:
        """Select ``path`` in the BIDS tree (cascades to all panes).

        Public so other widgets (the issues dialog, the validation
        pane's fix-button handler) can use it.
        """
        target = str(path)
        tree = self._tree_pane._tree

        def visit(item) -> bool:
            from .widgets.bids_tree_pane import PATH_ROLE
            stored = item.data(0, PATH_ROLE)
            if stored == target:
                tree.setCurrentItem(item)
                tree.scrollToItem(item)
                return True
            for i in range(item.childCount()):
                if visit(item.child(i)):
                    return True
            return False

        for i in range(tree.topLevelItemCount()):
            if visit(tree.topLevelItem(i)):
                return
        # If the tree row can't be found (e.g. the file got renamed
        # between validation and the click), fall back to loading
        # the sidecar / TSV viewer directly so the click still works.
        self._on_file_selected(path)

    def _on_fix_requested(self, path: Path, field: str) -> None:
        """A user clicked a ValMessage's fix button.

        Navigate to where the problem is actually edited, then highlight it:

        * a TSV column finding -> open the ``.tsv`` and select that column;
        * a sidecar field finding -> open the ``.json`` and focus that field.
          When the finding sits on a data file (``*_bold.nii.gz``, where bidsval
          attaches sidecar metadata findings), redirect to its editable
          ``*_bold.json`` sibling so the field can be focused.
        """
        if path is None:
            return
        name = path.name.lower()

        if name.endswith(".tsv") or name.endswith(".tsv.gz"):
            self.select_file_in_tree(path)
            verdict = self._verdict_for(path)
            if verdict is not None and field:
                # Highlight every bad cell for this column. A column finding
                # carries all offending rows in ``lines`` (the viewer resolves
                # each 1-based row to a cell).
                items = self._tsv_highlight_items(verdict.issues, field=field)
                if items:
                    self._tsv_viewer.highlight_findings(items)
            return

        if name.endswith(".json"):
            target = path
        else:
            from ..editor.bidsmgr_checks import sibling_json
            sib = sibling_json(path)
            target = sib if (sib is not None and sib.exists()) else path

        if target != self._sidecar_form.current_file():
            self.select_file_in_tree(target)
        if field and target.name.lower().endswith(".json"):
            self._sidecar_form.focus_field(field)

    def _on_highlight_all_requested(self, path: Path) -> None:
        """Highlight every shown error/warning field (JSON) or cell (TSV) for
        ``path`` in the editor. For a data file, the editable target is its
        ``.json`` sibling (where the sidecar fields live)."""
        if path is None or self._report is None:
            return
        name = path.name.lower()
        allowed = self._allowed_severities()

        # TSV: highlight each finding's specific cell (or whole column when the
        # finding has no row), so the bad values are pinpointed, not the column.
        if name.endswith(".tsv") or name.endswith(".tsv.gz"):
            verdict = self._verdict_for(path)
            if verdict is None:
                return
            items = self._tsv_highlight_items(verdict.issues, allowed=allowed)
            if not items:
                return
            if path != self._tsv_viewer.current_file():
                self.select_file_in_tree(path)
            self._tsv_viewer.highlight_findings(items)
            return

        # JSON sidecar (or a data file -> its editable .json sibling).
        if name.endswith(".json"):
            target = path
        else:
            from ..editor.bidsmgr_checks import sibling_json
            sib = sibling_json(path)
            target = sib if (sib is not None and sib.exists()) else path
        verdict = self._verdict_for(target)
        if verdict is None:
            return
        rank = {"err": 2, "warn": 1}
        mapping: dict[str, str] = {}
        for issue in verdict.issues:
            if not issue.field or issue.severity not in allowed:
                continue
            sv = issue.severity.value
            if rank.get(sv, 0) > rank.get(mapping.get(issue.field, ""), 0):
                mapping[issue.field] = sv
        if not mapping:
            return
        if target != self._sidecar_form.current_file():
            self.select_file_in_tree(target)
        self._sidecar_form.highlight_fields(mapping)

    @staticmethod
    def _tsv_highlight_items(issues, *, field=None, allowed=None) -> list:
        """Flatten TSV findings to ``(column, line, severity)`` items, one per
        offending row. A column finding lists every bad row in ``lines``; we
        expand them all so each bad cell is highlighted (falls back to ``line``
        for column-level findings without rows)."""
        items: list = []
        for issue in issues:
            if not issue.field:
                continue
            if field is not None and issue.field != field:
                continue
            if allowed is not None and issue.severity not in allowed:
                continue
            rows = issue.lines or (
                [issue.line] if issue.line is not None else [None]
            )
            for ln in rows:
                items.append((issue.field, ln, issue.severity.value))
        return items

    def _verdict_for(self, path: Path) -> Optional[FileVerdict]:
        """The :class:`FileVerdict` in the current report for ``path`` (matched
        on the absolute path), or ``None``."""
        report = self._report
        root = self.current_root()
        if report is None or root is None:
            return None
        try:
            target_abs = str(path.resolve())
        except OSError:
            target_abs = str(path)
        for fv in report.files:
            cand = fv.path if fv.path.is_absolute() else (root / fv.path)
            try:
                cand_abs = str(cand.resolve())
            except OSError:
                cand_abs = str(cand)
            if cand_abs == target_abs:
                return fv
        return None

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Undo / redo (delegates to the active editable center pane)
    # ------------------------------------------------------------------

    def _active_history_pane(self):
        """The current center pane if it supports undo/redo, else None."""
        w = self._center_stack.currentWidget()
        if w in (self._sidecar_form, self._tsv_viewer):
            return w
        return None

    def _sync_undo_redo(self, *args) -> None:
        del args
        pane = self._active_history_pane()
        self._undo_btn.setEnabled(bool(pane) and pane.can_undo())
        self._redo_btn.setEnabled(bool(pane) and pane.can_redo())

    def _on_editor_undo(self) -> None:
        pane = self._active_history_pane()
        if pane is not None:
            pane.undo()

    def _on_editor_redo(self) -> None:
        pane = self._active_history_pane()
        if pane is not None:
            pane.redo()

    def showEvent(self, event) -> None:  # noqa: N802
        """Refresh the tree when the Editor becomes visible.

        A conversion (or any external change) may have written into the open
        dataset while the user was in the Converter tab. ``refresh`` re-walks
        the root while preserving the user's expand / selection state, so the
        editor reflects the current disk contents the moment it is shown (and
        the live ``QFileSystemWatcher`` keeps it current thereafter).
        """
        super().showEvent(event)
        self._tree_pane.refresh()

    def repaint_for_palette(self, pal: dict) -> None:
        self._tree_pane.repaint_for_palette(pal)
        self._sidecar_form.repaint_for_palette(pal)
        self._tsv_viewer.repaint_for_palette(pal)
        self._nifti_viewer.repaint_for_palette(pal)
        self._recording_viewer.repaint_for_palette(pal)
        self._validation_pane.repaint_for_palette(pal)
        for frame in getattr(self, "_panel_frames", []):
            frame._refresh_icons()
        # Re-tint toolbar icons after the icon cache is cleared in
        # ``MainWindow._on_palette_changed``.
        icons.apply_button(self._open_btn, "open_folder")
        icons.apply_button(self._undo_btn, "undo")
        icons.apply_button(self._redo_btn, "redo")
        icons.apply_button(self._validate_dataset_btn, "dataset")
        icons.apply_button(self._strict_btn, "strict")
        # The two partial-validate buttons get their tint from the
        # current tree selection (green when their kind is selected,
        # accent blue otherwise) — defer to the sync helper so the
        # selection state survives a dark↔light swap.
        self._sync_validate_buttons_from_selection(self._tree_pane_current_path())


__all__ = ["EditorPanel"]
