"""Converter tab implementation for the BIDS Manager GUI.

This module houses the conversion workflow UI and supporting widgets that were
previously embedded in the monolithic ``gui.py`` file. Behaviour is preserved
while the code is organised for easier maintenance.
"""

import logging
import os
from typing import Any, Callable, Optional

from bids_manager.GUI.common import *  # noqa: F401,F403 - shared GUI helpers


# ---------------------------------------------------------------------------
# Helper widgets and worker threads used exclusively by the converter tab
# ---------------------------------------------------------------------------

class _ConflictScannerWorker(QObject):
    """Run the conflict detection scan in a background thread."""

    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, root_dir: str, finder: Callable[[str, int], dict], n_jobs: int):
        super().__init__()
        self._root_dir = root_dir
        self._finder = finder
        # ``n_jobs`` mirrors the CPU limit used for the main scanning step so we
        # do not overwhelm the system when running both operations back-to-back.
        self._n_jobs = max(1, n_jobs)

    @pyqtSlot()
    def run(self) -> None:
        """Execute the slow directory walk outside the GUI thread."""

        try:
            conflicts = self._finder(self._root_dir, self._n_jobs)
        except Exception as exc:  # pragma: no cover - runtime safety
            # Forward the error message back to the GUI thread so the caller
            # can decide how to handle it without freezing the interface.
            self.failed.emit(str(exc))
        else:
            self.finished.emit(conflicts)

class AutoFillTableWidget(QTableWidget):
    """``QTableWidget`` with an Excel-like autofill handle.

    The widget exposes a small square in the bottom-right corner of the current
    selection.  Dragging this handle extends the selection and automatically
    fills the new cells either by cloning the original content or by continuing
    simple sequences (numeric, datetime, or text with trailing digits).
    """

    HANDLE_SIZE = 8  # Square size in device pixels

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._handle_rect = QRect()
        self._autofill_active = False
        self._autofill_origin_range: Optional[QTableWidgetSelectionRange] = None
        self._autofill_current_range: Optional[QTableWidgetSelectionRange] = None
        # Whenever the selection changes we repaint so the handle follows it.
        self.itemSelectionChanged.connect(self._refresh_handle)

    # ------------------------------------------------------------------
    # Painting utilities
    def _index_is_editable(self, row: int, column: int) -> bool:
        """Return ``True`` when the ``(row, column)`` cell can be edited."""

        model = self.model()
        index = model.index(row, column)
        if not index.isValid():
            return False
        flags = model.flags(index)
        return bool(flags & Qt.ItemIsEditable)

    def _range_is_editable(
        self, rng: Optional[QTableWidgetSelectionRange]
    ) -> bool:
        """Return ``True`` when *all* cells in ``rng`` expose the edit flag."""

        if rng is None:
            return False
        for row in range(rng.topRow(), rng.bottomRow() + 1):
            for col in range(rng.leftColumn(), rng.rightColumn() + 1):
                if not self._index_is_editable(row, col):
                    return False
        return True

    def _clamp_to_editable(
        self,
        origin: QTableWidgetSelectionRange,
        candidate: QTableWidgetSelectionRange,
    ) -> QTableWidgetSelectionRange:
        """Restrict ``candidate`` so autofill never crosses non-editable cells."""

        # Limit horizontal growth by checking each additional column in order.
        max_right = origin.rightColumn()
        if candidate.rightColumn() > max_right:
            for col in range(origin.rightColumn() + 1, candidate.rightColumn() + 1):
                if all(
                    self._index_is_editable(row, col)
                    for row in range(origin.topRow(), origin.bottomRow() + 1)
                ):
                    max_right = col
                else:
                    break

        # Limit vertical growth by checking each extra row with the approved columns.
        max_bottom = origin.bottomRow()
        if candidate.bottomRow() > max_bottom:
            for row in range(origin.bottomRow() + 1, candidate.bottomRow() + 1):
                if all(
                    self._index_is_editable(row, col)
                    for col in range(origin.leftColumn(), max_right + 1)
                ):
                    max_bottom = row
                else:
                    break

        if (
            max_right == origin.rightColumn()
            and max_bottom == origin.bottomRow()
        ):
            return origin

        return QTableWidgetSelectionRange(
            origin.topRow(),
            origin.leftColumn(),
            max_bottom,
            max_right,
        )
    def _refresh_handle(self) -> None:
        """Trigger a repaint so the autofill handle reflects the selection."""

        self._handle_rect = QRect()
        self.viewport().update()

    def _current_selection_range(self) -> Optional[QTableWidgetSelectionRange]:
        """Return the single active selection range (if any)."""

        ranges = self.selectedRanges()
        if len(ranges) != 1:
            return None
        return ranges[0]

    def paintEvent(self, event):  # noqa: D401 - Qt override
        super().paintEvent(event)

        rng = self._current_selection_range()
        if not self._range_is_editable(rng):
            self._handle_rect = QRect()
            return

        model_index = self.model().index(rng.bottomRow(), rng.rightColumn())
        rect = self.visualRect(model_index)
        if not rect.isValid():
            self._handle_rect = QRect()
            return

        size = self.HANDLE_SIZE
        self._handle_rect = QRect(
            rect.right() - size + 1,
            rect.bottom() - size + 1,
            size,
            size,
        )

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self._handle_rect, self.palette().highlight())
        border = QPen(self.palette().color(QPalette.Dark))
        painter.setPen(border)
        painter.drawRect(self._handle_rect)
        painter.end()

    # ------------------------------------------------------------------
    # Mouse interaction handling
    def mousePressEvent(self, event):  # noqa: D401 - Qt override
        if (
            event.button() == Qt.LeftButton
            and self._handle_rect.contains(event.pos())
            and self._current_selection_range() is not None
        ):
            self._autofill_active = True
            self._autofill_origin_range = self._current_selection_range()
            self._autofill_current_range = self._autofill_origin_range
            self.viewport().setCursor(Qt.SizeAllCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: D401 - Qt override
        if self._autofill_active:
            new_range = self._compute_drag_range(event.pos())
            if new_range is not None:
                self._autofill_current_range = new_range
                # Replace the selection with the preview range so the user sees
                # the future extent of the autofill before releasing the mouse.
                self.blockSignals(True)
                self.clearSelection()
                self.setRangeSelected(new_range, True)
                self.blockSignals(False)
                self.viewport().update()
            event.accept()
            return

        if self._handle_rect.contains(event.pos()):
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().unsetCursor()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: D401 - Qt override
        if self._autofill_active and event.button() == Qt.LeftButton:
            try:
                self._finish_autofill()
            finally:
                self._autofill_active = False
                self.viewport().unsetCursor()
                self._autofill_origin_range = None
                self._autofill_current_range = None
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _compute_drag_range(self, pos: QPoint) -> Optional[QTableWidgetSelectionRange]:
        """Return the preview range while dragging the autofill handle."""

        if self._autofill_origin_range is None:
            return None

        origin = self._autofill_origin_range
        row = self.rowAt(pos.y())
        col = self.columnAt(pos.x())
        if row < 0 or col < 0:
            return origin

        # Only allow extending outward from the original bottom/right edge.
        if row < origin.bottomRow():
            row = origin.bottomRow()
        if col < origin.rightColumn():
            col = origin.rightColumn()

        if row == origin.bottomRow() and col == origin.rightColumn():
            return origin

        candidate = QTableWidgetSelectionRange(
            origin.topRow(),
            origin.leftColumn(),
            row,
            col,
        )

        return self._clamp_to_editable(origin, candidate)

    # ------------------------------------------------------------------
    # Autofill logic
    def _finish_autofill(self) -> None:
        """Apply the autofill operation once the user releases the mouse."""

        if self._autofill_origin_range is None:
            return

        final_range = self._autofill_current_range or self._autofill_origin_range
        origin = self._autofill_origin_range

        if (
            final_range.bottomRow() == origin.bottomRow()
            and final_range.rightColumn() == origin.rightColumn()
        ):
            # Nothing changed; restore the original selection highlight.
            self.blockSignals(True)
            self.clearSelection()
            self.setRangeSelected(origin, True)
            self.blockSignals(False)
            self.viewport().update()
            return

        self._apply_autofill(origin, final_range)

        # Keep the extended range selected after the fill to match spreadsheet UX.
        self.blockSignals(True)
        self.clearSelection()
        self.setRangeSelected(final_range, True)
        self.blockSignals(False)
        self.viewport().update()

    def _apply_autofill(
        self,
        origin: QTableWidgetSelectionRange,
        target: QTableWidgetSelectionRange,
    ) -> None:
        """Populate ``target`` based on ``origin`` and the autofill heuristics."""

        extend_down = target.bottomRow() > origin.bottomRow()
        extend_right = target.rightColumn() > origin.rightColumn()

        if extend_right:
            self._fill_right(origin, target.rightColumn())

        if extend_down:
            base_right = target.rightColumn() if extend_right else origin.rightColumn()
            base_range = QTableWidgetSelectionRange(
                origin.topRow(),
                origin.leftColumn(),
                origin.bottomRow(),
                base_right,
            )
            self._fill_down(base_range, target.bottomRow())

    def _fill_right(self, base: QTableWidgetSelectionRange, target_right: int) -> None:
        """Extend ``base`` horizontally until ``target_right`` inclusive."""

        extra = target_right - base.rightColumn()
        if extra <= 0:
            return

        for row in range(base.topRow(), base.bottomRow() + 1):
            values = [
                self._get_item_text(row, col)
                for col in range(base.leftColumn(), base.rightColumn() + 1)
            ]
            new_values = self._extend_series(values, extra)
            for offset, value in enumerate(new_values, start=1):
                self._set_item_text(row, base.rightColumn() + offset, value)

    def _fill_down(self, base: QTableWidgetSelectionRange, target_bottom: int) -> None:
        """Extend ``base`` vertically until ``target_bottom`` inclusive."""

        extra = target_bottom - base.bottomRow()
        if extra <= 0:
            return

        for col in range(base.leftColumn(), base.rightColumn() + 1):
            values = [
                self._get_item_text(row, col)
                for row in range(base.topRow(), base.bottomRow() + 1)
            ]
            new_values = self._extend_series(values, extra)
            for offset, value in enumerate(new_values, start=1):
                self._set_item_text(base.bottomRow() + offset, col, value)

    # ------------------------------------------------------------------
    # Sequence helpers
    def _extend_series(self, values: list[str], steps: int) -> list[str]:
        """Return ``steps`` new values continuing ``values`` when possible."""

        if steps <= 0 or not values:
            return []

        numeric = self._extend_numeric_series(values, steps)
        if numeric is not None:
            return numeric

        datelike = self._extend_datetime_series(values, steps)
        if datelike is not None:
            return datelike

        patterned = self._extend_text_pattern_series(values, steps)
        if patterned is not None:
            return patterned

        # Fallback: repeat the original pattern cyclically.
        repeated = []
        for i in range(steps):
            repeated.append(values[i % len(values)])
        return repeated

    def _extend_numeric_series(
        self,
        values: list[str],
        steps: int,
    ) -> Optional[list[str]]:
        """Continue integer/decimal sequences when the pattern is consistent."""

        stripped = [v.strip() for v in values]
        if any(not s for s in stripped):
            return None

        # Try integer sequences first so "01", "02" keep their padding.
        if all(re.fullmatch(r"[+-]?\d+", s) for s in stripped):
            numbers = [int(s) for s in stripped]
            diff = 0
            if len(numbers) >= 2:
                diffs = [numbers[i] - numbers[i - 1] for i in range(1, len(numbers))]
                if all(d == diffs[0] for d in diffs):
                    diff = diffs[0]
            pad_width = len(stripped[-1].lstrip("+-"))
            has_leading_zero = stripped[-1].lstrip("+-").startswith("0") and pad_width > 1
            force_plus = stripped[-1].startswith("+")
            current = numbers[-1]
            generated: list[str] = []
            for _ in range(steps):
                current += diff
                text = str(current)
                if has_leading_zero and current >= 0:
                    text = f"{current:0{pad_width}d}"
                if force_plus and not text.startswith("-") and not text.startswith("+"):
                    text = "+" + text
                generated.append(text)
            return generated

        # Fall back to decimals when integers are not appropriate.
        decimals: list[Decimal] = []
        decimal_places = 0
        for s in stripped:
            try:
                dec = Decimal(s)
            except InvalidOperation:
                return None
            decimals.append(dec)
            if "." in s:
                decimal_places = max(decimal_places, len(s.split(".")[-1]))

        diff = Decimal(0)
        if len(decimals) >= 2:
            diffs = [decimals[i] - decimals[i - 1] for i in range(1, len(decimals))]
            if all(d == diffs[0] for d in diffs):
                diff = diffs[0]
        current = decimals[-1]
        generated = []
        for _ in range(steps):
            current += diff
            if decimal_places:
                generated.append(f"{current:.{decimal_places}f}")
            else:
                generated.append(str(current))
        return generated

    def _extend_datetime_series(
        self,
        values: list[str],
        steps: int,
    ) -> Optional[list[str]]:
        """Continue datetime-like strings when intervals are consistent."""

        try:
            parsed = pd.to_datetime(values, errors="raise", infer_datetime_format=True)
        except Exception:
            return None

        if parsed.isna().any():
            return None

        delta = pd.Timedelta(0)
        if len(parsed) >= 2:
            diffs = parsed.diff().iloc[1:]
            if not diffs.empty and all(d == diffs.iloc[0] for d in diffs):
                delta = diffs.iloc[0]

        last = parsed.iloc[-1]
        template = values[-1]
        fmt = guess_datetime_format(template)
        generated = []
        current = last
        for _ in range(steps):
            current = current + delta
            if fmt:
                generated.append(current.strftime(fmt))
            else:
                if "T" in template:
                    generated.append(current.isoformat())
                elif ":" in template:
                    generated.append(current.strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    generated.append(current.date().isoformat())
        return generated

    def _extend_text_pattern_series(
        self,
        values: list[str],
        steps: int,
    ) -> Optional[list[str]]:
        """Continue strings ending with digits (e.g. "scan01")."""

        matches = [re.match(r"^(.*?)(\d+)$", v.strip()) for v in values]
        if any(m is None for m in matches):
            return None

        prefixes = [m.group(1) for m in matches if m is not None]
        numbers = [int(m.group(2)) for m in matches if m is not None]
        if not prefixes or not numbers:
            return None
        if any(p != prefixes[0] for p in prefixes):
            return None

        diff = 0
        if len(numbers) >= 2:
            diffs = [numbers[i] - numbers[i - 1] for i in range(1, len(numbers))]
            if all(d == diffs[0] for d in diffs):
                diff = diffs[0]
        pad_width = len(matches[-1].group(2)) if matches[-1] is not None else 0
        current = numbers[-1]
        prefix = prefixes[-1]
        generated = []
        for _ in range(steps):
            current += diff
            text = f"{current:0{pad_width}d}" if pad_width else str(current)
            generated.append(f"{prefix}{text}")
        return generated

    # ------------------------------------------------------------------
    # Cell helpers
    def _get_item_text(self, row: int, column: int) -> str:
        """Return the text stored at ``(row, column)`` (empty if missing)."""

        item = self.item(row, column)
        return item.text() if item is not None else ""

    def _set_item_text(self, row: int, column: int, value: str) -> None:
        """Assign ``value`` to ``(row, column)``, creating an item if required."""

        item = self.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.setItem(row, column, item)
        item.setText(value)

class MappingSortDialog(QDialog):
    """Dialog used to configure multi-level sorting for the metadata table."""

    def __init__(self, columns: list[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sort scanned metadata")
        self._columns = columns
        self._level_rows: list[tuple[QComboBox, QComboBox, QWidget]] = []

        layout = QVBoxLayout(self)
        info = QLabel(
            "Select the columns to sort by in priority order. "
            "Each level is applied sequentially, just like Excel's multi-column sort."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._levels_layout = QVBoxLayout()
        layout.addLayout(self._levels_layout)

        controls_layout = QHBoxLayout()
        self._add_level_btn = QPushButton("Add level")
        self._add_level_btn.clicked.connect(self._add_level)
        self._remove_level_btn = QPushButton("Remove level")
        self._remove_level_btn.clicked.connect(self._remove_level)
        controls_layout.addWidget(self._add_level_btn)
        controls_layout.addWidget(self._remove_level_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Always start with a single sorting level configured.
        self._add_level()
        self._update_button_state()

    # ------------------------------------------------------------------
    def _add_level(self) -> None:
        """Append a new sorting level to the dialog."""

        if len(self._level_rows) >= len(self._columns):
            return

        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        column_combo = QComboBox(row_widget)
        column_combo.addItems(self._columns)
        # Try to pre-select the first column that is not already used.
        used = {combo.currentText() for combo, _order, _widget in self._level_rows}
        for idx, name in enumerate(self._columns):
            if name not in used:
                column_combo.setCurrentIndex(idx)
                break

        order_combo = QComboBox(row_widget)
        order_combo.addItems(["Ascending", "Descending"])

        row_layout.addWidget(QLabel(f"Level {len(self._level_rows) + 1}", row_widget))
        row_layout.addWidget(column_combo)
        row_layout.addWidget(order_combo)
        row_layout.addStretch()

        self._levels_layout.addWidget(row_widget)
        self._level_rows.append((column_combo, order_combo, row_widget))
        self._update_button_state()

    def _remove_level(self) -> None:
        """Remove the last configured sorting level."""

        if not self._level_rows:
            return
        combo, order_combo, widget = self._level_rows.pop()
        combo.deleteLater()
        order_combo.deleteLater()
        widget.deleteLater()
        self._update_button_state()

    def _update_button_state(self) -> None:
        """Enable/disable controls based on the current dialog state."""

        self._add_level_btn.setEnabled(len(self._level_rows) < len(self._columns))
        self._remove_level_btn.setEnabled(len(self._level_rows) > 1)

    def sort_instructions(self) -> list[tuple[str, bool]]:
        """Return the configured sorting hierarchy as ``(column, ascending)``."""

        instructions: list[tuple[str, bool]] = []
        seen: set[str] = set()
        for column_combo, order_combo, _widget in self._level_rows:
            column = column_combo.currentText()
            if not column or column in seen:
                continue
            ascending = order_combo.currentText() == "Ascending"
            instructions.append((column, ascending))
            seen.add(column)
        return instructions

class SubjectDelegate(QStyledItemDelegate):
    """Delegate to edit BIDS subject IDs without altering the 'sub-' prefix."""

    def createEditor(self, parent, option, index):  # noqa: D401 - Qt override
        return QLineEdit(parent)

    def setEditorData(self, editor, index):  # noqa: D401 - Qt override
        text = index.model().data(index, Qt.EditRole)
        suffix = text[4:] if text.startswith("sub-") else text
        editor.setText(suffix)
        editor.selectAll()

    def setModelData(self, editor, model, index):  # noqa: D401 - Qt override
        model.setData(index, "sub-" + editor.text(), Qt.EditRole)

class ShrinkableScrollArea(QScrollArea):
    """``QScrollArea`` variant that allows the parent splitter to shrink."""

    def minimumSizeHint(self) -> QSize:  # noqa: D401 - Qt override
        return QSize(0, 0)

    def sizeHint(self) -> QSize:  # noqa: D401 - Qt override
        hint = super().sizeHint()
        return QSize(max(0, hint.width()), max(0, hint.height()))

class ConverterMixin:
    """Behaviour and UI for the Converter tab extracted from ``gui.py``."""

    def initConvertTab(self):
        """Create the Convert tab with a cleaner layout."""
        # This tab guides the user through the DICOM → BIDS workflow.
        # It contains controls to select directories, review the inventory TSV
        # and run the conversion pipeline while showing live logs.
        self.convert_tab = QWidget()
        main_layout = QVBoxLayout(self.convert_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        cfg_group = QGroupBox("Configuration")
        cfg_layout = QGridLayout(cfg_group)

        dicom_label = QLabel("<b>Raw data Dir:</b>")
        self.dicom_dir_edit = QLineEdit()
        self.dicom_dir_edit.setReadOnly(True)
        dicom_browse = QPushButton("Browse…")
        dicom_browse.clicked.connect(self.selectDicomDir)
        cfg_layout.addWidget(dicom_label, 0, 0)
        cfg_layout.addWidget(self.dicom_dir_edit, 0, 1)
        cfg_layout.addWidget(dicom_browse, 0, 2)

        bids_label = QLabel("<b>BIDS Out Dir:</b>")
        self.bids_out_edit = QLineEdit()
        self.bids_out_edit.setReadOnly(True)
        bids_browse = QPushButton("Browse…")
        bids_browse.clicked.connect(self.selectBIDSOutDir)
        cfg_layout.addWidget(bids_label, 1, 0)
        cfg_layout.addWidget(self.bids_out_edit, 1, 1)
        cfg_layout.addWidget(bids_browse, 1, 2)

        tsvname_label = QLabel("<b>TSV Name:</b>")
        self.tsv_name_edit = QLineEdit("subject_summary.tsv")
        cfg_layout.addWidget(tsvname_label, 2, 0)
        cfg_layout.addWidget(self.tsv_name_edit, 2, 1, 1, 2)

        self.tsv_button = QPushButton("Scan files")
        self.tsv_button.clicked.connect(self.runInventory)
        self.tsv_stop_button = QPushButton("Stop")
        self.tsv_stop_button.setEnabled(False)
        self.tsv_stop_button.clicked.connect(self.stopInventory)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.tsv_button)
        btn_row.addWidget(self.tsv_stop_button)
        btn_row.addStretch()
        cfg_layout.addLayout(btn_row, 3, 0, 1, 3)

        # Place the configuration group and the logo on the same row with
        # equal width. The logo stays centered even when the window resizes.
        top_row = QHBoxLayout()
        top_row.addWidget(cfg_group)
        self.logo_label = QLabel()
        logo_container = QWidget()
        lc_layout = QVBoxLayout(logo_container)
        lc_layout.setContentsMargins(0, 0, 0, 0)
        lc_layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)
        top_row.addWidget(logo_container)
        top_row.setStretch(0, 1)
        top_row.setStretch(1, 1)
        main_layout.addLayout(top_row)
        self._update_logo()

        self.left_split = QSplitter(Qt.Vertical)
        self.right_split = QSplitter(Qt.Vertical)

        self.tsv_group = QGroupBox("Scanned data viewer")
        tsv_layout = QVBoxLayout(self.tsv_group)
        self.tsv_detach_button = QPushButton("»")
        self.tsv_detach_button.setFixedWidth(20)
        self.tsv_detach_button.setFixedHeight(20)
        self.tsv_detach_button.clicked.connect(self.detachTSVWindow)
        self.tsv_detach_button.setFocusPolicy(Qt.NoFocus)
        header_row_tsv = QHBoxLayout()
        header_row_tsv.addStretch()
        header_row_tsv.addWidget(self.tsv_detach_button)
        tsv_layout.addLayout(header_row_tsv)
        self.tsv_tabs = QTabWidget()
        tsv_layout.addWidget(self.tsv_tabs)

        # --- Scanned metadata tab ---
        metadata_tab = QWidget()
        metadata_layout = QVBoxLayout(metadata_tab)
        metadata_toolbar = QHBoxLayout()
        self.tsv_actions_button = QToolButton()
        self.tsv_actions_button.setText("Actions")
        self.tsv_actions_button.setPopupMode(QToolButton.InstantPopup)
        self.tsv_actions_menu = QMenu(self.tsv_actions_button)

        self.tsv_sort_action = QAction("Sort", self)
        self.tsv_sort_action.setEnabled(False)
        self.tsv_sort_action.setToolTip("Sort the scanned metadata table")
        self.tsv_sort_action.triggered.connect(self._open_sort_dialog)
        self.tsv_actions_menu.addAction(self.tsv_sort_action)

        self.tsv_load_action = QAction("Load TSV…", self)
        self.tsv_load_action.triggered.connect(self.selectAndLoadTSV)
        self.tsv_actions_menu.addAction(self.tsv_load_action)

        self.tsv_generate_ids_action = QAction("Generate unique IDs", self)
        self.tsv_generate_ids_action.setEnabled(False)
        self.tsv_generate_ids_action.triggered.connect(self.generateUniqueIDs)
        self.tsv_actions_menu.addAction(self.tsv_generate_ids_action)

        self.tsv_detect_rep_action = QAction("Detect repeats", self)
        self.tsv_detect_rep_action.triggered.connect(self.detectRepeatedSequences)
        self.tsv_actions_menu.addAction(self.tsv_detect_rep_action)

        self.tsv_save_action = QAction("Save changes", self)
        self.tsv_save_action.setEnabled(False)
        self.tsv_save_action.triggered.connect(self.applyMappingChanges)
        self.tsv_actions_menu.addAction(self.tsv_save_action)

        self.tsv_actions_button.setMenu(self.tsv_actions_menu)
        metadata_toolbar.addWidget(self.tsv_actions_button)
        metadata_toolbar.addStretch()
        metadata_layout.addLayout(metadata_toolbar)
        self.mapping_table = AutoFillTableWidget()
        # Expose immutable DICOM metadata (StudyDescription, FamilyName,
        # PatientID) alongside the editable identifiers so users can see the
        # original values while editing BIDS-specific fields.
        self.mapping_table.setColumnCount(16)
        self.mapping_table.setHorizontalHeaderLabels([
            "include",
            "source_folder",
            "StudyDescription",
            "FamilyName",
            "PatientID",
            "BIDS_name",
            "subject",
            "GivenName",
            "session",
            "sequence",
            "Proposed BIDS name",
            "series_uid",
            "acq_time",
            "rep",
            "modality",
            "modality_bids",
        ])
        hdr = self.mapping_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setStretchLastSection(True)
        self.mapping_table.verticalHeader().setVisible(False)
        # Keep BIDS name edits constrained by the delegate despite the shifted
        # column indices introduced above.
        self.mapping_table.setItemDelegateForColumn(5, SubjectDelegate(self.mapping_table))
        self.mapping_table.itemChanged.connect(self._updateDetectRepeatEnabled)
        self.mapping_table.itemChanged.connect(self._onMappingItemChanged)
        metadata_layout.addWidget(self.mapping_table)

        self.tsv_tabs.addTab(metadata_tab, "Scanned metadata")

        # --- Suffix dictionary tab ---
        dict_tab = QWidget()
        dict_layout = QVBoxLayout(dict_tab)
        toggle_row = QHBoxLayout()
        self.use_custom_patterns_box = QCheckBox("Use custom suffix patterns")
        self.use_custom_patterns_box.toggled.connect(self._on_custom_toggle)
        toggle_row.addWidget(self.use_custom_patterns_box)
        toggle_row.addStretch()
        dict_layout.addLayout(toggle_row)

        self.seq_tabs_widget = QTabWidget()
        dict_layout.addWidget(self.seq_tabs_widget)
        dict_btn_row = QHBoxLayout()
        dict_btn_row.addStretch()
        self.seq_save_button = QPushButton("Save")
        # Saving the suffix dictionary also applies the changes immediately so
        # users do not need a separate "Apply" step.
        self.seq_save_button.clicked.connect(self.saveSequenceDictionary)
        dict_btn_row.addWidget(self.seq_save_button)
        restore_btn = QPushButton("Restore defaults")
        restore_btn.clicked.connect(self.restoreSequenceDefaults)
        dict_btn_row.addWidget(restore_btn)
        dict_btn_row.addStretch()
        dict_layout.addLayout(dict_btn_row)

        self.tsv_tabs.addTab(dict_tab, "Suffix dictionary")
        self.loadSequenceDictionary()

        self.tsv_scroll = ShrinkableScrollArea()
        self.tsv_scroll.setWidgetResizable(True)
        self.tsv_scroll.setWidget(self.tsv_group)
        self.tsv_container = QWidget()
        tsv_container_layout = QVBoxLayout(self.tsv_container)
        tsv_container_layout.setContentsMargins(0, 0, 0, 0)
        tsv_container_layout.addWidget(self.tsv_scroll)

        self.left_split.addWidget(self.tsv_container)

        self.filter_group = QGroupBox("Filter")
        modal_layout = QVBoxLayout(self.filter_group)
        self.modal_tabs = QTabWidget()
        full_tab = QWidget()
        full_layout = QVBoxLayout(full_tab)
        self.full_tree = QTreeWidget()
        self.full_tree.setColumnCount(1)
        self.full_tree.setHeaderLabels(["BIDS Modality"])
        hdr = self.full_tree.header()
        for i in range(self.full_tree.columnCount()):
            hdr.setSectionResizeMode(i, QHeaderView.Interactive)
        for i in range(self.full_tree.columnCount()):
            self.full_tree.resizeColumnToContents(i)
        full_layout.addWidget(self.full_tree)
        self.modal_tabs.addTab(full_tab, "General view")

        specific_tab = QWidget()
        specific_layout = QVBoxLayout(specific_tab)
        self.specific_tree = QTreeWidget()
        self.specific_tree.setColumnCount(3)
        self.specific_tree.setHeaderLabels(["Study/Subject", "Files", "Time"])
        s_hdr = self.specific_tree.header()
        for i in range(self.specific_tree.columnCount()):
            s_hdr.setSectionResizeMode(i, QHeaderView.Interactive)
        for i in range(self.specific_tree.columnCount()):
            self.specific_tree.resizeColumnToContents(i)
        specific_layout.addWidget(self.specific_tree)
        self.last_rep_box = QCheckBox("Only last repeats")
        self.last_rep_box.setEnabled(False)
        self.last_rep_box.toggled.connect(self._onLastRepToggled)
        specific_layout.addWidget(self.last_rep_box)
        self.modal_tabs.addTab(specific_tab, "Specific view")

        naming_tab = QWidget()
        naming_layout = QVBoxLayout(naming_tab)
        self.naming_table = QTableWidget()
        self.naming_table.setColumnCount(3)
        self.naming_table.setHorizontalHeaderLabels(["Study", "Given name", "BIDS name"])
        n_hdr = self.naming_table.horizontalHeader()
        for i in range(self.naming_table.columnCount()):
            n_hdr.setSectionResizeMode(i, QHeaderView.Interactive)
        n_hdr.setStretchLastSection(True)
        for i in range(self.naming_table.columnCount()):
            self.naming_table.resizeColumnToContents(i)
        self.naming_table.setItemDelegateForColumn(2, SubjectDelegate(self.naming_table))
        naming_layout.addWidget(self.naming_table)
        self.naming_table.itemChanged.connect(self._onNamingEdited)
        self.name_choice = QComboBox()
        self.name_choice.addItems(["Use BIDS names", "Use given names"])
        self.name_choice.setEnabled(False)
        self.name_choice.currentIndexChanged.connect(self._onNameChoiceChanged)
        naming_layout.addWidget(self.name_choice)
        self.modal_tabs.addTab(naming_tab, "Edit naming")

        # Always Exclude tab
        exclude_tab = QWidget()
        exclude_layout = QVBoxLayout(exclude_tab)
        self.exclude_table = QTableWidget()
        self.exclude_table.setColumnCount(2)
        self.exclude_table.setHorizontalHeaderLabels(["Active", "Pattern"])
        ex_hdr = self.exclude_table.horizontalHeader()
        ex_hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        ex_hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        exclude_layout.addWidget(self.exclude_table)

        ex_add_row = QHBoxLayout()
        self.exclude_edit = QLineEdit()
        ex_add_row.addWidget(self.exclude_edit)
        ex_add_btn = QPushButton("Add")
        ex_add_btn.clicked.connect(self._exclude_add)
        ex_add_row.addWidget(ex_add_btn)
        exclude_layout.addLayout(ex_add_row)

        ex_save_btn = QPushButton("Save")
        ex_save_btn.clicked.connect(self.saveExcludePatterns)
        exclude_layout.addWidget(ex_save_btn, alignment=Qt.AlignRight)
        self.modal_tabs.addTab(exclude_tab, "Always exclude")

        # Load saved exclude patterns now that the table exists
        self.loadExcludePatterns()

        header_row_filter = QHBoxLayout()
        self.filter_detach_button = QPushButton("»")
        self.filter_detach_button.setFixedWidth(20)
        self.filter_detach_button.setFixedHeight(20)
        self.filter_detach_button.clicked.connect(self.detachFilterWindow)
        self.filter_detach_button.setFocusPolicy(Qt.NoFocus)
        header_row_filter.addStretch()
        header_row_filter.addWidget(self.filter_detach_button)
        modal_layout.addLayout(header_row_filter)
        modal_layout.addWidget(self.modal_tabs)

        self.filter_scroll = ShrinkableScrollArea()
        self.filter_scroll.setWidgetResizable(True)
        self.filter_scroll.setWidget(self.filter_group)
        self.filter_container = QWidget()
        filter_container_layout = QVBoxLayout(self.filter_container)
        filter_container_layout.setContentsMargins(0, 0, 0, 0)
        filter_container_layout.addWidget(self.filter_scroll)

        self.right_split.addWidget(self.filter_container)
        self.left_split.setStretchFactor(0, 1)
        self.left_split.setStretchFactor(1, 1)
        self.right_split.setStretchFactor(0, 1)
        self.right_split.setStretchFactor(1, 1)

        self.preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(self.preview_group)
        self.preview_tabs = QTabWidget()

        text_tab = QWidget()
        text_lay = QVBoxLayout(text_tab)
        self.preview_text = QTreeWidget()
        self.preview_text.setColumnCount(2)
        self.preview_text.setHeaderLabels(["BIDS Path", "Original Sequence"])
        text_lay.addWidget(self.preview_text)
        self.preview_tabs.addTab(text_tab, "Text")

        tree_tab = QWidget()
        tree_lay = QVBoxLayout(tree_tab)
        self.preview_tree = QTreeWidget()
        self.preview_tree.setColumnCount(2)
        self.preview_tree.setHeaderLabels(["BIDS Structure", "Original Sequence"])
        tree_lay.addWidget(self.preview_tree)
        self.preview_tabs.addTab(tree_tab, "Tree")

        header_row_preview = QHBoxLayout()
        self.preview_detach_button = QPushButton("»")
        self.preview_detach_button.setFixedWidth(20)
        self.preview_detach_button.setFixedHeight(20)
        self.preview_detach_button.clicked.connect(self.detachPreviewWindow)
        self.preview_detach_button.setFocusPolicy(Qt.NoFocus)
        header_row_preview.addStretch()
        header_row_preview.addWidget(self.preview_detach_button)
        preview_layout.addLayout(header_row_preview)
        preview_layout.addWidget(self.preview_tabs)
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.generatePreview)
        preview_layout.addWidget(self.preview_button)

        btn_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.runFullConversion)
        self.run_stop_button = QPushButton("Stop")
        self.run_stop_button.setEnabled(False)
        self.run_stop_button.clicked.connect(self.stopConversion)
        btn_row.addStretch()
        btn_row.addWidget(self.run_button)
        btn_row.addWidget(self.run_stop_button)
        btn_row.addStretch()

        # Combine preview panel and run button so the splitter keeps the
        # original layout but allows resizing versus the log output.
        self.preview_container = QWidget()
        pv_lay = QVBoxLayout(self.preview_container)
        pv_lay.setContentsMargins(0, 0, 0, 0)
        pv_lay.setSpacing(6)
        self.preview_scroll = ShrinkableScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setWidget(self.preview_group)
        pv_lay.addWidget(self.preview_scroll)
        pv_lay.addLayout(btn_row)

        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout(log_group)
        self.terminal_cb = QCheckBox("Show output in terminal")
        log_layout.addWidget(self.terminal_cb)
        if sys.platform == "win32":
            # Always show terminal output on Windows and hide the option
            self.terminal_cb.setChecked(True)
            self.terminal_cb.setVisible(False)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.document().setMaximumBlockCount(1000)
        log_layout.addWidget(self.log_text)
        self.spinner_label = QLabel()
        self.spinner_label.setAlignment(Qt.AlignLeft)
        self.spinner_label.hide()
        log_layout.addWidget(self.spinner_label)

        self.left_split.addWidget(self.preview_container)
        self.right_split.addWidget(log_group)

        splitter = QSplitter()
        splitter.addWidget(self.left_split)
        splitter.addWidget(self.right_split)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

        self.tabs.addTab(self.convert_tab, "Converter")

    def _add_preview_path(self, parts, orig_seq):
        """Insert path components into the preview tree, storing ``orig_seq`` on the leaf."""
        # ``parts`` is a sequence of folder/file names comprising a BIDS path.
        parent = self.preview_tree.invisibleRootItem()
        for idx, part in enumerate(parts):
            match = None
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.text(0) == part:
                    match = child
                    break
            if match is None:
                match = QTreeWidgetItem([part, ""])  # second column filled on leaf
                parent.addChild(match)
            parent = match
            if idx == len(parts) - 1:
                parent.setText(1, orig_seq)

    def generatePreview(self):
        logging.info("generatePreview → Building preview tree …")
        """Populate preview tabs based on checked sequences."""
        self.preview_text.clear()
        self.preview_tree.clear()
        multi_study = len(self.study_set) > 1

        selected = []
        for i in range(self.mapping_table.rowCount()):
            if self.mapping_table.item(i, 0).checkState() == Qt.Checked:
                selected.append(self.row_info[i])

        rep_counts = defaultdict(int)
        for info in selected:
            subj_key = info['bids'] if self.use_bids_names else f"sub-{info['given']}"
            key = (subj_key, info['ses'], info['seq'])
            rep_counts[key] += 1

        for info in selected:
            subj = info['bids'] if self.use_bids_names else f"sub-{info['given']}"
            study = info['study']
            ses = info['ses']
            modb = info['modb']
            seq = info['seq']

            # Preview for DWI derivative maps moved to derivatives/
            tag = None
            if modb == 'dwi':
                seq_low = seq.lower()
                for t in ("adc", "fa", "tracew", "colfa"):
                    if t in seq_low:
                        tag = t.upper()
                        break
            if tag:
                path_parts = []
                if multi_study:
                    path_parts.append(study)
                path_parts.extend(["derivatives", DERIVATIVES_PIPELINE_NAME, subj, ses, "dwi"])
                fname_prefix = "_".join([p for p in [subj, ses] if p])
                fname = f"{fname_prefix}_desc-{tag}_dwi.nii.gz"
                full = [p for p in path_parts if p] + [fname]
                self.preview_text.addTopLevelItem(QTreeWidgetItem(["/".join(full), seq]))
                self._add_preview_path(full, seq)
                continue

            prop_dt = info.get('prop_dt')
            prop_base = info.get('prop_base')
            if prop_dt and prop_base:
                path_parts = []
                if multi_study:
                    path_parts.append(study)
                path_parts.extend([subj, ses, prop_dt])
                files = [f"{prop_base}.nii.gz"]
                if prop_base.endswith("_physio"):
                    files = [f"{prop_base}.tsv", f"{prop_base}.json"]
                for fname in files:
                    full = [p for p in path_parts if p] + [fname]
                    self.preview_text.addTopLevelItem(QTreeWidgetItem(["/".join(full), seq]))
                    self._add_preview_path(full, seq)
                continue

            path_parts = []
            if multi_study:
                path_parts.append(study)
            path_parts.extend([subj, ses, modb])

            base_parts = [subj, ses, seq]
            key = (subj, ses, seq)
            if rep_counts[key] > 1 and info['rep']:
                base_parts.append(f"rep-{info['rep']}")
            base = _dedup_parts(*base_parts)

            if modb == "fmap":
                for suffix in ["magnitude1", "magnitude2", "phasediff"]:
                    fname = f"{base}_{suffix}.nii.gz"
                    full = path_parts + [fname]
                    self.preview_text.addTopLevelItem(QTreeWidgetItem(["/".join(full), seq]))
                    self._add_preview_path(full, seq)
            else:
                fname = f"{base}.nii.gz"
                full = path_parts + [fname]
                self.preview_text.addTopLevelItem(QTreeWidgetItem(["/".join(full), seq]))
                self._add_preview_path(full, seq)

        self.preview_text.expandAll()
        self.preview_tree.expandAll()

    def selectDicomDir(self):
        """Select the raw DICOM input directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if directory:
            self.dicom_dir = directory
            self.dicom_dir_edit.setText(directory)

    def selectBIDSOutDir(self):
        """Select (or create) the BIDS output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select/Create BIDS Output Directory")
        if directory:
            self.bids_out_dir = directory
            self.bids_out_edit.setText(directory)
            self.loadExcludePatterns()

    def selectAndLoadTSV(self):
        """Choose an existing TSV and load it into the table."""
        path, _ = QFileDialog.getOpenFileName(self, "Select TSV", self.bids_out_dir or "", "TSV Files (*.tsv)")
        if path:
            self.tsv_path = path
            self.tsv_name_edit.setText(os.path.basename(path))
            self.loadMappingTable()

    def _open_sort_dialog(self) -> None:
        """Display a dialog allowing the user to configure a multi-level sort."""

        if self.mapping_table.rowCount() <= 1:
            QMessageBox.information(self, "Nothing to sort", "Load multiple rows before sorting.")
            return

        headers = []
        for col in range(self.mapping_table.columnCount()):
            header_item = self.mapping_table.horizontalHeaderItem(col)
            if header_item is not None:
                headers.append(header_item.text())

        dialog = MappingSortDialog(headers, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        instructions = dialog.sort_instructions()
        if not instructions:
            QMessageBox.information(self, "No columns selected", "Select at least one column to sort by.")
            return

        self._apply_mapping_sort(instructions)

    def _apply_mapping_sort(self, instructions: list[tuple[str, bool]]) -> None:
        """Reorder the mapping table using ``instructions`` as sort keys."""

        column_lookup: dict[str, int] = {}
        for idx in range(self.mapping_table.columnCount()):
            header_item = self.mapping_table.horizontalHeaderItem(idx)
            if header_item is not None:
                column_lookup[header_item.text()] = idx

        # Translate column names back into indices, ignoring stale entries.
        order = [(column_lookup.get(name), asc) for name, asc in instructions if name in column_lookup]
        order = [(idx, asc) for idx, asc in order if idx is not None]
        if not order:
            return

        row_count = self.mapping_table.rowCount()
        col_count = self.mapping_table.columnCount()
        rows: list[dict[str, Any]] = []
        for row in range(row_count):
            row_items: list[QTableWidgetItem] = []
            sort_values: list[tuple[Any, ...]] = []
            for col in range(col_count):
                item = self.mapping_table.item(row, col)
                if item is not None:
                    cloned = item.clone()
                else:
                    cloned = QTableWidgetItem()
                if col == 0:
                    # Preserve checkbox behaviour in the include column.
                    cloned.setFlags((cloned.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
                    state = item.checkState() if item is not None else Qt.Unchecked
                    cloned.setCheckState(state)
                    sort_value = (0, 1 if state == Qt.Checked else 0)
                else:
                    text = item.text() if item is not None else ""
                    sort_value = self._coerce_sort_value(text)
                row_items.append(cloned)
                sort_values.append(sort_value)
            info = self.row_info[row] if row < len(self.row_info) else {}
            rows.append(
                {
                    "items": row_items,
                    "sort_values": sort_values,
                    "row_info": info,
                    "original_index": row,
                }
            )

        def _compare(left: dict[str, Any], right: dict[str, Any]) -> int:
            for col_idx, ascending in order:
                l_val = left["sort_values"][col_idx]
                r_val = right["sort_values"][col_idx]
                if l_val == r_val:
                    continue
                if l_val < r_val:
                    return -1 if ascending else 1
                return 1 if ascending else -1
            return 0

        sorted_rows = sorted(rows, key=cmp_to_key(_compare))
        new_order = [row["original_index"] for row in sorted_rows]
        if new_order == list(range(len(sorted_rows))):
            return  # Already sorted

        previous_loading = self._loading_mapping_table
        self._loading_mapping_table = True
        self.mapping_table.blockSignals(True)
        self.mapping_table.setRowCount(0)
        for row_data in sorted_rows:
            idx = self.mapping_table.rowCount()
            self.mapping_table.insertRow(idx)
            for col_idx, item in enumerate(row_data["items"]):
                self.mapping_table.setItem(idx, col_idx, item)
        self.mapping_table.blockSignals(False)
        self._loading_mapping_table = previous_loading

        # Keep auxiliary structures synchronised with the new row order.
        self.row_info = [row["row_info"] for row in sorted_rows]
        if self.inventory_df is not None:
            try:
                self.inventory_df = self.inventory_df.iloc[new_order].reset_index(drop=True)
            except Exception:
                pass

        self._rebuild_lookup_maps()
        self._schedule_mapping_refresh()

        if hasattr(self, "log_text"):
            summary = ", ".join(
                f"{name} ({'Ascending' if asc else 'Descending'})" for name, asc in instructions
            )
            self.log_text.append(f"Sorted metadata table by {summary}.")

    def _coerce_sort_value(value: str) -> tuple[Any, ...]:
        """Return a comparable tuple for ``value`` suitable for sorting."""

        text = value.strip()
        if not text:
            return (5, "")

        try:
            numeric = Decimal(text)
            return (1, numeric)
        except (InvalidOperation, ValueError):
            pass

        fmt = guess_datetime_format(text)
        if fmt:
            try:
                dt_value = pd.to_datetime(text, errors="raise")
                return (2, dt_value.to_pydatetime())
            except Exception:
                pass

        lowered = text.lower()
        return (3, lowered, text)

    def detachTSVWindow(self):
        """Detach the scanned data viewer into a separate window."""
        if getattr(self, "tsv_dialog", None):
            self.tsv_dialog.activateWindow()
            return
        self.tsv_dialog = QDialog(self, flags=Qt.Window)
        self.tsv_dialog.setWindowTitle("Scanned data viewer")
        lay = QVBoxLayout(self.tsv_dialog)
        self.tsv_container.setParent(None)
        lay.addWidget(self.tsv_container)
        self.tsv_dialog.finished.connect(self._reattachTSVWindow)
        self.tsv_dialog.showMaximized()

    def _reattachTSVWindow(self, *args):
        self.tsv_container.setParent(None)
        self.left_split.insertWidget(0, self.tsv_container)
        self.tsv_dialog = None

    def detachFilterWindow(self):
        """Detach the filter panel into a separate window."""
        if getattr(self, "filter_dialog", None):
            self.filter_dialog.activateWindow()
            return
        self.filter_dialog = QDialog(self, flags=Qt.Window)
        self.filter_dialog.setWindowTitle("Filter")
        lay = QVBoxLayout(self.filter_dialog)
        self.filter_container.setParent(None)
        lay.addWidget(self.filter_container)
        self.filter_dialog.finished.connect(self._reattachFilterWindow)
        self.filter_dialog.showMaximized()

    def _reattachFilterWindow(self, *args):
        self.filter_container.setParent(None)
        # Insert after the TSV panel but before preview_container
        self.right_split.insertWidget(0, self.filter_container)
        self.filter_dialog = None

    def detachPreviewWindow(self):
        """Detach the preview panel into a separate window."""
        if getattr(self, "preview_dialog", None):
            self.preview_dialog.activateWindow()
            return
        self.preview_dialog = QDialog(self, flags=Qt.Window)
        self.preview_dialog.setWindowTitle("Preview")
        lay = QVBoxLayout(self.preview_dialog)
        self.preview_container.setParent(None)
        lay.addWidget(self.preview_container)
        self.preview_dialog.finished.connect(self._reattachPreviewWindow)
        self.preview_dialog.showMaximized()

    def _reattachPreviewWindow(self, *args):
        self.preview_container.setParent(None)
        if self.left_split.indexOf(self.tsv_container) == -1:
            self.left_split.addWidget(self.preview_container)
        else:
            self.left_split.insertWidget(1, self.preview_container)
        self.preview_dialog = None

    def runInventory(self):
        logging.info("runInventory → Generating TSV …")
        """
        Scan DICOMs and generate subject_summary.tsv in the selected output directory.
        """
        if not self.dicom_dir or not os.path.isdir(self.dicom_dir):
            QMessageBox.warning(self, "Invalid DICOM Directory", "Please select a valid DICOM input directory.")
            return
        if not self.bids_out_dir:
            QMessageBox.warning(self, "No BIDS Output Directory", "Please select a BIDS output directory.")
            return

        os.makedirs(self.bids_out_dir, exist_ok=True)

        name = self.tsv_name_edit.text().strip() or "subject_summary.tsv"
        self.tsv_path = os.path.join(self.bids_out_dir, name)

        # Run dicom_inventory asynchronously
        if self.inventory_process and self.inventory_process.state() != QProcess.NotRunning:
            return

        # Remember to reapply the suffix dictionary automatically once the new
        # scan results have been loaded.  This ensures custom patterns take
        # effect immediately after a rescan without requiring another manual
        # save on the suffix tab.
        self._apply_sequence_on_load = self.use_custom_patterns_box.isChecked()

        # Clear the log so each scan run starts with a fresh history for the
        # user.
        self.log_text.clear()
        self.log_text.append("Starting TSV generation…")
        self.tsv_button.setEnabled(False)
        self.tsv_stop_button.setEnabled(True)
        self._start_spinner("Scanning files")
        self.inventory_process = QProcess(self)
        if self.terminal_cb.isChecked():
            # Forward stdout and stderr when the user wants to see terminal output
            self.inventory_process.setProcessChannelMode(QProcess.ForwardedChannels)
        else:
            # Discard output to avoid hangs on Windows when not showing the terminal
            self.inventory_process.setStandardOutputFile(QProcess.nullDevice())
            self.inventory_process.setStandardErrorFile(QProcess.nullDevice())
        self.inventory_process.finished.connect(self._inventoryFinished)
        args = [
            "-m",
            "bids_manager.dicom_inventory",
            self.dicom_dir,
            self.tsv_path,
            "--jobs",
            str(self.num_cpus),
        ]
        self.inventory_process.start(sys.executable, args)

    def _find_conflicting_studies(self, root_dir: str, n_jobs: int = 1) -> dict:
        """Return folders containing more than one StudyInstanceUID.

        Parameters
        ----------
        root_dir : str
            Top level directory that may contain mixed-session folders.

        Returns
        -------
        dict
            Mapping of folder path → {study_uid: [file1, file2, ...]} for
            folders that contain DICOMs from multiple sessions.
        """

        conflicts: dict[str, dict[str, list[str]]] = {}

        # Gather the folders containing DICOM files so we can evaluate them in
        # parallel without paying the scheduling cost for empty directories.
        folders_to_scan: List[Tuple[str, List[str]]] = []
        for folder, _dirs, files in os.walk(root_dir):
            dicom_files: List[str] = []
            for fname in files:
                fpath = os.path.join(folder, fname)
                if is_dicom_file(fpath):
                    dicom_files.append(fpath)
            if dicom_files:
                folders_to_scan.append((folder, dicom_files))

        def _scan_folder(data: Tuple[str, List[str]]) -> Optional[Tuple[str, Dict[str, List[str]]]]:
            folder, dicom_files = data
            study_map: Dict[str, List[str]] = {}
            for fpath in dicom_files:
                try:
                    ds = pydicom.dcmread(
                        fpath,
                        stop_before_pixels=True,
                        specific_tags=["StudyInstanceUID"],
                    )
                    uid = str(getattr(ds, "StudyInstanceUID", "")).strip()
                except Exception:
                    # Maintain the previous best-effort behaviour: unreadable
                    # files are skipped silently so the scan keeps running.
                    continue
                study_map.setdefault(uid, []).append(fpath)
            if len(study_map) > 1:
                return folder, study_map
            return None

        workers = max(1, n_jobs)
        if workers == 1:
            results = (_scan_folder(entry) for entry in folders_to_scan)
        else:
            # ``Parallel`` confines the evaluation to the configured number of
            # workers, matching the CPU limit used for the initial DICOM scan.
            results = Parallel(n_jobs=workers)(
                delayed(_scan_folder)(entry) for entry in folders_to_scan
            )

        for result in results:
            if result is None:
                continue
            folder, study_map = result
            conflicts[folder] = study_map

        return conflicts

    def _reorganize_conflicting_sessions(self, conflicts: dict) -> None:
        """Move files for **all** sessions into separate subfolders.

        This method ensures that each unique ``StudyInstanceUID`` found within a
        folder is placed in its own subdirectory.  Previously only the
        additional sessions were moved, leaving the first session's files in the
        root folder, which could lead to HeuDiConv processing multiple sessions
        together and crashing.  By relocating every session we guarantee a clean
        one-session-per-folder layout.

        Parameters
        ----------
        conflicts : dict
            Output of :meth:`_find_conflicting_studies` mapping folder paths to
            ``StudyInstanceUID`` → list of files.
        """

        for folder, uid_map in conflicts.items():
            # Iterate over each StudyInstanceUID and move its files into a
            # unique ``sessionX`` directory.
            for idx, (uid, files) in enumerate(uid_map.items(), start=1):
                # Determine a unique destination directory.  We increment the
                # numeric suffix if a folder with the same name already exists.
                new_dir = os.path.join(folder, f"session{idx}")
                suffix = idx
                while os.path.exists(new_dir):
                    suffix += 1
                    new_dir = os.path.join(folder, f"session{suffix}")
                os.makedirs(new_dir, exist_ok=True)

                # Move each file belonging to the current UID into ``new_dir``.
                for fpath in files:
                    shutil.move(fpath, os.path.join(new_dir, os.path.basename(fpath)))

                self.log_text.append(
                    f"Moved {len(files)} files with StudyInstanceUID {uid} to {new_dir}."
                )

    def _cleanup_conflict_worker(self) -> None:
        """Release resources used by the background conflict scanner."""

        if self._conflict_worker is not None:
            self._conflict_worker.deleteLater()
            self._conflict_worker = None
        if self._conflict_thread is not None:
            if self._conflict_thread.isRunning():
                self._conflict_thread.quit()
                self._conflict_thread.wait()
            self._conflict_thread.deleteLater()
            self._conflict_thread = None

    def _start_conflict_scan(self) -> None:
        """Check for mixed sessions without blocking the GUI thread."""

        if not self.dicom_dir or not os.path.isdir(self.dicom_dir):
            # Nothing to scan – proceed directly to loading the generated TSV.
            self.log_text.append("TSV generation finished.")
            self.loadMappingTable()
            return

        # Avoid launching multiple background scanners simultaneously.
        if self._conflict_thread is not None:
            if not self._conflict_thread.isRunning():
                self._cleanup_conflict_worker()
            else:
                return

        self._start_spinner("Checking sessions")
        self.log_text.append("Checking for multiple sessions in scan folders…")

        self._conflict_worker = _ConflictScannerWorker(
            self.dicom_dir,
            self._find_conflicting_studies,
            self.num_cpus,
        )
        self._conflict_thread = QThread(self)
        self._conflict_worker.moveToThread(self._conflict_thread)
        self._conflict_thread.started.connect(self._conflict_worker.run)
        self._conflict_worker.finished.connect(self._on_conflict_scan_finished)
        self._conflict_worker.failed.connect(self._on_conflict_scan_failed)
        self._conflict_thread.start()

    def _on_conflict_scan_finished(self, conflicts: dict) -> None:
        """Handle successful completion of the background conflict scan."""

        self._stop_spinner()
        self._cleanup_conflict_worker()
        if conflicts:
            folders = "\n".join(conflicts.keys())
            msg = (
                "Multiple sessions were detected in the following folders:\n"
                f"{folders}\n\n"
                "Would you like to move each session into its own subfolder?"
            )
            resp = QMessageBox.question(
                self,
                "Multiple sessions detected",
                msg,
                QMessageBox.Yes | QMessageBox.No,
            )
            if resp == QMessageBox.Yes:
                self._reorganize_conflicting_sessions(conflicts)
                # Re-run the inventory now that the folders have been separated.
                self.runInventory()
                return

        self.log_text.append("TSV generation finished.")
        self.loadMappingTable()

    def _on_conflict_scan_failed(self, error_message: str) -> None:
        """Fallback when conflict detection fails for any reason."""

        self._stop_spinner()
        self._cleanup_conflict_worker()
        logging.error("Conflict detection failed: %s", error_message)
        self.log_text.append("Failed to check for mixed sessions; continuing anyway.")
        self.log_text.append("TSV generation finished.")
        self.loadMappingTable()

    def _inventoryFinished(self):
        ok = self.inventory_process.exitCode() == 0 if self.inventory_process else False
        self.inventory_process = None
        self.tsv_button.setEnabled(True)
        self.tsv_stop_button.setEnabled(False)
        self._stop_spinner()
        if ok:
            self._start_conflict_scan()
        else:
            self.log_text.append("TSV generation failed.")
            # Avoid auto-applying the suffix dictionary when the scan failed;
            # the mapping table will not be refreshed and we should not reuse
            # the pending flag for future loads.
            self._apply_sequence_on_load = False

    def stopInventory(self):
        if self.inventory_process and self.inventory_process.state() != QProcess.NotRunning:
            pid = int(self.inventory_process.processId())
            _terminate_process_tree(pid)
            self.inventory_process = None
            self.tsv_button.setEnabled(True)
            self.tsv_stop_button.setEnabled(False)
            self._stop_spinner()
            self.log_text.append("TSV generation cancelled.")
            # Reset pending suffix reapplication because no new TSV will be
            # loaded after a cancellation.
            self._apply_sequence_on_load = False

    def applyMappingChanges(self):
        """Save edits in the scanned data table back to the TSV and refresh."""
        if not self.tsv_path or not os.path.isfile(self.tsv_path):
            return
        try:
            df = pd.read_csv(self.tsv_path, sep="\t", keep_default_na=False)
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to load TSV: {exc}")
            return
        if df.shape[0] != self.mapping_table.rowCount():
            QMessageBox.warning(self, "Error", "Row count mismatch")
            return
        for i in range(self.mapping_table.rowCount()):
            df.at[i, "include"] = 1 if self.mapping_table.item(i, 0).checkState() == Qt.Checked else 0
            df.at[i, "source_folder"] = self.mapping_table.item(i, 1).text()
            study_item = self.mapping_table.item(i, 2)
            study_text = study_item.text() if study_item is not None else ""
            df.at[i, "StudyDescription"] = normalize_study_name(study_text.strip())
            df.at[i, "FamilyName"] = self.mapping_table.item(i, 3).text()
            df.at[i, "PatientID"] = self.mapping_table.item(i, 4).text()
            df.at[i, "BIDS_name"] = self.mapping_table.item(i, 5).text()
            df.at[i, "subject"] = self.mapping_table.item(i, 6).text()
            df.at[i, "GivenName"] = self.mapping_table.item(i, 7).text()
            df.at[i, "session"] = self.mapping_table.item(i, 8).text()
            df.at[i, "sequence"] = self.mapping_table.item(i, 9).text()
            df.at[i, "Proposed BIDS name"] = self.mapping_table.item(i, 10).text()
            df.at[i, "series_uid"] = self.mapping_table.item(i, 11).text()
            df.at[i, "acq_time"] = self.mapping_table.item(i, 12).text()
            df.at[i, "rep"] = self.mapping_table.item(i, 13).text()
            df.at[i, "modality"] = self.mapping_table.item(i, 14).text()
            df.at[i, "modality_bids"] = self.mapping_table.item(i, 15).text()

        # When editing the scanned data table we assume the user knows what
        # they are doing, so we do not enforce BIDS naming rules or uniqueness
        # here. Validation is still performed when editing via the naming table
        # and filter fields.
        try:
            df.to_csv(self.tsv_path, sep="\t", index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save TSV: {exc}")
            return
        self.loadMappingTable()

    def generateUniqueIDs(self):
        """Assign random 3-letter/3-digit IDs to subjects without an identifier."""
        # Load previously assigned IDs from all studies in the output directory
        existing: dict[str, dict[str, str]] = {}
        existing_ids: set[str] = set()
        out_dir = Path(self.bids_out_dir)
        if out_dir.is_dir():
            for study_dir in out_dir.iterdir():
                if not study_dir.is_dir():
                    continue
                s_path = study_dir / ".bids_manager" / "subject_summary.tsv"
                if s_path.exists():
                    try:
                        sdf = pd.read_csv(s_path, sep="\t", keep_default_na=False)
                        for _, row in sdf.iterrows():
                            study_desc = str(row.get("StudyDescription", "")).strip()
                            bids_name = str(row.get("BIDS_name", "")).strip()
                            sid = str(row.get("subject", "")).strip() or str(row.get("GivenName", "")).strip()
                            if study_desc and bids_name and sid:
                                existing.setdefault(study_desc, {})[bids_name] = sid
                                existing_ids.add(sid)
                    except Exception:
                        pass

        id_map: dict[tuple[str, str], str] = {}
        for i in range(self.mapping_table.rowCount()):
            bids = self.mapping_table.item(i, 5).text().strip()
            study = self.mapping_table.item(i, 2).text().strip()
            if not bids:
                continue

            subj_item = self.mapping_table.item(i, 6)
            given_item = self.mapping_table.item(i, 7)

            sid = None
            prior = existing.get(study, {}).get(bids)
            if prior and subj_item.text().strip() == "" and given_item.text().strip() == "":
                resp = QMessageBox.question(
                    self,
                    "Subject exists",
                    "This subject already exist in the study. Would you like to use the same unique ID?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if resp == QMessageBox.Yes:
                    sid = prior

            if sid is None:
                key = (study, bids)
                if key not in id_map:
                    id_map[key] = _random_subject_id(existing_ids | set(id_map.values()))
                sid = id_map[key]

            if subj_item.text().strip() == "":
                subj_item.setText(sid)
            if given_item.text().strip() == "":
                given_item.setText(sid)
            self.row_info[i]['given'] = given_item.text()
            existing_ids.add(sid)

        self._rebuild_lookup_maps()
        self._schedule_mapping_refresh()
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.generatePreview)
        QTimer.singleShot(0, self._updateDetectRepeatEnabled)
        QTimer.singleShot(0, self._updateMappingControlsEnabled)
        QTimer.singleShot(0, self._auto_apply_existing_study_mappings)

    def _updateDetectRepeatEnabled(self, _item=None):
        """Enable repeat detection when BIDS and Given names are filled."""
        if not hasattr(self, "tsv_detect_rep_action"):
            return
        enabled = self.mapping_table.rowCount() > 0
        if enabled:
            for r in range(self.mapping_table.rowCount()):
                bids = self.mapping_table.item(r, 5)
                given = self.mapping_table.item(r, 7)
                if bids is None or given is None or not bids.text().strip() or not given.text().strip():
                    enabled = False
                    break
        self.tsv_detect_rep_action.setEnabled(enabled)

    def _is_visual_only_sequence(sequence: str, modality: str = "") -> bool:
        """Return ``True`` when the sequence should remain view-only."""

        seq_lower = (sequence or "").lower()
        mod_lower = (modality or "").lower()
        # The check is intentionally substring-based so variations such as
        # "Phoenix Report" or "Scout Image" are also captured.
        visual_tokens = ("phoenix report", "scout", "report")
        return any(tok in seq_lower for tok in visual_tokens) or any(
            tok in mod_lower for tok in visual_tokens
        )

    def _apply_visual_only_rules(self, row: int) -> None:
        """Disable inclusion toggles for rows that must stay view-only."""

        if not (0 <= row < self.mapping_table.rowCount()):
            return
        include_item = self.mapping_table.item(row, 0)
        if include_item is None or row >= len(self.row_info):
            return

        is_visual_only = bool(self.row_info[row].get("visual_only"))
        # Temporarily silence table signals while we adjust checkbox flags so
        # we do not trigger downstream refreshes.
        prev_block = self.mapping_table.signalsBlocked()
        self.mapping_table.blockSignals(True)
        try:
            if is_visual_only:
                include_item.setCheckState(Qt.Unchecked)
                include_item.setFlags(
                    (include_item.flags() & ~Qt.ItemIsEditable)
                    & ~Qt.ItemIsUserCheckable
                    & ~Qt.ItemIsEnabled
                )
                include_item.setToolTip(
                    "Reports and scout images are shown for reference only."
                )
            else:
                include_item.setFlags(
                    (include_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                    & ~Qt.ItemIsEditable
                )
                include_item.setToolTip("")
        finally:
            self.mapping_table.blockSignals(prev_block)

    def _count_non_visual_repeats(self) -> int:
        """Return the number of repeated acquisitions excluding visual-only rows."""

        count = 0
        for info in self.row_info:
            if info.get("visual_only"):
                continue
            rep_val = str(info.get("rep") or "").strip()
            if rep_val.isdigit() and int(rep_val) > 1:
                count += 1
        return count

    def _maybe_notify_repeats(self) -> None:
        """Inform the user when new repeats are detected and sync selection."""

        if not hasattr(self, "last_rep_box"):
            return
        repeat_count = self._count_non_visual_repeats()
        if repeat_count > 0:
            if not self.last_rep_box.isChecked():
                self.last_rep_box.setChecked(True)
            if self._last_repeat_count == 0:
                QMessageBox.information(
                    self,
                    "Repeated sequences detected",
                    (
                        "Repeated acquisitions were found. The \"Only last repeats\" "
                        "option is now active to keep the latest instance selected."
                    ),
                )
        self._last_repeat_count = repeat_count

    def _auto_apply_existing_study_mappings(self) -> None:
        """Queue a silent sync of BIDS names with existing output datasets."""

        # ``QTimer.singleShot`` triggers this helper outside of the current
        # signal handler, so keep the method lightweight and delegate the heavy
        # lifting to the shared implementation below.
        self._apply_existing_study_mappings(silent=True)

    def _apply_existing_study_mappings(self, *, silent: bool = True) -> None:
        """Align BIDS subject names with prior conversions stored on disk.

        Parameters
        ----------
        silent
            When ``True`` the method skips modal warnings if the BIDS output
            directory has not been configured yet.  This is the default for the
            automatic calls that happen after scans or when generating IDs.
        """

        out_dir = Path(self.bids_out_dir or "")
        if not out_dir.is_dir():
            if silent:
                logging.info(
                    "Skipping existing-study sync because no BIDS output "
                    "directory is available."
                )
            else:
                QMessageBox.warning(
                    self,
                    "No BIDS Output Directory",
                    "Please select a BIDS output directory.",
                )
            return

        if self.naming_table.rowCount() == 0:
            return

        # Gather the studies present in the naming table so we only touch the
        # datasets that are relevant for the current scan.
        studies: set[str] = set()
        for row in range(self.naming_table.rowCount()):
            study_item = self.naming_table.item(row, 0)
            if study_item is None:
                continue
            study_text = study_item.text().strip()
            if study_text:
                studies.add(study_text)

        if not studies:
            return

        # ``existing_by_given`` and ``existing_by_uid`` map (study, key) pairs
        # to the BIDS names already present on disk.  We keep track of both the
        # ``GivenName`` (which becomes the generated unique ID) and the
        # ``subject`` column written by the converter so that we can preserve
        # names even if only one identifier is available.
        existing_by_given: dict[tuple[str, str], str] = {}
        existing_by_uid: dict[tuple[str, str], str] = {}
        used_by_study: dict[str, set[str]] = {}
        has_existing: dict[str, bool] = {}

        for study in studies:
            safe = _safe_stem(str(study))
            s_path = out_dir / safe / ".bids_manager" / "subject_summary.tsv"
            has_existing[study] = s_path.exists()
            if not s_path.exists():
                continue
            try:
                df = pd.read_csv(s_path, sep="\t", keep_default_na=False)
            except Exception as exc:  # pragma: no cover - runtime resilience
                logging.warning("Failed to read %s: %s", s_path, exc)
                continue

            for _, row in df.iterrows():
                given_val = str(row.get("GivenName", "")).strip()
                bids_val = str(row.get("BIDS_name", "")).strip()
                uid_val = str(row.get("subject", "")).strip()
                if bids_val:
                    used_by_study.setdefault(study, set()).add(bids_val)
                if given_val and bids_val:
                    existing_by_given[(study, given_val)] = bids_val
                if uid_val and bids_val:
                    existing_by_uid[(study, uid_val)] = bids_val

        if not any(has_existing.values()):
            # Nothing to reconcile – leave the manually assigned names untouched.
            return

        self.naming_table.blockSignals(True)
        self.mapping_table.blockSignals(True)

        for row in range(self.naming_table.rowCount()):
            study_item = self.naming_table.item(row, 0)
            given_item = self.naming_table.item(row, 1)
            bids_item = self.naming_table.item(row, 2)
            if None in (study_item, given_item, bids_item):
                continue

            study = study_item.text().strip()
            given = given_item.text().strip()
            current = bids_item.text().strip()
            if not study:
                continue

            # Derive the subject unique ID from the mapping table if present.
            subject_uid = ""
            for idx, info in enumerate(self.row_info):
                if info['study'] == study and info['given'] == given:
                    subj_col = self.mapping_table.item(idx, 6)
                    if subj_col is not None:
                        subject_uid = subj_col.text().strip()
                    break

            used = used_by_study.setdefault(study, set()).copy()

            mapped = None
            if subject_uid:
                mapped = existing_by_uid.get((study, subject_uid))
            if mapped is None and given:
                mapped = existing_by_given.get((study, given))

            if mapped:
                new_bids = mapped
            elif not has_existing.get(study, False):
                new_bids = current
            else:
                if current and current not in used:
                    new_bids = current
                else:
                    new_bids = _next_numeric_id(used)

            if new_bids != current:
                bids_item.setText(new_bids)

            # Update downstream structures so future duplicate detection and
            # preview generation operate on the adjusted BIDS identifiers.
            used_by_study.setdefault(study, set()).add(new_bids)
            for idx, info in enumerate(self.row_info):
                if info['study'] == study and info['given'] == given:
                    info['bids'] = new_bids
                    table_item = self.mapping_table.item(idx, 5)
                    if table_item is not None:
                        table_item.setText(new_bids)

            key_for_map = given or subject_uid
            if key_for_map:
                self.existing_maps.setdefault(study, {})[key_for_map] = new_bids
            if subject_uid and subject_uid != given:
                self.existing_maps.setdefault(study, {})[subject_uid] = new_bids
            self.existing_used.setdefault(study, set()).add(new_bids)

        self.naming_table.blockSignals(False)
        self.mapping_table.blockSignals(False)

        self._rebuild_lookup_maps()
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.generatePreview)
        QTimer.singleShot(0, self._updateDetectRepeatEnabled)
        QTimer.singleShot(0, self._updateMappingControlsEnabled)

    def _updateMappingControlsEnabled(self):
        """Enable controls that require scanned data."""
        if not hasattr(self, "tsv_generate_ids_action"):
            return
        has_data = self.mapping_table.rowCount() > 0
        self.tsv_generate_ids_action.setEnabled(has_data)
        if hasattr(self, "tsv_sort_action"):
            self.tsv_sort_action.setEnabled(has_data)
        if hasattr(self, "tsv_save_action"):
            self.tsv_save_action.setEnabled(has_data)
        self.last_rep_box.setEnabled(has_data)
        self.name_choice.setEnabled(has_data)
        if not has_data:
            self.last_rep_box.setChecked(False)

    def _sync_row_info_from_table(self, row: int) -> None:
        """Update cached ``row_info`` details after an in-place table edit."""
        if not (0 <= row < len(self.row_info)):
            return

        def _text(col: int) -> str:
            item = self.mapping_table.item(row, col)
            return item.text().strip() if item is not None else ""

        info = self.row_info[row]
        info['study'] = _text(2)
        info['bids'] = _text(5)
        info['given'] = _text(7)
        info['ses'] = _text(8)
        info['seq'] = _text(9)
        info['rep'] = _text(13)
        info['mod'] = _text(14)
        info['modb'] = _text(15)
        info['visual_only'] = self._is_visual_only_sequence(info['seq'], info['mod'])
        self._apply_visual_only_rules(row)

    def _schedule_mapping_refresh(self) -> None:
        """Queue UI updates that reflect the current mapping table."""
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.generatePreview)
        QTimer.singleShot(0, self._updateDetectRepeatEnabled)
        QTimer.singleShot(0, self._updateMappingControlsEnabled)

    def _update_study_set(self) -> None:
        """Recompute list of studies present in the mapping table."""
        self.study_set = set()
        for r in range(self.mapping_table.rowCount()):
            study_item = self.mapping_table.item(r, 2)
            if study_item is None:
                continue
            study_text = study_item.text().strip()
            if study_text:
                self.study_set.add(study_text)

    def _onMappingItemChanged(self, item):
        """Handle edits in the scanned data table."""
        if self._loading_mapping_table or item is None:
            return

        column = item.column()
        row = item.row()

        if column == 2:
            raw_text = item.text().strip()
            cleaned = normalize_study_name(raw_text)

            if cleaned != item.text():
                # Temporarily block signals so updating the text does not trigger
                # additional ``itemChanged`` notifications.
                self.mapping_table.blockSignals(True)
                item.setText(cleaned)
                self.mapping_table.blockSignals(False)

            self._sync_row_info_from_table(row)

            bids_item = self.mapping_table.item(row, 5)
            if bids_item is not None:
                bids_item.setData(Qt.UserRole, cleaned)

            self._update_study_set()
            self._rebuild_lookup_maps()
            self._schedule_mapping_refresh()
            return

        if column == 0:
            # Checkbox state changed; refresh dependent views.
            self._schedule_mapping_refresh()
            return

        if column in {5, 7, 8, 9, 13, 14, 15}:
            self._sync_row_info_from_table(row)
            self._rebuild_lookup_maps()
            self._schedule_mapping_refresh()

    def detectRepeatedSequences(self):
        """Detect repeated sequences within each subject and assign numbers."""
        if self.mapping_table.rowCount() == 0:
            return

        rows = []
        for i in range(self.mapping_table.rowCount()):
            rows.append({
                'StudyDescription': self.mapping_table.item(i, 2).text().strip(),
                'BIDS_name': self.mapping_table.item(i, 5).text().strip(),
                'session': self.mapping_table.item(i, 8).text().strip(),
                'modality_bids': self.mapping_table.item(i, 15).text().strip(),
                'modality': self.mapping_table.item(i, 14).text().strip(),
                'sequence': self.mapping_table.item(i, 9).text().strip(),
                'acq_time': self.mapping_table.item(i, 12).text().strip(),
            })

        df = pd.DataFrame(rows)
        df['acq_sort'] = pd.to_numeric(df['acq_time'].str.replace(':', ''), errors='coerce')
        key_cols = ['StudyDescription', 'BIDS_name', 'session', 'modality_bids', 'modality', 'sequence']
        df.sort_values(['acq_sort'], inplace=True)
        df['rep'] = df.groupby(key_cols).cumcount() + 1
        counts = df.groupby(key_cols)['rep'].transform('count')
        df.loc[counts == 1, 'rep'] = ''
        df.loc[(counts > 1) & (df['rep'] == 1), 'rep'] = ''

        for i in range(self.mapping_table.rowCount()):
            val = df.at[i, 'rep']
            self.mapping_table.item(i, 13).setText(str(val) if str(val) else '')
            self.row_info[i]['rep'] = str(val) if str(val) else ''

        self._rebuild_lookup_maps()
        self._maybe_notify_repeats()
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.generatePreview)

    def scanExistingStudies(self):
        """Manually trigger a sync with existing converted studies."""

        # Keep a public wrapper so automated tests and power users can still
        # force a rescan from the console if needed.  The heavy lifting is
        # performed by :meth:`_apply_existing_study_mappings` which is also
        # used by the automated workflows.
        self._apply_existing_study_mappings(silent=False)

    def loadMappingTable(self):
        logging.info("loadMappingTable → Loading TSV into table …")
        """
        Load the generated TSV into the mapping_table for user editing.
        Columns: include, source_folder, StudyDescription, FamilyName,
        PatientID, BIDS_name, subject, GivenName, session, sequence,
        Proposed BIDS name, series_uid, acq_time, rep, modality, modality_bids
        """
        if not self.tsv_path or not os.path.isfile(self.tsv_path):
            return

        self._loading_mapping_table = True
        try:
            df = pd.read_csv(self.tsv_path, sep="\t", keep_default_na=False)

            # Respect custom suffix patterns when building preview names by
            # re-deriving modalities with the active dictionary before
            # computing BIDS proposals.  This keeps freshly scanned tables in
            # sync with the "Use custom suffix patterns" toggle without
            # requiring a manual "Save" on the suffix tab first.
            if self.use_custom_patterns_box.isChecked():
                for idx, row in df.iterrows():
                    seq = str(row.get("sequence") or "")
                    mod = dicom_inventory.guess_modality(seq)
                    modb = dicom_inventory.modality_to_container(mod)
                    df.at[idx, "modality"] = mod
                    df.at[idx, "modality_bids"] = modb

            preview_map = _compute_bids_preview(df, self._schema)
            df["proposed_datatype"] = [preview_map.get(i, ("", ""))[0] for i in df.index]
            df["proposed_basename"] = [preview_map.get(i, ("", ""))[1] for i in df.index]

            def _prop_path(r):
                base = r.get("proposed_basename")
                dt = r.get("proposed_datatype")
                if not base:
                    return ""
                ext = ".tsv" if str(base).endswith("_physio") else ".nii.gz"
                return f"{dt}/{base}{ext}"

            df["Proposed BIDS name"] = df.apply(_prop_path, axis=1)
            self.inventory_df = df

            # ----- load existing mappings without altering the TSV -----
            self.existing_maps = {}
            self.existing_used = {}
            studies = df["StudyDescription"].fillna("").unique()
            for study in studies:
                safe = _safe_stem(str(study))
                mpath = Path(self.bids_out_dir) / safe / ".bids_manager" / "subject_mapping.tsv"
                mapping = {}
                used = set()
                if mpath.exists():
                    try:
                        mdf = pd.read_csv(mpath, sep="\t", keep_default_na=False)
                        mapping = dict(zip(mdf["GivenName"].astype(str), mdf["BIDS_name"].astype(str)))
                        used = set(mapping.values())
                    except Exception:
                        pass
                # Store mapping info so we can validate name edits later on
                self.existing_maps[study] = mapping
                self.existing_used[study] = used

            self.study_set.clear()
            self.modb_rows.clear()
            self.mod_rows.clear()
            self.seq_rows.clear()
            self.study_rows.clear()
            self.subject_rows.clear()
            self.session_rows.clear()
            self.spec_modb_rows.clear()
            self.spec_mod_rows.clear()
            self.spec_seq_rows.clear()
            self.row_info = []

            # Populate table rows
            self.mapping_table.setRowCount(0)

            def _clean(val):
                """Return string representation of val or empty string for NaN."""
                return "" if pd.isna(val) else str(val)

            for _, row in df.iterrows():
                r = self.mapping_table.rowCount()
                self.mapping_table.insertRow(r)
                include_item = QTableWidgetItem()
                include_item.setFlags(
                    (include_item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable
                )
                include_item.setCheckState(
                    Qt.Checked if row.get('include', 1) == 1 else Qt.Unchecked
                )
                self.mapping_table.setItem(r, 0, include_item)

                src_item = QTableWidgetItem(_clean(row.get('source_folder')))
                src_item.setFlags(src_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 1, src_item)

                study_raw = _clean(row.get('StudyDescription'))
                study = normalize_study_name(study_raw)

                study_item = QTableWidgetItem(study)
                study_item.setFlags(study_item.flags() | Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 2, study_item)

                family_item = QTableWidgetItem(_clean(row.get('FamilyName')))
                family_item.setFlags(family_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 3, family_item)

                patient_item = QTableWidgetItem(_clean(row.get('PatientID')))
                patient_item.setFlags(patient_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 4, patient_item)

                bids_name = _clean(row.get('BIDS_name'))
                bids_item = QTableWidgetItem(bids_name)
                bids_item.setFlags(bids_item.flags() | Qt.ItemIsEditable)
                bids_item.setData(Qt.UserRole, study)
                self.study_set.add(study)
                self.mapping_table.setItem(r, 5, bids_item)

                subj_item = QTableWidgetItem(_clean(row.get('subject')))
                # Subject identifiers come from generated mappings or utilities
                # such as ``generateUniqueIDs``; keep them read-only in the
                # scanned data viewer so users do not alter them manually.
                subj_item.setFlags(subj_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 6, subj_item)

                given_item = QTableWidgetItem(_clean(row.get('GivenName')))
                # "GivenName" (the generated unique ID) must not be edited
                # directly in the scanned data table to avoid inconsistencies
                # with the subject mapping utilities.
                given_item.setFlags(given_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 7, given_item)

                session = _clean(row.get('session'))
                ses_item = QTableWidgetItem(session)
                ses_item.setFlags(ses_item.flags() | Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 8, ses_item)

                seq_item = QTableWidgetItem(_clean(row.get('sequence')))
                # Preserve the original DICOM sequence information as read-only
                # so the viewer remains a faithful reflection of the scan data.
                seq_item.setFlags(seq_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 9, seq_item)

                preview_item = QTableWidgetItem(_clean(row.get('Proposed BIDS name')))
                preview_item.setFlags(preview_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 10, preview_item)

                uid_item = QTableWidgetItem(_clean(row.get('series_uid')))
                uid_item.setFlags(uid_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 11, uid_item)

                acq_item = QTableWidgetItem(_clean(row.get('acq_time')))
                acq_item.setFlags(acq_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 12, acq_item)

                rep_item = QTableWidgetItem(_clean(row.get('rep')))
                # Allow editing the repeat number directly in the table
                rep_item.setFlags(rep_item.flags() | Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 13, rep_item)

                mod_item = QTableWidgetItem(_clean(row.get('modality')))
                mod_item.setFlags(mod_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 14, mod_item)

                modb = _clean(row.get('modality_bids'))
                modb_item = QTableWidgetItem(modb)
                modb_item.setFlags(modb_item.flags() | Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 15, modb_item)

                mod = _clean(row.get('modality'))
                seq = _clean(row.get('sequence'))
                run = _clean(row.get('rep'))
                given = _clean(row.get('GivenName'))
                prop_dt = _clean(row.get('proposed_datatype'))
                prop_base = _clean(row.get('proposed_basename'))
                self.row_info.append({
                    'study': study,
                    'bids': bids_name,
                    'given': given,
                    'ses': session,
                    'modb': modb,
                    'mod': mod,
                    'seq': seq,
                    'rep': run,
                    'prop_dt': prop_dt,
                    'prop_base': prop_base,
                    'n_files': _clean(row.get('n_files')),
                    'acq_time': _clean(row.get('acq_time')),
                    'visual_only': self._is_visual_only_sequence(seq, mod),
                })
                self._apply_visual_only_rules(r)
            self.log_text.append("Loaded TSV into mapping table.")

            # Apply always-exclude patterns before building lookup tables
            self.applyExcludePatterns()

            # Build modality/sequence lookup for tree interactions
            self._rebuild_lookup_maps()
            self._maybe_notify_repeats()

            self.populateModalitiesTree()
            self.populateSpecificTree()
            if getattr(self, 'last_rep_box', None) is not None and self.last_rep_box.isChecked():
                self._onLastRepToggled(True)

            if self._apply_sequence_on_load:
                # Auto-apply the suffix dictionary after a fresh scan so the
                # latest custom patterns are reflected in the newly loaded
                # mapping table without requiring another manual save.
                self.applySequenceDictionary()
                self._apply_sequence_on_load = False

            # Populate naming table
            self.naming_table.blockSignals(True)
            self.naming_table.setRowCount(0)
            name_df = df[["StudyDescription", "GivenName", "BIDS_name"]].copy()
            name_df = name_df.drop_duplicates(subset=["StudyDescription", "BIDS_name"])
            for _, row in name_df.iterrows():
                nr = self.naming_table.rowCount()
                self.naming_table.insertRow(nr)
                sitem = QTableWidgetItem(_clean(row["StudyDescription"]))
                sitem.setFlags(sitem.flags() & ~Qt.ItemIsEditable)
                self.naming_table.setItem(nr, 0, sitem)
                gitem = QTableWidgetItem(_clean(row["GivenName"]))
                gitem.setFlags(gitem.flags() & ~Qt.ItemIsEditable)
                self.naming_table.setItem(nr, 1, gitem)
                bitem = QTableWidgetItem(_clean(row["BIDS_name"]))
                bitem.setFlags(bitem.flags() | Qt.ItemIsEditable)
                self.naming_table.setItem(nr, 2, bitem)
            self.naming_table.blockSignals(False)
            self._updateMappingControlsEnabled()
            QTimer.singleShot(0, self._auto_apply_existing_study_mappings)
        finally:
            self._loading_mapping_table = False

    def _build_series_list_from_df(self, df):
        rows = []

        # ``heudiconv`` initially names outputs using a simplified stem derived
        # from the DICOM SeriesDescription.  To later locate those files for
        # renaming we reconstruct that stem here.  We mirror the logic used by
        # :mod:`build_heuristic_from_tsv` which appends ``rep-<N>`` when a
        # sequence appears multiple times for a given subject/session.
        rep_counts = (
            df.groupby(["BIDS_name", "session", "sequence"], dropna=False)["sequence"].transform("count")
        )
        rep_index = (
            df.groupby(["BIDS_name", "session", "sequence"], dropna=False).cumcount() + 1
        )

        for idx, row in df.iterrows():
            subject = _extract_subject(row)
            session = row.get("session") or row.get("ses") or None
            modality = str(row.get("modality") or row.get("fine_modality") or row.get("BIDS_modality") or "")
            sequence = str(row.get("sequence") or row.get("SeriesDescription") or "")
            # ``rep`` encodes repeat acquisitions detected earlier.  Leave it as
            # ``None`` for non-repeated series and cast to ``int`` when present.
            rep_val = row.get("rep") or row.get("repeat")
            rep = int(rep_val) if rep_val else None

            extra: dict[str, str] = {}
            for key in ("task", "task_hits", "acq", "run", "dir", "echo"):
                if row.get(key):
                    extra[key] = str(row.get(key))

            # Reconstruct the basename produced by the converter so
            # :func:`apply_post_conversion_rename` can locate existing files even
            # when their names no longer contain the raw sequence.  This uses the
            # subject ID, optional session, a "safe" version of the sequence and
            # ``rep-<N>`` when duplicates exist.
            if row.get("BIDS_name") and sequence:
                base_parts = [str(row["BIDS_name"])]
                if session:
                    base_parts.append(session)
                base_parts.append(_safe_stem(sequence))
                if rep_counts.iloc[idx] > 1:
                    base_parts.append(f"rep-{rep_index.iloc[idx]}")
                current_base = _dedup_parts(*base_parts)
                extra["current_bids"] = current_base

            rows.append(SeriesInfo(subject, session, modality, sequence, rep, extra))

        return rows

    def _post_conversion_schema_rename(self, bids_root: str, df):
        if not (ENABLE_SCHEMA_RENAMER and self._schema):
            return {}
        series_list = self._build_series_list_from_df(df)
        proposals = build_preview_names(series_list, self._schema)
        rename_map = apply_post_conversion_rename(
            bids_root=bids_root,
            proposals=proposals,
            also_normalize_fieldmaps=ENABLE_FIELDMap_NORMALIZATION,
            handle_dwi_derivatives=ENABLE_DWI_DERIVATIVES_MOVE,
            derivatives_pipeline_name=DERIVATIVES_PIPELINE_NAME,
        )
        return rename_map

    def populateModalitiesTree(self):
        """Build modalities tree with checkboxes synced to the table."""
        self.full_tree.blockSignals(True)
        self.full_tree.clear()
        # build nested mapping: BIDS modality → non‑BIDS modality → seq → info
        modb_map = {}
        for info in self.row_info:
            modb_map.setdefault(info['modb'], {})\
                    .setdefault(info['mod'], {})[(info['seq'], info['rep'])] = info

        for modb, mod_map in sorted(modb_map.items()):
            modb_item = QTreeWidgetItem([modb])
            modb_item.setFlags(modb_item.flags() | Qt.ItemIsUserCheckable)
            rows = self.modb_rows.get(modb, [])
            states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
            if states and all(states):
                modb_item.setCheckState(0, Qt.Checked)
            elif states and any(states):
                modb_item.setCheckState(0, Qt.PartiallyChecked)
            else:
                modb_item.setCheckState(0, Qt.Unchecked)
            modb_item.setData(0, Qt.UserRole, ('modb', modb))

            for mod, seqs in sorted(mod_map.items()):
                mod_item = QTreeWidgetItem([mod])
                mod_item.setFlags(mod_item.flags() | Qt.ItemIsUserCheckable)
                rows = self.mod_rows.get((modb, mod), [])
                states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
                if states and all(states):
                    mod_item.setCheckState(0, Qt.Checked)
                elif states and any(states):
                    mod_item.setCheckState(0, Qt.PartiallyChecked)
                else:
                    mod_item.setCheckState(0, Qt.Unchecked)
                mod_item.setData(0, Qt.UserRole, ('mod', modb, mod))
                for (seq, rep), info in sorted(seqs.items()):
                    visual_only = bool(info.get("visual_only"))
                    label = seq
                    if rep:
                        label = f"{seq} (rep {rep})"
                    seq_item = QTreeWidgetItem([label])
                    if visual_only:
                        seq_item.setFlags(
                            (seq_item.flags() & ~Qt.ItemIsUserCheckable) & ~Qt.ItemIsEnabled
                        )
                    else:
                        seq_item.setFlags(seq_item.flags() | Qt.ItemIsUserCheckable)
                    rows = self.seq_rows.get((modb, mod, seq, rep), [])
                    states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
                    if states and all(states):
                        seq_item.setCheckState(0, Qt.Checked)
                    elif states and any(states):
                        seq_item.setCheckState(0, Qt.PartiallyChecked)
                    else:
                        seq_item.setCheckState(0, Qt.Unchecked)
                    seq_item.setData(0, Qt.UserRole, ('seq', modb, mod, seq, rep))
                    mod_item.addChild(seq_item)
                modb_item.addChild(mod_item)
            self.full_tree.addTopLevelItem(modb_item)

        self.full_tree.expandAll()
        self.full_tree.blockSignals(False)
        try:
            self.full_tree.itemChanged.disconnect(self.onModalityItemChanged)
        except TypeError:
            pass
        self.full_tree.itemChanged.connect(self.onModalityItemChanged)
        for i in range(self.full_tree.columnCount()):
            self.full_tree.resizeColumnToContents(i)

    def onSpecificItemChanged(self, item, column):
        role = item.data(0, Qt.UserRole)
        if not role:
            return
        state = item.checkState(0)
        tp = role[0]
        if tp == 'study':
            rows = self.study_rows.get(role[1], [])
        elif tp == 'subject':
            if self.use_bids_names:
                rows = self.subject_rows.get((role[1], role[2]), [])
            else:
                rows = self.subject_rows_given.get((role[1], role[2]), [])
        elif tp == 'session':
            if self.use_bids_names:
                rows = self.session_rows.get((role[1], role[2], role[3]), [])
            else:
                rows = self.session_rows_given.get((role[1], role[2], role[3]), [])
        elif tp == 'modb':
            if self.use_bids_names:
                rows = self.spec_modb_rows.get((role[1], role[2], role[3], role[4]), [])
            else:
                rows = self.spec_modb_rows_given.get((role[1], role[2], role[3], role[4]), [])
        elif tp == 'mod':
            if self.use_bids_names:
                rows = self.spec_mod_rows.get((role[1], role[2], role[3], role[4], role[5]), [])
            else:
                rows = self.spec_mod_rows_given.get((role[1], role[2], role[3], role[4], role[5]), [])
        elif tp == 'seq':
            if self.use_bids_names:
                rows = self.spec_seq_rows.get((role[1], role[2], role[3], role[4], role[5], role[6], role[7]), [])
            else:
                rows = self.spec_seq_rows_given.get((role[1], role[2], role[3], role[4], role[5], role[6], role[7]), [])
        else:
            rows = []
        for r in rows:
            if self.row_info[r].get("visual_only"):
                continue
            self.mapping_table.item(r, 0).setCheckState(state)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)

    def _onNamingEdited(self, item):
        if item.column() != 2:
            return
        study = self.naming_table.item(item.row(), 0).text()
        given = self.naming_table.item(item.row(), 1).text()
        # Remember the existing BIDS name so we can restore it if validation fails
        old_bids = None
        for info in self.row_info:
            if info['study'] == study and info['given'] == given:
                old_bids = info['bids']
                break
        new_bids = item.text()
        # Ensure the prefix is kept and that names remain unique
        if not new_bids.startswith('sub-'):
            QMessageBox.warning(self, "Invalid name", "BIDS names must start with 'sub-'.")
            if old_bids is not None:
                item.setText(old_bids)
            return
        other_names = [
            self.naming_table.item(r, 2).text()
            for r in range(self.naming_table.rowCount())
            if r != item.row() and self.naming_table.item(r, 0).text() == study
        ]
        # Also consider names already present in the converted dataset
        used_names = set(other_names)
        used_names.update(self.existing_used.get(study, set()))
        if new_bids in used_names and self.existing_maps.get(study, {}).get(given) != new_bids:
            QMessageBox.warning(
                self,
                "Duplicate name",
                "This name is already assigned in this study.",
            )
            if old_bids is not None:
                item.setText(old_bids)
            return
        for idx, info in enumerate(self.row_info):
            if info['study'] == study and info['given'] == given:
                info['bids'] = new_bids
                self.mapping_table.item(idx, 5).setText(new_bids)
        # Keep internal mapping updated
        self.existing_maps.setdefault(study, {})[given] = new_bids
        self.existing_used.setdefault(study, set()).add(new_bids)
        self._rebuild_lookup_maps()
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.generatePreview)

    def _onNameChoiceChanged(self, _index=None):
        self.use_bids_names = self.name_choice.currentIndex() == 0
        QTimer.singleShot(0, self.generatePreview)
        QTimer.singleShot(0, self.populateSpecificTree)

    def _onLastRepToggled(self, checked=False):
        groups = defaultdict(list)
        for idx, info in enumerate(self.row_info):
            key = (
                info['study'], info['bids'], info['ses'],
                info['modb'], info['mod'], info['seq']
            )
            rep_num = int(info['rep']) if str(info['rep']).isdigit() else 1
            groups.setdefault(key, []).append((rep_num, idx))
        for items in groups.values():
            filtered = [entry for entry in items if not self.row_info[entry[1]].get("visual_only")]
            if len(filtered) < 2:
                continue
            if checked:
                max_idx = max(filtered, key=lambda x: x[0])[1]
                for _, i in filtered:
                    st = Qt.Checked if i == max_idx else Qt.Unchecked
                    self.mapping_table.item(i, 0).setCheckState(st)
            else:
                for _, i in filtered:
                    self.mapping_table.item(i, 0).setCheckState(Qt.Checked)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.populateModalitiesTree)

    def _exclude_add(self) -> None:
        pattern = self.exclude_edit.text().strip()
        if not pattern:
            return
        r = self.exclude_table.rowCount()
        self.exclude_table.insertRow(r)
        chk = QTableWidgetItem()
        chk.setFlags(chk.flags() | Qt.ItemIsUserCheckable)
        chk.setCheckState(Qt.Checked)
        self.exclude_table.setItem(r, 0, chk)
        self.exclude_table.setItem(r, 1, QTableWidgetItem(pattern))
        self.exclude_edit.clear()

    def loadExcludePatterns(self) -> None:
        if not hasattr(self, "exclude_table"):
            return
        self.exclude_table.setRowCount(0)
        patterns = []
        if self.exclude_patterns_file.exists():
            try:
                df = pd.read_csv(self.exclude_patterns_file, sep="\t", keep_default_na=False)
                for _, row in df.iterrows():
                    pat = str(row.get("pattern", ""))
                    active = bool(int(row.get("active", 1)))
                    patterns.append((pat, active))
            except Exception:
                pass
        if not patterns:
            patterns = [
                ("localizer", True),
                ("scout", True),
                ("phoenixzipreport", True),
                ("phoenix document", True),
                (".pdf", True),
                ("report", True),
                ("physlog", True),
            ]
        for pat, active in patterns:
            r = self.exclude_table.rowCount()
            self.exclude_table.insertRow(r)
            chk = QTableWidgetItem()
            chk.setFlags(chk.flags() | Qt.ItemIsUserCheckable)
            chk.setCheckState(Qt.Checked if active else Qt.Unchecked)
            self.exclude_table.setItem(r, 0, chk)
            self.exclude_table.setItem(r, 1, QTableWidgetItem(pat))
        self.applyExcludePatterns()

    def saveExcludePatterns(self) -> None:
        self.exclude_patterns_file.parent.mkdir(exist_ok=True, parents=True)
        rows = []
        for r in range(self.exclude_table.rowCount()):
            pat = self.exclude_table.item(r, 1).text().strip()
            if not pat:
                continue
            active = self.exclude_table.item(r, 0).checkState() == Qt.Checked
            rows.append({"active": int(active), "pattern": pat})
        pd.DataFrame(rows).to_csv(self.exclude_patterns_file, sep="\t", index=False)
        QMessageBox.information(self, "Saved", f"Updated {self.exclude_patterns_file}")
        self.applyExcludePatterns()

    def applyExcludePatterns(self) -> None:
        if not hasattr(self, "exclude_table"):
            return
        patterns = [
            self.exclude_table.item(r, 1).text().strip().lower()
            for r in range(self.exclude_table.rowCount())
            if self.exclude_table.item(r, 0).checkState() == Qt.Checked
        ]
        for r in range(self.mapping_table.rowCount()):
            seq = self.mapping_table.item(r, 9).text().lower()
            if any(p in seq for p in patterns):
                self.mapping_table.item(r, 0).setCheckState(Qt.Unchecked)

    def _custom_add(self, suffix: str) -> None:
        table = self.custom_tables.get(suffix)
        edit = self.custom_inputs.get(suffix)
        if table is None or edit is None:
            return
        pat = edit.text().strip()
        if not pat:
            return
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(pat))
        edit.clear()

    def _custom_remove(self, suffix: str) -> None:
        table = self.custom_tables.get(suffix)
        if table is None:
            return
        rows = sorted({item.row() for item in table.selectedItems()}, reverse=True)
        for r in rows:
            table.removeRow(r)

    def _collect_custom_patterns(self) -> Dict[str, Tuple[str, ...]]:
        patterns: Dict[str, Tuple[str, ...]] = {}
        for suffix, table in self.custom_tables.items():
            entries: list[str] = []
            for r in range(table.rowCount()):
                item = table.item(r, 0)
                if item is None:
                    continue
                text = item.text().strip()
                if text:
                    entries.append(text)
            if entries:
                patterns[suffix] = tuple(entries)
        return patterns

    def _set_custom_pattern_enabled(self, enabled: bool) -> None:
        for table in self.custom_tables.values():
            table.setEnabled(enabled)
        for edit in self.custom_inputs.values():
            edit.setEnabled(enabled)
        for button in self.custom_add_buttons.values():
            button.setEnabled(enabled)
        for button in self.custom_remove_buttons.values():
            button.setEnabled(enabled)
        if hasattr(self, "seq_save_button"):
            self.seq_save_button.setEnabled(enabled)

    def _refresh_suffix_dictionary(self) -> None:
        active = dicom_inventory.get_sequence_hint_patterns()
        custom = dicom_inventory.get_custom_sequence_dictionary()
        use_custom = dicom_inventory.is_sequence_dictionary_enabled()
        for suffix, label in self.active_labels.items():
            active_list = active.get(suffix, ())
            source = "Custom" if use_custom and custom.get(suffix) else "Default"
            label.setText(f"Active source: {source} ({len(active_list)} patterns)")

    def _on_custom_toggle(self, enabled: bool) -> None:
        dicom_inventory.set_sequence_dictionary_enabled(enabled)
        self._set_custom_pattern_enabled(enabled)
        self._refresh_suffix_dictionary()
        self.applySequenceDictionary()

    def loadSequenceDictionary(self) -> None:
        if not hasattr(self, "seq_tabs_widget"):
            return

        default_patterns = dicom_inventory.get_sequence_hint_patterns(source="default")
        custom_patterns = dicom_inventory.get_custom_sequence_dictionary()
        suffixes = sorted(set(default_patterns) | set(custom_patterns))
        use_custom = dicom_inventory.is_sequence_dictionary_enabled()

        self.seq_tabs_widget.clear()
        self.default_pattern_lists = {}
        self.custom_tables = {}
        self.custom_inputs = {}
        self.custom_add_buttons = {}
        self.custom_remove_buttons = {}
        self.active_labels = {}

        for suffix in suffixes:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            active_label = QLabel()
            self.active_labels[suffix] = active_label
            layout.addWidget(active_label)

            default_group = QGroupBox("Default patterns")
            default_layout = QVBoxLayout(default_group)
            default_list = QListWidget()
            default_list.addItems(default_patterns.get(suffix, ()))
            default_list.setEnabled(False)
            self.default_pattern_lists[suffix] = default_list
            default_layout.addWidget(default_list)
            layout.addWidget(default_group)

            custom_group = QGroupBox("Custom patterns")
            custom_layout = QVBoxLayout(custom_group)
            table = QTableWidget()
            table.setColumnCount(1)
            table.setHorizontalHeaderLabels(["Pattern"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            for pat in custom_patterns.get(suffix, ()):  # populate stored entries
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(pat))
            self.custom_tables[suffix] = table
            custom_layout.addWidget(table)

            controls = QHBoxLayout()
            edit = QLineEdit()
            self.custom_inputs[suffix] = edit
            controls.addWidget(edit)
            add_btn = QPushButton("Add")
            add_btn.clicked.connect(lambda _=False, s=suffix: self._custom_add(s))
            self.custom_add_buttons[suffix] = add_btn
            controls.addWidget(add_btn)
            rm_btn = QPushButton("Remove")
            rm_btn.clicked.connect(lambda _=False, s=suffix: self._custom_remove(s))
            self.custom_remove_buttons[suffix] = rm_btn
            controls.addWidget(rm_btn)
            custom_layout.addLayout(controls)
            layout.addWidget(custom_group)

            self.seq_tabs_widget.addTab(tab, suffix)

        self.use_custom_patterns_box.blockSignals(True)
        self.use_custom_patterns_box.setChecked(use_custom)
        self.use_custom_patterns_box.blockSignals(False)
        self._set_custom_pattern_enabled(use_custom)
        self._refresh_suffix_dictionary()
        self.applySequenceDictionary()

    def saveSequenceDictionary(self) -> None:
        patterns = self._collect_custom_patterns()
        data = [
            {"modality": suffix, "pattern": pat}
            for suffix, pats in patterns.items()
            for pat in pats
        ]
        if data:
            self.seq_dict_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(data).to_csv(self.seq_dict_file, sep="\t", index=False)
        else:
            try:
                self.seq_dict_file.unlink()
            except Exception:
                pass
        dicom_inventory.update_sequence_dictionary(patterns)
        dicom_inventory.set_sequence_dictionary_enabled(self.use_custom_patterns_box.isChecked())
        QMessageBox.information(
            self,
            "Saved",
            f"Updated suffix patterns in {self.seq_dict_file}",
        )
        self._refresh_suffix_dictionary()
        # Applying the sequence dictionary immediately updates the scanned data
        # table so users do not need to trigger "Save changes" separately.
        self.applySequenceDictionary()
        self.applyMappingChanges()

    def restoreSequenceDefaults(self) -> None:
        dicom_inventory.restore_sequence_dictionary()
        self.loadSequenceDictionary()
        QMessageBox.information(
            self,
            "Restored",
            "Default suffix dictionary restored",
        )

    def applySequenceDictionary(self) -> None:
        if not hasattr(self, "custom_tables"):
            return

        patterns = self._collect_custom_patterns()
        dicom_inventory.update_sequence_dictionary(patterns)
        dicom_inventory.set_sequence_dictionary_enabled(self.use_custom_patterns_box.isChecked())
        self._refresh_suffix_dictionary()

        if self.mapping_table.rowCount() > 0:
            for i in range(self.mapping_table.rowCount()):
                seq = self.mapping_table.item(i, 9).text()
                mod = dicom_inventory.guess_modality(seq)
                modb = dicom_inventory.modality_to_container(mod)
                self.mapping_table.item(i, 14).setText(mod)
                self.mapping_table.item(i, 15).setText(modb)
                if i < len(self.row_info):
                    self.row_info[i]['mod'] = mod
                    self.row_info[i]['modb'] = modb
            self._rebuild_lookup_maps()
            QTimer.singleShot(0, self.populateModalitiesTree)
            QTimer.singleShot(0, self.populateSpecificTree)

    def _rebuild_lookup_maps(self):
        """Recompute internal lookup tables for tree interactions."""
        # Maps from tree nodes to row indices in ``self.mapping_table``. These
        # allow checkbox changes in the tree to update table rows and vice versa.
        self.modb_rows.clear()
        self.mod_rows.clear()
        self.seq_rows.clear()
        self.study_rows.clear()
        self.subject_rows.clear()
        self.session_rows.clear()
        self.spec_modb_rows.clear()
        self.spec_mod_rows.clear()
        self.spec_seq_rows.clear()

        # Same lookups but using the "given" subject names when that mode is
        # selected instead of the BIDS names
        self.subject_rows_given = {}
        self.session_rows_given = {}
        self.spec_modb_rows_given = {}
        self.spec_mod_rows_given = {}
        self.spec_seq_rows_given = {}
        for idx, info in enumerate(self.row_info):
            if info.get("visual_only"):
                continue
            # Populate lookup tables using BIDS subject names
            self.modb_rows.setdefault(info['modb'], []).append(idx)
            self.mod_rows.setdefault((info['modb'], info['mod']), []).append(idx)
            self.seq_rows.setdefault((info['modb'], info['mod'], info['seq'], info['rep']), []).append(idx)
            self.study_rows.setdefault(info['study'], []).append(idx)
            self.subject_rows.setdefault((info['study'], info['bids']), []).append(idx)
            self.session_rows.setdefault((info['study'], info['bids'], info['ses']), []).append(idx)
            self.spec_modb_rows.setdefault((info['study'], info['bids'], info['ses'], info['modb']), []).append(idx)
            self.spec_mod_rows.setdefault((info['study'], info['bids'], info['ses'], info['modb'], info['mod']), []).append(idx)
            self.spec_seq_rows.setdefault((info['study'], info['bids'], info['ses'], info['modb'], info['mod'], info['seq'], info['rep']), []).append(idx)
            gsub = f"sub-{info['given']}"
            # Equivalent lookups built from the given (non-BIDS) subject names
            self.subject_rows_given.setdefault((info['study'], gsub), []).append(idx)
            self.session_rows_given.setdefault((info['study'], gsub, info['ses']), []).append(idx)
            self.spec_modb_rows_given.setdefault((info['study'], gsub, info['ses'], info['modb']), []).append(idx)
            self.spec_mod_rows_given.setdefault((info['study'], gsub, info['ses'], info['modb'], info['mod']), []).append(idx)
            self.spec_seq_rows_given.setdefault((info['study'], gsub, info['ses'], info['modb'], info['mod'], info['seq'], info['rep']), []).append(idx)

    def _save_tree_expansion(self, tree):
        states = {}

        def recurse(item):
            path = []
            it = item
            while it is not None:
                path.insert(0, it.text(0))
                it = it.parent()
            states[tuple(path)] = item.isExpanded()
            for i in range(item.childCount()):
                recurse(item.child(i))

        for i in range(tree.topLevelItemCount()):
            recurse(tree.topLevelItem(i))
        return states

    def _restore_tree_expansion(self, tree, states):
        def recurse(item):
            path = []
            it = item
            while it is not None:
                path.insert(0, it.text(0))
                it = it.parent()
            if states.get(tuple(path)):
                item.setExpanded(True)
            for i in range(item.childCount()):
                recurse(item.child(i))

        for i in range(tree.topLevelItemCount()):
            recurse(tree.topLevelItem(i))

    def populateSpecificTree(self):
        """Build detailed tree (study→subject→session→modality)."""
        expanded = self._save_tree_expansion(self.specific_tree)
        self.specific_tree.blockSignals(True)
        self.specific_tree.clear()

        tree_map = {}
        for info in self.row_info:
            if self.use_bids_names:
                subj_key = info['bids']
            else:
                subj_key = f"sub-{info['given']}"
            tree_map.setdefault(info['study'], {})\
                    .setdefault(subj_key, {})\
                    .setdefault(info['ses'], {})\
                    .setdefault(info['modb'], {})\
                    .setdefault(info['mod'], {})[(info['seq'], info['rep'])] = info

        def _state(rows):
            states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
            if states and all(states):
                return Qt.Checked
            if states and any(states):
                return Qt.PartiallyChecked
            return Qt.Unchecked

        multi_study = len(tree_map) > 1

        for study, sub_map in sorted(tree_map.items()):
            st_item = QTreeWidgetItem([study])
            if multi_study:
                # Only expose the study-level checkbox when the dataset includes
                # multiple studies so that users do not see a redundant single
                # checkbox for the only available study.
                st_item.setFlags(st_item.flags() | Qt.ItemIsUserCheckable)
                st_item.setCheckState(0, _state(self.study_rows.get(study, [])))
                st_item.setData(0, Qt.UserRole, ('study', study))
            else:
                # Remove the user-checkable flag to hide the checkbox while
                # keeping the study label visible for context.
                st_item.setFlags(st_item.flags() & ~Qt.ItemIsUserCheckable)
            for subj, ses_map in sorted(sub_map.items()):
                su_item = QTreeWidgetItem([subj])
                su_item.setFlags(su_item.flags() | Qt.ItemIsUserCheckable)
                if self.use_bids_names:
                    rows = self.subject_rows.get((study, subj), [])
                else:
                    rows = self.subject_rows_given.get((study, subj), [])
                su_item.setCheckState(0, _state(rows))
                su_item.setData(0, Qt.UserRole, ('subject', study, subj))
                for ses, modb_map in sorted(ses_map.items()):
                    se_item = QTreeWidgetItem([ses])
                    se_item.setFlags(se_item.flags() | Qt.ItemIsUserCheckable)
                    if self.use_bids_names:
                        rows = self.session_rows.get((study, subj, ses), [])
                    else:
                        rows = self.session_rows_given.get((study, subj, ses), [])
                    se_item.setCheckState(0, _state(rows))
                    se_item.setData(0, Qt.UserRole, ('session', study, subj, ses))
                    for modb, mod_map in sorted(modb_map.items()):
                        mb_item = QTreeWidgetItem([modb, "", ""])
                        mb_item.setFlags(mb_item.flags() | Qt.ItemIsUserCheckable)
                        if self.use_bids_names:
                            rows = self.spec_modb_rows.get((study, subj, ses, modb), [])
                        else:
                            rows = self.spec_modb_rows_given.get((study, subj, ses, modb), [])
                        mb_item.setCheckState(0, _state(rows))
                        mb_item.setData(0, Qt.UserRole, ('modb', study, subj, ses, modb))
                        for mod, seqs in sorted(mod_map.items()):
                            mo_item = QTreeWidgetItem([mod, "", ""])
                            mo_item.setFlags(mo_item.flags() | Qt.ItemIsUserCheckable)
                            if self.use_bids_names:
                                rows = self.spec_mod_rows.get((study, subj, ses, modb, mod), [])
                            else:
                                rows = self.spec_mod_rows_given.get((study, subj, ses, modb, mod), [])
                            mo_item.setCheckState(0, _state(rows))
                            mo_item.setData(0, Qt.UserRole, ('mod', study, subj, ses, modb, mod))
                            for (seq, rep), info in sorted(seqs.items()):
                                label = seq
                                if rep:
                                    label = f"{seq} (rep {rep})"
                                files = str(info['n_files'])
                                time = info['acq_time']
                                sq_item = QTreeWidgetItem([label, files, time])
                                if info.get("visual_only"):
                                    sq_item.setFlags(
                                        (sq_item.flags() & ~Qt.ItemIsUserCheckable) & ~Qt.ItemIsEnabled
                                    )
                                else:
                                    sq_item.setFlags(sq_item.flags() | Qt.ItemIsUserCheckable)
                                if self.use_bids_names:
                                    rows = self.spec_seq_rows.get((study, subj, ses, modb, mod, seq, rep), [])
                                else:
                                    rows = self.spec_seq_rows_given.get((study, subj, ses, modb, mod, seq, rep), [])
                                sq_item.setCheckState(0, _state(rows))
                                sq_item.setData(0, Qt.UserRole, ('seq', study, subj, ses, modb, mod, seq, rep))
                                mo_item.addChild(sq_item)
                            mb_item.addChild(mo_item)
                        se_item.addChild(mb_item)
                    su_item.addChild(se_item)
                st_item.addChild(su_item)
            self.specific_tree.addTopLevelItem(st_item)

        self._restore_tree_expansion(self.specific_tree, expanded)
        if not expanded:
            self.specific_tree.expandAll()
        self.specific_tree.blockSignals(False)
        try:
            self.specific_tree.itemChanged.disconnect(self.onSpecificItemChanged)
        except TypeError:
            pass
        self.specific_tree.itemChanged.connect(self.onSpecificItemChanged)
        for i in range(self.specific_tree.columnCount()):
            self.specific_tree.resizeColumnToContents(i)

    def onModalityItemChanged(self, item, column):
        role = item.data(0, Qt.UserRole)
        if not role:
            return
        state = item.checkState(0)
        if role[0] == 'modb':
            modb = role[1]
            for r in self.modb_rows.get(modb, []):
                if self.row_info[r].get("visual_only"):
                    continue
                self.mapping_table.item(r, 0).setCheckState(state)
        elif role[0] == 'mod':
            modb, mod = role[1], role[2]
            for r in self.mod_rows.get((modb, mod), []):
                if self.row_info[r].get("visual_only"):
                    continue
                self.mapping_table.item(r, 0).setCheckState(state)
        elif role[0] == 'seq':
            modb, mod, seq, rep = role[1], role[2], role[3], role[4]
            for r in self.seq_rows.get((modb, mod, seq, rep), []):
                if self.row_info[r].get("visual_only"):
                    continue
                self.mapping_table.item(r, 0).setCheckState(state)
        QTimer.singleShot(0, self.populateModalitiesTree)

    def runFullConversion(self):
        logging.info("runFullConversion → Starting full pipeline …")
        if self.conv_process and self.conv_process.state() != QProcess.NotRunning:
            return
        if not self.tsv_path or not os.path.isfile(self.tsv_path):
            QMessageBox.warning(self, "No TSV", "Please generate the TSV first.")
            return
        if not self.bids_out_dir:
            QMessageBox.warning(self, "No BIDS Output", "Please select a BIDS output directory.")
            return

        # 1) Save updated TSV from table
        try:
            df_orig = pd.read_csv(self.tsv_path, sep="\t", keep_default_na=False)
            df_conv = df_orig.copy()
            for i in range(self.mapping_table.rowCount()):
                include = 1 if self.mapping_table.item(i, 0).checkState() == Qt.Checked else 0
                info = self.row_info[i]
                seq = self.mapping_table.item(i, 9).text()
                modb = self.mapping_table.item(i, 15).text()

                # Update df_orig with canonical BIDS name
                df_orig.at[i, 'BIDS_name'] = info['bids']
                df_orig.at[i, 'include'] = include
                df_orig.at[i, 'sequence'] = seq
                df_orig.at[i, 'modality_bids'] = modb

                # For conversion we may use given names
                conv_name = info['bids'] if self.use_bids_names else f"sub-{info['given']}"
                df_conv.at[i, 'BIDS_name'] = conv_name

            df_orig.to_csv(self.tsv_path, sep="\t", index=False)
            self.log_text.append("Saved updated TSV.")

            # Write temporary TSV for heuristic generation if using given names
            if self.use_bids_names:
                self.tsv_for_conv = self.tsv_path
            else:
                tmp_tsv = os.path.join(self.bids_out_dir, "tmp_subjects.tsv")
                df_conv.to_csv(tmp_tsv, sep="\t", index=False)
                self.tsv_for_conv = tmp_tsv
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save TSV: {e}")
            return

        # Paths for scripts
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.build_script = os.path.join(script_dir, "build_heuristic_from_tsv.py")
        self.run_script = os.path.join(script_dir, "run_heudiconv_from_heuristic.py")
        self.rename_script = os.path.join(script_dir, "post_conv_renamer.py")

        self.heuristic_dir = os.path.join(self.bids_out_dir, "heuristics")
        self.heurs_to_rename = []
        self.conv_stage = 0

        self.log_text.append("Building heuristics…")
        self._start_spinner("Converting")
        self.run_button.setEnabled(False)
        self.run_stop_button.setEnabled(True)
        self.conv_process = QProcess(self)
        if self.terminal_cb.isChecked():
            # Forward output so the user can monitor conversion progress
            self.conv_process.setProcessChannelMode(QProcess.ForwardedChannels)
        else:
            # Prevent blocked pipes on Windows when the terminal isn't shown
            self.conv_process.setStandardOutputFile(QProcess.nullDevice())
            self.conv_process.setStandardErrorFile(QProcess.nullDevice())
        self.conv_process.finished.connect(self._convStepFinished)
        args = [self.build_script, self.tsv_for_conv, self.heuristic_dir]
        self.conv_process.start(sys.executable, args)

    def _convStepFinished(self, exitCode, _status):
        if self.conv_stage == 0:
            if exitCode != 0:
                QMessageBox.critical(self, "Error", "build_heuristic failed")
                self.stopConversion()
                return
            self.log_text.append(f"Heuristics written to {self.heuristic_dir}")
            self.conv_stage = 1
            self.log_text.append("Running HeuDiConv…")
            args = [self.run_script, self.dicom_dir, self.heuristic_dir, self.bids_out_dir, '--subject-tsv', self.tsv_path]
            self.conv_process.start(sys.executable, args)
        elif self.conv_stage == 1:
            if exitCode != 0:
                QMessageBox.critical(self, "Error", "run_heudiconv failed")
                self.stopConversion()
                return
            self.log_text.append("HeuDiConv conversion complete.")
            self.conv_stage = 2
            self.heurs_to_rename = list(Path(self.heuristic_dir).glob("heuristic_*.py"))
            self._runNextRename()
        elif self.conv_stage == 2:
            if exitCode != 0:
                QMessageBox.critical(self, "Error", "post_conv_renamer failed")
                self.stopConversion()
                return
            if self.heurs_to_rename:
                self._runNextRename()
            else:
                self.log_text.append("Conversion pipeline finished successfully.")
                self._store_heuristics()
                if self.inventory_df is not None:
                    rename_map = self._post_conversion_schema_rename(self.bids_out_dir, self.inventory_df)
                    self.log_text.append(f"Schema renamer moved/renamed {len(rename_map)} files.")
                if getattr(self, 'tsv_for_conv', self.tsv_path) != self.tsv_path:
                    try:
                        os.remove(self.tsv_for_conv)
                    except Exception:
                        pass
                self.stopConversion(success=True)

    def _runNextRename(self):
        if not self.heurs_to_rename:
            self._convStepFinished(0, 0)
            return
        heur = self.heurs_to_rename.pop(0)
        dataset = heur.stem.replace("heuristic_", "")
        bids_path = os.path.join(self.bids_out_dir, dataset)
        self.log_text.append(f"Renaming fieldmaps for {dataset}…")
        args = [self.rename_script, bids_path]
        self.conv_process.start(sys.executable, args)

    def _store_heuristics(self):
        """Move heuristics into each dataset's .bids_manager folder."""
        try:
            hdir = Path(self.heuristic_dir)
            for heur in hdir.glob("heuristic_*.py"):
                dataset = heur.stem.replace("heuristic_", "")
                dst = Path(self.bids_out_dir) / dataset / ".bids_manager"
                dst.mkdir(exist_ok=True)
                shutil.move(str(heur), dst / heur.name)
            shutil.rmtree(hdir, ignore_errors=True)
        except Exception as exc:
            logging.warning(f"Failed to move heuristics: {exc}")

    def stopConversion(self, success: bool = False):
        if self.conv_process and self.conv_process.state() != QProcess.NotRunning:
            pid = int(self.conv_process.processId())
            _terminate_process_tree(pid)
        self.conv_process = None
        self._stop_spinner()
        self.run_button.setEnabled(True)
        self.run_stop_button.setEnabled(False)
        if not success:
            self.log_text.append("Conversion cancelled.")
