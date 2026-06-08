"""TSV / TSV.GZ viewer + editor (Editor center pane, table kind).

Sister widget to :class:`SidecarFormPane`. When the user clicks a
``.tsv`` (or ``.tsv.gz``) file in the BIDS tree, :class:`EditorPanel`
swaps its center pane to this viewer.

* Parsed with pandas' C engine on a :class:`bidsmgr.workers.TsvLoaderWorker`
  thread (releases the GIL, so the GUI stays responsive) and bounded to a
  preview cap so even a multi-million-row table reads quickly.
* Rendered through a **lazy** :class:`_TsvTableModel`
  (:class:`~PyQt6.QtCore.QAbstractTableModel`) backed by plain Python
  lists. Crucially this creates **no per-cell objects** and only the
  visible cells are queried, so binding the data is O(1) - no main-thread
  freeze regardless of how many rows / columns the file has.
* Cells are inline-editable (double-click / F2 / select-then-click).
* Toolbar: ``+ Add row`` / ``+ Add column`` / ``− Delete row`` /
  ``− Delete column`` plus ``Revert`` / ``Save`` (manual-save model, same
  as the JSON sidecar pane). ``Save`` flushes to disk; ``Revert`` reloads.
* Switching files silently discards unsaved edits.

Theme handling: every palette colour comes from the global QSS.
:meth:`repaint_for_palette` runs the same unpolish/polish dance the
sidecar pane uses.
"""

from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QStackedLayout,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .primitives import PaneHeader

log = logging.getLogger(__name__)


# Don't slurp arbitrarily-large TSVs into memory. BIDS TSVs (events /
# channels / participants / scans) are small in practice; if anyone passes
# a multi-million-row table we cap the preview here (the read is bounded to
# this, and the footer warns about truncation). Saving a truncated TSV
# would be destructive, so Save stays disabled when truncated on load.
_MAX_PREVIEW_ROWS = 5000


def _open_tsv_write(path: Path):
    """Open a ``.tsv`` or ``.tsv.gz`` for text writing (UTF-8)."""
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def _count_data_rows(path: Path) -> int:
    """Count data rows (newlines minus the header) at the C level.

    Used only when the preview was truncated, to report an exact total in
    the footer. Reads in binary chunks so the I/O (and zlib for ``.gz``)
    releases the GIL - it never freezes the worker.
    """
    is_gz = str(path).lower().endswith(".gz")
    opener = gzip.open if is_gz else open
    newlines = 0
    last = b""
    try:
        with opener(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                newlines += chunk.count(b"\n")
                last = chunk[-1:] or last
    except OSError:
        return 0
    if last and last != b"\n":
        newlines += 1  # final line had no trailing newline
    return max(0, newlines - 1)  # minus the header line


def _read_tsv(
    path: Path,
    *,
    max_rows: int = _MAX_PREVIEW_ROWS,
) -> tuple[list[str], list[list[str]], int]:
    """Read a TSV file. Returns ``(header, rows, total_rows)``.

    ``total_rows`` is the count of data rows on disk (so the footer can
    report truncation). Empty / malformed / missing files return
    ``([], [], 0)``.

    Parsed with pandas' **C engine** (releases the GIL during tokenising;
    Python's ``csv`` module is pure-Python and would hold the GIL, starving
    the worker so the loading spinner freezes). The read is **bounded** to
    ``max_rows + 1`` rows so a huge / wide file never materialises in full
    on the worker; the exact total is then counted cheaply at the C level.
    """
    import pandas as pd

    try:
        compression = "gzip" if str(path).lower().endswith(".gz") else "infer"
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            header=0,
            keep_default_na=False,
            na_filter=False,
            compression=compression,
            engine="c",
            on_bad_lines="skip",
            nrows=max_rows + 1,
        )
    except Exception as exc:  # noqa: BLE001 - empty / missing / malformed
        log.debug("could not read TSV %s: %s", path, exc)
        return [], [], 0

    header = [str(c) for c in df.columns]
    n_read = int(len(df))
    if not header and n_read == 0:
        return [], [], 0
    truncated = n_read > max_rows
    # Ragged rows: pandas pads short rows with NaN; coerce to "" so the
    # table mirrors the on-disk blanks.
    preview = df.head(max_rows).fillna("")
    rows = preview.astype(str).values.tolist()
    total = _count_data_rows(path) if truncated else n_read
    return header, rows, total


# ===========================================================================
# Lazy table model (no per-cell objects -> O(1) bind, no freeze)
# ===========================================================================
class _TsvTableModel(QAbstractTableModel):
    """A ``QAbstractTableModel`` over ``header: list[str]`` + ``rows``.

    Stores the parsed table as plain lists and serves cells on demand, so
    loading is O(1) (no ``QStandardItem`` per cell) and only visible cells
    are rendered. Supports inline cell editing + row/column add/remove for
    the toolbar.
    """

    # Emitted on any user cell edit (the pane uses it for dirty tracking).
    cellEdited = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._header: list[str] = []
        self._rows: list[list[str]] = []

    # --- data ----------------------------------------------------------
    def set_table(self, header, rows) -> None:
        """Replace the whole table (load / restore). Rectangular-padded."""
        self.beginResetModel()
        self._header = [str(h) for h in header]
        w = len(self._header)
        self._rows = []
        for r in rows:
            cells = [str(x) for x in list(r)[:w]]
            if len(cells) < w:
                cells += [""] * (w - len(cells))
            self._rows.append(cells)
        self.endResetModel()

    def header(self) -> list[str]:
        return list(self._header)

    def rows(self) -> list[list[str]]:
        return [list(r) for r in self._rows]

    # --- QAbstractTableModel API --------------------------------------
    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._header)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            r, c = index.row(), index.column()
            if 0 <= r < len(self._rows) and 0 <= c < len(self._rows[r]):
                return self._rows[r][c]
            return ""
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._header[section] if 0 <= section < len(self._header) else ""
        return section + 1

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        r, c = index.row(), index.column()
        if not (0 <= r < len(self._rows)):
            return False
        while len(self._rows[r]) <= c:
            self._rows[r].append("")
        self._rows[r][c] = str(value)
        self.dataChanged.emit(
            index, index,
            [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole],
        )
        self.cellEdited.emit()
        return True

    # --- structural edits (toolbar) -----------------------------------
    def insert_row(self) -> None:
        n = len(self._rows)
        self.beginInsertRows(QModelIndex(), n, n)
        self._rows.append([""] * max(1, len(self._header)))
        self.endInsertRows()

    def remove_row(self, r: int) -> None:
        if 0 <= r < len(self._rows):
            self.beginRemoveRows(QModelIndex(), r, r)
            del self._rows[r]
            self.endRemoveRows()

    def add_column(self, name: str) -> None:
        c = len(self._header)
        self.beginInsertColumns(QModelIndex(), c, c)
        self._header.append(str(name))
        for row in self._rows:
            row.append("")
        self.endInsertColumns()

    def remove_column(self, c: int) -> None:
        if 0 <= c < len(self._header):
            self.beginRemoveColumns(QModelIndex(), c, c)
            del self._header[c]
            for row in self._rows:
                if c < len(row):
                    del row[c]
            self.endRemoveColumns()


class TsvViewerPane(QWidget):
    """Editable table view for BIDS ``.tsv`` files."""

    # Emitted after a successful save. Per-file revalidation hooks here.
    file_saved = pyqtSignal(Path)
    # Emitted when the disk write fails. Args: (file_path, error_msg).
    save_failed = pyqtSignal(Path, str)
    # Emitted whenever the dirty state flips. ``True`` means unsaved edits.
    dirty_changed = pyqtSignal(bool)
    # Emitted whenever undo/redo availability changes (Editor toolbar sync).
    history_changed = pyqtSignal()
    # Emitted after a background load completes (success or empty). Tests
    # wait on this; the Editor mirrors it to its busy spinner.
    loaded = pyqtSignal(Path)
    # Emitted while a background load is in flight / finishes. Args:
    # ``(busy, message)``. The Editor mirrors it to its toolbar spinner.
    loading_changed = pyqtSignal(bool, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")

        self._current_file: Optional[Path] = None
        self._current_root: Optional[Path] = None
        # Background TSV parse worker. Kept on self so it isn't GC'd.
        self._loader = None
        # Disk snapshot — used to compute dirty + restore on revert.
        self._original_header: list[str] = []
        self._original_rows: list[list[str]] = []
        # True when the on-disk file is larger than the preview cap (Save
        # disabled so we never drop the tail rows).
        self._truncated_on_load: bool = False
        self._dirty = False
        # Undo/redo of in-memory table edits (snapshot = (header, rows)).
        from .edit_history import SnapshotHistory
        self._history = SnapshotHistory()
        self._pre_edit: Optional[tuple] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Table"))

        # --- Edit toolbar ----------------------------------------------
        self._edit_toolbar = QFrame()
        self._edit_toolbar.setObjectName("sidecar-toolbar")
        et = QHBoxLayout(self._edit_toolbar)
        et.setContentsMargins(14, 6, 14, 6)
        et.setSpacing(8)

        self._add_row_btn = QPushButton("+ Add row")
        self._add_row_btn.setObjectName("tb-btn")
        self._add_row_btn.clicked.connect(self._on_add_row)
        et.addWidget(self._add_row_btn)

        self._del_row_btn = QPushButton("− Delete row")
        self._del_row_btn.setObjectName("tb-btn")
        self._del_row_btn.setEnabled(False)
        self._del_row_btn.clicked.connect(self._on_delete_row)
        et.addWidget(self._del_row_btn)

        self._add_col_btn = QPushButton("+ Add column")
        self._add_col_btn.setObjectName("tb-btn")
        self._add_col_btn.clicked.connect(self._on_add_column)
        et.addWidget(self._add_col_btn)

        self._del_col_btn = QPushButton("− Delete column")
        self._del_col_btn.setObjectName("tb-btn")
        self._del_col_btn.setEnabled(False)
        self._del_col_btn.clicked.connect(self._on_delete_column)
        et.addWidget(self._del_col_btn)

        self._dirty_chip = QLabel("")
        self._dirty_chip.setObjectName("sidecar-dirty-chip")
        self._dirty_chip.setVisible(False)
        et.addWidget(self._dirty_chip)
        et.addStretch(1)

        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setObjectName("tb-btn")
        self._revert_btn.setEnabled(False)
        self._revert_btn.clicked.connect(self.revert)
        et.addWidget(self._revert_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("tb-btn")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save)
        et.addWidget(self._save_btn)

        self._edit_toolbar.setVisible(False)
        v.addWidget(self._edit_toolbar)

        # --- Stacked content -------------------------------------------
        self._stack = QStackedLayout()
        self._stack.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self._stack, 1)

        self._table = QTableView()
        self._table.setObjectName("tsv-view")
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self._table.horizontalHeader().setStretchLastSection(False)
        # Resize-to-contents must only SAMPLE a few rows, or it measures
        # every cell and freezes on big tables. 50 rows is plenty for a
        # sensible default width.
        self._table.horizontalHeader().setResizeContentsPrecision(50)
        self._table.verticalHeader().setVisible(True)
        self._model = _TsvTableModel(self)
        self._model.cellEdited.connect(self._on_cell_edited)
        self._table.setModel(self._model)
        sel_model = self._table.selectionModel()
        if sel_model is not None:
            sel_model.currentChanged.connect(self._sync_delete_button_state)

        self._empty_hint = QLabel(
            "Select a TSV file in the BIDS tree to view it."
        )
        self._empty_hint.setObjectName("pane-hint")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setWordWrap(True)

        # Loading page (index 2): an animated spinner, mirroring the NIfTI
        # viewer.
        from .spinner import BusySpinner

        self._loading_page = QWidget()
        self._loading_page.setObjectName("pane-dark")
        lp = QVBoxLayout(self._loading_page)
        lp.addStretch(1)
        srow = QHBoxLayout()
        srow.addStretch(1)
        self._loading_spinner = BusySpinner()
        srow.addWidget(self._loading_spinner)
        srow.addStretch(1)
        lp.addLayout(srow)
        self._loading_label = QLabel("")
        self._loading_label.setObjectName("pane-hint")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lp.addWidget(self._loading_label)
        lp.addStretch(1)

        # Index 0 = empty hint, 1 = table, 2 = loading.
        self._stack.addWidget(self._empty_hint)
        self._stack.addWidget(self._table)
        self._stack.addWidget(self._loading_page)
        self._stack.setCurrentIndex(0)

        # Footer (path + summary), QSS-driven so theme follows.
        self._footer = QFrame()
        self._footer.setObjectName("sidecar-footer")
        fl = QHBoxLayout(self._footer)
        fl.setContentsMargins(14, 6, 14, 6)
        fl.setSpacing(10)
        self._footer_path = QLabel("")
        self._footer_path.setObjectName("sidecar-footer-path")
        self._footer_summary = QLabel("")
        self._footer_summary.setObjectName("sidecar-footer-summary")
        fl.addWidget(self._footer_path, 1)
        fl.addWidget(self._footer_summary)
        v.addWidget(self._footer)

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def current_file(self) -> Optional[Path]:
        return self._current_file

    def is_dirty(self) -> bool:
        return self._dirty

    def set_file(
        self,
        path: Optional[Path],
        root: Optional[Path],
    ) -> None:
        """Bind the pane to a TSV (or ``None`` to clear).

        The parse runs on a :class:`bidsmgr.workers.TsvLoaderWorker`; the
        spinner page shows while it reads and the table (lazy model) is
        bound instantly when it lands.
        """
        if self._loader is not None:
            self._loader.cancel()
            self._loader = None

        self._current_file = path
        self._current_root = root
        self._history.clear()
        if path is None:
            self._reset_model()
            self._original_header = []
            self._original_rows = []
            self._truncated_on_load = False
            self._dirty = False
            self._stack.setCurrentIndex(0)
            self._empty_hint.setText(
                "Select a TSV file in the BIDS tree to view it."
            )
            self._footer_path.setText("")
            self._footer_summary.setText("")
            self._edit_toolbar.setVisible(False)
            self._refresh_dirty_ui()
            self._pre_edit = self._snapshot()
            self.history_changed.emit()
            self.loading_changed.emit(False, "")
            return

        self._reset_model()
        self._loading_label.setText(f"Loading {path.name}…")
        self._loading_spinner.set_busy(True, message="")
        self._stack.setCurrentIndex(2)
        self._edit_toolbar.setVisible(False)
        self._footer_path.setText("")
        self._footer_summary.setText("")
        self.loading_changed.emit(True, f"Loading {path.name}…")

        from ...workers import TsvLoaderWorker

        worker = TsvLoaderWorker(path, _MAX_PREVIEW_ROWS, parent=self)
        worker.finished_with_data.connect(self._on_loaded)
        worker.failed.connect(self._on_load_failed)
        worker.finished.connect(worker.deleteLater)
        self._loader = worker
        worker.start()

    def _on_loaded(
        self,
        header: list,
        rows: list,
        total: int,
        path: Path,
    ) -> None:
        """Background parse finished — bind the table (guarded for staleness)."""
        if path != self._current_file:
            return
        self._loader = None
        self._loading_spinner.set_busy(False)
        self._original_header = list(header)
        self._original_rows = [list(r) for r in rows]
        self._truncated_on_load = total > len(rows)
        self._dirty = False

        if not header and not rows:
            self._reset_model()
            self._stack.setCurrentIndex(0)
            self._empty_hint.setText(
                "This TSV is empty or could not be parsed."
            )
        else:
            self._populate_model(header, rows)
            self._stack.setCurrentIndex(1)
        self._update_footer(path, self._current_root, len(rows), len(header), total)
        self._edit_toolbar.setVisible(True)
        self._refresh_dirty_ui()
        self._pre_edit = self._snapshot()
        self.history_changed.emit()
        self.loading_changed.emit(False, "")
        self.loaded.emit(path)

    def _on_load_failed(self, path: Path, error: str) -> None:
        if path != self._current_file:
            return
        self._loader = None
        self._loading_spinner.set_busy(False)
        self._reset_model()
        self._stack.setCurrentIndex(0)
        self._empty_hint.setText(f"Could not load {path.name}:\n{error}")
        self._edit_toolbar.setVisible(False)
        self._dirty = False
        self._refresh_dirty_ui()
        self._pre_edit = self._snapshot()
        self.history_changed.emit()
        self.loading_changed.emit(False, "")
        self.loaded.emit(path)

    def save(self) -> bool:
        """Flush the model to disk. ``True`` on success / no-op."""
        if self._current_file is None:
            return True
        if not self._dirty:
            return True
        if self._truncated_on_load:
            msg = (
                f"File was truncated on load ({_MAX_PREVIEW_ROWS} rows "
                "shown); refusing to overwrite — saving would discard "
                "the tail rows."
            )
            log.warning("save refused for %s: %s", self._current_file, msg)
            self.save_failed.emit(self._current_file, msg)
            return False
        try:
            self._write_model_to_disk(self._current_file)
        except OSError as exc:
            log.warning("save failed for %s: %s", self._current_file, exc)
            self.save_failed.emit(self._current_file, str(exc))
            return False
        self._snapshot_current_model()
        self._dirty = False
        self._refresh_dirty_ui()
        self.file_saved.emit(self._current_file)
        return True

    def revert(self) -> None:
        """Reload the bound file from disk, dropping unsaved edits."""
        if self._current_file is None:
            return
        path, root = self._current_file, self._current_root
        self.set_file(path, root)

    def repaint_for_palette(self, pal: dict) -> None:
        """Same QSS-only refresh pattern as :class:`SidecarFormPane`."""
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()

    # ----------------------------------------------------------------------
    # Toolbar handlers
    # ----------------------------------------------------------------------

    def _on_add_row(self) -> None:
        if self._current_file is None:
            return
        if self._model.columnCount() == 0:
            self._model.add_column("col1")
        self._model.insert_row()
        self._mark_dirty()
        self._stack.setCurrentIndex(1)

    def _on_delete_row(self) -> None:
        idx = self._table.currentIndex()
        if not idx.isValid():
            return
        self._model.remove_row(idx.row())
        self._mark_dirty()

    def _on_add_column(self) -> None:
        if self._current_file is None:
            return
        name, ok = QInputDialog.getText(
            self, "Add column", "Column name:", text="newCol",
        )
        if not ok or not name.strip():
            return
        self._model.add_column(name.strip())
        self._mark_dirty()
        self._stack.setCurrentIndex(1)

    def _on_delete_column(self) -> None:
        idx = self._table.currentIndex()
        if not idx.isValid():
            return
        self._model.remove_column(idx.column())
        self._mark_dirty()

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _reset_model(self) -> None:
        self._model.set_table([], [])

    def _populate_model(
        self,
        header: list[str],
        rows: list[list[str]],
    ) -> None:
        # O(1) bind: the lazy model just stores the lists. Resize only
        # samples a few rows (precision set in __init__), so this is fast
        # no matter how big / wide the table is.
        self._model.set_table(header, rows)
        self._table.resizeColumnsToContents()

    def _snapshot_current_model(self) -> None:
        self._original_header = self._model.header()
        self._original_rows = self._model.rows()

    def _current_header(self) -> list[str]:
        return self._model.header()

    def _current_rows(self) -> list[list[str]]:
        return self._model.rows()

    def _write_model_to_disk(self, path: Path) -> None:
        header = self._model.header()
        rows = self._model.rows()
        with _open_tsv_write(path) as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)

    def _on_cell_edited(self) -> None:
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        if self._pre_edit is not None:
            self._history.record(self._pre_edit)
        self._pre_edit = self._snapshot()
        if not self._dirty:
            self._dirty = True
        self._refresh_dirty_ui()
        self.history_changed.emit()

    # ----------------------------------------------------------------------
    # Undo / redo (snapshot-based, in-memory; disk write still via Save)
    # ----------------------------------------------------------------------

    def _snapshot(self) -> tuple[list[str], list[list[str]]]:
        return (self._model.header(), self._model.rows())

    def _restore(self, snap: tuple[list[str], list[list[str]]]) -> None:
        header, rows = snap
        self._populate_model(list(header), [list(r) for r in rows])
        self._stack.setCurrentIndex(1 if (header or rows) else 0)
        self._dirty = (
            list(header) != list(self._original_header)
            or [list(r) for r in rows] != [list(r) for r in self._original_rows]
        )
        self._pre_edit = self._snapshot()
        self._refresh_dirty_ui()
        self.history_changed.emit()

    def can_undo(self) -> bool:
        return self._current_file is not None and self._history.can_undo

    def can_redo(self) -> bool:
        return self._current_file is not None and self._history.can_redo

    def undo(self) -> None:
        if self._current_file is None:
            return
        snap = self._history.undo(self._snapshot())
        if snap is not None:
            self._restore(snap)

    def redo(self) -> None:
        if self._current_file is None:
            return
        snap = self._history.redo(self._snapshot())
        if snap is not None:
            self._restore(snap)

    def _refresh_dirty_ui(self) -> None:
        has_file = self._current_file is not None
        editable = has_file and not self._truncated_on_load
        if self._dirty:
            self._dirty_chip.setText("unsaved changes")
            self._dirty_chip.setVisible(True)
        else:
            self._dirty_chip.setVisible(False)
        self._save_btn.setEnabled(editable and self._dirty)
        self._revert_btn.setEnabled(has_file and self._dirty)
        self._add_row_btn.setEnabled(editable)
        self._add_col_btn.setEnabled(editable)
        self._sync_delete_button_state()
        self.dirty_changed.emit(self._dirty)

    def _sync_delete_button_state(self, *args) -> None:
        del args
        editable = self._current_file is not None and not self._truncated_on_load
        idx = self._table.currentIndex()
        has_sel = idx.isValid()
        self._del_row_btn.setEnabled(
            editable and has_sel and self._model.rowCount() > 0
        )
        self._del_col_btn.setEnabled(
            editable and has_sel and self._model.columnCount() > 0
        )

    def _update_footer(
        self,
        path: Path,
        root: Optional[Path],
        shown_rows: int,
        cols: int,
        total_rows: int,
    ) -> None:
        if root is not None:
            try:
                rel = path.resolve().relative_to(root.resolve())
                self._footer_path.setText(str(rel))
            except ValueError:
                self._footer_path.setText(str(path))
        else:
            self._footer_path.setText(str(path))
        if shown_rows < total_rows:
            self._footer_summary.setText(
                f"{shown_rows} of {total_rows} rows shown · {cols} columns "
                f"· read-only (truncated)"
            )
        else:
            self._footer_summary.setText(
                f"{total_rows} rows · {cols} columns"
            )


__all__ = ["TsvViewerPane"]
