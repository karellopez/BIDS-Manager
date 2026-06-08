"""Background loader for TSV / TSV.GZ tables (Editor viewer).

BIDS sidecar TSVs (events / channels / participants / scans) are usually
small, but a phenotype table or a derivatives TSV can run to many
thousands of rows, and decompressing a ``.tsv.gz`` blocks while the file
is read. Doing that on the GUI thread freezes the window when the user
clicks such a file in the BIDS tree.

This worker pushes the parse onto a ``QThread`` and emits the parsed
header / rows back to :class:`bidsmgr.gui.widgets.TsvViewerPane`,
mirroring :class:`bidsmgr.workers.nifti_loader.NiftiLoaderWorker`.

Cancellation
------------
The pane often issues a new load before the previous one finishes (the
user clicks a different file). The worker can't interrupt the parse, so
it flips a flag and suppresses the result; the pane also guards on the
emitted ``path`` so a stale table never reaches the UI.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)


class TsvLoaderWorker(QThread):
    """Parse a ``.tsv`` / ``.tsv.gz`` on a background thread.

    Signals
    -------
    finished_with_data(list, list, int, Path)
        Emitted on success. Args: ``(header, rows, total_rows, path)``.
        ``total_rows`` is the on-disk data-row count (so the footer can
        report truncation). Always check ``path`` against the bound file.
    failed(Path, str)
        Emitted on read failure. Args: ``(path, error_message)``.
    """

    finished_with_data = pyqtSignal(list, list, int, Path)
    failed = pyqtSignal(Path, str)

    def __init__(self, path: Path, max_rows: int, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._max_rows = max_rows
        self._cancelled = False

    def path(self) -> Path:
        return self._path

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        from bidsmgr.gui.widgets.tsv_viewer_pane import _read_tsv

        try:
            header, rows, total = _read_tsv(self._path, max_rows=self._max_rows)
        except Exception as exc:  # noqa: BLE001 - surfaced to UI
            log.warning("TsvLoaderWorker failed on %s: %s", self._path, exc)
            if not self._cancelled:
                self.failed.emit(self._path, str(exc))
            return
        if self._cancelled:
            return
        self.finished_with_data.emit(header, rows, total, self._path)


__all__ = ["TsvLoaderWorker"]
