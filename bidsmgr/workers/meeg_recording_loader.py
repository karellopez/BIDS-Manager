"""Background loaders for EEG / MEG / iEEG recordings (Editor viewer).

The Editor's :class:`bidsmgr.gui.widgets.RecordingViewerPane` reads
recordings with MNE-Python. Even a metadata read can touch a multi-GB
file, and a full signal load (``preload=True``) can take several seconds
and allocate gigabytes. Doing either on the GUI thread freezes the
window, including the busy spinner that is supposed to say "loading".

These workers push the work onto ``QThread``s, mirroring
:class:`bidsmgr.workers.nifti_loader.NiftiLoaderWorker`.

Cancellation
------------
The pane often issues a new load before the previous one has finished
(the user clicks a different recording in the BIDS tree). The worker
cannot interrupt MNE mid-read, so it flips a flag and suppresses the
result emission; the pane additionally guards on the emitted ``path`` so
a stale result never reaches the UI.

CTF ``.ds`` + macOS note
------------------------
MNE's CTF reader uses lazy memory-mapping; the MEEGqc viewer reported
SIGBUS crashes when reading ``.ds`` inside a ``QThread`` on macOS. We
load with ``preload=True`` (the whole file is materialised into RAM
during the call, leaving no lingering memory-map to fault later), and our
datasets never use that path, so we thread uniformly. If a real
CTF-on-macOS SIGBUS is ever observed, the fix is a deferred main-thread
read guarded by ``sys.platform == "darwin" and ext == ".ds"`` in the
pane's :meth:`set_file` / load handler.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)


class RecordingMetaWorker(QThread):
    """Read a recording's metadata (``preload=False``) on a background thread.

    Signals
    -------
    finished_with_meta(dict, Path)
        Emitted on success. ``dict`` is the plain summary produced by
        :func:`bidsmgr.gui.widgets.recording_viewer_pane._summarize_raw`.
        Always check ``path`` against the currently-bound file.
    failed(Path, str)
        Emitted on read failure. Args: ``(path, error_message)``.
    """

    finished_with_meta = pyqtSignal(dict, Path)
    failed = pyqtSignal(Path, str)

    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._cancelled = False

    def path(self) -> Path:
        return self._path

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        from bidsmgr.gui.widgets.recording_viewer_pane import (
            _read_raw,
            _summarize_raw,
        )

        try:
            raw = _read_raw(self._path, preload=False)
            meta = _summarize_raw(raw, self._path)
        except Exception as exc:  # noqa: BLE001 - surfaced to UI
            log.warning("RecordingMetaWorker failed on %s: %s", self._path, exc)
            if not self._cancelled:
                self.failed.emit(self._path, str(exc))
            return
        if self._cancelled:
            return
        self.finished_with_meta.emit(meta, self._path)


class RecordingSignalWorker(QThread):
    """Read a recording's full signal (``preload=True``) on a background thread.

    Signals
    -------
    finished_with_raw(object, Path)
        Emitted on success with the preloaded ``mne.io.Raw``.
    failed(Path, str)
        Emitted on read failure.
    """

    finished_with_raw = pyqtSignal(object, Path)
    failed = pyqtSignal(Path, str)

    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._cancelled = False

    def path(self) -> Path:
        return self._path

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        from bidsmgr.gui.widgets.recording_viewer_pane import _read_raw

        try:
            raw = _read_raw(self._path, preload=True)
        except Exception as exc:  # noqa: BLE001 - surfaced to UI
            log.warning(
                "RecordingSignalWorker failed on %s: %s", self._path, exc
            )
            if not self._cancelled:
                self.failed.emit(self._path, str(exc))
            return
        if self._cancelled:
            return
        self.finished_with_raw.emit(raw, self._path)


class RecordingResampleWorker(QThread):
    """Resample an in-memory ``mne.io.Raw`` on a background thread.

    Signals
    -------
    finished_with_raw(object, float)
        Emitted on success. Args: ``(resampled_raw, new_sfreq)``.
    failed(str)
        Emitted on failure with the error message.
    """

    finished_with_raw = pyqtSignal(object, float)
    failed = pyqtSignal(str)

    def __init__(self, raw, sfreq: float, parent=None) -> None:
        super().__init__(parent)
        self._raw = raw
        self._sfreq = float(sfreq)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            resampled = self._raw.copy().resample(self._sfreq, verbose=False)
        except Exception as exc:  # noqa: BLE001 - surfaced to UI
            log.warning("RecordingResampleWorker failed: %s", exc)
            if not self._cancelled:
                self.failed.emit(str(exc))
            return
        if self._cancelled:
            return
        self.finished_with_raw.emit(resampled, self._sfreq)


class RecordingComputeWorker(QThread):
    """Run an arbitrary heavy callable on a background thread.

    Used for the PSD computation (an FFT over the whole recording can take
    a noticeable amount of time on long MEG files). The callable returns
    an opaque result object emitted back to the GUI thread.

    Signals
    -------
    finished_with_result(object)
        Emitted on success with whatever the callable returned.
    failed(str)
        Emitted on failure with the error message.
    """

    finished_with_result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, func: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self._func = func
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            result = self._func()
        except Exception as exc:  # noqa: BLE001 - surfaced to UI
            log.warning("RecordingComputeWorker failed: %s", exc)
            if not self._cancelled:
                self.failed.emit(str(exc))
            return
        if self._cancelled:
            return
        self.finished_with_result.emit(result)


__all__ = [
    "RecordingMetaWorker",
    "RecordingSignalWorker",
    "RecordingResampleWorker",
    "RecordingComputeWorker",
]
