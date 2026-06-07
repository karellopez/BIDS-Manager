"""Tests for the recording + TSV background loader workers."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from bidsmgr.workers import (  # noqa: E402
    RecordingComputeWorker,
    RecordingMetaWorker,
    RecordingResampleWorker,
    RecordingSignalWorker,
    TsvLoaderWorker,
)

pytestmark = pytest.mark.gui


@pytest.fixture
def fif_path(tmp_path: Path) -> Path:
    sfreq = 200.0
    n = int(sfreq * 4)
    data = np.random.default_rng(1).standard_normal((4, n)) * 1e-5
    info = mne.create_info(
        ["A", "B", "C", "STI"], sfreq, ["eeg", "eeg", "eeg", "stim"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.RawArray(data, info, verbose=False)
        path = tmp_path / "sub-01_task-x_eeg.fif"
        raw.save(path, overwrite=True, verbose=False)
    return path


def test_meta_worker_emits(qtbot, fif_path: Path) -> None:
    w = RecordingMetaWorker(fif_path)
    with qtbot.waitSignal(w.finished_with_meta, timeout=15000) as blocker:
        w.start()
    meta, path = blocker.args
    assert path == fif_path
    assert meta["n_channels"] == 4
    assert meta["sfreq"] == 200.0
    w.wait()


def test_meta_worker_failed_on_bad_path(qtbot, tmp_path: Path) -> None:
    w = RecordingMetaWorker(tmp_path / "missing_eeg.fif")
    with qtbot.waitSignal(w.failed, timeout=15000):
        w.start()
    w.wait()


def test_signal_worker_preloads(qtbot, fif_path: Path) -> None:
    w = RecordingSignalWorker(fif_path)
    with qtbot.waitSignal(w.finished_with_raw, timeout=15000) as blocker:
        w.start()
    raw, path = blocker.args
    assert path == fif_path
    assert raw.preload is True
    assert raw.n_times > 0
    w.wait()


def test_resample_worker(qtbot, fif_path: Path) -> None:
    raw = mne.io.read_raw(str(fif_path), preload=True, verbose=False)
    w = RecordingResampleWorker(raw, 100.0)
    with qtbot.waitSignal(w.finished_with_raw, timeout=15000) as blocker:
        w.start()
    out, freq = blocker.args
    assert freq == 100.0
    assert out.info["sfreq"] == 100.0
    w.wait()


def test_compute_worker_returns_result(qtbot) -> None:
    w = RecordingComputeWorker(lambda: 6 * 7)
    with qtbot.waitSignal(w.finished_with_result, timeout=5000) as blocker:
        w.start()
    assert blocker.args[0] == 42
    w.wait()


def test_compute_worker_failed(qtbot) -> None:
    def boom():
        raise ValueError("nope")

    w = RecordingComputeWorker(boom)
    with qtbot.waitSignal(w.failed, timeout=5000):
        w.start()
    w.wait()


def test_tsv_loader_worker(qtbot, tmp_path: Path) -> None:
    p = tmp_path / "big.tsv"
    p.write_text("a\tb\n1\t2\n3\t4\n")
    w = TsvLoaderWorker(p, 5000)
    with qtbot.waitSignal(w.finished_with_data, timeout=5000) as blocker:
        w.start()
    header, rows, total, path = blocker.args
    assert header == ["a", "b"]
    assert total == 2
    assert path == p
    w.wait()
