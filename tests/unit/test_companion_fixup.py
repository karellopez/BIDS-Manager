"""Tests for the companion-file copy fixup (task logs / events / beh / stim)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from bidsmgr.fixups.companion import attach_companion_files


def _stage(tmp_path: Path, basename="sub-001_task-rest_eeg", datatype="eeg") -> Path:
    d = tmp_path / ".tmp_bidsmgr" / "sub-001" / datatype
    d.mkdir(parents=True)
    (d / f"{basename}.json").write_text("{}", encoding="utf-8")
    return tmp_path / ".tmp_bidsmgr" / "sub-001"


def _task(basename="sub-001_task-rest_eeg", datatype="eeg", companions=()):
    return SimpleNamespace(
        basename=basename, datatype=datatype, companion_files=tuple(companions),
    )


def test_attach_copies_events_with_bids_name(tmp_path):
    staging = _stage(tmp_path)
    src = tmp_path / "my_events.tsv"
    src.write_text("onset\tduration\ttrial_type\n0\t1\tgo\n", encoding="utf-8")
    n = attach_companion_files(staging, [_task(companions=[("events", str(src))])])
    assert n == 1
    out = staging / "eeg" / "sub-001_task-rest_events.tsv"
    assert out.exists() and "go" in out.read_text()


def test_attach_skips_unsupported_suffix(tmp_path):
    staging = _stage(tmp_path)
    src = tmp_path / "x.tsv"
    src.write_text("a", encoding="utf-8")
    assert attach_companion_files(staging, [_task(companions=[("bogus", str(src))])]) == 0


def test_attach_skips_missing_source(tmp_path):
    staging = _stage(tmp_path)
    assert attach_companion_files(
        staging, [_task(companions=[("events", "/no/such/file.tsv")])]
    ) == 0


def test_attach_preserves_tsv_gz_extension(tmp_path):
    staging = _stage(tmp_path)
    src = tmp_path / "phys.tsv.gz"
    src.write_bytes(b"\x1f\x8b")
    attach_companion_files(staging, [_task(companions=[("physio", str(src))])])
    assert (staging / "eeg" / "sub-001_task-rest_physio.tsv.gz").exists()


def test_attach_noop_without_companions(tmp_path):
    staging = _stage(tmp_path)
    assert attach_companion_files(staging, [_task()]) == 0
