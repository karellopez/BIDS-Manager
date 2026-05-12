"""Tests for the EEG/MEG → BIDS pipeline (eeg_meg_inventory + run_mne_bids).

Builds tiny EDF files via mne's RawArray + mne.export.export_raw, runs the
inventory and the conversion, and asserts the resulting files / report.
The metadata engine is exercised in test_bids_metadata_engine.py — we just
spot-check it here in the integration test to confirm engine + mne-bids
output coexist.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")
mne_bids = pytest.importorskip("mne_bids")
from mne.export import export_raw  # noqa: E402

from bids_manager.eeg_meg_inventory import scan_eeg_meg
from bids_manager.run_mne_bids import run as run_mne_bids


def _make_edf(path: Path, sfreq: int = 200, n_chan: int = 4, dur_sec: int = 2):
    info = mne.create_info(
        ch_names=[f"E{i+1}" for i in range(n_chan)], sfreq=sfreq, ch_types="eeg"
    )
    data = np.random.RandomState(0).randn(n_chan, sfreq * dur_sec) * 1e-5
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path.parent.mkdir(parents=True, exist_ok=True)
    export_raw(str(path), raw, fmt="edf", overwrite=True, verbose="ERROR")


# --------------------------------------------------------------- inventory
def test_inventory_finds_edf_and_extracts_metadata(tmp_path):
    _make_edf(tmp_path / "rawdata" / "sub-001" / "ses-01" / "sub-001_task-rest_eeg.edf")
    inv = tmp_path / "inv.tsv"
    df = scan_eeg_meg(tmp_path / "rawdata", inv)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["datatype"] == "eeg"
    assert row["format"] == "EDF"
    assert row["sfreq"] == 200.0
    assert row["n_channels"] == 4
    assert row["session"] == "01"
    assert row["task"] == "rest"
    assert row["BIDS_name"] == "001"
    # has_positions is 0 for synthetic data — confirms the warning path
    # downstream is reachable.
    assert int(row["has_positions"]) == 0


def test_inventory_skips_unsupported_files(tmp_path):
    rawdir = tmp_path / "rawdata"
    rawdir.mkdir()
    (rawdir / "notes.txt").write_text("hello")
    (rawdir / "screenshot.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    df = scan_eeg_meg(rawdir, tmp_path / "inv.tsv")
    assert df.empty


def test_inventory_groups_subjects_consistently(tmp_path):
    _make_edf(tmp_path / "rawdata" / "sub-A" / "rest_eeg.edf")
    _make_edf(tmp_path / "rawdata" / "sub-A" / "task_eeg.edf")
    _make_edf(tmp_path / "rawdata" / "sub-B" / "rest_eeg.edf")
    df = scan_eeg_meg(tmp_path / "rawdata", tmp_path / "inv.tsv")
    # Two recordings for sub-A → same BIDS_name; sub-B → distinct.
    a_ids = set(df[df["subject"] == "A"]["BIDS_name"])
    b_ids = set(df[df["subject"] == "B"]["BIDS_name"])
    assert len(a_ids) == 1
    assert len(b_ids) == 1
    assert a_ids != b_ids


# --------------------------------------------------------------- conversion
def test_run_mne_bids_writes_bids_layout(tmp_path):
    _make_edf(tmp_path / "raw" / "sub-001" / "ses-01" / "sub-001_task-rest_eeg.edf")
    inv = tmp_path / "inv.tsv"
    scan_eeg_meg(tmp_path / "raw", inv)
    bids_root = tmp_path / "BIDS"
    report = run_mne_bids(inv, tmp_path / "raw", bids_root, overwrite=True)
    assert len(report.written_paths) == 1
    assert len(report.errors) == 0

    # mne-bids writes the data file + sidecars + dataset-level files.
    assert (bids_root / "sub-001" / "ses-01" / "eeg" /
            "sub-001_ses-01_task-rest_eeg.edf").exists()
    assert (bids_root / "sub-001" / "ses-01" / "eeg" /
            "sub-001_ses-01_task-rest_channels.tsv").exists()
    assert (bids_root / "sub-001" / "ses-01" / "eeg" /
            "sub-001_ses-01_task-rest_eeg.json").exists()
    assert (bids_root / "dataset_description.json").exists()


def test_run_mne_bids_reports_missing_positions(tmp_path):
    """Synthetic EDFs have no electrode positions — every recording must end
    up in ``missing_positions`` so the GUI can warn the user."""
    _make_edf(tmp_path / "raw" / "sub-001" / "rest_eeg.edf")
    _make_edf(tmp_path / "raw" / "sub-002" / "rest_eeg.edf")
    inv = tmp_path / "inv.tsv"
    scan_eeg_meg(tmp_path / "raw", inv)
    report = run_mne_bids(inv, tmp_path / "raw", tmp_path / "BIDS", overwrite=True)
    assert len(report.missing_positions) == 2


def test_excluded_rows_not_converted(tmp_path):
    _make_edf(tmp_path / "raw" / "sub-001" / "rest_eeg.edf")
    _make_edf(tmp_path / "raw" / "sub-002" / "rest_eeg.edf")
    inv = tmp_path / "inv.tsv"
    scan_eeg_meg(tmp_path / "raw", inv)

    df = pd.read_csv(inv, sep="\t", dtype=str, keep_default_na=False)
    df.loc[df["subject"] == "002", "include"] = "0"
    df.to_csv(inv, sep="\t", index=False)

    report = run_mne_bids(inv, tmp_path / "raw", tmp_path / "BIDS", overwrite=True)
    written_subs = {p.name.split("_")[0] for p in report.written_paths}
    assert written_subs == {"sub-001"}


# -------------------------------------------- engine integration spot-check
def test_metadata_engine_runs_after_mne_bids(tmp_path):
    _make_edf(tmp_path / "raw" / "sub-001" / "rest_eeg.edf")
    inv = tmp_path / "inv.tsv"
    scan_eeg_meg(tmp_path / "raw", inv)
    bids_root = tmp_path / "BIDS"
    run_mne_bids(inv, tmp_path / "raw", bids_root, overwrite=True)

    from bids_manager.bids_metadata_engine import BIDSMetadataEngine, DatasetMetadata
    BIDSMetadataEngine(bids_root=bids_root,
                       dataset_meta=DatasetMetadata(name="EEG pilot")).run()

    # mne-bids' age/sex/hand columns must survive the engine's merge.
    df = pd.read_csv(bids_root / "participants.tsv", sep="\t", dtype=str,
                     keep_default_na=False)
    assert "age" in df.columns
    assert "sex" in df.columns

    dd = json.loads((bids_root / "dataset_description.json").read_text())
    assert dd["Name"] == "EEG pilot"
    assert any(g["Name"] == "BIDS-Manager" for g in dd["GeneratedBy"])
