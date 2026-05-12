"""Tests for bids_manager.build_dcm2bids_config.

These exercise the TSV-to-config translation. We don't actually run dcm2bids
here — the contract under test is structural: every included TSV row
produces exactly one description matching that row's SeriesInstanceUID,
and the names match what schema_renamer.build_preview_names emits.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from bids_manager.build_dcm2bids_config import (
    _split_base,
    generate,
)


def _write_tsv(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _base_row(**overrides):
    row = {
        "subject": "OL_001",
        "BIDS_name": "001",
        "session": "",
        "source_folder": "OL_001/scan",
        "include": 1,
        "sequence": "mprage",
        "series_uid": "1.2.3",
        "rep": 1,
        "acq_time": "104500",
        "modality": "T1w",
        "modality_bids": "anat",
        "n_files": 192,
        "StudyDescription": "PILOT",
        "GivenName": "Alice",
        "FamilyName": "Doe",
        "patientID": "P1",
        "PatientAge": "030Y",
        "PatientSex": "F",
    }
    row.update(overrides)
    return row


def test_one_description_per_included_row(tmp_path):
    tsv = tmp_path / "inv.tsv"
    _write_tsv(tsv, [
        _base_row(series_uid="UID-001", sequence="mprage", modality="T1w", modality_bids="anat"),
        _base_row(series_uid="UID-002", sequence="rest",   modality="bold", modality_bids="func"),
    ])
    out = tmp_path / "cfg"
    paths = generate(tsv, out)
    assert len(paths) == 1
    cfg = json.loads(paths[0].read_text())
    assert len(cfg["descriptions"]) == 2
    uids = {d["criteria"]["SeriesInstanceUID"] for d in cfg["descriptions"]}
    assert uids == {"UID-001", "UID-002"}


def test_excluded_rows_omitted(tmp_path):
    tsv = tmp_path / "inv.tsv"
    _write_tsv(tsv, [
        _base_row(series_uid="UID-included", include=1),
        _base_row(series_uid="UID-excluded", include=0,
                  sequence="localizer", modality="scout", modality_bids="misc"),
    ])
    out = tmp_path / "cfg"
    paths = generate(tsv, out)
    cfg = json.loads(paths[0].read_text())
    uids = {d["criteria"]["SeriesInstanceUID"] for d in cfg["descriptions"]}
    assert uids == {"UID-included"}


def test_one_config_per_study(tmp_path):
    tsv = tmp_path / "inv.tsv"
    _write_tsv(tsv, [
        _base_row(StudyDescription="STUDY_A", series_uid="A1"),
        _base_row(StudyDescription="STUDY_B", series_uid="B1"),
    ])
    out = tmp_path / "cfg"
    paths = generate(tsv, out)
    assert len(paths) == 2
    names = {p.name for p in paths}
    assert names == {"dcm2bids_config_STUDY_A.json", "dcm2bids_config_STUDY_B.json"}


def test_skip_modalities_auto_excluded(tmp_path):
    """SKIP_MODALITIES (scout/report) must drop out automatically."""
    tsv = tmp_path / "inv.tsv"
    _write_tsv(tsv, [
        _base_row(series_uid="UID-keep"),
        _base_row(series_uid="UID-scout", modality="scout"),
        _base_row(series_uid="UID-report", modality="report"),
    ])
    out = tmp_path / "cfg"
    paths = generate(tsv, out)
    cfg = json.loads(paths[0].read_text())
    uids = {d["criteria"]["SeriesInstanceUID"] for d in cfg["descriptions"]}
    assert uids == {"UID-keep"}


def test_split_base_strips_sub_ses_and_returns_suffix():
    custom, suffix = _split_base("sub-001_ses-pre_task-rest_run-2_bold", "001", "pre")
    assert custom == "task-rest_run-2"
    assert suffix == "bold"


def test_split_base_no_session():
    custom, suffix = _split_base("sub-001_T1w", "001", "")
    assert custom == ""
    assert suffix == "T1w"


def test_dwi_derivative_routed_to_dwi_datatype(tmp_path):
    """DWI derivative maps (FA/ColFA/ADC/...) must go to dwi/ for dcm2bids;
    relocation under derivatives/ happens later via schema_renamer."""
    tsv = tmp_path / "inv.tsv"
    _write_tsv(tsv, [
        _base_row(series_uid="UID-fa", sequence="dti_FA", modality="dwi", modality_bids="dwi"),
    ])
    out = tmp_path / "cfg"
    paths = generate(tsv, out)
    cfg = json.loads(paths[0].read_text())
    assert cfg["descriptions"][0]["datatype"] == "dwi"


def test_uid_pipe_field_takes_first(tmp_path):
    """Multi-UID rows (separated by '|') match the first UID — same as the
    heudiconv heuristic builder."""
    tsv = tmp_path / "inv.tsv"
    _write_tsv(tsv, [
        _base_row(series_uid="UID-primary|UID-secondary"),
    ])
    out = tmp_path / "cfg"
    paths = generate(tsv, out)
    cfg = json.loads(paths[0].read_text())
    assert cfg["descriptions"][0]["criteria"]["SeriesInstanceUID"] == "UID-primary"
