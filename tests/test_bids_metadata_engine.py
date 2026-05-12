"""Tests for bids_manager.bids_metadata_engine.

We assemble a minimal fake BIDS layout on disk plus an inventory TSV with
demographics, run the engine, then assert the right files were created with
the right contents.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from bids_manager.bids_metadata_engine import (
    BIDSMetadataEngine,
    DatasetMetadata,
)


def _make_fake_bids(root: Path) -> Path:
    for sub in ("sub-001", "sub-002"):
        anat = root / sub / "anat"
        anat.mkdir(parents=True)
        (anat / f"{sub}_T1w.nii.gz").write_bytes(b"\x00")
        (anat / f"{sub}_T1w.json").write_text(
            json.dumps({"AcquisitionDateTime": "2026-05-07T10:45:00",
                       "RepetitionTime": 2.3})
        )
    func = root / "sub-001" / "func"
    func.mkdir(parents=True)
    (func / "sub-001_task-rest_bold.nii.gz").write_bytes(b"\x00")
    (func / "sub-001_task-rest_bold.json").write_text(
        json.dumps({"AcquisitionDateTime": "2026-05-07T10:50:00",
                   "RepetitionTime": 2.0})
    )
    return root


def _make_inventory_tsv(path: Path):
    pd.DataFrame([
        {"BIDS_name": "001", "GivenName": "Alice", "FamilyName": "Doe",
         "patientID": "P1", "PatientAge": "030Y", "PatientSex": "F",
         "StudyDescription": "PILOT"},
        {"BIDS_name": "002", "GivenName": "Bob", "FamilyName": "Smith",
         "patientID": "P2", "PatientAge": "025Y", "PatientSex": "M",
         "StudyDescription": "PILOT"},
    ]).to_csv(path, sep="\t", index=False)


def test_writes_required_dataset_files(tmp_path):
    bids = _make_fake_bids(tmp_path / "PILOT")
    engine = BIDSMetadataEngine(
        bids_root=bids,
        dataset_meta=DatasetMetadata(name="Pilot", authors=["ANCP Lab"]),
    )
    report = engine.run()
    written = {p.name for p in report.files_written}
    assert "dataset_description.json" in written
    assert "README" in written
    assert "CHANGES" in written
    assert "sub-001_scans.tsv" in written
    assert "sub-002_scans.tsv" in written


def test_dataset_description_json_has_required_fields(tmp_path):
    bids = _make_fake_bids(tmp_path / "PILOT")
    BIDSMetadataEngine(bids_root=bids,
                       dataset_meta=DatasetMetadata(name="Pilot")).run()
    dd = json.loads((bids / "dataset_description.json").read_text())
    assert dd["Name"] == "Pilot"
    assert dd["BIDSVersion"]
    assert dd["DatasetType"] in ("raw", "derivative")
    assert any(g["Name"] == "BIDS-Manager" for g in dd["GeneratedBy"])


def test_participants_tsv_includes_demographics(tmp_path):
    bids = _make_fake_bids(tmp_path / "PILOT")
    inv = tmp_path / "inv.tsv"
    _make_inventory_tsv(inv)
    BIDSMetadataEngine(bids_root=bids, inventory_tsv=inv,
                       dataset_meta=DatasetMetadata(name="Pilot")).run()

    df = pd.read_csv(bids / "participants.tsv", sep="\t")
    assert list(df["participant_id"]) == ["sub-001", "sub-002"]
    assert "age" in df.columns
    assert "sex" in df.columns
    assert df.loc[df.participant_id == "sub-001", "age"].iloc[0] == "030Y"
    assert df.loc[df.participant_id == "sub-002", "sex"].iloc[0] == "M"


def test_string_subject_ids_not_coerced_to_int(tmp_path):
    """Regression: BIDS_name like '001' must not become integer 1."""
    bids = _make_fake_bids(tmp_path / "PILOT")
    inv = tmp_path / "inv.tsv"
    _make_inventory_tsv(inv)
    BIDSMetadataEngine(bids_root=bids, inventory_tsv=inv,
                       dataset_meta=DatasetMetadata(name="Pilot")).run()
    df = pd.read_csv(bids / "participants.tsv", sep="\t", dtype=str)
    assert "sub-001" in df["participant_id"].values
    # If coercion happened the lookup would fail and demographic columns
    # would all contain "n/a" or the column would be dropped entirely.
    assert df.loc[df.participant_id == "sub-001", "given_name"].iloc[0] == "Alice"


def test_taskname_autofilled_for_bold(tmp_path):
    bids = _make_fake_bids(tmp_path / "PILOT")
    report = BIDSMetadataEngine(bids_root=bids,
                                dataset_meta=DatasetMetadata(name="Pilot")).run()
    bold_json = json.loads(
        (bids / "sub-001" / "func" / "sub-001_task-rest_bold.json").read_text()
    )
    assert bold_json["TaskName"] == "rest"
    # And it should be reported as a sidecar fill
    fills_paths = {p.name for p, _ in report.sidecar_fills}
    assert "sub-001_task-rest_bold.json" in fills_paths


def test_warns_when_required_field_missing(tmp_path):
    """A bold sidecar without RepetitionTime must trigger a warning."""
    bids = tmp_path / "PILOT"
    func = bids / "sub-001" / "func"
    func.mkdir(parents=True)
    (func / "sub-001_task-x_bold.nii.gz").write_bytes(b"\x00")
    (func / "sub-001_task-x_bold.json").write_text(json.dumps({}))
    report = BIDSMetadataEngine(bids_root=bids,
                                dataset_meta=DatasetMetadata(name="X")).run()
    assert any("RepetitionTime" in w for w in report.warnings)


def test_scans_tsv_lists_acq_time_from_sidecar(tmp_path):
    bids = _make_fake_bids(tmp_path / "PILOT")
    BIDSMetadataEngine(bids_root=bids,
                       dataset_meta=DatasetMetadata(name="Pilot")).run()
    scans = pd.read_csv(bids / "sub-001" / "sub-001_scans.tsv", sep="\t")
    # NIfTIs should be listed with relative paths and timestamps from JSON.
    assert any("anat/sub-001_T1w.nii" in fn for fn in scans["filename"])
    times = set(scans["acq_time"])
    assert "2026-05-07T10:45:00" in times


def test_readme_not_overwritten(tmp_path):
    bids = _make_fake_bids(tmp_path / "PILOT")
    (bids / "README").write_text("CUSTOM CONTENT")
    BIDSMetadataEngine(bids_root=bids,
                       dataset_meta=DatasetMetadata(name="Pilot")).run()
    assert (bids / "README").read_text() == "CUSTOM CONTENT"
