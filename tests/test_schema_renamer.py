import gzip
import json
import shutil
import types
from pathlib import Path

import sys

import pandas as pd
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bids_manager.renaming.schema_renamer import (
    load_bids_schema,
    SeriesInfo,
    build_preview_names,
    apply_post_conversion_rename,
    build_series_list_from_dataframe,
    convert_physio_from_proposals,
)
from bids_manager.renaming.config import DEFAULT_SCHEMA_DIR, DERIVATIVES_PIPELINE_NAME
from bids_manager.dicom_inventory import guess_modality


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy")


def _write_physio_dicom(path: Path, series_uid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    file_meta.MediaStorageSOPInstanceUID = "1.2.840.113619.2.5.1762583153.1"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = "PhysioLog"
    ds.PatientName = "Test^Subject"
    ds.AcquisitionTime = "123456"
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path))


def _patch_bidsphysio(monkeypatch, factory):
    pkg = types.ModuleType("bidsphysio")
    dcm2bids_pkg = types.ModuleType("bidsphysio.dcm2bids")
    submodule = types.ModuleType("bidsphysio.dcm2bids.dcm2bidsphysio")
    submodule.dcm2bids = factory
    dcm2bids_pkg.dcm2bidsphysio = submodule
    pkg.dcm2bids = dcm2bids_pkg
    monkeypatch.setitem(sys.modules, "bidsphysio", pkg)
    monkeypatch.setitem(sys.modules, "bidsphysio.dcm2bids", dcm2bids_pkg)
    monkeypatch.setitem(sys.modules, "bidsphysio.dcm2bids.dcm2bidsphysio", submodule)


def create_fake_dataset(root: Path):
    _touch(root / "sub-001" / "anat" / "sub-001_orig.nii.gz")
    _touch(root / "sub-001" / "anat" / "sub-001_orig.json")
    _touch(root / "sub-001" / "func" / "sub-001_run1.nii.gz")
    _touch(root / "sub-001" / "func" / "sub-001_run1.json")
    _touch(root / "sub-001" / "dwi" / "sub-001_raw.nii.gz")
    _touch(root / "sub-001" / "dwi" / "sub-001_raw.json")
    _touch(root / "sub-001" / "dwi" / "sub-001_raw.bval")
    _touch(root / "sub-001" / "dwi" / "sub-001_raw.bvec")
    for suffix in ["ADC", "FA", "TRACEW", "ColFA"]:
        _touch(root / "sub-001" / "dwi" / f"sub-001_raw_{suffix}.nii.gz")
        _touch(root / "sub-001" / "dwi" / f"sub-001_raw_{suffix}.json")
    _touch(root / "sub-001" / "fmap" / "sub-001_echo-1.nii.gz")
    _touch(root / "sub-001" / "fmap" / "sub-001_echo-2.nii.gz")
    _touch(root / "sub-001" / "fmap" / "sub-001_fmap.nii.gz")


def test_schema_renamer_end_to_end(tmp_path):
    create_fake_dataset(tmp_path)
    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    series = [
        SeriesInfo("001", None, "T1w", "mprage", None, {"current_bids": "sub-001_orig"}),
        SeriesInfo("001", None, "bold", "fmri_rest", None, {"current_bids": "sub-001_run1"}),
        SeriesInfo("001", None, "dwi", "ep2d_diff", None, {"current_bids": "sub-001_raw"}),
    ]
    proposals = build_preview_names(series, schema)
    rename_map = apply_post_conversion_rename(tmp_path, proposals)
    assert (tmp_path / "sub-001" / "anat" / "sub-001_T1w.nii.gz").exists()
    assert (tmp_path / "sub-001" / "func" / "sub-001_task-rest_bold.nii.gz").exists()
    assert (tmp_path / "sub-001" / "dwi" / "sub-001_dwi.nii.gz").exists()
    assert (tmp_path / "sub-001" / "dwi" / "sub-001_dwi.bval").exists()
    assert (tmp_path / "sub-001" / "dwi" / "sub-001_dwi.bvec").exists()
    for suffix in ["ADC", "FA", "TRACEW", "ColFA"]:
        out = tmp_path / "derivatives" / DERIVATIVES_PIPELINE_NAME / "sub-001" / "dwi" / f"sub-001_desc-{suffix}_dwi.nii.gz"
        assert out.exists()
    assert (tmp_path / "sub-001" / "fmap" / "sub-001_magnitude1.nii.gz").exists()
    assert (tmp_path / "sub-001" / "fmap" / "sub-001_magnitude2.nii.gz").exists()
    assert (tmp_path / "sub-001" / "fmap" / "sub-001_phasediff.nii.gz").exists()
    rename_map2 = apply_post_conversion_rename(tmp_path, proposals)
    assert rename_map2 == {}


def test_duplicate_names_numbered(tmp_path):
    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    _touch(tmp_path / "sub-001" / "anat" / "sub-001_orig1.nii.gz")
    _touch(tmp_path / "sub-001" / "anat" / "sub-001_orig1.json")
    _touch(tmp_path / "sub-001" / "anat" / "sub-001_orig2.nii.gz")
    _touch(tmp_path / "sub-001" / "anat" / "sub-001_orig2.json")
    series = [
        SeriesInfo("001", None, "T1w", "mprage", None, {"current_bids": "sub-001_orig1"}),
        SeriesInfo("001", None, "T1w", "mprage", 2, {"current_bids": "sub-001_orig2"}),
    ]
    proposals = build_preview_names(series, schema)
    rename_map = apply_post_conversion_rename(tmp_path, proposals)
    assert (tmp_path / "sub-001" / "anat" / "sub-001_T1w.nii.gz").exists()
    assert (tmp_path / "sub-001" / "anat" / "sub-001_T1w_rep-2.nii.gz").exists()


def test_fieldmap_runs_and_task_hits(tmp_path):
    """Fieldmaps with run numbers should keep distinct names and task_hits
    should influence task detection."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)

    # Two fieldmap series with run tokens in their sequence names
    fm1 = SeriesInfo("001", None, "phasediff", "fmap_run-1", None, {})
    fm2 = SeriesInfo("001", None, "phasediff", "fmap_run-2", None, {})

    # Series with custom task hits. The sequence itself has no known task
    # tokens but ``task_hits`` provides a hint.
    task_series = SeriesInfo(
        "001",
        None,
        "bold",
        "customsequence",
        None,
        {"task_hits": "custom"},
    )

    proposals = build_preview_names([fm1, fm2, task_series], schema)

    # Extract basenames for easier assertions
    fmap_bases = [base for (_, dt, base) in proposals[:2]]
    task_base = proposals[2][2]

    assert fmap_bases == [
        "sub-001_run-01_phasediff",
        "sub-001_run-02_phasediff",
    ]
    # Task hit "custom" should be used
    assert task_base == "sub-001_task-custom_bold"


def test_dwi_direction_and_acq_detection():
    """DWI series should capture dir/acq hints from their sequence names."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    series = [
        # Modality "dti" should normalise to dwi and pick up LR/RL directions
        SeriesInfo("001", None, "dti", "DTI_LR", None, {}),
        SeriesInfo("001", None, "dti", "DTI_RL", None, {}),
        # Numbers combined with direction should become an acq label
        SeriesInfo("001", None, "dwi", "15_AP", None, {}),
        SeriesInfo("001", None, "dwi", "15b0_AP", None, {}),
    ]

    proposals = build_preview_names(series, schema)
    bases = [base for (_, _, base) in proposals]

    assert bases == [
        "sub-001_dir-lr_dwi",
        "sub-001_dir-rl_dwi",
        "sub-001_acq-15_dir-ap_dwi",
        "sub-001_acq-15b0_dir-ap_dwi",
    ]


def test_sbref_and_physio_detection():
    """SBRef and physio sequences should not be misclassified as bold."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)

    # SeriesDescriptions containing "bold" tokens should still be detected
    # as SBRef or physio when those hints are present.
    sbref_series = SeriesInfo("001", None, "SBRef", "fmri_sbref", None, {})
    phys_series = SeriesInfo("001", None, "physio", "fmri_physio", None, {})

    proposals = build_preview_names([sbref_series, phys_series], schema)

    (_, dt_sbref, base_sbref), (_, dt_phys, base_phys) = proposals

    assert dt_sbref == "func"
    assert base_sbref.endswith("_sbref")

    assert dt_phys == "func"
    assert base_phys.endswith("_physio")


def test_physio_naming_preserves_task_and_run():
    """Physio recordings should share task/run labels with their BOLD runs."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)

    bold = SeriesInfo("001", None, "bold", "task-two_run-01_bold", None, {})
    phys = SeriesInfo("001", None, "physio", "task-two_run-01_physio", None, {})

    proposals = build_preview_names([bold, phys], schema)

    bases = {base for (_, _, base) in proposals}

    assert "sub-001_task-two_run-01_bold" in bases
    assert "sub-001_task-two_run-01_physio" in bases


def test_guess_modality_prefers_sbref_and_physio():
    """When sequences contain bold tokens, SBRef/physio patterns win."""

    assert guess_modality("fmri_sbref") == "SBRef"
    assert guess_modality("BOLD_SBRef") == "SBRef"
    assert guess_modality("fmri_physio") == "physio"


def test_physio_conversion_creates_outputs(tmp_path, monkeypatch):
    dicom_root = tmp_path / "dicoms"
    dicom_dir = dicom_root / "Session1"
    series_uid = "1.2.3.4"
    _write_physio_dicom(dicom_dir / "physio.dcm", series_uid)

    df = pd.DataFrame(
        [
            {
                "BIDS_name": "sub-001",
                "session": "",
                "sequence": "PhysioLog",
                "modality": "physio",
                "rep": "",
                "source_folder": "Session1",
                "series_uid": series_uid,
                "include": 1,
                "StudyDescription": "Demo",
            }
        ]
    )

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    series_list = build_series_list_from_dataframe(df)
    proposals = build_preview_names(series_list, schema)

    calls = []

    def fake_dcm2bids(path: str):
        calls.append(Path(path))

        class DummyPhysio:
            def save_to_bids(self_inner, prefix: str) -> None:
                prefix_path = Path(prefix)
                prefix_path.parent.mkdir(parents=True, exist_ok=True)
                tsv_path = prefix_path.with_name(f"{prefix_path.name}_physio.tsv.gz")
                with gzip.open(tsv_path, "wt") as f:
                    f.write("cardiac\n0.1\n")
                json_path = prefix_path.with_name(f"{prefix_path.name}_physio.json")
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump({"SamplingFrequency": 1, "StartTime": 0, "Columns": ["cardiac"]}, fh)

        return DummyPhysio()

    _patch_bidsphysio(monkeypatch, fake_dcm2bids)

    bids_root = tmp_path / "bids"
    converted = convert_physio_from_proposals(proposals, bids_root, dicom_root)
    assert converted == 1
    assert calls == [dicom_dir / "physio.dcm"]

    _, datatype, base = proposals[0]
    tsv_path = bids_root / "sub-001" / datatype / f"{base}_physio.tsv.gz"
    json_path = bids_root / "sub-001" / datatype / f"{base}_physio.json"
    assert tsv_path.exists()
    assert json_path.exists()

    # Re-running should detect existing outputs and skip conversion
    second = convert_physio_from_proposals(proposals, bids_root, dicom_root)
    assert second == 0
    assert len(calls) == 1


def test_physio_conversion_respects_include_flag(tmp_path, monkeypatch):
    dicom_root = tmp_path / "dicoms"
    series_uid = "9.9.9"
    _write_physio_dicom(dicom_root / "run" / "physio.dcm", series_uid)

    df = pd.DataFrame(
        [
            {
                "BIDS_name": "sub-002",
                "session": "",
                "sequence": "PhysioLog",
                "modality": "physio",
                "rep": "",
                "source_folder": "run",
                "series_uid": series_uid,
                "include": 0,
                "StudyDescription": "Demo",
            }
        ]
    )

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    series_list = build_series_list_from_dataframe(df)
    proposals = build_preview_names(series_list, schema)

    calls: list[Path] = []

    def fake_dcm2bids(path: str):
        calls.append(Path(path))

        class DummyPhysio:
            def save_to_bids(self_inner, prefix: str) -> None:  # pragma: no cover - should not run
                raise AssertionError("Should not be invoked when include=0")

        return DummyPhysio()

    _patch_bidsphysio(monkeypatch, fake_dcm2bids)

    bids_root = tmp_path / "bids"
    converted = convert_physio_from_proposals(proposals, bids_root, dicom_root)
    assert converted == 0
    assert calls == []
