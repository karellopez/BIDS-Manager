from __future__ import annotations

from pathlib import Path

import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from bids_manager.physio_conversion import (
    PHYSIO_PRIVATE_TAG,
    convert_physiological_data,
)


def _write_dicom(path: Path, series_description: str, *, include_physio: bool = True) -> None:
    """Create a minimal DICOM file for testing purposes."""

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SeriesDescription = series_description
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 1
    ds.AcquisitionNumber = 1
    ds.PatientName = "Test^Physio"
    ds.Modality = "MR"
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    if include_physio:
        # Use arbitrary but sufficiently long payload so detection succeeds
        ds.add_new(PHYSIO_PRIVATE_TAG, "OB", b"\0" * 2048)

    ds.save_as(str(path))


class _RecordingSpy:
    """Simple helper storing the prefix passed to ``save_to_bids``."""

    def __init__(self, store: list[str]) -> None:
        self._store = store

    def save_to_bids(self, prefix: str) -> None:
        self._store.append(prefix)


def test_convert_physio_creates_bids_prefix(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    bids_root = tmp_path / "bids"
    subject_dir = raw_root / "subjectA"
    subject_dir.mkdir(parents=True)

    dicom_path = subject_dir / "physio.dcm"
    _write_dicom(dicom_path, "task-rest_run-01_physio")

    saved_prefixes: list[str] = []

    def fake_converter(paths):
        assert paths == [str(dicom_path)]
        return _RecordingSpy(saved_prefixes)

    created = convert_physiological_data(
        raw_root,
        "subjectA",
        bids_root,
        "sub-001",
        converter=fake_converter,
    )

    expected_prefix = bids_root / "sub-001" / "func" / "sub-001_task-rest_run-01_physio"
    assert [Path(p) for p in saved_prefixes] == [expected_prefix]

    expected_tsv = expected_prefix.with_name("sub-001_task-rest_run-01_physio.tsv.gz")
    assert created == [expected_tsv]


def test_session_hint_preserved(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    bids_root = tmp_path / "bids"
    session_dir = raw_root / "subjectA" / "ses-02"
    session_dir.mkdir(parents=True)

    dicom_path = session_dir / "physio.dcm"
    _write_dicom(dicom_path, "task-movie_physio")

    saved_prefixes: list[str] = []

    def fake_converter(paths):
        assert paths == [str(dicom_path)]
        return _RecordingSpy(saved_prefixes)

    created = convert_physiological_data(
        raw_root,
        "subjectA/ses-02",
        bids_root,
        "sub-001",
        converter=fake_converter,
    )

    expected_prefix = bids_root / "sub-001" / "ses-02" / "func" / "sub-001_ses-02_task-movie_physio"
    assert [Path(p) for p in saved_prefixes] == [expected_prefix]

    expected_tsv = expected_prefix.with_name("sub-001_ses-02_task-movie_physio.tsv.gz")
    assert created == [expected_tsv]


def test_no_conversion_without_payload(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    bids_root = tmp_path / "bids"
    subject_dir = raw_root / "subjectA"
    subject_dir.mkdir(parents=True)

    dicom_path = subject_dir / "localizer.dcm"
    _write_dicom(dicom_path, "localizer", include_physio=False)

    def fake_converter(_paths):
        pytest.fail("converter should not be called when no physio data is present")

    created = convert_physiological_data(
        raw_root,
        "subjectA",
        bids_root,
        "sub-001",
        converter=fake_converter,
    )

    assert created == []
