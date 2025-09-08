import shutil
from pathlib import Path

from bids_manager.renaming.schema_renamer import (
    load_bids_schema,
    SeriesInfo,
    build_preview_names,
    apply_post_conversion_rename,
)
from bids_manager.renaming.config import DEFAULT_SCHEMA_DIR, DERIVATIVES_PIPELINE_NAME


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy")


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
        SeriesInfo("001", None, "T1w", "mprage", 1, {"current_bids": "sub-001_orig"}),
        SeriesInfo("001", None, "bold", "fmri_rest", 1, {"current_bids": "sub-001_run1"}),
        SeriesInfo("001", None, "dwi", "ep2d_diff", 1, {"current_bids": "sub-001_raw"}),
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
        SeriesInfo("001", None, "T1w", "mprage", 1, {"current_bids": "sub-001_orig1"}),
        SeriesInfo("001", None, "T1w", "mprage", 1, {"current_bids": "sub-001_orig2"}),
    ]
    proposals = build_preview_names(series, schema)
    rename_map = apply_post_conversion_rename(tmp_path, proposals)
    assert (tmp_path / "sub-001" / "anat" / "sub-001_T1w.nii.gz").exists()
    assert (tmp_path / "sub-001" / "anat" / "sub-001_T1w(2).nii.gz").exists()
