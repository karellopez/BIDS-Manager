import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_sbref_patterns_and_physio_naming(monkeypatch, tmp_path):
    """SBRef patterns should be detected and physio sequences named correctly."""

    from bids_manager import dicom_inventory

    # Ensure default dictionary loads when no preference file exists
    monkeypatch.setattr(dicom_inventory, "SEQ_DICT_FILE", tmp_path / "seq.tsv")
    dicom_inventory.BIDS_PATTERNS = {}
    dicom_inventory.load_sequence_dictionary()

    sbref_pats = dicom_inventory.BIDS_PATTERNS.get("SBRef", ())
    for token in ("sbref", "type-ref", "reference", "refscan", "ref"):
        assert token in sbref_pats
    assert "refscan" not in dicom_inventory.BIDS_PATTERNS

    # Guess modality using new patterns
    assert dicom_inventory.guess_modality("my refscan") == "SBRef"
    assert dicom_inventory.guess_modality("Reference") == "SBRef"

    # Physio naming should produce func/*_physio
    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    series = [SeriesInfo("001", None, "physio", "physio_rest", None, {})]
    proposals = build_preview_names(series, schema)
    assert proposals[0][1] == "func"
    assert proposals[0][2].endswith("_physio")
