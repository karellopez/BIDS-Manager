import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bids_manager.renaming.schema_renamer import (
    load_bids_schema,
    SeriesInfo,
    build_preview_names,
    apply_post_conversion_rename,
)


def create_dummy_dataset(root: Path):
    (root / "anat").mkdir()
    (root / "func").mkdir()
    (root / "dwi").mkdir()
    (root / "fmap").mkdir()

    # Anatomical
    (root / "anat" / "sub-001_mprage.nii.gz").write_text("")
    (root / "anat" / "sub-001_mprage.json").write_text("{}")

    # Functional (sequence sanitized to match pattern)
    (root / "func" / "sub-001_fmrirest.nii.gz").write_text("")
    (root / "func" / "sub-001_fmrirest.json").write_text("{}")

    # DWI raw
    (root / "dwi" / "sub-001_ep2ddiff.nii.gz").write_text("")
    (root / "dwi" / "sub-001_ep2ddiff.bval").write_text("0 0")
    (root / "dwi" / "sub-001_ep2ddiff.bvec").write_text("0 0 0")
    (root / "dwi" / "sub-001_ep2ddiff.json").write_text("{}")

    # Fieldmaps
    for tag in ["echo-1", "echo-2", "fmap"]:
        (root / "fmap" / f"sub-001_{tag}.nii.gz").write_text("")
        (root / "fmap" / f"sub-001_{tag}.json").write_text("{}")


def build_df():
    return pd.DataFrame([
        {"subject": "001", "modality": "T1w", "sequence": "mprage", "rep": 1},
        {"subject": "001", "modality": "bold", "sequence": "fmri_rest", "rep": 1},
        {"subject": "001", "modality": "dwi", "sequence": "ep2d_diff", "rep": 1},
    ])


def test_schema_renamer_preview_and_apply(tmp_path):
    create_dummy_dataset(tmp_path)
    schema = load_bids_schema(Path("bids_manager/miscellaneous/schema"))
    df = build_df()
    series = [
        SeriesInfo(r.subject, None, r.modality, r.sequence, r.rep, {})
        for r in df.itertuples()
    ]
    proposals = build_preview_names(series, schema)
    assert any(dt == "func" and "task-" in base for (_, dt, base) in proposals)

    rename_map = apply_post_conversion_rename(tmp_path, proposals)
    assert (tmp_path / "anat" / "sub-001_T1w.nii.gz").exists()
    assert (tmp_path / "func" / "sub-001_task-rest_bold.nii.gz").exists()
    assert (tmp_path / "dwi" / "sub-001_dwi.nii.gz").exists()
    assert (tmp_path / "dwi" / "sub-001_dwi.bval").exists()
    assert (tmp_path / "dwi" / "sub-001_dwi.bvec").exists()
    fmap_dir = tmp_path / "fmap"
    assert (fmap_dir / "sub-001_magnitude1.nii.gz").exists()
    assert (fmap_dir / "sub-001_magnitude2.nii.gz").exists()
    assert (fmap_dir / "sub-001_phasediff.nii.gz").exists()

    # Idempotency
    rename_map2 = apply_post_conversion_rename(tmp_path, proposals)
    assert rename_map2 == {}
