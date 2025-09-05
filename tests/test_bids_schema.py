import pytest

from bids_manager.bids_schema import parse_filename, build_filename


def roundtrip(name: str):
    parts = parse_filename(name, strict=False)
    rebuilt = build_filename(parts["entities"], parts["suffix"], parts["extension"])
    return parts, rebuilt


def test_anat_roundtrip():
    name = "sub-01_acq-highres_T1w.nii.gz"
    parts, rebuilt = roundtrip(name)
    assert parts["entities"] == {"sub": "01", "acq": "highres"}
    assert parts["suffix"] == "T1w"
    assert parts["extension"] == ".nii.gz"
    assert rebuilt == name


def test_func_roundtrip():
    name = "sub-01_ses-02_task-rest_run-1_bold.nii.gz"
    parts, rebuilt = roundtrip(name)
    assert parts["entities"]["task"] == "rest"
    assert parts["entities"]["run"] == "1"
    assert parts["suffix"] == "bold"
    assert rebuilt == name


def test_fieldmap_with_rep():
    name = "sub-01_ses-02_echo-1_rep-2_fmap.nii.gz"
    parts = parse_filename(name, strict=False)
    assert parts["entities"]["echo"] == "1"
    assert parts["entities"]["rep"] == "2"
    entities = dict(parts["entities"])
    entities.pop("echo")
    new_name = build_filename(entities, "magnitude1", parts["extension"], strict=False)
    assert new_name == "sub-01_ses-02_magnitude1_rep-2.nii.gz"

