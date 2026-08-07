"""Tests for the schema-driven, modality-agnostic sidecar repairs.

These assert against what the BIDS schema actually declares rather than against
a hardcoded expectation, because the whole point of the fixup is that it stops
being a hand-written list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.fixups.sidecar_schema import (
    fill_agnostic_fields,
    repair_array_types,
    repair_key_names,
    repair_sidecars,
)
from bidsmgr.recording_meta import RecordingMetaSpec


# ---------------------------------------------------------------------------
# Key casing
# ---------------------------------------------------------------------------


def test_eeg_misc_channel_count_is_respelled() -> None:
    """THE case this fixup was written for.

    BIDS spells the field MISCChannelCount for EEG and MiscChannelCount for MEG
    and iEEG. mne-bids writes the MEG spelling into EEG sidecars, so every EEG
    dataset carried a key the standard does not declare while the declared one
    read as missing. The value was never wrong, only its name.
    """
    data = {"MiscChannelCount": 0, "EEGChannelCount": 65}
    assert repair_key_names(data, "eeg", "eeg") == 1
    assert data["MISCChannelCount"] == 0
    assert "MiscChannelCount" not in data
    assert data["EEGChannelCount"] == 65


def test_meg_keeps_its_own_spelling() -> None:
    """The other half, and the reason this cannot be a blanket rename: the MEG
    spelling is correct FOR MEG. A fixup that 'corrected' it would break the
    datatype it came from."""
    data = {"MiscChannelCount": 2}
    assert repair_key_names(data, "meg", "meg") == 0
    assert data == {"MiscChannelCount": 2}


def test_a_key_the_schema_does_not_declare_is_left_alone() -> None:
    """Renaming is driven by the schema, so a private key keeps its name and
    its value rather than being guessed at."""
    data = {"MyLabsOwnField": 1, "bidsguess": "x"}
    assert repair_key_names(data, "anat", "T1w") == 0
    assert data == {"MyLabsOwnField": 1, "bidsguess": "x"}


def test_the_correct_spelling_wins_when_both_are_present() -> None:
    data = {"MiscChannelCount": 9, "MISCChannelCount": 0}
    repair_key_names(data, "eeg", "eeg")
    assert data == {"MISCChannelCount": 0}


# ---------------------------------------------------------------------------
# Array types
# ---------------------------------------------------------------------------


def test_a_scalar_is_wrapped_where_the_schema_wants_an_array() -> None:
    """dcm2niix writes a bare number for a single-frame PET acquisition where
    the schema declares one entry per frame."""
    data = {"FrameDuration": 60.0}
    assert repair_array_types(data, "pet", "pet") == 1
    assert data["FrameDuration"] == [60.0]


def test_an_existing_list_is_untouched() -> None:
    data = {"FrameDuration": [10.0, 20.0]}
    assert repair_array_types(data, "pet", "pet") == 0
    assert data["FrameDuration"] == [10.0, 20.0]


def test_a_field_the_schema_leaves_untyped_is_not_reshaped() -> None:
    """EchoTime is declared number-or-array. The schema permits both, so
    choosing one for the converter would be meddling, not repair."""
    data = {"EchoTime": 0.03}
    repair_array_types(data, "anat", "T1w")
    assert data["EchoTime"] == 0.03


# ---------------------------------------------------------------------------
# Agnostic fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("datatype, suffix", [
    ("anat", "T1w"), ("func", "bold"), ("pet", "pet"), ("eeg", "eeg"),
])
def test_institution_reaches_every_datatype(datatype, suffix) -> None:
    """Where the study was done has nothing to do with what recorded it, and
    the schema declares these fields for every datatype. They used to reach
    electrophysiology alone, because that was the only enrichment pass that
    ran."""
    spec = RecordingMetaSpec()
    spec.defaults.institution_name = "Uni Oldenburg"
    spec.defaults.institution_dept = "Neuropsychology"

    data: dict = {}
    assert fill_agnostic_fields(data, datatype, suffix, spec) == 2
    assert data["InstitutionName"] == "Uni Oldenburg"
    assert data["InstitutionalDepartmentName"] == "Neuropsychology"


def test_a_value_already_present_is_never_overwritten() -> None:
    """A converter that read the institution out of a DICOM header knows the
    scanner's own answer; a dataset-wide default must not clobber it."""
    spec = RecordingMetaSpec()
    spec.defaults.institution_name = "Default Site"
    data = {"InstitutionName": "From The DICOM Header"}
    fill_agnostic_fields(data, "anat", "T1w", spec)
    assert data["InstitutionName"] == "From The DICOM Header"


def test_no_spec_writes_nothing() -> None:
    data: dict = {}
    assert fill_agnostic_fields(data, "anat", "T1w", None) == 0
    assert data == {}


# ---------------------------------------------------------------------------
# End to end over a tree
# ---------------------------------------------------------------------------


def test_repair_sidecars_walks_the_tree(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    eeg = root / "sub-001" / "eeg"
    anat = root / "sub-001" / "anat"
    eeg.mkdir(parents=True)
    anat.mkdir(parents=True)
    (eeg / "sub-001_task-rest_eeg.json").write_text(
        json.dumps({"MiscChannelCount": 0, "SamplingFrequency": 500})
    )
    (anat / "sub-001_T1w.json").write_text(json.dumps({"EchoTime": 0.03}))

    spec = RecordingMetaSpec()
    spec.defaults.institution_name = "Uni Oldenburg"

    changed = repair_sidecars(root, [], spec)
    assert changed == 2

    eeg_data = json.loads((eeg / "sub-001_task-rest_eeg.json").read_text())
    assert eeg_data["MISCChannelCount"] == 0
    assert "MiscChannelCount" not in eeg_data
    assert eeg_data["InstitutionName"] == "Uni Oldenburg"

    # The MRI sidecar gets the agnostic value too: this is the datatype the
    # old enrichment pass skipped entirely.
    anat_data = json.loads((anat / "sub-001_T1w.json").read_text())
    assert anat_data["InstitutionName"] == "Uni Oldenburg"
    assert anat_data["EchoTime"] == 0.03


def test_rerunning_changes_nothing(tmp_path: Path) -> None:
    """Idempotent: a second pass over a repaired tree must be a no-op, or the
    convert step would report work it did not do."""
    root = tmp_path / "staging"
    eeg = root / "sub-001" / "eeg"
    eeg.mkdir(parents=True)
    (eeg / "sub-001_task-rest_eeg.json").write_text(
        json.dumps({"MiscChannelCount": 0})
    )
    assert repair_sidecars(root, [], None) == 1
    assert repair_sidecars(root, [], None) == 0
