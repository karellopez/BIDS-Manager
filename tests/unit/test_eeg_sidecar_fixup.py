"""Unit tests for the post-write EEG/MEG enrichment fixup."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bidsmgr.fixups.eeg_sidecar import enrich_recording_sidecars
from bidsmgr.recording_meta import (
    AcquisitionSpec,
    AuxChannelSpec,
    FilterSpec,
    RecordingMetaSpec,
    TaskProtocol,
)


def _task(datatype="eeg", basename="sub-001_task-rest_eeg", row_id="r1", task="rest"):
    return SimpleNamespace(
        datatype=datatype, basename=basename, row_id=row_id, entities={"task": task},
    )


def _stage_eeg(tmp_path: Path, datatype="eeg", basename="sub-001_task-rest_eeg") -> Path:
    """Create a synthetic staged datatype dir with the files mne-bids writes."""
    prefix = basename.rsplit("_", 1)[0]
    d = tmp_path / ".tmp_bidsmgr" / "sub-001" / datatype
    d.mkdir(parents=True)
    (d / f"{basename}.json").write_text(
        json.dumps({"SamplingFrequency": 1000.0, "PowerLineFrequency": 50}),
        encoding="utf-8",
    )
    # mne-bids writes TSVs with a UTF-8 BOM; replicate that (encoding
    # "utf-8-sig") so the fixup's BOM tolerance is exercised.
    (d / f"{prefix}_channels.tsv").write_text(
        "name\ttype\tunits\tdescription\n"
        "Fp1\tEEG\tµV\tn/a\n"
        "ECG\tMISC\tn/a\tn/a\n"
        "EOG\tMISC\tn/a\tn/a\n",
        encoding="utf-8-sig",
    )
    (d / f"{prefix}_events.tsv").write_text(
        "onset\tduration\ttrial_type\n"
        "12.0\t0\tS 20\n"
        "72.0\t0\tS 21\n",
        encoding="utf-8-sig",
    )
    return tmp_path / ".tmp_bidsmgr" / "sub-001"


def _full_spec() -> RecordingMetaSpec:
    return RecordingMetaSpec(
        defaults=AcquisitionSpec(
            eeg_reference="Cz",
            eeg_ground="AFz",
            manufacturer="Brain Products",
            amplifier_model="BrainAmp Standard",
            institution_name="University X",
            institution_dept="Psychology",
            filters=[FilterSpec(name="LP", kind="Hardware", info={"cutoff": 260})],
            aux_channels={
                "ECG": AuxChannelSpec(
                    mne_type="ecg", bids_type="ECG", units="mV", description="ecg"
                ),
                "EOG": AuxChannelSpec(bids_type="EOG"),
            },
        ),
        event_maps={"rest": {"S 20": "eyes_open", "S 21": "eyes_closed"}},
        task_protocols={"rest": TaskProtocol(task_description="resting", instructions="relax")},
    )


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def test_noop_when_spec_none(tmp_path):
    staging = _stage_eeg(tmp_path)
    assert enrich_recording_sidecars(staging, [_task()], None) == 0


def test_sidecar_fields_written(tmp_path):
    staging = _stage_eeg(tmp_path)
    n = enrich_recording_sidecars(staging, [_task()], _full_spec())
    assert n > 0
    side = _read_json(staging / "eeg" / "sub-001_task-rest_eeg.json")
    assert side["EEGReference"] == "Cz"
    assert side["EEGGround"] == "AFz"
    assert side["Manufacturer"] == "Brain Products"
    assert side["ManufacturersModelName"] == "BrainAmp Standard"
    assert side["InstitutionName"] == "University X"
    assert side["InstitutionalDepartmentName"] == "Psychology"
    assert side["HardwareFilters"] == {"LP": {"cutoff": 260}}
    assert side["TaskDescription"] == "resting"
    assert side["Instructions"] == "relax"
    # PowerLineFrequency is owned by the backend; the fixup must not touch it.
    assert side["PowerLineFrequency"] == 50


def test_channels_retyped(tmp_path):
    staging = _stage_eeg(tmp_path)
    enrich_recording_sidecars(staging, [_task()], _full_spec())
    text = (staging / "eeg" / "sub-001_task-rest_channels.tsv").read_text()
    lines = {ln.split("\t")[0]: ln.split("\t") for ln in text.strip().splitlines()}
    assert lines["ECG"][1] == "ECG"
    assert lines["ECG"][2] == "mV"
    assert lines["EOG"][1] == "EOG"
    # The real EEG channel is left alone (not generic).
    assert lines["Fp1"][1] == "EEG"


def test_events_mapped_and_json_written(tmp_path):
    staging = _stage_eeg(tmp_path)
    enrich_recording_sidecars(staging, [_task()], _full_spec())
    ev = (staging / "eeg" / "sub-001_task-rest_events.tsv").read_text()
    assert "eyes_open" in ev and "eyes_closed" in ev
    assert "S 20" not in ev
    ev_json = _read_json(staging / "eeg" / "sub-001_task-rest_events.json")
    assert set(ev_json["trial_type"]["Levels"]) == {"eyes_open", "eyes_closed"}


def test_meg_skips_eeg_only_keys(tmp_path):
    staging = _stage_eeg(tmp_path, datatype="meg", basename="sub-001_task-rest_meg")
    spec = RecordingMetaSpec(
        defaults=AcquisitionSpec(eeg_reference="Cz", manufacturer="Elekta"),
    )
    enrich_recording_sidecars(
        staging, [_task(datatype="meg", basename="sub-001_task-rest_meg")], spec,
    )
    side = _read_json(staging / "meg" / "sub-001_task-rest_meg.json")
    assert "EEGReference" not in side  # EEG-only key skipped for MEG
    assert side["Manufacturer"] == "Elekta"  # common key still written


def test_per_row_override_applies(tmp_path):
    staging = _stage_eeg(tmp_path)
    spec = RecordingMetaSpec(
        defaults=AcquisitionSpec(eeg_reference="Cz"),
        overrides={"r1": AcquisitionSpec(eeg_reference="FCz")},
    )
    enrich_recording_sidecars(staging, [_task(row_id="r1")], spec)
    side = _read_json(staging / "eeg" / "sub-001_task-rest_eeg.json")
    assert side["EEGReference"] == "FCz"
