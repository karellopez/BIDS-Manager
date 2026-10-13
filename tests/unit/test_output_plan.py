"""What the conversion will write, predicted before it runs.

The prediction was checked against real conversions of 49 vendor test sets
(dcm2niix QA, ICDC-Glioma, UPENN-GBM) and the multimodal sample: 39 matched
file for file, the rest differ only by vendor-specific files the conversion
does not rename. These tests pin the conventions that made it match.
"""

from __future__ import annotations

import json

import pandas as pd

from bidsmgr.classifier.acquisition.features import SeriesFeatures
from bidsmgr.converter.output_plan import UNKNOWN, PlanOptions, approximate, plan_output


def _row(sub, name, datatype, *, ses="", uid="", source="", modality="mri", include="1", **extra):
    row = {"participant_id": sub, "session": ses, "datatype": datatype,
           "bids_name": name, "bids_guess_suffix": name.rsplit("_", 1)[-1],
           "include": include, "modality": modality, "series_uid": uid, "source_file": source,
           "sequence": name, "acq_time": "", "study_date": "", "dataset": "ds"}
    row.update(extra)
    return row


def _paths(plan, *, sure: bool = True) -> set[str]:
    return {f.path for f in plan.files if (not f.conditional) == sure}


def test_an_mri_session_with_its_companions() -> None:
    rows = [
        _row("sub-01", "sub-01_ses-a_T1w", "anat", ses="ses-a", uid="u1"),
        _row("sub-01", "sub-01_ses-a_task-rest_bold", "func", ses="ses-a", uid="u2", task="rest"),
        _row("sub-01", "sub-01_ses-a_dir-AP_dwi", "dwi", ses="ses-a", uid="u3"),
        _row("sub-01", "sub-01_ses-a_dir-PA_epi", "fmap", ses="ses-a", uid="u4"),
        _row("sub-01", "sub-01_ses-a_magnitude1", "fmap", ses="ses-a", uid="u5|u6"),
        _row("sub-01", "sub-01_ses-a_localizer", "anat", ses="ses-a", uid="u7", include="0"),
    ]
    feats = {"u4": SeriesFeatures(image_type=["ORIGINAL", "PRIMARY", "DIFFUSION"]),
             "u5": SeriesFeatures(echo_times=[4.92, 7.38])}
    plan = plan_output(pd.DataFrame(rows), features=feats)
    sure = _paths(plan)
    folder = "sub-01/ses-a"
    for path in ("anat/sub-01_ses-a_T1w.nii.gz", "anat/sub-01_ses-a_T1w.json",
                 "func/sub-01_ses-a_task-rest_bold.nii.gz",
                 "dwi/sub-01_ses-a_dir-AP_dwi.bval", "dwi/sub-01_ses-a_dir-AP_dwi.bvec",
                 "fmap/sub-01_ses-a_dir-PA_epi.bval",
                 "fmap/sub-01_ses-a_magnitude2.nii.gz", "fmap/sub-01_ses-a_phasediff.json",
                 "sub-01_ses-a_scans.tsv"):
        assert f"{folder}/{path}" in sure, path
    assert not any("localizer" in p for p in sure), "a row left out writes nothing"
    assert {"dataset_description.json", ".bidsignore", "participants.tsv", "participants.json"} <= sure
    assert {"README", "CHANGES"} <= _paths(plan, sure=False), "written unless there is one"


def test_a_plain_epi_reference_gets_no_gradient_tables() -> None:
    """GE stores a b-value of 0 on a plain EPI: that is not diffusion."""
    rows = [_row("sub-01", "sub-01_dir-AP_epi", "fmap", uid="u1")]
    plan = plan_output(pd.DataFrame(rows), features={"u1": SeriesFeatures(b_values=[0.0])})
    assert not any(p.endswith((".bval", ".bvec")) for p in _paths(plan))


def test_a_series_with_several_echoes_is_split_and_said_so() -> None:
    rows = [_row("sub-01", "sub-01_task-x_bold", "func", uid="u1")]
    plan = plan_output(pd.DataFrame(rows), features={"u1": SeriesFeatures(echo_times=[12.0, 30.0])})
    split = [f for f in plan.files if "_e" in f.path]
    assert sorted(f.path.rsplit("/", 1)[-1] for f in split) == [
        "sub-01_task-x_bold_e1.json", "sub-01_task-x_bold_e1.nii.gz",
        "sub-01_task-x_bold_e2.json", "sub-01_task-x_bold_e2.nii.gz"]
    assert all("not included" in f.warning for f in split)
    assert "sub-01/func/sub-01_task-x_bold.nii.gz" not in _paths(plan)


def test_physio_is_one_file_per_signal_when_it_has_several() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_physio", "func", uid="p1")]
    plan = plan_output(pd.DataFrame(rows))
    maybe = _paths(plan, sure=False)
    for signal in ("cardiac", "ecg", "respiratory", "trigger"):
        assert f"sub-01/func/sub-01_task-rest_recording-{signal}_physio.tsv.gz" in maybe
    assert "sub-01/func/sub-01_task-rest_physio.tsv.gz" in maybe
    assert not any(p.endswith("_scans.tsv") for p in _paths(plan)), \
        "physio is not a row of the scans table"


def test_eeg_and_meg_as_mne_bids_writes_them() -> None:
    rows = [_row("sub-01", "sub-01_ses-1_task-rest_eeg", "eeg", ses="ses-1",
                 source="raw/rest.vhdr", modality="eeg"),
            _row("sub-01", "sub-01_ses-2_task-rest_meg", "meg", ses="ses-2",
                 source="raw/rest.fif", modality="meg")]
    plan = plan_output(pd.DataFrame(rows))
    sure, maybe = _paths(plan), _paths(plan, sure=False)
    for ext in (".vhdr", ".eeg", ".vmrk", ".json"):
        assert f"sub-01/ses-1/eeg/sub-01_ses-1_task-rest_eeg{ext}" in sure
    assert "sub-01/ses-1/eeg/sub-01_ses-1_task-rest_channels.tsv" in sure
    assert "sub-01/ses-1/eeg/sub-01_ses-1_task-rest_events.tsv" in maybe
    # Positions describe the session, not the run: no task in the name.
    assert "sub-01/ses-1/eeg/sub-01_ses-1_electrodes.tsv" in maybe
    assert "sub-01/ses-2/meg/sub-01_ses-2_coordsystem.json" in maybe
    assert "sub-01/ses-2/meg/sub-01_ses-2_task-rest_meg.fif" in sure
    assert {"sub-01/ses-1/sub-01_ses-1_scans.tsv", "sub-01/ses-2/sub-01_ses-2_scans.tsv"} <= sure


def test_force_edf_changes_the_eeg_file() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_eeg", "eeg", source="rest.cnt", modality="eeg")]
    assert "sub-01/eeg/sub-01_task-rest_eeg.vhdr" in _paths(plan_output(pd.DataFrame(rows)))
    plan = plan_output(pd.DataFrame(rows), options=PlanOptions(force_edf=True))
    assert "sub-01/eeg/sub-01_task-rest_eeg.edf" in _paths(plan)


def test_eye_tracking_writes_two_files_per_eye() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_recording-eye1_physio", "beh",
                 source="rec.edf", modality="eyetrack")]
    plan = plan_output(pd.DataFrame(rows))
    sure, maybe = _paths(plan), _paths(plan, sure=False)
    assert "sub-01/beh/sub-01_task-rest_recording-eye1_physio.tsv.gz" in sure
    assert "sub-01/beh/sub-01_task-rest_recording-eye1_physioevents.json" in sure
    assert "sub-01/beh/sub-01_task-rest_recording-eye2_physio.tsv.gz" in maybe
    assert not any(p.endswith("_scans.tsv") for p in sure)


def test_the_metadata_step_and_sourcedata_follow_the_settings() -> None:
    rows = [_row("sub-01", "sub-01_T1w", "anat", uid="u1", sequence="t1_mprage")]
    off = plan_output(pd.DataFrame(rows), options=PlanOptions(run_metadata=False))
    assert "participants.tsv" not in _paths(off)
    assert not any(p.endswith("_scans.tsv") for p in _paths(off)), \
        "an MRI session's scans table is the metadata step's"
    eeg = [_row("sub-01", "sub-01_task-x_eeg", "eeg", source="r.edf", modality="eeg")]
    assert "sub-01/sub-01_scans.tsv" in _paths(plan_output(
        pd.DataFrame(eeg), options=PlanOptions(run_metadata=False))), "MNE-BIDS writes it"
    copy = plan_output(pd.DataFrame(rows), options=PlanOptions(sourcedata="copy"))
    assert "sourcedata/sub-01/mri/t1_mprage/" in _paths(copy)
    assert "sourcedata/sub-01/manifest.tsv" in _paths(copy)
    archive = plan_output(pd.DataFrame(rows), options=PlanOptions(sourcedata="archive"))
    assert "sourcedata/sub-01/mri/t1_mprage.tar.gz" in _paths(archive)
    other = plan_output(pd.DataFrame(rows), options=PlanOptions(sourcedata="copy",
                                                                 sourcedata_modalities=("eeg",)))
    assert not any(p.startswith("sourcedata/sub-01/mri") for p in _paths(other))


# ---------------------------------------------------------------------------
# What a file will contain
# ---------------------------------------------------------------------------


def _fields(preview) -> dict:
    return {k: (v, s) for k, v, s in preview.fields}


def test_a_bold_sidecar_from_the_scan_the_table_and_the_settings() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_bold", "func", uid="b1"),
            _row("sub-01", "sub-01_phasediff", "fmap", uid="f1"),
            ]
    feats = {"b1": SeriesFeatures(manufacturer="SIEMENS", model="Prisma", field_strength=3.0,
                                  repetition_time=0.8, echo_times=[30.0], flip_angle=52.0,
                                  is_epi=True, pe_axis="COL", pe_positive=0.0,
                                  multiband=8.0, has_slice_timing=True)}
    df = pd.DataFrame(rows)
    plan = plan_output(df, features=feats)
    bold = plan.find("sub-01/func/sub-01_task-rest_bold.json")
    f = _fields(approximate(df, bold, PlanOptions(fill_todos=False), feats))
    assert f["TaskName"] == ("rest", "settings")
    assert f["Manufacturer"] == ("SIEMENS", "scan")
    assert f["RepetitionTime"] == (0.8, "scan")
    assert f["EchoTime"] == (0.03, "scan"), "ms in the scan, seconds in BIDS"
    assert f["PhaseEncodingDirection"] == ("j-", "scan")
    assert f["SliceTiming"] == (UNKNOWN, "converter")
    assert f["TotalReadoutTime"] == (UNKNOWN, "converter")
    assert f["B0FieldSource"][1] == "settings"
    assert not any(s == "todo" for _v, s in f.values())


def test_fieldmap_links_follow_the_settings() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_bold", "func", uid="b1", acq_time="100000"),
            _row("sub-01", "sub-01_phasediff", "fmap", uid="f1", acq_time="095000")]
    df = pd.DataFrame(rows)
    feats = {"b1": SeriesFeatures(), "f1": SeriesFeatures(echo_times=[4.92, 7.38])}
    plan = plan_output(df, features=feats)
    fmap = plan.find("sub-01/fmap/sub-01_phasediff.json")
    uri = _fields(approximate(df, fmap, PlanOptions(fill_todos=False), feats))
    assert uri["IntendedFor"][0] == ["bids::sub-01/func/sub-01_task-rest_bold.nii.gz"]
    assert uri["B0FieldIdentifier"] == ("fmap1", "settings")
    assert uri["EchoTime1"] == (0.00492, "scan") and uri["EchoTime2"] == (0.00738, "scan")
    rel = _fields(approximate(df, fmap, PlanOptions(intended_for="relative", b0_field=False,
                                                    fill_todos=False), feats))
    assert rel["IntendedFor"][0] == ["func/sub-01_task-rest_bold.nii.gz"]
    assert "B0FieldIdentifier" not in rel


def test_fill_todos_adds_the_placeholders_the_metadata_step_writes() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_eeg", "eeg", source="r.edf", modality="eeg",
                 sfreq="500", duration_sec="61.5", line_freq="50")]
    df = pd.DataFrame(rows)
    plan = plan_output(df)
    sidecar = plan.find("sub-01/eeg/sub-01_task-rest_eeg.json")
    on = _fields(approximate(df, sidecar, PlanOptions(fill_todos=True)))
    off = _fields(approximate(df, sidecar, PlanOptions(fill_todos=False)))
    assert on["SamplingFrequency"] == (500.0, "scan")
    assert on["PowerLineFrequency"] == (50.0, "settings")
    assert on["RecordingDuration"] == (61.5, "scan")
    todos = {k for k, (_v, s) in on.items() if s == "todo"}
    assert "TaskDescription" in todos and "Manufacturer" in todos
    assert not any(s == "todo" for _v, s in off.values())


def test_dataset_files() -> None:
    rows = [_row("sub-02", "sub-02_T1w", "anat", uid="u1", PatientAge="034Y", PatientSex="F"),
            _row("sub-01", "sub-01_T1w", "anat", uid="u2", PatientSex="M")]
    df = pd.DataFrame(rows)
    plan = plan_output(df, options=PlanOptions(dataset="study", bids_version="1.11.1"))
    desc = _fields(approximate(df, plan.find("dataset_description.json"),
                               PlanOptions(dataset="study", bids_version="1.11.1", version="9.9")))
    assert desc["Name"] == ("study", "settings") and desc["BIDSVersion"][0] == "1.11.1"
    assert desc["License"] == ("TODO", "todo")
    assert "Authors" not in desc, "optional: not in the default fill scope"
    everything = _fields(approximate(df, plan.find("dataset_description.json"),
                                     PlanOptions(fill_scope="optional")))
    assert everything["Authors"] == (["TODO"], "todo")
    assert [g["Description"] for g in desc["GeneratedBy"][0]] == ["conversion", "metadata engine"]
    parts = approximate(df, plan.find("participants.tsv"))
    assert parts.header == ("participant_id", "age", "sex")
    assert parts.rows == (("sub-01", "n/a", "M"), ("sub-02", "34", "F"))
    codebook = _fields(approximate(df, plan.find("participants.json")))
    assert set(codebook) == {"participant_id", "age", "sex"}
    from bidsmgr.metadata.engine import readme_text

    assert approximate(df, plan.find("README")).text == readme_text("<dataset>")
    assert approximate(df, plan.find(".bidsignore")).text == ".bidsmgr/\n.tmp_bidsmgr/\n"


def test_the_scans_table() -> None:
    rows = [_row("sub-01", "sub-01_T1w", "anat", uid="u1", study_date="20230914",
                 acq_time="102520.255"),
            _row("sub-01", "sub-01_dwi", "dwi", uid="u2", study_date="20230914", acq_time="103842")]
    df = pd.DataFrame(rows)
    plan = plan_output(df)
    scans = approximate(df, plan.find("sub-01/sub-01_scans.tsv"))
    assert scans.header == ("filename", "acq_time")
    assert scans.rows == (("anat/sub-01_T1w.nii.gz", "2023-09-14T10:25:20.255000"),
                          ("dwi/sub-01_dwi.bval", "2023-09-14T10:38:42"),
                          ("dwi/sub-01_dwi.bvec", "2023-09-14T10:38:42"),
                          ("dwi/sub-01_dwi.nii.gz", "2023-09-14T10:38:42"))


def test_events_name_the_labels_given_to_event_codes() -> None:
    class Spec:
        event_maps = {"task-x": {"1": "left", "2": "right"}}

    rows = [_row("sub-01", "sub-01_task-x_eeg", "eeg", source="r.edf", modality="eeg")]
    df = pd.DataFrame(rows)
    plan = plan_output(df)
    events = approximate(df, plan.find("sub-01/eeg/sub-01_task-x_events.tsv"), spec=Spec())
    assert events.header[:3] == ("onset", "duration", "trial_type")
    assert [r[2] for r in events.rows] == ["left", "right"]


def test_json_text_reads_as_the_file_would() -> None:
    rows = [_row("sub-01", "sub-01_T1w", "anat", uid="u1")]
    df = pd.DataFrame(rows)
    plan = plan_output(df, features={"u1": SeriesFeatures(manufacturer="GE MEDICAL SYSTEMS")})
    preview = approximate(df, plan.find("sub-01/anat/sub-01_T1w.json"),
                          PlanOptions(fill_todos=False), {"u1": SeriesFeatures(manufacturer="GE")})
    data = json.loads(preview.json_text())
    assert data["Modality"] == "MR" and data["Manufacturer"] == "GE"


def test_a_large_table_is_planned_quickly() -> None:
    import time

    rows = []
    for i in range(1500):
        rows.append(_row(f"sub-{i:04d}", f"sub-{i:04d}_task-rest_bold", "func", uid=f"u{i}"))
    start = time.perf_counter()
    plan = plan_output(pd.DataFrame(rows))
    assert time.perf_counter() - start < 2.0
    assert len(plan.files) > 4500
