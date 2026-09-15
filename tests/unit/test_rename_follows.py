"""Everything a rename has to carry with it, beyond the filenames.

BIDS names a file from inside other files in more ways than one, and for a
long time only three were followed: ``IntendedFor``, the ``filename`` column of
a ``*_scans.tsv``, and ``participant_id``. Everything else was left pointing at
a value that no longer existed, and none of it was reported, which is the worst
shape a defect can take: the dataset validates and is wrong.

Four kinds are covered here.

* **Path-valued sidecar fields.** ``AssociatedEmptyRoom``, ``Sources``,
  ``RawSources`` and ``AnatomicalImage`` are BIDS URIs exactly like
  ``IntendedFor``, and a rename breaks them exactly the same way.
* **``session_id``** in ``sub-XX_sessions.tsv``, which is ``participant_id``
  one level down.
* **``sample_id``** in ``samples.tsv``, the same again for microscopy.
* **``TaskName``**, where the sidecar holds the human name the filename label
  is derived from, so renaming one without the other puts them out of step.

The negative cases matter as much as the positive ones here, because each of
these edits reaches into a file the user did not name, and a rule that is too
eager silently rewrites metadata somebody wrote by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import rename as rn
from bidsmgr.editor.tsv_edit import read_table
from bidsmgr.project.operations import undo_last


def _write(root: Path, rel: str, text: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _column(path: Path, column: str) -> list[str]:
    table = read_table(path)
    assert table is not None, f"{path} is not a readable table"
    return list(table.column(column))


def _names(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and ".bidsmgr" not in p.parts
    }


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """One subject, one session, and every kind of back-reference."""
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "participants.tsv", "participant_id\tage\nsub-001\t31\n")
    _write(root, "sub-001/sub-001_sessions.tsv",
           "session_id\tacq_date\nses-01\t2026-01-01\n")

    _write(root, "sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz")
    _write(root, "sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.json",
           json.dumps({"TaskName": "rest", "RepetitionTime": 2.0}))

    _write(root, "sub-001/ses-01/meg/sub-001_ses-01_task-noise_meg.fif")
    _write(root, "sub-001/ses-01/meg/sub-001_ses-01_task-rest_meg.fif")
    _write(root, "sub-001/ses-01/meg/sub-001_ses-01_task-rest_meg.json",
           json.dumps({
               "TaskName": "rest",
               "AssociatedEmptyRoom":
                   "bids::sub-001/ses-01/meg/sub-001_ses-01_task-noise_meg.fif",
           }))

    _write(root, "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz")
    _write(root, "sub-001/ses-01/anat/sub-001_ses-01_T1w.json", json.dumps({
        "Sources": [
            "bids::sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz",
        ],
    }))
    _write(root, "sub-001/ses-01/sub-001_ses-01_scans.tsv",
           "filename\tacq_time\n"
           "func/sub-001_ses-01_task-rest_bold.nii.gz\tA\n")
    return root


# ---------------------------------------------------------------------------
# TaskName


def test_renaming_a_task_updates_the_sidecar_name(dataset: Path) -> None:
    """The filename said ``task-nback`` and the sidecar still said "rest".

    BIDS makes the two a pair: the label "MAY be derived from this TaskName
    field by removing all non-alphanumeric or + characters". Nothing reported
    the mismatch, and analysis code reads TaskName, not the filename.
    """
    rn.apply_rename(dataset, rn.plan_rename(dataset, "task", "rest", "nback"))

    bold = dataset / "sub-001/ses-01/func/sub-001_ses-01_task-nback_bold.json"
    assert _json(bold)["TaskName"] == "nback"
    assert _json(bold)["RepetitionTime"] == 2.0, "the rest is untouched"


def test_every_sidecar_for_that_task_follows(dataset: Path) -> None:
    """Not just the first one found: MEG and func both carry the task."""
    rn.apply_rename(dataset, rn.plan_rename(dataset, "task", "rest", "nback"))
    meg = dataset / "sub-001/ses-01/meg/sub-001_ses-01_task-nback_meg.json"
    assert _json(meg)["TaskName"] == "nback"


def test_a_derived_task_name_is_still_followed(tmp_path: Path) -> None:
    """``"faces n-back"`` derives to ``facesnback``, so a file labelled that
    way IS in step with its sidecar and both move together."""
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "sub-001/func/sub-001_task-facesnback_bold.nii.gz")
    _write(root, "sub-001/func/sub-001_task-facesnback_bold.json",
           json.dumps({"TaskName": "faces n-back"}))

    rn.apply_rename(
        root, rn.plan_rename(root, "task", "facesnback", "nback"),
    )
    assert _json(
        root / "sub-001/func/sub-001_task-nback_bold.json"
    )["TaskName"] == "nback"


def test_a_task_name_that_does_not_match_is_left_alone(
    tmp_path: Path,
) -> None:
    """The negative case, and the reason the rule is derivation and not
    "always overwrite".

    A ``TaskName`` that does not derive to the old label was ALREADY out of
    step with the filename before this rename, so the rename is not what broke
    it. Overwriting would be the tool asserting something nobody said.
    """
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "sub-001/func/sub-001_task-rest_bold.nii.gz")
    _write(root, "sub-001/func/sub-001_task-rest_bold.json",
           json.dumps({"TaskName": "something else entirely"}))

    plan = rn.plan_rename(root, "task", "rest", "nback")
    assert not [e for e in plan.content_edits if e.what == "TaskName"]
    rn.apply_rename(root, plan)
    assert _json(
        root / "sub-001/func/sub-001_task-nback_bold.json"
    )["TaskName"] == "something else entirely"


def test_a_sidecar_with_no_task_name_is_not_given_one(dataset: Path) -> None:
    """Adding a field the file never had is not following a rename, it is
    inventing metadata."""
    rn.apply_rename(dataset, rn.plan_rename(dataset, "task", "rest", "nback"))
    anat = dataset / "sub-001/ses-01/anat/sub-001_ses-01_T1w.json"
    assert "TaskName" not in _json(anat)


# ---------------------------------------------------------------------------
# The other path-valued fields


def test_associated_empty_room_follows_a_session_rename(
    dataset: Path,
) -> None:
    """An MEG sidecar points at the empty-room recording by BIDS URI. Renaming
    the session moved the file and left the pointer behind."""
    rn.apply_rename(dataset, rn.plan_rename(dataset, "ses", "01", "pre"))

    meg = dataset / "sub-001/ses-pre/meg/sub-001_ses-pre_task-rest_meg.json"
    assert _json(meg)["AssociatedEmptyRoom"] == (
        "bids::sub-001/ses-pre/meg/sub-001_ses-pre_task-noise_meg.fif"
    )


def test_sources_follows_a_task_rename(dataset: Path) -> None:
    """``Sources`` is a list of BIDS URIs and behaves exactly like
    ``IntendedFor``; it was simply never in the list."""
    rn.apply_rename(dataset, rn.plan_rename(dataset, "task", "rest", "nback"))

    anat = dataset / "sub-001/ses-01/anat/sub-001_ses-01_T1w.json"
    assert _json(anat)["Sources"] == [
        "bids::sub-001/ses-01/func/sub-001_ses-01_task-nback_bold.nii.gz"
    ]


def test_a_single_string_is_written_back_as_a_string(tmp_path: Path) -> None:
    """BIDS allows one URI or a list of them. Writing a string back as a
    one-element list would change the file's shape for no reason."""
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "sub-001/meg/sub-001_task-noise_meg.fif")
    _write(root, "sub-001/meg/sub-001_task-rest_meg.fif")
    _write(root, "sub-001/meg/sub-001_task-rest_meg.json", json.dumps({
        "AssociatedEmptyRoom": "bids::sub-001/meg/sub-001_task-noise_meg.fif",
    }))

    rn.apply_rename(root, rn.plan_rename(root, "sub", "001", "002"))
    written = _json(root / "sub-002/meg/sub-002_task-rest_meg.json")
    assert isinstance(written["AssociatedEmptyRoom"], str)
    assert written["AssociatedEmptyRoom"] == (
        "bids::sub-002/meg/sub-002_task-noise_meg.fif"
    )


@pytest.mark.parametrize("field", ["RawSources", "AnatomicalImage"])
def test_the_remaining_path_fields_follow_too(
    tmp_path: Path, field: str,
) -> None:
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "sub-001/anat/sub-001_T1w.nii.gz")
    _write(root, "sub-001/anat/sub-001_T1w.json", json.dumps({
        field: ["bids::sub-001/anat/sub-001_T1w.nii.gz"],
    }))

    rn.apply_rename(root, rn.plan_rename(root, "sub", "001", "002"))
    assert _json(root / "sub-002/anat/sub-002_T1w.json")[field] == [
        "bids::sub-002/anat/sub-002_T1w.nii.gz"
    ]


# ---------------------------------------------------------------------------
# The tables that LIST an entity


def test_renaming_a_session_updates_sessions_tsv(dataset: Path) -> None:
    """``sub-XX_sessions.tsv`` is ``participants.tsv`` one level down, and it
    was not being followed.

    The folder became ``ses-pre`` and the table still listed ``ses-01``, a
    session that no longer existed anywhere, with no validator error. That is
    the same dangling reference the subject case was fixed for.
    """
    rn.apply_rename(dataset, rn.plan_rename(dataset, "ses", "01", "pre"))
    assert _column(
        dataset / "sub-001/sub-001_sessions.tsv", "session_id",
    ) == ["ses-pre"]


def test_the_other_columns_of_sessions_tsv_survive(dataset: Path) -> None:
    rn.apply_rename(dataset, rn.plan_rename(dataset, "ses", "01", "pre"))
    assert _column(
        dataset / "sub-001/sub-001_sessions.tsv", "acq_date",
    ) == ["2026-01-01"]


def test_renaming_a_sample_updates_samples_tsv(tmp_path: Path) -> None:
    """The microscopy equivalent, for the same reason."""
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "samples.tsv",
           "sample_id\tparticipant_id\tsample_type\n"
           "sample-01\tsub-001\ttissue\n")
    _write(root, "sub-001/micr/sub-001_sample-01_TEM.ome.tif")

    rn.apply_rename(root, rn.plan_rename(root, "sample", "01", "A1"))
    assert _column(root / "samples.tsv", "sample_id") == ["sample-A1"]


def test_a_partial_session_rename_leaves_the_row(dataset: Path) -> None:
    """Half a session is still the old session.

    Renaming some of a session's files makes two sessions, which is a split
    rather than a rename, so the row follows only when the whole thing moved.
    This mirrors what ``participant_id`` already did.
    """
    _write(dataset, "sub-001/ses-01/dwi/sub-001_ses-01_dwi.nii.gz")
    plan = rn.plan_rename(dataset, "ses", "01", "pre")
    some = {
        plan.file_key(dataset, src) for src, _dst in plan.file_moves
        if "/anat/" in src.as_posix()
    }
    rn.apply_rename(dataset, plan, only=some)

    assert _column(
        dataset / "sub-001/sub-001_sessions.tsv", "session_id",
    ) == ["ses-01"], "ses-01 still exists, so its row must stay"


# ---------------------------------------------------------------------------
# Undo, and the guarantee that none of this is one-way


def test_all_of_it_undoes_as_one_step(dataset: Path) -> None:
    before = _names(dataset)
    sidecar = (dataset / "sub-001/ses-01/func"
               / "sub-001_ses-01_task-rest_bold.json").read_text()
    sessions = (dataset / "sub-001/sub-001_sessions.tsv").read_text()

    rn.apply_rename(dataset, rn.plan_rename(dataset, "task", "rest", "nback"))
    undo_last(dataset)

    assert _names(dataset) == before
    assert (dataset / "sub-001/ses-01/func"
            / "sub-001_ses-01_task-rest_bold.json").read_text() == sidecar
    assert (dataset / "sub-001/sub-001_sessions.tsv").read_text() == sessions


def test_the_preview_lists_every_edit_it_will_make(dataset: Path) -> None:
    """The dialog shows ``content_edits`` under "Follows automatically", so an
    edit the plan does not report is one the user never sees coming."""
    plan = rn.plan_rename(dataset, "ses", "01", "pre")
    what = {e.what for e in plan.content_edits}
    assert "session_id" in what
    assert "AssociatedEmptyRoom" in what
    assert "Sources" in what


# ---------------------------------------------------------------------------
# The BIDS URI scheme, which hid a subject rename from every path field


def test_a_bids_uri_keeps_its_directory_in_step(tmp_path: Path) -> None:
    """A pre-existing defect this work uncovered, and a silent one.

    ``_swap_token`` split on ``[/_]``, which makes the first chunk of
    ``bids::sub-001/anat/sub-001_T1w.nii.gz`` the string ``bids::sub-001``.
    That never equals ``sub-001``, so a SUBJECT rename rewrote the basename
    and left the directory naming the old subject:

        bids::sub-001/anat/sub-002_T1w.nii.gz

    The URI pointed at nothing and no validator reported it. Only subject
    renames could hit it, because ``sub-`` is the only entity in the first
    path segment, which is why it survived for so long: every other entity
    sits where the split handles it correctly.
    """
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "participants.tsv", "participant_id\nsub-001\n")
    _write(root, "sub-001/func/sub-001_task-rest_bold.nii.gz")
    _write(root, "sub-001/fmap/sub-001_epi.nii.gz")
    _write(root, "sub-001/fmap/sub-001_epi.json", json.dumps({
        "IntendedFor": ["bids::sub-001/func/sub-001_task-rest_bold.nii.gz"],
    }))

    rn.apply_rename(root, rn.plan_rename(root, "sub", "001", "002"))

    assert _json(root / "sub-002/fmap/sub-002_epi.json")["IntendedFor"] == [
        "bids::sub-002/func/sub-002_task-rest_bold.nii.gz"
    ]


def test_a_named_dataset_uri_is_understood(tmp_path: Path) -> None:
    """BIDS URIs are ``bids:[dataset-name]:path``, not only ``bids::``.

    A pipeline writing ``bids:rawdata:sub-001/...`` is spelling the same
    reference, and the scheme has to come off whatever sits between the
    colons or the first segment is mangled the same way.
    """
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "sub-001/anat/sub-001_T1w.nii.gz")
    _write(root, "sub-001/anat/sub-001_T1w.json", json.dumps({
        "Sources": ["bids:rawdata:sub-001/anat/sub-001_T1w.nii.gz"],
    }))

    rn.apply_rename(root, rn.plan_rename(root, "sub", "001", "002"))
    assert _json(root / "sub-002/anat/sub-002_T1w.json")["Sources"] == [
        "bids:rawdata:sub-002/anat/sub-002_T1w.nii.gz"
    ]
