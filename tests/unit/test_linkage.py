"""The six fields that point from one file to another.

BIDS Manager wrote one of them automatically and left the other five to be
typed into a JSON editor by hand, which is why a pointer at a file somebody
deleted was invisible: the validator reports that for ``IntendedFor`` and
for none of the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import linkage


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """A subject with two fieldmaps and three runs, timed like the real one."""
    root = tmp_path / "ds"
    (root / "sub-001/fmap").mkdir(parents=True)
    (root / "sub-001/func").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.10.0"})
    )

    def write(rel: str, meta: dict) -> None:
        path = root / rel
        path.write_bytes(b"x")
        path.with_name(path.name.split(".")[0] + ".json").write_text(
            json.dumps(meta)
        )

    for run, time in ((1, "09:33:59"), (2, "09:57:39"), (3, "10:17:1")):
        write(f"sub-001/func/sub-001_task-x_run-{run}_bold.nii.gz",
              {"AcquisitionTime": time})
    for fm, time in ((1, "09:31:31"), (2, "09:54:53")):
        for suffix in ("magnitude1", "phasediff"):
            write(f"sub-001/fmap/sub-001_run-{fm}_{suffix}.nii.gz",
                  {"AcquisitionTime": time})
    return root


def _fmap(root: Path, n: int) -> Path:
    return root / f"sub-001/fmap/sub-001_run-{n}_phasediff.json"


class TestWhichFieldsAreOffered:
    def test_a_fieldmap_is_offered_intended_for(self, dataset):
        assert "IntendedFor" in linkage.fields_for(dataset, _fmap(dataset, 1))

    def test_a_bold_is_not(self, dataset):
        """IntendedFor is written ON the fieldmap, not on what it corrects."""
        bold = dataset / "sub-001/func/sub-001_task-x_run-1_bold.json"
        assert "IntendedFor" not in linkage.fields_for(dataset, bold)

    def test_sources_is_offered_everywhere(self, dataset):
        bold = dataset / "sub-001/func/sub-001_task-x_run-1_bold.json"
        assert "Sources" in linkage.fields_for(dataset, bold)


class TestCandidates:
    def test_only_what_the_field_may_point_at(self, dataset):
        got = linkage.candidates(dataset, _fmap(dataset, 1), "IntendedFor")
        assert got, "runs exist"
        assert all("/func/" in p.as_posix() for p in got)

    def test_never_the_file_itself(self, dataset):
        fmap = _fmap(dataset, 1)
        assert fmap not in linkage.candidates(dataset, fmap, "IntendedFor")

    def test_never_another_subject(self, dataset):
        other = dataset / "sub-002/func"
        other.mkdir(parents=True)
        (other / "sub-002_task-x_run-1_bold.nii.gz").write_bytes(b"x")
        got = linkage.candidates(dataset, _fmap(dataset, 1), "IntendedFor")
        assert not any("sub-002" in p.as_posix() for p in got)


class TestProposal:
    def test_it_is_the_conversion_rule(self, dataset):
        """Fieldmap 1 covers run 1; fieldmap 2 covers runs 2 and 3."""
        first = linkage.propose(dataset, _fmap(dataset, 1), "IntendedFor")
        second = linkage.propose(dataset, _fmap(dataset, 2), "IntendedFor")
        assert [p.name for p in first.targets] == [
            "sub-001_task-x_run-1_bold.nii.gz"]
        assert [p.name for p in second.targets] == [
            "sub-001_task-x_run-2_bold.nii.gz",
            "sub-001_task-x_run-3_bold.nii.gz"]

    def test_the_reason_is_given(self, dataset):
        proposal = linkage.propose(dataset, _fmap(dataset, 1), "IntendedFor")
        assert "acquisition times" in proposal.reason

    def test_no_rule_proposes_the_other_fields(self, dataset):
        assert linkage.propose(dataset, _fmap(dataset, 1), "Sources") is None


class TestWriting:
    def test_a_link_is_written_as_a_uri(self, dataset):
        fmap = _fmap(dataset, 1)
        target = dataset / "sub-001/func/sub-001_task-x_run-2_bold.nii.gz"
        sidecar, uris = linkage.plan_write(dataset, fmap, "IntendedFor", [target])
        assert uris == ["bids::sub-001/func/sub-001_task-x_run-2_bold.nii.gz"]
        assert linkage.apply_links(dataset, [(sidecar, "IntendedFor", uris)]) == 1
        assert json.loads(fmap.read_text())["IntendedFor"] == uris

    def test_an_empty_list_removes_the_key(self, dataset):
        """``[]`` claims the file points at nothing, which is a different and
        wronger statement than not saying."""
        fmap = _fmap(dataset, 1)
        linkage.apply_links(dataset, [(fmap, "IntendedFor", ["bids::x"])])
        linkage.apply_links(dataset, [(fmap, "IntendedFor", [])])
        assert "IntendedFor" not in json.loads(fmap.read_text())

    def test_writing_the_same_value_changes_nothing(self, dataset):
        fmap = _fmap(dataset, 1)
        uris = ["bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"]
        assert linkage.apply_links(dataset, [(fmap, "IntendedFor", uris)]) == 1
        assert linkage.apply_links(dataset, [(fmap, "IntendedFor", uris)]) == 0

    def test_a_batch_is_one_operation(self, dataset):
        """Forty fieldmaps is one entry in the history and one undo."""
        from bidsmgr.project.operations import read_log

        edits = [
            (_fmap(dataset, n), "IntendedFor",
             ["bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"])
            for n in (1, 2)
        ]
        assert linkage.apply_links(dataset, edits) == 2
        assert len(read_log(dataset)) == 1


class TestResolving:
    def test_a_bids_uri(self, dataset):
        got = linkage.resolve(
            dataset, "bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz")
        assert got is not None and got.name.endswith("run-1_bold.nii.gz")

    def test_the_legacy_subject_relative_form(self, dataset):
        got = linkage.resolve(dataset, "func/sub-001_task-x_run-1_bold.nii.gz")
        assert got is not None

    def test_a_missing_file_resolves_to_nothing(self, dataset):
        assert linkage.resolve(dataset, "bids::sub-001/func/gone.nii.gz") is None


class TestBrokenLinks:
    def test_a_pointer_at_a_deleted_file_is_found(self, dataset):
        fmap = _fmap(dataset, 1)
        linkage.apply_links(dataset, [(
            fmap, "IntendedFor",
            ["bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"],
        )])
        (dataset / "sub-001/func/sub-001_task-x_run-1_bold.nii.gz").unlink()

        broken = linkage.broken_links(dataset)
        assert len(broken) == 1
        assert broken[0].field == "IntendedFor"
        assert broken[0].source == fmap

    def test_the_other_five_fields_are_checked_too(self, dataset):
        """The whole reason this exists: the validator checks IntendedFor and
        says nothing about the rest."""
        bold = dataset / "sub-001/func/sub-001_task-x_run-1_bold.json"
        linkage.apply_links(dataset, [(bold, "Sources", ["bids::nope.nii.gz"])])
        assert [b.field for b in linkage.broken_links(dataset)] == ["Sources"]

    def test_a_healthy_dataset_has_none(self, dataset):
        assert linkage.broken_links(dataset) == []
