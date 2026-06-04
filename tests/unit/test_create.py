"""Tests for ``bidsmgr-create`` (create/adopt a BIDS dataset workspace).

The verb scaffolds a minimal valid BIDS dataset (dataset_description.json +
README + .bidsignore) and an event-sourced project bundle at
``<bids_root>/.bidsmgr/project``. It adopts a pre-existing dataset without
overwriting it and is idempotent on re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

from bidsmgr.cli._scaffold import BIDSIGNORE_LINES, project_bundle_dir
from bidsmgr.cli.create import run_create
from bidsmgr.editor.validator import validate
from bidsmgr.project import Project
from bidsmgr.schema import bids_version as schema_bids_version


def test_create_scaffolds_a_valid_subjectless_dataset(tmp_path: Path):
    root = tmp_path / "my_study"
    rc = run_create(root)
    assert rc == 0

    dd = json.loads((root / "dataset_description.json").read_text())
    assert dd["Name"] == "my_study"
    assert dd["BIDSVersion"] == schema_bids_version()
    assert (root / "README").exists()
    bidsignore = (root / ".bidsignore").read_text().splitlines()
    assert all(line in bidsignore for line in BIDSIGNORE_LINES)

    # A scaffolded-but-subjectless dataset is valid BIDS: no errors.
    report = validate(root)
    assert report.counts["err"] == 0


def test_create_initializes_project_bundle(tmp_path: Path):
    root = tmp_path / "study"
    run_create(root, name="My Study", description="pilot")

    bundle = project_bundle_dir(root)
    assert (bundle / "events.jsonl").exists()
    assert (bundle / "meta.json").exists()
    # The bundle opens and replays (one ProjectCreated event).
    proj = Project.open(bundle)
    assert proj.state() is not None


def test_create_name_sets_dataset_description_name(tmp_path: Path):
    root = tmp_path / "folder_slug"
    run_create(root, name="Human Readable Name")
    dd = json.loads((root / "dataset_description.json").read_text())
    assert dd["Name"] == "Human Readable Name"


def test_create_adopts_existing_dataset_without_clobbering(tmp_path: Path):
    root = tmp_path / "existing"
    root.mkdir()
    # A dataset made by some other tool, with hand-edited fields.
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Pre-existing", "Authors": ["A. Person"]})
    )
    (root / "README").write_text("hand-written readme")

    rc = run_create(root)
    assert rc == 0

    dd = json.loads((root / "dataset_description.json").read_text())
    assert dd["Name"] == "Pre-existing"          # not overwritten
    assert dd["Authors"] == ["A. Person"]        # preserved
    assert dd["BIDSVersion"] == schema_bids_version()  # filled in
    assert (root / "README").read_text() == "hand-written readme"  # preserved
    assert project_bundle_dir(root).exists()     # bundle added


def test_create_is_idempotent(tmp_path: Path):
    root = tmp_path / "twice"
    run_create(root)
    bundle = project_bundle_dir(root)
    n_events_first = (bundle / "events.jsonl").read_text().count("\n")

    rc = run_create(root)  # second run must not error or re-init the bundle
    assert rc == 0
    n_events_second = (bundle / "events.jsonl").read_text().count("\n")
    assert n_events_second == n_events_first  # bundle untouched
