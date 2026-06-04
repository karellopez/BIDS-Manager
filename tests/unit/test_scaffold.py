"""Tests for the shared BIDS scaffold helpers (``bidsmgr.cli._scaffold``).

These centralize dataset_description.json creation + ``.bidsignore`` so the
convert verb and the future project "create dataset" flow stay consistent: the
BIDS version is sourced from the bundled schema, ``GeneratedBy`` is idempotent,
and BM's hidden working dirs are exempted from the official validator.
"""

from __future__ import annotations

import json
from pathlib import Path

from bidsmgr.cli._scaffold import (
    BIDSIGNORE_LINES,
    ensure_bidsignore,
    ensure_dataset_description,
)
from bidsmgr.schema import bids_version as schema_bids_version


def _dd(root: Path) -> dict:
    return json.loads((root / "dataset_description.json").read_text())


def test_dataset_description_minimal_valid(tmp_path: Path):
    ensure_dataset_description(tmp_path)
    dd = _dd(tmp_path)
    assert dd["Name"] == tmp_path.name
    assert dd["BIDSVersion"] == schema_bids_version()  # no hardcoded version
    assert dd["DatasetType"] == "raw"
    # No generator entry when none is passed (the "create empty dataset" path).
    assert "GeneratedBy" not in dd


def test_dataset_description_records_generated_by_once(tmp_path: Path):
    stamp = {"Name": "bidsmgr", "Version": "9.9.9", "Description": "x backend"}
    ensure_dataset_description(tmp_path, generated_by=stamp)
    ensure_dataset_description(tmp_path, generated_by=stamp)  # re-run
    gb = _dd(tmp_path)["GeneratedBy"]
    assert [g for g in gb if g.get("Name") == "bidsmgr"] == [stamp]


def test_dataset_description_preserves_user_fields(tmp_path: Path):
    p = tmp_path / "dataset_description.json"
    p.write_text(json.dumps({"Name": "My Study", "Authors": ["A. Person"]}))
    ensure_dataset_description(tmp_path)
    dd = _dd(tmp_path)
    assert dd["Name"] == "My Study"          # not overwritten
    assert dd["Authors"] == ["A. Person"]    # untouched
    assert dd["BIDSVersion"] == schema_bids_version()  # filled


def test_bidsignore_created_with_bm_dirs(tmp_path: Path):
    ensure_bidsignore(tmp_path)
    lines = (tmp_path / ".bidsignore").read_text().splitlines()
    for entry in BIDSIGNORE_LINES:
        assert entry in lines


def test_bidsignore_is_idempotent_and_preserves_user_lines(tmp_path: Path):
    p = tmp_path / ".bidsignore"
    p.write_text("derivatives/\n.bidsmgr/\n")  # user already ignored one BM dir
    ensure_bidsignore(tmp_path)
    ensure_bidsignore(tmp_path)  # second call must not duplicate
    lines = p.read_text().splitlines()
    assert lines.count(".bidsmgr/") == 1
    assert ".tmp_bidsmgr/" in lines
    assert "derivatives/" in lines  # user line preserved
