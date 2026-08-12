"""Every answer the metadata template accepts must reach a file.

The template asks its questions from the schema, but what it can STORE and what
gets WRITTEN were each decided elsewhere, so a field could be offered, typed
into, saved, and quietly dropped. Two ways that happened, both real:

* ``DatasetDescriptionSpec`` named nine fields because the CLI has nine flags.
  The schema declares twelve, the form offered twelve, and HEDVersion,
  DatasetLinks and Keywords went in and never came out.
* The agnostic answers reached ``dataset_description.json`` only through a
  separate ``bidsmgr-metadata`` run. Convert alone wrote none of them, so a user
  who filled in the form and pressed convert saw an empty file.

These compare the two lists rather than checking a handful of names, so a field
added to a future schema is covered without anyone remembering to add a case.
"""

from __future__ import annotations

import json
from pathlib import Path

from bidsmgr.cli._scaffold import ensure_dataset_description
from bidsmgr.metadata.template_plan import (
    build_template_tree,
    dataset_description_section,
    sidecar_section,
)
from bidsmgr.recording_meta import (
    DatasetDescriptionSpec,
    dataset_description_as_bids,
    dataset_description_from_bids,
)
from bidsmgr import schema as schema_mod


def _sample(field):
    """A value of the shape the schema declares for this field."""
    if field.enum:
        return field.enum[0]
    if field.type == "array":
        return [0] if field.item_type in ("number", "integer") else ["x"]
    if field.type in ("number", "integer"):
        return 1
    if field.type == "boolean":
        return True
    if field.type == "object":
        return {"k": "v"}
    return "stated"


def test_every_agnostic_answer_survives_a_save() -> None:
    offered = {f.name: _sample(f) for f in dataset_description_section().fields}
    spec = DatasetDescriptionSpec()
    dataset_description_from_bids(spec, offered)
    stored = dataset_description_as_bids(spec)
    assert set(stored) == set(offered), (
        f"discarded on save: {sorted(set(offered) - set(stored))}"
    )


def test_every_agnostic_answer_reaches_the_file_at_conversion(tmp_path: Path) -> None:
    """Convert alone, with no metadata run afterwards."""
    offered = {f.name: _sample(f) for f in dataset_description_section().fields}
    spec = DatasetDescriptionSpec()
    dataset_description_from_bids(spec, offered)

    root = tmp_path / "ds"
    root.mkdir()
    ensure_dataset_description(root, fields=dataset_description_as_bids(spec))

    written = json.loads((root / "dataset_description.json").read_text())
    missing = [name for name in offered if name not in written]
    assert not missing, f"never written: {missing}"


def test_a_stated_name_beats_the_folder_name(tmp_path: Path) -> None:
    """The form is the only place to say what a dataset is called, so saying it
    there has to win over the default taken from the directory."""
    root = tmp_path / "ds"
    root.mkdir()
    ensure_dataset_description(root)
    assert json.loads((root / "dataset_description.json").read_text())["Name"] == "ds"

    ensure_dataset_description(root, fields={"Name": "The Real Study"})
    assert (
        json.loads((root / "dataset_description.json").read_text())["Name"]
        == "The Real Study"
    )


def test_the_chain_accepts_every_field_the_form_offers() -> None:
    """The form asks bidsval what a file may carry; the chain asks it again per
    field before writing. If those two disagreed, an answer would be accepted by
    the form and silently refused by the writer."""
    pairs = [
        ("anat", "T1w"), ("func", "bold"), ("dwi", "dwi"), ("fmap", "epi"),
        ("eeg", "eeg"), ("meg", "meg"), ("ieeg", "ieeg"), ("pet", "pet"),
    ]
    refused: dict[str, list[str]] = {}
    for datatype, suffix in pairs:
        section = sidecar_section(datatype, suffix, include_derived=True)
        bad = [
            f.name for f in section.fields
            if not schema_mod.field_applies(f.name, datatype, suffix)
        ]
        if bad:
            refused[f"{datatype}/{suffix}"] = bad
    assert not refused, f"offered but unwritable: {refused}"


def test_every_file_node_stores_under_the_key_the_writer_reads() -> None:
    """A section's answers go to ``sequence_templates[storage_key]`` and the
    writer looks them up as ``<datatype>/<suffix>``. One typo apart and every
    per-file answer disappears."""
    from bidsmgr.recording_meta.templates import template_key

    tree = build_template_tree([("eeg", "eeg"), ("anat", "T1w"), ("pet", "pet")])
    for root in tree:
        for node in root.walk():
            if not node.section or node.section.datatype == "":
                continue
            assert node.key == template_key(
                node.section.datatype, node.section.suffix
            )


def test_a_correction_to_what_the_converter_wrote_survives(tmp_path: Path) -> None:
    """The whole reason "already answered by the conversion" is editable.

    A user who corrects the manufacturer there has said the header is wrong.
    Their answer used to be skipped, on the rule that a file's own header beats
    a statement about a class of files, so the correction vanished with no
    explanation on the next run.
    """
    import pandas as pd
    from bidsmgr.fixups.sidecar_schema import apply_stated_metadata
    from bidsmgr.recording_meta import RecordingMetaSpec

    root = tmp_path / "ds"
    eeg = root / "sub-001" / "eeg"
    eeg.mkdir(parents=True)
    (eeg / "sub-001_task-rest_eeg.json").write_text(json.dumps({
        "Manufacturer": "read from the header",
        "SamplingFrequency": 500.0,
    }))

    spec = RecordingMetaSpec()
    spec.sequence_templates["eeg/eeg"] = {"Manufacturer": "what the user says"}
    apply_stated_metadata(root, spec, pd.DataFrame([{
        "proposed_basename": "sub-001_task-rest_eeg", "source_file": "/raw/a.edf",
    }]))

    got = json.loads((eeg / "sub-001_task-rest_eeg.json").read_text())
    assert got["Manufacturer"] == "what the user says"
    # What nobody stated is left exactly as the converter wrote it.
    assert got["SamplingFrequency"] == 500.0
