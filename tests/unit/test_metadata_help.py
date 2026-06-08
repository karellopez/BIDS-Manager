"""The recording-metadata tooltips are sourced from the live BIDS schema,
which also confirms every exposed field is a genuine schema key."""

from __future__ import annotations

import pytest

from bidsmgr.gui.metadata_help import bids_tooltip, tooltip_for


@pytest.mark.parametrize(
    "ui_key, must_contain",
    [
        ("manufacturer", "Manufacturer"),
        ("line_freq", "Power"),               # PowerLineFrequency
        ("eeg_reference", "reference"),
        ("eeg_ground", "ground"),
        ("cap_manufacturer", "cap"),
        ("dewar_position", "dewar"),
        ("associated_empty_room", "empty"),
        ("subject_artefact_description", "artifact"),
    ],
)
def test_tooltip_from_schema(ui_key, must_contain):
    tip = tooltip_for(ui_key)
    assert tip, f"no tooltip for {ui_key}"
    assert must_contain.lower() in tip.lower()


def test_tooltip_markdown_stripped():
    # CapManufacturer's schema description has a backticked example.
    tip = bids_tooltip("CapManufacturer")
    assert "`" not in tip and tip


def test_fallback_for_non_metadata_fields():
    # montage is a convention (electrodes.tsv), demographics are participant cols.
    assert "electrodes.tsv" in tooltip_for("montage")
    assert "participants.tsv" in tooltip_for("PatientSex")
    assert "participants.tsv" in tooltip_for("Handedness")


def test_unknown_field_returns_empty():
    assert tooltip_for("definitely_not_a_field") == ""
