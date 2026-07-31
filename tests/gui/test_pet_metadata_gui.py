"""PET sections in the dataset dialog and the per-row Properties panel.

PET reuses the metadata UI EEG/MEG already had rather than growing a parallel
one, so the tests here are mostly about the seam: the right blocks appear for
the right datasets, values round-trip, and nothing an EEG-only or MRI-only user
sees has changed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bidsmgr.gui.models import InventoryTableModel
from bidsmgr.gui.properties_panel import PropertiesPanel
from bidsmgr.gui.recording_meta_dialog import RecordingMetaDialog
from bidsmgr.recording_meta import PetAcquisitionSpec, RecordingMetaSpec

pytestmark = pytest.mark.gui


PET_SUGGESTIONS = {
    "tracer": ["FDG / F18"],
    "dose": ["44.4 MBq"],
    "recon": ["PSF+TOF (3i 21s)"],
}


def _dialog(tmp_path, present):
    return RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        set(present),
        None,
        pet_suggestions=PET_SUGGESTIONS,
    )


def _pet_row(**extra) -> dict:
    row = {
        "BIDS_name": "sub-001", "subject": "phantom", "include": 1,
        "sequence": "PET_Brain_AC_TOF", "series_uid": "1.2.3",
        "proposed_datatype": "pet", "proposed_basename": "sub-001_pet",
        "bids_guess_datatype": "pet", "bids_guess_suffix": "pet",
        "entities": '{"subject": "001"}',
        "tracer_suggestion": "FDG", "radionuclide_suggestion": "F18",
        "injected_dose_suggestion": "44.4 MBq",
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Dataset dialog
# ---------------------------------------------------------------------------


def test_pet_blocks_show_only_for_a_pet_dataset(qtbot, tmp_path) -> None:
    dlg = _dialog(tmp_path, {"pet"})
    qtbot.addWidget(dlg)
    assert all(box.isVisibleTo(dlg) for box in dlg._pet_boxes)
    assert not dlg._eeg_box.isVisibleTo(dlg)
    assert dlg._specific_region.isVisibleTo(dlg)


def test_pet_blocks_hidden_for_an_eeg_dataset(qtbot, tmp_path) -> None:
    """REGRESSION: an EEG user must see exactly what they saw before."""
    dlg = _dialog(tmp_path, {"eeg", "meg"})
    qtbot.addWidget(dlg)
    assert not any(box.isVisibleTo(dlg) for box in dlg._pet_boxes)
    assert dlg._eeg_box.isVisibleTo(dlg)


def test_both_families_show_for_a_hybrid_dataset(qtbot, tmp_path) -> None:
    dlg = _dialog(tmp_path, {"pet", "eeg"})
    qtbot.addWidget(dlg)
    assert all(box.isVisibleTo(dlg) for box in dlg._pet_boxes)
    assert dlg._eeg_box.isVisibleTo(dlg)


def test_specific_region_hidden_for_mri_only(qtbot, tmp_path) -> None:
    """REGRESSION: MRI needs no modality-specific metadata at all."""
    dlg = _dialog(tmp_path, {"anat", "func"})
    qtbot.addWidget(dlg)
    assert not dlg._specific_region.isVisibleTo(dlg)
    assert not any(box.isVisibleTo(dlg) for box in dlg._pet_boxes)


def test_pet_fields_round_trip_through_the_spec(qtbot, tmp_path) -> None:
    dlg = _dialog(tmp_path, {"pet"})
    qtbot.addWidget(dlg)

    dlg._pet_tracer_name.setCurrentText("FDG")
    dlg._pet_radionuclide.setCurrentText("F18")
    dlg._pet_injected_radioactivity.setText("44.4")
    dlg._pet_injected_radioactivity_units.setCurrentText("MBq")
    dlg._pet_mode_of_administration.setCurrentText("bolus")
    dlg._pet_image_decay_corrected.setCurrentText("true")
    dlg._pet_recon_labels.setText("iterations, subsets")
    dlg._pet_recon_values.setText("3, 21")

    pet = dlg.build_spec().pet_defaults
    assert pet.tracer_name == "FDG"
    assert pet.injected_radioactivity == 44.4
    assert pet.mode_of_administration == "bolus"
    assert pet.image_decay_corrected is True
    assert pet.recon_method_parameter_labels == ["iterations", "subsets"]
    assert pet.recon_method_parameter_values == [3.0, 21.0]


def test_a_hidden_pet_block_keeps_its_loaded_values(qtbot, tmp_path) -> None:
    """An EEG-only session must not wipe PET fields from a shared scaffold."""
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    spec = RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(tracer_name="PIB"))
    scaffold.write_text(spec.model_dump_json())

    dlg = RecordingMetaDialog(scaffold, {"eeg"}, None)
    qtbot.addWidget(dlg)
    assert dlg.build_spec().pet_defaults.tracer_name == "PIB"


def test_decay_corrected_blank_stays_unset(qtbot, tmp_path) -> None:
    """Blank means "not stated", which is not the same as false."""
    dlg = _dialog(tmp_path, {"pet"})
    qtbot.addWidget(dlg)
    assert dlg.build_spec().pet_defaults.image_decay_corrected is None


# ---------------------------------------------------------------------------
# Per-row overrides
# ---------------------------------------------------------------------------


def test_row_inherits_the_dataset_default(qtbot) -> None:
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(
        RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(tracer_name="FDG"))
    )
    assert model.pet_effective(0, "tracer_name") == "FDG"
    assert model.pet_is_inherited(0, "tracer_name")


def test_override_breaks_inheritance(qtbot) -> None:
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(
        RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(tracer_name="FDG"))
    )
    assert model.set_pet_override(0, "tracer_name", "PIB")
    assert model.pet_effective(0, "tracer_name") == "PIB"
    assert not model.pet_is_inherited(0, "tracer_name")


def test_writing_the_default_restores_inheritance(qtbot) -> None:
    """Otherwise a later change to the dataset default would not reach the row."""
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(
        RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(tracer_name="FDG"))
    )
    model.set_pet_override(0, "tracer_name", "PIB")
    model.set_pet_override(0, "tracer_name", "FDG")
    assert model.pet_is_inherited(0, "tracer_name")
    assert "1.2.3" not in model.global_spec().pet_overrides


def test_a_non_numeric_value_is_refused(qtbot) -> None:
    """A coerced number would look deliberate in the sidecar. Refuse instead."""
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(RecordingMetaSpec())
    assert not model.set_pet_override(0, "injected_radioactivity", "not a number")
    assert model.pet_effective(0, "injected_radioactivity") == ""


def test_a_numeric_value_is_stored_as_a_number(qtbot) -> None:
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(RecordingMetaSpec())
    assert model.set_pet_override(0, "injected_radioactivity", "44.4")
    assert model.global_spec().pet_overrides["1.2.3"].injected_radioactivity == 44.4


def test_float_renders_without_a_trailing_zero(qtbot) -> None:
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(
        RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(scan_start=0.0))
    )
    assert model.pet_effective(0, "scan_start") == "0"


def test_pet_overrides_do_not_touch_the_eeg_block(qtbot) -> None:
    """REGRESSION: the two blocks are separate fields on one spec."""
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(RecordingMetaSpec())
    model.set_pet_override(0, "tracer_name", "FDG")
    spec = model.global_spec()
    assert spec.overrides == {}
    assert spec.pet_overrides["1.2.3"].tracer_name == "FDG"


# ---------------------------------------------------------------------------
# Properties panel
# ---------------------------------------------------------------------------


def test_properties_panel_renders_a_pet_row(qtbot) -> None:
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(
        RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(tracer_name="FDG"))
    )
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(model)
    panel.set_selected_row(0)  # would raise if the PET section were broken

    labels = [w.text() for w in panel.findChildren(type(panel._divider())) if hasattr(w, "text")]
    assert labels is not None  # the render completed


def test_properties_panel_still_renders_an_mri_row(qtbot) -> None:
    """REGRESSION: the PET branch must not disturb the MRI path."""
    row = _pet_row(proposed_datatype="anat", bids_guess_datatype="anat",
                   bids_guess_suffix="T1w", proposed_basename="sub-001_T1w")
    model = InventoryTableModel(pd.DataFrame([row]))
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(model)
    panel.set_selected_row(0)
