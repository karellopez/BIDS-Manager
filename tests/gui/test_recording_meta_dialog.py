"""GUI smoke tests for the Phase 4 recording-metadata surface:

* the constrained ``montage`` / ``line_freq`` dropdown delegate,
* the new editable per-row columns (reference / ground / demographics),
* the dataset-level :class:`RecordingMetaDialog` scaffold round-trip.

Marked ``gui`` so they run under ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import QComboBox, QTableWidgetItem

from bidsmgr.gui.delegates import ChoiceDelegate, builtin_montages
from bidsmgr.gui.models import COLUMNS
from bidsmgr.gui.recording_meta_dialog import RecordingMetaDialog
from bidsmgr.recording_meta import load_spec

pytestmark = pytest.mark.gui


def _spec(key: str):
    return next(c for c in COLUMNS if c.key == key)


def test_new_columns_registered_and_editable():
    for key in ("eeg_reference", "eeg_ground", "Handedness", "montage", "line_freq"):
        assert _spec(key).editable is True
    # Demographics are now editable so EEG/MEG subjects can be filled in-table.
    assert _spec("PatientSex").editable is True
    assert _spec("PatientAge").editable is True


def test_builtin_montages_nonempty():
    assert len(builtin_montages()) > 0


def test_choice_delegate_is_dropdown_only(qtbot):
    d = ChoiceDelegate(["50", "60"], blank_label="(blank)")
    editor = d.createEditor(None, None, None)
    assert isinstance(editor, QComboBox)
    assert editor.isEditable() is False  # never hand-typed
    assert [editor.itemText(i) for i in range(editor.count())] == ["(blank)", "50", "60"]


def test_choice_delegate_blank_maps_to_empty(qtbot):
    d = ChoiceDelegate(["50", "60"], blank_label="(blank)")
    model = QStandardItemModel(1, 1)
    idx = model.index(0, 0)
    model.setData(idx, "60", Qt.ItemDataRole.EditRole)

    editor = QComboBox()
    editor.addItems(["(blank)", "50", "60"])
    editor.setCurrentText("(blank)")
    d.setModelData(editor, model, idx)
    assert model.data(idx, Qt.ItemDataRole.EditRole) == ""  # blank -> empty string

    editor.setCurrentText("50")
    d.setModelData(editor, model, idx)
    assert model.data(idx, Qt.ItemDataRole.EditRole) == "50"


def test_dialog_round_trip(qtbot, tmp_path):
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold)
    qtbot.addWidget(dlg)

    dlg._manufacturer.setCurrentText("Brain Products")
    dlg._eeg_reference.setText("Cz")
    dlg._line_freq.setCurrentText("60")
    dlg._montage.setCurrentText("(none)")
    dlg._events.insertRow(0)
    dlg._events.setItem(0, 0, QTableWidgetItem("S 20"))
    dlg._events.setItem(0, 1, QTableWidgetItem("eyes_open"))

    spec = dlg.build_spec()
    assert spec.defaults.manufacturer == "Brain Products"
    assert spec.defaults.eeg_reference == "Cz"
    assert spec.defaults.power_line_freq == 60.0
    assert spec.defaults.montage is None  # "(none)" -> unset
    assert spec.event_maps["*"]["S 20"] == "eyes_open"

    dlg._on_save()
    assert scaffold.exists()
    reloaded = load_spec(scaffold)
    assert reloaded.defaults.manufacturer == "Brain Products"
    assert reloaded.event_maps["*"]["S 20"] == "eyes_open"


def test_dialog_loads_existing_scaffold(qtbot, tmp_path):
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    scaffold.write_text(
        '{"schema_version": 1, "defaults": {"manufacturer": "Elekta"}, '
        '"event_maps": {"*": {"T0": "rest"}}}',
        encoding="utf-8",
    )
    dlg = RecordingMetaDialog(scaffold)
    qtbot.addWidget(dlg)
    assert dlg._manufacturer.currentText() == "Elekta"
    assert dlg._events.rowCount() == 1
    assert dlg._events.item(0, 0).text() == "T0"


def test_dialog_phenotype_round_trip(qtbot, tmp_path):
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold)
    qtbot.addWidget(dlg)
    dlg._phenotype.addItem("/data/edinburgh.tsv")
    dlg._phenotype.addItem("/data/bdi.csv")
    assert dlg.build_spec().phenotype_files == ["/data/edinburgh.tsv", "/data/bdi.csv"]
    dlg._on_save()
    assert load_spec(scaffold).phenotype_files == ["/data/edinburgh.tsv", "/data/bdi.csv"]


def test_dialog_participants_file_round_trip(qtbot, tmp_path):
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold)
    qtbot.addWidget(dlg)
    dlg._participants_file.setText("/data/subjects.csv")
    assert dlg.build_spec().participants_file == "/data/subjects.csv"
    dlg._on_save()
    assert load_spec(scaffold).participants_file == "/data/subjects.csv"


def test_dialog_acquisition_hidden_for_mri_only(qtbot, tmp_path):
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold, present_datatypes={"anat", "func"})
    qtbot.addWidget(dlg)
    # The modality-specific sections are hidden for an MRI-only set...
    assert dlg._device_box.isVisibleTo(dlg) is False
    assert dlg._eeg_box.isVisibleTo(dlg) is False
    # ...but the agnostic event + phenotype sections still exist.
    assert dlg._events is not None and dlg._phenotype is not None


def test_dialog_acquisition_shown_for_eeg(qtbot, tmp_path):
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold, present_datatypes={"eeg"})
    qtbot.addWidget(dlg)
    assert dlg._device_box.isVisibleTo(dlg) is True
    assert dlg._eeg_box.isVisibleTo(dlg) is True
    assert dlg._eeg_reference.isEnabled() is True


def test_manufacturer_is_editable_dropdown_with_defaults(qtbot, tmp_path):
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold)
    qtbot.addWidget(dlg)
    assert dlg._manufacturer.isEditable() is True  # pick a default OR type another
    items = [dlg._manufacturer.itemText(i) for i in range(dlg._manufacturer.count())]
    assert "Brain Products" in items
    assert "MEGIN / Elekta / Neuromag" in items
    # A typed custom value flows through.
    dlg._manufacturer.setCurrentText("Custom Amp Co")
    assert dlg.build_spec().defaults.manufacturer == "Custom Amp Co"


def test_hidden_section_values_preserved_on_save(qtbot, tmp_path):
    """Editing on an MRI-only dataset must not wipe EEG fields a scaffold holds."""
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    scaffold.write_text(
        '{"schema_version": 1, "defaults": {"eeg_reference": "Cz", "montage": "standard_1005"}}',
        encoding="utf-8",
    )
    dlg = RecordingMetaDialog(scaffold, present_datatypes={"anat", "func"})
    qtbot.addWidget(dlg)
    spec = dlg.build_spec()
    assert spec.defaults.eeg_reference == "Cz"          # preserved, not wiped
    assert spec.defaults.montage == "standard_1005"


def test_dialog_whole_is_scrollable(qtbot, tmp_path):
    """Structural invariant: exactly ONE scroll area wraps the whole body, and
    the intro label + button box live OUTSIDE it (so the scroll surface is the
    whole metadata body, not an inner widget)."""
    from PyQt6.QtWidgets import QScrollArea
    dlg = RecordingMetaDialog(tmp_path / "inv.tsv.recording_meta.json")
    qtbot.addWidget(dlg)
    areas = dlg.findChildren(QScrollArea)
    assert len(areas) == 1, "the whole body should be one scroll surface"
    # Inner widgets are bounded so they cannot stretch greedily to fill.
    assert 0 < dlg._events.maximumHeight() <= 160
    assert 0 < dlg._phenotype.maximumHeight() <= 110


@pytest.mark.parametrize(
    "datatypes, expect_scroll",
    [
        ({"eeg"}, True),    # all sections stacked -> body taller than viewport
        ({"meg"}, True),    # device + meg + agnostic sections overflow
        ({"mri"}, True),    # institution + events + phenotype overflow too
    ],
)
def test_dialog_whole_body_scrolls_not_inner_tables(qtbot, tmp_path, datatypes, expect_scroll):
    """The OUTER scroll area moves the whole body; inner tables keep a natural
    bounded height (they do not stretch to fill).

    The user asked for a scrollable window rather than one that resizes to its
    content or whose inner tables expand. Scroll engagement at the fixed default
    size is deterministic per modality combination.
    """
    from PyQt6.QtWidgets import QScrollArea
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json", present_datatypes=datatypes)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    sa = dlg.findChild(QScrollArea)
    overflows = sa.widget().sizeHint().height() > sa.viewport().height()
    assert overflows is expect_scroll
    # Tables keep a bounded natural height regardless of modality.
    assert dlg._events.height() <= dlg._events.maximumHeight()
    assert dlg._phenotype.height() <= dlg._phenotype.maximumHeight()


def test_dialog_uses_shared_manufacturer_vocab(qtbot, tmp_path):
    """The dialog's manufacturer dropdown is the one shared list (no duplicate)."""
    from bidsmgr.recording_meta import COMMON_MANUFACTURERS
    dlg = RecordingMetaDialog(tmp_path / "inv.tsv.recording_meta.json")
    qtbot.addWidget(dlg)
    items = [dlg._manufacturer.itemText(i) for i in range(dlg._manufacturer.count())]
    assert items[0] == ""  # blank first entry
    assert items[1:] == list(COMMON_MANUFACTURERS)


def test_dialog_combined_modality_label(qtbot, tmp_path):
    """A field shared by several present modalities is labelled with all of
    them (e.g. 'EEG and MEG'); the device block names every electrophysiology
    modality, the reference block only EEG/iEEG."""
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold, present_datatypes={"eeg", "meg"})
    qtbot.addWidget(dlg)
    assert "EEG and MEG" in dlg._device_box.title()
    # MEG has no scalp reference/montage -> that block is EEG-only here.
    assert "EEG" in dlg._eeg_box.title() and "MEG" not in dlg._eeg_box.title()


def test_dialog_region_header_hidden_for_mri_only(qtbot, tmp_path):
    """An MRI-only dataset shows no modality-specific region (only agnostic)."""
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold, present_datatypes={"mri"})
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    assert not dlg._specific_region.isVisible()
    assert not dlg._device_box.isVisible()
    assert not dlg._eeg_box.isVisible()
    assert not dlg._meg_box.isVisible()


def test_dialog_meg_group_visibility_and_roundtrip(qtbot, tmp_path):
    """The MEG group shows only for MEG datasets and round-trips its manual
    fields. Channel-derived MEG fields are NOT exposed (mne-bids fills them)."""
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold, present_datatypes={"meg"})
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    assert dlg._meg_box.isVisible()
    assert not dlg._eeg_box.isVisible()      # MEG has no scalp reference/montage
    assert not hasattr(dlg, "_continuous_head_localization")  # auto -> not exposed
    dlg._dewar_position.setCurrentText("supine")
    dlg._associated_empty_room.setText("bids::sub-emptyroom")
    acq = dlg.build_spec().defaults
    assert acq.dewar_position == "supine"
    assert acq.associated_empty_room == "bids::sub-emptyroom"


def test_dialog_cap_is_editable_dropdown(qtbot, tmp_path):
    from PyQt6.QtWidgets import QComboBox
    from bidsmgr.recording_meta import COMMON_CAP_MANUFACTURERS
    dlg = RecordingMetaDialog(tmp_path / "inv.tsv.recording_meta.json",
                              present_datatypes={"eeg"})
    qtbot.addWidget(dlg)
    assert isinstance(dlg._cap_manufacturer, QComboBox)
    assert dlg._cap_manufacturer.isEditable()
    items = [dlg._cap_manufacturer.itemText(i) for i in range(dlg._cap_manufacturer.count())]
    assert "EasyCap" in items
    # A typed custom value flows through.
    dlg._cap_manufacturer.setCurrentText("Custom Cap Co")
    assert dlg.build_spec().defaults.cap_manufacturer == "Custom Cap Co"


def test_dialog_fields_have_schema_tooltips(qtbot, tmp_path):
    """Every metadata field carries an on-hover explanation from the schema."""
    dlg = RecordingMetaDialog(tmp_path / "inv.tsv.recording_meta.json",
                              present_datatypes={"eeg", "meg"})
    qtbot.addWidget(dlg)
    assert "Manufacturer" in dlg._manufacturer.toolTip()
    assert dlg._line_freq.toolTip()           # PowerLineFrequency description
    assert dlg._eeg_reference.toolTip()
    assert dlg._dewar_position.toolTip()
    assert dlg._cap_manufacturer.toolTip()


def test_dialog_montage_suggestion_summary(qtbot, tmp_path):
    """The scan montage suggestions surface in the global dialog as a summary."""
    from PyQt6.QtWidgets import QLabel
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json", present_datatypes={"eeg"},
        montage_suggestions=["standard_1005 (64/64)", "standard_1020 (19/19)"],
    )
    qtbot.addWidget(dlg)
    hints = [w.text() for w in dlg.findChildren(QLabel)
             if "scan suggests" in w.text() and "standard_1005 (64/64)" in w.text()]
    assert hints


def test_dialog_manufacturer_suggestion_summary(qtbot, tmp_path):
    """The scan manufacturer suggestions surface in the global dialog too."""
    from PyQt6.QtWidgets import QLabel
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json", present_datatypes={"meg"},
        manufacturer_suggestions=["MEGIN / Elekta / Neuromag"],
    )
    qtbot.addWidget(dlg)
    hints = [w.text() for w in dlg.findChildren(QLabel)
             if "scan suggests" in w.text() and "MEGIN" in w.text()]
    assert hints


def test_dialog_institution_is_agnostic_not_in_device_group(qtbot, tmp_path):
    """Institution is agnostic: it has its own group and shows even for an
    MRI-only dataset (where the device/EEG/MEG groups are hidden)."""
    from PyQt6.QtWidgets import QGroupBox
    dlg = RecordingMetaDialog(tmp_path / "inv.tsv.recording_meta.json",
                              present_datatypes={"mri"})
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    titles = [g.title() for g in dlg.findChildren(QGroupBox)]
    inst = [t for t in titles if "Institution" in t]
    assert inst and "any modality" in inst[0]
    # The device group does NOT mention institution any more.
    device = [t for t in titles if t.startswith("Acquisition system")]
    assert device and "institution" not in device[0].lower()
    # Institution fields are editable even on an MRI-only dataset.
    dlg._institution_name.setText("Uni Oldenburg")
    assert dlg.build_spec().defaults.institution_name == "Uni Oldenburg"
