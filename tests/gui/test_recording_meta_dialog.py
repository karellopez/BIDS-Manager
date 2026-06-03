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
    """The whole dialog scrolls (not just the inner table/list)."""
    from PyQt6.QtWidgets import QScrollArea
    dlg = RecordingMetaDialog(tmp_path / "inv.tsv.recording_meta.json")
    qtbot.addWidget(dlg)
    assert dlg.findChildren(QScrollArea), "body should be wrapped in a scroll area"
    # Inner widgets are bounded so the OUTER scroll engages for the overall layout.
    assert 0 < dlg._events.maximumHeight() <= 200
    assert 0 < dlg._phenotype.maximumHeight() <= 110
