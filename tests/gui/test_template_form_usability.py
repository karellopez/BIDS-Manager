"""The form has to be usable, not only correct.

Four things a user asked for after living with it: the curated dropdowns that
had quietly become plain boxes, being told which missing fields the data
normally supplies, the PSD action sitting with the field it answers, and panes
that can be squeezed narrow instead of setting their own floor.
"""

from __future__ import annotations

import pandas as pd
import pytest
from PyQt6.QtWidgets import QComboBox

from bidsmgr.gui.models import InventoryTableModel
from bidsmgr.gui.properties_panel import PropertiesPanel
from bidsmgr.gui.recording_meta_dialog import RecordingMetaDialog
from bidsmgr.gui.widgets.template_form import (
    build_field_widget,
    read_field_widget,
    write_field_widget,
)
from bidsmgr.gui.widgets.validation_pane import ValidationPane
from bidsmgr.metadata.template_plan import (
    ORIGIN_USUALLY_DERIVED,
    sidecar_section,
)
from bidsmgr.recording_meta import CURATED_SUGGESTIONS

pytestmark = pytest.mark.gui


def _field(datatype: str, suffix: str, name: str):
    return {f.name: f for f in sidecar_section(datatype, suffix).fields}[name]


# ---------------------------------------------------------------------------
# Common answers, offered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "datatype,suffix,name",
    [
        ("eeg", "eeg", "EEGReference"),
        ("eeg", "eeg", "EEGGround"),
        ("eeg", "eeg", "Manufacturer"),
        ("anat", "T1w", "PulseSequenceType"),
        ("anat", "T1w", "ParallelAcquisitionTechnique"),
        ("pet", "pet", "ReconFilterType"),
    ],
)
def test_a_field_with_common_answers_offers_them(qtbot, datatype, suffix, name):
    """BIDS leaves these free text, which is not a reason to leave the user a
    blank box. The list is a suggestion; anything can still be typed."""
    field = _field(datatype, suffix, name)
    widget = build_field_widget(field, CURATED_SUGGESTIONS.get(name, ()))
    qtbot.addWidget(widget)
    assert isinstance(widget, QComboBox), f"{name} should offer its common answers"
    assert widget.isEditable(), f"{name} must still accept anything typed"
    assert widget.count() > 1


def test_a_numeric_field_gets_the_list_too(qtbot):
    """The mains is 50 or 60 everywhere on earth. Typing it as a number is what
    BIDS wants, and it came back as the string "60" before."""
    field = _field("eeg", "eeg", "PowerLineFrequency")
    widget = build_field_widget(field, CURATED_SUGGESTIONS["PowerLineFrequency"])
    qtbot.addWidget(widget)
    assert isinstance(widget, QComboBox)
    write_field_widget(widget, 60)
    assert read_field_widget(widget, field) == 60


def test_a_non_numeric_answer_to_an_anyof_field_stays_text(qtbot):
    field = _field("eeg", "eeg", "PowerLineFrequency")
    widget = build_field_widget(field, CURATED_SUGGESTIONS["PowerLineFrequency"])
    qtbot.addWidget(widget)
    write_field_widget(widget, "n/a")
    assert read_field_widget(widget, field) == "n/a"


# ---------------------------------------------------------------------------
# Saying why a field is being asked about
# ---------------------------------------------------------------------------


def test_a_field_the_data_usually_carries_is_marked_as_such():
    """Two different problems wear the same face. A field no converter supplies
    has to be typed. A field a converter USUALLY supplies and did NOT often
    means the value is still at the scanner, or an anonymiser removed it, and
    the user can go and get it rather than invent one.

    The case only arises once the scan has measured this dataset: EEGChannelCount
    is normally answered, so it is asked about only when these files lack it.
    """
    measured = {"SamplingFrequency": 500.0}   # no channel counts in this dataset
    by_name = {
        f.name: f for f in sidecar_section("eeg", "eeg", answered=measured).fields
    }
    assert by_name["EEGChannelCount"].origin == ORIGIN_USUALLY_DERIVED
    assert by_name["EEGReference"].origin != ORIGIN_USUALLY_DERIVED
    assert "SamplingFrequency" not in by_name


# ---------------------------------------------------------------------------
# The PSD action belongs with the field it answers
# ---------------------------------------------------------------------------


def test_compute_psd_sits_with_the_line_frequency(qtbot):
    from PyQt6.QtWidgets import QPushButton

    df = pd.DataFrame([{
        "include": "1", "proposed_datatype": "eeg", "bids_guess_suffix": "eeg",
        "proposed_basename": "sub-001_task-rest_eeg", "source_file": "/raw/a.edf",
        "BIDS_name": "sub-001", "line_freq": "", "montage": "",
        "eeg_reference": "", "eeg_ground": "",
    }])
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(InventoryTableModel(df))
    panel.set_selected_row(0)

    psd = [b for b in panel.findChildren(QPushButton) if "PSD" in (b.text() or "")]
    assert psd, "the PSD action should be somewhere in the panel"
    # Reading the spectrum is how you decide 50 or 60, so it belongs in the same
    # section as the field, not in a block of its own.
    section = psd[0].parent()
    while section is not None and not hasattr(section, "_title_text"):
        section = section.parent()
    assert section is not None and section._title_text == "Sidecar fields"


# ---------------------------------------------------------------------------
# Panes that can be squeezed
# ---------------------------------------------------------------------------


def test_the_metadata_dialog_can_be_squeezed_narrow(qtbot, tmp_path):
    """Long group titles, a picker listing them, and section headers each
    reported their full text as a MINIMUM, so the dialog would not narrow past
    483 px however small its contents could be drawn."""
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"eeg"}, present_pairs=[("eeg", "eeg")],
    )
    qtbot.addWidget(dlg)
    dlg.show()
    assert dlg.minimumSizeHint().width() < 260


@pytest.mark.parametrize("factory", [PropertiesPanel, ValidationPane])
def test_a_side_pane_sets_no_floor_of_its_own(qtbot, factory):
    pane = factory()
    qtbot.addWidget(pane)
    pane.show()
    assert pane.minimumSizeHint().width() <= pane.minimumWidth()
