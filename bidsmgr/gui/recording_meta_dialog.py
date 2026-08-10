"""Dataset-level recording-metadata editor (the "global changes" surface).

Opened from the inspection footer's "Recording metadata" button, this dialog
edits the dataset-wide enrichment defaults plus the event-code labels, backed by
the same ``<inventory>.tsv.recording_meta.json`` scaffold the scan writes and the
convert verb auto-discovers. Per-row overrides (reference, ground, montage,
line_freq, demographics) live in the inspection table; this is for the values
shared across the dataset.

``montage`` and ``line_freq`` are dropdown-only here too (never hand-typed).
Auxiliary-channel / filter / extras tables are a later addition; this v1 covers
the highest-value fields: device, institution, reference/ground, and the event
map the scan seeded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..metadata.template_plan import dataset_description_section
from ..recording_meta import (
    COMMON_CAP_MANUFACTURERS,
    COMMON_MANUFACTURERS,
    COMMON_RADIONUCLIDES,
    COMMON_TRACERS,
    MASS_UNITS,
    MODES_OF_ADMINISTRATION,
    PET_ACQUISITION_MODES,
    PET_IMAGE_UNITS,
    RADIOACTIVITY_UNITS,
    SPECIFIC_RADIOACTIVITY_UNITS,
    AcquisitionSpec,
    PetAcquisitionSpec,
    RecordingMetaSpec,
    dump_spec,
    load_spec,
)
from .delegates import builtin_montages
from .metadata_help import tooltip_for
from .theme_manager import CUR

_NONE = "(none)"
_BLANK = "(blank)"

# Display names + an "EEG and MEG"-style joiner so a section whose field is
# shared by several present modalities is labelled with all of them.
_MODALITY_NAMES = {
    "eeg": "EEG", "meg": "MEG", "ieeg": "iEEG", "nirs": "NIRS", "pet": "PET",
}
_MODALITY_ORDER = ("eeg", "meg", "ieeg", "nirs", "pet")

# Datatypes whose presence makes the modality-specific region worth showing.
_SPECIFIC_DATATYPES = frozenset({"eeg", "meg", "ieeg", "nirs", "pet"})


def _join_modalities(mods: list[str]) -> str:
    names = [_MODALITY_NAMES.get(m, m.upper()) for m in mods]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _tighten_form(form: QFormLayout) -> None:
    """Compact a form layout so the stacked group boxes stay dense.

    Keeps the default window showing as many of the metadata fields as fit
    before the outer scroll area takes over for the rest.
    """
    form.setContentsMargins(8, 6, 8, 6)
    form.setVerticalSpacing(4)
    form.setHorizontalSpacing(8)



class _AuthorsEditor(QWidget):
    """A row per author, family name and given name kept apart.

    BIDS wants a list of strings and the convention is "Family, Given". A plain
    text box makes that a trap: split on commas and "Lopez, Karel" becomes two
    people; ask for one per line and the user has to guess the order. Two boxes
    per row remove the guess, and the value is composed the way the standard
    expects it.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[QWidget, QLineEdit, QLineEdit]] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        self._rows_box = QVBoxLayout()
        self._rows_box.setSpacing(2)
        outer.addLayout(self._rows_box)
        add = QPushButton("Add author")
        add.clicked.connect(lambda: self._add_row())
        outer.addWidget(add, 0, Qt.AlignmentFlag.AlignLeft)
        self._add_row()

    def _add_row(self, family: str = "", given: str = "") -> None:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        family_edit = QLineEdit(family)
        family_edit.setPlaceholderText("Family name")
        given_edit = QLineEdit(given)
        given_edit.setPlaceholderText("Given name")
        remove = QPushButton("\u2212")
        remove.setFixedWidth(26)
        remove.setToolTip("Remove this author")
        layout.addWidget(family_edit, 3)
        layout.addWidget(given_edit, 2)
        layout.addWidget(remove, 0)
        self._rows_box.addWidget(row)
        entry = (row, family_edit, given_edit)
        self._rows.append(entry)

        def drop() -> None:
            if entry in self._rows:
                self._rows.remove(entry)
            row.setParent(None)
            if not self._rows:
                self._add_row()

        remove.clicked.connect(drop)

    def value(self) -> list[str]:
        """The authors as BIDS wants them, blanks dropped."""
        out: list[str] = []
        for _row, family_edit, given_edit in self._rows:
            family = family_edit.text().strip()
            given = given_edit.text().strip()
            if family and given:
                out.append(f"{family}, {given}")
            elif family or given:
                out.append(family or given)
        return out

    def set_value(self, values) -> None:
        for row, _f, _g in list(self._rows):
            row.setParent(None)
        self._rows.clear()
        for entry in values or []:
            family, _, given = str(entry).partition(",")
            self._add_row(family.strip(), given.strip())
        if not self._rows:
            self._add_row()


class RecordingMetaDialog(QDialog):
    """Edit the dataset-level recording-metadata scaffold."""

    def __init__(
        self,
        scaffold_path: Path,
        present_datatypes: Optional[set[str]] = None,
        parent: Optional[QWidget] = None,
        montage_suggestions: Optional[list[str]] = None,
        manufacturer_suggestions: Optional[list[str]] = None,
        pet_suggestions: Optional[dict[str, list[str]]] = None,
        example_paths: Optional[dict] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset metadata")
        # Distinct per-recording suggestions the scan found, surfaced as read-only
        # hints beside the matching dataset defaults (not auto-applied).
        self._montage_suggestions = list(montage_suggestions or [])
        self._manufacturer_suggestions = list(manufacturer_suggestions or [])
        # A fixed, modest default size. The whole body lives in a scroll area,
        # so when the visible sections need more room the OUTER scroll bar moves
        # the entire window content as one - the user asked for a scrollable
        # window, not a window that resizes itself to the content or whose inner
        # tables stretch. Still freely resizable (enlarge to see more at once).
        self.resize(560, 640)
        self._scaffold_path = Path(scaffold_path)
        # Datatypes the scanned dataset actually contains. Drives which sections
        # / fields make sense: the recording-acquisition section is hidden for a
        # dataset with no EEG/MEG, and within it the scalp-EEG fields
        # (reference / ground / montage / cap) are hidden for MEG-only. The
        # agnostic sections (events, phenotype) always show.
        self._present = set(present_datatypes or {"eeg", "meg", "ieeg", "nirs"})
        # One real path per (datatype, suffix) from the scan, so a group can
        # name the file its answers reach instead of "sub-..._<datatype>.json".
        self._example_paths = dict(example_paths or {})
        # Read-only scan hints, shown beside the fields they inform. Same
        # contract as the montage and manufacturer hints: proposed, never
        # applied, because a vendor string can always parse wrongly.
        pet_suggestions = pet_suggestions or {}
        self._pet_tracer_suggestions = list(pet_suggestions.get("tracer", []))
        self._pet_dose_suggestions = list(pet_suggestions.get("dose", []))
        self._pet_recon_suggestions = list(pet_suggestions.get("recon", []))
        self._spec = self._load()

        outer = QVBoxLayout(self)
        outer.setSpacing(6)
        intro = QLabel(
            "Dataset-wide metadata, grouped by where it is written. "
            "Modality-agnostic sections apply to any dataset; modality-specific "
            "sections only affect their datatype's sidecars. Per-recording "
            "overrides live in the inspection table."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        # Region 1 - MODALITY-SPECIFIC defaults (the recording sidecars): a
        # device/site block (all electrophysiology) + an EEG/iEEG reference &
        # montage block. The region header is hidden for a dataset with no
        # EEG/MEG (only the agnostic region then shows).
        self._specific_region = self._region_label(
            "Modality-specific defaults", agnostic=False)
        bl.addWidget(self._specific_region)
        # One acquisition group per modality, not one shared between them: an
        # EEG amplifier and a MEG dewar are different instruments, and a study
        # running both could previously state only one manufacturer, one model
        # and one mains frequency for the pair.
        self._device_boxes: dict[str, QGroupBox] = {}
        self._device_widgets: dict[str, dict] = {}
        electrophysiology = [
            m for m in _MODALITY_ORDER if m in self._present and m != "pet"
        ]
        # Even with no electrophysiology present the widgets must exist: a
        # shared scaffold may carry EEG values, and build_spec preserves what it
        # cannot show rather than wiping it.
        for modality in electrophysiology or ["eeg"]:
            box = self._build_device_group(modality)
            self._device_boxes[modality] = box
            if not electrophysiology:
                box.setVisible(False)
        self._device_box = next(iter(self._device_boxes.values()))

        self._eeg_box = self._build_eeg_group()
        self._meg_box = self._build_meg_group()
        for box in self._device_boxes.values():
            bl.addWidget(box)
        bl.addWidget(self._eeg_box)
        bl.addWidget(self._meg_box)
        # PET: four groups, split the way the acquisition itself divides, so
        # the ~40 required fields read as four short forms instead of one wall.
        self._pet_boxes = [
            self._build_pet_tracer_group(),
            self._build_pet_administration_group(),
            self._build_pet_acquisition_group(),
            self._build_pet_recon_group(),
        ]
        for box in self._pet_boxes:
            bl.addWidget(box)
        # Region 2 - MODALITY-AGNOSTIC (apply to any dataset, incl. MRI):
        # institution (site info, written to every modality's sidecar), events
        # and phenotype, each writing to its own destination.
        bl.addWidget(self._region_label("Modality-agnostic", agnostic=True))
        bl.addWidget(self._build_dataset_description_group())
        bl.addWidget(self._build_institution_group())
        bl.addWidget(self._build_event_group())
        bl.addWidget(self._build_participants_group())
        bl.addWidget(self._build_phenotype_group())
        bl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)

        # Jump straight to a section. The layout is unchanged, so everything is
        # still there to scroll through if you prefer, but a dataset with four
        # modalities makes a long window and hunting for the PET reconstruction
        # box by dragging is not a good use of anyone's time.
        self._section_picker = QComboBox()
        self._section_picker.setToolTip("Jump to a section")
        picker_row = QWidget()
        picker_layout = QHBoxLayout(picker_row)
        picker_layout.setContentsMargins(0, 0, 0, 0)
        picker_layout.setSpacing(6)
        picker_layout.addWidget(QLabel("Go to:"))
        picker_layout.addWidget(self._section_picker, 1)
        outer.addWidget(picker_row)

        self._sections: list[QGroupBox] = [
            box for box in body.findChildren(QGroupBox) if box.title()
        ]
        for box in self._sections:
            self._section_picker.addItem(box.title())

        def jump(index: int) -> None:
            if 0 <= index < len(self._sections):
                scroll.ensureWidgetVisible(self._sections[index], 0, 0)

        self._section_picker.activated.connect(jump)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._populate()
        self._populate_dataset_description()
        self._populate_device_groups()
        self._apply_modality_constraints()

    def _region_label(self, text: str, *, agnostic: bool) -> QLabel:
        """A bold, colour-coded region divider (agnostic = teal, modality-
        specific = purple) separating the two metadata regions."""
        pal = CUR()
        color = pal["teal"] if agnostic else pal["purple"]
        lbl = QLabel(
            f'<span style="color:{color};font-weight:800;'
            f'letter-spacing:0.6px;">{text.upper()}</span>'
        )
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setStyleSheet(
            f"background: transparent; border-bottom: 1px solid {color}; "
            "padding-bottom: 2px;"
        )
        return lbl

    def _modalities_label(self, applicable: set[str]) -> str:
        """"EEG and MEG"-style label of the PRESENT modalities a field applies
        to (so a field shared by several modalities names them all)."""
        present_app = [
            m for m in _MODALITY_ORDER if m in self._present and m in applicable
        ]
        return _join_modalities(present_app)

    def _apply_modality_constraints(self) -> None:
        """Show/hide modality-specific sections by what was scanned.

        A section is shown only when its modality was actually scanned: the
        EEG/MEG device block for electrophysiology, the montage and reference
        block only with scalp EEG or iEEG, the MEG block only with MEG, and the
        four PET blocks only with PET. The region header hides when none apply,
        which is what an MRI-only dataset sees.
        """
        has_eeg_meg = bool(self._present & {"eeg", "meg", "ieeg", "nirs"})
        has_eeg = bool(self._present & {"eeg", "ieeg"})
        has_meg = "meg" in self._present
        has_pet = "pet" in self._present
        # Flags consulted by build_spec so a hidden section's loaded values are
        # preserved (not clobbered by the empty, never-shown widgets).
        self._device_applies = has_eeg_meg
        self._eeg_applies = has_eeg
        self._meg_applies = has_meg
        self._pet_applies = has_pet
        for box in self._device_boxes.values():
            box.setVisible(has_eeg_meg)
        self._eeg_box.setVisible(has_eeg)
        self._meg_box.setVisible(has_meg)
        for box in self._pet_boxes:
            box.setVisible(has_pet)
        self._specific_region.setVisible(has_eeg_meg or has_pet)

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def _form_row(self, form: QFormLayout, label_text: str, widget, ui_key: str) -> None:
        """Add a labelled field row with the schema tooltip on BOTH the label
        and the field (so hovering either shows the explanation)."""
        tip = tooltip_for(ui_key)
        label = QLabel(label_text)
        if tip:
            label.setToolTip(tip)
            widget.setToolTip(tip)
        form.addRow(label, widget)

    def _build_device_group(self, modality: str) -> QGroupBox:
        """The DEVICE that produced one modality's recordings.

        One group per modality: an EEG amplifier and a MEG dewar are different
        instruments, so a shared group could record only one of them and the
        other silently claimed it.
        """
        target = self._target_for(modality)
        box = QGroupBox(f"Acquisition system  ·  {modality.upper()}  →  {target}")
        form = QFormLayout(box)
        _tighten_form(form)

        manufacturer = QComboBox()
        manufacturer.setEditable(True)  # pick a default OR type another
        manufacturer.addItem("")
        manufacturer.addItems(COMMON_MANUFACTURERS)
        amplifier_model = QLineEdit()
        software_versions = QLineEdit()
        line_freq = QComboBox()
        line_freq.addItems([_BLANK, "50", "60"])

        self._form_row(form, "Manufacturer:", manufacturer, "manufacturer")
        hint = self._suggestion_label(self._manufacturer_suggestions)
        if hint is not None:
            form.addRow("", hint)
        self._form_row(form, "Amplifier / system model:", amplifier_model, "amplifier_model")
        self._form_row(form, "Software versions:", software_versions, "software_versions")
        self._form_row(form, "Power line frequency (Hz):", line_freq, "line_freq")

        self._device_widgets[modality] = {
            "manufacturer": manufacturer,
            "amplifier_model": amplifier_model,
            "software_versions": software_versions,
            "power_line_freq": line_freq,
        }
        # The first modality keeps the historical attribute names so the rest
        # of the dialog, and its tests, go on working unchanged.
        if not hasattr(self, "_manufacturer"):
            self._manufacturer = manufacturer
            self._amplifier_model = amplifier_model
            self._software_versions = software_versions
            self._line_freq = line_freq
        return box

    def _target_for(self, modality: str, suffix: str = "") -> str:
        """The file a modality's answers land in, from a real scanned row."""
        suffix = suffix or modality
        path = self._example_paths.get((modality, suffix), "")
        return path.split("/")[-1] if path else f"sub-..._{suffix}.json"

    def _read_device_groups(self, spec) -> None:
        """Store each modality's acquisition system in its own block.

        The shared block keeps what belongs to the whole study; an instrument
        belongs to the modality that used it, so a study running two systems
        records both instead of one overwriting the other.
        """
        for modality, widgets in self._device_widgets.items():
            values: dict = {}
            for attr, widget in widgets.items():
                text = (
                    widget.currentText() if isinstance(widget, QComboBox)
                    else widget.text()
                ).strip()
                if not text or text == _BLANK:
                    continue
                if attr == "power_line_freq":
                    try:
                        values[attr] = float(text)
                    except ValueError:
                        continue
                else:
                    values[attr] = text
            if values:
                block = spec.modality_defaults.get(modality) or AcquisitionSpec()
                spec.modality_defaults[modality] = block.model_copy(update=values)
            else:
                spec.modality_defaults.pop(modality, None)

    def _populate_device_groups(self) -> None:
        """Fill each modality's group from its own block, then the shared one."""
        for modality, widgets in self._device_widgets.items():
            block = self._spec.modality_defaults.get(modality)
            for attr, widget in widgets.items():
                value = getattr(block, attr, None) if block is not None else None
                if value in (None, ""):
                    value = getattr(self._spec.defaults, attr, None)
                if value in (None, ""):
                    continue
                text = (
                    str(int(value)) if attr == "power_line_freq" and float(value).is_integer()
                    else str(value)
                )
                if isinstance(widget, QComboBox):
                    widget.setCurrentText(text)
                else:
                    widget.setText(text)

    def _build_dataset_description_group(self) -> QGroupBox:
        """Who made the dataset and under what terms.

        Nothing here can be read from a recording, and until now there was
        nowhere to say it: the fields were modelled and writable from the
        command line but never offered, which is why every dataset the tool
        produced reported NO_AUTHORS.
        """
        box = QGroupBox(
            "Dataset description  ·  modality-agnostic  →  dataset_description.json"
        )
        form = QFormLayout(box)
        _tighten_form(form)

        self._dd_name = QLineEdit()
        self._dd_name.setPlaceholderText("the study's title")
        self._dd_authors = _AuthorsEditor()
        self._dd_license = QComboBox()
        self._dd_license.setEditable(True)
        self._dd_license.addItems(["", "CC0-1.0", "CC-BY-4.0", "PDDL-1.0"])
        self._dd_acknowledgements = QLineEdit()
        self._dd_how_to_acknowledge = QLineEdit()
        self._dd_funding = QLineEdit()
        self._dd_funding.setPlaceholderText("one grant per line is also fine")
        self._dd_ethics = QLineEdit()
        self._dd_references = QLineEdit()
        self._dd_doi = QLineEdit()
        self._dd_doi.setPlaceholderText("doi:10.…")

        # The level marks come from the schema, so they follow the BIDS version
        # in use rather than being asserted here.
        levels = {f.name: f.level for f in dataset_description_section().fields}

        def label(field: str, text: str) -> str:
            mark = {"required": " *", "recommended": " ·"}.get(levels.get(field, ""), "")
            return f"{text}{mark}:"

        form.addRow(label("Name", "Dataset name"), self._dd_name)
        form.addRow(label("Authors", "Authors"), self._dd_authors)
        form.addRow(label("License", "License"), self._dd_license)
        form.addRow(label("Acknowledgements", "Acknowledgements"), self._dd_acknowledgements)
        form.addRow(label("HowToAcknowledge", "How to acknowledge"), self._dd_how_to_acknowledge)
        form.addRow(label("Funding", "Funding"), self._dd_funding)
        form.addRow(label("EthicsApprovals", "Ethics approvals"), self._dd_ethics)
        form.addRow(label("ReferencesAndLinks", "References and links"), self._dd_references)
        form.addRow(label("DatasetDOI", "Dataset DOI"), self._dd_doi)
        return box

    def _populate_dataset_description(self) -> None:
        dd = self._spec.dataset_description
        self._dd_name.setText(dd.name or "")
        self._dd_authors.set_value(dd.authors)
        self._dd_license.setCurrentText(dd.license or "")
        self._dd_acknowledgements.setText(dd.acknowledgements or "")
        self._dd_how_to_acknowledge.setText(dd.how_to_acknowledge or "")
        self._dd_funding.setText("; ".join(dd.funding or []))
        self._dd_ethics.setText("; ".join(dd.ethics_approvals or []))
        self._dd_references.setText("; ".join(dd.references_and_links or []))
        self._dd_doi.setText(dd.dataset_doi or "")

    def _read_dataset_description(self, spec) -> None:
        def split(text: str) -> list[str]:
            # Semicolons, because a grant title may itself contain a comma.
            return [p.strip() for p in text.split(";") if p.strip()]

        dd = spec.dataset_description
        dd.name = self._dd_name.text().strip() or None
        dd.authors = self._dd_authors.value()
        dd.license = self._dd_license.currentText().strip() or None
        dd.acknowledgements = self._dd_acknowledgements.text().strip() or None
        dd.how_to_acknowledge = self._dd_how_to_acknowledge.text().strip() or None
        dd.funding = split(self._dd_funding.text())
        dd.ethics_approvals = split(self._dd_ethics.text())
        dd.references_and_links = split(self._dd_references.text())
        dd.dataset_doi = self._dd_doi.text().strip() or None

    def _build_institution_group(self) -> QGroupBox:
        # Institution / site info is modality-AGNOSTIC: it belongs in every
        # modality's sidecar (MRI gets it from DICOM; EEG/MEG from this default).
        box = QGroupBox("Institution / site  ·  any modality  →  sidecar / dataset")
        form = QFormLayout(box)
        _tighten_form(form)
        self._institution_name = QLineEdit()
        self._institution_dept = QLineEdit()
        self._form_row(form, "Institution name:", self._institution_name, "institution_name")
        self._form_row(form, "Institution department:", self._institution_dept, "institution_dept")
        return box

    def _build_eeg_group(self) -> QGroupBox:
        # Scalp-EEG / iEEG only sidecar fields (MEG has no scalp reference /
        # ground / montage). The title names the present EEG/iEEG modalities.
        mods = self._modalities_label({"eeg", "ieeg"}) or "EEG / iEEG"
        box = QGroupBox(
            f"Reference, ground & montage  ·  {mods}  →  sub-..._eeg.json + electrodes.tsv"
        )
        form = QFormLayout(box)
        _tighten_form(form)

        self._cap_manufacturer = QComboBox()
        self._cap_manufacturer.setEditable(True)  # pick a default OR type another
        self._cap_manufacturer.addItem("")
        self._cap_manufacturer.addItems(COMMON_CAP_MANUFACTURERS)
        self._eeg_reference = QLineEdit()
        self._eeg_ground = QLineEdit()
        self._montage = QComboBox()
        self._montage.addItem(_NONE)
        self._montage.addItems(builtin_montages())

        self._form_row(form, "Default EEG reference:", self._eeg_reference, "eeg_reference")
        self._form_row(form, "Default EEG ground:", self._eeg_ground, "eeg_ground")
        self._form_row(form, "Default montage:", self._montage, "montage")
        # Read-only summary of the per-recording montage matches the scan found
        # (the dataset default applies to all; this shows what was detected).
        hint = self._suggestion_label(self._montage_suggestions)
        if hint is not None:
            form.addRow("", hint)
        self._form_row(form, "Cap manufacturer:", self._cap_manufacturer, "cap_manufacturer")
        return box

    def _suggestion_label(self, suggestions: list[str]) -> Optional[QLabel]:
        """A dim 'scan suggests: ...' summary of distinct per-recording scan
        suggestions (montage or manufacturer), or ``None`` when there are none."""
        if not suggestions:
            return None
        pal = CUR()
        shown = suggestions[:3]
        more = len(suggestions) - len(shown)
        text = "; ".join(shown) + (f" (+{more} more)" if more > 0 else "")
        lbl = QLabel(
            f'<span style="color:{pal["dim"]};">scan suggests: </span>'
            f'<span style="color:{pal["teal"]};">{text}</span>'
        )
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        lbl.setToolTip(
            "Detected per recording at scan. Set a dataset default above, or "
            "override per recording in the inspection table; not auto-applied."
        )
        return lbl

    def _build_meg_group(self) -> QGroupBox:
        # MEG-only fields mne-bids CANNOT derive from the recording. The
        # channel-derived MEG facts (continuous head localization, digitized
        # landmarks / head points, head-coil frequency, ...) are filled by
        # mne-bids and are intentionally NOT exposed here.
        box = QGroupBox(
            "MEG acquisition  ·  MEG  →  sub-..._meg.json"
        )
        form = QFormLayout(box)
        _tighten_form(form)

        self._dewar_position = QComboBox()
        self._dewar_position.setEditable(True)
        self._dewar_position.addItems(["", "upright", "supine"])
        self._associated_empty_room = QLineEdit()
        self._subject_artefact_description = QLineEdit()

        self._form_row(form, "Dewar position:", self._dewar_position, "dewar_position")
        self._form_row(form, "Associated empty-room:", self._associated_empty_room, "associated_empty_room")
        self._form_row(form, "Subject artefact description:",
                       self._subject_artefact_description, "subject_artefact_description")
        return box

    # ------------------------------------------------------------------
    # PET
    # ------------------------------------------------------------------

    def _pet_combo(self, options, *, editable: bool = True) -> QComboBox:
        """A dropdown seeded with the common values but still free-text.

        The vocabularies are the common cases, not a closed set: a study using
        a tracer nobody has listed must still be describable.
        """
        combo = QComboBox()
        combo.setEditable(editable)
        combo.addItem("")
        combo.addItems(options)
        combo.setProperty("class", "ent-input")
        return combo

    def _build_pet_tracer_group(self) -> QGroupBox:
        # What was injected. None of this is in the DICOM header in a form BIDS
        # can use, so it comes from the radiochemistry record.
        box = QGroupBox("Tracer  ·  PET  →  sub-..._pet.json")
        form = QFormLayout(box)
        _tighten_form(form)

        self._pet_tracer_name = self._pet_combo(COMMON_TRACERS)
        self._pet_radionuclide = self._pet_combo(COMMON_RADIONUCLIDES)

        self._form_row(form, "Tracer name:", self._pet_tracer_name, "TracerName")
        self._form_row(form, "Radionuclide:", self._pet_radionuclide,
                       "TracerRadionuclide")
        hint = self._suggestion_label(self._pet_tracer_suggestions)
        if hint is not None:
            form.addRow("", hint)
        return box

    def _build_pet_administration_group(self) -> QGroupBox:
        # How much went in, in what form, and when. Purely from the injection
        # record: a scanner cannot know any of it.
        box = QGroupBox(
            "Radiochemistry & administration  ·  PET  →  sub-..._pet.json"
        )
        form = QFormLayout(box)
        _tighten_form(form)

        self._pet_injected_radioactivity = QLineEdit()
        self._pet_injected_radioactivity.setPlaceholderText("e.g. 44.4")
        self._pet_injected_radioactivity_units = self._pet_combo(RADIOACTIVITY_UNITS)
        self._pet_injected_mass = QLineEdit()
        self._pet_injected_mass_units = self._pet_combo(MASS_UNITS)
        self._pet_specific_radioactivity = QLineEdit()
        self._pet_specific_radioactivity_units = self._pet_combo(
            SPECIFIC_RADIOACTIVITY_UNITS)
        self._pet_mode_of_administration = self._pet_combo(MODES_OF_ADMINISTRATION)
        self._pet_injection_start = QLineEdit()
        self._pet_injection_start.setPlaceholderText("seconds relative to TimeZero")

        self._form_row(form, "Injected radioactivity:",
                       self._pet_injected_radioactivity, "InjectedRadioactivity")
        self._form_row(form, "     units:",
                       self._pet_injected_radioactivity_units,
                       "InjectedRadioactivityUnits")
        self._form_row(form, "Injected mass:", self._pet_injected_mass,
                       "InjectedMass")
        self._form_row(form, "     units:", self._pet_injected_mass_units,
                       "InjectedMassUnits")
        self._form_row(form, "Specific radioactivity:",
                       self._pet_specific_radioactivity, "SpecificRadioactivity")
        self._form_row(form, "     units:",
                       self._pet_specific_radioactivity_units,
                       "SpecificRadioactivityUnits")
        self._form_row(form, "Mode of administration:",
                       self._pet_mode_of_administration, "ModeOfAdministration")
        self._form_row(form, "Injection start:", self._pet_injection_start,
                       "InjectionStart")
        hint = self._suggestion_label(self._pet_dose_suggestions)
        if hint is not None:
            form.addRow("", hint)
        return box

    def _build_pet_acquisition_group(self) -> QGroupBox:
        box = QGroupBox("Acquisition  ·  PET  →  sub-..._pet.json")
        form = QFormLayout(box)
        _tighten_form(form)

        self._pet_time_zero = QLineEdit()
        self._pet_time_zero.setPlaceholderText("hh:mm:ss")
        self._pet_scan_start = QLineEdit()
        self._pet_scan_start.setPlaceholderText("seconds relative to TimeZero")
        self._pet_acquisition_mode = self._pet_combo(PET_ACQUISITION_MODES)
        self._pet_units = self._pet_combo(PET_IMAGE_UNITS)
        self._pet_body_part = QLineEdit()
        self._pet_attenuation_correction = QLineEdit()
        self._pet_image_decay_corrected = QComboBox()
        self._pet_image_decay_corrected.addItems(["", "true", "false"])
        self._pet_image_decay_correction_time = QLineEdit()

        self._form_row(form, "Time zero:", self._pet_time_zero, "TimeZero")
        self._form_row(form, "Scan start:", self._pet_scan_start, "ScanStart")
        self._form_row(form, "Acquisition mode:", self._pet_acquisition_mode,
                       "AcquisitionMode")
        self._form_row(form, "Image units:", self._pet_units, "Units")
        self._form_row(form, "Body part:", self._pet_body_part, "BodyPart")
        self._form_row(form, "Attenuation correction:",
                       self._pet_attenuation_correction, "AttenuationCorrection")
        self._form_row(form, "Image decay corrected:",
                       self._pet_image_decay_corrected, "ImageDecayCorrected")
        self._form_row(form, "     correction time:",
                       self._pet_image_decay_correction_time,
                       "ImageDecayCorrectionTime")
        return box

    def _build_pet_recon_group(self) -> QGroupBox:
        # The scanner DOES record this, but as free text whose grammar varies
        # by manufacturer, so the scan offers a parse and the user confirms it.
        box = QGroupBox("Reconstruction  ·  PET  →  sub-..._pet.json")
        form = QFormLayout(box)
        _tighten_form(form)

        self._pet_recon_method = QLineEdit()
        self._pet_recon_labels = QLineEdit()
        self._pet_recon_labels.setPlaceholderText("comma separated, e.g. iterations, subsets")
        self._pet_recon_values = QLineEdit()
        self._pet_recon_values.setPlaceholderText("comma separated, e.g. 3, 21")
        self._pet_recon_units = QLineEdit()
        self._pet_recon_units.setPlaceholderText("comma separated, e.g. none, none")
        self._pet_recon_filter_type = QLineEdit()
        self._pet_recon_filter_size = QLineEdit()

        self._form_row(form, "Method name:", self._pet_recon_method,
                       "ReconMethodName")
        self._form_row(form, "Parameter labels:", self._pet_recon_labels,
                       "ReconMethodParameterLabels")
        self._form_row(form, "Parameter values:", self._pet_recon_values,
                       "ReconMethodParameterValues")
        self._form_row(form, "Parameter units:", self._pet_recon_units,
                       "ReconMethodParameterUnits")
        self._form_row(form, "Filter type:", self._pet_recon_filter_type,
                       "ReconFilterType")
        self._form_row(form, "Filter size:", self._pet_recon_filter_size,
                       "ReconFilterSize")
        hint = self._suggestion_label(self._pet_recon_suggestions)
        if hint is not None:
            form.addRow("", hint)
        return box

    def _build_event_group(self) -> QGroupBox:
        box = QGroupBox("Events (trigger code -> label)  ·  modality-agnostic  →  events.tsv")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        self._events = QTableWidget(0, 2)
        self._events.setHorizontalHeaderLabels(["Code", "Label"])
        self._events.horizontalHeader().setStretchLastSection(True)
        # Bound the table to a natural, modest height. It keeps its own size
        # (it does not stretch to fill); when the stacked sections together
        # exceed the window, the OUTER scroll area moves the whole body.
        self._events.setMinimumHeight(96)
        self._events.setMaximumHeight(160)
        v.addWidget(self._events)

        row = QHBoxLayout()
        add = QPushButton("Add row")
        add.clicked.connect(lambda: self._events.insertRow(self._events.rowCount()))
        rem = QPushButton("Remove selected")
        rem.clicked.connect(self._remove_selected_event)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        v.addLayout(row)
        return box

    def _remove_selected_event(self) -> None:
        rows = sorted({i.row() for i in self._events.selectedIndexes()}, reverse=True)
        for r in rows:
            self._events.removeRow(r)

    def _build_participants_group(self) -> QGroupBox:
        box = QGroupBox("Participants spreadsheet  ·  modality-agnostic  →  participants.tsv")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        hint = QLabel(
            "Optional table keyed by participant_id. Its age / sex / handedness "
            "override the inventory; any extra columns (group, IQ, ...) are added "
            "to participants.tsv. A sibling <name>.json supplies column "
            "descriptions / Levels / Units."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)
        row = QHBoxLayout()
        self._participants_file = QLineEdit()
        self._participants_file.setObjectName("ent-input")
        self._participants_file.setReadOnly(True)
        self._participants_file.setPlaceholderText("(none)")
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._choose_participants_file)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self._participants_file.clear())
        row.addWidget(self._participants_file, 1)
        row.addWidget(browse)
        row.addWidget(clear)
        v.addLayout(row)
        return box

    def _choose_participants_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select participants spreadsheet", "",
            "Tables (*.tsv *.csv *.xlsx *.ods);;All files (*)",
        )
        if path:
            self._participants_file.setText(path)

    def _build_phenotype_group(self) -> QGroupBox:
        box = QGroupBox("Phenotype tables  ·  modality-agnostic  →  phenotype/")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        hint = QLabel(
            "Measure tables keyed by participant_id -> phenotype/<measure>.tsv + .json."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)
        self._phenotype = QListWidget()
        self._phenotype.setMinimumHeight(70)
        self._phenotype.setMaximumHeight(110)
        v.addWidget(self._phenotype)
        row = QHBoxLayout()
        add = QPushButton("Add file…")
        add.clicked.connect(self._add_phenotype_file)
        rem = QPushButton("Remove selected")
        rem.clicked.connect(self._remove_phenotype)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        v.addLayout(row)
        return box

    def _add_phenotype_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select phenotype table(s)", "",
            "Tables (*.tsv *.csv *.xlsx *.ods);;All files (*)",
        )
        existing = {self._phenotype.item(i).text() for i in range(self._phenotype.count())}
        for p in paths:
            if p and p not in existing:
                self._phenotype.addItem(p)

    def _remove_phenotype(self) -> None:
        for item in self._phenotype.selectedItems():
            self._phenotype.takeItem(self._phenotype.row(item))

    # ------------------------------------------------------------------
    # load / populate / save
    # ------------------------------------------------------------------

    def _load(self) -> RecordingMetaSpec:
        if self._scaffold_path.exists():
            try:
                return load_spec(self._scaffold_path)
            except Exception:
                pass
        return RecordingMetaSpec()

    def _populate(self) -> None:
        acq = self._spec.defaults
        self._manufacturer.setCurrentText(acq.manufacturer or "")
        self._amplifier_model.setText(acq.amplifier_model or "")
        self._software_versions.setText(acq.software_versions or "")
        self._cap_manufacturer.setCurrentText(acq.cap_manufacturer or "")
        self._institution_name.setText(acq.institution_name or "")
        self._institution_dept.setText(acq.institution_dept or "")
        self._eeg_reference.setText(acq.eeg_reference or "")
        self._eeg_ground.setText(acq.eeg_ground or "")
        self._set_combo(self._montage, acq.montage, _NONE)
        self._set_combo(
            self._line_freq,
            str(int(acq.power_line_freq)) if acq.power_line_freq else "",
            _BLANK,
        )
        # MEG-specific manual fields.
        self._dewar_position.setCurrentText(acq.dewar_position or "")
        self._associated_empty_room.setText(acq.associated_empty_room or "")
        self._subject_artefact_description.setText(acq.subject_artefact_description or "")

        self._populate_pet()

        event_map = self._spec.event_maps.get("*", {})
        self._events.setRowCount(0)
        for code, label in event_map.items():
            r = self._events.rowCount()
            self._events.insertRow(r)
            self._events.setItem(r, 0, QTableWidgetItem(str(code)))
            self._events.setItem(r, 1, QTableWidgetItem(str(label)))

        self._phenotype.clear()
        for p in self._spec.phenotype_files:
            self._phenotype.addItem(str(p))

        self._participants_file.setText(self._spec.participants_file or "")

    @staticmethod
    def _num_text(value) -> str:
        """Render a number for a line edit without a trailing ``.0``."""
        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _populate_pet(self) -> None:
        pet = self._spec.pet_defaults
        self._pet_tracer_name.setCurrentText(pet.tracer_name or "")
        self._pet_radionuclide.setCurrentText(pet.tracer_radionuclide or "")
        self._pet_injected_radioactivity.setText(
            self._num_text(pet.injected_radioactivity))
        self._pet_injected_radioactivity_units.setCurrentText(
            pet.injected_radioactivity_units or "")
        self._pet_injected_mass.setText(self._num_text(pet.injected_mass))
        self._pet_injected_mass_units.setCurrentText(pet.injected_mass_units or "")
        self._pet_specific_radioactivity.setText(
            self._num_text(pet.specific_radioactivity))
        self._pet_specific_radioactivity_units.setCurrentText(
            pet.specific_radioactivity_units or "")
        self._pet_mode_of_administration.setCurrentText(
            pet.mode_of_administration or "")
        self._pet_injection_start.setText(self._num_text(pet.injection_start))
        self._pet_time_zero.setText(pet.time_zero or "")
        self._pet_scan_start.setText(self._num_text(pet.scan_start))
        self._pet_acquisition_mode.setCurrentText(pet.acquisition_mode or "")
        self._pet_units.setCurrentText(pet.units or "")
        self._pet_body_part.setText(pet.body_part or "")
        self._pet_attenuation_correction.setText(pet.attenuation_correction or "")
        self._pet_image_decay_corrected.setCurrentText(
            "" if pet.image_decay_corrected is None
            else ("true" if pet.image_decay_corrected else "false")
        )
        self._pet_image_decay_correction_time.setText(
            self._num_text(pet.image_decay_correction_time))
        self._pet_recon_method.setText(pet.recon_method_name or "")
        self._pet_recon_labels.setText(", ".join(pet.recon_method_parameter_labels))
        self._pet_recon_values.setText(
            ", ".join(self._num_text(v) for v in pet.recon_method_parameter_values))
        self._pet_recon_units.setText(", ".join(pet.recon_method_parameter_units))
        self._pet_recon_filter_type.setText(pet.recon_filter_type or "")
        self._pet_recon_filter_size.setText(self._num_text(pet.recon_filter_size))

    def _build_pet_spec(self) -> PetAcquisitionSpec:
        """Read the PET form back into a spec block.

        A hidden PET section keeps its loaded values rather than reading the
        empty widgets, so an EEG-only dataset sharing a scaffold never wipes
        PET fields somebody else filled in.
        """
        if not self._pet_applies:
            return self._spec.pet_defaults

        def _txt(edit: QLineEdit) -> Optional[str]:
            return edit.text().strip() or None

        def _num(edit: QLineEdit) -> Optional[float]:
            raw = edit.text().strip()
            if not raw:
                return None
            try:
                return float(raw)
            except ValueError:
                # Keep the previous value rather than dropping what the user
                # typed on the floor; the field validates on the next edit.
                return None

        def _combo(combo: QComboBox) -> Optional[str]:
            return combo.currentText().strip() or None

        def _csv(edit: QLineEdit) -> list[str]:
            return [p.strip() for p in edit.text().split(",") if p.strip()]

        def _csv_num(edit: QLineEdit) -> list[float]:
            out = []
            for part in _csv(edit):
                try:
                    out.append(float(part))
                except ValueError:
                    continue
            return out

        decay = self._pet_image_decay_corrected.currentText().strip()

        return PetAcquisitionSpec(
            tracer_name=_combo(self._pet_tracer_name),
            tracer_radionuclide=_combo(self._pet_radionuclide),
            injected_radioactivity=_num(self._pet_injected_radioactivity),
            injected_radioactivity_units=_combo(
                self._pet_injected_radioactivity_units),
            injected_mass=_num(self._pet_injected_mass),
            injected_mass_units=_combo(self._pet_injected_mass_units),
            specific_radioactivity=_num(self._pet_specific_radioactivity),
            specific_radioactivity_units=_combo(
                self._pet_specific_radioactivity_units),
            mode_of_administration=_combo(self._pet_mode_of_administration),
            injection_start=_num(self._pet_injection_start),
            time_zero=_txt(self._pet_time_zero),
            scan_start=_num(self._pet_scan_start),
            acquisition_mode=_combo(self._pet_acquisition_mode),
            units=_combo(self._pet_units),
            body_part=_txt(self._pet_body_part),
            attenuation_correction=_txt(self._pet_attenuation_correction),
            image_decay_corrected=None if not decay else decay == "true",
            image_decay_correction_time=_num(
                self._pet_image_decay_correction_time),
            recon_method_name=_txt(self._pet_recon_method),
            recon_method_parameter_labels=_csv(self._pet_recon_labels),
            recon_method_parameter_values=_csv_num(self._pet_recon_values),
            recon_method_parameter_units=_csv(self._pet_recon_units),
            recon_filter_type=_txt(self._pet_recon_filter_type),
            recon_filter_size=_num(self._pet_recon_filter_size),
            # Preserved: not exposed in this dialog.
            tracer_molecular_weight=self._spec.pet_defaults.tracer_molecular_weight,
            tracer_molecular_weight_units=self._spec.pet_defaults.tracer_molecular_weight_units,
            tracer_radlex=self._spec.pet_defaults.tracer_radlex,
            tracer_snomed=self._spec.pet_defaults.tracer_snomed,
            molar_activity=self._spec.pet_defaults.molar_activity,
            molar_activity_units=self._spec.pet_defaults.molar_activity_units,
            injected_volume=self._spec.pet_defaults.injected_volume,
            purity=self._spec.pet_defaults.purity,
            injection_end=self._spec.pet_defaults.injection_end,
            infusion_radioactivity=self._spec.pet_defaults.infusion_radioactivity,
            infusion_start=self._spec.pet_defaults.infusion_start,
            infusion_speed=self._spec.pet_defaults.infusion_speed,
            infusion_speed_units=self._spec.pet_defaults.infusion_speed_units,
            manufacturer=self._spec.pet_defaults.manufacturer,
            manufacturers_model_name=self._spec.pet_defaults.manufacturers_model_name,
        )

    @staticmethod
    def _set_combo(combo: QComboBox, value: Optional[str], blank: str) -> None:
        value = (value or "").strip()
        if not value:
            combo.setCurrentText(blank)
            return
        pos = combo.findText(value)
        if pos < 0:
            combo.addItem(value)
            pos = combo.findText(value)
        combo.setCurrentIndex(pos)

    @staticmethod
    def _combo_value(combo: QComboBox, blank: str) -> Optional[str]:
        text = combo.currentText()
        return None if text == blank else text

    def build_spec(self) -> RecordingMetaSpec:
        """Assemble a :class:`RecordingMetaSpec` from the current form state.

        Sections hidden for the scanned modality keep their loaded values
        (read from ``self._spec``) rather than reading the empty, never-shown
        widgets - so a MEG-only or MRI-only dataset never wipes EEG fields a
        shared scaffold may carry.
        """
        def _opt(edit: QLineEdit) -> Optional[str]:
            t = edit.text().strip()
            return t or None

        prev = self._spec.defaults

        # Institution / site is agnostic - its group is always shown, so read it
        # unconditionally.
        institution_name = _opt(self._institution_name)
        institution_dept = _opt(self._institution_dept)

        # Device block (modality-specific; all EEG/MEG).
        if self._device_applies:
            manufacturer = self._manufacturer.currentText().strip() or None
            amplifier_model = _opt(self._amplifier_model)
            software_versions = _opt(self._software_versions)
            lf = self._combo_value(self._line_freq, _BLANK)
            power_line_freq = float(lf) if lf else None
        else:
            manufacturer = prev.manufacturer
            amplifier_model = prev.amplifier_model
            software_versions = prev.software_versions
            power_line_freq = prev.power_line_freq

        # EEG montage & references block (EEG/iEEG only).
        if self._eeg_applies:
            eeg_ref = _opt(self._eeg_reference)
            eeg_gnd = _opt(self._eeg_ground)
            cap = self._cap_manufacturer.currentText().strip() or None
            montage = self._combo_value(self._montage, _NONE)
        else:
            eeg_ref = prev.eeg_reference
            eeg_gnd = prev.eeg_ground
            cap = prev.cap_manufacturer
            montage = prev.montage

        # MEG-specific block (MEG only) - manual fields mne-bids cannot derive.
        if self._meg_applies:
            dewar_position = self._dewar_position.currentText().strip() or None
            associated_empty_room = _opt(self._associated_empty_room)
            subject_artefact_description = _opt(self._subject_artefact_description)
        else:
            dewar_position = prev.dewar_position
            associated_empty_room = prev.associated_empty_room
            subject_artefact_description = prev.subject_artefact_description

        acq = AcquisitionSpec(
            manufacturer=manufacturer,
            amplifier_model=amplifier_model,
            software_versions=software_versions,
            cap_manufacturer=cap,
            institution_name=institution_name,
            institution_dept=institution_dept,
            eeg_reference=eeg_ref,
            eeg_ground=eeg_gnd,
            montage=montage,
            power_line_freq=power_line_freq,
            dewar_position=dewar_position,
            associated_empty_room=associated_empty_room,
            subject_artefact_description=subject_artefact_description,
            # Preserve fields/blocks this dialog does not edit (legacy `software`,
            # aux channels, filters, extras, cap_model).
            software=self._spec.defaults.software,
            cap_model=self._spec.defaults.cap_model,
            aux_channels=self._spec.defaults.aux_channels,
            filters=self._spec.defaults.filters,
            extras=self._spec.defaults.extras,
        )

        event_map: dict[str, str] = {}
        for r in range(self._events.rowCount()):
            code_item = self._events.item(r, 0)
            label_item = self._events.item(r, 1)
            code = code_item.text().strip() if code_item else ""
            label = label_item.text().strip() if label_item else ""
            if code:
                event_map[code] = label

        event_maps = dict(self._spec.event_maps)
        if event_map:
            event_maps["*"] = event_map
        else:
            event_maps.pop("*", None)

        phenotype_files = [
            self._phenotype.item(i).text() for i in range(self._phenotype.count())
        ]

        return self._spec.model_copy(update={
            "defaults": acq,
            "pet_defaults": self._build_pet_spec(),
            "event_maps": event_maps,
            "phenotype_files": phenotype_files,
            "participants_file": self._participants_file.text().strip(),
        })

    def _on_save(self) -> None:
        spec = self.build_spec()
        self._read_dataset_description(spec)
        self._read_device_groups(spec)
        self._scaffold_path.parent.mkdir(parents=True, exist_ok=True)
        self._scaffold_path.write_text(dump_spec(spec), encoding="utf-8")
        self.accept()


__all__ = ["RecordingMetaDialog"]
