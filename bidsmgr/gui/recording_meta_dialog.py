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

from ..recording_meta import (
    COMMON_CAP_MANUFACTURERS,
    COMMON_MANUFACTURERS,
    AcquisitionSpec,
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
_MODALITY_NAMES = {"eeg": "EEG", "meg": "MEG", "ieeg": "iEEG", "nirs": "NIRS"}
_MODALITY_ORDER = ("eeg", "meg", "ieeg", "nirs")


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


class RecordingMetaDialog(QDialog):
    """Edit the dataset-level recording-metadata scaffold."""

    def __init__(
        self,
        scaffold_path: Path,
        present_datatypes: Optional[set[str]] = None,
        parent: Optional[QWidget] = None,
        montage_suggestions: Optional[list[str]] = None,
        manufacturer_suggestions: Optional[list[str]] = None,
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
        self._device_box = self._build_device_group()
        self._eeg_box = self._build_eeg_group()
        self._meg_box = self._build_meg_group()
        bl.addWidget(self._device_box)
        bl.addWidget(self._eeg_box)
        bl.addWidget(self._meg_box)
        # Region 2 - MODALITY-AGNOSTIC (apply to any dataset, incl. MRI):
        # institution (site info, written to every modality's sidecar), events
        # and phenotype, each writing to its own destination.
        bl.addWidget(self._region_label("Modality-agnostic", agnostic=True))
        bl.addWidget(self._build_institution_group())
        bl.addWidget(self._build_event_group())
        bl.addWidget(self._build_phenotype_group())
        bl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._populate()
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

        Both non-agnostic sections are hidden when the dataset has no
        EEG/MEG/iEEG/NIRS. The EEG montage/reference section is additionally
        hidden for a dataset with no scalp-EEG (e.g. MEG-only). The
        modality-specific region header hides when neither block applies.
        """
        has_eeg_meg = bool(self._present & {"eeg", "meg", "ieeg", "nirs"})
        has_eeg = bool(self._present & {"eeg", "ieeg"})
        has_meg = "meg" in self._present
        # Flags consulted by build_spec so a hidden section's loaded values are
        # preserved (not clobbered by the empty, never-shown widgets).
        self._device_applies = has_eeg_meg
        self._eeg_applies = has_eeg
        self._meg_applies = has_meg
        self._device_box.setVisible(has_eeg_meg)
        self._eeg_box.setVisible(has_eeg)
        self._meg_box.setVisible(has_meg)
        self._specific_region.setVisible(has_eeg_meg)

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

    def _build_device_group(self) -> QGroupBox:
        # Acquisition-system fields - the DEVICE that produced the recording.
        # These are modality-specific (the EEG amplifier and the MEG system are
        # different devices). Institution is agnostic and lives in its own
        # section. The title names every present modality (e.g. "EEG and MEG").
        mods = self._modalities_label({"eeg", "meg", "ieeg", "nirs"}) or "EEG / MEG"
        box = QGroupBox(
            f"Acquisition system  ·  {mods}  →  sub-..._<datatype>.json"
        )
        form = QFormLayout(box)
        _tighten_form(form)

        self._manufacturer = QComboBox()
        self._manufacturer.setEditable(True)  # pick a default OR type another
        self._manufacturer.addItem("")
        self._manufacturer.addItems(COMMON_MANUFACTURERS)
        self._amplifier_model = QLineEdit()
        self._software_versions = QLineEdit()
        self._line_freq = QComboBox()
        self._line_freq.addItems([_BLANK, "50", "60"])

        self._form_row(form, "Manufacturer:", self._manufacturer, "manufacturer")
        # Read-only summary of the manufacturers the scan detected/inferred.
        hint = self._suggestion_label(self._manufacturer_suggestions)
        if hint is not None:
            form.addRow("", hint)
        self._form_row(form, "Amplifier / system model:", self._amplifier_model, "amplifier_model")
        self._form_row(form, "Software versions:", self._software_versions, "software_versions")
        self._form_row(form, "Power line frequency (Hz):", self._line_freq, "line_freq")
        return box

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
            "event_maps": event_maps,
            "phenotype_files": phenotype_files,
        })

    def _on_save(self) -> None:
        spec = self.build_spec()
        self._scaffold_path.parent.mkdir(parents=True, exist_ok=True)
        self._scaffold_path.write_text(dump_spec(spec), encoding="utf-8")
        self.accept()


__all__ = ["RecordingMetaDialog"]
