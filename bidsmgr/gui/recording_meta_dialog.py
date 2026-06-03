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

from ..recording_meta import AcquisitionSpec, RecordingMetaSpec, dump_spec, load_spec
from .delegates import builtin_montages

_NONE = "(none)"
_BLANK = "(blank)"

# Common EEG + MEG amplifier/system manufacturers, offered as an editable
# dropdown (the user can still type any other value). MEG names follow the BIDS
# Manufacturer convention; EEG names are the common vendors.
_MANUFACTURERS: tuple[str, ...] = (
    # EEG
    "Brain Products", "BioSemi", "EGI / Philips Neuro", "ANT Neuro",
    "Compumedics Neuroscan", "g.tec", "Cognionics", "mBrainTrain", "OpenBCI",
    # MEG (BIDS convention)
    "MEGIN / Elekta / Neuromag", "CTF", "4D Neuroimaging / BTi",
    "KIT / Yokogawa", "ITAB", "KRISS", "FieldLine", "QuSpin",
)


class RecordingMetaDialog(QDialog):
    """Edit the dataset-level recording-metadata scaffold."""

    def __init__(
        self,
        scaffold_path: Path,
        present_datatypes: Optional[set[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset metadata")
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
        # Modality-specific (non-agnostic) -> the datatype sidecar, split into a
        # generic device/site block (all EEG/MEG) and an EEG-only montage block.
        self._device_box = self._build_device_group()
        self._eeg_box = self._build_eeg_group()
        bl.addWidget(self._device_box)
        bl.addWidget(self._eeg_box)
        # Modality-agnostic -> their own destination files.
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

    def _apply_modality_constraints(self) -> None:
        """Show/hide modality-specific sections by what was scanned.

        Both non-agnostic sections are hidden when the dataset has no
        EEG/MEG/iEEG/NIRS. The EEG montage/reference section is additionally
        hidden for a dataset with no scalp-EEG (e.g. MEG-only).
        """
        has_eeg_meg = bool(self._present & {"eeg", "meg", "ieeg", "nirs"})
        has_eeg = bool(self._present & {"eeg", "ieeg"})
        # Flags consulted by build_spec so a hidden section's loaded values are
        # preserved (not clobbered by the empty, never-shown widgets).
        self._device_applies = has_eeg_meg
        self._eeg_applies = has_eeg
        self._device_box.setVisible(has_eeg_meg)
        self._eeg_box.setVisible(has_eeg)

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def _build_device_group(self) -> QGroupBox:
        # Generic recording sidecar fields - apply to every EEG/MEG datatype's
        # sidecar (Manufacturer / InstitutionName etc. are not scalp-specific).
        box = QGroupBox(
            "Device & institution  ·  modality-specific  →  sub-..._<datatype>.json"
        )
        form = QFormLayout(box)

        self._manufacturer = QComboBox()
        self._manufacturer.setEditable(True)  # pick a default OR type another
        self._manufacturer.addItem("")
        self._manufacturer.addItems(_MANUFACTURERS)
        self._amplifier_model = QLineEdit()
        self._software_versions = QLineEdit()
        self._institution_name = QLineEdit()
        self._institution_dept = QLineEdit()
        self._line_freq = QComboBox()
        self._line_freq.addItems([_BLANK, "50", "60"])

        form.addRow("Manufacturer:", self._manufacturer)
        form.addRow("Amplifier / system model:", self._amplifier_model)
        form.addRow("Software versions:", self._software_versions)
        form.addRow("Institution name:", self._institution_name)
        form.addRow("Institution department:", self._institution_dept)
        form.addRow("Default line frequency (Hz):", self._line_freq)
        return box

    def _build_eeg_group(self) -> QGroupBox:
        # Scalp-EEG / iEEG only sidecar fields.
        box = QGroupBox(
            "EEG montage & references  ·  EEG / iEEG only  →  sub-..._eeg.json + electrodes.tsv"
        )
        form = QFormLayout(box)

        self._cap_manufacturer = QLineEdit()
        self._eeg_reference = QLineEdit()
        self._eeg_ground = QLineEdit()
        self._montage = QComboBox()
        self._montage.addItem(_NONE)
        self._montage.addItems(builtin_montages())

        form.addRow("Default EEG reference:", self._eeg_reference)
        form.addRow("Default EEG ground:", self._eeg_ground)
        form.addRow("Default montage:", self._montage)
        form.addRow("Cap manufacturer:", self._cap_manufacturer)
        return box

    def _build_event_group(self) -> QGroupBox:
        box = QGroupBox("Events (trigger code -> label)  ·  modality-agnostic  →  events.tsv")
        v = QVBoxLayout(box)
        self._events = QTableWidget(0, 2)
        self._events.setHorizontalHeaderLabels(["Code", "Label"])
        self._events.horizontalHeader().setStretchLastSection(True)
        # Bound the table height so it does not greedily expand: the whole
        # dialog then scrolls as one (rather than only this table scrolling).
        self._events.setMinimumHeight(120)
        self._events.setMaximumHeight(200)
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
        v.addWidget(QLabel(
            "Participant-level measure tables (TSV/CSV/XLSX/ODS keyed by "
            "participant_id). Each becomes phenotype/<measure>.tsv + .json."
        ))
        self._phenotype = QListWidget()
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
        self._cap_manufacturer.setText(acq.cap_manufacturer or "")
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

        # Device & institution block (generic sidecar; all EEG/MEG).
        if self._device_applies:
            manufacturer = self._manufacturer.currentText().strip() or None
            amplifier_model = _opt(self._amplifier_model)
            software_versions = _opt(self._software_versions)
            institution_name = _opt(self._institution_name)
            institution_dept = _opt(self._institution_dept)
            lf = self._combo_value(self._line_freq, _BLANK)
            power_line_freq = float(lf) if lf else None
        else:
            manufacturer = prev.manufacturer
            amplifier_model = prev.amplifier_model
            software_versions = prev.software_versions
            institution_name = prev.institution_name
            institution_dept = prev.institution_dept
            power_line_freq = prev.power_line_freq

        # EEG montage & references block (EEG/iEEG only).
        if self._eeg_applies:
            eeg_ref = _opt(self._eeg_reference)
            eeg_gnd = _opt(self._eeg_ground)
            cap = _opt(self._cap_manufacturer)
            montage = self._combo_value(self._montage, _NONE)
        else:
            eeg_ref = prev.eeg_reference
            eeg_gnd = prev.eeg_ground
            cap = prev.cap_manufacturer
            montage = prev.montage

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
            # Preserve blocks this dialog does not edit (aux channels, etc.).
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
