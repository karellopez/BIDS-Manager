"""Properties panel — col 4 of the Converter view.

Schema-driven editor for the selected row's BIDS entities. Reference:
``inspector_proto/proto.py`` lines 669-722.

Layout (top → bottom):

1. **datatype / suffix combos** — required (* in red). Editing either
   re-derives the entity allow-list below.
2. **Entities form** — one row per entity allowed by the schema for
   the current ``(datatype, suffix)``. Required entities marked with
   ``*``; optional entities labelled ``opt``. Each field is a
   ``QLineEdit`` whose ``editingFinished`` writes back through
   :meth:`InventoryTableModel.set_entity`.
3. **Predicted path preview** — token-coloured monospace string built
   from ``schema.build_relative_path``.
4. **Validation messages** — one ``ValMessage`` per
   :class:`schema.ValidationVerdict` from
   :func:`schema.validate_entity_set`.
5. **Why this name?** — small provenance section reading from the
   project's ``ProvenanceMap`` when one exists.

The panel never owns state — it reads from the model on every
``set_selected_row`` call and writes through the model's API. This
keeps the data flow unidirectional (model → panel → model) and means
the panel auto-refreshes when the model emits ``dataChanged``.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import pandas as pd

from .. import schema as schema_mod
from ..project import Project
from ..recording_meta import COMMON_CAP_MANUFACTURERS, COMMON_MANUFACTURERS
from .delegates import builtin_montages
from .metadata_help import tooltip_for
from .models import InventoryTableModel
from .theme_manager import CUR, scaled_px
from .widgets import PaneHeader, ValMessage

# Datatypes that carry recording-metadata (the per-row section appears only
# for these). MEG has no scalp montage / reference / ground concept.
_EEG_MEG_DATATYPES = frozenset({"eeg", "meg", "ieeg", "nirs"})

# Human display names for the datatypes that carry a recording sidecar.
_MODALITY_NAMES = {"eeg": "EEG", "meg": "MEG", "ieeg": "iEEG", "nirs": "NIRS"}


def _modality_label(datatype: str) -> str:
    """Display name for a single datatype (``eeg`` -> ``EEG``)."""
    return _MODALITY_NAMES.get(datatype, datatype.upper())


log = logging.getLogger(__name__)


# Stable display order for entities (matches BIDS spec). Entities not
# listed here are appended after these, schema-defined order.
_ENTITY_DISPLAY_ORDER: tuple[str, ...] = (
    "subject", "session", "task", "acquisition", "ceagent",
    "reconstruction", "direction", "run", "echo", "part", "chunk",
)


class _EntityRow(QWidget):
    """One ``[label] [QLineEdit]`` row for an entity.

    Self-contained so the form layout can drop in / pull out rows
    without leaking state. Emits no signal — the parent panel reads
    the value back when committing.
    """

    def __init__(
        self,
        entity_name: str,
        value: str,
        *,
        required: bool,
        deprecated: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.entity_name = entity_name
        self.setStyleSheet("background: transparent;")

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        label_text = entity_name
        lbl = QLabel(label_text)
        lbl.setMinimumWidth(76)
        lbl.setMaximumWidth(76)
        pal = CUR()
        css_color = pal["text"] if required else pal["dim"]
        suffix = f' <span style="color:{pal["error"]}">*</span>' if required else ""
        if deprecated:
            css_color = pal["muted"]
            lbl.setText(f'<span style="color:{css_color};text-decoration:line-through;">{label_text}</span>')
        else:
            lbl.setText(f'<span style="color:{css_color}">{label_text}</span>{suffix}')
        lbl.setTextFormat(Qt.TextFormat.RichText)
        h.addWidget(lbl)

        self.edit = QLineEdit(value)
        self.edit.setObjectName("ent-input")
        self.edit.setPlaceholderText("—")
        h.addWidget(self.edit, 1)

    def value(self) -> str:
        return self.edit.text().strip()


class PropertiesPanel(QWidget):
    """Right pane of the Converter view.

    Bind a model with :meth:`bind_model`, then call
    :meth:`set_selected_row` whenever the table's selection changes.
    Passing ``row=None`` blanks the form.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._model: Optional[InventoryTableModel] = None
        self._project: Optional[Project] = None
        self._row: Optional[int] = None
        self._suppress_writeback = False
        self._entity_rows: list[_EntityRow] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(PaneHeader("Properties · no row selected"))
        self._header = outer.itemAt(0).widget()

        # Body lives inside a scroll area so the form can grow without
        # bumping the splitter handles. The scroll area's contents are
        # ``self._body`` which we rebuild on every set_selected_row.
        self._body = QWidget()
        self._body.setObjectName("props-panel")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(14, 12, 14, 12)
        self._body_layout.setSpacing(7)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._body)
        outer.addWidget(scroll, 1)

        # Build the initial empty body (just a hint).
        self._render_empty()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind_model(self, model: Optional[InventoryTableModel]) -> None:
        """Attach (or detach with ``None``) the inventory model.

        Connects ``dataChanged`` so external edits to the active row
        (e.g. the table's mirror cells) reflow into the form.
        """
        if self._model is not None:
            try:
                self._model.dataChanged.disconnect(self._on_model_data_changed)
            except (TypeError, RuntimeError):
                pass
        self._model = model
        if model is not None:
            model.dataChanged.connect(self._on_model_data_changed)
        self.set_selected_row(None)

    def set_project(self, project: Optional[Project]) -> None:
        """Attach a project for provenance lookups. Not used yet for writes."""
        self._project = project

    def repaint_for_palette(self, _pal: dict) -> None:
        """Rebuild the body so inline palette reads pick up new colors.

        The form is constructed from ``CUR()`` reads at render time
        (label colors, predicted-path tokens, validation borders), so a
        full re-render is the cleanest way to refresh after a theme
        swap. Re-renders the currently-selected row (or the empty hint
        if nothing is selected).
        """
        self.set_selected_row(self._row)

    def set_selected_row(self, row: Optional[int]) -> None:
        """Render the form for ``row`` (or blank if ``None``).

        Called by the Converter view whenever the table selection
        changes. Cheap enough to call on every selection event — the
        body is rebuilt from scratch each time so we don't have to
        track per-row deltas.
        """
        self._row = row
        if self._model is None or row is None:
            self._render_empty()
            self._set_header("Properties · no row selected")
            return
        if not (0 <= row < self._model.rowCount()):
            self._render_empty()
            return
        self._set_header(f"Properties · row {row + 1} selected")
        self._render_for_row(row)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _clear_body(self) -> None:
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._entity_rows = []

    def _render_empty(self) -> None:
        self._clear_body()
        hint = QLabel(
            "Select a row in the inspection table to edit its entities."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {CUR()['dim']}; padding: 24px 0;")
        hint.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body_layout.addWidget(hint)
        self._body_layout.addStretch(1)

    def _render_for_row(self, row: int) -> None:
        assert self._model is not None
        self._clear_body()

        # 1. datatype + suffix combos
        datatype, suffix = self._model.datatype_suffix(row)
        self._body_layout.addWidget(self._build_combo_row(
            "datatype", datatype, options=sorted(schema_mod.list_datatypes()),
            required=True, slot=self._on_datatype_changed,
        ))
        suffix_options = sorted(schema_mod.list_suffixes(datatype)) if datatype else []
        self._body_layout.addWidget(self._build_combo_row(
            "suffix", suffix, options=suffix_options,
            required=True, slot=self._on_suffix_changed,
        ))

        self._body_layout.addSpacing(4)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addSpacing(2)

        # 2. Entities form
        ent_title = QLabel("Entities")
        ent_title.setStyleSheet(f"color: {CUR()['text']}; font-weight: 600;")
        sub = QLabel("  (schema-driven)")
        sub.setStyleSheet(f"color: {CUR()['dim']}; font-size: {scaled_px(10)}px;")
        head = QHBoxLayout()
        head.setSpacing(0)
        head.addWidget(ent_title)
        head.addWidget(sub)
        head.addStretch(1)
        head_wrap = QWidget()
        head_wrap.setLayout(head)
        head_wrap.setStyleSheet("background: transparent;")
        self._body_layout.addWidget(head_wrap)
        self._body_layout.addSpacing(2)

        entities = self._model.entities(row)
        ordered = self._ordered_entities(datatype, suffix)
        required_set = set(schema_mod.required_entities(datatype, suffix)) if datatype and suffix else set()
        deprecated_set = set(schema_mod.deprecated_entities(datatype, suffix)) if datatype and suffix else set()

        for entity in ordered:
            value = entities.get(entity, "")
            er = _EntityRow(
                entity,
                value,
                required=entity in required_set,
                deprecated=entity in deprecated_set,
            )
            er.edit.editingFinished.connect(
                lambda e=er: self._on_entity_committed(e.entity_name, e.value())
            )
            self._entity_rows.append(er)
            self._body_layout.addWidget(er)

        self._body_layout.addSpacing(6)
        self._body_layout.addWidget(self._divider())

        # 3. Predicted path preview
        sec = QLabel("PREDICTED PATH")
        sec.setStyleSheet(f"color: {CUR()['dim']}; font-size: {scaled_px(10)}px; font-weight: 600;")
        self._body_layout.addWidget(sec)
        self._body_layout.addWidget(self._build_path_preview(row, datatype, suffix, entities))

        # 4. Row-state notice (from the scanner's proposed_issues) +
        # schema validation. Two distinct sources of "what's wrong with
        # this row": scanner-detected operational issues vs. schema's
        # entity-set verdicts. Both render with the same ValMessage
        # widget so the user sees them as one ranked list.
        self._body_layout.addSpacing(8)
        for vmsg in self._build_row_issue_messages(row):
            self._body_layout.addWidget(vmsg)
        for vmsg in self._build_validation_messages(datatype, suffix, entities):
            self._body_layout.addWidget(vmsg)

        # 5. Per-row metadata, split into two clearly-separated regions:
        # modality-agnostic (participant demographics + companion files, written
        # for any modality incl. MRI) and modality-specific (the recording
        # sidecar, EEG/MEG/iEEG/NIRS only). Sections within each region are
        # labelled by their BIDS destination file.
        self._append_metadata_title()
        self._append_region_label("Modality-agnostic", agnostic=True)
        self._append_participant_section(row)
        self._append_companion_section(row)
        if datatype in _EEG_MEG_DATATYPES:
            self._append_region_label("Modality-specific", agnostic=False)
            self._append_recording_section(row, datatype)

        self._body_layout.addStretch(1)

    def _build_combo_row(self, label_text: str, value: str, *, options: list[str],
                         required: bool, slot) -> QWidget:
        row = QWidget()
        # Scoped so the transparent bg does not cascade into the combo popup.
        row.setObjectName("meta-row")
        row.setStyleSheet("#meta-row { background: transparent; }")
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        pal = CUR()
        lbl = QLabel(label_text)
        lbl.setMinimumWidth(76)
        lbl.setMaximumWidth(76)
        suffix = f' <span style="color:{pal["error"]}">*</span>' if required else ""
        lbl.setText(f'<span style="color:{pal["text"]}">{label_text}</span>{suffix}')
        lbl.setTextFormat(Qt.TextFormat.RichText)
        h.addWidget(lbl)

        combo = QComboBox()
        combo.setObjectName("ent-input")
        combo.setMinimumHeight(22)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        if options:
            combo.addItems(options)
        if value and value not in options:
            combo.addItem(value)
        if value:
            combo.setCurrentText(value)
        # Use ``activated`` (not ``currentTextChanged``) so we only fire on
        # user interaction, not on programmatic ``setCurrentText`` during
        # render.
        combo.activated.connect(lambda _i, c=combo: slot(c.currentText()))
        h.addWidget(combo, 1)
        return row

    # Source recording extensions, longest-first so ``.fif.gz`` wins over ``.fif``.
    _RECORDING_EXTS: tuple[str, ...] = (
        ".fif.gz", ".vhdr", ".edf", ".bdf", ".gdf", ".set", ".cnt", ".fif",
        ".con", ".sqd", ".ds", ".mff", ".snirf", ".nwb", ".mef",
    )
    # Fallback BIDS extension per non-MRI datatype when the source path is blank.
    _DATATYPE_DEFAULT_EXT: dict[str, str] = {
        "meg": ".fif", "eeg": ".edf", "ieeg": ".edf", "nirs": ".snirf",
    }

    def _preview_extension(self, row: int, datatype: str, suffix: str) -> str:
        """Pick the extension shown in the predicted-path preview.

        MRI datatypes render ``.nii.gz``; physio renders ``.tsv.gz``; EEG / MEG /
        iEEG / NIRS render the source recording's own extension (what mne-bids
        keeps by default) with a per-datatype fallback. Fixes the preview always
        showing ``.nii.gz`` for electrophysiology rows.
        """
        if suffix == "physio":
            return ".tsv.gz"
        if datatype in ("eeg", "ieeg", "meg", "nirs"):
            src = self._cell(row, "source_file").lower()
            for ext in self._RECORDING_EXTS:
                if src.endswith(ext):
                    return ext
            return self._DATATYPE_DEFAULT_EXT.get(datatype, "")
        return ".nii.gz"

    def _build_path_preview(
        self, row: int, datatype: str, suffix: str, entities: dict[str, str],
    ) -> QWidget:
        f = QFrame()
        f.setObjectName("path-preview")
        l = QVBoxLayout(f)
        l.setContentsMargins(11, 9, 11, 9)
        l.setSpacing(0)
        pal = CUR()

        # Compose the same token list the prototype rendered, derived
        # from the schema instead of hard-coded data.
        pieces: list[str] = []
        if datatype and suffix and entities.get("subject"):
            try:
                rel = str(schema_mod.build_relative_path(
                    entities, datatype, suffix,
                    self._preview_extension(row, datatype, suffix),
                ))
                # Insert a newline between the directory part and the
                # basename so the preview wraps clearly.
                head, _slash, base = rel.rpartition("/")
                pieces.append(self._color_path_segment(head + "/", pal["accent"]))
                pieces.append("<br>")
                # Basename split: tokens are ``key-value`` pairs joined
                # by underscores, ending in the suffix.
                tokens = base.split("_")
                last_index = len(tokens) - 1
                for i, tok in enumerate(tokens):
                    if i == last_index:
                        # split off extension
                        if "." in tok:
                            suf, _, ext = tok.partition(".")
                            pieces.append(f'<span style="color:{pal["teal"]}">{suf}</span>')
                            pieces.append(f'<span style="color:{pal["dim"]}">.{ext}</span>')
                        else:
                            pieces.append(f'<span style="color:{pal["teal"]}">{tok}</span>')
                    else:
                        if "-" in tok:
                            pieces.append(f'<span style="color:{pal["purple"]}">{tok}</span>')
                        else:
                            pieces.append(f'<span style="color:{pal["accent"]}">{tok}</span>')
                    if i < last_index:
                        pieces.append("_")
            except (ValueError, KeyError, TypeError) as exc:
                pieces.append(f'<span style="color:{pal["error"]}">cannot build path: {exc}</span>')
        else:
            pieces.append(
                f'<span style="color:{pal["muted"]}">'
                'Pick datatype + suffix to preview the path.'
                '</span>'
            )

        lbl = QLabel("".join(pieces))
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            'font-family: "SF Mono","Menlo","Monaco",monospace; '
            f'font-size: {scaled_px(11)}px; color: {pal["text"]}; '
            'background: transparent;'
        )
        l.addWidget(lbl)
        return f

    @staticmethod
    def _color_path_segment(text: str, color: str) -> str:
        return f'<span style="color:{color}">{text}</span>'

    def _build_row_issue_messages(self, row: int) -> list[QWidget]:
        """One ``ValMessage`` per scanner-detected issue on the selected row.

        Severity is derived from the model's ``row_state``: ``err`` →
        red badge, ``warn`` → amber, ``skip`` → muted (we surface it
        as ``warn`` here so the user sees the explanation). Returns an
        empty list when the row has no issues.
        """
        if self._model is None:
            return []
        issues = self._model.row_issues(row)
        if not issues:
            return []
        state = self._model.row_state(row)
        sev = {"err": "err", "warn": "warn", "skip": "warn"}.get(state, "warn")

        msgs: list[QWidget] = []
        for i, text in enumerate(issues):
            # First entry carries the rule_id "SCANNER · <row-state>";
            # follow-up entries are continuations of the same row so
            # the label is just blank to keep the column quiet.
            rule = f"SCANNER · {state}" if i == 0 else ""
            msgs.append(ValMessage(sev, rule, text, None))
        return msgs

    def _build_validation_messages(
        self, datatype: str, suffix: str, entities: dict[str, str],
    ) -> list[QWidget]:
        out: list[QWidget] = []
        if not datatype or not suffix:
            out.append(ValMessage(
                "warn", "SCHEMA",
                "Pick a datatype and suffix to validate the entity set.",
                None,
            ))
            return out

        verdicts = schema_mod.validate_entity_set(entities, datatype, suffix)
        if not verdicts:
            out.append(ValMessage(
                "ok", f"SCHEMA · {datatype}/{suffix}",
                "Entity set is valid.",
                None,
            ))
            return out
        for v in verdicts:
            sev = {
                schema_mod.Severity.ERROR: "err",
                schema_mod.Severity.WARNING: "warn",
                schema_mod.Severity.INFO: "ok",
            }.get(v.severity, "warn")
            out.append(ValMessage(sev, v.rule_id, v.message, None))
        return out

    @staticmethod
    def _ordered_entities(datatype: str, suffix: str) -> list[str]:
        """Display order: BIDS-spec order, fallback to schema-allow order."""
        if not datatype or not suffix:
            return list(_ENTITY_DISPLAY_ORDER)
        allowed = schema_mod.allowed_entities(datatype, suffix)
        # Put canonical order first (those that are in allowed), then any
        # remaining allowed entities in schema order.
        head = [e for e in _ENTITY_DISPLAY_ORDER if e in allowed]
        tail = [e for e in allowed if e not in head]
        return head + tail

    def _set_header(self, text: str) -> None:
        # The PaneHeader uppercases its constructor arg; we replace its
        # text directly to avoid restoring case rules.
        self._header.setText(text.upper())

    @staticmethod
    def _divider() -> QFrame:
        d = QFrame()
        d.setStyleSheet(
            f"background: {CUR()['subtle']}; max-height: 1px; "
            f"min-height: 1px; border: none;"
        )
        return d

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_entity_committed(self, entity: str, value: str) -> None:
        if self._suppress_writeback:
            return
        if self._model is None or self._row is None:
            return
        # Translate the schema entity name (``subject``) — already the
        # canonical form used by the schema engine and ProjectState.
        self._model.set_entity(self._row, entity, value or None)

    def _on_datatype_changed(self, new_value: str) -> None:
        if self._model is None or self._row is None:
            return
        _dt, suffix = self._model.datatype_suffix(self._row)
        self._model.set_datatype_suffix(self._row, new_value, suffix)
        # Re-render: the suffix combo's options change with datatype.
        self.set_selected_row(self._row)

    def _on_suffix_changed(self, new_value: str) -> None:
        if self._model is None or self._row is None:
            return
        datatype, _sf = self._model.datatype_suffix(self._row)
        self._model.set_datatype_suffix(self._row, datatype, new_value)
        self.set_selected_row(self._row)

    # ------------------------------------------------------------------
    # Per-row metadata, grouped by destination file
    # ------------------------------------------------------------------

    def _region_label(self, text: str, *, agnostic: bool) -> QWidget:
        """A bold, colour-coded region divider separating the two metadata
        regions (modality-agnostic vs modality-specific)."""
        pal = CUR()
        color = pal["teal"] if agnostic else pal["purple"]
        lbl = QLabel(
            f'<span style="color:{color};font-weight:800;'
            f'letter-spacing:0.6px;">{text.upper()}</span>'
        )
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setStyleSheet(
            f"font-size: {scaled_px(10)}px; background: transparent; "
            f"border-bottom: 1px solid {color}; padding-bottom: 2px;"
        )
        return lbl

    def _append_region_label(self, text: str, *, agnostic: bool) -> None:
        self._body_layout.addSpacing(10)
        self._body_layout.addWidget(self._region_label(text, agnostic=agnostic))

    def _append_metadata_title(self) -> None:
        """Heading that frames the per-row metadata block as the minimal,
        BIDS-recommended set (everything below is optional and inherits the
        dataset defaults)."""
        pal = CUR()
        self._body_layout.addSpacing(12)
        self._body_layout.addWidget(self._divider())
        title = QLabel("MINIMAL METADATA")
        title.setStyleSheet(
            f"color: {pal['text']}; font-weight: 800; letter-spacing: 0.6px; "
            f"font-size: {scaled_px(11)}px; background: transparent;"
        )
        self._body_layout.addWidget(title)
        sub = QLabel(
            "Essential BIDS fields for this recording. All optional; blank "
            "fields inherit the dataset defaults set in Dataset metadata."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(
            f"color: {pal['dim']}; font-size: {scaled_px(9)}px; background: transparent;"
        )
        self._body_layout.addWidget(sub)

    def _section_header(
        self, title: str, destination: str, *, agnostic: bool, tag: str,
    ) -> QWidget:
        """A colour-coded section title plus a dim ``<tag> -> <destination>`` note.

        ``agnostic`` colours the title; ``tag`` states which modalities the
        section applies to (``any modality`` for agnostic sections, or the
        recording's modality such as ``EEG``) so the destination is unambiguous.
        """
        pal = CUR()
        color = pal["teal"] if agnostic else pal["purple"]
        lbl = QLabel(
            f'<span style="color:{color};font-weight:700;">{title}</span>'
            f'<span style="color:{pal["dim"]};"> &middot; {tag} &rarr; {destination}</span>'
        )
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setStyleSheet(f"font-size: {scaled_px(10)}px; background: transparent;")
        return lbl

    def _append_participant_section(self, row: int) -> None:
        """Demographics for ANY row (incl. MRI) -> participants.tsv.

        Modality-agnostic: every subject has a participants.tsv row. Handedness
        is user-entered (never auto-assumed).
        """
        self._body_layout.addSpacing(8)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addWidget(self._section_header(
            "PARTICIPANT", "participants.tsv", agnostic=True, tag="any modality"))
        self._body_layout.addWidget(self._meta_combo_row(
            "sex", "PatientSex", ["", "M", "F", "O"], self._cell(row, "PatientSex"), "",
        ))
        self._body_layout.addWidget(self._meta_edit_row(
            "age", "PatientAge", self._cell(row, "PatientAge"),
        ))
        self._body_layout.addWidget(self._meta_combo_row(
            "hand", "Handedness", ["", "R", "L", "A"], self._cell(row, "Handedness"), "",
        ))

    def _append_recording_section(self, row: int, datatype: str) -> None:
        """EEG/MEG/iEEG/NIRS recording-sidecar fields -> sub-..._<datatype>.json.

        Up to three BIDS-aligned sub-sections: an Acquisition block (device +
        line frequency - shared by all electrophysiology sidecars), a Reference
        & montage block (EEG/iEEG only - MEG and NIRS have no scalp
        reference/ground/montage), and a MEG-acquisition block (MEG only).
        Institution is agnostic and lives in the Dataset-metadata dialog, NOT
        here. Inheritance fields show the EFFECTIVE value (per-row override, else
        the dataset default); writing the default clears the override.
        """
        mod = _modality_label(datatype)
        show_montage = datatype in ("eeg", "ieeg")
        show_ref_ground = datatype in ("eeg", "ieeg")
        show_cap = datatype == "eeg"

        # --- Acquisition: device + line frequency (the recording's hardware).
        # Institution is agnostic (set once for the dataset in Dataset metadata),
        # NOT here. Device overrides live in the scaffold's overrides[row_id]
        # (not a TSV column); each inherits the dataset default until set, and
        # clearing it (or matching the default) restores inheritance.
        self._body_layout.addSpacing(8)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addWidget(self._section_header(
            "ACQUISITION", f"sub-..._{datatype}.json", agnostic=False, tag=mod))
        lf = self._eff(row, "line_freq")
        lf = lf[:-2] if lf.endswith(".0") else lf
        self._body_layout.addWidget(self._meta_combo_row(
            "line_freq", "line_freq", ["(blank)", "50", "60"], lf, "(blank)",
        ))
        self._body_layout.addWidget(self._meta_combo_row(
            "manufacturer", "manufacturer",
            [""] + list(COMMON_MANUFACTURERS), self._acq_eff(row, "manufacturer"), "",
            setter=self._on_acq_field_changed, editable=True,
        ))
        # Scan-detected/inferred manufacturer, shown like the montage match (a
        # read-only suggestion; not auto-applied - mne-bids fills MEG itself).
        manuf_sugg = self._cell(row, "manufacturer_suggestion")
        if manuf_sugg:
            self._body_layout.addWidget(
                self._scan_hint("manufacturer", "scan detected", manuf_sugg))
        self._body_layout.addWidget(self._meta_edit_row(
            "model", "amplifier_model", self._acq_eff(row, "amplifier_model"),
            setter=self._on_acq_field_changed,
        ))
        self._body_layout.addWidget(self._meta_edit_row(
            "software", "software_versions", self._acq_eff(row, "software_versions"),
            setter=self._on_acq_field_changed,
        ))

        # --- Reference & montage: EEG/iEEG scalp fields only.
        if show_montage or show_ref_ground:
            self._body_layout.addSpacing(8)
            self._body_layout.addWidget(self._divider())
            self._body_layout.addWidget(self._section_header(
                "REFERENCE & MONTAGE",
                f"sub-..._{datatype}.json + electrodes.tsv",
                agnostic=False, tag=mod))
            if show_ref_ground:
                self._body_layout.addWidget(self._meta_edit_row(
                    "reference", "eeg_reference", self._eff(row, "eeg_reference"),
                ))
                self._body_layout.addWidget(self._meta_edit_row(
                    "ground", "eeg_ground", self._eff(row, "eeg_ground"),
                ))
            if show_montage:
                self._body_layout.addWidget(self._meta_combo_row(
                    "montage", "montage",
                    ["(none)"] + builtin_montages(), self._eff(row, "montage"), "(none)",
                ))
                # Surface the scan's best channel-name match as a read-only hint
                # so the user can pick the montage with confidence (the scan does
                # not auto-fill it).
                suggestion = self._cell(row, "montage_suggestion")
                if suggestion:
                    self._body_layout.addWidget(self._montage_hint(suggestion))
            if show_cap:
                self._body_layout.addWidget(self._meta_combo_row(
                    "cap", "cap_manufacturer",
                    [""] + list(COMMON_CAP_MANUFACTURERS),
                    self._acq_eff(row, "cap_manufacturer"), "",
                    setter=self._on_acq_field_changed, editable=True,
                ))

        # --- MEG-specific acquisition (MEG only). Only fields mne-bids cannot
        # derive from the recording: dewar position, the empty-room link, and
        # the artefact note. Channel-derived MEG facts (head localization,
        # digitized landmarks/head points, ...) are filled by mne-bids.
        if datatype == "meg":
            self._body_layout.addSpacing(8)
            self._body_layout.addWidget(self._divider())
            self._body_layout.addWidget(self._section_header(
                "MEG ACQUISITION", f"sub-..._{datatype}.json", agnostic=False, tag=mod))
            self._body_layout.addWidget(self._meta_combo_row(
                "dewar", "dewar_position", ["", "upright", "supine"],
                self._acq_eff(row, "dewar_position"), "",
                setter=self._on_acq_field_changed, editable=True,
            ))
            self._body_layout.addWidget(self._meta_edit_row(
                "empty room", "associated_empty_room",
                self._acq_eff(row, "associated_empty_room"),
                setter=self._on_acq_field_changed,
            ))
            self._body_layout.addWidget(self._meta_edit_row(
                "artefacts", "subject_artefact_description",
                self._acq_eff(row, "subject_artefact_description"),
                setter=self._on_acq_field_changed,
            ))

    def _montage_hint(self, suggestion: str) -> QWidget:
        return self._scan_hint("montage", "best montage match (from scan)", suggestion)

    def _scan_hint(self, field_key: str, prefix: str, value: str) -> QWidget:
        """A read-only scan-suggestion hint row (e.g. montage match, detected
        manufacturer). Aligned under the field column; not auto-applied. Carries
        an explicit QToolTip background so it never renders transparent."""
        pal = CUR()
        lbl = QLabel(
            f'<span style="color:{pal["dim"]};">{prefix}: </span>'
            f'<span style="color:{pal["teal"]};font-weight:700;">{value}</span>'
        )
        lbl.setObjectName("scan-hint")
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        # Scoped (objectName) so the transparent background does not leak into
        # this label's tooltip; the tooltip then uses the app QToolTip styling.
        lbl.setStyleSheet(
            f"#scan-hint {{ font-size: {scaled_px(10)}px; background: transparent; "
            "margin-left: 84px; }"
        )
        lbl.setToolTip(
            "Suggestion computed at scan; not applied automatically. Pick it in "
            "the field above if appropriate."
        )
        return lbl

    def _append_companion_section(self, row: int) -> None:
        """Link already-curated companion files (events/beh/stim/...) for a row.

        Modality-agnostic: any recording can carry curated sidecar companions
        the converter copies into the BIDS tree (place + name, no conversion).
        """
        self._body_layout.addSpacing(8)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addWidget(self._section_header(
            "COMPANION FILES", "events / beh / stim (copied into BIDS)",
            agnostic=True, tag="any modality"))

        self._companion_list = QListWidget()
        self._companion_list.setMaximumHeight(72)
        for suffix, path in self._companions(row):
            self._companion_list.addItem(f"{suffix}: {path}")
        self._body_layout.addWidget(self._companion_list)

        ctl = QWidget()
        ctl.setObjectName("companion-ctl")
        ctl.setStyleSheet("#companion-ctl { background: transparent; }")
        h = QHBoxLayout(ctl)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self._companion_suffix = QComboBox()
        self._companion_suffix.setObjectName("ent-input")  # opaque popup styling
        self._companion_suffix.addItems(
            ["events", "beh", "stim", "physio", "channels", "electrodes"]
        )
        link = QPushButton("Link file…")
        link.clicked.connect(lambda _=False, r=row: self._link_companion(r))
        rem = QPushButton("Remove")
        rem.clicked.connect(lambda _=False, r=row: self._remove_companion(r))
        h.addWidget(self._companion_suffix)
        h.addWidget(link)
        h.addWidget(rem)
        h.addStretch(1)
        self._body_layout.addWidget(ctl)

    def _companions(self, row: int) -> list[tuple[str, str]]:
        raw = self._cell(row, "companion_files")
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return []
        out: list[tuple[str, str]] = []
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict) and it.get("suffix") and it.get("path"):
                    out.append((str(it["suffix"]), str(it["path"])))
        return out

    def _write_companions(self, row: int, items: list[tuple[str, str]]) -> None:
        if self._model is None:
            return
        payload = (
            json.dumps([{"suffix": s, "path": p} for s, p in items]) if items else ""
        )
        # dataChanged from bulk_set re-renders this section with the new list.
        self._model.bulk_set([row], "companion_files", payload)

    def _link_companion(self, row: int) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Link a curated companion file", "", "All files (*)",
        )
        if not path:
            return
        items = self._companions(row)
        items.append((self._companion_suffix.currentText(), path))
        self._write_companions(row, items)

    def _remove_companion(self, row: int) -> None:
        sel = self._companion_list.currentRow()
        items = self._companions(row)
        if 0 <= sel < len(items):
            items.pop(sel)
            self._write_companions(row, items)

    def _meta_combo_row(self, label: str, key: str, options: list[str],
                        current: str, blank_label: str, *,
                        setter=None, editable: bool = False) -> QWidget:
        row_w = QWidget()
        # Scope the transparent background to THIS container only (objectName
        # selector) so it does not cascade into child combo popups / tooltips
        # and make them transparent. Those then use the app theme styling.
        row_w.setObjectName("meta-row")
        row_w.setStyleSheet("#meta-row { background: transparent; }")
        h = QHBoxLayout(row_w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        tip = tooltip_for(key)
        lbl = QLabel(label)
        lbl.setMinimumWidth(76)
        lbl.setMaximumWidth(76)
        lbl.setStyleSheet(f"color: {CUR()['dim']};")
        if tip:
            lbl.setToolTip(tip)
        h.addWidget(lbl)

        on_change = setter or self._on_meta_field_changed
        combo = QComboBox()
        combo.setObjectName("ent-input")
        if tip:
            combo.setToolTip(tip)
        combo.setEditable(editable)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        combo.addItems(options)
        cur = current.strip()
        if not cur:
            combo.setCurrentText(blank_label)
        else:
            if combo.findText(cur) < 0:
                combo.addItem(cur)
            combo.setCurrentText(cur)
        combo.activated.connect(
            lambda _i, c=combo, k=key, bl=blank_label:
            on_change(k, "" if c.currentText() == bl else c.currentText())
        )
        if editable:
            # An editable combo also commits a free-typed value on focus loss.
            combo.lineEdit().editingFinished.connect(
                lambda c=combo, k=key, bl=blank_label:
                on_change(k, "" if c.currentText() == bl else c.currentText().strip())
            )
        h.addWidget(combo, 1)
        return row_w

    def _meta_edit_row(self, label: str, key: str, current: str, *,
                       setter=None) -> QWidget:
        row_w = QWidget()
        # Scope the transparent background to THIS container only (objectName
        # selector) so it does not cascade into child combo popups / tooltips
        # and make them transparent. Those then use the app theme styling.
        row_w.setObjectName("meta-row")
        row_w.setStyleSheet("#meta-row { background: transparent; }")
        h = QHBoxLayout(row_w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        tip = tooltip_for(key)
        lbl = QLabel(label)
        lbl.setMinimumWidth(76)
        lbl.setMaximumWidth(76)
        lbl.setStyleSheet(f"color: {CUR()['dim']};")
        if tip:
            lbl.setToolTip(tip)
        h.addWidget(lbl)
        on_change = setter or self._on_meta_field_changed
        edit = QLineEdit(current)
        edit.setObjectName("ent-input")
        edit.setPlaceholderText("—")
        if tip:
            edit.setToolTip(tip)
        edit.editingFinished.connect(
            lambda e=edit, k=key: on_change(k, e.text().strip())
        )
        h.addWidget(edit, 1)
        return row_w

    def _eff(self, row: int, col: str) -> str:
        """Effective value (per-row override else inherited dataset default)."""
        if self._model is None:
            return ""
        return self._model.effective_value(row, col)

    def _acq_eff(self, row: int, field: str) -> str:
        """Effective per-row acquisition value (scaffold override else default)."""
        if self._model is None:
            return ""
        return self._model.acq_effective(row, field)

    def _cell(self, row: int, col: str) -> str:
        if self._model is None:
            return ""
        df = self._model.dataframe()
        if col not in df.columns or not (0 <= row < len(df)):
            return ""
        v = df.iloc[row][col]
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        s = str(v)
        return "" if s.lower() in ("nan", "none") else s

    def _on_meta_field_changed(self, key: str, value: str) -> None:
        if self._suppress_writeback or self._model is None or self._row is None:
            return
        self._model.bulk_set([self._row], key, value)

    def _on_acq_field_changed(self, key: str, value: str) -> None:
        """Commit a per-row device / institution override into the scaffold spec."""
        if self._suppress_writeback or self._model is None or self._row is None:
            return
        self._model.set_acq_override(self._row, key, value)

    def _on_model_data_changed(self, top_left, bottom_right, _roles=()) -> None:
        if self._model is None or self._row is None:
            return
        if top_left.row() <= self._row <= bottom_right.row():
            # Re-render but suppress the writeback that the rebuilt
            # ``QLineEdit`` widgets would trigger via ``editingFinished``
            # on focus loss during the rebuild.
            self._suppress_writeback = True
            try:
                self.set_selected_row(self._row)
            finally:
                self._suppress_writeback = False


__all__ = ["PropertiesPanel"]
