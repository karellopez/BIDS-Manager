"""Pure-data models for recording-level enrichment metadata.

These describe the information a recording file cannot carry on its own:
the reference and ground electrodes, hardware/software filters, amplifier and
cap details, institution, the meaning of trigger codes, and per-task protocol
notes. A scan seeds what it can detect; the user fills the rest. The models are
deliberately I/O-free (Pydantic v2 only): reading, writing, resolving, and
applying live in sibling modules.

Every leaf field is optional. Enrichment is *additive*: a field left unset means
"leave whatever the converter already wrote". The structure separates a single
dataset-wide ``defaults`` block from sparse per-recording ``overrides`` keyed by
the inventory row id, so a heterogeneous cohort (mixed caps, amplifiers, or
references) is expressible without repeating shared values.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator

from .schema_types import bids_channel_types, mne_channel_types

# A trigger-code -> human-label mapping (e.g. {"S 20": "eyes_open"}).
EventMap = dict[str, str]


class _Model(BaseModel):
    """Shared config: forbid unknown keys so typos surface at load time."""

    model_config = ConfigDict(extra="forbid")


class AcceptableImpedance(_Model):
    """Impedance acceptance threshold recorded during acquisition."""

    value: Optional[float] = None
    units: Optional[str] = None


class LightingConditions(_Model):
    """Ambient lighting of the recording environment."""

    description: Optional[str] = None
    measurement: Optional[str] = None


class ExtrasSpec(_Model):
    """Supplemental, non-required acquisition conditions.

    These are not BIDS-required sidecar keys; they are written as additional
    fields (BIDS permits extra keys) when present.
    """

    acceptable_impedance: Optional[AcceptableImpedance] = None
    electrode_type: Optional[str] = None
    conductive_medium: Optional[str] = None
    faraday_cage: Optional[bool] = None
    sound_proof: Optional[bool] = None
    lighting_conditions: Optional[LightingConditions] = None


class FilterSpec(_Model):
    """One hardware or software filter applied during acquisition.

    ``info`` is copied verbatim into the sidecar under the filter's ``name``,
    grouped by ``kind`` into ``HardwareFilters`` / ``SoftwareFilters``.
    """

    name: str
    kind: str  # "Hardware" | "Software"
    info: dict[str, Any] = {}

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, v: str) -> str:
        if v not in ("Hardware", "Software"):
            raise ValueError("filter kind must be 'Hardware' or 'Software'")
        return v


class AuxChannelSpec(_Model):
    """How one auxiliary (non-data) channel should be typed and described.

    Used to upgrade a channel the reader saw as generic ``misc`` to its real
    BIDS type and to fill its units / description in ``channels.tsv``.
    """

    mne_type: Optional[str] = None
    bids_type: Optional[str] = None
    description: Optional[str] = None
    units: Optional[str] = None
    location: Optional[Union[str, dict[str, str]]] = None

    @field_validator("bids_type")
    @classmethod
    def _check_bids_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in bids_channel_types():
            raise ValueError(
                f"{v!r} is not a BIDS channel type "
                f"(allowed: {', '.join(sorted(bids_channel_types()))})"
            )
        return v

    @field_validator("mne_type")
    @classmethod
    def _check_mne_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in mne_channel_types():
            raise ValueError(
                f"{v!r} is not an MNE channel type "
                f"(allowed: {', '.join(sorted(mne_channel_types()))})"
            )
        return v


class TaskProtocol(_Model):
    """Free-form per-task protocol notes that map into the task sidecar."""

    task_description: Optional[str] = None
    instructions: Optional[str] = None


class AcquisitionSpec(_Model):
    """The recording-level technical block (defaults and per-row overrides).

    Carries both the values applied during the write (``power_line_freq``,
    ``montage``) and the values folded into the sidecar afterwards (reference,
    ground, device, institution, filters, extras, aux channels).
    """

    power_line_freq: Optional[float] = None
    montage: Optional[str] = None
    eeg_reference: Optional[str] = None
    eeg_ground: Optional[str] = None
    manufacturer: Optional[str] = None
    amplifier_model: Optional[str] = None  # -> ManufacturersModelName
    software: Optional[str] = None
    software_versions: Optional[str] = None
    cap_manufacturer: Optional[str] = None
    cap_model: Optional[str] = None
    institution_name: Optional[str] = None
    institution_dept: Optional[str] = None
    aux_channels: dict[str, AuxChannelSpec] = {}
    filters: list[FilterSpec] = []
    extras: Optional[ExtrasSpec] = None


class RecordingMetaSpec(_Model):
    """Root enrichment object for one dataset.

    ``defaults`` applies to every recording; ``overrides`` carries the sparse
    per-recording deltas keyed by the inventory ``row_id`` (the recording's
    source path for EEG/MEG). ``event_maps`` and ``task_protocols`` are keyed by
    BIDS task label, with ``"*"`` as a fallback event map for all tasks.
    """

    schema_version: int = 1
    defaults: AcquisitionSpec = AcquisitionSpec()
    task_protocols: dict[str, TaskProtocol] = {}
    event_maps: dict[str, EventMap] = {}
    overrides: dict[str, AcquisitionSpec] = {}
    # Dataset-level phenotype measure tables (TSV/CSV/XLSX/ODS paths keyed by
    # participant_id). Written to ``phenotype/<measure>.tsv`` + ``.json`` by the
    # metadata engine. Agnostic: applies to any modality.
    phenotype_files: list[str] = []


__all__ = [
    "EventMap",
    "AcceptableImpedance",
    "LightingConditions",
    "ExtrasSpec",
    "FilterSpec",
    "AuxChannelSpec",
    "TaskProtocol",
    "AcquisitionSpec",
    "RecordingMetaSpec",
]
