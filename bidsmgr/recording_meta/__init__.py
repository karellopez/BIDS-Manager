"""Recording-level enrichment metadata: pure-data models + resolution + I/O.

This subpackage holds the information a raw EEG/MEG/iEEG recording cannot carry
on its own (reference, ground, filters, device, institution, event-code meaning,
task protocol) as I/O-free Pydantic models, plus the logic to resolve a
dataset-default-plus-per-row-override spec into a single effective view for one
recording. It imports only from :mod:`bidsmgr.schema` and standard scientific
deps; it is Qt-free and depends on no other ``bidsmgr`` subpackage.

The enrichment is *applied* to written BIDS files by
:mod:`bidsmgr.fixups.eeg_sidecar`; this subpackage only describes and resolves.
"""

from __future__ import annotations

from .models import (
    AcceptableImpedance,
    AcquisitionSpec,
    AuxChannelSpec,
    EventMap,
    ExtrasSpec,
    FilterSpec,
    LightingConditions,
    RecordingMetaSpec,
    TaskProtocol,
)
from .resolve import EffectiveSpec, merge_acquisition, resolve_effective
from .schema_types import bids_channel_types, mne_channel_types
from .serialize import (
    DEFAULT_POWER_LINE_FREQ,
    RECORDING_META_SIDECAR,
    default_spec,
    dump_spec,
    load_spec,
    scaffold_sidecar_path,
)

__all__ = [
    "AcceptableImpedance",
    "AcquisitionSpec",
    "AuxChannelSpec",
    "EventMap",
    "ExtrasSpec",
    "FilterSpec",
    "LightingConditions",
    "RecordingMetaSpec",
    "TaskProtocol",
    "EffectiveSpec",
    "merge_acquisition",
    "resolve_effective",
    "bids_channel_types",
    "mne_channel_types",
    "DEFAULT_POWER_LINE_FREQ",
    "RECORDING_META_SIDECAR",
    "default_spec",
    "dump_spec",
    "load_spec",
    "scaffold_sidecar_path",
]
