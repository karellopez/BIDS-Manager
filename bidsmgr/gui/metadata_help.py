"""Schema-sourced tooltips for the recording-metadata fields.

Every editable metadata field maps to a real BIDS sidecar key (or a documented
BIDS convention). The tooltip text is pulled live from the ``bidsschematools``
schema so it always matches the installed BIDS version - which also guarantees
the field we expose is a genuine schema key. A small set of fallbacks covers the
handful of fields that are conventions (montage) or participants.tsv columns
rather than sidecar-metadata keys.
"""

from __future__ import annotations

import functools
import re

from .. import schema as schema_mod

# UI field key -> BIDS ``objects.metadata`` key. ``None`` means the field is a
# convention / non-metadata column; use ``_FALLBACKS``.
_FIELD_BIDS_KEY: dict[str, str] = {
    "manufacturer": "Manufacturer",
    "amplifier_model": "ManufacturersModelName",
    "software_versions": "SoftwareVersions",
    "institution_name": "InstitutionName",
    "institution_dept": "InstitutionalDepartmentName",
    "line_freq": "PowerLineFrequency",
    "eeg_reference": "EEGReference",
    "eeg_ground": "EEGGround",
    "cap_manufacturer": "CapManufacturer",
    "cap_model": "CapManufacturersModelName",
    # MEG-specific (manual only - channel-derived MEG fields are left to mne-bids)
    "dewar_position": "DewarPosition",
    "associated_empty_room": "AssociatedEmptyRoom",
    "subject_artefact_description": "SubjectArtefactDescription",
    "manufacturer_suggestion": "Manufacturer",
}

# Fields that are BIDS conventions or participants.tsv columns (not sidecar
# metadata keys), so they are not in ``objects.metadata``.
_FALLBACKS: dict[str, str] = {
    "montage": (
        "MNE montage applied on conversion; sets the electrode positions written "
        "to electrodes.tsv. A BIDS convention, not a sidecar key."
    ),
    "PatientSex": "Participant sex (M / F / O). Written to participants.tsv (sex column).",
    "PatientAge": "Participant age in years at acquisition. Written to participants.tsv (age column).",
    "Handedness": "Participant handedness (R / L / A). Written to participants.tsv (handedness column).",
    "event": (
        "Map a recorded trigger code to a human-readable trial_type label in "
        "events.tsv. Blank labels are left untouched."
    ),
}


def _clean(text: str) -> str:
    """Strip BIDS markdown (links, backticks) and collapse whitespace."""
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    t = t.replace("`", "")
    return re.sub(r"\s+", " ", t).strip()


@functools.lru_cache(maxsize=256)
def bids_tooltip(bids_key: str) -> str:
    """Cleaned ``<display name>: <description>`` from the schema, or ``""``."""
    try:
        fi = schema_mod.field_metadata(bids_key)
    except Exception:
        return ""
    desc = _clean(fi.description)
    return f"{fi.display_name}: {desc}" if desc else fi.display_name


def tooltip_for(ui_field_key: str) -> str:
    """Tooltip for a metadata UI field (schema description, else a fallback)."""
    bids_key = _FIELD_BIDS_KEY.get(ui_field_key)
    if bids_key:
        t = bids_tooltip(bids_key)
        if t:
            return t
    return _FALLBACKS.get(ui_field_key, "")


__all__ = ["bids_tooltip", "tooltip_for"]
