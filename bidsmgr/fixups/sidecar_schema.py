"""Schema-driven sidecar repairs that apply to every datatype.

The datatype-specific fixups next door know what a particular modality needs.
This one knows only what the SCHEMA says, and so applies everywhere:

* :func:`repair_key_names` fixes a key whose spelling differs from the
  standard's only in case. Real example, and the reason this exists: BIDS
  spells the field ``MISCChannelCount`` for EEG and ``MiscChannelCount`` for
  MEG and iEEG, and mne-bids writes the MEG spelling into EEG sidecars. The
  value is right and the key is not, so every EEG dataset carried an undeclared
  field while the declared one read as missing.
* :func:`repair_array_types` wraps a bare scalar in a list where the schema
  types the field as an array of one entry per frame or per run. Generalised
  from the PET fixup, which had it first because dcm2niix does this on
  single-frame PET.
* :func:`fill_agnostic_fields` writes the values that belong to the DATASET
  rather than to a modality, chiefly where the study was done, into every
  sidecar whose datatype declares them. They used to reach electrophysiology
  alone, because the only enrichment pass that ran was the EEG/MEG one.

None of this guesses. A key is renamed only when the schema declares that exact
name modulo case, a value is wrapped only where the schema says array, and a
field is written only when the user supplied it and the schema declares it for
that datatype.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .. import schema as schema_mod
from ..recording_meta import RecordingMetaSpec

log = logging.getLogger(__name__)


# Values the dataset as a whole carries, mapped to the sidecar key that holds
# them. Agnostic by nature: which building a scanner sits in has nothing to do
# with whether it is a scanner, an amplifier or a dewar. The schema decides
# whether a given datatype actually declares each one.
_AGNOSTIC_FIELDS: dict[str, str] = {
    "institution_name": "InstitutionName",
    "institution_dept": "InstitutionalDepartmentName",
}


@lru_cache(maxsize=256)
def _declared_fields(datatype: str, suffix: str) -> tuple[str, ...]:
    """Every field name the schema declares for this kind of file."""
    try:
        return tuple(
            f.name for f in schema_mod.optional_sidecar_fields(datatype, suffix)
        ) + tuple(
            f.name for f in schema_mod.required_sidecar_fields(datatype, suffix)
        ) + tuple(
            f.name for f in schema_mod.recommended_sidecar_fields(datatype, suffix)
        ) + tuple(
            f.name for f in schema_mod.deprecated_sidecar_fields(datatype, suffix)
        )
    except Exception:  # noqa: BLE001 - an unknown datatype is not an error here
        return ()


@lru_cache(maxsize=256)
def _array_typed(datatype: str, suffix: str) -> frozenset[str]:
    """Fields the schema types as an array for this kind of file.

    Fields the schema leaves untyped (an ``anyOf`` that accepts both a scalar
    and an array) are excluded on purpose: it permits both shapes there, so
    reshaping the converter's choice would be meddling rather than repair.
    """
    out = set()
    for level in (
        schema_mod.required_sidecar_fields,
        schema_mod.recommended_sidecar_fields,
        schema_mod.optional_sidecar_fields,
    ):
        try:
            for f in level(datatype, suffix):
                if f.type == "array":
                    out.add(f.name)
        except Exception:  # noqa: BLE001
            continue
    return frozenset(out)


def repair_key_names(data: dict, datatype: str, suffix: str) -> int:
    """Rename keys that differ from the standard's spelling only in case.

    Returns the number of keys renamed. A key whose correct spelling is already
    present is dropped rather than merged: the correctly-spelled one wins,
    because that is the one every reader will look at.
    """
    declared = _declared_fields(datatype, suffix)
    if not declared:
        return 0
    canonical = {name.lower(): name for name in declared}

    renamed = 0
    for key in list(data):
        proper = canonical.get(key.lower())
        if proper is None or proper == key:
            continue
        if proper in data:
            data.pop(key)
        else:
            data[proper] = data.pop(key)
        renamed += 1
    return renamed


def repair_array_types(data: dict, datatype: str, suffix: str) -> int:
    """Wrap a bare scalar where the schema declares an array. Returns the count."""
    fixed = 0
    for name in _array_typed(datatype, suffix):
        if name in data and not isinstance(data[name], list):
            value = data[name]
            if value is None or value == "":
                continue
            data[name] = [value]
            fixed += 1
    return fixed


def fill_agnostic_fields(
    data: dict, datatype: str, suffix: str, spec: Optional[RecordingMetaSpec],
) -> int:
    """Write the dataset-wide values the schema declares for this datatype.

    Never overwrites a value already present: a converter that read the
    institution out of a DICOM header knows better than a dataset-wide default.
    """
    if spec is None:
        return 0
    declared = set(_declared_fields(datatype, suffix))
    if not declared:
        return 0

    written = 0
    for attr, key in _AGNOSTIC_FIELDS.items():
        if key not in declared or key in data:
            continue
        value = getattr(spec.defaults, attr, None)
        if value in (None, ""):
            continue
        data[key] = value
        written += 1
    return written


def repair_sidecars(
    staging: Path,
    tasks,
    spec: Optional[RecordingMetaSpec] = None,
) -> int:
    """Apply every schema-driven repair to the staged tree. Returns files changed.

    Walks the staged sidecars rather than the task list, so a sidecar the
    backend wrote for an output we did not enumerate (a fieldmap split, a
    multi-echo series) is repaired too.
    """
    from ..editor.bidsmgr_checks import infer_datatype_suffix

    changed = 0
    for sidecar in sorted(staging.rglob("*.json")):
        if ".bidsmgr" in sidecar.parts:
            continue
        datatype, suffix = infer_datatype_suffix(sidecar, staging)
        if not (datatype and suffix):
            continue
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue

        before = json.dumps(data, sort_keys=True)
        n = repair_key_names(data, datatype, suffix)
        n += repair_array_types(data, datatype, suffix)
        n += fill_agnostic_fields(data, datatype, suffix, spec)
        if not n or json.dumps(data, sort_keys=True) == before:
            continue
        try:
            sidecar.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:  # noqa: BLE001
            log.warning("could not rewrite %s: %s", sidecar, exc)
            continue
        changed += 1
    return changed


__all__ = [
    "fill_agnostic_fields",
    "repair_array_types",
    "repair_key_names",
    "repair_sidecars",
]
