"""Fill the BIDS PET sidecar after conversion.

Runs in convert phase 2, beside ``eeg_sidecar.enrich_recording_sidecars``, and
does three separable jobs on each staged ``*_pet.json``:

1. **Rename** what the converter wrote under a DICOM name to its BIDS name.
   dcm2niix writes ``ReconstructionMethod``; BIDS calls it ``ReconMethodName``.
2. **Fill** the fields no scanner records, from the user's metadata spec. This
   is most of the PET sidecar: a scanner knows how it reconstructed an image
   but not how much tracer went into the person, in what form, or when.
3. **Prune** keys that do not belong in a BIDS sidecar. dcm2niix is run with
   ``-ba n`` so BIDS Manager keeps ``SeriesInstanceUID`` for provenance, and
   that also keeps the patient identifiers alongside it.

Order matters: rename before fill, so a user value always wins over the
converter's, and prune last so a pruned key cannot be resurrected.

Everything is additive. A field the user left unset means "leave whatever the
converter wrote", which is what keeps a good dcm2niix value from being
clobbered by a blank.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from ..recording_meta import PetAcquisitionSpec, RecordingMetaSpec, resolve_pet

log = logging.getLogger(__name__)

_SUPPORTED_DATATYPES = frozenset({"pet"})

# DICOM-derived key -> BIDS key. dcm2niix emits the left-hand names; the BIDS
# PET spec asks for the right-hand ones. A rename never overwrites a key that
# is already present under its BIDS name.
_RENAMES: dict[str, str] = {
    "ReconstructionMethod": "ReconMethodName",
}

# Identifying and non-BIDS keys dcm2niix leaves in the sidecar. BIDS Manager
# passes ``-ba n`` to keep SeriesInstanceUID for provenance, which also keeps
# these. None is a BIDS field and several directly identify the participant, so
# they have no place in a shareable dataset.
#
# The study and series UIDs are deliberately NOT pruned: they are pseudonymous,
# they are what lets a converted image be traced back to its source series, and
# BIDS permits extra keys.
IDENTIFYING_KEYS: frozenset[str] = frozenset({
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientWeight",
    "PatientSize",
    "AccessionNumber",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "RequestingPhysician",
    "InstitutionalDepartmentAddress",
})

# Non-identifying keys that are simply not part of BIDS and add noise. Kept
# separate from the identifying set so the two can be reasoned about apart.
NON_BIDS_KEYS: frozenset[str] = frozenset({
    # dcm2niix's own classification hint. Purely internal to the scan, and
    # meaningless in a shipped dataset.
    "BidsGuess",
    "Modality",
    "ProtocolName",
    "SeriesDescription",
    "StudyDescription",
    "SeriesTime",
    "SeriesNumber",
    "DecayCorrection",
    "RadionuclideHalfLife",
    "RadionuclidePositronFraction",
    "ImageType",
    "ConvolutionKernel",
})

# spec field -> BIDS sidecar key, for the values that map straight across.
_SCALAR_FIELDS: dict[str, str] = {
    "tracer_name": "TracerName",
    "tracer_radionuclide": "TracerRadionuclide",
    "tracer_molecular_weight": "TracerMolecularWeight",
    "tracer_molecular_weight_units": "TracerMolecularWeightUnits",
    "tracer_radlex": "TracerRadLex",
    "tracer_snomed": "TracerSNOMED",
    "injected_radioactivity": "InjectedRadioactivity",
    "injected_radioactivity_units": "InjectedRadioactivityUnits",
    "injected_mass": "InjectedMass",
    "injected_mass_units": "InjectedMassUnits",
    "specific_radioactivity": "SpecificRadioactivity",
    "specific_radioactivity_units": "SpecificRadioactivityUnits",
    "molar_activity": "MolarActivity",
    "molar_activity_units": "MolarActivityUnits",
    "injected_volume": "InjectedVolume",
    "purity": "Purity",
    "mode_of_administration": "ModeOfAdministration",
    "injection_start": "InjectionStart",
    "injection_end": "InjectionEnd",
    "infusion_radioactivity": "InfusionRadioactivity",
    "infusion_start": "InfusionStart",
    "infusion_speed": "InfusionSpeed",
    "infusion_speed_units": "InfusionSpeedUnits",
    "time_zero": "TimeZero",
    "scan_start": "ScanStart",
    "acquisition_mode": "AcquisitionMode",
    "image_decay_corrected": "ImageDecayCorrected",
    "image_decay_correction_time": "ImageDecayCorrectionTime",
    "attenuation_correction": "AttenuationCorrection",
    "units": "Units",
    "body_part": "BodyPart",
    "recon_method_name": "ReconMethodName",
    "recon_filter_type": "ReconFilterType",
    "recon_filter_size": "ReconFilterSize",
    "manufacturer": "Manufacturer",
    "manufacturers_model_name": "ManufacturersModelName",
}

_LIST_FIELDS: dict[str, str] = {
    "recon_method_parameter_labels": "ReconMethodParameterLabels",
    "recon_method_parameter_units": "ReconMethodParameterUnits",
    "recon_method_parameter_values": "ReconMethodParameterValues",
}

@lru_cache(maxsize=1)
def _array_typed_keys() -> frozenset[str]:
    """Sidecar keys the BIDS schema types as arrays.

    dcm2niix writes a bare scalar for a single-frame acquisition where the
    schema wants a one-element array, and a validator rightly rejects that.
    Wrapping such a value is a type fix rather than a guess.

    Read from the schema instead of hardcoded, so the set follows the BIDS
    version in use. Fields the schema types with an ``anyOf`` (``type`` absent)
    are deliberately excluded: it accepts both shapes there, so touching the
    converter's choice would be meddling, not fixing.
    """
    from .. import schema as schema_mod

    keys = set()
    for key in _PET_SIDECAR_KEYS:
        try:
            if schema_mod.field_metadata(key).type == "array":
                keys.add(key)
        except KeyError:
            continue
    return frozenset(keys)


# Every key the fixup might write or touch, used to scope the schema lookup
# above. Values come from the spec maps plus the converter-written fields.
_PET_SIDECAR_KEYS: tuple[str, ...] = (
    "ScatterFraction", "DecayCorrectionFactor", "ReconFilterSize",
    "FrameDuration", "FrameTimesStart", "DoseCalibrationFactor",
    "ReconMethodParameterValues", "ReconMethodParameterLabels",
    "ReconMethodParameterUnits",
)


def enrich_pet_sidecars(
    subject_staging_dir: Path,
    tasks: Iterable,
    spec: Optional[RecordingMetaSpec],
    *,
    prune_identifiers: bool = True,
) -> int:
    """Enrich every staged PET sidecar for one subject.

    Parameters
    ----------
    subject_staging_dir
        Per-subject staging tree (``<bids_root>/.tmp_bidsmgr/sub-<id>/``).
    tasks
        The :class:`ConvertTask` objects for this subject. Non-PET tasks are
        ignored.
    spec
        The dataset's metadata spec. ``None`` still runs the rename, type-fix
        and prune passes, which need no user input.
    prune_identifiers
        Remove patient identifiers. On by default; the study and series UIDs
        are kept either way.

    Returns
    -------
    int
        Count of sidecar files modified.
    """
    n_modified = 0
    for task in tasks:
        if getattr(task, "datatype", "") not in _SUPPORTED_DATATYPES:
            continue
        basename = getattr(task, "basename", "") or ""
        if not basename:
            continue

        sidecar = _find_sidecar(subject_staging_dir, basename)
        if sidecar is None:
            continue

        pet = None
        if spec is not None:
            pet = resolve_pet(spec, str(getattr(task, "row_id", "")))

        if _apply_sidecar(sidecar, pet, prune_identifiers=prune_identifiers):
            n_modified += 1

    return n_modified


# ----------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------


def _find_sidecar(staging: Path, basename: str) -> Optional[Path]:
    """Locate ``<basename>.json`` anywhere under the subject staging tree."""
    matches = sorted(staging.rglob(f"{basename}.json"))
    return matches[0] if matches else None


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read PET sidecar %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _apply_sidecar(
    sidecar: Path,
    pet: Optional[PetAcquisitionSpec],
    *,
    prune_identifiers: bool,
) -> bool:
    """Rename, fill, type-fix and prune one sidecar. True if it changed."""
    data = _read_json(sidecar)
    if not data:
        return False
    before = json.dumps(data, sort_keys=True)

    # 1. rename DICOM-derived keys, never clobbering an existing BIDS key
    for old, new in _RENAMES.items():
        if old in data:
            value = data.pop(old)
            data.setdefault(new, value)

    # 2. fill from the spec. A user value always wins: it is a deliberate
    #    statement, where the converter's is an inference.
    if pet is not None:
        for field, key in _SCALAR_FIELDS.items():
            value = getattr(pet, field, None)
            if value is not None and value != "":
                data[key] = value
        for field, key in _LIST_FIELDS.items():
            value = getattr(pet, field, None)
            if value:
                data[key] = list(value)

    # 3. type fixes: BIDS types several PET fields as one-entry-per-frame
    #    arrays, and dcm2niix writes a bare scalar for a single frame.
    for key in _array_typed_keys():
        if key in data and not isinstance(data[key], list):
            data[key] = [data[key]]

    # 4. prune
    drop = set(NON_BIDS_KEYS)
    if prune_identifiers:
        drop |= IDENTIFYING_KEYS
    for key in drop:
        data.pop(key, None)

    if json.dumps(data, sort_keys=True) == before:
        return False
    _write_json(sidecar, data)
    return True


__all__ = [
    "IDENTIFYING_KEYS",
    "NON_BIDS_KEYS",
    "enrich_pet_sidecars",
]
