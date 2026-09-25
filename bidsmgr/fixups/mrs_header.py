"""Strip identifiers from inside a NIfTI-MRS file, and write its sidecar.

MR spectroscopy converts since dcm2niix v1.0.20260724, and it converts into
a shape nothing else in BIDS has: a complex free-induction decay in the
NIfTI data block, plus **a JSON header carried INSIDE the ``.nii.gz``** as
NIfTI extension code 44, which is where the NIfTI-MRS standard puts the
acquisition parameters a spectrum cannot be read without.

That extension is the problem this module exists for. Measured on this
lab's own data the moment MRS conversion was switched on::

    PatientName            OL_3846^YY11YY11
    PatientID              OL_3846
    PatientDoB             19930101
    PatientSex             M
    PatientWeight          66
    InstitutionAddress     Kuepkersweg 74,Oldenburg,District,DE,26129

Every other converter output is a NIfTI whose metadata lives in a sibling
``.json``, so pruning the sidecar was enough. Here the metadata is INSIDE
the image, where no sidecar pass can reach it, and a dataset that looks
de-identified because its JSON is clean is worse than one that obviously is
not.

So this rewrites the extension. It also copies the acquisition parameters
the spectrum genuinely needs (``SpectrometerFrequency``, ``ResonantNucleus``,
``EchoTime``, ``RepetitionTime``, the dimension labels) OUT to the BIDS
``.json`` sidecar, so a reader can see them without a NIfTI library, which
is the whole point of a sidecar.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from .identifiers import strip_identifiers

log = logging.getLogger(__name__)

#: NIfTI extension code the NIfTI-MRS standard reserves for its header.
MRS_EXTENSION_CODE = 44

#: Header fields a reader needs to interpret the spectrum, lifted into the
#: sidecar so they are visible without opening the image. Everything else
#: stays in the extension, where the standard puts it.
_SIDECAR_FIELDS: tuple[str, ...] = (
    "SpectrometerFrequency",
    "ResonantNucleus",
    "EchoTime",
    "RepetitionTime",
    "InversionTime",
    "ExcitationFlipAngle",
    "Manufacturer",
    "ManufacturersModelName",
    "InstitutionName",
    "dim_5", "dim_6", "dim_7",
)


def read_mrs_header(path: Path) -> Optional[dict]:
    """The NIfTI-MRS JSON header inside ``path``, or ``None``.

    ``None`` means this is not a NIfTI-MRS file, which is how every caller
    decides whether any of this applies.
    """
    try:
        import nibabel as nib

        img = nib.load(str(path))
        for ext in img.header.extensions:
            if ext.get_code() != MRS_EXTENSION_CODE:
                continue
            raw = ext.get_content()
            text = raw.decode("utf-8", "replace").rstrip("\x00")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
    except Exception as exc:  # noqa: BLE001 - not every .nii.gz is MRS
        log.debug("no NIfTI-MRS header in %s: %s", path, exc)
    return None


def clean_mrs_file(path: Path, *, write_sidecar: bool = True) -> list[str]:
    """Strip identifiers from ``path``'s MRS header. Return what was removed.

    Returns an empty list for a file that is not NIfTI-MRS, or that carried
    nothing identifying, so a caller can report honestly either way.

    The image is rewritten only when something actually changed. A
    conversion that rewrites every file it touches makes its own logs
    useless for telling which files it changed.
    """
    header = read_mrs_header(path)
    if header is None:
        return []

    removed = strip_identifiers(header)
    if write_sidecar:
        _write_sidecar(path, header)
    if not removed:
        return []

    try:
        import nibabel as nib
        from nibabel.nifti1 import Nifti1Extension

        img = nib.load(str(path))
        kept = [
            e for e in img.header.extensions
            if e.get_code() != MRS_EXTENSION_CODE
        ]
        payload = json.dumps(header).encode("utf-8")
        img.header.extensions.clear()
        for e in kept:
            img.header.extensions.append(e)
        img.header.extensions.append(
            Nifti1Extension(MRS_EXTENSION_CODE, payload)
        )
        # Written through a temporary name and moved into place: a half
        # written image is worse than an unmodified one, and a conversion
        # can be interrupted.
        #
        # The temporary name keeps the ORIGINAL extension, with the marker
        # in the stem. nibabel infers the format from the suffix, so
        # ``x.nii.gz.tmp`` is a file it refuses to write, and the refusal
        # is caught below and reported as "still carries", which is exactly
        # the outcome this function exists to prevent.
        suffix = ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix
        stem = path.name[: -len(suffix)]
        tmp = path.with_name(f"{stem}.bidsmgr-tmp{suffix}")
        nib.save(img, str(tmp))
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "could not rewrite the MRS header of %s (%s); "
            "the file still carries %s",
            path.name, exc, ", ".join(removed),
        )
        return []

    log.info(
        "%s: removed %d identifying field(s) from the NIfTI-MRS header: %s",
        path.name, len(removed), ", ".join(sorted(removed)),
    )
    return removed


def _write_sidecar(path: Path, header: dict) -> None:
    """Copy the reader-facing acquisition fields out to the ``.json``.

    Never overwrites a value already there: the sidecar is what the user
    and the metadata engine edit, and the extension is what the converter
    wrote.
    """
    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            sidecar = path.with_name(name[: -len(ext)] + ".json")
            break
    else:
        return

    try:
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {}
    except (OSError, ValueError):
        existing = {}

    added = False
    for field in _SIDECAR_FIELDS:
        value = header.get(field)
        if value in (None, "") or field in existing:
            continue
        existing[field] = value
        added = True
    if not added:
        return
    try:
        sidecar.write_text(
            json.dumps(existing, indent=4) + "\n", encoding="utf-8",
        )
    except OSError as exc:
        log.warning("could not write %s: %s", sidecar.name, exc)


def clean_mrs_outputs(staging_dir: Path, tasks: Iterable) -> int:
    """Clean every staged MRS file for one subject. Returns how many changed."""
    n = 0
    for task in tasks:
        if str(getattr(task, "datatype", "")) != "mrs":
            continue
        basename = str(getattr(task, "basename", "") or "")
        if not basename:
            continue
        for candidate in sorted(staging_dir.rglob(f"{basename}*.nii*")):
            if clean_mrs_file(candidate):
                n += 1
    return n


__all__ = [
    "MRS_EXTENSION_CODE",
    "clean_mrs_file",
    "clean_mrs_outputs",
    "read_mrs_header",
]
