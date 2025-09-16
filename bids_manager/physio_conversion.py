"""Utilities for detecting and converting physiological DICOM recordings.

This module centralises all logic required to locate physiological waveform
DICOM files in a raw dataset and convert them into BIDS-compliant outputs using
:mod:`bidsphysio`.  The conversion helpers are intentionally isolated from the
main HeuDiConv runner so they can be unit tested independently and reused by
other entry points in the future.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Callable, Iterable, List, Optional, Protocol, Sequence

import pydicom
from pydicom.errors import InvalidDicomError

try:  # pragma: no cover - import is validated via unit tests
    from bidsphysio.dcm2bids import dcm2bidsphysio
except Exception:  # pragma: no cover - handled gracefully below
    dcm2bidsphysio = None  # type: ignore

from .renaming.schema_renamer import (  # type: ignore  # circular import guard for mypy
    SeriesInfo,
    load_bids_schema,
    propose_bids_basename,
)
from .renaming.config import DEFAULT_SCHEMA_DIR


LOGGER = logging.getLogger(__name__)

# Physiological waveforms stored in Siemens CMRR DICOM files are contained in the
# private CSA tag ``(0x7fe1, 0x1010)``.  We keep the tag identifier centralised
# so it can be referenced consistently throughout the module and unit tests.
PHYSIO_PRIVATE_TAG = (0x7FE1, 0x1010)

# Common textual hints used by scanner vendors for physiological recordings.
# These are used as an additional sanity check before attempting a conversion so
# that unrelated private tags do not trigger unnecessary work.
_PHYSIO_KEYWORDS = ("physio", "physiolog", "pulse", "resp", "cardio")

# The BIDS schema is relatively expensive to load, therefore we cache it once at
# import time.  The GUI already ships with the schema directory bundled so the
# load call is inexpensive and deterministic.
_DEFAULT_SCHEMA = load_bids_schema(DEFAULT_SCHEMA_DIR)


class PhysioDataLike(Protocol):
    """Minimal protocol describing the API returned by :mod:`bidsphysio`."""

    def save_to_bids(self, bids_fName: str) -> None:  # pragma: no cover - protocol
        ...


Converter = Callable[[Sequence[str]], PhysioDataLike]


@dataclass
class PhysioSeries:
    """Metadata for a detected physiological DICOM recording."""

    path: Path
    series_description: str
    series_uid: str
    acquisition_number: Optional[int]
    session: Optional[str]


def _default_converter(paths: Sequence[str]) -> PhysioDataLike:
    """Wrapper around :func:`bidsphysio.dcm2bids.dcm2bidsphysio.dcm2bids`.

    The :mod:`bidsphysio` API expects either a single path or a list of paths.
    We always pass a list in order to present a consistent callable signature to
    the rest of the module.  A descriptive :class:`RuntimeError` is raised when
    the optional dependency is missing so that callers can provide a custom
    converter during tests.
    """

    if dcm2bidsphysio is None:  # pragma: no cover - environment guard
        raise RuntimeError(
            "bidsphysio is required to convert physiological recordings."
            " Install the `bidsphysio` extra or supply a custom converter."
        )
    return dcm2bidsphysio.dcm2bids(list(paths), verbose=False)


def _looks_like_dicom(path: Path) -> bool:
    """Return ``True`` when *path* resembles a DICOM file.

    The helper mirrors the lightweight checks used across the code base.  Files
    with known DICOM extensions are accepted immediately while extensionless
    files are inspected for the standard ``DICM`` marker located 128 bytes into
    the stream.  Any I/O error results in ``False`` to keep the scan robust.
    """

    lower = path.name.lower()
    if lower.endswith((".dcm", ".ima")):
        return True
    if "." in lower:
        return False
    try:
        with path.open("rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except Exception:
        return False


def _series_text(ds: pydicom.Dataset, fallback: str) -> str:
    """Extract a descriptive label for the sequence."""

    for attr in ("SeriesDescription", "ProtocolName"):
        value = getattr(ds, attr, None)
        if value:
            return str(value)
    return fallback


def _has_physio_payload(ds: pydicom.Dataset) -> bool:
    """Return ``True`` when the dataset contains physiological waveforms."""

    if PHYSIO_PRIVATE_TAG in ds:
        return True
    waveform = getattr(ds, "WaveformSequence", None)
    if waveform:
        return True
    image_type = getattr(ds, "ImageType", None)
    if image_type and any("phys" in str(item).lower() for item in image_type):
        return True
    return False


def _text_has_physio_hint(text: str) -> bool:
    low = text.lower()
    return any(token in low for token in _PHYSIO_KEYWORDS)


def _session_from_path(path_parts: Iterable[str]) -> Optional[str]:
    """Extract a BIDS session label from an iterable of path components."""

    for part in path_parts:
        low = part.lower()
        if low.startswith("ses-"):
            label = part.split("-", 1)[1]
            label = re.sub(r"[^0-9A-Za-z]+", "", label)
            if label:
                return label
    return None


def _normalise_subject_label(label: str) -> str:
    """Return a BIDS-compliant subject label without the ``sub-`` prefix."""

    text = label.strip()
    if text.lower().startswith("sub-"):
        text = text[4:]
    clean = re.sub(r"[^0-9A-Za-z]+", "", text)
    if not clean:
        raise ValueError("BIDS subject labels must contain alphanumeric characters")
    return clean


def _collect_physio_series(subject_root: Path, relative_subject: Path) -> List[PhysioSeries]:
    """Return physiological DICOM recordings discovered under *subject_root*."""

    series: List[PhysioSeries] = []
    seen_uids: set[str] = set()
    # Pre-compute any session hints embedded in the relative path.  When the raw
    # DICOM tree uses ``sub-XXX/ses-YYY`` style directories the ``relative_subject``
    # argument already contains the session token which should be propagated to
    # the resulting BIDS names.
    base_session = _session_from_path(relative_subject.parts)

    for path in sorted(subject_root.rglob("*")):
        if not path.is_file():
            continue
        if not _looks_like_dicom(path):
            continue
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                specific_tags=[
                    "SeriesDescription",
                    "ProtocolName",
                    "SeriesInstanceUID",
                    "AcquisitionNumber",
                    "SeriesNumber",
                    "ImageType",
                    PHYSIO_PRIVATE_TAG,
                    "WaveformSequence",
                ],
            )
        except (InvalidDicomError, FileNotFoundError, PermissionError):
            continue
        except Exception as exc:  # pragma: no cover - best effort guard
            LOGGER.debug("Skipping %s due to read error: %s", path, exc)
            continue

        text = _series_text(ds, path.stem)
        if not _has_physio_payload(ds):
            # Skip files that do not actually contain waveform data even if the
            # sequence name hints at physiology.  This avoids emitting empty
            # placeholders when the scanner exports template DICOMs without data.
            continue
        if not _text_has_physio_hint(text) and text == path.stem:
            # If we have no textual hint, ensure we only convert when the tag was
            # explicitly present (this condition is true when ``text`` fell back
            # to the filename).
            if PHYSIO_PRIVATE_TAG not in ds:
                continue

        uid = str(getattr(ds, "SeriesInstanceUID", "")) or str(path)
        if uid in seen_uids:
            continue
        seen_uids.add(uid)

        rel_folder = path.parent.relative_to(subject_root)
        session = base_session or _session_from_path((relative_subject / rel_folder).parts)

        acq = getattr(ds, "AcquisitionNumber", None)
        if acq is None:
            acq = getattr(ds, "SeriesNumber", None)
        acq_number = int(acq) if isinstance(acq, (int, float)) else None

        series.append(
            PhysioSeries(
                path=path,
                series_description=text,
                series_uid=uid,
                acquisition_number=acq_number,
                session=session,
            )
        )

    return series


def _prepare_bids_prefix(bids_root: Path, datatype: str, base: str) -> Path:
    """Return the target prefix path for a BIDS entity."""

    tokens = base.split("_")
    subject_token = next((token for token in tokens if token.startswith("sub-")), None)
    if not subject_token:
        raise ValueError(f"Unable to determine subject label from {base!r}")
    subject_dir = bids_root / subject_token

    session_token = next((token for token in tokens if token.startswith("ses-")), None)
    if session_token:
        prefix_dir = subject_dir / session_token / datatype
    else:
        prefix_dir = subject_dir / datatype

    prefix_dir.mkdir(parents=True, exist_ok=True)
    return prefix_dir / base


def convert_physiological_data(
    raw_root: Path,
    relative_subject: str,
    bids_root: Path,
    bids_subject: str,
    *,
    converter: Optional[Converter] = None,
    schema=None,
) -> List[Path]:
    """Convert physiological DICOM recordings for a single subject.

    Parameters
    ----------
    raw_root:
        Root directory containing the raw DICOM files.
    relative_subject:
        Path (relative to *raw_root*) pointing to the subject/session folder that
        was processed by HeuDiConv.
    bids_root:
        Output BIDS directory.
    bids_subject:
        BIDS subject label (``sub-XXX`` or ``XXX``).  The label is sanitized and
        reused when generating filenames so the physiologic data shares the same
        identifiers as the NIfTI volumes generated by HeuDiConv.
    converter:
        Optional callable used to perform the actual conversion.  Defaults to the
        :mod:`bidsphysio` converter but can be replaced in tests.
    schema:
        Optional BIDS schema instance.  Passing an explicit schema avoids the
        global cache when the caller already has one loaded.

    Returns
    -------
    list[Path]
        Paths to the generated ``*_physio.tsv.gz`` files.  Existing outputs are
        left untouched and therefore omitted from the returned list.
    """

    subject_rel_path = Path(relative_subject)
    subject_root = raw_root / subject_rel_path
    if not subject_root.exists():
        LOGGER.debug("Physio directory %s does not exist; skipping", subject_root)
        return []

    schema = schema or _DEFAULT_SCHEMA
    converter = converter or _default_converter

    subject_label = _normalise_subject_label(bids_subject)

    created: List[Path] = []

    for series in _collect_physio_series(subject_root, subject_rel_path):
        sequence = series.series_description or series.path.stem
        extra = {}

        info = SeriesInfo(
            subject=subject_label,
            session=series.session,
            modality="physio",
            sequence=sequence,
            rep=None,
            extra=extra,
        )

        datatype, base = propose_bids_basename(info, schema)
        prefix = _prepare_bids_prefix(bids_root, datatype, base)

        base_name = prefix.name
        if base_name.endswith("_physio"):
            base_root = base_name[:-len("_physio")]
        else:
            base_root = base_name
        tsv_path = prefix.with_name(f"{base_root}_physio.tsv.gz")
        if tsv_path.exists():
            LOGGER.debug("Physio output %s already exists; skipping", tsv_path)
            continue

        try:
            physio = converter([str(series.path)])
            physio.save_to_bids(str(prefix))
        except Exception as exc:  # pragma: no cover - exercised via integration
            LOGGER.error("Failed to convert %s: %s", series.path, exc)
            continue

        created.append(tsv_path)

    return created


__all__ = [
    "convert_physiological_data",
    "PHYSIO_PRIVATE_TAG",
]
