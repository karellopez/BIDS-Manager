"""ECAT PET scanner.

ECAT is the native format of Siemens HRRT and older CTI/ECAT scanners. dcm2niix
cannot read it, so PET support that stopped at DICOM would leave those scanners
out. nibabel reads ECAT and is already a BIDS Manager dependency, so this costs
nothing extra.

The module mirrors :mod:`bidsmgr.inventory.eeg_meg`: probe each candidate file,
emit one inventory row per recording, and let the shared orchestrator merge the
rows into the unified TSV.

Detection is by the ``MATRIX7x`` magic at byte 0, not by the ``.v`` extension.
``.v`` is far too generic to trust on its own (Verilog, GNU V, patch files all
use it), and conversely a valid ECAT file is sometimes shipped without it.

Qt-free.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ECAT7 main headers begin with this signature. The trailing digit is the
# sub-version (70, 72, 73 all appear in the OpenNeuroPET phantom set).
ECAT_MAGIC = b"MATRIX7"

# Candidate extensions, used only to skip an open() on obviously unrelated
# files. Anything extensionless still gets its magic checked.
ECAT_EXTS = (".v", ".img", ".ecat")


@dataclass(frozen=True)
class EcatProbe:
    """What one ECAT file tells us before any user input."""

    source: Path
    n_frames: int
    shape: tuple[int, ...]
    isotope: str = ""
    half_life: Optional[float] = None
    dose: Optional[float] = None
    tracer: str = ""
    facility: str = ""
    scan_start: Optional[int] = None
    frame_durations: tuple[float, ...] = ()
    frame_starts: tuple[float, ...] = ()
    patient_id: str = ""


def is_ecat_file(path: str | os.PathLike) -> bool:
    """True if ``path`` starts with the ECAT7 ``MATRIX7x`` signature."""
    p = Path(path)
    if p.suffix and p.suffix.lower() not in ECAT_EXTS:
        return False
    try:
        with open(p, "rb") as fh:
            return fh.read(len(ECAT_MAGIC)) == ECAT_MAGIC
    except OSError:
        return False


def _decode(value) -> str:
    """ECAT header strings are null-padded bytes inside a numpy record.

    nibabel hands them back as 0-d numpy arrays or numpy byte scalars, not as
    plain ``bytes``, so unwrap with ``.item()`` first. Skipping that step leaves
    the numpy repr (``np.bytes_(b'F-18')``) in the cell.
    """
    if value is None:
        return ""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, bytes):
        return value.split(b"\0")[0].decode("latin-1", "replace").strip()
    return str(value).strip()


def _number(value) -> Optional[float]:
    """Unwrap a numpy scalar to a plain float, or ``None`` if it is not one."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_ecat(path: Path) -> Optional[EcatProbe]:
    """Read one ECAT file's headers. Returns ``None`` if it cannot be read."""
    try:
        import nibabel
    except ImportError:  # pragma: no cover - nibabel is a hard dependency
        log.warning("nibabel unavailable; cannot probe ECAT %s", path)
        return None

    try:
        img = nibabel.ecat.load(str(path))
        mh = img.header
        subheaders = img.get_subheaders()
    except Exception as exc:  # noqa: BLE001 - a bad file must not kill the scan
        log.warning("could not read ECAT %s: %s", path, exc)
        return None

    def _main(key: str):
        try:
            return mh[key]
        except (KeyError, ValueError, TypeError):
            return None

    def _sub(key: str) -> tuple[float, ...]:
        out = []
        for sh in subheaders.subheaders:
            try:
                out.append(float(sh[key]))
            except (KeyError, ValueError, TypeError):
                out.append(0.0)
        return tuple(out)

    isotope = _decode(_main("isotope_name") or "")
    tracer = _decode(_main("radiopharmaceutical") or "")
    if tracer.lower() in ("unknown", "none"):
        tracer = ""

    dose = _number(_main("dosage"))
    half_life = _number(_main("isotope_halflife"))
    scan_start = _number(_main("scan_start_time"))

    # ECAT frame durations and start times are milliseconds; BIDS wants seconds.
    durations = tuple(v / 1000.0 for v in _sub("frame_duration"))
    starts = tuple(v / 1000.0 for v in _sub("frame_start_time"))

    return EcatProbe(
        source=path,
        n_frames=len(subheaders.subheaders),
        shape=tuple(img.shape),
        isotope=isotope,
        half_life=half_life,
        dose=dose,
        tracer=tracer,
        facility=_decode(_main("facility_name") or ""),
        scan_start=int(scan_start) if scan_start is not None else None,
        frame_durations=durations,
        frame_starts=starts,
        patient_id=_decode(_main("patient_id") or ""),
    )


def find_ecat_files(root: str | os.PathLike) -> list[Path]:
    """Every ECAT file under ``root``, detected by signature."""
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            fp = Path(dirpath) / name
            if is_ecat_file(fp):
                out.append(fp)
    return sorted(out)


def _subject_from_path(path: Path, root: Path) -> str:
    """Best-effort subject label, mirroring the EEG/MEG path heuristics.

    A literal ``sub-XXX`` component wins; otherwise the parent folder name,
    which is how the phantom archives are laid out (one folder per scanner).
    """
    parts = path.relative_to(root).parts
    for part in parts:
        if part.lower().startswith("sub-"):
            return part[4:]
    return parts[0] if len(parts) > 1 else path.stem


def scan_ecat(
    root_dir: str | os.PathLike,
    *,
    cancel_check=None,
) -> pd.DataFrame:
    """Inventory every ECAT file under ``root_dir``.

    Returns a DataFrame in the unified inventory's shape (the orchestrator
    backfills the columns this scanner does not own). One row per file, which
    for ECAT is also one row per recording: unlike DICOM, a whole dynamic study
    with all its frames lives in a single file.
    """
    from ..util.cancel import is_cancelled

    import json

    from .. import schema as schema_mod

    root = Path(root_dir)
    rows: list[dict] = []
    # One BIDS label per distinct source subject, numbered in encounter order.
    # Mirrors the EEG/MEG scanner: the label is provisional and the user
    # reconciles it by editing BIDS_name in the inventory.
    bids_id_for_subject: dict[str, str] = {}

    for idx, fp in enumerate(find_ecat_files(root)):
        if (idx & 7) == 0 and is_cancelled(cancel_check):
            from ..util.cancel import OperationCancelled
            raise OperationCancelled("scan cancelled by user")

        probe = probe_ecat(fp)
        if probe is None:
            continue

        rel = fp.relative_to(root)
        subject = _subject_from_path(fp, root)
        if subject not in bids_id_for_subject:
            bids_id_for_subject[subject] = f"sub-{len(bids_id_for_subject) + 1:03d}"
        bids_name = bids_id_for_subject[subject]

        entities = {"subject": bids_name[len("sub-"):]}
        try:
            basename = schema_mod.build_basename(entities, "pet", "pet")
        except Exception as exc:  # noqa: BLE001 - fall back to a literal name
            log.debug("schema.build_basename failed for %s: %s", bids_name, exc)
            basename = f"{bids_name}_pet"

        rows.append({
            "subject": subject,
            "BIDS_name": bids_name,
            "proposed_datatype": "pet",
            "proposed_basename": basename,
            "Proposed BIDS name": basename,
            "entities": json.dumps(entities, sort_keys=True),
            "source_folder": str(rel.parent) if rel.parent != Path(".") else root.name,
            "source_file": str(rel),
            "sequence": fp.stem,
            "include": 1,
            "n_files": 1,
            "modality": "pet",
            "modality_bids": "pet",
            "bids_guess_datatype": "pet",
            "bids_guess_suffix": "pet",
            "bids_guess_classifier": "ecat_header",
            "bids_guess_confidence": 0.90,
            "bids_guess_skip": False,
            "PatientID": probe.patient_id,
            # The PET suggestion columns this scanner can fill from the header.
            "radionuclide_suggestion": _isotope_to_bids(probe.isotope),
            "tracer_suggestion": probe.tracer,
            # No unit is appended on purpose. The ECAT "dosage" field has no
            # unit declared in the header and vendors disagree on whether it is
            # mCi or MBq, so asserting one would be a guess dressed up as a
            # fact. The user confirms the unit in the metadata step.
            "injected_dose_suggestion": (
                f"{round(probe.dose, 4)}" if probe.dose else ""
            ),
            "_ecat_frames": probe.n_frames,
            "_ecat_frame_durations": probe.frame_durations,
            "_ecat_frame_starts": probe.frame_starts,
            "_ecat_facility": probe.facility,
            "_ecat_scan_start": probe.scan_start,
        })

    return pd.DataFrame(rows)


def _isotope_to_bids(raw: str) -> str:
    """``"F-18"`` to ``"F18"``, reusing the DICOM-side normaliser."""
    from .pet import normalise_radionuclide

    return normalise_radionuclide(raw)


__all__ = [
    "ECAT_EXTS",
    "ECAT_MAGIC",
    "EcatProbe",
    "find_ecat_files",
    "is_ecat_file",
    "probe_ecat",
    "scan_ecat",
]
