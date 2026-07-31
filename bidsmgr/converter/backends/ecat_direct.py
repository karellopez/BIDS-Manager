"""ECAT to NIfTI backend, built on nibabel.

Siemens HRRT and the older CTI/ECAT scanners write ECAT rather than DICOM, and
dcm2niix cannot read it. nibabel can, and it is already a BIDS Manager
dependency, so ECAT support costs no new package and no subprocess.

The backend follows the same contract as every other one: take a
:class:`ConvertTask`, write into the per-subject staging directory, report what
landed on disk. Sidecar enrichment is not its job. It writes only what the ECAT
header states as fact (frame timing, units, radionuclide, institution); the
fields BIDS requires that no scanner records are filled later by
``fixups.pet_sidecar`` from the user's metadata spec.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..types import ConvertResult, ConvertTask

log = logging.getLogger(__name__)


class EcatDirect:
    """Convert an ECAT ``.v`` PET recording to NIfTI via nibabel.

    Stateless; safe to instantiate once per ``run_convert`` and reuse.
    """

    name = "ecat_direct"

    def can_handle(self, task: ConvertTask) -> bool:
        if task.datatype != "pet":
            return False
        if not task.source_files or not task.basename:
            return False
        # Signature check, not extension: ".v" is too generic to trust and a
        # valid ECAT file is sometimes shipped without it.
        from ...inventory.pet_ecat import is_ecat_file

        return any(is_ecat_file(fp) for fp in task.source_files if fp.exists())

    def convert(self, task: ConvertTask, staging_dir: Path) -> ConvertResult:
        t0 = time.monotonic()
        try:
            return self._convert_inner(task, staging_dir, t0)
        except Exception as exc:  # noqa: BLE001 - report, never crash the run
            log.exception("ecat_direct: unexpected error for %s", task.basename)
            return ConvertResult(
                task=task,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.monotonic() - t0,
            )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _convert_inner(
        self, task: ConvertTask, staging_dir: Path, t0: float,
    ) -> ConvertResult:
        import nibabel

        from ...inventory.pet_ecat import is_ecat_file, probe_ecat

        source = next(
            (fp for fp in task.source_files if fp.exists() and is_ecat_file(fp)),
            None,
        )
        if source is None:
            return ConvertResult(
                task=task, success=False,
                error="no readable ECAT file among the task's source files",
                duration_s=time.monotonic() - t0,
            )

        out_dir = staging_dir
        if task.session:
            out_dir = out_dir / f"ses-{task.session}"
        out_dir = out_dir / task.datatype
        out_dir.mkdir(parents=True, exist_ok=True)

        nii_path = out_dir / f"{task.basename}.nii.gz"
        json_path = out_dir / f"{task.basename}.json"

        try:
            img = nibabel.ecat.load(str(source))
            # get_fdata applies the per-frame scale factors ECAT stores in its
            # subheaders. Reading the raw array would silently drop them and
            # leave the image in arbitrary units.
            data = img.get_fdata()
            nifti = nibabel.Nifti1Image(data, img.affine)
            nifti.header.set_xyzt_units("mm", "sec")
            nibabel.save(nifti, str(nii_path))
        except Exception as exc:  # noqa: BLE001
            return ConvertResult(
                task=task, success=False,
                error=f"nibabel could not convert {source.name}: {exc}",
                duration_s=time.monotonic() - t0,
            )

        json_path.write_text(
            json.dumps(self._sidecar_from_header(source, probe_ecat(source)), indent=2),
            encoding="utf-8",
        )

        return ConvertResult(
            task=task,
            success=True,
            staged_files=(nii_path, json_path),
            duration_s=time.monotonic() - t0,
        )

    @staticmethod
    def _sidecar_from_header(source: Path, probe) -> dict:
        """The sidecar fields the ECAT header states outright.

        Deliberately narrow. Anything the header does not record (injected
        mass, administration mode, reconstruction parameters) is left for the
        metadata step rather than guessed here, so a missing field stays
        visibly missing in validation.
        """
        out: dict = {"Modality": "PT", "ConversionSoftware": "nibabel"}
        if probe is None:
            return out

        if probe.frame_durations:
            out["FrameDuration"] = [round(v, 6) for v in probe.frame_durations]
        if probe.frame_starts:
            out["FrameTimesStart"] = [round(v, 6) for v in probe.frame_starts]
        if probe.isotope:
            from ...inventory.pet import normalise_radionuclide

            out["TracerRadionuclide"] = normalise_radionuclide(probe.isotope)
        if probe.tracer:
            out["TracerName"] = probe.tracer
        if probe.facility:
            out["InstitutionName"] = probe.facility
        return out


__all__ = ["EcatDirect"]
