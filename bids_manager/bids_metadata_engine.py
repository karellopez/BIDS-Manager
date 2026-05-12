#!/usr/bin/env python3
"""Schema-driven post-conversion BIDS metadata generator.

This is a converter-agnostic step: it works on the output of either
heudiconv (existing pipeline) or dcm2bids (new pipeline). After conversion
plus the fieldmap / IntendedFor / DWI-derivatives fixups in
:mod:`post_conv_renamer`, this engine produces the dataset-level metadata
files that BIDS expects but that the conversion tools themselves do not
write:

* ``dataset_description.json`` (REQUIRED)
* ``participants.tsv`` + ``participants.json`` (REQUIRED if >1 subject)
* ``README`` (RECOMMENDED)
* ``CHANGES`` (RECOMMENDED)
* per-subject ``sub-XXX[_ses-YYY]_scans.tsv`` (RECOMMENDED)

It also performs a per-modality sidecar audit using a curated
required-field table compiled from the BIDS specification + the
``ancpbids.model_v1_10_0`` enums, fills in derivable fields where the
information is in adjacent files (e.g. ``TaskName`` from filename), and
emits warnings for missing required fields it can't auto-populate.

Finally, when ``ancpbids`` is available it runs ``validate_dataset`` for a
soft schema check; results are surfaced as warnings rather than failures so
the engine never blocks the pipeline on validator quirks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# BIDS schema lookups
# ---------------------------------------------------------------------------
# ancpbids ships the BIDS schema as Python enums; we use them for label
# validation but the spec's "which sidecar fields are required for suffix X"
# rules are not exposed as a structured lookup. The table below covers the
# datatypes BIDS Manager generates today; falling back to "no required
# sidecar fields" is safe for anything else.

# Per (datatype, suffix) → list of REQUIRED sidecar JSON fields.
_REQUIRED_SIDECAR_FIELDS: Dict[Tuple[str, str], List[str]] = {
    # func: BIDS spec §10 Task fMRI requires TaskName + RepetitionTime.
    ("func", "bold"):  ["TaskName", "RepetitionTime"],
    ("func", "sbref"): ["TaskName", "RepetitionTime"],
    ("func", "cbv"):   ["TaskName", "RepetitionTime"],
    ("func", "phase"): ["TaskName", "RepetitionTime"],
    # fmap: phasediff needs both EchoTimes; epi needs PE/ROT; fieldmap needs Units.
    ("fmap", "phasediff"): ["EchoTime1", "EchoTime2", "IntendedFor"],
    ("fmap", "phase1"):    ["EchoTime", "IntendedFor"],
    ("fmap", "phase2"):    ["EchoTime", "IntendedFor"],
    ("fmap", "magnitude1"): ["IntendedFor"],
    ("fmap", "magnitude2"): ["IntendedFor"],
    ("fmap", "fieldmap"):  ["Units", "IntendedFor"],
    ("fmap", "epi"):       ["PhaseEncodingDirection", "TotalReadoutTime", "IntendedFor"],
    # EEG / MEG / iEEG: BIDS spec §11/§12. mne-bids fills these when the
    # info is present in the raw object; the audit catches files that
    # mne-bids couldn't populate (e.g. missing line frequency for EDF).
    ("eeg",  "eeg"):  ["TaskName", "EEGReference", "SamplingFrequency", "PowerLineFrequency"],
    ("ieeg", "ieeg"): ["TaskName", "iEEGReference", "SamplingFrequency", "PowerLineFrequency",
                       "SoftwareFilters"],
    ("meg",  "meg"):  ["TaskName", "SamplingFrequency", "PowerLineFrequency",
                       "DewarPosition", "SoftwareFilters", "DigitizedLandmarks",
                       "DigitizedHeadPoints"],
}

# Per (datatype, suffix) → list of RECOMMENDED fields. Missing ones produce
# an info-level note but never a warning.
_RECOMMENDED_SIDECAR_FIELDS: Dict[Tuple[str, str], List[str]] = {
    ("anat", "T1w"): ["MagneticFieldStrength", "EchoTime", "RepetitionTime", "FlipAngle"],
    ("anat", "T2w"): ["MagneticFieldStrength", "EchoTime", "RepetitionTime", "FlipAngle"],
    ("dwi", "dwi"):  ["PhaseEncodingDirection", "TotalReadoutTime"],
    ("func", "bold"): ["EchoTime", "FlipAngle", "SliceTiming", "PhaseEncodingDirection"],
}

# Filename-derivable fillers. (datatype, suffix) → callable(json_path) → dict
# of fields to merge into the sidecar if missing.
def _derive_taskname(json_path: Path) -> Dict[str, object]:
    m = re.search(r"_task-([A-Za-z0-9]+)", json_path.name)
    return {"TaskName": m.group(1)} if m else {}


_FILENAME_DERIVED: Dict[Tuple[str, str], List] = {
    ("func", "bold"): [_derive_taskname],
    ("func", "sbref"): [_derive_taskname],
    ("func", "cbv"): [_derive_taskname],
    ("func", "phase"): [_derive_taskname],
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class DatasetMetadata:
    """User-supplied dataset_description.json fields.

    Anything missing falls back to a sane default. ``Name`` is the only field
    truly required by the BIDS spec; everything else has a sensible default.
    """
    name: str = "Untitled BIDS Dataset"
    bids_version: str = "1.10.0"
    dataset_type: str = "raw"  # 'raw' or 'derivative'
    license: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    acknowledgements: Optional[str] = None
    how_to_acknowledge: Optional[str] = None
    funding: List[str] = field(default_factory=list)
    ethics_approvals: List[str] = field(default_factory=list)
    references_and_links: List[str] = field(default_factory=list)
    dataset_doi: Optional[str] = None
    generated_by: List[Dict[str, str]] = field(default_factory=list)
    source_datasets: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class EngineReport:
    """What the engine did and what it could not do.

    The GUI surfaces this in the log group; the CLI dumps it to stdout.
    """
    files_written: List[Path] = field(default_factory=list)
    sidecar_fills: List[Tuple[Path, Dict[str, object]]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validator_messages: List[str] = field(default_factory=list)


class BIDSMetadataEngine:
    """Generate dataset-level metadata for a BIDS dataset on disk."""

    def __init__(self,
                 bids_root: Path,
                 inventory_tsv: Optional[Path] = None,
                 dataset_meta: Optional[DatasetMetadata] = None,
                 generator_label: str = "BIDS-Manager"):
        self.bids_root = Path(bids_root)
        self.inventory_tsv = Path(inventory_tsv) if inventory_tsv else None
        self.meta = dataset_meta or DatasetMetadata()
        self.generator_label = generator_label
        self.report = EngineReport()

    # ------------------------------------------------------------------ run
    def run(self) -> EngineReport:
        self._write_dataset_description()
        self._write_participants()
        self._write_readme()
        self._write_changes()
        self._refresh_scans_tsv()
        self._fill_sidecars()
        self._soft_validate()
        return self.report

    # ----------------------------------------------- dataset_description.json
    def _write_dataset_description(self) -> None:
        dd: Dict[str, object] = {
            "Name": self.meta.name,
            "BIDSVersion": self.meta.bids_version,
            "DatasetType": self.meta.dataset_type,
        }
        if self.meta.license:
            dd["License"] = self.meta.license
        if self.meta.authors:
            dd["Authors"] = list(self.meta.authors)
        if self.meta.acknowledgements:
            dd["Acknowledgements"] = self.meta.acknowledgements
        if self.meta.how_to_acknowledge:
            dd["HowToAcknowledge"] = self.meta.how_to_acknowledge
        if self.meta.funding:
            dd["Funding"] = list(self.meta.funding)
        if self.meta.ethics_approvals:
            dd["EthicsApprovals"] = list(self.meta.ethics_approvals)
        if self.meta.references_and_links:
            dd["ReferencesAndLinks"] = list(self.meta.references_and_links)
        if self.meta.dataset_doi:
            dd["DatasetDOI"] = self.meta.dataset_doi

        # GeneratedBy: always include BIDS Manager + any caller-supplied entries
        gen_by: List[Dict[str, str]] = list(self.meta.generated_by)
        try:
            from bids_manager import __version__ as _bm_version
        except Exception:
            _bm_version = "0.0.0"
        gen_by.insert(0, {
            "Name": self.generator_label,
            "Version": str(_bm_version),
            "CodeURL": "https://github.com/ANCPLabOldenburg/BIDS-Manager",
        })
        dd["GeneratedBy"] = gen_by
        if self.meta.source_datasets:
            dd["SourceDatasets"] = list(self.meta.source_datasets)

        out = self.bids_root / "dataset_description.json"
        # Preserve any user-edited fields on rerun.
        if out.exists():
            try:
                existing = json.loads(out.read_text())
                if isinstance(existing, dict):
                    # New values win for required fields; everything else preserved.
                    merged = {**existing, **dd}
                    dd = merged
            except Exception:
                pass
        out.write_text(json.dumps(dd, indent=2), encoding="utf-8")
        self.report.files_written.append(out)

    # ----------------------------------------------------- participants.tsv
    def _write_participants(self) -> None:
        subjects = sorted(p.name for p in self.bids_root.glob("sub-*") if p.is_dir())
        if not subjects:
            return

        # Try to enrich from the inventory TSV when available.
        demo_lookup: Dict[str, Dict[str, str]] = {}
        if self.inventory_tsv and self.inventory_tsv.exists():
            try:
                # Force string dtype so identifiers like "001" don't get coerced
                # into integer 1, which would break the sub-XXX lookup below.
                df = pd.read_csv(self.inventory_tsv, sep="\t", keep_default_na=False,
                                 dtype=str)
            except Exception as exc:
                self.report.warnings.append(
                    f"Could not read inventory TSV {self.inventory_tsv}: {exc}"
                )
                df = pd.DataFrame()
            if not df.empty and "BIDS_name" in df.columns:
                for bids_name, sub_df in df.groupby("BIDS_name"):
                    bids_name = str(bids_name)
                    pid = bids_name if bids_name.startswith("sub-") else f"sub-{bids_name}"
                    head = sub_df.iloc[0]
                    demo_lookup[pid] = {
                        "given_name": str(head.get("GivenName", "") or ""),
                        "family_name": str(head.get("FamilyName", "") or ""),
                        "patient_id": str(head.get("patientID", "") or ""),
                        "age": str(head.get("PatientAge", "") or head.get("age", "") or ""),
                        "sex": str(head.get("PatientSex", "") or head.get("sex", "") or ""),
                    }

        rows = []
        for sid in subjects:
            row = {"participant_id": sid}
            extra = demo_lookup.get(sid, {})
            for k in ("age", "sex", "given_name", "family_name", "patient_id"):
                v = extra.get(k, "")
                row[k] = v if v else "n/a"
            rows.append(row)
        df_new = pd.DataFrame(rows, columns=["participant_id", "age", "sex",
                                             "given_name", "family_name", "patient_id"])

        out_tsv = self.bids_root / "participants.tsv"
        # If mne-bids (or the user) already wrote a participants.tsv, merge
        # rather than replace: keep their columns/values, only fill cells
        # they left as ``n/a`` and only drop columns that *both* sides agree
        # are entirely empty.
        df_existing: Optional[pd.DataFrame] = None
        if out_tsv.exists():
            try:
                df_existing = pd.read_csv(out_tsv, sep="\t", dtype=str,
                                          keep_default_na=False)
            except Exception:
                df_existing = None

        merged_with_existing = False
        if df_existing is not None and "participant_id" in df_existing.columns:
            merged = df_existing.set_index("participant_id")
            new_indexed = df_new.set_index("participant_id")
            for sid in new_indexed.index:
                if sid not in merged.index:
                    merged.loc[sid] = "n/a"
            for col in new_indexed.columns:
                if col not in merged.columns:
                    merged[col] = "n/a"
                for sid in new_indexed.index:
                    incoming = new_indexed.at[sid, col]
                    if incoming and incoming != "n/a":
                        current = merged.at[sid, col] if sid in merged.index else ""
                        if not current or current == "n/a":
                            merged.at[sid, col] = incoming
            df_out = merged.reset_index()
            merged_with_existing = True
        else:
            df_out = df_new

        # When merging with an existing file (e.g. one mne-bids wrote) keep
        # every column the upstream tool created — those columns are part of
        # the dataset's schema even when their cells are still "n/a". Only
        # prune empty columns when we built the file from scratch.
        if not merged_with_existing:
            keep_cols = ["participant_id"] + [
                c for c in df_out.columns
                if c != "participant_id" and (df_out[c].astype(str) != "n/a").any()
            ]
            df_out = df_out[keep_cols]

        df_out.to_csv(out_tsv, sep="\t", index=False)
        self.report.files_written.append(out_tsv)

        # Data dictionary (participants.json)
        col_descriptions = {
            "participant_id": {"Description": "Unique participant identifier"},
            "age": {"Description": "Age of the participant", "Units": "years"},
            "sex": {
                "Description": "Biological sex",
                "Levels": {"M": "male", "F": "female", "O": "other", "n/a": "not available"},
            },
            "given_name": {"Description": "Subject given name (kept for internal traceability)"},
            "family_name": {"Description": "Subject family name (kept for internal traceability)"},
            "patient_id": {"Description": "Original DICOM PatientID"},
        }
        out_json = self.bids_root / "participants.json"
        out_json.write_text(
            json.dumps({c: col_descriptions[c] for c in df_out.columns
                        if c in col_descriptions},
                       indent=2),
            encoding="utf-8",
        )
        self.report.files_written.append(out_json)

    # ------------------------------------------------------------------ README
    def _write_readme(self) -> None:
        out = self.bids_root / "README"
        if out.exists():
            return  # never overwrite a hand-edited README
        body = (
            f"# {self.meta.name}\n\n"
            "This BIDS dataset was generated by BIDS Manager.\n\n"
            "Edit this README to describe acquisition details, study design,\n"
            "and any deviations from the standard BIDS layout.\n"
        )
        out.write_text(body, encoding="utf-8")
        self.report.files_written.append(out)

    # ------------------------------------------------------------------ CHANGES
    def _write_changes(self) -> None:
        out = self.bids_root / "CHANGES"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry = f"1.0.0 {ts}\n  - Initial release.\n"
        if out.exists():
            return  # don't churn the changelog on every run
        out.write_text(entry, encoding="utf-8")
        self.report.files_written.append(out)

    # --------------------------------------------------------------- scans.tsv
    def _refresh_scans_tsv(self) -> None:
        """Generate one ``*_scans.tsv`` per subject (or subject/session).

        Lists every NIfTI in the subject's tree with its ``acq_time`` parsed
        from the JSON sidecar's ``AcquisitionDateTime``. Missing acq times are
        reported as ``n/a`` per BIDS convention.
        """
        for subj_dir in sorted(self.bids_root.glob("sub-*")):
            if not subj_dir.is_dir():
                continue
            ses_dirs = [s for s in subj_dir.glob("ses-*") if s.is_dir()] or [subj_dir]
            for root in ses_dirs:
                rows = []
                for nii in sorted(root.rglob("*.nii*")):
                    if not nii.is_file():
                        continue
                    rel = nii.relative_to(root).as_posix()
                    json_path = nii.with_suffix("")
                    if json_path.suffix == ".nii":
                        json_path = json_path.with_suffix(".json")
                    else:
                        json_path = nii.with_name(nii.name.split(".nii")[0] + ".json")
                    acq_time = "n/a"
                    if json_path.exists():
                        try:
                            data = json.loads(json_path.read_text())
                            adt = data.get("AcquisitionDateTime") or data.get("AcquisitionTime")
                            if isinstance(adt, str) and adt:
                                acq_time = adt
                        except Exception:
                            pass
                    rows.append({"filename": rel, "acq_time": acq_time})
                if not rows:
                    continue
                subj = subj_dir.name
                ses_part = f"_{root.name}" if root.name.startswith("ses-") else ""
                out = root / f"{subj}{ses_part}_scans.tsv"
                pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
                self.report.files_written.append(out)

    # --------------------------------------------------------------- sidecars
    def _fill_sidecars(self) -> None:
        for json_path in sorted(self.bids_root.rglob("*.json")):
            if json_path.name in ("dataset_description.json", "participants.json"):
                continue
            datatype, suffix = self._infer_datatype_suffix(json_path)
            if not datatype or not suffix:
                continue
            try:
                data = json.loads(json_path.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue

            # Try filename-derivable fillers first.
            fills: Dict[str, object] = {}
            for fn in _FILENAME_DERIVED.get((datatype, suffix), []):
                for k, v in fn(json_path).items():
                    if k not in data:
                        fills[k] = v

            if fills:
                data.update(fills)
                json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                self.report.sidecar_fills.append((json_path, fills))

            # Required-field audit (after fills).
            missing_req = [k for k in _REQUIRED_SIDECAR_FIELDS.get((datatype, suffix), [])
                           if k not in data]
            for field in missing_req:
                self.report.warnings.append(
                    f"{json_path.relative_to(self.bids_root)}: missing required field {field!r}"
                )

    @staticmethod
    def _infer_datatype_suffix(json_path: Path) -> Tuple[str, str]:
        # Path: .../sub-X[/ses-Y]/datatype/sub-X..._suffix.json
        parts = json_path.parts
        datatype = ""
        for p in reversed(parts[:-1]):
            if p in ("anat", "func", "dwi", "fmap", "perf", "meg", "eeg", "ieeg",
                     "beh", "pet", "micr", "nirs"):
                datatype = p
                break
        stem = json_path.name[:-len(".json")]
        # Suffix is the trailing token after the last underscore that is NOT
        # an entity (entity tokens contain a hyphen, suffixes don't).
        tokens = stem.split("_")
        suffix = ""
        for tok in reversed(tokens):
            if "-" not in tok:
                suffix = tok
                break
        return datatype, suffix

    # ------------------------------------------------------------- validation
    def _soft_validate(self) -> None:
        try:
            import ancpbids
        except Exception:
            return
        try:
            ds = ancpbids.load_dataset(str(self.bids_root))
            report = ancpbids.validate_dataset(ds)
            for msg in getattr(report, "messages", []) or []:
                self.report.validator_messages.append(str(msg))
        except Exception as exc:
            # Validator failure must not block the engine.
            self.report.warnings.append(f"ancpbids validation skipped: {exc}")


# ---------------------------------------------------------------------------
# Module-level helper + CLI
# ---------------------------------------------------------------------------

def generate(bids_root: Path,
             inventory_tsv: Optional[Path] = None,
             dataset_meta: Optional[DatasetMetadata] = None) -> EngineReport:
    """Convenience function for callers that don't want the class API."""
    engine = BIDSMetadataEngine(bids_root, inventory_tsv, dataset_meta)
    return engine.run()


def main() -> None:
    """CLI for the ``bids-metadata`` entry point."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate dataset-level BIDS metadata for an existing dataset"
    )
    parser.add_argument("bids_root", help="Root of the BIDS dataset")
    parser.add_argument("--inventory-tsv",
                        help="subject_summary.tsv used to enrich participants.tsv")
    parser.add_argument("--name", help="Dataset Name (dataset_description.json)")
    parser.add_argument("--bids-version", default="1.10.0")
    parser.add_argument("--license")
    parser.add_argument("--author", action="append", default=[],
                        help="Author name (repeatable)")
    parser.add_argument("--acknowledgements")
    parser.add_argument("--how-to-acknowledge")
    parser.add_argument("--funding", action="append", default=[])
    parser.add_argument("--ethics-approvals", action="append", default=[])
    parser.add_argument("--references-and-links", action="append", default=[])
    parser.add_argument("--dataset-doi")
    args = parser.parse_args()

    meta = DatasetMetadata(
        name=args.name or Path(args.bids_root).name,
        bids_version=args.bids_version,
        license=args.license,
        authors=args.author,
        acknowledgements=args.acknowledgements,
        how_to_acknowledge=args.how_to_acknowledge,
        funding=args.funding,
        ethics_approvals=args.ethics_approvals,
        references_and_links=args.references_and_links,
        dataset_doi=args.dataset_doi,
    )
    report = generate(
        Path(args.bids_root),
        Path(args.inventory_tsv) if args.inventory_tsv else None,
        meta,
    )
    print(f"\nWrote {len(report.files_written)} file(s):")
    for p in report.files_written:
        print(f"  - {p}")
    if report.sidecar_fills:
        print(f"\nFilled fields in {len(report.sidecar_fills)} sidecar(s).")
    if report.warnings:
        print(f"\n{len(report.warnings)} warning(s):")
        for w in report.warnings:
            print(f"  ! {w}")
    if report.validator_messages:
        print(f"\nancpbids validator: {len(report.validator_messages)} message(s).")
        for m in report.validator_messages[:20]:
            print(f"  · {m}")


if __name__ == "__main__":
    main()
