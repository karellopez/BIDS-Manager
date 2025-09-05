#!/usr/bin/env python3
"""run_dcm2niix_from_scans.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convert DICOM series using :mod:`dcm2niix` based on instructions
provided in a ``scans.tsv`` file.  The TSV plays the same role as the
heuristic file previously generated for HeuDiConv and contains
information about which DICOM folders to convert and how to name the
resulting BIDS files.

The script performs a two stage process:

1. ``dcm2niix`` is executed for each row in the TSV to produce NIfTI
   images and JSON sidecars.
2. Basic BIDS metadata is injected using the schema shipped with
   :mod:`bids_manager` so that the output is immediately usable by BIDS
   aware tools.

The goal is to keep the conversion workflow similar to the original
HeuDiConv based pipeline while relying solely on ``dcm2niix`` for the
heavy lifting.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import pandas as pd
import json
import re

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def safe_stem(text: str) -> str:
    """Return a filename friendly version of *text*.

    Any character that is not alphanumeric, ``-`` or ``_`` is replaced with an
    underscore.  Trailing underscores are stripped to avoid awkward file names.
    """

    return re.sub(r"[^0-9A-Za-z_-]+", "_", str(text).strip()).strip("_")


def ensure_dataset_description(bids_root: Path) -> None:
    """Ensure that ``dataset_description.json`` exists.

    The BIDS version is read from the schema bundled with the project so the
    metadata is always in sync with the version of the specification used by
    the GUI.  Existing files are preserved with only the ``BIDSVersion`` field
    being added when missing.
    """

    schema_dir = Path(__file__).parent / "miscellaneous" / "schema"
    bids_version = (schema_dir / "BIDS_VERSION").read_text(encoding="utf-8").strip()
    desc_file = bids_root / "dataset_description.json"
    if desc_file.exists():
        with open(desc_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data.setdefault("BIDSVersion", bids_version)
    with open(desc_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def patch_metadata(json_file: Path) -> None:
    """Clean trivial metadata issues in ``json_file``.

    ``dcm2niix`` occasionally writes placeholder values such as ``"TODO"``.
    This helper removes those markers so the resulting dataset is BIDS
    compatible without manual fixes.
    """

    with open(json_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for key, value in list(meta.items()):
        if isinstance(value, str) and "todo" in value.lower():
            meta[key] = ""
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def convert_from_tsv(dicom_root: Path, scans_tsv: Path, bids_out: Path) -> None:
    """Run ``dcm2niix`` according to the instructions in ``scans_tsv``.

    Parameters
    ----------
    dicom_root : Path
        Directory containing the source DICOM folders.
    scans_tsv : Path
        TSV file describing which sequences to convert.  Expected columns are
        ``source_folder``, ``BIDS_name``, ``session``, ``sequence``,
        ``modality_bids`` and an optional ``include`` flag.
    bids_out : Path
        Destination BIDS root where converted data will be written.
    """

    bids_out.mkdir(parents=True, exist_ok=True)
    ensure_dataset_description(bids_out)

    df = pd.read_csv(scans_tsv, sep="\t", keep_default_na=False)
    if "include" in df.columns:
        df = df[df["include"] == 1]

    # Determine repetition counts so repeated sequences receive a ``rep-`` tag.
    rep_idx = (
        df.groupby(["BIDS_name", "session", "sequence"], dropna=False)
        .cumcount()
        + 1
    )
    rep_tot = (
        df.groupby(["BIDS_name", "session", "sequence"], dropna=False)["sequence"]
        .transform("count")
    )
    df = df.assign(_rep_idx=rep_idx, _rep_tot=rep_tot)

    for row in df.itertuples():
        subj = f"sub-{row.BIDS_name}" if getattr(row, "BIDS_name", "") else "sub-unknown"
        ses = f"ses-{row.session}" if getattr(row, "session", "") else ""
        container = getattr(row, "modality_bids", "misc") or "misc"
        stem = safe_stem(row.sequence)

        # Build output directory and filename
        out_dir = bids_out / subj
        if ses:
            out_dir /= ses
        out_dir /= container
        out_dir.mkdir(parents=True, exist_ok=True)

        fname_parts = [subj]
        if ses:
            fname_parts.append(ses)
        fname_parts.append(stem)
        if row._rep_tot > 1:
            fname_parts.append(f"rep-{row._rep_idx}")
        fname = "_".join(fname_parts)

        dicom_dir = Path(dicom_root) / str(getattr(row, "source_folder", "."))
        cmd = [
            "dcm2niix",
            "-b",
            "y",
            "-z",
            "y",
            "-f",
            fname,
            "-o",
            str(out_dir),
            str(dicom_dir),
        ]
        subprocess.run(cmd, check=True)

        js = out_dir / f"{fname}.json"
        if js.exists():
            patch_metadata(js)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Convert DICOMs using dcm2niix guided by scans.tsv"
    )
    parser.add_argument("dicom_root", help="Root directory containing DICOM data")
    parser.add_argument("scans_tsv", help="TSV file with conversion instructions")
    parser.add_argument("bids_out", help="Destination BIDS directory")
    args = parser.parse_args()

    convert_from_tsv(Path(args.dicom_root), Path(args.scans_tsv), Path(args.bids_out))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
