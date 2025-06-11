#!/usr/bin/env python3
"""
run_heudiconv_from_heuristic.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Launch HeuDiConv using *auto_heuristic.py*,
handling cleaned-vs-physical folder names automatically.
"""

from __future__ import annotations
from pathlib import Path
import importlib.util
import subprocess
import os
import sys
from typing import Dict, List, Optional
import pandas as pd
import re
import json


# ────────────────── helpers ──────────────────
def load_sid_map(heur: Path) -> Dict[str, str]:
    spec = importlib.util.spec_from_file_location("heuristic", heur)
    module = importlib.util.module_from_spec(spec)         # type: ignore
    assert spec.loader
    spec.loader.exec_module(module)                        # type: ignore
    return module.SID_MAP                                  # type: ignore


def clean_name(raw: str) -> str:
    return "".join(ch for ch in raw if ch.isalnum())

def safe_stem(text: str) -> str:
    """Return filename-friendly version of *text* (used for study names)."""
    return re.sub(r"[^0-9A-Za-z_-]+", "_", text.strip()).strip("_")


def physical_by_clean(raw_root: Path) -> Dict[str, str]:
    """Return mapping cleaned_name → relative folder path for all subdirs."""
    mapping: Dict[str, str] = {}
    for p in raw_root.rglob("*"):
        if not p.is_dir():
            continue
        rel = str(p.relative_to(raw_root))
        base = p.name
        mapping.setdefault(rel, rel)
        mapping.setdefault(clean_name(rel), rel)
        mapping.setdefault(base, rel)
        mapping.setdefault(clean_name(base), rel)
    return mapping



def detect_depth(folder: Path) -> int:
    """Minimum depth (#subdirs) from *folder* to any .dcm file."""
    for root, _dirs, files in os.walk(folder):
        if any(f.lower().endswith(".dcm") for f in files):
            rel = Path(root).relative_to(folder)
            return len(rel.parts)
    raise RuntimeError(f"No DICOMs under {folder}")


def heudi_cmd(raw_root: Path,
              phys_folders: List[str],
              heuristic: Path,
              bids_out: Path,
              depth: int,
              use_nipype: bool) -> List[str]:
    """Return argument list for invoking heudiconv."""

    wild = "*/" * depth
    template = raw_root.as_posix() + "/{subject}/" + wild + "*.dcm"

    phys_posix = [Path(p).as_posix() for p in phys_folders]

    converter = "dcm2niix" if use_nipype else "none"

    return [
        sys.executable,
        "-m",
        "heudiconv",
        "-d",
        template,
        "-s",
        *phys_posix,
        "-f",
        heuristic.as_posix(),
        "-c",
        converter,
        "-o",
        bids_out.as_posix(),
        "-b",
        "--minmeta",
        "--overwrite",
    ]


def _manual_convert(bids_root: Path, clean_id: str) -> None:
    """Run dcm2niix on filegroup.json results for *clean_id*."""
    info_dir = bids_root / ".heudiconv" / clean_id / "info"
    fg = info_dir / "filegroup.json"
    if not fg.exists():
        return

    with fg.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    for prefix, dicoms in mapping.items():
        if not dicoms:
            continue
        out_prefix = bids_root / prefix
        out_dir = out_prefix.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        dcm_dir = os.path.dirname(dicoms[0])
        cmd = [
            "dcm2niix",
            "-z", "y",
            "-b", "y",
            "-f", out_prefix.name,
            "-o", str(out_dir),
            dcm_dir,
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


def _parse_age(value: str) -> str:
    """Return numeric age from DICOM-style age strings (e.g. '032Y')."""
    m = re.match(r"(\d+)", str(value))
    if not m:
        return str(value)
    age = m.group(1).lstrip("0")
    return age or "0"


def write_participants(sub_df: pd.DataFrame, bids_root: Path) -> None:
    """Create or replace participants.tsv in *bids_root* using *sub_df*."""
    part_df = (
        sub_df[["BIDS_name", "GivenName", "PatientSex", "PatientAge"]]
        .drop_duplicates(subset=["BIDS_name"])
        .copy()
    )
    if part_df.empty:
        return

    part_df["PatientAge"] = part_df["PatientAge"].apply(_parse_age)
    part_df.rename(
        columns={
            "BIDS_name": "participant_id",
            "GivenName": "given_name",
            "PatientSex": "sex",
            "PatientAge": "age",
        },
        inplace=True,
    )
    part_df.to_csv(bids_root / "participants.tsv", sep="\t", index=False)


# ────────────────── main runner ──────────────────
def run_heudiconv(raw_root: Path,
                  heuristic: Path,
                  bids_out: Path,
                  per_folder: bool = True,
                  mapping_df: Optional[pd.DataFrame] = None,
                  use_nipype: bool = True) -> None:

    sid_map     = load_sid_map(heuristic)  # cleaned → sub-XXX
    clean2phys  = physical_by_clean(raw_root)
    cleaned_ids = sorted(sid_map.keys())

    missing = [c for c in cleaned_ids if c not in clean2phys]
    if missing:
        hint = (
            "Ensure --dicom-root matches the directory used to generate "
            "subject_summary.tsv"
        )
        raise KeyError(
            "Folders listed in heuristic not found under "
            f"{raw_root}: {', '.join(missing)}. {hint}"
        )

    phys_folders = [clean2phys[c] for c in cleaned_ids]

    depth = detect_depth(raw_root / phys_folders[0])

    print("Raw root    :", raw_root.as_posix())
    print("Heuristic   :", heuristic.as_posix())
    print("Output BIDS :", bids_out.as_posix())
    print("Folders     :", [Path(p).as_posix() for p in phys_folders])
    print("Depth       :", depth, "\n")

    bids_out.mkdir(parents=True, exist_ok=True)

    if per_folder:
        for phys in phys_folders:
            print(f"── {Path(phys).as_posix()} ──")
            cmd = heudi_cmd(raw_root, [phys], heuristic, bids_out, depth, use_nipype)
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            if not use_nipype:
                _manual_convert(bids_out, clean_name(phys))
                print("Re-applying heuristic to converted data …")
                subprocess.run(cmd, check=True)
            print()
    else:
        cmd = heudi_cmd(raw_root, phys_folders, heuristic, bids_out, depth, use_nipype)
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        if not use_nipype:
            for phys in phys_folders:
                _manual_convert(bids_out, clean_name(phys))
            print("Re-applying heuristic to converted data …")
            subprocess.run(cmd, check=True)

    if mapping_df is not None:
        dataset = bids_out.name
        mdir = bids_out / ".bids_manager"
        sub_df = mapping_df[mapping_df["StudyDescription"].fillna("").apply(safe_stem) == dataset]
        if not sub_df.empty:
            mdir.mkdir(exist_ok=True)
            sub_df.to_csv(mdir / "subject_summary.tsv", sep="\t", index=False)
            sub_df[["GivenName", "BIDS_name"]].drop_duplicates().to_csv(
                mdir / "subject_mapping.tsv", sep="\t", index=False
            )
            write_participants(sub_df, bids_out)


# ────────────────── CLI interface ──────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run HeuDiConv using one or more heuristics")
    parser.add_argument("dicom_root", help="Root directory containing DICOMs")
    parser.add_argument("heuristic", help="Heuristic file or directory with heuristic_*.py files")
    parser.add_argument("bids_out", help="Output BIDS directory")
    parser.add_argument("--subject-tsv", help="Path to subject_summary.tsv", default=None)
    parser.add_argument("--single-run", action="store_true", help="Use one heudiconv call for all subjects")
    parser.add_argument("--no-nipype", action="store_true", help="Skip Nipype and run dcm2niix separately")
    args = parser.parse_args()

    mapping_df = None
    if args.subject_tsv:
        mapping_df = pd.read_csv(args.subject_tsv, sep="\t")

    raw_root = Path(args.dicom_root).expanduser().resolve()
    heur_path = Path(args.heuristic).expanduser().resolve()
    bids_root = Path(args.bids_out).expanduser().resolve()

    heuristics = (
        [heur_path]
        if heur_path.is_file()
        else sorted(heur_path.glob("heuristic_*.py"))
    )

    for heur in heuristics:
        heur = heur.resolve()
        dataset = heur.stem.replace("heuristic_", "")
        out_dir = bids_root / dataset
        run_heudiconv(
            raw_root,
            heur,
            out_dir,
            per_folder=not args.single_run,
            mapping_df=mapping_df,
            use_nipype=not args.no_nipype,
        )


if __name__ == "__main__":
    main()

