#!/usr/bin/env python3
"""Build dcm2bids JSON config files from a DICOM inventory TSV.

This is the dcm2bids analogue of :mod:`bids_manager.build_heuristic_from_tsv`.
The two builders share the same naming contract via
:func:`bids_manager.schema_renamer.build_preview_names`, so a TSV row produces
the same BIDS basename regardless of which conversion engine downstream
consumes it.

Output: one ``dcm2bids_config_<study>.json`` per StudyDescription. Each
included TSV row becomes a single ``descriptions`` entry whose ``criteria``
matches on the row's ``SeriesInstanceUID`` — unambiguous, one-to-one, and
collision-free across subjects in the same study because SeriesInstanceUIDs
are globally unique.

Excluded rows (``include == 0``) are simply omitted from the config; the
underlying dcm2niix call will still process those series but dcm2bids will
not move them into the BIDS layout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from bids_manager.schema_renamer import (
        DEFAULT_SCHEMA_DIR,
        SKIP_MODALITIES,
        SeriesInfo,
        build_preview_names,
        load_bids_schema,
        normalize_study_name,
    )
except Exception:
    from schema_renamer import (  # type: ignore
        DEFAULT_SCHEMA_DIR,
        SKIP_MODALITIES,
        SeriesInfo,
        build_preview_names,
        load_bids_schema,
        normalize_study_name,
    )


def _safe_stem(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", str(text).strip()).strip("_") or "unknown"


def _normalize_subject(bids_name: str) -> str:
    s = str(bids_name or "").strip()
    if s.lower().startswith("sub-"):
        s = s[4:]
    return re.sub(r"[^0-9A-Za-z]+", "", s)


def _normalize_session(session: str) -> str:
    s = str(session or "").strip()
    if s.lower().startswith("ses-"):
        s = s[4:]
    return re.sub(r"[^0-9A-Za-z]+", "", s)


def _split_base(base: str, subject: str, session: str) -> Tuple[str, str]:
    """Split a final BIDS basename into (custom_entities, suffix).

    ``base`` is what :func:`build_preview_names` returns, e.g.
    ``sub-001_ses-pre_task-rest_run-2_bold``. dcm2bids prepends sub/ses
    automatically and takes the suffix as a separate field, so we strip
    those parts and return the entities in between plus the trailing suffix.
    """
    tokens = [t for t in base.split("_") if t]
    sub_tag = f"sub-{subject}" if subject else None
    ses_tag = f"ses-{session}" if session else None
    while tokens and (tokens[0] == sub_tag or tokens[0] == ses_tag):
        tokens.pop(0)
    if not tokens:
        return "", ""
    suffix = tokens.pop()
    custom_entities = "_".join(tokens)
    return custom_entities, suffix


def _detect_dwi_derivative(sequence: str) -> Optional[str]:
    s = (sequence or "").lower()
    if "colfa" in s:
        return "ColFA"
    if "fa" in s and "colfa" not in s:
        return "FA"
    if "tensor" in s:
        return "TENSOR"
    if "adc" in s:
        return "ADC"
    if "trace" in s or "tracew" in s:
        return "TRACE"
    return None


def _row_to_description(row: pd.Series, rep_num: int, rep_count: int,
                       schema, only_last_repeated: bool) -> Optional[Dict]:
    """Convert a single TSV row into a dcm2bids description dict.

    Returns ``None`` when the row should be skipped (e.g. only_last_repeated
    drops earlier repetitions, or the SeriesInstanceUID is missing).
    """
    if only_last_repeated and rep_count > 1 and rep_num != rep_count:
        return None

    series_uid = str(row.get("series_uid", "") or "").split("|")[0].strip()
    if not series_uid:
        # Without a UID we cannot match this series uniquely; let
        # post-conversion fixups deal with it manually if it shows up.
        return None

    subj = _normalize_subject(row["BIDS_name"])
    session_raw = row.get("session", "")
    if pd.isna(session_raw):
        session_raw = ""
    ses = _normalize_session(session_raw)

    sequence = str(row.get("sequence", ""))
    modality = str(row.get("modality", ""))

    extra: Dict[str, str] = {}
    for key in ("task", "task_hits", "acq", "run", "dir", "echo"):
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if str(val).strip():
            extra[key] = str(val)

    series = SeriesInfo(
        subject=subj,
        session=ses or None,
        modality=modality,
        sequence=sequence,
        rep=(None if only_last_repeated else int(rep_num) if rep_count > 1 else None),
        extra=extra,
    )
    datatype, base = build_preview_names([series], schema)[0][1:]

    custom_entities, suffix = _split_base(base, subj, ses)

    is_derivative = _detect_dwi_derivative(sequence) is not None
    if is_derivative:
        # dcm2bids has no native derivatives output path. We route the
        # series into the regular dwi/ folder; ``post_conv_renamer`` /
        # ``schema_renamer.apply_post_conversion_rename`` later relocates
        # it under derivatives/<pipeline>/sub-/ses-/dwi/.
        datatype = "dwi"

    description: Dict = {
        "datatype": datatype,
        "suffix": suffix,
        "criteria": {"SeriesInstanceUID": series_uid},
    }
    if custom_entities:
        description["custom_entities"] = custom_entities

    # Preserve the source row identity so downstream post-processing can
    # cross-reference dcm2bids descriptions with the inventory TSV.
    description["id"] = f"row_{int(row.get('_row_index', -1))}"
    return description


def write_config(df: pd.DataFrame, dst: Path, only_last_repeated: bool = False) -> None:
    """Write a dcm2bids JSON config to ``dst`` from a DataFrame slice."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)

    # Track repetition counts the same way build_heuristic_from_tsv does so
    # the rep-N suffix matches across both engines.
    rep_counts = (
        df.groupby(["BIDS_name", "session", "sequence"], dropna=False)["sequence"]
        .transform("count")
    )
    rep_index = (
        df.groupby(["BIDS_name", "session", "sequence"], dropna=False).cumcount() + 1
    )

    descriptions: List[Dict] = []
    for idx, row in df.iterrows():
        row = row.copy()
        row["_row_index"] = idx
        desc = _row_to_description(
            row,
            rep_num=int(rep_index.loc[idx]),
            rep_count=int(rep_counts.loc[idx]),
            schema=schema,
            only_last_repeated=only_last_repeated,
        )
        if desc is not None:
            descriptions.append(desc)

    config = {
        # Preserve dcm2niix flags BIDS Manager has used historically. ``-b y``
        # is implicit but listing it makes the config self-documenting.
        "dcm2niixOptions": "-b y -ba n -z y -f '%3s_%f_%p_%t'",
        # Default duplicate-handling: dcm2bids will append _dup-NN; we will
        # not encounter duplicates because criteria match unique
        # SeriesInstanceUIDs, but the field is required by some versions.
        "dup_method": "dup",
        "search_method": "fnmatch",
        "case_sensitive": True,
        "compKeys": ["SeriesNumber", "AcquisitionTime", "SeriesInstanceUID"],
        "descriptions": descriptions,
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"dcm2bids config written → {dst} ({len(descriptions)} descriptions)")


def generate(tsv: Path, out_dir: Path, only_last_repeated: bool = False) -> List[Path]:
    """Generate one dcm2bids config per StudyDescription in ``tsv``."""

    df = pd.read_csv(tsv, sep="\t", keep_default_na=False)

    if "StudyDescription" in df.columns:
        df["StudyDescription"] = df["StudyDescription"].apply(normalize_study_name)

    mask = df.modality.isin(SKIP_MODALITIES)
    if mask.any():
        df.loc[mask, "include"] = 0
        print(
            f"Auto-skipped {mask.sum()} rows "
            f"({', '.join(sorted(SKIP_MODALITIES))})"
        )

    df = df[df["include"] == 1]
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for study, sub_df in df.groupby("StudyDescription"):
        fname = _safe_stem(study or "unknown")
        cfg = out_dir / f"dcm2bids_config_{fname}.json"
        write_config(sub_df, cfg, only_last_repeated)
        written.append(cfg)
    return written


def main() -> None:
    """CLI entry point for the ``build-dcm2bids-config`` command."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate dcm2bids JSON config(s) from a DICOM inventory TSV"
    )
    parser.add_argument("tsv", help="Path to subject_summary.tsv")
    parser.add_argument("out_dir", help="Directory to write config files")
    parser.add_argument(
        "--only-last-repeated",
        action="store_true",
        help="Only keep the last repetition of repeated sequences",
    )
    args = parser.parse_args()
    generate(Path(args.tsv), Path(args.out_dir), args.only_last_repeated)


if __name__ == "__main__":
    main()
