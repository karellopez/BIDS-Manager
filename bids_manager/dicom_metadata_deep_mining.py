#!/usr/bin/env python3
"""DICOM metadata deep mining utility.

This script recursively scans a directory tree for DICOM files and writes
all metadata fields exposed by :mod:`pydicom` into a TSV table. Each row
in the output corresponds to a single DICOM file, and each column
represents a metadata field encountered across the dataset.

It complements :mod:`dicom_inventory` by providing an exhaustive view of
the headers rather than a curated subset.

Example
-------
Run from the command line using either the entry point or the module::

    dicom-metadata-deep-mining /path/to/dicoms output.tsv

    python -m bids_manager.dicom_metadata_deep_mining /path/to/dicoms output.tsv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pydicom
from pydicom.multival import MultiValue

# Reuse helper from dicom_inventory to recognise DICOM files
from bids_manager.dicom_inventory import is_dicom_file


def _ds_to_dict(ds: pydicom.Dataset) -> Dict[str, str]:
    """Convert a :class:`pydicom.Dataset` into a flat dictionary.

    Nested sequences are flattened using dotted keys (e.g. ``Seq[0].Field``).
    Values are coerced to strings suitable for TSV output.
    """

    info: Dict[str, str] = {}
    for elem in ds:
        key = elem.keyword or str(elem.tag)
        value = elem.value
        if isinstance(value, MultiValue):
            value = ';'.join(str(v) for v in value)
        elif isinstance(value, (bytes, bytearray)):
            value = value.hex()
        elif isinstance(value, pydicom.dataset.Dataset):
            nested = _ds_to_dict(value)
            for k, v in nested.items():
                info[f"{key}.{k}"] = v
            continue
        elif isinstance(value, pydicom.sequence.Sequence):
            for i, item in enumerate(value):
                nested = _ds_to_dict(item)
                for k, v in nested.items():
                    info[f"{key}[{i}].{k}"] = v
            continue
        info[key] = str(value)
    return info


def scan_dicoms_all_fields(dicom_dir: str, output_tsv: str | None = None) -> pd.DataFrame:
    """Scan *dicom_dir* recursively and extract all metadata fields."""

    rows: List[Dict[str, str]] = []
    all_keys: set[str] = set()
    root = Path(dicom_dir)

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            fpath = Path(dirpath) / name
            if not is_dicom_file(str(fpath)):
                continue
            try:
                ds = pydicom.dcmread(str(fpath), stop_before_pixels=True, force=True)
            except Exception:
                continue
            meta = _ds_to_dict(ds)
            record = {"file": fpath.relative_to(root).as_posix(), **meta}
            rows.append(record)
            all_keys.update(meta.keys())

    df = pd.DataFrame(rows)
    columns = ['file'] + sorted(all_keys)
    df = df.reindex(columns=columns)
    if output_tsv:
        df.to_csv(output_tsv, sep='	', index=False)
    return df


def main(argv: List[str] | None = None) -> None:
    """CLI entry point."""

    import argparse

    parser = argparse.ArgumentParser(
        description='Extract all DICOM metadata fields into a TSV table'
    )
    parser.add_argument('dicom_dir', help='Directory containing DICOM files')
    parser.add_argument('output_tsv', help='Path of the TSV file to create')
    args = parser.parse_args(argv)

    scan_dicoms_all_fields(args.dicom_dir, args.output_tsv)


if __name__ == '__main__':  # pragma: no cover
    main()
