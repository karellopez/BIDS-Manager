from __future__ import annotations

"""Light-weight access to the bundled BIDS schema.

This module loads the YAML files shipped under ``miscellaneous/schema`` and
exposes helpers to query entities, suffixes and their ordering. It also
provides :func:`parse_filename` and :func:`build_filename` utilities for working
with BIDS file names.
"""

from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple
import os
import yaml

SCHEMA_ROOT = Path(__file__).resolve().parent / "miscellaneous" / "schema"


@lru_cache
def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def get_entities() -> Dict[str, str]:
    """Return mapping of short entity labels to canonical names."""
    data = _load_yaml(SCHEMA_ROOT / "objects" / "entities.yaml")
    return {info["name"]: key for key, info in data.items()}


@lru_cache
def _canonical_to_short() -> Dict[str, str]:
    data = get_entities()
    return {canonical: short for short, canonical in data.items()}


@lru_cache
def get_suffixes() -> set[str]:
    """Return set of valid suffixes."""
    data = _load_yaml(SCHEMA_ROOT / "objects" / "suffixes.yaml")
    return {info["value"] for info in data.values()}


@lru_cache
def get_entity_order() -> List[str]:
    """Return canonical entity ordering (short names)."""
    canonical = _load_yaml(SCHEMA_ROOT / "rules" / "entities.yaml")
    c2s = _canonical_to_short()
    return [c2s.get(name, name) for name in canonical]


def _splitext(name: str) -> Tuple[str, str]:
    if name.endswith(".nii.gz"):
        return name[:-7], ".nii.gz"
    root, ext = os.path.splitext(name)
    return root, ext


def parse_filename(filename: str, *, strict: bool = True) -> Dict[str, object]:
    """Parse a BIDS ``filename`` into entities, suffix and extension.

    Parameters
    ----------
    filename : str
        Base file name or path.
    strict : bool, optional
        If ``True`` (default) validate suffix against the schema.
    """
    name = Path(filename).name
    stem, ext = _splitext(name)
    tokens = stem.split("_")
    entities: Dict[str, str] = {}
    suffix: str | None = None

    order_map = {e: i for i, e in enumerate(get_entity_order())}
    last_idx = -1

    for token in tokens:
        if suffix is None:
            if "-" in token:
                key, val = token.split("-", 1)
                entities[key] = val
                if key in order_map:
                    idx = order_map[key]
                    if idx < last_idx:
                        raise ValueError("Entities out of order")
                    last_idx = idx
            else:
                suffix = token
        else:
            if "-" in token:
                key, val = token.split("-", 1)
                entities[key] = val
            else:
                raise ValueError("Unexpected token after suffix")

    if suffix is None:
        raise ValueError("Missing suffix")
    if strict and suffix not in get_suffixes():
        raise ValueError(f"Invalid suffix: {suffix}")

    return {"entities": entities, "suffix": suffix, "extension": ext}


def build_filename(
    entities: Dict[str, str],
    suffix: str,
    extension: str = "",
    *,
    strict: bool = True,
) -> str:
    """Build a BIDS file name from ``entities`` and ``suffix``.

    Parameters
    ----------
    entities : dict
        Mapping of entity short names to values.
    suffix : str
        File suffix (e.g., ``T1w`` or ``bold``).
    extension : str, optional
        File extension including dot (e.g., ``.nii.gz``). If empty, no
        extension is appended.
    strict : bool, optional
        Validate ``suffix`` against the schema (default ``True").
    """
    if strict and suffix not in get_suffixes():
        raise ValueError(f"Invalid suffix: {suffix}")

    order = get_entity_order()
    known = [f"{key}-{entities[key]}" for key in order if key in entities]
    unknown = [f"{k}-{entities[k]}" for k in sorted(entities) if k not in order]
    parts = known + [suffix] + unknown

    name = "_".join(parts)
    if extension:
        if not extension.startswith("."):
            extension = "." + extension
        name += extension
    return name
