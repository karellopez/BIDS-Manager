"""Utilities to interact with the bundled BIDS schema.

This module provides helper functions that load the machine-readable
BIDS schema distributed with *BIDS-Manager*.  Only a very small subset
of the schema is required by the application: the list of valid file
suffixes and the canonical order of entities.  These are sufficient to
build or validate BIDS-style file names.

The schema files are taken from ``bids_manager/miscellaneous/schema``
and were copied from the official BIDS specification.
"""
from __future__ import annotations

from pathlib import Path
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is a runtime dependency
    yaml = None  # type: ignore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

SCHEMA_DIR = Path(__file__).resolve().parent / "miscellaneous" / "schema"
_SUFFIXES_FILE = SCHEMA_DIR / "objects" / "suffixes.yaml"
_ENTITIES_FILE = SCHEMA_DIR / "objects" / "entities.yaml"
_ENTITY_ORDER_FILE = SCHEMA_DIR / "rules" / "entities.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file from ``path`` using PyYAML.

    Parameters
    ----------
    path : Path
        YAML file to parse.
    """
    if yaml is None:  # pragma: no cover - handled at runtime
        raise RuntimeError("PyYAML is required to parse schema files")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _schema_entities() -> Tuple[List[str], Dict[str, str]]:
    """Return entity order and mapping of long to short names.

    Returns
    -------
    tuple
        ``(order, long_to_short)`` where ``order`` is a list of entity short
        names in the canonical order defined by the BIDS specification, and
        ``long_to_short`` maps the long entity names to their short forms
        (for example ``{"subject": "sub"}``).
    """
    raw_entities = _load_yaml(_ENTITIES_FILE)
    long_to_short = {long: data["name"] for long, data in raw_entities.items()}
    order_long: Iterable[str] = _load_yaml(_ENTITY_ORDER_FILE)
    order_short = [long_to_short.get(e, e) for e in order_long]
    return order_short, long_to_short


@lru_cache(maxsize=1)
def _valid_suffixes() -> List[str]:
    """Return the list of valid BIDS suffixes."""
    data = _load_yaml(_SUFFIXES_FILE)
    return [info.get("value", key) for key, info in data.items()]


ENTITY_RE = re.compile(r"([A-Za-z0-9]+)-([A-Za-z0-9]+)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_valid_bids_name(name: str) -> bool:
    """Return ``True`` if ``name`` looks like a valid BIDS filename.

    This is a *lightweight* check that validates the ordering of entities
    and the suffix against the official BIDS schema bundled with the
    package.  It does **not** perform a full specification validation but
    helps avoid obvious mistakes during renaming.
    """
    entities, suffix = _parse_name(name)
    if not suffix or suffix not in _valid_suffixes():
        return False
    order, _ = _schema_entities()
    known = set(order)
    # Ensure all entities are known
    if any(e not in known for e in entities):
        return False
    # Ensure entity order follows the canonical sequence
    last = -1
    index = {e: i for i, e in enumerate(order)}
    for e in entities:
        i = index[e]
        if i < last:
            return False
        last = i
    return True


def _parse_name(name: str) -> Tuple[List[str], str | None]:
    """Parse ``name`` into entity list and suffix.

    Parameters
    ----------
    name : str
        Filename to parse.  The extension is ignored.
    """
    base = Path(name).name
    if "." in base:
        base = base.split(".", 1)[0]
    parts = base.split("_")
    entities: List[str] = []
    suffix: str | None = None
    for part in parts:
        m = ENTITY_RE.fullmatch(part)
        if m:
            key, _ = m.groups()
            entities.append(key)
        else:
            suffix = part
    return entities, suffix


def build_bids_name(entities: Dict[str, str], suffix: str, ext: str) -> str:
    """Construct a BIDS filename from ``entities`` and ``suffix``.

    Parameters
    ----------
    entities : dict
        Mapping of entity short names (for example ``{"sub": "01"}``).
    suffix : str
        BIDS suffix such as ``"bold"``.
    ext : str
        File extension including leading dot, for example ``".nii.gz"`` or
        ``".json"``.
    """
    order, _ = _schema_entities()
    parts = [f"{e}-{entities[e]}" for e in order if e in entities]
    parts.append(suffix)
    return "_".join(parts) + ext

