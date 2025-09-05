"""Utilities to derive BIDS-compliant names from the bundled schema.

This module parses the BIDS schema distributed with the application in order to
provide a mapping between arbitrary textual descriptions (for example DICOM
``SeriesDescription`` values) and the canonical BIDS suffixes.  The goal is to
make renaming decisions data driven rather than relying on hard coded lists of
keywords.

Only a subset of the schema is required here.  The file
``miscellaneous/schema/objects/suffixes.yaml`` lists every valid suffix in BIDS
together with a human readable ``display_name``.  We load this file and create a
lookup table of *synonyms* → *suffix* where the synonyms are derived from the
following sources for each entry:

* the key of the entry itself,
* the ``value`` field (canonical suffix), and
* a simplified version of the ``display_name`` with common words such as
  ``image`` or ``map`` removed.

The lookup is intentionally permissive and normalises all strings by removing
non alphanumeric characters so that inputs like ``"3D_T1-weighted"`` correctly
match ``T1w``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml


# Paths ---------------------------------------------------------------------
SCHEMA_ROOT = Path(__file__).resolve().parent / "miscellaneous" / "schema"
SUFFIX_FILE = SCHEMA_ROOT / "objects" / "suffixes.yaml"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _normalise(text: str, sep: str = "") -> str:
    """Return ``text`` lower-cased with non alphanumerics stripped.

    Parameters
    ----------
    text : str
        Input string to normalise.
    sep : str, optional
        Replacement for non alphanumeric characters.  An empty string removes
        them entirely while ``"_"`` keeps word boundaries separated by
        underscores.  ``sep`` is not appended to the beginning or end of the
        result.
    """

    cleaned = re.sub(r"[^0-9A-Za-z]+", sep, str(text)).strip(sep)
    return cleaned.lower()


def _synonyms(entry: Dict[str, str]) -> set[str]:
    """Return a set of synonyms for a suffix entry.

    ``entry`` originates from the YAML schema.  The function derives a number of
    normalised strings that could be encountered in free text descriptions.  In
    addition to the canonical ``value`` and the dictionary key, the
    ``display_name`` is simplified by removing terms that are not helpful for
    matching (such as "image" or "map").
    """

    syns: set[str] = set()
    value = entry.get("value")
    display = entry.get("display_name", "")

    for cand in filter(None, [value]):
        syns.add(_normalise(cand))

    # Remove common trailing words which do not aid matching
    disp_clean = re.sub(r"\b(image|map|file|series)\b", "", display, flags=re.I)
    for cand in filter(None, [display, disp_clean]):
        syns.add(_normalise(cand))
        syns.add(_normalise(cand, sep="_"))
    return syns


@lru_cache()
def load_suffix_map() -> Dict[str, str]:
    """Load suffix definitions and return synonym → canonical suffix mapping."""

    with open(SUFFIX_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    mapping: Dict[str, str] = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        entry.setdefault("value", str(key))
        suffix = entry["value"]
        # Generate synonyms from key, value and display name
        syns = _synonyms({"value": suffix, "display_name": entry.get("display_name", "")})
        syns.add(_normalise(key))
        for syn in syns:
            # Only record first occurrence to avoid overriding more specific
            # matches with later entries
            mapping.setdefault(syn, suffix)
    return mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_suffix(name: str) -> str | None:
    """Return the BIDS suffix matching ``name`` if any.

    The function performs a normalised substring search across all synonyms
    derived from the schema.  If a match is found the canonical suffix value is
    returned, otherwise ``None`` is yielded.
    """

    norm = _normalise(name)
    if not norm:
        return None
    mapping = load_suffix_map()
    for syn, suffix in mapping.items():
        if syn and syn in norm:
            return suffix
    return None


def bidsify_stem(name: str) -> str:
    """Return a BIDS-compliant stem for ``name``.

    The function first attempts to match the input to a known suffix using the
    schema.  If no match is found the input is sanitised so that it can still be
    used in a filename, preserving the previous behaviour.
    """

    match = match_suffix(name)
    if match:
        return match
    # Fallback to a safe stem resembling the original text
    return _normalise(name, sep="_")

