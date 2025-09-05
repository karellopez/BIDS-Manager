"""Utilities for deriving BIDS compliant file names from the BIDS schema.

This module inspects the BIDS schema distributed with this project to map
arbitrary sequence descriptions to their canonical BIDS suffix.  The goal is to
avoid hard coded renaming rules and instead leverage the schema's knowledge of
valid suffixes and their human readable names.

The public function :func:`bidsify_sequence` accepts a free‑form sequence name
(e.g. ``"3D_T1-weighted"``) and returns the matching BIDS suffix (e.g.
``"T1w"``).  If no match is found the original sequence name is returned.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import yaml

# Location of the local copy of the BIDS schema
SCHEMA_ROOT = Path(__file__).parent / "miscellaneous" / "schema"


def _normalize(text: str) -> str:
    """Return a simplified version of *text* for fuzzy matching.

    The transformation removes all non alphanumeric characters and converts the
    string to lower case so that, for example, ``"T1-weighted"`` becomes
    ``"t1weighted"``.  This allows us to perform substring matching against the
    schema definitions.
    """
    return re.sub(r"[^0-9a-z]+", "", text.lower())


@lru_cache(None)
def _suffix_map() -> dict[str, str]:
    """Build a mapping of normalized synonyms → canonical suffix.

    The BIDS schema lists every valid suffix together with a human readable
    ``display_name``.  We normalise both the key itself, its ``value`` and the
    ``display_name`` (with generic words like "image" removed) and map each of
    those forms to the canonical ``value``.  The resulting dictionary enables
    us to look up a suffix purely based on textual matches.
    """
    suffix_file = SCHEMA_ROOT / "objects" / "suffixes.yaml"
    data = yaml.safe_load(suffix_file.read_text(encoding="utf-8"))

    mapping: dict[str, str] = {}
    for key, info in data.items():
        canonical = info.get("value", key)
        names = {key, canonical, info.get("display_name", "")}
        # Remove common trailing words from display names (image, data, map...)
        cleaned = re.sub(r"\b(image|data|map|file)\b", "", info.get("display_name", ""), flags=re.I)
        names.add(cleaned)
        for name in names:
            norm = _normalize(name)
            if norm:
                mapping[norm] = canonical

    # Longer keys first so that more specific matches win when used as a list
    return dict(sorted(mapping.items(), key=lambda kv: -len(kv[0])))


def bidsify_sequence(seq: str) -> str:
    """Return the BIDS suffix that best matches *seq*.

    Parameters
    ----------
    seq : str
        Arbitrary sequence description (for example DICOM ``SeriesDescription``)
        or an existing filename stem.

    Returns
    -------
    str
        Canonical BIDS suffix if a match is found, otherwise the original
        ``seq`` value.
    """
    norm_seq = _normalize(seq)
    for syn, canonical in _suffix_map().items():
        if syn in norm_seq:
            return canonical
    return seq
