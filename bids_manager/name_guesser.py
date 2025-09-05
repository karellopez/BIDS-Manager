"""Utility functions for guessing BIDS filenames.

This module loads suffix and entity information from the BIDS schema
(YAML files inside ``miscellaneous/schema/rules/files``) and exposes a
``guess_bids_name`` helper used by the GUI preview to suggest BIDS-like
file names. The goal is to provide sensible defaults that can be edited
by users before running the conversion pipeline.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Dict

# ``PyYAML`` is used to parse the BIDS schema at runtime.  Without it the
# suffix/entity map would be empty and filename suggestions would be
# essentially meaningless.  Importing it lazily gives a clear error if the
# dependency is missing instead of silently producing wrong names.
try:  # pragma: no cover - import guard exercised only when dependency missing
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - handled gracefully
    raise ModuleNotFoundError(
        "PyYAML is required for filename inference. Please install it to "
        "enable BIDS name suggestions."
    ) from exc

# ---------------------------------------------------------------------------
# Load suffix definitions from the BIDS schema
# ---------------------------------------------------------------------------

SCHEMA_DIR = Path(__file__).parent / "miscellaneous" / "schema" / "rules" / "files" / "raw"


def _build_suffix_map() -> Dict[str, Dict[str, object]]:
    """Parse YAML schema files to map suffixes to datatypes and entities."""
    suffix_map: Dict[str, Dict[str, object]] = {}
    if yaml is None:
        return suffix_map
    for yml in SCHEMA_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(yml.read_text()) or {}
        except Exception:
            continue
        for _name, spec in data.items():
            suffixes = spec.get("suffixes", [])
            datatypes = spec.get("datatypes", [])
            entities = spec.get("entities", {})
            datatype = datatypes[0] if datatypes else ""
            for suf in suffixes:
                suffix_map[suf] = {
                    "datatype": datatype,
                    "entities": entities,
                }
    return suffix_map


SUFFIX_MAP = _build_suffix_map()

# Precompiled regular expressions for common entities
ENTITY_REGEXES = {
    "task": re.compile(r"task-(?P<val>[^_]+)", re.I),
    "acq": re.compile(r"acq-(?P<val>[^_]+)", re.I),
    "run": re.compile(r"run-(?P<val>\d+)", re.I),
    "echo": re.compile(r"echo-(?P<val>\d+)", re.I),
    "dir": re.compile(r"dir-(?P<val>[^_]+)", re.I),
    "rec": re.compile(r"rec-(?P<val>[^_]+)", re.I),
    "ce": re.compile(r"ce-(?P<val>[^_]+)", re.I),
    "part": re.compile(r"part-(?P<val>[^_]+)", re.I),
}


def _extract_entities(seq: str, datatype: str, suffix: str) -> Dict[str, str]:
    """Extract entity values from ``seq`` based on known patterns."""
    entities: Dict[str, str] = {}
    for name, pattern in ENTITY_REGEXES.items():
        m = pattern.search(seq)
        if m:
            entities[name] = m.group("val")
    # For functional data, if no explicit ``task`` is present, derive it from
    # tokens that appear after the suffix.  The algorithm walks these tokens in
    # reverse order (nearest to the end first) and picks the first "human"
    # looking token—i.e., one that is not an entity tag and does not contain
    # digits which often represent acquisition parameters.
    if datatype == "func" and "task" not in entities:
        tokens = seq.split("_")
        try:
            idx = tokens.index(suffix)
        except ValueError:  # suffix not found; give up
            idx = -1
        candidates = tokens[idx + 1 :] if idx >= 0 else []
        for tok in reversed(candidates):
            if not tok or tok in SUFFIX_MAP:
                continue
            if any(tok.startswith(f"{e}-") for e in ENTITY_REGEXES):
                continue
            if any(ch.isdigit() for ch in tok):
                continue
            entities["task"] = tok
            break
    return entities


def guess_bids_name(subj: str, ses: str, seq: str, run: str | None = None) -> str:
    """Guess a BIDS compliant filename for a sequence.

    Parameters
    ----------
    subj, ses : str
        Subject and session identifiers (e.g., ``"sub-01"``).
    seq : str
        Original sequence name.
    run : str, optional
        Run number if available.
    """
    tokens = seq.split("_")
    suffix = ""
    datatype = ""
    # Identify the first token that matches a known BIDS suffix.  Matching is
    # case-insensitive and allows partial matches so that a sequence token like
    # "t1" will correctly map to the ``T1w`` suffix defined in the schema.
    for tok in tokens:
        tok_l = tok.lower()
        for suf, spec in SUFFIX_MAP.items():
            suf_l = suf.lower()
            if tok_l == suf_l or tok_l.startswith(suf_l) or suf_l.startswith(tok_l):
                suffix = suf
                datatype = str(spec.get("datatype", ""))
                break
        if suffix:
            break
    if not suffix:
        # Generic diffusion sequences often include ``diff`` or ``dwi`` in the
        # protocol name but not the exact ``dwi`` suffix.  Treat such cases as
        # diffusion images.
        for tok in tokens:
            tok_l = tok.lower()
            if tok_l.startswith("diff") or tok_l.startswith("dwi"):
                suffix = "dwi"
                datatype = str(SUFFIX_MAP.get("dwi", {}).get("datatype", ""))
                break
    if not suffix:
        # Fallback to the last token if nothing matched at all
        suffix = tokens[-1]
    entities: Dict[str, str] = {}
    if datatype == "dwi" and suffix.lower() not in {"dwi", "sbref"}:
        # Scanner-generated diffusion derivatives (e.g., ``ADC``) should be
        # treated as ``desc-<suffix>`` files with a ``dwi`` suffix.
        entities["desc"] = suffix
        suffix = "dwi"
    entities.update(_extract_entities(seq, datatype, suffix))
    if run and run.strip() and "run" not in entities:
        entities["run"] = run.strip()

    parts = [subj]
    if ses:
        parts.append(ses)
    for key in ["task", "acq", "ce", "rec", "dir", "run", "echo", "part", "desc"]:
        val = entities.get(key)
        if val:
            parts.append(f"{key}-{val}")
    parts.append(suffix)
    return "_".join(parts) + ".nii.gz"
