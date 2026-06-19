"""BIDS Manager-native validation supplements (Qt-free).

Validation findings come from the standalone :mod:`bidsval` package. These are
the two things bidsval does not produce that BIDS Manager's Editor still needs:

* :func:`sidecar_fields_for` - the schema-aware *form* data the sidecar editor
  renders (one row per schema-defined field at a ``(datatype, suffix)`` plus any
  extra keys the file carries). This is form scaffolding, not a finding; it
  drives the Editor's sidecar form and the HTML report's schema-audit table.
* :func:`todo_issues_for` - TODO-placeholder findings. A BIDS Manager
  convention: the metadata engine writes the literal string ``"TODO"`` into
  missing recommended fields, and the Editor flags those so the user fills them
  in. bidsval (correctly) does not know about this convention.

Both are derived straight from :mod:`bidsmgr.schema`, the same keystone the rest
of the package reads from. No Qt here (architectural rule 2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .. import schema as schema_mod
from .types import FieldLevel, Issue, Severity, SidecarField

# Recognised BIDS datatype directory names, used to infer a file's datatype from
# its path the same way the old in-house validator did.
_BIDS_DATATYPE_NAMES: tuple[str, ...] = (
    "anat", "func", "dwi", "fmap", "perf",
    "meg", "eeg", "ieeg", "beh", "pet", "micr", "nirs",
)

# Multi-part extensions stripped (longest first) to recover the BIDS suffix.
_KNOWN_EXTS: tuple[str, ...] = (
    ".nii.gz", ".tsv.gz", ".nii", ".json", ".tsv", ".bval", ".bvec",
    ".edf", ".bdf", ".set", ".vhdr", ".fif", ".fif.gz",
)


# Data-file extensions whose metadata lives in a sibling ``.json`` sidecar.
# Longest-first so multi-part extensions strip correctly. Used to find the
# editable sidecar for a finding attached to a data file.
_DATA_EXTS: tuple[str, ...] = (
    ".nii.gz", ".nii", ".fif.gz", ".fif", ".edf", ".bdf", ".gdf", ".set",
    ".vhdr", ".cnt", ".con", ".sqd", ".kdf", ".ds", ".mff", ".mef", ".nwb",
)


def _canonical(name: str) -> str:
    """``EchoTime__fmap`` -> ``EchoTime`` (strip the schema's context suffix)."""
    return name.split("__", 1)[0]


def sibling_json(path: Path) -> Optional[Path]:
    """The ``.json`` sidecar path for a data file (``*_bold.nii.gz`` ->
    ``*_bold.json``), or ``None`` if ``path`` is not a recognised data file.

    Purely a name transform - the caller checks whether the sidecar exists.
    """
    name = path.name
    lower = name.lower()
    for ext in _DATA_EXTS:
        if lower.endswith(ext):
            return path.with_name(name[: -len(ext)] + ".json")
    return None


def value_kind(value: object) -> str:
    """Classify a parsed JSON value for the sidecar form's value column."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value == "TODO":
        return "todo"
    return "string"


def infer_datatype_suffix(
    path: Path, bids_root: Path,
) -> tuple[Optional[str], Optional[str]]:
    """Best-effort ``(datatype, suffix)`` for a file from its path + name.

    Datatype is the first recognised datatype directory on the path; suffix is
    the last underscore-separated token of the basename with no ``-`` (so it is
    not an entity). Mirrors the old in-house validator so downstream consumers
    (HTML report typed annotation, sidecar form) see the same values.
    """
    try:
        rel_parts = path.relative_to(bids_root).parts
    except ValueError:
        rel_parts = path.parts

    datatype: Optional[str] = None
    for part in rel_parts:
        if part in _BIDS_DATATYPE_NAMES:
            datatype = part
            break

    stem = path.name
    for ext in _KNOWN_EXTS:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    suffix: Optional[str] = None
    for tok in reversed(stem.split("_")):
        if tok and "-" not in tok:
            suffix = tok
            break
    return datatype, suffix


def _load_json_object(fp: Path) -> Optional[dict]:
    """Parse ``fp`` as JSON, returning the object or ``None`` (unreadable / not
    an object). Errors are bidsval's job to report; here we just need the data."""
    try:
        data = json.loads(Path(fp).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def sidecar_fields_for(
    fp: Path, datatype: Optional[str], suffix: Optional[str],
) -> list[SidecarField]:
    """Build the schema-aware sidecar form rows for a JSON file.

    Returns one :class:`SidecarField` per schema-defined field at
    ``(datatype, suffix)`` (required + recommended + optional + deprecated),
    each flagged present/missing against the file's actual contents, plus any
    extra keys the file carries (as OPTIONAL). Returns ``[]`` for non-JSON
    files or untyped sidecars (the form falls back to a plain on-disk read).
    """
    if not str(fp).lower().endswith(".json"):
        return []
    if not (datatype and suffix):
        return []
    data = _load_json_object(fp)
    if data is None:
        return []

    seen: set[str] = set()
    fields: list[SidecarField] = []

    def _add_level(level: FieldLevel, getter) -> None:
        try:
            schema_fields = getter(datatype, suffix)
        except (KeyError, ValueError, AttributeError):
            schema_fields = []
        for fi in schema_fields or []:
            name = _canonical(fi.name)
            if name in seen:
                continue
            seen.add(name)
            present = name in data
            value = data.get(name) if present else None
            fields.append(SidecarField(
                level=level, name=name, value=value, present=present,
                value_kind=value_kind(value) if present else "missing",
                description=getattr(fi, "description", None),
            ))

    _add_level(FieldLevel.REQUIRED, schema_mod.required_sidecar_fields)
    _add_level(FieldLevel.RECOMMENDED, schema_mod.recommended_sidecar_fields)
    _add_level(FieldLevel.OPTIONAL, schema_mod.optional_sidecar_fields)
    _add_level(FieldLevel.DEPRECATED, schema_mod.deprecated_sidecar_fields)

    # Keys present in the file but absent from every schema list -> OPTIONAL.
    for k, v in data.items():
        if k in seen:
            continue
        seen.add(k)
        fields.append(SidecarField(
            level=FieldLevel.OPTIONAL, name=k, value=v, present=True,
            value_kind=value_kind(v),
        ))
    return fields


def todo_issues_for(fp: Path) -> list[Issue]:
    """Flag every field whose value is the literal placeholder ``"TODO"``.

    Works on any JSON object (typed sidecar or top-level file such as
    ``dataset_description.json``). The fix carries the field name so the
    Editor's fix button jumps straight to that row in the sidecar form.
    """
    if not str(fp).lower().endswith(".json"):
        return []
    data = _load_json_object(fp)
    if data is None:
        return []
    out: list[Issue] = []
    for key, value in data.items():
        if value == "TODO":
            out.append(Issue(
                severity=Severity.WARN,
                rule_id="bidsmgr.todo_placeholder",
                message=f"field {key!r} contains a TODO placeholder",
                field=key,
                fix_label="Set a real value",
                fix_action="set_field",
            ))
    return out


__all__ = [
    "value_kind",
    "infer_datatype_suffix",
    "sidecar_fields_for",
    "todo_issues_for",
    "sibling_json",
]
