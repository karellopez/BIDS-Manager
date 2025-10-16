"""Public wrapper for the schema-based renaming helpers."""

from __future__ import annotations

try:
    from ._renaming_loader import load_module
except ImportError:  # pragma: no cover - fallback for direct execution
    # When this module is imported from a checkout via ``python path/to/script``
    # the ``bids_manager`` package is not initialised, so relative imports fail.
    # Importing the helper directly keeps the lightweight wrappers usable in
    # that scenario (e.g. when ``build_heuristic_from_tsv`` falls back to
    # absolute imports during development).
    from _renaming_loader import load_module  # type: ignore

_impl = load_module("schema_renamer")

# Copy the main public API to this module's namespace.  Callers continue to
# import from :mod:`bids_manager.schema_renamer` while the implementation lives
# in ``bids_manager/renaming/schema_renamer.py``.
SequenceHint = getattr(_impl, "SequenceHint")
DEFAULT_SEQUENCE_HINTS = getattr(_impl, "DEFAULT_SEQUENCE_HINTS")
SEQUENCE_HINTS = getattr(_impl, "SEQUENCE_HINTS")
SchemaInfo = getattr(_impl, "SchemaInfo")
SeriesInfo = getattr(_impl, "SeriesInfo")
load_bids_schema = getattr(_impl, "load_bids_schema")
build_preview_names = getattr(_impl, "build_preview_names")
propose_bids_basename = getattr(_impl, "propose_bids_basename")
apply_post_conversion_rename = getattr(_impl, "apply_post_conversion_rename")
guess_modality = getattr(_impl, "guess_modality")
modality_to_container = getattr(_impl, "modality_to_container")
get_sequence_hint_patterns = getattr(_impl, "get_sequence_hint_patterns")
set_sequence_hint_patterns = getattr(_impl, "set_sequence_hint_patterns")
restore_default_sequence_hints = getattr(_impl, "restore_default_sequence_hints")
SKIP_MODALITIES = getattr(_impl, "SKIP_MODALITIES")

_exported = getattr(_impl, "__all__", ())
if _exported:
    __all__ = tuple(_exported)
else:
    __all__ = (
        "SchemaInfo",
        "SeriesInfo",
        "load_bids_schema",
        "build_preview_names",
        "propose_bids_basename",
        "apply_post_conversion_rename",
    )

