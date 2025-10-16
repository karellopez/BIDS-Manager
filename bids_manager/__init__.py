"""BIDS Manager package."""

from importlib import metadata

# Expose the schema-driven renaming helpers at the package root for backwards
# compatibility.  The implementation lives in ``bids_manager.schema_renamer``
# but callers historically imported the public API straight from
# :mod:`bids_manager`, so continue to re-export the key symbols here.
from .schema_renamer import (
    DEFAULT_SCHEMA_DIR,
    DERIVATIVES_PIPELINE_NAME,
    ENABLE_DWI_DERIVATIVES_MOVE,
    ENABLE_FIELDMap_NORMALIZATION,
    ENABLE_SCHEMA_RENAMER,
    SchemaInfo,
    SeriesInfo,
    apply_post_conversion_rename,
    build_preview_names,
    load_bids_schema,
    normalize_study_name,
    propose_bids_basename,
)

__all__ = [
    "__version__",
    "DEFAULT_SCHEMA_DIR",
    "DERIVATIVES_PIPELINE_NAME",
    "ENABLE_DWI_DERIVATIVES_MOVE",
    "ENABLE_FIELDMap_NORMALIZATION",
    "ENABLE_SCHEMA_RENAMER",
    "SchemaInfo",
    "SeriesInfo",
    "apply_post_conversion_rename",
    "build_preview_names",
    "load_bids_schema",
    "normalize_study_name",
    "propose_bids_basename",
]

try:  # pragma: no cover - version resolution
    __version__ = metadata.version("bids-manager")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

