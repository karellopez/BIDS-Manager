"""bidsval - a schema-driven, pydantic-typed, in-process BIDS validator.

The public surface grows as the validator does. Today it exposes the two pieces
that the rest of the engine is built on:

* the schema resolver (:func:`bidsval.schema.resolve`), the single place that
  turns a schema selector into one in-memory schema object, and
* the expression evaluator (:func:`bidsval.expr.evaluate_string`), which runs a
  BIDS schema expression against a context.

Result types (:class:`~bidsval.issues.Issue`, :class:`~bidsval.report.ValidationReport`)
are re-exported here so consumers can ``from bidsval import Issue, ValidationReport``.
"""

from __future__ import annotations

from .expr import evaluate_string
from .issues import DatasetIssues, Issue, Severity
from .report import FileVerdict, ValidationReport
from .schema import available_versions, bids_version, resolve, schema_version
from .validate import validate, validate_file, validate_subject

try:  # populated from package metadata once installed
    from importlib.metadata import version

    __version__ = version("bidsval")
except Exception:  # pragma: no cover - source checkout without metadata
    __version__ = "0.0.0"

__all__ = [
    "Severity",
    "Issue",
    "DatasetIssues",
    "FileVerdict",
    "ValidationReport",
    "resolve",
    "available_versions",
    "schema_version",
    "bids_version",
    "evaluate_string",
    "validate",
    "validate_subject",
    "validate_file",
    "__version__",
]
