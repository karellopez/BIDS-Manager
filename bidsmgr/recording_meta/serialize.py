"""Load and build :class:`RecordingMetaSpec` objects.

The spec is a plain JSON document (our own layout, not tied to any external
tool). The convert and metadata verbs read it via ``--recording-meta``; when no
spec is supplied they fall back to :func:`default_spec`, which preserves the
historical behaviour of writing ``PowerLineFrequency = 50`` by default while
keeping that value visible and overridable instead of buried in a CLI flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .models import AcquisitionSpec, RecordingMetaSpec

# Historical default written when nothing else specifies a power-line
# frequency, preserved so removing the old --line-freq flag does not silently
# change the default output.
DEFAULT_POWER_LINE_FREQ = 50.0

# Suffix for the scaffold the scan verb writes next to an inventory TSV
# (``<inventory>.tsv.recording_meta.json``). The convert/metadata verbs
# auto-discover it when ``--recording-meta`` is not given, mirroring the
# ``files_by_uid`` sidecar convention.
RECORDING_META_SIDECAR = ".recording_meta.json"


def scaffold_sidecar_path(tsv_path) -> Path:
    """Return the recording-metadata scaffold path beside an inventory TSV."""
    p = Path(tsv_path)
    return p.with_name(p.name + RECORDING_META_SIDECAR)


def default_spec(power_line_freq: Optional[float] = DEFAULT_POWER_LINE_FREQ) -> RecordingMetaSpec:
    """A spec with no enrichment beyond the default power-line frequency."""
    return RecordingMetaSpec(defaults=AcquisitionSpec(power_line_freq=power_line_freq))


def load_spec(path: Path) -> RecordingMetaSpec:
    """Read and validate a recording-metadata JSON document.

    Raises the underlying ``OSError`` / JSON / Pydantic error so the caller can
    report a precise message; callers that want a forgiving load should catch.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    return RecordingMetaSpec.model_validate(data)


def dump_spec(spec: RecordingMetaSpec) -> str:
    """Serialise a spec to pretty JSON (drops unset leaves for readability)."""
    return json.dumps(spec.model_dump(exclude_none=True), indent=2) + "\n"


__all__ = ["DEFAULT_POWER_LINE_FREQ", "default_spec", "load_spec", "dump_spec"]
