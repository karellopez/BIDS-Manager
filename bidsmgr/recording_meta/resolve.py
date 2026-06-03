"""Resolve the effective enrichment for one recording.

Merges the dataset-wide ``defaults`` with the per-recording ``overrides`` and
picks the task-scoped event map and protocol. The result is a flat object the
enrichment step consumes. Per-row inventory cells (montage, line_freq, reference,
ground) take precedence over the resolved spec and are layered on by the caller
(the convert verb), so this module stays spec-only and pure.

Precedence (low to high): ``defaults`` < ``overrides[row_id]`` < inventory cells.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from .models import AcquisitionSpec, EventMap, RecordingMetaSpec, TaskProtocol

_GLOBAL_EVENT_KEY = "*"


class EffectiveSpec(BaseModel):
    """The resolved enrichment for a single recording."""

    model_config = ConfigDict(extra="forbid")

    acquisition: AcquisitionSpec
    task_protocol: Optional[TaskProtocol] = None
    event_map: EventMap = {}


def merge_acquisition(
    base: AcquisitionSpec, over: Optional[AcquisitionSpec]
) -> AcquisitionSpec:
    """Return ``base`` with every set field of ``over`` layered on top.

    Scalars: a non-None override value wins. ``aux_channels``: the two maps are
    merged key-by-key (override entries replace same-named base entries).
    ``filters``: a non-empty override list replaces the base list outright (the
    set of filters is treated as a unit, not merged element-wise).
    """
    if over is None:
        return base.model_copy(deep=True)

    merged = base.model_dump()
    over_dump = over.model_dump(exclude_none=True)

    for key, value in over_dump.items():
        if key == "aux_channels":
            combined = dict(merged.get("aux_channels") or {})
            combined.update(value or {})
            merged["aux_channels"] = combined
        elif key == "filters":
            # exclude_none drops a None list; an explicit empty list would not
            # reach here, so a present value means "replace".
            if value:
                merged["filters"] = value
        else:
            merged[key] = value

    return AcquisitionSpec.model_validate(merged)


def resolve_effective(
    spec: RecordingMetaSpec,
    row_id: str,
    task_label: Optional[str] = None,
) -> EffectiveSpec:
    """Resolve the effective enrichment for the recording identified by ``row_id``.

    ``task_label`` selects the task protocol and event map; the event map falls
    back to the ``"*"`` global map when no task-specific map exists.
    """
    acquisition = merge_acquisition(spec.defaults, spec.overrides.get(row_id))

    task_protocol = spec.task_protocols.get(task_label) if task_label else None

    event_map: EventMap = {}
    if task_label and task_label in spec.event_maps:
        event_map = dict(spec.event_maps[task_label])
    elif _GLOBAL_EVENT_KEY in spec.event_maps:
        event_map = dict(spec.event_maps[_GLOBAL_EVENT_KEY])

    return EffectiveSpec(
        acquisition=acquisition,
        task_protocol=task_protocol,
        event_map=event_map,
    )


__all__ = ["EffectiveSpec", "merge_acquisition", "resolve_effective"]
