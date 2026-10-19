"""Anonymise a BIDS dataset's metadata (Qt-free).

``plan_anonymise`` lists every change a policy makes, ``apply_anonymise``
makes them (through an editor operation: one undoable step). Presets in
``policy``; what each kind of file carries in ``carriers``; the per-dataset
memory (date offsets, the pseudonym secret) in ``state``.
"""

from .plan import AnonymisePlan, FilePlan, anonymise, apply_anonymise, plan_anonymise
from .policy import (
    OFF,
    PRESET_LABELS,
    PRESETS,
    RECOMMENDED,
    STRICT,
    AnonymisePolicy,
    policy_for,
)

__all__ = [
    "AnonymisePlan", "AnonymisePolicy", "FilePlan", "OFF", "PRESETS", "PRESET_LABELS",
    "RECOMMENDED", "STRICT", "anonymise", "apply_anonymise", "plan_anonymise", "policy_for",
]
