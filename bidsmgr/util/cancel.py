"""Cooperative cancellation for the long-running scan / convert verbs.

The GUI workers expose a Stop button that flips a ``threading.Event``; the
CLI verbs receive its ``is_set`` bound method as a ``cancel_check`` callable
and poll it at unit-of-work boundaries (each batch of DICOM-read results,
each convert subject / task). When it returns True they raise
:class:`OperationCancelled`, which the worker catches and reports as a clean
stop.

This is cooperative, not pre-emptive: a dcm2niix subprocess or a single
DICOM read already in flight runs to completion before the stop takes
effect -- nothing is force-killed, so the BIDS tree is never left
half-written. No partial output is committed: a cancelled scan writes no
TSV, and a cancelled convert subject's staging is wiped by the existing
per-subject ``finally`` (subjects committed before the stop remain, which
is the desired "stop here" behaviour).
"""

from __future__ import annotations

from typing import Callable, Optional

# A zero-arg predicate that returns True when a stop has been requested.
CancelCheck = Optional[Callable[[], bool]]


class OperationCancelled(Exception):
    """Raised inside ``run_scan`` / ``run_convert`` when a stop was requested."""


def is_cancelled(cancel_check: CancelCheck) -> bool:
    """True if ``cancel_check`` is provided and currently signals a stop."""
    return bool(cancel_check is not None and cancel_check())


def raise_if_cancelled(cancel_check: CancelCheck, message: str = "operation cancelled") -> None:
    """Raise :class:`OperationCancelled` if a stop has been requested."""
    if is_cancelled(cancel_check):
        raise OperationCancelled(message)


__all__ = ["CancelCheck", "OperationCancelled", "is_cancelled", "raise_if_cancelled"]
