"""Tests for cooperative scan / convert cancellation.

The GUI Stop button flips a ``threading.Event`` whose ``is_set`` is passed
to ``run_scan`` / ``run_convert`` as ``cancel_check``. The verbs poll it at
unit-of-work boundaries and raise :class:`OperationCancelled`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bidsmgr.util.cancel import (
    OperationCancelled,
    is_cancelled,
    raise_if_cancelled,
)


def test_is_cancelled_none_and_predicate() -> None:
    assert is_cancelled(None) is False
    assert is_cancelled(lambda: False) is False
    assert is_cancelled(lambda: True) is True


def test_raise_if_cancelled() -> None:
    raise_if_cancelled(None)            # no-op
    raise_if_cancelled(lambda: False)   # no-op
    with pytest.raises(OperationCancelled):
        raise_if_cancelled(lambda: True)


def test_phase1_raises_when_cancelled_before_dispatch(tmp_path: Path) -> None:
    """``_phase1_parallel_dcm2niix`` bails out before touching any task when
    a stop is already pending — so a Stop click never launches new dcm2niix."""
    from bidsmgr.cli.convert import _phase1_parallel_dcm2niix

    with pytest.raises(OperationCancelled):
        _phase1_parallel_dcm2niix(
            [], tmp_path, [], n_jobs=1, cancel_check=lambda: True,
        )


def test_phase1_runs_normally_without_cancel(tmp_path: Path) -> None:
    """With no cancel pending and no tasks, it returns an empty result list."""
    from bidsmgr.cli.convert import _phase1_parallel_dcm2niix

    assert _phase1_parallel_dcm2niix([], tmp_path, [], n_jobs=1) == []
    assert _phase1_parallel_dcm2niix(
        [], tmp_path, [], n_jobs=1, cancel_check=lambda: False,
    ) == []
