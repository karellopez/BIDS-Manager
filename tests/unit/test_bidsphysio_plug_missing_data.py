"""Tests for the vendored bidsphysio ``PhysioSignal.plug_missing_data`` fix.

The upstream implementation inserted one missing sample at a time, each
insertion doing a full ``np.concatenate`` plus a re-scan from index 0 --
O(n^2), which hangs for minutes on large CMRR logs (a real 22M-sample ECG
with many gaps). bids-manager replaced it with a vectorised single-pass
rewrite. These tests lock in:

* identical output to the original algorithm whenever the original works
  (i.e. the first interval is not itself a gap -- see note below),
* a regular time grid with the original samples preserved and gaps filled
  with NaN, for arbitrary inputs, and
* that the pathological many-gap case completes quickly (no O(n^2) hang).

Note on the upstream edge bug: the original used
``np.argmax(diffs > dt*1.001)`` whose result is ``0`` both for "no gap" and
"first gap at index 0", and its ``while i != 0`` loop then treated index-0
gaps as "done" -- so a gap in the very first interval was silently left
unfilled. The vectorised rewrite fills those too (correct behaviour); the
identical-output test therefore only compares on inputs whose first interval
is not a gap, which is the case for real recordings.
"""

from __future__ import annotations

import time

import numpy as np

from bidsmgr.vendor.bidsphysio.base.bidsphysio import PhysioSignal


def _naive_plug(times, signal, sps, missing=np.nan):
    """The original O(n^2) algorithm, verbatim, as a reference oracle."""
    dt = 1 / sps
    t = np.array(times, dtype=float)
    arr = np.array(signal, dtype=float)
    i = np.argmax(np.ediff1d(t) > dt * 1.001)
    while i != 0:
        t = np.concatenate((t[: i + 1], [t[i] + dt], t[i + 1:]))
        arr = np.concatenate((arr[: i + 1], [missing], arr[i + 1:]))
        i = np.argmax(np.ediff1d(t) > dt * 1.001)
    return t, arr


def _plug(times, signal, sps=1.0):
    ps = PhysioSignal(
        label="x", signal=np.array(signal, dtype=float),
        samples_per_second=sps, sampling_times=np.array(times, dtype=float),
    )
    ps.plug_missing_data()
    return ps


def test_identical_to_original_when_first_interval_is_not_a_gap() -> None:
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = rng.integers(4, 50)
        rest = np.sort(rng.choice(np.arange(2, n * 2), size=n - 2, replace=False)).astype(float)
        times = np.concatenate(([0.0, 1.0], rest))  # first interval = 1 (no leading gap)
        signal = rng.normal(size=len(times))
        ref_t, ref_s = _naive_plug(times, signal, 1.0)
        ps = _plug(times, signal)
        assert np.allclose(ps.sampling_times, ref_t)
        assert np.array_equal(np.isnan(ps.signal), np.isnan(ref_s))
        assert np.allclose(ps.signal[~np.isnan(ps.signal)], ref_s[~np.isnan(ref_s)])


def test_always_regular_grid_with_originals_preserved() -> None:
    rng = np.random.default_rng(2)
    for _ in range(300):
        n = rng.integers(3, 50)
        times = np.sort(rng.choice(np.arange(0, n * 2), size=n, replace=False)).astype(float)
        signal = rng.normal(size=n)
        ps = _plug(times, signal)
        assert np.allclose(np.diff(ps.sampling_times), 1.0)  # regular grid
        assert np.allclose(ps.signal[~np.isnan(ps.signal)], signal)  # originals kept


def test_no_op_when_already_regular() -> None:
    ps = _plug([0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0])
    assert np.array_equal(ps.sampling_times, [0, 1, 2, 3])
    assert np.array_equal(ps.signal, [10, 11, 12, 13])
    assert ps.samples_count == 4


def test_fills_single_and_multi_sample_gaps() -> None:
    # gap of 1 missing (1->3) and a wide gap of 3 missing (3->7)
    ps = _plug([0.0, 1.0, 3.0, 7.0], [0.0, 1.0, 3.0, 7.0])
    assert np.allclose(ps.sampling_times, [0, 1, 2, 3, 4, 5, 6, 7])
    nan_positions = np.isnan(ps.signal)
    assert list(np.nonzero(nan_positions)[0]) == [2, 4, 5, 6]


def test_large_many_gap_input_does_not_hang() -> None:
    """1M samples with ~1M gaps: the old O(n^2) loop would hang for minutes."""
    rng = np.random.default_rng(3)
    times = np.sort(rng.choice(np.arange(0, 2_000_000), size=1_000_000, replace=False)).astype(float)
    signal = rng.normal(size=1_000_000)
    t0 = time.monotonic()
    ps = _plug(times, signal)
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f"plug_missing_data too slow ({elapsed:.1f}s) - O(n^2) regression?"
    assert np.allclose(np.diff(ps.sampling_times), 1.0)
