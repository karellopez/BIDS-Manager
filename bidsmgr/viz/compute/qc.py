"""Quality maps of a time series: mean, standard deviation, temporal SNR.

The three images a functional run is judged by before anything else: where
the signal is (mean), where it moves (standard deviation: motion at the
edges, vessels, ghosts), and how much of it is signal (temporal SNR, mean
over SD, the number acquisition protocols are compared by).

Accumulated frame by frame in float64 sums, never as a float copy of the
series: the 1224-volume BOLD the viewer is measured on is 679 MB as stored
and would be 2.7 GB in float64. Runs on a worker.

Each map comes with what it takes to READ it (:data:`HELP`, :func:`summary`):
a colour without a scale and a sentence on what it means is a picture, not
a measurement.

Qt-free.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np

Map = Literal["mean", "sd", "tsnr"]

#: What each map is called on screen.
TITLES: dict[str, str] = {"mean": "Mean", "sd": "Standard deviation",
                          "tsnr": "Temporal SNR"}


class Cancelled(Exception):
    pass


def non_steady_state(src, *, cancel: Optional[Callable[[], bool]] = None) -> int:
    """How many volumes at the START are non-steady-state (dummy scans not
    discarded by the scanner): they are much brighter than the rest, and
    left in they dominate every statistic. Counted from the mean of a
    central block of each of the first frames against the median of the
    run: brighter by more than 5 robust deviations. 0 when none."""
    n = src.loaded_frames
    if n < 8:
        return 0
    probe = min(n, 64)
    sx, sy, sz = (max(1, d // 4) for d in src.spatial)
    centre = tuple(slice(d // 2 - k, d // 2 + k) for d, k in zip(src.spatial, (sx, sy, sz)))
    means = []
    for t in range(probe):
        if cancel is not None and cancel():
            raise Cancelled()
        means.append(float(np.mean(src.scale(np.asarray(src.raw_frame(t))[centre]))))
    means = np.asarray(means)
    rest = means[min(10, probe // 2):]
    med = float(np.median(rest))
    mad = float(np.median(np.abs(rest - med))) * 1.4826 or 1e-9
    k = 0
    while k < min(10, probe) and means[k] > med + 5.0 * mad:
        k += 1
    return k


def series_moments(src, *, skip: int = 0, detrend: int = 2,
                   cancel: Optional[Callable[[], bool]] = None,
                   progress: Optional[Callable[[int, int], None]] = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, sd)`` per voxel over the loaded frames after the first
    ``skip``, the SD taken around a polynomial trend of order ``detrend``
    (0, 1 or 2): slow scanner drift is not instability, and left in it
    lowers every temporal SNR (AFNI and MRIQC detrend too).

    One pass in float64 sums, never a float copy of the series. The trend
    uses discrete orthogonal polynomials (1, u, u^2 - mean(u^2)), so the
    residual sum of squares is the plain sum of squares less each
    projection: ``S2 - S0^2/m - A1^2/N1 - A2^2/N2``.
    """
    n = src.loaded_frames
    skip = max(0, min(int(skip), n - 3))
    m = n - skip
    if m < 2:
        raise ValueError("a quality map needs a series of at least two volumes")
    order = max(0, min(int(detrend), 2, m - 2))
    u = np.linspace(-1.0, 1.0, m) if m > 1 else np.zeros(1)
    q1 = u
    q2 = u * u - float(np.mean(u * u))
    n1 = float((q1 * q1).sum()) or 1.0
    n2 = float((q2 * q2).sum()) or 1.0
    s0 = s2 = a1 = a2 = None
    for k in range(m):
        if cancel is not None and cancel():
            raise Cancelled()
        frame = src.scale(src.raw_frame(skip + k)).astype(np.float64, copy=False)
        if s0 is None:
            s0 = np.zeros(frame.shape, dtype=np.float64)
            s2 = np.zeros(frame.shape, dtype=np.float64)
            a1 = np.zeros(frame.shape, dtype=np.float64) if order >= 1 else None
            a2 = np.zeros(frame.shape, dtype=np.float64) if order >= 2 else None
        s0 += frame
        s2 += frame * frame
        if a1 is not None:
            a1 += q1[k] * frame
        if a2 is not None:
            a2 += q2[k] * frame
        if progress is not None and (k % 64 == 0 or k == m - 1):
            progress(k + 1, m)
    mean = s0 / m
    rss = s2 - s0 * s0 / m
    if a1 is not None:
        rss -= a1 * a1 / n1
    if a2 is not None:
        rss -= a2 * a2 / n2
    dof = max(m - 1 - order, 1)
    return mean, np.sqrt(np.maximum(rss, 0.0) / dof)


def brain_mask(mean: np.ndarray) -> np.ndarray:
    """Voxels bright enough to be in the head: above a fifth of the robust
    top of the mean image. Outside it every map is noise over noise."""
    finite = mean[np.isfinite(mean)]
    if finite.size == 0:
        return np.zeros(mean.shape, dtype=bool)
    top = float(np.percentile(finite, 98))
    return mean > 0.2 * top


def quality_map(src, which: Map, *, skip: Optional[int] = None, detrend: int = 2,
                cancel: Optional[Callable[[], bool]] = None,
                progress: Optional[Callable[[int, int], None]] = None
                ) -> tuple[np.ndarray, dict]:
    """``(map, facts)``: one map, (X, Y, Z) float32, NaN outside the head
    (drawn as nothing, not as the lowest colour), and how it was made:
    ``volumes`` used, ``skipped`` at the start, ``detrend`` order."""
    if skip is None:
        skip = non_steady_state(src, cancel=cancel)
    mean, sd = series_moments(src, skip=skip, detrend=detrend, cancel=cancel,
                              progress=progress)
    mask = brain_mask(mean)
    if which == "mean":
        out = mean
    elif which == "sd":
        out = sd
    else:
        out = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)
    facts = {"volumes": src.loaded_frames - skip, "skipped": skip, "detrend": detrend,
             "map": which}
    return np.where(mask, out, np.nan).astype(np.float32), facts


#: Under this temporal SNR a voxel's signal is hard to tell from its noise
#: in a single run (a reading aid, not a pass mark: see HELP).
TSNR_LOW = 20.0


def summary(values: np.ndarray, which: Map = "tsnr") -> dict:
    """The numbers a map is read by, over the head: ``median``, ``q1``,
    ``q3``, ``p5``, ``p95``, ``voxels`` and, for temporal SNR, the share of
    the head ``below`` :data:`TSNR_LOW`."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"voxels": 0}
    p5, q1, med, q3, p95 = (float(x) for x in np.percentile(v, (5, 25, 50, 75, 95)))
    out = {"voxels": int(v.size), "p5": p5, "q1": q1, "median": med, "q3": q3, "p95": p95}
    if which == "tsnr":
        out["below"] = float(np.mean(v < TSNR_LOW))
    return out


#: How to read each map: shown beside it. Careful wording on purpose:
#: temporal SNR depends on field strength, voxel size, acceleration, coil and
#: TR, so the honest comparison is with runs of the same protocol.
HELP: dict[str, str] = {
    "tsnr": (
        "Temporal SNR: each voxel's mean over its standard deviation through "
        "the run, after slow drift is removed. Higher is steadier. Compare "
        "runs of the SAME protocol: it depends on field strength, voxel size, "
        "acceleration, the coil and the repetition time, so no single number "
        "is a pass mark (Krueger and Glover 2001; Triantafyllou 2005). "
        "Patterns to look for: a low ring at the edge of the brain (motion), "
        "low values in the ventricles and along large vessels (pulsation), "
        "low where the mean image is also dark, as in orbitofrontal cortex "
        "and the temporal poles (susceptibility dropout), and alternating "
        "slices (slice-wise motion or spikes)."),
    "sd": (
        "Standard deviation through the run, after slow drift is removed, in "
        "the scanner's units. Bright means the signal moves: expected in "
        "vessels and ventricles, a warning as a rim at the brain's edge "
        "(motion) or as copies of the head outside it along the phase-encode "
        "direction (ghosting)."),
    "mean": (
        "Mean through the run. Where it is dark inside the head, signal is "
        "lost (susceptibility dropout near the sinuses and ear canals); the "
        "temporal SNR there will be low for that reason, not because of "
        "noise."),
}


def describe_summary(stats: dict, facts: dict, which: Map) -> str:
    """One line for the status bar and the controls column."""
    if not stats.get("voxels"):
        return "no voxels in the head"
    text = (f"median {stats['median']:.3g} (middle half {stats['q1']:.3g} to "
            f"{stats['q3']:.3g}) over {stats['voxels']:,} voxels")
    if which == "tsnr" and "below" in stats:
        text += f"; {stats['below'] * 100:.0f} % below {TSNR_LOW:g}"
    if facts:
        text += f"; {facts.get('volumes', '?')} volumes"
        if facts.get("skipped"):
            text += f" ({facts['skipped']} non-steady-state skipped)"
        if facts.get("detrend"):
            text += f", order-{facts['detrend']} drift removed"
    return text


def per_volume(src, *, skip: int = 0, cancel: Optional[Callable[[], bool]] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """What each VOLUME looks like, over the head: ``global`` (the mean
    signal), ``dvars`` (the root mean square change from the previous
    volume, in percent of the head's mean: Power 2012), and ``flagged``,
    the volumes whose DVARS is above the upper box-plot fence (75th
    percentile plus 1.5 interquartile ranges, FSL's default). Cheap: two
    passes over frames already in memory. Volume 0 has no DVARS (NaN)."""
    n = src.loaded_frames
    if n < 3:
        raise ValueError("a series of at least three volumes is needed")
    ref = src.scale(np.asarray(src.raw_frame(min(skip + 1, n - 1))))
    mask = brain_mask(np.asarray(ref, dtype=np.float64))
    if not mask.any():
        raise ValueError("no voxel is bright enough to be in the head")
    gs = np.empty(n)
    dvars = np.full(n, np.nan)
    prev = None
    for t in range(n):
        if cancel is not None and cancel():
            raise Cancelled()
        frame = src.scale(np.asarray(src.raw_frame(t)))[mask].astype(np.float64)
        gs[t] = frame.mean()
        if prev is not None:
            dvars[t] = float(np.sqrt(np.mean((frame - prev) ** 2)))
        prev = frame
        if progress is not None and (t % 64 == 0 or t == n - 1):
            progress(t + 1, n)
    level = float(np.median(gs[skip:])) or 1.0
    dvars_pct = dvars / level * 100.0
    valid = dvars_pct[skip + 1:]
    valid = valid[np.isfinite(valid)]
    if valid.size:
        q1, q3 = np.percentile(valid, (25, 75))
        fence = float(q3 + 1.5 * (q3 - q1))
    else:
        fence = float("inf")
    flagged = np.flatnonzero(np.isfinite(dvars_pct) & (dvars_pct > fence)
                             & (np.arange(n) > skip))
    return {"global": gs, "dvars": dvars_pct, "fence": fence, "flagged": flagged,
            "skipped": skip}


def display_for(which: Map, values: np.ndarray):
    """The look a quality map opens with: its own colour map and a window
    from ZERO (the low end is what a quality map is for: ventricles,
    dropout, edge rings) to the head's 98th percentile; the mean map, a
    picture of the anatomy, from its 2nd percentile. Outside the head the
    map is NaN, which is drawn as nothing."""
    from ..scene import VolumeDisplay

    inside = values[np.isfinite(values)]
    if inside.size:
        p2, p98 = (float(v) for v in np.percentile(inside, (2, 98)))
    else:
        p2, p98 = 0.0, 1.0
    lo = p2 if which == "mean" else 0.0
    hi = p98 if p98 > lo else lo + 1.0
    colormap = {"mean": "gray", "sd": "hot", "tsnr": "viridis"}[which]
    return VolumeDisplay(colormap=colormap, window=(lo, hi), threshold_mode="range",
                         gamma=1.0, opacity=1.0)


#: What each map's values are, for its colour bar.
QUANTITY: dict[str, str] = {"mean": "signal (a.u.)", "sd": "SD (a.u.)",
                            "tsnr": "tSNR (mean / SD)"}


def quality_overlay(src, which: Map, *, cancel=None, progress=None):
    """The map as an :class:`~bidsmgr.viz.overlays.Overlay` over ``src``."""
    from ..data.volume import array_volume
    from ..overlays import Overlay

    values, facts = quality_map(src, which, cancel=cancel, progress=progress)
    vol = array_volume(values, src.affine, path=src.path, name=TITLES[which])
    stats = summary(values, which)
    display = display_for(which, values)
    vol.quantity = QUANTITY[which]
    vol.notes = {"qc": which, "stats": stats, "facts": facts, "window": display.window,
                 "summary": describe_summary(stats, facts, which), "help": HELP[which]}
    note = f"{TITLES[which].lower()}: {describe_summary(stats, facts, which)}"
    return Overlay(source=vol, display=display, kind="image",
                   note=note, name=f"{TITLES[which]} of {src.path.name}")


__all__ = ["Cancelled", "HELP", "Map", "QUANTITY", "TITLES", "TSNR_LOW", "brain_mask",
           "describe_summary", "display_for", "non_steady_state", "per_volume",
           "quality_map", "quality_overlay", "series_moments", "summary"]
