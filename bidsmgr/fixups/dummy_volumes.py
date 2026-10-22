"""Remove the first volumes of a functional run, before it reaches the dataset.

The first volumes of a BOLD run are often acquired before the signal reaches
its steady state (dummy scans the scanner did not discard). A study that drops
them would otherwise do it in every analysis; here it is done once, at
conversion, and recorded where BIDS asks:

- the image (every echo) loses its first ``n`` volumes, header and stored
  values otherwise exactly as dcm2niix wrote them;
- the sidecar gains ``NumberOfVolumesDiscardedByUser`` (added to any value
  already there), and a ``VolumeTiming`` loses its first ``n`` entries and is
  measured from the new first volume;
- the run's events (``*_events.tsv`` beside it) move with the new first
  volume: BIDS measures an onset from the first volume of the image, so every
  onset becomes ``onset - n * RepetitionTime``. An event that now starts
  before the image is kept, with its negative onset, which the standard
  allows.

The run's physiology starts at the first volume kept too; that is done when
its tables are written (:mod:`bidsmgr.converter.physio_cmrr`), from the same
number, which the conversion hands the physio row of the run.

Only functional runs: removing volumes from diffusion or PET data would need
the gradient table or the frame timing rewritten with them.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

#: The suffixes volumes can be removed from: a functional time series.
DROPPABLE_SUFFIXES = frozenset({"bold", "cbv", "phase"})

FIELD = "NumberOfVolumesDiscardedByUser"


def applies(datatype: str, suffix: str) -> bool:
    """Whether volumes can be removed from a recording of this kind."""
    return datatype == "func" and suffix in DROPPABLE_SUFFIXES


def drop_leading_volumes(results: Iterable) -> int:
    """Remove the first ``task.drop_volumes`` volumes of every functional run
    converted in ``results``. Returns how many runs changed. A run too short
    for it is left whole, with a warning."""
    changed = 0
    for result in results:
        task = getattr(result, "task", None)
        n = int(getattr(task, "drop_volumes", 0) or 0)
        if n <= 0 or not getattr(result, "success", False) \
                or not applies(task.datatype, task.suffix):
            continue
        images = [Path(p) for p in result.staged_files
                  if str(p).endswith((".nii.gz", ".nii"))]
        if not images:
            continue
        try:
            done = [_drop_image(p, n) for p in images]
        except ValueError as exc:
            log.warning("%s: %s; the run is kept whole", task.basename, exc)
            continue
        tr = None
        for image in images:
            sidecar = _sidecar(image)
            if sidecar.exists():
                tr = _record(sidecar, n) or tr
        if tr is None and done:
            tr = done[0]
        if tr:
            for image in images:
                _shift_events(image, n * tr)
        log.info("%s: the first %d volume(s) removed", task.basename, n)
        changed += 1
    return changed


def _sidecar(image: Path) -> Path:
    name = image.name
    stem = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else name[: -len(".nii")]
    return image.with_name(stem + ".json")


def _drop_image(path: Path, n: int) -> float:
    """Cut the first ``n`` volumes; returns the repetition time in the
    header (seconds, 0 when unknown)."""
    import nibabel as nib
    import numpy as np

    img = nib.load(str(path))
    shape = img.shape
    if len(shape) < 4 or shape[3] <= n:
        volumes = shape[3] if len(shape) > 3 else 1
        raise ValueError(f"{n} volume(s) to remove, and {path.name} has {volumes}")
    raw = np.asanyarray(img.dataobj.get_unscaled()) if hasattr(img.dataobj, "get_unscaled") \
        else np.asanyarray(img.dataobj)
    out = img.__class__(raw[..., n:], img.affine, header=img.header)
    tmp = path.with_name(path.name + ".part" + (".nii.gz" if path.name.endswith(".gz") else ".nii"))
    nib.save(out, str(tmp))
    tmp.replace(path)
    units = img.header.get_xyzt_units()[1]
    scale = {"msec": 1e-3, "usec": 1e-6}.get(units, 1.0)
    return float(img.header["pixdim"][4]) * scale


def _record(sidecar: Path, n: int):
    """Write the record into the sidecar; returns its repetition time."""
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    before = data.get(FIELD, 0)
    data[FIELD] = (before if isinstance(before, int) and not isinstance(before, bool) else 0) + n
    timing = data.get("VolumeTiming")
    if isinstance(timing, list) and len(timing) > n:
        first = timing[n]
        data["VolumeTiming"] = [round(t - first, 6) for t in timing[n:]]
    sidecar.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tr = data.get("RepetitionTime")
    return float(tr) if isinstance(tr, (int, float)) and not isinstance(tr, bool) else None


def _shift_events(image: Path, seconds: float) -> None:
    """Move the onsets of the run's events by ``-seconds``."""
    name = image.name
    stem = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else name[: -len(".nii")]
    base = stem.rsplit("_", 1)[0]
    events = image.with_name(base + "_events.tsv")
    if not events.exists():
        return
    text = events.read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    if not rows or "onset" not in rows[0]:
        return
    col = rows[0].index("onset")
    for row in rows[1:]:
        if col < len(row):
            try:
                value = float(row[col]) - seconds
            except ValueError:
                continue
            row[col] = _number(value)
    out = io.StringIO()
    csv.writer(out, delimiter="\t", lineterminator="\n").writerows(rows)
    events.write_text(out.getvalue(), encoding="utf-8")
    log.info("%s: onsets moved by -%g s with the removed volumes", events.name, seconds)


def _number(value: float) -> str:
    text = f"{round(value, 6):.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


__all__ = ["DROPPABLE_SUFFIXES", "FIELD", "applies", "drop_leading_volumes"]
