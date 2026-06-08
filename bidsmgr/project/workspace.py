"""Per-scan version management inside a dataset's project bundle.

Each scan of a source folder is stored as its own version directory under
``<bids_root>/.bidsmgr/project/scans/<NNNN>__<slug>/``. A version directory is
itself a :class:`~bidsmgr.project.Project` bundle (``events.jsonl`` +
``meta.json``), so each scan's curation history is fully isolated, plus the
inventory snapshot (+ recording-metadata scaffold + files_by_uid sidecar) and a
``version.json`` descriptor used by the GUI version picker. A second scan of a
different source therefore never overwrites the first table.

Qt-free (architectural rule 2). Imports only ``util`` (a leaf utility); nothing
in this subpackage imports the GUI.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..util.paths import safe_path_component

VERSION_META = "version.json"
INVENTORY_NAME = "inventory.tsv"

_INDEX_RE = re.compile(r"^(\d{4})__")


def scans_dir(bids_root: Path) -> Path:
    """Return ``<bids_root>/.bidsmgr/project/scans``."""
    return Path(bids_root) / ".bidsmgr" / "project" / "scans"


def version_inventory(version_dir: Path) -> Path:
    """Return the inventory snapshot path inside a version directory."""
    return Path(version_dir) / INVENTORY_NAME


@dataclass(frozen=True)
class ScanVersion:
    """A single scan version (one source folder scanned into the dataset)."""

    dir: Path
    version_id: str          # e.g. "0001__neuro2"
    index: int
    source_label: str
    raw_root: Optional[str]
    status: str              # "curating" | "converted"
    inventory: Path


def _next_index(sd: Path) -> int:
    mx = 0
    if sd.is_dir():
        for child in sd.iterdir():
            m = _INDEX_RE.match(child.name)
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1


def allocate_version_dir(bids_root: Path, source_label: str) -> Path:
    """Return the next free version directory path (not created).

    ``<scans>/<NNNN>__<slug>`` where ``NNNN`` is one past the highest existing
    index and ``slug`` is a filesystem-safe form of the source folder name.
    """
    sd = scans_dir(bids_root)
    idx = _next_index(sd)
    slug = safe_path_component(source_label) or "scan"
    return sd / f"{idx:04d}__{slug}"


def write_version_meta(
    version_dir: Path,
    *,
    source_label: str,
    raw_root: Optional[str],
    status: str,
) -> None:
    """Write/merge the version.json descriptor (preserves unset fields)."""
    p = Path(version_dir) / VERSION_META
    meta: dict = {}
    if p.exists():
        try:
            old = json.loads(p.read_text())
            if isinstance(old, dict):
                meta = old
        except (OSError, json.JSONDecodeError):
            meta = {}
    meta["source_label"] = source_label
    meta["raw_root"] = raw_root
    meta["status"] = status
    p.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def read_version_meta(version_dir: Path) -> dict:
    p = Path(version_dir) / VERSION_META
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return d if isinstance(d, dict) else {}


def list_versions(bids_root: Path) -> list[ScanVersion]:
    """List complete scan versions (those with an inventory), oldest first."""
    sd = scans_dir(bids_root)
    out: list[ScanVersion] = []
    if not sd.is_dir():
        return out
    for child in sorted(sd.iterdir()):
        m = _INDEX_RE.match(child.name)
        if not (child.is_dir() and m):
            continue
        inv = version_inventory(child)
        if not inv.exists():
            continue  # incomplete / aborted scan
        meta = read_version_meta(child)
        out.append(ScanVersion(
            dir=child,
            version_id=child.name,
            index=int(m.group(1)),
            source_label=str(meta.get("source_label") or child.name),
            raw_root=meta.get("raw_root"),
            status=str(meta.get("status") or "curating"),
            inventory=inv,
        ))
    out.sort(key=lambda v: v.index)
    return out


__all__ = [
    "ScanVersion",
    "scans_dir",
    "version_inventory",
    "allocate_version_dir",
    "write_version_meta",
    "read_version_meta",
    "list_versions",
]
