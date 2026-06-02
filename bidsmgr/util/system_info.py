"""Cross-platform machine introspection: CPU thread/core count + RAM.

Used by the Settings dialog to (a) show a read-only "System info" panel and
(b) cap the parallel-workers spinboxes at the number of logical CPUs so a
user cannot ask for more workers than the machine has threads.

``psutil`` is the primary source (declared in ``pyproject.toml``); every
field degrades gracefully to a stdlib fallback so a missing or broken
``psutil`` never crashes the GUI -- the worst case is ``physical_cores`` /
``total_ram_bytes`` coming back ``None`` and the panel showing "unknown".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SystemInfo:
    """A snapshot of the host's CPU + memory capacity."""

    logical_threads: int                 # always >= 1 (the spinbox cap)
    physical_cores: Optional[int] = None  # None when undetectable
    total_ram_bytes: Optional[int] = None

    @property
    def total_ram_gib(self) -> Optional[float]:
        if self.total_ram_bytes is None:
            return None
        return self.total_ram_bytes / (1024 ** 3)

    def human(self) -> str:
        """One-line human summary, e.g. ``10 threads (10 cores) - 32.0 GiB RAM``."""
        if self.physical_cores and self.physical_cores != self.logical_threads:
            cpu = f"{self.logical_threads} threads ({self.physical_cores} cores)"
        else:
            cpu = f"{self.logical_threads} threads"
        if self.total_ram_gib is not None:
            return f"{cpu} - {self.total_ram_gib:.1f} GiB RAM"
        return f"{cpu} - RAM unknown"


def _logical_threads() -> int:
    # ``os.cpu_count()`` can return None on exotic platforms; ``sched_getaffinity``
    # (Linux) reflects the threads this process may actually use under cgroups.
    n: Optional[int] = None
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            n = len(getaffinity(0))
        except OSError:
            n = None
    if not n:
        n = os.cpu_count()
    return max(1, int(n or 1))


def _physical_cores() -> Optional[int]:
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        return int(cores) if cores else None
    except Exception:
        return None


def _total_ram_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    # POSIX fallback (Linux + macOS); Windows has no SC_PHYS_PAGES.
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def get_system_info() -> SystemInfo:
    """Return a :class:`SystemInfo` snapshot of the host machine."""
    return SystemInfo(
        logical_threads=_logical_threads(),
        physical_cores=_physical_cores(),
        total_ram_bytes=_total_ram_bytes(),
    )


__all__ = ["SystemInfo", "get_system_info"]
