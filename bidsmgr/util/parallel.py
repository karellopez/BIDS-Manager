"""Backend selection for the joblib pools used by scan / probe / convert.

joblib's default ``loky`` backend runs each task in a worker *process*, which
is what we want for DICOM header reads and dcm2niix probes: they are CPU- and
IO-heavy and sidestep the GIL entirely.

On **Windows with Python 3.14** that backend is unusable. loky's Windows spawn
path (``loky.backend.popen_loky_win32``) starts workers that die during
bootstrap, and joblib surfaces the corpse as::

    joblib.externals.loky.process_executor.TerminatedWorkerError: A worker
    process managed by the executor was unexpectedly terminated. ...

so a scan that works everywhere else fails outright there. joblib 1.5.3 /
loky 3.5.6 are the newest releases at the time of writing and both still carry
the bug, so there is no version to upgrade to — we have to route around it.

Two layers of defence:

* :func:`preferred_backend` picks ``"threading"`` up front on the affected
  combination, so users never hit the crash. Threads are slower than processes
  for this work (the GIL is only released around IO and inside pydicom /
  subprocess calls) but they are correct and keep the UI responsive.
* :func:`run_parallel` retries with threads if a process pool dies anyway.
  That covers the same failure appearing on a platform we did not predict,
  turning a hard failure into a slow success.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Iterable, Optional

from joblib import Parallel

log = logging.getLogger(__name__)


def loky_is_broken() -> bool:
    """True where joblib's process backend cannot be trusted.

    Windows + Python 3.14: loky's spawn path kills its own workers during
    bootstrap. Kept as a narrow, explicitly-versioned check so the fast
    process backend is given up only where it is actually broken — revisit
    when a joblib/loky release fixes it.
    """
    return sys.platform == "win32" and sys.version_info >= (3, 14)


def preferred_backend() -> Optional[str]:
    """Backend for :class:`joblib.Parallel`, or ``None`` for joblib's default.

    ``None`` means loky (worker processes). ``"threading"`` is returned only
    where processes are known to be broken.
    """
    return "threading" if loky_is_broken() else None


def _is_dead_worker_error(exc: BaseException) -> bool:
    """Whether ``exc`` means the process pool died rather than a task failing.

    Matched by name so importing joblib's private executor module (and pulling
    in its vendored loky) is not required, and so sibling errors from other
    joblib versions are still caught.
    """
    names = {type(exc).__name__ for exc in _causes(exc)}
    return bool(names & {
        "TerminatedWorkerError",     # worker segfaulted / was killed
        "BrokenProcessPool",         # pool unusable
        "BrokenExecutor",
        "WorkerInterrupt",
        "ProcessTerminatedError",
    })


def _causes(exc: BaseException) -> Iterable[BaseException]:
    """``exc`` plus its ``__cause__`` / ``__context__`` chain (bounded)."""
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen and len(seen) < 10:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def run_parallel(
    tasks: Callable[[], Iterable],
    *,
    n_jobs: int,
    consume: Callable[[Any], Any],
    backend: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Run ``tasks`` through :class:`joblib.Parallel`, falling back to threads.

    ``tasks`` is a *factory* returning a fresh iterable of ``delayed(...)``
    calls: a generator is consumed by the first attempt, so the retry needs to
    build its own. ``consume`` receives whatever ``Parallel`` returns (a list,
    or a generator when ``return_as="generator"``) and drives the iteration,
    which is what lets callers poll for cancellation.

    If the process pool dies, the whole run is retried on the threading
    backend. The retry restarts from the beginning — the failure happens as
    workers start up, so little work is lost, and a partial result would be
    worse than a slow one.
    """
    chosen = backend if backend is not None else preferred_backend()
    try:
        return consume(Parallel(n_jobs=n_jobs, backend=chosen, **kwargs)(tasks()))
    except Exception as exc:
        if chosen == "threading" or not _is_dead_worker_error(exc):
            raise
        log.warning(
            "Parallel worker processes died (%s); retrying with threads. "
            "This is slower but avoids the failure.",
            type(exc).__name__,
        )
    return consume(
        Parallel(n_jobs=n_jobs, backend="threading", **kwargs)(tasks())
    )
