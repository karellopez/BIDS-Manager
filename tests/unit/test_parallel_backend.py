"""Backend selection / fallback for the joblib pools (bidsmgr.util.parallel).

Regression cover for the Windows + Python 3.14 failure, where loky's worker
processes die during bootstrap and joblib raises ``TerminatedWorkerError``,
taking a whole DICOM scan down with them.
"""

from __future__ import annotations

import sys

import pytest
from joblib import delayed

from bidsmgr.util import parallel as P


def _square(i: int) -> int:
    return i * i


class TestBackendSelection:
    def test_loky_is_used_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        assert P.loky_is_broken() is False
        assert P.preferred_backend() is None       # None == joblib's loky

    def test_windows_py314_avoids_loky(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sys, "version_info", (3, 14, 0))
        assert P.loky_is_broken() is True
        assert P.preferred_backend() == "threading"

    @pytest.mark.parametrize(
        ("platform", "version"),
        [("win32", (3, 13, 0)), ("linux", (3, 14, 0)), ("darwin", (3, 14, 0))],
    )
    def test_only_the_broken_combination_is_downgraded(
        self, monkeypatch: pytest.MonkeyPatch, platform: str, version: tuple,
    ) -> None:
        """Windows alone or 3.14 alone must keep the fast process backend."""
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(sys, "version_info", version)
        assert P.loky_is_broken() is False
        assert P.preferred_backend() is None


class TestRunParallel:
    def test_runs_and_collects(self) -> None:
        out = P.run_parallel(
            lambda: (delayed(_square)(i) for i in range(6)),
            n_jobs=2, consume=list,
        )
        assert out == [0, 1, 4, 9, 16, 25]

    def test_generator_mode(self) -> None:
        """return_as="generator" is how the scan polls for cancellation."""
        out = P.run_parallel(
            lambda: (delayed(_square)(i) for i in range(4)),
            n_jobs=2, consume=lambda gen: [x for x in gen],
            return_as="generator",
        )
        assert out == [0, 1, 4, 9]

    def test_dead_pool_retries_on_threads(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from joblib.externals.loky.process_executor import TerminatedWorkerError

        real = P.Parallel
        seen: list = []

        class FlakyParallel:
            def __init__(self, n_jobs=1, backend=None, **kw):
                self.backend, self.kw = backend, kw
                seen.append(backend)

            def __call__(self, tasks):
                if self.backend != "threading":
                    raise TerminatedWorkerError("worker died")
                return real(n_jobs=1, backend="threading", **self.kw)(tasks)

        monkeypatch.setattr(P, "Parallel", FlakyParallel)
        out = P.run_parallel(
            lambda: (delayed(_square)(i) for i in range(4)),
            n_jobs=2, consume=list,
        )
        assert out == [0, 1, 4, 9]
        assert seen[-1] == "threading"      # fell back
        assert len(seen) == 2               # tried once, then retried

    def test_task_errors_are_not_swallowed(self) -> None:
        """Only a dead pool triggers the retry — real failures propagate."""
        def boom(_i: int) -> None:
            raise RuntimeError("task failed")

        with pytest.raises(RuntimeError, match="task failed"):
            P.run_parallel(
                lambda: (delayed(boom)(i) for i in range(2)),
                n_jobs=1, consume=list,
            )

    def test_threading_backend_does_not_retry(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Already on threads, a failure must surface rather than loop."""
        from joblib.externals.loky.process_executor import TerminatedWorkerError

        calls: list = []

        class AlwaysDead:
            def __init__(self, n_jobs=1, backend=None, **kw):
                calls.append(backend)

            def __call__(self, tasks):
                raise TerminatedWorkerError("worker died")

        monkeypatch.setattr(P, "Parallel", AlwaysDead)
        with pytest.raises(TerminatedWorkerError):
            P.run_parallel(
                lambda: (delayed(_square)(i) for i in range(2)),
                n_jobs=2, consume=list, backend="threading",
            )
        assert calls == ["threading"]
