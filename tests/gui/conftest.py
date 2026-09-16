"""Shared fixtures for the GUI test suite.

Defines :func:`isolated_settings` — sandbox ``QSettings`` per-test so
the GUI's persistence layer doesn't leak the real user's
preferences into tests (or vice versa) — and :func:`collect_garbage`,
which keeps Python's cyclic collector out of Qt's paint loop.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterator

import pytest
from PyQt6.QtCore import QCoreApplication, QSettings


@pytest.fixture(autouse=True)
def collect_garbage() -> Iterator[None]:
    """Reap each test's widget tree before the next test paints.

    A panel built in a test is a top-level widget owned by Python, and
    its tree is full of reference cycles (parent <-> child, signal
    closures, per-widget dicts), so dropping the last name for it does
    NOT free it. It becomes cyclic garbage, and the cyclic collector
    then runs at whatever unrelated allocation happens to cross the
    generation threshold.

    When that allocation falls inside ``qtbot.waitExposed`` — i.e. while
    Qt is halfway through a paint — the collector deletes the PREVIOUS
    test's C++ widgets mid-paint, which Qt does not allow, and the
    process dies with SIGSEGV inside ``QPainter::drawPixmap``. That is
    what made ``test_converter_panel.py`` segfault only as part of the
    full run: alone it never allocates enough to trigger a collection at
    the wrong moment, and the crash lands on whichever test is painting
    when the threshold happens to be crossed.

    Collecting here forces the reaping to happen at teardown, between
    tests, where no paint is in flight, which removes the ordering
    dependence entirely. The added cost did not stand out against the
    run-to-run variance of the tier.

    Not reproducible on every interpreter: 3.11 and 3.12 run the tier
    clean without this fixture. That is luck, not safety — they simply
    do not happen to cross a GC threshold mid-paint. The hazard is the
    same on all of them, and on every platform, because it is CPython's
    collector meeting Qt's paint loop and neither is OS-specific.
    """
    yield
    gc.collect()


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path) -> Iterator[None]:
    """Redirect ``QSettings`` into a per-test INI file.

    Forces the IniFormat default (macOS otherwise uses native plist
    and ignores ``setPath``) and points it at ``tmp_path``. The
    org/app names also get swapped so a leaked value from outside
    the sandbox cannot poison the test.

    ``autouse`` because opting in is a decision every test author has to
    remember and 17 of 37 files did not, so those wrote to the developer's
    REAL settings store. Two consequences, both seen: a test could pass or fail
    depending on what an earlier run had left behind, which made a genuine
    regression indistinguishable from noise, and running the suite quietly
    rewrote the preferences of whoever ran it.
    """
    orig_org = QCoreApplication.organizationName()
    orig_app = QCoreApplication.applicationName()
    orig_default = QSettings.defaultFormat()

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )
    QCoreApplication.setOrganizationName("bidsmgr-tests")
    QCoreApplication.setApplicationName("bidsmgr-tests")
    # Ensure an empty starting state.
    QSettings().clear()
    QSettings().sync()
    try:
        yield
    finally:
        QSettings().clear()
        QSettings().sync()
        QCoreApplication.setOrganizationName(orig_org)
        QCoreApplication.setApplicationName(orig_app)
        QSettings.setDefaultFormat(orig_default)
