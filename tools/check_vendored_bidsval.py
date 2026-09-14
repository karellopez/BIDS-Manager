#!/usr/bin/env python3
"""Compare the vendored bidsval against a source checkout, and report drift.

Vendoring is copy-and-forget right up until it is not. Two things go wrong,
and both have happened here:

* a file is edited in the copy and the change is never written down, so the
  next wholesale refresh silently deletes it (this is how the ``anyOf`` field
  typing nearly went missing);
* a file is NOT copied at all, so the copy looks complete and is not (this is
  how ``schema/bundled/*.json``, 2.9 MB of schema data, was left behind and
  the vendored validator quietly read the installed package's copy instead).

Neither is visible in a diff of BIDS Manager's own history, because the
vendored tree is checked in as a unit. This compares it against the real
thing.

Usage::

    python tools/check_vendored_bidsval.py [path-to-bidsval-checkout]

Defaults to the sibling ``../bidsval`` this workspace carries. Exits non-zero
on any undocumented divergence, so it can gate a refresh.
"""

from __future__ import annotations

import argparse
import filecmp
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDORED = REPO / "bidsmgr" / "vendor" / "bidsval"

# Files that are DELIBERATELY different, and why. Keep in step with the table
# in bidsmgr/vendor/README.md; the whole point is that an entry here is a
# decision somebody wrote down.
EXPECTED_DIVERGENCE: dict[str, str] = {
    "__init__.py":
        "__version__ stated literally, not read from installed metadata",
    "schema/resolve.py":
        "_bundled_dir() uses resources.files(__package__), not a string name",
    "schema/fields.py":
        "local addition: FieldSpec.accepts/accepts_free_text/accepts_na, "
        "the _variants helpers, and an anyOf lookup in _item_type",
}

# Present in the copy and not upstream, on purpose.
EXPECTED_EXTRA = {"LICENSE"}

# Never copied: editor and OS droppings.
IGNORE_NAMES = {".DS_Store", "Thumbs.db"}
IGNORE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache"}


def _files(root: Path) -> set[str]:
    out: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if set(rel.parts) & IGNORE_DIRS or rel.name in IGNORE_NAMES:
            continue
        out.add(rel.as_posix())
    return out


def _source_dir(checkout: Path) -> Path:
    """The package inside a bidsval checkout, src-layout or flat."""
    for candidate in (checkout / "src" / "bidsval", checkout / "bidsval"):
        if (candidate / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"no bidsval package found under {checkout}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkout", nargs="?", default=str(REPO.parent / "bidsval"),
        help="path to a bidsval source checkout (default: ../bidsval)",
    )
    args = parser.parse_args(argv)

    checkout = Path(args.checkout).expanduser().resolve()
    if not checkout.is_dir():
        print(f"no bidsval checkout at {checkout}; nothing to compare against")
        return 0
    source = _source_dir(checkout)

    upstream, vendored = _files(source), _files(VENDORED)
    problems: list[str] = []

    missing = sorted(upstream - vendored)
    if missing:
        problems.append(
            "NOT COPIED (upstream has them, the vendored tree does not):\n"
            + "\n".join(f"    {name}" for name in missing)
        )

    extra = sorted(vendored - upstream - EXPECTED_EXTRA)
    if extra:
        problems.append(
            "ONLY IN THE COPY, and not in EXPECTED_EXTRA:\n"
            + "\n".join(f"    {name}" for name in extra)
        )

    changed = sorted(
        name for name in upstream & vendored
        if not filecmp.cmp(source / name, VENDORED / name, shallow=False)
    )
    undocumented = [n for n in changed if n not in EXPECTED_DIVERGENCE]
    if undocumented:
        problems.append(
            "EDITED WITHOUT BEING WRITTEN DOWN (a refresh would delete these):\n"
            + "\n".join(f"    {name}" for name in undocumented)
        )

    stale = [n for n in EXPECTED_DIVERGENCE if n in upstream and n not in changed]
    if stale:
        problems.append(
            "DOCUMENTED AS CHANGED BUT IDENTICAL (upstream took the change? "
            "drop the entry):\n"
            + "\n".join(f"    {name}" for name in stale)
        )

    print(f"upstream: {source}")
    print(f"vendored: {VENDORED}")
    print(f"{len(upstream)} upstream file(s), {len(vendored)} vendored")
    for name in changed:
        why = EXPECTED_DIVERGENCE.get(name, "UNDOCUMENTED")
        print(f"  differs: {name}\n           {why}")

    if problems:
        print("\n" + "\n\n".join(problems))
        print("\nFAIL")
        return 1
    print("\nOK: the copy is complete and every difference is documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
