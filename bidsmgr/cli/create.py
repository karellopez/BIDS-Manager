"""``bidsmgr-create`` — create + scaffold a BIDS dataset workspace.

Creates the dataset folder (if needed), writes a minimal valid BIDS skeleton
(``dataset_description.json`` + ``README`` + ``.bidsignore``), and initializes
the event-sourced project bundle at ``<bids_root>/.bidsmgr/project`` so curation
can be resumed later. The folder name is the dataset slug you later pass to
``bidsmgr-scan --dataset`` / convert.

Safe to run on a directory that already holds a BIDS dataset: it ADOPTS it,
adding only the missing scaffold files + the project bundle, and never
overwriting an existing ``dataset_description.json`` / ``README``. Running it a
second time on a BM workspace is a no-op (the bundle already exists).

Architectural note (rule 3, no Pipeline orchestrator): orchestration is
straight-line code here; ``Project`` and the ``_scaffold`` helpers do the work.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..project import Project
from ._scaffold import (
    ensure_bidsignore,
    ensure_dataset_description,
    ensure_readme,
    project_bundle_dir,
)

log = logging.getLogger(__name__)


def open_or_create_workspace(
    bids_root: Path,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Project:
    """Create, adopt, or reopen the project workspace at ``bids_root``.

    Writes the minimal BIDS skeleton (never overwriting existing files) and
    returns the opened :class:`~bidsmgr.project.Project` bundle (created at
    ``<bids_root>/.bidsmgr/project`` if absent). Safe on a fresh folder, an
    external/non-BM BIDS dataset (adopted read-only), or an existing BM project
    (reopened). This is the single entry point the GUI Welcome tab uses.
    """
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)
    ds_name = name or bids_root.name

    # Minimal valid BIDS skeleton (each helper is never-overwrite).
    ensure_dataset_description(bids_root, name=ds_name)
    ensure_readme(bids_root, ds_name)
    ensure_bidsignore(bids_root)

    bundle = project_bundle_dir(bids_root)
    if bundle.exists():
        return Project.open(bundle)
    return Project.create(bundle, name=ds_name, description=description)


def run_create(
    output_dir: Path,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> int:
    """Create or adopt a BIDS dataset workspace at ``output_dir``.

    Returns 0 on success. ``name`` seeds the dataset_description ``Name``
    (defaults to the folder name); ``description`` is recorded in the project
    bundle's first event.
    """
    bids_root = Path(output_dir)
    existed = bids_root.exists()
    already = project_bundle_dir(bids_root).exists()

    open_or_create_workspace(bids_root, name=name, description=description)

    if already:
        log.info("%s is already a BIDS-Manager project", bids_root)
    else:
        verb = "Adopted existing dataset" if existed else "Created dataset"
        log.info("%s at %s", verb, bids_root)

    print(f"BIDS dataset ready at: {bids_root}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-create",
        description=(
            "Create + scaffold a BIDS dataset workspace (or adopt an existing "
            "BIDS folder). The folder name is the dataset slug used by "
            "bidsmgr-scan / bidsmgr-convert."
        ),
    )
    parser.add_argument(
        "output_dir", type=Path,
        help="Dataset folder to create or adopt.",
    )
    parser.add_argument(
        "--name", default=None,
        help="Human-readable dataset Name for dataset_description.json "
             "(defaults to the folder name).",
    )
    parser.add_argument(
        "--description", default=None,
        help="Optional project description recorded in the project bundle.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    return run_create(
        args.output_dir, name=args.name, description=args.description,
    )


if __name__ == "__main__":
    sys.exit(main())
