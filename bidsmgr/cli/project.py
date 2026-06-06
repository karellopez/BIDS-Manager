"""``bidsmgr-project`` — inspect a project dataset's scan versions.

A companion to ``bidsmgr-create`` / ``bidsmgr-scan --project`` /
``bidsmgr-convert --project``: lists the versioned scans recorded under
``<dataset>/.bidsmgr/project/scans/`` (the same versions the GUI's scan picker
shows), so a CLI user can see what is in a project and which scan is active
(the latest). Read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..cli._scaffold import project_bundle_dir
from ..project import workspace


def run_project_list(bids_root: Path) -> int:
    bids_root = Path(bids_root)
    if not project_bundle_dir(bids_root).exists():
        print(
            f"{bids_root} is not a BIDS-Manager project "
            f"(no .bidsmgr/project bundle). Create one with `bidsmgr-create`."
        )
        return 1

    versions = workspace.list_versions(bids_root)
    print(f"Project: {bids_root}")
    if not versions:
        print("  no scans yet (run `bidsmgr-scan <raw> --project " f"{bids_root}`)")
        return 0
    print(f"  {len(versions)} scan version(s); active = latest:")
    for v in versions:
        active = "  *" if v is versions[-1] else "   "
        raw = v.raw_root or "(raw moved/unknown)"
        try:
            import pandas as pd
            n_rows = len(pd.read_csv(v.inventory, sep="\t", dtype=str))
        except Exception:
            n_rows = "?"
        print(
            f"{active} {v.version_id}  [{v.status}]  "
            f"{n_rows} rows  source={v.source_label}  raw={raw}"
        )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-project",
        description=(
            "Inspect a BIDS-Manager project dataset: list its versioned scans "
            "(the active one is the latest, marked *)."
        ),
    )
    parser.add_argument("dataset", type=Path, help="Project dataset folder.")
    parser.add_argument(
        "--list", action="store_true",
        help="List scan versions (the default action).",
    )
    args = parser.parse_args(argv)
    # ``--list`` is the only (and default) action today.
    return run_project_list(Path(args.dataset))


if __name__ == "__main__":
    sys.exit(main())
