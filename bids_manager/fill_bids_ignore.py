from __future__ import annotations

import argparse
from pathlib import Path


def interactive_fill(bids_root: Path) -> None:
    """Interactively add paths to ``.bidsignore`` within ``bids_root``."""
    ignore_file = bids_root / ".bidsignore"
    existing = set()
    if ignore_file.exists():
        existing = {line.strip() for line in ignore_file.read_text().splitlines() if line.strip()}

    all_files = [p.relative_to(bids_root) for p in bids_root.rglob('*') if p.is_file()]
    while True:
        pattern = input("Search string (empty to finish): ").strip()
        if not pattern:
            break
        matches = [p for p in all_files if pattern in p.as_posix()]
        if not matches:
            print("No matches")
            continue
        for idx, p in enumerate(matches, 1):
            mark = "*" if p.as_posix() in existing else " "
            print(f"{idx:3d}{mark} {p.as_posix()}")
        sel = input("Select numbers separated by space: ").strip()
        if not sel:
            continue
        for num in sel.split():
            try:
                i = int(num) - 1
            except ValueError:
                continue
            if 0 <= i < len(matches):
                existing.add(matches[i].as_posix())
    if existing:
        ignore_file.write_text("\n".join(sorted(existing)) + "\n")
        print(f"Updated {ignore_file}")
    else:
        print("No entries added")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively populate .bidsignore")
    parser.add_argument("bids_root", help="Path to BIDS dataset")
    args = parser.parse_args()
    interactive_fill(Path(args.bids_root))


if __name__ == "__main__":
    main()
