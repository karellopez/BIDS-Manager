"""Compatibility wrapper for launching the modular BIDS Manager GUI."""

from bids_manager.GUI.main_window import BIDSManager, main

__all__ = ["BIDSManager", "main"]


if __name__ == "__main__":
    main()
