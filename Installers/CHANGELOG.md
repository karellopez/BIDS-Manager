# Installer changelog

The installers have their own version, separate from BIDS Manager's: an
installer puts BIDS Manager on a computer, and BIDS Manager then updates
itself. `INSTALLER_VERSION` in each script, the first heading below and the
installers in `Installers.zip` always agree; `tests/unit/test_installers.py`
checks it.

## 2.0.0 (2026-10-16)

The installers are built on uv instead of a portable Python kept in this
repository.

- uv 0.13.1 is downloaded into the install folder and checked against its
  published SHA-256 before it is used. A uv already on the computer is never
  called, and its settings (`uv.toml`, `UV_*` variables) are ignored.
- Python 3.12 comes from that uv, into the install folder. Nothing is added
  to the system's `PATH` or the Windows registry.
- New platforms: Intel Macs, Linux on ARM64, Windows 11 on ARM (through its
  x64 emulation, because dcm2niix has no ARM64 build for Windows).
- No administrator rights anywhere. On Windows the installer no longer
  installs the Visual C++ 2008 runtime that EyeLink `.edf` reading needs
  (that required elevation); it says when the runtime is missing.
- Running the installer again updates BIDS Manager to the newest release.
  Before, it reinstalled the version already there.
- An installation from installer 1.x is converted in place: its portable
  Python 3.10 is removed and the environment is rebuilt on Python 3.12.
- `installation.json` in the install folder records the installer, uv,
  Python and BIDS Manager versions, and how to call that uv.
- The scripts are tracked uncompressed in `Installers/`; `Installers.zip` is
  built from them by `tools/build_installers.py`, without macOS metadata.
- Optional settings for testing and for administrators: `BIDSMGR_INSTALL_DIR`,
  `BIDSMGR_PYTHON`, `BIDSMGR_PACKAGE`, `BIDSMGR_YES`, `BIDSMGR_NO_SHORTCUTS`.

## 1.0.2 (Windows) and 1.0.1 (macOS, Linux)

Download a portable Python 3.10 from `external/python-embed/` in this
repository, create an environment, `pip install bids-manager`, and register a
launcher. Apple Silicon Macs, Linux x86_64 and Windows x64 only. The Windows
installer 1.0.2 added the Visual C++ 2008 runtime for EyeLink, installed with
administrator rights.
