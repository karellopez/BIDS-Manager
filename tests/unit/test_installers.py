"""The installers agree with each other, with their changelog and with the zip.

Three scripts in three languages pin the same uv, carry the same version and
must never ask for administrator rights. Nothing but a test notices when one
of them drifts, and the zip on the website is a copy that can silently fall
behind the sources it was built from.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALLERS = ROOT / "Installers"
MACOS = INSTALLERS / "MacOS" / "install_BIDS_Manager.command"
LINUX = INSTALLERS / "Linux" / "install_BIDS_Manager.sh"
WINDOWS = INSTALLERS / "Windows" / "install_BIDS_Manager.bat"
UNIX = [MACOS, LINUX]
ALL = [MACOS, LINUX, WINDOWS]


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_installers", ROOT / "tools" / "build_installers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _installer_version(path: Path) -> str:
    pattern = (r"^\$InstallerVersion = '([^']+)'" if path == WINDOWS
               else r'^INSTALLER_VERSION="([^"]+)"')
    return re.search(pattern, _text(path), re.M).group(1)


def _uv_version(path: Path) -> str:
    pattern = r"^\$UvVersion = '([^']+)'" if path == WINDOWS else r'^UV_VERSION="([^"]+)"'
    return re.search(pattern, _text(path), re.M).group(1)


def _uv_checksums(path: Path) -> dict[str, str]:
    """Target triple -> SHA-256, as the script pins them."""
    text = _text(path)
    if path == WINDOWS:
        target = re.search(r"^\$UvTarget = '([^']+)'", text, re.M).group(1)
        return {target: re.search(r"^\$UvSha256 = '([^']+)'", text, re.M).group(1)}
    return {name.replace("_", "-").replace("x86-64", "x86_64"): value
            for name, value in re.findall(r'^UV_SHA256_(\w+)="([^"]+)"', text, re.M)}


def test_the_zip_holds_exactly_the_sources():
    builder = _builder()
    assert builder.contents(builder.ZIP) == builder.expected(), (
        "Installers.zip is out of date: run python tools/build_installers.py")


def test_the_zip_keeps_the_scripts_executable_and_carries_no_mac_metadata():
    names = _builder().contents()
    for name, (_data, mode) in names.items():
        assert "__MACOSX" not in name and "/._" not in name and ".DS_Store" not in name
        if name.endswith((".command", ".sh", ".bat")):
            assert mode == 0o755, name


def test_every_script_carries_the_changelog_version():
    newest = re.search(r"^## (\S+)", _text(INSTALLERS / "CHANGELOG.md"), re.M).group(1)
    assert {_installer_version(path) for path in ALL} == {newest}


def test_every_script_pins_the_same_uv_with_the_same_checksums():
    assert len({_uv_version(path) for path in ALL}) == 1
    pinned: dict[str, str] = {}
    for path in ALL:
        for target, digest in _uv_checksums(path).items():
            assert re.fullmatch(r"[0-9a-f]{64}", digest), (path.name, target)
            assert pinned.setdefault(target, digest) == digest, target
    assert set(pinned) == {
        "aarch64-apple-darwin", "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
    }


@pytest.mark.parametrize("path", UNIX, ids=lambda p: p.name)
def test_unix_scripts_never_run_anything_as_administrator(path):
    for number, line in enumerate(_text(path).splitlines(), 1):
        code = line.split("#", 1)[0].strip()
        if code.startswith("echo"):
            continue  # advice printed for the user, not a command run
        assert not re.search(r"(^|[|;&(`]\s*)sudo\b", code), f"{path.name}:{number}"
        assert not re.search(r"\b(su|doas|pkexec)\s", code), f"{path.name}:{number}"


def test_the_windows_script_never_asks_for_elevation():
    text = _text(WINDOWS)
    assert not re.search(r"runas", text, re.I)
    assert "Start-Process" not in text


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_every_script_isolates_its_uv(path):
    text = _text(path)
    for variable in ("UV_NO_CONFIG", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_MANAGED_PYTHON"):
        assert variable in text, variable
    # Variables set for another uv are removed before this one runs.
    assert "UV_*" in text
    # The Python it downloads is never put on PATH or in the registry.
    assert "--no-bin" in text
    if path == WINDOWS:
        assert "--no-registry" in text
    # An update installs the newest release, not the one already there.
    assert "--upgrade" in text


def test_the_windows_script_is_ascii_with_crlf_line_endings():
    data = WINDOWS.read_bytes()
    data.decode("ascii")
    assert data.count(b"\n") == data.count(b"\r\n")
    # The batch part ends where the PowerShell block comment closes.
    assert data.startswith(b"<# :")


@pytest.mark.parametrize("path", UNIX, ids=lambda p: p.name)
def test_unix_scripts_use_lf_line_endings(path):
    assert b"\r" not in path.read_bytes()
