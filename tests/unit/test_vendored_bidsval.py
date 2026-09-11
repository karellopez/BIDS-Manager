"""bidsval is vendored, and BIDS Manager must reach only the vendored copy.

bidsval is BIDS Manager's own package and the single implementation of two
things the tool rests on: validation, and the interpretation of the BIDS
schema. Depending on it through PyPI meant a schema fix could not be tested
from here without publishing a release first.

Two properties matter. Nothing in ``bidsmgr`` may import the INSTALLED
package, or the two copies diverge silently on a machine that has both; and
the vendored copy has to be the whole thing, not the parts that happened to be
needed on the day it was copied.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "bidsmgr"
VENDORED = PACKAGE / "vendor" / "bidsval"


def _imports(path: Path) -> list[str]:
    """Every absolute module name imported by a file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.append(node.module or "")
    return out


def _reaches_installed(paths) -> list[str]:
    return [
        str(path)
        for path in paths
        for name in _imports(path)
        if name == "bidsval" or name.startswith("bidsval.")
    ]


def test_nothing_reaches_the_installed_package() -> None:
    """A machine with both copies must not quietly use the wrong one."""
    sources = [
        p for p in PACKAGE.rglob("*.py")
        if "vendor" not in p.relative_to(PACKAGE).parts
    ]
    assert not _reaches_installed(sources)


def test_the_vendored_copy_imports_itself_relatively() -> None:
    """Which is why it works at a new path unmodified. An absolute
    self-import would reach the INSTALLED package from inside the vendored
    one, mixing the two."""
    assert not _reaches_installed(VENDORED.rglob("*.py"))


def test_the_vendored_copy_is_importable_and_whole() -> None:
    from bidsmgr.vendor import bidsval

    assert bidsval.__version__
    for name in ("schema", "validate", "issues", "report", "rules", "context"):
        __import__(f"bidsmgr.vendor.bidsval.{name}")


def test_the_schema_namespace_is_there() -> None:
    """``bidsmgr/schema`` is a thin adapter over this. If the namespace went
    missing the adapter would silently fall back and BIDS facts would get
    derived twice, which is what guard 8 exists to prevent."""
    from bidsmgr.vendor.bidsval import schema as bv

    for name in (
        "sidecar_fields", "dataset_description_fields", "metadata_by_name",
        "short_to_long", "entity_pattern", "field_applies", "datatypes",
        "suffixes", "bids_version",
    ):
        assert hasattr(bv, name), name


def test_the_licence_travels_with_the_code() -> None:
    assert (VENDORED / "LICENSE").is_file()


def test_no_packaging_metadata_was_copied() -> None:
    """Source and licence only. ``__pycache__`` is not checked: Python makes
    it on import and it is gitignored."""
    junk = [
        str(p.relative_to(VENDORED)) for p in VENDORED.rglob("*")
        if p.name == "PKG-INFO" or p.name.endswith(".egg-info")
        or p.suffix in (".whl", ".tar.gz")
    ]
    assert not junk, junk


def test_the_adapter_still_answers_schema_questions() -> None:
    """The end the vendoring is for: BIDS facts still arrive, from one place."""
    from bidsmgr import schema

    assert schema.bids_version()
    assert len(schema.sidecar_fields("anat", "T1w")) > 50
    assert "sub" in schema.entity_keys()


def test_validation_still_runs_through_the_vendored_engine(
    tmp_path: Path,
) -> None:
    import json

    from bidsmgr.editor.validator import validate

    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"")
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        json.dumps({"EchoTime": 0.03})
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    report = validate(root)
    assert report.files
    assert set(report.counts) >= {"ok", "warn", "err"}


@pytest.mark.parametrize("module", ["mne", "nibabel", "pandas", "yaml"])
def test_it_needs_no_dependency_we_did_not_already_ship(module: str) -> None:
    __import__(module)
