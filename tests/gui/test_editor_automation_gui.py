"""The Editor's two views must hold the same fields, and the tree must be
able to show the files a dataset hides.

The first is the defect this work started from: with a validation report bound,
the BIDS form offered every field the schema declares for the datatype and the
Tree view rendered the file, so the same sidecar showed 130 rows in one and 90
in the other with nothing to say why.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor.types import (
    FieldLevel,
    FileVerdict,
    Severity,
    SidecarField,
    ValidationReport,
)
from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.widgets.bids_tree_pane import BidsTreePane, is_hidden_name
from bidsmgr.gui.widgets.sidecar_form_pane import (
    SCOPE_ABSENT,
    SCOPE_ALL,
    SCOPE_PRESENT,
    SidecarFormPane,
)
from bidsmgr.gui.widgets.sidecar_row import SidecarRow

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    anat = root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_T1w.json").write_text(
        json.dumps({"EchoTime": 0.03, "Manufacturer": "Siemens"})
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    (root / ".bidsignore").write_text("derivatives/\n")
    (root / ".bidsmgr").mkdir()
    return root


def _report(root: Path) -> ValidationReport:
    """A verdict declaring one field the file does not carry."""
    return ValidationReport(
        bids_root=root,
        files=[FileVerdict(
            path=Path("sub-01/anat/sub-01_T1w.json"),
            severity=Severity.WARN,
            datatype="anat",
            suffix="T1w",
            sidecar_fields=[
                SidecarField(
                    level=FieldLevel.REQUIRED, name="MagneticFieldStrength",
                    value=None, present=False, value_kind="missing",
                ),
                SidecarField(
                    level=FieldLevel.OPTIONAL, name="EchoTime",
                    value=0.03, present=True, value_kind="number",
                ),
            ],
        )],
    )


def _form_keys(pane: SidecarFormPane) -> list[str]:
    return [r.key for r in pane.findChildren(SidecarRow)]


def _tree_keys(pane: SidecarFormPane) -> list[str]:
    tree = pane._tree_view
    return [
        tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())
    ]


# ---------------------------------------------------------------------------
# The two views agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", [SCOPE_ALL, SCOPE_PRESENT, SCOPE_ABSENT])
def test_both_views_hold_the_same_fields(qtbot, dataset: Path, scope) -> None:
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_field_scope(scope)
    pane.set_file(
        dataset / "sub-01" / "anat" / "sub-01_T1w.json", dataset,
        _report(dataset),
    )
    form = sorted(_form_keys(pane))
    pane._apply_view_mode("tree", persist=False)
    tree = sorted(_tree_keys(pane))
    assert form == tree, f"views disagree in scope {scope}"


def test_all_scope_shows_more_than_the_file_carries(
    qtbot, dataset: Path,
) -> None:
    """The point of the BIDS form: it offers what the standard declares."""
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_field_scope(SCOPE_ALL)
    pane.set_file(
        dataset / "sub-01" / "anat" / "sub-01_T1w.json", dataset,
        _report(dataset),
    )
    keys = _form_keys(pane)
    assert "MagneticFieldStrength" in keys   # declared, absent
    assert "Manufacturer" in keys            # in the file, not in the verdict


def test_present_scope_is_exactly_the_file(qtbot, dataset: Path) -> None:
    fp = dataset / "sub-01" / "anat" / "sub-01_T1w.json"
    on_disk = set(json.loads(fp.read_text()))
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_field_scope(SCOPE_PRESENT)
    pane.set_file(fp, dataset, _report(dataset))
    assert set(_form_keys(pane)) == on_disk


def test_a_declared_but_absent_field_is_not_written_back(
    qtbot, dataset: Path,
) -> None:
    """The tree shows absent fields so the views match. Showing them must not
    add them to the file."""
    fp = dataset / "sub-01" / "anat" / "sub-01_T1w.json"
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_field_scope(SCOPE_ALL)
    pane.set_file(fp, dataset, _report(dataset))
    pane._apply_view_mode("tree", persist=False)
    assert "MagneticFieldStrength" in _tree_keys(pane)
    written = pane._tree_view.to_dict()
    assert "MagneticFieldStrength" not in written
    assert set(written) == set(json.loads(fp.read_text()))


# ---------------------------------------------------------------------------
# Hidden files
# ---------------------------------------------------------------------------


def test_hidden_names_are_one_rule() -> None:
    assert is_hidden_name(".bidsignore")
    assert is_hidden_name(".bidsmgr")
    assert is_hidden_name(".git")
    assert not is_hidden_name("sub-01")
    assert not is_hidden_name("dataset_description.json")


def test_the_tree_can_show_and_hide_dotfiles(qtbot, dataset: Path) -> None:
    def _top_level_names() -> list[str]:
        pane = BidsTreePane()
        qtbot.addWidget(pane)
        pane.set_root(dataset)
        top = pane._tree.topLevelItem(0)
        return [top.child(i).text(0) for i in range(top.childCount())]

    previous = AppSettings.load().editor_show_hidden
    try:
        AppSettings.remember_editor_show_hidden(False)
        assert not [n for n in _top_level_names() if n.startswith(".")]

        AppSettings.remember_editor_show_hidden(True)
        shown = _top_level_names()
        assert ".bidsignore" in shown
        assert ".bidsmgr" in shown
    finally:
        AppSettings.remember_editor_show_hidden(previous)
