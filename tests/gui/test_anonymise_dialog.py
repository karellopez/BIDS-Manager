"""Tools, Anonymise metadata: the plan shown, then one undoable step."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from bidsmgr.gui.anonymise_dialog import AnonymiseDialog
from bidsmgr.project.operations import ORIGINALS_DIR, peek_undo, undo_last

pytestmark = pytest.mark.gui


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    anat = root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_T1w.json").write_text(json.dumps({
        "AcquisitionDateTime": "2025-05-26T11:14:30", "PatientName": "Doe^Jane",
        "ProtocolName": "t1 Jane", "EchoTime": 0.003}))
    (root / "sub-02" / "anat").mkdir(parents=True)
    (root / "sub-02" / "anat" / "sub-02_T1w.json").write_text(json.dumps({
        "AcquisitionDateTime": "2025-06-01T09:00:00", "PatientID": "X123"}))
    (root / "participants.tsv").write_text("participant_id\tgiven_name\tage\n"
                                           "sub-01\tJane\t93\nsub-02\tJoe\t30\n")
    (root / "dataset_description.json").write_text('{"Name": "d", "BIDSVersion": "1.10.0"}')
    (root / "sourcedata").mkdir()
    return root


def _open(qtbot, root: Path) -> AnonymiseDialog:
    dlg = AnonymiseDialog(root)
    qtbot.addWidget(dlg)
    qtbot.waitUntil(lambda: not dlg.busy(), timeout=20_000)
    return dlg


def _yes(monkeypatch) -> list:
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.Ok)
    return asked


def test_every_change_is_shown_before_anything_is_written(qtbot, ds: Path) -> None:
    before = {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()}
    dlg = _open(qtbot, ds)
    files = {dlg._tree.topLevelItem(i).text(0): dlg._tree.topLevelItem(i)
             for i in range(dlg._tree.topLevelItemCount())}
    assert set(files) == {"participants.tsv", "sub-01/anat/sub-01_T1w.json",
                          "sub-02/anat/sub-02_T1w.json"}
    side = files["sub-01/anat/sub-01_T1w.json"]
    rows = [side.child(i).text(0) for i in range(side.childCount())]
    assert any(r.startswith("PatientName") for r in rows)
    assert any(r.startswith("AcquisitionDateTime") for r in rows)
    others = [dlg._other.topLevelItem(i).text(0) for i in range(dlg._other.topLevelItemCount())]
    assert any(t.startswith("To read by hand") for t in others), "ProtocolName"
    assert any(t.startswith("Cannot be changed here") for t in others), "sourcedata/"
    assert dlg._ok.text() == "Anonymise 3 files"
    assert {p: p.read_bytes() for p in before} == before


def test_the_preset_plans_again(qtbot, ds: Path) -> None:
    dlg = _open(qtbot, ds)
    assert "midnight" not in dlg._what.text()
    dlg._preset.setCurrentIndex(dlg._preset.findData("strict"))
    qtbot.waitUntil(lambda: not dlg.busy(), timeout=20_000)
    assert "midnight" in dlg._what.text()


def test_anonymising_is_one_undoable_step(qtbot, ds: Path, monkeypatch) -> None:
    _yes(monkeypatch)
    before = {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()}
    dlg = _open(qtbot, ds)
    dlg._on_apply()
    qtbot.waitUntil(lambda: not dlg.busy(), timeout=20_000)
    assert dlg.written() == 3
    side = json.loads((ds / "sub-01/anat/sub-01_T1w.json").read_text())
    assert "PatientName" not in side and side["AcquisitionDateTime"] < "1926"
    assert peek_undo(ds).label == "Anonymise metadata (Recommended)"
    undo_last(ds)
    assert {p: p.read_bytes() for p in before} == before


def test_an_unticked_file_is_left_alone(qtbot, ds: Path, monkeypatch) -> None:
    _yes(monkeypatch)
    dlg = _open(qtbot, ds)
    for i in range(dlg._tree.topLevelItemCount()):
        item = dlg._tree.topLevelItem(i)
        if item.text(0).startswith("sub-02"):
            item.setCheckState(0, Qt.CheckState.Unchecked)
    original = (ds / "sub-02/anat/sub-02_T1w.json").read_bytes()
    dlg._on_apply()
    qtbot.waitUntil(lambda: not dlg.busy(), timeout=20_000)
    assert (ds / "sub-02/anat/sub-02_T1w.json").read_bytes() == original


def test_forgetting_the_originals_asks_and_deletes_them(qtbot, ds: Path, monkeypatch) -> None:
    asked = _yes(monkeypatch)
    dlg = _open(qtbot, ds)
    dlg._forget.setChecked(True)
    dlg._on_apply()
    qtbot.waitUntil(lambda: not dlg.busy(), timeout=20_000)
    assert "DELETED" in asked[0]
    assert not (ds / ORIGINALS_DIR).exists()
    assert peek_undo(ds) is None, "nothing left to undo, as it said"


def test_the_editor_offers_it_with_the_face_tools(qtbot) -> None:
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    labels = [a.text() for a in panel._tools_menu.actions()]
    assert labels.index("Anonymise metadata...") == labels.index("Put the face back...") + 1
