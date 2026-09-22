"""The three Editor tools added for links, padding and coherence.

Each is a thin dialog over a Qt-free engine that has its own unit tests, so
what is checked here is the wiring: that the dialog offers what the engine
found, that ticking and applying reaches the engine, and that the three
actions exist on the Tools menu.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from bidsmgr.gui.coherence_dialog import CoherenceDialog
from bidsmgr.gui.linkage_dialog import LinkageDialog
from bidsmgr.gui.pad_values_dialog import PadValuesDialog

pytestmark = pytest.mark.gui


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """Two fieldmaps and three runs, timed so the rule has something to say."""
    root = tmp_path / "ds"
    (root / "sub-001/fmap").mkdir(parents=True)
    (root / "sub-001/func").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.10.0"})
    )
    (root / "participants.tsv").write_text("participant_id\nsub-001\n")

    def write(rel: str, meta: dict) -> None:
        path = root / rel
        path.write_bytes(b"x")
        path.with_name(path.name.split(".")[0] + ".json").write_text(
            json.dumps(meta)
        )

    for run, time in ((1, "09:33:59"), (2, "09:57:39"), (3, "10:17:1")):
        write(f"sub-001/func/sub-001_task-x_run-{run}_bold.nii.gz",
              {"AcquisitionTime": time})
    for fm, time in ((1, "09:31:31"), (2, "09:54:53")):
        write(f"sub-001/fmap/sub-001_run-{fm}_phasediff.nii.gz",
              {"AcquisitionTime": time})
    (root / "sub-001/sub-001_scans.tsv").write_text(
        "filename\n"
        + "".join(
            f"func/sub-001_task-x_run-{r}_bold.nii.gz\n" for r in (1, 2, 3)
        )
        + "".join(
            f"fmap/sub-001_run-{f}_phasediff.nii.gz\n" for f in (1, 2)
        )
    )
    return root


def _fmap(root: Path, n: int) -> Path:
    return root / f"sub-001/fmap/sub-001_run-{n}_phasediff.json"


class TestReferencesDialog:
    """It draws the RELATIONSHIP, in both directions.

    Two earlier versions showed a flat list of sidecars and a field name,
    which says a relationship exists and nothing about what it relates.
    """

    def _tree_rows(self, dlg):
        rows = []

        def walk(item, depth):
            for i in range(item.childCount()):
                child = item.child(i)
                rows.append((depth, child.text(0), child.text(1), child.text(2)))
                walk(child, depth + 1)

        for i in range(dlg._tree.topLevelItemCount()):
            top = dlg._tree.topLevelItem(i)
            rows.append((0, top.text(0), top.text(1), top.text(2)))
            walk(top, 1)
        return rows

    def test_it_opens_on_the_whole_dataset(self, qtbot, dataset):
        """No file has to be selected first."""
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        assert dlg._tree.topLevelItemCount() >= 1

    def test_files_are_grouped_by_subject(self, qtbot, dataset):
        """The group key is DIRECTORIES only: a filename starts with
        ``sub-`` too, and including it gave every file its own group."""
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        tops = [dlg._tree.topLevelItem(i).text(0)
                for i in range(dlg._tree.topLevelItemCount())]
        assert tops == ["sub-001"]

    def test_an_outgoing_reference_names_its_target(self, qtbot, dataset):
        from bidsmgr.editor import linkage

        linkage.apply_links(dataset, [(
            _fmap(dataset, 1), "IntendedFor",
            ["bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"],
        )])
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        arrows = [r for r in self._tree_rows(dlg) if r[1].startswith("\u2192 ")]
        assert any("run-1_bold" in r[1] for r in arrows)
        assert any(r[3] == "ok" for r in arrows)

    def test_the_reverse_direction_is_shown(self, qtbot, dataset):
        """The question nothing else answers: has this run got a fieldmap?"""
        from bidsmgr.editor import linkage

        linkage.apply_links(dataset, [(
            _fmap(dataset, 1), "IntendedFor",
            ["bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"],
        )])
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        incoming = [r for r in self._tree_rows(dlg) if r[1].startswith("\u2190 ")]
        assert incoming
        assert "incoming" in incoming[0][2]

    def test_a_missing_target_is_flagged(self, qtbot, dataset):
        from bidsmgr.editor import linkage

        linkage.apply_links(dataset, [(
            _fmap(dataset, 1), "IntendedFor", ["bids::sub-001/func/gone.nii.gz"],
        )])
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        assert any(r[3] == "missing" for r in self._tree_rows(dlg))

    def test_what_the_times_imply_but_nothing_says(self, qtbot, dataset):
        """The fixture's fieldmaps carry no IntendedFor at all."""
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        assert any(r[3] == "implied, not set" for r in self._tree_rows(dlg))

    def test_the_summary_counts_the_states(self, qtbot, dataset):
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        assert "implied, not set" in dlg._summary.text()

    def test_the_editor_is_off_until_a_file_is_selected(self, qtbot, dataset):
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        assert not dlg._save_btn.isEnabled()

    def test_selecting_a_child_row_edits_its_parent_file(self, qtbot, dataset):
        """A child row is a relationship, not a file."""
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        top = dlg._tree.topLevelItem(0)
        parent = top.child(0)
        dlg._tree.setCurrentItem(parent.child(0))
        assert dlg._current is not None
        assert dlg._save_btn.isEnabled()

    def test_saving_writes_a_bids_uri(self, qtbot, dataset):
        dlg = LinkageDialog(dataset, _fmap(dataset, 1))
        qtbot.addWidget(dlg)
        dlg._on_propose()
        dlg._on_save()
        assert dlg.changed_count() == 1
        assert json.loads(_fmap(dataset, 1).read_text())["IntendedFor"] == [
            "bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"]

    def test_pointing_at_nothing_removes_the_field(self, qtbot, dataset):
        from bidsmgr.editor import linkage

        linkage.apply_links(dataset, [(
            _fmap(dataset, 1), "IntendedFor",
            ["bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"],
        )])
        dlg = LinkageDialog(dataset, _fmap(dataset, 1))
        qtbot.addWidget(dlg)
        dlg._set_all(False)
        dlg._on_save()
        assert "IntendedFor" not in json.loads(_fmap(dataset, 1).read_text())


class TestIndexWidthsDialog:
    def test_every_index_the_schema_defines_gets_a_row(self, qtbot, dataset):
        """All of them, not only those in use. Listing only what the dataset
        already has made a dataset with one ``run`` read as though ``run``
        were the only index entity there is."""
        from bidsmgr.editor.values import index_entities

        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert list(dlg._widths) == index_entities()
        assert "acq" not in dlg._widths, "a label has no width"

    def test_an_unused_entity_is_shown_and_disabled(self, qtbot, dataset):
        """Shown and disabled, not hidden: the difference between "you have
        none of these" and "these do not exist"."""
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert dlg._widths["run"].isEnabled()
        assert not dlg._widths["echo"].isEnabled()

    def test_the_editor_and_settings_offer_the_same_list(self, qtbot, dataset):
        """One list, read from the schema, so the two cannot disagree."""
        from bidsmgr.gui.app_settings import AppSettings
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        settings = SettingsDialog(AppSettings())
        qtbot.addWidget(settings)
        assert list(dlg._widths) == list(settings._index_widths)

    def test_a_consistent_dataset_says_so(self, qtbot, dataset):
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert "already uses one width" in dlg._warning.text()

    def test_the_width_starts_at_the_widest_in_use(self, qtbot, dataset):
        """So pressing Apply unchanged settles a dataset that disagrees with
        itself and leaves a consistent one alone."""
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert dlg._widths["run"].value() == 1
        assert not dlg._ok.isEnabled(), "nothing to change"

    def test_an_inconsistent_dataset_is_named(self, qtbot, dataset):
        f = dataset / "sub-001/func/sub-001_task-x_run-3_bold.nii.gz"
        f.rename(f.with_name("sub-001_task-x_run-30_bold.nii.gz"))
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert "disagree with themselves" in dlg._warning.text()
        assert dlg._widths["run"].value() == 2

    def test_a_clash_that_padding_cannot_settle_says_why(self, qtbot, dataset):
        """run-1 beside run-01 is the mess this is for, and padding would
        fuse two different runs."""
        (dataset / "sub-001/func/sub-001_task-x_run-01_bold.nii.gz").write_bytes(b"x")
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert "Cannot pad" in dlg._status.text()
        assert not dlg._ok.isEnabled()

    def test_the_preview_lists_what_would_change(self, qtbot, dataset):
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        dlg._widths["run"].setValue(2)
        head = dlg._tree.topLevelItem(0)
        assert head.text(0) == "run"
        rows = [(head.child(i).text(0), head.child(i).text(1))
                for i in range(head.childCount())]
        assert rows == [("run-1", "run-01"), ("run-2", "run-02"),
                        ("run-3", "run-03")]


class TestReplaceValueDialog:
    def test_it_offers_the_scopes(self, qtbot, dataset):
        from bidsmgr.gui.replace_value_dialog import ReplaceValueDialog

        dlg = ReplaceValueDialog(dataset)
        qtbot.addWidget(dlg)
        labels = [dlg._scope.itemText(i) for i in range(dlg._scope.count())]
        assert labels == ["The whole dataset", "sub-001"]

    def test_values_are_listed_with_their_file_counts(self, qtbot, dataset):
        from bidsmgr.gui.replace_value_dialog import ReplaceValueDialog

        dlg = ReplaceValueDialog(dataset, entity="task")
        qtbot.addWidget(dlg)
        assert dlg._old.itemText(0).startswith("x  (")
        assert "file(s)" in dlg._old.itemText(0)

    def test_replacing_renames_every_matching_file(self, qtbot, dataset):
        from bidsmgr.gui.replace_value_dialog import ReplaceValueDialog

        dlg = ReplaceValueDialog(dataset, entity="task")
        qtbot.addWidget(dlg)
        dlg._new.setText("rest")
        dlg._on_apply()
        assert dlg.applied_count() > 0
        assert (dataset / "sub-001/func/sub-001_task-rest_run-1_bold.nii.gz").exists()

    def test_the_same_value_is_refused(self, qtbot, dataset):
        from bidsmgr.gui.replace_value_dialog import ReplaceValueDialog

        dlg = ReplaceValueDialog(dataset, entity="task")
        qtbot.addWidget(dlg)
        dlg._new.setText("x")
        assert "already has" in dlg._status.text()
        assert not dlg._ok.isEnabled()


class TestCoherenceDialog:
    def test_a_clean_dataset_says_so(self, qtbot, dataset):
        """The fixture's fieldmaps carry no IntendedFor, which the times DO
        imply, so it is settled first and then checked."""
        from bidsmgr.editor import coherence

        coherence.apply(dataset, coherence.check(dataset))
        dlg = CoherenceDialog(dataset)
        qtbot.addWidget(dlg)
        assert dlg._tree.topLevelItemCount() == 0
        assert "Everything agrees" in dlg._status.text()
        assert not dlg._ok.isEnabled()

    def test_selecting_a_finding_shows_the_exact_change(self, qtbot, dataset):
        """The whole point of the rebuild: not "set TaskName" but what it
        says now and what it would say."""
        side = dataset / "sub-001/func/sub-001_task-x_run-1_bold.json"
        meta = json.loads(side.read_text())
        meta["TaskName"] = "something else"
        side.write_text(json.dumps(meta))

        dlg = CoherenceDialog(dataset)
        qtbot.addWidget(dlg)
        for item in dlg._leaves():
            finding = item.data(0, Qt.ItemDataRole.UserRole)
            if finding and finding.kind.name == "TASK_NAME_MISMATCH":
                dlg._show_detail(finding)
                rows = [
                    (dlg._detail.topLevelItem(i).text(0),
                     dlg._detail.topLevelItem(i).text(1))
                    for i in range(dlg._detail.topLevelItemCount())
                ]
                assert rows == [('TaskName: "something else"',
                                 'TaskName: "x"')]
                return
        raise AssertionError("the TaskName mismatch was not found")

    def test_findings_are_grouped_by_kind(self, qtbot, dataset):
        (dataset / "sub-001/func/sub-001_task-x_run-3_bold.nii.gz").unlink()
        (dataset / "sub-001/func/sub-001_task-x_run-3_bold.json").unlink()
        (dataset / "participants.tsv").write_text(
            "participant_id\nsub-001\nsub-999\n"
        )
        dlg = CoherenceDialog(dataset)
        qtbot.addWidget(dlg)
        heads = [dlg._tree.topLevelItem(i).text(0)
                 for i in range(dlg._tree.topLevelItemCount())]
        assert any("participants row" in h for h in heads)

    def test_repairing_settles_it(self, qtbot, dataset, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox

        monkeypatch.setattr(
            QMessageBox, "question",
            lambda *a, **k: QMessageBox.StandardButton.Ok,
        )
        (dataset / "participants.tsv").write_text(
            "participant_id\nsub-001\nsub-999\n"
        )
        dlg = CoherenceDialog(dataset)
        qtbot.addWidget(dlg)
        assert dlg._checked()
        dlg._on_apply()
        assert dlg.applied_count() >= 1
        assert "sub-999" not in (dataset / "participants.tsv").read_text()
        assert "Everything agrees" in dlg._status.text()

    def test_nothing_is_written_by_opening_it(self, qtbot, dataset):
        before = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
        dlg = CoherenceDialog(dataset)
        qtbot.addWidget(dlg)
        after = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
        assert before == after


class TestTheToolsMenu:
    def test_the_three_actions_exist(self, qtbot):
        from bidsmgr.gui.editor_panel import EditorPanel

        panel = EditorPanel()
        qtbot.addWidget(panel)
        labels = [a.text() for a in panel._tools_menu.actions()]
        for wanted in ("References (IntendedFor, Sources...)...",
                       "Find and replace a value...",
                       "Index widths...", "Check coherence..."):
            assert wanted in labels, labels
