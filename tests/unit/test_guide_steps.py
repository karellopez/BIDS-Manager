"""The guide's next step, decided from what each view reports.

One path per view (Home: where, name, create; Converter: raw data, scan,
check, convert; Editor: validate, fix), a busy step while the app works, and
a finished path marked done.
"""

from __future__ import annotations

import pytest

from bidsmgr.gui.guide_steps import next_step


def _keys(view, *facts):
    return [next_step(view, f).key for f in facts]


def test_home_walks_where_name_create() -> None:
    assert _keys("welcome", {}, {"location_chosen": True},
                 {"location_chosen": True, "name": True}) == [
        "home.location", "home.name", "home.create"]
    # Typing a name first accepts the folder shown.
    assert next_step("welcome", {"name": True}).key == "home.create"
    step = next_step("welcome", {})
    assert (step.number, step.total, step.counter) == (1, 3, "Step 1 of 3")
    assert step.target == "browse"


def test_home_with_a_project_open_points_to_the_converter() -> None:
    step = next_step("welcome", {"project": True, "project_name": "Motor_Study"})
    assert step.target == "converter" and "Motor_Study" in step.text
    assert step.counter == ""


def test_the_name_advice_is_one_home_accepts() -> None:
    """Home refuses spaces in a dataset name, so the example has none."""
    assert "My_Study" in next_step("welcome", {"location_chosen": True}).text


def test_the_converter_path() -> None:
    base = {"project": True}
    assert _keys(
        "converter",
        {},
        base,
        {**base, "raw_root": True},
        {**base, "raw_root": True, "rows": 28},
        {**base, "raw_root": True, "rows": 28, "reviewed": True},
        {**base, "raw_root": True, "rows": 28, "reviewed": True, "converted": True},
    ) == ["convert.no_project", "convert.raw", "convert.scan", "convert.review",
          "convert.run", "convert.done"]
    assert [next_step("converter", {**base, "raw_root": True}).target,
            next_step("converter", {**base, "raw_root": True, "rows": 3}).target] == [
        "scan", "table"]


def test_busy_steps_point_at_the_busy_control() -> None:
    scanning = next_step("converter", {"project": True, "raw_root": True, "scanning": True})
    converting = next_step("converter", {"project": True, "raw_root": True, "rows": 2,
                                         "reviewed": True, "converting": True})
    validating = next_step("editor", {"root": True, "subjects": True, "validating": True})
    assert (scanning.target, scanning.busy) == ("scan", True)
    assert (converting.target, converting.busy) == ("run", True)
    assert (validating.target, validating.busy) == ("validate", True)
    # Busy wins over what is otherwise next.
    assert next_step("converter", {"project": True, "scanning": True}).key == "convert.scanning"


def test_a_finished_conversion_sends_to_the_editor() -> None:
    step = next_step("converter", {"project": True, "raw_root": True, "rows": 1,
                                   "converted": True})
    assert (step.target, step.done) == ("editor", True)


def test_the_editor_path() -> None:
    base = {"root": True, "subjects": True}
    assert _keys("editor", {}, {"root": True}, base,
                 {**base, "validated": True, "errors": 3},
                 {**base, "validated": True, "warnings": 2},
                 {**base, "validated": True}) == [
        "editor.open", "editor.empty", "editor.validate", "editor.fix",
        "editor.valid", "editor.valid"]
    fix = next_step("editor", {**base, "validated": True, "errors": 1})
    assert "1 error to fix" in fix.text and fix.target == "findings"
    assert "3 errors" in next_step("editor", {**base, "validated": True, "errors": 3}).text
    valid = next_step("editor", {**base, "validated": True, "warnings": 2})
    assert valid.done and "2 warnings" in valid.text and valid.target == "findings"
    clean = next_step("editor", {**base, "validated": True})
    assert clean.done and clean.target == "validate"


def test_a_folder_opened_to_look_at_has_no_steps() -> None:
    assert next_step("editor", {"root": True, "view_only": True}) is None
    assert next_step("nowhere", {}) is None


@pytest.mark.parametrize("view", ["welcome", "converter", "editor"])
def test_every_text_is_plain(view) -> None:
    """No em dashes, and every step names its control."""
    facts = [{}, {"project": True}, {"project": True, "raw_root": True},
             {"root": True, "subjects": True, "validated": True, "errors": 2}]
    for f in facts:
        step = next_step(view, f)
        if step is None:
            continue
        assert "—" not in step.title + step.text
        assert step.target and step.title and step.text


# ---------------------------------------------------------------------------
# A new raw folder, the question, and the goals it sets
# ---------------------------------------------------------------------------

from bidsmgr.gui.guide_steps import choices, goal_done  # noqa: E402

_TABLE = {"project": True, "raw_root": True, "rows": 12}


def test_a_raw_folder_chosen_after_a_scan_leads_to_scan_again() -> None:
    for facts in ({**_TABLE, "raw_picked": True},
                  {**_TABLE, "reviewed": True, "raw_picked": True},
                  {**_TABLE, "reviewed": True, "converted": True, "raw_picked": True}):
        assert next_step("converter", facts).key == "convert.scan"


def test_one_path_asks_nothing() -> None:
    assert choices("converter", {"project": True}) == []
    assert choices("converter", {**_TABLE, "scanning": True}) == []
    assert choices("welcome", {}) == []
    assert choices("editor", {"root": True}) == []                       # nothing converted
    assert choices("editor", {"root": True, "subjects": True, "view_only": True}) == []


def test_the_converter_asks_with_the_paths_next_step_recommended() -> None:
    keys = lambda cs: [c.key for c in cs]                           # noqa: E731
    best = lambda cs: [c.key for c in cs if c.recommended]          # noqa: E731
    asked = choices("converter", _TABLE)
    assert keys(asked) == ["scan", "review", "convert"] and best(asked) == ["review"]
    asked = choices("converter", {**_TABLE, "reviewed": True, "subjects": True})
    assert keys(asked) == ["scan", "review", "convert", "editor"] and best(asked) == ["convert"]
    assert best(choices("converter", {**_TABLE, "reviewed": True, "converted": True})) == [
        "editor"]


def test_the_editor_and_home_ask() -> None:
    base = {"root": True, "subjects": True}
    first = choices("editor", base)
    assert [c.key for c in first] == ["validate", "edit", "tools"]
    assert first[0].recommended and first[0].label == "Validate the dataset"
    later = choices("editor", {**base, "validated": True, "errors": 2, "warnings": 5})
    fix = next(c for c in later if c.key == "fix")
    assert fix.recommended and "2 errors, 5 warnings" in fix.detail
    assert later[0].label == "Validate again"
    home = choices("welcome", {"project": True, "project_name": "Motor_Study"})
    assert [c.key for c in home] == ["continue", "create", "open"]
    assert home[0].label == "Continue with Motor_Study" and home[0].recommended


def test_a_goal_steers_the_steps() -> None:
    step = next_step("converter", {**_TABLE, "reviewed": True}, "scan")
    assert (step.key, step.target, step.heading) == (
        "convert.raw_new", "raw_folder", "Scan new data · Step 1 of 2")
    assert next_step("converter", {**_TABLE, "raw_picked": True}, "scan").key == \
        "convert.scan_new"
    assert next_step("converter", _TABLE, "convert").target == "run"       # no review first
    assert next_step("converter", _TABLE, "editor").target == "editor"
    assert next_step("converter", _TABLE, "review").kicker == "Check the table"
    base = {"root": True, "subjects": True, "validated": True, "errors": 1}
    assert next_step("editor", base, "edit").target == "tree"
    assert next_step("editor", {**base, "file_open": True}, "edit").target == "viewer"
    assert next_step("editor", base, "tools").target == "tools"
    assert next_step("editor", base, "validate").key == "editor.validate_goal"
    assert next_step("editor", base, "fix").target == "findings"
    assert next_step("welcome", {"project": True}, "create").key == "home.location"
    assert next_step("welcome", {"project": True}, "open").target == "open"
    # Busy beats a goal: the app is working on it.
    assert next_step("converter", {**_TABLE, "scanning": True}, "scan").busy


def test_a_goal_ends_when_it_is_reached() -> None:
    before = {**_TABLE, "tables": 1, "conversions": 0, "reports": 0}
    assert not goal_done("converter", "scan", before, {**before, "raw_picked": True})
    assert not goal_done("converter", "scan", before, {**before, "tables": 2, "scanning": True})
    assert goal_done("converter", "scan", before, {**before, "tables": 2})
    assert not goal_done("converter", "convert", before, {**before, "converting": True})
    assert goal_done("converter", "convert", before, {**before, "conversions": 1})
    assert goal_done("converter", "review", before, {**before, "reviewed": True})
    assert not goal_done("converter", "editor", before, before)
    assert goal_done("editor", "validate", {"reports": 3}, {"reports": 4})
    assert not goal_done("editor", "validate", {"reports": 3}, {"reports": 4, "validating": True})
    assert goal_done("editor", "fix", {}, {"errors": 0, "warnings": 0})
    assert not goal_done("editor", "edit", {}, {"file_open": True})
