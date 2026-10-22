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
