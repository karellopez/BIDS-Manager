"""What the guide points at next, decided from what the window shows.

The guide (switched on in the header, beside Settings) highlights ONE control:
the next thing to do. This module decides which, from plain facts each view
reports about itself (``guide_facts``), and names the control by a target
name the view resolves (``guide_targets``). No Qt here, so every path can be
tested without a window.

The paths, in the order a new user walks them:

- Home: choose where the dataset is saved, name it, create it.
- Converter: choose the raw data, scan it, check the table, convert. Choosing
  another raw data folder after a scan starts the path again from Scan.
- Editor: validate, fix what BIDS still needs.

Where a view offers more than one sensible thing to do (a table already
scanned, a dataset already converted), the guide ASKS when the user comes to
it (:func:`choices`), and the answer, a GOAL, steers the steps
(:func:`next_step`) until it is reached (:func:`goal_done`); then the path
continues as before.

A step that is waiting for the app (a scan, a conversion, a validation)
points at the control that is busy and says what is happening, so the guide
never sends the user somewhere while the app is working. A finished path
says so and is marked ``done``; the overlay lets it fade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

HOME = "welcome"
CONVERTER = "converter"
EDITOR = "editor"


@dataclass(frozen=True)
class Step:
    """One thing to do, and where."""

    key: str            # identity: a new key replays the entrance
    target: str         # a name the view (or the header) resolves to a widget
    title: str
    text: str
    number: int = 0     # position on this path; 0 when off a path
    total: int = 0
    busy: bool = False  # the app is working; nothing to press but Stop
    done: bool = False  # the path is finished
    #: Targets the caption must not cover: what its text sends the user to.
    keep_clear: tuple[str, ...] = ()
    #: The goal the user chose, shown above the step ("Scan new data").
    kicker: str = ""

    @property
    def counter(self) -> str:
        return f"Step {self.number} of {self.total}" if self.number else ""

    @property
    def heading(self) -> str:
        """The caption's first line: the goal, the step's place, or Done."""
        parts = [p for p in (self.kicker, self.counter) if p]
        return " · ".join(parts) if parts else ("Done" if self.done else "")


@dataclass(frozen=True)
class Choice:
    """One answer to "What would you like to do?"."""

    key: str            # the goal it sets
    label: str
    detail: str
    recommended: bool = False


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------


def _create_path(f: Mapping[str, object], kicker: str = "") -> Step:
    total = 3
    if f.get("name"):
        return Step(
            "home.create", "create", "Create the dataset",
            "Press Create dataset. The Converter opens next, ready for your "
            "raw data.", 3, total, kicker=kicker)
    if f.get("location_chosen"):
        return Step(
            "home.name", "name", "Name the dataset",
            "Type a short name for your study, without spaces, for example "
            "My_Study. It becomes the dataset's folder name.", 2, total, kicker=kicker)
    return Step(
        "home.location", "browse", "Choose where to save it",
        "Pick the folder your new dataset will be saved in. Already have a "
        "dataset? Use Open dataset folder instead.", 1, total, kicker=kicker)


def _home(f: Mapping[str, object], goal: Optional[str]) -> Step:
    if goal == "create":
        return _create_path(f, "New dataset")
    if goal == "open":
        return Step(
            "home.open", "open", "Open a dataset",
            "Pick its folder. A BIDS dataset opens as a project; any other "
            "folder can be looked at, or made into one.", kicker="Another dataset")
    if f.get("project"):
        return Step(
            "home.continue", "converter", "Continue in the Converter",
            f"{f.get('project_name') or 'Your dataset'} is open. The Converter "
            "turns your raw data into it.")
    return _create_path(f)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


def _converter(f: Mapping[str, object], goal: Optional[str]) -> Step:
    total = 4
    if not f.get("project"):
        return Step(
            "convert.no_project", "home", "Start on Home",
            "Create a new dataset or open one you have. The Converter works "
            "inside it.")
    if f.get("scanning"):
        return Step(
            "convert.scanning", "scan", "Scanning",
            "Reading the raw data. The table fills when the scan is done. "
            "Stop ends it early.", 2, total, busy=True)
    if f.get("converting"):
        return Step(
            "convert.converting", "run", "Converting",
            "The log below shows the progress. Stop ends it after the subject "
            "in progress.", 4, total, busy=True)
    rows = bool(f.get("rows"))
    if goal == "scan":
        if f.get("raw_root") and f.get("raw_picked"):
            return Step(
                "convert.scan_new", "scan", "Scan the new folder",
                "Its recordings become a new scan table. The table you had stays "
                "as an earlier version in Scan table.", 2, 2, kicker="Scan new data")
        return Step(
            "convert.raw_new", "raw_folder", "Choose the new data",
            "Pick the folder with the recordings to add: DICOM, EEG, MEG or PET "
            "files. Its subfolders are included.", 1, 2, kicker="Scan new data")
    if goal == "convert" and rows:
        return Step(
            "convert.run_goal", "run", "Convert",
            "Run conversion writes the ticked recordings into the dataset, with "
            "their metadata. Subjects already in it are kept or updated as "
            "Settings > Convert says.", kicker="Convert the table")
    if goal == "editor":
        return Step(
            "convert.editor_goal", "editor", "Open the Editor",
            "The Editor shows the dataset as it is on disk, file by file, and "
            "validates it.", kicker="Look at the dataset")
    scan = Step(
        "convert.scan", "scan", "Scan the raw data",
        "Scan reads the folder and lists every recording it finds, with the BIDS "
        "name it proposes.", 2, total)
    if rows and f.get("raw_picked"):
        return scan                    # a folder chosen after the table: scan it next
    if f.get("converted") and goal is None:
        return Step(
            "convert.done", "editor", "Check the result",
            "The conversion is done. Open the Editor to look at the dataset "
            "and validate it.", done=True)
    if not f.get("raw_root"):
        return Step(
            "convert.raw", "raw_folder", "Choose your raw data",
            "Pick the folder with the recordings to convert: DICOM, EEG, MEG "
            "or PET files. Its subfolders are included.", 1, total)
    if not rows:
        return scan
    if not f.get("reviewed"):
        return Step(
            "convert.review", "table", "Check the table",
            "Click a row to see its details in Properties. Untick what you "
            "do not want, and correct a name that is wrong.", 3, total,
            keep_clear=("properties",),
            kicker="Check the table" if goal == "review" else "")
    return Step(
        "convert.run", "run", "Convert",
        "Run conversion writes the ticked recordings into the dataset, with "
        "their metadata.", 4, total)


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------


def _editor(f: Mapping[str, object], goal: Optional[str]) -> Optional[Step]:
    total = 2
    if f.get("view_only"):
        return None                    # a folder to look at, nothing to do
    if not f.get("root"):
        return Step(
            "editor.open", "open", "Open a dataset",
            "Open folder shows a dataset's files. Your own project opens here "
            "by itself.")
    if not f.get("subjects"):
        return Step(
            "editor.empty", "converter", "Nothing converted yet",
            "This dataset has no subjects yet. Convert your raw data in the "
            "Converter first.")
    if f.get("validating"):
        return Step(
            "editor.validating", "validate", "Validating",
            "Checking every file. The findings appear in the Validation "
            "panel.", 1, total, busy=True)
    errors = int(f.get("errors") or 0)
    warnings = int(f.get("warnings") or 0)
    if goal == "validate":
        return Step(
            "editor.validate_goal", "validate", "Validate the dataset",
            "Validation checks every file against the BIDS standard and lists "
            "what still needs fixing.", kicker="Validate")
    if goal == "fix" and (errors or warnings):
        what = _count(errors, "error") if errors else _count(warnings, "warning")
        return Step(
            "editor.fix_goal", "findings", "Go through the findings",
            f"{what} left. Click a finding to open the file it is about, fix "
            "it, then validate again.", kicker="Fix what validation found")
    if goal == "edit":
        if not f.get("file_open"):
            return Step(
                "editor.pick_file", "tree", "Choose a file",
                "Click a file in the BIDS tree. Its metadata, table or image "
                "opens in the middle.", 1, 2, kicker="Look at or edit a file")
        return Step(
            "editor.edit_file", "viewer", "Look, and change what you need",
            "A change is saved with Save and taken back with Undo. Validate the "
            "file to see what it still lacks.", 2, 2, kicker="Look at or edit a file")
    if goal == "tools":
        return Step(
            "editor.tools", "tools", "Open the Tools",
            "Rename entities, add or remove sessions, anonymise, check image "
            "quality, repair sidecars and more, each with a preview first.",
            kicker="Rename, restructure or repair")
    if not f.get("validated"):
        return Step(
            "editor.validate", "validate", "Validate the dataset",
            "Validation checks every file against the BIDS standard and lists "
            "what still needs fixing.", 1, total)
    if errors:
        return Step(
            "editor.fix", "findings", "Fix what BIDS needs",
            f"{_count(errors, 'error')} to fix. Click a finding to open the "
            "file it is about, fix it, then validate again.", 2, total)
    if warnings:
        return Step(
            "editor.valid", "findings", "The dataset is valid",
            f"Nothing BIDS requires is missing. {_count(warnings, 'warning')} "
            "remain, worth a look but not required.", done=True)
    return Step(
        "editor.valid", "validate", "The dataset is valid",
        "Nothing BIDS requires is missing, and there are no warnings.",
        done=True)


# ---------------------------------------------------------------------------
# What the user may want, and when they have it
# ---------------------------------------------------------------------------


def choices(view: str, f: Mapping[str, object]) -> list[Choice]:
    """What the user may want to do in ``view`` now; fewer than two means
    there is one path and nothing to ask. The path's own next step is the
    ``recommended`` one."""
    if view == HOME:
        if not f.get("project"):
            return []
        name = f.get("project_name") or "this dataset"
        return [
            Choice("continue", f"Continue with {name}",
                   "Scan, convert and check its data in the Converter.", True),
            Choice("create", "Create a new dataset",
                   "Start another project, in a folder of its own."),
            Choice("open", "Open another dataset",
                   "A dataset you already have, BIDS or not."),
        ]
    if view == CONVERTER:
        if not f.get("project") or not f.get("rows") or f.get("scanning") \
                or f.get("converting"):
            return []
        if not f.get("reviewed"):
            nxt = "review"
        elif f.get("converted"):
            nxt = "editor"
        else:
            nxt = "convert"
        out = [
            Choice("scan", "Scan new data",
                   "Choose another raw data folder and scan it. This table stays "
                   "as an earlier version."),
            Choice("review", "Check and edit the table",
                   "Names, sessions and what is included, row by row."),
            Choice("convert", "Convert the table",
                   "Write the ticked recordings into the dataset."),
        ]
        if f.get("subjects") or f.get("converted"):
            out.append(Choice("editor", "Look at the dataset",
                              "Open the Editor to check and validate what was converted."))
        return [Choice(c.key, c.label, c.detail, c.key == nxt) for c in out]
    if view == EDITOR:
        if f.get("view_only") or not f.get("root") or not f.get("subjects") \
                or f.get("validating"):
            return []
        errors = int(f.get("errors") or 0)
        warnings = int(f.get("warnings") or 0)
        validated = bool(f.get("validated"))
        nxt = "validate" if not validated else ("fix" if errors else "")
        out = [Choice("validate", "Validate the dataset" if not validated else
                      "Validate again", "Check every file against the BIDS standard.")]
        if validated and (errors or warnings):
            out.append(Choice(
                "fix", "Fix what validation found",
                ", ".join(p for p in (_count(errors, "error") if errors else "",
                                      _count(warnings, "warning") if warnings else "") if p)
                + ", each a click away from its file."))
        out += [
            Choice("edit", "Look at or edit a file",
                   "Open a file from the BIDS tree: its metadata, table or image."),
            Choice("tools", "Rename, restructure or repair",
                   "Entities, sessions, anonymising, quality checks and more."),
        ]
        return [Choice(c.key, c.label, c.detail, c.key == nxt) for c in out]
    return []


def goal_done(view: str, goal: str, before: Mapping[str, object],
              now: Mapping[str, object]) -> bool:
    """Whether the goal chosen when the facts were ``before`` is reached now;
    the path then continues as it would have."""
    if view == CONVERTER:
        if goal == "scan":
            return now.get("tables") != before.get("tables") and not now.get("scanning")
        if goal == "convert":
            return (now.get("conversions") != before.get("conversions")
                    and not now.get("converting"))
        if goal == "review":
            return bool(now.get("reviewed"))
    if view == EDITOR:
        if goal == "validate":
            return now.get("reports") != before.get("reports") and not now.get("validating")
        if goal == "fix":
            return not (now.get("errors") or now.get("warnings"))
    return False


def next_step(view: str, facts: Mapping[str, object],
              goal: Optional[str] = None) -> Optional[Step]:
    """The step to show in ``view`` (``welcome``, ``converter`` or
    ``editor``) toward ``goal`` (a :class:`Choice` key) or along the view's
    path, or ``None`` when there is nothing to point at."""
    if view == HOME:
        return _home(facts, goal)
    if view == CONVERTER:
        return _converter(facts, goal)
    if view == EDITOR:
        return _editor(facts, goal)
    return None


__all__ = ["CONVERTER", "Choice", "EDITOR", "HOME", "Step", "choices", "goal_done",
           "next_step"]
