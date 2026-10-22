"""What the guide points at next, decided from what the window shows.

The guide (switched on in the header, beside Settings) highlights ONE control:
the next thing to do. This module decides which, from plain facts each view
reports about itself (``guide_facts``), and names the control by a target
name the view resolves (``guide_targets``). No Qt here, so every path can be
tested without a window.

The paths, in the order a new user walks them:

- Home: choose where the dataset is saved, name it, create it.
- Converter: choose the raw data, scan it, check the table, convert.
- Editor: validate, fix what BIDS still needs.

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
    number: int = 0     # position on this view's path; 0 when off the path
    total: int = 0
    busy: bool = False  # the app is working; nothing to press but Stop
    done: bool = False  # the path is finished
    #: Targets the caption must not cover: what its text sends the user to.
    keep_clear: tuple[str, ...] = ()

    @property
    def counter(self) -> str:
        return f"Step {self.number} of {self.total}" if self.number else ""


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _home(f: Mapping[str, object]) -> Step:
    if f.get("project"):
        return Step(
            "home.continue", "converter", "Continue in the Converter",
            f"{f.get('project_name') or 'Your dataset'} is open. The Converter "
            "turns your raw data into it.")
    total = 3
    if f.get("name"):
        return Step(
            "home.create", "create", "Create the dataset",
            "Press Create dataset. The Converter opens next, ready for your "
            "raw data.", 3, total)
    if f.get("location_chosen"):
        return Step(
            "home.name", "name", "Name the dataset",
            "Type a short name for your study, without spaces, for example "
            "My_Study. It becomes the dataset's folder name.", 2, total)
    return Step(
        "home.location", "browse", "Choose where to save it",
        "Pick the folder your new dataset will be saved in. Already have a "
        "dataset? Use Open dataset folder instead.", 1, total)


def _converter(f: Mapping[str, object]) -> Step:
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
    if f.get("converted"):
        return Step(
            "convert.done", "editor", "Check the result",
            "The conversion is done. Open the Editor to look at the dataset "
            "and validate it.", done=True)
    if not f.get("raw_root"):
        return Step(
            "convert.raw", "raw_folder", "Choose your raw data",
            "Pick the folder with the recordings to convert: DICOM, EEG, MEG "
            "or PET files. Its subfolders are included.", 1, total)
    if not f.get("rows"):
        return Step(
            "convert.scan", "scan", "Scan the raw data",
            "Scan reads the folder and lists every recording it finds, with "
            "the BIDS name it proposes.", 2, total)
    if not f.get("reviewed"):
        return Step(
            "convert.review", "table", "Check the table",
            "Click a row to see its details in Properties. Untick what you "
            "do not want, and correct a name that is wrong.", 3, total,
            keep_clear=("properties",))
    return Step(
        "convert.run", "run", "Convert",
        "Run conversion writes the ticked recordings into the dataset, with "
        "their metadata.", 4, total)


def _editor(f: Mapping[str, object]) -> Optional[Step]:
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
    if not f.get("validated"):
        return Step(
            "editor.validate", "validate", "Validate the dataset",
            "Validation checks every file against the BIDS standard and lists "
            "what still needs fixing.", 1, total)
    errors = int(f.get("errors") or 0)
    warnings = int(f.get("warnings") or 0)
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


def next_step(view: str, facts: Mapping[str, object]) -> Optional[Step]:
    """The step to show in ``view`` (``welcome``, ``converter`` or
    ``editor``), or ``None`` when there is nothing to point at."""
    if view == HOME:
        return _home(facts)
    if view == CONVERTER:
        return _converter(facts)
    if view == EDITOR:
        return _editor(facts)
    return None


__all__ = ["CONVERTER", "EDITOR", "HOME", "Step", "next_step"]
