"""The companion files a recording should have, and generating the missing ones.

BIDS associates files with each other by name: a task recording is expected to
have an ``events.tsv`` beside it, an electrophysiology recording a
``channels.tsv``, a data file a ``.json`` sidecar. A validator reports the ones
that are absent, and the user is then left to create them by hand in a text
editor, which is exactly the sort of work an editor should do.

One rule governs everything here, and it is the reason this module is careful
rather than eager:

**A generated file must not look authored.** An ``events.tsv`` with a header
and no rows satisfies a structural check while telling a reader nothing, and a
reader cannot tell it apart from a genuine recording with no events. So a stub
is written with the ``TODO`` convention BIDS Manager already uses for missing
recommended metadata, which the validator reports until a human replaces it.
The user is never quietly told a problem is solved when it has only been
hidden.

What can be derived is derived rather than stubbed: an electrophysiology
recording knows its own channel names, so ``channels.tsv`` is written from the
file rather than left blank.

Qt-free.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# The placeholder the metadata engine already writes into recommended fields
# it cannot answer, so validation keeps reporting a stub nobody filled in.
TODO = "TODO"

# Datatypes whose recordings BIDS associates an ``events.tsv`` with when the
# filename carries a task.
_TASK_DATATYPES = frozenset({"func", "eeg", "meg", "ieeg", "nirs", "beh", "pet"})

# Datatypes that carry a channels table.
_CHANNEL_DATATYPES = frozenset({"eeg", "meg", "ieeg", "nirs"})

# Recording extensions we know how to open to derive channel names.
_RECORDING_EXTS = (
    ".fif", ".fif.gz", ".edf", ".bdf", ".gdf", ".set", ".vhdr",
    ".cnt", ".con", ".sqd", ".kdf", ".mef", ".nwb", ".snirf",
)


@dataclass
class MissingAssociation:
    """A companion file the standard associates with a recording, absent."""

    target: Path            # what would be created
    rel: str                # relative to the dataset root, for display
    kind: str               # "events" | "channels" | "sidecar"
    source: Path            # the recording it belongs to
    derivable: bool         # can it be filled from the data, or only stubbed?
    note: str = ""

    @property
    def label(self) -> str:
        what = {
            "events": "events table",
            "channels": "channels table",
            "sidecar": "JSON sidecar",
        }.get(self.kind, self.kind)
        how = "from the recording" if self.derivable else "as a stub to fill in"
        return f"{what}, {how}"


def _entities(name: str) -> dict[str, str]:
    stem = _stem(name)
    out: dict[str, str] = {}
    for part in stem.split("_"):
        if "-" in part:
            key, _, val = part.partition("-")
            out[key] = val
    return out


def _stem(name: str) -> str:
    for ext in (".nii.gz", ".tsv.gz", ".fif.gz") + tuple(_RECORDING_EXTS) + (
        ".nii", ".tsv", ".json",
    ):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def _datatype_of(path: Path, root: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return rel.parts[-2] if len(rel.parts) >= 2 else None


def _is_recording(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return True
    return name.endswith(_RECORDING_EXTS)


def find_missing(root: Path) -> list[MissingAssociation]:
    """Every companion file the dataset's recordings should have and do not.

    Walks the recordings rather than the findings, so it answers the question
    even before the dataset has been validated.
    """
    root = Path(root)
    out: list[MissingAssociation] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".bidsmgr" in path.parts:
            continue
        if not _is_recording(path):
            continue
        datatype = _datatype_of(path, root)
        if datatype is None:
            continue
        stem = _stem(path.name)
        ent = _entities(path.name)

        # A JSON sidecar, for anything.
        sidecar = path.parent / f"{stem}.json"
        if not sidecar.exists():
            out.append(_missing(root, sidecar, "sidecar", path, False))

        # An events table, when the recording names a task.
        if datatype in _TASK_DATATYPES and "task" in ent:
            events = path.parent / f"{_strip_suffix(stem)}_events.tsv"
            if not events.exists():
                out.append(_missing(root, events, "events", path, False))

        # A channels table, derivable from the recording itself.
        if datatype in _CHANNEL_DATATYPES:
            channels = path.parent / f"{_strip_suffix(stem)}_channels.tsv"
            if not channels.exists():
                out.append(_missing(root, channels, "channels", path, True))
    return out


def _strip_suffix(stem: str) -> str:
    """``sub-01_task-rest_bold`` -> ``sub-01_task-rest``.

    A companion carries the entities but its own suffix, so the recording's
    suffix comes off before the companion's goes on.
    """
    parts = stem.split("_")
    if parts and "-" not in parts[-1]:
        return "_".join(parts[:-1])
    return stem


def _missing(
    root: Path, target: Path, kind: str, source: Path, derivable: bool,
) -> MissingAssociation:
    try:
        rel = str(target.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = str(target)
    return MissingAssociation(
        target=target, rel=rel, kind=kind, source=source, derivable=derivable,
    )


# --------------------------------------------------------------------------
# Generating


def _events_stub() -> str:
    """A header and one TODO row.

    Not an empty table: an empty ``events.tsv`` is indistinguishable from a
    recording that genuinely had no events, and would pass quietly forever.
    The TODO row is reported by validation until somebody deals with it.
    """
    return "onset\tduration\ttrial_type\n" + f"{TODO}\t{TODO}\t{TODO}\n"


def _channels_from_recording(source: Path) -> Optional[str]:
    """Read the recording's channel names and types, or ``None``.

    Real content, because the recording knows the answer. Falls back to
    ``None`` when the file cannot be opened, and the caller stubs instead.
    """
    try:
        import mne
    except ImportError:
        return None
    try:
        raw = mne.io.read_raw(str(source), preload=False, verbose="ERROR")
    except Exception as exc:  # noqa: BLE001 - many reader-specific failures
        log.debug("cannot read %s for channels: %s", source, exc)
        return None
    rows = ["name\ttype\tunits"]
    types = raw.get_channel_types()
    for name, kind in zip(raw.ch_names, types):
        units = "uV" if kind in ("eeg", "eog", "ecg", "emg") else "n/a"
        rows.append(f"{name}\t{kind.upper()}\t{units}")
    return "\n".join(rows) + "\n"


def _sidecar_stub(source: Path, root: Path) -> str:
    """A sidecar with the fields the standard requires, each a TODO."""
    datatype = _datatype_of(source, root)
    suffix = None
    parts = _stem(source.name).split("_")
    if parts and "-" not in parts[-1]:
        suffix = parts[-1]
    data: dict[str, object] = {}
    if datatype and suffix:
        try:
            from ..metadata.template_plan import sidecar_section

            # ``include_derived`` matters here: the usual form hides the
            # fields a conversion fills in, but this file has no sidecar at
            # all and no conversion is going to run, so the stub has to name
            # every field the standard requires. Without it a ``func/bold``
            # stub omitted RepetitionTime and TaskName.
            section = sidecar_section(datatype, suffix, include_derived=True)
            for f in section.fields:
                if str(getattr(f, "level", "")) == "required":
                    data[f.name] = TODO
        except Exception as exc:  # noqa: BLE001 - schema lookup is optional
            log.debug("no schema fields for %s/%s: %s", datatype, suffix, exc)
    if not data:
        data = {"Description": TODO}
    return json.dumps(data, indent=4, ensure_ascii=False) + "\n"


def generate(
    root: Path, items: list[MissingAssociation],
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Create the selected companion files, reversibly.

    Returns ``(created, failed)``. One operation for the batch, so undo puts
    the dataset back in a single step.
    """
    from ..project.operations import begin_operation

    created: list[Path] = []
    failed: list[tuple[Path, str]] = []
    if not items:
        return created, failed
    label = f"Generate {len(items)} companion file(s)"
    with begin_operation(Path(root), label) as op:
        for item in items:
            try:
                if item.kind == "channels":
                    text = _channels_from_recording(item.source)
                    if text is None:
                        text = f"name\ttype\tunits\n{TODO}\t{TODO}\t{TODO}\n"
                elif item.kind == "events":
                    text = _events_stub()
                else:
                    text = _sidecar_stub(item.source, Path(root))
                op.write_text(item.target, text)
            except OSError as exc:
                failed.append((item.target, str(exc)))
                continue
            created.append(item.target)
    return created, failed


__all__ = ["MissingAssociation", "TODO", "find_missing", "generate"]
