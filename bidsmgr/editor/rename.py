"""Rename a BIDS entity everywhere it appears, including a subject.

A dataset arrives with ``sub-Pilot01`` and it should be ``sub-01``; a task was
labelled ``rest`` in one session and ``resting`` in another; two runs were
numbered from zero. Doing that by hand means a folder, every filename under it,
and three kinds of cross-reference that are easy to forget, which is why
datasets end up with a renamed subject and a ``participants.tsv`` still naming
the old one.

**Entities are typed values, never substrings.** ``run-1`` becomes ``run-2``
without touching ``run-10``, and a subject called ``sub-01`` does not match the
``01`` inside ``acq-01``. Every match is made against a parsed
``key-value`` pair, so there is no regex over the whole path to get wrong.

**Three cross-references travel with the rename**, because BIDS points at
filenames from inside files:

* ``IntendedFor`` in fieldmap sidecars, which names the images a fieldmap
  corrects, as a path relative to the subject or as a BIDS URI;
* the ``filename`` column of every ``*_scans.tsv``;
* the ``participant_id`` column of ``participants.tsv``, for a subject rename.

**Renaming onto a name that already exists is a FUSION**, not an error, when
the caller asks for one. It is the common case nobody has a good tool for: the
same person was scanned twice and came back as ``sub-07`` and ``sub-12``, or an
EEG session and an MRI session were converted separately and are one
participant. Fusing moves everything under the source subject into the target,
merges the two ``*_scans.tsv`` tables, and folds the two ``participants.tsv``
rows into one, keeping every value either row states. Two files that would
claim the same name are still a conflict: that is genuinely two recordings, and
picking one is not a decision a tool should make silently.

**Nothing is written until the whole plan is known.** :func:`plan_rename`
returns what would move, what would be edited, what would merge, and what would
collide; :func:`apply_rename` performs it through the operation log, so a
rename that touched four hundred files is one undo.

``derivatives/``, ``sourcedata/`` and ``code/`` are out of scope, matching
BIDSvue's decision: their contents are produced by other tools and renaming
inside them silently breaks provenance.

Qt-free.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Directories the tool owns rather than the standard: our own state and the
# version control around it. Everything else that is out of scope comes from
# the schema (see :func:`_skip_top_level`).
_TOOL_DIRS = (".bidsmgr", ".git")


def folder_entities() -> tuple[str, ...]:
    """Entities that name a DIRECTORY as well as appearing in filenames.

    Read from the schema, not assumed: renaming one of these has to move a
    folder and renaming any other must not, and which is which is a fact
    about the standard rather than about this tool.
    """
    from ..schema import directory_entity_keys

    return directory_entity_keys()


def _skip_top_level() -> tuple[str, ...]:
    """Top-level directories a rename must not reach into.

    The schema marks ``derivatives`` / ``sourcedata`` / ``code`` and friends
    ``opaque``, which is exactly the property that matters here: their
    contents were produced by something else and renaming inside them
    silently breaks provenance.
    """
    from ..schema import opaque_directories

    return tuple(opaque_directories()) + _TOOL_DIRS


def _label_pattern(entity: str) -> re.Pattern:
    """What the standard allows this entity's value to be.

    Per-entity, because they differ and a single rule is wrong twice over: a
    hardcoded ``[A-Za-z0-9]+`` refuses the ``+`` the schema allows in a label
    (``acq-6p+s2`` is valid BIDS) and accepts ``run-abc``, which is not,
    because ``run`` is an index.
    """
    from ..schema import entity_key_info

    try:
        pattern = entity_key_info(entity).format.pattern
    except KeyError:
        pattern = r"[0-9a-zA-Z+]+"
    return re.compile(f"^(?:{pattern})$")


class RenameError(ValueError):
    """The rename cannot be performed as asked."""


@dataclass
class ContentEdit:
    """A file whose CONTENTS mention the old value."""

    path: Path
    rel: str
    what: str        # "IntendedFor" | "scans.tsv filename" | "participant_id"
    hits: int


@dataclass
class RenamePlan:
    """Everything a rename would do, before any of it is done."""

    entity: str
    old: str
    new: str
    dir_moves: list[tuple[Path, Path]] = dc_field(default_factory=list)
    file_moves: list[tuple[Path, Path]] = dc_field(default_factory=list)
    content_edits: list[ContentEdit] = dc_field(default_factory=list)
    conflicts: list[str] = dc_field(default_factory=list)
    # Fusion: the target name already exists and the caller asked to merge
    # into it rather than be refused.
    fusion: bool = False
    fused_dirs: list[tuple[Path, Path]] = dc_field(default_factory=list)
    table_merges: list[tuple[Path, Path]] = dc_field(default_factory=list)
    row_folds: list[tuple[Path, str]] = dc_field(default_factory=list)
    # Folders left behind once everything under them has moved out.
    emptied: list[Path] = dc_field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (
            self.dir_moves or self.file_moves or self.content_edits
            or self.table_merges or self.row_folds
        )

    @property
    def n_files(self) -> int:
        return len(self.file_moves)

    def summary(self) -> str:
        parts = []
        if self.dir_moves:
            parts.append(f"{len(self.dir_moves)} folder(s)")
        if self.file_moves:
            parts.append(f"{len(self.file_moves)} file(s)")
        if self.table_merges:
            parts.append(f"{len(self.table_merges)} table(s) merged")
        if self.row_folds:
            parts.append(f"{len(self.row_folds)} table row(s) folded")
        if self.content_edits:
            n = sum(e.hits for e in self.content_edits)
            parts.append(
                f"{n} reference(s) in {len(self.content_edits)} file(s)"
            )
        return ", ".join(parts) if parts else "nothing to do"

    def verb(self) -> str:
        """What this actually is, for a dialog that must not mislead."""
        if not self.fusion:
            return f"Rename {self.entity}-{self.old} to {self.entity}-{self.new}"
        return (
            f"Merge {self.entity}-{self.old} into the existing "
            f"{self.entity}-{self.new}"
        )

    # -- choosing a subset --------------------------------------------------
    #
    # A rename does not have to be all-or-nothing. "These three runs were
    # mislabelled, the rest are right" is an ordinary thing to want, and a
    # tool that can only do the whole thing forces the user to do the rest by
    # hand, which is where the errors come from.

    def file_key(self, root: Path, src: Path) -> str:
        """The stable identifier for one file move: its path within the root.

        A path rather than an index, so a caller that re-plans between showing
        the list and applying it cannot silently rename a different file.
        """
        return _rel(root, src)

    def file_keys(self, root: Path) -> list[str]:
        return [self.file_key(root, src) for src, _dst in self.file_moves]

    def selected_moves(
        self, root: Path, only: Optional[set[str]],
    ) -> list[tuple[Path, Path]]:
        """The moves ``only`` names, or all of them when it is ``None``."""
        if only is None:
            return list(self.file_moves)
        return [
            (src, dst) for src, dst in self.file_moves
            if self.file_key(root, src) in only
        ]


# --------------------------------------------------------------------------
# Parsing


def _stem_and_ext(name: str) -> tuple[str, str]:
    for ext in (".nii.gz", ".tsv.gz", ".fif.gz"):
        if name.endswith(ext):
            return name[: -len(ext)], ext
    p = Path(name)
    return p.stem, p.suffix


def entity_value(name: str, entity: str) -> Optional[str]:
    """The value of ``entity`` in a BIDS name, or ``None``.

    Matches the whole ``key-value`` part, which is what makes ``run-1``
    distinct from ``run-10`` without any regex escaping.
    """
    stem, _ = _stem_and_ext(name)
    for part in stem.split("_"):
        key, sep, value = part.partition("-")
        if sep and key == entity:
            return value
    return None


def rename_in_name(name: str, entity: str, old: str, new: str) -> str:
    """Return ``name`` with ``entity-old`` replaced by ``entity-new``."""
    stem, ext = _stem_and_ext(name)
    parts = stem.split("_")
    out = []
    for part in parts:
        key, sep, value = part.partition("-")
        if sep and key == entity and value == old:
            out.append(f"{entity}-{new}")
        else:
            out.append(part)
    return "_".join(out) + ext


def _skip(path: Path, root: Path) -> bool:
    """Is this path outside what a rename may touch?

    No ``resolve()``. It was called on BOTH sides for every path in the
    dataset, and ``realpath`` is a syscall per component: on a 1,262-file tree
    that alone was 216 ms of a 294 ms plan, and the dialog re-planned on every
    keystroke. Paths handed to this come from walking ``root``, so they are
    already under it and a plain ``relative_to`` is both correct and free.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        try:
            rel = path.resolve().relative_to(root.resolve())
        except (ValueError, OSError):
            return True
    return bool(rel.parts) and rel.parts[0] in _skip_top_level()


def walk_dataset(root: Path) -> list[Path]:
    """Every file a rename may touch, in one pass.

    Prunes the out-of-scope directories AS IT DESCENDS rather than walking
    them and filtering afterwards, so a dataset with a large ``derivatives/``
    costs nothing to skip. This used to be three separate ``rglob`` calls over
    the whole tree plus a per-path filter.
    """
    import os

    skip = set(_skip_top_level())
    out: list[Path] = []
    root_str = str(root)
    for dirpath, dirnames, filenames in os.walk(root_str):
        if dirpath == root_str:
            dirnames[:] = [d for d in dirnames if d not in skip]
        else:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        base = Path(dirpath)
        out.extend(base / name for name in filenames)
    return out


# --------------------------------------------------------------------------
# Planning


def plan_rename(
    root: Path, entity: str, old: str, new: str, *, fuse: bool = False,
) -> RenamePlan:
    """Work out everything a rename would touch.

    ``fuse`` decides what happens when the new name is already taken. Left off,
    a taken name is a conflict and the rename is refused. Turned on, the source
    is merged INTO the existing one: files move in, ``*_scans.tsv`` tables are
    concatenated, and the two ``participants.tsv`` rows become one. Files that
    would still collide after that are reported, because two recordings
    claiming one name is a decision for a person, not for a tool.

    Raises :class:`RenameError` for a request that cannot be valid: an empty
    or non-alphanumeric label (BIDS allows only those), or a no-op.
    """
    root = Path(root)
    entity = entity.strip()
    old, new = old.strip(), new.strip()
    if not entity or not old or not new:
        raise RenameError("entity, old value and new value are all required")
    if old == new:
        raise RenameError(f"{entity}-{old} is already its own name")
    if not _label_pattern(entity).match(new):
        raise RenameError(_bad_label_message(entity, new))

    plan = RenamePlan(entity=entity, old=old, new=new)
    old_token, new_token = f"{entity}-{old}", f"{entity}-{new}"

    # Folders first, so the file moves below can be expressed against their
    # final parents.
    if entity in folder_entities():
        for d in sorted(root.rglob(f"{old_token}")):
            if not d.is_dir() or _skip(d, root):
                continue
            target = d.with_name(new_token)
            if not target.exists():
                plan.dir_moves.append((d, target))
                continue
            if not fuse:
                plan.conflicts.append(
                    f"{_rel(root, target)} already exists, so "
                    f"{_rel(root, d)} cannot take its name. Merge them "
                    f"instead to keep both."
                )
                continue
            # The target is there and we were asked to merge into it. The
            # folder does not move; its contents do, one file at a time, so a
            # session the target does not have lands beside the ones it does.
            plan.fusion = True
            plan.fused_dirs.append((d, target))
            plan.emptied.append(d)

    # Fused directories behave exactly like moved ones when working out where
    # a file ends up. They differ only in that no folder is renamed.
    moved_dirs = {src: dst for src, dst in plan.dir_moves}
    moved_dirs.update({src: dst for src, dst in plan.fused_dirs})

    files = sorted(walk_dataset(root))
    for f in files:
        if entity_value(f.name, entity) != old:
            continue
        parent = _remap_parent(f.parent, moved_dirs)
        target = parent / rename_in_name(f.name, entity, old, new)
        if target.exists() and target != f:
            # Two tables of the same kind are merged rather than refused: a
            # scans table is a list of what a subject has, and fusing two
            # subjects means the list is the two lists.
            if plan.fusion and f.name.endswith("_scans.tsv"):
                plan.table_merges.append((f, target))
                continue
            plan.conflicts.append(
                f"{_rel(root, target)} already exists"
                + (
                    " - both subjects have this recording, so one of them "
                    "needs a different name before they can be merged"
                    if plan.fusion else ""
                )
            )
            continue
        plan.file_moves.append((f, target))

    plan.content_edits = _plan_content_edits(root, entity, old, new, files)

    if plan.fusion and entity == "sub":
        # participants.tsv must not simply have its label rewritten: that
        # leaves two rows claiming one participant_id. The two rows fold into
        # one instead, so the fold replaces the plain column rewrite.
        participants = root / "participants.tsv"
        if participants.exists():
            plan.content_edits = [
                e for e in plan.content_edits if e.path != participants
            ]
            if _has_row(participants, "participant_id", old_token):
                plan.row_folds.append((participants, "participant_id"))
    return plan


def _bad_label_message(entity: str, value: str) -> str:
    """Say what the standard asks for, in the standard's own words."""
    from ..schema import entity_key_info

    try:
        info = entity_key_info(entity)
    except KeyError:
        return f"{value!r} is not a valid value for {entity}-"
    kind = info.format.name
    if kind == "index":
        expected = "digits only, for example 01"
    elif kind == "label":
        expected = "letters, digits and + only, no spaces or underscores"
    else:
        expected = f"it must match {info.format.pattern}"
    return (
        f"{value!r} is not a valid {info.display_name.lower()} "
        f"{kind}: {expected}"
    )


def would_fuse(root: Path, entity: str, new: str) -> bool:
    """Is the target name already taken by a folder?

    Lets a caller ask the question before deciding which dialog to show, so a
    user is told "this will merge two subjects" rather than being handed a
    conflict list to interpret.
    """
    if entity not in folder_entities():
        return False
    import os

    root = Path(root)
    token = f"{entity}-{new.strip()}"
    skip = set(_skip_top_level())
    # A folder entity only ever names a directory at a fixed depth (subjects
    # at the top, sessions under one), so this looks there instead of walking
    # the whole tree, which it used to do on every keystroke.
    if (root / token).is_dir():
        return True
    try:
        for entry in os.scandir(root):
            if not entry.is_dir() or entry.name in skip:
                continue
            if (Path(entry.path) / token).is_dir():
                return True
    except OSError:
        return False
    return False


def _remap_parent(parent: Path, moved: dict[Path, Path]) -> Path:
    """Where a file's directory ends up once the folder moves are applied."""
    for src, dst in moved.items():
        if parent == src:
            return dst
        if src in parent.parents:
            return dst / parent.relative_to(src)
    return parent


def _plan_content_edits(
    root: Path, entity: str, old: str, new: str,
    files: Optional[list[Path]] = None,
) -> list[ContentEdit]:
    """The three places BIDS points at a filename from inside a file.

    Takes the file list the caller already walked. Re-globbing the tree twice
    more here was most of what made planning slow enough to freeze the dialog.
    """
    out: list[ContentEdit] = []
    old_token = f"{entity}-{old}"
    files = walk_dataset(root) if files is None else files

    for p in files:
        if p.suffix != ".json":
            continue
        data = _load_json(p)
        if not data or "IntendedFor" not in data:
            continue
        hits = _count_intended_for(data.get("IntendedFor"), old_token)
        if hits:
            out.append(ContentEdit(p, _rel(root, p), "IntendedFor", hits))

    for p in files:
        if not p.name.endswith("_scans.tsv"):
            continue
        hits = _count_column(p, "filename", old_token)
        if hits:
            out.append(ContentEdit(p, _rel(root, p), "scans.tsv filename", hits))

    if entity == "sub":
        participants = root / "participants.tsv"
        if participants.exists():
            hits = _count_column(participants, "participant_id", old_token)
            if hits:
                out.append(ContentEdit(
                    participants, "participants.tsv", "participant_id", hits,
                ))
    return out


def _count_intended_for(value, token: str) -> int:
    if value is None:
        return 0
    items = value if isinstance(value, list) else [value]
    return sum(1 for v in items if _token_in_path(str(v), token))


def _token_in_path(text: str, token: str) -> int:
    """Is ``token`` a whole path segment or entity in ``text``?

    Segment-aware so ``sub-01`` does not match ``sub-011``.
    """
    for chunk in re.split(r"[/_]", text):
        if chunk == token:
            return True
    return False


def _has_row(path: Path, column: str, token: str) -> bool:
    """Does the table carry a row whose ``column`` is exactly ``token``?"""
    from .tsv_edit import read_table

    table = read_table(path)
    if table is None or column not in table.header:
        return False
    return any(v == token for v in table.column(column))


def _count_column(path: Path, column: str, token: str) -> int:
    from .tsv_edit import read_table

    table = read_table(path)
    if table is None or column not in table.header:
        return 0
    return sum(1 for v in table.column(column) if _token_in_path(v, token))


def _load_json(path: Path) -> Optional[dict]:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _rel(root: Path, path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# Applying


def apply_rename(
    root: Path, plan: RenamePlan, *, only: Optional[set[str]] = None,
) -> tuple[int, list[str]]:
    """Perform ``plan``. Returns ``(files_touched, errors)``.

    ``only`` is a set of the relative source paths to rename, as
    :meth:`RenamePlan.file_key` reports them. ``None``, the default, renames
    everything the plan found, which is what every caller did before selection
    existed.

    A PARTIAL rename is a different operation from a whole one, and the
    difference is handled rather than papered over:

    * a folder is renamed as a unit only when every file inside it is
      selected. Otherwise the selected files are moved one at a time into the
      target folder and the rest stay where they are, which is what "rename
      these three sessions and leave the others" has to mean;
    * a cross-reference is rewritten only where it names a file that actually
      moved. Rewriting them all would leave ``IntendedFor`` and
      ``*_scans.tsv`` pointing at names nothing has;
    * ``participants.tsv`` keeps its row unless the WHOLE subject moved. Half
      a subject under a new label is still the old participant.

    Refuses outright when the plan has conflicts: a rename that would land on
    an existing name is the one case where doing most of it is worse than
    doing none.
    """
    from ..project.operations import begin_operation

    if plan.conflicts:
        raise RenameError(
            "this rename would overwrite existing files:\n  "
            + "\n  ".join(plan.conflicts)
        )
    root = Path(root)
    errors: list[str] = []
    touched = 0
    old_token, new_token = f"{plan.entity}-{plan.old}", \
        f"{plan.entity}-{plan.new}"

    moves = plan.selected_moves(root, only)
    if not moves and not plan.dir_moves and only is not None:
        return 0, []
    whole = only is None or len(moves) == len(plan.file_moves)
    chosen = {src for src, _dst in moves}
    # Which basenames the references may follow. ``None`` means all of them.
    moved_names: Optional[set[str]] = None if whole else {
        src.name for src in chosen
    }
    # When a SUBJECT rename moves only some of a subject's files, the result is
    # two subjects where there was one. That is a split, and it needs more than
    # the moves: the scans table has to follow the rows that left, the new
    # subject needs a participants row, and the folders the files vacated have
    # to go. Worked out before anything moves, from the moves themselves.
    splits = (
        _plan_splits(root, moves) if plan.entity == "sub" and not whole else []
    )
    # The folder renames that will ACTUALLY happen. A folder is only renamed
    # as a unit when nothing inside it is staying, so a partial selection
    # leaves it where it is and moves the chosen files one at a time. Deciding
    # this once, here, is load-bearing: the scans planning below asked the
    # PLAN which folders move, the plan said "sub-001 becomes sub-002", the
    # apply loop then declined to move it, and every row that should have
    # been relocated was left pointing at a file in another subject.
    renaming_dirs: dict[Path, Path] = {
        src: dst for src, dst in plan.dir_moves
        if _folder_fully_selected(src, chosen, whole)
    }
    # Where each directory's CONTENTS end up, which is not the same list. A
    # fused directory is not renamed (its target already exists) but its
    # contents do land somewhere new, and anything working out a destination
    # has to know that. Treating the two as one list tried to rename a folder
    # onto a directory that was already there.
    path_mapping: dict[Path, Path] = dict(renaming_dirs)
    path_mapping.update({src: dst for src, dst in plan.fused_dirs})

    # Which scans rows have to change table, worked out from the pre-move tree
    # so the answer does not depend on what has been applied yet. A row only
    # moves when its file leaves the scope of the table describing it; when the
    # table travels with the file, the ordinary column rewrite covers it.
    # ``renaming_dirs``, not ``path_mapping``: a table only travels when its
    # own directory is renamed or when the table file is itself in the move
    # list. A FUSED directory is never renamed, and its unselected contents,
    # the scans table among them, stay exactly where they are.
    row_moves = plan_scans_rows(root, moves, renaming_dirs)
    relocating = {move.from_table for move in row_moves}
    # Tables that are themselves being moved still need their column rewritten:
    # their rows name the old entity and are going with them.
    travelling = {src for src, _dst in moves if src.name.endswith("_scans.tsv")}

    label = (
        f"Rename {old_token} to {new_token}" if whole else
        f"Rename {len(moves)} file(s) from {old_token} to {new_token}"
    )
    with begin_operation(root, label) as op:
        # Contents first, while the paths they name still exist. The values
        # written are the post-rename ones.
        for edit in plan.content_edits:
            try:
                if edit.what == "IntendedFor":
                    _rewrite_intended_for(
                        op, edit.path, old_token, new_token, moved_names,
                    )
                elif edit.what == "participant_id":
                    if not whole:
                        continue     # half a subject is still the old one
                    _rewrite_column(
                        op, edit.path, "participant_id",
                        old_token, new_token, None,
                    )
                elif edit.path in relocating and edit.path not in travelling:
                    # Rows in this table are being moved to another table.
                    # Rewriting them here would leave it pointing at files
                    # that have left its scope, which is exactly the dangling
                    # reference this used to produce.
                    pass
                else:
                    _rewrite_column(
                        op, edit.path, "filename",
                        old_token, new_token, moved_names,
                    )
                touched += 1
            except OSError as exc:
                errors.append(f"{edit.rel}: {exc}")

        # Folders BEFORE files. The other order looked natural and was wrong:
        # moving a file into its new parent creates that folder as a side
        # effect, and the folder rename then collides with the directory it
        # just caused to exist.
        moved_dirs: dict[Path, Path] = {}
        for src in sorted(renaming_dirs, key=lambda p: len(p.parts)):
            dst = renaming_dirs[src]
            try:
                if src.exists():
                    op.rename(src, dst)
                    moved_dirs[src] = dst
                    touched += 1
            except OSError as exc:
                errors.append(f"{_rel(root, src)}: {exc}")

        # A file's recorded source is its path BEFORE the folder moved, so it
        # is remapped through the moves just made.
        for src, dst in sorted(
            moves, key=lambda m: len(m[0].parts), reverse=True,
        ):
            actual = _remap_parent(src.parent, moved_dirs) / src.name
            if actual == dst:
                continue          # the folder rename already did it
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                op.rename(actual, dst)
                touched += 1
            except OSError as exc:
                errors.append(f"{_rel(root, actual)}: {exc}")

        # Fusion only: the two tables that describe the subjects themselves.
        # Both are about the subject as a whole, so neither applies to a
        # partial move.
        if whole:
            # ``table_merges`` is NOT applied here. It used to append the
            # source table's rows verbatim, which left them naming the subject
            # they came from, and it ran before the general relocation and
            # deleted the table that relocation was about to read.
            # ``apply_scans_rows`` above moves every row correctly and deletes
            # the table it emptied, so the special case is gone. The plan
            # still reports the merge, because the dry run has to say it.
            for path, column in plan.row_folds:
                try:
                    _fold_row(op, path, column, old_token, new_token)
                    touched += 1
                except OSError as exc:
                    errors.append(f"{_rel(root, path)}: {exc}")

        # Every row that has to change table, whichever level the table is at.
        try:
            touched += apply_scans_rows(op, row_moves)
        except OSError as exc:
            errors.append(f"scans tables: {exc}")

        # A split: finish making the new subject a subject rather than a
        # folder of orphaned files.
        for split in splits:
            try:
                touched += _apply_split(op, root, split)
            except OSError as exc:
                errors.append(f"{split.target_label}: {exc}")

        # The source subject's folder, now that everything is out of it. Left
        # behind it would validate as an empty subject, which is worse than
        # the rename not having happened. ``_remove_if_empty`` is a no-op when
        # a partial selection left files in it.
        for d in sorted(plan.emptied, key=lambda p: len(p.parts), reverse=True):
            _remove_if_empty(op, d, root)

        # Every directory this operation could have emptied: the folders the
        # moved files came out of, and the source of any folder rename that
        # did not happen as a whole. An empty anat/ is not wrong exactly, but
        # it is a folder claiming a modality is there when it is not, and an
        # empty sub-XXX/ claims a subject.
        candidates = list(_emptied_by(moves))
        candidates += [src for src, _dst in plan.dir_moves]
        candidates += [src for src, _dst in plan.fused_dirs]
        # Deepest first, so a datatype folder goes before the session holding
        # it, and deduplicated: every moved file contributes its ancestors.
        for directory in sorted(
            dict.fromkeys(candidates), key=lambda p: len(p.parts), reverse=True,
        ):
            if directory != root:
                _remove_if_empty(op, directory, root)
    return touched, errors


@dataclass
class _Split:
    """One subject becoming two, because only some of its files were renamed.

    Renaming a few of ``sub-001``'s files to ``sub-002`` does not produce a
    folder of loose files: it produces a SUBJECT, and a subject is more than a
    directory. It needs its own ``*_scans.tsv`` carrying the rows that left,
    a row in ``participants.tsv``, and the folders its files vacated have to
    go, or the old subject keeps an empty ``anat/`` that every tool reading
    the tree will take for a modality that is there.
    """

    source: Path            # the subject directory files are leaving
    target: Path            # the subject directory they are joining
    moved: list[tuple[str, str]]   # (path within source, path within target)

    @property
    def source_label(self) -> str:
        return self.source.name

    @property
    def target_label(self) -> str:
        return self.target.name


def _subject_dir(root: Path, path: Path) -> Optional[Path]:
    """The ``sub-*`` directory a path lives under, if any."""
    try:
        parts = Path(path).resolve().relative_to(Path(root).resolve()).parts
    except (ValueError, OSError):
        return None
    return root / parts[0] if parts and parts[0].startswith("sub-") else None


def _plan_splits(root: Path, moves: list[tuple[Path, Path]]) -> list[_Split]:
    """Group the moves by the subject they leave and the one they join.

    Only moves that actually cross subjects count. A rename that keeps every
    file under the same subject is not a split, however partial it is.
    """
    grouped: dict[tuple[Path, Path], list[tuple[str, str]]] = {}
    for src, dst in moves:
        source = _subject_dir(root, src)
        target = _subject_dir(root, dst)
        if source is None or target is None or source == target:
            continue
        grouped.setdefault((source, target), []).append((
            _rel(source, src), _rel(target, dst),
        ))
    return [
        _Split(source=source, target=target, moved=moved)
        for (source, target), moved in grouped.items()
    ]


def _owns_any_move(path: Path, splits: list[_Split]) -> bool:
    """Is this table one a split is moving rows out of?"""
    return any(
        Path(path).parent == split.source for split in splits
    )


def _apply_split(op, root: Path, split: _Split) -> int:
    """Give the new subject the one thing only a split can know it needs.

    The scans rows are handled by :func:`apply_scans_rows` for EVERY move,
    split or not, because a row has to follow its file whether or not the
    subject changed. What is left here is the participants row, which exists
    only because a new subject exists.
    """
    return _add_participant_row(op, root, split)


def scans_home(path: Path, root: Path) -> Optional[Path]:
    """The directory whose ``*_scans.tsv`` governs ``path``.

    BIDS puts the table beside the thing it describes: at the SESSION level
    when there are sessions, at the subject level when there are not, and the
    ``filename`` column is relative to whichever of those it is. Looking only
    at the subject level, which is what this used to do, meant that every
    dataset with sessions was handled by accident or not at all.

    Returned whether or not a table exists there yet, because a split has to
    be able to CREATE one.
    """
    try:
        parts = Path(path).resolve().relative_to(Path(root).resolve()).parts
    except (ValueError, OSError):
        return None
    if not parts or not parts[0].startswith("sub-"):
        return None
    subject = root / parts[0]
    if len(parts) > 1 and parts[1].startswith("ses-"):
        session = subject / parts[1]
        # The session level is where BIDS puts it, and where a split should
        # create one. But a dataset that keeps a single subject-level table
        # despite having sessions is not unheard of, and quietly ignoring the
        # table it actually has would be worse than reading it: prefer an
        # EXISTING table, and only fall back to the standard location when
        # neither exists.
        if scans_table_in(session).exists():
            return session
        if scans_table_in(subject).exists():
            return subject
        return session
    return subject


def scans_table_in(directory: Path) -> Path:
    """Where this directory's scans table lives, existing or not.

    Named for the directory's own entities, which is what BIDS requires:
    ``sub-01/ses-pre`` holds ``sub-01_ses-pre_scans.tsv``.
    """
    try:
        rel = directory.relative_to(directory.parents[-1])
    except (ValueError, IndexError):
        rel = Path(directory.name)
    del rel
    parts = [p for p in (directory.parent.name, directory.name)
             if p.startswith(("sub-", "ses-"))]
    if directory.name.startswith("sub-"):
        parts = [directory.name]
    stem = "_".join(dict.fromkeys(parts))
    return directory / f"{stem}_scans.tsv"


@dataclass
class _RowMove:
    """One scans row that has to follow the file it describes."""

    from_table: Path
    to_table: Path
    old_rel: str
    new_rel: str


def plan_scans_rows(
    root: Path,
    moves: list[tuple[Path, Path]],
    moved_dirs: dict[Path, Path],
) -> list[_RowMove]:
    """Which scans rows have to move, and where to.

    A row only has to MOVE when the file leaves the scope of the table that
    described it. When the table travels with the file, which is what happens
    on a whole-subject or whole-session rename, the ordinary column rewrite
    has already done the work and there is nothing to relocate.

    Everything here is computed from the pre-move tree, so the answer does not
    depend on what has been applied yet.
    """
    # A scans table is a file, and a rename can move the table itself: it
    # carries the entity being renamed, so ``sub-001_ses-01_scans.tsv`` is in
    # the move list beside the recordings it describes. When that happens the
    # rows travel WITH it and only need the ordinary column rewrite. Missing
    # this left a moved table full of rows naming the subject it came from.
    moved_files = {src: dst for src, dst in moves}
    out: list[_RowMove] = []
    for src, dst in moves:
        home = scans_home(src, root)
        if home is None:
            continue
        # The destination table goes at the SAME LEVEL the source keeps its
        # tables at. BIDS puts them beside the session, but a dataset that
        # keeps one per subject despite having sessions is internally
        # consistent, and a split that silently switched convention would
        # make it inconsistent. Mirror what is there.
        target_home = _mirror_home(home, root, dst)
        if target_home is None:
            continue
        table = scans_table_in(home)
        if table in moved_files:
            # The table is moving too. Where does it land?
            if moved_files[table].parent == target_home:
                continue                 # with the file: the rewrite covers it
            landed = moved_files[table].parent
        else:
            landed = _remap_parent(home, moved_dirs)
        if landed == target_home:
            continue                     # the table came along; nothing to do
        out.append(_RowMove(
            from_table=scans_table_in(landed)
            if table not in moved_files else moved_files[table],
            to_table=scans_table_in(target_home),
            old_rel=_rel(home, src),
            new_rel=_rel(target_home, dst),
        ))
    return out


def _mirror_home(home: Path, root: Path, dst: Path) -> Optional[Path]:
    """Where ``dst``'s table belongs, at the same level as ``home``.

    ``home`` is either a subject directory or a session directory under one.
    The answer is the corresponding directory on the destination's side.
    """
    try:
        depth = len(home.relative_to(root).parts)
        parts = dst.relative_to(root).parts
    except ValueError:
        return scans_home(dst, root)
    if len(parts) < depth:
        return None
    return root.joinpath(*parts[:depth])


def apply_scans_rows(op, row_moves: list[_RowMove]) -> int:
    """Move each row out of its old table and into the one that now owns it.

    Grouped per table so each file is read and written once, however many rows
    changed hands.
    """
    from .tsv_edit import NA, TsvTable, read_table

    if not row_moves:
        return 0

    leaving: dict[Path, dict[str, str]] = {}
    for move in row_moves:
        leaving.setdefault(move.from_table, {})[move.old_rel] = move.new_rel
    destination: dict[Path, Path] = {
        move.from_table: move.to_table for move in row_moves
    }

    touched = 0
    for source_table, mapping in leaving.items():
        target_table = destination[source_table]
        if not source_table.exists():
            # Nothing described these files. A subject with no scans table is
            # allowed; inventing one here would state times nobody recorded.
            continue
        table = read_table(source_table)
        if table is None or "filename" not in table.header:
            continue
        index = table.header.index("filename")

        def cell(row: list[str]) -> str:
            return row[index] if index < len(row) else ""

        moving = [r for r in table.rows if cell(r) in mapping]
        if not moving:
            continue
        staying = [r for r in table.rows if cell(r) not in mapping]
        for row in moving:
            row[index] = mapping[cell(row)]

        existing = read_table(target_table) if target_table.exists() else None
        if existing is None or "filename" not in (existing.header or []):
            merged = TsvTable(header=list(table.header), rows=moving)
        else:
            header = list(existing.header) + [
                c for c in table.header if c not in existing.header
            ]
            rows = [_widen(r, existing.header, header, NA) for r in existing.rows]
            rows += [_widen(r, table.header, header, NA) for r in moving]
            merged = TsvTable(header=header, rows=rows)

        op.write_text(target_table, merged.to_text())
        if staying:
            op.write_text(
                source_table,
                TsvTable(header=table.header, rows=staying).to_text(),
            )
        else:
            # Every row left. An empty table is not a table.
            op.delete(source_table)
        touched += 1
    return touched


def _widen(row: list[str], header: list[str], target: list[str], fill: str):
    """One row re-expressed against a wider header."""
    values = {name: row[i] if i < len(row) else fill
              for i, name in enumerate(header)}
    return [values.get(name, fill) for name in target]


def _add_participant_row(op, root: Path, split: _Split) -> int:
    """Give the new subject a row, copied from the one it came from.

    ``participants.tsv`` has to list every subject, and a split creates one.
    The source's values are the only information that exists about the new
    subject, so they are copied rather than left blank, and the user can
    correct them: a split usually means the files were a different session or
    a different person, and only the user knows which.
    """
    from .tsv_edit import NA, read_table

    participants = root / "participants.tsv"
    if not participants.exists():
        return 0
    table = read_table(participants)
    if table is None or "participant_id" not in table.header:
        return 0
    index = table.header.index("participant_id")

    def cell(row: list[str]) -> str:
        return row[index] if index < len(row) else ""

    if any(cell(row) == split.target_label for row in table.rows):
        return 0
    source_row = next(
        (row for row in table.rows if cell(row) == split.source_label), None,
    )
    new_row = list(source_row) if source_row else [NA] * len(table.header)
    while len(new_row) < len(table.header):
        new_row.append(NA)
    new_row[index] = split.target_label
    table.rows.append(new_row)
    # Sorted, because a participants table out of order is hard to read and
    # every tool that writes one writes it sorted.
    table.rows.sort(key=cell)
    op.write_text(participants, table.to_text())
    return 1


def _emptied_by(moves: list[tuple[Path, Path]]) -> list[Path]:
    """Directories the moves may have emptied, deepest first.

    Only the ones files actually left. Checking the whole tree would be both
    slower and wrong: an empty folder that was already there is not this
    operation's business.
    """
    seen: dict[Path, None] = {}
    for src, _dst in moves:
        parent = src.parent
        while parent != parent.parent:
            seen.setdefault(parent, None)
            parent = parent.parent
    return sorted(seen, key=lambda p: len(p.parts), reverse=True)


def _within(path: Path, root: Path) -> bool:
    """Is this inside the dataset? Pruning must never climb out of it."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _folder_fully_selected(
    folder: Path, chosen: set[Path], whole: bool,
) -> bool:
    """May this folder be renamed as a unit?

    Only when nothing inside it is staying behind, because a folder rename
    moves everything under it whether it was chosen or not.
    """
    if whole:
        return True
    if not folder.is_dir():
        return False
    for path in folder.rglob("*"):
        if path.is_file() and path not in chosen:
            return False
    return True


def _fold_row(op, path: Path, column: str,
              old_token: str, new_token: str) -> None:
    """Fold the ``old_token`` row into the ``new_token`` one.

    The surviving row keeps everything it states and gains everything the
    folded row states that it does not. Neither row's information is lost,
    which is the whole point of a fusion: two records of one person.
    """
    from .tsv_edit import NA, read_table

    table = read_table(path)
    if table is None or column not in table.header:
        return
    idx = table.header.index(column)

    def _value(row: list[str], i: int) -> str:
        return row[i] if i < len(row) else NA

    target = next(
        (r for r in table.rows if _value(r, idx) == new_token), None,
    )
    source = next(
        (r for r in table.rows if _value(r, idx) == old_token), None,
    )
    if source is None:
        return
    if target is None:
        # Nothing to fold into: this is an ordinary rename of the one row.
        source[idx] = new_token
    else:
        for i in range(len(table.header)):
            if i == idx:
                continue
            current, incoming = _value(target, i), _value(source, i)
            if (not current or current == NA) and incoming and incoming != NA:
                while len(target) <= i:
                    target.append(NA)
                target[i] = incoming
        table.rows.remove(source)
    op.write_text(path, table.to_text())


def _remove_if_empty(op, directory: Path, root: Path) -> None:
    """Drop ``directory`` if nothing is in it, and any parent it empties.

    Just asks the filesystem. ``rmdir`` fails harmlessly on a directory that
    still holds something, which is exactly the test, so there is no need to
    look inside first.

    That matters more than it sounds: this used to ``rglob`` the whole subtree
    of every candidate directory, and callers hand it one candidate per
    ancestor of every moved file. On a twelve-file dataset that was 270,000
    ``rglob`` calls and 7.8 of a 8.7 second rename.

    Walking UP afterwards is what removes a session folder whose only
    datatype folder just went.
    """
    current = directory
    while current.is_dir() and _within(current, root):
        try:
            current.rmdir()
        except OSError:
            return              # still holds something; so does everything above
        current = current.parent


def _rewrite_intended_for(
    op, path: Path, old_token: str, new_token: str,
    moved_names: Optional[set[str]] = None,
) -> None:
    data = _load_json(path)
    if data is None or "IntendedFor" not in data:
        return
    value = data["IntendedFor"]
    items = value if isinstance(value, list) else [value]
    fixed = [
        _swap_token(str(v), old_token, new_token)
        if _names_a_moved_file(str(v), moved_names) else str(v)
        for v in items
    ]
    data["IntendedFor"] = fixed if isinstance(value, list) else fixed[0]
    op.write_json(path, data)


def _names_a_moved_file(text: str, moved_names: Optional[set[str]]) -> bool:
    """Does this reference point at one of the files that actually moved?

    ``None`` means everything moved, so every reference follows. Otherwise the
    basename decides: a reference to a file left behind must keep naming it,
    or the rename turns a working pointer into a dangling one.
    """
    if moved_names is None:
        return True
    return text.rsplit("/", 1)[-1] in moved_names


def _rewrite_column(op, path: Path, column: str,
                    old_token: str, new_token: str,
                    moved_names: Optional[set[str]] = None) -> None:
    from .tsv_edit import read_table

    table = read_table(path)
    if table is None or column not in table.header:
        return
    idx = table.header.index(column)
    for row in table.rows:
        if idx < len(row) and _names_a_moved_file(row[idx], moved_names):
            row[idx] = _swap_token(row[idx], old_token, new_token)
    op.write_text(path, table.to_text())


def _swap_token(text: str, old_token: str, new_token: str) -> str:
    """Replace whole ``key-value`` segments only, never substrings."""
    out = []
    for chunk in re.split(r"([/_])", text):
        out.append(new_token if chunk == old_token else chunk)
    return "".join(out)


def list_values(root: Path, entity: str) -> list[str]:
    """Every value ``entity`` currently takes in the dataset, sorted."""
    root = Path(root)
    seen: set[str] = set()
    for p in walk_dataset(root):
        value = entity_value(p.name, entity)
        if value:
            seen.add(value)
    return sorted(seen)


__all__ = [
    "folder_entities",
    "would_fuse",
    "ContentEdit",
    "RenameError",
    "RenamePlan",
    "apply_rename",
    "entity_value",
    "list_values",
    "plan_rename",
    "rename_in_name",
]
