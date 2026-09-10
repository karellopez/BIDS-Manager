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
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    return bool(rel.parts) and rel.parts[0] in _skip_top_level()


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

    for f in sorted(root.rglob("*")):
        if not f.is_file() or _skip(f, root):
            continue
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

    plan.content_edits = _plan_content_edits(root, entity, old, new)

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
    root = Path(root)
    token = f"{entity}-{new.strip()}"
    return any(
        d.is_dir() and not _skip(d, root) for d in root.rglob(token)
    )


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
) -> list[ContentEdit]:
    """The three places BIDS points at a filename from inside a file."""
    out: list[ContentEdit] = []
    old_token = f"{entity}-{old}"

    for p in sorted(root.rglob("*.json")):
        if _skip(p, root):
            continue
        data = _load_json(p)
        if not data or "IntendedFor" in data is None:
            continue
        hits = _count_intended_for(data.get("IntendedFor"), old_token)
        if hits:
            out.append(ContentEdit(p, _rel(root, p), "IntendedFor", hits))

    for p in sorted(root.rglob("*_scans.tsv")):
        if _skip(p, root):
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
        for src, dst in sorted(
            plan.dir_moves, key=lambda m: len(m[0].parts),
        ):
            if not _folder_fully_selected(src, chosen, whole):
                # Something inside is staying. Renaming the folder would drag
                # it along under a label that is not its own.
                continue
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
            for src, dst in plan.table_merges:
                try:
                    _merge_tables(op, src, dst)
                    touched += 1
                except OSError as exc:
                    errors.append(f"{_rel(root, src)}: {exc}")
            for path, column in plan.row_folds:
                try:
                    _fold_row(op, path, column, old_token, new_token)
                    touched += 1
                except OSError as exc:
                    errors.append(f"{_rel(root, path)}: {exc}")

        # The source subject's folder, now that everything is out of it. Left
        # behind it would validate as an empty subject, which is worse than
        # the rename not having happened. ``_remove_if_empty`` is a no-op when
        # a partial selection left files in it.
        for d in sorted(plan.emptied, key=lambda p: len(p.parts), reverse=True):
            _remove_if_empty(op, d)
    return touched, errors


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


def _merge_tables(op, src: Path, dst: Path) -> None:
    """Append ``src``'s rows to ``dst``, unioning the columns.

    Used when two subjects fuse and both carry a ``*_scans.tsv``. A row is
    never dropped and a column one table lacks is filled with ``n/a``, which
    is what BIDS says a table says when it has nothing to say.
    """
    from .tsv_edit import NA, TsvTable, read_table

    a, b = read_table(dst), read_table(src)
    if a is None or b is None:
        return
    header = list(a.header) + [c for c in b.header if c not in a.header]
    rows = [
        [row[a.header.index(c)] if c in a.header
         and a.header.index(c) < len(row) else NA for c in header]
        for row in a.rows
    ] + [
        [row[b.header.index(c)] if c in b.header
         and b.header.index(c) < len(row) else NA for c in header]
        for row in b.rows
    ]
    op.write_text(dst, TsvTable(header, rows).to_text())
    op.delete(src)


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


def _remove_if_empty(op, directory: Path) -> None:
    """Drop a directory tree that holds no files. Deepest first."""
    if not directory.is_dir():
        return
    for child in sorted(
        directory.rglob("*"), key=lambda p: len(p.parts), reverse=True,
    ):
        if child.is_dir():
            try:
                child.rmdir()
            except OSError:
                return          # something is still in it; leave the tree
    try:
        directory.rmdir()
    except OSError:
        log.debug("%s is not empty after the merge; left in place", directory)


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
    for p in root.rglob("*"):
        if not p.is_file() or _skip(p, root):
            continue
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
