"""Map :mod:`bidsval` results into BIDS Manager's ``editor/types`` shapes.

The GUI, workers, HTML report, and CLI all bind to
:class:`bidsmgr.editor.types.ValidationReport` / ``FileVerdict`` / ``Issue``.
bidsval returns near-isomorphic pydantic models; this module is the single
translation layer between the two, plus the BIDS Manager-native supplements
(``sidecar_fields`` + TODO findings from :mod:`bidsmgr.editor.bidsmgr_checks`).

Severity mapping: bidsval ``error`` / ``warning`` / ``ignore`` ->
``ERR`` / ``WARN`` / ``OK`` (a clean file, severity ``None``, also maps to ``OK``).

Finding attribution follows bidsval (and the BIDS spec): metadata findings for a
data file (for example a missing required ``RepetitionTime``) attach to the data
file (``*_bold.nii.gz``), not to its ``.json`` sidecar. The sidecar form still
shows which fields are missing because :func:`to_bm_file_verdict` populates
``sidecar_fields`` on the ``.json`` verdict from the schema.
"""

from __future__ import annotations

from pathlib import Path

from bidsval import Severity as BvSeverity

from . import bidsmgr_checks as bm
from .types import (
    FieldLevel,
    FileVerdict,
    Issue,
    Severity,
    SidecarField,
    ValidationReport,
    rollup_severity,
)

# bidsval severity -> BIDS Manager severity.
_SEVERITY: dict[BvSeverity, Severity] = {
    BvSeverity.ERROR: Severity.ERR,
    BvSeverity.WARNING: Severity.WARN,
    BvSeverity.IGNORE: Severity.OK,
}


def to_bm_severity(severity) -> Severity:
    """Map a bidsval severity (or ``None`` for a clean file) to a BM severity."""
    if severity is None:
        return Severity.OK
    return _SEVERITY.get(severity, Severity.WARN)


def _compose_message(bv_issue) -> str:
    """Build the BM message: bidsval message + its actionable suggestion.

    bidsval splits the *what* (``message``) from the *how to fix it*
    (``suggestion``); BIDS Manager's ``Issue`` has a single message field, so we
    join them. Kept plain text (no HTML) so both the rich-text validation pane
    and the HTML report - which escapes the message - render it correctly.
    """
    message = bv_issue.message or bv_issue.code or "validation finding"
    if bv_issue.suggestion:
        message = f"{message}  ·  {bv_issue.suggestion}"
    return message


def to_bm_issue(bv_issue) -> Issue:
    """Translate one bidsval :class:`Issue` into a BM :class:`Issue`.

    The JSON key / column / entity the finding refers to (``sub_code``, or the
    fix's target field) becomes ``field`` so the Editor's fix button can jump to
    that row in the sidecar form. The button is only offered when there is a
    field to jump to.
    """
    fix = bv_issue.fix
    field = (fix.field if (fix and fix.field) else bv_issue.sub_code) or None
    fix_label = (fix.label or "Fix") if (fix and field) else None
    fix_action = fix.action if fix else None
    return Issue(
        severity=to_bm_severity(bv_issue.severity),
        rule_id=bv_issue.code,
        message=_compose_message(bv_issue),
        field=field,
        line=getattr(bv_issue, "line", None),
        lines=list(getattr(bv_issue, "lines", []) or []),
        fix_label=fix_label,
        fix_action=fix_action,
    )


def to_bm_file_verdict(bv_verdict, bids_root: Path, *, flag_todos: bool = True) -> FileVerdict:
    """Translate a bidsval :class:`FileVerdict`, adding the BM-native supplements.

    For ``.json`` files this populates the schema-aware ``sidecar_fields`` the
    Editor's sidecar form renders and (when ``flag_todos``) appends
    TODO-placeholder findings. Severity is rolled up from the final issue set so
    the supplements count. ``flag_todos=False`` gives exact bidsval parity.
    """
    rel = Path(bv_verdict.path)
    abs_path = (Path(bids_root) / rel)
    datatype, suffix = bm.infer_datatype_suffix(abs_path, Path(bids_root))

    issues = [to_bm_issue(i) for i in bv_verdict.issues]
    sidecar_fields = []
    if rel.name.lower().endswith(".json"):
        if flag_todos:
            issues.extend(bm.todo_issues_for(abs_path))
        sidecar_fields = bm.sidecar_fields_for(abs_path, datatype, suffix)

    severity = rollup_severity([i.severity for i in issues])
    return FileVerdict(
        path=rel,
        severity=severity,
        datatype=datatype,
        suffix=suffix,
        issues=issues,
        sidecar_fields=sidecar_fields,
    )


def to_bm_report(
    bv_report,
    bids_root: Path,
    *,
    bidsmgr_version: str,
    generated_at: str,
    flag_todos: bool = True,
) -> ValidationReport:
    """Translate a whole bidsval :class:`ValidationReport`.

    ``folder_issues`` stays empty (bidsval has no folder-scoped findings; the
    validation pane handles an empty dict). ``counts`` / ``severity`` are rolled
    up with the same per-file + dataset-issue accounting the GUI's own
    ``_recompute_report_summary`` uses, so the chips and summary line are stable.
    """
    files = [
        to_bm_file_verdict(fv, bids_root, flag_todos=flag_todos)
        for fv in bv_report.files
    ]
    # Surface each data file's sidecar metadata findings on its editable .json
    # sibling too (so they show where the user fixes them, and the fix button
    # jumps to the field). Mirrors are flagged so they are not double-counted.
    _mirror_sidecar_findings(files)
    # Make sure every flagged field has a sidecar form row, so the Editor's
    # highlight / fix can locate it. bidsval flags a broader set of recommended
    # fields than the schema-derived form rows, so without this only fields that
    # already had a row (incl. present TODO fields) could be highlighted.
    _ensure_field_rows(files)
    dataset_issues = [to_bm_issue(i) for i in bv_report.dataset_issues.issues]

    report = ValidationReport(
        bids_root=Path(bids_root),
        bidsmgr_version=bidsmgr_version,
        bids_version=bv_report.bids_version or "",
        generated_at=generated_at,
        files=files,
        dataset_issues=dataset_issues,
        folder_issues={},
    )

    # Counts are PER-FINDING and exclude mirrors (the canonical copy is counted
    # once), so the GUI chips agree with bidsval's numbers. "ok" is the count of
    # clean files (a friendly "valid files" tally).
    counted: list[Issue] = [i for i in dataset_issues if not i.mirrored]
    for f in files:
        counted.extend(i for i in f.issues if not i.mirrored)
    report.severity = rollup_severity([i.severity for i in counted])
    report.counts = {
        "ok": sum(1 for f in files if f.severity is Severity.OK),
        "warn": sum(1 for i in counted if i.severity is Severity.WARN),
        "err": sum(1 for i in counted if i.severity is Severity.ERR),
    }
    return report


def _mirror_sidecar_findings(files: list[FileVerdict]) -> None:
    """Append a mirrored copy of each data file's field-bearing findings onto
    its sibling ``.json`` verdict (in place).

    A sidecar metadata finding canonically attaches to the data file
    (``*_bold.nii.gz``); this also surfaces it on the editable ``*_bold.json``
    so it shows where the user edits and the fix button can jump to the field.
    Mirrors carry ``mirrored=True`` so they render but are not counted twice.
    """
    index = {str(f.path): f for f in files}
    for f in files:
        sib = bm.sibling_json(f.path)
        if sib is None:
            continue
        target = index.get(str(sib))
        if target is None:
            continue
        mirrors = [
            issue.model_copy(update={"mirrored": True})
            for issue in f.issues
            if issue.field and not issue.mirrored
        ]
        if mirrors:
            target.issues.extend(mirrors)
            target.severity = rollup_severity([i.severity for i in target.issues])


_LEVEL_RANK = {Severity.ERR: 2, Severity.WARN: 1, Severity.OK: 0}


def _ensure_field_rows(files: list[FileVerdict]) -> None:
    """For every ``.json`` verdict, add a sidecar form row for each field that a
    finding refers to but that has no row yet (in place).

    The schema-derived rows (``sidecar_fields``) cover only part of what bidsval
    flags - it warns about a broader set of recommended fields. Without a row a
    finding's field cannot be scrolled to / highlighted / focused, which is why
    only already-present (e.g. TODO) fields could be highlighted before. Added
    rows are missing fields, leveled by the finding's worst severity.
    """
    for f in files:
        if not str(f.path).lower().endswith(".json"):
            continue
        existing = {sf.name for sf in f.sidecar_fields}
        worst: dict[str, Severity] = {}
        for issue in f.issues:
            if not issue.field or issue.field in existing:
                continue
            if _LEVEL_RANK.get(issue.severity, 0) > _LEVEL_RANK.get(
                worst.get(issue.field, Severity.OK), 0
            ):
                worst[issue.field] = issue.severity
        for name, sev in worst.items():
            level = FieldLevel.REQUIRED if sev is Severity.ERR else FieldLevel.RECOMMENDED
            f.sidecar_fields.append(SidecarField(
                level=level, name=name, value=None, present=False,
                value_kind="missing",
            ))


__all__ = ["to_bm_severity", "to_bm_issue", "to_bm_file_verdict", "to_bm_report"]
