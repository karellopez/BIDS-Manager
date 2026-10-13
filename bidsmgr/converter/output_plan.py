"""What a conversion will write, before it runs.

The Filter panel's BIDS preview shows the dataset the conversion and the
metadata step will produce: every data file AND what comes with it (a sidecar,
``.bval``/``.bvec``, MNE-BIDS' channel and event tables, an eye tracker's two
files per eye, the ``*_scans.tsv`` of each session, ``dataset_description.json``,
``participants.tsv``, a ``sourcedata/`` copy), and what each file will roughly
contain, so a user sees the result of their settings before converting.

Two halves:

* :func:`plan_output` names the files, the way each converter names them
  (dcm2niix, MNE-BIDS, the physio and eye-tracking writers, the metadata step).
  A file a converter writes only sometimes is :attr:`PlannedFile.conditional`
  with the reason ("when the recording has events").
* :func:`approximate` predicts one file's content from what the scan already
  knows. Every value says where it comes from (:data:`SOURCES`): read from the
  header during the scan, read by the converter at conversion (shown as ``…``,
  the value is not known yet), written by BIDS Manager from the settings, or a
  ``TODO`` placeholder the fill-todos setting writes.

It is a prediction and says so. Qt-free: the GUI calls it on a worker thread.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

#: Where a value in a predicted file comes from.
SOURCES = {
    "scan": "read from the header during the scan",
    "converter": "read by the converter at conversion (not known yet)",
    "settings": "written by BIDS Manager from your settings and the table",
    "todo": "a placeholder the fill-todos setting writes",
}
#: Shown for a value the converter reads at conversion.
UNKNOWN = "…"

_FILE_BASED_EEG = {".edf": ".edf", ".bdf": ".bdf", ".set": ".set", ".vhdr": ".vhdr"}
_MEG_FORMATS = {".fif": ".fif", ".fif.gz": ".fif", ".ds": ".ds", ".con": ".con", ".sqd": ".sqd",
                ".kdf": ".kdf"}


@dataclass(frozen=True)
class PlanOptions:
    """The settings that change what is written."""

    dataset: str = ""
    run_metadata: bool = True
    fill_todos: bool = True
    #: Which requirement levels fill-todos covers ("required", "recommended",
    #: "optional"), as Settings > Metadata says.
    fill_scope: str = "recommended"
    intended_for: str = "uri"            # "uri" | "relative"
    b0_field: bool = True
    sourcedata: str = "none"             # "none" | "copy" | "archive"
    sourcedata_modalities: tuple[str, ...] = ()
    force_edf: bool = False
    bids_version: str = ""
    version: str = ""                    # BIDS Manager's


@dataclass(frozen=True)
class PlannedFile:
    """One file the conversion or the metadata step will write."""

    path: str                      # relative to the dataset root, POSIX
    origin: str                    # who writes it
    rows: tuple[int, ...] = ()     # the inventory rows behind it
    conditional: str = ""          # why it may not be written; "" = always
    role: str = "data"             # data | sidecar | table | dataset | text | archive
    #: A file that will be written but is not valid BIDS, and why.
    warning: str = ""


@dataclass
class OutputPlan:
    files: list[PlannedFile] = field(default_factory=list)

    def paths(self) -> list[str]:
        return [f.path for f in self.files]

    def find(self, path: str) -> Optional[PlannedFile]:
        return next((f for f in self.files if f.path == path), None)

    def of_row(self, row: int) -> list[PlannedFile]:
        return [f for f in self.files if row in f.rows]


@dataclass(frozen=True)
class FilePreview:
    """A file's predicted content: ``fields`` for a JSON file (key, value,
    source), ``header`` and ``rows`` for a table, ``text`` for anything else,
    and what it cannot say in ``note``."""

    kind: str                                      # "json" | "tsv" | "text" | "binary" | "folder"
    fields: tuple[tuple[str, object, str], ...] = ()
    header: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    text: str = ""
    note: str = ""

    def json_text(self) -> str:
        """The JSON as it would read, ``…`` for what is not known yet."""
        return json.dumps({k: v for k, v, _s in self.fields}, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Reading the table
# ---------------------------------------------------------------------------


def _cell(df, i, col: str) -> str:
    if col not in df.columns:
        return ""
    v = df.at[i, col]
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v).strip()


def _included(df, included: Optional[Sequence[bool]]):
    if included is not None:
        return [bool(v) for v in included]
    if "include" not in df.columns:
        return [True] * len(df)
    return [str(v).strip() == "1" for v in df["include"].tolist()]


def _split_name(name: str) -> dict[str, str]:
    out = {}
    for token in name.split("_")[:-1]:
        key, sep, value = token.partition("-")
        if sep:
            out[key] = value
    return out


def _ext(path: str) -> str:
    low = path.lower()
    for double in (".fif.gz", ".nii.gz", ".tsv.gz"):
        if low.endswith(double):
            return double
    dot = low.rfind(".")
    return low[dot:] if dot >= 0 else ""


def _folder(subject: str, session: str, datatype: str) -> str:
    parts = [subject] + ([session] if session else []) + ([datatype] if datatype else [])
    return "/".join(parts)


def _safe(name: str) -> str:
    try:
        from ..util.paths import safe_path_component

        return safe_path_component(name)
    except Exception:  # noqa: BLE001 - only a name
        return re.sub(r"[^0-9A-Za-z._-]+", "_", name) or "series"


# ---------------------------------------------------------------------------
# What each converter writes
# ---------------------------------------------------------------------------


#: The signals a Siemens physio log can hold; bidsphysio writes one pair of
#: files per signal (``recording-<signal>``) when they were sampled apart.
_PHYSIO_SIGNALS = ("cardiac", "ecg", "respiratory", "trigger")


#: Suffixes whose echoes are the point of the series (a fieldmap's two
#: magnitudes): never split by echo.
_ECHO_SUFFIXES = {"magnitude", "magnitude1", "magnitude2", "phasediff", "phase1", "phase2",
                  "fieldmap", "epi"}
_SPLIT_WARNING = ("dcm2niix splits a series with several echoes into one file per echo "
                  "(_e1, _e2, ...), names BIDS does not allow; the validator will list them as "
                  "not included. Name each echo with the echo entity, or leave the series out.")


def _files_for_row(df, i, two_echoes: bool, force_edf: bool,
                   diffusion: bool = False, echoes: int = 1) -> list[PlannedFile]:
    name = _cell(df, i, "bids_name")
    datatype = _cell(df, i, "datatype")
    subject = _cell(df, i, "participant_id") or f"sub-{_split_name(name).get('sub', '')}"
    session = _cell(df, i, "session")
    if session and not session.startswith("ses-"):
        session = f"ses-{session}"
    folder = _folder(subject, session, datatype)
    base = f"{folder}/{name}"
    suffix = name.rsplit("_", 1)[-1]
    modality = _cell(df, i, "modality").lower()
    source = _cell(df, i, "source_file")
    rows = (int(i),)
    out: list[PlannedFile] = []

    def add(path, origin, role="data", conditional="", warning=""):
        out.append(PlannedFile(path=path, origin=origin, rows=rows, conditional=conditional,
                               role=role, warning=warning))

    if modality == "eyetrack":
        stem = base.rsplit("_", 1)[0]
        stem = re.sub(r"_recording-eye[12]", "", stem)
        for eye, cond in (("eye1", ""), ("eye2", "when the file records both eyes")):
            for kind in ("physio", "physioevents"):
                add(f"{stem}_recording-{eye}_{kind}.tsv.gz", "BIDS Manager (eye tracking)",
                    "data" if kind == "physio" else "table", cond)
                add(f"{stem}_recording-{eye}_{kind}.json", "BIDS Manager (eye tracking)",
                    "sidecar", cond)
        return out

    uid = _cell(df, i, "series_uid")
    if uid:                                              # a DICOM series
        if suffix == "physio":
            stem = base[: -len("_physio")]
            add(f"{base}.tsv.gz", "bidsphysio", conditional="when the log holds a single signal")
            add(f"{base}.json", "bidsphysio", "sidecar", "when the log holds a single signal")
            for signal in _PHYSIO_SIGNALS:
                cond = f"when the log holds a {signal} signal (one file per signal)"
                add(f"{stem}_recording-{signal}_physio.tsv.gz", "bidsphysio", conditional=cond)
                add(f"{stem}_recording-{signal}_physio.json", "bidsphysio", "sidecar", cond)
            return out
        origin = "dcm2niix"
        if echoes >= 2 and suffix not in _ECHO_SUFFIXES:
            for k in range(1, echoes + 1):
                add(f"{base}_e{k}.nii.gz", origin, warning=_SPLIT_WARNING)
                add(f"{base}_e{k}.json", origin, "sidecar", warning=_SPLIT_WARNING)
            return out
        add(f"{base}.nii.gz", origin)
        add(f"{base}.json", origin, "sidecar")
        if suffix == "dwi" or (suffix == "epi" and diffusion):
            add(f"{base}.bval", origin, "table")
            add(f"{base}.bvec", origin, "table")
        if suffix == "magnitude1":
            stem = base[: -len("magnitude1")]
            if two_echoes:
                add(f"{stem}magnitude2.nii.gz", origin)
                add(f"{stem}magnitude2.json", origin, "sidecar")
            if "|" in uid:                     # the magnitude and its phase, joined
                add(f"{stem}phasediff.nii.gz", origin)
                add(f"{stem}phasediff.json", origin, "sidecar")
        return out

    ext = _ext(source)
    if datatype == "pet":                               # ECAT
        add(f"{base}.nii.gz", "BIDS Manager (nibabel)")
        add(f"{base}.json", "BIDS Manager (nibabel)", "sidecar")
        return out
    if datatype in ("eeg", "ieeg", "meg", "nirs"):
        origin = "MNE-BIDS"
        if datatype == "meg":
            data_ext = _MEG_FORMATS.get(ext, ext or ".fif")
        elif force_edf and datatype != "nirs":
            data_ext = ".edf"
        elif datatype == "nirs":
            data_ext = ".snirf"
        else:
            data_ext = _FILE_BASED_EEG.get(ext, ".vhdr")
        add(f"{base}{data_ext}", origin)
        if data_ext == ".vhdr":
            add(f"{base}.eeg", origin)
            add(f"{base}.vmrk", origin)
        if data_ext == ".set" and ext == ".set":
            add(f"{base}.fdt", origin, conditional="when the EEGLAB file keeps its data apart")
        add(f"{base}.json", origin, "sidecar")
        stem = base.rsplit("_", 1)[0]
        # Positions describe the session, not a run: no task, no run.
        named = _split_name(name)
        place = "_".join(f"{k}-{named[k]}" for k in ("sub", "ses", "acq") if k in named)
        place = f"{folder}/{place}"
        add(f"{stem}_channels.tsv", origin, "table")
        add(f"{stem}_events.tsv", origin, "table",
            "when the recording has events or annotations")
        add(f"{stem}_events.json", origin, "sidecar",
            "when the recording has events or annotations")
        if datatype in ("eeg", "ieeg"):
            add(f"{place}_electrodes.tsv", origin, "table",
                "when electrode positions are known (a montage or digitised positions)")
            add(f"{place}_coordsystem.json", origin, "sidecar",
                "when electrode positions are known")
        if datatype == "meg":
            add(f"{place}_coordsystem.json", origin, "sidecar",
                "when the recording carries head digitisation")
        return out
    add(f"{base}{ext}", "BIDS Manager")
    return out


def _features_of(features: Optional[Mapping], uid: str):
    """A series' features; a row that joins two series (``uid|uid``) has
    the first one's."""
    if not features or not uid:
        return None
    return features.get(uid) or features.get(uid.split("|", 1)[0])


def _two_echoes(features: Mapping, uid: str) -> bool:
    feats = _features_of(features, uid)
    return bool(feats is not None and len(getattr(feats, "echo_times", []) or []) >= 2)


def _diffusion(features: Mapping, uid: str) -> bool:
    """A series acquired with diffusion encoding (its reference gets a
    ``.bval``/``.bvec`` of its own)."""
    feats = _features_of(features, uid)
    if feats is None:
        return False
    # GE stores a b-value of 0 on a plain EPI too: only a positive one counts.
    return any((b or 0) > 0 for b in (getattr(feats, "b_values", None) or [])) or \
        "DIFFUSION" in (getattr(feats, "image_type", []) or [])


def _echoes(features: Mapping, uid: str) -> int:
    feats = _features_of(features, uid)
    return len(getattr(feats, "echo_times", []) or []) if feats is not None else 1


def plan_output(df, included: Optional[Sequence[bool]] = None,
                options: PlanOptions = PlanOptions(),
                features: Optional[Mapping] = None) -> OutputPlan:
    """Every file the conversion and the metadata step will write."""
    flags = _included(df, included)
    plan = OutputPlan()
    sessions: dict[tuple[str, str], list[int]] = defaultdict(list)
    subjects: dict[str, list[int]] = defaultdict(list)
    keep = set(options.sourcedata_modalities or ())

    for pos, i in enumerate(df.index):
        if not flags[pos]:
            continue
        name, datatype = _cell(df, i, "bids_name"), _cell(df, i, "datatype")
        if not name or not datatype or datatype.startswith("derivatives"):
            continue
        uid = _cell(df, i, "series_uid")
        plan.files.extend(_files_for_row(df, i, _two_echoes(features or {}, uid),
                                         options.force_edf, _diffusion(features or {}, uid),
                                         _echoes(features or {}, uid)))
        subject = _cell(df, i, "participant_id")
        session = _cell(df, i, "session")
        if _lists_in_scans(df, i):
            sessions[(subject, session)].append(int(i))
        subjects[subject].append(int(i))
        if options.sourcedata in ("copy", "archive"):
            from .sourcedata import kind_for

            kind = kind_for(_cell(df, i, "modality"), datatype, name.rsplit("_", 1)[-1])
            if not keep or kind in keep:
                ses = f"/{session}" if session else ""
                if uid:
                    series = _safe(_cell(df, i, "sequence") or name)
                    dest = f"sourcedata/{subject}{ses}/{kind}/{series}"
                    dest += ".tar.gz" if options.sourcedata == "archive" else "/"
                else:
                    src = _cell(df, i, "source_file").replace("\\", "/").rsplit("/", 1)[-1]
                    dest = f"sourcedata/{subject}{ses}/{kind}/{src}"
                    dest += ".tar.gz" if options.sourcedata == "archive" else ""
                plan.files.append(PlannedFile(dest, "BIDS Manager (sourcedata)", (int(i),),
                                              role="archive" if dest.endswith(".tar.gz") else "data"))

    for (subject, session), rows in sessions.items():
        folder = _folder(subject, session, "")
        label = f"{subject}_{session}" if session else subject
        # MNE-BIDS writes the scans table of a session with EEG or MEG during
        # the conversion; for anything else the metadata step writes it.
        mne = any(_cell(df, r, "datatype") in ("eeg", "ieeg", "meg", "nirs")
                  and not _cell(df, r, "series_uid") for r in rows)
        if mne or options.run_metadata:
            plan.files.append(PlannedFile(f"{folder}/{label}_scans.tsv",
                                          "MNE-BIDS" if mne else "BIDS Manager (metadata)",
                                          tuple(rows), role="table"))
    if options.sourcedata in ("copy", "archive"):
        for subject, rows in subjects.items():
            plan.files.append(PlannedFile(f"sourcedata/{subject}/manifest.tsv",
                                          "BIDS Manager (sourcedata)", tuple(rows), role="table"))

    plan.files.append(PlannedFile("dataset_description.json", "BIDS Manager", role="dataset"))
    plan.files.append(PlannedFile(".bidsignore", "BIDS Manager", role="text"))
    if options.run_metadata:
        plan.files.append(PlannedFile("participants.tsv", "BIDS Manager (metadata)", role="dataset"))
        plan.files.append(PlannedFile("participants.json", "BIDS Manager (metadata)",
                                      role="dataset"))
        plan.files.append(PlannedFile("README", "BIDS Manager (metadata)", role="text",
                                      conditional="unless the dataset already has one"))
        plan.files.append(PlannedFile("CHANGES", "BIDS Manager (metadata)", role="text",
                                      conditional="unless the dataset already has one"))
    plan.files.sort(key=lambda f: _sort_key(f.path))
    return plan


def _lists_in_scans(df, i) -> bool:
    """Whether a row's files are rows of ``*_scans.tsv``: physio and eye
    tracking are not."""
    name = _cell(df, i, "bids_name")
    return not (_cell(df, i, "modality").lower() == "eyetrack" or name.endswith("_physio"))


def _sort_key(path: str) -> tuple:
    parts = path.split("/")
    top = parts[0]
    rank = 0 if len(parts) == 1 else (2 if top == "sourcedata" else 1)
    return (rank, [tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", part))
                   for part in parts])


# ---------------------------------------------------------------------------
# What a file will contain
# ---------------------------------------------------------------------------


def approximate(df, planned: PlannedFile, options: PlanOptions = PlanOptions(),
                features: Optional[Mapping] = None, spec=None,
                included: Optional[Sequence[bool]] = None) -> FilePreview:
    """``planned``'s predicted content (see :class:`FilePreview`)."""
    path = planned.path
    name = path.rsplit("/", 1)[-1]
    if path.endswith("/"):
        return FilePreview("folder", note="A copy of the series' DICOM files, as they are.")
    if name == "dataset_description.json":
        return _dataset_description(options)
    if name == "participants.tsv":
        return _participants_tsv(df, _included(df, included))
    if name == "participants.json":
        return _participants_json(df, _included(df, included))
    if name == "README":
        return FilePreview("text", text=_readme_text(options.dataset))
    if name == ".bidsignore":
        try:
            from ..cli._scaffold import BIDSIGNORE_LINES

            lines = list(BIDSIGNORE_LINES)
        except Exception:  # noqa: BLE001
            lines = [".bidsmgr/", ".tmp_bidsmgr/"]
        return FilePreview("text", text="\n".join(lines) + "\n",
                           note="BIDS Manager's own folders, so a validator skips them.")
    if name == "CHANGES":
        return FilePreview("text", text="1.0.0 <the date of the metadata step>\n  - Initial release.\n")
    if name.endswith("_scans.tsv"):
        return _scans_tsv(df, planned, features)
    if name == "manifest.tsv":
        return FilePreview("tsv", header=("source", "destination", "size", "sha256"),
                           note="One row per source file kept, with its checksum.")
    if not planned.rows:
        return FilePreview("binary")
    row = planned.rows[0]
    if name.endswith(".json"):
        return _sidecar(df, row, planned, options, features, spec, included)
    if name.endswith("_channels.tsv"):
        n = _cell(df, row, "n_channels")
        return FilePreview(
            "tsv", header=("name", "type", "units", "low_cutoff", "high_cutoff", "description",
                           "sampling_frequency", "status", "status_description"),
            note=f"One row per channel{f' ({n} channels)' if n else ''}, with the types the "
                 "scan found and the ones you set.")
    if name.endswith("_events.tsv"):
        return _events_tsv(df, row, spec)
    if name.endswith("_electrodes.tsv"):
        return FilePreview("tsv", header=("name", "x", "y", "z", "size"),
                           note="One row per electrode, from the montage or digitised positions.")
    if name.endswith((".bval", ".bvec")):
        feats = _features_of(features, _cell(df, row, "series_uid"))
        bvals = getattr(feats, "b_values", []) if feats else []
        n = getattr(feats, "n_directions", 0) if feats else 0
        what = ("b-values" if name.endswith(".bval") else "gradient directions (three rows: x, y, z)")
        return FilePreview("text", text=f"{UNKNOWN}\n",
                           note=f"The {what} of every volume, written by dcm2niix."
                                + (f" The scan found b = {', '.join(f'{b:g}' for b in bvals)}"
                                   f"{f' and {n} directions' if n else ''}." if bvals else ""))
    if name.endswith("_physioevents.tsv.gz"):
        return FilePreview("tsv", header=("onset", "duration", "trial_type", "message"),
                           note="Fixations, saccades, blinks and the tracker's messages.")
    if name.endswith("_physio.tsv.gz") or name.endswith(".tsv.gz"):
        return FilePreview("tsv", note="Samples, one column per signal, without a header "
                                       "(the columns are named in the JSON).")
    return FilePreview("binary", note=f"Written by {planned.origin}.")


def _todo(field_info) -> object:
    """The placeholder fill-todos writes for a field, or ``None`` for none
    (the metadata engine's own rule)."""
    from ..metadata.engine import NO_TODO, todo_value_for

    value = todo_value_for(field_info)
    return None if value is NO_TODO else value


def _levels(options: PlanOptions) -> frozenset:
    """The requirement levels the metadata step writes placeholders for."""
    if not (options.run_metadata and options.fill_todos):
        return frozenset()
    from ..metadata.engine import scope_levels

    return scope_levels(options.fill_scope)


def _dataset_description(options: PlanOptions) -> FilePreview:
    fields: list[tuple[str, object, str]] = [
        ("Name", options.dataset or UNKNOWN, "settings"),
        ("BIDSVersion", options.bids_version or UNKNOWN, "settings"),
        ("DatasetType", "raw", "settings"),
    ]
    levels = _levels(options)
    if levels:
        try:
            from .. import schema

            for f in schema.dataset_description_fields():
                if f.name in {k for k, _v, _s in fields} or f.name == "GeneratedBy":
                    continue
                if getattr(f, "level", "") in levels:
                    value = _todo(f)
                    if value is not None:
                        fields.append((f.name, value, "todo"))
        except Exception:  # noqa: BLE001 - the placeholders are a courtesy
            pass
    generated = [{"Name": "BIDS Manager", "Version": options.version or UNKNOWN,
                  "Description": "conversion"}]
    if options.run_metadata:
        generated.append({"Name": "BIDS Manager", "Version": options.version or UNKNOWN,
                          "Description": "metadata engine"})
    fields.append(("GeneratedBy", generated, "settings"))
    return FilePreview("json", fields=tuple(fields))


def _subjects(df, flags) -> list[str]:
    seen: list[str] = []
    for pos, i in enumerate(df.index):
        if flags[pos] and _cell(df, i, "bids_name"):
            s = _cell(df, i, "participant_id")
            if s and s not in seen:
                seen.append(s)
    return sorted(seen, key=lambda s: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)])


def _demographics(df, flags) -> dict[str, dict[str, str]]:
    from ..metadata.demographics import normalize_age, normalize_handedness, normalize_sex

    out: dict[str, dict[str, str]] = {}
    for pos, i in enumerate(df.index):
        if not flags[pos]:
            continue
        s = _cell(df, i, "participant_id")
        if not s:
            continue
        slot = out.setdefault(s, {})
        for col, key, norm in (("PatientAge", "age", normalize_age), ("PatientSex", "sex", normalize_sex),
                               ("Handedness", "handedness", normalize_handedness)):
            value = _cell(df, i, col)
            if value and not slot.get(key):
                slot[key] = norm(value) or ""
    return out


def _participants_tsv(df, flags) -> FilePreview:
    subjects = _subjects(df, flags)
    demo = _demographics(df, flags)
    columns = ["participant_id"] + [c for c in ("age", "sex", "handedness")
                                    if any(demo.get(s, {}).get(c) for s in subjects)]
    rows = tuple(tuple([s] + [demo.get(s, {}).get(c) or "n/a" for c in columns[1:]])
                 for s in subjects)
    note = ("Age, sex and handedness come from the DICOM header or the participants file you "
            "set; a column nobody answered is left out.")
    return FilePreview("tsv", header=tuple(columns), rows=rows, note=note)


def _participants_json(df, flags) -> FilePreview:
    from ..metadata.engine import PARTICIPANT_DESCRIPTIONS

    columns = _participants_tsv(df, flags).header
    return FilePreview("json", fields=tuple((c, PARTICIPANT_DESCRIPTIONS.get(c, {}), "settings")
                                           for c in columns))


def _readme_text(dataset: str) -> str:
    try:
        from ..metadata.engine import readme_text

        return readme_text(dataset or "<dataset>")
    except Exception:  # noqa: BLE001
        return f"# {dataset}\n"


def _iso_time(df, i) -> str:
    """The recording's time as the scans table writes it."""
    date = re.sub(r"\D", "", _cell(df, i, "study_date"))
    time = _cell(df, i, "acq_time")
    rec = _cell(df, i, "recording_time")
    if rec and re.match(r"\d{4}-\d{2}-\d{2}T", rec):
        return rec[:26].split("+")[0]
    if len(date) >= 8 and time:
        whole, _dot, frac = time.replace(":", "").partition(".")
        whole = re.sub(r"\D", "", whole)
        if len(whole) >= 6:
            stamp = f"{date[:4]}-{date[4:6]}-{date[6:8]}T{whole[:2]}:{whole[2:4]}:{whole[4:6]}"
            return stamp + (f".{(frac + '000000')[:6]}" if frac.isdigit() else "")
    # Not read by the scan: the converter takes it from the recording.
    return UNKNOWN


def _scans_tsv(df, planned: PlannedFile, features) -> FilePreview:
    """``filename`` relative to the session folder, and the time."""
    rows = []
    for i in planned.rows:
        datatype = _cell(df, i, "datatype")
        uid = _cell(df, i, "series_uid")
        for f in _files_for_row(df, i, _two_echoes(features or {}, uid), False,
                                _diffusion(features or {}, uid), _echoes(features or {}, uid)):
            if f.conditional or f.role == "sidecar" or f.path.endswith(("_channels.tsv",)):
                continue
            rows.append((f"{datatype}/{f.path.rsplit('/', 1)[-1]}", _iso_time(df, i)))
    rows.sort()
    return FilePreview("tsv", header=("filename", "acq_time"), rows=tuple(rows),
                       note="One row per recording of the session, its time from the scan.")


def _events_tsv(df, row, spec) -> FilePreview:
    labels = []
    maps = getattr(spec, "event_maps", None) or {}
    for _key, mapping in (maps.items() if isinstance(maps, dict) else []):
        for code, label in (mapping.items() if isinstance(mapping, dict) else []):
            labels.append((str(code), str(label)))
    rows = tuple((UNKNOWN, UNKNOWN, label, code, UNKNOWN) for code, label in labels[:6])
    note = ("One row per event or annotation in the recording, with the labels you gave "
            "event codes." if labels else "One row per event or annotation in the recording.")
    return FilePreview("tsv", header=("onset", "duration", "trial_type", "value", "sample"),
                       rows=rows, note=note)


def _seconds(ms) -> Optional[float]:
    try:
        return round(float(ms) / 1000.0, 6)
    except (TypeError, ValueError):
        return None


def _pe_direction(feats) -> Optional[str]:
    axis = getattr(feats, "pe_axis", "")
    if axis not in ("ROW", "COL"):
        return None
    letter = "i" if axis == "ROW" else "j"
    positive = getattr(feats, "pe_positive", None)
    if positive is None:
        return None
    return letter if positive else f"{letter}-"


def _links(df, row, options: PlanOptions, features) -> list[tuple[str, object, str]]:
    """IntendedFor / B0FieldIdentifier / B0FieldSource as the conversion writes them."""
    from ..fixups.intended_for import plan_links

    shims = {}
    for uid, feats in (features or {}).items():
        if getattr(feats, "shim", None):
            shims[uid] = feats.shim
    try:
        links, _why = plan_links(df, shims=shims)
    except Exception:  # noqa: BLE001
        return []
    # Number the fieldmaps' acquisitions per session, by their target lists.
    def key_of(i):
        return (_cell(df, i, "participant_id"), _cell(df, i, "session"))

    position = {r: n for n, r in enumerate(df.index)}
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for fm in sorted(links, key=lambda r: position.get(r, 0)):
        targets = tuple(sorted(links[fm]))
        if targets and targets not in groups[key_of(fm)]:
            groups[key_of(fm)].append(targets)

    def ident(fm) -> str:
        session = _cell(df, fm, "session").replace("ses-", "")
        targets = tuple(sorted(links[fm]))
        return f"fmap{re.sub(r'[^0-9A-Za-z]', '', session)}{groups[key_of(fm)].index(targets) + 1}"

    out: list[tuple[str, object, str]] = []
    if row in links and links[row]:
        subject = _cell(df, row, "participant_id")
        names = []
        for t in links[row]:
            datatype, name = _cell(df, t, "datatype"), _cell(df, t, "bids_name")
            session = _cell(df, t, "session")
            parts = ([session] if session else []) + [datatype, f"{name}.nii.gz"]
            rel = "/".join(parts)
            names.append(rel if options.intended_for == "relative" else f"bids::{subject}/{rel}")
        out.append(("IntendedFor", names, "settings"))
        if options.b0_field:
            out.append(("B0FieldIdentifier", ident(row), "settings"))
    elif options.b0_field:
        sources = [ident(fm) for fm, targets in links.items() if row in targets]
        sources = sorted(set(sources))
        if sources:
            out.append(("B0FieldSource", sources[0] if len(sources) == 1 else sources, "settings"))
    return out


def _mri_fields(df, row, feats) -> list[tuple[str, object, str]]:
    f: list[tuple[str, object, str]] = []

    def add(key, value, source="scan"):
        if value not in (None, "", [], ()):
            f.append((key, value, source))

    add("Modality", "MR")
    add("MagneticFieldStrength", getattr(feats, "field_strength", None))
    add("Manufacturer", getattr(feats, "manufacturer", ""))
    add("ManufacturersModelName", getattr(feats, "model", ""))
    f.append(("SoftwareVersions", UNKNOWN, "converter"))
    add("MRAcquisitionType", getattr(feats, "acquisition_type", ""))
    add("SeriesDescription", getattr(feats, "description", "") or _cell(df, row, "sequence"))
    add("ProtocolName", getattr(feats, "protocol", ""))
    scanning = [x for x, on in (("EP", getattr(feats, "is_epi", False)),
                                ("GR", getattr(feats, "is_gre", False)),
                                ("SE", getattr(feats, "is_se", False)),
                                ("IR", getattr(feats, "is_ir", False))) if on]
    add("ScanningSequence", "_".join(scanning))
    add("SequenceVariant", "_".join(getattr(feats, "sequence_variant", []) or []))
    add("ScanOptions", "_".join(getattr(feats, "scan_options", []) or []))
    add("SequenceName", getattr(feats, "sequence_name", ""))
    add("ImageType", list(getattr(feats, "image_type", []) or []))
    add("SliceThickness", getattr(feats, "slice_thickness", None))
    echoes = list(getattr(feats, "echo_times", []) or [])
    if len(echoes) >= 2 and _cell(df, row, "bids_name").endswith("phasediff"):
        add("EchoTime1", _seconds(min(echoes)))
        add("EchoTime2", _seconds(max(echoes)))
    elif echoes:
        add("EchoTime", _seconds(echoes[0]))
    add("RepetitionTime", getattr(feats, "repetition_time", None))
    add("InversionTime", getattr(feats, "inversion_time", None))
    add("FlipAngle", getattr(feats, "flip_angle", None))
    add("EchoTrainLength", getattr(feats, "echo_train_length", None))
    add("MultibandAccelerationFactor", getattr(feats, "multiband", None))
    add("ContrastBolusAgent", getattr(feats, "contrast_agent", ""))
    pe = _pe_direction(feats)
    f.append(("PhaseEncodingDirection", pe or UNKNOWN, "scan" if pe else "converter"))
    if getattr(feats, "is_epi", False):
        f.append(("EffectiveEchoSpacing", UNKNOWN, "converter"))
        f.append(("TotalReadoutTime", UNKNOWN, "converter"))
    if getattr(feats, "has_slice_timing", False):
        f.append(("SliceTiming", UNKNOWN, "converter"))
    f.append(("ConversionSoftware", "dcm2niix", "settings"))
    return f


def _eeg_fields(df, row, spec) -> list[tuple[str, object, str]]:
    datatype = _cell(df, row, "datatype")
    f: list[tuple[str, object, str]] = []
    task = _split_name(_cell(df, row, "bids_name")).get("task", "")
    if task:
        f.append(("TaskName", task, "settings"))
    sfreq = _cell(df, row, "sfreq")
    f.append(("SamplingFrequency", float(sfreq) if sfreq else UNKNOWN, "scan" if sfreq else "converter"))
    line = _cell(df, row, "line_freq")
    f.append(("PowerLineFrequency", float(line) if line else "n/a", "settings"))
    f.append(("SoftwareFilters", "n/a", "converter"))
    if datatype in ("eeg", "ieeg"):
        ref = _cell(df, row, "eeg_reference")
        f.append(("EEGReference", ref or UNKNOWN, "settings" if ref else "converter"))
        ground = _cell(df, row, "eeg_ground")
        if ground:
            f.append(("EEGGround", ground, "settings"))
        f.append(("EEGChannelCount", UNKNOWN, "converter"))
    if datatype == "meg":
        f.append(("MEGChannelCount", UNKNOWN, "converter"))
        f.append(("DewarPosition", UNKNOWN, "converter"))
    duration = _cell(df, row, "duration_sec")
    if duration:
        f.append(("RecordingDuration", round(float(duration), 3), "scan"))
    f.append(("RecordingType", "continuous", "converter"))
    manufacturer = _cell(df, row, "manufacturer_suggestion")
    if manufacturer:
        f.append(("Manufacturer", manufacturer, "scan"))
    return f


def _placeholders(datatype: str, suffix: str, have: set, levels: frozenset) -> list:
    """The TODO placeholders the metadata step writes into this kind of
    sidecar, by the engine's own rule: the schema's fields at the levels the
    fill scope covers, canonical names, a field it cannot mark left out."""
    from .. import schema
    from ..metadata.engine import canonical_field_name

    getters = (("required", schema.required_sidecar_fields),
               ("recommended", schema.recommended_sidecar_fields),
               ("optional", schema.optional_sidecar_fields))
    out: list[tuple[str, object, str]] = []
    seen = set(have)
    for level, get in getters:
        if level not in levels:
            continue
        try:
            infos = get(datatype, suffix)
        except (KeyError, ValueError, AttributeError):
            continue
        for info in infos:
            if level == "required" and getattr(info, "speculative", False):
                continue
            name = canonical_field_name(info.name)
            if name in seen:
                continue
            seen.add(name)
            value = _todo(info)
            if value is not None:
                out.append((name, value, "todo"))
    return out


def _sidecar(df, row, planned, options, features, spec, included) -> FilePreview:
    name = planned.path.rsplit("/", 1)[-1][:-len(".json")]
    datatype = _cell(df, row, "datatype")
    suffix = name.rsplit("_", 1)[-1]
    uid = _cell(df, row, "series_uid")
    feats = _features_of(features, uid)
    fields: list[tuple[str, object, str]] = []
    note = ""
    if suffix in ("channels", "events", "coordsystem", "electrodes"):
        if suffix == "events":
            fields = [("onset", {"Description": "Onset of the event", "Units": "s"}, "converter"),
                      ("trial_type", {"Description": "The event's label"}, "settings")]
        elif suffix == "coordsystem":
            fields = [("EEGCoordinateSystem" if datatype != "meg" else "MEGCoordinateSystem",
                       UNKNOWN, "converter"),
                      ("EEGCoordinateUnits" if datatype != "meg" else "MEGCoordinateUnits",
                       UNKNOWN, "converter")]
        return FilePreview("json", fields=tuple(fields))
    if suffix in ("physio", "physioevents"):
        if "recording-eye" in name:
            eye = "left" if "eye1" in name else "right"
            fields = [("PhysioType", "eyetrack", "settings"), ("RecordedEye", eye, "scan"),
                      ("SamplingFrequency", UNKNOWN, "converter"),
                      ("StartTime", UNKNOWN, "converter"),
                      ("Columns", ["timestamp", "x_coordinate", "y_coordinate", "pupil_size"]
                       if suffix == "physio" else ["onset", "duration", "trial_type", "message"],
                       "settings")]
        else:
            fields = [("SamplingFrequency", UNKNOWN, "converter"),
                      ("StartTime", UNKNOWN, "converter"),
                      ("Columns", UNKNOWN, "converter")]
            note = "One pair of files per signal (recording-<signal>) when they were sampled at different rates."
        return FilePreview("json", fields=tuple(fields), note=note)

    if datatype in ("eeg", "ieeg", "meg", "nirs"):
        fields = _eeg_fields(df, row, spec)
    else:
        if feats is not None:
            fields = _mri_fields(df, row, feats)
        else:
            fields = [("Modality", "MR", "scan"),
                      ("SeriesDescription", _cell(df, row, "sequence"), "scan")]
            note = ("The scan's acquisition details were not kept for this table; dcm2niix "
                    "reads them at conversion.")
        # From the name and the table, whatever the scan kept.
        if datatype == "func":
            task = _split_name(name).get("task", "")
            if task:
                fields.insert(0, ("TaskName", task, "settings"))
        if datatype in ("fmap", "func", "dwi", "perf"):
            fields.extend(_links(df, row, options, features))
    levels = _levels(options)
    if levels and datatype and suffix:
        fields.extend(_placeholders(datatype, suffix, {k for k, _v, _s in fields}, levels))
    if datatype == "pet" and spec is not None:
        note = note or "Dose and tracer fields come from the dataset metadata you set."
    return FilePreview("json", fields=tuple(fields), note=note)


__all__ = [
    "FilePreview",
    "OutputPlan",
    "PlanOptions",
    "PlannedFile",
    "SOURCES",
    "UNKNOWN",
    "approximate",
    "plan_output",
]
