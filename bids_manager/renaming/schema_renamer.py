from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# ----------------------------- Utilities -----------------------------

_BIDS_EXTS = (".nii.gz", ".nii", ".json", ".bval", ".bvec", ".tsv")

_SANITIZE_TOKEN = re.compile(r"[^a-zA-Z0-9]+")
_TASK_TOKEN = re.compile(r"(?:^|[_-])task-([a-zA-Z0-9]+)", re.IGNORECASE)


def _sanitize_token(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    return _SANITIZE_TOKEN.sub("", x).strip()


def _guess_task_from_text(*candidates: Optional[str]) -> Optional[str]:
    for c in candidates:
        if not c:
            continue
        m = _TASK_TOKEN.search(c)
        if m:
            return _sanitize_token(m.group(1))
    hints = ("rest", "resting", "movie", "nback", "flanker", "stroop", "motor", "checker", "checkerboard")
    for c in candidates:
        if not c:
            continue
        low = c.lower()
        for h in hints:
            if h in low:
                return _sanitize_token(h)
    return None


def _resolve_ext(name: str) -> str:
    for ext in _BIDS_EXTS:
        if name.endswith(ext):
            return ext
    return Path(name).suffix


def _replace_stem_keep_ext(src: Path, new_basename: str) -> Path:
    ext = _resolve_ext(src.name)
    return src.with_name(f"{new_basename}{ext}")


def _iter_schema_files(schema_dir: Path) -> Iterable[Path]:
    for p in schema_dir.rglob("*"):
        if p.suffix.lower() in (".json", ".yaml", ".yml") and p.is_file():
            yield p


# --------------------------- Schema parsing ---------------------------

@dataclass
class SchemaInfo:
    suffix_requirements: Dict[str, List[str]]
    suffix_to_datatypes: Dict[str, List[str]]


def _load_json_or_yaml(p: Path) -> Optional[dict]:
    try:
        if p.suffix.lower() == ".json":
            return json.loads(p.read_text(encoding="utf-8"))
        else:
            if yaml is None:
                return None
            return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _harvest_suffix_rules(obj: Union[dict, list], current_datatype: Optional[str], out_req: Dict[str, set],
                          out_dt: Dict[str, set]) -> None:
    if isinstance(obj, dict):
        if "datatype" in obj and isinstance(obj["datatype"], str):
            current_datatype = obj["datatype"]

        suffix = obj.get("suffix")
        if isinstance(suffix, str):
            sfx = suffix.strip()
            required = set()
            for key in ("required", "required_entities", "entities_required"):
                v = obj.get(key)
                if isinstance(v, list):
                    for e in v:
                        if isinstance(e, str):
                            required.add(e.strip("<>"))
                        elif isinstance(e, dict) and "name" in e:
                            required.add(str(e["name"]).strip("<>"))
            ents = obj.get("entities")
            if isinstance(ents, list):
                for e in ents:
                    if isinstance(e, dict) and e.get("required") is True and "name" in e:
                        required.add(str(e["name"]).strip("<>"))

            if required:
                out_req.setdefault(sfx, set()).update(required)

            if current_datatype:
                out_dt.setdefault(sfx, set()).add(current_datatype)

        for v in obj.values():
            _harvest_suffix_rules(v, current_datatype, out_req, out_dt)

    elif isinstance(obj, list):
        for it in obj:
            _harvest_suffix_rules(it, current_datatype, out_req, out_dt)


def load_bids_schema(schema_dir: Union[str, Path]) -> SchemaInfo:
    schema_dir = Path(schema_dir)
    suffix_requirements: Dict[str, set] = {}
    suffix_to_datatypes: Dict[str, set] = {}

    for p in _iter_schema_files(schema_dir):
        data = _load_json_or_yaml(p)
        if not isinstance(data, (dict, list)):
            continue
        _harvest_suffix_rules(data, current_datatype=None,
                              out_req=suffix_requirements, out_dt=suffix_to_datatypes)

    fallback_dt = {
        "T1w": "anat", "T2w": "anat", "FLAIR": "anat", "T2star": "anat", "PD": "anat",
        "bold": "func", "sbref": "func",
        "dwi": "dwi",
        "phasediff": "fmap", "fieldmap": "fmap", "magnitude1": "fmap", "magnitude2": "fmap", "epi": "fmap",
    }
    for sfx, dt in fallback_dt.items():
        suffix_to_datatypes.setdefault(sfx, set()).add(dt)
    for sfx in set(suffix_to_datatypes.keys()) | set(suffix_requirements.keys()):
        suffix_requirements.setdefault(sfx, set()).add("subject")

    return SchemaInfo(
        suffix_requirements={k: sorted(v) for k, v in suffix_requirements.items()},
        suffix_to_datatypes={k: sorted(v) for k, v in suffix_to_datatypes.items()},
    )


# --------------------------- Core Proposer ----------------------------

@dataclass
class SeriesInfo:
    subject: str
    session: Optional[str]
    modality: str
    sequence: str
    rep: Optional[int]
    extra: Dict[str, str]


def _normalize_suffix(modality: str) -> str:
    m = modality.strip()
    alias = {"SBRef": "sbref", "SBREF": "sbref", "T2*": "T2star", "t2star": "T2star"}
    return alias.get(m, m)


def _choose_datatype(suffix: str, schema: SchemaInfo) -> str:
    dts = schema.suffix_to_datatypes.get(suffix)
    if dts:
        pref = ("anat", "func", "dwi", "fmap", "perf", "pet", "meg", "eeg", "ieeg")
        for p in pref:
            if p in dts:
                return p
        return dts[0]
    return {
        "T1w": "anat", "T2w": "anat", "FLAIR": "anat", "T2star": "anat", "PD": "anat",
        "bold": "func", "sbref": "func", "dwi": "dwi",
        "phasediff": "fmap", "fieldmap": "fmap", "magnitude1": "fmap", "magnitude2": "fmap", "epi": "fmap",
    }.get(suffix, "misc")


def propose_bids_basename(series: SeriesInfo, schema: SchemaInfo) -> Tuple[str, str]:
    suffix = _normalize_suffix(series.modality)
    datatype = _choose_datatype(suffix, schema)
    required = set(schema.suffix_requirements.get(suffix, []))

    parts: List[str] = []
    sub = _sanitize_token(series.subject)
    if not sub:
        raise ValueError("SeriesInfo.subject is required and must be alphanumeric")
    parts.append(f"sub-{sub}")

    ses = _sanitize_token(series.session or "")
    if ses:
        parts.append(f"ses-{ses}")

    if "task" in required or suffix in ("bold", "sbref"):
        task = series.extra.get("task") if series.extra else None
        task = _sanitize_token(task) or _guess_task_from_text(series.sequence)
        parts.append(f"task-{task or 'unknown'}")

    acq = series.extra.get("acq") if series.extra else None
    if acq:
        acq = _sanitize_token(acq)
        if acq:
            parts.append(f"acq-{acq}")

    echo = series.extra.get("echo") if series.extra else None
    if echo:
        echo = _sanitize_token(str(echo))
        if echo:
            parts.append(f"echo-{echo}")

    direction = series.extra.get("dir") if series.extra else None
    if direction:
        direction = _sanitize_token(direction)
        if direction:
            parts.append(f"dir-{direction}")

    run = None
    if series.extra and "run" in series.extra:
        run = series.extra["run"]
    elif series.rep and int(series.rep) > 1:
        run = str(series.rep)
    if run:
        run_s = _sanitize_token(str(run))
        if run_s and run_s != "1":
            parts.append(f"run-{run_s}")

    parts.append(suffix)
    return datatype, "_".join(parts)


# -------------------------- Post-conv renaming ------------------------

def _glob_candidates(dt_dir: Path, subject: str, original_seq: str) -> List[Path]:
    seq_clean = _SANITIZE_TOKEN.sub("", original_seq or "").lower()
    seq_wc = _SANITIZE_TOKEN.sub("*", (original_seq or "").lower())
    globs = [
        f"sub-*{seq_clean}*.*",
        f"sub-*{seq_wc}*.*",
        f"sub-*{seq_clean}*.nii.gz",
        f"sub-*{seq_wc}*.nii.gz",
        f"sub-*{seq_clean}*.json",
        f"sub-*{seq_wc}*.json",
    ]
    out = []
    for g in globs:
        out.extend([p for p in dt_dir.glob(g) if p.is_file()])
    seen = set()
    unique = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def _rename_file_set(old: Path, new_basename: str, rename_map: Dict[Path, Path]) -> None:
    newp = _replace_stem_keep_ext(old, new_basename)
    if newp == old:
        return
    newp.parent.mkdir(parents=True, exist_ok=True)
    rename_map[old] = newp


def _normalize_fieldmaps(dt_dir: Path, rename_map: Dict[Path, Path]) -> None:
    for p in dt_dir.glob("*_echo-1.*"):
        newb = p.name.replace("_echo-1", "_magnitude1")
        rename_map[p] = p.with_name(newb)
    for p in dt_dir.glob("*_echo-2.*"):
        newb = p.name.replace("_echo-2", "_magnitude2")
        rename_map[p] = p.with_name(newb)
    for p in dt_dir.glob("*_fmap.*"):
        newb = p.name.replace("_fmap", "_phasediff")
        rename_map[p] = p.with_name(newb)


def _move_dwi_derivatives(bids_root: Path, pipeline_name: str, rename_map: Dict[Path, Path]) -> None:
    """
    Move vendor-derived DWI maps from raw dwi/ to derivatives/<pipeline>/dwi/
    and rename as sub-XXX_desc-<MAP>_dwi.<ext>.
    Maps handled: ADC, FA, TRACEW, ColFA
    """
    for sub_dir in bids_root.glob("sub-*"):
        dwi_dir = sub_dir / "dwi"
        if not dwi_dir.exists():
            continue
        # detect maps on disk
        for p in dwi_dir.glob("*_*"):
            stem = p.name
            # skip non-files and .bval/.bvec of raw runs
            if not p.is_file():
                continue
            if stem.endswith(".bval") or stem.endswith(".bvec"):
                continue

            # map suffix detection
            for tag in ("_ADC", "_FA", "_TRACEW", "_ColFA"):
                if tag in stem:
                    desc = tag[1:]  # remove leading underscore
                    # new location under derivatives
                    new_dir = bids_root / "derivatives" / pipeline_name / sub_dir.name / "dwi"
                    new_dir.mkdir(parents=True, exist_ok=True)
                    # build new basename: keep leading sub-XXX[_ses-YYY] if present, then desc-<MAP>_dwi
                    # try to extract sub- and ses- tokens
                    tokens = [t for t in stem.split("_") if t.startswith(("sub-", "ses-"))]
                    prefix = "_".join(tokens) if tokens else sub_dir.name
                    new_basename = f"{prefix}_desc-{desc}_dwi"
                    newp = _replace_stem_keep_ext(p, new_basename)
                    rename_map[p] = new_dir / newp.name
                    break


def build_preview_names(
    inventory_rows: Iterable[SeriesInfo], schema: SchemaInfo
) -> List[Tuple[SeriesInfo, str, str]]:
    out = []
    for s in inventory_rows:
        dt, base = propose_bids_basename(s, schema)
        out.append((s, dt, base))
    return out


def apply_post_conversion_rename(
    bids_root: Union[str, Path],
    proposals: Iterable[Tuple[SeriesInfo, str, str]],
    also_normalize_fieldmaps: bool = True,
    handle_dwi_derivatives: bool = True,
    derivatives_pipeline_name: str = "dcm2niix",
) -> Dict[Path, Path]:
    bids_root = Path(bids_root)
    rename_map: Dict[Path, Path] = {}

    # main renaming based on proposals
    for series, datatype, new_base in proposals:
        dt_dir = bids_root / f"sub-{_sanitize_token(series.subject)}"
        if series.session:
            dt_dir = dt_dir / f"ses-{_sanitize_token(series.session)}"
        dt_dir = dt_dir / datatype
        if not dt_dir.exists():
            continue
        candidates = _glob_candidates(dt_dir, series.subject, series.sequence)
        for p in candidates:
            if not any(p.name.endswith(ext) for ext in _BIDS_EXTS):
                continue
            _rename_file_set(p, new_base, rename_map)

    # fieldmaps normalization
    if also_normalize_fieldmaps:
        top_fmap = bids_root / "fmap"
        if top_fmap.exists():
            _normalize_fieldmaps(top_fmap, rename_map)
        for fmap_dir in bids_root.glob("sub-*/fmap"):
            _normalize_fieldmaps(fmap_dir, rename_map)
        for fmap_dir in bids_root.glob("sub-*/ses-*/fmap"):
            _normalize_fieldmaps(fmap_dir, rename_map)

    # DWI derivative maps → derivatives/...
    if handle_dwi_derivatives:
        _move_dwi_derivatives(bids_root, derivatives_pipeline_name, rename_map)

    # Execute rename ops
    for old, new in sorted(rename_map.items(), key=lambda kv: len(str(kv[0])), reverse=True):
        if new.exists():
            if old.resolve() == new.resolve():
                continue
            stem, ext = new.stem, new.suffix
            if new.name.endswith(".nii.gz"):
                stem = new.name[:-7]
                ext = ".nii.gz"
            k = 2
            cand = new
            while cand.exists():
                cand = new.with_name(f"{stem}__{k}{ext}")
                k += 1
            new = cand
            rename_map[old] = new
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))

    return rename_map
