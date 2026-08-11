"""What the conversion will answer by itself, per kind of file.

``derivable.py`` says WHICH fields a converter usually fills. That is enough to
keep the template from asking for them, and not enough to be reassuring: a form
that simply omits a required field looks like a form that forgot it, and a
per-row panel that shows the field empty looks like a dataset missing metadata.
Neither is true. dcm2niix has already read the flip angle out of the header.

This module answers the other half: what the value will BE. It reads the
sidecars a probe conversion produced and the header facts the scan read out of
each recording, and reports them per ``<datatype>/<suffix>`` so a form can show
"EchoTime 0.03, from the DICOM" instead of an empty box.

Two rules make it honest rather than decorative:

* A field is reported only when every probed file of that kind agrees. Where
  they differ the answer is :data:`~bidsmgr.recording_meta.VARIES`, which is
  exactly what that sentinel is for: it says the answer lives further down, per
  recording, not here.
* Nothing here is ever written to a sidecar. It is a preview of what the
  conversion does anyway, so writing it would at best duplicate the converter
  and at worst freeze one file's value onto a class of files.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import schema as schema_mod
from ..recording_meta import VARIES
from .derivable import CONVERTER_PRIVATE


def _key(datatype: str, suffix: str) -> str:
    return f"{datatype}/{suffix}"


def _agree(values: list[Any]) -> Any:
    """One answer for a group of files, or VARIES when they disagree."""
    first = values[0]
    for other in values[1:]:
        if other != first:
            return VARIES
    return first


def _collapse(
    seen: dict[str, dict[str, list[Any]]],
) -> dict[str, dict[str, Any]]:
    """Per kind, one value per field, dropping what the standard does not declare."""
    out: dict[str, dict[str, Any]] = {}
    for key, fields in seen.items():
        datatype, _, suffix = key.partition("/")
        answers = {
            name: _agree(values)
            for name, values in sorted(fields.items())
            if name not in CONVERTER_PRIVATE
            and schema_mod.field_applies(name, datatype, suffix)
        }
        if answers:
            out[key] = answers
    return out


# Inventory column -> the BIDS field the conversion writes it as. These are read
# from the recording by the scan itself, so an EEG or MEG dataset gets a preview
# without any probe conversion at all.
_RECORDING_COLUMNS: dict[str, str] = {
    "sfreq": "SamplingFrequency",
    "duration_sec": "RecordingDuration",
    "n_channels": "EEGChannelCount",
    "task": "TaskName",
}

# Which datatype takes which channel-count name. MEG counts its own channels
# separately, so the EEG name would be wrong there.
_CHANNEL_COUNT: dict[str, str] = {
    "eeg": "EEGChannelCount",
    "ieeg": "iEEGElectrodeCount",
    "meg": "MEGChannelCount",
}


def preview_from_inventory(df) -> dict[str, dict[str, Any]]:
    """What the scan already knows the conversion will write, per kind.

    Free: these are facts the scanner read out of each recording's header while
    building the inventory, so no conversion has to run to know them.
    """
    if df is None or not len(df):
        return {}

    seen: dict[str, dict[str, list[Any]]] = {}
    for _, row in df.iterrows():
        if str(row.get("include", "1")).strip() in ("0", "False", "false"):
            continue
        datatype = str(row.get("proposed_datatype", "") or "").strip()
        suffix = str(row.get("bids_guess_suffix", "") or "").strip()
        if not datatype or not suffix:
            continue

        fields = seen.setdefault(_key(datatype, suffix), {})
        for column, name in _RECORDING_COLUMNS.items():
            if name == "EEGChannelCount":
                name = _CHANNEL_COUNT.get(datatype, "")
                if not name:
                    continue
            raw = row.get(column)
            text = "" if raw is None else str(raw).strip()
            if not text or text.lower() in ("nan", "none"):
                continue
            fields.setdefault(name, []).append(_as_number(text))
    return _collapse(seen)


def _as_number(text: str) -> Any:
    """A count is an int and a rate is a float; anything else stays text."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def preview_from_probe(df, probe_stats: Optional[dict] = None) -> dict[str, dict[str, Any]]:
    """What a probe conversion actually produced, per kind.

    The join runs through the inventory rather than the scanned rows: a row
    knows its series UID but not its datatype, which the classifier decides
    afterwards and writes into the table.
    """
    if not probe_stats or df is None or not len(df):
        return {}

    kind_of: dict[str, tuple[str, str]] = {}
    for _, row in df.iterrows():
        uid = str(row.get("series_uid", "") or "").strip()
        datatype = str(row.get("proposed_datatype", "") or "").strip()
        if not datatype:
            datatype = str(row.get("bids_guess_datatype", "") or "").strip()
        suffix = str(row.get("bids_guess_suffix", "") or "").strip()
        if uid and datatype and suffix:
            kind_of[uid] = (datatype, suffix)

    seen: dict[str, dict[str, list[Any]]] = {}
    for uid, stats in probe_stats.items():
        kind = kind_of.get(str(uid))
        fields = getattr(stats, "sidecar_fields", None)
        if not kind or not fields:
            continue
        bucket = seen.setdefault(_key(*kind), {})
        for name, value in fields.items():
            bucket.setdefault(name, []).append(value)
    return _collapse(seen)


def merge_previews(*previews: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Combine previews from several sources; later ones win per field."""
    out: dict[str, dict[str, Any]] = {}
    for preview in previews:
        for key, fields in (preview or {}).items():
            out.setdefault(key, {}).update(fields)
    return out


__all__ = [
    "merge_previews",
    "preview_from_inventory",
    "preview_from_probe",
]
