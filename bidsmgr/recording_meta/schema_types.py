"""Allowed channel-type vocabularies, sourced from the live schema / MNE.

These back the validators on :class:`~bidsmgr.recording_meta.models.AuxChannelSpec`
so a user-authored auxiliary-channel mapping cannot smuggle in a misspelled or
non-standard channel type. We read the BIDS vocabulary straight from the bundled
``bidsschematools`` schema (so it tracks the BIDS version automatically) and the
MNE vocabulary from MNE itself where available, with a conservative fallback in
both cases so importing this module never hard-fails in a stripped environment.
"""

from __future__ import annotations

import functools

from .. import schema as schema_mod

# Conservative fallbacks, used only if the live source cannot be read. Kept
# small and obviously-correct; the live source is authoritative when present.
_BIDS_FALLBACK: frozenset[str] = frozenset(
    {
        "EEG", "ECG", "EOG", "EMG", "ECOG", "SEEG", "DBS", "MEGMAG",
        "MEGGRADAXIAL", "MEGGRADPLANAR", "MEGREFMAG", "MEGREFGRADAXIAL",
        "MEGREFGRADPLANAR", "MEGOTHER", "TRIG", "AUDIO", "PD", "RESP", "GSR",
        "TEMP", "PPG", "HEOG", "VEOG", "MISC", "OTHER", "REF",
    }
)

# MNE channel-type strings accepted by ``raw.set_channel_types``. Used only if
# MNE's own constant table cannot be read.
_MNE_FALLBACK: frozenset[str] = frozenset(
    {
        "bio", "chpi", "dbs", "dipole", "ecg", "ecog", "emg", "eog", "exci",
        "eyetrack", "fnirs", "gof", "gsr", "ias", "misc", "meg", "ref_meg",
        "resp", "seeg", "stim", "syst", "temperature", "mag", "grad",
    }
)


@functools.lru_cache(maxsize=1)
def bids_channel_types() -> frozenset[str]:
    """Allowed values for the ``type`` column of a BIDS ``channels.tsv``.

    Read from the schema's ``columns.type__channels`` enum so the set tracks
    whatever BIDS version the bundled schema pins.
    """
    try:
        column = schema_mod.get_schema().objects.columns["type__channels"]
        enum = column["enum"]
        values = frozenset(str(v) for v in enum)
        if values:
            return values
    except Exception:  # pragma: no cover - defensive only
        pass
    return _BIDS_FALLBACK


@functools.lru_cache(maxsize=1)
def mne_channel_types() -> frozenset[str]:
    """Channel-type strings MNE's ``set_channel_types`` accepts."""
    try:
        from mne.io.pick import get_channel_type_constants

        values = frozenset(str(k) for k in get_channel_type_constants().keys())
        if values:
            return values
    except Exception:  # pragma: no cover - older/newer MNE or absent MNE
        pass
    return _MNE_FALLBACK


__all__ = ["bids_channel_types", "mne_channel_types"]
