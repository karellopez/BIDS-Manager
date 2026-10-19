"""What anonymising a dataset changes, as a policy with two presets.

Every choice follows the standard where it has one: dates are SHIFTED rather
than removed ("shifting is RECOMMENDED over removal"), into 1925 or earlier,
by the same number of days for every date of one participant, so intervals
survive; ages are capped at 89. Where the standard says nothing, the
Recommended preset keeps what analyses need (times of day, the site name)
and Strict keeps as little as still makes a valid dataset.

Pure data. ``bidsmgr.anonymise.plan`` reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OFF = "off"
RECOMMENDED = "recommended"
STRICT = "strict"


@dataclass(frozen=True)
class AnonymisePolicy:
    """One answer per kind of identifying content."""

    name: str = RECOMMENDED
    #: Shift every date of a participant by one number of whole weeks, into
    #: 1902-1925 ("shift"), or leave dates alone ("keep").
    dates: Literal["shift", "keep"] = "shift"
    #: The time of day: kept, or set to midnight (and the scanner's own time
    #: fields removed, the NIfTI header's "Time=" too).
    times: Literal["keep", "midnight"] = "keep"
    #: DICOM UIDs: replaced by a pseudonym that is the same for the same UID
    #: everywhere (they embed the scanner and the time on many systems).
    uids: Literal["pseudonymise", "keep"] = "pseudonymise"
    #: The institution: "address" removes its address only; "all" removes
    #: name, department and address.
    institution: Literal["keep", "address", "all"] = "address"
    #: Device serial numbers and station names: pseudonymised.
    devices: Literal["pseudonymise", "keep"] = "pseudonymise"
    #: Ages in participants.tsv: capped at 89 (the standard's rule), or also
    #: rounded down to five-year bins.
    ages: Literal["cap", "bins"] = "cap"
    #: The gzip header of every .nii.gz and .tsv.gz: its time stamp and the
    #: original file name, removed.
    gzip_headers: bool = True
    #: Photographs: their EXIF and text metadata removed, the image untouched.
    photos: bool = True

    def describe(self) -> list[str]:
        """What this policy does, one line each, for a dialog or a log."""
        out = []
        if self.dates == "shift":
            out.append("Dates shifted into 1902-1925, by one number of whole weeks per "
                       "participant, so intervals and weekdays are kept")
        if self.times == "midnight":
            out.append("Times of day set to midnight; the scanner's own time fields "
                       "removed")
        out.append("Names, IDs, birth dates and the people who ran the session removed, "
                   "in sidecars, participants.tsv and recording headers")
        if self.uids == "pseudonymise":
            out.append("DICOM UIDs replaced by stable pseudonyms")
        if self.institution == "address":
            out.append("The institution's address written as n/a (its name kept)")
        elif self.institution == "all":
            out.append("The institution's name, department and address written as n/a")
        if self.devices == "pseudonymise":
            out.append("Device serial numbers and station names pseudonymised")
        out.append("Ages over 89 written as 89" if self.ages == "cap"
                   else "Ages rounded down to five-year bins, and capped at 89")
        if self.gzip_headers:
            out.append("The time stamp and original name in every gzip header removed")
        if self.photos:
            out.append("Photographs' EXIF and text metadata removed")
        return out


PRESETS: dict[str, AnonymisePolicy] = {
    RECOMMENDED: AnonymisePolicy(),
    STRICT: AnonymisePolicy(name=STRICT, times="midnight", institution="all", ages="bins"),
}

PRESET_LABELS: dict[str, str] = {
    OFF: "Off",
    RECOMMENDED: "Recommended",
    STRICT: "Strict",
}


def policy_for(name: str) -> AnonymisePolicy | None:
    """The preset called ``name``; ``None`` for off."""
    if name == OFF or not name:
        return None
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown anonymisation preset {name!r}; "
                         f"choose from {', '.join(PRESET_LABELS)}") from None


__all__ = ["AnonymisePolicy", "OFF", "PRESETS", "PRESET_LABELS", "RECOMMENDED", "STRICT",
           "policy_for"]
