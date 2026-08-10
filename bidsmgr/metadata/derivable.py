"""Which sidecar fields the converters fill by themselves.

A template should ask only for what the data cannot answer. Whether a field can
be answered from the data is not something the BIDS schema knows: it depends on
what dcm2niix reads out of a DICOM header and what mne-bids reads out of a
recording. So it is MEASURED, not asserted.

The sets below were produced by converting the real four-modality test tree with
an empty scaffold, which is exactly "what you get having filled nothing in", and
keeping the fields the schema also declares. The asymmetry they show is the
whole reason the template matters: DICOM carries most of what BIDS wants for
MRI, so an anatomical scan arrives with 29 of its 76 declared fields already
answered, while an EEG recording arrives with 15 of 43 and a PET scan 23 of 80.

``tests/unit/test_derivable.py`` re-measures against a converted dataset when
one is available, so an mne-bids or dcm2niix release that starts filling a field
shows up as a failing test rather than as a question the user is still being
asked for no reason.

Two caveats worth stating plainly.

First, this is one vendor's answer. A Siemens DICOM fills fields a Philips one
may not. Being listed here only means "usually supplied", and a field that goes
unfilled for some other scanner still shows up as missing at validation, where
it can be answered per row or per sequence.

Second, some fields are IN the recording but nothing extracts them today:
``MISCChannelCount`` and the channel counts MEG leaves at zero, or the device
serial a FIF header carries. Those belong in neither list. They are a backlog of
derivations, and asking a user to count their own channels would be the wrong
fix.
"""

from __future__ import annotations

# Per datatype, the fields a conversion produces without anyone filling in a
# template. Keyed by datatype rather than by (datatype, suffix): a field the
# converter writes for one suffix it writes for the others too, and the schema
# already decides which of them the file may carry.
DERIVED_BY_CONVERTER: dict[str, frozenset[str]] = {
    "anat": frozenset({
        "BodyPart", "CoilCombinationMethod", "DeviceSerialNumber",
        "DwellTime", "EchoTime", "FlipAngle", "InstitutionAddress",
        "InstitutionName", "InstitutionalDepartmentName", "InversionTime",
        "MRAcquisitionType", "MagneticFieldStrength", "Manufacturer",
        "ManufacturersModelName", "MatrixCoilMode",
        "NonlinearGradientCorrection", "ParallelReductionFactorInPlane",
        "PartialFourier", "PulseSequenceDetails",
        "ReceiveCoilActiveElements", "ReceiveCoilName", "ScanOptions",
        "ScanningSequence", "SequenceName", "SequenceVariant",
        "SoftwareVersions", "SpoilingState", "StationName", "TablePosition",
    }),
    "dwi": frozenset({
        "BodyPart", "CoilCombinationMethod", "DeviceSerialNumber",
        "DwellTime", "EchoTime", "EffectiveEchoSpacing", "FlipAngle",
        "InstitutionAddress", "InstitutionName",
        "InstitutionalDepartmentName", "MRAcquisitionType",
        "MagneticFieldStrength", "Manufacturer", "ManufacturersModelName",
        "MatrixCoilMode", "NonlinearGradientCorrection",
        "ParallelReductionFactorInPlane", "PartialFourier",
        "PhaseEncodingDirection", "PulseSequenceDetails",
        "ReceiveCoilActiveElements", "ReceiveCoilName", "ScanOptions",
        "ScanningSequence", "SequenceName", "SequenceVariant", "SliceTiming",
        "SoftwareVersions", "SpoilingState", "StationName", "TablePosition",
        "TotalReadoutTime",
    }),
    "fmap": frozenset({
        "BodyPart", "CoilCombinationMethod", "DeviceSerialNumber",
        "DwellTime", "EchoTime", "FlipAngle", "InstitutionAddress",
        "InstitutionName", "InstitutionalDepartmentName", "IntendedFor",
        "MRAcquisitionType", "MagneticFieldStrength", "Manufacturer",
        "ManufacturersModelName", "MatrixCoilMode",
        "NonlinearGradientCorrection", "PartialFourier",
        "PulseSequenceDetails", "ReceiveCoilActiveElements",
        "ReceiveCoilName", "ScanningSequence", "SequenceName",
        "SequenceVariant", "SliceTiming", "SoftwareVersions", "StationName",
        "TablePosition",
    }),
    "func": frozenset({
        "BodyPart", "CoilCombinationMethod", "Columns", "DeviceSerialNumber",
        "DwellTime", "EchoTime", "EffectiveEchoSpacing", "FlipAngle",
        "InstitutionAddress", "InstitutionName",
        "InstitutionalDepartmentName", "MRAcquisitionType",
        "MagneticFieldStrength", "Manufacturer", "ManufacturersModelName",
        "MatrixCoilMode", "MultibandAccelerationFactor",
        "NonlinearGradientCorrection", "ParallelReductionFactorInPlane",
        "PartialFourier", "PhaseEncodingDirection", "PulseSequenceDetails",
        "ReceiveCoilActiveElements", "ReceiveCoilName", "RepetitionTime",
        "SamplingFrequency", "ScanOptions", "ScanningSequence",
        "SequenceName", "SequenceVariant", "SliceTiming", "SoftwareVersions",
        "StartTime", "StationName", "TablePosition", "TaskName",
        "TotalReadoutTime",
    }),
    "eeg": frozenset({
        "ECGChannelCount", "EEGChannelCount", "EEGGround",
        "EEGPlacementScheme", "EEGReference", "EMGChannelCount",
        "EOGChannelCount", "Manufacturer", "PowerLineFrequency",
        "RecordingDuration", "RecordingType", "SamplingFrequency",
        "SoftwareFilters", "TaskName", "TriggerChannelCount",
    }),
    "meg": frozenset({
        "ContinuousHeadLocalization", "DewarPosition", "DigitizedHeadPoints",
        "DigitizedLandmarks", "ECGChannelCount", "EEGChannelCount",
        "EMGChannelCount", "EOGChannelCount", "HeadCoilFrequency",
        "MEGChannelCount", "MEGREFChannelCount", "Manufacturer",
        "MiscChannelCount", "PowerLineFrequency", "RecordingDuration",
        "RecordingType", "SamplingFrequency", "SoftwareFilters", "TaskName",
        "TriggerChannelCount",
    }),
    "pet": frozenset({
        "AttenuationCorrection", "BodyPart", "DecayCorrectionFactor",
        "DoseCalibrationFactor", "FrameDuration", "FrameTimesStart",
        "ImageDecayCorrected", "ImageDecayCorrectionTime",
        "InjectedRadioactivity", "InjectedRadioactivityUnits",
        "InjectedVolume", "InstitutionAddress", "InstitutionName",
        "InstitutionalDepartmentName", "Manufacturer",
        "ManufacturersModelName", "ReconFilterSize", "ReconMethodName",
        "ScanStart", "ScatterFraction", "TracerName", "TracerRadionuclide",
        "Units",
    }),
}

# Keys dcm2niix writes that BIDS does not define. They are not metadata a user
# should ever be asked for, and they are not the standard's either.
CONVERTER_PRIVATE: frozenset[str] = frozenset({
    "BidsGuess", "ConversionSoftware", "ConversionSoftwareVersion",
    "ImageOrientationText", "ProcedureStepDescription", "WipMemBlock",
})


def derived_fields(datatype: str) -> frozenset[str]:
    """Fields a conversion of this datatype usually supplies by itself."""
    return DERIVED_BY_CONVERTER.get(datatype, frozenset())


__all__ = ["CONVERTER_PRIVATE", "DERIVED_BY_CONVERTER", "derived_fields"]
