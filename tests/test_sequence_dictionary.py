from bids_manager import dicom_inventory


def test_sbref_patterns_cover_old_refscan():
    for seq in ["type-ref", "reference", "refscan", "ref"]:
        assert dicom_inventory.guess_modality(seq) == "SBRef"
