from bids_manager.renaming.schema_renamer import compose_proposed_name

def test_physio_uses_tsv_extension():
    assert (
        compose_proposed_name("physio", "func", "sub-001_task-rest_physio")
        == "func/sub-001_task-rest_physio.tsv"
    )


def test_images_use_nifti_extension():
    assert (
        compose_proposed_name("bold", "func", "sub-001_task-rest_bold")
        == "func/sub-001_task-rest_bold.nii.gz"
    )
