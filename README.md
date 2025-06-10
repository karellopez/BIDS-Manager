# BIDS Manager

This repository provides a PyQt-based GUI and helper scripts to convert DICOM folders into BIDS datasets and edit their metadata.

## Installation

```bash
pip install git+https://github.com/karellopez/BIDS-Manager.git
```

The package declares all dependencies including `heudiconv`, so installation
pulls everything required to run the GUI and helper scripts.

After installation the following commands become available:

- `bids-manager` – main GUI combining conversion and editing tools
- `dicom-inventory` – generate `subject_summary.tsv` from a DICOM directory
- `build-heuristic` – create a HeuDiConv heuristic from the TSV
- `run-heudiconv` – run HeuDiConv using the generated heuristic
- `post-conv-renamer` – rename fieldmap files after conversion
- `bids-editor` – standalone metadata editor

All utilities provide `-h/--help` for details.

### Recent updates

- The TSV produced by `dicom-inventory` can now be loaded directly in the GUI and
  its file name customised before generation.
- The Batch Rename tool previews changes and allows restricting the scope to
  specific subjects.
- `run-heudiconv` now keeps a copy of `subject_summary.tsv` under `.bids_manager`
  and generates a clean `participants.tsv` using demographics from that file.
- Dataset statistics in the Edit tab are now computed using `ancpbids` for
  BIDS Schema compliance.


