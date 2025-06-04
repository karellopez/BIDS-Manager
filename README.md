# BIDS Manager

This repository provides a PyQt-based GUI and helper scripts to convert DICOM folders into BIDS datasets and edit their metadata.

## Installation

```bash
pip install git+https://github.com/karellopez/BIDS-Manager.git
```

After installation the following commands become available:

- `bids-manager` – main GUI combining conversion and editing tools
- `dicom-inventory` – generate `subject_summary.tsv` from a DICOM directory
- `build-heuristic` – create a HeuDiConv heuristic from the TSV
- `run-heudiconv` – run HeuDiConv using the generated heuristic
- `post-conv-renamer` – rename fieldmap files after conversion
- `bids-editor` – standalone metadata editor

All utilities provide `-h/--help` for details.

