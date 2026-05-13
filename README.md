# BIDS Manager

**Schema-driven BIDS converter, curator, and editor.** PyQt6 GUI + CLI for
turning **DICOM**, **EEG**, and **MEG** trees into BIDS-compliant datasets,
with a post-conversion Editor for sidecar / TSV / NIfTI inspection and
metadata fixes.

📜 [documentation](https://ancplaboldenburg.github.io/bids_manager_documentation/) 📜

---

## What's new in v1.0.0

v1.0.0 is a **complete re-imagination** of BIDS-Manager (0.x). The old
HeuDiConv / dcm2bids two-engine pipeline + 9.2k-LOC single-file GUI has
been replaced by:

* A **schema-driven engine** — every layer (classification, naming,
  GUI forms, validation, sidecar generation) reads from the same
  machine-readable BIDS schema via
  [`bidsschematools`](https://github.com/bids-standard/bids-specification).
  Naming logic is no longer scattered across the heuristic, the config
  builder, and the renamer.
* **`dcm2niix` invoked directly** as the default MRI backend — no
  HeuDiConv / dcm2bids wrapper between us and the converter.
* A **two-tab PyQt6 GUI** — *Converter* (scan + plan + run) and
  *Editor* (post-conversion BIDS tree + sidecar form + TSV editor +
  NIfTI viewer + validator).
* A **NIfTI viewer** in the Editor with single-pane + tri-view
  (sagittal / coronal / axial) sharing one crosshair, click-and-drag
  scrubbing, and a 4-D time-series Graph with neighbour-scope grid.
  All NIfTI loads run on a `QThread` so the GUI doesn't freeze on
  large BOLDs.
* **Event-sourced project bundles** (`.bidsmgr/` directories) recording
  every user action for full undo + provenance.
* **Modular, testable architecture** — 16 sub-packages, no `Pipeline`
  god-object, no Qt outside `bidsmgr.gui`. ~730 unit/GUI tests + 49
  real-data tests passing.

The PyPI distribution name is **`bids-manager`** (unchanged from 0.x).
The import name is **`bidsmgr`** (new) — same pattern as
`pip install scikit-learn` → `import sklearn`.

If you're upgrading from v0.2.5: the old `from bids_manager import …`
imports and the old CLI entry points (`dicom-inventory`,
`build-heuristic`, `run-heudiconv`, `run-dcm2bids`, `post-conv-renamer`,
…) are gone. The replacement surface is documented below. The v0.2.5
codebase is preserved in git history.

---

## Install

```bash
pip install bids-manager
```

Requires Python ≥ 3.10. Installs the `bidsmgr` Python package and
seven console scripts (one GUI + five CLI verbs, see below).

---

## Use

### GUI

```bash
bidsmgr                  # launch the GUI
bidsmgr --theme dark     # force a theme on launch
bidsmgr --project PATH.bidsmgr   # open a saved project
```

The GUI has two tabs:

* **Converter** — point at a raw DICOM / EEG / MEG tree, review the
  schema-classified inventory in the inspection table, edit subject /
  session / sequence assignments via the Properties panel, run the
  conversion through `dcm2niix` (MRI) / `mne-bids` (EEG/MEG) /
  `bidsphysio` (Siemens physio). Per-subject staging with atomic
  commit; errors land under `<bids_root>/.bidsmgr/errors/`.
* **Editor** — open any BIDS dataset, browse it as a tree, edit JSON
  sidecars in a schema-aware form (required/recommended/optional
  fields colour-coded), edit TSVs in a table, view NIfTI volumes
  (2-D slice, tri-view, 4-D time-series graph), and run dataset /
  folder / file validation (layer 1 always; the official
  `bidsschematools.validator.validate_bids` as layer 2 via the Strict
  toggle).

### CLI

Five verbs covering the whole pipeline:

```
bidsmgr-scan      <raw_root>     <inv.tsv>      [--dataset NAME] [--line-freq 50|60] [--montage NAME] [-j N] [--probe-convert]
bidsmgr-rebuild   <inv.tsv>                     [--from {entities,columns}] [--dry-run]
bidsmgr-convert   <inv.tsv>      <bids_parent>  [--dataset NAME] [-j N] [--overwrite] [--dry-run]
bidsmgr-metadata  <bids_parent>                 [--inventory-tsv …] [--fill-todos] [--name …]
bidsmgr-validate  <bids_parent>                 [--strict] [--strict-warn] [--html]
```

Each verb has independent CLI dispatch under `bidsmgr.cli.<verb>:main`
and reads/writes files on disk, so any stage can be run standalone.

---

## Project layout

```
BIDS-Manager/                      ← repo root
├── pyproject.toml                  PEP 621, name = "bids-manager"
├── README.md                       ← you are here
├── LICENSE
├── CLAUDE.md                       project map for Claude Code
├── docs/                           architecture + planning documents
│   ├── architecture.md
│   ├── super_plan.md
│   ├── improvement_plan.md
│   ├── gui_mockups.html
│   └── inspector_proto/            standalone PyQt6 GUI prototype
├── miscellaneous/images/           non-package image assets
├── external/                       vendored Python embed (Windows installer)
├── Installers/                     packaged Windows installer
├── bidsmgr/                        ← the importable Python package
│   ├── __init__.py                 __version__ = "1.0.0"
│   ├── main.py                     GUI entry
│   ├── schema/                     bidsschematools wrapper (keystone)
│   ├── inventory/                  per-modality scanners
│   ├── classifier/                 chained classifiers
│   ├── planner/                    EntityPlan + edits
│   ├── converter/                  pluggable backends
│   │   └── backends/
│   ├── metadata/                   post-conv schema engine
│   ├── fixups/                     fmap, IntendedFor, scans.tsv
│   ├── project/                    event-sourced .bidsmgr files
│   ├── editor/                     post-conv editor logic (no Qt)
│   ├── gui/                        ← THE ONLY Qt subtree
│   │   ├── theme.qss
│   │   ├── theme_manager.py
│   │   ├── widgets/  delegates/  models/
│   ├── workers/                    QThread bridges
│   ├── cli/                        CLI verbs
│   └── util/                       cross-OS path safety, Qt platform helpers
└── tests/
    ├── unit/  integration/  real_data/  gui/  fixtures/
```

**Architectural prevention guards** (see `docs/architecture.md`):

1. `schema/` is the keystone — everything imports from it; it imports nothing.
2. `gui/` is the only Qt-coupled subtree; `workers/` also imports Qt
   (the QThread bridge); nothing else does.
3. **No `Pipeline` orchestrator.** Orchestration is explicit code in
   `cli/<verb>.py` and `gui/<panel>.py`.
4. Pure-data types only (Pydantic / dataclass; no I/O methods).
5. Functions over classes where possible.

---

## Develop

```bash
pip install -e ".[dev]"
python -c "import bidsmgr; print(bidsmgr.__version__)"
pytest
```

GUI tests run headless under `QT_QPA_PLATFORM=offscreen` via
`pytest-qt`. Real-data tests are gated on env vars
`BIDS_MANAGER_REAL_{MRI,EEG,MEG}_DATA=1` and datasets at
`/Users/karelo/Development/datasets/BIDS_Manager/raw_data/`.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

Authored by **Karel López Vilaret** and **Jochem Rieger**, ANCP Lab,
Carl von Ossietzky Universität Oldenburg. See the *About* dialog in
the GUI for the full acknowledgements.
