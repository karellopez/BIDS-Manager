import sys
import os
import subprocess
import json
import re
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout, QGridLayout,
    QTextEdit, QTreeView, QFileSystemModel, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QMessageBox, QAction, QSplitter, QDialog, QAbstractItemView,
    QMenuBar, QSizePolicy, QComboBox, QSlider)
from PyQt5.QtCore import Qt, QModelIndex, QTimer
from PyQt5.QtGui import QPalette, QColor, QFont, QImage, QPixmap
import logging  # debug logging


class _AutoUpdateLabel(QLabel):
    """QLabel that triggers a callback whenever it is resized."""

    def __init__(self, update_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_fn = update_fn

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if callable(self._update_fn):
            self._update_fn()

# Import the scan function directly to ensure TSV generation works. When the
# module is executed as a standalone script (``python gui.py``), ``__package__``
# will be ``None`` and relative imports fail.  In that case we prepend the
# package's parent directory to ``sys.path`` and fall back to an absolute
# import.
if __package__:
    from .dicom_inventory import scan_dicoms_long
else:  # running as a script
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bids_manager.dicom_inventory import scan_dicoms_long

# ---- basic logging config ----
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def _get_ext(path: Path) -> str:
    """Return file extension with special handling for .nii.gz."""
    name = path.name.lower()
    if name.endswith('.nii.gz'):
        return '.nii.gz'
    return path.suffix.lower()

class BIDSManager(QMainWindow):
    """
    Main GUI for BIDS Manager.
    Provides two tabs: Convert (DICOM→BIDS pipeline) and Edit (BIDS dataset explorer/editor).
    Supports Windows and Linux.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIDS Manager")
        self.resize(1400, 900)

        # Paths
        self.dicom_dir = ""         # Raw DICOM directory
        self.bids_out_dir = ""      # Output BIDS directory
        self.tsv_path = ""          # Path to subject_summary.tsv
        self.heuristic_dir = ""     # Directory with heuristics
        self.study_set = set()
        self.modb_rows = {}
        self.mod_rows = {}
        self.seq_rows = {}

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Tab widget
        self.tabs = QTabWidget()
        # Use larger font for tab labels
        font = QFont()
        font.setPointSize(10)
        self.tabs.setFont(font)
        main_layout.addWidget(self.tabs)

        # Initialize tabs
        self.initConvertTab()
        self.initEditTab()

    def initConvertTab(self):
        """Create the Convert tab with a cleaner layout."""
        self.convert_tab = QWidget()
        main_layout = QVBoxLayout(self.convert_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        cfg_group = QGroupBox("Configuration")
        cfg_layout = QGridLayout(cfg_group)

        dicom_label = QLabel("<b>DICOM Dir:</b>")
        self.dicom_dir_edit = QLineEdit()
        self.dicom_dir_edit.setReadOnly(True)
        dicom_browse = QPushButton("Browse…")
        dicom_browse.clicked.connect(self.selectDicomDir)
        cfg_layout.addWidget(dicom_label, 0, 0)
        cfg_layout.addWidget(self.dicom_dir_edit, 0, 1)
        cfg_layout.addWidget(dicom_browse, 0, 2)

        bids_label = QLabel("<b>BIDS Out Dir:</b>")
        self.bids_out_edit = QLineEdit()
        self.bids_out_edit.setReadOnly(True)
        bids_browse = QPushButton("Browse…")
        bids_browse.clicked.connect(self.selectBIDSOutDir)
        cfg_layout.addWidget(bids_label, 1, 0)
        cfg_layout.addWidget(self.bids_out_edit, 1, 1)
        cfg_layout.addWidget(bids_browse, 1, 2)

        tsvname_label = QLabel("<b>TSV Name:</b>")
        self.tsv_name_edit = QLineEdit("subject_summary.tsv")
        cfg_layout.addWidget(tsvname_label, 2, 0)
        cfg_layout.addWidget(self.tsv_name_edit, 2, 1, 1, 2)

        self.tsv_button = QPushButton("Generate TSV")
        self.tsv_button.clicked.connect(self.runInventory)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.tsv_button)
        btn_row.addStretch()
        cfg_layout.addLayout(btn_row, 3, 0, 1, 3)

        main_layout.addWidget(cfg_group)

        splitter = QSplitter()

        tsv_group = QGroupBox("TSV Viewer")
        tsv_layout = QVBoxLayout(tsv_group)
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(6)
        self.mapping_table.setHorizontalHeaderLabels([
            "Include", "Subject", "Session", "Sequence", "Modality", "BIDS Modality"
        ])
        hdr = self.mapping_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setStretchLastSection(True)
        self.mapping_table.verticalHeader().setVisible(False)
        tsv_layout.addWidget(self.mapping_table)
        self.tsv_load_button = QPushButton("Load TSV…")
        self.tsv_load_button.clicked.connect(self.selectAndLoadTSV)
        tsv_layout.addWidget(self.tsv_load_button)
        splitter.addWidget(tsv_group)

        modal_group = QGroupBox("Modalities")
        modal_layout = QVBoxLayout(modal_group)
        self.modal_tabs = QTabWidget()
        full_tab = QWidget()
        full_layout = QVBoxLayout(full_tab)
        self.full_tree = QTreeWidget()
        # Display only one column with the BIDS modality
        self.full_tree.setHeaderLabels(["BIDS Modality"])
        full_layout.addWidget(self.full_tree)
        self.modal_tabs.addTab(full_tab, "Full View")
        modal_layout.addWidget(self.modal_tabs)
        splitter.addWidget(modal_group)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_tree = QTreeWidget()
        self.preview_tree.setHeaderLabels(["BIDS Structure"])
        preview_layout.addWidget(self.preview_tree)
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.generatePreview)
        preview_layout.addWidget(self.preview_button)

        btn_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.runFullConversion)
        btn_row.addStretch()
        btn_row.addWidget(self.run_button)

        # Combine preview panel and run button so the splitter keeps the
        # original layout but allows resizing versus the log output.
        preview_container = QWidget()
        pv_lay = QVBoxLayout(preview_container)
        pv_lay.setContentsMargins(0, 0, 0, 0)
        pv_lay.setSpacing(6)
        pv_lay.addWidget(preview_group)
        pv_lay.addLayout(btn_row)

        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.document().setMaximumBlockCount(1000)
        log_layout.addWidget(self.log_text)

        main_layout.addWidget(splitter, 1)

        # Splitter to allow resizing between preview and log windows
        pv_split = QSplitter(Qt.Vertical)
        pv_split.addWidget(preview_container)
        pv_split.addWidget(log_group)
        pv_split.setStretchFactor(0, 1)
        pv_split.setStretchFactor(1, 1)
        main_layout.addWidget(pv_split, 1)

        self.tabs.addTab(self.convert_tab, "Convert")

    def generatePreview(self):
        logging.info("generatePreview → Building preview tree …")
        """Populate preview tree based on checked sequences."""
        self.preview_tree.clear()
        multi_study = len(self.study_set) > 1
        for i in range(self.mapping_table.rowCount()):
            include = (self.mapping_table.item(i, 0).checkState() == Qt.Checked)
            if include:
                subj = self.mapping_table.item(i, 1).text()
                study = self.mapping_table.item(i, 1).data(Qt.UserRole) or ""
                ses = self.mapping_table.item(i, 2).text()
                seq = self.mapping_table.item(i, 3).text()
                modb = self.mapping_table.item(i, 5).text()
                base = f"{subj}/{ses}/{modb}/{subj}_{ses}_{seq}"
                if multi_study:
                    base = f"{study}/" + base
                if modb == "fmap":
                    for suffix in ["magnitude1", "magnitude2", "phasediff"]:
                        fname = f"{base}_{suffix}.nii.gz"
                        self.preview_tree.addTopLevelItem(QTreeWidgetItem([fname]))
                else:
                    item = QTreeWidgetItem([f"{base}.nii.gz"])
                    self.preview_tree.addTopLevelItem(item)
        self.preview_tree.expandAll()

    # (Rest of code remains unchanged)

    def initEditTab(self):
        """
        Set up Edit tab to embed the full functionality of bids_editor_ancpbids.
        """
        self.edit_tab = QWidget()
        edit_layout = QVBoxLayout(self.edit_tab)
        edit_layout.setContentsMargins(10, 10, 10, 10)
        edit_layout.setSpacing(8)

        # Internal menu bar for Edit features
        menu = QMenuBar()
        menu.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        menu.setMaximumHeight(24)
        file_menu = menu.addMenu("File")
        open_act = QAction("Open BIDS…", self)
        open_act.triggered.connect(self.openBIDSForEdit)
        file_menu.addAction(open_act)
        tools_menu = menu.addMenu("Tools")
        rename_act = QAction("Batch Rename…", self)
        rename_act.triggered.connect(self.launchBatchRename)
        tools_menu.addAction(rename_act)
        edit_layout.addWidget(menu)

        # Splitter between left (tree & stats) and right (metadata)
        splitter = QSplitter()
        splitter.setHandleWidth(4)

        # Left panel: BIDSplorer and BIDStatistics
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_layout.addWidget(QLabel('<b>BIDSplorer</b>'))
        self.model = QFileSystemModel()
        self.model.setRootPath("")
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setEditTriggers(QAbstractItemView.EditKeyPressed | QAbstractItemView.SelectedClicked)
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(3, True)
        hdr = self.tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        hdr.setSectionResizeMode(2, QHeaderView.Interactive)
        left_layout.addWidget(self.tree)
        self.tree.clicked.connect(self.onTreeClicked)

        left_layout.addWidget(QLabel('<b>BIDStatistics</b>'))
        self.stats = QTreeWidget()
        self.stats.setHeaderLabels(["Metric", "Value"])
        self.stats.setAlternatingRowColors(True)
        s_hdr = self.stats.header()
        s_hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        s_hdr.setSectionResizeMode(1, QHeaderView.Interactive)
        left_layout.addWidget(self.stats)

        splitter.addWidget(left_panel)

        # Right panel: MetadataViewer (reused from original)
        self.viewer = MetadataViewer()
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 2)

        edit_layout.addWidget(splitter)
        self.tabs.addTab(self.edit_tab, "Edit")

    def selectDicomDir(self):
        """Select the raw DICOM input directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if directory:
            self.dicom_dir = directory
            self.dicom_dir_edit.setText(directory)

    def selectBIDSOutDir(self):
        """Select (or create) the BIDS output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select/Create BIDS Output Directory")
        if directory:
            self.bids_out_dir = directory
            self.bids_out_edit.setText(directory)

    def selectAndLoadTSV(self):
        """Choose an existing TSV and load it into the table."""
        path, _ = QFileDialog.getOpenFileName(self, "Select TSV", self.bids_out_dir or "", "TSV Files (*.tsv)")
        if path:
            self.tsv_path = path
            self.tsv_name_edit.setText(os.path.basename(path))
            self.loadMappingTable()

    def runInventory(self):
        logging.info("runInventory → Generating TSV …")
        """
        Scan DICOMs and generate subject_summary.tsv in the selected output directory.
        """
        if not self.dicom_dir or not os.path.isdir(self.dicom_dir):
            QMessageBox.warning(self, "Invalid DICOM Directory", "Please select a valid DICOM input directory.")
            return
        if not self.bids_out_dir:
            QMessageBox.warning(self, "No BIDS Output Directory", "Please select a BIDS output directory.")
            return

        os.makedirs(self.bids_out_dir, exist_ok=True)

        name = self.tsv_name_edit.text().strip() or "subject_summary.tsv"
        self.tsv_path = os.path.join(self.bids_out_dir, name)

        # Generate TSV via dicom_inventory
        try:
            scan_dicoms_long(self.dicom_dir, self.tsv_path)
            self.log_text.append(f"TSV generated at {self.tsv_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"dicom_inventory failed: {e}")
            return

        self.loadMappingTable()

    def loadMappingTable(self):
        logging.info("loadMappingTable → Loading TSV into table …")
        """
        Load the generated TSV into the mapping_table for user editing.
        Columns: include, subject, session, sequence, modality, modality_bids
        """
        if not self.tsv_path or not os.path.isfile(self.tsv_path):
            return
        df = pd.read_csv(self.tsv_path, sep="\t")

        self.study_set.clear()
        self.modb_rows.clear()
        self.mod_rows.clear()
        self.seq_rows.clear()
        self.row_info = []

        # Populate table rows
        self.mapping_table.setRowCount(0)
        def _clean(val):
            """Return string representation of val or empty string for NaN."""
            return "" if pd.isna(val) else str(val)

        for _, row in df.iterrows():
            r = self.mapping_table.rowCount()
            self.mapping_table.insertRow(r)
            # Include: checkbox
            include_item = QTableWidgetItem()
            include_item.setFlags(include_item.flags() | Qt.ItemIsUserCheckable)
            include_item.setCheckState(Qt.Checked if row.get('include', 1) == 1 else Qt.Unchecked)
            self.mapping_table.setItem(r, 0, include_item)
            # Subject (non-editable)
            subj_item = QTableWidgetItem(_clean(row.get('BIDS_name')))
            subj_item.setFlags(subj_item.flags() & ~Qt.ItemIsEditable)
            subj_item.setData(Qt.UserRole, _clean(row.get('StudyDescription')))
            self.study_set.add(_clean(row.get('StudyDescription')))
            self.mapping_table.setItem(r, 1, subj_item)
            # Session (non-editable)
            ses_item = QTableWidgetItem(_clean(row.get('session')))
            ses_item.setFlags(ses_item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(r, 2, ses_item)
            # Sequence (editable)
            seq_item = QTableWidgetItem(_clean(row.get('sequence')))
            seq_item.setFlags(seq_item.flags() | Qt.ItemIsEditable)
            self.mapping_table.setItem(r, 3, seq_item)
            # Modality (non-editable)
            mod_item = QTableWidgetItem(_clean(row.get('modality')))
            mod_item.setFlags(mod_item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(r, 4, mod_item)
            # BIDS Modality (editable)
            modb_item = QTableWidgetItem(_clean(row.get('modality_bids')))
            modb_item.setFlags(modb_item.flags() | Qt.ItemIsEditable)
            self.mapping_table.setItem(r, 5, modb_item)

            self.row_info.append({
                'modb': _clean(row.get('modality_bids')),
                'mod': _clean(row.get('modality')),
                'seq': _clean(row.get('sequence')),
            })
        self.log_text.append("Loaded TSV into mapping table.")

        # Build modality/sequence lookup for tree interactions
        for idx, info in enumerate(self.row_info):
            self.modb_rows.setdefault(info['modb'], []).append(idx)
            self.mod_rows.setdefault((info['modb'], info['mod']), []).append(idx)
            self.seq_rows.setdefault((info['modb'], info['mod'], info['seq']), []).append(idx)

        self.populateModalitiesTree()


    def populateModalitiesTree(self):
        """Build modalities tree with checkboxes synced to the table."""
        self.full_tree.blockSignals(True)
        self.full_tree.clear()
        # build nested mapping: BIDS modality → non‑BIDS modality → sequences
        modb_map = {}
        for info in self.row_info:
            modb_map.setdefault(info['modb'], {}).setdefault(info['mod'], set()).add(info['seq'])

        for modb, mod_map in sorted(modb_map.items()):
            modb_item = QTreeWidgetItem([modb])
            modb_item.setFlags(modb_item.flags() | Qt.ItemIsUserCheckable)
            rows = self.modb_rows.get(modb, [])
            states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
            if states and all(states):
                modb_item.setCheckState(0, Qt.Checked)
            elif states and any(states):
                modb_item.setCheckState(0, Qt.PartiallyChecked)
            else:
                modb_item.setCheckState(0, Qt.Unchecked)
            modb_item.setData(0, Qt.UserRole, ('modb', modb))

            for mod, seqs in sorted(mod_map.items()):
                mod_item = QTreeWidgetItem([mod])
                mod_item.setFlags(mod_item.flags() | Qt.ItemIsUserCheckable)
                rows = self.mod_rows.get((modb, mod), [])
                states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
                if states and all(states):
                    mod_item.setCheckState(0, Qt.Checked)
                elif states and any(states):
                    mod_item.setCheckState(0, Qt.PartiallyChecked)
                else:
                    mod_item.setCheckState(0, Qt.Unchecked)
                mod_item.setData(0, Qt.UserRole, ('mod', modb, mod))
                for seq in sorted(seqs):
                    seq_item = QTreeWidgetItem([seq])
                    seq_item.setFlags(seq_item.flags() | Qt.ItemIsUserCheckable)
                    rows = self.seq_rows.get((modb, mod, seq), [])
                    states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
                    if states and all(states):
                        seq_item.setCheckState(0, Qt.Checked)
                    elif states and any(states):
                        seq_item.setCheckState(0, Qt.PartiallyChecked)
                    else:
                        seq_item.setCheckState(0, Qt.Unchecked)
                    seq_item.setData(0, Qt.UserRole, ('seq', modb, mod, seq))
                    mod_item.addChild(seq_item)
                modb_item.addChild(mod_item)
            self.full_tree.addTopLevelItem(modb_item)

        self.full_tree.expandAll()
        self.full_tree.blockSignals(False)
        try:
            self.full_tree.itemChanged.disconnect(self.onModalityItemChanged)
        except TypeError:
            pass
        self.full_tree.itemChanged.connect(self.onModalityItemChanged)


    def onModalityItemChanged(self, item, column):
        role = item.data(0, Qt.UserRole)
        if not role:
            return
        state = item.checkState(0)
        if role[0] == 'modb':
            modb = role[1]
            for r in self.modb_rows.get(modb, []):
                self.mapping_table.item(r, 0).setCheckState(state)
        elif role[0] == 'mod':
            modb, mod = role[1], role[2]
            for r in self.mod_rows.get((modb, mod), []):
                self.mapping_table.item(r, 0).setCheckState(state)
        elif role[0] == 'seq':
            modb, mod, seq = role[1], role[2], role[3]
            for r in self.seq_rows.get((modb, mod, seq), []):
                self.mapping_table.item(r, 0).setCheckState(state)
        QTimer.singleShot(0, self.populateModalitiesTree)

    def runFullConversion(self):
        logging.info("runFullConversion → Starting full pipeline …")
        """
        Execute full pipeline:
        1) Save updated TSV
        2) build_heuristic_from_tsv.py → auto_heuristic.py
        3) run_heudiconv_from_heuristic.py to convert DICOMs
        4) post_conv_renamer.py on BIDS output
        """
        if not self.tsv_path or not os.path.isfile(self.tsv_path):
            QMessageBox.warning(self, "No TSV", "Please generate the TSV first.")
            return
        if not self.bids_out_dir:
            QMessageBox.warning(self, "No BIDS Output", "Please select a BIDS output directory.")
            return

        # 1) Save updated TSV from table
        try:
            df_updated = []
            for i in range(self.mapping_table.rowCount()):
                include = 1 if self.mapping_table.item(i, 0).checkState() == Qt.Checked else 0
                subj = self.mapping_table.item(i, 1).text()
                ses = self.mapping_table.item(i, 2).text()
                seq = self.mapping_table.item(i, 3).text()
                mod = self.mapping_table.item(i, 4).text()
                modb = self.mapping_table.item(i, 5).text()
                # Build a minimal row; full dicom_inventory already stored other columns
                df_updated.append({
                    'BIDS_name': subj,
                    'session': ses,
                    'include': include,
                    'sequence': seq,
                    'modality': mod,
                    'modality_bids': modb
                })
            # Write back: read original TSV to preserve other columns
            df_orig = pd.read_csv(self.tsv_path, sep="\t")
            # Update only include, sequence, modality_bids columns in df_orig
            for idx, row in enumerate(df_orig.itertuples()):
                df_orig.at[idx, 'include'] = df_updated[idx]['include']
                df_orig.at[idx, 'sequence'] = df_updated[idx]['sequence']
                df_orig.at[idx, 'modality_bids'] = df_updated[idx]['modality_bids']
            df_orig.to_csv(self.tsv_path, sep="\t", index=False)
            self.log_text.append("Saved updated TSV.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save TSV: {e}")
            return

        # Paths for scripts
        script_dir = os.path.dirname(os.path.abspath(__file__))
        build_script = os.path.join(script_dir, "build_heuristic_from_tsv.py")
        run_script = os.path.join(script_dir, "run_heudiconv_from_heuristic.py")
        rename_script = os.path.join(script_dir, "post_conv_renamer.py")

        # 2) Build heuristics directory per StudyDescription
        self.heuristic_dir = os.path.join(self.bids_out_dir, "heuristics")
        try:
            proc = subprocess.run([
                sys.executable, build_script,
                self.tsv_path, self.heuristic_dir
            ], check=True, capture_output=True, text=True)
            self.log_text.append(proc.stdout)
            self.log_text.append(f"Heuristics written to {self.heuristic_dir}")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"build_heuristic failed:\n{e.stderr}")
            return

        # 3) Run HeuDiConv conversion for each study
        try:
            proc = subprocess.run([
                sys.executable, run_script,
                self.dicom_dir, self.heuristic_dir, self.bids_out_dir
            ], check=True, capture_output=True, text=True)
            self.log_text.append(proc.stdout)
            self.log_text.append("HeuDiConv conversion complete.")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"run_heudiconv failed:\n{e.stderr}")
            return

        # 4) Post-conversion fieldmap renaming per dataset
        for heur in Path(self.heuristic_dir).glob("heuristic_*.py"):
            study = heur.stem.replace("heuristic_", "")
            bids_path = os.path.join(self.bids_out_dir, study)
            try:
                proc = subprocess.run([
                    sys.executable, rename_script, bids_path
                ], check=True, capture_output=True, text=True)
                self.log_text.append(proc.stdout)
            except subprocess.CalledProcessError as e:
                QMessageBox.critical(self, "Error", f"post_conv_renamer failed:\n{e.stderr}")
                return
        self.log_text.append("Fieldmap renaming applied.")

        self.log_text.append("Conversion pipeline finished successfully.")

    # ----- Edit tab methods (full bids_editor_ancpbids features) -----
    def openBIDSForEdit(self):
        """Prompt user to select a BIDS dataset root for editing."""
        p = QFileDialog.getExistingDirectory(self, "Select BIDS dataset")
        if p:
            self.bids_root = Path(p)
            self.model.setRootPath(p)
            self.tree.setRootIndex(self.model.index(p))
            self.viewer.clear()
            self.updateStats()
            self.setWindowTitle(f"BIDS Manager – {p}")

    def onTreeClicked(self, idx: QModelIndex):
        """When a file is clicked in the tree, load metadata if JSON/TSV."""
        p = Path(self.model.filePath(idx))
        self.selected = p
        if _get_ext(p) in ['.json', '.tsv', '.nii', '.nii.gz']:
            self.viewer.load_file(p)

    def updateStats(self):
        """Compute and display BIDS stats: total subjects, files, modalities."""
        root = self.bids_root
        self.stats.clear()
        subs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith('sub-')]
        self.stats.addTopLevelItem(QTreeWidgetItem(["Total subjects", str(len(subs))]))
        files = list(root.rglob('*.*'))
        self.stats.addTopLevelItem(QTreeWidgetItem(["Total files", str(len(files))]))
        for sub in subs:
            si = QTreeWidgetItem([sub.name, ""])
            sessions = [d for d in sub.iterdir() if d.is_dir() and d.name.startswith('ses-')]
            if len(sessions) > 1:
                for ses in sessions:
                    s2 = QTreeWidgetItem([ses.name, ""])
                    mods = set(p.parent.name for p in ses.rglob('*.nii*'))
                    s2.addChild(QTreeWidgetItem(["Modalities", str(len(mods))]))
                    for m in mods:
                        imgs = len(list(ses.rglob(f'{m}/*.*')))
                        meta = len(list(ses.rglob(f'{m}/*.json'))) + len(list(ses.rglob(f'{m}/*.tsv')))
                        s2.addChild(QTreeWidgetItem([m, f"imgs:{imgs}, meta:{meta}"]))
                    si.addChild(s2)
            else:
                mods = set(p.parent.name for p in sub.rglob('*.nii*'))
                si.addChild(QTreeWidgetItem(["Sessions", "1"]))
                si.addChild(QTreeWidgetItem(["Modalities", str(len(mods))]))
                for m in mods:
                    imgs = len(list(sub.rglob(f'{m}/*.*')))
                    meta = len(list(sub.rglob(f'{m}/*.json'))) + len(list(sub.rglob(f'{m}/*.tsv')))
                    si.addChild(QTreeWidgetItem([m, f"imgs:{imgs}, meta:{meta}"]))
            self.stats.addTopLevelItem(si)
        self.stats.expandAll()

    def launchBatchRename(self):
        """Open the Batch Rename dialog from bids_editor_ancpbids."""
        dlg = RemapDialog(self, self.bids_root)
        dlg.exec_()


class RemapDialog(QDialog):
    """
    Batch Rename Conditions dialog (from bids_editor_ancpbids).
    Allows regex-based renaming across the dataset.
    """
    def __init__(self, parent, default_scope: Path):
        super().__init__(parent)
        self.setWindowTitle("Batch Remap Conditions")
        self.resize(1000, 600)
        self.bids_root = default_scope
        layout = QVBoxLayout(self)

        # Tabs for multiple conditions
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.add_condition()  # initial condition

        # Button to add more conditions
        btn_add_cond = QPushButton("Add Condition")
        btn_add_cond.clicked.connect(self.add_condition)
        layout.addWidget(btn_add_cond)

        # Scope selector
        scope_layout = QHBoxLayout()
        scope_layout.addWidget(QLabel("Scope:"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Entire dataset", "Selected subjects"])
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("sub-001, sub-002")
        self.subject_edit.setEnabled(False)
        self.scope_combo.currentIndexChanged.connect(
            lambda i: self.subject_edit.setEnabled(i == 1)
        )
        scope_layout.addWidget(self.scope_combo)
        scope_layout.addWidget(self.subject_edit)
        layout.addLayout(scope_layout)

        # Preview and Apply buttons
        btn_layout = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply)
        btn_layout.addWidget(self.preview_button)
        btn_layout.addWidget(self.apply_button)
        layout.addLayout(btn_layout)

        # Preview tree
        self.preview_tree = QTreeWidget()
        self.preview_tree.setColumnCount(2)
        self.preview_tree.setHeaderLabels(["Original", "New Name"])
        hdr = self.preview_tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.preview_tree)

    def add_condition(self):
        """Add a new tab with a table for regex pattern/replacement pairs."""
        tab = QWidget()
        fl = QVBoxLayout(tab)
        rules_tbl = QTableWidget(0, 2)
        rules_tbl.setHorizontalHeaderLabels(["Pattern", "Replacement"])
        hdr = rules_tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        rule_btns = QHBoxLayout()
        btn_addr = QPushButton("Add Rule")
        btn_addr.clicked.connect(lambda: rules_tbl.insertRow(rules_tbl.rowCount()))
        btn_delr = QPushButton("Delete Rule")
        btn_delr.clicked.connect(lambda: rules_tbl.removeRow(rules_tbl.currentRow()))
        rule_btns.addWidget(btn_addr)
        rule_btns.addWidget(btn_delr)
        rule_btns.addStretch()
        fl.addLayout(rule_btns)
        fl.addWidget(rules_tbl)
        index = self.tabs.count() + 1
        self.tabs.addTab(tab, f"Condition {index}")

    def get_scope_paths(self):
        """Retrieve file paths under the selected scope."""
        all_files = [p for p in self.bids_root.rglob('*') if p.is_file()]
        if self.scope_combo.currentIndex() == 0:
            return all_files
        subs = [s.strip() for s in self.subject_edit.text().split(',') if s.strip()]
        if not subs:
            return all_files
        scoped = []
        for p in all_files:
            parts = p.relative_to(self.bids_root).parts
            if parts and parts[0] in subs:
                scoped.append(p)
        return scoped

    def preview(self):
        """Show potential renaming preview in a tree widget."""
        self.preview_tree.clear()
        paths = self.get_scope_paths()
        for path in sorted(paths):
            name = path.name
            new_name = name
            for i in range(self.tabs.count()):
                tbl = self.tabs.widget(i).findChild(QTableWidget)
                for r in range(tbl.rowCount()):
                    pat_item = tbl.item(r, 0)
                    rep_item = tbl.item(r, 1)
                    if pat_item and pat_item.text():
                        new_name = re.sub(pat_item.text(), rep_item.text() if rep_item else "", new_name)
            if new_name != name:
                item = QTreeWidgetItem([path.relative_to(self.bids_root).as_posix(), new_name])
                self.preview_tree.addTopLevelItem(item)
        self.preview_tree.expandAll()

    def apply(self):
        """Apply renaming to files as shown in preview."""
        for i in range(self.preview_tree.topLevelItemCount()):
            it = self.preview_tree.topLevelItem(i)
            orig = self.bids_root / it.text(0)
            new = orig.with_name(it.text(1))
            orig.rename(new)
        QMessageBox.information(self, "Batch Remap", "Rename applied.")
        self.accept()

class MetadataViewer(QWidget):
    """
    Metadata viewer/editor for JSON and TSV sidecars (from bids_editor_ancpbids).
    """
    def __init__(self):
        super().__init__()
        vlay = QVBoxLayout(self)
        self.welcome = QLabel(
            "<h3>Metadata BIDSualizer</h3><br>Load data via File → Open or select a file to begin editing."
        )
        self.welcome.setAlignment(Qt.AlignCenter)
        vlay.addWidget(self.welcome)
        self.toolbar = QHBoxLayout()
        vlay.addLayout(self.toolbar)
        self.viewer = None
        self.current_path = None
        self.data = None  # holds loaded NIfTI data when viewing images

    def clear(self):
        """Clear the toolbar and viewer when switching files."""
        while self.toolbar.count():
            w = self.toolbar.takeAt(0).widget()
            if w:
                w.deleteLater()
        if self.viewer:
            self.layout().removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None
        self.welcome.show()

    def load_file(self, path: Path):
        """Load JSON, TSV or NIfTI file into an editable viewer."""
        self.current_path = path
        self.clear()
        self.welcome.hide()
        ext = _get_ext(path)
        if ext == '.json':
            self._setup_json_toolbar()
            self.viewer = self._json_view(path)
        elif ext == '.tsv':
            self._setup_tsv_toolbar()
            self.viewer = self._tsv_view(path)
        elif ext in ['.nii', '.nii.gz']:
            self._setup_nifti_toolbar()
            self.viewer = self._nifti_view(path)
        self.layout().addWidget(self.viewer)

    def resizeEvent(self, event):
        """Ensure images rescale when the window size decreases."""
        super().resizeEvent(event)
        # If a NIfTI image is currently loaded, update the displayed slice
        if (
            self.data is not None
            and self.current_path
            and _get_ext(self.current_path) in ['.nii', '.nii.gz']
            and hasattr(self, 'img_label')
        ):
            self._update_slice()

    def _setup_json_toolbar(self):
        """Add buttons for JSON editing: Add Field, Delete Field, Save."""
        for txt, fn in [("Add Field", self._add_field), ("Delete Field", self._del_field), ("Save", self._save)]:
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            self.toolbar.addWidget(btn)
        self.toolbar.addStretch()

    def _setup_tsv_toolbar(self):
        """Add buttons for TSV editing: Add Row, Del Row, Add Col, Del Col, Save."""
        for txt, fn in [
            ("Add Row", self._add_row), ("Del Row", self._del_row),
            ("Add Col", self._add_col), ("Del Col", self._del_col), ("Save", self._save)
        ]:
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            self.toolbar.addWidget(btn)
        self.toolbar.addStretch()

    def _setup_nifti_toolbar(self):
        """Toolbar for NIfTI viewer with orientation buttons and sliders."""
        # Orientation buttons
        self.orientation = 2  # 0=sagittal, 1=coronal, 2=axial (default)
        self.ax_btn = QPushButton("Axial")
        self.co_btn = QPushButton("Coronal")
        self.sa_btn = QPushButton("Sagittal")
        for b in (self.ax_btn, self.co_btn, self.sa_btn):
            b.setCheckable(True)
        self.ax_btn.setChecked(True)
        self.ax_btn.clicked.connect(lambda: self._set_orientation(2))
        self.co_btn.clicked.connect(lambda: self._set_orientation(1))
        self.sa_btn.clicked.connect(lambda: self._set_orientation(0))
        self.toolbar.addWidget(self.sa_btn)
        self.toolbar.addWidget(self.co_btn)
        self.toolbar.addWidget(self.ax_btn)

        # Slice slider
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self._update_slice)
        self.toolbar.addWidget(QLabel("Slice:"))
        self.toolbar.addWidget(self.slice_slider)

        # Volume slider
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.valueChanged.connect(self._update_slice)
        self.toolbar.addWidget(QLabel("Volume:"))
        self.toolbar.addWidget(self.vol_slider)
        # Brightness slider
        self.bright_slider = QSlider(Qt.Horizontal)
        self.bright_slider.setRange(-100, 100)
        self.bright_slider.setValue(0)
        self.bright_slider.valueChanged.connect(self._update_slice)
        self.toolbar.addWidget(QLabel("Brightness:"))
        self.toolbar.addWidget(self.bright_slider)

        # Contrast slider
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self._update_slice)
        self.toolbar.addWidget(QLabel("Contrast:"))
        self.toolbar.addWidget(self.contrast_slider)
        self.toolbar.addStretch()

    def _add_field(self):
        """Insert a new key/value pair into JSON tree."""
        tree = self.viewer
        sel = tree.currentItem() or tree.invisibleRootItem()
        item = QTreeWidgetItem(["newKey", "newValue"])
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        sel.addChild(item)
        tree.editItem(item, 0)

    def _del_field(self):
        """Delete selected field from JSON tree."""
        tree = self.viewer
        it = tree.currentItem()
        if it:
            parent = it.parent() or tree.invisibleRootItem()
            parent.removeChild(it)

    def _add_row(self):
        """Insert a new row into TSV table."""
        tbl = self.viewer
        tbl.insertRow(tbl.rowCount())

    def _del_row(self):
        """Delete selected row from TSV table."""
        tbl = self.viewer
        r = tbl.currentRow()
        if r >= 0:
            tbl.removeRow(r)

    def _add_col(self):
        """Insert a new column into TSV table."""
        tbl = self.viewer
        c = tbl.columnCount()
        tbl.insertColumn(c)
        tbl.setHorizontalHeaderItem(c, QTableWidgetItem(f"col{c+1}"))

    def _del_col(self):
        """Delete selected column from TSV table."""
        tbl = self.viewer
        c = tbl.currentColumn()
        if c >= 0:
            tbl.removeColumn(c)

    def _set_orientation(self, axis: int) -> None:
        """Set viewing orientation and update slice slider."""
        self.orientation = axis
        vol_idx = getattr(self, 'vol_slider', None).value() if hasattr(self, 'vol_slider') else 0
        vol = self.data[..., vol_idx] if self.data.ndim == 4 else self.data
        axis_len = vol.shape[axis]
        self.slice_slider.setMaximum(max(axis_len - 1, 0))
        self.slice_slider.setEnabled(axis_len > 1)
        self.slice_slider.setValue(axis_len // 2)
        self._update_slice()

    def _nifti_view(self, path: Path) -> QWidget:
        """Create a simple viewer for NIfTI images with slice/volume controls."""
        self.nifti_img = nib.load(str(path))
        data = self.nifti_img.get_fdata()
        self.data = data
        widget = QWidget()
        vlay = QVBoxLayout(widget)
        self.img_label = _AutoUpdateLabel(self._update_slice)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vlay.addWidget(self.img_label)

        # Configure volume slider range
        n_vols = data.shape[3] if data.ndim == 4 else 1
        self.vol_slider.setMaximum(max(n_vols - 1, 0))
        self.vol_slider.setEnabled(n_vols > 1)

        # Initialize orientation and slice slider
        self._set_orientation(self.orientation)
        self._update_slice()
        return widget

    def _update_slice(self):
        """Update displayed slice when slider moves."""
        vol_idx = getattr(self, 'vol_slider', None).value() if hasattr(self, 'vol_slider') else 0
        vol = self.data[..., vol_idx] if self.data.ndim == 4 else self.data
        axis = getattr(self, 'orientation', 2)
        slice_idx = getattr(self, 'slice_slider', None).value() if hasattr(self, 'slice_slider') else vol.shape[axis] // 2
        if axis == 0:
            slice_img = vol[slice_idx, :, :]
        elif axis == 1:
            slice_img = vol[:, slice_idx, :]
        else:
            slice_img = vol[:, :, slice_idx]
        arr = slice_img.astype(np.float32)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()

        # Apply brightness/contrast adjustments
        bright = getattr(self, 'bright_slider', None)
        contrast = getattr(self, 'contrast_slider', None)
        b_val = bright.value() / 100.0 if bright else 0.0
        c_factor = (contrast.value() / 100.0) if contrast else 1.0
        arr = (arr - 0.5) * c_factor + 0.5 + b_val
        arr = np.clip(arr, 0, 1)

        arr = (arr * 255).astype(np.uint8)
        arr = np.rot90(arr)
        h, w = arr.shape
        img = QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(img)
        self.img_label.setPixmap(pix.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _json_view(self, path: Path) -> QTreeWidget:
        """Create a tree widget to show and edit JSON data."""
        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Key", "Value"])
        tree.setAlternatingRowColors(True)
        hdr = tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.Interactive)
        data = json.loads(path.read_text(encoding='utf-8'))
        self._populate_json(tree.invisibleRootItem(), data)
        tree.expandAll()
        tree.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        return tree

    def _populate_json(self, parent, data):
        """Recursively populate JSON dictionary into the tree widget."""
        if isinstance(data, dict):
            for k, v in data.items():
                it = QTreeWidgetItem([str(k), '' if isinstance(v, (dict, list)) else str(v)])
                it.setFlags(it.flags() | Qt.ItemIsEditable)
                parent.addChild(it)
                if isinstance(v, (dict, list)):
                    self._populate_json(it, v)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                it = QTreeWidgetItem([str(i), '' if isinstance(v, (dict, list)) else str(v)])
                it.setFlags(it.flags() | Qt.ItemIsEditable)
                parent.addChild(it)
                if isinstance(v, (dict, list)):
                    self._populate_json(it, v)

    def _tsv_view(self, path: Path) -> QTableWidget:
        """Create a table widget to show and edit TSV data."""
        df = pd.read_csv(path, sep="\t")
        tbl = QTableWidget(df.shape[0], df.shape[1])
        tbl.setAlternatingRowColors(True)
        hdr = tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        for j, col in enumerate(df.columns):
            tbl.setHorizontalHeaderItem(j, QTableWidgetItem(col))
            for i, val in enumerate(df[col].astype(str)):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                tbl.setItem(i, j, item)
        for j in range(1, tbl.columnCount()):
            hdr.setSectionResizeMode(j, QHeaderView.Interactive)
        tbl.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        return tbl

    def _save(self):
        """Save edits made to JSON or TSV sidecar back to disk."""
        path = self.current_path
        if path.suffix.lower() == '.json':
            data = self._tree_to_obj(self.viewer.invisibleRootItem())
            path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        else:
            tbl = self.viewer
            hdrs = [tbl.horizontalHeaderItem(c).text() for c in range(tbl.columnCount())]
            rows = [{hdrs[c]: tbl.item(r, c).text() for c in range(tbl.columnCount())} for r in range(tbl.rowCount())]
            pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
        QMessageBox.information(self, "Saved", f"Saved {path}")

    def _tree_to_obj(self, root):
        """Convert tree representation back to nested JSON object."""
        obj = {} if root.childCount() else None
        for i in range(root.childCount()):
            ch = root.child(i)
            k, val = ch.text(0), ch.text(1)
            child = self._tree_to_obj(ch)
            obj[k] = child if child is not None else val
        return obj

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(240, 240, 240))
    pal.setColor(QPalette.WindowText, Qt.black)
    app.setPalette(pal)
    win = BIDSManager()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

