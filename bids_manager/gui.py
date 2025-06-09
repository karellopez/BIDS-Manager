import sys
import os
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
    QMenuBar, QMenu, QSizePolicy, QComboBox, QSlider)
from PyQt5.QtCore import Qt, QModelIndex, QTimer, QProcess
from PyQt5.QtGui import QPalette, QColor, QFont, QImage, QPixmap
import logging  # debug logging
import signal
try:
    import psutil
    HAS_PSUTIL = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PSUTIL = False


class _AutoUpdateLabel(QLabel):
    """QLabel that triggers a callback whenever it is resized."""

    def __init__(self, update_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_fn = update_fn

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if callable(self._update_fn):
            self._update_fn()

# ---- basic logging config ----
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def _terminate_process_tree(pid: int):
    """Terminate a process and all of its children without killing the GUI."""
    if pid <= 0:
        return
    # Try killing the process group only if it's not the same as the GUI's
    # to avoid terminating the entire application or IDE.
    try:
        pgid = os.getpgid(pid)
        if pgid != os.getpgid(0):
            os.killpg(pgid, signal.SIGTERM)
            return
    except Exception:
        pass
    if HAS_PSUTIL:
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(children, timeout=3)
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

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
        self.study_rows = {}
        self.subject_rows = {}
        self.session_rows = {}
        self.spec_modb_rows = {}
        self.spec_mod_rows = {}
        self.spec_seq_rows = {}
        self.use_bids_names = True

        # Async process handles
        self.inventory_process = None
        self.conv_process = None
        self.conv_stage = 0
        self.heurs_to_rename = []

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

        # Theme support
        self.statusBar()
        self.themes = self._build_theme_dict()
        self.theme_btn = QPushButton("🌓")  # half-moon icon
        self.theme_btn.setFixedWidth(50)
        # Create a container widget with layout to adjust position
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 2, 0, 6)  # left, top, right, bottom
        layout.setSpacing(0)
        layout.addWidget(self.theme_btn)
        container.setLayout(layout)
        # Add the container to the status bar (left-aligned)
        self.statusBar().addWidget(container)
        # Create the theme menu
        theme_menu = QMenu(self)
        for name in self.themes.keys():
            act = theme_menu.addAction(name)
            act.triggered.connect(lambda _=False, n=name: self.apply_theme(n))
        self.theme_btn.setMenu(theme_menu)
        # Set default theme
        self.apply_theme("Light")

    def _build_theme_dict(self):
        """Return dictionary mapping theme names to QPalettes."""
        themes = {}

        dark_purple = QPalette()
        dark_purple.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_purple.setColor(QPalette.WindowText, Qt.white)
        dark_purple.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_purple.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_purple.setColor(QPalette.ToolTipBase, QColor(65, 65, 65))
        dark_purple.setColor(QPalette.ToolTipText, Qt.white)
        dark_purple.setColor(QPalette.Text, Qt.white)
        dark_purple.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_purple.setColor(QPalette.ButtonText, Qt.white)
        dark_purple.setColor(QPalette.Highlight, QColor(142, 45, 197))
        dark_purple.setColor(QPalette.HighlightedText, Qt.black)
        themes["Dark-purple"] = dark_purple

        dark_blue = QPalette()
        dark_blue.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_blue.setColor(QPalette.WindowText, Qt.white)
        dark_blue.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_blue.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_blue.setColor(QPalette.ToolTipBase, QColor(65, 65, 65))
        dark_blue.setColor(QPalette.ToolTipText, Qt.white)
        dark_blue.setColor(QPalette.Text, Qt.white)
        dark_blue.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_blue.setColor(QPalette.ButtonText, Qt.white)
        dark_blue.setColor(QPalette.Highlight, QColor(65, 105, 225))
        dark_blue.setColor(QPalette.HighlightedText, Qt.black)
        themes["Dark-blue"] = dark_blue

        dark_gold = QPalette()
        dark_gold.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_gold.setColor(QPalette.WindowText, Qt.white)
        dark_gold.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_gold.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_gold.setColor(QPalette.ToolTipBase, QColor(65, 65, 65))
        dark_gold.setColor(QPalette.ToolTipText, Qt.white)
        dark_gold.setColor(QPalette.Text, Qt.white)
        dark_gold.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_gold.setColor(QPalette.ButtonText, Qt.white)
        dark_gold.setColor(QPalette.Highlight, QColor(218, 165, 32))
        dark_gold.setColor(QPalette.HighlightedText, Qt.black)
        themes["Dark-gold"] = dark_gold

        light = QPalette()
        light.setColor(QPalette.Window, Qt.white)
        light.setColor(QPalette.WindowText, Qt.black)
        light.setColor(QPalette.Base, QColor(245, 245, 245))
        light.setColor(QPalette.AlternateBase, Qt.white)
        light.setColor(QPalette.ToolTipBase, Qt.white)
        light.setColor(QPalette.ToolTipText, Qt.black)
        light.setColor(QPalette.Text, Qt.black)
        light.setColor(QPalette.Button, QColor(240, 240, 240))
        light.setColor(QPalette.ButtonText, Qt.black)
        light.setColor(QPalette.Highlight, QColor(100, 149, 237))
        light.setColor(QPalette.HighlightedText, Qt.white)
        themes["Light"] = light

        beige = QPalette()
        beige.setColor(QPalette.Window, QColor(243, 232, 210))
        beige.setColor(QPalette.WindowText, Qt.black)
        beige.setColor(QPalette.Base, QColor(250, 240, 222))
        beige.setColor(QPalette.AlternateBase, QColor(246, 236, 218))
        beige.setColor(QPalette.ToolTipBase, QColor(236, 224, 200))
        beige.setColor(QPalette.ToolTipText, Qt.black)
        beige.setColor(QPalette.Text, Qt.black)
        beige.setColor(QPalette.Button, QColor(242, 231, 208))
        beige.setColor(QPalette.ButtonText, Qt.black)
        beige.setColor(QPalette.Highlight, QColor(196, 148, 70))
        beige.setColor(QPalette.HighlightedText, Qt.white)
        themes["Beige"] = beige

        ocean = QPalette()
        ocean.setColor(QPalette.Window, QColor(225, 238, 245))
        ocean.setColor(QPalette.WindowText, Qt.black)
        ocean.setColor(QPalette.Base, QColor(240, 248, 252))
        ocean.setColor(QPalette.AlternateBase, QColor(230, 240, 247))
        ocean.setColor(QPalette.ToolTipBase, QColor(215, 230, 240))
        ocean.setColor(QPalette.ToolTipText, Qt.black)
        ocean.setColor(QPalette.Text, Qt.black)
        ocean.setColor(QPalette.Button, QColor(213, 234, 242))
        ocean.setColor(QPalette.ButtonText, Qt.black)
        ocean.setColor(QPalette.Highlight, QColor(0, 123, 167))
        ocean.setColor(QPalette.HighlightedText, Qt.white)
        themes["Ocean"] = ocean

        hc = QPalette()
        hc.setColor(QPalette.Window, Qt.black)
        hc.setColor(QPalette.WindowText, Qt.white)
        hc.setColor(QPalette.Base, Qt.black)
        hc.setColor(QPalette.AlternateBase, Qt.black)
        hc.setColor(QPalette.ToolTipBase, Qt.black)
        hc.setColor(QPalette.ToolTipText, Qt.white)
        hc.setColor(QPalette.Text, Qt.white)
        hc.setColor(QPalette.BrightText, Qt.white)
        hc.setColor(QPalette.Button, Qt.black)
        hc.setColor(QPalette.ButtonText, Qt.white)
        hc.setColor(QPalette.Highlight, QColor(255, 215, 0))
        hc.setColor(QPalette.HighlightedText, Qt.black)
        themes["Contrast"] = hc

        hc_w = QPalette()
        hc_w.setColor(QPalette.Window, Qt.white)
        hc_w.setColor(QPalette.WindowText, Qt.black)
        hc_w.setColor(QPalette.Base, Qt.white)
        hc_w.setColor(QPalette.AlternateBase, Qt.white)
        hc_w.setColor(QPalette.ToolTipBase, Qt.white)
        hc_w.setColor(QPalette.ToolTipText, Qt.black)
        hc_w.setColor(QPalette.Text, Qt.black)
        hc_w.setColor(QPalette.BrightText, Qt.black)
        hc_w.setColor(QPalette.Button, Qt.white)
        hc_w.setColor(QPalette.ButtonText, Qt.black)
        hc_w.setColor(QPalette.Highlight, QColor(255, 215, 0))
        hc_w.setColor(QPalette.HighlightedText, Qt.black)
        themes["Contrast White"] = hc_w

        solar = QPalette()
        solar.setColor(QPalette.Window, QColor(253, 246, 227))
        solar.setColor(QPalette.WindowText, QColor(101, 123, 131))
        solar.setColor(QPalette.Base, QColor(255, 250, 240))
        solar.setColor(QPalette.AlternateBase, QColor(253, 246, 227))
        solar.setColor(QPalette.ToolTipBase, QColor(238, 232, 213))
        solar.setColor(QPalette.ToolTipText, QColor(88, 110, 117))
        solar.setColor(QPalette.Text, QColor(88, 110, 117))
        solar.setColor(QPalette.Button, QColor(238, 232, 213))
        solar.setColor(QPalette.ButtonText, QColor(88, 110, 117))
        solar.setColor(QPalette.Highlight, QColor(38, 139, 210))
        solar.setColor(QPalette.HighlightedText, Qt.white)
        themes["Solar"] = solar

        cyber = QPalette()
        cyber.setColor(QPalette.Window, QColor(20, 20, 30))
        cyber.setColor(QPalette.WindowText, QColor(0, 255, 255))
        cyber.setColor(QPalette.Base, QColor(30, 30, 45))
        cyber.setColor(QPalette.AlternateBase, QColor(25, 25, 35))
        cyber.setColor(QPalette.ToolTipBase, QColor(45, 45, 65))
        cyber.setColor(QPalette.ToolTipText, QColor(255, 0, 255))
        cyber.setColor(QPalette.Text, QColor(0, 255, 255))
        cyber.setColor(QPalette.Button, QColor(40, 40, 55))
        cyber.setColor(QPalette.ButtonText, QColor(255, 0, 255))
        cyber.setColor(QPalette.Highlight, QColor(255, 0, 128))
        cyber.setColor(QPalette.HighlightedText, Qt.white)
        themes["Cyber"] = cyber

        drac = QPalette()
        drac.setColor(QPalette.Window, QColor("#282a36"))
        drac.setColor(QPalette.WindowText, QColor("#f8f8f2"))
        drac.setColor(QPalette.Base, QColor("#1e1f29"))
        drac.setColor(QPalette.AlternateBase, QColor("#282a36"))
        drac.setColor(QPalette.ToolTipBase, QColor("#44475a"))
        drac.setColor(QPalette.ToolTipText, QColor("#f8f8f2"))
        drac.setColor(QPalette.Text, QColor("#f8f8f2"))
        drac.setColor(QPalette.Button, QColor("#44475a"))
        drac.setColor(QPalette.ButtonText, QColor("#f8f8f2"))
        drac.setColor(QPalette.Highlight, QColor("#bd93f9"))
        drac.setColor(QPalette.HighlightedText, Qt.black)
        themes["Dracula"] = drac

        nord = QPalette()
        nord.setColor(QPalette.Window, QColor("#2e3440"))
        nord.setColor(QPalette.WindowText, QColor("#d8dee9"))
        nord.setColor(QPalette.Base, QColor("#3b4252"))
        nord.setColor(QPalette.AlternateBase, QColor("#434c5e"))
        nord.setColor(QPalette.ToolTipBase, QColor("#4c566a"))
        nord.setColor(QPalette.ToolTipText, QColor("#eceff4"))
        nord.setColor(QPalette.Text, QColor("#e5e9f0"))
        nord.setColor(QPalette.Button, QColor("#4c566a"))
        nord.setColor(QPalette.ButtonText, QColor("#d8dee9"))
        nord.setColor(QPalette.Highlight, QColor("#88c0d0"))
        nord.setColor(QPalette.HighlightedText, Qt.black)
        themes["Nord"] = nord

        gruv = QPalette()
        gruv.setColor(QPalette.Window, QColor("#282828"))
        gruv.setColor(QPalette.WindowText, QColor("#ebdbb2"))
        gruv.setColor(QPalette.Base, QColor("#32302f"))
        gruv.setColor(QPalette.AlternateBase, QColor("#3c3836"))
        gruv.setColor(QPalette.ToolTipBase, QColor("#504945"))
        gruv.setColor(QPalette.ToolTipText, QColor("#fbf1c7"))
        gruv.setColor(QPalette.Text, QColor("#ebdbb2"))
        gruv.setColor(QPalette.Button, QColor("#504945"))
        gruv.setColor(QPalette.ButtonText, QColor("#ebdbb2"))
        gruv.setColor(QPalette.Highlight, QColor("#d79921"))
        gruv.setColor(QPalette.HighlightedText, Qt.black)
        themes["Gruvbox"] = gruv

        mono = QPalette()
        mono.setColor(QPalette.Window, QColor("#272822"))
        mono.setColor(QPalette.WindowText, QColor("#f8f8f2"))
        mono.setColor(QPalette.Base, QColor("#1e1f1c"))
        mono.setColor(QPalette.AlternateBase, QColor("#272822"))
        mono.setColor(QPalette.ToolTipBase, QColor("#3e3d32"))
        mono.setColor(QPalette.ToolTipText, QColor("#f8f8f2"))
        mono.setColor(QPalette.Text, QColor("#f8f8f2"))
        mono.setColor(QPalette.Button, QColor("#3e3d32"))
        mono.setColor(QPalette.ButtonText, QColor("#f8f8f2"))
        mono.setColor(QPalette.Highlight, QColor("#a6e22e"))
        mono.setColor(QPalette.HighlightedText, Qt.black)
        themes["Monokai"] = mono

        tokyo = QPalette()
        tokyo.setColor(QPalette.Window, QColor("#1a1b26"))
        tokyo.setColor(QPalette.WindowText, QColor("#c0caf5"))
        tokyo.setColor(QPalette.Base, QColor("#1f2335"))
        tokyo.setColor(QPalette.AlternateBase, QColor("#24283b"))
        tokyo.setColor(QPalette.ToolTipBase, QColor("#414868"))
        tokyo.setColor(QPalette.ToolTipText, QColor("#c0caf5"))
        tokyo.setColor(QPalette.Text, QColor("#c0caf5"))
        tokyo.setColor(QPalette.Button, QColor("#414868"))
        tokyo.setColor(QPalette.ButtonText, QColor("#c0caf5"))
        tokyo.setColor(QPalette.Highlight, QColor("#7aa2f7"))
        tokyo.setColor(QPalette.HighlightedText, Qt.white)
        themes["Tokyo"] = tokyo

        mocha = QPalette()
        mocha.setColor(QPalette.Window, QColor("#1e1e2e"))
        mocha.setColor(QPalette.WindowText, QColor("#cdd6f4"))
        mocha.setColor(QPalette.Base, QColor("#181825"))
        mocha.setColor(QPalette.AlternateBase, QColor("#1e1e2e"))
        mocha.setColor(QPalette.ToolTipBase, QColor("#313244"))
        mocha.setColor(QPalette.ToolTipText, QColor("#cdd6f4"))
        mocha.setColor(QPalette.Text, QColor("#cdd6f4"))
        mocha.setColor(QPalette.Button, QColor("#313244"))
        mocha.setColor(QPalette.ButtonText, QColor("#cdd6f4"))
        mocha.setColor(QPalette.Highlight, QColor("#f38ba8"))
        mocha.setColor(QPalette.HighlightedText, Qt.black)
        themes["Mocha"] = mocha

        pale = QPalette()
        pale.setColor(QPalette.Window, QColor("#292d3e"))
        pale.setColor(QPalette.WindowText, QColor("#a6accd"))
        pale.setColor(QPalette.Base, QColor("#1b1d2b"))
        pale.setColor(QPalette.AlternateBase, QColor("#222436"))
        pale.setColor(QPalette.ToolTipBase, QColor("#444267"))
        pale.setColor(QPalette.ToolTipText, QColor("#a6accd"))
        pale.setColor(QPalette.Text, QColor("#a6accd"))
        pale.setColor(QPalette.Button, QColor("#444267"))
        pale.setColor(QPalette.ButtonText, QColor("#a6accd"))
        pale.setColor(QPalette.Highlight, QColor("#82aaff"))
        pale.setColor(QPalette.HighlightedText, Qt.black)
        themes["Palenight"] = pale

        return themes

    def apply_theme(self, name: str):
        """Apply palette chosen from the Theme menu."""
        app = QApplication.instance()
        app.setPalette(self.themes[name])
        font = app.font()
        if name in ("Contrast", "Contrast White"):
            font.setWeight(QFont.Bold)
            font.setPointSize(font.pointSize() + 1)
        else:
            font.setWeight(QFont.Normal)
        app.setFont(font)

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
        self.tsv_stop_button = QPushButton("Stop")
        self.tsv_stop_button.setEnabled(False)
        self.tsv_stop_button.clicked.connect(self.stopInventory)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.tsv_button)
        btn_row.addWidget(self.tsv_stop_button)
        btn_row.addStretch()
        cfg_layout.addLayout(btn_row, 3, 0, 1, 3)

        main_layout.addWidget(cfg_group)

        left_split = QSplitter(Qt.Vertical)
        right_split = QSplitter(Qt.Vertical)

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
        left_split.addWidget(tsv_group)

        modal_group = QGroupBox("Modalities")
        modal_layout = QVBoxLayout(modal_group)
        self.modal_tabs = QTabWidget()
        full_tab = QWidget()
        full_layout = QVBoxLayout(full_tab)
        self.full_tree = QTreeWidget()
        # Display only one column with the BIDS modality
        self.full_tree.setHeaderLabels(["BIDS Modality"])
        full_layout.addWidget(self.full_tree)
        self.modal_tabs.addTab(full_tab, "General view")

        specific_tab = QWidget()
        specific_layout = QVBoxLayout(specific_tab)
        self.specific_tree = QTreeWidget()
        self.specific_tree.setHeaderLabels(["Study/Subject"])
        specific_layout.addWidget(self.specific_tree)
        self.modal_tabs.addTab(specific_tab, "Specific view")

        naming_tab = QWidget()
        naming_layout = QVBoxLayout(naming_tab)
        self.naming_table = QTableWidget()
        self.naming_table.setColumnCount(3)
        self.naming_table.setHorizontalHeaderLabels(["Study", "Given name", "BIDS name"])
        n_hdr = self.naming_table.horizontalHeader()
        n_hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        n_hdr.setStretchLastSection(True)
        naming_layout.addWidget(self.naming_table)
        self.naming_table.itemChanged.connect(self._onNamingEdited)
        self.name_choice = QComboBox()
        self.name_choice.addItems(["Use BIDS names", "Use given names"])
        self.name_choice.currentIndexChanged.connect(self._onNameChoiceChanged)
        naming_layout.addWidget(self.name_choice)
        self.modal_tabs.addTab(naming_tab, "Edit naming")

        modal_layout.addWidget(self.modal_tabs)
        right_split.addWidget(modal_group)
        left_split.setStretchFactor(0, 1)
        left_split.setStretchFactor(1, 1)
        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 1)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_tabs = QTabWidget()

        text_tab = QWidget()
        text_lay = QVBoxLayout(text_tab)
        self.preview_text = QTreeWidget()
        self.preview_text.setHeaderLabels(["BIDS Path"])
        text_lay.addWidget(self.preview_text)
        self.preview_tabs.addTab(text_tab, "Text")

        tree_tab = QWidget()
        tree_lay = QVBoxLayout(tree_tab)
        self.preview_tree = QTreeWidget()
        self.preview_tree.setHeaderLabels(["BIDS Structure"])
        tree_lay.addWidget(self.preview_tree)
        self.preview_tabs.addTab(tree_tab, "Tree")

        preview_layout.addWidget(self.preview_tabs)
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.generatePreview)
        preview_layout.addWidget(self.preview_button)

        btn_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.runFullConversion)
        self.run_stop_button = QPushButton("Stop")
        self.run_stop_button.setEnabled(False)
        self.run_stop_button.clicked.connect(self.stopConversion)
        btn_row.addStretch()
        btn_row.addWidget(self.run_button)
        btn_row.addWidget(self.run_stop_button)

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

        left_split.addWidget(preview_container)
        right_split.addWidget(log_group)

        splitter = QSplitter()
        splitter.addWidget(left_split)
        splitter.addWidget(right_split)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

        self.tabs.addTab(self.convert_tab, "Convert")

    def _add_preview_path(self, parts):
        parent = self.preview_tree.invisibleRootItem()
        for part in parts:
            match = None
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.text(0) == part:
                    match = child
                    break
            if match is None:
                match = QTreeWidgetItem([part])
                parent.addChild(match)
            parent = match

    def generatePreview(self):
        logging.info("generatePreview → Building preview tree …")
        """Populate preview tabs based on checked sequences."""
        self.preview_text.clear()
        self.preview_tree.clear()
        multi_study = len(self.study_set) > 1
        for i in range(self.mapping_table.rowCount()):
            include = (self.mapping_table.item(i, 0).checkState() == Qt.Checked)
            if not include:
                continue
            info = self.row_info[i]
            subj = info['bids'] if self.use_bids_names else info['given']
            study = self.mapping_table.item(i, 1).data(Qt.UserRole) or ""
            ses = self.mapping_table.item(i, 2).text()
            seq = self.mapping_table.item(i, 3).text()
            modb = self.mapping_table.item(i, 5).text()

            path_parts = []
            if multi_study:
                path_parts.append(study)
            path_parts.extend([subj, ses, modb])
            base = f"{subj}_{ses}_{seq}"

            if modb == "fmap":
                for suffix in ["magnitude1", "magnitude2", "phasediff"]:
                    fname = f"{base}_{suffix}.nii.gz"
                    full = path_parts + [fname]
                    self.preview_text.addTopLevelItem(QTreeWidgetItem(["/".join(full)]))
                    self._add_preview_path(full)
            else:
                fname = f"{base}.nii.gz"
                full = path_parts + [fname]
                self.preview_text.addTopLevelItem(QTreeWidgetItem(["/".join(full)]))
                self._add_preview_path(full)

        self.preview_text.expandAll()
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

        # Run dicom_inventory asynchronously
        if self.inventory_process and self.inventory_process.state() != QProcess.NotRunning:
            return

        self.log_text.append("Starting TSV generation…")
        self.tsv_button.setEnabled(False)
        self.tsv_stop_button.setEnabled(True)
        self.inventory_process = QProcess(self)
        self.inventory_process.finished.connect(self._inventoryFinished)
        args = ["-m", "bids_manager.dicom_inventory", self.dicom_dir, self.tsv_path]
        self.inventory_process.start(sys.executable, args)

    def _inventoryFinished(self):
        ok = self.inventory_process.exitCode() == 0 if self.inventory_process else False
        self.inventory_process = None
        self.tsv_button.setEnabled(True)
        self.tsv_stop_button.setEnabled(False)
        if ok:
            self.log_text.append("TSV generation finished.")
            self.loadMappingTable()
        else:
            self.log_text.append("TSV generation failed.")

    def stopInventory(self):
        if self.inventory_process and self.inventory_process.state() != QProcess.NotRunning:
            pid = int(self.inventory_process.processId())
            _terminate_process_tree(pid)
            self.inventory_process = None
            self.tsv_button.setEnabled(True)
            self.tsv_stop_button.setEnabled(False)
            self.log_text.append("TSV generation cancelled.")

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
        self.study_rows.clear()
        self.subject_rows.clear()
        self.session_rows.clear()
        self.spec_modb_rows.clear()
        self.spec_mod_rows.clear()
        self.spec_seq_rows.clear()
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
            bids_name = _clean(row.get('BIDS_name'))
            subj_item = QTableWidgetItem(bids_name)
            subj_item.setFlags(subj_item.flags() & ~Qt.ItemIsEditable)
            study = _clean(row.get('StudyDescription'))
            subj_item.setData(Qt.UserRole, study)
            self.study_set.add(study)
            self.mapping_table.setItem(r, 1, subj_item)
            # Session (non-editable)
            session = _clean(row.get('session'))
            ses_item = QTableWidgetItem(session)
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
            modb = _clean(row.get('modality_bids'))
            modb_item = QTableWidgetItem(modb)
            modb_item.setFlags(modb_item.flags() | Qt.ItemIsEditable)
            self.mapping_table.setItem(r, 5, modb_item)

            mod = _clean(row.get('modality'))
            seq = _clean(row.get('sequence'))
            given = _clean(row.get('subject'))
            self.row_info.append({
                'study': study,
                'bids': bids_name,
                'given': given,
                'ses': session,
                'modb': modb,
                'mod': mod,
                'seq': seq,
            })
        self.log_text.append("Loaded TSV into mapping table.")

        # Build modality/sequence lookup for tree interactions
        for idx, info in enumerate(self.row_info):
            self.modb_rows.setdefault(info['modb'], []).append(idx)
            self.mod_rows.setdefault((info['modb'], info['mod']), []).append(idx)
            self.seq_rows.setdefault((info['modb'], info['mod'], info['seq']), []).append(idx)
            self.study_rows.setdefault(info['study'], []).append(idx)
            self.subject_rows.setdefault((info['study'], info['bids']), []).append(idx)
            self.session_rows.setdefault((info['study'], info['bids'], info['ses']), []).append(idx)
            self.spec_modb_rows.setdefault((info['study'], info['bids'], info['ses'], info['modb']), []).append(idx)
            self.spec_mod_rows.setdefault((info['study'], info['bids'], info['ses'], info['modb'], info['mod']), []).append(idx)
            self.spec_seq_rows.setdefault((info['study'], info['bids'], info['ses'], info['modb'], info['mod'], info['seq']), []).append(idx)

        self.populateModalitiesTree()
        self.populateSpecificTree()

        # Populate naming table
        self.naming_table.blockSignals(True)
        self.naming_table.setRowCount(0)
        name_df = df[["StudyDescription", "subject", "BIDS_name"]].copy()
        name_df['subject'] = (name_df.replace({'subject': {"": pd.NA}})
                              .groupby(["StudyDescription", "BIDS_name"])
                              ['subject'].transform(lambda x: x.ffill().bfill()))
        name_df = name_df.drop_duplicates(subset=["StudyDescription", "BIDS_name"])
        for _, row in name_df.iterrows():
            nr = self.naming_table.rowCount()
            self.naming_table.insertRow(nr)
            sitem = QTableWidgetItem(_clean(row["StudyDescription"]))
            sitem.setFlags(sitem.flags() & ~Qt.ItemIsEditable)
            self.naming_table.setItem(nr, 0, sitem)
            gitem = QTableWidgetItem(_clean(row["subject"]))
            gitem.setFlags(gitem.flags() & ~Qt.ItemIsEditable)
            self.naming_table.setItem(nr, 1, gitem)
            bitem = QTableWidgetItem(_clean(row["BIDS_name"]))
            bitem.setFlags(bitem.flags() | Qt.ItemIsEditable)
            self.naming_table.setItem(nr, 2, bitem)
        self.naming_table.blockSignals(False)


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

    def onSpecificItemChanged(self, item, column):
        role = item.data(0, Qt.UserRole)
        if not role:
            return
        state = item.checkState(0)
        tp = role[0]
        if tp == 'study':
            rows = self.study_rows.get(role[1], [])
        elif tp == 'subject':
            rows = self.subject_rows.get((role[1], role[2]), [])
        elif tp == 'session':
            rows = self.session_rows.get((role[1], role[2], role[3]), [])
        elif tp == 'modb':
            rows = self.spec_modb_rows.get((role[1], role[2], role[3], role[4]), [])
        elif tp == 'mod':
            rows = self.spec_mod_rows.get((role[1], role[2], role[3], role[4], role[5]), [])
        elif tp == 'seq':
            rows = self.spec_seq_rows.get((role[1], role[2], role[3], role[4], role[5], role[6]), [])
        else:
            rows = []
        for r in rows:
            self.mapping_table.item(r, 0).setCheckState(state)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)

    def _onNamingEdited(self, item):
        if item.column() != 2:
            return
        study = self.naming_table.item(item.row(), 0).text()
        given = self.naming_table.item(item.row(), 1).text()
        new_bids = item.text()
        for idx, info in enumerate(self.row_info):
            if info['study'] == study and info['given'] == given:
                info['bids'] = new_bids
                self.mapping_table.item(idx, 1).setText(new_bids)
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.generatePreview)

    def _onNameChoiceChanged(self, _index=None):
        self.use_bids_names = self.name_choice.currentIndex() == 0
        QTimer.singleShot(0, self.generatePreview)

    def _save_tree_expansion(self, tree):
        states = {}

        def recurse(item):
            path = []
            it = item
            while it is not None:
                path.insert(0, it.text(0))
                it = it.parent()
            states[tuple(path)] = item.isExpanded()
            for i in range(item.childCount()):
                recurse(item.child(i))

        for i in range(tree.topLevelItemCount()):
            recurse(tree.topLevelItem(i))
        return states

    def _restore_tree_expansion(self, tree, states):
        def recurse(item):
            path = []
            it = item
            while it is not None:
                path.insert(0, it.text(0))
                it = it.parent()
            if states.get(tuple(path)):
                item.setExpanded(True)
            for i in range(item.childCount()):
                recurse(item.child(i))

        for i in range(tree.topLevelItemCount()):
            recurse(tree.topLevelItem(i))


    def populateSpecificTree(self):
        """Build detailed tree (study→subject→session→modality)."""
        expanded = self._save_tree_expansion(self.specific_tree)
        self.specific_tree.blockSignals(True)
        self.specific_tree.clear()

        tree_map = {}
        for info in self.row_info:
            tree_map.setdefault(info['study'], {})\
                    .setdefault(info['bids'], {})\
                    .setdefault(info['ses'], {})\
                    .setdefault(info['modb'], {})\
                    .setdefault(info['mod'], set()).add(info['seq'])

        def _state(rows):
            states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
            if states and all(states):
                return Qt.Checked
            if states and any(states):
                return Qt.PartiallyChecked
            return Qt.Unchecked

        for study, sub_map in sorted(tree_map.items()):
            st_item = QTreeWidgetItem([study])
            st_item.setFlags(st_item.flags() | Qt.ItemIsUserCheckable)
            st_item.setCheckState(0, _state(self.study_rows.get(study, [])))
            st_item.setData(0, Qt.UserRole, ('study', study))
            for subj, ses_map in sorted(sub_map.items()):
                su_item = QTreeWidgetItem([subj])
                su_item.setFlags(su_item.flags() | Qt.ItemIsUserCheckable)
                su_item.setCheckState(0, _state(self.subject_rows.get((study, subj), [])))
                su_item.setData(0, Qt.UserRole, ('subject', study, subj))
                for ses, modb_map in sorted(ses_map.items()):
                    se_item = QTreeWidgetItem([ses])
                    se_item.setFlags(se_item.flags() | Qt.ItemIsUserCheckable)
                    se_item.setCheckState(0, _state(self.session_rows.get((study, subj, ses), [])))
                    se_item.setData(0, Qt.UserRole, ('session', study, subj, ses))
                    for modb, mod_map in sorted(modb_map.items()):
                        mb_item = QTreeWidgetItem([modb])
                        mb_item.setFlags(mb_item.flags() | Qt.ItemIsUserCheckable)
                        mb_item.setCheckState(0, _state(self.spec_modb_rows.get((study, subj, ses, modb), [])))
                        mb_item.setData(0, Qt.UserRole, ('modb', study, subj, ses, modb))
                        for mod, seqs in sorted(mod_map.items()):
                            mo_item = QTreeWidgetItem([mod])
                            mo_item.setFlags(mo_item.flags() | Qt.ItemIsUserCheckable)
                            mo_item.setCheckState(0, _state(self.spec_mod_rows.get((study, subj, ses, modb, mod), [])))
                            mo_item.setData(0, Qt.UserRole, ('mod', study, subj, ses, modb, mod))
                            for seq in sorted(seqs):
                                sq_item = QTreeWidgetItem([seq])
                                sq_item.setFlags(sq_item.flags() | Qt.ItemIsUserCheckable)
                                sq_item.setCheckState(0, _state(self.spec_seq_rows.get((study, subj, ses, modb, mod, seq), [])))
                                sq_item.setData(0, Qt.UserRole, ('seq', study, subj, ses, modb, mod, seq))
                                mo_item.addChild(sq_item)
                            mb_item.addChild(mo_item)
                        se_item.addChild(mb_item)
                    su_item.addChild(se_item)
                st_item.addChild(su_item)
            self.specific_tree.addTopLevelItem(st_item)

        self._restore_tree_expansion(self.specific_tree, expanded)
        if not expanded:
            self.specific_tree.expandAll()
        self.specific_tree.blockSignals(False)
        try:
            self.specific_tree.itemChanged.disconnect(self.onSpecificItemChanged)
        except TypeError:
            pass
        self.specific_tree.itemChanged.connect(self.onSpecificItemChanged)


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
        if self.conv_process and self.conv_process.state() != QProcess.NotRunning:
            return
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
                info = self.row_info[i]
                subj = info['bids'] if self.use_bids_names else info['given']
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
                df_orig.at[idx, 'BIDS_name'] = df_updated[idx]['BIDS_name']
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
        self.build_script = os.path.join(script_dir, "build_heuristic_from_tsv.py")
        self.run_script = os.path.join(script_dir, "run_heudiconv_from_heuristic.py")
        self.rename_script = os.path.join(script_dir, "post_conv_renamer.py")

        self.heuristic_dir = os.path.join(self.bids_out_dir, "heuristics")
        self.heurs_to_rename = []
        self.conv_stage = 0

        self.log_text.append("Building heuristics…")
        self.run_button.setEnabled(False)
        self.run_stop_button.setEnabled(True)
        self.conv_process = QProcess(self)
        self.conv_process.finished.connect(self._convStepFinished)
        args = [self.build_script, self.tsv_path, self.heuristic_dir]
        self.conv_process.start(sys.executable, args)

    def _convStepFinished(self, exitCode, _status):
        if self.conv_stage == 0:
            if exitCode != 0:
                QMessageBox.critical(self, "Error", "build_heuristic failed")
                self.stopConversion()
                return
            self.log_text.append(f"Heuristics written to {self.heuristic_dir}")
            self.conv_stage = 1
            self.log_text.append("Running HeuDiConv…")
            args = [self.run_script, self.dicom_dir, self.heuristic_dir, self.bids_out_dir, '--subject-tsv', self.tsv_path]
            self.conv_process.start(sys.executable, args)
        elif self.conv_stage == 1:
            if exitCode != 0:
                QMessageBox.critical(self, "Error", "run_heudiconv failed")
                self.stopConversion()
                return
            self.log_text.append("HeuDiConv conversion complete.")
            self.conv_stage = 2
            self.heurs_to_rename = list(Path(self.heuristic_dir).glob("heuristic_*.py"))
            self._runNextRename()
        elif self.conv_stage == 2:
            if exitCode != 0:
                QMessageBox.critical(self, "Error", "post_conv_renamer failed")
                self.stopConversion()
                return
            if self.heurs_to_rename:
                self._runNextRename()
            else:
                self.log_text.append("Conversion pipeline finished successfully.")
                self.stopConversion(success=True)

    def _runNextRename(self):
        if not self.heurs_to_rename:
            self._convStepFinished(0, 0)
            return
        heur = self.heurs_to_rename.pop(0)
        dataset = heur.stem.replace("heuristic_", "")
        bids_path = os.path.join(self.bids_out_dir, dataset)
        self.log_text.append(f"Renaming fieldmaps for {dataset}…")
        args = [self.rename_script, bids_path]
        self.conv_process.start(sys.executable, args)

    def stopConversion(self, success: bool = False):
        if self.conv_process and self.conv_process.state() != QProcess.NotRunning:
            pid = int(self.conv_process.processId())
            _terminate_process_tree(pid)
        self.conv_process = None
        self.run_button.setEnabled(True)
        self.run_stop_button.setEnabled(False)
        if not success:
            self.log_text.append("Conversion cancelled.")

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
    win = BIDSManager()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

