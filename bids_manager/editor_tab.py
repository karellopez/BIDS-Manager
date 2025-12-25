"""Helper utilities for constructing the Editor tab UI."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QFileSystemModel,
    QHeaderView,
    QLabel,
    QMenuBar,
    QSizePolicy,
    QSplitter,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
)


def init_edit_tab(window, metadata_viewer_cls):
    """Set up the Editor tab on ``window`` using ``metadata_viewer_cls``."""

    window.edit_tab = QWidget()
    edit_layout = QVBoxLayout(window.edit_tab)
    edit_layout.setContentsMargins(10, 10, 10, 10)
    edit_layout.setSpacing(8)

    # Internal menu bar for Edit features
    menu = QMenuBar()
    menu.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    menu.setMaximumHeight(24)
    file_menu = menu.addMenu("File")
    open_act = QAction("Open BIDS…", window)
    open_act.triggered.connect(window.openBIDSForEdit)
    file_menu.addAction(open_act)
    tools_menu = menu.addMenu("Tools")
    rename_act = QAction("Batch Rename…", window)
    rename_act.triggered.connect(window.launchBatchRename)
    tools_menu.addAction(rename_act)

    intended_act = QAction("Set Intended For…", window)
    intended_act.triggered.connect(window.launchIntendedForEditor)
    tools_menu.addAction(intended_act)

    refresh_act = QAction("Refresh scans.tsv", window)
    refresh_act.triggered.connect(window.refreshScansTsv)
    tools_menu.addAction(refresh_act)

    ignore_act = QAction("Edit .bidsignore…", window)
    ignore_act.triggered.connect(window.launchBidsIgnore)
    tools_menu.addAction(ignore_act)
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
    window.model = QFileSystemModel()
    window.model.setRootPath("")
    window.tree = QTreeView()
    window.tree.setModel(window.model)
    window.tree.setEditTriggers(
        QAbstractItemView.EditKeyPressed | QAbstractItemView.SelectedClicked
    )
    window.tree.setColumnHidden(1, True)
    window.tree.setColumnHidden(3, True)
    hdr = window.tree.header()
    hdr.setSectionResizeMode(0, QHeaderView.Interactive)
    hdr.setSectionResizeMode(2, QHeaderView.Interactive)
    left_layout.addWidget(window.tree)
    window.tree.clicked.connect(window.onTreeClicked)

    left_layout.addWidget(QLabel('<b>BIDStatistics</b>'))
    window.stats = QTreeWidget()
    window.stats.setHeaderLabels(["Metric", "Value"])
    window.stats.setAlternatingRowColors(True)
    s_hdr = window.stats.header()
    s_hdr.setSectionResizeMode(0, QHeaderView.Interactive)
    s_hdr.setSectionResizeMode(1, QHeaderView.Interactive)
    left_layout.addWidget(window.stats)

    splitter.addWidget(left_panel)

    # Right panel: MetadataViewer (reused from original)
    window.viewer = metadata_viewer_cls()
    splitter.addWidget(window.viewer)
    splitter.setStretchFactor(1, 2)

    edit_layout.addWidget(splitter)
    window.tabs.addTab(window.edit_tab, "Editor")
