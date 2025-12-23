"""Editor tab implementation for the BIDS Manager GUI.

This module contains the embedded metadata editor, 3-D viewers, and related
dialogs. The logic mirrors the original monolithic implementation but is now
isolated for clarity.
"""

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from bids_manager.GUI.common import *  # noqa: F401,F403 - shared GUI helpers

class _SliceControl:
    """Stores widgets controlling a single anatomical slicer."""

    checkbox: QCheckBox
    min_slider: QSlider
    max_slider: QSlider
    min_value: QLabel
    max_value: QLabel
    axis: int
    negative_name: str
    positive_name: str

class _AutoUpdateLabel(QLabel):
    """QLabel that triggers a callback whenever it is resized."""

    def __init__(self, update_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_fn = update_fn

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if callable(self._update_fn):
            self._update_fn()

class _ImageLabel(_AutoUpdateLabel):
    """Label that notifies on resize and mouse clicks."""

    def __init__(self, update_fn, click_fn, *args, **kwargs):
        super().__init__(update_fn, *args, **kwargs)
        self._click_fn = click_fn

    def mousePressEvent(self, event):
        if callable(self._click_fn):
            self._click_fn(event)
        super().mousePressEvent(event)

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
        btn_addc = QPushButton("Add Condition")
        btn_addc.clicked.connect(self.add_condition)
        btn_delc = QPushButton("Delete Condition")
        btn_delc.clicked.connect(lambda: self.delete_condition(tab))
        rule_btns.addWidget(btn_addr)
        rule_btns.addWidget(btn_delr)
        rule_btns.addWidget(btn_addc)
        rule_btns.addWidget(btn_delc)
        rule_btns.addStretch()
        fl.addLayout(rule_btns)
        fl.addWidget(rules_tbl)
        index = self.tabs.count() + 1
        self.tabs.addTab(tab, f"Condition {index}")

    def delete_condition(self, tab):
        """Remove a condition tab."""
        idx = self.tabs.indexOf(tab)
        if idx != -1:
            self.tabs.removeTab(idx)
        if self.tabs.count() == 0:
            self.add_condition()

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
        rename_map = {}
        for i in range(self.preview_tree.topLevelItemCount()):
            it = self.preview_tree.topLevelItem(i)
            orig_rel = Path(it.text(0))
            orig = self.bids_root / orig_rel
            new = orig.with_name(it.text(1))
            orig.rename(new)
            rename_map[orig_rel.as_posix()] = (
                (orig_rel.parent / it.text(1)).as_posix()
            )
        if rename_map:
            try:
                from .scans_utils import update_scans_with_map

                update_scans_with_map(self.bids_root, rename_map)
            except Exception:
                pass
        QMessageBox.information(self, "Batch Remap", "Rename applied.")
        self.accept()

class IntendedForDialog(QDialog):
    """Manual editor for fieldmap IntendedFor lists."""

    def __init__(self, parent, bids_root: Path):
        super().__init__(parent)
        self.setWindowTitle("Set IntendedFor")
        self.resize(900, 500)
        self.bids_root = bids_root

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        (
            self.bold_tab,
            self.bold_tree,
            self.bold_intended,
            self.bold_func_list,
            self.bold_remove,
            self.bold_add,
            self.bold_save,
        ) = self._build_tab("bold")
        self.tabs.addTab(self.bold_tab, "BOLD")

        (
            self.dwi_tab,
            self.dwi_tree,
            self.dwi_intended,
            self.dwi_func_list,
            self.dwi_remove,
            self.dwi_add,
            self.dwi_save,
        ) = self._build_tab("dwi")

        self.b0_box = QCheckBox("Treat DWI b0 maps as fieldmaps")
        self.b0_box.toggled.connect(self._on_b0_toggle)
        layout.addWidget(self.b0_box)

        self.data = {}
        self._init_b0_state()
        self._collect()

    # ---- helpers ----
    def _build_tab(self, mode: str):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Fieldmap images:"))
        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        if mode == "bold":
            tree.itemSelectionChanged.connect(lambda: self._on_left_selected("bold"))
        else:
            tree.itemSelectionChanged.connect(lambda: self._on_left_selected("dwi"))
        left_layout.addWidget(tree)
        layout.addLayout(left_layout, 2)

        mid_layout = QVBoxLayout()
        mid_layout.addWidget(QLabel("IntendedFor:"))
        intended = QListWidget()
        intended.setSelectionMode(QAbstractItemView.ExtendedSelection)
        mid_layout.addWidget(intended)
        rm_save = QHBoxLayout()
        remove = QPushButton("Remove")
        remove.clicked.connect(lambda: self._remove_selected(mode))
        save = QPushButton("Save")
        save.clicked.connect(lambda: self._save_changes(mode))
        rm_save.addWidget(remove)
        rm_save.addWidget(save)
        rm_save.addStretch()
        mid_layout.addLayout(rm_save)
        layout.addLayout(mid_layout, 2)

        right_layout = QVBoxLayout()
        label = "Functional images:" if mode == "bold" else "Diffusion images:"
        right_layout.addWidget(QLabel(label))
        func_list = QListWidget()
        func_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        right_layout.addWidget(func_list)
        add_btn = QPushButton("← Add")
        add_btn.clicked.connect(lambda: self._add_selected(mode))
        right_layout.addWidget(add_btn)
        right_layout.addStretch()
        layout.addLayout(right_layout, 2)

        return widget, tree, intended, func_list, remove, add_btn, save

    def _init_b0_state(self) -> None:
        has_b0 = self._has_any_b0()
        fmap_b0 = self._fmap_has_b0()
        self.b0_box.setEnabled(has_b0)
        self.b0_box.setChecked(fmap_b0)
        if fmap_b0 and self.tabs.indexOf(self.dwi_tab) == -1:
            self.tabs.addTab(self.dwi_tab, "DWI")

    def _on_b0_toggle(self) -> None:
        if self.b0_box.isChecked():
            if self.tabs.indexOf(self.dwi_tab) == -1:
                self.tabs.addTab(self.dwi_tab, "DWI")
        else:
            idx = self.tabs.indexOf(self.dwi_tab)
            if idx != -1:
                self.tabs.removeTab(idx)
        self._collect()

    def _collect(self) -> None:
        self.bold_tree.clear()
        if self.tabs.indexOf(self.dwi_tab) != -1:
            self.dwi_tree.clear()
        self.data.clear()
        if self.b0_box.isChecked():
            self._move_b0_maps()
        else:
            self._restore_b0_maps()
        for sub in sorted(self.bids_root.glob('sub-*')):
            if not sub.is_dir():
                continue
            sub_item_bold = QTreeWidgetItem([sub.name])
            self.bold_tree.addTopLevelItem(sub_item_bold)
            sub_item_dwi = None
            if self.tabs.indexOf(self.dwi_tab) != -1:
                sub_item_dwi = QTreeWidgetItem([sub.name])
                self.dwi_tree.addTopLevelItem(sub_item_dwi)
            sessions = [s for s in sub.glob('ses-*') if s.is_dir()]
            if sessions:
                for ses in sessions:
                    ses_item_bold = QTreeWidgetItem([ses.name])
                    sub_item_bold.addChild(ses_item_bold)
                    ses_item_dwi = None
                    if sub_item_dwi is not None:
                        ses_item_dwi = QTreeWidgetItem([ses.name])
                        sub_item_dwi.addChild(ses_item_dwi)
                    self._add_fmaps(ses, ses_item_bold, ses_item_dwi, sub.name, ses.name)
            else:
                self._add_fmaps(sub, sub_item_bold, sub_item_dwi, sub.name, None)
            sub_item_bold.setExpanded(True)
            if sub_item_dwi is not None:
                sub_item_dwi.setExpanded(True)

    def _move_b0_maps(self) -> None:
        """Move DWI b0/epi images into the ``fmap`` folder."""
        rename_map: dict[str, str] = {}
        for sub in self.bids_root.glob('sub-*'):
            if not sub.is_dir():
                continue
            sessions = [s for s in sub.glob('ses-*') if s.is_dir()]
            roots = sessions or [sub]
            for root in roots:
                dwi_dir = root / 'dwi'
                if not dwi_dir.is_dir():
                    continue
                fmap_dir = root / 'fmap'
                fmap_dir.mkdir(exist_ok=True)
                for nii in dwi_dir.glob('*.nii*'):
                    name = nii.name.lower()
                    if 'b0' not in name and '_epi' not in name:
                        continue
                    dst = fmap_dir / nii.name
                    if not dst.exists():
                        nii.rename(dst)
                        rename_map[(nii.relative_to(self.bids_root)).as_posix()] = (
                            dst.relative_to(self.bids_root).as_posix()
                        )
                    base = re.sub(r'\.nii(\.gz)?$', '', nii.name, flags=re.I)
                    for ext in ['.json', '.bval', '.bvec']:
                        src = dwi_dir / (base + ext)
                        if src.exists():
                            dst_file = fmap_dir / src.name
                            if not dst_file.exists():
                                src.rename(dst_file)
                                rename_map[(src.relative_to(self.bids_root)).as_posix()] = (
                                    dst_file.relative_to(self.bids_root).as_posix()
                                )
        if rename_map:
            try:
                from .scans_utils import update_scans_with_map

                update_scans_with_map(self.bids_root, rename_map)
            except Exception:
                pass

    def _restore_b0_maps(self) -> None:
        """Move previously relocated b0/epi images back to ``dwi``."""
        rename_map: dict[str, str] = {}
        for sub in self.bids_root.glob('sub-*'):
            if not sub.is_dir():
                continue
            sessions = [s for s in sub.glob('ses-*') if s.is_dir()]
            roots = sessions or [sub]
            for root in roots:
                fmap_dir = root / 'fmap'
                dwi_dir = root / 'dwi'
                if not fmap_dir.is_dir() or not dwi_dir.is_dir():
                    continue
                for nii in fmap_dir.glob('*.nii*'):
                    name = nii.name.lower()
                    if 'b0' not in name and '_epi' not in name:
                        continue
                    dst = dwi_dir / nii.name
                    if not dst.exists():
                        nii.rename(dst)
                        rename_map[(nii.relative_to(self.bids_root)).as_posix()] = (
                            dst.relative_to(self.bids_root).as_posix()
                        )
                    base = re.sub(r'\.nii(\.gz)?$', '', nii.name, flags=re.I)
                    for ext in ['.json', '.bval', '.bvec']:
                        src = fmap_dir / (base + ext)
                        if src.exists():
                            dst_file = dwi_dir / src.name
                            if not dst_file.exists():
                                src.rename(dst_file)
                                rename_map[(src.relative_to(self.bids_root)).as_posix()] = (
                                    dst_file.relative_to(self.bids_root).as_posix()
                                )
        if rename_map:
            try:
                from .scans_utils import update_scans_with_map

                update_scans_with_map(self.bids_root, rename_map)
            except Exception:
                pass

    def _has_any_b0(self) -> bool:
        for sub in self.bids_root.glob('sub-*'):
            if not sub.is_dir():
                continue
            sessions = [s for s in sub.glob('ses-*') if s.is_dir()]
            roots = sessions or [sub]
            for root in roots:
                for folder in [root / 'dwi', root / 'fmap']:
                    if not folder.is_dir():
                        continue
                    for nii in folder.glob('*.nii*'):
                        if self._is_b0(nii.name):
                            return True
        return False

    def _fmap_has_b0(self) -> bool:
        for sub in self.bids_root.glob('sub-*'):
            if not sub.is_dir():
                continue
            sessions = [s for s in sub.glob('ses-*') if s.is_dir()]
            roots = sessions or [sub]
            for root in roots:
                fmap_dir = root / 'fmap'
                if not fmap_dir.is_dir():
                    continue
                for nii in fmap_dir.glob('*.nii*'):
                    if self._is_b0(nii.name):
                        return True
        return False

    @staticmethod
    def _is_b0(name: str) -> bool:
        lower = name.lower()
        return 'b0' in lower or '_epi' in lower

    def _add_fmaps(self, root: Path, bold_parent: QTreeWidgetItem,
                   dwi_parent: QTreeWidgetItem | None,
                   sub: str, ses: str | None) -> None:
        fmap_dir = root / 'fmap'
        func_dir = root / 'func'
        dwi_dir = root / 'dwi'
        func_files = [f.relative_to(root).as_posix()
                      for f in sorted(func_dir.glob('*.nii*')) if f.is_file()]
        dwi_files = [f.relative_to(root).as_posix()
                     for f in sorted(dwi_dir.glob('*.nii*')) if f.is_file()]
        groups: dict[str, list[Path]] = {}
        if fmap_dir.is_dir():
            for js in fmap_dir.glob('*.json'):
                base = re.sub(
                    r'_(magnitude1|magnitude2|phasediff|phase1|phase2)\.json$',
                    '', js.name, flags=re.I)
                groups.setdefault(base, []).append(js)
        for base, files in groups.items():
            key = (sub, ses, base)
            bold_int, dwi_int = self._load_intended(files[0], root)
            self.data[key] = {
                'jsons': files,
                'funcs_bold': func_files,
                'funcs_dwi': dwi_files,
                'intended_bold': bold_int,
                'intended_dwi': dwi_int,
                'root': root,
            }
            if not (self.b0_box.isChecked() and self._is_b0(base)):
                item_bold = QTreeWidgetItem([base])
                item_bold.setData(0, Qt.UserRole, key)
                bold_parent.addChild(item_bold)
            if dwi_parent is not None and self._is_b0(base):
                item_dwi = QTreeWidgetItem([base])
                item_dwi.setData(0, Qt.UserRole, key)
                dwi_parent.addChild(item_dwi)

    def _load_intended(self, path: Path, root: Path) -> tuple[list[str], list[str]]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            val = meta.get('IntendedFor', [])
            prefix = root.relative_to(self.bids_root)

            def _strip(p: str) -> str:
                parts = Path(p)
                try:
                    parts = parts.relative_to(prefix)
                except ValueError:
                    pass
                return parts.as_posix()

            if isinstance(val, str):
                vals = [_strip(val)]
            elif isinstance(val, list):
                vals = [_strip(v) for v in val]
            else:
                vals = []
            bold = [v for v in vals if '/func/' in v]
            dwi = [v for v in vals if '/dwi/' in v]
            return bold, dwi
        except Exception:
            pass
        return [], []

    def _on_left_selected(self, mode: str) -> None:
        tree = self.bold_tree if mode == "bold" else self.dwi_tree
        intended = self.bold_intended if mode == "bold" else self.dwi_intended
        funcs = self.bold_func_list if mode == "bold" else self.dwi_func_list
        it = tree.currentItem()
        if not it:
            return
        key = it.data(0, Qt.UserRole)
        if not key:
            return
        info = self.data.get(key, {})
        intended.clear()
        for f in info.get(f'intended_{mode}', []):
            intended.addItem(f)
        funcs.clear()
        for f in info.get(f'funcs_{mode}', []):
            funcs.addItem(f)

    def _add_selected(self, mode: str) -> None:
        tree = self.bold_tree if mode == "bold" else self.dwi_tree
        func_list = self.bold_func_list if mode == "bold" else self.dwi_func_list
        it = tree.currentItem()
        if not it:
            return
        key = it.data(0, Qt.UserRole)
        if not key:
            return
        info = self.data[key]
        for sel in func_list.selectedItems():
            path = sel.text()
            if path not in info[f'intended_{mode}']:
                info[f'intended_{mode}'].append(path)
        self._on_left_selected(mode)

    def _remove_selected(self, mode: str) -> None:
        tree = self.bold_tree if mode == "bold" else self.dwi_tree
        intended = self.bold_intended if mode == "bold" else self.dwi_intended
        it = tree.currentItem()
        if not it:
            return
        key = it.data(0, Qt.UserRole)
        if not key:
            return
        info = self.data[key]
        remove = [s.text() for s in intended.selectedItems()]
        info[f'intended_{mode}'] = [p for p in info[f'intended_{mode}'] if p not in remove]
        self._on_left_selected(mode)

    def _save_changes(self, mode: str) -> None:
        tree = self.bold_tree if mode == "bold" else self.dwi_tree
        it = tree.currentItem()
        if not it:
            return
        key = it.data(0, Qt.UserRole)
        if not key:
            return
        info = self.data[key]
        val = sorted(info['intended_bold'] + info['intended_dwi'])
        prefix = info['root'].relative_to(self.bids_root)
        cleaned = []
        for p in val:
            path = Path(p)
            try:
                path = path.relative_to(prefix)
            except ValueError:
                pass
            cleaned.append(path.as_posix())
        val = cleaned
        for js in info['jsons']:
            try:
                with open(js, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                meta['IntendedFor'] = val
                with open(js, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=4)
                    f.write('\n')
            except Exception as exc:
                QMessageBox.warning(self, 'Error', f'Failed to save {js}: {exc}')
                return
        QMessageBox.information(self, 'Saved', 'IntendedFor updated.')

class BidsIgnoreDialog(QDialog):
    """Dialog to edit ``.bidsignore`` entries using two selection panels."""

    def __init__(self, parent, bids_root: Path):
        super().__init__(parent)
        self.bids_root = bids_root
        self.setWindowTitle("Edit .bidsignore")
        self.resize(700, 400)

        main = QVBoxLayout(self)

        # --- lists ---
        lists = QHBoxLayout()
        main.addLayout(lists)

        # Left panel: existing entries
        left_box = QVBoxLayout()
        lists.addLayout(left_box)
        left_box.addWidget(QLabel("Ignored files:"))
        self.ignore_list = QListWidget()
        self.ignore_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        left_box.addWidget(self.ignore_list)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._remove_selected)
        left_box.addWidget(rm_btn)

        # Right panel: available files
        right_box = QVBoxLayout()
        lists.addLayout(right_box)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter files…")
        self.search.textChanged.connect(self._populate_lists)
        right_box.addWidget(self.search)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        right_box.addWidget(self.file_list)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_selected)
        right_box.addWidget(add_btn)

        # buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.save)
        btn_box.rejected.connect(self.reject)
        main.addWidget(btn_box)

        self.ignore_file = self.bids_root / ".bidsignore"
        self.entries: set[str] = set()
        if self.ignore_file.exists():
            self.entries = {
                line.strip()
                for line in self.ignore_file.read_text().splitlines()
                if line.strip()
            }

        self.all_files = [
            p.relative_to(self.bids_root).as_posix()
            for p in self.bids_root.rglob('*')
            if p.is_file()
        ]
        self._populate_lists()

    # --- helpers ---
    def _populate_lists(self) -> None:
        """Refresh both panels based on current entries and filter."""
        pattern = self.search.text().strip()

        self.ignore_list.clear()
        for path in sorted(self.entries):
            self.ignore_list.addItem(path)

        self.file_list.clear()
        for path in sorted(self.all_files):
            if path in self.entries:
                continue
            if pattern and pattern not in path:
                continue
            self.file_list.addItem(path)

    def _add_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.entries.add(item.text())
        self._populate_lists()

    def _remove_selected(self) -> None:
        for item in self.ignore_list.selectedItems():
            self.entries.discard(item.text())
        self._populate_lists()

    # --- save ---
    def save(self) -> None:
        self.ignore_file.write_text("\n".join(sorted(self.entries)) + "\n")
        QMessageBox.information(self, "Saved", f"Updated {self.ignore_file}")
        self.accept()

class Volume3DDialog(QDialog):
    """Interactive 3-D renderer for NIfTI volumes."""

    def __init__(
        self,
        parent,
        data: np.ndarray,
        meta: Optional[dict] = None,
        voxel_sizes: Optional[Sequence[float]] = None,
        default_mode: Optional[str] = None,
        title: Optional[str] = None,
        dark_theme: bool = False,
    ) -> None:
        super().__init__(parent)
        if not HAS_PYQTGRAPH:
            raise RuntimeError(
                "3-D volume rendering requires the optional 'pyqtgraph' dependency."
            )
        self.setWindowTitle(title or "3D Volume Viewer")
        self.resize(1800, 1000)
        self.setMinimumSize(780, 560)
        self.setSizeGripEnabled(True)

        self._raw = np.asarray(data)
        if self._raw.ndim < 3:
            raise ValueError("3-D rendering requires data with at least three axes")

        self._meta = meta or {}
        self._voxel_sizes = self._normalise_voxel_sizes(voxel_sizes)
        self._voxel_sizes_vec = np.asarray(self._voxel_sizes, dtype=np.float32)
        self._dark_theme = bool(dark_theme)
        self._max_points = 120_000
        self._surface_step = 1
        self._initialising = True
        self._scalar_min = 0.0
        self._scalar_max = 0.0
        self._scalar_volume: Optional[np.ndarray] = None
        self._normalised_volume: Optional[np.ndarray] = None
        self._downsampled: Optional[np.ndarray] = None
        self._downsample_step = 1
        self._point_sorted_values: Optional[np.ndarray] = None
        self._point_sorted_indices: Optional[np.ndarray] = None
        self._point_shape: Optional[tuple[int, int, int]] = None
        self._scalar_hist_upper_edges: Optional[np.ndarray] = None
        self._scalar_hist_cumulative: Optional[np.ndarray] = None
        self._scatter_item: Optional[gl.GLScatterPlotItem] = None
        self._mesh_item: Optional[gl.GLMeshItem] = None
        self._axis_item: Optional[gl.GLAxisItem] = None
        self._current_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._colormap_cache: dict[tuple[str, float], mcolors.Colormap] = {}
        self._light_shader = _create_directional_light_shader()
        self._flat_shader = _create_flat_color_shader()
        self._lighting_enabled = True

        self._fg_color = "#f0f0f0" if self._dark_theme else "#202020"
        self._canvas_bg = "#202020" if self._dark_theme else "#ffffff"
        self._axis_label_items: list[gl.GLTextItem] = []
        self._data_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._slice_controls: dict[str, _SliceControl] = {}

        layout = QVBoxLayout(self)

        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(12)
        layout.addWidget(self._splitter)

        self._view_container = QWidget()
        view_layout = QVBoxLayout(self._view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        # ``GLViewWidget`` renders using OpenGL so panning/zooming the scene does
        # not require recomputing the voxel subset on every interaction.
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setBackgroundColor(self._canvas_bg)
        self.view.opts["distance"] = 200
        self.view.opts["elevation"] = 20
        self.view.opts["azimuth"] = -60
        view_layout.addWidget(self.view)

        self._splitter.addWidget(self._view_container)

        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(6, 6, 6, 6)
        settings_layout.setSpacing(10)

        # Allow users to dock the settings pane to whichever edge feels most
        # comfortable when working on small screens.
        placement_layout = QHBoxLayout()
        placement_layout.setSpacing(6)
        placement_label = QLabel("Panel placement:")
        placement_layout.addWidget(placement_label)
        self.panel_location_combo = QComboBox()
        self.panel_location_combo.addItems(["Bottom", "Left", "Right"])
        # Default to a side-by-side layout so the viewport opens on the left.
        self.panel_location_combo.setCurrentText("Right")
        placement_layout.addWidget(self.panel_location_combo)
        placement_layout.addStretch()
        settings_layout.addLayout(placement_layout)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        agg_label = QLabel("Aggregation:")
        header_row.addWidget(agg_label)
        self.agg_combo = QComboBox()
        self._aggregators = {}
        self._init_aggregator_options()
        header_row.addWidget(self.agg_combo)
        header_row.addSpacing(12)

        render_label = QLabel("Render mode:")
        header_row.addWidget(render_label)
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["Point cloud", "Surface mesh"])
        self.render_mode_combo.setToolTip(
            "Switch between scattered voxels and iso-surface rendering.",
        )
        header_row.addWidget(self.render_mode_combo)
        header_row.addStretch()
        controls_layout.addLayout(header_row)

        # Stack the render-mode sliders beneath the mode selector so they are
        # visually associated with the chosen representation.
        slider_grid = QGridLayout()
        slider_grid.setContentsMargins(0, 0, 0, 0)
        slider_grid.setHorizontalSpacing(8)
        slider_grid.setVerticalSpacing(6)
        slider_grid.setColumnStretch(1, 1)

        self.thresh_text = QLabel("Threshold:")
        slider_grid.addWidget(self.thresh_text, 0, 0)
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(60)
        self.thresh_slider.setPageStep(5)
        self.thresh_slider.setToolTip(
            "Minimum normalised intensity included in the rendering.",
        )
        self.thresh_slider.valueChanged.connect(self._update_plot)
        slider_grid.addWidget(self.thresh_slider, 0, 1)
        self.thresh_label = QLabel("0.60")
        slider_grid.addWidget(self.thresh_label, 0, 2)

        self.point_label = QLabel("Point size:")
        slider_grid.addWidget(self.point_label, 1, 0)
        self.point_slider = QSlider(Qt.Horizontal)
        self.point_slider.setRange(1, 12)
        self.point_slider.setValue(4)
        self.point_slider.setToolTip("Marker diameter for point-cloud rendering.")
        self.point_slider.valueChanged.connect(self._update_plot)
        slider_grid.addWidget(self.point_slider, 1, 1)

        controls_layout.addLayout(slider_grid)
        controls_layout.addStretch()
        settings_layout.addWidget(controls_widget)

        options_group = QGroupBox("Rendering options")
        options = QGridLayout(options_group)
        options.setColumnStretch(1, 1)

        row = 0
        options.addWidget(QLabel("Colormap:"), row, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "twilight",
            "cubehelix",
            "Spectral",
            "coolwarm",
            "YlGnBu",
            "Greys",
            "bone",
        ])
        self.colormap_combo.setToolTip(
            "Select the matplotlib colormap for voxel intensities.",
        )
        self.colormap_combo.currentTextChanged.connect(lambda _: self._update_plot())
        options.addWidget(self.colormap_combo, row, 1)

        row += 1
        options.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.setToolTip(
            "Global alpha applied to rendered points or the surface mesh.",
        )
        options.addWidget(self.opacity_slider, row, 1)
        self.opacity_label = QLabel("0.80")
        options.addWidget(self.opacity_label, row, 2)

        row += 1
        options.addWidget(QLabel("Colour intensity:"), row, 0)
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(10, 200)
        self.intensity_slider.setValue(100)
        self.intensity_slider.setToolTip(
            "Scale factor applied to RGB values after colormap lookup (0.1–2.0×).",
        )
        options.addWidget(self.intensity_slider, row, 1)
        self.intensity_label = QLabel("1.00×")
        options.addWidget(self.intensity_label, row, 2)

        row += 1
        options.addWidget(QLabel("Maximum points:"), row, 0)
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(1_000, 20_000_000)
        self.max_points_spin.setSingleStep(5_000)
        self.max_points_spin.setValue(self._max_points)
        self.max_points_spin.setToolTip(
            "Upper bound on voxels displayed in point-cloud mode.",
        )
        options.addWidget(self.max_points_spin, row, 1)

        row += 1
        options.addWidget(QLabel("Downsample step:"), row, 0)
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(0, 8)
        self.downsample_spin.setValue(0)
        self.downsample_spin.setToolTip(
            "Manual voxel stride applied before rendering (0 = automatic).",
        )
        options.addWidget(self.downsample_spin, row, 1)

        row += 1
        self.surface_step_label = QLabel("Marching cubes step:")
        options.addWidget(self.surface_step_label, row, 0)
        self.surface_step_spin = QSpinBox()
        self.surface_step_spin.setRange(1, 6)
        self.surface_step_spin.setValue(self._surface_step)
        self.surface_step_spin.setToolTip(
            "Sampling stride for marching cubes when computing iso-surfaces.",
        )
        options.addWidget(self.surface_step_spin, row, 1)

        row += 1
        self.axes_checkbox = QCheckBox("Show axes")
        self.axes_checkbox.setChecked(False)
        options.addWidget(self.axes_checkbox, row, 0, 1, 2)

        row += 1
        options.addWidget(QLabel("Axis thickness:"), row, 0)
        self.axis_thickness_slider = QSlider(Qt.Horizontal)
        self.axis_thickness_slider.setRange(1, 12)
        self.axis_thickness_slider.setValue(3)
        self.axis_thickness_slider.setToolTip(
            "Line width of the anatomical axes (pixels).",
        )
        options.addWidget(self.axis_thickness_slider, row, 1)
        self.axis_thickness_value = QLabel("3 px")
        options.addWidget(self.axis_thickness_value, row, 2)

        row += 1
        self.axis_labels_checkbox = QCheckBox("Show axis labels")
        self.axis_labels_checkbox.setChecked(False)
        options.addWidget(self.axis_labels_checkbox, row, 0, 1, 2)
        self.axis_thickness_slider.setEnabled(False)
        self.axis_labels_checkbox.setEnabled(False)
        settings_layout.addWidget(options_group)

        # Embed the colour bar alongside the rest of the controls so it folds
        # away with the settings pane.
        colorbar_group = QGroupBox("Colour bar")
        colorbar_layout = QVBoxLayout(colorbar_group)
        colorbar_layout.setContentsMargins(8, 6, 8, 6)
        colorbar_layout.setSpacing(4)
        self._colorbar_canvas = FigureCanvas(plt.Figure(figsize=(2.6, 1.4)))
        self._colorbar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._colorbar_canvas.setFixedHeight(180)
        self._colorbar_canvas.figure.patch.set_facecolor(self._canvas_bg)
        colorbar_layout.addWidget(self._colorbar_canvas)

        # Host the colour bar in its own lightweight container so it can share
        # the retractable pane while remaining independent from the parameter list.
        self._colorbar_panel = QWidget()
        colorbar_panel_layout = QVBoxLayout(self._colorbar_panel)
        colorbar_panel_layout.setContentsMargins(6, 6, 6, 6)
        colorbar_panel_layout.setSpacing(6)
        colorbar_panel_layout.addWidget(colorbar_group)

        slice_group = QGroupBox("Slice planes")
        slice_layout = QVBoxLayout(slice_group)
        slice_layout.setContentsMargins(8, 4, 8, 4)
        slice_layout.setSpacing(6)
        self._slice_controls.clear()
        for key, axis_idx, neg_name, pos_name in _SLICE_ORIENTATIONS:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(6)
            checkbox = QCheckBox(key.capitalize())
            checkbox.setChecked(False)
            row_layout.addWidget(checkbox)
            row_layout.addSpacing(4)
            row_layout.addWidget(QLabel(f"{neg_name}:"))
            min_slider = QSlider(Qt.Horizontal)
            min_slider.setRange(0, 100)
            min_slider.setValue(0)
            min_slider.setEnabled(False)
            row_layout.addWidget(min_slider, 1)
            min_value = QLabel("0.0 mm")
            row_layout.addWidget(min_value)
            row_layout.addSpacing(6)
            row_layout.addWidget(QLabel(f"{pos_name}:"))
            max_slider = QSlider(Qt.Horizontal)
            max_slider.setRange(0, 100)
            max_slider.setValue(100)
            max_slider.setEnabled(False)
            row_layout.addWidget(max_slider, 1)
            max_value = QLabel("0.0 mm")
            row_layout.addWidget(max_value)
            row_layout.addStretch()
            slice_layout.addLayout(row_layout)
            control = _SliceControl(
                checkbox=checkbox,
                min_slider=min_slider,
                max_slider=max_slider,
                min_value=min_value,
                max_value=max_value,
                axis=axis_idx,
                negative_name=neg_name,
                positive_name=pos_name,
            )
            self._slice_controls[key] = control
            checkbox.toggled.connect(lambda checked, name=key: self._on_slice_toggle(name, checked))
            min_slider.valueChanged.connect(lambda value, name=key: self._on_slice_slider_change(name, "min", value))
            max_slider.valueChanged.connect(lambda value, name=key: self._on_slice_slider_change(name, "max", value))
        slice_layout.addStretch()
        settings_layout.addWidget(slice_group)

        self.lighting_group = QGroupBox("Lighting")
        light_layout = QGridLayout(self.lighting_group)
        light_row = 0
        self.light_enable_checkbox = QCheckBox("Enable lighting")
        self.light_enable_checkbox.setChecked(True)
        light_layout.addWidget(self.light_enable_checkbox, light_row, 0, 1, 3)
        light_row += 1
        light_layout.addWidget(QLabel("Azimuth:"), light_row, 0)
        self.light_azimuth_slider = QSlider(Qt.Horizontal)
        self.light_azimuth_slider.setRange(-180, 180)
        self.light_azimuth_slider.setValue(-45)
        self.light_azimuth_slider.setToolTip(
            "Horizontal direction of the light source relative to the mesh (°).",
        )
        light_layout.addWidget(self.light_azimuth_slider, light_row, 1)
        self.light_azimuth_label = QLabel("-45°")
        light_layout.addWidget(self.light_azimuth_label, light_row, 2)

        light_row += 1
        light_layout.addWidget(QLabel("Elevation:"), light_row, 0)
        self.light_elevation_slider = QSlider(Qt.Horizontal)
        self.light_elevation_slider.setRange(-90, 90)
        self.light_elevation_slider.setValue(30)
        self.light_elevation_slider.setToolTip(
            "Vertical angle of the light source relative to the mesh (°).",
        )
        light_layout.addWidget(self.light_elevation_slider, light_row, 1)
        self.light_elevation_label = QLabel("30°")
        light_layout.addWidget(self.light_elevation_label, light_row, 2)

        light_row += 1
        light_layout.addWidget(QLabel("Intensity:"), light_row, 0)
        self.light_intensity_slider = QSlider(Qt.Horizontal)
        self.light_intensity_slider.setRange(10, 300)
        self.light_intensity_slider.setValue(130)
        self.light_intensity_slider.setToolTip(
            "Diffuse lighting strength (0.1–3.0×). Ambient light stays constant.",
        )
        light_layout.addWidget(self.light_intensity_slider, light_row, 1)
        self.light_intensity_label = QLabel("1.30×")
        light_layout.addWidget(self.light_intensity_label, light_row, 2)
        if self._light_shader is None and self._flat_shader is None:
            self.lighting_group.setEnabled(False)
        settings_layout.addWidget(self.lighting_group)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        settings_layout.addWidget(self.status_label)
        settings_layout.addStretch()

        self._panel_scroll = ShrinkableScrollArea()
        self._panel_scroll.setWidget(settings_container)
        self._panel_scroll.setWidgetResizable(True)
        self._panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # A nested splitter keeps the scrollable controls and the colour bar in
        # separate panes so each can be resized without affecting the other.
        self._panel_splitter = QSplitter(Qt.Vertical)
        self._panel_splitter.setChildrenCollapsible(False)
        self._panel_splitter.addWidget(self._panel_scroll)
        self._panel_splitter.addWidget(self._colorbar_panel)
        self._panel_splitter.setStretchFactor(0, 3)
        self._panel_splitter.setStretchFactor(1, 1)

        self._panel_container = QWidget()
        panel_container_layout = QVBoxLayout(self._panel_container)
        panel_container_layout.setContentsMargins(0, 0, 0, 0)
        panel_container_layout.setSpacing(0)
        panel_container_layout.addWidget(self._panel_splitter)

        self._splitter.addWidget(self._panel_container)

        default_name = None
        if default_mode and default_mode in self._aggregators:
            default_name = default_mode
        elif self._raw.ndim > 3:
            if self._meta.get("is_rgb"):
                default_name = "Mean |value|"
            else:
                default_name = "RMS (DTI)"

        if default_name:
            self.agg_combo.setCurrentText(default_name)

        self.panel_location_combo.currentTextChanged.connect(self._on_panel_location_change)
        self.agg_combo.currentTextChanged.connect(self._on_agg_change)
        self.render_mode_combo.currentTextChanged.connect(self._on_render_mode_change)
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        self.intensity_slider.valueChanged.connect(self._on_intensity_change)
        self.max_points_spin.valueChanged.connect(self._on_max_points_change)
        self.downsample_spin.valueChanged.connect(self._on_downsample_change)
        self.surface_step_spin.valueChanged.connect(self._on_surface_step_change)
        self.axes_checkbox.toggled.connect(self._on_axes_toggle)
        self.axis_thickness_slider.valueChanged.connect(self._on_axis_thickness_change)
        self.axis_labels_checkbox.toggled.connect(self._on_axis_labels_toggle)
        self.light_enable_checkbox.toggled.connect(self._on_light_enabled_toggle)
        self.light_azimuth_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_elevation_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_intensity_slider.valueChanged.connect(self._on_light_setting_change)

        self._apply_panel_layout(self.panel_location_combo.currentText())
        self._update_light_controls_enabled()

        # Prime the UI labels without triggering expensive redraws while we are
        # still constructing the dialog.
        self._on_intensity_change(self.intensity_slider.value())
        self._on_axis_thickness_change(self.axis_thickness_slider.value())
        self._update_slice_labels()
        self._update_light_labels()
        self._update_light_shader()

        self._initialising = False
        self._update_mode_dependent_controls()
        self._compute_scalar_volume()
        self._update_plot()

    def _apply_panel_layout(self, location: str) -> None:
        """Reposition the settings pane relative to the 3-D viewport."""

        if not hasattr(self, "_splitter"):
            return

        normalised = (location or "bottom").strip().lower()
        if normalised not in {"bottom", "left", "right"}:
            normalised = "bottom"

        orientation = Qt.Vertical if normalised == "bottom" else Qt.Horizontal
        self._splitter.setOrientation(orientation)

        view_first = True
        if orientation == Qt.Horizontal:
            view_first = normalised != "left"

        panel_widget = getattr(self, "_panel_container", None)
        widgets = [self._view_container, panel_widget]
        if panel_widget is None:
            widgets = [self._view_container]
        if not view_first and len(widgets) == 2:
            widgets.reverse()
        for index, widget in enumerate(widgets):
            if widget is None:
                continue
            if self._splitter.indexOf(widget) != index:
                self._splitter.insertWidget(index, widget)

        # Bias the main splitter so the viewport consumes roughly three quarters
        # of the available space in the default layout.
        stretch = (3, 1) if view_first else (1, 3)
        self._splitter.setStretchFactor(0, stretch[0])
        if len(widgets) > 1:
            self._splitter.setStretchFactor(1, stretch[1])
            scale = 120
            self._splitter.setSizes([stretch[0] * scale, stretch[1] * scale])

        panel_splitter = getattr(self, "_panel_splitter", None)
        if isinstance(panel_splitter, QSplitter):
            # Rotate the internal splitter so the colour bar always occupies its
            # own pane next to the parameter controls regardless of docking side.
            if orientation == Qt.Vertical:
                panel_splitter.setOrientation(Qt.Horizontal)
                panel_splitter.setSizes([220, 160])
            else:
                panel_splitter.setOrientation(Qt.Vertical)
                panel_splitter.setSizes([300, 160])

    def _on_panel_location_change(self, location: str) -> None:
        self._apply_panel_layout(location)

    def _update_light_controls_enabled(self) -> None:
        enabled = bool(self.lighting_group.isEnabled() and self._lighting_enabled)
        for widget in (
            self.light_azimuth_slider,
            self.light_elevation_slider,
            self.light_intensity_slider,
        ):
            widget.setEnabled(enabled)
        for label in (
            self.light_azimuth_label,
            self.light_elevation_label,
            self.light_intensity_label,
        ):
            label.setEnabled(enabled)

    def _apply_mesh_shader(self) -> None:
        if self._mesh_item is None:
            return
        if self._light_shader is not None and self._lighting_enabled:
            self._mesh_item.setShader(self._light_shader)
        elif self._flat_shader is not None:
            self._mesh_item.setShader(self._flat_shader)
        else:
            self._mesh_item.setShader("shaded")

    def _on_light_enabled_toggle(self, checked: bool) -> None:
        self._lighting_enabled = bool(checked)
        self._update_light_controls_enabled()
        if self._lighting_enabled:
            self._update_light_shader()
        else:
            self._apply_mesh_shader()
        if not self._initialising and self._mesh_item is not None:
            self._mesh_item.update()
            self.view.update()

    def _normalise_voxel_sizes(self, voxel_sizes):
        if not voxel_sizes:
            return (1.0, 1.0, 1.0)
        values = tuple(float(v) for v in voxel_sizes[:3])
        if len(values) < 3:
            values = values + (1.0,) * (3 - len(values))
        return tuple(max(1e-6, abs(v)) for v in values)

    def _init_aggregator_options(self) -> None:
        if self._raw.ndim <= 3:
            self._aggregators["Intensity"] = "Intensity"
            self.agg_combo.addItem("Intensity")
            self.agg_combo.setEnabled(False)
        else:
            for name in ("RMS (DTI)", "Mean |value|", "Max |value|", "First volume"):
                self._aggregators[name] = name
                self.agg_combo.addItem(name)

    def _on_agg_change(self):
        if self._initialising:
            return
        self._compute_scalar_volume()
        self._update_plot()

    def _compute_scalar_volume(self) -> None:
        arr = np.asarray(self._raw, dtype=np.float32)
        if arr.ndim <= 3:
            scalar = arr
        else:
            axis = arr.ndim - 1
            mode = self.agg_combo.currentText()
            if mode == "RMS (DTI)":
                scalar = np.sqrt(np.nanmean(np.square(arr), axis=axis))
            elif mode == "Max |value|":
                scalar = np.nanmax(np.abs(arr), axis=axis)
            elif mode == "First volume":
                scalar = np.abs(np.take(arr, 0, axis=axis))
            else:
                scalar = np.nanmean(np.abs(arr), axis=axis)
        scalar = np.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0)
        self._scalar_volume = scalar
        finite = np.isfinite(scalar)
        if not finite.any():
            self._scalar_min = 0.0
            self._scalar_max = 0.0
            self._normalised_volume = np.zeros_like(scalar, dtype=np.float32)
        else:
            min_val = float(np.min(scalar[finite]))
            max_val = float(np.max(scalar[finite]))
            self._scalar_min = min_val
            self._scalar_max = max_val
            if abs(max_val - min_val) < 1e-6:
                self._normalised_volume = np.zeros_like(scalar, dtype=np.float32)
            else:
                norm = (scalar - min_val) / (max_val - min_val)
                self._normalised_volume = norm.astype(np.float32, copy=False)
        self._build_scalar_histogram()
        self._prepare_downsampled()

    def _build_scalar_histogram(self) -> None:
        self._scalar_hist_upper_edges = None
        self._scalar_hist_cumulative = None
        volume = self._normalised_volume
        if volume is None:
            return

        flattened = np.asarray(volume, dtype=np.float32).ravel()
        if flattened.size == 0:
            self._scalar_hist_upper_edges = np.empty(0, dtype=np.float32)
            self._scalar_hist_cumulative = np.empty(0, dtype=np.int64)
            return

        edges = np.linspace(0.0, 1.0, 513, dtype=np.float32)
        counts, _ = np.histogram(flattened, bins=edges)
        cumulative = np.cumsum(counts[::-1], dtype=np.int64)[::-1]
        self._scalar_hist_upper_edges = edges[1:]
        self._scalar_hist_cumulative = cumulative

    def _prepare_downsampled(self) -> None:
        vol = self._normalised_volume
        if vol is None:
            self._downsampled = None
            self._downsample_step = 1
            self._data_bounds = None
            self._update_slice_labels()
            return
        manual_step = 0
        if hasattr(self, "downsample_spin"):
            manual_step = int(self.downsample_spin.value())
        if manual_step > 0:
            step = manual_step
        else:
            step = 1
            total = float(vol.size)
            if total > self._max_points:
                step = int(np.ceil(np.cbrt(total / self._max_points)))
        step = max(1, step)
        slices = tuple(slice(None, None, step) for _ in range(3))
        self._downsampled = vol[slices]
        self._downsample_step = step
        if self._downsampled.size == 0:
            self._data_bounds = None
        else:
            ds_shape = np.ones(3, dtype=np.float32)
            dims = min(self._downsampled.ndim, 3)
            ds_shape[:dims] = np.asarray(self._downsampled.shape[:dims], dtype=np.float32)
            scale = self._voxel_sizes_vec * float(step)
            spans = np.maximum((ds_shape - 1.0) * scale, 0.0)
            mins = np.zeros(3, dtype=np.float32)
            self._data_bounds = (mins, mins + spans)
        self._update_slice_labels()
        self._rebuild_point_cache()

    def _rebuild_point_cache(self) -> None:
        self._point_sorted_values = None
        self._point_sorted_indices = None
        self._point_shape = None
        downsampled = self._downsampled
        if downsampled is None:
            return

        flat = np.asarray(downsampled, dtype=np.float32).ravel()
        self._point_shape = downsampled.shape
        if flat.size == 0:
            self._point_sorted_values = np.empty(0, dtype=np.float32)
            self._point_sorted_indices = np.empty(0, dtype=np.int64)
            return

        order = np.argsort(flat, kind="mergesort")
        self._point_sorted_indices = order.astype(np.int64, copy=False)
        self._point_sorted_values = flat.take(order)

    def _indices_to_world(
        self, indices: np.ndarray, shape: tuple[int, int, int], scale: np.ndarray
    ) -> np.ndarray:
        coords_mm = np.empty((indices.size, 3), dtype=np.float32)
        if indices.size == 0:
            return coords_mm

        plane = int(shape[1]) * int(shape[2])
        axis0 = indices // plane
        remainder = indices - axis0 * plane
        axis1 = remainder // int(shape[2])
        axis2 = remainder - axis1 * int(shape[2])

        coords_mm[:, 0] = axis0.astype(np.float32, copy=False)
        coords_mm[:, 1] = axis1.astype(np.float32, copy=False)
        coords_mm[:, 2] = axis2.astype(np.float32, copy=False)
        coords_mm *= scale
        return coords_mm

    def _estimate_voxels_above_threshold(self, thr: float) -> int:
        if self._scalar_hist_upper_edges is None or self._scalar_hist_cumulative is None:
            if self._normalised_volume is None:
                return 0
            return int(np.count_nonzero(self._normalised_volume >= thr))

        idx = int(np.searchsorted(self._scalar_hist_upper_edges, thr, side="left"))
        if idx >= self._scalar_hist_cumulative.size:
            return 0
        return int(self._scalar_hist_cumulative[idx])

    def _current_color_intensity(self) -> float:
        slider = getattr(self, "intensity_slider", None)
        if slider is None:
            return 1.0
        return max(0.1, slider.value() / 100.0)

    def _get_adjusted_colormap(self, cmap_name: str) -> mcolors.Colormap:
        intensity = round(self._current_color_intensity(), 3)
        key = (cmap_name, intensity)
        cached = self._colormap_cache.get(key)
        if cached is not None:
            return cached
        base = plt.get_cmap(cmap_name)
        samples = base(np.linspace(0.0, 1.0, 512))
        samples[:, :3] = np.clip(samples[:, :3] * intensity, 0.0, 1.0)
        cmap = mcolors.ListedColormap(samples, name=f"{cmap_name}_x{intensity:.3f}")
        self._colormap_cache[key] = cmap
        return cmap

    def _map_colors(
        self, values: np.ndarray, cmap: mcolors.Colormap, alpha: float
    ) -> np.ndarray:
        # Clamp to the normalised range and map to RGBA so OpenGL can upload a
        # single colour array for all voxels.
        colours = np.asarray(cmap(np.clip(values, 0.0, 1.0)), dtype=np.float32)
        colours[:, 3] = np.clip(colours[:, 3] * alpha, 0.0, 1.0)
        return colours

    def _update_slice_labels(self) -> None:
        if not self._slice_controls:
            return
        bounds = self._data_bounds
        has_bounds = bounds is not None
        mins: Optional[np.ndarray]
        maxs: Optional[np.ndarray]
        if has_bounds:
            mins, maxs = bounds  # type: ignore[assignment]
        else:
            mins = maxs = None
        for control in self._slice_controls.values():
            slider_enabled = bool(has_bounds and control.checkbox.isChecked())
            control.min_slider.setEnabled(slider_enabled)
            control.max_slider.setEnabled(slider_enabled)
            if mins is None or maxs is None:
                control.min_value.setText("--")
                control.max_value.setText("--")
                continue
            axis = control.axis
            axis_min = float(mins[axis])
            axis_max = float(maxs[axis])
            span = axis_max - axis_min
            min_frac = control.min_slider.value() / 100.0
            max_frac = control.max_slider.value() / 100.0
            min_mm = axis_min + span * min_frac
            max_mm = axis_min + span * max_frac
            control.min_value.setText(f"{min_mm:.1f} mm")
            control.max_value.setText(f"{max_mm:.1f} mm")

    def _active_slice_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not self._slice_controls or self._data_bounds is None:
            return None
        mins, maxs = self._data_bounds
        min_bounds = np.asarray(mins, dtype=np.float32).copy()
        max_bounds = np.asarray(maxs, dtype=np.float32).copy()
        active = False
        for control in self._slice_controls.values():
            if not control.checkbox.isChecked():
                continue
            axis = control.axis
            axis_min = float(mins[axis])
            axis_max = float(maxs[axis])
            span = axis_max - axis_min
            min_frac = control.min_slider.value() / 100.0
            max_frac = control.max_slider.value() / 100.0
            min_bounds[axis] = float(axis_min + span * min_frac)
            max_bounds[axis] = float(axis_min + span * max_frac)
            active = True
        if not active:
            return None
        return min_bounds, max_bounds

    def _slice_status_suffix(self) -> str:
        bounds = self._active_slice_bounds()
        if bounds is None:
            return ""
        mins, maxs = bounds
        parts: list[str] = []
        for control in self._slice_controls.values():
            if not control.checkbox.isChecked():
                continue
            axis = control.axis
            parts.append(
                f"{control.negative_name}–{control.positive_name} "
                f"{mins[axis]:.1f}–{maxs[axis]:.1f} mm"
            )
        if not parts:
            return ""
        return " Slice windows: " + ", ".join(parts) + "."

    def _mask_coords(
        self, coords: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        mins, maxs = bounds
        if coords.size == 0:
            return np.zeros(0, dtype=bool)
        mask = np.ones(coords.shape[0], dtype=bool)
        for axis in range(3):
            mask &= (coords[:, axis] >= mins[axis]) & (coords[:, axis] <= maxs[axis])
        return mask

    def _apply_slice_to_points(
        self, coords: np.ndarray, values: Optional[np.ndarray]
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[tuple[np.ndarray, np.ndarray]]]:
        bounds = self._active_slice_bounds()
        if bounds is None or coords.size == 0:
            return coords, values, None
        mask = self._mask_coords(coords, bounds)
        if not np.any(mask):
            empty_coords = coords[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_coords, empty_vals, bounds
        filtered_coords = coords[mask]
        filtered_values = values[mask] if values is not None else None
        return filtered_coords, filtered_values, bounds

    def _clip_mesh_to_slices(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        values: Optional[np.ndarray],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[tuple[np.ndarray, np.ndarray]],
    ]:
        bounds = self._active_slice_bounds()
        if bounds is None or verts.size == 0:
            return verts, faces, values, None
        mask = self._mask_coords(verts, bounds)
        if not np.any(mask):
            empty_verts = verts[:0].reshape(0, 3)
            empty_faces = faces[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_verts, empty_faces, empty_vals, bounds
        selected = np.where(mask)[0]
        index_map = -np.ones(mask.shape[0], dtype=np.int64)
        index_map[selected] = np.arange(selected.size, dtype=np.int64)
        mapped_faces = index_map[faces]
        valid = (mapped_faces >= 0).all(axis=1)
        mapped_faces = mapped_faces[valid]
        if mapped_faces.size == 0:
            empty_verts = verts[:0].reshape(0, 3)
            empty_faces = faces[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_verts, empty_faces, empty_vals, bounds
        new_verts = verts[selected]
        new_values = values[selected] if values is not None else None
        return new_verts, mapped_faces.astype(np.int32, copy=False), new_values, bounds

    def _on_slice_toggle(self, name: str, _checked: bool) -> None:
        if name not in self._slice_controls:
            return
        self._update_slice_labels()
        if not self._initialising:
            self._update_plot()

    def _on_slice_slider_change(self, name: str, which: str, value: int) -> None:
        control = self._slice_controls.get(name)
        if control is None:
            return
        if which == "min":
            other = control.max_slider
            if value > other.value():
                other.blockSignals(True)
                other.setValue(value)
                other.blockSignals(False)
        else:
            other = control.min_slider
            if value < other.value():
                other.blockSignals(True)
                other.setValue(value)
                other.blockSignals(False)
        self._update_slice_labels()
        if not self._initialising and control.checkbox.isChecked():
            self._update_plot()

    def _remove_axis_labels(self) -> None:
        if not self._axis_label_items:
            return
        for item in self._axis_label_items:
            try:
                self.view.removeItem(item)
            except Exception:  # pragma: no cover - defensive cleanup
                continue
        self._axis_label_items.clear()

    def _update_axis_labels(self) -> None:
        self._remove_axis_labels()
        if (
            getattr(self, "axis_labels_checkbox", None) is None
            or not self.axes_checkbox.isChecked()
            or not self.axis_labels_checkbox.isChecked()
            or self._current_bounds is None
        ):
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-6)
        safe_spans = np.where(spans > 0, spans, 1.0)
        centre = (mins + maxs) / 2.0
        offsets = safe_spans * 0.05
        colour = QColor(self._fg_color)
        font = QFont("Helvetica", 13)

        for _name, axis_idx, neg_name, pos_name in _SLICE_ORIENTATIONS:
            if axis_idx == 0:
                neg_pos = [
                    float(mins[0] - offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(maxs[0] + offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
            elif axis_idx == 1:
                neg_pos = [
                    float(centre[0]),
                    float(mins[1] - offsets[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(maxs[1] + offsets[1]),
                    float(centre[2]),
                ]
            else:
                neg_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(mins[2] - offsets[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(maxs[2] + offsets[2]),
                ]
            for text, position in ((neg_name, neg_pos), (pos_name, pos_pos)):
                label = gl.GLTextItem()
                label.setData(text=text, pos=position, color=colour, font=font)
                self.view.addItem(label)
                self._axis_label_items.append(label)

    def _update_axis_item(self) -> None:
        if self._axis_item is not None:
            self.view.removeItem(self._axis_item)
            self._axis_item = None
        if not self.axes_checkbox.isChecked() or self._current_bounds is None:
            self._remove_axis_labels()
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-3)
        axis = _AdjustableAxisItem() if "_AdjustableAxisItem" in globals() else gl.GLAxisItem()
        axis.setSize(spans[0], spans[1], spans[2])
        axis.translate(float(mins[0]), float(mins[1]), float(mins[2]))
        thickness = getattr(self, "axis_thickness_slider", None)
        if thickness is not None and hasattr(axis, "setLineWidth"):
            axis.setLineWidth(float(thickness.value()))
        self.view.addItem(axis)
        self._axis_item = axis
        self._update_axis_labels()

    def _on_axes_toggle(self, checked: bool) -> None:
        slider = getattr(self, "axis_thickness_slider", None)
        if slider is not None:
            slider.setEnabled(checked)
        label_cb = getattr(self, "axis_labels_checkbox", None)
        if label_cb is not None:
            label_cb.setEnabled(checked)
            if not checked:
                self._remove_axis_labels()
        self._update_axis_item()

    def _on_axis_thickness_change(self, value: int) -> None:
        if hasattr(self, "axis_thickness_value"):
            self.axis_thickness_value.setText(f"{value} px")
        if self._axis_item is not None and hasattr(self._axis_item, "setLineWidth"):
            self._axis_item.setLineWidth(float(value))
            self.view.update()

    def _on_axis_labels_toggle(self, checked: bool) -> None:
        if not checked:
            self._remove_axis_labels()
        else:
            self._update_axis_labels()

    def _on_intensity_change(self, value: int) -> None:
        self.intensity_label.setText(f"{value / 100.0:.2f}×")
        if not self._initialising:
            self._update_plot()

    def _update_light_labels(self) -> None:
        self.light_azimuth_label.setText(f"{self.light_azimuth_slider.value():+d}°")
        self.light_elevation_label.setText(
            f"{self.light_elevation_slider.value():+d}°"
        )
        self.light_intensity_label.setText(
            f"{self.light_intensity_slider.value() / 100.0:.2f}×"
        )

    def _update_light_shader(self) -> None:
        shader = self._light_shader
        if shader is None or not self._lighting_enabled:
            self._apply_mesh_shader()
            return
        azimuth = math.radians(self.light_azimuth_slider.value())
        elevation = math.radians(self.light_elevation_slider.value())
        cos_el = math.cos(elevation)
        direction = [
            float(cos_el * math.cos(azimuth)),
            float(cos_el * math.sin(azimuth)),
            float(math.sin(elevation)),
        ]
        shader["lightDir"] = direction
        shader["lightParams"] = [
            max(0.0, self.light_intensity_slider.value() / 100.0),
            0.35,
        ]
        self._apply_mesh_shader()

    def _on_light_setting_change(self, _value: int) -> None:
        self._update_light_labels()
        self._update_light_shader()
        if self._initialising:
            return
        if self.render_mode_combo.currentText() == "Surface mesh" and self._mesh_item is not None:
            self._mesh_item.update()
            self.view.update()

    def _clear_colorbar(self) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        fig.patch.set_facecolor(self._canvas_bg)
        self._colorbar_canvas.draw_idle()

    def _update_colorbar(
        self,
        cmap: mcolors.Colormap,
        vmin: float = 0.0,
        vmax: float = 1.0,
        label: str = "Normalised intensity",
    ) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        # ``ScalarMappable`` draws a classic matplotlib colour bar giving users a
        # persistent reference for the current normalisation range. We render it
        # horizontally so it reads naturally underneath the legend label while
        # leaving more vertical room for the surrounding controls.
        ax = fig.add_axes([0.12, 0.35, 0.76, 0.32])
        norm = plt.Normalize(vmin, vmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(mappable, cax=ax, orientation="horizontal")
        colorbar.set_label(label, color=self._fg_color, labelpad=6)
        colorbar.ax.xaxis.set_ticks_position("bottom")
        colorbar.ax.xaxis.set_label_position("bottom")
        colorbar.ax.tick_params(colors=self._fg_color, which="both")
        for spine in colorbar.ax.spines.values():
            spine.set_color(self._fg_color)
        colorbar.ax.set_facecolor(self._canvas_bg)
        fig.patch.set_facecolor(self._canvas_bg)
        self._colorbar_canvas.draw_idle()

    def _remove_axis_labels(self) -> None:
        if not self._axis_label_items:
            return
        for item in self._axis_label_items:
            try:
                self.view.removeItem(item)
            except Exception:  # pragma: no cover - defensive cleanup
                continue
        self._axis_label_items.clear()

    def _update_axis_labels(self) -> None:
        self._remove_axis_labels()
        if (
            getattr(self, "axis_labels_checkbox", None) is None
            or not self.axes_checkbox.isChecked()
            or not self.axis_labels_checkbox.isChecked()
            or self._current_bounds is None
        ):
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-6)
        safe_spans = np.where(spans > 0, spans, 1.0)
        centre = (mins + maxs) / 2.0
        offsets = safe_spans * 0.05
        colour = QColor(self._fg_color)
        font = QFont("Helvetica", 13)

        for _name, axis_idx, neg_name, pos_name in _SLICE_ORIENTATIONS:
            if axis_idx == 0:
                neg_pos = [
                    float(mins[0] - offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(maxs[0] + offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
            elif axis_idx == 1:
                neg_pos = [
                    float(centre[0]),
                    float(mins[1] - offsets[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(maxs[1] + offsets[1]),
                    float(centre[2]),
                ]
            else:
                neg_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(mins[2] - offsets[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(maxs[2] + offsets[2]),
                ]
            for text, position in ((neg_name, neg_pos), (pos_name, pos_pos)):
                label = gl.GLTextItem()
                label.setData(text=text, pos=position, color=colour, font=font)
                self.view.addItem(label)
                self._axis_label_items.append(label)

    def _update_scene_bounds(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        spans = np.maximum(maxs - mins, 1e-3)
        # Update the camera centre/distance so the current geometry fits snugly
        # inside the viewport regardless of the downsampling level.
        center = (mins + maxs) / 2.0
        self._current_bounds = (mins, maxs)
        self.view.opts["center"] = pg.Vector(center[0], center[1], center[2])
        self.view.opts["distance"] = float(np.linalg.norm(spans) * 1.2)
        self._update_axis_item()

    def _update_axis_item(self) -> None:
        if self._axis_item is not None:
            self.view.removeItem(self._axis_item)
            self._axis_item = None
        if not self.axes_checkbox.isChecked() or self._current_bounds is None:
            self._remove_axis_labels()
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-3)
        axis = _AdjustableAxisItem() if "_AdjustableAxisItem" in globals() else gl.GLAxisItem()
        axis.setSize(spans[0], spans[1], spans[2])
        axis.translate(float(mins[0]), float(mins[1]), float(mins[2]))
        thickness = getattr(self, "axis_thickness_slider", None)
        if thickness is not None and hasattr(axis, "setLineWidth"):
            axis.setLineWidth(float(thickness.value()))
        self.view.addItem(axis)
        self._axis_item = axis
        self._update_axis_labels()

    def _on_axes_toggle(self, checked: bool) -> None:
        slider = getattr(self, "axis_thickness_slider", None)
        if slider is not None:
            slider.setEnabled(checked)
        label_cb = getattr(self, "axis_labels_checkbox", None)
        if label_cb is not None:
            label_cb.setEnabled(checked)
            if not checked:
                self._remove_axis_labels()
        self._update_axis_item()

    def _on_axis_thickness_change(self, value: int) -> None:
        if hasattr(self, "axis_thickness_value"):
            self.axis_thickness_value.setText(f"{value} px")
        if self._axis_item is not None and hasattr(self._axis_item, "setLineWidth"):
            self._axis_item.setLineWidth(float(value))
            self.view.update()

    def _on_axis_labels_toggle(self, checked: bool) -> None:
        if not checked:
            self._remove_axis_labels()
        else:
            self._update_axis_labels()

    def _update_plot(self):
        if self._downsampled is None or self._normalised_volume is None:
            self.status_label.setText("No scalar volume available for rendering.")
            self._clear_colorbar()
            return

        thr = self.thresh_slider.value() / 100.0
        self.thresh_label.setText(f"{thr:.2f}")
        cmap = self.colormap_combo.currentText() or "viridis"
        alpha = self.opacity_slider.value() / 100.0
        mode = self.render_mode_combo.currentText()

        if mode == "Surface mesh":
            self._draw_surface_mesh(thr, cmap, alpha)
        else:
            self._draw_point_cloud(thr, cmap, alpha)

    def _handle_empty_point_cloud(self, shape: Sequence[int], scale: np.ndarray, thr: float) -> None:
        if self._scatter_item is not None:
            self.view.removeItem(self._scatter_item)
            self._scatter_item = None
        mins = np.zeros(3, dtype=np.float32)
        spans = np.maximum((np.asarray(shape, dtype=np.float32) - 1.0) * scale, 1e-3)
        maxs = mins + spans
        self._update_scene_bounds(mins, maxs)
        self._clear_colorbar()
        self.status_label.setText(
            f"No voxels above threshold {thr:.2f}. Lower the threshold to reveal data."
        )

    def _draw_point_cloud(self, thr: float, cmap_name: str, alpha: float) -> None:
        downsampled = self._downsampled
        if downsampled is None:
            self.status_label.setText("No data available for point-cloud rendering.")
            self._clear_colorbar()
            return

        if self._mesh_item is not None:
            # Switching from the iso-surface to point-cloud view discards the
            # cached ``GLMeshItem`` so only the scatter is redrawn.
            self.view.removeItem(self._mesh_item)
            self._mesh_item = None

        step = self._downsample_step
        scale = self._voxel_sizes_vec * float(step)
        shape = self._point_shape or downsampled.shape
        sorted_values = self._point_sorted_values
        sorted_indices = self._point_sorted_indices

        coords_mm: Optional[np.ndarray] = None
        values: Optional[np.ndarray] = None

        if (
            sorted_values is None
            or sorted_indices is None
            or self._point_shape is None
            or sorted_values.size == 0
        ):
            mask = downsampled >= thr
            coords = np.argwhere(mask)
            if coords.size == 0:
                self._handle_empty_point_cloud(shape, scale, thr)
                return
            coords_mm = coords.astype(np.float32, copy=False)
            coords_mm *= scale
            values = downsampled[mask].astype(np.float32, copy=False)
        else:
            start = int(np.searchsorted(sorted_values, thr, side="left"))
            if start >= sorted_values.size:
                self._handle_empty_point_cloud(shape, scale, thr)
                return

            selected_indices = sorted_indices[start:]
            selected_values = sorted_values[start:]
            if selected_indices.size > self._max_points:
                sample_idx = np.linspace(
                    0, selected_indices.size - 1, self._max_points, dtype=np.int64
                )
                selected_indices = selected_indices[sample_idx]
                selected_values = selected_values[sample_idx]

            if selected_indices.size == 0:
                self._handle_empty_point_cloud(shape, scale, thr)
                return

            if selected_indices.size > 1:
                order = np.argsort(selected_indices, kind="mergesort")
                selected_indices = selected_indices[order]
                selected_values = selected_values[order]

            coords_mm = self._indices_to_world(selected_indices, shape, scale)
            values = selected_values.astype(np.float32, copy=False)

        if coords_mm is None or values is None:
            self._handle_empty_point_cloud(shape, scale, thr)
            return

        coords_mm, values, slice_bounds = self._apply_slice_to_points(coords_mm, values)
        if coords_mm.size == 0:
            if slice_bounds is not None:
                if self._scatter_item is not None:
                    self.view.removeItem(self._scatter_item)
                    self._scatter_item = None
                self._clear_colorbar()
                mins, maxs = slice_bounds
                self._update_scene_bounds(mins, maxs)
                self.status_label.setText(
                    "Slice planes removed all voxels at threshold "
                    f"{thr:.2f}. Adjust the slicer sliders or disable slicing."
                )
                return
            self._handle_empty_point_cloud(shape, scale, thr)
            return

        if values is None:
            values = np.empty(coords_mm.shape[0], dtype=np.float32)
        cmap = self._get_adjusted_colormap(cmap_name)
        colors = self._map_colors(values, cmap, alpha)

        if self._scatter_item is None:
            # ``pxMode`` keeps the slider-controlled marker size in screen
            # pixels, mirroring the behaviour of the original matplotlib view.
            self._scatter_item = gl.GLScatterPlotItem(pxMode=True)
            self.view.addItem(self._scatter_item)
        self._scatter_item.setData(pos=coords_mm, color=colors, size=float(self.point_slider.value()))
        self._scatter_item.setGLOptions("translucent" if alpha < 0.999 else "opaque")

        mins = coords_mm.min(axis=0)
        maxs = coords_mm.max(axis=0)
        self._update_scene_bounds(mins, maxs)
        self._update_colorbar(cmap)

        total_voxels = self._estimate_voxels_above_threshold(thr)
        spin = getattr(self, "downsample_spin", None)
        downsample_source = "manual" if spin and spin.value() > 0 else "auto"
        lighting_suffix = ""
        if self._light_shader is not None or self._flat_shader is not None:
            lighting_suffix = " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        self.status_label.setText(
            "Point cloud: "
            f"{coords_mm.shape[0]:,} voxels (threshold {thr:.2f}, opacity {alpha:.2f}, "
            f"point size {self.point_slider.value()}). "
            f"Downsample step {step} ({downsample_source}); "
            f"≈ total voxels ≥ threshold {total_voxels:,}.{lighting_suffix}"
            f"{self._slice_status_suffix()}"
        )

    def _draw_surface_mesh(self, thr: float, cmap_name: str, alpha: float) -> None:
        if not HAS_SKIMAGE:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                "Surface rendering requires the optional 'scikit-image' dependency."
            )
            return

        volume = self._normalised_volume
        if volume is None:
            self.status_label.setText("No scalar volume available for rendering.")
            self._clear_colorbar()
            return

        if self._scatter_item is not None:
            self.view.removeItem(self._scatter_item)
            self._scatter_item = None

        step = self._downsample_step
        slices = tuple(slice(None, None, step) for _ in range(3))
        reduced = volume[slices]

        if min(reduced.shape) < 2:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                "Volume is too small after downsampling to compute a surface mesh."
            )
            return

        vol_min = float(np.min(reduced))
        vol_max = float(np.max(reduced))
        if not (vol_min < thr < vol_max):
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                f"Iso level {thr:.2f} is outside the volume range [{vol_min:.2f}, {vol_max:.2f}]."
            )
            return

        try:
            verts, faces, _normals, values = sk_measure.marching_cubes(
                reduced,
                level=thr,
                step_size=max(1, self._surface_step),
                spacing=(
                    step * self._voxel_sizes[0],
                    step * self._voxel_sizes[1],
                    step * self._voxel_sizes[2],
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(f"Marching cubes failed: {exc}")
            return

        vertex_values = np.clip(values, 0.0, 1.0)
        verts, faces, vertex_values, slice_bounds = self._clip_mesh_to_slices(
            verts, faces, vertex_values
        )
        if verts.size == 0 or faces.size == 0:
            if slice_bounds is not None:
                if self._mesh_item is not None:
                    self.view.removeItem(self._mesh_item)
                    self._mesh_item = None
                self._clear_colorbar()
                mins, maxs = slice_bounds
                self._update_scene_bounds(mins, maxs)
                self.status_label.setText(
                    f"Slice planes removed all surface triangles at iso level {thr:.2f}. "
                    "Adjust the slicer sliders or disable slicing."
                )
                return
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                f"No closed surface found at iso level {thr:.2f}; adjust the slider."
            )
            return

        if vertex_values is None:
            vertex_values = np.zeros(verts.shape[0], dtype=np.float32)

        meshdata = gl.MeshData(vertexes=verts, faces=faces)
        cmap = self._get_adjusted_colormap(cmap_name)
        vertex_colors = self._map_colors(vertex_values, cmap, alpha)
        meshdata.setVertexColors(vertex_colors)

        if self._mesh_item is None:
            # ``GLMeshItem`` retains the uploaded vertex buffers so subsequent
            # slider tweaks only update colours instead of reallocating the mesh.
            self._mesh_item = gl.GLMeshItem(
                meshdata=meshdata,
                smooth=False,
                shader="shaded",
                drawEdges=False,
            )
            self.view.addItem(self._mesh_item)
        else:
            self._mesh_item.setMeshData(meshdata=meshdata)

        self._update_light_shader()
        self._mesh_item.setGLOptions("translucent" if alpha < 0.999 else "opaque")

        mins = np.min(verts, axis=0)
        maxs = np.max(verts, axis=0)
        self._update_scene_bounds(mins, maxs)
        self._update_colorbar(cmap, vmin=thr, vmax=1.0, label="Iso level (normalised)")

        total_voxels = int(np.count_nonzero(self._normalised_volume >= thr))
        spin = getattr(self, "downsample_spin", None)
        downsample_source = "manual" if spin and spin.value() > 0 else "auto"
        lighting_suffix = ""
        if self._light_shader is not None or self._flat_shader is not None:
            lighting_suffix = " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        self.status_label.setText(
            "Surface mesh: "
            f"{verts.shape[0]:,} vertices / {faces.shape[0]:,} faces (iso {thr:.2f}, "
            f"opacity {alpha:.2f}). Downsample step {step} ({downsample_source}); "
            f"marching step {self._surface_step}. Total voxels ≥ level {total_voxels:,}.{lighting_suffix}"
            f"{self._slice_status_suffix()}"
        )

    def _on_render_mode_change(self, _mode: str) -> None:
        if self._initialising:
            return
        self._update_mode_dependent_controls()
        self._update_plot()

    def _update_mode_dependent_controls(self) -> None:
        mode = self.render_mode_combo.currentText()
        is_surface = mode == "Surface mesh"
        self.point_label.setVisible(not is_surface)
        self.point_slider.setVisible(not is_surface)
        self.point_slider.setEnabled(not is_surface)
        self.surface_step_label.setEnabled(is_surface)
        self.surface_step_spin.setEnabled(is_surface)
        if hasattr(self, "lighting_group"):
            self.lighting_group.setEnabled(
                is_surface
                and (self._light_shader is not None or self._flat_shader is not None)
            )
            self._update_light_controls_enabled()
        if is_surface:
            self.thresh_text.setText("Iso level:")
            self.thresh_slider.setToolTip(
                "Iso-surface level within the normalised volume (0–1)."
            )
        else:
            self.thresh_text.setText("Threshold:")
            self.thresh_slider.setToolTip(
                "Minimum normalised intensity included in the rendering."
            )

    def _on_opacity_change(self, value: int) -> None:
        self.opacity_label.setText(f"{value / 100.0:.2f}")
        if not self._initialising:
            self._update_plot()

    def _on_max_points_change(self, value: int) -> None:
        self._max_points = int(value)
        self._prepare_downsampled()
        if not self._initialising:
            self._update_plot()

    def _on_downsample_change(self, _value: int) -> None:
        self._prepare_downsampled()
        if not self._initialising:
            self._update_plot()

    def _on_surface_step_change(self, value: int) -> None:
        self._surface_step = max(1, int(value))
        if not self._initialising and self.render_mode_combo.currentText() == "Surface mesh":
            self._update_plot()

class Surface3DDialog(QDialog):
    """Interactive renderer for surface meshes from GIFTI or FreeSurfer."""

    def __init__(
        self,
        parent,
        vertices: np.ndarray,
        faces: np.ndarray,
        scalars: Optional[dict[str, np.ndarray]] = None,
        meta: Optional[dict[str, Any]] = None,
        title: Optional[str] = None,
        dark_theme: bool = False,
    ) -> None:
        super().__init__(parent)
        if not HAS_PYQTGRAPH:
            raise RuntimeError(
                "Surface rendering requires the optional 'pyqtgraph' dependency."
            )
        self._initialising = True
        self.setWindowTitle(title or "Surface Viewer")
        self.resize(1200, 900)
        self.setMinimumSize(780, 560)
        self.setSizeGripEnabled(True)

        verts = np.asarray(vertices, dtype=np.float32)
        faces_arr = np.asarray(faces, dtype=np.int32)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("Surface vertices must be an (N, 3) array")
        if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
            raise ValueError("Surface faces must be an (M, 3) array")
        if faces_arr.size and int(faces_arr.max(initial=-1)) >= verts.shape[0]:
            raise ValueError("Face indices exceed available vertices")

        self._vertices = verts
        self._faces = faces_arr
        self._meta = meta or {}
        self._dark_theme = bool(dark_theme)
        self._fg_color = "#f0f0f0" if self._dark_theme else "#202020"
        self._canvas_bg = "#202020" if self._dark_theme else "#ffffff"
        self._axis_label_items: list[gl.GLTextItem] = []
        self._data_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._slice_controls: dict[str, _SliceControl] = {}

        self._scalar_fields: dict[str, np.ndarray] = {}
        if scalars:
            for name, values in scalars.items():
                arr = np.asarray(values, dtype=np.float32).reshape(-1)
                if arr.shape[0] != verts.shape[0]:
                    continue
                safe_name = str(name).strip() or f"Field {len(self._scalar_fields) + 1}"
                self._scalar_fields[safe_name] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        self._mesh_item: Optional[gl.GLMeshItem] = None
        self._axis_item: Optional[gl.GLAxisItem] = None
        self._current_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._colormap_cache: dict[tuple[str, float], mcolors.Colormap] = {}
        self._light_shader = _create_directional_light_shader()
        self._flat_shader = _create_flat_color_shader()
        self._lighting_enabled = True

        if self._vertices.size:
            mins = np.min(self._vertices, axis=0).astype(np.float32, copy=False)
            maxs = np.max(self._vertices, axis=0).astype(np.float32, copy=False)
            self._data_bounds = (mins.copy(), maxs.copy())

        layout = QVBoxLayout(self)

        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(12)
        layout.addWidget(self._splitter)

        self._view_container = QWidget()
        view_layout = QVBoxLayout(self._view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        # ``GLViewWidget`` renders using OpenGL so panning/zooming the scene does
        # not require recomputing the mesh when interacting with the viewport.
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setBackgroundColor(self._canvas_bg)
        self.view.opts["distance"] = 200
        self.view.opts["elevation"] = 20
        self.view.opts["azimuth"] = -60
        view_layout.addWidget(self.view)

        self._splitter.addWidget(self._view_container)

        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(6, 6, 6, 6)
        settings_layout.setSpacing(10)

        # Allow the combined colour bar + control pane to be docked on the
        # side or bottom of the viewport.
        placement_layout = QHBoxLayout()
        placement_layout.setSpacing(6)
        placement_layout.addWidget(QLabel("Panel placement:"))
        self.panel_location_combo = QComboBox()
        self.panel_location_combo.addItems(["Bottom", "Left", "Right"])
        placement_layout.addWidget(self.panel_location_combo)
        placement_layout.addStretch()
        settings_layout.addLayout(placement_layout)

        scalar_row = QHBoxLayout()
        scalar_row.setContentsMargins(0, 0, 0, 0)
        scalar_row.setSpacing(8)
        scalar_row.addWidget(QLabel("Scalar:"))
        self.scalar_combo = QComboBox()
        self.scalar_combo.addItem("Constant colour", userData=None)
        for name in sorted(self._scalar_fields):
            self.scalar_combo.addItem(name, userData=name)
        self.scalar_combo.setEnabled(bool(self._scalar_fields))
        scalar_row.addWidget(self.scalar_combo, 1)
        scalar_row.addStretch()
        settings_layout.addLayout(scalar_row)

        appearance_group = QGroupBox("Appearance")
        appearance_layout = QGridLayout(appearance_group)
        appearance_layout.setColumnStretch(1, 1)

        row = 0
        appearance_layout.addWidget(QLabel("Colormap:"), row, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "twilight",
            "cubehelix",
            "Spectral",
            "coolwarm",
            "YlGnBu",
            "Greys",
            "bone",
        ])
        self.colormap_combo.setToolTip(
            "Select the matplotlib colormap for vertex intensities.",
        )
        appearance_layout.addWidget(self.colormap_combo, row, 1)

        row += 1
        appearance_layout.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.setToolTip(
            "Global alpha applied to rendered surfaces.",
        )
        appearance_layout.addWidget(self.opacity_slider, row, 1)
        self.opacity_label = QLabel("0.80")
        appearance_layout.addWidget(self.opacity_label, row, 2)

        row += 1
        appearance_layout.addWidget(QLabel("Colour intensity:"), row, 0)
        self.color_intensity_slider = QSlider(Qt.Horizontal)
        self.color_intensity_slider.setRange(10, 200)
        self.color_intensity_slider.setValue(100)
        self.color_intensity_slider.setToolTip(
            "Scale factor applied to RGB values after colormap lookup (0.1–2.0×).",
        )
        appearance_layout.addWidget(self.color_intensity_slider, row, 1)
        self.color_intensity_label = QLabel("1.00×")
        appearance_layout.addWidget(self.color_intensity_label, row, 2)
        settings_layout.addWidget(appearance_group)

        # Keep the colour bar within the scroll area so it collapses together
        # with the rest of the surface controls.
        colorbar_group = QGroupBox("Colour bar")
        colorbar_layout = QVBoxLayout(colorbar_group)
        colorbar_layout.setContentsMargins(8, 6, 8, 6)
        colorbar_layout.setSpacing(4)
        self._colorbar_canvas = FigureCanvas(plt.Figure(figsize=(2.6, 1.4)))
        self._colorbar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._colorbar_canvas.setFixedHeight(180)
        self._colorbar_canvas.figure.patch.set_facecolor(self._canvas_bg)
        colorbar_layout.addWidget(self._colorbar_canvas)
        settings_layout.addWidget(colorbar_group)

        axes_group = QGroupBox("Axes")
        axes_layout = QGridLayout(axes_group)
        axes_layout.setColumnStretch(1, 1)

        row = 0
        self.axes_checkbox = QCheckBox("Show axes")
        self.axes_checkbox.setChecked(False)
        axes_layout.addWidget(self.axes_checkbox, row, 0, 1, 3)

        row += 1
        axes_layout.addWidget(QLabel("Axis thickness:"), row, 0)
        self.axis_thickness_slider = QSlider(Qt.Horizontal)
        self.axis_thickness_slider.setRange(1, 12)
        self.axis_thickness_slider.setValue(3)
        self.axis_thickness_slider.setToolTip(
            "Line width of the anatomical axes (pixels).",
        )
        self.axis_thickness_slider.setEnabled(False)
        axes_layout.addWidget(self.axis_thickness_slider, row, 1)
        self.axis_thickness_value = QLabel("3 px")
        axes_layout.addWidget(self.axis_thickness_value, row, 2)

        row += 1
        self.axis_labels_checkbox = QCheckBox("Show axis labels")
        self.axis_labels_checkbox.setChecked(False)
        self.axis_labels_checkbox.setEnabled(False)
        axes_layout.addWidget(self.axis_labels_checkbox, row, 0, 1, 3)
        settings_layout.addWidget(axes_group)

        slice_group = QGroupBox("Slice planes")
        slice_layout = QVBoxLayout(slice_group)
        slice_layout.setContentsMargins(8, 4, 8, 4)
        slice_layout.setSpacing(6)
        self._slice_controls.clear()
        for key, axis_idx, neg_name, pos_name in _SLICE_ORIENTATIONS:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(6)
            checkbox = QCheckBox(key.capitalize())
            checkbox.setChecked(False)
            row_layout.addWidget(checkbox)
            row_layout.addSpacing(4)
            row_layout.addWidget(QLabel(f"{neg_name}:"))
            min_slider = QSlider(Qt.Horizontal)
            min_slider.setRange(0, 100)
            min_slider.setValue(0)
            min_slider.setEnabled(False)
            row_layout.addWidget(min_slider, 1)
            min_value = QLabel("0.0 mm")
            row_layout.addWidget(min_value)
            row_layout.addSpacing(6)
            row_layout.addWidget(QLabel(f"{pos_name}:"))
            max_slider = QSlider(Qt.Horizontal)
            max_slider.setRange(0, 100)
            max_slider.setValue(100)
            max_slider.setEnabled(False)
            row_layout.addWidget(max_slider, 1)
            max_value = QLabel("0.0 mm")
            row_layout.addWidget(max_value)
            row_layout.addStretch()
            slice_layout.addLayout(row_layout)
            control = _SliceControl(
                checkbox=checkbox,
                min_slider=min_slider,
                max_slider=max_slider,
                min_value=min_value,
                max_value=max_value,
                axis=axis_idx,
                negative_name=neg_name,
                positive_name=pos_name,
            )
            self._slice_controls[key] = control
            checkbox.toggled.connect(lambda checked, name=key: self._on_slice_toggle(name, checked))
            min_slider.valueChanged.connect(
                lambda value, name=key: self._on_slice_slider_change(name, "min", value)
            )
            max_slider.valueChanged.connect(
                lambda value, name=key: self._on_slice_slider_change(name, "max", value)
            )
        slice_layout.addStretch()
        settings_layout.addWidget(slice_group)

        self.lighting_group = QGroupBox("Lighting")
        light_layout = QGridLayout(self.lighting_group)
        light_row = 0
        self.light_enable_checkbox = QCheckBox("Enable lighting")
        self.light_enable_checkbox.setChecked(True)
        light_layout.addWidget(self.light_enable_checkbox, light_row, 0, 1, 3)
        light_row += 1
        light_layout.addWidget(QLabel("Azimuth:"), light_row, 0)
        self.light_azimuth_slider = QSlider(Qt.Horizontal)
        self.light_azimuth_slider.setRange(-180, 180)
        self.light_azimuth_slider.setValue(-45)
        self.light_azimuth_slider.setToolTip(
            "Horizontal direction of the light source relative to the mesh (°).",
        )
        light_layout.addWidget(self.light_azimuth_slider, light_row, 1)
        self.light_azimuth_label = QLabel("-45°")
        light_layout.addWidget(self.light_azimuth_label, light_row, 2)

        light_row += 1
        light_layout.addWidget(QLabel("Elevation:"), light_row, 0)
        self.light_elevation_slider = QSlider(Qt.Horizontal)
        self.light_elevation_slider.setRange(-90, 90)
        self.light_elevation_slider.setValue(30)
        self.light_elevation_slider.setToolTip(
            "Vertical angle of the light source relative to the mesh (°).",
        )
        light_layout.addWidget(self.light_elevation_slider, light_row, 1)
        self.light_elevation_label = QLabel("30°")
        light_layout.addWidget(self.light_elevation_label, light_row, 2)

        light_row += 1
        light_layout.addWidget(QLabel("Intensity:"), light_row, 0)
        self.light_intensity_slider = QSlider(Qt.Horizontal)
        self.light_intensity_slider.setRange(10, 300)
        self.light_intensity_slider.setValue(130)
        self.light_intensity_slider.setToolTip(
            "Diffuse lighting strength (0.1–3.0×). Ambient light stays constant.",
        )
        light_layout.addWidget(self.light_intensity_slider, light_row, 1)
        self.light_intensity_label = QLabel("1.30×")
        light_layout.addWidget(self.light_intensity_label, light_row, 2)
        if self._light_shader is None and self._flat_shader is None:
            self.lighting_group.setEnabled(False)
        settings_layout.addWidget(self.lighting_group)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        settings_layout.addWidget(self.status_label)
        settings_layout.addStretch()

        self._panel_scroll = ShrinkableScrollArea()
        self._panel_scroll.setWidget(settings_container)
        self._panel_scroll.setWidgetResizable(True)
        self._panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._splitter.addWidget(self._panel_scroll)
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 1)

        self.panel_location_combo.currentTextChanged.connect(self._on_panel_location_change)
        self.scalar_combo.currentIndexChanged.connect(self._on_scalar_change)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_change)
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        self.color_intensity_slider.valueChanged.connect(self._on_color_intensity_change)
        self.axes_checkbox.toggled.connect(self._on_axes_toggle)
        self.axis_thickness_slider.valueChanged.connect(self._on_axis_thickness_change)
        self.axis_labels_checkbox.toggled.connect(self._on_axis_labels_toggle)
        self.light_enable_checkbox.toggled.connect(self._on_light_enabled_toggle)
        self.light_azimuth_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_elevation_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_intensity_slider.valueChanged.connect(self._on_light_setting_change)

        self._apply_panel_layout(self.panel_location_combo.currentText())
        self._update_light_controls_enabled()

        # Prepare labels and shader uniforms before the first draw call.
        self._on_axis_thickness_change(self.axis_thickness_slider.value())
        self._update_slice_labels()
        self._on_color_intensity_change(self.color_intensity_slider.value())
        self._update_light_labels()
        self._update_light_shader()

        self._initialising = False
        self._update_plot()

    def _apply_panel_layout(self, location: str) -> None:
        """Reorder the settings splitter when users move the control pane."""

        if not hasattr(self, "_splitter"):
            return

        normalised = (location or "bottom").strip().lower()
        if normalised not in {"bottom", "left", "right"}:
            normalised = "bottom"

        view_first = True
        if normalised == "bottom":
            self._splitter.setOrientation(Qt.Vertical)
            view_first = True
        else:
            self._splitter.setOrientation(Qt.Horizontal)
            view_first = normalised != "left"

        widgets = [self._view_container, self._panel_scroll]
        if not view_first:
            widgets.reverse()
        for index, widget in enumerate(widgets):
            if self._splitter.indexOf(widget) != index:
                self._splitter.insertWidget(index, widget)

        if view_first:
            self._splitter.setStretchFactor(0, 4)
            self._splitter.setStretchFactor(1, 1)
        else:
            self._splitter.setStretchFactor(0, 1)
            self._splitter.setStretchFactor(1, 4)

    def _on_panel_location_change(self, location: str) -> None:
        self._apply_panel_layout(location)

    def _update_light_controls_enabled(self) -> None:
        enabled = bool(self.lighting_group.isEnabled() and self._lighting_enabled)
        for widget in (
            self.light_azimuth_slider,
            self.light_elevation_slider,
            self.light_intensity_slider,
        ):
            widget.setEnabled(enabled)
        for label in (
            self.light_azimuth_label,
            self.light_elevation_label,
            self.light_intensity_label,
        ):
            label.setEnabled(enabled)

    def _apply_mesh_shader(self) -> None:
        if self._mesh_item is None:
            return
        if self._light_shader is not None and self._lighting_enabled:
            self._mesh_item.setShader(self._light_shader)
        elif self._flat_shader is not None:
            self._mesh_item.setShader(self._flat_shader)
        else:
            self._mesh_item.setShader("shaded")

    def _on_light_enabled_toggle(self, checked: bool) -> None:
        self._lighting_enabled = bool(checked)
        self._update_light_controls_enabled()
        if self._lighting_enabled:
            self._update_light_shader()
        else:
            self._apply_mesh_shader()
        if not self._initialising and self._mesh_item is not None:
            self._mesh_item.update()
            self.view.update()

    def _current_color_intensity(self) -> float:
        return max(0.1, self.color_intensity_slider.value() / 100.0)

    def _get_adjusted_colormap(self, cmap_name: str) -> mcolors.Colormap:
        intensity = round(self._current_color_intensity(), 3)
        key = (cmap_name, intensity)
        cached = self._colormap_cache.get(key)
        if cached is not None:
            return cached
        base = plt.get_cmap(cmap_name)
        samples = base(np.linspace(0.0, 1.0, 512))
        samples[:, :3] = np.clip(samples[:, :3] * intensity, 0.0, 1.0)
        cmap = mcolors.ListedColormap(samples, name=f"{cmap_name}_x{intensity:.3f}")
        self._colormap_cache[key] = cmap
        return cmap

    def _map_colors(
        self, values: np.ndarray, cmap: mcolors.Colormap, alpha: float
    ) -> np.ndarray:
        # ``values`` are pre-normalised to [0, 1] so we can reuse matplotlib
        # colour maps and simply blend in the requested opacity.
        colours = np.asarray(cmap(np.clip(values, 0.0, 1.0)), dtype=np.float32)
        colours[:, 3] = np.clip(colours[:, 3] * alpha, 0.0, 1.0)
        return colours

    def _update_slice_labels(self) -> None:
        if not self._slice_controls:
            return
        bounds = self._data_bounds
        has_bounds = bounds is not None
        mins: Optional[np.ndarray]
        maxs: Optional[np.ndarray]
        if has_bounds:
            mins, maxs = bounds  # type: ignore[assignment]
        else:
            mins = maxs = None
        for control in self._slice_controls.values():
            slider_enabled = bool(has_bounds and control.checkbox.isChecked())
            control.min_slider.setEnabled(slider_enabled)
            control.max_slider.setEnabled(slider_enabled)
            if mins is None or maxs is None:
                control.min_value.setText("--")
                control.max_value.setText("--")
                continue
            axis = control.axis
            axis_min = float(mins[axis])
            axis_max = float(maxs[axis])
            span = axis_max - axis_min
            min_frac = control.min_slider.value() / 100.0
            max_frac = control.max_slider.value() / 100.0
            min_mm = axis_min + span * min_frac
            max_mm = axis_min + span * max_frac
            control.min_value.setText(f"{min_mm:.1f} mm")
            control.max_value.setText(f"{max_mm:.1f} mm")

    def _active_slice_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not self._slice_controls or self._data_bounds is None:
            return None
        mins, maxs = self._data_bounds
        min_bounds = np.asarray(mins, dtype=np.float32).copy()
        max_bounds = np.asarray(maxs, dtype=np.float32).copy()
        active = False
        for control in self._slice_controls.values():
            if not control.checkbox.isChecked():
                continue
            axis = control.axis
            axis_min = float(mins[axis])
            axis_max = float(maxs[axis])
            span = axis_max - axis_min
            min_frac = control.min_slider.value() / 100.0
            max_frac = control.max_slider.value() / 100.0
            min_bounds[axis] = float(axis_min + span * min_frac)
            max_bounds[axis] = float(axis_min + span * max_frac)
            active = True
        if not active:
            return None
        return min_bounds, max_bounds

    def _slice_status_suffix(self) -> str:
        bounds = self._active_slice_bounds()
        if bounds is None:
            return ""
        mins, maxs = bounds
        parts: list[str] = []
        for control in self._slice_controls.values():
            if not control.checkbox.isChecked():
                continue
            axis = control.axis
            parts.append(
                f"{control.negative_name}–{control.positive_name} "
                f"{mins[axis]:.1f}–{maxs[axis]:.1f} mm"
            )
        if not parts:
            return ""
        return " Slice windows: " + ", ".join(parts) + "."

    def _mask_coords(
        self, coords: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        mins, maxs = bounds
        if coords.size == 0:
            return np.zeros(0, dtype=bool)
        mask = np.ones(coords.shape[0], dtype=bool)
        for axis in range(3):
            mask &= (coords[:, axis] >= mins[axis]) & (coords[:, axis] <= maxs[axis])
        return mask

    def _apply_slice_to_points(
        self, coords: np.ndarray, values: Optional[np.ndarray]
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[tuple[np.ndarray, np.ndarray]]]:
        bounds = self._active_slice_bounds()
        if bounds is None or coords.size == 0:
            return coords, values, None
        mask = self._mask_coords(coords, bounds)
        if not np.any(mask):
            empty_coords = coords[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_coords, empty_vals, bounds
        filtered_coords = coords[mask]
        filtered_values = values[mask] if values is not None else None
        return filtered_coords, filtered_values, bounds

    def _clip_mesh_to_slices(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        values: Optional[np.ndarray],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[tuple[np.ndarray, np.ndarray]],
    ]:
        bounds = self._active_slice_bounds()
        if bounds is None or verts.size == 0:
            return verts, faces, values, None
        mask = self._mask_coords(verts, bounds)
        if not np.any(mask):
            empty_verts = verts[:0].reshape(0, 3)
            empty_faces = faces[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_verts, empty_faces, empty_vals, bounds
        selected = np.where(mask)[0]
        index_map = -np.ones(mask.shape[0], dtype=np.int64)
        index_map[selected] = np.arange(selected.size, dtype=np.int64)
        mapped_faces = index_map[faces]
        valid = (mapped_faces >= 0).all(axis=1)
        mapped_faces = mapped_faces[valid]
        if mapped_faces.size == 0:
            empty_verts = verts[:0].reshape(0, 3)
            empty_faces = faces[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_verts, empty_faces, empty_vals, bounds
        new_verts = verts[selected]
        new_values = values[selected] if values is not None else None
        return new_verts, mapped_faces.astype(np.int32, copy=False), new_values, bounds

    def _on_slice_toggle(self, name: str, _checked: bool) -> None:
        if name not in self._slice_controls:
            return
        self._update_slice_labels()
        if not self._initialising:
            self._update_plot()

    def _on_slice_slider_change(self, name: str, which: str, value: int) -> None:
        control = self._slice_controls.get(name)
        if control is None:
            return
        if which == "min":
            other = control.max_slider
            if value > other.value():
                other.blockSignals(True)
                other.setValue(value)
                other.blockSignals(False)
        else:
            other = control.min_slider
            if value < other.value():
                other.blockSignals(True)
                other.setValue(value)
                other.blockSignals(False)
        self._update_slice_labels()
        if not self._initialising and control.checkbox.isChecked():
            self._update_plot()
    def _clear_colorbar(self) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        fig.patch.set_facecolor(self._canvas_bg)
        self._colorbar_canvas.draw_idle()

    def _update_colorbar(
        self, cmap: mcolors.Colormap, vmin: float, vmax: float, label: str = "Scalar value"
    ) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        # Render the colour bar horizontally so it mirrors the slice legend above
        # while preserving vertical space for the rest of the controls.
        ax = fig.add_axes([0.12, 0.35, 0.76, 0.32])
        norm = plt.Normalize(vmin, vmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(mappable, cax=ax, orientation="horizontal")
        colorbar.set_label(label, color=self._fg_color, labelpad=6)
        colorbar.ax.xaxis.set_ticks_position("bottom")
        colorbar.ax.xaxis.set_label_position("bottom")
        colorbar.ax.tick_params(colors=self._fg_color, which="both")
        for spine in colorbar.ax.spines.values():
            spine.set_color(self._fg_color)
        colorbar.ax.set_facecolor(self._canvas_bg)
        fig.patch.set_facecolor(self._canvas_bg)
        self._colorbar_canvas.draw_idle()

    def _on_color_intensity_change(self, value: int) -> None:
        self.color_intensity_label.setText(f"{value / 100.0:.2f}×")
        if not self._initialising:
            self._update_plot()

    def _update_light_labels(self) -> None:
        self.light_azimuth_label.setText(f"{self.light_azimuth_slider.value():+d}°")
        self.light_elevation_label.setText(
            f"{self.light_elevation_slider.value():+d}°"
        )
        self.light_intensity_label.setText(
            f"{self.light_intensity_slider.value() / 100.0:.2f}×"
        )

    def _update_light_shader(self) -> None:
        shader = self._light_shader
        if shader is None or not self._lighting_enabled:
            self._apply_mesh_shader()
            return
        azimuth = math.radians(self.light_azimuth_slider.value())
        elevation = math.radians(self.light_elevation_slider.value())
        cos_el = math.cos(elevation)
        direction = [
            float(cos_el * math.cos(azimuth)),
            float(cos_el * math.sin(azimuth)),
            float(math.sin(elevation)),
        ]
        shader["lightDir"] = direction
        shader["lightParams"] = [
            max(0.0, self.light_intensity_slider.value() / 100.0),
            0.35,
        ]
        self._apply_mesh_shader()

    def _on_light_setting_change(self, _value: int) -> None:
        self._update_light_labels()
        self._update_light_shader()
        if self._initialising:
            return
        if self._mesh_item is not None:
            self._mesh_item.update()
            self.view.update()

    def _update_scene_bounds(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        spans = np.maximum(maxs - mins, 1e-3)
        # Keep the camera centred on the mesh to avoid jarring jumps when users
        # switch between scalar fields or adjust opacity.
        center = (mins + maxs) / 2.0
        self._current_bounds = (mins, maxs)
        self.view.opts["center"] = pg.Vector(center[0], center[1], center[2])
        self.view.opts["distance"] = float(np.linalg.norm(spans) * 1.2)
        self._update_axis_item()

    def _on_scalar_change(self, _index: int) -> None:
        self._update_plot()

    def _on_colormap_change(self, _name: str) -> None:
        self._update_plot()

    def _on_opacity_change(self, value: int) -> None:
        self.opacity_label.setText(f"{value / 100.0:.2f}")
        self._update_plot()

    def _update_plot(self) -> None:
        self._clear_colorbar()
        if self._faces.size == 0 or self._vertices.size == 0:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self.status_label.setText("Surface mesh has no vertices or faces to render.")
            return

        alpha = self.opacity_slider.value() / 100.0
        cmap_name = self.colormap_combo.currentText() or "viridis"
        scalar_key = self.scalar_combo.currentData()
        cmap = self._get_adjusted_colormap(cmap_name)

        summary = (
            f"Surface mesh: {self._vertices.shape[0]:,} vertices / {self._faces.shape[0]:,} faces. "
        )

        normalised_values: Optional[np.ndarray] = None
        vmin: Optional[float] = None
        vmax: Optional[float] = None
        if scalar_key and scalar_key in self._scalar_fields:
            raw_values = self._scalar_fields[scalar_key]
            vmin = float(np.min(raw_values))
            vmax = float(np.max(raw_values))
            if math.isclose(vmin, vmax):
                normalised_values = np.zeros_like(raw_values, dtype=np.float32)
                vmin, vmax = vmin - 0.5, vmax + 0.5
            else:
                norm = (raw_values - vmin) / (vmax - vmin)
                normalised_values = np.clip(norm.astype(np.float32, copy=False), 0.0, 1.0)

        clipped_verts, clipped_faces, clipped_values, slice_bounds = self._clip_mesh_to_slices(
            self._vertices, self._faces, normalised_values
        )

        if clipped_verts.size == 0 or clipped_faces.size == 0:
            if slice_bounds is not None:
                if self._mesh_item is not None:
                    self.view.removeItem(self._mesh_item)
                    self._mesh_item = None
                self._clear_colorbar()
                mins, maxs = slice_bounds
                self._update_scene_bounds(mins, maxs)
                self.status_label.setText(
                    "Slice planes removed all surface triangles. "
                    "Adjust the slicer sliders or disable slicing."
                )
                return
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self.status_label.setText("Surface mesh has no vertices or faces to render.")
            return

        meshdata = gl.MeshData(vertexes=clipped_verts, faces=clipped_faces)
        if clipped_values is not None and scalar_key and scalar_key in self._scalar_fields:
            colours = self._map_colors(clipped_values, cmap, alpha)
            meshdata.setVertexColors(colours)
            assert vmin is not None and vmax is not None
            self._update_colorbar(cmap, vmin, vmax)
            summary += f"Scalar '{scalar_key}' range {vmin:.4g} – {vmax:.4g}. "
        else:
            base_color = cmap(0.6)
            colour = np.array(
                [[base_color[0], base_color[1], base_color[2], base_color[3] * alpha]],
                dtype=np.float32,
            )
            meshdata.setVertexColors(np.repeat(colour, clipped_verts.shape[0], axis=0))
            self._update_colorbar(cmap, 0.0, 1.0, label="Colormap preview")
            summary += "Constant colouring applied. "

        if self._mesh_item is None:
            self._mesh_item = gl.GLMeshItem(
                meshdata=meshdata,
                smooth=False,
                shader="shaded",
                drawEdges=False,
            )
            self.view.addItem(self._mesh_item)
        else:
            self._mesh_item.setMeshData(meshdata=meshdata)
        self._update_light_shader()
        self._mesh_item.setGLOptions("translucent" if alpha < 0.999 else "opaque")

        mins = np.min(clipped_verts, axis=0)
        maxs = np.max(clipped_verts, axis=0)
        self._update_scene_bounds(mins, maxs)

        summary += f"Opacity {alpha:.2f}."
        if self._light_shader is not None or self._flat_shader is not None:
            summary += " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        if self._scalar_fields and (not scalar_key or scalar_key not in self._scalar_fields):
            summary += " Select a scalar field to colour the surface."
        summary += self._slice_status_suffix()
        self.status_label.setText(summary)

class FreeSurferSurfaceDialog(QDialog):
    """Simplified viewer tailored for FreeSurfer ``*.pial``/``*.white`` meshes."""

    def __init__(
        self,
        parent,
        vertices: np.ndarray,
        faces: np.ndarray,
        title: Optional[str] = None,
        dark_theme: bool = False,
    ) -> None:
        super().__init__(parent)
        if not HAS_PYQTGRAPH:
            raise RuntimeError(
                "Surface rendering requires the optional 'pyqtgraph' dependency."
            )

        self.setWindowTitle(title or "FreeSurfer Surface")
        self.resize(900, 720)
        self.setMinimumSize(620, 480)
        self.setSizeGripEnabled(True)

        verts = np.asarray(vertices, dtype=np.float32)
        faces_arr = np.asarray(faces, dtype=np.int32)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("Surface vertices must be an (N, 3) array")
        if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
            raise ValueError("Surface faces must be an (M, 3) array")
        if faces_arr.size and int(faces_arr.max(initial=-1)) >= verts.shape[0]:
            raise ValueError("Face indices exceed available vertices")

        self._vertices = verts
        self._faces = faces_arr
        self._fg_color = "#f0f0f0" if dark_theme else "#202020"
        self._canvas_bg = "#202020" if dark_theme else "#ffffff"

        self._mesh_item: Optional[gl.GLMeshItem] = None
        self._axis_item: Optional[gl.GLAxisItem] = None
        self._current_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._light_shader = _create_directional_light_shader()
        self._flat_shader = _create_flat_color_shader()
        self._lighting_enabled = self._light_shader is not None

        layout = QVBoxLayout(self)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(self._canvas_bg)
        self.view.opts["distance"] = 200
        self.view.opts["elevation"] = 20
        self.view.opts["azimuth"] = -60
        layout.addWidget(self.view, 1)

        controls = QGroupBox("Display settings")
        controls_layout = QGridLayout(controls)
        controls_layout.setColumnStretch(1, 1)

        # Provide a handful of constant colours so users can quickly switch the
        # appearance without diving into advanced scalar options.
        self._colour_presets: list[tuple[str, str]] = [
            ("Clay", "#d87c5a"),
            ("Slate", "#4f6d7a"),
            ("Bone", "#e9d8a6"),
            ("Forest", "#2a9d8f"),
            ("Carbon", "#5c5f66"),
        ]

        row = 0
        controls_layout.addWidget(QLabel("Surface colour:"), row, 0)
        self.color_combo = QComboBox()
        for name, colour in self._colour_presets:
            self.color_combo.addItem(name, userData=colour)
        controls_layout.addWidget(self.color_combo, row, 1, 1, 2)

        row += 1
        controls_layout.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(85)
        controls_layout.addWidget(self.opacity_slider, row, 1)
        self.opacity_label = QLabel("0.85")
        controls_layout.addWidget(self.opacity_label, row, 2)

        row += 1
        self.axes_checkbox = QCheckBox("Show axes")
        controls_layout.addWidget(self.axes_checkbox, row, 0, 1, 3)

        row += 1
        self.light_checkbox = QCheckBox("Enable lighting")
        self.light_checkbox.setChecked(self._lighting_enabled)
        controls_layout.addWidget(self.light_checkbox, row, 0, 1, 3)
        if self._light_shader is None and self._flat_shader is None:
            self._lighting_enabled = False
            self.light_checkbox.setEnabled(False)
            self.light_checkbox.setChecked(False)

        layout.addWidget(controls)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._initialising = True

        self.color_combo.currentIndexChanged.connect(self._on_color_change)
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        self.axes_checkbox.toggled.connect(self._on_axes_toggle)
        self.light_checkbox.toggled.connect(self._on_light_toggle)

        if self._vertices.size:
            mins = np.min(self._vertices, axis=0).astype(np.float32, copy=False)
            maxs = np.max(self._vertices, axis=0).astype(np.float32, copy=False)
        else:
            mins = np.zeros(3, dtype=np.float32)
            maxs = np.zeros(3, dtype=np.float32)
        self._update_scene_bounds(mins, maxs)
        self._update_axis_item()
        self._update_light_shader()
        self._update_mesh()

        self._initialising = False
        self._update_status()

    def _current_colour_rgba(self) -> np.ndarray:
        colour_hex = self.color_combo.currentData()
        if not colour_hex:
            colour_hex = "#d87c5a"
        rgb = np.array(mcolors.to_rgb(str(colour_hex)), dtype=np.float32)
        alpha = np.clip(self.opacity_slider.value() / 100.0, 0.1, 1.0)
        colours = np.empty((self._vertices.shape[0], 4), dtype=np.float32)
        colours[:, :3] = rgb
        colours[:, 3] = alpha
        return colours

    def _update_mesh(self) -> None:
        if self._vertices.size == 0 or self._faces.size == 0:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self.status_label.setText("Surface mesh has no vertices or faces to render.")
            return

        meshdata = gl.MeshData(vertexes=self._vertices, faces=self._faces)
        colours = self._current_colour_rgba()
        meshdata.setVertexColors(colours)

        if self._mesh_item is None:
            self._mesh_item = gl.GLMeshItem(
                meshdata=meshdata,
                smooth=False,
                shader="shaded",
                drawEdges=False,
            )
            self.view.addItem(self._mesh_item)
        else:
            self._mesh_item.setMeshData(meshdata=meshdata)

        self._apply_mesh_shader()
        alpha = self.opacity_slider.value() / 100.0
        self._mesh_item.setGLOptions("translucent" if alpha < 0.999 else "opaque")
        if not self._initialising:
            self.view.update()

    def _update_status(self) -> None:
        if self._vertices.size == 0 or self._faces.size == 0:
            summary = "Surface mesh has no vertices or faces to render."
        else:
            mins, maxs = self._current_bounds if self._current_bounds else (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            )
            summary = (
                f"Surface mesh: {self._vertices.shape[0]:,} vertices / "
                f"{self._faces.shape[0]:,} faces. Opacity {self.opacity_slider.value() / 100.0:.2f}. "
                f"Colour preset: {self.color_combo.currentText()}. "
                f"Bounds X {mins[0]:.1f}–{maxs[0]:.1f} mm, "
                f"Y {mins[1]:.1f}–{maxs[1]:.1f} mm, Z {mins[2]:.1f}–{maxs[2]:.1f} mm. "
            )
        if self._light_shader is None and self._flat_shader is None:
            summary += " Lighting controls unavailable."
        else:
            summary += " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        self.status_label.setText(summary)

    def _apply_mesh_shader(self) -> None:
        if self._mesh_item is None:
            return
        if self._light_shader is not None and self._lighting_enabled:
            self._mesh_item.setShader(self._light_shader)
        elif self._flat_shader is not None:
            self._mesh_item.setShader(self._flat_shader)
        else:
            self._mesh_item.setShader("shaded")

    def _update_light_shader(self) -> None:
        if self._light_shader is None:
            return
        self._light_shader["lightDir"] = [0.5, 0.3, 0.8]
        self._light_shader["lightParams"] = [1.0, 0.35]
        self._apply_mesh_shader()

    def _update_scene_bounds(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        spans = np.maximum(maxs - mins, 1e-3)
        center = (mins + maxs) / 2.0
        self._current_bounds = (mins, maxs)
        self.view.opts["center"] = pg.Vector(center[0], center[1], center[2])
        self.view.opts["distance"] = float(np.linalg.norm(spans) * 1.2)

    def _update_axis_item(self) -> None:
        if not self.axes_checkbox.isChecked():
            if self._axis_item is not None:
                self.view.removeItem(self._axis_item)
                self._axis_item = None
            return
        if self._axis_item is None:
            axis = _AdjustableAxisItem() if '_AdjustableAxisItem' in globals() else gl.GLAxisItem()
            axis.setSize(1, 1, 1)
            self.view.addItem(axis)
            self._axis_item = axis
        mins, maxs = self._current_bounds if self._current_bounds else (
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )
        spans = np.maximum(maxs - mins, 1e-3)
        self._axis_item.setSize(spans[0], spans[1], spans[2])
        if hasattr(self._axis_item, "setLineWidth"):
            self._axis_item.setLineWidth(2.0)

    def _on_color_change(self, _index: int) -> None:
        self._update_mesh()
        if not self._initialising:
            self._update_status()

    def _on_opacity_change(self, value: int) -> None:
        self.opacity_label.setText(f"{value / 100.0:.2f}")
        self._update_mesh()
        if not self._initialising:
            self._update_status()

    def _on_axes_toggle(self, _checked: bool) -> None:
        self._update_axis_item()
        if not self._initialising:
            self.view.update()

    def _on_light_toggle(self, checked: bool) -> None:
        self._lighting_enabled = bool(checked)
        self._apply_mesh_shader()
        if not self._initialising and self._mesh_item is not None:
            self._mesh_item.update()
        self._update_status()

class MetadataViewer(QWidget):
    """
    Metadata viewer/editor for JSON and TSV sidecars (from bids_editor_ancpbids).
    """
    def __init__(self):
        super().__init__()
        # Layout consists of a small toolbar followed by the actual viewer
        # widget.  A welcome message is shown when no file is loaded.
        vlay = QVBoxLayout(self)
        self.welcome = QLabel(
            "<h3>Metadata BIDSualizer</h3><br>Load data via File → Open or select a file to begin editing."
        )
        self.welcome.setAlignment(Qt.AlignCenter)
        vlay.addWidget(self.welcome)
        self.toolbar = QHBoxLayout()
        vlay.addLayout(self.toolbar)
        self.value_row = QHBoxLayout()
        vlay.addLayout(self.value_row)
        # Label used to show a spinner while a file is being loaded. It is
        # created as an overlay so it always appears above whatever viewer is
        # currently shown.  The font is enlarged for better visibility.
        self.loading_label = QLabel(self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet(
            "font-size: 24px; background-color: rgba(0, 0, 0, 128);"
            "color: white;"
        )
        self.loading_label.hide()

        # Timer and frame sequence driving the spinner animation.
        self._load_timer = QTimer()
        self._load_timer.timeout.connect(self._spin_loading)
        self._load_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self._load_index = 0  # Current index into ``_load_frames``
        self._load_message = ""  # Message displayed next to the spinner
        self.viewer = None
        self.current_path = None
        self.data = None  # holds loaded NIfTI data when viewing images
        self.surface_data: Optional[dict[str, Any]] = None

    def clear(self):
        """Clear the toolbar and viewer when switching files."""
        def _clear_layout(lay):
            while lay.count():
                item = lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    _clear_layout(item.layout())
            lay.deleteLater()

        while self.toolbar.count():
            item = self.toolbar.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                _clear_layout(item.layout())
        while self.value_row.count():
            item = self.value_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                _clear_layout(item.layout())
        if self.viewer:
            self.layout().removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None
        self.data = None
        self.surface_data = None
        self.nifti_img = None
        self._nifti_meta = {}
        self.loading_label.hide()
        self._load_timer.stop()
        self.welcome.show()

    def _is_dark_theme(self) -> bool:
        """Detect whether the current palette is dark or light."""
        color = self.palette().color(QPalette.Window)
        brightness = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        return brightness < 128

    def _start_loading(self, message: str) -> None:
        """Begin showing the loading spinner with ``message``."""
        self._load_message = message
        self._load_index = 0
        self.loading_label.setText(f"{message} {self._load_frames[0]}")
        self.loading_label.setGeometry(0, 0, self.width(), self.height())
        self.loading_label.raise_()
        self.loading_label.show()
        self._load_timer.start(100)

    def _spin_loading(self) -> None:
        """Advance the spinner animation by one frame."""
        if not self.loading_label.isVisible():
            return
        self._load_index = (self._load_index + 1) % len(self._load_frames)
        self.loading_label.setText(
            f"{self._load_message} {self._load_frames[self._load_index]}"
        )

    def _stop_loading(self) -> None:
        """Hide the loading spinner and stop the timer."""
        self._load_timer.stop()
        self.loading_label.hide()

    def _get_nifti_data(self, img):
        """Load a NIfTI image into an ``ndarray`` while handling RGB/struct dtypes.

        Some derived NIfTI maps (for example colour fractional anisotropy
        volumes) store multi-component voxels in a structured ``void`` dtype.
        ``nibabel`` cannot promote those to floats, which previously caused the
        viewer to crash.  We detect this situation, convert the array into a
        plain ``float`` ndarray and record metadata describing the additional
        vector axis so the rest of the viewer can render it correctly.
        """

        dtype_exc = None
        try:
            data = img.get_fdata()
            return data, {}
        except Exception as exc:
            # Only handle the structured-dtype failure; re-raise other errors so
            # they surface during debugging instead of being silently masked.
            is_dtype_error = False
            if np_exceptions is not None and isinstance(exc, np_exceptions.DTypePromotionError):
                is_dtype_error = True
            elif exc.__class__.__name__ == "DTypePromotionError":
                is_dtype_error = True
            elif "VoidDType" in str(exc):
                is_dtype_error = True
            if not is_dtype_error:
                raise
            dtype_exc = exc

        if rfn is None:
            raise RuntimeError(
                "Structured NIfTI data requires NumPy's recfunctions module to "
                "convert it into a regular array."
            ) from dtype_exc

        dataobj = np.asanyarray(img.dataobj)
        if not getattr(dataobj.dtype, "fields", None):
            # Unexpected dtype: ``get_fdata`` failed but the array is not
            # structured.  Re-raise the original exception so the caller knows.
            raise dtype_exc

        # ``structured_to_unstructured`` flattens the fields into the last axis
        # and keeps the voxel geometry intact, giving us a standard ndarray.
        unstructured = rfn.structured_to_unstructured(dataobj)
        vector_length = (
            int(unstructured.shape[-1])
            if unstructured.ndim == dataobj.ndim + 1
            else 1
        )
        meta = {
            "vector_axis": len(img.shape),
            "vector_length": vector_length,
            # Treat 3/4-component vectors as colour channels so they can be
            # rendered directly instead of exposing them as separate volumes.
            "is_rgb": vector_length in (3, 4)
            and unstructured.ndim == dataobj.ndim + 1,
        }
        # ``structured_to_unstructured`` already yields a float array when the
        # original data was floating point.  Cast explicitly to float32 so we do
        # not unnecessarily inflate the array when the original values were
        # stored as 32-bit floats.
        return unstructured.astype(np.float32, copy=False), meta

    def load_file(self, path: Path):
        """Load JSON, TSV, NIfTI or DICOM file into an editable viewer.

        The file is read in a background thread to keep the UI responsive and
        animate the loading spinner while potentially large datasets are processed.
        """

        self.current_path = path
        self.clear()
        self.welcome.hide()
        ext = _get_ext(path)
        lower_name = path.name.lower()
        gifti_candidate = any(lower_name.endswith(suffix) for suffix in GIFTI_SURFACE_SUFFIXES)
        freesurfer_candidate = any(lower_name.endswith(suffix) for suffix in FREESURFER_SURFACE_SUFFIXES)
        dicom = is_dicom_file(str(path))
        self._start_loading("Loading")
        # ``worker`` reads the file in a separate thread so the UI can keep
        # updating the spinner while potentially large data is loaded.
        result = {}
        load_error: Optional[Exception] = None

        def worker():
            nonlocal load_error
            try:
                if ext == '.json':
                    result['data'] = json.loads(path.read_text(encoding='utf-8'))
                elif ext == '.tsv':
                    result['df'] = pd.read_csv(path, sep='\t', keep_default_na=False)
                elif ext in ['.nii', '.nii.gz']:
                    img = nib.load(str(path))
                    result['img'] = img
                    data, meta = self._get_nifti_data(img)
                    result['data'] = data
                    result['nifti_meta'] = meta
                elif gifti_candidate:
                    surface = self._load_gifti_surface(path)
                    result['surface'] = surface
                    result['surface_type'] = 'gifti'
                elif freesurfer_candidate:
                    surface = self._load_freesurfer_surface(path)
                    result['surface'] = surface
                    result['surface_type'] = 'freesurfer'
                elif dicom:
                    # ``stop_before_pixels`` avoids loading heavy pixel data
                    dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
                    result['ds'] = dataset
                    # Preparing the nested metadata representation inside the
                    # worker keeps the GUI responsive when large headers are
                    # parsed.
                    result['dicom_tree'] = self._dicom_dataset_to_tree(dataset)
            except Exception as exc:  # pragma: no cover - interactive load errors
                load_error = exc

        if ext in ['.json', '.tsv', '.nii', '.nii.gz'] or dicom or gifti_candidate or freesurfer_candidate:
            t = threading.Thread(target=worker)
            t.start()
            while t.is_alive():
                QApplication.processEvents()
                time.sleep(0.05)
            t.join()
        self._stop_loading()

        if load_error is not None:
            logging.exception("Failed to load %s", path)
            QMessageBox.critical(
                self,
                "Open",
                f"Unable to load {path.name}: {load_error}",
            )
            self.welcome.show()
            return

        if ext == '.json':
            self._setup_json_toolbar()
            self.viewer = self._json_view(path, result.get('data'))
        elif ext == '.tsv':
            self._setup_tsv_toolbar()
            self.viewer = self._tsv_view(path, result.get('df'))
        elif ext in ['.nii', '.nii.gz']:
            self._setup_nifti_toolbar()
            self.viewer = self._nifti_view(
                path,
                (
                    result.get('img'),
                    result.get('data'),
                    result.get('nifti_meta'),
                ),
            )
        elif result.get('surface'):
            self.surface_data = result['surface']
            if result.get('surface_type'):
                self.surface_data['type'] = result['surface_type']
            self._setup_surface_toolbar()
            self.viewer = self._surface_view(path, self.surface_data)
        elif dicom:
            self.viewer = self._dicom_view(path, result.get('dicom_tree'))
            self.toolbar.addStretch()
        elif ext in ['.html', '.htm']:
            self.viewer = self._html_view(path)
            self.toolbar.addStretch()

        self.layout().addWidget(self.viewer)

    def resizeEvent(self, event):
        """Ensure images rescale when the window size decreases."""
        super().resizeEvent(event)
        self.loading_label.setGeometry(0, 0, self.width(), self.height())
        # If a NIfTI image is currently loaded, update the displayed slice
        if (
            self.data is not None
            and self.current_path
            and _get_ext(self.current_path) in ['.nii', '.nii.gz']
            and hasattr(self, 'img_label')
        ):
            self._update_slice()

    def _setup_surface_toolbar(self) -> None:
        """Toolbar for surface files featuring a single 3-D view button."""

        self.view_surface_btn = QPushButton("Surface View")
        self.view_surface_btn.clicked.connect(self._show_surface_view)
        self.toolbar.addWidget(self.view_surface_btn)
        self.toolbar.addStretch()

    def _surface_view(self, path: Path, surface: dict[str, Any]) -> QWidget:
        """Display a textual summary for a loaded surface mesh."""

        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setMinimumHeight(220)

        raw_vertices = surface.get('vertices')
        raw_faces = surface.get('faces')
        vertices = np.asarray(
            raw_vertices if raw_vertices is not None else np.empty((0, 3), dtype=np.float32),
            dtype=np.float32,
        )
        faces = np.asarray(
            raw_faces if raw_faces is not None else np.empty((0, 3), dtype=np.int32),
            dtype=np.int32,
        )
        scalars = surface.get('scalars') or {}
        meta = surface.get('meta') or {}

        lines = [
            f"File: {path.name}",
            f"Vertices: {vertices.shape[0]:,}",
            f"Faces: {faces.shape[0]:,}",
        ]

        if vertices.size:
            mins = vertices.min(axis=0)
            maxs = vertices.max(axis=0)
            extent_text = (
                f"Bounds (mm): X {mins[0]:.2f} – {maxs[0]:.2f}, "
                f"Y {mins[1]:.2f} – {maxs[1]:.2f}, "
                f"Z {mins[2]:.2f} – {maxs[2]:.2f}"
            )
            lines.append(extent_text)

        if scalars:
            lines.append("Scalar fields:")
            for name, values in scalars.items():
                arr = np.asarray(values, dtype=np.float32)
                if arr.size == 0:
                    continue
                vmin = float(np.min(arr))
                vmax = float(np.max(arr))
                lines.append(f"  • {name}: range {vmin:.4g} – {vmax:.4g}")
        else:
            lines.append("No embedded scalar fields detected.")

        structure = meta.get('structure')
        if structure:
            lines.append(f"Structure: {structure}")

        coord = meta.get('coordinate_system') or {}
        if coord:
            dataspace = coord.get('dataspace') or 'unknown'
            xformspace = coord.get('xformspace') or 'unknown'
            lines.append(f"Coordinate system: {dataspace} → {xformspace}")

        gifti_meta = meta.get('gifti_meta') or {}
        if gifti_meta:
            lines.append("GIFTI metadata:")
            for key in sorted(gifti_meta):
                lines.append(f"  • {key}: {gifti_meta[key]}")

        fs_header = meta.get('freesurfer_header') or {}
        if fs_header:
            lines.append("FreeSurfer metadata:")
            for key in sorted(fs_header):
                lines.append(f"  • {key}: {fs_header[key]}")

        info.setPlainText("\n".join(lines))
        layout.addWidget(info)

        hint = QLabel(
            "Use the Surface View button above to explore the mesh interactively."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()
        return widget

    def _load_gifti_surface(self, path: Path) -> dict[str, Any]:
        """Load a GIFTI surface, returning vertices, faces and scalar data."""

        img = nib.load(str(path))
        coords = None
        faces = None
        scalars: dict[str, np.ndarray] = {}
        structure = ""
        coord_info: dict[str, Any] = {}

        for arr in img.darrays:
            intent = getattr(arr, 'intent', '') or ''
            data = np.asarray(arr.data)
            if intent == 'NIFTI_INTENT_POINTSET':
                coords = data.astype(np.float32, copy=False)
                structure = arr.meta.get('AnatomicalStructurePrimary') or structure
                cs = getattr(arr, 'coordsys', None)
                if cs is not None:
                    coord_info = {
                        'dataspace': getattr(cs, 'dataspace', ''),
                        'xformspace': getattr(cs, 'xformspace', ''),
                        'xform': getattr(cs, 'xform', None),
                    }
            elif intent == 'NIFTI_INTENT_TRIANGLE':
                faces = data.astype(np.int32, copy=False)
            else:
                if coords is None or data.shape[0] != coords.shape[0]:
                    continue
                if data.ndim == 1:
                    candidate = np.nan_to_num(
                        data.astype(np.float32, copy=False),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    name = arr.meta.get('Name') or f"Field {len(scalars) + 1}"
                    scalars[str(name)] = candidate
                elif data.ndim == 2 and data.shape[1] == 1:
                    candidate = np.nan_to_num(
                        data[:, 0].astype(np.float32, copy=False),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    name = arr.meta.get('Name') or f"Field {len(scalars) + 1}"
                    scalars[str(name)] = candidate

        if coords is None or faces is None:
            raise ValueError("GIFTI file does not contain surface geometry")

        gifti_meta = {}
        try:
            pairs = getattr(img.meta, 'data', None) or []
            gifti_meta = {str(item.name): str(item.value) for item in pairs}
        except Exception:  # pragma: no cover - defensive fallback
            gifti_meta = {}

        return {
            'vertices': coords,
            'faces': faces,
            'scalars': scalars,
            'meta': {
                'structure': structure,
                'coordinate_system': coord_info,
                'gifti_meta': gifti_meta,
            },
        }

    def _load_freesurfer_surface(self, path: Path) -> dict[str, Any]:
        """Load a FreeSurfer ``*.pial``/``*.white`` style surface mesh."""

        try:
            from nibabel.freesurfer import io as fsio
        except Exception as exc:  # pragma: no cover - optional dependency issues
            raise RuntimeError("nibabel FreeSurfer support is unavailable") from exc

        try:
            verts, faces, header = fsio.read_geometry(str(path), read_metadata=True)
        except TypeError:  # pragma: no cover - older nibabel releases
            verts, faces = fsio.read_geometry(str(path))
            header = {}

        header_dict = {}
        if isinstance(header, dict):
            header_dict = {str(k): str(v) for k, v in header.items()}

        return {
            'vertices': verts.astype(np.float32, copy=False),
            'faces': faces.astype(np.int32, copy=False),
            'scalars': {},
            'meta': {
                'structure': Path(path).stem,
                'freesurfer_header': header_dict,
            },
        }

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

        self.view3d_btn = QPushButton("3D View")
        self.view3d_btn.clicked.connect(self._show_3d_view)
        self.toolbar.addWidget(self.view3d_btn)

        self.graph_btn = QPushButton("Graph")
        self.graph_btn.setCheckable(True)
        self.graph_btn.clicked.connect(self._toggle_graph)
        self.toolbar.addWidget(self.graph_btn)

        # Helper to add slider+label vertically
        def add_slider(title, slider, val_label=None):
            box = QVBoxLayout()
            lab = QLabel(title)
            lab.setAlignment(Qt.AlignCenter)
            box.addWidget(lab)
            row = QHBoxLayout()
            row.addWidget(slider)
            if val_label is not None:
                row.addWidget(val_label)
            box.addLayout(row)
            self.toolbar.addLayout(box)

        # Slice slider
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self._update_slice)
        self.slice_val = QLabel("0")
        add_slider("Slice", self.slice_slider, self.slice_val)

        # Volume slider
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.valueChanged.connect(self._update_slice)
        self.vol_val = QLabel("0")
        add_slider("Volume", self.vol_slider, self.vol_val)
        # Brightness slider
        self.bright_slider = QSlider(Qt.Horizontal)
        self.bright_slider.setRange(-100, 100)
        self.bright_slider.setValue(0)
        self.bright_slider.valueChanged.connect(self._update_slice)
        add_slider("Brightness", self.bright_slider)

        # Contrast slider
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self._update_slice)
        add_slider("Contrast", self.contrast_slider)
        self.voxel_val_label = QLabel("N/A")
        self.value_row.addWidget(QLabel("Voxel value:"))
        self.value_row.addWidget(self.voxel_val_label)
        self.value_row.addStretch()
        self.toolbar.addStretch()

    def _show_3d_view(self) -> None:
        """Launch the interactive 3-D renderer for the loaded volume."""

        if getattr(self, "data", None) is None:
            QMessageBox.warning(self, "3D Viewer", "No NIfTI volume is currently loaded.")
            return

        voxel_sizes = None
        if getattr(self, "nifti_img", None) is not None:
            try:
                zooms = self.nifti_img.header.get_zooms()
                voxel_sizes = tuple(float(v) for v in zooms[:3])
            except Exception:  # pragma: no cover - defensive: malformed header
                voxel_sizes = None

        default_mode = None
        if self.current_path:
            name = self.current_path.name.lower()
            if any(tag in name for tag in ("dwi", "dti", "diffusion")):
                default_mode = "RMS (DTI)"

        title = (
            f"3D View – {self.current_path.name}"
            if self.current_path is not None
            else "3D Volume Viewer"
        )

        try:
            dialog = Volume3DDialog(
                self,
                self.data,
                meta=getattr(self, "_nifti_meta", {}),
                voxel_sizes=voxel_sizes,
                default_mode=default_mode,
                title=title,
                dark_theme=self._is_dark_theme(),
            )
        except Exception as exc:  # pragma: no cover - interactive error reporting
            logging.exception("Failed to initialise 3-D viewer")
            QMessageBox.critical(
                self,
                "3D Viewer",
                f"Unable to open 3D view: {exc}",
            )
            return

        dialog.exec_()

    def _show_surface_view(self) -> None:
        """Launch the interactive surface renderer for the loaded mesh."""

        if not self.surface_data:
            QMessageBox.warning(self, "Surface Viewer", "No surface mesh is currently loaded.")
            return

        title = (
            f"Surface View – {self.current_path.name}"
            if self.current_path is not None
            else "Surface Viewer"
        )

        surface_type = str(self.surface_data.get('type') or '').lower()

        try:
            if surface_type == 'freesurfer':
                dialog = FreeSurferSurfaceDialog(
                    self,
                    self.surface_data.get('vertices'),
                    self.surface_data.get('faces'),
                    title=title,
                    dark_theme=self._is_dark_theme(),
                )
            else:
                dialog = Surface3DDialog(
                    self,
                    self.surface_data.get('vertices'),
                    self.surface_data.get('faces'),
                    scalars=self.surface_data.get('scalars'),
                    meta=self.surface_data.get('meta'),
                    title=title,
                    dark_theme=self._is_dark_theme(),
                )
        except Exception as exc:  # pragma: no cover - interactive error reporting
            logging.exception("Failed to initialise surface viewer")
            QMessageBox.critical(
                self,
                "Surface Viewer",
                f"Unable to open surface view: {exc}",
            )
            return

        dialog.exec_()

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
        slider = getattr(self, 'vol_slider', None)
        vol_idx = slider.value() if slider is not None else 0
        vol = self._get_volume_data(vol_idx)
        axis_len = vol.shape[axis]
        self.slice_slider.setMaximum(max(axis_len - 1, 0))
        self.slice_slider.setEnabled(axis_len > 1)
        self.slice_slider.setValue(axis_len // 2)
        self.slice_val.setText(str(axis_len // 2))
        self._update_slice()

    def _nifti_view(self, path: Path, img_data=None) -> QWidget:
        """Create a simple viewer for NIfTI images with slice/volume controls."""
        meta = {}
        if img_data is None:
            self.nifti_img = nib.load(str(path))
            data, meta = self._get_nifti_data(self.nifti_img)
        else:
            if isinstance(img_data, tuple) and len(img_data) >= 2:
                self.nifti_img = img_data[0]
                data = img_data[1]
                if len(img_data) >= 3 and isinstance(img_data[2], dict):
                    meta = img_data[2] or {}
            else:
                self.nifti_img = img_data
                data, meta = self._get_nifti_data(self.nifti_img)

        if data is None:
            data, meta = self._get_nifti_data(self.nifti_img)

        self.data = data
        self._nifti_meta = meta or {}
        self._nifti_is_color = bool(self._nifti_meta.get("is_rgb"))
        widget = QWidget()
        vlay = QVBoxLayout(widget)

        self.cross_voxel = [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2]
        self._img_scale = (1.0, 1.0)

        self.img_label = _ImageLabel(self._update_slice, self._on_image_clicked)
        self.img_label.setAlignment(Qt.AlignCenter)
        # Allow the image to shrink as well as expand when resizing
        self.img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.img_label.setMinimumSize(1, 1)

        # Center the image label within a container so resizing doesn't shift
        # it vertically when the splitter changes size.
        img_container = QWidget()
        ic_layout = QVBoxLayout(img_container)
        ic_layout.setContentsMargins(0, 0, 0, 0)
        ic_layout.addWidget(self.img_label)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(img_container)

        # Graph panel with scope selector
        self.graph_panel = QWidget()
        g_lay = QVBoxLayout(self.graph_panel)
        g_lay.setContentsMargins(0, 0, 0, 0)
        g_lay.setSpacing(2)

        self.graph_canvas = FigureCanvas(plt.Figure(figsize=(4, 2)))
        self.graph_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.graph_canvas.figure.subplots()
        g_lay.addWidget(self.graph_canvas)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self.scope_spin = QSpinBox()
        self.scope_spin.setRange(1, 4)
        self.scope_spin.setValue(1)
        self.scope_spin.valueChanged.connect(self._update_graph)
        scope_row.addWidget(self.scope_spin)
        scope_row.addSpacing(10)
        scope_row.addWidget(QLabel("Dot size:"))
        self.dot_size_spin = QSpinBox()
        self.dot_size_spin.setRange(1, 20)
        self.dot_size_spin.setValue(6)
        self.dot_size_spin.valueChanged.connect(self._update_graph)
        scope_row.addWidget(self.dot_size_spin)
        scope_row.addSpacing(15)
        self.mark_neighbors_box = QCheckBox("Mark neighbors")
        self.mark_neighbors_box.setChecked(True)
        self.mark_neighbors_box.stateChanged.connect(self._update_graph)
        scope_row.addWidget(self.mark_neighbors_box)
        scope_row.addStretch()
        g_lay.addLayout(scope_row)

        self.graph_panel.setVisible(False)
        self.splitter.addWidget(self.graph_panel)
        # Allow the image and graph to share space evenly when the graph is shown
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        vlay.addWidget(self.splitter)

        # Configure volume slider range. Structured RGB maps expose the last
        # axis as colour channels, not temporal volumes, so we keep the slider
        # disabled for those datasets.
        if data.ndim == 4 and not self._nifti_is_color:
            n_vols = data.shape[3]
        else:
            n_vols = 1
        self.vol_slider.setMaximum(max(n_vols - 1, 0))
        self.vol_slider.setEnabled(n_vols > 1)
        self.vol_slider.setValue(0)
        self.vol_val.setText("0")
        self.graph_btn.setVisible(n_vols > 1)

        # Initialize orientation and slice slider
        self._set_orientation(self.orientation)
        self._update_slice()
        return widget

    def _get_volume_data(self, vol_idx: int | None = None):
        """Return the 3-D volume used for display.

        Regular 4-D datasets encode the temporal dimension on the last axis and
        should be indexed by ``vol_idx``.  RGB/structured data keeps colour
        channels on the last axis which must remain untouched so they can be
        rendered as true-colour images.
        """

        if vol_idx is None:
            slider = getattr(self, 'vol_slider', None)
            vol_idx = slider.value() if slider is not None else 0

        if self.data.ndim == 4 and not getattr(self, '_nifti_is_color', False):
            vol_idx = max(0, min(vol_idx, self.data.shape[3] - 1))
            return self.data[..., vol_idx]
        return self.data

    def _update_slice(self):
        """Update displayed slice when slider moves."""
        slider = getattr(self, 'vol_slider', None)
        vol_idx = slider.value() if slider is not None else 0
        vol = self._get_volume_data(vol_idx)
        axis = getattr(self, 'orientation', 2)
        slice_idx = getattr(self, 'slice_slider', None).value() if hasattr(self, 'slice_slider') else vol.shape[axis] // 2
        self.slice_val.setText(str(slice_idx))
        self.vol_val.setText(str(vol_idx))
        if axis == 0:
            slice_img = vol[slice_idx, :, :]
        elif axis == 1:
            slice_img = vol[:, slice_idx, :]
        else:
            slice_img = vol[:, :, slice_idx]
        # Normalise the slice to 0..1 before applying display adjustments
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
        if arr.ndim == 2:
            h, w = arr.shape
            img = QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, c = arr.shape
            fmt = QImage.Format_RGB888 if c == 3 else QImage.Format_RGBA8888
            bytes_per_line = w * c
            img = QImage(arr.tobytes(), w, h, bytes_per_line, fmt)
        pix = QPixmap.fromImage(img)

        scaled = pix.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img_scale = (scaled.width() / w, scaled.height() / h)

        # Draw crosshair after scaling for consistent width
        if self.cross_voxel is not None:
            x_rot, y_rot = self._voxel_to_arr(self.cross_voxel)
            scale_x, scale_y = self._img_scale
            x_s = int(x_rot * scale_x)
            y_s = int(y_rot * scale_y)
            # Use the highlight color so the crosshair is visible on any theme
            painter = QPainter(scaled)
            theme_color = self.palette().color(QPalette.Highlight)
            pen = QPen(theme_color)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(x_s, 0, x_s, scaled.height())
            painter.drawLine(0, y_s, scaled.width(), y_s)
            square = max(2, int(min(scaled.width(), scaled.height()) * 0.02))
            half = square // 2
            painter.drawRect(x_s - half, y_s - half, square, square)
            painter.end()

        self.img_label.setPixmap(scaled)

        self._update_value()
        if self.graph_panel.isVisible():
            self._update_graph_marker()

    def _label_pos_to_img_coords(self, pos):
        # Convert click position on the scaled QLabel back to image coordinates
        pix = self.img_label.pixmap()
        if pix is None:
            return None
        pw, ph = pix.width(), pix.height()
        lw, lh = self.img_label.width(), self.img_label.height()
        off_x, off_y = (lw - pw) / 2, (lh - ph) / 2
        x = pos.x() - off_x
        y = pos.y() - off_y
        if 0 <= x < pw and 0 <= y < ph:
            scale_x, scale_y = self._img_scale
            return int(x / scale_x), int(y / scale_y)
        return None

    def _arr_to_voxel(self, x, y):
        # Map 2D display coordinates back to a voxel index within the volume
        vol_idx = self.vol_slider.value()
        vol = self._get_volume_data(vol_idx)
        axis = self.orientation
        slice_idx = self.slice_slider.value()
        if axis == 0:
            j = x
            k = vol.shape[2] - 1 - y
            return slice_idx, j, k
        elif axis == 1:
            i = x
            k = vol.shape[2] - 1 - y
            return i, slice_idx, k
        else:
            i = x
            j = vol.shape[1] - 1 - y
            return i, j, slice_idx

    def _voxel_to_arr(self, voxel):
        """Convert a voxel index back to 2-D array coordinates for drawing."""
        i, j, k = voxel
        vol_idx = self.vol_slider.value()
        vol = self._get_volume_data(vol_idx)
        axis = self.orientation
        if axis == 0:
            x = j
            y = vol.shape[2] - 1 - k
        elif axis == 1:
            x = i
            y = vol.shape[2] - 1 - k
        else:
            x = i
            y = vol.shape[1] - 1 - j
        return x, y

    def _on_image_clicked(self, event):
        coords = self._label_pos_to_img_coords(event.pos())
        if coords:
            voxel = self._arr_to_voxel(*coords)
            self.cross_voxel = list(voxel)
            self._update_slice()
            if self.graph_panel.isVisible():
                self._update_graph()

    def _update_value(self):
        if self.cross_voxel is None:
            self.voxel_val_label.setText("N/A")
            return
        vol_idx = self.vol_slider.value()
        i, j, k = self.cross_voxel
        if getattr(self, '_nifti_is_color', False):
            vec = np.asarray(self.data[i, j, k, :], dtype=float)
            if vec.size == 0:
                self.voxel_val_label.setText("N/A")
            else:
                components = ", ".join(f"{v:.3g}" for v in vec)
                self.voxel_val_label.setText(f"[{components}]")
            return

        if self.data.ndim == 4:
            val = self.data[i, j, k, vol_idx]
        else:
            val = self.data[i, j, k]
        self.voxel_val_label.setText(f"{float(val):.3g}")

    def _toggle_graph(self):
        visible = self.graph_btn.isChecked()
        self.graph_panel.setVisible(visible)
        total = self.splitter.size().height()
        if visible:
            self.splitter.setSizes([total // 2, total // 2])
            self._update_graph()
        else:
            self.splitter.setSizes([total, 0])

    def _update_graph(self):
        # Redraw the time-series graph for all voxels in the selected neighborhood
        # around ``self.cross_voxel``. Only valid for 4-D data.
        if (
            self.data.ndim != 4
            or self.cross_voxel is None
            or getattr(self, '_nifti_is_color', False)
        ):
            return

        level = self.scope_spin.value()
        dim = 2 * (level - 1) + 1
        half = dim // 2
        i0, j0, k0 = self.cross_voxel
        orient = self.orientation

        self.graph_canvas.figure.clf()
        axes = self.graph_canvas.figure.subplots(
            dim, dim, squeeze=False, sharex=True, sharey=True
        )

        line_color = "#000000" if not self._is_dark_theme() else "#ffffff"
        marker_color = self.palette().color(QPalette.Highlight).name()
        bg_color = self.palette().color(QPalette.Base).name()
        dot_size = self.dot_size_spin.value()
        self.graph_canvas.figure.set_facecolor(bg_color)
        self.markers = []
        self.marker_ts = []
        global_min = float("inf")
        global_max = float("-inf")

        for r, di in enumerate(range(-half, half + 1)):
            for c, dj in enumerate(range(-half, half + 1)):
                ax = axes[r][c]
                i, j, k = i0, j0, k0
                if orient == 0:
                    j = j0 + di
                    k = k0 + dj
                elif orient == 1:
                    i = i0 + di
                    k = k0 + dj
                else:
                    i = i0 + di
                    j = j0 + dj

                if not (0 <= i < self.data.shape[0] and 0 <= j < self.data.shape[1] and 0 <= k < self.data.shape[2]):
                    ax.axis("off")
                    continue

                ts_orig = self.data[i, j, k, :]
                ts = ts_orig
                global_min = min(global_min, ts_orig.min())
                global_max = max(global_max, ts_orig.max())
                ax.set_facecolor(bg_color)
                ax.plot(ts, color=line_color, linewidth=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False)
                if self.mark_neighbors_box.isChecked() or (r == half and c == half):
                    self.marker_ts.append(ts)
                    idx = self.vol_slider.value()
                    marker, = ax.plot([idx], [ts[idx]], "o", color=marker_color, markersize=dot_size)
                    self.markers.append(marker)

        if global_min < global_max:
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_ylim(global_min, global_max)

        self.graph_canvas.figure.tight_layout(pad=0.1)
        self.graph_canvas.draw()

    def _update_graph_marker(self):
        # Update the marker showing the current volume index on all axes
        if (
            not getattr(self, "markers", None)
            or not getattr(self, "marker_ts", None)
            or getattr(self, '_nifti_is_color', False)
        ):
            return
        marker_color = self.palette().color(QPalette.Highlight).name()
        idx = self.vol_slider.value()
        for marker, ts in zip(self.markers, self.marker_ts):
            i = max(0, min(idx, len(ts) - 1))
            marker.set_data([i], [ts[i]])
            marker.set_color(marker_color)
            marker.set_markersize(self.dot_size_spin.value())
        self.graph_canvas.draw_idle()

    def _json_view(self, path: Path, data=None) -> QTreeWidget:
        """Create a tree widget to show and edit JSON data."""
        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Key", "Value"])
        tree.setAlternatingRowColors(True)
        hdr = tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.Interactive)
        if data is None:
            data = json.loads(path.read_text(encoding='utf-8'))
        self._populate_json(tree.invisibleRootItem(), data)
        tree.expandAll()
        tree.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        return tree

    def _populate_json(self, parent, data, editable: bool = True):
        """Recursively populate JSON-like data into the tree widget."""

        if isinstance(data, dict):
            for k, v in data.items():
                it = QTreeWidgetItem([str(k), '' if isinstance(v, (dict, list)) else str(v)])
                if editable:
                    it.setFlags(it.flags() | Qt.ItemIsEditable)
                parent.addChild(it)
                if isinstance(v, (dict, list)):
                    self._populate_json(it, v, editable)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                it = QTreeWidgetItem([str(i), '' if isinstance(v, (dict, list)) else str(v)])
                if editable:
                    it.setFlags(it.flags() | Qt.ItemIsEditable)
                parent.addChild(it)
                if isinstance(v, (dict, list)):
                    self._populate_json(it, v, editable)

    def _tsv_view(self, path: Path, df=None) -> QTableWidget:
        """Create a table widget to show and edit TSV data."""
        if df is None:
            df = pd.read_csv(path, sep="\t", keep_default_na=False)
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

    def _dicom_dataset_to_tree(self, dataset) -> Dict[str, Any]:
        """Return a nested mapping representing the supplied DICOM dataset.

        Iterating over all elements, especially sequences, can be expensive for
        large headers.  Performing the conversion in a worker thread keeps the
        GUI responsive while still producing a structure that mirrors the
        hierarchical layout of the dataset for display.
        """

        def ds_to_dict(ds) -> Dict[str, Any]:
            mapping: Dict[str, Any] = {}
            if ds is None:
                return mapping
            for elem in ds:
                name = elem.keyword or elem.name
                if elem.VR == "SQ":  # Sequence elements contain nested datasets
                    mapping[name] = [ds_to_dict(item) for item in elem.value]
                else:
                    # ``str`` ensures even binary/text values are rendered in a
                    # readable form inside the tree widget.
                    try:
                        mapping[name] = str(elem.value)
                    except Exception:
                        mapping[name] = "<unavailable>"
            return mapping

        file_meta = getattr(dataset, "file_meta", None)
        return {
            "File Meta Information": ds_to_dict(file_meta),
            "Dataset": ds_to_dict(dataset),
        }

    def _dicom_view(self, path: Path, metadata: Optional[Dict[str, Any]]) -> QTreeWidget:
        """Display precomputed DICOM metadata in a read-only tree."""

        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Tag", "Value"])
        tree.setAlternatingRowColors(True)
        hdr = tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.Interactive)

        data = metadata or {"File Meta Information": {}, "Dataset": {}}
        self._populate_json(tree.invisibleRootItem(), data, editable=False)
        tree.expandAll()
        tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        return tree

    def _html_view(self, path: Path) -> QWebEngineView:
        """Display HTML file using a QWebEngineView for full rendering."""
        view = QWebEngineView()
        view.setUrl(QUrl.fromLocalFile(str(Path(path).resolve())))
        return view

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

class EditorMixin:
    """Behaviour and UI for the Editor tab extracted from ``gui.py``."""

    def initEditTab(self):
        """
        Set up Edit tab to embed the full functionality of bids_editor_ancpbids.
        """
        # This tab provides a file browser, statistics viewer and the metadata
        # editor used to inspect and modify BIDS sidecars.  It mirrors the
        # standalone "bids-editor" utility but is embedded in this application.
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

        intended_act = QAction("Set Intended For…", self)
        intended_act.triggered.connect(self.launchIntendedForEditor)
        tools_menu.addAction(intended_act)

        refresh_act = QAction("Refresh scans.tsv", self)
        refresh_act.triggered.connect(self.refreshScansTsv)
        tools_menu.addAction(refresh_act)

        ignore_act = QAction("Edit .bidsignore…", self)
        ignore_act.triggered.connect(self.launchBidsIgnore)
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
        self.tabs.addTab(self.edit_tab, "Editor")

    def openBIDSForEdit(self):
        """Prompt user to select a BIDS dataset root for editing."""
        p = QFileDialog.getExistingDirectory(self, "Select BIDS dataset")
        if p:
            self.bids_root = Path(p)
            self.model.setRootPath(p)
            self.tree.setRootIndex(self.model.index(p))
            self.viewer.clear()
            self.updateStats()
            self.loadExcludePatterns()

    def onTreeClicked(self, idx: QModelIndex):
        """When a file is clicked, load metadata if supported."""

        p = Path(self.model.filePath(idx))
        self.selected = p
        ext = _get_ext(p)
        lower_name = p.name.lower()
        gifti_candidate = any(lower_name.endswith(suffix) for suffix in GIFTI_SURFACE_SUFFIXES)
        freesurfer_candidate = any(
            lower_name.endswith(suffix) for suffix in FREESURFER_SURFACE_SUFFIXES
        )
        # ``is_dicom_file`` also checks for files without an extension.
        dicom_like = is_dicom_file(str(p))
        if (
            ext in ['.json', '.tsv', '.nii', '.nii.gz', '.html', '.htm']
            or dicom_like
            or gifti_candidate
            or freesurfer_candidate
        ):
            self.viewer.load_file(p)

    def updateStats(self):
        """Compute and display BIDS stats: total subjects, files, modalities."""
        root = self.bids_root
        if not root:
            return
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
                        imgs = len(list(ses.rglob(f'{m}/*.nii*')))
                        meta = len(list(ses.rglob(f'{m}/*.json'))) + len(list(ses.rglob(f'{m}/*.tsv')))
                        s2.addChild(QTreeWidgetItem([m, f"imgs:{imgs}, meta:{meta}"]))
                    si.addChild(s2)
            else:
                mods = set(p.parent.name for p in sub.rglob('*.nii*'))
                si.addChild(QTreeWidgetItem(["Sessions", "1"]))
                si.addChild(QTreeWidgetItem(["Modalities", str(len(mods))]))
                for m in mods:
                    imgs = len(list(sub.rglob(f'{m}/*.nii*')))
                    meta = len(list(sub.rglob(f'{m}/*.json'))) + len(list(sub.rglob(f'{m}/*.tsv')))
                    si.addChild(QTreeWidgetItem([m, f"imgs:{imgs}, meta:{meta}"]))
            self.stats.addTopLevelItem(si)
        self.stats.expandAll()

    def launchBatchRename(self):
        """Open the Batch Rename dialog from bids_editor_ancpbids."""
        if not self.bids_root:
            QMessageBox.critical(
                self,
                "Error",
                "Dataset not detected. Please load a dataset in File → Open BIDS",
            )
            return
        dlg = RemapDialog(self, self.bids_root)
        dlg.exec_()

    def launchIntendedForEditor(self):
        """Open the manual IntendedFor editor dialog."""
        if not self.bids_root:
            QMessageBox.critical(
                self,
                "Error",
                "Dataset not detected. Please load a dataset in File → Open BIDS",
            )
            return
        dlg = IntendedForDialog(self, self.bids_root)
        dlg.exec_()

    def refreshScansTsv(self):
        """Update ``*_scans.tsv`` files to match current filenames."""
        if not self.bids_root:
            QMessageBox.critical(
                self,
                "Error",
                "Dataset not detected. Please load a dataset in File → Open BIDS",
            )
            return
        try:
            from .post_conv_renamer import update_scans_tsv

            update_scans_tsv(self.bids_root)
            QMessageBox.information(self, "Refresh", "Updated scans.tsv files")
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to update: {exc}")

    def launchBidsIgnore(self):
        """Open dialog to edit ``.bidsignore``."""
        if not self.bids_root:
            QMessageBox.critical(
                self,
                "Error",
                "Dataset not detected. Please load a dataset in File → Open BIDS",
            )
            return
        dlg = BidsIgnoreDialog(self, self.bids_root)
        dlg.exec_()
