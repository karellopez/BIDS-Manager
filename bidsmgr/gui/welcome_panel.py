"""Welcome panel - the project-first landing (VS Code-style home tab).

Shown when no project is open. A hero header plus three self-contained section
cards: Create a new dataset, Open an existing one, and Recent projects. Emits
:pyattr:`project_opened` with the opened ``Project`` and its dataset root so
:class:`MainWindow` can bind it to the Converter.

The create/open *logic* lives in small testable methods (``create_project`` /
``open_project``); the controls gather input then call those, so the flow can be
exercised offscreen without driving modal dialogs.

Styling is fully QSS-driven (object names keyed in ``theme.qss``);
``repaint_for_palette`` runs the unpolish/polish dance so a dark<->light swap
recomputes every card's colours (no stale white/black patches).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import shutil

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..cli._scaffold import slugify_name
from ..cli.create import open_or_create_workspace
from .app_settings import AppSettings

log = logging.getLogger(__name__)


def _section_card(title: str, description: str) -> tuple[QFrame, QVBoxLayout]:
    """Build a titled, self-contained section card; return (frame, body layout)."""
    card = QFrame()
    card.setObjectName("welcome-card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(22, 18, 22, 18)
    lay.setSpacing(10)
    head = QLabel(title)
    head.setObjectName("welcome-section")
    desc = QLabel(description)
    desc.setObjectName("welcome-section-desc")
    desc.setWordWrap(True)
    lay.addWidget(head)
    lay.addWidget(desc)
    return card, lay


class WelcomePanel(QWidget):
    """Create / open / recent for BIDS dataset projects.

    Emits ``project_opened(project, bids_root)`` (a
    :class:`~bidsmgr.project.Project` and a :class:`pathlib.Path`).
    """

    project_opened = pyqtSignal(object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("welcome-panel")

        # Default the create-location to the last BIDS-output parent (or home).
        last_parent = AppSettings.load().bids_parent
        self._create_location = Path(last_parent) if last_parent else Path.home()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("welcome-scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        body = QWidget()
        body.setObjectName("welcome-body")
        scroll.setWidget(body)

        # Centred, capped-width container that uses the horizontal space: a
        # full-width hero on top, then a two-column row (Create on the left,
        # Open + Recent stacked on the right) so the page reads as a desktop
        # layout rather than a tall single phone-width column.
        body_row = QHBoxLayout(body)
        body_row.setContentsMargins(28, 28, 28, 28)
        body_row.addStretch(1)
        col_host = QWidget()
        col_host.setObjectName("welcome-col")
        col_host.setMaximumWidth(1080)
        host_v = QVBoxLayout(col_host)
        host_v.setContentsMargins(0, 0, 0, 0)
        host_v.setSpacing(18)
        body_row.addWidget(col_host, 6)
        body_row.addStretch(1)

        host_v.addWidget(self._build_hero())

        content = QHBoxLayout()
        content.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(16)
        left.addWidget(self._build_create_card())
        left.addStretch(1)
        right = QVBoxLayout()
        right.setSpacing(16)
        right.addWidget(self._build_open_card())
        right.addWidget(self._build_recent_card(), 1)
        content.addLayout(left, 1)
        content.addLayout(right, 1)
        host_v.addLayout(content, 1)

        self.refresh_recent()

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _build_hero(self) -> QWidget:
        hero = QWidget()
        hero.setObjectName("welcome-hero")
        h = QHBoxLayout(hero)
        h.setContentsMargins(2, 0, 2, 4)
        h.setSpacing(14)

        icon = QLabel()
        icon.setObjectName("welcome-logo")
        png = Path(__file__).parent / "assets" / "macos" / "AppIcon128.png"
        if png.exists():
            pix = QPixmap(str(png))
            if not pix.isNull():
                icon.setPixmap(pix.scaledToHeight(
                    56, Qt.TransformationMode.SmoothTransformation,
                ))
                h.addWidget(icon, 0, Qt.AlignmentFlag.AlignVCenter)

        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel("Welcome to BIDS-Manager")
        title.setObjectName("welcome-title")
        subtitle = QLabel(
            "Schema-driven BIDS conversion, curation, and editing. "
            "Create a dataset to start, or reopen one to pick up where you left off."
        )
        subtitle.setObjectName("welcome-subtitle")
        subtitle.setWordWrap(True)
        text.addWidget(title)
        text.addWidget(subtitle)
        h.addLayout(text, 1)
        return hero

    def _build_create_card(self) -> QFrame:
        card, lay = _section_card(
            "Create a new dataset",
            "Scaffold a fresh BIDS dataset (dataset_description.json, README, "
            ".bidsignore) and start scanning raw data into it.",
        )

        self._name_edit = QLineEdit()
        self._name_edit.setObjectName("welcome-input")
        self._name_edit.setPlaceholderText("Dataset name, e.g. My Study")
        self._name_edit.returnPressed.connect(self._on_inline_create)
        lay.addWidget(self._name_edit)

        loc_row = QHBoxLayout()
        loc_row.setSpacing(8)
        self._location_edit = QLineEdit(str(self._create_location))
        self._location_edit.setObjectName("welcome-input")
        self._location_edit.setReadOnly(True)
        browse = QPushButton("Browse…")
        browse.setObjectName("tb-btn-ghost")
        browse.setCursor(Qt.CursorShape.PointingHandCursor)
        browse.clicked.connect(self._on_browse_location)
        loc_row.addWidget(self._location_edit, 1)
        loc_row.addWidget(browse)
        lay.addLayout(loc_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._create_btn = QPushButton("Create dataset")
        self._create_btn.setObjectName("tb-btn-primary")
        self._create_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._create_btn.clicked.connect(self._on_inline_create)
        btn_row.addWidget(self._create_btn)
        lay.addLayout(btn_row)
        return card

    def _build_open_card(self) -> QFrame:
        card, lay = _section_card(
            "Open an existing dataset",
            "Continue curating a BIDS-Manager project, or adopt a dataset "
            "created elsewhere (it is opened read-only, never overwritten).",
        )
        btn_row = QHBoxLayout()
        self._open_btn = QPushButton("Open dataset folder…")
        self._open_btn.setObjectName("tb-btn-ghost")
        self._open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._open_btn.clicked.connect(self._on_open_clicked)
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)
        return card

    def _build_recent_card(self) -> QFrame:
        card, lay = _section_card(
            "Recent projects",
            "Datasets you recently created or opened. Double-click to reopen.",
        )
        self._recent = QListWidget()
        self._recent.setObjectName("welcome-recent")
        self._recent.setMinimumHeight(160)
        self._recent.itemActivated.connect(self._on_recent_activated)
        self._recent.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._recent.customContextMenuRequested.connect(self._on_recent_menu)
        lay.addWidget(self._recent, 1)
        hint = QLabel("Right-click a project to remove or delete it.")
        hint.setObjectName("welcome-section-desc")
        lay.addWidget(hint)
        self._recent_empty = QLabel("No recent projects yet.")
        self._recent_empty.setObjectName("welcome-section-desc")
        lay.addWidget(self._recent_empty)
        return card

    # ------------------------------------------------------------------
    # Recent list
    # ------------------------------------------------------------------

    def refresh_recent(self) -> None:
        """Repopulate the recent-projects list from AppSettings."""
        self._recent.clear()
        recents = AppSettings.load().recent_projects
        for p in recents:
            item = QListWidgetItem(p)
            item.setData(Qt.ItemDataRole.UserRole, p)
            if not Path(p).exists():
                item.setText(f"{p}   (missing)")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self._recent.addItem(item)
        has = bool(recents)
        self._recent.setVisible(has)
        self._recent_empty.setVisible(not has)

    # ------------------------------------------------------------------
    # Core (testable) actions
    # ------------------------------------------------------------------

    def create_project(self, parent_dir: Path, name: str) -> Optional[Path]:
        """Create ``<parent_dir>/<slug(name)>`` and open it. Returns the root."""
        name = (name or "").strip()
        if not name:
            return None
        bids_root = Path(parent_dir) / slugify_name(name)
        proj = open_or_create_workspace(bids_root, name=name)
        AppSettings.remember_recent_project(bids_root)
        self.refresh_recent()
        self.project_opened.emit(proj, bids_root)
        return bids_root

    def open_project(self, bids_root: Path) -> Optional[Path]:
        """Open (or adopt) the dataset at ``bids_root``. Returns the root."""
        bids_root = Path(bids_root)
        proj = open_or_create_workspace(bids_root)
        AppSettings.remember_recent_project(bids_root)
        self.refresh_recent()
        self.project_opened.emit(proj, bids_root)
        return bids_root

    # ------------------------------------------------------------------
    # Control handlers
    # ------------------------------------------------------------------

    def _on_browse_location(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Choose where to create the dataset",
            str(self._create_location),
        )
        if d:
            self._create_location = Path(d)
            self._location_edit.setText(d)

    def _on_inline_create(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            self._name_edit.setFocus()
            return
        parent = self._location_edit.text().strip() or str(Path.home())
        self.create_project(Path(parent), name)
        self._name_edit.clear()

    def _on_open_clicked(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Open BIDS dataset folder", str(self._create_location),
        )
        if not d:
            return
        self.open_project(Path(d))

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        p = item.data(Qt.ItemDataRole.UserRole)
        if p and Path(p).exists():
            self.open_project(Path(p))

    # ------------------------------------------------------------------
    # Remove / delete projects
    # ------------------------------------------------------------------

    def delete_project(self, bids_root: Path, *, from_disk: bool) -> bool:
        """Forget a project (and optionally delete its folder from disk).

        With ``from_disk=False`` the dataset is only dropped from the recent
        list. With ``from_disk=True`` the folder is permanently removed, but only
        when it actually looks like a BIDS dataset / BM project (has
        ``.bidsmgr/`` or ``dataset_description.json``) so an arbitrary path can
        never be nuked. Returns ``True`` if anything was removed.
        """
        bids_root = Path(bids_root)
        if from_disk and bids_root.exists():
            looks_bids = (
                (bids_root / ".bidsmgr").exists()
                or (bids_root / "dataset_description.json").exists()
            )
            if not looks_bids:
                return False  # refuse to delete a non-dataset folder
            shutil.rmtree(bids_root, ignore_errors=True)
        AppSettings.forget_recent_project(bids_root)
        self.refresh_recent()
        return True

    def _on_recent_menu(self, pos) -> None:
        item = self._recent.itemAt(pos)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        bids_root = Path(path)
        menu = QMenu(self)
        act_open = menu.addAction("Open")
        act_open.setEnabled(bids_root.exists())
        act_forget = menu.addAction("Remove from list")
        act_delete = menu.addAction("Delete project from disk…")
        chosen = menu.exec(self._recent.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_open:
            if bids_root.exists():
                self.open_project(bids_root)
        elif chosen is act_forget:
            self.delete_project(bids_root, from_disk=False)
        elif chosen is act_delete:
            confirm = QMessageBox.warning(
                self,
                "Delete project",
                f"Permanently delete this dataset and everything in it?\n\n{bids_root}\n\n"
                "This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                if not self.delete_project(bids_root, from_disk=True):
                    QMessageBox.information(
                        self, "Delete project",
                        "Not deleted: that folder does not look like a BIDS "
                        "dataset, so it was left untouched (removed from the "
                        "list only).",
                    )

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def repaint_for_palette(self, pal: dict) -> None:
        """Force QSS recomputation on a dark<->light swap.

        The same unpolish/polish dance the other panels use, so the cards /
        inputs / recent list pick up the re-applied stylesheet instead of
        keeping stale background colours.
        """
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()


__all__ = ["WelcomePanel"]
