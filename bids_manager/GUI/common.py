"""Shared GUI utilities, helpers, and base widgets for BIDS Manager."""

import sys
import os
import json
import re
import shutil
import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation
try:  # NumPy exposes structured-array helpers from ``numpy.lib``
    from numpy.lib import recfunctions as rfn
except Exception:  # pragma: no cover - extremely old NumPy releases
    rfn = None
try:  # NumPy >= 1.20 provides dedicated exception classes
    from numpy import exceptions as np_exceptions  # type: ignore
except Exception:  # pragma: no cover - fallback for older versions
    np_exceptions = None
import threading
import time
import pydicom  # used to inspect DICOM headers when checking for mixed sessions
try:  # prefer absolute import but fall back to direct when running as a script
    from bids_manager.run_heudiconv_from_heuristic import is_dicom_file  # reuse existing helper
except Exception:  # pragma: no cover - packaging edge cases
    from run_heudiconv_from_heuristic import is_dicom_file  # type: ignore
try:
    import nibabel as nib
except ModuleNotFoundError as exc:
    if exc.name == '_bz2':
        import sys
        import types
        import io
        import subprocess

        class _SubprocessBZ2File(io.BufferedReader):
            """Minimal BZ2File replacement using the external ``bzip2`` binary."""

            def __init__(self, filename, mode="r", buffering=None, compresslevel=9):
                if "r" not in mode:
                    raise NotImplementedError(
                        "Writing not supported without Python bz2 module"
                    )
                proc = subprocess.Popen(
                    ["bzip2", "-dc", filename], stdout=subprocess.PIPE
                )
                if proc.stdout is None:  # pragma: no cover - should not happen
                    raise RuntimeError("Failed to open bzip2 subprocess")
                self._proc = proc
                super().__init__(proc.stdout)

            def close(self):
                try:
                    super().close()
                finally:
                    self._proc.stdout.close()
                    self._proc.wait()

        stub = types.ModuleType("bz2")
        stub.BZ2File = _SubprocessBZ2File
        sys.modules.setdefault("bz2", stub)
        import nibabel as nib
    else:  # pragma: no cover - unrelated import failure
        raise
from pathlib import Path
from collections import defaultdict
from functools import cmp_to_key
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from joblib import Parallel, delayed
from pandas.core.tools.datetimes import guess_datetime_format
from dataclasses import dataclass
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QTableWidgetSelectionRange, QGroupBox, QGridLayout,
    QTextEdit, QTreeView, QFileSystemModel, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QMessageBox, QAction, QSplitter, QDialog, QAbstractItemView,
    QMenuBar, QMenu, QSizePolicy, QComboBox, QSlider, QSpinBox,
    QCheckBox, QStyledItemDelegate, QDialogButtonBox, QListWidget, QScrollArea,
    QToolButton
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import (
    Qt,
    QModelIndex,
    QTimer,
    QProcess,
    QUrl,
    QRect,
    QPoint,
    QObject,
    QThread,
    QSize,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import (
    QPalette,
    QColor,
    QFont,
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QIcon,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import logging  # debug logging
import signal
import random
import string
import math
from bids_manager import dicom_inventory
from bids_manager.schema_renamer import (
    DEFAULT_SCHEMA_DIR,
    DERIVATIVES_PIPELINE_NAME,
    ENABLE_DWI_DERIVATIVES_MOVE,
    ENABLE_FIELDMap_NORMALIZATION,
    ENABLE_SCHEMA_RENAMER,
    SeriesInfo,
    apply_post_conversion_rename,
    build_preview_names,
    load_bids_schema,
)
from bids_manager.schema_renamer import normalize_study_name
try:
    import psutil
    HAS_PSUTIL = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PSUTIL = False

try:  # Surface reconstruction for the 3-D viewer (optional dependency)
    from skimage import measure as sk_measure
    HAS_SKIMAGE = True
except Exception:  # pragma: no cover - optional dependency
    sk_measure = None
    HAS_SKIMAGE = False

try:  # Hardware-accelerated 3-D rendering for volumes and surfaces
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import shaders as gl_shaders
    HAS_PYQTGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    pg = None
    gl = None
    gl_shaders = None
    HAS_PYQTGRAPH = False

# Paths to images bundled with the application
ROOT_DIR = Path(__file__).resolve().parent.parent
LOGO_FILE = ROOT_DIR / "miscellaneous" / "images" / "Logo.png"
ICON_FILE = ROOT_DIR / "miscellaneous" / "images" / "Icon.png"
ANCP_LAB_FILE = ROOT_DIR / "miscellaneous" / "images" / "ANCP_lab.png"
KAREL_IMG_FILE = ROOT_DIR / "miscellaneous" / "images" / "Karel.jpeg"
JOCHEM_IMG_FILE = ROOT_DIR / "miscellaneous" / "images" / "Jochem.jpg"

# Directory used to store persistent user preferences
PREF_DIR = ROOT_DIR / "user_preferences"


class _ConflictScannerWorker(QObject):
    """Run the conflict detection scan in a background thread."""

    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, root_dir: str, finder: Callable[[str, int], dict], n_jobs: int):
        super().__init__()
        self._root_dir = root_dir
        self._finder = finder
        # ``n_jobs`` mirrors the CPU limit used for the main scanning step so we
        # do not overwhelm the system when running both operations back-to-back.
        self._n_jobs = max(1, n_jobs)

    @pyqtSlot()
    def run(self) -> None:
        """Execute the slow directory walk outside the GUI thread."""

        try:
            conflicts = self._finder(self._root_dir, self._n_jobs)
        except Exception as exc:  # pragma: no cover - runtime safety
            # Forward the error message back to the GUI thread so the caller
            # can decide how to handle it without freezing the interface.
            self.failed.emit(str(exc))
        else:
            self.finished.emit(conflicts)


def _create_directional_light_shader():
    """Build a lightweight directional lighting shader for ``GLMeshItem``.

    PyQtGraph ships with a fixed light direction inside the built-in
    ``'shaded'`` program which makes the surface difficult to interpret when the
    user wants to change the perceived light source.  Creating our own shader
    allows us to expose the light direction and intensity as uniforms that can
    be updated on the fly without recompiling the OpenGL program each time the
    sliders move.
    """

    if not HAS_PYQTGRAPH or gl is None:
        return None

    shader_mod = getattr(gl, "shaders", None)
    if shader_mod is None:
        return None

    try:
        vertex = shader_mod.VertexShader(
            """
            varying vec3 normal;
            void main() {
                normal = normalize(gl_NormalMatrix * gl_Normal);
                gl_FrontColor = gl_Color;
                gl_BackColor = gl_Color;
                gl_Position = ftransform();
            }
            """
        )
        fragment = shader_mod.FragmentShader(
            """
            uniform float lightDir[3];
            uniform float lightParams[2];
            varying vec3 normal;
            void main() {
                vec3 norm = normalize(normal);
                vec3 lightVec = normalize(vec3(lightDir[0], lightDir[1], lightDir[2]));
                float diffuse = max(dot(norm, lightVec), 0.0) * lightParams[0];
                float ambient = lightParams[1];
                float lighting = clamp(ambient + diffuse, 0.0, 1.0);
                vec4 colour = gl_Color;
                colour.rgb *= lighting;
                gl_FragColor = colour;
            }
            """
        )
    except Exception:  # pragma: no cover - shader compilation errors are runtime only
        return None

    return shader_mod.ShaderProgram(
        None,
        [vertex, fragment],
        uniforms={
            "lightDir": [0.0, 0.0, 1.0],
            # ``lightParams`` stores [diffuse_scale, ambient_strength].  We keep
            # a modest ambient component so that the mesh never becomes entirely
            # black when the light points away from the surface.
            "lightParams": [1.0, 0.35],
        },
    )


def _create_flat_color_shader():
    """Return an unlit shader for meshes when lighting is disabled."""

    if not HAS_PYQTGRAPH or gl is None:
        return None

    shader_mod = getattr(gl, "shaders", None)
    if shader_mod is None:
        return None

    try:
        vertex = shader_mod.VertexShader(
            """
            void main() {
                gl_FrontColor = gl_Color;
                gl_BackColor = gl_Color;
                gl_Position = ftransform();
            }
            """
        )
        fragment = shader_mod.FragmentShader(
            """
            void main() {
                gl_FragColor = gl_Color;
            }
            """
        )
    except Exception:  # pragma: no cover - shader compilation errors occur at runtime
        return None

    return shader_mod.ShaderProgram(None, [vertex, fragment], uniforms={})


_SLICE_ORIENTATIONS = (
    ("sagittal", 0, "Left", "Right"),
    ("coronal", 1, "Posterior", "Anterior"),
    ("axial", 2, "Inferior", "Superior"),
)


if HAS_PYQTGRAPH:

    class _AdjustableAxisItem(gl.GLAxisItem):
        """Axis item with configurable line width for better visibility."""

        def __init__(self, *args, **kwargs):
            self._line_width = 2.0
            super().__init__(*args, **kwargs)

        def setLineWidth(self, width: float) -> None:
            self._line_width = max(1.0, float(width))
            self.update()

        def paint(self):  # pragma: no cover - requires OpenGL context
            from OpenGL.GL import (
                GL_LINES,
                GL_LINE_SMOOTH,
                GL_LINE_SMOOTH_HINT,
                GL_NICEST,
                glBegin,
                glColor4f,
                glEnable,
                glEnd,
                glHint,
                glLineWidth,
                glVertex3f,
            )

            self.setupGLState()
            if self.antialias:
                glEnable(GL_LINE_SMOOTH)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glLineWidth(self._line_width)
            glBegin(GL_LINES)
            x, y, z = self.size()
            glColor4f(0, 1, 0, 0.6)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, z)
            glColor4f(1, 1, 0, 0.6)
            glVertex3f(0, 0, 0)
            glVertex3f(0, y, 0)
            glColor4f(0, 0, 1, 0.6)
            glVertex3f(0, 0, 0)
            glVertex3f(x, 0, 0)
            glEnd()


@dataclass
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

def _extract_subject(row) -> str:
    """Return subject identifier prioritising ``BIDS_name`` and stripping ``sub-``."""
    subj = str(row.get("BIDS_name") or row.get("subject") or row.get("sub") or "UNK")
    if subj.lower().startswith("sub-"):
        subj = subj[4:]
    return subj

def _compute_bids_preview(df, schema):
    """Returns a dict {row_index: (datatype, basename)} for preview; safe if schema is None."""
    out = {}
    if not schema:
        return out
    rows = []
    idxs = []
    for i, row in df.iterrows():
        subject = _extract_subject(row)
        session = row.get("session") or row.get("ses") or None
        modality = str(row.get("modality") or row.get("fine_modality") or row.get("BIDS_modality") or "")
        sequence = str(row.get("sequence") or row.get("SeriesDescription") or "")
        rep = row.get("rep") or row.get("repeat") or 1

        extra = {}
        for key in ("task", "task_hits", "acq", "run", "dir", "echo"):
            if row.get(key):
                extra[key] = str(row.get(key))

        rows.append(SeriesInfo(subject, session, modality, sequence, int(rep or 1), extra))
        idxs.append(i)

    proposals = build_preview_names(rows, schema)
    for (series, dt, base), idx in zip(proposals, idxs):
        out[idx] = (dt, base)
    return out

# ---- basic logging config ----
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def _terminate_process_tree(pid: int):
    """Terminate a process and all of its children without killing the GUI."""
    # Protect against invalid PIDs which may occur if a process fails to start
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


def _style_axes_3d(axis, fg_color: str, axis_bg: str) -> None:
    """Apply consistent styling to Matplotlib 3-D axes."""

    axis.set_facecolor(axis_bg)
    base = axis.get_facecolor()
    axis.xaxis.set_pane_color((*base[:3], 0.15))
    axis.yaxis.set_pane_color((*base[:3], 0.15))
    axis.zaxis.set_pane_color((*base[:3], 0.15))
    for ax in (axis.xaxis, axis.yaxis, axis.zaxis):
        ax.label.set_color(fg_color)
        ax.set_tick_params(colors=fg_color)
    axis.grid(False)


GIFTI_SURFACE_SUFFIXES = (".surf.gii", ".surf.gii.gz", ".gii", ".gii.gz")
FREESURFER_SURFACE_SUFFIXES = (
    ".pial",
    ".pial.gz",
    ".white",
    ".white.gz",
    ".inflated",
    ".inflated.gz",
    ".sphere",
    ".sphere.gz",
    ".orig",
    ".orig.gz",
    ".smoothwm",
    ".smoothwm.gz",
    ".midthickness",
    ".midthickness.gz",
)


def _dedup_parts(*parts: str) -> str:
    """Return underscore-joined parts with consecutive repeats removed."""
    # ``parts`` may contain elements that themselves contain underscores.  The
    # goal is to produce a clean path component without duplicate separators.
    tokens: list[str] = []
    for part in parts:
        # Split each piece on underscores so ``seq__name`` becomes ["seq", "name"]
        for t in str(part).split('_'):
            # Only keep tokens that are not a repeat of the previous one
            if t and (not tokens or t != tokens[-1]):
                tokens.append(t)
    return "_".join(tokens)


def _safe_stem(text: str) -> str:
    """Return filename-friendly version of ``text`` used for study folders."""

    cleaned = normalize_study_name(text)
    return re.sub(r"[^0-9A-Za-z_-]+", "_", cleaned).strip("_")


def _format_subject_id(num: int) -> str:
    """Return ID as three letters followed by three digits."""
    letters_idx = (num - 1) // 1000
    digits = (num - 1) % 1000 + 1
    letters = []
    for _ in range(3):
        letters.append(chr(ord("A") + letters_idx % 26))
        letters_idx //= 26
    return "".join(reversed(letters)) + f"{digits:03d}"


def _random_subject_id(existing: set[str]) -> str:
    """Return a unique random 3-letter/3-digit identifier."""
    while True:
        letters = ''.join(random.choices(string.ascii_uppercase, k=3))
        digits = ''.join(random.choices(string.digits, k=3))
        sid = letters + digits
        if sid not in existing:
            return sid


def _next_numeric_id(used: set[str]) -> str:
    """Return the next "sub-XXX" style identifier."""
    nums = []
    for name in used:
        m = re.fullmatch(r"sub-(\d+)", name)
        if m:
            try:
                nums.append(int(m.group(1)))
            except Exception:
                pass
    nxt = max(nums, default=0) + 1
    while True:
        candidate = f"sub-{nxt:03d}"
        if candidate not in used:
            return candidate
        nxt += 1


class AutoFillTableWidget(QTableWidget):
    """``QTableWidget`` with an Excel-like autofill handle.

    The widget exposes a small square in the bottom-right corner of the current
    selection.  Dragging this handle extends the selection and automatically
    fills the new cells either by cloning the original content or by continuing
    simple sequences (numeric, datetime, or text with trailing digits).
    """

    HANDLE_SIZE = 8  # Square size in device pixels

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._handle_rect = QRect()
        self._autofill_active = False
        self._autofill_origin_range: Optional[QTableWidgetSelectionRange] = None
        self._autofill_current_range: Optional[QTableWidgetSelectionRange] = None
        # Whenever the selection changes we repaint so the handle follows it.
        self.itemSelectionChanged.connect(self._refresh_handle)

    # ------------------------------------------------------------------
    # Painting utilities
    def _index_is_editable(self, row: int, column: int) -> bool:
        """Return ``True`` when the ``(row, column)`` cell can be edited."""

        model = self.model()
        index = model.index(row, column)
        if not index.isValid():
            return False
        flags = model.flags(index)
        return bool(flags & Qt.ItemIsEditable)

    def _range_is_editable(
        self, rng: Optional[QTableWidgetSelectionRange]
    ) -> bool:
        """Return ``True`` when *all* cells in ``rng`` expose the edit flag."""

        if rng is None:
            return False
        for row in range(rng.topRow(), rng.bottomRow() + 1):
            for col in range(rng.leftColumn(), rng.rightColumn() + 1):
                if not self._index_is_editable(row, col):
                    return False
        return True

    def _clamp_to_editable(
        self,
        origin: QTableWidgetSelectionRange,
        candidate: QTableWidgetSelectionRange,
    ) -> QTableWidgetSelectionRange:
        """Restrict ``candidate`` so autofill never crosses non-editable cells."""

        # Limit horizontal growth by checking each additional column in order.
        max_right = origin.rightColumn()
        if candidate.rightColumn() > max_right:
            for col in range(origin.rightColumn() + 1, candidate.rightColumn() + 1):
                if all(
                    self._index_is_editable(row, col)
                    for row in range(origin.topRow(), origin.bottomRow() + 1)
                ):
                    max_right = col
                else:
                    break

        # Limit vertical growth by checking each extra row with the approved columns.
        max_bottom = origin.bottomRow()
        if candidate.bottomRow() > max_bottom:
            for row in range(origin.bottomRow() + 1, candidate.bottomRow() + 1):
                if all(
                    self._index_is_editable(row, col)
                    for col in range(origin.leftColumn(), max_right + 1)
                ):
                    max_bottom = row
                else:
                    break

        if (
            max_right == origin.rightColumn()
            and max_bottom == origin.bottomRow()
        ):
            return origin

        return QTableWidgetSelectionRange(
            origin.topRow(),
            origin.leftColumn(),
            max_bottom,
            max_right,
        )
    def _refresh_handle(self) -> None:
        """Trigger a repaint so the autofill handle reflects the selection."""

        self._handle_rect = QRect()
        self.viewport().update()

    def _current_selection_range(self) -> Optional[QTableWidgetSelectionRange]:
        """Return the single active selection range (if any)."""

        ranges = self.selectedRanges()
        if len(ranges) != 1:
            return None
        return ranges[0]

    def paintEvent(self, event):  # noqa: D401 - Qt override
        super().paintEvent(event)

        rng = self._current_selection_range()
        if not self._range_is_editable(rng):
            self._handle_rect = QRect()
            return

        model_index = self.model().index(rng.bottomRow(), rng.rightColumn())
        rect = self.visualRect(model_index)
        if not rect.isValid():
            self._handle_rect = QRect()
            return

        size = self.HANDLE_SIZE
        self._handle_rect = QRect(
            rect.right() - size + 1,
            rect.bottom() - size + 1,
            size,
            size,
        )

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self._handle_rect, self.palette().highlight())
        border = QPen(self.palette().color(QPalette.Dark))
        painter.setPen(border)
        painter.drawRect(self._handle_rect)
        painter.end()

    # ------------------------------------------------------------------
    # Mouse interaction handling
    def mousePressEvent(self, event):  # noqa: D401 - Qt override
        if (
            event.button() == Qt.LeftButton
            and self._handle_rect.contains(event.pos())
            and self._current_selection_range() is not None
        ):
            self._autofill_active = True
            self._autofill_origin_range = self._current_selection_range()
            self._autofill_current_range = self._autofill_origin_range
            self.viewport().setCursor(Qt.SizeAllCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: D401 - Qt override
        if self._autofill_active:
            new_range = self._compute_drag_range(event.pos())
            if new_range is not None:
                self._autofill_current_range = new_range
                # Replace the selection with the preview range so the user sees
                # the future extent of the autofill before releasing the mouse.
                self.blockSignals(True)
                self.clearSelection()
                self.setRangeSelected(new_range, True)
                self.blockSignals(False)
                self.viewport().update()
            event.accept()
            return

        if self._handle_rect.contains(event.pos()):
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().unsetCursor()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: D401 - Qt override
        if self._autofill_active and event.button() == Qt.LeftButton:
            try:
                self._finish_autofill()
            finally:
                self._autofill_active = False
                self.viewport().unsetCursor()
                self._autofill_origin_range = None
                self._autofill_current_range = None
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _compute_drag_range(self, pos: QPoint) -> Optional[QTableWidgetSelectionRange]:
        """Return the preview range while dragging the autofill handle."""

        if self._autofill_origin_range is None:
            return None

        origin = self._autofill_origin_range
        row = self.rowAt(pos.y())
        col = self.columnAt(pos.x())
        if row < 0 or col < 0:
            return origin

        # Only allow extending outward from the original bottom/right edge.
        if row < origin.bottomRow():
            row = origin.bottomRow()
        if col < origin.rightColumn():
            col = origin.rightColumn()

        if row == origin.bottomRow() and col == origin.rightColumn():
            return origin

        candidate = QTableWidgetSelectionRange(
            origin.topRow(),
            origin.leftColumn(),
            row,
            col,
        )

        return self._clamp_to_editable(origin, candidate)

    # ------------------------------------------------------------------
    # Autofill logic
    def _finish_autofill(self) -> None:
        """Apply the autofill operation once the user releases the mouse."""

        if self._autofill_origin_range is None:
            return

        final_range = self._autofill_current_range or self._autofill_origin_range
        origin = self._autofill_origin_range

        if (
            final_range.bottomRow() == origin.bottomRow()
            and final_range.rightColumn() == origin.rightColumn()
        ):
            # Nothing changed; restore the original selection highlight.
            self.blockSignals(True)
            self.clearSelection()
            self.setRangeSelected(origin, True)
            self.blockSignals(False)
            self.viewport().update()
            return

        self._apply_autofill(origin, final_range)

        # Keep the extended range selected after the fill to match spreadsheet UX.
        self.blockSignals(True)
        self.clearSelection()
        self.setRangeSelected(final_range, True)
        self.blockSignals(False)
        self.viewport().update()

    def _apply_autofill(
        self,
        origin: QTableWidgetSelectionRange,
        target: QTableWidgetSelectionRange,
    ) -> None:
        """Populate ``target`` based on ``origin`` and the autofill heuristics."""

        extend_down = target.bottomRow() > origin.bottomRow()
        extend_right = target.rightColumn() > origin.rightColumn()

        if extend_right:
            self._fill_right(origin, target.rightColumn())

        if extend_down:
            base_right = target.rightColumn() if extend_right else origin.rightColumn()
            base_range = QTableWidgetSelectionRange(
                origin.topRow(),
                origin.leftColumn(),
                origin.bottomRow(),
                base_right,
            )
            self._fill_down(base_range, target.bottomRow())

    def _fill_right(self, base: QTableWidgetSelectionRange, target_right: int) -> None:
        """Extend ``base`` horizontally until ``target_right`` inclusive."""

        extra = target_right - base.rightColumn()
        if extra <= 0:
            return

        for row in range(base.topRow(), base.bottomRow() + 1):
            values = [
                self._get_item_text(row, col)
                for col in range(base.leftColumn(), base.rightColumn() + 1)
            ]
            new_values = self._extend_series(values, extra)
            for offset, value in enumerate(new_values, start=1):
                self._set_item_text(row, base.rightColumn() + offset, value)

    def _fill_down(self, base: QTableWidgetSelectionRange, target_bottom: int) -> None:
        """Extend ``base`` vertically until ``target_bottom`` inclusive."""

        extra = target_bottom - base.bottomRow()
        if extra <= 0:
            return

        for col in range(base.leftColumn(), base.rightColumn() + 1):
            values = [
                self._get_item_text(row, col)
                for row in range(base.topRow(), base.bottomRow() + 1)
            ]
            new_values = self._extend_series(values, extra)
            for offset, value in enumerate(new_values, start=1):
                self._set_item_text(base.bottomRow() + offset, col, value)

    # ------------------------------------------------------------------
    # Sequence helpers
    def _extend_series(self, values: list[str], steps: int) -> list[str]:
        """Return ``steps`` new values continuing ``values`` when possible."""

        if steps <= 0 or not values:
            return []

        numeric = self._extend_numeric_series(values, steps)
        if numeric is not None:
            return numeric

        datelike = self._extend_datetime_series(values, steps)
        if datelike is not None:
            return datelike

        patterned = self._extend_text_pattern_series(values, steps)
        if patterned is not None:
            return patterned

        # Fallback: repeat the original pattern cyclically.
        repeated = []
        for i in range(steps):
            repeated.append(values[i % len(values)])
        return repeated

    def _extend_numeric_series(
        self,
        values: list[str],
        steps: int,
    ) -> Optional[list[str]]:
        """Continue integer/decimal sequences when the pattern is consistent."""

        stripped = [v.strip() for v in values]
        if any(not s for s in stripped):
            return None

        # Try integer sequences first so "01", "02" keep their padding.
        if all(re.fullmatch(r"[+-]?\d+", s) for s in stripped):
            numbers = [int(s) for s in stripped]
            diff = 0
            if len(numbers) >= 2:
                diffs = [numbers[i] - numbers[i - 1] for i in range(1, len(numbers))]
                if all(d == diffs[0] for d in diffs):
                    diff = diffs[0]
            pad_width = len(stripped[-1].lstrip("+-"))
            has_leading_zero = stripped[-1].lstrip("+-").startswith("0") and pad_width > 1
            force_plus = stripped[-1].startswith("+")
            current = numbers[-1]
            generated: list[str] = []
            for _ in range(steps):
                current += diff
                text = str(current)
                if has_leading_zero and current >= 0:
                    text = f"{current:0{pad_width}d}"
                if force_plus and not text.startswith("-") and not text.startswith("+"):
                    text = "+" + text
                generated.append(text)
            return generated

        # Fall back to decimals when integers are not appropriate.
        decimals: list[Decimal] = []
        decimal_places = 0
        for s in stripped:
            try:
                dec = Decimal(s)
            except InvalidOperation:
                return None
            decimals.append(dec)
            if "." in s:
                decimal_places = max(decimal_places, len(s.split(".")[-1]))

        diff = Decimal(0)
        if len(decimals) >= 2:
            diffs = [decimals[i] - decimals[i - 1] for i in range(1, len(decimals))]
            if all(d == diffs[0] for d in diffs):
                diff = diffs[0]
        current = decimals[-1]
        generated = []
        for _ in range(steps):
            current += diff
            if decimal_places:
                generated.append(f"{current:.{decimal_places}f}")
            else:
                generated.append(str(current))
        return generated

    def _extend_datetime_series(
        self,
        values: list[str],
        steps: int,
    ) -> Optional[list[str]]:
        """Continue datetime-like strings when intervals are consistent."""

        try:
            parsed = pd.to_datetime(values, errors="raise", infer_datetime_format=True)
        except Exception:
            return None

        if parsed.isna().any():
            return None

        delta = pd.Timedelta(0)
        if len(parsed) >= 2:
            diffs = parsed.diff().iloc[1:]
            if not diffs.empty and all(d == diffs.iloc[0] for d in diffs):
                delta = diffs.iloc[0]

        last = parsed.iloc[-1]
        template = values[-1]
        fmt = guess_datetime_format(template)
        generated = []
        current = last
        for _ in range(steps):
            current = current + delta
            if fmt:
                generated.append(current.strftime(fmt))
            else:
                if "T" in template:
                    generated.append(current.isoformat())
                elif ":" in template:
                    generated.append(current.strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    generated.append(current.date().isoformat())
        return generated

    def _extend_text_pattern_series(
        self,
        values: list[str],
        steps: int,
    ) -> Optional[list[str]]:
        """Continue strings ending with digits (e.g. "scan01")."""

        matches = [re.match(r"^(.*?)(\d+)$", v.strip()) for v in values]
        if any(m is None for m in matches):
            return None

        prefixes = [m.group(1) for m in matches if m is not None]
        numbers = [int(m.group(2)) for m in matches if m is not None]
        if not prefixes or not numbers:
            return None
        if any(p != prefixes[0] for p in prefixes):
            return None

        diff = 0
        if len(numbers) >= 2:
            diffs = [numbers[i] - numbers[i - 1] for i in range(1, len(numbers))]
            if all(d == diffs[0] for d in diffs):
                diff = diffs[0]
        pad_width = len(matches[-1].group(2)) if matches[-1] is not None else 0
        current = numbers[-1]
        prefix = prefixes[-1]
        generated = []
        for _ in range(steps):
            current += diff
            text = f"{current:0{pad_width}d}" if pad_width else str(current)
            generated.append(f"{prefix}{text}")
        return generated

    # ------------------------------------------------------------------
    # Cell helpers
    def _get_item_text(self, row: int, column: int) -> str:
        """Return the text stored at ``(row, column)`` (empty if missing)."""

        item = self.item(row, column)
        return item.text() if item is not None else ""

    def _set_item_text(self, row: int, column: int, value: str) -> None:
        """Assign ``value`` to ``(row, column)``, creating an item if required."""

        item = self.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.setItem(row, column, item)
        item.setText(value)


class MappingSortDialog(QDialog):
    """Dialog used to configure multi-level sorting for the metadata table."""

    def __init__(self, columns: list[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sort scanned metadata")
        self._columns = columns
        self._level_rows: list[tuple[QComboBox, QComboBox, QWidget]] = []

        layout = QVBoxLayout(self)
        info = QLabel(
            "Select the columns to sort by in priority order. "
            "Each level is applied sequentially, just like Excel's multi-column sort."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._levels_layout = QVBoxLayout()
        layout.addLayout(self._levels_layout)

        controls_layout = QHBoxLayout()
        self._add_level_btn = QPushButton("Add level")
        self._add_level_btn.clicked.connect(self._add_level)
        self._remove_level_btn = QPushButton("Remove level")
        self._remove_level_btn.clicked.connect(self._remove_level)
        controls_layout.addWidget(self._add_level_btn)
        controls_layout.addWidget(self._remove_level_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Always start with a single sorting level configured.
        self._add_level()
        self._update_button_state()

    # ------------------------------------------------------------------
    def _add_level(self) -> None:
        """Append a new sorting level to the dialog."""

        if len(self._level_rows) >= len(self._columns):
            return

        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        column_combo = QComboBox(row_widget)
        column_combo.addItems(self._columns)
        # Try to pre-select the first column that is not already used.
        used = {combo.currentText() for combo, _order, _widget in self._level_rows}
        for idx, name in enumerate(self._columns):
            if name not in used:
                column_combo.setCurrentIndex(idx)
                break

        order_combo = QComboBox(row_widget)
        order_combo.addItems(["Ascending", "Descending"])

        row_layout.addWidget(QLabel(f"Level {len(self._level_rows) + 1}", row_widget))
        row_layout.addWidget(column_combo)
        row_layout.addWidget(order_combo)
        row_layout.addStretch()

        self._levels_layout.addWidget(row_widget)
        self._level_rows.append((column_combo, order_combo, row_widget))
        self._update_button_state()

    def _remove_level(self) -> None:
        """Remove the last configured sorting level."""

        if not self._level_rows:
            return
        combo, order_combo, widget = self._level_rows.pop()
        combo.deleteLater()
        order_combo.deleteLater()
        widget.deleteLater()
        self._update_button_state()

    def _update_button_state(self) -> None:
        """Enable/disable controls based on the current dialog state."""

        self._add_level_btn.setEnabled(len(self._level_rows) < len(self._columns))
        self._remove_level_btn.setEnabled(len(self._level_rows) > 1)

    def sort_instructions(self) -> list[tuple[str, bool]]:
        """Return the configured sorting hierarchy as ``(column, ascending)``."""

        instructions: list[tuple[str, bool]] = []
        seen: set[str] = set()
        for column_combo, order_combo, _widget in self._level_rows:
            column = column_combo.currentText()
            if not column or column in seen:
                continue
            ascending = order_combo.currentText() == "Ascending"
            instructions.append((column, ascending))
            seen.add(column)
        return instructions


class SubjectDelegate(QStyledItemDelegate):
    """Delegate to edit BIDS subject IDs without altering the 'sub-' prefix."""

    def createEditor(self, parent, option, index):  # noqa: D401 - Qt override
        return QLineEdit(parent)

    def setEditorData(self, editor, index):  # noqa: D401 - Qt override
        text = index.model().data(index, Qt.EditRole)
        suffix = text[4:] if text.startswith("sub-") else text
        editor.setText(suffix)
        editor.selectAll()

    def setModelData(self, editor, model, index):  # noqa: D401 - Qt override
        model.setData(index, "sub-" + editor.text(), Qt.EditRole)


class ShrinkableScrollArea(QScrollArea):
    """``QScrollArea`` variant that allows the parent splitter to shrink."""

    def minimumSizeHint(self) -> QSize:  # noqa: D401 - Qt override
        return QSize(0, 0)

    def sizeHint(self) -> QSize:  # noqa: D401 - Qt override
        hint = super().sizeHint()
        return QSize(max(0, hint.width()), max(0, hint.height()))
