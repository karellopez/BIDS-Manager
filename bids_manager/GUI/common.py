"""Shared utilities, imports, and helpers for the BIDS Manager GUI.

This module centralizes the heavy imports, optional dependency handling, and
pure helper functions that are reused by both the converter and editor
submodules. Keeping them here avoids duplication while preserving the original
behaviour of the monolithic ``gui.py`` implementation.
"""

import json
import math
import os
import random
import re
import shutil
import signal
import string
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.core.tools.datetimes import guess_datetime_format

try:  # NumPy exposes structured-array helpers from ``numpy.lib``
    from numpy.lib import recfunctions as rfn
except Exception:  # pragma: no cover - extremely old NumPy releases
    rfn = None
try:  # NumPy >= 1.20 provides dedicated exception classes
    from numpy import exceptions as np_exceptions  # type: ignore
except Exception:  # pragma: no cover - fallback for older versions
    np_exceptions = None

import pydicom  # used to inspect DICOM headers when checking for mixed sessions
try:  # prefer relative import but fall back to direct when running as a script
    from bids_manager.run_heudiconv_from_heuristic import is_dicom_file
except Exception:  # pragma: no cover - packaging edge cases
    from run_heudiconv_from_heuristic import is_dicom_file  # type: ignore
try:
    import nibabel as nib
except ModuleNotFoundError as exc:
    if exc.name == '_bz2':
        import io
        import subprocess
        import types

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

from PyQt5.QtCore import (
    QModelIndex,
    QObject,
    QPoint,
    QRect,
    QThread,
    QTimer,
    QUrl,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPalette, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTableWidgetSelectionRange,
    QTextEdit,
    QToolButton,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QFileSystemModel,
    QAbstractItemView,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView

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
    normalize_study_name,
)

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

# Paths to images bundled with the application. Use the package root rather
# than the GUI folder so existing assets continue to load after the refactor.
_PKG_ROOT = Path(__file__).resolve().parent.parent
LOGO_FILE = _PKG_ROOT / "miscellaneous" / "images" / "Logo.png"
ICON_FILE = _PKG_ROOT / "miscellaneous" / "images" / "Icon.png"
ANCP_LAB_FILE = _PKG_ROOT / "miscellaneous" / "images" / "ANCP_lab.png"
KAREL_IMG_FILE = _PKG_ROOT / "miscellaneous" / "images" / "Karel.jpeg"
JOCHEM_IMG_FILE = _PKG_ROOT / "miscellaneous" / "images" / "Jochem.jpg"

# Directory used to store persistent user preferences
PREF_DIR = _PKG_ROOT / "user_preferences"


def _create_directional_light_shader():
    """Build a lightweight directional lighting shader for ``GLMeshItem``."""

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


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _terminate_process_tree(pid: int):
    """Terminate a process and all of its children without killing the GUI."""

    if pid <= 0:
        return
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

    tokens: list[str] = []
    for part in parts:
        for t in str(part).split('_'):
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


__all__ = [
    "ANCP_LAB_FILE",
    "DEFAULT_SCHEMA_DIR",
    "DERIVATIVES_PIPELINE_NAME",
    "ENABLE_DWI_DERIVATIVES_MOVE",
    "ENABLE_FIELDMap_NORMALIZATION",
    "ENABLE_SCHEMA_RENAMER",
    "FREESURFER_SURFACE_SUFFIXES",
    "GIFTI_SURFACE_SUFFIXES",
    "ICON_FILE",
    "JOCHEM_IMG_FILE",
    "KAREL_IMG_FILE",
    "LOGO_FILE",
    "PREF_DIR",
    "Any",
    "Callable",
    "Dict",
    "List",
    "Optional",
    "Sequence",
    "Tuple",
    "QAction",
    "QAbstractItemView",
    "QApplication",
    "QCheckBox",
    "QComboBox",
    "QDialog",
    "QDialogButtonBox",
    "QGridLayout",
    "QGroupBox",
    "QHeaderView",
    "QHBoxLayout",
    "QLabel",
    "QLineEdit",
    "QListWidget",
    "QMainWindow",
    "QMenu",
    "QMenuBar",
    "QMessageBox",
    "QPalette",
    "QPixmap",
    "QScrollArea",
    "QSizePolicy",
    "QSize",
    "QSlider",
    "QSpinBox",
    "QSplitter",
    "QStyledItemDelegate",
    "QTabWidget",
    "QTableWidget",
    "QTableWidgetItem",
    "QTableWidgetSelectionRange",
    "QTextEdit",
    "QToolButton",
    "QTreeView",
    "QTreeWidget",
    "QTreeWidgetItem",
    "QVBoxLayout",
    "QWidget",
    "QFileDialog",
    "QFileSystemModel",
    "QAbstractItemView",
    "QTimer",
    "QProcess",
    "QUrl",
    "QRect",
    "QPoint",
    "QThread",
    "QIcon",
    "QImage",
    "QPainter",
    "QPen",
    "Qt",
    "QModelIndex",
    "QWebEngineView",
    "SeriesInfo",
    "_SLICE_ORIENTATIONS",
    "_compute_bids_preview",
    "_create_directional_light_shader",
    "_create_flat_color_shader",
    "_dedup_parts",
    "_format_subject_id",
    "_get_ext",
    "_next_numeric_id",
    "_random_subject_id",
    "_safe_stem",
    "_style_axes_3d",
    "apply_post_conversion_rename",
    "build_preview_names",
    "cmp_to_key",
    "defaultdict",
    "delayed",
    "dicom_inventory",
    "gl",
    "gl_shaders",
    "guess_datetime_format",
    "is_dicom_file",
    "load_bids_schema",
    "math",
    "nib",
    "np",
    "np_exceptions",
    "os",
    "pd",
    "pg",
    "pydicom",
    "random",
    "rfn",
    "shutil",
    "signal",
    "sk_measure",
    "time",
    "threading",
    "Path",
    "Parallel",
    "HAS_PYQTGRAPH",
    "HAS_PSUTIL",
    "HAS_SKIMAGE",
]
