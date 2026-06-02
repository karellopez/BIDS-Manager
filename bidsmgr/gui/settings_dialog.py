"""Settings dialog — surface CLI knobs the GUI uses.

Reads / writes :class:`bidsmgr.gui.app_settings.AppSettings` via
``QSettings``. All changes are applied on **Save** (no live binding) so
the user can experiment with values and cancel without commit.

Tabs: Display / System / Scan / Convert + post-convert. The Convert tab
lays the post-convert chain out as an indented hierarchy (parent step +
its sub-options), and a "Restore defaults" button resets every widget to
the :class:`AppSettings` field defaults.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..util.system_info import SystemInfo, get_system_info
from .app_settings import AppSettings


def _indented(child: QWidget, *, indent: int = 22) -> QWidget:
    """Wrap ``child`` in a left-indented container so it reads as a
    sub-option nested under the checkbox above it (the post-convert
    hierarchy tree)."""
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(indent, 0, 0, 0)
    lay.setSpacing(6)
    lay.addWidget(child)
    lay.addStretch(1)
    return box


def _bind_children(parent_cb: QCheckBox, *children: QWidget) -> None:
    """Enable ``children`` only while ``parent_cb`` is checked, and sync
    immediately so the initial state is correct."""
    def _sync(checked: bool) -> None:
        for c in children:
            c.setEnabled(checked)
    parent_cb.toggled.connect(_sync)
    _sync(parent_cb.isChecked())


class SettingsDialog(QDialog):
    """Settings dialog: Display / System / Scan / Convert + post-convert.

    Theme + post-convert chain live under their natural homes. The
    inspector column visibility is NOT here — it's controlled via the
    table header's right-click menu, and that menu writes through to
    the same QSettings namespace.
    """

    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BIDS-Manager — Settings")
        self.resize(560, 600)
        self._settings = settings
        # Detected once: the worker-count spinboxes are capped at the host's
        # logical thread count so the user can never ask for more workers
        # than the machine has threads.
        self._sys: SystemInfo = get_system_info()

        v = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._build_display_tab(), "Display")
        tabs.addTab(self._build_system_tab(), "System")
        tabs.addTab(self._build_scan_tab(), "Scan")
        tabs.addTab(self._build_convert_tab(), "Convert + post-convert")
        v.addWidget(tabs, 1)

        # Save / Cancel / Restore defaults.
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        buttons.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        ).clicked.connect(self._on_restore_defaults)
        v.addWidget(buttons)

        # Populate every widget from the current settings.
        self._load_into_widgets(self._settings)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    # Font scale presets shown in the Display tab. The combo stores the
    # human-readable label; the float multiplier is the second element.
    _FONT_SCALE_PRESETS: list[tuple[str, float]] = [
        ("Compact (0.85x)",        0.85),
        ("Normal (1.00x)",         1.00),
        ("Comfortable (1.15x)",    1.15),
        ("Large (1.30x)",          1.30),
        ("Extra large (1.50x)",    1.50),
    ]

    # Combo presets for the header brand mark.
    _HEADER_LOGO_PRESETS: list[tuple[str, str]] = [
        ("Default (monochrome mark)", "default"),
        ("App icon (full color)",     "app_icon"),
    ]

    def _build_display_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "light"])
        form.addRow("Theme:", self._theme_combo)

        # Font scale: multiplies every font-size (QSS + delegate paints +
        # inline stylesheets + icon sizes) so the user can comfortably
        # nudge the whole UI up or down. Persisted under ``ui/font_scale``.
        self._font_scale_combo = QComboBox()
        for label, _value in self._FONT_SCALE_PRESETS:
            self._font_scale_combo.addItem(label)
        form.addRow("Font scale:", self._font_scale_combo)

        # Header brand artwork.
        self._header_logo_combo = QComboBox()
        for label, _value in self._HEADER_LOGO_PRESETS:
            self._header_logo_combo.addItem(label)
        form.addRow("Header logo:", self._header_logo_combo)

        hint = QLabel(
            "Theme can also be toggled live via the sun / moon button "
            "in the top header. Font scale and header logo apply on Save."
        )
        hint.setStyleSheet("color: #8b949e;")
        hint.setWordWrap(True)
        form.addRow("", hint)

        return w

    def _build_system_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        info = QGroupBox("System info (detected)")
        form = QFormLayout(info)

        threads = self._sys.logical_threads
        cores = self._sys.physical_cores
        cpu_txt = f"{threads} logical threads"
        if cores:
            cpu_txt += f"  /  {cores} physical cores"
        form.addRow("CPU:", QLabel(cpu_txt))

        ram_gib = self._sys.total_ram_gib
        form.addRow(
            "Memory:",
            QLabel(f"{ram_gib:.1f} GiB total" if ram_gib is not None else "unknown"),
        )

        v.addWidget(info)

        note = QLabel(
            "Parallel-worker counts (Scan and Convert) are capped at the "
            f"detected thread count ({threads}). Asking for more workers than "
            "the machine has threads only adds scheduling overhead, so the "
            "spinboxes will not go higher."
        )
        note.setStyleSheet("color: #8b949e;")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        return w

    def _build_scan_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        defaults = QGroupBox("Scan defaults")
        form = QFormLayout(defaults)

        self._scan_jobs = QSpinBox()
        self._scan_jobs.setRange(1, self._sys.logical_threads)
        self._scan_jobs.setToolTip(
            f"Capped at the detected thread count ({self._sys.logical_threads})."
        )
        form.addRow("Parallel workers (-j):", self._scan_jobs)

        self._scan_dataset = QLineEdit()
        self._scan_dataset.setPlaceholderText("(auto-derive from raw folder name)")
        form.addRow("Default dataset slug:", self._scan_dataset)

        self._scan_line_freq = QDoubleSpinBox()
        self._scan_line_freq.setRange(0.0, 100.0)
        self._scan_line_freq.setDecimals(1)
        self._scan_line_freq.setSingleStep(1.0)
        form.addRow("EEG/MEG line frequency (Hz):", self._scan_line_freq)

        self._scan_montage = QLineEdit()
        self._scan_montage.setPlaceholderText("e.g. standard_1005, biosemi64")
        form.addRow("EEG/MEG montage:", self._scan_montage)

        self._scan_probe = QCheckBox(
            "Enable --probe-convert (run dcm2niix per series to enrich "
            "naming with the actual file count + extensions)"
        )
        form.addRow("Probe:", self._scan_probe)

        self._scan_skip_bids_guess = QCheckBox(
            "Skip dcm2niix BidsGuess classifier (use only the legacy "
            "regex fallback layer)"
        )
        form.addRow("Classifier:", self._scan_skip_bids_guess)

        v.addWidget(defaults)
        v.addStretch(1)
        return w

    def _build_convert_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        convert = QGroupBox("Convert defaults")
        form = QFormLayout(convert)

        self._convert_jobs = QSpinBox()
        self._convert_jobs.setRange(1, self._sys.logical_threads)
        self._convert_jobs.setToolTip(
            f"Capped at the detected thread count ({self._sys.logical_threads})."
        )
        form.addRow("Parallel workers (-j):", self._convert_jobs)

        self._convert_overwrite = QCheckBox(
            "Overwrite existing subjects in the BIDS output (otherwise "
            "the convert verb refuses to clobber)"
        )
        form.addRow("Overwrite:", self._convert_overwrite)

        self._convert_skip_residuals = QCheckBox(
            "Skip residual volumes (drop dcm2niix secondary duplicates such "
            "as ..._bolda / _Eq_ / _ROI that are not real images)"
        )
        self._convert_skip_residuals.setToolTip(
            "dcm2niix splits a single input series into the real image plus "
            "derived single-volume duplicates it names ..._bolda, ..._Eq_1, "
            "etc. These have no valid BIDS suffix. Recommended: on."
        )
        form.addRow("Residuals:", self._convert_skip_residuals)

        v.addWidget(convert)

        # Post-convert chain laid out as an indented hierarchy: each step is
        # a parent checkbox; its sub-options sit indented beneath and are
        # enabled only while the parent is on.
        post = QGroupBox("Post-convert chain (run after every conversion)")
        pv = QVBoxLayout(post)
        pv.setSpacing(4)

        self._post_run_metadata = QCheckBox(
            "Generate metadata (dataset_description, participants.tsv, "
            "*_scans.tsv, sidecar audit)"
        )
        pv.addWidget(self._post_run_metadata)
        self._post_metadata_fill_todos = QCheckBox(
            "Insert 'TODO' placeholders for missing recommended fields"
        )
        pv.addWidget(_indented(self._post_metadata_fill_todos))

        self._post_run_validate = QCheckBox(
            "Validate dataset (schema + bidsschematools structural)"
        )
        pv.addWidget(self._post_run_validate)
        self._post_validate_strict = QCheckBox(
            "Strict: treat warnings as errors (--strict)"
        )
        self._post_validate_html = QCheckBox(
            "Write a self-contained validation_report.html (--html)"
        )
        pv.addWidget(_indented(self._post_validate_strict))
        pv.addWidget(_indented(self._post_validate_html))

        _bind_children(self._post_run_metadata, self._post_metadata_fill_todos)
        _bind_children(
            self._post_run_validate,
            self._post_validate_strict,
            self._post_validate_html,
        )

        v.addWidget(post)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------
    # Widget <-> settings
    # ------------------------------------------------------------------

    @classmethod
    def _header_logo_index(cls, value: str) -> int:
        for i, (_label, key) in enumerate(cls._HEADER_LOGO_PRESETS):
            if key == value:
                return i
        return 0

    @classmethod
    def _closest_font_scale_index(cls, value: float) -> int:
        """Return the preset index whose multiplier is nearest *value*."""
        try:
            return min(
                range(len(cls._FONT_SCALE_PRESETS)),
                key=lambda i: abs(cls._FONT_SCALE_PRESETS[i][1] - value),
            )
        except Exception:
            return 1  # "Normal"

    def _load_into_widgets(self, s: AppSettings) -> None:
        """Push every value from ``s`` into the dialog's widgets.

        Used both on open (with the live settings) and by "Restore
        defaults" (with a fresh ``AppSettings()``), so the two paths can
        never drift. Worker counts are clamped to the detected thread cap.
        """
        cap = self._sys.logical_threads

        self._theme_combo.setCurrentText(s.theme)
        self._font_scale_combo.setCurrentIndex(
            self._closest_font_scale_index(s.font_scale)
        )
        self._header_logo_combo.setCurrentIndex(
            self._header_logo_index(s.header_logo)
        )

        self._scan_jobs.setValue(max(1, min(s.scan_n_jobs, cap)))
        self._scan_dataset.setText(s.dataset_slug)
        self._scan_line_freq.setValue(s.scan_line_freq)
        self._scan_montage.setText(s.scan_montage)
        self._scan_probe.setChecked(s.scan_probe_convert)
        self._scan_skip_bids_guess.setChecked(s.scan_skip_bids_guess)

        self._convert_jobs.setValue(max(1, min(s.convert_n_jobs, cap)))
        self._convert_overwrite.setChecked(s.convert_overwrite)
        self._convert_skip_residuals.setChecked(s.convert_skip_residuals)

        self._post_run_metadata.setChecked(s.post_run_metadata)
        self._post_metadata_fill_todos.setChecked(s.post_metadata_fill_todos)
        self._post_run_validate.setChecked(s.post_run_validate)
        self._post_validate_strict.setChecked(s.post_validate_strict)
        self._post_validate_html.setChecked(s.post_validate_html)

    def _on_restore_defaults(self) -> None:
        """Reset all widgets to the AppSettings field defaults (not saved
        until the user clicks Save)."""
        self._load_into_widgets(AppSettings())

    def _on_save(self) -> None:
        s = self._settings
        s.theme = self._theme_combo.currentText()
        s.font_scale = self._FONT_SCALE_PRESETS[
            self._font_scale_combo.currentIndex()
        ][1]
        s.header_logo = self._HEADER_LOGO_PRESETS[
            self._header_logo_combo.currentIndex()
        ][1]

        s.scan_n_jobs = self._scan_jobs.value()
        s.dataset_slug = self._scan_dataset.text().strip()
        s.scan_line_freq = self._scan_line_freq.value()
        s.scan_montage = self._scan_montage.text().strip()
        s.scan_probe_convert = self._scan_probe.isChecked()
        s.scan_skip_bids_guess = self._scan_skip_bids_guess.isChecked()

        s.convert_n_jobs = self._convert_jobs.value()
        s.convert_overwrite = self._convert_overwrite.isChecked()
        s.convert_skip_residuals = self._convert_skip_residuals.isChecked()

        s.post_run_metadata = self._post_run_metadata.isChecked()
        s.post_metadata_fill_todos = self._post_metadata_fill_todos.isChecked()
        s.post_run_validate = self._post_run_validate.isChecked()
        s.post_validate_strict = self._post_validate_strict.isChecked()
        s.post_validate_html = self._post_validate_html.isChecked()

        s.save()
        self.accept()


__all__ = ["SettingsDialog"]
