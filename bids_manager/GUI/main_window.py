"""Main window assembly combining conversion and editor tabs."""

from bids_manager.GUI.common import *
from bids_manager.GUI.converter import ConverterMixin
from bids_manager.GUI.editor import EditorMixin, CpuSettingsDialog, AuthorshipDialog, DpiSettingsDialog

class BIDSManager(QMainWindow, ConverterMixin, EditorMixin):
    """Primary GUI window coordinating editor and converter modules."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIDS Manager")
        if ICON_FILE.exists():
            self.setWindowIcon(QIcon(str(ICON_FILE)))
        self.resize(900, 900)
        self.setMinimumSize(640, 480)
    
        app = QApplication.instance()
        self._base_font = app.font()
        screen = app.primaryScreen()
        # Detect the OS DPI scaling in a resilient way.  ``logicalDotsPerInch``
        # is usually the most accurate value because it already factors in the
        # operating-system scale factor.  On some platforms (e.g. X11 with
        # unusual monitor reporting) this can be inaccurate, so we fall back to
        # the device pixel ratio when needed.  The result is stored as a
        # percentage relative to the 96 DPI base expected by Qt.
        self._os_dpi = self._detect_system_dpi(screen)
    
        # User-requested DPI scale.  By default we start from the detected
        # system value so that the UI matches the OS scaling out of the box.
        # Users can still fine-tune the value manually via the DPI dialog.
        self.dpi_scale = self._os_dpi
    
        # Paths
        self.dicom_dir = ""         # Raw DICOM directory
        self.bids_out_dir = ""      # Output BIDS directory
        self.tsv_path = ""          # Path to subject_summary.tsv
        self.heuristic_dir = ""     # Directory with heuristics
        # Lookup containers used to synchronise the mapping table with the
        # modality trees.  Keys are different path elements and values are lists
        # of row indices in ``self.mapping_table``.
        self.study_set = set()        # All study names encountered
        self.modb_rows = {}           # BIDS modality → [row, ...]
        self.mod_rows = {}            # (BIDS modality, modality) → [row, ...]
        self.seq_rows = {}            # (BIDS modality, modality, sequence) → rows
        self.study_rows = {}
        self.subject_rows = {}
        self.session_rows = {}
        self.spec_modb_rows = {}
        self.spec_mod_rows = {}
        self.spec_seq_rows = {}
        # Equivalent lookups when displaying the "given" subject names instead
        # of the BIDS names
        self.subject_rows_given = {}
        self.session_rows_given = {}
        self.spec_modb_rows_given = {}
        self.spec_mod_rows_given = {}
        self.spec_seq_rows_given = {}
        # Existing mappings found in output datasets
        self.existing_maps = {}
        self.existing_used = {}
        self.use_bids_names = True
    
        # Track repeat-detection state so we can notify the user when new
        # duplicates are found and keep the "Only last repeats" option in sync.
        self._last_repeat_count = 0
    
        # Flag used to skip expensive updates while the mapping table is being
        # populated programmatically.  ``QTableWidget`` emits ``itemChanged``
        # signals even when values are assigned in code, so we guard callbacks
        # that rebuild caches with this boolean.
        self._loading_mapping_table = False
    
        # Async process handles for inventory and conversion steps
        self.inventory_process = None  # QProcess for dicom_inventory
        self.conv_process = None       # QProcess for the conversion pipeline
        self.conv_stage = 0            # Tracks which step of the pipeline ran
        self.heurs_to_rename = []      # List of heuristics pending rename
        # Background worker used to detect mixed StudyInstanceUID folders
        self._conflict_thread: Optional[QThread] = None
        self._conflict_worker: Optional[_ConflictScannerWorker] = None
    
        # Root of the currently loaded BIDS dataset (None until loaded)
        self.bids_root = None
    
        # Schema information for proposed BIDS names
        self._schema = None
        if ENABLE_SCHEMA_RENAMER:
            try:
                self._schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
            except Exception as e:
                print(f"[WARN] Could not load BIDS schema: {e}")
                self._schema = None
        self.inventory_df = None
    
        # Path to persistent user preferences
        self.pref_dir = PREF_DIR
        try:
            self.pref_dir.mkdir(exist_ok=True, parents=True)
        except Exception:
            pass
        self.exclude_patterns_file = self.pref_dir / "exclude_patterns.tsv"
        self.theme_file = self.pref_dir / "theme.txt"
        self.dpi_file = self.pref_dir / "dpi_scale.txt"
        # Load any previously stored DPI preference so the UI scale persists
        # across sessions.  Invalid or out-of-range values fall back to the
        # detected system scale.  This keeps the first launch aligned with the
        # host environment while preserving user tweaks afterwards.
        if self.dpi_file.exists():
            try:
                saved_dpi = int(self.dpi_file.read_text().strip())
            except ValueError:
                saved_dpi = None
            except Exception:
                saved_dpi = None
            if saved_dpi is not None:
                self.dpi_scale = max(50, min(200, saved_dpi))
        self.seq_dict_file = dicom_inventory.SEQ_DICT_FILE
        # Flag used to automatically reapply the suffix dictionary after a fresh
        # scan.  This keeps the scanned data table in sync with the latest
        # custom patterns without overriding manual edits made later via
        # ``applyMappingChanges``.
        self._apply_sequence_on_load = False
    
        # Spinner for long-running tasks
        self.spinner_label = None
        # Timer and unicode characters for the small animated spinner that
        # appears while long-running subprocesses are running
        self._spinner_timer = QTimer()
        self._spinner_timer.timeout.connect(self._spin)
        self._spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self._spinner_index = 0
        self._spinner_message = ""
    
        # Parallel settings
        # Use ~80% of available CPUs by default to avoid saturating the system
        # when running external tools in parallel.  ``os.cpu_count`` may return
        # ``None`` so fall back to 1 in that case.
        total_cpu = os.cpu_count() or 1
        # ``current`` is pre-set to about 80% of this value in the main window.
        self.num_cpus = max(1, round(total_cpu * 0.8))
    
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
        self._updateMappingControlsEnabled()
    
        # Theme support
        self.statusBar()
        self.themes = self._build_theme_dict()
        self.current_theme = None
        self.theme_btn = QPushButton("🌓")  # half-moon icon
        self.theme_btn.setFixedWidth(50)
        self.cpu_btn = QPushButton(f"CPU: {self.num_cpus}")
        self.cpu_btn.setFixedWidth(70)
        self.cpu_btn.clicked.connect(self.show_cpu_dialog)
        self.authorship_btn = QPushButton("Authorship")
        self.authorship_btn.setFixedWidth(90)
        self.authorship_btn.clicked.connect(self.show_authorship_dialog)
        self.dpi_btn = QPushButton(f"DPI: {self.dpi_scale}%")
        self.dpi_btn.setFixedWidth(80)
        self.dpi_btn.clicked.connect(self.show_dpi_dialog)
        # Create a container widget with layout to adjust position
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 2, 0, 6)  # left, top, right, bottom
        layout.setSpacing(8)
        layout.addWidget(self.theme_btn)
        layout.addWidget(self.cpu_btn)
        layout.addWidget(self.dpi_btn)
        layout.addWidget(self.authorship_btn)
        container.setLayout(layout)
        # Add the container to the status bar (left-aligned)
        self.statusBar().addWidget(container)
        # Create the theme menu
        theme_menu = QMenu(self)
        for name in self.themes.keys():
            act = theme_menu.addAction(name)
            act.triggered.connect(lambda _=False, n=name: self.apply_theme(n))
        self.theme_btn.setMenu(theme_menu)
    
        # Load previously saved theme preference
        default_theme = "Light"
        if self.theme_file.exists():
            try:
                default_theme = self.theme_file.read_text().strip() or default_theme
            except Exception:
                pass
        self.apply_theme(default_theme)

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
        self.current_theme = name
        try:
            self.theme_file.write_text(name)
        except Exception:
            pass
        self._update_logo()
        self._apply_font_scale()

    def show_cpu_dialog(self) -> None:
        """Display dialog to choose number of CPUs."""
        dlg = CpuSettingsDialog(self, self.num_cpus)
        if dlg.exec_() == QDialog.Accepted:
            self.num_cpus = dlg.spin.value()
            self.cpu_btn.setText(f"CPU: {self.num_cpus}")

    def show_authorship_dialog(self) -> None:
        """Display authorship information dialog."""
        dlg = AuthorshipDialog(self)
        dlg.exec_()

    def show_dpi_dialog(self) -> None:
        """Display dialog to adjust UI scale."""
        dlg = DpiSettingsDialog(self, self.dpi_scale)
        if dlg.exec_() == QDialog.Accepted:
            self.dpi_scale = dlg.spin.value()
            self.dpi_btn.setText(f"DPI: {self.dpi_scale}%")
            self._apply_font_scale()
            # Persist the user-selected DPI so the preference is restored next
            # time the application starts.
            try:
                self.dpi_file.write_text(str(self.dpi_scale))
            except Exception:
                pass

    def _start_spinner(self, message: str) -> None:
        """Show animated spinner with *message* in the log group."""
        self._spinner_message = message
        self._spinner_index = 0
        if self.spinner_label is not None:
            self.spinner_label.setText(f"{message} {self._spinner_frames[0]}")
            self.spinner_label.show()
        self._spinner_timer.start(100)

    def _spin(self) -> None:
        if not self.spinner_label or not self.spinner_label.isVisible():
            return
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        self.spinner_label.setText(
            f"{self._spinner_message} {self._spinner_frames[self._spinner_index]}"
        )

    def _stop_spinner(self) -> None:
        self._spinner_timer.stop()
        if self.spinner_label is not None:
            self.spinner_label.hide()

    def _is_dark_theme(self) -> bool:
        color = self.palette().color(QPalette.Window)
        brightness = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        return brightness < 128

    def _apply_font_scale(self) -> None:
        """Apply current DPI scaling to the application font."""
        app = QApplication.instance()
        font = QFont(self._base_font)
        base_size = self._base_font.pointSizeF() or float(self._base_font.pointSize())
        # Calculate the font size relative to the detected system DPI so that
        # ``dpi_scale`` always represents the intended scale percentage of the
        # 96-DPI baseline.  Using a ratio keeps the default matching the OS while
        # letting users make incremental adjustments without surprises.
        scale_ratio = max(0.5, min(2.0, self.dpi_scale / max(self._os_dpi, 1)))
        scaled = max(1.0, base_size * scale_ratio)
        if self.current_theme in ("Contrast", "Contrast White"):
            font.setWeight(QFont.Bold)
            font.setPointSizeF(scaled + 1)
        else:
            font.setWeight(QFont.Normal)
            font.setPointSizeF(scaled)
        app.setFont(font)
    
        # Ensure the tab labels also scale with the selected DPI
        if hasattr(self, "tabs"):
            tab_font = QFont(font)
            tab_font.setPointSizeF(font.pointSizeF() + 1)
            self.tabs.setFont(tab_font)

    def _detect_system_dpi(self, screen):
        """Return the system DPI percentage relative to 96 with fallbacks.
    
        The logic prefers ``logicalDotsPerInch`` (Qt's effective DPI that already
        accounts for the OS scale).  When that value is implausible we try the
        physical DPI and the device pixel ratio, selecting the first reasonable
        candidate.  This helps avoid wildly large or tiny UI defaults on
        platforms that misreport DPI values.
        """
    
        if screen is None:
            return 100
    
        candidates = []
        try:
            logical = screen.logicalDotsPerInch()
            if logical > 1:
                candidates.append(logical / 96 * 100)
        except Exception:
            pass
    
        try:
            physical = screen.physicalDotsPerInch()
            if physical > 1:
                candidates.append(physical / 96 * 100)
        except Exception:
            pass
    
        try:
            ratio = screen.devicePixelRatio()
            if ratio > 0:
                candidates.append(ratio * 100)
        except Exception:
            pass
    
        for candidate in candidates:
            if 25 <= candidate <= 400:
                return int(round(candidate))
    
        # Fallback to a safe default when nothing sensible is reported.
        return 100

    def _update_logo(self) -> None:
        """Update logo pixmap based on current theme."""
        if not hasattr(self, "logo_label"):
            return
        if not LOGO_FILE.exists():
            return
        pix = QPixmap(str(LOGO_FILE))
        if self._is_dark_theme():
            img = pix.toImage()
            img.invertPixels()
            pix = QPixmap.fromImage(img)
        self.logo_label.setPixmap(pix.scaledToHeight(120, Qt.SmoothTransformation))

def main() -> None:
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                u"bids.manager.1.0"
            )
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    if ICON_FILE.exists():
        app.setWindowIcon(QIcon(str(ICON_FILE)))
    win = BIDSManager()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

