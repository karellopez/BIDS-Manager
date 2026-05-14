#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# BIDS-Manager Installer, Linux (x86_64)
# Version:  1.0.1
# License:  MIT, ANCP Lab, University of Oldenburg
# Homepage: https://github.com/ANCPLabOldenburg/BIDS-Manager
#
# Downloads a portable Python 3.10 runtime, creates an isolated virtual
# environment, installs bids-manager from PyPI, and registers it in the
# application menu with its own icon.
#
# Icon source: the installed bids-manager package ships its logo at
#   bidsmgr/gui/assets/logo.png
# The installer pads it to a square canvas and writes a 256x256 PNG into
# ~/.local/share/icons/hicolor/256x256/apps/. No pre-built icon required.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# Resolve the installer's own directory immediately, before any `cd`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colour helpers ────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; RESET=''
fi

log_info()  { echo -e "${GREEN}[✓]${RESET} $*"; }
log_step()  { echo -e "${CYAN}${BOLD}[*]${RESET} $*"; }
log_warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
log_error() { echo -e "${RED}[✗]${RESET} $*" >&2; }
die()       { log_error "$*"; echo "    Log file: ${LOG_FILE}"; exit 1; }

# ── Configuration ─────────────────────────────────────────────────────
INSTALLER_VERSION="1.0.1"
INSTALL_DIR="$HOME/BIDS-Manager"
ENV_DIR="$INSTALL_DIR/env"
LOG_FILE="$INSTALL_DIR/install.log"
PYPI_PKG="bids-manager"
DESKTOP_DIR="$HOME/Desktop"
APPDIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/256x256/apps"
PYTHON_URL="https://raw.githubusercontent.com/ANCPLabOldenburg/BIDS-Manager/main/external/python-embed/python-3.10.13%2B20240107-x86_64-unknown-linux-gnu-install_only.zip"
PYTHON_BIN="$INSTALL_DIR/bin/python3.10"
MIN_DISK_MB=2500

# ── Banner ────────────────────────────────────────────────────────────
clear 2>/dev/null || true
echo -e "${BOLD}"
cat << 'BANNER'

    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║   B I D S - M a n a g e r   I n s t a l l e r    ║
    ║                                                  ║
    ║    Schema-driven BIDS converter and editor       ║
    ║        ANCP Lab · University of Oldenburg        ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝

BANNER
echo -e "${RESET}"
echo -e "  Installer version:  ${BOLD}${INSTALLER_VERSION}${RESET}"
echo -e "  Platform:           Linux (x86_64)"
echo -e "  Install directory:  ${BOLD}${INSTALL_DIR}${RESET}"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────

# 1. Required tools
for tool in unzip wget; do
    if ! command -v "$tool" &>/dev/null; then
        die "'$tool' is required but not installed. Please install it:\n    sudo apt install $tool"
    fi
done

# 2. Disk space
AVAIL_MB=$(df -m "$HOME" | awk 'NR==2 {print $4}')
if (( AVAIL_MB < MIN_DISK_MB )); then
    log_warn "Low disk space: ${AVAIL_MB} MB available, ${MIN_DISK_MB} MB recommended."
    echo "    The installation may fail if there is insufficient space."
fi

# 3. Network
if command -v curl &>/dev/null; then
    curl -fsS --max-time 5 https://pypi.org > /dev/null 2>&1 \
        || log_warn "Cannot reach pypi.org, please check your internet connection."
fi

# 4. Existing installation
if [[ -d "$ENV_DIR" ]]; then
    echo -e "${YELLOW}  An existing BIDS-Manager installation was detected at:${RESET}"
    echo "    $INSTALL_DIR"
    echo ""
    echo "  [u] Upgrade: keep settings, reinstall the package"
    echo "  [r] Reinstall: remove everything and start fresh"
    echo "  [a] Abort"
    echo ""
    while true; do
        read -rp "  Your choice [u/r/a]: " choice
        case "${choice,,}" in
            u) log_info "Upgrading existing installation..."; UPGRADE_MODE=true;  break ;;
            r) log_info "Removing existing installation..."; rm -rf "$INSTALL_DIR"; UPGRADE_MODE=false; break ;;
            a) echo "  Installation aborted."; exit 0 ;;
            *) echo "  Please enter u, r, or a." ;;
        esac
    done
else
    UPGRADE_MODE=false
fi

# 5. Confirmation
echo ""
echo -e "${BOLD}  The installer will:${RESET}"
echo "    1. Download a portable Python 3.10 runtime (~65 MB)"
echo "    2. Create an isolated virtual environment"
echo "    3. Install bids-manager and all dependencies (~1.5 GB)"
echo "    4. Register BIDS-Manager in the application menu with its own icon"
echo "       BIDS-Manager opens in a terminal so you can follow its progress."
echo ""
read -rp "  Continue with installation? [Y/n] " confirm
if [[ "${confirm,,}" == "n" ]]; then
    echo "  Installation cancelled."
    exit 0
fi

# ── Prepare ───────────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR" "$APPDIR"
echo "" > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "BIDS-Manager installation started at $(date)"
echo "Installer version: $INSTALLER_VERSION"
echo ""

# ═════════════════════════════════════════════════════════════════════
# Step 1/4, Download portable Python runtime
# ═════════════════════════════════════════════════════════════════════
if [[ "$UPGRADE_MODE" == true && -x "$PYTHON_BIN" ]]; then
    log_info "Portable Python already present, skipping download."
else
    log_step "Step 1/4, Downloading portable Python 3.10 (x86_64)..."
    cd "$INSTALL_DIR"
    ZIP_NAME="$(basename "$PYTHON_URL" | sed 's/%2B/+/g')"
    wget -q --show-progress -O "$ZIP_NAME" "$PYTHON_URL" \
        || die "Download failed. Please check your internet connection and try again."

    log_step "Extracting Python runtime..."
    unzip -qo "$ZIP_NAME" || die "Extraction failed."
    rm -f "$ZIP_NAME"

    if [[ -d "python" ]]; then
        cp -a python/* . 2>/dev/null || true
        rm -rf python
    fi

    [[ -x "$PYTHON_BIN" ]] || die "Python binary not found at $PYTHON_BIN after extraction."
    log_info "Python 3.10 ready."
fi

# ═════════════════════════════════════════════════════════════════════
# Step 2/4, Create virtual environment
# ═════════════════════════════════════════════════════════════════════
log_step "Step 2/4, Creating virtual environment..."
"$PYTHON_BIN" -m venv "$ENV_DIR" || die "Failed to create virtual environment."
source "$ENV_DIR/bin/activate"

# python-build-standalone ships without CA certificates.
# Point pip to the system trust store so HTTPS to PyPI works.
for _cert in \
        /etc/ssl/certs/ca-certificates.crt \
        /etc/pki/tls/certs/ca-bundle.crt \
        /etc/ssl/ca-bundle.pem \
        /etc/ssl/cert.pem; do
    if [[ -f "$_cert" ]]; then
        export SSL_CERT_FILE="$_cert"
        export REQUESTS_CA_BUNDLE="$_cert"
        break
    fi
done

# ═════════════════════════════════════════════════════════════════════
# Step 3/4, Install bids-manager
# ═════════════════════════════════════════════════════════════════════
log_step "Step 3/4, Installing bids-manager (this may take a few minutes)..."
echo "    Upgrading pip..."
pip install --upgrade pip || log_warn "pip upgrade failed, continuing with existing version."

echo "    Installing $PYPI_PKG..."
if pip install "$PYPI_PKG"; then
    log_info "bids-manager installed successfully."
else
    log_warn "Standard install failed. Retrying with --trusted-host fallback..."
    pip install \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        "$PYPI_PKG" \
        || die "Package installation failed. Check the log file: $LOG_FILE"
    log_info "bids-manager installed successfully (via trusted-host fallback)."
fi
deactivate

# ═════════════════════════════════════════════════════════════════════
# Step 4/4, Create launcher, uninstaller, icon and shortcuts
# ═════════════════════════════════════════════════════════════════════
log_step "Step 4/4, Creating launcher scripts, icon and shortcuts..."

# ── App icon ──────────────────────────────────────────────────────────
# Ask the installed package where it lives, then locate the bundled
# logo. The installer pads / resizes on the fly because BIDS-Manager
# does not currently ship pre-sized PNGs.
PKG_PATH=$("$ENV_DIR/bin/python3" -c \
    "import bidsmgr, os; print(os.path.dirname(bidsmgr.__file__))" 2>/dev/null || true)
ASSETS_PNG=""
LOGO_PNG=""
if [[ -n "$PKG_PATH" ]]; then
    ASSETS_PNG="$PKG_PATH/gui/assets/macos/AppIcon256.png"
    LOGO_PNG="$PKG_PATH/gui/assets/logo.png"
fi
ICON_FILE="$ICON_DIR/bids-manager.png"

_make_square_png() {
    local src="$1" dst="$2" size="${3:-256}"
    "$ENV_DIR/bin/python3" - "$src" "$dst" "$size" << 'PYEOF'
import sys
from PIL import Image
src, dst, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
img = Image.open(src).convert("RGBA")
w, h = img.size
side = max(w, h)
canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
canvas.paste(img, ((side - w) // 2, (side - h) // 2))
canvas = canvas.resize((size, size), Image.LANCZOS)
canvas.save(dst, "PNG")
PYEOF
}

if [[ -f "$ASSETS_PNG" ]]; then
    mkdir -p "$ICON_DIR"
    cp "$ASSETS_PNG" "$ICON_FILE"
    gtk-update-icon-cache --force --ignore-theme-index \
        "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
    log_info "App icon installed from bundled assets (AppIcon256.png)."
    DESKTOP_ICON="bids-manager"
elif [[ -f "$LOGO_PNG" ]]; then
    log_step "Padding package logo to square canvas for app icon..."
    mkdir -p "$ICON_DIR"
    if _make_square_png "$LOGO_PNG" "$ICON_FILE" 256; then
        log_info "App icon created from package logo."
    else
        log_warn "Pillow padding failed, copying raw logo (may appear distorted)."
        cp "$LOGO_PNG" "$ICON_FILE"
    fi
    gtk-update-icon-cache --force --ignore-theme-index \
        "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
    log_info "App icon installed to icon theme."
    DESKTOP_ICON="bids-manager"
else
    log_warn "No icon source found, shortcuts will use the default icon."
    DESKTOP_ICON="utilities-terminal"
fi

# ── Launcher ──────────────────────────────────────────────────────────
RUN_SCRIPT="$INSTALL_DIR/run_BIDS_Manager.sh"
cat > "$RUN_SCRIPT" << LAUNCHER_EOF
#!/usr/bin/env bash
source "$ENV_DIR/bin/activate"
exec bidsmgr "\$@"
LAUNCHER_EOF
chmod +x "$RUN_SCRIPT"

# ── Uninstaller ───────────────────────────────────────────────────────
UNINSTALL_SCRIPT="$INSTALL_DIR/uninstall_BIDS_Manager.sh"
cat > "$UNINSTALL_SCRIPT" << UNINSTALLER_EOF
#!/usr/bin/env bash
echo ""
echo "  BIDS-Manager Uninstaller"
echo ""
echo "  This will completely remove BIDS-Manager from:"
echo "    $INSTALL_DIR"
echo ""
read -rp "  Are you sure? [y/N] " confirm
if [[ "\${confirm,,}" != "y" ]]; then
    echo "  Uninstallation cancelled."
    exit 0
fi
echo ""
echo "  [*] Removing installation directory..."
rm -rf "$INSTALL_DIR"

echo "  [*] Removing Desktop shortcuts..."
rm -f "$DESKTOP_DIR/BIDS-Manager.desktop"
rm -f "$DESKTOP_DIR/Uninstall_BIDS_Manager.desktop"
rm -f "$APPDIR/BIDS-Manager.desktop"
rm -f "$APPDIR/Uninstall_BIDS_Manager.desktop"

echo "  [*] Removing app icon..."
rm -f "$ICON_FILE"
gtk-update-icon-cache "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

echo ""
echo "  ✅ BIDS-Manager has been completely removed."
echo ""
UNINSTALLER_EOF
chmod +x "$UNINSTALL_SCRIPT"

# ── Desktop .desktop entries ──────────────────────────────────────────
make_desktop_entry() {
    local dest="$1" name="$2" comment="$3" exec_cmd="$4" icon="$5"
    cat > "$dest" << DEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$name
Comment=$comment
Exec=bash -c "$exec_cmd"
Terminal=true
Icon=$icon
Categories=Science;Education;
DEOF
    chmod +x "$dest"
}

make_desktop_entry "$APPDIR/BIDS-Manager.desktop" \
    "BIDS-Manager" "Schema-driven BIDS converter and editor" \
    "$RUN_SCRIPT" "$DESKTOP_ICON"

make_desktop_entry "$APPDIR/Uninstall_BIDS_Manager.desktop" \
    "Uninstall BIDS-Manager" "Completely remove BIDS-Manager" \
    "$UNINSTALL_SCRIPT" "system-software-update"

if [[ -d "$DESKTOP_DIR" ]]; then
    cp "$APPDIR/BIDS-Manager.desktop"           "$DESKTOP_DIR/"
    cp "$APPDIR/Uninstall_BIDS_Manager.desktop" "$DESKTOP_DIR/"
    gio set "$DESKTOP_DIR/BIDS-Manager.desktop"           metadata::trusted true 2>/dev/null || true
    gio set "$DESKTOP_DIR/Uninstall_BIDS_Manager.desktop" metadata::trusted true 2>/dev/null || true
    log_info "Desktop shortcuts created."
else
    log_warn "Desktop directory not found, shortcuts available in the application menu."
fi

# ═════════════════════════════════════════════════════════════════════
# Done.
# ═════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  ✅  BIDS-Manager was successfully installed!${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${BOLD}Launch BIDS-Manager:${RESET}"
echo -e "    • Search ${CYAN}BIDS-Manager${RESET} in your application menu"
echo -e "    • Or double-click ${CYAN}BIDS-Manager${RESET} on your Desktop"
echo -e "    • Or run: bash $RUN_SCRIPT"
echo -e "    A terminal will open showing BIDS-Manager's progress."
echo ""
echo -e "  ${BOLD}Uninstall:${RESET}  Double-click ${CYAN}Uninstall BIDS-Manager${RESET} on your Desktop"
echo -e "  ${BOLD}Log file:${RESET}   $LOG_FILE"
echo ""
echo "  Thank you for using BIDS-Manager!"
echo ""
