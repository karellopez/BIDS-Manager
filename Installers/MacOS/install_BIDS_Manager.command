#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# BIDS-Manager Installer, macOS (Apple Silicon, arm64)
# Version:  1.0.1
# License:  MIT, ANCP Lab, University of Oldenburg
# Homepage: https://github.com/ANCPLabOldenburg/BIDS-Manager
#
# Downloads a portable Python 3.10 runtime, creates an isolated virtual
# environment, installs bids-manager from PyPI, and registers it as a
# native macOS application (~/Applications/BIDS-Manager.app) with icon.
#
# Icon source: the installed bids-manager package ships its logo at
#   bidsmgr/gui/assets/logo.png
# The installer pads it to a square canvas, renders a 1024x1024 PNG,
# and builds an .icns via iconutil. No pre-built .icns is required.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

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
APPLICATIONS_DIR="$HOME/Applications"
APP_BUNDLE="$APPLICATIONS_DIR/BIDS-Manager.app"
PYTHON_URL="https://raw.githubusercontent.com/ANCPLabOldenburg/BIDS-Manager/main/external/python-embed/cpython-3.10.13%2B20240107-aarch64-apple-darwin-install_only.tar.gz"
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
echo -e "  Platform:           macOS · Apple Silicon (arm64)"
echo -e "  Install directory:  ${BOLD}${INSTALL_DIR}${RESET}"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────

# 1. Disk space
AVAIL_MB=$(df -m "$HOME" | awk 'NR==2 {print $4}')
if (( AVAIL_MB < MIN_DISK_MB )); then
    log_warn "Low disk space: ${AVAIL_MB} MB available, ${MIN_DISK_MB} MB recommended."
    echo "    The installation may fail if there is insufficient space."
fi

# 2. Network
if ! curl -fsS --max-time 5 https://pypi.org > /dev/null 2>&1; then
    log_warn "Cannot reach pypi.org, please check your internet connection."
fi

# 3. Existing installation
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
        case "$choice" in
            [uU]) log_info "Upgrading existing installation..."; UPGRADE_MODE=true;  break ;;
            [rR]) log_info "Removing existing installation..."; rm -rf "$INSTALL_DIR"; UPGRADE_MODE=false; break ;;
            [aA]) echo "  Installation aborted."; exit 0 ;;
            *)    echo "  Please enter u, r, or a." ;;
        esac
    done
else
    UPGRADE_MODE=false
fi

# 4. Confirmation
echo ""
echo -e "${BOLD}  The installer will:${RESET}"
echo "    1. Download a portable Python 3.10 runtime (~60 MB)"
echo "    2. Create an isolated virtual environment"
echo "    3. Install bids-manager and all dependencies (~1.5 GB)"
echo "    4. Register BIDS-Manager as a macOS app (~/Applications/BIDS-Manager.app)"
echo "       Launch from Spotlight, Finder, or the Dock, just like any Mac app."
echo ""
read -rp "  Continue with installation? [Y/n] " confirm
if [[ "$confirm" == [nN] ]]; then
    echo "  Installation cancelled."
    exit 0
fi

# ── Prepare ───────────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR" "$APPLICATIONS_DIR"
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
    log_step "Step 1/4, Downloading portable Python 3.10 (arm64)..."
    cd "$INSTALL_DIR"
    ARCHIVE_NAME="$(basename "$PYTHON_URL" | sed 's/%2B/+/g')"
    curl -fSL --retry 3 --retry-delay 5 -o "$ARCHIVE_NAME" "$PYTHON_URL" \
        || die "Download failed. Please check your internet connection and try again."

    if ! file "$ARCHIVE_NAME" | grep -qi 'gzip'; then
        die "Downloaded file is not a valid archive. The download may be corrupted."
    fi

    log_step "Extracting Python runtime..."
    tar -xzf "$ARCHIVE_NAME" || die "Extraction failed."
    rm -f "$ARCHIVE_NAME"

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
# Point pip to the macOS system trust store so HTTPS to PyPI works.
if [[ -f /etc/ssl/cert.pem ]]; then
    export SSL_CERT_FILE=/etc/ssl/cert.pem
    export REQUESTS_CA_BUNDLE=/etc/ssl/cert.pem
fi

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
# Step 4/4, Create BIDS-Manager.app, uninstaller and Desktop shortcut
# ═════════════════════════════════════════════════════════════════════
log_step "Step 4/4, Building BIDS-Manager.app and shortcuts..."

# ── App bundle structure ───────────────────────────────────────────────
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# Executable: runs the `bidsmgr` console entry directly so macOS treats
# the app as a real, long-running application: Dock icon, Spotlight,
# pin-to-Dock all work. stdout / stderr are appended to
# ~/BIDS-Manager/bidsmgr.log for later inspection.
cat > "$APP_BUNDLE/Contents/MacOS/BIDS-Manager" << APP_EXEC_EOF
#!/bin/bash
source "$ENV_DIR/bin/activate"
exec bidsmgr "\$@" >> "$INSTALL_DIR/bidsmgr.log" 2>&1
APP_EXEC_EOF
chmod +x "$APP_BUNDLE/Contents/MacOS/BIDS-Manager"

# Info.plist: makes macOS recognise this directory as an application.
cat > "$APP_BUNDLE/Contents/Info.plist" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>       <string>BIDS-Manager</string>
    <key>CFBundleIdentifier</key>       <string>de.uni-oldenburg.ancplab.bidsmanager</string>
    <key>CFBundleName</key>             <string>BIDS-Manager</string>
    <key>CFBundleDisplayName</key>      <string>BIDS-Manager</string>
    <key>CFBundleIconFile</key>         <string>AppIcon</string>
    <key>CFBundleVersion</key>          <string>1.0.1</string>
    <key>CFBundleShortVersionString</key> <string>1.0.1</string>
    <key>CFBundlePackageType</key>      <string>APPL</string>
    <key>NSHighResolutionCapable</key>  <true/>
    <key>LSMinimumSystemVersion</key>   <string>11.0</string>
</dict>
</plist>
PLIST_EOF

# Icon: ask the installed bidsmgr package where its logo lives.
# Resolution order: pre-built .icns -> pre-built 1024 PNG -> logo.png + on-the-fly pad.
PKG_PATH=$("$ENV_DIR/bin/python3" -c \
    "import bidsmgr, os; print(os.path.dirname(bidsmgr.__file__))" 2>/dev/null || true)
ASSETS_ICNS=""
ASSETS_PNG1024=""
LOGO_PNG=""
if [[ -n "$PKG_PATH" ]]; then
    ASSETS_ICNS="$PKG_PATH/gui/assets/macos/AppIcon.icns"
    ASSETS_PNG1024="$PKG_PATH/gui/assets/macos/AppIcon1024.png"
    LOGO_PNG="$PKG_PATH/gui/assets/logo.png"
fi

SQUARE_PNG="/tmp/bids_manager_square_$$.png"
ICONSET_TMP="/tmp/bids_manager_$$.iconset"
ICON_OUT="$APP_BUNDLE/Contents/Resources/AppIcon.icns"

_build_icns_from_png() {
    # Build an .icns from a 1024x1024 PNG using pre-sized PNGs when
    # available (assets/macos/), otherwise resize on-the-fly with sips.
    local src="$1"
    local assets_dir
    assets_dir="$(dirname "$src")"
    mkdir -p "$ICONSET_TMP"

    _slot() {
        local size="$1" name="$2" preset="$3"
        if [[ -n "$preset" && -f "$preset" ]]; then
            cp "$preset" "$ICONSET_TMP/$name"
        else
            sips -z "$size" "$size" "$src" --out "$ICONSET_TMP/$name" >/dev/null 2>&1
        fi
    }

    _slot 16   icon_16x16.png      "$assets_dir/AppIcon16.png"
    _slot 32   icon_16x16@2x.png   "$assets_dir/AppIcon32.png"
    _slot 32   icon_32x32.png      "$assets_dir/AppIcon32.png"
    _slot 64   icon_32x32@2x.png   "$assets_dir/AppIcon64.png"
    _slot 128  icon_128x128.png    "$assets_dir/AppIcon128.png"
    _slot 256  icon_128x128@2x.png "$assets_dir/AppIcon256.png"
    _slot 256  icon_256x256.png    "$assets_dir/AppIcon256.png"
    _slot 512  icon_256x256@2x.png "$assets_dir/AppIcon512.png"
    _slot 512  icon_512x512.png    "$assets_dir/AppIcon512.png"
    _slot 1024 icon_512x512@2x.png "$assets_dir/AppIcon1024.png"

    iconutil -c icns "$ICONSET_TMP" -o "$ICON_OUT" 2>/dev/null
    local rc=$?
    rm -rf "$ICONSET_TMP"
    return $rc
}

_make_square_icon() {
    local src="$1" dst="$2"
    "$ENV_DIR/bin/python3" - "$src" "$dst" << 'PYEOF'
import sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
img = Image.open(src).convert("RGBA")
w, h = img.size
side = max(w, h)
square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
square.paste(img, ((side - w) // 2, (side - h) // 2))
pad = int(side * 0.15)
padded_side = side + 2 * pad
padded = Image.new("RGBA", (padded_side, padded_side), (0, 0, 0, 0))
padded.paste(square, (pad, pad))
padded = padded.resize((1024, 1024), Image.LANCZOS)
padded.save(dst, "PNG")
PYEOF
}

if [[ -f "$ASSETS_ICNS" ]]; then
    # Best path: a pre-built .icns ships in the package.
    cp "$ASSETS_ICNS" "$ICON_OUT"
    log_info "App icon installed from bundled assets (AppIcon.icns)."
elif [[ -f "$ASSETS_PNG1024" ]]; then
    log_step "Building app icon from bundled PNG assets..."
    if _build_icns_from_png "$ASSETS_PNG1024"; then
        log_info "App icon created from bundled PNG assets."
    else
        log_warn "iconutil failed, app will use the default macOS icon."
    fi
elif [[ -f "$LOGO_PNG" ]]; then
    log_step "Padding package logo to square canvas for app icon..."
    if _make_square_icon "$LOGO_PNG" "$SQUARE_PNG"; then
        _build_icns_from_png "$SQUARE_PNG" \
            && log_info "App icon created from package logo." \
            || log_warn "iconutil failed, app will use the default macOS icon."
    else
        log_warn "Icon generation failed, app will use the default macOS icon."
    fi
    rm -f "$SQUARE_PNG"
else
    log_warn "No icon source found, app will use the default macOS icon."
fi

# Remove Gatekeeper quarantine so the app launches on first double-click.
xattr -rd com.apple.quarantine "$APP_BUNDLE" 2>/dev/null || true
# Register with Launch Services so Spotlight and Finder see the app immediately.
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f "$APP_BUNDLE" 2>/dev/null || true

log_info "BIDS-Manager.app registered in ~/Applications"

# ── Uninstaller ───────────────────────────────────────────────────────
UNINSTALL_SCRIPT="$INSTALL_DIR/uninstall_BIDS_Manager.command"
cat > "$UNINSTALL_SCRIPT" << UNINSTALLER_EOF
#!/bin/bash
echo ""
echo "  BIDS-Manager Uninstaller"
echo ""
echo "  This will completely remove BIDS-Manager from:"
echo "    $INSTALL_DIR"
echo "    $APP_BUNDLE"
echo ""
read -rp "  Are you sure? [y/N] " confirm
if [[ "\$confirm" != [yY] ]]; then
    echo "  Uninstallation cancelled."
    exit 0
fi
echo ""
echo "  [*] Removing installation directory..."
rm -rf "$INSTALL_DIR"

echo "  [*] Removing BIDS-Manager.app..."
rm -rf "$APP_BUNDLE"

echo "  [*] Removing Desktop shortcut..."
rm -f "$DESKTOP_DIR/Uninstall_BIDS_Manager.command"

echo ""
echo "  ✅ BIDS-Manager has been completely removed."
echo ""
UNINSTALLER_EOF
chmod +x "$UNINSTALL_SCRIPT"

# ── Desktop shortcut (uninstaller only, launch via the .app) ─────────
if [[ -d "$DESKTOP_DIR" ]]; then
    cp "$UNINSTALL_SCRIPT" "$DESKTOP_DIR/Uninstall_BIDS_Manager.command"
    chmod +x "$DESKTOP_DIR/Uninstall_BIDS_Manager.command"
    xattr -d com.apple.quarantine "$DESKTOP_DIR/Uninstall_BIDS_Manager.command" 2>/dev/null || true
    log_info "Uninstaller shortcut placed on Desktop."
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
echo -e "    • Open ${CYAN}Finder → Applications${RESET} and double-click ${BOLD}BIDS-Manager${RESET}"
echo -e "    • Or search ${BOLD}BIDS-Manager${RESET} with Spotlight (⌘ Space)"
echo -e "    • Or drag ${BOLD}BIDS-Manager${RESET} from Applications to your Dock (right-click → Keep in Dock)"
echo -e "    BIDS-Manager runs as a native app, its icon appears in the Dock while active."
echo -e "    Log output is saved to: ${CYAN}$INSTALL_DIR/bidsmgr.log${RESET}"
echo ""
echo -e "  ${BOLD}Uninstall:${RESET}  Double-click ${CYAN}Uninstall_BIDS_Manager.command${RESET} on your Desktop"
echo -e "  ${BOLD}Log file:${RESET}   $LOG_FILE"
echo ""
echo "  Thank you for using BIDS-Manager!"
echo ""
