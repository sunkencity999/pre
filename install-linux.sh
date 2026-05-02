#!/bin/bash
# install-linux.sh — Full setup for PRE (Personal Reasoning Engine) on Linux
#
# This script handles everything from Ollama to ready-to-run:
#   0. Checks system requirements (distro, GPU/VRAM, RAM, disk)
#   1. Installs/verifies Ollama
#   1b. Configures Ollama environment (FA, KV cache, keep-alive)
#   2. Pulls the base model (q8_0 ~28GB or q4_K_M ~15GB, auto-selected by VRAM)
#   2b. Pulls embedding model (nomic-embed-text for experience ledger + RAG)
#   3. Creates optimized custom model (pre-gemma4) from Modelfile
#   4. Checks for Node.js
#   5. Sets up Web GUI (npm install)
#   6. Sets up ~/.pre/ directories
#   7. Auto-sizes context window based on RAM headroom
#   8. Optional: voice tools (Whisper, FFmpeg, espeak-ng)
#   9. Optional: GNOME PIM integration (evolution-data-server)
#   10. Optional: desktop automation deps (xdotool, scrot, xclip)
#   11. Optional: auto-start (systemd user service)
#   12. Pre-warms model into GPU memory
#   13. Creates ~/bin/pre launcher script
#
# Target distros: Ubuntu/Debian (primary), Fedora/RHEL (secondary), Arch (best-effort)
#
# Usage:
#   ./install-linux.sh           Interactive install
#   ./install-linux.sh --yes     Non-interactive: accept all defaults, skip optional steps
#   ./install-linux.sh -y        Same as --yes

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$REPO_DIR/engine"
CUSTOM_MODEL="pre-gemma4"

# --- Flag parsing ---
AUTO_YES=false
for arg in "$@"; do
    case "$arg" in
        --yes|-y) AUTO_YES=true ;;
    esac
done

ask_yn() {
    local prompt="$1" default="$2"
    if [ "$AUTO_YES" = true ]; then
        [ "$default" = "Y" ] && return 0 || return 1
    fi
    read -p "$prompt" -n 1 -r
    echo ""
    if [ "$default" = "Y" ]; then
        [[ ! $REPLY =~ ^[Nn]$ ]]
    else
        [[ $REPLY =~ ^[Yy]$ ]]
    fi
}

step() { echo -e "\n${BOLD}${CYAN}=== $1 ===${RESET}\n"; }
ok()   { echo -e "${GREEN}$1${RESET}"; }
warn() { echo -e "${YELLOW}$1${RESET}"; }
fail() { echo -e "${RED}$1${RESET}"; exit 1; }

trap 'echo -e "\n${RED}Installation failed at line $LINENO.${RESET}\n${DIM}Re-run the script to retry — it will pick up where it left off.${RESET}"' ERR

# ── Distro detection ──
detect_pkg_manager() {
    if command -v apt-get &>/dev/null; then
        PKG_MGR="apt"
        PKG_INSTALL="sudo apt-get install -y"
        PKG_UPDATE="sudo apt-get update"
    elif command -v dnf &>/dev/null; then
        PKG_MGR="dnf"
        PKG_INSTALL="sudo dnf install -y"
        PKG_UPDATE="sudo dnf check-update || true"
    elif command -v pacman &>/dev/null; then
        PKG_MGR="pacman"
        PKG_INSTALL="sudo pacman -S --noconfirm"
        PKG_UPDATE="sudo pacman -Sy"
    else
        PKG_MGR="unknown"
        PKG_INSTALL=""
        PKG_UPDATE=""
    fi
}

detect_pkg_manager

# ============================================================================
# Step 0: System requirements
# ============================================================================
step "Checking system requirements"

OS=$(uname -s)
if [ "$OS" != "Linux" ]; then
    fail "This installer is for Linux. Detected: $OS. Use install.sh for macOS."
fi

# Distro info
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO_NAME="$NAME"
    DISTRO_VER="$VERSION_ID"
    ok "  Distro: $DISTRO_NAME $DISTRO_VER"
else
    DISTRO_NAME="Unknown"
    warn "  Unable to detect distro (no /etc/os-release)"
fi

ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "aarch64" ]; then
    ok "  Architecture: $ARCH"
else
    fail "Unsupported architecture: $ARCH (need x86_64 or aarch64)"
fi

# ── GPU detection ──
GPU_BACKEND=""
GPU_NAME=""
GPU_VRAM_GB=0

# NVIDIA GPU detection (native — no Docker needed on Linux)
if command -v nvidia-smi &>/dev/null; then
    NVOUT=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
    if [ -n "$NVOUT" ]; then
        GPU_BACKEND="cuda"
        GPU_NAME=$(echo "$NVOUT" | cut -d',' -f1 | xargs)
        VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
        if [ -n "$VRAM_MIB" ] && [ "$VRAM_MIB" -gt 0 ] 2>/dev/null; then
            GPU_VRAM_GB=$((VRAM_MIB / 1024))
        fi
        ok "  GPU: $GPU_NAME (NVIDIA CUDA)"
        if [ "$GPU_VRAM_GB" -gt 0 ]; then
            ok "  VRAM: ${GPU_VRAM_GB}GB"
        fi
    fi
fi

# AMD GPU detection (ROCm — future support)
if [ -z "$GPU_BACKEND" ] && command -v rocm-smi &>/dev/null; then
    AMD_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -i "card" | head -1 | sed 's/.*: //' || true)
    if [ -n "$AMD_NAME" ]; then
        GPU_BACKEND="rocm"
        GPU_NAME="$AMD_NAME"
        # ROCm VRAM detection
        AMD_VRAM=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total" | head -1 | grep -oE '[0-9]+' || true)
        if [ -n "$AMD_VRAM" ] && [ "$AMD_VRAM" -gt 0 ] 2>/dev/null; then
            GPU_VRAM_GB=$((AMD_VRAM / 1024 / 1024 / 1024))
        fi
        ok "  GPU: $GPU_NAME (AMD ROCm)"
        if [ "$GPU_VRAM_GB" -gt 0 ]; then
            ok "  VRAM: ${GPU_VRAM_GB}GB"
        fi
    fi
fi

# Fallback: lspci detection
if [ -z "$GPU_BACKEND" ]; then
    LSPCI_GPU=$(lspci 2>/dev/null | grep -i 'vga\|3d\|display' | head -1 || true)
    if [ -n "$LSPCI_GPU" ]; then
        warn "  GPU: $LSPCI_GPU (no CUDA/ROCm driver detected)"
        warn "  Install NVIDIA or AMD GPU drivers for GPU acceleration."
    else
        warn "  No GPU detected — Ollama will use CPU only (very slow for 26B model)."
    fi
fi

# RAM
RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
RAM_GB=$((RAM_KB / 1048576))

if [ "$RAM_GB" -lt 16 ]; then
    fail "PRE requires at least 16GB RAM. Detected: ${RAM_GB}GB"
fi
if [ "$RAM_GB" -lt 32 ]; then
    warn "  RAM: ${RAM_GB}GB — functional, but 32GB+ recommended."
else
    ok "  RAM: ${RAM_GB}GB"
fi

# ── Quantization selection (VRAM-aware) ──
if [ "$GPU_VRAM_GB" -ge 28 ]; then
    BASE_MODEL="gemma4:26b-a4b-it-q8_0"
    QUANT="q8_0"
    MODEL_SIZE_GB=28
    ok "  Quant: q8_0 (near-lossless, ~28GB) — fits in ${GPU_VRAM_GB}GB VRAM"
elif [ "$GPU_VRAM_GB" -gt 0 ]; then
    BASE_MODEL="gemma4:26b-a4b-it-q4_K_M"
    QUANT="q4_K_M"
    MODEL_SIZE_GB=15
    ok "  Quant: q4_K_M (~15GB) — fits in ${GPU_VRAM_GB}GB VRAM for full GPU acceleration"
elif [ "$RAM_GB" -ge 32 ]; then
    # No GPU detected — CPU mode, enough RAM for q8_0
    BASE_MODEL="gemma4:26b-a4b-it-q8_0"
    QUANT="q8_0"
    MODEL_SIZE_GB=28
    warn "  Quant: q8_0 (CPU mode — inference will be slow without GPU)"
else
    BASE_MODEL="gemma4:26b-a4b-it-q4_K_M"
    QUANT="q4_K_M"
    MODEL_SIZE_GB=15
    warn "  Quant: q4_K_M (CPU mode, limited RAM)"
fi

# Context window sizing (headroom-based, same thresholds as macOS/Windows)
HEADROOM=$((RAM_GB - MODEL_SIZE_GB))
if [ "$HEADROOM" -ge 68 ]; then
    OPTIMAL_CTX=131072
elif [ "$HEADROOM" -ge 36 ]; then
    OPTIMAL_CTX=65536
elif [ "$HEADROOM" -ge 20 ]; then
    OPTIMAL_CTX=32768
elif [ "$HEADROOM" -ge 8 ]; then
    OPTIMAL_CTX=16384
elif [ "$HEADROOM" -ge 4 ]; then
    OPTIMAL_CTX=8192
else
    OPTIMAL_CTX=4096
fi
CTX_HUMAN="$(( OPTIMAL_CTX / 1024 ))K"
ok "  Context window: ${CTX_HUMAN} (${OPTIMAL_CTX} tokens, ${HEADROOM}GB headroom after ${QUANT} model)"

# Disk space
AVAIL_KB=$(df -k "$REPO_DIR" | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1048576))
DISK_NEEDED=$((MODEL_SIZE_GB + 5))
if [ "$AVAIL_GB" -lt "$DISK_NEEDED" ]; then
    warn "  Disk: ${AVAIL_GB}GB available — need ~${MODEL_SIZE_GB}GB for ${QUANT} model download."
    if ! ask_yn "  Continue anyway? [y/N] " "N"; then exit 1; fi
else
    ok "  Disk: ${AVAIL_GB}GB available"
fi

echo ""
ok "System requirements met."

# ============================================================================
# Step 1: Ollama
# ============================================================================
step "Checking Ollama"

if command -v ollama &>/dev/null; then
    OLLAMA_VER=$(ollama --version 2>&1 | head -1)
    ok "  Ollama installed: $OLLAMA_VER"
else
    echo "  Ollama is not installed."
    if ask_yn "  Install Ollama via official installer? [Y/n] " "Y"; then
        echo "  Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        ok "  Ollama installed."
    else
        echo ""
        echo "  Install Ollama manually:"
        echo "    curl -fsSL https://ollama.com/install.sh | sh"
        echo "    — or —"
        echo "    Download from https://ollama.com/download"
        exit 1
    fi
fi

# Ensure Ollama is running
PORT="${PRE_PORT:-11434}"
if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "  Starting Ollama..."
    # On Linux, Ollama typically runs as a systemd service
    if systemctl is-active ollama &>/dev/null; then
        ok "  Ollama service already active."
    elif systemctl start ollama &>/dev/null 2>&1; then
        sleep 3
        ok "  Started Ollama systemd service."
    else
        # Fallback: start manually
        ollama serve >/dev/null 2>&1 &
        sleep 3
    fi
    if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        warn "  Could not start Ollama automatically."
        echo "  Please run 'ollama serve' in another terminal, then re-run this script."
        exit 1
    fi
fi
ok "  Ollama running on port $PORT."

# ============================================================================
# Step 1b: Ollama environment
# ============================================================================
step "Configuring Ollama environment"

# Flash Attention: ON for CUDA (NVIDIA benefits), OFF if no GPU / ROCm (safe default)
FA_VAL=0
if [ "$GPU_BACKEND" = "cuda" ]; then
    FA_VAL=1
fi

KV_CACHE_TYPE="q8_0"
if [ "$QUANT" = "q4_K_M" ]; then
    KV_CACHE_TYPE="q4_0"
fi

# Write Ollama env vars to systemd override (if Ollama runs as systemd service)
OLLAMA_SERVICE_DIR="/etc/systemd/system/ollama.service.d"
if systemctl is-enabled ollama &>/dev/null 2>&1; then
    if [ -d "/etc/systemd/system" ]; then
        sudo mkdir -p "$OLLAMA_SERVICE_DIR"
        cat <<ENVEOF | sudo tee "$OLLAMA_SERVICE_DIR/pre-environment.conf" >/dev/null
[Service]
Environment="OLLAMA_FLASH_ATTENTION=${FA_VAL}"
Environment="OLLAMA_KV_CACHE_TYPE=${KV_CACHE_TYPE}"
Environment="OLLAMA_KEEP_ALIVE=24h"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
ENVEOF
        sudo systemctl daemon-reload
        sudo systemctl restart ollama
        sleep 3
        ok "  Ollama systemd environment configured (FA=${FA_VAL}, KV=${KV_CACHE_TYPE})"
    fi
fi

# Also persist in shell profile
SHELL_RC=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.profile" ]; then
    SHELL_RC="$HOME/.profile"
fi

OLLAMA_ENVS=(
    "OLLAMA_FLASH_ATTENTION:${FA_VAL}"
    "OLLAMA_KV_CACHE_TYPE:${KV_CACHE_TYPE}"
    "OLLAMA_KEEP_ALIVE:24h"
    "OLLAMA_NUM_PARALLEL:1"
    "OLLAMA_MAX_LOADED_MODELS:1"
)

if [ -n "$SHELL_RC" ]; then
    CHANGED=false
    for entry in "${OLLAMA_ENVS[@]}"; do
        KEY="${entry%%:*}"
        VAL="${entry#*:}"
        if ! grep -q "^export ${KEY}=" "$SHELL_RC" 2>/dev/null; then
            echo "export ${KEY}=${VAL}" >> "$SHELL_RC"
            CHANGED=true
        fi
    done
    if [ "$CHANGED" = true ]; then
        ok "  Ollama env vars added to $SHELL_RC"
    else
        ok "  Ollama env vars already in $SHELL_RC"
    fi
fi

# ============================================================================
# Step 2: Pull base model
# ============================================================================
step "Pulling base model ($BASE_MODEL)"

MODEL_TAG=$(echo "$BASE_MODEL" | sed 's/.*://')
if ollama list 2>/dev/null | grep -q "gemma4.*${MODEL_TAG}"; then
    ok "  Base model already available."
else
    echo "  Downloading ~${MODEL_SIZE_GB}GB model ($QUANT). This may take a while..."
    echo "  (Download is resumable — safe to interrupt and re-run)"
    ollama pull "$BASE_MODEL" || fail "Failed to pull $BASE_MODEL"
    ok "  Base model downloaded."
fi

# ============================================================================
# Step 2b: Pull embedding model
# ============================================================================
step "Pulling embedding model (nomic-embed-text — experience ledger + RAG)"

if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
    ok "  Embedding model already available."
else
    echo "  Downloading nomic-embed-text (~274MB)..."
    ollama pull nomic-embed-text || warn "  Failed to pull nomic-embed-text — RAG will use keyword fallback."
    if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        ok "  Embedding model downloaded."
    fi
fi

# ============================================================================
# Step 3: Create custom model from Modelfile
# ============================================================================
step "Creating optimized model ($CUSTOM_MODEL)"

EFFECTIVE_MODELFILE="$ENGINE_DIR/Modelfile"
if [ ! -f "$EFFECTIVE_MODELFILE" ]; then
    fail "Modelfile not found at $EFFECTIVE_MODELFILE"
fi
if [ "$QUANT" != "q8_0" ]; then
    TEMP_MODELFILE="/tmp/pre-Modelfile-${QUANT}"
    sed "s|FROM gemma4:26b-a4b-it-q8_0|FROM $BASE_MODEL|" "$EFFECTIVE_MODELFILE" > "$TEMP_MODELFILE"
    EFFECTIVE_MODELFILE="$TEMP_MODELFILE"
    ok "  Using modified Modelfile with $QUANT base"
fi

if ollama list 2>/dev/null | grep -q "$CUSTOM_MODEL"; then
    ok "  Custom model already exists."
    if ask_yn "  Recreate from Modelfile? [y/N] " "N"; then
        ollama create "$CUSTOM_MODEL" -f "$EFFECTIVE_MODELFILE" || fail "Failed to create $CUSTOM_MODEL"
        ok "  Custom model recreated."
    fi
else
    ollama create "$CUSTOM_MODEL" -f "$EFFECTIVE_MODELFILE" || fail "Failed to create $CUSTOM_MODEL"
    ok "  Custom model created ($QUANT, dynamic context, optimized batch size)."
fi

# ============================================================================
# Step 4: Node.js
# ============================================================================
step "Checking Node.js"

if command -v node &>/dev/null; then
    NODE_VER=$(node --version)
    NODE_MAJOR=$(echo "$NODE_VER" | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_MAJOR" -ge 18 ]; then
        ok "  Node.js: $NODE_VER"
    else
        warn "  Node.js $NODE_VER is too old — need v18+."
        echo "  Install a newer version via:"
        echo "    nvm install 22"
        echo "    — or —"
        echo "    https://nodejs.org/en/download"
        if [ "$PKG_MGR" = "apt" ]; then
            echo "    — or —"
            echo "    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs"
        fi
        exit 1
    fi
else
    warn "  Node.js not found."
    echo "  Install Node.js (v18+ required):"
    if [ "$PKG_MGR" = "apt" ]; then
        echo "    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -"
        echo "    sudo apt-get install -y nodejs"
    elif [ "$PKG_MGR" = "dnf" ]; then
        echo "    sudo dnf install nodejs"
    elif [ "$PKG_MGR" = "pacman" ]; then
        echo "    sudo pacman -S nodejs npm"
    fi
    echo "    — or —"
    echo "    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash"
    echo "    nvm install 22"
    exit 1
fi

# ============================================================================
# Step 5: Web GUI dependencies
# ============================================================================
step "Setting up Web GUI"

WEB_DIR="$REPO_DIR/web"
if [ -f "$WEB_DIR/package.json" ]; then
    echo "  Installing web dependencies..."
    cd "$WEB_DIR"
    npm install --silent 2>&1 | tail -3
    ok "  Web GUI dependencies installed"
    cd "$REPO_DIR"
else
    warn "  Web GUI not found at $WEB_DIR — skipping."
fi

# ============================================================================
# Step 6: Create ~/.pre/ directories
# ============================================================================
step "Setting up data directories"

PRE_DATA="$HOME/.pre"
mkdir -p "$PRE_DATA"/{sessions,memory/experience,artifacts,rag,workflows,custom_tools}
ok "  ~/.pre/ directory structure created"

# Write context window
echo "$OPTIMAL_CTX" > "$PRE_DATA/context"
ok "  Context window set to ${CTX_HUMAN} (${OPTIMAL_CTX} tokens)"

# ============================================================================
# Step 7: Optional — voice tools
# ============================================================================
step "Optional: Voice Tools"

if ask_yn "  Install voice tools (Whisper STT + espeak-ng TTS + FFmpeg)? [y/N] " "N"; then
    if [ -n "$PKG_INSTALL" ]; then
        echo "  Installing FFmpeg and espeak-ng..."
        $PKG_UPDATE 2>/dev/null || true
        $PKG_INSTALL ffmpeg espeak-ng 2>&1 | tail -3
        ok "  FFmpeg and espeak-ng installed"
    else
        warn "  Unknown package manager — install ffmpeg and espeak-ng manually."
    fi
    echo ""
    echo "  For Whisper STT, install via pip:"
    echo "    pip install openai-whisper"
    echo "  (requires Python 3.9+ and ~1GB disk for the base model)"
else
    ok "  Skipping voice tools."
fi

# ============================================================================
# Step 8: Optional — GNOME PIM integration
# ============================================================================
step "Optional: GNOME Calendar/Contacts/Reminders"

# Check if running a GNOME desktop
GNOME_DETECTED=false
if [ -n "$GNOME_DESKTOP_SESSION_ID" ] || echo "$XDG_CURRENT_DESKTOP" | grep -qi "gnome"; then
    GNOME_DETECTED=true
fi

if [ "$GNOME_DETECTED" = true ]; then
    if command -v gdbus &>/dev/null; then
        ok "  GNOME desktop detected with gdbus — Calendar/Contacts/Reminders will work via Evolution Data Server."
    else
        warn "  GNOME detected but gdbus not found."
        if [ -n "$PKG_INSTALL" ]; then
            if ask_yn "  Install evolution-data-server for calendar/contacts/reminders? [Y/n] " "Y"; then
                $PKG_INSTALL evolution-data-server 2>&1 | tail -3
                ok "  evolution-data-server installed"
            fi
        fi
    fi
else
    ok "  Non-GNOME desktop detected. Calendar/Contacts/Reminders require GNOME Evolution Data Server."
    if [ -n "$PKG_INSTALL" ]; then
        if ask_yn "  Install evolution-data-server anyway? (works if GNOME libs are available) [y/N] " "N"; then
            $PKG_INSTALL evolution-data-server 2>&1 | tail -3
            ok "  evolution-data-server installed"
        else
            ok "  Skipping — PRE will show a helpful message if you try to use these tools."
        fi
    fi
fi

# ============================================================================
# Step 9: Optional — desktop automation deps
# ============================================================================
step "Optional: Desktop Automation (Computer Use)"

if ask_yn "  Install desktop automation tools (xdotool, scrot, xclip, wmctrl)? [y/N] " "N"; then
    if [ -n "$PKG_INSTALL" ]; then
        $PKG_UPDATE 2>/dev/null || true
        if [ "$PKG_MGR" = "apt" ]; then
            $PKG_INSTALL xdotool scrot xclip wmctrl libnotify-bin 2>&1 | tail -3
        elif [ "$PKG_MGR" = "dnf" ]; then
            $PKG_INSTALL xdotool scrot xclip wmctrl libnotify 2>&1 | tail -3
        elif [ "$PKG_MGR" = "pacman" ]; then
            $PKG_INSTALL xdotool scrot xclip wmctrl libnotify 2>&1 | tail -3
        fi
        ok "  Desktop automation tools installed (X11)"
    else
        warn "  Unknown package manager. Install manually: xdotool scrot xclip wmctrl"
    fi
else
    ok "  Skipping desktop automation tools."
fi

# ============================================================================
# Step 10: Optional — auto-start (systemd user service)
# ============================================================================
step "Optional: Auto-start at Login"

if ask_yn "  Set up PRE web server to start automatically at login? [y/N] " "N"; then
    SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
    mkdir -p "$SYSTEMD_USER_DIR"

    cat > "$SYSTEMD_USER_DIR/pre-server.service" <<SVCEOF
[Unit]
Description=PRE Personal Reasoning Engine — Web GUI
After=network.target

[Service]
Type=simple
WorkingDirectory=$WEB_DIR
ExecStart=$(command -v node) $WEB_DIR/server.js
Environment=PRE_WEB_PORT=7749
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
SVCEOF

    systemctl --user daemon-reload
    systemctl --user enable pre-server
    ok "  systemd user service created and enabled"
    ok "  PRE will auto-start at login on port 7749"
    echo -e "  ${DIM}Manage: systemctl --user start|stop|status pre-server${RESET}"
else
    ok "  Skipping auto-start."
fi

# ============================================================================
# Step 11: Pre-warm model
# ============================================================================
step "Pre-warming model into GPU memory"

if ollama ps 2>/dev/null | grep -q "$CUSTOM_MODEL"; then
    ok "  Model already loaded."
else
    echo "  Loading model into GPU (context: ${OPTIMAL_CTX} tokens)..."
    curl -sf --max-time 300 "http://localhost:${PORT}/api/chat" \
        -d "{\"model\":\"${CUSTOM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"keep_alive\":\"24h\",\"options\":{\"num_predict\":1,\"num_ctx\":${OPTIMAL_CTX}}}" \
        >/dev/null 2>&1 || true
    if ollama ps 2>/dev/null | grep -q "$CUSTOM_MODEL"; then
        ok "  Model loaded and warm."
    else
        warn "  Model may not have fully loaded. First request may be slow."
    fi
fi

# ============================================================================
# Step 12: Create launcher script
# ============================================================================
step "Setting up launcher"

LAUNCHER_DIR="$HOME/.local/bin"
mkdir -p "$LAUNCHER_DIR"
LAUNCHER="$LAUNCHER_DIR/pre-web"

cat > "$LAUNCHER" <<LAUNCHEOF
#!/bin/bash
# pre-web — Launch PRE Web GUI
cd "$WEB_DIR" && exec node server.js
LAUNCHEOF
chmod +x "$LAUNCHER"
ok "  Launcher created at $LAUNCHER"

# Add ~/.local/bin to PATH if not already there
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    if [ -n "$SHELL_RC" ]; then
        if ! grep -q '\.local/bin' "$SHELL_RC" 2>/dev/null; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
            ok "  Added ~/.local/bin to PATH in $SHELL_RC"
        fi
    fi
fi

# ============================================================================
# Step 13: Create XDG desktop entry
# ============================================================================
DESKTOP_DIR="$HOME/.local/share/applications"
mkdir -p "$DESKTOP_DIR"

# Check for a logo
ICON_PATH="$WEB_DIR/public/img/logo.png"
if [ ! -f "$ICON_PATH" ]; then
    ICON_PATH=""
fi

cat > "$DESKTOP_DIR/pre.desktop" <<DESKTOPEOF
[Desktop Entry]
Type=Application
Name=PRE - Personal Reasoning Engine
Comment=Local AI assistant with 70+ tools
Exec=bash -c 'cd $WEB_DIR && node server.js & sleep 2 && xdg-open http://localhost:7749'
${ICON_PATH:+Icon=$ICON_PATH}
Terminal=false
Categories=Development;AI;Science;
StartupNotify=true
DESKTOPEOF
chmod +x "$DESKTOP_DIR/pre.desktop"
ok "  Desktop entry created (find PRE in your app launcher)"

# ============================================================================
# Done!
# ============================================================================
echo ""
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  PRE is installed and ready!${RESET}"
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${BOLD}Start PRE:${RESET}"
echo -e "    cd $WEB_DIR && node server.js"
echo -e "    Then open: ${CYAN}http://localhost:7749${RESET}"
echo ""
echo -e "  ${BOLD}Quick launch:${RESET}"
echo -e "    pre-web      ${DIM}(if ~/.local/bin is in PATH)${RESET}"
echo ""
echo -e "  ${BOLD}Hardware profile:${RESET}"
echo -e "    GPU: ${GPU_NAME:-CPU only}  |  VRAM: ${GPU_VRAM_GB}GB  |  RAM: ${RAM_GB}GB"
echo -e "    Model: ${QUANT}  |  Context: ${CTX_HUMAN}  |  Backend: ${GPU_BACKEND:-cpu}"
echo ""
if [ -n "$GPU_BACKEND" ]; then
    echo -e "  ${DIM}Expect ~30-70+ tok/s with GPU acceleration.${RESET}"
else
    echo -e "  ${YELLOW}No GPU detected — expect ~3-6 tok/s (CPU mode).${RESET}"
    echo -e "  ${DIM}Install NVIDIA drivers + CUDA toolkit for GPU acceleration.${RESET}"
fi
echo ""
