#!/bin/bash
# install.sh — Full setup for PRE (Personal Reasoning Engine)
#
# This script handles everything from Ollama to ready-to-run:
#   1. Checks system requirements (Apple Silicon, macOS, RAM, disk)
#   2. Installs/verifies Ollama
#   3. Pulls the base model (gemma4:26b-a4b-it-q4_K_M, ~17GB)
#   4. Creates optimized custom model (pre-gemma4) from Modelfile
#   5. Installs Xcode CLI tools if needed
#   6. Compiles PRE CLI binary
#   7. Installs pre-launch command to ~/.local/bin
#   8. Sets up ~/.pre/ directories
#
# Run time: 5-20 minutes depending on internet speed.
# Disk space required: ~17GB for model + negligible for binaries.

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
BASE_MODEL="gemma4:26b-a4b-it-q4_K_M"
CUSTOM_MODEL="pre-gemma4"

step() { echo -e "\n${BOLD}${CYAN}=== $1 ===${RESET}\n"; }
ok()   { echo -e "${GREEN}$1${RESET}"; }
warn() { echo -e "${YELLOW}$1${RESET}"; }
fail() { echo -e "${RED}$1${RESET}"; exit 1; }

# ============================================================================
# Step 0: System requirements
# ============================================================================
step "Checking system requirements"

# Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    fail "PRE requires Apple Silicon (arm64). Detected: $ARCH"
fi
ok "  Apple Silicon: $ARCH"

# macOS
OS=$(uname -s)
if [ "$OS" != "Darwin" ]; then
    fail "PRE requires macOS. Detected: $OS"
fi
MACOS_VER=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo "$MACOS_VER" | cut -d. -f1)
if [ "$MACOS_MAJOR" -lt 14 ]; then
    fail "PRE requires macOS 14 (Sonoma) or later. Detected: $MACOS_VER"
fi
ok "  macOS: $MACOS_VER"

# RAM
RAM_BYTES=$(sysctl -n hw.memsize)
RAM_GB=$((RAM_BYTES / 1073741824))
if [ "$RAM_GB" -lt 16 ]; then
    fail "PRE requires at least 16GB unified memory. Detected: ${RAM_GB}GB"
fi
if [ "$RAM_GB" -lt 32 ]; then
    warn "  RAM: ${RAM_GB}GB — functional, but 32GB+ recommended for full 262K context."
else
    ok "  RAM: ${RAM_GB}GB unified memory"
fi

# Disk space (~17GB for model, ~1GB headroom)
AVAIL_KB=$(df -k "$REPO_DIR" | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1048576))
if [ "$AVAIL_GB" -lt 20 ]; then
    warn "  Disk: ${AVAIL_GB}GB available — need ~17GB for model download."
    echo -n "  Continue anyway? [y/N] "
    read -r ans
    if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then exit 1; fi
else
    ok "  Disk: ${AVAIL_GB}GB available"
fi

# GPU info
GPU_INFO=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | sed 's/.*: //')
if [ -n "$GPU_INFO" ]; then
    ok "  GPU: $GPU_INFO (Metal)"
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
    if command -v brew &>/dev/null; then
        echo -n "  Install Ollama via Homebrew? [Y/n] "
        read -r ans
        if [ "$ans" = "n" ] || [ "$ans" = "N" ]; then
            echo ""
            echo "  Install Ollama manually:"
            echo "    brew install ollama"
            echo "    — or —"
            echo "    Download from https://ollama.com/download"
            exit 1
        fi
        echo "  Installing Ollama..."
        brew install ollama
        ok "  Ollama installed."
    else
        echo ""
        echo "  Install Ollama:"
        echo "    Download from https://ollama.com/download"
        echo "    — or —"
        echo "    Install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "    Then: brew install ollama"
        exit 1
    fi
fi

# Ensure Ollama is running
PORT="${PRE_PORT:-11434}"
if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "  Starting Ollama..."
    ollama serve >/dev/null 2>&1 &
    sleep 3
    if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        warn "  Could not start Ollama automatically."
        echo "  Please start Ollama.app or run 'ollama serve' in another terminal, then re-run this script."
        exit 1
    fi
fi
ok "  Ollama running on port $PORT."

# ============================================================================
# Step 2: Pull base model
# ============================================================================
step "Pulling base model ($BASE_MODEL)"

if ollama list 2>/dev/null | grep -q "gemma4.*26b-a4b-it-q4_K_M"; then
    ok "  Base model already available."
else
    echo "  Downloading ~17GB model. This may take a while..."
    echo "  (Download is resumable — safe to interrupt and re-run)"
    ollama pull "$BASE_MODEL" || fail "Failed to pull $BASE_MODEL"
    ok "  Base model downloaded."
fi

# ============================================================================
# Step 2b: Pull embedding model (for experience ledger semantic search)
# ============================================================================
step "Pulling embedding model (nomic-embed-text)"

if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
    ok "  Embedding model already available."
else
    echo "  Downloading nomic-embed-text (~274MB)..."
    ollama pull nomic-embed-text || warn "  Failed to pull nomic-embed-text — experience ledger will use keyword search fallback."
    if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        ok "  Embedding model downloaded."
    fi
fi

# ============================================================================
# Step 3: Create custom model from Modelfile
# ============================================================================
step "Creating optimized model ($CUSTOM_MODEL)"

if ollama list 2>/dev/null | grep -q "$CUSTOM_MODEL"; then
    ok "  Custom model already exists."
    echo -n "  Recreate from Modelfile? [y/N] "
    read -r ans
    if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
        ollama create "$CUSTOM_MODEL" -f "$ENGINE_DIR/Modelfile" || fail "Failed to create $CUSTOM_MODEL"
        ok "  Custom model recreated."
    fi
else
    if [ ! -f "$ENGINE_DIR/Modelfile" ]; then
        fail "Modelfile not found at $ENGINE_DIR/Modelfile"
    fi
    ollama create "$CUSTOM_MODEL" -f "$ENGINE_DIR/Modelfile" || fail "Failed to create $CUSTOM_MODEL"
    ok "  Custom model created (dynamic context, optimized batch size)."
fi

# ============================================================================
# Step 4: Xcode Command Line Tools
# ============================================================================
step "Checking build tools"

if ! command -v clang &>/dev/null; then
    warn "  Clang not found. Installing Xcode Command Line Tools..."
    xcode-select --install 2>/dev/null || true
    echo "  Please complete the Xcode installation and re-run this script."
    exit 1
fi
ok "  Compiler: $(clang --version | head -1)"

# Check for Metal framework (should always be present on Apple Silicon macOS)
if ! xcrun --sdk macosx --show-sdk-path &>/dev/null; then
    warn "  macOS SDK not found. You may need to run: xcode-select --install"
fi

# ============================================================================
# Step 5: Build PRE
# ============================================================================
step "Building PRE"

cd "$ENGINE_DIR"
make clean 2>/dev/null || true
make pre telegram 2>&1 | tail -5
if [ ! -x "$ENGINE_DIR/pre" ]; then
    fail "Build failed — 'pre' binary not found."
fi
ok "  Built: pre ($(du -sh pre | cut -f1))"
if [ -x "$ENGINE_DIR/pre-telegram" ]; then
    ok "  Built: pre-telegram ($(du -sh pre-telegram | cut -f1))"
fi

# ============================================================================
# Step 6: Web GUI dependencies
# ============================================================================
step "Setting up Web GUI"

WEB_DIR="$REPO_DIR/web"
if [ -f "$WEB_DIR/package.json" ]; then
    if command -v node &>/dev/null; then
        NODE_VER=$(node --version)
        NODE_MAJOR=$(echo "$NODE_VER" | sed 's/v//' | cut -d. -f1)
        if [ "$NODE_MAJOR" -ge 18 ]; then
            ok "  Node.js: $NODE_VER"
            echo "  Installing web dependencies..."
            cd "$WEB_DIR"
            npm install --silent 2>&1 | tail -3
            ok "  Web GUI dependencies installed (express, ws, docx, exceljs, pdfkit)"

            # Install terminal-notifier for clickable cron notifications
            if ! command -v terminal-notifier &>/dev/null; then
                if command -v brew &>/dev/null; then
                    echo "  Installing terminal-notifier (clickable cron notifications)..."
                    brew install terminal-notifier --quiet 2>&1 | tail -1
                    ok "  terminal-notifier installed"
                else
                    warn "  terminal-notifier not found — cron notifications will use basic osascript."
                    echo -e "  ${DIM}Install later with: brew install terminal-notifier${RESET}"
                fi
            else
                ok "  terminal-notifier available (clickable cron notifications)"
            fi
            cd "$REPO_DIR"
        else
            warn "  Node.js $NODE_VER is too old — need v18+. Web GUI will not be available."
            warn "  Install a newer version: brew install node"
        fi
    else
        warn "  Node.js not found — Web GUI will not be available."
        warn "  Install Node.js: brew install node"
        echo -e "  ${DIM}The CLI works without Node.js. Web GUI is optional.${RESET}"
    fi
else
    warn "  Web GUI not found at $WEB_DIR — skipping."
fi

# ============================================================================
# Step 7: Install pre-launch command
# ============================================================================
step "Installing pre-launch command"

chmod +x "$ENGINE_DIR/pre-launch"
make install 2>&1

# Ensure ~/.local/bin is in PATH
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    SHELL_RC=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_RC="$HOME/.bash_profile"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    fi
    if [ -n "$SHELL_RC" ]; then
        if ! grep -q '.local/bin' "$SHELL_RC" 2>/dev/null; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
            ok "  Added ~/.local/bin to PATH in $SHELL_RC"
            warn "  Run 'source $SHELL_RC' or open a new terminal to use pre-launch."
        fi
    else
        warn "  Add to your shell profile: export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
fi

# ============================================================================
# Step 8: Set up ~/.pre/ directories
# ============================================================================
step "Setting up PRE data directories"

mkdir -p "$HOME/.pre/sessions"
mkdir -p "$HOME/.pre/memory"
mkdir -p "$HOME/.pre/memory/experience"
mkdir -p "$HOME/.pre/checkpoints"
mkdir -p "$HOME/.pre/artifacts"
ok "  Created ~/.pre/sessions/"
ok "  Created ~/.pre/memory/"
ok "  Created ~/.pre/memory/experience/"
ok "  Created ~/.pre/checkpoints/"
ok "  Created ~/.pre/artifacts/"

# Create default config files if they don't exist
if [ ! -f "$HOME/.pre/hooks.json" ]; then
    echo '{"hooks":[]}' > "$HOME/.pre/hooks.json"
    ok "  Created ~/.pre/hooks.json"
fi
if [ ! -f "$HOME/.pre/mcp.json" ]; then
    echo '{"servers":{}}' > "$HOME/.pre/mcp.json"
    ok "  Created ~/.pre/mcp.json"
fi

# Migrate from old Flash-MoE layout if present
if [ -d "$HOME/.flash-moe" ] && [ ! -f "$HOME/.pre/.migrated" ]; then
    if [ -f "$HOME/.flash-moe/pre_history" ]; then
        cp "$HOME/.flash-moe/pre_history" "$HOME/.pre/pre_history" 2>/dev/null || true
        ok "  Migrated command history from ~/.flash-moe/"
    fi
    touch "$HOME/.pre/.migrated"
fi

# ============================================================================
# Step 8b: ComfyUI setup (optional — local image generation)
# ============================================================================
step "Image Generation Setup (optional)"

COMFYUI_DIR="$HOME/.pre/comfyui"
COMFYUI_VENV="$HOME/.pre/comfyui-venv"
COMFYUI_CONFIG="$HOME/.pre/comfyui.json"

# Checkpoint options
TURBO_CHECKPOINT="sd_xl_turbo_1.0_fp16.safetensors"
TURBO_URL="https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors"
QUALITY_CHECKPOINT="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
QUALITY_REPO="RunDiffusion/Juggernaut-XL-v9"

if [ -f "$COMFYUI_CONFIG" ]; then
    ok "  ComfyUI already configured — skipping."
else
    # Check available RAM
    RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1073741824}')
    if [ "$RAM_GB" -lt 48 ] 2>/dev/null; then
        warn "  Note: Your system has ${RAM_GB}GB RAM. Image generation (SDXL + Gemma 4) works"
        warn "  best with 48GB+. You can skip this and add it later with /connections."
    fi

    echo ""
    echo -e "  ${BOLD}Install local image generation?${RESET}"
    echo -e "  This sets up ComfyUI + Stable Diffusion XL for generating images from text."
    echo -e "  ${DIM}Requires: ~8GB disk, Python 3.10-3.13 (PyTorch), ~6.5GB model download${RESET}"
    echo ""
    read -p "  Install ComfyUI? [y/N] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Find a compatible Python (3.10-3.13; 3.14+ lacks PyTorch support)
        COMFYUI_PYTHON=""
        for pyver in python3.12 python3.13 python3.11 python3.10; do
            if command -v "$pyver" &>/dev/null; then
                COMFYUI_PYTHON="$pyver"
                break
            fi
        done
        # Fallback to python3 if it's in the supported range
        if [ -z "$COMFYUI_PYTHON" ] && command -v python3 &>/dev/null; then
            PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
            if [ "$PY_MINOR" -ge 10 ] && [ "$PY_MINOR" -le 13 ]; then
                COMFYUI_PYTHON="python3"
            fi
        fi

        if [ -z "$COMFYUI_PYTHON" ]; then
            warn "  Python 3.10-3.13 not found — skipping ComfyUI. PyTorch requires Python <=3.13."
            warn "  Install Python 3.12 (brew install python@3.12) and re-run."
        else
            PYTHON_VERSION=$($COMFYUI_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            ok "  Python $PYTHON_VERSION found ($COMFYUI_PYTHON)"

            # Create virtual environment
            echo "  Creating Python virtual environment..."
            $COMFYUI_PYTHON -m venv "$COMFYUI_VENV" || { warn "  Failed to create venv — skipping."; REPLY=n; }

            if [[ $REPLY =~ ^[Yy]$ ]]; then
                # Install PyTorch with MPS support
                echo "  Installing PyTorch (Metal/MPS)..."
                "$COMFYUI_VENV/bin/pip" install --quiet torch torchvision torchaudio 2>&1 | tail -1
                ok "  PyTorch installed with MPS support"

                # Clone ComfyUI
                if [ ! -d "$COMFYUI_DIR" ]; then
                    echo "  Cloning ComfyUI..."
                    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFYUI_DIR" 2>&1 | tail -1
                fi
                ok "  ComfyUI cloned"

                # Install ComfyUI requirements
                echo "  Installing ComfyUI dependencies..."
                "$COMFYUI_VENV/bin/pip" install --quiet -r "$COMFYUI_DIR/requirements.txt" 2>&1 | tail -1
                ok "  Dependencies installed"

                # Choose checkpoint: Quality (Juggernaut XL) or Speed (SDXL Turbo)
                CKPT_DIR="$COMFYUI_DIR/models/checkpoints"
                mkdir -p "$CKPT_DIR"

                echo ""
                echo -e "  ${BOLD}Choose image model:${RESET}"
                echo -e "  ${CYAN}1)${RESET} ${BOLD}Juggernaut XL${RESET} (recommended) — Photorealistic, excellent faces"
                echo -e "     ~30-45s per image, 25 steps at 1024x1024"
                echo -e "  ${CYAN}2)${RESET} ${BOLD}SDXL Turbo${RESET} — Fast but lower quality"
                echo -e "     ~5s per image, 4 steps at 512x512"
                echo ""
                read -p "  Choice [1/2, default=1]: " -n 1 -r IMG_CHOICE
                echo ""

                SELECTED_CHECKPOINT=""
                if [ "$IMG_CHOICE" = "2" ]; then
                    # SDXL Turbo (speed)
                    if [ -f "$CKPT_DIR/$TURBO_CHECKPOINT" ]; then
                        ok "  SDXL Turbo checkpoint already downloaded"
                    else
                        echo "  Downloading SDXL Turbo (~6.5GB)..."
                        if command -v wget &>/dev/null; then
                            wget -q --show-progress -O "$CKPT_DIR/$TURBO_CHECKPOINT" "$TURBO_URL"
                        else
                            curl -L --progress-bar -o "$CKPT_DIR/$TURBO_CHECKPOINT" "$TURBO_URL"
                        fi
                    fi
                    if [ -f "$CKPT_DIR/$TURBO_CHECKPOINT" ]; then
                        CKPT_SIZE=$(du -sh "$CKPT_DIR/$TURBO_CHECKPOINT" | cut -f1)
                        ok "  SDXL Turbo downloaded ($CKPT_SIZE)"
                        SELECTED_CHECKPOINT="$TURBO_CHECKPOINT"
                    else
                        warn "  Download failed — you can download manually later."
                    fi
                else
                    # Juggernaut XL (quality — default)
                    if [ -f "$CKPT_DIR/$QUALITY_CHECKPOINT" ]; then
                        ok "  Juggernaut XL checkpoint already downloaded"
                        SELECTED_CHECKPOINT="$QUALITY_CHECKPOINT"
                    else
                        echo "  Downloading Juggernaut XL v9 (~6.6GB)..."
                        if command -v huggingface-cli &>/dev/null; then
                            huggingface-cli download "$QUALITY_REPO" "$QUALITY_CHECKPOINT" \
                                --local-dir "$CKPT_DIR" 2>&1 | tail -3
                            # Clean up HF cache
                            rm -rf "$CKPT_DIR/.cache" 2>/dev/null
                        elif command -v wget &>/dev/null; then
                            wget -q --show-progress -O "$CKPT_DIR/$QUALITY_CHECKPOINT" \
                                "https://huggingface.co/$QUALITY_REPO/resolve/main/$QUALITY_CHECKPOINT"
                        else
                            curl -L --progress-bar -o "$CKPT_DIR/$QUALITY_CHECKPOINT" \
                                "https://huggingface.co/$QUALITY_REPO/resolve/main/$QUALITY_CHECKPOINT"
                        fi
                    fi
                    if [ -f "$CKPT_DIR/$QUALITY_CHECKPOINT" ]; then
                        CKPT_SIZE=$(du -sh "$CKPT_DIR/$QUALITY_CHECKPOINT" | cut -f1)
                        ok "  Juggernaut XL downloaded ($CKPT_SIZE)"
                        SELECTED_CHECKPOINT="$QUALITY_CHECKPOINT"
                    else
                        warn "  Download failed — falling back to SDXL Turbo..."
                        echo "  Downloading SDXL Turbo (~6.5GB)..."
                        if command -v wget &>/dev/null; then
                            wget -q --show-progress -O "$CKPT_DIR/$TURBO_CHECKPOINT" "$TURBO_URL"
                        else
                            curl -L --progress-bar -o "$CKPT_DIR/$TURBO_CHECKPOINT" "$TURBO_URL"
                        fi
                        if [ -f "$CKPT_DIR/$TURBO_CHECKPOINT" ]; then
                            SELECTED_CHECKPOINT="$TURBO_CHECKPOINT"
                            ok "  SDXL Turbo downloaded as fallback"
                        else
                            warn "  Both downloads failed. You can download manually later."
                        fi
                    fi
                fi

                if [ -n "$SELECTED_CHECKPOINT" ]; then

                    # Create config file
                    cat > "$COMFYUI_CONFIG" << CFGEOF
{
  "installed": true,
  "port": 8188,
  "checkpoint": "$SELECTED_CHECKPOINT",
  "venv": "~/.pre/comfyui-venv",
  "comfyui_dir": "~/.pre/comfyui"
}
CFGEOF
                    ok "  ComfyUI configuration saved (using $SELECTED_CHECKPOINT)"
                    echo ""
                    ok "  Image generation ready! PRE will start ComfyUI automatically when needed."
                    echo -e "  ${DIM}Tip: To switch models later, edit ~/.pre/comfyui.json${RESET}"
                fi
            fi
        fi
    else
        echo "  Skipped. You can install later by re-running this script."
    fi
fi

# ============================================================================
# Step 9: Pre-warm the model
# ============================================================================
step "Pre-warming model into GPU memory"

echo "  Loading $CUSTOM_MODEL into GPU (this takes 10-30 seconds)..."
curl -sf "http://localhost:${PORT}/api/chat" -d "{\"model\":\"$CUSTOM_MODEL\",\"messages\":[],\"keep_alive\":\"24h\"}" >/dev/null 2>&1 || true

for i in $(seq 1 30); do
    if ollama ps 2>/dev/null | grep -q "$CUSTOM_MODEL"; then
        ok "  Model loaded into GPU memory."
        break
    fi
    sleep 1
done

if ! ollama ps 2>/dev/null | grep -q "$CUSTOM_MODEL"; then
    warn "  Model may not be fully loaded yet — it will load on first launch."
fi

# ============================================================================
# Done!
# ============================================================================
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║${RESET}${BOLD}  PRE installation complete!                              ${GREEN}║${RESET}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}Launch:${RESET}"
echo -e "    ${CYAN}pre-launch${RESET}                  Start PRE (CLI + Web GUI)"
echo -e "    ${CYAN}pre-launch --show-think${RESET}     Start with visible reasoning"
echo -e "    Web GUI auto-starts at ${CYAN}http://localhost:7749${RESET}"
echo ""
echo -e "  ${BOLD}First Launch:${RESET}"
echo -e "    PRE will ask you to name your agent and optionally"
echo -e "    configure connections (Google, GitHub, Brave, Wolfram)."
echo ""
echo -e "  ${BOLD}Connections (optional):${RESET}"
echo -e "    Type ${BOLD}/connections${RESET} inside PRE or use the Web GUI Settings to set up:"
echo -e "    • ${DIM}Google        — Gmail, Drive, Docs (built-in OAuth)${RESET}"
echo -e "    • ${DIM}Telegram      — chat from your phone + cron delivery${RESET}"
echo -e "    • ${DIM}Slack         — channel messaging${RESET}"
echo -e "    • ${DIM}Brave Search  — web search${RESET}"
echo -e "    • ${DIM}GitHub        — repos, issues, PRs${RESET}"
echo -e "    • ${DIM}Jira          — issues, projects, transitions${RESET}"
echo -e "    • ${DIM}Confluence    — wiki pages, search${RESET}"
echo -e "    • ${DIM}Smartsheet    — spreadsheets, rows, search${RESET}"
echo -e "    • ${DIM}Wolfram Alpha — computation${RESET}"
if [ -f "$HOME/.pre/comfyui.json" ]; then
echo -e "    • ${GREEN}ComfyUI       — image generation (installed ✓)${RESET}"
else
echo -e "    • ${DIM}ComfyUI       — image generation (re-run install.sh to add)${RESET}"
fi
echo ""
echo -e "  ${BOLD}Features:${RESET}"
echo -e "    50+ tools including sub-agents, browser automation, hooks,"
echo -e "    experience ledger (learns from past tasks), and temporal"
echo -e "    memory awareness. Chrome enables headless browser control."
echo ""
echo -e "  ${BOLD}Data:${RESET}"
echo -e "    ${DIM}~/.pre/identity.json${RESET}      Agent name"
echo -e "    ${DIM}~/.pre/sessions/${RESET}          Session history (shared by CLI + Web)"
echo -e "    ${DIM}~/.pre/memory/${RESET}            Persistent memory"
echo -e "    ${DIM}~/.pre/memory/experience/${RESET} Experience ledger (lessons learned)"
echo -e "    ${DIM}~/.pre/artifacts/${RESET}         Generated documents, images & artifacts"
echo -e "    ${DIM}~/.pre/cron.json${RESET}          Scheduled recurring tasks"
echo -e "    ${DIM}~/.pre/hooks.json${RESET}         Pre/post tool execution hooks"
echo -e "    ${DIM}~/.pre/mcp.json${RESET}           MCP server configuration"
echo -e "    ${DIM}~/.pre/telegram.log${RESET}       Telegram bot log"
echo ""
echo -e "  Type ${BOLD}/help${RESET} inside PRE for all commands."
echo ""
