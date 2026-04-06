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
# Step 6: Install pre-launch command
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
# Step 7: Set up ~/.pre/ directories
# ============================================================================
step "Setting up PRE data directories"

mkdir -p "$HOME/.pre/sessions"
mkdir -p "$HOME/.pre/memory"
mkdir -p "$HOME/.pre/checkpoints"
ok "  Created ~/.pre/sessions/"
ok "  Created ~/.pre/memory/"
ok "  Created ~/.pre/checkpoints/"

# Migrate from old Flash-MoE layout if present
if [ -d "$HOME/.flash-moe" ] && [ ! -f "$HOME/.pre/.migrated" ]; then
    if [ -f "$HOME/.flash-moe/pre_history" ]; then
        cp "$HOME/.flash-moe/pre_history" "$HOME/.pre/pre_history" 2>/dev/null || true
        ok "  Migrated command history from ~/.flash-moe/"
    fi
    touch "$HOME/.pre/.migrated"
fi

# ============================================================================
# Step 7b: ComfyUI setup (optional — local image generation)
# ============================================================================
step "Image Generation Setup (optional)"

COMFYUI_DIR="$HOME/.pre/comfyui"
COMFYUI_VENV="$HOME/.pre/comfyui-venv"
COMFYUI_CONFIG="$HOME/.pre/comfyui.json"
SDXL_CHECKPOINT="sd_xl_turbo_1.0_fp16.safetensors"
SDXL_URL="https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors"

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
    echo -e "  This sets up ComfyUI + SDXL Turbo for generating images from text."
    echo -e "  ${DIM}Requires: ~8GB disk, Python 3.10+, ~6.5GB model download${RESET}"
    echo ""
    read -p "  Install ComfyUI? [y/N] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Check Python
        if ! command -v python3 &>/dev/null; then
            warn "  Python 3 not found — skipping ComfyUI. Install Python 3.10+ and re-run."
        else
            PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            ok "  Python $PYTHON_VERSION found"

            # Create virtual environment
            echo "  Creating Python virtual environment..."
            python3 -m venv "$COMFYUI_VENV" || { warn "  Failed to create venv — skipping."; REPLY=n; }

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

                # Download SDXL Turbo checkpoint
                CKPT_DIR="$COMFYUI_DIR/models/checkpoints"
                mkdir -p "$CKPT_DIR"
                if [ -f "$CKPT_DIR/$SDXL_CHECKPOINT" ]; then
                    ok "  SDXL Turbo checkpoint already downloaded"
                else
                    echo "  Downloading SDXL Turbo (~6.5GB — this may take a while)..."
                    if command -v wget &>/dev/null; then
                        wget -q --show-progress -O "$CKPT_DIR/$SDXL_CHECKPOINT" "$SDXL_URL"
                    else
                        curl -L --progress-bar -o "$CKPT_DIR/$SDXL_CHECKPOINT" "$SDXL_URL"
                    fi

                    if [ -f "$CKPT_DIR/$SDXL_CHECKPOINT" ]; then
                        CKPT_SIZE=$(du -sh "$CKPT_DIR/$SDXL_CHECKPOINT" | cut -f1)
                        ok "  SDXL Turbo downloaded ($CKPT_SIZE)"
                    else
                        warn "  Download failed — you can download manually later:"
                        warn "    wget -O $CKPT_DIR/$SDXL_CHECKPOINT $SDXL_URL"
                    fi
                fi

                # Create config file
                cat > "$COMFYUI_CONFIG" << 'CFGEOF'
{
  "installed": true,
  "port": 8188,
  "checkpoint": "sd_xl_turbo_1.0_fp16.safetensors"
}
CFGEOF
                ok "  ComfyUI configuration saved"
                echo ""
                ok "  ✓ Image generation ready! PRE will start ComfyUI automatically when needed."
            fi
        fi
    else
        echo "  Skipped. You can install later by re-running this script."
    fi
fi

# ============================================================================
# Step 8: Pre-warm the model
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
echo -e "    ${CYAN}pre-launch${RESET}                  Start PRE from any directory"
echo -e "    ${CYAN}pre-launch --show-think${RESET}     Start with visible reasoning"
echo ""
echo -e "  ${BOLD}First Launch:${RESET}"
echo -e "    PRE will ask you to name your agent and optionally"
echo -e "    configure connections (Google, GitHub, Brave, Wolfram)."
echo ""
echo -e "  ${BOLD}Connections (optional):${RESET}"
echo -e "    Type ${BOLD}/connections${RESET} inside PRE to set up:"
echo -e "    • ${DIM}Google        — Gmail, Drive, Docs (built-in OAuth)${RESET}"
echo -e "    • ${DIM}Telegram      — chat from your phone${RESET}"
echo -e "    • ${DIM}Brave Search  — web search${RESET}"
echo -e "    • ${DIM}GitHub        — repos, issues, PRs${RESET}"
echo -e "    • ${DIM}Wolfram Alpha — computation${RESET}"
if [ -f "$HOME/.pre/comfyui.json" ]; then
echo -e "    • ${GREEN}ComfyUI       — image generation (installed ✓)${RESET}"
else
echo -e "    • ${DIM}ComfyUI       — image generation (re-run install.sh to add)${RESET}"
fi
echo ""
echo -e "  ${BOLD}Data:${RESET}"
echo -e "    ${DIM}~/.pre/identity.json${RESET}  Agent name"
echo -e "    ${DIM}~/.pre/sessions/${RESET}      Session history"
echo -e "    ${DIM}~/.pre/memory/${RESET}        Persistent memory"
echo -e "    ${DIM}~/.pre/telegram.log${RESET}   Telegram bot log"
echo ""
echo -e "  Type ${BOLD}/help${RESET} inside PRE for all commands."
echo ""
