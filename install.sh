#!/bin/bash
# install.sh — Full setup for PRE (Personal Reasoning Engine)
#
# This script handles everything from model download to ready-to-run:
#   1. Checks system requirements (Apple Silicon, RAM, disk space)
#   2. Installs Python dependencies (numpy, safetensors, huggingface-hub)
#   3. Downloads Qwen3.5-397B-A17B-4bit from HuggingFace (~214GB)
#   4. Extracts non-expert weights → engine/model_weights.bin (5.5GB)
#   5. Exports tokenizer → engine/tokenizer.bin + engine/vocab.bin
#   6. Repacks expert weights → packed_experts/ (209GB)
#   7. Compiles inference engine and PRE CLI
#   8. Installs pre-launch command
#   9. Sets up system prompt
#
# Run time: 30-90 minutes depending on internet speed and SSD performance.
# Disk space required: ~430GB (214GB download cache + 214GB packed weights).

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

step() { echo -e "\n${BOLD}${CYAN}=== $1 ===${RESET}\n"; }
ok()   { echo -e "${GREEN}$1${RESET}"; }
warn() { echo -e "${YELLOW}$1${RESET}"; }
fail() { echo -e "${RED}$1${RESET}"; exit 1; }

# ============================================================================
# Step 0: System requirements check
# ============================================================================
step "Checking system requirements"

# Apple Silicon check
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    fail "PRE requires Apple Silicon (arm64). Detected: $ARCH"
fi
ok "  Apple Silicon: $ARCH"

# macOS check
OS=$(uname -s)
if [ "$OS" != "Darwin" ]; then
    fail "PRE requires macOS. Detected: $OS"
fi
MACOS_VER=$(sw_vers -productVersion)
ok "  macOS: $MACOS_VER"

# RAM check
RAM_BYTES=$(sysctl -n hw.memsize)
RAM_GB=$((RAM_BYTES / 1073741824))
if [ "$RAM_GB" -lt 36 ]; then
    fail "PRE requires at least 48GB unified memory. Detected: ${RAM_GB}GB"
fi
if [ "$RAM_GB" -lt 48 ]; then
    warn "  RAM: ${RAM_GB}GB — minimum viable. 48GB+ recommended for best performance."
else
    ok "  RAM: ${RAM_GB}GB unified memory"
fi

# Disk space check (need ~430GB total)
AVAIL_KB=$(df -k "$REPO_DIR" | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1048576))
if [ "$AVAIL_GB" -lt 250 ]; then
    warn "  Disk: ${AVAIL_GB}GB available — you may need ~430GB total."
    warn "  If the model is already downloaded, ~215GB is sufficient."
    echo -n "  Continue anyway? [y/N] "
    read -r ans
    if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then exit 1; fi
else
    ok "  Disk: ${AVAIL_GB}GB available"
fi

# Xcode / clang check
if ! command -v clang &>/dev/null; then
    warn "  Clang not found. Installing Xcode Command Line Tools..."
    xcode-select --install 2>/dev/null || true
    echo "  Please complete the Xcode installation and re-run this script."
    exit 1
fi
ok "  Compiler: $(clang --version | head -1)"

# Python check
if ! command -v python3 &>/dev/null; then
    fail "Python 3 is required. Install via: brew install python3"
fi
ok "  Python: $(python3 --version)"

# GPU check
GPU_INFO=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | sed 's/.*: //')
if [ -n "$GPU_INFO" ]; then
    ok "  GPU: $GPU_INFO (Metal)"
fi

echo ""
ok "System requirements met."

# ============================================================================
# Step 1: Python dependencies
# ============================================================================
step "Installing Python dependencies"

pip3 install --quiet numpy safetensors huggingface-hub 2>&1 | tail -5
ok "  Python packages installed."

# ============================================================================
# Step 2: Download model from HuggingFace
# ============================================================================
step "Downloading Qwen3.5-397B-A17B-4bit"

MODEL_REPO="mlx-community/Qwen3.5-397B-A17B-4bit"
MODEL_DIR="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit"

# Check if model is already downloaded
if [ -d "$MODEL_DIR" ]; then
    # Find the snapshot directory
    SNAPSHOT_DIR=$(find "$MODEL_DIR/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
    if [ -n "$SNAPSHOT_DIR" ] && [ -f "$SNAPSHOT_DIR/model.safetensors.index.json" ]; then
        ok "  Model already downloaded at $SNAPSHOT_DIR"
    else
        warn "  Partial download detected. Resuming..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', local_dir=None)
" 2>&1 | tail -5
        SNAPSHOT_DIR=$(find "$MODEL_DIR/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
    fi
else
    echo "  Downloading ~214GB model. This will take a while..."
    echo "  (Download is resumable — safe to interrupt and re-run)"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', local_dir=None)
" 2>&1 | tail -5
    SNAPSHOT_DIR=$(find "$MODEL_DIR/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
fi

if [ -z "$SNAPSHOT_DIR" ] || [ ! -f "$SNAPSHOT_DIR/model.safetensors.index.json" ]; then
    fail "Model download failed. Check your internet connection and try again."
fi
ok "  Model ready: $SNAPSHOT_DIR"

# ============================================================================
# Step 3: Extract non-expert weights
# ============================================================================
step "Extracting non-expert weights"

if [ -f "$ENGINE_DIR/model_weights.bin" ] && [ -f "$ENGINE_DIR/model_weights.json" ]; then
    ok "  model_weights.bin already exists — skipping."
else
    python3 "$ENGINE_DIR/extract_weights.py" --model "$SNAPSHOT_DIR" --output "$ENGINE_DIR/"
    ok "  Done: engine/model_weights.bin ($(du -sh "$ENGINE_DIR/model_weights.bin" | cut -f1))"
fi

# ============================================================================
# Step 4: Export tokenizer
# ============================================================================
step "Exporting tokenizer"

if [ -f "$ENGINE_DIR/tokenizer.bin" ]; then
    ok "  tokenizer.bin already exists — skipping."
else
    python3 "$ENGINE_DIR/export_tokenizer.py" "$SNAPSHOT_DIR/tokenizer.json" "$ENGINE_DIR/tokenizer.bin"
    ok "  Done: engine/tokenizer.bin"
fi

# Export vocab.bin if missing (needed for token decoding in server)
if [ ! -f "$ENGINE_DIR/vocab.bin" ]; then
    warn "  vocab.bin not found — will be generated on first server start."
fi

# ============================================================================
# Step 5: Repack expert weights
# ============================================================================
step "Repacking expert weights (209GB — this takes 30-60 minutes)"

PACKED_DIR="$REPO_DIR/packed_experts"
if [ -d "$PACKED_DIR" ] && [ -f "$PACKED_DIR/layer_59.bin" ]; then
    ok "  packed_experts/ already exists — skipping."
else
    cd "$REPO_DIR"
    python3 "$ENGINE_DIR/repack_experts.py" --index "$ENGINE_DIR/expert_index.json"
    ok "  Done: packed_experts/ ($(du -sh "$PACKED_DIR" | cut -f1))"
fi

# ============================================================================
# Step 6: Compile binaries
# ============================================================================
step "Compiling inference engine and PRE"

cd "$ENGINE_DIR"
make clean 2>/dev/null || true
make all 2>&1 | tail -3
ok "  Built: infer ($(du -sh infer | cut -f1)) + pre ($(du -sh pre | cut -f1))"

# ============================================================================
# Step 7: Install pre-launch command
# ============================================================================
step "Installing pre-launch command"

make install 2>&1
# Ensure ~/.local/bin is in PATH
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    SHELL_RC=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bash_profile"
    fi
    if [ -n "$SHELL_RC" ]; then
        if ! grep -q '.local/bin' "$SHELL_RC" 2>/dev/null; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
            ok "  Added ~/.local/bin to PATH in $SHELL_RC"
        fi
    fi
fi

# ============================================================================
# Step 8: Set up system prompt
# ============================================================================
step "Setting up system prompt"

mkdir -p "$HOME/.flash-moe/sessions"
if [ ! -f "$HOME/.flash-moe/system.md" ]; then
    cp "$REPO_DIR/system.md" "$HOME/.flash-moe/system.md"
    ok "  Installed system prompt at ~/.flash-moe/system.md"
else
    ok "  System prompt already exists — keeping existing."
fi

# ============================================================================
# Done!
# ============================================================================
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║${RESET}${BOLD}  PRE installation complete!                              ${GREEN}║${RESET}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}Quick start:${RESET}"
echo -e "    ${CYAN}pre-launch${RESET}                  Launch from any directory"
echo -e "    ${CYAN}pre-launch --show-think${RESET}     Launch with visible reasoning"
echo ""
echo -e "  ${BOLD}Or manually:${RESET}"
echo -e "    ${DIM}cd $ENGINE_DIR${RESET}"
echo -e "    ${DIM}./infer --serve 8000  ${RESET}${DIM}# Terminal 1: start server${RESET}"
echo -e "    ${DIM}./pre                 ${RESET}${DIM}# Terminal 2: start PRE${RESET}"
echo ""
echo -e "  ${BOLD}System prompt:${RESET} ~/.flash-moe/system.md (edit to customize)"
echo -e "  ${BOLD}Sessions:${RESET}      ~/.flash-moe/sessions/"
echo -e "  ${BOLD}History:${RESET}       ~/.flash-moe/pre_history"
echo ""
echo -e "  Type ${BOLD}/help${RESET} inside PRE for all commands."
echo ""
