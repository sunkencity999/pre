#!/bin/bash
# install.sh — Full setup for PRE (Personal Reasoning Engine)
#
# This script handles everything from Ollama to ready-to-run:
#   0. Checks system requirements (Apple Silicon, macOS, RAM, disk)
#   1. Installs/verifies Ollama
#   1b. Configures Ollama environment (FA, keep-alive, parallelism)
#   2. Pulls the base model (gemma4:26b-a4b-it-q8_0, ~28GB)
#   2b. Pulls embedding model (nomic-embed-text for experience ledger + RAG)
#   3. Creates optimized custom model (pre-gemma4) from Modelfile
#   4. Installs Xcode CLI tools if needed
#   5. Compiles PRE CLI binary
#   6. Sets up Web GUI (npm install, terminal-notifier)
#   6b. Configures MCP integration (Claude, Codex, Antigravity)
#   7. Installs pre-launch command to ~/.local/bin
#   8. Sets up ~/.pre/ directories (sessions, memory, rag, workflows, triggers)
#   8b. Optional: ComfyUI image generation setup
#   8c. Optional: cliclick for Computer Use (desktop automation)
#   8d. Optional: Whisper + FFmpeg for voice interface
#   9. Pre-warms model into GPU memory
#   10. Optional: auto-start at login (LaunchAgent)
#
# Run time: 5-20 minutes depending on internet speed.
# Disk space required: ~28GB for model + negligible for binaries.
#
# Usage:
#   ./install.sh           Interactive install (prompts for optional steps)
#   ./install.sh --yes     Non-interactive: accept all defaults, skip optional steps
#   ./install.sh -y        Same as --yes

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
BASE_MODEL="gemma4:26b-a4b-it-q8_0"
CUSTOM_MODEL="pre-gemma4"

# --- Flag parsing ---
AUTO_YES=false
for arg in "$@"; do
    case "$arg" in
        --yes|-y) AUTO_YES=true ;;
    esac
done

# In --yes mode, prompts auto-accept Y/n defaults and skip y/N defaults
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

# Error trap — show which step failed
trap 'echo -e "\n${RED}Installation failed at line $LINENO.${RESET}\n${DIM}Re-run the script to retry — it will pick up where it left off.${RESET}"' ERR

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

# RAM — q8_0 model loads at ~32GB, need headroom for KV cache + OS
RAM_BYTES=$(sysctl -n hw.memsize)
RAM_GB=$((RAM_BYTES / 1073741824))
if [ "$RAM_GB" -lt 32 ]; then
    fail "PRE requires at least 32GB unified memory (q8_0 model loads at ~32GB). Detected: ${RAM_GB}GB"
fi
if [ "$RAM_GB" -lt 64 ]; then
    warn "  RAM: ${RAM_GB}GB — functional, but 64GB+ recommended for large context windows."
else
    ok "  RAM: ${RAM_GB}GB unified memory"
fi

# Auto-size context window based on available RAM.
# Gemma 4 26B q8_0 uses ~28GB for weights. KV cache scales linearly with context:
#   128K ≈ 4GB, 64K ≈ 2GB, 32K ≈ 1GB, 16K ≈ 0.5GB, 8K ≈ 0.25GB
# We need headroom for macOS, apps, and swap avoidance.
if [ "$RAM_GB" -ge 96 ]; then
    OPTIMAL_CTX=131072   # 128K — full Gemma 4 capacity
elif [ "$RAM_GB" -ge 64 ]; then
    OPTIMAL_CTX=65536    # 64K
elif [ "$RAM_GB" -ge 48 ]; then
    OPTIMAL_CTX=32768    # 32K
elif [ "$RAM_GB" -ge 36 ]; then
    OPTIMAL_CTX=16384    # 16K
else
    OPTIMAL_CTX=8192     # 8K — tight fit on 32GB
fi
CTX_HUMAN="$(( OPTIMAL_CTX / 1024 ))K"
ok "  Context window: ${CTX_HUMAN} (${OPTIMAL_CTX} tokens, auto-sized for ${RAM_GB}GB RAM)"

# Disk space (~28GB for q8_0 model, ~2GB headroom)
AVAIL_KB=$(df -k "$REPO_DIR" | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1048576))
if [ "$AVAIL_GB" -lt 35 ]; then
    warn "  Disk: ${AVAIL_GB}GB available — need ~28GB for q8_0 model download."
    if ! ask_yn "  Continue anyway? [y/N] " "N"; then exit 1; fi
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
        if ! ask_yn "  Install Ollama via Homebrew? [Y/n] " "Y"; then
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
# Step 1b: Ollama environment optimizations for Gemma 4 MoE
# ============================================================================
step "Configuring Ollama environment"

# These settings are tuned for Gemma 4 26B MoE on Apple Silicon:
#   FA=0:            Gemma 4's hybrid attention (sliding-window + global) is slower/unstable with Flash Attention
#   KEEP_ALIVE=24h:  Keep model loaded in GPU memory between requests
#   NUM_PARALLEL=1:  Single-user — don't split KV cache across parallel slots
#   MAX_LOADED=1:    Only one model at a time to avoid memory waste

OLLAMA_ENVS=(
    "OLLAMA_FLASH_ATTENTION:0"
    "OLLAMA_KEEP_ALIVE:24h"
    "OLLAMA_NUM_PARALLEL:1"
    "OLLAMA_MAX_LOADED_MODELS:1"
)

# Set via launchctl for the macOS Ollama.app
NEEDS_RESTART=false
for entry in "${OLLAMA_ENVS[@]}"; do
    KEY="${entry%%:*}"
    VAL="${entry#*:}"
    CURRENT=$(launchctl getenv "$KEY" 2>/dev/null || echo "")
    if [ "$CURRENT" != "$VAL" ]; then
        launchctl setenv "$KEY" "$VAL"
        NEEDS_RESTART=true
    fi
done
ok "  launchctl environment set (FA=0, KEEP_ALIVE=24h, NUM_PARALLEL=1, MAX_LOADED=1)"

# Persist in shell profile so new terminals inherit the values
SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_RC="$HOME/.bash_profile"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

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

# Restart Ollama if env vars changed and it's running
if [ "$NEEDS_RESTART" = true ]; then
    if pgrep -f "Ollama.app" >/dev/null 2>&1; then
        echo "  Restarting Ollama to pick up new environment..."
        pkill -f "Ollama.app" 2>/dev/null
        sleep 3
        open -a Ollama 2>/dev/null || ollama serve >/dev/null 2>&1 &
        # Wait for API
        for i in $(seq 1 30); do
            if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then break; fi
            sleep 1
        done
        ok "  Ollama restarted with optimized settings"
    fi
fi

# ============================================================================
# Step 2: Pull base model
# ============================================================================
step "Pulling base model ($BASE_MODEL)"

if ollama list 2>/dev/null | grep -q "gemma4.*26b-a4b-it-q8_0"; then
    ok "  Base model already available."
else
    echo "  Downloading ~28GB model. This may take a while..."
    echo "  (Download is resumable — safe to interrupt and re-run)"
    ollama pull "$BASE_MODEL" || fail "Failed to pull $BASE_MODEL"
    ok "  Base model downloaded."
fi

# ============================================================================
# Step 2b: Pull embedding model (for experience ledger semantic search)
# ============================================================================
step "Pulling embedding model (nomic-embed-text — experience ledger + RAG)"

if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
    ok "  Embedding model already available."
else
    echo "  Downloading nomic-embed-text (~274MB)..."
    ollama pull nomic-embed-text || warn "  Failed to pull nomic-embed-text — experience ledger and RAG will use keyword search fallback."
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
    if ask_yn "  Recreate from Modelfile? [y/N] " "N"; then
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

# Swift compiler (needed for EventKit-based Calendar/Reminders tools)
if command -v swiftc &>/dev/null; then
    ok "  Swift: $(swiftc --version 2>&1 | head -1 | sed 's/.*version: //')"
else
    warn "  swiftc not found — Calendar/Reminders tools will compile on first use."
    warn "  Install Xcode CLI tools to resolve: xcode-select --install"
fi

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
make pre telegram menubar 2>&1 | tail -5
if [ ! -x "$ENGINE_DIR/pre" ]; then
    fail "Build failed — 'pre' binary not found."
fi
ok "  Built: pre ($(du -sh pre | cut -f1))"
if [ -x "$ENGINE_DIR/pre-telegram" ]; then
    ok "  Built: pre-telegram ($(du -sh pre-telegram | cut -f1))"
fi
if [ -x "$ENGINE_DIR/PRE.app/Contents/MacOS/pre-menubar" ]; then
    ok "  Built: PRE.app (menu bar helper)"
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
            ok "  Web GUI dependencies installed"

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
# Step 6b: MCP integration (Claude Code / Claude Desktop / Codex / Antigravity)
# ============================================================================
step "MCP Integration (use PRE from Claude, Codex, Antigravity, or other AI tools)"

MCP_STDIO_PATH="$WEB_DIR/mcp-stdio.js"
MCP_SNIPPET="{\"mcpServers\":{\"pre\":{\"command\":\"node\",\"args\":[\"$MCP_STDIO_PATH\"]}}}"
MCP_TOOLS_CONFIGURED=0

# Shared delegation instructions — written to each tool's instructions file
PRE_DELEGATION_BLOCK="## PRE — Local AI Agent (MCP)

PRE is a local AI agent running on this machine (Ollama + Gemma 4, ${CTX_HUMAN} context).
It is available as an MCP tool to offload execution-heavy tasks at zero API token cost.

### When to delegate to PRE (\`pre_agent\` or \`pre_chat\`):
- System info, disk, network, process checks
- File search, reading, and directory listing
- Git operations and status checks
- Shell command execution
- Summarization, translation, data formatting
- Simple factual questions
- Memory search (\`pre_memory_search\`)

### When to handle directly (do NOT delegate):
- Complex code generation and software architecture
- Security reviews and vulnerability analysis
- Nuanced multi-factor analysis and bug reasoning
- Any task where frontier reasoning quality is critical

### MCP tools:
- \`pre_agent\` — Full agentic task with 63+ tools (file I/O, shell, web, macOS apps, RAG, etc.)
- \`pre_chat\` — Quick Q&A without tools (fastest)
- \`pre_memory_search\` — Search PRE's persistent memory store
- \`pre_sessions\` — List recent PRE sessions"

# Helper: append PRE delegation instructions to a file if not already present
write_delegation_instructions() {
    local target_file="$1"
    local tool_name="$2"
    if [ -f "$target_file" ] && grep -q "## PRE — Local AI Agent" "$target_file" 2>/dev/null; then
        ok "  PRE delegation instructions already in $tool_name instructions."
    else
        # Append with a blank line separator
        if [ -f "$target_file" ] && [ -s "$target_file" ]; then
            printf '\n\n%s\n' "$PRE_DELEGATION_BLOCK" >> "$target_file"
        else
            printf '%s\n' "$PRE_DELEGATION_BLOCK" > "$target_file"
        fi
        ok "  Wrote PRE delegation guidelines to $tool_name instructions."
    fi
}

# ── Claude ──────────────────────────────────────────────────────────────────

CLAUDE_DESKTOP_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
CLAUDE_DETECTED=0

# Claude Desktop
if [ -d "$HOME/Library/Application Support/Claude" ]; then
    CLAUDE_DETECTED=1
    echo "  Claude Desktop detected."
    echo ""
    echo -e "  PRE can serve as an MCP tool for Claude, letting Claude delegate"
    echo -e "  execution-heavy tasks to your local model ${BOLD}at zero token cost${RESET}."
    echo ""
    if ask_yn "  Add PRE to Claude Desktop? [Y/n] " "Y"; then
        if [ -f "$CLAUDE_DESKTOP_CONFIG" ]; then
            if grep -q '"pre"' "$CLAUDE_DESKTOP_CONFIG" 2>/dev/null; then
                ok "  PRE already configured in Claude Desktop."
            else
                node -e "
const fs = require('fs');
const p = '$CLAUDE_DESKTOP_CONFIG';
let cfg = {};
try { cfg = JSON.parse(fs.readFileSync(p, 'utf-8')); } catch {}
if (!cfg.mcpServers) cfg.mcpServers = {};
cfg.mcpServers.pre = { command: 'node', args: ['$MCP_STDIO_PATH'] };
fs.writeFileSync(p, JSON.stringify(cfg, null, 2) + '\n');
"
                ok "  PRE added to Claude Desktop config."
                warn "  Restart Claude Desktop to activate."
            fi
        else
            mkdir -p "$(dirname "$CLAUDE_DESKTOP_CONFIG")"
            echo "$MCP_SNIPPET" | node -e "process.stdout.write(JSON.stringify(JSON.parse(require('fs').readFileSync('/dev/stdin','utf-8')),null,2)+'\n')" > "$CLAUDE_DESKTOP_CONFIG" 2>/dev/null \
                || echo "$MCP_SNIPPET" > "$CLAUDE_DESKTOP_CONFIG"
            ok "  Created Claude Desktop config with PRE."
            warn "  Restart Claude Desktop to activate."
        fi
        MCP_TOOLS_CONFIGURED=$((MCP_TOOLS_CONFIGURED + 1))
    else
        echo "  Skipped. You can add it later — see the README."
    fi
fi

# Claude Code
if [ -d "$HOME/.claude" ]; then
    CLAUDE_DETECTED=1
    ok "  Claude Code detected."
    echo ""
    echo -e "  ${BOLD}To use PRE from Claude Code:${RESET}"
    echo -e "  Add to your project ${CYAN}.mcp.json${RESET} or global settings:"
    echo ""
    echo -e "    ${DIM}{${RESET}"
    echo -e "    ${DIM}  \"mcpServers\": {${RESET}"
    echo -e "    ${DIM}    \"pre\": {${RESET}"
    echo -e "    ${DIM}      \"command\": \"node\",${RESET}"
    echo -e "    ${DIM}      \"args\": [\"${MCP_STDIO_PATH}\"]${RESET}"
    echo -e "    ${DIM}    }${RESET}"
    echo -e "    ${DIM}  }${RESET}"
    echo -e "    ${DIM}}${RESET}"
    echo ""
fi

# Write delegation instructions for Claude (CLAUDE.md in home dir)
if [ "$CLAUDE_DETECTED" -eq 1 ]; then
    CLAUDE_MD="$HOME/CLAUDE.md"
    write_delegation_instructions "$CLAUDE_MD" "CLAUDE.md"
fi

# ── Codex (OpenAI) ──────────────────────────────────────────────────────────

CODEX_CONFIG="$HOME/.codex/config.toml"
if [ -d "$HOME/.codex" ] || command -v codex &>/dev/null; then
    echo ""
    echo "  Codex detected."
    echo ""
    if ask_yn "  Add PRE as an MCP server for Codex? [Y/n] " "Y"; then
        # Add MCP server to config.toml
        if [ -f "$CODEX_CONFIG" ] && grep -q '\[mcp_servers\.pre\]' "$CODEX_CONFIG" 2>/dev/null; then
            ok "  PRE already configured in Codex."
        else
            # Append TOML block to config
            {
                echo ""
                echo "[mcp_servers.pre]"
                echo "command = \"node\""
                echo "args = [\"$MCP_STDIO_PATH\"]"
            } >> "$CODEX_CONFIG"
            ok "  PRE added to Codex config ($CODEX_CONFIG)."
            MCP_TOOLS_CONFIGURED=$((MCP_TOOLS_CONFIGURED + 1))
        fi

        # Write delegation instructions to Codex instructions file
        CODEX_INSTRUCTIONS="$HOME/.codex/instructions.md"
        write_delegation_instructions "$CODEX_INSTRUCTIONS" "Codex (instructions.md)"
    else
        echo "  Skipped. Add manually to ~/.codex/config.toml — see the README."
    fi
fi

# ── Antigravity (Google) ────────────────────────────────────────────────────

ANTIGRAVITY_SETTINGS="$HOME/.gemini/settings.json"
if [ -d "$HOME/.gemini" ] || [ -d "$HOME/.antigravity" ] || command -v agy &>/dev/null; then
    echo ""
    echo "  Antigravity detected."
    echo ""
    if ask_yn "  Add PRE as an MCP server for Antigravity? [Y/n] " "Y"; then
        # Merge PRE into settings.json mcpServers
        if [ -f "$ANTIGRAVITY_SETTINGS" ] && grep -q '"pre"' "$ANTIGRAVITY_SETTINGS" 2>/dev/null; then
            ok "  PRE already configured in Antigravity."
        else
            if command -v node &>/dev/null; then
                node -e "
const fs = require('fs');
const p = '$ANTIGRAVITY_SETTINGS';
let cfg = {};
try { cfg = JSON.parse(fs.readFileSync(p, 'utf-8')); } catch {}
if (!cfg.mcpServers) cfg.mcpServers = {};
cfg.mcpServers.pre = { command: 'node', args: ['$MCP_STDIO_PATH'] };
fs.writeFileSync(p, JSON.stringify(cfg, null, 2) + '\n');
"
                ok "  PRE added to Antigravity settings ($ANTIGRAVITY_SETTINGS)."
                MCP_TOOLS_CONFIGURED=$((MCP_TOOLS_CONFIGURED + 1))
            else
                warn "  Node.js needed to update Antigravity config — add manually."
            fi
        fi

        # Write delegation instructions to GEMINI.md
        GEMINI_MD="$HOME/.gemini/GEMINI.md"
        write_delegation_instructions "$GEMINI_MD" "Antigravity (GEMINI.md)"
    else
        echo "  Skipped. Add manually to ~/.gemini/settings.json — see the README."
    fi
fi

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
if [ "$MCP_TOOLS_CONFIGURED" -gt 0 ]; then
    ok "  Configured PRE as MCP tool for $MCP_TOOLS_CONFIGURED AI tool(s)."
fi
echo -e "  The MCP server auto-starts Ollama and PRE — no manual launch needed."
echo -e "  See ${CYAN}web/README.md${RESET} for delegation guidelines and cost savings data."

# ============================================================================
# Step 7: Install pre-launch command + menu bar app
# ============================================================================
step "Installing pre-launch command + menu bar app"

chmod +x "$ENGINE_DIR/pre-launch"
make -C "$ENGINE_DIR" install 2>&1

# Install PRE.app menu bar helper
if [ -d "$ENGINE_DIR/PRE.app" ] && [ -x "$ENGINE_DIR/PRE.app/Contents/MacOS/pre-menubar" ]; then
    mkdir -p "$HOME/Applications"
    rm -rf "$HOME/Applications/PRE.app"
    cp -R "$ENGINE_DIR/PRE.app" "$HOME/Applications/PRE.app"
    ok "  Installed PRE.app → ~/Applications/ (menu bar helper)"
    echo -e "  ${DIM}Start/stop PRE from the menu bar — no terminal needed.${RESET}"
fi

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
mkdir -p "$HOME/.pre/rag"
mkdir -p "$HOME/.pre/workflows"
mkdir -p "$HOME/.pre/custom_tools"
ok "  Created ~/.pre/sessions/"
ok "  Created ~/.pre/memory/"
ok "  Created ~/.pre/memory/experience/"
ok "  Created ~/.pre/checkpoints/"
ok "  Created ~/.pre/artifacts/"
ok "  Created ~/.pre/rag/"
ok "  Created ~/.pre/workflows/"
ok "  Created ~/.pre/custom_tools/"

# Write context window config (read at runtime by constants.js and pre-launch)
echo "$OPTIMAL_CTX" > "$HOME/.pre/context"
ok "  Context window: ${CTX_HUMAN} → ~/.pre/context"

# Create default config files if they don't exist
if [ ! -f "$HOME/.pre/hooks.json" ]; then
    echo '{"hooks":[]}' > "$HOME/.pre/hooks.json"
    ok "  Created ~/.pre/hooks.json"
fi
if [ ! -f "$HOME/.pre/mcp.json" ]; then
    echo '{"servers":{}}' > "$HOME/.pre/mcp.json"
    ok "  Created ~/.pre/mcp.json"
fi
if [ ! -f "$HOME/.pre/cron.json" ]; then
    echo '[]' > "$HOME/.pre/cron.json"
    ok "  Created ~/.pre/cron.json"
fi
if [ ! -f "$HOME/.pre/triggers.json" ]; then
    echo '[]' > "$HOME/.pre/triggers.json"
    ok "  Created ~/.pre/triggers.json"
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
    if [ "$RAM_GB" -lt 64 ] 2>/dev/null; then
        warn "  Note: Your system has ${RAM_GB}GB RAM. Image generation (SDXL + Gemma 4 q8_0) works"
        warn "  best with 64GB+. You can skip this and add it later with /connections."
    fi

    echo ""
    echo -e "  ${BOLD}Install local image generation?${RESET}"
    echo -e "  This sets up ComfyUI + Stable Diffusion XL for generating images from text."
    echo -e "  ${DIM}Requires: ~8GB disk, Python 3.10-3.13 (PyTorch), ~6.5GB model download${RESET}"
    echo ""
    if ask_yn "  Install ComfyUI? [y/N] " "N"; then
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
            VENV_OK=true
            echo "  Creating Python virtual environment..."
            $COMFYUI_PYTHON -m venv "$COMFYUI_VENV" || { warn "  Failed to create venv — skipping."; VENV_OK=false; }

            if [ "$VENV_OK" = true ]; then
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
                IMG_CHOICE="1"
                if [ "$AUTO_YES" = false ]; then
                    read -p "  Choice [1/2, default=1]: " -n 1 -r IMG_CHOICE
                    echo ""
                fi

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
# Step 8c: Computer Use setup (optional — desktop automation)
# ============================================================================
step "Desktop Automation Setup (optional)"

echo -e "  ${BOLD}Computer Use${RESET} lets PRE see your screen and control any application"
echo -e "  — click, type, scroll, drag, press key combos. It uses ${CYAN}cliclick${RESET}"
echo -e "  for mouse/keyboard control and ${CYAN}screencapture${RESET} (built-in) for screenshots."
echo ""

if command -v cliclick &>/dev/null; then
    ok "  cliclick already installed: $(cliclick -V 2>&1 | head -1)"
    echo -e "  ${DIM}macOS will prompt for Accessibility permissions on first use.${RESET}"
else
    if command -v brew &>/dev/null; then
        if ask_yn "  Install cliclick for desktop automation? [Y/n] " "Y"; then
            echo "  Installing cliclick..."
            brew install cliclick --quiet 2>&1 | tail -1
            if command -v cliclick &>/dev/null; then
                ok "  cliclick installed."
                echo -e "  ${DIM}macOS will prompt for Accessibility permissions on first use.${RESET}"
            else
                warn "  cliclick install failed — Computer Use will be unavailable."
                echo -e "  ${DIM}Install later with: brew install cliclick${RESET}"
            fi
        else
            echo "  Skipped. Install later with: brew install cliclick"
        fi
    else
        warn "  Homebrew not found — install cliclick manually for Computer Use."
        echo -e "  ${DIM}brew install cliclick${RESET}"
    fi
fi

# ============================================================================
# Step 8d: Voice interface setup (optional — speech-to-text)
# ============================================================================
step "Voice Interface Setup (optional)"

echo -e "  ${BOLD}Voice input${RESET} lets you talk to PRE — hold the microphone button in the"
echo -e "  web GUI to record, release to transcribe. Uses ${CYAN}OpenAI Whisper${RESET} (local,"
echo -e "  no cloud API) for speech-to-text and macOS ${CYAN}say${RESET} for text-to-speech."
echo ""

VOICE_CHANGED=false

if command -v whisper &>/dev/null; then
    ok "  Whisper already installed: $(whisper --version 2>&1 || echo 'available')"
else
    if command -v brew &>/dev/null; then
        if ask_yn "  Install Whisper for voice input? [Y/n] " "Y"; then
            echo "  Installing OpenAI Whisper via Homebrew (~46MB + dependencies)..."
            echo -e "  ${DIM}This may take a minute — Homebrew manages the Python environment.${RESET}"
            brew install openai-whisper --quiet 2>&1 | tail -3
            if command -v whisper &>/dev/null; then
                ok "  Whisper installed."
                VOICE_CHANGED=true
            else
                warn "  Whisper install failed."
                echo -e "  ${DIM}Install later with: brew install openai-whisper${RESET}"
            fi
        else
            echo "  Skipped. Install later with: brew install openai-whisper"
        fi
    else
        warn "  Homebrew not found — install Whisper manually for voice input."
        echo -e "  ${DIM}brew install openai-whisper${RESET}"
    fi
fi

# FFmpeg improves audio format handling (WebM → WAV conversion)
if command -v ffmpeg &>/dev/null; then
    ok "  FFmpeg available (audio format conversion)"
else
    if command -v brew &>/dev/null; then
        if ask_yn "  Install FFmpeg for audio format conversion? [Y/n] " "Y"; then
            echo "  Installing FFmpeg..."
            brew install ffmpeg --quiet 2>&1 | tail -1
            if command -v ffmpeg &>/dev/null; then
                ok "  FFmpeg installed."
                VOICE_CHANGED=true
            else
                warn "  FFmpeg install failed — voice input will still work, but some audio formats may not convert."
            fi
        else
            echo "  Skipped. Voice input works without FFmpeg, but some audio formats may not convert."
        fi
    fi
fi

if [ "$VOICE_CHANGED" = true ]; then
    echo ""
    ok "  Voice interface ready! Look for the microphone icon in the web GUI."
    echo -e "  ${DIM}Whisper downloads its model (~140MB) on first transcription.${RESET}"
fi

# ============================================================================
# Step 9: Pre-warm the model
# ============================================================================
step "Pre-warming model into GPU memory"

echo "  Loading $CUSTOM_MODEL into GPU with ${CTX_HUMAN} context (this takes 30-60 seconds)..."
# Send a real message with full num_ctx to force Ollama to allocate the complete KV cache.
# An empty messages array defers KV allocation to the first real request.
curl -sf --max-time 300 "http://localhost:${PORT}/api/chat" \
    -d "{\"model\":\"$CUSTOM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"keep_alive\":\"24h\",\"options\":{\"num_predict\":1,\"num_ctx\":$OPTIMAL_CTX}}" \
    >/dev/null 2>&1 || true

for i in $(seq 1 60); do
    if ollama ps 2>/dev/null | grep -q "$CUSTOM_MODEL"; then
        MEM=$(ollama ps 2>/dev/null | grep "$CUSTOM_MODEL" | awk '{print $4}')
        ok "  Model loaded into GPU memory (${MEM})."
        break
    fi
    sleep 1
done

if ! ollama ps 2>/dev/null | grep -q "$CUSTOM_MODEL"; then
    warn "  Model may not be fully loaded yet — it will load on first launch."
fi

# ============================================================================
# Step 10: Auto-start at login (optional LaunchAgent)
# ============================================================================
step "Auto-start at login (optional)"

PLIST_LABEL="com.pre.server"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
PRE_SERVER_SH="$WEB_DIR/pre-server.sh"
NODE_BIN=$(command -v node 2>/dev/null || echo "/opt/homebrew/bin/node")

echo -e "  PRE can start automatically when you log in, so the server and"
echo -e "  model are always ready. The web GUI, MCP server, and cron jobs"
echo -e "  will all be available immediately — no manual launch needed."
echo ""

if [ -f "$PLIST_PATH" ]; then
    ok "  LaunchAgent already installed ($PLIST_LABEL)."
    if ask_yn "  Reinstall with current paths? [y/N] " "N"; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
    else
        echo "  Keeping existing LaunchAgent."
        # Skip to Done
        SKIP_LAUNCHAGENT=true
    fi
fi

if [ "${SKIP_LAUNCHAGENT:-}" != "true" ]; then
    if ask_yn "  Start PRE automatically at login? [Y/n] " "Y"; then
        chmod +x "$PRE_SERVER_SH"
        mkdir -p "$HOME/Library/LaunchAgents"

        cat > "$PLIST_PATH" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${PRE_SERVER_SH}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>WorkingDirectory</key>
    <string>${WEB_DIR}</string>
    <key>StandardOutPath</key>
    <string>/tmp/pre-server.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/pre-server.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>OLLAMA_FLASH_ATTENTION</key>
        <string>0</string>
        <key>OLLAMA_KEEP_ALIVE</key>
        <string>24h</string>
        <key>OLLAMA_NUM_PARALLEL</key>
        <string>1</string>
        <key>OLLAMA_MAX_LOADED_MODELS</key>
        <string>1</string>
    </dict>
</dict>
</plist>
PLISTEOF

        # Load the LaunchAgent (starts immediately)
        launchctl load "$PLIST_PATH" 2>/dev/null || true

        ok "  LaunchAgent installed — PRE will start automatically at login."
        echo -e "  ${DIM}Server log: /tmp/pre-server.log${RESET}"
        echo -e "  ${DIM}To stop:    pre-server.sh --stop${RESET}"
        echo -e "  ${DIM}To disable: launchctl unload ~/Library/LaunchAgents/${PLIST_LABEL}.plist${RESET}"
    else
        echo "  Skipped. Launch PRE manually with: pre-launch"
    fi
fi

# ============================================================================
# Done!
# ============================================================================
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║${RESET}${BOLD}  PRE installation complete!                              ${GREEN}║${RESET}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
if [ -f "$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist" ]; then
echo -e "  ${BOLD}Launch:${RESET}  ${GREEN}PRE starts automatically at login${RESET}"
echo -e "    Web GUI is at ${CYAN}http://localhost:7749${RESET}"
echo -e "    ${CYAN}pre-launch${RESET}                  Open CLI (server already running)"
echo -e "    ${CYAN}pre-launch --show-think${RESET}     CLI with visible reasoning"
echo -e "    ${CYAN}pre-server.sh --status${RESET}      Check server status"
echo -e "    ${CYAN}pre-server.sh --stop${RESET}        Stop the server"
else
echo -e "  ${BOLD}Launch:${RESET}"
echo -e "    ${CYAN}pre-launch${RESET}                  Start PRE (CLI + Web GUI)"
echo -e "    ${CYAN}pre-launch --show-think${RESET}     Start with visible reasoning"
echo -e "    Web GUI auto-starts at ${CYAN}http://localhost:7749${RESET}"
fi
if [ -d "$HOME/Applications/PRE.app" ]; then
echo -e "    ${GREEN}PRE.app${RESET}                     Menu bar helper (~/Applications/)"
echo -e "    ${DIM}Start, stop, restart, and open PRE from the menu bar.${RESET}"
fi
echo ""
echo -e "  ${BOLD}Hardware Profile:${RESET}"
echo -e "    ${RAM_GB}GB unified memory → ${GREEN}${CTX_HUMAN} context window${RESET} (${OPTIMAL_CTX} tokens)"
echo -e "    ${DIM}To change: edit ~/.pre/context and restart PRE${RESET}"
echo ""
echo -e "  ${BOLD}First Launch:${RESET}"
echo -e "    PRE will ask you to name your agent and optionally"
echo -e "    configure cloud connections via ${BOLD}/connections${RESET} or Web GUI Settings."
echo ""
echo -e "  ${BOLD}Native macOS (works immediately — no setup):${RESET}"
echo -e "    • ${GREEN}Mail, Calendar, Contacts, Reminders, Notes, Spotlight${RESET}"
echo -e "    ${DIM}Uses your existing macOS accounts (iCloud, Gmail, Exchange, etc.)${RESET}"
echo -e "    ${DIM}macOS will prompt for Automation permissions on first use.${RESET}"
echo ""
echo -e "  ${BOLD}Cloud Connections (optional):${RESET}"
echo -e "    Type ${BOLD}/connections${RESET} inside PRE or use the Web GUI Settings to set up:"
echo -e "    • ${DIM}Google         — Gmail, Drive, Docs (built-in OAuth)${RESET}"
echo -e "    • ${DIM}Microsoft      — SharePoint, OneDrive (Azure AD OAuth)${RESET}"
echo -e "    • ${DIM}Slack          — channel messaging${RESET}"
echo -e "    • ${DIM}Telegram       — chat from your phone + cron delivery${RESET}"
echo -e "    • ${DIM}GitHub         — repos, issues, PRs${RESET}"
echo -e "    • ${DIM}Jira           — issues, projects, transitions${RESET}"
echo -e "    • ${DIM}Confluence     — wiki pages, search${RESET}"
echo -e "    • ${DIM}Smartsheet     — spreadsheets, rows, search${RESET}"
echo -e "    • ${DIM}Linear         — issue tracking, projects, cycles${RESET}"
echo -e "    • ${DIM}Zoom           — meetings, recordings${RESET}"
echo -e "    • ${DIM}Figma          — design files, comments, export${RESET}"
echo -e "    • ${DIM}Asana          — tasks, projects, search${RESET}"
echo -e "    • ${DIM}Brave Search   — web search${RESET}"
echo -e "    • ${DIM}Wolfram Alpha  — computation${RESET}"
if [ -f "$HOME/.pre/comfyui.json" ]; then
echo -e "    • ${GREEN}ComfyUI        — image generation (installed ✓)${RESET}"
else
echo -e "    • ${DIM}ComfyUI        — image generation (re-run install.sh to add)${RESET}"
fi
echo ""
echo -e "  ${BOLD}Features:${RESET}"
echo -e "    63+ tools including computer use (desktop automation), local RAG,"
echo -e "    event-driven triggers, voice interface, workflow capture and replay,"
echo -e "    sub-agents, browser automation, document export, hooks, experience"
echo -e "    ledger, and temporal memory awareness."
echo ""
echo -e "  ${BOLD}MCP Integration:${RESET}"
echo -e "    PRE serves as an MCP tool for Claude, Codex, Antigravity, and other AI tools."
echo -e "    Delegate execution-heavy tasks to your local model at ${GREEN}zero token cost${RESET}."
echo -e "    Delegation instructions auto-written to CLAUDE.md, instructions.md, GEMINI.md."
echo -e "    See ${CYAN}web/README.md${RESET} for setup and cost savings data."
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
echo -e "    ${DIM}~/.pre/triggers.json${RESET}      Event-driven triggers"
echo -e "    ${DIM}~/.pre/argus.json${RESET}         Argus companion config"
echo -e "    ${DIM}~/.pre/custom_tools/${RESET}      Dynamic virtual tool definitions"
echo -e "    ${DIM}~/.pre/rag/${RESET}               RAG indexes (semantic search)"
echo -e "    ${DIM}~/.pre/workflows/${RESET}          Recorded workflow sequences"
echo -e "    ${DIM}~/.pre/telegram.log${RESET}       Telegram bot log"
if ! command -v cliclick &>/dev/null || ! command -v whisper &>/dev/null; then
echo -e "  ${BOLD}Skipped Optional Features:${RESET}"
if ! command -v cliclick &>/dev/null; then
echo -e "    ${DIM}brew install cliclick${RESET}           Desktop automation (Computer Use)"
fi
if ! command -v whisper &>/dev/null; then
echo -e "    ${DIM}brew install openai-whisper${RESET}     Voice input (speech-to-text)"
fi
if ! command -v ffmpeg &>/dev/null; then
echo -e "    ${DIM}brew install ffmpeg${RESET}             Audio format conversion"
fi
echo -e "    ${DIM}Re-run install.sh to add these interactively.${RESET}"
fi
echo ""
echo -e "  Type ${BOLD}/help${RESET} inside PRE for all commands."
echo ""

# Keep Terminal window open if launched via Finder double-click
if [ "$AUTO_YES" = false ] && [ -t 0 ]; then
    read -n 1 -s -r -p "  Press any key to close this window."
    echo ""
fi
