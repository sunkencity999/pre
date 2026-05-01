#!/bin/bash
# pre-server.sh — Headless PRE server launcher (for LaunchAgent auto-start)
#
# Ensures Ollama is running, pre-warms the model into GPU memory,
# then exec's the web server so launchd can manage the process directly.
#
# Usage:
#   ./pre-server.sh              Start server (foreground, for launchd)
#   ./pre-server.sh --status     Check if server is running
#   ./pre-server.sh --stop       Stop the server

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PRE_PORT:-11434}"
WEB_PORT="${PRE_WEB_PORT:-7749}"

# Read configured context window from ~/.pre/context
MODEL_CTX=131072
CTX_FILE="$HOME/.pre/context"
if [ -f "$CTX_FILE" ]; then
    read -r _ctx < "$CTX_FILE" 2>/dev/null
    _ctx="${_ctx//[!0-9]/}"
    if [ -n "$_ctx" ] && [ "$_ctx" -ge 2048 ] 2>/dev/null && [ "$_ctx" -le 262144 ] 2>/dev/null; then
        MODEL_CTX="$_ctx"
    fi
fi

# Status check
if [ "${1:-}" = "--status" ]; then
    if curl -sf "http://localhost:${WEB_PORT}/api/status" >/dev/null 2>&1; then
        echo "PRE server is running on port $WEB_PORT"
        if ollama ps 2>/dev/null | grep -q "pre-gemma4"; then
            MEM=$(ollama ps 2>/dev/null | grep "pre-gemma4" | awk '{print $3, $4}')
            echo "Model loaded (${MEM}, context: ${MODEL_CTX})"
        fi
        exit 0
    else
        echo "PRE server is not running"
        exit 1
    fi
fi

# Stop
if [ "${1:-}" = "--stop" ]; then
    PLIST="$HOME/Library/LaunchAgents/com.pre.server.plist"
    if [ -f "$PLIST" ]; then
        launchctl unload "$PLIST" 2>/dev/null || true
        echo "PRE server stopped (LaunchAgent unloaded)"
        echo "To re-enable: launchctl load $PLIST"
    else
        # No LaunchAgent — try to kill the process directly
        PID=$(lsof -ti :"$WEB_PORT" -sTCP:LISTEN 2>/dev/null | head -1)
        if [ -n "$PID" ]; then
            kill "$PID" 2>/dev/null
            echo "PRE server stopped (PID $PID)"
        else
            echo "PRE server is not running"
        fi
    fi
    exit 0
fi

# --- Startup sequence ---

# Set Ollama env vars here so manual invocations (outside LaunchAgent) also get
# the optimized settings. The LaunchAgent plist carries them too, but exporting
# here means `pre-server.sh` works correctly however it's launched.
export OLLAMA_KEEP_ALIVE=24h
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

# Detect GPU backend: NVIDIA eGPU (CUDA via TinyGPU) vs Apple Silicon (Metal)
# Flash Attention: ON for CUDA, OFF for Metal (Gemma 4 hybrid attention is slower on Metal)
GPU_BACKEND="metal"
if systemextensionsctl list 2>/dev/null | grep -qi "tinygpu"; then
    # TinyGPU driver loaded — check for NVIDIA via Docker
    if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
        EGPU_VRAM_MIB=$(docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 \
            nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
        if [ -n "$EGPU_VRAM_MIB" ] && [ "$EGPU_VRAM_MIB" -gt 0 ] 2>/dev/null; then
            GPU_BACKEND="cuda"
            EGPU_VRAM_GB=$((EGPU_VRAM_MIB / 1024))
        fi
    fi
fi

if [ "$GPU_BACKEND" = "cuda" ]; then
    export OLLAMA_FLASH_ATTENTION=1
    # KV cache type: q8_0 only if eGPU VRAM >= 28GB (q8_0 model installed)
    if [ "${EGPU_VRAM_GB:-0}" -ge 28 ]; then
        export OLLAMA_KV_CACHE_TYPE=q8_0
    else
        export OLLAMA_KV_CACHE_TYPE=q4_0
    fi
    export OLLAMA_GPU_OVERHEAD=256000000
else
    export OLLAMA_FLASH_ATTENTION=0
fi

# Ensure Ollama is running
if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    # Try starting Ollama.app first (preferred — handles updates, UI icon)
    if [ -d "/Applications/Ollama.app" ]; then
        open -g -a Ollama 2>/dev/null || true
    else
        ollama serve >/dev/null 2>&1 &
    fi
    # Wait for Ollama API
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "ERROR: Ollama failed to start within 30 seconds" >&2
        exit 1
    fi
fi

# Pre-warm model into GPU memory (avoids cold-start delay on first request)
if ! ollama ps 2>/dev/null | grep -q "pre-gemma4"; then
    curl -sf --max-time 300 "http://localhost:${PORT}/api/chat" \
        -d "{\"model\":\"pre-gemma4\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"keep_alive\":\"24h\",\"options\":{\"num_predict\":1,\"num_ctx\":${MODEL_CTX}}}" \
        >/dev/null 2>&1 || true
fi

# Start web server — exec replaces this process so launchd manages it directly
cd "$SCRIPT_DIR"
exec node server.js
