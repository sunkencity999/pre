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
