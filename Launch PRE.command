#!/bin/bash
# ============================================================
#  PRE - Personal Reasoning Engine
#  Double-click this file in Finder to launch PRE.
# ============================================================
#
# This starts Ollama (if needed), launches the web server,
# and opens your browser to http://localhost:7749.
# Close this terminal window or press Ctrl+C to stop.

cd "$(dirname "$0")" || exit 1

WEB_DIR="$(pwd)/web"
PRE_PORT="${PRE_PORT:-11434}"
PRE_WEB_PORT="${PRE_WEB_PORT:-7749}"
PRE_URL="http://localhost:${PRE_WEB_PORT}"

clear

echo ""
echo "  PRE - Personal Reasoning Engine"
echo "  ================================"
echo ""

# ── Check if already running ──────────────────────────────────

if curl -sf "${PRE_URL}/api/status" >/dev/null 2>&1; then
    echo "  PRE is already running on port ${PRE_WEB_PORT}."
    echo "  Opening browser..."
    open "${PRE_URL}"
    echo ""
    echo "  Press any key to close this window..."
    read -n 1 -s
    exit 0
fi

# ── Start Ollama ──────────────────────────────────────────────

echo "  Checking Ollama..."
if ! curl -sf "http://127.0.0.1:${PRE_PORT}/v1/models" >/dev/null 2>&1; then
    if [ -d "/Applications/Ollama.app" ]; then
        echo "  Starting Ollama.app..."
        open -g -a Ollama
    elif command -v ollama >/dev/null 2>&1; then
        echo "  Starting Ollama..."
        ollama serve >/dev/null 2>&1 &
    else
        echo "  ERROR: Ollama not found. Install from https://ollama.com"
        echo ""
        echo "  Press any key to close..."
        read -n 1 -s
        exit 1
    fi

    # Wait for Ollama API
    printf "  Waiting for Ollama"
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:${PRE_PORT}/v1/models" >/dev/null 2>&1; then
            echo " ready."
            break
        fi
        printf "."
        sleep 1
    done
    echo ""
fi

# ── Pre-warm model ────────────────────────────────────────────

CTX_FILE="$HOME/.pre/context"
CTX=8192
if [ -f "$CTX_FILE" ]; then
    CTX=$(cat "$CTX_FILE" 2>/dev/null | tr -d '[:space:]')
fi

echo "  Pre-warming model..."
curl -sf "http://127.0.0.1:${PRE_PORT}/api/generate" \
    -d "{\"model\":\"pre-gemma4\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1,\"num_ctx\":${CTX}}}" \
    >/dev/null 2>&1 &

# ── Start web server ──────────────────────────────────────────

if [ ! -f "${WEB_DIR}/server.js" ]; then
    echo "  ERROR: web/server.js not found. Run the installer first."
    echo ""
    echo "  Press any key to close..."
    read -n 1 -s
    exit 1
fi

echo "  Starting PRE web server on port ${PRE_WEB_PORT}..."
echo ""

# Open browser after a short delay (server needs a moment to bind)
(sleep 2 && open "${PRE_URL}") &

# Run server in foreground so closing the terminal stops it
cd "${WEB_DIR}" && exec node server.js
