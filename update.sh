#!/bin/bash
# update.sh - Update PRE (Personal Reasoning Engine) on macOS
#
# Detects whether PRE was installed via git clone or zip download,
# compares local version against the latest release on GitHub,
# and updates accordingly. User data (~/.pre/) is never touched.
#
# Usage:
#   bash update.sh          # Interactive
#   bash update.sh --yes    # Non-interactive (accept all defaults)

set -e

REPO_URL="https://github.com/sunkencity999/pre"
RAW_URL="https://raw.githubusercontent.com/sunkencity999/pre/main"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_DIR="$REPO_DIR/web"
ENGINE_DIR="$REPO_DIR/engine"
PRE_PORT="${PRE_PORT:-7749}"
AUTO_YES=false

if [[ "$1" == "--yes" || "$1" == "-y" ]]; then
    AUTO_YES=true
fi

# -- Helpers ----------------------------------------------------------------

step()  { printf "\n\033[1;36m=== %s ===\033[0m\n\n" "$1"; }
ok()    { printf "  \033[32m%s\033[0m\n" "$1"; }
warn()  { printf "  \033[33m%s\033[0m\n" "$1"; }
fail()  { printf "\n  \033[31m%s\033[0m\n" "$1"; exit 1; }

ask_yn() {
    local prompt="$1" default="$2"
    if $AUTO_YES; then
        [[ "$default" == "Y" ]] && return 0 || return 1
    fi
    read -p "$prompt " -n 1 -r
    echo
    if [[ "$default" == "Y" ]]; then
        [[ ! $REPLY =~ ^[Nn]$ ]]
    else
        [[ $REPLY =~ ^[Yy]$ ]]
    fi
}

# -- Detect install type ----------------------------------------------------

step "Checking installation"

IS_GIT=false
if [ -d "$REPO_DIR/.git" ]; then
    IS_GIT=true
    ok "Install type: git repository"
    # Check for local modifications
    if [ -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ]; then
        warn "Local modifications detected."
        git -C "$REPO_DIR" status --short
        echo ""
    fi
else
    ok "Install type: zip download (no git repository)"
fi

# -- Get local version ------------------------------------------------------

LOCAL_VERSION="unknown"
VERSION_FILE="$REPO_DIR/VERSION"
if [ -f "$VERSION_FILE" ]; then
    LOCAL_VERSION="$(cat "$VERSION_FILE" | tr -d '[:space:]')"
fi
ok "Local version: $LOCAL_VERSION"

# -- Get remote version -----------------------------------------------------

step "Checking for updates"

REMOTE_VERSION=""
if command -v curl >/dev/null 2>&1; then
    REMOTE_VERSION="$(curl -sf "${RAW_URL}/VERSION" 2>/dev/null | tr -d '[:space:]')" || true
fi

if [ -z "$REMOTE_VERSION" ]; then
    warn "Could not fetch remote version. Check your internet connection."
    if ! ask_yn "  Continue anyway? [y/N]" "N"; then
        exit 0
    fi
    REMOTE_VERSION="unknown"
fi

ok "Remote version: $REMOTE_VERSION"

# Compare versions
if [ "$LOCAL_VERSION" = "$REMOTE_VERSION" ] && [ "$REMOTE_VERSION" != "unknown" ]; then
    ok "You're already on the latest version ($LOCAL_VERSION)."
    if ! ask_yn "  Force update anyway? [y/N]" "N"; then
        echo ""
        echo "  No update needed."
        exit 0
    fi
else
    if [ "$REMOTE_VERSION" != "unknown" ]; then
        ok "Update available: $LOCAL_VERSION -> $REMOTE_VERSION"
    fi
fi

# -- Check if server is running ---------------------------------------------

SERVER_WAS_RUNNING=false
if curl -sf "http://localhost:${PRE_PORT}/api/status" >/dev/null 2>&1; then
    SERVER_WAS_RUNNING=true
    warn "PRE server is currently running."
    if ask_yn "  Stop it for the update? [Y/n]" "Y"; then
        echo "  Stopping server..."
        # Try graceful shutdown via the API, fall back to kill
        curl -sf "http://localhost:${PRE_PORT}/api/shutdown" >/dev/null 2>&1 || true
        sleep 1
        # Kill by port if still running
        lsof -ti :${PRE_PORT} -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null || true
        sleep 1
        ok "Server stopped."
    else
        warn "Updating while server is running - restart manually after."
    fi
fi

# -- Perform update ---------------------------------------------------------

step "Updating PRE"

if $IS_GIT; then
    # ---- Git-based update ----
    echo "  Pulling latest changes from origin..."

    # Stash local changes if any
    STASHED=false
    if [ -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ]; then
        warn "Stashing local changes..."
        git -C "$REPO_DIR" stash push -m "pre-update-$(date +%Y%m%d-%H%M%S)" >/dev/null 2>&1
        STASHED=true
    fi

    # Pull
    PULL_OUTPUT="$(git -C "$REPO_DIR" pull origin main 2>&1)" || {
        echo "$PULL_OUTPUT"
        if $STASHED; then
            warn "Restoring stashed changes..."
            git -C "$REPO_DIR" stash pop >/dev/null 2>&1 || true
        fi
        fail "Git pull failed. Resolve conflicts manually."
    }
    echo "  $PULL_OUTPUT"

    # Restore stashed changes
    if $STASHED; then
        echo ""
        warn "Restoring stashed changes..."
        if git -C "$REPO_DIR" stash pop >/dev/null 2>&1; then
            ok "Local changes restored."
        else
            warn "Could not auto-restore local changes. Run 'git stash pop' manually."
        fi
    fi

    ok "Git update complete."

else
    # ---- Zip-based update ----
    echo "  Downloading latest version from GitHub..."

    TEMP_DIR="$(mktemp -d)"
    ZIP_FILE="$TEMP_DIR/pre-latest.zip"
    EXTRACT_DIR="$TEMP_DIR/pre-main"

    # Download
    curl -fSL "${REPO_URL}/archive/refs/heads/main.zip" -o "$ZIP_FILE" || {
        rm -rf "$TEMP_DIR"
        fail "Download failed. Check your internet connection."
    }
    ok "Downloaded."

    # Extract
    echo "  Extracting..."
    unzip -q "$ZIP_FILE" -d "$TEMP_DIR" || {
        rm -rf "$TEMP_DIR"
        fail "Extraction failed."
    }

    # The zip extracts to a directory like "pre-main/"
    if [ ! -d "$EXTRACT_DIR" ]; then
        # Try to find the extracted directory
        EXTRACT_DIR="$(find "$TEMP_DIR" -maxdepth 1 -type d -name 'pre-*' | head -1)"
    fi

    if [ ! -d "$EXTRACT_DIR" ]; then
        rm -rf "$TEMP_DIR"
        fail "Could not find extracted files."
    fi

    # Copy files, preserving user data
    echo "  Updating files..."

    # Update engine/ (source code, Modelfile, scripts)
    if [ -d "$EXTRACT_DIR/engine" ]; then
        rsync -a --exclude='*.o' --exclude='pre' --exclude='telegram' \
            "$EXTRACT_DIR/engine/" "$ENGINE_DIR/"
    fi

    # Update web/ (server, tools, frontend)
    if [ -d "$EXTRACT_DIR/web" ]; then
        rsync -a --exclude='node_modules/' \
            "$EXTRACT_DIR/web/" "$WEB_DIR/"
    fi

    # Update root files (scripts, README, VERSION, etc.)
    for f in VERSION README.md install.sh install.ps1 install.cmd \
             "Launch PRE.command" "Launch PRE.cmd" "PRE Tray.cmd" \
             "Install PRE.command" "Update PRE.command" "Update PRE.cmd" \
             update.sh update.ps1 system.md benchmark.sh; do
        if [ -f "$EXTRACT_DIR/$f" ]; then
            cp "$EXTRACT_DIR/$f" "$REPO_DIR/$f"
        fi
    done

    # Make scripts executable
    chmod +x "$REPO_DIR"/*.sh "$REPO_DIR"/*.command 2>/dev/null || true

    # Clean up
    rm -rf "$TEMP_DIR"
    ok "Files updated."
fi

# -- Update dependencies ----------------------------------------------------

step "Updating dependencies"

if [ -f "$WEB_DIR/package.json" ]; then
    echo "  Running npm install..."
    cd "$WEB_DIR"
    npm install --silent 2>&1 | tail -3
    ok "Dependencies updated."
else
    warn "web/package.json not found - skipping npm install."
fi

# -- Rebuild CLI (macOS only) -----------------------------------------------

if [ -f "$ENGINE_DIR/Makefile" ] && command -v make >/dev/null 2>&1; then
    step "Rebuilding CLI (optional)"
    if ask_yn "  Rebuild the CLI binary? [y/N]" "N"; then
        cd "$ENGINE_DIR"
        make pre 2>&1 | tail -5
        ok "CLI rebuilt."
    else
        echo "  Skipped. Run 'cd engine && make pre' to rebuild later."
    fi
fi

# -- Check if Modelfile changed ---------------------------------------------

# If the Modelfile changed, the custom model may need recreation
if $IS_GIT; then
    MODELFILE_CHANGED="$(git -C "$REPO_DIR" diff HEAD~1 --name-only 2>/dev/null | grep -c 'Modelfile' || true)"
    if [ "$MODELFILE_CHANGED" -gt 0 ]; then
        warn "Modelfile has changed. You may want to recreate the custom model:"
        echo "    ollama create pre-gemma4 -f engine/Modelfile"
    fi
fi

# -- Restart server if it was running ----------------------------------------

if $SERVER_WAS_RUNNING; then
    step "Restarting server"
    if ask_yn "  Restart PRE server? [Y/n]" "Y"; then
        cd "$WEB_DIR"
        nohup node server.js > /tmp/pre-server.log 2>&1 &
        sleep 2
        if curl -sf "http://localhost:${PRE_PORT}/api/status" >/dev/null 2>&1; then
            ok "Server restarted on port $PRE_PORT."
        else
            warn "Server may still be starting. Check http://localhost:$PRE_PORT"
        fi
    fi
fi

# -- Done -------------------------------------------------------------------

step "Update Complete!"

NEW_VERSION="unknown"
if [ -f "$REPO_DIR/VERSION" ]; then
    NEW_VERSION="$(cat "$REPO_DIR/VERSION" | tr -d '[:space:]')"
fi

echo ""
ok "PRE updated to version $NEW_VERSION"
echo ""
echo "  Changes in this update:"
if $IS_GIT; then
    echo "    git log --oneline -10"
else
    echo "    See: ${REPO_URL}/commits/main"
fi
echo ""
echo "  If you encounter issues after updating:"
echo "    1. Restart the server: cd web && node server.js"
echo "    2. Recreate the model: ollama create pre-gemma4 -f engine/Modelfile"
echo "    3. Report issues: ${REPO_URL}/issues"
echo ""
