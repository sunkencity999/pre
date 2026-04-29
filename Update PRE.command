#!/bin/bash
# Update PRE.command - Double-click in Finder to update PRE
# This is the macOS double-click launcher for the update script.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UPDATE_SCRIPT="$SCRIPT_DIR/update.sh"

if [ ! -f "$UPDATE_SCRIPT" ]; then
    echo "ERROR: update.sh not found in $SCRIPT_DIR"
    echo "Press Enter to exit..."
    read -r
    exit 1
fi

bash "$UPDATE_SCRIPT"
