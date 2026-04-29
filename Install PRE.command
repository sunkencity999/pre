#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  PRE — Personal Reasoning Engine                            ║
# ║  Double-click this file in Finder to install.               ║
# ╚══════════════════════════════════════════════════════════════╝
#
# This wrapper makes the install script launchable from Finder.
# It opens in Terminal, navigates to the correct directory, and
# runs install.sh with all the same checks and steps.

# Navigate to the directory containing this script
cd "$(dirname "$0")" || exit 1

# Clear the terminal for a clean look
clear

echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║                                                      ║"
echo "  ║   ██████╗ ██████╗ ███████╗                           ║"
echo "  ║   ██╔══██╗██╔══██╗██╔════╝                           ║"
echo "  ║   ██████╔╝██████╔╝█████╗                             ║"
echo "  ║   ██╔═══╝ ██╔══██╗██╔══╝                             ║"
echo "  ║   ██║     ██║  ██║███████╗                            ║"
echo "  ║   ╚═╝     ╚═╝  ╚═╝╚══════╝                           ║"
echo "  ║                                                      ║"
echo "  ║   Personal Reasoning Engine                          ║"
echo "  ║   Local AI Agent for Apple Silicon                   ║"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""
echo "  This installer will set up PRE on your Mac."
echo ""
echo "  What it does:"
echo "    • Checks system requirements (Apple Silicon, RAM, disk)"
echo "    • Installs Ollama (if needed)"
echo "    • Downloads Gemma 4 26B model (~28 GB)"
echo "    • Builds the PRE binary"
echo "    • Sets up the Web GUI"
echo "    • Optionally installs local image generation"
echo ""
echo "  Estimated time: 5-20 minutes (mostly model download)"
echo ""
read -p "  Ready to install? [Y/n] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo "  Installation cancelled. You can run this again anytime."
    echo ""
    echo "  Press any key to close..."
    read -n 1 -s
    exit 0
fi

echo ""

# Run the actual install script
if [ -f "./install.sh" ]; then
    bash ./install.sh
    EXIT_CODE=$?
else
    echo "  Error: install.sh not found in $(pwd)"
    echo "  Make sure this file is in the PRE project root directory."
    EXIT_CODE=1
fi

# Keep terminal open so user can read the output
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ Installation complete. You can close this window."
else
    echo "  ✗ Installation encountered an error. Review the output above."
fi
echo ""
echo "  Press any key to close..."
read -n 1 -s
