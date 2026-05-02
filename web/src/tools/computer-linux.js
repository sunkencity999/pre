// PRE Web GUI — Linux Desktop Automation Helpers (X11)
// xdotool for mouse/keyboard, scrot for screenshots, xdpyinfo for display info.
// Called by computer.js when running on Linux.

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

function run(cmd, timeout = 10000) {
  try {
    return execSync(cmd, { encoding: 'utf-8', timeout, maxBuffer: 128 * 1024 }).trim();
  } catch (err) {
    return (err.stderr || err.message || '').trim();
  }
}

// ── Availability ──────────────────────────────────────────────────────────

let _xdotoolPath = null;
let _scrotPath = null;

try { _xdotoolPath = execSync('which xdotool', { encoding: 'utf-8', timeout: 3000 }).trim(); } catch {}
try { _scrotPath = execSync('which scrot', { encoding: 'utf-8', timeout: 3000 }).trim(); } catch {}

/**
 * Check if desktop automation is available on Linux.
 * Requires xdotool (mouse/keyboard) and scrot (screenshots).
 */
function isAvailable() {
  return !!_xdotoolPath && !!_scrotPath;
}

// ── Screen Size ───────────────────────────────────────────────────────────

function getScreenSize() {
  try {
    const out = run('xdotool getdisplaygeometry');
    const parts = out.split(/\s+/).map(Number);
    if (parts.length >= 2 && parts[0] > 0) return { width: parts[0], height: parts[1] };
  } catch {}
  // Fallback: xdpyinfo
  try {
    const out = run("xdpyinfo 2>/dev/null | grep 'dimensions:'");
    const match = out.match(/(\d+)x(\d+)/);
    if (match) return { width: parseInt(match[1]), height: parseInt(match[2]) };
  } catch {}
  return { width: 1920, height: 1080 };
}

// ── Screenshot ────────────────────────────────────────────────────────────

/**
 * Take a screenshot using scrot.
 * @param {Object|null} region - { x, y, width, height } or null for full screen
 * @param {number} maxWidth - Max screenshot width (downscale if larger)
 * @returns {{ base64: string, width: number, height: number, scale: number }}
 */
function takeScreenshot(region, maxWidth = 1600) {
  const outPath = path.join(os.tmpdir(), `pre_computer_${Date.now()}.png`);
  let captureW, captureH;

  if (region && region.width > 0 && region.height > 0) {
    // scrot region capture: --select is interactive, use import (ImageMagick) for non-interactive region
    try {
      run(`import -window root -crop ${region.width}x${region.height}+${region.x}+${region.y} '${outPath}'`, 10000);
      captureW = region.width;
      captureH = region.height;
    } catch {
      // Fallback: full screen capture then crop
      run(`scrot -o '${outPath}'`, 10000);
      const size = getScreenSize();
      captureW = size.width;
      captureH = size.height;
    }
  } else {
    run(`scrot -o '${outPath}'`, 10000);
    const size = getScreenSize();
    captureW = size.width;
    captureH = size.height;
  }

  // Downscale if needed (requires ImageMagick convert)
  let scale = 1.0;
  let finalW = captureW;
  let finalH = captureH;

  if (captureW > maxWidth) {
    scale = captureW / maxWidth;
    finalW = maxWidth;
    finalH = Math.round(captureH / scale);
    try {
      run(`convert '${outPath}' -resize ${finalW}x${finalH} '${outPath}'`, 10000);
    } catch {}
  }

  let base64 = '';
  try {
    const buf = fs.readFileSync(outPath);
    base64 = buf.toString('base64');
    fs.unlinkSync(outPath);
  } catch {}

  return { base64, width: finalW, height: finalH, scale };
}

// ── Mouse Actions ─────────────────────────────────────────────────────────

/**
 * Move cursor, click, double-click, or right-click using xdotool.
 * @param {string} action - 'click' | 'double_click' | 'right_click' | 'move'
 * @param {number} x - Screen X coordinate
 * @param {number} y - Screen Y coordinate
 */
function mouseAction(action, x, y) {
  const ix = Math.round(x);
  const iy = Math.round(y);

  switch (action) {
    case 'click':
      run(`xdotool mousemove ${ix} ${iy} click 1`);
      break;
    case 'double_click':
      run(`xdotool mousemove ${ix} ${iy} click --repeat 2 --delay 50 1`);
      break;
    case 'right_click':
      run(`xdotool mousemove ${ix} ${iy} click 3`);
      break;
    case 'move':
      run(`xdotool mousemove ${ix} ${iy}`);
      break;
  }
}

/**
 * Drag from one position to another.
 */
function mouseDrag(fromX, fromY, toX, toY) {
  run(`xdotool mousemove ${Math.round(fromX)} ${Math.round(fromY)} mousedown 1`);
  run(`xdotool mousemove --delay 100 ${Math.round(toX)} ${Math.round(toY)} mouseup 1`);
}

/**
 * Scroll the mouse wheel.
 * @param {number} x - Screen X coordinate
 * @param {number} y - Screen Y coordinate
 * @param {string} direction - 'up' or 'down'
 * @param {number} amount - Number of scroll ticks
 */
function mouseScroll(x, y, direction, amount) {
  run(`xdotool mousemove ${Math.round(x)} ${Math.round(y)}`);
  // xdotool: button 4 = scroll up, button 5 = scroll down
  const button = direction === 'up' ? 4 : 5;
  run(`xdotool click --repeat ${amount} --delay 50 ${button}`);
}

// ── Keyboard Actions ──────────────────────────────────────────────────────

/**
 * Type text using xdotool.
 * @param {string} text - Text to type
 */
function typeText(text) {
  // xdotool type handles most characters; use --clearmodifiers to avoid modifier interference
  const escaped = text.replace(/'/g, "'\\''");
  run(`xdotool type --clearmodifiers --delay 12 '${escaped}'`, 30000);
}

/**
 * Press a key or key combination.
 * @param {string} key - Key name or combo like 'ctrl+c', 'enter', 'f5'
 */
function pressKey(key) {
  // Map common key names to xdotool key names
  const keyMap = {
    'enter': 'Return', 'return': 'Return',
    'tab': 'Tab', 'escape': 'Escape', 'esc': 'Escape',
    'space': 'space', 'delete': 'Delete', 'backspace': 'BackSpace', 'bs': 'BackSpace',
    'up': 'Up', 'down': 'Down', 'left': 'Left', 'right': 'Right',
    'arrow-up': 'Up', 'arrow-down': 'Down', 'arrow-left': 'Left', 'arrow-right': 'Right',
    'home': 'Home', 'end': 'End',
    'page-up': 'Prior', 'page-down': 'Next', 'pageup': 'Prior', 'pagedown': 'Next',
    'insert': 'Insert',
    'f1': 'F1', 'f2': 'F2', 'f3': 'F3', 'f4': 'F4',
    'f5': 'F5', 'f6': 'F6', 'f7': 'F7', 'f8': 'F8',
    'f9': 'F9', 'f10': 'F10', 'f11': 'F11', 'f12': 'F12',
  };

  const modMap = {
    'ctrl': 'ctrl', 'control': 'ctrl', 'cmd': 'ctrl', 'command': 'ctrl',
    'alt': 'alt', 'option': 'alt',
    'shift': 'shift', 'super': 'super', 'meta': 'super',
  };

  const parts = key.toLowerCase().split('+').map(s => s.trim());

  if (parts.length > 1) {
    const mods = [];
    const mainKey = parts[parts.length - 1];
    for (let i = 0; i < parts.length - 1; i++) {
      const mapped = modMap[parts[i]];
      if (mapped) mods.push(mapped);
    }
    const xKey = keyMap[mainKey] || mainKey;
    const combo = [...mods, xKey].join('+');
    run(`xdotool key --clearmodifiers ${combo}`);
  } else {
    const mapped = keyMap[key.toLowerCase()] || key;
    run(`xdotool key --clearmodifiers ${mapped}`);
  }
}

// ── Window Management ─────────────────────────────────────────────────────

/**
 * Get window bounds by searching for a window matching appName.
 * @param {string} appName - Application name or window title substring
 * @returns {{ x: number, y: number, width: number, height: number }|null}
 */
function getWindowBounds(appName) {
  try {
    // Find window ID by name
    const wid = run(`xdotool search --name '${appName.replace(/'/g, "'\\''")}' | head -1`);
    if (!wid || !/^\d+$/.test(wid)) return null;

    // Get geometry using xdotool
    const geo = run(`xdotool getwindowgeometry --shell ${wid}`);
    const xMatch = geo.match(/X=(\d+)/);
    const yMatch = geo.match(/Y=(\d+)/);
    const wMatch = geo.match(/WIDTH=(\d+)/);
    const hMatch = geo.match(/HEIGHT=(\d+)/);

    if (xMatch && yMatch && wMatch && hMatch) {
      return {
        x: parseInt(xMatch[1]),
        y: parseInt(yMatch[1]),
        width: parseInt(wMatch[1]),
        height: parseInt(hMatch[1]),
      };
    }
  } catch {}
  return null;
}

/**
 * Focus a window by name.
 */
function focusWindow(appName) {
  try {
    const wid = run(`xdotool search --name '${appName.replace(/'/g, "'\\''")}' | head -1`);
    if (wid && /^\d+$/.test(wid)) {
      run(`xdotool windowactivate ${wid}`);
    }
  } catch {}
}

/**
 * Get the currently focused window's name.
 */
function getForegroundApp() {
  try {
    const wid = run('xdotool getactivewindow');
    if (wid && /^\d+$/.test(wid)) {
      return run(`xdotool getactivewindow getwindowname`);
    }
  } catch {}
  return '';
}

/**
 * Get cursor position.
 */
function getCursorPosition() {
  try {
    const out = run('xdotool getmouselocation --shell');
    const xMatch = out.match(/X=(\d+)/);
    const yMatch = out.match(/Y=(\d+)/);
    if (xMatch && yMatch) return `${xMatch[1]},${yMatch[1]}`;
  } catch {}
  return '0,0';
}

module.exports = {
  isAvailable,
  getScreenSize,
  takeScreenshot,
  mouseAction,
  mouseDrag,
  mouseScroll,
  typeText,
  pressKey,
  getWindowBounds,
  focusWindow,
  getForegroundApp,
  getCursorPosition,
};
