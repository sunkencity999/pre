// PRE Web GUI — Computer Use tool (desktop automation)
// Uses screencapture for screenshots and cliclick for mouse/keyboard.
// Returns base64 screenshots after each action for the vision loop.
//
// When a target app is set (via open_app or window_focus), screenshots capture
// only that app's window and coordinates are window-relative. This dramatically
// improves accuracy by removing desktop clutter and simplifying the coordinate space.

const { execSync } = require('child_process');
const fs = require('fs');

// Check if cliclick is installed
let cliclickPath = null;
try {
  cliclickPath = execSync('which cliclick', { encoding: 'utf-8', timeout: 3000 }).trim();
} catch {}

// Track the target app for window-specific capture and auto-focus.
let _targetApp = null;
// Track the previously focused app so we can restore focus after computer use.
let _previousApp = null;
// Window offset: model outputs window-relative coords, we translate to screen-absolute for cliclick.
let _windowOffset = { x: 0, y: 0 };
// Coordinate scale factor: when screenshots are downscaled (large monitors), multiply model coords back up.
let _coordScale = 1.0;
// Max screenshot width sent to model — larger images are downscaled for better vision accuracy.
const MAX_CAPTURE_WIDTH = 1600;

// Click-loop detection: track recent click coordinates to warn/block when stuck.
// Each entry is { x, y, ts } — we check if the last N clicks hit the same area.
const _recentClicks = [];
const CLICK_HISTORY_SIZE = 8;
const CLICK_PROXIMITY_THRESHOLD = 50; // pixels — clicks within this distance count as "same spot"

function recordClick(x, y) {
  _recentClicks.push({ x: parseInt(x), y: parseInt(y), ts: Date.now() });
  if (_recentClicks.length > CLICK_HISTORY_SIZE) _recentClicks.shift();
}

/**
 * Check if the model is stuck in a click loop.
 * Returns { level: 'warn'|'block', message } if detected, null otherwise.
 * - 'warn' (3+ nearby): warning in tool output
 * - 'block' (5+ nearby): REFUSE to execute the click, force model to change approach
 */
function checkClickLoop(x, y) {
  if (_recentClicks.length < 3) return null;
  const cx = parseInt(x);
  const cy = parseInt(y);
  let nearby = 0;
  for (const click of _recentClicks) {
    const dist = Math.sqrt((click.x - cx) ** 2 + (click.y - cy) ** 2);
    if (dist < CLICK_PROXIMITY_THRESHOLD) nearby++;
  }
  if (nearby >= 5) {
    return {
      level: 'block',
      message: `ERROR: CLICK BLOCKED — You have clicked near (${cx}, ${cy}) ${nearby} times with no progress. This click was NOT executed. You MUST try a completely different approach NOW: use key action with "cmd+f" to open search, or type in a visible search field, or use keyboard shortcuts, or scroll, or report your findings to the user. Do NOT click this area again.`,
    };
  }
  if (nearby >= 3) {
    return {
      level: 'warn',
      message: `⚠️ WARNING: You have clicked near (${cx}, ${cy}) ${nearby} times. This area is not responding to clicks. Try a DIFFERENT approach: use key "cmd+f" to search, click a different UI element, or report what you see.`,
    };
  }
  return null;
}

function resetClickHistory() {
  _recentClicks.length = 0;
}

function setTargetApp(appName) {
  // Save the currently focused app so we can restore later
  if (!_targetApp && appName) {
    try {
      _previousApp = runAppleScript(
        'tell application "System Events" to get name of first application process whose frontmost is true'
      );
    } catch {}
  }
  _targetApp = appName;
  _windowOffset = { x: 0, y: 0 };
  _coordScale = 1.0;
  resetClickHistory();
}

function ensureTargetFocused() {
  if (_targetApp) {
    try {
      runAppleScript(`tell application "${_targetApp}" to activate`);
    } catch {}
  }
}

/**
 * Run an AppleScript via stdin to avoid shell quoting issues with app names
 * that contain spaces (e.g. "Microsoft Word").
 */
function runAppleScript(script, timeout = 5000) {
  return execSync('osascript', {
    input: script,
    encoding: 'utf-8',
    timeout,
  }).trim();
}

/**
 * Maximize the target app's front window to fill the screen.
 * Activates the app first, then uses the frontmost process name (avoids
 * mismatches between app name and System Events process name).
 */
function maximizeTargetWindow() {
  if (!_targetApp) return;
  try {
    // Activate the app so it becomes frontmost
    ensureTargetFocused();
    // Query the actual System Events process name of the frontmost app
    const processName = runAppleScript(
      'tell application "System Events" to get name of first application process whose frontmost is true'
    );
    if (!processName) return;
    const size = getScreenSize();
    const w = size.width;
    const h = size.height - 25;
    runAppleScript(`
      tell application "System Events"
        tell process "${processName}"
          set position of window 1 to {0, 25}
          set size of window 1 to {${w}, ${h}}
        end tell
      end tell
    `);
    console.log(`[computer] Maximized ${processName} to ${w}x${h}`);
  } catch (err) {
    console.log(`[computer] maximizeTargetWindow failed: ${err.message}`);
  }
}

/**
 * Restore focus to the app that was active before computer use started.
 * Used after task completion to bring back the browser/terminal.
 */
function restoreFocus() {
  if (_previousApp) {
    try {
      runAppleScript(`tell application "${_previousApp}" to activate`);
      console.log(`[computer] Restored focus to ${_previousApp}`);
    } catch {}
  }
  _previousApp = null;
  _targetApp = null;
}

/**
 * Get the target app's window bounds (position + size) in logical points.
 * Returns { x, y, width, height } or null if unavailable.
 */
function getWindowBounds(appName) {
  try {
    // Use stdin to avoid shell quoting issues with names containing spaces
    const out = runAppleScript(`
      tell application "System Events"
        tell process "${appName}"
          get {position, size} of window 1
        end tell
      end tell
    `);
    // Returns "x, y, w, h"
    const parts = out.split(',').map(s => parseInt(s.trim()));
    if (parts.length >= 4) {
      return { x: parts[0], y: parts[1], width: parts[2], height: parts[3] };
    }
  } catch {}
  return null;
}

function isAvailable() {
  return !!cliclickPath;
}

/**
 * Take a screenshot and return as base64 PNG.
 * If a target app is set, captures only that window (window-relative coordinates).
 * Otherwise captures the full main display.
 * On Retina displays, resizes to logical resolution so pixel coords match cliclick points.
 */
function takeScreenshot() {
  const outPath = `/tmp/pre_computer_${Date.now()}.png`;
  let logicalWidth, logicalHeight;

  if (_targetApp) {
    ensureTargetFocused();
    const bounds = getWindowBounds(_targetApp);
    if (bounds && bounds.width > 50 && bounds.height > 50) {
      // Capture just the target window region
      _windowOffset = { x: bounds.x, y: bounds.y };
      execSync(`screencapture -x -R ${bounds.x},${bounds.y},${bounds.width},${bounds.height} '${outPath}'`, { timeout: 10000 });
      logicalWidth = bounds.width;
      logicalHeight = bounds.height;
    } else {
      // Fallback to full screen if window bounds unavailable
      _windowOffset = { x: 0, y: 0 };
      execSync(`screencapture -x -D 1 '${outPath}'`, { timeout: 10000 });
      const size = getScreenSize();
      logicalWidth = size.width;
      logicalHeight = size.height;
    }
  } else {
    _windowOffset = { x: 0, y: 0 };
    execSync(`screencapture -x -D 1 '${outPath}'`, { timeout: 10000 });
    const size = getScreenSize();
    logicalWidth = size.width;
    logicalHeight = size.height;
  }

  // First resize to logical dimensions (Retina 2x → 1x), then downscale if still too large.
  // This ensures the model sees a manageable image and coordinates scale correctly.
  let finalWidth = logicalWidth;
  let finalHeight = logicalHeight;
  if (logicalWidth > MAX_CAPTURE_WIDTH) {
    _coordScale = logicalWidth / MAX_CAPTURE_WIDTH;
    finalWidth = MAX_CAPTURE_WIDTH;
    finalHeight = Math.round(logicalHeight / _coordScale);
  } else {
    _coordScale = 1.0;
  }
  try {
    execSync(`sips --resampleWidth ${finalWidth} '${outPath}' > /dev/null 2>&1`, { timeout: 5000 });
  } catch {}

  const buffer = fs.readFileSync(outPath);
  fs.unlinkSync(outPath);
  return { base64: buffer.toString('base64'), width: finalWidth, height: finalHeight };
}

/**
 * Get screen dimensions (logical points)
 */
function getScreenSize() {
  try {
    const out = execSync(
      "osascript -e 'tell application \"Finder\" to get bounds of window of desktop'",
      { encoding: 'utf-8', timeout: 5000 }
    ).trim();
    const parts = out.split(',').map(s => parseInt(s.trim()));
    if (parts.length >= 4) return { width: parts[2], height: parts[3] };
  } catch {}
  try {
    const out = execSync(
      "system_profiler SPDisplaysDataType 2>/dev/null | grep Resolution",
      { encoding: 'utf-8', timeout: 5000 }
    ).trim();
    const match = out.match(/(\d+)\s*x\s*(\d+)/);
    if (match) return { width: parseInt(match[1]), height: parseInt(match[2]) };
  } catch {}
  return { width: 1920, height: 1080 };
}

function run(cmd, timeout = 10000) {
  return execSync(cmd, { encoding: 'utf-8', timeout, maxBuffer: 64 * 1024 }).trim();
}

/**
 * Translate model coordinates to screen-absolute for cliclick.
 * Applies coordinate scaling (for downscaled screenshots) and window offset.
 */
function toScreen(x, y) {
  return {
    x: Math.round(parseInt(x) * _coordScale) + _windowOffset.x,
    y: Math.round(parseInt(y) * _coordScale) + _windowOffset.y,
  };
}

/**
 * Main computer use dispatcher
 */
async function computerUse(args) {
  const { action } = args;
  if (!action) return 'Error: action required (screenshot|click|double_click|right_click|type|key|scroll|move|drag|screen_size|cursor_position)';

  if (!cliclickPath && action !== 'screenshot' && action !== 'screen_size') {
    return 'Error: cliclick not installed. Run: brew install cliclick';
  }

  try {
  switch (action) {
    case 'screenshot': {
      const { base64, width, height } = takeScreenshot();
      const label = _targetApp ? `${_targetApp} window` : 'Desktop';
      return JSON.stringify({
        action: 'screenshot',
        screenshot: base64,
        screen: { width, height },
        message: `${label} screenshot (${width}x${height})`,
      });
    }

    case 'click': {
      const { x, y } = args;
      if (x === undefined || y === undefined) return 'Error: x and y coordinates required';
      const loopCheck = checkClickLoop(x, y);
      recordClick(x, y);
      // HARD BLOCK: refuse to execute after too many repeated clicks
      if (loopCheck && loopCheck.level === 'block') {
        const { base64 } = takeScreenshot();
        return JSON.stringify({
          action: 'click',
          x: parseInt(x), y: parseInt(y),
          screenshot: base64,
          blocked: true,
          message: loopCheck.message,
        });
      }
      ensureTargetFocused();
      const abs = toScreen(x, y);
      run(`${cliclickPath} c:${abs.x},${abs.y}`);
      await sleep(300);
      const { base64 } = takeScreenshot();
      const msg = loopCheck
        ? `Clicked at (${x}, ${y}). ${loopCheck.message}`
        : `Clicked at (${x}, ${y})`;
      return JSON.stringify({
        action: 'click',
        x: parseInt(x), y: parseInt(y),
        screenshot: base64,
        message: msg,
      });
    }

    case 'double_click': {
      const { x, y } = args;
      if (x === undefined || y === undefined) return 'Error: x and y coordinates required';
      const loopCheck = checkClickLoop(x, y);
      recordClick(x, y);
      if (loopCheck && loopCheck.level === 'block') {
        const { base64 } = takeScreenshot();
        return JSON.stringify({
          action: 'double_click',
          x: parseInt(x), y: parseInt(y),
          screenshot: base64,
          blocked: true,
          message: loopCheck.message,
        });
      }
      ensureTargetFocused();
      const abs = toScreen(x, y);
      run(`${cliclickPath} dc:${abs.x},${abs.y}`);
      await sleep(300);
      const { base64 } = takeScreenshot();
      const msg = loopCheck
        ? `Double-clicked at (${x}, ${y}). ${loopCheck.message}`
        : `Double-clicked at (${x}, ${y})`;
      return JSON.stringify({
        action: 'double_click',
        x: parseInt(x), y: parseInt(y),
        screenshot: base64,
        message: msg,
      });
    }

    case 'right_click': {
      const { x, y } = args;
      if (x === undefined || y === undefined) return 'Error: x and y coordinates required';
      const loopCheck = checkClickLoop(x, y);
      recordClick(x, y);
      if (loopCheck && loopCheck.level === 'block') {
        const { base64 } = takeScreenshot();
        return JSON.stringify({
          action: 'right_click',
          x: parseInt(x), y: parseInt(y),
          screenshot: base64,
          blocked: true,
          message: loopCheck.message,
        });
      }
      ensureTargetFocused();
      const abs = toScreen(x, y);
      run(`${cliclickPath} rc:${abs.x},${abs.y}`);
      await sleep(300);
      const { base64 } = takeScreenshot();
      const msg = loopWarning
        ? `Right-clicked at (${x}, ${y}). ${loopWarning}`
        : `Right-clicked at (${x}, ${y})`;
      return JSON.stringify({
        action: 'right_click',
        x: parseInt(x), y: parseInt(y),
        screenshot: base64,
        message: msg,
      });
    }

    case 'type': {
      const { text } = args;
      if (!text) return 'Error: text required';
      resetClickHistory(); // Typing = new approach, reset loop detection
      ensureTargetFocused();
      await sleep(150);

      // Normalize newlines: models often send literal "\n" (backslash + n) instead of
      // actual newline characters. Convert both forms to real newlines before splitting.
      const normalized = text.replace(/\\n/g, '\n').replace(/\\r/g, '\r');

      // Split on newlines — type each line, press Return between them.
      // cliclick t: cannot handle literal newlines in shell quotes.
      const lines = normalized.split(/\r?\n/);
      for (let li = 0; li < lines.length; li++) {
        const line = lines[li];
        if (line.length > 0) {
          // Chunk long lines to avoid shell command length limits and app lag
          const chunks = line.length > 5 ? (line.match(/.{1,4}/g) || [line]) : [line];
          const cmd = chunks.map(c => `t:'${c.replace(/'/g, "'\\''")}'`).join(' w:50 ');
          run(`${cliclickPath} ${cmd}`, 30000);
        }
        // Press Return between lines (not after the last one)
        if (li < lines.length - 1) {
          run(`${cliclickPath} kp:return`, 5000);
          await sleep(100);
        }
      }

      await sleep(200);
      const { base64 } = takeScreenshot();
      return JSON.stringify({
        action: 'type',
        text: text.slice(0, 50),
        screenshot: base64,
        message: `Typed "${text.slice(0, 50).replace(/\\n/g, '↵').replace(/\n/g, '↵')}"`,
      });
    }

    case 'key': {
      const { key } = args;
      if (!key) return 'Error: key required (e.g. return, tab, escape, space, delete, arrow-up, arrow-down, arrow-left, arrow-right, cmd+c, cmd+v)';
      resetClickHistory(); // Key press = new approach, reset loop detection
      ensureTargetFocused();
      const keyMap = {
        'enter': 'return', 'return': 'return',
        'tab': 'tab', 'escape': 'esc', 'esc': 'esc',
        'space': 'space', 'delete': 'delete', 'backspace': 'delete',
        'forward-delete': 'fwd-delete', 'fwd-delete': 'fwd-delete',
        'up': 'arrow-up', 'down': 'arrow-down',
        'left': 'arrow-left', 'right': 'arrow-right',
        'arrow-up': 'arrow-up', 'arrow-down': 'arrow-down',
        'arrow-left': 'arrow-left', 'arrow-right': 'arrow-right',
        'home': 'home', 'end': 'end',
        'page-up': 'page-up', 'page-down': 'page-down',
        'pageup': 'page-up', 'pagedown': 'page-down',
        'f1': 'f1', 'f2': 'f2', 'f3': 'f3', 'f4': 'f4',
        'f5': 'f5', 'f6': 'f6', 'f7': 'f7', 'f8': 'f8',
        'f9': 'f9', 'f10': 'f10', 'f11': 'f11', 'f12': 'f12',
        'f13': 'f13', 'f14': 'f14', 'f15': 'f15', 'f16': 'f16',
        'mute': 'mute', 'volume-up': 'volume-up', 'volume-down': 'volume-down',
        'play-pause': 'play-pause', 'play-next': 'play-next', 'play-previous': 'play-previous',
      };

      const isSpecialKey = (k) => Object.values(keyMap).includes(k) || /^(num-\d|keys-light|brightness)/.test(k);

      const parts = key.toLowerCase().split('+').map(s => s.trim());
      if (parts.length > 1) {
        const modifiers = [];
        let mainKey = parts[parts.length - 1];
        for (let i = 0; i < parts.length - 1; i++) {
          const mod = parts[i];
          if (mod === 'cmd' || mod === 'command') modifiers.push('cmd');
          else if (mod === 'ctrl' || mod === 'control') modifiers.push('ctrl');
          else if (mod === 'alt' || mod === 'option') modifiers.push('alt');
          else if (mod === 'shift') modifiers.push('shift');
        }
        mainKey = keyMap[mainKey] || mainKey;
        const modStr = modifiers.join(',');
        const keyCmd = isSpecialKey(mainKey) ? `kp:${mainKey}` : `t:${mainKey}`;
        run(`${cliclickPath} kd:${modStr} ${keyCmd} ku:${modStr}`);
      } else {
        const mapped = keyMap[key.toLowerCase()] || key.toLowerCase();
        if (isSpecialKey(mapped)) {
          run(`${cliclickPath} kp:${mapped}`);
        } else {
          run(`${cliclickPath} t:${mapped}`);
        }
      }
      await sleep(300);
      const { base64, width, height } = takeScreenshot();
      return JSON.stringify({
        action: 'key',
        key,
        screenshot: base64,
        message: `Pressed ${key}`,
      });
    }

    case 'scroll': {
      resetClickHistory(); // Scrolling = new approach, reset loop detection
      ensureTargetFocused();
      const { x, y, direction, amount } = args;
      const clicks = parseInt(amount) || 3;
      if (x !== undefined && y !== undefined) {
        const abs = toScreen(x, y);
        run(`${cliclickPath} m:${abs.x},${abs.y}`);
      }
      const dir = direction === 'up' ? clicks : -clicks;
      const script = `tell application "System Events" to scroll area 1 by ${dir}`;
      try {
        run(`osascript -e '${script.replace(/'/g, "'\\''")}'`);
      } catch {
        const keyDir = direction === 'up' ? 'arrow-up' : 'arrow-down';
        for (let i = 0; i < Math.abs(clicks); i++) {
          run(`${cliclickPath} kp:${keyDir}`);
        }
      }
      await sleep(300);
      const { base64, width, height } = takeScreenshot();
      return JSON.stringify({
        action: 'scroll',
        direction: direction || 'down',
        amount: clicks,
        screenshot: base64,
        message: `Scrolled ${direction || 'down'} ${clicks} units`,
      });
    }

    case 'move': {
      const { x, y } = args;
      if (x === undefined || y === undefined) return 'Error: x and y coordinates required';
      const abs = toScreen(x, y);
      run(`${cliclickPath} m:${abs.x},${abs.y}`);
      const { base64, width, height } = takeScreenshot();
      return JSON.stringify({
        action: 'move',
        x: parseInt(x), y: parseInt(y),
        screenshot: base64,
        message: `Moved cursor to (${x}, ${y})`,
      });
    }

    case 'drag': {
      ensureTargetFocused();
      const { from_x, from_y, to_x, to_y } = args;
      if (from_x === undefined || from_y === undefined || to_x === undefined || to_y === undefined) {
        return 'Error: from_x, from_y, to_x, and to_y required';
      }
      const absFrom = toScreen(from_x, from_y);
      const absTo = toScreen(to_x, to_y);
      run(`${cliclickPath} dd:${absFrom.x},${absFrom.y} du:${absTo.x},${absTo.y}`);
      await sleep(300);
      const { base64, width, height } = takeScreenshot();
      return JSON.stringify({
        action: 'drag',
        from: { x: parseInt(from_x), y: parseInt(from_y) },
        to: { x: parseInt(to_x), y: parseInt(to_y) },
        screenshot: base64,
        message: `Dragged from (${from_x}, ${from_y}) to (${to_x}, ${to_y})`,
      });
    }

    case 'screen_size': {
      const size = getScreenSize();
      return JSON.stringify({
        action: 'screen_size',
        ...size,
        message: `Screen size: ${size.width}x${size.height}`,
      });
    }

    case 'cursor_position': {
      const pos = run(`${cliclickPath} p:`);
      return JSON.stringify({
        action: 'cursor_position',
        position: pos,
        message: `Cursor at ${pos}`,
      });
    }

    default:
      return `Error: unknown computer action '${action}'. Available: screenshot, click, double_click, right_click, type, key, scroll, move, drag, screen_size, cursor_position`;
  }
  } catch (err) {
    const msg = (err.stderr || err.message || '').trim().split('\n')[0];
    return `Error: ${msg}`;
  }
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

module.exports = { computerUse, isAvailable, setTargetApp, maximizeTargetWindow, restoreFocus };
