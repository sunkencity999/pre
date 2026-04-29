// PRE Web GUI — Windows Desktop Automation Helpers
// PowerShell/.NET implementations for screenshot, mouse, keyboard, and window management.
// Called by computer.js when running on Windows.

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

function run(cmd, timeout = 10000) {
  try {
    return execSync(cmd, { encoding: 'utf-8', timeout, maxBuffer: 128 * 1024, windowsHide: true }).trim();
  } catch (err) {
    return (err.stderr || err.message || '').trim();
  }
}

function ps(script, timeout = 15000) {
  return run(`powershell.exe -NoProfile -Command "${script.replace(/"/g, '\\"')}"`, timeout);
}

// ── Availability ──────────────────────────────────────────────────────────

/**
 * Check if desktop automation is available on Windows.
 * Requires System.Windows.Forms + user32.dll (built into Windows 10+).
 */
function isAvailable() {
  try {
    const result = ps("Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width");
    return /^\d+$/.test(result.trim());
  } catch {
    return false;
  }
}

// ── Screen Size ───────────────────────────────────────────────────────────

function getScreenSize() {
  try {
    const out = ps("Add-Type -AssemblyName System.Windows.Forms; $s = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; Write-Output \"$($s.Width),$($s.Height)\"");
    const parts = out.split(',').map(Number);
    if (parts.length >= 2 && parts[0] > 0) return { width: parts[0], height: parts[1] };
  } catch {}
  return { width: 1920, height: 1080 };
}

// ── Screenshot ────────────────────────────────────────────────────────────

/**
 * Take a screenshot using System.Drawing + System.Windows.Forms.
 * @param {Object|null} region - { x, y, width, height } or null for full screen
 * @param {number} maxWidth - Max screenshot width (downscale if larger)
 * @returns {{ base64: string, width: number, height: number, scale: number }}
 */
function takeScreenshot(region, maxWidth = 1600) {
  const outPath = path.join(os.tmpdir(), `pre_computer_${Date.now()}.png`).replace(/\//g, '\\');
  const safeOutPath = outPath.replace(/'/g, "''");

  let script;
  if (region && region.width > 0 && region.height > 0) {
    script = `
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
$bmp = New-Object System.Drawing.Bitmap(${region.width}, ${region.height})
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen(${region.x}, ${region.y}, 0, 0, $bmp.Size)
$g.Dispose()
$bmp.Save('${safeOutPath}', [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
Write-Output '${region.width},${region.height}'
`.trim().replace(/\n/g, '; ');
  } else {
    script = `
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
$s = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($s.Width, $s.Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($s.Location, [System.Drawing.Point]::Empty, $s.Size)
$g.Dispose()
$bmp.Save('${safeOutPath}', [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
Write-Output "$($s.Width),$($s.Height)"
`.trim().replace(/\n/g, '; ');
  }

  const out = ps(script, 15000);
  const parts = out.split(',').map(Number);
  const captureW = parts[0] || 1920;
  const captureH = parts[1] || 1080;

  // Read and optionally downscale
  let scale = 1.0;
  let finalW = captureW;
  let finalH = captureH;

  if (captureW > maxWidth) {
    scale = captureW / maxWidth;
    finalW = maxWidth;
    finalH = Math.round(captureH / scale);
    // Resize using System.Drawing
    const resizeScript = `
Add-Type -AssemblyName System.Drawing
$src = [System.Drawing.Image]::FromFile('${safeOutPath}')
$dst = New-Object System.Drawing.Bitmap(${finalW}, ${finalH})
$g = [System.Drawing.Graphics]::FromImage($dst)
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$g.DrawImage($src, 0, 0, ${finalW}, ${finalH})
$g.Dispose()
$src.Dispose()
$dst.Save('${safeOutPath}', [System.Drawing.Imaging.ImageFormat]::Png)
$dst.Dispose()
`.trim().replace(/\n/g, '; ');
    try { ps(resizeScript, 10000); } catch {}
  }

  let base64 = '';
  try {
    const buf = fs.readFileSync(outPath.replace(/\\/g, '/'));
    base64 = buf.toString('base64');
    fs.unlinkSync(outPath.replace(/\\/g, '/'));
  } catch {}

  return { base64, width: finalW, height: finalH, scale };
}

// ── Mouse Actions ─────────────────────────────────────────────────────────

/**
 * Move cursor, click, double-click, or right-click using user32.dll P/Invoke.
 * @param {string} action - 'click' | 'double_click' | 'right_click' | 'move'
 * @param {number} x - Screen X coordinate
 * @param {number} y - Screen Y coordinate
 */
function mouseAction(action, x, y) {
  const ix = Math.round(x);
  const iy = Math.round(y);

  // Build the P/Invoke and action script
  const pInvoke = `
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class WinInput {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint data, int extra);
  public const uint LDOWN = 0x02, LUP = 0x04, RDOWN = 0x08, RUP = 0x10;
  public static void Click(int x, int y) { SetCursorPos(x, y); mouse_event(LDOWN, 0, 0, 0, 0); mouse_event(LUP, 0, 0, 0, 0); }
  public static void DoubleClick(int x, int y) { SetCursorPos(x, y); mouse_event(LDOWN,0,0,0,0); mouse_event(LUP,0,0,0,0); System.Threading.Thread.Sleep(50); mouse_event(LDOWN,0,0,0,0); mouse_event(LUP,0,0,0,0); }
  public static void RightClick(int x, int y) { SetCursorPos(x, y); mouse_event(RDOWN, 0, 0, 0, 0); mouse_event(RUP, 0, 0, 0, 0); }
  public static void Move(int x, int y) { SetCursorPos(x, y); }
}
'@
`.trim().replace(/\n/g, '; ');

  let call;
  switch (action) {
    case 'click': call = `[WinInput]::Click(${ix}, ${iy})`; break;
    case 'double_click': call = `[WinInput]::DoubleClick(${ix}, ${iy})`; break;
    case 'right_click': call = `[WinInput]::RightClick(${ix}, ${iy})`; break;
    case 'move': call = `[WinInput]::Move(${ix}, ${iy})`; break;
    default: return;
  }

  ps(`${pInvoke}; ${call}`);
}

/**
 * Drag from one position to another.
 */
function mouseDrag(fromX, fromY, toX, toY) {
  const script = `
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class WinDrag {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, int e);
  public const uint LDOWN = 0x02, LUP = 0x04;
}
'@
[WinDrag]::SetCursorPos(${Math.round(fromX)}, ${Math.round(fromY)})
[WinDrag]::mouse_event([WinDrag]::LDOWN, 0, 0, 0, 0)
Start-Sleep -Milliseconds 100
[WinDrag]::SetCursorPos(${Math.round(toX)}, ${Math.round(toY)})
Start-Sleep -Milliseconds 100
[WinDrag]::mouse_event([WinDrag]::LUP, 0, 0, 0, 0)
`.trim().replace(/\n/g, '; ');

  ps(script);
}

/**
 * Scroll the mouse wheel.
 * @param {number} x - Screen X coordinate
 * @param {number} y - Screen Y coordinate
 * @param {string} direction - 'up' or 'down'
 * @param {number} amount - Number of scroll ticks
 */
function mouseScroll(x, y, direction, amount) {
  const delta = direction === 'up' ? (amount * 120) : -(amount * 120);
  const script = `
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class WinScroll {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, int e);
  public const uint WHEEL = 0x0800;
}
'@
[WinScroll]::SetCursorPos(${Math.round(x)}, ${Math.round(y)})
[WinScroll]::mouse_event([WinScroll]::WHEEL, 0, 0, ${delta}, 0)
`.trim().replace(/\n/g, '; ');

  ps(script);
}

// ── Keyboard Actions ──────────────────────────────────────────────────────

/**
 * Type text using SendKeys.
 * @param {string} text - Text to type
 */
function typeText(text) {
  // Escape SendKeys special characters: +^%~{}[]()
  const escaped = text.replace(/([+^%~{}[\]()])/g, '{$1}');
  // Split into chunks to avoid command length limits
  const chunks = escaped.match(/.{1,200}/g) || [escaped];
  for (const chunk of chunks) {
    const safeChunk = chunk.replace(/'/g, "''");
    ps(`Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('${safeChunk}')`);
  }
}

/**
 * Press a key or key combination.
 * @param {string} key - Key name or combo like 'ctrl+c', 'enter', 'f5'
 */
function pressKey(key) {
  // Map key names to SendKeys syntax
  const keyMap = {
    'enter': '{ENTER}', 'return': '{ENTER}',
    'tab': '{TAB}', 'escape': '{ESC}', 'esc': '{ESC}',
    'space': ' ', 'delete': '{DELETE}', 'backspace': '{BACKSPACE}', 'bs': '{BACKSPACE}',
    'up': '{UP}', 'down': '{DOWN}', 'left': '{LEFT}', 'right': '{RIGHT}',
    'arrow-up': '{UP}', 'arrow-down': '{DOWN}', 'arrow-left': '{LEFT}', 'arrow-right': '{RIGHT}',
    'home': '{HOME}', 'end': '{END}',
    'page-up': '{PGUP}', 'page-down': '{PGDN}', 'pageup': '{PGUP}', 'pagedown': '{PGDN}',
    'insert': '{INSERT}',
    'f1': '{F1}', 'f2': '{F2}', 'f3': '{F3}', 'f4': '{F4}',
    'f5': '{F5}', 'f6': '{F6}', 'f7': '{F7}', 'f8': '{F8}',
    'f9': '{F9}', 'f10': '{F10}', 'f11': '{F11}', 'f12': '{F12}',
  };

  const parts = key.toLowerCase().split('+').map(s => s.trim());

  if (parts.length > 1) {
    // Key combination: ctrl+c, alt+f4, shift+enter, etc.
    let sendStr = '';
    const mainKey = parts[parts.length - 1];
    for (let i = 0; i < parts.length - 1; i++) {
      const mod = parts[i];
      if (mod === 'ctrl' || mod === 'control' || mod === 'cmd' || mod === 'command') sendStr += '^';
      else if (mod === 'alt' || mod === 'option') sendStr += '%';
      else if (mod === 'shift') sendStr += '+';
    }
    sendStr += keyMap[mainKey] || mainKey;
    const safe = sendStr.replace(/'/g, "''");
    ps(`Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('${safe}')`);
  } else {
    const mapped = keyMap[key.toLowerCase()];
    if (mapped) {
      const safe = mapped.replace(/'/g, "''");
      ps(`Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('${safe}')`);
    } else {
      // Single character
      const safe = key.replace(/'/g, "''");
      ps(`Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('${safe}')`);
    }
  }
}

// ── Window Management ─────────────────────────────────────────────────────

/**
 * Get window bounds for a process by name.
 * @param {string} appName - Process name (without .exe)
 * @returns {{ x: number, y: number, width: number, height: number }|null}
 */
function getWindowBounds(appName) {
  const safeName = appName.replace(/'/g, "''").replace(/\.exe$/i, '');
  const script = `
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class WinRect {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT r);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
'@
$p = Get-Process -Name '${safeName}' -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne [IntPtr]::Zero } | Select-Object -First 1
if ($p) {
  $r = New-Object WinRect+RECT
  [WinRect]::GetWindowRect($p.MainWindowHandle, [ref]$r) | Out-Null
  Write-Output "$($r.Left),$($r.Top),$($r.Right - $r.Left),$($r.Bottom - $r.Top)"
}
`.trim().replace(/\n/g, '; ');

  try {
    const out = ps(script);
    const parts = out.split(',').map(Number);
    if (parts.length >= 4 && parts[2] > 0) {
      return { x: parts[0], y: parts[1], width: parts[2], height: parts[3] };
    }
  } catch {}
  return null;
}

/**
 * Focus a window by process name.
 */
function focusWindow(appName) {
  const safeName = appName.replace(/'/g, "''").replace(/\.exe$/i, '');
  const script = `
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class WinFocus {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
'@
$p = Get-Process -Name '${safeName}' -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne [IntPtr]::Zero } | Select-Object -First 1
if ($p) {
  [WinFocus]::ShowWindow($p.MainWindowHandle, 9) | Out-Null
  [WinFocus]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
}
`.trim().replace(/\n/g, '; ');

  ps(script);
}

/**
 * Get the currently focused window's process name.
 */
function getForegroundApp() {
  const script = `
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class WinFG {
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
}
'@
$hwnd = [WinFG]::GetForegroundWindow()
$pid = 0
[WinFG]::GetWindowThreadProcessId($hwnd, [ref]$pid) | Out-Null
(Get-Process -Id $pid -ErrorAction SilentlyContinue).ProcessName
`.trim().replace(/\n/g, '; ');

  try {
    return ps(script).trim();
  } catch {
    return '';
  }
}

/**
 * Get cursor position.
 */
function getCursorPosition() {
  try {
    const out = ps("Add-Type -AssemblyName System.Windows.Forms; $p = [System.Windows.Forms.Cursor]::Position; Write-Output \"$($p.X),$($p.Y)\"");
    return out.trim();
  } catch {
    return '0,0';
  }
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
