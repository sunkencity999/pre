// PRE Web GUI — Platform abstraction layer
// Centralizes OS detection and platform-specific helpers so tool files
// stay clean. Every macOS-specific call routes through this module.

const { execSync, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// ── Platform detection ────────────────────────────────────────────────────

const IS_WIN = process.platform === 'win32';
const IS_MAC = process.platform === 'darwin';
const IS_LINUX = process.platform === 'linux';

// ── Shell ─────────────────────────────────────────────────────────────────

/**
 * Returns the default shell command and args for the current platform.
 * Usage: execSync(cmd, { shell: getShell().cmd })
 *        spawn(getShell().cmd, [...getShell().args, command])
 */
function getShell() {
  if (IS_WIN) {
    return { cmd: 'powershell.exe', args: ['-NoProfile', '-Command'] };
  }
  if (IS_LINUX) {
    return { cmd: '/bin/bash', args: ['-c'] };
  }
  return { cmd: '/bin/zsh', args: ['-c'] };
}

/**
 * Returns the shell string suitable for execSync's `shell` option.
 */
function getShellPath() {
  return getShell().cmd;
}

// ── Chrome discovery ──────────────────────────────────────────────────────

function getChromePaths() {
  if (IS_WIN) {
    return [
      'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
      path.join(os.homedir(), 'AppData', 'Local', 'Google', 'Chrome', 'Application', 'chrome.exe'),
      'C:\\Program Files\\Chromium\\Application\\chrome.exe',
    ];
  }
  if (IS_LINUX) {
    return [
      '/usr/bin/google-chrome',
      '/usr/bin/google-chrome-stable',
      '/usr/bin/chromium-browser',
      '/usr/bin/chromium',
      '/snap/bin/chromium',
      '/opt/google/chrome/chrome',
    ];
  }
  return [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    path.join(os.homedir(), 'Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
  ];
}

// ── File manager reveal ───────────────────────────────────────────────────

function revealInFileManager(filePath) {
  const safe = filePath.replace(/"/g, '\\"');
  if (IS_WIN) {
    exec(`explorer.exe /select,"${safe.replace(/\//g, '\\')}"`);
  } else if (IS_LINUX) {
    // xdg-open opens the parent directory (no select-file equivalent on Linux)
    exec(`xdg-open "${path.dirname(safe)}"`);
  } else {
    exec(`open -R "${safe}"`);
  }
}

// ── Clipboard ─────────────────────────────────────────────────────────────

function clipboardRead() {
  try {
    if (IS_WIN) {
      return execSync('powershell.exe -NoProfile -Command "Get-Clipboard"', {
        encoding: 'utf-8', timeout: 5000,
      }).trim();
    }
    if (IS_LINUX) {
      return execSync('xclip -selection clipboard -o 2>/dev/null || xsel -b -o 2>/dev/null', {
        encoding: 'utf-8', timeout: 5000,
      }).trim();
    }
    return execSync('pbpaste 2>/dev/null', { encoding: 'utf-8', timeout: 5000 }).trim();
  } catch {
    return '';
  }
}

function clipboardWrite(text) {
  try {
    if (IS_WIN) {
      execSync('powershell.exe -NoProfile -Command "Set-Clipboard -Value $input"', {
        input: text, encoding: 'utf-8', timeout: 5000,
      });
    } else if (IS_LINUX) {
      execSync('xclip -selection clipboard 2>/dev/null || xsel -b -i 2>/dev/null', {
        input: text, encoding: 'utf-8', timeout: 5000,
      });
    } else {
      execSync('pbcopy', { input: text, encoding: 'utf-8', timeout: 5000 });
    }
    return true;
  } catch {
    return false;
  }
}

// ── Notifications ─────────────────────────────────────────────────────────

function notify(title, message) {
  const safeTitle = (title || '').replace(/"/g, '\\"').replace(/`/g, '');
  const safeMsg = (message || '').replace(/"/g, '\\"').replace(/`/g, '');
  try {
    if (IS_WIN) {
      // Use PowerShell BalloonTip as a universal fallback (works on all Windows 10+)
      execSync(`powershell.exe -NoProfile -Command "[void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms'); $n = New-Object System.Windows.Forms.NotifyIcon; $n.Icon = [System.Drawing.SystemIcons]::Information; $n.Visible = $true; $n.ShowBalloonTip(5000, '${safeTitle}', '${safeMsg}', 'Info'); Start-Sleep -Milliseconds 100; $n.Dispose()"`, {
        timeout: 10000, windowsHide: true,
      });
    } else if (IS_LINUX) {
      execSync(`notify-send "${safeTitle}" "${safeMsg}" 2>/dev/null`, { timeout: 5000 });
    } else {
      execSync(`osascript -e 'display notification "${safeMsg}" with title "${safeTitle}"'`, { timeout: 5000 });
    }
  } catch { /* non-critical */ }
}

// ── Open target (app, URL, file) ──────────────────────────────────────────

function openTarget(target) {
  const safe = target.replace(/"/g, '\\"');
  try {
    if (IS_WIN) {
      execSync(`start "" "${safe}"`, { shell: 'cmd.exe', timeout: 10000 });
    } else if (IS_LINUX) {
      execSync(`xdg-open "${safe}"`, { timeout: 10000 });
    } else {
      execSync(`open "${safe}"`, { timeout: 10000 });
    }
    return true;
  } catch {
    return false;
  }
}

// ── Command lookup (which / where) ────────────────────────────────────────

function whichCmd(name) {
  try {
    if (IS_WIN) {
      return execSync(`where ${name} 2>NUL`, { encoding: 'utf-8', timeout: 5000 }).trim().split('\n')[0];
    }
    return execSync(`which ${name} 2>/dev/null`, { encoding: 'utf-8', timeout: 5000 }).trim();
  } catch {
    return null;
  }
}

// ── HTML to text ──────────────────────────────────────────────────────────

/**
 * Convert HTML string to plain text.
 * macOS: pipes through `textutil`. Windows/fallback: regex tag stripping.
 */
function htmlToText(html) {
  if (!html) return '';

  // macOS — use textutil (best quality)
  if (IS_MAC) {
    try {
      return execSync('textutil -stdin -format html -convert txt -stdout 2>/dev/null', {
        input: html, encoding: 'utf-8', timeout: 10000, maxBuffer: 256 * 1024,
      }).trim();
    } catch { /* fall through to regex */ }
  }

  // Universal fallback — regex tag stripping
  return html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
    .replace(/<head[^>]*>[\s\S]*?<\/head>/gi, '')
    .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, '')
    .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, '')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n\n')
    .replace(/<\/div>/gi, '\n')
    .replace(/<\/h[1-6]>/gi, '\n\n')
    .replace(/<\/li>/gi, '\n')
    .replace(/<li[^>]*>/gi, '  - ')
    .replace(/<[^>]+>/g, '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// ── System information helpers ────────────────────────────────────────────

function run(cmd, timeout = 10000) {
  try {
    return execSync(cmd, { encoding: 'utf-8', timeout, maxBuffer: 64 * 1024 }).trim();
  } catch {
    return '';
  }
}

function getCpuInfo() {
  if (IS_WIN) {
    const name = run('powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"');
    const cores = os.cpus().length;
    return { name: name || `${cores}-core processor`, cores };
  }
  if (IS_LINUX) {
    const name = run("grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2") || '';
    const cores = os.cpus().length;
    return { name: name.trim() || `${cores}-core processor`, cores };
  }
  const name = run("sysctl -n machdep.cpu.brand_string 2>/dev/null");
  const cores = parseInt(run("sysctl -n hw.ncpu 2>/dev/null")) || os.cpus().length;
  return { name, cores };
}

function getCpuUsage() {
  if (IS_WIN) {
    const pct = run('powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Processor).LoadPercentage"');
    return parseFloat(pct) || 0;
  }
  // Unix: aggregate from ps, normalized by core count
  const ps = run("ps -A -o %cpu | awk '{s+=$1} END {print s}'");
  const cores = IS_LINUX ? os.cpus().length : (parseInt(run("sysctl -n hw.ncpu")) || 1);
  return (parseFloat(ps) || 0) / cores;
}

function getMemoryInfo() {
  const totalBytes = os.totalmem();
  const freeBytes = os.freemem();

  if (IS_WIN) {
    // os.freemem() on Windows returns available memory (includes standby cache)
    const usagePct = totalBytes > 0 ? ((1 - (freeBytes / totalBytes)) * 100) : 0;
    return {
      totalGB: (totalBytes / (1024 ** 3)).toFixed(1),
      usagePct: Math.round(usagePct * 10) / 10,
    };
  }

  if (IS_LINUX) {
    // Linux: parse /proc/meminfo for MemAvailable (includes reclaimable caches)
    try {
      const meminfo = run("cat /proc/meminfo 2>/dev/null");
      const totalKB = parseFloat((meminfo.match(/MemTotal:\s+(\d+)/) || [])[1]) || 0;
      const availKB = parseFloat((meminfo.match(/MemAvailable:\s+(\d+)/) || [])[1]) || 0;
      const totalB = totalKB * 1024;
      const usagePct = totalB > 0 ? ((1 - ((availKB * 1024) / totalB)) * 100) : 0;
      return {
        totalGB: (totalB / (1024 ** 3)).toFixed(1),
        usagePct: Math.round(usagePct * 10) / 10,
      };
    } catch {
      const usagePct = totalBytes > 0 ? ((1 - (freeBytes / totalBytes)) * 100) : 0;
      return {
        totalGB: (totalBytes / (1024 ** 3)).toFixed(1),
        usagePct: Math.round(usagePct * 10) / 10,
      };
    }
  }

  // macOS: count inactive + speculative pages as available (reclaimable under pressure)
  try {
    const pageSize = 16384;
    const vmstat = run("vm_stat 2>/dev/null");
    const free = parseFloat((vmstat.match(/Pages free:\s+(\d+)/) || [])[1]) || 0;
    const inactive = parseFloat((vmstat.match(/Pages inactive:\s+(\d+)/) || [])[1]) || 0;
    const speculative = parseFloat((vmstat.match(/Pages speculative:\s+(\d+)/) || [])[1]) || 0;
    const available = (free + inactive + speculative) * pageSize;
    const usagePct = totalBytes > 0 ? ((1 - (available / totalBytes)) * 100) : 0;
    return {
      totalGB: (totalBytes / (1024 ** 3)).toFixed(1),
      usagePct: Math.round(usagePct * 10) / 10,
    };
  } catch {
    // Fallback to Node.js os module
    const usagePct = totalBytes > 0 ? ((1 - (freeBytes / totalBytes)) * 100) : 0;
    return {
      totalGB: (totalBytes / (1024 ** 3)).toFixed(1),
      usagePct: Math.round(usagePct * 10) / 10,
    };
  }
}

function getDiskUsage() {
  if (IS_WIN) {
    // Get the system drive (usually C:)
    const drive = process.env.SystemDrive || 'C:';
    const info = run(`powershell.exe -NoProfile -Command "$d = Get-PSDrive ${drive.replace(':', '')}; Write-Output ('{0},{1}' -f $d.Used, ($d.Used + $d.Free))"`);
    const parts = info.split(',');
    if (parts.length === 2) {
      const used = parseFloat(parts[0]) || 0;
      const total = parseFloat(parts[1]) || 1;
      return Math.round((used / total) * 100);
    }
    return 0;
  }
  if (IS_LINUX) {
    const df = run("df -h / 2>/dev/null");
    const lastLine = df.trim().split('\n').pop();
    const pctMatch = (lastLine || '').match(/(\d+)%/);
    return pctMatch ? parseInt(pctMatch[1]) : 0;
  }
  // macOS: use Data volume (APFS root shows only system volume)
  const df = run("df -h /System/Volumes/Data 2>/dev/null || df -h /");
  const lastLine = df.trim().split('\n').pop();
  const pctMatch = (lastLine || '').match(/(\d+)%/);
  return pctMatch ? parseInt(pctMatch[1]) : 0;
}

function getBatteryInfo() {
  if (IS_WIN) {
    const pct = run('powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Battery).EstimatedChargeRemaining"');
    const status = run('powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus"');
    // BatteryStatus: 1=discharging, 2=AC, 3=charged, 4=low, 5=critical, 6-9=charging variants
    const percent = parseInt(pct) || null;
    const charging = ['2', '3', '6', '7', '8', '9'].includes(status);
    const state = !percent ? 'unknown' : charging ? 'charging' : 'discharging';
    return { percent, state };
  }
  if (IS_LINUX) {
    // Read from /sys/class/power_supply/BAT*
    try {
      const batDirs = fs.readdirSync('/sys/class/power_supply').filter(d => d.startsWith('BAT'));
      if (batDirs.length === 0) return { percent: null, state: 'unknown' };
      const batPath = `/sys/class/power_supply/${batDirs[0]}`;
      const capacity = run(`cat ${batPath}/capacity 2>/dev/null`);
      const statusRaw = run(`cat ${batPath}/status 2>/dev/null`).toLowerCase();
      const percent = parseInt(capacity) || null;
      const state = statusRaw.includes('discharging') ? 'discharging'
        : statusRaw.includes('charging') ? 'charging' : 'unknown';
      return { percent, state };
    } catch {
      return { percent: null, state: 'unknown' };
    }
  }
  // macOS
  const raw = run("pmset -g batt 2>/dev/null | tail -1");
  const pctMatch = raw.match(/(\d+)%/);
  const percent = pctMatch ? parseInt(pctMatch[1]) : null;
  const isDischarging = /discharging/.test(raw);
  const isCharging = /charging/.test(raw) && !isDischarging;
  const state = isDischarging ? 'discharging' : isCharging ? 'charging' : 'unknown';
  return { percent, state };
}

function getGpuInfo() {
  if (IS_WIN) {
    // Try NVIDIA first (most common for ML workloads)
    const nvidia = run('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>NUL');
    if (nvidia) return `GPU: ${nvidia}`;
    // Fallback to WMI
    const wmi = run('powershell.exe -NoProfile -Command "(Get-CimInstance Win32_VideoController).Name"');
    return wmi ? `GPU: ${wmi}` : '';
  }
  if (IS_LINUX) {
    // Try NVIDIA first (direct — no Docker needed on Linux)
    const nvidia = run('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null');
    if (nvidia) return `GPU: ${nvidia}`;
    // Fallback to lspci
    const lspci = run("lspci 2>/dev/null | grep -i 'vga\\|3d\\|display'");
    return lspci ? `GPU: ${lspci}` : '';
  }
  return run("system_profiler SPDisplaysDataType 2>/dev/null | grep -E 'Chipset|Chip|VRAM|Metal'");
}

// ── TTS (Text-to-Speech) ─────────────────────────────────────────────────

function hasTTS() {
  if (IS_WIN) {
    // Windows SAPI is always available on Windows 10+
    return true;
  }
  if (IS_LINUX) {
    return !!(whichCmd('espeak-ng') || whichCmd('espeak'));
  }
  return !!whichCmd('say');
}

function ttsSpeak(text, opts = {}) {
  if (!text) return { error: 'No text provided' };

  const safeText = text.replace(/[`$\\]/g, '').replace(/"/g, '\\"').slice(0, 5000);

  if (IS_WIN) {
    const voice = opts.voice || '';
    const rate = opts.rate || 0; // SAPI rate: -10 to 10 (0 = default)

    if (opts.output) {
      // Generate WAV file
      const outPath = opts.output.endsWith('.wav') ? opts.output : opts.output + '.wav';
      try {
        const voiceSelect = voice
          ? `$synth.SelectVoice('${voice}');`
          : '';
        execSync(`powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; ${voiceSelect} $synth.Rate = ${rate}; $synth.SetOutputToWaveFile('${outPath}'); $synth.Speak('${safeText}'); $synth.Dispose()"`, {
          timeout: 30000, windowsHide: true,
        });
        return { file: outPath, voice: voice || 'default', rate };
      } catch (err) {
        return { error: `TTS generation failed: ${err.message}` };
      }
    }

    // Speak directly
    try {
      const voiceSelect = voice
        ? `$synth.SelectVoice('${voice}');`
        : '';
      exec(`powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; ${voiceSelect} $synth.Rate = ${rate}; $synth.Speak('${safeText}'); $synth.Dispose()"`);
      return { spoken: true, voice: voice || 'default', rate, length: safeText.length };
    } catch (err) {
      return { error: `TTS failed: ${err.message}` };
    }
  }

  if (IS_LINUX) {
    // Linux — use espeak-ng (or espeak fallback)
    const ttsCmd = whichCmd('espeak-ng') ? 'espeak-ng' : 'espeak';
    const voice = opts.voice || 'en';
    const rate = opts.rate || 175; // espeak WPM (default ~175)

    if (opts.output) {
      const outPath = opts.output.endsWith('.wav') ? opts.output : opts.output + '.wav';
      try {
        execSync(`${ttsCmd} -v "${voice}" -s ${rate} -w "${outPath}" "${safeText}"`, { timeout: 30000 });
        return { file: outPath, voice, rate };
      } catch (err) {
        return { error: `TTS generation failed: ${err.message}` };
      }
    }

    exec(`${ttsCmd} -v "${voice}" -s ${rate} "${safeText}"`);
    return { spoken: true, voice, rate, length: safeText.length };
  }

  // macOS — use `say`
  const voice = opts.voice || 'Samantha';
  const rate = opts.rate || 185;

  if (opts.output) {
    const outPath = opts.output.endsWith('.aiff') ? opts.output : opts.output + '.aiff';
    try {
      execSync(`say -v "${voice}" -r ${rate} -o "${outPath}" "${safeText}"`, { timeout: 30000 });
      return { file: outPath, voice, rate };
    } catch (err) {
      return { error: `TTS generation failed: ${err.message}` };
    }
  }

  exec(`say -v "${voice}" -r ${rate} "${safeText}"`);
  return { spoken: true, voice, rate, length: safeText.length };
}

function ttsListVoices() {
  if (IS_WIN) {
    try {
      const output = run('powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name + \'|\' + $_.VoiceInfo.Culture.Name }"');
      const voices = output.split('\n').filter(Boolean).map(line => {
        const [name, locale] = line.split('|');
        return { name: name.trim(), locale: (locale || '').trim(), sample: '' };
      });
      const english = voices.filter(v => v.locale.startsWith('en'));
      return { voices: english, total: voices.length };
    } catch {
      return { voices: [], total: 0 };
    }
  }

  if (IS_LINUX) {
    // espeak-ng / espeak
    const ttsCmd = whichCmd('espeak-ng') ? 'espeak-ng' : 'espeak';
    try {
      const output = run(`${ttsCmd} --voices 2>/dev/null`, 15000);
      const voices = output.split('\n').filter(Boolean).slice(1).map(line => {
        // Format: Pty Language Age/Gender VoiceName File Other Languages
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 4) {
          return { name: parts[3], locale: parts[1] || '', sample: '' };
        }
        return null;
      }).filter(Boolean);
      const english = voices.filter(v => v.locale.startsWith('en'));
      return { voices: english, total: voices.length };
    } catch {
      return { voices: [], total: 0 };
    }
  }

  // macOS
  try {
    const output = run('say -v "?" 2>/dev/null', 15000);
    const voices = output.split('\n').filter(Boolean).map(line => {
      const match = line.match(/^(\S+)\s+(\S+)\s+#\s*(.*)$/);
      if (match) return { name: match[1], locale: match[2], sample: match[3] };
      return null;
    }).filter(Boolean);
    const english = voices.filter(v => v.locale.startsWith('en'));
    return { voices: english, total: voices.length };
  } catch {
    return { voices: [], total: 0 };
  }
}

// ── Pure-JS file search (replaces POSIX `find`) ──────────────────────────

/**
 * Recursive file search using Node.js fs. Cross-platform replacement for `find`.
 * @param {string} basePath - Directory to search
 * @param {string} pattern - Glob-like pattern (supports * wildcard)
 * @param {number} maxDepth - Maximum directory depth (default 5)
 * @param {number} maxResults - Maximum results to return (default 100)
 * @returns {string[]} Array of matching file paths
 */
function nodeGlob(basePath, pattern, maxDepth = 5, maxResults = 100) {
  const results = [];
  // Convert glob pattern to regex: * → [^/]*, ** → .*
  const regexStr = pattern
    .replace(/[.+^${}()|[\]\\]/g, '\\$&') // escape regex special chars (except * and ?)
    .replace(/\*\*/g, '{{GLOBSTAR}}')
    .replace(/\*/g, '[^/\\\\]*')
    .replace(/\?/g, '[^/\\\\]')
    .replace(/\{\{GLOBSTAR\}\}/g, '.*');
  const regex = new RegExp(regexStr, IS_WIN ? 'i' : '');

  function walk(dir, depth) {
    if (depth > maxDepth || results.length >= maxResults) return;
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (results.length >= maxResults) return;
      if (entry.name.startsWith('.')) continue;
      const fullPath = path.join(dir, entry.name);
      const relativePath = path.relative(basePath, fullPath);
      if (regex.test(relativePath) || regex.test(entry.name)) {
        results.push(fullPath);
      }
      if (entry.isDirectory()) {
        walk(fullPath, depth + 1);
      }
    }
  }

  walk(basePath, 0);
  return results;
}

// ── Pure-JS content search (replaces POSIX `grep -rn`) ───────────────────

/**
 * Recursive content search using Node.js fs. Cross-platform replacement for `grep -rn`.
 * @param {string} pattern - Regex pattern to search for
 * @param {string} searchPath - Directory or file to search
 * @param {string} [include] - Glob filter for filenames (e.g., "*.js")
 * @param {number} [maxResults=100] - Maximum matching lines to return
 * @returns {string} Grep-style output: "filepath:line:content"
 */
function nodeGrep(pattern, searchPath, include, maxResults = 100) {
  const results = [];
  let regex;
  try {
    regex = new RegExp(pattern, 'i');
  } catch {
    return `Error: invalid pattern '${pattern}'`;
  }

  // Convert include glob to regex (e.g., "*.js" → /\.js$/i)
  let includeRegex = null;
  if (include) {
    const inc = include
      .replace(/[.+^${}()|[\]\\]/g, '\\$&')
      .replace(/\*/g, '.*')
      .replace(/\?/g, '.');
    includeRegex = new RegExp('^' + inc + '$', 'i');
  }

  function searchFile(filePath) {
    if (results.length >= maxResults) return;
    if (includeRegex && !includeRegex.test(path.basename(filePath))) return;

    // Skip binary files (check first 512 bytes for null bytes)
    try {
      const fd = fs.openSync(filePath, 'r');
      const buf = Buffer.alloc(512);
      const bytesRead = fs.readSync(fd, buf, 0, 512, 0);
      fs.closeSync(fd);
      if (buf.subarray(0, bytesRead).includes(0)) return; // binary file
    } catch {
      return;
    }

    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const lines = content.split('\n');
      for (let i = 0; i < lines.length && results.length < maxResults; i++) {
        if (regex.test(lines[i])) {
          results.push(`${filePath}:${i + 1}:${lines[i]}`);
        }
      }
    } catch { /* skip unreadable files */ }
  }

  function walk(dir, depth) {
    if (depth > 8 || results.length >= maxResults) return;
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (results.length >= maxResults) return;
      if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath, depth + 1);
      } else if (entry.isFile()) {
        searchFile(fullPath);
      }
    }
  }

  try {
    const stat = fs.statSync(searchPath);
    if (stat.isFile()) {
      searchFile(searchPath);
    } else {
      walk(searchPath, 0);
    }
  } catch {
    return `Error: cannot access '${searchPath}'`;
  }

  return results.join('\n') || 'No matches found';
}

// ── Process management ────────────────────────────────────────────────────

function processList(filter) {
  if (IS_WIN) {
    if (filter) {
      return run(`powershell.exe -NoProfile -Command "Get-Process | Where-Object { $_.ProcessName -like '*${filter}*' } | Format-Table Id, CPU, WorkingSet64, ProcessName -AutoSize | Out-String -Width 200"`, 15000) || 'No matching processes';
    }
    return run('powershell.exe -NoProfile -Command "Get-Process | Sort-Object CPU -Descending | Select-Object -First 30 | Format-Table Id, CPU, WorkingSet64, ProcessName -AutoSize | Out-String -Width 200"', 15000);
  }
  if (filter) {
    return run(`ps aux | head -1; ps aux | grep '${filter.replace(/'/g, '')}' | grep -v grep`, 15000) || 'No matching processes';
  }
  return run('ps aux | head -30', 15000);
}

function netInfo() {
  if (IS_WIN) {
    return run('ipconfig', 10000) || 'Unable to gather network info';
  }
  if (IS_LINUX) {
    return run("ip addr 2>/dev/null | head -40") || run("ifconfig 2>/dev/null | head -30") || 'Unable to gather network info';
  }
  return run("ifconfig 2>/dev/null | grep -E 'flags|inet |ether' | head -30") || 'Unable to gather network info';
}

function netConnections(filter) {
  if (IS_WIN) {
    if (filter) {
      return run(`netstat -an | findstr "${filter}"`, 10000) || 'No matching connections';
    }
    return run('netstat -an | more', 10000) || 'Unable to list connections';
  }
  if (filter) {
    return run(`lsof -i -n -P 2>/dev/null | grep '${(filter || '').replace(/'/g, '')}' | head -30`) || 'No matching connections';
  }
  return run("lsof -i -n -P 2>/dev/null | head -30") || 'Unable to list connections';
}

function serviceStatus(service) {
  if (!service) return 'Error: service name required';
  const safe = service.replace(/['"]/g, '');
  if (IS_WIN) {
    return run(`powershell.exe -NoProfile -Command "Get-Service -Name '*${safe}*' | Format-Table Status, Name, DisplayName -AutoSize | Out-String -Width 200"`) || `Service '${service}' not found`;
  }
  if (IS_LINUX) {
    return run(`systemctl status '${safe}' 2>/dev/null`) || run(`systemctl --user status '${safe}' 2>/dev/null`) || `Service '${service}' not found`;
  }
  return run(`launchctl list 2>/dev/null | grep '${safe}'`) || run(`brew services list 2>/dev/null | grep '${safe}'`) || `Service '${service}' not found`;
}

function windowList() {
  if (IS_WIN) {
    return run('powershell.exe -NoProfile -Command "Get-Process | Where-Object { $_.MainWindowTitle -ne \'\' } | Format-Table Id, ProcessName, MainWindowTitle -AutoSize | Out-String -Width 300"') || 'Unable to list windows';
  }
  if (IS_LINUX) {
    return run("wmctrl -l 2>/dev/null") || 'Unable to list windows (install wmctrl)';
  }
  return run("osascript -e 'tell application \"System Events\" to get name of every application process whose visible is true' 2>/dev/null") || 'Unable to list windows';
}

function windowFocus(app) {
  if (!app) return false;
  const safe = app.replace(/"/g, '\\"');
  if (IS_WIN) {
    run(`powershell.exe -NoProfile -Command "$p = Get-Process | Where-Object { $_.MainWindowTitle -like '*${safe}*' } | Select-Object -First 1; if ($p) { Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class W { [DllImport(\\\"user32.dll\\\")] public static extern bool SetForegroundWindow(IntPtr hWnd); }'; [W]::SetForegroundWindow($p.MainWindowHandle) }"`);
    return true;
  }
  if (IS_LINUX) {
    // Find window by name and activate it via xdotool
    run(`xdotool search --name "${safe}" windowactivate 2>/dev/null`);
    return true;
  }
  run(`osascript -e 'tell application "${safe}" to activate' 2>&1`);
  return true;
}

// ── Disk usage (formatted) ────────────────────────────────────────────────

function diskUsageFormatted(targetPath) {
  if (IS_WIN) {
    const drive = (targetPath || 'C:').match(/^[A-Za-z]:/)?.[0] || 'C:';
    return run(`powershell.exe -NoProfile -Command "Get-PSDrive ${drive.replace(':', '')} | Format-Table Used, Free, @{N='Total';E={$_.Used+$_.Free}} -AutoSize | Out-String"`) || `Error: cannot check disk for ${targetPath}`;
  }
  return run(`df -h '${(targetPath || '/').replace(/'/g, '')}' 2>/dev/null`) || `Error: cannot check disk for ${targetPath}`;
}

// ── Screenshot (system tool) ──────────────────────────────────────────────

function screenshot(region) {
  const ts = Date.now();
  if (IS_WIN) {
    const outPath = path.join(os.tmpdir(), `pre_screenshot_${ts}.png`);
    try {
      execSync(`powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; $bmp = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height); $g = [System.Drawing.Graphics]::FromImage($bmp); $g.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size); $bmp.Save('${outPath}'); $g.Dispose(); $bmp.Dispose()"`, {
        timeout: 10000, windowsHide: true,
      });
      return outPath;
    } catch (err) {
      return null;
    }
  }
  if (IS_LINUX) {
    const outPath = `/tmp/pre_screenshot_${ts}.png`;
    try {
      if (region === 'selection') {
        execSync(`scrot -s '${outPath}' 2>/dev/null || import '${outPath}'`, { timeout: 30000 });
      } else if (region === 'window') {
        execSync(`scrot -u '${outPath}' 2>/dev/null || import -window "$(xdotool getactivewindow)" '${outPath}'`, { timeout: 30000 });
      } else {
        execSync(`scrot '${outPath}' 2>/dev/null || import -window root '${outPath}'`, { timeout: 10000 });
      }
      return outPath;
    } catch {
      return null;
    }
  }
  // macOS
  const outPath = `/tmp/pre_screenshot_${ts}.png`;
  try {
    if (region === 'selection') {
      execSync(`screencapture -i '${outPath}'`, { timeout: 30000 });
    } else if (region === 'window') {
      execSync(`screencapture -w '${outPath}'`, { timeout: 30000 });
    } else {
      execSync(`screencapture '${outPath}'`, { timeout: 10000 });
    }
    return outPath;
  } catch {
    return null;
  }
}

module.exports = {
  // Detection
  IS_WIN,
  IS_MAC,
  IS_LINUX,
  // Shell
  getShell,
  getShellPath,
  // Chrome
  getChromePaths,
  // File manager
  revealInFileManager,
  // Clipboard
  clipboardRead,
  clipboardWrite,
  // Notifications
  notify,
  // Open
  openTarget,
  // Command lookup
  whichCmd,
  // HTML
  htmlToText,
  // System info
  getCpuInfo,
  getCpuUsage,
  getMemoryInfo,
  getDiskUsage,
  getBatteryInfo,
  getGpuInfo,
  // TTS
  hasTTS,
  ttsSpeak,
  ttsListVoices,
  // File search
  nodeGlob,
  nodeGrep,
  // Process/network/services
  processList,
  netInfo,
  netConnections,
  serviceStatus,
  // Window management
  windowList,
  windowFocus,
  // Disk
  diskUsageFormatted,
  // Screenshot
  screenshot,
};
