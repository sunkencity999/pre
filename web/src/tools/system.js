// PRE Web GUI — System tools (cross-platform)
// system_info, process_list, process_kill, hardware_info, disk_usage, etc.
// Platform-specific calls are routed through ../platform.js

const { execSync } = require('child_process');
const os = require('os');
const platform = require('../platform');

function run(cmd, timeout = 10000) {
  try {
    return execSync(cmd, { encoding: 'utf-8', timeout, maxBuffer: 64 * 1024 }).trim();
  } catch {
    return '';
  }
}

function systemInfo() {
  const parts = [];
  const cpu = platform.getCpuInfo();
  if (cpu.name) parts.push(`CPU: ${cpu.name}`);

  const mem = platform.getMemoryInfo();
  parts.push(`Memory: ${mem.totalGB} GB (${mem.usagePct}% used)`);

  const disk = platform.getDiskUsage();
  parts.push(`Disk: ${disk}% used`);

  const battery = platform.getBatteryInfo();
  if (battery.percent !== null) {
    parts.push(`Battery: ${battery.percent}% (${battery.state})`);
  }

  parts.push(`OS: ${os.type()} ${os.release()} (${os.arch()})`);
  parts.push(`Hostname: ${os.hostname()}`);
  parts.push(`Uptime: ${(os.uptime() / 3600).toFixed(1)} hours`);

  return parts.join('\n');
}

function processList(args) {
  return platform.processList(args?.filter);
}

function processKill(args) {
  const pid = args?.pid;
  if (!pid) return 'Error: no pid provided';
  if (!/^\d+$/.test(pid)) return `Error: invalid pid '${pid}'`;

  try {
    process.kill(parseInt(pid), platform.IS_WIN ? undefined : 'SIGTERM');
    return `Sent termination signal to pid ${pid}`;
  } catch (err) {
    return `Error killing pid ${pid}: ${err.message}`;
  }
}

function hardwareInfo() {
  const parts = [];
  const cpu = platform.getCpuInfo();
  if (cpu.name) parts.push(`CPU: ${cpu.name}`);
  if (cpu.cores) parts.push(`Cores: ${cpu.cores}`);

  const mem = platform.getMemoryInfo();
  parts.push(`Memory: ${mem.totalGB} GB`);

  const gpu = platform.getGpuInfo();
  if (gpu) parts.push(gpu.startsWith('GPU:') ? gpu : `GPU:\n  ${gpu.replace(/\n/g, '\n  ')}`);

  return parts.join('\n') || 'Unable to gather hardware info';
}

function diskUsage(args) {
  return platform.diskUsageFormatted(args?.path);
}

function netInfo() {
  return platform.netInfo();
}

function netConnections(args) {
  return platform.netConnections(args?.filter);
}

function serviceStatus(args) {
  return platform.serviceStatus(args?.service);
}

function displayInfo() {
  if (platform.IS_WIN) {
    return run('powershell.exe -NoProfile -Command "Get-CimInstance Win32_VideoController | Format-List Name, AdapterRAM, DriverVersion, VideoModeDescription | Out-String"') || 'Unable to gather display info';
  }
  return run("system_profiler SPDisplaysDataType 2>/dev/null") || 'Unable to gather display info';
}

function clipboardRead() {
  return platform.clipboardRead() || '(clipboard empty)';
}

function clipboardWrite(args) {
  const content = args?.content;
  if (!content) return 'Error: no content provided';
  if (platform.clipboardWrite(content)) {
    return `Copied ${Buffer.byteLength(content)} bytes to clipboard`;
  }
  return 'Error: cannot write to clipboard';
}

function openApp(args) {
  const target = args?.target;
  if (!target) return 'Error: no target provided';
  if (platform.openTarget(target)) {
    return `Opened ${target}`;
  }
  return `Error: failed to open ${target}`;
}

function notify(args) {
  const { title, message } = args || {};
  if (!title || !message) return 'Error: title and message required';
  platform.notify(title, message);
  return `Notification sent: ${title}`;
}

function screenshot(args) {
  const outPath = platform.screenshot(args?.region);
  if (outPath) return `Screenshot saved: ${outPath}`;
  return 'Error taking screenshot';
}

function windowList() {
  return platform.windowList();
}

function windowFocus(args) {
  const app = args?.app;
  if (!app) return 'Error: app name required';
  platform.windowFocus(app);
  return `Focused: ${app}`;
}

function applescript(args) {
  if (platform.IS_WIN) return 'AppleScript is not available on Windows. Use PowerShell or bash instead.';
  const script = args?.script;
  if (!script) return 'Error: no script provided';
  try {
    return execSync(`osascript -e '${script.replace(/'/g, "'\\''")}'`, {
      encoding: 'utf-8', timeout: 30000, maxBuffer: 64 * 1024,
    }).trim() || 'Script executed';
  } catch (err) {
    return `Error: ${(err.stderr || err.message).trim()}`;
  }
}

module.exports = {
  systemInfo, processList, processKill, hardwareInfo, diskUsage,
  netInfo, netConnections, serviceStatus, displayInfo,
  clipboardRead, clipboardWrite, openApp, notify, screenshot,
  windowList, windowFocus, applescript,
};
