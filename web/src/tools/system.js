// PRE Web GUI — System tools
// system_info, process_list, process_kill, hardware_info, disk_usage, etc.

const { execSync } = require('child_process');
const os = require('os');

function run(cmd, timeout = 10000) {
  try {
    return execSync(cmd, { encoding: 'utf-8', timeout, maxBuffer: 64 * 1024 }).trim();
  } catch {
    return '';
  }
}

function systemInfo() {
  const parts = [];
  const cpu = run("sysctl -n machdep.cpu.brand_string 2>/dev/null");
  if (cpu) parts.push(`CPU: ${cpu}`);

  const memBytes = parseInt(run("sysctl -n hw.memsize 2>/dev/null") || '0');
  if (memBytes) parts.push(`Memory: ${(memBytes / (1024 ** 3)).toFixed(1)} GB`);

  const vmstat = run("vm_stat 2>/dev/null | head -5");
  if (vmstat) parts.push(`VM Stats:\n  ${vmstat.replace(/\n/g, '\n  ')}`);

  const disk = run("df -h / 2>/dev/null");
  if (disk) parts.push(`Disk:\n  ${disk.replace(/\n/g, '\n  ')}`);

  const battery = run("pmset -g batt 2>/dev/null | tail -1");
  if (battery) parts.push(`Battery: ${battery.trim()}`);

  parts.push(`OS: ${os.type()} ${os.release()} (${os.arch()})`);
  parts.push(`Hostname: ${os.hostname()}`);
  parts.push(`Uptime: ${(os.uptime() / 3600).toFixed(1)} hours`);

  return parts.join('\n');
}

function processList(args) {
  const filter = args?.filter;
  if (filter) {
    return run(`ps aux | head -1; ps aux | grep '${filter.replace(/'/g, '')}' | grep -v grep`, 15000) || 'No matching processes';
  }
  return run('ps aux | head -30', 15000);
}

function processKill(args) {
  const pid = args?.pid;
  if (!pid) return 'Error: no pid provided';
  if (!/^\d+$/.test(pid)) return `Error: invalid pid '${pid}'`;

  try {
    process.kill(parseInt(pid), 'SIGTERM');
    return `Sent SIGTERM to pid ${pid}`;
  } catch (err) {
    return `Error killing pid ${pid}: ${err.message}`;
  }
}

function hardwareInfo() {
  const parts = [];
  const cpu = run("sysctl -n machdep.cpu.brand_string 2>/dev/null");
  if (cpu) parts.push(`CPU: ${cpu}`);

  const cores = run("sysctl -n hw.ncpu 2>/dev/null");
  if (cores) parts.push(`Cores: ${cores}`);

  const mem = parseInt(run("sysctl -n hw.memsize 2>/dev/null") || '0');
  if (mem) parts.push(`Memory: ${(mem / (1024 ** 3)).toFixed(0)} GB`);

  const gpu = run("system_profiler SPDisplaysDataType 2>/dev/null | grep 'Chipset\\|Chip\\|VRAM\\|Metal'");
  if (gpu) parts.push(`GPU:\n  ${gpu.replace(/\n/g, '\n  ')}`);

  return parts.join('\n') || 'Unable to gather hardware info';
}

function diskUsage(args) {
  const target = args?.path || '/';
  return run(`df -h '${target.replace(/'/g, '')}' 2>/dev/null`) || `Error: cannot check disk usage for ${target}`;
}

function netInfo() {
  return run("ifconfig 2>/dev/null | grep -E 'flags|inet |ether' | head -30") || 'Unable to gather network info';
}

function netConnections(args) {
  const filter = args?.filter;
  if (filter) {
    return run(`lsof -i -n -P 2>/dev/null | grep '${filter.replace(/'/g, '')}' | head -30`) || 'No matching connections';
  }
  return run("lsof -i -n -P 2>/dev/null | head -30") || 'Unable to list connections';
}

function serviceStatus(args) {
  const service = args?.service;
  if (!service) return 'Error: service name required';
  const safe = service.replace(/'/g, '');
  return run(`launchctl list 2>/dev/null | grep '${safe}'`) || run(`brew services list 2>/dev/null | grep '${safe}'`) || `Service '${service}' not found`;
}

function displayInfo() {
  return run("system_profiler SPDisplaysDataType 2>/dev/null") || 'Unable to gather display info';
}

function clipboardRead() {
  return run("pbpaste 2>/dev/null") || '(clipboard empty)';
}

function clipboardWrite(args) {
  const content = args?.content;
  if (!content) return 'Error: no content provided';
  try {
    execSync('pbcopy', { input: content, encoding: 'utf-8', timeout: 5000 });
    return `Copied ${Buffer.byteLength(content)} bytes to clipboard`;
  } catch (err) {
    return `Error: cannot write to clipboard: ${err.message}`;
  }
}

function openApp(args) {
  const target = args?.target;
  if (!target) return 'Error: no target provided';
  const result = run(`open '${target.replace(/'/g, "\\'")}' 2>&1`);
  return result || `Opened ${target}`;
}

function notify(args) {
  const { title, message } = args || {};
  if (!title || !message) return 'Error: title and message required';
  const safeTitle = title.replace(/"/g, '\\"');
  const safeMsg = message.replace(/"/g, '\\"');
  run(`osascript -e 'display notification "${safeMsg}" with title "${safeTitle}"' 2>&1`);
  return `Notification sent: ${title}`;
}

function screenshot(args) {
  const region = args?.region || 'full';
  const ts = Date.now();
  const outPath = `/tmp/pre_screenshot_${ts}.png`;
  try {
    if (region === 'selection') {
      execSync(`screencapture -i '${outPath}'`, { timeout: 30000 });
    } else if (region === 'window') {
      execSync(`screencapture -w '${outPath}'`, { timeout: 30000 });
    } else {
      execSync(`screencapture '${outPath}'`, { timeout: 10000 });
    }
    return `Screenshot saved: ${outPath}`;
  } catch (err) {
    return `Error taking screenshot: ${err.message}`;
  }
}

function windowList() {
  return run("osascript -e 'tell application \"System Events\" to get name of every application process whose visible is true' 2>/dev/null") || 'Unable to list windows';
}

function windowFocus(args) {
  const app = args?.app;
  if (!app) return 'Error: app name required';
  const safe = app.replace(/"/g, '\\"');
  run(`osascript -e 'tell application "${safe}" to activate' 2>&1`);
  return `Focused: ${app}`;
}

function applescript(args) {
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
