// PRE Web GUI — Background Process Monitor
// Starts, streams, and manages long-running background processes.
// Output is captured and can be read by the model or observed by Argus.

const { spawn } = require('child_process');

// Active monitors: id → { process, name, output[], startedAt, exitCode }
const monitors = new Map();
let monitorCounter = 0;

const MAX_OUTPUT_LINES = 500;

/**
 * Start a background process and capture its output.
 * @param {object} args - { command, name }
 * @returns {string} Monitor ID and status
 */
function startMonitor(args) {
  const { command, name } = args;
  if (!command) return 'Error: command is required';

  const id = `mon_${++monitorCounter}`;
  const displayName = name || command.slice(0, 60);

  const proc = spawn('sh', ['-c', command], {
    cwd: process.env.HOME || '/tmp',
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  const monitor = {
    id,
    name: displayName,
    command,
    pid: proc.pid,
    output: [],
    startedAt: Date.now(),
    exitCode: null,
    running: true,
    subscribers: new Set(), // WebSocket broadcast functions
  };

  const pushLine = (source, data) => {
    const lines = data.toString().split('\n').filter(l => l.length > 0);
    for (const line of lines) {
      const entry = { ts: Date.now(), source, text: line };
      monitor.output.push(entry);
      if (monitor.output.length > MAX_OUTPUT_LINES) monitor.output.shift();
      // Notify subscribers (Argus, live chat)
      for (const fn of monitor.subscribers) {
        try { fn({ type: 'monitor_output', id, name: displayName, source, text: line }); } catch {}
      }
    }
  };

  proc.stdout.on('data', (data) => pushLine('stdout', data));
  proc.stderr.on('data', (data) => pushLine('stderr', data));

  proc.on('close', (code) => {
    monitor.exitCode = code;
    monitor.running = false;
    for (const fn of monitor.subscribers) {
      try { fn({ type: 'monitor_exit', id, name: displayName, exitCode: code }); } catch {}
    }
  });

  proc.on('error', (err) => {
    monitor.running = false;
    pushLine('stderr', `Process error: ${err.message}`);
  });

  monitors.set(id, monitor);

  return `Monitor started: ${id} (${displayName})\nPID: ${proc.pid}\nUse monitor with action "read" and id "${id}" to see output.`;
}

/**
 * Read recent output from a monitor.
 * @param {object} args - { id, tail }
 * @returns {string} Recent output lines
 */
function readMonitor(args) {
  const { id, tail } = args;
  if (!id) return 'Error: id is required';

  const monitor = monitors.get(id);
  if (!monitor) return `Error: monitor ${id} not found`;

  const count = parseInt(tail) || 30;
  const lines = monitor.output.slice(-count);

  let result = `Monitor: ${monitor.name} (${monitor.running ? 'running' : `exited: ${monitor.exitCode}`})\n`;
  result += `PID: ${monitor.pid} | Started: ${new Date(monitor.startedAt).toLocaleTimeString()} | Lines: ${monitor.output.length}\n`;
  result += '---\n';
  result += lines.map(l => `[${l.source}] ${l.text}`).join('\n');

  return result || 'No output yet.';
}

/**
 * Stop a running monitor.
 * @param {object} args - { id }
 * @returns {string} Status message
 */
function stopMonitor(args) {
  const { id } = args;
  if (!id) return 'Error: id is required';

  const monitor = monitors.get(id);
  if (!monitor) return `Error: monitor ${id} not found`;
  if (!monitor.running) return `Monitor ${id} already stopped (exit code: ${monitor.exitCode})`;

  try {
    process.kill(monitor.pid, 'SIGTERM');
    return `Monitor ${id} (${monitor.name}) stopped.`;
  } catch (err) {
    return `Error stopping monitor: ${err.message}`;
  }
}

/**
 * List all monitors (active and recent).
 * @returns {string} Monitor list
 */
function listMonitors() {
  if (monitors.size === 0) return 'No monitors active.';

  return Array.from(monitors.values()).map(m => {
    const status = m.running ? 'running' : `exited (${m.exitCode})`;
    const duration = ((Date.now() - m.startedAt) / 1000).toFixed(0);
    return `[${m.id}] ${m.name} — ${status} — ${duration}s — ${m.output.length} lines`;
  }).join('\n');
}

/**
 * Subscribe a broadcast function to a monitor's output.
 * Used by Argus and WebSocket forwarding.
 */
function subscribe(id, fn) {
  const monitor = monitors.get(id);
  if (monitor) monitor.subscribers.add(fn);
}

function unsubscribe(id, fn) {
  const monitor = monitors.get(id);
  if (monitor) monitor.subscribers.delete(fn);
}

/**
 * Main dispatcher
 */
function monitor(args) {
  const action = args.action || 'list';
  switch (action) {
    case 'start': return startMonitor(args);
    case 'read': return readMonitor(args);
    case 'stop': return stopMonitor(args);
    case 'list': return listMonitors();
    default: return `Unknown monitor action: ${action}. Use start, read, stop, or list.`;
  }
}

module.exports = { monitor, subscribe, unsubscribe };
