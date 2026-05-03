// PRE Web GUI — Event-Driven Trigger Engine
// Extends the cron system with event-based execution: file watchers, webhooks,
// polling monitors, and git commit detection. Triggers fire prompts through the
// same execution pipeline as cron (session creation, tool loop, notifications).
//
// Storage: ~/.pre/triggers.json
// Trigger types: file_watch, webhook, polling

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { PRE_DIR, CONNECTIONS_FILE } = require('./constants');

const TRIGGERS_FILE = path.join(PRE_DIR, 'triggers.json');

// Active file watchers keyed by trigger ID
const activeWatchers = new Map();

// Active polling intervals keyed by trigger ID
const activePollers = new Map();

// Debounce timers keyed by trigger ID
const debounceTimers = new Map();

// Execution callback — set by server.js via init()
let executeCallback = null;

// ── Persistence ────────────────────────────────────────────────────────────

function loadTriggers() {
  try {
    const data = JSON.parse(fs.readFileSync(TRIGGERS_FILE, 'utf-8'));
    // Handle both formats: plain array or {triggers: [...]}
    return Array.isArray(data) ? data : Array.isArray(data.triggers) ? data.triggers : [];
  } catch {
    return [];
  }
}

function saveTriggers(triggers) {
  fs.writeFileSync(TRIGGERS_FILE, JSON.stringify(triggers, null, 2));
}

function generateId() {
  return crypto.randomBytes(4).toString('hex');
}

// ── File Watcher ───────────────────────────────────────────────────────────

/**
 * Start watching a file or directory for changes.
 * Uses Node.js fs.watch (FSEvents on macOS — very efficient).
 */
function startFileWatcher(trigger) {
  const watchPath = trigger.config.path;
  if (!watchPath || !fs.existsSync(watchPath)) {
    console.log(`[triggers] Cannot watch ${watchPath} — path not found`);
    return false;
  }

  // Stop existing watcher if any
  stopWatcher(trigger.id);

  const isDir = fs.statSync(watchPath).isDirectory();
  const recursive = isDir && trigger.config.recursive !== false;
  const debounceMs = trigger.config.debounce || 120000; // 2 min — batches rapid file events
  const globPattern = trigger.config.glob || null;

  try {
    const watcher = fs.watch(watchPath, { recursive }, (eventType, filename) => {
      if (!filename) return;

      // Glob filter
      if (globPattern) {
        const match = matchGlob(filename, globPattern);
        if (!match) return;
      }

      // Skip common noise
      if (filename.includes('.git/') && !trigger.config.watchGit) return;
      if (filename.includes('node_modules/')) return;
      if (filename.endsWith('.swp') || filename.endsWith('.tmp') || filename.endsWith('~')) return;

      // Debounce — batch rapid changes
      const existing = debounceTimers.get(trigger.id);
      if (existing) {
        existing.files.add(filename);
        existing.events.add(eventType);
        return;
      }

      const state = { files: new Set([filename]), events: new Set([eventType]) };
      debounceTimers.set(trigger.id, state);

      setTimeout(() => {
        debounceTimers.delete(trigger.id);
        fireTrigger(trigger, {
          files: [...state.files],
          events: [...state.events],
          watchPath,
        });
      }, debounceMs);
    });

    watcher.on('error', (err) => {
      console.log(`[triggers] Watcher error for ${trigger.id}: ${err.message}`);
    });

    activeWatchers.set(trigger.id, watcher);
    console.log(`[triggers] Watching ${watchPath} (trigger: ${trigger.id}, recursive: ${recursive})`);
    return true;
  } catch (err) {
    console.log(`[triggers] Failed to start watcher for ${trigger.id}: ${err.message}`);
    return false;
  }
}

/**
 * Simple glob matching (supports *, **, and ? patterns).
 */
function matchGlob(filename, pattern) {
  // Convert glob to regex
  const regexStr = pattern
    .replace(/\./g, '\\.')
    .replace(/\*\*/g, '__DOUBLESTAR__')
    .replace(/\*/g, '[^/]*')
    .replace(/__DOUBLESTAR__/g, '.*')
    .replace(/\?/g, '.');
  try {
    return new RegExp(`^${regexStr}$`).test(filename) || new RegExp(regexStr).test(filename);
  } catch {
    return filename.includes(pattern.replace(/[*?]/g, ''));
  }
}

function stopWatcher(triggerId) {
  const watcher = activeWatchers.get(triggerId);
  if (watcher) {
    watcher.close();
    activeWatchers.delete(triggerId);
  }
  const timer = debounceTimers.get(triggerId);
  if (timer) {
    debounceTimers.delete(triggerId);
  }
}

// ── Trigger Execution ──────────────────────────────────────────────────────

/**
 * Fire a trigger — substitute variables in the prompt and execute.
 */
function fireTrigger(trigger, context = {}) {
  if (!executeCallback) {
    console.log(`[triggers] Cannot fire trigger ${trigger.id} — no execution callback`);
    return;
  }

  // Build the prompt with variable substitution
  let prompt = trigger.prompt;

  if (context.files) {
    prompt = prompt.replace(/\{files?\}/gi, context.files.join(', '));
  }
  if (context.events) {
    prompt = prompt.replace(/\{events?\}/gi, context.events.join(', '));
  }
  if (context.watchPath) {
    prompt = prompt.replace(/\{path\}/gi, context.watchPath);
  }
  if (context.payload) {
    const payloadStr = typeof context.payload === 'string'
      ? context.payload : JSON.stringify(context.payload, null, 2);
    prompt = prompt.replace(/\{payload\}/gi, payloadStr);
  }
  if (context.headers) {
    prompt = prompt.replace(/\{headers\}/gi, JSON.stringify(context.headers));
  }
  if (context.services) {
    prompt = prompt.replace(/\{services?\}/gi, context.services.join(', '));
  }

  // Update trigger stats
  const triggers = loadTriggers();
  const t = triggers.find(tr => tr.id === trigger.id);
  if (t) {
    t.last_fired_at = Date.now();
    t.fire_count = (t.fire_count || 0) + 1;
    saveTriggers(triggers);
  }

  console.log(`[triggers] Firing trigger ${trigger.id}: "${trigger.name}"`);

  // Execute through the cron runner pipeline
  // Pass the watch path as cwd so the model operates in the watched directory
  executeCallback({
    id: trigger.id,
    description: trigger.name,
    prompt,
    cwd: context.watchPath || undefined,
  }).catch(err => {
    console.error(`[triggers] Execution error for ${trigger.id}: ${err.message}`);
  });
}

// ── Polling Monitor ───────────────────────────────────────────────────────

/**
 * Available polling services and what they check.
 * Each returns a summary string (or null if nothing changed).
 */
const POLLING_SERVICES = {
  github: {
    label: 'GitHub',
    requires: 'github_key',
    description: 'Check for new PRs, issues, and notifications',
  },
  gmail: {
    label: 'Gmail',
    requires: 'google_refresh_token',
    description: 'Check for new unread emails',
  },
  jira: {
    label: 'Jira',
    requires: 'jira_token',
    description: 'Check for assigned issue updates',
  },
  slack: {
    label: 'Slack',
    requires: 'slack_token',
    description: 'Check for new unread messages',
  },
  calendar: {
    label: 'Calendar',
    requires: null,  // Native macOS — always available
    description: 'Check for upcoming events',
  },
};

/**
 * Get currently connected services that support polling.
 */
function getAvailablePollingServices() {
  let connections = {};
  try {
    connections = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
  } catch {}

  return Object.entries(POLLING_SERVICES).map(([key, svc]) => ({
    service: key,
    label: svc.label,
    description: svc.description,
    available: svc.requires === null || !!connections[svc.requires],
  }));
}

/**
 * Start a polling monitor for a trigger.
 * Runs at the configured interval and fires when the model should be briefed.
 */
function startPoller(trigger) {
  stopPoller(trigger.id);

  const intervalMs = (trigger.config.interval_minutes || 60) * 60 * 1000;
  const services = trigger.config.services || ['all'];

  console.log(`[triggers] Polling ${services.join(', ')} every ${trigger.config.interval_minutes || 60}m (trigger: ${trigger.id})`);

  // Fire immediately on start, then at intervals
  runPollingCheck(trigger, services);

  const interval = setInterval(() => {
    runPollingCheck(trigger, services);
  }, intervalMs);

  activePollers.set(trigger.id, interval);
  return true;
}

function stopPoller(triggerId) {
  const interval = activePollers.get(triggerId);
  if (interval) {
    clearInterval(interval);
    activePollers.delete(triggerId);
  }
}

/**
 * Run a single polling check: gather state from services and fire if there are changes.
 */
function runPollingCheck(trigger, services) {
  // Build a dynamic prompt that asks the model to check the specified services
  const serviceList = services.includes('all')
    ? Object.keys(POLLING_SERVICES)
    : services.filter(s => POLLING_SERVICES[s]);

  if (serviceList.length === 0) return;

  // Load connections to filter to only available services
  let connections = {};
  try {
    connections = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
  } catch {}

  const available = serviceList.filter(s => {
    const svc = POLLING_SERVICES[s];
    return svc.requires === null || !!connections[svc.requires];
  });

  if (available.length === 0) return;

  // Build a check prompt
  const serviceInstructions = available.map(s => {
    switch (s) {
      case 'github': return '- GitHub: Check my notifications and any PRs/issues assigned to me. Use the github tool with action "notifications" and "my_issues".';
      case 'gmail': return '- Gmail: Check for unread emails in my inbox. Use the gmail tool with action "list" and filter for unread messages.';
      case 'jira': return '- Jira: Check for issues assigned to me that have been updated recently. Use the jira tool with action "my_issues".';
      case 'slack': return '- Slack: Check for unread messages and mentions. Use the slack tool with action "unread".';
      case 'calendar': return '- Calendar: Check for upcoming events in the next 4 hours. Use the apple_calendar tool with action "today".';
      default: return '';
    }
  }).filter(Boolean).join('\n');

  const briefingPrompt = `You are running a proactive monitoring check. Check the following services for noteworthy updates and compile a brief status report.

${serviceInstructions}

RULES:
- Only report items that are NEW or CHANGED since you last checked.
- Be concise: one line per item, grouped by service.
- If nothing noteworthy has changed, simply respond: "No notable updates."
- Format as a clean briefing the user can scan quickly.
- Include actionable items (things that need attention) at the top.`;

  // Use the trigger's custom prompt if provided, otherwise use the default briefing
  const prompt = trigger.prompt || briefingPrompt;

  fireTrigger(trigger, {
    services: available,
    checkType: 'polling',
  });
}

// ── Webhook Handling ───────────────────────────────────────────────────────

/**
 * Handle an incoming webhook request.
 * Called by the server's POST /api/triggers/webhook/:id endpoint.
 */
function handleWebhook(triggerId, { body, headers }) {
  const triggers = loadTriggers();
  const trigger = triggers.find(t => t.id === triggerId && t.type === 'webhook');

  if (!trigger) return { error: 'Trigger not found', status: 404 };
  if (!trigger.enabled) return { error: 'Trigger is disabled', status: 403 };

  // Verify secret if configured
  if (trigger.config.secret) {
    const provided = headers['x-webhook-secret'] || headers['x-hub-signature'] || '';
    if (provided !== trigger.config.secret) {
      return { error: 'Invalid secret', status: 401 };
    }
  }

  fireTrigger(trigger, {
    payload: body,
    headers: Object.fromEntries(
      Object.entries(headers).filter(([k]) =>
        !['host', 'content-length', 'connection'].includes(k.toLowerCase())
      )
    ),
  });

  return { ok: true, triggerId: trigger.id, name: trigger.name };
}

// ── Lifecycle ──────────────────────────────────────────────────────────────

/**
 * Initialize the trigger engine.
 * Call once from server.js after the execution pipeline is ready.
 *
 * @param {Function} executeFn — async (job) => { sessionId, response }
 */
function init(executeFn) {
  executeCallback = executeFn;

  // Start all enabled file watchers and pollers
  const triggers = loadTriggers();
  let watcherCount = 0;
  let pollerCount = 0;

  for (const trigger of triggers) {
    if (!trigger.enabled) continue;
    if (trigger.type === 'file_watch') {
      if (startFileWatcher(trigger)) watcherCount++;
    } else if (trigger.type === 'polling') {
      if (startPoller(trigger)) pollerCount++;
    }
  }

  if (watcherCount > 0) {
    console.log(`  [triggers] ${watcherCount} file watcher(s) active`);
  }
  if (pollerCount > 0) {
    console.log(`  [triggers] ${pollerCount} polling monitor(s) active`);
  }

  return { watcherCount, pollerCount, total: triggers.length };
}

/**
 * Stop all watchers (for graceful shutdown).
 */
function shutdown() {
  for (const [id] of activeWatchers) {
    stopWatcher(id);
  }
  for (const [id] of activePollers) {
    stopPoller(id);
  }
  console.log('[triggers] All watchers stopped');
}

// ── Tool Actions ───────────────────────────────────────────────────────────

function addTrigger(args) {
  const type = (args.type || '').toLowerCase();
  if (!['file_watch', 'webhook', 'polling'].includes(type)) {
    return 'Error: type must be "file_watch", "webhook", or "polling"';
  }
  if (!args.prompt && type !== 'polling') return 'Error: prompt is required';

  const config = {};
  const trigger = {
    id: generateId(),
    type,
    name: args.name || (args.prompt || 'Proactive Monitor').slice(0, 60),
    prompt: args.prompt || '',
    config,
    enabled: true,
    created_at: Date.now(),
    last_fired_at: null,
    fire_count: 0,
  };

  if (type === 'file_watch') {
    if (!args.path) return 'Error: path is required for file_watch triggers';
    const watchPath = path.resolve(args.path);
    if (!fs.existsSync(watchPath)) return `Error: path not found: ${watchPath}`;
    config.path = watchPath;
    config.recursive = args.recursive !== false;
    config.debounce = args.debounce || 120000;
    if (args.glob) config.glob = args.glob;
  }

  if (type === 'webhook') {
    if (args.secret) config.secret = args.secret;
  }

  if (type === 'polling') {
    config.services = args.services || ['all'];
    config.interval_minutes = args.interval_minutes || 60;
    if (Array.isArray(config.services)) {
      config.services = config.services.map(s => s.toLowerCase().trim());
    }
  }

  const triggers = loadTriggers();
  triggers.push(trigger);
  saveTriggers(triggers);

  // Start watcher/poller immediately
  if (type === 'file_watch') {
    startFileWatcher(trigger);
  } else if (type === 'polling') {
    startPoller(trigger);
  }

  const lines = [`Trigger created:`, `  ID: ${trigger.id}`, `  Type: ${type}`, `  Name: ${trigger.name}`];
  if (type === 'file_watch') {
    lines.push(`  Watching: ${config.path}`);
    if (config.glob) lines.push(`  Glob: ${config.glob}`);
  }
  if (type === 'webhook') {
    lines.push(`  Endpoint: POST /api/triggers/webhook/${trigger.id}`);
    if (config.secret) lines.push(`  Secret: configured`);
  }
  if (type === 'polling') {
    lines.push(`  Services: ${config.services.join(', ')}`);
    lines.push(`  Interval: every ${config.interval_minutes} minutes`);
  }
  if (trigger.prompt) lines.push(`  Prompt: ${trigger.prompt}`);
  return lines.join('\n');
}

function listTriggers() {
  const triggers = loadTriggers();
  if (triggers.length === 0) return 'No triggers configured. Use action `add` to create one.';

  const lines = [`${triggers.length} trigger(s):`, ''];
  for (const t of triggers) {
    const status = t.enabled ? 'ACTIVE' : 'DISABLED';
    const lastFired = t.last_fired_at ? new Date(t.last_fired_at).toLocaleString() : 'never';
    const watcherStatus = activeWatchers.has(t.id) ? ' [watching]' : '';
    lines.push(`[${status}] ${t.id}  ${t.type}  "${t.name}"${watcherStatus}`);
    lines.push(`  Fires: ${t.fire_count || 0}, Last: ${lastFired}`);
    if (t.type === 'file_watch') lines.push(`  Path: ${t.config.path}${t.config.glob ? ` (${t.config.glob})` : ''}`);
    if (t.type === 'webhook') lines.push(`  Endpoint: POST /api/triggers/webhook/${t.id}`);
    if (t.type === 'polling') {
      const pollerStatus = activePollers.has(t.id) ? ' [polling]' : '';
      lines[lines.length - 2] = lines[lines.length - 2].replace(watcherStatus, watcherStatus + pollerStatus);
      lines.push(`  Services: ${(t.config.services || []).join(', ')}`);
      lines.push(`  Interval: every ${t.config.interval_minutes || 60} minutes`);
    }
    lines.push('');
  }
  return lines.join('\n');
}

function removeTrigger(args) {
  if (!args.id) return 'Error: id is required';
  const triggers = loadTriggers();
  const idx = triggers.findIndex(t => t.id === args.id);
  if (idx === -1) return `Error: no trigger with ID "${args.id}"`;

  stopWatcher(args.id);
  stopPoller(args.id);
  const removed = triggers.splice(idx, 1)[0];
  saveTriggers(triggers);
  return `Removed trigger ${removed.id}: "${removed.name}"`;
}

function enableTrigger(args) {
  if (!args.id) return 'Error: id is required';
  const triggers = loadTriggers();
  const trigger = triggers.find(t => t.id === args.id);
  if (!trigger) return `Error: no trigger with ID "${args.id}"`;

  trigger.enabled = true;
  saveTriggers(triggers);

  if (trigger.type === 'file_watch') startFileWatcher(trigger);
  if (trigger.type === 'polling') startPoller(trigger);
  return `Enabled trigger ${trigger.id}: "${trigger.name}"`;
}

function disableTrigger(args) {
  if (!args.id) return 'Error: id is required';
  const triggers = loadTriggers();
  const trigger = triggers.find(t => t.id === args.id);
  if (!trigger) return `Error: no trigger with ID "${args.id}"`;

  trigger.enabled = false;
  saveTriggers(triggers);
  stopWatcher(args.id);
  stopPoller(args.id);
  return `Disabled trigger ${trigger.id}: "${trigger.name}"`;
}

/**
 * Tool dispatcher — called from tools.js
 */
function trigger(args) {
  const action = (args.action || '').toLowerCase();
  switch (action) {
    case 'add': case 'create': return addTrigger(args);
    case 'list': case 'ls': return listTriggers();
    case 'remove': case 'rm': case 'delete': return removeTrigger(args);
    case 'enable': return enableTrigger(args);
    case 'disable': return disableTrigger(args);
    default: return 'Error: unknown trigger action. Use: add, list, remove, enable, disable';
  }
}

/**
 * Check if a watcher is actively running for a given trigger ID.
 */
function isWatching(triggerId) {
  return activeWatchers.has(triggerId);
}

/**
 * Restart the file watcher for a given trigger.
 * Useful when the watcher died or wasn't started.
 */
function restartWatcher(triggerId) {
  const triggers = loadTriggers();
  const trigger = triggers.find(t => t.id === triggerId);
  if (!trigger) return { error: 'Trigger not found' };
  if (trigger.type !== 'file_watch') return { error: 'Not a file watcher trigger' };
  if (!trigger.enabled) return { error: 'Trigger is disabled — enable it first' };

  stopWatcher(triggerId);
  const ok = startFileWatcher(trigger);
  return ok
    ? { ok: true, watching: true }
    : { error: `Failed to start watcher for ${trigger.config?.path || triggerId}` };
}

module.exports = {
  trigger,
  init,
  shutdown,
  handleWebhook,
  loadTriggers,
  saveTriggers,
  isWatching,
  restartWatcher,
  getAvailablePollingServices,
};
