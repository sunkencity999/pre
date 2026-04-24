// PRE Web GUI — Event-Driven Trigger Engine
// Extends the cron system with event-based execution: file watchers, webhooks,
// and git commit detection. Triggers fire prompts through the same execution
// pipeline as cron (session creation, tool loop, notifications).
//
// Storage: ~/.pre/triggers.json
// Trigger types: file_watch, webhook

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { PRE_DIR } = require('./constants');

const TRIGGERS_FILE = path.join(PRE_DIR, 'triggers.json');

// Active file watchers keyed by trigger ID
const activeWatchers = new Map();

// Debounce timers keyed by trigger ID
const debounceTimers = new Map();

// Execution callback — set by server.js via init()
let executeCallback = null;

// ── Persistence ────────────────────────────────────────────────────────────

function loadTriggers() {
  try {
    return JSON.parse(fs.readFileSync(TRIGGERS_FILE, 'utf-8'));
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
  const debounceMs = trigger.config.debounce || 3000;
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
  executeCallback({
    id: trigger.id,
    description: trigger.name,
    prompt,
  }).catch(err => {
    console.error(`[triggers] Execution error for ${trigger.id}: ${err.message}`);
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

  // Start all enabled file watchers
  const triggers = loadTriggers();
  let watcherCount = 0;

  for (const trigger of triggers) {
    if (!trigger.enabled) continue;
    if (trigger.type === 'file_watch') {
      if (startFileWatcher(trigger)) watcherCount++;
    }
  }

  if (watcherCount > 0) {
    console.log(`  [triggers] ${watcherCount} file watcher(s) active`);
  }

  return { watcherCount, total: triggers.length };
}

/**
 * Stop all watchers (for graceful shutdown).
 */
function shutdown() {
  for (const [id] of activeWatchers) {
    stopWatcher(id);
  }
  console.log('[triggers] All watchers stopped');
}

// ── Tool Actions ───────────────────────────────────────────────────────────

function addTrigger(args) {
  const type = (args.type || '').toLowerCase();
  if (!['file_watch', 'webhook'].includes(type)) {
    return 'Error: type must be "file_watch" or "webhook"';
  }
  if (!args.prompt) return 'Error: prompt is required';

  const config = {};
  const trigger = {
    id: generateId(),
    type,
    name: args.name || args.prompt.slice(0, 60),
    prompt: args.prompt,
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
    config.debounce = args.debounce || 3000;
    if (args.glob) config.glob = args.glob;
  }

  if (type === 'webhook') {
    if (args.secret) config.secret = args.secret;
  }

  const triggers = loadTriggers();
  triggers.push(trigger);
  saveTriggers(triggers);

  // Start watcher immediately if file_watch
  if (type === 'file_watch') {
    startFileWatcher(trigger);
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
  lines.push(`  Prompt: ${trigger.prompt}`);
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

module.exports = {
  trigger,
  init,
  shutdown,
  handleWebhook,
  loadTriggers,
  saveTriggers,
  isWatching,
};
