// PRE Web GUI — Hooks system
// Pre/post tool execution hooks defined in ~/.pre/hooks.json
// Hooks can log, block, or modify tool calls.

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const HOOKS_PATH = path.join(process.env.HOME || '/tmp', '.pre', 'hooks.json');

// ── Cached hooks config ──────────────────────────────────────────────────────
// runHooks() is called twice per tool call (pre + post). With 10 tools per turn
// and 35 turns, that's ~700 readFileSync calls per session for a file that only
// changes when the user manually edits hooks. Cache with 30s TTL.
let _hooksCache = null;
let _hooksCacheTime = 0;
const _HOOKS_TTL = 30000;

/**
 * Hook config format (hooks.json):
 * {
 *   "hooks": [
 *     {
 *       "id": "log-bash",
 *       "event": "pre_tool",           // pre_tool | post_tool | pre_message | post_message
 *       "tool": "bash",                // Tool name filter (optional, "*" = all)
 *       "command": "echo $PRE_TOOL_ARGS >> /tmp/pre-audit.log",
 *       "enabled": true,
 *       "can_block": true,             // If true, non-zero exit blocks the tool call
 *       "timeout": 5000,               // Max ms for hook execution
 *       "description": "Audit all bash commands"
 *     }
 *   ]
 * }
 *
 * Environment variables passed to hook commands:
 *   PRE_HOOK_EVENT  — "pre_tool", "post_tool", etc.
 *   PRE_TOOL_NAME   — Tool being called (e.g. "bash")
 *   PRE_TOOL_ARGS   — JSON-encoded tool arguments
 *   PRE_TOOL_OUTPUT  — Tool output (post_tool only)
 *   PRE_SESSION_ID  — Current session ID
 *   PRE_CWD         — Working directory
 */

function loadHooks() {
  const now = Date.now();
  if (_hooksCache && now - _hooksCacheTime < _HOOKS_TTL) return _hooksCache;
  try {
    if (fs.existsSync(HOOKS_PATH)) {
      const raw = fs.readFileSync(HOOKS_PATH, 'utf-8');
      const config = JSON.parse(raw);
      _hooksCache = config.hooks || [];
      _hooksCacheTime = now;
      return _hooksCache;
    }
  } catch (err) {
    console.error('[hooks] Failed to load config:', err.message);
  }
  _hooksCache = [];
  _hooksCacheTime = now;
  return _hooksCache;
}

function saveHooks(hooks) {
  const dir = path.dirname(HOOKS_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(HOOKS_PATH, JSON.stringify({ hooks }, null, 2));
  // Invalidate cache so next loadHooks() reads the updated file
  _hooksCache = null;
  _hooksCacheTime = 0;
}

/**
 * Run hooks matching a given event
 * @param {string} event - "pre_tool" | "post_tool" | "pre_message" | "post_message"
 * @param {object} context - { tool, args, output, sessionId, cwd }
 * @returns {{ blocked: boolean, reason?: string }}
 */
function runHooks(event, context = {}) {
  const hooks = loadHooks();
  const matching = hooks.filter(h => {
    if (!h.enabled) return false;
    if (h.event !== event) return false;
    // Tool filter: match specific tool or "*" for all
    if (h.tool && h.tool !== '*' && h.tool !== context.tool) return false;
    return true;
  });

  if (matching.length === 0) return { blocked: false };

  const env = {
    ...process.env,
    PRE_HOOK_EVENT: event,
    PRE_TOOL_NAME: context.tool || '',
    PRE_TOOL_ARGS: JSON.stringify(context.args || {}),
    PRE_TOOL_OUTPUT: context.output || '',
    PRE_SESSION_ID: context.sessionId || '',
    PRE_CWD: context.cwd || '',
  };

  for (const hook of matching) {
    const timeout = hook.timeout || 5000;
    try {
      execSync(hook.command, {
        env,
        timeout,
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: context.cwd || process.env.HOME,
      });
    } catch (err) {
      // Non-zero exit = hook failed
      if (hook.can_block) {
        const stderr = err.stderr?.toString().trim() || '';
        const reason = stderr || `Hook '${hook.id || hook.description}' blocked this action (exit code ${err.status})`;
        console.log(`[hooks] BLOCKED by ${hook.id}: ${reason}`);
        return { blocked: true, reason };
      }
      // Non-blocking hook failure — just log
      console.log(`[hooks] Hook ${hook.id} failed (non-blocking): ${err.message}`);
    }
  }

  return { blocked: false };
}

/**
 * Add a new hook
 */
function addHook(hook) {
  const hooks = loadHooks();
  if (!hook.id) {
    hook.id = `hook_${Date.now().toString(36)}`;
  }
  if (!hook.event) return { error: 'event is required (pre_tool, post_tool, pre_message, post_message)' };
  if (!hook.command) return { error: 'command is required' };
  hook.enabled = hook.enabled !== false;
  hook.can_block = !!hook.can_block;
  hook.timeout = hook.timeout || 5000;
  hooks.push(hook);
  saveHooks(hooks);
  return hook;
}

/**
 * Remove a hook by ID
 */
function removeHook(id) {
  const hooks = loadHooks();
  const idx = hooks.findIndex(h => h.id === id);
  if (idx === -1) return { error: 'Hook not found' };
  const removed = hooks.splice(idx, 1)[0];
  saveHooks(hooks);
  return removed;
}

/**
 * Toggle a hook enabled/disabled
 */
function toggleHook(id) {
  const hooks = loadHooks();
  const hook = hooks.find(h => h.id === id);
  if (!hook) return { error: 'Hook not found' };
  hook.enabled = !hook.enabled;
  saveHooks(hooks);
  return hook;
}

/**
 * List all hooks
 */
function listHooks() {
  return loadHooks();
}

module.exports = { loadHooks, saveHooks, runHooks, addHook, removeHook, toggleHook, listHooks };
