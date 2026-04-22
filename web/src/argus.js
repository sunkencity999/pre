// PRE Web GUI — Argus Companion
// Observes session events in realtime and provides brief, contextual reactions.
// Reactions are generated via a lightweight Ollama call (fire-and-forget, never blocks).

const fs = require('fs');
const path = require('path');
const { streamChat } = require('./ollama');
const { PRE_DIR } = require('./constants');

const BUDDY_FILE = path.join(PRE_DIR, 'argus.json');

// ── Default config ──
const DEFAULTS = {
  enabled: true,
  name: 'Argus',
  personality: 'thoughtful mentor',
  cooldownMs: 30000,
  maxReactionTokens: 150,
};

// ── State ──
let config = null;
let lastReactionAt = 0;
let generating = false;
let broadcastFn = null;
const eventWindow = [];       // sliding window of recent event summaries
const MAX_WINDOW = 12;
let currentUserMessage = '';  // the user's most recent prompt (for context)
let recentResponseText = '';  // accumulates assistant response tokens

// ── Config management ──

function loadConfig() {
  try {
    config = { ...DEFAULTS, ...JSON.parse(fs.readFileSync(BUDDY_FILE, 'utf-8')) };
  } catch {
    config = { ...DEFAULTS };
    fs.writeFileSync(BUDDY_FILE, JSON.stringify(config, null, 2));
  }
  return config;
}

function saveConfig(updates) {
  if (!config) loadConfig();
  config = { ...config, ...updates };
  fs.writeFileSync(BUDDY_FILE, JSON.stringify(config, null, 2));
  return config;
}

function getConfig() {
  if (!config) loadConfig();
  return config;
}

function getStatus() {
  return {
    enabled: getConfig().enabled,
    name: getConfig().name,
    lastReactionAt,
    generating,
    windowSize: eventWindow.length,
  };
}

// ── Event observation ──

// Events that can trigger a reaction
const TRIGGER_TYPES = new Set([
  'tool_result', 'done', 'artifact', 'document',
  'image_generated', 'error', 'agent_status',
]);

// Tools that are particularly interesting to comment on
const INTERESTING_TOOLS = new Set([
  'bash', 'web_search', 'file_write', 'file_read',
  'memory_save', 'rag_search', 'rag_index',
  'apple_mail', 'apple_calendar', 'apple_reminders',
  'github', 'slack', 'computer_use', 'browser',
  'image_generate', 'delegate',
]);

function summarizeEvent(event) {
  const summary = { type: event.type, ts: Date.now() };
  if (event.name) summary.tool = event.name;
  if (event.status) summary.status = event.status;
  if (event.type === 'error' && event.message) {
    summary.error = event.message.slice(0, 200);
  }
  if (event.type === 'tool_result' && event.output) {
    const out = typeof event.output === 'string' ? event.output : JSON.stringify(event.output);
    summary.result = out.slice(0, 600);
    // Flag errors in tool output
    if (event.status === 'error' || /error|failed|exception|denied|not found/i.test(out.slice(0, 500))) {
      summary.hasError = true;
    }
  }
  if (event.type === 'artifact') {
    summary.title = event.title;
    summary.artifactType = event.artifactType;
  }
  if (event.type === 'done' && event.stats) {
    summary.tokPerSec = event.stats.tok_s;
    summary.toolTurns = event.stats.eval_count ? 'multi-turn' : 'single';
  }
  if (event.type === 'agent_status') {
    summary.agentType = event.agentType || event.type;
    if (event.task) summary.task = event.task.slice(0, 100);
  }
  return summary;
}

function shouldReact(event) {
  const cfg = getConfig();
  if (!cfg.enabled) return false;
  if (generating) return false;
  if (!TRIGGER_TYPES.has(event.type)) return false;

  // 'done' events get a reduced cooldown — they're the best trigger for
  // content-level insight since the full response is available.
  const cooldown = event.type === 'done' ? Math.min(cfg.cooldownMs, 10000) : cfg.cooldownMs;
  if (Date.now() - lastReactionAt < cooldown) return false;

  // Always react to errors
  if (event.type === 'error') return true;

  // React to tool results — prioritize interesting tools and errors
  if (event.type === 'tool_result') {
    const out = typeof event.output === 'string' ? event.output : JSON.stringify(event.output || '');
    if (event.status === 'error' || /error|failed|exception/i.test(out.slice(0, 500))) return true;
    if (INTERESTING_TOOLS.has(event.name)) return true;
    // React to ~30% of other tool results for variety
    return Math.random() < 0.3;
  }

  // React to task completion, artifacts, images
  if (event.type === 'done' || event.type === 'artifact' ||
      event.type === 'image_generated' || event.type === 'document') return true;

  // React to sub-agent events occasionally
  if (event.type === 'agent_status') return Math.random() < 0.5;

  return false;
}

// ── Reaction generation ──
//
// Key design decisions:
// 1. think: false — Gemma 4's thinking mode causes over-reasoning and meta-text
//    in short-form generation. Disabling it forces direct, terse output.
// 2. Completion-style prompting — present a session transcript that Argus
//    naturally continues, not an instruction the model follows.
// 3. Quality filtering — reject narration, instruction echoing, and meta-text.

function buildReactionContext(trigger) {
  const now = new Date();
  let ctx = `Today: ${now.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })}\n`;

  if (currentUserMessage) {
    ctx += `User: "${currentUserMessage.slice(0, 250)}"\n`;
  }

  // Recent tool activity as terse transcript lines
  for (const e of eventWindow.slice(-6)) {
    let line = `[${e.tool || e.type}]`;
    if (e.hasError || e.status === 'error') line += ' ERROR';
    if (e.result) line += ` → ${e.result.slice(0, 400)}`;
    if (e.error) line += ` → ${e.error}`;
    if (e.title) line += ` "${e.title}"`;
    ctx += line + '\n';
  }

  if (recentResponseText && recentResponseText.length > 30) {
    ctx += `Assistant: "${recentResponseText.slice(0, 1000).trim()}"\n`;
  }

  // The triggering event
  if (trigger.type === 'done') {
    ctx += '\nTask complete.\n';
  } else {
    let tLine = 'Latest: ';
    if (trigger.tool) tLine += trigger.tool;
    if (trigger.hasError) tLine += ' [ERROR]';
    if (trigger.result) tLine += ` → ${trigger.result.slice(0, 400)}`;
    if (trigger.error) tLine += ` → ${trigger.error}`;
    if (trigger.title) tLine += ` "${trigger.title}"`;
    ctx += tLine + '\n';
  }

  return ctx;
}

/**
 * Check whether a reaction is genuinely useful vs. narration/meta-text.
 */
function isQualityReaction(text) {
  if (!text || text.length < 10 || text.length > 400) return false;

  const lower = text.toLowerCase();

  // Label prefixes — model wrapping output in a category
  if (/^(insight|observation|note|comment|tip|suggestion|goal|option|example|argus|response|answer)\s*:/i.test(text)) return false;

  // Guillemet markers — model treating examples as numbered options
  if (/[«»]/.test(text)) return false;

  // Self-referential meta-text
  if (/^i (notice|see|can see|observe|would|should|think)\b/i.test(text)) return false;
  if (lower.includes('the user asked') || lower.includes('the assistant ')) return false;

  // Narrating tool activity instead of adding insight — only reject clearly narrative phrases
  if (/^(the |this |that |a )?(search|tool|command|query|request|bash|script|model) (returned|found|was |is |ran |executed|completed|started|called|produced|showed|used|performed)/i.test(text)) return false;
  if (/^pre (is |has |used |started |ran |called )/i.test(text)) return false;

  // Instruction echoing — model repeating its own prompt constraints
  if (lower.includes('one sentence') || lower.includes('no preamble') || lower.includes('terse')) return false;

  // Silence signals
  if (/^[.…—\-*\s]+$/.test(text)) return false;

  return true;
}

async function generateReaction(trigger) {
  if (generating) return;
  generating = true;
  lastReactionAt = Date.now();

  const cfg = getConfig();
  console.log(`[argus] Generating reaction for ${trigger.type}${trigger.tool ? ':' + trigger.tool : ''}`);

  try {
    // Choose prompt based on trigger type — 'done' events get the substantive
    // commentary prompt since the full response is available for analysis.
    const isContentTrigger = trigger.type === 'done' || trigger.type === 'artifact' || trigger.type === 'document';
    const groundingRule = ' IMPORTANT: Tool results (especially web_search) are real-time data — trust them over your training knowledge. Never claim a feature or product does not exist if search results show otherwise.';
    const systemPrompt = isContentTrigger
      ? 'You are Argus, a wise and perceptive advisor. The user just received a response. Offer ONE brief insight that deepens understanding — a connection they might miss, a question worth considering, a broader implication, or a practical next step. Speak to the substance, not the mechanics. No labels, no preamble.' + groundingRule
      : 'You are Argus, a sharp technical observer. Respond with ONE brief, specific observation about what just happened — a fact worth knowing, a potential issue, or a useful tip. No labels, no preamble.' + groundingRule;

    const result = await streamChat({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: buildReactionContext(trigger) },
      ],
      maxTokens: isContentTrigger ? 120 : 80,
      think: false,
      extraOptions: { temperature: 0.6 },
    });

    let reaction = (result.response || '').trim();

    // Strip "Argus:" prefix if model adds it
    reaction = reaction.replace(/^argus:\s*/i, '');

    // Take first line only (prevent multi-line rambling)
    reaction = reaction.split('\n')[0].trim();

    // If still long, truncate to first sentence
    if (reaction.length > 200) {
      const end = reaction.search(/[.!?]\s/);
      if (end > 10) reaction = reaction.slice(0, end + 1).trim();
    }

    // Quality gate
    if (!isQualityReaction(reaction)) {
      console.log(`[argus] Filtered: "${reaction.slice(0, 80)}"`);
      reaction = '';
    }

    if (reaction && broadcastFn) {
      console.log(`[argus] Broadcasting: "${reaction.slice(0, 80)}"`);
      broadcastFn({
        type: 'argus_reaction',
        content: reaction,
        name: cfg.name,
        trigger: trigger.type,
        tool: trigger.tool || null,
        timestamp: Date.now(),
      });
    }
  } catch (err) {
    console.log(`[argus] Reaction error: ${err.message}`);
  } finally {
    generating = false;
  }
}

// ── Public API ──

function observeEvent(event) {
  const cfg = getConfig();
  if (!cfg.enabled) return;

  // Capture the user's message for conversational context
  if (event.type === 'user_message' && event.content) {
    currentUserMessage = event.content.slice(0, 300);
    recentResponseText = ''; // reset for new turn
    eventWindow.length = 0;  // clear stale events from previous session/turn
    return;
  }

  // Accumulate response text (cap at 1500 chars for substantive context)
  if (event.type === 'token' && event.content) {
    if (recentResponseText.length < 1500) {
      recentResponseText += event.content;
    }
    return;
  }

  // Skip thinking events
  if (event.type === 'thinking') return;

  // Don't reset response text on 'done' — the done-triggered reaction needs it.
  // It resets on the next user_message instead (line 289-291).

  const summary = summarizeEvent(event);
  eventWindow.push(summary);
  if (eventWindow.length > MAX_WINDOW) eventWindow.shift();

  if (shouldReact(event)) {
    // Fire and forget — never block
    generateReaction(summary).catch(() => {});
  }
}

function setBroadcast(fn) {
  broadcastFn = fn;
}

function init(broadcast) {
  loadConfig();
  broadcastFn = broadcast;
  console.log(`[argus] Initialized — ${config.enabled ? 'enabled' : 'disabled'}, name: ${config.name}`);
}

module.exports = {
  init,
  observeEvent,
  setBroadcast,
  getConfig,
  saveConfig,
  getStatus,
  loadConfig,
};
