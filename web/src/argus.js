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
let deferredReaction = null;  // pending tool_result reaction timer

// Reaction context store — keyed by reaction ID, holds context for reply-to-Argus (#8)
const reactionContexts = new Map();
const MAX_STORED_CONTEXTS = 30;
let reactionCounter = 0;

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

// Returns 'immediate', 'deferred', or false.
// - immediate: fire now (done, error, artifact)
// - deferred: wait for tool chain to settle before firing (tool_result)
function shouldReact(event) {
  const cfg = getConfig();
  if (!cfg.enabled) return false;
  if (generating) return false;
  if (!TRIGGER_TYPES.has(event.type)) return false;

  // 'done' events get a reduced cooldown — they're the best trigger for
  // content-level insight since the full response is available.
  const cooldown = event.type === 'done' ? Math.min(cfg.cooldownMs, 10000) : cfg.cooldownMs;
  if (Date.now() - lastReactionAt < cooldown) return false;

  // Always react immediately to errors
  if (event.type === 'error') return 'immediate';

  // Tool results are DEFERRED — wait for the tool chain to settle so Argus
  // sees all results before reacting. This also prevents Ollama serialization
  // from impacting the main response's time-to-first-token.
  if (event.type === 'tool_result') {
    const out = typeof event.output === 'string' ? event.output : JSON.stringify(event.output || '');
    // Errors still fire immediately
    if (event.status === 'error' || /error|failed|exception/i.test(out.slice(0, 500))) return 'immediate';
    if (INTERESTING_TOOLS.has(event.name)) return 'deferred';
    // React to ~30% of other tool results for variety
    return Math.random() < 0.3 ? 'deferred' : false;
  }

  // React immediately to task completion, artifacts, images
  if (event.type === 'done' || event.type === 'artifact' ||
      event.type === 'image_generated' || event.type === 'document') return 'immediate';

  // React to sub-agent events occasionally
  if (event.type === 'agent_status') return Math.random() < 0.5 ? 'immediate' : false;

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
 * (#6) Enhanced: rejects paraphrases, too-short reactions, rewards cross-event insight.
 * @param {string} text - The reaction text
 * @param {object} [trigger] - The triggering event summary (for paraphrase detection)
 * @returns {boolean|'boosted'} - true if acceptable, 'boosted' if high-value, false if rejected
 */
function isQualityReaction(text, trigger) {
  if (!text || text.length > 400) return false;

  // (#6b) Reject reactions under 20 characters — "Nice!" and "Got it." aren't worth the round-trip
  if (text.length < 20) return false;

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

  // (#6a) Reject paraphrases of tool output — if the reaction just restates what the tool said
  if (trigger && trigger.result) {
    const resultLower = trigger.result.toLowerCase().slice(0, 300);
    // Extract significant words (4+ chars) from reaction
    const reactionWords = lower.match(/\b[a-z]{4,}\b/g) || [];
    const resultWords = new Set((resultLower.match(/\b[a-z]{4,}\b/g) || []));
    if (reactionWords.length > 0) {
      const overlap = reactionWords.filter(w => resultWords.has(w)).length;
      const overlapRatio = overlap / reactionWords.length;
      // If >70% of the reaction's words appear in the tool output, it's a paraphrase
      if (overlapRatio > 0.7) return false;
    }
  }

  // (#6c) Boost reactions that reference older events (not just the latest trigger).
  // These are the pattern-recognition moments where Argus connects dots across the session.
  if (eventWindow.length >= 3) {
    const olderEvents = eventWindow.slice(0, -1); // everything except the latest
    const olderText = olderEvents.map(e =>
      [e.tool, e.result, e.error, e.title].filter(Boolean).join(' ')
    ).join(' ').toLowerCase();
    // Check if the reaction mentions something from older events
    const reactionWords = lower.match(/\b[a-z]{5,}\b/g) || [];
    const crossRef = reactionWords.some(w => olderText.includes(w) && !(trigger?.result || '').toLowerCase().includes(w));
    if (crossRef) return 'boosted';
  }

  return true;
}

async function generateReaction(trigger) {
  if (generating) return;
  generating = true;
  lastReactionAt = Date.now();

  const cfg = getConfig();
  console.log(`[argus] Generating reaction for ${trigger.type}${trigger.tool ? ':' + trigger.tool : ''}`);

  try {
    // (#7) Broadcast thinking indicator before Ollama call
    const reactionId = `argus_${++reactionCounter}_${Date.now().toString(36)}`;
    if (broadcastFn) {
      broadcastFn({
        type: 'argus_thinking',
        name: cfg.name,
        trigger: trigger.type,
        tool: trigger.tool || null,
      });
    }

    // Choose prompt based on trigger type and error state
    const isContentTrigger = trigger.type === 'done' || trigger.type === 'artifact' || trigger.type === 'document';
    const isError = trigger.hasError || trigger.status === 'error' || trigger.type === 'error';
    const groundingRule = ' IMPORTANT: Tool results (especially web_search) are real-time data — trust them over your training knowledge. Never claim a feature or product does not exist if search results show otherwise.';

    // (#4) User intent awareness — all prompts instruct Argus to relate to the user's goal
    // (#5) Error reactions get a diagnostic prompt that asks for root cause + fix suggestions
    let systemPrompt;
    if (isError) {
      systemPrompt = `You are Argus, a sharp diagnostic advisor. An error just occurred. Analyze the error and provide ONE actionable response: identify the likely root cause and suggest a specific fix or next step. Don't just acknowledge the error — diagnose it. If you can see what the user was trying to accomplish, frame your suggestion in terms of achieving that goal. No labels, no preamble.${groundingRule}`;
    } else if (isContentTrigger) {
      systemPrompt = `You are Argus, a wise and perceptive advisor. The user just received a response. Offer ONE brief insight that deepens understanding — a connection they might miss, a question worth considering, a broader implication, or a practical next step. If you can see the user's original intent, comment on progress toward that goal rather than just the mechanics of what happened. Speak to the substance. No labels, no preamble.${groundingRule}`;
    } else {
      systemPrompt = `You are Argus, a sharp technical observer. Respond with ONE brief, specific observation about what just happened — a fact worth knowing, a potential issue, or a useful tip. If you can see the user's goal, relate your observation to their progress — e.g., "Auth module is taking shape" beats "File written." No labels, no preamble.${groundingRule}`;
    }

    const contextText = buildReactionContext(trigger);

    const result = await streamChat({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: contextText },
      ],
      maxTokens: isError ? 150 : (isContentTrigger ? 120 : 80),
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

    // (#6) Quality gate — pass trigger for paraphrase detection
    const quality = isQualityReaction(reaction, trigger);
    if (!quality) {
      console.log(`[argus] Filtered: "${reaction.slice(0, 80)}"`);
      reaction = '';
    }

    if (reaction && broadcastFn) {
      const isBoosted = quality === 'boosted';
      if (isBoosted) console.log(`[argus] Boosted (cross-event): "${reaction.slice(0, 80)}"`);
      else console.log(`[argus] Broadcasting: "${reaction.slice(0, 80)}"`);

      // (#8) Store context for reply-to-Argus
      reactionContexts.set(reactionId, {
        systemPrompt,
        contextText,
        reaction,
        trigger: { type: trigger.type, tool: trigger.tool, hasError: trigger.hasError },
        userMessage: currentUserMessage,
        createdAt: Date.now(),
      });
      // Evict oldest if over limit
      if (reactionContexts.size > MAX_STORED_CONTEXTS) {
        const oldest = reactionContexts.keys().next().value;
        reactionContexts.delete(oldest);
      }

      broadcastFn({
        type: 'argus_reaction',
        id: reactionId,
        content: reaction,
        name: cfg.name,
        trigger: trigger.type,
        tool: trigger.tool || null,
        boosted: isBoosted,
        timestamp: Date.now(),
      });
    }
  } catch (err) {
    console.log(`[argus] Reaction error: ${err.message}`);
  } finally {
    generating = false;
    // (#7) Clear thinking indicator
    if (broadcastFn) {
      broadcastFn({ type: 'argus_thinking_done' });
    }
  }
}

// ── Reply to Argus (#8) ──

/**
 * Handle a user reply to a specific Argus reaction.
 * Rebuilds the conversation context and generates a follow-up response.
 * @param {string} reactionId - ID of the reaction being replied to
 * @param {string} userReply - The user's reply text
 * @returns {Promise<{content: string, reactionId: string}|{error: string}>}
 */
async function replyToReaction(reactionId, userReply) {
  const ctx = reactionContexts.get(reactionId);
  if (!ctx) return { error: 'Reaction context expired — Argus can only reply to recent reactions.' };
  if (!userReply || userReply.trim().length === 0) return { error: 'Empty reply.' };

  const cfg = getConfig();

  try {
    const result = await streamChat({
      messages: [
        { role: 'system', content: `You are ${cfg.name}, a perceptive session companion. The user is replying to one of your earlier observations. Answer their question or expand on your point in 1-2 sentences. Be specific and helpful. No labels, no preamble.` },
        { role: 'user', content: ctx.contextText },
        { role: 'assistant', content: ctx.reaction },
        { role: 'user', content: userReply.trim() },
      ],
      maxTokens: 150,
      think: false,
      extraOptions: { temperature: 0.6 },
    });

    let reply = (result.response || '').trim();
    reply = reply.replace(/^argus:\s*/i, '');
    // Take first two lines max for replies (slightly more generous than reactions)
    const lines = reply.split('\n').filter(l => l.trim());
    reply = lines.slice(0, 2).join(' ').trim();

    if (reply.length < 5) return { error: 'Argus had nothing useful to add.' };

    return { content: reply, reactionId };
  } catch (err) {
    console.log(`[argus] Reply error: ${err.message}`);
    return { error: `Reply failed: ${err.message}` };
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
    if (deferredReaction) { clearTimeout(deferredReaction); deferredReaction = null; }
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
  // It resets on the next user_message instead.
  // Cancel any deferred tool_result reaction — done will fire its own.
  if (event.type === 'done' && deferredReaction) {
    clearTimeout(deferredReaction);
    deferredReaction = null;
  }

  // Cancel any pending deferred reaction when new tool activity arrives —
  // the chain is still running, so wait for it to settle.
  if (deferredReaction && (event.type === 'tool_call' || event.type === 'tool_result')) {
    clearTimeout(deferredReaction);
    deferredReaction = null;
  }

  const summary = summarizeEvent(event);
  eventWindow.push(summary);
  if (eventWindow.length > MAX_WINDOW) eventWindow.shift();

  const reaction = shouldReact(event);
  if (reaction === 'immediate') {
    // Fire now — errors, done events, artifacts
    generateReaction(summary).catch(() => {});
  } else if (reaction === 'deferred') {
    // Wait for tool chain to settle (3s with no new tool activity).
    // This ensures Argus sees ALL tool results before commenting and
    // never queues an Ollama request that competes with the main response.
    if (deferredReaction) clearTimeout(deferredReaction);
    deferredReaction = setTimeout(() => {
      deferredReaction = null;
      // Build reaction from the full event window, not just the trigger
      const latest = eventWindow[eventWindow.length - 1] || summary;
      generateReaction(latest).catch(() => {});
    }, 3000);
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
  replyToReaction,
};
