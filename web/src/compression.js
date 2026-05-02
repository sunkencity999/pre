// PRE Web GUI — Session Context Compression
// Summarizes old conversation turns when approaching context limits.
// Keeps recent turns intact; replaces oldest messages with a compact summary.

const { MODEL_CTX } = require('./constants');
const { streamChat } = require('./ollama');

// ── Token Estimation ──────────────────────────────────────────────────────────

// Char→token ratio, calibrated at runtime from actual Ollama prompt_eval_count.
// Starts at 3.5 (reasonable default for Gemma-class models), then adjusts toward
// reality using exponential moving average. Avoids over/under-triggering compression.
let charsPerToken = 3.5;
const CALIBRATION_ALPHA = 0.15; // Smoothing factor — responsive but stable
let calibrationSamples = 0;

function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(String(text).length / charsPerToken);
}

/**
 * Calibrate the token estimator using actual Ollama stats.
 * Call after each inference with the messages sent and the actual token count.
 * @param {Array} messages - Messages that were sent to Ollama
 * @param {number} actualTokens - prompt_eval_count from Ollama response
 */
function calibrate(messages, actualTokens) {
  if (!actualTokens || actualTokens < 100) return; // Too few tokens to be meaningful
  // Estimate total chars in the messages
  let totalChars = 0;
  for (const m of messages) {
    if (m.content) totalChars += String(m.content).length;
    if (m.tool_calls) totalChars += JSON.stringify(m.tool_calls).length;
    if (m.thinking) totalChars += String(m.thinking).length;
  }
  if (totalChars < 100) return;

  const observedRatio = totalChars / actualTokens;
  if (observedRatio < 1 || observedRatio > 10) return; // Sanity bounds

  if (calibrationSamples === 0) {
    // First sample: jump to observed value (fast convergence on session start)
    charsPerToken = observedRatio;
  } else {
    // EMA: blend toward observed ratio
    charsPerToken = charsPerToken * (1 - CALIBRATION_ALPHA) + observedRatio * CALIBRATION_ALPHA;
  }
  calibrationSamples++;
}

function estimateMessageTokens(messages) {
  let total = 0;
  for (const m of messages) {
    total += estimateTokens(m.content);
    // Tool calls in assistant messages add overhead
    if (m.tool_calls) total += estimateTokens(JSON.stringify(m.tool_calls));
    // Thinking tokens count toward context too
    if (m.thinking) total += estimateTokens(m.thinking);
  }
  return total;
}

// ── Budget Constants ──────────────────────────────────────────────────────────

const SYSTEM_PROMPT_TOKENS = 3000;   // System prompt + memory context
const OUTPUT_RESERVE = 32768;        // Max tokens for model output (from tools.js)
const TOOL_TOKENS_PER_DOMAIN = 1500; // ~1500 tokens per activated domain
const CORE_TOOL_TOKENS = 3000;       // Core tools overhead

// Minimum number of recent turns to keep uncompressed (3 user-assistant pairs)
const KEEP_RECENT_TURNS = 6;

// Compression fires when token usage exceeds this fraction of budget
const COMPRESSION_THRESHOLD = 0.80;

// ── Core Compression ──────────────────────────────────────────────────────────

/**
 * Check if messages need compression and compress if necessary.
 * Returns the (possibly compressed) message array.
 *
 * @param {string} sessionId - Session identifier
 * @param {Array} messages - Full message history
 * @param {number} [activeDomainCount=0] - Number of active tool domains
 * @returns {Promise<Array>} Messages, possibly with old turns compressed
 */
async function compressIfNeeded(sessionId, messages, activeDomainCount = 0) {
  if (!messages || messages.length <= KEEP_RECENT_TURNS) return messages;

  // Calculate available token budget for conversation history
  const toolTokens = CORE_TOOL_TOKENS + (activeDomainCount * TOOL_TOKENS_PER_DOMAIN);
  const overhead = SYSTEM_PROMPT_TOKENS + toolTokens + OUTPUT_RESERVE;
  const budget = MODEL_CTX - overhead;

  if (budget <= 0) return messages; // Shouldn't happen, but guard

  const currentTokens = estimateMessageTokens(messages);
  const threshold = budget * COMPRESSION_THRESHOLD;

  // Under threshold — no compression needed
  if (currentTokens <= threshold) return messages;

  console.log(`[compression] Session ${sessionId}: ${currentTokens} tokens exceeds ${Math.round(threshold)} threshold (budget: ${budget}). Compressing...`);

  // Split messages: old (to compress) and recent (to keep)
  const splitIdx = messages.length - KEEP_RECENT_TURNS;
  const oldMessages = messages.slice(0, splitIdx);
  const recentMessages = messages.slice(splitIdx);

  // Build a compact text representation of old messages for summarization
  const oldText = oldMessages.map(m => {
    const role = m.role || 'unknown';
    const content = (m.content || '').slice(0, 2000); // Cap per-message for summarization
    return `[${role}] ${content}`;
  }).join('\n\n');

  // If old text is very short, not worth compressing
  if (estimateTokens(oldText) < 500) return messages;

  try {
    const summary = await summarize(oldText);
    if (!summary) return messages; // Summarization failed, return original

    const compressedMsg = {
      role: 'user',
      content: `[Compressed conversation context — ${oldMessages.length} earlier messages summarized]\n\n${summary}\n\n[End of compressed context. Recent conversation follows.]`,
    };

    const result = [compressedMsg, ...recentMessages];
    const newTokens = estimateMessageTokens(result);
    console.log(`[compression] Compressed ${oldMessages.length} messages (${currentTokens} → ${newTokens} tokens, saved ${currentTokens - newTokens})`);
    return result;
  } catch (err) {
    console.log(`[compression] Error: ${err.message}. Returning uncompressed.`);
    return messages;
  }
}

// ── Summarization via Ollama ──────────────────────────────────────────────────

async function summarize(text) {
  // Cap input to avoid overwhelming the model
  const maxInput = 8000;
  const truncated = text.length > maxInput ? text.slice(0, maxInput) + '\n\n[...truncated]' : text;

  const result = await streamChat({
    messages: [
      {
        role: 'system',
        content: 'You are a conversation compressor. Summarize the following conversation history concisely while preserving ALL essential information: key decisions, tool results (especially file paths, error messages, and solutions), user requests and preferences, and important context. Be terse but complete. Use bullet points. Do not add commentary.',
      },
      {
        role: 'user',
        content: `Summarize this conversation history:\n\n${truncated}`,
      },
    ],
    maxTokens: 2048,
    think: false,
    onToken: () => {}, // Silent — no streaming to client
  });

  return result.response?.trim() || null;
}

module.exports = {
  compressIfNeeded,
  estimateTokens,
  estimateMessageTokens,
  calibrate,
  COMPRESSION_THRESHOLD,
  KEEP_RECENT_TURNS,
  SYSTEM_PROMPT_TOKENS,
  OUTPUT_RESERVE,
  CORE_TOOL_TOKENS,
  TOOL_TOKENS_PER_DOMAIN,
};
