// PRE Web GUI — Self-Improving Skills Engine
// Analyzes completed tool chains and auto-creates reusable custom tools.
// Runs as a background task after tool loops (alongside memory + experience).

const { streamChat } = require('./ollama');
const { loadCustomTools, saveCustomTool } = require('./custom-tools');

// ── Constants ─────────────────────────────────────────────────────────────────

const SKILL_COOLDOWN_MS = 180000; // 3 minutes between analyses
const MIN_TOOL_CALLS = 3;         // Minimum tool calls to trigger analysis
const MAX_CHAIN_LENGTH = 15;      // Max tool calls to analyze (trim oldest)
const OVERLAP_THRESHOLD = 0.70;   // Reject proposed skills with >70% step overlap

// ── State ─────────────────────────────────────────────────────────────────────

const state = {
  running: false,
  lastAnalyzeTime: 0,
};

// ── Chain Extraction ──────────────────────────────────────────────────────────

/**
 * Extract tool call chains from conversation messages.
 * Returns an array of { tool, args, result, success } objects.
 */
function extractToolChains(messages) {
  const chains = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    // Assistant messages with tool_calls
    if (msg.role === 'assistant' && msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        const name = tc.function?.name || '';
        const args = tc.function?.arguments || {};
        chains.push({ tool: name, args, result: null, success: true });
      }
    }

    // Tool result messages — match to the most recent chain entry
    if (msg.role === 'tool' && msg.content) {
      const content = msg.content || '';
      // Parse tool_response tags
      const responses = content.match(/<tool_response name="([^"]+)">([\s\S]*?)<\/tool_response>/g) || [];
      for (const resp of responses) {
        const nameMatch = resp.match(/name="([^"]+)"/);
        const bodyMatch = resp.match(/<tool_response[^>]*>([\s\S]*?)<\/tool_response>/);
        if (nameMatch && bodyMatch) {
          const toolName = nameMatch[1];
          const body = bodyMatch[1].trim();
          const isError = body.startsWith('Error:') || body.includes('Error:');
          // Find matching chain entry (last one with this tool name and no result)
          for (let j = chains.length - 1; j >= 0; j--) {
            if (chains[j].tool === toolName && chains[j].result === null) {
              chains[j].result = body.slice(0, 500); // Cap for prompt
              chains[j].success = !isError;
              break;
            }
          }
        }
      }
    }
  }

  return chains.slice(-MAX_CHAIN_LENGTH);
}

// ── Deduplication ─────────────────────────────────────────────────────────────

/**
 * Check if a proposed skill overlaps with existing custom tools.
 * Compares tool step sequences — if >OVERLAP_THRESHOLD of steps match, it's a dupe.
 */
function isDuplicate(proposed, existingTools) {
  if (!proposed.implementation?.steps) return false;
  const proposedSteps = proposed.implementation.steps.map(s => s.tool);

  for (const existing of existingTools) {
    // Check name similarity
    if (existing.name === proposed.name) return true;

    if (!existing.implementation?.steps) continue;
    const existingSteps = existing.implementation.steps.map(s => s.tool);

    // Calculate step overlap
    const common = proposedSteps.filter(s => existingSteps.includes(s));
    const maxLen = Math.max(proposedSteps.length, existingSteps.length);
    if (maxLen > 0 && common.length / maxLen >= OVERLAP_THRESHOLD) return true;
  }
  return false;
}

// ── Main Entry Point ──────────────────────────────────────────────────────────

/**
 * Analyze a completed conversation for reusable tool patterns.
 * If found, auto-creates a custom tool (chain type).
 *
 * @param {Array} messages - Full session messages
 * @param {Object} opts - { sessionId, cwd }
 * @returns {Promise<Array>} Created skill objects (name + description)
 */
async function analyzeForSkills(messages, opts = {}) {
  // Gate: cooldown
  if (state.running) return [];
  const elapsed = Date.now() - state.lastAnalyzeTime;
  if (elapsed < SKILL_COOLDOWN_MS) return [];

  // Gate: minimum tool usage
  const toolMessages = messages.filter(m => m.role === 'tool');
  if (toolMessages.length < MIN_TOOL_CALLS) return [];

  state.running = true;
  state.lastAnalyzeTime = Date.now();

  try {
    const chains = extractToolChains(messages);
    if (chains.length < 2) return [];

    // Only analyze successful chains
    const successChains = chains.filter(c => c.success);
    if (successChains.length < 2) return [];

    const existingTools = loadCustomTools();

    // Build compact chain summary for the model
    const chainSummary = successChains.map((c, i) =>
      `${i + 1}. ${c.tool}(${JSON.stringify(c.args).slice(0, 200)}) → ${(c.result || '').slice(0, 150)}`
    ).join('\n');

    const result = await streamChat({
      messages: [
        {
          role: 'system',
          content: `You analyze tool execution chains to find reusable patterns. Given a sequence of tool calls that completed successfully, determine if they form a reusable workflow worth saving as a custom tool.

CRITERIA for a good skill:
- 2+ tools used in sequence that form a logical unit
- The pattern would be useful in future, similar tasks
- The steps have clear input parameters that can be generalized
- NOT a one-off investigation or debugging session

OUTPUT: Return ONLY valid JSON (no markdown, no explanation):
{ "worth_saving": false }
OR
{
  "worth_saving": true,
  "tool": {
    "name": "skill_name_here",
    "description": "What this skill does",
    "parameters": [{"name": "param1", "type": "string", "description": "...", "required": true}],
    "implementation": {
      "type": "chain",
      "steps": [{"tool": "tool_name", "args": {"key": "\${param1}"}}]
    }
  }
}

Use snake_case for names. Use \${param} syntax for parameter substitution in step args.`,
        },
        {
          role: 'user',
          content: `Analyze this tool chain for reusable patterns:\n\n${chainSummary}`,
        },
      ],
      maxTokens: 1024,
      think: false,
      onToken: () => {},
    });

    // Parse the response
    let parsed;
    try {
      // Extract JSON from response (model might wrap in backticks)
      const jsonStr = (result.response || '').replace(/```json?\n?/g, '').replace(/```/g, '').trim();
      parsed = JSON.parse(jsonStr);
    } catch {
      return [];
    }

    if (!parsed.worth_saving || !parsed.tool) return [];

    const proposed = parsed.tool;

    // Validate required fields
    if (!proposed.name || !proposed.description || !proposed.implementation) return [];

    // Check for duplicates
    if (isDuplicate(proposed, existingTools)) {
      console.log(`[skills] Proposed skill "${proposed.name}" overlaps with existing tool — skipped`);
      return [];
    }

    // Add metadata
    proposed.source = 'auto';
    proposed.created = new Date().toISOString();
    proposed.usage_count = 0;
    proposed.last_used = null;

    saveCustomTool(proposed);
    console.log(`[skills] Created skill: ${proposed.name} — ${proposed.description}`);
    return [{ name: proposed.name, description: proposed.description }];
  } finally {
    state.running = false;
  }
}

module.exports = {
  analyzeForSkills,
  extractToolChains,
  isDuplicate,
  SKILL_COOLDOWN_MS,
  MIN_TOOL_CALLS,
  OVERLAP_THRESHOLD,
};
