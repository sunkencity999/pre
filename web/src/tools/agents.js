// PRE Web GUI — Sub-agent spawning and management
// Allows the model to spawn parallel research tasks as independent Ollama requests.

const http = require('http');
const { streamChat, healthCheck } = require('../ollama');
const { buildToolDefs } = require('../tools-defs');
const { MODEL, OLLAMA_PORT } = require('../constants');
const { createSession, appendMessage: appendSessionMessage, renameSession } = require('../sessions');

// Timeout defaults (ms)
// At 128K context, each inference takes 12-30s. With 10 tool turns + tool execution,
// a full agent run can legitimately take 7-8 minutes. 10 minutes gives headroom.
const AGENT_TIMEOUT = 10 * 60 * 1000;      // 10 minutes overall per agent
const INFERENCE_TIMEOUT = 2 * 60 * 1000;    // 2 minutes per Ollama call (stuck-model guard)
const MODEL_READY_TIMEOUT = 30 * 1000;      // 30 seconds to wait for model load

/**
 * Wrap a promise with a timeout. Rejects with a descriptive error if the
 * promise doesn't settle within `ms` milliseconds.
 */
function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/**
 * Verify Ollama is running and the model is loaded before spawning an agent.
 * If the model isn't loaded, sends a lightweight request to trigger loading
 * and waits up to MODEL_READY_TIMEOUT for it to become available.
 */
async function ensureModelReady() {
  const healthy = await healthCheck();
  if (!healthy) throw new Error('Ollama is not running — cannot spawn agent');

  // Check if model is loaded via /api/ps
  const loaded = await new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${OLLAMA_PORT}/api/ps`, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          const ps = JSON.parse(data);
          const models = ps.models || [];
          resolve(models.some(m => m.name && m.name.startsWith(MODEL)));
        } catch { resolve(false); }
      });
    });
    req.on('error', () => resolve(false));
    req.setTimeout(5000, () => { req.destroy(); resolve(false); });
  });

  if (!loaded) {
    // Trigger a minimal inference to load the model
    const loadReq = new Promise((resolve, reject) => {
      const body = JSON.stringify({
        model: MODEL,
        prompt: 'hi',
        stream: false,
        options: { num_predict: 1, num_ctx: 8192 },
      });
      const req = http.request({
        hostname: '127.0.0.1',
        port: OLLAMA_PORT,
        path: '/api/generate',
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
      }, (res) => {
        let data = '';
        res.on('data', (chunk) => data += chunk);
        res.on('end', () => resolve(true));
      });
      req.on('error', (err) => reject(err));
      req.write(body);
      req.end();
    });
    await withTimeout(loadReq, MODEL_READY_TIMEOUT, 'Model loading');
  }
}

// Active agents: { id: { status, task, result, startedAt, ... } }
const agents = {};
let agentCounter = 0;

// Tools available to sub-agents (safe, read-only subset)
const AGENT_TOOLS = [
  'bash', 'read_file', 'list_dir', 'glob', 'grep',
  'web_fetch', 'web_search', 'memory_search', 'memory_list',
  'system_info', 'hardware_info', 'disk_usage',
];

/**
 * Spawn a sub-agent to perform a task autonomously
 * @param {object} args - { task, tools_allowed }
 * @param {string} cwd - working directory
 * @param {Function} onStatus - callback for status updates
 * @param {object} [overrides] - { maxTurns, allowedTools, systemPrompt } for research mode
 * @returns {string} result summary
 */
async function spawnAgent(args, cwd, onStatus, overrides) {
  const { task } = args;
  if (!task) return 'Error: task description is required';

  // Verify Ollama is healthy and model is loaded before committing resources
  try {
    await ensureModelReady();
  } catch (err) {
    return `Error: Cannot spawn agent — ${err.message}. Try again in a moment.`;
  }

  const id = `agent_${++agentCounter}`;

  // Create a dedicated session for this agent
  const ts = Date.now().toString(36);
  const sessionId = createSession('web', `agent-${id}-${ts}`, true);
  const shortTask = task.length > 60 ? task.slice(0, 57) + '...' : task;
  renameSession(sessionId, `Agent: ${shortTask}`);
  appendSessionMessage(sessionId, { role: 'user', content: task });

  const agent = {
    id,
    task,
    sessionId,
    status: 'running',
    startedAt: Date.now(),
    messages: [],
    result: null,
  };
  agents[id] = agent;

  if (onStatus) onStatus({ type: 'agent_started', id, task, sessionId });

  try {
    const result = await withTimeout(
      runAgent(agent, cwd, onStatus, overrides),
      AGENT_TIMEOUT,
      `Agent "${id}"`,
    );
    agent.status = 'completed';
    agent.result = result;
    agent.completedAt = Date.now();
    // Save final result to agent session
    appendSessionMessage(sessionId, { role: 'assistant', content: result });
    if (onStatus) onStatus({ type: 'agent_completed', id, sessionId, duration: agent.completedAt - agent.startedAt });
    return result;
  } catch (err) {
    agent.status = 'failed';
    agent.result = `Error: ${err.message}`;
    agent.completedAt = Date.now();
    appendSessionMessage(sessionId, { role: 'assistant', content: agent.result });
    if (onStatus) onStatus({ type: 'agent_failed', id, sessionId, error: err.message });
    return agent.result;
  }
}

/**
 * Run the agent's internal loop — single-turn or multi-turn with tools
 * @param {object} agent
 * @param {string} cwd
 * @param {Function} onStatus
 * @param {object} [overrides] - { maxTurns, allowedTools, systemPrompt }
 */
async function runAgent(agent, cwd, onStatus, overrides) {
  const systemPrompt = (overrides && overrides.systemPrompt) || `You are a research sub-agent spawned by PRE (Personal Reasoning Engine). Your task is to complete the following assignment and return a concise, factual summary of your findings.

RULES:
- Focus only on the assigned task
- Use tools to gather information (read files, search, fetch web pages)
- Be thorough but concise
- Return your findings as a clear summary
- Do NOT ask follow-up questions — complete the task autonomously
- Maximum 10 tool calls`;

  const deniedTools = overrides && overrides.deniedTools;
  const allowedTools = (overrides && overrides.allowedTools) || AGENT_TOOLS;
  const tools = buildToolDefs().filter(t => {
    const name = t.function?.name;
    if (!name) return false;
    // MCP tools always allowed
    if (name.startsWith('mcp__')) return true;
    // Denylist mode: include everything EXCEPT denied tools
    if (deniedTools) return !deniedTools.includes(name);
    // Allowlist mode (default): include only specified tools
    return allowedTools.includes(name);
  });

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: agent.task },
  ];

  const MAX_TURNS = (overrides && overrides.maxTurns) || 10;
  let turn = 0;
  let didToolCalls = false;

  while (turn < MAX_TURNS) {
    turn++;

    let fullResponse = '';
    const chatResult = await withTimeout(
      streamChat({
        messages,
        tools: tools.length > 0 ? tools : undefined,
        onToken: (event) => {
          if (event && event.type === 'token') fullResponse += event.content || '';
          else if (event && event.type === 'thinking') { /* skip */ }
          else if (event && event.type === 'tool_calls') { /* handled via chatResult */ }
        },
      }),
      INFERENCE_TIMEOUT,
      `Agent inference (turn ${turn})`,
    );
    // Prefer the resolved response string from the Promise
    if (chatResult && chatResult.response) fullResponse = chatResult.response;

    // Check for tool calls — prefer native Ollama tool_calls, fall back to XML parsing
    let toolCalls = [];
    if (chatResult && chatResult.toolCalls && chatResult.toolCalls.length > 0) {
      toolCalls = chatResult.toolCalls.map(tc => ({
        name: tc.function?.name || tc.name,
        args: tc.function?.arguments || tc.arguments || {},
      }));
    } else {
      toolCalls = parseToolCalls(fullResponse);
    }

    if (toolCalls.length === 0) {
      // No tool calls — agent is done, return its text response
      const cleaned = cleanResponse(fullResponse);
      if (cleaned) return cleaned;
      // Empty response after tool calls — fall through to summary
      break;
    }

    didToolCalls = true;

    // Execute tool calls — include tool_calls so the model can see its own calls
    const assistantMsg = { role: 'assistant', content: fullResponse };
    if (chatResult && chatResult.toolCalls) assistantMsg.tool_calls = chatResult.toolCalls;
    messages.push(assistantMsg);
    // Save assistant turn to agent session
    if (agent.sessionId) appendSessionMessage(agent.sessionId, assistantMsg);

    for (const tc of toolCalls) {
      if (onStatus) onStatus({ type: 'agent_tool', id: agent.id, tool: tc.name });

      let output;
      try {
        // Import here to avoid circular dependency
        const { executeTool } = require('../tools');
        output = await executeTool(tc.name, tc.args, cwd);
      } catch (err) {
        output = `Error: ${err.message}`;
      }

      // Strip base64 screenshots from browser/computer results — sub-agents
      // can't see images, and the base64 noise wastes context. The text field
      // (added by navigate/read) gives the model what it actually needs.
      if (typeof output === 'string' && output.startsWith('{')) {
        try {
          const parsed = JSON.parse(output);
          if (parsed.screenshot) {
            delete parsed.screenshot;
            output = JSON.stringify(parsed);
          }
        } catch {}
      }

      const toolMsg = {
        role: 'tool',
        content: typeof output === 'string' ? output : JSON.stringify(output),
      };
      messages.push(toolMsg);
      // Save tool result to agent session
      if (agent.sessionId) appendSessionMessage(agent.sessionId, toolMsg);
    }
  }

  // Turns exhausted (or empty response after tools) — force a summary WITHOUT tools
  if (didToolCalls) {
    messages.push({
      role: 'user',
      content: 'Your tool turns are complete. Now summarize ALL findings from your research above. Include every specific fact, number, date, statistic, and source URL you gathered. Output raw findings only.',
    });
    let summary = '';
    const summaryResult = await withTimeout(
      streamChat({
        messages,
        maxTokens: 16384, // Generous limit — agent summaries feed directly into report synthesis
        // No tools — force a text response
        onToken: (event) => {
          if (event && event.type === 'token') summary += event.content || '';
        },
      }),
      INFERENCE_TIMEOUT,
      'Agent summary generation',
    );
    summary = (summaryResult && summaryResult.response) || summary;
    if (summary.trim()) return summary.trim();
  }

  return 'Agent reached maximum turns without producing findings.';
}

/**
 * Spawn multiple agents in parallel and collect results.
 * Although Ollama serializes model requests, agents spend significant time
 * on tool execution (web_fetch, bash, file reads) which runs concurrently.
 * This yields real speedups for research-heavy multi-agent workflows.
 */
async function spawnMulti(args, cwd, onStatus) {
  const { tasks } = args;
  if (!tasks || !Array.isArray(tasks) || tasks.length === 0) {
    return 'Error: tasks must be a non-empty array of task descriptions';
  }

  if (tasks.length > 5) {
    return 'Error: maximum 5 tasks allowed';
  }

  if (onStatus) onStatus({ type: 'agent_progress', current: 0, total: tasks.length, task: `Launching ${tasks.length} parallel agents` });

  let completed = 0;
  const promises = tasks.map((task, i) => {
    return spawnAgent({ task }, cwd, (event) => {
      // Forward all status events with the task index
      if (onStatus) onStatus({ ...event, taskIndex: i, taskTotal: tasks.length });
      // Track completion for progress updates
      if (event.type === 'agent_completed' || event.type === 'agent_failed') {
        completed++;
        if (onStatus) onStatus({ type: 'agent_task_done', current: completed, total: tasks.length, task, preview: (event.type === 'agent_completed' ? 'Completed' : 'Failed') });
      }
    }).then(result => ({ task, result, index: i, status: 'fulfilled' }))
      .catch(err => ({ task, result: `Error: ${err.message}`, index: i, status: 'rejected' }));
  });

  const results = await Promise.all(promises);

  // Sort by original index for consistent output order
  results.sort((a, b) => a.index - b.index);

  return results.map(r =>
    `## Task ${r.index + 1}: ${r.task}\n${r.result}`
  ).join('\n\n---\n\n');
}

/**
 * List active and recent agents
 */
function listAgents() {
  const list = Object.values(agents);
  if (list.length === 0) return 'No agents have been spawned in this session.';

  return list.map(a => {
    const duration = a.completedAt
      ? `${((a.completedAt - a.startedAt) / 1000).toFixed(1)}s`
      : 'running...';
    return `[${a.id}] ${a.status} (${duration}) — ${a.task.slice(0, 100)}`;
  }).join('\n');
}

/**
 * Parse tool calls from model response (supports both XML and JSON formats)
 */
function parseToolCalls(text) {
  const calls = [];

  // Try XML format: <tool_call>{"name":"...","arguments":{...}}</tool_call>
  const xmlRegex = /<tool_call>\s*(\{[\s\S]*?\})\s*<\/tool_call>/g;
  let match;
  while ((match = xmlRegex.exec(text)) !== null) {
    try {
      const parsed = JSON.parse(match[1]);
      if (parsed.name) {
        calls.push({
          name: parsed.name,
          args: parsed.arguments || {},
        });
      }
    } catch {}
  }

  return calls;
}

/**
 * Clean up response text by removing tool call tags
 */
function cleanResponse(text) {
  return text
    .replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '')
    .trim();
}

module.exports = { spawnAgent, spawnMulti, listAgents };
