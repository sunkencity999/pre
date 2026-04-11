// PRE Web GUI — Sub-agent spawning and management
// Allows the model to spawn parallel research tasks as independent Ollama requests.

const { streamChat } = require('../ollama');
const { buildSystemPrompt } = require('../context');
const { buildToolDefs } = require('../tools-defs');
const { MODEL_CTX } = require('../constants');

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
 * @returns {string} result summary
 */
async function spawnAgent(args, cwd, onStatus) {
  const { task } = args;
  if (!task) return 'Error: task description is required';

  const id = `agent_${++agentCounter}`;
  const agent = {
    id,
    task,
    status: 'running',
    startedAt: Date.now(),
    messages: [],
    result: null,
  };
  agents[id] = agent;

  if (onStatus) onStatus({ type: 'agent_started', id, task });

  try {
    const result = await runAgent(agent, cwd, onStatus);
    agent.status = 'completed';
    agent.result = result;
    agent.completedAt = Date.now();
    if (onStatus) onStatus({ type: 'agent_completed', id, duration: agent.completedAt - agent.startedAt });
    return result;
  } catch (err) {
    agent.status = 'failed';
    agent.result = `Error: ${err.message}`;
    agent.completedAt = Date.now();
    if (onStatus) onStatus({ type: 'agent_failed', id, error: err.message });
    return agent.result;
  }
}

/**
 * Run the agent's internal loop — single-turn or multi-turn with tools
 */
async function runAgent(agent, cwd, onStatus) {
  const systemPrompt = `You are a research sub-agent spawned by PRE (Personal Reasoning Engine). Your task is to complete the following assignment and return a concise, factual summary of your findings.

RULES:
- Focus only on the assigned task
- Use tools to gather information (read files, search, fetch web pages)
- Be thorough but concise
- Return your findings as a clear summary
- Do NOT ask follow-up questions — complete the task autonomously
- Maximum 10 tool calls`;

  const tools = buildToolDefs().filter(t =>
    AGENT_TOOLS.includes(t.function?.name) || t.function?.name?.startsWith('mcp__')
  );

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: agent.task },
  ];

  const MAX_TURNS = 10;
  let turn = 0;

  while (turn < MAX_TURNS) {
    turn++;

    // Call Ollama — streamChat returns a Promise directly
    let fullResponse = '';
    const chatResult = await streamChat({
      messages,
      tools: tools.length > 0 ? tools : undefined,
      onToken: (event) => {
        if (event && event.type === 'token') fullResponse += event.content || '';
        else if (event && event.type === 'thinking') { /* skip */ }
        else if (event && event.type === 'tool_calls') { /* handled via chatResult */ }
      },
    });
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
      // No tool calls — agent is done
      return cleanResponse(fullResponse);
    }

    // Execute tool calls
    messages.push({ role: 'assistant', content: fullResponse });

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

      messages.push({
        role: 'tool',
        content: typeof output === 'string' ? output : JSON.stringify(output),
      });
    }
  }

  // Max turns reached — return whatever we have
  const lastAssistant = messages.filter(m => m.role === 'assistant').pop();
  return lastAssistant ? cleanResponse(lastAssistant.content) : 'Agent reached maximum turns without a final answer.';
}

/**
 * Spawn multiple agents sequentially and collect results.
 * Runs one at a time because Ollama processes requests serially per model.
 * Streams status updates so the user sees progress.
 */
async function spawnMulti(args, cwd, onStatus) {
  const { tasks } = args;
  if (!tasks || !Array.isArray(tasks) || tasks.length === 0) {
    return 'Error: tasks must be a non-empty array of task descriptions';
  }

  if (tasks.length > 5) {
    return 'Error: maximum 5 tasks allowed';
  }

  const results = [];
  for (let i = 0; i < tasks.length; i++) {
    const task = tasks[i];
    if (onStatus) onStatus({ type: 'agent_progress', current: i + 1, total: tasks.length, task });

    const result = await spawnAgent({ task }, cwd, onStatus);
    results.push({ task, result, index: i });

    if (onStatus) onStatus({ type: 'agent_task_done', current: i + 1, total: tasks.length, task, preview: result.slice(0, 200) });
  }

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
