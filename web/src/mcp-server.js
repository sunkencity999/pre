// PRE MCP Server — Exposes PRE as an MCP tool provider
// Allows Claude Desktop, Claude Code, or any MCP client to delegate tasks
// to PRE's local Gemma 4 model, saving API tokens on execution-heavy work.

'use strict';

const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { z } = require('zod');

const PORT = process.env.PRE_WEB_PORT || 7749;
const BASE_URL = `http://localhost:${PORT}`;

/**
 * Create an MCP server instance with PRE's tools.
 * All tool handlers call the PRE REST API — works identically
 * whether connected via SSE (in-process) or stdio (separate process).
 */
function createMcpServer() {
  const server = new McpServer({
    name: 'pre',
    version: '1.0.0',
  });

  // ── pre_agent: Full agentic task execution ──
  server.tool(
    'pre_agent',
    {
      description: [
        "Run a task through PRE's local AI agent (Gemma 4 26B on Apple Silicon) with 60+ tools.",
        'Tools include: bash, file ops, web search/scrape, email (Mail.app), calendar, contacts,',
        'reminders, notes, GitHub, Jira, Slack, Confluence, SharePoint, Linear, Zoom, Figma,',
        'Asana, image generation, document creation, computer use, and more.',
        'The agent runs a multi-turn tool loop locally — no API tokens consumed.',
        'Use this for execution-heavy tasks that would burn tokens on tool loops.',
      ].join(' '),
    },
    {
      task: z.string().describe('The task for the agent to execute'),
      max_turns: z.number().optional().describe('Maximum tool loop iterations (default 15, max 30)'),
    },
    async ({ task, max_turns }) => {
      try {
        const res = await fetch(`${BASE_URL}/api/mcp-server/agent`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task, maxTurns: max_turns }),
          signal: AbortSignal.timeout(300_000),
        });
        if (!res.ok) {
          const body = await res.text();
          throw new Error(`PRE server returned ${res.status}: ${body}`);
        }
        const data = await res.json();

        let text = data.response || 'No response generated.';
        if (data.toolsUsed?.length) {
          text += `\n\n[Tools used: ${data.toolsUsed.join(', ')}]`;
        }
        if (data.turns > 1) {
          text += `\n[Agent turns: ${data.turns}]`;
        }
        return { content: [{ type: 'text', text }] };
      } catch (err) {
        return { content: [{ type: 'text', text: `PRE agent error: ${err.message}` }], isError: true };
      }
    }
  );

  // ── pre_chat: Simple one-shot query (no tools) ──
  server.tool(
    'pre_chat',
    {
      description: [
        'Send a one-shot query to the local Gemma 4 26B model. No tools, no agent loop —',
        'just a direct question/answer. Fast and free. Good for: summarization, analysis,',
        'brainstorming, code review, translation, or any task that needs LLM reasoning',
        'without tool access.',
      ].join(' '),
      readOnlyHint: true,
    },
    {
      message: z.string().describe('The message to send'),
      system_prompt: z.string().optional().describe('Optional system prompt to set context'),
    },
    async ({ message, system_prompt }) => {
      try {
        const res = await fetch(`${BASE_URL}/api/mcp-server/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, systemPrompt: system_prompt }),
          signal: AbortSignal.timeout(120_000),
        });
        if (!res.ok) throw new Error(`PRE server returned ${res.status}`);
        const data = await res.json();
        return { content: [{ type: 'text', text: data.response || 'No response.' }] };
      } catch (err) {
        return { content: [{ type: 'text', text: `PRE chat error: ${err.message}` }], isError: true };
      }
    }
  );

  // ── pre_memory_search: Query persistent memory ──
  server.tool(
    'pre_memory_search',
    {
      description: [
        "Search PRE's persistent memory. PRE accumulates memories across sessions about the user,",
        'their projects, preferences, feedback, and reference material. Returns matching entries.',
      ].join(' '),
      readOnlyHint: true,
    },
    {
      query: z.string().describe('Search query for memory (keyword match)'),
    },
    async ({ query }) => {
      try {
        const res = await fetch(`${BASE_URL}/api/mcp-server/memory`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        });
        if (!res.ok) throw new Error(`PRE server returned ${res.status}`);
        const data = await res.json();

        if (!data.memories?.length) {
          return { content: [{ type: 'text', text: 'No matching memories found.' }] };
        }

        const text = data.memories.map(m =>
          `## ${m.name} (${m.type})\n${m.description || ''}\n\n${m.body}`
        ).join('\n\n---\n\n');

        return { content: [{ type: 'text', text }] };
      } catch (err) {
        return { content: [{ type: 'text', text: `Memory search error: ${err.message}` }], isError: true };
      }
    }
  );

  // ── pre_sessions: List recent sessions ──
  server.tool(
    'pre_sessions',
    {
      description: 'List recent PRE conversation sessions with names and message counts. Useful for understanding what tasks have been worked on recently.',
      readOnlyHint: true,
    },
    {
      limit: z.number().optional().describe('Max sessions to return (default 10)'),
    },
    async ({ limit }) => {
      try {
        const res = await fetch(`${BASE_URL}/api/mcp-server/sessions?limit=${limit || 10}`);
        if (!res.ok) throw new Error(`PRE server returned ${res.status}`);
        const data = await res.json();

        if (!data.sessions?.length) {
          return { content: [{ type: 'text', text: 'No sessions found.' }] };
        }

        const text = data.sessions.map(s =>
          `- **${s.displayName || s.id}** — ${s.messageCount} messages (${s.updated || 'unknown'})`
        ).join('\n');

        return { content: [{ type: 'text', text }] };
      } catch (err) {
        return { content: [{ type: 'text', text: `Sessions error: ${err.message}` }], isError: true };
      }
    }
  );

  return server;
}

module.exports = { createMcpServer };
