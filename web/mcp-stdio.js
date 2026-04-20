#!/usr/bin/env node
// PRE MCP — Stdio transport entry point
// Auto-starts Ollama and PRE server if not already running.
// Configure in Claude Desktop / Claude Code as:
//   { "command": "node", "args": ["/path/to/pre/web/mcp-stdio.js"] }

'use strict';

const { spawn } = require('child_process');
const path = require('path');

const PORT = process.env.PRE_WEB_PORT || 7749;
const BASE_URL = `http://localhost:${PORT}`;
const OLLAMA_URL = `http://localhost:${process.env.OLLAMA_PORT || 11434}`;

function log(msg) {
  process.stderr.write(`[pre-mcp] ${msg}\n`);
}

async function isReachable(url, timeout = 2000) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(timeout) });
    return res.ok;
  } catch { return false; }
}

async function waitFor(checkFn, label, maxWaitSec) {
  for (let i = 0; i < maxWaitSec; i++) {
    if (await checkFn()) return true;
    if (i === 0) log(`Waiting for ${label}...`);
    await new Promise(r => setTimeout(r, 1000));
  }
  return false;
}

async function ensureOllama() {
  if (await isReachable(`${OLLAMA_URL}/api/version`)) return;

  log('Starting Ollama...');
  const child = spawn('ollama', ['serve'], {
    detached: true,
    stdio: 'ignore',
  });
  child.unref();

  const ok = await waitFor(
    () => isReachable(`${OLLAMA_URL}/api/version`),
    'Ollama', 15
  );
  if (!ok) throw new Error('Ollama failed to start within 15 seconds');
  log('Ollama ready.');
}

async function ensureServer() {
  if (await isReachable(`${BASE_URL}/api/status`)) return;

  log('Starting PRE server...');
  const serverPath = path.join(__dirname, 'server.js');
  const child = spawn(process.execPath, [serverPath], {
    detached: true,
    stdio: 'ignore',
    cwd: __dirname,
    env: { ...process.env, PRE_MCP_CHILD: '1' },
  });
  child.unref();

  const ok = await waitFor(
    () => isReachable(`${BASE_URL}/api/status`),
    'PRE server', 30
  );
  if (!ok) throw new Error('PRE server failed to start within 30 seconds');
  log('PRE server ready.');
}

async function main() {
  await ensureOllama();
  await ensureServer();

  // Lazy-require SDK after startup checks (keeps initial load fast)
  const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
  const { createMcpServer } = require('./src/mcp-server');

  const server = createMcpServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  log('MCP stdio transport connected.');
}

main().catch(err => {
  log(`Fatal: ${err.message}`);
  process.exit(1);
});
