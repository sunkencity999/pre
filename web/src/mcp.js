// PRE Web GUI — MCP (Model Context Protocol) Client Manager
// Connects to MCP servers, discovers tools, and routes tool calls.

const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { StreamableHTTPClientTransport } = require('@modelcontextprotocol/sdk/client/streamableHttp.js');
const fs = require('fs');
const path = require('path');

const CONFIG_PATH = path.join(process.env.HOME || '/tmp', '.pre', 'mcp.json');

// Active MCP client connections: { serverName: { client, transport, tools, info } }
const connections = {};

/**
 * Load MCP server configuration from ~/.pre/mcp.json
 * Format: { "servers": { "name": { "command": "...", "args": [...], "env": {...} } } }
 * Or for HTTP: { "servers": { "name": { "url": "https://..." } } }
 */
function loadConfig() {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const raw = fs.readFileSync(CONFIG_PATH, 'utf-8');
      return JSON.parse(raw);
    }
  } catch (err) {
    console.error('[mcp] Failed to load config:', err.message);
  }
  return { servers: {} };
}

function saveConfig(config) {
  const dir = path.dirname(CONFIG_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
}

/**
 * Connect to a single MCP server
 */
async function connectServer(name, serverConfig) {
  if (connections[name]) {
    console.log(`[mcp] ${name} already connected`);
    return connections[name];
  }

  console.log(`[mcp] Connecting to ${name}...`);
  const client = new Client({ name: 'pre', version: '1.0.0' });

  let transport;
  if (serverConfig.url) {
    // HTTP/SSE transport
    transport = new StreamableHTTPClientTransport(new URL(serverConfig.url));
  } else if (serverConfig.command) {
    // Stdio transport — spawn process
    const env = { ...process.env, ...(serverConfig.env || {}) };
    transport = new StdioClientTransport({
      command: serverConfig.command,
      args: serverConfig.args || [],
      env,
    });
  } else {
    throw new Error(`MCP server '${name}' has no command or url configured`);
  }

  try {
    await client.connect(transport);

    // Discover tools
    const toolsResult = await client.listTools();
    const tools = toolsResult.tools || [];

    const conn = {
      client,
      transport,
      tools,
      info: { name, ...serverConfig },
      connectedAt: Date.now(),
    };
    connections[name] = conn;
    console.log(`[mcp] ${name} connected — ${tools.length} tools available`);
    return conn;
  } catch (err) {
    console.error(`[mcp] Failed to connect to ${name}:`, err.message);
    throw err;
  }
}

/**
 * Disconnect from a single MCP server
 */
async function disconnectServer(name) {
  const conn = connections[name];
  if (!conn) return;
  try {
    await conn.client.close();
  } catch {}
  delete connections[name];
  console.log(`[mcp] ${name} disconnected`);
}

/**
 * Connect to all configured MCP servers
 */
async function connectAll() {
  const config = loadConfig();
  const servers = config.servers || {};
  const results = {};

  for (const [name, serverConfig] of Object.entries(servers)) {
    if (serverConfig.disabled) continue;
    try {
      await connectServer(name, serverConfig);
      results[name] = { connected: true, tools: connections[name].tools.length };
    } catch (err) {
      results[name] = { connected: false, error: err.message };
    }
  }
  return results;
}

/**
 * Disconnect all MCP servers
 */
async function disconnectAll() {
  for (const name of Object.keys(connections)) {
    await disconnectServer(name);
  }
}

/**
 * Get all tools from all connected MCP servers, formatted for Ollama
 * Returns array of tool definitions in Ollama's function-calling format
 */
function getAllTools() {
  const tools = [];
  for (const [serverName, conn] of Object.entries(connections)) {
    for (const tool of conn.tools) {
      // Prefix tool name with server name to avoid collisions
      // Use __ as separator so server names with underscores don't break parsing
      const prefixedName = `mcp__${serverName}__${tool.name}`;
      tools.push({
        type: 'function',
        function: {
          name: prefixedName,
          description: `[MCP: ${serverName}] ${tool.description || tool.name}`,
          parameters: tool.inputSchema || { type: 'object', properties: {} },
        },
        _mcp: { server: serverName, originalName: tool.name },
      });
    }
  }
  return tools;
}

/**
 * Call an MCP tool by its prefixed name
 * @param {string} prefixedName - e.g. "mcp_myserver_tool_name"
 * @param {object} args - tool arguments
 * @returns {string} tool result text
 */
async function callTool(prefixedName, args) {
  // Parse the prefixed name to find server and tool
  const match = prefixedName.match(/^mcp__(.+?)__(.+)$/);
  if (!match) throw new Error(`Invalid MCP tool name: ${prefixedName}`);

  const [, serverName, toolName] = match;
  const conn = connections[serverName];
  if (!conn) throw new Error(`MCP server '${serverName}' is not connected`);

  try {
    const result = await conn.client.callTool({ name: toolName, arguments: args });

    // Format the result content array into a string
    if (!result.content || result.content.length === 0) {
      return result.isError ? 'Error: tool returned no content' : '(no output)';
    }

    return result.content.map(block => {
      if (block.type === 'text') return block.text;
      if (block.type === 'image') return `[Image: ${block.mimeType}, ${block.data?.length || 0} bytes]`;
      if (block.type === 'resource') return `[Resource: ${block.resource?.uri || 'unknown'}]`;
      return JSON.stringify(block);
    }).join('\n');
  } catch (err) {
    throw new Error(`MCP ${serverName}/${toolName}: ${err.message}`);
  }
}

/**
 * Check if a tool name is an MCP tool
 */
function isMCPTool(name) {
  return name.startsWith('mcp__');
}

/**
 * Get status of all configured and connected servers
 */
function getStatus() {
  const config = loadConfig();
  const servers = config.servers || {};
  const status = {};

  for (const [name, serverConfig] of Object.entries(servers)) {
    const conn = connections[name];
    status[name] = {
      configured: true,
      disabled: !!serverConfig.disabled,
      connected: !!conn,
      tools: conn ? conn.tools.length : 0,
      toolNames: conn ? conn.tools.map(t => t.name) : [],
      type: serverConfig.url ? 'http' : 'stdio',
      command: serverConfig.command,
      url: serverConfig.url,
    };
  }
  return status;
}

/**
 * Add a new MCP server to the config
 */
function addServer(name, serverConfig) {
  const config = loadConfig();
  if (!config.servers) config.servers = {};
  config.servers[name] = serverConfig;
  saveConfig(config);
  return true;
}

/**
 * Remove an MCP server from the config and disconnect it
 */
async function removeServer(name) {
  await disconnectServer(name);
  const config = loadConfig();
  if (config.servers) {
    delete config.servers[name];
    saveConfig(config);
  }
  return true;
}

module.exports = {
  loadConfig,
  saveConfig,
  connectServer,
  disconnectServer,
  connectAll,
  disconnectAll,
  getAllTools,
  callTool,
  isMCPTool,
  getStatus,
  addServer,
  removeServer,
};
