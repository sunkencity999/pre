// PRE Web GUI — Express + WebSocket server
// Phase 2: Full tool execution loop

const express = require('express');
const http = require('http');
const { WebSocketServer } = require('ws');
const path = require('path');
const os = require('os');
const fs = require('fs');
const { execSync } = require('child_process');

const { healthCheck, streamChat } = require('./src/ollama');
const { runToolLoop } = require('./src/tools');
const {
  listSessions, getSession, getSessionMessages, appendMessage,
  createSession, deleteSession, renameSession, rewindSession,
  listProjects, createProject, renameProject, deleteProject, moveSessionToProject,
} = require('./src/sessions');
const memorySystem = require('./src/memory');
const {
  listConnections, setApiKey, removeConnection,
  setGoogleCredentials, getGoogleAuthUrl, exchangeGoogleCode,
  setMicrosoftCredentials, getMicrosoftAuthUrl, exchangeMicrosoftCode,
  setTelegramChatId, testTelegramToken,
  setJiraConfig,
  setConfluenceConfig,
  setZoomConfig,
} = require('./src/connections');
const { MODEL_CTX, ARTIFACTS_DIR } = require('./src/constants');
const cronSystem = require('./src/tools/cron');
const { executeCronJob, checkMissedJobs } = require('./src/cron-runner');
const { runDeepResearch } = require('./src/research');
const exportTool = require('./src/tools/export');
const delegate = require('./src/tools/delegate');
const mcp = require('./src/mcp');
const hooksSystem = require('./src/hooks');
const experienceSystem = require('./src/experience');
const chronosSystem = require('./src/chronos');
const { createMcpServer } = require('./src/mcp-server');
const { StreamableHTTPServerTransport } = require('@modelcontextprotocol/sdk/server/streamableHttp.js');

const PORT = parseInt(process.env.PRE_WEB_PORT || '7749', 10);
const CWD = process.env.PRE_CWD || os.homedir();

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server, maxPayload: 50 * 1024 * 1024 }); // 50MB for image uploads

// Serve static files (no caching in dev to avoid stale JS/CSS)
app.use(express.static(path.join(__dirname, 'public'), {
  etag: false,
  maxAge: 0,
  setHeaders: (res) => {
    res.set('Cache-Control', 'no-store, no-cache, must-revalidate');
  },
}));
app.use(express.json({ limit: '10mb' }));

// Serve artifacts (for iframe viewing and document downloads)
app.use('/artifacts', express.static(ARTIFACTS_DIR, {
  setHeaders: (res, filePath) => {
    // Force download for document types
    const ext = path.extname(filePath).toLowerCase();
    if (['.docx', '.xlsx', '.pdf', '.txt', '.xml'].includes(ext)) {
      res.set('Content-Disposition', `attachment; filename="${path.basename(filePath)}"`);
    }
  },
}));

// ── REST API ──

app.get('/api/sessions', (_req, res) => {
  res.json(listSessions());
});

app.get('/api/sessions/:id', (req, res) => {
  const messages = getSession(decodeURIComponent(req.params.id));
  res.json(messages);
});

app.post('/api/sessions/new', (req, res) => {
  const { project, channel, projectSlug } = req.body || {};
  const id = createSession(project || 'web', channel || 'general', true, projectSlug || null);
  res.json({ id });
});

app.delete('/api/sessions/:id', (req, res) => {
  const id = decodeURIComponent(req.params.id);
  const ok = deleteSession(id);
  if (!ok) return res.status(404).json({ error: 'Session not found' });
  res.json({ deleted: id });
});

app.post('/api/sessions/:id/rename', (req, res) => {
  const id = decodeURIComponent(req.params.id);
  const { name } = req.body || {};
  renameSession(id, name);
  res.json({ id, name: name || null });
});

app.post('/api/rewind', (req, res) => {
  const { sessionId, turns } = req.body || {};
  if (!sessionId) return res.status(400).json({ error: 'sessionId required' });
  const remaining = rewindSession(sessionId, turns || 1);
  res.json({ remaining: remaining.length });
});

// ── Projects API ──

app.get('/api/projects', (_req, res) => {
  res.json(listProjects());
});

app.post('/api/projects', (req, res) => {
  const { name } = req.body || {};
  if (!name || !name.trim()) return res.status(400).json({ error: 'name required' });
  const project = createProject(name);
  res.json(project);
});

app.post('/api/projects/:slug/rename', (req, res) => {
  const slug = decodeURIComponent(req.params.slug);
  const { name } = req.body || {};
  if (!name || !name.trim()) return res.status(400).json({ error: 'name required' });
  const ok = renameProject(slug, name);
  if (!ok) return res.status(404).json({ error: 'Project not found' });
  res.json({ slug, name: name.trim() });
});

app.delete('/api/projects/:slug', (req, res) => {
  const slug = decodeURIComponent(req.params.slug);
  const ok = deleteProject(slug);
  if (!ok) return res.status(404).json({ error: 'Project not found' });
  res.json({ deleted: slug });
});

app.post('/api/sessions/:id/move', (req, res) => {
  const id = decodeURIComponent(req.params.id);
  const { projectSlug } = req.body || {};
  moveSessionToProject(id, projectSlug || null);
  res.json({ id, projectSlug: projectSlug || null });
});

// ── Connections API ──

app.get('/api/connections', (_req, res) => {
  res.json(listConnections());
});

app.post('/api/connections/:service/key', (req, res) => {
  const service = req.params.service;
  const { key } = req.body || {};
  if (!key) return res.status(400).json({ error: 'key required' });
  const ok = setApiKey(service, key);
  if (!ok) return res.status(400).json({ error: 'Invalid service' });
  res.json({ service, active: true });
});

app.delete('/api/connections/:service', (req, res) => {
  const service = req.params.service;
  removeConnection(service);
  res.json({ service, active: false });
});

app.post('/api/connections/telegram/test', async (req, res) => {
  const { token } = req.body || {};
  if (!token) return res.status(400).json({ error: 'token required' });
  try {
    const bot = await testTelegramToken(token);
    res.json({ success: true, bot });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.post('/api/connections/telegram/chat-id', (req, res) => {
  const { chatId } = req.body || {};
  if (!chatId) return res.status(400).json({ error: 'chatId required' });
  setTelegramChatId(chatId);
  res.json({ success: true, chatId });
});

app.post('/api/connections/jira/config', (req, res) => {
  const { url, token } = req.body || {};
  if (!url || !token) return res.status(400).json({ error: 'url and token required' });
  setJiraConfig(url, token);
  res.json({ success: true });
});

app.post('/api/connections/confluence/config', (req, res) => {
  const { url, token } = req.body || {};
  if (!url || !token) return res.status(400).json({ error: 'url and token required' });
  setConfluenceConfig(url, token);
  res.json({ success: true });
});

app.post('/api/connections/zoom/config', (req, res) => {
  const { accountId, clientId, clientSecret } = req.body || {};
  if (!accountId || !clientId || !clientSecret) {
    return res.status(400).json({ error: 'accountId, clientId, and clientSecret required' });
  }
  setZoomConfig(accountId, clientId, clientSecret);
  res.json({ success: true });
});

app.post('/api/connections/google/credentials', (req, res) => {
  const { clientId, clientSecret } = req.body || {};
  if (!clientId || !clientSecret) return res.status(400).json({ error: 'clientId and clientSecret required' });
  setGoogleCredentials(clientId, clientSecret);
  res.json({ success: true });
});

app.get('/api/connections/google/auth-url', (req, res) => {
  const url = getGoogleAuthUrl(PORT);
  if (!url) return res.status(400).json({ error: 'Google credentials not configured' });
  res.json({ url });
});

// ── Microsoft / SharePoint OAuth ──

app.post('/api/connections/microsoft/credentials', (req, res) => {
  const { tenantId, clientId, clientSecret } = req.body || {};
  if (!tenantId || !clientId || !clientSecret) {
    return res.status(400).json({ error: 'tenantId, clientId, and clientSecret required' });
  }
  setMicrosoftCredentials(tenantId, clientId, clientSecret);
  res.json({ success: true });
});

app.get('/api/connections/microsoft/auth-url', (req, res) => {
  const url = getMicrosoftAuthUrl(PORT);
  if (!url) return res.status(400).json({ error: 'Microsoft credentials not configured' });
  res.json({ url });
});

// Microsoft OAuth callback — receives the authorization code
app.get('/oauth/microsoft/callback', async (req, res) => {
  const { code, error, error_description } = req.query;
  if (error) {
    return res.send(`<html><body style="font-family:system-ui;background:#0a0a0a;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh"><div style="text-align:center"><h2 style="color:#f87171">Authorization Failed</h2><p>${error_description || error}</p><p>You can close this tab.</p></div></body></html>`);
  }
  if (!code) {
    return res.status(400).send('Missing authorization code');
  }
  try {
    await exchangeMicrosoftCode(code, PORT);
    res.send(`<html><body style="font-family:system-ui;background:#0a0a0a;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh"><div style="text-align:center"><h2 style="color:#4ade80">Microsoft Connected!</h2><p>SharePoint search, files, lists, and pages are now available.</p><p>You can close this tab and return to PRE.</p><script>setTimeout(()=>window.close(),3000)</script></div></body></html>`);
  } catch (err) {
    res.send(`<html><body style="font-family:system-ui;background:#0a0a0a;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh"><div style="text-align:center"><h2 style="color:#f87171">Token Exchange Failed</h2><p>${err.message}</p><p>You can close this tab.</p></div></body></html>`);
  }
});

// Google OAuth callback — receives the authorization code
app.get('/oauth/callback', async (req, res) => {
  const { code, error } = req.query;
  if (error) {
    return res.send(`<html><body style="font-family:system-ui;background:#0a0a0a;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh"><div style="text-align:center"><h2 style="color:#f87171">Authorization Failed</h2><p>${error}</p><p>You can close this tab.</p></div></body></html>`);
  }
  if (!code) {
    return res.status(400).send('Missing authorization code');
  }
  try {
    await exchangeGoogleCode(code, PORT);
    res.send(`<html><body style="font-family:system-ui;background:#0a0a0a;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh"><div style="text-align:center"><h2 style="color:#4ade80">Google Connected!</h2><p>Gmail, Drive, and Docs are now available.</p><p>You can close this tab and return to PRE.</p><script>setTimeout(()=>window.close(),3000)</script></div></body></html>`);
  } catch (err) {
    res.send(`<html><body style="font-family:system-ui;background:#0a0a0a;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh"><div style="text-align:center"><h2 style="color:#f87171">Token Exchange Failed</h2><p>${err.message}</p><p>You can close this tab.</p></div></body></html>`);
  }
});

// ── Memory API ──

app.get('/api/memory', (req, res) => {
  const query = req.query.q || '';
  if (query) {
    const results = memorySystem.searchMemories(query);
    res.json(results.map(m => ({
      filename: m.filename,
      name: m.name,
      description: m.description,
      type: m.type,
      scope: m.scope,
      body: m.body,
      age: memorySystem.memoryAge(m.mtimeMs),
      modified: new Date(m.mtimeMs).toISOString().slice(0, 10),
    })));
  } else {
    res.json(memorySystem.listForAPI());
  }
});

app.get('/api/memory/:filename', (req, res) => {
  const memory = memorySystem.getMemory(req.params.filename);
  if (!memory) return res.status(404).json({ error: 'Memory not found' });
  res.json(memory);
});

app.post('/api/memory', (req, res) => {
  const { name, type, description, content, scope } = req.body || {};
  const result = memorySystem.saveMemory({ name, type, description, content, scope });
  if (result.error) return res.status(400).json({ error: result.error });
  res.json(result);
});

app.delete('/api/memory/:filename', (req, res) => {
  const result = memorySystem.deleteMemory(req.params.filename);
  if (result.error) return res.status(404).json({ error: result.error });
  res.json(result);
});

// Reveal an artifact in Finder (macOS)
app.post('/api/artifacts/reveal', (req, res) => {
  const { filePath } = req.body || {};
  if (!filePath) return res.status(400).json({ error: 'filePath required' });

  // Resolve the web path to an absolute filesystem path
  const resolved = filePath.startsWith('/artifacts/')
    ? path.join(ARTIFACTS_DIR, filePath.replace(/^\/artifacts\//, ''))
    : filePath;

  const fs = require('fs');
  if (!fs.existsSync(resolved)) {
    return res.status(404).json({ error: 'File not found' });
  }

  // Use osascript to reveal in Finder (non-blocking)
  const { exec } = require('child_process');
  exec(`open -R "${resolved.replace(/"/g, '\\"')}"`, (err) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json({ ok: true });
  });
});

// Export an artifact to PDF, PNG, or self-contained HTML
app.post('/api/artifacts/export', async (req, res) => {
  const { path: artPath, format, title } = req.body || {};
  if (!artPath) return res.status(400).json({ error: 'path required' });
  if (!['pdf', 'png', 'html'].includes(format)) {
    return res.status(400).json({ error: 'format must be pdf, png, or html' });
  }

  // Derive title from filename if not provided
  const exportTitle = title || artPath.replace(/^\/artifacts\//, '').replace(/\.[^.]+$/, '').replace(/-[a-z0-9]+$/, '').replace(/-/g, ' ');

  try {
    let result;
    switch (format) {
      case 'pdf':
        result = await exportTool.exportPdf(artPath, exportTitle);
        break;
      case 'png':
        result = await exportTool.exportPng(artPath, exportTitle);
        break;
      case 'html':
        result = await exportTool.exportSelfContainedHtml(artPath, exportTitle);
        break;
    }
    res.json({ path: result.webPath, filename: result.filename });
  } catch (err) {
    console.error(`[export] Error: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Cron API ──

app.get('/api/cron', (_req, res) => {
  res.json(cronSystem.loadJobs());
});

app.post('/api/cron', (req, res) => {
  const { schedule, prompt, description } = req.body || {};
  if (!schedule || !prompt) {
    return res.status(400).json({ error: 'schedule and prompt are required' });
  }
  const fields = schedule.trim().split(/\s+/);
  if (fields.length !== 5) {
    return res.status(400).json({ error: 'Schedule must be 5 fields: min hour dom month dow' });
  }
  const jobs = cronSystem.loadJobs();
  const job = {
    id: cronSystem.generateId(),
    schedule: schedule.trim(),
    prompt,
    description: description || prompt.slice(0, 80),
    enabled: true,
    created_at: Date.now(),
    last_run_at: null,
    run_count: 0,
  };
  jobs.push(job);
  cronSystem.saveJobs(jobs);
  res.json(job);
});

app.patch('/api/cron/:id', (req, res) => {
  const jobs = cronSystem.loadJobs();
  const job = jobs.find(j => j.id === req.params.id);
  if (!job) return res.status(404).json({ error: 'Job not found' });
  if (req.body.enabled !== undefined) job.enabled = !!req.body.enabled;
  if (req.body.schedule) job.schedule = req.body.schedule;
  if (req.body.prompt) job.prompt = req.body.prompt;
  if (req.body.description) job.description = req.body.description;
  cronSystem.saveJobs(jobs);
  res.json(job);
});

app.delete('/api/cron/:id', (req, res) => {
  const jobs = cronSystem.loadJobs();
  const idx = jobs.findIndex(j => j.id === req.params.id);
  if (idx === -1) return res.status(404).json({ error: 'Job not found' });
  const removed = jobs.splice(idx, 1)[0];
  cronSystem.saveJobs(jobs);
  res.json(removed);
});

app.post('/api/cron/:id/run', (req, res) => {
  const jobs = cronSystem.loadJobs();
  const job = jobs.find(j => j.id === req.params.id);
  if (!job) return res.status(404).json({ error: 'Job not found' });
  // Update run stats
  job.last_run_at = Date.now();
  job.run_count = (job.run_count || 0) + 1;
  cronSystem.saveJobs(jobs);
  // Execute server-side (non-blocking) and notify
  executeCronJob(job, { broadcastWS }).catch(err => {
    console.error(`[cron] Manual run error for ${job.id}: ${err.message}`);
  });
  res.json({ ok: true, id: job.id, run_count: job.run_count });
});

// Broadcast a WS event to all connected clients
function broadcastWS(event) {
  const msg = JSON.stringify(event);
  wss.clients.forEach((client) => {
    if (client.readyState === 1) client.send(msg);
  });
}

// ── Delegate (frontier AI) routes ──
app.get('/api/delegates', (_req, res) => {
  res.json(delegate.checkAvailability());
});

// ── MCP (Model Context Protocol) routes ──
app.get('/api/mcp', (_req, res) => {
  res.json(mcp.getStatus());
});

app.post('/api/mcp/connect', async (req, res) => {
  const { name } = req.body;
  if (!name) return res.status(400).json({ error: 'name required' });
  const config = mcp.loadConfig();
  const serverConfig = config.servers?.[name];
  if (!serverConfig) return res.status(404).json({ error: `Server '${name}' not configured` });
  try {
    await mcp.connectServer(name, serverConfig);
    res.json({ connected: true, tools: mcp.getStatus()[name]?.tools || 0 });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/mcp/disconnect', async (req, res) => {
  const { name } = req.body;
  if (!name) return res.status(400).json({ error: 'name required' });
  await mcp.disconnectServer(name);
  res.json({ disconnected: true });
});

app.post('/api/mcp/add', async (req, res) => {
  const { name, command, args, env, url } = req.body;
  if (!name) return res.status(400).json({ error: 'name required' });
  if (!command && !url) return res.status(400).json({ error: 'command or url required' });
  const serverConfig = url ? { url } : { command, args: args || [], env: env || {} };
  mcp.addServer(name, serverConfig);
  try {
    await mcp.connectServer(name, serverConfig);
    res.json({ added: true, connected: true, tools: mcp.getStatus()[name]?.tools || 0 });
  } catch (err) {
    res.json({ added: true, connected: false, error: err.message });
  }
});

app.delete('/api/mcp/:name', async (req, res) => {
  await mcp.removeServer(req.params.name);
  res.json({ removed: true });
});

// ── Hooks API ──
app.get('/api/hooks', (_req, res) => {
  res.json(hooksSystem.listHooks());
});

app.post('/api/hooks', (req, res) => {
  const result = hooksSystem.addHook(req.body || {});
  if (result.error) return res.status(400).json({ error: result.error });
  res.json(result);
});

app.patch('/api/hooks/:id/toggle', (req, res) => {
  const result = hooksSystem.toggleHook(req.params.id);
  if (result.error) return res.status(404).json({ error: result.error });
  res.json(result);
});

app.delete('/api/hooks/:id', (req, res) => {
  const result = hooksSystem.removeHook(req.params.id);
  if (result.error) return res.status(404).json({ error: result.error });
  res.json(result);
});

// ── Experience Ledger API ──
app.get('/api/experience', (req, res) => {
  const query = req.query.q || '';
  if (query) {
    experienceSystem.searchExperiences(query).then(results => res.json(results)).catch(() => res.json([]));
  } else {
    res.json(experienceSystem.listExperiences());
  }
});

// ── Chronos API ──
app.get('/api/chronos/health', (_req, res) => {
  res.json(chronosSystem.maintenanceSummary());
});

app.get('/api/chronos/staleness', (_req, res) => {
  res.json(chronosSystem.stalenessReport());
});

app.post('/api/chronos/verify/:filename', (req, res) => {
  const result = chronosSystem.verifyMemory(req.params.filename);
  if (result.error) return res.status(404).json({ error: result.error });
  res.json(result);
});

app.post('/api/chronos/maintenance', async (_req, res) => {
  try {
    const result = await chronosSystem.runMaintenance();
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/status', async (_req, res) => {
  const ollamaUp = await healthCheck();
  res.json({
    ollama: ollamaUp,
    cwd: CWD,
    model_ctx: MODEL_CTX,
  });
});

// ── System settings API ──

const PLIST_LABEL = 'com.pre.server';
const PLIST_PATH = path.join(os.homedir(), 'Library', 'LaunchAgents', `${PLIST_LABEL}.plist`);
const PRE_SERVER_SH = path.join(__dirname, 'pre-server.sh');

app.get('/api/system/autostart', (_req, res) => {
  const installed = fs.existsSync(PLIST_PATH);
  let running = false;
  if (installed) {
    try {
      const out = execSync(`launchctl list ${PLIST_LABEL} 2>/dev/null`, { encoding: 'utf-8' });
      running = !out.includes('"Label"') || true; // if launchctl list succeeds, it's loaded
    } catch { /* not loaded */ }
  }
  res.json({ installed, running, plistPath: PLIST_PATH });
});

app.post('/api/system/autostart', (req, res) => {
  const { enabled } = req.body || {};

  if (enabled) {
    // Create and load LaunchAgent
    const webDir = __dirname;
    const plist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${PRE_SERVER_SH}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>WorkingDirectory</key>
    <string>${webDir}</string>
    <key>StandardOutPath</key>
    <string>/tmp/pre-server.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/pre-server.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key>
        <string>${os.homedir()}</string>
        <key>OLLAMA_FLASH_ATTENTION</key>
        <string>0</string>
        <key>OLLAMA_KEEP_ALIVE</key>
        <string>24h</string>
        <key>OLLAMA_NUM_PARALLEL</key>
        <string>1</string>
        <key>OLLAMA_MAX_LOADED_MODELS</key>
        <string>1</string>
    </dict>
</dict>
</plist>`;
    const dir = path.dirname(PLIST_PATH);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(PLIST_PATH, plist);
    try { execSync(`launchctl load "${PLIST_PATH}" 2>/dev/null`); } catch { /* already loaded */ }
    res.json({ installed: true, running: true });
  } else {
    // Remove LaunchAgent plist — don't launchctl unload since that would
    // kill this server process (we ARE the LaunchAgent-managed process).
    // The server keeps running for this session but won't auto-start on next login.
    if (fs.existsSync(PLIST_PATH)) fs.unlinkSync(PLIST_PATH);
    res.json({ installed: false, running: false });
  }
});

// ── MCP Server API (PRE as an MCP tool provider) ──

// Health check — used by mcp-stdio.js to detect if server is up
app.get('/api/health', (_req, res) => res.json({ ok: true }));

// Run a full agent task (multi-turn tool loop)
app.post('/api/mcp-server/agent', async (req, res) => {
  const { task, maxTurns } = req.body || {};
  if (!task) return res.status(400).json({ error: 'task is required' });

  const sessionId = createSession('mcp', 'agent', true);
  appendMessage(sessionId, { role: 'user', content: task });

  const events = [];
  const send = (msg) => events.push(msg);
  const onConfirmRequest = async () => true; // Auto-approve in MCP mode
  const effectiveMaxTurns = Math.min(Math.max(maxTurns || 15, 1), 30);

  try {
    await runToolLoop({
      sessionId,
      cwd: CWD,
      send,
      signal: null,
      onConfirmRequest,
      userMessage: task,
      needsTitle: true,
      maxTurns: effectiveMaxTurns,
    });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }

  // Extract the final assistant response
  const messages = getSessionMessages(sessionId);
  const assistantMsgs = messages.filter(m => m.role === 'assistant');
  const lastAssistant = assistantMsgs[assistantMsgs.length - 1];

  const toolsUsed = events
    .filter(e => e.type === 'tool_call')
    .map(e => e.name)
    .filter(Boolean);

  res.json({
    response: lastAssistant?.content || 'No response generated.',
    sessionId,
    toolsUsed: [...new Set(toolsUsed)],
    turns: assistantMsgs.length,
  });
});

// One-shot chat (no tools, no agent loop)
app.post('/api/mcp-server/chat', async (req, res) => {
  const { message, systemPrompt } = req.body || {};
  if (!message) return res.status(400).json({ error: 'message is required' });

  const messages = [];
  if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
  messages.push({ role: 'user', content: message });

  try {
    const result = await streamChat({ messages, maxTokens: 4096 });
    res.json({
      response: result.response || '',
      thinking: result.thinking || '',
      stats: result.stats,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Search persistent memory
app.post('/api/mcp-server/memory', (req, res) => {
  const { query } = req.body || {};
  const memories = memorySystem.searchMemories(query || '');
  res.json({
    memories: memories.map(m => ({
      name: m.name,
      type: m.type,
      description: m.description,
      body: m.body,
      scope: m.scope,
      created: m.created,
    })),
  });
});

// List recent sessions
app.get('/api/mcp-server/sessions', (req, res) => {
  const limit = parseInt(req.query.limit) || 10;
  const sessions = listSessions();
  res.json({
    sessions: sessions.slice(0, limit).map(s => ({
      id: s.id,
      displayName: s.displayName,
      messageCount: s.messageCount,
      updated: s.updatedAt,
      project: s.project,
    })),
  });
});

// ── MCP HTTP Transport ──
// Connect MCP clients via StreamableHTTP at /mcp
const mcpSessions = new Map();

app.post('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id'];

  // Existing session — route to its transport
  if (sessionId && mcpSessions.has(sessionId)) {
    const session = mcpSessions.get(sessionId);
    return session.transport.handleRequest(req, res);
  }

  // New session — create transport + server
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: () => `mcp-${Date.now()}-${Math.random().toString(36).slice(2, 6)}` });
  const mcpSrv = createMcpServer();

  transport.onclose = () => {
    const sid = transport.sessionId;
    if (sid) mcpSessions.delete(sid);
    console.log(`[mcp-server] HTTP client disconnected (${sid})`);
  };

  await mcpSrv.connect(transport);

  // Store session for subsequent requests
  if (transport.sessionId) {
    mcpSessions.set(transport.sessionId, { transport, server: mcpSrv });
  }

  console.log(`[mcp-server] HTTP client connected (session ${transport.sessionId})`);
  return transport.handleRequest(req, res);
});

app.get('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id'];
  if (sessionId && mcpSessions.has(sessionId)) {
    return mcpSessions.get(sessionId).transport.handleRequest(req, res);
  }
  res.status(400).json({ error: 'MCP session ID required. POST /mcp to start a new session.' });
});

app.delete('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id'];
  if (sessionId && mcpSessions.has(sessionId)) {
    const session = mcpSessions.get(sessionId);
    await session.transport.close();
    mcpSessions.delete(sessionId);
    return res.status(200).json({ closed: true });
  }
  res.status(404).json({ error: 'Unknown MCP session' });
});

// ── WebSocket handler ──

wss.on('connection', (ws) => {
  let sessionId = 'web:general';
  let abortController = null;
  // Pending confirmation requests
  const pendingConfirms = new Map();

  // Ensure session exists
  createSession('web', 'general');

  ws.on('message', async (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }

    if (msg.type === 'switch_session') {
      sessionId = msg.sessionId || 'web:general';
      const history = getSession(sessionId);
      ws.send(JSON.stringify({ type: 'session_history', messages: history, sessionId }));
      return;
    }

    if (msg.type === 'cancel') {
      if (abortController) abortController.abort();
      return;
    }

    // Handle tool confirmation responses
    if (msg.type === 'confirm') {
      const resolver = pendingConfirms.get(msg.id);
      if (resolver) {
        resolver(msg.approved);
        pendingConfirms.delete(msg.id);
      }
      return;
    }

    // ── Delegate to frontier AI ──
    if (msg.type === 'delegate') {
      const target = msg.target; // 'claude', 'codex', 'gemini'
      const content = msg.content;
      if (!target || !content?.trim()) return;

      console.log(`[ws] Delegating to ${target}: ${content.slice(0, 80)}...`);

      // Save user message to session
      appendMessage(sessionId, { role: 'user', content, delegate: target });
      ws.send(JSON.stringify({ type: 'delegate_start', target }));

      try {
        const result = await delegate.execute(target, content, {
          signal: abortController?.signal,
          onToken: (chunk) => {
            if (ws.readyState === 1) {
              ws.send(JSON.stringify({ type: 'delegate_token', target, content: chunk }));
            }
          },
        });

        // Save assistant response to session
        appendMessage(sessionId, { role: 'assistant', content: result.response, delegate: target });
        if (ws.readyState === 1) {
          ws.send(JSON.stringify({
            type: 'delegate_done',
            target,
            response: result.response,
            duration: result.duration,
          }));
        }
      } catch (err) {
        if (ws.readyState === 1) {
          ws.send(JSON.stringify({ type: 'delegate_error', target, message: err.message }));
        }
      }
      return;
    }

    if (msg.type === 'message') {
      console.log(`[ws] Received message: ${msg.content?.slice(0, 80)}...`);
      const userContent = msg.content;
      if (!userContent || !userContent.trim()) return;

      // Check if this session needs an auto-generated title
      const history = getSession(sessionId);
      const userMsgCount = history.filter(m => m.role === 'user').length;
      const sessions = listSessions();
      const sessionMeta = sessions.find(s => s.id === sessionId);
      // Generate a title if the session has no display name yet — not just on the first message.
      // Previous logic (userMsgCount === 0) meant sessions that hit max tool turns or got
      // aborted on the first message would never get titled.
      const needsTitle = !sessionMeta?.displayName;
      console.log(`[ws] needsTitle=${needsTitle} userMsgCount=${userMsgCount} displayName=${sessionMeta?.displayName}`);

      // Save user message to session
      const userMsg = { role: 'user', content: userContent };
      // Support single image (legacy) or multiple images array
      if (msg.images && msg.images.length > 0) {
        userMsg.images = msg.images;
      } else if (msg.image) {
        userMsg.images = [msg.image];
      }
      // Auto-save uploaded images to artifacts so they can be referenced in HTML artifacts.
      // Stores paths in userMsg.imagePaths for the model to use.
      if (userMsg.images && userMsg.images.length > 0) {
        const uploadsDir = path.join(ARTIFACTS_DIR, 'uploads');
        if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
        userMsg.imagePaths = [];
        for (let i = 0; i < userMsg.images.length; i++) {
          try {
            const buf = Buffer.from(userMsg.images[i], 'base64');
            // Detect format from magic bytes
            const ext = (buf[0] === 0x89 && buf[1] === 0x50) ? 'png'
              : (buf[0] === 0xFF && buf[1] === 0xD8) ? 'jpg'
              : (buf[0] === 0x47 && buf[1] === 0x49) ? 'gif' : 'png';
            const fname = `upload-${Date.now()}-${i}.${ext}`;
            fs.writeFileSync(path.join(uploadsDir, fname), buf);
            userMsg.imagePaths.push(`/artifacts/uploads/${fname}`);
          } catch {}
        }
        if (userMsg.imagePaths.length > 0) {
          // Append path info to message so the model knows where to find the images
          userMsg.content += `\n[Uploaded image(s) saved to: ${userMsg.imagePaths.join(', ')}. Use these paths in HTML artifacts with <img src="..."> to embed them.]`;
        }
      }
      appendMessage(sessionId, userMsg);

      // Helper to send WS events safely
      const wsSend = (event) => {
        if (ws.readyState === 1) ws.send(JSON.stringify(event));
      };

      abortController = new AbortController();

      // ── Deep Research mode ──
      if (msg.researchMode) {
        const frontier = msg.useFrontier || false;
        console.log(`[ws] Deep Research mode for: ${userContent.slice(0, 80)}... (frontier: ${frontier || 'off'})`);
        try {
          await runDeepResearch({
            sessionId,
            query: userContent,
            cwd: CWD,
            send: wsSend,
            signal: abortController.signal,
            useFrontier: frontier,
          });
        } catch (err) {
          if (err.message !== 'Research cancelled' && ws.readyState === 1) {
            ws.send(JSON.stringify({ type: 'error', message: err.message }));
          }
          wsSend({ type: 'done', stats: {}, context: { used: 0, max: 0, pct: 0 } });
        }
        abortController = null;
        return;
      }

      // ── Normal tool loop ──
      try {
        await runToolLoop({
          sessionId,
          cwd: CWD,
          signal: abortController.signal,
          userMessage: needsTitle ? userContent : null,
          needsTitle,
          send: wsSend,
          onConfirmRequest: (id, toolName, toolArgs) => {
            return new Promise((resolve) => {
              pendingConfirms.set(id, resolve);
              if (ws.readyState === 1) {
                ws.send(JSON.stringify({
                  type: 'confirm_request',
                  id,
                  tool: toolName,
                  args: toolArgs,
                }));
              }
              // Auto-timeout after 60s
              setTimeout(() => {
                if (pendingConfirms.has(id)) {
                  pendingConfirms.delete(id);
                  resolve(false);
                }
              }, 60000);
            });
          },
        });
      } catch (err) {
        if (err.message !== 'Request aborted' && ws.readyState === 1) {
          ws.send(JSON.stringify({ type: 'error', message: err.message }));
        }
      }
      abortController = null;
    }
  });
});

// ── Graceful shutdown ──

function shutdown() {
  console.log('\n  PRE Web GUI shutting down...');
  wss.clients.forEach(ws => ws.close());
  server.close();
  process.exit(0);
}

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

// ── Cron background loop ──
// Check every 30 seconds. Jobs fire at most once per minute.

let lastCronMinute = -1;

setInterval(() => {
  const now = new Date();
  const currentMinute = now.getHours() * 60 + now.getMinutes();
  // Only check once per minute
  if (currentMinute === lastCronMinute) return;
  lastCronMinute = currentMinute;

  const jobs = cronSystem.loadJobs();
  let changed = false;

  for (const job of jobs) {
    if (!job.enabled) continue;
    if (!cronSystem.matchesNow(job.schedule)) continue;

    // Prevent double-fire within the same minute
    if (job.last_run_at) {
      const lastRun = new Date(job.last_run_at);
      if (lastRun.getHours() === now.getHours() && lastRun.getMinutes() === now.getMinutes()
          && lastRun.getDate() === now.getDate()) continue;
    }

    console.log(`[cron] Firing job ${job.id}: "${job.description}"`);
    job.last_run_at = Date.now();
    job.run_count = (job.run_count || 0) + 1;
    changed = true;

    // Execute server-side — runs headlessly, stores in own session, sends notifications
    executeCronJob(job, { broadcastWS }).catch(err => {
      console.error(`[cron] Execution error for job ${job.id}: ${err.message}`);
    });
  }

  if (changed) cronSystem.saveJobs(jobs);
}, 30000);

// ── Start ──

server.listen(PORT, async () => {
  const jobs = cronSystem.loadJobs();
  const enabledCount = jobs.filter(j => j.enabled).length;
  console.log(`\n  PRE Web GUI running at http://localhost:${PORT}`);
  console.log(`  Working directory: ${CWD}`);
  console.log(`  Model context: ${MODEL_CTX} tokens`);
  console.log(`  Cron jobs: ${enabledCount} active / ${jobs.length} total`);
  console.log(`  MCP server: HTTP at /mcp | stdio via mcp-stdio.js`);

  // Check for cron jobs that were missed while the system was down
  if (enabledCount > 0) {
    checkMissedJobs({ broadcastWS }).then(fired => {
      if (fired) console.log('  [cron] Missed jobs queued for execution');
    }).catch(err => {
      console.error(`  [cron] Missed job check failed: ${err.message}`);
    });
  }

  // Auto-connect MCP servers
  const mcpConfig = mcp.loadConfig();
  const mcpCount = Object.keys(mcpConfig.servers || {}).length;
  if (mcpCount > 0) {
    const results = await mcp.connectAll();
    const connected = Object.values(results).filter(r => r.connected).length;
    const totalTools = Object.values(results).reduce((sum, r) => sum + (r.tools || 0), 0);
    console.log(`  MCP servers: ${connected}/${mcpCount} connected (${totalTools} tools)`);
  }
  console.log('');
});
