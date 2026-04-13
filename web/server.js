// PRE Web GUI — Express + WebSocket server
// Phase 2: Full tool execution loop

const express = require('express');
const http = require('http');
const { WebSocketServer } = require('ws');
const path = require('path');
const os = require('os');

const { healthCheck } = require('./src/ollama');
const { runToolLoop } = require('./src/tools');
const {
  listSessions, getSession, appendMessage,
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
const delegate = require('./src/tools/delegate');
const mcp = require('./src/mcp');
const hooksSystem = require('./src/hooks');
const experienceSystem = require('./src/experience');
const chronosSystem = require('./src/chronos');

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
      const needsTitle = userMsgCount === 0 && !sessionMeta?.displayName;
      console.log(`[ws] needsTitle=${needsTitle} userMsgCount=${userMsgCount} displayName=${sessionMeta?.displayName}`);

      // Save user message to session
      const userMsg = { role: 'user', content: userContent };
      // Support single image (legacy) or multiple images array
      if (msg.images && msg.images.length > 0) {
        userMsg.images = msg.images;
      } else if (msg.image) {
        userMsg.images = [msg.image];
      }
      appendMessage(sessionId, userMsg);

      // Run the full tool loop
      abortController = new AbortController();
      try {
        await runToolLoop({
          sessionId,
          cwd: CWD,
          signal: abortController.signal,
          userMessage: needsTitle ? userContent : null,
          needsTitle,
          send: (event) => {
            if (ws.readyState === 1) {
              ws.send(JSON.stringify(event));
            }
          },
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
