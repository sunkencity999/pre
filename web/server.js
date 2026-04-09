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
const { MODEL_CTX, ARTIFACTS_DIR } = require('./src/constants');

const PORT = parseInt(process.env.PRE_WEB_PORT || '7749', 10);
const CWD = process.env.PRE_CWD || os.homedir();

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

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
      if (msg.image) userMsg.images = [msg.image];
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

// ── Start ──

server.listen(PORT, () => {
  console.log(`\n  PRE Web GUI running at http://localhost:${PORT}`);
  console.log(`  Working directory: ${CWD}`);
  console.log(`  Model context: ${MODEL_CTX} tokens\n`);
});
