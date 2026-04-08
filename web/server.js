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
  createSession, deleteSession, rewindSession,
} = require('./src/sessions');
const { MODEL_CTX, ARTIFACTS_DIR } = require('./src/constants');

const PORT = parseInt(process.env.PRE_WEB_PORT || '7749', 10);
const CWD = process.env.PRE_CWD || os.homedir();

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json({ limit: '10mb' }));

// Serve artifacts (for iframe viewing)
app.use('/artifacts', express.static(ARTIFACTS_DIR));

// ── REST API ──

app.get('/api/sessions', (_req, res) => {
  res.json(listSessions());
});

app.get('/api/sessions/:id', (req, res) => {
  const messages = getSession(req.params.id);
  res.json(messages);
});

app.post('/api/sessions/new', (req, res) => {
  const { project, channel } = req.body || {};
  const id = createSession(project || 'web', channel || 'general');
  res.json({ id });
});

app.delete('/api/sessions/:id', (req, res) => {
  const ok = deleteSession(req.params.id);
  if (!ok) return res.status(404).json({ error: 'Session not found' });
  res.json({ deleted: req.params.id });
});

app.post('/api/rewind', (req, res) => {
  const { sessionId, turns } = req.body || {};
  if (!sessionId) return res.status(400).json({ error: 'sessionId required' });
  const remaining = rewindSession(sessionId, turns || 1);
  res.json({ remaining: remaining.length });
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
      const userContent = msg.content;
      if (!userContent || !userContent.trim()) return;

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

// ── Start ──

server.listen(PORT, () => {
  console.log(`\n  PRE Web GUI running at http://localhost:${PORT}`);
  console.log(`  Working directory: ${CWD}`);
  console.log(`  Model context: ${MODEL_CTX} tokens\n`);
});
