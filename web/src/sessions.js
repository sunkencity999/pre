// PRE Web GUI — Session management (JSONL format, shared with CLI)

const fs = require('fs');
const path = require('path');
const { SESSIONS_DIR } = require('./constants');

// Ensure sessions directory exists
if (!fs.existsSync(SESSIONS_DIR)) {
  fs.mkdirSync(SESSIONS_DIR, { recursive: true });
}

/**
 * List all sessions with metadata
 */
function listSessions() {
  const files = fs.readdirSync(SESSIONS_DIR)
    .filter(f => f.endsWith('.jsonl'))
    .map(f => {
      const fullPath = path.join(SESSIONS_DIR, f);
      const stat = fs.statSync(fullPath);
      const id = f.replace('.jsonl', '');
      // Parse ID format: project:channel
      const [project, channel] = id.includes(':') ? id.split(':') : ['global', id];
      // Get first user message as preview
      let preview = '';
      let turnCount = 0;
      try {
        const content = fs.readFileSync(fullPath, 'utf-8');
        const lines = content.split('\n').filter(Boolean);
        turnCount = lines.length;
        for (const line of lines) {
          try {
            const msg = JSON.parse(line);
            if (msg.role === 'user' && !msg.content.startsWith('[Previous conversation')) {
              preview = msg.content.slice(0, 120);
              break;
            }
          } catch {}
        }
      } catch {}

      return {
        id,
        project,
        channel,
        preview,
        turnCount,
        modified: stat.mtime.toISOString(),
        size: stat.size,
      };
    })
    .sort((a, b) => new Date(b.modified) - new Date(a.modified));

  return files;
}

/**
 * Get full session history as messages array
 */
function getSession(sessionId) {
  const filePath = path.join(SESSIONS_DIR, `${sessionId}.jsonl`);
  if (!fs.existsSync(filePath)) return [];

  const content = fs.readFileSync(filePath, 'utf-8');
  const messages = [];
  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      messages.push(JSON.parse(line));
    } catch {}
  }
  return messages;
}

/**
 * Append a message to session JSONL
 */
function appendMessage(sessionId, message) {
  const filePath = path.join(SESSIONS_DIR, `${sessionId}.jsonl`);
  const line = JSON.stringify(message) + '\n';
  fs.appendFileSync(filePath, line);
}

/**
 * Create a new session, return its ID
 */
function createSession(project = 'web', channel = 'general') {
  const id = `${project}:${channel}`;
  const filePath = path.join(SESSIONS_DIR, `${id}.jsonl`);
  // Don't overwrite existing session
  if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, '');
  }
  return id;
}

/**
 * Rewind a session by removing the last N turns (user + assistant pairs)
 */
function rewindSession(sessionId, turns = 1) {
  const messages = getSession(sessionId);
  // Remove last N*2 entries (user + assistant)
  const removeCount = turns * 2;
  const kept = messages.slice(0, Math.max(0, messages.length - removeCount));
  const filePath = path.join(SESSIONS_DIR, `${sessionId}.jsonl`);
  fs.writeFileSync(filePath, kept.map(m => JSON.stringify(m)).join('\n') + (kept.length ? '\n' : ''));
  return kept;
}

/**
 * Get session messages formatted for Ollama API
 */
function getSessionMessages(sessionId) {
  return getSession(sessionId);
}

module.exports = {
  listSessions,
  getSession,
  appendMessage,
  createSession,
  rewindSession,
  getSessionMessages,
};
