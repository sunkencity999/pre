// PRE Web GUI — Session management (JSONL format, shared with CLI)

const fs = require('fs');
const path = require('path');
const { SESSIONS_DIR } = require('./constants');

// Ensure sessions directory exists
if (!fs.existsSync(SESSIONS_DIR)) {
  fs.mkdirSync(SESSIONS_DIR, { recursive: true });
}

// Session display names stored in a metadata file
const META_FILE = path.join(SESSIONS_DIR, '.meta.json');

function loadMeta() {
  try {
    return JSON.parse(fs.readFileSync(META_FILE, 'utf-8'));
  } catch {
    return {};
  }
}

function saveMeta(meta) {
  fs.writeFileSync(META_FILE, JSON.stringify(meta, null, 2));
}

/**
 * List all sessions with metadata, optionally filtered by project
 */
function listSessions(filterProjectSlug) {
  const meta = loadMeta();
  const projectMap = meta._projectMap || {};
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
        displayName: meta[id] || null,
        projectSlug: projectMap[id] || null,
        preview,
        turnCount,
        modified: stat.mtime.toISOString(),
        size: stat.size,
      };
    })
    .sort((a, b) => new Date(b.modified) - new Date(a.modified));

  if (filterProjectSlug !== undefined) {
    return files.filter(s => s.projectSlug === filterProjectSlug);
  }
  return files;
}

// ── Project management ──

function listProjects() {
  const meta = loadMeta();
  const projects = meta._projects || {};
  const projectMap = meta._projectMap || {};
  // Count sessions per project
  const counts = {};
  for (const slug of Object.values(projectMap)) {
    counts[slug] = (counts[slug] || 0) + 1;
  }
  return Object.entries(projects)
    .map(([slug, p]) => ({
      slug,
      name: p.name,
      created: p.created,
      order: p.order ?? 0,
      sessionCount: counts[slug] || 0,
    }))
    .sort((a, b) => a.order - b.order);
}

function createProject(name) {
  const meta = loadMeta();
  if (!meta._projects) meta._projects = {};
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
    || `project-${Date.now().toString(36)}`;
  // Avoid duplicates
  let finalSlug = slug;
  if (meta._projects[finalSlug]) {
    finalSlug = `${slug}-${Date.now().toString(36)}`;
  }
  meta._projects[finalSlug] = {
    name: name.trim(),
    created: new Date().toISOString(),
    order: Object.keys(meta._projects).length,
  };
  saveMeta(meta);
  return { slug: finalSlug, name: name.trim() };
}

function renameProject(slug, newName) {
  const meta = loadMeta();
  if (!meta._projects || !meta._projects[slug]) return false;
  meta._projects[slug].name = newName.trim();
  saveMeta(meta);
  return true;
}

function deleteProject(slug) {
  const meta = loadMeta();
  if (!meta._projects || !meta._projects[slug]) return false;
  delete meta._projects[slug];
  // Ungroup all sessions in this project
  if (meta._projectMap) {
    for (const [sid, ps] of Object.entries(meta._projectMap)) {
      if (ps === slug) delete meta._projectMap[sid];
    }
  }
  saveMeta(meta);
  return true;
}

/**
 * Ensure a project exists by slug, creating it if needed. Returns the slug.
 */
function ensureProject(slug, name) {
  const meta = loadMeta();
  if (!meta._projects) meta._projects = {};
  if (!meta._projects[slug]) {
    meta._projects[slug] = {
      name: name || slug,
      created: new Date().toISOString(),
      order: Object.keys(meta._projects).length,
    };
    saveMeta(meta);
  }
  return slug;
}

function moveSessionToProject(sessionId, projectSlug) {
  const meta = loadMeta();
  if (!meta._projectMap) meta._projectMap = {};
  if (projectSlug) {
    meta._projectMap[sessionId] = projectSlug;
  } else {
    delete meta._projectMap[sessionId];
  }
  saveMeta(meta);
  return true;
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
function createSession(project = 'web', channel = 'general', forceNew = false, projectSlug = null) {
  let id = `${project}:${channel}`;
  let filePath = path.join(SESSIONS_DIR, `${id}.jsonl`);
  // If forceNew and base ID exists, append a timestamp to make it unique
  if (forceNew && fs.existsSync(filePath)) {
    const suffix = Date.now().toString(36);
    id = `${project}:${channel}-${suffix}`;
    filePath = path.join(SESSIONS_DIR, `${id}.jsonl`);
  }
  if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, '');
  }
  // Assign to project if specified
  if (projectSlug) {
    moveSessionToProject(id, projectSlug);
  }
  return id;
}

/**
 * Rename a session (display name only — file stays the same)
 */
function renameSession(sessionId, newName) {
  const meta = loadMeta();
  if (newName && newName.trim()) {
    meta[sessionId] = newName.trim();
  } else {
    delete meta[sessionId];
  }
  saveMeta(meta);
  return true;
}

/**
 * Delete a session file
 */
function deleteSession(sessionId) {
  const filePath = path.join(SESSIONS_DIR, `${sessionId}.jsonl`);
  if (!fs.existsSync(filePath)) return false;
  fs.unlinkSync(filePath);
  // Clean up display name and project mapping
  const meta = loadMeta();
  let changed = false;
  if (meta[sessionId]) { delete meta[sessionId]; changed = true; }
  if (meta._projectMap && meta._projectMap[sessionId]) { delete meta._projectMap[sessionId]; changed = true; }
  if (changed) saveMeta(meta);
  return true;
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
 * Get session messages formatted for Ollama API.
 * Filters out display-only records (artifacts, documents, images) that are
 * persisted for UI rendering but should not be sent to the model.
 */
function getSessionMessages(sessionId) {
  return getSession(sessionId).filter(m => m.role !== 'display');
}

module.exports = {
  listSessions,
  getSession,
  appendMessage,
  createSession,
  deleteSession,
  renameSession,
  rewindSession,
  getSessionMessages,
  listProjects,
  createProject,
  renameProject,
  deleteProject,
  moveSessionToProject,
  ensureProject,
};
