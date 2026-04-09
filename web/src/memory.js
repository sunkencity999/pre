// PRE Web GUI — Enhanced Memory System
// Inspired by Claude Code's memory architecture
// File-based persistent memory with frontmatter, index, age annotations,
// auto-extraction, and type-aware context injection.

const fs = require('fs');
const path = require('path');
const os = require('os');
const { MEMORY_DIR } = require('./constants');
const { streamChat } = require('./ollama');

// Ensure global memory directory exists
if (!fs.existsSync(MEMORY_DIR)) {
  fs.mkdirSync(MEMORY_DIR, { recursive: true });
}

const INDEX_FILE = 'MEMORY.md';
const VALID_TYPES = ['user', 'feedback', 'project', 'reference'];
const MAX_MEMORIES_IN_CONTEXT = 30;
const MAX_BODY_LINES = 30;

// ── Frontmatter parsing ──

function parseFrontmatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { meta: {}, body: content.trim() };

  const meta = {};
  for (const line of match[1].split('\n')) {
    const idx = line.indexOf(':');
    if (idx > 0) {
      const key = line.slice(0, idx).trim();
      const val = line.slice(idx + 1).trim();
      meta[key] = val;
    }
  }
  return { meta, body: match[2].trim() };
}

function buildFrontmatter(meta) {
  return '---\n' + Object.entries(meta).map(([k, v]) => `${k}: ${v}`).join('\n') + '\n---';
}

// ── File operations ──

/**
 * Scan a memory directory and return parsed entries
 */
function scanMemoryDir(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter(f => f.endsWith('.md') && f !== INDEX_FILE && f !== 'index.md')
    .map(filename => {
      try {
        const filePath = path.join(dir, filename);
        const stat = fs.statSync(filePath);
        const content = fs.readFileSync(filePath, 'utf-8');
        const { meta, body } = parseFrontmatter(content);

        return {
          filename,
          filePath,
          name: meta.name || filename.replace('.md', ''),
          description: meta.description || '',
          type: VALID_TYPES.includes(meta.type) ? meta.type : 'project',
          scope: meta.scope || 'global',
          body,
          mtimeMs: stat.mtimeMs,
          created: meta.created || stat.birthtime?.toISOString().slice(0, 10),
        };
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .sort((a, b) => b.mtimeMs - a.mtimeMs);
}

/**
 * Get all memories (global + optional project scope)
 */
function getAllMemories(projectDir) {
  const memories = scanMemoryDir(MEMORY_DIR);
  if (projectDir) {
    const projMemDir = path.join(projectDir, 'memory');
    const projMemories = scanMemoryDir(projMemDir);
    projMemories.forEach(m => { m.scope = 'project'; });
    memories.push(...projMemories);
  }
  return memories;
}

/**
 * Sanitize a memory name to a safe filename
 */
function toFilename(name) {
  let fn = name.replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
  if (!fn) fn = 'untitled';
  return fn + '.md';
}

/**
 * Save a memory file and update the index
 */
function saveMemory({ name, type, description, content, scope, projectDir }) {
  if (!name) return { error: 'name is required' };
  if (!content) return { error: 'content is required' };
  type = VALID_TYPES.includes(type) ? type : 'project';

  const filename = toFilename(name);
  const targetDir = (scope === 'project' && projectDir)
    ? path.join(projectDir, 'memory')
    : MEMORY_DIR;

  if (!fs.existsSync(targetDir)) {
    fs.mkdirSync(targetDir, { recursive: true });
  }

  const filePath = path.join(targetDir, filename);
  const meta = {
    name,
    description: description || name,
    type,
    scope: scope || 'global',
    created: new Date().toISOString().slice(0, 10),
  };
  const md = buildFrontmatter(meta) + '\n\n' + content + '\n';
  fs.writeFileSync(filePath, md, 'utf-8');

  // Update index
  updateIndex(targetDir, filename, name, description || name);

  return { success: true, filename, path: filePath };
}

/**
 * Delete a memory by name or filename
 */
function deleteMemory(query, projectDir) {
  if (!query) return { error: 'query is required' };
  const q = query.toLowerCase();

  const dirs = [MEMORY_DIR];
  if (projectDir) dirs.push(path.join(projectDir, 'memory'));

  for (const dir of dirs) {
    const memories = scanMemoryDir(dir);
    const match = memories.find(m =>
      m.name.toLowerCase().includes(q) || m.filename.toLowerCase().includes(q)
    );
    if (match) {
      fs.unlinkSync(match.filePath);
      removeFromIndex(dir, match.filename);
      return { success: true, deleted: match.name, filename: match.filename };
    }
  }
  return { error: `No memory found matching '${query}'` };
}

/**
 * Search memories by query
 */
function searchMemories(query, projectDir) {
  const memories = getAllMemories(projectDir);
  if (!query) return memories;

  const q = query.toLowerCase();
  return memories.filter(m => {
    const text = `${m.name} ${m.description} ${m.type} ${m.body}`.toLowerCase();
    return text.includes(q);
  });
}

// ── Index management ──

function updateIndex(dir, filename, name, description) {
  const idxPath = path.join(dir, INDEX_FILE);
  let lines = [];

  if (fs.existsSync(idxPath)) {
    lines = fs.readFileSync(idxPath, 'utf-8').split('\n');
  }

  // Find and replace existing entry, or append
  const ref = `(${filename})`;
  const newLine = `- [${name}](${filename}) — ${description}`;
  let found = false;

  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes(ref)) {
      lines[i] = newLine;
      found = true;
      break;
    }
  }

  if (!found) {
    if (lines.length === 0) lines.push('# PRE Memory', '');
    lines.push(newLine);
  }

  fs.writeFileSync(idxPath, lines.join('\n') + '\n', 'utf-8');

  // Also update legacy index.md if it exists
  const legacyIdx = path.join(dir, 'index.md');
  if (fs.existsSync(legacyIdx) && legacyIdx !== idxPath) {
    try {
      let legacy = fs.readFileSync(legacyIdx, 'utf-8');
      if (!legacy.includes(ref)) {
        legacy += newLine + '\n';
        fs.writeFileSync(legacyIdx, legacy, 'utf-8');
      }
    } catch {}
  }
}

function removeFromIndex(dir, filename) {
  for (const idxName of [INDEX_FILE, 'index.md']) {
    const idxPath = path.join(dir, idxName);
    if (!fs.existsSync(idxPath)) continue;
    try {
      const content = fs.readFileSync(idxPath, 'utf-8');
      const filtered = content.split('\n')
        .filter(line => !line.includes(`(${filename})`))
        .join('\n');
      fs.writeFileSync(idxPath, filtered, 'utf-8');
    } catch {}
  }
}

// ── Age annotations ──

/**
 * Generate an age annotation for a memory based on modification time
 */
function memoryAge(mtimeMs) {
  const days = Math.floor((Date.now() - mtimeMs) / (1000 * 60 * 60 * 24));
  if (days <= 1) return '';
  if (days <= 7) return `[${days} days old]`;
  if (days <= 30) return `[${Math.floor(days / 7)} weeks old — verify before acting on it]`;
  return `[${Math.floor(days / 30)} months old — may be outdated, verify against current state]`;
}

// ── Context injection ──

/**
 * Build memory context for system prompt injection.
 * Type-aware ordering: feedback first (behavioral guidance), then user, project, reference.
 */
function buildMemoryContext(projectDir) {
  const memories = getAllMemories(projectDir);
  if (memories.length === 0) return '';

  // Sort by type priority, then recency
  const typePriority = { feedback: 0, user: 1, project: 2, reference: 3 };
  memories.sort((a, b) => {
    const pa = typePriority[a.type] ?? 4;
    const pb = typePriority[b.type] ?? 4;
    if (pa !== pb) return pa - pb;
    return b.mtimeMs - a.mtimeMs;
  });

  const capped = memories.slice(0, MAX_MEMORIES_IN_CONTEXT);

  let ctx = '<memory>\n';
  for (const m of capped) {
    const age = memoryAge(m.mtimeMs);
    const bodyLines = m.body.split('\n').slice(0, MAX_BODY_LINES).join('\n');
    ctx += `## ${m.name} [${m.type}] ${age}\n${bodyLines}\n\n`;
  }
  ctx += '</memory>';
  return ctx;
}

/**
 * Build the memory instructions block for the system prompt.
 * Tells the model how and when to use memory tools.
 */
function buildMemoryInstructions() {
  return `
MEMORY SYSTEM:
You have persistent memory across sessions stored in ~/.pre/memory/.
Memory types: user (who they are), feedback (how to work), project (what's happening), reference (where to look).

When to save memories:
- User shares role, preferences, or expertise → save as "user" type
- User corrects your approach or confirms a non-obvious choice → save as "feedback" type
- You learn about ongoing work, decisions, deadlines → save as "project" type
- You discover external resource locations → save as "reference" type

Rules:
- Check existing memories before saving (avoid duplicates)
- One fact per memory. Update or replace stale ones.
- Convert relative dates to absolute (e.g. "next Thursday" → "2026-04-16")
- Do NOT save: code patterns (read the code), git history (use git), debugging fixes (they're in the code)
- For feedback type: include Why and How to apply lines
- If the user explicitly asks you to remember something, save it immediately
`;
}

// ── Auto-extraction ──

// Throttling state: avoid hammering the GPU after every turn
const extractionState = {
  turnsSinceLastExtract: 0,
  lastExtractTime: 0,
  running: false,
};

const EXTRACT_EVERY_N_TURNS = 3;       // Only run extraction every N conversation turns
const EXTRACT_COOLDOWN_MS = 60 * 1000; // Minimum 60s between extractions
const EXTRACT_MIN_USER_CHARS = 50;     // Skip if user messages are too short (just commands)

/**
 * Check whether extraction should run this turn.
 * Returns false if throttled, the model is busy, or there's nothing substantive.
 */
function shouldExtract(messages) {
  // Don't run concurrent extractions
  if (extractionState.running) {
    console.log('[memory-extract] Skipped: already running');
    return false;
  }

  extractionState.turnsSinceLastExtract++;

  // Frequency gate
  if (extractionState.turnsSinceLastExtract < EXTRACT_EVERY_N_TURNS) {
    return false;
  }

  // Cooldown gate
  const elapsed = Date.now() - extractionState.lastExtractTime;
  if (elapsed < EXTRACT_COOLDOWN_MS) {
    return false;
  }

  // Content gate: need substantive user messages in recent window
  const recentUserMsgs = messages.slice(-6)
    .filter(m => m.role === 'user')
    .map(m => m.content || '');
  const totalChars = recentUserMsgs.reduce((sum, c) => sum + c.length, 0);
  if (totalChars < EXTRACT_MIN_USER_CHARS) {
    return false;
  }

  return true;
}

/**
 * Extract memories from a conversation using a lightweight LLM pass.
 * Runs after qualifying assistant responses to capture implicit memory-worthy facts.
 *
 * @param {Array} messages - Recent conversation messages
 * @param {Array} existingMemories - Already saved memories (to avoid duplicates)
 * @returns {Array} Extracted memory objects [{name, type, description, content}]
 */
async function extractMemories(messages, existingMemories) {
  // Only look at the last few turns to keep extraction focused
  const recentMessages = messages.slice(-6);

  // Skip if no user messages in recent window
  const hasUserMsg = recentMessages.some(m => m.role === 'user');
  if (!hasUserMsg) return [];

  // Build a compact summary of what's already remembered
  const existingSummary = existingMemories.slice(0, 30)
    .map(m => `- [${m.type}] ${m.name}: ${m.description}`)
    .join('\n');

  const extractionPrompt = `You are the memory extraction subagent for PRE (Personal Reasoning Engine).
Analyze the recent conversation and extract facts worth remembering across sessions.

EXISTING MEMORIES (do not duplicate):
${existingSummary || '(none)'}

MEMORY TYPES:
- user: User's role, preferences, expertise, how they like to work
- feedback: Corrections to approach, confirmed good approaches (include Why + How to apply)
- project: Ongoing work, decisions, deadlines, context not in code/git
- reference: Pointers to external systems, URLs, tool locations

DO NOT EXTRACT:
- Code patterns, architecture, file paths (derivable from code)
- Git history or recent changes (use git log)
- Debugging solutions (the fix is in the code)
- Ephemeral task state (what's happening right now in this conversation)
- Anything already in existing memories

OUTPUT FORMAT:
Return a JSON array of memory objects. If nothing is worth saving, return [].
Each object: {"name": "short_name", "type": "user|feedback|project|reference", "description": "one line", "content": "the memory content"}

Be conservative — only extract what will genuinely help in future conversations. Most turns produce no memories.`;

  const conversationText = recentMessages
    .filter(m => m.role === 'user' || m.role === 'assistant')
    .map(m => `${m.role}: ${(m.content || '').slice(0, 1000)}`)
    .join('\n\n');

  try {
    const result = await streamChat({
      messages: [
        { role: 'system', content: extractionPrompt },
        { role: 'user', content: `Extract memories from this conversation:\n\n${conversationText}` },
      ],
      maxTokens: 1024,
    });

    const response = result.response || '';
    // Extract JSON from response (may be wrapped in markdown code fence)
    const jsonMatch = response.match(/\[[\s\S]*?\]/);
    if (!jsonMatch) return [];

    const parsed = JSON.parse(jsonMatch[0]);
    if (!Array.isArray(parsed)) return [];

    // Validate each entry
    return parsed.filter(m =>
      m.name && m.type && m.content &&
      VALID_TYPES.includes(m.type) &&
      m.name.length < 128
    );
  } catch (err) {
    console.log(`[memory-extract] Error: ${err.message}`);
    return [];
  }
}

/**
 * Run auto-extraction: analyze recent messages and save any new memories.
 * Designed to run in the background after a conversation turn.
 * Throttled to avoid GPU contention — runs every 3 turns with 60s cooldown.
 */
async function autoExtract(messages, projectDir) {
  // Throttle check
  if (!shouldExtract(messages)) {
    return [];
  }

  extractionState.running = true;
  extractionState.turnsSinceLastExtract = 0;
  extractionState.lastExtractTime = Date.now();

  try {
    const existing = getAllMemories(projectDir);
    const extracted = await extractMemories(messages, existing);

    if (extracted.length === 0) return [];

    const saved = [];
    for (const mem of extracted) {
      // Check for duplicates (fuzzy name match)
      const isDupe = existing.some(e =>
        e.name.toLowerCase() === mem.name.toLowerCase() ||
        (e.description.toLowerCase() === (mem.description || '').toLowerCase() && e.type === mem.type)
      );
      if (isDupe) continue;

      const result = saveMemory({
        name: mem.name,
        type: mem.type,
        description: mem.description || mem.name,
        content: mem.content,
        projectDir,
      });
      if (result.success) {
        saved.push(mem);
        console.log(`[memory-extract] Saved: [${mem.type}] ${mem.name}`);
      }
    }
    return saved;
  } catch (err) {
    console.log(`[memory-extract] Auto-extract error: ${err.message}`);
    return [];
  } finally {
    extractionState.running = false;
  }
}

// ── REST API helpers ──

/**
 * List all memories formatted for API response
 */
function listForAPI(projectDir) {
  return getAllMemories(projectDir).map(m => ({
    filename: m.filename,
    name: m.name,
    description: m.description,
    type: m.type,
    scope: m.scope,
    body: m.body,
    age: memoryAge(m.mtimeMs),
    modified: new Date(m.mtimeMs).toISOString().slice(0, 10),
  }));
}

/**
 * Get a single memory by filename
 */
function getMemory(filename, projectDir) {
  const dirs = [MEMORY_DIR];
  if (projectDir) dirs.push(path.join(projectDir, 'memory'));

  for (const dir of dirs) {
    const filePath = path.join(dir, filename);
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf-8');
      const stat = fs.statSync(filePath);
      const { meta, body } = parseFrontmatter(content);
      return {
        filename,
        name: meta.name || filename.replace('.md', ''),
        description: meta.description || '',
        type: meta.type || 'project',
        scope: meta.scope || 'global',
        body,
        age: memoryAge(stat.mtimeMs),
        modified: new Date(stat.mtimeMs).toISOString().slice(0, 10),
      };
    }
  }
  return null;
}

module.exports = {
  // Core operations
  saveMemory,
  deleteMemory,
  searchMemories,
  getAllMemories,
  getMemory,

  // Context injection
  buildMemoryContext,
  buildMemoryInstructions,

  // Auto-extraction
  autoExtract,

  // API helpers
  listForAPI,

  // Utilities
  parseFrontmatter,
  memoryAge,
  scanMemoryDir,
};
