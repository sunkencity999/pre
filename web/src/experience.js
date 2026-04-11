// PRE Web GUI — Experience Ledger
// Post-task reflection: extracts lessons learned from completed tool loops.
// Unlike auto-extraction (which saves facts about the user), the experience
// ledger captures metacognitive insights: what worked, what failed, and why.
//
// Entries are saved as "experience" type memories with structured fields:
//   task, approach, outcome, lesson
//
// Uses embedding-based similarity to avoid duplicate lessons and to enable
// semantic retrieval at inference time.

const fs = require('fs');
const path = require('path');
const { streamChat, embed } = require('./ollama');
const { MEMORY_DIR } = require('./constants');
const { saveMemory, getAllMemories, parseFrontmatter } = require('./memory');

const EXPERIENCE_DIR = path.join(MEMORY_DIR, 'experience');
const EMBEDDINGS_FILE = path.join(EXPERIENCE_DIR, 'embeddings.json');

// Ensure directory exists
if (!fs.existsSync(EXPERIENCE_DIR)) {
  fs.mkdirSync(EXPERIENCE_DIR, { recursive: true });
}

// In-memory embedding cache: { filename: { vector, text } }
let embeddingCache = null;

function loadEmbeddings() {
  if (embeddingCache) return embeddingCache;
  try {
    if (fs.existsSync(EMBEDDINGS_FILE)) {
      embeddingCache = JSON.parse(fs.readFileSync(EMBEDDINGS_FILE, 'utf-8'));
      return embeddingCache;
    }
  } catch {}
  embeddingCache = {};
  return embeddingCache;
}

function saveEmbeddings() {
  if (!embeddingCache) return;
  fs.writeFileSync(EMBEDDINGS_FILE, JSON.stringify(embeddingCache));
}

/**
 * Cosine similarity between two vectors
 */
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB) || 1);
}

/**
 * Check if a lesson is too similar to an existing one
 */
function isDuplicate(lessonText, threshold = 0.85) {
  const cache = loadEmbeddings();
  // Without embeddings yet, fall back to substring check
  if (Object.keys(cache).length === 0) return false;
  // Will be checked properly after embedding is computed
  return false;
}

async function isDuplicateWithEmbedding(vector, threshold = 0.85) {
  const cache = loadEmbeddings();
  for (const entry of Object.values(cache)) {
    if (entry.vector && cosineSim(vector, entry.vector) > threshold) {
      return true;
    }
  }
  return false;
}

// Throttle: don't reflect on every single tool loop
const reflectionState = {
  lastReflectTime: 0,
  running: false,
};
const REFLECT_COOLDOWN_MS = 120 * 1000; // 2 minutes between reflections
const MIN_TOOL_CALLS = 2; // Only reflect if the loop used multiple tools

/**
 * Run a post-task reflection on a completed tool loop.
 * Extracts lessons learned and saves them to the experience ledger.
 *
 * @param {Array} messages - Full conversation messages from the session
 * @param {Object} opts - { sessionId, cwd }
 * @returns {Array} Saved experience entries
 */
async function reflect(messages, opts = {}) {
  // Gate: cooldown
  if (reflectionState.running) return [];
  const elapsed = Date.now() - reflectionState.lastReflectTime;
  if (elapsed < REFLECT_COOLDOWN_MS) return [];

  // Gate: only reflect on conversations with substantive tool use
  const toolMessages = messages.filter(m => m.role === 'tool');
  if (toolMessages.length < MIN_TOOL_CALLS) return [];

  reflectionState.running = true;
  reflectionState.lastReflectTime = Date.now();

  try {
    // Build a compact summary of the conversation for reflection
    const recentMessages = messages.slice(-20); // Last 20 messages max
    const conversationSummary = recentMessages
      .filter(m => m.role === 'user' || m.role === 'assistant' || m.role === 'tool')
      .map(m => {
        if (m.role === 'tool') {
          // Summarize tool results compactly
          const content = m.content || '';
          const toolMatch = content.match(/<tool_response name="([^"]+)">/);
          const toolName = toolMatch ? toolMatch[1] : 'tool';
          const result = content.replace(/<\/?tool_response[^>]*>/g, '').trim();
          const preview = result.length > 300 ? result.slice(0, 300) + '...' : result;
          return `[tool:${toolName}] ${preview}`;
        }
        const text = (m.content || '').slice(0, 500);
        return `${m.role}: ${text}`;
      })
      .join('\n\n');

    // Load existing experience entries to avoid duplicates
    const existingExperiences = listExperiences();
    const existingSummary = existingExperiences.slice(0, 20)
      .map(e => `- ${e.name}: ${e.description}`)
      .join('\n');

    const reflectionPrompt = `You are the reflection engine for PRE (Personal Reasoning Engine). After a task completes, you analyze what happened and extract reusable lessons.

EXISTING LESSONS (do not duplicate):
${existingSummary || '(none yet)'}

Analyze the conversation and extract lessons learned. Focus on:
1. **Tool strategies** — which tools were effective, which approaches failed, what sequence worked
2. **Error patterns** — errors encountered and how they were resolved (or not)
3. **Domain knowledge** — non-obvious facts discovered during the task
4. **Efficiency insights** — faster approaches discovered, unnecessary steps identified

OUTPUT FORMAT:
Return a JSON array. If nothing is worth capturing, return [].
Each entry:
{
  "name": "short_descriptive_name",
  "task": "what was being done",
  "approach": "what was tried",
  "outcome": "what happened (success/failure/partial)",
  "lesson": "the reusable insight for future tasks"
}

Be selective — only capture insights that will help with DIFFERENT future tasks. Skip trivial observations.`;

    const result = await streamChat({
      messages: [
        { role: 'system', content: reflectionPrompt },
        { role: 'user', content: conversationSummary },
      ],
      maxTokens: 2048,
    });

    const response = result.response || '';
    const jsonMatch = response.match(/\[[\s\S]*?\]/);
    if (!jsonMatch) return [];

    let entries;
    try {
      entries = JSON.parse(jsonMatch[0]);
    } catch { return []; }
    if (!Array.isArray(entries)) return [];

    // Save each valid entry
    const saved = [];
    for (const entry of entries) {
      if (!entry.name || !entry.lesson) continue;

      const content = [
        entry.task ? `**Task:** ${entry.task}` : '',
        entry.approach ? `**Approach:** ${entry.approach}` : '',
        entry.outcome ? `**Outcome:** ${entry.outcome}` : '',
        `**Lesson:** ${entry.lesson}`,
      ].filter(Boolean).join('\n');

      const description = entry.lesson.slice(0, 120);

      // Check for embedding-based duplicates if possible
      try {
        const embeddings = await embed(description);
        if (embeddings.length > 0) {
          const vector = embeddings[0];
          if (await isDuplicateWithEmbedding(vector)) {
            console.log(`[experience] Skipping duplicate: ${entry.name}`);
            continue;
          }
          // Save embedding
          const cache = loadEmbeddings();
          cache[entry.name] = { vector, text: description };
          saveEmbeddings();
        }
      } catch (err) {
        console.log(`[experience] Embedding check skipped: ${err.message}`);
        // Continue without embedding dedup — still save the entry
      }

      const result = saveExperience({
        name: entry.name,
        description,
        content,
      });

      if (result.success) {
        saved.push(entry);
        console.log(`[experience] Saved: ${entry.name}`);
      }
    }

    return saved;
  } catch (err) {
    console.log(`[experience] Reflection error: ${err.message}`);
    return [];
  } finally {
    reflectionState.running = false;
  }
}

/**
 * Save an experience entry to the experience ledger
 */
function saveExperience({ name, description, content }) {
  if (!name || !content) return { error: 'name and content required' };

  const filename = name.replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase() + '.md';
  const filePath = path.join(EXPERIENCE_DIR, filename);

  const meta = {
    name,
    description: description || name,
    type: 'experience',
    created: new Date().toISOString().slice(0, 10),
    verified: new Date().toISOString().slice(0, 10),
  };

  const md = '---\n' + Object.entries(meta).map(([k, v]) => `${k}: ${v}`).join('\n') + '\n---\n\n' + content + '\n';
  fs.writeFileSync(filePath, md, 'utf-8');

  return { success: true, filename, path: filePath };
}

/**
 * List all experience entries
 */
function listExperiences() {
  if (!fs.existsSync(EXPERIENCE_DIR)) return [];
  return fs.readdirSync(EXPERIENCE_DIR)
    .filter(f => f.endsWith('.md'))
    .map(filename => {
      try {
        const filePath = path.join(EXPERIENCE_DIR, filename);
        const stat = fs.statSync(filePath);
        const content = fs.readFileSync(filePath, 'utf-8');
        const { meta, body } = parseFrontmatter(content);
        return {
          filename,
          name: meta.name || filename.replace('.md', ''),
          description: meta.description || '',
          type: 'experience',
          body,
          created: meta.created,
          verified: meta.verified,
          mtimeMs: stat.mtimeMs,
        };
      } catch { return null; }
    })
    .filter(Boolean)
    .sort((a, b) => b.mtimeMs - a.mtimeMs);
}

/**
 * Search experiences by semantic similarity
 * Falls back to keyword search if embeddings aren't available
 */
async function searchExperiences(query) {
  const experiences = listExperiences();
  if (experiences.length === 0) return [];

  // Try embedding-based search first
  try {
    const queryEmbeddings = await embed(query);
    if (queryEmbeddings.length > 0) {
      const queryVector = queryEmbeddings[0];
      const cache = loadEmbeddings();

      const scored = experiences
        .map(exp => {
          const cached = cache[exp.name];
          const sim = cached?.vector ? cosineSim(queryVector, cached.vector) : 0;
          return { ...exp, similarity: sim };
        })
        .filter(e => e.similarity > 0.3)
        .sort((a, b) => b.similarity - a.similarity);

      if (scored.length > 0) return scored.slice(0, 10);
    }
  } catch {}

  // Fallback: keyword search
  const q = query.toLowerCase();
  return experiences.filter(e => {
    const text = `${e.name} ${e.description} ${e.body}`.toLowerCase();
    return text.includes(q);
  });
}

/**
 * Build experience context for system prompt injection.
 * Returns the most relevant experiences for the current task.
 */
function buildExperienceContext() {
  const experiences = listExperiences();
  if (experiences.length === 0) return '';

  // Include the 10 most recent experiences
  const recent = experiences.slice(0, 10);

  let ctx = '<experience_ledger>\n';
  ctx += 'Lessons from past tasks (use these to inform your approach):\n\n';
  for (const exp of recent) {
    const age = Math.floor((Date.now() - exp.mtimeMs) / (1000 * 60 * 60 * 24));
    const ageStr = age <= 1 ? 'today' : age <= 7 ? `${age}d ago` : `${Math.floor(age / 7)}w ago`;
    ctx += `- **${exp.name}** (${ageStr}): ${exp.description}\n`;
  }
  ctx += '</experience_ledger>\n';
  return ctx;
}

module.exports = {
  reflect,
  saveExperience,
  listExperiences,
  searchExperiences,
  buildExperienceContext,
};
