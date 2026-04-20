// PRE Web GUI — Local RAG (Retrieval-Augmented Generation)
// Indexes local files into vector databases for semantic search.
// Uses nomic-embed-text (768-dim) via Ollama for embeddings.
//
// Storage layout per index:
//   ~/.pre/rag/{index_name}/
//     meta.json        — index metadata (name, paths, stats, file mtimes)
//     chunks.json      — array of { id, file, idx, content }
//     vectors.json     — { chunkId: float[768] }

const fs = require('fs');
const path = require('path');
const { embed } = require('../ollama');

const RAG_DIR = path.join(require('os').homedir(), '.pre', 'rag');

// Ensure base directory exists
if (!fs.existsSync(RAG_DIR)) fs.mkdirSync(RAG_DIR, { recursive: true });

// ── File type support ──────────────────────────────────────────────────────

const TEXT_EXTENSIONS = new Set([
  '.md', '.txt', '.rst', '.adoc',
  '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
  '.py', '.pyi', '.rb', '.php', '.java', '.kt', '.scala',
  '.c', '.h', '.cpp', '.hpp', '.cc', '.m', '.mm', '.swift',
  '.go', '.rs', '.zig',
  '.cs', '.fs', '.vb',
  '.sh', '.bash', '.zsh', '.fish', '.ps1',
  '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
  '.xml', '.html', '.htm', '.css', '.scss', '.less', '.sass',
  '.sql', '.graphql', '.gql',
  '.csv', '.tsv',
  '.env', '.gitignore', '.dockerignore',
  '.r', '.R', '.jl', '.lua', '.pl', '.pm',
  '.tf', '.hcl', '.nix', '.dhall',
  '.proto', '.thrift', '.avsc',
  '.tex', '.bib', '.sty',
  '.el', '.clj', '.cljs', '.edn', '.ex', '.exs', '.erl', '.hrl',
  '.hs', '.lhs', '.ml', '.mli', '.v', '.sv',
  '.makefile', '.mk', '.cmake',
]);

// Files to always skip
const SKIP_NAMES = new Set([
  'node_modules', '.git', '.svn', '.hg', '__pycache__', '.DS_Store',
  'venv', '.venv', 'env', '.env', 'dist', 'build', '.next', '.nuxt',
  'target', 'bin', 'obj', '.tox', '.mypy_cache', '.pytest_cache',
  'coverage', '.nyc_output', '.cache', 'vendor',
]);

const MAX_FILE_SIZE = 512 * 1024; // 512KB per file
const CHUNK_SIZE = 1500;          // ~375 tokens per chunk
const CHUNK_OVERLAP = 200;        // overlap between chunks for continuity
const EMBED_BATCH_SIZE = 40;      // embeddings per Ollama API call

// ── Helpers ────────────────────────────────────────────────────────────────

function indexDir(name) {
  return path.join(RAG_DIR, name);
}

function loadMeta(name) {
  const p = path.join(indexDir(name), 'meta.json');
  if (!fs.existsSync(p)) return null;
  try { return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch { return null; }
}

function saveMeta(name, meta) {
  const dir = indexDir(name);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify(meta, null, 2));
}

function loadChunks(name) {
  const p = path.join(indexDir(name), 'chunks.json');
  if (!fs.existsSync(p)) return [];
  try { return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch { return []; }
}

function saveChunks(name, chunks) {
  const dir = indexDir(name);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'chunks.json'), JSON.stringify(chunks));
}

function loadVectors(name) {
  const p = path.join(indexDir(name), 'vectors.json');
  if (!fs.existsSync(p)) return {};
  try { return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch { return {}; }
}

function saveVectors(name, vectors) {
  const dir = indexDir(name);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'vectors.json'), JSON.stringify(vectors));
}

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

function isTextFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const base = path.basename(filePath).toLowerCase();
  // Handle extensionless files that are commonly text
  if (!ext && ['makefile', 'dockerfile', 'vagrantfile', 'gemfile', 'rakefile', 'procfile'].includes(base)) return true;
  return TEXT_EXTENSIONS.has(ext);
}

// ── Chunking ───────────────────────────────────────────────────────────────

/**
 * Split text into overlapping chunks, respecting paragraph boundaries.
 */
function chunkText(text) {
  const chunks = [];
  // Normalize line endings
  text = text.replace(/\r\n/g, '\n');

  // For very short files, return as a single chunk
  if (text.length <= CHUNK_SIZE) {
    if (text.trim().length > 0) {
      chunks.push(text.trim());
    }
    return chunks;
  }

  // Split on double newlines first (paragraph boundaries)
  const paragraphs = text.split(/\n\n+/);
  let current = '';

  for (const para of paragraphs) {
    const trimmed = para.trim();
    if (!trimmed) continue;

    if (current.length + trimmed.length + 2 <= CHUNK_SIZE) {
      current += (current ? '\n\n' : '') + trimmed;
    } else {
      // Current chunk is full
      if (current.trim()) {
        chunks.push(current.trim());
      }

      // If a single paragraph exceeds CHUNK_SIZE, split it by lines
      if (trimmed.length > CHUNK_SIZE) {
        const lines = trimmed.split('\n');
        current = '';
        for (const line of lines) {
          if (current.length + line.length + 1 <= CHUNK_SIZE) {
            current += (current ? '\n' : '') + line;
          } else {
            if (current.trim()) chunks.push(current.trim());
            // If a single line exceeds CHUNK_SIZE, hard-split it
            if (line.length > CHUNK_SIZE) {
              for (let i = 0; i < line.length; i += CHUNK_SIZE - CHUNK_OVERLAP) {
                chunks.push(line.slice(i, i + CHUNK_SIZE));
              }
              current = '';
            } else {
              current = line;
            }
          }
        }
      } else {
        current = trimmed;
      }
    }
  }

  if (current.trim()) {
    chunks.push(current.trim());
  }

  // Add overlap: prepend the tail of the previous chunk to each subsequent chunk
  if (CHUNK_OVERLAP > 0 && chunks.length > 1) {
    const overlapped = [chunks[0]];
    for (let i = 1; i < chunks.length; i++) {
      const prev = chunks[i - 1];
      const overlapText = prev.slice(-CHUNK_OVERLAP);
      overlapped.push(overlapText + '\n...\n' + chunks[i]);
    }
    return overlapped;
  }

  return chunks;
}

// ── File collection ────────────────────────────────────────────────────────

/**
 * Recursively collect indexable files from a directory.
 */
function collectFiles(dirPath, recursive = true) {
  const files = [];

  function walk(dir) {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }

    for (const ent of entries) {
      if (SKIP_NAMES.has(ent.name)) continue;
      if (ent.name.startsWith('.') && ent.name !== '.env') continue;

      const full = path.join(dir, ent.name);

      if (ent.isDirectory()) {
        if (recursive) walk(full);
      } else if (ent.isFile()) {
        if (!isTextFile(full)) continue;
        try {
          const stat = fs.statSync(full);
          if (stat.size > MAX_FILE_SIZE) continue;
          if (stat.size === 0) continue;
          files.push({ path: full, size: stat.size, mtime: stat.mtimeMs });
        } catch { /* skip unreadable */ }
      }
    }
  }

  walk(dirPath);
  return files;
}

// ── Core operations ────────────────────────────────────────────────────────

/**
 * Index files from a path into a named index.
 * Supports incremental updates — only re-processes changed files.
 */
async function indexPath(args, cwd) {
  const targetPath = args.path ? path.resolve(cwd || '', args.path) : cwd;
  const indexName = (args.index_name || 'default').replace(/[^a-zA-Z0-9_-]/g, '_');
  const recursive = args.recursive !== false;

  if (!targetPath || !fs.existsSync(targetPath)) {
    return `Error: path not found: ${targetPath}`;
  }

  const stat = fs.statSync(targetPath);
  const isDir = stat.isDirectory();
  const files = isDir ? collectFiles(targetPath, recursive) : [{ path: targetPath, size: stat.size, mtime: stat.mtimeMs }];

  if (files.length === 0) {
    return `No indexable text files found in ${targetPath}`;
  }

  // Load existing index for incremental update
  const existingMeta = loadMeta(indexName);
  const existingChunks = loadChunks(indexName);
  const existingVectors = loadVectors(indexName);
  const existingMtimes = existingMeta?.fileMtimes || {};

  // Determine which files need (re)processing
  const changed = [];
  const unchanged = [];
  for (const f of files) {
    if (existingMtimes[f.path] && existingMtimes[f.path] >= f.mtime) {
      unchanged.push(f.path);
    } else {
      changed.push(f);
    }
  }

  if (changed.length === 0) {
    return `Index "${indexName}" is up to date. ${unchanged.length} files, ${existingChunks.length} chunks. No changes detected.`;
  }

  // Remove old chunks for changed files
  const changedPaths = new Set(changed.map(f => f.path));
  const retainedChunks = existingChunks.filter(c => !changedPaths.has(c.file));
  const retainedVectors = {};
  for (const c of retainedChunks) {
    if (existingVectors[c.id]) retainedVectors[c.id] = existingVectors[c.id];
  }

  // Chunk changed files
  const newChunks = [];
  let chunkId = retainedChunks.length > 0
    ? Math.max(...retainedChunks.map(c => parseInt(c.id.split('_').pop()) || 0)) + 1
    : 0;

  for (const f of changed) {
    let content;
    try { content = fs.readFileSync(f.path, 'utf-8'); } catch { continue; }

    const relativePath = isDir ? path.relative(targetPath, f.path) : path.basename(f.path);
    const textChunks = chunkText(content);

    for (let i = 0; i < textChunks.length; i++) {
      const id = `c_${chunkId++}`;
      newChunks.push({
        id,
        file: f.path,
        relPath: relativePath,
        idx: i,
        content: textChunks[i],
      });
    }
  }

  // Embed new chunks in batches
  const newVectors = {};
  const totalBatches = Math.ceil(newChunks.length / EMBED_BATCH_SIZE);

  for (let b = 0; b < totalBatches; b++) {
    const batch = newChunks.slice(b * EMBED_BATCH_SIZE, (b + 1) * EMBED_BATCH_SIZE);
    const texts = batch.map(c => {
      // Prefix with file path for better context
      const prefix = `File: ${c.relPath}\n`;
      return prefix + c.content;
    });

    try {
      const embeddings = await embed(texts);
      for (let i = 0; i < batch.length; i++) {
        if (embeddings[i]) newVectors[batch[i].id] = embeddings[i];
      }
    } catch (err) {
      return `Error embedding batch ${b + 1}/${totalBatches}: ${err.message}`;
    }
  }

  // Merge retained + new
  const allChunks = [...retainedChunks, ...newChunks];
  const allVectors = { ...retainedVectors, ...newVectors };

  // Build file mtime map
  const fileMtimes = {};
  for (const f of files) fileMtimes[f.path] = f.mtime;
  // Keep mtimes for files that were in the index but not in this scan (other paths)
  if (existingMeta?.fileMtimes) {
    for (const [p, mt] of Object.entries(existingMeta.fileMtimes)) {
      if (!(p in fileMtimes) && !changedPaths.has(p)) fileMtimes[p] = mt;
    }
  }

  // Save
  const meta = {
    name: indexName,
    description: args.description || existingMeta?.description || `Index of ${targetPath}`,
    created: existingMeta?.created || new Date().toISOString(),
    updated: new Date().toISOString(),
    paths: [...new Set([...(existingMeta?.paths || []), targetPath])],
    fileCount: Object.keys(fileMtimes).length,
    chunkCount: allChunks.length,
    embeddingModel: 'nomic-embed-text',
    vectorDim: 768,
    fileMtimes,
  };

  saveMeta(indexName, meta);
  saveChunks(indexName, allChunks);
  saveVectors(indexName, allVectors);

  const lines = [
    `Indexed "${indexName}": ${changed.length} files processed, ${unchanged.length} unchanged`,
    `Total: ${meta.fileCount} files, ${allChunks.length} chunks, ${Object.keys(allVectors).length} vectors`,
    `Path: ${targetPath}`,
  ];
  return lines.join('\n');
}

/**
 * Semantic search across one or all indexes.
 */
async function search(args) {
  const query = args.query;
  if (!query) return 'Error: query is required';

  const indexName = args.index_name;
  const topK = Math.min(args.top_k || 5, 20);
  const minScore = args.min_score || 0.3;

  // Get query embedding
  let queryVector;
  try {
    const embeddings = await embed(query);
    queryVector = embeddings[0];
    if (!queryVector) return 'Error: failed to compute query embedding';
  } catch (err) {
    return `Error computing query embedding: ${err.message}`;
  }

  // Collect indexes to search
  let indexNames;
  if (indexName) {
    if (!fs.existsSync(indexDir(indexName))) return `Error: index "${indexName}" not found`;
    indexNames = [indexName];
  } else {
    // Search all indexes
    try {
      indexNames = fs.readdirSync(RAG_DIR, { withFileTypes: true })
        .filter(d => d.isDirectory() && fs.existsSync(path.join(RAG_DIR, d.name, 'meta.json')))
        .map(d => d.name);
    } catch { indexNames = []; }
    if (indexNames.length === 0) return 'No RAG indexes found. Use `rag` with action `index` to index a directory first.';
  }

  // Search each index
  const allResults = [];

  for (const idx of indexNames) {
    const chunks = loadChunks(idx);
    const vectors = loadVectors(idx);

    for (const chunk of chunks) {
      const vec = vectors[chunk.id];
      if (!vec) continue;
      const score = cosineSim(queryVector, vec);
      if (score >= minScore) {
        allResults.push({
          score,
          index: idx,
          file: chunk.relPath || path.basename(chunk.file),
          fullPath: chunk.file,
          chunkIdx: chunk.idx,
          content: chunk.content,
        });
      }
    }
  }

  // Sort by score and take top K
  allResults.sort((a, b) => b.score - a.score);
  const results = allResults.slice(0, topK);

  if (results.length === 0) {
    return `No results above similarity threshold (${minScore}) for: "${query}"`;
  }

  // Format results
  const lines = [`**RAG Search Results** (${results.length} matches for "${query}")`, ''];
  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    const snippet = r.content.length > 500 ? r.content.slice(0, 500) + '...' : r.content;
    lines.push(`**${i + 1}. ${r.file}** (score: ${r.score.toFixed(3)}, index: ${r.index})`);
    lines.push(`   Path: ${r.fullPath}`);
    lines.push('```');
    lines.push(snippet);
    lines.push('```');
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * List all RAG indexes.
 */
function listIndexes() {
  if (!fs.existsSync(RAG_DIR)) return 'No RAG indexes. Use action `index` to create one.';

  const dirs = fs.readdirSync(RAG_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory());

  if (dirs.length === 0) return 'No RAG indexes. Use action `index` to create one.';

  const lines = ['**RAG Indexes**', ''];
  for (const d of dirs) {
    const meta = loadMeta(d.name);
    if (!meta) continue;
    const age = timeSince(meta.updated || meta.created);
    lines.push(`**${meta.name}** — ${meta.fileCount} files, ${meta.chunkCount} chunks (updated ${age})`);
    lines.push(`  Paths: ${meta.paths?.join(', ') || '(unknown)'}`);
    lines.push(`  Description: ${meta.description || '(none)'}`);
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Detailed status of a specific index.
 */
function statusIndex(args) {
  const indexName = args.index_name || 'default';
  const meta = loadMeta(indexName);
  if (!meta) return `Index "${indexName}" not found.`;

  const chunks = loadChunks(indexName);
  const vectors = loadVectors(indexName);

  // Calculate storage sizes
  const dir = indexDir(indexName);
  let totalSize = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      totalSize += fs.statSync(path.join(dir, f)).size;
    }
  } catch {}

  // File type breakdown
  const extCounts = {};
  for (const c of chunks) {
    const ext = path.extname(c.file).toLowerCase() || '(none)';
    extCounts[ext] = (extCounts[ext] || 0) + 1;
  }

  const lines = [
    `**Index: ${meta.name}**`,
    `Description: ${meta.description}`,
    `Created: ${meta.created}`,
    `Updated: ${meta.updated}`,
    `Files: ${meta.fileCount}`,
    `Chunks: ${meta.chunkCount}`,
    `Vectors: ${Object.keys(vectors).length}`,
    `Storage: ${(totalSize / 1024).toFixed(1)} KB`,
    `Embedding model: ${meta.embeddingModel} (${meta.vectorDim}d)`,
    '',
    '**File types:**',
    ...Object.entries(extCounts).sort((a, b) => b[1] - a[1]).map(([ext, n]) => `  ${ext}: ${n} chunks`),
    '',
    `**Indexed paths:**`,
    ...(meta.paths || []).map(p => `  ${p}`),
  ];

  return lines.join('\n');
}

/**
 * Delete an index.
 */
function deleteIndex(args) {
  const indexName = args.index_name;
  if (!indexName) return 'Error: index_name is required';

  const dir = indexDir(indexName);
  if (!fs.existsSync(dir)) return `Index "${indexName}" not found.`;

  // Remove directory recursively
  fs.rmSync(dir, { recursive: true, force: true });
  return `Deleted index "${indexName}".`;
}

function timeSince(isoStr) {
  if (!isoStr) return 'unknown';
  const ms = Date.now() - new Date(isoStr).getTime();
  const mins = Math.floor(ms / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

// ── Main dispatcher ────────────────────────────────────────────────────────

async function rag(args, cwd) {
  const action = (args.action || '').toLowerCase();

  switch (action) {
    case 'index':
      return indexPath(args, cwd);
    case 'search':
      return search(args);
    case 'list':
      return listIndexes();
    case 'status':
      return statusIndex(args);
    case 'delete':
      return deleteIndex(args);
    default:
      return 'Unknown RAG action. Available: index, search, list, status, delete';
  }
}

module.exports = { rag, search, indexPath, listIndexes, statusIndex, deleteIndex };
