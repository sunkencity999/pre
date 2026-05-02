// PRE Web GUI — Full-Text Search across Sessions
// Searches all JSONL session files for matching text or patterns.

const fs = require('fs');
const path = require('path');
const { SESSIONS_DIR } = require('./constants');

/**
 * Search across all session JSONL files for matching content.
 *
 * @param {string} query - Search text or regex pattern
 * @param {Object} [opts]
 * @param {number} [opts.maxResults=20] - Maximum results to return
 * @param {string} [opts.project] - Limit to sessions in this project
 * @returns {string} Formatted search results
 */
function searchSessions(query, opts = {}) {
  if (!query) return 'Error: query is required for session_search';

  const maxResults = opts.maxResults || 20;
  const projectFilter = opts.project || null;

  let regex;
  try {
    regex = new RegExp(query, 'i');
  } catch {
    // Not a valid regex — treat as literal
    regex = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
  }

  if (!fs.existsSync(SESSIONS_DIR)) return 'No sessions found.';

  const results = [];
  const sessionFiles = [];

  // Collect session files, optionally filtered by project
  const entries = fs.readdirSync(SESSIONS_DIR);
  for (const entry of entries) {
    const full = path.join(SESSIONS_DIR, entry);
    const stat = fs.statSync(full);
    if (stat.isFile() && entry.endsWith('.jsonl')) {
      // Session filenames: <project>:<channel>.jsonl or just <sessionId>.jsonl
      if (projectFilter) {
        if (!entry.startsWith(projectFilter + ':') && !entry.startsWith(projectFilter + '_')) {
          continue;
        }
      }
      sessionFiles.push({ path: full, name: entry, mtime: stat.mtimeMs });
    } else if (stat.isDirectory()) {
      // Sessions can be in project subdirectories
      const subEntries = fs.readdirSync(full).filter(f => f.endsWith('.jsonl'));
      for (const sub of subEntries) {
        const subFull = path.join(full, sub);
        if (projectFilter && entry !== projectFilter) continue;
        sessionFiles.push({ path: subFull, name: `${entry}/${sub}`, mtime: fs.statSync(subFull).mtimeMs });
      }
    }
  }

  // Sort by most recent first
  sessionFiles.sort((a, b) => b.mtime - a.mtime);

  for (const sf of sessionFiles) {
    if (results.length >= maxResults) break;

    try {
      const content = fs.readFileSync(sf.path, 'utf-8');
      const lines = content.split('\n').filter(Boolean);

      for (const line of lines) {
        if (results.length >= maxResults) break;

        try {
          const msg = JSON.parse(line);
          const text = msg.content || '';
          if (!regex.test(text)) continue;

          // Extract a snippet around the match
          const match = text.match(regex);
          const matchIdx = match ? text.indexOf(match[0]) : 0;
          const start = Math.max(0, matchIdx - 80);
          const end = Math.min(text.length, matchIdx + match[0].length + 80);
          let snippet = text.slice(start, end).replace(/\n/g, ' ');
          if (start > 0) snippet = '...' + snippet;
          if (end < text.length) snippet += '...';

          // Parse session name for display
          const sessionName = sf.name.replace('.jsonl', '');
          const date = new Date(sf.mtime).toISOString().slice(0, 10);

          results.push({
            session: sessionName,
            date,
            role: msg.role || 'unknown',
            snippet,
          });
        } catch {
          // Skip malformed JSON lines
        }
      }
    } catch {
      // Skip unreadable files
    }
  }

  if (results.length === 0) return `No matches found for "${query}" across sessions.`;

  const formatted = results.map((r, i) =>
    `${i + 1}. [${r.date}] ${r.session} (${r.role})\n   ${r.snippet}`
  ).join('\n\n');

  return `Found ${results.length} match(es) for "${query}":\n\n${formatted}`;
}

module.exports = { searchSessions };
