// PRE Web GUI — Memory tools (save, search, list, delete)
// Operates on ~/.pre/memory/ directory

const fs = require('fs');
const path = require('path');
const { MEMORY_DIR } = require('../constants');

// Ensure memory directory exists
if (!fs.existsSync(MEMORY_DIR)) {
  fs.mkdirSync(MEMORY_DIR, { recursive: true });
}

function save(args) {
  const { name, type, description, content, scope } = args;
  if (!name) return 'Error: name is required';
  if (!type) return 'Error: type is required';
  if (!content) return 'Error: content is required';

  // Sanitize filename
  const filename = name.replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase() + '.md';
  const filePath = path.join(MEMORY_DIR, filename);

  const md = `---
name: ${name}
description: ${description || name}
type: ${type}
scope: ${scope || 'project'}
---

${content}
`;

  fs.writeFileSync(filePath, md, 'utf-8');
  return `Memory saved: ${name} (${filePath})`;
}

function search(args) {
  const query = (args.query || '').toLowerCase();
  const files = listMemoryFiles();
  const results = [];

  for (const { name, description, type, content, filename } of files) {
    const text = `${name} ${description} ${content}`.toLowerCase();
    if (!query || text.includes(query)) {
      results.push(`[${type}] ${name}: ${description}\n  ${content.slice(0, 200)}`);
    }
  }

  if (results.length === 0) return query ? `No memories matching '${query}'` : 'No memories found';
  return results.join('\n\n');
}

function list() {
  const files = listMemoryFiles();
  if (files.length === 0) return 'No memories saved';

  return files.map(m => `[${m.type}] ${m.name}: ${m.description}`).join('\n');
}

function del(args) {
  const query = (args.query || '').toLowerCase();
  if (!query) return 'Error: query is required';

  const files = listMemoryFiles();
  for (const m of files) {
    if (m.name.toLowerCase().includes(query) || m.filename.toLowerCase().includes(query)) {
      fs.unlinkSync(path.join(MEMORY_DIR, m.filename));
      return `Deleted memory: ${m.name} (${m.filename})`;
    }
  }
  return `No memory found matching '${query}'`;
}

function listMemoryFiles() {
  if (!fs.existsSync(MEMORY_DIR)) return [];
  return fs.readdirSync(MEMORY_DIR)
    .filter(f => f.endsWith('.md'))
    .map(filename => {
      try {
        const content = fs.readFileSync(path.join(MEMORY_DIR, filename), 'utf-8');
        const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
        if (!match) return null;

        const frontmatter = match[1];
        const body = match[2].trim();
        const nameMatch = frontmatter.match(/name:\s*(.+)/);
        const descMatch = frontmatter.match(/description:\s*(.+)/);
        const typeMatch = frontmatter.match(/type:\s*(.+)/);

        return {
          filename,
          name: nameMatch ? nameMatch[1].trim() : filename,
          description: descMatch ? descMatch[1].trim() : '',
          type: typeMatch ? typeMatch[1].trim() : 'unknown',
          content: body,
        };
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

module.exports = { save, search, list, del };
