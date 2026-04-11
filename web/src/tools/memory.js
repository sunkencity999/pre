// PRE Web GUI — Memory tools (save, search, list, delete)
// Thin wrappers around the enhanced memory module

const { saveMemory, deleteMemory, searchMemories, getAllMemories } = require('../memory');

function save(args) {
  const { name, type, description, content, scope } = args;
  const result = saveMemory({ name, type, description, content, scope });
  if (result.error) return `Error: ${result.error}`;
  return `Memory saved: ${name} (${result.filename})`;
}

function search(args) {
  const query = args.query || '';
  const results = searchMemories(query);

  if (results.length === 0) return query ? `No memories matching '${query}'` : 'No memories found';
  return results.map(m =>
    `[${m.type}] ${m.name}: ${m.description}\n  ${m.body.slice(0, 200)}`
  ).join('\n\n');
}

function list() {
  const memories = getAllMemories();
  if (memories.length === 0) return 'No memories saved';

  return memories.map(m =>
    `[${m.type}] ${m.name}: ${m.description}\n  ${m.body.slice(0, 200)}`
  ).join('\n\n');
}

function del(args) {
  const query = args.query || '';
  const result = deleteMemory(query);
  if (result.error) return result.error;
  return `Deleted memory: ${result.deleted} (${result.filename})`;
}

module.exports = { save, search, list, del };
