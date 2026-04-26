// Tests for src/memory.js — persistent memory system
const fs = require('fs');
const path = require('path');

jest.mock('../src/constants');
jest.mock('../src/ollama', () => ({
  streamChat: jest.fn().mockResolvedValue({ response: '[]' }),
}));

const { MEMORY_DIR } = require('../src/constants');
const memory = require('../src/memory');

describe('memory', () => {
  afterEach(() => {
    for (const f of fs.readdirSync(MEMORY_DIR).filter(f => f.endsWith('.md'))) {
      fs.unlinkSync(path.join(MEMORY_DIR, f));
    }
  });

  // ── Frontmatter ──

  describe('parseFrontmatter', () => {
    test('parses valid frontmatter', () => {
      const { meta, body } = memory.parseFrontmatter(
        '---\nname: Test\ntype: user\n---\n\nBody text.'
      );
      expect(meta.name).toBe('Test');
      expect(meta.type).toBe('user');
      expect(body).toBe('Body text.');
    });

    test('returns raw content when no frontmatter', () => {
      const { meta, body } = memory.parseFrontmatter('Plain text.');
      expect(Object.keys(meta)).toHaveLength(0);
      expect(body).toBe('Plain text.');
    });
  });

  describe('buildFrontmatter', () => {
    test('builds YAML frontmatter from object', () => {
      expect(memory.buildFrontmatter({ name: 'X', type: 'user' }))
        .toBe('---\nname: X\ntype: user\n---');
    });
  });

  // ── saveMemory ──

  describe('saveMemory', () => {
    test('saves a memory file with frontmatter', () => {
      const result = memory.saveMemory({
        name: 'Test Memory', type: 'user', description: 'A test', content: 'Likes dark mode.',
      });
      expect(result.success).toBe(true);
      expect(fs.existsSync(result.path)).toBe(true);
      const raw = fs.readFileSync(result.path, 'utf-8');
      expect(raw).toContain('name: Test Memory');
      expect(raw).toContain('Likes dark mode.');
    });

    test('rejects missing name', () => {
      expect(memory.saveMemory({ type: 'user', content: 'x' }).error).toBeTruthy();
    });

    test('rejects missing content', () => {
      expect(memory.saveMemory({ name: 'X', type: 'user' }).error).toBeTruthy();
    });
  });

  // ── getAllMemories ──

  describe('getAllMemories', () => {
    test('returns saved memories with parsed metadata', () => {
      memory.saveMemory({ name: 'A', type: 'user', description: 'a', content: 'a' });
      memory.saveMemory({ name: 'B', type: 'feedback', description: 'b', content: 'b' });
      const all = memory.getAllMemories();
      expect(all.length).toBeGreaterThanOrEqual(2);
      expect(all.some(m => m.name === 'A')).toBe(true);
    });
  });

  // ── searchMemories ──

  describe('searchMemories', () => {
    test('finds memories matching by name or content', () => {
      memory.saveMemory({ name: 'Dark Mode Pref', type: 'feedback', description: 'ui', content: 'Prefers dark.' });
      memory.saveMemory({ name: 'Python', type: 'user', description: 'skill', content: 'Knows Python.' });
      const results = memory.searchMemories('dark mode');
      expect(results.some(m => m.name === 'Dark Mode Pref')).toBe(true);
    });
  });

  // ── deleteMemory ──

  describe('deleteMemory', () => {
    test('deletes by filename', () => {
      const saved = memory.saveMemory({ name: 'Del', type: 'project', description: 't', content: 'x' });
      expect(memory.deleteMemory(saved.filename).success).toBe(true);
      expect(fs.existsSync(saved.path)).toBe(false);
    });

    test('returns error for nonexistent', () => {
      expect(memory.deleteMemory('ghost.md').error).toBeTruthy();
    });
  });

  // ── buildMemoryContext ──

  describe('buildMemoryContext', () => {
    test('returns a string', () => {
      expect(typeof memory.buildMemoryContext()).toBe('string');
    });

    test('includes saved memories in context', () => {
      memory.saveMemory({ name: 'Ctx Test', type: 'user', description: 'c', content: 'User is a pilot.' });
      expect(memory.buildMemoryContext()).toContain('Ctx Test');
    });
  });

  // ── scanMemoryDir ──

  describe('scanMemoryDir', () => {
    test('returns parsed entries', () => {
      memory.saveMemory({ name: 'Scan', type: 'reference', description: 'r', content: 'body' });
      const entries = memory.scanMemoryDir(MEMORY_DIR);
      expect(entries.length).toBeGreaterThanOrEqual(1);
      expect(entries[0]).toHaveProperty('name');
      expect(entries[0]).toHaveProperty('type');
    });

    test('returns empty for nonexistent dir', () => {
      expect(memory.scanMemoryDir('/tmp/no-such-dir-xyz')).toEqual([]);
    });
  });

  // ── memoryAge ──

  describe('memoryAge', () => {
    test('returns age string for recent timestamp', () => {
      expect(typeof memory.memoryAge(Date.now() - 1000)).toBe('string');
    });
  });
});
