// Tests for src/chronos.js — temporal awareness and memory staleness
const fs = require('fs');
const path = require('path');

jest.mock('../src/constants');
jest.mock('../src/ollama', () => ({
  streamChat: jest.fn().mockResolvedValue({ response: '[]' }),
}));

const { MEMORY_DIR } = require('../src/constants');
const chronos = require('../src/chronos');
const memory = require('../src/memory');

describe('chronos', () => {
  afterEach(() => {
    for (const f of fs.readdirSync(MEMORY_DIR).filter(f => f.endsWith('.md'))) {
      fs.unlinkSync(path.join(MEMORY_DIR, f));
    }
  });

  describe('STALENESS_THRESHOLDS', () => {
    test('exports thresholds for all memory types', () => {
      const t = chronos.STALENESS_THRESHOLDS;
      expect(t).toHaveProperty('project');
      expect(t).toHaveProperty('reference');
      expect(t).toHaveProperty('feedback');
      expect(t).toHaveProperty('user');
      expect(t).toHaveProperty('experience');
    });

    test('project has shortest threshold', () => {
      const t = chronos.STALENESS_THRESHOLDS;
      expect(t.project).toBeLessThan(t.user);
      expect(t.project).toBeLessThan(t.feedback);
    });
  });

  describe('stalenessReport', () => {
    test('returns empty buckets when no memories', () => {
      const r = chronos.stalenessReport();
      expect(r.fresh).toEqual([]);
      expect(r.aging).toEqual([]);
      expect(r.stale).toEqual([]);
      expect(r.unverified).toEqual([]);
    });
  });

  describe('verifyMemory', () => {
    test('marks a memory as verified', () => {
      const saved = memory.saveMemory({ name: 'V', type: 'project', description: 't', content: 'b' });
      const result = chronos.verifyMemory(saved.filename);
      expect(result.success).toBe(true);
      expect(result.verified).toBe(new Date().toISOString().slice(0, 10));
      expect(fs.readFileSync(saved.path, 'utf-8')).toContain('verified:');
    });

    test('returns error for nonexistent', () => {
      expect(chronos.verifyMemory('ghost.md').error).toBeTruthy();
    });
  });

  describe('buildTemporalContext', () => {
    test('returns empty string when no stale memories', () => {
      expect(chronos.buildTemporalContext()).toBe('');
    });
  });

  describe('maintenanceSummary', () => {
    test('returns summary with health percentage', () => {
      const s = chronos.maintenanceSummary();
      expect(s).toHaveProperty('total');
      expect(s).toHaveProperty('healthPct');
      expect(s.healthPct).toBeLessThanOrEqual(100);
    });

    test('100% health when no memories', () => {
      const s = chronos.maintenanceSummary();
      expect(s.total).toBe(0);
      expect(s.healthPct).toBe(100);
    });
  });
});
