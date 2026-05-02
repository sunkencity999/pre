// Tests for src/context.js — system prompt builder
const fs = require('fs');

jest.mock('../src/constants');
jest.mock('../src/ollama', () => ({
  streamChat: jest.fn().mockResolvedValue({ response: '[]' }),
}));

const { CONNECTIONS_FILE } = require('../src/constants');
const context = require('../src/context');

describe('context', () => {
  beforeEach(() => {
    context.invalidateConnectionsCache();
  });

  describe('getActiveConnections', () => {
    test('returns boolean fields for each service', () => {
      const c = context.getActiveConnections();
      expect(c).toHaveProperty('brave');
      expect(c).toHaveProperty('github');
      expect(c).toHaveProperty('google');
      expect(c).toHaveProperty('telegram');
      expect(c).toHaveProperty('jira');
      expect(c).toHaveProperty('dynamics365');
    });

    test('all false with empty connections', () => {
      fs.writeFileSync(CONNECTIONS_FILE, '{}');
      const c = context.getActiveConnections();
      for (const val of Object.values(c)) {
        expect(val).toBe(false);
      }
    });

    test('detects active github key', () => {
      fs.writeFileSync(CONNECTIONS_FILE, JSON.stringify({ github_key: 'ghp_x' }));
      expect(context.getActiveConnections().github).toBe(true);
      fs.writeFileSync(CONNECTIONS_FILE, '{}');
    });
  });

  describe('buildSystemPrompt', () => {
    test('returns a non-empty string', () => {
      const prompt = context.buildSystemPrompt('/tmp');
      expect(typeof prompt).toBe('string');
      expect(prompt.length).toBeGreaterThan(100);
    });

    test('includes current date', () => {
      const prompt = context.buildSystemPrompt('/tmp');
      const year = new Date().getFullYear().toString();
      expect(prompt).toContain(year);
    });
  });

  describe('isComfyUIInstalled', () => {
    test('returns false when comfyui.json missing', () => {
      expect(context.isComfyUIInstalled()).toBe(false);
    });
  });
});
