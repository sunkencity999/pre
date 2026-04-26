// Tests for src/tools-defs.js — Ollama tool definition builder
const fs = require('fs');

jest.mock('../src/constants');
jest.mock('../src/ollama', () => ({
  streamChat: jest.fn().mockResolvedValue({ response: '[]' }),
}));
jest.mock('../src/mcp', () => ({
  getConnectedTools: jest.fn().mockReturnValue([]),
  getAllTools: jest.fn().mockReturnValue([]),
  loadConfig: jest.fn().mockReturnValue({ servers: {} }),
}));

const { CONNECTIONS_FILE } = require('../src/constants');
const toolsDefs = require('../src/tools-defs');

describe('tools-defs', () => {
  beforeEach(() => {
    fs.writeFileSync(CONNECTIONS_FILE, '{}');
  });

  describe('buildToolDefs', () => {
    test('returns an array of tool definitions', () => {
      const tools = toolsDefs.buildToolDefs();
      expect(Array.isArray(tools)).toBe(true);
      expect(tools.length).toBeGreaterThan(30);
    });

    test('each tool has correct structure', () => {
      for (const t of toolsDefs.buildToolDefs()) {
        expect(t.type).toBe('function');
        expect(t.function).toHaveProperty('name');
        expect(t.function).toHaveProperty('description');
        expect(t.function.parameters).toHaveProperty('type', 'object');
        expect(t.function.parameters).toHaveProperty('properties');
      }
    });

    test('includes core tools', () => {
      const names = toolsDefs.buildToolDefs().map(t => t.function.name);
      expect(names).toContain('bash');
      expect(names).toContain('read_file');
      expect(names).toContain('file_write');
      expect(names).toContain('file_edit');
      expect(names).toContain('web_fetch');
      expect(names).toContain('memory_save');
    });

    test('includes Apple tools', () => {
      const names = toolsDefs.buildToolDefs().map(t => t.function.name);
      expect(names).toContain('apple_mail');
      expect(names).toContain('apple_calendar');
      expect(names).toContain('apple_contacts');
      expect(names).toContain('apple_reminders');
      expect(names).toContain('apple_notes');
      expect(names).toContain('spotlight');
    });

    test('includes scheduling tools', () => {
      const names = toolsDefs.buildToolDefs().map(t => t.function.name);
      expect(names).toContain('cron');
      expect(names).toContain('trigger');
    });

    test('tool names are unique', () => {
      const names = toolsDefs.buildToolDefs().map(t => t.function.name);
      expect(new Set(names).size).toBe(names.length);
    });

    test('required fields are arrays', () => {
      for (const t of toolsDefs.buildToolDefs()) {
        if (t.function.parameters.required) {
          expect(Array.isArray(t.function.parameters.required)).toBe(true);
        }
      }
    });

    test('conditional tools absent without credentials', () => {
      const names = toolsDefs.buildToolDefs().map(t => t.function.name);
      expect(names).not.toContain('github');
      expect(names).not.toContain('jira');
      expect(names).not.toContain('dynamics365');
    });

    test('conditional tools appear with credentials', () => {
      fs.writeFileSync(CONNECTIONS_FILE, JSON.stringify({
        github_key: 'ghp_test',
        jira_url: 'https://jira.test', jira_token: 'pat',
      }));
      const names = toolsDefs.buildToolDefs().map(t => t.function.name);
      expect(names).toContain('github');
      expect(names).toContain('jira');
      fs.writeFileSync(CONNECTIONS_FILE, '{}');
    });
  });
});
