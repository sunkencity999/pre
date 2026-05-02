// Tests for Self-Improving Skills (skills.js)

const fs = require('fs');

jest.mock('../src/constants');
jest.mock('../src/ollama', () => ({
  streamChat: jest.fn(),
}));
jest.mock('../src/mcp', () => ({
  getConnectedTools: jest.fn().mockReturnValue([]),
  getAllTools: jest.fn().mockReturnValue([]),
  loadConfig: jest.fn().mockReturnValue({ servers: {} }),
}));

const { CONNECTIONS_FILE } = require('../src/constants');

beforeEach(() => {
  fs.writeFileSync(CONNECTIONS_FILE, '{}');
});

describe('skills — chain extraction', () => {
  const skills = require('../src/skills');

  test('extracts tool calls from assistant messages', () => {
    const messages = [
      { role: 'user', content: 'find files' },
      {
        role: 'assistant',
        content: 'Let me search.',
        tool_calls: [
          { function: { name: 'glob', arguments: { pattern: '**/*.js' } } },
        ],
      },
      {
        role: 'tool',
        content: '<tool_response name="glob">file1.js\nfile2.js</tool_response>',
      },
      {
        role: 'assistant',
        content: 'Now searching contents.',
        tool_calls: [
          { function: { name: 'grep', arguments: { pattern: 'TODO' } } },
        ],
      },
      {
        role: 'tool',
        content: '<tool_response name="grep">file1.js:5: // TODO fix\nfile2.js:10: // TODO refactor</tool_response>',
      },
    ];
    const chains = skills.extractToolChains(messages);
    expect(chains.length).toBe(2);
    expect(chains[0].tool).toBe('glob');
    expect(chains[0].success).toBe(true);
    expect(chains[0].result).toContain('file1.js');
    expect(chains[1].tool).toBe('grep');
    expect(chains[1].success).toBe(true);
  });

  test('marks error results as unsuccessful', () => {
    const messages = [
      {
        role: 'assistant',
        tool_calls: [{ function: { name: 'bash', arguments: { command: 'bad-cmd' } } }],
      },
      {
        role: 'tool',
        content: '<tool_response name="bash">Error: command not found</tool_response>',
      },
    ];
    const chains = skills.extractToolChains(messages);
    expect(chains.length).toBe(1);
    expect(chains[0].success).toBe(false);
  });

  test('returns empty array for no tool calls', () => {
    const messages = [
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hi' },
    ];
    expect(skills.extractToolChains(messages)).toEqual([]);
  });

  test('handles multiple tool calls in one message', () => {
    const messages = [
      {
        role: 'assistant',
        tool_calls: [
          { function: { name: 'glob', arguments: { pattern: '*.js' } } },
          { function: { name: 'grep', arguments: { pattern: 'test' } } },
        ],
      },
      {
        role: 'tool',
        content: '<tool_response name="glob">a.js</tool_response>\n<tool_response name="grep">b.js:1: test</tool_response>',
      },
    ];
    const chains = skills.extractToolChains(messages);
    expect(chains.length).toBe(2);
  });
});

describe('skills — deduplication', () => {
  const skills = require('../src/skills');

  test('detects duplicate by name', () => {
    const proposed = { name: 'my_skill', implementation: { type: 'chain', steps: [] } };
    const existing = [{ name: 'my_skill', implementation: { type: 'chain', steps: [] } }];
    expect(skills.isDuplicate(proposed, existing)).toBe(true);
  });

  test('detects duplicate by step overlap', () => {
    const proposed = {
      name: 'new_skill',
      implementation: {
        type: 'chain',
        steps: [{ tool: 'glob' }, { tool: 'grep' }, { tool: 'read_file' }],
      },
    };
    const existing = [{
      name: 'old_skill',
      implementation: {
        type: 'chain',
        steps: [{ tool: 'glob' }, { tool: 'grep' }, { tool: 'file_write' }],
      },
    }];
    // 2/3 = 67% overlap, under 70% threshold
    expect(skills.isDuplicate(proposed, existing)).toBe(false);
  });

  test('detects high overlap as duplicate', () => {
    const proposed = {
      name: 'new_skill',
      implementation: {
        type: 'chain',
        steps: [{ tool: 'glob' }, { tool: 'grep' }, { tool: 'read_file' }, { tool: 'file_write' }],
      },
    };
    const existing = [{
      name: 'old_skill',
      implementation: {
        type: 'chain',
        steps: [{ tool: 'glob' }, { tool: 'grep' }, { tool: 'read_file' }],
      },
    }];
    // 3/4 = 75% overlap, above threshold
    expect(skills.isDuplicate(proposed, existing)).toBe(true);
  });

  test('non-chain tools are not duplicates', () => {
    const proposed = { name: 'new_skill', implementation: { type: 'prompt', template: 'hi' } };
    const existing = [{ name: 'other', implementation: { type: 'prompt', template: 'hello' } }];
    expect(skills.isDuplicate(proposed, existing)).toBe(false);
  });
});

describe('skills — gate checks', () => {
  const skills = require('../src/skills');

  test('returns empty for too few tool calls', async () => {
    const messages = [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
      { role: 'tool', content: '<tool_response name="bash">ok</tool_response>' },
    ];
    // Only 1 tool message, need MIN_TOOL_CALLS (3)
    const result = await skills.analyzeForSkills(messages, { sessionId: 'test' });
    expect(result).toEqual([]);
  });

  test('exports expected constants', () => {
    expect(skills.SKILL_COOLDOWN_MS).toBe(180000);
    expect(skills.MIN_TOOL_CALLS).toBe(3);
    expect(skills.OVERLAP_THRESHOLD).toBe(0.70);
  });
});

describe('skills — usage tracking in custom-tools', () => {
  const customTools = require('../src/custom-tools');

  test('exports saveCustomTool and deleteCustomTool', () => {
    expect(typeof customTools.saveCustomTool).toBe('function');
    expect(typeof customTools.deleteCustomTool).toBe('function');
  });

  test('loadCustomTools returns array', () => {
    const tools = customTools.loadCustomTools();
    expect(Array.isArray(tools)).toBe(true);
  });
});

describe('skills — FTS integration', () => {
  const fts = require('../src/fts');

  test('searchSessions is a function', () => {
    expect(typeof fts.searchSessions).toBe('function');
  });

  test('searchSessions returns error for empty query', () => {
    const result = fts.searchSessions('');
    expect(result).toContain('Error');
  });

  test('searchSessions returns no matches message for nonexistent text', () => {
    const result = fts.searchSessions('zzz_totally_nonexistent_query_xyz');
    expect(result).toContain('No matches');
  });
});
