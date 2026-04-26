// Integration tests for custom tool dispatch chain:
// tools-defs.js builds defs → tools.js dispatches → custom-tools.js executes
const fs = require('fs');
const path = require('path');

jest.mock('../src/constants');
jest.mock('../src/mcp', () => ({
  getConnectedTools: jest.fn().mockReturnValue([]),
  getAllTools: jest.fn().mockReturnValue([]),
  loadConfig: jest.fn().mockReturnValue({ servers: {} }),
}));

const { PRE_DIR } = require('../src/constants');
const customTools = require('../src/custom-tools');
const { buildToolDefs } = require('../src/tools-defs');

const CUSTOM_TOOLS_DIR = path.join(PRE_DIR, 'custom_tools');

describe('custom tool dispatch — integration', () => {
  beforeEach(() => {
    // Clean custom tools dir
    if (fs.existsSync(CUSTOM_TOOLS_DIR)) {
      for (const f of fs.readdirSync(CUSTOM_TOOLS_DIR)) {
        fs.unlinkSync(path.join(CUSTOM_TOOLS_DIR, f));
      }
    } else {
      fs.mkdirSync(CUSTOM_TOOLS_DIR, { recursive: true });
    }
  });

  // ── Tool definition injection ──

  describe('tool definitions include custom tools', () => {
    test('custom tools appear in buildToolDefs() output', () => {
      customTools.customTool({
        action: 'create',
        name: 'greet_user',
        description: 'Greet a user by name',
        template: 'Hello, ${name}!',
        parameters: JSON.stringify([
          { name: 'name', type: 'string', required: true },
        ]),
      });

      const defs = buildToolDefs();
      const customDef = defs.find(d => d.function?.name === 'custom_greet_user');

      expect(customDef).toBeDefined();
      expect(customDef.function.description).toContain('Greet a user');
      expect(customDef.function.parameters.properties).toHaveProperty('name');
      expect(customDef.function.parameters.required).toContain('name');
    });

    test('deleted custom tools disappear from buildToolDefs()', () => {
      customTools.customTool({
        action: 'create',
        name: 'temp_tool',
        description: 'Temporary',
        template: 'x',
      });

      let defs = buildToolDefs();
      expect(defs.find(d => d.function?.name === 'custom_temp_tool')).toBeDefined();

      customTools.customTool({ action: 'delete', name: 'temp_tool' });

      defs = buildToolDefs();
      expect(defs.find(d => d.function?.name === 'custom_temp_tool')).toBeUndefined();
    });

    test('multiple custom tools all appear in definitions', () => {
      for (let i = 1; i <= 3; i++) {
        customTools.customTool({
          action: 'create',
          name: `tool_${i}`,
          description: `Tool number ${i}`,
          template: `Output ${i}: \${input}`,
        });
      }

      const defs = buildToolDefs();
      const customDefs = defs.filter(d => d.function?.name?.startsWith('custom_tool_'));
      expect(customDefs.length).toBe(3);
    });
  });

  // ── isCustomTool detection ──

  describe('tool name resolution', () => {
    test('isCustomTool recognizes created tools', () => {
      customTools.customTool({
        action: 'create',
        name: 'my_checker',
        description: 'Check something',
        template: 'Checking ${target}',
      });

      expect(customTools.isCustomTool('custom_my_checker')).toBe(true);
      expect(customTools.isCustomTool('custom_nonexistent')).toBe(false);
      expect(customTools.isCustomTool('bash')).toBe(false);
      expect(customTools.isCustomTool('my_checker')).toBe(false);
    });
  });

  // ── End-to-end execution ──

  describe('prompt template execution', () => {
    test('substitutes all parameters into template', async () => {
      customTools.customTool({
        action: 'create',
        name: 'email_template',
        description: 'Generate email',
        template: 'Dear ${name},\n\nRe: ${subject}\n\n${body}\n\nBest,\n${sender}',
        parameters: JSON.stringify([
          { name: 'name', type: 'string', required: true },
          { name: 'subject', type: 'string', required: true },
          { name: 'body', type: 'string', required: true },
          { name: 'sender', type: 'string', required: true },
        ]),
      });

      const result = await customTools.executeCustomTool('email_template', {
        name: 'Alice',
        subject: 'Project Update',
        body: 'The build is green.',
        sender: 'Bob',
      });

      expect(result).toContain('Dear Alice');
      expect(result).toContain('Re: Project Update');
      expect(result).toContain('The build is green.');
      expect(result).toContain('Best,\nBob');
    });

    test('handles special characters in parameter values', async () => {
      customTools.customTool({
        action: 'create',
        name: 'code_wrap',
        description: 'Wrap code in markdown',
        template: '```${lang}\n${code}\n```',
      });

      const result = await customTools.executeCustomTool('code_wrap', {
        lang: 'javascript',
        code: 'const x = "hello $world"; // comment\nconst y = `template ${literal}`;',
      });

      expect(result).toContain('```javascript');
      expect(result).toContain('const x = "hello $world"');
    });

    test('unsubstituted parameters remain as placeholders', async () => {
      customTools.customTool({
        action: 'create',
        name: 'partial',
        description: 'Test partial substitution',
        template: 'Hello ${name}, your role is ${role}.',
      });

      const result = await customTools.executeCustomTool('partial', { name: 'Chris' });

      expect(result).toContain('Hello Chris');
      expect(result).toContain('${role}'); // unsubstituted
    });
  });

  // ── Chain execution ──

  describe('chain tool execution', () => {
    test('chain steps reference parameters and previous results', async () => {
      customTools.customTool({
        action: 'create',
        name: 'search_chain',
        description: 'Search and save',
        steps: JSON.stringify([
          { tool: 'web_search', args: { query: '${topic} latest news' } },
          { tool: 'file_write', args: { path: '${output}', content: 'Results: ${step1}' } },
        ]),
        parameters: JSON.stringify([
          { name: 'topic', type: 'string', required: true },
          { name: 'output', type: 'string', required: true },
        ]),
      });

      const result = await customTools.executeCustomTool('search_chain', {
        topic: 'AI safety',
        output: '/tmp/results.txt',
      });

      expect(result).toContain('Chain');
      expect(result).toContain('web_search');
      expect(result).toContain('AI safety latest news');
      expect(result).toContain('file_write');
      expect(result).toContain('/tmp/results.txt');
    });
  });

  // ── Error cases ──

  describe('error handling', () => {
    test('executeCustomTool returns error for nonexistent tool', async () => {
      const result = await customTools.executeCustomTool('ghost_tool', {});
      expect(result).toContain('Error');
      expect(result).toContain('not found');
    });

    test('create rejects tools without implementation', () => {
      const result = customTools.customTool({
        action: 'create',
        name: 'empty_tool',
        description: 'Has no implementation',
        // no template, steps, or workflow_name
      });
      expect(result).toContain('Error');
    });
  });
});
