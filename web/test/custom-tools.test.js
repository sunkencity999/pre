// Tests for src/custom-tools.js — dynamic virtual tool system
const fs = require('fs');
const path = require('path');

jest.mock('../src/constants');

const { PRE_DIR } = require('../src/constants');
const customTools = require('../src/custom-tools');

// Custom tools dir for this test worker
const CUSTOM_TOOLS_DIR = path.join(PRE_DIR, 'custom_tools');

describe('custom-tools', () => {
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

  describe('customTool dispatcher', () => {
    test('create a prompt-type tool', () => {
      const result = customTools.customTool({
        action: 'create',
        name: 'greet',
        description: 'Greet someone by name',
        template: 'Hello, ${name}! Welcome to PRE.',
        parameters: JSON.stringify([
          { name: 'name', type: 'string', description: 'Person to greet', required: true },
        ]),
      });
      expect(result).toContain('Custom tool created');
      expect(result).toContain('custom_greet');
    });

    test('create a chain-type tool', () => {
      const result = customTools.customTool({
        action: 'create',
        name: 'search_and_save',
        description: 'Search web and save result',
        steps: JSON.stringify([
          { tool: 'web_search', args: { query: '${query}' } },
          { tool: 'file_write', args: { path: '${output_path}', content: '${step1}' } },
        ]),
        parameters: JSON.stringify([
          { name: 'query', type: 'string', required: true },
          { name: 'output_path', type: 'string', required: true },
        ]),
      });
      expect(result).toContain('Custom tool created');
      expect(result).toContain('chain');
    });

    test('list shows created tools', () => {
      customTools.customTool({
        action: 'create', name: 'test_tool',
        description: 'A test', template: 'test ${x}',
      });
      const list = customTools.customTool({ action: 'list' });
      expect(list).toContain('custom_test_tool');
      expect(list).toContain('A test');
    });

    test('show returns tool details', () => {
      customTools.customTool({
        action: 'create', name: 'detail_tool',
        description: 'Detail test', template: 'hello ${name}',
        parameters: JSON.stringify([{ name: 'name', type: 'string' }]),
      });
      const detail = customTools.customTool({ action: 'show', name: 'detail_tool' });
      expect(detail).toContain('detail_tool');
      expect(detail).toContain('prompt');
    });

    test('delete removes a tool', () => {
      customTools.customTool({
        action: 'create', name: 'to_delete',
        description: 'Will be deleted', template: 'x',
      });
      const result = customTools.customTool({ action: 'delete', name: 'to_delete' });
      expect(result).toContain('Deleted');
      const list = customTools.customTool({ action: 'list' });
      expect(list).not.toContain('to_delete');
    });

    test('create requires name and description', () => {
      expect(customTools.customTool({ action: 'create' })).toContain('Error');
      expect(customTools.customTool({ action: 'create', name: 'x' })).toContain('Error');
    });
  });

  describe('isCustomTool', () => {
    test('returns false for non-custom tools', () => {
      expect(customTools.isCustomTool('bash')).toBe(false);
      expect(customTools.isCustomTool('read_file')).toBe(false);
    });

    test('returns true for created custom tools', () => {
      customTools.customTool({
        action: 'create', name: 'my_tool',
        description: 'test', template: 'hello',
      });
      expect(customTools.isCustomTool('custom_my_tool')).toBe(true);
    });

    test('returns false for non-existent custom tools', () => {
      expect(customTools.isCustomTool('custom_nonexistent')).toBe(false);
    });
  });

  describe('executeCustomTool', () => {
    test('executes prompt template with parameter substitution', async () => {
      customTools.customTool({
        action: 'create', name: 'greeter',
        description: 'Greet', template: 'Hello, ${name}! You are ${role}.',
      });
      const result = await customTools.executeCustomTool('greeter', { name: 'Chris', role: 'admin' });
      expect(result).toContain('Hello, Chris!');
      expect(result).toContain('You are admin.');
    });

    test('returns error for non-existent tool', async () => {
      const result = await customTools.executeCustomTool('nope', {});
      expect(result).toContain('Error');
    });
  });

  describe('buildCustomToolDefs', () => {
    test('returns empty array with no tools', () => {
      const defs = customTools.buildCustomToolDefs();
      expect(defs).toEqual([]);
    });

    test('returns tool definitions for created tools', () => {
      customTools.customTool({
        action: 'create', name: 'def_test',
        description: 'Test tool', template: '${x}',
        parameters: JSON.stringify([{ name: 'x', type: 'string', required: true }]),
      });
      const defs = customTools.buildCustomToolDefs();
      expect(defs.length).toBe(1);
      expect(defs[0].type).toBe('function');
      expect(defs[0].function.name).toBe('custom_def_test');
      expect(defs[0].function.parameters.properties).toHaveProperty('x');
      expect(defs[0].function.parameters.required).toContain('x');
    });
  });
});
