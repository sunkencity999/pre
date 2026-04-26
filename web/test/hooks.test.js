// Tests for src/hooks.js — pre/post tool execution hooks
const fs = require('fs');
const path = require('path');
const os = require('os');

// Use a temp hooks file
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pre-hooks-'));
const HOOKS_PATH = path.join(tmpDir, 'hooks.json');

// Mock the HOOKS_PATH by setting HOME to our temp parent
const origHome = process.env.HOME;
const fakeHome = fs.mkdtempSync(path.join(os.tmpdir(), 'pre-home-'));
const fakePreDir = path.join(fakeHome, '.pre');
fs.mkdirSync(fakePreDir, { recursive: true });
fs.writeFileSync(path.join(fakePreDir, 'hooks.json'), JSON.stringify({ hooks: [] }));

beforeAll(() => {
  process.env.HOME = fakeHome;
});

afterAll(() => {
  process.env.HOME = origHome;
  fs.rmSync(tmpDir, { recursive: true, force: true });
  fs.rmSync(fakeHome, { recursive: true, force: true });
});

// Require after HOME is set
const hooks = require('../src/hooks');

describe('hooks', () => {
  beforeEach(() => {
    // Reset hooks file
    hooks.saveHooks([]);
  });

  // ── addHook ──

  describe('addHook', () => {
    test('adds a hook with auto-generated ID', () => {
      const hook = hooks.addHook({
        event: 'pre_tool',
        command: 'echo test',
        description: 'Test hook',
      });
      expect(hook.id).toBeTruthy();
      expect(hook.event).toBe('pre_tool');
      expect(hook.enabled).toBe(true);
    });

    test('rejects hook without event', () => {
      const result = hooks.addHook({ command: 'echo hi' });
      expect(result.error).toBeTruthy();
    });

    test('rejects hook without command', () => {
      const result = hooks.addHook({ event: 'pre_tool' });
      expect(result.error).toBeTruthy();
    });

    test('preserves custom ID', () => {
      const hook = hooks.addHook({
        id: 'my-hook',
        event: 'post_tool',
        command: 'echo done',
      });
      expect(hook.id).toBe('my-hook');
    });
  });

  // ── listHooks ──

  describe('listHooks', () => {
    test('returns empty array when no hooks', () => {
      expect(hooks.listHooks()).toEqual([]);
    });

    test('returns added hooks', () => {
      hooks.addHook({ event: 'pre_tool', command: 'echo a' });
      hooks.addHook({ event: 'post_tool', command: 'echo b' });
      expect(hooks.listHooks()).toHaveLength(2);
    });
  });

  // ── removeHook ──

  describe('removeHook', () => {
    test('removes by ID', () => {
      hooks.addHook({ id: 'removable', event: 'pre_tool', command: 'echo x' });
      const result = hooks.removeHook('removable');
      expect(result.id).toBe('removable');
      expect(hooks.listHooks()).toHaveLength(0);
    });

    test('returns error for missing ID', () => {
      expect(hooks.removeHook('nonexistent').error).toBeTruthy();
    });
  });

  // ── toggleHook ──

  describe('toggleHook', () => {
    test('toggles enabled state', () => {
      hooks.addHook({ id: 'toggler', event: 'pre_tool', command: 'echo x' });
      const disabled = hooks.toggleHook('toggler');
      expect(disabled.enabled).toBe(false);
      const enabled = hooks.toggleHook('toggler');
      expect(enabled.enabled).toBe(true);
    });

    test('returns error for missing hook', () => {
      expect(hooks.toggleHook('nope').error).toBeTruthy();
    });
  });

  // ── runHooks ──

  describe('runHooks', () => {
    test('returns not-blocked when no hooks match', () => {
      const result = hooks.runHooks('pre_tool', { tool: 'bash' });
      expect(result.blocked).toBe(false);
    });

    test('runs matching hook command', () => {
      const logFile = path.join(fakeHome, 'hook-test.log');
      hooks.addHook({
        id: 'logger',
        event: 'pre_tool',
        tool: 'bash',
        command: `echo "ran" > "${logFile}"`,
      });
      hooks.runHooks('pre_tool', { tool: 'bash' });
      expect(fs.existsSync(logFile)).toBe(true);
    });

    test('does not run hooks for non-matching tool', () => {
      const logFile = path.join(fakeHome, 'should-not-exist.log');
      hooks.addHook({
        event: 'pre_tool',
        tool: 'bash',
        command: `touch "${logFile}"`,
      });
      hooks.runHooks('pre_tool', { tool: 'memory' });
      expect(fs.existsSync(logFile)).toBe(false);
    });

    test('wildcard tool matches all tools', () => {
      const logFile = path.join(fakeHome, 'wildcard.log');
      hooks.addHook({
        event: 'pre_tool',
        tool: '*',
        command: `touch "${logFile}"`,
      });
      hooks.runHooks('pre_tool', { tool: 'anything' });
      expect(fs.existsSync(logFile)).toBe(true);
    });

    test('blocking hook prevents execution', () => {
      hooks.addHook({
        id: 'blocker',
        event: 'pre_tool',
        tool: 'bash',
        command: 'exit 1',
        can_block: true,
      });
      const result = hooks.runHooks('pre_tool', { tool: 'bash' });
      expect(result.blocked).toBe(true);
    });

    test('non-blocking hook failure does not block', () => {
      hooks.addHook({
        event: 'pre_tool',
        tool: 'bash',
        command: 'exit 1',
        can_block: false,
      });
      const result = hooks.runHooks('pre_tool', { tool: 'bash' });
      expect(result.blocked).toBe(false);
    });

    test('disabled hooks are skipped', () => {
      const logFile = path.join(fakeHome, 'disabled.log');
      hooks.addHook({
        id: 'off',
        event: 'pre_tool',
        tool: '*',
        command: `touch "${logFile}"`,
        enabled: true,
      });
      hooks.toggleHook('off'); // disable it
      hooks.runHooks('pre_tool', { tool: 'bash' });
      expect(fs.existsSync(logFile)).toBe(false);
    });
  });
});
