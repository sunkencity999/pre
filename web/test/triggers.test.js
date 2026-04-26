// Tests for src/triggers.js — event-driven trigger engine
const fs = require('fs');
const path = require('path');

jest.mock('../src/constants');
const { PRE_DIR } = require('../src/constants');
const triggers = require('../src/triggers');

const triggersFile = path.join(PRE_DIR, 'triggers.json');

describe('triggers', () => {
  beforeEach(() => {
    fs.writeFileSync(triggersFile, '[]');
  });

  afterAll(() => triggers.shutdown());

  // ── dispatcher ──

  test('returns error for unknown action', () => {
    expect(triggers.trigger({ action: 'bogus' })).toMatch(/error/i);
  });

  // ── add ──

  describe('add trigger', () => {
    test('creates a file_watch trigger', () => {
      const result = triggers.trigger({
        action: 'add', type: 'file_watch', name: 'Watcher',
        prompt: 'Process {path}', path: PRE_DIR,
      });
      expect(result).toContain('Trigger created');
      expect(triggers.loadTriggers()).toHaveLength(1);
    });

    test('creates a webhook trigger with secret', () => {
      const result = triggers.trigger({
        action: 'add', type: 'webhook', name: 'CI', prompt: '{payload}', secret: 's3cret',
      });
      expect(result).toContain('webhook');
      expect(triggers.loadTriggers()[0].config.secret).toBe('s3cret');
    });

    test('rejects invalid type', () => {
      expect(triggers.trigger({ action: 'add', type: 'bad', prompt: 'x' })).toMatch(/error/i);
    });

    test('rejects missing prompt', () => {
      expect(triggers.trigger({ action: 'add', type: 'webhook' })).toMatch(/error/i);
    });

    test('rejects file_watch without path', () => {
      expect(triggers.trigger({ action: 'add', type: 'file_watch', prompt: 'x' })).toMatch(/error/i);
    });
  });

  // ── list ──

  describe('list', () => {
    test('empty message when none', () => {
      expect(triggers.trigger({ action: 'list' })).toMatch(/no triggers/i);
    });

    test('lists created triggers', () => {
      triggers.trigger({ action: 'add', type: 'webhook', name: 'T', prompt: 'x' });
      expect(triggers.trigger({ action: 'list' })).toContain('1 trigger');
    });
  });

  // ── remove ──

  describe('remove', () => {
    test('removes by ID', () => {
      triggers.trigger({ action: 'add', type: 'webhook', name: 'R', prompt: 'x' });
      const id = triggers.loadTriggers()[0].id;
      expect(triggers.trigger({ action: 'remove', id })).toContain('Removed');
      expect(triggers.loadTriggers()).toHaveLength(0);
    });

    test('error for bad ID', () => {
      expect(triggers.trigger({ action: 'remove', id: 'nope' })).toMatch(/error/i);
    });
  });

  // ── enable / disable ──

  test('disables and re-enables', () => {
    triggers.trigger({ action: 'add', type: 'webhook', name: 'E', prompt: 'x' });
    const id = triggers.loadTriggers()[0].id;
    triggers.trigger({ action: 'disable', id });
    expect(triggers.loadTriggers()[0].enabled).toBe(false);
    triggers.trigger({ action: 'enable', id });
    expect(triggers.loadTriggers()[0].enabled).toBe(true);
  });

  // ── isWatching ──

  test('isWatching returns false for nonexistent', () => {
    expect(triggers.isWatching('fake')).toBe(false);
  });

  // ── handleWebhook ──

  describe('handleWebhook', () => {
    test('404 for unknown ID', () => {
      expect(triggers.handleWebhook('x', { body: {}, headers: {} }).status).toBe(404);
    });

    test('403 for disabled trigger', () => {
      triggers.trigger({ action: 'add', type: 'webhook', name: 'D', prompt: 'x' });
      const id = triggers.loadTriggers()[0].id;
      triggers.trigger({ action: 'disable', id });
      expect(triggers.handleWebhook(id, { body: {}, headers: {} }).status).toBe(403);
    });

    test('401 for wrong secret', () => {
      triggers.trigger({ action: 'add', type: 'webhook', name: 'S', prompt: 'x', secret: 'ok' });
      const id = triggers.loadTriggers()[0].id;
      expect(triggers.handleWebhook(id, { body: {}, headers: { 'x-webhook-secret': 'bad' } }).status).toBe(401);
    });
  });

  // ── restartWatcher ──

  test('restartWatcher error for nonexistent', () => {
    expect(triggers.restartWatcher('fake').error).toBeTruthy();
  });

  test('restartWatcher error for webhook type', () => {
    triggers.trigger({ action: 'add', type: 'webhook', name: 'W', prompt: 'x' });
    const id = triggers.loadTriggers()[0].id;
    expect(triggers.restartWatcher(id).error).toMatch(/not a file watcher/i);
  });

  // ── init ──

  test('init returns stats', () => {
    const stats = triggers.init(jest.fn().mockResolvedValue({}));
    expect(stats).toHaveProperty('watcherCount');
    expect(stats).toHaveProperty('total');
  });
});
