// Tests for src/sessions.js — JSONL session management
const fs = require('fs');
const path = require('path');

jest.mock('../src/constants');
const { SESSIONS_DIR } = require('../src/constants');
const sessions = require('../src/sessions');

describe('sessions', () => {
  afterEach(() => {
    // Clean session files and meta between tests
    for (const f of fs.readdirSync(SESSIONS_DIR)) {
      fs.unlinkSync(path.join(SESSIONS_DIR, f));
    }
  });

  // ── createSession ──

  describe('createSession', () => {
    test('creates a new session file and returns an ID', () => {
      const id = sessions.createSession('web', 'test');
      expect(id).toBe('web:test');
      expect(fs.existsSync(path.join(SESSIONS_DIR, 'web:test.jsonl'))).toBe(true);
    });

    test('returns existing ID if session exists and forceNew=false', () => {
      sessions.createSession('web', 'dup');
      const id2 = sessions.createSession('web', 'dup');
      expect(id2).toBe('web:dup');
    });

    test('creates unique ID with forceNew=true', () => {
      const id1 = sessions.createSession('cron', 'job1', true);
      fs.writeFileSync(path.join(SESSIONS_DIR, `${id1}.jsonl`), '');
      const id2 = sessions.createSession('cron', 'job1', true);
      expect(id2).not.toBe(id1);
      expect(id2).toMatch(/^cron:job1-/);
    });
  });

  // ── appendMessage / getSession ──

  describe('appendMessage / getSession', () => {
    test('appends messages and reads them back', () => {
      const id = sessions.createSession('web', 'chat1');
      sessions.appendMessage(id, { role: 'user', content: 'Hello' });
      sessions.appendMessage(id, { role: 'assistant', content: 'Hi there' });

      const msgs = sessions.getSession(id);
      expect(msgs).toHaveLength(2);
      expect(msgs[0]).toEqual({ role: 'user', content: 'Hello' });
      expect(msgs[1]).toEqual({ role: 'assistant', content: 'Hi there' });
    });

    test('returns empty array for nonexistent session', () => {
      expect(sessions.getSession('nonexistent:session')).toEqual([]);
    });
  });

  // ── getSessionMessages ──

  describe('getSessionMessages', () => {
    test('filters out display-role messages', () => {
      const id = sessions.createSession('web', 'filter');
      sessions.appendMessage(id, { role: 'user', content: 'Hello' });
      sessions.appendMessage(id, { role: 'display', content: '<artifact>' });
      sessions.appendMessage(id, { role: 'assistant', content: 'Response' });

      const msgs = sessions.getSessionMessages(id);
      expect(msgs).toHaveLength(2);
      expect(msgs.every(m => m.role !== 'display')).toBe(true);
    });
  });

  // ── renameSession ──

  describe('renameSession', () => {
    test('sets display name for a session', () => {
      const id = sessions.createSession('web', 'named');
      sessions.renameSession(id, 'My Chat');
      const list = sessions.listSessions();
      const found = list.find(s => s.id === id);
      expect(found.displayName).toBe('My Chat');
    });

    test('clears display name when given empty string', () => {
      const id = sessions.createSession('web', 'clearname');
      sessions.renameSession(id, 'Temp');
      sessions.renameSession(id, '');
      const list = sessions.listSessions();
      const found = list.find(s => s.id === id);
      expect(found.displayName).toBeNull();
    });
  });

  // ── deleteSession ──

  describe('deleteSession', () => {
    test('deletes session file and metadata', () => {
      const id = sessions.createSession('web', 'todelete');
      sessions.appendMessage(id, { role: 'user', content: 'delete me' });
      expect(sessions.deleteSession(id)).toBe(true);
      expect(fs.existsSync(path.join(SESSIONS_DIR, `${id}.jsonl`))).toBe(false);
    });

    test('returns false for nonexistent session', () => {
      expect(sessions.deleteSession('nope:nope')).toBe(false);
    });
  });

  // ── rewindSession ──

  describe('rewindSession', () => {
    test('removes last N turn pairs', () => {
      const id = sessions.createSession('web', 'rewind');
      sessions.appendMessage(id, { role: 'user', content: 'Q1' });
      sessions.appendMessage(id, { role: 'assistant', content: 'A1' });
      sessions.appendMessage(id, { role: 'user', content: 'Q2' });
      sessions.appendMessage(id, { role: 'assistant', content: 'A2' });

      const kept = sessions.rewindSession(id, 1);
      expect(kept).toHaveLength(2);
      expect(kept[0].content).toBe('Q1');

      const reloaded = sessions.getSession(id);
      expect(reloaded).toHaveLength(2);
    });
  });

  // ── listSessions ──

  describe('listSessions', () => {
    test('includes preview from first user message', () => {
      const id = sessions.createSession('web', 'preview');
      sessions.appendMessage(id, { role: 'user', content: 'What is the weather?' });
      const list = sessions.listSessions();
      const found = list.find(s => s.id === id);
      expect(found.preview).toBe('What is the weather?');
    });

    test('filters by project slug', () => {
      const id1 = sessions.createSession('web', 'proj1');
      sessions.createSession('web', 'proj2');
      sessions.ensureProject('test-proj', 'Test');
      sessions.moveSessionToProject(id1, 'test-proj');

      const filtered = sessions.listSessions('test-proj');
      expect(filtered).toHaveLength(1);
      expect(filtered[0].id).toBe(id1);
    });
  });

  // ── Project management ──

  describe('project management', () => {
    test('createProject creates a project with slug', () => {
      const proj = sessions.createProject('My Project');
      expect(proj.slug).toBe('my-project');
      expect(proj.name).toBe('My Project');
    });

    test('createProject avoids duplicate slugs', () => {
      const p1 = sessions.createProject('Dup Test');
      const p2 = sessions.createProject('Dup Test');
      expect(p1.slug).not.toBe(p2.slug);
    });

    test('listProjects returns created projects', () => {
      sessions.createProject('Alpha');
      const projects = sessions.listProjects();
      expect(projects.some(p => p.name === 'Alpha')).toBe(true);
    });

    test('renameProject updates name', () => {
      const proj = sessions.createProject('RenameMe');
      sessions.renameProject(proj.slug, 'Renamed');
      const projects = sessions.listProjects();
      expect(projects.find(p => p.slug === proj.slug).name).toBe('Renamed');
    });

    test('deleteProject removes project and ungroups sessions', () => {
      const proj = sessions.createProject('Temp');
      const sid = sessions.createSession('web', 'tempsess');
      sessions.moveSessionToProject(sid, proj.slug);
      expect(sessions.deleteProject(proj.slug)).toBe(true);
      expect(sessions.listProjects().find(p => p.slug === proj.slug)).toBeUndefined();
    });

    test('ensureProject creates if missing, no-ops if exists', () => {
      expect(sessions.ensureProject('auto', 'Auto')).toBe('auto');
      expect(sessions.ensureProject('auto', 'Auto')).toBe('auto');
    });

    test('moveSessionToProject assigns and unassigns', () => {
      const proj = sessions.createProject('Move Test');
      const sid = sessions.createSession('web', 'movable');
      sessions.moveSessionToProject(sid, proj.slug);
      expect(sessions.listSessions(proj.slug).some(s => s.id === sid)).toBe(true);
      sessions.moveSessionToProject(sid, null);
      expect(sessions.listSessions(proj.slug).some(s => s.id === sid)).toBe(false);
    });
  });
});
