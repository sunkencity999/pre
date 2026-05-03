// Tests for src/cron-runner.js — headless cron job execution
jest.mock('../src/constants');
jest.mock('../src/tools', () => ({
  runToolLoop: jest.fn(async ({ send }) => {
    send({ type: 'token', content: 'Test ' });
    send({ type: 'token', content: 'response.' });
    send({ type: 'done' });
  }),
  waitForIdle: jest.fn().mockResolvedValue(),
}));
jest.mock('../src/ollama', () => ({
  streamChat: jest.fn().mockResolvedValue({ response: '[]' }),
}));

const { executeCronJob } = require('../src/cron-runner');
const sessions = require('../src/sessions');

describe('cron-runner', () => {
  describe('executeCronJob', () => {
    test('creates a session and returns response', async () => {
      const result = await executeCronJob({
        id: 'test-job', description: 'Test', prompt: 'Say hello',
      });
      expect(result.sessionId).toBeTruthy();
      expect(result.response).toBe('Test response.');
    });

    test('session is named after description', async () => {
      const result = await executeCronJob({
        id: 'named', description: 'Daily Briefing', prompt: 'Brief me',
      });
      const all = sessions.listSessions();
      const found = all.find(s => s.id === result.sessionId);
      expect(found.displayName).toContain('Daily Briefing');
    });

    test('uses job.cwd when provided', async () => {
      const { runToolLoop } = require('../src/tools');
      runToolLoop.mockClear();
      await executeCronJob({
        id: 'cwd', description: 'CWD', prompt: 'ls', cwd: '/tmp/custom',
      });
      expect(runToolLoop).toHaveBeenCalledWith(
        expect.objectContaining({ cwd: '/tmp/custom' })
      );
    });

    test('falls back to HOME when no cwd', async () => {
      const { runToolLoop } = require('../src/tools');
      runToolLoop.mockClear();
      await executeCronJob({ id: 'home', description: 'H', prompt: 'x' });
      expect(runToolLoop).toHaveBeenCalledWith(
        expect.objectContaining({ cwd: process.env.HOME || '/tmp' })
      );
    });

    test('each job creates a unique session', async () => {
      const r1 = await executeCronJob({ id: 'dup', description: 'A', prompt: 'a' });
      const r2 = await executeCronJob({ id: 'dup', description: 'B', prompt: 'b' });
      expect(r1.sessionId).not.toBe(r2.sessionId);
    });

    test('broadcasts WS events', async () => {
      const broadcast = jest.fn();
      await executeCronJob(
        { id: 'ws', description: 'WS', prompt: 'x' },
        { broadcastWS: broadcast }
      );
      expect(broadcast).toHaveBeenCalled();
      const types = broadcast.mock.calls.map(c => c[0].type);
      expect(types).toContain('token');
      expect(types).toContain('done');
    });
  });
});
