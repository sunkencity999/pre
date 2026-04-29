// Integration tests for src/tools/agents.js — agent spawn, timeouts, tool execution
jest.mock('../src/constants');
// Mock ollama at the module boundary — tests cross-module flow, not Ollama HTTP
jest.mock('../src/ollama');
// Mock MCP so tool-defs doesn't fail
jest.mock('../src/mcp', () => ({
  getConnectedTools: jest.fn().mockReturnValue([]),
  getAllTools: jest.fn().mockReturnValue([]),
  loadConfig: jest.fn().mockReturnValue({ servers: {} }),
}));

// Mock http so ensureModelReady's raw Ollama calls don't hit the network.
// This is needed because agents.js uses http.get(/api/ps) and http.request(/api/generate)
// directly, bypassing the mocked ollama module.
jest.mock('http', () => {
  const actual = jest.requireActual('http');
  return {
    ...actual,
    get: jest.fn((url, cb) => {
      // Simulate /api/ps returning the model as loaded
      const res = new (require('events').EventEmitter)();
      res.statusCode = 200;
      process.nextTick(() => {
        res.emit('data', JSON.stringify({ models: [{ name: 'pre-gemma4:latest' }] }));
        res.emit('end');
      });
      if (cb) cb(res);
      const req = new (require('events').EventEmitter)();
      req.setTimeout = jest.fn();
      req.destroy = jest.fn();
      return req;
    }),
    request: jest.fn((opts, cb) => {
      // Simulate /api/generate returning success
      const res = new (require('events').EventEmitter)();
      res.statusCode = 200;
      process.nextTick(() => {
        res.emit('data', JSON.stringify({ response: 'ok' }));
        res.emit('end');
      });
      if (cb) cb(res);
      const req = new (require('events').EventEmitter)();
      req.write = jest.fn();
      req.end = jest.fn();
      return req;
    }),
  };
});

const ollama = require('../src/ollama');
const { spawnAgent, spawnMulti, listAgents } = require('../src/tools/agents');

// Helper: mock streamChat to return a simple text response (no tool calls)
function mockSimpleResponse(response) {
  return () => Promise.resolve({
    response,
    thinking: '',
    toolCalls: null,
    stats: { eval_count: 50, eval_duration: 1e9 },
  });
}

// Helper: mock streamChat that returns tool calls on first call, text on subsequent
function mockToolThenText(toolCalls, finalResponse) {
  let callCount = 0;
  return () => {
    callCount++;
    if (callCount === 1) {
      return Promise.resolve({
        response: '',
        thinking: '',
        toolCalls,
        stats: {},
      });
    }
    return Promise.resolve({
      response: finalResponse,
      thinking: '',
      toolCalls: null,
      stats: {},
    });
  };
}

describe('agents — integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Default: healthy Ollama
    ollama.healthCheck.mockResolvedValue(true);
  });

  // ── ensureModelReady integration ──

  describe('pre-flight health check', () => {
    test('agent fails fast when Ollama is down', async () => {
      ollama.healthCheck.mockResolvedValue(false);
      ollama.streamChat.mockImplementation(mockSimpleResponse('should not reach'));

      const result = await spawnAgent({ task: 'test task' }, '/tmp');
      expect(result).toContain('Error');
      expect(result).toContain('Ollama is not running');
      expect(ollama.streamChat).not.toHaveBeenCalled();
    });

    test('agent proceeds when Ollama is healthy', async () => {
      ollama.streamChat.mockImplementation(mockSimpleResponse('Agent completed the task.'));

      const result = await spawnAgent({ task: 'list files' }, '/tmp');
      expect(result).toContain('Agent completed the task');
      expect(ollama.streamChat).toHaveBeenCalled();
    });
  });

  // ── Basic spawn and result ──

  describe('spawnAgent', () => {
    test('requires a task description', async () => {
      const result = await spawnAgent({}, '/tmp');
      expect(result).toContain('Error');
      expect(result).toContain('task');
    });

    test('creates a dedicated session for the agent', async () => {
      ollama.streamChat.mockImplementation(mockSimpleResponse('Research findings here.'));

      const statusEvents = [];
      await spawnAgent({ task: 'research project' }, '/tmp', (event) => {
        statusEvents.push(event);
      });

      const startEvent = statusEvents.find(e => e.type === 'agent_started');
      expect(startEvent).toBeDefined();
      expect(startEvent.id).toMatch(/^agent_/);
      expect(startEvent.sessionId).toBeTruthy();
      expect(startEvent.task).toBe('research project');

      const completeEvent = statusEvents.find(e => e.type === 'agent_completed');
      expect(completeEvent).toBeDefined();
      expect(completeEvent.duration).toBeGreaterThanOrEqual(0);
    });

    test('returns text response when model produces no tool calls', async () => {
      ollama.streamChat.mockImplementation(mockSimpleResponse('Here are 5 key findings about the topic.'));

      const result = await spawnAgent({ task: 'summarize findings' }, '/tmp');
      expect(result).toBe('Here are 5 key findings about the topic.');
    });

    test('executes tool calls and returns final summary', async () => {
      ollama.streamChat.mockImplementation(
        mockToolThenText(
          [{ function: { name: 'read_file', arguments: { path: '/tmp/test.txt' } } }],
          'The file contains test data.'
        )
      );

      const statusEvents = [];
      await spawnAgent({ task: 'read /tmp/test.txt' }, '/tmp', (event) => {
        statusEvents.push(event);
      });

      // Should have tried to execute the tool
      const toolEvent = statusEvents.find(e => e.type === 'agent_tool');
      expect(toolEvent).toBeDefined();
      expect(toolEvent.tool).toBe('read_file');

      // streamChat called at least twice: once for tool call, once for final response
      expect(ollama.streamChat.mock.calls.length).toBeGreaterThanOrEqual(2);
    });
  });

  // ── Timeout behavior ──

  describe('timeouts', () => {
    test('per-inference timeout rejects hung Ollama calls', async () => {
      // Mock a streamChat that never resolves
      ollama.streamChat.mockImplementation(() => new Promise(() => {}));

      const result = await spawnAgent({ task: 'should timeout' }, '/tmp');

      expect(result).toContain('Error');
      expect(result).toContain('timed out');
    }, 150000); // Jest timeout must exceed the 2-min inference timeout

    test('agent completes within max turns even with continuous tool calls', async () => {
      // Always return tool calls to keep looping
      ollama.streamChat.mockImplementation(() => Promise.resolve({
        response: '',
        toolCalls: [{ function: { name: 'bash', arguments: { command: 'echo hi' } } }],
        stats: {},
      }));

      const result = await spawnAgent({ task: 'infinite loop' }, '/tmp');

      // Agent should hit max turns (10), then try a summary (which also returns tool calls
      // but is called without tools, so it returns empty), then finish
      expect(ollama.streamChat.mock.calls.length).toBeGreaterThanOrEqual(10);
    }, 30000);
  });

  // ── spawnMulti (parallel agents) ──

  describe('spawnMulti', () => {
    test('requires non-empty tasks array', async () => {
      const result = await spawnMulti({}, '/tmp');
      expect(result).toContain('Error');

      const result2 = await spawnMulti({ tasks: [] }, '/tmp');
      expect(result2).toContain('Error');
    });

    test('rejects more than 5 tasks', async () => {
      const result = await spawnMulti({
        tasks: ['a', 'b', 'c', 'd', 'e', 'f'],
      }, '/tmp');
      expect(result).toContain('Error');
      expect(result).toContain('maximum 5');
    });

    test('runs multiple agents and collects results in order', async () => {
      let callNum = 0;
      ollama.streamChat.mockImplementation(() => {
        callNum++;
        return Promise.resolve({
          response: `Result for agent call ${callNum}`,
          toolCalls: null,
          stats: {},
        });
      });

      const result = await spawnMulti({
        tasks: ['find files', 'check status', 'read logs'],
      }, '/tmp');

      // Results should contain all 3 task headers in order
      expect(result).toContain('Task 1: find files');
      expect(result).toContain('Task 2: check status');
      expect(result).toContain('Task 3: read logs');
    });

    test('fires progress events for each agent', async () => {
      ollama.streamChat.mockImplementation(mockSimpleResponse('done'));

      const events = [];
      await spawnMulti({ tasks: ['a', 'b'] }, '/tmp', (event) => {
        events.push(event);
      });

      const progressEvent = events.find(e => e.type === 'agent_progress');
      expect(progressEvent).toBeDefined();
      expect(progressEvent.total).toBe(2);

      const startEvents = events.filter(e => e.type === 'agent_started');
      expect(startEvents.length).toBe(2);
    });
  });

  // ── listAgents ──

  describe('listAgents', () => {
    test('tracks spawned agents', async () => {
      ollama.streamChat.mockImplementation(mockSimpleResponse('done'));
      await spawnAgent({ task: 'tracked task' }, '/tmp');

      const list = listAgents();
      expect(list).toContain('tracked task');
      expect(list).toContain('completed');
    });
  });
});
