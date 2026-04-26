// Tests for src/ollama.js — streamChat abort handling, error scoping, repetition detection
const http = require('http');

// Do NOT mock constants for ollama tests — we need the real MODULE/PORT values
// but we DO need to avoid hitting a real Ollama instance, so we spin up a
// local HTTP server that mimics Ollama's NDJSON streaming.

let mockServer;
let mockPort;

beforeAll((done) => {
  mockServer = http.createServer();
  mockServer.listen(0, '127.0.0.1', () => {
    mockPort = mockServer.address().port;
    done();
  });
});

afterAll((done) => {
  mockServer.close(done);
});

// Dynamically require ollama after we know the port — we need to override
// the OLLAMA_PORT constant before the module reads it.
let streamChat;

beforeAll(() => {
  // Override constants to point at our mock server
  jest.resetModules();
  jest.mock('../src/constants', () => ({
    MODEL: 'test-model',
    MODEL_CTX: 8192,
    OLLAMA_PORT: mockPort,
    PRE_DIR: '/tmp/pre-test-ollama',
    MAX_TOOL_TURNS: 35,
  }));
  const ollama = require('../src/ollama');
  streamChat = ollama.streamChat;
});

/**
 * Helper: configure the mock server to respond with NDJSON chunks
 * @param {Array<object>} chunks - array of JSON objects to send as NDJSON lines
 * @param {number} [delayMs=0] - delay between chunks
 */
function setMockResponse(chunks, delayMs = 0) {
  mockServer.removeAllListeners('request');
  mockServer.on('request', (req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
    let i = 0;
    function sendNext() {
      if (i >= chunks.length) {
        res.end();
        return;
      }
      res.write(JSON.stringify(chunks[i]) + '\n');
      i++;
      if (delayMs > 0) {
        setTimeout(sendNext, delayMs);
      } else {
        sendNext();
      }
    }
    sendNext();
  });
}

/**
 * Helper: configure mock server to hang (never respond) — for timeout/abort testing
 */
function setMockHang() {
  mockServer.removeAllListeners('request');
  mockServer.on('request', (req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
    // Write one chunk then hang
    res.write(JSON.stringify({ message: { content: 'start...' } }) + '\n');
    // Never send done or end — simulates a stuck model
  });
}

describe('ollama streamChat', () => {
  // ── Basic streaming ──

  test('streams tokens and returns complete response', async () => {
    setMockResponse([
      { message: { content: 'Hello' } },
      { message: { content: ' world' } },
      { done: true, eval_count: 2, eval_duration: 1e9, prompt_eval_duration: 5e8 },
    ]);

    const tokens = [];
    const result = await streamChat({
      messages: [{ role: 'user', content: 'hi' }],
      onToken: (event) => tokens.push(event),
    });

    expect(result.response).toBe('Hello world');
    expect(result.stats.eval_count).toBe(2);
    expect(tokens.length).toBeGreaterThanOrEqual(2);
  });

  test('handles thinking tokens separately', async () => {
    setMockResponse([
      { message: { thinking: 'Let me think...' } },
      { message: { content: 'The answer is 42.' } },
      { done: true, eval_count: 10 },
    ]);

    const result = await streamChat({
      messages: [{ role: 'user', content: 'meaning of life' }],
      onToken: () => {},
    });

    expect(result.thinking).toBe('Let me think...');
    expect(result.response).toBe('The answer is 42.');
  });

  test('extracts native tool calls', async () => {
    setMockResponse([
      { message: { tool_calls: [{ function: { name: 'bash', arguments: { command: 'ls' } } }] } },
      { done: true },
    ]);

    const result = await streamChat({
      messages: [{ role: 'user', content: 'list files' }],
      onToken: () => {},
    });

    expect(result.toolCalls).toBeDefined();
    expect(result.toolCalls[0].function.name).toBe('bash');
  });

  // ── Abort signal handling ──

  describe('abort signal', () => {
    test('aborts a hanging request when signal fires', async () => {
      setMockHang();

      const controller = new AbortController();

      // Fire abort after 100ms
      setTimeout(() => controller.abort(), 100);

      await expect(
        streamChat({
          messages: [{ role: 'user', content: 'test' }],
          signal: controller.signal,
          onToken: () => {},
        })
      ).rejects.toThrow('Request aborted');
    });

    test('aborts mid-stream and rejects cleanly', async () => {
      // Server sends tokens slowly
      setMockResponse([
        { message: { content: 'token1 ' } },
        { message: { content: 'token2 ' } },
        { message: { content: 'token3 ' } },
        { message: { content: 'token4 ' } },
        { done: true },
      ], 50);

      const controller = new AbortController();
      const tokens = [];

      // Abort after receiving some tokens
      setTimeout(() => controller.abort(), 80);

      await expect(
        streamChat({
          messages: [{ role: 'user', content: 'test' }],
          signal: controller.signal,
          onToken: (event) => {
            if (event.type === 'token') tokens.push(event.content);
          },
        })
      ).rejects.toThrow('Request aborted');

      // Should have received some but not all tokens
      expect(tokens.length).toBeGreaterThan(0);
      expect(tokens.length).toBeLessThan(4);
    });

    test('pre-aborted signal rejects immediately', async () => {
      setMockResponse([
        { message: { content: 'should not see this' } },
        { done: true },
      ]);

      const controller = new AbortController();
      controller.abort(); // Already aborted

      await expect(
        streamChat({
          messages: [{ role: 'user', content: 'test' }],
          signal: controller.signal,
          onToken: () => {},
        })
      ).rejects.toThrow();
    });
  });

  // ── Error handling ──

  describe('error handling', () => {
    test('handles server connection refused', async () => {
      // Stop the mock server temporarily so the port refuses connections
      await new Promise((resolve) => mockServer.close(resolve));

      await expect(
        streamChat({
          messages: [{ role: 'user', content: 'hi' }],
          onToken: () => {},
        })
      ).rejects.toThrow();

      // Restart the mock server on the same port
      await new Promise((resolve) => {
        mockServer = http.createServer();
        mockServer.listen(mockPort, '127.0.0.1', resolve);
      });
    });

    test('handles malformed JSON in stream gracefully', async () => {
      mockServer.removeAllListeners('request');
      mockServer.on('request', (req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
        res.write('{"message":{"content":"ok"}}\n');
        res.write('NOT VALID JSON\n');
        res.write('{"done":true}\n');
        res.end();
      });

      const result = await streamChat({
        messages: [{ role: 'user', content: 'test' }],
        onToken: () => {},
      });

      // Should recover and return what it got
      expect(result.response).toBe('ok');
    });
  });

  // ── Repetition detection ──

  describe('repetition detection', () => {
    test('detects and aborts repetitive output', async () => {
      // Generate a response with a repeating pattern.
      // repeatCheckCounter fires every 50 chunks, and the window needs 600+ chars,
      // so we need enough chunks to cross both thresholds.
      const repeatingChunk = 'the quick brown fox ';
      const chunks = [];
      for (let i = 0; i < 120; i++) {
        chunks.push({ message: { content: repeatingChunk } });
      }
      chunks.push({ done: true });

      setMockResponse(chunks);

      const result = await streamChat({
        messages: [{ role: 'user', content: 'test' }],
        onToken: () => {},
      });

      // Should have detected repetition and stopped early
      expect(result.response).toContain('repetitive output detected');
    });
  });
});
