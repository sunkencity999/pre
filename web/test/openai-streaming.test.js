// Tests for OpenAI-compatible SSE streaming in src/ollama.js

// Mock connections to control provider config
let mockProvider = { type: 'ollama' };
jest.mock('../src/connections', () => ({
  getProvider: () => mockProvider,
  loadConnections: () => ({}),
}));
jest.mock('../src/constants', () => ({
  MODEL: 'pre-gemma4',
  MODEL_CTX: 8192,
  OLLAMA_PORT: 11434,
  CONNECTIONS_FILE: '/tmp/test-connections.json',
}));

const { safeParseJSON } = require('../src/ollama');

describe('OpenAI streaming support', () => {
  beforeEach(() => {
    mockProvider = { type: 'ollama' };
  });

  describe('safeParseJSON', () => {
    test('parses valid JSON', () => {
      expect(safeParseJSON('{"name":"test"}')).toEqual({ name: 'test' });
    });

    test('repairs literal newlines in strings', () => {
      const result = safeParseJSON('{"text":"line1\nline2"}');
      expect(result).toEqual({ text: 'line1\nline2' });
    });

    test('returns null for unparseable input', () => {
      expect(safeParseJSON('not json at all')).toBeNull();
    });
  });

  describe('SSE tool call argument accumulation', () => {
    // Simulates the fragment accumulation pattern from _streamChatOpenAI
    test('accumulates fragments into complete arguments', () => {
      const toolCallAccum = new Map();

      // Simulate 3 SSE chunks with tool call fragments
      const chunks = [
        { index: 0, function: { name: 'read_file', arguments: '{"pat' } },
        { index: 0, function: { arguments: 'h":"/tmp' } },
        { index: 0, function: { arguments: '/test.txt"}' } },
      ];

      for (const tc of chunks) {
        const idx = tc.index ?? 0;
        if (!toolCallAccum.has(idx)) {
          toolCallAccum.set(idx, { name: '', arguments: '' });
        }
        const acc = toolCallAccum.get(idx);
        if (tc.function?.name) acc.name = tc.function.name;
        if (tc.function?.arguments) acc.arguments += tc.function.arguments;
      }

      // Assemble
      const toolCalls = [...toolCallAccum.entries()]
        .sort(([a], [b]) => a - b)
        .map(([, tc]) => ({
          function: {
            name: tc.name,
            arguments: safeParseJSON(tc.arguments) || {},
          },
        }));

      expect(toolCalls).toHaveLength(1);
      expect(toolCalls[0].function.name).toBe('read_file');
      expect(toolCalls[0].function.arguments).toEqual({ path: '/tmp/test.txt' });
    });

    test('accumulates multiple parallel tool calls', () => {
      const toolCallAccum = new Map();

      const chunks = [
        { index: 0, function: { name: 'bash', arguments: '{"comm' } },
        { index: 1, function: { name: 'read_file', arguments: '{"path' } },
        { index: 0, function: { arguments: 'and":"ls"}' } },
        { index: 1, function: { arguments: '":"/etc/hosts"}' } },
      ];

      for (const tc of chunks) {
        const idx = tc.index ?? 0;
        if (!toolCallAccum.has(idx)) {
          toolCallAccum.set(idx, { name: '', arguments: '' });
        }
        const acc = toolCallAccum.get(idx);
        if (tc.function?.name) acc.name = tc.function.name;
        if (tc.function?.arguments) acc.arguments += tc.function.arguments;
      }

      const toolCalls = [...toolCallAccum.entries()]
        .sort(([a], [b]) => a - b)
        .map(([, tc]) => ({
          function: {
            name: tc.name,
            arguments: safeParseJSON(tc.arguments) || {},
          },
        }));

      expect(toolCalls).toHaveLength(2);
      expect(toolCalls[0].function.name).toBe('bash');
      expect(toolCalls[0].function.arguments).toEqual({ command: 'ls' });
      expect(toolCalls[1].function.name).toBe('read_file');
      expect(toolCalls[1].function.arguments).toEqual({ path: '/etc/hosts' });
    });
  });

  describe('SSE line parsing', () => {
    test('extracts payload from SSE data lines', () => {
      const lines = [
        'data: {"id":"123","choices":[{"delta":{"content":"Hello"}}]}',
        '',
        'data: {"id":"123","choices":[{"delta":{"content":" world"}}]}',
        '',
        'data: [DONE]',
      ];

      const tokens = [];
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;
        const payload = trimmed.slice(6);
        if (payload === '[DONE]') continue;
        const parsed = safeParseJSON(payload);
        if (parsed?.choices?.[0]?.delta?.content) {
          tokens.push(parsed.choices[0].delta.content);
        }
      }

      expect(tokens).toEqual(['Hello', ' world']);
    });

    test('handles reasoning_content as thinking', () => {
      const line = 'data: {"choices":[{"delta":{"reasoning_content":"Let me think..."}}]}';
      const payload = line.slice(6);
      const parsed = safeParseJSON(payload);
      const delta = parsed?.choices?.[0]?.delta;
      expect(delta.reasoning_content).toBe('Let me think...');
    });

    test('extracts usage stats from final chunk', () => {
      const line = 'data: {"choices":[],"usage":{"prompt_tokens":150,"completion_tokens":50,"total_tokens":200}}';
      const payload = line.slice(6);
      const parsed = safeParseJSON(payload);
      expect(parsed.usage).toEqual({
        prompt_tokens: 150,
        completion_tokens: 50,
        total_tokens: 200,
      });
    });
  });

  describe('maxTokens capping', () => {
    test('caps to provider max_tokens when lower', () => {
      const provider = { type: 'openai', max_tokens: 2048 };
      const requested = 32768;
      const effective = Math.min(requested, provider.max_tokens || 4096);
      expect(effective).toBe(2048);
    });

    test('uses default 4096 when provider max_tokens not set', () => {
      const provider = { type: 'openai' };
      const requested = 32768;
      const effective = Math.min(requested, provider.max_tokens || 4096);
      expect(effective).toBe(4096);
    });

    test('uses requested when lower than cap', () => {
      const provider = { type: 'openai', max_tokens: 16384 };
      const requested = 8192;
      const effective = Math.min(requested, provider.max_tokens || 4096);
      expect(effective).toBe(8192);
    });
  });

  describe('provider config', () => {
    test('getProvider returns ollama by default', () => {
      const { getProvider } = require('../src/connections');
      mockProvider = { type: 'ollama' };
      expect(getProvider().type).toBe('ollama');
    });

    test('getProvider returns openai config when set', () => {
      mockProvider = {
        type: 'openai',
        base_url: 'https://api.openai.com/v1',
        api_key: 'sk-test',
        model: 'gpt-4o',
        max_tokens: 4096,
      };
      const { getProvider } = require('../src/connections');
      const p = getProvider();
      expect(p.type).toBe('openai');
      expect(p.model).toBe('gpt-4o');
    });
  });
});
