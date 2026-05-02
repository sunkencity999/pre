// Tests for Session Compression (compression.js)

jest.mock('../src/constants', () => {
  const path = require('path');
  const os = require('os');
  const fs = require('fs');
  const PRE_DIR = path.join(os.tmpdir(), '.pre-test-compression-' + process.pid);
  fs.mkdirSync(PRE_DIR, { recursive: true });
  fs.mkdirSync(path.join(PRE_DIR, 'sessions'), { recursive: true });
  fs.mkdirSync(path.join(PRE_DIR, 'memory'), { recursive: true });
  fs.mkdirSync(path.join(PRE_DIR, 'artifacts'), { recursive: true });
  return {
    PRE_DIR,
    MODEL: 'test-model',
    MODEL_CTX: 131072,  // Standard context window
    OLLAMA_PORT: 11434,
    MAX_TOOL_TURNS: 35,
    SESSIONS_DIR: path.join(PRE_DIR, 'sessions'),
    MEMORY_DIR: path.join(PRE_DIR, 'memory'),
    ARTIFACTS_DIR: path.join(PRE_DIR, 'artifacts'),
    CONNECTIONS_FILE: path.join(PRE_DIR, 'connections.json'),
    COMFYUI_FILE: path.join(PRE_DIR, 'comfyui.json'),
    CRON_FILE: path.join(PRE_DIR, 'cron.json'),
    RESEARCH_AGENT_MAX_TURNS: 30,
    RESEARCH_MAX_SECTIONS: 8,
    RESEARCH_MAX_SOURCES: 5,
  };
});
jest.mock('../src/ollama', () => ({
  streamChat: jest.fn().mockResolvedValue({ response: 'Compressed summary of conversation.' }),
}));

describe('compression — token estimation', () => {
  const compression = require('../src/compression');

  test('estimateTokens returns positive number for text', () => {
    const tokens = compression.estimateTokens('hello world');
    expect(tokens).toBeGreaterThan(0);
    expect(typeof tokens).toBe('number');
  });

  test('estimateTokens returns 0 for null/empty', () => {
    expect(compression.estimateTokens(null)).toBe(0);
    expect(compression.estimateTokens('')).toBe(0);
    expect(compression.estimateTokens(undefined)).toBe(0);
  });

  test('estimateTokens scales with text length', () => {
    const short = compression.estimateTokens('hello');
    const long = compression.estimateTokens('hello '.repeat(100));
    expect(long).toBeGreaterThan(short);
  });

  test('estimateMessageTokens handles message array', () => {
    const messages = [
      { role: 'user', content: 'Hello, how are you?' },
      { role: 'assistant', content: 'I am fine, thank you!' },
    ];
    const tokens = compression.estimateMessageTokens(messages);
    expect(tokens).toBeGreaterThan(0);
  });

  test('estimateMessageTokens accounts for tool_calls', () => {
    const withoutTools = [{ role: 'assistant', content: 'hello' }];
    const withTools = [{
      role: 'assistant',
      content: 'hello',
      tool_calls: [{ function: { name: 'bash', arguments: { command: 'echo test' } } }],
    }];
    const tokensWithout = compression.estimateMessageTokens(withoutTools);
    const tokensWith = compression.estimateMessageTokens(withTools);
    expect(tokensWith).toBeGreaterThan(tokensWithout);
  });

  test('estimateMessageTokens accounts for thinking', () => {
    const withoutThinking = [{ role: 'assistant', content: 'hello' }];
    const withThinking = [{
      role: 'assistant',
      content: 'hello',
      thinking: 'Let me think about this carefully...',
    }];
    const tokensWithout = compression.estimateMessageTokens(withoutThinking);
    const tokensWith = compression.estimateMessageTokens(withThinking);
    expect(tokensWith).toBeGreaterThan(tokensWithout);
  });
});

describe('compression — budget constants', () => {
  const compression = require('../src/compression');

  test('exports expected constants', () => {
    expect(compression.COMPRESSION_THRESHOLD).toBeGreaterThan(0);
    expect(compression.COMPRESSION_THRESHOLD).toBeLessThan(1);
    expect(compression.KEEP_RECENT_TURNS).toBeGreaterThanOrEqual(4);
    expect(compression.SYSTEM_PROMPT_TOKENS).toBeGreaterThan(0);
    expect(compression.OUTPUT_RESERVE).toBeGreaterThan(0);
    expect(compression.CORE_TOOL_TOKENS).toBeGreaterThan(0);
    expect(compression.TOOL_TOKENS_PER_DOMAIN).toBeGreaterThan(0);
  });
});

describe('compression — compressIfNeeded', () => {
  const compression = require('../src/compression');

  test('returns messages unchanged if under threshold', async () => {
    const messages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' },
    ];
    const result = await compression.compressIfNeeded('test-session', messages, 0);
    expect(result).toEqual(messages);
  });

  test('returns messages unchanged if too few messages', async () => {
    const messages = [
      { role: 'user', content: 'test' },
    ];
    const result = await compression.compressIfNeeded('test-short', messages, 0);
    expect(result).toEqual(messages);
  });

  test('compresses when over threshold', async () => {
    // Create a large message set that would exceed the budget
    const messages = [];
    for (let i = 0; i < 100; i++) {
      messages.push({
        role: i % 2 === 0 ? 'user' : 'assistant',
        content: 'A'.repeat(5000), // ~1400 tokens each, 100 messages = ~140K tokens
      });
    }
    const result = await compression.compressIfNeeded('test-compress', messages, 0);
    // Should have fewer messages than original
    expect(result.length).toBeLessThan(messages.length);
    // First message should be compressed context
    expect(result[0].content).toContain('[Compressed conversation context');
    // Recent messages preserved
    expect(result.length).toBeGreaterThanOrEqual(compression.KEEP_RECENT_TURNS + 1);
  });

  test('preserves recent turns during compression', async () => {
    const messages = [];
    for (let i = 0; i < 100; i++) {
      messages.push({
        role: i % 2 === 0 ? 'user' : 'assistant',
        content: `Message ${i}: ${'A'.repeat(5000)}`,
      });
    }
    const result = await compression.compressIfNeeded('test-preserve', messages, 0);
    // Last KEEP_RECENT_TURNS messages should be preserved
    const lastOriginal = messages.slice(-compression.KEEP_RECENT_TURNS);
    const lastResult = result.slice(-compression.KEEP_RECENT_TURNS);
    for (let i = 0; i < lastOriginal.length; i++) {
      expect(lastResult[i].content).toBe(lastOriginal[i].content);
    }
  });
});
