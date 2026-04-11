// PRE Web GUI — Ollama NDJSON streaming client
// Streams /api/chat responses and emits parsed events via callback

const http = require('http');
const { MODEL, MODEL_CTX, OLLAMA_PORT } = require('./constants');

/**
 * Send a chat request to Ollama and stream the response.
 * @param {Object} opts
 * @param {Array} opts.messages - Chat messages array
 * @param {Array} [opts.tools] - Tool definitions array
 * @param {number} [opts.maxTokens=8192] - Max tokens to generate
 * @param {Function} opts.onToken - Called with {type, content} for each event
 * @param {AbortSignal} [opts.signal] - Abort signal to cancel request
 * @returns {Promise<{response: string, toolCalls: Array|null, stats: Object}>}
 */
function streamChat({ messages, tools, maxTokens = 8192, onToken, signal }) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: MODEL,
      stream: true,
      keep_alive: '24h',
      options: {
        num_predict: maxTokens,
        num_ctx: MODEL_CTX,
      },
      messages,
      ...(tools && tools.length > 0 ? { tools } : {}),
    });

    const req = http.request({
      hostname: '127.0.0.1',
      port: OLLAMA_PORT,
      path: '/api/chat',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    }, (res) => {
      let response = '';
      let thinking = '';
      let toolCalls = null;
      let stats = {};
      let buffer = '';

      res.on('data', (chunk) => {
        buffer += chunk.toString();
        // Process complete NDJSON lines
        let nlIdx;
        while ((nlIdx = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, nlIdx).trim();
          buffer = buffer.slice(nlIdx + 1);
          if (!line) continue;

          let parsed;
          try { parsed = JSON.parse(line); } catch { continue; }

          // Extract message content
          const msg = parsed.message || {};

          // Thinking content
          if (msg.thinking) {
            thinking += msg.thinking;
            if (onToken) onToken({ type: 'thinking', content: msg.thinking });
          }

          // Regular content
          if (msg.content) {
            response += msg.content;
            if (onToken) onToken({ type: 'token', content: msg.content });
          }

          // Native tool calls
          if (msg.tool_calls && msg.tool_calls.length > 0) {
            toolCalls = msg.tool_calls;
            if (onToken) onToken({ type: 'tool_calls', calls: msg.tool_calls });
          }

          // Done message — extract stats
          if (parsed.done) {
            stats = {
              prompt_eval_count: parsed.prompt_eval_count || 0,
              eval_count: parsed.eval_count || 0,
              eval_duration: parsed.eval_duration || 0,
              prompt_eval_duration: parsed.prompt_eval_duration || 0,
              total_duration: parsed.total_duration || 0,
            };
            // Compute tok/s
            if (stats.eval_duration > 0) {
              stats.tok_s = (stats.eval_count / (stats.eval_duration / 1e9)).toFixed(1);
            }
            if (stats.prompt_eval_duration > 0) {
              stats.ttft_ms = (stats.prompt_eval_duration / 1e6).toFixed(0);
            }
          }
        }
      });

      res.on('end', () => {
        resolve({ response, thinking, toolCalls, stats });
      });

      res.on('error', reject);
    });

    req.on('error', reject);

    // Handle abort
    if (signal) {
      signal.addEventListener('abort', () => {
        req.destroy();
        reject(new Error('Request aborted'));
      });
    }

    req.write(body);
    req.end();
  });
}

/**
 * Check if Ollama is running
 */
function healthCheck() {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${OLLAMA_PORT}/v1/models`, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => resolve(true));
    });
    req.on('error', () => resolve(false));
    req.setTimeout(3000, () => { req.destroy(); resolve(false); });
  });
}

/**
 * Parse <tool_call> tags from model text output.
 * Fallback for when model outputs tool calls as text instead of native API.
 * Returns array of tool calls in Ollama format, or null if none found.
 */
function parseTextToolCalls(text) {
  if (!text) return null;
  const calls = [];
  let scan = text;
  let idx;

  while ((idx = scan.indexOf('<tool_call>')) !== -1) {
    scan = scan.slice(idx + 11); // skip past <tool_call>

    let body;
    const closeIdx = scan.indexOf('</tool_call>');
    if (closeIdx !== -1) {
      body = scan.slice(0, closeIdx).trim();
      scan = scan.slice(closeIdx + 12);
    } else {
      // No closing tag — try brace matching
      const braceStart = scan.indexOf('{');
      if (braceStart === -1) break;
      let depth = 0;
      let inStr = false;
      let end = -1;
      for (let i = braceStart; i < scan.length; i++) {
        const c = scan[i];
        if (inStr) {
          if (c === '\\') { i++; continue; }
          if (c === '"') inStr = false;
          continue;
        }
        if (c === '"') { inStr = true; continue; }
        if (c === '{') depth++;
        if (c === '}') { depth--; if (depth === 0) { end = i; break; } }
      }
      if (end === -1) break;
      body = scan.slice(braceStart, end + 1).trim();
      scan = scan.slice(end + 1);
    }

    // Parse the JSON
    try {
      const obj = JSON.parse(body);
      const name = obj.name;
      const args = obj.arguments || obj.parameters || obj.params || {};
      // If no nested args object, treat all non-name keys as args
      let finalArgs = args;
      if (typeof args === 'string') {
        try { finalArgs = JSON.parse(args); } catch { finalArgs = { input: args }; }
      } else if (args === obj.arguments && Object.keys(args).length === 0) {
        finalArgs = {};
        for (const [k, v] of Object.entries(obj)) {
          if (k !== 'name') finalArgs[k] = v;
        }
      }
      if (name) {
        calls.push({
          function: { name, arguments: finalArgs },
        });
      }
    } catch {
      // Try to find raw JSON without tags
      continue;
    }
  }

  // Fallback: raw JSON with "name" and "arguments" but no <tool_call> tags
  if (calls.length === 0 && text.includes('"name"') && (text.includes('"arguments"') || text.includes('"parameters"'))) {
    const jsonMatch = text.match(/\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*(?:"arguments"|"parameters"|"params")\s*:\s*\{[\s\S]*?\}\s*\}/);
    if (jsonMatch) {
      try {
        const obj = JSON.parse(jsonMatch[0]);
        if (obj.name) {
          calls.push({
            function: {
              name: obj.name,
              arguments: obj.arguments || obj.parameters || obj.params || {},
            },
          });
        }
      } catch {}
    }
  }

  return calls.length > 0 ? calls : null;
}

/**
 * Strip <tool_call> tags and their content from response text,
 * returning only the natural language portion.
 */
function stripToolCallText(text) {
  if (!text) return '';
  // Remove <tool_call>...</tool_call> blocks
  let cleaned = text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();
  // Remove unclosed <tool_call> blocks (model got cut off)
  const unclosedIdx = cleaned.indexOf('<tool_call>');
  if (unclosedIdx !== -1) {
    cleaned = cleaned.slice(0, unclosedIdx).trim();
  }
  return cleaned;
}

/**
 * Generate embeddings via Ollama /api/embed
 * Uses nomic-embed-text (274MB) — lightweight and already installed.
 * @param {string|string[]} input - Text(s) to embed
 * @returns {Promise<number[][]>} Array of embedding vectors
 */
const EMBED_MODEL = 'nomic-embed-text';

function embed(input) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: EMBED_MODEL,
      input: Array.isArray(input) ? input : [input],
    });

    const req = http.request({
      hostname: '127.0.0.1',
      port: OLLAMA_PORT,
      path: '/api/embed',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          resolve(parsed.embeddings || []);
        } catch (err) {
          reject(new Error(`Embed parse error: ${err.message}`));
        }
      });
    });

    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Embed request timed out')); });
    req.write(body);
    req.end();
  });
}

module.exports = { streamChat, healthCheck, parseTextToolCalls, stripToolCallText, embed };
