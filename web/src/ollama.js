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

module.exports = { streamChat, healthCheck };
