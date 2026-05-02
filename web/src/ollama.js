// PRE Web GUI — Ollama NDJSON streaming client
// Streams /api/chat responses and emits parsed events via callback

const http = require('http');
const { MODEL, MODEL_CTX, OLLAMA_PORT } = require('./constants');

// Persistent HTTP agent — reuses TCP connections to Ollama across requests.
// Eliminates ~5-50ms TCP handshake overhead per inference call.
const ollamaAgent = new http.Agent({
  keepAlive: true,
  keepAliveMsecs: 30000,
  maxSockets: 4, // Allow background tasks + main inference concurrently
});

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
function streamChat({ messages, tools, maxTokens = 8192, onToken, signal, think, extraOptions }) {
  return new Promise((resolve, reject) => {
    let aborted = false; // Shared across res and req error handlers

    const body = JSON.stringify({
      model: MODEL,
      stream: true,
      keep_alive: '24h',
      ...(think === false ? { think: false } : {}),
      options: {
        num_predict: maxTokens,
        num_ctx: MODEL_CTX,
        repeat_penalty: 1.1,
        repeat_last_n: 256,
        ...(extraOptions || {}),
      },
      messages,
      ...(tools && tools.length > 0 ? { tools } : {}),
    });

    const req = http.request({
      hostname: '127.0.0.1',
      port: OLLAMA_PORT,
      path: '/api/chat',
      method: 'POST',
      agent: ollamaAgent,
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

      // Repetition loop detector — watches the last N characters for
      // repeating patterns.  If a short phrase (8-60 chars) repeats
      // 6+ times consecutively we abort the request early.
      const REPEAT_WINDOW = 600; // chars to scan
      const REPEAT_THRESHOLD = 6; // how many repeats trigger abort
      let repeatCheckCounter = 0;
      let thinkRepeatCounter = 0;

      function detectRepetition(text) {
        if (text.length < REPEAT_WINDOW) return false;
        const tail = text.slice(-REPEAT_WINDOW);
        // Try pattern lengths from 8 to 60 characters
        for (let pLen = 8; pLen <= 60; pLen++) {
          const pattern = tail.slice(-pLen);
          let count = 0;
          let pos = tail.length - pLen;
          while (pos >= 0) {
            if (tail.slice(pos, pos + pLen) === pattern) {
              count++;
              pos -= pLen;
            } else {
              break;
            }
          }
          if (count >= REPEAT_THRESHOLD) return pattern;
        }
        return false;
      }

      res.on('data', (chunk) => {
        if (aborted) return;
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

            // Check for repetition in thinking every ~50 tokens
            thinkRepeatCounter++;
            if (thinkRepeatCounter >= 50) {
              thinkRepeatCounter = 0;
              const repeatedPattern = detectRepetition(thinking);
              if (repeatedPattern) {
                console.warn(`[ollama] Thinking loop detected: "${repeatedPattern.slice(0, 40)}..." — aborting generation`);
                aborted = true;
                req.destroy();
                // Trim the repeated tail from thinking
                const cleanIdx = thinking.lastIndexOf(repeatedPattern);
                if (cleanIdx > 100) {
                  let trimPos = cleanIdx;
                  while (trimPos > 0 && thinking.slice(trimPos - repeatedPattern.length, trimPos) === repeatedPattern) {
                    trimPos -= repeatedPattern.length;
                  }
                  thinking = thinking.slice(0, trimPos).trimEnd();
                }
                // If the model already produced a response, keep it; otherwise flag it
                if (!response.trim()) {
                  response = '*(Generation stopped — thinking loop detected)*';
                  if (onToken) onToken({ type: 'token', content: response });
                }
                resolve({ response, thinking, toolCalls, stats });
                return;
              }
            }
          }

          // Regular content
          if (msg.content) {
            response += msg.content;
            if (onToken) onToken({ type: 'token', content: msg.content });

            // Check for repetition every ~50 tokens (don't scan every token)
            repeatCheckCounter++;
            if (repeatCheckCounter >= 50) {
              repeatCheckCounter = 0;
              const repeatedPattern = detectRepetition(response);
              if (repeatedPattern) {
                console.warn(`[ollama] Repetition loop detected: "${repeatedPattern.slice(0, 40)}..." — aborting generation`);
                aborted = true;
                req.destroy();
                // Trim the repeated tail from the response
                const cleanIdx = response.lastIndexOf(repeatedPattern);
                if (cleanIdx > 100) {
                  // Find where the repetition started by walking backwards
                  let trimPos = cleanIdx;
                  while (trimPos > 0 && response.slice(trimPos - repeatedPattern.length, trimPos) === repeatedPattern) {
                    trimPos -= repeatedPattern.length;
                  }
                  response = response.slice(0, trimPos).trimEnd();
                  response += '\n\n*(Generation stopped — repetitive output detected)*';
                }
                if (onToken) onToken({ type: 'token', content: '\n\n*(Generation stopped — repetitive output detected)*' });
                resolve({ response, thinking, toolCalls, stats });
                return;
              }
            }
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
        if (!aborted) resolve({ response, thinking, toolCalls, stats });
      });

      res.on('error', (err) => { if (!aborted) reject(err); });
    });

    req.on('error', (err) => { if (!aborted) reject(err); });

    // Handle abort
    if (signal) {
      if (signal.aborted) {
        aborted = true;
        req.destroy();
        reject(new Error('Request aborted'));
        return;
      }
      signal.addEventListener('abort', () => {
        aborted = true;
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
    const req = http.get({ hostname: '127.0.0.1', port: OLLAMA_PORT, path: '/v1/models', agent: ollamaAgent }, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => resolve(true));
    });
    req.on('error', () => resolve(false));
    req.setTimeout(3000, () => { req.destroy(); resolve(false); });
  });
}

/**
 * Repair common JSON issues from model output.
 * Models often output literal newlines, tabs, and unescaped control chars
 * inside JSON string values instead of proper escape sequences.
 */
function repairJSON(str) {
  let result = '';
  let inString = false;
  let escaped = false;

  for (let i = 0; i < str.length; i++) {
    const ch = str[i];

    if (escaped) {
      result += ch;
      escaped = false;
      continue;
    }

    if (ch === '\\' && inString) {
      result += ch;
      escaped = true;
      continue;
    }

    if (ch === '"') {
      inString = !inString;
      result += ch;
      continue;
    }

    if (inString) {
      if (ch === '\n') { result += '\\n'; continue; }
      if (ch === '\r') { result += '\\r'; continue; }
      if (ch === '\t') { result += '\\t'; continue; }
    }

    result += ch;
  }

  return result;
}

/**
 * Fix Python-style triple-quoted strings in JSON.
 * Models sometimes output """...""" for multiline string values.
 */
function fixTripleQuotes(str) {
  if (!str.includes('"""')) return str;
  return str.replace(/"""([\s\S]*?)"""/g, (_match, inner) => {
    const escaped = inner
      .replace(/\\/g, '\\\\')
      .replace(/"/g, '\\"')
      .replace(/\n/g, '\\n')
      .replace(/\r/g, '\\r')
      .replace(/\t/g, '\\t');
    return '"' + escaped + '"';
  });
}

/**
 * Try to parse JSON, falling back to repaired JSON on failure.
 */
function safeParseJSON(str) {
  try {
    return JSON.parse(str);
  } catch {
    // Try repairing literal newlines/tabs in string values
    try {
      return JSON.parse(repairJSON(str));
    } catch {
      // Try fixing triple quotes (Python-style multiline) + newlines
      try {
        return JSON.parse(repairJSON(fixTripleQuotes(str)));
      } catch (err) {
        console.log(`[tool-parse] JSON repair failed: ${err.message.slice(0, 100)}`);
        return null;
      }
    }
  }
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

    // Parse the JSON (with repair fallback for literal newlines in strings)
    const obj = safeParseJSON(body);
    if (obj) {
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
    }
  }

  // Fallback: raw JSON with "name" and "arguments" but no <tool_call> tags
  if (calls.length === 0 && text.includes('"name"') && (text.includes('"arguments"') || text.includes('"parameters"'))) {
    const jsonMatch = text.match(/\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*(?:"arguments"|"parameters"|"params")\s*:\s*\{[\s\S]*?\}\s*\}/);
    if (jsonMatch) {
      const obj = safeParseJSON(jsonMatch[0]);
      if (obj && obj.name) {
        calls.push({
          function: {
            name: obj.name,
            arguments: obj.arguments || obj.parameters || obj.params || {},
          },
        });
      }
    }
  }

  // Fallback: Claude-style <artifact> tags (model trained on Claude outputs)
  // Convert <artifact title="..." type="html">...content...</artifact> into artifact tool calls
  if (calls.length === 0 && text.includes('<artifact')) {
    const artifactMatch = text.match(/<artifact\s+([^>]*)>([\s\S]*?)<\/artifact>/);
    if (artifactMatch) {
      const attrs = artifactMatch[1];
      const content = artifactMatch[2].trim();
      const titleMatch = attrs.match(/title\s*=\s*"([^"]*)"/);
      const typeMatch = attrs.match(/type\s*=\s*"([^"]*)"/);
      const title = titleMatch ? titleMatch[1] : 'Untitled';
      const type = typeMatch ? typeMatch[1] : 'html';
      console.log(`[tool-parse] Converted <artifact> tag → artifact tool call: "${title}"`);
      calls.push({
        function: {
          name: 'artifact',
          arguments: { title, content, type },
        },
      });
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
  // Remove Claude-style <artifact>...</artifact> blocks
  cleaned = cleaned.replace(/<artifact\s[^>]*>[\s\S]*?<\/artifact>/g, '').trim();
  const unclosedArt = cleaned.indexOf('<artifact');
  if (unclosedArt !== -1) {
    cleaned = cleaned.slice(0, unclosedArt).trim();
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
      agent: ollamaAgent,
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

module.exports = { streamChat, healthCheck, parseTextToolCalls, stripToolCallText, embed, safeParseJSON };
