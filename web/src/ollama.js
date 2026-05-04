// PRE Web GUI — Ollama NDJSON streaming client
// Streams /api/chat responses and emits parsed events via callback

const http = require('http');
const https = require('https');
const { MODEL, MODEL_CTX, OLLAMA_PORT } = require('./constants');
const { getProvider } = require('./connections');

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
function _streamChatOllama({ messages, tools, maxTokens = 8192, onToken, signal, think, extraOptions }) {
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
 * Stream a chat request to an OpenAI-compatible API via SSE.
 * Returns the same { response, toolCalls, stats } shape as the Ollama path.
 */
function _streamChatOpenAI({ messages, tools, maxTokens = 8192, onToken, signal, provider }) {
  return new Promise((resolve, reject) => {
    let aborted = false;

    const effectiveMax = Math.min(maxTokens, provider.max_tokens || 4096);

    // Build request body — OpenAI / Azure chat completions format
    const isAzure = provider.type === 'azure';
    // Azure newer models (GPT 5.5+, o-series) require max_completion_tokens;
    // standard OpenAI-compatible APIs use max_tokens.
    const tokenParam = isAzure
      ? { max_completion_tokens: effectiveMax }
      : { max_tokens: effectiveMax };

    // Normalize messages for OpenAI/Azure compliance:
    // - Assistant messages with tool_calls need type:"function" and id on each call
    // - Tool result messages need tool_call_id matching the call they respond to
    // - PRE stores combined tool results as a single role:"tool" message; split them
    const normalizedMessages = [];
    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      if (msg.role === 'assistant' && msg.tool_calls) {
        // Ensure each tool_call has type and id
        const fixedCalls = msg.tool_calls.map((tc, idx) => ({
          id: tc.id || `call_${i}_${idx}`,
          type: tc.type || 'function',
          function: {
            name: tc.function?.name || '',
            arguments: typeof tc.function?.arguments === 'string'
              ? tc.function.arguments
              : JSON.stringify(tc.function?.arguments || {}),
          },
        }));
        normalizedMessages.push({ ...msg, tool_calls: fixedCalls });
      } else if (msg.role === 'tool') {
        // Find the preceding assistant message's tool_calls to match IDs
        let prevAssistant = null;
        for (let j = i - 1; j >= 0; j--) {
          if (messages[j].role === 'assistant' && messages[j].tool_calls) {
            prevAssistant = normalizedMessages.find(
              (m, mi) => mi >= j && m.role === 'assistant' && m.tool_calls
            ) || messages[j];
            break;
          }
        }
        if (prevAssistant && prevAssistant.tool_calls) {
          // Split combined tool results into individual messages, one per tool_call
          for (const tc of prevAssistant.tool_calls) {
            const callId = tc.id || `call_${i}_0`;
            const toolName = tc.function?.name || '';
            // Extract the matching <tool_response> block if available
            const tagRegex = new RegExp(`<tool_response name="${toolName}">\\n?([\\s\\S]*?)</tool_response>`);
            const match = (msg.content || '').match(tagRegex);
            normalizedMessages.push({
              role: 'tool',
              tool_call_id: callId,
              content: match ? match[1] : (msg.content || ''),
            });
          }
        } else {
          // Fallback: pass through with a synthetic ID
          normalizedMessages.push({ ...msg, tool_call_id: msg.tool_call_id || `call_${i}_0` });
        }
      } else {
        normalizedMessages.push(msg);
      }
    }

    const bodyObj = {
      // Azure uses the deployment name in the URL, not a model field
      ...(isAzure ? {} : { model: provider.model }),
      stream: true,
      stream_options: { include_usage: true },
      ...tokenParam,
      messages: normalizedMessages,
      ...(tools && tools.length > 0 ? { tools } : {}),
    };
    const body = JSON.stringify(bodyObj);

    // Azure: base_url already points to the deployment endpoint; append api-version
    // OpenAI: append /chat/completions to the base URL
    let endpoint;
    if (isAzure) {
      const sep = provider.base_url.includes('?') ? '&' : '?';
      endpoint = provider.base_url + sep + 'api-version=' + (provider.api_version || '2024-10-21');
    } else {
      endpoint = provider.base_url + '/chat/completions';
    }
    const url = new URL(endpoint);
    const useHttps = url.protocol === 'https:';
    const lib = useHttps ? https : http;

    // Azure uses api-key header; OpenAI uses Authorization: Bearer
    const authHeaders = isAzure
      ? (provider.api_key ? { 'api-key': provider.api_key } : {})
      : (provider.api_key ? { 'Authorization': `Bearer ${provider.api_key}` } : {});

    const req = lib.request({
      hostname: url.hostname,
      port: url.port || (useHttps ? 443 : 80),
      path: url.pathname + url.search,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
        ...authHeaders,
      },
    }, (res) => {
      // Handle non-200 responses
      if (res.statusCode && res.statusCode >= 400) {
        let errData = '';
        res.on('data', (chunk) => errData += chunk);
        res.on('end', () => {
          let msg = `HTTP ${res.statusCode}`;
          try {
            const parsed = JSON.parse(errData);
            msg = parsed.error?.message || parsed.error?.type || msg;
          } catch {}
          reject(new Error(`Remote API error: ${msg}`));
        });
        return;
      }

      let response = '';
      let thinking = '';
      let stats = {};
      let buffer = '';

      // Tool call argument accumulator — OpenAI streams fragments
      const toolCallAccum = new Map(); // index → { name, arguments }

      // Repetition detection (same logic as Ollama path)
      const REPEAT_WINDOW = 600;
      const REPEAT_THRESHOLD = 6;
      let repeatCheckCounter = 0;

      function detectRepetition(text) {
        if (text.length < REPEAT_WINDOW) return false;
        const tail = text.slice(-REPEAT_WINDOW);
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

        // Process SSE lines: "data: {...}\n\n"
        let lineEnd;
        while ((lineEnd = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, lineEnd).trim();
          buffer = buffer.slice(lineEnd + 1);

          if (!line) continue;
          if (!line.startsWith('data: ')) continue;

          const payload = line.slice(6); // strip "data: "
          if (payload === '[DONE]') continue;

          let parsed;
          try { parsed = JSON.parse(payload); } catch { continue; }

          const choice = parsed.choices?.[0];
          if (!choice) {
            // Final chunk may have usage without choices
            if (parsed.usage) {
              stats = {
                prompt_eval_count: parsed.usage.prompt_tokens || 0,
                eval_count: parsed.usage.completion_tokens || 0,
                total_tokens: parsed.usage.total_tokens || 0,
              };
            }
            continue;
          }

          const delta = choice.delta || {};

          // Thinking / reasoning content (DeepSeek, some providers)
          if (delta.reasoning_content) {
            thinking += delta.reasoning_content;
            if (onToken) onToken({ type: 'thinking', content: delta.reasoning_content });
          }

          // Regular content
          if (delta.content) {
            response += delta.content;
            if (onToken) onToken({ type: 'token', content: delta.content });

            // Repetition detection every ~50 tokens
            repeatCheckCounter++;
            if (repeatCheckCounter >= 50) {
              repeatCheckCounter = 0;
              const repeatedPattern = detectRepetition(response);
              if (repeatedPattern) {
                console.warn(`[openai] Repetition loop detected: "${repeatedPattern.slice(0, 40)}..." — aborting`);
                aborted = true;
                req.destroy();
                const cleanIdx = response.lastIndexOf(repeatedPattern);
                if (cleanIdx > 100) {
                  let trimPos = cleanIdx;
                  while (trimPos > 0 && response.slice(trimPos - repeatedPattern.length, trimPos) === repeatedPattern) {
                    trimPos -= repeatedPattern.length;
                  }
                  response = response.slice(0, trimPos).trimEnd();
                  response += '\n\n*(Generation stopped — repetitive output detected)*';
                }
                if (onToken) onToken({ type: 'token', content: '\n\n*(Generation stopped — repetitive output detected)*' });
                resolve({ response, thinking, toolCalls: null, stats });
                return;
              }
            }
          }

          // Tool call fragments — accumulate across chunks
          if (delta.tool_calls) {
            for (const tc of delta.tool_calls) {
              const idx = tc.index ?? 0;
              if (!toolCallAccum.has(idx)) {
                toolCallAccum.set(idx, { id: '', name: '', arguments: '' });
              }
              const acc = toolCallAccum.get(idx);
              if (tc.id) acc.id = tc.id;
              if (tc.function?.name) acc.name = tc.function.name;
              if (tc.function?.arguments) acc.arguments += tc.function.arguments;
            }
          }

          // Usage in the final chunk (some providers include it per-choice)
          if (parsed.usage) {
            stats = {
              prompt_eval_count: parsed.usage.prompt_tokens || 0,
              eval_count: parsed.usage.completion_tokens || 0,
              total_tokens: parsed.usage.total_tokens || 0,
            };
          }
        }
      });

      res.on('end', () => {
        if (aborted) return;

        // Assemble accumulated tool calls
        let toolCalls = null;
        if (toolCallAccum.size > 0) {
          toolCalls = [...toolCallAccum.entries()]
            .sort(([a], [b]) => a - b)
            .map(([, tc]) => ({
              id: tc.id || `call_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
              type: 'function',
              function: {
                name: tc.name,
                arguments: safeParseJSON(tc.arguments) || {},
              },
            }));
          if (onToken) onToken({ type: 'tool_calls', calls: toolCalls });
        }

        // If no native tool calls, try text-based fallback
        if (!toolCalls && response) {
          toolCalls = parseTextToolCalls(response);
          if (toolCalls) {
            response = stripToolCallText(response);
          }
        }

        resolve({ response, thinking, toolCalls, stats });
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
 * Stream a chat request to an Anthropic Messages API endpoint (Azure AI Foundry or direct).
 * Handles the event-based SSE format: message_start, content_block_delta, message_delta, etc.
 * Returns the same { response, toolCalls, stats } shape as the other paths.
 */
function _streamChatAnthropic({ messages, tools, maxTokens = 8192, onToken, signal, provider }) {
  return new Promise((resolve, reject) => {
    let aborted = false;
    const effectiveMax = Math.min(maxTokens, provider.max_tokens || 4096);

    // Convert OpenAI-format messages to Anthropic format:
    // - Extract system message into top-level `system` field
    // - Anthropic messages array contains only user/assistant roles
    let systemPrompt = '';
    const anthropicMessages = [];
    for (const msg of messages) {
      if (msg.role === 'system') {
        systemPrompt += (systemPrompt ? '\n\n' : '') + (typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content));
      } else if (msg.role === 'user' || msg.role === 'assistant') {
        // Convert tool results from OpenAI format to Anthropic format
        if (msg.role === 'assistant' && msg.tool_calls) {
          const content = [];
          if (msg.content) content.push({ type: 'text', text: msg.content });
          for (const tc of msg.tool_calls) {
            content.push({
              type: 'tool_use',
              id: tc.id || `toolu_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
              name: tc.function?.name || 'unknown',
              input: tc.function?.arguments || {},
            });
          }
          anthropicMessages.push({ role: 'assistant', content });
        } else {
          anthropicMessages.push({ role: msg.role, content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content || '') });
        }
      } else if (msg.role === 'tool') {
        // Tool results — Anthropic expects these as user messages with tool_result content blocks.
        // Each tool_result needs a tool_use_id matching a tool_use block in the previous assistant message.
        const toolResultBlocks = [];

        if (msg._tool_results && Array.isArray(msg._tool_results)) {
          // Preferred: use structured per-tool results with preserved IDs
          for (const tr of msg._tool_results) {
            if (tr.tool_call_id) {
              toolResultBlocks.push({ type: 'tool_result', tool_use_id: tr.tool_call_id, content: tr.output || '' });
            }
          }
        }

        // Fallback: match tool results to preceding assistant tool_use blocks by name
        if (toolResultBlocks.length === 0) {
          let prevAssistant = null;
          for (let j = anthropicMessages.length - 1; j >= 0; j--) {
            if (anthropicMessages[j].role === 'assistant' && Array.isArray(anthropicMessages[j].content)) {
              prevAssistant = anthropicMessages[j];
              break;
            }
          }
          if (prevAssistant) {
            const toolUseBlocks = prevAssistant.content.filter(b => b.type === 'tool_use');
            for (const tu of toolUseBlocks) {
              // Try to extract matching output from combined content
              const tagRegex = new RegExp(`<tool_response name="${tu.name}">\\n?([\\s\\S]*?)</tool_response>`);
              const match = (msg.content || '').match(tagRegex);
              toolResultBlocks.push({
                type: 'tool_result',
                tool_use_id: tu.id,
                content: match ? match[1] : (msg.content || ''),
              });
            }
          }
          // Last resort: use whatever ID we have
          if (toolResultBlocks.length === 0) {
            toolResultBlocks.push({ type: 'tool_result', tool_use_id: msg.tool_call_id || 'unknown', content: msg.content || '' });
          }
        }

        const lastMsg = anthropicMessages[anthropicMessages.length - 1];
        if (lastMsg && lastMsg.role === 'user' && Array.isArray(lastMsg.content)) {
          lastMsg.content.push(...toolResultBlocks);
        } else {
          anthropicMessages.push({ role: 'user', content: toolResultBlocks });
        }
      }
    }

    // Convert OpenAI tool defs to Anthropic format
    let anthropicTools;
    if (tools && tools.length > 0) {
      anthropicTools = tools.map(t => ({
        name: t.function?.name || t.name,
        description: t.function?.description || t.description || '',
        input_schema: t.function?.parameters || t.parameters || { type: 'object', properties: {} },
      }));
    }

    const bodyObj = {
      model: provider.model,
      max_tokens: effectiveMax,
      stream: true,
      messages: anthropicMessages,
      ...(systemPrompt ? { system: systemPrompt } : {}),
      ...(anthropicTools ? { tools: anthropicTools } : {}),
    };
    const body = JSON.stringify(bodyObj);

    const url = new URL(provider.base_url);
    const useHttps = url.protocol === 'https:';
    const lib = useHttps ? https : http;

    const req = lib.request({
      hostname: url.hostname,
      port: url.port || (useHttps ? 443 : 80),
      path: url.pathname + url.search,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
        'anthropic-version': provider.api_version || '2023-06-01',
        ...(provider.api_key ? { 'x-api-key': provider.api_key } : {}),
      },
    }, (res) => {
      if (res.statusCode && res.statusCode >= 400) {
        let errData = '';
        res.on('data', (chunk) => errData += chunk);
        res.on('end', () => {
          let msg = `HTTP ${res.statusCode}`;
          try { const p = JSON.parse(errData); msg = p.error?.message || msg; } catch {}
          reject(new Error(`Anthropic API error: ${msg}`));
        });
        return;
      }

      let response = '';
      let thinking = '';
      let stats = {};
      let buffer = '';
      let toolCalls = null;

      // Track current tool use blocks being built
      const toolUseBlocks = []; // { id, name, input_json }

      // Repetition detection
      const REPEAT_WINDOW = 600;
      const REPEAT_THRESHOLD = 6;
      let repeatCheckCounter = 0;

      function detectRepetition(text) {
        if (text.length < REPEAT_WINDOW) return false;
        const tail = text.slice(-REPEAT_WINDOW);
        for (let pLen = 8; pLen <= 60; pLen++) {
          const pattern = tail.slice(-pLen);
          let count = 0;
          let pos = tail.length - pLen;
          while (pos >= 0) {
            if (tail.slice(pos, pos + pLen) === pattern) { count++; pos -= pLen; } else break;
          }
          if (count >= REPEAT_THRESHOLD) return pattern;
        }
        return false;
      }

      res.on('data', (chunk) => {
        if (aborted) return;
        buffer += chunk.toString();

        let lineEnd;
        while ((lineEnd = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, lineEnd).trim();
          buffer = buffer.slice(lineEnd + 1);
          if (!line) continue;

          // Anthropic SSE uses "event: <type>" and "data: <json>" lines
          if (line.startsWith('event:')) continue; // We parse from data lines

          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6);
          let parsed;
          try { parsed = JSON.parse(payload); } catch { continue; }

          const type = parsed.type;

          if (type === 'content_block_start') {
            const block = parsed.content_block;
            if (block?.type === 'tool_use') {
              toolUseBlocks.push({ id: block.id, name: block.name, input_json: '' });
            }
          } else if (type === 'content_block_delta') {
            const delta = parsed.delta;
            if (delta?.type === 'text_delta' && delta.text) {
              response += delta.text;
              if (onToken) onToken({ type: 'token', content: delta.text });

              repeatCheckCounter++;
              if (repeatCheckCounter >= 50) {
                repeatCheckCounter = 0;
                const rp = detectRepetition(response);
                if (rp) {
                  console.warn(`[anthropic] Repetition loop detected — aborting`);
                  aborted = true;
                  req.destroy();
                  response += '\n\n*(Generation stopped — repetitive output detected)*';
                  if (onToken) onToken({ type: 'token', content: '\n\n*(Generation stopped — repetitive output detected)*' });
                  resolve({ response, thinking, toolCalls, stats });
                  return;
                }
              }
            } else if (delta?.type === 'thinking_delta' && delta.thinking) {
              thinking += delta.thinking;
              if (onToken) onToken({ type: 'thinking', content: delta.thinking });
            } else if (delta?.type === 'input_json_delta' && delta.partial_json) {
              // Accumulate tool input JSON fragments
              if (toolUseBlocks.length > 0) {
                toolUseBlocks[toolUseBlocks.length - 1].input_json += delta.partial_json;
              }
            }
          } else if (type === 'message_delta') {
            // Final usage stats
            if (parsed.usage) {
              stats = {
                prompt_eval_count: parsed.usage.input_tokens || 0,
                eval_count: parsed.usage.output_tokens || 0,
              };
            }
          } else if (type === 'message_start' && parsed.message?.usage) {
            stats.prompt_eval_count = parsed.message.usage.input_tokens || 0;
          }
        }
      });

      res.on('end', () => {
        if (aborted) return;

        // Convert tool_use blocks to the standard toolCalls format
        if (toolUseBlocks.length > 0) {
          toolCalls = toolUseBlocks.map(tb => ({
            id: tb.id,
            type: 'function',
            function: {
              name: tb.name,
              arguments: safeParseJSON(tb.input_json) || {},
            },
          }));
          if (onToken) onToken({ type: 'tool_calls', calls: toolCalls });
        }

        resolve({ response, thinking, toolCalls, stats });
      });

      res.on('error', (err) => { if (!aborted) reject(err); });
    });

    req.on('error', (err) => { if (!aborted) reject(err); });

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
 * Dispatch to Ollama, OpenAI-compatible, Azure, or Anthropic streaming.
 */
function streamChat(opts) {
  const provider = getProvider();
  if (provider.type === 'anthropic') {
    return _streamChatAnthropic({ ...opts, provider });
  }
  if (provider.type === 'openai' || provider.type === 'azure') {
    return _streamChatOpenAI({ ...opts, provider });
  }
  return _streamChatOllama(opts);
}

/**
 * Check if the inference backend is reachable.
 * For remote providers, pings the /models endpoint with auth.
 */
function healthCheck() {
  const provider = getProvider();

  if (provider.type === 'openai' || provider.type === 'azure' || provider.type === 'anthropic') {
    return new Promise((resolve) => {
      try {
        let checkUrl;
        let authHeaders;
        let method = 'GET';

        if (provider.type === 'anthropic') {
          // Anthropic: send a minimal messages request (no /models endpoint)
          checkUrl = new URL(provider.base_url);
          authHeaders = {
            'x-api-key': provider.api_key || '',
            'anthropic-version': provider.api_version || '2023-06-01',
            'Content-Type': 'application/json',
          };
          method = 'POST';
        } else if (provider.type === 'azure') {
          const sep = provider.base_url.includes('?') ? '&' : '?';
          checkUrl = new URL(provider.base_url + sep + 'api-version=' + (provider.api_version || '2024-10-21'));
          authHeaders = provider.api_key ? { 'api-key': provider.api_key } : {};
        } else {
          checkUrl = new URL(provider.base_url + '/models');
          authHeaders = provider.api_key ? { 'Authorization': `Bearer ${provider.api_key}` } : {};
        }

        const useHttps = checkUrl.protocol === 'https:';
        const lib = useHttps ? https : http;

        if (method === 'POST') {
          // Anthropic health check: minimal messages request
          const body = JSON.stringify({
            model: provider.model || 'claude-sonnet-4-20250514',
            messages: [{ role: 'user', content: 'Hi' }],
            max_tokens: 1,
          });
          authHeaders['Content-Length'] = Buffer.byteLength(body);
          const req = lib.request({
            hostname: checkUrl.hostname,
            port: checkUrl.port || (useHttps ? 443 : 80),
            path: checkUrl.pathname + checkUrl.search,
            method: 'POST',
            headers: authHeaders,
          }, (res) => {
            let data = '';
            res.on('data', (chunk) => data += chunk);
            res.on('end', () => resolve(res.statusCode < 400));
          });
          req.on('error', () => resolve(false));
          req.setTimeout(5000, () => { req.destroy(); resolve(false); });
          req.write(body);
          req.end();
        } else {
          const req = lib.get({
            hostname: checkUrl.hostname,
            port: checkUrl.port || (useHttps ? 443 : 80),
            path: checkUrl.pathname + checkUrl.search,
            headers: authHeaders,
          }, (res) => {
            let data = '';
            res.on('data', (chunk) => data += chunk);
            res.on('end', () => resolve(res.statusCode < 400));
          });
          req.on('error', () => resolve(false));
          req.setTimeout(5000, () => { req.destroy(); resolve(false); });
        }
      } catch {
        resolve(false);
      }
    });
  }

  // Default: local Ollama
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
