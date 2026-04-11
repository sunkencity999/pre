// PRE Web GUI — Tool dispatcher + execution loop
// Mirrors the CLI's execute_tool() + tool loop from pre.m

const { streamChat, parseTextToolCalls, stripToolCallText } = require('./ollama');
const { buildSystemPrompt } = require('./context');
const { buildToolDefs } = require('./tools-defs');
const { appendMessage, getSessionMessages, renameSession } = require('./sessions');
const { MODEL_CTX, MAX_TOOL_TURNS } = require('./constants');
const { autoExtract } = require('./memory');
const mcp = require('./mcp');
const hooks = require('./hooks');
const experience = require('./experience');
const chronos = require('./chronos');

/**
 * Generate a short session title from the first user message.
 * Runs in the background — does not block the main response.
 */
function generateSessionTitle(sessionId, userMessage, send) {
  streamChat({
    messages: [
      {
        role: 'system',
        content: 'You are a title generator. Given a user message, output a concise 3-6 word conversation title. Output ONLY the title text, nothing else. No quotes. No punctuation. No explanation. Examples: "Weather in Boulder Creek", "Python Prime Number Function", "Snake Game for Browser"',
      },
      { role: 'user', content: userMessage },
    ],
    maxTokens: 512, // Model needs room for thinking before producing the title
  }).then((result) => {
    console.log(`[title-gen] response: "${result.response}" thinking: "${(result.thinking || '').slice(0, 200)}"`);
    // Model often puts everything in thinking. Extract a title-like line.
    let title = '';
    if (result.response && result.response.trim()) {
      title = result.response.trim().split('\n')[0];
    } else if (result.thinking) {
      // Find the last short line that looks like a title (not a bullet, not the input)
      const lines = result.thinking.split('\n')
        .map(l => l.trim())
        .filter(l => l.length > 0 && l.length < 60
          && !l.startsWith('*') && !l.startsWith('-') && !l.startsWith('#')
          && !l.toLowerCase().includes('input') && !l.toLowerCase().includes('message')
          && !l.toLowerCase().includes('generate') && !l.toLowerCase().includes('title'));
      if (lines.length > 0) {
        title = lines[lines.length - 1]; // last qualifying line is usually the answer
      }
    }
    title = title.replace(/^["'`]|["'`]$/g, '').replace(/[.!:]+$/, '').trim();
    if (title && title.length > 0 && title.length < 80) {
      console.log(`[title-gen] Saving title: "${title}" for session ${sessionId}`);
      renameSession(sessionId, title);
      send({ type: 'session_renamed', sessionId, name: title });
    } else {
      console.log(`[title-gen] Title rejected: "${title}"`);
    }
  }).catch((err) => {
    console.log(`[title-gen] Error: ${err.message}`);
  });
}

// Tool implementations
const bashTool = require('./tools/bash');
const filesTool = require('./tools/files');
const webTool = require('./tools/web');
const memoryTool = require('./tools/memory');
const systemTool = require('./tools/system');
const artifactTool = require('./tools/artifact');
const documentTool = require('./tools/document');
const googleTool = require('./tools/google');
const telegramTool = require('./tools/telegram');
const githubTool = require('./tools/github');
const jiraTool = require('./tools/jira');
const confluenceTool = require('./tools/confluence');
const smartsheetTool = require('./tools/smartsheet');
const slackTool = require('./tools/slack');
const imageTool = require('./tools/image');
const cronTool = require('./tools/cron');
const agentsTool = require('./tools/agents');
const browserTool = require('./tools/browser');

// Tool name aliases — models hallucinate wrong names frequently
const ALIASES = {
  shell: 'bash', run: 'bash', terminal: 'bash', sh: 'bash',
  cmd: 'bash', execute: 'bash', exec: 'bash', command: 'bash',
  run_command: 'bash', shell_exec: 'bash',
  file_read: 'read_file', read: 'read_file', cat: 'read_file',
  view_file: 'read_file', get_file: 'read_file',
  write_file: 'file_write', write: 'file_write', create_file: 'file_write',
  save_file: 'file_write', save: 'file_write',
  edit_file: 'file_edit', edit: 'file_edit', replace: 'file_edit', patch: 'file_edit',
  ls: 'list_dir', list: 'list_dir', dir: 'list_dir', list_directory: 'list_dir',
  search: 'grep', find: 'glob', rg: 'grep', ripgrep: 'grep',
  find_files: 'glob', search_files: 'grep',
  fetch: 'web_fetch', curl: 'web_fetch', http: 'web_fetch',
  browse: 'web_fetch', web: 'web_fetch',
  create_artifact: 'artifact', html: 'artifact', render: 'artifact', display: 'artifact',
  create_document: 'document', doc: 'document', make_document: 'document',
  generate_document: 'document', export: 'document', write_document: 'document',
  copy: 'clipboard_write', paste: 'clipboard_read',
  remember: 'memory_save', recall: 'memory_search', forget: 'memory_delete',
  email: 'gmail', mail: 'gmail', send_email: 'gmail',
  google_drive: 'gdrive', drive: 'gdrive',
  google_docs: 'gdocs', docs: 'gdocs',
  tg: 'telegram', send_telegram: 'telegram', telegram_send: 'telegram',
  jira_search: 'jira', jira_issue: 'jira', jira_create: 'jira',
  wiki: 'confluence', confluence_search: 'confluence', confluence_page: 'confluence',
  spreadsheet: 'smartsheet', ss: 'smartsheet', smartsheets: 'smartsheet',
  slack_send: 'slack', slack_message: 'slack', send_slack: 'slack',
  generate_image: 'image_generate', create_image: 'image_generate', img: 'image_generate',
  image: 'image_generate', image_gen: 'image_generate', gen_image: 'image_generate',
  draw: 'image_generate', paint: 'image_generate', dalle: 'image_generate',
  schedule: 'cron', cron_job: 'cron', recurring: 'cron', timer: 'cron',
  agent: 'spawn_agent', sub_agent: 'spawn_agent', research: 'spawn_agent',
  spawn_parallel: 'spawn_multi', parallel: 'spawn_multi', agents: 'list_agents',
  lessons: 'experience_list', experiences: 'experience_list',
  health: 'memory_health', staleness: 'memory_health',
  browse_web: 'browser', web_browser: 'browser', chrome: 'browser',
  open_browser: 'browser', puppeteer: 'browser', navigate: 'browser',
};

// Tools that require user confirmation before execution
const CONFIRM_TOOLS = new Set([
  'process_kill', 'applescript', 'memory_delete',
]);

// Dispatch a tool call to its handler
// opts.onStatus — callback for streaming status events (used by sub-agents)
async function executeTool(name, args, cwd, opts) {
  // Resolve aliases
  name = ALIASES[name] || name;

  switch (name) {
    // Shell
    case 'bash': return bashTool.bash(args, cwd);

    // Files
    case 'read_file': return filesTool.readFile(args, cwd);
    case 'list_dir': return filesTool.listDir(args, cwd);
    case 'glob': return filesTool.glob(args, cwd);
    case 'grep': return filesTool.grep(args, cwd);
    case 'file_write': return filesTool.fileWrite(args, cwd);
    case 'file_edit': return filesTool.fileEdit(args, cwd);

    // Web
    case 'web_fetch': return webTool.webFetch(args);
    case 'web_search': return webTool.webSearch(args);

    // Memory
    case 'memory_save': return memoryTool.save(args);
    case 'memory_search': return memoryTool.search(args);
    case 'memory_list': return memoryTool.list();
    case 'memory_delete': return memoryTool.del(args);

    // System
    case 'system_info': return systemTool.systemInfo();
    case 'process_list': return systemTool.processList(args);
    case 'process_kill': return systemTool.processKill(args);
    case 'hardware_info': return systemTool.hardwareInfo();
    case 'disk_usage': return systemTool.diskUsage(args);
    case 'net_info': return systemTool.netInfo();
    case 'net_connections': return systemTool.netConnections(args);
    case 'service_status': return systemTool.serviceStatus(args);
    case 'display_info': return systemTool.displayInfo();
    case 'clipboard_read': return systemTool.clipboardRead();
    case 'clipboard_write': return systemTool.clipboardWrite(args);
    case 'open_app': return systemTool.openApp(args);
    case 'notify': return systemTool.notify(args);
    case 'screenshot': return systemTool.screenshot(args);
    case 'window_list': return systemTool.windowList();
    case 'window_focus': return systemTool.windowFocus(args);
    case 'applescript': return systemTool.applescript(args);

    // Artifacts
    case 'artifact': return artifactTool.createArtifact(args);

    // GitHub
    case 'github': return githubTool.github(args);

    // Jira
    case 'jira': return jiraTool.jira(args);

    // Confluence
    case 'confluence': return confluenceTool.confluence(args);

    // Smartsheet
    case 'smartsheet': return smartsheetTool.smartsheet(args);

    // Slack
    case 'slack': return slackTool.slack(args);

    // Google
    case 'gmail': return googleTool.gmail(args);
    case 'gdrive': return googleTool.gdrive(args);
    case 'gdocs': return googleTool.gdocs(args);

    // Telegram
    case 'telegram': return telegramTool.telegram(args);

    // Image generation
    case 'image_generate': return imageTool.imageGenerate(args);

    // Browser
    case 'browser': return browserTool.browserAction(args);

    // Cron / scheduling
    case 'cron': return cronTool.cron(args);

    // Sub-agents — onStatus passed through from runToolLoop for WS streaming
    case 'spawn_agent': return agentsTool.spawnAgent(args, cwd, opts?.onStatus);
    case 'spawn_multi': {
      // Parse tasks if passed as JSON string
      if (args.tasks && typeof args.tasks === 'string') {
        try { args.tasks = JSON.parse(args.tasks); } catch {}
      }
      return agentsTool.spawnMulti(args, cwd, opts?.onStatus);
    }
    case 'list_agents': return agentsTool.listAgents();

    // Experience ledger
    case 'experience_search': {
      const results = await experience.searchExperiences(args.query || '');
      if (results.length === 0) return 'No relevant past experience found.';
      return results.map(r => `**${r.name}** (similarity: ${r.similarity?.toFixed(2) || 'keyword'})\n${r.body}`).join('\n\n---\n\n');
    }
    case 'experience_list': {
      const entries = experience.listExperiences();
      if (entries.length === 0) return 'No experience entries yet. The ledger builds automatically as you complete tasks.';
      return entries.map(e => `[${e.created}] **${e.name}**: ${e.description}`).join('\n');
    }

    // Chronos
    case 'memory_health': {
      const summary = chronos.maintenanceSummary();
      return `Memory Health: ${summary.healthPct}% fresh\n`
        + `Total: ${summary.total} | Fresh: ${summary.fresh} | Aging: ${summary.aging} | Stale: ${summary.stale} | Unverified: ${summary.unverified}`
        + (summary.oldestStale ? `\nOldest stale: "${summary.oldestStale.name}" (${summary.oldestStale.verifiedAge}d)` : '');
    }

    // Documents
    case 'document': {
      // Parse sheets if passed as JSON string
      if (args.sheets && typeof args.sheets === 'string') {
        try { args.sheets = JSON.parse(args.sheets); } catch {}
      }
      return documentTool.createDocument(args);
    }

    default:
      // Check if this is an MCP tool
      if (mcp.isMCPTool(name)) {
        return mcp.callTool(name, args);
      }
      return `Error: unknown tool '${name}'. Available tools: bash, read_file, list_dir, glob, grep, file_write, file_edit, web_fetch, web_search, memory_save, memory_search, memory_list, memory_delete, system_info, image_generate, cron, spawn_agent, spawn_multi, list_agents, github, jira, confluence, smartsheet, slack, gmail, gdrive, gdocs, telegram, artifact, document`;
  }
}

/**
 * Run the full tool execution loop for a chat turn.
 * Streams the response, executes tool calls, sends follow-ups.
 *
 * @param {Object} opts
 * @param {string} opts.sessionId
 * @param {string} opts.cwd - Working directory
 * @param {Function} opts.send - Send WS message to client
 * @param {AbortSignal} opts.signal
 * @param {Function} opts.onConfirmRequest - Ask client to confirm dangerous tool
 * @returns {Promise<void>}
 */
async function runToolLoop({ sessionId, cwd, send, signal, onConfirmRequest, userMessage, needsTitle }) {
  let tokensIn = 0;
  let tokensOut = 0;

  for (let turn = 0; turn < MAX_TOOL_TURNS; turn++) {
    if (signal?.aborted) break;

    const systemPrompt = buildSystemPrompt(cwd);
    const history = getSessionMessages(sessionId);
    const messages = [
      { role: 'system', content: systemPrompt },
      ...history,
    ];
    const tools = buildToolDefs();

    // Budget must be generous — Gemma 4 extended thinking consumes num_predict
    // before producing visible response tokens
    const maxTokens = turn === 0 ? 16384 : 24576;

    const result = await streamChat({
      messages,
      tools,
      maxTokens,
      signal,
      onToken: (event) => send(event),
    });

    // Update token counts
    tokensIn += result.stats.prompt_eval_count || 0;
    tokensOut += result.stats.eval_count || 0;

    // Fallback: parse <tool_call> tags from text if no native tool calls
    const hasNative = result.toolCalls && result.toolCalls.length > 0;
    const hasTag = result.response && result.response.includes('<tool_call>');
    console.log(`[tool-loop] turn=${turn} hasNative=${hasNative} hasTag=${hasTag} responseLen=${(result.response||'').length}`);
    if (!hasNative && result.response) {
      const textCalls = parseTextToolCalls(result.response);
      if (textCalls) {
        console.log(`[tool-loop] Parsed ${textCalls.length} text tool call(s):`, textCalls.map(c => c.function?.name));
        result.toolCalls = textCalls;
        // Strip tool call text from displayed response
        result.response = stripToolCallText(result.response);
        // Notify client about parsed tool calls
        send({ type: 'tool_calls', calls: textCalls });
      }
    }

    // Save assistant message
    if (result.response || result.toolCalls) {
      const assistantMsg = { role: 'assistant', content: result.response || '' };
      if (result.toolCalls) {
        assistantMsg.tool_calls = result.toolCalls;
      }
      appendMessage(sessionId, assistantMsg);
    }

    // No tool calls — we're done
    if (!result.toolCalls || result.toolCalls.length === 0) {
      send({
        type: 'done',
        stats: { ...result.stats },
        context: {
          used: tokensIn + tokensOut,
          max: MODEL_CTX,
          pct: Math.round((tokensIn + tokensOut) * 100 / MODEL_CTX),
        },
      });
      // Generate session title in background after first response
      if (needsTitle && userMessage) {
        generateSessionTitle(sessionId, userMessage, send);
      }
      // Auto-extract memories in background (don't block response)
      const historyForExtract = getSessionMessages(sessionId);
      autoExtract(historyForExtract).then(saved => {
        if (saved.length > 0) {
          console.log(`[memory-extract] Auto-saved ${saved.length} memory(ies)`);
          send({ type: 'memory_saved', memories: saved.map(m => ({ name: m.name, type: m.type })) });
        }
      }).catch(err => {
        console.log(`[memory-extract] Background error: ${err.message}`);
      });
      // Experience ledger: reflect on what worked/failed (don't block response)
      experience.reflect(historyForExtract, { sessionId, cwd }).then(saved => {
        if (saved.length > 0) {
          console.log(`[experience] Saved ${saved.length} lesson(s)`);
          send({ type: 'experience_saved', lessons: saved.map(l => ({ name: l.name, lesson: l.lesson })) });
        }
      }).catch(err => {
        console.log(`[experience] Background error: ${err.message}`);
      });
      return;
    }

    // Execute tool calls
    send({ type: 'done_partial', stats: result.stats });

    const toolResults = [];
    for (const tc of result.toolCalls) {
      const toolName = ALIASES[tc.function?.name] || tc.function?.name || 'unknown';
      const toolArgs = tc.function?.arguments || {};
      const toolId = `tc_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

      // Notify client about tool call
      send({
        type: 'tool_call',
        name: toolName,
        args: toolArgs,
        id: toolId,
        status: 'running',
      });

      // Check if confirmation is needed
      if (CONFIRM_TOOLS.has(toolName)) {
        if (onConfirmRequest) {
          const approved = await onConfirmRequest(toolId, toolName, toolArgs);
          if (!approved) {
            send({ type: 'tool_result', id: toolId, name: toolName, output: 'Skipped by user', status: 'skipped' });
            toolResults.push({ name: toolName, output: 'Tool execution skipped by user.' });
            continue;
          }
        }
      }

      // Run pre_tool hooks (can block)
      const preHook = hooks.runHooks('pre_tool', { tool: toolName, args: toolArgs, sessionId, cwd });
      if (preHook.blocked) {
        send({ type: 'tool_result', id: toolId, name: toolName, output: `Blocked by hook: ${preHook.reason}`, status: 'blocked' });
        toolResults.push({ name: toolName, output: `Blocked by hook: ${preHook.reason}` });
        continue;
      }

      // Execute the tool
      let output;
      try {
        output = await executeTool(toolName, toolArgs, cwd, {
          onStatus: (event) => send({ type: 'agent_status', ...event }),
        });
      } catch (err) {
        output = `Error: ${err.message}`;
      }

      // Run post_tool hooks (non-blocking, for logging/auditing)
      hooks.runHooks('post_tool', { tool: toolName, args: toolArgs, output: output?.slice(0, 4000), sessionId, cwd });

      // Notify client about generated images so they can display inline
      if (toolName === 'image_generate' && output && output.includes('/artifacts/')) {
        const urlMatch = output.match(/View at: (\/artifacts\/[^\s]+)/);
        if (urlMatch) {
          send({
            type: 'image_generated',
            prompt: toolArgs.prompt || '',
            path: urlMatch[1],
          });
        }
      }

      // Notify client about artifacts/documents so they can display download cards
      if ((toolName === 'artifact' || toolName === 'document') && output && output.includes('/artifacts/')) {
        const urlMatch = output.match(/\/artifacts\/[^\s]+/);
        if (urlMatch) {
          const isDoc = toolName === 'document';
          send({
            type: isDoc ? 'document' : 'artifact',
            title: toolArgs.title || (isDoc ? 'Document' : 'Artifact'),
            path: urlMatch[0],
            artifactType: isDoc ? (toolArgs.format || 'txt') : (toolArgs.type || 'html'),
          });
        }
      }

      // Handle browser tool screenshots — extract base64 image for vision
      let browserScreenshot = null;
      if (toolName === 'browser' && output) {
        try {
          const parsed = JSON.parse(output);
          if (parsed.screenshot) {
            browserScreenshot = parsed.screenshot;
            // Send screenshot to client for display
            send({ type: 'browser_screenshot', id: toolId, screenshot: parsed.screenshot, message: parsed.message });
            // Strip screenshot from text output (too large for text context)
            delete parsed.screenshot;
            output = JSON.stringify(parsed);
          }
        } catch {}
      }

      // Truncate very large outputs
      const MAX_OUTPUT = 32000;
      if (output.length > MAX_OUTPUT) {
        output = output.slice(0, MAX_OUTPUT) + `\n\n[...truncated ${output.length - MAX_OUTPUT} bytes]`;
      }

      send({
        type: 'tool_result',
        id: toolId,
        name: toolName,
        output: output.slice(0, 2000), // Send abbreviated to client
        status: 'done',
      });

      toolResults.push({ name: toolName, output, image: browserScreenshot });
    }

    // Save tool results to session (combined format matching CLI)
    const combinedResult = toolResults
      .map(r => `<tool_response name="${r.name}">\n${r.output}</tool_response>`)
      .join('\n');
    const toolMsg = { role: 'tool', content: combinedResult };
    // Attach browser screenshots as images so the model can see them
    const screenshots = toolResults.filter(r => r.image).map(r => r.image);
    if (screenshots.length > 0) {
      toolMsg.images = screenshots;
    }
    appendMessage(sessionId, toolMsg);

    // Start next iteration (model sees tool results and may call more tools)
  }

  // Hit max tool turns
  send({ type: 'error', message: `Tool loop reached maximum of ${MAX_TOOL_TURNS} turns` });
}

module.exports = { executeTool, runToolLoop, ALIASES, CONFIRM_TOOLS };
