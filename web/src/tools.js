// PRE Web GUI — Tool dispatcher + execution loop
// Mirrors the CLI's execute_tool() + tool loop from pre.m

const fs = require('fs');
const path = require('path');
const { streamChat, parseTextToolCalls, stripToolCallText } = require('./ollama');
const { buildSystemPrompt, buildSystemPromptAsync } = require('./context');
const { buildToolDefs } = require('./tools-defs');
const { appendMessage, getSessionMessages, renameSession } = require('./sessions');
const { MODEL_CTX, MAX_TOOL_TURNS } = require('./constants');
const { autoExtract } = require('./memory');
const mcp = require('./mcp');
const customTools = require('./custom-tools');
const hooks = require('./hooks');
const experience = require('./experience');
const chronos = require('./chronos');
const argus = require('./argus');
const tiers = require('./tool-tiers');
const compression = require('./compression');
const { getProvider } = require('./connections');

/**
 * Generate a short session title from the first user message.
 * Runs in the background — does not block the main response.
 */
function generateSessionTitle(sessionId, userMessage, send) {
  // Truncate very long messages — we only need enough to understand the topic
  const truncated = userMessage.length > 300 ? userMessage.slice(0, 300) : userMessage;
  return streamChat({
    messages: [
      {
        role: 'system',
        content: 'You are a title generator. Your ENTIRE response must be exactly one short title (3-6 words). Output NOTHING else — no preamble, no options, no list, no explanation. Just the title.\n\nExamples of correct output:\nWeather in Boulder Creek\nPython Prime Number Function\nDocusign Certificate Email Search',
      },
      { role: 'user', content: truncated },
    ],
    maxTokens: 128,
    think: false,
  }).then((result) => {
    console.log(`[title-gen] response: "${result.response}" thinking: "${(result.thinking || '').slice(0, 200)}"`);
    let title = '';

    // Preamble/meta phrases that are not actual titles
    const junkPattern = /^(possible titles?|here are|some options|options?:|sure|okay|certainly|of course|title:|titles:)/i;

    // Prefer the actual response text — skip preamble lines
    if (result.response && result.response.trim()) {
      const lines = result.response.trim().split('\n')
        .map(l => l.replace(/^[\d.)\-*]+\s*/, '').trim())  // strip list markers
        .filter(l => l.length >= 3 && l.length < 60 && !junkPattern.test(l));
      if (lines.length > 0) title = lines[0];
    }

    // Fallback: extract from thinking if response is empty (Gemma 4 thinking mode)
    if (!title && result.thinking) {
      // Strip bullet/list prefixes and common labels, then filter
      const lines = result.thinking.split('\n')
        .map(l => l.trim()
          .replace(/^[*\-#>]+\s*/, '')          // strip bullet/heading markers
          .replace(/^(topic|title|subject)\s*[:=]\s*/i, '')  // strip label prefixes
          .trim()
        )
        .filter(l =>
          l.length >= 3 && l.length < 60
          && !junkPattern.test(l)
          && !l.toLowerCase().includes('user message') && !l.toLowerCase().includes('the user')
          && !l.toLowerCase().includes('generate') && !l.toLowerCase().includes('should be')
          && !l.toLowerCase().includes('let me') && !l.toLowerCase().includes("let's")
          && !l.toLowerCase().includes('i need') && !l.toLowerCase().includes('i think')
          && !l.toLowerCase().includes('how about') && !l.toLowerCase().includes('go with')
          && !l.toLowerCase().includes('→')
          && !l.toLowerCase().includes('possible') && !l.toLowerCase().includes('option')
          && !/^(ok|so|now|here|this|that)\b/i.test(l)
        );
      if (lines.length > 0) {
        title = lines[lines.length - 1];
      }
    }

    // Clean up: strip quotes, trailing punctuation, "Title:" prefix
    title = title
      .replace(/^(title\s*[:=]\s*)/i, '')
      .replace(/^["'`"""'']+|["'`"""'']+$/g, '')
      .replace(/[.!:;,]+$/, '')
      .trim();

    if (title && title.length >= 3 && title.length < 80) {
      console.log(`[title-gen] Saving title: "${title}" for session ${sessionId}`);
      renameSession(sessionId, title);
      send({ type: 'session_renamed', sessionId, name: title });
    } else {
      console.log(`[title-gen] Title rejected (len=${title.length}): "${title}"`);
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
const sharepointTool = require('./tools/sharepoint');
const imageTool = require('./tools/image');
const exportTool = require('./tools/export');
const mailTool = require('./tools/mail');
const calendarTool = require('./tools/calendar');
const contactsTool = require('./tools/contacts');
const spotlightTool = require('./tools/spotlight');
const remindersTool = require('./tools/reminders');
const notesTool = require('./tools/notes');
const cronTool = require('./tools/cron');
const agentsTool = require('./tools/agents');
const monitorTool = require('./tools/monitor');
const browserTool = require('./tools/browser');
const computerTool = require('./tools/computer');
const linearTool = require('./tools/linear');
const zoomTool = require('./tools/zoom');
const figmaTool = require('./tools/figma');
const asanaTool = require('./tools/asana');
const dynamics365Tool = require('./tools/dynamics365');
const ragTool = require('./tools/rag');
const voiceTool = require('./tools/voice');
const workflowTool = require('./tools/workflow');
const triggerSystem = require('./triggers');

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
  sharepoint_search: 'sharepoint', sp: 'sharepoint', microsoft: 'sharepoint',
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
  desktop: 'computer', computer_use: 'computer', screen: 'computer',
  mouse: 'computer', keyboard: 'computer', click: 'computer',
  linear_issue: 'linear', linear_search: 'linear', issues: 'linear',
  zoom_meeting: 'zoom', meeting: 'zoom', zoom_create: 'zoom',
  figma_file: 'figma', design: 'figma', figma_comments: 'figma',
  asana_task: 'asana', task_manager: 'asana', asana_search: 'asana',
  d365: 'dynamics365', dynamics: 'dynamics365', dataverse: 'dynamics365', crm: 'dynamics365',
  dynamics_365: 'dynamics365', d365_search: 'dynamics365', d365_record: 'dynamics365',
  rag_search: 'rag', rag_index: 'rag', vector_search: 'rag', semantic_search: 'rag',
  document_search: 'rag', knowledge_base: 'rag', kb: 'rag',
  triggers: 'trigger', watch: 'trigger', file_watch: 'trigger', webhook: 'trigger',
  event_trigger: 'trigger', watcher: 'trigger',
  speak: 'voice', say: 'voice', tts: 'voice', stt: 'voice',
  transcribe: 'voice', speech: 'voice', dictate: 'voice',
  record_workflow: 'workflow', replay_workflow: 'workflow', macro: 'workflow',
  automation: 'workflow', playback: 'workflow',
  export_pdf: 'pdf_export', share_pdf: 'pdf_export', artifact_pdf: 'pdf_export',
  export_artifact: 'pdf_export', share: 'pdf_export',
  send_mail: 'apple_mail', email_send: 'apple_mail', read_mail: 'apple_mail',
  inbox: 'apple_mail', mailbox: 'apple_mail',
  events: 'apple_calendar', schedule_event: 'apple_calendar', create_event: 'apple_calendar',
  my_calendar: 'apple_calendar', today_events: 'apple_calendar', agenda: 'apple_calendar',
  find_contact: 'apple_contacts', lookup: 'apple_contacts', address_book: 'apple_contacts',
  people: 'apple_contacts', phone: 'apple_contacts',
  mdfind: 'spotlight', find_file: 'spotlight', search_files_spotlight: 'spotlight',
  file_search: 'spotlight', locate: 'spotlight',
  reminder: 'apple_reminders', reminders: 'apple_reminders', todo: 'apple_reminders',
  todos: 'apple_reminders', add_reminder: 'apple_reminders', tasks: 'apple_reminders',
  note: 'apple_notes', notes: 'apple_notes', create_note: 'apple_notes',
  search_notes: 'apple_notes', apple_note: 'apple_notes',
};

// Tools that require user confirmation before execution
const CONFIRM_TOOLS = new Set([
  'process_kill', 'memory_delete',
]);

// Tool+action combinations that require confirmation (for multi-action tools)
const CONFIRM_ACTIONS = {
  sharepoint: new Set(['delete_file']),
  zoom: new Set(['delete_meeting']),
  apple_mail: new Set(['send']),
  apple_calendar: new Set(['delete_event']),
  apple_reminders: new Set(['delete']),
  rag: new Set(['delete']),
  trigger: new Set(['delete']),
  workflow: new Set(['delete']),
  dynamics365: new Set(['delete_record']),
};

// Dispatch a tool call to its handler
// opts.onStatus — callback for streaming status events (used by sub-agents)
async function executeTool(name, args, cwd, opts) {
  // Strip domain prefix (e.g. "media:image_generate" → "image_generate")
  // Models hallucinate this pattern after seeing request_tools domain activation.
  if (name.includes(':')) {
    name = name.split(':').pop();
  }
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
    case 'open_app': {
      // Track target app for computer use auto-focus
      if (args?.target) computerTool.setTargetApp(args.target);
      const openResult = systemTool.openApp(args);
      // Maximize the window so the model gets a clear, full-size view
      if (args?.target && computerTool.isAvailable()) {
        try {
          // Wait for the app to finish opening, then maximize
          await new Promise(r => setTimeout(r, 1000));
          computerTool.maximizeTargetWindow();
          // Second attempt after brief delay — some apps need time to fully render
          await new Promise(r => setTimeout(r, 500));
          computerTool.maximizeTargetWindow();
        } catch {}
      }
      return openResult;
    }
    case 'notify': return systemTool.notify(args);
    case 'screenshot':
      // When computer use is available, redirect to the computer tool's vision pipeline
      // so the model actually sees the screenshot (base64 for vision) instead of just saving a file
      if (computerTool.isAvailable()) {
        return computerTool.computerUse({ action: 'screenshot' });
      }
      return systemTool.screenshot(args);
    case 'window_list': return systemTool.windowList();
    case 'window_focus':
      // Track target app for computer use auto-focus
      if (args?.app) computerTool.setTargetApp(args.app);
      return systemTool.windowFocus(args);
    case 'applescript': return systemTool.applescript(args);
    case 'powershell_script': return systemTool.powershellScript(args);

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

    // SharePoint
    case 'sharepoint': return sharepointTool.sharepoint(args);

    // Google
    case 'gmail': return googleTool.gmail(args);
    case 'gdrive': return googleTool.gdrive(args);
    case 'gdocs': return googleTool.gdocs(args);

    // Telegram
    case 'telegram': return telegramTool.telegram(args);

    // Linear
    case 'linear': return linearTool.linear(args);

    // Zoom
    case 'zoom': return zoomTool.zoom(args);

    // Figma
    case 'figma': return figmaTool.figma(args);

    // Asana
    case 'asana': return asanaTool.asana(args);

    // Dynamics 365
    case 'dynamics365': return dynamics365Tool.dynamics365(args);

    // Image generation
    case 'image_generate': return imageTool.imageGenerate(args);

    // Computer Use (desktop automation)
    case 'computer': {
      const result = await computerTool.computerUse(args);
      // Record step for workflow capture
      if (workflowTool.isRecording()) {
        workflowTool.recordStep(args.action, args);
      }
      return result;
    }

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

    // Background process monitor
    case 'monitor': return monitorTool.monitor(args);

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

    // RAG (local document intelligence)
    case 'rag': return ragTool.rag(args, cwd);

    // Event-driven triggers
    case 'trigger': return triggerSystem.trigger(args);

    // Voice (STT/TTS)
    case 'voice': return voiceTool.voice(args);

    // Workflow capture and replay
    case 'workflow': return workflowTool.workflow(args);

    // Custom tools (self-architected)
    case 'custom_tool': return customTools.customTool(args);

    // Progressive disclosure: request_tools is handled in runToolLoop (needs session context)
    case 'request_tools': return 'Error: request_tools must be handled by the tool loop';

    // Session search (full-text search across past conversations)
    case 'session_search': {
      const fts = require('./fts');
      return fts.searchSessions(args.query, { maxResults: args.count || 20, project: args.project });
    }

    // Chronos
    case 'memory_health': {
      const summary = chronos.maintenanceSummary();
      return `Memory Health: ${summary.healthPct}% fresh\n`
        + `Total: ${summary.total} | Fresh: ${summary.fresh} | Aging: ${summary.aging} | Stale: ${summary.stale} | Unverified: ${summary.unverified}`
        + (summary.oldestStale ? `\nOldest stale: "${summary.oldestStale.name}" (${summary.oldestStale.verifiedAge}d)` : '');
    }

    // macOS native Mail, Calendar, Contacts (zero-config)
    case 'apple_mail': return mailTool.mail(args);
    case 'apple_calendar': return calendarTool.calendar(args);
    case 'apple_contacts': return contactsTool.contacts(args);
    case 'spotlight': return spotlightTool.spotlight(args);
    case 'apple_reminders': return remindersTool.reminders(args);
    case 'apple_notes': return notesTool.notes(args);

    // PDF Export (artifact → PDF via Puppeteer)
    case 'pdf_export': {
      // Find the artifact to export — by title search or explicit path
      const artTitle = args.title || 'latest';
      const artPath = args.path; // optional explicit web path
      let webPath = artPath;

      if (!webPath) {
        // Search artifacts directory for a matching HTML file
        const artDir = require('./constants').ARTIFACTS_DIR;
        const files = fs.readdirSync(artDir)
          .filter(f => f.endsWith('.html'))
          .map(f => ({ name: f, mtime: fs.statSync(path.join(artDir, f)).mtimeMs }))
          .sort((a, b) => b.mtime - a.mtime);

        if (artTitle.toLowerCase() === 'latest') {
          if (files.length === 0) return 'No HTML artifacts found to export.';
          webPath = `/artifacts/${files[0].name}`;
        } else {
          const slug = artTitle.toLowerCase().replace(/[^a-z0-9]+/g, '-');
          const match = files.find(f => f.name.includes(slug));
          if (match) {
            webPath = `/artifacts/${match.name}`;
          } else if (files.length > 0) {
            webPath = `/artifacts/${files[0].name}`;
          } else {
            return `No artifact found matching "${artTitle}".`;
          }
        }
      }

      const result = await exportTool.exportPdf(webPath, artTitle);
      return `PDF exported: ${result.filename}\nDownload: ${result.webPath}`;
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
      // Check if this is a custom (self-architected) tool
      if (customTools.isCustomTool(name)) {
        return customTools.executeCustomTool(name.slice(7), args); // strip 'custom_' prefix
      }
      // Check if this is an MCP tool
      if (mcp.isMCPTool(name)) {
        return mcp.callTool(name, args);
      }
      return `Error: unknown tool '${name}'. Available tools: bash, read_file, list_dir, glob, grep, file_write, file_edit, web_fetch, web_search, memory_save, memory_search, memory_list, memory_delete, system_info, image_generate, cron, spawn_agent, spawn_multi, list_agents, rag, github, jira, confluence, smartsheet, slack, sharepoint, gmail, gdrive, gdocs, telegram, artifact, document, pdf_export, apple_mail, apple_calendar, apple_contacts, spotlight, apple_reminders, apple_notes`;
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
// Track active user loops so background jobs (cron, triggers) can queue
let _activeUserLoops = 0;
const _loopWaiters = []; // Resolve callbacks waiting for loops to drain

function isUserLoopActive() { return _activeUserLoops > 0; }

/**
 * Returns a promise that resolves when no user loops are active.
 * Resolves immediately if already idle.
 */
function waitForIdle() {
  if (_activeUserLoops <= 0) return Promise.resolve();
  return new Promise(resolve => _loopWaiters.push(resolve));
}

async function runToolLoop({ sessionId, cwd, send, signal, onConfirmRequest, userMessage, needsTitle, maxTurns }) {
  // Track user-initiated loops (non-cron) for background job queuing
  const isUserLoop = !sessionId.startsWith('cron:') && !sessionId.startsWith('mcp:');
  if (isUserLoop) _activeUserLoops++;

  // Wrap send to let Argus observe all events (fire-and-forget, never blocks)
  const originalSend = send;
  send = (event) => {
    originalSend(event);
    argus.observeEvent(event);
  };
  // Give Argus the user's prompt for conversational context
  if (userMessage) {
    argus.observeEvent({ type: 'user_message', content: userMessage });
  }

  // Signal Argus that the tool loop is active — suppresses deferred reactions
  // that would compete with Ollama inference during the loop.
  argus.setToolLoopActive(true);

  const turnLimit = maxTurns || MAX_TOOL_TURNS;
  let tokensIn = 0;
  let tokensOut = 0;
  let pendingScreenshots = []; // Screenshots from computer/browser tools, injected transiently
  let usedComputerUse = false; // Track if computer use was invoked for completion notification

  // Progressive tool disclosure: auto-activate domains from user message keywords
  const activeDomains = tiers.getActiveDomains(sessionId);
  if (userMessage) {
    tiers.resolveKeywords(sessionId, userMessage);
  }
  let pendingClickWarning = null; // Click-loop warning to inject with next screenshot
  let emptyResponseNudged = false; // One-shot: nudge model if thinking consumed entire budget

  // Build system prompt once — it doesn't change between tool-loop turns.
  // Turn 0 uses async version (relevance-ranked memory/experience), cached for reuse.
  let cachedSystemPrompt = null;
  // Tool definitions are rebuilt only when domains change (via request_tools).
  let cachedTools = null;
  let toolDefsStale = true; // Start stale so first turn builds

  for (let turn = 0; turn < turnLimit; turn++) {
    if (signal?.aborted) break;

    if (!cachedSystemPrompt) {
      cachedSystemPrompt = userMessage
        ? await buildSystemPromptAsync(cwd, userMessage)
        : buildSystemPrompt(cwd);
    }
    const systemPrompt = cachedSystemPrompt;
    const history = getSessionMessages(sessionId);
    // Compress old turns if approaching context limit (runs only when needed)
    const compressedHistory = await compression.compressIfNeeded(sessionId, history, activeDomains.size);
    const messages = [
      { role: 'system', content: systemPrompt },
      ...compressedHistory,
    ];

    // Inject pending screenshots as a transient user message (not persisted to session).
    // Ollama only processes images on user role messages. This message is ephemeral —
    // it gets the model to see the screenshot without polluting the session history.
    if (pendingScreenshots.length > 0) {
      let screenshotPrompt = '[Screenshot captured by the computer tool is attached. Analyze the image and continue completing the task by calling tools. Do not just describe what you see — take action.';
      if (pendingClickWarning) {
        screenshotPrompt += `\n\n${pendingClickWarning}`;
        pendingClickWarning = null;
      }
      screenshotPrompt += ']';
      messages.push({
        role: 'user',
        content: screenshotPrompt,
        images: pendingScreenshots,
      });
      pendingScreenshots = [];
    }

    if (toolDefsStale || !cachedTools) {
      cachedTools = buildToolDefs({ activeDomains });
      toolDefsStale = false;
    }
    const tools = cachedTools;

    // Budget must be generous — Gemma 4 extended thinking consumes num_predict
    // before producing visible response tokens. With 128K context and free local
    // inference, we maximize output capacity so reports and research are thorough.
    // Remote providers get capped to their configured max_tokens to control costs.
    const provider = getProvider();
    const baseMax = turn === 0 ? 32768 : 49152;
    const isRemote = provider.type === 'openai' || provider.type === 'azure' || provider.type === 'anthropic';
    const maxTokens = isRemote
      ? Math.min(baseMax, provider.max_tokens || 4096)
      : baseMax;

    const result = await streamChat({
      messages,
      tools,
      maxTokens,
      signal,
      onToken: (event) => send(event),
      // Lower temperature for tool-calling turns reduces hallucinated tool names
      // and inconsistent parameter formatting. Modelfile default is 1.0 (creative),
      // but agent mode benefits from deterministic tool selection.
      extraOptions: { temperature: 0.4 },
      // Disable thinking for remote providers (most don't support it, wastes tokens)
      ...(isRemote ? { think: false } : {}),
    });

    // Update token counts
    tokensIn += result.stats.prompt_eval_count || 0;
    tokensOut += result.stats.eval_count || 0;

    // Calibrate token estimator with actual Ollama counts
    compression.calibrate(messages, result.stats.prompt_eval_count);

    // Title generation deferred to after tool loop (see background tasks below)
    // to avoid GPU contention with main inference.

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

    // No tool calls — check for cases where the model needs a nudge before finishing.
    if (!result.toolCalls || result.toolCalls.length === 0) {
      const responseText = (result.response || '').trim();
      const thinkingLen = (result.thinking || '').length;

      // (#A) Empty response after substantial thinking — the model put the entire
      // analysis in the thinking block and produced no visible output. One-shot nudge
      // to get it to write the actual response. Only fires once per conversation.
      if (!responseText && thinkingLen > 500 && !emptyResponseNudged) {
        emptyResponseNudged = true;
        console.log(`[tool-loop] Empty response with ${thinkingLen} chars of thinking — nudging for visible output`);
        messages.push({ role: 'assistant', content: '(thinking completed)' });
        messages.push({ role: 'user', content: 'Please write your full response now. Your analysis should be visible to me, not just in your reasoning.' });
        send({ type: 'done_partial', stats: result.stats });
        continue;
      }

      // (#B) Unfulfilled artifact intent — model plans to create an artifact but stops.
      if (turn === 1 && responseText) {
        const resp = responseText.toLowerCase();
        const mentionsArtifact = /\b(artifact|dashboard|html|webpage|visualization)\b/.test(resp);
        const mentionsIntent = /\b(will (write|create|build|generate|produce|make)|let me (create|build|write)|now (create|write|build)|here'?s the)\b/.test(resp);
        if (mentionsArtifact && mentionsIntent) {
          console.log('[tool-loop] Detected unfulfilled artifact intent — sending continuation nudge');
          messages.push({ role: 'assistant', content: responseText });
          messages.push({ role: 'user', content: 'Go ahead — call the artifact tool now with the full HTML content.' });
          send({ type: 'done_partial', stats: result.stats });
          continue;
        }
      }

      send({
        type: 'done',
        stats: {
          ...result.stats,
          // Override with accumulated totals across all tool loop turns
          eval_count: tokensOut,
          prompt_eval_count: tokensIn,
        },
        context: {
          used: tokensIn + tokensOut,
          max: MODEL_CTX,
          pct: Math.round((tokensIn + tokensOut) * 100 / MODEL_CTX),
        },
      });
      // When computer use session completes: notify and restore focus
      if (usedComputerUse) {
        try {
          const { execSync } = require('child_process');
          // Sanitize for AppleScript — strip all quotes and special chars
          const preview = (result.response || '')
            .replace(/["""''`]/g, "'")
            .replace(/\\/g, '')
            .replace(/[\n\r]/g, ' ')
            .slice(0, 120);
          execSync(`osascript -e 'display notification "${preview}" with title "PRE — Task Complete" sound name "Glass"'`, { timeout: 3000 });
        } catch (err) {
          console.log(`[computer] notification failed: ${err.message}`);
        }
        // Restore focus to the app that was active before computer use (browser/terminal)
        try {
          computerTool.restoreFocus();
        } catch {}
      }
      // Background intelligence tasks — serialized to avoid GPU contention.
      // Each calls Ollama; running them concurrently halves throughput on a single GPU.
      // Chained sequentially: title → memory → experience → skills. Non-blocking.
      const historyForExtract = getSessionMessages(sessionId);
      const skills = require('./skills');
      (async () => {
        // Session title generation (deferred from turn 0 to avoid GPU contention)
        if (needsTitle && userMessage) {
          try {
            await generateSessionTitle(sessionId, userMessage, send);
          } catch (err) {
            console.log(`[title-gen] Deferred error: ${err.message}`);
          }
        }
        try {
          const saved = await autoExtract(historyForExtract);
          if (saved.length > 0) {
            console.log(`[memory-extract] Auto-saved ${saved.length} memory(ies)`);
            send({ type: 'memory_saved', memories: saved.map(m => ({ name: m.name, type: m.type })) });
          }
        } catch (err) {
          console.log(`[memory-extract] Background error: ${err.message}`);
        }
        try {
          const saved = await experience.reflect(historyForExtract, { sessionId, cwd });
          if (saved.length > 0) {
            console.log(`[experience] Saved ${saved.length} lesson(s)`);
            send({ type: 'experience_saved', lessons: saved.map(l => ({ name: l.name, lesson: l.lesson })) });
          }
        } catch (err) {
          console.log(`[experience] Background error: ${err.message}`);
        }
        try {
          const created = await skills.analyzeForSkills(historyForExtract, { sessionId, cwd });
          if (created.length > 0) {
            console.log(`[skills] Auto-created ${created.length} skill(s)`);
            send({ type: 'skill_created', skills: created.map(s => ({ name: s.name, description: s.description })) });
          }
        } catch (err) {
          console.log(`[skills] Background error: ${err.message}`);
        }
      })();
      argus.setToolLoopActive(false);
      if (isUserLoop && --_activeUserLoops <= 0) {
        _activeUserLoops = 0;
        while (_loopWaiters.length) _loopWaiters.shift()();
      }
      return;
    }

    // Execute tool calls
    send({ type: 'done_partial', stats: result.stats });

    // Normalize tool names: strip domain prefix, resolve aliases
    for (const tc of result.toolCalls) {
      if (tc.function?.name) {
        let n = tc.function.name;
        if (n.includes(':')) n = n.split(':').pop(); // strip "media:image_generate" → "image_generate"
        n = ALIASES[n] || n;
        tc.function.name = n;
      }
    }

    // Deduplicate tool calls — model sometimes emits the same call twice in one turn.
    // Compare by name + JSON-serialized args; keep the first occurrence.
    // Also cap at 2 calls to the same tool per turn to prevent retry-spam loops.
    const seenCalls = new Set();
    const toolNameCounts = {};
    const MAX_SAME_TOOL_PER_TURN = 2;
    const dedupedCalls = result.toolCalls.filter(tc => {
      const name = tc.function?.name || '';
      const args = JSON.stringify(tc.function?.arguments || {});
      const key = `${name}::${args}`;
      if (seenCalls.has(key)) {
        console.log(`[tool-loop] Skipping duplicate tool call: ${name}`);
        return false;
      }
      seenCalls.add(key);
      // Cap same-tool calls per turn (e.g. model retrying image_generate 5x)
      toolNameCounts[name] = (toolNameCounts[name] || 0) + 1;
      if (toolNameCounts[name] > MAX_SAME_TOOL_PER_TURN) {
        console.log(`[tool-loop] Capping ${name} at ${MAX_SAME_TOOL_PER_TURN} calls per turn`);
        return false;
      }
      return true;
    });

    // Tools that must run sequentially (confirmation, vision loop, state mutation)
    const SEQUENTIAL_TOOLS = new Set([
      'request_tools', 'computer', 'browser', 'screenshot',
      'process_kill', 'memory_delete',
    ]);

    // Determine if all tools in this batch can run in parallel.
    // Parallel requires: 2+ tools, none sequential, none needing confirmation.
    const resolvedCalls = dedupedCalls.map(tc => ({
      tc,
      name: ALIASES[tc.function?.name] || tc.function?.name || 'unknown',
      args: tc.function?.arguments || {},
    }));

    const canParallelize = resolvedCalls.length >= 2 && resolvedCalls.every(({ name, args }) =>
      !SEQUENTIAL_TOOLS.has(name)
      && !CONFIRM_TOOLS.has(name)
      && !(CONFIRM_ACTIONS[name] && CONFIRM_ACTIONS[name].has(args.action))
    );

    // Execute a single tool call and return its result object.
    // Shared between parallel and sequential paths.
    async function executeOne({ name: toolName, args: toolArgs }, toolId) {
      // Run pre_tool hooks (can block)
      const preHook = hooks.runHooks('pre_tool', { tool: toolName, args: toolArgs, sessionId, cwd });
      if (preHook.blocked) {
        send({ type: 'tool_result', id: toolId, name: toolName, output: `Blocked by hook: ${preHook.reason}`, status: 'blocked' });
        return { name: toolName, output: `Blocked by hook: ${preHook.reason}` };
      }

      // Execute the tool (with abort-race so Stop cancels hung tools)
      let output;
      try {
        const toolPromise = executeTool(toolName, toolArgs, cwd, {
          onStatus: (event) => send({ type: 'agent_status', ...event }),
        });
        if (signal) {
          const abortPromise = new Promise((_, reject) => {
            if (signal.aborted) return reject(new Error('Cancelled by user'));
            signal.addEventListener('abort', () => reject(new Error('Cancelled by user')), { once: true });
          });
          output = await Promise.race([toolPromise, abortPromise]);
        } else {
          output = await toolPromise;
        }
      } catch (err) {
        output = `Error: ${err.message}`;
      }

      // Run post_tool hooks (non-blocking, for logging/auditing)
      hooks.runHooks('post_tool', { tool: toolName, args: toolArgs, output: output?.slice(0, 4000), sessionId, cwd });

      if (signal?.aborted) {
        send({ type: 'tool_result', id: toolId, name: toolName, output: 'Cancelled by user', status: 'cancelled' });
        return null; // signals abort
      }

      // Post-processing: images, artifacts, PDFs, screenshots
      let browserScreenshot = null;

      if (toolName === 'image_generate' && output && output.includes('/artifacts/')) {
        const urlMatch = output.match(/View at: (\/artifacts\/[^\s]+)/);
        if (urlMatch) {
          const event = { type: 'image_generated', prompt: toolArgs.prompt || '', path: urlMatch[1] };
          send(event);
          appendMessage(sessionId, { role: 'display', display: 'image', prompt: event.prompt, path: event.path });
        }
      }

      if ((toolName === 'artifact' || toolName === 'document') && output && output.includes('/artifacts/')) {
        const urlMatch = output.match(/\/artifacts\/[^\s]+/);
        if (urlMatch) {
          const isDoc = toolName === 'document';
          const event = {
            type: isDoc ? 'document' : 'artifact',
            title: toolArgs.title || (isDoc ? 'Document' : 'Artifact'),
            path: urlMatch[0],
            artifactType: isDoc ? (toolArgs.format || 'txt') : (toolArgs.type || 'html'),
          };
          send(event);
          appendMessage(sessionId, { role: 'display', display: isDoc ? 'document' : 'artifact', title: event.title, path: event.path, artifactType: event.artifactType });
        }
      }

      if (toolName === 'pdf_export' && output && output.includes('/artifacts/')) {
        const urlMatch = output.match(/Download: (\/artifacts\/[^\s]+)/);
        if (urlMatch) {
          const event = {
            type: 'document',
            title: `${toolArgs.title || 'Export'} (PDF)`,
            path: urlMatch[1],
            artifactType: 'pdf',
          };
          send(event);
          appendMessage(sessionId, { role: 'display', display: 'document', title: event.title, path: event.path, artifactType: 'pdf' });
        }
      }

      if (toolName === 'computer' || toolName === 'screenshot') {
        usedComputerUse = true;
      }

      if ((toolName === 'browser' || toolName === 'computer' || toolName === 'screenshot') && output) {
        try {
          const parsed = JSON.parse(output);
          if (parsed.screenshot) {
            browserScreenshot = parsed.screenshot;
            send({ type: 'browser_screenshot', id: toolId, screenshot: parsed.screenshot, message: parsed.message });
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
        output: output.slice(0, 2000),
        status: 'done',
      });

      return { name: toolName, output, image: browserScreenshot };
    }

    const toolResults = [];

    if (canParallelize) {
      // ── Parallel execution: all tools run concurrently ──
      console.log(`[tool-loop] Executing ${resolvedCalls.length} tools in parallel`);
      const ids = resolvedCalls.map((_, i) =>
        `tc_${Date.now()}_${i}_${Math.random().toString(36).slice(2, 6)}`
      );
      // Notify client about all calls upfront
      for (let i = 0; i < resolvedCalls.length; i++) {
        send({ type: 'tool_call', name: resolvedCalls[i].name, args: resolvedCalls[i].args, id: ids[i], status: 'running' });
      }
      const promises = resolvedCalls.map((rc, i) => executeOne(rc, ids[i]));
      const results = await Promise.all(promises);
      for (const r of results) {
        if (r === null) break; // aborted
        toolResults.push(r);
      }
    } else {
      // ── Sequential execution: handles confirmation, vision, request_tools ──
      for (const rc of resolvedCalls) {
        if (signal?.aborted) break;

        const { name: toolName, args: toolArgs } = rc;
        const toolId = `tc_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

        send({ type: 'tool_call', name: toolName, args: toolArgs, id: toolId, status: 'running' });

        // Check if confirmation is needed
        const needsConfirm = CONFIRM_TOOLS.has(toolName)
          || (CONFIRM_ACTIONS[toolName] && CONFIRM_ACTIONS[toolName].has(toolArgs.action));
        if (needsConfirm) {
          if (onConfirmRequest) {
            const approved = await onConfirmRequest(toolId, toolName, toolArgs);
            if (!approved) {
              send({ type: 'tool_result', id: toolId, name: toolName, output: 'Skipped by user', status: 'skipped' });
              toolResults.push({ name: toolName, output: 'Tool execution skipped by user.' });
              continue;
            }
          }
        }

        // Progressive disclosure: handle request_tools inline
        if (toolName === 'request_tools') {
          const domain = (toolArgs.domain || '').toLowerCase();
          const activated = tiers.activateDomain(sessionId, domain);
          if (!activated) {
            const available = tiers.listDomains().map(d => d.name).join(', ');
            const output = `Error: unknown domain "${domain}". Available: ${available}, all`;
            send({ type: 'tool_result', id: toolId, name: toolName, output, status: 'done' });
            toolResults.push({ name: toolName, output });
            continue;
          }
          const newTools = [];
          for (const d of activated) {
            newTools.push(...tiers.domainToolList(d));
          }
          const output = `Activated domain(s): ${activated.join(', ')}\nNewly available tools: ${newTools.join(', ')}\n\nYou can now use these tools in this conversation.`;
          send({ type: 'tool_result', id: toolId, name: toolName, output, status: 'done' });
          toolResults.push({ name: toolName, output });
          toolDefsStale = true; // Rebuild tool defs on next turn to include new domain
          continue;
        }

        const r = await executeOne(rc, toolId);
        if (r === null) break; // aborted
        toolResults.push(r);

        // Vision tools: break so the model sees the screenshot before next action
        if (r.image && result.toolCalls.length > 1) {
          break;
        }
      }
    }

    // Save tool results to session (combined format matching CLI)
    const combinedResult = toolResults
      .map(r => `<tool_response name="${r.name}">\n${r.output}</tool_response>`)
      .join('\n');
    const toolMsg = { role: 'tool', content: combinedResult };
    appendMessage(sessionId, toolMsg);

    // Collect screenshots for transient injection on the next loop iteration.
    // NOT persisted to session — injected into the messages array at call time only.
    const screenshots = toolResults.filter(r => r.image).map(r => r.image);
    if (screenshots.length > 0) {
      pendingScreenshots = screenshots;
      // Check for click-loop warnings/blocks in computer tool results
      for (const r of toolResults) {
        if (r.name === 'computer' && r.output) {
          try {
            const parsed = JSON.parse(r.output);
            if (parsed.blocked) {
              // Hard block — force this into the transient user message
              pendingClickWarning = parsed.message;
            } else if (parsed.message && (parsed.message.includes('WARNING:') || parsed.message.includes('CLICK BLOCKED'))) {
              pendingClickWarning = parsed.message.slice(parsed.message.indexOf('⚠️'));
            }
          } catch {}
        }
      }
    }

    // Start next iteration (model sees tool results and may call more tools)
  }

  // Loop ended without a natural `done` — either max turns or signal abort
  argus.setToolLoopActive(false);
  if (isUserLoop && --_activeUserLoops <= 0) {
    _activeUserLoops = 0;
    while (_loopWaiters.length) _loopWaiters.shift()();
  }
  if (signal?.aborted) {
    send({ type: 'done', stats: {}, context: { used: tokensIn + tokensOut, max: MODEL_CTX, pct: Math.round((tokensIn + tokensOut) * 100 / MODEL_CTX) }, aborted: true });
  } else {
    send({ type: 'error', message: `Tool loop reached maximum of ${turnLimit} turns` });
  }
}

module.exports = { executeTool, runToolLoop, isUserLoopActive, waitForIdle, ALIASES, CONFIRM_TOOLS };
