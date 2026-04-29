// PRE Web GUI — System prompt builder
// Mirrors the CLI's build_context_preamble() logic

const fs = require('fs');
const path = require('path');
const os = require('os');
const { PRE_DIR, CONNECTIONS_FILE, COMFYUI_FILE } = require('./constants');
const { buildMemoryContext: buildMemCtx, buildMemoryContextAsync, buildMemoryInstructions } = require('./memory');
const { buildExperienceContext, buildExperienceContextAsync } = require('./experience');
const { buildTemporalContext } = require('./chronos');

/**
 * Check which connections are active
 */
function getActiveConnections() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    return {
      brave: !!data.brave_search_key,
      github: !!data.github_key,
      google: !!data.google_client_id,
      microsoft: !!data.microsoft_client_id && !!data.microsoft_refresh_token,
      wolfram: !!data.wolfram_key,
      telegram: !!data.telegram_key,
      jira: !!data.jira_url && !!data.jira_token,
      confluence: !!data.confluence_url && !!data.confluence_token,
      smartsheet: !!data.smartsheet_token,
      slack: !!data.slack_token,
      linear: !!data.linear_token,
      zoom: !!data.zoom_account_id && !!data.zoom_client_id && !!data.zoom_client_secret,
      figma: !!data.figma_token,
      asana: !!data.asana_token,
      dynamics365: !!data.d365_url && !!data.d365_client_id && !!data.d365_tenant_id && (!!data.d365_client_secret || !!data.d365_refresh_token),
    };
  } catch {
    return { brave: false, github: false, google: false, microsoft: false, wolfram: false, telegram: false, jira: false, confluence: false, smartsheet: false, slack: false, linear: false, zoom: false, figma: false, asana: false, dynamics365: false };
  }
}

/**
 * Check if ComfyUI is installed
 */
function isComfyUIInstalled() {
  try {
    const data = JSON.parse(fs.readFileSync(COMFYUI_FILE, 'utf-8'));
    return data.installed === true;
  } catch {
    return false;
  }
}

/**
 * Build the full system prompt preamble
 */
function buildSystemPrompt(cwd) {
  const connections = getActiveConnections();
  const comfyui = isComfyUIInstalled();
  const memory = buildMemCtx();
  const memoryInstructions = buildMemoryInstructions();

  // Directory listing
  let filesList = '';
  try {
    const entries = fs.readdirSync(cwd);
    for (const entry of entries.slice(0, 40)) {
      if (entry.startsWith('.')) continue;
      try {
        const stat = fs.statSync(path.join(cwd, entry));
        if (stat.isDirectory()) {
          filesList += `  ${entry}/\n`;
        } else {
          const size = stat.size < 1024 ? `${stat.size}B`
            : stat.size < 1048576 ? `${(stat.size / 1024).toFixed(0)}K`
            : `${(stat.size / 1048576).toFixed(1)}M`;
          filesList += `  ${entry} (${size})\n`;
        }
      } catch {}
    }
  } catch {}

  // Date
  const now = new Date();
  const days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
  const months = ['January','February','March','April','May','June',
    'July','August','September','October','November','December'];
  const dateStr = `${days[now.getDay()]}, ${months[now.getMonth()]} ${now.getDate()}, ${now.getFullYear()}`;

  const { IS_WIN } = require('./platform');
  const platformDesc = IS_WIN
    ? 'running locally on Windows with GPU acceleration'
    : 'running on Apple Silicon';
  let prompt = `You are PRE (Personal Reasoning Engine), a fully local agentic assistant ${platformDesc}. `
    + `All data stays on this machine. You have persistent memory across sessions.\n\n`;

  if (memory) prompt += memory + '\n';
  prompt += memoryInstructions + '\n';

  // Experience ledger — lessons from past tasks
  const experienceCtx = buildExperienceContext();
  if (experienceCtx) prompt += experienceCtx + '\n';

  // Temporal awareness — stale memory warnings, time context
  const temporalCtx = buildTemporalContext();
  if (temporalCtx) prompt += temporalCtx + '\n';

  prompt += `<context>\nWorking directory: ${cwd}\nFiles:\n${filesList}</context>\n\n`;
  prompt += `Today is ${dateStr}. Use this when interpreting relative dates.\n\n`;

  // Rules
  prompt += `RULES (follow these exactly):\n`
    + `1. NEVER output code, HTML, or file contents in chat. Use tools instead.\n`
    + `2. ALL tools (including artifact) are native function calls. Call the artifact tool directly with title, content, and type.\n`
    + `   Do NOT describe or plan what you will build — call the artifact tool IMMEDIATELY with the full HTML content.\n`
    + `   NEVER output raw HTML, <artifact> tags, or code blocks in chat.\n`
    + `   NEVER use bash with printf/cat/echo to write files. Use the file_write tool.\n`
    + `3. One tool call per turn. STOP after each call and wait for the result.\n`
    + `4. For research: call web_search 5-8 times with DIFFERENT specific queries before writing. More searches = richer content.\n`
    + `5. For reports with images: web_search first, then image_generate for each image, then artifact last.\n`
    + `6. In HTML artifacts: load CDN scripts in <head>.\n`
    + `7. For long reports: write the COMPLETE report in a single artifact call. Do NOT truncate — you have ample token budget.\n`;

  if (comfyui) {
    prompt += `8. image_generate is a WORKING native function call. It creates photorealistic images on the local GPU in ~30-45 seconds. `
      + `ALWAYS call it when images are requested. NEVER use Unsplash or external URLs instead. `
      + `The image is automatically displayed in the chat after generation — just respond naturally describing what was created. `
      + `For reports/artifacts that need the image embedded, use the returned /artifacts/ path as the src.\n`;
  }

  // Computer Use guidance
  const computerTool = require('./tools/computer');
  if (computerTool.isAvailable()) {
    prompt += `\nCOMPUTER USE (Desktop Automation):\n`
      + `You have a \`computer\` tool that takes screenshots and controls the mouse/keyboard.\n`
      + `- For ANY task involving a desktop GUI app, use the computer tool's vision loop: screenshot → see screen → click/type/key → screenshot → repeat.\n`
      + `- Do NOT use applescript to inspect or enumerate GUI elements. AppleScript UI introspection is fragile and fails on most modern apps. Use the computer tool to SEE the screen instead.\n`
      + `- AppleScript is fine for non-visual tasks (open an app, activate a window), but NOT for reading GUI content.\n`
      + `- After opening an app, IMMEDIATELY take a screenshot to see what's on screen, then act on what you see.\n`
      + `- SEARCH FIRST: When looking for specific content in an app, ALWAYS use the search bar or Cmd+F FIRST:\n`
      + `  1. Look at the screenshot for a search bar, search icon (magnifying glass), or filter field\n`
      + `  2. If you see one, CLICK on it, then TYPE your search query, then press Return\n`
      + `  3. If no visible search bar, try Cmd+F (universal find shortcut) before scrolling manually\n`
      + `  4. Only browse/scroll manually if search is not available\n`
      + `- BEFORE TYPING: Always CLICK on the target text field or search bar first. Never assume a field has focus. Look at the screenshot, find the input field, click it, THEN type.\n`
      + `- MULTI-LINE TEXT: Use \\n in your type text for line breaks. Example: "Line one\\nLine two\\nLine three" — each \\n becomes a Return key press.\n`
      + `- DIALOGS & SAVE: When a dialog/popup appears (Save, Open, Confirm):\n`
      + `  • Press Return to activate the default (blue/highlighted) button — this is fastest and most reliable\n`
      + `  • Press Escape to cancel\n`
      + `  • For Save dialogs: use Cmd+S, then type the filename in the "Save As" field, then press Return to save\n`
      + `  • Do NOT try to click small dialog buttons — just use Return/Escape keyboard shortcuts\n`
      + `- Always take a screenshot after important actions (clicking, typing, pressing keys) to verify the result before proceeding.\n`
      + `- Keep your computer use sessions focused: screenshot → act → verify. Don't over-plan — react to what you see.\n`
      + `- AVOID CLICK LOOPS: If clicking an area does NOT produce results after 2 attempts, STOP and try a different approach:\n`
      + `  • Try a keyboard shortcut instead (e.g. Cmd+F for search, Return to confirm)\n`
      + `  • Click a different UI element nearby\n`
      + `  • Scroll to reveal hidden content\n`
      + `  • Take a fresh screenshot and reassess the entire screen layout\n`
      + `  • If truly stuck after 3 different approaches, STOP and report what you see to the user\n`
      + `- You have a LIMITED number of turns (~25). Be efficient — don't waste turns repeating failed actions.\n`
      + `- Once you have the information the user asked for, STOP taking actions and report your findings immediately.\n`;
  }

  // Experience ledger guidance
  prompt += `\nEXPERIENCE LEDGER:\n`
    + `You have an experience ledger that captures lessons from past tasks.\n`
    + `- Before attempting complex or multi-step tasks, use experience_search to check for relevant prior lessons\n`
    + `- The ledger updates automatically after tasks complete — you don't need to save lessons manually\n`
    + `- Use memory_health to check if any memories need verification or are becoming stale\n`;

  // Sub-agent guidance
  prompt += `\nSUB-AGENTS:\n`
    + `You have spawn_agent and spawn_multi tools for delegating research:\n`
    + `- Use spawn_agent when a task requires deep research (multiple searches/reads) that would consume too much of this conversation's context window\n`
    + `- Use spawn_multi when the user asks to compare, analyze, or investigate multiple independent topics (e.g. "compare frameworks A, B, and C") — agents run sequentially with progress updates\n`
    + `- Sub-agents keep the main context clean — their tool calls and outputs stay in their own session\n`
    + `- Do NOT use sub-agents for simple tasks (one file read, one search, one web fetch) — handle those directly, it's faster\n`
    + `- Each sub-agent has broad tool access: bash, file reading, web_fetch, web_search, browser (headless Chrome), cloud integrations, and more — only destructive tools (file_write, process_kill, etc.) are blocked\n`;

  // Report quality
  prompt += `\nREPORT QUALITY STANDARDS:\n`
    + `You run locally with NO per-token cost. Output should be COMPREHENSIVE — never truncate, abbreviate, or summarize prematurely.\n`
    + `When creating reports, analyses, or research documents:\n`
    + `\n`
    + `DEPTH & LENGTH:\n`
    + `- Reports must be REFERENCE-QUALITY material — thorough enough to serve as a standalone briefing document.\n`
    + `- Each major section needs 4-8 substantive paragraphs with specific facts, data points, named sources, dates, and figures.\n`
    + `- Include analysis, not just description: explain WHY things matter, compare alternatives, assess implications, identify risks.\n`
    + `- Add sub-sections, tables, and structured breakdowns where they add clarity.\n`
    + `- A good report is 3,000-8,000 words. Short summaries are only acceptable if the user explicitly asks for a summary.\n`
    + `- Do NOT stop early. Do NOT add "Further reading" or "For more information" cop-outs. Write the actual content.\n`
    + `- When using spawn_agent for research, instruct each agent to gather DETAILED findings with specific data points, not brief summaries.\n`
    + `\n`
    + `RESEARCH RIGOR:\n`
    + `- USE SPECIFIC DATA from web_search results: names, dates, dollar amounts, percentages, named sources.\n`
    + `- INCLUDE EVERY SECTION the user requested.\n`
    + `- CHARTS must use REAL DATA from research, not placeholder numbers.\n`
    + `- Cross-reference multiple sources. Note conflicting data where it exists.\n`
    + `- Include a timeline or chronology for evolving situations.\n`
    + `- End with pdf_export if the user requested PDF output.\n`
    + `\n`
    + `HTML QUALITY:\n`
    + `- Validate your HTML: matching quotes, no duplicate CSS keywords.\n`
    + `- STYLING: Use a clean, modern light theme. White/soft grey backgrounds (#fafafa, #ffffff), dark text (#1a1a2e), readable serif headings (Georgia), generous line-height (1.65+). NEVER use dark backgrounds or prefers-color-scheme:dark media queries. Tables: light header (#334155 or similar slate, white text), subtle striped rows (#f8fafc). Emphasize whitespace, typography, and readability.\n`;

  // Image embedding in artifacts
  prompt += `\nIMAGES IN ARTIFACTS:\n`
    + `When embedding images in HTML artifacts/reports:\n`
    + `- GENERATED IMAGES: After image_generate returns a path like "/artifacts/2026-04-13/image.png", use that EXACT path in <img src="...">. NEVER use placeholder URLs or CSS gradients instead.\n`
    + `- USER-UPLOADED IMAGES: When a user uploads an image, it is auto-saved and the path appears in the message (e.g. "/artifacts/uploads/upload-123.png"). Use that path in <img src="...">.\n`
    + `- WEB IMAGES: Use web_fetch to download images from URLs. If the response is an image, it will be saved locally and a /artifacts/ path returned. Use that local path, not the original URL.\n`
    + `- NEVER use placeholder gradient divs, broken external URLs, or "image not available" fallbacks when you have an actual local image path.\n`
    + `- ALWAYS use relative paths starting with /artifacts/ for locally stored images.\n`;

  // Live artifacts
  prompt += `\nLIVE ARTIFACTS (auto-refreshing dashboards):\n`
    + `When the user asks for a dashboard, status board, or any artifact that should show CURRENT data:\n`
    + `- Create an HTML artifact with JavaScript that fetches from PRE's live data API endpoints.\n`
    + `- Each endpoint returns { data: <structured JSON>, timestamp: ISO8601 }.\n`
    + `- Endpoints and their data shapes:\n`
    + `  /api/live/calendar → data: [{title, date ("YYYY-MM-DD"), start (ISO datetime), end (ISO datetime), start_time ("HH:MM"), end_time ("HH:MM"), calendar}, ...]\n`
    + `  /api/live/calendar/week → same shape as /calendar\n`
    + `  /api/live/mail?count=N → data: [{unread, date, from, subject}, ...]\n`
    + `  /api/live/reminders → data: [{title, due, list, priority}, ...]\n`
    + `  /api/live/system → data: {hostname, os, cpu, cpu_usage (number %), memory, memory_usage (number %), disk_usage (number %), uptime, battery_percent (number or null), battery_state ("charging"|"discharging"|"unknown")}\n`
    + `  /api/live/sessions → data: [{id, name, modified, turns}, ...]\n`
    + `- Include a refresh button AND an auto-refresh interval (e.g. every 60s).\n`
    + `- Show a "Last updated: <timestamp>" indicator.\n`
    + `- Example: fetch("/api/live/calendar").then(r=>r.json()).then(d=> d.data.forEach(e=> /* e.title, e.start */))\n`
    + `- The artifact is served from the same origin (localhost:7749), so no CORS issues.\n`;

  // Connection status
  const hasAny = connections.brave || connections.github || connections.google || connections.wolfram;
  if (!hasAny) {
    prompt += `\n- Additional tools (web_search, github, gmail, gdrive, wolfram) require /connections setup.\n`;
  } else {
    if (!connections.brave) prompt += `- web_search available via /connections add brave_search.\n`;
    if (!connections.github) prompt += `- github available via /connections add github.\n`;
    if (!connections.google) prompt += `- gmail/gdrive/gdocs available via /connections add google.\n`;
    if (!connections.wolfram) prompt += `- wolfram available via /connections add wolfram.\n`;
  }

  return prompt;
}

/**
 * Async version: uses embedding-based relevance ranking for memory and experience context.
 * When a userQuery is provided, memories and experiences are ranked by semantic similarity
 * to the current conversation, surfacing the most relevant context instead of dumping all.
 */
async function buildSystemPromptAsync(cwd, userQuery) {
  // Build the base prompt synchronously (connections, rules, etc.)
  const basePrompt = buildSystemPrompt(cwd);

  if (!userQuery) return basePrompt;

  // Replace the sync memory/experience blocks with relevance-ranked versions
  try {
    const [rankedMemory, rankedExperience] = await Promise.all([
      buildMemoryContextAsync(null, userQuery),
      buildExperienceContextAsync(userQuery),
    ]);

    let prompt = basePrompt;

    // Replace the memory block if we got a ranked version
    if (rankedMemory) {
      const memStart = prompt.indexOf('<memory>');
      const memEnd = prompt.indexOf('</memory>');
      if (memStart >= 0 && memEnd >= 0) {
        prompt = prompt.slice(0, memStart) + rankedMemory + prompt.slice(memEnd + '</memory>'.length);
      }
    }

    // Replace the experience block if we got a ranked version
    if (rankedExperience) {
      const expStart = prompt.indexOf('<experience_ledger>');
      const expEnd = prompt.indexOf('</experience_ledger>');
      if (expStart >= 0 && expEnd >= 0) {
        prompt = prompt.slice(0, expStart) + rankedExperience + prompt.slice(expEnd + '</experience_ledger>\n'.length);
      }
    }

    return prompt;
  } catch {
    // If embedding fails, return the base prompt with default memory/experience
    return basePrompt;
  }
}

module.exports = { buildSystemPrompt, buildSystemPromptAsync, getActiveConnections, isComfyUIInstalled };
