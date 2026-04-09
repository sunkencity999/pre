// PRE Web GUI — System prompt builder
// Mirrors the CLI's build_context_preamble() logic

const fs = require('fs');
const path = require('path');
const os = require('os');
const { PRE_DIR, CONNECTIONS_FILE, COMFYUI_FILE } = require('./constants');
const { buildMemoryContext: buildMemCtx, buildMemoryInstructions } = require('./memory');

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
      wolfram: !!data.wolfram_key,
      telegram: !!data.telegram_key,
      jira: !!data.jira_url && !!data.jira_token,
      confluence: !!data.confluence_url && !!data.confluence_token,
      smartsheet: !!data.smartsheet_token,
      slack: !!data.slack_token,
    };
  } catch {
    return { brave: false, github: false, google: false, wolfram: false, telegram: false, jira: false, confluence: false, smartsheet: false, slack: false };
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

  let prompt = `You are PRE (Personal Reasoning Engine), a fully local agentic assistant running on Apple Silicon. `
    + `All data stays on this machine. You have persistent memory across sessions.\n\n`;

  if (memory) prompt += memory + '\n';
  prompt += memoryInstructions + '\n';

  prompt += `<context>\nWorking directory: ${cwd}\nFiles:\n${filesList}</context>\n\n`;
  prompt += `Today is ${dateStr}. Use this when interpreting relative dates.\n\n`;

  // Rules
  prompt += `RULES (follow these exactly):\n`
    + `1. NEVER output code, HTML, or file contents in chat. Use tools instead.\n`
    + `2. artifact uses text-based <tool_call> tags (not function calls):\n`
    + `   <tool_call>\n`
    + `   {"name": "artifact", "arguments": {"title": "...", "content": "...HTML...", "type": "html"}}\n`
    + `   </tool_call>\n`
    + `   All other tools (including file_write) are native function calls.\n`
    + `   NEVER use bash with printf/cat/echo to write files. Use the file_write tool.\n`
    + `3. One tool call per turn. STOP after each call and wait for the result.\n`
    + `4. For research: call web_search 3-5 times with DIFFERENT specific queries before writing.\n`
    + `5. For reports with images: web_search first, then image_generate for each image, then artifact last.\n`
    + `6. In HTML artifacts: load CDN scripts in <head>.\n`
    + `7. For long reports: use append_to to add sections to an existing artifact.\n`;

  if (comfyui) {
    prompt += `8. image_generate is a WORKING native function call. It creates photorealistic images on the local GPU in ~30-45 seconds. `
      + `ALWAYS call it when images are requested. NEVER use Unsplash or external URLs instead. `
      + `After generating, use the returned path in artifacts: <img src='file:///path/from/tool'>\n`;
  }

  // Report quality
  prompt += `\nREPORT QUALITY STANDARDS:\n`
    + `When creating reports or documents:\n`
    + `- USE SPECIFIC DATA from your web_search results: names, dates, numbers, quotes.\n`
    + `- INCLUDE EVERY SECTION the user requested.\n`
    + `- CHARTS must use REAL DATA from research, not placeholder numbers.\n`
    + `- Each section needs 2-3 substantive paragraphs minimum with specific facts.\n`
    + `- End with pdf_export if the user requested PDF output.\n`
    + `- Validate your HTML: matching quotes, no duplicate CSS keywords.\n`;

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

module.exports = { buildSystemPrompt, getActiveConnections, isComfyUIInstalled };
