// PRE Web GUI — Progressive Tool Disclosure
// Organizes 70+ tools into domains. Only CORE tools load by default;
// additional domains activate via keyword detection or the request_tools meta-tool.
// Saves 40-60% of tool-definition tokens on most conversations.

// ── Domain Taxonomy ───────────────────────────────────────────────────────────

const CORE_TOOLS = new Set([
  'bash', 'read_file', 'list_dir', 'glob', 'grep',
  'file_write', 'file_edit', 'web_fetch', 'web_search',
  'memory_save', 'memory_search', 'memory_list', 'memory_delete',
  'system_info', 'spotlight', 'artifact', 'document',
  'screenshot', 'request_tools', 'session_search',
]);

const DOMAINS = {
  devops: [
    'process_list', 'process_kill', 'monitor', 'service_status',
    'net_info', 'net_connections', 'disk_usage', 'hardware_info',
  ],
  desktop: [
    'computer', 'window_list', 'window_focus', 'display_info',
    'clipboard_read', 'clipboard_write', 'open_app', 'notify',
  ],
  pim: [
    'apple_mail', 'apple_calendar', 'apple_contacts',
    'apple_reminders', 'apple_notes',
  ],
  code: [
    'applescript', 'powershell_script',
  ],
  media: [
    'image_generate', 'voice', 'pdf_export', 'browser',
  ],
  automation: [
    'cron', 'spawn_agent', 'spawn_multi', 'list_agents',
    'trigger', 'custom_tool', 'experience_search', 'experience_list',
    'memory_health', 'workflow', 'rag',
  ],
  cloud: [
    'github', 'gmail', 'gdrive', 'gdocs', 'jira', 'confluence',
    'smartsheet', 'slack', 'sharepoint', 'linear', 'zoom', 'figma',
    'asana', 'dynamics365', 'telegram', 'wolfram',
  ],
};

// ── Keyword → Domain Mapping ──────────────────────────────────────────────────
// Each keyword/phrase maps to one or more domains to auto-activate.

const KEYWORD_MAP = {
  // devops
  'process':    ['devops'], 'kill':        ['devops'], 'monitor':     ['devops'],
  'service':    ['devops'], 'network':     ['devops'], 'disk':        ['devops'],
  'hardware':   ['devops'], 'cpu':         ['devops'], 'memory usage':['devops'],
  'ram':        ['devops'], 'gpu':         ['devops'], 'port':        ['devops'],
  'daemon':     ['devops'], 'systemctl':   ['devops'], 'ps aux':      ['devops'],
  'connections':['devops'],
  // desktop
  'click':      ['desktop'], 'screenshot': ['desktop'], 'window':     ['desktop'],
  'clipboard':  ['desktop'], 'copy to clipboard': ['desktop'],
  'paste':      ['desktop'], 'notification':['desktop'], 'notify':    ['desktop'],
  'open app':   ['desktop'], 'launch app': ['desktop'], 'focus':     ['desktop'],
  'automate':   ['desktop', 'automation'],
  // pim
  'email':      ['pim'], 'mail':        ['pim'], 'calendar':    ['pim'],
  'contacts':   ['pim'], 'reminders':   ['pim'], 'reminder':    ['pim'],
  'notes':      ['pim'], 'meeting':     ['pim'], 'appointment': ['pim'],
  'schedule':   ['pim'], 'inbox':       ['pim'], 'send email':  ['pim'],
  'events':     ['pim'],
  // code
  'applescript':['code'], 'osascript':   ['code'], 'powershell':  ['code'],
  // media
  'generate image': ['media'], 'image':  ['media'], 'voice':      ['media'],
  'speak':      ['media'], 'transcribe': ['media'], 'tts':        ['media'],
  'stt':        ['media'], 'pdf export': ['media'], 'browse':     ['media'],
  'headless':   ['media'], 'puppeteer':  ['media'], 'whisper':    ['media'],
  // automation
  'cron':       ['automation'], 'agent':      ['automation'],
  'sub-agent':  ['automation'], 'subagent':   ['automation'],
  'spawn':      ['automation'], 'experience': ['automation'],
  'workflow':   ['automation'], 'rag':        ['automation'],
  'index':      ['automation'], 'custom tool':['automation'],
  'trigger':    ['automation'], 'webhook':    ['automation'],
  'file watch': ['automation'],
  // cloud
  'github':     ['cloud'], 'jira':       ['cloud'], 'confluence': ['cloud'],
  'smartsheet': ['cloud'], 'slack':      ['cloud'], 'sharepoint': ['cloud'],
  'linear':     ['cloud'], 'zoom':       ['cloud'], 'figma':      ['cloud'],
  'asana':      ['cloud'], 'dynamics':   ['cloud'], 'telegram':   ['cloud'],
  'wolfram':    ['cloud'], 'gdrive':     ['cloud'], 'google drive':['cloud'],
  'google docs':['cloud'], 'gmail api':  ['cloud'],
  'pull request':['cloud'], 'PR':        ['cloud'], 'issue':      ['cloud'],
};

// ── Per-Session State ─────────────────────────────────────────────────────────

const sessionDomains = new Map(); // sessionId → Set<domainName>

function getActiveDomains(sessionId) {
  if (!sessionDomains.has(sessionId)) {
    sessionDomains.set(sessionId, new Set());
  }
  return sessionDomains.get(sessionId);
}

function activateDomain(sessionId, domain) {
  if (domain === 'all') {
    const active = getActiveDomains(sessionId);
    for (const d of Object.keys(DOMAINS)) active.add(d);
    return Object.keys(DOMAINS);
  }
  if (!DOMAINS[domain]) return null;
  getActiveDomains(sessionId).add(domain);
  return [domain];
}

/**
 * Scan a user message for keywords and auto-activate matching domains.
 * Returns the set of newly activated domain names.
 */
function resolveKeywords(sessionId, userMessage) {
  if (!userMessage) return [];
  const lower = userMessage.toLowerCase();
  const active = getActiveDomains(sessionId);
  const newDomains = [];

  for (const [keyword, domains] of Object.entries(KEYWORD_MAP)) {
    if (lower.includes(keyword)) {
      for (const d of domains) {
        if (!active.has(d)) {
          active.add(d);
          if (!newDomains.includes(d)) newDomains.push(d);
        }
      }
    }
  }
  return newDomains;
}

/**
 * Check if a tool name is allowed given the current active domains.
 * CORE tools are always allowed. Domain tools need their domain active.
 * Tools not in any domain (conditional/dynamic) are always allowed.
 */
function isToolAllowed(toolName, activeDomains) {
  if (CORE_TOOLS.has(toolName)) return true;

  for (const [domain, tools] of Object.entries(DOMAINS)) {
    if (tools.includes(toolName)) {
      return activeDomains.has(domain);
    }
  }
  // Tool not in any domain (custom tools, MCP tools, etc.) — always allowed
  return true;
}

/**
 * Get a human-readable list of tools in a domain.
 */
function domainToolList(domain) {
  return DOMAINS[domain] || [];
}

/**
 * List all available domains with their tool counts.
 */
function listDomains() {
  return Object.entries(DOMAINS).map(([name, tools]) => ({
    name,
    tools: tools.length,
    toolNames: tools,
  }));
}

/**
 * Clean up session state (call on session delete/end).
 */
function clearSession(sessionId) {
  sessionDomains.delete(sessionId);
}

module.exports = {
  CORE_TOOLS,
  DOMAINS,
  KEYWORD_MAP,
  getActiveDomains,
  activateDomain,
  resolveKeywords,
  isToolAllowed,
  domainToolList,
  listDomains,
  clearSession,
};
