// PRE Web GUI — Tool definitions for Ollama structured function calling
// Mirrors build_tools_json() from pre.m

const { getActiveConnections, isComfyUIInstalled } = require('./context');
const mcp = require('./mcp');

function buildToolDefs() {
  const tools = [
    tool('bash', 'Run a shell command and return output', {
      command: { type: 'string', description: 'The shell command to execute' },
    }, ['command']),

    tool('read_file', 'Read file contents from disk', {
      path: { type: 'string', description: 'File path to read' },
    }, ['path']),

    tool('list_dir', 'List directory contents', {
      path: { type: 'string', description: 'Directory path (default: cwd)' },
    }),

    tool('glob', 'Find files matching a glob pattern', {
      pattern: { type: 'string', description: 'Glob pattern (e.g. **/*.js)' },
      path: { type: 'string', description: 'Base directory (default: cwd)' },
    }, ['pattern']),

    tool('grep', 'Search file contents with regex', {
      pattern: { type: 'string', description: 'Regex pattern to search for' },
      path: { type: 'string', description: 'File or directory to search' },
      include: { type: 'string', description: 'File glob filter (e.g. *.py)' },
    }, ['pattern']),

    tool('file_write', 'Create or overwrite a file with content. Use this instead of bash/printf/cat for writing files.', {
      path: { type: 'string', description: 'File path to write' },
      content: { type: 'string', description: 'Full file content to write' },
    }, ['path', 'content']),

    tool('file_edit', 'Replace exact text in a file', {
      path: { type: 'string', description: 'File path to edit' },
      old_string: { type: 'string', description: 'Exact text to find (must match once)' },
      new_string: { type: 'string', description: 'Replacement text' },
    }, ['path', 'old_string', 'new_string']),

    tool('web_fetch', 'Fetch URL content (HTML/JSON/text)', {
      url: { type: 'string', description: 'URL to fetch' },
    }, ['url']),

    tool('system_info', 'Get system information (OS, CPU, memory, disk)', {}),
    tool('process_list', 'List running processes', {
      filter: { type: 'string', description: 'Filter processes by name' },
    }),
    tool('process_kill', 'Kill a process by PID', {
      pid: { type: 'string', description: 'Process ID to kill' },
    }, ['pid']),

    tool('clipboard_read', 'Read clipboard contents', {}),
    tool('clipboard_write', 'Write text to clipboard', {
      content: { type: 'string', description: 'Text to copy to clipboard' },
    }, ['content']),

    tool('open_app', 'Open a file, URL, or application', {
      target: { type: 'string', description: 'Path, URL, or app name to open' },
    }, ['target']),

    tool('notify', 'Show a macOS notification', {
      title: { type: 'string', description: 'Notification title' },
      message: { type: 'string', description: 'Notification message' },
    }, ['title', 'message']),

    tool('memory_save', 'Save a persistent memory', {
      name: { type: 'string', description: 'Memory name/key' },
      type: { type: 'string', description: 'Type: user|feedback|project|reference' },
      description: { type: 'string', description: 'One-line description' },
      content: { type: 'string', description: 'Memory content' },
      scope: { type: 'string', description: 'Scope: project|global (default: project)' },
    }, ['name', 'type', 'description', 'content']),

    tool('memory_search', 'Search saved memories', {
      query: { type: 'string', description: 'Search query' },
    }),
    tool('memory_list', 'List all saved memories', {}),
    tool('memory_delete', 'Delete a saved memory', {
      query: { type: 'string', description: 'Memory name or search query to delete' },
    }, ['query']),

    tool('screenshot', 'Take a screenshot', {
      region: { type: 'string', description: 'Region: full|window|selection (default: full)' },
    }),
    tool('window_list', 'List open windows', {}),
    tool('window_focus', 'Focus/activate an application window', {
      app: { type: 'string', description: 'Application name to focus' },
    }, ['app']),
    tool('display_info', 'Get display/screen information', {}),
    tool('net_info', 'Get network interface information', {}),
    tool('net_connections', 'List active network connections', {
      filter: { type: 'string', description: 'Filter connections' },
    }),
    tool('service_status', 'Check service/daemon status', {
      service: { type: 'string', description: 'Service name to check' },
    }),
    tool('disk_usage', 'Show disk usage', {
      path: { type: 'string', description: 'Path to check (default: /)' },
    }),
    tool('hardware_info', 'Get hardware details (CPU, GPU, memory)', {}),
    tool('applescript', 'Run an AppleScript', {
      script: { type: 'string', description: 'AppleScript code to execute' },
    }, ['script']),

    tool('document', 'Create a downloadable document file. Supports txt, xml, docx, xlsx, and pdf formats. For xlsx with structured data, pass a sheets array with {name, headers, rows}.', {
      title: { type: 'string', description: 'Document title / filename' },
      content: { type: 'string', description: 'Document content (text, markdown, XML, or pipe-delimited table for xlsx)' },
      format: { type: 'string', description: 'File format: txt, xml, docx, xlsx, or pdf' },
      sheets: { type: 'string', description: 'For xlsx: JSON array of {name, headers, rows} objects for multi-sheet workbooks' },
    }, ['title', 'content', 'format']),

    tool('artifact', 'Create an interactive HTML artifact (webpage, game, visualization, dashboard) that can be viewed in the browser', {
      title: { type: 'string', description: 'Title for the artifact' },
      content: { type: 'string', description: 'Full HTML content of the artifact' },
      type: { type: 'string', description: 'Type: html, svg, or markdown (default: html)' },
    }, ['title', 'content']),

    tool('pdf_export', 'Export an artifact to PDF for sharing', {
      title: { type: 'string', description: "Artifact title to export (or 'latest')" },
      path: { type: 'string', description: 'Output PDF path (optional)' },
    }, ['title']),

    tool('cron', 'Manage recurring scheduled tasks', {
      action: { type: 'string', description: 'Action: add|list|remove|enable|disable' },
      schedule: { type: 'string', description: '5-field cron schedule' },
      prompt: { type: 'string', description: 'Prompt to send when job triggers' },
      description: { type: 'string', description: 'Human-readable description' },
      id: { type: 'string', description: 'Job ID (for remove/enable/disable)' },
    }, ['action']),

    tool('spawn_agent', 'Spawn an autonomous sub-agent for research-heavy tasks. Use this when a task requires multiple searches, file reads, or web fetches that would clutter the main conversation. The agent runs independently, calls tools on its own, and returns a concise summary. Best for: deep research, codebase analysis, gathering background info.', {
      task: { type: 'string', description: 'Detailed task description — be specific about what to find and return' },
    }, ['task']),

    tool('spawn_multi', 'Run multiple research sub-agents sequentially, collecting results from each. Use when the user asks to compare, contrast, or investigate multiple independent topics. Each agent runs to completion before the next starts. Maximum 5 tasks. Progress is streamed to the user.', {
      tasks: { type: 'string', description: 'JSON array of task description strings, e.g. ["research X", "analyze Y"]' },
    }, ['tasks']),

    tool('list_agents', 'List all spawned sub-agents and their status (running, completed, failed)', {}),

    tool('experience_search', 'Search the experience ledger for lessons learned from past tasks. Use this before attempting complex tasks to check if you have relevant prior experience.', {
      query: { type: 'string', description: 'Search query describing the task or problem' },
    }, ['query']),

    tool('experience_list', 'List all entries in the experience ledger (lessons from past tasks)', {}),

    tool('memory_health', 'Check the health of the memory system: staleness report, aging warnings, maintenance status', {}),

    // macOS native apps (zero-config — works with whatever accounts are set up in macOS)
    tool('apple_mail', 'Send, read, and search email using the macOS Mail app. Works with any configured email account (iCloud, Gmail, Exchange, Outlook). No API keys needed — uses the native mail client.', {
      action: { type: 'string', description: 'Action: send|draft|search|read|list_recent|list_mailboxes|list_accounts' },
      to: { type: 'string', description: 'Recipient email address' },
      cc: { type: 'string', description: 'CC email address' },
      bcc: { type: 'string', description: 'BCC email address' },
      subject: { type: 'string', description: 'Email subject line' },
      body: { type: 'string', description: 'Email body text' },
      query: { type: 'string', description: 'Search query (subject or sender)' },
      id: { type: 'string', description: 'Message ID (from search/list results)' },
      mailbox: { type: 'string', description: 'Mailbox name (default: inbox)' },
      account: { type: 'string', description: 'Account name filter (optional)' },
      count: { type: 'integer', description: 'Max results (default: 15)' },
    }, ['action']),

    tool('apple_calendar', 'View and create calendar events using the macOS Calendar app. Works with any configured calendar (iCloud, Google, Exchange). No API keys needed.', {
      action: { type: 'string', description: 'Action: today|week|list_events|create_event|search|list_calendars|delete_event' },
      title: { type: 'string', description: 'Event title (for create_event)' },
      start: { type: 'string', description: 'Start date/time, e.g. "April 15, 2026 10:00:00 AM"' },
      end: { type: 'string', description: 'End date/time (defaults to 1 hour after start)' },
      location: { type: 'string', description: 'Event location' },
      notes: { type: 'string', description: 'Event notes/description' },
      calendar: { type: 'string', description: 'Calendar name (optional — defaults to first calendar)' },
      query: { type: 'string', description: 'Search query for event titles' },
      days: { type: 'integer', description: 'Number of days to show (default: 7)' },
      id: { type: 'string', description: 'Event UID (for delete_event)' },
    }, ['action']),

    tool('apple_contacts', 'Search and read contacts from the macOS Contacts app. Works with any synced account (iCloud, Google, Exchange). No API keys needed.', {
      action: { type: 'string', description: 'Action: search|read|list_groups|count' },
      query: { type: 'string', description: 'Search by name or organization' },
      name: { type: 'string', description: 'Contact name to search or read' },
      id: { type: 'string', description: 'Contact ID (from search results)' },
      count: { type: 'integer', description: 'Max search results (default: 20)' },
    }, ['action']),

    // Spotlight (mdfind — full-text search across the machine)
    tool('spotlight', 'Search the entire Mac using Spotlight (mdfind). Finds files by name, content, or metadata. Can filter by file type and folder. Dramatically faster than manual file searches.', {
      action: { type: 'string', description: 'Action: search|find_files|preview' },
      query: { type: 'string', description: 'Search query (searches file names AND content)' },
      folder: { type: 'string', description: 'Limit search to a specific folder path' },
      type: { type: 'string', description: 'File type filter: pdf|image|audio|video|email|presentation|spreadsheet|document|text|code|app' },
      path: { type: 'string', description: 'File path (for preview action — shows metadata)' },
      count: { type: 'integer', description: 'Max results (default: 20)' },
    }, ['action']),

    // macOS Reminders (zero-config)
    tool('apple_reminders', 'Create, list, complete, and search reminders using the macOS Reminders app. Works with any configured account (iCloud, Exchange). No API keys needed.', {
      action: { type: 'string', description: 'Action: add|list|complete|search|list_lists|delete' },
      title: { type: 'string', description: 'Reminder title (for add/complete)' },
      notes: { type: 'string', description: 'Reminder notes/description' },
      list: { type: 'string', description: 'Reminder list name (default: default list)' },
      due: { type: 'string', description: 'Due date, e.g. "April 15, 2026 10:00:00 AM"' },
      priority: { type: 'string', description: 'Priority: high|medium|low|none' },
      id: { type: 'string', description: 'Reminder ID (from search/list results)' },
      query: { type: 'string', description: 'Search query' },
      count: { type: 'integer', description: 'Max results (default: 25)' },
      completed: { type: 'boolean', description: 'Show completed reminders too (default: false)' },
    }, ['action']),

    // macOS Notes (zero-config)
    tool('apple_notes', 'Search, read, and create notes using the macOS Notes app. Works with any configured account (iCloud, Google, Exchange). No API keys needed.', {
      action: { type: 'string', description: 'Action: search|read|create|list_recent|list_folders' },
      title: { type: 'string', description: 'Note title (for create/read)' },
      body: { type: 'string', description: 'Note body content (for create)' },
      folder: { type: 'string', description: 'Folder name (for create/list_recent)' },
      query: { type: 'string', description: 'Search query (searches titles and content)' },
      id: { type: 'string', description: 'Note ID (from search/list results)' },
      count: { type: 'integer', description: 'Max results (default: 20)' },
    }, ['action']),
  ];

  // Conditional: computer use (desktop automation)
  const computerTool = require('./tools/computer');
  if (computerTool.isAvailable()) {
    tools.push(tool('computer', 'Control the desktop — click, type, scroll, and take screenshots of any application. Returns a screenshot after each action so you can see the result. Use this for automating desktop applications, filling forms, navigating GUIs, and any task that requires interacting with non-browser applications.', {
      action: { type: 'string', description: 'Action: screenshot|click|double_click|right_click|type|key|scroll|move|drag|screen_size|cursor_position' },
      x: { type: 'integer', description: 'X coordinate for click/move/scroll' },
      y: { type: 'integer', description: 'Y coordinate for click/move/scroll' },
      text: { type: 'string', description: 'Text to type' },
      key: { type: 'string', description: 'Key to press (return, tab, escape, space, cmd+c, ctrl+a, etc.)' },
      direction: { type: 'string', description: 'Scroll direction: up|down' },
      amount: { type: 'integer', description: 'Scroll amount (default: 3)' },
      from_x: { type: 'integer', description: 'Drag start X' },
      from_y: { type: 'integer', description: 'Drag start Y' },
      to_x: { type: 'integer', description: 'Drag end X' },
      to_y: { type: 'integer', description: 'Drag end Y' },
    }, ['action']));
  }

  // Conditional: browser
  const browserTool = require('./tools/browser');
  if (browserTool.isAvailable()) {
    tools.push(tool('browser', 'Control a headless Chrome browser. Navigate web pages, take screenshots, click elements, type text, read content. The browser returns screenshots as base64 images after each action so you can see what happened.', {
      action: { type: 'string', description: 'Action: navigate|screenshot|click|type|press|scroll|read|evaluate|select|back|forward|wait|close' },
      url: { type: 'string', description: 'URL to navigate to (for navigate)' },
      selector: { type: 'string', description: 'CSS selector for click/type/wait' },
      text: { type: 'string', description: 'Text to type, or text content to click on' },
      x: { type: 'integer', description: 'X coordinate for click' },
      y: { type: 'integer', description: 'Y coordinate for click' },
      key: { type: 'string', description: 'Key to press (Enter, Tab, Escape, etc.)' },
      direction: { type: 'string', description: 'Scroll direction: up|down' },
      amount: { type: 'integer', description: 'Scroll amount in pixels' },
      script: { type: 'string', description: 'JavaScript to evaluate in page context' },
      clear: { type: 'boolean', description: 'Clear field before typing' },
      full_page: { type: 'boolean', description: 'Take full-page screenshot' },
    }, ['action']));
  }

  // Conditional: image_generate
  if (isComfyUIInstalled()) {
    tools.push(tool('image_generate',
      'Generate an image locally on the GPU. INSTALLED AND OPERATIONAL. Returns a file path to the generated PNG.',
      {
        prompt: { type: 'string', description: 'Detailed image description' },
        width: { type: 'integer', description: 'Width in pixels (default: 1024, max: 1536)' },
        height: { type: 'integer', description: 'Height in pixels (default: 1024, max: 1536)' },
        style: { type: 'string', description: 'Optional style prefix: photorealistic, artistic, cartoon, illustration, cinematic' },
      }, ['prompt']));
  }

  // web_search: always available (Brave API if configured, DuckDuckGo fallback otherwise)
  const conns = getActiveConnections();
  tools.push(tool('web_search', 'Search the web (returns titles, URLs, and snippets)', {
    query: { type: 'string', description: 'Search query' },
    count: { type: 'integer', description: 'Number of results (default: 5)' },
  }, ['query']));

  if (conns.github) {
    tools.push(tool('github', 'Interact with GitHub', {
      action: { type: 'string', description: 'Action: list_repos|search_repos|list_issues|read_issue|list_prs|read_pr|user' },
      repo: { type: 'string', description: 'Repository (owner/name)' },
      query: { type: 'string', description: 'Search query' },
      number: { type: 'integer', description: 'Issue/PR number' },
      username: { type: 'string', description: 'GitHub username (omit for authenticated user)' },
      state: { type: 'string', description: 'State filter: open|closed|all' },
    }, ['action']));
  }

  if (conns.google) {
    tools.push(tool('gmail', 'Gmail operations', {
      action: { type: 'string', description: 'Action: search|read|send|draft|trash|labels|profile' },
      query: { type: 'string', description: 'Search query' },
      id: { type: 'string', description: 'Message/thread ID' },
      to: { type: 'string', description: 'Recipient email' },
      subject: { type: 'string', description: 'Email subject' },
      body: { type: 'string', description: 'Email body' },
      account: { type: 'string', description: 'Google account' },
    }, ['action']));

    tools.push(tool('gdrive', 'Google Drive operations', {
      action: { type: 'string', description: 'Action: list|search|download|upload|mkdir|share|delete' },
      id: { type: 'string', description: 'File/folder ID' },
      path: { type: 'string', description: 'Local file path' },
      name: { type: 'string', description: 'File name' },
      query: { type: 'string', description: 'Search query' },
      account: { type: 'string', description: 'Google account' },
    }, ['action']));

    tools.push(tool('gdocs', 'Google Docs operations', {
      action: { type: 'string', description: 'Action: create|read|append' },
      id: { type: 'string', description: 'Document ID' },
      title: { type: 'string', description: 'Document title' },
      content: { type: 'string', description: 'Content to write' },
      account: { type: 'string', description: 'Google account' },
    }, ['action']));
  }

  if (conns.jira) {
    tools.push(tool('jira', 'Interact with Jira Server', {
      action: { type: 'string', description: 'Action: search|get_issue|create_issue|comment|transition|assign|list_projects|get_project|my_issues' },
      key: { type: 'string', description: 'Issue key (e.g. PROJ-123)' },
      project: { type: 'string', description: 'Project key (e.g. PROJ)' },
      jql: { type: 'string', description: 'JQL query for search' },
      summary: { type: 'string', description: 'Issue summary' },
      description: { type: 'string', description: 'Issue description' },
      type: { type: 'string', description: 'Issue type (default: Task)' },
      priority: { type: 'string', description: 'Priority name' },
      assignee: { type: 'string', description: 'Assignee username' },
      status: { type: 'string', description: 'Target status for transition' },
      body: { type: 'string', description: 'Comment body text' },
      count: { type: 'integer', description: 'Max results (default: 20)' },
    }, ['action']));
  }

  if (conns.confluence) {
    tools.push(tool('confluence', 'Interact with Confluence Server wiki', {
      action: { type: 'string', description: 'Action: search|get_page|create_page|update_page|list_spaces|get_space|get_children|get_comments|add_comment' },
      id: { type: 'string', description: 'Page or content ID' },
      title: { type: 'string', description: 'Page title' },
      space: { type: 'string', description: 'Space key' },
      cql: { type: 'string', description: 'CQL query for search (e.g. type=page AND text~"keyword")' },
      query: { type: 'string', description: 'Search query text' },
      content: { type: 'string', description: 'Page content (Confluence storage format HTML or plain text)' },
      parent_id: { type: 'string', description: 'Parent page ID for creating child pages' },
      count: { type: 'integer', description: 'Max results (default: 20)' },
    }, ['action']));
  }

  if (conns.smartsheet) {
    tools.push(tool('smartsheet', 'Interact with Smartsheet', {
      action: { type: 'string', description: 'Action: list_sheets|get_sheet|search|create_sheet|add_rows|update_rows|delete_rows|get_columns|add_column|list_workspaces|get_workspace|add_comment|me' },
      id: { type: 'string', description: 'Sheet, workspace, or row ID' },
      name: { type: 'string', description: 'Sheet name (for get_sheet lookup or create_sheet)' },
      query: { type: 'string', description: 'Search query' },
      columns: { type: 'string', description: 'Comma-separated column names, or JSON array of {title, type} objects' },
      rows: { type: 'string', description: 'Pipe-delimited rows (val1|val2\\nval3|val4), or JSON array' },
      row_ids: { type: 'string', description: 'Comma-separated row IDs for deletion' },
      title: { type: 'string', description: 'Column title (for add_column)' },
      type: { type: 'string', description: 'Column type: TEXT_NUMBER, DATE, CONTACT_LIST, PICKLIST, CHECKBOX, etc.' },
      options: { type: 'string', description: 'Comma-separated picklist options' },
      text: { type: 'string', description: 'Comment text' },
      workspace_id: { type: 'string', description: 'Workspace ID (for creating sheets in a workspace)' },
      count: { type: 'integer', description: 'Max results (default: 25)' },
    }, ['action']));
  }

  if (conns.microsoft) {
    tools.push(tool('sharepoint', 'Interact with Microsoft SharePoint. Actions: search, list_sites, list_drives, list_files, read_file, upload_file, list_lists, list_items, get_page, site_usage, get_recent, get_columns, create_list_item, update_list_item, create_folder, get_file_metadata, move_file, copy_file, delete_file, list_subsites', {
      action: { type: 'string', description: 'Action to perform' },
      query: { type: 'string', description: 'Search query text' },
      site_id: { type: 'string', description: 'SharePoint site ID' },
      drive_id: { type: 'string', description: 'Drive ID' },
      folder_path: { type: 'string', description: 'Folder path (e.g. /General/Reports)' },
      item_id: { type: 'string', description: 'File, item, or drive item ID' },
      list_id: { type: 'string', description: 'List ID' },
      page_id: { type: 'string', description: 'Page ID' },
      fields: { type: 'string', description: 'JSON object of field values for create_list_item/update_list_item' },
      folder_name: { type: 'string', description: 'Name for new folder (create_folder)' },
      file_name: { type: 'string', description: 'Filename for upload' },
      content: { type: 'string', description: 'File content for upload' },
      dest_folder: { type: 'string', description: 'Destination folder path (upload, move, copy)' },
      dest_drive_id: { type: 'string', description: 'Destination drive ID (move/copy across drives)' },
      filename: { type: 'string', description: 'New filename (rename via move, or copy target name)' },
      filter: { type: 'string', description: 'OData filter for list_items' },
      count: { type: 'integer', description: 'Max results (default: 25)' },
    }, ['action']));
  }

  if (conns.slack) {
    tools.push(tool('slack', 'Interact with Slack', {
      action: { type: 'string', description: 'Action: list_channels|get_channel|history|send|reply|update|react|search|list_users|get_user|me' },
      channel: { type: 'string', description: 'Channel name (#general) or ID (C0123456)' },
      text: { type: 'string', description: 'Message text' },
      thread_ts: { type: 'string', description: 'Thread timestamp for replies' },
      ts: { type: 'string', description: 'Message timestamp (for update/react)' },
      emoji: { type: 'string', description: 'Emoji name for reactions (e.g. thumbsup)' },
      query: { type: 'string', description: 'Search query' },
      user: { type: 'string', description: 'User ID or @username' },
      count: { type: 'integer', description: 'Max results (default varies by action)' },
    }, ['action']));
  }

  if (conns.wolfram) {
    tools.push(tool('wolfram', 'Query Wolfram Alpha for computation/facts', {
      query: { type: 'string', description: 'Query for Wolfram Alpha' },
    }, ['query']));
  }

  if (conns.telegram) {
    tools.push(tool('telegram', 'Send messages via Telegram bot', {
      action: { type: 'string', description: 'Action: send|get_me|get_updates|get_chat' },
      chat_id: { type: 'string', description: 'Chat ID to send to (auto-detected from recent messages if omitted)' },
      text: { type: 'string', description: 'Message text to send' },
      parse_mode: { type: 'string', description: 'Parse mode: Markdown|HTML (default: Markdown)' },
    }, ['action']));
  }

  if (conns.linear) {
    tools.push(tool('linear', 'Interact with Linear project tracking. Actions: list_issues, search, get_issue, create_issue, update_issue, add_comment, list_projects, list_teams, list_cycles, list_labels, me', {
      action: { type: 'string', description: 'Action to perform' },
      query: { type: 'string', description: 'Search query' },
      id: { type: 'string', description: 'Issue ID or identifier (e.g. ENG-123)' },
      key: { type: 'string', description: 'Issue identifier (e.g. ENG-123)' },
      title: { type: 'string', description: 'Issue title (for create)' },
      description: { type: 'string', description: 'Issue description' },
      team: { type: 'string', description: 'Team key (e.g. ENG)' },
      project: { type: 'string', description: 'Project name' },
      status: { type: 'string', description: 'Status name or state ID' },
      priority: { type: 'string', description: 'Priority (0=none, 1=urgent, 2=high, 3=medium, 4=low)' },
      assignee: { type: 'string', description: 'Assignee ID' },
      body: { type: 'string', description: 'Comment body text' },
      count: { type: 'integer', description: 'Max results (default: 25)' },
    }, ['action']));
  }

  if (conns.zoom) {
    tools.push(tool('zoom', 'Interact with Zoom. Actions: list_meetings, get_meeting, create_meeting, update_meeting, delete_meeting, list_recordings, list_users, me', {
      action: { type: 'string', description: 'Action to perform' },
      id: { type: 'string', description: 'Meeting ID' },
      topic: { type: 'string', description: 'Meeting topic/title' },
      start_time: { type: 'string', description: 'Start time ISO 8601 (e.g. 2026-04-15T10:00:00Z)' },
      duration: { type: 'integer', description: 'Duration in minutes' },
      agenda: { type: 'string', description: 'Meeting agenda' },
      timezone: { type: 'string', description: 'Timezone (e.g. America/Los_Angeles)' },
      type: { type: 'string', description: 'Meeting type: scheduled|instant' },
      user_id: { type: 'string', description: 'User ID (default: me)' },
      from: { type: 'string', description: 'Start date for recordings (YYYY-MM-DD)' },
      to: { type: 'string', description: 'End date for recordings (YYYY-MM-DD)' },
      count: { type: 'integer', description: 'Max results' },
    }, ['action']));
  }

  if (conns.figma) {
    tools.push(tool('figma', 'Interact with Figma design files. Actions: get_file, get_file_nodes, get_comments, post_comment, list_projects, get_project_files, get_team_projects, get_file_versions, get_images, me', {
      action: { type: 'string', description: 'Action to perform' },
      file_key: { type: 'string', description: 'Figma file key (from URL: figma.com/file/<key>/...)' },
      ids: { type: 'string', description: 'Comma-separated node IDs' },
      team_id: { type: 'string', description: 'Team ID' },
      project_id: { type: 'string', description: 'Project ID' },
      message: { type: 'string', description: 'Comment message' },
      comment_id: { type: 'string', description: 'Comment ID (for replies)' },
      format: { type: 'string', description: 'Export format: png|jpg|svg|pdf' },
      scale: { type: 'string', description: 'Export scale (default: 2)' },
      depth: { type: 'integer', description: 'Traversal depth for get_file' },
    }, ['action']));
  }

  if (conns.asana) {
    tools.push(tool('asana', 'Interact with Asana project management. Actions: list_workspaces, list_projects, get_project, list_tasks, get_task, create_task, update_task, add_comment, search, list_sections, me', {
      action: { type: 'string', description: 'Action to perform' },
      id: { type: 'string', description: 'Task or project GID' },
      gid: { type: 'string', description: 'Asana GID (same as id)' },
      workspace: { type: 'string', description: 'Workspace GID' },
      project: { type: 'string', description: 'Project GID' },
      section: { type: 'string', description: 'Section GID' },
      team: { type: 'string', description: 'Team GID' },
      name: { type: 'string', description: 'Task name' },
      notes: { type: 'string', description: 'Task description' },
      assignee: { type: 'string', description: 'Assignee (GID or email)' },
      due_on: { type: 'string', description: 'Due date (YYYY-MM-DD)' },
      completed: { type: 'boolean', description: 'Mark task complete/incomplete' },
      query: { type: 'string', description: 'Search query' },
      text: { type: 'string', description: 'Comment text' },
      count: { type: 'integer', description: 'Max results (default: 25)' },
    }, ['action']));
  }

  // Workflow capture and replay
  tools.push(tool('workflow', 'Record, replay, and manage desktop automation workflows. Record a sequence of Computer Use actions, save as a replayable workflow, and replay on demand or schedule via cron. Actions: record (start capturing), stop (save recording), status, list, replay (re-execute), show (view steps), delete, export', {
    action: { type: 'string', description: 'Action: record, stop, status, list, replay, show, delete, export' },
    name: { type: 'string', description: 'Workflow name (for record, replay, show, delete, export)' },
    description: { type: 'string', description: 'Workflow description (for record)' },
    speed: { type: 'number', description: 'Replay speed multiplier (default: 1.0, 2.0 = double speed)' },
  }, ['action']));

  // Voice (speech-to-text / text-to-speech)
  if (require('./tools/voice').hasWhisper || require('./tools/voice').hasSay) {
    tools.push(tool('voice', 'Voice interface — speech-to-text via Whisper (local) and text-to-speech via macOS say. Actions: transcribe (audio → text), speak/say (text → audio), voices (list available), status (check capabilities)', {
      action: { type: 'string', description: 'Action: transcribe, speak, voices, status' },
      text: { type: 'string', description: 'Text to speak (for speak/say action)' },
      path: { type: 'string', description: 'Path to audio file (for transcribe action)' },
      audio_base64: { type: 'string', description: 'Base64-encoded audio data (for transcribe from browser)' },
      mime_type: { type: 'string', description: 'Audio MIME type (default: audio/webm)' },
      voice: { type: 'string', description: 'TTS voice name (default: Samantha)' },
      rate: { type: 'integer', description: 'TTS speaking rate in WPM (default: 185)' },
      output: { type: 'string', description: 'Output file path for TTS audio' },
      format: { type: 'string', description: 'Output format: aiff (default) or mp3' },
      model: { type: 'string', description: 'Whisper model: tiny.en, base.en, small.en, medium.en (default: base.en)' },
      language: { type: 'string', description: 'Language code for transcription (default: en)' },
    }, ['action']));
  }

  // Event-driven triggers
  tools.push(tool('trigger', 'Event-driven triggers that fire prompts in response to events (file changes, webhooks). Like cron but reactive instead of scheduled. Actions: add, list, remove, enable, disable', {
    action: { type: 'string', description: 'Action: add (create trigger), list (show all), remove (delete), enable, disable' },
    type: { type: 'string', description: 'Trigger type: file_watch (fires on file changes) or webhook (fires on HTTP POST)' },
    name: { type: 'string', description: 'Human-readable name for the trigger' },
    prompt: { type: 'string', description: 'Prompt to execute when triggered. Variables: {file} {event} {path} for file_watch; {payload} {headers} for webhook' },
    path: { type: 'string', description: 'Directory or file to watch (for file_watch type)' },
    glob: { type: 'string', description: 'Glob pattern to filter watched files (e.g. "*.log", "*.py")' },
    recursive: { type: 'boolean', description: 'Watch subdirectories recursively (default: true)' },
    debounce: { type: 'integer', description: 'Debounce interval in ms — batches rapid changes (default: 3000)' },
    secret: { type: 'string', description: 'Secret for webhook authentication (X-Webhook-Secret header)' },
    id: { type: 'string', description: 'Trigger ID (for remove/enable/disable)' },
  }, ['action']));

  // RAG (local document intelligence)
  tools.push(tool('rag', 'Local document RAG (Retrieval-Augmented Generation). Index files and directories into searchable vector databases for semantic search. Use this to find information across codebases, documents, notes, and any text files. Actions: index, search, list, status, delete', {
    action: { type: 'string', description: 'Action: index (index files/directory), search (semantic search), list (show all indexes), status (detailed index info), delete (remove an index)' },
    path: { type: 'string', description: 'File or directory path to index (for index action)' },
    query: { type: 'string', description: 'Natural language search query (for search action)' },
    index_name: { type: 'string', description: 'Index name (default: "default"). Use different names to organize separate document collections.' },
    top_k: { type: 'integer', description: 'Number of results to return (for search, default: 5, max: 20)' },
    min_score: { type: 'number', description: 'Minimum similarity threshold 0-1 (for search, default: 0.3)' },
    recursive: { type: 'boolean', description: 'Index subdirectories recursively (default: true)' },
    description: { type: 'string', description: 'Human-readable description for the index (for index action)' },
  }, ['action']));

  // Append MCP tools from all connected servers
  const mcpTools = mcp.getAllTools();
  tools.push(...mcpTools);

  return tools;
}

function tool(name, description, properties, required = []) {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: {
        type: 'object',
        properties,
        required,
        additionalProperties: false,
      },
    },
  };
}

module.exports = { buildToolDefs };
