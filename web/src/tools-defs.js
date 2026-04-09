// PRE Web GUI — Tool definitions for Ollama structured function calling
// Mirrors build_tools_json() from pre.m

const { getActiveConnections, isComfyUIInstalled } = require('./context');

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
  ];

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

  // Conditional: web_search
  const conns = getActiveConnections();
  if (conns.brave) {
    tools.push(tool('web_search', 'Search the web', {
      query: { type: 'string', description: 'Search query' },
      count: { type: 'integer', description: 'Number of results (default: 5)' },
    }, ['query']));
  }

  if (conns.github) {
    tools.push(tool('github', 'Interact with GitHub', {
      action: { type: 'string', description: 'Action: search_repos|list_issues|read_issue|list_prs|user' },
      repo: { type: 'string', description: 'Repository (owner/name)' },
      query: { type: 'string', description: 'Search query' },
      number: { type: 'integer', description: 'Issue/PR number' },
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

  if (conns.wolfram) {
    tools.push(tool('wolfram', 'Query Wolfram Alpha for computation/facts', {
      query: { type: 'string', description: 'Query for Wolfram Alpha' },
    }, ['query']));
  }

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
