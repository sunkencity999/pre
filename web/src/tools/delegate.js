// PRE Web GUI — Delegate to frontier AI models
// Sends prompts to Claude Code, Codex, or Gemini CLI tools
// and captures the response for display in the GUI.

const { execSync, spawn } = require('child_process');
const path = require('path');

// CLI configurations for each delegate
const DELEGATES = {
  claude: {
    name: 'Claude',
    description: 'Anthropic Claude (Opus/Sonnet) via Claude Code CLI',
    command: 'claude',
    // -p = print mode, --output-format text = plain text, --tools "" = no tool use
    buildArgs: (prompt) => ['-p', prompt, '--output-format', 'text', '--tools', ''],
    icon: 'C',
    color: '#cc785c',
  },
  codex: {
    name: 'Codex',
    description: 'OpenAI Codex CLI',
    command: 'codex',
    // -q = quiet, -a suggest = don't auto-execute tools
    buildArgs: (prompt) => ['-q', '-a', 'suggest', prompt],
    // Codex -q echoes the user message as JSON to stdout before the response;
    // buffer output and strip leading JSON lines
    bufferOutput: true,
    cleanOutput: (raw) => {
      const lines = raw.split('\n');
      // Drop leading lines that look like JSON objects (user message echo)
      let start = 0;
      for (let i = 0; i < lines.length; i++) {
        const trimmed = lines[i].trim();
        if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
          start = i + 1;
        } else if (trimmed) {
          break;
        }
      }
      return lines.slice(start).join('\n').trim();
    },
    icon: 'X',
    color: '#10a37f',
  },
  gemini: {
    name: 'Gemini',
    description: 'Google Gemini CLI',
    command: 'gemini',
    // Positional prompt with text output format for clean output
    buildArgs: (prompt) => ['-o', 'text', prompt],
    // Gemini may prefix MCP server status messages (sometimes without newlines);
    // buffer and strip them
    bufferOutput: true,
    cleanOutput: (raw) => {
      // Strip MCP auth messages that may be concatenated without newlines
      let cleaned = raw;
      // Remove "MCP server 'name' requires authentication using: /mcp auth name" patterns
      // The server name after "/mcp auth" is always a single word matching the quoted name
      cleaned = cleaned.replace(/MCP server '([^']*)' requires authentication using: \/mcp auth \1/gi, '');
      // Remove any remaining MCP status lines
      cleaned = cleaned.split('\n').filter(line => {
        const t = line.trim();
        return !(t.startsWith('MCP server') || t.includes('/mcp auth'));
      }).join('\n');
      return cleaned.trim();
    },
    icon: 'G',
    color: '#4285f4',
  },
};

/**
 * Check which AI CLIs are installed and available
 * @returns {Object} { claude: { available, version, path }, ... }
 */
function checkAvailability() {
  const result = {};
  for (const [key, config] of Object.entries(DELEGATES)) {
    try {
      const cmdPath = execSync(`which ${config.command} 2>/dev/null`, { encoding: 'utf-8' }).trim();
      let version = '';
      try {
        version = execSync(`${config.command} --version 2>/dev/null`, { encoding: 'utf-8' }).trim().split('\n')[0];
      } catch {}
      result[key] = {
        available: true,
        path: cmdPath,
        version,
        name: config.name,
        description: config.description,
        icon: config.icon,
        color: config.color,
      };
    } catch {
      result[key] = {
        available: false,
        name: config.name,
        description: config.description,
        icon: config.icon,
        color: config.color,
        installHint: getInstallHint(key),
      };
    }
  }
  return result;
}

function getInstallHint(key) {
  switch (key) {
    case 'claude': return 'npm install -g @anthropic-ai/claude-code';
    case 'codex': return 'npm install -g @openai/codex';
    case 'gemini': return 'npm install -g @anthropic-ai/gemini-cli';
    default: return '';
  }
}

/**
 * Execute a prompt via a delegate AI CLI
 * @param {string} target - 'claude', 'codex', or 'gemini'
 * @param {string} prompt - The prompt to send
 * @param {Object} opts
 * @param {Function} opts.onToken - Called with partial output chunks
 * @param {AbortSignal} opts.signal - For cancellation
 * @param {number} opts.timeout - Timeout in ms (default: 5 minutes)
 * @returns {Promise<{response: string, duration: number}>}
 */
function execute(target, prompt, { onToken, signal, timeout = 300000 } = {}) {
  return new Promise((resolve, reject) => {
    const config = DELEGATES[target];
    if (!config) return reject(new Error(`Unknown delegate: ${target}`));

    // Verify CLI is available
    try {
      execSync(`which ${config.command} 2>/dev/null`, { encoding: 'utf-8' });
    } catch {
      return reject(new Error(`${config.name} CLI not installed. Install with: ${getInstallHint(target)}`));
    }

    const args = config.buildArgs(prompt);
    const startTime = Date.now();
    let output = '';
    let timedOut = false;

    const child = spawn(config.command, args, {
      env: { ...process.env, NO_COLOR: '1', TERM: 'dumb' },
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout,
    });

    // Handle abort signal
    if (signal) {
      const onAbort = () => {
        child.kill('SIGTERM');
        reject(new Error('Cancelled'));
      };
      signal.addEventListener('abort', onAbort, { once: true });
      child.on('exit', () => signal.removeEventListener('abort', onAbort));
    }

    // Timeout handler
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGTERM');
    }, timeout);

    child.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      output += text;
      if (onToken && !config.bufferOutput) onToken(text);
    });

    // Capture stderr but don't fail on it (CLIs often print progress to stderr)
    let stderr = '';
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('close', (code) => {
      clearTimeout(timer);
      const duration = Date.now() - startTime;

      if (timedOut) {
        return reject(new Error(`${config.name} timed out after ${Math.round(timeout / 1000)}s`));
      }

      // Clean up ANSI escape codes from output
      let cleaned = stripAnsi(output).trim();

      // Apply delegate-specific output cleaning
      if (config.cleanOutput) {
        cleaned = config.cleanOutput(cleaned);
      }

      if (code !== 0 && !cleaned) {
        // Extract a human-readable error from stderr (skip minified JS stack traces)
        const errMsg = extractErrorMessage(stderr) ||
          `exited with code ${code}`;
        return reject(new Error(`${config.name}: ${errMsg}`));
      }

      // For buffered delegates, send the cleaned output as a single token
      if (config.bufferOutput && onToken && cleaned) {
        onToken(cleaned);
      }

      resolve({ response: cleaned, duration });
    });

    child.on('error', (err) => {
      clearTimeout(timer);
      reject(new Error(`Failed to start ${config.name}: ${err.message}`));
    });

    // Close stdin immediately
    child.stdin.end();
  });
}

/**
 * Extract a human-readable error message from CLI stderr output.
 * CLIs often dump minified stack traces — find the actual error line.
 */
function extractErrorMessage(stderr) {
  if (!stderr) return '';
  const cleaned = stripAnsi(stderr).trim();
  // Look for common error patterns
  const patterns = [
    /Error:\s*(.+)/i,
    /error\s*[:\-]\s*(.+)/i,
    /quota/i,
    /unauthorized/i,
    /authentication/i,
    /API key/i,
    /rate limit/i,
    /not found/i,
  ];
  // Try to find a line with a recognizable error message
  const lines = cleaned.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    // Skip minified JS lines (very long with no spaces, or starting with file://)
    if (trimmed.startsWith('file://') || (trimmed.length > 200 && !trimmed.includes('. '))) continue;
    for (const pat of patterns) {
      const m = trimmed.match(pat);
      if (m) {
        // Return the matched error line, capped at 200 chars
        return trimmed.length > 200 ? trimmed.slice(0, 200) + '...' : trimmed;
      }
    }
  }
  // Fallback: first non-minified line under 200 chars
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith('file://') && trimmed.length < 200) {
      return trimmed;
    }
  }
  return cleaned.slice(0, 200);
}

/**
 * Strip ANSI escape codes from a string
 */
function stripAnsi(str) {
  // eslint-disable-next-line no-control-regex
  return str.replace(/\x1B\[[0-9;]*[a-zA-Z]/g, '').replace(/\x1B\][^\x07]*\x07/g, '');
}

module.exports = { checkAvailability, execute, DELEGATES };
