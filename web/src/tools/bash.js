// PRE Web GUI — Bash tool execution
// Mirrors the CLI's bash tool with stderr capture

const { execSync } = require('child_process');
const path = require('path');
const { getShellPath } = require('../platform');

const MAX_OUTPUT = 65536;

function bash(args, cwd) {
  const command = args.command || args.cmd;
  if (!command) return 'Error: no command provided';

  try {
    // Execute with stderr merged into stdout (matches CLI's 2>&1 behavior)
    const output = execSync(command, {
      cwd,
      encoding: 'utf-8',
      maxBuffer: MAX_OUTPUT,
      timeout: 60000, // 60s timeout
      shell: getShellPath(),
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return output || '(no output)';
  } catch (err) {
    // execSync throws on non-zero exit — return combined stdout+stderr
    const stdout = err.stdout || '';
    const stderr = err.stderr || '';
    const combined = (stdout + stderr).trim();
    if (combined) return combined;
    return `Error: command failed with exit code ${err.status || 'unknown'}: ${err.message}`;
  }
}

module.exports = { bash };
