// PRE Web GUI — File operation tools
// read_file, list_dir, glob, grep, file_write, file_edit

const fs = require('fs');
const pathMod = require('path');
const { execSync } = require('child_process');

const MAX_FILE_SIZE = 512 * 1024; // 512KB read limit

function resolvePath(p, cwd) {
  if (!p) return cwd;
  if (pathMod.isAbsolute(p)) return p;
  if (p.startsWith('~/')) return pathMod.join(require('os').homedir(), p.slice(2));
  return pathMod.resolve(cwd, p);
}

function readFile(args, cwd) {
  const filePath = resolvePath(args.path, cwd);
  try {
    const stat = fs.statSync(filePath);
    if (stat.size > MAX_FILE_SIZE) {
      const content = fs.readFileSync(filePath, 'utf-8').slice(0, MAX_FILE_SIZE);
      return content + `\n\n[...truncated ${stat.size - MAX_FILE_SIZE} bytes — use bash to see full file]`;
    }
    return fs.readFileSync(filePath, 'utf-8');
  } catch (err) {
    return `Error: cannot read file '${args.path}': ${err.message}`;
  }
}

function listDir(args, cwd) {
  const dirPath = resolvePath(args.path, cwd);
  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    const lines = [];
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      try {
        const full = pathMod.join(dirPath, entry.name);
        const stat = fs.statSync(full);
        if (entry.isDirectory()) {
          lines.push(`  ${entry.name}/`);
        } else {
          const size = stat.size < 1024 ? `${stat.size}B`
            : stat.size < 1048576 ? `${(stat.size / 1024).toFixed(0)}K`
            : `${(stat.size / 1048576).toFixed(1)}M`;
          lines.push(`  ${entry.name} (${size})`);
        }
      } catch {}
    }
    return lines.join('\n') || '(empty directory)';
  } catch (err) {
    return `Error: cannot list directory '${args.path}': ${err.message}`;
  }
}

function glob(args, cwd) {
  const pattern = args.pattern;
  if (!pattern) return 'Error: no pattern provided';

  const basePath = resolvePath(args.path, cwd);
  try {
    // Use find + shell globbing for cross-platform compatibility
    const output = execSync(
      `find "${basePath}" -path "*${pattern.replace(/\*/g, '*')}" -maxdepth 5 2>/dev/null | head -100`,
      { encoding: 'utf-8', timeout: 10000 }
    ).trim();
    return output || 'No matches found';
  } catch {
    return 'No matches found';
  }
}

function grep(args, cwd) {
  const pattern = args.pattern;
  if (!pattern) return 'Error: no pattern provided';

  const searchPath = resolvePath(args.path, cwd);
  const include = args.include;

  try {
    let cmd = `grep -rn`;
    if (include) cmd += ` --include='${include}'`;
    cmd += ` '${pattern.replace(/'/g, "'\\''")}' '${searchPath}' 2>/dev/null | head -100`;

    const output = execSync(cmd, { encoding: 'utf-8', timeout: 15000 }).trim();
    return output || 'No matches found';
  } catch {
    return 'No matches found';
  }
}

function fileWrite(args, cwd) {
  const filePath = resolvePath(args.path, cwd);
  const content = args.content;
  if (!content && content !== '') return 'Error: no content provided';

  try {
    // Ensure parent directory exists
    const dir = pathMod.dirname(filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(filePath, content, 'utf-8');
    const size = Buffer.byteLength(content, 'utf-8');
    return `File written: ${filePath} (${size} bytes)`;
  } catch (err) {
    return `Error: cannot write file '${args.path}': ${err.message}`;
  }
}

function fileEdit(args, cwd) {
  const filePath = resolvePath(args.path, cwd);
  const oldString = args.old_string;
  const newString = args.new_string;

  if (!oldString) return 'Error: old_string is required';
  if (newString === undefined) return 'Error: new_string is required';

  try {
    let content = fs.readFileSync(filePath, 'utf-8');
    const count = content.split(oldString).length - 1;

    if (count === 0) return `Error: old_string not found in ${args.path}`;
    if (count > 1) return `Error: old_string found ${count} times (must be unique)`;

    content = content.replace(oldString, newString);
    fs.writeFileSync(filePath, content, 'utf-8');
    return `File edited: ${filePath}`;
  } catch (err) {
    return `Error: cannot edit file '${args.path}': ${err.message}`;
  }
}

module.exports = { readFile, listDir, glob, grep, fileWrite, fileEdit };
