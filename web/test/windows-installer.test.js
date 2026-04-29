// Tests for Windows installer and startup script structure
// Validates file contents and PowerShell syntax patterns without executing them.

const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..', '..');

describe('install.ps1', () => {
  const installPath = path.join(REPO_ROOT, 'install.ps1');
  let content;

  beforeAll(() => {
    content = fs.readFileSync(installPath, 'utf-8');
  });

  test('file exists', () => {
    expect(fs.existsSync(installPath)).toBe(true);
  });

  test('is a PowerShell script with param block', () => {
    expect(content).toContain('param(');
    expect(content).toContain('[switch]$Yes');
  });

  test('checks system requirements', () => {
    expect(content).toContain('Checking system requirements');
    expect(content).toContain('Win32_OperatingSystem');
    expect(content).toContain('nvidia-smi');
    expect(content).toContain('TotalPhysicalMemory');
  });

  test('handles Ollama installation', () => {
    expect(content).toContain('Checking Ollama');
    expect(content).toContain('ollama');
    expect(content).toContain('winget install Ollama.Ollama');
  });

  test('configures Ollama environment variables', () => {
    expect(content).toContain('OLLAMA_KEEP_ALIVE');
    expect(content).toContain('OLLAMA_NUM_PARALLEL');
    expect(content).toContain('OLLAMA_MAX_LOADED_MODELS');
    expect(content).toContain('SetEnvironmentVariable');
  });

  test('pulls base model', () => {
    expect(content).toContain('gemma4:26b-a4b-it-q8_0');
    expect(content).toContain('ollama pull');
  });

  test('pulls embedding model', () => {
    expect(content).toContain('nomic-embed-text');
  });

  test('creates custom model from Modelfile', () => {
    expect(content).toContain('pre-gemma4');
    expect(content).toContain('ollama create');
    expect(content).toContain('Modelfile');
  });

  test('checks/installs Node.js', () => {
    expect(content).toContain('Checking Node.js');
    expect(content).toContain('OpenJS.NodeJS.LTS');
  });

  test('runs npm install for web GUI', () => {
    expect(content).toContain('npm install');
  });

  test('creates ~/.pre/ directory structure', () => {
    expect(content).toContain('.pre');
    expect(content).toContain('sessions');
    expect(content).toContain('memory');
    expect(content).toContain('artifacts');
    expect(content).toContain('rag');
    expect(content).toContain('workflows');
  });

  test('auto-sizes context window', () => {
    expect(content).toContain('131072');  // 128K
    expect(content).toContain('65536');   // 64K
    expect(content).toContain('32768');   // 32K
    expect(content).toContain('16384');   // 16K
    expect(content).toContain('8192');    // 8K
    expect(content).toContain('context');
  });

  test('creates default config files', () => {
    expect(content).toContain('hooks.json');
    expect(content).toContain('mcp.json');
    expect(content).toContain('cron.json');
    expect(content).toContain('triggers.json');
  });

  test('offers optional voice setup', () => {
    expect(content).toContain('Whisper');
    expect(content).toContain('FFmpeg');
  });

  test('offers optional auto-start', () => {
    expect(content).toContain('PRE-Server.vbs');
    expect(content).toContain('pre-server.ps1');
    expect(content).toContain('Startup');
  });

  test('pre-warms the model', () => {
    expect(content).toContain('Pre-warming model');
    expect(content).toContain('/api/generate');
    expect(content).toContain('num_ctx');
  });

  test('creates pre.cmd launcher', () => {
    expect(content).toContain('pre.cmd');
    expect(content).toContain('.local\\bin');
  });

  test('does not reference macOS-specific tools', () => {
    expect(content).not.toContain('xcode');
    expect(content).not.toContain('swiftc');
    expect(content).not.toContain('launchctl');
    expect(content).not.toContain('brew install');
    expect(content).not.toContain('Apple Silicon');
  });

  test('has matching context sizes with install.sh', () => {
    const installSh = fs.readFileSync(path.join(REPO_ROOT, 'install.sh'), 'utf-8');
    // Both installers should use the same RAM→context mapping
    expect(installSh).toContain('131072');
    expect(installSh).toContain('65536');
    expect(installSh).toContain('32768');
    expect(installSh).toContain('16384');
    expect(installSh).toContain('8192');
  });
});

describe('pre-server.ps1', () => {
  const ps1Path = path.join(REPO_ROOT, 'web', 'pre-server.ps1');
  let content;

  beforeAll(() => {
    content = fs.readFileSync(ps1Path, 'utf-8');
  });

  test('file exists', () => {
    expect(fs.existsSync(ps1Path)).toBe(true);
  });

  test('supports --status flag', () => {
    expect(content).toContain('$status');
    expect(content).toContain('is running');
    expect(content).toContain('is not running');
  });

  test('supports --stop flag', () => {
    expect(content).toContain('$stop');
    expect(content).toContain('Stop-Process');
  });

  test('starts Ollama if not running', () => {
    expect(content).toContain('Start-Process ollama');
    expect(content).toContain('ArgumentList "serve"');
    expect(content).toContain('WindowStyle Hidden');
  });

  test('pre-warms model', () => {
    expect(content).toContain('Pre-warming model');
    expect(content).toContain('pre-gemma4');
    expect(content).toContain('num_ctx');
  });

  test('starts Node.js web server', () => {
    expect(content).toContain('node server.js');
    expect(content).toContain('web server');
  });

  test('uses correct port', () => {
    expect(content).toContain('7749');
  });

  test('reads context window from ~/.pre/context', () => {
    expect(content).toContain('.pre');
    expect(content).toContain('context');
    expect(content).toContain('131072');  // fallback
  });

  test('sets Ollama environment variables', () => {
    expect(content).toContain('OLLAMA_KEEP_ALIVE');
    expect(content).toContain('OLLAMA_NUM_PARALLEL');
    expect(content).toContain('OLLAMA_MAX_LOADED_MODELS');
  });
});

describe('server.js autostart endpoints', () => {
  test('server.js contains Windows autostart constants', () => {
    const serverJs = fs.readFileSync(path.join(REPO_ROOT, 'web', 'server.js'), 'utf-8');
    expect(serverJs).toContain('WIN_STARTUP_DIR');
    expect(serverJs).toContain('WIN_STARTUP_VBS');
    expect(serverJs).toContain('PRE_SERVER_PS1');
  });

  test('server.js has both macOS and Windows autostart branches', () => {
    const serverJs = fs.readFileSync(path.join(REPO_ROOT, 'web', 'server.js'), 'utf-8');
    // Windows branch
    expect(serverJs).toContain('if (IS_WIN)');
    expect(serverJs).toContain('VBScript');
    // macOS branch
    expect(serverJs).toContain('PLIST_PATH');
    expect(serverJs).toContain('launchctl');
  });
});
