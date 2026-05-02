// Tests for Windows-specific features across the PRE codebase
// Covers: web.js native https, powershellScript, native app dispatchers,
// sessions filename translation, autostart VBScript, delegate platform-awareness

const fs = require('fs');
const path = require('path');
const http = require('http');

jest.mock('../src/constants');
jest.mock('../src/ollama', () => ({
  streamChat: jest.fn().mockResolvedValue({ response: '[]' }),
}));
jest.mock('../src/mcp', () => ({
  getConnectedTools: jest.fn().mockReturnValue([]),
  getAllTools: jest.fn().mockReturnValue([]),
  loadConfig: jest.fn().mockReturnValue({ servers: {} }),
}));

const { CONNECTIONS_FILE, SESSIONS_DIR } = require('../src/constants');

// ═══════════════════════════════════════════════════════════════════════════
// Web Tool — Native HTTPS (no curl, no shell)
// ═══════════════════════════════════════════════════════════════════════════

describe('web tool — native https implementation', () => {
  beforeEach(() => {
    fs.writeFileSync(CONNECTIONS_FILE, '{}');
  });

  const web = require('../src/tools/web');

  test('exports webFetch and webSearch as async functions', () => {
    expect(typeof web.webFetch).toBe('function');
    expect(typeof web.webSearch).toBe('function');
  });

  test('webFetch returns error for missing URL', async () => {
    const result = await web.webFetch({});
    expect(result).toContain('Error');
    expect(result).toContain('no url');
  });

  test('webFetch returns error for non-HTTP URL', async () => {
    const result = await web.webFetch({ url: 'ftp://example.com' });
    expect(result).toContain('Error');
    expect(result).toContain('http');
  });

  test('webFetch returns error for unreachable host', async () => {
    const result = await web.webFetch({ url: 'http://192.0.2.1:1' });
    expect(result).toContain('Error');
    expect(result).toContain('failed to fetch');
  }, 20000);

  test('webSearch returns error for missing query', async () => {
    const result = await web.webSearch({});
    expect(result).toContain('Error');
    expect(result).toContain('no query');
  });

  test('webFetch fetches a real HTTP page', async () => {
    // Use a local HTTP server to avoid external dependency
    const server = http.createServer((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end('<html><body><p>Hello from test server</p></body></html>');
    });
    await new Promise(resolve => server.listen(0, resolve));
    const port = server.address().port;

    try {
      const result = await web.webFetch({ url: `http://127.0.0.1:${port}/test` });
      expect(result).toContain('Hello from test server');
    } finally {
      server.close();
    }
  });

  test('webFetch follows redirects', async () => {
    let reqCount = 0;
    const server = http.createServer((req, res) => {
      reqCount++;
      if (req.url === '/redirect') {
        res.writeHead(302, { Location: '/final' });
        res.end();
      } else {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end('<html><body><p>Redirected content</p></body></html>');
      }
    });
    await new Promise(resolve => server.listen(0, resolve));
    const port = server.address().port;

    try {
      const result = await web.webFetch({ url: `http://127.0.0.1:${port}/redirect` });
      expect(result).toContain('Redirected content');
      expect(reqCount).toBe(2);
    } finally {
      server.close();
    }
  });

  test('webFetch converts HTML to text (strips tags)', async () => {
    const server = http.createServer((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end('<html><body><h1>Title</h1><p>Paragraph <b>bold</b></p><script>evil()</script></body></html>');
    });
    await new Promise(resolve => server.listen(0, resolve));
    const port = server.address().port;

    try {
      const result = await web.webFetch({ url: `http://127.0.0.1:${port}/` });
      expect(result).toContain('Title');
      expect(result).toContain('Paragraph');
      expect(result).toContain('bold');
      expect(result).not.toContain('<h1>');
      expect(result).not.toContain('evil');
    } finally {
      server.close();
    }
  });

  test('web.js source has no shell dependencies', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/web.js'), 'utf-8');
    expect(source).not.toContain('execSync');
    expect(source).not.toContain('spawn');
    expect(source).not.toContain('child_process');
    // No curl shell-out (the file has a comment "no curl dependency" which is fine)
    expect(source).not.toMatch(/exec.*curl/);
    expect(source).toContain("require('https')");
    expect(source).toContain("require('http')");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// PowerShell Script Tool
// ═══════════════════════════════════════════════════════════════════════════

describe('powershellScript tool', () => {
  const system = require('../src/tools/system');
  const { IS_WIN, IS_MAC } = require('../src/platform');

  test('powershellScript is exported', () => {
    expect(typeof system.powershellScript).toBe('function');
  });

  if (IS_MAC) {
    test('returns platform unavailability message on macOS (regardless of args)', () => {
      // On macOS, platform check fires before arg validation
      expect(system.powershellScript({ script: 'test' })).toContain('not available');
      expect(system.powershellScript({})).toContain('not available');
      expect(system.powershellScript(null)).toContain('not available');
    });
  }

  if (IS_WIN) {
    test('returns error for missing script on Windows', () => {
      const result = system.powershellScript({});
      expect(result).toContain('Error');
      expect(result).toContain('no script');
    });

    test('returns error for null args on Windows', () => {
      const result = system.powershellScript(null);
      expect(result).toContain('Error');
    });

    test('executes a simple PowerShell command on Windows', () => {
      const result = system.powershellScript({ script: 'Write-Host "hello"' });
      expect(result).toContain('hello');
    });

    test('applescript returns unavailability message on Windows', () => {
      const result = system.applescript({ script: 'return "test"' });
      expect(result).toContain('not available');
      expect(result).toContain('powershell');
    });
  }
});

// ═══════════════════════════════════════════════════════════════════════════
// Native App Tool Dispatchers (Calendar, Mail, Notes, Contacts, Reminders)
// ═══════════════════════════════════════════════════════════════════════════

describe('native app tools — dispatcher routing', () => {
  const { IS_MAC, IS_WIN } = require('../src/platform');
  const isSupported = IS_MAC || IS_WIN;

  beforeEach(() => {
    fs.writeFileSync(CONNECTIONS_FILE, '{}');
  });

  // On Linux CI, native app tools return "only supported on macOS/Windows"
  // before reaching the action dispatcher. Skip behavioral tests on Linux.

  describe('calendar', () => {
    const { calendar } = require('../src/tools/calendar');

    if (isSupported) {
      test('returns error without action', async () => {
        const result = await calendar({});
        expect(result).toContain('Error');
        expect(result).toContain('action required');
      });

      test('returns error for unknown action', async () => {
        const result = await calendar({ action: 'nonexistent' });
        expect(result).toContain('Error');
        expect(result).toContain('unknown action');
      });

      test('all expected actions are listed in error message', async () => {
        const result = await calendar({});
        for (const action of ['list_events', 'create_event', 'search', 'list_calendars', 'delete_event', 'today', 'week']) {
          expect(result).toContain(action);
        }
      });
    } else {
      test('returns EDS guidance on Linux without EDS', async () => {
        const result = await calendar({ action: 'today' });
        // On Linux, calendar dispatches to EDS which returns install guidance
        expect(result).toContain('Evolution Data Server');
      });
    }

    test('source has IS_WIN dispatch branch', () => {
      const source = fs.readFileSync(require.resolve('../src/tools/calendar.js'), 'utf-8');
      expect(source).toContain('if (IS_WIN)');
      expect(source).toContain('winListEvents');
      expect(source).toContain('winCreateEvent');
      expect(source).toContain('winSearchEvents');
      expect(source).toContain('winListCalendars');
      expect(source).toContain('winDeleteEvent');
    });
  });

  describe('mail', () => {
    const { mail } = require('../src/tools/mail');

    if (isSupported) {
      test('returns error without action', async () => {
        const result = await mail({});
        expect(result).toContain('Error');
        expect(result).toContain('action required');
      });

      test('returns error for unknown action', async () => {
        const result = await mail({ action: 'nonexistent' });
        expect(result).toContain('Error');
        expect(result).toContain('unknown');
      });
    } else {
      test('returns unsupported platform error on Linux', async () => {
        const result = await mail({ action: 'list_recent' });
        expect(result).toContain('Error');
        expect(result).toContain('only supported');
      });
    }

    test('source has IS_WIN dispatch branch', () => {
      const source = fs.readFileSync(require.resolve('../src/tools/mail.js'), 'utf-8');
      expect(source).toContain('if (IS_WIN)');
      expect(source).toMatch(/function win\w+/);
    });
  });

  describe('notes', () => {
    const { notes } = require('../src/tools/notes');

    if (isSupported) {
      test('returns error without action', async () => {
        const result = await notes({});
        expect(result).toContain('Error');
        expect(result).toContain('action required');
      });

      test('returns error for unknown action', async () => {
        const result = await notes({ action: 'nonexistent' });
        expect(result).toContain('Error');
        expect(result).toContain('unknown');
      });

      test('all expected actions are listed', async () => {
        const result = await notes({});
        for (const action of ['search', 'read', 'create', 'list_recent', 'list_folders']) {
          expect(result).toContain(action);
        }
      });
    } else {
      test('returns unsupported platform error on Linux', async () => {
        const result = await notes({ action: 'list_recent' });
        expect(result).toContain('Error');
        expect(result).toContain('only supported');
      });
    }

    test('source has IS_WIN dispatch branch', () => {
      const source = fs.readFileSync(require.resolve('../src/tools/notes.js'), 'utf-8');
      expect(source).toContain('if (IS_WIN)');
      expect(source).toMatch(/function win\w+/);
    });
  });

  describe('contacts', () => {
    const { contacts } = require('../src/tools/contacts');

    if (isSupported) {
      test('returns error without action', async () => {
        const result = await contacts({});
        expect(result).toContain('Error');
        expect(result).toContain('action required');
      });

      test('returns error for unknown action', async () => {
        const result = await contacts({ action: 'nonexistent' });
        expect(result).toContain('Error');
        expect(result).toContain('unknown');
      });
    } else {
      test('returns EDS guidance on Linux without EDS', async () => {
        const result = await contacts({ action: 'search' });
        // On Linux, contacts dispatches to EDS which returns install guidance
        expect(result).toContain('Evolution Data Server');
      });
    }

    test('source has IS_WIN dispatch branch', () => {
      const source = fs.readFileSync(require.resolve('../src/tools/contacts.js'), 'utf-8');
      expect(source).toContain('if (IS_WIN)');
      expect(source).toMatch(/function win\w+/);
    });
  });

  describe('reminders', () => {
    const { reminders } = require('../src/tools/reminders');

    if (isSupported) {
      test('returns error without action', async () => {
        const result = await reminders({});
        expect(result).toContain('Error');
        expect(result).toContain('action required');
      });

      test('returns error for unknown action', async () => {
        const result = await reminders({ action: 'nonexistent' });
        expect(result).toContain('Error');
        expect(result).toContain('unknown');
      });

      test('all expected actions are listed', async () => {
        const result = await reminders({});
        for (const action of ['add', 'list', 'complete', 'search', 'delete']) {
          expect(result).toContain(action);
        }
      });
    } else {
      test('returns EDS guidance on Linux without EDS', async () => {
        const result = await reminders({ action: 'list' });
        // On Linux, reminders dispatches to EDS which returns install guidance
        expect(result).toContain('Evolution Data Server');
      });
    }

    test('source has IS_WIN dispatch branch', () => {
      const source = fs.readFileSync(require.resolve('../src/tools/reminders.js'), 'utf-8');
      expect(source).toContain('if (IS_WIN)');
      expect(source).toMatch(/function win\w+/);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Sessions — Windows Filename Translation
// ═══════════════════════════════════════════════════════════════════════════

describe('sessions — Windows filename compatibility', () => {
  const { IS_WIN } = require('../src/platform');

  test('sessions.js imports IS_WIN from platform', () => {
    const source = fs.readFileSync(require.resolve('../src/sessions.js'), 'utf-8');
    expect(source).toContain("{ IS_WIN }");
    expect(source).toContain("require('./platform')");
  });

  test('idToFilename replaces colons with double-dashes on Windows', () => {
    const source = fs.readFileSync(require.resolve('../src/sessions.js'), 'utf-8');
    // Verify the function body has the right translation logic
    expect(source).toContain("id.replace(/:/g, '--')");
  });

  test('filenameToId converts double-dashes back to colons on Windows', () => {
    const source = fs.readFileSync(require.resolve('../src/sessions.js'), 'utf-8');
    expect(source).toContain("name.replace(/--/g, ':')");
  });

  test('session CRUD works with colon-containing IDs', () => {
    const sessions = require('../src/sessions');

    // Create a session with a colon in the ID (project:channel pattern)
    const id = sessions.createSession('test', 'wincheck', true);
    expect(id).toContain(':');

    // Append and read back
    sessions.appendMessage(id, { role: 'user', content: 'Windows filename test' });
    const messages = sessions.getSessionMessages(id);
    expect(messages.length).toBeGreaterThanOrEqual(1);
    expect(messages.some(m => m.content === 'Windows filename test')).toBe(true);

    // Verify file on disk uses correct separator
    const files = fs.readdirSync(SESSIONS_DIR);
    if (IS_WIN) {
      // On Windows, file should use '--' not ':'
      expect(files.some(f => f.includes('--') && f.includes('wincheck'))).toBe(true);
      expect(files.some(f => f.includes(':') && f.includes('wincheck'))).toBe(false);
    } else {
      // On macOS, file uses ':' (valid in HFS+)
      expect(files.some(f => f.includes(':') && f.includes('wincheck'))).toBe(true);
    }

    // List sessions and verify the ID is returned with ':'
    const list = sessions.listSessions();
    const found = list.find(s => s.id.includes('wincheck'));
    expect(found).toBeTruthy();
    expect(found.id).toContain(':');

    // Cleanup
    sessions.deleteSession(id);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Autostart — Windows VBScript Path
// ═══════════════════════════════════════════════════════════════════════════

describe('autostart — Windows VBScript structure', () => {
  test('server.js defines Windows startup constants', () => {
    const source = fs.readFileSync(require.resolve('../server.js'), 'utf-8');
    // Verify VBScript startup file path uses Windows conventions
    expect(source).toContain('WIN_STARTUP_DIR');
    expect(source).toContain('WIN_STARTUP_VBS');
    expect(source).toContain('PRE_SERVER_PS1');
    expect(source).toContain("'PRE-Server.vbs'");
  });

  test('Windows startup directory is under AppData', () => {
    const source = fs.readFileSync(require.resolve('../server.js'), 'utf-8');
    expect(source).toContain("'AppData', 'Roaming', 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup'");
  });

  test('VBScript template uses WScript.Shell and PowerShell', () => {
    const source = fs.readFileSync(require.resolve('../server.js'), 'utf-8');
    expect(source).toContain('WScript.Shell');
    expect(source).toContain('powershell.exe');
    expect(source).toContain('-ExecutionPolicy Bypass');
    expect(source).toContain('pre-server.ps1');
  });

  test('autostart GET endpoint handles IS_WIN branch', () => {
    const source = fs.readFileSync(require.resolve('../server.js'), 'utf-8');
    // The GET /api/system/autostart endpoint should check IS_WIN
    expect(source).toMatch(/api\/system\/autostart[\s\S]*?IS_WIN/);
  });

  test('autostart POST endpoint has Windows VBS creation branch', () => {
    const source = fs.readFileSync(require.resolve('../server.js'), 'utf-8');
    // The POST endpoint creates VBS on Windows
    expect(source).toMatch(/post.*autostart[\s\S]*?WIN_STARTUP_VBS/i);
  });

  test('pre-server.ps1 exists', () => {
    const ps1Path = path.join(__dirname, '..', 'pre-server.ps1');
    expect(fs.existsSync(ps1Path)).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Delegate Tool — Platform-Aware stderr Redirect
// ═══════════════════════════════════════════════════════════════════════════

describe('delegate tool — platform awareness', () => {
  const delegate = require('../src/tools/delegate');

  test('DELEGATES includes claude, codex, gemini', () => {
    expect(delegate.DELEGATES).toHaveProperty('claude');
    expect(delegate.DELEGATES).toHaveProperty('codex');
    expect(delegate.DELEGATES).toHaveProperty('gemini');
  });

  test('each delegate has required config fields', () => {
    for (const [, config] of Object.entries(delegate.DELEGATES)) {
      expect(config).toHaveProperty('name');
      expect(config).toHaveProperty('command');
      expect(config).toHaveProperty('buildArgs');
      expect(typeof config.buildArgs).toBe('function');
      expect(config).toHaveProperty('icon');
      expect(config).toHaveProperty('color');
    }
  });

  test('stderr redirect uses platform branch', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/delegate.js'), 'utf-8');
    // Must not have bare 2>/dev/null — should use platform-branched suppress
    expect(source).toContain("process.platform === 'win32'");
    expect(source).toContain('2>NUL');
    expect(source).toContain('2>/dev/null');
  });

  test('checkAvailability returns structured results', () => {
    const result = delegate.checkAvailability();
    expect(typeof result).toBe('object');
    // Each key should have available boolean and name
    for (const [, info] of Object.entries(result)) {
      expect(typeof info.available).toBe('boolean');
      expect(typeof info.name).toBe('string');
    }
  });

  test('execute rejects unknown delegate', async () => {
    await expect(delegate.execute('nonexistent', 'test')).rejects.toThrow('Unknown delegate');
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Computer-Win32 — Desktop Automation Module
// ═══════════════════════════════════════════════════════════════════════════

describe('computer-win32 module', () => {
  const win32 = require('../src/tools/computer-win32');

  test('all required functions are exported', () => {
    const requiredExports = [
      'isAvailable', 'getScreenSize', 'takeScreenshot',
      'mouseAction', 'mouseDrag', 'mouseScroll',
      'typeText', 'pressKey',
      'getWindowBounds', 'focusWindow', 'getForegroundApp',
      'getCursorPosition',
    ];
    for (const fn of requiredExports) {
      expect(typeof win32[fn]).toBe('function');
    }
  });

  test('source uses PowerShell for Windows automation', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/computer-win32.js'), 'utf-8');
    // Should use PowerShell, P/Invoke, or .NET for automation
    expect(source).toMatch(/powershell|System\.Windows\.Forms|user32/i);
  });

  test('source has key mapping for modifier keys', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/computer-win32.js'), 'utf-8');
    // Windows uses different key names than macOS
    expect(source).toMatch(/ctrl|alt|shift|win/i);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Platform-Gated Tool Registration in tools-defs.js
// ═══════════════════════════════════════════════════════════════════════════

describe('tools-defs — platform-conditional registration', () => {
  beforeEach(() => {
    fs.writeFileSync(CONNECTIONS_FILE, '{}');
  });

  const { IS_MAC, IS_WIN } = require('../src/platform');
  const { buildToolDefs } = require('../src/tools-defs');

  test('notification tool description is platform-neutral', () => {
    const tools = buildToolDefs();
    const notifyTool = tools.find(t => t.function.name === 'notify');
    expect(notifyTool).toBeTruthy();
    // Should say "desktop notification" not "macOS notification"
    expect(notifyTool.function.description.toLowerCase()).toContain('notification');
    expect(notifyTool.function.description.toLowerCase()).not.toContain('macos notification');
  });

  test('native app tools have platform-appropriate descriptions', () => {
    if (!IS_MAC && !IS_WIN) return;

    const tools = buildToolDefs();
    const mailTool = tools.find(t => t.function.name === 'apple_mail');
    expect(mailTool).toBeTruthy();

    if (IS_WIN) {
      expect(mailTool.function.description).toContain('Outlook');
    } else {
      expect(mailTool.function.description).toContain('macOS Mail');
    }
  });

  test('tools-defs source uses IS_MAC || IS_WIN for native apps', () => {
    const source = fs.readFileSync(require.resolve('../src/tools-defs.js'), 'utf-8');
    expect(source).toContain('IS_MAC || IS_WIN');
  });

  test('tools-defs source gates applescript to IS_MAC only', () => {
    const source = fs.readFileSync(require.resolve('../src/tools-defs.js'), 'utf-8');
    expect(source).toMatch(/IS_MAC.*applescript/);
  });

  test('tools-defs source gates powershell_script to IS_WIN only', () => {
    const source = fs.readFileSync(require.resolve('../src/tools-defs.js'), 'utf-8');
    expect(source).toMatch(/IS_WIN.*powershell_script/);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Installer & Launch Scripts — Windows Files Exist
// ═══════════════════════════════════════════════════════════════════════════

describe('Windows launch/install scripts exist', () => {
  const repoRoot = path.join(__dirname, '..', '..');

  test('install.ps1 exists and references NVIDIA optimizations', () => {
    const ps1Path = path.join(repoRoot, 'install.ps1');
    expect(fs.existsSync(ps1Path)).toBe(true);
    const source = fs.readFileSync(ps1Path, 'utf-8');
    expect(source).toContain('OLLAMA_FLASH_ATTENTION');
    expect(source).toContain('OLLAMA_KV_CACHE_TYPE');
    expect(source).toContain('OLLAMA_GPU_OVERHEAD');
  });

  test('Launch PRE.cmd exists and has NVIDIA detection', () => {
    const cmdPath = path.join(repoRoot, 'Launch PRE.cmd');
    expect(fs.existsSync(cmdPath)).toBe(true);
    const source = fs.readFileSync(cmdPath, 'utf-8');
    expect(source).toContain('nvidia-smi');
    expect(source).toContain('OLLAMA_FLASH_ATTENTION');
  });

  test('pre-server.ps1 exists and has NVIDIA env vars', () => {
    const ps1Path = path.join(__dirname, '..', 'pre-server.ps1');
    expect(fs.existsSync(ps1Path)).toBe(true);
    const source = fs.readFileSync(ps1Path, 'utf-8');
    expect(source).toContain('OLLAMA_FLASH_ATTENTION');
    expect(source).toContain('OLLAMA_KV_CACHE_TYPE');
  });

  test('install.ps1 uses VRAM-aware quant and KV cache selection', () => {
    const ps1Path = path.join(repoRoot, 'install.ps1');
    const source = fs.readFileSync(ps1Path, 'utf-8');
    // Should select q4_0 or q8_0 KV cache based on quant type
    expect(source).toMatch(/q4_0|q8_0/);
    expect(source).toContain('QUANT');
    // VRAM-aware: model must fit in VRAM for full GPU acceleration
    expect(source).toContain('vramGB');
    expect(source).toContain('q4_K_M');
    expect(source).toContain('q8_0');
  });

  test('pre-server.ps1 uses VRAM-based KV cache detection', () => {
    const ps1Path = path.join(__dirname, '..', 'pre-server.ps1');
    const source = fs.readFileSync(ps1Path, 'utf-8');
    // Should detect VRAM via nvidia-smi, not rely on RAM alone
    expect(source).toContain('memory.total');
    expect(source).toContain('detectedVramGB');
  });

  test('Launch PRE.cmd uses VRAM-based KV cache detection', () => {
    const cmdPath = path.join(repoRoot, 'Launch PRE.cmd');
    const source = fs.readFileSync(cmdPath, 'utf-8');
    // Should detect VRAM via nvidia-smi, not rely on RAM alone
    expect(source).toContain('memory.total');
    expect(source).toMatch(/geq 28/);
  });
});
