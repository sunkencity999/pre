// Tests for Windows compatibility across all modified tool modules
// Validates that:
//   1. All tool modules load without errors on any platform
//   2. Platform-specific imports resolve correctly
//   3. Tool dispatchers handle missing platform gracefully
//   4. Windows code path structure is correct (even on macOS runner)

const fs = require('fs');

jest.mock('../src/constants');
jest.mock('../src/ollama', () => ({
  streamChat: jest.fn().mockResolvedValue({ response: '[]' }),
}));
jest.mock('../src/mcp', () => ({
  getConnectedTools: jest.fn().mockReturnValue([]),
  getAllTools: jest.fn().mockReturnValue([]),
  loadConfig: jest.fn().mockReturnValue({ servers: {} }),
}));

const { CONNECTIONS_FILE } = require('../src/constants');

describe('Windows compatibility — module loading', () => {
  beforeEach(() => {
    fs.writeFileSync(CONNECTIONS_FILE, '{}');
  });

  // ── Core tool modules load ──────────────────────────────────────────────

  const toolModules = [
    'bash', 'files', 'web', 'delegate', 'browser', 'export',
    'system', 'voice', 'spotlight', 'computer', 'monitor',
    'calendar', 'reminders', 'mail', 'notes', 'contacts',
    'artifact', 'document', 'cron', 'agents', 'image', 'rag',
  ];

  test.each(toolModules)('tools/%s loads without error', (mod) => {
    expect(() => require(`../src/tools/${mod}`)).not.toThrow();
  });

  test('computer-win32 module loads without error', () => {
    // On macOS this module loads but its functions would fail when called
    // (since PowerShell isn't available). We just verify it parses.
    expect(() => require('../src/tools/computer-win32')).not.toThrow();
  });

  // ── Platform module ─────────────────────────────────────────────────────

  test('platform module loads and exports IS_WIN/IS_MAC', () => {
    const platform = require('../src/platform');
    expect(typeof platform.IS_WIN).toBe('boolean');
    expect(typeof platform.IS_MAC).toBe('boolean');
  });

  // ── tools-defs builds successfully ──────────────────────────────────────

  test('buildToolDefs returns valid tool array', () => {
    const { buildToolDefs } = require('../src/tools-defs');
    const tools = buildToolDefs();
    expect(Array.isArray(tools)).toBe(true);
    expect(tools.length).toBeGreaterThan(30);
  });

  test('spotlight tool is always registered (cross-platform)', () => {
    const { buildToolDefs } = require('../src/tools-defs');
    const names = buildToolDefs().map(t => t.function.name);
    expect(names).toContain('spotlight');
  });

  test('core tools present regardless of platform', () => {
    const { buildToolDefs } = require('../src/tools-defs');
    const names = buildToolDefs().map(t => t.function.name);
    const coreTool = [
      'bash', 'read_file', 'list_dir', 'glob', 'grep',
      'file_write', 'file_edit', 'web_fetch', 'memory_save',
      'system_info', 'spotlight', 'cron', 'trigger',
    ];
    for (const t of coreTool) {
      expect(names).toContain(t);
    }
  });

  test('native app tools present on macOS AND Windows', () => {
    const { IS_MAC, IS_WIN } = require('../src/platform');
    const { buildToolDefs } = require('../src/tools-defs');
    const names = buildToolDefs().map(t => t.function.name);

    // PIM tools (calendar, contacts, reminders, notes) are cross-platform
    // (macOS: EventKit/AppleScript, Windows: Outlook COM, Linux: GNOME EDS)
    const crossPlatformPIM = [
      'apple_calendar', 'apple_contacts', 'apple_reminders', 'apple_notes',
    ];
    for (const t of crossPlatformPIM) {
      expect(names).toContain(t);
    }

    // Mail is macOS + Windows only (no stable Linux mail DBus API)
    if (IS_MAC || IS_WIN) {
      expect(names).toContain('apple_mail');
    } else {
      expect(names).not.toContain('apple_mail');
    }
  });

  test('applescript only on macOS, powershell_script only on Windows', () => {
    const { IS_MAC, IS_WIN } = require('../src/platform');
    const { buildToolDefs } = require('../src/tools-defs');
    const names = buildToolDefs().map(t => t.function.name);

    if (IS_MAC) {
      expect(names).toContain('applescript');
      expect(names).not.toContain('powershell_script');
    } else if (IS_WIN) {
      expect(names).toContain('powershell_script');
      expect(names).not.toContain('applescript');
    }
  });
});

describe('Windows compatibility — tool behavior', () => {
  // ── Voice tool ──────────────────────────────────────────────────────────

  describe('voice', () => {
    const voice = require('../src/tools/voice');

    test('exports voice dispatcher and capabilities', () => {
      expect(typeof voice.voice).toBe('function');
      expect(typeof voice.hasSay).toBe('boolean');
      expect(typeof voice.hasWhisper).toBe('boolean');
      expect(typeof voice.hasFfmpeg).toBe('boolean');
    });

    test('status action returns capability info', async () => {
      const result = await voice.voice({ action: 'status' });
      expect(result).toContain('Voice capabilities');
      expect(result).toContain('TTS');
    });

    test('speak returns error for empty text', async () => {
      const result = await voice.voice({ action: 'speak' });
      expect(result).toContain('Error');
    });

    test('unknown action returns error', async () => {
      const result = await voice.voice({ action: 'nonexistent' });
      expect(result).toContain('Unknown voice action');
    });
  });

  // ── Spotlight tool ──────────────────────────────────────────────────────

  describe('spotlight', () => {
    const { spotlight } = require('../src/tools/spotlight');

    test('returns error without action', async () => {
      const result = await spotlight({});
      expect(result).toContain('Error');
      expect(result).toContain('action required');
    });

    test('search requires query', async () => {
      const result = await spotlight({ action: 'search' });
      expect(result).toContain('Error');
      expect(result).toContain('query');
    });

    test('find_files requires query or type', async () => {
      const result = await spotlight({ action: 'find_files' });
      expect(result).toContain('Error');
    });

    test('preview requires path', async () => {
      const result = await spotlight({ action: 'preview' });
      expect(result).toContain('Error');
    });

    test('unknown action returns error', async () => {
      const result = await spotlight({ action: 'nonexistent' });
      expect(result).toContain('Error');
      expect(result).toContain('unknown action');
    });
  });

  // ── System tool ─────────────────────────────────────────────────────────

  describe('system', () => {
    const system = require('../src/tools/system');

    test('systemInfo returns formatted string', () => {
      const result = system.systemInfo();
      expect(typeof result).toBe('string');
      // CPU name may be empty on Linux CI (no sysctl), but Memory is always present
      expect(result).toContain('Memory');
      expect(result).toContain('OS:');
    });

    test('hardwareInfo returns formatted string', () => {
      const result = system.hardwareInfo();
      expect(typeof result).toBe('string');
      // On Linux CI, CPU name may be empty but cores/memory are always present
      expect(result).toContain('Memory');
    });

    test('processKill returns error for missing pid', () => {
      expect(system.processKill({})).toContain('Error');
    });

    test('processKill returns error for invalid pid', () => {
      expect(system.processKill({ pid: 'abc' })).toContain('Error');
    });

    test('clipboardWrite returns error for empty content', () => {
      expect(system.clipboardWrite({})).toContain('Error');
    });

    test('openApp returns error for missing target', () => {
      expect(system.openApp({})).toContain('Error');
    });

    test('notify returns error for missing title/message', () => {
      expect(system.notify({})).toContain('Error');
    });

    test('windowFocus returns error for missing app', () => {
      expect(system.windowFocus({})).toContain('Error');
    });

    test('applescript returns unavailable message on Windows', () => {
      const { IS_WIN } = require('../src/platform');
      if (IS_WIN) {
        expect(system.applescript({ script: 'test' })).toContain('not available');
      }
    });
  });

  // ── Computer tool ───────────────────────────────────────────────────────

  describe('computer', () => {
    const computer = require('../src/tools/computer');

    test('exports isAvailable function', () => {
      expect(typeof computer.isAvailable).toBe('function');
    });

    test('exports setTargetApp function', () => {
      expect(typeof computer.setTargetApp).toBe('function');
    });

    test('exports maximizeTargetWindow function', () => {
      expect(typeof computer.maximizeTargetWindow).toBe('function');
    });

    test('exports restoreFocus function', () => {
      expect(typeof computer.restoreFocus).toBe('function');
    });

    test('computerUse returns error without action', async () => {
      const result = await computer.computerUse({});
      expect(result).toContain('Error');
      expect(result).toContain('action required');
    });
  });

  // ── Computer-Win32 module structure ─────────────────────────────────────

  describe('computer-win32', () => {
    const win32 = require('../src/tools/computer-win32');

    test('exports all required functions', () => {
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
  });

  // ── Files tool (nodeGlob/nodeGrep integration) ─────────────────────────

  describe('files', () => {
    const files = require('../src/tools/files');

    test('exports glob function', () => {
      expect(typeof files.glob).toBe('function');
    });

    test('exports grep function', () => {
      expect(typeof files.grep).toBe('function');
    });

    test('glob returns error for missing pattern', () => {
      const result = files.glob({}, '/tmp');
      expect(result).toContain('Error');
    });

    test('grep returns error for missing pattern', () => {
      const result = files.grep({}, '/tmp');
      expect(result).toContain('Error');
    });
  });

  // ── Bash tool ───────────────────────────────────────────────────────────

  describe('bash', () => {
    const bash = require('../src/tools/bash');

    test('exports bash function', () => {
      expect(typeof bash.bash).toBe('function');
    });

    test('executes a simple command', async () => {
      // On Linux CI, /bin/zsh may not exist — the bash tool uses platform.getShell()
      // which defaults to /bin/zsh on macOS. Skip if shell isn't available.
      const { getShell } = require('../src/platform');
      const fs = require('fs');
      const shell = getShell();
      if (!fs.existsSync(shell.cmd)) {
        return; // skip on CI where the shell binary doesn't exist
      }
      const result = await bash.bash({ command: 'echo hello' }, '/tmp');
      expect(result).toContain('hello');
    });

    test('returns error for empty command', async () => {
      const result = await bash.bash({}, '/tmp');
      expect(result).toContain('Error');
    });
  });

  // ── Web tool ────────────────────────────────────────────────────────────

  describe('web', () => {
    const web = require('../src/tools/web');

    test('exports webFetch function', () => {
      expect(typeof web.webFetch).toBe('function');
    });
  });

  // ── Delegate tool ───────────────────────────────────────────────────────

  describe('delegate', () => {
    const delegate = require('../src/tools/delegate');

    test('exports execute and DELEGATES', () => {
      expect(typeof delegate.execute).toBe('function');
      expect(typeof delegate.DELEGATES).toBe('object');
    });
  });

  // ── Export tool ─────────────────────────────────────────────────────────

  describe('export', () => {
    const exp = require('../src/tools/export');

    test('exports findChrome function', () => {
      expect(typeof exp.findChrome).toBe('function');
    });

    test('exports export functions', () => {
      expect(typeof exp.exportPdf).toBe('function');
      expect(typeof exp.exportPng).toBe('function');
      expect(typeof exp.exportSelfContainedHtml).toBe('function');
    });
  });

  // ── Browser tool ────────────────────────────────────────────────────────

  describe('browser', () => {
    const browser = require('../src/tools/browser');

    test('exports isAvailable function', () => {
      expect(typeof browser.isAvailable).toBe('function');
    });
  });

  // ── Monitor tool ────────────────────────────────────────────────────────

  describe('monitor', () => {
    const monitor = require('../src/tools/monitor');

    test('exports monitor function', () => {
      expect(typeof monitor.monitor).toBe('function');
    });

    test('returns status without command', async () => {
      const result = await monitor.monitor({});
      expect(typeof result).toBe('string');
    });
  });
});

describe('Windows compatibility — cross-platform patterns', () => {
  // ── Shell command redirection ───────────────────────────────────────────

  describe('no POSIX redirects in Windows code paths', () => {
    const filesToCheck = [
      '../src/tools/voice.js',
      '../src/tools/spotlight.js',
      '../src/tools/delegate.js',
      '../src/tools/system.js',
      '../src/tools/monitor.js',
    ];

    test.each(filesToCheck)('%s does not use bare 2>/dev/null in execSync calls', (mod) => {
      const source = fs.readFileSync(require.resolve(mod), 'utf-8');
      // Bare 2>/dev/null inside execSync is not cross-platform. Allowed patterns:
      //   - Platform-branched: process.platform === 'win32' ? '2>NUL' : '2>/dev/null'
      //   - stdio: 'pipe' (captures stderr without redirects)
      //   - Inside Linux-only functions (linuxSearch, linuxFindFiles, linuxPreview)
      //     which use Linux-only commands (stat --format, file --brief, locate, find)
      const execCalls = source.match(/execSync\([^)]*2>\/dev\/null[^)]*\)/g) || [];
      // Filter out platform-branched and Linux-only patterns (those are fine)
      const bare = execCalls.filter(c =>
        !c.includes('process.platform') && !c.includes('IS_WIN') && !c.includes('IS_LINUX') &&
        !c.includes('stat --format') && !c.includes('file --brief') &&
        !c.includes('locate ') && !c.includes('find "'));
      expect(bare).toEqual([]);
    });

    test('web.js uses no shell commands at all (native https)', () => {
      const source = fs.readFileSync(require.resolve('../src/tools/web.js'), 'utf-8');
      expect(source).not.toContain('execSync');
      expect(source).not.toContain('child_process');
      expect(source).toContain("require('https')");
    });
  });

  // ── Platform imports ────────────────────────────────────────────────────

  describe('platform imports', () => {
    const modifiedTools = [
      '../src/tools/bash.js',
      '../src/tools/files.js',
      '../src/tools/web.js',
      '../src/tools/delegate.js',
      '../src/tools/browser.js',
      '../src/tools/export.js',
      '../src/tools/system.js',
      '../src/tools/voice.js',
      '../src/tools/spotlight.js',
      '../src/tools/computer.js',
    ];

    test.each(modifiedTools)('%s imports from platform.js', (mod) => {
      const source = fs.readFileSync(require.resolve(mod), 'utf-8');
      expect(source).toMatch(/require\(['"]\.\.\/platform['"]\)/);
    });
  });

  // ── Context system prompt ───────────────────────────────────────────────

  describe('context.js platform awareness', () => {
    test('system prompt references correct platform', () => {
      const context = require('../src/context');
      const prompt = context.buildSystemPrompt('/tmp');
      const { IS_MAC, IS_WIN } = require('../src/platform');

      if (IS_MAC) {
        expect(prompt).toContain('Apple Silicon');
      } else if (IS_WIN) {
        expect(prompt).toContain('Windows');
      } else {
        // Linux CI — falls through to macOS default path (no Linux-specific prompt)
        expect(typeof prompt).toBe('string');
        expect(prompt.length).toBeGreaterThan(100);
      }
    });
  });
});
