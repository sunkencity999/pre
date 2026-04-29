// Tests for src/platform.js — Cross-platform abstraction layer
// Validates both macOS and Windows code paths.
// Since we can't change process.platform at runtime, we test:
//   1. Module exports and structure
//   2. Pure-JS functions that work identically on both platforms
//   3. Current platform (macOS) behavior
//   4. Windows code path structure via mock/spy where possible

const fs = require('fs');
const path = require('path');
const os = require('os');

const platform = require('../src/platform');

describe('platform', () => {
  // ── Constants ───────────────────────────────────────────────────────────

  describe('constants', () => {
    test('IS_WIN is boolean', () => {
      expect(typeof platform.IS_WIN).toBe('boolean');
    });

    test('IS_MAC is boolean', () => {
      expect(typeof platform.IS_MAC).toBe('boolean');
    });

    test('IS_WIN and IS_MAC are mutually exclusive on Mac/Win', () => {
      // On macOS test runner, IS_MAC=true, IS_WIN=false
      // On Windows test runner, IS_MAC=false, IS_WIN=true
      expect(platform.IS_MAC).not.toBe(platform.IS_WIN);
    });

    test('platform matches process.platform', () => {
      if (process.platform === 'darwin') {
        expect(platform.IS_MAC).toBe(true);
        expect(platform.IS_WIN).toBe(false);
      } else if (process.platform === 'win32') {
        expect(platform.IS_WIN).toBe(true);
        expect(platform.IS_MAC).toBe(false);
      }
    });
  });

  // ── Exports ─────────────────────────────────────────────────────────────

  describe('exports', () => {
    const expectedExports = [
      'IS_WIN', 'IS_MAC',
      'getShell', 'getShellPath', 'getChromePaths',
      'revealInFileManager', 'clipboardRead', 'clipboardWrite',
      'notify', 'openTarget', 'whichCmd', 'htmlToText',
      'getCpuInfo', 'getCpuUsage', 'getMemoryInfo',
      'getDiskUsage', 'getBatteryInfo', 'getGpuInfo',
      'hasTTS', 'ttsSpeak', 'ttsListVoices',
      'nodeGlob', 'nodeGrep',
      'processList', 'netInfo', 'netConnections', 'serviceStatus',
      'windowList', 'windowFocus', 'diskUsageFormatted', 'screenshot',
    ];

    test('exports all expected functions and constants', () => {
      for (const name of expectedExports) {
        expect(platform).toHaveProperty(name);
      }
    });

    test('function exports are callable', () => {
      const funcNames = expectedExports.filter(n => n !== 'IS_WIN' && n !== 'IS_MAC');
      for (const name of funcNames) {
        expect(typeof platform[name]).toBe('function');
      }
    });
  });

  // ── Shell ───────────────────────────────────────────────────────────────

  describe('getShell', () => {
    test('returns object with cmd and args', () => {
      const shell = platform.getShell();
      expect(shell).toHaveProperty('cmd');
      expect(shell).toHaveProperty('args');
      expect(typeof shell.cmd).toBe('string');
      expect(Array.isArray(shell.args)).toBe(true);
    });

    test('returns appropriate shell for current platform', () => {
      const shell = platform.getShell();
      if (platform.IS_MAC) {
        expect(shell.cmd).toBe('/bin/zsh');
        expect(shell.args).toEqual(['-c']);
      } else {
        expect(shell.cmd).toBe('powershell.exe');
        expect(shell.args).toContain('-NoProfile');
        expect(shell.args).toContain('-Command');
      }
    });
  });

  describe('getShellPath', () => {
    test('returns a string', () => {
      expect(typeof platform.getShellPath()).toBe('string');
    });

    test('returns valid shell path for current platform', () => {
      const shellPath = platform.getShellPath();
      if (platform.IS_MAC) {
        expect(shellPath).toMatch(/\/(zsh|bash|sh)$/);
      } else {
        expect(shellPath).toMatch(/powershell/i);
      }
    });
  });

  // ── Chrome Paths ────────────────────────────────────────────────────────

  describe('getChromePaths', () => {
    test('returns an array of strings', () => {
      const paths = platform.getChromePaths();
      expect(Array.isArray(paths)).toBe(true);
      expect(paths.length).toBeGreaterThan(0);
      for (const p of paths) {
        expect(typeof p).toBe('string');
      }
    });

    test('paths include Chrome or Chromium', () => {
      const paths = platform.getChromePaths();
      const hasChrome = paths.some(p => /chrome|chromium/i.test(p));
      expect(hasChrome).toBe(true);
    });

    test('macOS paths are under /Applications', () => {
      if (!platform.IS_MAC) return;
      const paths = platform.getChromePaths();
      const hasMacPath = paths.some(p => p.startsWith('/Applications'));
      expect(hasMacPath).toBe(true);
    });
  });

  // ── whichCmd ────────────────────────────────────────────────────────────

  describe('whichCmd', () => {
    test('finds node executable', () => {
      const result = platform.whichCmd('node');
      expect(result).toBeTruthy();
      expect(result).toContain('node');
    });

    test('returns falsy value for nonexistent command', () => {
      const result = platform.whichCmd('definitely_not_a_real_command_xyz123');
      expect(result).toBeFalsy();
    });

    test('returns falsy value for empty input', () => {
      const result = platform.whichCmd('');
      expect(result).toBeFalsy();
    });
  });

  // ── htmlToText ──────────────────────────────────────────────────────────

  describe('htmlToText', () => {
    test('strips HTML tags', () => {
      const result = platform.htmlToText('<p>Hello <b>world</b></p>');
      expect(result).toContain('Hello');
      expect(result).toContain('world');
      expect(result).not.toContain('<p>');
      expect(result).not.toContain('<b>');
    });

    test('converts common entities', () => {
      const result = platform.htmlToText('&amp; &lt; &gt; &quot;');
      expect(result).toContain('&');
      expect(result).toContain('<');
      expect(result).toContain('>');
    });

    test('handles empty input', () => {
      expect(platform.htmlToText('')).toBe('');
    });

    test('strips script and style blocks', () => {
      const html = '<html><head><style>body{color:red}</style></head><body><script>alert("x")</script><p>Content</p></body></html>';
      const result = platform.htmlToText(html);
      expect(result).toContain('Content');
      expect(result).not.toContain('alert');
      expect(result).not.toContain('color:red');
    });

    test('preserves text from nested elements', () => {
      const result = platform.htmlToText('<div><span>A</span> <span>B</span></div>');
      expect(result).toContain('A');
      expect(result).toContain('B');
    });
  });

  // ── nodeGlob (pure-JS file search) ─────────────────────────────────────

  describe('nodeGlob', () => {
    const tmpDir = path.join(os.tmpdir(), `pre-test-glob-${Date.now()}`);

    beforeAll(() => {
      // Create a test directory structure
      fs.mkdirSync(path.join(tmpDir, 'sub', 'deep'), { recursive: true });
      fs.writeFileSync(path.join(tmpDir, 'file1.txt'), 'hello');
      fs.writeFileSync(path.join(tmpDir, 'file2.js'), 'const x = 1;');
      fs.writeFileSync(path.join(tmpDir, 'sub', 'file3.txt'), 'world');
      fs.writeFileSync(path.join(tmpDir, 'sub', 'deep', 'file4.txt'), 'nested');
      fs.writeFileSync(path.join(tmpDir, 'sub', 'deep', 'file5.json'), '{}');
    });

    afterAll(() => {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    });

    test('finds files matching wildcard pattern', () => {
      const results = platform.nodeGlob(tmpDir, '*.txt');
      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results.some(r => r.endsWith('file1.txt'))).toBe(true);
    });

    test('searches recursively into subdirectories', () => {
      const results = platform.nodeGlob(tmpDir, '*.txt', 5, 100);
      expect(results.length).toBeGreaterThanOrEqual(3);
      expect(results.some(r => r.includes('file3.txt'))).toBe(true);
      expect(results.some(r => r.includes('file4.txt'))).toBe(true);
    });

    test('respects maxDepth', () => {
      const shallow = platform.nodeGlob(tmpDir, '*.txt', 1, 100);
      const deep = platform.nodeGlob(tmpDir, '*.txt', 5, 100);
      expect(deep.length).toBeGreaterThanOrEqual(shallow.length);
    });

    test('respects maxResults limit', () => {
      const results = platform.nodeGlob(tmpDir, '*.*', 5, 2);
      expect(results.length).toBeLessThanOrEqual(2);
    });

    test('returns empty array for no matches', () => {
      const results = platform.nodeGlob(tmpDir, '*.xyz');
      expect(results).toEqual([]);
    });

    test('handles nonexistent directory gracefully', () => {
      const results = platform.nodeGlob('/tmp/nonexistent_dir_xyz123', '*.txt');
      expect(Array.isArray(results)).toBe(true);
      expect(results.length).toBe(0);
    });

    test('matches specific extensions', () => {
      const jsFiles = platform.nodeGlob(tmpDir, '*.js', 5, 100);
      expect(jsFiles.length).toBeGreaterThanOrEqual(1);
      expect(jsFiles.some(f => f.endsWith('file2.js'))).toBe(true);

      const jsonFiles = platform.nodeGlob(tmpDir, '*.json', 5, 100);
      expect(jsonFiles.length).toBeGreaterThanOrEqual(1);
      expect(jsonFiles.some(f => f.endsWith('file5.json'))).toBe(true);
    });
  });

  // ── nodeGrep (pure-JS content search) ──────────────────────────────────

  describe('nodeGrep', () => {
    const tmpDir = path.join(os.tmpdir(), `pre-test-grep-${Date.now()}`);

    beforeAll(() => {
      fs.mkdirSync(path.join(tmpDir, 'sub'), { recursive: true });
      fs.writeFileSync(path.join(tmpDir, 'alpha.txt'), 'Hello World\nFoo Bar\nHello Again\n');
      fs.writeFileSync(path.join(tmpDir, 'beta.js'), 'function hello() {\n  return "world";\n}\n');
      fs.writeFileSync(path.join(tmpDir, 'sub', 'gamma.txt'), 'No match here\nJust data\n');
      fs.writeFileSync(path.join(tmpDir, 'sub', 'delta.txt'), 'Hello from subdirectory\n');
    });

    afterAll(() => {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    });

    test('finds matching lines across files', () => {
      const result = platform.nodeGrep('Hello', tmpDir);
      expect(result).toContain('Hello');
      expect(result).toContain('alpha.txt');
    });

    test('searches recursively', () => {
      const result = platform.nodeGrep('Hello', tmpDir);
      expect(result).toContain('delta.txt');
    });

    test('includes line numbers', () => {
      const result = platform.nodeGrep('Hello', tmpDir);
      // Should contain "filename:linenum:content" format
      expect(result).toMatch(/:\d+:/);
    });

    test('respects include filter', () => {
      const result = platform.nodeGrep('Hello', tmpDir, '*.js');
      expect(result).toContain('beta.js');
      expect(result).not.toContain('alpha.txt');
    });

    test('returns "No matches" for no results', () => {
      const result = platform.nodeGrep('zzzzNotInAnyFile', tmpDir);
      expect(result).toMatch(/no matches/i);
    });

    test('handles regex special characters in pattern', () => {
      // Should not throw on regex chars
      const result = platform.nodeGrep('function\\(', tmpDir);
      expect(typeof result).toBe('string');
    });

    test('respects max results limit', () => {
      const result = platform.nodeGrep('Hello', tmpDir, null, 1);
      const lines = result.split('\n').filter(l => l.includes(':'));
      expect(lines.length).toBeLessThanOrEqual(1);
    });
  });

  // ── System Info ─────────────────────────────────────────────────────────

  describe('getCpuInfo', () => {
    test('returns object with name and cores', () => {
      const info = platform.getCpuInfo();
      expect(info).toHaveProperty('name');
      expect(info).toHaveProperty('cores');
      expect(typeof info.name).toBe('string');
    });

    test('cores is a positive number', () => {
      const info = platform.getCpuInfo();
      expect(Number(info.cores)).toBeGreaterThan(0);
    });
  });

  describe('getMemoryInfo', () => {
    test('returns object with totalGB and usagePct', () => {
      const info = platform.getMemoryInfo();
      expect(info).toHaveProperty('totalGB');
      expect(info).toHaveProperty('usagePct');
    });

    test('totalGB is reasonable', () => {
      const info = platform.getMemoryInfo();
      const gb = parseFloat(info.totalGB);
      expect(gb).toBeGreaterThan(1);
      expect(gb).toBeLessThan(2048);
    });

    test('usagePct is between 0 and 100', () => {
      const info = platform.getMemoryInfo();
      const pct = parseFloat(info.usagePct);
      expect(pct).toBeGreaterThanOrEqual(0);
      expect(pct).toBeLessThanOrEqual(100);
    });
  });

  describe('getDiskUsage', () => {
    test('returns a percentage string', () => {
      const usage = platform.getDiskUsage();
      const pct = parseFloat(usage);
      expect(pct).toBeGreaterThanOrEqual(0);
      expect(pct).toBeLessThanOrEqual(100);
    });
  });

  describe('getBatteryInfo', () => {
    test('returns object with percent and state', () => {
      const info = platform.getBatteryInfo();
      expect(info).toHaveProperty('percent');
      expect(info).toHaveProperty('state');
    });

    test('percent is null or a number', () => {
      const info = platform.getBatteryInfo();
      if (info.percent !== null) {
        expect(typeof info.percent).toBe('number');
        expect(info.percent).toBeGreaterThanOrEqual(0);
        expect(info.percent).toBeLessThanOrEqual(100);
      }
    });
  });

  describe('getGpuInfo', () => {
    test('returns a string', () => {
      const info = platform.getGpuInfo();
      expect(typeof info).toBe('string');
    });
  });

  // ── TTS ─────────────────────────────────────────────────────────────────

  describe('hasTTS', () => {
    test('returns a boolean', () => {
      expect(typeof platform.hasTTS()).toBe('boolean');
    });
  });

  describe('ttsSpeak', () => {
    test('returns error for empty text', () => {
      const result = platform.ttsSpeak('');
      expect(result).toHaveProperty('error');
    });

    test('returns error for null text', () => {
      const result = platform.ttsSpeak(null);
      expect(result).toHaveProperty('error');
    });
  });

  describe('ttsListVoices', () => {
    test('returns object with voices array', () => {
      const result = platform.ttsListVoices();
      expect(result).toHaveProperty('voices');
      expect(Array.isArray(result.voices)).toBe(true);
    });

    test('each voice has name and locale', () => {
      const result = platform.ttsListVoices();
      for (const voice of result.voices) {
        expect(voice).toHaveProperty('name');
        expect(voice).toHaveProperty('locale');
      }
    });
  });

  // ── getCpuUsage ─────────────────────────────────────────────────────────

  describe('getCpuUsage', () => {
    test('returns a number between 0 and 100', () => {
      const usage = platform.getCpuUsage();
      expect(typeof usage).toBe('number');
      expect(usage).toBeGreaterThanOrEqual(0);
      expect(usage).toBeLessThanOrEqual(100);
    });
  });
});
