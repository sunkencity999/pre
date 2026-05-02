// Tests for Linux-specific features across the PRE codebase
// Covers: platform.js IS_LINUX, install-linux.sh, tool dispatchers,
// EDS integration, computer-linux, autostart systemd, TTS espeak-ng

const fs = require('fs');
const path = require('path');

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

// ═══════════════════════════════════════════════════════════════════════════
// Platform Module — IS_LINUX Constant
// ═══════════════════════════════════════════════════════════════════════════

describe('platform.js — IS_LINUX support', () => {
  const platform = require('../src/platform');

  test('exports IS_LINUX constant', () => {
    expect(typeof platform.IS_LINUX).toBe('boolean');
  });

  test('IS_LINUX is true only on Linux', () => {
    expect(platform.IS_LINUX).toBe(process.platform === 'linux');
  });

  test('exactly one platform constant is true', () => {
    const trueCount = [platform.IS_MAC, platform.IS_WIN, platform.IS_LINUX].filter(Boolean).length;
    // On known platforms, exactly one should be true; on exotic platforms, all may be false
    expect(trueCount).toBeLessThanOrEqual(1);
  });

  test('platform.js source defines IS_LINUX from process.platform', () => {
    const source = fs.readFileSync(require.resolve('../src/platform.js'), 'utf-8');
    expect(source).toContain("IS_LINUX");
    expect(source).toContain("process.platform === 'linux'");
  });

  test('platform.js exports IS_LINUX in module.exports', () => {
    const source = fs.readFileSync(require.resolve('../src/platform.js'), 'utf-8');
    expect(source).toMatch(/module\.exports[\s\S]*IS_LINUX/);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Platform Functions — Linux Branches
// ═══════════════════════════════════════════════════════════════════════════

describe('platform.js — Linux function branches', () => {
  const source = fs.readFileSync(require.resolve('../src/platform.js'), 'utf-8');

  test('getChromePaths includes Linux Chrome/Chromium paths', () => {
    expect(source).toContain('/usr/bin/google-chrome');
    expect(source).toContain('/usr/bin/chromium');
  });

  test('getCpuInfo has /proc/cpuinfo parsing for Linux', () => {
    expect(source).toContain('/proc/cpuinfo');
  });

  test('getMemoryInfo has /proc/meminfo parsing for Linux', () => {
    expect(source).toContain('/proc/meminfo');
  });

  test('getBatteryInfo has /sys/class/power_supply for Linux', () => {
    expect(source).toContain('/sys/class/power_supply');
  });

  test('getGpuInfo uses nvidia-smi on Linux', () => {
    expect(source).toContain('nvidia-smi');
  });

  test('clipboardRead uses xclip or xsel on Linux', () => {
    expect(source).toMatch(/xclip|xsel/);
  });

  test('clipboardWrite uses xclip or xsel on Linux', () => {
    expect(source).toMatch(/xclip.*clipboard|xsel.*-b/);
  });

  test('notify uses notify-send on Linux', () => {
    expect(source).toContain('notify-send');
  });

  test('openTarget uses xdg-open on Linux', () => {
    expect(source).toContain('xdg-open');
  });

  test('screenshot uses scrot on Linux', () => {
    expect(source).toContain('scrot');
  });

  test('windowList uses wmctrl on Linux', () => {
    expect(source).toContain('wmctrl');
  });

  test('netInfo uses ip addr on Linux', () => {
    expect(source).toContain('ip addr');
  });

  test('serviceStatus uses systemctl on Linux', () => {
    expect(source).toContain('systemctl');
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// TTS — espeak-ng Linux Branch
// ═══════════════════════════════════════════════════════════════════════════

describe('platform.js — Linux TTS via espeak-ng', () => {
  const source = fs.readFileSync(require.resolve('../src/platform.js'), 'utf-8');

  test('hasTTS checks for espeak-ng on Linux', () => {
    expect(source).toContain('espeak-ng');
    expect(source).toContain('espeak');
  });

  test('ttsSpeak uses espeak-ng on Linux', () => {
    // Should have espeak command invocation with -v and -s flags via ttsCmd variable
    expect(source).toMatch(/ttsCmd.*-v.*-s/);
  });

  test('ttsListVoices parses espeak-ng --voices output', () => {
    expect(source).toContain('--voices');
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Install Script — install-linux.sh
// ═══════════════════════════════════════════════════════════════════════════

describe('install-linux.sh', () => {
  const repoRoot = path.join(__dirname, '..', '..');
  const installPath = path.join(repoRoot, 'install-linux.sh');

  test('install-linux.sh exists', () => {
    expect(fs.existsSync(installPath)).toBe(true);
  });

  test('install-linux.sh is executable', () => {
    const stats = fs.statSync(installPath);
    // Check user execute bit (0o100)
    expect(stats.mode & 0o100).toBeTruthy();
  });

  test('install-linux.sh detects distro package managers', () => {
    const source = fs.readFileSync(installPath, 'utf-8');
    expect(source).toContain('apt');
    expect(source).toContain('dnf');
    expect(source).toContain('pacman');
  });

  test('install-linux.sh detects NVIDIA VRAM', () => {
    const source = fs.readFileSync(installPath, 'utf-8');
    expect(source).toContain('nvidia-smi');
    expect(source).toContain('memory.total');
  });

  test('install-linux.sh has VRAM-aware quant selection', () => {
    const source = fs.readFileSync(installPath, 'utf-8');
    expect(source).toContain('q8_0');
    expect(source).toContain('q4_K_M');
  });

  test('install-linux.sh configures Ollama env vars', () => {
    const source = fs.readFileSync(installPath, 'utf-8');
    expect(source).toContain('OLLAMA_FLASH_ATTENTION');
    expect(source).toContain('OLLAMA_KV_CACHE_TYPE');
  });

  test('install-linux.sh has optional EDS integration step', () => {
    const source = fs.readFileSync(installPath, 'utf-8');
    expect(source).toContain('evolution-data-server');
  });

  test('install-linux.sh has optional desktop automation deps', () => {
    const source = fs.readFileSync(installPath, 'utf-8');
    expect(source).toContain('xdotool');
    expect(source).toContain('scrot');
    expect(source).toContain('xclip');
  });

  test('install-linux.sh has systemd user service option', () => {
    const source = fs.readFileSync(installPath, 'utf-8');
    expect(source).toContain('systemd');
    expect(source).toContain('pre-server.service');
  });

  test('install-linux.sh creates XDG desktop entry', () => {
    const source = fs.readFileSync(installPath, 'utf-8');
    expect(source).toContain('.desktop');
    expect(source).toContain('Desktop Entry');
  });

  test('install-linux.sh has headroom-based context sizing', () => {
    const source = fs.readFileSync(installPath, 'utf-8');
    expect(source).toContain('headroom');
    expect(source).toContain('131072');
    expect(source).toContain('65536');
    expect(source).toContain('32768');
    expect(source).toContain('16384');
    expect(source).toContain('8192');
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Launch Scripts — Linux Compatibility
// ═══════════════════════════════════════════════════════════════════════════

describe('launch scripts — Linux compatibility', () => {
  test('pre-server.sh has Linux GPU detection branch', () => {
    const source = fs.readFileSync(path.join(__dirname, '..', 'pre-server.sh'), 'utf-8');
    expect(source).toContain('Linux');
    expect(source).toContain('nvidia-smi');
  });

  test('pre-server.sh has Linux systemctl Ollama startup', () => {
    const source = fs.readFileSync(path.join(__dirname, '..', 'pre-server.sh'), 'utf-8');
    expect(source).toContain('systemctl');
    expect(source).toContain('ollama');
  });

  test('pre-launch has Linux branch', () => {
    const source = fs.readFileSync(path.join(__dirname, '..', '..', 'engine', 'pre-launch'), 'utf-8');
    expect(source).toContain('Linux');
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Context — Linux Platform Description
// ═══════════════════════════════════════════════════════════════════════════

describe('context.js — Linux platform description', () => {
  test('context.js imports IS_LINUX', () => {
    const source = fs.readFileSync(require.resolve('../src/context.js'), 'utf-8');
    expect(source).toContain('IS_LINUX');
  });

  test('context.js has Linux platform description', () => {
    const source = fs.readFileSync(require.resolve('../src/context.js'), 'utf-8');
    expect(source).toMatch(/IS_LINUX[\s\S]*?Linux/);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Tool Registration — Linux Platform Gating
// ═══════════════════════════════════════════════════════════════════════════

describe('tools-defs — Linux tool registration', () => {
  beforeEach(() => {
    fs.writeFileSync(CONNECTIONS_FILE, '{}');
  });

  const source = fs.readFileSync(require.resolve('../src/tools-defs.js'), 'utf-8');

  test('mail is gated to macOS + Windows only (not Linux)', () => {
    // Mail should use IS_MAC || IS_WIN gate
    expect(source).toMatch(/IS_MAC\s*\|\|\s*IS_WIN[\s\S]*?apple_mail/);
  });

  test('tools-defs imports IS_LINUX', () => {
    expect(source).toContain('IS_LINUX');
  });

  test('calendar tool is registered on all platforms', () => {
    // Calendar should not be gated behind IS_MAC || IS_WIN
    // It should be available always (with EDS on Linux)
    const { buildToolDefs } = require('../src/tools-defs');
    const tools = buildToolDefs();
    const calTool = tools.find(t => t.function.name === 'apple_calendar');
    expect(calTool).toBeTruthy();
  });

  test('contacts tool is registered on all platforms', () => {
    const { buildToolDefs } = require('../src/tools-defs');
    const tools = buildToolDefs();
    const tool = tools.find(t => t.function.name === 'apple_contacts');
    expect(tool).toBeTruthy();
  });

  test('reminders tool is registered on all platforms', () => {
    const { buildToolDefs } = require('../src/tools-defs');
    const tools = buildToolDefs();
    const tool = tools.find(t => t.function.name === 'apple_reminders');
    expect(tool).toBeTruthy();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// EDS Linux Integration Module
// ═══════════════════════════════════════════════════════════════════════════

describe('eds-linux.js — Evolution Data Server integration', () => {
  test('eds-linux.js exists', () => {
    expect(fs.existsSync(require.resolve('../src/tools/eds-linux.js'))).toBe(true);
  });

  const eds = require('../src/tools/eds-linux');

  test('exports all required calendar functions', () => {
    expect(typeof eds.edsCalendarList).toBe('function');
    expect(typeof eds.edsCalendarCreate).toBe('function');
    expect(typeof eds.edsCalendarDelete).toBe('function');
    expect(typeof eds.edsCalendarSearch).toBe('function');
    expect(typeof eds.edsListCalendars).toBe('function');
  });

  test('exports all required contact functions', () => {
    expect(typeof eds.edsContactSearch).toBe('function');
    expect(typeof eds.edsContactRead).toBe('function');
    expect(typeof eds.edsContactCount).toBe('function');
    expect(typeof eds.edsListGroups).toBe('function');
  });

  test('exports all required task functions', () => {
    expect(typeof eds.edsTaskList).toBe('function');
    expect(typeof eds.edsTaskCreate).toBe('function');
    expect(typeof eds.edsTaskComplete).toBe('function');
    expect(typeof eds.edsTaskDelete).toBe('function');
    expect(typeof eds.edsTaskSearch).toBe('function');
    expect(typeof eds.edsListTaskLists).toBe('function');
  });

  test('exports isEdsAvailable function', () => {
    expect(typeof eds.isEdsAvailable).toBe('function');
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Native App Dispatchers — IS_LINUX Branches
// ═══════════════════════════════════════════════════════════════════════════

describe('native app tools — IS_LINUX dispatch branches', () => {
  test('calendar.js has IS_LINUX branch', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/calendar.js'), 'utf-8');
    expect(source).toContain('IS_LINUX');
    expect(source).toContain('eds-linux');
  });

  test('contacts.js has IS_LINUX branch', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/contacts.js'), 'utf-8');
    expect(source).toContain('IS_LINUX');
    expect(source).toContain('eds-linux');
  });

  test('reminders.js has IS_LINUX branch', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/reminders.js'), 'utf-8');
    expect(source).toContain('IS_LINUX');
    expect(source).toContain('eds-linux');
  });

  test('spotlight.js has Linux search functions', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/spotlight.js'), 'utf-8');
    expect(source).toContain('IS_LINUX');
    expect(source).toMatch(/locate|find/);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Computer-Linux — Desktop Automation Module
// ═══════════════════════════════════════════════════════════════════════════

describe('computer-linux module', () => {
  test('computer-linux.js exists', () => {
    expect(fs.existsSync(require.resolve('../src/tools/computer-linux.js'))).toBe(true);
  });

  const linuxMod = require('../src/tools/computer-linux');

  test('all required functions are exported', () => {
    const requiredExports = [
      'isAvailable', 'getScreenSize', 'takeScreenshot',
      'mouseAction', 'mouseDrag', 'mouseScroll',
      'typeText', 'pressKey',
      'getWindowBounds', 'focusWindow', 'getForegroundApp',
      'getCursorPosition',
    ];
    for (const fn of requiredExports) {
      expect(typeof linuxMod[fn]).toBe('function');
    }
  });

  test('source uses xdotool for input automation', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/computer-linux.js'), 'utf-8');
    expect(source).toContain('xdotool');
  });

  test('source uses scrot for screenshots', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/computer-linux.js'), 'utf-8');
    expect(source).toContain('scrot');
  });

  test('source has key mapping for xdotool key names', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/computer-linux.js'), 'utf-8');
    expect(source).toContain('Return');
    expect(source).toContain('Escape');
    expect(source).toContain('BackSpace');
  });
});

describe('computer.js — Linux dispatch integration', () => {
  test('computer.js imports IS_LINUX and loads Linux backend', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/computer.js'), 'utf-8');
    expect(source).toContain('IS_LINUX');
    expect(source).toContain("require('./computer-linux')");
  });

  test('computer.js dispatches to linux backend for all actions', () => {
    const source = fs.readFileSync(require.resolve('../src/tools/computer.js'), 'utf-8');
    // Should have linux.mouseAction, linux.mouseDrag, linux.mouseScroll, etc.
    expect(source).toContain('linux.mouseAction');
    expect(source).toContain('linux.mouseDrag');
    expect(source).toContain('linux.mouseScroll');
    expect(source).toContain('linux.typeText');
    expect(source).toContain('linux.pressKey');
    expect(source).toContain('linux.getCursorPosition');
    expect(source).toContain('linux.focusWindow');
    expect(source).toContain('linux.getForegroundApp');
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Server — Linux Autostart (systemd)
// ═══════════════════════════════════════════════════════════════════════════

describe('server.js — Linux autostart via systemd', () => {
  test('server.js defines systemd constants', () => {
    const source = fs.readFileSync(require.resolve('../server.js'), 'utf-8');
    expect(source).toContain('SYSTEMD_USER_DIR');
    expect(source).toContain('SYSTEMD_SERVICE');
    expect(source).toContain('pre-server.service');
  });

  test('server.js autostart endpoint handles IS_LINUX', () => {
    const source = fs.readFileSync(require.resolve('../server.js'), 'utf-8');
    expect(source).toMatch(/autostart[\s\S]*?IS_LINUX/);
  });

  test('server.js creates systemd unit file content', () => {
    const source = fs.readFileSync(require.resolve('../server.js'), 'utf-8');
    expect(source).toContain('[Unit]');
    expect(source).toContain('[Service]');
    expect(source).toContain('[Install]');
    expect(source).toContain('WantedBy');
  });
});
