// Tests for Progressive Tool Disclosure (tool-tiers.js + buildToolDefs filtering)

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

beforeEach(() => {
  fs.writeFileSync(CONNECTIONS_FILE, '{}');
});

describe('tool-tiers — domain taxonomy', () => {
  const tiers = require('../src/tool-tiers');

  test('exports CORE_TOOLS as a Set', () => {
    expect(tiers.CORE_TOOLS).toBeInstanceOf(Set);
    expect(tiers.CORE_TOOLS.size).toBeGreaterThan(15);
  });

  test('exports DOMAINS as an object with expected keys', () => {
    const expected = ['devops', 'desktop', 'pim', 'code', 'media', 'automation', 'cloud'];
    for (const d of expected) {
      expect(tiers.DOMAINS).toHaveProperty(d);
      expect(Array.isArray(tiers.DOMAINS[d])).toBe(true);
      expect(tiers.DOMAINS[d].length).toBeGreaterThan(0);
    }
  });

  test('CORE_TOOLS includes essential tools', () => {
    const essential = ['bash', 'read_file', 'file_write', 'web_fetch', 'request_tools', 'session_search'];
    for (const t of essential) {
      expect(tiers.CORE_TOOLS.has(t)).toBe(true);
    }
  });

  test('domain tools do not overlap with CORE', () => {
    for (const [domain, tools] of Object.entries(tiers.DOMAINS)) {
      for (const tool of tools) {
        expect(tiers.CORE_TOOLS.has(tool)).toBe(false);
      }
    }
  });

  test('no tool appears in multiple domains', () => {
    const seen = new Map();
    for (const [domain, tools] of Object.entries(tiers.DOMAINS)) {
      for (const tool of tools) {
        if (seen.has(tool)) {
          fail(`Tool "${tool}" appears in both "${seen.get(tool)}" and "${domain}"`);
        }
        seen.set(tool, domain);
      }
    }
  });
});

describe('tool-tiers — keyword detection', () => {
  const tiers = require('../src/tool-tiers');

  test('resolveKeywords activates pim for email-related messages', () => {
    const sid = 'test-kw-pim';
    tiers.clearSession(sid);
    const activated = tiers.resolveKeywords(sid, 'Please check my email inbox');
    expect(activated).toContain('pim');
    expect(tiers.getActiveDomains(sid).has('pim')).toBe(true);
  });

  test('resolveKeywords activates devops for process/kill messages', () => {
    const sid = 'test-kw-devops';
    tiers.clearSession(sid);
    const activated = tiers.resolveKeywords(sid, 'Kill the node process on port 3000');
    expect(activated).toContain('devops');
  });

  test('resolveKeywords activates cloud for github messages', () => {
    const sid = 'test-kw-cloud';
    tiers.clearSession(sid);
    const activated = tiers.resolveKeywords(sid, 'Create a pull request on github');
    expect(activated).toContain('cloud');
  });

  test('resolveKeywords activates desktop for click/screenshot messages', () => {
    const sid = 'test-kw-desktop';
    tiers.clearSession(sid);
    const activated = tiers.resolveKeywords(sid, 'Take a screenshot and click the button');
    expect(activated).toContain('desktop');
  });

  test('resolveKeywords returns empty for unrelated messages', () => {
    const sid = 'test-kw-none';
    tiers.clearSession(sid);
    const activated = tiers.resolveKeywords(sid, 'Hello, how are you today?');
    expect(activated).toEqual([]);
  });

  test('resolveKeywords does not re-activate already active domains', () => {
    const sid = 'test-kw-dedup';
    tiers.clearSession(sid);
    tiers.resolveKeywords(sid, 'Check my email');
    const second = tiers.resolveKeywords(sid, 'Also check my calendar');
    // pim already active from first call, so it should not be in second result
    expect(second).toEqual([]);
  });

  test('resolveKeywords handles null/empty input', () => {
    const sid = 'test-kw-null';
    expect(tiers.resolveKeywords(sid, null)).toEqual([]);
    expect(tiers.resolveKeywords(sid, '')).toEqual([]);
  });
});

describe('tool-tiers — session state', () => {
  const tiers = require('../src/tool-tiers');

  test('getActiveDomains returns empty Set for new session', () => {
    const sid = 'test-state-new';
    tiers.clearSession(sid);
    const domains = tiers.getActiveDomains(sid);
    expect(domains).toBeInstanceOf(Set);
    expect(domains.size).toBe(0);
  });

  test('activateDomain adds domain to session', () => {
    const sid = 'test-state-activate';
    tiers.clearSession(sid);
    const result = tiers.activateDomain(sid, 'devops');
    expect(result).toEqual(['devops']);
    expect(tiers.getActiveDomains(sid).has('devops')).toBe(true);
  });

  test('activateDomain with "all" activates all domains', () => {
    const sid = 'test-state-all';
    tiers.clearSession(sid);
    const result = tiers.activateDomain(sid, 'all');
    expect(result.length).toBe(Object.keys(tiers.DOMAINS).length);
    for (const d of Object.keys(tiers.DOMAINS)) {
      expect(tiers.getActiveDomains(sid).has(d)).toBe(true);
    }
  });

  test('activateDomain returns null for unknown domain', () => {
    const sid = 'test-state-unknown';
    tiers.clearSession(sid);
    expect(tiers.activateDomain(sid, 'nonexistent')).toBeNull();
  });

  test('clearSession removes session state', () => {
    const sid = 'test-state-clear';
    tiers.activateDomain(sid, 'pim');
    expect(tiers.getActiveDomains(sid).size).toBeGreaterThan(0);
    tiers.clearSession(sid);
    expect(tiers.getActiveDomains(sid).size).toBe(0);
  });
});

describe('tool-tiers — isToolAllowed', () => {
  const tiers = require('../src/tool-tiers');

  test('CORE tools are always allowed', () => {
    const empty = new Set();
    expect(tiers.isToolAllowed('bash', empty)).toBe(true);
    expect(tiers.isToolAllowed('request_tools', empty)).toBe(true);
    expect(tiers.isToolAllowed('session_search', empty)).toBe(true);
  });

  test('domain tools blocked without active domain', () => {
    const empty = new Set();
    expect(tiers.isToolAllowed('apple_mail', empty)).toBe(false);
    expect(tiers.isToolAllowed('github', empty)).toBe(false);
    expect(tiers.isToolAllowed('computer', empty)).toBe(false);
  });

  test('domain tools allowed with active domain', () => {
    const active = new Set(['pim', 'cloud']);
    expect(tiers.isToolAllowed('apple_mail', active)).toBe(true);
    expect(tiers.isToolAllowed('github', active)).toBe(true);
    expect(tiers.isToolAllowed('computer', active)).toBe(false); // desktop not active
  });

  test('unknown tools (custom, MCP) always allowed', () => {
    const empty = new Set();
    expect(tiers.isToolAllowed('custom_my_tool', empty)).toBe(true);
    expect(tiers.isToolAllowed('mcp_some_server_tool', empty)).toBe(true);
  });
});

describe('tool-tiers — listDomains', () => {
  const tiers = require('../src/tool-tiers');

  test('returns array of domain info objects', () => {
    const domains = tiers.listDomains();
    expect(Array.isArray(domains)).toBe(true);
    expect(domains.length).toBe(7);
    for (const d of domains) {
      expect(d).toHaveProperty('name');
      expect(d).toHaveProperty('tools');
      expect(d).toHaveProperty('toolNames');
      expect(typeof d.tools).toBe('number');
      expect(Array.isArray(d.toolNames)).toBe(true);
    }
  });
});

describe('buildToolDefs — progressive disclosure integration', () => {
  const { buildToolDefs } = require('../src/tools-defs');

  test('no opts returns all tools (backward compatible)', () => {
    const all = buildToolDefs();
    expect(all.length).toBeGreaterThan(30);
  });

  test('empty activeDomains returns only core + undomained tools', () => {
    const core = buildToolDefs({ activeDomains: new Set() });
    const all = buildToolDefs();
    expect(core.length).toBeLessThan(all.length);
    // Core tools present
    const names = core.map(t => t.function.name);
    expect(names).toContain('bash');
    expect(names).toContain('request_tools');
    expect(names).toContain('session_search');
  });

  test('activating pim adds PIM tools', () => {
    const withPim = buildToolDefs({ activeDomains: new Set(['pim']) });
    const names = withPim.map(t => t.function.name);
    expect(names).toContain('apple_calendar');
    expect(names).toContain('apple_contacts');
    expect(names).toContain('apple_reminders');
    expect(names).toContain('apple_notes');
  });

  test('activating all domains gives same count as no filtering', () => {
    const tiers = require('../src/tool-tiers');
    const allDomains = new Set(Object.keys(tiers.DOMAINS));
    const withAll = buildToolDefs({ activeDomains: allDomains });
    const noFilter = buildToolDefs();
    expect(withAll.length).toBe(noFilter.length);
  });

  test('domain tools excluded when domain not active', () => {
    const core = buildToolDefs({ activeDomains: new Set() });
    const names = core.map(t => t.function.name);
    // These should NOT be in core-only
    expect(names).not.toContain('process_kill');
    expect(names).not.toContain('computer');
    expect(names).not.toContain('cron');
    expect(names).not.toContain('github');
  });
});
