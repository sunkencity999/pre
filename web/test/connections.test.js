// Tests for src/connections.js — credential management
const fs = require('fs');

jest.mock('../src/constants');
const { CONNECTIONS_FILE } = require('../src/constants');
const connections = require('../src/connections');

describe('connections', () => {
  beforeEach(() => {
    fs.writeFileSync(CONNECTIONS_FILE, '{}');
  });

  describe('setApiKey', () => {
    test('saves an API key', () => {
      connections.setApiKey('github', 'ghp_test123');
      const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
      expect(data.github_key).toBe('ghp_test123');
    });

    test('key appears as active in listConnections', () => {
      connections.setApiKey('brave_search', 'BSA-test');
      const list = connections.listConnections();
      expect(list.find(c => c.service === 'brave_search').active).toBe(true);
    });
  });

  describe('removeConnection', () => {
    test('removes connection data', () => {
      connections.setApiKey('github', 'ghp_xxx');
      connections.removeConnection('github');
      const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
      expect(data.github_key).toBeUndefined();
    });
  });

  describe('listConnections', () => {
    test('returns array of all services', () => {
      const list = connections.listConnections();
      expect(Array.isArray(list)).toBe(true);
      expect(list.length).toBeGreaterThan(5);
      const services = list.map(c => c.service);
      expect(services).toContain('github');
      expect(services).toContain('telegram');
    });

    test('masks API keys', () => {
      connections.setApiKey('github', 'ghp_verylongtoken123456');
      const gh = connections.listConnections().find(c => c.service === 'github');
      expect(gh.masked).not.toBe('ghp_verylongtoken123456');
      expect(gh.masked).toContain('...');
    });
  });

  describe('setJiraConfig', () => {
    test('saves Jira URL and token', () => {
      connections.setJiraConfig('https://jira.test', 'pat-123');
      const d = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
      expect(d.jira_url).toBe('https://jira.test');
      expect(d.jira_token).toBe('pat-123');
    });
  });

  describe('setConfluenceConfig', () => {
    test('saves Confluence URL and token', () => {
      connections.setConfluenceConfig('https://wiki.test', 'pat-456');
      const d = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
      expect(d.confluence_url).toBe('https://wiki.test');
    });
  });

  describe('setZoomConfig', () => {
    test('saves Zoom S2S credentials', () => {
      connections.setZoomConfig('acc', 'cli', 'sec');
      const d = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
      expect(d.zoom_account_id).toBe('acc');
      expect(d.zoom_client_id).toBe('cli');
      expect(d.zoom_client_secret).toBe('sec');
    });
  });

  describe('setD365Config', () => {
    test('saves D365 credentials', () => {
      connections.setD365Config('https://org.crm.dynamics.com', 't1', 'c2', 's3');
      const d = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
      expect(d.d365_url).toBe('https://org.crm.dynamics.com');
      expect(d.d365_tenant_id).toBe('t1');
    });
  });

  describe('setTelegramChatId', () => {
    test('saves chat ID', () => {
      connections.setTelegramChatId('12345');
      const d = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
      expect(d.telegram_chat_id).toBe('12345');
    });
  });
});
