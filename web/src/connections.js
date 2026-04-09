// PRE Web GUI — Connections management
// Reads/writes ~/.pre/connections.json, handles Google OAuth flow

const fs = require('fs');
const http = require('http');
const https = require('https');
const path = require('path');
const os = require('os');
const { URL } = require('url');
const { CONNECTIONS_FILE } = require('./constants');

const TELEGRAM_OWNER_FILE = path.join(os.homedir(), '.pre', 'telegram_owner');

const GOOGLE_AUTH_URL = 'https://accounts.google.com/o/oauth2/v2/auth';
const GOOGLE_TOKEN_URL = 'https://oauth2.googleapis.com/token';
const GOOGLE_SCOPES = [
  'https://www.googleapis.com/auth/gmail.modify',
  'https://www.googleapis.com/auth/gmail.compose',
  'https://www.googleapis.com/auth/gmail.send',
  'https://www.googleapis.com/auth/drive',
  'https://www.googleapis.com/auth/documents',
].join(' ');

function loadConnections() {
  try {
    return JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
  } catch {
    return {};
  }
}

function saveConnections(data) {
  fs.writeFileSync(CONNECTIONS_FILE, JSON.stringify(data, null, 2), { mode: 0o600 });
}

/**
 * Get the Telegram owner chat_id from connections or the CLI's owner file
 */
function getTelegramChatId() {
  const data = loadConnections();
  if (data.telegram_chat_id) return data.telegram_chat_id;
  try {
    const id = fs.readFileSync(TELEGRAM_OWNER_FILE, 'utf-8').trim();
    return id || null;
  } catch {
    return null;
  }
}

/**
 * Save the Telegram chat_id (writes to both connections.json and telegram_owner)
 */
function setTelegramChatId(chatId) {
  const data = loadConnections();
  data.telegram_chat_id = String(chatId);
  saveConnections(data);
  // Also write to the CLI's owner file for compatibility
  try {
    fs.writeFileSync(TELEGRAM_OWNER_FILE, String(chatId), { mode: 0o600 });
  } catch {}
}

/**
 * Test a Telegram bot token by calling getMe
 */
function testTelegramToken(token) {
  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'api.telegram.org',
      path: `/bot${token}/getMe`,
      method: 'GET',
    }, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try {
          const data = JSON.parse(body);
          if (!data.ok) return reject(new Error(data.description || 'Invalid token'));
          resolve(data.result);
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.end();
  });
}

/**
 * List connections with status (no secrets exposed)
 */
function listConnections() {
  const data = loadConnections();
  return [
    {
      service: 'brave_search',
      label: 'Brave Search',
      type: 'api_key',
      active: !!data.brave_search_key,
      masked: data.brave_search_key ? maskKey(data.brave_search_key) : null,
    },
    {
      service: 'github',
      label: 'GitHub',
      type: 'api_key',
      active: !!data.github_key,
      masked: data.github_key ? maskKey(data.github_key) : null,
    },
    {
      service: 'google',
      label: 'Google (Gmail, Drive, Docs)',
      type: 'oauth',
      active: !!data.google_client_id && !!data.google_refresh_token,
      hasCredentials: !!data.google_client_id,
      hasTokens: !!data.google_refresh_token,
      tokenExpiry: data.google_token_expiry ? new Date(data.google_token_expiry * 1000).toISOString() : null,
    },
    {
      service: 'wolfram',
      label: 'Wolfram Alpha',
      type: 'api_key',
      active: !!data.wolfram_key,
      masked: data.wolfram_key ? maskKey(data.wolfram_key) : null,
    },
    {
      service: 'telegram',
      label: 'Telegram Bot',
      type: 'telegram',
      active: !!data.telegram_key,
      masked: data.telegram_key ? maskKey(data.telegram_key) : null,
      chatId: getTelegramChatId(),
    },
    {
      service: 'jira',
      label: 'Jira Server',
      type: 'jira',
      active: !!data.jira_url && !!data.jira_token,
      url: data.jira_url || null,
      masked: data.jira_token ? maskKey(data.jira_token) : null,
    },
    {
      service: 'confluence',
      label: 'Confluence Server',
      type: 'confluence',
      active: !!data.confluence_url && !!data.confluence_token,
      url: data.confluence_url || null,
      masked: data.confluence_token ? maskKey(data.confluence_token) : null,
    },
    {
      service: 'smartsheet',
      label: 'Smartsheet',
      type: 'api_key',
      active: !!data.smartsheet_token,
      masked: data.smartsheet_token ? maskKey(data.smartsheet_token) : null,
    },
    {
      service: 'slack',
      label: 'Slack',
      type: 'slack',
      active: !!data.slack_token,
      masked: data.slack_token ? maskKey(data.slack_token) : null,
    },
  ];
}

function maskKey(key) {
  if (!key || key.length < 8) return '****';
  return key.slice(0, 4) + '...' + key.slice(-4);
}

/**
 * Set an API key connection
 */
function setApiKey(service, key) {
  const data = loadConnections();
  const keyMap = {
    brave_search: 'brave_search_key',
    github: 'github_key',
    wolfram: 'wolfram_key',
    telegram: 'telegram_key',
    smartsheet: 'smartsheet_token',
    slack: 'slack_token',
  };
  const field = keyMap[service];
  if (!field) return false;
  if (key) {
    data[field] = key;
  } else {
    delete data[field];
  }
  saveConnections(data);
  return true;
}

/**
 * Remove a connection
 */
function removeConnection(service) {
  const data = loadConnections();
  if (service === 'google') {
    delete data.google_client_id;
    delete data.google_client_secret;
    delete data.google_access_token;
    delete data.google_refresh_token;
    delete data.google_token_expiry;
  } else if (service === 'jira') {
    delete data.jira_url;
    delete data.jira_token;
  } else if (service === 'confluence') {
    delete data.confluence_url;
    delete data.confluence_token;
  } else {
    return setApiKey(service, null);
  }
  saveConnections(data);
  return true;
}

/**
 * Set Google OAuth client credentials
 */
function setGoogleCredentials(clientId, clientSecret) {
  const data = loadConnections();
  data.google_client_id = clientId;
  data.google_client_secret = clientSecret;
  saveConnections(data);
  return true;
}

/**
 * Start Google OAuth flow — opens a temporary server to receive the callback
 * Returns the authorization URL to redirect the user to
 */
function getGoogleAuthUrl(redirectPort) {
  const data = loadConnections();
  if (!data.google_client_id) return null;
  const redirectUri = `http://localhost:${redirectPort}/oauth/callback`;
  const params = new URLSearchParams({
    client_id: data.google_client_id,
    redirect_uri: redirectUri,
    response_type: 'code',
    scope: GOOGLE_SCOPES,
    access_type: 'offline',
    prompt: 'consent',
  });
  return `${GOOGLE_AUTH_URL}?${params.toString()}`;
}

/**
 * Exchange authorization code for tokens
 */
function exchangeGoogleCode(code, redirectPort) {
  return new Promise((resolve, reject) => {
    const data = loadConnections();
    if (!data.google_client_id || !data.google_client_secret) {
      return reject(new Error('Google credentials not configured'));
    }
    const redirectUri = `http://localhost:${redirectPort}/oauth/callback`;
    const postData = new URLSearchParams({
      code,
      client_id: data.google_client_id,
      client_secret: data.google_client_secret,
      redirect_uri: redirectUri,
      grant_type: 'authorization_code',
    }).toString();

    const req = https.request(GOOGLE_TOKEN_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Content-Length': Buffer.byteLength(postData),
      },
    }, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try {
          const tokens = JSON.parse(body);
          if (tokens.error) return reject(new Error(tokens.error_description || tokens.error));
          // Save tokens
          const connData = loadConnections();
          connData.google_access_token = tokens.access_token;
          if (tokens.refresh_token) connData.google_refresh_token = tokens.refresh_token;
          connData.google_token_expiry = Math.floor(Date.now() / 1000) + (tokens.expires_in || 3600);
          saveConnections(connData);
          resolve({ success: true });
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

/**
 * Refresh Google access token using refresh token
 */
function refreshGoogleToken() {
  return new Promise((resolve, reject) => {
    const data = loadConnections();
    if (!data.google_refresh_token || !data.google_client_id || !data.google_client_secret) {
      return reject(new Error('No refresh token available'));
    }
    const postData = new URLSearchParams({
      refresh_token: data.google_refresh_token,
      client_id: data.google_client_id,
      client_secret: data.google_client_secret,
      grant_type: 'refresh_token',
    }).toString();

    const req = https.request(GOOGLE_TOKEN_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Content-Length': Buffer.byteLength(postData),
      },
    }, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try {
          const tokens = JSON.parse(body);
          if (tokens.error) return reject(new Error(tokens.error_description || tokens.error));
          data.google_access_token = tokens.access_token;
          data.google_token_expiry = Math.floor(Date.now() / 1000) + (tokens.expires_in || 3600);
          saveConnections(data);
          resolve({ success: true });
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

/**
 * Set Jira Server URL and personal access token
 */
function setJiraConfig(url, token) {
  const data = loadConnections();
  data.jira_url = url.replace(/\/+$/, '');
  data.jira_token = token;
  saveConnections(data);
  return true;
}

/**
 * Set Confluence Server URL and personal access token
 */
function setConfluenceConfig(url, token) {
  const data = loadConnections();
  data.confluence_url = url.replace(/\/+$/, '');
  data.confluence_token = token;
  saveConnections(data);
  return true;
}

module.exports = {
  listConnections,
  setApiKey,
  removeConnection,
  setGoogleCredentials,
  getGoogleAuthUrl,
  exchangeGoogleCode,
  refreshGoogleToken,
  setTelegramChatId,
  testTelegramToken,
  setJiraConfig,
  setConfluenceConfig,
};
