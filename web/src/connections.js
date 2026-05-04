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

// Microsoft / SharePoint OAuth 2.0
const MICROSOFT_SCOPES = 'https://graph.microsoft.com/Sites.Read.All https://graph.microsoft.com/Files.ReadWrite.All https://graph.microsoft.com/User.Read offline_access';

// Dynamics 365 / Dataverse OAuth 2.0 (delegated)
// Scope is built dynamically: `${d365_url}/.default offline_access`

function microsoftAuthUrl(tenantId) {
  return `https://login.microsoftonline.com/${tenantId}/oauth2/v2.0/authorize`;
}
function microsoftTokenUrl(tenantId) {
  return `https://login.microsoftonline.com/${tenantId}/oauth2/v2.0/token`;
}

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
    req.setTimeout(15000, () => { req.destroy(); reject(new Error('Telegram token test timed out')); });
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
    {
      service: 'microsoft',
      label: 'Microsoft SharePoint',
      type: 'oauth',
      active: !!data.microsoft_client_id && !!data.microsoft_refresh_token,
      hasCredentials: !!data.microsoft_client_id,
      hasTokens: !!data.microsoft_refresh_token,
      tenantId: data.microsoft_tenant_id || null,
      tokenExpiry: data.microsoft_token_expiry ? new Date(data.microsoft_token_expiry * 1000).toISOString() : null,
    },
    {
      service: 'linear',
      label: 'Linear',
      type: 'api_key',
      active: !!data.linear_token,
      masked: data.linear_token ? maskKey(data.linear_token) : null,
    },
    {
      service: 'zoom',
      label: 'Zoom',
      type: 'zoom',
      active: !!data.zoom_account_id && !!data.zoom_client_id && !!data.zoom_client_secret,
      hasCredentials: !!data.zoom_client_id,
      accountId: data.zoom_account_id || null,
    },
    {
      service: 'figma',
      label: 'Figma',
      type: 'api_key',
      active: !!data.figma_token,
      masked: data.figma_token ? maskKey(data.figma_token) : null,
    },
    {
      service: 'asana',
      label: 'Asana',
      type: 'api_key',
      active: !!data.asana_token,
      masked: data.asana_token ? maskKey(data.asana_token) : null,
    },
    {
      service: 'dynamics365',
      label: 'Dynamics 365',
      type: 'dynamics365',
      active: !!data.d365_url && !!data.d365_client_id && !!data.d365_client_secret && !!data.d365_tenant_id,
      url: data.d365_url || null,
      masked: data.d365_client_id ? maskKey(data.d365_client_id) : null,
      hasCredentials: !!data.d365_client_id && !!data.d365_tenant_id,
      hasTokens: !!data.d365_refresh_token,
      authMode: data.d365_refresh_token ? 'delegated' : 'client_credentials',
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
    linear: 'linear_token',
    figma: 'figma_token',
    asana: 'asana_token',
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
  } else if (service === 'microsoft') {
    delete data.microsoft_tenant_id;
    delete data.microsoft_client_id;
    delete data.microsoft_client_secret;
    delete data.microsoft_access_token;
    delete data.microsoft_refresh_token;
    delete data.microsoft_token_expiry;
  } else if (service === 'zoom') {
    delete data.zoom_account_id;
    delete data.zoom_client_id;
    delete data.zoom_client_secret;
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
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Google OAuth exchange timed out')); });
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
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Google token refresh timed out')); });
    req.write(postData);
    req.end();
  });
}

/**
 * Set Microsoft / SharePoint OAuth client credentials
 */
function setMicrosoftCredentials(tenantId, clientId, clientSecret) {
  const data = loadConnections();
  data.microsoft_tenant_id = tenantId;
  data.microsoft_client_id = clientId;
  data.microsoft_client_secret = clientSecret;
  saveConnections(data);
  return true;
}

/**
 * Get Microsoft OAuth authorization URL
 */
function getMicrosoftAuthUrl(redirectPort) {
  const data = loadConnections();
  if (!data.microsoft_client_id || !data.microsoft_tenant_id) return null;
  const redirectUri = `http://localhost:${redirectPort}/oauth/microsoft/callback`;
  const params = new URLSearchParams({
    client_id: data.microsoft_client_id,
    redirect_uri: redirectUri,
    response_type: 'code',
    scope: MICROSOFT_SCOPES,
    response_mode: 'query',
  });
  return `${microsoftAuthUrl(data.microsoft_tenant_id)}?${params.toString()}`;
}

/**
 * Exchange Microsoft authorization code for tokens
 */
function exchangeMicrosoftCode(code, redirectPort) {
  return new Promise((resolve, reject) => {
    const data = loadConnections();
    if (!data.microsoft_client_id || !data.microsoft_client_secret || !data.microsoft_tenant_id) {
      return reject(new Error('Microsoft credentials not configured'));
    }
    const redirectUri = `http://localhost:${redirectPort}/oauth/microsoft/callback`;
    const postData = new URLSearchParams({
      code,
      client_id: data.microsoft_client_id,
      client_secret: data.microsoft_client_secret,
      redirect_uri: redirectUri,
      grant_type: 'authorization_code',
      scope: MICROSOFT_SCOPES,
    }).toString();

    const tokenUrl = new URL(microsoftTokenUrl(data.microsoft_tenant_id));
    const req = https.request({
      hostname: tokenUrl.hostname,
      path: tokenUrl.pathname,
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
          const connData = loadConnections();
          connData.microsoft_access_token = tokens.access_token;
          if (tokens.refresh_token) connData.microsoft_refresh_token = tokens.refresh_token;
          connData.microsoft_token_expiry = Math.floor(Date.now() / 1000) + (tokens.expires_in || 3600);
          saveConnections(connData);
          resolve({ success: true });
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Microsoft OAuth exchange timed out')); });
    req.write(postData);
    req.end();
  });
}

/**
 * Refresh Microsoft access token using refresh token
 */
function refreshMicrosoftToken() {
  return new Promise((resolve, reject) => {
    const data = loadConnections();
    if (!data.microsoft_refresh_token || !data.microsoft_client_id || !data.microsoft_client_secret || !data.microsoft_tenant_id) {
      return reject(new Error('No Microsoft refresh token available'));
    }
    const postData = new URLSearchParams({
      refresh_token: data.microsoft_refresh_token,
      client_id: data.microsoft_client_id,
      client_secret: data.microsoft_client_secret,
      grant_type: 'refresh_token',
      scope: MICROSOFT_SCOPES,
    }).toString();

    const tokenUrl = new URL(microsoftTokenUrl(data.microsoft_tenant_id));
    const req = https.request({
      hostname: tokenUrl.hostname,
      path: tokenUrl.pathname,
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
          data.microsoft_access_token = tokens.access_token;
          if (tokens.refresh_token) data.microsoft_refresh_token = tokens.refresh_token;
          data.microsoft_token_expiry = Math.floor(Date.now() / 1000) + (tokens.expires_in || 3600);
          saveConnections(data);
          resolve({ success: true });
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Microsoft token refresh timed out')); });
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

/**
 * Set Dynamics 365 environment URL and Azure AD app credentials
 */
function setD365Config(url, tenantId, clientId, clientSecret) {
  const data = loadConnections();
  data.d365_url = url.replace(/\/+$/, '');
  data.d365_tenant_id = tenantId;
  data.d365_client_id = clientId;
  data.d365_client_secret = clientSecret;
  saveConnections(data);
  return true;
}

/**
 * Set D365 delegated OAuth credentials (for the Authorize flow)
 */
function setD365Credentials(url, tenantId, clientId, clientSecret) {
  const data = loadConnections();
  data.d365_url = url.replace(/\/+$/, '');
  data.d365_tenant_id = tenantId;
  data.d365_client_id = clientId;
  data.d365_client_secret = clientSecret;
  // Clear any existing client-credentials-only token so delegated flow takes over
  delete data.d365_access_token;
  delete data.d365_token_expiry;
  saveConnections(data);
  return true;
}

/**
 * Get D365 OAuth authorization URL (delegated flow)
 */
function getD365AuthUrl(redirectPort) {
  const data = loadConnections();
  if (!data.d365_client_id || !data.d365_tenant_id || !data.d365_url) return null;
  const redirectUri = `http://localhost:${redirectPort}/oauth/dynamics365/callback`;
  const scope = `${data.d365_url}/.default offline_access`;
  const params = new URLSearchParams({
    client_id: data.d365_client_id,
    redirect_uri: redirectUri,
    response_type: 'code',
    scope,
    response_mode: 'query',
  });
  return `${microsoftAuthUrl(data.d365_tenant_id)}?${params.toString()}`;
}

/**
 * Exchange D365 authorization code for tokens (delegated flow)
 */
function exchangeD365Code(code, redirectPort) {
  return new Promise((resolve, reject) => {
    const data = loadConnections();
    if (!data.d365_client_id || !data.d365_client_secret || !data.d365_tenant_id || !data.d365_url) {
      return reject(new Error('D365 credentials not configured'));
    }
    const redirectUri = `http://localhost:${redirectPort}/oauth/dynamics365/callback`;
    const scope = `${data.d365_url}/.default offline_access`;
    const postData = new URLSearchParams({
      code,
      client_id: data.d365_client_id,
      client_secret: data.d365_client_secret,
      redirect_uri: redirectUri,
      grant_type: 'authorization_code',
      scope,
    }).toString();

    const tokenUrl = new URL(microsoftTokenUrl(data.d365_tenant_id));
    const req = https.request({
      hostname: tokenUrl.hostname,
      path: tokenUrl.pathname,
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
          const connData = loadConnections();
          connData.d365_access_token = tokens.access_token;
          if (tokens.refresh_token) connData.d365_refresh_token = tokens.refresh_token;
          connData.d365_token_expiry = Math.floor(Date.now() / 1000) + (tokens.expires_in || 3600);
          saveConnections(connData);
          resolve({ success: true });
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('D365 OAuth exchange timed out')); });
    req.write(postData);
    req.end();
  });
}

/**
 * Refresh D365 access token using refresh token (delegated flow)
 */
function refreshD365Token() {
  return new Promise((resolve, reject) => {
    const data = loadConnections();
    if (!data.d365_refresh_token || !data.d365_client_id || !data.d365_client_secret || !data.d365_tenant_id || !data.d365_url) {
      return reject(new Error('No D365 refresh token available'));
    }
    const scope = `${data.d365_url}/.default offline_access`;
    const postData = new URLSearchParams({
      refresh_token: data.d365_refresh_token,
      client_id: data.d365_client_id,
      client_secret: data.d365_client_secret,
      grant_type: 'refresh_token',
      scope,
    }).toString();

    const tokenUrl = new URL(microsoftTokenUrl(data.d365_tenant_id));
    const req = https.request({
      hostname: tokenUrl.hostname,
      path: tokenUrl.pathname,
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
          data.d365_access_token = tokens.access_token;
          if (tokens.refresh_token) data.d365_refresh_token = tokens.refresh_token;
          data.d365_token_expiry = Math.floor(Date.now() / 1000) + (tokens.expires_in || 3600);
          saveConnections(data);
          resolve({ success: true });
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('D365 token refresh timed out')); });
    req.write(postData);
    req.end();
  });
}

/**
 * Set Zoom Server-to-Server OAuth credentials
 */
function setZoomConfig(accountId, clientId, clientSecret) {
  const data = loadConnections();
  data.zoom_account_id = accountId;
  data.zoom_client_id = clientId;
  data.zoom_client_secret = clientSecret;
  saveConnections(data);
  return true;
}

// ── Model Provider (Ollama / OpenAI-compatible / Azure) ──────────────────

/**
 * Normalize an Azure URL to the chat/completions endpoint.
 * Azure AI Foundry shows users a "Target URI" like:
 *   https://{resource}.cognitiveservices.azure.com/openai/responses?api-version=...
 * But the Chat Completions API needs:
 *   https://{resource}.cognitiveservices.azure.com/openai/deployments/{model}/chat/completions
 * If the URL contains /openai/responses and a model/deployment name is available,
 * auto-convert it. Also strips any existing ?api-version= query params (we add our own).
 */
function normalizeAzureUrl(baseUrl, model) {
  if (!baseUrl) return baseUrl;
  // Strip trailing slashes and query params
  let url = baseUrl.replace(/\/+$/, '').replace(/\?.*$/, '');
  // If URL ends with /openai/responses or /openai, rebuild with deployment path
  if (model && /\/openai(\/responses)?$/.test(url)) {
    url = url.replace(/\/openai(\/responses)?$/, `/openai/deployments/${model}/chat/completions`);
  }
  return url;
}

/**
 * Read raw provider storage. Handles migration from old single-config format
 * to new multi-config format: { active: 'azure', openai: {...}, azure: {...}, anthropic: {...} }
 */
function _readProviderStore() {
  const data = loadConnections();
  const p = data._provider;
  if (!p) return { active: 'ollama' };
  // Migrate old format: { type: 'azure', base_url: '...' } → new multi-config
  if (p.type && !p.active) {
    const type = p.type;
    const cfg = { ...p };
    delete cfg.type;
    return { active: type, [type]: cfg };
  }
  return p;
}

/**
 * Get the active model provider config.
 * Returns { type: 'ollama' } when no remote provider is configured.
 */
function getProvider() {
  const store = _readProviderStore();
  const active = store.active || 'ollama';
  if (active === 'ollama') return { type: 'ollama' };
  const p = store[active];
  if (!p) return { type: 'ollama' };
  let baseUrl = (p.base_url || '').replace(/\/+$/, '');
  const model = p.model || '';
  if (active === 'azure') baseUrl = normalizeAzureUrl(baseUrl, model);
  const result = {
    type: active,
    base_url: baseUrl,
    api_key: p.api_key || '',
    model,
    max_tokens: parseInt(p.max_tokens, 10) || 4096,
  };
  if (active === 'azure') result.api_version = p.api_version || '2024-10-21';
  if (active === 'anthropic') result.api_version = p.api_version || '2023-06-01';
  return result;
}

/**
 * Get all saved provider configs (for the settings panel).
 * Returns { active, openai: {...}, azure: {...}, anthropic: {...} }
 * API keys are NOT masked here — callers must mask before sending to the client.
 */
function getAllProviders() {
  return _readProviderStore();
}

/**
 * Save a remote model provider config.
 * Stores the config under its type key and sets it as active.
 */
function setProvider(config) {
  if (!config || !config.base_url) {
    throw new Error('base_url is required');
  }
  const type = config.type || 'openai';
  if (type !== 'azure' && type !== 'anthropic' && !config.model) {
    throw new Error('model is required for non-Azure/Anthropic providers');
  }
  const data = loadConnections();
  const store = data._provider && data._provider.active ? { ...data._provider } : _readProviderStore();
  store.active = type;
  store[type] = {
    base_url: config.base_url.replace(/\/+$/, ''),
    api_key: config.api_key || '',
    model: config.model || '',
    max_tokens: parseInt(config.max_tokens, 10) || 4096,
    ...(type === 'azure' ? { api_version: config.api_version || '2024-10-21' } : {}),
    ...(type === 'anthropic' ? { api_version: config.api_version || '2023-06-01' } : {}),
  };
  data._provider = store;
  saveConnections(data);
  return true;
}

/**
 * Remove the active provider, reverting to local Ollama.
 * Saved configs for each type are preserved.
 */
function removeProvider() {
  const data = loadConnections();
  const store = data._provider && data._provider.active ? { ...data._provider } : _readProviderStore();
  store.active = 'ollama';
  data._provider = store;
  saveConnections(data);
  return true;
}

/**
 * Test connectivity to an OpenAI-compatible or Azure endpoint.
 * Sends a minimal non-streaming request and checks for a valid response.
 * @param {{ type?: string, base_url: string, api_key: string, model?: string, api_version?: string }} config
 * @returns {Promise<{ success: boolean, model: string, message?: string }>}
 */
function testProvider(config) {
  return new Promise((resolve, reject) => {
    let baseUrl = (config.base_url || '').replace(/\/+$/, '');
    if (!baseUrl) return reject(new Error('base_url is required'));

    const isAzure = config.type === 'azure';
    const isAnthropic = config.type === 'anthropic';

    // Azure: normalize Foundry-provided URLs
    if (isAzure) baseUrl = normalizeAzureUrl(baseUrl, config.model);

    // Anthropic uses a different request format and auth
    if (isAnthropic) {
      const url = new URL(baseUrl);
      const useHttps = url.protocol === 'https:';
      const lib = useHttps ? https : http;
      const modelName = config.model || 'claude-sonnet-4-20250514';
      const body = JSON.stringify({
        model: modelName,
        messages: [{ role: 'user', content: 'Hi' }],
        max_tokens: 5,
      });
      const req = lib.request({
        hostname: url.hostname,
        port: url.port || (useHttps ? 443 : 80),
        path: url.pathname + url.search,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
          'x-api-key': config.api_key || '',
          'anthropic-version': config.api_version || '2023-06-01',
        },
      }, (res) => {
        let data = '';
        res.on('data', (chunk) => data += chunk);
        res.on('end', () => {
          try {
            const parsed = JSON.parse(data);
            if (parsed.error) {
              return resolve({ success: false, model: modelName, message: parsed.error.message || parsed.error.type || 'Unknown error' });
            }
            if (parsed.content && parsed.content.length > 0) {
              return resolve({ success: true, model: parsed.model || modelName });
            }
            resolve({ success: false, model: modelName, message: 'Unexpected response format' });
          } catch (err) {
            resolve({ success: false, model: modelName, message: `Parse error: ${err.message}` });
          }
        });
      });
      req.on('error', (err) => resolve({ success: false, model: modelName, message: err.message }));
      req.setTimeout(15000, () => { req.destroy(); resolve({ success: false, model: modelName, message: 'Connection timed out' }); });
      req.write(body);
      req.end();
      return;
    }

    let endpoint;
    if (isAzure) {
      const sep = baseUrl.includes('?') ? '&' : '?';
      endpoint = baseUrl + sep + 'api-version=' + (config.api_version || '2024-10-21');
    } else {
      endpoint = baseUrl + '/chat/completions';
    }
    const url = new URL(endpoint);
    const useHttps = url.protocol === 'https:';
    const lib = useHttps ? https : http;

    const modelName = config.model || (isAzure ? 'azure-deployment' : 'gpt-4o');
    const tokenParam = isAzure ? { max_completion_tokens: 5 } : { max_tokens: 5 };
    const body = JSON.stringify({
      ...(isAzure ? {} : { model: modelName }),
      messages: [{ role: 'user', content: 'Hi' }],
      ...tokenParam,
      stream: false,
    });

    const authHeaders = isAzure
      ? (config.api_key ? { 'api-key': config.api_key } : {})
      : (config.api_key ? { 'Authorization': `Bearer ${config.api_key}` } : {});

    const req = lib.request({
      hostname: url.hostname,
      port: url.port || (useHttps ? 443 : 80),
      path: url.pathname + url.search,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
        ...authHeaders,
      },
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (parsed.error) {
            return resolve({ success: false, model: modelName, message: parsed.error.message || parsed.error.type || 'Unknown error' });
          }
          if (parsed.choices && parsed.choices.length > 0) {
            return resolve({ success: true, model: parsed.model || modelName });
          }
          resolve({ success: false, model: modelName, message: 'Unexpected response format' });
        } catch (err) {
          resolve({ success: false, model: modelName, message: `Parse error: ${err.message}` });
        }
      });
    });

    req.on('error', (err) => resolve({ success: false, model: modelName, message: err.message }));
    req.setTimeout(15000, () => { req.destroy(); resolve({ success: false, model: modelName, message: 'Connection timed out' }); });
    req.write(body);
    req.end();
  });
}

module.exports = {
  listConnections,
  setApiKey,
  removeConnection,
  setGoogleCredentials,
  getGoogleAuthUrl,
  exchangeGoogleCode,
  refreshGoogleToken,
  setMicrosoftCredentials,
  getMicrosoftAuthUrl,
  exchangeMicrosoftCode,
  refreshMicrosoftToken,
  setTelegramChatId,
  testTelegramToken,
  setJiraConfig,
  setConfluenceConfig,
  setD365Config,
  setD365Credentials,
  getD365AuthUrl,
  exchangeD365Code,
  refreshD365Token,
  setZoomConfig,
  getProvider,
  getAllProviders,
  setProvider,
  removeProvider,
  testProvider,
};
