// PRE Web GUI — Google tools (Gmail, GDrive, GDocs)
// Uses OAuth2 tokens from ~/.pre/connections.json

const https = require('https');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('../constants');

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
 * Ensure we have a valid access token, refreshing if needed
 */
async function ensureToken() {
  const data = loadConnections();
  if (!data.google_refresh_token || !data.google_client_id || !data.google_client_secret) {
    throw new Error('Google not configured. Use Settings to set up Google connection.');
  }
  // Check if token is expired or will expire within 60 seconds
  const now = Math.floor(Date.now() / 1000);
  if (data.google_access_token && data.google_token_expiry && data.google_token_expiry > now + 60) {
    return data.google_access_token;
  }
  // Refresh
  const tokens = await refreshToken(data);
  return tokens.access_token;
}

function refreshToken(data) {
  return new Promise((resolve, reject) => {
    const postData = new URLSearchParams({
      refresh_token: data.google_refresh_token,
      client_id: data.google_client_id,
      client_secret: data.google_client_secret,
      grant_type: 'refresh_token',
    }).toString();

    const req = https.request('https://oauth2.googleapis.com/token', {
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
          resolve(tokens);
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
 * Make an authenticated Google API request
 */
function googleRequest(method, url, token, body) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const options = {
      hostname: parsed.hostname,
      path: parsed.pathname + parsed.search,
      method,
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/json',
      },
    };
    if (body) {
      const bodyStr = typeof body === 'string' ? body : JSON.stringify(body);
      options.headers['Content-Type'] = 'application/json';
      options.headers['Content-Length'] = Buffer.byteLength(bodyStr);
    }

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          return reject(new Error(`Google API ${res.statusCode}: ${data.slice(0, 500)}`));
        }
        try {
          resolve(JSON.parse(data));
        } catch {
          resolve(data);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Google API request timed out')); });
    if (body) {
      req.write(typeof body === 'string' ? body : JSON.stringify(body));
    }
    req.end();
  });
}

// ── Gmail ──

async function gmail(args) {
  const action = args.action;
  if (!action) return 'Error: action required (search|read|send|draft|trash|labels|profile)';

  const token = await ensureToken();
  const base = 'https://gmail.googleapis.com/gmail/v1/users/me';

  switch (action) {
    case 'profile': {
      const data = await googleRequest('GET', `${base}/profile`, token);
      return `Email: ${data.emailAddress}\nMessages: ${data.messagesTotal}\nThreads: ${data.threadsTotal}`;
    }

    case 'labels': {
      const data = await googleRequest('GET', `${base}/labels`, token);
      const labels = (data.labels || []).map(l => `${l.name} (${l.type})`).join('\n');
      return `Gmail Labels:\n${labels}`;
    }

    case 'search': {
      const query = args.query || 'in:inbox';
      const maxResults = Math.min(args.count || 10, 20);
      const url = `${base}/messages?q=${encodeURIComponent(query)}&maxResults=${maxResults}`;
      const data = await googleRequest('GET', url, token);
      const messages = data.messages || [];
      if (messages.length === 0) return `No messages found for: ${query}`;

      // Fetch snippet for each message
      const results = [];
      for (const msg of messages.slice(0, maxResults)) {
        try {
          const detail = await googleRequest('GET', `${base}/messages/${msg.id}?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date`, token);
          const headers = detail.payload?.headers || [];
          const from = headers.find(h => h.name === 'From')?.value || '';
          const subject = headers.find(h => h.name === 'Subject')?.value || '(no subject)';
          const date = headers.find(h => h.name === 'Date')?.value || '';
          results.push(`ID: ${msg.id}\nFrom: ${from}\nSubject: ${subject}\nDate: ${date}\nSnippet: ${detail.snippet || ''}\n`);
        } catch {
          results.push(`ID: ${msg.id}\n`);
        }
      }
      return `Found ${data.resultSizeEstimate || messages.length} messages:\n\n${results.join('\n')}`;
    }

    case 'read': {
      if (!args.id) return 'Error: id required for read action';
      const data = await googleRequest('GET', `${base}/messages/${args.id}?format=full`, token);
      const headers = data.payload?.headers || [];
      const from = headers.find(h => h.name === 'From')?.value || '';
      const to = headers.find(h => h.name === 'To')?.value || '';
      const subject = headers.find(h => h.name === 'Subject')?.value || '';
      const date = headers.find(h => h.name === 'Date')?.value || '';

      // Extract body text
      let body = '';
      function extractText(part) {
        if (part.mimeType === 'text/plain' && part.body?.data) {
          body += Buffer.from(part.body.data, 'base64url').toString('utf-8');
        }
        if (part.parts) {
          for (const p of part.parts) extractText(p);
        }
      }
      if (data.payload) extractText(data.payload);
      if (!body && data.snippet) body = data.snippet;

      return `From: ${from}\nTo: ${to}\nSubject: ${subject}\nDate: ${date}\n\n${body}`;
    }

    case 'send': {
      if (!args.to) return 'Error: to required for send action';
      if (!args.subject && !args.body) return 'Error: subject or body required';
      const raw = buildRawEmail(args.to, args.subject || '', args.body || '', args.cc, args.bcc);
      const data = await googleRequest('POST', `${base}/messages/send`, token, { raw });
      return `Message sent! ID: ${data.id}`;
    }

    case 'draft': {
      if (!args.to) return 'Error: to required for draft action';
      const raw = buildRawEmail(args.to, args.subject || '', args.body || '', args.cc, args.bcc);
      const data = await googleRequest('POST', `${base}/drafts`, token, { message: { raw } });
      return `Draft created! ID: ${data.id}`;
    }

    case 'trash': {
      if (!args.id) return 'Error: id required for trash action';
      await googleRequest('POST', `${base}/messages/${args.id}/trash`, token, {});
      return `Message ${args.id} moved to trash.`;
    }

    default:
      return `Error: unknown gmail action '${action}'. Use: search, read, send, draft, trash, labels, profile`;
  }
}

function buildRawEmail(to, subject, body, cc, bcc) {
  let msg = `To: ${to}\r\n`;
  if (cc) msg += `Cc: ${cc}\r\n`;
  if (bcc) msg += `Bcc: ${bcc}\r\n`;
  msg += `Subject: ${subject}\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n${body}`;
  return Buffer.from(msg).toString('base64url');
}

// ── Google Drive ──

async function gdrive(args) {
  const action = args.action;
  if (!action) return 'Error: action required (list|search|download|mkdir|delete)';

  const token = await ensureToken();
  const base = 'https://www.googleapis.com/drive/v3';

  switch (action) {
    case 'list': {
      const query = args.query ? `&q=${encodeURIComponent(args.query)}` : '';
      const data = await googleRequest('GET', `${base}/files?pageSize=20&fields=files(id,name,mimeType,size,modifiedTime)${query}`, token);
      const files = data.files || [];
      if (files.length === 0) return 'No files found.';
      return files.map(f => `${f.name} (${f.mimeType}) ID: ${f.id} Modified: ${f.modifiedTime}`).join('\n');
    }

    case 'search': {
      if (!args.query) return 'Error: query required for search';
      const q = encodeURIComponent(`fullText contains '${args.query}'`);
      const data = await googleRequest('GET', `${base}/files?q=${q}&pageSize=20&fields=files(id,name,mimeType,size,modifiedTime)`, token);
      const files = data.files || [];
      if (files.length === 0) return `No files found for: ${args.query}`;
      return files.map(f => `${f.name} (${f.mimeType}) ID: ${f.id}`).join('\n');
    }

    case 'mkdir': {
      if (!args.name) return 'Error: name required for mkdir';
      const data = await googleRequest('POST', `${base}/files`, token, {
        name: args.name,
        mimeType: 'application/vnd.google-apps.folder',
      });
      return `Folder created: ${data.name} ID: ${data.id}`;
    }

    case 'delete': {
      if (!args.id) return 'Error: id required for delete';
      await googleRequest('DELETE', `${base}/files/${args.id}`, token);
      return `File ${args.id} deleted.`;
    }

    default:
      return `Error: unknown gdrive action '${action}'. Use: list, search, mkdir, delete`;
  }
}

// ── Google Docs ──

async function gdocs(args) {
  const action = args.action;
  if (!action) return 'Error: action required (create|read|append)';

  const token = await ensureToken();

  switch (action) {
    case 'create': {
      if (!args.title) return 'Error: title required for create';
      const data = await googleRequest('POST', 'https://docs.googleapis.com/v1/documents', token, {
        title: args.title,
      });
      return `Document created: "${data.title}" ID: ${data.documentId}`;
    }

    case 'read': {
      if (!args.id) return 'Error: id required for read';
      const data = await googleRequest('GET', `https://docs.googleapis.com/v1/documents/${args.id}`, token);
      // Extract text content
      let text = '';
      function extractContent(elements) {
        for (const el of elements || []) {
          if (el.paragraph) {
            for (const pe of el.paragraph.elements || []) {
              if (pe.textRun) text += pe.textRun.content;
            }
          }
          if (el.table) {
            for (const row of el.table.tableRows || []) {
              for (const cell of row.tableCells || []) {
                extractContent(cell.content);
              }
            }
          }
        }
      }
      extractContent(data.body?.content);
      return `Title: ${data.title}\n\n${text}`;
    }

    case 'append': {
      if (!args.id) return 'Error: id required for append';
      if (!args.content) return 'Error: content required for append';
      // Get current document length first
      const doc = await googleRequest('GET', `https://docs.googleapis.com/v1/documents/${args.id}`, token);
      const endIndex = (doc.body?.content || []).reduce((max, el) => Math.max(max, el.endIndex || 0), 1) - 1;
      await googleRequest('POST', `https://docs.googleapis.com/v1/documents/${args.id}:batchUpdate`, token, {
        requests: [{
          insertText: {
            location: { index: Math.max(1, endIndex) },
            text: args.content,
          },
        }],
      });
      return `Content appended to document ${args.id}.`;
    }

    default:
      return `Error: unknown gdocs action '${action}'. Use: create, read, append`;
  }
}

module.exports = { gmail, gdrive, gdocs };
