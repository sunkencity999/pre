// PRE Web GUI — SharePoint tool
// Uses Microsoft Graph API with OAuth 2.0 (Azure AD / Entra ID)

const https = require('https');
const fs = require('fs');
const path = require('path');
const { CONNECTIONS_FILE } = require('../constants');

const GRAPH_HOST = 'graph.microsoft.com';
const TOKEN_HOST = 'login.microsoftonline.com';

// Encode folder path segments individually (preserves / separators)
function encodePath(p) {
  return p.split('/').map(s => encodeURIComponent(s)).join('/');
}

// ── Token management ──

function getConfig() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    if (!data.microsoft_client_id || !data.microsoft_refresh_token) return null;
    return {
      tenantId: data.microsoft_tenant_id || 'common',
      clientId: data.microsoft_client_id,
      clientSecret: data.microsoft_client_secret,
      accessToken: data.microsoft_access_token,
      refreshToken: data.microsoft_refresh_token,
      tokenExpiry: data.microsoft_token_expiry || 0,
    };
  } catch {
    return null;
  }
}

function saveTokens(accessToken, refreshToken, expiresIn) {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    data.microsoft_access_token = accessToken;
    if (refreshToken) data.microsoft_refresh_token = refreshToken;
    data.microsoft_token_expiry = Math.floor(Date.now() / 1000) + (expiresIn || 3600);
    fs.writeFileSync(CONNECTIONS_FILE, JSON.stringify(data, null, 2), { mode: 0o600 });
  } catch {}
}

function refreshAccessToken(config) {
  return new Promise((resolve, reject) => {
    const postData = new URLSearchParams({
      client_id: config.clientId,
      client_secret: config.clientSecret,
      refresh_token: config.refreshToken,
      grant_type: 'refresh_token',
      scope: 'https://graph.microsoft.com/.default offline_access',
    }).toString();

    const req = https.request({
      hostname: TOKEN_HOST,
      path: `/${config.tenantId}/oauth2/v2.0/token`,
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
          saveTokens(tokens.access_token, tokens.refresh_token, tokens.expires_in);
          resolve(tokens.access_token);
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('SharePoint token refresh timed out')); });
    req.write(postData);
    req.end();
  });
}

async function getAccessToken() {
  const config = getConfig();
  if (!config) throw new Error('Microsoft/SharePoint not configured. Set up via Settings > SharePoint.');

  // Refresh if expired or expiring within 5 minutes
  const now = Math.floor(Date.now() / 1000);
  if (!config.accessToken || now >= (config.tokenExpiry - 300)) {
    return refreshAccessToken(config);
  }
  return config.accessToken;
}

// ── Graph API client ──

function graphRequest(method, apiPath, body, rawResponse) {
  return new Promise(async (resolve, reject) => {
    let token;
    try {
      token = await getAccessToken();
    } catch (err) {
      return reject(new Error(`Auth failed: ${err.message}`));
    }

    const options = {
      hostname: GRAPH_HOST,
      path: encodeURI(`/v1.0${apiPath}`),
      method,
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/json',
        'User-Agent': 'PRE-Web-GUI',
      },
    };

    if (body) {
      const bodyStr = JSON.stringify(body);
      options.headers['Content-Type'] = 'application/json';
      options.headers['Content-Length'] = Buffer.byteLength(bodyStr);
    }

    const req = https.request(options, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const buf = Buffer.concat(chunks);
        if (res.statusCode >= 400) {
          let msg = `Graph API ${res.statusCode}`;
          try { const e = JSON.parse(buf.toString()); msg += ': ' + (e.error?.message || JSON.stringify(e.error)); } catch {}
          return reject(new Error(msg));
        }
        if (rawResponse) return resolve(buf);
        try {
          resolve(buf.length > 0 ? JSON.parse(buf.toString()) : {});
        } catch {
          resolve(buf.toString());
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Graph API request timed out')); });
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

function graphUpload(apiPath, content) {
  return new Promise(async (resolve, reject) => {
    let token;
    try {
      token = await getAccessToken();
    } catch (err) {
      return reject(new Error(`Auth failed: ${err.message}`));
    }

    const buf = Buffer.from(content, 'utf-8');
    const options = {
      hostname: GRAPH_HOST,
      path: encodeURI(`/v1.0${apiPath}`),
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/octet-stream',
        'Content-Length': buf.length,
      },
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          let msg = `Upload failed (${res.statusCode})`;
          try { msg += ': ' + JSON.parse(data).error?.message; } catch {}
          return reject(new Error(msg));
        }
        try { resolve(JSON.parse(data)); } catch { resolve(data); }
      });
    });
    req.on('error', reject);
    req.setTimeout(60000, () => { req.destroy(); reject(new Error('Upload timed out')); });
    req.write(buf);
    req.end();
  });
}

// ── Tool handlers ──

async function searchSharePoint(query, entityTypes, maxResults) {
  const types = entityTypes || ['driveItem', 'site', 'listItem'];
  const result = await graphRequest('POST', '/search/query', {
    requests: [{
      entityTypes: types,
      query: { queryString: query },
      from: 0,
      size: maxResults || 10,
    }],
  });

  const hits = [];
  for (const response of (result.value || [])) {
    for (const hitContainer of (response.hitsContainers || [])) {
      for (const hit of (hitContainer.hits || [])) {
        const r = hit.resource || {};
        hits.push({
          type: hit.resource?.['@odata.type']?.replace('#microsoft.graph.', '') || 'unknown',
          name: r.name || r.displayName || 'Untitled',
          webUrl: r.webUrl || '',
          summary: hit.summary || '',
          lastModified: r.lastModifiedDateTime || '',
          id: r.id || '',
          siteId: r.parentReference?.siteId || '',
        });
      }
    }
  }
  return hits;
}

async function listSites(query) {
  const searchTerm = query || '*';
  const result = await graphRequest('GET', `/sites?search=${encodeURIComponent(searchTerm)}&$top=25`);
  return (result.value || []).map(s => ({
    id: s.id,
    name: s.displayName,
    webUrl: s.webUrl,
    description: s.description || '',
  }));
}

async function listFiles(siteId, driveId, folderPath, top) {
  let apiPath;
  if (driveId && folderPath) {
    apiPath = `/sites/${siteId}/drives/${driveId}/root:/${encodePath(folderPath)}:/children`;
  } else if (driveId) {
    apiPath = `/sites/${siteId}/drives/${driveId}/root/children`;
  } else if (folderPath) {
    apiPath = `/sites/${siteId}/drive/root:/${encodePath(folderPath)}:/children`;
  } else {
    apiPath = `/sites/${siteId}/drive/root/children`;
  }
  apiPath += `?$top=${top || 50}&$orderby=lastModifiedDateTime desc`;

  const result = await graphRequest('GET', apiPath);
  return (result.value || []).map(item => ({
    id: item.id,
    name: item.name,
    type: item.folder ? 'folder' : 'file',
    size: item.size || 0,
    webUrl: item.webUrl || '',
    lastModified: item.lastModifiedDateTime || '',
    mimeType: item.file?.mimeType || '',
    driveId: item.parentReference?.driveId || '',
  }));
}

async function readFile(siteId, itemId, driveId) {
  // First get file metadata to check type and size
  let metaPath;
  if (driveId) {
    metaPath = `/sites/${siteId}/drives/${driveId}/items/${itemId}`;
  } else {
    metaPath = `/sites/${siteId}/drive/items/${itemId}`;
  }
  const meta = await graphRequest('GET', metaPath);

  // Reject very large files
  if (meta.size > 5 * 1024 * 1024) {
    return `File "${meta.name}" is ${(meta.size / 1024 / 1024).toFixed(1)} MB — too large to read inline. Download via webUrl: ${meta.webUrl}`;
  }

  // Download content
  const contentPath = `${metaPath}/content`;
  const buf = await graphRequest('GET', contentPath, null, true);

  // For text-based files, return as string
  const textTypes = ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.css', '.js', '.ts', '.py', '.sh', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.log', '.sql'];
  const ext = path.extname(meta.name || '').toLowerCase();
  const isText = textTypes.includes(ext) || (meta.file?.mimeType || '').startsWith('text/');

  if (isText) {
    return `File: ${meta.name} (${(meta.size / 1024).toFixed(1)} KB)\n\n${buf.toString('utf-8')}`;
  }
  // For Office documents, return metadata with download link
  return `File: ${meta.name} (${(meta.size / 1024).toFixed(1)} KB, ${meta.file?.mimeType || 'binary'})\nThis is a binary file. Download URL: ${meta['@microsoft.graph.downloadUrl'] || meta.webUrl}`;
}

async function uploadFile(siteId, folderPath, fileName, content, driveId) {
  const fullPath = folderPath ? `${folderPath}/${fileName}` : fileName;
  let apiPath;
  if (driveId) {
    apiPath = `/sites/${siteId}/drives/${driveId}/root:/${encodePath(fullPath)}:/content`;
  } else {
    apiPath = `/sites/${siteId}/drive/root:/${encodePath(fullPath)}:/content`;
  }
  const result = await graphUpload(apiPath, content);
  return {
    id: result.id,
    name: result.name,
    webUrl: result.webUrl,
    size: result.size,
  };
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return 'unknown';
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${units[i]}`;
}

async function listDrives(siteId) {
  const result = await graphRequest('GET', `/sites/${siteId}/drives?$select=id,name,driveType,webUrl,quota`);
  return (result.value || []).map(d => ({
    id: d.id,
    name: d.name,
    driveType: d.driveType,
    webUrl: d.webUrl,
    quota: d.quota ? {
      used: d.quota.used,
      usedHuman: formatBytes(d.quota.used),
      total: d.quota.total,
      totalHuman: formatBytes(d.quota.total),
      remaining: d.quota.remaining,
      remainingHuman: formatBytes(d.quota.remaining),
      deleted: d.quota.deleted,
      deletedHuman: formatBytes(d.quota.deleted),
      state: d.quota.state || 'unknown',
    } : null,
  }));
}

async function getSiteUsage(siteId) {
  // Get site info
  const site = await graphRequest('GET', `/sites/${siteId}?$select=id,displayName,webUrl`);
  // Get all drives with quota
  const drives = await listDrives(siteId);
  const totalUsed = drives.reduce((sum, d) => sum + (d.quota?.used || 0), 0);
  const totalQuota = drives.reduce((sum, d) => sum + (d.quota?.total || 0), 0);
  return {
    site: site.displayName || siteId,
    webUrl: site.webUrl,
    driveCount: drives.length,
    totalUsed,
    totalUsedHuman: formatBytes(totalUsed),
    totalQuota,
    totalQuotaHuman: formatBytes(totalQuota),
    percentUsed: totalQuota > 0 ? `${(totalUsed / totalQuota * 100).toFixed(1)}%` : 'unknown',
    drives: drives.map(d => ({
      name: d.name,
      driveType: d.driveType,
      quota: d.quota,
    })),
  };
}

async function listItems(siteId, listId, top, filter) {
  let apiPath = `/sites/${siteId}/lists/${listId}/items?$expand=fields&$top=${top || 50}`;
  if (filter) apiPath += `&$filter=${encodeURIComponent(filter)}`;

  const result = await graphRequest('GET', apiPath);
  return (result.value || []).map(item => ({
    id: item.id,
    webUrl: item.webUrl,
    fields: item.fields || {},
    lastModified: item.lastModifiedDateTime || '',
  }));
}

async function listLists(siteId) {
  const result = await graphRequest('GET', `/sites/${siteId}/lists?$top=50`);
  return (result.value || []).map(l => ({
    id: l.id,
    name: l.displayName,
    description: l.description || '',
    webUrl: l.webUrl,
    template: l.list?.template || '',
    itemCount: l.list?.contentTypesEnabled ? undefined : undefined,
  }));
}

async function getRecent(siteId, driveId, top) {
  let apiPath;
  if (driveId) {
    apiPath = `/sites/${siteId}/drives/${driveId}/recent`;
  } else {
    apiPath = `/sites/${siteId}/drive/recent`;
  }
  const result = await graphRequest('GET', apiPath);
  const items = (result.value || []).slice(0, top || 25);
  return items.map(item => ({
    id: item.id,
    name: item.name,
    type: item.folder ? 'folder' : 'file',
    size: item.size || 0,
    webUrl: item.webUrl || '',
    lastModified: item.lastModifiedDateTime || '',
    lastModifiedBy: item.lastModifiedBy?.user?.displayName || '',
    mimeType: item.file?.mimeType || '',
    driveId: item.parentReference?.driveId || '',
    folderPath: item.parentReference?.path?.replace(/^.*root:/, '') || '/',
  }));
}

async function getColumns(siteId, listId) {
  const result = await graphRequest('GET', `/sites/${siteId}/lists/${listId}/columns?$top=100`);
  return (result.value || []).filter(c => !c.readOnly || c.name === 'Title').map(c => ({
    name: c.name,
    displayName: c.displayName,
    type: c.text ? 'text' : c.number ? 'number' : c.dateTime ? 'dateTime' : c.boolean ? 'boolean' : c.choice ? 'choice' : c.lookup ? 'lookup' : c.personOrGroup ? 'personOrGroup' : c.currency ? 'currency' : c.calculated ? 'calculated' : 'other',
    description: c.description || '',
    required: c.required || false,
    choices: c.choice?.choices || undefined,
    readOnly: c.readOnly || false,
  }));
}

async function createListItem(siteId, listId, fields) {
  const result = await graphRequest('POST', `/sites/${siteId}/lists/${listId}/items`, { fields });
  return {
    id: result.id,
    webUrl: result.webUrl,
    fields: result.fields || {},
  };
}

async function updateListItem(siteId, listId, itemId, fields) {
  const result = await graphRequest('PATCH', `/sites/${siteId}/lists/${listId}/items/${itemId}/fields`, fields);
  return { id: itemId, fields: result || {} };
}

async function createFolder(siteId, folderName, parentPath, driveId) {
  let apiPath;
  if (driveId && parentPath) {
    apiPath = `/sites/${siteId}/drives/${driveId}/root:/${encodePath(parentPath)}:/children`;
  } else if (driveId) {
    apiPath = `/sites/${siteId}/drives/${driveId}/root/children`;
  } else if (parentPath) {
    apiPath = `/sites/${siteId}/drive/root:/${encodePath(parentPath)}:/children`;
  } else {
    apiPath = `/sites/${siteId}/drive/root/children`;
  }
  const result = await graphRequest('POST', apiPath, {
    name: folderName,
    folder: {},
    '@microsoft.graph.conflictBehavior': 'fail',
  });
  return {
    id: result.id,
    name: result.name,
    webUrl: result.webUrl,
  };
}

async function getFileMetadata(siteId, itemId, driveId) {
  let apiPath;
  if (driveId) {
    apiPath = `/sites/${siteId}/drives/${driveId}/items/${itemId}?$select=id,name,size,webUrl,file,folder,createdDateTime,lastModifiedDateTime,createdBy,lastModifiedBy,parentReference`;
  } else {
    apiPath = `/sites/${siteId}/drive/items/${itemId}?$select=id,name,size,webUrl,file,folder,createdDateTime,lastModifiedDateTime,createdBy,lastModifiedBy,parentReference`;
  }
  const item = await graphRequest('GET', apiPath);
  return {
    id: item.id,
    name: item.name,
    type: item.folder ? 'folder' : 'file',
    size: item.size || 0,
    sizeHuman: formatBytes(item.size || 0),
    webUrl: item.webUrl || '',
    mimeType: item.file?.mimeType || '',
    created: item.createdDateTime || '',
    createdBy: item.createdBy?.user?.displayName || '',
    lastModified: item.lastModifiedDateTime || '',
    lastModifiedBy: item.lastModifiedBy?.user?.displayName || '',
    driveId: item.parentReference?.driveId || '',
    folderPath: item.parentReference?.path?.replace(/^.*root:/, '') || '/',
    childCount: item.folder?.childCount,
    downloadUrl: item['@microsoft.graph.downloadUrl'] || null,
  };
}

async function moveFile(siteId, itemId, destFolderPath, newName, driveId, destDriveId) {
  let apiPath;
  if (driveId) {
    apiPath = `/sites/${siteId}/drives/${driveId}/items/${itemId}`;
  } else {
    apiPath = `/sites/${siteId}/drive/items/${itemId}`;
  }
  const body = {};
  if (destFolderPath) {
    body.parentReference = {
      driveId: destDriveId || driveId,
      path: `/root:/${encodePath(destFolderPath)}`,
    };
  }
  if (newName) body.name = newName;
  const result = await graphRequest('PATCH', apiPath, body);
  return {
    id: result.id,
    name: result.name,
    webUrl: result.webUrl,
    parentPath: result.parentReference?.path?.replace(/^.*root:/, '') || '/',
  };
}

async function copyFile(siteId, itemId, destFolderPath, newName, driveId, destDriveId) {
  let apiPath;
  if (driveId) {
    apiPath = `/sites/${siteId}/drives/${driveId}/items/${itemId}/copy`;
  } else {
    apiPath = `/sites/${siteId}/drive/items/${itemId}/copy`;
  }
  const body = {};
  if (destFolderPath) {
    body.parentReference = {
      driveId: destDriveId || driveId,
      path: `/root:/${encodePath(destFolderPath)}`,
    };
  }
  if (newName) body.name = newName;
  // Copy returns 202 Accepted with a Location header for async status — Graph API returns empty body
  await graphRequest('POST', apiPath, body);
  return { success: true, name: newName || '(same name)', destFolder: destFolderPath || '/' };
}

async function deleteFile(siteId, itemId, driveId) {
  let apiPath;
  if (driveId) {
    apiPath = `/sites/${siteId}/drives/${driveId}/items/${itemId}`;
  } else {
    apiPath = `/sites/${siteId}/drive/items/${itemId}`;
  }
  await graphRequest('DELETE', apiPath);
  return { deleted: true };
}

async function listSubsites(siteId) {
  const result = await graphRequest('GET', `/sites/${siteId}/sites?$top=50`);
  return (result.value || []).map(s => ({
    id: s.id,
    name: s.displayName,
    webUrl: s.webUrl,
    description: s.description || '',
  }));
}

async function getPage(siteId, pageId) {
  // Pages API uses beta endpoint
  const result = await new Promise(async (resolve, reject) => {
    let token;
    try { token = await getAccessToken(); } catch (err) { return reject(err); }

    const req = https.request({
      hostname: GRAPH_HOST,
      path: `/v1.0/sites/${siteId}/pages/${pageId}`,
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/json',
      },
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          return reject(new Error(`Page API ${res.statusCode}: ${data.slice(0, 200)}`));
        }
        try { resolve(JSON.parse(data)); } catch { resolve(data); }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('SharePoint page request timed out')); });
    req.end();
  });

  return {
    id: result.id,
    title: result.title || result.name || 'Untitled',
    webUrl: result.webUrl,
    description: result.description || '',
    lastModified: result.lastModifiedDateTime || '',
    content: result.content || result.body?.content || '(No content available — use webUrl to view in browser)',
  };
}

// ── Main tool dispatcher ──

async function sharepoint(args) {
  const config = getConfig();
  if (!config) return 'SharePoint not configured. Set up Microsoft connection in Settings.';

  const action = (args.action || '').toLowerCase();

  try {
    switch (action) {
      case 'search': {
        if (!args.query) return 'Error: "query" is required for search.';
        const types = args.entity_types ? args.entity_types.split(',').map(s => s.trim()) : undefined;
        const hits = await searchSharePoint(args.query, types, args.max_results);
        if (hits.length === 0) return `No results found for "${args.query}".`;
        let out = `Found ${hits.length} result(s) for "${args.query}":\n\n`;
        for (const h of hits) {
          out += `[${h.type}] ${h.name}\n`;
          if (h.webUrl) out += `  URL: ${h.webUrl}\n`;
          if (h.summary) out += `  Summary: ${h.summary}\n`;
          if (h.lastModified) out += `  Modified: ${h.lastModified}\n`;
          if (h.siteId) out += `  Site ID: ${h.siteId}\n`;
          out += `  ID: ${h.id}\n\n`;
        }
        return out.trim();
      }

      case 'list_sites': {
        const sites = await listSites(args.query);
        if (sites.length === 0) return 'No SharePoint sites found.';
        let out = `${sites.length} site(s):\n\n`;
        for (const s of sites) {
          out += `${s.name}\n  ID: ${s.id}\n  URL: ${s.webUrl}\n`;
          if (s.description) out += `  Description: ${s.description}\n`;
          out += '\n';
        }
        return out.trim();
      }

      case 'list_drives': {
        if (!args.site_id) return 'Error: "site_id" is required for list_drives.';
        const drives = await listDrives(args.site_id);
        if (drives.length === 0) return 'No document libraries found.';
        let out = `${drives.length} document library/libraries:\n\n`;
        for (const d of drives) {
          out += `${d.name} (${d.driveType})\n  ID: ${d.id}\n  URL: ${d.webUrl}`;
          if (d.quota) {
            out += `\n  Storage: ${d.quota.usedHuman} used of ${d.quota.totalHuman} (${d.quota.state})`;
            if (d.quota.remaining != null) out += ` — ${d.quota.remainingHuman} remaining`;
          }
          out += '\n\n';
        }
        return out.trim();
      }

      case 'site_usage': {
        if (!args.site_id) return 'Error: "site_id" is required for site_usage.';
        const usage = await getSiteUsage(args.site_id);
        let out = `Site: ${usage.site}\nURL: ${usage.webUrl}\n\n`;
        out += `Total storage used: ${usage.totalUsedHuman}\n`;
        out += `Total quota: ${usage.totalQuotaHuman}\n`;
        out += `Usage: ${usage.percentUsed}\n`;
        out += `Document libraries: ${usage.driveCount}\n\n`;
        for (const d of usage.drives) {
          out += `  ${d.name} (${d.driveType}): ${d.quota?.usedHuman || 'unknown'} used`;
          if (d.quota?.totalHuman) out += ` of ${d.quota.totalHuman}`;
          out += '\n';
        }
        return out.trim();
      }

      case 'list_files': {
        if (!args.site_id) return 'Error: "site_id" is required for list_files.';
        const files = await listFiles(args.site_id, args.drive_id, args.folder_path, args.max_results);
        if (files.length === 0) return 'No files found.';
        let out = `${files.length} item(s):\n\n`;
        for (const f of files) {
          const sizeStr = f.type === 'folder' ? 'folder' : `${(f.size / 1024).toFixed(1)} KB`;
          out += `${f.type === 'folder' ? '📁' : '📄'} ${f.name} (${sizeStr})\n`;
          out += `  ID: ${f.id}\n`;
          if (f.driveId) out += `  Drive ID: ${f.driveId}\n`;
          if (f.lastModified) out += `  Modified: ${f.lastModified}\n`;
          out += '\n';
        }
        return out.trim();
      }

      case 'read_file': {
        if (!args.site_id || !args.item_id) return 'Error: "site_id" and "item_id" are required for read_file.';
        return readFile(args.site_id, args.item_id, args.drive_id);
      }

      case 'upload_file': {
        if (!args.site_id || !args.file_name || !args.content) {
          return 'Error: "site_id", "file_name", and "content" are required for upload_file.';
        }
        const result = await uploadFile(args.site_id, args.folder_path, args.file_name, args.content, args.drive_id);
        return `Uploaded: ${result.name} (${(result.size / 1024).toFixed(1)} KB)\nURL: ${result.webUrl}`;
      }

      case 'list_lists': {
        if (!args.site_id) return 'Error: "site_id" is required for list_lists.';
        const lists = await listLists(args.site_id);
        if (lists.length === 0) return 'No lists found.';
        let out = `${lists.length} list(s):\n\n`;
        for (const l of lists) {
          out += `${l.name}\n  ID: ${l.id}\n  URL: ${l.webUrl}\n`;
          if (l.description) out += `  Description: ${l.description}\n`;
          out += '\n';
        }
        return out.trim();
      }

      case 'list_items': {
        if (!args.site_id || !args.list_id) return 'Error: "site_id" and "list_id" are required for list_items.';
        const items = await listItems(args.site_id, args.list_id, args.max_results, args.filter);
        if (items.length === 0) return 'No items found.';
        let out = `${items.length} item(s):\n\n`;
        for (const item of items) {
          out += `Item ${item.id}\n`;
          const fields = item.fields || {};
          for (const [k, v] of Object.entries(fields)) {
            if (k.startsWith('@odata') || k === 'id') continue;
            out += `  ${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}\n`;
          }
          out += '\n';
        }
        return out.trim();
      }

      case 'get_page': {
        if (!args.site_id || !args.page_id) return 'Error: "site_id" and "page_id" are required for get_page.';
        const page = await getPage(args.site_id, args.page_id);
        let out = `Page: ${page.title}\n`;
        if (page.webUrl) out += `URL: ${page.webUrl}\n`;
        if (page.lastModified) out += `Modified: ${page.lastModified}\n`;
        out += `\n${page.content}`;
        return out;
      }

      case 'get_recent': {
        if (!args.site_id) return 'Error: "site_id" is required for get_recent.';
        const recent = await getRecent(args.site_id, args.drive_id, args.count);
        if (recent.length === 0) return 'No recently modified files found.';
        let out = `${recent.length} recently modified item(s):\n\n`;
        for (const f of recent) {
          const sizeStr = f.type === 'folder' ? 'folder' : formatBytes(f.size);
          out += `${f.type === 'folder' ? '📁' : '📄'} ${f.name} (${sizeStr})\n`;
          out += `  ID: ${f.id}\n`;
          if (f.driveId) out += `  Drive ID: ${f.driveId}\n`;
          if (f.folderPath) out += `  Path: ${f.folderPath}\n`;
          if (f.lastModified) out += `  Modified: ${f.lastModified}`;
          if (f.lastModifiedBy) out += ` by ${f.lastModifiedBy}`;
          out += '\n\n';
        }
        return out.trim();
      }

      case 'get_columns': {
        if (!args.site_id || !args.list_id) return 'Error: "site_id" and "list_id" are required for get_columns.';
        const columns = await getColumns(args.site_id, args.list_id);
        if (columns.length === 0) return 'No writable columns found.';
        let out = `${columns.length} column(s):\n\n`;
        for (const c of columns) {
          out += `${c.displayName} (${c.name}) — ${c.type}`;
          if (c.required) out += ' [required]';
          if (c.readOnly) out += ' [read-only]';
          out += '\n';
          if (c.description) out += `  Description: ${c.description}\n`;
          if (c.choices) out += `  Choices: ${c.choices.join(', ')}\n`;
        }
        return out.trim();
      }

      case 'create_list_item': {
        if (!args.site_id || !args.list_id || !args.fields) {
          return 'Error: "site_id", "list_id", and "fields" (JSON object) are required for create_list_item.';
        }
        let fields = args.fields;
        if (typeof fields === 'string') {
          try { fields = JSON.parse(fields); } catch { return 'Error: "fields" must be valid JSON.'; }
        }
        const item = await createListItem(args.site_id, args.list_id, fields);
        let out = `Created item ${item.id}\nURL: ${item.webUrl}\nFields:\n`;
        for (const [k, v] of Object.entries(item.fields)) {
          if (k.startsWith('@odata') || k === 'id') continue;
          out += `  ${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}\n`;
        }
        return out.trim();
      }

      case 'update_list_item': {
        if (!args.site_id || !args.list_id || !args.item_id || !args.fields) {
          return 'Error: "site_id", "list_id", "item_id", and "fields" (JSON object) are required for update_list_item.';
        }
        let uFields = args.fields;
        if (typeof uFields === 'string') {
          try { uFields = JSON.parse(uFields); } catch { return 'Error: "fields" must be valid JSON.'; }
        }
        const updated = await updateListItem(args.site_id, args.list_id, args.item_id, uFields);
        let out = `Updated item ${updated.id}\nFields:\n`;
        for (const [k, v] of Object.entries(updated.fields)) {
          if (k.startsWith('@odata') || k === 'id') continue;
          out += `  ${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}\n`;
        }
        return out.trim();
      }

      case 'create_folder': {
        if (!args.site_id || !args.folder_name) {
          return 'Error: "site_id" and "folder_name" are required for create_folder.';
        }
        const folder = await createFolder(args.site_id, args.folder_name, args.dest_folder, args.drive_id);
        return `Created folder: ${folder.name}\nID: ${folder.id}\nURL: ${folder.webUrl}`;
      }

      case 'get_file_metadata': {
        if (!args.site_id || !args.item_id) return 'Error: "site_id" and "item_id" are required for get_file_metadata.';
        const meta = await getFileMetadata(args.site_id, args.item_id, args.drive_id);
        let out = `${meta.type === 'folder' ? '📁' : '📄'} ${meta.name}\n`;
        out += `  Type: ${meta.type}\n`;
        out += `  Size: ${meta.sizeHuman}\n`;
        if (meta.mimeType) out += `  MIME: ${meta.mimeType}\n`;
        out += `  Created: ${meta.created} by ${meta.createdBy}\n`;
        out += `  Modified: ${meta.lastModified} by ${meta.lastModifiedBy}\n`;
        out += `  Path: ${meta.folderPath}\n`;
        out += `  Drive ID: ${meta.driveId}\n`;
        out += `  URL: ${meta.webUrl}\n`;
        if (meta.childCount != null) out += `  Children: ${meta.childCount}\n`;
        if (meta.downloadUrl) out += `  Download: ${meta.downloadUrl}\n`;
        return out.trim();
      }

      case 'move_file': {
        if (!args.site_id || !args.item_id) return 'Error: "site_id" and "item_id" are required for move_file.';
        if (!args.dest_folder && !args.filename) return 'Error: "dest_folder" and/or "filename" (for rename) required.';
        const moved = await moveFile(args.site_id, args.item_id, args.dest_folder, args.filename, args.drive_id, args.dest_drive_id);
        return `Moved: ${moved.name}\nNew path: ${moved.parentPath}\nURL: ${moved.webUrl}`;
      }

      case 'copy_file': {
        if (!args.site_id || !args.item_id) return 'Error: "site_id" and "item_id" are required for copy_file.';
        if (!args.dest_folder) return 'Error: "dest_folder" is required for copy_file.';
        const copied = await copyFile(args.site_id, args.item_id, args.dest_folder, args.filename, args.drive_id, args.dest_drive_id);
        return `Copy started: ${copied.name} → ${copied.destFolder}\nNote: Large file copies are async and may take a moment to appear.`;
      }

      case 'delete_file': {
        if (!args.site_id || !args.item_id) return 'Error: "site_id" and "item_id" are required for delete_file.';
        await deleteFile(args.site_id, args.item_id, args.drive_id);
        return 'File deleted successfully.';
      }

      case 'list_subsites': {
        if (!args.site_id) return 'Error: "site_id" is required for list_subsites.';
        const subsites = await listSubsites(args.site_id);
        if (subsites.length === 0) return 'No subsites found.';
        let out = `${subsites.length} subsite(s):\n\n`;
        for (const s of subsites) {
          out += `${s.name}\n  ID: ${s.id}\n  URL: ${s.webUrl}\n`;
          if (s.description) out += `  Description: ${s.description}\n`;
          out += '\n';
        }
        return out.trim();
      }

      default:
        return 'Error: Unknown action. Use: search, list_sites, list_drives, list_files, read_file, upload_file, list_lists, list_items, get_page, site_usage, get_recent, get_columns, create_list_item, update_list_item, create_folder, get_file_metadata, move_file, copy_file, delete_file, list_subsites';
    }
  } catch (err) {
    return `SharePoint error: ${err.message}`;
  }
}

module.exports = { sharepoint, getConfig };
