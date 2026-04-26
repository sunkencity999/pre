// PRE Web GUI — Dynamics 365 tool
// Uses Dataverse Web API with Azure AD OAuth 2.0 (client credentials)

const https = require('https');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('../constants');

const TOKEN_HOST = 'login.microsoftonline.com';

// ── Config & token management ──

function getConfig() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    if (!data.d365_url || !data.d365_client_id || !data.d365_tenant_id) return null;
    // Need either client_secret (for client_credentials) or refresh_token (for delegated)
    if (!data.d365_client_secret && !data.d365_refresh_token) return null;
    return {
      url: data.d365_url.replace(/\/+$/, ''),
      tenantId: data.d365_tenant_id,
      clientId: data.d365_client_id,
      clientSecret: data.d365_client_secret,
      refreshToken: data.d365_refresh_token,
      accessToken: data.d365_access_token,
      tokenExpiry: data.d365_token_expiry || 0,
    };
  } catch {
    return null;
  }
}

function saveTokens(accessToken, refreshToken, expiresIn) {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    data.d365_access_token = accessToken;
    if (refreshToken) data.d365_refresh_token = refreshToken;
    data.d365_token_expiry = Math.floor(Date.now() / 1000) + (expiresIn || 3600);
    fs.writeFileSync(CONNECTIONS_FILE, JSON.stringify(data, null, 2), { mode: 0o600 });
  } catch {}
}

/**
 * Fetch token via client credentials grant (app-only, no user context)
 */
function fetchClientCredentialsToken(config) {
  return new Promise((resolve, reject) => {
    const scope = `${config.url}/.default`;
    const postData = new URLSearchParams({
      client_id: config.clientId,
      client_secret: config.clientSecret,
      grant_type: 'client_credentials',
      scope,
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
          saveTokens(tokens.access_token, null, tokens.expires_in);
          resolve(tokens.access_token);
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('D365 auth request timed out')); });
    req.write(postData);
    req.end();
  });
}

/**
 * Refresh token via delegated flow (user context)
 */
function refreshDelegatedToken(config) {
  return new Promise((resolve, reject) => {
    const scope = `${config.url}/.default offline_access`;
    const postData = new URLSearchParams({
      refresh_token: config.refreshToken,
      client_id: config.clientId,
      client_secret: config.clientSecret,
      grant_type: 'refresh_token',
      scope,
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
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('D365 token refresh timed out')); });
    req.write(postData);
    req.end();
  });
}

async function getAccessToken() {
  const config = getConfig();
  if (!config) throw new Error('Dynamics 365 not configured. Use Settings to add your D365 environment URL and authorize access.');

  // If token is still fresh, reuse it
  const now = Math.floor(Date.now() / 1000);
  if (config.accessToken && now < (config.tokenExpiry - 300)) {
    return config.accessToken;
  }

  // Delegated flow (refresh token) takes priority over client credentials
  if (config.refreshToken) {
    return refreshDelegatedToken(config);
  }

  // Fall back to client credentials (app-only)
  if (config.clientSecret) {
    return fetchClientCredentialsToken(config);
  }

  throw new Error('D365 auth incomplete — either authorize via OAuth or provide a client secret.');
}

// ── Dataverse Web API client ──

function dataverseRequest(method, apiPath, body) {
  return new Promise(async (resolve, reject) => {
    const config = getConfig();
    if (!config) return reject(new Error('Dynamics 365 not configured.'));

    let token;
    try {
      token = await getAccessToken();
    } catch (err) {
      return reject(new Error(`Auth failed: ${err.message}`));
    }

    const baseUrl = new URL(config.url);
    const fullPath = `/api/data/v9.2${apiPath}`;

    const options = {
      hostname: baseUrl.hostname,
      port: 443,
      path: fullPath,
      method,
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/json',
        'OData-MaxVersion': '4.0',
        'OData-Version': '4.0',
        'User-Agent': 'PRE-Web-GUI',
        'Prefer': 'odata.include-annotations="*"',
      },
    };

    if (body) {
      const bodyStr = JSON.stringify(body);
      options.headers['Content-Type'] = 'application/json';
      options.headers['Content-Length'] = Buffer.byteLength(bodyStr);
    }

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        if (res.statusCode === 204) return resolve({});
        if (res.statusCode >= 400) {
          let msg = `Dataverse API ${res.statusCode}`;
          try {
            const e = JSON.parse(data);
            msg += ': ' + (e.error?.message || JSON.stringify(e.error));
          } catch {}
          return reject(new Error(msg));
        }
        try {
          resolve(data ? JSON.parse(data) : {});
        } catch {
          resolve(data);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Dataverse request timed out')); });
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

// ── Helper: format entity record for display ──

function formatRecord(record, fields) {
  const lines = [];
  for (const [key, val] of Object.entries(record)) {
    // Skip OData metadata fields unless specifically requested
    if (key.startsWith('@odata') || key.startsWith('_') && key.endsWith('_value')) continue;
    if (fields && !fields.includes(key)) continue;
    if (val === null || val === undefined) continue;
    lines.push(`  ${key}: ${val}`);
  }
  return lines.join('\n');
}

// ── Lookup-friendly field resolver ──
// Dataverse returns lookup values as _fieldname_value; include the annotation for display name

function formatLookup(record, field) {
  const raw = record[`_${field}_value`];
  const name = record[`_${field}_value@OData.Community.Display.V1.FormattedValue`]
    || record[`_${field}_value@Microsoft.Dynamics.CRM.lookuplogicalname`];
  if (!raw) return null;
  return name ? `${name} (${raw})` : raw;
}

// ── Main dispatcher ──

async function dynamics365(args) {
  const action = args.action;
  if (!action) return 'Error: action required (search|get_record|create_record|update_record|delete_record|list_records|get_entity_metadata|list_entities|whoami)';

  const config = getConfig();
  if (!config) return 'Error: Dynamics 365 not configured. Use Settings to add your D365 environment URL, Tenant ID, Client ID, and Client Secret.';

  switch (action) {
    case 'whoami': {
      const data = await dataverseRequest('GET', '/WhoAmI');
      return `Authenticated to Dynamics 365\n  User ID: ${data.UserId}\n  Organization ID: ${data.OrganizationId}\n  Business Unit ID: ${data.BusinessUnitId}\n  Environment: ${config.url}`;
    }

    case 'list_entities': {
      const filter = args.filter || "IsCustomizable/Value eq true";
      const count = Math.min(args.count || 50, 100);
      const data = await dataverseRequest('GET',
        `/EntityDefinitions?$select=LogicalName,DisplayName,EntitySetName,IsCustomEntity&$filter=${encodeURIComponent(filter)}&$top=${count}`);
      const entities = data.value || [];
      if (entities.length === 0) return 'No entities found matching filter.';
      return `Found ${entities.length} entities:\n\n` +
        entities.map(e => {
          const display = e.DisplayName?.UserLocalizedLabel?.Label || e.LogicalName;
          return `${display} (${e.LogicalName})\n  Entity Set: ${e.EntitySetName} | Custom: ${e.IsCustomEntity ? 'Yes' : 'No'}`;
        }).join('\n\n');
    }

    case 'get_entity_metadata': {
      if (!args.entity) return 'Error: entity logical name required (e.g. account, contact, incident)';
      const data = await dataverseRequest('GET',
        `/EntityDefinitions(LogicalName='${encodeURIComponent(args.entity)}')?$expand=Attributes($select=LogicalName,DisplayName,AttributeType,IsCustomAttribute;$top=${args.count || 50})`);
      const display = data.DisplayName?.UserLocalizedLabel?.Label || data.LogicalName;
      let output = `${display} (${data.LogicalName})\n`;
      output += `Entity Set: ${data.EntitySetName} | Primary Key: ${data.PrimaryIdAttribute} | Primary Name: ${data.PrimaryNameAttribute}\n`;
      output += `Custom: ${data.IsCustomEntity ? 'Yes' : 'No'} | Ownership: ${data.OwnershipType}\n`;

      if (data.Attributes?.length) {
        output += `\nAttributes (${data.Attributes.length}):\n`;
        output += data.Attributes.map(a => {
          const attrDisplay = a.DisplayName?.UserLocalizedLabel?.Label || a.LogicalName;
          return `  ${attrDisplay} (${a.LogicalName}) — ${a.AttributeType}${a.IsCustomAttribute ? ' [custom]' : ''}`;
        }).join('\n');
      }
      return output;
    }

    case 'list_records': {
      if (!args.entity) return 'Error: entity set name required (e.g. accounts, contacts, incidents)';
      const count = Math.min(args.count || 20, 50);
      let path = `/${encodeURIComponent(args.entity)}?$top=${count}`;
      if (args.select) path += `&$select=${encodeURIComponent(args.select)}`;
      if (args.filter) path += `&$filter=${encodeURIComponent(args.filter)}`;
      if (args.orderby) path += `&$orderby=${encodeURIComponent(args.orderby)}`;
      if (args.expand) path += `&$expand=${encodeURIComponent(args.expand)}`;

      const data = await dataverseRequest('GET', path);
      const records = data.value || [];
      if (records.length === 0) return `No records found in ${args.entity}.`;

      return `Found ${records.length} record(s) in ${args.entity}:\n\n` +
        records.map((r, i) => {
          const name = r.name || r.fullname || r.title || r.subject || r.ticketnumber || r[Object.keys(r).find(k => !k.startsWith('@') && !k.startsWith('_'))] || `Record ${i + 1}`;
          return `[${i + 1}] ${name}\n${formatRecord(r)}`;
        }).join('\n\n');
    }

    case 'get_record': {
      if (!args.entity) return 'Error: entity set name required (e.g. accounts, contacts, incidents)';
      if (!args.id) return 'Error: record ID (GUID) required';
      let path = `/${encodeURIComponent(args.entity)}(${args.id})`;
      if (args.select) path += `?$select=${encodeURIComponent(args.select)}`;
      if (args.expand) {
        path += (args.select ? '&' : '?') + `$expand=${encodeURIComponent(args.expand)}`;
      }

      const data = await dataverseRequest('GET', path);
      const name = data.name || data.fullname || data.title || data.subject || data.ticketnumber || args.id;
      return `${name}\n${formatRecord(data)}`;
    }

    case 'search': {
      if (!args.query) return 'Error: query text required';
      if (!args.entity) return 'Error: entity set name required (e.g. accounts, contacts, incidents)';

      // Use OData $filter with contains() for text search
      const nameField = args.field || 'name';
      const count = Math.min(args.count || 20, 50);
      let path = `/${encodeURIComponent(args.entity)}?$filter=contains(${encodeURIComponent(nameField)},\'${encodeURIComponent(args.query)}\')&$top=${count}`;
      if (args.select) path += `&$select=${encodeURIComponent(args.select)}`;
      if (args.orderby) path += `&$orderby=${encodeURIComponent(args.orderby)}`;

      const data = await dataverseRequest('GET', path);
      const records = data.value || [];
      if (records.length === 0) return `No records found matching "${args.query}" in ${args.entity}.`;

      return `Found ${records.length} record(s) matching "${args.query}" in ${args.entity}:\n\n` +
        records.map((r, i) => {
          const name = r.name || r.fullname || r.title || r.subject || r.ticketnumber || `Record ${i + 1}`;
          return `[${i + 1}] ${name}\n${formatRecord(r)}`;
        }).join('\n\n');
    }

    case 'create_record': {
      if (!args.entity) return 'Error: entity set name required (e.g. accounts, contacts, incidents)';
      if (!args.data) return 'Error: data required (JSON object of field values)';
      let recordData;
      try {
        recordData = typeof args.data === 'string' ? JSON.parse(args.data) : args.data;
      } catch {
        return 'Error: data must be valid JSON (e.g. {"name": "Acme Corp", "telephone1": "555-0100"})';
      }

      const data = await dataverseRequest('POST', `/${encodeURIComponent(args.entity)}`, recordData);
      // New record ID is in the OData-EntityId header or response body
      const id = data[Object.keys(data).find(k => k.endsWith('id'))] || 'created';
      return `Record created in ${args.entity}\n  ID: ${id}\n${formatRecord(data)}`;
    }

    case 'update_record': {
      if (!args.entity) return 'Error: entity set name required';
      if (!args.id) return 'Error: record ID (GUID) required';
      if (!args.data) return 'Error: data required (JSON object of fields to update)';
      let updateData;
      try {
        updateData = typeof args.data === 'string' ? JSON.parse(args.data) : args.data;
      } catch {
        return 'Error: data must be valid JSON';
      }

      await dataverseRequest('PATCH', `/${encodeURIComponent(args.entity)}(${args.id})`, updateData);
      return `Record ${args.id} updated in ${args.entity}`;
    }

    case 'delete_record': {
      if (!args.entity) return 'Error: entity set name required';
      if (!args.id) return 'Error: record ID (GUID) required';

      await dataverseRequest('DELETE', `/${encodeURIComponent(args.entity)}(${args.id})`);
      return `Record ${args.id} deleted from ${args.entity}`;
    }

    default:
      return `Error: unknown dynamics365 action '${action}'. Use: search, get_record, create_record, update_record, delete_record, list_records, get_entity_metadata, list_entities, whoami`;
  }
}

module.exports = { dynamics365 };
