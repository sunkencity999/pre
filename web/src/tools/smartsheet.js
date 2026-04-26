// PRE Web GUI — Smartsheet tool
// Uses Smartsheet REST API 2.0 with API Access Token

const https = require('https');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('../constants');

function getSmartsheetConfig() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    if (!data.smartsheet_token) return null;
    return { token: data.smartsheet_token };
  } catch {
    return null;
  }
}

function ssRequest(method, path, config, body) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.smartsheet.com',
      path: `/2.0${path}`,
      method,
      headers: {
        'Authorization': `Bearer ${config.token}`,
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
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          let msg = `Smartsheet API ${res.statusCode}`;
          try {
            const parsed = JSON.parse(data);
            if (parsed.message) msg += ': ' + parsed.message;
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
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Smartsheet request timed out')); });
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

async function smartsheet(args) {
  const action = args.action;
  if (!action) return 'Error: action required (list_sheets|get_sheet|search|create_sheet|add_rows|update_rows|delete_rows|get_columns|add_column|list_workspaces|get_workspace|add_comment|me)';

  const config = getSmartsheetConfig();
  if (!config) return 'Error: Smartsheet not configured. Use Settings to add your Smartsheet API access token.';

  switch (action) {
    case 'me': {
      const data = await ssRequest('GET', '/users/me', config);
      return `User: ${data.email}\nName: ${data.firstName} ${data.lastName}\nAccount: ${data.account?.name || 'Unknown'}\nAdmin: ${data.admin ? 'Yes' : 'No'}`;
    }

    case 'list_sheets': {
      const pageSize = Math.min(args.count || 25, 100);
      const data = await ssRequest('GET', `/sheets?pageSize=${pageSize}&includeAll=false`, config);
      const sheets = data.data || [];
      if (sheets.length === 0) return 'No sheets found.';
      return `Found ${data.totalCount || sheets.length} sheets (showing ${sheets.length}):\n\n` +
        sheets.map(s => {
          return `${s.name} (ID: ${s.id})\n  Rows: ${s.totalRowCount || 0} | Columns: ${s.columns?.length || '?'} | Modified: ${s.modifiedAt?.slice(0, 10) || 'Unknown'} | Access: ${s.accessLevel || 'Unknown'}`;
        }).join('\n\n');
    }

    case 'get_sheet': {
      if (!args.id && !args.name) return 'Error: id or name required';
      let sheetId = args.id;

      // If name provided, search for it
      if (!sheetId && args.name) {
        const search = await ssRequest('GET', `/search?query=${encodeURIComponent(args.name)}`, config);
        const match = (search.results || []).find(r =>
          r.objectType === 'sheet' && r.text.toLowerCase() === args.name.toLowerCase()
        );
        if (!match) {
          // Try partial match
          const partial = (search.results || []).find(r => r.objectType === 'sheet');
          if (!partial) return `Sheet not found: "${args.name}"`;
          sheetId = partial.objectId;
        } else {
          sheetId = match.objectId;
        }
      }

      const rowCount = Math.min(args.count || 50, 500);
      const data = await ssRequest('GET', `/sheets/${sheetId}?rowsModifiedSince=&pageSize=${rowCount}`, config);

      let output = `${data.name} (ID: ${data.id})\n`;
      output += `Rows: ${data.totalRowCount || 0} | Created: ${data.createdAt?.slice(0, 10)} | Modified: ${data.modifiedAt?.slice(0, 10)}\n`;
      output += `Access: ${data.accessLevel || 'Unknown'} | Permalink: ${data.permalink || ''}\n`;

      // Column headers
      const columns = data.columns || [];
      if (columns.length > 0) {
        output += `\nColumns: ${columns.map(c => `${c.title} [${c.type}]`).join(' | ')}\n`;
      }

      // Rows
      const rows = data.rows || [];
      if (rows.length > 0) {
        output += `\n--- Data (${rows.length} rows) ---\n`;
        // Header row
        output += columns.map(c => c.title).join(' | ') + '\n';
        output += columns.map(() => '---').join(' | ') + '\n';

        for (const row of rows) {
          const cells = row.cells || [];
          const values = columns.map(col => {
            const cell = cells.find(c => c.columnId === col.id);
            if (!cell) return '';
            return cell.displayValue || cell.value || '';
          });
          output += values.join(' | ') + '\n';
        }
      } else {
        output += '\n(no rows)';
      }
      return output;
    }

    case 'search': {
      if (!args.query) return 'Error: query required';
      const data = await ssRequest('GET', `/search?query=${encodeURIComponent(args.query)}`, config);
      const results = data.results || [];
      if (results.length === 0) return `No results found for: ${args.query}`;
      return `Found ${data.totalCount || results.length} results:\n\n` +
        results.map(r => {
          return `${r.text} [${r.objectType}]\n  Context: ${r.contextData?.join(' > ') || 'N/A'}`;
        }).join('\n\n');
    }

    case 'create_sheet': {
      if (!args.name) return 'Error: name required';
      if (!args.columns) return 'Error: columns required (comma-separated column names, or JSON array of {title, type} objects)';

      let columns;
      if (typeof args.columns === 'string') {
        try {
          columns = JSON.parse(args.columns);
        } catch {
          // Treat as comma-separated names
          const names = args.columns.split(',').map(n => n.trim()).filter(Boolean);
          columns = names.map((name, i) => ({
            title: name,
            type: 'TEXT_NUMBER',
            primary: i === 0,
          }));
        }
      } else {
        columns = args.columns;
      }

      // Ensure first column is primary
      if (columns.length > 0 && !columns.some(c => c.primary)) {
        columns[0].primary = true;
      }

      const sheetData = { name: args.name, columns };

      let data;
      if (args.workspace_id) {
        data = await ssRequest('POST', `/workspaces/${args.workspace_id}/sheets`, config, sheetData);
      } else {
        data = await ssRequest('POST', '/sheets', config, sheetData);
      }

      const result = data.result || data;
      return `Sheet created: ${result.name || args.name}\nID: ${result.id}\nPermalink: ${result.permalink || ''}`;
    }

    case 'add_rows': {
      if (!args.id) return 'Error: sheet id required';
      if (!args.rows) return 'Error: rows required (JSON array of row objects, or pipe-delimited text matching column order)';

      // Get sheet columns for mapping
      const sheet = await ssRequest('GET', `/sheets/${args.id}?pageSize=0`, config);
      const columns = sheet.columns || [];

      let rows;
      if (typeof args.rows === 'string') {
        try {
          rows = JSON.parse(args.rows);
        } catch {
          // Parse pipe-delimited rows: "val1|val2|val3\nval4|val5|val6"
          const lines = args.rows.split('\n').filter(l => l.trim());
          rows = lines.map(line => {
            const values = line.split('|').map(v => v.trim());
            const cells = values.map((val, i) => {
              if (i >= columns.length) return null;
              return { columnId: columns[i].id, value: val };
            }).filter(Boolean);
            return { toBottom: true, cells };
          });
        }
      } else {
        rows = args.rows;
      }

      // If rows are simple value arrays, map to cells
      const mappedRows = rows.map(row => {
        if (row.cells) return row;
        if (Array.isArray(row)) {
          return {
            toBottom: true,
            cells: row.map((val, i) => {
              if (i >= columns.length) return null;
              return { columnId: columns[i].id, value: val };
            }).filter(Boolean),
          };
        }
        // Object with column names as keys
        return {
          toBottom: true,
          cells: Object.entries(row).map(([key, val]) => {
            const col = columns.find(c => c.title.toLowerCase() === key.toLowerCase());
            if (!col) return null;
            return { columnId: col.id, value: val };
          }).filter(Boolean),
        };
      });

      const data = await ssRequest('POST', `/sheets/${args.id}/rows`, config, mappedRows);
      const added = data.result?.length || mappedRows.length;
      return `Added ${added} row(s) to sheet ${args.id}`;
    }

    case 'update_rows': {
      if (!args.id) return 'Error: sheet id required';
      if (!args.rows) return 'Error: rows required (JSON array with rowId and cells)';

      let rows;
      if (typeof args.rows === 'string') {
        try { rows = JSON.parse(args.rows); } catch { return 'Error: rows must be valid JSON'; }
      } else {
        rows = args.rows;
      }

      const data = await ssRequest('PUT', `/sheets/${args.id}/rows`, config, rows);
      const updated = data.result?.length || rows.length;
      return `Updated ${updated} row(s) in sheet ${args.id}`;
    }

    case 'delete_rows': {
      if (!args.id) return 'Error: sheet id required';
      if (!args.row_ids) return 'Error: row_ids required (comma-separated row IDs)';

      const ids = typeof args.row_ids === 'string'
        ? args.row_ids.split(',').map(id => id.trim()).join(',')
        : args.row_ids.join(',');

      await ssRequest('DELETE', `/sheets/${args.id}/rows?ids=${ids}`, config);
      return `Deleted rows from sheet ${args.id}`;
    }

    case 'get_columns': {
      if (!args.id) return 'Error: sheet id required';
      const data = await ssRequest('GET', `/sheets/${args.id}/columns`, config);
      const columns = data.data || [];
      if (columns.length === 0) return 'No columns found.';
      return columns.map(c => {
        let info = `${c.title} (ID: ${c.id}) [${c.type}]`;
        if (c.primary) info += ' *PRIMARY*';
        if (c.options?.length) info += ` Options: ${c.options.join(', ')}`;
        return info;
      }).join('\n');
    }

    case 'add_column': {
      if (!args.id) return 'Error: sheet id required';
      if (!args.title) return 'Error: title required';
      const colData = {
        title: args.title,
        type: args.type || 'TEXT_NUMBER',
        index: args.index !== undefined ? parseInt(args.index) : undefined,
      };
      if (args.options) {
        colData.options = typeof args.options === 'string'
          ? args.options.split(',').map(o => o.trim())
          : args.options;
        if (!args.type) colData.type = 'PICKLIST';
      }
      const data = await ssRequest('POST', `/sheets/${args.id}/columns`, config, [colData]);
      const result = data.result?.[0] || data;
      return `Column added: ${result.title || args.title} (ID: ${result.id || 'N/A'}) [${result.type || colData.type}]`;
    }

    case 'list_workspaces': {
      const data = await ssRequest('GET', '/workspaces', config);
      const workspaces = data.data || [];
      if (workspaces.length === 0) return 'No workspaces found.';
      return workspaces.map(w =>
        `${w.name} (ID: ${w.id}) | Access: ${w.accessLevel || 'Unknown'}`
      ).join('\n');
    }

    case 'get_workspace': {
      if (!args.id) return 'Error: workspace id required';
      const data = await ssRequest('GET', `/workspaces/${args.id}`, config);
      let output = `${data.name} (ID: ${data.id})\n`;
      output += `Access: ${data.accessLevel || 'Unknown'} | Permalink: ${data.permalink || ''}\n`;
      const sheets = data.sheets || [];
      if (sheets.length > 0) {
        output += `\nSheets (${sheets.length}):\n`;
        output += sheets.map(s => `  ${s.name} (ID: ${s.id})`).join('\n');
      }
      const folders = data.folders || [];
      if (folders.length > 0) {
        output += `\nFolders (${folders.length}):\n`;
        output += folders.map(f => `  ${f.name} (ID: ${f.id})`).join('\n');
      }
      return output;
    }

    case 'add_comment': {
      if (!args.id) return 'Error: sheet id required';
      if (!args.text && !args.body) return 'Error: text/body required';
      const commentText = args.text || args.body;
      const data = await ssRequest('POST', `/sheets/${args.id}/discussions`, config, {
        comment: { text: commentText },
      });
      return `Discussion created on sheet ${args.id} (ID: ${data.result?.id || 'N/A'})`;
    }

    default:
      return `Error: unknown smartsheet action '${action}'. Use: list_sheets, get_sheet, search, create_sheet, add_rows, update_rows, delete_rows, get_columns, add_column, list_workspaces, get_workspace, add_comment, me`;
  }
}

module.exports = { smartsheet };
