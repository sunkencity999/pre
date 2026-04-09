// PRE Web GUI — Confluence Server tool
// Uses Confluence REST API with Personal Access Token

const https = require('https');
const http = require('http');
const fs = require('fs');
const { URL } = require('url');
const { CONNECTIONS_FILE } = require('../constants');

function getConfluenceConfig() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    if (!data.confluence_url || !data.confluence_token) return null;
    return { url: data.confluence_url.replace(/\/+$/, ''), token: data.confluence_token };
  } catch {
    return null;
  }
}

function confluenceRequest(method, path, config, body) {
  return new Promise((resolve, reject) => {
    const baseUrl = new URL(config.url);
    const fullPath = `/rest/api${path}`;
    const isHttps = baseUrl.protocol === 'https:';
    const transport = isHttps ? https : http;

    const options = {
      hostname: baseUrl.hostname,
      port: baseUrl.port || (isHttps ? 443 : 80),
      path: fullPath,
      method,
      headers: {
        'Authorization': `Bearer ${config.token}`,
        'Accept': 'application/json',
        'User-Agent': 'PRE-Web-GUI',
      },
      rejectUnauthorized: false,
    };
    if (body) {
      const bodyStr = JSON.stringify(body);
      options.headers['Content-Type'] = 'application/json';
      options.headers['Content-Length'] = Buffer.byteLength(bodyStr);
    }

    const req = transport.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          let msg = `Confluence API ${res.statusCode}`;
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
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

/**
 * Strip Confluence storage format HTML to plain text
 */
function stripHtml(html) {
  if (!html) return '';
  return html
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n\n')
    .replace(/<\/li>/gi, '\n')
    .replace(/<li>/gi, '  • ')
    .replace(/<\/h[1-6]>/gi, '\n')
    .replace(/<h[1-6][^>]*>/gi, '\n## ')
    .replace(/<[^>]+>/g, '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

async function confluence(args) {
  const action = args.action;
  if (!action) return 'Error: action required (search|get_page|create_page|update_page|get_space|list_spaces|get_children|get_comments|add_comment)';

  const config = getConfluenceConfig();
  if (!config) return 'Error: Confluence not configured. Use Settings to add your Confluence Server URL and personal access token.';

  switch (action) {
    case 'search': {
      const cql = args.cql || args.query;
      if (!cql) return 'Error: cql or query required';
      const limit = Math.min(args.count || 20, 50);
      const encodedCql = encodeURIComponent(cql);
      const data = await confluenceRequest('GET',
        `/content/search?cql=${encodedCql}&limit=${limit}&expand=space,version`, config);
      const results = data.results || [];
      if (results.length === 0) return `No results found for: ${cql}`;
      return `Found ${data.totalSize || results.length} results (showing ${results.length}):\n\n` +
        results.map(r => {
          return `${r.title} [${r.type}]\n  Space: ${r.space?.name || r.space?.key || 'Unknown'} | Version: ${r.version?.number || 1} | Last modified by: ${r.version?.by?.displayName || 'Unknown'}\n  URL: ${config.url}${r._links?.webui || ''}`;
        }).join('\n\n');
    }

    case 'get_page': {
      if (!args.id && !args.title) return 'Error: id or title required';
      let page;
      if (args.id) {
        page = await confluenceRequest('GET',
          `/content/${args.id}?expand=body.storage,version,space,children.page,ancestors`, config);
      } else {
        // Search by title (optionally in a space)
        let cql = `title="${args.title}"`;
        if (args.space) cql += ` AND space="${args.space}"`;
        cql += ' AND type=page';
        const search = await confluenceRequest('GET',
          `/content/search?cql=${encodeURIComponent(cql)}&limit=1&expand=body.storage,version,space,ancestors`, config);
        if (!search.results || search.results.length === 0) return `Page not found: "${args.title}"`;
        page = search.results[0];
      }

      let output = `${page.title}\n`;
      output += `ID: ${page.id} | Space: ${page.space?.name || page.space?.key || 'Unknown'} | Type: ${page.type}\n`;
      output += `Version: ${page.version?.number || 1} | Last modified by: ${page.version?.by?.displayName || 'Unknown'} (${page.version?.when?.slice(0, 10) || ''})\n`;
      if (page.ancestors?.length) {
        output += `Path: ${page.ancestors.map(a => a.title).join(' > ')}\n`;
      }
      output += `URL: ${config.url}${page._links?.webui || ''}\n`;

      const body = page.body?.storage?.value;
      if (body) {
        output += `\n--- Content ---\n${stripHtml(body)}`;
      }
      return output;
    }

    case 'create_page': {
      if (!args.space) return 'Error: space key required';
      if (!args.title) return 'Error: title required';
      if (!args.content && !args.body) return 'Error: content/body required';
      const pageData = {
        type: 'page',
        title: args.title,
        space: { key: args.space },
        body: {
          storage: {
            value: args.content || args.body,
            representation: 'storage',
          },
        },
      };
      if (args.parent_id) {
        pageData.ancestors = [{ id: args.parent_id }];
      }
      const data = await confluenceRequest('POST', '/content', config, pageData);
      return `Page created: ${data.title}\nID: ${data.id}\nURL: ${config.url}${data._links?.webui || ''}`;
    }

    case 'update_page': {
      if (!args.id) return 'Error: page id required';
      if (!args.content && !args.body && !args.title) return 'Error: content/body or title required';

      // Get current version
      const current = await confluenceRequest('GET', `/content/${args.id}?expand=version,body.storage`, config);
      const newVersion = (current.version?.number || 0) + 1;

      const updateData = {
        type: 'page',
        title: args.title || current.title,
        version: { number: newVersion },
        body: {
          storage: {
            value: args.content || args.body || current.body?.storage?.value || '',
            representation: 'storage',
          },
        },
      };
      const data = await confluenceRequest('PUT', `/content/${args.id}`, config, updateData);
      return `Page updated: ${data.title} (version ${newVersion})\nURL: ${config.url}${data._links?.webui || ''}`;
    }

    case 'list_spaces': {
      const limit = Math.min(args.count || 25, 100);
      const data = await confluenceRequest('GET', `/space?limit=${limit}&expand=description.plain`, config);
      const spaces = data.results || [];
      if (spaces.length === 0) return 'No spaces found.';
      return spaces.map(s => {
        const desc = s.description?.plain?.value ? ` — ${s.description.plain.value.slice(0, 80)}` : '';
        return `${s.key}: ${s.name} [${s.type}]${desc}`;
      }).join('\n');
    }

    case 'get_space': {
      if (!args.key && !args.space) return 'Error: key/space required';
      const spaceKey = args.key || args.space;
      const data = await confluenceRequest('GET',
        `/space/${spaceKey}?expand=description.plain,homepage`, config);
      let output = `${data.key}: ${data.name}\n`;
      output += `Type: ${data.type}\n`;
      if (data.description?.plain?.value) output += `Description: ${data.description.plain.value}\n`;
      if (data.homepage) output += `Homepage: ${data.homepage.title} (ID: ${data.homepage.id})\n`;
      output += `URL: ${config.url}/display/${data.key}`;
      return output;
    }

    case 'get_children': {
      if (!args.id) return 'Error: page id required';
      const limit = Math.min(args.count || 25, 50);
      const data = await confluenceRequest('GET',
        `/content/${args.id}/child/page?limit=${limit}&expand=version`, config);
      const pages = data.results || [];
      if (pages.length === 0) return 'No child pages found.';
      return pages.map(p =>
        `${p.title} (ID: ${p.id}) | Version: ${p.version?.number || 1} | Modified: ${p.version?.when?.slice(0, 10) || ''}`
      ).join('\n');
    }

    case 'get_comments': {
      if (!args.id) return 'Error: page id required';
      const limit = Math.min(args.count || 20, 50);
      const data = await confluenceRequest('GET',
        `/content/${args.id}/child/comment?limit=${limit}&expand=body.storage,version`, config);
      const comments = data.results || [];
      if (comments.length === 0) return 'No comments found.';
      return comments.map(c => {
        const author = c.version?.by?.displayName || 'Unknown';
        const date = c.version?.when?.slice(0, 10) || '';
        const body = stripHtml(c.body?.storage?.value || '');
        return `${author} (${date}):\n${body}`;
      }).join('\n\n');
    }

    case 'add_comment': {
      if (!args.id) return 'Error: page id required';
      if (!args.body && !args.text && !args.content) return 'Error: body/text/content required';
      const commentBody = args.body || args.text || args.content;
      const commentData = {
        type: 'comment',
        container: { id: args.id, type: 'page' },
        body: {
          storage: {
            value: `<p>${commentBody}</p>`,
            representation: 'storage',
          },
        },
      };
      const data = await confluenceRequest('POST', '/content', config, commentData);
      return `Comment added to page ${args.id} (comment ID: ${data.id})`;
    }

    default:
      return `Error: unknown confluence action '${action}'. Use: search, get_page, create_page, update_page, list_spaces, get_space, get_children, get_comments, add_comment`;
  }
}

module.exports = { confluence };
