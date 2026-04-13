// PRE Web GUI — Figma tool
// Uses Figma REST API with Personal Access Token

const https = require('https');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('../constants');

function getFigmaConfig() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    if (!data.figma_token) return null;
    return { token: data.figma_token };
  } catch {
    return null;
  }
}

function figmaRequest(method, path, config, body) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.figma.com',
      path,
      method,
      headers: {
        'X-Figma-Token': config.token,
        'Accept': 'application/json',
        'User-Agent': 'PRE-Web-GUI',
      },
    };

    let postData;
    if (body) {
      postData = JSON.stringify(body);
      options.headers['Content-Type'] = 'application/json';
      options.headers['Content-Length'] = Buffer.byteLength(postData);
    }

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          let msg = `Figma API ${res.statusCode}`;
          try { msg += ': ' + JSON.parse(data).message; } catch {}
          return reject(new Error(msg));
        }
        try { resolve(JSON.parse(data)); } catch { resolve(data); }
      });
    });
    req.on('error', reject);
    if (postData) req.write(postData);
    req.end();
  });
}

function formatDate(d) {
  if (!d) return '';
  return new Date(d).toISOString().slice(0, 16).replace('T', ' ');
}

async function figma(args) {
  const action = args.action;
  if (!action) return 'Error: action required (get_file|get_file_nodes|get_comments|post_comment|list_projects|get_project_files|get_team_projects|get_file_versions|get_images|me)';

  const config = getFigmaConfig();
  if (!config) return 'Error: Figma not configured. Use Settings to add your Figma Personal Access Token.';

  switch (action) {
    case 'me': {
      const data = await figmaRequest('GET', '/v1/me', config);
      return `${data.handle} (${data.email})\nID: ${data.id} | Image: ${data.img_url || 'none'}`;
    }

    case 'get_file': {
      if (!args.file_key) return 'Error: file_key required (from Figma URL: figma.com/file/<file_key>/...)';
      const params = [];
      if (args.depth) params.push(`depth=${args.depth}`);
      const qs = params.length ? `?${params.join('&')}` : '';
      const data = await figmaRequest('GET', `/v1/files/${args.file_key}${qs}`, config);

      let output = `${data.name}\n`;
      output += `Last modified: ${formatDate(data.lastModified)}\n`;
      output += `Version: ${data.version}\n`;
      if (data.thumbnailUrl) output += `Thumbnail: ${data.thumbnailUrl}\n`;

      // List top-level pages
      const pages = data.document?.children || [];
      if (pages.length > 0) {
        output += `\nPages (${pages.length}):\n`;
        for (const page of pages) {
          const childCount = page.children?.length || 0;
          output += `  ${page.name} — ${childCount} top-level frames (ID: ${page.id})\n`;
        }
      }

      // Component count
      const compCount = Object.keys(data.components || {}).length;
      const styleCount = Object.keys(data.styles || {}).length;
      if (compCount || styleCount) {
        output += `\nComponents: ${compCount} | Styles: ${styleCount}`;
      }
      return output;
    }

    case 'get_file_nodes': {
      if (!args.file_key) return 'Error: file_key required';
      if (!args.ids) return 'Error: ids required (comma-separated node IDs)';
      const ids = encodeURIComponent(args.ids);
      const data = await figmaRequest('GET', `/v1/files/${args.file_key}/nodes?ids=${ids}`, config);

      const nodes = data.nodes || {};
      const results = [];
      for (const [id, info] of Object.entries(nodes)) {
        if (!info?.document) continue;
        const n = info.document;
        results.push(`${n.name} (${n.type}) — ID: ${id}\n  Size: ${n.absoluteBoundingBox ? `${n.absoluteBoundingBox.width}x${n.absoluteBoundingBox.height}` : 'N/A'}\n  Children: ${n.children?.length || 0}`);
      }
      return results.length > 0 ? results.join('\n\n') : 'No nodes found.';
    }

    case 'get_comments': {
      if (!args.file_key) return 'Error: file_key required';
      const data = await figmaRequest('GET', `/v1/files/${args.file_key}/comments`, config);
      const comments = data.comments || [];
      if (comments.length === 0) return 'No comments on this file.';

      return `Comments (${comments.length}):\n\n` + comments.slice(-20).map(c => {
        const resolved = c.resolved_at ? ' [RESOLVED]' : '';
        return `${c.user?.handle || 'Unknown'} (${formatDate(c.created_at)})${resolved}:\n${c.message}\n  ID: ${c.id}${c.order_id ? ` | Thread: ${c.order_id}` : ''}`;
      }).join('\n\n');
    }

    case 'post_comment': {
      if (!args.file_key) return 'Error: file_key required';
      if (!args.message) return 'Error: message required';
      const body = { message: args.message };
      if (args.comment_id) body.comment_id = args.comment_id; // reply to thread
      const data = await figmaRequest('POST', `/v1/files/${args.file_key}/comments`, config, body);
      return `Comment posted (ID: ${data.id})`;
    }

    case 'get_team_projects': {
      if (!args.team_id) return 'Error: team_id required';
      const data = await figmaRequest('GET', `/v1/teams/${args.team_id}/projects`, config);
      const projects = data.projects || [];
      if (projects.length === 0) return 'No projects found.';
      return projects.map(p => `${p.name}\n  ID: ${p.id}`).join('\n\n');
    }

    case 'list_projects': {
      if (!args.team_id) return 'Error: team_id required (find via get_team_projects or Figma URL)';
      const data = await figmaRequest('GET', `/v1/teams/${args.team_id}/projects`, config);
      const projects = data.projects || [];
      if (projects.length === 0) return 'No projects found.';
      return projects.map(p => `${p.name}\n  ID: ${p.id}`).join('\n\n');
    }

    case 'get_project_files': {
      if (!args.project_id) return 'Error: project_id required';
      const data = await figmaRequest('GET', `/v1/projects/${args.project_id}/files`, config);
      const files = data.files || [];
      if (files.length === 0) return 'No files in project.';
      return `Files (${files.length}):\n\n` + files.map(f =>
        `${f.name}\n  Key: ${f.key} | Last modified: ${formatDate(f.last_modified)}\n  Thumbnail: ${f.thumbnail_url || 'N/A'}`
      ).join('\n\n');
    }

    case 'get_file_versions': {
      if (!args.file_key) return 'Error: file_key required';
      const data = await figmaRequest('GET', `/v1/files/${args.file_key}/versions`, config);
      const versions = data.versions || [];
      if (versions.length === 0) return 'No versions found.';
      return `Versions (${versions.length}):\n\n` + versions.slice(0, 20).map(v =>
        `${v.label || '(unnamed)'} — ${formatDate(v.created_at)}\n  By: ${v.user?.handle || 'Unknown'} | ID: ${v.id}\n  ${v.description || ''}`
      ).join('\n\n');
    }

    case 'get_images': {
      if (!args.file_key) return 'Error: file_key required';
      if (!args.ids) return 'Error: ids required (comma-separated node IDs to export)';
      const format = args.format || 'png';
      const scale = args.scale || '2';
      const ids = encodeURIComponent(args.ids);
      const data = await figmaRequest('GET', `/v1/images/${args.file_key}?ids=${ids}&format=${format}&scale=${scale}`, config);

      if (data.err) return `Error: ${data.err}`;
      const images = data.images || {};
      const results = [];
      for (const [id, url] of Object.entries(images)) {
        results.push(`Node ${id}: ${url || '(no image generated)'}`);
      }
      return results.length > 0 ? `Exported ${format.toUpperCase()} images:\n\n` + results.join('\n') : 'No images generated.';
    }

    default:
      return `Error: unknown figma action '${action}'. Use: get_file, get_file_nodes, get_comments, post_comment, list_projects, get_project_files, get_team_projects, get_file_versions, get_images, me`;
  }
}

module.exports = { figma };
