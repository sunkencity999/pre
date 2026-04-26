// PRE Web GUI — Asana tool
// Uses Asana REST API with Personal Access Token

const https = require('https');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('../constants');

function getAsanaConfig() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    if (!data.asana_token) return null;
    return { token: data.asana_token };
  } catch {
    return null;
  }
}

function asanaRequest(method, path, config, body) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'app.asana.com',
      path: `/api/1.0${path}`,
      method,
      headers: {
        'Authorization': `Bearer ${config.token}`,
        'Accept': 'application/json',
        'User-Agent': 'PRE-Web-GUI',
      },
    };

    let postData;
    if (body) {
      postData = JSON.stringify({ data: body });
      options.headers['Content-Type'] = 'application/json';
      options.headers['Content-Length'] = Buffer.byteLength(postData);
    }

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          let msg = `Asana API ${res.statusCode}`;
          try {
            const err = JSON.parse(data);
            msg += ': ' + (err.errors?.map(e => e.message).join(', ') || data.slice(0, 300));
          } catch {}
          return reject(new Error(msg));
        }
        try {
          const parsed = JSON.parse(data);
          resolve(parsed.data !== undefined ? parsed.data : parsed);
        } catch {
          resolve(data);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Asana request timed out')); });
    if (postData) req.write(postData);
    req.end();
  });
}

function formatDate(d) {
  if (!d) return '';
  return d.slice(0, 10);
}

async function asana(args) {
  const action = args.action;
  if (!action) return 'Error: action required (list_workspaces|list_projects|get_project|list_tasks|get_task|create_task|update_task|add_comment|search|list_sections|me)';

  const config = getAsanaConfig();
  if (!config) return 'Error: Asana not configured. Use Settings to add your Asana Personal Access Token.';

  switch (action) {
    case 'me': {
      const data = await asanaRequest('GET', '/users/me', config);
      let output = `${data.name} (${data.email})\nGID: ${data.gid}`;
      if (data.workspaces?.length) {
        output += `\n\nWorkspaces:\n${data.workspaces.map(w => `  ${w.name} (${w.gid})`).join('\n')}`;
      }
      return output;
    }

    case 'list_workspaces': {
      const data = await asanaRequest('GET', '/workspaces?limit=50', config);
      if (!data || data.length === 0) return 'No workspaces found.';
      return data.map(w => `${w.name}\n  GID: ${w.gid}`).join('\n\n');
    }

    case 'list_projects': {
      const workspace = args.workspace;
      let path = '/projects?opt_fields=name,color,current_status,due_on,created_at,modified_at,owner.name&limit=50';
      if (workspace) path += `&workspace=${workspace}`;
      if (args.team) path += `&team=${args.team}`;

      const data = await asanaRequest('GET', path, config);
      if (!data || data.length === 0) return 'No projects found.';
      return `Projects (${data.length}):\n\n` + data.map(p => {
        const status = p.current_status?.text || p.current_status?.color || 'No status';
        const owner = p.owner?.name || 'No owner';
        return `${p.name}\n  GID: ${p.gid} | Owner: ${owner} | Due: ${formatDate(p.due_on) || 'None'} | Status: ${status}`;
      }).join('\n\n');
    }

    case 'get_project': {
      if (!args.id && !args.gid) return 'Error: id/gid required';
      const gid = args.id || args.gid;
      const data = await asanaRequest('GET', `/projects/${gid}?opt_fields=name,notes,color,current_status,due_on,start_on,created_at,modified_at,owner.name,team.name,members.name`, config);

      let output = `${data.name}\n`;
      output += `GID: ${data.gid}\n`;
      if (data.owner?.name) output += `Owner: ${data.owner.name}\n`;
      if (data.team?.name) output += `Team: ${data.team.name}\n`;
      if (data.start_on) output += `Start: ${formatDate(data.start_on)}\n`;
      if (data.due_on) output += `Due: ${formatDate(data.due_on)}\n`;
      if (data.current_status) output += `Status: ${data.current_status.text || data.current_status.color}\n`;
      if (data.members?.length) output += `Members: ${data.members.map(m => m.name).join(', ')}\n`;
      output += `\nNotes:\n${data.notes || '(no description)'}`;
      return output;
    }

    case 'list_tasks': {
      if (!args.project && !args.section && !args.assignee) {
        return 'Error: project, section, or assignee required';
      }
      const count = Math.min(args.count || 25, 100);
      let path = `/tasks?opt_fields=name,completed,assignee.name,due_on,modified_at,tags.name&limit=${count}`;
      if (args.project) path += `&project=${args.project}`;
      if (args.section) path += `&section=${args.section}`;
      if (args.assignee && args.workspace) {
        path = `/tasks?opt_fields=name,completed,assignee.name,due_on,modified_at,tags.name&limit=${count}&assignee=${args.assignee}&workspace=${args.workspace}`;
      }

      const data = await asanaRequest('GET', path, config);
      if (!data || data.length === 0) return 'No tasks found.';
      return `Tasks (${data.length}):\n\n` + data.map(t => {
        const done = t.completed ? ' [DONE]' : '';
        const assignee = t.assignee?.name || 'Unassigned';
        const tags = t.tags?.map(tag => tag.name).join(', ') || '';
        return `${t.name}${done}\n  GID: ${t.gid} | Assignee: ${assignee} | Due: ${formatDate(t.due_on) || 'None'}${tags ? ` | Tags: ${tags}` : ''}`;
      }).join('\n\n');
    }

    case 'get_task': {
      if (!args.id && !args.gid) return 'Error: id/gid required';
      const gid = args.id || args.gid;
      const data = await asanaRequest('GET', `/tasks/${gid}?opt_fields=name,notes,html_notes,completed,assignee.name,due_on,start_on,created_at,modified_at,tags.name,projects.name,memberships.section.name,parent.name,num_subtasks,custom_fields`, config);

      let output = `${data.name}${data.completed ? ' [COMPLETED]' : ''}\n`;
      output += `GID: ${data.gid}\n`;
      if (data.assignee?.name) output += `Assignee: ${data.assignee.name}\n`;
      if (data.start_on) output += `Start: ${formatDate(data.start_on)}\n`;
      if (data.due_on) output += `Due: ${formatDate(data.due_on)}\n`;
      if (data.projects?.length) output += `Projects: ${data.projects.map(p => p.name).join(', ')}\n`;
      if (data.memberships?.length) {
        const sections = data.memberships.filter(m => m.section?.name).map(m => m.section.name);
        if (sections.length) output += `Sections: ${sections.join(', ')}\n`;
      }
      if (data.tags?.length) output += `Tags: ${data.tags.map(t => t.name).join(', ')}\n`;
      if (data.parent?.name) output += `Parent: ${data.parent.name}\n`;
      if (data.num_subtasks) output += `Subtasks: ${data.num_subtasks}\n`;
      if (data.custom_fields?.length) {
        const fields = data.custom_fields.filter(f => f.display_value).map(f => `${f.name}: ${f.display_value}`);
        if (fields.length) output += `Custom fields: ${fields.join(', ')}\n`;
      }
      output += `Created: ${formatDate(data.created_at)} | Modified: ${formatDate(data.modified_at)}\n`;
      output += `\nDescription:\n${data.notes || '(no description)'}`;

      // Fetch stories (comments)
      try {
        const stories = await asanaRequest('GET', `/tasks/${gid}/stories?opt_fields=text,created_by.name,created_at,type&limit=15`, config);
        const comments = (stories || []).filter(s => s.type === 'comment');
        if (comments.length > 0) {
          output += '\n\n--- Comments ---\n';
          for (const c of comments.slice(-10)) {
            output += `\n${c.created_by?.name || 'Unknown'} (${formatDate(c.created_at)}):\n${c.text}\n`;
          }
        }
      } catch {}
      return output;
    }

    case 'create_task': {
      if (!args.name) return 'Error: name required';
      const body = { name: args.name };
      if (args.project) body.projects = [args.project];
      if (args.section) body.memberships = [{ project: args.project, section: args.section }];
      if (args.assignee) body.assignee = args.assignee;
      if (args.due_on) body.due_on = args.due_on;
      if (args.notes) body.notes = args.notes;
      if (args.workspace) body.workspace = args.workspace;

      const data = await asanaRequest('POST', '/tasks', config, body);
      return `Task created: ${data.name}\nGID: ${data.gid}${data.permalink_url ? `\nURL: ${data.permalink_url}` : ''}`;
    }

    case 'update_task': {
      if (!args.id && !args.gid) return 'Error: id/gid required';
      const gid = args.id || args.gid;
      const body = {};
      if (args.name) body.name = args.name;
      if (args.notes) body.notes = args.notes;
      if (args.assignee) body.assignee = args.assignee;
      if (args.due_on) body.due_on = args.due_on;
      if (args.completed !== undefined) body.completed = args.completed === true || args.completed === 'true';

      const data = await asanaRequest('PUT', `/tasks/${gid}`, config, body);
      return `Task updated: ${data.name}${data.completed ? ' [COMPLETED]' : ''}\nGID: ${data.gid}`;
    }

    case 'add_comment': {
      if (!args.id && !args.gid) return 'Error: id/gid required';
      if (!args.text) return 'Error: text required';
      const gid = args.id || args.gid;

      const data = await asanaRequest('POST', `/tasks/${gid}/stories`, config, { text: args.text });
      return `Comment added to task ${gid} (story GID: ${data.gid})`;
    }

    case 'search': {
      if (!args.query) return 'Error: query required';
      if (!args.workspace) return 'Error: workspace required for search';

      const params = [`text=${encodeURIComponent(args.query)}`, 'opt_fields=name,completed,assignee.name,due_on,projects.name'];
      if (args.assignee) params.push(`assignee.any=${args.assignee}`);
      if (args.project) params.push(`projects.any=${args.project}`);
      params.push('limit=25');

      const data = await asanaRequest('GET', `/workspaces/${args.workspace}/tasks/search?${params.join('&')}`, config);
      if (!data || data.length === 0) return `No tasks found for: ${args.query}`;
      return `Search results for "${args.query}" (${data.length}):\n\n` + data.map(t => {
        const done = t.completed ? ' [DONE]' : '';
        const projects = t.projects?.map(p => p.name).join(', ') || '';
        return `${t.name}${done}\n  GID: ${t.gid} | Assignee: ${t.assignee?.name || 'Unassigned'} | Due: ${formatDate(t.due_on) || 'None'}${projects ? ` | Projects: ${projects}` : ''}`;
      }).join('\n\n');
    }

    case 'list_sections': {
      if (!args.project) return 'Error: project required';
      const data = await asanaRequest('GET', `/projects/${args.project}/sections?opt_fields=name,created_at`, config);
      if (!data || data.length === 0) return 'No sections found.';
      return data.map(s => `${s.name}\n  GID: ${s.gid}`).join('\n\n');
    }

    default:
      return `Error: unknown asana action '${action}'. Use: list_workspaces, list_projects, get_project, list_tasks, get_task, create_task, update_task, add_comment, search, list_sections, me`;
  }
}

module.exports = { asana };
