// PRE Web GUI — Jira Server tool
// Uses Jira REST API v2 with Personal Access Token

const https = require('https');
const http = require('http');
const fs = require('fs');
const { URL } = require('url');
const { CONNECTIONS_FILE } = require('../constants');

function getJiraConfig() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    if (!data.jira_url || !data.jira_token) return null;
    return { url: data.jira_url.replace(/\/+$/, ''), token: data.jira_token };
  } catch {
    return null;
  }
}

function jiraRequest(method, path, config, body) {
  return new Promise((resolve, reject) => {
    const baseUrl = new URL(config.url);
    const fullPath = `/rest/api/2${path}`;
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
      // Allow self-signed certs for on-prem Jira
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
          let msg = `Jira API ${res.statusCode}`;
          try { msg += ': ' + (JSON.parse(data).errorMessages || []).join(', '); } catch {}
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

async function jira(args) {
  const action = args.action;
  if (!action) return 'Error: action required (search|get_issue|create_issue|comment|transition|assign|list_projects|get_project|my_issues)';

  const config = getJiraConfig();
  if (!config) return 'Error: Jira not configured. Use Settings to add your Jira Server URL and personal access token.';

  switch (action) {
    case 'search': {
      const jql = args.jql || args.query || 'assignee = currentUser() ORDER BY updated DESC';
      const maxResults = Math.min(args.count || 20, 50);
      const fields = 'summary,status,assignee,priority,issuetype,created,updated';
      const data = await jiraRequest('GET',
        `/search?jql=${encodeURIComponent(jql)}&maxResults=${maxResults}&fields=${fields}`, config);
      const issues = data.issues || [];
      if (issues.length === 0) return `No issues found for JQL: ${jql}`;
      return `Found ${data.total} issues (showing ${issues.length}):\n\n` +
        issues.map(i => {
          const f = i.fields;
          return `${i.key} [${f.status?.name}] ${f.summary}\n  Type: ${f.issuetype?.name} | Priority: ${f.priority?.name || 'None'} | Assignee: ${f.assignee?.displayName || 'Unassigned'} | Updated: ${f.updated?.slice(0, 10)}`;
        }).join('\n\n');
    }

    case 'my_issues': {
      const jql = 'assignee = currentUser() AND status != Done ORDER BY updated DESC';
      const maxResults = Math.min(args.count || 20, 50);
      const fields = 'summary,status,assignee,priority,issuetype,updated';
      const data = await jiraRequest('GET',
        `/search?jql=${encodeURIComponent(jql)}&maxResults=${maxResults}&fields=${fields}`, config);
      const issues = data.issues || [];
      if (issues.length === 0) return 'No open issues assigned to you.';
      return `Your open issues (${data.total} total):\n\n` +
        issues.map(i => {
          const f = i.fields;
          return `${i.key} [${f.status?.name}] ${f.summary}\n  Type: ${f.issuetype?.name} | Priority: ${f.priority?.name || 'None'} | Updated: ${f.updated?.slice(0, 10)}`;
        }).join('\n\n');
    }

    case 'get_issue': {
      if (!args.key && !args.id) return 'Error: key or id required (e.g. PROJ-123)';
      const issueKey = args.key || args.id;
      const data = await jiraRequest('GET', `/issue/${issueKey}`, config);
      const f = data.fields;
      let output = `${data.key}: ${f.summary}\n`;
      output += `Status: ${f.status?.name} | Type: ${f.issuetype?.name} | Priority: ${f.priority?.name || 'None'}\n`;
      output += `Assignee: ${f.assignee?.displayName || 'Unassigned'} | Reporter: ${f.reporter?.displayName || 'Unknown'}\n`;
      output += `Created: ${f.created?.slice(0, 10)} | Updated: ${f.updated?.slice(0, 10)}\n`;
      if (f.labels?.length) output += `Labels: ${f.labels.join(', ')}\n`;
      if (f.components?.length) output += `Components: ${f.components.map(c => c.name).join(', ')}\n`;
      if (f.fixVersions?.length) output += `Fix Versions: ${f.fixVersions.map(v => v.name).join(', ')}\n`;
      output += `\nDescription:\n${f.description || '(no description)'}`;

      // Fetch comments
      if (f.comment?.comments?.length) {
        output += '\n\n--- Comments ---\n';
        for (const c of f.comment.comments.slice(-10)) {
          output += `\n${c.author?.displayName} (${c.created?.slice(0, 10)}):\n${c.body}\n`;
        }
      }
      return output;
    }

    case 'create_issue': {
      if (!args.project) return 'Error: project required (project key, e.g. PROJ)';
      if (!args.summary) return 'Error: summary required';
      const issueData = {
        fields: {
          project: { key: args.project },
          summary: args.summary,
          issuetype: { name: args.type || 'Task' },
        },
      };
      if (args.description) issueData.fields.description = args.description;
      if (args.priority) issueData.fields.priority = { name: args.priority };
      if (args.assignee) issueData.fields.assignee = { name: args.assignee };
      if (args.labels) {
        issueData.fields.labels = typeof args.labels === 'string' ? args.labels.split(',').map(l => l.trim()) : args.labels;
      }

      const data = await jiraRequest('POST', '/issue', config, issueData);
      return `Issue created: ${data.key}\nURL: ${config.url}/browse/${data.key}`;
    }

    case 'comment': {
      if (!args.key && !args.id) return 'Error: key required (e.g. PROJ-123)';
      if (!args.body && !args.text) return 'Error: body/text required';
      const issueKey = args.key || args.id;
      const commentBody = args.body || args.text;
      const data = await jiraRequest('POST', `/issue/${issueKey}/comment`, config, { body: commentBody });
      return `Comment added to ${issueKey} (comment ID: ${data.id})`;
    }

    case 'transition': {
      if (!args.key && !args.id) return 'Error: key required (e.g. PROJ-123)';
      if (!args.status && !args.transition) return 'Error: status or transition name required';
      const issueKey = args.key || args.id;
      const targetName = (args.status || args.transition).toLowerCase();

      // Get available transitions
      const transitions = await jiraRequest('GET', `/issue/${issueKey}/transitions`, config);
      const match = (transitions.transitions || []).find(t =>
        t.name.toLowerCase() === targetName || t.to?.name?.toLowerCase() === targetName
      );
      if (!match) {
        const available = (transitions.transitions || []).map(t => `${t.name} → ${t.to?.name}`).join(', ');
        return `Error: transition "${args.status || args.transition}" not available. Available: ${available}`;
      }

      await jiraRequest('POST', `/issue/${issueKey}/transitions`, config, {
        transition: { id: match.id },
      });
      return `${issueKey} transitioned to: ${match.to?.name || match.name}`;
    }

    case 'assign': {
      if (!args.key && !args.id) return 'Error: key required (e.g. PROJ-123)';
      const issueKey = args.key || args.id;
      const assignee = args.assignee || args.user || null;
      await jiraRequest('PUT', `/issue/${issueKey}/assignee`, config, { name: assignee });
      return assignee ? `${issueKey} assigned to ${assignee}` : `${issueKey} unassigned`;
    }

    case 'list_projects': {
      const data = await jiraRequest('GET', '/project', config);
      if (!data || data.length === 0) return 'No projects found.';
      return data.map(p =>
        `${p.key} — ${p.name}${p.projectCategory ? ` [${p.projectCategory.name}]` : ''}`
      ).join('\n');
    }

    case 'get_project': {
      if (!args.key && !args.project) return 'Error: key/project required';
      const projectKey = args.key || args.project;
      const data = await jiraRequest('GET', `/project/${projectKey}`, config);
      let output = `${data.key}: ${data.name}\n`;
      output += `Lead: ${data.lead?.displayName || 'Unknown'}\n`;
      if (data.description) output += `Description: ${data.description}\n`;
      if (data.issueTypes?.length) output += `Issue Types: ${data.issueTypes.map(t => t.name).join(', ')}\n`;
      output += `URL: ${config.url}/browse/${data.key}`;
      return output;
    }

    default:
      return `Error: unknown jira action '${action}'. Use: search, get_issue, create_issue, comment, transition, assign, list_projects, get_project, my_issues`;
  }
}

module.exports = { jira };
