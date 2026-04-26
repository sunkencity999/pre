// PRE Web GUI — Linear tool
// Uses Linear GraphQL API with Personal API Key

const https = require('https');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('../constants');

function getLinearConfig() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    if (!data.linear_token) return null;
    return { token: data.linear_token };
  } catch {
    return null;
  }
}

function graphql(query, variables, config) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({ query, variables });
    const req = https.request({
      hostname: 'api.linear.app',
      path: '/graphql',
      method: 'POST',
      headers: {
        'Authorization': config.token,
        'Content-Type': 'application/json',
        'User-Agent': 'PRE-Web-GUI',
        'Content-Length': Buffer.byteLength(body),
      },
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          return reject(new Error(`Linear API HTTP ${res.statusCode}: ${data.slice(0, 300)}`));
        }
        try {
          const parsed = JSON.parse(data);
          if (parsed.errors && parsed.errors.length > 0) {
            return reject(new Error(`Linear API: ${parsed.errors.map(e => e.message).join(', ')}`));
          }
          resolve(parsed.data);
        } catch {
          resolve(data);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Linear request timed out')); });
    req.write(body);
    req.end();
  });
}

async function linear(args) {
  const action = args.action;
  if (!action) return 'Error: action required (list_issues|search|get_issue|create_issue|update_issue|add_comment|list_projects|list_teams|list_cycles|list_labels|me)';

  const config = getLinearConfig();
  if (!config) return 'Error: Linear not configured. Use Settings to add your Linear API key.';

  switch (action) {
    case 'me': {
      const data = await graphql(`{ viewer { id name email displayName admin } }`, {}, config);
      const v = data.viewer;
      return `${v.displayName || v.name} (${v.email})\nID: ${v.id} | Admin: ${v.admin ? 'Yes' : 'No'}`;
    }

    case 'list_teams': {
      const data = await graphql(`{ teams { nodes { id name key description } } }`, {}, config);
      const teams = data.teams?.nodes || [];
      if (teams.length === 0) return 'No teams found.';
      return teams.map(t => `${t.key} — ${t.name}${t.description ? `\n  ${t.description}` : ''}\n  ID: ${t.id}`).join('\n\n');
    }

    case 'list_projects': {
      const count = Math.min(args.count || 25, 50);
      const data = await graphql(`query($first: Int!) {
        projects(first: $first, orderBy: updatedAt) {
          nodes { id name state progress startDate targetDate lead { name } teams { nodes { key } } }
        }
      }`, { first: count }, config);
      const projects = data.projects?.nodes || [];
      if (projects.length === 0) return 'No projects found.';
      return projects.map(p => {
        const teamKeys = p.teams?.nodes?.map(t => t.key).join(', ') || '';
        const lead = p.lead?.name || 'No lead';
        const dates = [p.startDate, p.targetDate].filter(Boolean).join(' - ');
        return `${p.name} [${p.state}] ${Math.round(p.progress * 100)}%\n  Lead: ${lead} | Teams: ${teamKeys}${dates ? ` | ${dates}` : ''}\n  ID: ${p.id}`;
      }).join('\n\n');
    }

    case 'list_issues': {
      const count = Math.min(args.count || 25, 50);
      const team = args.team;
      const project = args.project;
      const status = args.status;

      let filter = '';
      const filterParts = [];
      if (team) filterParts.push(`team: { key: { eq: "${team}" } }`);
      if (project) filterParts.push(`project: { name: { eq: "${project}" } }`);
      if (status) filterParts.push(`state: { name: { eq: "${status}" } }`);
      if (filterParts.length > 0) filter = `filter: { ${filterParts.join(', ')} },`;

      const data = await graphql(`query($first: Int!) {
        issues(first: $first, ${filter} orderBy: updatedAt) {
          nodes { id identifier title state { name } priority assignee { name } project { name } updatedAt labels { nodes { name } } }
        }
      }`, { first: count }, config);
      const issues = data.issues?.nodes || [];
      if (issues.length === 0) return 'No issues found.';
      return `Issues (${issues.length}):\n\n` + issues.map(i => {
        const labels = i.labels?.nodes?.map(l => l.name).join(', ') || '';
        const proj = i.project?.name || '';
        return `${i.identifier} [${i.state?.name}] ${i.title}\n  Priority: ${i.priority} | Assignee: ${i.assignee?.name || 'Unassigned'}${proj ? ` | Project: ${proj}` : ''}${labels ? ` | Labels: ${labels}` : ''}\n  Updated: ${i.updatedAt?.slice(0, 10)}`;
      }).join('\n\n');
    }

    case 'search': {
      if (!args.query) return 'Error: query required';
      const data = await graphql(`query($q: String!) {
        searchIssues(query: $q, first: 20) {
          nodes { id identifier title state { name } priority assignee { name } updatedAt }
        }
      }`, { q: args.query }, config);
      const issues = data.searchIssues?.nodes || [];
      if (issues.length === 0) return `No issues found for: ${args.query}`;
      return `Search results for "${args.query}" (${issues.length}):\n\n` + issues.map(i =>
        `${i.identifier} [${i.state?.name}] ${i.title}\n  Priority: ${i.priority} | Assignee: ${i.assignee?.name || 'Unassigned'} | Updated: ${i.updatedAt?.slice(0, 10)}`
      ).join('\n\n');
    }

    case 'get_issue': {
      if (!args.id && !args.key) return 'Error: id or key required (e.g. ENG-123)';
      const issueId = args.id || args.key;
      const data = await graphql(`query($id: String!) {
        issue(id: $id) { id identifier title description state { name } priority priorityLabel
          assignee { name email } creator { name } project { name }
          labels { nodes { name } } comments { nodes { body user { name } createdAt } }
          createdAt updatedAt estimate cycle { name number } parent { identifier title } }
      }`, { id: issueId }, config);
      const i = data.issue;
      if (!i) return `Issue not found: ${issueId}`;
      let output = `${i.identifier}: ${i.title}\n`;
      output += `Status: ${i.state?.name} | Priority: ${i.priorityLabel || i.priority} | Estimate: ${i.estimate || 'None'}\n`;
      output += `Assignee: ${i.assignee?.name || 'Unassigned'} | Creator: ${i.creator?.name || 'Unknown'}\n`;
      if (i.project?.name) output += `Project: ${i.project.name}\n`;
      if (i.cycle?.name) output += `Cycle: ${i.cycle.name} (#${i.cycle.number})\n`;
      if (i.parent) output += `Parent: ${i.parent.identifier} ${i.parent.title}\n`;
      if (i.labels?.nodes?.length) output += `Labels: ${i.labels.nodes.map(l => l.name).join(', ')}\n`;
      output += `Created: ${i.createdAt?.slice(0, 10)} | Updated: ${i.updatedAt?.slice(0, 10)}\n`;
      output += `\nDescription:\n${i.description || '(no description)'}`;
      if (i.comments?.nodes?.length) {
        output += '\n\n--- Comments ---\n';
        for (const c of i.comments.nodes.slice(-10)) {
          output += `\n${c.user?.name || 'Unknown'} (${c.createdAt?.slice(0, 10)}):\n${c.body}\n`;
        }
      }
      return output;
    }

    case 'create_issue': {
      if (!args.title) return 'Error: title required';
      if (!args.team) return 'Error: team required (team key, e.g. ENG)';

      // Look up team ID from key
      const teamData = await graphql(`query($key: String!) {
        teams(filter: { key: { eq: $key } }) { nodes { id } }
      }`, { key: args.team }, config);
      const teamId = teamData.teams?.nodes?.[0]?.id;
      if (!teamId) return `Error: team not found: ${args.team}`;

      const input = { title: args.title, teamId };
      if (args.description) input.description = args.description;
      if (args.priority) input.priority = parseInt(args.priority);
      if (args.assignee) input.assigneeId = args.assignee;

      const data = await graphql(`mutation($input: IssueCreateInput!) {
        issueCreate(input: $input) { success issue { id identifier title url } }
      }`, { input }, config);
      const result = data.issueCreate;
      if (!result?.success) return 'Error: failed to create issue';
      return `Issue created: ${result.issue.identifier} — ${result.issue.title}\nURL: ${result.issue.url}`;
    }

    case 'update_issue': {
      if (!args.id && !args.key) return 'Error: id or key required';
      const issueId = args.id || args.key;
      const input = {};
      if (args.title) input.title = args.title;
      if (args.description) input.description = args.description;
      if (args.status) input.stateId = args.status;
      if (args.priority) input.priority = parseInt(args.priority);
      if (args.assignee) input.assigneeId = args.assignee;

      const data = await graphql(`mutation($id: String!, $input: IssueUpdateInput!) {
        issueUpdate(id: $id, input: $input) { success issue { identifier title state { name } } }
      }`, { id: issueId, input }, config);
      const result = data.issueUpdate;
      if (!result?.success) return `Error: failed to update issue ${issueId}`;
      return `Updated ${result.issue.identifier}: ${result.issue.title} [${result.issue.state?.name}]`;
    }

    case 'add_comment': {
      if (!args.id && !args.key) return 'Error: id or key required';
      if (!args.body && !args.text) return 'Error: body/text required';
      const issueId = args.id || args.key;
      const body = args.body || args.text;

      const data = await graphql(`mutation($issueId: String!, $body: String!) {
        commentCreate(input: { issueId: $issueId, body: $body }) { success comment { id } }
      }`, { issueId, body }, config);
      if (!data.commentCreate?.success) return `Error: failed to add comment to ${issueId}`;
      return `Comment added to ${issueId}`;
    }

    case 'list_cycles': {
      const team = args.team;
      let teamFilter = '';
      if (team) teamFilter = `filter: { team: { key: { eq: "${team}" } } },`;

      const data = await graphql(`{
        cycles(${teamFilter} first: 10, orderBy: updatedAt) {
          nodes { id number name startsAt endsAt progress { completed total } team { key } }
        }
      }`, {}, config);
      const cycles = data.cycles?.nodes || [];
      if (cycles.length === 0) return 'No cycles found.';
      return cycles.map(c => {
        const pct = c.progress?.total ? Math.round((c.progress.completed / c.progress.total) * 100) : 0;
        return `Cycle ${c.number}${c.name ? ` — ${c.name}` : ''} (${c.team?.key})\n  ${c.startsAt?.slice(0, 10)} to ${c.endsAt?.slice(0, 10)} | ${pct}% complete (${c.progress?.completed || 0}/${c.progress?.total || 0})`;
      }).join('\n\n');
    }

    case 'list_labels': {
      const data = await graphql(`{ issueLabels(first: 50) { nodes { id name color description } } }`, {}, config);
      const labels = data.issueLabels?.nodes || [];
      if (labels.length === 0) return 'No labels found.';
      return labels.map(l => `${l.name}${l.description ? ` — ${l.description}` : ''}\n  Color: ${l.color} | ID: ${l.id}`).join('\n');
    }

    default:
      return `Error: unknown linear action '${action}'. Use: list_issues, search, get_issue, create_issue, update_issue, add_comment, list_projects, list_teams, list_cycles, list_labels, me`;
  }
}

module.exports = { linear };
