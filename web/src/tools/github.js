// PRE Web GUI — GitHub tool
// Uses GitHub REST API with personal access token

const https = require('https');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('../constants');

function getToken() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    return data.github_key || null;
  } catch {
    return null;
  }
}

function ghRequest(method, path, token, body) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.github.com',
      path,
      method,
      headers: {
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'PRE-Web-GUI',
        'X-GitHub-Api-Version': '2022-11-28',
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
          return reject(new Error(`GitHub API ${res.statusCode}: ${data.slice(0, 300)}`));
        }
        try {
          resolve(JSON.parse(data));
        } catch {
          resolve(data);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('GitHub request timed out')); });
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

async function github(args) {
  const action = args.action;
  if (!action) return 'Error: action required (search_repos|list_repos|list_issues|read_issue|list_prs|read_pr|user)';

  const token = getToken();
  if (!token) return 'Error: GitHub not configured. Use Settings to add your token.';

  switch (action) {
    case 'user': {
      const user = args.username
        ? await ghRequest('GET', `/users/${args.username}`, token)
        : await ghRequest('GET', '/user', token);
      return `User: ${user.login}\nName: ${user.name || ''}\nPublic repos: ${user.public_repos}\nFollowers: ${user.followers}\nBio: ${user.bio || ''}`;
    }

    case 'list_repos': {
      const perPage = Math.min(args.count || 20, 100);
      const sort = args.sort || 'updated';
      const data = args.username
        ? await ghRequest('GET', `/users/${args.username}/repos?per_page=${perPage}&sort=${sort}`, token)
        : await ghRequest('GET', `/user/repos?per_page=${perPage}&sort=${sort}&affiliation=owner`, token);
      if (!data || data.length === 0) return 'No repositories found.';
      return data.map(r =>
        `${r.full_name}${r.private ? ' (private)' : ''} - ${r.description || 'No description'}\n  Stars: ${r.stargazers_count} | Forks: ${r.forks_count} | Lang: ${r.language || 'N/A'} | Updated: ${r.updated_at?.slice(0, 10)}`
      ).join('\n\n');
    }

    case 'search_repos': {
      if (!args.query) return 'Error: query required for search_repos';
      const perPage = Math.min(args.count || 10, 30);
      const data = await ghRequest('GET', `/search/repositories?q=${encodeURIComponent(args.query)}&per_page=${perPage}`, token);
      const items = data.items || [];
      if (items.length === 0) return `No repositories found for: ${args.query}`;
      return items.map(r =>
        `${r.full_name} - ${r.description || 'No description'}\n  Stars: ${r.stargazers_count} | Lang: ${r.language || 'N/A'}`
      ).join('\n\n');
    }

    case 'list_issues': {
      if (!args.repo) return 'Error: repo required (owner/name)';
      const state = args.state || 'open';
      const perPage = Math.min(args.count || 10, 30);
      const data = await ghRequest('GET', `/repos/${args.repo}/issues?state=${state}&per_page=${perPage}`, token);
      if (!data || data.length === 0) return `No ${state} issues in ${args.repo}.`;
      return data
        .filter(i => !i.pull_request) // exclude PRs
        .map(i => `#${i.number} [${i.state}] ${i.title}\n  By: ${i.user?.login} | ${i.comments} comments | ${i.created_at?.slice(0, 10)}`)
        .join('\n\n');
    }

    case 'read_issue': {
      if (!args.repo || !args.number) return 'Error: repo and number required';
      const issue = await ghRequest('GET', `/repos/${args.repo}/issues/${args.number}`, token);
      let output = `#${issue.number} [${issue.state}] ${issue.title}\nBy: ${issue.user?.login} | Created: ${issue.created_at?.slice(0, 10)}\n`;
      if (issue.labels?.length) output += `Labels: ${issue.labels.map(l => l.name).join(', ')}\n`;
      if (issue.assignees?.length) output += `Assignees: ${issue.assignees.map(a => a.login).join(', ')}\n`;
      output += `\n${issue.body || '(no body)'}`;

      // Fetch comments
      if (issue.comments > 0) {
        const comments = await ghRequest('GET', `/repos/${args.repo}/issues/${args.number}/comments?per_page=10`, token);
        output += '\n\n--- Comments ---\n';
        for (const c of comments) {
          output += `\n${c.user?.login} (${c.created_at?.slice(0, 10)}):\n${c.body}\n`;
        }
      }
      return output;
    }

    case 'list_prs': {
      if (!args.repo) return 'Error: repo required (owner/name)';
      const state = args.state || 'open';
      const perPage = Math.min(args.count || 10, 30);
      const data = await ghRequest('GET', `/repos/${args.repo}/pulls?state=${state}&per_page=${perPage}`, token);
      if (!data || data.length === 0) return `No ${state} PRs in ${args.repo}.`;
      return data.map(pr =>
        `#${pr.number} [${pr.state}] ${pr.title}\n  By: ${pr.user?.login} | ${pr.created_at?.slice(0, 10)} | ${pr.head?.ref} → ${pr.base?.ref}`
      ).join('\n\n');
    }

    case 'read_pr': {
      if (!args.repo || !args.number) return 'Error: repo and number required';
      const pr = await ghRequest('GET', `/repos/${args.repo}/pulls/${args.number}`, token);
      let output = `#${pr.number} [${pr.state}] ${pr.title}\n`;
      output += `By: ${pr.user?.login} | ${pr.head?.ref} → ${pr.base?.ref}\n`;
      output += `Created: ${pr.created_at?.slice(0, 10)} | Merged: ${pr.merged ? 'Yes' : 'No'}\n`;
      output += `+${pr.additions} -${pr.deletions} across ${pr.changed_files} files\n`;
      output += `\n${pr.body || '(no body)'}`;
      return output;
    }

    default:
      return `Error: unknown github action '${action}'. Use: search_repos, list_repos, list_issues, read_issue, list_prs, read_pr, user`;
  }
}

module.exports = { github };
