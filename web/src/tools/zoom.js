// PRE Web GUI — Zoom tool
// Uses Zoom Server-to-Server OAuth (account_id + client_id + client_secret)
// Docs: https://developers.zoom.us/docs/internal-apps/s2s-oauth/

const https = require('https');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('../constants');

let cachedToken = null;
let tokenExpiry = 0;

function getZoomConfig() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    if (!data.zoom_account_id || !data.zoom_client_id || !data.zoom_client_secret) return null;
    return {
      accountId: data.zoom_account_id,
      clientId: data.zoom_client_id,
      clientSecret: data.zoom_client_secret,
    };
  } catch {
    return null;
  }
}

/**
 * Get an access token via Server-to-Server OAuth
 */
function getAccessToken(config) {
  return new Promise((resolve, reject) => {
    // Return cached token if still valid
    if (cachedToken && Date.now() < tokenExpiry - 60000) {
      return resolve(cachedToken);
    }

    const auth = Buffer.from(`${config.clientId}:${config.clientSecret}`).toString('base64');
    const postData = `grant_type=account_credentials&account_id=${encodeURIComponent(config.accountId)}`;

    const req = https.request({
      hostname: 'zoom.us',
      path: '/oauth/token',
      method: 'POST',
      headers: {
        'Authorization': `Basic ${auth}`,
        'Content-Type': 'application/x-www-form-urlencoded',
        'Content-Length': Buffer.byteLength(postData),
      },
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (parsed.error) return reject(new Error(parsed.reason || parsed.error));
          cachedToken = parsed.access_token;
          tokenExpiry = Date.now() + (parsed.expires_in || 3600) * 1000;
          resolve(cachedToken);
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

function zoomRequest(method, path, token, body) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.zoom.us',
      path: `/v2${path}`,
      method,
      headers: {
        'Authorization': `Bearer ${token}`,
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
        if (res.statusCode === 204) return resolve({});
        if (res.statusCode >= 400) {
          let msg = `Zoom API ${res.statusCode}`;
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

async function zoom(args) {
  const action = args.action;
  if (!action) return 'Error: action required (list_meetings|get_meeting|create_meeting|update_meeting|delete_meeting|list_recordings|list_users|me)';

  const config = getZoomConfig();
  if (!config) return 'Error: Zoom not configured. Use Settings to add your Zoom Server-to-Server OAuth credentials (Account ID, Client ID, Client Secret).';

  const token = await getAccessToken(config);

  switch (action) {
    case 'me': {
      const data = await zoomRequest('GET', '/users/me', token);
      return `${data.first_name} ${data.last_name} (${data.email})\nID: ${data.id} | Type: ${data.type === 1 ? 'Basic' : data.type === 2 ? 'Licensed' : data.type} | Status: ${data.status}\nTimezone: ${data.timezone || 'Not set'}`;
    }

    case 'list_users': {
      const count = Math.min(args.count || 30, 300);
      const data = await zoomRequest('GET', `/users?page_size=${count}&status=active`, token);
      const users = data.users || [];
      if (users.length === 0) return 'No users found.';
      return `Users (${data.total_records || users.length}):\n\n` + users.map(u =>
        `${u.first_name} ${u.last_name} (${u.email})\n  ID: ${u.id} | Type: ${u.type === 1 ? 'Basic' : u.type === 2 ? 'Licensed' : u.type} | Status: ${u.status}`
      ).join('\n\n');
    }

    case 'list_meetings': {
      const userId = args.user_id || 'me';
      const type = args.type || 'upcoming';
      const count = Math.min(args.count || 20, 100);
      const data = await zoomRequest('GET', `/users/${userId}/meetings?type=${type}&page_size=${count}`, token);
      const meetings = data.meetings || [];
      if (meetings.length === 0) return `No ${type} meetings found.`;
      return `Meetings (${data.total_records || meetings.length}):\n\n` + meetings.map(m => {
        const start = formatDate(m.start_time);
        const duration = m.duration ? `${m.duration} min` : '';
        return `${m.topic}\n  ID: ${m.id} | ${start} | ${duration} | Type: ${m.type === 1 ? 'Instant' : m.type === 2 ? 'Scheduled' : m.type === 8 ? 'Recurring' : m.type}\n  Join: ${m.join_url || 'N/A'}`;
      }).join('\n\n');
    }

    case 'get_meeting': {
      if (!args.id) return 'Error: id required (meeting ID)';
      const data = await zoomRequest('GET', `/meetings/${args.id}`, token);
      let output = `${data.topic}\n`;
      output += `ID: ${data.id} | Type: ${data.type === 2 ? 'Scheduled' : data.type === 8 ? 'Recurring' : data.type}\n`;
      output += `Start: ${formatDate(data.start_time)} | Duration: ${data.duration || 0} min\n`;
      output += `Host: ${data.host_email || 'Unknown'}\n`;
      output += `Status: ${data.status || 'N/A'}\n`;
      if (data.agenda) output += `Agenda: ${data.agenda}\n`;
      if (data.join_url) output += `Join URL: ${data.join_url}\n`;
      if (data.password) output += `Passcode: ${data.password}\n`;
      const settings = data.settings || {};
      output += `Settings: Waiting Room: ${settings.waiting_room ? 'Yes' : 'No'} | Recording: ${settings.auto_recording || 'none'}`;
      return output;
    }

    case 'create_meeting': {
      if (!args.topic) return 'Error: topic required';
      const body = {
        topic: args.topic,
        type: args.type === 'instant' ? 1 : 2,
      };
      if (args.start_time) body.start_time = args.start_time;
      if (args.duration) body.duration = parseInt(args.duration);
      if (args.agenda) body.agenda = args.agenda;
      if (args.timezone) body.timezone = args.timezone;
      if (args.password) body.password = args.password;

      const userId = args.user_id || 'me';
      const data = await zoomRequest('POST', `/users/${userId}/meetings`, token, body);
      return `Meeting created: ${data.topic}\nID: ${data.id}\nJoin URL: ${data.join_url}\nStart URL: ${data.start_url}\n${data.password ? `Passcode: ${data.password}` : ''}`;
    }

    case 'update_meeting': {
      if (!args.id) return 'Error: id required (meeting ID)';
      const body = {};
      if (args.topic) body.topic = args.topic;
      if (args.start_time) body.start_time = args.start_time;
      if (args.duration) body.duration = parseInt(args.duration);
      if (args.agenda) body.agenda = args.agenda;
      if (args.timezone) body.timezone = args.timezone;

      await zoomRequest('PATCH', `/meetings/${args.id}`, token, body);
      return `Meeting ${args.id} updated.`;
    }

    case 'delete_meeting': {
      if (!args.id) return 'Error: id required (meeting ID)';
      await zoomRequest('DELETE', `/meetings/${args.id}`, token);
      return `Meeting ${args.id} deleted.`;
    }

    case 'list_recordings': {
      const userId = args.user_id || 'me';
      const from = args.from || new Date(Date.now() - 30 * 86400000).toISOString().slice(0, 10);
      const to = args.to || new Date().toISOString().slice(0, 10);
      const data = await zoomRequest('GET', `/users/${userId}/recordings?from=${from}&to=${to}`, token);
      const meetings = data.meetings || [];
      if (meetings.length === 0) return `No recordings found from ${from} to ${to}.`;
      return `Recordings (${data.total_records || meetings.length}):\n\n` + meetings.map(m => {
        const files = (m.recording_files || []).map(f =>
          `    ${f.file_type} (${f.file_size ? (f.file_size / 1048576).toFixed(1) + 'MB' : '?'}): ${f.download_url || 'N/A'}`
        ).join('\n');
        return `${m.topic} — ${formatDate(m.start_time)} (${m.duration || 0} min)\n  ID: ${m.id}\n${files || '    (no files)'}`;
      }).join('\n\n');
    }

    default:
      return `Error: unknown zoom action '${action}'. Use: list_meetings, get_meeting, create_meeting, update_meeting, delete_meeting, list_recordings, list_users, me`;
  }
}

module.exports = { zoom };
