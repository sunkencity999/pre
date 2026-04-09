// PRE Web GUI — Slack tool
// Uses Slack Web API with Bot User OAuth Token (xoxb-)

const https = require('https');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('../constants');

function getSlackConfig() {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    if (!data.slack_token) return null;
    return { token: data.slack_token };
  } catch {
    return null;
  }
}

function slackRequest(method, path, config, body) {
  return new Promise((resolve, reject) => {
    const isGet = method === 'GET';
    let fullPath = `/api${path}`;
    if (isGet && body) {
      const params = new URLSearchParams(body).toString();
      fullPath += (fullPath.includes('?') ? '&' : '?') + params;
    }

    const options = {
      hostname: 'slack.com',
      path: fullPath,
      method,
      headers: {
        'Authorization': `Bearer ${config.token}`,
        'Accept': 'application/json',
        'User-Agent': 'PRE-Web-GUI',
      },
    };

    let postData;
    if (!isGet && body) {
      postData = JSON.stringify(body);
      options.headers['Content-Type'] = 'application/json; charset=utf-8';
      options.headers['Content-Length'] = Buffer.byteLength(postData);
    }

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          return reject(new Error(`Slack API HTTP ${res.statusCode}: ${data.slice(0, 300)}`));
        }
        try {
          const parsed = JSON.parse(data);
          if (!parsed.ok) {
            return reject(new Error(`Slack API error: ${parsed.error || 'unknown'}`));
          }
          resolve(parsed);
        } catch {
          resolve(data);
        }
      });
    });
    req.on('error', reject);
    if (postData) req.write(postData);
    req.end();
  });
}

/**
 * Format a Slack timestamp to readable date
 */
function formatTs(ts) {
  if (!ts) return '';
  const ms = parseFloat(ts) * 1000;
  return new Date(ms).toISOString().slice(0, 16).replace('T', ' ');
}

async function slack(args) {
  const action = args.action;
  if (!action) return 'Error: action required (list_channels|get_channel|history|send|reply|update|react|search|list_users|get_user|me)';

  const config = getSlackConfig();
  if (!config) return 'Error: Slack not configured. Use Settings to add your Slack Bot User OAuth Token.';

  switch (action) {
    case 'me': {
      const data = await slackRequest('GET', '/auth.test', config, {});
      return `Bot: ${data.user}\nTeam: ${data.team}\nUser ID: ${data.user_id}\nTeam ID: ${data.team_id}\nURL: ${data.url || ''}`;
    }

    case 'list_channels': {
      const limit = Math.min(args.count || 50, 200);
      const types = args.types || 'public_channel,private_channel';
      const params = {
        limit,
        types,
        exclude_archived: args.include_archived ? 'false' : 'true',
      };
      if (args.cursor) params.cursor = args.cursor;

      const data = await slackRequest('GET', '/conversations.list', config, params);
      const channels = data.channels || [];
      if (channels.length === 0) return 'No channels found.';

      let output = `Channels (${channels.length}):\n\n`;
      output += channels.map(ch => {
        const members = ch.num_members !== undefined ? `${ch.num_members} members` : '';
        const priv = ch.is_private ? ' (private)' : '';
        const archived = ch.is_archived ? ' [archived]' : '';
        const purpose = ch.purpose?.value ? ` — ${ch.purpose.value.slice(0, 80)}` : '';
        return `#${ch.name}${priv}${archived} (ID: ${ch.id})\n  ${members}${purpose}`;
      }).join('\n\n');

      if (data.response_metadata?.next_cursor) {
        output += `\n\n[More results available — use cursor: "${data.response_metadata.next_cursor}"]`;
      }
      return output;
    }

    case 'get_channel': {
      if (!args.channel) return 'Error: channel required (ID or #name)';
      const channelId = await resolveChannel(args.channel, config);
      if (!channelId) return `Error: channel not found: ${args.channel}`;

      const data = await slackRequest('GET', '/conversations.info', config, { channel: channelId });
      const ch = data.channel;
      let output = `#${ch.name} (ID: ${ch.id})\n`;
      output += `Private: ${ch.is_private ? 'Yes' : 'No'} | Archived: ${ch.is_archived ? 'Yes' : 'No'} | Members: ${ch.num_members || '?'}\n`;
      if (ch.topic?.value) output += `Topic: ${ch.topic.value}\n`;
      if (ch.purpose?.value) output += `Purpose: ${ch.purpose.value}\n`;
      output += `Created: ${new Date(ch.created * 1000).toISOString().slice(0, 10)}`;
      return output;
    }

    case 'history': {
      if (!args.channel) return 'Error: channel required (ID or #name)';
      const channelId = await resolveChannel(args.channel, config);
      if (!channelId) return `Error: channel not found: ${args.channel}`;

      const limit = Math.min(args.count || 20, 100);
      const params = { channel: channelId, limit };
      if (args.oldest) params.oldest = args.oldest;
      if (args.latest) params.latest = args.latest;

      const data = await slackRequest('GET', '/conversations.history', config, params);
      const messages = data.messages || [];
      if (messages.length === 0) return `No messages in channel.`;

      // Reverse to chronological order
      messages.reverse();

      let output = `Messages from <#${channelId}> (${messages.length}):\n\n`;
      output += messages.map(m => {
        const user = m.user || m.bot_id || 'unknown';
        const time = formatTs(m.ts);
        const thread = m.reply_count ? ` [${m.reply_count} replies]` : '';
        const edited = m.edited ? ' (edited)' : '';
        let text = m.text || '';
        // Truncate long messages
        if (text.length > 500) text = text.slice(0, 500) + '...';
        return `${user} (${time})${thread}${edited}:\n${text}`;
      }).join('\n\n');

      return output;
    }

    case 'send': {
      if (!args.channel) return 'Error: channel required (ID or #name)';
      if (!args.text) return 'Error: text required';
      const channelId = await resolveChannel(args.channel, config);
      if (!channelId) return `Error: channel not found: ${args.channel}`;

      const payload = {
        channel: channelId,
        text: args.text,
      };
      if (args.thread_ts) payload.thread_ts = args.thread_ts;

      const data = await slackRequest('POST', '/chat.postMessage', config, payload);
      const where = args.thread_ts ? ` (in thread ${args.thread_ts})` : '';
      return `Message sent to <#${channelId}>${where}\nTimestamp: ${data.ts}`;
    }

    case 'reply': {
      if (!args.channel) return 'Error: channel required';
      if (!args.thread_ts) return 'Error: thread_ts required (timestamp of parent message)';
      if (!args.text) return 'Error: text required';
      const channelId = await resolveChannel(args.channel, config);
      if (!channelId) return `Error: channel not found: ${args.channel}`;

      const data = await slackRequest('POST', '/chat.postMessage', config, {
        channel: channelId,
        text: args.text,
        thread_ts: args.thread_ts,
      });
      return `Reply sent in thread ${args.thread_ts}\nTimestamp: ${data.ts}`;
    }

    case 'update': {
      if (!args.channel) return 'Error: channel required';
      if (!args.ts) return 'Error: ts required (message timestamp to update)';
      if (!args.text) return 'Error: text required';
      const channelId = await resolveChannel(args.channel, config);
      if (!channelId) return `Error: channel not found: ${args.channel}`;

      await slackRequest('POST', '/chat.update', config, {
        channel: channelId,
        ts: args.ts,
        text: args.text,
      });
      return `Message ${args.ts} updated in <#${channelId}>`;
    }

    case 'react': {
      if (!args.channel) return 'Error: channel required';
      if (!args.ts) return 'Error: ts required (message timestamp)';
      if (!args.emoji && !args.name) return 'Error: emoji/name required (e.g. thumbsup, white_check_mark)';
      const channelId = await resolveChannel(args.channel, config);
      if (!channelId) return `Error: channel not found: ${args.channel}`;

      const emoji = (args.emoji || args.name).replace(/^:|:$/g, '');
      await slackRequest('POST', '/reactions.add', config, {
        channel: channelId,
        timestamp: args.ts,
        name: emoji,
      });
      return `Reaction :${emoji}: added to message ${args.ts}`;
    }

    case 'search': {
      if (!args.query) return 'Error: query required';
      const count = Math.min(args.count || 10, 50);

      const data = await slackRequest('GET', '/search.messages', config, {
        query: args.query,
        count,
        sort: args.sort || 'timestamp',
        sort_dir: 'desc',
      });

      const matches = data.messages?.matches || [];
      if (matches.length === 0) return `No messages found for: ${args.query}`;

      let output = `Found ${data.messages?.total || matches.length} results (showing ${matches.length}):\n\n`;
      output += matches.map(m => {
        const time = formatTs(m.ts);
        const channel = m.channel?.name ? `#${m.channel.name}` : '';
        const user = m.user || m.username || 'unknown';
        let text = m.text || '';
        if (text.length > 300) text = text.slice(0, 300) + '...';
        return `${user} in ${channel} (${time}):\n${text}`;
      }).join('\n\n');
      return output;
    }

    case 'list_users': {
      const limit = Math.min(args.count || 50, 200);
      const params = { limit };
      if (args.cursor) params.cursor = args.cursor;

      const data = await slackRequest('GET', '/users.list', config, params);
      const members = (data.members || []).filter(u => !u.deleted && !u.is_bot);

      if (members.length === 0) return 'No users found.';
      return members.map(u => {
        const status = u.profile?.status_text ? ` — ${u.profile.status_text}` : '';
        const title = u.profile?.title ? ` (${u.profile.title})` : '';
        return `@${u.name}${title} — ${u.real_name || u.profile?.real_name || ''}${status}\n  ID: ${u.id} | TZ: ${u.tz || 'Unknown'}`;
      }).join('\n\n');
    }

    case 'get_user': {
      if (!args.user && !args.id) return 'Error: user or id required';
      let userId = args.user || args.id;

      // If it looks like a @mention or name, look up the user
      if (!userId.startsWith('U') && !userId.startsWith('W')) {
        const name = userId.replace(/^@/, '');
        const list = await slackRequest('GET', '/users.list', config, { limit: 200 });
        const match = (list.members || []).find(u =>
          u.name === name || u.real_name?.toLowerCase() === name.toLowerCase()
        );
        if (!match) return `User not found: ${userId}`;
        userId = match.id;
      }

      const data = await slackRequest('GET', '/users.info', config, { user: userId });
      const u = data.user;
      let output = `@${u.name} — ${u.real_name || ''}\n`;
      output += `ID: ${u.id} | Admin: ${u.is_admin ? 'Yes' : 'No'} | Bot: ${u.is_bot ? 'Yes' : 'No'}\n`;
      if (u.profile?.title) output += `Title: ${u.profile.title}\n`;
      if (u.profile?.email) output += `Email: ${u.profile.email}\n`;
      if (u.profile?.status_text) output += `Status: ${u.profile.status_emoji || ''} ${u.profile.status_text}\n`;
      output += `TZ: ${u.tz || 'Unknown'}`;
      return output;
    }

    default:
      return `Error: unknown slack action '${action}'. Use: list_channels, get_channel, history, send, reply, update, react, search, list_users, get_user, me`;
  }
}

/**
 * Resolve a channel name (e.g. #general or "general") to a channel ID.
 * If already an ID (starts with C/G/D), return as-is.
 */
async function resolveChannel(channel, config) {
  if (!channel) return null;
  channel = channel.replace(/^#/, '');
  // Already an ID
  if (/^[CGD][A-Z0-9]+$/.test(channel)) return channel;

  // Look up by name
  const data = await slackRequest('GET', '/conversations.list', config, {
    limit: 200,
    types: 'public_channel,private_channel',
    exclude_archived: 'true',
  });
  const match = (data.channels || []).find(ch => ch.name === channel);
  return match ? match.id : null;
}

module.exports = { slack };
