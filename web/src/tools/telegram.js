// PRE Web GUI — Telegram Bot tool
// Uses Bot API to send messages, photos, and check updates

const https = require('https');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('../constants');

function loadConnections() {
  try {
    return JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
  } catch {
    return {};
  }
}

function saveConnections(data) {
  fs.writeFileSync(CONNECTIONS_FILE, JSON.stringify(data, null, 2), { mode: 0o600 });
}

const TG_MAX_LENGTH = 4096;

/**
 * Split a message into chunks that fit within Telegram's 4096-char limit.
 * Splits at newlines first, then spaces, to avoid breaking mid-word.
 */
function chunkMessage(text) {
  if (text.length <= TG_MAX_LENGTH) return [text];

  const chunks = [];
  let remaining = text;
  while (remaining.length > 0) {
    if (remaining.length <= TG_MAX_LENGTH) {
      chunks.push(remaining);
      break;
    }
    // Find the best split point within the limit
    let splitAt = remaining.lastIndexOf('\n', TG_MAX_LENGTH);
    if (splitAt < TG_MAX_LENGTH * 0.3) {
      // Newline too far back — try a space
      splitAt = remaining.lastIndexOf(' ', TG_MAX_LENGTH);
    }
    if (splitAt < TG_MAX_LENGTH * 0.3) {
      // No good split point — hard cut
      splitAt = TG_MAX_LENGTH;
    }
    chunks.push(remaining.slice(0, splitAt));
    remaining = remaining.slice(splitAt).replace(/^\n/, ''); // trim leading newline from next chunk
  }
  return chunks;
}

function getBotToken() {
  return loadConnections().telegram_key || null;
}

function getSavedChatId() {
  // Check connections.json first, then the CLI's telegram_owner file
  const data = loadConnections();
  if (data.telegram_chat_id) return data.telegram_chat_id;
  try {
    const ownerFile = require('path').join(require('os').homedir(), '.pre', 'telegram_owner');
    const ownerId = require('fs').readFileSync(ownerFile, 'utf-8').trim();
    if (ownerId) return ownerId;
  } catch {}
  return null;
}

function saveChatId(chatId) {
  const data = loadConnections();
  data.telegram_chat_id = String(chatId);
  saveConnections(data);
}

function botRequest(method, params) {
  return new Promise((resolve, reject) => {
    const token = getBotToken();
    if (!token) return reject(new Error('Telegram not configured. Use Settings to add your bot token.'));

    const postData = JSON.stringify(params || {});
    const req = https.request({
      hostname: 'api.telegram.org',
      path: `/bot${token}/${method}`,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData),
      },
    }, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try {
          const data = JSON.parse(body);
          if (!data.ok) return reject(new Error(data.description || 'Telegram API error'));
          resolve(data.result);
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

async function telegram(args) {
  const action = args.action;
  if (!action) return 'Error: action required (send|get_me|get_updates|get_chat)';

  switch (action) {
    case 'get_me': {
      const me = await botRequest('getMe');
      return `Bot: @${me.username}\nName: ${me.first_name}\nID: ${me.id}`;
    }

    case 'get_updates': {
      const limit = Math.min(args.count || 10, 100);
      const updates = await botRequest('getUpdates', { limit });
      if (!updates || updates.length === 0) return 'No recent updates. (The pre-telegram bot process may be consuming updates.)';
      const msgs = updates
        .filter(u => u.message)
        .map(u => {
          const m = u.message;
          const from = m.from ? `${m.from.first_name || ''} ${m.from.last_name || ''}`.trim() : 'unknown';
          // Save chat_id for future sends
          if (m.chat?.id) saveChatId(m.chat.id);
          return `[chat_id: ${m.chat.id}] ${from}: ${m.text || '(non-text message)'}`;
        });
      return msgs.length > 0 ? msgs.join('\n') : 'No text messages in recent updates.';
    }

    case 'get_chat': {
      if (!args.chat_id) return 'Error: chat_id required';
      const chat = await botRequest('getChat', { chat_id: args.chat_id });
      const type = chat.type;
      const title = chat.title || `${chat.first_name || ''} ${chat.last_name || ''}`.trim();
      return `Chat: ${title}\nType: ${type}\nID: ${chat.id}`;
    }

    case 'send': {
      if (!args.chat_id) {
        // Try saved chat_id first, then fall back to getUpdates
        const saved = getSavedChatId();
        if (saved) {
          args.chat_id = saved;
        } else {
          try {
            const updates = await botRequest('getUpdates', { limit: 5 });
            const lastMsg = updates.reverse().find(u => u.message);
            if (lastMsg) {
              args.chat_id = lastMsg.message.chat.id;
              saveChatId(args.chat_id);
            } else {
              return 'Error: chat_id not known. Send a message to the bot first so I can learn your chat ID, or provide chat_id explicitly.';
            }
          } catch {
            return 'Error: chat_id not known and could not auto-detect. Provide chat_id explicitly.';
          }
        }
      } else {
        // Save provided chat_id for future use
        saveChatId(args.chat_id);
      }
      if (!args.text && !args.message) return 'Error: text required for send action';
      const text = args.text || args.message;
      const parseMode = args.parse_mode || 'Markdown';
      const chunks = chunkMessage(text);
      let lastResult;
      for (const chunk of chunks) {
        lastResult = await botRequest('sendMessage', {
          chat_id: args.chat_id,
          text: chunk,
          parse_mode: parseMode,
        });
      }
      const plural = chunks.length > 1 ? ` (${chunks.length} messages)` : '';
      return `Message sent to chat ${lastResult.chat.id}${plural}! Message ID: ${lastResult.message_id}`;
    }

    default:
      return `Error: unknown telegram action '${action}'. Use: send, get_me, get_updates, get_chat`;
  }
}

module.exports = { telegram };
