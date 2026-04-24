// PRE Web GUI — Telegram Background Receiver
// Long-polls Telegram's getUpdates API for incoming messages, routes them
// through the cron-runner execution pipeline, and sends responses back.
// Initialized from server.js alongside triggers and cron.

const https = require('https');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('./constants');

// Polling state
let polling = false;
let pollTimeout = null;
let lastOffset = 0;
let executeFn = null;
let botUsername = null;

// Telegram long-poll timeout (seconds) — Telegram holds the connection open
const LONG_POLL_TIMEOUT = 30;

// Delay between poll cycles (ms) — brief pause to avoid tight loops on errors
const POLL_INTERVAL = 1000;

// Max message age to process (5 minutes) — skip stale messages from before boot
const MAX_MESSAGE_AGE_S = 300;

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

function getBotToken() {
  return loadConnections().telegram_key || null;
}

function saveChatId(chatId) {
  const data = loadConnections();
  if (data.telegram_chat_id !== String(chatId)) {
    data.telegram_chat_id = String(chatId);
    saveConnections(data);
  }
}

/**
 * Make a Telegram Bot API request.
 */
function botRequest(token, method, params) {
  return new Promise((resolve, reject) => {
    const postData = JSON.stringify(params || {});
    const req = https.request({
      hostname: 'api.telegram.org',
      path: `/bot${token}/${method}`,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData),
      },
      timeout: (LONG_POLL_TIMEOUT + 10) * 1000,
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
    req.on('timeout', () => { req.destroy(); reject(new Error('Request timeout')); });
    req.write(postData);
    req.end();
  });
}

const TG_MAX_LENGTH = 4096;

/**
 * Split a message into chunks that fit within Telegram's 4096-char limit.
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
    let splitAt = remaining.lastIndexOf('\n', TG_MAX_LENGTH);
    if (splitAt < TG_MAX_LENGTH * 0.3) {
      splitAt = remaining.lastIndexOf(' ', TG_MAX_LENGTH);
    }
    if (splitAt < TG_MAX_LENGTH * 0.3) {
      splitAt = TG_MAX_LENGTH;
    }
    chunks.push(remaining.slice(0, splitAt));
    remaining = remaining.slice(splitAt).replace(/^\n/, '');
  }
  return chunks;
}

/**
 * Send a reply back to a Telegram chat.
 */
async function sendReply(token, chatId, text) {
  const chunks = chunkMessage(text);
  for (const chunk of chunks) {
    try {
      await botRequest(token, 'sendMessage', {
        chat_id: chatId,
        text: chunk,
        parse_mode: 'Markdown',
      });
    } catch (err) {
      // Retry without parse_mode if Markdown fails
      if (err.message && err.message.includes("can't parse")) {
        await botRequest(token, 'sendMessage', {
          chat_id: chatId,
          text: chunk,
        });
      } else {
        throw err;
      }
    }
  }
}

/**
 * Process a single incoming Telegram message.
 * Creates a session, runs the tool loop, sends the response back.
 */
async function processMessage(token, message) {
  const chatId = message.chat.id;
  const text = message.text;
  const from = message.from
    ? `${message.from.first_name || ''} ${message.from.last_name || ''}`.trim()
    : 'Unknown';

  if (!text) return; // Skip non-text messages (photos, stickers, etc.)

  // Save chat_id for future sends
  saveChatId(chatId);

  console.log(`[telegram] Message from ${from} (${chatId}): ${text.slice(0, 80)}${text.length > 80 ? '...' : ''}`);

  // Send typing indicator while processing
  botRequest(token, 'sendChatAction', { chat_id: chatId, action: 'typing' }).catch(() => {});

  try {
    // Run through the same execution pipeline as cron/triggers
    const result = await executeFn({
      id: `tg-${chatId}`,
      description: `Telegram: ${from}`,
      prompt: text,
      cwd: process.env.HOME || '/tmp',
      fromTelegram: true,
    });

    if (result && result.response) {
      await sendReply(token, chatId, result.response);
    } else {
      await sendReply(token, chatId, '(No response generated)');
    }
  } catch (err) {
    console.error(`[telegram] Processing error: ${err.message}`);
    try {
      await sendReply(token, chatId, `Error processing your message: ${err.message}`);
    } catch {}
  }
}

/**
 * Single poll cycle — fetch updates and process them.
 */
async function pollOnce(token) {
  const params = {
    timeout: LONG_POLL_TIMEOUT,
    allowed_updates: ['message'],
  };
  if (lastOffset > 0) {
    params.offset = lastOffset;
  }

  const updates = await botRequest(token, 'getUpdates', params);

  if (!updates || updates.length === 0) return;

  const now = Math.floor(Date.now() / 1000);

  for (const update of updates) {
    // Advance offset to acknowledge this update (Telegram won't re-deliver it)
    lastOffset = update.update_id + 1;

    const message = update.message;
    if (!message || !message.text) continue;

    // Skip messages older than MAX_MESSAGE_AGE_S (stale from before server boot)
    if (now - message.date > MAX_MESSAGE_AGE_S) {
      console.log(`[telegram] Skipping stale message (${now - message.date}s old): "${message.text.slice(0, 40)}..."`);
      continue;
    }

    // Skip bot's own messages (shouldn't happen, but be safe)
    if (message.from?.is_bot) continue;

    // Process asynchronously — don't block the poll loop
    processMessage(token, message).catch(err => {
      console.error(`[telegram] Failed to process message ${message.message_id}: ${err.message}`);
    });
  }
}

/**
 * Main polling loop.
 */
async function pollLoop() {
  if (!polling) return;

  const token = getBotToken();
  if (!token) {
    console.log('[telegram] No bot token configured — receiver stopped');
    polling = false;
    return;
  }

  try {
    await pollOnce(token);
  } catch (err) {
    // Don't log timeout errors — they're normal for long polling
    if (!err.message?.includes('timeout') && !err.message?.includes('ECONNRESET')) {
      console.error(`[telegram] Poll error: ${err.message}`);
    }
  }

  // Schedule next poll
  if (polling) {
    pollTimeout = setTimeout(pollLoop, POLL_INTERVAL);
  }
}

/**
 * Initialize the Telegram receiver.
 * Call once from server.js after the execution pipeline is ready.
 *
 * @param {Function} execute — async (job) => { sessionId, response }
 * @param {Function} broadcast — WebSocket broadcast function
 * @returns {{ active: boolean, username: string|null }}
 */
async function init(execute) {
  executeFn = execute;

  const token = getBotToken();
  if (!token) {
    return { active: false, username: null };
  }

  // Verify bot token and get username
  try {
    const me = await botRequest(token, 'getMe', {});
    botUsername = me.username;
  } catch (err) {
    console.error(`[telegram] Bot verification failed: ${err.message}`);
    return { active: false, username: null, error: err.message };
  }

  // Flush stale updates by fetching with offset -1, then advancing past them
  try {
    const stale = await botRequest(token, 'getUpdates', { offset: -1, limit: 1, timeout: 0 });
    if (stale && stale.length > 0) {
      lastOffset = stale[stale.length - 1].update_id + 1;
      console.log(`[telegram] Flushed ${stale.length} stale update(s), starting from offset ${lastOffset}`);
    }
  } catch {}

  // Start polling
  polling = true;
  pollLoop();

  return { active: true, username: botUsername };
}

/**
 * Stop the receiver (for graceful shutdown).
 */
function shutdown() {
  polling = false;
  if (pollTimeout) {
    clearTimeout(pollTimeout);
    pollTimeout = null;
  }
  console.log('[telegram] Receiver stopped');
}

/**
 * Check if the receiver is actively polling.
 */
function isActive() {
  return polling;
}

/**
 * Get receiver status.
 */
function getStatus() {
  return {
    active: polling,
    username: botUsername,
    lastOffset,
  };
}

module.exports = { init, shutdown, isActive, getStatus };
