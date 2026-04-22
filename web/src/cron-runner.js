// PRE Web GUI — Headless cron job runner
// Executes cron prompts server-side, stores results in dedicated sessions,
// and delivers notifications via macOS, Telegram, and Slack.

const { execSync } = require('child_process');
const { createSession, appendMessage, renameSession, ensureProject, moveSessionToProject } = require('./sessions');
const { runToolLoop } = require('./tools');
const { loadJobs, saveJobs, previousMatchTime } = require('./tools/cron');
const telegramTool = require('./tools/telegram');
const fs = require('fs');
const { CONNECTIONS_FILE } = require('./constants');

const PORT = parseInt(process.env.PRE_WEB_PORT || '7749', 10);

// Max execution time per cron job (5 minutes)
const CRON_TIMEOUT_MS = 5 * 60 * 1000;

function loadConnections() {
  try {
    return JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
  } catch {
    return {};
  }
}

/**
 * Run a cron job headlessly — creates a session, executes the prompt
 * through the full tool loop, collects the response, and sends notifications.
 *
 * @param {Object} job - The cron job object from cron.json
 * @param {Object} opts
 * @param {Function} opts.broadcastWS - Optional: broadcast WS event to connected clients
 * @returns {Promise<{sessionId: string, response: string}>}
 */
async function executeCronJob(job, { broadcastWS } = {}) {
  const ts = Date.now().toString(36);
  const sessionId = createSession('cron', `${job.id}-${ts}`, true);

  // Name the session after the job and group under Cron Jobs project
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
  renameSession(sessionId, `${job.description} — ${dateStr}`);
  ensureProject('scheduled-jobs', 'Scheduled Jobs');
  moveSessionToProject(sessionId, 'scheduled-jobs');

  // Add the cron prompt as a user message
  appendMessage(sessionId, {
    role: 'user',
    content: job.prompt,
  });

  // Collect the assistant's final response
  let finalResponse = '';
  const collectedTokens = [];

  // Abort controller with timeout
  const abortController = new AbortController();
  const timeout = setTimeout(() => abortController.abort(), CRON_TIMEOUT_MS);

  try {
    await runToolLoop({
      sessionId,
      cwd: process.env.HOME || '/tmp',
      signal: abortController.signal,
      userMessage: job.prompt,
      needsTitle: false, // we already set the title
      send: (event) => {
        // Collect response tokens
        if (event.type === 'token' && event.content) {
          collectedTokens.push(event.content);
        }
        if (event.type === 'done') {
          finalResponse = collectedTokens.join('');
        }
        // Forward events to connected WS clients so they can see live progress
        if (broadcastWS) {
          broadcastWS({ ...event, cronSessionId: sessionId, cronJobId: job.id, cronDescription: job.description });
        }
      },
      onConfirmRequest: async () => {
        // Auto-deny confirmation-required tools in headless mode
        console.log(`[cron-runner] Auto-denied confirmation for job ${job.id} — headless mode`);
        return false;
      },
    });
  } catch (err) {
    finalResponse = collectedTokens.join('') || `Error: ${err.message}`;
    console.error(`[cron-runner] Job ${job.id} error: ${err.message}`);
  } finally {
    clearTimeout(timeout);
  }

  if (!finalResponse) {
    finalResponse = collectedTokens.join('') || '(No response generated)';
  }

  console.log(`[cron-runner] Job ${job.id} complete. Session: ${sessionId}, Response: ${finalResponse.slice(0, 200)}...`);

  // Save last session ID to the job record for GUI access
  try {
    const jobs = loadJobs();
    const jobRecord = jobs.find(j => j.id === job.id);
    if (jobRecord) {
      jobRecord.last_session_id = sessionId;
      saveJobs(jobs);
    }
  } catch (err) {
    console.log(`[cron-runner] Failed to save last_session_id: ${err.message}`);
  }

  // Send notifications
  await sendNotifications(job, sessionId, finalResponse);

  // Notify connected GUI clients about the completed cron run
  if (broadcastWS) {
    broadcastWS({
      type: 'cron_complete',
      jobId: job.id,
      sessionId,
      description: job.description,
      preview: finalResponse.slice(0, 300),
    });
  }

  return { sessionId, response: finalResponse };
}

/**
 * Send notifications via all configured channels
 */
async function sendNotifications(job, sessionId, response) {
  const guiUrl = `http://localhost:${PORT}?session=${encodeURIComponent(sessionId)}`;

  // Short preview for macOS notification (limited display space)
  const macPreview = response.length > 500
    ? response.slice(0, 500) + '...'
    : response;

  // 1. macOS notification (always — it's local)
  sendMacNotification(job.description, macPreview, guiUrl);

  // 2. Telegram (if configured) — send full response, chunking handles the 4096 limit
  const conns = loadConnections();
  if (conns.telegram_key) {
    sendTelegramNotification(job, response).catch(err => {
      console.log(`[cron-runner] Telegram notification failed: ${err.message}`);
    });
  }

}

/**
 * macOS notification that opens the GUI when clicked
 * Uses terminal-notifier if available (supports click-to-open),
 * falls back to osascript.
 */
function sendMacNotification(title, message, url) {
  const safeTitle = title.replace(/"/g, '\\"').replace(/'/g, "'");
  // Truncate for notification display
  const shortMsg = message.length > 200 ? message.slice(0, 200) + '...' : message;
  const safeMsg = shortMsg.replace(/"/g, '\\"').replace(/'/g, "'").replace(/\n/g, ' ');

  try {
    // Try terminal-notifier first (supports -open for click action)
    execSync('which terminal-notifier', { stdio: 'pipe' });
    execSync(
      `terminal-notifier -title "PRE: ${safeTitle}" -message "${safeMsg}" -open "${url}" -sound default -group "pre-cron"`,
      { stdio: 'pipe', timeout: 5000 }
    );
    console.log(`[cron-runner] macOS notification sent via terminal-notifier`);
  } catch {
    // Fallback to osascript (no click-to-open, but always available)
    try {
      execSync(
        `osascript -e 'display notification "${safeMsg}" with title "PRE: ${safeTitle}" sound name "default"'`,
        { stdio: 'pipe', timeout: 5000 }
      );
      // Also open the GUI URL since osascript notifications aren't clickable
      execSync(`open "${url}"`, { stdio: 'pipe', timeout: 5000 });
      console.log(`[cron-runner] macOS notification sent via osascript + browser open`);
    } catch (err) {
      console.log(`[cron-runner] macOS notification failed: ${err.message}`);
    }
  }
}

/**
 * Send cron result via Telegram
 */
async function sendTelegramNotification(job, preview) {
  const header = `📋 *PRE Cron: ${job.description}*\n\n`;
  const text = header + preview;
  await telegramTool.telegram({
    action: 'send',
    text,
    parse_mode: 'Markdown',
  });
  console.log(`[cron-runner] Telegram notification sent for job ${job.id}`);
}


/**
 * Check for cron jobs that should have fired while the system was down.
 * For each enabled job, compute the most recent scheduled time. If that
 * time is after the job's last_run_at, the job was missed — fire it now.
 *
 * @param {Object} opts
 * @param {Function} opts.broadcastWS - WebSocket broadcast function
 */
async function checkMissedJobs({ broadcastWS } = {}) {
  const jobs = loadJobs();
  let changed = false;
  const now = Date.now();

  for (const job of jobs) {
    if (!job.enabled) continue;

    const prevMatch = previousMatchTime(job.schedule);
    if (!prevMatch) continue;

    const lastRun = job.last_run_at || 0;

    // Job was missed if the previous scheduled time is after its last run
    if (lastRun < prevMatch.getTime()) {
      const missedAt = prevMatch.toLocaleString('en-US', {
        month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
      });
      console.log(`[cron] Missed job ${job.id}: "${job.description}" (was due ${missedAt})`);

      job.last_run_at = now;
      job.run_count = (job.run_count || 0) + 1;
      changed = true;

      executeCronJob(job, { broadcastWS }).catch(err => {
        console.error(`[cron] Missed job execution error for ${job.id}: ${err.message}`);
      });
    }
  }

  if (changed) saveJobs(jobs);
  return changed;
}

module.exports = { executeCronJob, checkMissedJobs };
