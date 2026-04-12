// PRE Web GUI — Cron job management
// Mirrors the CLI's cron system from pre.m, sharing ~/.pre/cron.json

const fs = require('fs');
const crypto = require('crypto');
const { CRON_FILE } = require('../constants');

/**
 * Load all cron jobs from disk
 * @returns {Array} Array of job objects
 */
function loadJobs() {
  try {
    return JSON.parse(fs.readFileSync(CRON_FILE, 'utf-8'));
  } catch {
    return [];
  }
}

/**
 * Save jobs array to disk
 */
function saveJobs(jobs) {
  fs.writeFileSync(CRON_FILE, JSON.stringify(jobs, null, 2));
}

/**
 * Generate a short random hex ID
 */
function generateId() {
  return crypto.randomBytes(4).toString('hex');
}

// Check if a cron field matches the current value.
// Supports: *, star-slash-N, N, N-M, N,M,O
function fieldMatches(field, value) {
  if (field === '*') return true;
  // Step: */N
  if (field.startsWith('*/')) {
    const step = parseInt(field.slice(2));
    return step > 0 && value % step === 0;
  }
  // List: N,M,O
  const parts = field.split(',');
  for (const part of parts) {
    // Range: N-M
    if (part.includes('-')) {
      const [lo, hi] = part.split('-').map(Number);
      if (value >= lo && value <= hi) return true;
    } else {
      if (parseInt(part) === value) return true;
    }
  }
  return false;
}

/**
 * Check if a job's schedule matches the current time
 * Schedule format: "min hour dom month dow" (5-field cron)
 */
function matchesNow(schedule) {
  const fields = schedule.trim().split(/\s+/);
  if (fields.length !== 5) return false;
  const now = new Date();
  return fieldMatches(fields[0], now.getMinutes())
    && fieldMatches(fields[1], now.getHours())
    && fieldMatches(fields[2], now.getDate())
    && fieldMatches(fields[3], now.getMonth() + 1)
    && fieldMatches(fields[4], now.getDay());
}

/**
 * Tool handler — dispatches cron actions
 */
function cron(args) {
  const action = (args.action || '').toLowerCase();
  const jobs = loadJobs();

  switch (action) {
    case 'add': {
      if (!args.schedule || !args.prompt) {
        return 'Error: "schedule" (5-field cron) and "prompt" are required.';
      }
      // Validate schedule format
      const fields = args.schedule.trim().split(/\s+/);
      if (fields.length !== 5) {
        return 'Error: Schedule must be 5 fields: minute hour day_of_month month day_of_week (e.g. "0 9 * * 1-5")';
      }
      const job = {
        id: generateId(),
        schedule: args.schedule.trim(),
        prompt: args.prompt,
        description: args.description || args.prompt.slice(0, 80),
        enabled: true,
        created_at: Date.now(),
        last_run_at: null,
        run_count: 0,
      };
      jobs.push(job);
      saveJobs(jobs);
      return `Cron job added:\n  ID: ${job.id}\n  Schedule: ${job.schedule}\n  Description: ${job.description}\n  Prompt: ${job.prompt}`;
    }

    case 'list': case 'ls': {
      if (jobs.length === 0) return 'No cron jobs configured.';
      const lines = jobs.map(j => {
        const status = j.enabled ? '✓' : '✗';
        const lastRun = j.last_run_at ? new Date(j.last_run_at).toLocaleString() : 'never';
        return `[${status}] ${j.id}  ${j.schedule}  "${j.description}"  (runs: ${j.run_count}, last: ${lastRun})`;
      });
      return `${jobs.length} cron job(s):\n${lines.join('\n')}`;
    }

    case 'remove': case 'rm': case 'delete': {
      if (!args.id) return 'Error: "id" is required for remove.';
      const idx = jobs.findIndex(j => j.id === args.id);
      if (idx === -1) return `Error: No job with ID "${args.id}".`;
      const removed = jobs.splice(idx, 1)[0];
      saveJobs(jobs);
      return `Removed cron job ${removed.id}: "${removed.description}"`;
    }

    case 'enable': {
      if (!args.id) return 'Error: "id" is required for enable.';
      const job = jobs.find(j => j.id === args.id);
      if (!job) return `Error: No job with ID "${args.id}".`;
      job.enabled = true;
      saveJobs(jobs);
      return `Enabled cron job ${job.id}: "${job.description}"`;
    }

    case 'disable': {
      if (!args.id) return 'Error: "id" is required for disable.';
      const job = jobs.find(j => j.id === args.id);
      if (!job) return `Error: No job with ID "${args.id}".`;
      job.enabled = false;
      saveJobs(jobs);
      return `Disabled cron job ${job.id}: "${job.description}"`;
    }

    default:
      return 'Error: Unknown cron action. Use: add, list, remove, enable, disable';
  }
}

/**
 * Find the most recent time before `before` that matches the cron schedule.
 * Walks backwards minute by minute, up to 7 days.
 * @param {string} schedule - 5-field cron expression
 * @param {Date} [before] - Start searching before this time (default: now)
 * @returns {Date|null} The most recent matching time, or null
 */
function previousMatchTime(schedule, before) {
  const fields = schedule.trim().split(/\s+/);
  if (fields.length !== 5) return null;

  const check = new Date(before || Date.now());
  check.setSeconds(0, 0);
  check.setMinutes(check.getMinutes() - 1); // start one minute before

  const MAX_LOOKBACK = 7 * 24 * 60; // 7 days in minutes
  for (let i = 0; i < MAX_LOOKBACK; i++) {
    if (fieldMatches(fields[0], check.getMinutes())
      && fieldMatches(fields[1], check.getHours())
      && fieldMatches(fields[2], check.getDate())
      && fieldMatches(fields[3], check.getMonth() + 1)
      && fieldMatches(fields[4], check.getDay())) {
      return new Date(check);
    }
    check.setMinutes(check.getMinutes() - 1);
  }
  return null;
}

module.exports = { cron, loadJobs, saveJobs, matchesNow, previousMatchTime, generateId };
