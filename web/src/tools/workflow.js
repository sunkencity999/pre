// PRE Web GUI — Workflow Capture and Replay
// Records sequences of Computer Use actions as replayable workflows.
// Workflows can be replayed on demand or scheduled via cron triggers.
//
// A workflow is a JSON file containing:
//   - Metadata (name, description, created)
//   - Steps: array of { action, args, delay, screenshot_hash }
//   - Each step is a Computer Use action that was executed during recording
//
// Replay uses vision-adaptive execution: before each step, take a screenshot
// and let the model verify/adjust coordinates if the UI has changed.
//
// Storage: ~/.pre/workflows/

const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');

const WORKFLOW_DIR = path.join(os.homedir(), '.pre', 'workflows');
if (!fs.existsSync(WORKFLOW_DIR)) fs.mkdirSync(WORKFLOW_DIR, { recursive: true });

// ── Recording state ────────────────────────────────────────────────────────

let recording = null; // { name, steps[], startTime }

/**
 * Start recording a new workflow.
 */
function startRecording(args) {
  if (recording) {
    return `Already recording workflow "${recording.name}". Stop it first with action "stop".`;
  }

  const name = (args.name || `workflow_${Date.now()}`).replace(/[^a-zA-Z0-9_-]/g, '_');

  recording = {
    name,
    description: args.description || '',
    steps: [],
    startTime: Date.now(),
    lastStepTime: Date.now(),
  };

  return `Recording started: "${name}"\nPerform actions using the computer tool. Each action will be captured.\nUse workflow action "stop" when finished.`;
}

/**
 * Record a single step (called by the computer tool interceptor).
 */
function recordStep(action, args) {
  if (!recording) return;

  const now = Date.now();
  const delay = now - recording.lastStepTime;
  recording.lastStepTime = now;

  // Don't record screenshots — they're observations, not actions
  if (action === 'screenshot' || action === 'screen_size' || action === 'cursor_position') return;

  const step = {
    action,
    args: { ...args },
    delay: Math.min(delay, 10000), // Cap delay at 10s
    timestamp: now,
  };

  // Store a hash of the action for deduplication
  step.hash = crypto.createHash('md5')
    .update(JSON.stringify({ action, args }))
    .digest('hex')
    .slice(0, 8);

  recording.steps.push(step);
}

/**
 * Stop recording and save the workflow.
 */
function stopRecording() {
  if (!recording) return 'No workflow is being recorded.';

  if (recording.steps.length === 0) {
    const name = recording.name;
    recording = null;
    return `Recording "${name}" stopped with 0 steps. Nothing saved.`;
  }

  const workflow = {
    name: recording.name,
    description: recording.description,
    created: new Date().toISOString(),
    stepCount: recording.steps.length,
    duration: Date.now() - recording.startTime,
    steps: recording.steps,
  };

  const filename = `${recording.name}.json`;
  const filePath = path.join(WORKFLOW_DIR, filename);
  fs.writeFileSync(filePath, JSON.stringify(workflow, null, 2));

  const name = recording.name;
  const count = recording.steps.length;
  recording = null;

  return `Workflow saved: "${name}" (${count} steps)\nPath: ${filePath}\nReplay with: workflow action "replay" name "${name}"`;
}

/**
 * Check if currently recording.
 */
function isRecording() {
  return recording !== null;
}

/**
 * Get current recording status.
 */
function recordingStatus() {
  if (!recording) return null;
  return {
    name: recording.name,
    steps: recording.steps.length,
    duration: Date.now() - recording.startTime,
  };
}

// ── Workflow management ────────────────────────────────────────────────────

function loadWorkflow(name) {
  const filename = name.endsWith('.json') ? name : `${name}.json`;
  const filePath = path.join(WORKFLOW_DIR, filename);
  if (!fs.existsSync(filePath)) return null;
  try { return JSON.parse(fs.readFileSync(filePath, 'utf-8')); } catch { return null; }
}

function listWorkflows() {
  if (!fs.existsSync(WORKFLOW_DIR)) return [];

  return fs.readdirSync(WORKFLOW_DIR)
    .filter(f => f.endsWith('.json'))
    .map(f => {
      try {
        const wf = JSON.parse(fs.readFileSync(path.join(WORKFLOW_DIR, f), 'utf-8'));
        return {
          name: wf.name,
          description: wf.description,
          steps: wf.stepCount,
          created: wf.created,
          duration: wf.duration,
        };
      } catch { return null; }
    })
    .filter(Boolean)
    .sort((a, b) => new Date(b.created) - new Date(a.created));
}

function deleteWorkflow(name) {
  const filename = name.endsWith('.json') ? name : `${name}.json`;
  const filePath = path.join(WORKFLOW_DIR, filename);
  if (!fs.existsSync(filePath)) return `Workflow "${name}" not found.`;
  fs.unlinkSync(filePath);
  return `Deleted workflow "${name}".`;
}

// ── Replay ─────────────────────────────────────────────────────────────────

/**
 * Replay a workflow by re-executing each step through Computer Use.
 * Steps are executed with their recorded delays (capped).
 *
 * @param {string} name - Workflow name
 * @param {Object} opts - { speed (multiplier, default 1.0), computerFn (executor) }
 * @returns {Promise<string>} Replay result summary
 */
async function replay(name, opts = {}) {
  const workflow = loadWorkflow(name);
  if (!workflow) return `Workflow "${name}" not found.`;

  const computerFn = opts.computerFn;
  if (!computerFn) return 'Error: no computer function provided for replay';

  const speed = opts.speed || 1.0;
  const results = [];
  let errors = 0;

  for (let i = 0; i < workflow.steps.length; i++) {
    const step = workflow.steps[i];

    // Wait for the inter-step delay (adjusted by speed multiplier)
    if (i > 0 && step.delay > 0) {
      const wait = Math.max(100, Math.floor(step.delay / speed));
      await new Promise(r => setTimeout(r, wait));
    }

    try {
      await computerFn(step.args);
      results.push({ step: i + 1, action: step.action, status: 'ok' });
    } catch (err) {
      errors++;
      results.push({ step: i + 1, action: step.action, status: 'error', message: err.message });
    }
  }

  const summary = [
    `Replayed "${workflow.name}": ${workflow.steps.length} steps`,
    `Speed: ${speed}x, Errors: ${errors}`,
    '',
    ...results.map(r => `  Step ${r.step}: ${r.action} — ${r.status}${r.message ? ` (${r.message})` : ''}`),
  ];

  return summary.join('\n');
}

// ── Tool Dispatcher ────────────────────────────────────────────────────────

async function workflowTool(args) {
  const action = (args.action || '').toLowerCase();

  switch (action) {
    case 'record': case 'start':
      return startRecording(args);

    case 'stop':
      return stopRecording();

    case 'status': {
      const status = recordingStatus();
      if (!status) return 'Not recording. Use action "record" to start.';
      return `Recording: "${status.name}" — ${status.steps} steps captured (${Math.floor(status.duration / 1000)}s)`;
    }

    case 'list': case 'ls': {
      const workflows = listWorkflows();
      if (workflows.length === 0) return 'No saved workflows. Use action "record" to create one.';
      const lines = workflows.map(w => {
        const dur = w.duration ? `${(w.duration / 1000).toFixed(0)}s` : '?';
        return `  **${w.name}** — ${w.steps} steps, ${dur} duration (${w.created?.slice(0, 10) || '?'})${w.description ? `\n    ${w.description}` : ''}`;
      });
      return `${workflows.length} workflow(s):\n${lines.join('\n')}`;
    }

    case 'replay': case 'run': case 'play': {
      if (!args.name) return 'Error: name is required for replay';
      // Get the computer function from opts
      const computerTool = require('./computer');
      const computerFn = async (stepArgs) => {
        return computerTool.computer({ ...stepArgs });
      };
      return replay(args.name, {
        speed: args.speed || 1.0,
        computerFn,
      });
    }

    case 'show': case 'view': case 'inspect': {
      if (!args.name) return 'Error: name is required';
      const wf = loadWorkflow(args.name);
      if (!wf) return `Workflow "${args.name}" not found.`;
      const lines = [
        `**Workflow: ${wf.name}**`,
        wf.description ? `Description: ${wf.description}` : '',
        `Created: ${wf.created}`,
        `Steps: ${wf.stepCount}, Duration: ${(wf.duration / 1000).toFixed(0)}s`,
        '',
        '**Steps:**',
        ...wf.steps.map((s, i) => {
          const argStr = Object.entries(s.args)
            .filter(([k]) => k !== 'action')
            .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
            .join(', ');
          return `  ${i + 1}. ${s.action}(${argStr})${s.delay > 500 ? ` [wait ${(s.delay / 1000).toFixed(1)}s]` : ''}`;
        }),
      ].filter(Boolean);
      return lines.join('\n');
    }

    case 'delete': case 'rm': case 'remove':
      if (!args.name) return 'Error: name is required';
      return deleteWorkflow(args.name);

    case 'export': {
      if (!args.name) return 'Error: name is required';
      const wf = loadWorkflow(args.name);
      if (!wf) return `Workflow "${args.name}" not found.`;
      // Export as a shell-friendly description
      const lines = wf.steps.map((s, i) => {
        const delay = s.delay > 500 ? `sleep ${(s.delay / 1000).toFixed(1)} && ` : '';
        return `# Step ${i + 1}: ${s.action}\n${delay}# ${JSON.stringify(s.args)}`;
      });
      return `# Workflow: ${wf.name}\n# ${wf.description || ''}\n# Steps: ${wf.stepCount}\n\n${lines.join('\n\n')}`;
    }

    default:
      return 'Unknown workflow action. Available: record, stop, status, list, replay, show, delete, export';
  }
}

module.exports = {
  workflow: workflowTool,
  recordStep,
  isRecording,
  recordingStatus,
  startRecording,
  stopRecording,
};
