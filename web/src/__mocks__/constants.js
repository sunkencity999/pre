// Mock constants module for tests — uses /tmp/pre-test-env/ as base
const path = require('path');
const fs = require('fs');

// Each test worker gets a unique dir based on its worker ID
const workerId = process.env.JEST_WORKER_ID || '1';
const PRE_DIR = `/tmp/pre-test-env-${workerId}`;

// Ensure directories exist
const SESSIONS_DIR = path.join(PRE_DIR, 'sessions');
const MEMORY_DIR = path.join(PRE_DIR, 'memory');
const ARTIFACTS_DIR = path.join(PRE_DIR, 'artifacts');
const CONNECTIONS_FILE = path.join(PRE_DIR, 'connections.json');
const COMFYUI_FILE = path.join(PRE_DIR, 'comfyui.json');
const CRON_FILE = path.join(PRE_DIR, 'cron.json');

fs.mkdirSync(SESSIONS_DIR, { recursive: true });
fs.mkdirSync(MEMORY_DIR, { recursive: true });
fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
fs.mkdirSync(path.join(MEMORY_DIR, 'experience'), { recursive: true });

if (!fs.existsSync(CONNECTIONS_FILE)) fs.writeFileSync(CONNECTIONS_FILE, '{}');
if (!fs.existsSync(CRON_FILE)) fs.writeFileSync(CRON_FILE, JSON.stringify({ jobs: [] }));
if (!fs.existsSync(path.join(PRE_DIR, 'triggers.json'))) fs.writeFileSync(path.join(PRE_DIR, 'triggers.json'), '[]');

module.exports = {
  PRE_DIR,
  MODEL: 'test-model',
  MODEL_CTX: 8192,
  OLLAMA_PORT: 11434,
  MAX_TOOL_TURNS: 35,
  SESSIONS_DIR,
  MEMORY_DIR,
  ARTIFACTS_DIR,
  CONNECTIONS_FILE,
  COMFYUI_FILE,
  CRON_FILE,
  RESEARCH_AGENT_MAX_TURNS: 30,
  RESEARCH_MAX_SECTIONS: 8,
  RESEARCH_MAX_SOURCES: 5,
};
