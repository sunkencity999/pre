// PRE Web GUI — shared constants
// Must match Modelfile values exactly to avoid Ollama model reload

const path = require('path');
const os = require('os');

const PRE_DIR = path.join(os.homedir(), '.pre');
const MODEL = 'pre-gemma4';
const MODEL_CTX = 65536;
const OLLAMA_PORT = parseInt(process.env.PRE_PORT || '11434', 10);
const MAX_TOOL_TURNS = 35;

// Session files live in ~/.pre/sessions/<project>:<channel>.jsonl
const SESSIONS_DIR = path.join(PRE_DIR, 'sessions');
const MEMORY_DIR = path.join(PRE_DIR, 'memory');
const ARTIFACTS_DIR = path.join(PRE_DIR, 'artifacts');
const CONNECTIONS_FILE = path.join(PRE_DIR, 'connections.json');
const COMFYUI_FILE = path.join(PRE_DIR, 'comfyui.json');
const CRON_FILE = path.join(PRE_DIR, 'cron.json');

// Deep Research mode constants
const RESEARCH_AGENT_MAX_TURNS = 20;   // Each research sub-agent gets more turns than normal
const RESEARCH_MAX_SECTIONS = 8;       // Max sections in a research outline
const RESEARCH_MAX_SOURCES = 5;        // Sources to gather per section

module.exports = {
  PRE_DIR,
  MODEL,
  MODEL_CTX,
  OLLAMA_PORT,
  MAX_TOOL_TURNS,
  SESSIONS_DIR,
  MEMORY_DIR,
  ARTIFACTS_DIR,
  CONNECTIONS_FILE,
  COMFYUI_FILE,
  CRON_FILE,
  RESEARCH_AGENT_MAX_TURNS,
  RESEARCH_MAX_SECTIONS,
  RESEARCH_MAX_SOURCES,
};
