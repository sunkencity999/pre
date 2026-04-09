// PRE Web GUI — Image generation via ComfyUI
// Mirrors the CLI's comfyui_generate() from pre.m

const fs = require('fs');
const path = require('path');
const http = require('http');
const { execSync, spawn } = require('child_process');
const { PRE_DIR, ARTIFACTS_DIR } = require('../constants');

const COMFYUI_FILE = path.join(PRE_DIR, 'comfyui.json');

/** Load ComfyUI configuration from ~/.pre/comfyui.json */
function loadConfig() {
  try {
    return JSON.parse(fs.readFileSync(COMFYUI_FILE, 'utf-8'));
  } catch {
    return null;
  }
}

/** Check if ComfyUI is reachable on the configured port */
async function isRunning(port) {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${port}/system_stats`, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(3000, () => { req.destroy(); resolve(false); });
  });
}

/** Start ComfyUI as a background process and wait for it to be ready */
async function startComfyUI(config) {
  const port = config.port || 8188;
  if (await isRunning(port)) return;

  const home = process.env.HOME;
  const venvPython = path.join(home, '.pre/comfyui-venv/bin/python3');
  const comfyuiMain = path.join(home, '.pre/comfyui/main.py');
  const logPath = path.join(PRE_DIR, 'comfyui.log');

  if (!fs.existsSync(venvPython) || !fs.existsSync(comfyuiMain)) {
    throw new Error('ComfyUI files not found — run the PRE installer to set up image generation');
  }

  console.log(`[comfyui] Starting on port ${port}...`);

  const child = spawn(venvPython, [
    comfyuiMain, '--listen', '127.0.0.1', '--port', String(port), '--force-fp16',
  ], {
    detached: true,
    stdio: ['ignore', fs.openSync(logPath, 'w'), fs.openSync(logPath, 'a')],
  });
  child.unref();

  // Poll until ready (max 60s — first launch loads model into GPU)
  for (let i = 0; i < 120; i++) {
    await new Promise(r => setTimeout(r, 500));
    if (await isRunning(port)) {
      console.log(`[comfyui] Ready after ${Math.ceil((i + 1) / 2)}s`);
      return;
    }
  }
  throw new Error('ComfyUI failed to start within 60s — check ~/.pre/comfyui.log');
}

/** HTTP POST JSON to ComfyUI */
function postJSON(port, endpoint, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = http.request({
      hostname: '127.0.0.1',
      port,
      path: endpoint,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
    }, (res) => {
      let buf = '';
      res.on('data', (c) => buf += c);
      res.on('end', () => {
        try { resolve(JSON.parse(buf)); }
        catch { reject(new Error(`Invalid JSON from ComfyUI: ${buf.slice(0, 200)}`)); }
      });
    });
    req.on('error', reject);
    req.setTimeout(10000, () => { req.destroy(); reject(new Error('ComfyUI POST timeout')); });
    req.write(data);
    req.end();
  });
}

/** HTTP GET from ComfyUI */
function getJSON(port, endpoint) {
  return new Promise((resolve, reject) => {
    const req = http.get(`http://127.0.0.1:${port}${endpoint}`, (res) => {
      let buf = '';
      res.on('data', (c) => buf += c);
      res.on('end', () => {
        try { resolve(JSON.parse(buf)); }
        catch { resolve(null); }
      });
    });
    req.on('error', reject);
    req.setTimeout(5000, () => { req.destroy(); reject(new Error('ComfyUI GET timeout')); });
  });
}

/**
 * Generate an image via ComfyUI.
 * @param {Object} args - { prompt, width?, height?, style?, negative? }
 * @returns {string} Result message with file path and web URL
 */
async function imageGenerate(args) {
  const config = loadConfig();
  if (!config || !config.installed) {
    throw new Error('ComfyUI is not installed. Run the PRE installer to set up local image generation.');
  }

  const port = config.port || 8188;
  const checkpoint = config.checkpoint || 'sd_xl_turbo_1.0_fp16.safetensors';

  // Ensure ComfyUI is running
  await startComfyUI(config);

  // Build prompt with optional style prefix
  let prompt = args.prompt || '';
  if (args.style) prompt = `${args.style}, ${prompt}`;

  // Detect model type — Turbo vs full SDXL
  const isTurbo = checkpoint.toLowerCase().includes('turbo');

  let width = parseInt(args.width) || (isTurbo ? 512 : 1024);
  let height = parseInt(args.height) || (isTurbo ? 512 : 1024);
  const maxDim = isTurbo ? 1024 : 1536;
  width = Math.min(width, maxDim);
  height = Math.min(height, maxDim);

  // Default negative prompt
  const negative = args.negative ||
    'blurry, low quality, distorted, deformed, disfigured, bad anatomy, wrong proportions, ' +
    'extra limbs, mutated hands, ugly face, distorted face, malformed face, crossed eyes, ' +
    'watermark, text, signature, jpeg artifacts';

  // Workflow parameters adapt to checkpoint
  const steps = isTurbo ? 4 : 25;
  const cfg = isTurbo ? 1.0 : 5.5;
  const sampler = isTurbo ? 'euler_ancestral' : 'dpmpp_2m';
  const scheduler = isTurbo ? 'normal' : 'karras';
  const seed = Math.floor(Math.random() * 2147483647);
  const prefix = `pre_${Date.now()}`;

  // Build ComfyUI workflow
  const workflow = {
    '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: checkpoint } },
    '5': { class_type: 'EmptyLatentImage', inputs: { width, height, batch_size: 1 } },
    '6': { class_type: 'CLIPTextEncode', inputs: { text: prompt, clip: ['4', 1] } },
    '7': { class_type: 'CLIPTextEncode', inputs: { text: negative, clip: ['4', 1] } },
    '3': {
      class_type: 'KSampler',
      inputs: {
        seed, steps, cfg, sampler_name: sampler, scheduler, denoise: 1.0,
        model: ['4', 0], positive: ['6', 0], negative: ['7', 0], latent_image: ['5', 0],
      },
    },
    '8': { class_type: 'VAEDecode', inputs: { samples: ['3', 0], vae: ['4', 2] } },
    '9': { class_type: 'SaveImage', inputs: { filename_prefix: prefix, images: ['8', 0] } },
  };

  // Submit prompt to ComfyUI
  console.log(`[comfyui] Generating ${width}x${height}, ${steps} steps, seed ${seed}`);
  const submitResult = await postJSON(port, '/prompt', { prompt: workflow });
  const promptId = submitResult.prompt_id;
  if (!promptId) {
    throw new Error(`ComfyUI rejected prompt: ${JSON.stringify(submitResult).slice(0, 300)}`);
  }

  // Poll /history/{prompt_id} until complete (max 300s)
  let filename = null;
  for (let poll = 0; poll < 600; poll++) {
    await new Promise(r => setTimeout(r, 500));

    try {
      const history = await getJSON(port, `/history/${promptId}`);
      if (!history || !history[promptId]) continue;

      // Extract filename from outputs
      const outputs = history[promptId].outputs;
      if (outputs && outputs['9'] && outputs['9'].images && outputs['9'].images.length > 0) {
        filename = outputs['9'].images[0].filename;
        break;
      }
    } catch {
      // Polling error — keep trying
    }
  }

  if (!filename) {
    throw new Error('Image generation timed out (300s). Check ~/.pre/comfyui.log for errors.');
  }

  // Copy from ComfyUI output to artifacts
  const home = process.env.HOME;
  const srcPath = path.join(home, '.pre/comfyui/output', filename);

  // Create date subdirectory
  const now = new Date();
  const dateStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  const dateDir = path.join(ARTIFACTS_DIR, dateStr);
  fs.mkdirSync(dateDir, { recursive: true });

  // Build descriptive filename from prompt
  const safeName = prompt
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_|_$/g, '')
    .slice(0, 50) || 'generated';
  const destPath = path.join(dateDir, `${safeName}.png`);

  try {
    fs.copyFileSync(srcPath, destPath);
  } catch {
    // Fallback to source path
    return `Image generated: ${srcPath}\nView at: /artifacts/${path.relative(ARTIFACTS_DIR, srcPath)}`;
  }

  const webPath = `/artifacts/${dateStr}/${safeName}.png`;
  const stats = fs.statSync(destPath);
  const sizeMB = (stats.size / 1024 / 1024).toFixed(1);

  console.log(`[comfyui] Done: ${destPath} (${sizeMB} MB)`);

  return `Image generated: ${destPath}\nView at: ${webPath}\nSize: ${sizeMB} MB | ${width}x${height} | ${steps} steps`;
}

module.exports = { imageGenerate };
