// PRE Web GUI — Voice Interface
// Speech-to-text via Whisper (local), text-to-speech via macOS `say`.
// Two modes:
//   1. Server-side: record via sox/rec, transcribe via Whisper CLI
//   2. Browser-side: record via Web Audio API, POST audio to /api/voice/transcribe
//
// All processing happens locally — no audio leaves the machine.

const { execSync, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const VOICE_DIR = path.join(os.tmpdir(), 'pre-voice');
if (!fs.existsSync(VOICE_DIR)) fs.mkdirSync(VOICE_DIR, { recursive: true });

// ── Configuration ──────────────────────────────────────────────────────────

const WHISPER_MODEL = 'base.en';  // Good balance of speed and accuracy for English
const TTS_VOICE = 'Samantha';     // Default macOS voice (clear, natural)
const TTS_RATE = 185;             // Words per minute (default ~175, slightly faster)
// Max recording: 120s (enforced browser-side)

// Check what's available
const hasWhisper = (() => {
  try { execSync('which whisper', { stdio: 'pipe' }); return true; } catch { return false; }
})();

const hasSox = (() => {
  try { execSync('which rec', { stdio: 'pipe' }); return true; } catch { return false; }
})();

const hasSay = (() => {
  try { execSync('which say', { stdio: 'pipe' }); return true; } catch { return false; }
})();

const hasFfmpeg = (() => {
  try { execSync('which ffmpeg', { stdio: 'pipe' }); return true; } catch { return false; }
})();

// ── Speech-to-Text ─────────────────────────────────────────────────────────

/**
 * Transcribe an audio file using Whisper.
 * @param {string} audioPath - Path to audio file (wav, mp3, m4a, webm, etc.)
 * @param {Object} opts - { language, model }
 * @returns {Promise<string>} Transcribed text
 */
async function transcribe(audioPath, opts = {}) {
  if (!hasWhisper) {
    return { error: 'Whisper not installed. Run: pip install openai-whisper' };
  }

  if (!fs.existsSync(audioPath)) {
    return { error: `Audio file not found: ${audioPath}` };
  }

  const model = opts.model || WHISPER_MODEL;
  const language = opts.language || 'en';
  const outputDir = VOICE_DIR;
  // Convert webm to wav if needed (Whisper handles most formats, but webm can be iffy)
  let inputPath = audioPath;
  if (audioPath.endsWith('.webm') && hasFfmpeg) {
    const wavPath = audioPath.replace('.webm', '.wav');
    try {
      execSync(`ffmpeg -y -i "${audioPath}" -ar 16000 -ac 1 "${wavPath}" 2>/dev/null`, { timeout: 30000 });
      inputPath = wavPath;
    } catch {
      // Fall through — Whisper might handle it
    }
  }

  try {
    const cmd = `whisper "${inputPath}" --model ${model} --language ${language} --output_format txt --output_dir "${outputDir}" --fp16 False 2>/dev/null`;
    execSync(cmd, { timeout: 60000, maxBuffer: 10 * 1024 * 1024 });

    // Read the transcript
    const baseName = path.basename(inputPath, path.extname(inputPath));
    const txtPath = path.join(outputDir, `${baseName}.txt`);

    if (fs.existsSync(txtPath)) {
      const text = fs.readFileSync(txtPath, 'utf-8').trim();
      // Clean up
      try { fs.unlinkSync(txtPath); } catch {}
      return { text, model, language };
    }

    return { error: 'Transcription produced no output' };
  } catch (err) {
    return { error: `Transcription failed: ${err.message}` };
  }
}

/**
 * Transcribe audio from a base64-encoded buffer.
 * Used by the browser-side recording flow.
 */
async function transcribeBuffer(base64Audio, mimeType = 'audio/webm') {
  const ext = mimeType.includes('wav') ? '.wav'
    : mimeType.includes('mp4') || mimeType.includes('m4a') ? '.m4a'
    : mimeType.includes('ogg') ? '.ogg'
    : '.webm';

  const tmpFile = path.join(VOICE_DIR, `recording_${Date.now()}${ext}`);

  try {
    fs.writeFileSync(tmpFile, Buffer.from(base64Audio, 'base64'));
    const result = await transcribe(tmpFile);
    return result;
  } finally {
    try { fs.unlinkSync(tmpFile); } catch {}
    // Also clean up any converted wav
    const wavFile = tmpFile.replace(ext, '.wav');
    try { fs.unlinkSync(wavFile); } catch {}
  }
}

// ── Text-to-Speech ─────────────────────────────────────────────────────────

/**
 * Speak text using macOS `say` command.
 * Generates audio and optionally plays it.
 *
 * @param {string} text - Text to speak
 * @param {Object} opts - { voice, rate, output (file path), play (boolean) }
 */
function speak(text, opts = {}) {
  if (!hasSay) return { error: 'macOS say command not available' };
  if (!text) return { error: 'No text provided' };

  const voice = opts.voice || TTS_VOICE;
  const rate = opts.rate || TTS_RATE;

  // Sanitize text for shell (remove problematic characters)
  const safeText = text
    .replace(/[`$\\]/g, '')
    .replace(/"/g, '\\"')
    .slice(0, 5000); // Cap length

  if (opts.output) {
    // Generate audio file (AIFF format — macOS native)
    const outPath = opts.output.endsWith('.aiff') ? opts.output : opts.output + '.aiff';
    try {
      execSync(`say -v "${voice}" -r ${rate} -o "${outPath}" "${safeText}"`, { timeout: 30000 });

      // Convert to mp3 if ffmpeg is available and requested
      if (opts.format === 'mp3' && hasFfmpeg) {
        const mp3Path = outPath.replace('.aiff', '.mp3');
        execSync(`ffmpeg -y -i "${outPath}" -codec:a libmp3lame -q:a 2 "${mp3Path}" 2>/dev/null`, { timeout: 30000 });
        try { fs.unlinkSync(outPath); } catch {}
        return { file: mp3Path, voice, rate };
      }

      return { file: outPath, voice, rate };
    } catch (err) {
      return { error: `TTS generation failed: ${err.message}` };
    }
  }

  // Play directly (non-blocking)
  if (opts.play !== false) {
    exec(`say -v "${voice}" -r ${rate} "${safeText}"`);
    return { spoken: true, voice, rate, length: safeText.length };
  }

  return { error: 'No output or play option specified' };
}

/**
 * List available macOS voices.
 */
function listVoices() {
  if (!hasSay) return { error: 'macOS say command not available' };
  try {
    const output = execSync('say -v "?"', { encoding: 'utf-8', timeout: 10000 });
    const voices = output.trim().split('\n')
      .map(line => {
        const match = line.match(/^(\S+)\s+(\S+)\s+#\s*(.*)$/);
        if (match) return { name: match[1], locale: match[2], sample: match[3] };
        return null;
      })
      .filter(Boolean);

    // Filter to English voices for the summary
    const englishVoices = voices.filter(v => v.locale.startsWith('en'));
    return { voices: englishVoices, total: voices.length };
  } catch (err) {
    return { error: `Failed to list voices: ${err.message}` };
  }
}

// ── Tool Dispatcher ────────────────────────────────────────────────────────

async function voice(args) {
  const action = (args.action || '').toLowerCase();

  switch (action) {
    case 'transcribe': {
      if (args.audio_base64) {
        const result = await transcribeBuffer(args.audio_base64, args.mime_type);
        if (result.error) return `Error: ${result.error}`;
        return `Transcribed (${result.model}, ${result.language}): ${result.text}`;
      }
      if (args.path) {
        const result = await transcribe(args.path, { model: args.model, language: args.language });
        if (result.error) return `Error: ${result.error}`;
        return `Transcribed (${result.model}, ${result.language}): ${result.text}`;
      }
      return 'Error: provide audio_base64 or path for transcription';
    }

    case 'speak': case 'say': case 'tts': {
      if (!args.text) return 'Error: text is required';
      if (args.output) {
        const result = speak(args.text, { voice: args.voice, rate: args.rate, output: args.output, format: args.format });
        if (result.error) return `Error: ${result.error}`;
        return `Audio saved to: ${result.file} (voice: ${result.voice}, rate: ${result.rate})`;
      }
      const result = speak(args.text, { voice: args.voice, rate: args.rate, play: true });
      if (result.error) return `Error: ${result.error}`;
      return `Speaking ${result.length} chars (voice: ${result.voice}, rate: ${result.rate})`;
    }

    case 'voices': case 'list_voices': {
      const result = listVoices();
      if (result.error) return `Error: ${result.error}`;
      const lines = result.voices.map(v => `  ${v.name} (${v.locale})`);
      return `English voices (${result.voices.length} of ${result.total} total):\n${lines.join('\n')}`;
    }

    case 'status': {
      const capabilities = [];
      if (hasWhisper) capabilities.push('Whisper STT (installed)');
      else capabilities.push('Whisper STT (not installed — pip install openai-whisper)');
      if (hasSay) capabilities.push('macOS TTS (available)');
      if (hasFfmpeg) capabilities.push('FFmpeg (available — format conversion)');
      if (hasSox) capabilities.push('Sox/rec (available — mic recording)');
      else capabilities.push('Sox/rec (not installed — brew install sox)');
      return `Voice capabilities:\n${capabilities.map(c => `  ${c}`).join('\n')}`;
    }

    default:
      return 'Unknown voice action. Available: transcribe, speak, voices, status';
  }
}

module.exports = {
  voice,
  transcribe,
  transcribeBuffer,
  speak,
  listVoices,
  hasWhisper,
  hasSay,
  hasFfmpeg,
};
