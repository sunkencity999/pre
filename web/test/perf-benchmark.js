#!/usr/bin/env node
// PRE Performance Benchmark — Progressive Tool Disclosure
// Measures real Ollama latency with different tool configurations.
// Run: node test/perf-benchmark.js

const http = require('http');
const { MODEL, MODEL_CTX, OLLAMA_PORT } = require('../src/constants');
const { buildToolDefs } = require('../src/tools-defs');
const { buildSystemPrompt } = require('../src/context');
const { estimateTokens } = require('../src/compression');
const tiers = require('../src/tool-tiers');

// ── Ollama request helper ─────────────────────────────────────────────────────

function ollamaChat(messages, tools, maxTokens = 256) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: MODEL,
      stream: false,
      keep_alive: '24h',
      think: false,
      options: { num_predict: maxTokens, num_ctx: MODEL_CTX },
      messages,
      ...(tools && tools.length > 0 ? { tools } : {}),
    });

    const start = Date.now();
    const req = http.request({
      hostname: '127.0.0.1',
      port: OLLAMA_PORT,
      path: '/api/chat',
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        const elapsed = Date.now() - start;
        try {
          const parsed = JSON.parse(data);
          resolve({
            elapsed,
            prompt_eval_count: parsed.prompt_eval_count || 0,
            eval_count: parsed.eval_count || 0,
            prompt_eval_duration: parsed.prompt_eval_duration || 0,
            eval_duration: parsed.eval_duration || 0,
            total_duration: parsed.total_duration || 0,
            response: (parsed.message?.content || '').slice(0, 200),
            tool_calls: parsed.message?.tool_calls || null,
          });
        } catch (err) {
          reject(new Error(`Parse error: ${err.message}`));
        }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ── Benchmark configs ─────────────────────────────────────────────────────────

const WARMUP_PROMPT = 'Say "ready" and nothing else.';
const TEST_PROMPTS = [
  { label: 'Simple question (no tools needed)', msg: 'What is 2 + 2?' },
  { label: 'File task (core tools)', msg: 'List the files in the current directory.' },
  { label: 'PIM task (domain: pim)', msg: 'Show me my calendar events for today.' },
  { label: 'DevOps task (domain: devops)', msg: 'Show me running processes.' },
];

const TOOL_CONFIGS = [
  { name: 'ALL tools (baseline)', getDomains: () => null },
  { name: 'CORE only', getDomains: () => new Set() },
  { name: 'CORE + auto-detect', getDomains: (msg) => {
    const sid = `bench-${Date.now()}`;
    tiers.clearSession(sid);
    tiers.resolveKeywords(sid, msg);
    return tiers.getActiveDomains(sid);
  }},
];

// ── Main ──────────────────────────────────────────────────────────────────────

async function run() {
  console.log('╔══════════════════════════════════════════════════════════════════╗');
  console.log('║     PRE Performance Benchmark — Progressive Tool Disclosure     ║');
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log(`║ Model: ${MODEL.padEnd(56)}║`);
  console.log(`║ Context: ${String(MODEL_CTX).padEnd(54)}║`);
  console.log(`║ Ollama port: ${String(OLLAMA_PORT).padEnd(50)}║`);
  console.log('╚══════════════════════════════════════════════════════════════════╝\n');

  // Warmup — ensure model is loaded in memory
  process.stdout.write('Warming up model... ');
  const systemPrompt = buildSystemPrompt('/tmp');
  try {
    await ollamaChat([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: WARMUP_PROMPT },
    ], [], 32);
    console.log('done.\n');
  } catch (err) {
    console.log(`FAILED: ${err.message}\nIs Ollama running with ${MODEL}?`);
    process.exit(1);
  }

  const results = [];

  for (const prompt of TEST_PROMPTS) {
    console.log(`━━━ ${prompt.label} ━━━`);
    console.log(`Prompt: "${prompt.msg}"\n`);

    const row = { prompt: prompt.label };

    for (const cfg of TOOL_CONFIGS) {
      const domains = cfg.getDomains(prompt.msg);
      const opts = domains !== null ? { activeDomains: domains } : {};
      const tools = buildToolDefs(opts);
      const toolTokens = estimateTokens(JSON.stringify(tools));

      const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: prompt.msg },
      ];

      process.stdout.write(`  ${cfg.name.padEnd(25)} `);

      try {
        const r = await ollamaChat(messages, tools, 256);
        const promptMs = Math.round(r.prompt_eval_duration / 1e6);
        const evalMs = Math.round(r.eval_duration / 1e6);
        const promptTokPerSec = r.prompt_eval_count > 0 && r.prompt_eval_duration > 0
          ? Math.round(r.prompt_eval_count / (r.prompt_eval_duration / 1e9))
          : 0;

        console.log(
          `${String(tools.length).padStart(3)} tools | ` +
          `~${String(toolTokens).padStart(5)} tok defs | ` +
          `${String(r.prompt_eval_count).padStart(5)} prompt tok | ` +
          `${String(promptMs).padStart(5)}ms prompt | ` +
          `${String(evalMs).padStart(5)}ms gen | ` +
          `${promptTokPerSec} tok/s`
        );

        row[cfg.name] = {
          tools: tools.length,
          toolTokens,
          promptTokens: r.prompt_eval_count,
          promptMs,
          evalMs,
          totalMs: r.elapsed,
          promptTokPerSec,
          hadToolCalls: !!r.tool_calls,
        };
      } catch (err) {
        console.log(`ERROR: ${err.message}`);
        row[cfg.name] = { error: err.message };
      }
    }
    results.push(row);
    console.log();
  }

  // Summary table
  console.log('╔══════════════════════════════════════════════════════════════════╗');
  console.log('║                        SUMMARY                                 ║');
  console.log('╚══════════════════════════════════════════════════════════════════╝\n');

  const baseline = 'ALL tools (baseline)';
  const progressive = 'CORE + auto-detect';

  for (const row of results) {
    const b = row[baseline];
    const p = row[progressive];
    if (!b || !p || b.error || p.error) continue;

    const tokenSavings = Math.round((1 - p.promptTokens / b.promptTokens) * 100);
    const promptSpeedup = b.promptMs > 0 ? ((b.promptMs - p.promptMs) / b.promptMs * 100).toFixed(1) : 'N/A';
    const totalSpeedup = b.totalMs > 0 ? ((b.totalMs - p.totalMs) / b.totalMs * 100).toFixed(1) : 'N/A';

    console.log(`${row.prompt}`);
    console.log(`  Prompt tokens:  ${b.promptTokens} → ${p.promptTokens} (${tokenSavings > 0 ? '-' : '+'}${Math.abs(tokenSavings)}%)`);
    console.log(`  Prompt eval:    ${b.promptMs}ms → ${p.promptMs}ms (${promptSpeedup}% faster)`);
    console.log(`  Total latency:  ${b.totalMs}ms → ${p.totalMs}ms (${totalSpeedup}% faster)`);
    console.log(`  Tool defs:      ${b.toolTokens} → ${p.toolTokens} tokens`);
    console.log();
  }
}

run().catch(err => {
  console.error('Benchmark failed:', err.message);
  process.exit(1);
});
