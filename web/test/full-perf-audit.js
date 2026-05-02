#!/usr/bin/env node
// PRE Full Performance Audit — Comprehensive Feature & Latency Test
// Tests all major features with real Ollama inference.
// Measures: TTFT, tok/s, tool call accuracy, hallucinations, failures.
// Run: node test/full-perf-audit.js

const http = require('http');
const { MODEL, MODEL_CTX, OLLAMA_PORT } = require('../src/constants');
const { buildToolDefs } = require('../src/tools-defs');
const { buildSystemPrompt } = require('../src/context');
const { estimateTokens } = require('../src/compression');
const tiers = require('../src/tool-tiers');

// ── Ollama streaming helper ──────────────────────────────────────────────────

function ollamaChat(messages, tools, { maxTokens = 512, temperature = null, think = false } = {}) {
  return new Promise((resolve, reject) => {
    const options = { num_predict: maxTokens, num_ctx: MODEL_CTX };
    if (temperature !== null) options.temperature = temperature;

    const body = JSON.stringify({
      model: MODEL,
      stream: true,
      keep_alive: '24h',
      ...(think === false ? { think: false } : {}),
      options,
      messages,
      ...(tools && tools.length > 0 ? { tools } : {}),
    });

    let response = '';
    let thinking = '';
    let toolCalls = null;
    let stats = {};
    let buffer = '';
    let firstTokenTime = null;
    const start = Date.now();

    const req = http.request({
      hostname: '127.0.0.1',
      port: OLLAMA_PORT,
      path: '/api/chat',
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    }, (res) => {
      res.on('data', (chunk) => {
        buffer += chunk.toString();
        let nlIdx;
        while ((nlIdx = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, nlIdx).trim();
          buffer = buffer.slice(nlIdx + 1);
          if (!line) continue;
          let parsed;
          try { parsed = JSON.parse(line); } catch { continue; }
          const msg = parsed.message || {};
          if (msg.thinking) thinking += msg.thinking;
          if (msg.content) {
            if (!firstTokenTime) firstTokenTime = Date.now();
            response += msg.content;
          }
          if (msg.tool_calls && msg.tool_calls.length > 0) {
            if (!firstTokenTime) firstTokenTime = Date.now();
            toolCalls = msg.tool_calls;
          }
          if (parsed.done) {
            stats = {
              prompt_eval_count: parsed.prompt_eval_count || 0,
              eval_count: parsed.eval_count || 0,
              eval_duration: parsed.eval_duration || 0,
              prompt_eval_duration: parsed.prompt_eval_duration || 0,
              total_duration: parsed.total_duration || 0,
            };
          }
        }
      });
      res.on('end', () => {
        const elapsed = Date.now() - start;
        const ttft = firstTokenTime ? firstTokenTime - start : elapsed;
        const tokPerSec = stats.eval_duration > 0
          ? (stats.eval_count / (stats.eval_duration / 1e9)).toFixed(1)
          : '0';
        const promptTokPerSec = stats.prompt_eval_duration > 0
          ? Math.round(stats.prompt_eval_count / (stats.prompt_eval_duration / 1e9))
          : 0;
        resolve({
          response, thinking, toolCalls, stats, elapsed, ttft,
          tokPerSec: parseFloat(tokPerSec),
          promptTokPerSec,
          promptTokens: stats.prompt_eval_count,
          genTokens: stats.eval_count,
        });
      });
      res.on('error', reject);
    });
    req.on('error', reject);
    req.setTimeout(120000, () => { req.destroy(); reject(new Error('Request timed out (120s)')); });
    req.write(body);
    req.end();
  });
}

// ── Test framework ───────────────────────────────────────────────────────────

const PASS = '\x1b[32m✓\x1b[0m';
const FAIL = '\x1b[31m✗\x1b[0m';
const WARN = '\x1b[33m⚠\x1b[0m';
const BOLD = '\x1b[1m';
const RESET = '\x1b[0m';
const DIM = '\x1b[2m';

const allResults = [];
let testNum = 0;
let passes = 0;
let fails = 0;
let warnings = 0;

function record(category, name, result) {
  testNum++;
  const status = result.pass ? (result.warning ? 'WARN' : 'PASS') : 'FAIL';
  const icon = result.pass ? (result.warning ? WARN : PASS) : FAIL;
  if (result.pass && !result.warning) passes++;
  else if (result.pass && result.warning) { passes++; warnings++; }
  else fails++;

  console.log(`  ${icon} ${name}`);
  if (result.ttft !== undefined) {
    console.log(`    ${DIM}TTFT: ${result.ttft}ms | Gen: ${result.tokPerSec} tok/s | Prompt: ${result.promptTokPerSec} tok/s | Prompt tokens: ${result.promptTokens} | Gen tokens: ${result.genTokens} | Elapsed: ${result.elapsed}ms${RESET}`);
  }
  if (result.detail) {
    console.log(`    ${DIM}${result.detail}${RESET}`);
  }
  if (result.warning) {
    console.log(`    ${WARN} ${result.warning}`);
  }

  allResults.push({ testNum, category, name, status, ...result });
}

// ── Test categories ──────────────────────────────────────────────────────────

async function testBasicInference(systemPrompt) {
  console.log(`\n${BOLD}━━━ 1. Basic Inference (no tools) ━━━${RESET}`);

  // 1a. Simple factual question
  const r1 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'What is the capital of France? Answer in one word.' },
  ], [], { maxTokens: 64, think: false });

  const hasCorrect = r1.response.toLowerCase().includes('paris');
  record('inference', 'Simple factual question (capital of France)', {
    pass: hasCorrect,
    detail: `Response: "${r1.response.trim().slice(0, 100)}"`,
    warning: !hasCorrect ? 'Expected "Paris" in response' : undefined,
    ...r1,
  });

  // 1b. Math question
  const r2 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'What is 17 * 23? Reply with just the number.' },
  ], [], { maxTokens: 64, think: false });

  const hasMath = r2.response.includes('391');
  record('inference', 'Math: 17 * 23 = 391', {
    pass: hasMath,
    detail: `Response: "${r2.response.trim().slice(0, 100)}"`,
    warning: !hasMath ? `Expected 391, got: "${r2.response.trim().slice(0, 50)}"` : undefined,
    ...r2,
  });

  // 1c. Instruction following — format compliance
  const r3 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'List exactly 3 colors, one per line. No numbering, no bullets, just the color names.' },
  ], [], { maxTokens: 128, think: false });

  const lines = r3.response.trim().split('\n').filter(l => l.trim());
  const isThree = lines.length >= 3 && lines.length <= 5;
  record('inference', 'Instruction following (3 colors, one per line)', {
    pass: isThree,
    detail: `Got ${lines.length} lines: ${lines.slice(0, 5).map(l => l.trim()).join(', ')}`,
    warning: !isThree ? `Expected 3 lines, got ${lines.length}` : undefined,
    ...r3,
  });
}

async function testCoreTools(systemPrompt) {
  console.log(`\n${BOLD}━━━ 2. Core Tool Calling ━━━${RESET}`);

  // Build CORE-only tool set
  const sid = `perf-core-${Date.now()}`;
  tiers.clearSession(sid);
  const activeDomains = tiers.getActiveDomains(sid);
  const tools = buildToolDefs({ activeDomains });

  // 2a. File listing — should call bash or list_dir
  const r1 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'List the files in /tmp. Use a tool.' },
  ], tools, { maxTokens: 512, temperature: 0.4 });

  const calledTool1 = r1.toolCalls && r1.toolCalls.length > 0;
  const toolName1 = calledTool1 ? r1.toolCalls[0].function?.name : null;
  const validTool1 = ['bash', 'list_dir', 'glob'].includes(toolName1);
  record('tools-core', 'File listing → should call bash/list_dir/glob', {
    pass: calledTool1 && validTool1,
    detail: `Tool: ${toolName1 || 'none'} | Args: ${JSON.stringify(r1.toolCalls?.[0]?.function?.arguments || {}).slice(0, 120)}`,
    warning: calledTool1 && !validTool1 ? `Unexpected tool: ${toolName1}` : (!calledTool1 ? 'No tool called' : undefined),
    ...r1,
  });

  // 2b. File read — should call read_file
  const r2 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'Read the contents of /etc/hosts' },
  ], tools, { maxTokens: 512, temperature: 0.4 });

  const calledTool2 = r2.toolCalls && r2.toolCalls.length > 0;
  const toolName2 = calledTool2 ? r2.toolCalls[0].function?.name : null;
  const validTool2 = ['read_file', 'bash'].includes(toolName2);
  record('tools-core', 'File read → should call read_file or bash cat', {
    pass: calledTool2 && validTool2,
    detail: `Tool: ${toolName2 || 'none'} | Args: ${JSON.stringify(r2.toolCalls?.[0]?.function?.arguments || {}).slice(0, 120)}`,
    warning: calledTool2 && !validTool2 ? `Unexpected tool: ${toolName2}` : (!calledTool2 ? 'No tool called' : undefined),
    ...r2,
  });

  // 2c. Grep/Search — should call grep or glob
  const r3 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'Search for files containing "TODO" in the current directory recursively.' },
  ], tools, { maxTokens: 512, temperature: 0.4 });

  const calledTool3 = r3.toolCalls && r3.toolCalls.length > 0;
  const toolName3 = calledTool3 ? r3.toolCalls[0].function?.name : null;
  const validTool3 = ['grep', 'bash', 'glob'].includes(toolName3);
  record('tools-core', 'Code search → should call grep/bash/glob', {
    pass: calledTool3 && validTool3,
    detail: `Tool: ${toolName3 || 'none'} | Args: ${JSON.stringify(r3.toolCalls?.[0]?.function?.arguments || {}).slice(0, 120)}`,
    warning: calledTool3 && !validTool3 ? `Unexpected tool: ${toolName3}` : (!calledTool3 ? 'No tool called' : undefined),
    ...r3,
  });

  // 2d. System info — should call system_info
  const r4 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'What operating system am I running? Use the system_info tool.' },
  ], tools, { maxTokens: 512, temperature: 0.4 });

  const calledTool4 = r4.toolCalls && r4.toolCalls.length > 0;
  const toolName4 = calledTool4 ? r4.toolCalls[0].function?.name : null;
  const validTool4 = ['system_info', 'bash'].includes(toolName4);
  record('tools-core', 'System info → should call system_info', {
    pass: calledTool4 && validTool4,
    detail: `Tool: ${toolName4 || 'none'}`,
    warning: calledTool4 && !validTool4 ? `Unexpected tool: ${toolName4}` : (!calledTool4 ? 'No tool called' : undefined),
    ...r4,
  });

  tiers.clearSession(sid);
}

async function testProgressiveDisclosure(systemPrompt) {
  console.log(`\n${BOLD}━━━ 3. Progressive Tool Disclosure ━━━${RESET}`);

  // 3a. Core-only tool count
  const sid = `perf-ptd-${Date.now()}`;
  tiers.clearSession(sid);
  const coreTools = buildToolDefs({ activeDomains: tiers.getActiveDomains(sid) });
  const allTools = buildToolDefs();

  const corePct = Math.round((1 - coreTools.length / allTools.length) * 100);
  record('progressive', `Core-only: ${coreTools.length} tools vs All: ${allTools.length} (${corePct}% reduction)`, {
    pass: coreTools.length < allTools.length,
    detail: `Token savings: ~${estimateTokens(JSON.stringify(allTools)) - estimateTokens(JSON.stringify(coreTools))} tokens`,
  });

  // 3b. Keyword auto-detection: PIM domain
  tiers.clearSession(sid);
  tiers.resolveKeywords(sid, 'Check my calendar events for today');
  const pimActive = tiers.getActiveDomains(sid).has('pim');
  record('progressive', 'Keyword auto-detect: "calendar" → PIM domain', {
    pass: pimActive,
    detail: `Active domains: ${[...tiers.getActiveDomains(sid)].join(', ') || 'none'}`,
    warning: !pimActive ? 'PIM domain not activated by "calendar" keyword' : undefined,
  });

  // 3c. Keyword auto-detection: DevOps domain
  tiers.clearSession(sid);
  tiers.resolveKeywords(sid, 'Show me running processes and kill node');
  const devopsActive = tiers.getActiveDomains(sid).has('devops');
  record('progressive', 'Keyword auto-detect: "process"+"kill" → DevOps domain', {
    pass: devopsActive,
    detail: `Active domains: ${[...tiers.getActiveDomains(sid)].join(', ') || 'none'}`,
  });

  // 3d. Keyword auto-detection: Cloud domain
  tiers.clearSession(sid);
  tiers.resolveKeywords(sid, 'Create a github pull request for this branch');
  const cloudActive = tiers.getActiveDomains(sid).has('cloud');
  record('progressive', 'Keyword auto-detect: "github"+"pull request" → Cloud domain', {
    pass: cloudActive,
    detail: `Active domains: ${[...tiers.getActiveDomains(sid)].join(', ') || 'none'}`,
  });

  // 3e. Token savings measurement
  tiers.clearSession(sid);
  tiers.resolveKeywords(sid, 'List the files in the current directory');
  const nodomainTools = buildToolDefs({ activeDomains: tiers.getActiveDomains(sid) });
  const allTokens = estimateTokens(JSON.stringify(allTools));
  const coreTokens = estimateTokens(JSON.stringify(nodomainTools));
  const savings = Math.round((1 - coreTokens / allTokens) * 100);

  record('progressive', `Token savings for generic query: ${savings}% (${allTokens} → ${coreTokens} tokens)`, {
    pass: savings > 30,
    detail: `${nodomainTools.length} tools loaded for "list files" (no domain activated)`,
    warning: savings <= 30 ? `Expected >30% savings, got ${savings}%` : undefined,
  });

  // 3f. Latency comparison: CORE-only vs ALL tools
  const msgSimple = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'What is 2+2? Answer briefly.' },
  ];

  const rCore = await ollamaChat(msgSimple, nodomainTools, { maxTokens: 128, think: false });
  const rAll = await ollamaChat(msgSimple, allTools, { maxTokens: 128, think: false });

  const speedup = rAll.ttft > 0 ? Math.round((1 - rCore.ttft / rAll.ttft) * 100) : 0;
  record('progressive', `TTFT: CORE ${rCore.ttft}ms vs ALL ${rAll.ttft}ms (${speedup}% faster)`, {
    pass: true, // informational
    detail: `Core prompt tokens: ${rCore.promptTokens} | All prompt tokens: ${rAll.promptTokens}`,
    ...rCore,
  });

  tiers.clearSession(sid);
}

async function testToolCallQuality(systemPrompt) {
  console.log(`\n${BOLD}━━━ 4. Tool Call Quality (hallucination check) ━━━${RESET}`);

  const sid = `perf-quality-${Date.now()}`;
  tiers.clearSession(sid);
  const tools = buildToolDefs({ activeDomains: tiers.getActiveDomains(sid) });

  // 4a. No-tool question should NOT trigger tool calls
  const r1 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'What is the meaning of life? Answer philosophically in 2 sentences.' },
  ], tools, { maxTokens: 256, temperature: 0.4 });

  const noToolsNeeded = !r1.toolCalls || r1.toolCalls.length === 0;
  record('quality', 'No-tool question should not trigger tools', {
    pass: noToolsNeeded,
    detail: `Tool calls: ${r1.toolCalls ? r1.toolCalls.map(tc => tc.function?.name).join(', ') : 'none'} | Response: "${r1.response.trim().slice(0, 100)}"`,
    warning: !noToolsNeeded ? `Hallucinated tool call: ${r1.toolCalls?.[0]?.function?.name}` : undefined,
    ...r1,
  });

  // 4b. Tool call should have valid parameters (not hallucinated)
  const r2 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'Read the file /etc/hostname' },
  ], tools, { maxTokens: 256, temperature: 0.4 });

  const hasTool2 = r2.toolCalls && r2.toolCalls.length > 0;
  let validParams2 = false;
  if (hasTool2) {
    const tc = r2.toolCalls[0];
    const args = tc.function?.arguments || {};
    // Check: args should have a "path" or "command" key with a string value
    validParams2 = (typeof args.path === 'string' && args.path.includes('/etc/hostname'))
      || (typeof args.command === 'string' && args.command.includes('/etc/hostname'));
  }
  record('quality', 'Tool args should reference correct file path', {
    pass: hasTool2 && validParams2,
    detail: `Tool: ${r2.toolCalls?.[0]?.function?.name} | Args: ${JSON.stringify(r2.toolCalls?.[0]?.function?.arguments || {}).slice(0, 150)}`,
    warning: hasTool2 && !validParams2 ? 'Tool args do not reference /etc/hostname' : (!hasTool2 ? 'No tool called' : undefined),
    ...r2,
  });

  // 4c. Should not call a non-existent tool
  const r3 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'Show me the weather forecast for Boulder Creek, CA' },
  ], tools, { maxTokens: 512, temperature: 0.4 });

  let hallucinated3 = false;
  if (r3.toolCalls) {
    const toolNames = new Set(tools.map(t => t.function.name));
    for (const tc of r3.toolCalls) {
      if (!toolNames.has(tc.function?.name)) {
        hallucinated3 = true;
      }
    }
  }
  record('quality', 'No hallucinated tool names (weather query)', {
    pass: !hallucinated3,
    detail: `Tool calls: ${r3.toolCalls ? r3.toolCalls.map(tc => tc.function?.name).join(', ') : 'none'}`,
    warning: hallucinated3 ? `Hallucinated tool: ${r3.toolCalls.map(tc => tc.function?.name).join(', ')}` : undefined,
    ...r3,
  });

  // 4d. Multi-tool: should call 2+ tools for a complex request
  const r4 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'Find all .js files in /tmp and tell me how many there are. You may need multiple tool calls.' },
  ], tools, { maxTokens: 512, temperature: 0.4 });

  const hasTool4 = r4.toolCalls && r4.toolCalls.length > 0;
  record('quality', 'Complex query triggers at least one tool call', {
    pass: hasTool4,
    detail: `Tool calls: ${r4.toolCalls ? r4.toolCalls.map(tc => tc.function?.name).join(', ') : 'none'}`,
    warning: !hasTool4 ? 'Expected tool call for file search task' : undefined,
    ...r4,
  });

  tiers.clearSession(sid);
}

async function testDomainToolCalling(systemPrompt) {
  console.log(`\n${BOLD}━━━ 5. Domain-Specific Tool Calling ━━━${RESET}`);

  // 5a. DevOps: process listing with devops domain active
  const sid5a = `perf-domain-${Date.now()}`;
  tiers.clearSession(sid5a);
  tiers.activateDomain(sid5a, 'devops');
  const devopsTools = buildToolDefs({ activeDomains: tiers.getActiveDomains(sid5a) });

  const r1 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'Show me running processes. Use the process_list tool.' },
  ], devopsTools, { maxTokens: 512, temperature: 0.4 });

  const calledProcess = r1.toolCalls?.some(tc => ['process_list', 'bash'].includes(tc.function?.name));
  record('domain', 'DevOps: process listing', {
    pass: !!calledProcess,
    detail: `Tools: ${r1.toolCalls ? r1.toolCalls.map(tc => tc.function?.name).join(', ') : 'none'}`,
    warning: !calledProcess ? 'Expected process_list or bash tool' : undefined,
    ...r1,
  });

  // 5b. PIM: calendar check with pim domain active
  const sid5b = `perf-pim-${Date.now()}`;
  tiers.clearSession(sid5b);
  tiers.activateDomain(sid5b, 'pim');
  const pimTools = buildToolDefs({ activeDomains: tiers.getActiveDomains(sid5b) });

  const r2 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'Check my calendar for events today.' },
  ], pimTools, { maxTokens: 512, temperature: 0.4 });

  const calledCalendar = r2.toolCalls?.some(tc => ['apple_calendar', 'bash'].includes(tc.function?.name));
  record('domain', 'PIM: calendar event check', {
    pass: !!calledCalendar,
    detail: `Tools: ${r2.toolCalls ? r2.toolCalls.map(tc => tc.function?.name).join(', ') : 'none'}`,
    warning: !calledCalendar ? 'Expected apple_calendar tool' : undefined,
    ...r2,
  });

  // 5c. Automation: cron listing with automation domain active
  const sid5c = `perf-auto-${Date.now()}`;
  tiers.clearSession(sid5c);
  tiers.activateDomain(sid5c, 'automation');
  const autoTools = buildToolDefs({ activeDomains: tiers.getActiveDomains(sid5c) });

  const r3 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'List my scheduled cron jobs.' },
  ], autoTools, { maxTokens: 512, temperature: 0.4 });

  const calledCron = r3.toolCalls?.some(tc => ['cron', 'bash'].includes(tc.function?.name));
  record('domain', 'Automation: cron listing', {
    pass: !!calledCron,
    detail: `Tools: ${r3.toolCalls ? r3.toolCalls.map(tc => tc.function?.name).join(', ') : 'none'}`,
    warning: !calledCron ? 'Expected cron tool' : undefined,
    ...r3,
  });

  tiers.clearSession(sid5a);
  tiers.clearSession(sid5b);
  tiers.clearSession(sid5c);
}

async function testTemperatureEffect(systemPrompt) {
  console.log(`\n${BOLD}━━━ 6. Temperature Impact (0.4 vs 1.0) ━━━${RESET}`);

  const tools = buildToolDefs({ activeDomains: new Set() }); // CORE only
  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'Read the file /etc/hosts' },
  ];

  // Run 3 trials at each temperature and check consistency
  const results04 = [];
  const results10 = [];

  for (let i = 0; i < 3; i++) {
    const r04 = await ollamaChat(messages, tools, { maxTokens: 256, temperature: 0.4, think: false });
    results04.push(r04.toolCalls?.[0]?.function?.name || 'none');

    const r10 = await ollamaChat(messages, tools, { maxTokens: 256, temperature: 1.0, think: false });
    results10.push(r10.toolCalls?.[0]?.function?.name || 'none');
  }

  const consistent04 = new Set(results04).size === 1;
  const consistent10 = new Set(results10).size === 1;

  record('temperature', `temp=0.4 consistency: ${results04.join(', ')} → ${consistent04 ? 'consistent' : 'varied'}`, {
    pass: true, // informational
    detail: `3 trials: [${results04.join(', ')}]`,
    warning: !consistent04 ? 'Tool selection varied across trials at temp=0.4' : undefined,
  });

  record('temperature', `temp=1.0 consistency: ${results10.join(', ')} → ${consistent10 ? 'consistent' : 'varied'}`, {
    pass: true, // informational
    detail: `3 trials: [${results10.join(', ')}]`,
    warning: !consistent10 ? 'Tool selection varied across trials at temp=1.0 (expected)' : undefined,
  });
}

async function testCompression() {
  console.log(`\n${BOLD}━━━ 7. Compression System ━━━${RESET}`);

  const compression = require('../src/compression');

  // 7a. Token estimation accuracy (static)
  const shortText = 'Hello world, this is a test.';
  const est = compression.estimateTokens(shortText);
  record('compression', `Token estimation: "${shortText}" → ${est} tokens`, {
    pass: est > 0 && est < 20,
    detail: `chars=${shortText.length}, ratio=${(shortText.length / est).toFixed(2)} chars/tok`,
  });

  // 7b. Budget constants are reasonable
  const budget = MODEL_CTX - compression.SYSTEM_PROMPT_TOKENS - compression.OUTPUT_RESERVE - compression.CORE_TOOL_TOKENS;
  record('compression', `Context budget: ${budget} tokens (of ${MODEL_CTX})`, {
    pass: budget > 50000,
    detail: `System: ${compression.SYSTEM_PROMPT_TOKENS} | Output: ${compression.OUTPUT_RESERVE} | Core tools: ${compression.CORE_TOOL_TOKENS} | Available: ${budget}`,
    warning: budget <= 50000 ? `Budget seems low: ${budget}` : undefined,
  });

  // 7c. Compression threshold
  const threshold = Math.round(budget * compression.COMPRESSION_THRESHOLD);
  record('compression', `Compression triggers at ${threshold} tokens (${compression.COMPRESSION_THRESHOLD * 100}% of ${budget})`, {
    pass: threshold > 40000 && threshold < budget,
    detail: `Threshold: ${threshold} | Keep recent: ${compression.KEEP_RECENT_TURNS} turns`,
  });
}

async function testSessionSearch() {
  console.log(`\n${BOLD}━━━ 8. Session Search (FTS) ━━━${RESET}`);

  const fts = require('../src/fts');
  const sessions = require('../src/sessions');

  // Create a test session with searchable content
  const sid = sessions.createSession('perf', 'fts-test', true);
  sessions.appendMessage(sid, { role: 'user', content: 'Tell me about quantum computing' });
  sessions.appendMessage(sid, { role: 'assistant', content: 'Quantum computing uses qubits instead of classical bits.' });

  // 8a. Search for known content
  const start = Date.now();
  const result = fts.searchSessions('quantum computing');
  const elapsed = Date.now() - start;

  const found = result.includes('quantum');
  record('fts', `Search "quantum computing" across sessions (${elapsed}ms)`, {
    pass: found,
    detail: `Found: ${found} | Elapsed: ${elapsed}ms | Result length: ${result.length} chars`,
    warning: !found ? 'Expected to find "quantum" in search results' : undefined,
  });

  // 8b. Search with no matches
  const r2 = fts.searchSessions('xyznonexistent12345');
  const noMatch = r2.includes('No matches') || r2.includes('No results') || r2.includes('0 result');
  record('fts', 'Search for non-existent term returns "No matches"', {
    pass: noMatch,
    detail: `Result: "${r2.slice(0, 100)}"`,
  });

  // Clean up
  sessions.deleteSession(sid);
}

async function testRequestTools(systemPrompt) {
  console.log(`\n${BOLD}━━━ 9. request_tools Meta-Tool ━━━${RESET}`);

  const sid = `perf-reqtools-${Date.now()}`;
  tiers.clearSession(sid);
  const tools = buildToolDefs({ activeDomains: tiers.getActiveDomains(sid) });

  // Model should call request_tools when it needs a domain tool
  const r1 = await ollamaChat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'I need to check my email. You may need to load additional tools first using the request_tools tool with domain "pim".' },
  ], tools, { maxTokens: 512, temperature: 0.4 });

  const calledReqTools = r1.toolCalls?.some(tc => tc.function?.name === 'request_tools');
  const calledPIM = r1.toolCalls?.some(tc => {
    if (tc.function?.name !== 'request_tools') return false;
    const domain = (tc.function?.arguments?.domain || '').toLowerCase();
    return domain === 'pim' || domain === 'all';
  });

  record('meta-tool', 'Model calls request_tools for email (PIM domain)', {
    pass: !!calledReqTools,
    detail: `Tools called: ${r1.toolCalls ? r1.toolCalls.map(tc => `${tc.function?.name}(${JSON.stringify(tc.function?.arguments || {})})`).join(', ') : 'none'}`,
    warning: !calledReqTools ? 'Model did not call request_tools — may have used available tools instead' : (!calledPIM ? 'Called request_tools but not with "pim" domain' : undefined),
    ...r1,
  });

  tiers.clearSession(sid);
}

async function testLatencyProfile(systemPrompt) {
  console.log(`\n${BOLD}━━━ 10. Latency Profile ━━━${RESET}`);

  const tools = buildToolDefs({ activeDomains: new Set() });
  const allTools = buildToolDefs();

  const configs = [
    { label: 'No tools, think=false', tools: [], opts: { maxTokens: 128, think: false } },
    { label: 'CORE tools, think=false', tools: tools, opts: { maxTokens: 128, think: false, temperature: 0.4 } },
    { label: 'ALL tools, think=false', tools: allTools, opts: { maxTokens: 128, think: false, temperature: 0.4 } },
    { label: 'CORE tools, think=true', tools: tools, opts: { maxTokens: 512, temperature: 0.4 } },
  ];

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: 'What time is it? Answer briefly.' },
  ];

  for (const cfg of configs) {
    const r = await ollamaChat(messages, cfg.tools, cfg.opts);
    record('latency', cfg.label, {
      pass: r.ttft < 30000, // 30s timeout
      detail: `TTFT: ${r.ttft}ms | Gen: ${r.tokPerSec} tok/s | Prompt: ${r.promptTokPerSec} tok/s | ${r.promptTokens} prompt tok | ${r.genTokens} gen tok | Tools: ${cfg.tools.length}`,
      warning: r.ttft > 10000 ? `TTFT high: ${r.ttft}ms` : undefined,
      ...r,
    });
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function run() {
  console.log('╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║          PRE Full Performance Audit — Feature & Latency Test        ║');
  console.log('╠══════════════════════════════════════════════════════════════════════╣');
  console.log(`║ Model: ${MODEL.padEnd(62)}║`);
  console.log(`║ Context: ${String(MODEL_CTX).padEnd(60)}║`);
  console.log(`║ Port: ${String(OLLAMA_PORT).padEnd(63)}║`);
  console.log('╚══════════════════════════════════════════════════════════════════════╝');

  // Warmup
  process.stdout.write('\nWarming up model... ');
  const systemPrompt = buildSystemPrompt('/tmp');
  try {
    await ollamaChat([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: 'Say "ready".' },
    ], [], { maxTokens: 32, think: false });
    console.log('done.\n');
  } catch (err) {
    console.log(`FAILED: ${err.message}\nIs Ollama running with ${MODEL}?`);
    process.exit(1);
  }

  const totalStart = Date.now();

  // Run all test categories
  await testBasicInference(systemPrompt);
  await testCoreTools(systemPrompt);
  await testProgressiveDisclosure(systemPrompt);
  await testToolCallQuality(systemPrompt);
  await testDomainToolCalling(systemPrompt);
  await testTemperatureEffect(systemPrompt);
  await testCompression();
  await testSessionSearch();
  await testRequestTools(systemPrompt);
  await testLatencyProfile(systemPrompt);

  const totalElapsed = ((Date.now() - totalStart) / 1000).toFixed(1);

  // ── Summary ──
  console.log(`\n${'═'.repeat(72)}`);
  console.log(`${BOLD}SUMMARY${RESET}`);
  console.log(`${'═'.repeat(72)}\n`);

  console.log(`  ${PASS} Passed:   ${passes}`);
  console.log(`  ${FAIL} Failed:   ${fails}`);
  console.log(`  ${WARN} Warnings: ${warnings}`);
  console.log(`  Total time: ${totalElapsed}s\n`);

  // Latency summary table
  const latencyResults = allResults.filter(r => r.ttft !== undefined && r.ttft > 0);
  if (latencyResults.length > 0) {
    console.log(`${BOLD}Latency Summary:${RESET}`);
    console.log(`  ${'Test'.padEnd(50)} ${'TTFT'.padStart(8)} ${'tok/s'.padStart(8)} ${'Prompt'.padStart(8)} ${'Tokens'.padStart(8)}`);
    console.log(`  ${'─'.repeat(82)}`);
    for (const r of latencyResults) {
      console.log(`  ${r.name.padEnd(50)} ${(r.ttft + 'ms').padStart(8)} ${(r.tokPerSec + '').padStart(8)} ${(r.promptTokPerSec + '/s').padStart(8)} ${(r.promptTokens + '').padStart(8)}`);
    }

    const avgTTFT = Math.round(latencyResults.reduce((s, r) => s + r.ttft, 0) / latencyResults.length);
    const avgTokS = (latencyResults.reduce((s, r) => s + r.tokPerSec, 0) / latencyResults.length).toFixed(1);
    const avgPromptTokS = Math.round(latencyResults.reduce((s, r) => s + r.promptTokPerSec, 0) / latencyResults.length);
    console.log(`  ${'─'.repeat(82)}`);
    console.log(`  ${'AVERAGE'.padEnd(50)} ${(avgTTFT + 'ms').padStart(8)} ${(avgTokS + '').padStart(8)} ${(avgPromptTokS + '/s').padStart(8)}`);
  }

  // Failures detail
  const failures = allResults.filter(r => r.status === 'FAIL');
  if (failures.length > 0) {
    console.log(`\n${BOLD}${FAIL} Failures:${RESET}`);
    for (const f of failures) {
      console.log(`  [${f.category}] ${f.name}`);
      if (f.detail) console.log(`    ${f.detail}`);
      if (f.warning) console.log(`    ${f.warning}`);
    }
  }

  // Warnings detail
  const warns = allResults.filter(r => r.warning && r.status === 'WARN');
  if (warns.length > 0) {
    console.log(`\n${BOLD}${WARN} Warnings:${RESET}`);
    for (const w of warns) {
      console.log(`  [${w.category}] ${w.name}`);
      console.log(`    ${w.warning}`);
    }
  }

  console.log();
  process.exit(fails > 0 ? 1 : 0);
}

run().catch(err => {
  console.error('Audit failed:', err.message);
  process.exit(1);
});
