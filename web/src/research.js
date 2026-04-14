// PRE Web GUI — Deep Research orchestration
// Four-phase pipeline: Outline → Gather → Synthesize → Assemble
// Produces thorough, multi-source research documents with proper citations.

const { streamChat } = require('./ollama');
const { appendMessage, renameSession } = require('./sessions');
const { spawnAgent } = require('./tools/agents');
const { createArtifact } = require('./tools/artifact');
const delegate = require('./tools/delegate');
const { exportPdf } = require('./tools/export');
const {
  MODEL_CTX,
  RESEARCH_AGENT_MAX_TURNS,
  RESEARCH_MAX_SECTIONS,
  RESEARCH_MAX_SOURCES,
} = require('./constants');

/**
 * Run a deep research pipeline for a user query.
 *
 * @param {Object} opts
 * @param {string} opts.sessionId - Session to write results into
 * @param {string} opts.query - The user's research question
 * @param {string} opts.cwd - Working directory
 * @param {Function} opts.send - Send WS event to client
 * @param {AbortSignal} opts.signal - Abort signal
 * @param {string|false} [opts.useFrontier] - Frontier model name for synthesis ('claude','codex','gemini') or false for local multi-pass
 * @returns {Promise<{response: string}>}
 */
async function runDeepResearch({ sessionId, query, cwd, send, signal, useFrontier }) {
  const startTime = Date.now();

  send({ type: 'research_status', phase: 'starting', message: 'Starting deep research...' });

  // ── Phase 1: Outline ──
  send({ type: 'research_status', phase: 'outline', message: 'Planning research outline...' });

  const outline = await generateOutline(query, cwd, signal);
  if (signal?.aborted) throw new Error('Research cancelled');

  send({
    type: 'research_status',
    phase: 'outline_done',
    message: `Research outline: ${outline.sections.length} sections planned`,
    sections: outline.sections.map(s => s.title),
  });

  // ── Phase 2: Gather ──
  send({ type: 'research_status', phase: 'gather', message: `Researching ${outline.sections.length} sections...` });

  const gathered = [];
  for (let i = 0; i < outline.sections.length; i++) {
    if (signal?.aborted) throw new Error('Research cancelled');

    const section = outline.sections[i];
    send({
      type: 'research_status',
      phase: 'gather_section',
      message: `Researching: ${section.title} (${i + 1}/${outline.sections.length})`,
      current: i + 1,
      total: outline.sections.length,
    });

    const sectionData = await gatherSection(section, query, cwd, (status) => {
      send({ type: 'research_status', phase: 'gather_detail', message: status.message || status.tool || '' });
    });

    gathered.push({
      title: section.title,
      queries: section.queries,
      findings: sectionData,
    });

    send({
      type: 'research_status',
      phase: 'gather_section_done',
      message: `Completed: ${section.title}`,
      current: i + 1,
      total: outline.sections.length,
    });
  }

  // ── Phase 3: Synthesize ──
  let report;
  if (useFrontier) {
    send({ type: 'research_status', phase: 'synthesize', message: `Synthesizing with ${useFrontier} (frontier model)...` });
    if (signal?.aborted) throw new Error('Research cancelled');
    report = await synthesizeWithFrontier(query, outline, gathered, useFrontier, signal, send);
  } else {
    send({ type: 'research_status', phase: 'synthesize', message: 'Multi-pass synthesis: drafting report...' });
    if (signal?.aborted) throw new Error('Research cancelled');
    report = await synthesizeMultiPass(query, outline, gathered, signal, send);
  }

  // ── Phase 4: Assemble ──
  send({ type: 'research_status', phase: 'assemble', message: 'Creating final document...' });

  if (signal?.aborted) throw new Error('Research cancelled');

  const reportTitle = outline.title || `Research: ${query.slice(0, 60)}`;

  const artifactResult = createArtifact({
    title: reportTitle,
    content: report,
    type: 'html',
  });

  // Parse artifact path from result
  const pathMatch = artifactResult.match(/\/artifacts\/[^\s]+\.html/);
  if (pathMatch) {
    send({ type: 'artifact', title: reportTitle, path: pathMatch[0], artifactType: 'html' });
  }

  // ── PDF Export ──
  let pdfPath = null;
  if (pathMatch) {
    send({ type: 'research_status', phase: 'pdf', message: 'Generating PDF...' });
    try {
      const pdfResult = await exportPdf(pathMatch[0], reportTitle);
      pdfPath = pdfResult.webPath;
      send({ type: 'document', title: `${reportTitle} (PDF)`, path: pdfPath, artifactType: 'pdf' });
    } catch (err) {
      console.log(`[research] PDF generation failed: ${err.message}`);
      // Not fatal — the HTML artifact is still available
    }
  }

  // Save the full response to session
  const duration = ((Date.now() - startTime) / 1000).toFixed(0);
  const pdfNote = pdfPath ? ` PDF available at ${pdfPath}.` : '';
  const summary = `Research complete: "${reportTitle}" — ${outline.sections.length} sections, ${duration}s elapsed.${pdfNote}`;

  appendMessage(sessionId, {
    role: 'assistant',
    content: summary,
  });

  send({
    type: 'research_status',
    phase: 'complete',
    message: summary,
  });

  send({
    type: 'done',
    stats: { duration_s: parseInt(duration), sections: outline.sections.length },
    context: { used: 0, max: MODEL_CTX, pct: 0 },
  });

  // Rename session to reflect the research topic
  renameSession(sessionId, reportTitle);
  send({ type: 'session_renamed', sessionId, name: reportTitle });

  return { response: summary };
}


// ═══════════════════════════════════════════════
// Phase 1: Generate research outline
// ═══════════════════════════════════════════════

async function generateOutline(query, cwd, signal) {
  const messages = [
    {
      role: 'system',
      content: `You are a research planner. Given a research question, create a structured outline for a comprehensive research document.

Output ONLY valid JSON with this structure:
{
  "title": "Document Title",
  "sections": [
    {
      "title": "Section Title",
      "description": "What this section should cover",
      "queries": ["search query 1", "search query 2", "search query 3"]
    }
  ]
}

RULES:
- Create ${RESEARCH_MAX_SECTIONS} or fewer focused sections
- Each section needs 2-${RESEARCH_MAX_SOURCES} specific web search queries
- Queries should be diverse — different angles, sources, data types
- Include sections for: background/context, current state, key findings, analysis, implications
- The title should be descriptive and professional
- Output ONLY the JSON — no markdown, no explanation, no code fences`,
    },
    { role: 'user', content: query },
  ];

  const result = await streamChat({ messages, maxTokens: 4096, signal });

  // Parse the outline from the response
  let outline;
  try {
    // Try to extract JSON from the response (model may wrap in code fences)
    let jsonStr = result.response || '';
    const fenceMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (fenceMatch) jsonStr = fenceMatch[1];
    // Also try to find bare JSON object
    const objMatch = jsonStr.match(/\{[\s\S]*\}/);
    if (objMatch) jsonStr = objMatch[0];
    outline = JSON.parse(jsonStr);
  } catch (err) {
    // Fallback: create a basic outline
    outline = {
      title: `Research: ${query.slice(0, 80)}`,
      sections: [
        { title: 'Overview', description: 'Background and context', queries: [query] },
        { title: 'Key Findings', description: 'Main research findings', queries: [`${query} latest research`, `${query} key findings`] },
        { title: 'Analysis', description: 'In-depth analysis', queries: [`${query} analysis`, `${query} implications`] },
      ],
    };
  }

  // Enforce limits
  if (outline.sections.length > RESEARCH_MAX_SECTIONS) {
    outline.sections = outline.sections.slice(0, RESEARCH_MAX_SECTIONS);
  }
  for (const section of outline.sections) {
    if (section.queries && section.queries.length > RESEARCH_MAX_SOURCES) {
      section.queries = section.queries.slice(0, RESEARCH_MAX_SOURCES);
    }
  }

  return outline;
}


// ═══════════════════════════════════════════════
// Phase 2: Gather data for a single section
// ═══════════════════════════════════════════════

async function gatherSection(section, originalQuery, cwd, onStatus) {
  const agentTask = `You are researching the following section for a comprehensive report.

RESEARCH TOPIC: ${originalQuery}
SECTION: ${section.title}
SECTION GOAL: ${section.description}
SUGGESTED SEARCH QUERIES: ${section.queries.join(', ')}

INSTRUCTIONS:
1. Use web_search with each suggested query (and add your own if needed)
2. Use web_fetch to read the most relevant pages from search results
3. Extract SPECIFIC facts, data, statistics, quotes, and source URLs
4. Collect at least 3 distinct sources

Return your findings as structured text with:
- Key facts and data points (with specific numbers, dates, names)
- Direct quotes where relevant (with attribution)
- Source URLs for each finding
- Any data suitable for charts or tables

Be thorough — this data will be used to write a detailed report section. Do not summarize prematurely; include raw findings.`;

  const overrides = {
    maxTurns: RESEARCH_AGENT_MAX_TURNS,
    allowedTools: [
      'bash', 'read_file', 'list_dir', 'glob', 'grep',
      'web_fetch', 'web_search', 'memory_search', 'memory_list',
      'system_info',
    ],
    systemPrompt: `You are a deep research agent for PRE (Personal Reasoning Engine). You gather thorough, factual information using web searches and page fetches. Be methodical: search → read → extract facts → repeat. Collect SPECIFIC data, not vague summaries.

RULES:
- Execute ALL suggested search queries
- Fetch and read the top 2-3 results from each search
- Extract specific facts: numbers, dates, names, statistics
- Note the source URL for every fact
- Return raw findings, not a polished summary
- Maximum ${RESEARCH_AGENT_MAX_TURNS} tool calls`,
  };

  const result = await spawnAgent({ task: agentTask }, cwd, onStatus, overrides);
  return result;
}


// ═══════════════════════════════════════════════
// Phase 3a: Multi-pass local synthesis (default)
//   Pass 1 — Draft: generate the full HTML report
//   Pass 2 — Critique: identify gaps, weak sections, missing data
//   Pass 3 — Revise: produce the final improved report
// ═══════════════════════════════════════════════

const REPORT_SYSTEM_PROMPT = `You are a professional report writer for PRE (Personal Reasoning Engine). You synthesize raw research data into polished, comprehensive HTML documents.

OUTPUT FORMAT:
Create a COMPLETE, standalone HTML document with embedded CSS. The document should be professional and well-formatted.

STRUCTURE:
- Clean, modern HTML with embedded <style> in <head>
- Professional typography (system font stack)
- Responsive layout
- Table of contents with anchor links
- Each section: title, 2-4 substantive paragraphs with SPECIFIC facts from the research
- Citations as numbered footnotes or inline [Source: URL]
- Data tables where appropriate (with real data from research, NOT placeholders)
- A Sources/References section at the end with all URLs

QUALITY STANDARDS:
- USE SPECIFIC DATA from the research findings: names, dates, numbers, statistics, quotes
- Every claim must be backed by data from the research
- Do not invent or assume data — only use what was gathered
- Include analysis and insights, not just data regurgitation
- Professional tone appropriate for a research report
- Minimum 2-3 paragraphs per section with substantive content

Output ONLY the complete HTML document. No markdown wrapping, no explanation.`;

function buildGatherContext(gathered) {
  let ctx = '';
  for (const section of gathered) {
    ctx += `\n\n== SECTION: ${section.title} ==\n`;
    ctx += `Search queries used: ${(section.queries || []).join(', ')}\n`;
    ctx += `Findings:\n${section.findings}\n`;
  }
  if (ctx.length > 40000) {
    ctx = ctx.slice(0, 40000) + '\n\n[... truncated for length ...]';
  }
  return ctx;
}

function buildReportUserPrompt(query, outline, gatherContext) {
  return `Write a comprehensive research report on: "${query}"

REPORT TITLE: ${outline.title}

SECTIONS TO INCLUDE:
${outline.sections.map((s, i) => `${i + 1}. ${s.title}: ${s.description}`).join('\n')}

RAW RESEARCH DATA:
${gatherContext}

Create the full HTML report now. Use ALL the specific data gathered above.`;
}

function cleanHtml(raw, title) {
  let html = raw;
  const fenceMatch = html.match(/```(?:html)?\s*([\s\S]*?)```/);
  if (fenceMatch) html = fenceMatch[1];
  if (!html.trim().toLowerCase().startsWith('<!doctype') && !html.trim().toLowerCase().startsWith('<html')) {
    html = `<!DOCTYPE html>\n<html>\n<head><meta charset="UTF-8"><title>${title}</title></head>\n<body>\n${html}\n</body>\n</html>`;
  }
  return html;
}

async function synthesizeMultiPass(query, outline, gathered, signal, send) {
  const gatherContext = buildGatherContext(gathered);

  // ── Pass 1: Draft ──
  send({ type: 'research_status', phase: 'synthesize_progress', message: 'Pass 1/3: Writing first draft...' });

  let draftHtml = '';
  const draftResult = await streamChat({
    messages: [
      { role: 'system', content: REPORT_SYSTEM_PROMPT },
      { role: 'user', content: buildReportUserPrompt(query, outline, gatherContext) },
    ],
    maxTokens: 16384,
    signal,
    onToken: (event) => {
      if (event.type === 'token') {
        draftHtml += event.content || '';
        send({ type: 'research_status', phase: 'synthesize_progress', message: `Pass 1/3: Drafting... (${draftHtml.length} chars)` });
      }
    },
  });
  draftHtml = draftResult.response || draftHtml;

  if (signal?.aborted) throw new Error('Research cancelled');

  // ── Pass 2: Critique ──
  send({ type: 'research_status', phase: 'synthesize_progress', message: 'Pass 2/3: Reviewing and critiquing draft...' });

  let critique = '';
  const critiqueResult = await streamChat({
    messages: [
      {
        role: 'system',
        content: `You are a senior research editor. You review draft reports and provide detailed, actionable critique to improve quality.

Evaluate the draft on these dimensions:
1. DATA USAGE — Are specific facts, numbers, dates, and statistics from the research included? Are any findings from the raw data NOT used that should be?
2. COMPLETENESS — Does every section have substantial content (2-4 paragraphs)? Are any sections thin or vague?
3. CITATIONS — Are sources properly attributed? Are any claims unsourced?
4. ANALYSIS — Does the report merely list facts, or does it provide insight and synthesis?
5. STRUCTURE — Is the table of contents complete? Do section transitions flow logically?
6. HTML QUALITY — Is the CSS professional? Are there formatting issues?

For each issue found, cite the specific section and provide the EXACT improvement needed — referencing data from the raw research that should be incorporated.

Be thorough and specific. This critique drives the final revision pass.`,
      },
      {
        role: 'user',
        content: `DRAFT REPORT:\n${draftHtml}\n\nRAW RESEARCH DATA (for verifying completeness):\n${gatherContext}\n\nProvide your detailed critique now.`,
      },
    ],
    maxTokens: 8192,
    signal,
    onToken: (event) => {
      if (event.type === 'token') {
        critique += event.content || '';
        send({ type: 'research_status', phase: 'synthesize_progress', message: `Pass 2/3: Critiquing... (${critique.length} chars)` });
      }
    },
  });
  critique = critiqueResult.response || critique;

  if (signal?.aborted) throw new Error('Research cancelled');

  // ── Pass 3: Revise ──
  send({ type: 'research_status', phase: 'synthesize_progress', message: 'Pass 3/3: Revising final report...' });

  let revisedHtml = '';
  const reviseResult = await streamChat({
    messages: [
      {
        role: 'system',
        content: REPORT_SYSTEM_PROMPT + `\n\nYou are producing the FINAL REVISION of this report. A senior editor has reviewed the first draft and provided detailed critique. You MUST address every point in the critique.

Key priorities for revision:
- Incorporate ALL specific data points the critique identifies as missing
- Expand any sections flagged as thin or vague
- Add proper citations where the critique notes they are missing
- Improve analysis and insight where flagged
- Fix any HTML/CSS issues noted
- The final document must be substantially improved over the draft

Output the COMPLETE revised HTML document.`,
      },
      {
        role: 'user',
        content: `ORIGINAL DRAFT:\n${draftHtml}\n\nEDITOR CRITIQUE:\n${critique}\n\nRAW RESEARCH DATA (for incorporating missing facts):\n${gatherContext}\n\nProduce the complete, revised HTML report now. Address every critique point.`,
      },
    ],
    maxTokens: 16384,
    signal,
    onToken: (event) => {
      if (event.type === 'token') {
        revisedHtml += event.content || '';
        send({ type: 'research_status', phase: 'synthesize_progress', message: `Pass 3/3: Revising... (${revisedHtml.length} chars)` });
      }
    },
  });
  revisedHtml = reviseResult.response || revisedHtml;

  return cleanHtml(revisedHtml, outline.title);
}


// ═══════════════════════════════════════════════
// Phase 3b: Frontier model synthesis (hybrid mode)
//   Sends all gathered data to a frontier model (Claude, Codex, Gemini)
//   for a single high-quality synthesis pass with deep reasoning.
// ═══════════════════════════════════════════════

async function synthesizeWithFrontier(query, outline, gathered, frontierModel, signal, send) {
  const gatherContext = buildGatherContext(gathered);

  const prompt = `You are producing a comprehensive, professional research report. I have gathered raw research data from multiple web sources on the topic below. Synthesize ALL of this data into a polished, well-structured HTML document.

RESEARCH TOPIC: "${query}"
REPORT TITLE: ${outline.title}

SECTIONS TO INCLUDE:
${outline.sections.map((s, i) => `${i + 1}. ${s.title}: ${s.description}`).join('\n')}

RAW RESEARCH DATA:
${gatherContext}

REQUIREMENTS:
- Output a COMPLETE standalone HTML document with embedded <style> CSS
- Professional typography (system font stack), responsive layout
- Table of contents with anchor links to each section
- Each section: 2-4 substantive paragraphs with SPECIFIC facts, numbers, dates, statistics from the research data above
- Citations as numbered footnotes or inline [Source: URL] — every claim backed by data
- Data tables where the research contains structured data
- A Sources/References section at the end listing all URLs
- Include analysis, insights, and cross-references between sections — not just data listing
- Do NOT invent or assume data — use only what is provided above

Think deeply about how to best organize and present this information. Take your time to produce a thorough, high-quality document.

Output ONLY the complete HTML document. No markdown wrapping, no preamble, no explanation.`;

  const synthStart = Date.now();
  send({ type: 'research_status', phase: 'synthesize_progress', message: `Frontier synthesis via ${frontierModel} (this may take several minutes)...` });

  let responseText = '';
  // Elapsed-time ticker so users know it's still working
  const ticker = setInterval(() => {
    const elapsed = Math.round((Date.now() - synthStart) / 1000);
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    const timeStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
    const charsNote = responseText.length > 0 ? ` — ${responseText.length} chars received` : ' — waiting for response';
    send({ type: 'research_status', phase: 'synthesize_progress', message: `Frontier synthesis via ${frontierModel} (${timeStr}${charsNote})...` });
  }, 5000);

  try {
    const result = await delegate.execute(frontierModel, prompt, {
      signal,
      timeout: 900000, // 15 minutes — frontier models with deep thinking need time
      onToken: (chunk) => {
        responseText += chunk;
      },
    });
    responseText = result.response || responseText;
    if (result.partial) {
      console.log(`[research] Frontier synthesis returned partial response (${responseText.length} chars)`);
    }
  } catch (err) {
    if (responseText.length > 500) {
      // Partial response is usable — proceed with what we have
      console.log(`[research] Frontier synthesis partial (${responseText.length} chars): ${err.message}`);
    } else {
      clearInterval(ticker);
      throw new Error(`Frontier synthesis failed: ${err.message}`);
    }
  } finally {
    clearInterval(ticker);
  }

  const synthDuration = Math.round((Date.now() - synthStart) / 1000);
  send({ type: 'research_status', phase: 'synthesize_progress', message: `Frontier synthesis complete (${synthDuration}s, ${responseText.length} chars)` });

  return cleanHtml(responseText, outline.title);
}

module.exports = { runDeepResearch };
