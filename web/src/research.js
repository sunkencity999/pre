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
      content: `You are a research planner. Given a research question, create a structured outline for a comprehensive, in-depth research document.

Output ONLY valid JSON with this structure:
{
  "title": "Document Title",
  "sections": [
    {
      "title": "Section Title",
      "description": "Detailed description of what this section should cover, including specific subtopics, data tables to include, and comparison angles to explore",
      "queries": ["search query 1", "search query 2", "search query 3", "search query 4"]
    }
  ]
}

RULES:
- Create ${RESEARCH_MAX_SECTIONS} focused sections (use all ${RESEARCH_MAX_SECTIONS})
- Each section needs 3-${RESEARCH_MAX_SOURCES} specific web search queries — be creative and varied
- Queries should target different angles: official sources, reviews, technical specs, comparisons, community discussion, news articles
- Section descriptions must be DETAILED (2-3 sentences each) listing specific subtopics, data points to gather, and any tables or comparisons that would enrich the section
- Include sections for: background/history, current state, detailed technical/feature analysis, comparative analysis (vs alternatives or predecessors), community/market reception, future outlook
- At least one section should be explicitly about comparison or competitive analysis
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
  const agentTask = `You are researching the following section for a comprehensive, in-depth report.

RESEARCH TOPIC: ${originalQuery}
SECTION: ${section.title}
SECTION GOAL: ${section.description}
SUGGESTED SEARCH QUERIES: ${section.queries.join(', ')}

INSTRUCTIONS:
1. Execute EVERY suggested search query — use web_search for each one
2. For the top 2-3 results from EACH search, fetch the full page content with web_fetch (or browser for JS-heavy pages)
3. Read thoroughly — extract EVERY useful specific fact, not just the first paragraph
4. If you find links to additional relevant pages (e.g., spec sheets, patch notes, reviews), follow them
5. Collect at least 4-5 distinct sources minimum

WHAT TO EXTRACT (be exhaustive):
- Every specific number, date, price, measurement, version, spec
- Direct quotes from developers, reviewers, or official sources (with attribution)
- Comparison data: how does this compare to alternatives/competitors/predecessors?
- Structured data suitable for tables (feature lists, specifications, timelines)
- Community reception: ratings, user counts, sentiment
- Source URL for EVERY fact (format: [Source: URL])

OUTPUT FORMAT:
Return raw findings organized by source. For each source:
SOURCE: [URL]
FACTS:
- fact 1
- fact 2
QUOTES:
- "quote" — attribution
TABLE DATA:
- any structured/tabular data found

Be EXHAUSTIVE — this data directly determines report quality. Every fact you miss is a fact missing from the final report. Do not summarize prematurely; include all raw data.`;

  const overrides = {
    maxTurns: RESEARCH_AGENT_MAX_TURNS,
    // Denylist: block destructive/recursive tools, allow everything else
    // (browser, web_search, cloud integrations, etc. all available for research)
    deniedTools: [
      'file_write', 'file_edit', 'process_kill', 'memory_delete',
      'cron', 'spawn_agent', 'spawn_multi', 'image_generate',
    ],
    systemPrompt: `You are a deep research agent for PRE (Personal Reasoning Engine). You gather thorough, factual information using all available tools. Be methodical and use the best tool for each task:

TOOL STRATEGY:
- Use web_search (if available) for broad topic queries
- Use browser to navigate and read JavaScript-heavy or interactive pages
- Use web_fetch for direct URL retrieval (articles, APIs, docs)
- Use bash for curl, jq, or other CLI data gathering when useful
- Cloud tools (github, jira, slack, etc.) are available if the research topic involves those platforms

RULES:
- Execute ALL suggested search queries using the best available tool
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

// ── CSS template embedded in every local report ──
// Providing this eliminates CSS bugs from the model and ensures professional styling.
const REPORT_CSS = `
  :root { --bg: #ffffff; --fg: #1a1a2e; --accent: #2563eb; --accent-light: #dbeafe;
    --border: #e2e8f0; --muted: #64748b; --callout-bg: #f0f9ff; --callout-border: #38bdf8;
    --blockquote-border: #6366f1; --table-stripe: #f8fafc; --shadow: rgba(0,0,0,0.06); }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #0f172a; --fg: #e2e8f0; --accent: #60a5fa; --accent-light: #1e3a5f;
      --border: #334155; --muted: #94a3b8; --callout-bg: #1e293b; --callout-border: #38bdf8;
      --blockquote-border: #818cf8; --table-stripe: #1e293b; --shadow: rgba(0,0,0,0.3); } }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; font-size: 17px;
    line-height: 1.7; color: var(--fg); background: var(--bg); padding: 2rem 1rem; }
  .container { max-width: 52rem; margin: 0 auto; }
  h1 { font-family: Georgia, 'Times New Roman', serif; font-size: 2.2rem; font-weight: 700;
    margin-bottom: 0.5rem; line-height: 1.25; border-bottom: 3px solid var(--accent); padding-bottom: 0.5rem; }
  .subtitle { color: var(--muted); font-size: 0.95rem; margin-bottom: 2rem; }
  h2 { font-family: Georgia, 'Times New Roman', serif; font-size: 1.6rem; font-weight: 600;
    margin: 2.5rem 0 1rem; padding-bottom: 0.3rem; border-bottom: 1px solid var(--border); }
  h3 { font-size: 1.2rem; font-weight: 600; margin: 1.5rem 0 0.75rem; }
  p { max-width: 72ch; margin-bottom: 1rem; }
  a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
  /* Table of Contents */
  .toc { background: var(--callout-bg); border: 1px solid var(--border); border-radius: 8px;
    padding: 1.25rem 1.5rem; margin: 1.5rem 0 2rem; }
  .toc h2 { font-size: 1.1rem; margin: 0 0 0.75rem; border: none; }
  .toc ol { padding-left: 1.5rem; } .toc li { margin: 0.3rem 0; }
  .toc a { font-weight: 500; }
  /* Data tables */
  table { width: 100%; border-collapse: collapse; margin: 1.5rem 0; font-size: 0.95rem;
    box-shadow: 0 1px 3px var(--shadow); border-radius: 6px; overflow: hidden; }
  thead { background: var(--accent); color: #fff; }
  th { padding: 0.65rem 1rem; text-align: left; font-weight: 600; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 0.03em; }
  td { padding: 0.6rem 1rem; border-bottom: 1px solid var(--border); }
  tbody tr:nth-child(even) { background: var(--table-stripe); }
  tbody tr:hover { background: var(--accent-light); }
  /* Callout boxes */
  .callout { background: var(--callout-bg); border-left: 4px solid var(--callout-border);
    border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; margin: 1.5rem 0; }
  .callout strong { display: block; margin-bottom: 0.3rem; color: var(--accent); }
  /* Blockquotes */
  blockquote { border-left: 4px solid var(--blockquote-border); padding: 0.75rem 1.25rem;
    margin: 1.25rem 0; font-style: italic; color: var(--muted); background: var(--callout-bg);
    border-radius: 0 6px 6px 0; }
  blockquote cite { display: block; font-style: normal; font-size: 0.85rem; margin-top: 0.5rem; color: var(--accent); }
  /* Footnotes */
  .footnotes { margin-top: 3rem; padding-top: 1.5rem; border-top: 2px solid var(--border); font-size: 0.9rem; }
  .footnotes ol { padding-left: 1.5rem; } .footnotes li { margin: 0.5rem 0; }
  sup a { color: var(--accent); font-weight: 600; text-decoration: none; font-size: 0.75rem; }
  /* Back to top */
  .back-to-top { display: inline-block; margin-top: 1rem; font-size: 0.85rem; color: var(--muted); }
  /* Print */
  @media print { body { font-size: 11pt; padding: 0; }
    .back-to-top, .toc { break-inside: avoid; } table { font-size: 9pt; }
    h2 { break-after: avoid; } a { color: inherit; } }
`;

const REPORT_SYSTEM_PROMPT = `You are a professional report writer for PRE (Personal Reasoning Engine). You synthesize raw research data into polished, comprehensive HTML documents.

OUTPUT FORMAT:
Create a COMPLETE standalone HTML document. The CSS is pre-built — you MUST use the exact stylesheet provided below by embedding it in a <style> tag. Focus your effort on writing RICH, DETAILED HTML content.

PRE-BUILT CSS — embed this exactly in <style>:
${REPORT_CSS}

HTML STRUCTURE REQUIRED (use these exact elements and classes):
- Wrap all content in <div class="container">
- <h1> for the report title, followed by <p class="subtitle"> with date and source count
- <nav class="toc"> containing <h2>Contents</h2> and <ol> with <li><a href="#section-N"> links
- Each section: <h2 id="section-N"> title, then 3-5 detailed paragraphs
- Use <h3> for subsections within sections
- <div class="callout"><strong>Key Insight:</strong> text</div> for important takeaways (at least 2-3 per report)
- <blockquote>quote text<cite>— Attribution, Source</cite></blockquote> for notable quotes
- <table> with <thead>/<tbody> for any structured data, comparisons, or specifications
- Footnote references as <sup><a href="#fn-N">[N]</a></sup> in text
- <section class="footnotes"> at the end with <ol> of <li id="fn-N"> entries
- <a href="#top" class="back-to-top">↑ Back to top</a> after each major section

CONTENT QUALITY — THIS IS CRITICAL:
- Write 3-5 SUBSTANTIVE paragraphs per section, each 4-8 sentences long
- USE EVERY specific fact, number, date, statistic from the research data — do not leave data on the table
- Include comparison/analysis tables where you can contrast items (e.g., feature comparisons, timelines, specifications)
- Add cross-references between sections ("As discussed in Section 3, ...")
- Provide analytical insight: don't just state facts, explain significance and implications
- Include at least one callout box per section highlighting a key takeaway
- Use blockquotes for direct quotes from developers, press, or official sources
- Every factual claim needs a footnote reference to its source URL
- The Sources section must list EVERY URL from the research data with descriptive titles
- Target report length: 40,000-60,000 characters of HTML (this is a COMPREHENSIVE report)

Output ONLY the complete HTML document starting with <!DOCTYPE html>. No markdown, no explanation.`;

function buildGatherContext(gathered) {
  let ctx = '';
  for (const section of gathered) {
    ctx += `\n\n== SECTION: ${section.title} ==\n`;
    ctx += `Search queries used: ${(section.queries || []).join(', ')}\n`;
    ctx += `Findings:\n${section.findings}\n`;
  }
  if (ctx.length > 120000) {
    ctx = ctx.slice(0, 120000) + '\n\n[... truncated for length ...]';
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
    html = `<!DOCTYPE html>\n<html>\n<head><meta charset="UTF-8"><title>${title}</title><style>${REPORT_CSS}</style></head>\n<body>\n${html}\n</body>\n</html>`;
  }
  // If the model produced a full HTML doc but forgot the CSS, inject it
  if (html.includes('<head>') && !html.includes('--callout-bg')) {
    html = html.replace('</head>', `<style>${REPORT_CSS}</style>\n</head>`);
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
    maxTokens: 65536,
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
        content: `You are a senior research editor. You review draft reports and provide detailed, actionable critique to improve quality. The revision author will use your critique to produce the final report — be SPECIFIC and THOROUGH.

Evaluate the draft on these dimensions, flagging EVERY issue:

1. DATA COMPLETENESS — List every specific fact, number, statistic, date, or quote from the raw research data that is NOT present in the draft. Quote the missing data verbatim so the revision author can paste it in.

2. SECTION DEPTH — Each section needs 3-5 paragraphs of 4-8 sentences each. Flag any section with fewer than 3 paragraphs or with vague/generic content. State exactly what additional content each thin section needs.

3. CITATIONS — Every factual claim needs a footnote (<sup>[N]</sup>). Count how many footnotes exist vs. how many should exist. List uncited claims.

4. RICH HTML FEATURES — Check for presence of:
   - Callout boxes (<div class="callout">) — need at least 2-3 across the report. List sections that would benefit from one.
   - Blockquotes with <cite> — at least 1-2 for notable quotes. List good candidates from the raw data.
   - Comparison/data tables — list any structured data in the raw research that should be in a table but isn't.
   - Cross-references between sections — flag opportunities where one section should reference another.

5. ANALYSIS DEPTH — Does the report explain WHY facts matter, or just list them? Flag sections that read like bullet points converted to prose. Suggest specific analytical angles.

6. REPORT LENGTH — A comprehensive report should be 40,000-60,000 characters. If the draft is under 30,000, flag this and list which sections need the most expansion.

For each issue, state the section, the problem, and the specific fix (including data from the raw research to incorporate). Be relentless — every improvement you flag makes the final report better.`,
      },
      {
        role: 'user',
        content: `DRAFT REPORT:\n${draftHtml}\n\nRAW RESEARCH DATA (for verifying completeness):\n${gatherContext}\n\nProvide your detailed critique now.`,
      },
    ],
    maxTokens: 16384,
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
- Expand any sections flagged as thin or vague — target 3-5 substantial paragraphs per section
- Add footnote citations (<sup><a href="#fn-N">[N]</a></sup>) where the critique notes they are missing
- Add callout boxes (<div class="callout">) for key insights in every section
- Add blockquotes with <cite> tags for notable direct quotes
- Add comparison or data tables wherever structured data exists
- Include cross-references between sections ("As noted in Section N, ...")
- Improve analysis and insight where flagged — explain WHY facts matter, not just WHAT they are
- The final document must be 40,000+ characters of HTML — this is a comprehensive deep-dive

Output the COMPLETE revised HTML document starting with <!DOCTYPE html>.`,
      },
      {
        role: 'user',
        content: `ORIGINAL DRAFT:\n${draftHtml}\n\nEDITOR CRITIQUE:\n${critique}\n\nRAW RESEARCH DATA (for incorporating missing facts):\n${gatherContext}\n\nProduce the complete, revised HTML report now. Address every critique point. Remember: use the pre-built CSS classes (callout, toc, footnotes, back-to-top, blockquote with cite). Target 40,000+ chars total.`,
      },
    ],
    maxTokens: 65536,
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
