// PRE Web GUI — Chronos: Temporal Awareness & Self-Maintenance
//
// Adds temporal intelligence to PRE:
// 1. Memory staleness detection — flags memories that may be outdated
// 2. Temporal context injection — deadlines, time-sensitive project state
// 3. Self-maintenance scheduling — automatic memory auditing and cleanup
// 4. Memory verification — marks memories as verified/stale with timestamps

const fs = require('fs');
const path = require('path');
const { MEMORY_DIR } = require('./constants');
const { getAllMemories, parseFrontmatter, buildFrontmatter } = require('./memory');
const { streamChat } = require('./ollama');

// Staleness thresholds by memory type (in days)
const STALENESS_THRESHOLDS = {
  project: 14,    // Project context changes fast
  reference: 60,  // External references may move
  feedback: 90,   // Work preferences are relatively stable
  user: 180,      // User profile changes slowly
  experience: 120, // Lessons may become outdated as tools/libs change
};

/**
 * Get the age of a memory in days
 */
function ageDays(mtimeMs) {
  return Math.floor((Date.now() - mtimeMs) / (1000 * 60 * 60 * 24));
}

/**
 * Analyze all memories and return a staleness report
 */
function stalenessReport(projectDir) {
  const memories = getAllMemories(projectDir);
  const report = { fresh: [], aging: [], stale: [], unverified: [] };

  for (const m of memories) {
    const age = ageDays(m.mtimeMs);
    const threshold = STALENESS_THRESHOLDS[m.type] || 60;

    // Check verified date from frontmatter
    const verifiedDate = m.verified || m.created;
    const verifiedAge = verifiedDate
      ? Math.floor((Date.now() - new Date(verifiedDate).getTime()) / (1000 * 60 * 60 * 24))
      : age;

    const entry = {
      name: m.name,
      type: m.type,
      filename: m.filename,
      ageDays: age,
      verifiedAge,
      threshold,
      scope: m.scope,
      description: m.description,
    };

    if (verifiedAge > threshold * 2) {
      report.stale.push(entry);
    } else if (verifiedAge > threshold) {
      report.aging.push(entry);
    } else if (!verifiedDate) {
      report.unverified.push(entry);
    } else {
      report.fresh.push(entry);
    }
  }

  return report;
}

/**
 * Mark a memory as verified (updates the 'verified' frontmatter field)
 */
function verifyMemory(filename, dir) {
  const targetDir = dir || MEMORY_DIR;
  const filePath = path.join(targetDir, filename);
  if (!fs.existsSync(filePath)) return { error: 'Memory not found' };

  const content = fs.readFileSync(filePath, 'utf-8');
  const { meta, body } = parseFrontmatter(content);
  meta.verified = new Date().toISOString().slice(0, 10);

  const updated = buildFrontmatter(meta) + '\n\n' + body + '\n';
  fs.writeFileSync(filePath, updated, 'utf-8');
  return { success: true, filename, verified: meta.verified };
}

/**
 * Build temporal context for the system prompt.
 * Includes: current time awareness, approaching deadlines from project memories,
 * stale memory warnings, and experience ledger hints.
 */
function buildTemporalContext(projectDir) {
  const now = new Date();
  const report = stalenessReport(projectDir);

  let ctx = '';

  // Stale memory warnings
  if (report.stale.length > 0) {
    ctx += '<temporal_awareness>\n';
    ctx += `${report.stale.length} memories may be outdated and should be verified before acting on them:\n`;
    for (const m of report.stale.slice(0, 5)) {
      ctx += `- "${m.name}" [${m.type}] — last verified ${m.verifiedAge} days ago (threshold: ${m.threshold}d)\n`;
    }
    if (report.stale.length > 5) {
      ctx += `  ...and ${report.stale.length - 5} more\n`;
    }
    ctx += 'When using information from stale memories, verify against current state first.\n';
    ctx += '</temporal_awareness>\n';
  }

  // Aging warnings (approaching staleness)
  if (report.aging.length > 0) {
    ctx += `<aging_memories count="${report.aging.length}">\n`;
    for (const m of report.aging.slice(0, 3)) {
      ctx += `- "${m.name}" [${m.type}] — ${m.verifiedAge}d since verified\n`;
    }
    ctx += '</aging_memories>\n';
  }

  return ctx;
}

/**
 * Run automated memory maintenance.
 * Called by the cron system or manually.
 * Returns a report of actions taken.
 */
async function runMaintenance(projectDir) {
  const report = stalenessReport(projectDir);
  const actions = [];

  // For stale memories, ask the model to assess if they're still valid
  const staleMemories = report.stale.slice(0, 10); // Cap at 10 per run
  if (staleMemories.length === 0) {
    return { actions: [{ type: 'info', message: 'No stale memories found. All memories are current.' }] };
  }

  // Build a compact list of stale memories for the model to review
  const memoryList = staleMemories.map(m => {
    // Read the full body
    const allMems = getAllMemories(projectDir);
    const full = allMems.find(mem => mem.filename === m.filename);
    return `[${m.filename}] "${m.name}" (${m.type}, ${m.verifiedAge}d old): ${full?.body?.slice(0, 300) || m.description}`;
  }).join('\n\n');

  try {
    const result = await streamChat({
      messages: [
        {
          role: 'system',
          content: `You are PRE's memory maintenance system. Review stale memories and determine which should be:
- KEEP: Still likely valid, mark as re-verified
- UPDATE: Contains outdated info, suggest correction
- DELETE: No longer relevant or useful

Output a JSON array of actions:
[{"filename": "...", "action": "keep|update|delete", "reason": "why"}]

Be conservative — prefer KEEP over DELETE. Only DELETE memories that are clearly about completed/abandoned work.`,
        },
        {
          role: 'user',
          content: `Review these stale memories:\n\n${memoryList}`,
        },
      ],
      maxTokens: 2048,
    });

    const response = result.response || '';
    const jsonMatch = response.match(/\[[\s\S]*?\]/);
    if (jsonMatch) {
      const decisions = JSON.parse(jsonMatch[0]);
      for (const d of decisions) {
        if (!d.filename || !d.action) continue;

        if (d.action === 'keep') {
          const vResult = verifyMemory(d.filename);
          if (vResult.success) {
            actions.push({ type: 'verified', filename: d.filename, reason: d.reason });
          }
        } else if (d.action === 'delete') {
          actions.push({ type: 'flagged_for_deletion', filename: d.filename, reason: d.reason });
          // Don't auto-delete — flag for user review
        } else if (d.action === 'update') {
          actions.push({ type: 'needs_update', filename: d.filename, reason: d.reason });
        }
      }
    }
  } catch (err) {
    actions.push({ type: 'error', message: `Maintenance error: ${err.message}` });
  }

  return {
    reviewed: staleMemories.length,
    fresh: report.fresh.length,
    aging: report.aging.length,
    stale: report.stale.length,
    actions,
  };
}

/**
 * Get a maintenance summary suitable for display
 */
function maintenanceSummary(projectDir) {
  const report = stalenessReport(projectDir);
  const total = report.fresh.length + report.aging.length + report.stale.length + report.unverified.length;

  return {
    total,
    fresh: report.fresh.length,
    aging: report.aging.length,
    stale: report.stale.length,
    unverified: report.unverified.length,
    healthPct: total > 0 ? Math.round(report.fresh.length * 100 / total) : 100,
    oldestStale: report.stale[0] || null,
  };
}

module.exports = {
  stalenessReport,
  verifyMemory,
  buildTemporalContext,
  runMaintenance,
  maintenanceSummary,
  STALENESS_THRESHOLDS,
};
