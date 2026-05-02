// PRE Web GUI — Dynamic Virtual Tools (Self-Architecting)
// Allows the model to create, manage, and execute custom tools at runtime.
// Custom tools can be:
//   1. Prompt templates — parameterized prompts executed through the tool loop
//   2. Workflow-backed — replay a recorded workflow with parameter substitution
//   3. Multi-step — chain multiple existing tools in sequence
//
// Storage: ~/.pre/custom_tools/
// Each tool is a JSON file with: name, description, parameters, implementation

const fs = require('fs');
const path = require('path');
const { PRE_DIR } = require('./constants');

const CUSTOM_TOOLS_DIR = path.join(PRE_DIR, 'custom_tools');
if (!fs.existsSync(CUSTOM_TOOLS_DIR)) fs.mkdirSync(CUSTOM_TOOLS_DIR, { recursive: true });

// ── Persistence ────────────────────────────────────────────────────────────

function loadCustomTools() {
  if (!fs.existsSync(CUSTOM_TOOLS_DIR)) return [];
  return fs.readdirSync(CUSTOM_TOOLS_DIR)
    .filter(f => f.endsWith('.json'))
    .map(f => {
      try {
        const data = JSON.parse(fs.readFileSync(path.join(CUSTOM_TOOLS_DIR, f), 'utf-8'));
        data._filename = f;
        return data;
      } catch { return null; }
    })
    .filter(Boolean);
}

function saveCustomTool(tool) {
  const filename = `${tool.name}.json`;
  const filePath = path.join(CUSTOM_TOOLS_DIR, filename);
  fs.writeFileSync(filePath, JSON.stringify(tool, null, 2));
  return filePath;
}

function deleteCustomTool(name) {
  const filename = `${name}.json`;
  const filePath = path.join(CUSTOM_TOOLS_DIR, filename);
  if (!fs.existsSync(filePath)) return false;
  fs.unlinkSync(filePath);
  return true;
}

// ── Tool Definition Builder ────────────────────────────────────────────────

/**
 * Build Ollama-compatible tool definitions for all custom tools.
 * These get appended to the main tool list in tools-defs.js.
 */
function buildCustomToolDefs() {
  const tools = loadCustomTools();
  return tools.map(t => ({
    type: 'function',
    function: {
      name: `custom_${t.name}`,
      description: `[Custom Tool] ${t.description}`,
      parameters: {
        type: 'object',
        properties: buildParameterProperties(t.parameters || []),
        required: (t.parameters || []).filter(p => p.required).map(p => p.name),
      },
    },
  }));
}

function buildParameterProperties(params) {
  const props = {};
  for (const p of params) {
    props[p.name] = {
      type: p.type || 'string',
      description: p.description || p.name,
    };
    if (p.enum) props[p.name].enum = p.enum;
    if (p.default !== undefined) props[p.name].default = p.default;
  }
  return props;
}

// ── Execution ──────────────────────────────────────────────────────────────

/**
 * Execute a custom tool by name.
 * Substitutes parameters into the tool's implementation template.
 */
async function executeCustomTool(name, args) {
  const tools = loadCustomTools();
  const tool = tools.find(t => t.name === name);
  if (!tool) return `Error: custom tool "${name}" not found`;

  // Track usage for self-improving skills analytics
  tool.usage_count = (tool.usage_count || 0) + 1;
  tool.last_used = new Date().toISOString();
  try { saveCustomTool(tool); } catch {}

  const impl = tool.implementation;
  if (!impl) return `Error: custom tool "${name}" has no implementation`;

  switch (impl.type) {
    case 'prompt': {
      // Prompt template: substitute parameters and return for model execution
      let prompt = impl.template;
      for (const [key, value] of Object.entries(args)) {
        prompt = prompt.replace(new RegExp(`\\$\\{${key}\\}`, 'g'), String(value));
      }
      return `[Custom Tool Prompt]\n${prompt}`;
    }

    case 'workflow': {
      // Delegate to workflow replay with parameter substitution
      const workflowTool = require('./tools/workflow');
      return workflowTool.workflow({
        action: 'replay',
        name: impl.workflow_name,
        speed: impl.speed || 1.0,
        ...args,
      });
    }

    case 'chain': {
      // Execute a sequence of tool calls
      const steps = impl.steps || [];
      const results = [];
      for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        // Substitute parameters in step args
        const stepArgs = {};
        for (const [key, value] of Object.entries(step.args || {})) {
          if (typeof value === 'string') {
            let resolved = value;
            for (const [paramKey, paramValue] of Object.entries(args)) {
              resolved = resolved.replace(new RegExp(`\\$\\{${paramKey}\\}`, 'g'), String(paramValue));
            }
            // Also substitute previous step results
            for (let j = 0; j < results.length; j++) {
              resolved = resolved.replace(new RegExp(`\\$\\{step${j + 1}\\}`, 'g'), String(results[j]));
            }
            stepArgs[key] = resolved;
          } else {
            stepArgs[key] = value;
          }
        }
        results.push(`Step ${i + 1} (${step.tool}): queued with args ${JSON.stringify(stepArgs)}`);
      }
      return `[Custom Tool Chain]\nThis tool chains ${steps.length} steps. Execute these in order:\n${
        steps.map((s, i) => `${i + 1}. Call "${s.tool}" with: ${JSON.stringify(s.args)}`).join('\n')
      }\n\nSubstituted parameters:\n${results.join('\n')}`;
    }

    default:
      return `Error: unknown implementation type "${impl.type}" for custom tool "${name}"`;
  }
}

/**
 * Check if a tool name is a custom tool.
 */
function isCustomTool(name) {
  if (!name.startsWith('custom_')) return false;
  const baseName = name.slice(7); // remove 'custom_' prefix
  const tools = loadCustomTools();
  return tools.some(t => t.name === baseName);
}

// ── Tool Dispatcher (management actions) ───────────────────────────────────

function customToolDispatcher(args) {
  const action = (args.action || '').toLowerCase();

  switch (action) {
    case 'create': case 'add': return createTool(args);
    case 'list': case 'ls': return listTools();
    case 'show': case 'inspect': return showTool(args);
    case 'delete': case 'rm': case 'remove': return removeTool(args);
    case 'from_workflow': return createFromWorkflow(args);
    default:
      return 'Error: unknown custom_tool action. Available: create, list, show, delete, from_workflow';
  }
}

function createTool(args) {
  if (!args.name) return 'Error: name is required';
  if (!args.description) return 'Error: description is required';

  const name = args.name.replace(/[^a-zA-Z0-9_]/g, '_').toLowerCase();

  // Parse parameters
  let parameters = [];
  if (args.parameters) {
    if (typeof args.parameters === 'string') {
      try { parameters = JSON.parse(args.parameters); } catch {
        return 'Error: parameters must be a valid JSON array of {name, type, description, required}';
      }
    } else if (Array.isArray(args.parameters)) {
      parameters = args.parameters;
    }
  }

  // Build implementation
  let implementation;
  if (args.template) {
    implementation = { type: 'prompt', template: args.template };
  } else if (args.workflow_name) {
    implementation = { type: 'workflow', workflow_name: args.workflow_name, speed: args.speed || 1.0 };
  } else if (args.steps) {
    let steps = args.steps;
    if (typeof steps === 'string') {
      try { steps = JSON.parse(steps); } catch {
        return 'Error: steps must be a valid JSON array of {tool, args}';
      }
    }
    implementation = { type: 'chain', steps };
  } else {
    return 'Error: one of template, workflow_name, or steps is required for the implementation';
  }

  const tool = {
    name,
    description: args.description,
    parameters,
    implementation,
    created: new Date().toISOString(),
    version: 1,
  };

  const filePath = saveCustomTool(tool);

  const paramDesc = parameters.length > 0
    ? `\n  Parameters: ${parameters.map(p => `${p.name} (${p.type || 'string'})`).join(', ')}`
    : '';

  return [
    `Custom tool created:`,
    `  Name: custom_${name}`,
    `  Type: ${implementation.type}`,
    `  Description: ${args.description}`,
    paramDesc,
    `  Path: ${filePath}`,
    ``,
    `The tool is now available as "custom_${name}" in your tool list.`,
  ].filter(Boolean).join('\n');
}

function listTools() {
  const tools = loadCustomTools();
  if (tools.length === 0) {
    return 'No custom tools defined. Use action "create" to build one, or "from_workflow" to convert a workflow.';
  }

  const lines = [`${tools.length} custom tool(s):`, ''];
  for (const t of tools) {
    const params = (t.parameters || []).map(p => p.name).join(', ');
    lines.push(`  **custom_${t.name}** [${t.implementation?.type || '?'}] — ${t.description}`);
    if (params) lines.push(`    Parameters: ${params}`);
    lines.push(`    Created: ${t.created?.slice(0, 10) || '?'}`);
    lines.push('');
  }
  return lines.join('\n');
}

function showTool(args) {
  if (!args.name) return 'Error: name is required';
  const name = args.name.replace(/^custom_/, '');
  const tools = loadCustomTools();
  const tool = tools.find(t => t.name === name);
  if (!tool) return `Custom tool "${name}" not found.`;

  const lines = [
    `**Custom Tool: custom_${tool.name}**`,
    `Description: ${tool.description}`,
    `Type: ${tool.implementation?.type || '?'}`,
    `Created: ${tool.created || '?'}`,
    `Version: ${tool.version || 1}`,
    '',
    '**Parameters:**',
    ...(tool.parameters || []).map(p =>
      `  - ${p.name} (${p.type || 'string'})${p.required ? ' [required]' : ''}: ${p.description || ''}`
    ),
    '',
    '**Implementation:**',
    JSON.stringify(tool.implementation, null, 2),
  ];
  return lines.join('\n');
}

function removeTool(args) {
  if (!args.name) return 'Error: name is required';
  const name = args.name.replace(/^custom_/, '');
  if (deleteCustomTool(name)) {
    return `Deleted custom tool "${name}".`;
  }
  return `Custom tool "${name}" not found.`;
}

/**
 * Create a custom tool from an existing workflow.
 * Analyzes the workflow steps and generates parameter definitions.
 */
function createFromWorkflow(args) {
  if (!args.workflow_name) return 'Error: workflow_name is required';
  if (!args.name) return 'Error: name is required for the custom tool';

  const workflowModule = require('./tools/workflow');
  const workflows = workflowModule.workflow({ action: 'list' });

  // Verify workflow exists by trying to show it
  const details = workflowModule.workflow({ action: 'show', name: args.workflow_name });
  if (details.includes('not found')) return `Workflow "${args.workflow_name}" not found.`;

  return createTool({
    name: args.name,
    description: args.description || `Replay workflow: ${args.workflow_name}`,
    parameters: args.parameters || '[]',
    workflow_name: args.workflow_name,
    speed: args.speed || 1.0,
  });
}

module.exports = {
  customTool: customToolDispatcher,
  executeCustomTool,
  isCustomTool,
  buildCustomToolDefs,
  loadCustomTools,
  saveCustomTool,
  deleteCustomTool,
};
