# PRE — Personal Reasoning Engine

> A fully local agentic assistant that actually works. No cloud. No API keys. No data leaves your machine.

PRE is not a chatbot with tools bolted on. It is a **purpose-built agent** — a single-binary Objective-C application engineered from the ground up around one specific model on one specific platform. Every architectural decision, from socket-level I/O to dynamic memory allocation to prompt compression, exists to make **Google Gemma 4 26B-A4B** run at its absolute ceiling on Apple Silicon. The result is a local agent that doesn't feel local: **~70 tokens/second**, sub-second time to first token, 65K context window, 38+ integrated tools, persistent memory, local image generation, a built-in web GUI, and real agentic workflows — all running on your MacBook.

The reference system is a **MacBook Pro with an M4 Max (128 GB unified memory)**.

---

## Why This Works (When Other Local Agents Don't)

Most local AI tools follow a generic pattern: wrap an OpenAI-compatible API, connect a few tools, hope for the best. The result is sluggish, fragile, and useful mainly as a novelty. PRE takes the opposite approach — it is a **model-specific, platform-specific, vertically integrated agent** — and that specificity is what makes it actually usable for real work.

### The Right Model for Agency

**Gemma 4 26B-A4B** is a Mixture-of-Experts (MoE) architecture: 26 billion total parameters, but only **3.8 billion active per token** (128 experts, 8 active per forward pass). This is the key to the entire system. MoE means:

- **Speed without sacrifice.** You get the knowledge and reasoning quality of a 26B-parameter model at the computational cost of a ~4B model. On Apple Silicon with q4_K_M quantization (~17 GB on disk), this translates to **~70 tokens/second** — fast enough that the agent's tool-call-execute-respond loop feels interactive, not glacial.

- **Context without collapse.** Gemma 4 supports up to 262K tokens natively. PRE allocates a **65K token window** — large enough for extended multi-step workflows (read 20 files, chain tool calls, iterate) while cold-loading in just 1.5 seconds. Models that top out at 8K choke on the second tool call. 65K with auto-compaction means PRE can hold an entire debugging session, a full codebase exploration, or a multi-step refactoring in a single coherent conversation.

- **Strong instruction following.** MoE models with dedicated routing learn to follow structured tool-call formats reliably. Gemma 4 handles PRE's `<tool_call>` JSON format consistently — it doesn't hallucinate partial calls, forget to stop after calling a tool, or mangle JSON arguments. This sounds basic, but it's the #1 failure mode that makes local agents unusable: the model calls tools incorrectly and the whole loop breaks down.

- **Native multimodal input.** Gemma 4 accepts images natively. PRE supports Ctrl+V image paste — screenshot a UI bug, paste it in, and ask the model to analyze it. No separate vision model, no preprocessing pipeline, no base64 workaround. It just works because the model was trained for it.

### Platform-Specific Engineering

PRE doesn't abstract away the hardware — it leans into it. Every layer of the stack is optimized for the specific reality of running a 26B MoE model on Apple Silicon via Ollama:

**Raw socket streaming, not buffered I/O.** Most LLM clients use `fgets()` or equivalent stdio-buffered reads to consume server-sent events. This adds latency — sometimes hundreds of milliseconds per token — because the C runtime waits to fill its internal buffer before returning data. PRE uses raw `recv()` on the TCP socket with a 64KB manual ring buffer and `memchr()`-based newline scanning. Every NDJSON chunk from Ollama is parsed and rendered the instant it arrives. The difference is visceral: tokens appear one at a time as the model generates them, not in jerky batches.

**Fixed context window with fast startup.** Changing `num_ctx` at runtime triggers a full model unload/reload in Ollama — 300+ seconds for large models. PRE avoids this entirely: the Modelfile sets `num_ctx=65536` once, `pre-launch` pre-warms the model with a real request (forcing full KV cache allocation), and every request from the code sends the exact same `num_ctx=65536`. The model cold-loads in ~1.5 seconds on M4 Max 128GB, and subsequent requests benefit from Ollama's KV cache prefix reuse with sub-second TTFT. Auto-compaction at 75% keeps conversations within the 65K budget.

**KV cache prefix reuse.** Ollama caches the KV state of previously processed tokens. If the prefix of a new request matches a previous one, those tokens don't need re-processing — only the new tokens go through the model. PRE exploits this by always sending the system prompt as the first `role:system` message in the messages array, identically formatted every turn. After the first exchange, the system prompt (tool descriptions, project context, memories) is essentially free — the KV cache already has it. This is why multi-turn conversations maintain fast TTFT even with a large system prompt.

**Server-reported token metrics.** Client-side timing (start a clock, measure when tokens arrive) includes network latency, JSON parsing overhead, and rendering time. It systematically underreports performance. PRE extracts `eval_duration`, `prompt_eval_duration`, `eval_count`, and `prompt_eval_count` from Ollama's final `done:true` NDJSON message — these are the server's own measurements of time spent in the model, giving you ground-truth tok/s and TTFT numbers.

**Model pre-warming.** Cold-starting a model — loading weights from disk into GPU memory — can take 30+ seconds and causes terrible TTFT on the first request. PRE's launcher script (`pre-launch`) sends a zero-message warmup request with `keep_alive: "24h"` before starting the CLI. By the time you see the prompt, the model is already in GPU memory and the KV cache is primed.

**Compact prompt engineering.** The system prompt is where most local agents waste their context budget — verbose tool descriptions, lengthy instructions, boilerplate. PRE compresses all 35 tool definitions into a **function-signature format** (`bash(command) read_file(path) glob(pattern,path?)...`) instead of numbered lists with multi-line descriptions. The entire tool catalog fits in ~800 tokens. This leaves more room for actual conversation and means fewer tokens to prefill on every turn (directly improving TTFT through the KV cache reuse described above).

**Whitespace-tolerant JSON parsing.** Different APIs (Ollama, Google, GitHub) format their JSON differently — some minified, some pretty-printed. A naive `strstr(json, "\"id\":\"")` fails when the server sends `"id" : "value"` with spaces around the colon. PRE's `json_find_key()` helper scans for the key string, then skips arbitrary whitespace before matching the colon. This eliminated an entire class of parsing failures that plagued early versions.

### What Makes It an Agent, Not a Chatbot

The difference between a chatbot and an agent is the **tool-call loop**. A chatbot generates text. An agent generates text, decides it needs more information or needs to take an action, calls a tool, reads the result, decides what to do next, and repeats until the task is done. This requires:

1. **Reliable tool calling** — the model must produce valid JSON tool calls consistently
2. **Fast execution** — each tool call adds a round-trip; if each one takes 10 seconds, a 5-step task takes a minute of waiting
3. **Sufficient context** — the conversation must hold the system prompt, tool definitions, all previous turns, tool results, and still have room for the model to reason
4. **Autonomy with safety** — most actions should just execute; only genuinely dangerous ones should ask

PRE delivers all four. The model produces well-formed tool calls via Ollama's native structured function calling. Each round-trip (generate → parse → execute → inject result → generate) completes in 1-3 seconds for typical tools. The 65K context with auto-compaction holds extended multi-step sessions. And the permission model is designed for power users: **35 of 38+ tools auto-execute** — only `process_kill`, `memory_delete`, and `applescript` require confirmation.

This means PRE can do things like:

- Read a stack trace, search the codebase for the relevant function, read the file, identify the bug, edit the fix, run the tests, and report the result — all from a single prompt
- Search your Gmail for a thread, summarize it, draft a reply, and save it — without you touching a browser
- Inspect network connections, check disk usage, review running processes, and generate a system health report
- Navigate your codebase, understand the architecture, and make coordinated edits across multiple files

These aren't demos. These are workflows that complete in under a minute because every layer of the stack was built to make the loop fast.

---

## Table of Contents

- [Quick Start](#quick-start)
- [What PRE Can Do](#what-pre-can-do)
- [Installation](#installation)
- [Documentation](#documentation)
  - [Commands](#commands)
  - [Tools](#tools)
  - [Memory System](#memory-system)
  - [Projects & PRE.md](#projects--premd)
  - [Channels](#channels)
  - [Context Management](#context-management)
- [Best Practices](#best-practices)
- [Web GUI](#web-gui)
- [Telegram Integration](#telegram-integration)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

```bash
# Full automated install (checks requirements, installs Ollama, pulls model, builds PRE)
cd pre
./install.sh

# Or manual setup:

# 1. Install Ollama (if not already installed)
brew install ollama

# 2. Pull the base model (~17 GB)
ollama pull gemma4:26b-a4b-it-q4_K_M

# 3. Clone and build PRE
git clone https://github.com/sunkencity999/pre.git
cd pre/engine
make pre

# 4. Launch (auto-creates optimized model on first run)
./pre-launch
```

PRE detects your project, loads memories, and drops you into an interactive prompt:

```
╔══════════════════════════════════════════════════╗
║  Personal Reasoning Engine (PRE)                ║
║  Gemma 4 26B-A4B                                ║
╚══════════════════════════════════════════════════╝
  Server:  http://localhost:11434
  Project: my-project  /Users/you/my-project
  Channel: #general
  Memory:  3 entries loaded
  Type /help for commands

my-project #general>
```

---

## What PRE Can Do

PRE is not a chatbot — it's a local agent with deep system access.

**Read and modify your codebase** — The model reads files, searches with glob/grep, writes and edits files with checkpointed undo, all through structured tool calls.

**Run commands autonomously** — Bash execution with a streamlined permission model. 35 of 38 tools auto-execute; only genuinely destructive operations ask for confirmation.

**Remember across sessions** — Persistent memory stores your preferences, project context, workflow patterns, and reference pointers. Memories survive restarts.

**Manage multiple workstreams** — Channels let you run parallel conversations with separate contexts within the same project.

**Deep system inspection** — Network interfaces, running processes, disk usage, hardware info, window management, screenshots, and arbitrary AppleScript automation.

**Connect to external services** — Optional integrations with Brave Search, GitHub, Google (Gmail, Drive, Docs), Wolfram Alpha, and Telegram via `/connections`. Google uses built-in OAuth — just sign in, no Cloud Console setup. API keys for the rest. Multi-account Google is supported (`/connections add google work`). Tokens stored locally.

**Use from your browser** — PRE includes a built-in web GUI at `http://localhost:7749` that launches automatically alongside the CLI. Three themes (Dark, Light, Evangelion), Calendas Plus serif typography, real-time streaming, full tool execution, session management — all sharing the same sessions and memory as the CLI. No framework, no build step, just `node server.js`.

**Chat from your phone** — Configure a Telegram bot and PRE automatically bridges it when you launch. Full system access from Telegram — same 35 tools, same memory, same agentic workflows. No separate process to manage.

**Personalize your agent** — On first launch, PRE asks you to name your assistant. The name appears in the banner, system prompt, and Telegram bot. Change it anytime with `/name`.

**Generate images locally** — `image_generate` tool creates photorealistic images via ComfyUI running on Apple Silicon (MPS). The installer offers two checkpoints: **Juggernaut XL v9** (recommended — 25-step, 1024x1024, photorealistic faces and scenes) or **SDXL Turbo** (fast — 4-step, 512x512, speed-optimized). No cloud API required. Optional install via `install.sh`.

**Build rich reports** — Multi-part artifacts let the model build long documents across multiple tool calls. Each `append_to` call adds a new section to an existing artifact. Combined with image generation and `/pdf` export, PRE can produce complete visual reports.

**Paste images for analysis** — Ctrl+V pastes clipboard images directly into the prompt. Gemma 4 is multimodal — it can analyze screenshots, diagrams, photos, and more.

**Export to PDF** — `/pdf` exports any artifact to a clean PDF via WebKit rendering. The model can also call `pdf_export` programmatically to generate shareable documents.

**Clean shutdown** — Ctrl+C stops all services (including the Telegram bot and ComfyUI) and unloads the model from GPU memory, freeing VRAM immediately.

**Respect your privacy** — Everything runs locally on your machine. Ollama serves the model, PRE manages the conversation. Connection-dependent tools make API calls to their respective services; all other tools are fully local.

---

## Installation

### Prerequisites

| Component | Required |
|-----------|----------|
| **macOS** | 14.0+ (Sonoma or later) |
| **Chip** | Apple Silicon (M1 or later) |
| **RAM** | 16 GB minimum, 32+ GB recommended |
| **Disk** | ~17 GB for model, +8 GB for image generation (optional) |
| **Ollama** | [ollama.ai](https://ollama.ai) or `brew install ollama` |
| **Xcode CLI** | `xcode-select --install` |
| **Python 3.10-3.13** | Optional — for ComfyUI image generation (`brew install python@3.12`) |

### Install

The easiest path is the automated installer:

```bash
git clone https://github.com/sunkencity999/pre.git
cd pre
./install.sh
```

This checks system requirements, installs Ollama if needed, pulls the base model, creates the optimized `pre-gemma4` model from the Modelfile, builds the PRE binary, installs `pre-launch` to your PATH, sets up data directories, and pre-warms the model into GPU memory.

Or install manually:

```bash
# Pull the base model (Gemma 4 26B-A4B, MoE, ~17 GB)
ollama pull gemma4:26b-a4b-it-q4_K_M

# Create the optimized model
cd pre/engine
ollama create pre-gemma4 -f Modelfile

# Build PRE
make pre

# Optional: add to PATH
make install
# or: ln -sf "$(pwd)/pre-launch" ~/.local/bin/pre-launch
```

### Launch

```bash
pre-launch                         # From any directory
pre-launch --dir /path/to/project  # Override working directory
pre-launch --max-tokens 16384      # Allow longer responses
```

The launcher checks that Ollama is running (starts it if not), creates `pre-gemma4` from the Modelfile if needed, pre-warms the model into GPU memory with a real request (matching `num_ctx=65536` for full KV cache allocation), starts the web GUI server on port 7749 (if Node.js is available), and launches the PRE binary.

---

## Documentation

### Commands

PRE supports slash commands for managing sessions, files, and configuration. Type `/help` for the full list, or `/help <topic>` for detailed guides.

#### Chat & Input

| Command | Description |
|---------|-------------|
| *(type a message)* | Send to the model |
| `!command` | Run a shell command (output stays local, not sent to model) |
| `/edit` | Open `$EDITOR` for multi-line prompts |
| `/file <path>` | Attach a file to next message (stackable) |
| `/run <cmd>` | Run a command, optionally feed output to model |

#### Session Management

| Command | Description |
|---------|-------------|
| `/new` | Fresh session in current channel |
| `/sessions` | List all saved sessions |
| `/resume <id>` | Resume a previous session |
| `/rename <name>` | Name the current session |
| `/rewind [N]` | Remove last N turns (default: 1) |
| `/save <path>` | Save last response to a file |
| `/export [path]` | Export conversation to markdown |
| `/summary` | Model-generated session summary |

#### Navigation & Project

| Command | Description |
|---------|-------------|
| `/cd <path>` | Change directory (re-detects project) |
| `/ls [path]` | List directory |
| `/tree [path]` | Directory tree (depth 3) |
| `/project` | Show detected project info |
| `/channel [name]` | List or switch channels |

#### Memory & Status

| Command | Description |
|---------|-------------|
| `/memory [query]` | List or search memories |
| `/forget <query>` | Delete a memory |
| `/context` | Context window usage bar |
| `/status` | Current state overview |
| `/stats` | Detailed session statistics |
| `/undo` | Revert last file change |
| `/think` | Toggle reasoning visibility |
| `/name <name>` | Rename your agent |
| `/pdf [title]` | Export artifact to PDF |
| `/cron add <schedule> <prompt>` | Schedule a recurring task (5-field cron) |
| `/cron list` | List all scheduled tasks |
| `/cron rm <id>` | Remove a scheduled task |

#### Help

| Command | Description |
|---------|-------------|
| `/help` | Command overview |
| `/help tools` | All 38+ tools with permission levels |
| `/help memory` | Memory system guide |
| `/help channels` | Channel system guide |
| `/help projects` | Project detection & PRE.md |
| `/help tips` | Best practices and tips |
| `/help all` | Everything at once |

---

### Tools

PRE has 32 built-in tools plus up to 6 connection-dependent tools (38 total) that the model can call autonomously. PRE is designed for power users — nearly all tools auto-execute without confirmation:

- **Auto** — executes immediately, no confirmation needed (35 of 38 tools)
- **Confirm always** — asks every time (only 3 tools: `process_kill`, `memory_delete`, `applescript`)

#### File & Code

| # | Tool | Args | Description |
|---|------|------|-------------|
| 1 | `read_file` | `path` | Read file contents |
| 2 | `list_dir` | `path` | List directory contents |
| 3 | `glob` | `pattern`, `path`? | Find files by glob pattern |
| 4 | `grep` | `pattern`, `path`?, `include`? | Search file contents (regex) |
| 5 | `file_write` | `path`, `content` | Create or overwrite a file (checkpointed, `/undo`-able) |
| 6 | `file_edit` | `path`, `old_string`, `new_string` | Find-and-replace in a file (checkpointed, `/undo`-able) |

#### Shell & Process

| # | Tool | Args | Description |
|---|------|------|-------------|
| 7 | `bash` | `command` | Execute a shell command |
| 8 | `process_list` | `filter`? | List running processes |
| 9 | `process_kill` | `pid` | Send SIGTERM to a process *(confirm always)* |

#### System Inspection

| # | Tool | Args | Description |
|---|------|------|-------------|
| 10 | `system_info` | *(none)* | CPU, memory, disk, battery overview |
| 11 | `hardware_info` | *(none)* | Detailed hardware, thermal sensors, GPU info |
| 12 | `disk_usage` | `path`? | Volume and directory usage |
| 13 | `display_info` | *(none)* | Display resolution and GPU details |

#### Network

| # | Tool | Args | Description |
|---|------|------|-------------|
| 14 | `net_info` | *(none)* | Interfaces, IPs, DNS, routes |
| 15 | `net_connections` | `filter`? | TCP connections (listening/established/port) |
| 16 | `service_status` | `service`? | List or search launchd services |

#### Desktop Integration

| # | Tool | Args | Description |
|---|------|------|-------------|
| 17 | `screenshot` | `region`? | Capture screen (full/window/region/x,y,w,h) |
| 18 | `window_list` | *(none)* | List open windows with positions |
| 19 | `window_focus` | `app` | Bring an app to front |
| 20 | `clipboard_read` | *(none)* | Read clipboard contents |
| 21 | `clipboard_write` | `content` | Write to clipboard |
| 22 | `open_app` | `target` | Open files/apps/URLs via macOS `open` |
| 23 | `notify` | `title`, `message` | Show a macOS notification |
| 24 | `applescript` | `script` | Run arbitrary AppleScript *(confirm always)* |

#### Web

| # | Tool | Args | Description |
|---|------|------|-------------|
| 25 | `web_fetch` | `url` | Fetch a URL (HTML→text conversion) |

#### Creative & Export

| # | Tool | Args | Description |
|---|------|------|-------------|
| 26 | `artifact` | `title`, `content`, `type`, `append_to`? | Create/append rich HTML artifacts in pop-out viewer |
| 27 | `image_generate` | `prompt`, `width`?, `height`?, `style`? | Generate images locally via ComfyUI (Juggernaut XL or SDXL Turbo, MPS) |
| 28 | `pdf_export` | `title`, `path`? | Export an artifact to PDF via WebKit |

#### Memory

| # | Tool | Args | Description |
|---|------|------|-------------|
| 29 | `memory_save` | `name`, `type`, `description`, `content`, `scope`? | Save a persistent memory (global or project-scoped) |
| 30 | `memory_search` | `query`? | Search saved memories |
| 31 | `memory_list` | *(none)* | List all memories |
| 32 | `memory_delete` | `query` | Delete a memory *(confirm always)* |

#### Scheduling

| # | Tool | Args | Description |
|---|------|------|-------------|
| 33 | `cron` | `action`, `schedule`?, `prompt`?, `description`?, `id`? | Manage recurring scheduled tasks |

#### Connection-Dependent Tools

These tools require external API keys or OAuth setup via `/connections`. Run `/connections` to configure.

| # | Tool | Connection | Args | Description |
|---|------|------------|------|-------------|
| 34 | `web_search` | Brave Search | `query`, `count`? | Web search via Brave Search API |
| 35 | `github` | GitHub | `action`, `repo`?, `query`?, `number`?, `state`? | GitHub API (search repos, issues, PRs, user info) |
| 36 | `gmail` | Google | `action`, `query`?, `id`?, `to`?, `subject`?, `body`?, `cc`?, `bcc`?, `max_results`? | Gmail (search, read, send, draft, trash, labels, profile) |
| 37 | `gdrive` | Google | `action`, `id`?, `path`?, `name`?, `folder_id`?, `query`?, `email`?, `role`?, `count`? | Google Drive (list, search, download, upload, mkdir, share, delete) |
| 38 | `gdocs` | Google | `action`, `id`?, `title`?, `content`? | Google Docs (create, read, append) |
| 39 | `wolfram` | Wolfram Alpha | `query` | Computation, math, science, data queries |

---

### Memory System

PRE has persistent, file-based memory that survives across sessions and restarts.

#### How It Works

Memories are markdown files with YAML frontmatter stored in `~/.pre/memory/` (global) or `~/.pre/projects/{name}/memory/` (project-scoped). All relevant memories are injected into context at the start of each session.

#### Memory Types

| Type | Purpose | Example |
|------|---------|---------|
| `user` | About you — role, preferences, expertise | "Senior Python dev, prefers type hints" |
| `feedback` | How to work — corrections and confirmations | "Don't add docstrings to unchanged code" |
| `project` | Current work — decisions, deadlines, context | "Auth rewrite driven by compliance, not tech debt" |
| `reference` | External pointers — where info lives | "Bugs tracked in Linear project INGEST" |

#### Saving Memories

The model saves memories proactively when it learns something about you or your project. You can also be explicit:

```
pre> Remember that I prefer functional style over class-based components.
pre> Save a project memory: we're migrating from REST to GraphQL by Q3.
```

Project-scoped memories stay with that project and don't leak into other contexts.

#### Managing Memories

```
/memory              # List all memories
/memory auth         # Search for "auth" in memories
/forget "old rule"   # Delete a memory (with confirmation)
```

#### File Format

```markdown
---
name: User prefers functional style
description: Coding style preference for React components
type: feedback
---

User prefers functional components with hooks over class-based components.
Confirmed when reviewing frontend code on 2026-03-15.
```

---

### Projects & PRE.md

#### Auto-Detection

PRE detects project boundaries by walking up from your working directory looking for:

`.git` · `package.json` · `pyproject.toml` · `Cargo.toml` · `go.mod` · `Makefile` · `CMakeLists.txt` · `pom.xml` · `PRE.md`

When a project is detected, PRE:
1. Shows the project name in the prompt and banner
2. Creates `~/.pre/projects/{name}/` for project data
3. Loads project-scoped memories alongside global ones
4. Scopes channels to the project

#### PRE.md — Project Configuration

Place a `PRE.md` file in your project root to give the model project-specific instructions. It's loaded into context on the first turn of every session.

```markdown
# My API Service

FastAPI application with PostgreSQL and Redis.

## Conventions
- Use async/await for all I/O
- Type hints required on all public functions
- Tests in tests/ — run with: pytest -xvs
- Deploy target: AWS ECS on arm64

## Architecture
- src/api/ — FastAPI routes
- src/models/ — SQLAlchemy models
- src/services/ — Business logic
- migrations/ — Alembic migrations
```

Use `/project` to see detected project info and verify PRE.md is loaded.

---

### Channels

Channels are named conversation threads scoped to a project. Each channel has its own message history and context.

```
/channel                 # List channels
/channel refactor        # Switch to #refactor (creates if new)
/channel debug-auth      # Separate thread for debugging
/channel general         # Back to default
/new                     # Fresh session in current channel
```

**Why channels?** Long conversations accumulate context. If you're deep into a refactoring discussion and need to debug something unrelated, switching channels gives you a clean context without losing your refactoring thread.

Channels are scoped to the detected project. When you `/cd` into a different project, PRE switches to that project's `#general` channel.

---

### Connections

PRE can integrate with external services via API keys and OAuth. Run `/connections` to see available integrations, or `/connections add <service>` to set one up.

| Service | Auth Type | Tools Unlocked |
|---------|-----------|----------------|
| **Google** | Built-in OAuth 2.0 | `gmail`, `gdrive`, `gdocs` |
| **Telegram** | Bot token (via @BotFather) | Phone access to PRE |
| **Brave Search** | API key | `web_search` |
| **GitHub** | Personal access token | `github` |
| **Wolfram Alpha** | API key | `wolfram` |

**Google** uses built-in OAuth credentials — just run `/connections add google` and sign in via your browser. No Google Cloud Console setup required. For advanced users who want their own OAuth app, option 2 in the setup menu supports custom client ID/secret.

**Multi-account Google** is supported: `/connections add google work` and `/connections add google personal` create named accounts. The model uses the `account` parameter to target the right one.

**Telegram** automatically starts when PRE launches (if configured). See [Telegram Integration](#telegram-integration) for details.

All tokens are stored locally in `~/.pre/connections.json` and refreshed automatically.

---

### Context Management

PRE manages Gemma 4's context window with a **fixed 65K token allocation** (set once in the Modelfile and matched exactly in every request). This avoids Ollama's model reload penalty — changing `num_ctx` at runtime triggers a full unload/reload cycle (300+ seconds for large models). Instead, PRE allocates 65K at startup, pre-warms the KV cache with a real request, and all subsequent requests reuse the same allocation with sub-second TTFT.

**Fixed 65K context** — Large enough for extended multi-step agentic sessions (read 20 files, chain tool calls, iterate) while cold-loading in ~1.5 seconds on M4 Max 128GB. Auto-compaction keeps conversations within budget.

**Context bar** — Shown after every response (color-coded: grey → yellow at 50% → red at 75%). Also available via `/context`.

**Auto-compaction** — When estimated tokens exceed 75% of the 65K budget (~49K tokens), older conversation turns are automatically summarized and compressed. The last 6 exchanges are kept intact.

**Server-reported tokens** — PRE uses Ollama's native API which reports exact prompt and generation token counts from the server, giving you ground-truth context usage instead of client-side estimates.

**Tool response cap** — Tool outputs are truncated to 8KB to prevent a single large file from consuming the entire context.

**Rewind** — `/rewind N` removes the last N turns to free context space.

**Channels** — Use separate channels for different tasks to avoid context pollution.

---

## Best Practices

### Getting Good Results

1. **Be specific.** "Review auth.py for SQL injection" > "check my code"
2. **Attach files first.** `/file src/main.py` then ask your question.
3. **Use /edit for complex prompts.** Opens your $EDITOR for multi-line input.
4. **Watch the thinking.** `/think` toggles the model's reasoning — useful for debugging and understanding its approach.
5. **Use PRE.md.** A project config file saves you from re-explaining your stack every session.

### Managing Context

1. **Check /context regularly.** Know how much budget you've used.
2. **Use channels for separate tasks.** Don't mix debugging and feature work.
3. **Rewind noisy turns.** `/rewind` removes turns that added bulk without value.
4. **Let auto-compaction work.** It kicks in at 75% and preserves recent context.

### Tool Calling

1. **Almost everything auto-executes.** 32 of 35 tools run without confirmation. Only `process_kill`, `memory_delete`, and `applescript` prompt before executing.
2. **Undo mistakes.** `/undo` reverts the last `file_write` or `file_edit`.
3. **The model is agentic.** It will chain tool calls to accomplish multi-step tasks — read code, search for patterns, make edits, run commands, check results.

### Memory

1. **Tell PRE what to remember.** "Remember that I prefer X" works.
2. **Use project-scoped memory** for things specific to one codebase.
3. **Review memories periodically.** `/memory` lists everything. `/forget` removes stale entries.
4. **The model saves proactively.** It learns your preferences from corrections and confirmations.

### Shell Integration

1. **`!` for quick commands.** `!git status`, `!docker ps` — runs immediately, output not sent to model.
2. **`/run` to feed output to model.** Runs the command and asks if you want the model to see it.
3. **`bash` tool for model-driven commands.** The model calls `bash` autonomously when it needs to run something.

---

## Web GUI

PRE includes a built-in browser interface that provides full access to all PRE features through a modern, responsive web application.

### How It Works

The web GUI is a **Node.js (Express + WebSocket) backend** with a **vanilla JS SPA frontend** — no React, no Vue, no bundler. It talks directly to Ollama, reads/writes the same JSONL session files as the CLI, executes tools server-side, and streams responses via WebSocket. Sessions are fully interchangeable between CLI and web.

### Features

- **Real-time streaming** — Tokens appear as the model generates them, with a thinking block, streaming cursor, and live tool status cards
- **Full tool execution** — All 37+ tools run server-side with the same multi-turn tool loop as the CLI (up to 25 autonomous tool calls per prompt)
- **Three themes** — Dark, Light, and Evangelion (NERV-inspired orange-on-purple with hexagonal grid background and scan-line animations)
- **Calendas Plus typography** — Serif display font for headings, system-ui stack for body text
- **Session management** — Sidebar with session list, create/switch sessions, shared with CLI
- **Context tracking** — Live context window usage bar in the topbar
- **Tool confirmation** — Dangerous tools (process_kill, applescript, memory_delete) show a confirmation dialog before executing
- **Responsive layout** — Three-panel design (sidebar, chat, artifact panel) with mobile hamburger drawer at <768px

### Auto-Launch

The web GUI starts automatically when you run `pre-launch` (requires Node.js). It runs in the background on port **7749** (configurable via `PRE_WEB_PORT`). If the server is already running, the launcher detects it and skips.

### Manual Launch

```bash
cd pre/web
npm install          # First time only
node server.js       # Starts on http://localhost:7749

# Or with custom settings:
PRE_WEB_PORT=8080 PRE_CWD=/path/to/project node server.js
```

### Architecture

```
Browser (localhost:7749)
  ├── WebSocket ──► Express + ws server
  │                   ├── Ollama NDJSON streaming client
  │                   ├── Tool dispatcher (48 aliases, 37 tools)
  │                   ├── Session JSONL read/write (shared with CLI)
  │                   └── System prompt builder (memory, connections, rules)
  └── REST API ──► /api/sessions, /api/status, /api/rewind
```

The web GUI never exposes raw API keys to the browser — all tool execution and credential access happens server-side.

---

## Telegram Integration

PRE includes a built-in Telegram bot that gives you full agent access from your phone.

### Setup

1. Message [@BotFather](https://t.me/BotFather) on Telegram and create a new bot
2. In PRE, run `/connections add telegram` and paste the bot token
3. Restart PRE — the Telegram bot starts automatically

### How It Works

When PRE launches and a Telegram connection is configured, it automatically spawns `pre-telegram` as a background process. The bot long-polls the Telegram API (no webhook, no public URL required) and routes messages through the same Ollama instance as the TUI.

- **Full system access** — all 38+ tools, same as the TUI. File operations, bash, process management, clipboard, screenshots, Google services, everything.
- **Owner authorization** — the first Telegram user to message the bot becomes the owner. All other users are blocked.
- **Conversation management** — per-chat history with `/new` to reset
- **Automatic lifecycle** — starts with PRE, stops with PRE (Ctrl+C kills both)
- **Logs** — output goes to `~/.pre/telegram.log`

### Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/new` | New conversation |
| `/status` | Bot status (model, connections, memory count) |
| `/memory` | List saved memories |
| `/help` | Show commands |

### Architecture

The Telegram bot (`telegram.m` / `pre-telegram`) is a separate binary that shares the same Ollama model, memory system, connections, and identity. It uses non-streaming Ollama chat with dynamic context scaling (matching the TUI), and sends typing indicators while the model generates.

---

## Architecture

### How PRE Works

```
┌─────────────┐    Ollama Native    ┌─────────────────┐
│   PRE CLI   │  ◄─ /api/chat ──►  │     Ollama       │
│  (pre.m)    │   raw recv() NDJSON │  localhost:11434  │
│ ~10000 lines│                     │                   │
│  Obj-C / C  │                     │  Gemma 4 26B-A4B  │
└──┬──────┬───┘                     │  MoE, q4_K_M      │
   │      │                         │  num_ctx=65536     │
   │      │ fork/exec               │  ~70 tok/s         │
   │      │                         └──────────▲────────┘
   │  ┌───▼──────────┐  /api/chat (non-stream) │
   │  │ pre-telegram  │ ───────────────────────┘
   │  │ (telegram.m)  │                         │
   │  │ Telegram Bot  │  ◄──► Telegram Bot API  │
   │  └───────────────┘       (long-poll)       │
   │                                            │
   │ auto-start/stop          /api/chat NDJSON  │
   │                    ┌───────────────────────┘
┌──▼──────────────┐  ┌──┴──────────────┐
│    ComfyUI      │  │   PRE Web GUI   │  localhost:7749
│ Juggernaut XL   │  │  (Node.js/WS)   │  3 themes, streaming
│  (MPS/Metal)    │  │  vanilla JS SPA  │  full tool execution
│ 25-step, 1024px │  │  shared sessions │  Calendas Plus typography
└─────────────────┘  └─────────────────┘

  ~/.pre/
  ├── identity.json       # Agent name
  ├── connections.json    # API keys and OAuth tokens
  ├── sessions/           # Conversation JSONL files (shared by CLI + web)
  ├── history             # Input history (arrow-key recall)
  ├── checkpoints/        # File backups for /undo
  ├── artifacts/          # HTML artifacts and generated images
  ├── cron.json           # Scheduled recurring tasks
  ├── comfyui.json        # ComfyUI configuration (if installed)
  ├── comfyui/            # ComfyUI installation (if installed)
  ├── comfyui-venv/       # Python venv for ComfyUI (if installed)
  ├── telegram.log        # Telegram bot output
  ├── telegram_owner      # Authorized Telegram user ID
  ├── memory/             # Global persistent memories
  │   ├── index.md
  │   └── *.md
  └── projects/
      └── {name}/
          ├── memory/     # Project-scoped memories
          └── channels/   # Channel metadata
```

### Optimization Stack

PRE's performance comes from a stack of reinforcing optimizations, not any single trick:

| Layer | Optimization | Effect |
|-------|-------------|--------|
| **Model selection** | Gemma 4 26B-A4B MoE — 3.8B active params of 26B total | 26B quality at ~4B speed |
| **Quantization** | q4_K_M (~17 GB) — sweet spot of quality vs. memory | Fits comfortably in unified memory |
| **Context allocation** | Fixed `num_ctx=65536` matching Modelfile exactly | 1.5s cold load, no runtime reload penalty |
| **KV cache reuse** | Identical system prompt prefix every turn | System prompt is free after turn 1 |
| **Streaming I/O** | Raw `recv()` with 64KB ring buffer, `memchr()` line scan | Zero-latency token delivery |
| **Prompt compression** | Function-signature tool format (~800 tokens for 35 tools) | More room for conversation, faster prefill |
| **Model pre-warming** | `keep_alive: "24h"` + real warmup request (with matching `num_ctx`) at launch | Full KV cache pre-allocated, no cold-start penalty |
| **Server metrics** | Ollama-reported `eval_duration` / `prompt_eval_duration` | Ground-truth performance numbers |
| **Hybrid tool calling** | Native Ollama `tools` API for small tools, text-based for artifacts | Reliable structured calls + large content generation |
| **Artifact compaction** | Strip HTML from session after saving to disk | Prevents prefill stalls on follow-up turns |
| **Image data URIs** | Convert `file://` image refs to base64 at render time | Local images display in WebKit without cross-origin blocks |
| **Auto-compaction** | Summarize old turns at 75% context usage | Extends effective session length |
| **Tool response cap** | 8KB limit per tool result | Prevents context blowout from large files |

### The PRE Binary

**PRE CLI** (`pre.m`) is a single-file Objective-C/C application (~10,000 lines). It handles:
- Ollama native API client with raw `recv()` NDJSON streaming
- Fixed `num_ctx=65536` matching Modelfile (avoids Ollama reload penalty)
- System prompt as `role:system` message for KV cache prefix reuse
- Streaming markdown renderer with ANSI formatting
- Hybrid tool calling: native Ollama `tools` API + text-based `<tool_call>` for large content
- 38+ tool implementations with two-tier permissions
- Local image generation via ComfyUI + Juggernaut XL/SDXL Turbo (MPS-accelerated, checkpoint-adaptive workflow)
- Multi-part artifacts with incremental append mode
- PDF export via native WebKit rendering
- Cron registry for recurring scheduled tasks (`/cron`, persisted in `~/.pre/cron.json`)
- Built-in Google OAuth 2.0 with multi-account support
- Connection management for external services (Brave, GitHub, Google, Wolfram, Telegram)
- Automatic Telegram bot subprocess management
- Agent identity system (custom naming, persisted across sessions)
- Persistent memory with per-project scoping
- Channel-based conversation management
- Project detection and PRE.md loading
- Context compaction and token budget management (65K fixed context)
- Artifact session compaction (strips HTML from session after saving to disk)
- Model unload on exit (frees GPU memory on Ctrl+C)
- File checkpointing and undo
- Ctrl+V image paste for multimodal queries (via AppKit)
- linenoise-based line editor with tab completion and ANSI-aware cursor

**Telegram Bot** (`telegram.m`) is a companion binary (~2000 lines) auto-launched by PRE:
- Telegram Bot API long-polling (no webhook/public URL needed)
- Full tool access matching the TUI (all 38+ tools)
- Per-chat conversation history with dynamic context scaling
- Owner-based authorization (first user to message becomes owner)
- Typing indicators via forked child process
- Shared identity, connections, and memory with the TUI

**Ollama** serves the model via a custom Modelfile (`pre-gemma4`) with tuned parameters and a small default context. PRE uses the native `/api/chat` endpoint — not the OpenAI-compatible layer — for access to server-reported metrics, native multimodal support, and streaming NDJSON.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRE_PORT` | `11434` | Ollama server port |
| `PRE_WEB_PORT` | `7749` | Web GUI server port |
| `EDITOR` | `vi` | Editor for `/edit` command |

### CLI Options

```bash
pre-launch [options]
  --port N          Override Ollama port
  --max-tokens N    Max response tokens (default: 8192)
  --show-think      Show reasoning blocks by default
  --resume ID       Resume a previous session
  --sessions        List saved sessions and exit
  --dir PATH        Override working directory
```

### Data Location

All PRE data lives in `~/.pre/`:

```
~/.pre/
├── identity.json       # Agent name
├── connections.json    # API keys, OAuth tokens (chmod 600)
├── sessions/           # Conversation JSONL (one per channel)
├── history             # Readline history
├── checkpoints/        # File backups (auto-cleaned)
├── telegram.log        # Telegram bot output
├── telegram_owner      # Authorized Telegram user ID
├── memory/
│   ├── index.md        # Memory index
│   └── *.md            # Individual memory files
└── projects/
    └── {project-id}/
        ├── memory/     # Project-scoped memories
        └── channels/   # Channel metadata
```

---

## Project Structure

```
pre/
├── README.md               # This file
├── install.sh              # Automated installer
├── system.md               # Model system prompt reference
├── engine/
│   ├── pre.m               # PRE CLI (single-file, ~10000 lines)
│   ├── telegram.m          # Telegram bot bridge (~2000 lines)
│   ├── linenoise.c/h       # Terminal line editor (patched: Ctrl+V, ANSI-aware)
│   ├── Makefile             # Build: make pre telegram
│   ├── Modelfile            # Ollama model config (pre-gemma4, num_ctx=65536)
│   ├── pre-launch           # Universal launcher script (starts CLI + web GUI)
│   └── launch-telegram      # Standalone Telegram launcher (optional)
├── web/                     # Web GUI (Node.js + vanilla JS)
│   ├── server.js            # Express + WebSocket server
│   ├── package.json
│   ├── src/
│   │   ├── ollama.js        # Ollama NDJSON streaming client
│   │   ├── sessions.js      # JSONL read/write (shared with CLI)
│   │   ├── tools.js         # Tool dispatcher + execution loop
│   │   ├── tools-defs.js    # 37 tool definitions for Ollama
│   │   ├── context.js       # System prompt builder
│   │   ├── constants.js     # MODEL_CTX=65536, paths
│   │   └── tools/           # Tool implementations (bash, files, web, memory, system)
│   └── public/
│       ├── index.html       # SPA shell
│       ├── fonts/           # Calendas Plus (regular, italic, bold)
│       ├── css/             # base, themes (dark/light/evangelion), chat, sidebar, components, animations
│       └── js/              # app, ws, chat, markdown, themes
├── docs/
│   └── *.md                 # Research notes from Flash-MoE era
└── benchmark_results/       # Performance benchmarks
```

---

## Contributing

PRE is built and maintained by **Christopher Bradford** — systems administrator at Joby Aviation and AI engineer.

**Areas for contribution:**
- **New tools** — Calendar integration, git operations, Slack, Linear/Jira
- **Smarter memory** — Auto-extraction of important facts, memory relevance ranking
- **Better rendering** — Syntax highlighting in code blocks, image display in terminal
- **Model support** — Test with other Ollama models (Llama 4, Qwen 3, etc.)
- **New connections** — Slack, Discord, Notion, Linear, and other service integrations
- **Web GUI** — Sidebar features (channels, artifacts viewer), image generation UI, cron management, slash command palette
- **Telegram enhancements** — Inline keyboards, photo/document handling, group chat support
- **Documentation** — Tutorials, use-case guides, video walkthroughs

**How to contribute:**
1. Fork and create a branch
2. Make changes
3. Open a pull request

**Contact:**
- GitHub: [@sunkencity999](https://github.com/sunkencity999)

---

## Acknowledgments

- **Google** for the Gemma 4 model family
- **Ollama** for making local model serving effortless
- **Apple** for unified memory architecture that makes this possible
- The original **Flash-MoE** research that informed PRE's early design

## License

MIT License. See [LICENSE](LICENSE) for details.

The Gemma 4 model weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
