# PRE — Personal Reasoning Engine

> A fully local agentic assistant that actually works. No cloud. No API keys required. No data leaves your machine.

PRE is not a chatbot with tools bolted on. It is a **purpose-built agent** — a single-binary Objective-C application engineered from the ground up around one specific model on one specific platform. Every architectural decision, from socket-level I/O to dynamic memory allocation to prompt compression, exists to make **Google Gemma 4 26B-A4B** run at its absolute ceiling on Apple Silicon. The result is a local agent that doesn't feel local: **~70 tokens/second**, sub-second time to first token, 65K context window, 45+ integrated tools, persistent memory, local image generation, autonomous scheduling, a built-in web GUI, and real agentic workflows — all running on your MacBook.

The reference system is a **MacBook Pro with an M4 Max (128 GB unified memory)**.

---

## Why This Exists

Most local AI tools follow a generic pattern: wrap an OpenAI-compatible API, connect a few tools, hope for the best. The result is sluggish, fragile, and useful mainly as a novelty. PRE takes the opposite approach — it is a **model-specific, platform-specific, vertically integrated agent** — and that specificity is what makes it competitive with cloud-based offerings like ChatGPT Pro, Claude, and Gemini Advanced, while keeping everything private and running on your own hardware.

### The Right Model for Agency

**Gemma 4 26B-A4B** is a Mixture-of-Experts (MoE) architecture: 26 billion total parameters, but only **3.8 billion active per token** (128 experts, 8 active per forward pass). This is the key to the entire system:

- **Speed without sacrifice.** 26B-parameter quality at ~4B computational cost. On Apple Silicon with q4_K_M quantization (~17 GB), this means **~70 tokens/second** — fast enough that the agent's tool-call-execute-respond loop feels interactive, not glacial.

- **Context without collapse.** Gemma 4 supports up to 262K tokens natively. PRE allocates a **65K token window** — large enough for extended multi-step workflows (read 20 files, chain tool calls, iterate) while cold-loading in just 1.5 seconds. Auto-compaction at 75% extends effective session length indefinitely.

- **Strong instruction following.** Gemma 4 handles PRE's `<tool_call>` JSON format consistently — it doesn't hallucinate partial calls, forget to stop after calling a tool, or mangle JSON arguments. This sounds basic, but it's the #1 failure mode that makes local agents unusable.

- **Native multimodal input.** Gemma 4 accepts images natively — screenshot a UI bug, paste it in, and ask the model to analyze it. No separate vision model, no preprocessing pipeline.

### Platform-Specific Engineering

PRE doesn't abstract away the hardware — it leans into it:

| Layer | Optimization | Effect |
|-------|-------------|--------|
| **Streaming I/O** | Raw `recv()` with 64KB ring buffer, `memchr()` line scan | Zero-latency token delivery |
| **Context allocation** | Fixed `num_ctx=65536` matching Modelfile exactly | 1.5s cold load, no runtime reload penalty |
| **KV cache reuse** | Identical system prompt prefix every turn | System prompt is free after turn 1 |
| **Prompt compression** | Function-signature tool format (~800 tokens for 45+ tools) | More room for conversation, faster prefill |
| **Model pre-warming** | `keep_alive: "24h"` + real warmup request at launch | Full KV cache pre-allocated, sub-second TTFT |
| **Server metrics** | Ollama-reported `eval_duration` / `prompt_eval_duration` | Ground-truth tok/s and TTFT numbers |

### What Makes It an Agent, Not a Chatbot

The difference is the **tool-call loop**. A chatbot generates text. An agent generates text, decides it needs more information, calls a tool, reads the result, decides what to do next, and repeats until the task is done. PRE delivers:

1. **Reliable tool calling** via Ollama's native structured function calling
2. **Fast execution** — each round-trip completes in 1-3 seconds for typical tools
3. **Sufficient context** — 65K window with auto-compaction holds extended multi-step sessions
4. **Autonomous scheduling** — cron jobs execute server-side, even when you're not at the computer

This means PRE can:

- Read a stack trace, search the codebase, identify the bug, edit the fix, run the tests, and report the result — all from a single prompt
- Search your Gmail, summarize a thread, draft a reply, and save it — without you touching a browser
- Generate images locally, create documents, build artifacts, and export PDFs
- Run recurring tasks on a schedule — morning briefings, system checks, report generation — and deliver results via macOS notification and Telegram
- Manage Jira tickets, search Confluence wikis, update Smartsheet rows, and send Slack messages
- Delegate prompts to frontier AI models (Claude, Codex, Gemini) when cloud-level intelligence is needed — PRE becomes an agentic orchestration layer

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
  - [Connections](#connections)
  - [Scheduling (Cron)](#scheduling-cron)
  - [Context Management](#context-management)
- [Best Practices](#best-practices)
- [Web GUI](#web-gui)
- [Frontier AI Delegation](#frontier-ai-delegation)
- [Telegram Integration](#telegram-integration)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

```bash
# Full automated install (checks requirements, installs dependencies, pulls model, builds PRE)
git clone https://github.com/sunkencity999/pre.git
cd pre
./install.sh

# Launch
pre-launch
```

The installer checks system requirements, installs Ollama if needed, pulls the base model (~17 GB), creates the optimized `pre-gemma4` model, builds the PRE binary, sets up the web GUI, installs `terminal-notifier` for clickable notifications, creates data directories, and pre-warms the model into GPU memory.

PRE detects your project, loads memories, and drops you into an interactive prompt:

```
╔══════════════════════════════════════════════════╗
║  Personal Reasoning Engine (PRE)                ║
║  Gemma 4 26B-A4B                                ║
╚══════════════════════════════════════════════════╝
  Server:  http://localhost:11434
  Web GUI: http://localhost:7749
  Project: my-project  /Users/you/my-project
  Channel: #general
  Memory:  3 entries loaded
  Type /help for commands

my-project #general>
```

---

## What PRE Can Do

PRE is not a chatbot — it's a local agent with deep system access and 45+ tools.

**Read and modify your codebase** — Reads files, searches with glob/grep, writes and edits files with checkpointed undo, all through structured tool calls.

**Run commands autonomously** — Bash execution with a streamlined permission model. 42 of 45+ tools auto-execute; only genuinely destructive operations ask for confirmation.

**Remember across sessions** — Persistent memory stores your preferences, project context, workflow patterns, and reference pointers. Memories survive restarts. Auto-extraction learns from conversations without being asked.

**Generate images locally** — Photorealistic image generation via ComfyUI on Apple Silicon (MPS). Juggernaut XL v9 (1024x1024, 25-step) or SDXL Turbo (512x512, 4-step). No cloud API.

**Create documents and artifacts** — Multi-part HTML artifacts, Excel spreadsheets, Word documents, and PDFs. Download, open, or reveal in Finder directly from the GUI.

**Schedule recurring tasks** — Cron-based scheduling with natural language input ("every morning at 9am"). Jobs execute server-side even when the browser is closed. Results are stored as individual chat sessions and delivered via macOS notification (click to open), Telegram, and in-browser toasts.

**Upload files and images for analysis** — Drag-and-drop, clipboard paste, or file picker. Gemma 4's native multimodal support analyzes screenshots, diagrams, photos, CSVs, code files, and more.

**Connect to external services** — Google (Gmail, Drive, Docs), GitHub, Telegram, Slack, Jira, Confluence, Smartsheet, Brave Search, and Wolfram Alpha. Configure via `/connections` or the web GUI Settings panel.

**Use from your browser** — Built-in web GUI at `http://localhost:7749` with three themes (Dark, Light, Evangelion), real-time streaming, full tool execution, session management, project organization, memory browser, cron scheduler, and connection management.

**Delegate to frontier AI** — When a task requires cloud-level intelligence, route it to Claude (Anthropic), Codex (OpenAI), or Gemini (Google) with a single click. PRE detects which CLIs are installed, streams the response back in real-time, and stores the result in the session alongside local messages. Use local inference for 95% of tasks; escalate on demand.

**Chat from your phone** — Configure a Telegram bot and PRE bridges it automatically. Full system access — same tools, same memory, same agentic workflows.

**Manage multiple workstreams** — Projects with drag-and-drop session organization. Channels for separate conversation threads within the same project.

**Export and share** — PDF export, markdown export, document generation (DOCX, XLSX, TXT, CSV, HTML). Artifacts open in the browser or download directly.

**Respect your privacy** — Everything runs locally. Ollama serves the model, PRE manages the conversation. Connection-dependent tools make API calls to their respective services; all other tools are fully local.

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
| **Node.js 18+** | For web GUI (`brew install node`) |
| **Python 3.10-3.13** | Optional — for ComfyUI image generation |

### Install

```bash
git clone https://github.com/sunkencity999/pre.git
cd pre
./install.sh
```

The installer handles everything: system validation, Ollama, model pull, binary compilation, web GUI dependencies, terminal-notifier, ComfyUI (optional), data directories, and model pre-warming.

Or install manually:

```bash
# Pull the base model (~17 GB)
ollama pull gemma4:26b-a4b-it-q4_K_M

# Create the optimized model
cd pre/engine
ollama create pre-gemma4 -f Modelfile

# Build PRE CLI + Telegram bot
make pre telegram

# Web GUI
cd ../web && npm install

# Optional: clickable cron notifications
brew install terminal-notifier

# Add to PATH
make install
# or: ln -sf "$(pwd)/pre-launch" ~/.local/bin/pre-launch
```

### Launch

```bash
pre-launch                         # From any directory
pre-launch --dir /path/to/project  # Override working directory
pre-launch --show-think            # Start with visible reasoning
pre-launch --max-tokens 16384      # Allow longer responses
```

The launcher checks Ollama, creates `pre-gemma4` if needed, pre-warms the model into GPU memory, starts the web GUI on port 7749, and launches the PRE CLI.

---

## Documentation

### Commands

Type `/help` for the full list, or `/help <topic>` for detailed guides.

#### Chat & Input

| Command | Description |
|---------|-------------|
| *(type a message)* | Send to the model |
| `!command` | Run a shell command (output stays local) |
| `/edit` | Open `$EDITOR` for multi-line prompts |
| `/file <path>` | Attach a file to next message |
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

#### Scheduling

| Command | Description |
|---------|-------------|
| `/cron add <schedule> <prompt>` | Schedule a recurring task |
| `/cron list` | List all scheduled tasks |
| `/cron rm <id>` | Remove a scheduled task |

---

### Tools

PRE has 45+ tools that the model calls autonomously. Nearly all auto-execute without confirmation:

- **Auto** — executes immediately (42+ tools)
- **Confirm always** — asks every time (3 tools: `process_kill`, `memory_delete`, `applescript`)

#### File & Code

| Tool | Args | Description |
|------|------|-------------|
| `read_file` | `path` | Read file contents |
| `list_dir` | `path` | List directory contents |
| `glob` | `pattern`, `path`? | Find files by glob pattern |
| `grep` | `pattern`, `path`?, `include`? | Search file contents (regex) |
| `file_write` | `path`, `content` | Create or overwrite a file (checkpointed, `/undo`-able) |
| `file_edit` | `path`, `old_string`, `new_string` | Find-and-replace in a file (checkpointed, `/undo`-able) |

#### Shell & Process

| Tool | Args | Description |
|------|------|-------------|
| `bash` | `command` | Execute a shell command |
| `process_list` | `filter`? | List running processes |
| `process_kill` | `pid` | Send SIGTERM to a process *(confirm always)* |

#### System Inspection

| Tool | Args | Description |
|------|------|-------------|
| `system_info` | *(none)* | CPU, memory, disk, battery overview |
| `hardware_info` | *(none)* | Detailed hardware, thermal sensors, GPU info |
| `disk_usage` | `path`? | Volume and directory usage |
| `display_info` | *(none)* | Display resolution and GPU details |

#### Network

| Tool | Args | Description |
|------|------|-------------|
| `net_info` | *(none)* | Interfaces, IPs, DNS, routes |
| `net_connections` | `filter`? | TCP connections (listening/established/port) |
| `service_status` | `service`? | List or search launchd services |

#### Desktop Integration

| Tool | Args | Description |
|------|------|-------------|
| `screenshot` | `region`? | Capture screen (full/window/region/x,y,w,h) |
| `window_list` | *(none)* | List open windows with positions |
| `window_focus` | `app` | Bring an app to front |
| `clipboard_read` | *(none)* | Read clipboard contents |
| `clipboard_write` | `content` | Write to clipboard |
| `open_app` | `target` | Open files/apps/URLs via macOS `open` |
| `notify` | `title`, `message` | Show a macOS notification |
| `applescript` | `script` | Run arbitrary AppleScript *(confirm always)* |

#### Web

| Tool | Args | Description |
|------|------|-------------|
| `web_fetch` | `url` | Fetch a URL (HTML to text conversion) |

#### Creative & Export

| Tool | Args | Description |
|------|------|-------------|
| `artifact` | `title`, `content`, `type`, `append_to`? | Create/append rich HTML artifacts |
| `document` | `title`, `format`, `sheets`? | Generate DOCX, XLSX, TXT, CSV, HTML documents |
| `image_generate` | `prompt`, `width`?, `height`?, `style`? | Generate images via ComfyUI (MPS) |

#### Memory

| Tool | Args | Description |
|------|------|-------------|
| `memory_save` | `name`, `type`, `description`, `content` | Save a persistent memory |
| `memory_search` | `query`? | Search saved memories |
| `memory_list` | *(none)* | List all memories |
| `memory_delete` | `query` | Delete a memory *(confirm always)* |

#### Scheduling

| Tool | Args | Description |
|------|------|-------------|
| `cron` | `action`, `schedule`?, `prompt`?, `id`? | Manage recurring scheduled tasks |

#### Connection-Dependent Tools

These require API keys or OAuth setup via `/connections` or the web GUI Settings panel.

| Tool | Connection | Description |
|------|------------|-------------|
| `web_search` | Brave Search | Web search via Brave Search API |
| `github` | GitHub | Search repos, issues, PRs, user info |
| `gmail` | Google | Search, read, send, draft, trash, labels |
| `gdrive` | Google | List, search, download, upload, share |
| `gdocs` | Google | Create, read, append documents |
| `telegram` | Telegram | Send messages, photos, check updates |
| `jira` | Jira | Search issues, create, update, transition |
| `confluence` | Confluence | Search pages, read, create, update |
| `smartsheet` | Smartsheet | List sheets, search rows, add/update rows |
| `slack` | Slack | Send messages, list channels, reply, react |
| `wolfram` | Wolfram Alpha | Computation, math, science, data queries |

---

### Memory System

PRE has persistent, file-based memory that survives across sessions and restarts.

#### How It Works

Memories are markdown files with YAML frontmatter stored in `~/.pre/memory/` (global) or `~/.pre/projects/{name}/memory/` (project-scoped). Relevant memories are injected into context at the start of each session.

#### Memory Types

| Type | Purpose | Example |
|------|---------|---------|
| `user` | About you — role, preferences, expertise | "Senior Python dev, prefers type hints" |
| `feedback` | How to work — corrections and confirmations | "Don't add docstrings to unchanged code" |
| `project` | Current work — decisions, deadlines, context | "Auth rewrite driven by compliance" |
| `reference` | External pointers — where info lives | "Bugs tracked in Linear project INGEST" |

#### Three Ways Memories Are Created

1. **Auto-extraction** — The web GUI analyzes conversations and automatically saves important facts (throttled: every 3 turns, 60-second cooldown)
2. **Model-initiated** — The model saves memories proactively when it learns something important
3. **Explicit** — Tell PRE what to remember: "Remember that I prefer functional style over classes"

#### Managing Memories

```
/memory              # List all memories
/memory auth         # Search for "auth" in memories
/forget "old rule"   # Delete a memory (with confirmation)
```

The web GUI includes a **Memory Browser** panel (book icon in sidebar footer) for viewing and managing memories visually.

---

### Projects & PRE.md

#### Auto-Detection

PRE detects project boundaries by walking up from your working directory looking for:

`.git` · `package.json` · `pyproject.toml` · `Cargo.toml` · `go.mod` · `Makefile` · `CMakeLists.txt` · `pom.xml` · `PRE.md`

When a project is detected, PRE loads project-scoped memories, scopes channels to the project, and shows the project name in the prompt.

#### PRE.md — Project Configuration

Place a `PRE.md` file in your project root to give the model project-specific instructions:

```markdown
# My API Service

FastAPI application with PostgreSQL and Redis.

## Conventions
- Use async/await for all I/O
- Type hints required on all public functions
- Tests in tests/ — run with: pytest -xvs
```

The web GUI supports **Projects** with drag-and-drop session organization — group related sessions under a named project.

---

### Channels

Channels are named conversation threads scoped to a project. Each channel has its own message history and context.

```
/channel                 # List channels
/channel refactor        # Switch to #refactor
/channel debug-auth      # Separate thread for debugging
/new                     # Fresh session in current channel
```

Long conversations accumulate context. Switching channels gives you a clean context without losing your previous thread.

---

### Connections

PRE integrates with external services via API keys and OAuth. Configure via `/connections` in the CLI or the **Settings** panel in the web GUI.

| Service | Auth Type | Tools Unlocked |
|---------|-----------|----------------|
| **Google** | Built-in OAuth 2.0 | `gmail`, `gdrive`, `gdocs` |
| **Telegram** | Bot token (@BotFather) | Phone access + cron delivery |
| **Slack** | Bot OAuth token (xoxb-) | `slack` |
| **Brave Search** | API key | `web_search` |
| **GitHub** | Personal access token | `github` |
| **Jira** | URL + API token | `jira` |
| **Confluence** | URL + API token | `confluence` |
| **Smartsheet** | API token | `smartsheet` |
| **Wolfram Alpha** | API key | `wolfram` |

**Google** uses built-in OAuth — just sign in via your browser. No Google Cloud Console setup required. Multi-account supported: `/connections add google work`.

**Telegram** automatically starts as a background process when PRE launches.

All tokens are stored locally in `~/.pre/connections.json` and refreshed automatically.

---

### Scheduling (Cron)

PRE includes a full cron scheduling system for recurring autonomous tasks. Jobs execute **server-side** — no browser or terminal needs to be open.

#### How It Works

1. **Create a job** — via the web GUI cron panel (clock icon in sidebar) or the CLI (`/cron add`)
2. **Natural language scheduling** — type "every morning at 9am" or "weekdays at 8:30am" instead of raw cron expressions
3. **Server-side execution** — when a job fires, the server runs the prompt through the full agent tool loop headlessly
4. **Dedicated sessions** — each run creates its own chat session you can review, continue, or reference
5. **Multi-channel delivery** — results are delivered via all configured channels:

| Channel | Behavior |
|---------|----------|
| **macOS notification** | Always sent. Click to open the result in the web GUI. Uses `terminal-notifier` for clickable notifications (installed automatically). |
| **Telegram** | Sent if bot token is configured. Full result preview in your Telegram chat. |
| **GUI toast** | Shown if the browser is open. Slide-in notification with "View Result" button. |

#### Natural Language Schedule Input

The web GUI accepts plain English for scheduling:

| Input | Cron Expression | Description |
|-------|----------------|-------------|
| "every morning at 9am" | `0 9 * * *` | Daily at 9:00 AM |
| "weekdays at 8:30am" | `30 8 * * 1-5` | Weekdays at 8:30 AM |
| "every 15 minutes" | `*/15 * * * *` | Every 15 minutes |
| "monday and friday at 3pm" | `0 15 * * 1,5` | Mon & Fri at 3:00 PM |
| "monthly on the 1st at 10am" | `0 10 1 * *` | 1st of each month |
| "twice a day" | `0 9,17 * * *` | 9 AM and 5 PM |
| "every half hour" | `*/30 * * * *` | Every 30 minutes |

Raw cron expressions (e.g. `0 */2 * * 1-5`) are also accepted with passthrough.

#### Use Cases

- **Morning briefing** — "Summarize my calendar and top priorities for today"
- **System health check** — "Check disk usage, memory, and running services. Alert if anything is abnormal."
- **Competitive monitoring** — "Search for news about [company] and summarize findings"
- **Report generation** — "Generate a weekly project status report from git activity"
- **Data pipeline monitoring** — "Check the ETL job status and report any failures"

#### Managing Jobs

| Method | How |
|--------|-----|
| **Web GUI** | Clock icon in sidebar footer — visual panel with live preview, enable/disable toggle, run now, view last result |
| **CLI** | `/cron add "0 9 * * 1-5" "Morning briefing"`, `/cron list`, `/cron rm <id>` |
| **Model tool** | The model can create cron jobs autonomously via the `cron` tool |

---

### Context Management

PRE manages Gemma 4's context window with a **fixed 65K token allocation**.

**Why fixed?** Changing `num_ctx` at runtime triggers a full model unload/reload in Ollama — 300+ seconds for large models. PRE sets 65K once in the Modelfile, matches it exactly in every request, and the model cold-loads in ~1.5 seconds on M4 Max 128GB.

- **Context bar** — Shown after every response (color-coded: grey > yellow at 50% > red at 75%). Also visible in the web GUI topbar.
- **Auto-compaction** — At 75% usage (~49K tokens), older turns are summarized and compressed. The last 6 exchanges are kept intact.
- **Tool response cap** — Tool outputs truncated to 8KB to prevent context blowout.
- **Rewind** — `/rewind N` removes the last N turns to free space.
- **Channels** — Separate channels for separate tasks to avoid context pollution.

---

## Best Practices

### Getting Good Results

1. **Be specific.** "Review auth.py for SQL injection" > "check my code"
2. **Attach files.** `/file src/main.py` (CLI) or drag-and-drop (web GUI) then ask your question.
3. **Use PRE.md.** A project config file saves you from re-explaining your stack every session.
4. **Watch the thinking.** `/think` toggles the model's reasoning — useful for understanding its approach.
5. **Upload images.** Paste screenshots, drag photos — Gemma 4 analyzes them natively.

### Managing Context

1. **Check context regularly.** `/context` or the topbar bar in the web GUI.
2. **Use channels for separate tasks.** Don't mix debugging and feature work.
3. **Rewind noisy turns.** `/rewind` removes turns that added bulk without value.
4. **Let auto-compaction work.** It preserves recent context while compressing old turns.

### Scheduling

1. **Start simple.** "every day at 9am" — verify results before adding complex schedules.
2. **Be specific in prompts.** Good: "Check disk usage on / and /Users, alert if >80%". Bad: "check things".
3. **Use the Result button.** Each cron job run is a full chat session you can review and continue.
4. **Configure delivery channels.** Add Telegram and/or Slack tokens for push notifications even when away from the computer.

---

## Web GUI

PRE includes a built-in browser interface at `http://localhost:7749` that provides full access to all features through a modern, responsive web application.

### How It Works

**Node.js (Express + WebSocket) backend** with a **vanilla JS SPA frontend** — no React, no Vue, no bundler. It talks directly to Ollama, reads/writes the same JSONL session files as the CLI, executes tools server-side, and streams responses via WebSocket. Sessions are fully interchangeable between CLI and web.

### Features

- **Real-time streaming** — tokens appear as the model generates them, with thinking blocks, streaming cursor, and live tool status cards
- **Full tool execution** — all 45+ tools run server-side with the same multi-turn tool loop as the CLI (up to 25 autonomous tool calls per prompt)
- **File and image upload** — drag-and-drop, clipboard paste (Ctrl/Cmd+V), or file picker button. Images sent to model for analysis; text files included as context.
- **Image generation** — generates images locally via ComfyUI. Results display inline with Full Size, Download, and Show in Finder actions.
- **Document generation** — creates DOCX, XLSX, TXT, CSV, HTML documents. Download or open directly from the chat.
- **Artifact viewer** — rich HTML artifacts render in a slide-out right panel. Download or reveal in Finder.
- **Frontier AI delegation** — toggle between PRE (local), Claude, Codex, or Gemini via a dropdown next to the input. Unavailable CLIs are auto-hidden. Responses stream in real-time with color-coded model badges. Sessions preserve which model generated each message.
- **Cron scheduler** — visual panel for managing recurring jobs with natural language input, live cron preview, enable/disable toggles, and run-now buttons. Results delivered via macOS notification, Telegram, and in-browser toast.
- **Session management** — sidebar with searchable session list, create/rename/delete sessions, project organization with drag-and-drop
- **Memory browser** — view, search, and manage persistent memories from a dedicated panel
- **Connection manager** — configure all external services from the Settings panel (API keys, OAuth)
- **Context tracking** — live context window usage bar in the topbar
- **Tool confirmation** — dangerous tools show a confirmation dialog before executing
- **Three themes** — Dark, Light, and Evangelion (NERV-inspired orange-on-purple with hexagonal grid background)
- **Calendas Plus typography** — serif display font for headings, system-ui stack for body text
- **Auto-generated titles** — sessions are automatically named based on the first message
- **Responsive layout** — three-panel design (sidebar, chat, artifact panel) with mobile hamburger drawer

### Auto-Launch

The web GUI starts automatically when you run `pre-launch` (requires Node.js 18+). It runs in the background on port **7749** (configurable via `PRE_WEB_PORT`).

### Manual Launch

```bash
cd pre/web
npm install          # First time only
node server.js       # Starts on http://localhost:7749

# With custom settings:
PRE_WEB_PORT=8080 PRE_CWD=/path/to/project node server.js
```

---

## Frontier AI Delegation

PRE can delegate prompts to frontier AI models when cloud-level intelligence is needed, turning it into an **agentic orchestration layer** — use local inference for everyday tasks and escalate to the most powerful models on demand.

### Supported Models

| Delegate | CLI | Install |
|----------|-----|---------|
| **Claude** (Anthropic) | `claude` | `npm install -g @anthropic-ai/claude-code` |
| **Codex** (OpenAI) | `codex` | `npm install -g @openai/codex` |
| **Gemini** (Google) | `gemini` | `npm install -g @anthropic-ai/gemini-cli` |

### How It Works

1. PRE detects which CLIs are installed at startup and hides unavailable options
2. In the GUI, click the **delegate toggle** (next to the attachment button) to switch between PRE, Claude, Codex, or Gemini
3. Your prompt is sent to the selected CLI in text-only mode (no tool use) and the response streams back in real-time
4. Responses display with a **color-coded model badge** (orange for Claude, green for Codex, blue for Gemini)
5. Session history preserves which model generated each message — badges render on reload

The delegates run in non-interactive mode with tool use disabled (`claude --tools ""`, `codex -a suggest`, `gemini -o text`) so they return text responses only. The frontier model thinks; PRE acts.

---

## Telegram Integration

PRE includes a built-in Telegram bot for full agent access from your phone.

### Setup

1. Message [@BotFather](https://t.me/BotFather) on Telegram and create a new bot
2. In PRE, run `/connections add telegram` and paste the bot token
3. Restart PRE — the Telegram bot starts automatically

### How It Works

The bot long-polls the Telegram API (no webhook, no public URL required) and routes messages through the same Ollama instance. Full system access — all tools, same memory, same agentic workflows.

- **Owner authorization** — the first user to message becomes the owner
- **Cron delivery** — scheduled job results are sent to your Telegram chat
- **Automatic lifecycle** — starts with PRE, stops with PRE (Ctrl+C kills both)

### Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/new` | New conversation |
| `/status` | Bot status |
| `/memory` | List saved memories |
| `/help` | Show commands |

---

## Architecture

### System Diagram

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
│  (MPS/Metal)    │  │  vanilla JS SPA  │  45+ tools, cron runner
│ 25-step, 1024px │  │  shared sessions │  file upload, image gen
└─────────────────┘  └─────────────────┘

  ~/.pre/
  ├── identity.json       # Agent name
  ├── connections.json    # API keys and OAuth tokens
  ├── cron.json           # Scheduled recurring tasks
  ├── sessions/           # Conversation JSONL (shared by CLI + web + cron)
  ├── history             # Input history (arrow-key recall)
  ├── checkpoints/        # File backups for /undo
  ├── artifacts/          # HTML artifacts, documents, and generated images
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

### The PRE Binary

**PRE CLI** (`pre.m`) is a single-file Objective-C/C application (~10,000 lines) handling:
- Raw `recv()` NDJSON streaming with 64KB ring buffer
- Fixed `num_ctx=65536` matching Modelfile (avoids Ollama reload)
- 45+ tool implementations with two-tier permissions
- Hybrid tool calling: native Ollama `tools` API + text-based `<tool_call>` fallback
- Local image generation via ComfyUI (checkpoint-adaptive workflow)
- Multi-part artifacts with incremental append
- PDF export via native WebKit rendering
- Cron registry for recurring tasks
- Built-in Google OAuth 2.0 with multi-account support
- Persistent memory with per-project scoping
- Context compaction and token budget management
- File checkpointing and undo
- Ctrl+V image paste for multimodal queries
- linenoise-based line editor with tab completion

**Web GUI** (`web/`) is a Node.js Express + WebSocket server with vanilla JS SPA:
- NDJSON streaming client for Ollama
- Server-side tool execution (same tool loop as CLI)
- Headless cron runner with multi-channel notification delivery
- Session JSONL read/write (shared format with CLI)
- File/image upload (drag-and-drop, clipboard paste, file picker)
- Document generation (DOCX, XLSX via docx/exceljs, PDF via pdfkit)
- Three themes with CSS custom properties
- No framework, no bundler — under 3K lines total JS

**Telegram Bot** (`telegram.m`) is a companion binary (~2000 lines):
- Long-polling (no webhook needed)
- Full tool access matching CLI
- Owner-based authorization
- Automatic lifecycle management

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
├── cron.json           # Scheduled recurring tasks
├── sessions/           # Conversation JSONL (one per session)
├── history             # Readline history
├── checkpoints/        # File backups (auto-cleaned)
├── artifacts/          # Generated documents, images, artifacts
├── comfyui.json        # ComfyUI config (optional)
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
├── benchmark.sh            # Performance benchmarking tool
├── engine/
│   ├── pre.m               # PRE CLI (single-file, ~10000 lines)
│   ├── telegram.m          # Telegram bot bridge (~2000 lines)
│   ├── linenoise.c/h       # Terminal line editor (patched)
│   ├── Makefile             # Build: make pre telegram
│   ├── Modelfile            # Ollama model config (num_ctx=65536)
│   ├── pre-launch           # Universal launcher script
│   └── launch-telegram      # Standalone Telegram launcher
├── web/                     # Web GUI (Node.js + vanilla JS)
│   ├── server.js            # Express + WebSocket server
│   ├── package.json
│   ├── src/
│   │   ├── ollama.js        # Ollama NDJSON streaming client
│   │   ├── sessions.js      # JSONL read/write (shared with CLI)
│   │   ├── tools.js         # Tool dispatcher + execution loop
│   │   ├── tools-defs.js    # 45+ tool definitions for Ollama
│   │   ├── context.js       # System prompt builder
│   │   ├── memory.js        # Auto-extraction engine
│   │   ├── connections.js   # Credential management
│   │   ├── constants.js     # MODEL_CTX=65536, paths
│   │   ├── cron-runner.js   # Headless cron execution + notifications
│   │   └── tools/           # Tool implementations
│   │       ├── bash.js      # Shell execution
│   │       ├── files.js     # File operations
│   │       ├── web.js       # Web fetch/search
│   │       ├── memory.js    # Memory CRUD
│   │       ├── system.js    # System inspection + desktop
│   │       ├── artifact.js  # HTML artifact generation
│   │       ├── document.js  # DOCX/XLSX/PDF/TXT/CSV generation
│   │       ├── image.js     # ComfyUI image generation
│   │       ├── cron.js      # Cron job management
│   │       ├── github.js    # GitHub API
│   │       ├── google.js    # Gmail, Drive, Docs
│   │       ├── telegram.js  # Telegram Bot API
│   │       ├── jira.js      # Jira API
│   │       ├── confluence.js # Confluence API
│   │       ├── smartsheet.js # Smartsheet API
│   │       └── slack.js     # Slack API
│   └── public/
│       ├── index.html       # SPA shell
│       ├── favicon.svg
│       ├── fonts/           # Calendas Plus (regular, italic, bold)
│       ├── css/             # base, themes, chat, sidebar, components, animations
│       └── js/              # app, ws, chat, markdown, themes
└── docs/
    └── *.md                 # Research notes
```

---

## Contributing

PRE is built and maintained by **Christopher Bradford** — systems administrator at Joby Aviation and AI engineer.

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

## License

MIT License. See [LICENSE](LICENSE) for details.

The Gemma 4 model weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
