# PRE — Personal Reasoning Engine

> A fully local agentic assistant. No cloud. No API keys. No data leaves your machine.

PRE is a tool-calling, memory-equipped AI agent that runs entirely on your Mac. Powered by **Google Gemma 4 26B-A4B** (a Mixture-of-Experts model with 3.8B active parameters) via [Ollama](https://ollama.ai), PRE delivers **~70 tokens/second** on Apple Silicon with up to 35 integrated tools, persistent memory, project detection, and channel-based conversations.

The reference system is a **MacBook Pro with an M4 Max (128 GB unified memory)**.

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
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

```bash
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

**Run commands autonomously** — Bash execution with a three-tier permission model. Read-only tools auto-approve; writes confirm once; destructive operations confirm every time.

**Remember across sessions** — Persistent memory stores your preferences, project context, workflow patterns, and reference pointers. Memories survive restarts.

**Manage multiple workstreams** — Channels let you run parallel conversations with separate contexts within the same project.

**Deep system inspection** — Network interfaces, running processes, disk usage, hardware info, window management, screenshots, and arbitrary AppleScript automation.

**Connect to external services** — Optional integrations with Brave Search, GitHub, Google (Gmail, Drive, Docs), and Wolfram Alpha via `/connections`. OAuth 2.0 for Google, API keys for the rest. Tokens stored locally.

**Paste images for analysis** — Ctrl+V pastes clipboard images directly into the prompt. Gemma 4 is multimodal — it can analyze screenshots, diagrams, photos, and more.

**Respect your privacy** — Everything runs locally on your machine. Ollama serves the model, PRE manages the conversation. Connection-dependent tools make API calls to their respective services; all other tools are fully local.

---

## Installation

### Prerequisites

| Component | Required |
|-----------|----------|
| **macOS** | 14.0+ (Sonoma or later) |
| **Chip** | Apple Silicon (M1 or later) |
| **RAM** | 16 GB minimum, 32+ GB recommended |
| **Ollama** | [ollama.ai](https://ollama.ai) or `brew install ollama` |
| **Xcode CLI** | `xcode-select --install` |

### Install

```bash
# Pull the base model (Gemma 4 26B-A4B, MoE, ~17 GB)
ollama pull gemma4:26b-a4b-it-q4_K_M

# Clone and build
git clone https://github.com/sunkencity999/pre.git
cd pre/engine
make pre

# Optional: add to PATH
ln -sf "$(pwd)/pre-launch" ~/.local/bin/pre-launch
```

### Launch

```bash
pre-launch                         # From any directory
pre-launch --dir /path/to/project  # Override working directory
pre-launch --max-tokens 16384      # Allow longer responses
```

The launcher checks that Ollama is running (starts it if not), creates the optimized `pre-gemma4` model from the Modelfile if needed (sets `num_ctx 262144`, `num_batch 512`), and launches the PRE binary.

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

#### Help

| Command | Description |
|---------|-------------|
| `/help` | Command overview |
| `/help tools` | All 35 tools with permission levels |
| `/help memory` | Memory system guide |
| `/help channels` | Channel system guide |
| `/help projects` | Project detection & PRE.md |
| `/help tips` | Best practices and tips |
| `/help all` | Everything at once |

---

### Tools

PRE has 29 built-in tools plus up to 6 connection-dependent tools (35 total) that the model can call autonomously. PRE is designed for power users — nearly all tools auto-execute without confirmation:

- **Auto** — executes immediately, no confirmation needed (32 of 35 tools)
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

#### Memory

| # | Tool | Args | Description |
|---|------|------|-------------|
| 26 | `memory_save` | `name`, `type`, `description`, `content`, `scope`? | Save a persistent memory (global or project-scoped) |
| 27 | `memory_search` | `query`? | Search saved memories |
| 28 | `memory_list` | *(none)* | List all memories |
| 29 | `memory_delete` | `query` | Delete a memory *(confirm always)* |

#### Connection-Dependent Tools

These tools require external API keys or OAuth setup via `/connections`. Run `/connections` to configure.

| # | Tool | Connection | Args | Description |
|---|------|------------|------|-------------|
| 30 | `web_search` | Brave Search | `query`, `count`? | Web search via Brave Search API |
| 31 | `github` | GitHub | `action`, `repo`?, `query`?, `number`?, `state`? | GitHub API (search repos, issues, PRs, user info) |
| 32 | `gmail` | Google | `action`, `query`?, `id`?, `to`?, `subject`?, `body`?, `cc`?, `bcc`?, `max_results`? | Gmail (search, read, send, draft, trash, labels, profile) |
| 33 | `gdrive` | Google | `action`, `id`?, `path`?, `name`?, `folder_id`?, `query`?, `email`?, `role`?, `count`? | Google Drive (list, search, download, upload, mkdir, share, delete) |
| 34 | `gdocs` | Google | `action`, `id`?, `title`?, `content`? | Google Docs (create, read, append) |
| 35 | `wolfram` | Wolfram Alpha | `query` | Computation, math, science, data queries |

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
| **Brave Search** | API key | `web_search` |
| **GitHub** | Personal access token | `github` |
| **Google** | OAuth 2.0 | `gmail`, `gdrive`, `gdocs` |
| **Wolfram Alpha** | API key | `wolfram` |

Google OAuth uses a local HTTP callback server — no data leaves your machine except for the API calls themselves. Tokens are stored locally and refreshed automatically.

---

### Context Management

PRE uses Gemma 4's full 262K token context window. A few features help you stay within budget:

**Context bar** — Shown after every response (color-coded: grey → yellow at 75% → red at 90%). Also available via `/context`.

**Auto-compaction** — When estimated tokens exceed 75% of the budget, older conversation turns are automatically summarized and compressed. The last 6 exchanges are kept intact.

**Server-reported tokens** — PRE uses Ollama's native API which reports exact prompt and generation token counts, giving you accurate context usage instead of estimates.

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

## Architecture

### How PRE Works

```
┌─────────────┐    Ollama Native    ┌─────────────────┐
│   PRE CLI   │  ◄─ /api/chat ──►  │     Ollama       │
│  (pre.m)    │   raw recv() NDJSON │  localhost:11434  │
│  ~6000 lines│                     │                   │
│  Obj-C / C  │                     │  Gemma 4 26B-A4B  │
└─────────────┘                     │  MoE, q4_K_M      │
      │                             │  ~70 tok/s         │
      ▼                             └───────────────────┘
  ~/.pre/
  ├── sessions/     # Conversation JSONL files
  ├── history       # Input history (arrow-key recall)
  ├── checkpoints/  # File backups for /undo
  ├── connections/  # API keys and OAuth tokens
  ├── memory/       # Global persistent memories
  │   ├── index.md
  │   └── *.md
  └── projects/
      └── {name}/
          ├── memory/    # Project-scoped memories
          └── channels/  # Channel metadata
```

**PRE CLI** (`pre.m`) is a single-file Objective-C/C application. It handles:
- Ollama native API client with raw `recv()` NDJSON streaming (no stdio buffering)
- System prompt as `role:system` message for KV cache prefix reuse across turns
- Streaming markdown renderer with ANSI formatting
- Multi-format tool call parser (JSON, XML, bare)
- Up to 35 tool implementations with three-tier permissions
- OAuth 2.0 flow with local HTTP callback for Google APIs
- Connection management for external services (Brave, GitHub, Google, Wolfram)
- Persistent memory with per-project scoping
- Channel-based conversation management
- Project detection and PRE.md loading
- Context compaction and token budget management (262K window)
- File checkpointing and undo
- Ctrl+V image paste for multimodal queries (via AppKit)
- linenoise-based line editor with tab completion and ANSI-aware cursor

**Ollama** serves the model via a custom Modelfile (`pre-gemma4`) with tuned `num_ctx`, `num_batch`, and `keep_alive` for optimal performance. PRE uses the native `/api/chat` endpoint with server-reported token counts.

**Gemma 4 26B-A4B** is a Mixture-of-Experts model: 26B total parameters, 3.8B active per token, 128 experts with 8 active. At q4_K_M quantization (~17 GB), it runs at ~70 tok/s on M4 Max with 262K context — fast enough for real-time agentic workflows.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRE_PORT` | `11434` | Ollama server port |
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
├── sessions/           # Conversation JSONL (one per channel)
├── history             # Readline history
├── checkpoints/        # File backups (auto-cleaned)
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
├── system.md               # Model system prompt reference
├── engine/
│   ├── pre.m               # PRE CLI (single-file, ~6000 lines)
│   ├── linenoise.c/h       # Terminal line editor (patched: Ctrl+V, ANSI-aware)
│   ├── Makefile             # Build: make pre
│   ├── Modelfile            # Ollama model config (pre-gemma4)
│   └── pre-launch           # Universal launcher script
├── docs/
│   └── *.md                 # Research notes from Flash-MoE era
└── benchmark_results/       # Performance benchmarks
```

---

## Contributing

PRE is built and maintained by **Christopher Bradford** — systems administrator at Joby Aviation and AI engineer.

**Areas for contribution:**
- **New tools** — Calendar integration, git operations, Slack, linear/Jira
- **Smarter memory** — Auto-extraction of important facts, memory relevance ranking
- **Better rendering** — Syntax highlighting in code blocks, image display in terminal
- **Model support** — Test with other Ollama models (Llama 4, Qwen 3, etc.)
- **New connections** — Slack, Discord, Notion, Linear, and other service integrations
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
- **Apple** for unified memory architecture
- The original **Flash-MoE** research that informed PRE's early design

## License

MIT License. See [LICENSE](LICENSE) for details.

The Gemma 4 model weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
