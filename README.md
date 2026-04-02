# PRE — Personal Reasoning Engine

> A fully local agentic assistant. No cloud. No API keys. No data leaves your machine.

PRE is a tool-calling, memory-equipped AI agent that runs entirely on your Mac. Powered by **Google Gemma 4 26B-A4B** (a Mixture-of-Experts model with 3.8B active parameters) via [Ollama](https://ollama.ai), PRE delivers **~56 tokens/second** on Apple Silicon with 29 integrated tools, persistent memory, project detection, and channel-based conversations.

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

# 2. Pull the model (~17 GB)
ollama pull gemma4:26b-a4b-it-q4_K_M

# 3. Clone and build PRE
git clone https://github.com/sunkencity999/pre.git
cd pre/engine
make pre

# 4. Launch
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

**Respect your privacy** — Everything runs locally on your machine. Ollama serves the model, PRE manages the conversation. Nothing phones home.

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
# Pull the model (Gemma 4 26B-A4B, MoE, ~17 GB)
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

The launcher checks that Ollama is running (starts it if not), verifies the model is pulled, and launches the PRE binary.

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
| `/help tools` | All 29 tools with permission levels |
| `/help memory` | Memory system guide |
| `/help channels` | Channel system guide |
| `/help projects` | Project detection & PRE.md |
| `/help tips` | Best practices and tips |
| `/help all` | Everything at once |

---

### Tools

PRE has 29 built-in tools the model can call autonomously. Each has a permission level:

#### File & Code (auto-approved)

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `list_dir` | List directory |
| `glob` | Find files by pattern |
| `grep` | Search file contents (regex) |

#### File Modification (confirm once)

| Tool | Description |
|------|-------------|
| `file_write` | Create or overwrite a file (checkpointed) |
| `file_edit` | Find-and-replace in a file (checkpointed) |

#### Shell (confirm always)

| Tool | Description |
|------|-------------|
| `bash` | Execute a shell command |

#### System Inspection (auto-approved)

| Tool | Description |
|------|-------------|
| `system_info` | CPU, memory, disk, battery |
| `hardware_info` | Detailed hardware, thermal, GPU |
| `process_list` | Running processes (with optional filter) |
| `disk_usage` | Volume and directory usage |
| `display_info` | Display and GPU details |

#### Network (auto-approved)

| Tool | Description |
|------|-------------|
| `net_info` | Interfaces, IPs, DNS, routes |
| `net_connections` | TCP connections (listening/established/port) |
| `service_status` | launchd service listing |

#### Desktop Integration

| Tool | Permission | Description |
|------|-----------|-------------|
| `screenshot` | Confirm once | Capture screen (full/window/region) |
| `window_list` | Auto | List open windows with positions |
| `window_focus` | Confirm once | Bring an app to front |
| `clipboard_read` | Auto | Read clipboard |
| `clipboard_write` | Confirm once | Write to clipboard |
| `open_app` | Confirm first | Open files/apps/URLs via macOS `open` |
| `notify` | Confirm once | macOS notification |
| `applescript` | Confirm always | Run arbitrary AppleScript |

#### Web

| Tool | Permission | Description |
|------|-----------|-------------|
| `web_fetch` | Confirm once | Fetch a URL (HTML→text conversion) |

#### Memory

| Tool | Permission | Description |
|------|-----------|-------------|
| `memory_save` | Auto | Save a persistent memory (global or project-scoped) |
| `memory_search` | Auto | Search saved memories |
| `memory_list` | Auto | List all memories |
| `memory_delete` | Confirm once | Delete a memory |

#### Process Control

| Tool | Permission | Description |
|------|-----------|-------------|
| `process_kill` | Confirm always | Send SIGTERM to a process |

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

### Context Management

PRE uses Gemma 4's 128K token context window. A few features help you stay within budget:

**Context bar** — `/context` shows a visual progress bar of usage.

**Auto-compaction** — When estimated tokens exceed 75% of the budget, older conversation turns are automatically summarized and compressed. The last 6 exchanges are kept intact.

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

1. **Trust the permission model.** Read-only tools are auto-approved. Write tools ask once. Destructive tools ask every time.
2. **Use 'a' for auto-approve.** When prompted for confirmation, answer `a` to auto-approve all confirm-once tools for the session.
3. **Undo mistakes.** `/undo` reverts the last file_write or file_edit.
4. **The model is agentic.** It will chain tool calls to accomplish multi-step tasks — read code, search for patterns, make edits, verify results.

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
┌─────────────┐     HTTP/SSE      ┌─────────────────┐
│   PRE CLI   │ ◄──────────────► │     Ollama       │
│  (pre.m)    │                   │  localhost:11434  │
│  4100 lines │                   │                   │
│  Obj-C / C  │                   │  Gemma 4 26B-A4B  │
└─────────────┘                   │  MoE, q4_K_M      │
      │                           │  ~56 tok/s         │
      ▼                           └───────────────────┘
  ~/.pre/
  ├── sessions/     # Conversation JSONL files
  ├── history       # Input history (arrow-key recall)
  ├── checkpoints/  # File backups for /undo
  ├── memory/       # Global persistent memories
  │   ├── index.md
  │   └── *.md
  └── projects/
      └── {name}/
          ├── memory/    # Project-scoped memories
          └── channels/  # Channel metadata
```

**PRE CLI** (`pre.m`) is a single-file Objective-C/C application. It handles:
- OpenAI-compatible HTTP/SSE client for Ollama
- Streaming markdown renderer with ANSI formatting
- Multi-format tool call parser (JSON, XML, bare)
- 29 tool implementations with three-tier permissions
- Persistent memory with per-project scoping
- Channel-based conversation management
- Project detection and PRE.md loading
- Context compaction and token budget management
- File checkpointing and undo
- linenoise-based line editor with tab completion

**Ollama** serves the model. PRE connects as a standard OpenAI API client, sending the full conversation history with each request (Ollama is stateless).

**Gemma 4 26B-A4B** is a Mixture-of-Experts model: 26B total parameters, 3.8B active per token, 128 experts with 8 active. At q4_K_M quantization (~17 GB), it runs at ~56 tok/s on M4 Max — fast enough for real-time agentic workflows.

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
├── system.md               # Model system prompt
├── engine/
│   ├── pre.m               # PRE CLI (single-file, ~4100 lines)
│   ├── linenoise.c/h       # Terminal line editor
│   ├── Makefile             # Build: make pre
│   └── pre-launch           # Universal launcher script
├── docs/
│   └── *.md                 # Research notes from Flash-MoE era
└── benchmark_results/       # Performance benchmarks
```

---

## Contributing

PRE is built and maintained by **Christopher Bradford** — systems administrator at Joby Aviation and AI engineer.

**Areas for contribution:**
- **New tools** — Calendar integration, git operations, image analysis (Gemma 4 has vision)
- **Smarter memory** — Auto-extraction of important facts, memory relevance ranking
- **Better rendering** — Syntax highlighting in code blocks, image display
- **Model support** — Test with other Ollama models (Llama 4, Qwen 3, etc.)
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
