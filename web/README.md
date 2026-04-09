# PRE Web GUI

Browser interface for the Personal Reasoning Engine. Runs alongside the CLI, sharing the same sessions, memory, and tools.

## Quick Start

```bash
npm install      # First time only
node server.js   # http://localhost:7749
```

Or let `pre-launch` start it automatically — the web GUI launches in the background when you run PRE.

## Features

- **Real-time streaming** via WebSocket — tokens appear as the model generates them
- **Full tool execution** — 45+ tools run server-side with multi-turn agentic loop (up to 25 tool calls per prompt)
- **Shared sessions** — same JSONL format as CLI, fully interchangeable
- **Projects** — group related sessions into collapsible project folders with drag-and-drop
- **Connections GUI** — configure all integrations from Settings (gear icon in sidebar)
- **Persistent memory** — file-based memory system with auto-extraction, age annotations, and a GUI browser
- **Three themes** — Dark, Light, and Evangelion (NERV-inspired)
- **Calendas Plus typography** — serif display font for headings
- **Thinking indicator** — animated bouncing dots during model's first-token latency
- **Tool confirmation dialogs** — dangerous tools require approval
- **Context tracking** — live context window usage bar
- **Auto-generated titles** — sessions are named from the first message
- **Responsive** — three-panel desktop, hamburger drawer on mobile

## Integrations

All integrations are configured through the Settings panel (gear icon in sidebar footer). Credentials are stored in `~/.pre/connections.json` and never exposed to the browser.

| Service | Auth | Actions |
|---------|------|---------|
| **Brave Search** | API Key | Web search |
| **GitHub** | Personal Access Token | Repos, issues, PRs, user profile |
| **Google** (Gmail, Drive, Docs) | OAuth 2.0 | Email search/read/send/draft, Drive list/search/upload, Docs create/read/append |
| **Telegram** | Bot Token + Chat ID | Send/receive messages, auto-detects chat ID |
| **Jira Server** | URL + Personal Access Token | Search, issues (CRUD), comments, transitions, assignments, projects |
| **Confluence Server** | URL + Personal Access Token | Search (CQL), pages (CRUD), spaces, child pages, comments |
| **Smartsheet** | API Access Token | Sheets (CRUD), rows (add/update/delete), columns, workspaces, search, comments |
| **Slack** | Bot User OAuth Token | Channels, message history, send/reply/update, reactions, search, users |
| **Wolfram Alpha** | API Key | Computational queries |

## Memory System

PRE has a persistent, file-based memory system shared between the CLI and web GUI. Memories survive across sessions and are automatically injected into the system prompt.

**Storage:** `~/.pre/memory/*.md` — one fact per file, YAML frontmatter + markdown body. `MEMORY.md` serves as a human-readable index.

**Memory types:**
- **user** — Who the user is (role, preferences, expertise)
- **feedback** — How to work (corrections, confirmed approaches)
- **project** — What's happening (decisions, deadlines, context not in code/git)
- **reference** — Where to look (external system pointers, URLs)

**Auto-extraction:** After each conversation turn, a lightweight LLM pass analyzes recent messages and silently saves memory-worthy facts. Duplicate detection prevents redundant saves.

**Age annotations:** When memories are injected into context, stale memories (>7 days) get a warning so the model knows to verify before acting on them.

**GUI browser:** Click the memory icon (sidebar footer) to browse, search, create, and delete memories. Memories are grouped by type with color coding.

**Context injection priority:** feedback > user > project > reference (behavioral guidance first).

## Architecture

```
server.js                  Express + WebSocket, REST API
src/
  ollama.js                Ollama /api/chat NDJSON streaming
  sessions.js              JSONL read/write + project management
  tools.js                 Tool dispatcher + execution loop
  tools-defs.js            45+ tool definitions for Ollama
  context.js               System prompt builder
  memory.js                 Enhanced memory system (save, extract, age, context injection)
  connections.js            Connection management, Google OAuth, Telegram setup
  constants.js             MODEL_CTX=65536, paths
  tools/
    bash.js                Shell execution (stderr capture)
    files.js               read_file, list_dir, glob, grep, file_write, file_edit
    web.js                 web_fetch, web_search (Brave API)
    memory.js              save, search, list, delete
    system.js              17 system tools (info, processes, clipboard, etc.)
    artifact.js            Interactive HTML artifacts
    document.js            Document export (txt, xml, docx, xlsx, pdf)
    google.js              Gmail, Google Drive, Google Docs
    telegram.js            Telegram Bot API
    github.js              GitHub REST API
    jira.js                Jira Server REST API v2
    confluence.js          Confluence Server REST API
    smartsheet.js          Smartsheet REST API 2.0
    slack.js               Slack Web API
public/
  index.html               SPA shell
  fonts/                   Calendas Plus (regular, italic, bold)
  css/                     6 stylesheets (base, themes, chat, sidebar, components, animations)
  js/                      5 modules (app, ws, chat, markdown, themes)
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PRE_WEB_PORT` | `7749` | Server port |
| `PRE_CWD` | `$HOME` | Working directory for tools |
| `PRE_PORT` | `11434` | Ollama server port |

## Themes

Switch between themes using the buttons in the sidebar footer.

- **Dark** — `#0a0a0a` background, blue primary, clean and minimal
- **Light** — `#fafafa` background, blue primary, bright and readable
- **Evangelion** — deep purple-black background, NERV orange primary, amber text, hexagonal grid pattern, scan-line animations on focused elements

## Key Design Decisions

1. **Always send `num_ctx=65536`** — must match the Modelfile exactly to prevent Ollama model reload (300s+ penalty)
2. **Vanilla JS, no framework** — local tool, one user, loads instantly, no build step
3. **Credentials stay server-side** — API keys never sent to the browser
4. **Tool execution server-side** — security, CORS, file system access
5. **Session format compatibility** — CLI and web can interleave on the same session files
