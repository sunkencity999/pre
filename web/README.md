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
- **Full tool execution** — all 37+ tools run server-side with multi-turn agentic loop (up to 25 tool calls per prompt)
- **Shared sessions** — same JSONL format as CLI, fully interchangeable
- **Three themes** — Dark, Light, and Evangelion (NERV-inspired)
- **Calendas Plus typography** — serif display font for headings
- **Tool confirmation dialogs** — dangerous tools require approval
- **Context tracking** — live context window usage bar
- **Responsive** — three-panel desktop, hamburger drawer on mobile

## Architecture

```
server.js                  Express + WebSocket, REST API
src/
  ollama.js                Ollama /api/chat NDJSON streaming
  sessions.js              JSONL read/write (shared with CLI)
  tools.js                 Tool dispatcher + execution loop
  tools-defs.js            37 tool definitions for Ollama
  context.js               System prompt builder
  constants.js             MODEL_CTX=65536, paths
  tools/
    bash.js                Shell execution (stderr capture)
    files.js               read_file, list_dir, glob, grep, file_write, file_edit
    web.js                 web_fetch, web_search (Brave API)
    memory.js              save, search, list, delete
    system.js              17 system tools (info, processes, clipboard, etc.)
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
