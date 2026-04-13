# PRE Web GUI

Browser interface for the Personal Reasoning Engine. Runs alongside the CLI, sharing the same sessions, memory, and tools.

## Quick Start

```bash
npm install      # First time only
node server.js   # http://localhost:7749
```

Or let `pre-launch` start it automatically — the web GUI launches in the background when you run PRE.

## Features

- **Computer Use** — desktop automation via screenshot + mouse/keyboard control; the model sees your screen and operates any application
- **Real-time streaming** via WebSocket — tokens appear as the model generates them
- **Full tool execution** — 55+ tools run server-side with multi-turn agentic loop (up to 25 tool calls per prompt)
- **Local image generation** — ComfyUI + Stable Diffusion XL on Apple Silicon GPU, inline image preview in chat
- **Shared sessions** — same JSONL format as CLI, fully interchangeable
- **Projects** — group related sessions into collapsible project folders with drag-and-drop
- **Connections GUI** — configure all integrations from Settings (gear icon in sidebar)
- **Persistent memory** — file-based memory system with auto-extraction, age annotations, and a GUI browser
- **Two themes** — Dark and Light
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
| **Microsoft SharePoint** | OAuth 2.0 (Azure AD) | Search sites, browse drives/files, read/upload files, list items, pages |
| **Wolfram Alpha** | API Key | Computational queries |
| **Linear** | Personal API Key | Issues (CRUD), search, projects, teams, cycles, labels, comments |
| **Zoom** | Server-to-Server OAuth | Meetings (CRUD), recordings, users |
| **Figma** | Personal Access Token | Files, nodes, comments, projects, versions, image export |
| **Asana** | Personal Access Token | Tasks (CRUD), projects, workspaces, sections, search, comments |

## Computer Use (Desktop Automation)

PRE's standout capability: control **any desktop application** through a vision loop — the model takes a screenshot, sees what's on screen via its vision model, and issues click/type/scroll commands. Each action returns a fresh screenshot, so the model can verify results and self-correct. This is the same paradigm as Anthropic's Computer Use, running entirely locally.

### Requirements

- **cliclick** (Homebrew): `brew install cliclick`
- **Accessibility permissions**: macOS will prompt to allow terminal access for screen recording and input events

### How It Works

1. The model calls `computer` with `action: screenshot` to see the desktop
2. The screenshot is sent as a base64 image — the model sees it via vision
3. The model decides what to do: click at coordinates, type text, press keys
4. Each action returns a new screenshot so the model sees the result
5. The loop continues until the task is complete

This is the same vision-action loop used by the browser tool, extended to the entire desktop.

### Available Actions

| Action | Parameters | Description |
|--------|------------|-------------|
| `screenshot` | — | Capture the full desktop (returns base64 PNG for vision) |
| `click` | `x`, `y` | Left-click at pixel coordinates |
| `double_click` | `x`, `y` | Double-click at coordinates (open files, select words) |
| `right_click` | `x`, `y` | Right-click for context menus |
| `type` | `text` | Type text character-by-character into the focused element |
| `key` | `key` | Press a key or combo — see Key Reference below |
| `scroll` | `direction`, `amount`, `x`, `y` (optional) | Scroll up or down at a position |
| `move` | `x`, `y` | Move cursor without clicking |
| `drag` | `from_x`, `from_y`, `to_x`, `to_y` | Click-and-drag between two points |
| `screen_size` | — | Get display resolution (width × height) |
| `cursor_position` | — | Get current cursor pixel coordinates |

### Key Reference

Single keys: `return`, `tab`, `esc`, `space`, `delete`, `arrow-up`, `arrow-down`, `arrow-left`, `arrow-right`, `home`, `end`, `page-up`, `page-down`, `f1`–`f16`

Modifier combos: `cmd+c`, `cmd+v`, `cmd+a`, `cmd+z`, `ctrl+a`, `shift+tab`, `cmd+shift+z`, `alt+tab`

### Example Uses

**Open an application and navigate menus:**
> "Open System Preferences and navigate to the Display settings"

The model takes a screenshot, finds and clicks the System Preferences icon, waits for it to open (new screenshot), then clicks the Display pane.

**Fill out a form in a desktop app:**
> "Open the Notes app, create a new note, and type a shopping list"

The model launches Notes, clicks the new-note button, then types the content line by line.

**Multi-step file management:**
> "In Finder, go to my Downloads folder and move all PDFs into a new folder called 'Documents'"

The model uses screenshots to navigate Finder, selects files, creates folders, and drags items.

**Keyboard shortcuts for power users:**
> "Take a screenshot of the current window and save it to the desktop"

The model presses `cmd+shift+4` then `space` to capture a window screenshot.

**Interact with any app PRE doesn't have an API for:**
> "Open Slack desktop and send 'heading to lunch' in the #general channel"

Computer Use works with any GUI application — it doesn't need an API integration.

## Linear (Issue Tracking)

PRE integrates with [Linear](https://linear.app) for issue tracking, project management, and cycle planning via the GraphQL API.

### Setup

1. Go to [Linear Settings → API](https://linear.app/settings/api) and create a **Personal API Key**
2. In PRE, open Settings (gear icon) → find **Linear** → paste your API key → Save

### Available Actions

| Action | Description |
|--------|-------------|
| `me` | Get your Linear profile (name, email, admin status) |
| `list_teams` | List all teams with key, name, and description |
| `list_projects` | List projects with state, progress %, lead, and dates |
| `list_issues` | List issues with optional filters by team, project, or status |
| `search` | Full-text search across all issues |
| `get_issue` | Get issue details including description, comments, labels, and cycle |
| `create_issue` | Create a new issue (requires team key, e.g. `ENG`) |
| `update_issue` | Update issue title, description, status, priority, or assignee |
| `add_comment` | Add a comment to an issue |
| `list_cycles` | List sprint cycles with progress and completion stats |
| `list_labels` | List all labels with colors and descriptions |

### Example Uses

> "Show me all open issues assigned to me in the ENG team"

> "Create a bug report in the PLATFORM project: 'OAuth refresh token expires prematurely'"

> "What's the progress on the current sprint cycle for the Mobile team?"

> "Search Linear for issues mentioning 'database migration'"

---

## Zoom (Meetings & Recordings)

PRE integrates with [Zoom](https://zoom.us) for meeting management and recording access via Server-to-Server OAuth.

### Setup

Zoom uses Server-to-Server OAuth (no browser sign-in flow needed):

1. Go to the [Zoom App Marketplace](https://marketplace.zoom.us/) → **Develop** → **Build App**
2. Choose **Server-to-Server OAuth** app type
3. Fill in the required info and note your **Account ID**, **Client ID**, and **Client Secret**
4. Under **Scopes**, add: `meeting:read:admin`, `meeting:write:admin`, `recording:read:admin`, `user:read:admin`
5. Activate the app
6. In PRE, open Settings → find **Zoom** → click **Setup** → enter the three credentials → Save

### Available Actions

| Action | Description |
|--------|-------------|
| `me` | Get your Zoom profile (name, email, timezone, account type) |
| `list_users` | List active users in your Zoom account |
| `list_meetings` | List upcoming, previous, or all meetings for a user |
| `get_meeting` | Get meeting details (agenda, join URL, passcode, settings) |
| `create_meeting` | Schedule a new meeting with topic, time, duration, and agenda |
| `update_meeting` | Update meeting topic, time, duration, or agenda |
| `delete_meeting` | Delete a meeting (requires confirmation) |
| `list_recordings` | List cloud recordings with download URLs |

### Example Uses

> "Schedule a team standup for tomorrow at 9:30 AM Pacific, 30 minutes, recurring daily"

> "List all my meetings for this week"

> "Get the recording download links for last Friday's all-hands meeting"

> "Delete the meeting with ID 12345678"

---

## Figma (Design)

PRE integrates with [Figma](https://figma.com) for inspecting designs, exporting assets, and managing comments via the REST API.

### Setup

1. Go to [Figma Settings → Personal Access Tokens](https://www.figma.com/settings) and generate a new token
2. In PRE, open Settings → find **Figma** → paste your token → Save

### Available Actions

| Action | Description |
|--------|-------------|
| `me` | Get your Figma profile (handle, email) |
| `get_file` | Get file metadata — pages, components, styles, thumbnail |
| `get_file_nodes` | Get specific node details by ID (name, type, dimensions) |
| `get_comments` | List comments on a file with resolved status |
| `post_comment` | Post a comment or reply to a thread on a file |
| `list_projects` | List projects in a team |
| `get_project_files` | List files in a project with thumbnails |
| `get_file_versions` | List version history for a file |
| `get_images` | Export nodes as PNG, SVG, or PDF at a specified scale |

### Example Uses

> "Show me the structure of the Figma file at figma.com/file/abc123/MyDesign"

> "Export the hero banner component from file abc123 as a 2x PNG"

> "List all unresolved comments on the mobile app design file"

> "What files are in the 'Q2 Redesign' project for team 12345?"

---

## Asana (Project Management)

PRE integrates with [Asana](https://asana.com) for task management, project tracking, and team collaboration via the REST API.

### Setup

1. Go to [Asana Developer Console](https://app.asana.com/0/developer-console) → **Personal Access Tokens** → **Create new token**
2. In PRE, open Settings → find **Asana** → paste your token → Save

### Available Actions

| Action | Description |
|--------|-------------|
| `me` | Get your profile and list of accessible workspaces |
| `list_workspaces` | List all workspaces with name and GID |
| `list_projects` | List projects with owner, due date, and status |
| `get_project` | Get project details including members, dates, and notes |
| `list_tasks` | List tasks in a project, section, or by assignee |
| `get_task` | Get task details with custom fields, subtasks, and comments |
| `create_task` | Create a task with name, assignee, due date, and project |
| `update_task` | Update task fields or mark as complete |
| `add_comment` | Add a comment to a task |
| `search` | Search tasks across a workspace with optional filters |
| `list_sections` | List sections in a project |

### Example Uses

> "Show me all tasks assigned to me in the 'Website Redesign' project"

> "Create a task in the Engineering project: 'Update API rate limiting' due next Friday, assigned to me"

> "Search for tasks mentioning 'onboarding' in my workspace"

> "Mark task 1234567890 as complete and add a comment: 'Deployed to production'"

---

## Microsoft SharePoint Setup

PRE integrates with Microsoft SharePoint via the Microsoft Graph API. One Azure AD app registration serves your entire organization — each user authenticates with their own Microsoft credentials.

### 1. Register an Azure AD App (admin, one-time)

1. Go to [Azure App Registrations](https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade)
2. Click **New registration**
   - **Name:** `PRE - Personal Reasoning Engine` (or any name)
   - **Supported account types:** "Accounts in this organizational directory only" (single-tenant)
   - **Redirect URI:** Select **Web** platform, enter `http://localhost:7749/oauth/microsoft/callback`
3. After creation, note the **Application (client) ID** and **Directory (tenant) ID** from the Overview page
4. Go to **Certificates & secrets** > **New client secret** — copy the **Value** (not the Secret ID)
5. Go to **API permissions** > **Add a permission** > **Microsoft Graph** > **Delegated permissions**:
   - `Sites.Read.All` — Read SharePoint sites
   - `Files.ReadWrite.All` — Read and write files in SharePoint document libraries
   - `User.Read` — Sign in and read user profile
6. Click **Grant admin consent** (requires Global Admin)

### 2. Connect in PRE (each user)

1. Open PRE Web GUI (`http://localhost:7749`)
2. Click the gear icon (Settings) in the sidebar footer
3. Find **Microsoft SharePoint** and click **Setup**
4. Enter:
   - **Tenant ID** — Directory (tenant) ID from the app registration
   - **Client ID** — Application (client) ID from the app registration
   - **Client Secret** — The secret value you created
5. Click **Save & Authorize** — a Microsoft sign-in window will open
6. Sign in with your Microsoft work account and grant the requested permissions
7. The window will close automatically and SharePoint tools become available

### 3. Available Actions

| Action | Description |
|--------|-------------|
| `search` | Search across all SharePoint sites |
| `list_sites` | List available SharePoint sites |
| `list_subsites` | List subsites under a site |
| `list_drives` | List document libraries with storage quota |
| `site_usage` | Aggregate storage usage across all drives |
| `list_files` | Browse files in a drive or folder |
| `get_recent` | Recently modified files in a drive |
| `read_file` | Download and read a file's contents |
| `get_file_metadata` | File details (size, author, dates) without downloading |
| `upload_file` | Upload a file to a SharePoint drive |
| `create_folder` | Create a new folder in a drive |
| `move_file` | Move or rename a file |
| `copy_file` | Copy a file to another location |
| `delete_file` | Delete a file from a drive (requires user confirmation) |
| `list_lists` | List SharePoint lists in a site |
| `get_columns` | Get column schema for a list (field names, types, choices) |
| `list_items` | Read items from a SharePoint list |
| `create_list_item` | Add a new item to a list |
| `update_list_item` | Update fields on an existing list item |
| `get_page` | Read a SharePoint page |

### FAQ

**Do other users in my company need their own app registration?**
No. One Azure AD app registration serves the entire organization. Each user just needs the Tenant ID, Client ID, and Client Secret (shared by IT), then signs in with their own Microsoft account.

**Does the token refresh automatically?**
Yes. PRE automatically refreshes the access token when it expires. You only need to sign in once.

## Image Generation

PRE includes local image generation via ComfyUI + Stable Diffusion XL, running entirely on Apple Silicon GPU. No cloud APIs, no data leaves your machine.

### Setup

Image generation is set up during `install.sh` (Step 8b). If you skipped it, re-run the installer — it will detect existing components and only install what's missing.

**Requirements:** ~8GB disk, Python 3.10-3.13, 48GB+ unified memory recommended

### How It Works

1. Ask PRE to generate an image (e.g., "generate a photo of Twin Lakes Beach at sunset")
2. The `image_generate` tool starts ComfyUI automatically on first use (port 8188)
3. A workflow is submitted with the appropriate parameters for your checkpoint
4. The generated PNG is saved to `~/.pre/artifacts/{date}/` and displayed inline in chat

### Model Options

| Model | Speed | Quality | Resolution |
|-------|-------|---------|------------|
| **Juggernaut XL v9** (default) | ~30-45s | Photorealistic, excellent faces | 1024x1024 |
| **SDXL Turbo** | ~5s | Good for drafts | 512x512 |

Switch models by editing the `checkpoint` field in `~/.pre/comfyui.json`.

### Architecture

ComfyUI runs as a background process, started on-demand by either the CLI or web GUI:

- **CLI** (`pre.m`) — `comfyui_ensure()` starts ComfyUI before generating
- **Web GUI** (`src/tools/image.js`) — `startComfyUI()` spawns the server and polls until ready
- Both write to the same artifacts directory and share `~/.pre/comfyui.json` config
- ComfyUI is **not** started by `pre-launch` — it only runs when needed to save GPU memory

## Memory System

PRE has a persistent, file-based memory system shared between the CLI (`pre.m`) and web GUI. Memories survive across sessions and are automatically injected into the system prompt so the model remembers who you are, how you work, and what's going on.

### How It Works

Memories are stored as individual Markdown files in `~/.pre/memory/`, each with YAML frontmatter:

```markdown
---
name: user_role
description: User is a systems admin and AI engineer at Joby Aviation
type: user
scope: global
created: 2026-04-09
---

Christopher is a Systems Administrator at Joby Aviation and AI Engineer
based in Boulder Creek, CA. Focus areas include fine-tuning transformers,
aviation domain models, and system administration.
```

`MEMORY.md` serves as a human-readable index of all memories (auto-maintained).

### Memory Types

| Type | Purpose | Example |
|------|---------|---------|
| **user** | Who the user is — role, preferences, expertise | "Deep Go expertise, new to React" |
| **feedback** | How to work — corrections and confirmed approaches | "Don't mock the database in integration tests" |
| **project** | What's happening — decisions, deadlines, context | "Auth rewrite driven by legal compliance" |
| **reference** | Where to look — external system pointers | "Pipeline bugs tracked in Linear project INGEST" |

### Three Ways to Create Memories

1. **Automatic extraction** — After every ~3 conversation turns, a background LLM pass silently analyzes recent messages and saves anything worth remembering. Most turns produce nothing; the model is instructed to be conservative.

2. **Model-initiated** — Ask the model to "remember that I prefer..." or it may proactively save feedback when you correct its approach. Uses the `memory_save` tool.

3. **Manual via GUI** — Click the memory icon (sidebar footer) to open the browser. Click "+ New" to create a memory with name, type, description, and content.

### Auto-Extraction Performance

Auto-extraction is **throttled to minimize GPU impact**:

- Runs every **3 turns** (not every turn)
- **60-second cooldown** between extractions
- Skipped if user messages are trivially short (<50 chars)
- Skipped if an extraction is already running
- Uses a small token budget (1024 max output)
- Runs fully in the background — never blocks the response

On Apple Silicon M4 Max with Gemma 4 26B, a typical extraction takes 3-8 seconds and runs while the user is reading the response. If you send another message during extraction, your query takes priority — Ollama queues requests, and the next response won't be delayed.

### Age Annotations

When memories are injected into the system prompt, stale memories get a warning:

- **<7 days** — no annotation
- **1-4 weeks** — `[2 weeks old — verify before acting on it]`
- **>30 days** — `[3 months old — may be outdated, verify against current state]`

This prevents the model from confidently acting on information that may have changed.

### Context Injection

Memories are injected into the system prompt in priority order: **feedback** (behavioral guidance) > **user** (who you are) > **project** (what's happening) > **reference** (where to look). Up to 30 memories, 30 lines each.

### GUI Memory Browser

The memory icon in the sidebar footer opens a right panel with:

- All memories grouped by type with color coding (blue/amber/green/purple)
- Click any memory to view its full content, metadata, and age
- Create new memories with the "+ New" button
- Delete memories with confirmation
- Modification dates and scope (global/project) displayed on each card

### CLI Compatibility

The web GUI reads and writes the same `~/.pre/memory/` directory as the CLI. Memories created in one are immediately available in the other. Both maintain the `MEMORY.md` index (and the legacy `index.md` for backwards compatibility).

## Architecture

```
server.js                  Express + WebSocket, REST API
src/
  ollama.js                Ollama /api/chat NDJSON streaming
  sessions.js              JSONL read/write + project management
  tools.js                 Tool dispatcher + execution loop
  tools-defs.js            55+ tool definitions for Ollama
  context.js               System prompt builder
  memory.js                 Enhanced memory system (save, extract, age, context injection)
  connections.js            Connection management, Google/Microsoft OAuth, Telegram setup
  constants.js             MODEL_CTX=65536, paths
  tools/
    bash.js                Shell execution (stderr capture)
    files.js               read_file, list_dir, glob, grep, file_write, file_edit
    web.js                 web_fetch, web_search (Brave API)
    memory.js              save, search, list, delete
    system.js              17 system tools (info, processes, clipboard, etc.)
    artifact.js            Interactive HTML artifacts
    image.js               ComfyUI image generation (SDXL/Juggernaut XL)
    document.js            Document export (txt, xml, docx, xlsx, pdf)
    google.js              Gmail, Google Drive, Google Docs
    telegram.js            Telegram Bot API
    github.js              GitHub REST API
    jira.js                Jira Server REST API v2
    confluence.js          Confluence Server REST API
    smartsheet.js          Smartsheet REST API 2.0
    slack.js               Slack Web API
    sharepoint.js          Microsoft SharePoint (Graph API)
    computer.js            Computer Use — desktop automation (cliclick)
    linear.js              Linear GraphQL API
    zoom.js                Zoom REST API (S2S OAuth)
    figma.js               Figma REST API
    asana.js               Asana REST API
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

Switch between themes using the dropdown in the sidebar footer.

- **Dark** — `#0a0a0a` background, blue primary, clean and minimal
- **Light** — `#fafafa` background, blue primary, bright and readable

## Key Design Decisions

1. **Always send `num_ctx=65536`** — must match the Modelfile exactly to prevent Ollama model reload (300s+ penalty)
2. **Vanilla JS, no framework** — local tool, one user, loads instantly, no build step
3. **Credentials stay server-side** — API keys never sent to the browser
4. **Tool execution server-side** — security, CORS, file system access
5. **Session format compatibility** — CLI and web can interleave on the same session files
