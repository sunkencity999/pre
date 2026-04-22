# PRE Web GUI

A local AI operating system for macOS. 63+ tools, desktop automation, document intelligence, voice interface, 15 enterprise integrations, event-driven triggers, persistent memory, and a full management GUI — all running on your Apple Silicon Mac with zero cloud dependency.

PRE is not a chatbot with a few tools bolted on. It is a **vertically integrated AI platform** that controls your desktop, reads your email, searches your documents semantically, reacts to file changes, speaks and listens, records and replays workflows, schedules autonomous tasks, and connects to every enterprise tool your team uses — through a single conversational interface backed by Google Gemma 4 26B running locally at ~73 tokens/second.

---

## What Sets PRE Apart

| Capability | What It Means |
|-----------|---------------|
| **Data sovereignty** | Every token, prompt, screenshot, and tool result stays on your machine. Nothing leaves. ITAR-safe, HIPAA-compatible, zero exposure. |
| **Desktop automation** | Computer Use + Workflow recording: PRE sees your screen and operates any application. Record actions, replay on demand. |
| **Document intelligence** | Local RAG indexes your files and searches by meaning, not keywords. Powered by `nomic-embed-text` embeddings. |
| **Voice interface** | Hold-to-record microphone in the GUI. Whisper transcribes locally; macOS `say` speaks responses. No audio leaves your machine. |
| **Event-driven triggers** | File watchers and webhooks that fire prompts automatically. PRE reacts to your environment without being asked. |
| **Enterprise integrations** | 15 services (Jira, Confluence, SharePoint, Slack, Linear, Zoom, Figma, Asana, Gmail, Drive, GitHub, Smartsheet, Telegram, Brave, Wolfram) in one interface. |
| **Autonomous scheduling** | Cron jobs run server-side — daily briefings, monitoring, reports — even when you're not at the computer. |
| **Persistent memory** | Auto-extracted from conversations, age-tracked, semantically searchable. PRE gets smarter over weeks and months. |
| **Zero marginal cost** | No API pricing, no rate limits, no seat fees. Every prompt, every scheduled job, every sub-agent loop — free. |
| **MCP server** | Frontier models (Claude, Codex, Gemini) can delegate execution-heavy tasks to PRE, saving thousands in API tokens. |
| **Full GUI management** | Sidebar panels for triggers, RAG indexes, workflows, memory, cron jobs, and settings — everything is visual, not just CLI. |

---

## Quick Start

The fastest path is the one-touch installer — double-click `install.sh` (or run it from a terminal) and it handles everything: Ollama, model pull, CLI build, npm install, ComfyUI, context sizing, and optional auto-start at login.

```bash
./install.sh           # Interactive — prompts for optional steps
./install.sh --yes     # Non-interactive — accepts all defaults
```

Or start the web GUI manually:

```bash
cd web && npm install   # First time only
node server.js          # http://localhost:7749
```

The `pre-launch` command also starts the web GUI automatically in the background when you launch the CLI.

## Features

- **Computer Use** — desktop automation via screenshot + mouse/keyboard control; the model sees your screen and operates any application
- **Real-time streaming** via WebSocket — tokens appear as the model generates them
- **Full tool execution** — 63+ tools run server-side with multi-turn agentic loop (up to 25 tool calls per prompt)
- **Local image generation** — ComfyUI + Stable Diffusion XL on Apple Silicon GPU, inline image preview in chat
- **Auto-start at login** — optional macOS LaunchAgent keeps the web GUI running in the background; toggle from Settings
- **Local RAG** — index directories and search by meaning using `nomic-embed-text` embeddings; incremental re-indexing, paragraph-aware chunking
- **Event-driven triggers** — file watchers and webhooks that fire prompts automatically when things change
- **Voice interface** — speech-to-text via local Whisper, text-to-speech via macOS `say`; all audio stays on your machine
- **Workflow capture and replay** — record Computer Use action sequences and replay them at configurable speed
- **Auto-sized context window** — the installer detects your Mac's RAM and sets the optimal context window (8K–128K); no manual tuning needed
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

## Native macOS Integrations (Zero Config)

These tools work immediately with no setup — they use the native macOS apps and whatever accounts you've already configured on your Mac.

| Tool | App | Actions |
|------|-----|---------|
| **Mail** | Mail.app | Send, draft, search, read, list recent, list mailboxes, list accounts |
| **Calendar** | Calendar.app | Today's events, this week, list events by range, create events, search, list calendars, delete events |
| **Contacts** | Contacts.app | Search by name/org, read full contact details, list groups, count |
| **Reminders** | Reminders.app | Add, list, complete, search, list lists, delete |
| **Notes** | Notes.app | Search (title + content), read, create, list recent, list folders |
| **Spotlight** | mdfind | Full-text search across entire machine, find files by type, file metadata preview |

These work with **any email/calendar provider** — iCloud, Gmail, Exchange, Outlook, Yahoo — whatever is configured in System Settings. No OAuth flows, no API keys, no developer console. Just ask PRE to check your calendar or send an email.

> "What's on my calendar today?"

> "Remind me to review the Q3 report on Friday"

> "Search my notes for anything about the API migration"

> "Find all PDFs in my Documents folder about Azure"

> "Search my contacts for someone at Acme Corp"

The Calendar and Reminders tools use compiled EventKit binaries for fast queries (~50ms), even across many calendars/lists. Binaries are compiled on first use and cached at `/tmp/pre-cal-events` and `/tmp/pre-reminders`. Spotlight uses `mdfind` directly — the same engine behind Cmd+Space.

**Note:** macOS will prompt for Automation permissions on first use (allow Terminal to control Mail/Calendar/Contacts/Notes/Reminders). This is a one-time approval. Sending email and deleting reminders require user confirmation in the chat.

For advanced email features (threading, labels, batch operations, Drive/Docs), the Google OAuth integration is still available as an optional upgrade in Settings.

---

## Cloud Integrations

All cloud integrations are configured through the Settings panel (gear icon in sidebar footer). Credentials are stored in `~/.pre/connections.json` and never exposed to the browser.

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

## Local RAG (Document Intelligence)

PRE can index directories of files and search them semantically — ask questions in natural language and find relevant content even when the exact words don't match. All indexing and search runs locally using `nomic-embed-text` embeddings through Ollama.

### How It Works

1. Index a directory: PRE reads all text files, splits them into chunks (paragraph-boundary-aware), and generates 768-dimensional embeddings via Ollama
2. Search by meaning: queries are embedded and matched against stored chunks using cosine similarity
3. Incremental updates: re-indexing skips unchanged files (mtime tracking), so only new/modified files are processed

### Available Actions

| Action | Description |
|--------|-------------|
| `index` | Index a directory (recursive by default). Supports incremental re-indexing. |
| `search` | Semantic search across an index — returns top-K matching chunks with scores |
| `list` | List all indexes with file counts, chunk counts, and sizes |
| `status` | Detailed status of a specific index (files, chunks, last indexed) |
| `delete` | Delete an index and all its data (requires confirmation) |

### Configuration

- **Chunk size:** ~1,500 characters with 200-character overlap at paragraph boundaries
- **File types:** `.txt`, `.md`, `.js`, `.ts`, `.py`, `.java`, `.c`, `.h`, `.css`, `.html`, `.json`, `.yaml`, `.yml`, `.toml`, `.xml`, `.csv`, `.sh`, `.rb`, `.go`, `.rs`, `.swift`, `.m`, `.sql`, `.r`, `.scala`, `.kt`, `.lua`, `.pl`, `.ex`, `.clj`, `.hs`, `.ml`
- **Max file size:** 512 KB per file
- **Batch embedding:** 40 chunks per Ollama API call
- **Storage:** `~/.pre/rag/{index_name}/` (meta.json, chunks.json, vectors.json)

### Example Uses

> "Index my project at ~/code/api-service"

> "Search the api-service index for authentication middleware"

> "What files mention database migration in the api-service index?"

> "List all my RAG indexes"

### Performance

On an M4 Max with 128GB: indexing 33 files (266 chunks) completes in ~4 seconds. Semantic search returns results in under 10ms. Incremental re-indexing of an unchanged directory takes ~0ms.

### GUI Panel

Click the **search+ icon** in the sidebar footer to open the RAG management panel. Index directories, run semantic searches, and delete indexes visually. See the [GUI Management Panels](#gui-management-panels) section for details.

---

## Event-Driven Triggers

PRE can react to events automatically — watch files for changes or listen for webhooks, then execute prompts when triggered. Triggers use the same execution pipeline as cron jobs, so they create sessions, run tool loops, and deliver notifications.

### Trigger Types

| Type | How It Fires |
|------|-------------|
| **File watcher** | Monitors a directory (recursive) for file changes. Debounced (default 3s) to batch rapid changes. Optional glob filter. |
| **Webhook** | Listens at `/api/triggers/webhook/:id`. Fires when an HTTP POST is received. Optional secret verification via `X-Webhook-Secret` header. |

### Available Actions

| Action | Description |
|--------|-------------|
| `add` | Create a new trigger (file_watch or webhook) |
| `list` | List all triggers with status and hit counts |
| `remove` | Delete a trigger (requires confirmation) |
| `enable` | Re-enable a disabled trigger |
| `disable` | Temporarily disable a trigger without deleting it |

### Variable Substitution

Trigger prompts support variables that are replaced with event data at fire time:

**File watcher:** `{file}` (changed filename), `{event}` (change/rename), `{path}` (watched directory)

**Webhook:** `{payload}` (POST body as JSON), `{headers}` (request headers)

### Example Uses

> "Watch ~/Documents/reports for new files and summarize any new PDFs"

> "Create a webhook trigger that processes incoming JSON payloads and posts a summary to Slack"

> "List all my active triggers"

> "Disable the reports-watcher trigger"

### Storage

Triggers are stored in `~/.pre/triggers.json`. File watchers start automatically when the server boots (via `triggers.init()`). Webhooks are stateless listeners on the REST API.

### GUI Panel

Click the **lightning icon** in the sidebar footer to open the Triggers management panel. Create, enable/disable, and delete triggers visually. See the [GUI Management Panels](#gui-management-panels) section for details.

---

## Voice Interface

PRE supports speech-to-text and text-to-speech, all running locally. Transcribe audio with OpenAI Whisper, speak responses with macOS `say`.

### Requirements

| Component | Install | Purpose |
|-----------|---------|---------|
| **Whisper** | `pip install openai-whisper` | Speech-to-text (local, no API) |
| **macOS `say`** | Built-in | Text-to-speech (25+ English voices) |
| **FFmpeg** (optional) | `brew install ffmpeg` | Audio format conversion (WebM→WAV, AIFF→MP3) |

### Available Actions

| Action | Description |
|--------|-------------|
| `transcribe` | Transcribe audio from a file path or base64-encoded buffer |
| `speak` / `say` / `tts` | Speak text aloud or save to audio file (AIFF or MP3) |
| `voices` | List available macOS English voices |
| `status` | Check which voice capabilities are installed |

### GUI: Microphone Button

When Whisper is installed, a **microphone icon** appears in the chat input area. Hold it to record, release to transcribe:

1. Press and hold the microphone button — it pulses red while recording
2. Speak your message
3. Release — the audio is transcribed locally via Whisper and appears in the input box
4. Edit if needed, then send

Audio is recorded via the Web Audio API, encoded as base64 WebM, and sent to `POST /api/voice/transcribe` for local Whisper transcription. No audio leaves your machine.

### Example Uses

> "Speak this summary aloud: The quarterly report shows a 15% increase in throughput"

> "What voices are available for text-to-speech?"

> "Check my voice capabilities — is Whisper installed?"

### Configuration

- **Whisper model:** `base.en` (good balance of speed and accuracy for English)
- **Default voice:** Samantha (clear, natural macOS voice)
- **Speech rate:** 185 words per minute
- **Max recording:** 120 seconds (enforced browser-side)

---

## Workflow Capture and Replay

Record sequences of Computer Use actions as replayable workflows. Automate repetitive desktop tasks by recording them once and replaying on demand.

### How It Works

1. Start recording — PRE begins capturing every Computer Use action (click, type, key, scroll, drag)
2. Perform the task — use Computer Use normally; each action is recorded with inter-step timing
3. Stop recording — the workflow is saved as a JSON file in `~/.pre/workflows/`
4. Replay — re-execute the recorded steps with configurable speed

Observation-only actions (screenshot, screen_size, cursor_position) are automatically filtered out — only actions that change state are recorded.

### Available Actions

| Action | Description |
|--------|-------------|
| `record` / `start` | Start recording a new workflow |
| `stop` | Stop recording and save the workflow |
| `status` | Check if currently recording and how many steps captured |
| `list` | List all saved workflows with step counts and durations |
| `replay` / `run` | Replay a workflow with optional speed multiplier |
| `show` / `inspect` | View the steps of a saved workflow |
| `delete` | Delete a saved workflow (requires confirmation) |
| `export` | Export a workflow as a shell-friendly description |

### Example Uses

> "Start recording a workflow called 'daily-standup-setup'"

(Perform Computer Use actions: open Zoom, join meeting, open notes app, etc.)

> "Stop recording"

> "Replay the daily-standup-setup workflow at 2x speed"

> "Show me the steps in the daily-standup-setup workflow"

> "List all my saved workflows"

### Storage

Workflows are stored as JSON in `~/.pre/workflows/`. Each file contains metadata (name, description, created date, duration) and an array of steps with action, arguments, inter-step delay, and a deduplication hash.

### GUI Panel

Click the **grid icon** in the sidebar footer to open the Workflows panel. From here you can:
- View all saved workflows with step counts and durations
- Click **View** to inspect individual steps
- Click **Replay** to re-execute a workflow (with confirmation prompt)
- Delete workflows you no longer need

---

## GUI Management Panels

Every major PRE capability has a dedicated management panel accessible from the sidebar footer. No CLI needed — everything can be configured and monitored visually.

### Sidebar Footer Icons

The sidebar footer contains these buttons (left to right):

| Icon | Panel | What You Can Do |
|------|-------|-----------------|
| Sun/Moon | Theme | Toggle between Dark and Light themes |
| Book | **Memory Browser** | View, create, search, and delete persistent memories grouped by type |
| Clock | **Cron Scheduler** | Create, enable/disable, run, and delete scheduled jobs with natural language input |
| Lightning | **Triggers** | Create file watchers and webhooks, enable/disable, view fire counts, delete |
| Search+ | **RAG Indexes** | Index directories, semantic search across documents, manage and delete indexes |
| Grid | **Workflows** | View saved workflows, inspect steps, replay, delete |
| Gear | **Settings** | Configure all 15 enterprise integrations, MCP servers, auto-start at login |

### Voice Input (Microphone Button)

The **microphone icon** appears in the chat input area (next to the model selector) when Whisper is installed. It provides push-to-talk voice input:

1. **Hold** the microphone button (or press and hold on mobile)
2. **Speak** your message — the button pulses red while recording
3. **Release** — audio is sent to local Whisper for transcription
4. Transcribed text appears in the input box, ready to send or edit

The microphone button is hidden if Whisper is not installed. Install it with `pip install openai-whisper`.

### Triggers Panel

Click the **lightning icon** to manage event-driven triggers:

- **+ New Trigger** — opens a form with:
  - **Type selector**: File Watcher or Webhook
  - **File Watcher fields**: watch path, glob filter (e.g., `*.pdf`)
  - **Webhook fields**: optional shared secret for verification
  - **Prompt**: the text sent to the model when triggered (supports `{file}`, `{event}`, `{path}`, `{payload}`, `{headers}` variables)
- Each trigger shows: name, type badge, status dot (green=enabled), watch path or webhook endpoint, fire count, last fired time
- **Disable/Enable** toggles without deleting
- **Delete** removes the trigger permanently

### RAG Panel

Click the **search+ icon** to manage document indexes:

- **+ Index Directory** — enter a directory path and optional index name; PRE reads all text files, chunks them, and generates embeddings
- **Semantic Search** — enter a natural language query and optional index name; results show matched content chunks ranked by relevance score
- **Index list** shows all indexes with file counts, chunk counts, and creation dates
- **Delete** removes an index and all its embeddings

Indexing progress is shown inline. Large directories may take 10-30 seconds; incremental re-indexing of unchanged files is instant.

### Memory Browser

Click the **book icon** to browse and manage persistent memories:

- Memories are grouped by type with color coding: **user** (blue), **feedback** (amber), **project** (green), **reference** (purple)
- Click any memory to view its full content, metadata, and age
- **+ New** to create a memory manually with name, type, description, and content
- **Delete** with confirmation
- Modification dates and scope (global/project) displayed on each card

### Cron Scheduler

Click the **clock icon** to manage scheduled jobs:

- **+ New Job** — enter schedule in natural language ("every weekday at 9am"), description, and prompt
- Live preview shows the parsed cron expression and human-readable description
- Each job shows: status dot, description, schedule, run count, last run time
- **Run Now** executes immediately; **Result** opens the output session
- **Enable/Disable** toggles; **Delete** removes the job
- Results are delivered via macOS notification, Telegram, and in-browser toast

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

## Auto-Start at Login

PRE can optionally start the web GUI server automatically when you log in to your Mac, so it's always available at `http://localhost:7749` without manually running anything.

### How It Works

A macOS LaunchAgent (`com.pre.server`) runs `pre-server.sh` at login. The launcher ensures Ollama is running, pre-warms the model into GPU memory, and starts the Node.js server. If the server crashes, launchd restarts it automatically.

### Enabling Auto-Start

**From the GUI:** Open Settings (gear icon) → scroll to the **System** section → toggle **Start at Login**.

**From the installer:** The install script (`install.sh`) offers to enable auto-start during installation (Step 10).

**Manually:**

```bash
# Enable
web/pre-server.sh           # Verify it starts correctly first
launchctl load ~/Library/LaunchAgents/com.pre.server.plist

# Disable
launchctl unload ~/Library/LaunchAgents/com.pre.server.plist
rm ~/Library/LaunchAgents/com.pre.server.plist
```

### Management Commands

`pre-server.sh` doubles as a management tool:

```bash
web/pre-server.sh --status   # Check if server and model are running
web/pre-server.sh --stop     # Stop the server (unloads LaunchAgent if active)
```

### Details

- **Plist:** `~/Library/LaunchAgents/com.pre.server.plist`
- **Log:** `~/.pre/server.log`
- **Restart policy:** `KeepAlive` with `SuccessfulExit: false` — launchd restarts on crash but not on clean shutdown
- **Coexistence:** `pre-launch` detects the LaunchAgent-managed server (via PPID=1) and reuses it instead of killing and restarting

---

## Context Window Auto-Sizing

The install script detects your Mac's unified memory and sets the optimal context window size. The value is written to `~/.pre/context` and read at runtime by the CLI, web GUI, and launcher — no manual sync needed.

| RAM | Context Window | Tokens |
|-----|---------------|--------|
| 128GB+ | 128K | 131,072 |
| 64–95GB | 64K | 65,536 |
| 48–63GB | 32K | 32,768 |
| 36–47GB | 16K | 16,384 |
| <36GB | 8K | 8,192 |

The context window can be overridden by editing `~/.pre/context` directly (any value from 2048 to 262144). Changes take effect the next time the CLI or web GUI starts.

### Where It's Read

| Component | How |
|-----------|-----|
| `web/src/constants.js` | `fs.readFileSync('~/.pre/context')` at require-time |
| `engine/pre.m` | `NSString` file read in `main()` |
| `engine/pre-launch` | Bash `read` for model warmup `num_ctx` |

All three fall back to 131,072 if the file is missing or invalid.

---

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

## MCP Server — Use PRE from Claude, Codex, Antigravity, or Any MCP Client

PRE doubles as an MCP (Model Context Protocol) server. This lets frontier models like Claude, GPT, or Gemini delegate execution-heavy tasks to PRE's local Gemma 4 agent instead of burning API tokens on tool loops. The local model handles the grunt work — file searches, multi-step tool chains, email lookups, data gathering — while the frontier model stays focused on complex reasoning and planning.

**Why this matters:** A single Claude conversation that searches files, reads 15 documents, calls 8 tools, and synthesizes results can consume 100K+ tokens at API pricing. Delegating the search/tool phase to PRE costs zero tokens — Gemma 4 runs locally on your Apple Silicon GPU.

### Installation

#### Claude Code (CLI / IDE extensions)

Add PRE to your project-level `.mcp.json` or global settings:

```json
{
  "mcpServers": {
    "pre": {
      "command": "node",
      "args": ["/path/to/pre/web/mcp-stdio.js"]
    }
  }
}
```

Replace `/path/to/pre` with your actual PRE install path (e.g., `~/pre`).

The stdio transport auto-starts Ollama and the PRE server if they're not already running — no need to launch PRE separately.

#### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pre": {
      "command": "node",
      "args": ["/path/to/pre/web/mcp-stdio.js"]
    }
  }
}
```

Restart Claude Desktop after saving. PRE will appear as an available tool.

#### Other MCP Clients (HTTP transport)

If the PRE server is already running, any MCP client that supports StreamableHTTP can connect directly:

```
POST http://localhost:7749/mcp
```

No additional setup needed — the HTTP transport is always available when the server is running.

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `pre_agent` | Run a full agentic task through PRE's local model with 63+ tools. Multi-turn tool loop, zero API cost. |
| `pre_chat` | One-shot query to the local model. No tools, no agent loop — just fast, free LLM reasoning. |
| `pre_memory_search` | Search PRE's persistent memory for context about the user, projects, and preferences. |
| `pre_sessions` | List recent PRE sessions to understand what tasks have been worked on. |

### Teaching Claude When to Delegate

The MCP tool descriptions tell Claude *what* PRE can do. To tell Claude *when* to prefer PRE over doing work itself, add delegation guidance to your `CLAUDE.md`:

```markdown
## Local Agent Delegation (PRE)

You have access to PRE, a local AI agent running Gemma 4 26B with 63+ tools
on Apple Silicon. PRE runs at zero token cost. Delegate to PRE (`pre_agent`)
when the task is execution-heavy but not reasoning-heavy:

**Delegate to PRE:**
- File searches, code grep, reading many files to gather context
- Sending emails, creating calendar events, posting to Slack
- Multi-step tool chains (search Jira → cross-reference Confluence → summarize)
- Data gathering and formatting that requires many tool calls
- Running shell commands, system checks, process management
- Searching PRE's memory for user context (`pre_memory_search`)

**Keep for yourself (Claude):**
- Complex reasoning, planning, and architecture decisions
- Code generation requiring deep understanding of large codebases
- Nuanced writing where tone and precision matter
- Tasks requiring your full context window and conversation history
- Anything the user is actively discussing with you interactively

**Use `pre_chat` for:**
- Quick questions you'd otherwise answer yourself but that burn tokens
- Summarization of long text you've already gathered
- Translation, reformatting, or analysis of data

When delegating, be specific in the task description — PRE is an agent, not
a search engine. "Search my email for messages from Jane about the Q3 budget
and summarize what she said" will work. "Email stuff" will not.
```

This snippet goes in any `CLAUDE.md` file — project-level or global. Claude reads it at the start of every conversation and will proactively delegate appropriate tasks.

#### Codex (OpenAI)

Add PRE to your Codex config at `~/.codex/config.toml`:

```toml
[mcp_servers.pre]
command = "node"
args = ["/path/to/pre/web/mcp-stdio.js"]
```

Add delegation guidance to `~/.codex/instructions.md` (global) or `AGENTS.md` (project-level). Codex reads `AGENTS.md` files from the directory hierarchy, the same way Claude reads `CLAUDE.md`.

#### Antigravity (Google)

Add PRE to your Antigravity settings at `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "pre": {
      "command": "node",
      "args": ["/path/to/pre/web/mcp-stdio.js"]
    }
  }
}
```

Add delegation guidance to `~/.gemini/GEMINI.md` (global). Antigravity also reads `AGENTS.md` for cross-tool instructions.

#### Automatic Setup

The install script (`install.sh`) auto-detects Claude Desktop, Claude Code, Codex, and Antigravity. It offers to configure MCP and writes delegation instructions to the appropriate file for each tool:

| Tool | MCP Config | Instructions File |
|------|-----------|-------------------|
| Claude Desktop | `claude_desktop_config.json` | `~/CLAUDE.md` |
| Claude Code | `.mcp.json` (manual) | `~/CLAUDE.md` |
| Codex | `~/.codex/config.toml` | `~/.codex/instructions.md` |
| Antigravity | `~/.gemini/settings.json` | `~/.gemini/GEMINI.md` |

### Architecture

```
Claude/GPT (frontier model)
    │
    ├─ Complex reasoning, planning, code generation
    │
    └─ pre_agent("search my email for the DocuSign issue")
         │
         PRE Server (localhost:7749)
              │
              └─ Gemma 4 26B + 63+ tools
                   ├─ apple_mail.search → 8 results
                   ├─ apple_mail.read → full thread
                   └─ Returns summary to Claude
```

The frontier model sees only the final result — not the intermediate tool calls, screenshots, or token overhead from the agent loop. This is where the cost savings compound: a 15-turn tool loop that would cost ~$2 in Claude API tokens runs for free on local hardware.

### Cost Savings: PRE as a Token-Offloading Engine

We benchmarked 20 tasks across PRE's MCP tools to measure quality and estimate real-world token savings. The results confirm that a significant share of typical AI workloads can be offloaded to local inference at zero marginal cost without sacrificing quality.

#### Benchmark Results (20 tasks, Gemma 4 26B q8_0, M4 Max 128GB)

| Task Type | Tasks | Passed | Avg Quality | Avg Time |
|-----------|-------|--------|-------------|----------|
| **Agent tasks** (tool loops) | 10 | 9/10 | 4.0/5 | 40s |
| **Chat tasks** (delegatable) | 5 | 5/5 | 4.2/5 | 9s |
| **Chat tasks** (frontier-grade) | 5 | 5/5 | 5.0/5 | 37s |

Agent tasks included: system info gathering, file search, process listing, git log, file reading, directory listing, code grep, disk/network checks, memory search, and multi-step file analysis. All used real tools (bash, grep, read_file, system_info, etc.) and returned correct results.

Chat tasks included: summarization, translation, code explanation, data formatting, factual Q&A, architecture design, security review, bug diagnosis, and code generation. Even the "frontier-grade" tasks scored 5/5 — but these were self-contained questions. Where Claude still wins is **in-context reasoning** across your full conversation history and active codebase.

#### Per-Task Token Savings

| Task Type | Tokens (input + output) | Claude Opus Cost | Claude Sonnet Cost | PRE Cost |
|-----------|------------------------|------------------|-------------------|----------|
| Agent task (tool loop) | ~25K + ~2K | $0.53 | $0.11 | **$0.00** |
| Chat task (no tools) | ~2K + ~800 | $0.09 | $0.02 | **$0.00** |

#### Monthly Projection: 10M Tokens/Month User

A user consuming 10M tokens/month (roughly 80% input, 20% output) has a baseline monthly cost of **$270 (Opus)** or **$54 (Sonnet)**. Not all tokens are delegatable — complex reasoning, code generation in active codebases, and interactive debugging should stay with the frontier model. Based on our benchmark data, 20-40% of typical usage is offloadable.

| Delegation Level | Tokens to PRE | Opus Savings | Sonnet Savings |
|-----------------|---------------|--------------|----------------|
| **Conservative (20%)** | 2M | $54/mo ($648/yr) | $11/mo ($130/yr) |
| **Moderate (30%)** | 3M | $81/mo ($972/yr) | $16/mo ($194/yr) |
| **Aggressive (40%)** | 4M | $108/mo ($1,296/yr) | $22/mo ($259/yr) |

For a daily workflow of 5 agent delegations + 15 chat delegations (a moderate pattern), savings are **$3.97/day on Opus** or **$87/month** over 22 work days.

#### What to Delegate vs. Keep

Based on benchmark quality scores:

**Safe to delegate (4-5/5 quality):**
- System checks, disk/network info, process listing
- File reading, searching, grepping across codebases
- Git history, status checks
- Email, calendar, contacts, reminders (native macOS tools)
- Memory lookups and cross-session context
- Summarization, translation, data formatting
- Factual Q&A, code explanations
- Multi-step tool chains (search Jira + read Confluence + summarize)

**Keep for the frontier model:**
- Code generation that touches the active codebase (needs full project context)
- Architecture decisions informed by the conversation history
- Interactive debugging where back-and-forth matters
- Nuanced writing where tone and precision are critical
- Security analysis of code you're actively editing

**The cost compounds:** A 15-turn tool loop where Claude calls bash, reads 8 files, greps 3 directories, and summarizes — that's easily 50K+ tokens. Delegated to PRE, Claude sees only the final 500-token summary. The 50K tokens of tool-loop overhead simply never hit the API.

#### Hardware Amortization

PRE requires Apple Silicon with sufficient unified memory. At moderate delegation (30%, Opus pricing), the **$972/year savings** pays for itself against the incremental hardware cost. For teams of 3-5 engineers sharing a Mac Studio, the economics are even stronger — one machine serves multiple MCP clients simultaneously via Ollama's request queuing.

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PRE_WEB_PORT` | `7749` | Port for the PRE server (MCP endpoints share this) |
| `PRE_MCP_CHILD` | — | Set automatically when stdio transport spawns the server |

---

## Tutorial — Getting the Most Out of PRE

This tutorial walks you through every major feature of PRE using real example prompts. Each section builds on the last, but you can jump to any section that interests you. Copy any prompt below and paste it directly into PRE.

```
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║     ██████╗ ██████╗ ███████╗    ████████╗██╗   ██╗████████╗ ║
  ║     ██╔══██╗██╔══██╗██╔════╝    ╚══██╔══╝██║   ██║╚══██╔══╝ ║
  ║     ██████╔╝██████╔╝█████╗         ██║   ██║   ██║   ██║    ║
  ║     ██╔═══╝ ██╔══██╗██╔══╝         ██║   ██║   ██║   ██║    ║
  ║     ██║     ██║  ██║███████╗        ██║   ╚██████╔╝   ██║    ║
  ║     ╚═╝     ╚═╝  ╚═╝╚══════╝        ╚═╝    ╚═════╝    ╚═╝    ║
  ║                                                              ║
  ║          Personal Reasoning Engine — Tutorial Guide          ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
```

### Table of Contents

1. [Your First Conversation](#1-your-first-conversation)
2. [Working with Files](#2-working-with-files)
3. [Your Mac, Your Way — Native App Control](#3-your-mac-your-way--native-app-control)
4. [Desktop Automation — Computer Use](#4-desktop-automation--computer-use)
5. [Browser Automation](#5-browser-automation)
6. [Memory — PRE Learns Over Time](#6-memory--pre-learns-over-time)
7. [RAG — Search Your Documents by Meaning](#7-rag--search-your-documents-by-meaning)
8. [Scheduling & Automation](#8-scheduling--automation)
9. [Sub-Agents — Parallel Research](#9-sub-agents--parallel-research)
10. [Cloud Integrations](#10-cloud-integrations)
11. [Artifacts — Interactive Documents](#11-artifacts--interactive-documents)
12. [Voice Interface](#12-voice-interface)
13. [Deep Research Mode](#13-deep-research-mode)
14. [Power User Workflows](#14-power-user-workflows)

---

### 1. Your First Conversation

PRE is a conversational AI that runs entirely on your Mac. Just type naturally.

```
┌─────────────────────────────────────────────────────┐
│  YOU:  What can you help me with?                   │
│                                                     │
│  PRE:  I have 63+ tools at my disposal...           │
│        [lists capabilities]                         │
│                                                     │
│  YOU:  Summarize what's in my Downloads folder       │
│                                                     │
│  PRE:  ┌─ Tool: list_dir ──────────┐               │
│        │ ~/Downloads               │               │
│        └───────────────────────────┘               │
│        Your Downloads folder has 47 items...        │
└─────────────────────────────────────────────────────┘
```

**Try these prompts:**

> What's the weather like in my area? Search the web for the current forecast.

> What day of the week was July 4, 1776?

> Explain quantum computing like I'm 10 years old.

> What's running on my Mac right now? Show me the top 10 processes by CPU usage.

**Tip:** PRE auto-generates a session title from your first message. Use the **+ Session** button to start fresh conversations, or **+ Project** to organize them into folders.

---

### 2. Working with Files

PRE can read, write, search, and edit files across your entire system.

```
  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │   read_file  │      │   file_write │      │   file_edit  │
  │              │      │              │      │              │
  │  Read any    │ ──── │  Create new  │ ──── │  Edit with   │
  │  file on     │      │  files from  │      │  surgical    │
  │  your Mac    │      │  scratch     │      │  precision   │
  └──────────────┘      └──────────────┘      └──────────────┘
         │                                           │
         ▼                                           ▼
  ┌──────────────┐                           ┌──────────────┐
  │   glob       │                           │   grep       │
  │              │                           │              │
  │  Find files  │                           │  Search file │
  │  by pattern  │                           │  contents    │
  └──────────────┘                           └──────────────┘
```

**Try these prompts:**

> Find all Python files in my home directory that import pandas.

> Read my ~/.zshrc and suggest improvements for performance.

> Create a shell script at ~/Desktop/cleanup.sh that finds and lists files larger than 100MB in my home directory.

> Search my Documents folder for any file mentioning "quarterly review".

> Find all images on my Desktop that were modified in the last week.

**Advanced file operations:**

> Read my package.json, find all outdated patterns, and update the dependencies section to use ES modules.

> Find every TODO comment in the src/ directory and create a summary report.

> Compare the two most recent log files in /var/log/ and highlight the differences.

---

### 3. Your Mac, Your Way — Native App Control

PRE controls Mail, Calendar, Contacts, Reminders, Notes, and Spotlight directly through macOS — no API keys, no OAuth, no setup. Whatever accounts you have configured on your Mac just work.

```
  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐
  │  Mail   │   │ Calendar │   │ Contacts │   │ Reminders │
  │  ✉️      │   │  📅       │   │  👤       │   │  ☑️        │
  └────┬────┘   └────┬─────┘   └────┬─────┘   └─────┬─────┘
       │             │              │               │
       └─────────────┴──────────────┴───────────────┘
                          │
                    ┌─────┴─────┐
                    │   Notes   │   ┌───────────┐
                    │   📝       │   │ Spotlight │
                    └───────────┘   │  🔍        │
                                    └───────────┘
```

**Email:**

> Check my email for anything from my boss in the last 3 days.

> Draft an email to sarah@example.com about rescheduling our Wednesday meeting to Thursday at 2pm. Keep it professional but friendly.

> Search my email for the latest shipping confirmation and tell me the tracking number.

**Calendar:**

> What's on my calendar today? Include meeting links if available.

> Create a "Dentist Appointment" event on May 5th from 2pm to 3pm at "123 Main St, Suite 200".

> What does my week look like? Flag any conflicts or back-to-back meetings.

**Contacts & Reminders:**

> Find the phone number for John at Acme Corp in my contacts.

> Remind me to submit the expense report by Friday at 5pm with high priority.

> Show me all my incomplete reminders from the "Work" list.

**Notes & Spotlight:**

> Search my notes for anything about the API migration project.

> Create a new note in my "Work" folder titled "Meeting Notes - April 21" with the key discussion points from today.

> Find all PDF documents on my Mac that contain "budget proposal".

---

### 4. Desktop Automation — Computer Use

This is PRE's most powerful feature. It takes a screenshot, analyzes what's on screen, and controls your mouse and keyboard to operate **any** application — even ones without APIs.

```
  ┌─────────────────────────────────────────────────┐
  │              THE VISION LOOP                     │
  │                                                  │
  │   ┌──────────┐    ┌───────────┐    ┌─────────┐ │
  │   │Screenshot│───>│  Analyze  │───>│  Act    │ │
  │   │  📸      │    │  with AI  │    │ 🖱️ ⌨️    │ │
  │   └──────────┘    └───────────┘    └────┬────┘ │
  │        ▲                                │      │
  │        └────────────────────────────────┘      │
  │               Repeat until done                │
  └─────────────────────────────────────────────────┘
```

**Try these prompts:**

> Take a screenshot and describe what's on my screen.

> Open System Settings and navigate to the Wi-Fi section. Tell me what networks are available.

> Open TextEdit, create a new document, and type "Meeting notes for today" as the title.

> Open Finder, navigate to my Downloads folder, and sort the files by date modified.

**Multi-step automation:**

> Open Safari, go to news.ycombinator.com, and tell me the top 5 stories right now.

> Open the Calculator app, compute 15% tip on $86.50, and tell me the result.

> Take a screenshot of my desktop, then open Preview and annotate the screenshot with a red circle around any Finder windows.

**Keyboard shortcuts and combos:**

> Press Cmd+Space to open Spotlight, type "Activity Monitor", and press Enter to launch it.

> In the current window, press Cmd+A to select all, then Cmd+C to copy the text, and paste it into a new note.

**Workflow recording (record once, replay anytime):**

> Start recording a workflow called "morning-setup". Open Mail, Safari, and Slack, then arrange them side by side. Stop recording when done.

> Replay the "morning-setup" workflow.

> Show me all my saved workflows.

---

### 5. Browser Automation

PRE has a built-in headless Chrome browser for web scraping, form filling, and automated browsing.

**Try these prompts:**

> Open the browser, go to https://example.com, and read the page content.

> Navigate to Wikipedia's main page and tell me what today's featured article is about.

> Go to https://httpbin.org/forms/post, fill in the form with test data, and submit it.

> Search Google for "best hiking trails near Santa Cruz" and give me the top 5 results with links.

**Data extraction:**

> Navigate to my company's status page and check if all services are operational.

> Go to a recipe site, find a recipe for chocolate chip cookies, and save the ingredients list to a file on my Desktop.

---

### 6. Memory — PRE Learns Over Time

PRE remembers things between conversations. Memories are auto-extracted from your chats or you can save them explicitly. They're searchable by meaning, not just keywords.

```
  ┌────────────────────────────────────────────────┐
  │              MEMORY LIFECYCLE                    │
  │                                                  │
  │  Conversation ──> Auto-Extract ──> Memory Store │
  │       │                               │         │
  │       │          Manual Save          │         │
  │       └──────────────────────────────>│         │
  │                                       │         │
  │  Future Conversations <── Recall <────┘         │
  │                                                  │
  │  Types: user | feedback | project | reference   │
  └────────────────────────────────────────────────┘
```

**Try these prompts:**

> Remember that our team standup is at 9:15 AM Pacific every weekday, and the Zoom link is in the #engineering channel on Slack.

> Remember that I prefer TypeScript over JavaScript and always use strict mode.

> Search my memories for anything about deployment procedures.

> List all my saved memories.

> What do you remember about my work projects?

**The Memory Panel** (book icon in the sidebar footer) lets you browse, search, and manage all memories visually.

**Pro tip:** PRE auto-extracts memories every ~3 conversation turns. The more you chat, the smarter it gets. Memories are age-tracked — PRE knows when it learned something and flags potentially stale information.

---

### 7. RAG — Search Your Documents by Meaning

Local RAG (Retrieval-Augmented Generation) indexes your files and lets you search by concept, not just keywords. Everything stays on your machine.

```
  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
  │  Your Files  │       │  nomic-embed │       │  Semantic    │
  │              │       │    -text     │       │  Search      │
  │  ~/project/  │──────>│              │──────>│              │
  │  *.md, *.py  │ chunk │  768-dim     │ index │  "How does   │
  │  *.js, *.txt │       │  embeddings  │       │   auth work?"│
  └──────────────┘       └──────────────┘       └──────────────┘
```

**Try these prompts:**

> Index my ~/Documents/notes folder and call the index "my-notes".

> Search my "my-notes" index for anything about project deadlines or milestones.

> Index this Git repository and search for how error handling works.

> List all my RAG indexes and their stats.

> Delete the "old-project" index, I don't need it anymore.

**The RAG Panel** (magnifying glass + icon in the sidebar footer) lets you manage indexes, trigger re-indexing, and search visually.

**Supported file types:** Markdown, Python, JavaScript/TypeScript, JSON, YAML, HTML, CSS, SQL, shell scripts, C/C++/Rust/Go/Java, plain text, and 30+ more.

---

### 8. Scheduling & Automation

PRE can run tasks on a schedule (cron) or react to file changes and webhooks (triggers) — even when you're not at the computer.

```
  ┌─────────────────────────────────────────────────────┐
  │                AUTOMATION ENGINE                      │
  │                                                       │
  │  ┌───────────┐    ┌────────────┐    ┌─────────────┐ │
  │  │   Cron    │    │  Triggers  │    │  Workflows  │ │
  │  │ ⏰ Time   │    │ 📁 Files   │    │ 🔄 Replay   │ │
  │  │  based    │    │ 🌐 Webhooks│    │   recorded  │ │
  │  └─────┬─────┘    └─────┬──────┘    └──────┬──────┘ │
  │        │               │                  │         │
  │        └───────────────┴──────────────────┘         │
  │                        │                             │
  │              ┌─────────▼──────────┐                 │
  │              │  Headless Agent    │                 │
  │              │  (runs in          │                 │
  │              │   background)      │                 │
  │              └─────────┬──────────┘                 │
  │                        │                             │
  │              ┌─────────▼──────────┐                 │
  │              │  Notifications     │                 │
  │              │  macOS + Telegram  │                 │
  │              └────────────────────┘                 │
  └─────────────────────────────────────────────────────┘
```

**Cron jobs — scheduled recurring tasks:**

> Schedule a daily morning briefing at 8am that checks my calendar, unread emails, and any high-priority Jira tickets. Run it Monday through Friday.

> Schedule a job that runs every 6 hours to check disk usage and alert me if any volume is over 80% full.

> List all my scheduled jobs and their next run times.

> Disable the "morning-briefing" cron job — I'm on vacation this week.

**Triggers — react to changes:**

> Create a trigger that watches my ~/Downloads folder. When a new PDF appears, summarize its contents and save the summary to ~/Documents/summaries/.

> Create a trigger that watches the server.log file and alerts me via notification if it detects any ERROR lines.

> Create a webhook trigger called "deploy-notify" that sends me a Slack message when our CI/CD pipeline posts to it.

**The Cron Panel** (clock icon) and **Triggers Panel** (lightning icon) in the sidebar footer let you manage everything visually.

---

### 9. Sub-Agents — Parallel Research

When a task requires deep research, PRE can spawn sub-agents that work autonomously — reading files, searching the web, and gathering data without cluttering your main conversation.

```
  ┌──────────────────────────────────────────────────┐
  │                 AGENT ARCHITECTURE                │
  │                                                    │
  │  YOU ──> PRE (Main) ──┬──> Agent 1: "Research X" │
  │                       │                           │
  │                       ├──> Agent 2: "Research Y" │
  │                       │                           │
  │                       └──> Agent 3: "Research Z" │
  │                                                    │
  │  Each agent gets its own session, tools, and      │
  │  context. Results feed back to the main model.    │
  │                                                    │
  │  ┌─ Agent Feed (sidebar) ────────────────────┐   │
  │  │ ◉ Agent 1: Research X — 4 tools — 12.3s  │   │
  │  │ ◉ Agent 2: Research Y — 2 tools — 8.1s   │   │
  │  │ ✓ Agent 3: Research Z — 6 tools — 22.5s  │   │
  │  │                          [Open Session]    │   │
  │  └───────────────────────────────────────────┘   │
  └──────────────────────────────────────────────────┘
```

**Try these prompts:**

> Research the pros and cons of PostgreSQL vs. MySQL for a high-write workload. Spawn agents to investigate each database independently, then synthesize the findings.

> I need a competitive analysis. Spawn agents to research Notion, Obsidian, and Logseq — features, pricing, and user sentiment — then give me a comparison table.

> Spawn an agent to read all the README files in my ~/projects directory and create a summary of what each project does.

**The Agent Feed** (robot icon in the sidebar footer) shows real-time progress, tool usage, and results for all running agents. Each agent gets its own session you can click into for the full transcript.

---

### 10. Cloud Integrations

PRE connects to 15 enterprise services. Configure them in **Settings** (gear icon in the sidebar footer). Each requires a simple API key or OAuth setup.

**GitHub:**

> List my open pull requests on the frontend repo.

> Search GitHub for repositories related to "local AI assistants" with more than 100 stars.

> Show me the latest issues on my project, sorted by most recent.

**Slack:**

> Search Slack for messages about the production deployment in the last 24 hours.

> Send a message to #engineering saying "Deployment complete — all services green."

> Who's been most active in the #support channel this week?

**Jira:**

> Show me all Jira tickets assigned to me that are in "In Progress" status.

> Create a new Jira bug report: "Login page crashes on Safari 17 when using SSO". Set priority to High.

> What tickets were completed in the current sprint?

**Linear:**

> List my active Linear issues sorted by priority.

> Create a Linear issue in the Backend project: "Add rate limiting to the /api/search endpoint."

**Figma:**

> Show me the file structure of our Design System Figma file.

> Export the "Hero Section" component from our landing page Figma file as a PNG.

> Post a comment on the login screen frame: "The spacing between the input fields needs to match the 8px grid."

**Zoom:**

> What Zoom meetings do I have scheduled this week?

> Create a Zoom meeting for Thursday at 3pm called "Sprint Planning" with a 60-minute duration.

**Asana:**

> List all tasks assigned to me in the "Q2 Launch" Asana project.

> Create a new task: "Update API documentation for v2 endpoints" in the Engineering project, due next Friday.

**SharePoint:**

> Search SharePoint for documents about "onboarding procedures".

> List the files in our team's shared drive.

---

### 11. Artifacts — Interactive Documents

PRE can create rich, interactive HTML documents — dashboards, visualizations, games, and reports — that open directly in your browser.

**Try these prompts:**

> Create an interactive HTML dashboard showing a sample project timeline with milestones, progress bars, and status indicators. Make it visually stunning.

> Build a simple Pomodoro timer as an HTML artifact with start/pause/reset buttons, a circular progress indicator, and sound notifications.

> Create an HTML artifact that visualizes the Fibonacci sequence as a spiral using SVG.

> Generate a professional project status report as a formatted HTML document with charts, RAG status indicators, and an executive summary.

**Image generation (requires ComfyUI):**

> Generate an image of a serene Japanese garden at sunset with cherry blossoms, painted in watercolor style.

> Create a photorealistic image of a modern home office with warm lighting, a standing desk, and large monitors.

**Document export:**

> Create a Word document summarizing our meeting notes with action items and owners.

> Export the current conversation as a PDF.

---

### 12. Voice Interface

If Whisper is installed, PRE can listen to you speak and respond with voice. The microphone button appears in the chat input area.

**How it works:**

```
  ┌───────────┐      ┌──────────────┐      ┌───────────┐
  │ Hold mic  │      │  Whisper     │      │  PRE      │
  │ button    │─────>│  (local STT) │─────>│  processes│
  │ & speak   │      │  on your Mac │      │  & replies│
  └───────────┘      └──────────────┘      └─────┬─────┘
                                                   │
                                            ┌──────▼──────┐
                                            │  macOS say  │
                                            │  (TTS)      │
                                            │  speaks the │
                                            │  response   │
                                            └─────────────┘
```

**Try these prompts (type or speak):**

> Read this text aloud: "Good morning! Here's your daily briefing."

> What voices are available for text-to-speech?

> Speak my calendar events for today.

**All audio is processed locally** — nothing leaves your machine. No cloud transcription services, no recordings sent anywhere.

---

### 13. Deep Research Mode

For complex questions that need multi-step investigation, toggle **Deep Research** mode (beaker icon next to the send button). PRE will perform multiple rounds of tool use, searching, reading, and synthesizing before delivering a comprehensive answer.

**Try these prompts in Deep Research mode:**

> What are the current best practices for securing a Node.js REST API in 2026? Cover authentication, rate limiting, input validation, and dependency management.

> Research the history of the Scotts Valley, CA area — from the indigenous peoples to the present day tech industry. Include key dates and events.

> Analyze the architectural patterns used in large-scale local-first applications. Compare CRDTs, event sourcing, and operational transforms.

**Frontier AI delegation** (available from the dropdown next to the send button):

If you have Claude, Codex, or Gemini CLI installed, PRE can delegate complex reasoning tasks to frontier models while handling all local execution itself:

> [Delegate to Claude] Review this code architecture and suggest improvements for scalability.

> [Delegate to Gemini] Write a comprehensive technical specification for a real-time collaboration feature.

---

### 14. Power User Workflows

These prompts combine multiple features into real-world workflows.

**Morning startup routine:**

> Check my calendar for today, summarize any unread emails flagged as important, list my top-priority Jira tickets, and check if any cron jobs ran overnight. Give me a concise briefing.

**Project onboarding:**

> Index the entire ~/projects/new-app repository with RAG. Then search it for the main entry point, database schema, and authentication flow. Give me a developer onboarding summary.

**Automated documentation:**

> Read all the source files in src/tools/, identify every exported function, and create a comprehensive API reference document as an HTML artifact with a table of contents.

**System health check:**

> Check disk usage on all volumes, list the top 20 processes by memory, check if my Time Machine backup ran in the last 24 hours, and verify my network connectivity. Format as a system health report.

**Meeting preparation:**

> Search my email for the latest messages from the product team. Check my calendar for any meetings with them this week. Search Confluence for their latest project spec. Summarize everything so I'm prepared for tomorrow's meeting.

**End-of-day wrap-up:**

> Summarize what I worked on today based on our conversation history. Create a reminder for tomorrow's top priorities. Draft a standup update for Slack with what I accomplished and what's planned for tomorrow.

---

### GUI Quick Reference

```
  ┌────────────────────────────────────────────────────────────────┐
  │ SIDEBAR FOOTER — Your Control Center                           │
  │                                                                │
  │  ┌──────────────────────────────────────────────────────┐     │
  │  │ [Theme Toggle]                        [Settings ⚙️ ]  │     │
  │  ├──────────────────────────────────────────────────────┤     │
  │  │ [📖 Memory] [⏰ Cron] [⚡ Triggers]                   │     │
  │  │ [🔍 RAG] [🔄 Workflows] [🤖 Agent Feed]              │     │
  │  └──────────────────────────────────────────────────────┘     │
  │                                                                │
  │  📖 Memory     — Browse, search, and manage saved memories     │
  │  ⏰ Cron       — Create and manage scheduled recurring tasks   │
  │  ⚡ Triggers    — Set up file watchers and webhook listeners    │
  │  🔍 RAG        — Manage document indexes and semantic search   │
  │  🔄 Workflows  — View, replay, and manage recorded workflows   │
  │  🤖 Agent Feed — Monitor sub-agents and background jobs        │
  │  ⚙️ Settings    — Configure cloud integrations and preferences │
  └────────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
server.js                  Express + WebSocket + MCP server, REST API
mcp-stdio.js               MCP stdio transport (auto-starts server + Ollama)
pre-server.sh              Headless launcher for LaunchAgent auto-start (--status, --stop)
src/
  ollama.js                Ollama /api/chat NDJSON streaming
  sessions.js              JSONL read/write + project management
  tools.js                 Tool dispatcher + execution loop
  tools-defs.js            63+ tool definitions for Ollama
  context.js               System prompt builder
  memory.js                 Enhanced memory system (save, extract, age, context injection)
  mcp-server.js            MCP server definition (pre_agent, pre_chat, pre_memory_search, pre_sessions)
  connections.js            Connection management, Google/Microsoft OAuth, Telegram setup
  constants.js             MODEL_CTX (from ~/.pre/context), paths
  triggers.js              Event-driven trigger engine (file watchers, webhooks)
  tools/
    bash.js                Shell execution (stderr capture)
    files.js               read_file, list_dir, glob, grep, file_write, file_edit
    web.js                 web_fetch, web_search (Brave API)
    memory.js              save, search, list, delete
    system.js              17 system tools (info, processes, clipboard, etc.)
    artifact.js            Interactive HTML artifacts
    export.js              Artifact sharing (PDF, PNG, self-contained HTML via Puppeteer)
    mail.js                macOS Mail.app (AppleScript, zero-config)
    calendar.js            macOS Calendar.app (EventKit via compiled Swift binary)
    contacts.js            macOS Contacts.app (AppleScript, zero-config)
    spotlight.js           macOS Spotlight/mdfind (full-text file search)
    reminders.js           macOS Reminders.app (EventKit via compiled Swift binary)
    notes.js               macOS Notes.app (AppleScript, zero-config)
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
    rag.js                 Local RAG (directory indexing + semantic search)
    voice.js               Voice interface (Whisper STT + macOS say TTS)
    workflow.js            Workflow capture and replay (Computer Use sequences)
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

1. **Context window must match across CLI and Web GUI** — both read `~/.pre/context` at startup so the `num_ctx` value stays in sync; a mismatch triggers a 300s+ Ollama model reload
2. **Vanilla JS, no framework** — local tool, one user, loads instantly, no build step
3. **Credentials stay server-side** — API keys never sent to the browser
4. **Tool execution server-side** — security, CORS, file system access
5. **Session format compatibility** — CLI and web can interleave on the same session files
