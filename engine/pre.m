/*
 * pre.m — Personal Reasoning Engine (PRE)
 *
 * Fully local agentic assistant powered by Gemma 4 via Ollama.
 * Tool-calling, persistent memory, file-aware reasoning interface.
 *
 * Build:  make pre
 * Run:    ./pre-launch          (manages Ollama + PRE)
 *         ./pre [--port 11434] [--dir /path]
 */

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <getopt.h>
#include <dirent.h>
#include <limits.h>
#include <errno.h>
#include <glob.h>
#include <signal.h>
#include "linenoise.h"

// ============================================================================
// Constants
// ============================================================================

#define MAX_RESPONSE    (1024 * 1024)   // 1MB response buffer
#define MAX_ATTACH      (128 * 1024)    // 128KB max file attachment
#define MAX_BODY        (256 * 1024)    // 256KB max HTTP body
#define SESSIONS_DIR    ".pre/sessions"
#define HISTORY_FILE    ".pre/history"
#define MEMORY_DIR      ".pre/memory"
#define MEMORY_INDEX    ".pre/memory/index.md"
#define MAX_MEMORIES    128
#define MAX_MEMORY_VAL  4096
#define MODEL_NAME      "Gemma 4 26B-A4B"
#define MAX_TOOL_ARGS   8
#define MAX_ARG_VAL_LEN 65536
#define MAX_TOOL_LOOP_TURNS 25
#define CHECKPOINTS_DIR ".pre/checkpoints"
#define PROJECTS_DIR    ".pre/projects"
#define MAX_CHANNELS    64

// ANSI escape codes
#define ANSI_RESET      "\033[0m"
#define ANSI_BOLD       "\033[1m"
#define ANSI_DIM        "\033[2m"
#define ANSI_ITALIC     "\033[3m"
#define ANSI_CODE       "\033[36m"
#define ANSI_CODEBLK    "\033[48;5;236m\033[38;5;252m"
#define ANSI_CODEBLK_LINE "\033[48;5;236m\033[K"
#define ANSI_HEADER     "\033[1;34m"
#define ANSI_GREEN      "\033[32m"
#define ANSI_YELLOW     "\033[33m"
#define ANSI_RED        "\033[31m"
#define ANSI_CYAN       "\033[36m"
#define ANSI_MAGENTA    "\033[35m"
#define ANSI_REV        "\033[7m"

// ============================================================================
// Global state
// ============================================================================

// Approximate context window budget for the model (tokens)
#define MAX_CONTEXT     131072

typedef struct {
    int port;
    int max_tokens;
    char session_id[64];
    char cwd[PATH_MAX];
    char sessions_dir[1024];
    char history_path[1024];

    // Stats from last response
    int last_token_count;
    double last_tok_s;
    double last_ttft_ms;

    // Cumulative session stats
    int total_tokens_out;       // total generated tokens this session
    int total_tokens_in;        // estimated input tokens this session
    double cumulative_gen_ms;   // total generation time
    double session_start_ms;    // when session began

    // Conversation state
    char *last_response;
    int turn_count;
    char session_title[128];    // human-readable session name

    // Feature toggles
    int show_thinking;
    int auto_approve_tools;   // 'a' during tool confirm

    // File checkpoints for undo
    #define MAX_CHECKPOINTS 64
    struct { char path[PATH_MAX]; char backup[PATH_MAX]; } checkpoints[MAX_CHECKPOINTS];
    int checkpoint_count;
    int open_app_approved;  // track first-time approval for open_app

    // Project detection
    char project_root[PATH_MAX];  // detected project root (e.g. git root)
    char project_name[128];       // human-readable project name
    char project_id[128];         // sanitized id for directory names
    char project_dir[PATH_MAX];   // ~/.pre/projects/{project_id}/

    // Channel system
    char channel[64];             // active channel name (default: "general")
    char channel_session[128];    // session_id for current channel
} PreState;

static PreState g = {
    .port = 11434,
    .max_tokens = 8192,
    .show_thinking = 1,
    .turn_count = 0,
};

// Model to request from Ollama
static const char *g_model = "gemma4:26b-a4b-it-q4_K_M";

// ============================================================================
// ToolCall struct and helpers
// ============================================================================

typedef struct {
    char name[64];
    int argc;
    char keys[MAX_TOOL_ARGS][64];
    char *vals[MAX_TOOL_ARGS];  // heap-allocated
} ToolCall;

static void tool_call_free(ToolCall *tc) {
    for (int i = 0; i < tc->argc; i++) {
        free(tc->vals[i]);
        tc->vals[i] = NULL;
    }
    tc->argc = 0;
}

static const char *tool_call_get(const ToolCall *tc, const char *key) {
    for (int i = 0; i < tc->argc; i++) {
        if (strcmp(tc->keys[i], key) == 0) return tc->vals[i];
    }
    return NULL;
}

// ============================================================================
// Permission model
// ============================================================================

typedef enum { PERM_AUTO, PERM_CONFIRM_ONCE, PERM_CONFIRM_ALWAYS } PermLevel;

static PermLevel tool_permission(const char *name) {
    if (strcmp(name, "read_file") == 0 || strcmp(name, "list_dir") == 0 ||
        strcmp(name, "glob") == 0 || strcmp(name, "grep") == 0 ||
        strcmp(name, "clipboard_read") == 0 || strcmp(name, "system_info") == 0 ||
        strcmp(name, "process_list") == 0 || strcmp(name, "memory_search") == 0 ||
        strcmp(name, "memory_list") == 0 || strcmp(name, "memory_save") == 0 ||
        strcmp(name, "window_list") == 0 || strcmp(name, "display_info") == 0 ||
        strcmp(name, "net_info") == 0 || strcmp(name, "net_connections") == 0 ||
        strcmp(name, "service_status") == 0 || strcmp(name, "disk_usage") == 0 ||
        strcmp(name, "hardware_info") == 0) return PERM_AUTO;
    if (strcmp(name, "file_write") == 0 || strcmp(name, "file_edit") == 0 ||
        strcmp(name, "clipboard_write") == 0 || strcmp(name, "web_fetch") == 0 ||
        strcmp(name, "notify") == 0 || strcmp(name, "memory_delete") == 0 ||
        strcmp(name, "screenshot") == 0 || strcmp(name, "window_focus") == 0) return PERM_CONFIRM_ONCE;
    return PERM_CONFIRM_ALWAYS; // bash, process_kill, open_app (first time)
}

// Pending file attachment(s) — prepended to next message
static char *g_pending_attach = NULL;

// ============================================================================
// Utilities
// ============================================================================

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// JSON-escape into a malloc'd buffer (handles arbitrarily large input)
static char *json_escape_alloc(const char *src) {
    size_t slen = strlen(src);
    size_t cap = slen * 2 + 16;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    size_t j = 0;
    for (size_t i = 0; i < slen; i++) {
        if (j + 8 > cap) {
            cap *= 2;
            buf = realloc(buf, cap);
            if (!buf) return NULL;
        }
        switch (src[i]) {
            case '"':  buf[j++]='\\'; buf[j++]='"'; break;
            case '\\': buf[j++]='\\'; buf[j++]='\\'; break;
            case '\n': buf[j++]='\\'; buf[j++]='n'; break;
            case '\r': buf[j++]='\\'; buf[j++]='r'; break;
            case '\t': buf[j++]='\\'; buf[j++]='t'; break;
            default:
                if ((unsigned char)src[i] < 0x20) {
                    j += sprintf(buf + j, "\\u%04x", (unsigned char)src[i]);
                } else {
                    buf[j++] = src[i];
                }
                break;
        }
    }
    buf[j] = 0;
    return buf;
}


// Format file size for display
static const char *fmt_size(off_t bytes) {
    static char buf[32];
    if (bytes >= 1024 * 1024 * 1024) snprintf(buf, sizeof(buf), "%.1fG", bytes / (1024.0*1024*1024));
    else if (bytes >= 1024 * 1024) snprintf(buf, sizeof(buf), "%.1fM", bytes / (1024.0*1024));
    else if (bytes >= 1024) snprintf(buf, sizeof(buf), "%.1fK", bytes / 1024.0);
    else snprintf(buf, sizeof(buf), "%lld", (long long)bytes);
    return buf;
}

// Format elapsed time for display
static const char *fmt_elapsed(double ms) {
    static char buf[32];
    double secs = ms / 1000.0;
    if (secs < 60) snprintf(buf, sizeof(buf), "%.0fs", secs);
    else if (secs < 3600) snprintf(buf, sizeof(buf), "%dm%02ds", (int)secs/60, (int)secs%60);
    else snprintf(buf, sizeof(buf), "%dh%02dm", (int)secs/3600, ((int)secs%3600)/60);
    return buf;
}

// Resolve path relative to CWD
static void resolve_path(const char *input, char *out, size_t outsize) {
    if (input[0] == '/' || input[0] == '~') {
        if (input[0] == '~') {
            const char *home = getenv("HOME") ?: "/tmp";
            snprintf(out, outsize, "%s%s", home, input + 1);
        } else {
            strncpy(out, input, outsize - 1);
            out[outsize - 1] = 0;
        }
    } else {
        snprintf(out, outsize, "%s/%s", g.cwd, input);
    }
}

// ============================================================================
// PRE directories and file checkpoints
// ============================================================================

static void init_pre_dirs(void) {
    char path[PATH_MAX];
    const char *home = getenv("HOME");
    snprintf(path, sizeof(path), "%s/.pre", home);
    mkdir(path, 0755);
    snprintf(path, sizeof(path), "%s/.pre/checkpoints", home);
    mkdir(path, 0755);
}

static int checkpoint_file(const char *path) {
    struct stat st;
    if (stat(path, &st) < 0) return 0; // new file, nothing to backup

    if (g.checkpoint_count >= MAX_CHECKPOINTS) return 0;

    char backup[PATH_MAX];
    const char *home = getenv("HOME");
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    snprintf(backup, sizeof(backup), "%s/.pre/checkpoints/%s_%d_%s",
             home, g.session_id, g.checkpoint_count, base);

    // Copy file
    FILE *src = fopen(path, "r");
    FILE *dst = fopen(backup, "w");
    if (!src || !dst) { if (src) fclose(src); if (dst) fclose(dst); return 0; }
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), src)) > 0) fwrite(buf, 1, n, dst);
    fclose(src); fclose(dst);

    strncpy(g.checkpoints[g.checkpoint_count].path, path, PATH_MAX - 1);
    strncpy(g.checkpoints[g.checkpoint_count].backup, backup, PATH_MAX - 1);
    g.checkpoint_count++;

    printf(ANSI_DIM "  [checkpointed %s]" ANSI_RESET "\n", base);
    return 1;
}

// Forward declarations for functions used by project/channel code
static int session_load_title(const char *session_id, char *out, size_t outsize);

// ============================================================================
// Project detection
// ============================================================================

// Detect project root by walking up from cwd looking for markers
static void detect_project(void) {
    g.project_root[0] = 0;
    g.project_name[0] = 0;
    g.project_id[0] = 0;
    g.project_dir[0] = 0;

    static const char *markers[] = {
        ".git", "package.json", "pyproject.toml", "Cargo.toml",
        "go.mod", "Makefile", "CMakeLists.txt", "pom.xml",
        ".pre.md", "PRE.md", NULL
    };

    char dir[PATH_MAX];
    strncpy(dir, g.cwd, sizeof(dir) - 1);

    while (strlen(dir) > 1) {
        for (int i = 0; markers[i]; i++) {
            char check[PATH_MAX];
            snprintf(check, sizeof(check), "%s/%s", dir, markers[i]);
            struct stat st;
            if (stat(check, &st) == 0) {
                strncpy(g.project_root, dir, sizeof(g.project_root) - 1);
                goto found;
            }
        }
        // Walk up
        char *slash = strrchr(dir, '/');
        if (!slash || slash == dir) break;
        *slash = 0;
    }
    return; // no project found

found:;
    // Derive project name from directory basename
    const char *base = strrchr(g.project_root, '/');
    base = base ? base + 1 : g.project_root;
    strncpy(g.project_name, base, sizeof(g.project_name) - 1);

    // Sanitize into project_id (lowercase, replace non-alnum with _)
    int pi = 0;
    for (int i = 0; base[i] && pi < 126; i++) {
        char c = base[i];
        if (c >= 'A' && c <= 'Z') g.project_id[pi++] = c + 32;
        else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) g.project_id[pi++] = c;
        else if (c == '-' || c == '_' || c == '.') g.project_id[pi++] = c;
        else g.project_id[pi++] = '_';
    }
    g.project_id[pi] = 0;

    // Create project directory under ~/.pre/projects/
    const char *home = getenv("HOME");
    char proj_base[PATH_MAX];
    snprintf(proj_base, sizeof(proj_base), "%s/.pre/projects", home);
    mkdir(proj_base, 0755);

    snprintf(g.project_dir, sizeof(g.project_dir), "%s/%s", proj_base, g.project_id);
    mkdir(g.project_dir, 0755);

    // Create project subdirs
    char sub[PATH_MAX];
    snprintf(sub, sizeof(sub), "%s/memory", g.project_dir);
    mkdir(sub, 0755);
    snprintf(sub, sizeof(sub), "%s/channels", g.project_dir);
    mkdir(sub, 0755);
}

// ============================================================================
// Channel system
// ============================================================================

// Initialize channel — sets g.channel, g.channel_session, and session paths
static void channel_init(const char *name) {
    if (!name || !name[0]) name = "general";
    strncpy(g.channel, name, sizeof(g.channel) - 1);

    // Channel session id: project_id:channel (or just channel if no project)
    if (g.project_id[0])
        snprintf(g.channel_session, sizeof(g.channel_session), "%s:%s", g.project_id, name);
    else
        snprintf(g.channel_session, sizeof(g.channel_session), "global:%s", name);

    // Use channel_session as the session_id for JSONL storage
    strncpy(g.session_id, g.channel_session, sizeof(g.session_id) - 1);

    // Reset conversation state for the new channel
    g.turn_count = 0;
    g.total_tokens_in = 0;
    g.total_tokens_out = 0;
    g.cumulative_gen_ms = 0;
    g.session_start_ms = now_ms();
    free(g.last_response); g.last_response = NULL;
    g.session_title[0] = 0;
    g.auto_approve_tools = 0;
}

// List channels for current project (or global)
static void channel_list(void) {
    // Channels are discovered from session JSONL files matching the project prefix
    const char *prefix = g.project_id[0] ? g.project_id : "global";
    size_t plen = strlen(prefix);

    DIR *dir = opendir(g.sessions_dir);
    if (!dir) { printf("  No channels found.\n\n"); return; }

    printf(ANSI_BOLD "  Channels" ANSI_RESET);
    if (g.project_name[0]) printf(" (%s)", g.project_name);
    printf(":\n");

    struct dirent *entry;
    int count = 0;
    while ((entry = readdir(dir))) {
        if (entry->d_name[0] == '.') continue;
        char *dot = strrchr(entry->d_name, '.');
        if (!dot || strcmp(dot, ".jsonl") != 0) continue;

        // Check if it matches our project prefix
        if (strncmp(entry->d_name, prefix, plen) != 0) continue;
        if (entry->d_name[plen] != ':') continue;

        // Extract channel name
        const char *chan = entry->d_name + plen + 1;
        char chan_name[64];
        strncpy(chan_name, chan, 63);
        char *ext = strrchr(chan_name, '.');
        if (ext) *ext = 0;

        // Count lines in session file
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", g.sessions_dir, entry->d_name);
        FILE *f = fopen(path, "r");
        int lines = 0;
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) lines++;
            fclose(f);
        }

        int active = (strcmp(chan_name, g.channel) == 0);
        printf("  %s " ANSI_CYAN "%s" ANSI_RESET "  (%d messages)\n",
               active ? ANSI_GREEN "▸" ANSI_RESET : " ",
               chan_name, lines);
        count++;
    }
    closedir(dir);

    if (count == 0) {
        printf("  " ANSI_GREEN "▸" ANSI_RESET " " ANSI_CYAN "general" ANSI_RESET "  (new)\n");
    }
    printf("\n");
}

// Switch to a channel — loads existing session or starts fresh
static void channel_switch(const char *name) {
    if (!name || !name[0]) {
        printf(ANSI_YELLOW "  Usage: /channel <name>" ANSI_RESET "\n\n");
        return;
    }

    // Sanitize channel name
    char clean[64];
    int ci = 0;
    for (int i = 0; name[i] && ci < 62; i++) {
        char c = name[i];
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_')
            clean[ci++] = c;
        else if (c >= 'A' && c <= 'Z')
            clean[ci++] = c + 32;
        else if (c == ' ')
            clean[ci++] = '-';
    }
    clean[ci] = 0;
    if (ci == 0) {
        printf(ANSI_YELLOW "  Invalid channel name" ANSI_RESET "\n\n");
        return;
    }

    channel_init(clean);

    // Try to load existing session for this channel
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.jsonl", g.sessions_dir, g.session_id);
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 0) {
        // Count turns
        FILE *f = fopen(path, "r");
        if (f) {
            char buf[4096];
            int lines = 0;
            while (fgets(buf, sizeof(buf), f)) lines++;
            fclose(f);
            g.turn_count = lines / 2;
        }
        printf(ANSI_GREEN "  [channel: #%s — %d turns]" ANSI_RESET "\n\n", clean, g.turn_count);
    } else {
        printf(ANSI_GREEN "  [channel: #%s — new]" ANSI_RESET "\n\n", clean);
    }

    // Load channel title if exists
    session_load_title(g.session_id, g.session_title, sizeof(g.session_title));
}

// ============================================================================
// Memory system — persistent file-based memory
// ============================================================================

// Memory entry loaded from disk
typedef struct {
    char name[128];
    char type[32];       // user, feedback, project, reference
    char description[256];
    char file[PATH_MAX]; // path to the .md file
} MemoryEntry;

static MemoryEntry g_memories[MAX_MEMORIES];
static int g_memory_count = 0;

static void init_memory_dir(void) {
    char path[PATH_MAX];
    const char *home = getenv("HOME");
    snprintf(path, sizeof(path), "%s/.pre/memory", home);
    mkdir(path, 0755);

    // Create index.md index if it doesn't exist
    char idx[PATH_MAX];
    snprintf(idx, sizeof(idx), "%s/.pre/memory/index.md", home);
    struct stat st;
    if (stat(idx, &st) < 0) {
        FILE *f = fopen(idx, "w");
        if (f) {
            fprintf(f, "# PRE Memory\n\n");
            fclose(f);
        }
    }
}

// Parse frontmatter from a memory .md file
// Returns 1 on success, fills entry
static int parse_memory_file(const char *path, MemoryEntry *entry) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    memset(entry, 0, sizeof(*entry));
    strncpy(entry->file, path, PATH_MAX - 1);

    char line[1024];
    int in_front = 0;
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\n\r")] = 0;
        if (strcmp(line, "---") == 0) {
            if (in_front) break; // end of frontmatter
            in_front = 1;
            continue;
        }
        if (!in_front) continue;

        // Parse key: value
        char *colon = strchr(line, ':');
        if (!colon) continue;
        *colon = 0;
        char *key = line;
        char *val = colon + 1;
        while (*val == ' ') val++;
        // Trim trailing whitespace from key
        int kl = (int)strlen(key);
        while (kl > 0 && key[kl-1] == ' ') key[--kl] = 0;

        if (strcmp(key, "name") == 0) strncpy(entry->name, val, sizeof(entry->name) - 1);
        else if (strcmp(key, "type") == 0) strncpy(entry->type, val, sizeof(entry->type) - 1);
        else if (strcmp(key, "description") == 0) strncpy(entry->description, val, sizeof(entry->description) - 1);
    }
    fclose(f);
    return entry->name[0] != 0;
}

// Load all memories from ~/.pre/memory/*.md (not index.md)
// Load .md files from a directory into g_memories (skips index.md)
static void load_memories_from_dir(const char *dir_path) {
    char pattern[PATH_MAX];
    snprintf(pattern, sizeof(pattern), "%s/*.md", dir_path);

    glob_t gl;
    if (glob(pattern, GLOB_TILDE, NULL, &gl) != 0) return;

    for (size_t i = 0; i < gl.gl_pathc && g_memory_count < MAX_MEMORIES; i++) {
        const char *base = strrchr(gl.gl_pathv[i], '/');
        base = base ? base + 1 : gl.gl_pathv[i];
        if (strcmp(base, "index.md") == 0) continue;

        if (parse_memory_file(gl.gl_pathv[i], &g_memories[g_memory_count])) {
            g_memory_count++;
        }
    }
    globfree(&gl);
}

static void load_memories(void) {
    g_memory_count = 0;
    const char *home = getenv("HOME");

    // Load global memories
    char global_dir[PATH_MAX];
    snprintf(global_dir, sizeof(global_dir), "%s/.pre/memory", home);
    load_memories_from_dir(global_dir);

    // Load project-scoped memories if in a project
    if (g.project_dir[0]) {
        char proj_mem[PATH_MAX];
        snprintf(proj_mem, sizeof(proj_mem), "%s/memory", g.project_dir);
        load_memories_from_dir(proj_mem);
    }
}

// Read the body (content after frontmatter) of a memory file
static char *read_memory_body(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;

    char line[1024];
    int dashes = 0;
    // Skip frontmatter
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "---", 3) == 0) {
            dashes++;
            if (dashes >= 2) break;
        }
    }

    // Read rest
    size_t cap = 4096, len = 0;
    char *body = malloc(cap);
    while (fgets(line, sizeof(line), f)) {
        size_t ll = strlen(line);
        if (len + ll + 1 > cap) { cap *= 2; body = realloc(body, cap); }
        memcpy(body + len, line, ll);
        len += ll;
    }
    body[len] = 0;
    fclose(f);

    // Trim leading newlines
    char *start = body;
    while (*start == '\n' || *start == '\r') start++;
    if (start != body) memmove(body, start, strlen(start) + 1);

    return body;
}

// Save a memory: writes the .md file and updates index.md index
// scope: "project" saves to project memory dir, anything else saves globally
static int save_memory(const char *name, const char *type, const char *description, const char *content, const char *scope) {
    const char *home = getenv("HOME");

    // Generate filename from name: lowercase, replace spaces with _
    char filename[128];
    int fi = 0;
    for (int i = 0; name[i] && fi < 120; i++) {
        char c = name[i];
        if (c == ' ' || c == '/' || c == '\\') filename[fi++] = '_';
        else if (c >= 'A' && c <= 'Z') filename[fi++] = c + 32;
        else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')
            filename[fi++] = c;
    }
    filename[fi] = 0;
    if (fi == 0) strncpy(filename, "untitled", sizeof(filename));

    // Determine target directory
    char mem_dir[PATH_MAX];
    int is_project = (scope && strcmp(scope, "project") == 0 && g.project_dir[0]);
    if (is_project) {
        snprintf(mem_dir, sizeof(mem_dir), "%s/memory", g.project_dir);
    } else {
        snprintf(mem_dir, sizeof(mem_dir), "%s/.pre/memory", home);
    }

    char filepath[PATH_MAX];
    snprintf(filepath, sizeof(filepath), "%s/%s.md", mem_dir, filename);

    // Write the memory file with frontmatter
    FILE *f = fopen(filepath, "w");
    if (!f) return 0;
    fprintf(f, "---\nname: %s\ndescription: %s\ntype: %s\n---\n\n%s\n", name, description, type, content);
    fclose(f);

    // Update index.md in the appropriate directory
    char idx_path[PATH_MAX];
    snprintf(idx_path, sizeof(idx_path), "%s/index.md", mem_dir);

    // Read existing index
    FILE *idx = fopen(idx_path, "r");
    size_t idx_cap = 8192, idx_len = 0;
    char *idx_buf = malloc(idx_cap);
    idx_buf[0] = 0;
    int found = 0;
    if (idx) {
        char line[512];
        while (fgets(line, sizeof(line), idx)) {
            // Check if this line references the same file
            char ref[256];
            snprintf(ref, sizeof(ref), "(%s.md)", filename);
            if (strstr(line, ref)) {
                // Replace this line with updated entry
                int wrote = snprintf(idx_buf + idx_len, idx_cap - idx_len,
                    "- [%s](%s.md) — %s\n", name, filename, description);
                if (wrote > 0) idx_len += wrote;
                found = 1;
            } else {
                size_t ll = strlen(line);
                if (idx_len + ll + 1 > idx_cap) { idx_cap *= 2; idx_buf = realloc(idx_buf, idx_cap); }
                memcpy(idx_buf + idx_len, line, ll);
                idx_len += ll;
            }
        }
        fclose(idx);
    }

    if (!found) {
        // Append new entry
        if (idx_len == 0) {
            idx_len = snprintf(idx_buf, idx_cap, "# PRE Memory\n\n");
        }
        idx_len += snprintf(idx_buf + idx_len, idx_cap - idx_len,
            "- [%s](%s.md) — %s\n", name, filename, description);
    }
    idx_buf[idx_len] = 0;

    idx = fopen(idx_path, "w");
    if (idx) { fputs(idx_buf, idx); fclose(idx); }
    free(idx_buf);

    // Reload memories
    load_memories();
    return 1;
}

// Delete a memory by name (partial match)
static int delete_memory(const char *query) {
    int deleted = 0;
    for (int i = 0; i < g_memory_count; i++) {
        if (strcasestr(g_memories[i].name, query) ||
            strcasestr(g_memories[i].description, query)) {
            // Remove the .md file
            remove(g_memories[i].file);

            // Remove from index.md index
            const char *base = strrchr(g_memories[i].file, '/');
            base = base ? base + 1 : g_memories[i].file;

            char idx_path[PATH_MAX];
            const char *home = getenv("HOME");
            snprintf(idx_path, sizeof(idx_path), "%s/.pre/memory/index.md", home);

            FILE *idx = fopen(idx_path, "r");
            if (idx) {
                size_t cap = 8192, len = 0;
                char *buf = malloc(cap);
                buf[0] = 0;
                char line[512];
                while (fgets(line, sizeof(line), idx)) {
                    if (!strstr(line, base)) {
                        size_t ll = strlen(line);
                        if (len + ll + 1 > cap) { cap *= 2; buf = realloc(buf, cap); }
                        memcpy(buf + len, line, ll);
                        len += ll;
                    }
                }
                fclose(idx);
                buf[len] = 0;
                idx = fopen(idx_path, "w");
                if (idx) { fputs(buf, idx); fclose(idx); }
                free(buf);
            }
            deleted++;
            break; // delete first match only
        }
    }
    if (deleted) load_memories();
    return deleted;
}

// Search memories by query — returns heap-allocated result string
static char *search_memories(const char *query) {
    size_t cap = 4096, len = 0;
    char *result = malloc(cap);
    result[0] = 0;
    int matches = 0;

    for (int i = 0; i < g_memory_count; i++) {
        int match = 0;
        if (!query || !query[0]) match = 1; // list all
        else if (strcasestr(g_memories[i].name, query)) match = 1;
        else if (strcasestr(g_memories[i].description, query)) match = 1;
        else if (strcasestr(g_memories[i].type, query)) match = 1;
        else {
            // Search body
            char *body = read_memory_body(g_memories[i].file);
            if (body && strcasestr(body, query)) match = 1;
            free(body);
        }

        if (match) {
            char *body = read_memory_body(g_memories[i].file);
            int wrote = snprintf(result + len, cap - len,
                "[%s] (%s) %s\n%s\n\n",
                g_memories[i].type, g_memories[i].name,
                g_memories[i].description,
                body ? body : "");
            free(body);
            if (wrote > 0) {
                len += wrote;
                if (len + 2048 > cap) { cap *= 2; result = realloc(result, cap); }
            }
            matches++;
        }
    }

    if (matches == 0) {
        snprintf(result, cap, "No memories found%s%s%s.",
                 query && query[0] ? " matching '" : "",
                 query && query[0] ? query : "",
                 query && query[0] ? "'" : "");
    }
    return result;
}

// Build a compact memory summary for context injection
static char *build_memory_context(void) {
    if (g_memory_count == 0) return NULL;

    size_t cap = 8192, len = 0;
    char *buf = malloc(cap);
    len = snprintf(buf, cap, "<memory>\n");

    for (int i = 0; i < g_memory_count; i++) {
        char *body = read_memory_body(g_memories[i].file);
        int wrote = snprintf(buf + len, cap - len,
            "## %s (%s)\n%s\n\n",
            g_memories[i].name, g_memories[i].type,
            body ? body : g_memories[i].description);
        free(body);
        if (wrote > 0) len += wrote;
        if (len + 2048 > cap) { cap *= 2; buf = realloc(buf, cap); }
    }
    len += snprintf(buf + len, cap - len, "</memory>\n");
    buf[len] = 0;
    return buf;
}

// ============================================================================
// TUI — status bar, banner, spinner
// ============================================================================

// Status bar removed — was using cursor positioning that conflicted with linenoise.
// Status info is now printed inline after each response.

static void tui_banner(void) {
    printf(ANSI_BOLD ANSI_CYAN
           "╔══════════════════════════════════════════════════╗\n"
           "║" ANSI_RESET ANSI_BOLD "  Personal Reasoning Engine (PRE)                "
           ANSI_CYAN "║\n"
           "║" ANSI_RESET "  " ANSI_DIM "%s" ANSI_RESET
           "                   " ANSI_CYAN "║\n"
           "╚══════════════════════════════════════════════════╝\n" ANSI_RESET,
           MODEL_NAME);
    printf("  Server:  " ANSI_GREEN "http://localhost:%d" ANSI_RESET "\n", g.port);
    if (g.project_name[0])
        printf("  Project: " ANSI_BOLD "%s" ANSI_RESET "  " ANSI_DIM "%s" ANSI_RESET "\n",
               g.project_name, g.project_root);
    printf("  Channel: " ANSI_CYAN "#%s" ANSI_RESET "\n", g.channel[0] ? g.channel : "general");
    printf("  CWD:     %s\n", g.cwd);
    if (g_memory_count > 0)
        printf("  Memory:  %d entries loaded\n", g_memory_count);
    printf("  Type " ANSI_BOLD "/help" ANSI_RESET " for commands\n\n");
}

// ============================================================================
// Session management
// ============================================================================

static void init_sessions_dir(void) {
    const char *home = getenv("HOME") ?: "/tmp";
    char parent[1024];
    snprintf(parent, sizeof(parent), "%s/.pre", home);
    mkdir(parent, 0755);
    snprintf(g.sessions_dir, sizeof(g.sessions_dir), "%s/%s", home, SESSIONS_DIR);
    mkdir(g.sessions_dir, 0755);
    snprintf(g.history_path, sizeof(g.history_path), "%s/%s", home, HISTORY_FILE);
}


static void session_save_title(const char *session_id, const char *title) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.title", g.sessions_dir, session_id);
    FILE *f = fopen(path, "w");
    if (f) { fputs(title, f); fclose(f); }
}

static int session_load_title(const char *session_id, char *out, size_t outsize) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.title", g.sessions_dir, session_id);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    if (!fgets(out, (int)outsize, f)) { fclose(f); return 0; }
    fclose(f);
    // Strip trailing whitespace/newlines
    int len = (int)strlen(out);
    while (len > 0 && (out[len-1] == '\n' || out[len-1] == '\r' || out[len-1] == ' ')) out[--len] = 0;
    return len > 0;
}

static void session_save_turn(const char *session_id, const char *role, const char *content) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.jsonl", g.sessions_dir, session_id);
    FILE *f = fopen(path, "a");
    if (!f) return;
    char *escaped = json_escape_alloc(content);
    if (escaped) {
        fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
        free(escaped);
    }
    fclose(f);
}

static int session_load(const char *session_id) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.jsonl", g.sessions_dir, session_id);
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    printf(ANSI_DIM "[resuming session %s]" ANSI_RESET "\n\n", session_id);
    int turns = 0;
    char line[MAX_RESPONSE];
    while (fgets(line, sizeof(line), f)) {
        char *role_start = strstr(line, "\"role\":\"");
        char *content_start = strstr(line, "\"content\":\"");
        if (!role_start || !content_start) continue;

        role_start += 8;
        char role[32]; int ri = 0;
        while (*role_start && *role_start != '"' && ri < 31) role[ri++] = *role_start++;
        role[ri] = 0;

        content_start += 11;
        char content[MAX_RESPONSE]; int ci = 0;
        for (int i = 0; content_start[i] && ci < MAX_RESPONSE - 1; i++) {
            if (content_start[i] == '"' && (i == 0 || content_start[i-1] != '\\')) break;
            if (content_start[i] == '\\' && content_start[i+1]) {
                i++;
                switch (content_start[i]) {
                    case 'n': content[ci++] = '\n'; break;
                    case 't': content[ci++] = '\t'; break;
                    case '"': content[ci++] = '"'; break;
                    case '\\': content[ci++] = '\\'; break;
                    default: content[ci++] = content_start[i]; break;
                }
            } else {
                content[ci++] = content_start[i];
            }
        }
        content[ci] = 0;

        if (strcmp(role, "user") == 0) {
            printf(ANSI_BOLD ANSI_GREEN "> " ANSI_RESET ANSI_BOLD "%s" ANSI_RESET "\n\n", content);
        } else if (strcmp(role, "assistant") == 0) {
            // Truncate long responses in replay
            if (ci > 500) {
                content[497] = '.'; content[498] = '.'; content[499] = '.'; content[500] = 0;
            }
            printf("%s\n\n", content);
        }
        turns++;
    }
    fclose(f);
    if (turns > 0) printf(ANSI_DIM "[%d turns loaded]" ANSI_RESET "\n\n", turns);
    return turns;
}

static void session_list(void) {
    DIR *dir = opendir(g.sessions_dir);
    if (!dir) { printf("  No sessions found.\n\n"); return; }

    printf(ANSI_BOLD "  Sessions:" ANSI_RESET "\n");
    struct dirent *entry;
    int count = 0;
    while ((entry = readdir(dir))) {
        if (entry->d_name[0] == '.') continue;
        char *dot = strrchr(entry->d_name, '.');
        if (!dot || strcmp(dot, ".jsonl") != 0) continue;
        *dot = 0;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.jsonl", g.sessions_dir, entry->d_name);
        struct stat st;
        stat(path, &st);

        // Count lines and get first user message preview
        FILE *f = fopen(path, "r");
        int lines = 0;
        char preview[80] = "";
        if (f) {
            char buf[4096];
            while (fgets(buf, sizeof(buf), f)) {
                lines++;
                if (preview[0] == 0 && strstr(buf, "\"user\"")) {
                    char *cs = strstr(buf, "\"content\":\"");
                    if (cs) {
                        cs += 11;
                        int pi = 0;
                        while (cs[pi] && cs[pi] != '"' && pi < 60) {
                            if (cs[pi] == '\\') { pi++; if (cs[pi]) pi++; continue; }
                            preview[pi] = cs[pi]; pi++;
                        }
                        preview[pi] = 0;
                    }
                }
            }
            fclose(f);
        }

        // Check for title sidecar file
        char title[128] = "";
        session_load_title(entry->d_name, title, sizeof(title));

        if (title[0])
            printf("  " ANSI_CYAN "%s" ANSI_RESET "  " ANSI_BOLD "%s" ANSI_RESET "  (%d turns)\n",
                   entry->d_name, title, lines);
        else
            printf("  " ANSI_CYAN "%s" ANSI_RESET "  (%d turns)  " ANSI_DIM "%s" ANSI_RESET "\n",
                   entry->d_name, lines, preview);
        count++;
    }
    closedir(dir);
    if (count == 0) printf("  (none)\n");
    printf("\n");
}

// ============================================================================
// File & directory operations
// ============================================================================

// Read file contents wrapped in fences for context injection
static char *file_read_for_context(const char *path) {
    struct stat st;
    if (stat(path, &st) < 0) return NULL;
    if (S_ISDIR(st.st_mode)) return NULL;

    size_t size = (size_t)st.st_size;
    if (size > MAX_ATTACH) {
        fprintf(stderr, ANSI_YELLOW "  [warning: truncating to %dKB]" ANSI_RESET "\n", MAX_ATTACH / 1024);
        size = MAX_ATTACH;
    }

    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    char *data = malloc(size + 1);
    size_t nread = fread(data, 1, size, f);
    fclose(f);
    data[nread] = 0;

    // Check for binary content
    int binary = 0;
    for (size_t i = 0; i < nread && i < 512; i++) {
        if (data[i] == 0) { binary = 1; break; }
    }
    if (binary) {
        free(data);
        return NULL;
    }

    // Wrap in fences
    size_t total = nread + strlen(path) + 128;
    char *result = malloc(total);
    snprintf(result, total, "--- FILE: %s (%s) ---\n%s\n--- END FILE ---\n", path, fmt_size(st.st_size), data);
    free(data);
    return result;
}

// Directory listing
static char *dir_listing(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) return NULL;

    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    len += snprintf(buf + len, cap - len, "Directory: %s\n\n", path);

    struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (entry->d_name[0] == '.') continue;
        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", path, entry->d_name);
        struct stat st;
        if (stat(full, &st) < 0) continue;

        if (len + 256 > cap) { cap *= 2; buf = realloc(buf, cap); }

        if (S_ISDIR(st.st_mode)) {
            len += snprintf(buf + len, cap - len, "  %s/\n", entry->d_name);
        } else {
            len += snprintf(buf + len, cap - len, "  %-40s %s\n", entry->d_name, fmt_size(st.st_size));
        }
    }
    closedir(dir);
    buf[len] = 0;
    return buf;
}

// Directory tree (recursive)
static void dir_tree_recurse(const char *path, int depth, int max_depth, char **buf, size_t *len, size_t *cap) {
    if (depth >= max_depth) return;
    DIR *dir = opendir(path);
    if (!dir) return;

    // Skip common noise directories
    static const char *skip[] = {".git", "node_modules", "__pycache__", ".venv", "venv",
                                  ".cache", ".DS_Store", NULL};

    struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (entry->d_name[0] == '.') continue;
        int skip_it = 0;
        for (int i = 0; skip[i]; i++) {
            if (strcmp(entry->d_name, skip[i]) == 0) { skip_it = 1; break; }
        }
        if (skip_it) continue;

        if (*len + 512 > *cap) { *cap *= 2; *buf = realloc(*buf, *cap); }

        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", path, entry->d_name);
        struct stat st;
        if (stat(full, &st) < 0) continue;

        for (int i = 0; i < depth; i++) *len += snprintf(*buf + *len, *cap - *len, "  ");

        if (S_ISDIR(st.st_mode)) {
            *len += snprintf(*buf + *len, *cap - *len, "%s/\n", entry->d_name);
            dir_tree_recurse(full, depth + 1, max_depth, buf, len, cap);
        } else {
            *len += snprintf(*buf + *len, *cap - *len, "%s  (%s)\n", entry->d_name, fmt_size(st.st_size));
        }
    }
    closedir(dir);
}

static char *dir_tree(const char *path, int max_depth) {
    size_t cap = 8192, len = 0;
    char *buf = malloc(cap);
    len += snprintf(buf + len, cap - len, "%s/\n", path);
    dir_tree_recurse(path, 1, max_depth, &buf, &len, &cap);
    buf[len] = 0;
    return buf;
}

// ============================================================================
// Context injection — build preamble for first message in session
// ============================================================================

static char *build_context_preamble(void) {
    // Get a compact directory listing for context
    DIR *dir = opendir(g.cwd);
    char files_list[4096] = "";
    int flen = 0;
    if (dir) {
        struct dirent *entry;
        while ((entry = readdir(dir))) {
            if (entry->d_name[0] == '.') continue;
            char full[PATH_MAX];
            snprintf(full, sizeof(full), "%s/%s", g.cwd, entry->d_name);
            struct stat st;
            if (stat(full, &st) < 0) continue;
            if (flen + 80 > (int)sizeof(files_list)) break;
            if (S_ISDIR(st.st_mode))
                flen += snprintf(files_list + flen, sizeof(files_list) - flen, "  %s/\n", entry->d_name);
            else
                flen += snprintf(files_list + flen, sizeof(files_list) - flen, "  %s (%s)\n", entry->d_name, fmt_size(st.st_size));
        }
        closedir(dir);
    }

    // Check for PRE.md in CWD (project-specific instructions)
    char pre_md_path[PATH_MAX];
    snprintf(pre_md_path, sizeof(pre_md_path), "%s/PRE.md", g.cwd);
    char *pre_md_content = NULL;
    struct stat pre_md_st;
    if (stat(pre_md_path, &pre_md_st) == 0 && pre_md_st.st_size < 8192) {
        FILE *pmf = fopen(pre_md_path, "r");
        if (pmf) {
            pre_md_content = malloc((size_t)pre_md_st.st_size + 1);
            size_t nr = fread(pre_md_content, 1, (size_t)pre_md_st.st_size, pmf);
            pre_md_content[nr] = 0;
            fclose(pmf);
        }
    }

    // Get git branch and status if in a repo
    char git_info[2048] = "";
    int git_len = 0;
    {
        char git_dir[PATH_MAX];
        snprintf(git_dir, sizeof(git_dir), "%s/.git", g.cwd);
        struct stat gs;
        if (stat(git_dir, &gs) == 0) {
            // Get branch name
            FILE *p = popen("git -C \"$PWD\" branch --show-current 2>/dev/null", "r");
            if (p) {
                char branch[128] = "";
                if (fgets(branch, sizeof(branch), p)) {
                    branch[strcspn(branch, "\n")] = 0;
                    git_len += snprintf(git_info + git_len, sizeof(git_info) - git_len,
                        "Git branch: %s\n", branch);
                }
                pclose(p);
            }
            // Get short status
            p = popen("git -C \"$PWD\" status --short 2>/dev/null | head -20", "r");
            if (p) {
                char line[256];
                int status_lines = 0;
                while (fgets(line, sizeof(line), p) && status_lines < 20 &&
                       git_len < (int)sizeof(git_info) - 256) {
                    git_len += snprintf(git_info + git_len, sizeof(git_info) - git_len,
                        "  %s", line);
                    status_lines++;
                }
                pclose(p);
            }
        }
    }

    // Build memory context
    char *memory_ctx = build_memory_context();

    size_t cap = 32768 + flen + (pre_md_content ? strlen(pre_md_content) : 0) +
                 (memory_ctx ? strlen(memory_ctx) : 0) + git_len;
    char *preamble = malloc(cap);
    int plen = 0;

    // System identity and role
    plen += snprintf(preamble + plen, cap - plen,
        "You are PRE (Personal Reasoning Engine), a fully local agentic assistant running on Apple Silicon. "
        "All data stays on this machine. You have persistent memory across sessions.\n\n");

    // Project instructions
    if (pre_md_content) {
        plen += snprintf(preamble + plen, cap - plen,
            "<project_instructions>\n%s\n</project_instructions>\n\n", pre_md_content);
        free(pre_md_content);
    }

    // Memory context
    if (memory_ctx && g_memory_count > 0) {
        plen += snprintf(preamble + plen, cap - plen, "%s\n", memory_ctx);
    }
    free(memory_ctx);

    // Environment context
    plen += snprintf(preamble + plen, cap - plen,
        "<context>\n"
        "Working directory: %s\n", g.cwd);
    if (g.project_name[0]) {
        plen += snprintf(preamble + plen, cap - plen,
            "Project: %s (root: %s)\n"
            "Channel: #%s\n",
            g.project_name, g.project_root, g.channel);
    }
    if (git_len > 0) {
        plen += snprintf(preamble + plen, cap - plen, "%s", git_info);
    }
    plen += snprintf(preamble + plen, cap - plen,
        "Files:\n%s"
        "</context>\n\n", files_list);

    // Tool instructions
    plen += snprintf(preamble + plen, cap - plen,
        "You have tools to interact with the local system. To use a tool, output EXACTLY this format:\n\n"
        "<tool_call>\n"
        "{\"name\": \"TOOL_NAME\", \"arguments\": {\"KEY\": \"VALUE\"}}\n"
        "</tool_call>\n\n"
        "Available tools:\n"
        "1. bash - Run a shell command. Args: command\n"
        "2. read_file - Read a file. Args: path\n"
        "3. list_dir - List directory contents. Args: path\n"
        "4. glob - Find files by pattern. Args: pattern, path (optional)\n"
        "5. grep - Search file contents. Args: pattern, path (optional), include (optional glob)\n"
        "6. file_write - Write/create a file. Args: path, content\n"
        "7. file_edit - Edit a file (find & replace). Args: path, old_string, new_string\n"
        "8. web_fetch - Fetch a URL. Args: url\n"
        "9. system_info - Get system information. No args.\n"
        "10. process_list - List running processes. Args: filter (optional)\n"
        "11. process_kill - Kill a process. Args: pid\n"
        "12. clipboard_read - Read clipboard. No args.\n"
        "13. clipboard_write - Write to clipboard. Args: content\n"
        "14. open_app - Open file/app/URL with macOS 'open'. Args: target\n"
        "15. notify - Show macOS notification. Args: title, message\n"
        "16. memory_save - Save a persistent memory. Args: name, type (user/feedback/project/reference), description, content, scope (optional: 'project' or 'global', default global)\n"
        "17. memory_search - Search saved memories. Args: query (optional, omit to list all)\n"
        "18. memory_list - List all saved memories. No args.\n"
        "19. memory_delete - Delete a memory by name. Args: query\n"
        "20. screenshot - Capture screen. Args: region (optional: 'full', 'window', or 'x,y,w,h')\n"
        "21. window_list - List all open windows with positions. No args.\n"
        "22. window_focus - Bring an app to front. Args: app\n"
        "23. display_info - Display/GPU information. No args.\n"
        "24. net_info - Network interfaces, IPs, DNS. No args.\n"
        "25. net_connections - Active network connections. Args: filter (optional: 'listening', 'established', or port number)\n"
        "26. service_status - List or search launchd services. Args: service (optional)\n"
        "27. disk_usage - Disk/volume usage. Args: path (optional)\n"
        "28. hardware_info - Detailed hardware/thermal/battery info. No args.\n"
        "29. applescript - Run AppleScript for deep macOS automation. Args: script\n"
        "\n"
        "Example — edit a file:\n"
        "<tool_call>\n"
        "{\"name\": \"file_edit\", \"arguments\": {\"path\": \"src/main.py\", \"old_string\": \"def old_func():\\n    pass\", \"new_string\": \"def new_func():\\n    return 42\"}}\n"
        "</tool_call>\n\n"
        "Example — save a memory:\n"
        "<tool_call>\n"
        "{\"name\": \"memory_save\", \"arguments\": {\"name\": \"User prefers concise output\", \"type\": \"feedback\", \"description\": \"User wants terse responses\", \"content\": \"User prefers short, direct answers without trailing summaries.\"}}\n"
        "</tool_call>\n\n"
        "RULES:\n"
        "- ALWAYS include JSON with \"name\" and \"arguments\" inside <tool_call> tags.\n"
        "- NEVER output empty <tool_call></tool_call>.\n"
        "- After a tool call, STOP and wait for <tool_response>.\n"
        "- file_edit old_string must match exactly once in the file.\n"
        "- Paths are relative to working directory unless absolute.\n"
        "- Save memories proactively when you learn the user's preferences, project context, or workflow patterns.\n"
        "- Memory types: user (about the person), feedback (how to work), project (current work), reference (where info lives).\n"
        "- Use scope 'project' for project-specific memories, 'global' for cross-project knowledge.\n"
        "\n");

    return preamble;
}

// Build the full message: context preamble (first turn) + attachment + user input
static char *build_message(const char *input, int is_first_turn) {
    char *preamble = is_first_turn ? build_context_preamble() : NULL;
    size_t plen = preamble ? strlen(preamble) : 0;
    size_t alen = g_pending_attach ? strlen(g_pending_attach) : 0;
    size_t ilen = strlen(input);

    char *msg = malloc(plen + alen + ilen + 8);
    size_t off = 0;
    if (preamble) { memcpy(msg + off, preamble, plen); off += plen; free(preamble); }
    if (g_pending_attach) {
        memcpy(msg + off, g_pending_attach, alen); off += alen;
        msg[off++] = '\n'; msg[off++] = '\n';
        free(g_pending_attach);
        g_pending_attach = NULL;
    }
    memcpy(msg + off, input, ilen); off += ilen;
    msg[off] = 0;
    return msg;
}

// ============================================================================
// Context compaction — auto-summarize when approaching context limit
// ============================================================================

// Compact the session JSONL by replacing older turns with a summary.
// Keeps the last `keep_turns` user/assistant pairs intact.
// Returns 1 if compaction happened, 0 if not needed.
static int compact_session(const char *session_id, int keep_turns) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.jsonl", g.sessions_dir, session_id);

    FILE *f = fopen(path, "r");
    if (!f) return 0;

    // Read all lines
    char **lines = NULL;
    int line_count = 0, line_cap = 0;
    char buf[MAX_RESPONSE];
    while (fgets(buf, sizeof(buf), f)) {
        if (line_count >= line_cap) {
            line_cap = line_cap ? line_cap * 2 : 64;
            lines = realloc(lines, (size_t)line_cap * sizeof(char *));
        }
        lines[line_count++] = strdup(buf);
    }
    fclose(f);

    // Count user turns (each user+assistant pair = 1 turn)
    int user_count = 0;
    for (int i = 0; i < line_count; i++) {
        if (strstr(lines[i], "\"user\"")) user_count++;
    }

    if (user_count <= keep_turns + 1) {
        // Not enough turns to compact
        for (int i = 0; i < line_count; i++) free(lines[i]);
        free(lines);
        return 0;
    }

    // Find the split point: keep last `keep_turns` user messages and everything after
    int turns_from_end = 0;
    int split_idx = line_count;
    for (int i = line_count - 1; i >= 0; i--) {
        if (strstr(lines[i], "\"user\"")) {
            turns_from_end++;
            if (turns_from_end >= keep_turns) {
                split_idx = i;
                break;
            }
        }
    }

    // Build summary of older turns
    size_t summary_cap = 4096;
    char *summary = malloc(summary_cap);
    int summary_len = snprintf(summary, summary_cap,
        "[Previous conversation summary — %d turns compacted]\n", split_idx);

    for (int i = 0; i < split_idx && summary_len < (int)summary_cap - 512; i++) {
        // Extract role and abbreviated content
        char *role_p = strstr(lines[i], "\"role\":\"");
        char *content_p = strstr(lines[i], "\"content\":\"");
        if (!role_p || !content_p) continue;

        role_p += 8;
        char role[32]; int ri = 0;
        while (*role_p && *role_p != '"' && ri < 31) role[ri++] = *role_p++;
        role[ri] = 0;

        content_p += 11;
        // Get first 150 chars of content for summary
        char preview[160] = "";
        int pi = 0;
        for (int j = 0; content_p[j] && content_p[j] != '"' && pi < 150; j++) {
            if (content_p[j] == '\\' && content_p[j+1]) {
                j++;
                if (content_p[j] == 'n') preview[pi++] = ' ';
                else preview[pi++] = content_p[j];
            } else {
                preview[pi++] = content_p[j];
            }
        }
        preview[pi] = 0;

        // Skip tool responses in summary (too verbose)
        if (strcmp(role, "tool") == 0) continue;

        summary_len += snprintf(summary + summary_len, summary_cap - summary_len,
            "- %s: %s%s\n", role, preview, pi >= 150 ? "..." : "");
    }
    summary[summary_len] = 0;

    // Rewrite the session file: summary as first "system" message + kept turns
    f = fopen(path, "w");
    if (f) {
        char *escaped = json_escape_alloc(summary);
        if (escaped) {
            fprintf(f, "{\"role\":\"user\",\"content\":\"%s\"}\n", escaped);
            free(escaped);
        }
        // Write a synthetic assistant acknowledgment
        fprintf(f, "{\"role\":\"assistant\",\"content\":\"Understood, I have context from our previous %d turns.\"}\n",
                split_idx);

        for (int i = split_idx; i < line_count; i++) {
            fputs(lines[i], f);
        }
        fclose(f);
    }

    for (int i = 0; i < line_count; i++) free(lines[i]);
    free(lines);
    free(summary);

    printf(ANSI_DIM "  [compacted %d older turns to stay within context budget]" ANSI_RESET "\n",
           split_idx);
    return 1;
}

// Check if we should compact and do it
static void maybe_compact(void) {
    int estimated_tokens = g.total_tokens_in + g.total_tokens_out;
    // Compact when we've used 75% of context budget
    int threshold = MAX_CONTEXT * 3 / 4;
    if (estimated_tokens > threshold) {
        // Keep last 6 turns (12 messages) for immediate context
        compact_session(g.session_id, 6);
        // Reset token estimates after compaction
        g.total_tokens_in = estimated_tokens / 4; // rough estimate of compacted size
    }
}

// ============================================================================
// HTTP / SSE client
// ============================================================================

static int send_request(const char *user_message, int max_tokens, const char *session_id) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return -1; }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(g.port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "\n" ANSI_RED "  [error] Cannot connect to server on port %d." ANSI_RESET "\n", g.port);
        close(sock);
        return -1;
    }

    // Build full conversation history from session file
    // Each line in the JSONL is: {"role":"...","content":"..."}
    char session_path[1024];
    snprintf(session_path, sizeof(session_path), "%s/%s.jsonl", g.sessions_dir, session_id);

    // Start with generous capacity; grow if needed
    size_t body_cap = 1024 * 1024; // 1MB initial
    char *body = malloc(body_cap);
    int body_len = snprintf(body, body_cap, "{\"model\":\"%s\",\"messages\":[", g_model);

    // Replay session history
    FILE *sf = fopen(session_path, "r");
    if (sf) {
        char sline[MAX_RESPONSE];
        int first_msg = 1;
        while (fgets(sline, sizeof(sline), sf)) {
            // Each line is already a JSON object with role+content
            // Strip trailing newline
            int sl = (int)strlen(sline);
            while (sl > 0 && (sline[sl-1] == '\n' || sline[sl-1] == '\r')) sline[--sl] = 0;
            if (sl == 0) continue;

            // Ensure capacity
            if ((int)(body_len + sl + 10) > (int)body_cap - 100) {
                body_cap *= 2;
                body = realloc(body, body_cap);
            }

            if (!first_msg) body[body_len++] = ',';
            memcpy(body + body_len, sline, sl);
            body_len += sl;
            first_msg = 0;
        }
        fclose(sf);
    }

    // Append the new user message
    char *escaped = json_escape_alloc(user_message);
    if (!escaped) { free(body); close(sock); return -1; }
    size_t elen = strlen(escaped);

    // Ensure capacity for the new message
    if ((int)(body_len + elen + 256) > (int)body_cap - 100) {
        body_cap = body_len + elen + 4096;
        body = realloc(body, body_cap);
    }

    body_len += snprintf(body + body_len, body_cap - body_len,
        "%s{\"role\":\"user\",\"content\":\"%s\"}],\"max_tokens\":%d,\"stream\":true}",
        (body_len > 50 ? "," : ""), escaped, max_tokens);
    free(escaped);

    size_t req_cap = body_len + 256;
    char *request = malloc(req_cap);
    int req_len = snprintf(request, req_cap,
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        g.port, body_len, body);

    write(sock, request, req_len);
    free(body);
    free(request);
    return sock;
}

static int health_check(void) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(g.port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(sock);
        return 0;
    }
    close(sock);
    return 1;
}

// ============================================================================
// Streaming markdown renderer (from chat.m, unchanged)
// ============================================================================

typedef struct {
    int bold, italic, code_inline, code_block, skip_lang, line_start;
    int in_blockquote, in_table_row;
    char lang[32];
    int lang_idx;
} MdState;

static MdState g_md;

static void md_reset(void) {
    memset(&g_md, 0, sizeof(g_md));
    g_md.line_start = 1;
}

static void md_print(const char *text) {
    for (int i = 0; text[i]; i++) {
        char c = text[i];

        if (g_md.skip_lang) {
            if (c == '\n') {
                g_md.skip_lang = 0;
                g_md.lang[g_md.lang_idx] = 0;
                if (g_md.lang_idx > 0)
                    printf(ANSI_DIM " %s " ANSI_RESET "\n", g_md.lang);
                printf(ANSI_CODEBLK ANSI_CODEBLK_LINE "\n");
            } else if (g_md.lang_idx < 31) {
                g_md.lang[g_md.lang_idx++] = c;
            }
            continue;
        }

        if (c == '`' && text[i+1] == '`' && text[i+2] == '`') {
            if (g_md.code_block) { printf(ANSI_RESET "\n"); g_md.code_block = 0; }
            else { g_md.code_block = 1; g_md.skip_lang = 1; g_md.lang_idx = 0; }
            i += 2; continue;
        }

        if (g_md.code_block) {
            printf(ANSI_CODEBLK);
            if (c == '\n') printf(ANSI_CODEBLK_LINE "\n");
            else putchar(c);
            continue;
        }

        if (c == '`') {
            if (g_md.code_inline) { printf(ANSI_RESET); g_md.code_inline = 0; }
            else { printf(ANSI_CODE); g_md.code_inline = 1; }
            continue;
        }

        if (g_md.code_inline) { putchar(c); continue; }

        if (g_md.line_start && c == '#') {
            int level = 0;
            while (text[i + level] == '#') level++;
            i += level;
            while (text[i] == ' ') i++;
            // Header styling: h1 bold+underline, h2 bold, h3+ bold+dim
            if (level == 1) printf(ANSI_BOLD "\033[4m" ANSI_HEADER);
            else if (level == 2) printf(ANSI_BOLD ANSI_HEADER);
            else printf(ANSI_BOLD ANSI_CYAN);
            while (text[i] && text[i] != '\n') { putchar(text[i]); i++; }
            printf(ANSI_RESET);
            if (text[i] == '\n') { putchar('\n'); g_md.line_start = 1; }
            continue;
        }

        // Horizontal rule: --- or *** or ___ (3+ chars, nothing else on line)
        if (g_md.line_start && (c == '-' || c == '*' || c == '_')) {
            int peek = i;
            int count = 0;
            char ruler = c;
            while (text[peek] == ruler || text[peek] == ' ') {
                if (text[peek] == ruler) count++;
                peek++;
            }
            if (count >= 3 && (text[peek] == '\n' || text[peek] == '\0')) {
                printf(ANSI_DIM);
                for (int d = 0; d < 50; d++) printf("─");
                printf(ANSI_RESET "\n");
                i = peek;
                if (text[i] == '\n') g_md.line_start = 1;
                else g_md.line_start = 0;
                continue;
            }
        }

        // Blockquote: > text
        if (g_md.line_start && c == '>' && (text[i+1] == ' ' || text[i+1] == '\t')) {
            printf(ANSI_DIM "  │ " ANSI_RESET ANSI_ITALIC);
            g_md.in_blockquote = 1;
            i += 1; // skip '>', the space after is consumed naturally
            g_md.line_start = 0;
            continue;
        }

        // Table row: | col | col | col |
        if (g_md.line_start && c == '|') {
            // Check if this is a separator row (|---|---|)
            int peek = i + 1;
            int is_sep = 1;
            while (text[peek] && text[peek] != '\n') {
                if (text[peek] != '-' && text[peek] != '|' && text[peek] != ' '
                    && text[peek] != ':' && text[peek] != '+') {
                    is_sep = 0;
                    break;
                }
                peek++;
            }
            if (is_sep && peek > i + 2) {
                // Render separator as a dim line
                printf(ANSI_DIM);
                while (text[i] && text[i] != '\n') {
                    if (text[i] == '|') putchar('|');
                    else if (text[i] == '-' || text[i] == '+') printf("─");
                    else putchar(text[i]);
                    i++;
                }
                printf(ANSI_RESET);
                if (text[i] == '\n') { putchar('\n'); g_md.line_start = 1; }
                continue;
            }
            // Data row — colorize pipes as delimiters
            printf(ANSI_DIM "|" ANSI_RESET);
            g_md.in_table_row = 1;
            g_md.line_start = 0;
            continue;
        }

        if (g_md.line_start && (c == '-' || c == '*' || c == ' ')) {
            int indent = 0, peek = i;
            while (text[peek] == ' ' || text[peek] == '\t') { indent++; peek++; }
            char marker = text[peek];
            if ((marker == '-' || marker == '*') && marker != '\0') {
                char after = text[peek + 1];
                if (marker == '-' && (after == ' ' || after == '\0')) {
                    for (int d = 0; d < indent / 2 + 1; d++) printf("  ");
                    printf(ANSI_YELLOW "•" ANSI_RESET " ");
                    i = peek + 1;
                    while (text[i] == ' ' || text[i] == '\t') i++;
                    i--; g_md.line_start = 0; continue;
                }
                if (marker == '*' && after != '*' && (after == ' ' || after == '\0' || after == '\t')) {
                    for (int d = 0; d < indent / 2 + 1; d++) printf("  ");
                    printf(ANSI_YELLOW "•" ANSI_RESET " ");
                    i = peek + 1;
                    while (text[i] == ' ' || text[i] == '\t') i++;
                    i--; g_md.line_start = 0; continue;
                }
            }
        }

        if (g_md.line_start && c >= '0' && c <= '9') {
            int num_start = i;
            while (text[i] >= '0' && text[i] <= '9') i++;
            if (text[i] == '.' && text[i+1] == ' ') {
                printf("  " ANSI_YELLOW);
                for (int j = num_start; j <= i; j++) putchar(text[j]);
                printf(ANSI_RESET);
                i++; g_md.line_start = 0; continue;
            }
            i = num_start; c = text[i];
        }

        if (c == '*' && text[i+1] == '*') {
            if (g_md.bold) { printf(ANSI_RESET); g_md.bold = 0; }
            else { printf(ANSI_BOLD); g_md.bold = 1; }
            i++; continue;
        }

        if (c == '*' && text[i+1] != '*') {
            if (g_md.italic) { printf(ANSI_RESET); g_md.italic = 0; }
            else { printf(ANSI_ITALIC); g_md.italic = 1; }
            continue;
        }

        if (c == '\n') {
            if (g_md.in_blockquote) { printf(ANSI_RESET); g_md.in_blockquote = 0; }
            if (g_md.in_table_row) { printf(ANSI_DIM "|" ANSI_RESET); g_md.in_table_row = 0; }
            g_md.line_start = 1;
        } else {
            // Mid-line pipe in a table row
            if (c == '|' && g_md.in_table_row) {
                printf(ANSI_DIM "|" ANSI_RESET);
                continue;
            }
            g_md.line_start = 0;
        }

        putchar(c);
    }
}

// ============================================================================
// Stream response with spinner and stats
// ============================================================================

static char *stream_response(int sock) {
    FILE *stream = fdopen(sock, "r");
    if (!stream) { close(sock); return NULL; }

    int header_done = 0, in_think = 0, tokens = 0;
    double t_start = now_ms(), t_first = 0;
    md_reset();

    char *response = calloc(1, MAX_RESPONSE);
    int resp_len = 0;

    // Spinner during prefill
    static const char *spin[] = {"⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"};
    int spin_idx = 0;
    int spinning = 1;

    char line[65536];

    // Use select() for spinner animation during prefill wait
    int fd = fileno(stream);
    while (1) {
        if (spinning) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd, &fds);
            struct timeval tv = {0, 120000}; // 120ms timeout for spinner
            int ready = select(fd + 1, &fds, NULL, NULL, &tv);
            if (ready <= 0) {
                // Timeout — animate spinner
                printf("\r  " ANSI_DIM "[reasoning %s]" ANSI_RESET "  ", spin[spin_idx % 10]);
                fflush(stdout);
                spin_idx++;
                continue;
            }
        }

        if (!fgets(line, sizeof(line), stream)) break;

        if (!header_done) {
            if (strcmp(line, "\r\n") == 0 || strcmp(line, "\n") == 0) header_done = 1;
            continue;
        }
        if (strncmp(line, "data: ", 6) != 0) continue;
        if (strncmp(line + 6, "[DONE]", 6) == 0) break;

        // Decode a JSON string value starting at p into buf, return length
        // Handles \n \t \" \\ and \uXXXX (BMP codepoints, ASCII subset)
        #define DECODE_JSON_STR(p, buf, bufsz) ({ \
            int _di = 0; \
            for (int _i = 0; (p)[_i] && (p)[_i] != '"' && _di < (bufsz)-1; _i++) { \
                if ((p)[_i] == '\\' && (p)[_i+1]) { \
                    _i++; \
                    if ((p)[_i] == 'u' && (p)[_i+1] && (p)[_i+2] && (p)[_i+3] && (p)[_i+4]) { \
                        /* Parse 4 hex digits */ \
                        char _hex[5] = {(p)[_i+1],(p)[_i+2],(p)[_i+3],(p)[_i+4],0}; \
                        unsigned int _cp = (unsigned int)strtol(_hex, NULL, 16); \
                        _i += 4; \
                        if (_cp < 0x80) { \
                            (buf)[_di++] = (char)_cp; \
                        } else if (_cp < 0x800) { \
                            (buf)[_di++] = (char)(0xC0 | (_cp >> 6)); \
                            if (_di < (bufsz)-1) (buf)[_di++] = (char)(0x80 | (_cp & 0x3F)); \
                        } else { \
                            (buf)[_di++] = (char)(0xE0 | (_cp >> 12)); \
                            if (_di < (bufsz)-1) (buf)[_di++] = (char)(0x80 | ((_cp >> 6) & 0x3F)); \
                            if (_di < (bufsz)-1) (buf)[_di++] = (char)(0x80 | (_cp & 0x3F)); \
                        } \
                    } else { \
                        switch ((p)[_i]) { \
                            case 'n': (buf)[_di++]='\n'; break; \
                            case 't': (buf)[_di++]='\t'; break; \
                            case '"': (buf)[_di++]='"'; break; \
                            case '\\': (buf)[_di++]='\\'; break; \
                            default: (buf)[_di++]=(p)[_i]; break; \
                        } \
                    } \
                } else (buf)[_di++] = (p)[_i]; \
            } \
            (buf)[_di] = 0; \
            _di; \
        })

        // Check for reasoning field (Gemma 4 / Ollama thinking)
        char *rk = strstr(line + 6, "\"reasoning\":\"");
        if (rk) {
            rk += 13;
            char rdecoded[4096]; int rdi = DECODE_JSON_STR(rk, rdecoded, 4096);
            if (rdi > 0) {
                if (spinning) { printf("\r\033[K"); fflush(stdout); spinning = 0; }
                tokens++;
                if (!t_first) t_first = now_ms();
                if (g.show_thinking) { printf(ANSI_DIM "%s" ANSI_RESET, rdecoded); fflush(stdout); }
            }
        }

        // Check for content field
        char *ck = strstr(line + 6, "\"content\":\"");
        if (!ck) continue;
        ck += 11;

        char decoded[4096]; int di = DECODE_JSON_STR(ck, decoded, 4096);
        if (!di) continue;

        // Clear spinner on first real token
        if (spinning) {
            printf("\r\033[K");
            fflush(stdout);
            spinning = 0;
        }

        // Handle Qwen-style <think> tags in content (backwards compat)
        if (strstr(decoded, "<think>")) in_think = 1;
        if (strstr(decoded, "</think>")) { in_think = 0; tokens++; continue; }
        tokens++;
        if (!t_first) t_first = now_ms();

        if (!in_think && resp_len + di < MAX_RESPONSE - 1) {
            memcpy(response + resp_len, decoded, di);
            resp_len += di;
            response[resp_len] = 0;
        }

        if (in_think && !g.show_thinking) continue;
        if (in_think) printf(ANSI_DIM "%s" ANSI_RESET, decoded);
        else md_print(decoded);
        fflush(stdout);
    }
    fclose(stream);

    printf(ANSI_RESET);

    // Stats
    double t_end = now_ms();
    double gen_time = t_first > 0 ? t_end - t_first : 0;
    int gen_tokens = tokens > 1 ? tokens - 1 : tokens;
    double tok_s = (gen_tokens > 0 && gen_time > 0) ? gen_tokens * 1000.0 / gen_time : 0;
    double ttft = t_first > 0 ? t_first - t_start : t_end - t_start;

    g.last_token_count = tokens;
    g.last_tok_s = tok_s;
    g.last_ttft_ms = ttft;
    g.total_tokens_out += tokens;
    g.cumulative_gen_ms += gen_time;

    printf("\n\n");
    if (tokens > 0) {
        printf(ANSI_DIM "  [%d tokens, %.1f tok/s, TTFT %.1fs]" ANSI_RESET "\n\n",
               tokens, tok_s, ttft / 1000.0);
    }

    return response;
}

// ============================================================================
// Tool call handling
// ============================================================================

// Decode a JSON string starting at p (just after the opening quote) into a malloc'd buffer.
// Returns the decoded string (caller must free) and advances *endp past the closing quote.
// Handles \n, \t, \", \\, \uXXXX (UTF-8 output).
static char *decode_json_string(const char *p, const char **endp) {
    size_t cap = 256;
    char *buf = malloc(cap);
    size_t di = 0;

    for (int i = 0; p[i] && p[i] != '"'; i++) {
        if (di + 8 > cap) { cap *= 2; buf = realloc(buf, cap); }
        if (p[i] == '\\' && p[i+1]) {
            i++;
            if (p[i] == 'u' && p[i+1] && p[i+2] && p[i+3] && p[i+4]) {
                char hex[5] = {p[i+1], p[i+2], p[i+3], p[i+4], 0};
                unsigned int cp = (unsigned int)strtol(hex, NULL, 16);
                i += 4;
                if (cp < 0x80) {
                    buf[di++] = (char)cp;
                } else if (cp < 0x800) {
                    buf[di++] = (char)(0xC0 | (cp >> 6));
                    buf[di++] = (char)(0x80 | (cp & 0x3F));
                } else {
                    buf[di++] = (char)(0xE0 | (cp >> 12));
                    buf[di++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    buf[di++] = (char)(0x80 | (cp & 0x3F));
                }
            } else {
                switch (p[i]) {
                    case 'n': buf[di++] = '\n'; break;
                    case 't': buf[di++] = '\t'; break;
                    case 'r': buf[di++] = '\r'; break;
                    case '"': buf[di++] = '"'; break;
                    case '\\': buf[di++] = '\\'; break;
                    case '/': buf[di++] = '/'; break;
                    default: buf[di++] = p[i]; break;
                }
            }
        } else {
            buf[di++] = p[i];
        }
    }
    buf[di] = 0;

    // Advance endp past the closing quote
    if (endp) {
        int i = 0;
        while (p[i] && p[i] != '"') {
            if (p[i] == '\\' && p[i+1]) i += 2;
            else i++;
        }
        *endp = p[i] == '"' ? p + i + 1 : p + i;
    }
    return buf;
}

// Extract tool call with full multi-argument support.
// Fills ToolCall struct. Returns 1 on success.
static int extract_tool_call_v2(const char *tc_body, ToolCall *tc) {
    memset(tc, 0, sizeof(*tc));

    // === Try JSON format: {"name":"...","arguments":{...}} ===
    char *nk = strstr(tc_body, "\"name\"");
    if (nk) {
        // Find the opening quote of the name value
        nk += 6;
        while (*nk && *nk != '"') nk++;
        if (*nk == '"') {
            nk++;
            int ni = 0;
            while (*nk && *nk != '"' && ni < 63) tc->name[ni++] = *nk++;
            tc->name[ni] = 0;
        }
    }

    // Find "arguments" then the opening {
    char *args_start = strstr(tc_body, "\"arguments\"");
    if (args_start) {
        args_start += 11;
        while (*args_start && *args_start != '{') args_start++;
        if (*args_start == '{') {
            args_start++; // skip {
            // Parse key-value pairs
            const char *p = args_start;
            while (*p && tc->argc < MAX_TOOL_ARGS) {
                // Skip whitespace and commas
                while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',')) p++;
                if (*p == '}') break; // end of arguments
                if (*p != '"') break; // expect key string

                // Parse key
                p++; // skip opening "
                int ki = 0;
                while (*p && *p != '"' && ki < 63) {
                    tc->keys[tc->argc][ki++] = *p++;
                }
                tc->keys[tc->argc][ki] = 0;
                if (*p == '"') p++; // skip closing "

                // Skip colon and whitespace
                while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;

                if (*p == '"') {
                    // String value
                    p++; // skip opening "
                    const char *endp;
                    tc->vals[tc->argc] = decode_json_string(p, &endp);
                    p = endp;
                } else {
                    // Non-string value (number, bool, null) — read until , or }
                    const char *start = p;
                    while (*p && *p != ',' && *p != '}') p++;
                    size_t vlen = (size_t)(p - start);
                    tc->vals[tc->argc] = malloc(vlen + 1);
                    memcpy(tc->vals[tc->argc], start, vlen);
                    tc->vals[tc->argc][vlen] = 0;
                    // Trim trailing whitespace
                    while (vlen > 0 && (tc->vals[tc->argc][vlen-1] == ' ' ||
                           tc->vals[tc->argc][vlen-1] == '\n')) {
                        tc->vals[tc->argc][--vlen] = 0;
                    }
                }
                tc->argc++;
            }
            if (tc->argc > 0) return 1;
        }
    }

    // === Fallback: Qwen XML-style format ===
    if (tc->name[0] == 0) {
        const char *p = tc_body;
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r')) p++;
        if (*p && *p != '{' && *p != '<') {
            int ni = 0;
            while (*p && *p != '\n' && *p != '\r' && *p != '<' && *p != ' ' && ni < 63)
                tc->name[ni++] = *p++;
            tc->name[ni] = 0;
        }
    }

    char *ak = strstr(tc_body, "<arg_key>");
    char *av = strstr(tc_body, "<arg_value>");
    if (ak && av) {
        ak += 9;
        char *ak_end = strstr(ak, "</arg_key>");
        if (ak_end) {
            int kl = (int)(ak_end - ak);
            if (kl > 63) kl = 63;
            memcpy(tc->keys[0], ak, kl);
            tc->keys[0][kl] = 0;
        }
        av += 11;
        char *av_end = strstr(av, "</arg_value>");
        if (!av_end) av_end = strstr(av, "</function>");
        if (!av_end) av_end = strstr(av, "</tool_call>");
        if (av_end) {
            int vl = (int)(av_end - av);
            tc->vals[0] = malloc(vl + 1);
            memcpy(tc->vals[0], av, vl);
            tc->vals[0][vl] = 0;
        }
        if (tc->vals[0]) { tc->argc = 1; return 1; }
    }

    // Bare argument on the line after the tool name
    if (tc->name[0] != 0 && tc->argc == 0) {
        const char *p = tc_body;
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r')) p++;
        while (*p && *p != '\n') p++;
        while (*p == '\n' || *p == '\r') p++;
        if (*p && *p != '<') {
            const char *end = p + strlen(p);
            while (end > p && (end[-1] == '\n' || end[-1] == '\r' || end[-1] == ' ')) end--;
            int vl = (int)(end - p);
            if (vl > 0) {
                // Default key based on tool name
                if (strcmp(tc->name, "bash") == 0)
                    strncpy(tc->keys[0], "command", 63);
                else
                    strncpy(tc->keys[0], "path", 63);
                tc->vals[0] = malloc(vl + 1);
                memcpy(tc->vals[0], p, vl);
                tc->vals[0][vl] = 0;
                tc->argc = 1;
                return 1;
            }
        }
    }

    if (tc->name[0] == 0 && strstr(tc_body, "bash")) strncpy(tc->name, "bash", 64);

    // Default path for list_dir/read_file when no argument given
    if (tc->argc == 0 && (strcmp(tc->name, "list_dir") == 0 || strcmp(tc->name, "read_file") == 0)) {
        strncpy(tc->keys[0], "path", 63);
        tc->vals[0] = strdup(".");
        tc->argc = 1;
    }

    return tc->argc > 0;
}

// Max bytes per tool response injected into context (keeps prefill fast)
#define MAX_TOOL_RESPONSE (8 * 1024)

// Execute a single tool call and write output into buf. Returns output length.
static int execute_tool(ToolCall *tc, char *output, size_t output_sz) {
    int out_len = 0;
    output[0] = 0;
    const char *name = tc->name;

    // --- Permission check ---
    PermLevel perm = tool_permission(name);

    // Special case: open_app is confirm first time only
    if (strcmp(name, "open_app") == 0 && g.open_app_approved) perm = PERM_AUTO;

    if (perm == PERM_CONFIRM_ALWAYS) {
        const char *detail = tool_call_get(tc, "command");
        if (!detail) detail = tool_call_get(tc, "pid");
        if (!detail) detail = tool_call_get(tc, "target");
        if (!detail) detail = name;
        printf(ANSI_YELLOW "  [%s: %s]" ANSI_RESET "\n", name, detail);
        printf(ANSI_DIM "  [execute? y/n] " ANSI_RESET);
        fflush(stdout);
        int ch = getchar(); while (getchar() != '\n');
        if (ch != 'y' && ch != 'Y') {
            printf(ANSI_DIM "  [skipped]" ANSI_RESET "\n");
            return -1;
        }
        if (strcmp(name, "open_app") == 0) g.open_app_approved = 1;
    } else if (perm == PERM_CONFIRM_ONCE && !g.auto_approve_tools) {
        const char *detail = tool_call_get(tc, "path");
        if (!detail) detail = tool_call_get(tc, "url");
        if (!detail) detail = tool_call_get(tc, "title");
        if (!detail) detail = name;
        printf(ANSI_YELLOW "  [%s: %s]" ANSI_RESET "\n", name, detail);
        printf(ANSI_DIM "  [allow? y/n/a(lways)] " ANSI_RESET);
        fflush(stdout);
        int ch = getchar(); while (getchar() != '\n');
        if (ch == 'a' || ch == 'A') { g.auto_approve_tools = 1; }
        else if (ch != 'y' && ch != 'Y') {
            printf(ANSI_DIM "  [skipped]" ANSI_RESET "\n");
            return -1;
        }
    }

    // --- Dispatch tools ---

    if (strcmp(name, "read_file") == 0) {
        const char *path_arg = tool_call_get(tc, "path");
        if (!path_arg) path_arg = ".";
        char resolved[PATH_MAX];
        resolve_path(path_arg, resolved, sizeof(resolved));
        printf(ANSI_DIM "  [reading %s]" ANSI_RESET "\n", resolved);
        char *content = file_read_for_context(resolved);
        if (content) {
            int cl = (int)strlen(content);
            int limit = MAX_TOOL_RESPONSE < (int)output_sz - 1 ? MAX_TOOL_RESPONSE : (int)output_sz - 1;
            if (cl > limit) {
                memcpy(output, content, limit);
                output[limit] = 0;
                out_len = limit + snprintf(output + limit, output_sz - limit,
                    "\n\n[... truncated %d/%d bytes — use bash to see full file]", limit, cl);
                printf(ANSI_DIM "  [truncated to %dKB of %dKB]" ANSI_RESET "\n",
                    limit / 1024, cl / 1024);
            } else {
                memcpy(output, content, cl);
                out_len = cl;
                output[cl] = 0;
            }
            free(content);
        } else {
            out_len = snprintf(output, output_sz, "Error: cannot read file '%s'", path_arg);
        }

    } else if (strcmp(name, "list_dir") == 0) {
        const char *path_arg = tool_call_get(tc, "path");
        if (!path_arg) path_arg = ".";
        char resolved[PATH_MAX];
        resolve_path(path_arg, resolved, sizeof(resolved));
        printf(ANSI_DIM "  [listing %s]" ANSI_RESET "\n", resolved);
        char *listing = dir_listing(resolved);
        if (listing) {
            int ll = (int)strlen(listing);
            if (ll > (int)output_sz - 1) ll = (int)output_sz - 1;
            memcpy(output, listing, ll);
            out_len = ll;
            output[ll] = 0;
            free(listing);
        } else {
            out_len = snprintf(output, output_sz, "Error: cannot list directory '%s'", path_arg);
        }

    } else if (strcmp(name, "bash") == 0) {
        const char *cmd = tool_call_get(tc, "command");
        if (!cmd) { out_len = snprintf(output, output_sz, "Error: no command provided"); return out_len; }
        printf(ANSI_DIM "  [$ %s]" ANSI_RESET "\n", cmd);
        FILE *proc = popen(cmd, "r");
        if (proc) {
            while (out_len < (int)output_sz - 1) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }
        if (out_len > 0) {
            // Truncate display if very long
            if (out_len > 2000) {
                printf(ANSI_DIM "%.2000s\n  [...%d more bytes]" ANSI_RESET "\n", output, out_len - 2000);
            } else {
                printf(ANSI_DIM "%s" ANSI_RESET, output);
                if (output[out_len - 1] != '\n') printf("\n");
            }
        }

    } else if (strcmp(name, "system_info") == 0) {
        printf(ANSI_DIM "  [gathering system info]" ANSI_RESET "\n");
        out_len = 0;
        // CPU
        FILE *p = popen("sysctl -n machdep.cpu.brand_string 2>/dev/null", "r");
        if (p) {
            out_len += snprintf(output + out_len, output_sz - out_len, "CPU: ");
            char buf[256];
            if (fgets(buf, sizeof(buf), p)) {
                buf[strcspn(buf, "\n")] = 0;
                out_len += snprintf(output + out_len, output_sz - out_len, "%s\n", buf);
            }
            pclose(p);
        }
        // Memory
        p = popen("sysctl -n hw.memsize 2>/dev/null", "r");
        if (p) {
            char buf[64];
            if (fgets(buf, sizeof(buf), p)) {
                long long mem = atoll(buf);
                out_len += snprintf(output + out_len, output_sz - out_len,
                    "Memory: %.1f GB\n", mem / (1024.0*1024*1024));
            }
            pclose(p);
        }
        // VM stats
        p = popen("vm_stat 2>/dev/null | head -5", "r");
        if (p) {
            out_len += snprintf(output + out_len, output_sz - out_len, "VM Stats:\n");
            char buf[256];
            while (fgets(buf, sizeof(buf), p) && out_len < (int)output_sz - 256)
                out_len += snprintf(output + out_len, output_sz - out_len, "  %s", buf);
            pclose(p);
        }
        // Disk
        p = popen("df -h / 2>/dev/null", "r");
        if (p) {
            out_len += snprintf(output + out_len, output_sz - out_len, "Disk:\n");
            char buf[256];
            while (fgets(buf, sizeof(buf), p) && out_len < (int)output_sz - 256)
                out_len += snprintf(output + out_len, output_sz - out_len, "  %s", buf);
            pclose(p);
        }
        // Battery
        p = popen("pmset -g batt 2>/dev/null | tail -1", "r");
        if (p) {
            char buf[256];
            if (fgets(buf, sizeof(buf), p)) {
                buf[strcspn(buf, "\n")] = 0;
                out_len += snprintf(output + out_len, output_sz - out_len, "Battery: %s\n", buf);
            }
            pclose(p);
        }

    } else if (strcmp(name, "process_list") == 0) {
        const char *filter = tool_call_get(tc, "filter");
        printf(ANSI_DIM "  [listing processes%s%s]" ANSI_RESET "\n",
               filter ? " filter=" : "", filter ? filter : "");
        char cmd[512];
        if (filter && filter[0])
            snprintf(cmd, sizeof(cmd), "ps aux | head -1; ps aux | grep '%s' | grep -v grep", filter);
        else
            snprintf(cmd, sizeof(cmd), "ps aux");
        FILE *proc = popen(cmd, "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }

    } else if (strcmp(name, "process_kill") == 0) {
        const char *pid_str = tool_call_get(tc, "pid");
        if (!pid_str) { out_len = snprintf(output, output_sz, "Error: no pid provided"); return out_len; }
        // Validate numeric
        for (int i = 0; pid_str[i]; i++) {
            if (pid_str[i] < '0' || pid_str[i] > '9') {
                out_len = snprintf(output, output_sz, "Error: invalid pid '%s'", pid_str);
                return out_len;
            }
        }
        pid_t pid = (pid_t)atoi(pid_str);
        printf(ANSI_DIM "  [killing pid %d]" ANSI_RESET "\n", pid);
        if (kill(pid, SIGTERM) == 0) {
            out_len = snprintf(output, output_sz, "Sent SIGTERM to pid %d", pid);
        } else {
            out_len = snprintf(output, output_sz, "Error killing pid %d: %s", pid, strerror(errno));
        }

    } else if (strcmp(name, "clipboard_read") == 0) {
        printf(ANSI_DIM "  [reading clipboard]" ANSI_RESET "\n");
        FILE *proc = popen("pbpaste", "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        } else {
            out_len = snprintf(output, output_sz, "Error: cannot read clipboard");
        }

    } else if (strcmp(name, "clipboard_write") == 0) {
        const char *content = tool_call_get(tc, "content");
        if (!content) { out_len = snprintf(output, output_sz, "Error: no content provided"); return out_len; }
        printf(ANSI_DIM "  [writing to clipboard]" ANSI_RESET "\n");
        FILE *proc = popen("pbcopy", "w");
        if (proc) {
            fwrite(content, 1, strlen(content), proc);
            pclose(proc);
            out_len = snprintf(output, output_sz, "Copied %zu bytes to clipboard", strlen(content));
        } else {
            out_len = snprintf(output, output_sz, "Error: cannot write to clipboard");
        }

    } else if (strcmp(name, "open_app") == 0) {
        const char *target = tool_call_get(tc, "target");
        if (!target) { out_len = snprintf(output, output_sz, "Error: no target provided"); return out_len; }
        printf(ANSI_DIM "  [opening %s]" ANSI_RESET "\n", target);
        char cmd[PATH_MAX + 32];
        snprintf(cmd, sizeof(cmd), "open '%s' 2>&1", target);
        FILE *proc = popen(cmd, "r");
        if (proc) {
            while (out_len < (int)output_sz - 1) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            int ret = pclose(proc);
            if (ret == 0 && out_len == 0)
                out_len = snprintf(output, output_sz, "Opened %s", target);
        }

    } else if (strcmp(name, "notify") == 0) {
        const char *title = tool_call_get(tc, "title");
        const char *message = tool_call_get(tc, "message");
        if (!title || !message) {
            out_len = snprintf(output, output_sz, "Error: title and message required");
            return out_len;
        }
        printf(ANSI_DIM "  [notify: %s]" ANSI_RESET "\n", title);
        // Shell-escape by replacing single quotes
        char safe_title[256], safe_msg[1024];
        int ti = 0, mi = 0;
        for (int i = 0; title[i] && ti < 250; i++) {
            if (title[i] == '\'') { safe_title[ti++] = '\\'; safe_title[ti++] = '\''; }
            else safe_title[ti++] = title[i];
        }
        safe_title[ti] = 0;
        for (int i = 0; message[i] && mi < 1018; i++) {
            if (message[i] == '\'') { safe_msg[mi++] = '\\'; safe_msg[mi++] = '\''; }
            else safe_msg[mi++] = message[i];
        }
        safe_msg[mi] = 0;
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "osascript -e 'display notification \"%s\" with title \"%s\"' 2>&1",
            safe_msg, safe_title);
        FILE *proc = popen(cmd, "r");
        if (proc) { pclose(proc); }
        out_len = snprintf(output, output_sz, "Notification sent: %s", title);

    } else if (strcmp(name, "glob") == 0) {
        const char *pattern = tool_call_get(tc, "pattern");
        const char *path_arg = tool_call_get(tc, "path");
        if (!pattern) { out_len = snprintf(output, output_sz, "Error: no pattern provided"); return out_len; }

        char full_pattern[PATH_MAX];
        if (path_arg && path_arg[0]) {
            char resolved[PATH_MAX];
            resolve_path(path_arg, resolved, sizeof(resolved));
            snprintf(full_pattern, sizeof(full_pattern), "%s/%s", resolved, pattern);
        } else {
            // If pattern is already absolute, use as-is
            if (pattern[0] == '/' || pattern[0] == '~') {
                strncpy(full_pattern, pattern, sizeof(full_pattern) - 1);
            } else {
                snprintf(full_pattern, sizeof(full_pattern), "%s/%s", g.cwd, pattern);
            }
        }
        printf(ANSI_DIM "  [glob %s]" ANSI_RESET "\n", full_pattern);

        glob_t gl;
        int ret = glob(full_pattern, GLOB_TILDE | GLOB_MARK, NULL, &gl);
        if (ret == 0) {
            for (size_t i = 0; i < gl.gl_pathc && out_len < (int)output_sz - PATH_MAX; i++) {
                out_len += snprintf(output + out_len, output_sz - out_len, "%s\n", gl.gl_pathv[i]);
            }
            if (gl.gl_pathc == 0)
                out_len = snprintf(output, output_sz, "No matches found");
            globfree(&gl);
        } else {
            out_len = snprintf(output, output_sz, "No matches found for pattern '%s'", full_pattern);
        }

    } else if (strcmp(name, "grep") == 0) {
        const char *pattern = tool_call_get(tc, "pattern");
        const char *path_arg = tool_call_get(tc, "path");
        const char *include = tool_call_get(tc, "include");
        if (!pattern) { out_len = snprintf(output, output_sz, "Error: no pattern provided"); return out_len; }

        char resolved[PATH_MAX];
        if (path_arg && path_arg[0])
            resolve_path(path_arg, resolved, sizeof(resolved));
        else
            strncpy(resolved, g.cwd, sizeof(resolved) - 1);

        printf(ANSI_DIM "  [grep '%s' in %s]" ANSI_RESET "\n", pattern, resolved);

        // Build grep command — shell-escape pattern by replacing single quotes
        char safe_pattern[1024];
        int pi = 0;
        for (int i = 0; pattern[i] && pi < 1018; i++) {
            if (pattern[i] == '\'') { safe_pattern[pi++] = '\''; safe_pattern[pi++] = '\\';
                                      safe_pattern[pi++] = '\''; safe_pattern[pi++] = '\''; }
            else safe_pattern[pi++] = pattern[i];
        }
        safe_pattern[pi] = 0;

        char cmd[2048];
        if (include && include[0])
            snprintf(cmd, sizeof(cmd), "grep -rn --include='%s' '%s' '%s' 2>/dev/null", include, safe_pattern, resolved);
        else
            snprintf(cmd, sizeof(cmd), "grep -rn '%s' '%s' 2>/dev/null", safe_pattern, resolved);

        FILE *proc = popen(cmd, "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }
        if (out_len == 0)
            out_len = snprintf(output, output_sz, "No matches found");

    } else if (strcmp(name, "web_fetch") == 0) {
        const char *url = tool_call_get(tc, "url");
        if (!url) { out_len = snprintf(output, output_sz, "Error: no url provided"); return out_len; }
        // Validate URL and reject command injection
        if (strncmp(url, "http://", 7) != 0 && strncmp(url, "https://", 8) != 0) {
            out_len = snprintf(output, output_sz, "Error: URL must start with http:// or https://");
            return out_len;
        }
        if (strchr(url, '\'') || strchr(url, '`') || strchr(url, '$') || strchr(url, ';')) {
            out_len = snprintf(output, output_sz, "Error: URL contains invalid characters");
            return out_len;
        }
        printf(ANSI_DIM "  [fetching %s]" ANSI_RESET "\n", url);
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "curl -sL --max-time 15 '%s' | textutil -stdin -format html -convert txt -stdout 2>/dev/null || curl -sL --max-time 15 '%s'",
            url, url);
        FILE *proc = popen(cmd, "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }
        if (out_len == 0)
            out_len = snprintf(output, output_sz, "Error: no content fetched from %s", url);

    } else if (strcmp(name, "file_write") == 0) {
        const char *path_arg = tool_call_get(tc, "path");
        const char *content = tool_call_get(tc, "content");
        if (!path_arg || !content) {
            out_len = snprintf(output, output_sz, "Error: path and content required");
            return out_len;
        }
        char resolved[PATH_MAX];
        resolve_path(path_arg, resolved, sizeof(resolved));
        printf(ANSI_DIM "  [writing %s]" ANSI_RESET "\n", resolved);
        checkpoint_file(resolved);
        FILE *f = fopen(resolved, "w");
        if (f) {
            size_t written = fwrite(content, 1, strlen(content), f);
            fclose(f);
            out_len = snprintf(output, output_sz, "Wrote %zu bytes to %s", written, resolved);
        } else {
            out_len = snprintf(output, output_sz, "Error: cannot write '%s': %s", resolved, strerror(errno));
        }

    } else if (strcmp(name, "file_edit") == 0) {
        const char *path_arg = tool_call_get(tc, "path");
        const char *old_str = tool_call_get(tc, "old_string");
        const char *new_str = tool_call_get(tc, "new_string");
        if (!path_arg || !old_str || !new_str) {
            out_len = snprintf(output, output_sz, "Error: path, old_string, and new_string required");
            return out_len;
        }
        char resolved[PATH_MAX];
        resolve_path(path_arg, resolved, sizeof(resolved));
        printf(ANSI_DIM "  [editing %s]" ANSI_RESET "\n", resolved);

        // Read file
        struct stat st;
        if (stat(resolved, &st) < 0) {
            out_len = snprintf(output, output_sz, "Error: file not found '%s'", resolved);
            return out_len;
        }
        FILE *f = fopen(resolved, "r");
        if (!f) {
            out_len = snprintf(output, output_sz, "Error: cannot read '%s'", resolved);
            return out_len;
        }
        char *file_content = malloc((size_t)st.st_size + 1);
        size_t nread = fread(file_content, 1, (size_t)st.st_size, f);
        file_content[nread] = 0;
        fclose(f);

        // Find old_string — must appear exactly once
        char *first = strstr(file_content, old_str);
        if (!first) {
            free(file_content);
            out_len = snprintf(output, output_sz, "Error: old_string not found in %s", resolved);
            return out_len;
        }
        char *second = strstr(first + strlen(old_str), old_str);
        if (second) {
            free(file_content);
            out_len = snprintf(output, output_sz, "Error: old_string appears multiple times in %s", resolved);
            return out_len;
        }

        // Checkpoint and replace
        checkpoint_file(resolved);

        size_t old_len = strlen(old_str);
        size_t new_len = strlen(new_str);
        size_t prefix_len = (size_t)(first - file_content);
        size_t suffix_len = nread - prefix_len - old_len;
        size_t result_len = prefix_len + new_len + suffix_len;
        char *result = malloc(result_len + 1);
        memcpy(result, file_content, prefix_len);
        memcpy(result + prefix_len, new_str, new_len);
        memcpy(result + prefix_len + new_len, first + old_len, suffix_len);
        result[result_len] = 0;
        free(file_content);

        f = fopen(resolved, "w");
        if (f) {
            fwrite(result, 1, result_len, f);
            fclose(f);
            out_len = snprintf(output, output_sz, "Edited %s: replaced %zu bytes with %zu bytes",
                             resolved, old_len, new_len);
        } else {
            out_len = snprintf(output, output_sz, "Error: cannot write '%s'", resolved);
        }
        free(result);

    } else if (strcmp(name, "memory_save") == 0) {
        const char *mem_name = tool_call_get(tc, "name");
        const char *mem_type = tool_call_get(tc, "type");
        const char *mem_desc = tool_call_get(tc, "description");
        const char *mem_content = tool_call_get(tc, "content");
        const char *mem_scope = tool_call_get(tc, "scope"); // "project" or "global" (default)
        if (!mem_name || !mem_type || !mem_content) {
            out_len = snprintf(output, output_sz, "Error: name, type, and content required");
            return out_len;
        }
        if (!mem_desc) mem_desc = mem_name;
        int is_proj = (mem_scope && strcmp(mem_scope, "project") == 0);
        printf(ANSI_DIM "  [saving %s memory: %s (%s)]" ANSI_RESET "\n",
               is_proj ? "project" : "global", mem_name, mem_type);
        if (save_memory(mem_name, mem_type, mem_desc, mem_content, mem_scope)) {
            out_len = snprintf(output, output_sz, "Memory saved: %s [%s] (%s)",
                             mem_name, mem_type, is_proj ? "project" : "global");
        } else {
            out_len = snprintf(output, output_sz, "Error: failed to save memory");
        }

    } else if (strcmp(name, "memory_search") == 0) {
        const char *query = tool_call_get(tc, "query");
        printf(ANSI_DIM "  [searching memories%s%s]" ANSI_RESET "\n",
               query ? ": " : "", query ? query : "");
        char *result = search_memories(query);
        if (result) {
            int rl = (int)strlen(result);
            if (rl > (int)output_sz - 1) rl = (int)output_sz - 1;
            memcpy(output, result, rl);
            output[rl] = 0;
            out_len = rl;
            free(result);
        }

    } else if (strcmp(name, "memory_list") == 0) {
        printf(ANSI_DIM "  [listing %d memories]" ANSI_RESET "\n", g_memory_count);
        for (int i = 0; i < g_memory_count && out_len < (int)output_sz - 512; i++) {
            out_len += snprintf(output + out_len, output_sz - out_len,
                "%d. [%s] %s — %s\n",
                i + 1, g_memories[i].type, g_memories[i].name, g_memories[i].description);
        }
        if (g_memory_count == 0)
            out_len = snprintf(output, output_sz, "No memories saved yet.");

    } else if (strcmp(name, "memory_delete") == 0) {
        const char *query = tool_call_get(tc, "query");
        if (!query) { out_len = snprintf(output, output_sz, "Error: query required"); return out_len; }
        printf(ANSI_DIM "  [deleting memory matching: %s]" ANSI_RESET "\n", query);
        if (delete_memory(query)) {
            out_len = snprintf(output, output_sz, "Deleted memory matching '%s'", query);
        } else {
            out_len = snprintf(output, output_sz, "No memory found matching '%s'", query);
        }

    // === Priority 3: Deep System Access ===

    } else if (strcmp(name, "screenshot") == 0) {
        const char *region = tool_call_get(tc, "region"); // "full", "window", or x,y,w,h
        printf(ANSI_DIM "  [capturing screenshot%s%s]" ANSI_RESET "\n",
               region ? ": " : "", region ? region : "");

        // Generate temp path
        char tmp_path[PATH_MAX];
        snprintf(tmp_path, sizeof(tmp_path), "/tmp/pre_screenshot_%d.png", (int)getpid());

        char cmd[512];
        if (region && strcmp(region, "window") == 0) {
            snprintf(cmd, sizeof(cmd), "screencapture -w '%s' 2>&1", tmp_path);
        } else if (region && strcmp(region, "full") != 0 && region[0] >= '0' && region[0] <= '9') {
            // Parse x,y,w,h
            int x = 0, y = 0, w = 0, h = 0;
            sscanf(region, "%d,%d,%d,%d", &x, &y, &w, &h);
            snprintf(cmd, sizeof(cmd), "screencapture -R%d,%d,%d,%d '%s' 2>&1", x, y, w, h, tmp_path);
        } else {
            snprintf(cmd, sizeof(cmd), "screencapture -x '%s' 2>&1", tmp_path);
        }
        int ret = system(cmd);

        struct stat ss;
        if (ret == 0 && stat(tmp_path, &ss) == 0 && ss.st_size > 0) {
            // Base64 encode for model (Gemma 4 vision)
            char b64_cmd[PATH_MAX + 64];
            snprintf(b64_cmd, sizeof(b64_cmd), "base64 -i '%s' 2>/dev/null", tmp_path);
            FILE *proc = popen(b64_cmd, "r");
            if (proc) {
                while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                    int ch = fgetc(proc);
                    if (ch == EOF) break;
                    output[out_len++] = (char)ch;
                }
                output[out_len] = 0;
                pclose(proc);
            }
            // Prepend metadata
            char meta[256];
            int ml = snprintf(meta, sizeof(meta),
                "Screenshot saved: %s (%s)\nBase64 PNG data (%s):\n",
                tmp_path, fmt_size(ss.st_size), fmt_size(ss.st_size));
            if (out_len + ml < (int)output_sz) {
                memmove(output + ml, output, out_len + 1);
                memcpy(output, meta, ml);
                out_len += ml;
            }
        } else {
            out_len = snprintf(output, output_sz,
                "Screenshot saved to %s (use bash to inspect). Note: screencapture may need Screen Recording permission.",
                tmp_path);
        }

    } else if (strcmp(name, "window_list") == 0) {
        printf(ANSI_DIM "  [listing windows]" ANSI_RESET "\n");
        const char *script =
            "osascript -e '"
            "set output to \"\"\n"
            "tell application \"System Events\"\n"
            "  repeat with proc in (every process whose background only is false)\n"
            "    set appName to name of proc\n"
            "    try\n"
            "      repeat with w in (every window of proc)\n"
            "        set winName to name of w\n"
            "        set {x, y} to position of w\n"
            "        set {ww, wh} to size of w\n"
            "        set output to output & appName & \" | \" & winName & \" | \" & x & \",\" & y & \" \" & ww & \"x\" & wh & \"\n\"\n"
            "      end repeat\n"
            "    end try\n"
            "  end repeat\n"
            "end tell\n"
            "return output' 2>&1";
        FILE *proc = popen(script, "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }
        if (out_len == 0)
            out_len = snprintf(output, output_sz, "No windows found (may need Accessibility permission)");

    } else if (strcmp(name, "window_focus") == 0) {
        const char *app = tool_call_get(tc, "app");
        if (!app) { out_len = snprintf(output, output_sz, "Error: app name required"); return out_len; }
        printf(ANSI_DIM "  [focusing %s]" ANSI_RESET "\n", app);
        // Shell-escape app name
        char safe_app[256];
        int ai = 0;
        for (int i = 0; app[i] && ai < 250; i++) {
            if (app[i] == '\'') { safe_app[ai++] = '\\'; safe_app[ai++] = '\''; }
            else safe_app[ai++] = app[i];
        }
        safe_app[ai] = 0;
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
            "osascript -e 'tell application \"%s\" to activate' 2>&1", safe_app);
        FILE *proc = popen(cmd, "r");
        if (proc) {
            while (out_len < (int)output_sz - 1) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            int ret = pclose(proc);
            if (ret == 0 && out_len == 0)
                out_len = snprintf(output, output_sz, "Focused %s", app);
        }

    } else if (strcmp(name, "display_info") == 0) {
        printf(ANSI_DIM "  [getting display info]" ANSI_RESET "\n");
        FILE *proc = popen("system_profiler SPDisplaysDataType 2>/dev/null", "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }

    } else if (strcmp(name, "net_info") == 0) {
        printf(ANSI_DIM "  [gathering network info]" ANSI_RESET "\n");
        // Active interfaces and IPs
        FILE *proc = popen(
            "echo '=== Active Interfaces ===' && "
            "ifconfig | grep -E '^[a-z]|inet ' | grep -v '127.0.0.1' && "
            "echo '' && echo '=== Default Route ===' && "
            "route -n get default 2>/dev/null | grep -E 'gateway|interface' && "
            "echo '' && echo '=== DNS ===' && "
            "scutil --dns 2>/dev/null | head -20", "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }

    } else if (strcmp(name, "net_connections") == 0) {
        const char *filter = tool_call_get(tc, "filter"); // "listening", "established", or port number
        printf(ANSI_DIM "  [listing connections%s%s]" ANSI_RESET "\n",
               filter ? ": " : "", filter ? filter : "");
        char cmd[512];
        if (filter && strcmp(filter, "listening") == 0) {
            snprintf(cmd, sizeof(cmd), "lsof -iTCP -sTCP:LISTEN -P -n 2>/dev/null | head -50");
        } else if (filter && strcmp(filter, "established") == 0) {
            snprintf(cmd, sizeof(cmd), "lsof -iTCP -sTCP:ESTABLISHED -P -n 2>/dev/null | head -50");
        } else if (filter && filter[0] >= '0' && filter[0] <= '9') {
            snprintf(cmd, sizeof(cmd), "lsof -iTCP:%s -P -n 2>/dev/null", filter);
        } else {
            snprintf(cmd, sizeof(cmd), "lsof -iTCP -P -n 2>/dev/null | head -80");
        }
        FILE *proc = popen(cmd, "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }

    } else if (strcmp(name, "service_status") == 0) {
        const char *svc = tool_call_get(tc, "service");
        printf(ANSI_DIM "  [checking services%s%s]" ANSI_RESET "\n",
               svc ? ": " : "", svc ? svc : "");
        char cmd[512];
        if (svc && svc[0]) {
            // Shell-escape
            char safe[256]; int si = 0;
            for (int i = 0; svc[i] && si < 250; i++) {
                if (svc[i] == '\'') { safe[si++] = '\''; safe[si++] = '\\'; safe[si++] = '\''; safe[si++] = '\''; }
                else safe[si++] = svc[i];
            }
            safe[si] = 0;
            snprintf(cmd, sizeof(cmd),
                "launchctl list 2>/dev/null | grep -i '%s'", safe);
        } else {
            snprintf(cmd, sizeof(cmd),
                "echo '=== User Services ===' && launchctl list 2>/dev/null | head -30 && "
                "echo '' && echo '=== System Services (running) ===' && "
                "sudo launchctl list 2>/dev/null | grep -v '\"0\"' | head -30 || "
                "launchctl list 2>/dev/null | head -40");
        }
        FILE *proc = popen(cmd, "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }

    } else if (strcmp(name, "disk_usage") == 0) {
        const char *path_arg = tool_call_get(tc, "path");
        printf(ANSI_DIM "  [checking disk usage]" ANSI_RESET "\n");
        char cmd[512];
        if (path_arg && path_arg[0]) {
            char resolved[PATH_MAX];
            resolve_path(path_arg, resolved, sizeof(resolved));
            snprintf(cmd, sizeof(cmd), "du -sh '%s'/* 2>/dev/null | sort -rh | head -30", resolved);
        } else {
            snprintf(cmd, sizeof(cmd),
                "echo '=== Volumes ===' && df -h 2>/dev/null && "
                "echo '' && echo '=== Largest in CWD ===' && "
                "du -sh '%s'/* 2>/dev/null | sort -rh | head -20", g.cwd);
        }
        FILE *proc = popen(cmd, "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }

    } else if (strcmp(name, "hardware_info") == 0) {
        printf(ANSI_DIM "  [gathering hardware info]" ANSI_RESET "\n");
        FILE *proc = popen(
            "echo '=== Hardware ===' && "
            "sysctl -n hw.model 2>/dev/null && "
            "sysctl -n machdep.cpu.brand_string 2>/dev/null && "
            "echo \"Cores: $(sysctl -n hw.ncpu 2>/dev/null) ($(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo '?')P + $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo '?')E)\" && "
            "echo \"Memory: $(( $(sysctl -n hw.memsize 2>/dev/null) / 1073741824 )) GB\" && "
            "echo \"GPU Cores: $(system_profiler SPDisplaysDataType 2>/dev/null | grep 'Total Number of Cores' | awk -F: '{print $2}' | xargs)\" && "
            "echo \"Memory Bandwidth: $(sysctl -n hw.memsize 2>/dev/null | awk '{printf \"%.0f GB/s (theoretical)\", $1/1073741824 * 4.266}')\" && "
            "echo '' && echo '=== Thermal ===' && "
            "pmset -g therm 2>/dev/null && "
            "echo '' && echo '=== Battery ===' && "
            "pmset -g batt 2>/dev/null", "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }

    } else if (strcmp(name, "applescript") == 0) {
        const char *script = tool_call_get(tc, "script");
        if (!script) { out_len = snprintf(output, output_sz, "Error: script required"); return out_len; }
        printf(ANSI_DIM "  [running AppleScript]" ANSI_RESET "\n");

        // Write script to temp file to avoid shell escaping issues
        char tmp[PATH_MAX];
        snprintf(tmp, sizeof(tmp), "/tmp/pre_applescript_%d.scpt", (int)getpid());
        FILE *sf = fopen(tmp, "w");
        if (!sf) {
            out_len = snprintf(output, output_sz, "Error: cannot create temp script");
            return out_len;
        }
        fputs(script, sf);
        fclose(sf);

        char cmd[PATH_MAX + 64];
        snprintf(cmd, sizeof(cmd), "osascript '%s' 2>&1", tmp);
        FILE *proc = popen(cmd, "r");
        if (proc) {
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }
        remove(tmp);
        if (out_len == 0)
            out_len = snprintf(output, output_sz, "AppleScript executed (no output)");

    } else {
        out_len = snprintf(output, output_sz, "Error: unknown tool '%s'", name);
    }

    return out_len;
}

static char *handle_tool_calls(char *response) {
    g.auto_approve_tools = 0;
    int loop_turns = 0;

    while (response && strstr(response, "<tool_call>")) {
        if (++loop_turns > MAX_TOOL_LOOP_TURNS) {
            printf(ANSI_YELLOW "\n  [tool loop limit reached (%d turns)]" ANSI_RESET "\n", MAX_TOOL_LOOP_TURNS);
            break;
        }

        // Collect all tool calls from the current response
        #define MAX_BATCH_TOOLS 8
        ToolCall batch[MAX_BATCH_TOOLS];
        int batch_count = 0;

        char *scan = response;
        while (batch_count < MAX_BATCH_TOOLS) {
            char *tc_start = strstr(scan, "<tool_call>");
            if (!tc_start) break;
            char *tc_end = strstr(tc_start, "</tool_call>");
            if (!tc_end) break;

            char *body_start = tc_start + 11;
            int tc_len = (int)(tc_end - body_start);

            // Heap-allocate tc_body for large payloads (file_write content)
            char *tc_body = malloc(tc_len + 1);
            memcpy(tc_body, body_start, tc_len);
            tc_body[tc_len] = 0;

            if (extract_tool_call_v2(tc_body, &batch[batch_count])) {
                batch_count++;
            }
            free(tc_body);
            scan = tc_end + 12; // past </tool_call>
        }

        if (batch_count == 0) break;

        // Execute all collected tool calls and combine responses
        size_t combined_cap = 65536 * batch_count + 512;
        char *combined = malloc(combined_cap);
        combined[0] = 0;
        size_t combined_len = 0;
        int denied = 0;

        for (int i = 0; i < batch_count && !denied; i++) {
            char output[65536];
            int out_len = execute_tool(&batch[i], output, sizeof(output));
            if (out_len < 0) { denied = 1; break; }

            int wrote = snprintf(combined + combined_len, combined_cap - combined_len,
                "<tool_response name=\"%s\">\n%s</tool_response>\n",
                batch[i].name, output);
            if (wrote > 0) combined_len += wrote;
        }

        // Free all ToolCall heap memory
        for (int i = 0; i < batch_count; i++) tool_call_free(&batch[i]);

        if (denied) {
            free(combined);
            free(response);
            return NULL;
        }

        session_save_turn(g.session_id, "tool", combined);
        free(response);

        int sock = send_request(combined, g.max_tokens, g.session_id);
        free(combined);
        if (sock < 0) return NULL;

        printf("\n");
        response = stream_response(sock);

        if (response && strlen(response) > 0) {
            session_save_turn(g.session_id, "assistant", response);
        }
    }

    return response;
}

// ============================================================================
// Slash command handlers
// ============================================================================

static void cmd_help(const char *args);
static void cmd_quit(const char *args);
static void cmd_file(const char *args);
static void cmd_ls(const char *args);
static void cmd_tree(const char *args);
static void cmd_save(const char *args);
static void cmd_cd(const char *args);
static void cmd_new(const char *args);
static void cmd_sessions(const char *args);
static void cmd_resume(const char *args);
static void cmd_think(const char *args);
static void cmd_clear(const char *args);
static void cmd_status(const char *args);
static void cmd_stats(const char *args);
static void cmd_export(const char *args);
static void cmd_summary(const char *args);
static void cmd_rewind(const char *args);
static void cmd_rename(const char *args);
static void cmd_context(const char *args);
static void cmd_run(const char *args);
static void cmd_edit(const char *args);
static void cmd_undo(const char *args);
static void cmd_memory(const char *args);
static void cmd_forget(const char *args);
static void cmd_channel(const char *args);
static void cmd_project(const char *args);

typedef struct {
    const char *name;
    const char *args_hint;
    const char *description;
    void (*handler)(const char *args);
} SlashCommand;

static SlashCommand commands[] = {
    {"/help",     "[topic]",  "Show help (tools/memory/channels/projects/tips)", cmd_help},
    {"/quit",     NULL,       "Exit PRE",                        cmd_quit},
    {"/exit",     NULL,       "Exit PRE",                        cmd_quit},
    {"/file",     "<path>",   "Attach file to next message",     cmd_file},
    {"/ls",       "[path]",   "List directory contents",         cmd_ls},
    {"/tree",     "[path]",   "Show directory tree (depth 3)",   cmd_tree},
    {"/save",     "<path>",   "Save last response to file",      cmd_save},
    {"/cd",       "<path>",   "Change working directory",        cmd_cd},
    {"/new",      NULL,       "Start a new session",             cmd_new},
    {"/sessions", NULL,       "List saved sessions",             cmd_sessions},
    {"/resume",   "<id>",     "Resume a previous session",       cmd_resume},
    {"/think",    NULL,       "Toggle think block visibility",   cmd_think},
    {"/clear",    NULL,       "Clear screen",                    cmd_clear},
    {"/status",   NULL,       "Show current status",             cmd_status},
    {"/stats",    NULL,       "Detailed session statistics",     cmd_stats},
    {"/export",   "[path]",   "Export conversation to markdown", cmd_export},
    {"/summary",  NULL,       "Ask model to summarize session",  cmd_summary},
    {"/rewind",   "[N]",      "Remove last N turns (default 1)", cmd_rewind},
    {"/rename",   "<name>",   "Name this session",               cmd_rename},
    {"/context",  NULL,       "Show context/token budget",       cmd_context},
    {"/run",      "<cmd>",    "Run shell command, optionally feed to model", cmd_run},
    {"/edit",     NULL,       "Open $EDITOR for multi-line input", cmd_edit},
    {"/undo",     NULL,       "Undo last file change",             cmd_undo},
    {"/memory",   "[query]",  "List or search saved memories",     cmd_memory},
    {"/forget",   "<query>",  "Delete a memory matching query",    cmd_forget},
    {"/channel",  "[name]",   "Switch channel or list channels",   cmd_channel},
    {"/project",  NULL,       "Show detected project info",        cmd_project},
    {NULL, NULL, NULL, NULL}
};

static int g_should_quit = 0;

static void help_commands(void) {
    printf("\n" ANSI_BOLD "  Commands:" ANSI_RESET "\n");
    for (int i = 0; commands[i].name; i++) {
        if (strcmp(commands[i].name, "/exit") == 0) continue;
        if (commands[i].args_hint)
            printf("  " ANSI_CYAN "%-12s" ANSI_RESET " %-10s %s\n",
                   commands[i].name, commands[i].args_hint, commands[i].description);
        else
            printf("  " ANSI_CYAN "%-12s" ANSI_RESET " %-10s %s\n",
                   commands[i].name, "", commands[i].description);
    }
    printf("\n");
}

static void help_tools(void) {
    printf("\n" ANSI_BOLD "  Agent Tools (29)" ANSI_RESET "\n");
    printf("  ─────────────────────────────\n");
    printf("  The model can call these tools autonomously during conversations.\n");
    printf("  Permission levels control what runs automatically vs. needs approval.\n\n");

    printf("  " ANSI_GREEN "Auto-approved" ANSI_RESET " (read-only, safe):\n");
    printf("    read_file, list_dir, glob, grep, system_info, process_list,\n");
    printf("    clipboard_read, memory_save, memory_search, memory_list,\n");
    printf("    window_list, display_info, net_info, net_connections,\n");
    printf("    service_status, disk_usage, hardware_info\n\n");

    printf("  " ANSI_YELLOW "Confirm once" ANSI_RESET " (write ops — approve once or 'a' for session):\n");
    printf("    file_write, file_edit, clipboard_write, web_fetch, notify,\n");
    printf("    memory_delete, screenshot, window_focus\n\n");

    printf("  " ANSI_RED "Confirm always" ANSI_RESET " (potentially destructive):\n");
    printf("    bash, process_kill, open_app, applescript\n\n");

    printf("  " ANSI_DIM "Tip: answer 'a' at a confirm prompt to auto-approve for the session." ANSI_RESET "\n\n");
}

static void help_memory(void) {
    printf("\n" ANSI_BOLD "  Memory System" ANSI_RESET "\n");
    printf("  ─────────────────────────────\n");
    printf("  PRE has persistent memory that survives across sessions.\n\n");

    printf("  " ANSI_BOLD "How it works:" ANSI_RESET "\n");
    printf("    Memories are stored as .md files in " ANSI_DIM "~/.pre/memory/" ANSI_RESET "\n");
    printf("    Project-scoped memories live in " ANSI_DIM "~/.pre/projects/{name}/memory/" ANSI_RESET "\n");
    printf("    All memories are loaded into context at the start of each session.\n\n");

    printf("  " ANSI_BOLD "Memory types:" ANSI_RESET "\n");
    printf("    " ANSI_CYAN "user" ANSI_RESET "       Your role, preferences, expertise\n");
    printf("    " ANSI_CYAN "feedback" ANSI_RESET "   How you want PRE to work (corrections & confirmations)\n");
    printf("    " ANSI_CYAN "project" ANSI_RESET "    Ongoing work, decisions, deadlines\n");
    printf("    " ANSI_CYAN "reference" ANSI_RESET "  Pointers to external resources\n\n");

    printf("  " ANSI_BOLD "Commands:" ANSI_RESET "\n");
    printf("    " ANSI_CYAN "/memory" ANSI_RESET "          List all memories\n");
    printf("    " ANSI_CYAN "/memory <query>" ANSI_RESET "  Search memories\n");
    printf("    " ANSI_CYAN "/forget <query>" ANSI_RESET "  Delete a memory\n\n");

    printf("  " ANSI_BOLD "Automatic:" ANSI_RESET " The model saves memories proactively when it learns\n");
    printf("  your preferences or discovers project context. You can also ask it\n");
    printf("  explicitly: " ANSI_DIM "\"Remember that I prefer tabs over spaces.\"" ANSI_RESET "\n\n");
}

static void help_channels(void) {
    printf("\n" ANSI_BOLD "  Channels" ANSI_RESET "\n");
    printf("  ─────────────────────────────\n");
    printf("  Channels are separate conversation threads within a project.\n");
    printf("  Each channel has its own history, context, and turn count.\n\n");

    printf("  " ANSI_BOLD "Commands:" ANSI_RESET "\n");
    printf("    " ANSI_CYAN "/channel" ANSI_RESET "          List channels for current project\n");
    printf("    " ANSI_CYAN "/channel <name>" ANSI_RESET "   Switch to a channel (creates if new)\n");
    printf("    " ANSI_CYAN "/new" ANSI_RESET "              Fresh session in current channel\n\n");

    printf("  " ANSI_BOLD "Examples:" ANSI_RESET "\n");
    printf("    " ANSI_DIM "/channel refactor" ANSI_RESET "    — work on refactoring in isolation\n");
    printf("    " ANSI_DIM "/channel debug-auth" ANSI_RESET "  — debug auth without polluting main context\n");
    printf("    " ANSI_DIM "/channel general" ANSI_RESET "     — back to default channel\n\n");

    printf("  " ANSI_DIM "Channels are scoped to the detected project. Changing projects\n");
    printf("  via /cd automatically switches to that project's #general channel." ANSI_RESET "\n\n");
}

static void help_projects(void) {
    printf("\n" ANSI_BOLD "  Projects" ANSI_RESET "\n");
    printf("  ─────────────────────────────\n");
    printf("  PRE auto-detects projects by looking for marker files when you\n");
    printf("  launch or " ANSI_CYAN "/cd" ANSI_RESET " into a directory.\n\n");

    printf("  " ANSI_BOLD "Detected markers:" ANSI_RESET "\n");
    printf("    .git  package.json  pyproject.toml  Cargo.toml  go.mod\n");
    printf("    Makefile  CMakeLists.txt  pom.xml  PRE.md\n\n");

    printf("  " ANSI_BOLD "Project config:" ANSI_RESET "  " ANSI_CYAN "PRE.md" ANSI_RESET "\n");
    printf("    Place a PRE.md in your project root with instructions for the model.\n");
    printf("    It's loaded into context on the first turn — like a briefing document.\n\n");
    printf("    Example PRE.md:\n");
    printf(ANSI_DIM
        "    # My Project\n"
        "    This is a FastAPI app with PostgreSQL.\n"
        "    - Always use async/await for database calls.\n"
        "    - Tests are in tests/ — run with: pytest -xvs\n"
        "    - Deploy target: AWS ECS on arm64.\n" ANSI_RESET "\n\n");

    printf("  " ANSI_BOLD "Project data:" ANSI_RESET "  " ANSI_DIM "~/.pre/projects/{name}/" ANSI_RESET "\n");
    printf("    " ANSI_DIM "memory/" ANSI_RESET "    Project-scoped memories\n");
    printf("    " ANSI_DIM "channels/" ANSI_RESET "   Channel metadata\n\n");

    printf("  " ANSI_BOLD "Commands:" ANSI_RESET "\n");
    printf("    " ANSI_CYAN "/project" ANSI_RESET "   Show detected project info\n");
    printf("    " ANSI_CYAN "/cd" ANSI_RESET "        Change directory (re-detects project)\n\n");
}

static void help_tips(void) {
    printf("\n" ANSI_BOLD "  Tips & Best Practices" ANSI_RESET "\n");
    printf("  ─────────────────────────────\n\n");

    printf("  " ANSI_BOLD "Getting the best results:" ANSI_RESET "\n");
    printf("    • Be specific. " ANSI_DIM "\"Review auth.py for SQL injection\"" ANSI_RESET " > " ANSI_DIM "\"check my code\"" ANSI_RESET "\n");
    printf("    • Attach files before asking. " ANSI_CYAN "/file src/main.py" ANSI_RESET " then ask your question.\n");
    printf("    • Use " ANSI_CYAN "/edit" ANSI_RESET " for complex prompts — opens your $EDITOR.\n");
    printf("    • Use " ANSI_CYAN "/think" ANSI_RESET " to watch the model's reasoning process.\n\n");

    printf("  " ANSI_BOLD "Context management:" ANSI_RESET "\n");
    printf("    • Check " ANSI_CYAN "/context" ANSI_RESET " to see how much of the 128K window you've used.\n");
    printf("    • PRE auto-compacts old turns at 75%% capacity to stay within budget.\n");
    printf("    • Use " ANSI_CYAN "/rewind" ANSI_RESET " to undo turns that added noise to context.\n");
    printf("    • Use channels to keep different tasks in separate contexts.\n\n");

    printf("  " ANSI_BOLD "Tool calling:" ANSI_RESET "\n");
    printf("    • The model reads files, searches code, and runs commands autonomously.\n");
    printf("    • Answer " ANSI_BOLD "'a'" ANSI_RESET " at a permission prompt to auto-approve for the session.\n");
    printf("    • " ANSI_CYAN "/undo" ANSI_RESET " reverts the last file_write or file_edit.\n");
    printf("    • Tool responses are capped at 8KB to preserve context budget.\n\n");

    printf("  " ANSI_BOLD "Shell integration:" ANSI_RESET "\n");
    printf("    • " ANSI_BOLD "!" ANSI_RESET "command runs a shell command (output not sent to model).\n");
    printf("    • " ANSI_CYAN "/run" ANSI_RESET " command runs it and offers to feed output to the model.\n\n");

    printf("  " ANSI_BOLD "Privacy:" ANSI_RESET "\n");
    printf("    • Everything runs locally. No data leaves your machine.\n");
    printf("    • Model: Gemma 4 26B-A4B via Ollama on port 11434.\n");
    printf("    • Session data: " ANSI_DIM "~/.pre/" ANSI_RESET "\n\n");
}

static void cmd_help(const char *args) {
    if (!args || !args[0]) {
        // Default: show overview with topic hints
        help_commands();
        printf("  Type a message to chat. Prefix with " ANSI_BOLD "!" ANSI_RESET " for shell commands.\n\n");
        printf("  " ANSI_BOLD "Help topics:" ANSI_RESET "  " ANSI_CYAN "/help tools" ANSI_RESET
               "  " ANSI_CYAN "/help memory" ANSI_RESET
               "  " ANSI_CYAN "/help channels" ANSI_RESET
               "  " ANSI_CYAN "/help projects" ANSI_RESET
               "  " ANSI_CYAN "/help tips" ANSI_RESET "\n\n");
        return;
    }
    while (*args == ' ') args++;

    if (strcmp(args, "tools") == 0) { help_tools(); return; }
    if (strcmp(args, "memory") == 0 || strcmp(args, "memories") == 0) { help_memory(); return; }
    if (strcmp(args, "channels") == 0 || strcmp(args, "channel") == 0) { help_channels(); return; }
    if (strcmp(args, "projects") == 0 || strcmp(args, "project") == 0) { help_projects(); return; }
    if (strcmp(args, "tips") == 0 || strcmp(args, "best") == 0 || strcmp(args, "practices") == 0) { help_tips(); return; }
    if (strcmp(args, "all") == 0) {
        help_commands();
        help_tools();
        help_memory();
        help_channels();
        help_projects();
        help_tips();
        return;
    }

    // Per-command help: /help <command>
    char lookup[32];
    if (args[0] == '/') strncpy(lookup, args, 31);
    else { lookup[0] = '/'; strncpy(lookup + 1, args, 30); }
    lookup[31] = 0;

    for (int i = 0; commands[i].name; i++) {
        if (strcmp(lookup, commands[i].name) == 0) {
            printf("\n  " ANSI_CYAN "%s" ANSI_RESET, commands[i].name);
            if (commands[i].args_hint) printf(" %s", commands[i].args_hint);
            printf("\n  %s\n\n", commands[i].description);
            return;
        }
    }

    printf(ANSI_YELLOW "  Unknown help topic: %s" ANSI_RESET "\n", args);
    printf("  Try: " ANSI_CYAN "/help tools" ANSI_RESET ", "
           ANSI_CYAN "/help memory" ANSI_RESET ", "
           ANSI_CYAN "/help channels" ANSI_RESET ", "
           ANSI_CYAN "/help projects" ANSI_RESET ", "
           ANSI_CYAN "/help tips" ANSI_RESET ", or "
           ANSI_CYAN "/help all" ANSI_RESET "\n\n");
}

static void cmd_quit(const char *args __attribute__((unused))) {
    g_should_quit = 1;
}

static void cmd_file(const char *args) {
    if (!args || !args[0]) {
        printf(ANSI_YELLOW "  Usage: /file <path>" ANSI_RESET "\n\n");
        return;
    }
    // Strip leading/trailing whitespace
    while (*args == ' ') args++;
    char resolved[PATH_MAX];
    resolve_path(args, resolved, sizeof(resolved));

    char *content = file_read_for_context(resolved);
    if (!content) {
        printf(ANSI_RED "  [error: cannot read '%s']" ANSI_RESET "\n\n", resolved);
        return;
    }

    // Append to existing attachment (support multiple /file calls)
    if (g_pending_attach) {
        size_t old_len = strlen(g_pending_attach);
        size_t new_len = strlen(content);
        g_pending_attach = realloc(g_pending_attach, old_len + new_len + 2);
        g_pending_attach[old_len] = '\n';
        memcpy(g_pending_attach + old_len + 1, content, new_len + 1);
        free(content);
    } else {
        g_pending_attach = content;
    }

    struct stat st;
    stat(resolved, &st);
    printf(ANSI_GREEN "  [attached: %s (%s)]" ANSI_RESET "\n\n", resolved, fmt_size(st.st_size));
}

static void cmd_ls(const char *args) {
    const char *path = (args && args[0]) ? args : g.cwd;
    char resolved[PATH_MAX];
    resolve_path(path, resolved, sizeof(resolved));

    char *listing = dir_listing(resolved);
    if (listing) {
        printf("\n%s\n", listing);
        free(listing);
    } else {
        printf(ANSI_RED "  [error: cannot list '%s']" ANSI_RESET "\n\n", resolved);
    }
}

static void cmd_tree(const char *args) {
    const char *path = (args && args[0]) ? args : g.cwd;
    char resolved[PATH_MAX];
    resolve_path(path, resolved, sizeof(resolved));

    char *tree = dir_tree(resolved, 3);
    if (tree) {
        printf("\n%s\n", tree);
        free(tree);
    } else {
        printf(ANSI_RED "  [error: cannot tree '%s']" ANSI_RESET "\n\n", resolved);
    }
}

static void cmd_save(const char *args) {
    if (!args || !args[0]) {
        printf(ANSI_YELLOW "  Usage: /save <path>" ANSI_RESET "\n\n");
        return;
    }
    if (!g.last_response || !g.last_response[0]) {
        printf(ANSI_YELLOW "  [no response to save]" ANSI_RESET "\n\n");
        return;
    }
    while (*args == ' ') args++;
    char resolved[PATH_MAX];
    resolve_path(args, resolved, sizeof(resolved));

    FILE *f = fopen(resolved, "w");
    if (!f) {
        printf(ANSI_RED "  [error: cannot write '%s': %s]" ANSI_RESET "\n\n", resolved, strerror(errno));
        return;
    }
    fputs(g.last_response, f);
    fclose(f);
    printf(ANSI_GREEN "  [saved to %s (%s)]" ANSI_RESET "\n\n", resolved, fmt_size(strlen(g.last_response)));
}

static void cmd_cd(const char *args) {
    if (!args || !args[0]) {
        printf("  CWD: %s\n\n", g.cwd);
        return;
    }
    while (*args == ' ') args++;
    char resolved[PATH_MAX];
    resolve_path(args, resolved, sizeof(resolved));

    // Normalize with realpath
    char real[PATH_MAX];
    if (!realpath(resolved, real)) {
        printf(ANSI_RED "  [error: '%s' not found]" ANSI_RESET "\n\n", resolved);
        return;
    }
    struct stat st;
    if (stat(real, &st) < 0 || !S_ISDIR(st.st_mode)) {
        printf(ANSI_RED "  [error: '%s' is not a directory]" ANSI_RESET "\n\n", real);
        return;
    }
    strncpy(g.cwd, real, PATH_MAX - 1);
    chdir(g.cwd);

    // Re-detect project for new directory
    char old_project[128];
    strncpy(old_project, g.project_id, sizeof(old_project));
    detect_project();
    load_memories(); // reload with new project scope

    printf(ANSI_GREEN "  [cwd: %s]" ANSI_RESET "\n", g.cwd);
    if (g.project_name[0] && strcmp(old_project, g.project_id) != 0) {
        printf(ANSI_GREEN "  [project: %s]" ANSI_RESET "\n", g.project_name);
        channel_init("general");
        printf(ANSI_GREEN "  [channel: #general]" ANSI_RESET "\n");
    }
    printf("\n");
}

static void cmd_new(const char *args __attribute__((unused))) {
    // Start fresh in current channel
    channel_init(g.channel[0] ? g.channel : "general");
    g.last_tok_s = 0;
    free(g_pending_attach); g_pending_attach = NULL;
    printf(ANSI_GREEN "  [new session in #%s]" ANSI_RESET "\n\n", g.channel);
}

static void cmd_sessions(const char *args __attribute__((unused))) {
    printf("\n");
    session_list();
}

static void cmd_resume(const char *args) {
    if (!args || !args[0]) {
        printf(ANSI_YELLOW "  Usage: /resume <session-id>" ANSI_RESET "\n\n");
        return;
    }
    while (*args == ' ') args++;
    char id[64];
    strncpy(id, args, 63); id[63] = 0;
    // Strip trailing whitespace
    int len = (int)strlen(id);
    while (len > 0 && id[len-1] == ' ') id[--len] = 0;

    int turns = session_load(id);
    if (turns == 0) {
        printf(ANSI_YELLOW "  [session '%s' not found]" ANSI_RESET "\n\n", id);
        return;
    }
    strncpy(g.session_id, id, sizeof(g.session_id) - 1);
    g.turn_count = turns / 2; // approximate
}

static void cmd_think(const char *args __attribute__((unused))) {
    g.show_thinking = !g.show_thinking;
    printf("  [thinking blocks: %s]\n\n", g.show_thinking ? "visible" : "hidden");
}

static void cmd_clear(const char *args __attribute__((unused))) {
    printf("\033[2J\033[H");
    fflush(stdout);
}

static void cmd_status(const char *args __attribute__((unused))) {
    printf("\n");
    printf("  " ANSI_BOLD "Model:" ANSI_RESET "     %s\n", MODEL_NAME);
    printf("  " ANSI_BOLD "Server:" ANSI_RESET "    http://localhost:%d\n", g.port);
    if (g.project_name[0])
        printf("  " ANSI_BOLD "Project:" ANSI_RESET "   %s (%s)\n", g.project_name, g.project_root);
    printf("  " ANSI_BOLD "Channel:" ANSI_RESET "   #%s\n", g.channel);
    printf("  " ANSI_BOLD "CWD:" ANSI_RESET "       %s\n", g.cwd);
    printf("  " ANSI_BOLD "Turn:" ANSI_RESET "      %d\n", g.turn_count);
    printf("  " ANSI_BOLD "Memory:" ANSI_RESET "    %d entries\n", g_memory_count);
    printf("  " ANSI_BOLD "Thinking:" ANSI_RESET "  %s\n", g.show_thinking ? "visible" : "hidden");
    if (g.last_tok_s > 0) {
        printf("  " ANSI_BOLD "Last:" ANSI_RESET "     %d tokens, %.1f tok/s, TTFT %.1fs\n",
               g.last_token_count, g.last_tok_s, g.last_ttft_ms / 1000.0);
    }
    if (g_pending_attach) {
        printf("  " ANSI_BOLD "Attached:" ANSI_RESET "  %s pending\n", fmt_size(strlen(g_pending_attach)));
    }
    printf("\n");
}

static void cmd_stats(const char *args __attribute__((unused))) {
    double elapsed = g.session_start_ms > 0 ? now_ms() - g.session_start_ms : 0;
    double avg_tok_s = (g.total_tokens_out > 0 && g.cumulative_gen_ms > 0)
        ? g.total_tokens_out * 1000.0 / g.cumulative_gen_ms : 0;

    printf("\n" ANSI_BOLD "  Session Statistics" ANSI_RESET "\n");
    printf("  ─────────────────────────────\n");
    printf("  " ANSI_DIM "Session:    " ANSI_RESET "%s\n", g.session_id);
    if (g.session_title[0])
        printf("  " ANSI_DIM "Title:      " ANSI_RESET "%s\n", g.session_title);
    printf("  " ANSI_DIM "Turns:      " ANSI_RESET "%d\n", g.turn_count);
    printf("  " ANSI_DIM "Tokens out: " ANSI_RESET "%d\n", g.total_tokens_out);
    printf("  " ANSI_DIM "Tokens in:  " ANSI_RESET "~%d (estimated)\n", g.total_tokens_in);
    if (avg_tok_s > 0)
        printf("  " ANSI_DIM "Avg speed:  " ANSI_RESET "%.1f tok/s\n", avg_tok_s);
    if (g.last_tok_s > 0)
        printf("  " ANSI_DIM "Last speed: " ANSI_RESET "%.1f tok/s (TTFT %.1fs)\n",
               g.last_tok_s, g.last_ttft_ms / 1000.0);
    if (elapsed > 0)
        printf("  " ANSI_DIM "Elapsed:    " ANSI_RESET "%s\n", fmt_elapsed(elapsed));
    printf("\n");
}

static void cmd_export(const char *args) {
    // Default output path
    char resolved[PATH_MAX];
    if (args && args[0]) {
        while (*args == ' ') args++;
        resolve_path(args, resolved, sizeof(resolved));
    } else {
        snprintf(resolved, sizeof(resolved), "%s/%s.md", g.cwd, g.session_id);
    }

    // Read session JSONL
    char src[1024];
    snprintf(src, sizeof(src), "%s/%s.jsonl", g.sessions_dir, g.session_id);
    FILE *in = fopen(src, "r");
    if (!in) { printf(ANSI_YELLOW "  [no conversation to export]" ANSI_RESET "\n\n"); return; }

    FILE *out = fopen(resolved, "w");
    if (!out) { fclose(in); printf(ANSI_RED "  [error: cannot write '%s']" ANSI_RESET "\n\n", resolved); return; }

    fprintf(out, "# PRE Session: %s\n\n", g.session_title[0] ? g.session_title : g.session_id);

    char line[MAX_RESPONSE];
    while (fgets(line, sizeof(line), in)) {
        char *role_start = strstr(line, "\"role\":\"");
        char *content_start = strstr(line, "\"content\":\"");
        if (!role_start || !content_start) continue;

        role_start += 8;
        char role[32]; int ri = 0;
        while (*role_start && *role_start != '"' && ri < 31) role[ri++] = *role_start++;
        role[ri] = 0;

        content_start += 11;
        char content[MAX_RESPONSE]; int ci = 0;
        for (int i = 0; content_start[i] && ci < MAX_RESPONSE - 1; i++) {
            if (content_start[i] == '"' && (i == 0 || content_start[i-1] != '\\')) break;
            if (content_start[i] == '\\' && content_start[i+1]) {
                i++;
                switch (content_start[i]) {
                    case 'n': content[ci++] = '\n'; break;
                    case 't': content[ci++] = '\t'; break;
                    case '"': content[ci++] = '"'; break;
                    case '\\': content[ci++] = '\\'; break;
                    default: content[ci++] = content_start[i]; break;
                }
            } else { content[ci++] = content_start[i]; }
        }
        content[ci] = 0;

        if (strcmp(role, "user") == 0)
            fprintf(out, "## User\n\n%s\n\n", content);
        else if (strcmp(role, "assistant") == 0)
            fprintf(out, "## Assistant\n\n%s\n\n---\n\n", content);
        else if (strcmp(role, "tool") == 0)
            fprintf(out, "## Tool Response\n\n```\n%s\n```\n\n", content);
    }
    fclose(in);
    fclose(out);

    struct stat st;
    stat(resolved, &st);
    printf(ANSI_GREEN "  [exported to %s (%s)]" ANSI_RESET "\n\n", resolved, fmt_size(st.st_size));
}

static void cmd_summary(const char *args __attribute__((unused))) {
    if (g.turn_count == 0) {
        printf(ANSI_YELLOW "  [no conversation to summarize]" ANSI_RESET "\n\n");
        return;
    }
    const char *prompt = "Please provide a concise summary of our conversation so far in 3-5 bullet points.";
    session_save_turn(g.session_id, "user", prompt);

    int sock = send_request(prompt, g.max_tokens, g.session_id);
    if (sock < 0) return;

    printf("\n");
    char *response = stream_response(sock);
    if (response && strlen(response) > 0) {
        session_save_turn(g.session_id, "assistant", response);
        free(g.last_response);
        g.last_response = strdup(response);
    }
    free(response);
    g.turn_count++;
}

static void cmd_rewind(const char *args) {
    int n = 1;
    if (args && args[0]) n = atoi(args);
    if (n < 1) n = 1;

    // Read all lines from session JSONL
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.jsonl", g.sessions_dir, g.session_id);
    FILE *f = fopen(path, "r");
    if (!f) { printf(ANSI_YELLOW "  [no conversation to rewind]" ANSI_RESET "\n\n"); return; }

    // Read all lines into memory
    char **lines = NULL;
    int line_count = 0, line_cap = 0;
    char buf[MAX_RESPONSE];
    while (fgets(buf, sizeof(buf), f)) {
        if (line_count >= line_cap) {
            line_cap = line_cap ? line_cap * 2 : 32;
            lines = realloc(lines, line_cap * sizeof(char *));
        }
        lines[line_count++] = strdup(buf);
    }
    fclose(f);

    // Remove last n*2 lines (user+assistant pairs)
    int remove = n * 2;
    if (remove > line_count) remove = line_count;
    int keep = line_count - remove;

    // Rewrite file
    f = fopen(path, "w");
    if (f) {
        for (int i = 0; i < keep; i++) fputs(lines[i], f);
        fclose(f);
    }
    for (int i = 0; i < line_count; i++) free(lines[i]);
    free(lines);

    g.turn_count -= n;
    if (g.turn_count < 0) g.turn_count = 0;

    printf(ANSI_GREEN "  [rewound %d turn%s — %d remaining]" ANSI_RESET "\n", n, n > 1 ? "s" : "", keep);
    printf(ANSI_DIM "  Note: server-side context is unchanged. Use /new for a fresh context." ANSI_RESET "\n\n");
}

static void cmd_rename(const char *args) {
    if (!args || !args[0]) {
        if (g.session_title[0])
            printf("  Session title: " ANSI_BOLD "%s" ANSI_RESET "\n\n", g.session_title);
        else
            printf(ANSI_YELLOW "  Usage: /rename <name>" ANSI_RESET "\n\n");
        return;
    }
    while (*args == ' ') args++;
    strncpy(g.session_title, args, sizeof(g.session_title) - 1);
    g.session_title[sizeof(g.session_title) - 1] = 0;
    // Strip trailing whitespace
    int len = (int)strlen(g.session_title);
    while (len > 0 && g.session_title[len-1] == ' ') g.session_title[--len] = 0;

    session_save_title(g.session_id, g.session_title);
    printf(ANSI_GREEN "  [session renamed: %s]" ANSI_RESET "\n\n", g.session_title);
}

static void cmd_context(const char *args __attribute__((unused))) {
    int used = g.total_tokens_in + g.total_tokens_out;
    int pct = MAX_CONTEXT > 0 ? (used * 100 / MAX_CONTEXT) : 0;
    if (pct > 100) pct = 100;

    printf("\n" ANSI_BOLD "  Context Window" ANSI_RESET "\n");
    printf("  ─────────────────────────────\n");

    // Visual bar (40 chars wide)
    int filled = pct * 40 / 100;
    printf("  [");
    for (int i = 0; i < 40; i++) {
        if (i < filled) {
            if (pct > 80) printf(ANSI_RED "█" ANSI_RESET);
            else if (pct > 60) printf(ANSI_YELLOW "█" ANSI_RESET);
            else printf(ANSI_GREEN "█" ANSI_RESET);
        } else printf(ANSI_DIM "░" ANSI_RESET);
    }
    printf("] %d%%\n", pct);

    printf("  " ANSI_DIM "Used:      " ANSI_RESET "~%d tokens\n", used);
    printf("  " ANSI_DIM "Budget:    " ANSI_RESET "%d tokens\n", MAX_CONTEXT);
    printf("  " ANSI_DIM "Remaining: " ANSI_RESET "~%d tokens\n", MAX_CONTEXT - used);
    printf("  " ANSI_DIM "Tokens in: " ANSI_RESET "~%d  " ANSI_DIM "Tokens out:" ANSI_RESET " %d\n",
           g.total_tokens_in, g.total_tokens_out);
    printf("\n");
}

static void cmd_run(const char *args) {
    if (!args || !args[0]) {
        printf(ANSI_YELLOW "  Usage: /run <command>" ANSI_RESET "\n\n");
        return;
    }
    while (*args == ' ') args++;

    printf(ANSI_DIM "  $ %s" ANSI_RESET "\n", args);
    FILE *proc = popen(args, "r");
    if (!proc) {
        printf(ANSI_RED "  [error: failed to execute]" ANSI_RESET "\n\n");
        return;
    }

    // Capture output
    size_t cap = 16384, len = 0;
    char *output = malloc(cap);
    int ch;
    while ((ch = fgetc(proc)) != EOF) {
        if (len + 2 > cap) { cap *= 2; output = realloc(output, cap); }
        output[len++] = (char)ch;
    }
    output[len] = 0;
    pclose(proc);

    // Display
    if (len > 0) {
        printf("%s", output);
        if (output[len-1] != '\n') printf("\n");
    }

    // Offer to feed to model
    printf(ANSI_DIM "  [feed to model? y/n] " ANSI_RESET);
    fflush(stdout);
    int c = getchar();
    while (getchar() != '\n');

    if (c == 'y' || c == 'Y') {
        // Wrap in code fences and attach
        size_t total = len + strlen(args) + 128;
        char *attached = malloc(total);
        snprintf(attached, total, "--- COMMAND: %s ---\n```\n%s```\n--- END OUTPUT ---\n", args, output);

        if (g_pending_attach) {
            size_t old_len = strlen(g_pending_attach);
            size_t new_len = strlen(attached);
            g_pending_attach = realloc(g_pending_attach, old_len + new_len + 2);
            g_pending_attach[old_len] = '\n';
            memcpy(g_pending_attach + old_len + 1, attached, new_len + 1);
            free(attached);
        } else {
            g_pending_attach = attached;
        }
        printf(ANSI_GREEN "  [output attached — type your message]" ANSI_RESET "\n\n");
    } else {
        printf("\n");
    }
    free(output);
}

static void cmd_edit(const char *args __attribute__((unused))) {
    const char *editor = getenv("EDITOR");
    if (!editor) editor = getenv("VISUAL");
    if (!editor) editor = "vi";

    char tmppath[] = "/tmp/pre_edit_XXXXXX";
    int fd = mkstemp(tmppath);
    if (fd < 0) {
        printf(ANSI_RED "  [error: cannot create temp file]" ANSI_RESET "\n\n");
        return;
    }
    close(fd);

    char cmd[PATH_MAX + 64];
    snprintf(cmd, sizeof(cmd), "%s %s", editor, tmppath);
    int ret = system(cmd);
    if (ret != 0) {
        unlink(tmppath);
        printf(ANSI_YELLOW "  [editor exited with error]" ANSI_RESET "\n\n");
        return;
    }

    // Read back the file
    struct stat st;
    if (stat(tmppath, &st) < 0 || st.st_size == 0) {
        unlink(tmppath);
        printf(ANSI_DIM "  [empty — cancelled]" ANSI_RESET "\n\n");
        return;
    }

    FILE *f = fopen(tmppath, "r");
    if (!f) { unlink(tmppath); return; }
    char *content = malloc(st.st_size + 1);
    size_t nread = fread(content, 1, st.st_size, f);
    content[nread] = 0;
    fclose(f);
    unlink(tmppath);

    // Strip trailing whitespace
    while (nread > 0 && (content[nread-1] == '\n' || content[nread-1] == ' ')) content[--nread] = 0;

    if (nread == 0) {
        free(content);
        printf(ANSI_DIM "  [empty — cancelled]" ANSI_RESET "\n\n");
        return;
    }

    // Attach as pending message content
    if (g_pending_attach) {
        size_t old_len = strlen(g_pending_attach);
        g_pending_attach = realloc(g_pending_attach, old_len + nread + 2);
        g_pending_attach[old_len] = '\n';
        memcpy(g_pending_attach + old_len + 1, content, nread + 1);
        free(content);
    } else {
        g_pending_attach = content;
    }

    // Count lines for display
    int lines = 1;
    for (size_t i = 0; i < nread; i++) if (g_pending_attach[i] == '\n') lines++;
    printf(ANSI_GREEN "  [attached %d lines from editor — type your message or press Enter to send]" ANSI_RESET "\n\n", lines);
}

static void cmd_undo(const char *args) {
    (void)args;
    if (g.checkpoint_count == 0) {
        printf("  [no file changes to undo]\n\n");
        return;
    }
    g.checkpoint_count--;
    const char *orig = g.checkpoints[g.checkpoint_count].path;
    const char *back = g.checkpoints[g.checkpoint_count].backup;

    FILE *src = fopen(back, "r");
    FILE *dst = fopen(orig, "w");
    if (!src || !dst) {
        printf("  [error restoring %s]\n\n", orig);
        if (src) fclose(src); if (dst) fclose(dst);
        return;
    }
    char buf[8192]; size_t n;
    while ((n = fread(buf, 1, sizeof(buf), src)) > 0) fwrite(buf, 1, n, dst);
    fclose(src); fclose(dst);
    remove(back);
    printf("  [restored %s]\n\n", orig);
}

static void cmd_memory(const char *args) {
    const char *query = (args && args[0]) ? args : NULL;
    if (query) while (*query == ' ') query++;

    if (g_memory_count == 0 && (!query || !query[0])) {
        printf("  [no memories saved yet]\n\n");
        return;
    }

    printf("\n" ANSI_BOLD "  Memories" ANSI_RESET " (%d total)\n", g_memory_count);
    printf("  ─────────────────────────────\n");

    for (int i = 0; i < g_memory_count; i++) {
        int match = 1;
        if (query && query[0]) {
            match = 0;
            if (strcasestr(g_memories[i].name, query)) match = 1;
            else if (strcasestr(g_memories[i].description, query)) match = 1;
            else if (strcasestr(g_memories[i].type, query)) match = 1;
        }
        if (!match) continue;

        printf("  " ANSI_CYAN "[%s]" ANSI_RESET " " ANSI_BOLD "%s" ANSI_RESET "\n",
               g_memories[i].type, g_memories[i].name);
        printf("  " ANSI_DIM "%s" ANSI_RESET "\n", g_memories[i].description);

        // Show first 2 lines of body
        char *body = read_memory_body(g_memories[i].file);
        if (body && body[0]) {
            char preview[256];
            strncpy(preview, body, 255);
            preview[255] = 0;
            // Truncate at 2nd newline
            int nl = 0;
            for (int j = 0; preview[j]; j++) {
                if (preview[j] == '\n') {
                    nl++;
                    if (nl >= 2) { preview[j] = 0; break; }
                }
            }
            printf("  " ANSI_DIM "%s" ANSI_RESET "\n", preview);
        }
        free(body);
        printf("\n");
    }
}

static void cmd_forget(const char *args) {
    if (!args || !args[0]) {
        printf(ANSI_YELLOW "  Usage: /forget <query>" ANSI_RESET "\n\n");
        return;
    }
    while (*args == ' ') args++;

    // Show what will be deleted
    for (int i = 0; i < g_memory_count; i++) {
        if (strcasestr(g_memories[i].name, args) ||
            strcasestr(g_memories[i].description, args)) {
            printf("  Found: " ANSI_BOLD "%s" ANSI_RESET " [%s]\n",
                   g_memories[i].name, g_memories[i].type);
            printf(ANSI_DIM "  [delete? y/n] " ANSI_RESET);
            fflush(stdout);
            int ch = getchar(); while (getchar() != '\n');
            if (ch == 'y' || ch == 'Y') {
                if (delete_memory(args)) {
                    printf(ANSI_GREEN "  [memory deleted]" ANSI_RESET "\n\n");
                } else {
                    printf(ANSI_RED "  [error deleting memory]" ANSI_RESET "\n\n");
                }
            } else {
                printf("  [cancelled]\n\n");
            }
            return;
        }
    }
    printf(ANSI_YELLOW "  No memory found matching '%s'" ANSI_RESET "\n\n", args);
}

static void cmd_channel(const char *args) {
    if (!args || !args[0]) {
        // No args: list channels
        channel_list();
        return;
    }
    while (*args == ' ') args++;

    // /channel list
    if (strcmp(args, "list") == 0) {
        channel_list();
        return;
    }

    // /channel <name> — switch to channel
    channel_switch(args);
}

static void cmd_project(const char *args __attribute__((unused))) {
    printf("\n" ANSI_BOLD "  Project" ANSI_RESET "\n");
    printf("  ─────────────────────────────\n");
    if (g.project_root[0]) {
        printf("  " ANSI_DIM "Name:    " ANSI_RESET ANSI_BOLD "%s" ANSI_RESET "\n", g.project_name);
        printf("  " ANSI_DIM "Root:    " ANSI_RESET "%s\n", g.project_root);
        printf("  " ANSI_DIM "ID:      " ANSI_RESET "%s\n", g.project_id);
        printf("  " ANSI_DIM "Data:    " ANSI_RESET "%s\n", g.project_dir);
        printf("  " ANSI_DIM "Channel: " ANSI_RESET "#%s\n", g.channel);

        // Check for PRE.md
        char pre_md[PATH_MAX];
        snprintf(pre_md, sizeof(pre_md), "%s/PRE.md", g.project_root);
        struct stat st;
        if (stat(pre_md, &st) == 0)
            printf("  " ANSI_DIM "Config:  " ANSI_RESET "%s (%s)\n", pre_md, fmt_size(st.st_size));
        else
            printf("  " ANSI_DIM "Config:  " ANSI_RESET ANSI_DIM "none (create PRE.md in project root)" ANSI_RESET "\n");

        // Count project memories
        int proj_mems = 0;
        for (int i = 0; i < g_memory_count; i++) {
            if (strstr(g_memories[i].file, g.project_dir)) proj_mems++;
        }
        printf("  " ANSI_DIM "Memories:" ANSI_RESET " %d project, %d global\n",
               proj_mems, g_memory_count - proj_mems);
    } else {
        printf("  " ANSI_DIM "No project detected." ANSI_RESET "\n");
        printf("  " ANSI_DIM "Projects are detected via .git, package.json, pyproject.toml, etc." ANSI_RESET "\n");
        printf("  " ANSI_DIM "CWD: %s" ANSI_RESET "\n", g.cwd);
    }
    printf("\n");
}

// ============================================================================
// Command dispatch
// ============================================================================

static int dispatch_command(const char *input) {
    if (input[0] != '/') return 0;

    const char *space = strchr(input, ' ');
    int cmd_len = space ? (int)(space - input) : (int)strlen(input);
    const char *args = space ? space + 1 : "";

    for (int i = 0; commands[i].name; i++) {
        if ((int)strlen(commands[i].name) == cmd_len &&
            strncmp(input, commands[i].name, cmd_len) == 0) {
            commands[i].handler(args);
            return 1;
        }
    }

    printf(ANSI_YELLOW "  Unknown command: %.*s" ANSI_RESET "\n", cmd_len, input);
    printf("  Type /help for available commands.\n\n");
    return 1;
}

// ============================================================================
// Linenoise callbacks — tab completion and hints
// ============================================================================

static void completion_cb(const char *buf, linenoiseCompletions *lc) {
    if (buf[0] != '/') return;
    int blen = (int)strlen(buf);
    for (int i = 0; commands[i].name; i++) {
        if (strncmp(buf, commands[i].name, blen) == 0) {
            linenoiseAddCompletion(lc, commands[i].name);
        }
    }
}

static char *hints_cb(const char *buf, int *color, int *bold) {
    if (buf[0] == '!' && buf[1] == 0) {
        *color = 90; *bold = 0;
        return " <shell command>";
    }
    if (buf[0] != '/') return NULL;

    for (int i = 0; commands[i].name; i++) {
        if (strcmp(buf, commands[i].name) == 0) {
            *color = 90; // dark gray
            *bold = 0;
            static char hint[128];
            if (commands[i].args_hint)
                snprintf(hint, sizeof(hint), " %s  %s", commands[i].args_hint, commands[i].description);
            else
                snprintf(hint, sizeof(hint), "  %s", commands[i].description);
            return hint;
        }
    }
    return NULL;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    @autoreleasepool {
        // Defaults
        g.port = 8000;
        g.max_tokens = 8192;
        getcwd(g.cwd, sizeof(g.cwd));
        const char *resume_id = NULL;

        static struct option long_options[] = {
            {"port",        required_argument, 0, 'p'},
            {"max-tokens",  required_argument, 0, 't'},
            {"show-think",  no_argument,       0, 's'},
            {"resume",      required_argument, 0, 'r'},
            {"sessions",    no_argument,       0, 'l'},
            {"dir",         required_argument, 0, 'd'},
            {"help",        no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        init_sessions_dir();
        init_pre_dirs();

        int c;
        while ((c = getopt_long(argc, argv, "p:t:sr:ld:h", long_options, NULL)) != -1) {
            switch (c) {
                case 'p': g.port = atoi(optarg); break;
                case 't': g.max_tokens = atoi(optarg); break;
                case 's': g.show_thinking = 1; break;
                case 'r': resume_id = optarg; break;
                case 'l': init_sessions_dir(); session_list(); return 0;
                case 'd': {
                    char real[PATH_MAX];
                    if (realpath(optarg, real)) {
                        strncpy(g.cwd, real, PATH_MAX - 1);
                        chdir(g.cwd);
                    } else {
                        fprintf(stderr, "Invalid directory: %s\n", optarg);
                        return 1;
                    }
                    break;
                }
                case 'h':
                    printf("Usage: %s [options]\n", argv[0]);
                    printf("  --port N         Server port (default: 8000)\n");
                    printf("  --max-tokens N   Max response tokens (default: 8192)\n");
                    printf("  --show-think     Show <think> blocks (dimmed)\n");
                    printf("  --resume ID      Resume a previous session\n");
                    printf("  --sessions       List saved sessions\n");
                    printf("  --dir PATH       Set working directory\n");
                    printf("  --help           This message\n");
                    return 0;
                default: return 1;
            }
        }

        // Handle positional arg as directory
        if (optind < argc) {
            char real[PATH_MAX];
            if (realpath(argv[optind], real)) {
                struct stat st;
                if (stat(real, &st) == 0 && S_ISDIR(st.st_mode)) {
                    strncpy(g.cwd, real, PATH_MAX - 1);
                    chdir(g.cwd);
                }
            }
        }

        // Detect project from CWD
        detect_project();

        // Initialize memory (must come after detect_project for per-project memories)
        init_memory_dir();
        load_memories();

        // Session / channel
        if (resume_id) {
            strncpy(g.session_id, resume_id, sizeof(g.session_id) - 1);
            // Try to extract channel from session id (format: project:channel)
            char *colon = strchr(g.session_id, ':');
            if (colon) {
                strncpy(g.channel, colon + 1, sizeof(g.channel) - 1);
            }
        } else {
            // Start in 'general' channel for this project
            channel_init("general");
        }

        // Banner
        tui_banner();

        // Health check
        if (!health_check()) {
            fprintf(stderr, ANSI_RED "  Server not running on port %d.\n" ANSI_RESET, g.port);
            fprintf(stderr, "  Start it: " ANSI_BOLD "./infer --serve %d" ANSI_RESET "\n\n", g.port);
            return 1;
        }
        printf(ANSI_GREEN "  Server connected." ANSI_RESET "\n\n");

        g.session_start_ms = now_ms();

        // Resume session
        if (resume_id) {
            int turns = session_load(g.session_id);
            if (turns == 0) printf(ANSI_YELLOW "  [session '%s' not found — starting fresh]" ANSI_RESET "\n\n", g.session_id);
            else g.turn_count = turns / 2;
            session_load_title(g.session_id, g.session_title, sizeof(g.session_title));
        }

        // Linenoise setup
        linenoiseSetMultiLine(1);
        linenoiseSetCompletionCallback(completion_cb);
        linenoiseSetHintsCallback(hints_cb);
        linenoiseHistoryLoad(g.history_path);
        linenoiseHistorySetMaxLen(500);

        // Main loop
        for (;;) {
            // Build prompt with channel name
            char prompt[128];
            if (g.project_name[0])
                snprintf(prompt, sizeof(prompt),
                    ANSI_DIM "%s" ANSI_RESET ANSI_BOLD " #%s> " ANSI_RESET,
                    g.project_name, g.channel);
            else
                snprintf(prompt, sizeof(prompt), ANSI_BOLD "pre #%s> " ANSI_RESET, g.channel);
            char *line = linenoise(prompt);

            if (!line) { printf("\n"); break; }
            if (strlen(line) == 0) { free(line); continue; }

            linenoiseHistoryAdd(line);
            linenoiseHistorySave(g.history_path);

            // ! bash mode — execute shell command directly
            if (line[0] == '!') {
                const char *cmd = line + 1;
                while (*cmd == ' ') cmd++;
                if (*cmd) system(cmd);
                free(line);
                continue;
            }

            // Command dispatch
            if (dispatch_command(line)) {
                free(line);
                if (g_should_quit) break;
                continue;
            }

            // /edit sends pending attachment as the message if no other input
            // If user types just Enter after /edit, send the attachment
            if (g_pending_attach && strlen(line) == 0) {
                free(line);
                continue;
            }

            // Build message with context preamble (first turn) and file attachment
            char *message = build_message(line, g.turn_count == 0);
            free(line);

            // Estimate input tokens (~4 chars per token)
            g.total_tokens_in += (int)(strlen(message) / 4);

            // Save user turn (save original input, not the preamble-augmented version)
            session_save_turn(g.session_id, "user", message);

            // Auto-compact if context is getting large
            maybe_compact();

            // Send to server
            int sock = send_request(message, g.max_tokens, g.session_id);
            free(message);
            if (sock < 0) continue;

            printf("\n");
            char *response = stream_response(sock);

            // Save assistant turn
            if (response && strlen(response) > 0) {
                session_save_turn(g.session_id, "assistant", response);
                free(g.last_response);
                g.last_response = strdup(response);
            }

            // Handle tool calls
            response = handle_tool_calls(response);

            free(response);
            g.turn_count++;

            // Show brief status after response
            {
                double elapsed = g.session_start_ms > 0 ? now_ms() - g.session_start_ms : 0;
                printf(ANSI_DIM);
                if (g.total_tokens_out > 0)
                    printf("  %d tok", g.total_tokens_out);
                if (g.last_tok_s > 0)
                    printf(" | %.1f t/s", g.last_tok_s);
                if (elapsed > 0)
                    printf(" | %s", fmt_elapsed(elapsed));
                printf(ANSI_RESET "\n");
            }
        }

        // Cleanup
        free(g.last_response);
        free(g_pending_attach);

        printf(ANSI_DIM "Goodbye." ANSI_RESET "\n");
        return 0;
    }
}
