/*
 * pre.m — Personal Reasoning Engine (PRE)
 *
 * Rich CLI for chatting with the local Qwen3.5-397B-A17B model.
 * Directory-aware, file-attachable, tool-calling reasoning interface.
 *
 * Build:  make pre
 * Run:    ./infer --serve 8000  (in another terminal)
 *         ./pre [--port 8000] [--dir /path]
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
#include "linenoise.h"

// ============================================================================
// Constants
// ============================================================================

#define MAX_RESPONSE    (1024 * 1024)   // 1MB response buffer
#define MAX_ATTACH      (128 * 1024)    // 128KB max file attachment
#define MAX_BODY        (256 * 1024)    // 256KB max HTTP body
#define SESSIONS_DIR    ".flash-moe/sessions"
#define HISTORY_FILE    ".flash-moe/pre_history"
#define MODEL_NAME      "Qwen3.5-397B-A17B"

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
#define MAX_CONTEXT     32768

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
} PreState;

static PreState g = {
    .port = 8000,
    .max_tokens = 8192,
    .show_thinking = 0,
    .turn_count = 0,
};

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

static void get_terminal_size(int *cols, int *rows) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        *cols = ws.ws_col;
        *rows = ws.ws_row;
    } else {
        *cols = 80;
        *rows = 24;
    }
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
// TUI — status bar, banner, spinner
// ============================================================================

static void tui_status_bar(void) {
    int cols, rows;
    get_terminal_size(&cols, &rows);

    // Session display name: title > truncated ID
    const char *name = g.session_title[0] ? g.session_title : g.session_id;
    char name_short[32];
    strncpy(name_short, name, 28); name_short[28] = 0;

    // Context usage estimate
    int ctx_used = g.total_tokens_in + g.total_tokens_out;
    int ctx_pct = MAX_CONTEXT > 0 ? (ctx_used * 100 / MAX_CONTEXT) : 0;
    if (ctx_pct > 100) ctx_pct = 100;

    // Elapsed
    double elapsed = g.session_start_ms > 0 ? now_ms() - g.session_start_ms : 0;

    char bar[512];
    int off = 0;
    off += snprintf(bar + off, sizeof(bar) - off, " PRE | %s", name_short);
    if (g.total_tokens_out > 0)
        off += snprintf(bar + off, sizeof(bar) - off, " | %d tok", g.total_tokens_out);
    if (ctx_pct > 0)
        off += snprintf(bar + off, sizeof(bar) - off, " | ctx %d%%", ctx_pct);
    if (g.last_tok_s > 0)
        off += snprintf(bar + off, sizeof(bar) - off, " | %.1f t/s", g.last_tok_s);
    if (elapsed > 0)
        off += snprintf(bar + off, sizeof(bar) - off, " | %s", fmt_elapsed(elapsed));
    off += snprintf(bar + off, sizeof(bar) - off, " | %s ", g.cwd);

    // Truncate to terminal width
    int blen = (int)strlen(bar);
    if (blen > cols) { bar[cols] = 0; blen = cols; }

    printf("\0337");                        // save cursor
    printf("\033[%d;1H", rows);             // move to last row
    printf(ANSI_REV "%s", bar);             // reverse video
    for (int i = blen; i < cols; i++) putchar(' ');
    printf(ANSI_RESET);
    printf("\0338");                        // restore cursor
    fflush(stdout);
}

static void tui_clear_status(void) {
    int cols, rows;
    get_terminal_size(&cols, &rows);
    printf("\0337\033[%d;1H\033[K\0338", rows);
    fflush(stdout);
}

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
    printf("  Session: %s\n", g.session_id);
    printf("  CWD:     %s\n", g.cwd);
    printf("  Type " ANSI_BOLD "/help" ANSI_RESET " for commands\n\n");
}

// ============================================================================
// Session management
// ============================================================================

static void init_sessions_dir(void) {
    const char *home = getenv("HOME") ?: "/tmp";
    char parent[1024];
    snprintf(parent, sizeof(parent), "%s/.flash-moe", home);
    mkdir(parent, 0755);
    snprintf(g.sessions_dir, sizeof(g.sessions_dir), "%s/%s", home, SESSIONS_DIR);
    mkdir(g.sessions_dir, 0755);
    snprintf(g.history_path, sizeof(g.history_path), "%s/%s", home, HISTORY_FILE);
}

static void generate_session_id(char *buf, size_t bufsize) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    snprintf(buf, bufsize, "pre-%d-%ld%06d",
             (int)getpid(), (long)tv.tv_sec, (int)tv.tv_usec);
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

    size_t cap = 8192 + flen;
    char *preamble = malloc(cap);
    snprintf(preamble, cap,
        "<context>\n"
        "Working directory: %s\n"
        "Files:\n%s\n"
        "Available tools (wrap calls in <tool_call>JSON</tool_call> tags):\n"
        "- bash: Run a shell command. {\"name\":\"bash\",\"arguments\":{\"command\":\"...\"}}\n"
        "- read_file: Read a file. {\"name\":\"read_file\",\"arguments\":{\"path\":\"...\"}}\n"
        "- list_dir: List directory. {\"name\":\"list_dir\",\"arguments\":{\"path\":\"...\"}}\n"
        "\n"
        "Guidelines:\n"
        "- Running locally on Apple Silicon. All data stays on this machine.\n"
        "- File paths are relative to the working directory unless absolute.\n"
        "- When using tools, stop and wait for <tool_response> before continuing.\n"
        "- For complex tasks, think step by step.\n"
        "</context>\n\n",
        g.cwd, files_list);
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

    char *escaped = json_escape_alloc(user_message);
    if (!escaped) { close(sock); return -1; }

    size_t elen = strlen(escaped);
    size_t body_cap = elen + 256;
    char *body = malloc(body_cap);
    int body_len = snprintf(body, body_cap,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],"
        "\"max_tokens\":%d,\"stream\":true,\"session_id\":\"%s\"}",
        escaped, max_tokens, session_id);
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
            while (text[i] == '#') i++;
            while (text[i] == ' ') i++;
            printf(ANSI_HEADER);
            while (text[i] && text[i] != '\n') { putchar(text[i]); i++; }
            printf(ANSI_RESET);
            if (text[i] == '\n') { putchar('\n'); g_md.line_start = 1; }
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

        if (c == '\n') g_md.line_start = 1;
        else g_md.line_start = 0;

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

        char *ck = strstr(line + 6, "\"content\":\"");
        if (!ck) continue;
        ck += 11;

        char decoded[4096]; int di = 0;
        for (int i = 0; ck[i] && ck[i] != '"' && di < 4095; i++) {
            if (ck[i] == '\\' && ck[i+1]) {
                i++;
                switch (ck[i]) {
                    case 'n': decoded[di++]='\n'; break;
                    case 't': decoded[di++]='\t'; break;
                    case '"': decoded[di++]='"'; break;
                    case '\\': decoded[di++]='\\'; break;
                    default: decoded[di++]=ck[i]; break;
                }
            } else decoded[di++] = ck[i];
        }
        decoded[di] = 0;
        if (!di) continue;

        // Clear spinner on first real token
        if (spinning) {
            printf("\r\033[K");
            fflush(stdout);
            spinning = 0;
        }

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

// Extract command string from tool call body (handles JSON and fallback formats)
static int extract_tool_call(const char *tc_body, char *name, size_t name_sz,
                             char *arg_key, size_t key_sz, char *arg_val, size_t val_sz) {
    name[0] = arg_key[0] = arg_val[0] = 0;

    // JSON format: {"name":"bash","arguments":{"command":"..."}}
    char *nk = strstr(tc_body, "\"name\"");
    if (nk) {
        nk = strchr(nk + 6, '"'); if (nk) { nk++;
            int ni = 0;
            while (*nk && *nk != '"' && ni < (int)name_sz - 1) name[ni++] = *nk++;
            name[ni] = 0;
        }
    }

    // Find the argument value
    // For "command":"..." or "path":"..."
    static const char *keys[] = {"command", "path", NULL};
    for (int k = 0; keys[k]; k++) {
        char search[64];
        snprintf(search, sizeof(search), "\"%s\"", keys[k]);
        char *ck = strstr(tc_body, search);
        if (!ck) continue;
        strncpy(arg_key, keys[k], key_sz - 1);
        ck = strchr(ck + strlen(search), '"');
        if (!ck) continue; ck++;
        int vi = 0;
        for (int i = 0; ck[i] && ck[i] != '"' && vi < (int)val_sz - 1; i++) {
            if (ck[i] == '\\' && ck[i+1]) {
                i++;
                switch (ck[i]) {
                    case 'n': arg_val[vi++] = '\n'; break;
                    case 't': arg_val[vi++] = '\t'; break;
                    case '"': arg_val[vi++] = '"'; break;
                    case '\\': arg_val[vi++] = '\\'; break;
                    default: arg_val[vi++] = ck[i]; break;
                }
            } else arg_val[vi++] = ck[i];
        }
        arg_val[vi] = 0;
        return 1;
    }

    // Fallback: look for bash command text
    if (name[0] == 0 && strstr(tc_body, "bash")) strncpy(name, "bash", name_sz);
    return arg_val[0] != 0;
}

static char *handle_tool_calls(char *response) {
    g.auto_approve_tools = 0;

    while (response && strstr(response, "<tool_call>")) {
        char *tc_start = strstr(response, "<tool_call>");
        char *tc_end = strstr(tc_start, "</tool_call>");
        if (!tc_start || !tc_end) break;

        tc_start += 11;
        char tc_body[8192] = {0};
        int tc_len = (int)(tc_end - tc_start);
        if (tc_len > 8191) tc_len = 8191;
        memcpy(tc_body, tc_start, tc_len);

        char name[64], arg_key[64], arg_val[4096];
        if (!extract_tool_call(tc_body, name, sizeof(name),
                               arg_key, sizeof(arg_key), arg_val, sizeof(arg_val))) {
            break;
        }

        char output[65536] = {0};
        int out_len = 0;

        if (strcmp(name, "read_file") == 0) {
            // Read-only: no confirmation needed
            char resolved[PATH_MAX];
            resolve_path(arg_val, resolved, sizeof(resolved));
            printf(ANSI_DIM "  [reading %s]" ANSI_RESET "\n", resolved);
            char *content = file_read_for_context(resolved);
            if (content) {
                int cl = (int)strlen(content);
                if (cl > 65535) cl = 65535;
                memcpy(output, content, cl);
                out_len = cl;
                free(content);
            } else {
                out_len = snprintf(output, sizeof(output), "Error: cannot read file '%s'", arg_val);
            }
        } else if (strcmp(name, "list_dir") == 0) {
            // Read-only: no confirmation needed
            char resolved[PATH_MAX];
            resolve_path(arg_val, resolved, sizeof(resolved));
            printf(ANSI_DIM "  [listing %s]" ANSI_RESET "\n", resolved);
            char *listing = dir_listing(resolved);
            if (listing) {
                int ll = (int)strlen(listing);
                if (ll > 65535) ll = 65535;
                memcpy(output, listing, ll);
                out_len = ll;
                free(listing);
            } else {
                out_len = snprintf(output, sizeof(output), "Error: cannot list directory '%s'", arg_val);
            }
        } else {
            // bash: needs confirmation
            printf(ANSI_YELLOW "  $ %s" ANSI_RESET "\n", arg_val);

            int approved = g.auto_approve_tools;
            if (!approved) {
                printf(ANSI_DIM "  [execute? y/n/a(lways)] " ANSI_RESET);
                fflush(stdout);
                int ch = getchar();
                while (getchar() != '\n');
                if (ch == 'a' || ch == 'A') { approved = 1; g.auto_approve_tools = 1; }
                else if (ch == 'y' || ch == 'Y') approved = 1;
            }

            if (!approved) {
                printf(ANSI_DIM "  [skipped]" ANSI_RESET "\n");
                free(response);
                return NULL;
            }

            FILE *proc = popen(arg_val, "r");
            if (proc) {
                while (out_len < 65535) {
                    int ch = fgetc(proc);
                    if (ch == EOF) break;
                    output[out_len++] = (char)ch;
                }
                output[out_len] = 0;
                pclose(proc);
            }

            if (out_len > 0) {
                printf(ANSI_DIM "%s" ANSI_RESET, output);
                if (output[out_len - 1] != '\n') printf("\n");
            }
        }

        // Send tool response back
        char *tool_msg = malloc(out_len + 256);
        snprintf(tool_msg, out_len + 256, "<tool_response>\n%s</tool_response>", output);

        session_save_turn(g.session_id, "tool", tool_msg);
        free(response);

        int sock = send_request(tool_msg, g.max_tokens, g.session_id);
        free(tool_msg);
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

typedef struct {
    const char *name;
    const char *args_hint;
    const char *description;
    void (*handler)(const char *args);
} SlashCommand;

static SlashCommand commands[] = {
    {"/help",     NULL,       "Show available commands",         cmd_help},
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
    {NULL, NULL, NULL, NULL}
};

static int g_should_quit = 0;

static void cmd_help(const char *args __attribute__((unused))) {
    printf("\n" ANSI_BOLD "  Commands:" ANSI_RESET "\n");
    for (int i = 0; commands[i].name; i++) {
        if (strcmp(commands[i].name, "/exit") == 0) continue; // skip alias
        if (commands[i].args_hint) {
            printf("  " ANSI_CYAN "%-12s" ANSI_RESET " %-10s %s\n",
                   commands[i].name, commands[i].args_hint, commands[i].description);
        } else {
            printf("  " ANSI_CYAN "%-12s" ANSI_RESET " %-10s %s\n",
                   commands[i].name, "", commands[i].description);
        }
    }
    printf("\n  Type a message to chat. Prefix with " ANSI_BOLD "!" ANSI_RESET
           " for shell commands.\n"
           "  Use " ANSI_BOLD "/file" ANSI_RESET " to attach files, "
           ANSI_BOLD "/edit" ANSI_RESET " for multi-line input.\n\n");
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
    printf(ANSI_GREEN "  [cwd: %s]" ANSI_RESET "\n\n", g.cwd);
}

static void cmd_new(const char *args __attribute__((unused))) {
    generate_session_id(g.session_id, sizeof(g.session_id));
    g.turn_count = 0;
    g.last_tok_s = 0;
    g.total_tokens_in = 0;
    g.total_tokens_out = 0;
    g.cumulative_gen_ms = 0;
    g.session_start_ms = now_ms();
    g.session_title[0] = 0;
    free(g.last_response); g.last_response = NULL;
    free(g_pending_attach); g_pending_attach = NULL;
    printf(ANSI_GREEN "  [new session: %s]" ANSI_RESET "\n\n", g.session_id);
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
    printf("  " ANSI_BOLD "Session:" ANSI_RESET "   %s\n", g.session_id);
    printf("  " ANSI_BOLD "CWD:" ANSI_RESET "       %s\n", g.cwd);
    printf("  " ANSI_BOLD "Turn:" ANSI_RESET "      %d\n", g.turn_count);
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

        // Session
        if (resume_id) {
            strncpy(g.session_id, resume_id, sizeof(g.session_id) - 1);
        } else {
            generate_session_id(g.session_id, sizeof(g.session_id));
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
            tui_status_bar();

            char *line = linenoise(ANSI_BOLD "pre> " ANSI_RESET);
            tui_clear_status();

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
        }

        // Cleanup
        free(g.last_response);
        free(g_pending_attach);

        printf(ANSI_DIM "Goodbye." ANSI_RESET "\n");
        return 0;
    }
}
