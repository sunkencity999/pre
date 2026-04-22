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
#import <AppKit/AppKit.h>
#import <WebKit/WebKit.h>
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
#include <sys/wait.h>
#include <mach-o/dyld.h>
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
#define CONNECTIONS_FILE ".pre/connections.json"
#define MAX_CHANNELS    64
#define MAX_CONNECTIONS 16
#define CRON_FILE       ".pre/cron.json"
#define MAX_CRON_JOBS   32

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
#define MAX_CONTEXT     262144
// Actual num_ctx sent per-request — must match across CLI and Web GUI.
// Sending any different value to Ollama triggers a full model reload (300s+).
// Read from ~/.pre/context (written by install.sh) at startup; fallback 131072.
#define MODEL_CTX_DEFAULT 131072
static int MODEL_CTX = MODEL_CTX_DEFAULT;

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
    int last_num_ctx;           // last num_ctx sent to Ollama (avoid KV cache rebuilds)
    double cumulative_gen_ms;   // total generation time
    double session_start_ms;    // when session began

    // Conversation state
    char *last_response;
    int turn_count;
    char session_title[128];    // human-readable session name

    // Feature toggles
    int show_thinking;
    int _reserved_approve;    // unused, kept for struct compat

    // File checkpoints for undo
    #define MAX_CHECKPOINTS 64
    struct { char path[PATH_MAX]; char backup[PATH_MAX]; } checkpoints[MAX_CHECKPOINTS];
    int checkpoint_count;
    int _reserved_open;     // unused, kept for struct compat

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
    .max_tokens = 16384,
    .show_thinking = 1,
    .turn_count = 0,
};

// Model to request from Ollama
static const char *g_model = "pre-gemma4";

// Path to our own executable (for exec-based subprocess launching)
static char g_exe_path[PATH_MAX];

// Agent identity
static char g_agent_name[128] = "PRE";  // default, overridden by identity file

static const char *identity_path(void) {
    static char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/.pre/identity.json", getenv("HOME") ?: "/tmp");
    return path;
}

static void load_identity(void) {
    FILE *f = fopen(identity_path(), "r");
    if (!f) return;
    char buf[512];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);
    // Extract "name" field
    const char *p = strstr(buf, "\"name\"");
    if (!p) return;
    p = strchr(p, ':');
    if (!p) return;
    p++;
    while (*p == ' ' || *p == '"') p++;
    int i = 0;
    while (*p && *p != '"' && i < (int)sizeof(g_agent_name) - 1)
        g_agent_name[i++] = *p++;
    g_agent_name[i] = 0;
}

static void save_identity(const char *name) {
    strlcpy(g_agent_name, name, sizeof(g_agent_name));
    FILE *f = fopen(identity_path(), "w");
    if (!f) return;
    fprintf(f, "{\"name\":\"%s\"}\n", name);
    fclose(f);
    chmod(identity_path(), 0600);
}

// Forward declarations for functions used before their definition
static int json_extract_str(const char *json, const char *key, char *dst, size_t dsz);
static void comfyui_stop(void);
static int g_argus_enabled = 0;  // Argus companion — toggle with /argus
static void argus_react(const char *tool_name, const char *tool_output);

// Telegram bot subprocess
static pid_t g_telegram_pid = 0;

// Unload model from GPU on exit
static void ollama_unload(void) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "curl -s --max-time 5 -X POST -H 'Content-Type: application/json' "
        "-d '{\"model\":\"%s\",\"keep_alive\":\"0\"}' "
        "'http://127.0.0.1:%d/api/chat' >/dev/null 2>&1",
        g_model, g.port);
    system(cmd);
}

static void stop_telegram(void) {
    if (g_telegram_pid > 0) {
        kill(g_telegram_pid, SIGTERM);
        // Wait briefly, then force kill if still alive
        for (int i = 0; i < 10; i++) {
            int status;
            pid_t r = waitpid(g_telegram_pid, &status, WNOHANG);
            if (r != 0) { g_telegram_pid = 0; return; }
            usleep(200000); // 200ms
        }
        // Still alive after 2s — force kill
        kill(g_telegram_pid, SIGKILL);
        waitpid(g_telegram_pid, NULL, 0);
        g_telegram_pid = 0;
    }
}

static void start_telegram(void); // defined after connection infrastructure

static void handle_exit_signal(int sig) {
    (void)sig;
    printf("\n" ANSI_DIM "Stopping services..." ANSI_RESET "\n");
    stop_telegram();
    comfyui_stop();
    ollama_unload();
    printf(ANSI_DIM "Goodbye." ANSI_RESET "\n");
    _exit(0);
}

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
    // Destructive operations that permanently delete data — always confirm
    if (strcmp(name, "process_kill") == 0 ||
        strcmp(name, "memory_delete") == 0 ||
        strcmp(name, "applescript") == 0) return PERM_CONFIRM_ALWAYS;
    // Everything else auto-approves — PRE is a power-user tool
    return PERM_AUTO;
}

// Pending file attachment(s) — prepended to next message
static char *g_pending_attach = NULL;

// Pending image attachment — base64 PNG, sent as multimodal content
static char *g_pending_image = NULL;

// Native tool calls from Ollama's structured response (set by stream_response, consumed by handle_tool_calls)
static char *g_native_tool_calls = NULL;  // raw JSON array string: [{"id":"...","function":{"name":"...","arguments":{...}}}]

// ComfyUI image generation state
static pid_t g_comfyui_pid = 0;
static int g_comfyui_port = 8188;
static char g_comfyui_checkpoint[256] = "sd_xl_turbo_1.0_fp16.safetensors";
static int g_comfyui_installed = 0;  // set on startup by checking config file

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
    snprintf(path, sizeof(path), "%s/.pre/artifacts", home);
    mkdir(path, 0755);
}

// ============================================================================
// Artifact system — visual pop-outs for HTML, markdown, CSV, etc.
// ============================================================================

// Session artifact tracking
#define MAX_SESSION_ARTIFACTS 64
static struct {
    char path[PATH_MAX];
    char title[128];
    char type[16];
    int section_count;  // for multi-part append tracking
} g_artifacts[MAX_SESSION_ARTIFACTS];
static int g_artifact_count = 0;

// Minimal markdown → HTML converter
static char *markdown_to_html(const char *md) {
    size_t cap = strlen(md) * 3 + 4096;
    char *html = malloc(cap);
    if (!html) return NULL;
    size_t pos = 0;

    #define HCAT(s) do { \
        size_t _l = strlen(s); \
        if (pos + _l < cap) { memcpy(html + pos, s, _l); pos += _l; } \
    } while(0)

    // Dark-mode HTML wrapper
    HCAT("<!DOCTYPE html><html><head><meta charset='utf-8'>"
         "<meta name='viewport' content='width=device-width,initial-scale=1'>"
         "<style>"
         "body{font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',Helvetica,sans-serif;"
         "background:#1a1a2e;color:#e0e0e0;max-width:800px;margin:40px auto;padding:0 20px;"
         "line-height:1.6;font-size:15px}"
         "h1,h2,h3,h4{color:#00d4ff;margin-top:1.5em}"
         "h1{border-bottom:1px solid #333;padding-bottom:8px}"
         "a{color:#64b5f6;text-decoration:none}"
         "a:hover{text-decoration:underline}"
         "code{background:#2a2a3e;padding:2px 6px;border-radius:3px;font-size:0.9em;"
         "font-family:'SF Mono',Menlo,monospace}"
         "pre{background:#0d0d1a;border:1px solid #333;border-radius:6px;padding:16px;"
         "overflow-x:auto}"
         "pre code{background:none;padding:0}"
         "blockquote{border-left:3px solid #00d4ff;margin-left:0;padding-left:16px;color:#aaa}"
         "table{border-collapse:collapse;width:100%}"
         "th,td{border:1px solid #333;padding:8px;text-align:left}"
         "th{background:#2a2a3e}"
         "hr{border:none;border-top:1px solid #333}"
         "ul,ol{padding-left:24px}"
         "li{margin:4px 0}"
         ".artifact-badge{background:#00d4ff;color:#1a1a2e;padding:2px 8px;border-radius:3px;"
         "font-size:0.75em;font-weight:600;float:right}"
         "</style></head><body>"
         "<span class='artifact-badge'>PRE Artifact</span>");

    const char *p = md;
    int in_code_block = 0;
    int in_list = 0;

    while (*p) {
        // Find end of current line
        const char *eol = strchr(p, '\n');
        if (!eol) eol = p + strlen(p);
        size_t line_len = (size_t)(eol - p);
        char line[4096];
        if (line_len >= sizeof(line)) line_len = sizeof(line) - 1;
        memcpy(line, p, line_len);
        line[line_len] = 0;

        // Code blocks (```)
        if (strncmp(line, "```", 3) == 0) {
            if (in_code_block) {
                HCAT("</code></pre>");
                in_code_block = 0;
            } else {
                if (in_list) { HCAT("</ul>"); in_list = 0; }
                HCAT("<pre><code>");
                in_code_block = 1;
            }
            p = (*eol) ? eol + 1 : eol;
            continue;
        }

        if (in_code_block) {
            // HTML-escape inside code blocks
            for (size_t i = 0; i < line_len; i++) {
                if (line[i] == '<') HCAT("&lt;");
                else if (line[i] == '>') HCAT("&gt;");
                else if (line[i] == '&') HCAT("&amp;");
                else { html[pos++] = line[i]; }
            }
            HCAT("\n");
            p = (*eol) ? eol + 1 : eol;
            continue;
        }

        // Close list if this line isn't a list item
        if (in_list && line[0] != '-' && line[0] != '*' &&
            !(line[0] >= '0' && line[0] <= '9' && strchr(line, '.'))) {
            HCAT("</ul>");
            in_list = 0;
        }

        // Empty line
        if (line_len == 0) {
            HCAT("<br>");
            p = (*eol) ? eol + 1 : eol;
            continue;
        }

        // Horizontal rule
        if ((strncmp(line, "---", 3) == 0 || strncmp(line, "***", 3) == 0) && line_len <= 5) {
            HCAT("<hr>");
            p = (*eol) ? eol + 1 : eol;
            continue;
        }

        // Headers
        int hlevel = 0;
        const char *htext = line;
        while (*htext == '#' && hlevel < 6) { hlevel++; htext++; }
        if (hlevel > 0 && (*htext == ' ' || *htext == 0)) {
            while (*htext == ' ') htext++;
            char tag[8];
            snprintf(tag, sizeof(tag), "<h%d>", hlevel);
            HCAT(tag);
            HCAT(htext);
            snprintf(tag, sizeof(tag), "</h%d>", hlevel);
            HCAT(tag);
            p = (*eol) ? eol + 1 : eol;
            continue;
        }

        // Blockquote
        if (line[0] == '>' && (line[1] == ' ' || line[1] == 0)) {
            HCAT("<blockquote>");
            HCAT(line + (line[1] == ' ' ? 2 : 1));
            HCAT("</blockquote>");
            p = (*eol) ? eol + 1 : eol;
            continue;
        }

        // Unordered list
        if ((line[0] == '-' || line[0] == '*') && line[1] == ' ') {
            if (!in_list) { HCAT("<ul>"); in_list = 1; }
            HCAT("<li>");
            HCAT(line + 2);
            HCAT("</li>");
            p = (*eol) ? eol + 1 : eol;
            continue;
        }

        // Ordered list (1. 2. etc)
        if (line[0] >= '0' && line[0] <= '9') {
            const char *dot = strchr(line, '.');
            if (dot && dot < line + 4 && dot[1] == ' ') {
                if (!in_list) { HCAT("<ul>"); in_list = 1; }
                HCAT("<li>");
                HCAT(dot + 2);
                HCAT("</li>");
                p = (*eol) ? eol + 1 : eol;
                continue;
            }
        }

        // Regular paragraph — apply inline formatting
        HCAT("<p>");
        const char *lp = line;
        while (*lp) {
            // Bold **text**
            if (lp[0] == '*' && lp[1] == '*') {
                const char *end = strstr(lp + 2, "**");
                if (end) {
                    HCAT("<strong>");
                    size_t slen = (size_t)(end - lp - 2);
                    if (pos + slen < cap) { memcpy(html + pos, lp + 2, slen); pos += slen; }
                    HCAT("</strong>");
                    lp = end + 2;
                    continue;
                }
            }
            // Italic *text*
            if (lp[0] == '*' && lp[1] != '*') {
                const char *end = strchr(lp + 1, '*');
                if (end && end != lp + 1) {
                    HCAT("<em>");
                    size_t slen = (size_t)(end - lp - 1);
                    if (pos + slen < cap) { memcpy(html + pos, lp + 1, slen); pos += slen; }
                    HCAT("</em>");
                    lp = end + 1;
                    continue;
                }
            }
            // Inline code `text`
            if (lp[0] == '`') {
                const char *end = strchr(lp + 1, '`');
                if (end) {
                    HCAT("<code>");
                    size_t slen = (size_t)(end - lp - 1);
                    if (pos + slen < cap) { memcpy(html + pos, lp + 1, slen); pos += slen; }
                    HCAT("</code>");
                    lp = end + 1;
                    continue;
                }
            }
            // Link [text](url)
            if (lp[0] == '[') {
                const char *cb = strchr(lp, ']');
                if (cb && cb[1] == '(') {
                    const char *cp = strchr(cb + 2, ')');
                    if (cp) {
                        HCAT("<a href='");
                        size_t ulen = (size_t)(cp - cb - 2);
                        if (pos + ulen < cap) { memcpy(html + pos, cb + 2, ulen); pos += ulen; }
                        HCAT("'>");
                        size_t tlen = (size_t)(cb - lp - 1);
                        if (pos + tlen < cap) { memcpy(html + pos, lp + 1, tlen); pos += tlen; }
                        HCAT("</a>");
                        lp = cp + 1;
                        continue;
                    }
                }
            }
            // HTML escape
            if (*lp == '<') { HCAT("&lt;"); lp++; }
            else if (*lp == '>') { HCAT("&gt;"); lp++; }
            else if (*lp == '&') { HCAT("&amp;"); lp++; }
            else { html[pos++] = *lp++; }
        }
        HCAT("</p>");
        p = (*eol) ? eol + 1 : eol;
    }

    if (in_list) HCAT("</ul>");
    if (in_code_block) HCAT("</code></pre>");
    HCAT("</body></html>");
    html[pos] = 0;

    #undef HCAT
    return html;
}

// Show artifact in a native WebKit pop-out window.
// Uses fork+exec to get a clean process — WebKit's XPC backend doesn't
// survive fork() in a multithreaded process.
static void show_artifact_window(const char *filepath, const char *title) {
    pid_t pid = fork();
    if (pid < 0) return;  // fork failed
    if (pid > 0) return;  // parent returns immediately

    // Child: exec ourselves with --artifact flag for a clean process
    execl(g_exe_path, g_exe_path, "--artifact", filepath, title, NULL);
    // If exec fails, fall back to opening in default browser
    execl("/usr/bin/open", "open", filepath, NULL);
    _exit(1);
}

// Entry point for --artifact mode: creates a native WebKit pop-out window
static int artifact_window_main(const char *filepath, const char *title) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];

        // Window dimensions
        NSRect frame = NSMakeRect(100, 100, 900, 700);
        NSWindow *window = [[NSWindow alloc]
            initWithContentRect:frame
            styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                      NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable)
            backing:NSBackingStoreBuffered
            defer:NO];

        NSString *winTitle = [NSString stringWithFormat:@"PRE — %s", title];
        [window setTitle:winTitle];
        [window setLevel:NSFloatingWindowLevel];
        [window setBackgroundColor:[NSColor colorWithRed:0.1 green:0.1 blue:0.18 alpha:1.0]];

        // WebKit view
        WKWebViewConfiguration *config = [[WKWebViewConfiguration alloc] init];
        WKWebpagePreferences *pagePrefs = [[WKWebpagePreferences alloc] init];
        pagePrefs.allowsContentJavaScript = YES;
        config.defaultWebpagePreferences = pagePrefs;
        WKWebView *webView = [[WKWebView alloc] initWithFrame:window.contentView.bounds
                                                configuration:config];
        webView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        [window.contentView addSubview:webView];

        // Load the HTML file. We use loadHTMLString with an HTTPS base URL so
        // external CDN resources (Chart.js, Leaflet, etc.) load correctly.
        // Local file:// image references (from image_generate) are converted to
        // base64 data URIs before loading, since WKWebView blocks cross-origin file access.
        NSString *filePath = [NSString stringWithUTF8String:filepath];
        NSError *readErr = nil;
        NSMutableString *htmlContent = [NSMutableString stringWithContentsOfFile:filePath
                                        encoding:NSUTF8StringEncoding
                                        error:&readErr];
        if (htmlContent) {
            // Convert file:// image references to base64 data URIs
            NSRegularExpression *fileRegex = [NSRegularExpression
                regularExpressionWithPattern:@"(src=['\"])file://([^'\"]+)(['\"])"
                options:0 error:nil];
            NSArray *matches = [fileRegex matchesInString:htmlContent
                options:0 range:NSMakeRange(0, htmlContent.length)];
            // Process in reverse order so ranges stay valid after replacements
            for (NSTextCheckingResult *match in [matches reverseObjectEnumerator]) {
                NSString *prefix = [htmlContent substringWithRange:[match rangeAtIndex:1]];
                NSString *localPath = [htmlContent substringWithRange:[match rangeAtIndex:2]];
                NSString *suffix = [htmlContent substringWithRange:[match rangeAtIndex:3]];
                NSData *imgData = [NSData dataWithContentsOfFile:localPath];
                if (imgData) {
                    NSString *ext = [localPath pathExtension].lowercaseString;
                    NSString *mime = @"image/png";
                    if ([ext isEqualToString:@"jpg"] || [ext isEqualToString:@"jpeg"])
                        mime = @"image/jpeg";
                    else if ([ext isEqualToString:@"gif"]) mime = @"image/gif";
                    else if ([ext isEqualToString:@"webp"]) mime = @"image/webp";
                    NSString *b64 = [imgData base64EncodedStringWithOptions:0];
                    NSString *dataURI = [NSString stringWithFormat:@"%@data:%@;base64,%@%@",
                                         prefix, mime, b64, suffix];
                    [htmlContent replaceCharactersInRange:[match range] withString:dataURI];
                }
            }
            NSURL *baseURL = [NSURL URLWithString:@"https://localhost/"];
            [webView loadHTMLString:htmlContent baseURL:baseURL];
        } else {
            NSURL *fileURL = [NSURL fileURLWithPath:filePath];
            [webView loadFileURL:fileURL
                allowingReadAccessToURL:[fileURL URLByDeletingLastPathComponent]];
        }

        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        // Run until window is closed
        [[NSNotificationCenter defaultCenter]
            addObserverForName:NSWindowWillCloseNotification
            object:window queue:nil
            usingBlock:^(NSNotification *note __attribute__((unused))) {
                [NSApp terminate:nil];
            }];

        [NSApp run];
    }
    return 0;
}

// Entry point for --pdf mode: renders HTML to PDF via WebKit
// Uses WKWebView's createPDFWithConfiguration (macOS 13+)
static int pdf_export_main(const char *html_path, const char *pdf_path) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];

        NSString *filePath = [NSString stringWithUTF8String:html_path];
        NSString *outPath = [NSString stringWithUTF8String:pdf_path];
        NSError *readErr = nil;
        NSMutableString *htmlContent = [NSMutableString stringWithContentsOfFile:filePath
                                        encoding:NSUTF8StringEncoding
                                        error:&readErr];
        if (!htmlContent) {
            fprintf(stderr, "pdf: cannot read %s\n", html_path);
            return 1;
        }

        // Convert file:// image references to base64 data URIs
        NSRegularExpression *fileRegex = [NSRegularExpression
            regularExpressionWithPattern:@"(src=['\"])file://([^'\"]+)(['\"])"
            options:0 error:nil];
        NSArray *matches = [fileRegex matchesInString:htmlContent
            options:0 range:NSMakeRange(0, htmlContent.length)];
        for (NSTextCheckingResult *match in [matches reverseObjectEnumerator]) {
            NSString *prefix = [htmlContent substringWithRange:[match rangeAtIndex:1]];
            NSString *localPath = [htmlContent substringWithRange:[match rangeAtIndex:2]];
            NSString *suffix = [htmlContent substringWithRange:[match rangeAtIndex:3]];
            NSData *imgData = [NSData dataWithContentsOfFile:localPath];
            if (imgData) {
                NSString *ext = [localPath pathExtension].lowercaseString;
                NSString *mime = @"image/png";
                if ([ext isEqualToString:@"jpg"] || [ext isEqualToString:@"jpeg"])
                    mime = @"image/jpeg";
                else if ([ext isEqualToString:@"gif"]) mime = @"image/gif";
                else if ([ext isEqualToString:@"webp"]) mime = @"image/webp";
                NSString *b64 = [imgData base64EncodedStringWithOptions:0];
                NSString *dataURI = [NSString stringWithFormat:@"%@data:%@;base64,%@%@",
                                     prefix, mime, b64, suffix];
                [htmlContent replaceCharactersInRange:[match range] withString:dataURI];
            }
        }

        // Create an off-screen WebKit view for rendering
        WKWebViewConfiguration *config = [[WKWebViewConfiguration alloc] init];
        WKWebpagePreferences *pagePrefs = [[WKWebpagePreferences alloc] init];
        pagePrefs.allowsContentJavaScript = YES;
        config.defaultWebpagePreferences = pagePrefs;

        // Use a large frame so content renders at full width
        NSRect frame = NSMakeRect(0, 0, 1024, 768);
        WKWebView *webView = [[WKWebView alloc] initWithFrame:frame configuration:config];

        // Load HTML with HTTPS base URL for CDN resources
        NSURL *baseURL = [NSURL URLWithString:@"https://localhost/"];
        [webView loadHTMLString:htmlContent baseURL:baseURL];

        // Wait for navigation to finish, then export PDF
        // Use a simple polling approach with the run loop
        __block BOOL done = NO;
        __block int result = 0;

        // Use KVO on loading property
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.5 * NSEC_PER_SEC)),
                       dispatch_get_main_queue(), ^{
            // Poll until loaded (check every 0.3s, max 15s)
            __block int polls = 0;
            [NSTimer scheduledTimerWithTimeInterval:0.3 repeats:YES block:^(NSTimer *t) {
                polls++;
                if (!webView.loading || polls > 50) {
                    [t invalidate];

                    // Give JS a moment to execute after load
                    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.0 * NSEC_PER_SEC)),
                                   dispatch_get_main_queue(), ^{
                        // Export PDF — A4 size in points (595.28 x 841.89)
                        if (@available(macOS 13.0, *)) {
                            WKPDFConfiguration *pdfConfig = [[WKPDFConfiguration alloc] init];
                            pdfConfig.rect = CGRectMake(0, 0, 595.28, 841.89);

                            [webView createPDFWithConfiguration:pdfConfig
                                             completionHandler:^(NSData *pdfData, NSError *pdfErr) {
                                if (pdfData && !pdfErr) {
                                    [pdfData writeToFile:outPath atomically:YES];
                                    fprintf(stderr, "pdf: exported %lu bytes to %s\n",
                                            (unsigned long)pdfData.length, pdf_path);
                                    result = 0;
                                } else {
                                    fprintf(stderr, "pdf: export failed: %s\n",
                                            pdfErr.localizedDescription.UTF8String ?: "unknown error");
                                    result = 1;
                                }
                                done = YES;
                                [NSApp terminate:nil];
                            }];
                        } else {
                            fprintf(stderr, "pdf: requires macOS 13+\n");
                            result = 1;
                            done = YES;
                            [NSApp terminate:nil];
                        }
                    });
                }
            }];
        });

        // Set a hard timeout
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(20 * NSEC_PER_SEC)),
                       dispatch_get_main_queue(), ^{
            if (!done) {
                fprintf(stderr, "pdf: timeout waiting for render\n");
                [NSApp terminate:nil];
            }
        });

        [NSApp run];
        return result;
    }
}

// Helper: export artifact to PDF using fork+exec --pdf
static int export_to_pdf(const char *html_path, const char *pdf_path) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        // Child
        execl(g_exe_path, g_exe_path, "--pdf", html_path, pdf_path, NULL);
        _exit(1);
    }
    // Parent: wait for child
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        struct stat st;
        if (stat(pdf_path, &st) == 0 && st.st_size > 0) return 0;
    }
    return -1;
}

// ============================================================================
// ComfyUI Image Generation
// ============================================================================

// Check if ComfyUI is installed and load config
static void comfyui_load_config(void) {
    const char *home = getenv("HOME");
    if (!home) return;
    char cfg_path[PATH_MAX];
    snprintf(cfg_path, sizeof(cfg_path), "%s/.pre/comfyui.json", home);
    FILE *f = fopen(cfg_path, "r");
    if (!f) return;
    char buf[2048];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);

    if (strstr(buf, "\"installed\":true") || strstr(buf, "\"installed\": true")) {
        g_comfyui_installed = 1;
        // Extract checkpoint name
        char ckpt[256];
        if (json_extract_str(buf, "checkpoint", ckpt, sizeof(ckpt)) > 0)
            strlcpy(g_comfyui_checkpoint, ckpt, sizeof(g_comfyui_checkpoint));
        // Extract port
        char port_str[16];
        if (json_extract_str(buf, "port", port_str, sizeof(port_str)) > 0) {
            int p = atoi(port_str);
            if (p > 0 && p < 65536) g_comfyui_port = p;
        }
    }
}

// Check if ComfyUI is running on the configured port
static int comfyui_is_running(void) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "curl -sf http://127.0.0.1:%d/system_stats >/dev/null 2>&1", g_comfyui_port);
    return system(cmd) == 0;
}

// Start ComfyUI as a background process
static int comfyui_start(void) {
    if (comfyui_is_running()) return 0;

    const char *home = getenv("HOME");
    if (!home) return -1;

    char venv_python[PATH_MAX], comfyui_main[PATH_MAX], log_path[PATH_MAX];
    snprintf(venv_python, sizeof(venv_python), "%s/.pre/comfyui-venv/bin/python3", home);
    snprintf(comfyui_main, sizeof(comfyui_main), "%s/.pre/comfyui/main.py", home);
    snprintf(log_path, sizeof(log_path), "%s/.pre/comfyui.log", home);

    // Verify files exist
    struct stat st;
    if (stat(venv_python, &st) != 0 || stat(comfyui_main, &st) != 0) return -1;

    printf(ANSI_DIM "  [starting ComfyUI on port %d...]" ANSI_RESET "\n", g_comfyui_port);

    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        // Child: redirect stdout/stderr to log, start ComfyUI
        int logfd = open(log_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (logfd >= 0) { dup2(logfd, STDOUT_FILENO); dup2(logfd, STDERR_FILENO); close(logfd); }
        setsid();  // detach from terminal

        char port_str[16];
        snprintf(port_str, sizeof(port_str), "%d", g_comfyui_port);
        execl(venv_python, "python3", comfyui_main,
              "--listen", "127.0.0.1", "--port", port_str,
              "--force-fp16", NULL);
        _exit(1);
    }

    g_comfyui_pid = pid;

    // Poll until ready (max 60s — first launch loads model into GPU)
    for (int i = 0; i < 120; i++) {
        usleep(500000);  // 500ms
        if (comfyui_is_running()) {
            printf(ANSI_DIM "  [ComfyUI ready after %ds]" ANSI_RESET "\n", (i + 1) / 2);
            return 0;
        }
        if (i % 4 == 0) {
            printf("\r" ANSI_DIM "  [ComfyUI loading model... %ds]" ANSI_RESET, (i + 1) / 2);
            fflush(stdout);
        }
    }
    printf("\n" ANSI_YELLOW "  [ComfyUI failed to start — check ~/.pre/comfyui.log]" ANSI_RESET "\n");
    return -1;
}

// Ensure ComfyUI is running, starting it if necessary
static int comfyui_ensure(void) {
    if (!g_comfyui_installed) return -1;
    if (comfyui_is_running()) return 0;
    return comfyui_start();
}

// Stop ComfyUI if we started it
static void comfyui_stop(void) {
    if (g_comfyui_pid > 0) {
        kill(g_comfyui_pid, SIGTERM);
        // Give it 2s to shut down gracefully, then force kill
        for (int i = 0; i < 20; i++) {
            int status;
            if (waitpid(g_comfyui_pid, &status, WNOHANG) != 0) goto done;
            usleep(100000);
        }
        kill(g_comfyui_pid, SIGKILL);
        waitpid(g_comfyui_pid, NULL, 0);
    done:
        g_comfyui_pid = 0;
    }
}

// Generate an image via ComfyUI. Returns 0 on success, -1 on failure.
// output_path receives the path to the generated PNG.
static int comfyui_generate(const char *prompt, const char *negative,
                            int width, int height, char *output_path, size_t path_sz) {
    if (comfyui_ensure() != 0) return -1;

    const char *home = getenv("HOME");
    if (!home) return -1;

    // Build filename prefix for this generation
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "pre_%ld", (long)time(NULL));

    // Detect if using SDXL Turbo (speed mode) vs a full SDXL model (quality mode).
    // Turbo is trained at 512x512 with 4 steps; full SDXL at 1024x1024 with 20-30 steps.
    int is_turbo = (strstr(g_comfyui_checkpoint, "turbo") != NULL);

    if (is_turbo) {
        if (width <= 0) width = 512;
        if (height <= 0) height = 512;
        if (width > 1024) width = 1024;
        if (height > 1024) height = 1024;
    } else {
        if (width <= 0) width = 1024;
        if (height <= 0) height = 1024;
        if (width > 1536) width = 1536;
        if (height > 1536) height = 1536;
    }

    // JSON-escape the prompt (replace " with \", newlines with \n)
    char esc_prompt[2048] = {0};
    int ei = 0;
    for (int i = 0; prompt[i] && ei < (int)sizeof(esc_prompt) - 4; i++) {
        if (prompt[i] == '"') { esc_prompt[ei++] = '\\'; esc_prompt[ei++] = '"'; }
        else if (prompt[i] == '\n') { esc_prompt[ei++] = '\\'; esc_prompt[ei++] = 'n'; }
        else if (prompt[i] == '\\') { esc_prompt[ei++] = '\\'; esc_prompt[ei++] = '\\'; }
        else esc_prompt[ei++] = prompt[i];
    }

    char esc_neg[1024] = {0};
    if (negative && negative[0]) {
        ei = 0;
        for (int i = 0; negative[i] && ei < (int)sizeof(esc_neg) - 4; i++) {
            if (negative[i] == '"') { esc_neg[ei++] = '\\'; esc_neg[ei++] = '"'; }
            else if (negative[i] == '\n') { esc_neg[ei++] = '\\'; esc_neg[ei++] = 'n'; }
            else esc_neg[ei++] = negative[i];
        }
    } else {
        strlcpy(esc_neg,
            "blurry, low quality, distorted, deformed, disfigured, "
            "bad anatomy, wrong proportions, extra limbs, mutated hands, "
            "ugly face, distorted face, malformed face, crossed eyes, "
            "watermark, text, signature, jpeg artifacts",
            sizeof(esc_neg));
    }

    long seed = (long)arc4random();

    // Workflow parameters adapt to the checkpoint type
    int steps = is_turbo ? 4 : 25;
    double cfg = is_turbo ? 1.0 : 5.5;
    const char *sampler = is_turbo ? "euler_ancestral" : "dpmpp_2m";
    const char *scheduler = is_turbo ? "normal" : "karras";

    // Build the workflow JSON
    char workflow[8192];
    snprintf(workflow, sizeof(workflow),
        "{"
        "\"4\":{\"class_type\":\"CheckpointLoaderSimple\",\"inputs\":{\"ckpt_name\":\"%s\"}},"
        "\"5\":{\"class_type\":\"EmptyLatentImage\",\"inputs\":{\"width\":%d,\"height\":%d,\"batch_size\":1}},"
        "\"6\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"%s\",\"clip\":[\"4\",1]}},"
        "\"7\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"%s\",\"clip\":[\"4\",1]}},"
        "\"3\":{\"class_type\":\"KSampler\",\"inputs\":{"
          "\"seed\":%ld,\"steps\":%d,\"cfg\":%.1f,\"sampler_name\":\"%s\","
          "\"scheduler\":\"%s\",\"denoise\":1.0,"
          "\"model\":[\"4\",0],\"positive\":[\"6\",0],\"negative\":[\"7\",0],\"latent_image\":[\"5\",0]}},"
        "\"8\":{\"class_type\":\"VAEDecode\",\"inputs\":{\"samples\":[\"3\",0],\"vae\":[\"4\",2]}},"
        "\"9\":{\"class_type\":\"SaveImage\",\"inputs\":{\"filename_prefix\":\"%s\",\"images\":[\"8\",0]}}"
        "}",
        g_comfyui_checkpoint, width, height,
        esc_prompt, esc_neg, seed, steps, cfg, sampler, scheduler, prefix);

    // POST to /prompt
    char post_body[16384];
    snprintf(post_body, sizeof(post_body), "{\"prompt\":%s}", workflow);

    // Write post body to temp file for curl
    char tmp_path[PATH_MAX];
    snprintf(tmp_path, sizeof(tmp_path), "%s/.pre/comfyui_request.json", home);
    FILE *tf = fopen(tmp_path, "w");
    if (!tf) return -1;
    fputs(post_body, tf);
    fclose(tf);

    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "curl -sf -X POST http://127.0.0.1:%d/prompt "
        "-H 'Content-Type: application/json' "
        "-d @%s 2>/dev/null",
        g_comfyui_port, tmp_path);

    FILE *proc = popen(cmd, "r");
    if (!proc) { unlink(tmp_path); return -1; }
    char resp[4096] = {0};
    fread(resp, 1, sizeof(resp) - 1, proc);
    pclose(proc);
    unlink(tmp_path);

    // Extract prompt_id from response
    char prompt_id[128] = {0};
    json_extract_str(resp, "prompt_id", prompt_id, sizeof(prompt_id));
    if (!prompt_id[0]) {
        fprintf(stderr, "comfyui: no prompt_id in response: %s\n", resp);
        return -1;
    }

    // Poll /history/{prompt_id} until complete (max 300s for 25-step generation at 1024x1024)
    printf(ANSI_DIM "  [generating image...]" ANSI_RESET);
    fflush(stdout);

    char filename[256] = {0};
    for (int poll = 0; poll < 600; poll++) {
        usleep(500000);
        if (poll % 4 == 0) {
            printf("\r" ANSI_DIM "  [generating image... %ds]" ANSI_RESET, (poll + 1) / 2);
            fflush(stdout);
        }

        snprintf(cmd, sizeof(cmd),
            "curl -sf http://127.0.0.1:%d/history/%s 2>/dev/null",
            g_comfyui_port, prompt_id);
        proc = popen(cmd, "r");
        if (!proc) continue;
        char hist[8192] = {0};
        fread(hist, 1, sizeof(hist) - 1, proc);
        pclose(proc);

        // Check if our prompt_id exists in the response (means it completed)
        if (!strstr(hist, prompt_id)) continue;

        // Extract filename from outputs.9.images[0].filename
        char *fname_key = strstr(hist, "\"filename\":");
        if (fname_key) {
            json_extract_str(fname_key, "filename", filename, sizeof(filename));
            if (filename[0]) break;
        }
    }
    printf("\r\033[K");

    if (!filename[0]) {
        printf(ANSI_YELLOW "  [image generation timed out]" ANSI_RESET "\n");
        return -1;
    }

    // Copy the image from ComfyUI output to artifacts directory
    char src_path[PATH_MAX];
    snprintf(src_path, sizeof(src_path), "%s/.pre/comfyui/output/%s", home, filename);

    // Create date directory in artifacts
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    char date_dir[PATH_MAX];
    snprintf(date_dir, sizeof(date_dir), "%s/.pre/artifacts/%04d-%02d-%02d",
             home, tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday);
    mkdir(date_dir, 0755);

    // Build descriptive filename from prompt
    char safe_name[64] = {0};
    int si = 0;
    for (int i = 0; prompt[i] && si < 50; i++) {
        char c = prompt[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
            safe_name[si++] = c;
        else if (c == ' ' && si > 0 && safe_name[si-1] != '_')
            safe_name[si++] = '_';
    }
    if (!safe_name[0]) strlcpy(safe_name, "generated", sizeof(safe_name));

    snprintf(output_path, path_sz, "%s/%s.png", date_dir, safe_name);

    // Copy file
    snprintf(cmd, sizeof(cmd), "cp '%s' '%s'", src_path, output_path);
    if (system(cmd) != 0) {
        // Fallback: try the source path directly
        strlcpy(output_path, src_path, path_sz);
    }

    struct stat ist;
    if (stat(output_path, &ist) == 0) {
        printf(ANSI_DIM "  [image generated: %lld bytes]" ANSI_RESET "\n", (long long)ist.st_size);
        return 0;
    }
    return -1;
}

// Wrap non-HTML content types in a styled HTML page for WebKit rendering
static char *wrap_content_as_html(const char *content, const char *type, const char *title) {
    size_t clen = strlen(content);
    size_t cap = clen * 3 + 4096;
    char *html = malloc(cap);
    if (!html) return NULL;

    const char *style =
        "body{font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',sans-serif;"
        "background:#1a1a2e;color:#e0e0e0;margin:0;padding:24px;line-height:1.6}"
        "h1{color:#00d4ff;font-size:1.3em;margin:0 0 16px 0;padding-bottom:8px;"
        "border-bottom:1px solid #333}"
        "pre{background:#0d0d1a;border:1px solid #333;border-radius:8px;padding:16px;"
        "overflow-x:auto;font-family:'SF Mono',Menlo,monospace;font-size:13px;line-height:1.5}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #333;padding:8px 12px;text-align:left}"
        "th{background:#2a2a3e;color:#00d4ff;font-weight:600}"
        "tr:nth-child(even){background:#1f1f33}"
        ".badge{background:#00d4ff;color:#1a1a2e;padding:2px 8px;border-radius:3px;"
        "font-size:0.7em;font-weight:700;margin-left:8px}";

    if (strcmp(type, "csv") == 0) {
        // Render CSV as a styled table
        int pos = snprintf(html, cap,
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>%s</style></head><body>"
            "<h1>%s <span class='badge'>CSV</span></h1><table>", style, title);

        const char *p = content;
        int row = 0;
        while (*p) {
            const char *eol = strchr(p, '\n');
            if (!eol) eol = p + strlen(p);
            pos += snprintf(html + pos, cap - pos, "<tr>");
            const char *cell = p;
            const char *tag = (row == 0) ? "th" : "td";
            while (cell < eol) {
                const char *comma = cell;
                while (comma < eol && *comma != ',') comma++;
                pos += snprintf(html + pos, cap - pos, "<%s>", tag);
                size_t cellen = (size_t)(comma - cell);
                if ((size_t)pos + cellen < cap) { memcpy(html + pos, cell, cellen); pos += cellen; }
                pos += snprintf(html + pos, cap - pos, "</%s>", tag);
                cell = (comma < eol) ? comma + 1 : eol;
            }
            pos += snprintf(html + pos, cap - pos, "</tr>");
            p = (*eol) ? eol + 1 : eol;
            row++;
        }
        pos += snprintf(html + pos, cap - pos, "</table></body></html>");
        html[pos] = 0;

    } else if (strcmp(type, "json") == 0) {
        snprintf(html, cap,
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>%s .key{color:#64b5f6} .str{color:#81c784} .num{color:#ffb74d} "
            ".bool{color:#ce93d8} .null{color:#888}</style></head><body>"
            "<h1>%s <span class='badge'>JSON</span></h1>"
            "<pre id='json'></pre>"
            "<script>"
            "try{const d=JSON.parse(document.getElementById('json').textContent=%s);"
            "document.getElementById('json').innerHTML=JSON.stringify(d,null,2)"
            ".replace(/\"([^\"]+)\":/g,'<span class=key>\"$1\"</span>:')"
            ".replace(/: \"([^\"]*)\"/g,': <span class=str>\"$1\"</span>')"
            ".replace(/: (\\d+)/g,': <span class=num>$1</span>')"
            ".replace(/: (true|false)/g,': <span class=bool>$1</span>')"
            ".replace(/: (null)/g,': <span class=null>$1</span>')"
            "}catch(e){}</script></body></html>",
            style, title, content);

    } else if (strcmp(type, "svg") == 0) {
        snprintf(html, cap,
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>%s svg{max-width:100%%;height:auto}</style></head><body>"
            "<h1>%s <span class='badge'>SVG</span></h1>%s</body></html>",
            style, title, content);

    } else {
        // code / text — show in a pre block
        // HTML-escape the content
        size_t elen = 0;
        char *escaped = malloc(clen * 5 + 1);
        if (escaped) {
            for (size_t i = 0; i < clen; i++) {
                if (content[i] == '<') { memcpy(escaped + elen, "&lt;", 4); elen += 4; }
                else if (content[i] == '>') { memcpy(escaped + elen, "&gt;", 4); elen += 4; }
                else if (content[i] == '&') { memcpy(escaped + elen, "&amp;", 5); elen += 5; }
                else escaped[elen++] = content[i];
            }
            escaped[elen] = 0;
            snprintf(html, cap,
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<style>%s</style></head><body>"
                "<h1>%s <span class='badge'>%s</span></h1><pre>%s</pre></body></html>",
                style, title, type, escaped);
            free(escaped);
        } else {
            snprintf(html, cap,
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<style>%s</style></head><body><pre>%s</pre></body></html>",
                style, content);
        }
    }
    return html;
}

// Create artifact: writes to ~/.pre/artifacts/, opens in native pop-out window.
// If append_to is non-NULL, appends content to an existing artifact instead of creating new.
static int execute_artifact(const char *title, const char *content, const char *type,
                            const char *append_to, char *output, size_t output_sz) {
    const char *home = getenv("HOME");
    if (!home) return snprintf(output, output_sz, "Error: HOME not set");
    if (!content) return snprintf(output, output_sz, "Error: content required");
    if (!type) type = "text";

    // ---- APPEND MODE ----
    // Find the existing artifact and inject content before </body>
    if (append_to && append_to[0]) {
        int found = -1;
        for (int i = g_artifact_count - 1; i >= 0; i--) {
            if (strcasestr(g_artifacts[i].title, append_to)) { found = i; break; }
        }
        if (found < 0)
            return snprintf(output, output_sz, "Error: no artifact matching '%s' found", append_to);

        // Read existing HTML
        FILE *f = fopen(g_artifacts[found].path, "r");
        if (!f) return snprintf(output, output_sz, "Error: cannot read %s", g_artifacts[found].path);
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *existing = malloc(fsize + 1);
        fread(existing, 1, fsize, f);
        existing[fsize] = 0;
        fclose(f);

        // Find insertion point: before </body> or </html> or end of file
        char *insert_point = strstr(existing, "</body>");
        if (!insert_point) insert_point = strstr(existing, "</html>");
        if (!insert_point) insert_point = existing + fsize;

        g_artifacts[found].section_count++;
        int sec_num = g_artifacts[found].section_count;

        // Build the section wrapper with a visual divider
        f = fopen(g_artifacts[found].path, "w");
        if (!f) { free(existing); return snprintf(output, output_sz, "Error: cannot write %s", g_artifacts[found].path); }

        // Write everything before insertion point
        fwrite(existing, 1, insert_point - existing, f);

        // Inject section divider + content
        fprintf(f, "\n<!-- Section %d -->\n"
                   "<hr style='border:none;border-top:2px solid #444;margin:32px 0'>\n"
                   "<section id='section-%d'>\n%s\n</section>\n",
                sec_num, sec_num, content);

        // Write the rest (</body></html>)
        fputs(insert_point, f);
        fclose(f);
        free(existing);

        struct stat art_st;
        long long new_size = 0;
        if (stat(g_artifacts[found].path, &art_st) == 0) new_size = (long long)art_st.st_size;

        printf(ANSI_DIM "  [artifact section %d appended: %lld bytes total]" ANSI_RESET "\n", sec_num, new_size);

        // Refresh the viewer window
        show_artifact_window(g_artifacts[found].path, g_artifacts[found].title);

        printf(ANSI_CYAN "  ◆ Artifact updated: %s" ANSI_RESET " (section %d)\n",
               g_artifacts[found].title, sec_num);

        return snprintf(output, output_sz,
            "Section %d appended to artifact \"%s\" (%lld bytes total).\n"
            "The updated artifact is displayed in the pop-out window.",
            sec_num, g_artifacts[found].title, new_size);
    }

    // ---- CREATE MODE (original path) ----
    if (!title || !title[0]) return snprintf(output, output_sz, "Error: title required");

    // Create date directory
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    char date_dir[PATH_MAX];
    snprintf(date_dir, sizeof(date_dir), "%s/.pre/artifacts/%04d-%02d-%02d",
             home, tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday);
    mkdir(date_dir, 0755);

    // Sanitize title for filename
    char safe_title[128];
    int si = 0;
    for (int i = 0; title[i] && si < (int)sizeof(safe_title) - 1; i++) {
        char c = title[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_')
            safe_title[si++] = c;
        else if (c == ' ')
            safe_title[si++] = '_';
    }
    safe_title[si] = 0;
    if (!safe_title[0]) strlcpy(safe_title, "artifact", sizeof(safe_title));

    // All artifacts become HTML for the WebKit viewer
    char filepath[PATH_MAX];
    snprintf(filepath, sizeof(filepath), "%s/%s.html", date_dir, safe_title);

    // Generate HTML content based on type
    FILE *f = fopen(filepath, "w");
    if (!f) return snprintf(output, output_sz, "Error: cannot create %s", filepath);

    if (strcmp(type, "html") == 0) {
        // HTML: wrap bare fragments, pass through full documents
        if (!strstr(content, "<html") && !strstr(content, "<!DOCTYPE")) {
            fprintf(f, "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;"
                "background:#1a1a2e;color:#e0e0e0;margin:0;padding:24px;line-height:1.6}"
                "a{color:#64b5f6}code{background:#2a2a3e;padding:2px 6px;border-radius:3px}"
                "pre{background:#0d0d1a;border:1px solid #333;border-radius:6px;padding:16px}"
                "</style></head><body>%s</body></html>", content);
        } else {
            fputs(content, f);
        }
    } else if (strcmp(type, "markdown") == 0 || strcmp(type, "md") == 0) {
        char *html = markdown_to_html(content);
        if (html) { fputs(html, f); free(html); }
        else fputs(content, f);
    } else {
        // csv, json, svg, code, text — wrap in styled HTML
        char *html = wrap_content_as_html(content, type, title);
        if (html) { fputs(html, f); free(html); }
        else fputs(content, f);
    }
    fclose(f);

    // Also save raw content for non-HTML types
    if (strcmp(type, "html") != 0 && strcmp(type, "markdown") != 0 && strcmp(type, "md") != 0) {
        const char *raw_ext = "txt";
        if (strcmp(type, "csv") == 0) raw_ext = "csv";
        else if (strcmp(type, "json") == 0) raw_ext = "json";
        else if (strcmp(type, "svg") == 0) raw_ext = "svg";
        char raw_path[PATH_MAX];
        snprintf(raw_path, sizeof(raw_path), "%s/%s.%s", date_dir, safe_title, raw_ext);
        FILE *rf = fopen(raw_path, "w");
        if (rf) { fputs(content, rf); fclose(rf); }
    }

    // Track in session
    if (g_artifact_count < MAX_SESSION_ARTIFACTS) {
        strlcpy(g_artifacts[g_artifact_count].path, filepath,
                sizeof(g_artifacts[g_artifact_count].path));
        strlcpy(g_artifacts[g_artifact_count].title, title,
                sizeof(g_artifacts[g_artifact_count].title));
        strlcpy(g_artifacts[g_artifact_count].type, type,
                sizeof(g_artifacts[g_artifact_count].type));
        g_artifact_count++;
    }

    // Verify file was written and check for truncation
    struct stat art_st;
    int truncated = 0;
    if (stat(filepath, &art_st) == 0) {
        printf(ANSI_DIM "  [artifact: %lld bytes written]" ANSI_RESET "\n", (long long)art_st.st_size);

        // Detect truncated artifacts
        size_t content_len = strlen(content);
        if (strcmp(type, "html") == 0) {
            // HTML artifacts should have closing tags and reasonable size
            if (content_len < 200 && (strstr(content, "<html") || strstr(content, "<!DOCTYPE"))) {
                truncated = 1; // Full doc declared but tiny content
            } else if (content_len > 50 && !strstr(content, "</html") &&
                       (strstr(content, "<html") || strstr(content, "<!DOCTYPE"))) {
                truncated = 2; // Has opening but no closing
            }
        }
    }

    if (truncated) {
        printf(ANSI_YELLOW "  [warning: artifact content appears truncated (%zu bytes)]" ANSI_RESET "\n",
               strlen(content));
        return snprintf(output, output_sz,
            "ERROR: Artifact content was truncated — only %zu bytes received, missing </html> closing tag. "
            "This usually means double quotes in HTML attributes broke the JSON string. "
            "Please retry: use SINGLE QUOTES for all HTML attributes (e.g. <div class='foo'> not <div class=\"foo\">), "
            "and ensure the content is complete with proper closing tags.",
            strlen(content));
    }

    // Open in native pop-out window
    show_artifact_window(filepath, title);

    // Print clickable artifact link using OSC 8 terminal hyperlinks
    printf(ANSI_CYAN "  ◆ Artifact: %s" ANSI_RESET "\n", title);
    printf("    \033]8;;file://%s\033\\", filepath);  // OSC 8 open
    printf(ANSI_BOLD ANSI_CYAN "▸ %s" ANSI_RESET, filepath);
    printf("\033]8;;\033\\\n");  // OSC 8 close
    return snprintf(output, output_sz, "Artifact created and displayed in pop-out window: \"%s\"\n"
                    "Saved at: %s\n"
                    "The user can see the pop-out window and can click the link in terminal to reopen it.",
                    title, filepath);
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
    g.last_num_ctx = 0;
    g.cumulative_gen_ms = 0;
    g.session_start_ms = now_ms();
    free(g.last_response); g.last_response = NULL;
    g.session_title[0] = 0;
    g._reserved_approve = 0;
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

// ============================================================================
// Connections — external service credentials
// ============================================================================

typedef struct {
    char name[32];       // e.g. "brave_search", "github", "wolfram", "google"
    char label[64];      // human-friendly label
    char key[512];       // API key / token (simple auth)
    int  active;         // 1 if configured
    int  is_oauth;       // 1 if this uses OAuth2
    char client_id[256];
    char client_secret[128];
    char access_token[2048];
    char refresh_token[512];
    long token_expiry;   // unix timestamp when access_token expires
} Connection;

static Connection g_connections[MAX_CONNECTIONS];
static int g_connections_count = 0;
static int g_connections_loaded = 0;

// Forward declarations for connection helpers
static Connection *get_connection(const char *name);
static void save_connections(void);

// ============================================================================
// Cron registry — recurring scheduled tasks
// ============================================================================

// Forward declarations for functions used by cron_check_and_run
static int json_extract_str(const char *json, const char *key, char *dst, size_t dsz);
static void session_save_turn(const char *session_id, const char *role, const char *content);
static void session_save_assistant_with_tool_calls(const char *session_id, const char *content,
                                                    const char *tool_calls_json);
static int send_request(const char *user_message, int max_tokens, const char *session_id);
static char *stream_response(int sock, int num_predict);
static char *handle_tool_calls(char *response);

typedef struct {
    char id[16];            // short unique id (hex)
    char schedule[64];      // 5-field cron: min hour dom month dow
    char prompt[1024];      // prompt to send to model when triggered
    char description[256];  // human-readable description
    int  enabled;           // 1 = active, 0 = disabled
    long created_at;        // unix timestamp
    long last_run_at;       // unix timestamp of last execution
    int  run_count;         // total executions
} CronJob;

static CronJob g_cron_jobs[MAX_CRON_JOBS];
static int g_cron_count = 0;
static double g_cron_last_check_ms = 0;

static const char *cron_path(void) {
    static char path[PATH_MAX];
    const char *home = getenv("HOME") ?: "/tmp";
    snprintf(path, sizeof(path), "%s/%s", home, CRON_FILE);
    return path;
}

static void cron_generate_id(char *buf, size_t bufsz) {
    unsigned int r;
    arc4random_buf(&r, sizeof(r));
    snprintf(buf, bufsz, "%08x", r);
}

// Save cron jobs to disk as JSON array
static void cron_save(void) {
    FILE *f = fopen(cron_path(), "w");
    if (!f) return;
    fprintf(f, "[\n");
    for (int i = 0; i < g_cron_count; i++) {
        CronJob *j = &g_cron_jobs[i];
        char *esc_prompt = json_escape_alloc(j->prompt);
        char *esc_desc = json_escape_alloc(j->description);
        fprintf(f, "%s{\"id\":\"%s\",\"schedule\":\"%s\",\"prompt\":\"%s\","
                   "\"description\":\"%s\",\"enabled\":%s,"
                   "\"created_at\":%ld,\"last_run_at\":%ld,\"run_count\":%d}",
                i > 0 ? ",\n" : "", j->id, j->schedule,
                esc_prompt ? esc_prompt : j->prompt,
                esc_desc ? esc_desc : j->description,
                j->enabled ? "true" : "false",
                j->created_at, j->last_run_at, j->run_count);
        free(esc_prompt);
        free(esc_desc);
    }
    fprintf(f, "\n]\n");
    fclose(f);
}

// Load cron jobs from disk
static void cron_load(void) {
    g_cron_count = 0;
    FILE *f = fopen(cron_path(), "r");
    if (!f) return;

    // Read entire file
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 2 || sz > 65536) { fclose(f); return; }
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = 0;
    fclose(f);

    // Simple parsing: find each {..."id":"..."...} object
    const char *p = buf;
    while (g_cron_count < MAX_CRON_JOBS) {
        const char *obj = strchr(p, '{');
        if (!obj) break;
        const char *obj_end = strchr(obj, '}');
        if (!obj_end) break;

        CronJob *j = &g_cron_jobs[g_cron_count];
        memset(j, 0, sizeof(CronJob));

        char tmp[2048];
        if (json_extract_str(obj, "id", tmp, sizeof(tmp))) strlcpy(j->id, tmp, sizeof(j->id));
        if (json_extract_str(obj, "schedule", tmp, sizeof(tmp))) strlcpy(j->schedule, tmp, sizeof(j->schedule));
        if (json_extract_str(obj, "prompt", tmp, sizeof(tmp))) strlcpy(j->prompt, tmp, sizeof(j->prompt));
        if (json_extract_str(obj, "description", tmp, sizeof(tmp))) strlcpy(j->description, tmp, sizeof(j->description));

        // Parse boolean/number fields
        if (strstr(obj, "\"enabled\":true")) j->enabled = 1;
        else if (strstr(obj, "\"enabled\":false")) j->enabled = 0;
        else j->enabled = 1; // default enabled

        char numtmp[32];
        if (json_extract_str(obj, "created_at", numtmp, sizeof(numtmp))) j->created_at = atol(numtmp);
        if (json_extract_str(obj, "last_run_at", numtmp, sizeof(numtmp))) j->last_run_at = atol(numtmp);
        if (json_extract_str(obj, "run_count", numtmp, sizeof(numtmp))) j->run_count = atoi(numtmp);

        if (j->id[0] && j->schedule[0] && j->prompt[0]) {
            g_cron_count++;
        }
        p = obj_end + 1;
    }
    free(buf);
}

// Parse a 5-field cron expression and check if it matches the given time.
// Fields: minute hour day-of-month month day-of-week
// Supports: numbers, *, */N (step), comma-separated lists, ranges (N-M)
static int cron_field_matches(const char *field, int value, int min_val, int max_val) {
    (void)min_val; (void)max_val;
    if (strcmp(field, "*") == 0) return 1;

    // */N step
    if (field[0] == '*' && field[1] == '/') {
        int step = atoi(field + 2);
        return step > 0 && (value % step) == 0;
    }

    // Comma-separated: "1,5,10"
    char buf[64];
    strlcpy(buf, field, sizeof(buf));
    char *tok = strtok(buf, ",");
    while (tok) {
        // Range: "1-5"
        char *dash = strchr(tok, '-');
        if (dash) {
            int lo = atoi(tok), hi = atoi(dash + 1);
            if (value >= lo && value <= hi) return 1;
        } else {
            if (atoi(tok) == value) return 1;
        }
        tok = strtok(NULL, ",");
    }
    return 0;
}

static int cron_matches_now(const char *schedule) {
    // Parse 5 fields
    char buf[64];
    strlcpy(buf, schedule, sizeof(buf));

    char *fields[5] = {NULL};
    char *p = buf;
    for (int i = 0; i < 5 && *p; i++) {
        while (*p == ' ') p++;
        fields[i] = p;
        while (*p && *p != ' ') p++;
        if (*p) *p++ = 0;
    }
    if (!fields[0] || !fields[1] || !fields[2] || !fields[3] || !fields[4])
        return 0;

    time_t now = time(NULL);
    struct tm *tm = localtime(&now);

    return cron_field_matches(fields[0], tm->tm_min, 0, 59) &&
           cron_field_matches(fields[1], tm->tm_hour, 0, 23) &&
           cron_field_matches(fields[2], tm->tm_mday, 1, 31) &&
           cron_field_matches(fields[3], tm->tm_mon + 1, 1, 12) &&
           cron_field_matches(fields[4], tm->tm_wday, 0, 6);
}

// Check if a schedule matches a specific time (not just "now")
static int cron_matches_time(const char *schedule, struct tm *tm) {
    char buf[64];
    strlcpy(buf, schedule, sizeof(buf));

    char *fields[5] = {NULL};
    char *p = buf;
    for (int i = 0; i < 5 && *p; i++) {
        while (*p == ' ') p++;
        fields[i] = p;
        while (*p && *p != ' ') p++;
        if (*p) *p++ = 0;
    }
    if (!fields[0] || !fields[1] || !fields[2] || !fields[3] || !fields[4])
        return 0;

    return cron_field_matches(fields[0], tm->tm_min, 0, 59) &&
           cron_field_matches(fields[1], tm->tm_hour, 0, 23) &&
           cron_field_matches(fields[2], tm->tm_mday, 1, 31) &&
           cron_field_matches(fields[3], tm->tm_mon + 1, 1, 12) &&
           cron_field_matches(fields[4], tm->tm_wday, 0, 6);
}

// Find the most recent time before `before` that matches the cron schedule.
// Walks backwards minute by minute, up to 7 days. Returns 0 if no match.
static time_t cron_previous_match_time(const char *schedule, time_t before) {
    time_t check = before - 60; // start one minute before
    // Zero out seconds
    struct tm *tm = localtime(&check);
    tm->tm_sec = 0;
    check = mktime(tm);

    int max_lookback = 7 * 24 * 60; // 7 days in minutes
    for (int i = 0; i < max_lookback; i++) {
        tm = localtime(&check);
        if (cron_matches_time(schedule, tm)) return check;
        check -= 60;
    }
    return 0;
}

// Check for cron jobs that should have fired while the system was down.
// Called once at startup after cron_load().
static void cron_check_missed(void) {
    if (g_cron_count == 0) return;

    time_t now = time(NULL);
    int any_ran = 0;

    for (int i = 0; i < g_cron_count; i++) {
        CronJob *j = &g_cron_jobs[i];
        if (!j->enabled) continue;

        time_t prev = cron_previous_match_time(j->schedule, now);
        if (prev == 0) continue;

        // Job was missed if the previous scheduled time is after its last run
        if (j->last_run_at < prev) {
            struct tm *missed_tm = localtime(&prev);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%b %d %I:%M %p", missed_tm);

            printf(ANSI_YELLOW "  ⏰ Missed cron: \"%s\" (was due %s)" ANSI_RESET "\n",
                   j->description[0] ? j->description : j->id, time_str);

            // Fire it
            session_save_turn(g.session_id, "user", j->prompt);
            free(g_native_tool_calls);
            g_native_tool_calls = NULL;

            int sock = send_request(j->prompt, g.max_tokens, g.session_id);
            if (sock >= 0) {
                printf("\n");
                char *response = stream_response(sock, g.max_tokens);
                if (response && (strlen(response) > 0 || g_native_tool_calls)) {
                    if (g_native_tool_calls) {
                        session_save_assistant_with_tool_calls(g.session_id, response, g_native_tool_calls);
                    } else {
                        session_save_turn(g.session_id, "assistant", response);
                    }
                    free(g.last_response);
                    g.last_response = strdup(response);
                }
                if (response && (g_native_tool_calls || strstr(response, "<tool_call>"))) {
                    char *final = handle_tool_calls(response);
                    free(final);
                } else {
                    free(response);
                }
                g.turn_count++;
            }

            j->last_run_at = now;
            j->run_count++;
            any_ran = 1;
        }
    }

    if (any_ran) {
        cron_save();
        printf("\n");
    }
}

// Check all cron jobs and execute any that are due.
// Called from the main loop. Only checks once per minute.
static void cron_check_and_run(void) {
    if (g_cron_count == 0) return;

    double now = now_ms();
    if (now - g_cron_last_check_ms < 60000) return; // check at most once per minute
    g_cron_last_check_ms = now;

    time_t now_t = time(NULL);
    int any_ran = 0;

    for (int i = 0; i < g_cron_count; i++) {
        CronJob *j = &g_cron_jobs[i];
        if (!j->enabled) continue;

        // Don't run more than once per minute (prevent double-fire)
        if (now_t - j->last_run_at < 60) continue;

        if (!cron_matches_now(j->schedule)) continue;

        // Fire this cron job
        printf("\n" ANSI_CYAN "  ⏰ [cron: %s]" ANSI_RESET "\n", j->description[0] ? j->description : j->id);
        printf(ANSI_DIM "  schedule: %s" ANSI_RESET "\n", j->schedule);

        // Save user turn and send to model
        session_save_turn(g.session_id, "user", j->prompt);

        // Clear stale native tool calls
        free(g_native_tool_calls);
        g_native_tool_calls = NULL;

        int sock = send_request(j->prompt, g.max_tokens, g.session_id);
        if (sock >= 0) {
            printf("\n");
            char *response = stream_response(sock, g.max_tokens);

            if (response && (strlen(response) > 0 || g_native_tool_calls)) {
                if (g_native_tool_calls) {
                    session_save_assistant_with_tool_calls(g.session_id, response, g_native_tool_calls);
                } else {
                    session_save_turn(g.session_id, "assistant", response);
                }
                free(g.last_response);
                g.last_response = strdup(response);
            }

            // Process tool calls if any
            if (response && (g_native_tool_calls || strstr(response, "<tool_call>"))) {
                char *final = handle_tool_calls(response);
                free(final);
            } else {
                free(response);
            }

            g.turn_count++;
        }

        // Update job state
        j->last_run_at = now_t;
        j->run_count++;
        any_ran = 1;
    }

    if (any_ran) {
        cron_save();
        printf("\n"); // blank line before next prompt
    }
}

static const char *connections_path(void) {
    static char path[PATH_MAX];
    const char *home = getenv("HOME") ?: "/tmp";
    snprintf(path, sizeof(path), "%s/%s", home, CONNECTIONS_FILE);
    return path;
}

// Find a JSON key in text, tolerating whitespace: matches "key" followed by optional spaces and ":"
// Returns pointer to the "key" or NULL. Use for scanning pretty-printed API JSON.
static const char *json_find_key(const char *json, const char *key) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = json;
    while ((p = strstr(p, needle)) != NULL) {
        const char *after = p + strlen(needle);
        while (*after == ' ' || *after == '\t' || *after == '\n' || *after == '\r') after++;
        if (*after == ':') return p;
        p++;
    }
    return NULL;
}

// Simple JSON key extractor — finds "key": "value" (with optional whitespace) and copies value into dst
static int json_extract_str(const char *json, const char *key, char *dst, size_t dsz) {
    // Search for "key" then find the colon and opening quote, tolerating whitespace
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return 0;
    p += strlen(needle);
    // Skip whitespace and colon
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != ':') return 0;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    // Handle both string values ("...") and bare numbers
    if (*p == '"') {
        p++; // skip opening quote
    } else {
        // Bare value (number, bool) — read until comma/whitespace/}
        size_t i = 0;
        while (*p && *p != ',' && *p != '}' && *p != ' ' && *p != '\n' && i < dsz - 1)
            dst[i++] = *p++;
        dst[i] = 0;
        return i > 0;
    }
    size_t i = 0;
    while (*p && *p != '"' && i < dsz - 1) {
        if (*p == '\\' && p[1]) { dst[i++] = p[1]; p += 2; }
        else dst[i++] = *p++;
    }
    dst[i] = 0;
    return 1;
}

static void load_connections(void) {
    g_connections_count = 0;
    g_connections_loaded = 1;

    // Pre-populate known service slots
    struct { const char *name; const char *label; int oauth; } services[] = {
        {"brave_search", "Brave Search API", 0},
        {"github",       "GitHub",           0},
        {"google",       "Google (Gmail/Drive/Docs)", 1},
        {"wolfram",      "Wolfram Alpha",    0},
        {"telegram",     "Telegram Bot",     0},
        {NULL, NULL, 0}
    };
    for (int i = 0; services[i].name; i++) {
        strlcpy(g_connections[i].name, services[i].name, sizeof(g_connections[i].name));
        strlcpy(g_connections[i].label, services[i].label, sizeof(g_connections[i].label));
        g_connections[i].key[0] = 0;
        g_connections[i].active = 0;
        g_connections[i].is_oauth = services[i].oauth;
        g_connections_count++;
    }

    FILE *f = fopen(connections_path(), "r");
    if (!f) return;

    char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);

    // Parse each known service key from JSON
    for (int i = 0; i < g_connections_count; i++) {
        Connection *c = &g_connections[i];
        if (c->is_oauth) {
            // OAuth services store multiple fields
            char field[64];
            snprintf(field, sizeof(field), "%s_client_id", c->name);
            json_extract_str(buf, field, c->client_id, sizeof(c->client_id));
            snprintf(field, sizeof(field), "%s_client_secret", c->name);
            json_extract_str(buf, field, c->client_secret, sizeof(c->client_secret));
            snprintf(field, sizeof(field), "%s_access_token", c->name);
            json_extract_str(buf, field, c->access_token, sizeof(c->access_token));
            snprintf(field, sizeof(field), "%s_refresh_token", c->name);
            json_extract_str(buf, field, c->refresh_token, sizeof(c->refresh_token));
            snprintf(field, sizeof(field), "%s_token_expiry", c->name);
            char expiry_s[32];
            if (json_extract_str(buf, field, expiry_s, sizeof(expiry_s)))
                c->token_expiry = atol(expiry_s);
            if (c->refresh_token[0]) c->active = 1;
        } else {
            char key_field[64];
            snprintf(key_field, sizeof(key_field), "%s_key", c->name);
            if (json_extract_str(buf, key_field, c->key, sizeof(c->key))) {
                if (c->key[0]) c->active = 1;
            }
        }
    }

    // Load additional Google accounts (google_work, google_personal, etc.)
    // Scan for any "google_*_refresh_token" keys in the JSON
    const char *scan = buf;
    while ((scan = strstr(scan, "\"google_")) != NULL) {
        scan++; // skip opening quote
        const char *end = strchr(scan, '"');
        if (!end) break;
        // Extract field name
        char field_name[128];
        size_t flen = end - scan;
        if (flen >= sizeof(field_name)) { scan = end; continue; }
        memcpy(field_name, scan, flen);
        field_name[flen] = 0;
        scan = end + 1;

        // Only care about refresh_token fields for additional accounts
        char *suffix = strstr(field_name, "_refresh_token");
        if (!suffix) continue;
        *suffix = 0; // field_name is now e.g. "google_work"

        // Skip the default "google" — already loaded above
        if (strcmp(field_name, "google") == 0) continue;
        // Skip if already loaded
        if (get_connection(field_name)) continue;
        // Must not exceed limit
        if (g_connections_count >= MAX_CONNECTIONS) break;

        // Create a new connection slot
        const char *short_label = field_name + 7; // skip "google_"
        Connection *c = &g_connections[g_connections_count++];
        memset(c, 0, sizeof(*c));
        strlcpy(c->name, field_name, sizeof(c->name));
        snprintf(c->label, sizeof(c->label), "Google (%s)", short_label);
        c->is_oauth = 1;

        // Load its OAuth fields
        char fld[64];
        snprintf(fld, sizeof(fld), "%s_client_id", c->name);
        json_extract_str(buf, fld, c->client_id, sizeof(c->client_id));
        snprintf(fld, sizeof(fld), "%s_client_secret", c->name);
        json_extract_str(buf, fld, c->client_secret, sizeof(c->client_secret));
        snprintf(fld, sizeof(fld), "%s_access_token", c->name);
        json_extract_str(buf, fld, c->access_token, sizeof(c->access_token));
        snprintf(fld, sizeof(fld), "%s_refresh_token", c->name);
        json_extract_str(buf, fld, c->refresh_token, sizeof(c->refresh_token));
        snprintf(fld, sizeof(fld), "%s_token_expiry", c->name);
        char exp_s[32];
        if (json_extract_str(buf, fld, exp_s, sizeof(exp_s)))
            c->token_expiry = atol(exp_s);
        if (c->refresh_token[0]) c->active = 1;
    }
}

static void save_connections(void) {
    FILE *f = fopen(connections_path(), "w");
    if (!f) return;
    chmod(connections_path(), 0600);

    fprintf(f, "{\n");
    int first = 1;
    for (int i = 0; i < g_connections_count; i++) {
        Connection *c = &g_connections[i];
        if (!c->active) continue;
        if (c->is_oauth) {
            // Save OAuth fields
            char *esc_cid = json_escape_alloc(c->client_id);
            char *esc_cs  = json_escape_alloc(c->client_secret);
            char *esc_at  = json_escape_alloc(c->access_token);
            char *esc_rt  = json_escape_alloc(c->refresh_token);
            if (!first) fprintf(f, ",\n");
            fprintf(f, "  \"%s_client_id\": \"%s\",\n", c->name, esc_cid ?: "");
            fprintf(f, "  \"%s_client_secret\": \"%s\",\n", c->name, esc_cs ?: "");
            fprintf(f, "  \"%s_access_token\": \"%s\",\n", c->name, esc_at ?: "");
            fprintf(f, "  \"%s_refresh_token\": \"%s\",\n", c->name, esc_rt ?: "");
            fprintf(f, "  \"%s_token_expiry\": \"%ld\"", c->name, c->token_expiry);
            free(esc_cid); free(esc_cs); free(esc_at); free(esc_rt);
        } else {
            if (!first) fprintf(f, ",\n");
            char *escaped = json_escape_alloc(c->key);
            fprintf(f, "  \"%s_key\": \"%s\"", c->name, escaped ?: "");
            free(escaped);
        }
        first = 0;
    }
    fprintf(f, "\n}\n");
    fclose(f);
}

static Connection *get_connection(const char *name) {
    for (int i = 0; i < g_connections_count; i++) {
        if (strcmp(g_connections[i].name, name) == 0) return &g_connections[i];
    }
    return NULL;
}

// Resolve a Google connection by optional account label.
// "account" param from tool call: NULL → default "google", "work" → "google_work", etc.
// Also accepts full names like "google_work" directly.
static Connection *get_google_connection(const char *account) {
    if (!account || !account[0]) return get_connection("google");
    // Try exact match first (e.g. "google_work")
    Connection *c = get_connection(account);
    if (c && c->is_oauth) return c;
    // Try prefixed (e.g. "work" → "google_work")
    char full[64];
    snprintf(full, sizeof(full), "google_%s", account);
    c = get_connection(full);
    if (c && c->is_oauth) return c;
    // Fall back to default
    return get_connection("google");
}

// Add a new Google account connection slot and return it
static Connection *add_google_account(const char *label) {
    if (g_connections_count >= MAX_CONNECTIONS) return NULL;
    Connection *c = &g_connections[g_connections_count++];
    memset(c, 0, sizeof(*c));
    snprintf(c->name, sizeof(c->name), "google_%s", label);
    snprintf(c->label, sizeof(c->label), "Google (%s)", label);
    c->is_oauth = 1;
    return c;
}

// List all active Google account names (for tool descriptions)
static int list_google_accounts(char *buf, size_t bufsz) {
    int off = 0;
    for (int i = 0; i < g_connections_count; i++) {
        Connection *c = &g_connections[i];
        if (!c->active || !c->is_oauth) continue;
        if (strncmp(c->name, "google", 6) != 0) continue;
        if (off > 0 && off < (int)bufsz - 2) buf[off++] = ',';
        const char *short_name = (strcmp(c->name, "google") == 0) ? "default" : c->name + 7;
        int n = snprintf(buf + off, bufsz - off, "%s", short_name);
        if (n > 0) off += n;
    }
    buf[off] = 0;
    return off;
}

// ============================================================================
// Telegram bot subprocess
// ============================================================================

static void start_telegram(void) {
    if (!g_connections_loaded) load_connections();
    Connection *tg = get_connection("telegram");
    if (!tg || !tg->active) return;

    // Find the pre-telegram binary next to this binary
    char tg_bin[PATH_MAX];
    char self[PATH_MAX] = {0};
    uint32_t sz = sizeof(self);
    if (_NSGetExecutablePath(self, &sz) == 0) {
        char real[PATH_MAX];
        if (realpath(self, real)) strlcpy(self, real, sizeof(self));
        char *slash = strrchr(self, '/');
        if (slash) {
            *slash = 0;
            snprintf(tg_bin, sizeof(tg_bin), "%s/pre-telegram", self);
        } else {
            snprintf(tg_bin, sizeof(tg_bin), "./pre-telegram");
        }
    } else {
        snprintf(tg_bin, sizeof(tg_bin), "./pre-telegram");
    }

    struct stat st;
    if (stat(tg_bin, &st) != 0 || !(st.st_mode & S_IXUSR)) {
        // Binary not found — silently skip
        return;
    }

    pid_t pid = fork();
    if (pid == 0) {
        // Child — redirect output to log file so it doesn't interleave with TUI
        const char *home = getenv("HOME") ?: "/tmp";
        char logpath[PATH_MAX];
        snprintf(logpath, sizeof(logpath), "%s/.pre/telegram.log", home);
        FILE *logf = fopen(logpath, "a");
        if (logf) {
            dup2(fileno(logf), STDOUT_FILENO);
            dup2(fileno(logf), STDERR_FILENO);
            fclose(logf);
        }
        char port_s[16];
        snprintf(port_s, sizeof(port_s), "%d", g.port);
        execl(tg_bin, "pre-telegram", "--port", port_s, NULL);
        _exit(1);
    } else if (pid > 0) {
        g_telegram_pid = pid;
        printf(ANSI_DIM "  Telegram bot active" ANSI_RESET "\n");
    }
}

// ============================================================================
// OAuth2 helpers
// ============================================================================

// Built-in Google OAuth credentials — users don't need their own Cloud project.
// Google documents that client secrets for "installed applications" (desktop/native)
// are not treated as confidential. Users just sign in via browser.
#define PRE_GOOGLE_CLIENT_ID "1062005591474-bj9c932m52vrvh5cl8fr02j1dti99jdh.apps.googleusercontent.com"
#define PRE_GOOGLE_CLIENT_SECRET "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"

#define GOOGLE_AUTH_URL "https://accounts.google.com/o/oauth2/v2/auth"
#define GOOGLE_TOKEN_URL "https://oauth2.googleapis.com/token"
#define GOOGLE_SCOPES "https://www.googleapis.com/auth/gmail.modify" \
    "%20https://www.googleapis.com/auth/gmail.compose" \
    "%20https://www.googleapis.com/auth/gmail.send" \
    "%20https://www.googleapis.com/auth/drive" \
    "%20https://www.googleapis.com/auth/documents"

// Tiny local HTTP server to catch OAuth callback — returns auth code
static int oauth_listen_for_code(int port, char *code_out, size_t code_sz) {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) return 0;

    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) { close(srv); return 0; }
    if (listen(srv, 1) < 0) { close(srv); return 0; }

    // Set a 120-second timeout
    struct timeval tv = { .tv_sec = 120 };
    setsockopt(srv, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    int client = accept(srv, NULL, NULL);
    close(srv);
    if (client < 0) return 0;

    char buf[4096];
    ssize_t n = read(client, buf, sizeof(buf) - 1);
    if (n <= 0) { close(client); return 0; }
    buf[n] = 0;

    // Extract code from "GET /?code=XXXX&..."
    char *cp = strstr(buf, "code=");
    if (cp) {
        cp += 5;
        size_t i = 0;
        while (cp[i] && cp[i] != '&' && cp[i] != ' ' && cp[i] != '\r' && i < code_sz - 1) {
            code_out[i] = cp[i];
            i++;
        }
        code_out[i] = 0;
    }

    // Check for error
    char *ep = strstr(buf, "error=");
    int got_code = (code_out[0] != 0);

    // Send a nice response page
    const char *body = got_code
        ? "<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
          "<h2 style='color:#22c55e'>&#10003; Connected to Google</h2>"
          "<p>You can close this tab and return to PRE.</p></body></html>"
        : "<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
          "<h2 style='color:#ef4444'>Authorization failed</h2>"
          "<p>Please try again in PRE.</p></body></html>";

    char resp[2048];
    int rlen = snprintf(resp, sizeof(resp),
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n%s", body);
    write(client, resp, rlen);
    close(client);

    (void)ep;
    return got_code;
}

// Exchange auth code for access + refresh tokens
static int oauth_exchange_code(Connection *c, const char *code, int port) {
    // Write POST body to temp file to avoid shell escaping issues with the code
    char tmp[128];
    snprintf(tmp, sizeof(tmp), "/tmp/pre_oauth_%d.txt", getpid());
    FILE *tf = fopen(tmp, "w");
    if (!tf) return 0;
    fprintf(tf, "code=%s&client_id=%s&client_secret=%s&redirect_uri=http://127.0.0.1:%d&grant_type=authorization_code",
            code, c->client_id, c->client_secret, port);
    fclose(tf);

    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "curl -s -X POST '%s' "
        "-H 'Content-Type: application/x-www-form-urlencoded' "
        "-d @%s 2>&1",
        GOOGLE_TOKEN_URL, tmp);

    FILE *p = popen(cmd, "r");
    if (!p) { remove(tmp); return 0; }

    char resp[8192];
    int rlen = 0;
    while (rlen < (int)sizeof(resp) - 1) {
        int ch = fgetc(p);
        if (ch == EOF) break;
        resp[rlen++] = (char)ch;
    }
    resp[rlen] = 0;
    pclose(p);
    remove(tmp);

    // Extract tokens from JSON response
    char expires_s[32] = {0};
    json_extract_str(resp, "access_token", c->access_token, sizeof(c->access_token));
    json_extract_str(resp, "refresh_token", c->refresh_token, sizeof(c->refresh_token));
    json_extract_str(resp, "expires_in", expires_s, sizeof(expires_s));

    if (c->access_token[0] && c->refresh_token[0]) {
        int expires_in = atoi(expires_s);
        if (expires_in <= 0) expires_in = 3600;
        c->token_expiry = (long)time(NULL) + expires_in;
        c->active = 1;
        return 1;
    }

    // Show error for debugging
    char error[256] = {0}, error_desc[512] = {0};
    json_extract_str(resp, "error", error, sizeof(error));
    json_extract_str(resp, "error_description", error_desc, sizeof(error_desc));
    if (error[0])
        printf("\n" ANSI_RED "  Google error: %s — %s" ANSI_RESET "\n", error, error_desc);
    else if (rlen > 0)
        printf("\n" ANSI_RED "  Exchange response (%d bytes): %.500s" ANSI_RESET "\n", rlen, resp);
    else
        printf("\n" ANSI_RED "  Exchange got empty response (curl may have failed)" ANSI_RESET "\n");

    return 0;
}

// Refresh an expired access token
static int oauth_refresh_token(Connection *c) {
    if (!c->refresh_token[0] || !c->client_id[0] || !c->client_secret[0]) return 0;

    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "curl -s -X POST '%s' "
        "-d 'refresh_token=%s' "
        "-d 'client_id=%s' "
        "-d 'client_secret=%s' "
        "-d 'grant_type=refresh_token' 2>/dev/null",
        GOOGLE_TOKEN_URL, c->refresh_token, c->client_id, c->client_secret);

    FILE *p = popen(cmd, "r");
    if (!p) return 0;

    char resp[8192];
    int rlen = 0;
    while (rlen < (int)sizeof(resp) - 1) {
        int ch = fgetc(p);
        if (ch == EOF) break;
        resp[rlen++] = (char)ch;
    }
    resp[rlen] = 0;
    pclose(p);

    char new_token[2048] = {0}, expires_s[32] = {0};
    json_extract_str(resp, "access_token", new_token, sizeof(new_token));
    json_extract_str(resp, "expires_in", expires_s, sizeof(expires_s));

    if (new_token[0]) {
        strlcpy(c->access_token, new_token, sizeof(c->access_token));
        int expires_in = atoi(expires_s);
        if (expires_in <= 0) expires_in = 3600;
        c->token_expiry = (long)time(NULL) + expires_in;
        save_connections();
        return 1;
    }
    return 0;
}

// Ensure we have a valid access token, refreshing if needed
static int oauth_ensure_token(Connection *c) {
    if (!c->active || !c->refresh_token[0]) return 0;
    // Refresh if expired or within 60 seconds of expiring
    if (c->access_token[0] && c->token_expiry > (long)time(NULL) + 60) return 1;
    printf(ANSI_DIM "  [refreshing Google token...]" ANSI_RESET "\n");
    return oauth_refresh_token(c);
}

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
    printf("\n");
    // Banner box — 48 inner columns between ║ walls
    printf(ANSI_BOLD ANSI_CYAN "  ╔════════════════════════════════════════════════╗\n" ANSI_RESET);
    if (strcmp(g_agent_name, "PRE") != 0) {
        char title[64];
        snprintf(title, sizeof(title), "  %s  —  Personal Reasoning Engine", g_agent_name);
        printf(ANSI_BOLD ANSI_CYAN "  ║" ANSI_RESET ANSI_BOLD "  %-46s" ANSI_RESET ANSI_BOLD ANSI_CYAN "║\n" ANSI_RESET, title);
    } else {
        printf(ANSI_BOLD ANSI_CYAN "  ║" ANSI_RESET ANSI_BOLD "  P R E  —  Personal Reasoning Engine           " ANSI_RESET ANSI_BOLD ANSI_CYAN "║\n" ANSI_RESET);
    }
    printf(ANSI_BOLD ANSI_CYAN "  ║" ANSI_RESET ANSI_DIM "  %-46s" ANSI_RESET ANSI_BOLD ANSI_CYAN "║\n" ANSI_RESET, MODEL_NAME);
    printf(ANSI_BOLD ANSI_CYAN "  ╚════════════════════════════════════════════════╝\n" ANSI_RESET);
    printf("\n");

    // Info block with dim labels
    printf(ANSI_DIM "  server  " ANSI_RESET ANSI_GREEN "localhost:%d" ANSI_RESET, g.port);
    printf(ANSI_DIM "   model  " ANSI_RESET "%s\n", g_model);

    if (g.project_name[0])
        printf(ANSI_DIM "  project " ANSI_RESET ANSI_BOLD "%s" ANSI_RESET
               ANSI_DIM "  (%s)" ANSI_RESET "\n", g.project_name, g.project_root);

    printf(ANSI_DIM "  channel " ANSI_RESET ANSI_CYAN "#%s" ANSI_RESET, g.channel[0] ? g.channel : "general");
    printf(ANSI_DIM "   cwd  " ANSI_RESET "%s\n", g.cwd);

    if (g_memory_count > 0)
        printf(ANSI_DIM "  memory  " ANSI_RESET "%d entries loaded\n", g_memory_count);

    // Show active connections
    if (!g_connections_loaded) load_connections();
    int n_active = 0;
    char svc_list[256] = "";
    int svc_off = 0;
    for (int i = 0; i < g_connections_count; i++) {
        if (g_connections[i].active) {
            if (n_active > 0) svc_off += snprintf(svc_list + svc_off, sizeof(svc_list) - svc_off, ", ");
            svc_off += snprintf(svc_list + svc_off, sizeof(svc_list) - svc_off, "%s", g_connections[i].label);
            n_active++;
        }
    }
    if (n_active > 0)
        printf(ANSI_DIM "  online  " ANSI_RESET ANSI_GREEN "%s" ANSI_RESET "\n", svc_list);
    else
        printf(ANSI_DIM "  online  " ANSI_RESET ANSI_DIM "none — /connections setup to add" ANSI_RESET "\n");

    printf("\n" ANSI_DIM "  /help for commands  •  /help tips for best practices" ANSI_RESET "\n\n");
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

// Strip <think>...</think> blocks from assistant responses before saving.
// Thinking blocks are enormous (500-1000+ tokens) and add no value when replayed
// in future turns — they just bloat the context and slow prefill.
static char *strip_thinking_blocks(const char *text) {
    // Quick check: no thinking blocks to strip
    if (!strstr(text, "<think>")) return NULL;

    size_t len = strlen(text);
    char *out = malloc(len + 1);
    size_t oi = 0;
    const char *p = text;

    while (*p) {
        char *ts = strstr(p, "<think>");
        if (!ts) {
            // No more think blocks — copy rest
            strcpy(out + oi, p);
            oi += strlen(p);
            break;
        }
        // Copy everything before <think>
        size_t before = ts - p;
        memcpy(out + oi, p, before);
        oi += before;

        // Skip past </think>
        char *te = strstr(ts, "</think>");
        if (te) {
            p = te + 8; // past </think>
            // Skip whitespace/newlines after closing tag
            while (*p == '\n' || *p == '\r' || *p == ' ') p++;
        } else {
            // Unclosed think block — skip to end
            break;
        }
    }
    out[oi] = 0;

    // If stripping produced only whitespace, return NULL
    const char *check = out;
    while (*check == ' ' || *check == '\n' || *check == '\r') check++;
    if (!*check) { free(out); return NULL; }

    return out;
}

static void session_save_turn(const char *session_id, const char *role, const char *content) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.jsonl", g.sessions_dir, session_id);
    FILE *f = fopen(path, "a");
    if (!f) return;

    // Strip thinking blocks from assistant responses to keep session compact
    char *stripped = NULL;
    if (strcmp(role, "assistant") == 0) {
        stripped = strip_thinking_blocks(content);
    }

    char *escaped = json_escape_alloc(stripped ? stripped : content);
    if (escaped) {
        fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
        free(escaped);
    }
    free(stripped);
    fclose(f);
}

// Save an assistant turn with native tool_calls JSON for correct session replay.
// tool_calls_json is the raw JSON array (e.g. [{"id":"...","function":{...}}]).
static void session_save_assistant_with_tool_calls(const char *session_id, const char *content,
                                                    const char *tool_calls_json) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.jsonl", g.sessions_dir, session_id);
    FILE *f = fopen(path, "a");
    if (!f) return;

    char *stripped = strip_thinking_blocks(content);
    char *escaped = json_escape_alloc(stripped ? stripped : (content ? content : ""));
    if (escaped) {
        fprintf(f, "{\"role\":\"assistant\",\"content\":\"%s\",\"tool_calls\":%s}\n",
                escaped, tool_calls_json);
        free(escaped);
    }
    free(stripped);
    fclose(f);
}

// Replace the last line in a session file (used to fix truncated tool calls).
static void session_replace_last_turn(const char *session_id, const char *role, const char *content) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.jsonl", g.sessions_dir, session_id);

    // Read all lines
    FILE *f = fopen(path, "r");
    if (!f) return;
    char **lines = NULL;
    int nlines = 0;
    char buf[MAX_RESPONSE];
    while (fgets(buf, sizeof(buf), f)) {
        lines = realloc(lines, sizeof(char *) * (nlines + 1));
        lines[nlines++] = strdup(buf);
    }
    fclose(f);

    // Rewrite: all lines except the last, then the replacement (or drop if content is NULL/empty)
    f = fopen(path, "w");
    if (!f) { for (int i = 0; i < nlines; i++) free(lines[i]); free(lines); return; }
    for (int i = 0; i < nlines - 1; i++) {
        fputs(lines[i], f);
        free(lines[i]);
    }
    if (nlines > 0) free(lines[nlines - 1]);
    free(lines);

    if (content && content[0]) {
        char *escaped = json_escape_alloc(content);
        if (escaped) {
            fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
            free(escaped);
        }
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
// Native tool definitions for Ollama /api/chat tools parameter
// ============================================================================

// Helper macro: emit one tool definition with JSON Schema
#define TOOL_DEF(NAME, DESC, PROPS, REQ) \
    "{\"type\":\"function\",\"function\":{\"name\":\"" NAME "\",\"description\":\"" DESC "\"," \
    "\"parameters\":{\"type\":\"object\",\"properties\":{" PROPS "}," \
    "\"required\":[" REQ "],\"additionalProperties\":false}}}"

static char *build_tools_json(void) {
    size_t cap = 32768;
    char *buf = malloc(cap);
    int len = 0;

    len += snprintf(buf + len, cap - len, "[");

    // Core tools — always available
    static const char *core_tools[] = {
        TOOL_DEF("bash", "Run a shell command and return output",
                 "\"command\":{\"type\":\"string\",\"description\":\"The shell command to execute\"}",
                 "\"command\""),

        TOOL_DEF("read_file", "Read file contents from disk",
                 "\"path\":{\"type\":\"string\",\"description\":\"File path to read\"}",
                 "\"path\""),

        TOOL_DEF("list_dir", "List directory contents",
                 "\"path\":{\"type\":\"string\",\"description\":\"Directory path (default: cwd)\"}",
                 ""),

        TOOL_DEF("glob", "Find files matching a glob pattern",
                 "\"pattern\":{\"type\":\"string\",\"description\":\"Glob pattern (e.g. **/*.js)\"},"
                 "\"path\":{\"type\":\"string\",\"description\":\"Base directory (default: cwd)\"}",
                 "\"pattern\""),

        TOOL_DEF("grep", "Search file contents with regex",
                 "\"pattern\":{\"type\":\"string\",\"description\":\"Regex pattern to search for\"},"
                 "\"path\":{\"type\":\"string\",\"description\":\"File or directory to search\"},"
                 "\"include\":{\"type\":\"string\",\"description\":\"File glob filter (e.g. *.py)\"}",
                 "\"pattern\""),

        TOOL_DEF("file_write", "Create or overwrite a file with content. Use this instead of bash/printf/cat for writing files.",
                 "\"path\":{\"type\":\"string\",\"description\":\"File path to write (relative to cwd or absolute)\"},"
                 "\"content\":{\"type\":\"string\",\"description\":\"Full file content to write\"}",
                 "\"path\",\"content\""),

        TOOL_DEF("file_edit", "Replace exact text in a file",
                 "\"path\":{\"type\":\"string\",\"description\":\"File path to edit\"},"
                 "\"old_string\":{\"type\":\"string\",\"description\":\"Exact text to find (must match once)\"},"
                 "\"new_string\":{\"type\":\"string\",\"description\":\"Replacement text\"}",
                 "\"path\",\"old_string\",\"new_string\""),

        TOOL_DEF("web_fetch", "Fetch URL content (HTML/JSON/text)",
                 "\"url\":{\"type\":\"string\",\"description\":\"URL to fetch\"}",
                 "\"url\""),

        TOOL_DEF("system_info", "Get system information (OS, CPU, memory, disk)", "", ""),

        TOOL_DEF("process_list", "List running processes",
                 "\"filter\":{\"type\":\"string\",\"description\":\"Filter processes by name\"}",
                 ""),

        TOOL_DEF("process_kill", "Kill a process by PID",
                 "\"pid\":{\"type\":\"string\",\"description\":\"Process ID to kill\"}",
                 "\"pid\""),

        TOOL_DEF("clipboard_read", "Read clipboard contents", "", ""),

        TOOL_DEF("clipboard_write", "Write text to clipboard",
                 "\"content\":{\"type\":\"string\",\"description\":\"Text to copy to clipboard\"}",
                 "\"content\""),

        TOOL_DEF("open_app", "Open a file, URL, or application",
                 "\"target\":{\"type\":\"string\",\"description\":\"Path, URL, or app name to open\"}",
                 "\"target\""),

        TOOL_DEF("notify", "Show a macOS notification",
                 "\"title\":{\"type\":\"string\",\"description\":\"Notification title\"},"
                 "\"message\":{\"type\":\"string\",\"description\":\"Notification message\"}",
                 "\"title\",\"message\""),

        TOOL_DEF("memory_save", "Save a persistent memory",
                 "\"name\":{\"type\":\"string\",\"description\":\"Memory name/key\"},"
                 "\"type\":{\"type\":\"string\",\"description\":\"Type: user|feedback|project|reference\"},"
                 "\"description\":{\"type\":\"string\",\"description\":\"One-line description\"},"
                 "\"content\":{\"type\":\"string\",\"description\":\"Memory content\"},"
                 "\"scope\":{\"type\":\"string\",\"description\":\"Scope: project|global (default: project)\"}",
                 "\"name\",\"type\",\"description\",\"content\""),

        TOOL_DEF("memory_search", "Search saved memories",
                 "\"query\":{\"type\":\"string\",\"description\":\"Search query\"}",
                 ""),

        TOOL_DEF("memory_list", "List all saved memories", "", ""),

        TOOL_DEF("memory_delete", "Delete a saved memory",
                 "\"query\":{\"type\":\"string\",\"description\":\"Memory name or search query to delete\"}",
                 "\"query\""),

        TOOL_DEF("screenshot", "Take a screenshot",
                 "\"region\":{\"type\":\"string\",\"description\":\"Region: full|window|selection (default: full)\"}",
                 ""),

        TOOL_DEF("window_list", "List open windows", "", ""),

        TOOL_DEF("window_focus", "Focus/activate an application window",
                 "\"app\":{\"type\":\"string\",\"description\":\"Application name to focus\"}",
                 "\"app\""),

        TOOL_DEF("display_info", "Get display/screen information", "", ""),

        TOOL_DEF("net_info", "Get network interface information", "", ""),

        TOOL_DEF("net_connections", "List active network connections",
                 "\"filter\":{\"type\":\"string\",\"description\":\"Filter connections (e.g. port or process)\"}",
                 ""),

        TOOL_DEF("service_status", "Check service/daemon status",
                 "\"service\":{\"type\":\"string\",\"description\":\"Service name to check\"}",
                 ""),

        TOOL_DEF("disk_usage", "Show disk usage",
                 "\"path\":{\"type\":\"string\",\"description\":\"Path to check (default: /)\"}",
                 ""),

        TOOL_DEF("hardware_info", "Get hardware details (CPU, GPU, memory)", "", ""),

        TOOL_DEF("applescript", "Run an AppleScript",
                 "\"script\":{\"type\":\"string\",\"description\":\"AppleScript code to execute\"}",
                 "\"script\""),

        // NOTE: artifact is excluded from native tools — the model refuses to stuff
        // thousands of tokens of HTML into a JSON string argument. It works via text-based
        // <tool_call> format instead (see system prompt).

        // NOTE: image_generate is added conditionally below (only when ComfyUI is installed).
        // Including it when unavailable causes the model to reason about it in <think>
        // blocks and then decide it can't work, wasting tokens and confusing itself.

        TOOL_DEF("pdf_export", "Export an artifact to PDF for sharing",
                 "\"title\":{\"type\":\"string\",\"description\":\"Artifact title to export (or 'latest' for most recent)\"},"
                 "\"path\":{\"type\":\"string\",\"description\":\"Output PDF path (optional — defaults to same dir as artifact)\"}",
                 "\"title\""),

        TOOL_DEF("cron", "Manage recurring scheduled tasks",
                 "\"action\":{\"type\":\"string\",\"description\":\"Action: add|list|remove|enable|disable\"},"
                 "\"schedule\":{\"type\":\"string\",\"description\":\"5-field cron schedule: min hour dom month dow (e.g. 0 9 * * 1-5)\"},"
                 "\"prompt\":{\"type\":\"string\",\"description\":\"Prompt to send when job triggers\"},"
                 "\"description\":{\"type\":\"string\",\"description\":\"Human-readable description\"},"
                 "\"id\":{\"type\":\"string\",\"description\":\"Job ID (for remove/enable/disable)\"}",
                 "\"action\""),

        NULL
    };

    for (int i = 0; core_tools[i]; i++) {
        if (i > 0) buf[len++] = ',';
        int tl = (int)strlen(core_tools[i]);
        if ((size_t)(len + tl + 64) > cap) { cap *= 2; buf = realloc(buf, cap); }
        memcpy(buf + len, core_tools[i], tl);
        len += tl;
    }

    // Conditional tool: image_generate (only when ComfyUI is installed)
    if (g_comfyui_installed) {
        const char *t = "," TOOL_DEF("image_generate", "Generate an image locally on the GPU. INSTALLED AND OPERATIONAL. Returns a file path to the generated PNG. Produces high-quality photorealistic output. Call this tool — do not use external image URLs instead.",
            "\"prompt\":{\"type\":\"string\",\"description\":\"Detailed image description\"},"
            "\"width\":{\"type\":\"integer\",\"description\":\"Width in pixels (default: 512, max: 1024)\"},"
            "\"height\":{\"type\":\"integer\",\"description\":\"Height in pixels (default: 512, max: 1024)\"},"
            "\"style\":{\"type\":\"string\",\"description\":\"Optional style prefix: photorealistic, artistic, cartoon, illustration, cinematic\"}",
            "\"prompt\"");
        int tl = (int)strlen(t);
        if ((size_t)(len + tl + 64) > cap) { cap *= 2; buf = realloc(buf, cap); }
        memcpy(buf + len, t, tl); len += tl;
    }

    // Conditional service tools
    if (!g_connections_loaded) load_connections();
    Connection *brave = get_connection("brave_search");
    Connection *gh = get_connection("github");
    Connection *goog = get_connection("google");
    Connection *wolf = get_connection("wolfram");

    if (brave && brave->active) {
        const char *t = "," TOOL_DEF("web_search", "Search the web",
            "\"query\":{\"type\":\"string\",\"description\":\"Search query\"},"
            "\"count\":{\"type\":\"integer\",\"description\":\"Number of results (default: 5)\"}",
            "\"query\"");
        int tl = (int)strlen(t);
        if ((size_t)(len + tl + 64) > cap) { cap *= 2; buf = realloc(buf, cap); }
        memcpy(buf + len, t, tl); len += tl;
    }

    if (gh && gh->active) {
        const char *t = "," TOOL_DEF("github", "Interact with GitHub",
            "\"action\":{\"type\":\"string\",\"description\":\"Action: search_repos|list_issues|read_issue|list_prs|user\"},"
            "\"repo\":{\"type\":\"string\",\"description\":\"Repository (owner/name)\"},"
            "\"query\":{\"type\":\"string\",\"description\":\"Search query\"},"
            "\"number\":{\"type\":\"integer\",\"description\":\"Issue/PR number\"},"
            "\"state\":{\"type\":\"string\",\"description\":\"State filter: open|closed|all\"}",
            "\"action\"");
        int tl = (int)strlen(t);
        if ((size_t)(len + tl + 64) > cap) { cap *= 2; buf = realloc(buf, cap); }
        memcpy(buf + len, t, tl); len += tl;
    }

    if (goog && goog->active) {
        const char *gmail_t = "," TOOL_DEF("gmail", "Gmail operations",
            "\"action\":{\"type\":\"string\",\"description\":\"Action: search|read|send|draft|trash|labels|profile\"},"
            "\"query\":{\"type\":\"string\",\"description\":\"Search query\"},"
            "\"id\":{\"type\":\"string\",\"description\":\"Message/thread ID\"},"
            "\"to\":{\"type\":\"string\",\"description\":\"Recipient email\"},"
            "\"subject\":{\"type\":\"string\",\"description\":\"Email subject\"},"
            "\"body\":{\"type\":\"string\",\"description\":\"Email body\"},"
            "\"account\":{\"type\":\"string\",\"description\":\"Google account\"}",
            "\"action\"");
        int tl = (int)strlen(gmail_t);
        if ((size_t)(len + tl + 64) > cap) { cap *= 2; buf = realloc(buf, cap); }
        memcpy(buf + len, gmail_t, tl); len += tl;

        const char *gdrive_t = "," TOOL_DEF("gdrive", "Google Drive operations",
            "\"action\":{\"type\":\"string\",\"description\":\"Action: list|search|download|upload|mkdir|share|delete\"},"
            "\"id\":{\"type\":\"string\",\"description\":\"File/folder ID\"},"
            "\"path\":{\"type\":\"string\",\"description\":\"Local file path\"},"
            "\"name\":{\"type\":\"string\",\"description\":\"File name\"},"
            "\"query\":{\"type\":\"string\",\"description\":\"Search query\"},"
            "\"account\":{\"type\":\"string\",\"description\":\"Google account\"}",
            "\"action\"");
        tl = (int)strlen(gdrive_t);
        if ((size_t)(len + tl + 64) > cap) { cap *= 2; buf = realloc(buf, cap); }
        memcpy(buf + len, gdrive_t, tl); len += tl;

        const char *gdocs_t = "," TOOL_DEF("gdocs", "Google Docs operations",
            "\"action\":{\"type\":\"string\",\"description\":\"Action: create|read|append\"},"
            "\"id\":{\"type\":\"string\",\"description\":\"Document ID\"},"
            "\"title\":{\"type\":\"string\",\"description\":\"Document title\"},"
            "\"content\":{\"type\":\"string\",\"description\":\"Content to write\"},"
            "\"account\":{\"type\":\"string\",\"description\":\"Google account\"}",
            "\"action\"");
        tl = (int)strlen(gdocs_t);
        if ((size_t)(len + tl + 64) > cap) { cap *= 2; buf = realloc(buf, cap); }
        memcpy(buf + len, gdocs_t, tl); len += tl;
    }

    if (wolf && wolf->active) {
        const char *t = "," TOOL_DEF("wolfram", "Query Wolfram Alpha for computation/facts",
            "\"query\":{\"type\":\"string\",\"description\":\"Query for Wolfram Alpha\"}",
            "\"query\"");
        int tl = (int)strlen(t);
        if ((size_t)(len + tl + 64) > cap) { cap *= 2; buf = realloc(buf, cap); }
        memcpy(buf + len, t, tl); len += tl;
    }

    len += snprintf(buf + len, cap - len, "]");
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
    if (strcmp(g_agent_name, "PRE") != 0)
        plen += snprintf(preamble + plen, cap - plen,
            "You are %s, a Personal Reasoning Engine (PRE) — a fully local agentic assistant running on Apple Silicon. "
            "All data stays on this machine. You have persistent memory across sessions.\n\n", g_agent_name);
    else
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

    // Current date/time so the model knows "today"
    {
        time_t now = time(NULL);
        struct tm *tm = localtime(&now);
        char datebuf[128];
        const char *wday[] = {"Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"};
        const char *mon[] = {"January","February","March","April","May","June",
                             "July","August","September","October","November","December"};
        snprintf(datebuf, sizeof(datebuf), "%s, %s %d, %d",
                 wday[tm->tm_wday], mon[tm->tm_mon], tm->tm_mday, tm->tm_year + 1900);
        plen += snprintf(preamble + plen, cap - plen,
            "Today is %s. Use this when interpreting relative dates like \"today\", \"yesterday\", \"this week\".\n\n",
            datebuf);
    }

    // Tool usage guidance — tool definitions are now provided via Ollama's native
    // tools parameter (structured function calling), so we don't need format instructions.
    // Keep behavioral guidance that the schema can't express.
    // Tool rules — consolidated into one block to avoid overwhelming the model.
    // Gemma 4 26B loses instruction-following when there are too many competing
    // CRITICAL sections. One clear rules block works better than four verbose ones.
    plen += snprintf(preamble + plen, cap - plen,
        "RULES (follow these exactly):\n"
        "1. NEVER output code, HTML, or file contents in chat. Use tools instead.\n"
        "2. artifact uses text-based <tool_call> tags (not function calls):\n"
        "   <tool_call>\n"
        "   {\"name\": \"artifact\", \"arguments\": {\"title\": \"...\", \"content\": \"...HTML...\", \"type\": \"html\"}}\n"
        "   </tool_call>\n"
        "   All other tools (including file_write) are native function calls.\n"
        "   NEVER use bash with printf/cat/echo to write files. Use the file_write tool.\n"
        "3. One tool call per turn. STOP after each call and wait for the result.\n"
        "4. For research: call web_search 3-5 times with DIFFERENT specific queries before writing.\n"
        "5. For reports with images: web_search first, then image_generate for each image, then artifact last.\n"
        "6. In HTML artifacts: load CDN scripts in <head>.\n"
        "7. For long reports: use append_to to add sections to an existing artifact.\n");

    if (g_comfyui_installed) {
        plen += snprintf(preamble + plen, cap - plen,
            "8. image_generate is a WORKING native function call. It creates photorealistic images on the local GPU in ~30-45 seconds. "
            "ALWAYS call it when images are requested. NEVER use Unsplash or external URLs instead. "
            "After generating, use the returned path in artifacts: <img src='file:///path/from/tool'>\n");
    }

    // Report quality guidance — prevents generic filler content
    plen += snprintf(preamble + plen, cap - plen,
        "\nREPORT QUALITY STANDARDS:\n"
        "When creating reports or documents:\n"
        "- USE SPECIFIC DATA from your web_search results: names, dates, numbers, quotes. "
        "Never write vague summaries like 'showed incredible promise' or 'significant progress.'\n"
        "- INCLUDE EVERY SECTION the user requested. Check their prompt and ensure nothing is missing.\n"
        "- CHARTS must use REAL DATA from research, not arbitrary placeholder numbers. "
        "Label axes meaningfully. Include all chart types the user requested.\n"
        "- Each section needs 2-3 substantive paragraphs minimum with specific facts.\n"
        "- For multi-section reports: create the header/CSS + first 2 sections in the initial artifact, "
        "then use append_to for remaining sections. This produces richer content.\n"
        "- End with pdf_export if the user requested PDF output — do not forget this step.\n"
        "- Validate your HTML: matching quotes on attributes, no duplicate CSS keywords.\n");

    plen += snprintf(preamble + plen, cap - plen, "\n");

    // Tell the model about connection-dependent tools (even if not active)
    if (!g_connections_loaded) load_connections();
    Connection *brave = get_connection("brave_search");
    Connection *gh = get_connection("github");
    Connection *goog = get_connection("google");
    Connection *wolf = get_connection("wolfram");
    int has_any = (brave && brave->active) || (gh && gh->active) || (goog && goog->active) || (wolf && wolf->active);
    if (!has_any) {
        plen += snprintf(preamble + plen, cap - plen,
            "- Additional tools (gmail, gdrive, gdocs, web_search, github, wolfram) are available "
            "but require the user to run /connections setup first. If the user asks about email, "
            "web search, GitHub, Google Drive, or Wolfram queries, tell them to run /connections setup.\n");
    } else {
        // Note which are NOT connected
        if (!brave || !brave->active)
            plen += snprintf(preamble + plen, cap - plen,
                "- web_search is available if the user runs /connections add brave_search.\n");
        if (!gh || !gh->active)
            plen += snprintf(preamble + plen, cap - plen,
                "- github tool is available if the user runs /connections add github.\n");
        if (!goog || !goog->active)
            plen += snprintf(preamble + plen, cap - plen,
                "- gmail, gdrive, gdocs tools are available if the user runs /connections add google.\n");
        if (!wolf || !wolf->active)
            plen += snprintf(preamble + plen, cap - plen,
                "- wolfram tool is available if the user runs /connections add wolfram.\n");
    }

    // Tell the model about available Google accounts for multi-account support
    char accounts[256];
    int acct_count = 0;
    for (int i = 0; i < g_connections_count; i++) {
        if (g_connections[i].active && g_connections[i].is_oauth &&
            strncmp(g_connections[i].name, "google", 6) == 0)
            acct_count++;
    }
    if (acct_count > 1 && list_google_accounts(accounts, sizeof(accounts))) {
        plen += snprintf(preamble + plen, cap - plen,
            "Google accounts: %s. Use account param to target a specific account (e.g. account:\"work\").\n",
            accounts);
    }

    // System status — presented as live facts, not instructions.
    // Models trust factual assertions about their environment more than commands.
    plen += snprintf(preamble + plen, cap - plen, "\n--- SYSTEM STATUS ---\n");
    if (g_comfyui_installed) {
        plen += snprintf(preamble + plen, cap - plen,
            "image_generate: ONLINE (ComfyUI + %s, port %d, GPU-accelerated)\n",
            g_comfyui_checkpoint, g_comfyui_port);
    }
    Connection *bs = get_connection("brave_search");
    if (bs && bs->active) {
        plen += snprintf(preamble + plen, cap - plen, "web_search: ONLINE (Brave Search API)\n");
    }
    plen += snprintf(preamble + plen, cap - plen, "---\n\n");

    return preamble;
}

// Build the user message content: attachment (if any) + user input
// System preamble is now sent separately as a "system" role message
static char *build_message(const char *input, int is_first_turn) {
    (void)is_first_turn; // system prompt handled in send_request now
    size_t alen = g_pending_attach ? strlen(g_pending_attach) : 0;
    size_t ilen = strlen(input);

    char *msg = malloc(alen + ilen + 8);
    size_t off = 0;
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
    // Compact when we've used 75% of the model's actual context window (32768).
    // Can't grow num_ctx at runtime — any change triggers a 300s+ model reload.
    // 75% of 32768 = ~24576 tokens — enough room for the next tool round-trip.
    int threshold = MODEL_CTX * 3 / 4;
    if (estimated_tokens > threshold) {
        // Keep last 6 turns (12 messages) for immediate context
        compact_session(g.session_id, 6);
        // Reset token estimates after compaction
        g.total_tokens_in = estimated_tokens / 4; // rough estimate of compacted size
        g.total_tokens_out = 0;
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

    // Build request body for Ollama native /api/chat
    char session_path[1024];
    snprintf(session_path, sizeof(session_path), "%s/%s.jsonl", g.sessions_dir, session_id);

    size_t body_cap = 1024 * 1024; // 1MB initial
    char *body = malloc(body_cap);
    int body_len = snprintf(body, body_cap, "{\"model\":\"%s\",\"stream\":true,\"keep_alive\":\"24h\"", g_model);

    // Context window strategy: ALWAYS send the same num_ctx value matching the Modelfile.
    // Any num_ctx change triggers a full model reload in Ollama (300s+ for Gemma 4 26B).
    // The Modelfile sets num_ctx=32768 and pre-launch pre-warms the model with a real
    // request to fully allocate the KV cache. By sending the same value every time,
    // Ollama recognizes it matches the loaded model and reuses the existing KV allocation.
    // Auto-compaction (maybe_compact) keeps conversations within this 32K budget.
    body_len += snprintf(body + body_len, body_cap - body_len,
        ",\"options\":{\"num_predict\":%d,\"num_ctx\":%d}",
        max_tokens, MODEL_CTX);

    // Messages array — starts with system message for tool/context instructions
    // Using role:system in messages array (not top-level "system" field) for
    // reliable instruction following with Gemma 4 and Ollama KV cache reuse
    body_len += snprintf(body + body_len, body_cap - body_len, ",\"messages\":[");

    char *sys_prompt = build_context_preamble();
    if (sys_prompt) {
        char *sys_esc = json_escape_alloc(sys_prompt);
        free(sys_prompt);
        if (sys_esc) {
            size_t slen = strlen(sys_esc);
            if ((size_t)body_len + slen + 64 > body_cap) {
                body_cap = body_len + slen + 4096;
                body = realloc(body, body_cap);
            }
            body_len += snprintf(body + body_len, body_cap - body_len,
                "{\"role\":\"system\",\"content\":\"%s\"}", sys_esc);
            free(sys_esc);
        }
    }

    // Replay session history (system message is always first, so always comma-prefix)
    FILE *sf = fopen(session_path, "r");
    if (sf) {
        char sline[MAX_RESPONSE];
        while (fgets(sline, sizeof(sline), sf)) {
            int sl = (int)strlen(sline);
            while (sl > 0 && (sline[sl-1] == '\n' || sline[sl-1] == '\r')) sline[--sl] = 0;
            if (sl == 0) continue;

            if ((int)(body_len + sl + 10) > (int)body_cap - 100) {
                body_cap *= 2;
                body = realloc(body, body_cap);
            }

            body[body_len++] = ',';
            memcpy(body + body_len, sline, sl);
            body_len += sl;
        }
        fclose(sf);
    }

    // Append the new user message (NULL = replay session only, no new message)
    if (user_message) {
        char *escaped = json_escape_alloc(user_message);
        if (!escaped) { free(body); close(sock); return -1; }
        size_t elen = strlen(escaped);
        size_t img_len = g_pending_image ? strlen(g_pending_image) : 0;

        size_t need = body_len + elen + img_len + 512;
        if (need > body_cap) {
            body_cap = need + 4096;
            body = realloc(body, body_cap);
        }

        if (g_pending_image) {
            // Ollama native multimodal: images as base64 array at message level
            body_len += snprintf(body + body_len, body_cap - body_len,
                ",{\"role\":\"user\",\"content\":\"%s\",\"images\":[\"%s\"]}",
                escaped, g_pending_image);
            free(g_pending_image);
            g_pending_image = NULL;
        } else {
            body_len += snprintf(body + body_len, body_cap - body_len,
                ",{\"role\":\"user\",\"content\":\"%s\"}",
                escaped);
        }
        free(escaped);
    }

    // Close messages array
    body_len += snprintf(body + body_len, body_cap - body_len, "]");

    // Add native tool definitions — Ollama uses these for structured tool calling.
    // The model returns tool_calls as structured JSON instead of text-based <tool_call> tags.
    char *tools_json = build_tools_json();
    if (tools_json) {
        size_t tlen = strlen(tools_json);
        if ((size_t)body_len + tlen + 32 > body_cap) {
            body_cap = body_len + tlen + 4096;
            body = realloc(body, body_cap);
        }
        body_len += snprintf(body + body_len, body_cap - body_len, ",\"tools\":%s", tools_json);
        free(tools_json);
    }

    // Close request body
    body_len += snprintf(body + body_len, body_cap - body_len, "}");

    size_t req_cap = body_len + 256;
    char *request = malloc(req_cap);
    int req_len = snprintf(request, req_cap,
        "POST /api/chat HTTP/1.1\r\n"
        "Host: localhost:%d\r\n"
        "Content-Type: application/json\r\n"
        "Connection: close\r\n"
        "Content-Length: %d\r\n"
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

static char *stream_response(int sock, int num_predict) {
    int header_done = 0, in_think = 0, tokens = 0;
    int had_thinking = 0, in_tool_call_display = 0;
    double t_start = now_ms(), t_first = 0, last_progress_ms = 0;
    md_reset();

    char *response = calloc(1, MAX_RESPONSE);
    int resp_len = 0;

    // Spinner during prefill
    static const char *spin[] = {"⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"};
    int spin_idx = 0;
    int spinning = 1;

    // Raw recv buffer — avoids stdio buffering latency on SSE/NDJSON chunks
    char recvbuf[65536];
    int rb_len = 0;  // valid bytes in recvbuf
    int rb_pos = 0;  // current scan position

    // Server-reported stats from the final "done":true message
    int server_prompt_tokens = 0, server_eval_tokens = 0;
    double server_eval_duration = 0, server_prompt_duration = 0;

    // Decode a JSON string value starting at p into buf, return length
    // Handles \n \t \" \\ and \uXXXX (BMP codepoints, ASCII subset)
    #define DECODE_JSON_STR(p, buf, bufsz) ({ \
        int _di = 0; \
        for (int _i = 0; (p)[_i] && (p)[_i] != '"' && _di < (bufsz)-1; _i++) { \
            if ((p)[_i] == '\\' && (p)[_i+1]) { \
                _i++; \
                if ((p)[_i] == 'u' && (p)[_i+1] && (p)[_i+2] && (p)[_i+3] && (p)[_i+4]) { \
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

    int done = 0;
    double last_data_ms = now_ms();  // stall detection
    // Prefill (before first token) can be slow for large contexts — allow 5 minutes.
    // Mid-generation stalls (after first token) are caught faster at 90 seconds.
    // Base stall timeouts. Prefill timeout scales with estimated context size:
    // at 8K tokens, 5 min is plenty; at 128K+, allow up to 10 min.
    int estimated_ctx = g.total_tokens_in + g.total_tokens_out;
    // Prefill timeout: scale with context size. A KV cache rebuild at 32K tokens
    // can take 250+ seconds on M4 Max; at 64K it can exceed 5 minutes. Allow enough
    // time for legitimate rebuilds without hanging indefinitely on true stalls.
    double prefill_stall_ms = 300000.0; // 5 min base
    if (estimated_ctx > 16384) prefill_stall_ms = 300000.0 + (estimated_ctx - 16384) * 3.0;
    if (prefill_stall_ms > 600000.0) prefill_stall_ms = 600000.0; // cap at 10 min
    #define STALL_GENERATE_MS  90000   // 90s — no token mid-generation = hung
    #define STALL_TOOLCALL_MS  600000  // 10 min — native tool call generation (no streaming)
    int expect_native_tool = 1; // we always send tools param now, so tool calls are possible
    double tool_wait_start_ms = 0; // when we started waiting for a potential tool call

    while (!done) {
        // Extract next newline-delimited line from recv buffer
        char *nl = NULL;
        while (!(nl = memchr(recvbuf + rb_pos, '\n', rb_len - rb_pos))) {
            // Need more data — compact buffer first
            if (rb_pos > 0) {
                memmove(recvbuf, recvbuf + rb_pos, rb_len - rb_pos);
                rb_len -= rb_pos;
                rb_pos = 0;
            }

            // Use select() for spinner animation during prefill
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(sock, &fds);
            struct timeval tv = {0, spinning ? 120000 : 500000};
            int ready = select(sock + 1, &fds, NULL, NULL, &tv);
            if (ready <= 0) {
                if (spinning) {
                    double elapsed = (now_ms() - t_start) / 1000.0;
                    if (elapsed < 10)
                        printf("\r  " ANSI_DIM "[reasoning %s]" ANSI_RESET "  ", spin[spin_idx % 10]);
                    else
                        printf("\r  " ANSI_DIM "[reasoning %s  %.0fs]" ANSI_RESET "  ", spin[spin_idx % 10], elapsed);
                    fflush(stdout);
                    spin_idx++;
                }
                // Stall detection. Three modes:
                // 1. Prefill (before first token): 5-10 min depending on context
                // 2. Mid-generation (text streaming): 90s between tokens
                // 3. Native tool call wait (text done, waiting for tool_calls): 10 min
                //    With native function calling, Ollama buffers the entire tool call
                //    and sends it as one message. No data arrives during generation.
                double stall_limit;
                const char *stall_phase;
                if (!t_first) {
                    stall_limit = prefill_stall_ms;
                    stall_phase = "prefill";
                } else if (expect_native_tool && tokens > 0 &&
                           now_ms() - last_data_ms > 5000) {
                    // We've received text but no data for 5+ seconds.
                    // Likely the model finished text and is generating a tool call.
                    stall_limit = STALL_TOOLCALL_MS;
                    stall_phase = "tool call generation";
                    if (tool_wait_start_ms == 0) tool_wait_start_ms = now_ms();

                    // Show progress so user knows we're waiting for a tool call
                    double wait_s = (now_ms() - tool_wait_start_ms) / 1000.0;
                    if (now_ms() - last_progress_ms >= 2000) {
                        printf("\r\033[K" ANSI_DIM "  [waiting for tool call... %.0fs]"
                               ANSI_RESET, wait_s);
                        fflush(stdout);
                        last_progress_ms = now_ms();
                    }
                } else {
                    stall_limit = STALL_GENERATE_MS;
                    stall_phase = "generation";
                }
                if (now_ms() - last_data_ms > stall_limit) {
                    printf("\n" ANSI_YELLOW "  [server stall — no data for %.0fs during %s, aborting]"
                           ANSI_RESET "\n", (now_ms() - last_data_ms) / 1000.0,
                           stall_phase);
                    done = 1; break;
                }
                continue;
            }

            ssize_t n = recv(sock, recvbuf + rb_len, sizeof(recvbuf) - rb_len - 1, 0);
            if (n <= 0) { done = 1; break; }
            rb_len += (int)n;
            last_data_ms = now_ms();  // reset stall timer on data received
        }
        if (done) break;

        // We have a complete line from rb_pos to nl
        *nl = 0;
        char *line = recvbuf + rb_pos;
        rb_pos = (int)(nl - recvbuf) + 1;

        // Skip HTTP headers
        if (!header_done) {
            if (line[0] == '\r' || line[0] == 0) header_done = 1;
            continue;
        }

        // Skip chunked transfer encoding size lines (hex digits only)
        if (line[0] == 0) continue;
        int is_chunk_size = 1;
        for (int ci = 0; line[ci] && line[ci] != '\r'; ci++) {
            if (!((line[ci] >= '0' && line[ci] <= '9') ||
                  (line[ci] >= 'a' && line[ci] <= 'f') ||
                  (line[ci] >= 'A' && line[ci] <= 'F'))) {
                is_chunk_size = 0; break;
            }
        }
        if (is_chunk_size && line[0] != '{') continue;

        // Parse Ollama native NDJSON: {"message":{"content":"...","thinking":"...","tool_calls":[...]},"done":bool}

        // Check for native tool_calls (structured function calling from Ollama).
        // These arrive in a single NDJSON line with "tool_calls":[ before the "done":true line.
        char *tc_field = strstr(line, "\"tool_calls\":[");
        if (tc_field) {
            // Extract the tool_calls JSON array
            char *arr_start = tc_field + 13; // points to '['
            int depth = 0;
            const char *p = arr_start;
            while (*p) {
                if (*p == '[') depth++;
                else if (*p == ']') { depth--; if (depth == 0) break; }
                else if (*p == '"') {
                    p++;
                    while (*p && !(*p == '"' && *(p-1) != '\\')) p++;
                }
                p++;
            }
            if (*p == ']') {
                size_t arr_len = p - arr_start + 1;
                free(g_native_tool_calls);
                g_native_tool_calls = malloc(arr_len + 1);
                memcpy(g_native_tool_calls, arr_start, arr_len);
                g_native_tool_calls[arr_len] = 0;

                // Show status — clear any progress line first
                printf("\r\033[K");
                if (spinning) { spinning = 0; }
                if (tool_wait_start_ms > 0) {
                    double wait_s = (now_ms() - tool_wait_start_ms) / 1000.0;
                    printf(ANSI_DIM "  [native tool call received after %.0fs]" ANSI_RESET "\n", wait_s);
                    tool_wait_start_ms = 0;
                } else {
                    printf(ANSI_DIM "  [native tool call received]" ANSI_RESET "\n");
                }
                fflush(stdout);
                tokens++;
                if (!t_first) t_first = now_ms();
            }
        }

        // Check for done
        if (strstr(line, "\"done\":true")) {
            // Extract server-reported token counts and durations
            char eval_str[32] = {0}, prompt_str[32] = {0};
            char eval_dur[32] = {0}, prompt_dur[32] = {0};
            json_extract_str(line, "eval_count", eval_str, sizeof(eval_str));
            json_extract_str(line, "prompt_eval_count", prompt_str, sizeof(prompt_str));
            json_extract_str(line, "eval_duration", eval_dur, sizeof(eval_dur));
            json_extract_str(line, "prompt_eval_duration", prompt_dur, sizeof(prompt_dur));
            if (eval_str[0]) server_eval_tokens = atoi(eval_str);
            if (prompt_str[0]) server_prompt_tokens = atoi(prompt_str);
            if (eval_dur[0]) server_eval_duration = strtod(eval_dur, NULL) / 1e9;
            if (prompt_dur[0]) server_prompt_duration = strtod(prompt_dur, NULL) / 1e9;
            break;
        }

        // Check for thinking field (Ollama native: "thinking":"...")
        char *rk = strstr(line, "\"thinking\":\"");
        if (rk) {
            rk += 12;
            char rdecoded[4096]; int rdi = DECODE_JSON_STR(rk, rdecoded, 4096);
            if (rdi > 0) {
                if (spinning) { printf("\r\033[K"); fflush(stdout); spinning = 0; }
                tokens++;
                if (!t_first) t_first = now_ms();
                if (g.show_thinking) {
                    if (!had_thinking) {
                        printf("\n" ANSI_DIM "  ┌─ thinking ");
                        for (int d = 0; d < 36; d++) printf("─");
                        printf("\n  │ ");
                    }
                    for (int ri = 0; ri < rdi; ri++) {
                        if (rdecoded[ri] == '\n') printf("\n  │ ");
                        else putchar(rdecoded[ri]);
                    }
                    fflush(stdout);
                }
                had_thinking = 1;
                in_think = 1;
            }
        }

        // Check for content field
        char *ck = strstr(line, "\"content\":\"");
        if (!ck) continue;
        ck += 11;

        char decoded[4096]; int di = DECODE_JSON_STR(ck, decoded, 4096);
        if (!di) continue;

        // If we were thinking, content means thinking is done
        if (in_think && had_thinking) in_think = 0;

        if (spinning) { printf("\r\033[K"); fflush(stdout); spinning = 0; }

        // Handle Qwen-style <think> tags in content (backwards compat)
        if (strstr(decoded, "<think>")) { in_think = 1; had_thinking = 1; }
        if (strstr(decoded, "</think>")) { in_think = 0; tokens++; continue; }
        tokens++;
        if (!t_first) t_first = now_ms();

        if (!in_think && resp_len + di < MAX_RESPONSE - 1) {
            if (had_thinking && resp_len == 0 && g.show_thinking) {
                printf("\n  └");
                for (int d = 0; d < 48; d++) printf("─");
                printf(ANSI_RESET "\n\n");
                md_reset();
            } else if (had_thinking && resp_len == 0 && !g.show_thinking) {
                printf("\n");
            }
            memcpy(response + resp_len, decoded, di);
            resp_len += di;
            response[resp_len] = 0;
        }

        if (in_think && !g.show_thinking) continue;
        if (in_think) {
            if (g.show_thinking) {
                printf(ANSI_DIM);
                for (int ci = 0; ci < di; ci++) {
                    if (decoded[ci] == '\n') printf("\n  │ ");
                    else putchar(decoded[ci]);
                }
            }
        } else {
            // Suppress tool call content from display — show status instead
            // Track: once we see <tool_call> start suppressing, until </tool_call>
            if (!in_tool_call_display) {
                // Check if this chunk or recent content starts a tool call
                char *tc_tag = strstr(response + (resp_len > 64 ? resp_len - 64 : 0), "<tool_call>");
                if (tc_tag && !strstr(tc_tag, "</tool_call>")) {
                    in_tool_call_display = 1;
                    printf("\r\033[K");
                    printf(ANSI_DIM "  [using tool...]" ANSI_RESET);
                    fflush(stdout);
                } else {
                    md_print(decoded);
                }
            } else {
                // Still inside tool call — check for closing tag
                if (strstr(response + (resp_len > 64 ? resp_len - 64 : 0), "</tool_call>")) {
                    in_tool_call_display = 2;  // done, don't print remaining in this response
                }
                // Show live progress every 2 seconds so user knows it's not frozen
                if (in_tool_call_display == 1) {
                    double now = now_ms();
                    if (now - last_progress_ms >= 2000) {
                        double elapsed = (now - t_start) / 1000.0;
                        double toks = elapsed > 0 ? tokens / elapsed : 0;
                        printf("\r\033[K" ANSI_DIM "  [using tool... %d tokens, %.0fs, %.1f tok/s]"
                               ANSI_RESET, tokens, elapsed, toks);
                        fflush(stdout);
                        last_progress_ms = now;
                    }

                    // Early abort: if we're inside an incomplete tool call and approaching
                    // the num_predict limit, stop now instead of waiting to exhaust the
                    // budget. This saves minutes of wasted generation on large artifacts.
                    if (num_predict > 0 && tokens > (num_predict * 9 / 10)) {
                        printf("\r\033[K" ANSI_YELLOW
                               "  [tool call approaching limit (%d/%d tokens) — aborting for retry]"
                               ANSI_RESET "\n", tokens, num_predict);
                        done = 1; break;
                    }
                }
            }
        }
        fflush(stdout);
    }
    close(sock);

    printf(ANSI_RESET);

    // Stats — use server-reported timing for accurate tok/s
    double t_end = now_ms();
    int reported_tokens = server_eval_tokens > 0 ? server_eval_tokens : tokens;

    // Server-reported eval rate is the ground truth — no client overhead
    double tok_s;
    if (server_eval_duration > 0 && server_eval_tokens > 0)
        tok_s = server_eval_tokens / server_eval_duration;
    else {
        double gen_time = t_first > 0 ? t_end - t_first : 0;
        int gen_tokens = reported_tokens > 1 ? reported_tokens - 1 : reported_tokens;
        tok_s = (gen_tokens > 0 && gen_time > 0) ? gen_tokens * 1000.0 / gen_time : 0;
    }

    // TTFT: server prompt eval duration if available, else client-measured
    double ttft;
    if (server_prompt_duration > 0)
        ttft = server_prompt_duration * 1000.0; // convert to ms
    else
        ttft = t_first > 0 ? t_first - t_start : t_end - t_start;

    double gen_time_ms = server_eval_duration > 0 ? server_eval_duration * 1000.0 :
                         (t_first > 0 ? t_end - t_first : 0);

    g.last_token_count = reported_tokens;
    g.last_tok_s = tok_s;
    g.last_ttft_ms = ttft;
    g.total_tokens_out += reported_tokens;
    if (server_prompt_tokens > 0) g.total_tokens_in = server_prompt_tokens;
    g.cumulative_gen_ms += gen_time_ms;

    printf("\n\n");
    if (tokens > 0) {
        printf(ANSI_DIM "  ");
        for (int d = 0; d < 50; d++) printf("·");
        printf("\n  %d tokens  │  %.1f tok/s  │  TTFT %.1fs\n",
               tokens, tok_s, ttft / 1000.0);

        // Context window usage bar (relative to model's actual num_ctx, not MAX_CONTEXT)
        int used = g.total_tokens_in + g.total_tokens_out;
        int pct = used * 100 / MODEL_CTX;
        if (pct > 100) pct = 100;
        int bar_width = 30;
        int filled = pct * bar_width / 100;

        printf("  context [");
        for (int i = 0; i < bar_width; i++) {
            if (i < filled) {
                if (pct >= 90)       printf(ANSI_RESET ANSI_RED "█" ANSI_DIM);
                else if (pct >= 75)  printf(ANSI_RESET ANSI_YELLOW "█" ANSI_DIM);
                else                 printf("█");
            } else {
                printf("░");
            }
        }
        printf("] %d%% of %dK", pct, MODEL_CTX / 1024);
        if (pct >= 75) printf("  ⟳ auto-compact active");
        printf(ANSI_RESET "\n\n");
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

// Greedy JSON string extraction: handles unescaped double quotes inside string values.
// When decode_json_string stops at a " that isn't followed by valid JSON continuation
// (, or }), this function keeps scanning until it finds the real string boundary.
static char *decode_json_string_greedy(const char *p, const char **endp) {
    size_t cap = 4096;
    char *buf = malloc(cap);
    size_t di = 0;
    const char *scan = p;

    while (*scan) {
        if (di + 8 > cap) { cap *= 2; buf = realloc(buf, cap); }

        if (*scan == '\\' && scan[1]) {
            scan++;
            switch (*scan) {
                case 'n': buf[di++] = '\n'; break;
                case 't': buf[di++] = '\t'; break;
                case 'r': buf[di++] = '\r'; break;
                case '"': buf[di++] = '"'; break;
                case '\\': buf[di++] = '\\'; break;
                case '/': buf[di++] = '/'; break;
                case 'u':
                    if (scan[1] && scan[2] && scan[3] && scan[4]) {
                        char hex[5] = {scan[1], scan[2], scan[3], scan[4], 0};
                        unsigned int cp = (unsigned int)strtol(hex, NULL, 16);
                        scan += 4;
                        if (cp < 0x80) buf[di++] = (char)cp;
                        else if (cp < 0x800) {
                            buf[di++] = (char)(0xC0 | (cp >> 6));
                            buf[di++] = (char)(0x80 | (cp & 0x3F));
                        } else {
                            buf[di++] = (char)(0xE0 | (cp >> 12));
                            buf[di++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            buf[di++] = (char)(0x80 | (cp & 0x3F));
                        }
                    } else { buf[di++] = 'u'; }
                    break;
                default: buf[di++] = *scan; break;
            }
            scan++;
            continue;
        }

        if (*scan == '"') {
            // Check if this is a real JSON string boundary
            const char *after = scan + 1;
            while (*after == ' ' || *after == '\n' || *after == '\r' || *after == '\t') after++;
            if (*after == ',' || *after == '}' || *after == '\0') {
                // Real boundary — stop here
                if (endp) *endp = scan + 1;
                buf[di] = 0;
                return buf;
            }
            // Not a real boundary — include the " as literal content
            buf[di++] = '"';
            scan++;
            continue;
        }

        if (*scan == '\0') break;
        buf[di++] = *scan++;
    }

    buf[di] = 0;
    if (endp) *endp = scan;
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

    // Find "arguments", "parameters", or "params" then the opening {
    // Models hallucinate different key names — claw-code handles all three.
    char *args_start = strstr(tc_body, "\"arguments\"");
    if (args_start) { args_start += 11; }
    else if ((args_start = strstr(tc_body, "\"parameters\"")) != NULL) { args_start += 12; }
    else if ((args_start = strstr(tc_body, "\"params\"")) != NULL) { args_start += 8; }
    if (args_start) {
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

                    // Check if decode_json_string stopped at a real JSON boundary
                    // If model forgot to escape " in HTML, we'll hit a false end
                    const char *boundary_check = endp;
                    while (*boundary_check == ' ' || *boundary_check == '\n' ||
                           *boundary_check == '\r' || *boundary_check == '\t')
                        boundary_check++;
                    if (*boundary_check && *boundary_check != ',' && *boundary_check != '}') {
                        // False boundary — re-extract with greedy parser
                        free(tc->vals[tc->argc]);
                        tc->vals[tc->argc] = decode_json_string_greedy(p, &endp);
                    }
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
            // Tool call is valid if we have a name (args may be empty for no-arg tools)
            if (tc->name[0]) return 1;
        }
    }

    // If we found a name but no "arguments" block, try parsing top-level keys
    // as arguments. Models often hallucinate {"name":"bash","command":"ls"} without
    // wrapping in an "arguments" object. Re-scan for all string keys that aren't "name".
    if (tc->name[0] && tc->argc == 0) {
        const char *p = tc_body;
        while (*p && tc->argc < MAX_TOOL_ARGS) {
            // Find next quoted key
            char *q = strchr(p, '"');
            if (!q) break;
            q++; // skip opening "
            // Read key
            char key[64] = {0};
            int ki = 0;
            while (*q && *q != '"' && ki < 63) key[ki++] = *q++;
            key[ki] = 0;
            if (*q == '"') q++; // skip closing "

            // Skip to value
            while (*q && (*q == ' ' || *q == ':' || *q == '\t')) q++;

            if (strcmp(key, "name") == 0 || strcmp(key, "arguments") == 0 ||
                strcmp(key, "parameters") == 0 || strcmp(key, "params") == 0) {
                // Skip known wrapper keys — advance past their value
                if (*q == '"') {
                    q++; // skip opening "
                    while (*q && !(*q == '"' && *(q-1) != '\\')) q++;
                    if (*q == '"') q++;
                } else if (*q == '{') {
                    int depth = 1; q++;
                    while (*q && depth > 0) {
                        if (*q == '{') depth++;
                        else if (*q == '}') depth--;
                        q++;
                    }
                }
                p = q;
                continue;
            }

            // Parse value for this key
            if (*q == '"') {
                q++; // skip opening "
                const char *endp;
                tc->vals[tc->argc] = decode_json_string(q, &endp);
                q = (char *)endp;
            } else {
                // Non-string value
                const char *start = q;
                while (*q && *q != ',' && *q != '}') q++;
                size_t vlen = (size_t)(q - start);
                tc->vals[tc->argc] = malloc(vlen + 1);
                memcpy(tc->vals[tc->argc], start, vlen);
                tc->vals[tc->argc][vlen] = 0;
            }
            strncpy(tc->keys[tc->argc], key, 63);
            tc->argc++;
            p = q;
        }
        if (tc->argc > 0) return 1;
    }

    // If we found a name via JSON but no arguments at all, still valid (no-arg tool)
    if (nk && tc->name[0]) return 1;

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

// Parse native Ollama tool_calls JSON array into ToolCall structs.
// Input: JSON like [{"id":"call_xxx","function":{"name":"bash","arguments":{"command":"ls"}}}]
// Returns number of tool calls parsed. Fills batch[] up to max_batch.
static int parse_native_tool_calls(const char *json, ToolCall *batch, int max_batch) {
    int count = 0;
    const char *p = json;
    if (*p != '[') return 0;
    p++;

    while (*p && count < max_batch) {
        // Skip whitespace/commas
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',')) p++;
        if (*p == ']' || *p == '\0') break;
        if (*p != '{') { p++; continue; }

        // Find "function" object within this tool_call object
        const char *func_key = strstr(p, "\"function\"");
        if (!func_key) break;

        // Find function name: "name":"tool_name"
        const char *name_key = strstr(func_key, "\"name\":\"");
        if (!name_key) break;
        name_key += 8; // past "name":"
        const char *name_end = strchr(name_key, '"');
        if (!name_end) break;

        ToolCall *tc = &batch[count];
        memset(tc, 0, sizeof(ToolCall));
        size_t name_len = name_end - name_key;
        if (name_len >= sizeof(tc->name)) name_len = sizeof(tc->name) - 1;
        memcpy(tc->name, name_key, name_len);
        tc->name[name_len] = 0;

        // Find "arguments":{...} — this is a JSON object with pre-parsed values
        const char *args_key = strstr(func_key, "\"arguments\":{");
        if (!args_key) args_key = strstr(func_key, "\"arguments\": {");
        if (!args_key) {
            // No arguments — tool with no params (like system_info, memory_list)
            count++;
            // Skip past this tool_call object
            int depth = 0;
            while (*p) {
                if (*p == '{') depth++;
                else if (*p == '}') { depth--; if (depth == 0) { p++; break; } }
                else if (*p == '"') { p++; while (*p && !(*p == '"' && *(p-1) != '\\')) p++; }
                p++;
            }
            continue;
        }

        // Find the opening { of arguments
        const char *args_start = strchr(args_key + 11, '{');
        if (!args_start) break;
        args_start++; // past {

        // Parse key-value pairs from arguments object
        const char *ap = args_start;
        while (*ap && tc->argc < MAX_TOOL_ARGS) {
            // Skip whitespace
            while (*ap && (*ap == ' ' || *ap == '\n' || *ap == '\r' || *ap == '\t' || *ap == ',')) ap++;
            if (*ap == '}') break;

            // Expect "key":value
            if (*ap != '"') { ap++; continue; }
            ap++; // past opening "
            const char *key_end = ap;
            while (*key_end && *key_end != '"') key_end++;
            if (!*key_end) break;

            size_t klen = key_end - ap;
            if (klen >= 64) klen = 63;
            memcpy(tc->keys[tc->argc], ap, klen);
            tc->keys[tc->argc][klen] = 0;

            ap = key_end + 1; // past closing "
            while (*ap && (*ap == ' ' || *ap == ':' || *ap == '\t')) ap++;

            // Parse value — could be string, number, boolean, or null
            if (*ap == '"') {
                // String value — decode JSON string
                ap++; // past opening "
                size_t val_cap = 4096;
                char *val = malloc(val_cap);
                size_t vi = 0;
                while (*ap && !(*ap == '"' && *(ap > args_start ? ap-1 : ap) != '\\')) {
                    if (*ap == '\\' && *(ap+1)) {
                        ap++;
                        switch (*ap) {
                            case 'n': val[vi++] = '\n'; break;
                            case 't': val[vi++] = '\t'; break;
                            case 'r': val[vi++] = '\r'; break;
                            case '"': val[vi++] = '"'; break;
                            case '\\': val[vi++] = '\\'; break;
                            case '/': val[vi++] = '/'; break;
                            default: val[vi++] = *ap; break;
                        }
                    } else {
                        val[vi++] = *ap;
                    }
                    ap++;
                    if (vi + 4 >= val_cap) { val_cap *= 2; val = realloc(val, val_cap); }
                }
                if (*ap == '"') ap++; // past closing "
                val[vi] = 0;
                tc->vals[tc->argc] = val;
            } else {
                // Non-string value (number, bool, null) — copy as-is
                const char *vstart = ap;
                while (*ap && *ap != ',' && *ap != '}' && *ap != ' ' && *ap != '\n') ap++;
                size_t vlen = ap - vstart;
                tc->vals[tc->argc] = malloc(vlen + 1);
                memcpy(tc->vals[tc->argc], vstart, vlen);
                tc->vals[tc->argc][vlen] = 0;
            }
            tc->argc++;
        }

        count++;

        // Skip past this tool_call object
        int depth = 0;
        const char *skip = p;
        while (*skip) {
            if (*skip == '{') depth++;
            else if (*skip == '}') { depth--; if (depth == 0) { skip++; break; } }
            else if (*skip == '"') { skip++; while (*skip && !(*skip == '"' && *(skip-1) != '\\')) skip++; }
            skip++;
        }
        p = skip;
    }

    return count;
}

// Max bytes per tool response injected into context (keeps prefill fast)
#define MAX_TOOL_RESPONSE (8 * 1024)

// Execute a single tool call and write output into buf. Returns output length.
static int execute_tool(ToolCall *tc, char *output, size_t output_sz) {
    int out_len = 0;
    output[0] = 0;
    const char *name = tc->name;

    // --- Permission check ---
    // Only destructive operations (process_kill, memory_delete, applescript) require confirmation
    PermLevel perm = tool_permission(name);

    if (perm == PERM_CONFIRM_ALWAYS) {
        const char *detail = tool_call_get(tc, "command");
        if (!detail) detail = tool_call_get(tc, "pid");
        if (!detail) detail = tool_call_get(tc, "script");
        if (!detail) detail = tool_call_get(tc, "query");
        if (!detail) detail = name;
        printf(ANSI_YELLOW "  [%s: %s]" ANSI_RESET "\n", name, detail);
        printf(ANSI_DIM "  [execute? y/n] " ANSI_RESET);
        fflush(stdout);
        int ch = getchar(); while (getchar() != '\n');
        if (ch != 'y' && ch != 'Y') {
            printf(ANSI_DIM "  [skipped]" ANSI_RESET "\n");
            return -1;
        }
    }

    // --- Resolve tool name aliases (handle model hallucinations) ---
    // Models frequently hallucinate similar but wrong tool names. This maps
    // common variants to the canonical name, inspired by LocalClaw's 134-alias table.
    static const struct { const char *alias; const char *canonical; } tool_aliases[] = {
        // Shell
        {"shell", "bash"}, {"run", "bash"}, {"terminal", "bash"}, {"sh", "bash"},
        {"cmd", "bash"}, {"execute", "bash"}, {"exec", "bash"}, {"command", "bash"},
        {"run_command", "bash"}, {"shell_exec", "bash"},
        // File read
        {"file_read", "read_file"}, {"read", "read_file"}, {"cat", "read_file"},
        {"view_file", "read_file"}, {"get_file", "read_file"},
        // File write
        {"write_file", "file_write"}, {"write", "file_write"}, {"create_file", "file_write"},
        {"save_file", "file_write"}, {"save", "file_write"},
        // File edit
        {"edit_file", "file_edit"}, {"edit", "file_edit"}, {"replace", "file_edit"},
        {"patch", "file_edit"},
        // Directory
        {"ls", "list_dir"}, {"list", "list_dir"}, {"dir", "list_dir"},
        {"list_directory", "list_dir"},
        // Search
        {"search", "grep"}, {"find", "glob"}, {"rg", "grep"}, {"ripgrep", "grep"},
        {"find_files", "glob"}, {"search_files", "grep"},
        // Web
        {"fetch", "web_fetch"}, {"curl", "web_fetch"}, {"http", "web_fetch"},
        {"browse", "web_fetch"}, {"web", "web_fetch"},
        // Artifact
        {"create_artifact", "artifact"}, {"html", "artifact"},
        {"render", "artifact"}, {"display", "artifact"},
        // Clipboard
        {"copy", "clipboard_write"}, {"paste", "clipboard_read"},
        // Memory
        {"remember", "memory_save"}, {"recall", "memory_search"},
        {"forget", "memory_delete"},
        {NULL, NULL}
    };

    // First try case-insensitive exact match, then alias lookup
    for (int a = 0; tool_aliases[a].alias; a++) {
        if (strcasecmp(name, tool_aliases[a].alias) == 0) {
            // Copy canonical name into tc->name (it's a fixed-size buffer)
            strlcpy(tc->name, tool_aliases[a].canonical, sizeof(tc->name));
            name = tc->name;
            break;
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
        // Redirect stderr to stdout so shell errors are captured in the tool result.
        // Without this, popen() only reads stdout — shell errors go to the terminal
        // display but not the tool response, causing the model to hallucinate success.
        char cmd_with_stderr[4096];
        snprintf(cmd_with_stderr, sizeof(cmd_with_stderr), "%s 2>&1", cmd);
        FILE *proc = popen(cmd_with_stderr, "r");
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
        // Check if URL is GitHub and we have a token — add auth header
        char auth_header[600] = "";
        if (strstr(url, "github.com") || strstr(url, "api.github.com") || strstr(url, "raw.githubusercontent.com")) {
            if (!g_connections_loaded) load_connections();
            Connection *gh = get_connection("github");
            if (gh && gh->active)
                snprintf(auth_header, sizeof(auth_header),
                    "-H 'Authorization: token %s' -H 'Accept: application/vnd.github+json' ", gh->key);
        }
        char cmd[4096];
        snprintf(cmd, sizeof(cmd),
            "curl -sL --max-time 15 %s'%s' | textutil -stdin -format html -convert txt -stdout 2>/dev/null || curl -sL --max-time 15 %s'%s'",
            auth_header, url, auth_header, url);
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

    } else if (strcmp(name, "web_search") == 0) {
        const char *query = tool_call_get(tc, "query");
        if (!query) { out_len = snprintf(output, output_sz, "Error: no query provided"); return out_len; }
        Connection *c = get_connection("brave_search");
        if (!c || !c->active) {
            out_len = snprintf(output, output_sz,
                "Error: Brave Search not configured. Run /connections add brave_search");
            return out_len;
        }
        // URL-encode query (simple: spaces to +, escape dangerous chars)
        char encoded[1024] = {0};
        int ei = 0;
        for (int qi = 0; query[qi] && ei < (int)sizeof(encoded) - 4; qi++) {
            if (query[qi] == ' ') encoded[ei++] = '+';
            else if (query[qi] == '\'' || query[qi] == '`' || query[qi] == '$' || query[qi] == ';')
                continue; // skip injection chars
            else if (query[qi] == '&') { encoded[ei++] = '%'; encoded[ei++] = '2'; encoded[ei++] = '6'; }
            else encoded[ei++] = query[qi];
        }
        encoded[ei] = 0;

        printf(ANSI_DIM "  [searching: %s]" ANSI_RESET "\n", query);
        char cmd[2048];
        const char *count = tool_call_get(tc, "count");
        int n = count ? atoi(count) : 5;
        if (n < 1) n = 1; if (n > 20) n = 20;
        snprintf(cmd, sizeof(cmd),
            "curl -s --max-time 10 "
            "-H 'Accept: application/json' "
            "-H 'X-Subscription-Token: %s' "
            "'https://api.search.brave.com/res/v1/web/search?q=%s&count=%d' 2>/dev/null",
            c->key, encoded, n);

        FILE *proc = popen(cmd, "r");
        if (!proc) { out_len = snprintf(output, output_sz, "Error: failed to execute search"); return out_len; }

        // Read raw JSON response
        char *raw = malloc(65536);
        int raw_len = 0;
        while (raw_len < 65535) {
            int ch = fgetc(proc);
            if (ch == EOF) break;
            raw[raw_len++] = (char)ch;
        }
        raw[raw_len] = 0;
        pclose(proc);

        // Extract results into readable format
        // Look for "title":"..." and "url":"..." and "description":"..." patterns
        out_len = snprintf(output, output_sz, "Search results for: %s\n\n", query);
        int result_num = 0;
        char *scan = raw;
        while ((scan = (char *)json_find_key(scan, "title")) != NULL && result_num < n) {
            char title[256] = {0}, rurl[512] = {0}, desc[512] = {0};
            json_extract_str(scan, "title", title, sizeof(title));
            json_extract_str(scan, "url", rurl, sizeof(rurl));
            json_extract_str(scan, "description", desc, sizeof(desc));
            if (title[0] && rurl[0]) {
                result_num++;
                out_len += snprintf(output + out_len, output_sz - out_len,
                    "%d. %s\n   %s\n   %s\n\n", result_num, title, rurl, desc);
            }
            scan++;
        }
        free(raw);

        if (result_num == 0)
            out_len = snprintf(output, output_sz, "No search results found for: %s", query);

    } else if (strcmp(name, "github") == 0) {
        const char *action = tool_call_get(tc, "action");
        if (!action) { out_len = snprintf(output, output_sz, "Error: no action provided"); return out_len; }
        Connection *c = get_connection("github");
        if (!c || !c->active) {
            out_len = snprintf(output, output_sz,
                "Error: GitHub not configured. Run /connections add github");
            return out_len;
        }

        char cmd[2048];
        const char *repo = tool_call_get(tc, "repo");
        const char *query = tool_call_get(tc, "query");

        if (strcmp(action, "search_repos") == 0) {
            if (!query) { out_len = snprintf(output, output_sz, "Error: query required for search_repos"); return out_len; }
            printf(ANSI_DIM "  [github: searching repos for '%s']" ANSI_RESET "\n", query);
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 "
                "-H 'Authorization: token %s' -H 'Accept: application/vnd.github+json' "
                "'https://api.github.com/search/repositories?q=%s&per_page=5' 2>/dev/null",
                c->key, query);
        } else if (strcmp(action, "list_issues") == 0) {
            if (!repo) { out_len = snprintf(output, output_sz, "Error: repo required (owner/name)"); return out_len; }
            printf(ANSI_DIM "  [github: listing issues for %s]" ANSI_RESET "\n", repo);
            const char *state = tool_call_get(tc, "state");
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 "
                "-H 'Authorization: token %s' -H 'Accept: application/vnd.github+json' "
                "'https://api.github.com/repos/%s/issues?state=%s&per_page=10' 2>/dev/null",
                c->key, repo, (state && state[0]) ? state : "open");
        } else if (strcmp(action, "read_issue") == 0) {
            if (!repo) { out_len = snprintf(output, output_sz, "Error: repo required (owner/name)"); return out_len; }
            const char *number = tool_call_get(tc, "number");
            if (!number) { out_len = snprintf(output, output_sz, "Error: issue number required"); return out_len; }
            printf(ANSI_DIM "  [github: reading %s#%s]" ANSI_RESET "\n", repo, number);
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 "
                "-H 'Authorization: token %s' -H 'Accept: application/vnd.github+json' "
                "'https://api.github.com/repos/%s/issues/%s' 2>/dev/null",
                c->key, repo, number);
        } else if (strcmp(action, "list_prs") == 0) {
            if (!repo) { out_len = snprintf(output, output_sz, "Error: repo required (owner/name)"); return out_len; }
            printf(ANSI_DIM "  [github: listing PRs for %s]" ANSI_RESET "\n", repo);
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 "
                "-H 'Authorization: token %s' -H 'Accept: application/vnd.github+json' "
                "'https://api.github.com/repos/%s/pulls?per_page=10' 2>/dev/null",
                c->key, repo);
        } else if (strcmp(action, "user") == 0) {
            printf(ANSI_DIM "  [github: fetching user profile]" ANSI_RESET "\n");
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 "
                "-H 'Authorization: token %s' -H 'Accept: application/vnd.github+json' "
                "'https://api.github.com/user' 2>/dev/null",
                c->key);
        } else {
            out_len = snprintf(output, output_sz,
                "Error: unknown action '%s'. Available: search_repos, list_issues, read_issue, list_prs, user",
                action);
            return out_len;
        }

        FILE *proc = popen(cmd, "r");
        if (!proc) { out_len = snprintf(output, output_sz, "Error: failed to execute github request"); return out_len; }
        while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
            int ch = fgetc(proc);
            if (ch == EOF) break;
            output[out_len++] = (char)ch;
        }
        output[out_len] = 0;
        pclose(proc);
        if (out_len == 0)
            out_len = snprintf(output, output_sz, "Error: no response from GitHub API");

    } else if (strcmp(name, "wolfram") == 0) {
        const char *query = tool_call_get(tc, "query");
        if (!query) { out_len = snprintf(output, output_sz, "Error: no query provided"); return out_len; }
        Connection *c = get_connection("wolfram");
        if (!c || !c->active) {
            out_len = snprintf(output, output_sz,
                "Error: Wolfram Alpha not configured. Run /connections add wolfram");
            return out_len;
        }

        // URL-encode query
        char encoded[1024] = {0};
        int ei = 0;
        for (int qi = 0; query[qi] && ei < (int)sizeof(encoded) - 4; qi++) {
            if (query[qi] == ' ') { encoded[ei++] = '%'; encoded[ei++] = '2'; encoded[ei++] = '0'; }
            else if (query[qi] == '+') { encoded[ei++] = '%'; encoded[ei++] = '2'; encoded[ei++] = 'B'; }
            else if (query[qi] == '\'' || query[qi] == '`' || query[qi] == '$' || query[qi] == ';')
                continue;
            else encoded[ei++] = query[qi];
        }
        encoded[ei] = 0;

        printf(ANSI_DIM "  [wolfram: %s]" ANSI_RESET "\n", query);
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "curl -s --max-time 15 "
            "'https://api.wolframalpha.com/v1/result?appid=%s&i=%s' 2>/dev/null",
            c->key, encoded);

        FILE *proc = popen(cmd, "r");
        if (!proc) { out_len = snprintf(output, output_sz, "Error: failed to query Wolfram Alpha"); return out_len; }
        while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
            int ch = fgetc(proc);
            if (ch == EOF) break;
            output[out_len++] = (char)ch;
        }
        output[out_len] = 0;
        pclose(proc);
        if (out_len == 0)
            out_len = snprintf(output, output_sz, "Error: no response from Wolfram Alpha");

    } else if (strcmp(name, "gmail") == 0) {
        const char *action = tool_call_get(tc, "action");
        if (!action) { out_len = snprintf(output, output_sz, "Error: no action provided"); return out_len; }
        const char *account = tool_call_get(tc, "account");
        Connection *c = get_google_connection(account);
        if (!c || !c->active) {
            out_len = snprintf(output, output_sz,
                "Error: Google not configured. Run /connections add google");
            return out_len;
        }
        if (!oauth_ensure_token(c)) {
            out_len = snprintf(output, output_sz,
                "Error: Google token expired and refresh failed. Run /connections add google");
            return out_len;
        }

        char cmd[4096];
        if (strcmp(action, "search") == 0) {
            const char *query = tool_call_get(tc, "query");
            if (!query) { out_len = snprintf(output, output_sz, "Error: query required for search"); return out_len; }
            // URL-encode query
            char encoded[512] = {0};
            int ei = 0;
            for (int qi = 0; query[qi] && ei < (int)sizeof(encoded) - 4; qi++) {
                if (query[qi] == ' ') { encoded[ei++] = '%'; encoded[ei++] = '2'; encoded[ei++] = '0'; }
                else if (query[qi] == '\'') continue;
                else encoded[ei++] = query[qi];
            }
            encoded[ei] = 0;

            const char *max_results = tool_call_get(tc, "max_results");
            int mr = max_results ? atoi(max_results) : 10;
            if (mr < 1) mr = 1; if (mr > 50) mr = 50;

            printf(ANSI_DIM "  [gmail: searching '%s']" ANSI_RESET "\n", query);
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 "
                "-H 'Authorization: Bearer %s' "
                "'https://gmail.googleapis.com/gmail/v1/users/me/messages?q=%s&maxResults=%d' 2>/dev/null",
                c->access_token, encoded, mr);

            FILE *proc = popen(cmd, "r");
            if (!proc) { out_len = snprintf(output, output_sz, "Error: failed to search Gmail"); return out_len; }
            char *raw = malloc(65536);
            int raw_len = 0;
            while (raw_len < 65535) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                raw[raw_len++] = (char)ch;
            }
            raw[raw_len] = 0;
            pclose(proc);

            // Extract message IDs and fetch snippets
            out_len = snprintf(output, output_sz, "Gmail search: %s\n\n", query);
            int msg_num = 0;
            char *scan = raw;
            while ((scan = (char *)json_find_key(scan, "id")) != NULL && msg_num < mr) {
                char msg_id[64] = {0};
                json_extract_str(scan, "id", msg_id, sizeof(msg_id));
                if (!msg_id[0]) { scan++; continue; }

                // Fetch message metadata
                char fetch_cmd[2048];
                snprintf(fetch_cmd, sizeof(fetch_cmd),
                    "curl -s --max-time 5 "
                    "-H 'Authorization: Bearer %s' "
                    "'https://gmail.googleapis.com/gmail/v1/users/me/messages/%s?format=metadata"
                    "&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date' 2>/dev/null",
                    c->access_token, msg_id);

                FILE *mp = popen(fetch_cmd, "r");
                if (mp) {
                    char meta[8192];
                    int mlen = 0;
                    while (mlen < (int)sizeof(meta) - 1) {
                        int ch = fgetc(mp);
                        if (ch == EOF) break;
                        meta[mlen++] = (char)ch;
                    }
                    meta[mlen] = 0;
                    pclose(mp);

                    char snippet[512] = {0};
                    json_extract_str(meta, "snippet", snippet, sizeof(snippet));

                    // Extract headers (From, Subject, Date)
                    char from[256] = "?", subject[256] = "(no subject)", date[64] = "";
                    char *hp = meta;
                    while ((hp = (char *)json_find_key(hp, "name")) != NULL) {
                        char hname[32] = {0}, hval[256] = {0};
                        json_extract_str(hp, "name", hname, sizeof(hname));
                        json_extract_str(hp, "value", hval, sizeof(hval));
                        if (strcasecmp(hname, "From") == 0) strlcpy(from, hval, sizeof(from));
                        else if (strcasecmp(hname, "Subject") == 0) strlcpy(subject, hval, sizeof(subject));
                        else if (strcasecmp(hname, "Date") == 0) strlcpy(date, hval, sizeof(date));
                        hp++;
                    }

                    msg_num++;
                    out_len += snprintf(output + out_len, output_sz - out_len,
                        "%d. [%s] %s\n   From: %s\n   Date: %s\n   %s\n\n",
                        msg_num, msg_id, subject, from, date, snippet);
                }
                scan++;
            }
            free(raw);
            if (msg_num == 0)
                out_len += snprintf(output + out_len, output_sz - out_len, "No messages found.");

        } else if (strcmp(action, "read") == 0) {
            const char *msg_id = tool_call_get(tc, "id");
            if (!msg_id) { out_len = snprintf(output, output_sz, "Error: message id required"); return out_len; }
            printf(ANSI_DIM "  [gmail: reading message %s]" ANSI_RESET "\n", msg_id);
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 "
                "-H 'Authorization: Bearer %s' "
                "'https://gmail.googleapis.com/gmail/v1/users/me/messages/%s?format=full' 2>/dev/null",
                c->access_token, msg_id);

            FILE *proc = popen(cmd, "r");
            if (!proc) { out_len = snprintf(output, output_sz, "Error: failed to read message"); return out_len; }
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);

        } else if (strcmp(action, "labels") == 0) {
            printf(ANSI_DIM "  [gmail: listing labels]" ANSI_RESET "\n");
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 "
                "-H 'Authorization: Bearer %s' "
                "'https://gmail.googleapis.com/gmail/v1/users/me/labels' 2>/dev/null",
                c->access_token);

            FILE *proc = popen(cmd, "r");
            if (!proc) { out_len = snprintf(output, output_sz, "Error: failed to list labels"); return out_len; }
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);

        } else if (strcmp(action, "profile") == 0) {
            printf(ANSI_DIM "  [gmail: fetching profile]" ANSI_RESET "\n");
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 "
                "-H 'Authorization: Bearer %s' "
                "'https://gmail.googleapis.com/gmail/v1/users/me/profile' 2>/dev/null",
                c->access_token);

            FILE *proc = popen(cmd, "r");
            if (!proc) { out_len = snprintf(output, output_sz, "Error: failed to fetch profile"); return out_len; }
            while (out_len < (int)output_sz - 1 && out_len < MAX_TOOL_RESPONSE) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);

        } else if (strcmp(action, "send") == 0 || strcmp(action, "draft") == 0) {
            const char *to = tool_call_get(tc, "to");
            const char *subject = tool_call_get(tc, "subject");
            const char *body = tool_call_get(tc, "body");
            if (!to || !subject || !body) {
                out_len = snprintf(output, output_sz, "Error: to, subject, and body required");
                return out_len;
            }
            const char *cc = tool_call_get(tc, "cc");
            const char *bcc = tool_call_get(tc, "bcc");

            // Build RFC 2822 message
            char *raw_msg = malloc(65536);
            int rlen = 0;
            rlen += snprintf(raw_msg + rlen, 65536 - rlen, "To: %s\r\n", to);
            if (cc && cc[0]) rlen += snprintf(raw_msg + rlen, 65536 - rlen, "Cc: %s\r\n", cc);
            if (bcc && bcc[0]) rlen += snprintf(raw_msg + rlen, 65536 - rlen, "Bcc: %s\r\n", bcc);
            rlen += snprintf(raw_msg + rlen, 65536 - rlen,
                "Subject: %s\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n%s",
                subject, body);

            // Base64url encode (Gmail API requires URL-safe base64)
            @autoreleasepool {
                NSData *data = [NSData dataWithBytes:raw_msg length:rlen];
                NSString *b64 = [data base64EncodedStringWithOptions:0];
                // Convert to base64url: + → -, / → _, strip =
                NSString *b64url = [[b64 stringByReplacingOccurrencesOfString:@"+" withString:@"-"]
                                         stringByReplacingOccurrencesOfString:@"/" withString:@"_"];
                b64url = [b64url stringByReplacingOccurrencesOfString:@"=" withString:@""];

                int is_draft = (strcmp(action, "draft") == 0);
                printf(ANSI_DIM "  [gmail: %s to %s]" ANSI_RESET "\n",
                       is_draft ? "creating draft" : "sending", to);

                // Write JSON body to temp file (too large for command line)
                char tmp[128];
                snprintf(tmp, sizeof(tmp), "/tmp/pre_gmail_%d.json", getpid());
                FILE *tf = fopen(tmp, "w");
                if (tf) {
                    if (is_draft)
                        fprintf(tf, "{\"message\":{\"raw\":\"%s\"}}", [b64url UTF8String]);
                    else
                        fprintf(tf, "{\"raw\":\"%s\"}", [b64url UTF8String]);
                    fclose(tf);

                    const char *endpoint = is_draft
                        ? "https://gmail.googleapis.com/gmail/v1/users/me/drafts"
                        : "https://gmail.googleapis.com/gmail/v1/users/me/messages/send";

                    snprintf(cmd, sizeof(cmd),
                        "curl -s --max-time 15 -X POST "
                        "-H 'Authorization: Bearer %s' "
                        "-H 'Content-Type: application/json' "
                        "-d @%s '%s' 2>/dev/null",
                        c->access_token, tmp, endpoint);

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
                }
            }
            free(raw_msg);
            if (out_len == 0)
                out_len = snprintf(output, output_sz, "Error: failed to %s message",
                                   strcmp(action, "draft") == 0 ? "draft" : "send");

        } else if (strcmp(action, "trash") == 0) {
            const char *msg_id = tool_call_get(tc, "id");
            if (!msg_id) { out_len = snprintf(output, output_sz, "Error: message id required"); return out_len; }
            printf(ANSI_DIM "  [gmail: trashing message %s]" ANSI_RESET "\n", msg_id);
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -X POST "
                "-H 'Authorization: Bearer %s' "
                "'https://gmail.googleapis.com/gmail/v1/users/me/messages/%s/trash' 2>/dev/null",
                c->access_token, msg_id);

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
                out_len = snprintf(output, output_sz, "Error: failed to trash message");

        } else {
            out_len = snprintf(output, output_sz,
                "Error: unknown action '%s'. Available: search, read, send, draft, trash, labels, profile", action);
        }

    } else if (strcmp(name, "gdrive") == 0) {
        const char *action = tool_call_get(tc, "action");
        if (!action) { out_len = snprintf(output, output_sz, "Error: no action provided"); return out_len; }
        const char *account = tool_call_get(tc, "account");
        Connection *c = get_google_connection(account);
        if (!c || !c->active) {
            out_len = snprintf(output, output_sz, "Error: Google not configured. Run /connections add google");
            return out_len;
        }
        if (!oauth_ensure_token(c)) {
            out_len = snprintf(output, output_sz, "Error: Google token expired. Run /connections add google");
            return out_len;
        }

        char cmd[4096];

        if (strcmp(action, "list") == 0) {
            const char *folder_id = tool_call_get(tc, "folder_id");
            const char *count = tool_call_get(tc, "count");
            int n = count ? atoi(count) : 20;
            if (n < 1) n = 1; if (n > 100) n = 100;
            printf(ANSI_DIM "  [gdrive: listing files]" ANSI_RESET "\n");
            if (folder_id && folder_id[0]) {
                snprintf(cmd, sizeof(cmd),
                    "curl -s --max-time 10 "
                    "-H 'Authorization: Bearer %s' "
                    "'https://www.googleapis.com/drive/v3/files?q=%%27%s%%27+in+parents&pageSize=%d"
                    "&fields=files(id,name,mimeType,size,modifiedTime,webViewLink)' 2>/dev/null",
                    c->access_token, folder_id, n);
            } else {
                snprintf(cmd, sizeof(cmd),
                    "curl -s --max-time 10 "
                    "-H 'Authorization: Bearer %s' "
                    "'https://www.googleapis.com/drive/v3/files?pageSize=%d"
                    "&fields=files(id,name,mimeType,size,modifiedTime,webViewLink)"
                    "&orderBy=modifiedTime+desc' 2>/dev/null",
                    c->access_token, n);
            }

        } else if (strcmp(action, "search") == 0) {
            const char *query = tool_call_get(tc, "query");
            if (!query) { out_len = snprintf(output, output_sz, "Error: query required"); return out_len; }
            char encoded[512] = {0};
            int ei = 0;
            for (int qi = 0; query[qi] && ei < (int)sizeof(encoded) - 4; qi++) {
                if (query[qi] == ' ') { encoded[ei++] = '%'; encoded[ei++] = '2'; encoded[ei++] = '0'; }
                else if (query[qi] == '\'') { encoded[ei++] = '%'; encoded[ei++] = '2'; encoded[ei++] = '7'; }
                else encoded[ei++] = query[qi];
            }
            encoded[ei] = 0;
            printf(ANSI_DIM "  [gdrive: searching '%s']" ANSI_RESET "\n", query);
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 "
                "-H 'Authorization: Bearer %s' "
                "'https://www.googleapis.com/drive/v3/files?q=name+contains+%%27%s%%27"
                "&pageSize=10&fields=files(id,name,mimeType,size,modifiedTime,webViewLink)' 2>/dev/null",
                c->access_token, encoded);

        } else if (strcmp(action, "download") == 0) {
            const char *file_id = tool_call_get(tc, "id");
            const char *dest = tool_call_get(tc, "path");
            if (!file_id) { out_len = snprintf(output, output_sz, "Error: id required"); return out_len; }
            if (!dest) { out_len = snprintf(output, output_sz, "Error: path required (local destination)"); return out_len; }
            char resolved[PATH_MAX];
            resolve_path(dest, resolved, sizeof(resolved));
            printf(ANSI_DIM "  [gdrive: downloading %s → %s]" ANSI_RESET "\n", file_id, resolved);
            snprintf(cmd, sizeof(cmd),
                "curl -sL --max-time 30 "
                "-H 'Authorization: Bearer %s' "
                "'https://www.googleapis.com/drive/v3/files/%s?alt=media' -o '%s' 2>/dev/null && echo 'Downloaded to %s'",
                c->access_token, file_id, resolved, resolved);

        } else if (strcmp(action, "upload") == 0) {
            const char *src = tool_call_get(tc, "path");
            const char *fname = tool_call_get(tc, "name");
            const char *folder_id = tool_call_get(tc, "folder_id");
            if (!src) { out_len = snprintf(output, output_sz, "Error: path required (local file)"); return out_len; }
            char resolved[PATH_MAX];
            resolve_path(src, resolved, sizeof(resolved));
            if (!fname) fname = strrchr(resolved, '/') ? strrchr(resolved, '/') + 1 : resolved;

            printf(ANSI_DIM "  [gdrive: uploading %s]" ANSI_RESET "\n", fname);

            // Build metadata JSON
            char meta_json[512];
            if (folder_id && folder_id[0])
                snprintf(meta_json, sizeof(meta_json),
                    "{\"name\":\"%s\",\"parents\":[\"%s\"]}", fname, folder_id);
            else
                snprintf(meta_json, sizeof(meta_json), "{\"name\":\"%s\"}", fname);

            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 30 -X POST "
                "-H 'Authorization: Bearer %s' "
                "-F 'metadata=%s;type=application/json' "
                "-F 'file=@%s' "
                "'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
                "&fields=id,name,webViewLink' 2>/dev/null",
                c->access_token, meta_json, resolved);

        } else if (strcmp(action, "mkdir") == 0) {
            const char *fname = tool_call_get(tc, "name");
            const char *parent = tool_call_get(tc, "folder_id");
            if (!fname) { out_len = snprintf(output, output_sz, "Error: name required"); return out_len; }
            printf(ANSI_DIM "  [gdrive: creating folder '%s']" ANSI_RESET "\n", fname);
            char tmp[256];
            snprintf(tmp, sizeof(tmp), "/tmp/pre_gdrive_%d.json", getpid());
            FILE *tf = fopen(tmp, "w");
            if (!tf) { out_len = snprintf(output, output_sz, "Error: temp file failed"); return out_len; }
            if (parent && parent[0])
                fprintf(tf, "{\"name\":\"%s\",\"mimeType\":\"application/vnd.google-apps.folder\",\"parents\":[\"%s\"]}", fname, parent);
            else
                fprintf(tf, "{\"name\":\"%s\",\"mimeType\":\"application/vnd.google-apps.folder\"}", fname);
            fclose(tf);
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -X POST "
                "-H 'Authorization: Bearer %s' "
                "-H 'Content-Type: application/json' "
                "-d @%s 'https://www.googleapis.com/drive/v3/files?fields=id,name,webViewLink' 2>/dev/null && rm -f %s",
                c->access_token, tmp, tmp);

        } else if (strcmp(action, "share") == 0) {
            const char *file_id = tool_call_get(tc, "id");
            const char *email = tool_call_get(tc, "email");
            const char *role = tool_call_get(tc, "role");
            if (!file_id || !email) { out_len = snprintf(output, output_sz, "Error: id and email required"); return out_len; }
            if (!role) role = "reader";
            printf(ANSI_DIM "  [gdrive: sharing %s with %s as %s]" ANSI_RESET "\n", file_id, email, role);
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -X POST "
                "-H 'Authorization: Bearer %s' "
                "-H 'Content-Type: application/json' "
                "-d '{\"role\":\"%s\",\"type\":\"user\",\"emailAddress\":\"%s\"}' "
                "'https://www.googleapis.com/drive/v3/files/%s/permissions' 2>/dev/null",
                c->access_token, role, email, file_id);

        } else if (strcmp(action, "delete") == 0) {
            const char *file_id = tool_call_get(tc, "id");
            if (!file_id) { out_len = snprintf(output, output_sz, "Error: id required"); return out_len; }
            printf(ANSI_DIM "  [gdrive: deleting %s]" ANSI_RESET "\n", file_id);
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -X DELETE "
                "-H 'Authorization: Bearer %s' "
                "'https://www.googleapis.com/drive/v3/files/%s' -w '%%{http_code}' 2>/dev/null",
                c->access_token, file_id);

        } else {
            out_len = snprintf(output, output_sz,
                "Error: unknown action '%s'. Available: list, search, download, upload, mkdir, share, delete", action);
            return out_len;
        }

        // Execute (for actions that didn't return early)
        if (out_len == 0) {
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
                out_len = snprintf(output, output_sz, "Error: no response from Google Drive API");
        }

    } else if (strcmp(name, "gdocs") == 0) {
        const char *action = tool_call_get(tc, "action");
        if (!action) { out_len = snprintf(output, output_sz, "Error: no action provided"); return out_len; }
        const char *account = tool_call_get(tc, "account");
        Connection *c = get_google_connection(account);
        if (!c || !c->active) {
            out_len = snprintf(output, output_sz, "Error: Google not configured. Run /connections add google");
            return out_len;
        }
        if (!oauth_ensure_token(c)) {
            out_len = snprintf(output, output_sz, "Error: Google token expired. Run /connections add google");
            return out_len;
        }

        char cmd[4096];

        if (strcmp(action, "create") == 0) {
            const char *title = tool_call_get(tc, "title");
            const char *content = tool_call_get(tc, "content");
            if (!title) { out_len = snprintf(output, output_sz, "Error: title required"); return out_len; }
            printf(ANSI_DIM "  [gdocs: creating '%s']" ANSI_RESET "\n", title);

            // Step 1: Create empty doc
            char tmp[128];
            snprintf(tmp, sizeof(tmp), "/tmp/pre_gdocs_%d.json", getpid());
            FILE *tf = fopen(tmp, "w");
            if (!tf) { out_len = snprintf(output, output_sz, "Error: temp file failed"); return out_len; }
            fprintf(tf, "{\"title\":\"%s\"}", title);
            fclose(tf);

            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -X POST "
                "-H 'Authorization: Bearer %s' "
                "-H 'Content-Type: application/json' "
                "-d @%s 'https://docs.googleapis.com/v1/documents' 2>/dev/null",
                c->access_token, tmp);

            FILE *proc = popen(cmd, "r");
            char create_resp[8192] = {0};
            int cr_len = 0;
            if (proc) {
                while (cr_len < (int)sizeof(create_resp) - 1) {
                    int ch = fgetc(proc);
                    if (ch == EOF) break;
                    create_resp[cr_len++] = (char)ch;
                }
                create_resp[cr_len] = 0;
                pclose(proc);
            }
            remove(tmp);

            char doc_id[128] = {0};
            json_extract_str(create_resp, "documentId", doc_id, sizeof(doc_id));

            // Step 2: Insert content if provided
            if (content && content[0] && doc_id[0]) {
                char *esc_content = json_escape_alloc(content);
                if (esc_content) {
                    tf = fopen(tmp, "w");
                    if (tf) {
                        fprintf(tf, "{\"requests\":[{\"insertText\":{\"location\":{\"index\":1},\"text\":\"%s\"}}]}", esc_content);
                        fclose(tf);
                        snprintf(cmd, sizeof(cmd),
                            "curl -s --max-time 10 -X POST "
                            "-H 'Authorization: Bearer %s' "
                            "-H 'Content-Type: application/json' "
                            "-d @%s 'https://docs.googleapis.com/v1/documents/%s:batchUpdate' 2>/dev/null",
                            c->access_token, tmp, doc_id);
                        system(cmd); // fire and forget, we already have the doc
                        remove(tmp);
                    }
                    free(esc_content);
                }
            }

            // Return creation result
            out_len = snprintf(output, output_sz, "%s", create_resp);

        } else if (strcmp(action, "read") == 0) {
            const char *doc_id = tool_call_get(tc, "id");
            if (!doc_id) { out_len = snprintf(output, output_sz, "Error: document id required"); return out_len; }
            printf(ANSI_DIM "  [gdocs: reading %s]" ANSI_RESET "\n", doc_id);
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 "
                "-H 'Authorization: Bearer %s' "
                "'https://docs.googleapis.com/v1/documents/%s' 2>/dev/null",
                c->access_token, doc_id);

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

        } else if (strcmp(action, "append") == 0) {
            const char *doc_id = tool_call_get(tc, "id");
            const char *content = tool_call_get(tc, "content");
            if (!doc_id || !content) {
                out_len = snprintf(output, output_sz, "Error: id and content required");
                return out_len;
            }
            printf(ANSI_DIM "  [gdocs: appending to %s]" ANSI_RESET "\n", doc_id);

            char *esc_content = json_escape_alloc(content);
            if (!esc_content) { out_len = snprintf(output, output_sz, "Error: encoding failed"); return out_len; }

            char tmp[128];
            snprintf(tmp, sizeof(tmp), "/tmp/pre_gdocs_%d.json", getpid());
            FILE *tf = fopen(tmp, "w");
            if (!tf) { free(esc_content); out_len = snprintf(output, output_sz, "Error: temp file failed"); return out_len; }
            fprintf(tf, "{\"requests\":[{\"insertText\":{\"endOfSegmentLocation\":{\"segmentId\":\"\"},\"text\":\"%s\"}}]}", esc_content);
            fclose(tf);
            free(esc_content);

            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -X POST "
                "-H 'Authorization: Bearer %s' "
                "-H 'Content-Type: application/json' "
                "-d @%s 'https://docs.googleapis.com/v1/documents/%s:batchUpdate' 2>/dev/null",
                c->access_token, tmp, doc_id);

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

        } else {
            out_len = snprintf(output, output_sz,
                "Error: unknown action '%s'. Available: create, read, append", action);
        }

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
        for (int i = 0; i < g_memory_count && out_len < (int)output_sz - 1024; i++) {
            out_len += snprintf(output + out_len, output_sz - out_len,
                "%d. [%s] %s — %s\n",
                i + 1, g_memories[i].type, g_memories[i].name, g_memories[i].description);
            char *body = read_memory_body(g_memories[i].file);
            if (body && body[0]) {
                char preview[256];
                strlcpy(preview, body, sizeof(preview));
                out_len += snprintf(output + out_len, output_sz - out_len, "   %s\n\n", preview);
            }
            free(body);
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

    } else if (strcmp(name, "artifact") == 0) {
        const char *atitle = tool_call_get(tc, "title");
        const char *acontent = tool_call_get(tc, "content");
        const char *atype = tool_call_get(tc, "type");
        // Auto-detect type from content if not provided
        if (!atype || !atype[0]) {
            if (acontent && (strstr(acontent, "<html") || strstr(acontent, "<!DOCTYPE") ||
                strstr(acontent, "<div") || strstr(acontent, "<script")))
                atype = "html";
            else if (acontent && acontent[0] == '{')
                atype = "json";
            else if (acontent && (strstr(acontent, "<svg") || strstr(acontent, "xmlns=\"http://www.w3.org/2000/svg")))
                atype = "svg";
            else
                atype = "html";
        }
        // Default title from timestamp if not provided
        if (!atitle || !atitle[0]) {
            static char auto_title[64];
            snprintf(auto_title, sizeof(auto_title), "artifact_%ld", (long)time(NULL));
            atitle = auto_title;
        }
        const char *aappend = tool_call_get(tc, "append_to");
        out_len = execute_artifact(atitle, acontent, atype, aappend, output, output_sz);

    } else if (strcmp(name, "image_generate") == 0) {
        const char *iprompt = tool_call_get(tc, "prompt");
        const char *istyle = tool_call_get(tc, "style");
        const char *iwidth_s = tool_call_get(tc, "width");
        const char *iheight_s = tool_call_get(tc, "height");

        if (!iprompt || !iprompt[0]) {
            out_len = snprintf(output, output_sz, "Error: prompt is required");
        } else if (!g_comfyui_installed) {
            out_len = snprintf(output, output_sz,
                "Error: ComfyUI is not installed. Run the install script with image generation "
                "support, or manually install ComfyUI to ~/.pre/comfyui/ and create ~/.pre/comfyui.json");
        } else {
            int iw = iwidth_s ? atoi(iwidth_s) : 512;
            int ih = iheight_s ? atoi(iheight_s) : 512;

            // Prepend style hint to prompt if provided
            char full_prompt[2048];
            if (istyle && istyle[0]) {
                snprintf(full_prompt, sizeof(full_prompt), "%s %s", istyle, iprompt);
            } else {
                strlcpy(full_prompt, iprompt, sizeof(full_prompt));
            }

            char img_path[PATH_MAX] = {0};
            int rc = comfyui_generate(full_prompt, NULL, iw, ih, img_path, sizeof(img_path));
            if (rc == 0) {
                printf(ANSI_CYAN "  ◆ Image: %s" ANSI_RESET "\n", img_path);
                printf("    \033]8;;file://%s\033\\", img_path);
                printf(ANSI_BOLD ANSI_CYAN "▸ %s" ANSI_RESET, img_path);
                printf("\033]8;;\033\\\n");

                out_len = snprintf(output, output_sz,
                    "Image generated successfully: %s\n"
                    "To use in an HTML artifact, reference it as: <img src='file://%s'>\n"
                    "The image is saved locally and visible in the terminal link above.",
                    img_path, img_path);
            } else {
                out_len = snprintf(output, output_sz,
                    "Error: image generation failed. Check that ComfyUI is running "
                    "(port %d) and the SDXL model is loaded. See ~/.pre/comfyui.log for details.",
                    g_comfyui_port);
            }
        }

    } else if (strcmp(name, "pdf_export") == 0) {
        const char *ptitle = tool_call_get(tc, "title");
        const char *ppath = tool_call_get(tc, "path");

        // Find the artifact to export
        int found = -1;
        if (ptitle && (strcmp(ptitle, "latest") == 0 || strcmp(ptitle, "last") == 0)) {
            if (g_artifact_count > 0) found = g_artifact_count - 1;
        } else if (ptitle) {
            for (int i = g_artifact_count - 1; i >= 0; i--) {
                if (strcasestr(g_artifacts[i].title, ptitle)) { found = i; break; }
            }
        }
        if (found < 0) {
            out_len = snprintf(output, output_sz, "Error: no artifact matching '%s' found. "
                "Create an artifact first, then export it.", ptitle ?: "(none)");
        } else {
            // Build output PDF path
            char pdf_path[PATH_MAX];
            if (ppath && ppath[0]) {
                strlcpy(pdf_path, ppath, sizeof(pdf_path));
            } else {
                // Same directory as HTML, with .pdf extension
                strlcpy(pdf_path, g_artifacts[found].path, sizeof(pdf_path));
                char *dot = strrchr(pdf_path, '.');
                if (dot) strlcpy(dot, ".pdf", sizeof(pdf_path) - (dot - pdf_path));
                else strlcat(pdf_path, ".pdf", sizeof(pdf_path));
            }

            printf(ANSI_DIM "  [exporting to PDF...]" ANSI_RESET "\n");
            int rc = export_to_pdf(g_artifacts[found].path, pdf_path);
            if (rc == 0) {
                struct stat pst;
                long long psz = 0;
                if (stat(pdf_path, &pst) == 0) psz = (long long)pst.st_size;

                printf(ANSI_CYAN "  ◆ PDF exported: %s" ANSI_RESET " (%lld bytes)\n", pdf_path, psz);
                // Print clickable link
                printf("    \033]8;;file://%s\033\\", pdf_path);
                printf(ANSI_BOLD ANSI_CYAN "▸ %s" ANSI_RESET, pdf_path);
                printf("\033]8;;\033\\\n");

                out_len = snprintf(output, output_sz,
                    "PDF exported successfully: %s (%lld bytes)\n"
                    "The user can click the terminal link to open it.", pdf_path, psz);
            } else {
                out_len = snprintf(output, output_sz,
                    "Error: PDF export failed. The artifact HTML may have issues, "
                    "or macOS 13+ is required for WebKit PDF export.");
            }
        }

    } else if (strcmp(name, "cron") == 0) {
        const char *action = tool_call_get(tc, "action");
        if (!action) action = "list";

        // Ensure cron jobs are loaded
        if (g_cron_count == 0 && g_cron_last_check_ms == 0) {
            cron_load();
            g_cron_last_check_ms = now_ms();
        }

        if (strcmp(action, "add") == 0) {
            const char *sched = tool_call_get(tc, "schedule");
            const char *prompt_text = tool_call_get(tc, "prompt");
            const char *desc = tool_call_get(tc, "description");
            if (!sched || !prompt_text) {
                out_len = snprintf(output, output_sz, "Error: cron add requires 'schedule' and 'prompt'");
            } else if (g_cron_count >= MAX_CRON_JOBS) {
                out_len = snprintf(output, output_sz, "Error: maximum cron jobs (%d) reached", MAX_CRON_JOBS);
            } else {
                CronJob *j = &g_cron_jobs[g_cron_count];
                memset(j, 0, sizeof(CronJob));
                cron_generate_id(j->id, sizeof(j->id));
                strlcpy(j->schedule, sched, sizeof(j->schedule));
                strlcpy(j->prompt, prompt_text, sizeof(j->prompt));
                if (desc && desc[0]) {
                    strlcpy(j->description, desc, sizeof(j->description));
                } else {
                    size_t dlen = strlen(prompt_text);
                    if (dlen > 60) { memcpy(j->description, prompt_text, 57); strcpy(j->description + 57, "..."); }
                    else strlcpy(j->description, prompt_text, sizeof(j->description));
                }
                j->enabled = 1;
                j->created_at = time(NULL);
                g_cron_count++;
                cron_save();
                out_len = snprintf(output, output_sz, "Created cron job %s (schedule: %s)", j->id, j->schedule);
                printf(ANSI_GREEN "  [cron: created %s — %s]" ANSI_RESET "\n", j->id, j->schedule);
            }
        } else if (strcmp(action, "list") == 0) {
            out_len = 0;
            if (g_cron_count == 0) {
                out_len = snprintf(output, output_sz, "No cron jobs configured.");
            } else {
                for (int ci = 0; ci < g_cron_count && out_len < (int)output_sz - 256; ci++) {
                    CronJob *j = &g_cron_jobs[ci];
                    out_len += snprintf(output + out_len, output_sz - out_len,
                        "%s [%s] %s — %s (runs: %d, %s)\n",
                        j->id, j->schedule, j->enabled ? "enabled" : "DISABLED",
                        j->description, j->run_count,
                        j->last_run_at ? "has run" : "never run");
                }
            }
        } else if (strcmp(action, "remove") == 0) {
            const char *job_id = tool_call_get(tc, "id");
            if (!job_id) {
                out_len = snprintf(output, output_sz, "Error: cron remove requires 'id'");
            } else {
                int found = 0;
                for (int ci = 0; ci < g_cron_count; ci++) {
                    if (strcmp(g_cron_jobs[ci].id, job_id) == 0) {
                        out_len = snprintf(output, output_sz, "Removed cron job %s: %s", job_id, g_cron_jobs[ci].description);
                        for (int k = ci; k < g_cron_count - 1; k++) g_cron_jobs[k] = g_cron_jobs[k + 1];
                        g_cron_count--;
                        cron_save();
                        found = 1;
                        break;
                    }
                }
                if (!found) out_len = snprintf(output, output_sz, "Error: no cron job with id '%s'", job_id);
            }
        } else if (strcmp(action, "enable") == 0 || strcmp(action, "disable") == 0) {
            const char *job_id = tool_call_get(tc, "id");
            int enable = (strcmp(action, "enable") == 0);
            if (!job_id) {
                out_len = snprintf(output, output_sz, "Error: cron %s requires 'id'", action);
            } else {
                int found = 0;
                for (int ci = 0; ci < g_cron_count; ci++) {
                    if (strcmp(g_cron_jobs[ci].id, job_id) == 0) {
                        g_cron_jobs[ci].enabled = enable;
                        cron_save();
                        out_len = snprintf(output, output_sz, "%s cron job %s",
                                           enable ? "Enabled" : "Disabled", job_id);
                        found = 1;
                        break;
                    }
                }
                if (!found) out_len = snprintf(output, output_sz, "Error: no cron job with id '%s'", job_id);
            }
        } else {
            out_len = snprintf(output, output_sz, "Error: unknown cron action '%s'. Use: add|list|remove|enable|disable", action);
        }

    } else {
        out_len = snprintf(output, output_sz, "Error: unknown tool '%s'", name);
    }

    // Truncate tool results to prevent context bloat (inspired by LocalClaw's 8000-char cap).
    // Large tool outputs (file reads, grep results) get saved to session and replayed every turn.
    #define MAX_TOOL_RESULT_CHARS 8000
    if (out_len > MAX_TOOL_RESULT_CHARS) {
        snprintf(output + MAX_TOOL_RESULT_CHARS - 80, 80,
                 "\n...[truncated — %d chars total, showing first %d]",
                 out_len, MAX_TOOL_RESULT_CHARS - 80);
        out_len = (int)strlen(output);
    }

    return out_len;
}

// Extract raw JSON tool calls from text when model doesn't use <tool_call> tags.
// Looks for patterns like: {"name":"tool_name","arguments":{...}} or {"name":"tool_name","parameters":{...}}
// Returns 1 if a tool call was found and wrapped in <tool_call> tags in the response.
static int extract_json_tool_calls(char **response_ptr) {
    char *response = *response_ptr;
    if (!response || strstr(response, "<tool_call>")) return 0; // already has tags

    // Scan for JSON objects that look like tool calls
    char *scan = response;
    char *best_start = NULL;
    int best_len = 0;

    while (*scan) {
        // Find a { that might start a tool call JSON
        char *brace = strchr(scan, '{');
        if (!brace) break;

        // Quick check: does it contain "name" near the start?
        char *name_key = strstr(brace, "\"name\"");
        if (!name_key || name_key - brace > 30) { scan = brace + 1; continue; }

        // Check for "arguments" or "parameters"
        char *args_key = strstr(brace, "\"arguments\"");
        if (!args_key) args_key = strstr(brace, "\"parameters\"");
        if (!args_key) args_key = strstr(brace, "\"params\"");
        if (!args_key) { scan = brace + 1; continue; }

        // Find the matching closing brace (counting depth)
        int depth = 0;
        int in_string = 0;
        const char *p = brace;
        while (*p) {
            if (*p == '"' && (p == brace || *(p-1) != '\\')) in_string = !in_string;
            if (!in_string) {
                if (*p == '{') depth++;
                else if (*p == '}') { depth--; if (depth == 0) break; }
            }
            p++;
        }
        if (*p == '}' && depth == 0) {
            int len = (int)(p - brace + 1);
            if (len > best_len) {
                best_start = brace;
                best_len = len;
            }
        }
        scan = brace + 1;
    }

    if (!best_start || best_len < 20) return 0;

    // Verify it parses as a valid tool call
    char *tc_body = malloc(best_len + 1);
    memcpy(tc_body, best_start, best_len);
    tc_body[best_len] = 0;

    ToolCall tc;
    if (!extract_tool_call_v2(tc_body, &tc) || !tc.name[0]) {
        free(tc_body);
        return 0;
    }
    tool_call_free(&tc);

    // Wrap the JSON in <tool_call> tags so the normal parser handles it
    printf(ANSI_YELLOW "  [found raw JSON tool call for '%s' — wrapping in tags]" ANSI_RESET "\n",
           tc.name);

    size_t resp_len = strlen(response);
    size_t new_len = resp_len + 25 + best_len; // <tool_call>\n...\n</tool_call>
    char *new_response = malloc(new_len + 1);

    // Copy everything before the JSON
    size_t prefix_len = best_start - response;
    memcpy(new_response, response, prefix_len);
    int pos = (int)prefix_len;

    // Insert <tool_call> wrapper
    pos += sprintf(new_response + pos, "<tool_call>\n%s\n</tool_call>", tc_body);

    // Copy everything after the JSON
    char *after = best_start + best_len;
    strcpy(new_response + pos, after);

    free(tc_body);
    free(*response_ptr);
    *response_ptr = new_response;
    return 1;
}

// Strip artifact HTML content from ALL assistant turns in the session file.
// Replaces the content value with a short placeholder to prevent context bloat.
// Works on the raw JSONL bytes — no JSON parsing needed.
static void compact_artifact_content(const char *session_id) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.jsonl", g.sessions_dir, session_id);

    FILE *f = fopen(path, "r");
    if (!f) return;

    // Read all lines
    char **lines = NULL;
    int nlines = 0;
    char buf[MAX_RESPONSE];
    while (fgets(buf, sizeof(buf), f)) {
        lines = realloc(lines, sizeof(char *) * (nlines + 1));
        lines[nlines++] = strdup(buf);
    }
    fclose(f);

    int modified = 0;

    for (int i = 0; i < nlines; i++) {
        // Only process assistant lines that mention "artifact"
        if (!strstr(lines[i], "\"role\":\"assistant\"")) continue;
        if (!strstr(lines[i], "artifact")) continue;
        // Skip lines already compacted
        if (strstr(lines[i], "[artifact content")) continue;

        // Find the artifact content value. In the session JSONL, the assistant's
        // content field is a JSON string, so inner quotes are escaped as \".
        // The artifact tool call inside looks like:
        //   \"content\": \"<!DOCTYPE html>...\"
        // or:
        //   \"content\":\"<!DOCTYPE html>...\"
        // We search for \"content\" followed by : then optional space then \"
        // then look for HTML-like content start.
        char *search = lines[i];
        while ((search = strstr(search, "\\\"content\\\""))) {
            // Found \"content\" — now look for :  then \"
            char *after_key = search + 11; // past \"content\"
            if (*after_key != ':') { search = after_key; continue; }
            after_key++; // past :
            while (*after_key == ' ') after_key++; // skip optional space

            // Expect \" (the opening quote of the value)
            if (after_key[0] != '\\' || after_key[1] != '"') { search = after_key; continue; }
            char *val_start = after_key + 2; // past \"

            // Check if this looks like HTML content (the artifact payload)
            if (strncmp(val_start, "<!DOCTYPE", 9) != 0 &&
                strncmp(val_start, "<html", 5) != 0 &&
                strncmp(val_start, "<HTML", 5) != 0 &&
                strncmp(val_start, "<svg", 4) != 0) {
                search = val_start;
                continue;
            }

            // Find the end of the value: scan for \" that's a JSON string boundary.
            // The value ends at \" followed by optional space then , or } or ]
            // IMPORTANT: Inside the HTML content, JS strings are double-escaped:
            //   \\\" (4 bytes: \ \ \ ") — these are NOT boundaries.
            // Real boundaries use just \" (2 bytes: \ ") — NOT preceded by another \.
            const char *p = val_start;
            const char *val_end = NULL;
            while (*p) {
                if (p[0] == '\\' && p[1] == '"') {
                    // Check this isn't \\\" (double-escaped quote inside HTML/JS)
                    int preceded_by_backslash = (p > val_start && *(p-1) == '\\');
                    if (!preceded_by_backslash) {
                        const char *after = p + 2;
                        while (*after == ' ') after++;
                        if (*after == ',' || *after == '}' || *after == ']' ||
                            *after == '\n' || *after == 0) {
                            val_end = p;
                            break;
                        }
                    }
                }
                p++;
            }
            if (!val_end || (size_t)(val_end - val_start) < 200) {
                search = val_start;
                continue;
            }

            // Replace val_start..val_end with a placeholder
            size_t content_len = val_end - val_start;
            char placeholder[128];
            snprintf(placeholder, sizeof(placeholder),
                     "[artifact content — %zu bytes saved to disk]", content_len);

            size_t prefix_len = val_start - lines[i];
            size_t suffix_len = strlen(val_end);
            size_t new_len = prefix_len + strlen(placeholder) + suffix_len + 1;
            char *compacted = malloc(new_len);
            memcpy(compacted, lines[i], prefix_len);
            memcpy(compacted + prefix_len, placeholder, strlen(placeholder));
            memcpy(compacted + prefix_len + strlen(placeholder), val_end, suffix_len);
            compacted[prefix_len + strlen(placeholder) + suffix_len] = 0;

            free(lines[i]);
            lines[i] = compacted;
            modified = 1;

            printf(ANSI_DIM "  [compacted artifact content: %zu → %zu bytes]" ANSI_RESET "\n",
                   content_len, strlen(placeholder));
            break; // one artifact per line is typical
        }
    }

    if (modified) {
        f = fopen(path, "w");
        if (f) {
            for (int i = 0; i < nlines; i++) {
                fputs(lines[i], f);
                // Ensure each line ends with newline
                size_t ll = strlen(lines[i]);
                if (ll > 0 && lines[i][ll-1] != '\n') fputc('\n', f);
            }
            fclose(f);
        }
    }

    for (int i = 0; i < nlines; i++) free(lines[i]);
    free(lines);
}

static char *handle_tool_calls(char *response) {
    g._reserved_approve = 0;
    int loop_turns = 0;
    int truncation_retries = 0;
    int native_mode = 0; // 1 if using Ollama native tool calls

    // Check for native tool calls first (from Ollama's structured function calling).
    // These are pre-parsed and don't need text-based extraction.
    if (g_native_tool_calls && g_native_tool_calls[0] == '[') {
        native_mode = 1;
    }

    // Fallback: if no native calls and model output raw JSON tool calls without <tool_call> tags,
    // extract and wrap them so the normal parser can handle them.
    if (!native_mode && response && !strstr(response, "<tool_call>")) {
        extract_json_tool_calls(&response);
    }

    // Check if we have anything to process
    int has_tool_calls = native_mode ||
                         (response && strstr(response, "<tool_call>"));

    while (has_tool_calls) {
        if (++loop_turns > MAX_TOOL_LOOP_TURNS) {
            printf(ANSI_YELLOW "\n  [tool loop limit reached (%d turns)]" ANSI_RESET "\n", MAX_TOOL_LOOP_TURNS);
            break;
        }

        // Collect all tool calls from the current response
        #define MAX_BATCH_TOOLS 8
        ToolCall batch[MAX_BATCH_TOOLS];
        int batch_count = 0;

        if (native_mode && g_native_tool_calls) {
            // Parse native tool calls from structured JSON
            batch_count = parse_native_tool_calls(g_native_tool_calls, batch, MAX_BATCH_TOOLS);
            for (int i = 0; i < batch_count; i++) {
                printf(ANSI_DIM "  [tool: %s (native)]" ANSI_RESET "\n", batch[i].name);
            }
            // Consume the native tool calls
            free(g_native_tool_calls);
            g_native_tool_calls = NULL;
            native_mode = 0; // subsequent iterations use text-based parsing
        } else {
            // Text-based <tool_call> tag extraction (fallback path)
            char *scan = response;
            while (batch_count < MAX_BATCH_TOOLS) {
                char *tc_start = strstr(scan, "<tool_call>");
                if (!tc_start) break;
                char *tc_end = strstr(tc_start, "</tool_call>");

                char *body_start = tc_start + 11;
                int tc_len;
                if (tc_end) {
                    tc_len = (int)(tc_end - body_start);
                } else {
                    // Model omitted closing tag — use rest of response
                    tc_len = (int)strlen(body_start);
                    // Try to find end of JSON object by matching braces
                    int depth = 0;
                    int found_end = 0;
                    for (int j = 0; j < tc_len; j++) {
                        if (body_start[j] == '{') depth++;
                        else if (body_start[j] == '}') {
                            depth--;
                            if (depth == 0) {
                                tc_len = j + 1;
                                found_end = 1;
                                break;
                            }
                        } else if (body_start[j] == '"') {
                            // Skip string contents (don't count braces inside strings)
                            j++;
                            while (j < tc_len && body_start[j] != '"') {
                                if (body_start[j] == '\\') j++; // skip escaped char
                                j++;
                            }
                        }
                    }
                if (!found_end) break; // can't parse, give up
            }

            // Heap-allocate tc_body for large payloads (file_write content)
            char *tc_body = malloc(tc_len + 1);
            memcpy(tc_body, body_start, tc_len);
            tc_body[tc_len] = 0;

            if (extract_tool_call_v2(tc_body, &batch[batch_count])) {
                printf(ANSI_DIM "  [tool: %s%s]" ANSI_RESET "\n",
                       batch[batch_count].name,
                       tc_end ? "" : " (no closing tag)");
                batch_count++;
            } else {
                printf(ANSI_YELLOW "  [tool call parse failed — first 100 chars: %.100s]" ANSI_RESET "\n", tc_body);
            }
            free(tc_body);
            scan = tc_end ? tc_end + 12 : body_start + tc_len; // past </tool_call> or end of JSON
        }
        } // end else (text-based parsing)

        // Dynamic token budgets:
        //   - Truncation retry: 4x base (we know 1x wasn't enough)
        //   - Tool follow-ups:  2x base (room for another tool call, not wasteful)
        int retry_budget = g.max_tokens * 4;
        int followup_budget = g.max_tokens * 2;
        if (retry_budget > MAX_CONTEXT / 2) retry_budget = MAX_CONTEXT / 2;
        if (followup_budget > MAX_CONTEXT / 2) followup_budget = MAX_CONTEXT / 2;

        if (batch_count == 0) {
            // Tool call detected but couldn't be parsed — likely truncated by num_predict limit.
            // Drop the broken assistant response and retry with guidance to produce shorter output.
            session_replace_last_turn(g.session_id, "assistant", NULL);

            // Undo the output token count from the dropped response
            g.total_tokens_out -= g.last_token_count;
            if (g.total_tokens_out < 0) g.total_tokens_out = 0;

            int actual_tokens = g.last_token_count;

            // Cap retries: after 2 truncation retries, give up — the model isn't
            // producing compact enough output and more retries just waste time.
            truncation_retries++;
            if (truncation_retries > 2) {
                printf(ANSI_YELLOW "\n  [tool call too large after %d retries (%d tokens) — "
                       "try asking for a simpler version]" ANSI_RESET "\n", truncation_retries, actual_tokens);

                // Synthesize a tool result so the session stays consistent.
                // Without this, the session has an assistant turn with a tool call
                // but no corresponding tool response, confusing the model on future turns.
                const char *synthetic = "<tool_response name=\"unknown\">\n"
                    "Error: tool call was too large and could not be completed after multiple "
                    "retries. Please try a simpler approach with less code.\n"
                    "</tool_response>";
                session_save_turn(g.session_id, "tool", synthetic);

                free(response);
                return strdup("");
            }

            // Escalate retry budget based on actual usage
            if (actual_tokens * 2 > retry_budget) {
                retry_budget = actual_tokens * 2;
                if (retry_budget > MAX_CONTEXT / 2) retry_budget = MAX_CONTEXT / 2;
            }

            printf(ANSI_YELLOW "\n  [tool call truncated at %d tokens — retrying with %d]"
                   ANSI_RESET "\n", actual_tokens, retry_budget);

            // Inject truncation feedback so the model knows to produce shorter output.
            // Without this, the model has no signal that its approach was too verbose
            // and will regenerate the same bloated response.
            char feedback[512];
            snprintf(feedback, sizeof(feedback),
                "Your previous response was truncated at %d tokens — the artifact was too long. "
                "Please try again with shorter code (~3000 tokens). Tips: skip comments, use short "
                "variable names, implement only the core game loop. Go directly to the tool call "
                "without drafting code in your response text.",
                actual_tokens);
            session_save_turn(g.session_id, "user", feedback);

            free(response);
            int sock = send_request(NULL, retry_budget, g.session_id);
            if (sock < 0) return NULL;

            printf("\n");
            response = stream_response(sock, retry_budget);

            // Drop the injected feedback from session (it was only for the retry)
            // and save the new assistant response in its place
            session_replace_last_turn(g.session_id, "user", NULL); // remove feedback
            if (response && strlen(response) > 0) {
                if (g_native_tool_calls) {
                    session_save_assistant_with_tool_calls(g.session_id, response, g_native_tool_calls);
                } else {
                    session_save_turn(g.session_id, "assistant", response);
                }
            }

            continue;
        }

        // Execute all collected tool calls and combine responses
        size_t combined_cap = 65536 * batch_count + 512;
        char *combined = malloc(combined_cap);
        combined[0] = 0;
        size_t combined_len = 0;
        int denied = 0;
        int had_artifact = 0;

        for (int i = 0; i < batch_count && !denied; i++) {
            char output[65536];
            int out_len = execute_tool(&batch[i], output, sizeof(output));
            if (out_len < 0) { denied = 1; break; }
            if (strcmp(batch[i].name, "artifact") == 0) had_artifact = 1;

            int wrote = snprintf(combined + combined_len, combined_cap - combined_len,
                "<tool_response name=\"%s\">\n%s</tool_response>\n",
                batch[i].name, output);
            if (wrote > 0) combined_len += wrote;
        }

        // Free all ToolCall heap memory
        for (int i = 0; i < batch_count; i++) tool_call_free(&batch[i]);

        // Argus reaction (fire after tools, before follow-up)
        if (g_argus_enabled && batch_count > 0 && !denied) {
            // Use the last tool name and combined output for context
            char last_tool[64] = "tool";
            // Extract last tool name from combined output
            char *last_resp = strstr(combined, "<tool_response name=\"");
            char *found = last_resp;
            while (found) {
                last_resp = found;
                found = strstr(found + 1, "<tool_response name=\"");
            }
            if (last_resp) {
                const char *name_start = last_resp + strlen("<tool_response name=\"");
                int ni = 0;
                while (name_start[ni] && name_start[ni] != '"' && ni < 63) {
                    last_tool[ni] = name_start[ni];
                    ni++;
                }
                last_tool[ni] = 0;
            }
            argus_react(last_tool, combined);
        }

        // After successful artifact creation, compact the assistant's session turn
        // to strip the large HTML content. The full content is saved to disk by
        // execute_artifact; keeping it in the session bloats every future prefill.
        if (had_artifact && !denied) {
            // Compact artifact content from the session file. Scans ALL assistant
            // lines (not just the last) and strips HTML content from artifact tool calls,
            // replacing it with a short placeholder. This prevents context bloat on
            // subsequent turns — the full content is already saved to disk by execute_artifact.
            //
            // In the session JSONL, text-based <tool_call> content looks like:
            //   {"role":"assistant","content":"...<tool_call>\n{\"name\": \"artifact\", \"arguments\": {\"title\": \"...\", \"content\": \"<!DOCTYPE html>...\", ...
            // The artifact content is JSON-escaped inside the content string, so the
            // boundary markers are \" (backslash-quote on disk = two bytes: '\' '"').
            compact_artifact_content(g.session_id);
        }

        if (denied) {
            free(combined);
            free(response);
            return NULL;
        }

        session_save_turn(g.session_id, "tool", combined);

        // For artifact-only batches, skip the follow-up request entirely —
        // UNLESS the artifact was truncated/errored. The model already described
        // what it built in the prose before the tool call, and the user can see
        // the pop-out window. Skipping avoids a costly prefill (often 2-5 minutes)
        // just for a redundant "Here's your game!" summary.
        // But if the artifact was truncated, the model MUST see the error so it
        // can retry with a shorter version or use append_to for multi-part output.
        if (had_artifact && batch_count == 1 && !strstr(combined, "ERROR:")) {
            free(combined);
            free(response);
            return strdup("");  // empty response = no more tool calls, loop exits
        }

        free(response);

        // Auto-compact before follow-up to prevent context bloat
        maybe_compact();

        // Follow-up requests in tool loop get 2x budget (room for another tool call).
        // Pass NULL as user_message — the tool result is already saved in the session
        // JSONL (line above), so send_request will replay it from there. Passing
        // `combined` here would duplicate the tool result (once as role:tool from
        // JSONL, once as role:user appended by send_request), which doubles the
        // token count and invalidates Ollama's KV cache prefix match, causing a
        // full reprocess (observed: 300s stall after image_generate).
        int sock = send_request(NULL, followup_budget, g.session_id);
        free(combined);
        if (sock < 0) return NULL;

        printf("\n");
        response = stream_response(sock, followup_budget);

        if (response && (strlen(response) > 0 || g_native_tool_calls)) {
            if (g_native_tool_calls) {
                session_save_assistant_with_tool_calls(g.session_id, response, g_native_tool_calls);
            } else {
                session_save_turn(g.session_id, "assistant", response);
            }
        }

        // Check if the follow-up response has tool calls (native or text-based)
        if (g_native_tool_calls && g_native_tool_calls[0] == '[') {
            native_mode = 1;
            has_tool_calls = 1;
        } else if (response && strstr(response, "<tool_call>")) {
            has_tool_calls = 1;
        } else {
            // Try JSON extraction fallback
            if (response && !strstr(response, "<tool_call>")) {
                extract_json_tool_calls(&response);
            }
            has_tool_calls = (response && strstr(response, "<tool_call>"));
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
static void cmd_connections(const char *args);
static void cmd_name(const char *args);
static void cmd_artifacts(const char *args);
static void cmd_cron(const char *args);
static void cmd_pdf(const char *args);
static void cmd_tutorial(const char *args);
static void cmd_argus(const char *args);

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
    {"/connections", "[service]", "Manage API connections",        cmd_connections},
    {"/name",    "<name>",   "Rename this agent",                  cmd_name},
    {"/artifacts","[open|dir]","List session artifacts or open one", cmd_artifacts},
    {"/pdf",     "[title|N]", "Export artifact to PDF",              cmd_pdf},
    {"/cron",    "[add|rm|ls]","Manage recurring scheduled tasks",   cmd_cron},
    {"/tutorial","[topic]",   "Interactive tutorial with examples",  cmd_tutorial},
    {"/argus",  NULL,       "Toggle Argus companion on/off",       cmd_argus},
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
    // Count active tools
    int tool_count = 29;
    if (!g_connections_loaded) load_connections();
    Connection *brave = get_connection("brave_search");
    Connection *gh = get_connection("github");
    Connection *goog = get_connection("google");
    Connection *wolf = get_connection("wolfram");
    if (brave && brave->active) tool_count++;
    if (gh && gh->active) tool_count++;
    if (goog && goog->active) tool_count += 3; // gmail + gdrive + gdocs
    if (wolf && wolf->active) tool_count++;

    printf("\n" ANSI_BOLD "  Agent Tools (%d)" ANSI_RESET "\n", tool_count);
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
    printf("    memory_delete, screenshot, window_focus");
    if (brave && brave->active) printf(", web_search");
    if (gh && gh->active) printf(", github");
    if (goog && goog->active) printf(", gmail, gdrive, gdocs");
    if (wolf && wolf->active) printf(", wolfram");
    printf("\n\n");

    printf("  " ANSI_RED "Confirm always" ANSI_RESET " (potentially destructive):\n");
    printf("    bash, process_kill, open_app, applescript\n\n");

    // Show connected services
    int any = (brave && brave->active) || (gh && gh->active) || (goog && goog->active) || (wolf && wolf->active);
    if (any) {
        printf("  " ANSI_BOLD "Connected services:" ANSI_RESET "\n");
        if (brave && brave->active) printf("    " ANSI_GREEN "●" ANSI_RESET " web_search  — Brave Search API\n");
        if (gh && gh->active)       printf("    " ANSI_GREEN "●" ANSI_RESET " github      — GitHub API (repos, issues, PRs)\n");
        if (goog && goog->active) {
            printf("    " ANSI_GREEN "●" ANSI_RESET " gmail       — Gmail (search, read, send, draft, trash)\n");
            printf("    " ANSI_GREEN "●" ANSI_RESET " gdrive      — Google Drive (list, search, upload, download, share)\n");
            printf("    " ANSI_GREEN "●" ANSI_RESET " gdocs       — Google Docs (create, read, append)\n");
        }
        if (wolf && wolf->active)   printf("    " ANSI_GREEN "●" ANSI_RESET " wolfram     — Wolfram Alpha (math, science, data)\n");
        printf("\n");
    } else {
        printf("  " ANSI_DIM "No external services connected." ANSI_RESET "\n");
        printf("  " ANSI_DIM "Run /connections setup to enable web_search, github, google, wolfram." ANSI_RESET "\n\n");
    }

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
               "  " ANSI_CYAN "/help tips" ANSI_RESET
               "  " ANSI_CYAN "/tutorial" ANSI_RESET "\n\n");
        return;
    }
    while (*args == ' ') args++;

    if (strcmp(args, "tools") == 0) { help_tools(); return; }
    if (strcmp(args, "connections") == 0 || strcmp(args, "connect") == 0) {
        cmd_connections(""); return;
    }
    if (strcmp(args, "memory") == 0 || strcmp(args, "memories") == 0) { help_memory(); return; }
    if (strcmp(args, "channels") == 0 || strcmp(args, "channel") == 0) { help_channels(); return; }
    if (strcmp(args, "projects") == 0 || strcmp(args, "project") == 0) { help_projects(); return; }
    if (strcmp(args, "tips") == 0 || strcmp(args, "best") == 0 || strcmp(args, "practices") == 0) { help_tips(); return; }
    if (strcmp(args, "tutorial") == 0 || strcmp(args, "examples") == 0) { cmd_tutorial(""); return; }
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
    free(g_pending_image); g_pending_image = NULL;
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
    char *response = stream_response(sock, g.max_tokens);
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

static void cmd_name(const char *args) {
    if (!args || !args[0] || *args == ' ') {
        // Trim leading spaces
        while (args && *args == ' ') args++;
        if (!args || !*args) {
            printf("  Agent name: " ANSI_BOLD "%s" ANSI_RESET "\n", g_agent_name);
            printf(ANSI_DIM "  Usage: /name <new name>" ANSI_RESET "\n\n");
            return;
        }
    }
    while (*args == ' ') args++;
    char name[128];
    strlcpy(name, args, sizeof(name));
    // Trim trailing whitespace
    int len = (int)strlen(name);
    while (len > 0 && (name[len-1] == ' ' || name[len-1] == '\n')) name[--len] = 0;
    if (len == 0) { printf(ANSI_YELLOW "  Usage: /name <new name>" ANSI_RESET "\n\n"); return; }
    save_identity(name);
    save_memory("agent_identity", "user", "The agent's chosen name and identity",
        g_agent_name, "global");
    printf(ANSI_GREEN "  Agent renamed to: %s" ANSI_RESET "\n\n", g_agent_name);
}

static void cmd_artifacts(const char *args) {
    while (args && *args == ' ') args++;

    // /artifacts dir — open artifacts folder in Finder
    if (args && strncmp(args, "dir", 3) == 0) {
        const char *home = getenv("HOME");
        char dir[PATH_MAX];
        snprintf(dir, sizeof(dir), "%s/.pre/artifacts", home);
        char cmd[PATH_MAX + 32];
        snprintf(cmd, sizeof(cmd), "open '%s'", dir);
        system(cmd);
        printf("  Opened artifacts directory in Finder.\n\n");
        return;
    }

    // /artifacts open <n> — re-open artifact by number
    if (args && strncmp(args, "open", 4) == 0) {
        const char *num = args + 4;
        while (*num == ' ') num++;
        int idx = atoi(num) - 1;
        if (idx < 0 || idx >= g_artifact_count) {
            printf(ANSI_YELLOW "  Invalid artifact number. Use /artifacts to list." ANSI_RESET "\n\n");
            return;
        }
        show_artifact_window(g_artifacts[idx].path, g_artifacts[idx].title);
        printf("  Reopened: %s\n\n", g_artifacts[idx].title);
        return;
    }

    // Default: list session artifacts
    if (g_artifact_count == 0) {
        printf("  No artifacts created this session.\n");
        printf(ANSI_DIM "  The model can create artifacts using the artifact tool." ANSI_RESET "\n");
        printf(ANSI_DIM "  Use /artifacts dir to browse all saved artifacts." ANSI_RESET "\n\n");
        return;
    }

    printf("\n" ANSI_BOLD "  Session Artifacts" ANSI_RESET "\n");
    printf("  ─────────────────────────────\n");
    for (int i = 0; i < g_artifact_count; i++) {
        printf("  %d. " ANSI_CYAN "%s" ANSI_RESET " [%s]\n",
               i + 1, g_artifacts[i].title, g_artifacts[i].type);
        printf("     \033]8;;file://%s\033\\", g_artifacts[i].path);
        printf(ANSI_DIM "▸ %s" ANSI_RESET, g_artifacts[i].path);
        printf("\033]8;;\033\\\n");
    }
    printf("\n  " ANSI_DIM "/artifacts open <n>  — reopen artifact" ANSI_RESET "\n");
    printf("  " ANSI_DIM "/artifacts dir       — open folder in Finder" ANSI_RESET "\n\n");
}

// /cron — manage recurring scheduled tasks
// Usage: /cron add <schedule> <prompt>   — add a new cron job
//        /cron ls                        — list all cron jobs
//        /cron rm <id>                   — remove a cron job
//        /cron enable <id>               — enable a disabled job
//        /cron disable <id>              — disable a job
static void cmd_pdf(const char *args) {
    if (g_artifact_count == 0) {
        printf(ANSI_YELLOW "  No artifacts in this session. Create one first." ANSI_RESET "\n");
        return;
    }

    int idx = -1;
    if (!args || !args[0]) {
        // Default: most recent artifact
        idx = g_artifact_count - 1;
    } else if (args[0] >= '0' && args[0] <= '9') {
        idx = atoi(args) - 1;
        if (idx < 0 || idx >= g_artifact_count) {
            printf(ANSI_YELLOW "  Invalid index. Use 1-%d." ANSI_RESET "\n", g_artifact_count);
            return;
        }
    } else {
        // Search by title
        for (int i = g_artifact_count - 1; i >= 0; i--) {
            if (strcasestr(g_artifacts[i].title, args)) { idx = i; break; }
        }
        if (idx < 0) {
            printf(ANSI_YELLOW "  No artifact matching '%s'." ANSI_RESET "\n", args);
            return;
        }
    }

    // Build PDF path
    char pdf_path[PATH_MAX];
    strlcpy(pdf_path, g_artifacts[idx].path, sizeof(pdf_path));
    char *dot = strrchr(pdf_path, '.');
    if (dot) strlcpy(dot, ".pdf", sizeof(pdf_path) - (dot - pdf_path));
    else strlcat(pdf_path, ".pdf", sizeof(pdf_path));

    printf(ANSI_DIM "  Exporting \"%s\" to PDF..." ANSI_RESET "\n", g_artifacts[idx].title);

    int rc = export_to_pdf(g_artifacts[idx].path, pdf_path);
    if (rc == 0) {
        struct stat pst;
        long long psz = 0;
        if (stat(pdf_path, &pst) == 0) psz = (long long)pst.st_size;

        printf(ANSI_CYAN "  ◆ PDF: %s" ANSI_RESET " (%lld bytes)\n", pdf_path, psz);
        // Open in Preview
        pid_t opid = fork();
        if (opid == 0) { execl("/usr/bin/open", "open", pdf_path, NULL); _exit(1); }
    } else {
        printf(ANSI_RED "  PDF export failed. Requires macOS 13+." ANSI_RESET "\n");
    }
}

// ============================================================================
// Tutorial
// ============================================================================

static void tutorial_section(const char *icon, const char *title, const char **prompts, int count, const char *tip) {
    printf("\n  %s " ANSI_BOLD "%s" ANSI_RESET "\n", icon, title);
    printf("  " ANSI_DIM "──────────────────────────────────────" ANSI_RESET "\n");
    for (int i = 0; i < count; i++) {
        printf("  " ANSI_CYAN "→" ANSI_RESET " %s\n", prompts[i]);
    }
    if (tip) printf("  " ANSI_DIM "💡 %s" ANSI_RESET "\n", tip);
}

static void cmd_tutorial(const char *args) {
    while (args && *args == ' ') args++;

    printf("\n");
    printf(ANSI_BOLD "  ╔══════════════════════════════════════════════════════╗\n");
    printf("  ║                                                      ║\n");
    printf("  ║   ██████╗ ██████╗ ███████╗  Tutorial                 ║\n");
    printf("  ║   ██╔══██╗██╔══██╗██╔════╝  ─────────               ║\n");
    printf("  ║   ██████╔╝██████╔╝█████╗    Copy any prompt below   ║\n");
    printf("  ║   ██╔═══╝ ██╔══██╗██╔══╝    and paste it to try     ║\n");
    printf("  ║   ██║     ██║  ██║███████╗   it out.                 ║\n");
    printf("  ║   ╚═╝     ╚═╝  ╚═╝╚══════╝                          ║\n");
    printf("  ║                                                      ║\n");
    printf("  ╚══════════════════════════════════════════════════════╝" ANSI_RESET "\n");

    // Topic filter
    int show_all = (!args || !args[0] || strcmp(args, "all") == 0);
    int show_topic = 0;

    #define MATCH_TOPIC(t) (show_all || (args && (strcasecmp(args, t) == 0)))

    if (MATCH_TOPIC("basics") || MATCH_TOPIC("start") || show_all) {
        show_topic = 1;
        const char *p[] = {
            "What can you help me with? Give me a quick overview.",
            "What's running on my Mac right now? Show me top 10 processes by CPU.",
            "Summarize what's in my Downloads folder.",
            "Search the web for the latest news about Apple Silicon.",
        };
        tutorial_section("💬", "Getting Started", p, 4,
            "PRE auto-titles sessions. Use /new for fresh conversations.");
    }

    if (MATCH_TOPIC("files") || MATCH_TOPIC("file") || show_all) {
        show_topic = 1;
        const char *p[] = {
            "Find all Python files in my home directory that import pandas.",
            "Read my ~/.zshrc and suggest improvements.",
            "Search my Documents folder for any file mentioning \"quarterly review\".",
            "Find every TODO comment in this project and create a summary.",
        };
        tutorial_section("📁", "File Operations", p, 4,
            "PRE can read, write, search, and edit any file on your system.");
    }

    if (MATCH_TOPIC("macos") || MATCH_TOPIC("mac") || MATCH_TOPIC("native") || show_all) {
        show_topic = 1;
        const char *p[] = {
            "What's on my calendar today? Include meeting links.",
            "Check my email for anything from my boss in the last 3 days.",
            "Remind me to submit the expense report by Friday at 5pm.",
            "Search my notes for anything about the API migration.",
            "Find all PDFs on my Mac that contain \"budget proposal\".",
        };
        tutorial_section("🍎", "Native macOS (Mail, Calendar, Contacts, etc.)", p, 5,
            "Works with any provider configured on your Mac — no API keys needed.");
    }

    if (MATCH_TOPIC("desktop") || MATCH_TOPIC("computer") || MATCH_TOPIC("automation") || show_all) {
        show_topic = 1;
        const char *p[] = {
            "Take a screenshot and describe what's on my screen.",
            "Open System Settings and navigate to the Wi-Fi section.",
            "Open TextEdit and type \"Meeting notes for today\" as the title.",
            "Press Cmd+Space, type \"Activity Monitor\", and press Enter.",
        };
        tutorial_section("🖥️ ", "Desktop Automation (Computer Use)", p, 4,
            "PRE sees your screen and operates any app via mouse/keyboard.");
    }

    if (MATCH_TOPIC("memory") || MATCH_TOPIC("rag") || show_all) {
        show_topic = 1;
        const char *p[] = {
            "Remember that our standup is at 9:15 AM Pacific every weekday.",
            "Search my memories for anything about deployment procedures.",
            "Index my ~/Documents/notes folder and call the index \"my-notes\".",
            "Search the \"my-notes\" index for anything about project deadlines.",
            "What do you remember about my work projects?",
        };
        tutorial_section("🧠", "Memory & RAG", p, 5,
            "Memories persist across sessions. /memory to browse. RAG searches by meaning.");
    }

    if (MATCH_TOPIC("schedule") || MATCH_TOPIC("cron") || MATCH_TOPIC("triggers") || show_all) {
        show_topic = 1;
        const char *p[] = {
            "Schedule a daily morning briefing at 8am — calendar, emails, Jira tickets. Monday-Friday.",
            "Create a trigger watching ~/Downloads for new PDFs — summarize each one.",
            "Schedule a job every 6 hours to check disk usage. Alert if any volume over 80%%.",
            "List all my scheduled jobs and their next run times.",
        };
        tutorial_section("⏰", "Scheduling & Triggers", p, 4,
            "Cron jobs run in the background. /cron to manage. Triggers react to file changes.");
    }

    if (MATCH_TOPIC("agents") || MATCH_TOPIC("agent") || MATCH_TOPIC("research") || show_all) {
        show_topic = 1;
        const char *p[] = {
            "Research PostgreSQL vs MySQL for high-write workloads. Spawn agents for each, then compare.",
            "Spawn an agent to read all README files in ~/projects and summarize each one.",
            "Do a deep research pass on best practices for securing a Node.js REST API in 2026.",
        };
        tutorial_section("🤖", "Sub-Agents & Deep Research", p, 3,
            "Agents work autonomously with their own tools and sessions.");
    }

    if (MATCH_TOPIC("cloud") || MATCH_TOPIC("integrations") || show_all) {
        show_topic = 1;
        const char *p[] = {
            "Show me all Jira tickets assigned to me in \"In Progress\" status.",
            "Search Slack for messages about the production deployment.",
            "List my open pull requests on GitHub.",
            "Create a Linear issue: \"Add rate limiting to /api/search\".",
            "What Zoom meetings do I have scheduled this week?",
        };
        tutorial_section("☁️ ", "Cloud Integrations (15 services)", p, 5,
            "Configure in /connections. Supports Jira, Slack, GitHub, Linear, Zoom, and more.");
    }

    if (MATCH_TOPIC("artifacts") || MATCH_TOPIC("export") || MATCH_TOPIC("create") || show_all) {
        show_topic = 1;
        const char *p[] = {
            "Create an interactive HTML dashboard with a project timeline and progress bars.",
            "Build a Pomodoro timer as an HTML artifact with start/pause/reset buttons.",
            "Create a Word document summarizing today's meeting notes with action items.",
            "Export the current conversation as a PDF.",
        };
        tutorial_section("🎨", "Artifacts & Exports", p, 4,
            "/artifacts to list, /pdf to export. HTML artifacts open in your browser.");
    }

    if (MATCH_TOPIC("power") || MATCH_TOPIC("workflows") || MATCH_TOPIC("advanced") || show_all) {
        show_topic = 1;
        const char *p[] = {
            "Check my calendar, summarize important emails, list top Jira tickets — morning briefing.",
            "Index this repo with RAG, find the auth flow and DB schema, give me a developer onboarding summary.",
            "Check disk usage, top 20 processes by memory, network connectivity — system health report.",
            "Summarize what I worked on today, create a reminder for tomorrow, draft a standup update.",
        };
        tutorial_section("🔥", "Power Workflows", p, 4,
            "Combine multiple features for real-world multi-step workflows.");
    }

    if (!show_topic) {
        printf(ANSI_YELLOW "  Unknown topic: %s" ANSI_RESET "\n\n", args);
        printf("  Available topics: " ANSI_CYAN "basics files macos desktop memory schedule agents cloud artifacts power" ANSI_RESET "\n");
        printf("  Or just " ANSI_CYAN "/tutorial" ANSI_RESET " for everything.\n\n");
        return;
    }

    printf("\n  " ANSI_DIM "Topics: /tutorial [basics|files|macos|desktop|memory|schedule|agents|cloud|artifacts|power]" ANSI_RESET "\n\n");

    #undef MATCH_TOPIC
}

// ============================================================================
// Argus companion
// ============================================================================

static const char *argus_config_path(void) {
    static char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/.pre/argus.json", getenv("HOME") ?: "/tmp");
    return path;
}

static void argus_load(void) {
    FILE *f = fopen(argus_config_path(), "r");
    if (!f) return;
    char buf[512];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);
    // Check "enabled" field
    const char *p = strstr(buf, "\"enabled\"");
    if (p) {
        p = strchr(p, ':');
        if (p) {
            p++;
            while (*p == ' ') p++;
            g_argus_enabled = (strncmp(p, "true", 4) == 0) ? 1 : 0;
        }
    }
}

static void argus_save(void) {
    FILE *f = fopen(argus_config_path(), "r");
    char buf[1024] = {0};
    if (f) {
        fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
    }
    // Update or write config with enabled state
    f = fopen(argus_config_path(), "w");
    if (!f) return;
    // Simple: rewrite the whole config preserving other fields
    if (strstr(buf, "\"enabled\"")) {
        // Replace the enabled value
        char *p = strstr(buf, "\"enabled\"");
        char *colon = strchr(p, ':');
        if (colon) {
            // Find the value (true or false)
            char *val = colon + 1;
            while (*val == ' ') val++;
            char *end = val;
            while (*end && *end != ',' && *end != '}' && *end != '\n') end++;
            // Write: prefix + new value + suffix
            fwrite(buf, 1, val - buf, f);
            fprintf(f, "%s", g_argus_enabled ? "true" : "false");
            fputs(end, f);
        }
    } else {
        // No existing config — write fresh
        fprintf(f, "{\n  \"enabled\": %s,\n  \"name\": \"Argus\",\n  \"personality\": \"thoughtful mentor\",\n  \"cooldownMs\": 30000,\n  \"maxReactionTokens\": 150\n}\n",
                g_argus_enabled ? "true" : "false");
    }
    fclose(f);
}

// Print an Argus reaction after tool execution. Makes a quick Ollama API
// call to generate a reaction. Fire-and-forget.
static void argus_react(const char *tool_name, const char *tool_output) {
    if (!g_argus_enabled) return;

    // Build a simple request body for Argus
    static time_t last_reaction = 0;
    time_t now = time(NULL);
    if (now - last_reaction < 30) return;  // 30s cooldown
    last_reaction = now;

    // Truncate output for context
    char output_preview[512];
    size_t olen = strlen(tool_output);
    if (olen > 500) {
        memcpy(output_preview, tool_output, 500);
        output_preview[500] = 0;
    } else {
        strlcpy(output_preview, tool_output, sizeof(output_preview));
    }

    // Build minimal Ollama request directly (no web server dependency)
    char body[4096];
    // JSON-escape the output preview and tool name
    char escaped_output[1200];
    char escaped_tool[256];
    int ei = 0;
    for (int i = 0; output_preview[i] && ei < (int)sizeof(escaped_output) - 2; i++) {
        if (output_preview[i] == '"' || output_preview[i] == '\\') escaped_output[ei++] = '\\';
        if (output_preview[i] == '\n') { escaped_output[ei++] = '\\'; escaped_output[ei++] = 'n'; continue; }
        if (output_preview[i] == '\r') continue;
        if (output_preview[i] == '\t') { escaped_output[ei++] = ' '; continue; }
        escaped_output[ei++] = output_preview[i];
    }
    escaped_output[ei] = 0;
    ei = 0;
    for (int i = 0; tool_name[i] && ei < (int)sizeof(escaped_tool) - 2; i++) {
        if (tool_name[i] == '"' || tool_name[i] == '\\') escaped_tool[ei++] = '\\';
        escaped_tool[ei++] = tool_name[i];
    }
    escaped_tool[ei] = 0;

    snprintf(body, sizeof(body),
        "{\"model\":\"%s\",\"stream\":false,\"options\":{\"num_predict\":150,\"num_ctx\":%d},"
        "\"messages\":["
        "{\"role\":\"system\",\"content\":\"You are Argus, a brief companion observer. You watch an AI assistant work and offer 1-2 short sentences. Be genuinely helpful: suggestions, encouragement, or gentle observations. Never repeat what the tool said. Be specific, not generic.\"},"
        "{\"role\":\"user\",\"content\":\"Tool '%s' just ran.\\nResult: %s\\n\\nReact briefly.\"}"
        "]}",
        g_model, MODEL_CTX, escaped_tool, escaped_output);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return;

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(g.port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    // Non-blocking connect with 5s timeout
    struct timeval tv = {5, 0};
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(sock);
        return;
    }

    char header[256];
    int body_len = (int)strlen(body);
    int hlen = snprintf(header, sizeof(header),
        "POST /api/chat HTTP/1.1\r\n"
        "Host: 127.0.0.1:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n\r\n",
        g.port, body_len);

    send(sock, header, hlen, 0);
    send(sock, body, body_len, 0);

    // Read response
    char resp[8192];
    int total = 0;
    while (total < (int)sizeof(resp) - 1) {
        int n = (int)recv(sock, resp + total, sizeof(resp) - 1 - total, 0);
        if (n <= 0) break;
        total += n;
    }
    resp[total] = 0;
    close(sock);

    // Extract the response content from Ollama JSON
    // Look for "content":"..." in the response body
    char *json_start = strstr(resp, "\r\n\r\n");
    if (!json_start) return;
    json_start += 4;

    char reaction[1024];
    if (json_extract_str(json_start, "content", reaction, sizeof(reaction)) > 0 && strlen(reaction) >= 5) {
        printf("\n  " ANSI_MAGENTA "✦ " ANSI_RESET ANSI_CYAN "%s" ANSI_RESET "\n", reaction);
    }
}

static void cmd_argus(const char *args) {
    (void)args;
    g_argus_enabled = !g_argus_enabled;
    argus_save();
    printf("\n  " ANSI_MAGENTA "✦ Argus" ANSI_RESET " %s\n",
           g_argus_enabled ? "enabled — reactions will appear between tool outputs"
                          : "disabled");
    printf("\n");
}

static void cmd_cron(const char *args) {
    while (args && *args == ' ') args++;

    // Load jobs if not already loaded
    if (g_cron_count == 0 && g_cron_last_check_ms == 0) {
        cron_load();
        g_cron_last_check_ms = now_ms();
    }

    // /cron (no args) or /cron ls — list jobs
    if (!args || !args[0] || strncmp(args, "ls", 2) == 0 || strncmp(args, "list", 4) == 0) {
        if (g_cron_count == 0) {
            printf("\n  No cron jobs configured.\n");
            printf(ANSI_DIM "  Use /cron add <schedule> <prompt> to create one.\n");
            printf("  Schedule is standard 5-field cron: min hour dom month dow\n");
            printf("  Example: /cron add */30 * * * * check disk usage and alert if > 90%%" ANSI_RESET "\n\n");
            return;
        }

        printf("\n" ANSI_BOLD "  Cron Jobs" ANSI_RESET "\n");
        printf("  ─────────────────────────────────────\n");
        for (int i = 0; i < g_cron_count; i++) {
            CronJob *j = &g_cron_jobs[i];
            printf("  %s%s" ANSI_RESET " " ANSI_DIM "[%s]" ANSI_RESET " %s\n",
                   j->enabled ? ANSI_GREEN : ANSI_RED,
                   j->id,
                   j->schedule,
                   j->description[0] ? j->description : j->prompt);
            if (!j->enabled)
                printf("     " ANSI_DIM "(disabled)" ANSI_RESET "\n");
            if (j->run_count > 0) {
                char timebuf[64];
                struct tm *tm = localtime(&j->last_run_at);
                strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M", tm);
                printf("     " ANSI_DIM "runs: %d, last: %s" ANSI_RESET "\n", j->run_count, timebuf);
            }
        }
        printf("\n");
        return;
    }

    // /cron add <schedule> <prompt>
    // The schedule is 5 space-separated fields, then the rest is the prompt.
    // To disambiguate, we parse exactly 5 cron fields then treat the rest as prompt.
    if (strncmp(args, "add ", 4) == 0) {
        const char *p = args + 4;
        while (*p == ' ') p++;

        // Parse 5 cron fields
        char schedule[64] = "";
        int slen = 0;
        int field_count = 0;
        const char *field_start = p;

        while (field_count < 5 && *p) {
            while (*p && *p != ' ') p++;
            if (field_count > 0) schedule[slen++] = ' ';
            int flen = (int)(p - field_start);
            if (slen + flen >= (int)sizeof(schedule) - 1) break;
            memcpy(schedule + slen, field_start, flen);
            slen += flen;
            field_count++;
            while (*p == ' ') p++;
            field_start = p;
        }
        schedule[slen] = 0;

        if (field_count < 5 || !*p) {
            printf(ANSI_YELLOW "  Usage: /cron add <min> <hour> <dom> <month> <dow> <prompt>" ANSI_RESET "\n");
            printf(ANSI_DIM "  Example: /cron add 0 9 * * 1-5 summarize yesterday's git activity" ANSI_RESET "\n\n");
            return;
        }

        if (g_cron_count >= MAX_CRON_JOBS) {
            printf(ANSI_YELLOW "  Maximum cron jobs (%d) reached." ANSI_RESET "\n\n", MAX_CRON_JOBS);
            return;
        }

        CronJob *j = &g_cron_jobs[g_cron_count];
        memset(j, 0, sizeof(CronJob));
        cron_generate_id(j->id, sizeof(j->id));
        strlcpy(j->schedule, schedule, sizeof(j->schedule));
        strlcpy(j->prompt, p, sizeof(j->prompt));
        j->enabled = 1;
        j->created_at = time(NULL);

        // Auto-generate description: first 60 chars of prompt
        size_t dlen = strlen(p);
        if (dlen > 60) {
            memcpy(j->description, p, 57);
            strcpy(j->description + 57, "...");
        } else {
            strlcpy(j->description, p, sizeof(j->description));
        }

        g_cron_count++;
        cron_save();

        printf(ANSI_GREEN "  Created cron job %s" ANSI_RESET "\n", j->id);
        printf(ANSI_DIM "  schedule: %s" ANSI_RESET "\n", j->schedule);
        printf(ANSI_DIM "  prompt:   %s" ANSI_RESET "\n\n", j->prompt);
        return;
    }

    // /cron rm <id>
    if (strncmp(args, "rm ", 3) == 0 || strncmp(args, "remove ", 7) == 0 ||
        strncmp(args, "del ", 4) == 0 || strncmp(args, "delete ", 7) == 0) {
        const char *id = strchr(args, ' ');
        while (id && *id == ' ') id++;
        if (!id || !*id) {
            printf(ANSI_YELLOW "  Usage: /cron rm <id>" ANSI_RESET "\n\n");
            return;
        }

        for (int i = 0; i < g_cron_count; i++) {
            if (strcmp(g_cron_jobs[i].id, id) == 0) {
                printf("  Removed cron job %s: %s\n\n", g_cron_jobs[i].id, g_cron_jobs[i].description);
                // Shift remaining jobs down
                for (int k = i; k < g_cron_count - 1; k++)
                    g_cron_jobs[k] = g_cron_jobs[k + 1];
                g_cron_count--;
                cron_save();
                return;
            }
        }
        printf(ANSI_YELLOW "  No cron job with id '%s'" ANSI_RESET "\n\n", id);
        return;
    }

    // /cron enable <id>
    if (strncmp(args, "enable ", 7) == 0) {
        const char *id = args + 7;
        while (*id == ' ') id++;
        for (int i = 0; i < g_cron_count; i++) {
            if (strcmp(g_cron_jobs[i].id, id) == 0) {
                g_cron_jobs[i].enabled = 1;
                cron_save();
                printf(ANSI_GREEN "  Enabled cron job %s" ANSI_RESET "\n\n", id);
                return;
            }
        }
        printf(ANSI_YELLOW "  No cron job with id '%s'" ANSI_RESET "\n\n", id);
        return;
    }

    // /cron disable <id>
    if (strncmp(args, "disable ", 8) == 0) {
        const char *id = args + 8;
        while (*id == ' ') id++;
        for (int i = 0; i < g_cron_count; i++) {
            if (strcmp(g_cron_jobs[i].id, id) == 0) {
                g_cron_jobs[i].enabled = 0;
                cron_save();
                printf(ANSI_YELLOW "  Disabled cron job %s" ANSI_RESET "\n\n", id);
                return;
            }
        }
        printf(ANSI_YELLOW "  No cron job with id '%s'" ANSI_RESET "\n\n", id);
        return;
    }

    // /cron run <id> — manually trigger a job now
    if (strncmp(args, "run ", 4) == 0) {
        const char *id = args + 4;
        while (*id == ' ') id++;
        for (int i = 0; i < g_cron_count; i++) {
            if (strcmp(g_cron_jobs[i].id, id) == 0) {
                CronJob *j = &g_cron_jobs[i];
                printf(ANSI_CYAN "  Running cron job %s: %s" ANSI_RESET "\n", j->id, j->description);

                session_save_turn(g.session_id, "user", j->prompt);
                free(g_native_tool_calls);
                g_native_tool_calls = NULL;

                int sock = send_request(j->prompt, g.max_tokens, g.session_id);
                if (sock >= 0) {
                    printf("\n");
                    char *response = stream_response(sock, g.max_tokens);
                    if (response && (strlen(response) > 0 || g_native_tool_calls)) {
                        if (g_native_tool_calls) {
                            session_save_assistant_with_tool_calls(g.session_id, response, g_native_tool_calls);
                        } else {
                            session_save_turn(g.session_id, "assistant", response);
                        }
                        free(g.last_response);
                        g.last_response = strdup(response);
                    }
                    if (response && (g_native_tool_calls || strstr(response, "<tool_call>"))) {
                        char *final = handle_tool_calls(response);
                        free(final);
                    } else {
                        free(response);
                    }
                    g.turn_count++;
                }

                j->last_run_at = time(NULL);
                j->run_count++;
                cron_save();
                printf("\n");
                return;
            }
        }
        printf(ANSI_YELLOW "  No cron job with id '%s'" ANSI_RESET "\n\n", id);
        return;
    }

    printf(ANSI_YELLOW "  Unknown cron subcommand. Usage:" ANSI_RESET "\n");
    printf("  /cron add <min> <hr> <dom> <mon> <dow> <prompt>\n");
    printf("  /cron ls          — list all jobs\n");
    printf("  /cron rm <id>     — remove a job\n");
    printf("  /cron enable <id> — enable a disabled job\n");
    printf("  /cron disable <id>— disable a job\n");
    printf("  /cron run <id>    — manually trigger a job now\n\n");
}

static void cmd_context(const char *args __attribute__((unused))) {
    int used = g.total_tokens_in + g.total_tokens_out;
    int pct = MODEL_CTX > 0 ? (used * 100 / MODEL_CTX) : 0;
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
// Connections command and setup wizard
// ============================================================================

static void connections_show_status(void) {
    printf("\n" ANSI_BOLD "  Connections" ANSI_RESET "\n");
    printf("  ─────────────────────────────────────\n");
    int any_active = 0;
    for (int i = 0; i < g_connections_count; i++) {
        Connection *c = &g_connections[i];
        if (c->active) {
            if (c->is_oauth && strncmp(c->name, "google", 6) == 0) {
                // Show OAuth status for Google accounts
                printf("  " ANSI_GREEN "●" ANSI_RESET " %-20s " ANSI_DIM "authorized" ANSI_RESET "\n",
                       c->label);
            } else {
                // Mask key: show first 4 and last 4 chars
                char masked[32];
                int klen = (int)strlen(c->key);
                if (klen > 12) {
                    snprintf(masked, sizeof(masked), "%.4s····%.4s", c->key, c->key + klen - 4);
                } else {
                    snprintf(masked, sizeof(masked), "****");
                }
                printf("  " ANSI_GREEN "●" ANSI_RESET " %-20s " ANSI_DIM "%s" ANSI_RESET "\n",
                       c->label, masked);
            }
            any_active = 1;
        } else {
            // Don't show unconfigured additional google account slots
            if (strncmp(c->name, "google_", 7) == 0) continue;
            printf("  " ANSI_DIM "○ %-20s not configured" ANSI_RESET "\n", c->label);
        }
    }
    if (!any_active) {
        printf("\n  " ANSI_DIM "No connections configured. Run " ANSI_RESET
               "/connections setup" ANSI_DIM " to add them." ANSI_RESET "\n");
    }
    printf("  " ANSI_DIM "Add more Google accounts: /connections add google <label>" ANSI_RESET "\n");
    printf("\n");
}

// Test a connection by making a lightweight API call
static int connection_test(Connection *c) {
    char cmd[1024];
    if (strcmp(c->name, "brave_search") == 0) {
        snprintf(cmd, sizeof(cmd),
            "curl -sf -o /dev/null -w '%%{http_code}' "
            "-H 'X-Subscription-Token: %s' "
            "'https://api.search.brave.com/res/v1/web/search?q=test&count=1' 2>/dev/null",
            c->key);
    } else if (strcmp(c->name, "github") == 0) {
        snprintf(cmd, sizeof(cmd),
            "curl -sf -o /dev/null -w '%%{http_code}' "
            "-H 'Authorization: token %s' "
            "'https://api.github.com/user' 2>/dev/null",
            c->key);
    } else if (strcmp(c->name, "wolfram") == 0) {
        snprintf(cmd, sizeof(cmd),
            "curl -sf -o /dev/null -w '%%{http_code}' "
            "'https://api.wolframalpha.com/v2/query?input=1%%2B1&appid=%s&output=json' 2>/dev/null",
            c->key);
    } else if (strcmp(c->name, "telegram") == 0) {
        snprintf(cmd, sizeof(cmd),
            "curl -sf -o /dev/null -w '%%{http_code}' "
            "'https://api.telegram.org/bot%s/getMe' 2>/dev/null",
            c->key);
    } else if (strncmp(c->name, "google", 6) == 0 && c->is_oauth) {
        if (!oauth_ensure_token(c)) return 0;
        snprintf(cmd, sizeof(cmd),
            "curl -sf -o /dev/null -w '%%{http_code}' "
            "-H 'Authorization: Bearer %s' "
            "'https://gmail.googleapis.com/gmail/v1/users/me/profile' 2>/dev/null",
            c->access_token);
    } else {
        return -1;
    }
    FILE *p = popen(cmd, "r");
    if (!p) return -1;
    char resp[16];
    if (fgets(resp, sizeof(resp), p)) {
        pclose(p);
        return (resp[0] == '2') ? 1 : 0; // 2xx = success
    }
    pclose(p);
    return -1;
}

// OAuth2 browser-based setup flow for Google
static void connections_setup_google(Connection *c) {
    printf("\n" ANSI_BOLD "  Setting up: %s" ANSI_RESET "\n", c->label);
    printf(ANSI_DIM "  Gmail, Google Drive, Google Docs via browser sign-in" ANSI_RESET "\n\n");

    // Use built-in credentials unless user has already provided custom ones
    if (!c->client_id[0]) {
        printf("  " ANSI_BOLD "[1]" ANSI_RESET " Sign in with Google (recommended)\n");
        printf("  " ANSI_BOLD "[2]" ANSI_RESET " Use my own OAuth credentials (advanced)\n");
        printf("  " ANSI_DIM "[s]" ANSI_RESET " Skip\n\n  > ");
        fflush(stdout);
        char choice[16];
        if (!fgets(choice, sizeof(choice), stdin)) return;
        if (choice[0] == 's' || choice[0] == 'S') {
            printf(ANSI_DIM "  Skipped." ANSI_RESET "\n");
            return;
        }
        if (choice[0] == '2') {
            // Advanced: user provides their own Cloud project credentials
            printf("\n  " ANSI_BOLD "Custom OAuth setup:" ANSI_RESET "\n");
            printf("  1. Go to " ANSI_CYAN "https://console.cloud.google.com/apis/credentials" ANSI_RESET "\n");
            printf("  2. Create a project (or select existing)\n");
            printf("  3. Enable " ANSI_BOLD "Gmail API" ANSI_RESET ", " ANSI_BOLD "Google Drive API" ANSI_RESET ", " ANSI_BOLD "Google Docs API" ANSI_RESET "\n");
            printf("  4. Configure OAuth consent screen (add your email as test user)\n");
            printf("  5. Create credentials → OAuth client ID → Desktop app\n\n");

            printf("  " ANSI_BOLD "Client ID" ANSI_RESET " (or 'back'): ");
            fflush(stdout);
            char cid[256];
            if (!fgets(cid, sizeof(cid), stdin)) return;
            int len = (int)strlen(cid);
            while (len > 0 && (cid[len-1] == '\n' || cid[len-1] == '\r')) cid[--len] = 0;
            if (len == 0 || strcasecmp(cid, "back") == 0) {
                printf(ANSI_DIM "  Skipped." ANSI_RESET "\n");
                return;
            }

            printf("  " ANSI_BOLD "Client Secret" ANSI_RESET ": ");
            fflush(stdout);
            char cs[128];
            if (!fgets(cs, sizeof(cs), stdin)) return;
            len = (int)strlen(cs);
            while (len > 0 && (cs[len-1] == '\n' || cs[len-1] == '\r')) cs[--len] = 0;
            if (len == 0) { printf(ANSI_DIM "  Skipped." ANSI_RESET "\n"); return; }

            strlcpy(c->client_id, cid, sizeof(c->client_id));
            strlcpy(c->client_secret, cs, sizeof(c->client_secret));
        } else {
            // Default: use built-in PRE credentials
            strlcpy(c->client_id, PRE_GOOGLE_CLIENT_ID, sizeof(c->client_id));
            strlcpy(c->client_secret, PRE_GOOGLE_CLIENT_SECRET, sizeof(c->client_secret));
        }
    }

    // OAuth authorization via browser
    int port = 18492; // fixed port for redirect URI
    printf("  Opening your browser to sign in with Google...\n");

    char auth_url[2048];
    snprintf(auth_url, sizeof(auth_url),
        "%s?client_id=%s"
        "&redirect_uri=http://127.0.0.1:%d"
        "&response_type=code"
        "&scope=%s"
        "&access_type=offline"
        "&prompt=consent",
        GOOGLE_AUTH_URL, c->client_id, port, GOOGLE_SCOPES);

    // Open browser
    char open_cmd[2200];
    snprintf(open_cmd, sizeof(open_cmd), "open '%s'", auth_url);
    system(open_cmd);

    printf(ANSI_DIM "  Waiting for authorization..." ANSI_RESET "\n");
    printf(ANSI_DIM "  If the browser didn't open, visit:" ANSI_RESET "\n");
    printf(ANSI_CYAN "  %s" ANSI_RESET "\n\n", auth_url);

    char code[512] = {0};
    if (!oauth_listen_for_code(port, code, sizeof(code)) || !code[0]) {
        printf(ANSI_RED "  ✗ Authorization failed or timed out." ANSI_RESET "\n");
        return;
    }

    // Exchange code for tokens
    int codelen = (int)strlen(code);
    printf(ANSI_DIM "  Code received (%d chars): %.8s...%s" ANSI_RESET "\n",
           codelen, code, codelen > 12 ? code + codelen - 4 : "");
    printf(ANSI_DIM "  Exchanging authorization code..." ANSI_RESET);
    fflush(stdout);

    if (oauth_exchange_code(c, code, port)) {
        // Fetch the email address for this account
        char email_cmd[2048];
        snprintf(email_cmd, sizeof(email_cmd),
            "curl -s -H 'Authorization: Bearer %s' "
            "'https://gmail.googleapis.com/gmail/v1/users/me/profile' 2>/dev/null",
            c->access_token);
        FILE *ep = popen(email_cmd, "r");
        char email_resp[2048] = {0};
        if (ep) {
            fread(email_resp, 1, sizeof(email_resp) - 1, ep);
            pclose(ep);
        }
        char email[128] = {0};
        json_extract_str(email_resp, "emailAddress", email, sizeof(email));

        save_connections();
        if (email[0])
            printf("\r" ANSI_GREEN "  ● Connected: %s                              " ANSI_RESET "\n", email);
        else
            printf("\r" ANSI_GREEN "  ● Google connected! Gmail, Drive, Docs ready. " ANSI_RESET "\n");
    } else {
        printf("\r" ANSI_RED "  ✗ Token exchange failed.                     " ANSI_RESET "\n");
        c->active = 0;
    }
}

static void connections_setup_service(Connection *c) {
    // Delegate to OAuth flow for Google
    if (c->is_oauth && strcmp(c->name, "google") == 0) {
        connections_setup_google(c);
        return;
    }

    printf("\n" ANSI_BOLD "  Setting up: %s" ANSI_RESET "\n", c->label);

    // Show instructions per service
    if (strcmp(c->name, "brave_search") == 0) {
        printf(ANSI_DIM "  Free tier: 2,000 queries/month" ANSI_RESET "\n");
        printf("  1. Go to " ANSI_CYAN "https://brave.com/search/api/" ANSI_RESET "\n");
        printf("  2. Sign up and get your API key\n");
        printf("  3. Paste it below\n\n");
    } else if (strcmp(c->name, "github") == 0) {
        printf(ANSI_DIM "  Personal Access Token for repo/issue/PR access" ANSI_RESET "\n");
        printf("  1. Go to " ANSI_CYAN "https://github.com/settings/tokens" ANSI_RESET "\n");
        printf("  2. Generate a new token (classic) with repo scope\n");
        printf("  3. Paste it below\n\n");
    } else if (strcmp(c->name, "wolfram") == 0) {
        printf(ANSI_DIM "  Computational knowledge engine — math, science, data" ANSI_RESET "\n");
        printf("  1. Go to " ANSI_CYAN "https://developer.wolframalpha.com/access" ANSI_RESET "\n");
        printf("  2. Create an app and get your AppID\n");
        printf("  3. Paste it below\n\n");
    } else if (strcmp(c->name, "telegram") == 0) {
        printf(ANSI_DIM "  Chat with PRE from your phone via Telegram" ANSI_RESET "\n");
        printf("  1. Open Telegram and search for " ANSI_CYAN "@BotFather" ANSI_RESET "\n");
        printf("  2. Send " ANSI_BOLD "/newbot" ANSI_RESET " and follow the prompts to create a bot\n");
        printf("  3. Copy the bot token and paste it below\n\n");
        printf(ANSI_DIM "  After setup, run: pre-telegram" ANSI_RESET "\n\n");
    }

    printf("  " ANSI_BOLD "API key" ANSI_RESET " (or 'skip' / 'remove'): ");
    fflush(stdout);

    char input[512];
    if (!fgets(input, sizeof(input), stdin)) return;
    // Strip newline
    int len = (int)strlen(input);
    while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r')) input[--len] = 0;

    if (len == 0 || strcasecmp(input, "skip") == 0) {
        printf(ANSI_DIM "  Skipped." ANSI_RESET "\n");
        return;
    }
    if (strcasecmp(input, "remove") == 0) {
        c->key[0] = 0;
        c->active = 0;
        save_connections();
        printf(ANSI_GREEN "  Removed." ANSI_RESET "\n");
        return;
    }

    // Store and test
    strlcpy(c->key, input, sizeof(c->key));
    printf(ANSI_DIM "  Testing connection..." ANSI_RESET);
    fflush(stdout);

    int result = connection_test(c);
    if (result == 1) {
        c->active = 1;
        save_connections();
        printf("\r" ANSI_GREEN "  ● Connected!              " ANSI_RESET "\n");
    } else if (result == 0) {
        printf("\r" ANSI_RED "  ✗ Authentication failed — key not saved." ANSI_RESET "\n");
        c->key[0] = 0;
        c->active = 0;
    } else {
        // Can't reach API — save anyway, might be network issue
        c->active = 1;
        save_connections();
        printf("\r" ANSI_YELLOW "  ? Could not verify (network issue?) — key saved." ANSI_RESET "\n");
    }
}

static void connections_setup_wizard(void) {
    printf("\n" ANSI_BOLD ANSI_CYAN "  ┌─ Connection Setup ──────────────────────┐" ANSI_RESET "\n");
    printf(ANSI_BOLD ANSI_CYAN "  │" ANSI_RESET "  Connect external services to unlock      " ANSI_BOLD ANSI_CYAN "│" ANSI_RESET "\n");
    printf(ANSI_BOLD ANSI_CYAN "  │" ANSI_RESET "  web search, GitHub, and more.             " ANSI_BOLD ANSI_CYAN "│" ANSI_RESET "\n");
    printf(ANSI_BOLD ANSI_CYAN "  │" ANSI_RESET ANSI_DIM "  All keys stored locally in ~/.pre/        " ANSI_RESET ANSI_BOLD ANSI_CYAN "│" ANSI_RESET "\n");
    printf(ANSI_BOLD ANSI_CYAN "  │" ANSI_RESET ANSI_DIM "  Skip any you don't need — add later with  " ANSI_RESET ANSI_BOLD ANSI_CYAN "│" ANSI_RESET "\n");
    printf(ANSI_BOLD ANSI_CYAN "  │" ANSI_RESET ANSI_DIM "  /connections setup                        " ANSI_RESET ANSI_BOLD ANSI_CYAN "│" ANSI_RESET "\n");
    printf(ANSI_BOLD ANSI_CYAN "  └─────────────────────────────────────────┘" ANSI_RESET "\n");

    for (int i = 0; i < g_connections_count; i++) {
        connections_setup_service(&g_connections[i]);
    }

    printf("\n");
    connections_show_status();
}

static void cmd_connections(const char *args) {
    if (!g_connections_loaded) load_connections();

    if (!args || !args[0]) {
        connections_show_status();
        printf(ANSI_DIM "  Usage:" ANSI_RESET "\n");
        printf("    /connections setup              Run full setup wizard\n");
        printf("    /connections add <service>      Configure one service\n");
        printf("    /connections add google <label> Add another Google account\n");
        printf("    /connections remove <service>   Remove a service\n");
        printf("    /connections test               Test all connections\n\n");
        printf(ANSI_DIM "  Services: brave_search, github, google, wolfram, telegram" ANSI_RESET "\n\n");
        return;
    }

    // Skip leading whitespace
    while (*args == ' ') args++;

    if (strcasecmp(args, "setup") == 0) {
        connections_setup_wizard();
        return;
    }

    if (strcasecmp(args, "test") == 0) {
        printf("\n" ANSI_BOLD "  Testing connections..." ANSI_RESET "\n\n");
        for (int i = 0; i < g_connections_count; i++) {
            Connection *c = &g_connections[i];
            if (!c->active) {
                printf("  " ANSI_DIM "○ %-20s skipped (not configured)" ANSI_RESET "\n", c->label);
                continue;
            }
            printf("  " ANSI_DIM "  %-20s testing..." ANSI_RESET, c->label);
            fflush(stdout);
            int r = connection_test(c);
            if (r == 1)
                printf("\r  " ANSI_GREEN "● %-20s ok" ANSI_RESET "                \n", c->label);
            else if (r == 0)
                printf("\r  " ANSI_RED "✗ %-20s auth failed" ANSI_RESET "         \n", c->label);
            else
                printf("\r  " ANSI_YELLOW "? %-20s unreachable" ANSI_RESET "        \n", c->label);
        }
        printf("\n");
        return;
    }

    // /connections add <service> [label]
    // e.g. /connections add google        → default Google account
    //      /connections add google work   → additional Google account named "work"
    if (strncasecmp(args, "add ", 4) == 0) {
        const char *svc = args + 4;
        while (*svc == ' ') svc++;

        // Check for "google <label>" pattern for multi-account
        if (strncasecmp(svc, "google ", 7) == 0) {
            const char *label = svc + 7;
            while (*label == ' ') label++;
            if (*label) {
                // Check if this account already exists
                char full_name[64];
                snprintf(full_name, sizeof(full_name), "google_%s", label);
                Connection *c = get_connection(full_name);
                if (c && c->active) {
                    printf(ANSI_YELLOW "  Google account '%s' already connected." ANSI_RESET "\n", label);
                    printf("  Use " ANSI_BOLD "/connections remove %s" ANSI_RESET " first to re-authorize.\n\n", full_name);
                    return;
                }
                if (!c) {
                    c = add_google_account(label);
                    if (!c) {
                        printf(ANSI_RED "  Too many connections (max %d)." ANSI_RESET "\n\n", MAX_CONNECTIONS);
                        return;
                    }
                }
                connections_setup_google(c);
                printf("\n");
                return;
            }
        }

        Connection *c = get_connection(svc);
        if (!c) {
            printf(ANSI_RED "  Unknown service: %s" ANSI_RESET "\n", svc);
            printf(ANSI_DIM "  Available: brave_search, github, google, wolfram" ANSI_RESET "\n");
            printf(ANSI_DIM "  Multi-account: /connections add google <label>" ANSI_RESET "\n\n");
            return;
        }
        connections_setup_service(c);
        printf("\n");
        return;
    }

    // /connections remove <service>
    if (strncasecmp(args, "remove ", 7) == 0) {
        const char *svc = args + 7;
        while (*svc == ' ') svc++;
        Connection *c = get_connection(svc);
        if (!c) {
            // Try google_<label> pattern
            char full_name[64];
            snprintf(full_name, sizeof(full_name), "google_%s", svc);
            c = get_connection(full_name);
        }
        if (!c) {
            printf(ANSI_RED "  Unknown service: %s" ANSI_RESET "\n\n", svc);
            return;
        }
        if (c->is_oauth) {
            c->client_id[0] = c->client_secret[0] = 0;
            c->access_token[0] = c->refresh_token[0] = 0;
            c->token_expiry = 0;
        }
        c->key[0] = 0;
        c->active = 0;
        save_connections();
        printf(ANSI_GREEN "  Removed %s connection." ANSI_RESET "\n\n", c->label);
        return;
    }

    printf(ANSI_YELLOW "  Unknown subcommand: %s" ANSI_RESET "\n", args);
    printf(ANSI_DIM "  Try: /connections setup, test, add <service>, remove <service>" ANSI_RESET "\n\n");
}

// First-run check — offer setup if no connections.json exists
static void first_run_check(void) {
    if (!g_connections_loaded) load_connections();

    struct stat st;
    int is_first = (stat(connections_path(), &st) != 0);
    int needs_name = (stat(identity_path(), &st) != 0);

    if (!is_first && !needs_name) return;

    if (needs_name) {
        printf(ANSI_DIM "  ─────────────────────────────────────────" ANSI_RESET "\n");
        printf("  " ANSI_BOLD "Welcome to PRE." ANSI_RESET
               " Let's personalize your agent.\n\n");
        printf("  What would you like to name your assistant?\n");
        printf("  " ANSI_DIM "(This becomes its identity — used in prompts and the banner)" ANSI_RESET "\n\n");
        printf("  Name: ");
        fflush(stdout);

        char name_buf[128];
        if (fgets(name_buf, sizeof(name_buf), stdin)) {
            // Trim whitespace
            char *p = name_buf;
            while (*p == ' ') p++;
            size_t len = strlen(p);
            while (len > 0 && (p[len-1] == '\n' || p[len-1] == '\r' || p[len-1] == ' '))
                p[--len] = 0;
            if (len > 0) {
                save_identity(p);
                printf("\n  " ANSI_GREEN "Hello! I'm %s." ANSI_RESET "\n\n", g_agent_name);
                // Also save as a memory so the model knows its name
                save_memory("agent_identity", "user", "The agent's chosen name and identity",
                    g_agent_name, "global");
            } else {
                printf(ANSI_DIM "  Using default name: PRE" ANSI_RESET "\n\n");
                save_identity("PRE");
            }
        }
    }

    if (is_first) {
        printf(ANSI_DIM "  ─────────────────────────────────────────" ANSI_RESET "\n");
        printf("  Would you like to set up external connections\n"
               "  (web search, GitHub, etc.)?\n\n");
        printf("  " ANSI_BOLD "[y]" ANSI_RESET " Run setup    "
               ANSI_BOLD "[n]" ANSI_RESET " Skip (run /connections later)\n\n  > ");
        fflush(stdout);

        char ch[16];
        if (fgets(ch, sizeof(ch), stdin) && (ch[0] == 'y' || ch[0] == 'Y')) {
            connections_setup_wizard();
        } else {
            // Create empty file so we don't prompt again
            FILE *f = fopen(connections_path(), "w");
            if (f) { fprintf(f, "{}\n"); fclose(f); chmod(connections_path(), 0600); }
            printf(ANSI_DIM "  Skipped. Run /connections setup anytime." ANSI_RESET "\n\n");
        }
    }
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
// Clipboard image paste (Ctrl+V)
// ============================================================================

static const char *ctrlv_paste_cb(void) {
    @autoreleasepool {
        NSPasteboard *pb = [NSPasteboard generalPasteboard];

        // Check for image data on clipboard
        NSData *pngData = [pb dataForType:NSPasteboardTypePNG];
        if (!pngData) {
            // Try TIFF (screenshots, Preview copies) and convert to PNG
            NSData *tiffData = [pb dataForType:NSPasteboardTypeTIFF];
            if (tiffData) {
                NSBitmapImageRep *rep = [NSBitmapImageRep imageRepWithData:tiffData];
                if (rep) {
                    pngData = [rep representationUsingType:NSBitmapImageFileTypePNG
                                               properties:@{}];
                }
            }
        }

        if (pngData && pngData.length > 0) {
            // Base64-encode the PNG data
            NSString *b64 = [pngData base64EncodedStringWithOptions:0];
            const char *b64c = [b64 UTF8String];

            // Store as pending image
            free(g_pending_image);
            g_pending_image = strdup(b64c);

            // Return placeholder text to insert into the input line
            static char label[64];
            double kb = pngData.length / 1024.0;
            if (kb >= 1024.0)
                snprintf(label, sizeof(label), "[image: %.1fMB]", kb / 1024.0);
            else
                snprintf(label, sizeof(label), "[image: %.0fKB]", kb);
            return label;
        }

        // No image — check for plain text and paste that instead
        NSString *text = [pb stringForType:NSPasteboardTypeString];
        if (text && text.length > 0) {
            static char textbuf[4096];
            const char *utf8 = [text UTF8String];
            size_t len = strlen(utf8);
            if (len >= sizeof(textbuf)) len = sizeof(textbuf) - 1;
            memcpy(textbuf, utf8, len);
            textbuf[len] = 0;
            return textbuf;
        }

        return NULL;
    }
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
        // Save our executable path for fork+exec subprocess launching
        if (argv[0][0] == '/') {
            strlcpy(g_exe_path, argv[0], sizeof(g_exe_path));
        } else {
            // Resolve relative path
            char *rp = realpath(argv[0], NULL);
            if (rp) { strlcpy(g_exe_path, rp, sizeof(g_exe_path)); free(rp); }
            else strlcpy(g_exe_path, argv[0], sizeof(g_exe_path));
        }

        // --artifact mode: launched by show_artifact_window via fork+exec
        if (argc >= 3 && strcmp(argv[1], "--artifact") == 0) {
            return artifact_window_main(argv[2], argc >= 4 ? argv[3] : "Artifact");
        }

        // --pdf mode: launched by pdf_export via fork+exec
        if (argc >= 4 && strcmp(argv[1], "--pdf") == 0) {
            return pdf_export_main(argv[2], argv[3]);
        }

        // Unload model on Ctrl+C or kill
        signal(SIGINT, handle_exit_signal);
        signal(SIGTERM, handle_exit_signal);

        // Read context window from ~/.pre/context (written by install.sh)
        {
            NSString *ctxPath = [NSHomeDirectory() stringByAppendingPathComponent:@".pre/context"];
            NSString *ctxStr = [NSString stringWithContentsOfFile:ctxPath encoding:NSUTF8StringEncoding error:nil];
            if (ctxStr) {
                int v = [[ctxStr stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] intValue];
                if (v >= 2048 && v <= MAX_CONTEXT) MODEL_CTX = v;
            }
        }

        // Defaults
        g.port = 11434;
        g.max_tokens = 16384;
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
                    printf("  --max-tokens N   Max response tokens (default: 16384)\n");
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

        // Load agent identity
        load_identity();

        // Load Argus state
        argus_load();

        // Banner
        tui_banner();

        // Health check
        if (!health_check()) {
            fprintf(stderr, ANSI_RED "  Ollama not running on port %d.\n" ANSI_RESET, g.port);
            fprintf(stderr, "  Start it: " ANSI_BOLD "ollama serve" ANSI_RESET " or launch " ANSI_BOLD "Ollama.app" ANSI_RESET "\n\n");
            return 1;
        }
        printf(ANSI_GREEN "  Server connected." ANSI_RESET "\n\n");

        // First-run: offer connection setup
        first_run_check();

        // Auto-start Telegram bot if configured
        start_telegram();

        g.session_start_ms = now_ms();

        // Load cron jobs and catch up on any missed while system was down
        cron_load();
        if (g_cron_count > 0) {
            printf(ANSI_DIM "  %d cron job%s loaded" ANSI_RESET "\n",
                   g_cron_count, g_cron_count == 1 ? "" : "s");
            cron_check_missed();
        }

        // Load ComfyUI config
        comfyui_load_config();
        if (g_comfyui_installed)
            printf(ANSI_DIM "  ComfyUI available (image generation)" ANSI_RESET "\n");

        // Resume session
        if (resume_id) {
            int turns = session_load(g.session_id);
            if (turns == 0) printf(ANSI_YELLOW "  [session '%s' not found — starting fresh]" ANSI_RESET "\n\n", g.session_id);
            else g.turn_count = turns / 2;
            session_load_title(g.session_id, g.session_title, sizeof(g.session_title));
        }

        // Compact any uncompacted artifact content from previous runs.
        // This prevents context bloat from HTML that wasn't compacted due to bugs.
        compact_artifact_content(g.session_id);

        // Linenoise setup
        linenoiseSetMultiLine(1);
        linenoiseSetCompletionCallback(completion_cb);
        linenoiseSetHintsCallback(hints_cb);
        linenoiseSetCtrlVCallback(ctrlv_paste_cb);
        linenoiseHistoryLoad(g.history_path);
        linenoiseHistorySetMaxLen(500);

        // Main loop
        for (;;) {
            // Check cron jobs (runs at most once per minute)
            cron_check_and_run();

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

            // Clear any stale native tool calls from previous turn
            free(g_native_tool_calls);
            g_native_tool_calls = NULL;

            // Save user turn (save original input, not the preamble-augmented version)
            session_save_turn(g.session_id, "user", message);

            // Auto-compact if context is getting large
            maybe_compact();

            // Send to server
            int sock = send_request(message, g.max_tokens, g.session_id);
            free(message);
            if (sock < 0) continue;

            printf("\n");
            char *response = stream_response(sock, g.max_tokens);

            // Save assistant turn — use native format if tool_calls are present.
            // With native tool calls, content may be empty but we still need to save.
            if (response && (strlen(response) > 0 || g_native_tool_calls)) {
                if (g_native_tool_calls) {
                    session_save_assistant_with_tool_calls(g.session_id, response, g_native_tool_calls);
                } else {
                    session_save_turn(g.session_id, "assistant", response);
                }
                free(g.last_response);
                g.last_response = strdup(response);
            }

            // Detect HTML code dumped in chat without a tool call. Instead of
            // nudging the model to retry (which doesn't work — it just dumps code
            // again), auto-extract the HTML and create the artifact ourselves.
            if (response && !g_native_tool_calls && !strstr(response, "<tool_call>")) {
                // Find the last/best HTML block in the response
                // Try ```html fenced block first, then raw <!DOCTYPE or <html
                char *html_content = NULL;
                char *fence = strstr(response, "```html");
                if (!fence) fence = strstr(response, "```HTML");
                if (fence) {
                    // Extract content between ```html and closing ```
                    char *start = strchr(fence + 7, '\n');
                    if (start) {
                        start++; // past newline
                        char *end = strstr(start, "```");
                        if (end) {
                            html_content = malloc(end - start + 1);
                            memcpy(html_content, start, end - start);
                            html_content[end - start] = 0;
                        }
                    }
                }
                // Also check for the LAST ```html block (model often writes multiple)
                if (fence) {
                    char *last_fence = fence;
                    char *scan = fence + 7; // past first "```html"
                    char *next;
                    while ((next = strstr(scan, "```html")) != NULL) {
                        last_fence = next;
                        scan = next + 7;
                    }
                    // Also check for ```HTML variant
                    scan = fence + 7;
                    while ((next = strstr(scan, "```HTML")) != NULL) {
                        if (next > last_fence) last_fence = next;
                        scan = next + 7;
                    }
                    if (last_fence != fence) {
                        char *start = strchr(last_fence + 7, '\n');
                        if (start) {
                            start++;
                            char *end = strstr(start, "```");
                            if (end && (end - start) > (int)(html_content ? strlen(html_content) : 0)) {
                                free(html_content);
                                html_content = malloc(end - start + 1);
                                memcpy(html_content, start, end - start);
                                html_content[end - start] = 0;
                            }
                        }
                    }
                }
                // Check for <artifact ...>...</artifact> XML format (wrong format but recoverable)
                if (!html_content) {
                    char *art_tag = strstr(response, "<artifact ");
                    if (art_tag) {
                        // Find the closing > of the opening tag
                        char *tag_end = strchr(art_tag, '>');
                        if (tag_end) {
                            tag_end++; // past >
                            char *art_close = strstr(tag_end, "</artifact>");
                            if (!art_close) art_close = response + strlen(response);
                            size_t len = art_close - tag_end;
                            if (len > 100) {
                                html_content = malloc(len + 1);
                                memcpy(html_content, tag_end, len);
                                html_content[len] = 0;
                                // Strip leading/trailing whitespace
                                while (html_content[0] == '\n') memmove(html_content, html_content + 1, strlen(html_content));
                            }
                        }
                    }
                }
                // Fallback: find raw <!DOCTYPE, <html, or HTML fragment tags
                if (!html_content) {
                    char *doctype = strstr(response, "<!DOCTYPE");
                    if (!doctype) doctype = strstr(response, "<html");
                    // Also detect HTML fragments dumped without a full document wrapper
                    if (!doctype) doctype = strstr(response, "<article>");
                    if (!doctype) doctype = strstr(response, "<article ");
                    if (doctype) {
                        // Take everything from doctype to end of </html> or </article> or end of response
                        char *html_end = strstr(doctype, "</html>");
                        if (html_end) html_end += 7;
                        if (!html_end) { html_end = strstr(doctype, "</article>"); if (html_end) html_end += 10; }
                        if (!html_end) html_end = response + strlen(response);
                        size_t len = html_end - doctype;
                        // For fragments, also grab any trailing <script> and <style> blocks
                        if (html_end < response + strlen(response)) {
                            char *trail = html_end;
                            while (*trail == '\n' || *trail == ' ') trail++;
                            if (strncmp(trail, "<script", 7) == 0 || strncmp(trail, "<style", 6) == 0) {
                                html_end = response + strlen(response); // grab everything
                                len = html_end - doctype;
                            }
                        }
                        html_content = malloc(len + 1);
                        memcpy(html_content, doctype, len);
                        html_content[len] = 0;
                    }
                }

                if (html_content && strlen(html_content) > 100) {
                    printf(ANSI_YELLOW "\n  [HTML detected in chat — auto-creating artifact]"
                           ANSI_RESET "\n");
                    // If it's an HTML fragment (no <!DOCTYPE or <html), wrap it
                    if (!strstr(html_content, "<!DOCTYPE") && !strstr(html_content, "<html")) {
                        size_t frag_len = strlen(html_content);
                        // Check if there's a <style> block to extract
                        char *style_block = strstr(html_content, "<style>");
                        char *style_end = style_block ? strstr(style_block, "</style>") : NULL;
                        char *style_str = "";
                        size_t style_len = 0;
                        if (style_block && style_end) {
                            style_len = (style_end + 8) - style_block;
                            style_str = style_block;
                        }
                        // Check for <script> blocks
                        char *script_block = strstr(html_content, "<script");
                        size_t wrap_cap = frag_len + 512;
                        char *wrapped = malloc(wrap_cap);
                        int wlen = snprintf(wrapped, wrap_cap,
                            "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
                            "<meta name='viewport' content='width=device-width,initial-scale=1'>");
                        // Copy style into head if found
                        if (style_len > 0) {
                            memcpy(wrapped + wlen, style_str, style_len);
                            wlen += style_len;
                        }
                        wlen += snprintf(wrapped + wlen, wrap_cap - wlen, "</head><body>");
                        // Copy content (skip the style block since we moved it to head)
                        if (style_block && style_block < script_block) {
                            // Copy up to style, skip style, copy rest up to script
                            memcpy(wrapped + wlen, html_content, style_block - html_content);
                            wlen += style_block - html_content;
                            char *after_style = style_end ? style_end + 8 : style_block;
                            size_t rest = frag_len - (after_style - html_content);
                            memcpy(wrapped + wlen, after_style, rest);
                            wlen += rest;
                        } else {
                            memcpy(wrapped + wlen, html_content, frag_len);
                            wlen += frag_len;
                        }
                        wlen += snprintf(wrapped + wlen, wrap_cap - wlen, "</body></html>");
                        wrapped[wlen] = 0;
                        free(html_content);
                        html_content = wrapped;
                    }
                    // Extract title from <title>...</title> tag, or <h1>, or <header> text
                    char auto_title[256] = "Auto-Extracted Content";
                    // Try <title> first, then <h1>
                    char *title_start = strstr(html_content, "<title>");
                    int title_tag_len = 7;
                    char *title_close = "</title>";
                    if (!title_start) {
                        title_start = strstr(html_content, "<h1>");
                        title_tag_len = 4;
                        title_close = "</h1>";
                        if (!title_start) {
                            title_start = strstr(html_content, "<h1 ");
                            if (title_start) {
                                char *gt = strchr(title_start, '>');
                                if (gt) { title_tag_len = (int)(gt + 1 - title_start); }
                            }
                        }
                    }
                    if (title_start) {
                        title_start += title_tag_len;
                        char *title_end = strstr(title_start, title_close);
                        if (title_end && (title_end - title_start) > 0 && (title_end - title_start) < 200) {
                            size_t tlen = title_end - title_start;
                            memcpy(auto_title, title_start, tlen);
                            auto_title[tlen] = 0;
                            // Strip HTML tags from title (e.g. <strong>)
                            char clean[256]; int ci2 = 0;
                            for (int j = 0; auto_title[j] && ci2 < 254; j++) {
                                if (auto_title[j] == '<') { while (auto_title[j] && auto_title[j] != '>') j++; }
                                else clean[ci2++] = auto_title[j];
                            }
                            clean[ci2] = 0;
                            strlcpy(auto_title, clean, sizeof(auto_title));
                            // Strip leading/trailing whitespace
                            char *t = auto_title;
                            while (*t == ' ' || *t == '\n' || *t == '\t') t++;
                            if (t != auto_title) memmove(auto_title, t, strlen(t) + 1);
                            size_t at_len = strlen(auto_title);
                            while (at_len > 0 && (auto_title[at_len-1] == ' ' || auto_title[at_len-1] == '\n'))
                                auto_title[--at_len] = 0;
                        }
                    }
                    char auto_output[65536];
                    execute_artifact(auto_title, html_content, "html",
                                    NULL, auto_output, sizeof(auto_output));
                    size_t html_len = strlen(html_content);
                    free(html_content);

                    // Compact the session: replace the full response with a brief note.
                    // The model's response often has thousands of tokens of HTML code
                    // that was dumped in chat. Keeping ANY of it bloats every future prefill.
                    // Extract just the first ~200 chars of non-code text as context.
                    char compact_buf[512];
                    const char *src = response;
                    int ci = 0;
                    // Copy leading text up to the first code fence or HTML tag
                    while (*src && ci < 200) {
                        if (*src == '`' && *(src+1) == '`' && *(src+2) == '`') break;
                        if (*src == '<' && (strncmp(src, "<!DOCTYPE", 9) == 0 ||
                                           strncmp(src, "<html", 5) == 0)) break;
                        compact_buf[ci++] = *src++;
                    }
                    // Trim trailing whitespace
                    while (ci > 0 && (compact_buf[ci-1] == ' ' || compact_buf[ci-1] == '\n')) ci--;
                    snprintf(compact_buf + ci, sizeof(compact_buf) - ci,
                             "\n[code auto-extracted to artifact — %zu bytes]", html_len);
                    session_replace_last_turn(g.session_id, "assistant", compact_buf);
                } else {
                    free(html_content);
                }
            }

            // Nudge: if the model described creating something but didn't call a tool,
            // send a follow-up to trigger the actual tool call. Safety net for when the
            // model narrates what it plans to build but forgets the <tool_call> tags.
            if (response && !g_native_tool_calls && !strstr(response, "<tool_call>")) {
                // Check if the response suggests the model intended to create an artifact
                int intended_tool = (strstr(response, "I'll create") || strstr(response, "I'll build") ||
                                     strstr(response, "I will create") || strstr(response, "I will build") ||
                                     strstr(response, "Creating") || strstr(response, "creating the") ||
                                     strstr(response, "I'll use") || strstr(response, "the artifact") ||
                                     strstr(response, "generate the") || strstr(response, "I'll make") ||
                                     strstr(response, "here it is") || strstr(response, "Here it is") ||
                                     strstr(response, "Here's the"));
                if (intended_tool) {
                    printf(ANSI_YELLOW "\n  [model described creating content but didn't call tool — nudging]"
                           ANSI_RESET "\n");
                    const char *nudge = "You did not use a tool call. You MUST use this exact format:\n"
                        "<tool_call>\n"
                        "{\"name\": \"artifact\", \"arguments\": {\"title\": \"Your Title\", \"content\": \"...HTML...\", \"type\": \"html\"}}\n"
                        "</tool_call>\n"
                        "Do NOT output raw HTML or use any other format. Use <tool_call> tags NOW with the artifact tool.";
                    session_save_turn(g.session_id, "user", nudge);

                    free(g_native_tool_calls);
                    g_native_tool_calls = NULL;

                    int nudge_budget = g.max_tokens * 4;
                    if (nudge_budget > MAX_CONTEXT / 2) nudge_budget = MAX_CONTEXT / 2;
                    int sock = send_request(NULL, nudge_budget, g.session_id);
                    if (sock >= 0) {
                        printf("\n");
                        char *nudge_response = stream_response(sock, nudge_budget);

                        // Drop the nudge from session and save the new response
                        session_replace_last_turn(g.session_id, "user", NULL);
                        if (nudge_response && (strlen(nudge_response) > 0 || g_native_tool_calls)) {
                            if (g_native_tool_calls) {
                                session_save_assistant_with_tool_calls(g.session_id, nudge_response, g_native_tool_calls);
                            } else {
                                session_save_turn(g.session_id, "assistant", nudge_response);
                            }
                        }

                        // If the nudge produced HTML in chat, auto-extract it
                        if (nudge_response && !g_native_tool_calls && !strstr(nudge_response, "<tool_call>")) {
                            char *html_start = strstr(nudge_response, "```html");
                            if (!html_start) html_start = strstr(nudge_response, "<!DOCTYPE");
                            if (!html_start) html_start = strstr(nudge_response, "<html");
                            if (html_start) {
                                char *content_start = html_start;
                                char *content_end = NULL;
                                if (html_start[0] == '`') {
                                    content_start = strchr(html_start + 7, '\n');
                                    if (content_start) { content_start++; content_end = strstr(content_start, "```"); }
                                } else {
                                    content_end = strstr(content_start, "</html>");
                                    if (content_end) content_end += 7;
                                    else content_end = nudge_response + strlen(nudge_response);
                                }
                                if (content_start && content_end && content_end > content_start) {
                                    size_t clen = content_end - content_start;
                                    char *extracted = malloc(clen + 1);
                                    memcpy(extracted, content_start, clen);
                                    extracted[clen] = 0;
                                    if (clen > 100) {
                                        printf(ANSI_YELLOW "  [HTML extracted from nudge response — creating artifact]"
                                               ANSI_RESET "\n");
                                        // Extract title from <title> tag
                                        char nudge_title[256] = "Auto-Extracted Content";
                                        char *nt_start = strstr(extracted, "<title>");
                                        if (nt_start) {
                                            nt_start += 7;
                                            char *nt_end = strstr(nt_start, "</title>");
                                            if (nt_end && (nt_end - nt_start) > 0 && (nt_end - nt_start) < 200) {
                                                size_t ntlen = nt_end - nt_start;
                                                memcpy(nudge_title, nt_start, ntlen);
                                                nudge_title[ntlen] = 0;
                                                // Strip whitespace
                                                char *nt = nudge_title;
                                                while (*nt == ' ' || *nt == '\n' || *nt == '\t') nt++;
                                                if (nt != nudge_title) memmove(nudge_title, nt, strlen(nt) + 1);
                                                size_t nl = strlen(nudge_title);
                                                while (nl > 0 && (nudge_title[nl-1] == ' ' || nudge_title[nl-1] == '\n'))
                                                    nudge_title[--nl] = 0;
                                            }
                                        }
                                        char auto_output[65536];
                                        execute_artifact(nudge_title, extracted, "html",
                                                        NULL, auto_output, sizeof(auto_output));
                                        // Compact session
                                        char compact_note[256];
                                        snprintf(compact_note, sizeof(compact_note),
                                                 "[artifact auto-created — %zu bytes]", clen);
                                        session_replace_last_turn(g.session_id, "assistant", compact_note);
                                    }
                                    free(extracted);
                                }
                            }
                        }

                        // Replace the original response with the nudge response for tool handling
                        free(response);
                        response = nudge_response;
                    }
                }
            }

            // Handle tool calls
            response = handle_tool_calls(response);

            free(response);
            g.turn_count++;

            // Stats are shown inline by stream_response()
        }

        // Cleanup — stop services, unload model from GPU
        printf(ANSI_DIM "\nStopping services..." ANSI_RESET "\n");
        stop_telegram();
        comfyui_stop();
        ollama_unload();
        free(g.last_response);
        free(g_pending_attach);
        free(g_pending_image);
        g_pending_image = NULL;

        printf(ANSI_DIM "Goodbye." ANSI_RESET "\n");
        return 0;
    }
}
