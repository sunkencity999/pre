// telegram.m — Telegram bot bridge for PRE (Personal Reasoning Engine)
//
// Bridges Telegram Bot API to Ollama, giving phone access to PRE's AI agent.
// Uses the same model, system prompt style, memory, and connections as the CLI.
//
// Usage:
//   ./pre-telegram [--port 11434]
//
// Requires:
//   - Telegram bot token in ~/.pre/connections.json (via /connections add telegram)
//   - Ollama running with pre-gemma4 model loaded
//
// Build:
//   make telegram   (or: clang -O3 ... telegram.m -o pre-telegram)
//
// Security:
//   - Only the first Telegram user to message the bot is authorized (owner)
//   - Owner ID is persisted in ~/.pre/telegram_owner
//   - Only read-only + web + memory tools available (no bash, file_write, etc.)

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <dirent.h>
#include <glob.h>
#include <time.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <limits.h>

// ============================================================================
// Constants
// ============================================================================

#define TELEGRAM_API       "https://api.telegram.org/bot"
#define MAX_TOOL_ARGS      8
#define MAX_TOOL_RESPONSE  8192
#define MAX_RESPONSE       (256 * 1024)
#define MAX_MSG_LEN        4096   // Telegram message limit
#define MAX_HISTORY        50     // conversation turns to keep
#define MAX_MEMORIES       128
#define MAX_CONNECTIONS    16
#define MAX_TOOL_LOOP      10
#define CONNECTIONS_FILE   ".pre/connections.json"

// Built-in Google OAuth credentials (same as CLI)
#define PRE_GOOGLE_CLIENT_ID "1062005591474-bj9c932m52vrvh5cl8fr02j1dti99jdh.apps.googleusercontent.com"
#define PRE_GOOGLE_CLIENT_SECRET "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"

static char g_agent_name[128] = "PRE";

static void load_identity(void) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/.pre/identity.json", getenv("HOME") ?: "/tmp");
    FILE *f = fopen(path, "r");
    if (!f) return;
    char buf[512];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);
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

static volatile int g_running = 1;
static void sighandler(int sig) { (void)sig; g_running = 0; }

// ============================================================================
// Structures
// ============================================================================

typedef struct {
    char name[64];
    int argc;
    char keys[MAX_TOOL_ARGS][64];
    char *vals[MAX_TOOL_ARGS];
} ToolCall;

static void tool_call_free(ToolCall *tc) {
    for (int i = 0; i < tc->argc; i++) { free(tc->vals[i]); tc->vals[i] = NULL; }
    tc->argc = 0;
}

static const char *tool_call_get(const ToolCall *tc, const char *key) {
    for (int i = 0; i < tc->argc; i++)
        if (strcmp(tc->keys[i], key) == 0) return tc->vals[i];
    return NULL;
}

typedef struct {
    char name[32];
    char label[64];
    char key[512];
    int active;
    int is_oauth;
    char client_id[256];
    char client_secret[128];
    char access_token[2048];
    char refresh_token[512];
    long token_expiry;
} Connection;

typedef struct {
    char name[128];
    char type[32];
    char description[256];
    char file[PATH_MAX];
} MemoryEntry;

// Per-chat conversation history
typedef struct {
    char role[16];     // "user", "assistant", "system"
    char *content;     // heap-allocated
} ChatMsg;

typedef struct {
    long chat_id;
    ChatMsg history[MAX_HISTORY * 2];
    int history_count;
} Conversation;

#define MAX_CONVERSATIONS 32
static Conversation g_convos[MAX_CONVERSATIONS];
static int g_convo_count = 0;

// ============================================================================
// Globals
// ============================================================================

static Connection g_connections[MAX_CONNECTIONS];
static int g_connections_count = 0;

static MemoryEntry g_memories[MAX_MEMORIES];
static int g_memory_count = 0;

static char g_bot_token[512] = {0};
static long g_owner_id = 0;       // authorized Telegram user ID
static long g_active_chat_id = 0; // current chat for artifact delivery
static int g_port = 11434;
static const char *g_model = "pre-gemma4";

// ============================================================================
// JSON helpers (same patterns as pre.m)
// ============================================================================

static int json_extract_str(const char *json, const char *key, char *dst, size_t dsz) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return 0;
    p += strlen(needle);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != ':') return 0;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p == '"') {
        p++;
    } else {
        size_t i = 0;
        while (*p && *p != ',' && *p != '}' && *p != ' ' && *p != '\n' && i < dsz - 1)
            dst[i++] = *p++;
        dst[i] = 0;
        return i > 0;
    }
    size_t i = 0;
    while (*p && *p != '"' && i < dsz - 1) {
        if (*p == '\\' && p[1]) {
            switch (p[1]) {
                case 'n': dst[i++] = '\n'; break;
                case 't': dst[i++] = '\t'; break;
                case 'r': dst[i++] = '\r'; break;
                case '"': dst[i++] = '"'; break;
                case '\\': dst[i++] = '\\'; break;
                case '/': dst[i++] = '/'; break;
                default: dst[i++] = p[1]; break;
            }
            p += 2;
        }
        else dst[i++] = *p++;
    }
    dst[i] = 0;
    return 1;
}

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

static char *json_escape_alloc(const char *s) {
    if (!s) return strdup("");
    size_t len = strlen(s);
    char *out = malloc(len * 2 + 1);
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        switch (s[i]) {
            case '"':  out[j++] = '\\'; out[j++] = '"'; break;
            case '\\': out[j++] = '\\'; out[j++] = '\\'; break;
            case '\n': out[j++] = '\\'; out[j++] = 'n'; break;
            case '\r': out[j++] = '\\'; out[j++] = 'r'; break;
            case '\t': out[j++] = '\\'; out[j++] = 't'; break;
            default:   out[j++] = s[i]; break;
        }
    }
    out[j] = 0;
    return out;
}

// ============================================================================
// Curl helper — run curl and capture output
// ============================================================================

static char *curl_exec(const char *cmd, size_t max_sz) {
    FILE *p = popen(cmd, "r");
    if (!p) return NULL;
    char *buf = malloc(max_sz);
    size_t len = 0;
    while (len < max_sz - 1) {
        size_t n = fread(buf + len, 1, max_sz - 1 - len, p);
        if (n == 0) break;
        len += n;
    }
    buf[len] = 0;
    pclose(p);
    return buf;
}

// ============================================================================
// Telegram API
// ============================================================================

static char *tg_api(const char *method, const char *json_body) {
    char cmd[8192];
    if (json_body) {
        // Write body to temp file to avoid shell escaping
        char tmp[64];
        snprintf(tmp, sizeof(tmp), "/tmp/pre_tg_%d.json", getpid());
        FILE *f = fopen(tmp, "w");
        if (!f) return NULL;
        fputs(json_body, f);
        fclose(f);
        snprintf(cmd, sizeof(cmd),
            "curl -sg --max-time 35 -X POST "
            "-H 'Content-Type: application/json' "
            "-d @%s '%s%s/%s' 2>/dev/null",
            tmp, TELEGRAM_API, g_bot_token, method);
        char *result = curl_exec(cmd, 65536);
        remove(tmp);
        return result;
    } else {
        snprintf(cmd, sizeof(cmd),
            "curl -sg --max-time 35 '%s%s/%s' 2>/dev/null",
            TELEGRAM_API, g_bot_token, method);
        return curl_exec(cmd, 65536);
    }
}

static void tg_send_message(long chat_id, const char *text) {
    // Split long messages at Telegram's 4096-char limit
    size_t total = strlen(text);
    size_t offset = 0;
    while (offset < total) {
        size_t chunk = total - offset;
        if (chunk > MAX_MSG_LEN - 1) {
            // Find last newline within limit for clean split
            chunk = MAX_MSG_LEN - 1;
            for (size_t i = chunk; i > chunk / 2; i--) {
                if (text[offset + i] == '\n') { chunk = i + 1; break; }
            }
        }
        char *part = malloc(chunk + 1);
        memcpy(part, text + offset, chunk);
        part[chunk] = 0;

        char *escaped = json_escape_alloc(part);
        char *body = NULL;
        asprintf(&body, "{\"chat_id\":%ld,\"text\":\"%s\"}", chat_id, escaped);
        char *resp = tg_api("sendMessage", body);
        free(resp); free(body); free(escaped); free(part);
        offset += chunk;
    }
}

static void tg_send_document(long chat_id, const char *filepath, const char *caption) {
    char cmd[4096];
    char *esc_cap = caption ? json_escape_alloc(caption) : NULL;
    if (esc_cap) {
        snprintf(cmd, sizeof(cmd),
            "curl -sg --max-time 60 -F 'chat_id=%ld' "
            "-F 'document=@%s' -F 'caption=%s' "
            "'%s%s/sendDocument' 2>/dev/null",
            chat_id, filepath, caption, TELEGRAM_API, g_bot_token);
        free(esc_cap);
    } else {
        snprintf(cmd, sizeof(cmd),
            "curl -sg --max-time 60 -F 'chat_id=%ld' "
            "-F 'document=@%s' "
            "'%s%s/sendDocument' 2>/dev/null",
            chat_id, filepath, TELEGRAM_API, g_bot_token);
    }
    char *result = curl_exec(cmd, 65536);
    free(result);
}

static void tg_send_typing(long chat_id) {
    char body[128];
    snprintf(body, sizeof(body), "{\"chat_id\":%ld,\"action\":\"typing\"}", chat_id);
    char *resp = tg_api("sendChatAction", body);
    free(resp);
}

// ============================================================================
// Connection loading (adapted from pre.m)
// ============================================================================

static Connection *get_connection(const char *name) {
    for (int i = 0; i < g_connections_count; i++)
        if (strcmp(g_connections[i].name, name) == 0) return &g_connections[i];
    return NULL;
}

static Connection *get_google_connection(const char *account) {
    if (!account || !account[0]) return get_connection("google");
    Connection *c = get_connection(account);
    if (c && c->is_oauth) return c;
    char full[64];
    snprintf(full, sizeof(full), "google_%s", account);
    c = get_connection(full);
    if (c && c->is_oauth) return c;
    return get_connection("google");
}

#define GOOGLE_TOKEN_URL "https://oauth2.googleapis.com/token"

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
    char *resp = curl_exec(cmd, 8192);
    if (!resp) return 0;
    char new_token[2048] = {0}, expires_s[32] = {0};
    json_extract_str(resp, "access_token", new_token, sizeof(new_token));
    json_extract_str(resp, "expires_in", expires_s, sizeof(expires_s));
    free(resp);
    if (new_token[0]) {
        strlcpy(c->access_token, new_token, sizeof(c->access_token));
        int ei = atoi(expires_s);
        if (ei <= 0) ei = 3600;
        c->token_expiry = (long)time(NULL) + ei;
        return 1;
    }
    return 0;
}

static int oauth_ensure_token(Connection *c) {
    if (!c->active || !c->refresh_token[0]) return 0;
    if (c->access_token[0] && c->token_expiry > (long)time(NULL) + 60) return 1;
    return oauth_refresh_token(c);
}

static void load_connections(void) {
    const char *home = getenv("HOME") ?: "/tmp";
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", home, CONNECTIONS_FILE);

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
        g_connections[i].is_oauth = services[i].oauth;
        g_connections_count++;
    }

    FILE *f = fopen(path, "r");
    if (!f) return;
    char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);

    for (int i = 0; i < g_connections_count; i++) {
        Connection *c = &g_connections[i];
        if (c->is_oauth) {
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
            if (json_extract_str(buf, key_field, c->key, sizeof(c->key)))
                if (c->key[0]) c->active = 1;
        }
    }

    // Load additional Google accounts
    const char *scan = buf;
    while ((scan = strstr(scan, "\"google_")) != NULL) {
        scan++;
        const char *end = strchr(scan, '"');
        if (!end) break;
        char field_name[128];
        size_t flen = end - scan;
        if (flen >= sizeof(field_name)) { scan = end; continue; }
        memcpy(field_name, scan, flen);
        field_name[flen] = 0;
        scan = end + 1;
        char *suffix = strstr(field_name, "_refresh_token");
        if (!suffix) continue;
        *suffix = 0;
        if (strcmp(field_name, "google") == 0) continue;
        if (get_connection(field_name)) continue;
        if (g_connections_count >= MAX_CONNECTIONS) break;
        const char *short_label = field_name + 7;
        Connection *c = &g_connections[g_connections_count++];
        memset(c, 0, sizeof(*c));
        strlcpy(c->name, field_name, sizeof(c->name));
        snprintf(c->label, sizeof(c->label), "Google (%s)", short_label);
        c->is_oauth = 1;
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

    // Extract bot token
    Connection *tg = get_connection("telegram");
    if (tg && tg->active)
        strlcpy(g_bot_token, tg->key, sizeof(g_bot_token));
}

// ============================================================================
// Owner persistence
// ============================================================================

static void load_owner(void) {
    const char *home = getenv("HOME") ?: "/tmp";
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/.pre/telegram_owner", home);
    FILE *f = fopen(path, "r");
    if (f) {
        char buf[32];
        if (fgets(buf, sizeof(buf), f))
            g_owner_id = atol(buf);
        fclose(f);
    }
}

static void save_owner(long id) {
    const char *home = getenv("HOME") ?: "/tmp";
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/.pre/telegram_owner", home);
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "%ld\n", id);
        fclose(f);
    }
    g_owner_id = id;
}

// ============================================================================
// Memory system (adapted from pre.m)
// ============================================================================

static void load_memories(void) {
    const char *home = getenv("HOME") ?: "/tmp";
    char dir[PATH_MAX];
    snprintf(dir, sizeof(dir), "%s/.pre/memory", home);
    g_memory_count = 0;

    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && g_memory_count < MAX_MEMORIES) {
        if (ent->d_name[0] == '.') continue;
        size_t nlen = strlen(ent->d_name);
        if (nlen < 4 || strcmp(ent->d_name + nlen - 3, ".md") != 0) continue;
        if (strcmp(ent->d_name, "index.md") == 0) continue;

        char fpath[PATH_MAX];
        snprintf(fpath, sizeof(fpath), "%s/%s", dir, ent->d_name);
        FILE *f = fopen(fpath, "r");
        if (!f) continue;
        char content[4096];
        size_t n = fread(content, 1, sizeof(content) - 1, f);
        content[n] = 0;
        fclose(f);

        MemoryEntry *m = &g_memories[g_memory_count];
        strlcpy(m->file, fpath, sizeof(m->file));
        m->name[0] = m->type[0] = m->description[0] = 0;

        // Parse YAML frontmatter
        if (strncmp(content, "---", 3) == 0) {
            char *end = strstr(content + 3, "---");
            if (end) {
                char *line = content + 3;
                while (line < end) {
                    while (*line == '\n' || *line == '\r') line++;
                    if (line >= end) break;
                    char *nl = strchr(line, '\n');
                    if (!nl || nl > end) nl = end;
                    char linestr[512];
                    size_t ll = nl - line;
                    if (ll >= sizeof(linestr)) ll = sizeof(linestr) - 1;
                    memcpy(linestr, line, ll);
                    linestr[ll] = 0;

                    if (strncmp(linestr, "name:", 5) == 0) {
                        char *v = linestr + 5; while (*v == ' ') v++;
                        strlcpy(m->name, v, sizeof(m->name));
                    } else if (strncmp(linestr, "type:", 5) == 0) {
                        char *v = linestr + 5; while (*v == ' ') v++;
                        strlcpy(m->type, v, sizeof(m->type));
                    } else if (strncmp(linestr, "description:", 12) == 0) {
                        char *v = linestr + 12; while (*v == ' ') v++;
                        strlcpy(m->description, v, sizeof(m->description));
                    }
                    line = nl + 1;
                }
            }
        }
        if (m->name[0]) g_memory_count++;
    }
    closedir(d);
}

static char *build_memory_context(void) {
    if (g_memory_count == 0) return NULL;
    size_t cap = 4096;
    char *buf = malloc(cap);
    int len = snprintf(buf, cap, "Memories:\n");
    for (int i = 0; i < g_memory_count && len < (int)cap - 256; i++) {
        len += snprintf(buf + len, cap - len, "- [%s] %s: %s\n",
                        g_memories[i].type, g_memories[i].name, g_memories[i].description);
    }
    return buf;
}

static int save_memory(const char *name, const char *type, const char *description, const char *content) {
    const char *home = getenv("HOME") ?: "/tmp";
    char dir[PATH_MAX];
    snprintf(dir, sizeof(dir), "%s/.pre/memory", home);
    mkdir(dir, 0755);

    // Generate filename
    char fname[256];
    int fi = 0;
    for (int i = 0; name[i] && fi < (int)sizeof(fname) - 5; i++) {
        char c = name[i];
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')
            fname[fi++] = c;
        else if (c >= 'A' && c <= 'Z')
            fname[fi++] = c + 32;
        else if (c == ' ')
            fname[fi++] = '_';
    }
    fname[fi] = 0;
    if (!fname[0]) strlcpy(fname, "memory", sizeof(fname));

    char fpath[PATH_MAX];
    snprintf(fpath, sizeof(fpath), "%s/%s.md", dir, fname);

    FILE *f = fopen(fpath, "w");
    if (!f) return 0;
    fprintf(f, "---\nname: %s\ndescription: %s\ntype: %s\n---\n\n%s\n",
            name, description && description[0] ? description : name, type, content);
    fclose(f);

    // Update index
    char idx[PATH_MAX];
    snprintf(idx, sizeof(idx), "%s/index.md", dir);
    FILE *fi2 = fopen(idx, "a");
    if (fi2) {
        fprintf(fi2, "- [%s](%s.md) — %s\n", name, fname,
                description && description[0] ? description : name);
        fclose(fi2);
    }
    load_memories(); // reload
    return 1;
}

// ============================================================================
// Conversation management
// ============================================================================

static Conversation *get_conversation(long chat_id) {
    for (int i = 0; i < g_convo_count; i++)
        if (g_convos[i].chat_id == chat_id) return &g_convos[i];
    // Create new
    if (g_convo_count >= MAX_CONVERSATIONS) {
        // Evict oldest
        for (int j = 0; j < g_convos[0].history_count; j++)
            free(g_convos[0].history[j].content);
        memmove(&g_convos[0], &g_convos[1], sizeof(Conversation) * (MAX_CONVERSATIONS - 1));
        g_convo_count--;
    }
    Conversation *c = &g_convos[g_convo_count++];
    memset(c, 0, sizeof(*c));
    c->chat_id = chat_id;
    return c;
}

static void convo_add(Conversation *c, const char *role, const char *content) {
    if (c->history_count >= MAX_HISTORY * 2) {
        // Drop oldest pair
        free(c->history[0].content);
        free(c->history[1].content);
        memmove(&c->history[0], &c->history[2], sizeof(ChatMsg) * (c->history_count - 2));
        c->history_count -= 2;
    }
    ChatMsg *m = &c->history[c->history_count++];
    strlcpy(m->role, role, sizeof(m->role));
    m->content = strdup(content);
}

static void convo_clear(Conversation *c) {
    for (int i = 0; i < c->history_count; i++)
        free(c->history[i].content);
    c->history_count = 0;
}

// ============================================================================
// System prompt builder
// ============================================================================

static char *build_system_prompt(void) {
    size_t cap = 8192;
    char *buf = malloc(cap);
    int len = 0;

    if (strcmp(g_agent_name, "PRE") != 0)
        len += snprintf(buf + len, cap - len,
            "You are %s, a Personal Reasoning Engine (PRE) — a fully capable AI assistant with full system access. "
            "The user is chatting with you via Telegram. "
            "Keep responses concise and mobile-friendly — avoid very long code blocks or tables.\n\n", g_agent_name);
    else
        len += snprintf(buf + len, cap - len,
            "You are PRE (Personal Reasoning Engine), a fully capable AI assistant with full system access. "
            "The user is chatting with you via Telegram. "
            "Keep responses concise and mobile-friendly — avoid very long code blocks or tables.\n\n");

    // Date
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    char datebuf[128];
    const char *wday[] = {"Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"};
    const char *mon[] = {"January","February","March","April","May","June",
                         "July","August","September","October","November","December"};
    snprintf(datebuf, sizeof(datebuf), "%s, %s %d, %d",
             wday[tm->tm_wday], mon[tm->tm_mon], tm->tm_mday, tm->tm_year + 1900);
    len += snprintf(buf + len, cap - len,
        "Today is %s.\n\n", datebuf);

    // Memory context
    char *mem = build_memory_context();
    if (mem) {
        len += snprintf(buf + len, cap - len, "%s\n", mem);
        free(mem);
    }

    // Tool instructions — same as CLI, full access
    len += snprintf(buf + len, cap - len,
        "To use a tool: <tool_call>\n{\"name\":\"TOOL\",\"arguments\":{...}}\n</tool_call>\n"
        "After calling, STOP and wait for <tool_response>.\n\n"
        "CRITICAL — FILE CREATION RULES:\n"
        "You must NEVER output raw code, HTML, scripts, or file contents in your response text. "
        "Instead, ALWAYS use a tool to save content to disk:\n"
        "- For visual content (HTML, markdown, CSV, SVG, diagrams, reports): use artifact(title,content,type). "
        "This saves the file and sends it as a document automatically.\n"
        "- For executable content (scripts, source code, configs): use file_write(path,content).\n"
        "- Describe what you created in your response text. Do NOT include the file contents in chat.\n\n"
        "Tools:\n"
        "bash(command) read_file(path) list_dir(path) glob(pattern,path?) grep(pattern,path?,include?) "
        "file_write(path,content) file_edit(path,old_string,new_string) web_fetch(url) "
        "system_info() process_list(filter?) process_kill(pid) "
        "clipboard_read() clipboard_write(content) open_app(target) notify(title,message) "
        "memory_save(name,type,description,content) memory_search(query?) memory_list() memory_delete(query) "
        "screenshot(region?) window_list() window_focus(app) display_info() "
        "net_info() net_connections(filter?) service_status(service?) disk_usage(path?) "
        "hardware_info() applescript(script) "
        "artifact(title,content,type)\n");

    // Connection-dependent tools
    Connection *brave = get_connection("brave_search");
    Connection *gh = get_connection("github");
    Connection *goog = get_connection("google");
    Connection *wolf = get_connection("wolfram");

    if (brave && brave->active)
        len += snprintf(buf + len, cap - len, "web_search(query,count?) ");
    if (gh && gh->active)
        len += snprintf(buf + len, cap - len,
            "github(action:search_repos|list_issues|read_issue|list_prs|user,repo?,query?,number?,state?) ");
    if (goog && goog->active)
        len += snprintf(buf + len, cap - len,
            "gmail(action:search|read|send|draft|trash|labels|profile,query?,id?,to?,subject?,body?,cc?,bcc?,max_results?,account?) "
            "gdrive(action:list|search|download,id?,path?,name?,folder_id?,query?,count?,account?) "
            "gdocs(action:create|read|append,id?,title?,content?,account?) ");
    if (wolf && wolf->active)
        len += snprintf(buf + len, cap - len, "wolfram(query) ");

    len += snprintf(buf + len, cap - len, "\n");

    len += snprintf(buf + len, cap - len,
        "memory_save types: user|feedback|project|reference.\n"
        "artifact types: html|markdown|csv|json|svg|code|text.\n"
        "Save memories proactively for user prefs and important context.\n");

    return buf;
}

// ============================================================================
// Tool call parser (simplified from pre.m)
// ============================================================================

static int decode_json_string(const char *p, char *buf, size_t bufsz) {
    int di = 0;
    for (int i = 0; p[i] && p[i] != '"' && di < (int)bufsz - 1; i++) {
        if (p[i] == '\\' && p[i+1]) {
            i++;
            if (p[i] == 'u' && p[i+1] && p[i+2] && p[i+3] && p[i+4]) {
                char hex[5] = {p[i+1],p[i+2],p[i+3],p[i+4],0};
                unsigned cp = (unsigned)strtol(hex, NULL, 16);
                i += 4;
                if (cp < 0x80) buf[di++] = (char)cp;
                else if (cp < 0x800) {
                    buf[di++] = (char)(0xC0 | (cp >> 6));
                    if (di < (int)bufsz-1) buf[di++] = (char)(0x80 | (cp & 0x3F));
                } else {
                    buf[di++] = (char)(0xE0 | (cp >> 12));
                    if (di < (int)bufsz-1) buf[di++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    if (di < (int)bufsz-1) buf[di++] = (char)(0x80 | (cp & 0x3F));
                }
            } else {
                switch (p[i]) {
                    case 'n': buf[di++]='\n'; break;
                    case 't': buf[di++]='\t'; break;
                    case '"': buf[di++]='"'; break;
                    case '\\': buf[di++]='\\'; break;
                    default: buf[di++]=p[i]; break;
                }
            }
        } else buf[di++] = p[i];
    }
    buf[di] = 0;
    return di;
}

static int extract_tool_call(const char *text, ToolCall *tc) {
    memset(tc, 0, sizeof(*tc));
    const char *start = strstr(text, "<tool_call>");
    if (!start) start = strstr(text, "```json\n{\"name\"");
    if (!start) start = strstr(text, "{\"name\"");
    if (!start) return 0;

    if (strstr(start, "<tool_call>")) start += 11;
    while (*start == '\n' || *start == ' ' || *start == '`') start++;
    if (*start != '{') return 0;

    // Extract name
    const char *np = strstr(start, "\"name\"");
    if (!np) return 0;
    np += 6;
    while (*np == ' ' || *np == ':' || *np == ' ') np++;
    if (*np == ':') np++;
    while (*np == ' ') np++;
    if (*np != '"') return 0;
    np++;
    int ni = 0;
    while (np[ni] && np[ni] != '"' && ni < 63) { tc->name[ni] = np[ni]; ni++; }
    tc->name[ni] = 0;

    // Extract arguments
    const char *ap = strstr(start, "\"arguments\"");
    if (!ap) ap = strstr(start, "\"args\"");
    if (!ap) { if (tc->name[0]) return 1; return 0; }
    ap = strchr(ap, '{');
    if (!ap) { if (tc->name[0]) return 1; return 0; }
    ap++; // skip {

    while (*ap && *ap != '}' && tc->argc < MAX_TOOL_ARGS) {
        while (*ap == ' ' || *ap == '\n' || *ap == '\r' || *ap == '\t' || *ap == ',') ap++;
        if (*ap == '}' || !*ap) break;
        if (*ap != '"') { ap++; continue; }
        ap++; // skip opening quote of key
        int ki = 0;
        while (ap[ki] && ap[ki] != '"' && ki < 63) { tc->keys[tc->argc][ki] = ap[ki]; ki++; }
        tc->keys[tc->argc][ki] = 0;
        ap += ki;
        if (*ap == '"') ap++;
        while (*ap == ' ' || *ap == ':') ap++;

        if (*ap == '"') {
            ap++;
            char vbuf[65536];
            int vl = decode_json_string(ap, vbuf, sizeof(vbuf));
            tc->vals[tc->argc] = strdup(vbuf);
            // Skip past closing quote
            for (int i = 0; ap[i]; i++) {
                if (ap[i] == '"' && (i == 0 || ap[i-1] != '\\')) { ap += i + 1; break; }
            }
            (void)vl;
        } else {
            // Non-string value
            char vbuf[256];
            int vi = 0;
            while (*ap && *ap != ',' && *ap != '}' && *ap != '\n' && vi < 255)
                vbuf[vi++] = *ap++;
            while (vi > 0 && (vbuf[vi-1] == ' ' || vbuf[vi-1] == '\t')) vi--;
            vbuf[vi] = 0;
            tc->vals[tc->argc] = strdup(vbuf);
        }
        tc->argc++;
    }

    return tc->name[0] ? 1 : 0;
}

// ============================================================================
// Tool implementations — full system access (matches TUI)
// ============================================================================

static void resolve_path(const char *path, char *out, size_t out_sz) {
    if (path[0] == '~') {
        snprintf(out, out_sz, "%s%s", getenv("HOME") ?: "/tmp", path + 1);
    } else if (path[0] != '/') {
        snprintf(out, out_sz, "%s/%s", getenv("HOME") ?: "/tmp", path);
    } else {
        strlcpy(out, path, out_sz);
    }
}

static int execute_tool(const ToolCall *tc, char *output, size_t output_sz) {
    const char *name = tc->name;
    int out_len = 0;

    // ---- Shell / bash ----

    if (strcmp(name, "bash") == 0) {
        const char *cmd = tool_call_get(tc, "command");
        if (!cmd) return snprintf(output, output_sz, "Error: no command provided");
        printf("  [$ %s]\n", cmd);
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

    // ---- File tools ----

    } else if (strcmp(name, "read_file") == 0) {
        const char *path = tool_call_get(tc, "path");
        if (!path) return snprintf(output, output_sz, "Error: path required");
        char resolved[PATH_MAX];
        resolve_path(path, resolved, sizeof(resolved));
        FILE *f = fopen(resolved, "r");
        if (!f) return snprintf(output, output_sz, "Error: cannot read %s", resolved);
        out_len = (int)fread(output, 1, output_sz - 1, f);
        output[out_len] = 0;
        fclose(f);

    } else if (strcmp(name, "file_write") == 0) {
        const char *path_arg = tool_call_get(tc, "path");
        const char *content = tool_call_get(tc, "content");
        if (!path_arg || !content) return snprintf(output, output_sz, "Error: path and content required");
        char resolved[PATH_MAX];
        resolve_path(path_arg, resolved, sizeof(resolved));
        printf("  [writing %s]\n", resolved);
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
        if (!path_arg || !old_str || !new_str)
            return snprintf(output, output_sz, "Error: path, old_string, and new_string required");
        char resolved[PATH_MAX];
        resolve_path(path_arg, resolved, sizeof(resolved));
        printf("  [editing %s]\n", resolved);
        struct stat st;
        if (stat(resolved, &st) < 0) return snprintf(output, output_sz, "Error: file not found '%s'", resolved);
        FILE *f = fopen(resolved, "r");
        if (!f) return snprintf(output, output_sz, "Error: cannot read '%s'", resolved);
        char *file_content = malloc((size_t)st.st_size + 1);
        size_t nread = fread(file_content, 1, (size_t)st.st_size, f);
        file_content[nread] = 0;
        fclose(f);
        char *first = strstr(file_content, old_str);
        if (!first) { free(file_content); return snprintf(output, output_sz, "Error: old_string not found in %s", resolved); }
        char *second = strstr(first + strlen(old_str), old_str);
        if (second) { free(file_content); return snprintf(output, output_sz, "Error: old_string appears multiple times in %s", resolved); }
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
            out_len = snprintf(output, output_sz, "Edited %s: replaced %zu bytes with %zu bytes", resolved, old_len, new_len);
        } else {
            out_len = snprintf(output, output_sz, "Error: cannot write '%s'", resolved);
        }
        free(result);

    } else if (strcmp(name, "list_dir") == 0) {
        const char *path = tool_call_get(tc, "path");
        if (!path) path = ".";
        char resolved[PATH_MAX];
        resolve_path(path, resolved, sizeof(resolved));
        DIR *d = opendir(resolved);
        if (!d) return snprintf(output, output_sz, "Error: cannot open %s", resolved);
        struct dirent *ent;
        while ((ent = readdir(d)) != NULL && out_len < (int)output_sz - 128) {
            if (ent->d_name[0] == '.') continue;
            out_len += snprintf(output + out_len, output_sz - out_len, "%s\n", ent->d_name);
        }
        closedir(d);

    } else if (strcmp(name, "glob") == 0) {
        const char *pattern = tool_call_get(tc, "pattern");
        if (!pattern) return snprintf(output, output_sz, "Error: pattern required");
        glob_t g;
        int r = glob(pattern, GLOB_TILDE | GLOB_BRACE, NULL, &g);
        if (r != 0) return snprintf(output, output_sz, "No matches for: %s", pattern);
        for (size_t i = 0; i < g.gl_pathc && out_len < (int)output_sz - 256; i++)
            out_len += snprintf(output + out_len, output_sz - out_len, "%s\n", g.gl_pathv[i]);
        globfree(&g);

    } else if (strcmp(name, "grep") == 0) {
        const char *pattern = tool_call_get(tc, "pattern");
        const char *path = tool_call_get(tc, "path");
        const char *include = tool_call_get(tc, "include");
        if (!pattern) return snprintf(output, output_sz, "Error: pattern required");
        char cmd[2048];
        int ci = snprintf(cmd, sizeof(cmd), "grep -rn --max-count=30 ");
        if (include) ci += snprintf(cmd + ci, sizeof(cmd) - ci, "--include='%s' ", include);
        ci += snprintf(cmd + ci, sizeof(cmd) - ci, "-- '%s' '%s' 2>/dev/null", pattern, path ?: ".");
        char *result = curl_exec(cmd, output_sz);
        if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

    // ---- System info ----

    } else if (strcmp(name, "system_info") == 0) {
        char cmd[] = "echo '=== System ===' && uname -a && echo && "
                     "echo '=== CPU ===' && sysctl -n machdep.cpu.brand_string && "
                     "echo && echo '=== Memory ===' && vm_stat 2>/dev/null | head -5 && "
                     "echo && echo '=== Disk ===' && df -h / 2>/dev/null | tail -1 && "
                     "echo && echo '=== Uptime ===' && uptime";
        char *result = curl_exec(cmd, output_sz);
        if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

    // ---- Process management ----

    } else if (strcmp(name, "process_list") == 0) {
        const char *filter = tool_call_get(tc, "filter");
        char cmd[512];
        if (filter && filter[0])
            snprintf(cmd, sizeof(cmd), "ps aux | head -1; ps aux | grep '%s' | grep -v grep", filter);
        else
            snprintf(cmd, sizeof(cmd), "ps aux");
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

    } else if (strcmp(name, "process_kill") == 0) {
        const char *pid_str = tool_call_get(tc, "pid");
        if (!pid_str) return snprintf(output, output_sz, "Error: no pid provided");
        for (int i = 0; pid_str[i]; i++) {
            if (pid_str[i] < '0' || pid_str[i] > '9')
                return snprintf(output, output_sz, "Error: invalid pid '%s'", pid_str);
        }
        pid_t pid = (pid_t)atoi(pid_str);
        printf("  [killing pid %d]\n", pid);
        if (kill(pid, SIGTERM) == 0)
            out_len = snprintf(output, output_sz, "Sent SIGTERM to pid %d", pid);
        else
            out_len = snprintf(output, output_sz, "Error killing pid %d: %s", pid, strerror(errno));

    // ---- Clipboard ----

    } else if (strcmp(name, "clipboard_read") == 0) {
        FILE *proc = popen("pbpaste", "r");
        if (proc) {
            while (out_len < (int)output_sz - 1) {
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
        if (!content) return snprintf(output, output_sz, "Error: no content provided");
        FILE *proc = popen("pbcopy", "w");
        if (proc) {
            fwrite(content, 1, strlen(content), proc);
            pclose(proc);
            out_len = snprintf(output, output_sz, "Copied %zu bytes to clipboard", strlen(content));
        } else {
            out_len = snprintf(output, output_sz, "Error: cannot write to clipboard");
        }

    // ---- App / notification ----

    } else if (strcmp(name, "open_app") == 0) {
        const char *target = tool_call_get(tc, "target");
        if (!target) return snprintf(output, output_sz, "Error: no target provided");
        printf("  [opening %s]\n", target);
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
        if (!title || !message) return snprintf(output, output_sz, "Error: title and message required");
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

    // ---- Screenshot / window / display ----

    } else if (strcmp(name, "screenshot") == 0) {
        const char *region = tool_call_get(tc, "region");
        char tmp_path[PATH_MAX];
        snprintf(tmp_path, sizeof(tmp_path), "/tmp/pre_screenshot_%d.png", (int)getpid());
        char cmd[512];
        if (region && strcmp(region, "window") == 0) {
            snprintf(cmd, sizeof(cmd), "screencapture -w '%s' 2>&1", tmp_path);
        } else if (region && strcmp(region, "full") != 0 && region[0] >= '0' && region[0] <= '9') {
            int x = 0, y = 0, w = 0, h = 0;
            sscanf(region, "%d,%d,%d,%d", &x, &y, &w, &h);
            snprintf(cmd, sizeof(cmd), "screencapture -R%d,%d,%d,%d '%s' 2>&1", x, y, w, h, tmp_path);
        } else {
            snprintf(cmd, sizeof(cmd), "screencapture -x '%s' 2>&1", tmp_path);
        }
        int ret = system(cmd);
        struct stat ss;
        if (ret == 0 && stat(tmp_path, &ss) == 0 && ss.st_size > 0) {
            out_len = snprintf(output, output_sz, "Screenshot saved: %s (%lld bytes)", tmp_path, (long long)ss.st_size);
        } else {
            out_len = snprintf(output, output_sz, "Screenshot saved to %s (screencapture may need Screen Recording permission)", tmp_path);
        }

    } else if (strcmp(name, "window_list") == 0) {
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
            while (out_len < (int)output_sz - 1) {
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
        if (!app) return snprintf(output, output_sz, "Error: app name required");
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
        FILE *proc = popen("system_profiler SPDisplaysDataType 2>/dev/null", "r");
        if (proc) {
            while (out_len < (int)output_sz - 1) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }

    // ---- Network / services / hardware ----

    } else if (strcmp(name, "net_info") == 0) {
        FILE *proc = popen(
            "echo '=== Active Interfaces ===' && "
            "ifconfig | grep -E '^[a-z]|inet ' | grep -v '127.0.0.1' && "
            "echo '' && echo '=== Default Route ===' && "
            "route -n get default 2>/dev/null | grep -E 'gateway|interface' && "
            "echo '' && echo '=== DNS ===' && "
            "scutil --dns 2>/dev/null | head -20", "r");
        if (proc) {
            while (out_len < (int)output_sz - 1) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }

    } else if (strcmp(name, "net_connections") == 0) {
        const char *filter = tool_call_get(tc, "filter");
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
            while (out_len < (int)output_sz - 1) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }

    } else if (strcmp(name, "service_status") == 0) {
        const char *svc = tool_call_get(tc, "service");
        char cmd[512];
        if (svc && svc[0]) {
            char safe[256]; int si = 0;
            for (int i = 0; svc[i] && si < 250; i++) {
                if (svc[i] == '\'') { safe[si++] = '\''; safe[si++] = '\\'; safe[si++] = '\''; safe[si++] = '\''; }
                else safe[si++] = svc[i];
            }
            safe[si] = 0;
            snprintf(cmd, sizeof(cmd), "launchctl list 2>/dev/null | grep -i '%s'", safe);
        } else {
            snprintf(cmd, sizeof(cmd),
                "echo '=== User Services ===' && launchctl list 2>/dev/null | head -30 && "
                "echo '' && echo '=== System Services (running) ===' && "
                "sudo launchctl list 2>/dev/null | grep -v '\"0\"' | head -30 || "
                "launchctl list 2>/dev/null | head -40");
        }
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

    } else if (strcmp(name, "disk_usage") == 0) {
        const char *path_arg = tool_call_get(tc, "path");
        char cmd[512];
        if (path_arg && path_arg[0]) {
            char resolved[PATH_MAX];
            resolve_path(path_arg, resolved, sizeof(resolved));
            snprintf(cmd, sizeof(cmd), "du -sh '%s'/* 2>/dev/null | sort -rh | head -30", resolved);
        } else {
            snprintf(cmd, sizeof(cmd),
                "echo '=== Volumes ===' && df -h 2>/dev/null && "
                "echo '' && echo '=== Largest in Home ===' && "
                "du -sh '%s'/* 2>/dev/null | sort -rh | head -20", getenv("HOME") ?: "/tmp");
        }
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

    } else if (strcmp(name, "hardware_info") == 0) {
        FILE *proc = popen(
            "echo '=== Hardware ===' && "
            "sysctl -n hw.model 2>/dev/null && "
            "sysctl -n machdep.cpu.brand_string 2>/dev/null && "
            "echo \"Cores: $(sysctl -n hw.ncpu 2>/dev/null) ($(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo '?')P + $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo '?')E)\" && "
            "echo \"Memory: $(( $(sysctl -n hw.memsize 2>/dev/null) / 1073741824 )) GB\" && "
            "echo \"GPU Cores: $(system_profiler SPDisplaysDataType 2>/dev/null | grep 'Total Number of Cores' | awk -F: '{print $2}' | xargs)\" && "
            "echo '' && echo '=== Thermal ===' && "
            "pmset -g therm 2>/dev/null && "
            "echo '' && echo '=== Battery ===' && "
            "pmset -g batt 2>/dev/null", "r");
        if (proc) {
            while (out_len < (int)output_sz - 1) {
                int ch = fgetc(proc);
                if (ch == EOF) break;
                output[out_len++] = (char)ch;
            }
            output[out_len] = 0;
            pclose(proc);
        }

    } else if (strcmp(name, "applescript") == 0) {
        const char *script = tool_call_get(tc, "script");
        if (!script) return snprintf(output, output_sz, "Error: script required");
        printf("  [running AppleScript]\n");
        char tmp[PATH_MAX];
        snprintf(tmp, sizeof(tmp), "/tmp/pre_applescript_%d.scpt", (int)getpid());
        FILE *sf = fopen(tmp, "w");
        if (!sf) return snprintf(output, output_sz, "Error: cannot create temp script");
        fputs(script, sf);
        fclose(sf);
        char cmd[PATH_MAX + 64];
        snprintf(cmd, sizeof(cmd), "osascript '%s' 2>&1", tmp);
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
        remove(tmp);
        if (out_len == 0)
            out_len = snprintf(output, output_sz, "AppleScript executed (no output)");

    // ---- Web tools ----

    } else if (strcmp(name, "web_fetch") == 0) {
        const char *url = tool_call_get(tc, "url");
        if (!url) return snprintf(output, output_sz, "Error: url required");
        // Validate URL
        if (strncmp(url, "http://", 7) != 0 && strncmp(url, "https://", 8) != 0)
            return snprintf(output, output_sz, "Error: URL must start with http:// or https://");
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "curl -sL --max-time 15 '%s' 2>/dev/null | "
            "textutil -stdin -format html -convert txt -stdout 2>/dev/null || "
            "curl -sL --max-time 15 '%s' 2>/dev/null", url, url);
        char *result = curl_exec(cmd, output_sz);
        if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

    } else if (strcmp(name, "web_search") == 0) {
        Connection *brave = get_connection("brave_search");
        if (!brave || !brave->active)
            return snprintf(output, output_sz, "Error: Brave Search not configured");
        const char *query = tool_call_get(tc, "query");
        if (!query) return snprintf(output, output_sz, "Error: query required");
        const char *count_s = tool_call_get(tc, "count");
        int n = count_s ? atoi(count_s) : 5;
        if (n < 1) n = 1; if (n > 20) n = 20;
        // URL-encode query
        char encoded[512] = {0};
        int ei = 0;
        for (int qi = 0; query[qi] && ei < (int)sizeof(encoded) - 4; qi++) {
            if (query[qi] == ' ') { encoded[ei++] = '+'; }
            else if (query[qi] == '&') { encoded[ei++] = '%'; encoded[ei++] = '2'; encoded[ei++] = '6'; }
            else encoded[ei++] = query[qi];
        }
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "curl -s --max-time 10 "
            "-H 'Accept: application/json' "
            "-H 'X-Subscription-Token: %s' "
            "'https://api.search.brave.com/res/v1/web/search?q=%s&count=%d' 2>/dev/null",
            brave->key, encoded, n);
        char *resp = curl_exec(cmd, 32768);
        if (!resp) return snprintf(output, output_sz, "Error: search request failed");
        // Parse results
        const char *scan = resp;
        int rn = 0;
        while ((scan = (char *)json_find_key(scan, "title")) != NULL && rn < n) {
            char title[256] = {0}, url_r[512] = {0}, desc[512] = {0};
            json_extract_str(scan, "title", title, sizeof(title));
            json_extract_str(scan, "url", url_r, sizeof(url_r));
            json_extract_str(scan, "description", desc, sizeof(desc));
            if (title[0] && out_len < (int)output_sz - 512) {
                out_len += snprintf(output + out_len, output_sz - out_len,
                    "%d. %s\n   %s\n   %s\n\n", ++rn, title, url_r, desc);
            }
            scan++;
        }
        free(resp);
        if (rn == 0) out_len = snprintf(output, output_sz, "No results found.");

    // ---- Wolfram ----

    } else if (strcmp(name, "wolfram") == 0) {
        Connection *wolf = get_connection("wolfram");
        if (!wolf || !wolf->active)
            return snprintf(output, output_sz, "Error: Wolfram Alpha not configured");
        const char *query = tool_call_get(tc, "query");
        if (!query) return snprintf(output, output_sz, "Error: query required");
        char encoded[512] = {0};
        int ei = 0;
        for (int qi = 0; query[qi] && ei < (int)sizeof(encoded) - 4; qi++) {
            if (query[qi] == ' ') { encoded[ei++] = '%'; encoded[ei++] = '2'; encoded[ei++] = '0'; }
            else encoded[ei++] = query[qi];
        }
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "curl -s --max-time 15 'https://api.wolframalpha.com/v1/result?appid=%s&i=%s' 2>/dev/null",
            wolf->key, encoded);
        char *result = curl_exec(cmd, output_sz);
        if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

    // ---- GitHub ----

    } else if (strcmp(name, "github") == 0) {
        Connection *gh = get_connection("github");
        if (!gh || !gh->active)
            return snprintf(output, output_sz, "Error: GitHub not configured");
        const char *action = tool_call_get(tc, "action");
        if (!action) return snprintf(output, output_sz, "Error: action required");
        char cmd[4096];

        if (strcmp(action, "user") == 0) {
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -H 'Authorization: token %s' "
                "'https://api.github.com/user' 2>/dev/null", gh->key);
        } else if (strcmp(action, "search_repos") == 0) {
            const char *query = tool_call_get(tc, "query");
            if (!query) return snprintf(output, output_sz, "Error: query required");
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -H 'Authorization: token %s' "
                "'https://api.github.com/search/repositories?q=%s&per_page=5' 2>/dev/null",
                gh->key, query);
        } else if (strcmp(action, "list_issues") == 0) {
            const char *repo = tool_call_get(tc, "repo");
            const char *state = tool_call_get(tc, "state");
            if (!repo) return snprintf(output, output_sz, "Error: repo required");
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -H 'Authorization: token %s' "
                "'https://api.github.com/repos/%s/issues?state=%s&per_page=10' 2>/dev/null",
                gh->key, repo, state ?: "open");
        } else if (strcmp(action, "read_issue") == 0) {
            const char *repo = tool_call_get(tc, "repo");
            const char *number = tool_call_get(tc, "number");
            if (!repo || !number) return snprintf(output, output_sz, "Error: repo and number required");
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -H 'Authorization: token %s' "
                "'https://api.github.com/repos/%s/issues/%s' 2>/dev/null",
                gh->key, repo, number);
        } else if (strcmp(action, "list_prs") == 0) {
            const char *repo = tool_call_get(tc, "repo");
            const char *state = tool_call_get(tc, "state");
            if (!repo) return snprintf(output, output_sz, "Error: repo required");
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -H 'Authorization: token %s' "
                "'https://api.github.com/repos/%s/pulls?state=%s&per_page=10' 2>/dev/null",
                gh->key, repo, state ?: "open");
        } else {
            return snprintf(output, output_sz, "Error: unknown action '%s'", action);
        }
        char *result = curl_exec(cmd, output_sz);
        if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

    // ---- Gmail ----

    } else if (strcmp(name, "gmail") == 0) {
        const char *action = tool_call_get(tc, "action");
        const char *account = tool_call_get(tc, "account");
        Connection *c = get_google_connection(account);
        if (!c || !c->active) return snprintf(output, output_sz, "Error: Google not configured");
        if (!oauth_ensure_token(c)) return snprintf(output, output_sz, "Error: Google token expired");
        if (!action) return snprintf(output, output_sz, "Error: action required");
        char cmd[4096];

        if (strcmp(action, "profile") == 0) {
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -H 'Authorization: Bearer %s' "
                "'https://gmail.googleapis.com/gmail/v1/users/me/profile' 2>/dev/null",
                c->access_token);
            char *result = curl_exec(cmd, output_sz);
            if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

        } else if (strcmp(action, "search") == 0) {
            const char *query = tool_call_get(tc, "query");
            if (!query) return snprintf(output, output_sz, "Error: query required");
            const char *mr_s = tool_call_get(tc, "max_results");
            int mr = mr_s ? atoi(mr_s) : 10;
            if (mr < 1) mr = 1; if (mr > 20) mr = 20;
            char encoded[512] = {0};
            int ei = 0;
            for (int qi = 0; query[qi] && ei < (int)sizeof(encoded) - 4; qi++) {
                if (query[qi] == ' ') { encoded[ei++] = '%'; encoded[ei++] = '2'; encoded[ei++] = '0'; }
                else encoded[ei++] = query[qi];
            }
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -H 'Authorization: Bearer %s' "
                "'https://gmail.googleapis.com/gmail/v1/users/me/messages?q=%s&maxResults=%d' 2>/dev/null",
                c->access_token, encoded, mr);
            char *result = curl_exec(cmd, 32768);
            if (!result) return snprintf(output, output_sz, "Error: Gmail request failed");

            // Parse message IDs and fetch metadata
            const char *scan = result;
            int msg_num = 0;
            while ((scan = (char *)json_find_key(scan, "id")) != NULL && msg_num < mr) {
                char mid[64] = {0};
                json_extract_str(scan, "id", mid, sizeof(mid));
                if (!mid[0]) { scan++; continue; }
                // Fetch metadata
                char mcmd[1024];
                snprintf(mcmd, sizeof(mcmd),
                    "curl -s --max-time 5 -H 'Authorization: Bearer %s' "
                    "'https://gmail.googleapis.com/gmail/v1/users/me/messages/%s"
                    "?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date' 2>/dev/null",
                    c->access_token, mid);
                char *mresp = curl_exec(mcmd, 8192);
                if (mresp) {
                    char from[256] = {0}, subj[256] = {0}, date[64] = {0}, snippet[256] = {0};
                    // Parse headers
                    const char *hp = mresp;
                    while ((hp = (char *)json_find_key(hp, "name")) != NULL) {
                        char hname[64] = {0}, hval[256] = {0};
                        json_extract_str(hp, "name", hname, sizeof(hname));
                        json_extract_str(hp, "value", hval, sizeof(hval));
                        if (strcasecmp(hname, "From") == 0) strlcpy(from, hval, sizeof(from));
                        else if (strcasecmp(hname, "Subject") == 0) strlcpy(subj, hval, sizeof(subj));
                        else if (strcasecmp(hname, "Date") == 0) strlcpy(date, hval, sizeof(date));
                        hp++;
                    }
                    json_extract_str(mresp, "snippet", snippet, sizeof(snippet));
                    if (out_len < (int)output_sz - 512) {
                        out_len += snprintf(output + out_len, output_sz - out_len,
                            "%d. [%s] %s\n   From: %s | Date: %s\n   %s\n\n",
                            ++msg_num, mid, subj[0] ? subj : "(no subject)", from, date, snippet);
                    }
                    free(mresp);
                }
                scan++;
            }
            free(result);
            if (msg_num == 0) out_len = snprintf(output, output_sz, "No messages found.");

        } else if (strcmp(action, "read") == 0) {
            const char *id = tool_call_get(tc, "id");
            if (!id) return snprintf(output, output_sz, "Error: id required");
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -H 'Authorization: Bearer %s' "
                "'https://gmail.googleapis.com/gmail/v1/users/me/messages/%s?format=full' 2>/dev/null",
                c->access_token, id);
            char *result = curl_exec(cmd, output_sz);
            if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

        } else if (strcmp(action, "send") == 0) {
            const char *to = tool_call_get(tc, "to");
            const char *subject = tool_call_get(tc, "subject");
            const char *body = tool_call_get(tc, "body");
            if (!to || !subject || !body) return snprintf(output, output_sz, "Error: to, subject, body required");
            // Build RFC 2822 message and base64 encode
            char raw[8192];
            int rl = snprintf(raw, sizeof(raw), "To: %s\r\nSubject: %s\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n%s",
                              to, subject, body);
            // Base64 encode via command
            char tmp[64];
            snprintf(tmp, sizeof(tmp), "/tmp/pre_tg_mail_%d.txt", getpid());
            FILE *tf = fopen(tmp, "w");
            if (tf) { fwrite(raw, 1, rl, tf); fclose(tf); }
            char b64cmd[256];
            snprintf(b64cmd, sizeof(b64cmd), "base64 -i '%s' | tr '+/' '-_' | tr -d '='", tmp);
            char *b64 = curl_exec(b64cmd, 16384);
            remove(tmp);
            if (!b64) return snprintf(output, output_sz, "Error: encoding failed");
            // Strip newlines from base64
            for (char *p2 = b64; *p2; p2++) if (*p2 == '\n') *p2 = 0;
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -X POST "
                "-H 'Authorization: Bearer %s' "
                "-H 'Content-Type: application/json' "
                "-d '{\"raw\":\"%s\"}' "
                "'https://gmail.googleapis.com/gmail/v1/users/me/messages/send' 2>/dev/null",
                c->access_token, b64);
            free(b64);
            char *result = curl_exec(cmd, output_sz);
            if (result) {
                char mid[64] = {0};
                json_extract_str(result, "id", mid, sizeof(mid));
                if (mid[0]) out_len = snprintf(output, output_sz, "Email sent (id: %s)", mid);
                else out_len = snprintf(output, output_sz, "Send result: %.500s", result);
                free(result);
            }

        } else if (strcmp(action, "labels") == 0) {
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -H 'Authorization: Bearer %s' "
                "'https://gmail.googleapis.com/gmail/v1/users/me/labels' 2>/dev/null",
                c->access_token);
            char *result = curl_exec(cmd, output_sz);
            if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

        } else {
            out_len = snprintf(output, output_sz, "Error: unknown gmail action '%s'", action);
        }

    // ---- Google Drive ----

    } else if (strcmp(name, "gdrive") == 0) {
        const char *action = tool_call_get(tc, "action");
        const char *account = tool_call_get(tc, "account");
        Connection *c = get_google_connection(account);
        if (!c || !c->active) return snprintf(output, output_sz, "Error: Google not configured");
        if (!oauth_ensure_token(c)) return snprintf(output, output_sz, "Error: Google token expired");
        if (!action) return snprintf(output, output_sz, "Error: action required");
        char cmd[4096];

        if (strcmp(action, "list") == 0 || strcmp(action, "search") == 0) {
            const char *query = tool_call_get(tc, "query");
            const char *count_s = tool_call_get(tc, "count");
            int n = count_s ? atoi(count_s) : 20;
            if (n < 1) n = 1; if (n > 100) n = 100;
            if (query) {
                char encoded[512] = {0};
                int ei = 0;
                for (int qi = 0; query[qi] && ei < (int)sizeof(encoded) - 4; qi++) {
                    if (query[qi] == ' ') { encoded[ei++] = '%'; encoded[ei++] = '2'; encoded[ei++] = '0'; }
                    else if (query[qi] == '\'') { encoded[ei++] = '%'; encoded[ei++] = '2'; encoded[ei++] = '7'; }
                    else encoded[ei++] = query[qi];
                }
                snprintf(cmd, sizeof(cmd),
                    "curl -s --max-time 10 -H 'Authorization: Bearer %s' "
                    "'https://www.googleapis.com/drive/v3/files?q=name%%20contains%%20%%27%s%%27"
                    "&fields=files(id,name,mimeType,modifiedTime)&pageSize=%d' 2>/dev/null",
                    c->access_token, encoded, n);
            } else {
                snprintf(cmd, sizeof(cmd),
                    "curl -s --max-time 10 -H 'Authorization: Bearer %s' "
                    "'https://www.googleapis.com/drive/v3/files"
                    "?fields=files(id,name,mimeType,modifiedTime)&pageSize=%d' 2>/dev/null",
                    c->access_token, n);
            }
            char *result = curl_exec(cmd, output_sz);
            if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

        } else if (strcmp(action, "upload") == 0) {
            const char *src = tool_call_get(tc, "path");
            const char *fname = tool_call_get(tc, "name");
            const char *folder_id = tool_call_get(tc, "folder_id");
            if (!src) return snprintf(output, output_sz, "Error: path required (local file)");
            char resolved[PATH_MAX];
            resolve_path(src, resolved, sizeof(resolved));
            if (!fname) fname = strrchr(resolved, '/') ? strrchr(resolved, '/') + 1 : resolved;
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
            char *result = curl_exec(cmd, output_sz);
            if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

        } else if (strcmp(action, "mkdir") == 0) {
            const char *fname = tool_call_get(tc, "name");
            const char *parent = tool_call_get(tc, "folder_id");
            if (!fname) return snprintf(output, output_sz, "Error: name required");
            char tmp[256];
            snprintf(tmp, sizeof(tmp), "/tmp/pre_tg_gdrive_%d.json", getpid());
            FILE *tf = fopen(tmp, "w");
            if (!tf) return snprintf(output, output_sz, "Error: temp file failed");
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
            char *result = curl_exec(cmd, output_sz);
            if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

        } else if (strcmp(action, "share") == 0) {
            const char *file_id = tool_call_get(tc, "id");
            const char *email = tool_call_get(tc, "email");
            const char *role = tool_call_get(tc, "role");
            if (!file_id || !email) return snprintf(output, output_sz, "Error: id and email required");
            if (!role) role = "reader";
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -X POST "
                "-H 'Authorization: Bearer %s' "
                "-H 'Content-Type: application/json' "
                "-d '{\"role\":\"%s\",\"type\":\"user\",\"emailAddress\":\"%s\"}' "
                "'https://www.googleapis.com/drive/v3/files/%s/permissions' 2>/dev/null",
                c->access_token, role, email, file_id);
            char *result = curl_exec(cmd, output_sz);
            if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

        } else if (strcmp(action, "delete") == 0) {
            const char *file_id = tool_call_get(tc, "id");
            if (!file_id) return snprintf(output, output_sz, "Error: id required");
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -X DELETE "
                "-H 'Authorization: Bearer %s' "
                "'https://www.googleapis.com/drive/v3/files/%s' -w '%%{http_code}' 2>/dev/null",
                c->access_token, file_id);
            char *result = curl_exec(cmd, output_sz);
            if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

        } else {
            out_len = snprintf(output, output_sz,
                "Error: unknown action '%s'. Available: list, search, upload, mkdir, share, delete", action);
        }

    // ---- Google Docs ----

    } else if (strcmp(name, "gdocs") == 0) {
        const char *action = tool_call_get(tc, "action");
        const char *account = tool_call_get(tc, "account");
        Connection *c = get_google_connection(account);
        if (!c || !c->active) return snprintf(output, output_sz, "Error: Google not configured");
        if (!oauth_ensure_token(c)) return snprintf(output, output_sz, "Error: Google token expired");
        if (!action) return snprintf(output, output_sz, "Error: action required");

        if (strcmp(action, "read") == 0) {
            const char *id = tool_call_get(tc, "id");
            if (!id) return snprintf(output, output_sz, "Error: id required");
            char cmd[2048];
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -H 'Authorization: Bearer %s' "
                "'https://docs.googleapis.com/v1/documents/%s' 2>/dev/null",
                c->access_token, id);
            char *result = curl_exec(cmd, output_sz);
            if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

        } else if (strcmp(action, "create") == 0) {
            const char *title = tool_call_get(tc, "title");
            const char *content = tool_call_get(tc, "content");
            if (!title) return snprintf(output, output_sz, "Error: title required");
            char tmp[128];
            snprintf(tmp, sizeof(tmp), "/tmp/pre_tg_gdocs_%d.json", getpid());
            FILE *tf = fopen(tmp, "w");
            if (!tf) return snprintf(output, output_sz, "Error: temp file failed");
            fprintf(tf, "{\"title\":\"%s\"}", title);
            fclose(tf);
            char cmd[4096];
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -X POST "
                "-H 'Authorization: Bearer %s' "
                "-H 'Content-Type: application/json' "
                "-d @%s 'https://docs.googleapis.com/v1/documents' 2>/dev/null",
                c->access_token, tmp);
            char *create_resp = curl_exec(cmd, 8192);
            remove(tmp);
            if (!create_resp) return snprintf(output, output_sz, "Error: create request failed");
            char doc_id[128] = {0};
            json_extract_str(create_resp, "documentId", doc_id, sizeof(doc_id));
            // Insert content if provided
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
                        system(cmd);
                        remove(tmp);
                    }
                    free(esc_content);
                }
            }
            strlcpy(output, create_resp, output_sz);
            out_len = (int)strlen(output);
            free(create_resp);

        } else if (strcmp(action, "append") == 0) {
            const char *doc_id = tool_call_get(tc, "id");
            const char *content = tool_call_get(tc, "content");
            if (!doc_id || !content) return snprintf(output, output_sz, "Error: id and content required");
            char *esc_content = json_escape_alloc(content);
            if (!esc_content) return snprintf(output, output_sz, "Error: encoding failed");
            char tmp[128];
            snprintf(tmp, sizeof(tmp), "/tmp/pre_tg_gdocs_%d.json", getpid());
            FILE *tf = fopen(tmp, "w");
            if (!tf) { free(esc_content); return snprintf(output, output_sz, "Error: temp file failed"); }
            fprintf(tf, "{\"requests\":[{\"insertText\":{\"endOfSegmentLocation\":{\"segmentId\":\"\"},\"text\":\"%s\"}}]}", esc_content);
            fclose(tf);
            free(esc_content);
            char cmd[4096];
            snprintf(cmd, sizeof(cmd),
                "curl -s --max-time 10 -X POST "
                "-H 'Authorization: Bearer %s' "
                "-H 'Content-Type: application/json' "
                "-d @%s 'https://docs.googleapis.com/v1/documents/%s:batchUpdate' 2>/dev/null",
                c->access_token, tmp, doc_id);
            char *result = curl_exec(cmd, output_sz);
            remove(tmp);
            if (result) { strlcpy(output, result, output_sz); out_len = (int)strlen(output); free(result); }

        } else {
            out_len = snprintf(output, output_sz,
                "Error: unknown action '%s'. Available: create, read, append", action);
        }

    // ---- Memory tools ----

    } else if (strcmp(name, "memory_save") == 0) {
        const char *mname = tool_call_get(tc, "name");
        const char *mtype = tool_call_get(tc, "type");
        const char *mdesc = tool_call_get(tc, "description");
        const char *mcont = tool_call_get(tc, "content");
        if (!mname || !mtype || !mcont)
            return snprintf(output, output_sz, "Error: name, type, content required");
        if (save_memory(mname, mtype, mdesc, mcont))
            out_len = snprintf(output, output_sz, "Memory saved: %s [%s]", mname, mtype);
        else
            out_len = snprintf(output, output_sz, "Error: failed to save memory");

    } else if (strcmp(name, "memory_list") == 0) {
        load_memories();
        if (g_memory_count == 0) return snprintf(output, output_sz, "No memories saved.");
        for (int i = 0; i < g_memory_count && out_len < (int)output_sz - 256; i++)
            out_len += snprintf(output + out_len, output_sz - out_len,
                "%d. [%s] %s — %s\n", i+1, g_memories[i].type, g_memories[i].name, g_memories[i].description);

    } else if (strcmp(name, "memory_search") == 0) {
        load_memories();
        const char *query = tool_call_get(tc, "query");
        if (g_memory_count == 0) return snprintf(output, output_sz, "No memories saved.");
        int found = 0;
        for (int i = 0; i < g_memory_count && out_len < (int)output_sz - 512; i++) {
            if (query && query[0]) {
                if (!strcasestr(g_memories[i].name, query) &&
                    !strcasestr(g_memories[i].type, query) &&
                    !strcasestr(g_memories[i].description, query))
                    continue;
            }
            out_len += snprintf(output + out_len, output_sz - out_len,
                "%d. [%s] %s — %s\n", ++found, g_memories[i].type, g_memories[i].name, g_memories[i].description);
        }
        if (found == 0) out_len = snprintf(output, output_sz, "No memories matching '%s'", query ?: "");

    } else if (strcmp(name, "memory_delete") == 0) {
        const char *query = tool_call_get(tc, "query");
        if (!query) return snprintf(output, output_sz, "Error: query required");
        load_memories();
        for (int i = 0; i < g_memory_count; i++) {
            if (strcasestr(g_memories[i].name, query) || strcasestr(g_memories[i].description, query)) {
                remove(g_memories[i].file);
                out_len = snprintf(output, output_sz, "Deleted memory: %s", g_memories[i].name);
                load_memories();
                return out_len;
            }
        }
        out_len = snprintf(output, output_sz, "No memory found matching '%s'", query);

    } else if (strcmp(name, "artifact") == 0) {
        const char *atitle = tool_call_get(tc, "title");
        const char *acontent = tool_call_get(tc, "content");
        const char *atype = tool_call_get(tc, "type");
        if (!acontent) return snprintf(output, output_sz, "Error: content required");
        // Auto-detect type from content if not provided
        if (!atype || !atype[0]) {
            if (acontent && (strstr(acontent, "<html") || strstr(acontent, "<!DOCTYPE") ||
                strstr(acontent, "<div") || strstr(acontent, "<script")))
                atype = "html";
            else
                atype = "html";
        }
        // Default title
        static char auto_title[64];
        if (!atitle || !atitle[0]) {
            snprintf(auto_title, sizeof(auto_title), "artifact_%ld", (long)time(NULL));
            atitle = auto_title;
        }

        const char *home = getenv("HOME");
        if (!home) return snprintf(output, output_sz, "Error: HOME not set");

        // Create date directory
        time_t now = time(NULL);
        struct tm *tm_now = localtime(&now);
        char date_dir[PATH_MAX];
        snprintf(date_dir, sizeof(date_dir), "%s/.pre/artifacts/%04d-%02d-%02d",
                 home, tm_now->tm_year + 1900, tm_now->tm_mon + 1, tm_now->tm_mday);
        mkdir(date_dir, 0755);

        // Sanitize title
        char safe_title[128];
        int si = 0;
        for (int i = 0; atitle[i] && si < (int)sizeof(safe_title) - 1; i++) {
            char c = atitle[i];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '-' || c == '_')
                safe_title[si++] = c;
            else if (c == ' ')
                safe_title[si++] = '_';
        }
        safe_title[si] = 0;
        if (!safe_title[0]) strlcpy(safe_title, "artifact", sizeof(safe_title));

        // Determine extension
        const char *ext = "txt";
        if (strcmp(atype, "html") == 0) ext = "html";
        else if (strcmp(atype, "markdown") == 0 || strcmp(atype, "md") == 0) ext = "md";
        else if (strcmp(atype, "csv") == 0) ext = "csv";
        else if (strcmp(atype, "json") == 0) ext = "json";
        else if (strcmp(atype, "svg") == 0) ext = "svg";

        // Write file
        char filepath[PATH_MAX];
        snprintf(filepath, sizeof(filepath), "%s/%s.%s", date_dir, safe_title, ext);
        FILE *af = fopen(filepath, "w");
        if (!af) return snprintf(output, output_sz, "Error: cannot create %s", filepath);
        fputs(acontent, af);
        fclose(af);

        printf("  Artifact: %s (%s)\n", atitle, filepath);

        // Send as document to Telegram chat
        if (g_active_chat_id > 0) {
            char caption[256];
            snprintf(caption, sizeof(caption), "📎 %s [%s]", atitle, atype);
            tg_send_document(g_active_chat_id, filepath, caption);
        }

        out_len = snprintf(output, output_sz, "Artifact created and sent: %s (%s)", atitle, filepath);

    } else {
        out_len = snprintf(output, output_sz, "Error: unknown tool '%s'", name);
    }

    if (out_len == 0) out_len = snprintf(output, output_sz, "(no output)");
    return out_len;
}

// ============================================================================
// Ollama chat (non-streaming with typing indicator)
// ============================================================================

static char *ollama_chat(Conversation *convo, const char *user_message, long chat_id) {
    // Build request body
    size_t body_cap = 512 * 1024;
    char *body = malloc(body_cap);
    int blen = snprintf(body, body_cap,
        "{\"model\":\"%s\",\"stream\":false,\"keep_alive\":\"24h\"", g_model);

    // Dynamic context: estimate tokens used + headroom. Keep small to avoid
    // KV cache rebuilds (which cause massive TTFT spikes). Must stay at or below
    // whatever num_ctx the TUI is using to avoid thrashing.
    int est_tokens = 0;
    // System prompt ~500 tokens, history ~4 chars/token
    est_tokens += 600; // system prompt estimate
    for (int i = 0; i < convo->history_count; i++)
        est_tokens += (int)(strlen(convo->history[i].content) / 4) + 10;
    est_tokens += 2048; // headroom
    int num_ctx = 8192;
    while (num_ctx < est_tokens && num_ctx < 131072) num_ctx *= 2;

    blen += snprintf(body + blen, body_cap - blen,
        ",\"options\":{\"num_predict\":4096,\"num_ctx\":%d}"
        ",\"messages\":[", num_ctx);

    // System prompt
    char *sys = build_system_prompt();
    if (sys) {
        char *esc = json_escape_alloc(sys);
        blen += snprintf(body + blen, body_cap - blen,
            "{\"role\":\"system\",\"content\":\"%s\"}", esc);
        free(esc); free(sys);
    }

    // History
    for (int i = 0; i < convo->history_count; i++) {
        char *esc = json_escape_alloc(convo->history[i].content);
        blen += snprintf(body + blen, body_cap - blen,
            ",{\"role\":\"%s\",\"content\":\"%s\"}", convo->history[i].role, esc);
        free(esc);
    }

    // Current user message
    char *esc = json_escape_alloc(user_message);
    blen += snprintf(body + blen, body_cap - blen,
        ",{\"role\":\"user\",\"content\":\"%s\"}", esc);
    free(esc);

    blen += snprintf(body + blen, body_cap - blen, "]}");

    // Fork a child to send typing indicators
    pid_t typing_pid = fork();
    if (typing_pid == 0) {
        // Child: send typing every 4 seconds until killed
        signal(SIGTERM, SIG_DFL);
        while (1) {
            tg_send_typing(chat_id);
            sleep(4);
        }
        _exit(0);
    }

    // Send to Ollama
    char tmp[64];
    snprintf(tmp, sizeof(tmp), "/tmp/pre_tg_ollama_%d.json", getpid());
    FILE *tf = fopen(tmp, "w");
    if (tf) { fwrite(body, 1, blen, tf); fclose(tf); }
    free(body);

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "curl -s --max-time 300 -X POST "
        "-H 'Content-Type: application/json' "
        "-d @%s 'http://127.0.0.1:%d/api/chat' 2>/dev/null",
        tmp, g_port);
    char *resp = curl_exec(cmd, MAX_RESPONSE);
    remove(tmp);

    // Kill typing indicator
    if (typing_pid > 0) {
        kill(typing_pid, SIGTERM);
        waitpid(typing_pid, NULL, 0);
    }

    if (!resp) return strdup("Error: no response from model");

    // Extract assistant message content
    char *content = malloc(MAX_RESPONSE);
    content[0] = 0;
    json_extract_str(resp, "content", content, MAX_RESPONSE);
    free(resp);

    return content;
}

// ============================================================================
// Agentic message handler — tool loop
// ============================================================================

static void handle_message(long chat_id, long user_id, const char *text) {
    // Authorization check
    if (g_owner_id == 0) {
        save_owner(user_id);
        printf("  Owner set: %ld\n", user_id);
    } else if (user_id != g_owner_id) {
        tg_send_message(chat_id, "This bot is private.");
        return;
    }

    Conversation *convo = get_conversation(chat_id);

    // Bot commands
    if (text[0] == '/') {
        if (strcmp(text, "/start") == 0) {
            tg_send_message(chat_id,
                "Welcome to PRE (Personal Reasoning Engine).\n\n"
                "I'm your local AI assistant, running on your Mac via Ollama.\n\n"
                "Commands:\n"
                "/new - New conversation\n"
                "/status - Bot status\n"
                "/memory - List memories\n"
                "/help - Show this message");
            return;
        }
        if (strcmp(text, "/new") == 0) {
            convo_clear(convo);
            tg_send_message(chat_id, "New conversation started.");
            return;
        }
        if (strcmp(text, "/status") == 0) {
            load_memories();
            char status[1024];
            snprintf(status, sizeof(status),
                "PRE Telegram Bot\n"
                "Model: %s\n"
                "Ollama: localhost:%d\n"
                "Memories: %d\n"
                "History: %d messages\n\n"
                "Connections:\n",
                g_model, g_port, g_memory_count, convo->history_count);
            int sl = (int)strlen(status);
            for (int i = 0; i < g_connections_count; i++) {
                Connection *c = &g_connections[i];
                if (strcmp(c->name, "telegram") == 0) continue;
                sl += snprintf(status + sl, sizeof(status) - sl,
                    "%s %s\n", c->active ? "●" : "○", c->label);
            }
            tg_send_message(chat_id, status);
            return;
        }
        if (strcmp(text, "/memory") == 0) {
            load_memories();
            if (g_memory_count == 0) {
                tg_send_message(chat_id, "No memories saved.");
                return;
            }
            char buf[4096];
            int bl = 0;
            for (int i = 0; i < g_memory_count && bl < (int)sizeof(buf) - 256; i++)
                bl += snprintf(buf + bl, sizeof(buf) - bl,
                    "%d. [%s] %s\n", i+1, g_memories[i].type, g_memories[i].name);
            tg_send_message(chat_id, buf);
            return;
        }
        if (strcmp(text, "/help") == 0) {
            tg_send_message(chat_id,
                "PRE Telegram Bot\n\n"
                "Just type a message to chat with PRE. I can:\n"
                "- Search the web (if Brave Search configured)\n"
                "- Read your email (if Google connected)\n"
                "- Browse Google Drive\n"
                "- Query Wolfram Alpha\n"
                "- Check GitHub issues/PRs\n"
                "- Read files on your Mac\n"
                "- Remember things across sessions\n\n"
                "Commands:\n"
                "/new - Fresh conversation\n"
                "/status - Show connections & stats\n"
                "/memory - List saved memories\n"
                "/help - This message");
            return;
        }
        // Unknown command — treat as message
    }

    // Send typing immediately
    tg_send_typing(chat_id);
    g_active_chat_id = chat_id;

    // Agentic tool loop
    convo_add(convo, "user", text);

    for (int loop = 0; loop < MAX_TOOL_LOOP; loop++) {
        char *response = ollama_chat(convo, loop == 0 ? text : convo->history[convo->history_count - 1].content, chat_id);

        // Check for tool call
        ToolCall tc;
        if (extract_tool_call(response, &tc)) {
            printf("  Tool call: %s\n", tc.name);

            // Execute tool
            char tool_output[MAX_TOOL_RESPONSE];
            execute_tool(&tc, tool_output, sizeof(tool_output));
            tool_call_free(&tc);

            // Add assistant response + tool result to history
            convo_add(convo, "assistant", response);
            free(response);

            char tool_msg[MAX_TOOL_RESPONSE + 64];
            snprintf(tool_msg, sizeof(tool_msg), "<tool_response>\n%s\n</tool_response>", tool_output);
            convo_add(convo, "user", tool_msg);

            // Send typing for next round
            tg_send_typing(chat_id);
            continue;
        }

        // No tool call — this is the final response
        convo_add(convo, "assistant", response);

        // Strip any trailing <tool_call> fragments or empty blocks
        char *tc_start = strstr(response, "<tool_call>");
        if (tc_start) *tc_start = 0;

        // Send to Telegram
        size_t rlen = strlen(response);
        if (rlen > 0) {
            tg_send_message(chat_id, response);
        } else {
            tg_send_message(chat_id, "(empty response)");
        }
        free(response);
        break;
    }
}

// ============================================================================
// Main loop — long-poll Telegram getUpdates
// ============================================================================

int main(int argc, char **argv) {
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            g_port = atoi(argv[++i]);
    }

    signal(SIGINT, sighandler);
    signal(SIGTERM, sighandler);
    // Note: typing indicator children are reaped explicitly via waitpid()
    // in ollama_chat(). Do NOT set SIGCHLD to SIG_IGN — it breaks popen/pclose.

    printf("PRE Telegram Bot starting...\n");

    // Load config
    load_connections();
    load_identity();
    if (!g_bot_token[0]) {
        fprintf(stderr, "Error: No Telegram bot token configured.\n");
        fprintf(stderr, "Run PRE and use: /connections add telegram\n");
        return 1;
    }

    load_owner();
    load_memories();

    // Verify bot token
    char *me = tg_api("getMe", NULL);
    if (!me) {
        fprintf(stderr, "Error: Cannot reach Telegram API.\n");
        return 1;
    }
    char bot_name[128] = {0};
    json_extract_str(me, "first_name", bot_name, sizeof(bot_name));
    char ok_str[8] = {0};
    json_extract_str(me, "ok", ok_str, sizeof(ok_str));
    free(me);

    if (strcmp(ok_str, "true") != 0) {
        fprintf(stderr, "Error: Invalid bot token.\n");
        return 1;
    }

    printf("  Bot: %s\n", bot_name);
    printf("  Ollama: localhost:%d\n", g_port);
    printf("  Model: %s\n", g_model);
    printf("  Owner: %s\n", g_owner_id > 0 ? "set" : "first user to message");
    printf("  Memories: %d\n", g_memory_count);
    printf("  Waiting for messages...\n\n");

    // Verify Ollama is running
    char check[256];
    snprintf(check, sizeof(check),
        "curl -sf http://localhost:%d/v1/models >/dev/null 2>&1", g_port);
    if (system(check) != 0) {
        fprintf(stderr, "Warning: Ollama not responding on port %d. Make sure it's running.\n", g_port);
    }

    // Long-poll loop
    long update_offset = 0;

    while (g_running) {
        char url[512];
        snprintf(url, sizeof(url),
            "getUpdates?offset=%ld&timeout=30&allowed_updates=[%%22message%%22]",
            update_offset);
        char *resp = tg_api(url, NULL);
        if (!resp) { sleep(2); continue; }

        // Check ok
        char ok[8] = {0};
        json_extract_str(resp, "ok", ok, sizeof(ok));
        if (strcmp(ok, "true") != 0) {
            fprintf(stderr, "  getUpdates error: %.200s\n", resp);
            free(resp);
            sleep(5);
            continue;
        }

        // Parse updates — scan for update_id fields
        const char *scan = resp;
        while ((scan = json_find_key(scan, "update_id")) != NULL) {
            char uid_s[32] = {0};
            json_extract_str(scan, "update_id", uid_s, sizeof(uid_s));
            long uid = atol(uid_s);
            if (uid >= update_offset) update_offset = uid + 1;

            // Find the message text and chat info
            // Look for "text" field after this update_id
            const char *msg_section = scan;
            char text[4096] = {0}, chat_id_s[32] = {0}, from_id_s[32] = {0}, first_name[64] = {0};

            // Find the "message" object's fields
            const char *chat_key = json_find_key(msg_section, "chat");
            if (chat_key) {
                // Extract chat.id — look for "id" after "chat"
                const char *id_key = json_find_key(chat_key, "id");
                if (id_key) json_extract_str(id_key, "id", chat_id_s, sizeof(chat_id_s));
            }

            const char *from_key = json_find_key(msg_section, "from");
            if (from_key) {
                const char *id_key = json_find_key(from_key, "id");
                if (id_key) json_extract_str(id_key, "id", from_id_s, sizeof(from_id_s));
                json_extract_str(from_key, "first_name", first_name, sizeof(first_name));
            }

            json_extract_str(msg_section, "text", text, sizeof(text));

            if (text[0] && chat_id_s[0]) {
                long cid = atol(chat_id_s);
                long fid = atol(from_id_s);
                printf("  [%s] %s\n", first_name, text);
                handle_message(cid, fid, text);
            }

            scan++;
        }

        free(resp);
    }

    // Unload model from GPU
    printf("\nUnloading model...\n");
    char unload_cmd[512];
    snprintf(unload_cmd, sizeof(unload_cmd),
        "curl -s --max-time 5 -X POST -H 'Content-Type: application/json' "
        "-d '{\"model\":\"%s\",\"keep_alive\":\"0\"}' "
        "'http://127.0.0.1:%d/api/chat' >/dev/null 2>&1",
        g_model, g_port);
    system(unload_cmd);
    printf("Goodbye.\n");
    return 0;
}
