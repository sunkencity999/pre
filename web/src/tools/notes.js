// PRE Web GUI — Notes integration (macOS Notes.app + Windows local markdown)
// macOS: Zero-config notes via AppleScript — works with iCloud or any configured account
// Windows: Local markdown notes in ~/.pre/notes/ — simple, portable, no COM dependency

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { IS_WIN, IS_MAC } = require('../platform');

function runAS(script, timeout = 30000) {
  try {
    return execSync('osascript -', {
      input: script,
      encoding: 'utf-8',
      timeout,
      maxBuffer: 256 * 1024,
    }).trim();
  } catch (err) {
    const msg = (err.stderr || err.message || '').trim();
    throw new Error(msg || 'AppleScript execution failed');
  }
}

// ── Windows: local markdown notes directory ──────────────────────────────────
const NOTES_DIR = path.join(process.env.HOME || process.env.USERPROFILE || '/tmp', '.pre', 'notes');

function ensureNotesDir() {
  if (!fs.existsSync(NOTES_DIR)) {
    fs.mkdirSync(NOTES_DIR, { recursive: true });
  }
}

/**
 * Notes tool dispatcher.
 * @param {Object} args - { action, title, body, folder, query, id, count }
 */
async function notes(args) {
  const action = args?.action;
  if (!action) return 'Error: action required. Actions: search, read, create, list_recent, list_folders';

  if (IS_WIN) {
    switch (action) {
      case 'search': return winSearchNotes(args);
      case 'read': return winReadNote(args);
      case 'create': return winCreateNote(args);
      case 'list_recent': return winListRecent(args);
      case 'list_folders': return winListFolders();
      default:
        return `Error: unknown action '${action}'. Available: search, read, create, list_recent, list_folders`;
    }
  }

  if (IS_MAC) {
    switch (action) {
      case 'search': return searchNotes(args);
      case 'read': return readNote(args);
      case 'create': return createNote(args);
      case 'list_recent': return listRecent(args);
      case 'list_folders': return listFolders();
      default:
        return `Error: unknown action '${action}'. Available: search, read, create, list_recent, list_folders`;
    }
  }

  return 'Error: notes tool is only supported on macOS (Notes.app) and Windows (local markdown notes)';
}

function searchNotes(args) {
  const query = args.query;
  const count = Math.min(args.count || 20, 50);

  if (!query) return 'Error: "query" is required for search';

  const escQuery = query.replace(/\\/g, '\\\\').replace(/"/g, '\\"');

  const script = `
tell application "Notes"
  set resultList to {}
  set matchCount to 0

  set matchedNotes to (every note whose name contains "${escQuery}")
  repeat with n in matchedNotes
    if matchCount ≥ ${count} then exit repeat
    set nDate to modification date of n
    set nDateStr to (year of nDate as text) & "-" & text -2 thru -1 of ("0" & ((month of nDate) as integer) as text) & "-" & text -2 thru -1 of ("0" & (day of nDate) as text)
    set nInfo to "ID:" & (id of n) & " | " & nDateStr & " | " & (name of n)
    try
      set fName to name of container of n
      set nInfo to nInfo & " | Folder: " & fName
    end try
    -- Preview first 100 chars of body
    try
      set bodyText to plaintext of n
      if bodyText is not "" then
        set previewLen to min of {(length of bodyText), 100}
        set nInfo to nInfo & " | Preview: " & (text 1 thru previewLen of bodyText)
      end if
    end try
    set end of resultList to nInfo
    set matchCount to matchCount + 1
  end repeat

  -- Also search body content if title search found nothing
  if (count of resultList) = 0 then
    set bodyMatches to (every note whose plaintext contains "${escQuery}")
    repeat with n in bodyMatches
      if matchCount ≥ ${count} then exit repeat
      set nDate to modification date of n
      set nDateStr to (year of nDate as text) & "-" & text -2 thru -1 of ("0" & ((month of nDate) as integer) as text) & "-" & text -2 thru -1 of ("0" & (day of nDate) as text)
      set nInfo to "ID:" & (id of n) & " | " & nDateStr & " | " & (name of n)
      try
        set fName to name of container of n
        set nInfo to nInfo & " | Folder: " & fName
      end try
      set end of resultList to nInfo
      set matchCount to matchCount + 1
    end repeat
  end if

  if (count of resultList) = 0 then
    return "No notes found matching: ${escQuery}"
  end if

  set AppleScript's text item delimiters to "\\n"
  return resultList as text
end tell
`;

  try {
    return runAS(script, 60000);
  } catch (err) {
    return `Error searching notes: ${err.message}`;
  }
}

function readNote(args) {
  const noteId = args.id;
  const title = args.title || args.query;

  if (!noteId && !title) return 'Error: "id" or "title" is required';

  let findClause;
  if (noteId) {
    findClause = `first note whose id is "${noteId}"`;
  } else {
    findClause = `first note whose name contains "${(title || '').replace(/"/g, '\\"')}"`;
  }

  const script = `
tell application "Notes"
  try
    set n to ${findClause}
  on error
    return "Note not found"
  end try

  set nDate to modification date of n
  set nDateStr to (year of nDate as text) & "-" & text -2 thru -1 of ("0" & ((month of nDate) as integer) as text) & "-" & text -2 thru -1 of ("0" & (day of nDate) as text)

  set cDate to creation date of n
  set cDateStr to (year of cDate as text) & "-" & text -2 thru -1 of ("0" & ((month of cDate) as integer) as text) & "-" & text -2 thru -1 of ("0" & (day of cDate) as text)

  set info to "Title: " & (name of n) & "\\nCreated: " & cDateStr & "\\nModified: " & nDateStr

  try
    set fName to name of container of n
    set info to info & "\\nFolder: " & fName
  end try

  set bodyText to plaintext of n
  -- Truncate very long notes
  set maxLen to 8000
  if (length of bodyText) > maxLen then
    set bodyText to (text 1 thru maxLen of bodyText) & "\\n\\n[...truncated]"
  end if

  set info to info & "\\n\\n" & bodyText
  return info
end tell
`;

  try {
    return runAS(script, 30000);
  } catch (err) {
    return `Error reading note: ${err.message}`;
  }
}

function createNote(args) {
  const title = args.title || 'Untitled';
  const body = args.body || '';
  const folder = args.folder || '';

  const escTitle = title.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  const escBody = body.replace(/\\/g, '\\\\').replace(/"/g, '\\"');

  const folderTarget = folder
    ? `folder "${folder.replace(/"/g, '\\"')}"`
    : 'default account';

  // Notes.app expects HTML-ish body content
  // We'll set the body as plaintext by using the name + body approach
  const script = `
tell application "Notes"
  set newNote to make new note at ${folderTarget} with properties {name:"${escTitle}", body:"${escBody}"}
  return "Note created: ${escTitle}\\nID: " & (id of newNote)
end tell
`;

  try {
    return runAS(script);
  } catch (err) {
    return `Error creating note: ${err.message}`;
  }
}

function listRecent(args) {
  const count = Math.min(args.count || 20, 50);
  const folder = args.folder || '';

  let noteSource;
  if (folder) {
    noteSource = `notes of folder "${folder.replace(/"/g, '\\"')}"`;
  } else {
    noteSource = 'notes';
  }

  const script = `
tell application "Notes"
  set resultList to {}
  set matchCount to 0

  set allNotes to ${noteSource}
  repeat with n in allNotes
    if matchCount ≥ ${count} then exit repeat

    set nDate to modification date of n
    set nDateStr to (year of nDate as text) & "-" & text -2 thru -1 of ("0" & ((month of nDate) as integer) as text) & "-" & text -2 thru -1 of ("0" & (day of nDate) as text)
    set nInfo to "ID:" & (id of n) & " | " & nDateStr & " | " & (name of n)

    try
      set fName to name of container of n
      set nInfo to nInfo & " | Folder: " & fName
    end try

    set end of resultList to nInfo
    set matchCount to matchCount + 1
  end repeat

  if (count of resultList) = 0 then
    return "No notes found"
  end if

  set AppleScript's text item delimiters to "\\n"
  return resultList as text
end tell
`;

  try {
    return runAS(script, 30000);
  } catch (err) {
    return `Error listing notes: ${err.message}`;
  }
}

function listFolders() {
  const script = `
tell application "Notes"
  set resultList to {}
  repeat with acct in accounts
    set acctName to name of acct
    repeat with f in folders of acct
      set fName to name of f
      set noteCount to count of notes of f
      set end of resultList to acctName & " / " & fName & " (" & noteCount & " notes)"
    end repeat
  end repeat

  if (count of resultList) = 0 then
    return "No note folders found"
  end if

  set AppleScript's text item delimiters to "\\n"
  return resultList as text
end tell
`;

  try {
    return runAS(script);
  } catch (err) {
    return `Error listing folders: ${err.message}`;
  }
}

// ── Windows local markdown note implementations ─────────────────────────────

/**
 * Generate a filesystem-safe slug from a title.
 */
function slugify(title) {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 80) || 'untitled';
}

/**
 * Read frontmatter + body from a markdown note file.
 * Files use a simple format: first line "# Title", optional metadata lines, then body.
 */
function parseNoteFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const stat = fs.statSync(filePath);
  const lines = content.split('\n');
  let title = path.basename(filePath, '.md');
  let body = content;

  if (lines[0] && lines[0].startsWith('# ')) {
    title = lines[0].slice(2).trim();
    body = lines.slice(1).join('\n').trim();
  }

  return {
    title,
    body,
    id: path.basename(filePath),
    folder: path.basename(path.dirname(filePath)),
    created: stat.birthtime,
    modified: stat.mtime,
    filePath,
  };
}

function winSearchNotes(args) {
  const query = args.query;
  const count = Math.min(args.count || 20, 50);
  if (!query) return 'Error: "query" is required for search';

  ensureNotesDir();
  const q = query.toLowerCase();
  const results = [];

  function searchDir(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (results.length >= count) break;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        searchDir(full);
      } else if (entry.name.endsWith('.md')) {
        try {
          const note = parseNoteFile(full);
          if (note.title.toLowerCase().includes(q) || note.body.toLowerCase().includes(q)) {
            const dateStr = note.modified.toISOString().slice(0, 10);
            const preview = note.body.slice(0, 100).replace(/\n/g, ' ');
            let line = `ID:${note.id} | ${dateStr} | ${note.title}`;
            if (note.folder !== 'notes') line += ` | Folder: ${note.folder}`;
            if (preview) line += ` | Preview: ${preview}`;
            results.push(line);
          }
        } catch {}
      }
    }
  }

  searchDir(NOTES_DIR);
  return results.length > 0 ? results.join('\n') : `No notes found matching: ${query}`;
}

function winReadNote(args) {
  const noteId = args.id;
  const title = args.title || args.query;
  if (!noteId && !title) return 'Error: "id" or "title" is required';

  ensureNotesDir();

  // Try direct file match by id
  if (noteId) {
    const direct = path.join(NOTES_DIR, noteId);
    if (fs.existsSync(direct)) {
      const note = parseNoteFile(direct);
      const created = note.created.toISOString().slice(0, 10);
      const modified = note.modified.toISOString().slice(0, 10);
      let info = `Title: ${note.title}\nCreated: ${created}\nModified: ${modified}`;
      if (note.folder !== 'notes') info += `\nFolder: ${note.folder}`;
      const bodyPreview = note.body.length > 8000 ? note.body.slice(0, 8000) + '\n\n[...truncated]' : note.body;
      info += `\n\n${bodyPreview}`;
      return info;
    }
  }

  // Search by title
  const q = (title || noteId || '').toLowerCase();
  function findInDir(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        const found = findInDir(full);
        if (found) return found;
      } else if (entry.name.endsWith('.md')) {
        try {
          const note = parseNoteFile(full);
          if (note.title.toLowerCase().includes(q) || entry.name.toLowerCase().includes(q)) {
            return note;
          }
        } catch {}
      }
    }
    return null;
  }

  const note = findInDir(NOTES_DIR);
  if (!note) return 'Note not found';

  const created = note.created.toISOString().slice(0, 10);
  const modified = note.modified.toISOString().slice(0, 10);
  let info = `Title: ${note.title}\nCreated: ${created}\nModified: ${modified}`;
  if (note.folder !== 'notes') info += `\nFolder: ${note.folder}`;
  const bodyPreview = note.body.length > 8000 ? note.body.slice(0, 8000) + '\n\n[...truncated]' : note.body;
  info += `\n\n${bodyPreview}`;
  return info;
}

function winCreateNote(args) {
  const title = args.title || 'Untitled';
  const body = args.body || '';
  const folder = args.folder || '';

  ensureNotesDir();

  let targetDir = NOTES_DIR;
  if (folder) {
    targetDir = path.join(NOTES_DIR, slugify(folder));
    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }
  }

  const slug = slugify(title);
  let fileName = `${slug}.md`;
  let filePath = path.join(targetDir, fileName);

  // Avoid overwriting existing notes
  let counter = 1;
  while (fs.existsSync(filePath)) {
    fileName = `${slug}-${counter}.md`;
    filePath = path.join(targetDir, fileName);
    counter++;
  }

  const content = `# ${title}\n\n${body}`;
  fs.writeFileSync(filePath, content, 'utf-8');

  return `Note created: ${title}\nID: ${fileName}`;
}

function winListRecent(args) {
  const count = Math.min(args.count || 20, 50);
  const folder = args.folder || '';

  ensureNotesDir();

  let targetDir = NOTES_DIR;
  if (folder) {
    const folderPath = path.join(NOTES_DIR, folder);
    if (fs.existsSync(folderPath)) targetDir = folderPath;
  }

  const allNotes = [];
  function collectNotes(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        collectNotes(full);
      } else if (entry.name.endsWith('.md')) {
        try {
          allNotes.push(parseNoteFile(full));
        } catch {}
      }
    }
  }

  collectNotes(targetDir);

  // Sort by modification time descending
  allNotes.sort((a, b) => b.modified - a.modified);

  const results = allNotes.slice(0, count).map(note => {
    const dateStr = note.modified.toISOString().slice(0, 10);
    let line = `ID:${note.id} | ${dateStr} | ${note.title}`;
    if (note.folder !== 'notes') line += ` | Folder: ${note.folder}`;
    return line;
  });

  return results.length > 0 ? results.join('\n') : 'No notes found';
}

function winListFolders() {
  ensureNotesDir();

  const results = [];

  // Count notes in root
  const rootNotes = fs.readdirSync(NOTES_DIR).filter(f => f.endsWith('.md'));
  results.push(`notes (${rootNotes.length} notes)`);

  // Count notes in subfolders
  const entries = fs.readdirSync(NOTES_DIR, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.isDirectory()) {
      const subDir = path.join(NOTES_DIR, entry.name);
      const subNotes = fs.readdirSync(subDir).filter(f => f.endsWith('.md'));
      results.push(`${entry.name} (${subNotes.length} notes)`);
    }
  }

  return results.length > 0 ? results.join('\n') : 'No note folders found';
}

module.exports = { notes };
