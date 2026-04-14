// PRE Web GUI — macOS Notes.app integration
// Zero-config notes via AppleScript — works with iCloud or any configured account

const { execSync } = require('child_process');

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

/**
 * Notes tool dispatcher.
 * @param {Object} args - { action, title, body, folder, query, id, count }
 */
async function notes(args) {
  const action = args?.action;
  if (!action) return 'Error: action required. Actions: search, read, create, list_recent, list_folders';

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

module.exports = { notes };
