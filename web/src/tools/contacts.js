// PRE Web GUI — macOS Contacts.app integration
// Zero-config contacts via AppleScript — works with iCloud, Google, Exchange, or any synced account

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
 * Contacts tool dispatcher.
 * @param {Object} args - { action, query, id, name, count }
 */
async function contacts(args) {
  const action = args?.action;
  if (!action) return 'Error: action required. Actions: search, read, list_groups, count';

  switch (action) {
    case 'search': return searchContacts(args);
    case 'read': return readContact(args);
    case 'list_groups': return listGroups();
    case 'count': return countContacts();
    default:
      return `Error: unknown action '${action}'. Available: search, read, list_groups, count`;
  }
}

function searchContacts(args) {
  const query = args.query || args.name;
  const count = Math.min(args.count || 20, 50);

  if (!query) return 'Error: "query" or "name" is required for search';

  const escQuery = query.replace(/\\/g, '\\\\').replace(/"/g, '\\"');

  const script = `
tell application "Contacts"
  set resultList to {}
  set matchCount to 0

  -- Search by name (first, last, or full name)
  set matches to (every person whose name contains "${escQuery}")

  repeat with p in matches
    if matchCount ≥ ${count} then exit repeat

    set pName to name of p
    set pInfo to "ID:" & (id of p) & " | " & pName

    -- Emails
    try
      set emailList to {}
      repeat with e in emails of p
        set end of emailList to (label of e) & ": " & (value of e)
      end repeat
      if (count of emailList) > 0 then
        set AppleScript's text item delimiters to ", "
        set pInfo to pInfo & " | Email: " & (emailList as text)
      end if
    end try

    -- Phone numbers
    try
      set phoneList to {}
      repeat with ph in phones of p
        set end of phoneList to (label of ph) & ": " & (value of ph)
      end repeat
      if (count of phoneList) > 0 then
        set AppleScript's text item delimiters to ", "
        set pInfo to pInfo & " | Phone: " & (phoneList as text)
      end if
    end try

    -- Organization
    try
      set org to organization of p
      if org is not missing value and org is not "" then
        set pInfo to pInfo & " | Org: " & org
      end if
    end try

    -- Job title
    try
      set jt to job title of p
      if jt is not missing value and jt is not "" then
        set pInfo to pInfo & " | Title: " & jt
      end if
    end try

    set end of resultList to pInfo
    set matchCount to matchCount + 1
  end repeat

  if (count of resultList) = 0 then
    -- Try searching by email or organization
    set orgMatches to (every person whose organization contains "${escQuery}")
    repeat with p in orgMatches
      if matchCount ≥ ${count} then exit repeat
      set pInfo to "ID:" & (id of p) & " | " & (name of p)
      try
        set pInfo to pInfo & " | Org: " & (organization of p)
      end try
      try
        set e1 to value of first email of p
        set pInfo to pInfo & " | Email: " & e1
      end try
      set end of resultList to pInfo
      set matchCount to matchCount + 1
    end repeat
  end if

  if (count of resultList) = 0 then
    return "No contacts found matching: ${escQuery}"
  end if

  set AppleScript's text item delimiters to "\\n"
  return resultList as text
end tell
`;

  try {
    return runAS(script, 30000);
  } catch (err) {
    return `Error searching contacts: ${err.message}`;
  }
}

function readContact(args) {
  const contactId = args.id;
  const name = args.name;

  if (!contactId && !name) return 'Error: "id" or "name" is required';

  const lookupClause = contactId
    ? `first person whose id is "${contactId}"`
    : `first person whose name contains "${(name || '').replace(/"/g, '\\"')}"`;

  const script = `
tell application "Contacts"
  try
    set p to ${lookupClause}
  on error
    return "Contact not found"
  end try

  set info to "Name: " & (name of p)

  -- Organization & title
  try
    set org to organization of p
    if org is not missing value and org is not "" then set info to info & "\\nOrganization: " & org
  end try
  try
    set jt to job title of p
    if jt is not missing value and jt is not "" then set info to info & "\\nTitle: " & jt
  end try
  try
    set dept to department of p
    if dept is not missing value and dept is not "" then set info to info & "\\nDepartment: " & dept
  end try

  -- Emails
  try
    repeat with e in emails of p
      set info to info & "\\nEmail (" & (label of e) & "): " & (value of e)
    end repeat
  end try

  -- Phones
  try
    repeat with ph in phones of p
      set info to info & "\\nPhone (" & (label of ph) & "): " & (value of ph)
    end repeat
  end try

  -- Addresses
  try
    repeat with addr in addresses of p
      set addrStr to ""
      try
        set addrStr to (street of addr)
        if (city of addr) is not missing value then set addrStr to addrStr & ", " & (city of addr)
        if (state of addr) is not missing value then set addrStr to addrStr & ", " & (state of addr)
        if (zip of addr) is not missing value then set addrStr to addrStr & " " & (zip of addr)
      end try
      if addrStr is not "" then set info to info & "\\nAddress (" & (label of addr) & "): " & addrStr
    end repeat
  end try

  -- Birthday
  try
    set bd to birth date of p
    if bd is not missing value then
      set bdStr to (year of bd as text) & "-" & text -2 thru -1 of ("0" & ((month of bd) as integer) as text) & "-" & text -2 thru -1 of ("0" & (day of bd) as text)
      set info to info & "\\nBirthday: " & bdStr
    end if
  end try

  -- Notes
  try
    set n to note of p
    if n is not missing value and n is not "" then set info to info & "\\nNotes: " & (text 1 thru (min of {(length of n), 500}) of n)
  end try

  return info
end tell
`;

  try {
    return runAS(script, 15000);
  } catch (err) {
    return `Error reading contact: ${err.message}`;
  }
}

function listGroups() {
  const script = `
tell application "Contacts"
  set resultList to {}
  repeat with g in groups
    set gInfo to (name of g) & " (" & (count of people of g) & " contacts)"
    set end of resultList to gInfo
  end repeat

  if (count of resultList) = 0 then
    return "No contact groups found"
  end if

  set AppleScript's text item delimiters to "\\n"
  return resultList as text
end tell
`;

  try {
    return runAS(script);
  } catch (err) {
    return `Error listing groups: ${err.message}`;
  }
}

function countContacts() {
  const script = `
tell application "Contacts"
  set totalCount to count of people
  return "Total contacts: " & totalCount
end tell
`;

  try {
    return runAS(script);
  } catch (err) {
    return `Error counting contacts: ${err.message}`;
  }
}

module.exports = { contacts };
