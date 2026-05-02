// PRE Web GUI — Contacts integration (macOS Contacts.app + Windows Outlook COM)
// macOS: Zero-config contacts via AppleScript — works with iCloud, Google, Exchange, or any synced account
// Windows: Outlook COM automation via PowerShell — works with any account configured in Outlook

const { execSync } = require('child_process');
const { IS_WIN, IS_MAC, IS_LINUX } = require('../platform');

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
 * Run a PowerShell script and return the result.
 */
function runPS(script, timeout = 30000) {
  try {
    return execSync('powershell.exe -NoProfile -Command -', {
      input: `$ErrorActionPreference = 'Stop'\n${script}`,
      encoding: 'utf-8',
      timeout,
      maxBuffer: 256 * 1024,
    }).trim();
  } catch (err) {
    const msg = (err.stderr || err.message || '').trim();
    throw new Error(msg || 'PowerShell execution failed');
  }
}

/**
 * Escape a string for safe embedding in PowerShell single-quoted strings.
 */
function escPS(str) {
  return (str || '').replace(/'/g, "''");
}

/**
 * Contacts tool dispatcher.
 * @param {Object} args - { action, query, id, name, count }
 */
async function contacts(args) {
  const action = args?.action;
  if (!action) return 'Error: action required. Actions: search, read, list_groups, count';

  if (IS_WIN) {
    switch (action) {
      case 'search': return winSearchContacts(args);
      case 'read': return winReadContact(args);
      case 'list_groups': return winListGroups();
      case 'count': return winCountContacts();
      default:
        return `Error: unknown action '${action}'. Available: search, read, list_groups, count`;
    }
  }

  if (IS_MAC) {
    switch (action) {
      case 'search': return searchContacts(args);
      case 'read': return readContact(args);
      case 'list_groups': return listGroups();
      case 'count': return countContacts();
      default:
        return `Error: unknown action '${action}'. Available: search, read, list_groups, count`;
    }
  }

  if (IS_LINUX) {
    const eds = require('./eds-linux');
    switch (action) {
      case 'search': return eds.edsContactSearch(args.query || args.name, args.count);
      case 'read': return eds.edsContactRead(args.id || args.name);
      case 'list_groups': return eds.edsListGroups();
      case 'count': return eds.edsContactCount();
      default:
        return `Error: unknown action '${action}'. Available: search, read, list_groups, count`;
    }
  }

  return 'Error: contacts tool is not supported on this platform';
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

// ── Windows Outlook COM implementations ──────────────────────────────────────

function winSearchContacts(args) {
  const query = args.query || args.name;
  const count = Math.min(args.count || 20, 50);

  if (!query) return 'Error: "query" or "name" is required for search';

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace('MAPI')
$folder = $ns.GetDefaultFolder(10)
$q = '${escPS(query)}'.ToLower()
$results = @()
$matchCount = 0
foreach ($item in $folder.Items) {
  if ($matchCount -ge ${count}) { break }
  $fullName = if ($item.FullName) { $item.FullName } else { '' }
  $company = if ($item.CompanyName) { $item.CompanyName } else { '' }
  $email1 = if ($item.Email1Address) { $item.Email1Address } else { '' }
  if ($fullName.ToLower().Contains($q) -or $company.ToLower().Contains($q) -or $email1.ToLower().Contains($q)) {
    $line = "ID:$($item.EntryID.Substring(0,20)) | $fullName"
    if ($email1) { $line += " | Email: $email1" }
    $phone = if ($item.BusinessTelephoneNumber) { $item.BusinessTelephoneNumber } elseif ($item.MobileTelephoneNumber) { $item.MobileTelephoneNumber } else { '' }
    if ($phone) { $line += " | Phone: $phone" }
    if ($company) { $line += " | Org: $company" }
    $title = if ($item.JobTitle) { $item.JobTitle } else { '' }
    if ($title) { $line += " | Title: $title" }
    $results += $line
    $matchCount++
  }
}
if ($results.Count -eq 0) { Write-Output 'No contacts found matching: ${escPS(query)}' }
else { $results | ForEach-Object { Write-Output $_ } }
`;

  try {
    return runPS(script, 30000);
  } catch (err) {
    return `Error searching contacts: ${err.message}`;
  }
}

function winReadContact(args) {
  const contactId = args.id;
  const name = args.name;

  if (!contactId && !name) return 'Error: "id" or "name" is required';

  const lookupClause = contactId
    ? `$item = $ns.GetItemFromID('${escPS(contactId)}')`
    : `$folder = $ns.GetDefaultFolder(10)
$item = $null
$q = '${escPS(name)}'.ToLower()
foreach ($c in $folder.Items) {
  if ($c.FullName -and $c.FullName.ToLower().Contains($q)) { $item = $c; break }
}
if (-not $item) { Write-Output 'Contact not found'; exit }`;

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace('MAPI')
${lookupClause}
$info = @()
$info += "Name: $($item.FullName)"
if ($item.CompanyName) { $info += "Organization: $($item.CompanyName)" }
if ($item.JobTitle) { $info += "Title: $($item.JobTitle)" }
if ($item.Department) { $info += "Department: $($item.Department)" }
if ($item.Email1Address) { $info += "Email (Work): $($item.Email1Address)" }
if ($item.Email2Address) { $info += "Email (Other): $($item.Email2Address)" }
if ($item.Email3Address) { $info += "Email (Other 2): $($item.Email3Address)" }
if ($item.BusinessTelephoneNumber) { $info += "Phone (Business): $($item.BusinessTelephoneNumber)" }
if ($item.MobileTelephoneNumber) { $info += "Phone (Mobile): $($item.MobileTelephoneNumber)" }
if ($item.HomeTelephoneNumber) { $info += "Phone (Home): $($item.HomeTelephoneNumber)" }
if ($item.BusinessAddress) { $info += "Address (Business): $($item.BusinessAddress)" }
if ($item.HomeAddress) { $info += "Address (Home): $($item.HomeAddress)" }
if ($item.Birthday -and $item.Birthday.Year -gt 1900) { $info += "Birthday: $($item.Birthday.ToString('yyyy-MM-dd'))" }
if ($item.Body) {
  $preview = if ($item.Body.Length -gt 500) { $item.Body.Substring(0,500) + '...' } else { $item.Body }
  $info += "Notes: $preview"
}
$info | ForEach-Object { Write-Output $_ }
`;

  try {
    return runPS(script, 15000);
  } catch (err) {
    return `Error reading contact: ${err.message}`;
  }
}

function winListGroups() {
  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace('MAPI')
$folder = $ns.GetDefaultFolder(10)
$results = @()
foreach ($sub in $folder.Folders) {
  $results += "$($sub.Name) ($($sub.Items.Count) contacts)"
}
# Also list distribution lists in the main contacts folder
foreach ($item in $folder.Items) {
  if ($item.Class -eq 69) {  # olDistributionList
    $results += "Distribution List: $($item.DLName) ($($item.MemberCount) members)"
  }
}
if ($results.Count -eq 0) { Write-Output 'No contact groups found' }
else { $results | ForEach-Object { Write-Output $_ } }
`;

  try {
    return runPS(script);
  } catch (err) {
    return `Error listing groups: ${err.message}`;
  }
}

function winCountContacts() {
  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace('MAPI')
$folder = $ns.GetDefaultFolder(10)
Write-Output "Total contacts: $($folder.Items.Count)"
`;

  try {
    return runPS(script);
  } catch (err) {
    return `Error counting contacts: ${err.message}`;
  }
}

module.exports = { contacts };
