// PRE Web GUI — Mail integration (macOS Mail.app + Windows Outlook COM)
// macOS: Zero-config email via AppleScript — works with any account configured in Mail.app
// Windows: Outlook COM automation via PowerShell — works with any account configured in Outlook

const { execSync } = require('child_process');
const { IS_WIN, IS_MAC } = require('../platform');

/**
 * Run an AppleScript and return the result.
 * Uses heredoc-style input to handle complex scripts with quotes.
 */
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
 * Uses -NoProfile -Command for clean, non-interactive execution.
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
 * Single quotes are doubled per PowerShell convention.
 */
function escPS(str) {
  return (str || '').replace(/'/g, "''");
}

/**
 * Mail tool dispatcher.
 * @param {Object} args - { action, to, cc, bcc, subject, body, query, id, count, mailbox, account }
 */
async function mail(args) {
  const action = args?.action;
  if (!action) return 'Error: action required. Actions: send, draft, search, read, list_recent, list_mailboxes, list_accounts';

  if (IS_WIN) {
    switch (action) {
      case 'send': return winSendMail(args);
      case 'draft': return winDraftMail(args);
      case 'search': return winSearchMail(args);
      case 'read': return winReadMail(args);
      case 'list_recent': return winListRecent(args);
      case 'list_mailboxes': return winListMailboxes(args);
      case 'list_accounts': return winListAccounts();
      default:
        return `Error: unknown action '${action}'. Available: send, draft, search, read, list_recent, list_mailboxes, list_accounts`;
    }
  }

  if (IS_MAC) {
    switch (action) {
      case 'send': return sendMail(args);
      case 'draft': return draftMail(args);
      case 'search': return searchMail(args);
      case 'read': return readMail(args);
      case 'list_recent': return listRecent(args);
      case 'list_mailboxes': return listMailboxes(args);
      case 'list_accounts': return listAccounts();
      default:
        return `Error: unknown action '${action}'. Available: send, draft, search, read, list_recent, list_mailboxes, list_accounts`;
    }
  }

  return 'Error: mail tool is only supported on macOS (Mail.app) and Windows (Outlook)';
}

// ── Windows Outlook COM implementations ──────────────────────────────────────

function winSendMail(args) {
  const to = args.to;
  const subject = args.subject || '';
  const body = args.body || '';
  const cc = args.cc || '';
  const bcc = args.bcc || '';

  if (!to) return 'Error: "to" recipient is required';

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$mail = $outlook.CreateItem(0)
$mail.To = '${escPS(to)}'
${cc ? `$mail.CC = '${escPS(cc)}'` : ''}
${bcc ? `$mail.BCC = '${escPS(bcc)}'` : ''}
$mail.Subject = '${escPS(subject)}'
$mail.Body = '${escPS(body)}'
$mail.Send()
Write-Output 'Email sent to ${escPS(to)}'
`;

  try {
    return runPS(script);
  } catch (err) {
    return `Error sending email: ${err.message}`;
  }
}

function winDraftMail(args) {
  const to = args.to || '';
  const subject = args.subject || '';
  const body = args.body || '';

  const toLine = to ? `\$mail.To = '${escPS(to)}'` : '';
  const draftMsg = 'Draft created' + (to ? ' for ' + escPS(to) : '') + ': ' + escPS(subject);

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$mail = $outlook.CreateItem(0)
${toLine}
$mail.Subject = '${escPS(subject)}'
$mail.Body = '${escPS(body)}'
$mail.Display()
$mail.Save()
Write-Output '${escPS(draftMsg)}'
`;

  try {
    return runPS(script);
  } catch (err) {
    return `Error creating draft: ${err.message}`;
  }
}

function winSearchMail(args) {
  const query = args.query;
  const count = Math.min(args.count || 20, 50);

  if (!query) return 'Error: "query" is required for search';

  const escaped = escPS(query);

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace("MAPI")
$inbox = $ns.GetDefaultFolder(6)
$filter = '@SQL="urn:schemas:httpmail:subject" LIKE ''%${escaped}%'' OR "urn:schemas:httpmail:fromemail" LIKE ''%${escaped}%'''
$items = $inbox.Items.Restrict($filter)
$items.Sort("[ReceivedTime]", $true)
$results = @()
$max = ${count}
$i = 0
foreach ($msg in $items) {
  if ($i -ge $max) { break }
  $dateStr = $msg.ReceivedTime.ToString("yyyy-MM-dd")
  $readStatus = if ($msg.UnRead) { "false" } else { "true" }
  $results += "ID:" + $msg.EntryID + " | " + $dateStr + " | From: " + $msg.SenderName + " | Subject: " + $msg.Subject + " | Read: " + $readStatus
  $i++
}
if ($results.Count -eq 0) {
  Write-Output 'No messages found matching: ${escaped}'
} else {
  Write-Output ($results -join "\`n")
}
`;

  try {
    return runPS(script, 60000);
  } catch (err) {
    return `Error searching mail: ${err.message}`;
  }
}

function winReadMail(args) {
  const msgId = args.id;
  if (!msgId) return 'Error: "id" is required (from search or list_recent results)';

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace("MAPI")
try {
  $m = $ns.GetItemFromID('${escPS(msgId)}')
  $from = $m.SenderName
  $toAddr = $m.To
  $dateStr = $m.ReceivedTime.ToString("yyyy-MM-dd")
  $subj = $m.Subject
  $readStatus = if ($m.UnRead) { "false" } else { "true" }
  $body = $m.Body
  if ($body.Length -gt 7000) {
    $body = $body.Substring(0, 7000)
  }
  Write-Output ("From: " + $from + "\`nTo: " + $toAddr + "\`nDate: " + $dateStr + "\`nSubject: " + $subj + "\`nRead: " + $readStatus + "\`n\`n" + $body)
} catch {
  Write-Output "Message not found with ID: ${escPS(msgId)}"
}
`;

  try {
    return runPS(script);
  } catch (err) {
    return `Error reading message: ${err.message}`;
  }
}

function winListRecent(args) {
  const count = Math.min(args.count || 15, 50);

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace("MAPI")
$inbox = $ns.GetDefaultFolder(6)
$items = $inbox.Items
$items.Sort("[ReceivedTime]", $true)
$results = @()
$max = ${count}
$i = 0
foreach ($msg in $items) {
  if ($i -ge $max) { break }
  $dateStr = $msg.ReceivedTime.ToString("yyyy-MM-dd HH:mm")
  $prefix = if ($msg.UnRead) { "[UNREAD] " } else { "" }
  $results += $prefix + "ID:" + $msg.EntryID + " | " + $dateStr + " | From: " + $msg.SenderName + " | Subject: " + $msg.Subject
  $i++
}
if ($results.Count -eq 0) {
  Write-Output 'No messages found in Inbox'
} else {
  Write-Output ($results -join "\`n")
}
`;

  try {
    return runPS(script);
  } catch (err) {
    return `Error listing mail: ${err.message}`;
  }
}

function winListMailboxes(args) {
  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace("MAPI")
$results = @()
$store = $ns.DefaultStore
$root = $store.GetRootFolder()
$storeName = $store.DisplayName
function List-Folders($parent, $storeName) {
  foreach ($folder in $parent.Folders) {
    $script:results += $storeName + " / " + $folder.Name + " (" + $folder.Items.Count + " messages)"
    List-Folders $folder $storeName
  }
}
List-Folders $root $storeName
if ($results.Count -eq 0) {
  Write-Output 'No mailboxes found'
} else {
  Write-Output ($results -join "\`n")
}
`;

  try {
    return runPS(script, 30000);
  } catch (err) {
    return `Error listing mailboxes: ${err.message}`;
  }
}

function winListAccounts() {
  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace("MAPI")
$results = @()
foreach ($acct in $ns.Accounts) {
  $results += $acct.DisplayName + " | " + $acct.SmtpAddress + " | Enabled: true"
}
if ($results.Count -eq 0) {
  Write-Output 'No accounts found'
} else {
  Write-Output ($results -join "\`n")
}
`;

  try {
    return runPS(script);
  } catch (err) {
    return `Error listing accounts: ${err.message}`;
  }
}

// ── macOS Mail.app implementations ───────────────────────────────────────────

function sendMail(args) {
  const to = args.to;
  const subject = args.subject || '';
  const body = args.body || '';
  const cc = args.cc || '';
  const bcc = args.bcc || '';

  if (!to) return 'Error: "to" recipient is required';

  const escBody = body.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  const escSubject = subject.replace(/\\/g, '\\\\').replace(/"/g, '\\"');

  let recipientLines = `make new to recipient at end of to recipients with properties {address:"${to}"}`;
  if (cc) {
    recipientLines += `\nmake new cc recipient at end of cc recipients with properties {address:"${cc}"}`;
  }
  if (bcc) {
    recipientLines += `\nmake new bcc recipient at end of bcc recipients with properties {address:"${bcc}"}`;
  }

  const script = `
tell application "Mail"
  set newMsg to make new outgoing message with properties {subject:"${escSubject}", content:"${escBody}", visible:false}
  tell newMsg
    ${recipientLines}
  end tell
  send newMsg
end tell
return "Email sent to ${to}"
`;

  try {
    return runAS(script);
  } catch (err) {
    return `Error sending email: ${err.message}`;
  }
}

function draftMail(args) {
  const to = args.to || '';
  const subject = args.subject || '';
  const body = args.body || '';

  const escBody = body.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  const escSubject = subject.replace(/\\/g, '\\\\').replace(/"/g, '\\"');

  let recipientLine = '';
  if (to) {
    recipientLine = `tell newMsg\nmake new to recipient at end of to recipients with properties {address:"${to}"}\nend tell`;
  }

  const script = `
tell application "Mail"
  set newMsg to make new outgoing message with properties {subject:"${escSubject}", content:"${escBody}", visible:true}
  ${recipientLine}
  activate
end tell
return "Draft created${to ? ' for ' + to : ''}: ${escSubject}"
`;

  try {
    return runAS(script);
  } catch (err) {
    return `Error creating draft: ${err.message}`;
  }
}

function searchMail(args) {
  const query = args.query;
  const count = Math.min(args.count || 20, 50);
  const mailbox = args.mailbox || 'inbox';

  if (!query) return 'Error: "query" is required for search';

  const escQuery = query.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  // Always use mailbox "..." syntax — the bare `inbox` keyword fails on Exchange/IMAP accounts
  const mboxName = mailbox.toLowerCase() === 'inbox' ? 'INBOX' : mailbox;

  // Build account filter
  const accountArg = args.account || '';
  let accountFilter = '';
  if (accountArg) {
    accountFilter = `set acctList to {account "${accountArg.replace(/"/g, '\\"')}"}`;
  } else {
    accountFilter = 'set acctList to accounts';
  }

  // Search by subject and sender — AppleScript's "whose" clause
  const script = `
tell application "Mail"
  ${accountFilter}
  set resultList to {}
  set matchCount to 0
  set maxResults to ${count}

  -- Search across accounts
  repeat with acct in acctList
    try
      set msgs to messages of mailbox "${mboxName}" of acct whose subject contains "${escQuery}" or sender contains "${escQuery}"
      repeat with m in msgs
        if matchCount ≥ maxResults then exit repeat
        set msgDate to date received of m
        set msgDateStr to (year of msgDate as text) & "-" & text -2 thru -1 of ("0" & ((month of msgDate) as integer) as text) & "-" & text -2 thru -1 of ("0" & (day of msgDate) as text)
        set msgInfo to "ID:" & (id of m) & " | " & msgDateStr & " | From: " & (sender of m) & " | Subject: " & (subject of m) & " | Read: " & (read status of m)
        set end of resultList to msgInfo
        set matchCount to matchCount + 1
      end repeat
    end try
    if matchCount ≥ maxResults then exit repeat
  end repeat

  if (count of resultList) = 0 then
    return "No messages found matching: ${escQuery}"
  end if

  set AppleScript's text item delimiters to "\\n"
  return resultList as text
end tell
`;

  try {
    return runAS(script, 60000);
  } catch (err) {
    return `Error searching mail: ${err.message}`;
  }
}

function readMail(args) {
  const msgId = args.id;
  if (!msgId) return 'Error: "id" is required (from search or list_recent results)';

  const script = `
tell application "Mail"
  repeat with acct in accounts
    repeat with mbox in mailboxes of acct
      try
        set m to first message of mbox whose id is ${msgId}
        set sndr to sender of m
        set s to subject of m
        set readStat to read status of m as text
        set msgDate to date received of m
        set msgDateStr to (year of msgDate as text) & "-" & text -2 thru -1 of ("0" & ((month of msgDate) as integer) as text) & "-" & text -2 thru -1 of ("0" & (day of msgDate) as text)
        try
          set toAddr to address of to recipients of m as text
        on error
          set toAddr to "(unknown)"
        end try
        set c to content of m
        if (length of c) > 7000 then
          set c to text 1 thru 7000 of c
        end if
        return "From: " & sndr & return & "To: " & toAddr & return & "Date: " & msgDateStr & return & "Subject: " & s & return & "Read: " & readStat & return & return & c
      end try
    end repeat
  end repeat
  return "Message not found with ID: ${msgId}"
end tell
`;

  try {
    return runAS(script, 30000);
  } catch (err) {
    return `Error reading message: ${err.message}`;
  }
}

function listRecent(args) {
  const count = Math.min(args.count || 15, 50);
  const mailbox = args.mailbox || 'inbox';
  const account = args.account || '';

  // Always use mailbox "..." syntax — the bare `inbox` keyword fails on Exchange/IMAP accounts
  const mboxName = mailbox.toLowerCase() === 'inbox' ? 'INBOX' : mailbox;

  let accountFilter = '';
  if (account) {
    accountFilter = `set acctList to {account "${account.replace(/"/g, '\\"')}"}`;
  } else {
    accountFilter = 'set acctList to accounts';
  }

  const script = `
tell application "Mail"
  ${accountFilter}
  set resultList to {}
  set msgCount to 0

  repeat with acct in acctList
    try
      set msgs to messages 1 thru ${count} of mailbox "${mboxName}" of acct
      repeat with m in msgs
        if msgCount ≥ ${count} then exit repeat
        set msgDate to date received of m
        set msgDateStr to (year of msgDate as text) & "-" & text -2 thru -1 of ("0" & ((month of msgDate) as integer) as text) & "-" & text -2 thru -1 of ("0" & (day of msgDate) as text)
        set msgTime to text -2 thru -1 of ("0" & (hours of msgDate) as text) & ":" & text -2 thru -1 of ("0" & (minutes of msgDate) as text)
        set msgInfo to "ID:" & (id of m) & " | " & msgDateStr & " " & msgTime & " | From: " & (sender of m) & " | Subject: " & (subject of m)
        if not (read status of m) then set msgInfo to "[UNREAD] " & msgInfo
        set end of resultList to msgInfo
        set msgCount to msgCount + 1
      end repeat
    end try
    if msgCount ≥ ${count} then exit repeat
  end repeat

  if (count of resultList) = 0 then
    return "No messages found in ${mboxName}"
  end if

  set AppleScript's text item delimiters to "\\n"
  return resultList as text
end tell
`;

  try {
    return runAS(script, 30000);
  } catch (err) {
    return `Error listing mail: ${err.message}`;
  }
}

function listMailboxes(args) {
  const account = args.account || '';

  let accountFilter = '';
  if (account) {
    accountFilter = `set acctList to {account "${account.replace(/"/g, '\\"')}"}`;
  } else {
    accountFilter = 'set acctList to accounts';
  }

  const script = `
tell application "Mail"
  ${accountFilter}
  set resultList to {}

  repeat with acct in acctList
    set acctName to name of acct
    set mboxes to mailboxes of acct
    repeat with mbox in mboxes
      set mboxName to name of mbox
      set msgCount to count of messages of mbox
      set end of resultList to acctName & " / " & mboxName & " (" & msgCount & " messages)"
    end repeat
  end repeat

  set AppleScript's text item delimiters to "\\n"
  return resultList as text
end tell
`;

  try {
    return runAS(script, 30000);
  } catch (err) {
    return `Error listing mailboxes: ${err.message}`;
  }
}

function listAccounts() {
  const script = `
tell application "Mail"
  set resultList to {}
  repeat with acct in accounts
    set acctInfo to (name of acct) & " | " & (user name of acct) & " | Enabled: " & (enabled of acct)
    set end of resultList to acctInfo
  end repeat
  set AppleScript's text item delimiters to "\\n"
  return resultList as text
end tell
`;

  try {
    return runAS(script);
  } catch (err) {
    return `Error listing accounts: ${err.message}`;
  }
}

module.exports = { mail };
