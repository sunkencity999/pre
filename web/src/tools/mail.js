// PRE Web GUI — macOS Mail.app integration
// Zero-config email via AppleScript — works with any account configured in Mail.app

const { execSync } = require('child_process');

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
 * Mail tool dispatcher.
 * @param {Object} args - { action, to, cc, bcc, subject, body, query, id, count, mailbox, account }
 */
async function mail(args) {
  const action = args?.action;
  if (!action) return 'Error: action required. Actions: send, draft, search, read, list_recent, list_mailboxes, list_accounts';

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
  const mailboxRef = mailbox.toLowerCase() === 'inbox' ? 'inbox' : `mailbox "${mailbox}"`;

  // Search by subject and sender — AppleScript's "whose" clause
  const script = `
tell application "Mail"
  set resultList to {}
  set matchCount to 0
  set maxResults to ${count}

  -- Search across all accounts
  repeat with acct in accounts
    try
      set msgs to messages of ${mailboxRef} of acct whose subject contains "${escQuery}" or sender contains "${escQuery}"
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
        set msgDate to date received of m
        set msgDateStr to (year of msgDate as text) & "-" & text -2 thru -1 of ("0" & ((month of msgDate) as integer) as text) & "-" & text -2 thru -1 of ("0" & (day of msgDate) as text)
        set msgInfo to "From: " & (sender of m) & "\\nTo: " & (address of to recipients of m as text) & "\\nDate: " & msgDateStr & "\\nSubject: " & (subject of m) & "\\nRead: " & (read status of m) & "\\n\\n" & (content of m)
        return text 1 thru (min of {(length of msgInfo), 8000}) of msgInfo
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

  const mailboxRef = mailbox.toLowerCase() === 'inbox' ? 'inbox' : `mailbox "${mailbox}"`;

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
      set msgs to messages 1 thru ${count} of ${mailboxRef} of acct
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
    return "No messages found in ${mailbox}"
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
