// PRE Web GUI — Reminders/Tasks integration (macOS Reminders.app + Windows Outlook Tasks)
// macOS: Zero-config reminders via EventKit (compiled Swift) — works with iCloud or any configured account
// Windows: Outlook Tasks COM automation via PowerShell — works with any account configured in Outlook

const { execSync } = require('child_process');
const fs = require('fs');
const { IS_WIN, IS_MAC } = require('../platform');

const BIN_PATH = '/tmp/pre-reminders';
const SRC_PATH = '/tmp/pre-reminders.swift';

/**
 * Ensure the Swift helper binary is compiled.
 */
function ensureBinary() {
  if (fs.existsSync(BIN_PATH)) return true;
  fs.writeFileSync(SRC_PATH, REMINDERS_SWIFT);
  try {
    execSync(`swiftc -O -o ${BIN_PATH} ${SRC_PATH} -framework EventKit`, {
      timeout: 60000, encoding: 'utf-8',
    });
    return true;
  } catch (err) {
    return `Error compiling reminders helper: ${(err.stderr || err.message).trim().split('\n').pop()}`;
  }
}

function runBin(flags) {
  const ready = ensureBinary();
  if (typeof ready === 'string') return ready;

  try {
    const output = execSync(`${BIN_PATH} ${flags}`, {
      encoding: 'utf-8',
      timeout: 10000,
      maxBuffer: 256 * 1024,
    }).trim();
    return output;
  } catch (err) {
    // If binary is stale/corrupt, remove and let it recompile next time
    try { fs.unlinkSync(BIN_PATH); } catch {}
    return `Error: ${(err.stderr || err.message).trim().split('\n').pop()}`;
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
 * Reminders tool dispatcher.
 * @param {Object} args - { action, title, notes, list, due, priority, id, query, count, completed }
 */
async function reminders(args) {
  const action = args?.action;
  if (!action) return 'Error: action required. Actions: add, list, complete, search, list_lists, delete';

  if (IS_WIN) {
    switch (action) {
      case 'add': return winAddTask(args);
      case 'list': return winListTasks(args);
      case 'complete': return winCompleteTask(args);
      case 'search': return winSearchTasks(args);
      case 'list_lists': return winListTaskFolders();
      case 'delete': return winDeleteTask(args);
      default:
        return `Error: unknown action '${action}'. Available: add, list, complete, search, list_lists, delete`;
    }
  }

  if (IS_MAC) {
    switch (action) {
      case 'add': return addReminder(args);
      case 'list': return listReminders(args);
      case 'complete': return completeReminder(args);
      case 'search': return searchReminders(args);
      case 'list_lists': return listLists();
      case 'delete': return deleteReminder(args);
      default:
        return `Error: unknown action '${action}'. Available: add, list, complete, search, list_lists, delete`;
    }
  }

  return 'Error: reminders tool is only supported on macOS (Reminders.app) and Windows (Outlook Tasks)';
}

function addReminder(args) {
  const title = args.title;
  const notes = args.notes || '';
  const listName = args.list || '';
  const due = args.due || '';
  const priority = args.priority || '';

  if (!title) return 'Error: "title" is required';

  // Use the Swift binary for adding — AppleScript is too slow
  let flags = `--add "${title.replace(/"/g, '\\"')}"`;
  if (notes) flags += ` --notes "${notes.replace(/"/g, '\\"')}"`;
  if (listName) flags += ` --list "${listName.replace(/"/g, '\\"')}"`;
  if (due) flags += ` --due "${due.replace(/"/g, '\\"')}"`;
  if (priority) flags += ` --priority ${priority}`;

  return runBin(flags);
}

function listReminders(args) {
  const listName = args.list || '';
  const count = Math.min(args.count || 25, 100);
  const showCompleted = args.completed === true || args.completed === 'true';

  let flags = `--list-reminders --count ${count}`;
  if (listName) flags += ` --list "${listName.replace(/"/g, '\\"')}"`;
  if (showCompleted) flags += ' --show-completed';

  const result = runBin(flags);
  return result || 'No reminders found';
}

function completeReminder(args) {
  const id = args.id;
  const title = args.title;
  if (!id && !title) return 'Error: "id" or "title" is required';

  let flags = '--complete';
  if (id) flags += ` --id "${id.replace(/"/g, '\\"')}"`;
  else flags += ` --title "${title.replace(/"/g, '\\"')}"`;

  return runBin(flags);
}

function searchReminders(args) {
  const query = args.query;
  const count = Math.min(args.count || 20, 50);
  if (!query) return 'Error: "query" is required for search';

  let flags = `--search "${query.replace(/"/g, '\\"')}" --count ${count}`;
  return runBin(flags) || `No reminders found matching: ${query}`;
}

function listLists() {
  return runBin('--list-lists') || 'No reminder lists found';
}

function deleteReminder(args) {
  const id = args.id;
  const title = args.title;
  if (!id && !title) return 'Error: "id" or "title" is required';

  let flags = '--delete';
  if (id) flags += ` --id "${id.replace(/"/g, '\\"')}"`;
  else flags += ` --title "${title.replace(/"/g, '\\"')}"`;

  return runBin(flags);
}

// Swift source for the compiled reminders helper
const REMINDERS_SWIFT = `
import EventKit
import Foundation

let store = EKEventStore()
let sem = DispatchSemaphore(value: 0)

if #available(macOS 14.0, *) {
    store.requestFullAccessToReminders { granted, error in sem.signal() }
} else {
    store.requestAccess(to: .reminder) { granted, error in sem.signal() }
}
sem.wait()

// Parse arguments
var args = Array(CommandLine.arguments.dropFirst())
var mode = ""
var listName = ""
var query = ""
var title = ""
var notes = ""
var due = ""
var priority = ""
var id = ""
var count = 25
var showCompleted = false

while !args.isEmpty {
    let arg = args.removeFirst()
    switch arg {
    case "--list-reminders": mode = "list-reminders"
    case "--list-lists": mode = "list-lists"
    case "--search":
        mode = "search"
        if !args.isEmpty { query = args.removeFirst() }
    case "--add":
        mode = "add"
        if !args.isEmpty { title = args.removeFirst() }
    case "--complete": mode = "complete"
    case "--delete": mode = "delete"
    case "--list":
        if !args.isEmpty { listName = args.removeFirst() }
    case "--title":
        if !args.isEmpty { title = args.removeFirst() }
    case "--notes":
        if !args.isEmpty { notes = args.removeFirst() }
    case "--due":
        if !args.isEmpty { due = args.removeFirst() }
    case "--priority":
        if !args.isEmpty { priority = args.removeFirst() }
    case "--id":
        if !args.isEmpty { id = args.removeFirst() }
    case "--count":
        if !args.isEmpty { count = Int(args.removeFirst()) ?? 25 }
    case "--show-completed":
        showCompleted = true
    default: break
    }
}

let df = DateFormatter()
df.dateFormat = "yyyy-MM-dd HH:mm"
let dayFmt = DateFormatter()
dayFmt.dateFormat = "yyyy-MM-dd"

func priorityStr(_ p: Int) -> String {
    switch p {
    case 1: return "HIGH"
    case 5: return "MEDIUM"
    case 9: return "LOW"
    default: return ""
    }
}

func findCalendar(named name: String) -> EKCalendar? {
    let calendars = store.calendars(for: .reminder)
    return calendars.first { $0.title.lowercased() == name.lowercased() }
}

switch mode {
case "list-lists":
    let calendars = store.calendars(for: .reminder)
    for cal in calendars {
        // Fetch incomplete count
        let pred = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: [cal])
        var incomplete = 0
        let s = DispatchSemaphore(value: 0)
        store.fetchReminders(matching: pred) { rems in
            incomplete = rems?.count ?? 0
            s.signal()
        }
        s.wait()
        print("\\(cal.title) | \\(incomplete) pending | Source: \\(cal.source.title)")
    }

case "list-reminders":
    var cals: [EKCalendar]? = nil
    if !listName.isEmpty {
        if let cal = findCalendar(named: listName) {
            cals = [cal]
        } else {
            print("List not found: \\(listName)")
            exit(0)
        }
    }

    let pred: NSPredicate
    if showCompleted {
        // Get both complete and incomplete
        let p1 = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: cals)
        let p2 = store.predicateForCompletedReminders(withCompletionDateStarting: nil, ending: nil, calendars: cals)
        pred = NSCompoundPredicate(orPredicateWithSubpredicates: [p1, p2])
    } else {
        pred = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: cals)
    }

    let s = DispatchSemaphore(value: 0)
    var results: [EKReminder] = []
    store.fetchReminders(matching: pred) { rems in
        results = rems ?? []
        s.signal()
    }
    s.wait()

    // Sort by due date (nil dates last), then by creation date
    results.sort {
        let d0 = $0.dueDateComponents?.date
        let d1 = $1.dueDateComponents?.date
        if let d0 = d0, let d1 = d1 { return d0 < d1 }
        if d0 != nil { return true }
        if d1 != nil { return false }
        return ($0.creationDate ?? Date.distantPast) > ($1.creationDate ?? Date.distantPast)
    }

    for r in results.prefix(count) {
        var line = ""
        if r.isCompleted { line += "[DONE] " }
        line += "ID:\\(r.calendarItemIdentifier) | \\(r.title ?? "(no title)")"
        if let dc = r.dueDateComponents, let d = dc.date {
            line += " | Due: \\(dayFmt.string(from: d))"
        }
        let p = priorityStr(Int(r.priority))
        if !p.isEmpty { line += " | Priority: \\(p)" }
        line += " | [\\(r.calendar.title)]"
        if let body = r.notes, !body.isEmpty {
            let preview = String(body.prefix(100))
            line += " | Notes: \\(preview)"
        }
        print(line)
    }

    if results.isEmpty {
        print("No reminders found")
    }

case "search":
    let pred = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: nil)
    let s = DispatchSemaphore(value: 0)
    var results: [EKReminder] = []
    store.fetchReminders(matching: pred) { rems in
        results = rems ?? []
        s.signal()
    }
    s.wait()

    let q = query.lowercased()
    let matched = results.filter {
        ($0.title ?? "").lowercased().contains(q) ||
        ($0.notes ?? "").lowercased().contains(q)
    }

    for r in matched.prefix(count) {
        var line = ""
        if r.isCompleted { line += "[DONE] " }
        line += "ID:\\(r.calendarItemIdentifier) | \\(r.title ?? "(no title)") | [\\(r.calendar.title)]"
        if let dc = r.dueDateComponents, let d = dc.date {
            line += " | Due: \\(dayFmt.string(from: d))"
        }
        print(line)
    }

    if matched.isEmpty {
        print("No reminders found matching: \\(query)")
    }

case "add":
    guard !title.isEmpty else {
        print("Error: title is required")
        exit(1)
    }

    let reminder = EKReminder(eventStore: store)
    reminder.title = title
    if !notes.isEmpty { reminder.notes = notes }

    // Calendar (list)
    if !listName.isEmpty, let cal = findCalendar(named: listName) {
        reminder.calendar = cal
    } else {
        reminder.calendar = store.defaultCalendarForNewReminders()
    }

    // Due date
    if !due.isEmpty {
        let parseFmts = ["yyyy-MM-dd HH:mm", "yyyy-MM-dd", "MMMM d, yyyy h:mm:ss a", "MMMM d, yyyy h:mm a", "MMMM d, yyyy"]
        var parsed: Date? = nil
        for fmt in parseFmts {
            let f = DateFormatter()
            f.dateFormat = fmt
            f.locale = Locale(identifier: "en_US")
            if let d = f.date(from: due) { parsed = d; break }
        }
        if let d = parsed {
            reminder.dueDateComponents = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: d)
        } else {
            print("Warning: Could not parse due date '\\(due)', creating without due date")
        }
    }

    // Priority
    switch priority.lowercased() {
    case "high": reminder.priority = 1
    case "medium": reminder.priority = 5
    case "low": reminder.priority = 9
    default: break
    }

    do {
        try store.save(reminder, commit: true)
        var msg = "Reminder created: \\(title)"
        if !due.isEmpty { msg += " (due: \\(due))" }
        msg += " [\\(reminder.calendar.title)]"
        print(msg)
    } catch {
        print("Error creating reminder: \\(error.localizedDescription)")
    }

case "complete":
    let pred = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: nil)
    let s = DispatchSemaphore(value: 0)
    var results: [EKReminder] = []
    store.fetchReminders(matching: pred) { rems in
        results = rems ?? []
        s.signal()
    }
    s.wait()

    var target: EKReminder? = nil
    if !id.isEmpty {
        target = results.first { $0.calendarItemIdentifier == id }
    } else if !title.isEmpty {
        let t = title.lowercased()
        target = results.first { ($0.title ?? "").lowercased().contains(t) }
    }

    if let r = target {
        r.isCompleted = true
        do {
            try store.save(r, commit: true)
            print("Completed: \\(r.title ?? "(no title)")")
        } catch {
            print("Error completing reminder: \\(error.localizedDescription)")
        }
    } else {
        print("Reminder not found")
    }

case "delete":
    // Search across all reminders (including completed)
    let p1 = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: nil)
    let p2 = store.predicateForCompletedReminders(withCompletionDateStarting: nil, ending: nil, calendars: nil)
    let pred = NSCompoundPredicate(orPredicateWithSubpredicates: [p1, p2])
    let s = DispatchSemaphore(value: 0)
    var results: [EKReminder] = []
    store.fetchReminders(matching: pred) { rems in
        results = rems ?? []
        s.signal()
    }
    s.wait()

    var target: EKReminder? = nil
    if !id.isEmpty {
        target = results.first { $0.calendarItemIdentifier == id }
    } else if !title.isEmpty {
        let t = title.lowercased()
        target = results.first { ($0.title ?? "").lowercased().contains(t) }
    }

    if let r = target {
        let rName = r.title ?? "(no title)"
        do {
            try store.remove(r, commit: true)
            print("Deleted reminder: \\(rName)")
        } catch {
            print("Error deleting reminder: \\(error.localizedDescription)")
        }
    } else {
        print("Reminder not found")
    }

default:
    print("Usage: pre-reminders --list-lists | --list-reminders | --search <query> | --add <title> | --complete | --delete")
}
`;

// ── Windows Outlook Tasks COM implementations ───────────────────────────────

function winAddTask(args) {
  const title = args.title;
  const taskNotes = args.notes || '';
  const due = args.due || '';
  const priority = args.priority || '';

  if (!title) return 'Error: "title" is required';

  let priorityVal = '1'; // olImportanceNormal
  if (priority === 'high') priorityVal = '2'; // olImportanceHigh
  else if (priority === 'low') priorityVal = '0'; // olImportanceLow

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$task = $outlook.CreateItem(3)
$task.Subject = '${escPS(title)}'
${taskNotes ? `$task.Body = '${escPS(taskNotes)}'` : ''}
$task.Importance = ${priorityVal}
${due ? `$task.DueDate = [DateTime]::Parse('${escPS(due)}')` : ''}
$task.Save()
$msg = 'Reminder created: ${escPS(title)}'
${due ? `$msg += " (due: $($task.DueDate.ToString('yyyy-MM-dd')))"` : ''}
Write-Output $msg
`;

  try {
    return runPS(script);
  } catch (err) {
    return `Error creating task: ${err.message}`;
  }
}

function winListTasks(args) {
  const count = Math.min(args.count || 25, 100);
  const showCompleted = args.completed === true || args.completed === 'true';

  const statusFilter = showCompleted
    ? ''
    : `if ($item.Complete) { continue }`;

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace('MAPI')
$folder = $ns.GetDefaultFolder(13)
$results = @()
$matchCount = 0
foreach ($item in $folder.Items) {
  if ($matchCount -ge ${count}) { break }
  ${statusFilter}
  $line = ''
  if ($item.Complete) { $line += '[DONE] ' }
  $line += "ID:$($item.EntryID.Substring(0,20)) | $($item.Subject)"
  if ($item.DueDate -and $item.DueDate.Year -gt 1900 -and $item.DueDate.Year -lt 4000) {
    $line += " | Due: $($item.DueDate.ToString('yyyy-MM-dd'))"
  }
  switch ($item.Importance) {
    2 { $line += ' | Priority: HIGH' }
    0 { $line += ' | Priority: LOW' }
  }
  if ($item.Body) {
    $preview = if ($item.Body.Length -gt 100) { $item.Body.Substring(0,100) } else { $item.Body }
    $preview = $preview -replace '\\r?\\n', ' '
    $line += " | Notes: $preview"
  }
  $results += $line
  $matchCount++
}
if ($results.Count -eq 0) { Write-Output 'No reminders found' }
else { $results | ForEach-Object { Write-Output $_ } }
`;

  try {
    return runPS(script, 15000);
  } catch (err) {
    return `Error listing tasks: ${err.message}`;
  }
}

function winCompleteTask(args) {
  const id = args.id;
  const title = args.title;
  if (!id && !title) return 'Error: "id" or "title" is required';

  const lookupClause = id
    ? `$item = $ns.GetItemFromID('${escPS(id)}')`
    : `$folder = $ns.GetDefaultFolder(13)
$item = $null
$q = '${escPS(title)}'.ToLower()
foreach ($t in $folder.Items) {
  if (-not $t.Complete -and $t.Subject -and $t.Subject.ToLower().Contains($q)) { $item = $t; break }
}
if (-not $item) { Write-Output 'Reminder not found'; exit }`;

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace('MAPI')
${lookupClause}
$item.MarkComplete()
$item.Save()
Write-Output "Completed: $($item.Subject)"
`;

  try {
    return runPS(script);
  } catch (err) {
    return `Error completing task: ${err.message}`;
  }
}

function winSearchTasks(args) {
  const query = args.query;
  const count = Math.min(args.count || 20, 50);
  if (!query) return 'Error: "query" is required for search';

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace('MAPI')
$folder = $ns.GetDefaultFolder(13)
$q = '${escPS(query)}'.ToLower()
$results = @()
$matchCount = 0
foreach ($item in $folder.Items) {
  if ($matchCount -ge ${count}) { break }
  $subj = if ($item.Subject) { $item.Subject.ToLower() } else { '' }
  $body = if ($item.Body) { $item.Body.ToLower() } else { '' }
  if ($subj.Contains($q) -or $body.Contains($q)) {
    $line = ''
    if ($item.Complete) { $line += '[DONE] ' }
    $line += "ID:$($item.EntryID.Substring(0,20)) | $($item.Subject)"
    if ($item.DueDate -and $item.DueDate.Year -gt 1900 -and $item.DueDate.Year -lt 4000) {
      $line += " | Due: $($item.DueDate.ToString('yyyy-MM-dd'))"
    }
    $results += $line
    $matchCount++
  }
}
if ($results.Count -eq 0) { Write-Output 'No reminders found matching: ${escPS(query)}' }
else { $results | ForEach-Object { Write-Output $_ } }
`;

  try {
    return runPS(script, 15000);
  } catch (err) {
    return `Error searching tasks: ${err.message}`;
  }
}

function winListTaskFolders() {
  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace('MAPI')
$folder = $ns.GetDefaultFolder(13)
$results = @()
# Count incomplete tasks in default folder
$incomplete = 0
foreach ($item in $folder.Items) { if (-not $item.Complete) { $incomplete++ } }
$results += "$($folder.Name) | $incomplete pending | Default"
foreach ($sub in $folder.Folders) {
  $subIncomplete = 0
  foreach ($item in $sub.Items) { if (-not $item.Complete) { $subIncomplete++ } }
  $results += "$($sub.Name) | $subIncomplete pending"
}
if ($results.Count -eq 0) { Write-Output 'No task folders found' }
else { $results | ForEach-Object { Write-Output $_ } }
`;

  try {
    return runPS(script);
  } catch (err) {
    return `Error listing task folders: ${err.message}`;
  }
}

function winDeleteTask(args) {
  const id = args.id;
  const title = args.title;
  if (!id && !title) return 'Error: "id" or "title" is required';

  const lookupClause = id
    ? `$item = $ns.GetItemFromID('${escPS(id)}')`
    : `$folder = $ns.GetDefaultFolder(13)
$item = $null
$q = '${escPS(title)}'.ToLower()
foreach ($t in $folder.Items) {
  if ($t.Subject -and $t.Subject.ToLower().Contains($q)) { $item = $t; break }
}
if (-not $item) { Write-Output 'Reminder not found'; exit }`;

  const script = `
$outlook = New-Object -ComObject Outlook.Application
$ns = $outlook.GetNamespace('MAPI')
${lookupClause}
$name = $item.Subject
$item.Delete()
Write-Output "Deleted reminder: $name"
`;

  try {
    return runPS(script);
  } catch (err) {
    return `Error deleting task: ${err.message}`;
  }
}

module.exports = { reminders };
