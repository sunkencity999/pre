// PRE Web GUI — macOS Calendar.app integration
// Zero-config calendar via AppleScript — works with iCloud, Google, Exchange, or any configured account

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
 * Calendar tool dispatcher.
 * @param {Object} args - { action, title, start, end, location, notes, calendar, query, days, id }
 */
async function calendar(args) {
  const action = args?.action;
  if (!action) return 'Error: action required. Actions: list_events, create_event, search, list_calendars, delete_event, today, week';

  switch (action) {
    case 'today': return listEvents({ ...args, days: 1 });
    case 'week': return listEvents({ ...args, days: 7 });
    case 'list_events': return listEvents(args);
    case 'create_event': return createEvent(args);
    case 'search': return searchEvents(args);
    case 'list_calendars': return listCalendars();
    case 'delete_event': return deleteEvent(args);
    default:
      return `Error: unknown action '${action}'. Available: list_events, create_event, search, list_calendars, delete_event, today, week`;
  }
}

function listEvents(args) {
  const days = Math.min(args.days || 7, 90);
  const calName = args.calendar || '';

  // Use EventKit via Swift — dramatically faster than AppleScript for date queries
  return listEventsSwift(days, calName);
}

/**
 * Fast event listing via EventKit (compiled Swift binary).
 * Compiles once to /tmp/pre-cal-events, reuses on subsequent calls.
 */
function listEventsSwift(days, calName) {
  const { execSync } = require('child_process');
  const fs = require('fs');
  const binPath = '/tmp/pre-cal-events';
  const srcPath = '/tmp/pre-cal-events.swift';

  // Compile the helper binary if missing
  if (!fs.existsSync(binPath)) {
    fs.writeFileSync(srcPath, CAL_EVENTS_SWIFT);
    try {
      execSync(`swiftc -O -o ${binPath} ${srcPath} -framework EventKit`, {
        timeout: 60000, encoding: 'utf-8',
      });
    } catch (err) {
      return `Error compiling calendar helper: ${(err.stderr || err.message).trim().split('\n').pop()}`;
    }
  }

  try {
    let cmd = `${binPath} --days ${days}`;
    if (calName) cmd += ` --calendar "${calName}"`;
    const output = execSync(cmd, {
      encoding: 'utf-8',
      timeout: 10000,
      maxBuffer: 256 * 1024,
    }).trim();
    return output || `No events found in the next ${days} day(s)`;
  } catch (err) {
    // If binary is stale/corrupt, remove and let it recompile next time
    try { fs.unlinkSync(binPath); } catch {}
    return `Error listing events: ${(err.stderr || err.message).trim().split('\n').pop()}`;
  }
}

// Swift source for the compiled calendar helper
const CAL_EVENTS_SWIFT = `
import EventKit
import Foundation

let store = EKEventStore()
let sem = DispatchSemaphore(value: 0)

if #available(macOS 14.0, *) {
    store.requestFullAccessToEvents { granted, error in sem.signal() }
} else {
    store.requestAccess(to: .event) { granted, error in sem.signal() }
}
sem.wait()

var days = 7
var targetCal = ""
var mode = "list" // list or search
var query = ""

var args = CommandLine.arguments.dropFirst()
while let arg = args.first {
    args = args.dropFirst()
    switch arg {
    case "--days":
        if let next = args.first { days = Int(next) ?? 7; args = args.dropFirst() }
    case "--calendar":
        if let next = args.first { targetCal = next; args = args.dropFirst() }
    case "--search":
        mode = "search"
        if let next = args.first { query = next; args = args.dropFirst() }
    default: break
    }
}

let now = Calendar.current.startOfDay(for: Date())
let startDate: Date
let endDate: Date

if mode == "search" {
    startDate = Calendar.current.date(byAdding: .day, value: -days, to: now)!
    endDate = Calendar.current.date(byAdding: .day, value: days, to: now)!
} else {
    startDate = now
    endDate = Calendar.current.date(byAdding: .day, value: days, to: now)!
}

let predicate = store.predicateForEvents(withStart: startDate, end: endDate, calendars: nil)
var events = store.events(matching: predicate).sorted { $0.startDate < $1.startDate }

if !targetCal.isEmpty {
    events = events.filter { $0.calendar.title == targetCal }
}

if mode == "search" && !query.isEmpty {
    let q = query.lowercased()
    events = events.filter {
        ($0.title ?? "").lowercased().contains(q) ||
        ($0.location ?? "").lowercased().contains(q)
    }
}

let df = DateFormatter()
df.dateFormat = "yyyy-MM-dd HH:mm"
let tf = DateFormatter()
tf.dateFormat = "HH:mm"

for e in events {
    var line = "UID:\\(e.calendarItemIdentifier) | \\(df.string(from: e.startDate))-\\(tf.string(from: e.endDate)) | \\(e.title ?? "(no title)")"
    if let loc = e.location, !loc.isEmpty { line += " | @ \\(loc)" }
    line += " | [\\(e.calendar.title)]"
    print(line)
}

if events.isEmpty {
    print("No events found")
}
`;

function createEvent(args) {
  const title = args.title;
  const startTime = args.start;
  const endTime = args.end;
  const location = args.location || '';
  const notes = args.notes || '';
  const calName = args.calendar || '';

  if (!title) return 'Error: "title" is required';
  if (!startTime) return 'Error: "start" is required (e.g. "2026-04-15 10:00")';

  const escTitle = title.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  const escLocation = location.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  const escNotes = notes.replace(/\\/g, '\\\\').replace(/"/g, '\\"');

  // Parse start/end times — accept ISO-ish formats
  // AppleScript needs: date "April 15, 2026 10:00:00 AM"
  // We'll use a flexible approach: pass to osascript as a parseable string
  const calTarget = calName
    ? `calendar "${calName.replace(/"/g, '\\"')}"`
    : 'first calendar';

  let endDateClause = '';
  if (endTime) {
    endDateClause = `set end date of newEvent to date "${endTime}"`;
  } else {
    // Default: 1 hour duration
    endDateClause = 'set end date of newEvent to (start date of newEvent) + 3600';
  }

  const script = `
tell application "Calendar"
  tell ${calTarget}
    set newEvent to make new event with properties {summary:"${escTitle}", start date:date "${startTime}"}
    ${endDateClause}
    ${location ? `set location of newEvent to "${escLocation}"` : ''}
    ${notes ? `set description of newEvent to "${escNotes}"` : ''}
    set eventUID to uid of newEvent
  end tell
  return "Event created: ${escTitle}\\nUID: " & eventUID
end tell
`;

  try {
    return runAS(script);
  } catch (err) {
    // Common error: date parsing failure. Provide helpful guidance.
    if (err.message.includes('date') || err.message.includes('Can')) {
      return `Error creating event: Could not parse date "${startTime}". Use a format like "April 15, 2026 10:00:00 AM" or "2026-04-15 10:00 AM". Original error: ${err.message}`;
    }
    return `Error creating event: ${err.message}`;
  }
}

function searchEvents(args) {
  const query = args.query;
  const days = Math.min(args.days || 30, 365);

  if (!query) return 'Error: "query" is required for search';

  // Reuse the compiled EventKit binary with --search flag
  const { execSync } = require('child_process');
  const fs = require('fs');
  const binPath = '/tmp/pre-cal-events';

  // Ensure binary exists (compile if needed)
  if (!fs.existsSync(binPath)) {
    // Trigger compilation via listEvents
    listEventsSwift(1, '');
  }

  try {
    const cmd = `${binPath} --days ${days} --search "${query.replace(/"/g, '\\"')}"`;
    const output = execSync(cmd, {
      encoding: 'utf-8',
      timeout: 10000,
      maxBuffer: 256 * 1024,
    }).trim();
    return output || `No events found matching: ${query}`;
  } catch (err) {
    return `Error searching calendar: ${(err.stderr || err.message).trim().split('\n').pop()}`;
  }
}

function listCalendars() {
  const script = `
tell application "Calendar"
  set resultList to {}
  repeat with cal in calendars
    set calInfo to (name of cal) & " | Color: " & (color of cal) & " | Writable: " & (writable of cal)
    try
      set calInfo to calInfo & " | Description: " & (description of cal)
    end try
    set end of resultList to calInfo
  end repeat
  set AppleScript's text item delimiters to "\\n"
  return resultList as text
end tell
`;

  try {
    return runAS(script);
  } catch (err) {
    return `Error listing calendars: ${err.message}`;
  }
}

function deleteEvent(args) {
  const uid = args.id || args.uid;
  if (!uid) return 'Error: "id" (event UID) is required';

  const script = `
tell application "Calendar"
  repeat with cal in calendars
    try
      set evts to (every event of cal whose uid is "${uid}")
      repeat with e in evts
        set evtName to summary of e
        delete e
        return "Deleted event: " & evtName
      end repeat
    end try
  end repeat
  return "Event not found with UID: ${uid}"
end tell
`;

  try {
    return runAS(script);
  } catch (err) {
    return `Error deleting event: ${err.message}`;
  }
}

module.exports = { calendar };
