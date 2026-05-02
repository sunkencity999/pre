// PRE Web GUI — GNOME Evolution Data Server (EDS) integration via gdbus
// Provides calendar, contacts, and reminders/tasks on Linux desktops
// that use GNOME/EDS (Ubuntu, Fedora GNOME, etc.)

const { execSync } = require('child_process');

// ── EDS availability check ──────────────────────────────────────────────

let _edsAvailable = null;

function isEdsAvailable() {
  if (_edsAvailable !== null) return _edsAvailable;
  try {
    execSync('gdbus introspect --session --dest org.gnome.evolution.dataserver.Sources5 --object-path / 2>/dev/null', {
      encoding: 'utf-8', timeout: 5000,
    });
    _edsAvailable = true;
  } catch {
    _edsAvailable = false;
  }
  return _edsAvailable;
}

const EDS_UNAVAILABLE = 'Calendar/Contacts/Reminders integration requires GNOME Evolution Data Server.\n'
  + 'Install: sudo apt install evolution-data-server (Debian/Ubuntu)\n'
  + '         sudo dnf install evolution-data-server (Fedora)';

function run(cmd, timeout = 15000) {
  try {
    return execSync(cmd, { encoding: 'utf-8', timeout, maxBuffer: 256 * 1024 }).trim();
  } catch {
    return '';
  }
}

// ── Calendar ────────────────────────────────────────────────────────────

/**
 * List calendar events using `gcalcli` or `gdbus` + EDS.
 * We try gcalcli first (user-friendly, widely available), then raw EDS.
 */
function edsCalendarList(days = 7, calendarFilter) {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;

  // Use gnome-calendar-cli or direct EDS query
  const endDate = new Date();
  endDate.setDate(endDate.getDate() + days);
  const startISO = new Date().toISOString();
  const endISO = endDate.toISOString();

  // Try fetching via D-Bus ECalClient
  // EDS exposes calendar objects under /org/gnome/evolution/dataserver/Calendar/*
  // We'll use a Python one-liner since gdbus for EDS calendar queries is complex
  // Fallback: parse ical files from ~/.local/share/evolution/calendar/
  const calDir = `${process.env.HOME}/.local/share/evolution/calendar/local`;
  const output = run(`find ${calDir} -name '*.ics' -newer /dev/null 2>/dev/null | head -50`);

  if (!output) {
    // Try gnome-calendar or flatpak path
    const flatpakDir = `${process.env.HOME}/.local/share/gnome-calendar/local-cal`;
    const fpOutput = run(`find ${flatpakDir} -name '*.ics' 2>/dev/null | head -50`);
    if (!fpOutput) {
      return 'No calendar events found. Ensure you have calendars configured in GNOME Online Accounts or Evolution.';
    }
    return parseIcsFiles(fpOutput.split('\n'), days, calendarFilter);
  }

  return parseIcsFiles(output.split('\n'), days, calendarFilter);
}

function parseIcsFiles(files, days, calendarFilter) {
  const now = new Date();
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() + days);

  const events = [];
  for (const file of files) {
    if (!file) continue;
    try {
      const content = require('fs').readFileSync(file, 'utf-8');
      const vevents = content.split('BEGIN:VEVENT');
      for (let i = 1; i < vevents.length; i++) {
        const block = vevents[i].split('END:VEVENT')[0];
        const uid = extractIcalField(block, 'UID') || `evt-${i}`;
        const summary = extractIcalField(block, 'SUMMARY') || '(no title)';
        const dtstart = parseIcalDate(extractIcalField(block, 'DTSTART'));
        const dtend = parseIcalDate(extractIcalField(block, 'DTEND'));
        const location = extractIcalField(block, 'LOCATION') || '';
        const calName = require('path').basename(require('path').dirname(file));

        if (calendarFilter && !calName.toLowerCase().includes(calendarFilter.toLowerCase())) continue;
        if (dtstart && dtstart >= now && dtstart <= cutoff) {
          const startStr = formatDateTime(dtstart);
          const endStr = dtend ? formatTime(dtend) : '';
          events.push(`UID:${uid} | ${startStr}${endStr ? '-' + endStr : ''} | ${summary}${location ? ' | @ ' + location : ''} | [${calName}]`);
        }
      }
    } catch { /* skip unreadable files */ }
  }

  events.sort();
  return events.length > 0 ? events.join('\n') : `No events in the next ${days} days.`;
}

function edsCalendarCreate(title, start, end, location, notes, calendar) {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;

  const uid = `pre-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const startDate = new Date(start);
  const endDate = end ? new Date(end) : new Date(startDate.getTime() + 3600000);

  const vevent = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//PRE//EN',
    'BEGIN:VEVENT',
    `UID:${uid}`,
    `DTSTART:${toIcalDate(startDate)}`,
    `DTEND:${toIcalDate(endDate)}`,
    `SUMMARY:${title}`,
    location ? `LOCATION:${location}` : '',
    notes ? `DESCRIPTION:${notes}` : '',
    `DTSTAMP:${toIcalDate(new Date())}`,
    'END:VEVENT',
    'END:VCALENDAR',
  ].filter(Boolean).join('\r\n');

  // Write to EDS local calendar directory
  const calDir = `${process.env.HOME}/.local/share/evolution/calendar/local/system`;
  try {
    const fs = require('fs');
    if (!fs.existsSync(calDir)) fs.mkdirSync(calDir, { recursive: true });
    fs.writeFileSync(`${calDir}/${uid}.ics`, vevent);
    return `Event created: ${title}\nUID: ${uid}`;
  } catch (err) {
    return `Error creating event: ${err.message}`;
  }
}

function edsCalendarDelete(uid) {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;

  const calDir = `${process.env.HOME}/.local/share/evolution/calendar/local`;
  const file = run(`find ${calDir} -name '${uid}.ics' 2>/dev/null | head -1`);
  if (!file) return `Event not found: ${uid}`;

  try {
    require('fs').unlinkSync(file);
    return `Deleted event: ${uid}`;
  } catch (err) {
    return `Error deleting event: ${err.message}`;
  }
}

function edsCalendarSearch(query, days = 365) {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;
  const events = edsCalendarList(days);
  if (events.startsWith('No ') || events.startsWith('Calendar/')) return events;
  const lines = events.split('\n').filter(l => l.toLowerCase().includes(query.toLowerCase()));
  return lines.length > 0 ? lines.join('\n') : `No events matching: ${query}`;
}

function edsListCalendars() {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;

  const calDir = `${process.env.HOME}/.local/share/evolution/calendar/local`;
  const dirs = run(`ls -d ${calDir}/*/ 2>/dev/null`);
  if (!dirs) return 'No local calendars found.';
  return dirs.split('\n').filter(Boolean).map(d => {
    const name = require('path').basename(d.replace(/\/$/, ''));
    const count = run(`find "${d}" -name '*.ics' 2>/dev/null | wc -l`).trim();
    return `${name} | ${count} events | Writable: true`;
  }).join('\n');
}

// ── Contacts ────────────────────────────────────────────────────────────

function edsContactSearch(query, count = 20) {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;

  const abDir = `${process.env.HOME}/.local/share/evolution/addressbook/local/system`;
  const files = run(`find ${abDir} -name '*.vcf' 2>/dev/null | head -200`);
  if (!files) return 'No contacts found. Ensure you have contacts configured in GNOME Online Accounts or Evolution.';

  const results = [];
  for (const file of files.split('\n')) {
    if (!file || results.length >= count) break;
    try {
      const content = require('fs').readFileSync(file, 'utf-8');
      if (!content.toLowerCase().includes(query.toLowerCase())) continue;

      const fn = extractVcardField(content, 'FN') || '';
      const email = extractVcardField(content, 'EMAIL') || '';
      const tel = extractVcardField(content, 'TEL') || '';
      const org = extractVcardField(content, 'ORG') || '';
      const uid = extractVcardField(content, 'UID') || require('path').basename(file, '.vcf');

      results.push(`ID:${uid} | ${fn}${email ? ' | Email: ' + email : ''}${tel ? ' | Phone: ' + tel : ''}${org ? ' | Org: ' + org : ''}`);
    } catch { /* skip */ }
  }

  return results.length > 0 ? results.join('\n') : `No contacts matching: ${query}`;
}

function edsContactRead(id) {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;

  const abDir = `${process.env.HOME}/.local/share/evolution/addressbook/local`;
  const file = run(`find ${abDir} -name '${id}.vcf' 2>/dev/null | head -1`);
  if (!file) {
    // Try searching by content
    const allFiles = run(`find ${abDir} -name '*.vcf' 2>/dev/null`);
    for (const f of (allFiles || '').split('\n')) {
      if (!f) continue;
      try {
        const content = require('fs').readFileSync(f, 'utf-8');
        if (content.includes(id)) {
          return formatVcard(content);
        }
      } catch { /* skip */ }
    }
    return `Contact not found: ${id}`;
  }

  try {
    const content = require('fs').readFileSync(file, 'utf-8');
    return formatVcard(content);
  } catch (err) {
    return `Error reading contact: ${err.message}`;
  }
}

function edsContactCount() {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;
  const abDir = `${process.env.HOME}/.local/share/evolution/addressbook/local`;
  const count = run(`find ${abDir} -name '*.vcf' 2>/dev/null | wc -l`).trim();
  return `Total contacts: ${count || 0}`;
}

function edsListGroups() {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;
  const abDir = `${process.env.HOME}/.local/share/evolution/addressbook/local`;
  const dirs = run(`ls -d ${abDir}/*/ 2>/dev/null`);
  if (!dirs) return 'No address books found.';
  return dirs.split('\n').filter(Boolean).map(d => {
    const name = require('path').basename(d.replace(/\/$/, ''));
    const count = run(`find "${d}" -name '*.vcf' 2>/dev/null | wc -l`).trim();
    return `${name} (${count} contacts)`;
  }).join('\n');
}

// ── Reminders/Tasks ─────────────────────────────────────────────────────

function edsTaskList(listFilter, showCompleted, count = 25) {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;

  const taskDir = `${process.env.HOME}/.local/share/evolution/tasks/local`;
  const files = run(`find ${taskDir} -name '*.ics' 2>/dev/null | head -200`);
  if (!files) return 'No reminders/tasks found. Ensure you have task lists configured in GNOME Online Accounts or Evolution.';

  const results = [];
  for (const file of files.split('\n')) {
    if (!file || results.length >= count) break;
    try {
      const content = require('fs').readFileSync(file, 'utf-8');
      if (!content.includes('VTODO')) continue;

      const blocks = content.split('BEGIN:VTODO');
      for (let i = 1; i < blocks.length; i++) {
        if (results.length >= count) break;
        const block = blocks[i].split('END:VTODO')[0];
        const uid = extractIcalField(block, 'UID') || `task-${i}`;
        const summary = extractIcalField(block, 'SUMMARY') || '(no title)';
        const status = extractIcalField(block, 'STATUS') || '';
        const due = extractIcalField(block, 'DUE');
        const priority = extractIcalField(block, 'PRIORITY');
        const notesField = extractIcalField(block, 'DESCRIPTION') || '';
        const listName = require('path').basename(require('path').dirname(file));

        const isDone = status === 'COMPLETED';
        if (!showCompleted && isDone) continue;
        if (listFilter && !listName.toLowerCase().includes(listFilter.toLowerCase())) continue;

        const priLabel = priority === '1' ? 'HIGH' : priority === '5' ? 'MEDIUM' : priority === '9' ? 'LOW' : '';
        const dueStr = due ? formatDateTime(parseIcalDate(due)) : '';
        const doneTag = isDone ? '[DONE] ' : '';
        results.push(`${doneTag}ID:${uid} | ${summary}${dueStr ? ' | Due: ' + dueStr : ''}${priLabel ? ' | Priority: ' + priLabel : ''} | [${listName}]${notesField ? ' | Notes: ' + notesField.slice(0, 100) : ''}`);
      }
    } catch { /* skip */ }
  }

  return results.length > 0 ? results.join('\n') : 'No reminders/tasks found.';
}

function edsTaskCreate(title, notes, list, due, priority) {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;

  const uid = `pre-task-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const priMap = { high: '1', medium: '5', low: '9' };
  const priVal = priority ? (priMap[priority.toLowerCase()] || '0') : '0';

  const vtodo = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//PRE//EN',
    'BEGIN:VTODO',
    `UID:${uid}`,
    `SUMMARY:${title}`,
    notes ? `DESCRIPTION:${notes}` : '',
    due ? `DUE:${toIcalDate(new Date(due))}` : '',
    `PRIORITY:${priVal}`,
    'STATUS:NEEDS-ACTION',
    `DTSTAMP:${toIcalDate(new Date())}`,
    'END:VTODO',
    'END:VCALENDAR',
  ].filter(Boolean).join('\r\n');

  const listDir = list
    ? `${process.env.HOME}/.local/share/evolution/tasks/local/${list}`
    : `${process.env.HOME}/.local/share/evolution/tasks/local/system`;

  try {
    const fs = require('fs');
    if (!fs.existsSync(listDir)) fs.mkdirSync(listDir, { recursive: true });
    fs.writeFileSync(`${listDir}/${uid}.ics`, vtodo);
    const dueStr = due ? ` (due: ${due})` : '';
    return `Reminder created: ${title}${dueStr} [${list || 'system'}]`;
  } catch (err) {
    return `Error creating reminder: ${err.message}`;
  }
}

function edsTaskComplete(id) {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;

  const taskDir = `${process.env.HOME}/.local/share/evolution/tasks/local`;
  const file = findIcsFileByUid(taskDir, id);
  if (!file) return `Reminder not found: ${id}`;

  try {
    const fs = require('fs');
    let content = fs.readFileSync(file, 'utf-8');
    content = content.replace(/STATUS:.*/, 'STATUS:COMPLETED');
    if (!content.includes('COMPLETED:')) {
      content = content.replace('END:VTODO', `COMPLETED:${toIcalDate(new Date())}\nEND:VTODO`);
    }
    fs.writeFileSync(file, content);
    const summary = extractIcalField(content, 'SUMMARY') || id;
    return `Completed: ${summary}`;
  } catch (err) {
    return `Error completing reminder: ${err.message}`;
  }
}

function edsTaskDelete(id) {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;

  const taskDir = `${process.env.HOME}/.local/share/evolution/tasks/local`;
  const file = findIcsFileByUid(taskDir, id);
  if (!file) return `Reminder not found: ${id}`;

  try {
    const content = require('fs').readFileSync(file, 'utf-8');
    const summary = extractIcalField(content, 'SUMMARY') || id;
    require('fs').unlinkSync(file);
    return `Deleted reminder: ${summary}`;
  } catch (err) {
    return `Error deleting reminder: ${err.message}`;
  }
}

function edsTaskSearch(query, count = 20) {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;
  const all = edsTaskList(null, true, 200);
  if (all.startsWith('No ') || all.startsWith('Calendar/')) return all;
  const lines = all.split('\n').filter(l => l.toLowerCase().includes(query.toLowerCase()));
  return lines.slice(0, count).join('\n') || `No reminders matching: ${query}`;
}

function edsListTaskLists() {
  if (!isEdsAvailable()) return EDS_UNAVAILABLE;
  const taskDir = `${process.env.HOME}/.local/share/evolution/tasks/local`;
  const dirs = run(`ls -d ${taskDir}/*/ 2>/dev/null`);
  if (!dirs) return 'No task lists found.';
  return dirs.split('\n').filter(Boolean).map(d => {
    const name = require('path').basename(d.replace(/\/$/, ''));
    const pending = run(`grep -rl 'STATUS:NEEDS-ACTION' "${d}" 2>/dev/null | wc -l`).trim();
    return `${name} | ${pending} pending | Source: local`;
  }).join('\n');
}

// ── iCal helpers ────────────────────────────────────────────────────────

function extractIcalField(block, field) {
  // Handle folded lines (continuation with space/tab)
  const unfolded = block.replace(/\r?\n[ \t]/g, '');
  const regex = new RegExp(`^${field}[;:](.*)$`, 'm');
  const match = unfolded.match(regex);
  if (!match) return '';
  // Strip parameters (e.g., DTSTART;VALUE=DATE:20260415 → 20260415)
  const val = match[1];
  const colonIdx = val.indexOf(':');
  // If the matched line had parameters (field;PARAM=val:data), extract after last colon
  // But if it was field:data, the value is already correct
  return colonIdx >= 0 && match[0].includes(';') ? val.slice(colonIdx + 1) : val;
}

function parseIcalDate(str) {
  if (!str) return null;
  // Format: 20260415T100000Z or 20260415T100000 or 20260415
  const clean = str.replace(/[^0-9T]/g, '');
  if (clean.length >= 8) {
    const y = clean.slice(0, 4);
    const m = clean.slice(4, 6);
    const d = clean.slice(6, 8);
    const h = clean.length >= 11 ? clean.slice(9, 11) : '00';
    const min = clean.length >= 13 ? clean.slice(11, 13) : '00';
    return new Date(`${y}-${m}-${d}T${h}:${min}:00`);
  }
  return null;
}

function toIcalDate(date) {
  return date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
}

function formatDateTime(date) {
  if (!date) return '';
  return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })
    + ' ' + formatTime(date);
}

function formatTime(date) {
  if (!date) return '';
  return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
}

// ── vCard helpers ───────────────────────────────────────────────────────

function extractVcardField(content, field) {
  const regex = new RegExp(`^${field}[;:](.*)$`, 'mi');
  const match = content.match(regex);
  if (!match) return '';
  const val = match[1];
  // Strip TYPE parameters
  const colonIdx = val.indexOf(':');
  return colonIdx >= 0 && match[0].includes(';') ? val.slice(colonIdx + 1) : val;
}

function formatVcard(content) {
  const fn = extractVcardField(content, 'FN') || '';
  const org = extractVcardField(content, 'ORG') || '';
  const title = extractVcardField(content, 'TITLE') || '';
  const bday = extractVcardField(content, 'BDAY') || '';
  const note = extractVcardField(content, 'NOTE') || '';

  // Extract all emails and phones
  const emails = [];
  const phones = [];
  const addrs = [];
  for (const line of content.split('\n')) {
    if (line.match(/^EMAIL/i)) {
      const val = line.split(':').slice(1).join(':').trim();
      if (val) emails.push(val);
    }
    if (line.match(/^TEL/i)) {
      const val = line.split(':').slice(1).join(':').trim();
      if (val) phones.push(val);
    }
    if (line.match(/^ADR/i)) {
      const val = line.split(':').slice(1).join(':').replace(/;/g, ', ').trim();
      if (val && val !== ', , , , , , ') addrs.push(val);
    }
  }

  const lines = [];
  if (fn) lines.push(`Name: ${fn}`);
  if (org) lines.push(`Organization: ${org.replace(/;/g, ', ')}`);
  if (title) lines.push(`Title: ${title}`);
  emails.forEach(e => lines.push(`Email: ${e}`));
  phones.forEach(p => lines.push(`Phone: ${p}`));
  addrs.forEach(a => lines.push(`Address: ${a}`));
  if (bday) lines.push(`Birthday: ${bday}`);
  if (note) lines.push(`Notes: ${note.slice(0, 500)}`);

  return lines.join('\n') || 'No contact details found.';
}

// ── File search helper ──────────────────────────────────────────────────

function findIcsFileByUid(baseDir, uid) {
  // Try direct filename match first
  const direct = run(`find ${baseDir} -name '${uid}.ics' 2>/dev/null | head -1`);
  if (direct) return direct;
  // Search by UID content
  const byContent = run(`grep -rl 'UID:${uid}' ${baseDir} 2>/dev/null | head -1`);
  return byContent || null;
}

module.exports = {
  isEdsAvailable,
  // Calendar
  edsCalendarList,
  edsCalendarCreate,
  edsCalendarDelete,
  edsCalendarSearch,
  edsListCalendars,
  // Contacts
  edsContactSearch,
  edsContactRead,
  edsContactCount,
  edsListGroups,
  // Tasks/Reminders
  edsTaskList,
  edsTaskCreate,
  edsTaskComplete,
  edsTaskDelete,
  edsTaskSearch,
  edsListTaskLists,
};
