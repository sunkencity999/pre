// PRE Web GUI — File Search (cross-platform)
// macOS: Spotlight/mdfind (zero-config full-text search)
// Windows: Windows Search via ADODB SystemIndex + Get-ChildItem fallback

const { execSync } = require('child_process');
const os = require('os');
const { IS_WIN } = require('../platform');

/**
 * Spotlight tool dispatcher.
 * @param {Object} args - { action, query, folder, type, count }
 */
async function spotlight(args) {
  const action = args?.action;
  if (!action) return 'Error: action required. Actions: search, find_files, preview';

  switch (action) {
    case 'search': return searchFiles(args);
    case 'find_files': return findFiles(args);
    case 'preview': return previewFile(args);
    default:
      return `Error: unknown action '${action}'. Available: search, find_files, preview`;
  }
}

// ── Search (full-text + filename) ─────────────────────────────────────────

function searchFiles(args) {
  const query = args.query;
  const folder = args.folder || '';
  const count = Math.min(args.count || 20, 100);

  if (!query) return 'Error: "query" is required for search';

  if (IS_WIN) return winSearch(query, folder, '', count);
  return macSearch(query, folder, count);
}

function macSearch(query, folder, count) {
  let cmd = `mdfind`;
  if (folder) cmd += ` -onlyin "${folder.replace(/"/g, '\\"')}"`;
  cmd += ` "${query.replace(/"/g, '\\"')}"`;

  try {
    const output = execSync(cmd, {
      encoding: 'utf-8', timeout: 15000, maxBuffer: 512 * 1024,
    }).trim();

    if (!output) return `No files found matching: ${query}`;
    return output.split('\n').slice(0, count).join('\n');
  } catch (err) {
    return `Error searching Spotlight: ${(err.stderr || err.message).trim()}`;
  }
}

/**
 * Windows Search via ADODB SystemIndex query.
 * Falls back to Get-ChildItem if the Search index is unavailable.
 */
function winSearch(query, folder, typeFilter, count) {
  const safeQuery = query.replace(/'/g, "''");
  const scope = folder
    ? `AND SCOPE='file:${folder.replace(/\\/g, '/').replace(/'/g, "''")}'`
    : '';
  const typeClause = typeFilter ? `AND System.ItemType = '${typeFilter}'` : '';

  // Try Windows Search index first (fast, full-text)
  const ps = `
$conn = New-Object -ComObject ADODB.Connection
$conn.Open("Provider=Search.CollatorDSO;Extended Properties='Application=Windows'")
$rs = $conn.Execute("SELECT TOP ${count} System.ItemPathDisplay FROM SystemIndex WHERE FREETEXT('${safeQuery}') ${scope} ${typeClause}")
$results = @()
while (-not $rs.EOF) { $results += $rs.Fields.Item(0).Value; $rs.MoveNext() }
$conn.Close()
$results -join [char]10
`.trim().replace(/\n/g, ' ');

  try {
    const output = execSync(`powershell.exe -NoProfile -Command "${ps.replace(/"/g, '\\"')}"`, {
      encoding: 'utf-8', timeout: 15000, maxBuffer: 512 * 1024, windowsHide: true,
    }).trim();

    if (output) return output;
  } catch {
    // Windows Search index unavailable — fall through to Get-ChildItem
  }

  // Fallback: recursive file name search
  return winFallbackSearch(query, folder, count);
}

function winFallbackSearch(query, folder, count) {
  const searchPath = folder || os.homedir();
  const safeQuery = query.replace(/'/g, "''");
  const ps = `Get-ChildItem -Path '${searchPath}' -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -like '*${safeQuery}*' } | Select-Object -First ${count} -ExpandProperty FullName`;

  try {
    const output = execSync(`powershell.exe -NoProfile -Command "${ps.replace(/"/g, '\\"')}"`, {
      encoding: 'utf-8', timeout: 30000, maxBuffer: 512 * 1024, windowsHide: true,
    }).trim();

    if (!output) return `No files found matching: ${query}`;
    return output;
  } catch (err) {
    return `Error searching files: ${(err.stderr || err.message).trim()}`;
  }
}

// ── Find Files by Type ────────────────────────────────────────────────────

function findFiles(args) {
  const query = args.query || '';
  const type = args.type || '';
  const folder = args.folder || '';
  const count = Math.min(args.count || 20, 100);

  if (!query && !type) return 'Error: "query" and/or "type" is required';

  if (IS_WIN) return winFindFiles(query, type, folder, count);
  return macFindFiles(query, type, folder, count);
}

function macFindFiles(query, type, folder, count) {
  // Map friendly type names to kMDItemContentType values
  const typeMap = {
    pdf: 'com.adobe.pdf',
    image: 'public.image',
    photo: 'public.image',
    audio: 'public.audio',
    music: 'public.audio',
    video: 'public.movie',
    movie: 'public.movie',
    email: 'com.apple.mail.emlx',
    presentation: 'public.presentation',
    spreadsheet: 'public.spreadsheet',
    document: 'public.composite-content',
    folder: 'public.folder',
    text: 'public.text',
    code: 'public.source-code',
    app: 'com.apple.application-bundle',
  };

  let conditions = [];
  if (type && typeMap[type.toLowerCase()]) {
    conditions.push(`kMDItemContentTypeTree == "${typeMap[type.toLowerCase()]}"`);
  } else if (type) {
    conditions.push(`kMDItemContentTypeTree == "public.${type}" || kMDItemFSName == "*.${type}"`);
  }
  if (query) {
    conditions.push(`(kMDItemDisplayName == "*${query.replace(/"/g, '\\"')}*"wcd || kMDItemTextContent == "*${query.replace(/"/g, '\\"')}*"cd)`);
  }

  let cmd = `mdfind`;
  if (folder) cmd += ` -onlyin "${folder.replace(/"/g, '\\"')}"`;
  cmd += ` '${conditions.join(' && ')}'`;

  try {
    const output = execSync(cmd, {
      encoding: 'utf-8', timeout: 15000, maxBuffer: 512 * 1024,
    }).trim();

    if (!output) return `No ${type || ''} files found${query ? ` matching: ${query}` : ''}`;
    return output.split('\n').slice(0, count).join('\n');
  } catch (err) {
    return `Error finding files: ${(err.stderr || err.message).trim()}`;
  }
}

function winFindFiles(query, type, folder, count) {
  // Map friendly type names to Windows file extensions
  const extMap = {
    pdf: '.pdf',
    image: '.jpg,.jpeg,.png,.gif,.bmp,.webp,.svg,.tiff',
    photo: '.jpg,.jpeg,.png,.gif,.bmp,.webp,.tiff',
    audio: '.mp3,.wav,.flac,.aac,.ogg,.wma,.m4a',
    music: '.mp3,.wav,.flac,.aac,.ogg,.wma,.m4a',
    video: '.mp4,.avi,.mkv,.mov,.wmv,.flv,.webm',
    movie: '.mp4,.avi,.mkv,.mov,.wmv,.flv,.webm',
    presentation: '.pptx,.ppt,.key,.odp',
    spreadsheet: '.xlsx,.xls,.csv,.ods',
    document: '.docx,.doc,.pdf,.rtf,.odt,.txt',
    text: '.txt,.md,.log,.csv,.json,.xml,.yaml,.yml',
    code: '.js,.ts,.py,.java,.c,.cpp,.cs,.go,.rs,.rb,.php,.sh,.ps1',
    app: '.exe,.msi',
  };

  const searchPath = folder || os.homedir();
  const exts = type ? (extMap[type.toLowerCase()] || `.${type}`) : '';

  let filter = '';
  if (exts) {
    const extList = exts.split(',').map(e => `$_.Extension -eq '${e}'`).join(' -or ');
    filter = `| Where-Object { ${extList} }`;
  }
  if (query) {
    const safeQuery = query.replace(/'/g, "''");
    const nameFilter = `| Where-Object { $_.Name -like '*${safeQuery}*' }`;
    filter = filter ? `${filter} ${nameFilter}` : nameFilter;
  }

  const ps = `Get-ChildItem -Path '${searchPath}' -Recurse -File -ErrorAction SilentlyContinue ${filter} | Select-Object -First ${count} -ExpandProperty FullName`;

  try {
    const output = execSync(`powershell.exe -NoProfile -Command "${ps.replace(/"/g, '\\"')}"`, {
      encoding: 'utf-8', timeout: 30000, maxBuffer: 512 * 1024, windowsHide: true,
    }).trim();

    if (!output) return `No ${type || ''} files found${query ? ` matching: ${query}` : ''}`;
    return output;
  } catch (err) {
    return `Error finding files: ${(err.stderr || err.message).trim()}`;
  }
}

// ── File Metadata Preview ─────────────────────────────────────────────────

function previewFile(args) {
  const filePath = args.path || args.query;
  if (!filePath) return 'Error: "path" or "query" (file path) is required for preview';

  if (IS_WIN) return winPreview(filePath);
  return macPreview(filePath);
}

function macPreview(filePath) {
  try {
    const output = execSync(`mdls "${filePath.replace(/"/g, '\\"')}"`, {
      encoding: 'utf-8', timeout: 10000, maxBuffer: 256 * 1024,
    }).trim();

    if (!output) return `No metadata found for: ${filePath}`;

    const useful = [
      'kMDItemDisplayName', 'kMDItemContentType', 'kMDItemFSSize',
      'kMDItemFSCreationDate', 'kMDItemContentModificationDate',
      'kMDItemAuthors', 'kMDItemTitle', 'kMDItemKind',
      'kMDItemPageHeight', 'kMDItemPageWidth', 'kMDItemNumberOfPages',
      'kMDItemDurationSeconds', 'kMDItemPixelHeight', 'kMDItemPixelWidth',
      'kMDItemCodecs', 'kMDItemAudioBitRate', 'kMDItemVideoBitRate',
      'kMDItemWhereFroms', 'kMDItemLastUsedDate',
    ];

    const lines = output.split('\n');
    const filtered = lines.filter(line => {
      const attr = line.split('=')[0]?.trim();
      return useful.includes(attr);
    });

    return filtered.length > 0 ? filtered.join('\n') : output.split('\n').slice(0, 30).join('\n');
  } catch (err) {
    return `Error getting file metadata: ${(err.stderr || err.message).trim()}`;
  }
}

function winPreview(filePath) {
  const safePath = filePath.replace(/'/g, "''");
  const ps = `
$f = Get-Item -LiteralPath '${safePath}' -ErrorAction SilentlyContinue
if (-not $f) { Write-Output 'File not found'; exit }
$props = @(
  "Name: $($f.Name)"
  "Type: $($f.Extension)"
  "Size: $([math]::Round($f.Length / 1KB, 1)) KB"
  "Created: $($f.CreationTime)"
  "Modified: $($f.LastWriteTime)"
  "Accessed: $($f.LastAccessTime)"
  "ReadOnly: $($f.IsReadOnly)"
  "Attributes: $($f.Attributes)"
  "FullPath: $($f.FullName)"
)
$props -join [char]10
`.trim().replace(/\n/g, ' ');

  try {
    const output = execSync(`powershell.exe -NoProfile -Command "${ps.replace(/"/g, '\\"')}"`, {
      encoding: 'utf-8', timeout: 10000, maxBuffer: 256 * 1024, windowsHide: true,
    }).trim();

    return output || `No metadata found for: ${filePath}`;
  } catch (err) {
    return `Error getting file metadata: ${(err.stderr || err.message).trim()}`;
  }
}

module.exports = { spotlight };
