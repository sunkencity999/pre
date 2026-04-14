// PRE Web GUI — macOS Spotlight/mdfind integration
// Zero-config full-text search across the entire machine via mdfind CLI

const { execSync } = require('child_process');

/**
 * Spotlight tool dispatcher.
 * @param {Object} args - { action, query, folder, type, count }
 */
async function spotlight(args) {
  const action = args?.action;
  if (!action) return 'Error: action required. Actions: search, find_files, preview';

  switch (action) {
    case 'search': return searchSpotlight(args);
    case 'find_files': return findFiles(args);
    case 'preview': return previewFile(args);
    default:
      return `Error: unknown action '${action}'. Available: search, find_files, preview`;
  }
}

/**
 * Full-text content search via mdfind.
 * Searches file names AND content — this is what makes Spotlight powerful.
 */
function searchSpotlight(args) {
  const query = args.query;
  const folder = args.folder || '';
  const count = Math.min(args.count || 20, 100);

  if (!query) return 'Error: "query" is required for search';

  let cmd = `mdfind`;
  if (folder) cmd += ` -onlyin "${folder.replace(/"/g, '\\"')}"`;
  cmd += ` "${query.replace(/"/g, '\\"')}"`;

  try {
    const output = execSync(cmd, {
      encoding: 'utf-8',
      timeout: 15000,
      maxBuffer: 512 * 1024,
    }).trim();

    if (!output) return `No files found matching: ${query}`;

    const lines = output.split('\n').slice(0, count);
    return lines.join('\n');
  } catch (err) {
    return `Error searching Spotlight: ${(err.stderr || err.message).trim()}`;
  }
}

/**
 * Find files by type using Spotlight metadata attributes.
 * Common types: pdf, image, audio, video, email, presentation, spreadsheet, folder
 */
function findFiles(args) {
  const query = args.query || '';
  const type = args.type || '';
  const folder = args.folder || '';
  const count = Math.min(args.count || 20, 100);

  if (!query && !type) return 'Error: "query" and/or "type" is required';

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
    // Try using it as a raw UTI or file extension
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
      encoding: 'utf-8',
      timeout: 15000,
      maxBuffer: 512 * 1024,
    }).trim();

    if (!output) return `No ${type || ''} files found${query ? ` matching: ${query}` : ''}`;

    const lines = output.split('\n').slice(0, count);
    return lines.join('\n');
  } catch (err) {
    return `Error finding files: ${(err.stderr || err.message).trim()}`;
  }
}

/**
 * Preview file metadata via mdls (Spotlight metadata attributes).
 */
function previewFile(args) {
  const filePath = args.path || args.query;
  if (!filePath) return 'Error: "path" or "query" (file path) is required for preview';

  try {
    const output = execSync(`mdls "${filePath.replace(/"/g, '\\"')}"`, {
      encoding: 'utf-8',
      timeout: 10000,
      maxBuffer: 256 * 1024,
    }).trim();

    if (!output) return `No metadata found for: ${filePath}`;

    // Filter to the most useful attributes
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

module.exports = { spotlight };
