// PRE Web GUI — Artifact tool
// Saves HTML/code artifacts to ~/.pre/artifacts/ for viewing in browser

const fs = require('fs');
const path = require('path');
const { ARTIFACTS_DIR } = require('../constants');

// Ensure artifacts directory exists
if (!fs.existsSync(ARTIFACTS_DIR)) {
  fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
}

/**
 * Create an artifact file and return its path/URL.
 * @param {Object} args - { title, content, type }
 * @returns {string} Result message with artifact path
 */
function createArtifact(args) {
  const title = args.title || 'Untitled';
  const content = args.content || '';
  const type = args.type || 'html';

  // Generate filename from title
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 60);
  const timestamp = Date.now().toString(36);
  const ext = type === 'html' ? 'html' : type === 'svg' ? 'svg' : type === 'markdown' ? 'md' : 'html';
  const filename = `${slug}-${timestamp}.${ext}`;
  const filePath = path.join(ARTIFACTS_DIR, filename);

  fs.writeFileSync(filePath, content);

  return `Artifact saved: ${filePath}\nView at: /artifacts/${filename}`;
}

module.exports = { createArtifact };
