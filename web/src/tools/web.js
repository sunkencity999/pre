// PRE Web GUI — Web tools (web_fetch, web_search)
// Mirrors the CLI's implementation using child_process for curl

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { CONNECTIONS_FILE, ARTIFACTS_DIR } = require('../constants');

function getConnectionKey(name) {
  try {
    const data = JSON.parse(fs.readFileSync(CONNECTIONS_FILE, 'utf-8'));
    switch (name) {
      case 'brave_search': return data.brave_search_key || null;
      case 'github': return data.github_key || null;
      case 'wolfram': return data.wolfram_key || null;
      default: return null;
    }
  } catch {
    return null;
  }
}

/**
 * Check if a URL points to an image based on extension or explicit save_image flag.
 */
function isImageUrl(url) {
  const imageExts = /\.(png|jpg|jpeg|gif|webp|svg|bmp|ico)(\?|$)/i;
  return imageExts.test(url);
}

function webFetch(args) {
  const url = args.url;
  if (!url) return 'Error: no url provided';

  // Validate URL
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return 'Error: URL must start with http:// or https://';
  }
  // Reject command injection characters
  if (/[`$;']/.test(url)) {
    return 'Error: URL contains invalid characters';
  }

  try {
    // Add GitHub auth header if applicable
    let authHeader = '';
    if (url.includes('github.com') || url.includes('api.github.com') || url.includes('raw.githubusercontent.com')) {
      const ghKey = getConnectionKey('github');
      if (ghKey) {
        authHeader = `-H 'Authorization: token ${ghKey}' -H 'Accept: application/vnd.github+json' `;
      }
    }

    // If URL is an image or save_image flag is set, download as binary and save locally
    if (isImageUrl(url) || args.save_image) {
      const downloadsDir = path.join(ARTIFACTS_DIR, 'downloads');
      if (!fs.existsSync(downloadsDir)) fs.mkdirSync(downloadsDir, { recursive: true });
      const ext = (url.match(/\.(png|jpg|jpeg|gif|webp|svg)/) || [null, 'png'])[1];
      const fname = `web-${Date.now()}.${ext}`;
      const outPath = path.join(downloadsDir, fname);
      execSync(`curl -sL --max-time 15 ${authHeader}-o '${outPath}' '${url}'`, { timeout: 20000 });
      const stat = fs.statSync(outPath);
      if (stat.size < 100) {
        fs.unlinkSync(outPath);
        return `Error: downloaded file too small (${stat.size} bytes), likely not a valid image`;
      }
      const sizeFmt = stat.size < 1024 ? `${stat.size}B` : `${(stat.size / 1024).toFixed(0)}KB`;
      return `Image downloaded and saved locally.\nPath: /artifacts/downloads/${fname}\nSize: ${sizeFmt}\nUse this path in HTML artifacts: <img src="/artifacts/downloads/${fname}">`;
    }

    // Try HTML-to-text conversion first, fall back to raw
    const cmd = `curl -sL --max-time 15 ${authHeader}'${url}' | textutil -stdin -format html -convert txt -stdout 2>/dev/null || curl -sL --max-time 15 ${authHeader}'${url}'`;
    const output = execSync(cmd, { encoding: 'utf-8', maxBuffer: 256 * 1024, timeout: 20000 }).trim();
    return output || `Error: no content fetched from ${url}`;
  } catch (err) {
    return `Error: failed to fetch ${url}: ${err.message}`;
  }
}

function webSearch(args) {
  const query = args.query;
  if (!query) return 'Error: no query provided';

  const apiKey = getConnectionKey('brave_search');
  if (!apiKey) {
    return 'Error: Brave Search not configured. Run /connections add brave_search';
  }

  // URL-encode query
  const encoded = query.replace(/ /g, '+').replace(/[`$;']/g, '').replace(/&/g, '%26');
  const count = Math.min(Math.max(args.count || 5, 1), 20);

  try {
    const cmd = `curl -s --max-time 10 -H 'Accept: application/json' -H 'X-Subscription-Token: ${apiKey}' 'https://api.search.brave.com/res/v1/web/search?q=${encoded}&count=${count}' 2>/dev/null`;
    const raw = execSync(cmd, { encoding: 'utf-8', maxBuffer: 128 * 1024, timeout: 15000 });

    // Parse JSON and extract results
    try {
      const data = JSON.parse(raw);
      const results = data.web?.results || [];
      if (results.length === 0) return `No search results found for: ${query}`;

      let output = `Search results for: ${query}\n\n`;
      for (let i = 0; i < results.length && i < count; i++) {
        const r = results[i];
        output += `${i + 1}. ${r.title || '(no title)'}\n`;
        output += `   ${r.url || ''}\n`;
        output += `   ${r.description || ''}\n\n`;
      }
      return output;
    } catch {
      // Fallback: return raw if JSON parse fails
      return raw.slice(0, 8000);
    }
  } catch (err) {
    return `Error: search failed: ${err.message}`;
  }
}

module.exports = { webFetch, webSearch };
