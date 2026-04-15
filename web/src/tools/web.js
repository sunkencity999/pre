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
  const count = Math.min(Math.max(args.count || 5, 1), 20);

  // Use Brave API if configured, otherwise fall back to DuckDuckGo HTML
  if (apiKey) {
    return braveSearch(query, count, apiKey);
  }
  return duckDuckGoSearch(query, count);
}

function braveSearch(query, count, apiKey) {
  const encoded = query.replace(/ /g, '+').replace(/[`$;']/g, '').replace(/&/g, '%26');
  try {
    const cmd = `curl -s --max-time 10 -H 'Accept: application/json' -H 'X-Subscription-Token: ${apiKey}' 'https://api.search.brave.com/res/v1/web/search?q=${encoded}&count=${count}' 2>/dev/null`;
    const raw = execSync(cmd, { encoding: 'utf-8', maxBuffer: 128 * 1024, timeout: 15000 });

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
      return raw.slice(0, 8000);
    }
  } catch (err) {
    return `Error: search failed: ${err.message}`;
  }
}

function duckDuckGoSearch(query, count) {
  // DuckDuckGo HTML search — no API key needed
  const encoded = encodeURIComponent(query);
  try {
    const cmd = `curl -sL --max-time 15 -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36' 'https://html.duckduckgo.com/html/?q=${encoded}' 2>/dev/null`;
    const html = execSync(cmd, { encoding: 'utf-8', maxBuffer: 512 * 1024, timeout: 20000 });

    // Parse results from DuckDuckGo HTML response
    const results = [];
    // Each result is in a <div class="result ..."> block
    const resultBlocks = html.match(/<a class="result__a"[^>]*>[\s\S]*?<\/a>[\s\S]*?<a class="result__snippet"[^>]*>[\s\S]*?<\/a>/g)
      || html.match(/<a class="result__a"[^>]*>[\s\S]*?<\/a>[\s\S]*?<td class="result__snippet">[\s\S]*?<\/td>/g)
      || [];

    for (const block of resultBlocks) {
      if (results.length >= count) break;
      // Extract URL and title from the result link
      const linkMatch = block.match(/<a class="result__a"[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/);
      // Extract snippet
      const snippetMatch = block.match(/class="result__snippet"[^>]*>([\s\S]*?)<\/(?:a|td)>/);

      if (linkMatch) {
        let url = linkMatch[1];
        // DuckDuckGo wraps URLs in a redirect — extract the actual URL
        const uddgMatch = url.match(/uddg=([^&]+)/);
        if (uddgMatch) url = decodeURIComponent(uddgMatch[1]);

        const title = linkMatch[2].replace(/<[^>]+>/g, '').trim();
        const snippet = snippetMatch
          ? snippetMatch[1].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#x27;/g, "'").trim()
          : '';

        if (url && title) {
          results.push({ title, url, snippet });
        }
      }
    }

    if (results.length === 0) {
      // Fallback: try simpler regex for older DDG format
      const simpleLinks = html.match(/<a[^>]*class="result__url"[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/g) || [];
      for (const link of simpleLinks) {
        if (results.length >= count) break;
        const m = link.match(/href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/);
        if (m) {
          let url = m[1];
          const uddgMatch = url.match(/uddg=([^&]+)/);
          if (uddgMatch) url = decodeURIComponent(uddgMatch[1]);
          results.push({ title: m[2].replace(/<[^>]+>/g, '').trim(), url, snippet: '' });
        }
      }
    }

    if (results.length === 0) return `No search results found for: ${query}`;

    let output = `Search results for: ${query}\n\n`;
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      output += `${i + 1}. ${r.title}\n`;
      output += `   ${r.url}\n`;
      if (r.snippet) output += `   ${r.snippet}\n`;
      output += '\n';
    }
    return output;
  } catch (err) {
    return `Error: search failed: ${err.message}`;
  }
}

module.exports = { webFetch, webSearch };
