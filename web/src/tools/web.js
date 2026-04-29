// PRE Web GUI — Web tools (web_fetch, web_search)
// Uses Node.js native https — no curl dependency, fully cross-platform.

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { CONNECTIONS_FILE, ARTIFACTS_DIR } = require('../constants');
const { htmlToText } = require('../platform');

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
 * Perform an HTTP(S) GET request using Node.js native modules.
 * Follows redirects (up to 5), returns { data, headers } or throws.
 */
function httpGet(url, headers = {}, maxRedirects = 5) {
  return new Promise((resolve, reject) => {
    const mod = url.startsWith('https') ? https : http;
    const req = mod.get(url, { headers, timeout: 15000 }, (res) => {
      // Follow redirects
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
        if (maxRedirects <= 0) return reject(new Error('Too many redirects'));
        let loc = res.headers.location;
        if (loc.startsWith('/')) {
          const u = new URL(url);
          loc = `${u.protocol}//${u.host}${loc}`;
        }
        res.resume();
        return resolve(httpGet(loc, headers, maxRedirects - 1));
      }
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => resolve({ data: Buffer.concat(chunks), headers: res.headers, statusCode: res.statusCode }));
      res.on('error', reject);
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('Request timed out')); });
  });
}

/**
 * Check if a URL points to an image based on extension or explicit save_image flag.
 */
function isImageUrl(url) {
  const imageExts = /\.(png|jpg|jpeg|gif|webp|svg|bmp|ico)(\?|$)/i;
  return imageExts.test(url);
}

async function webFetch(args) {
  const url = args.url;
  if (!url) return 'Error: no url provided';

  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return 'Error: URL must start with http:// or https://';
  }

  try {
    const headers = {};

    // Add GitHub auth header if applicable
    if (url.includes('github.com') || url.includes('api.github.com') || url.includes('raw.githubusercontent.com')) {
      const ghKey = getConnectionKey('github');
      if (ghKey) {
        headers['Authorization'] = `token ${ghKey}`;
        headers['Accept'] = 'application/vnd.github+json';
      }
    }

    // If URL is an image or save_image flag is set, download as binary and save locally
    if (isImageUrl(url) || args.save_image) {
      const downloadsDir = path.join(ARTIFACTS_DIR, 'downloads');
      if (!fs.existsSync(downloadsDir)) fs.mkdirSync(downloadsDir, { recursive: true });
      const ext = (url.match(/\.(png|jpg|jpeg|gif|webp|svg)/) || [null, 'png'])[1];
      const fname = `web-${Date.now()}.${ext}`;
      const outPath = path.join(downloadsDir, fname);

      const { data } = await httpGet(url, headers);
      fs.writeFileSync(outPath, data);
      const size = data.length;
      if (size < 100) {
        fs.unlinkSync(outPath);
        return `Error: downloaded file too small (${size} bytes), likely not a valid image`;
      }
      const sizeFmt = size < 1024 ? `${size}B` : `${(size / 1024).toFixed(0)}KB`;
      return `Image downloaded and saved locally.\nPath: /artifacts/downloads/${fname}\nSize: ${sizeFmt}\nUse this path in HTML artifacts: <img src="/artifacts/downloads/${fname}">`;
    }

    // Fetch HTML and convert to readable text
    const { data } = await httpGet(url, headers);
    const rawHtml = data.toString('utf-8').trim();
    if (!rawHtml) return `Error: no content fetched from ${url}`;
    const output = htmlToText(rawHtml);
    return output || rawHtml.slice(0, 50000);
  } catch (err) {
    return `Error: failed to fetch ${url}: ${err.message}`;
  }
}

async function webSearch(args) {
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

async function braveSearch(query, count, apiKey) {
  const encoded = encodeURIComponent(query);
  try {
    const url = `https://api.search.brave.com/res/v1/web/search?q=${encoded}&count=${count}`;
    const { data } = await httpGet(url, {
      'Accept': 'application/json',
      'X-Subscription-Token': apiKey,
    });

    try {
      const json = JSON.parse(data.toString('utf-8'));
      const results = json.web?.results || [];
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
      return data.toString('utf-8').slice(0, 8000);
    }
  } catch (err) {
    return `Error: search failed: ${err.message}`;
  }
}

async function duckDuckGoSearch(query, count) {
  const encoded = encodeURIComponent(query);
  try {
    const url = `https://html.duckduckgo.com/html/?q=${encoded}`;
    const { data } = await httpGet(url, {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    });
    const html = data.toString('utf-8');

    // Parse results from DuckDuckGo HTML response
    const results = [];
    // Each result is in a <div class="result ..."> block
    const resultBlocks = html.match(/<a class="result__a"[^>]*>[\s\S]*?<\/a>[\s\S]*?<a class="result__snippet"[^>]*>[\s\S]*?<\/a>/g)
      || html.match(/<a class="result__a"[^>]*>[\s\S]*?<\/a>[\s\S]*?<td class="result__snippet">[\s\S]*?<\/td>/g)
      || [];

    for (const block of resultBlocks) {
      if (results.length >= count) break;
      const linkMatch = block.match(/<a class="result__a"[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/);
      const snippetMatch = block.match(/class="result__snippet"[^>]*>([\s\S]*?)<\/(?:a|td)>/);

      if (linkMatch) {
        let resultUrl = linkMatch[1];
        const uddgMatch = resultUrl.match(/uddg=([^&]+)/);
        if (uddgMatch) resultUrl = decodeURIComponent(uddgMatch[1]);

        const title = linkMatch[2].replace(/<[^>]+>/g, '').trim();
        const snippet = snippetMatch
          ? snippetMatch[1].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#x27;/g, "'").trim()
          : '';

        if (resultUrl && title) {
          results.push({ title, url: resultUrl, snippet });
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
          let resultUrl = m[1];
          const uddgMatch = resultUrl.match(/uddg=([^&]+)/);
          if (uddgMatch) resultUrl = decodeURIComponent(uddgMatch[1]);
          results.push({ title: m[2].replace(/<[^>]+>/g, '').trim(), url: resultUrl, snippet: '' });
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
