// PRE Web GUI — Artifact export utilities
// Shared module for PDF, PNG, and self-contained HTML export via Puppeteer

const fs = require('fs');
const path = require('path');
const { ARTIFACTS_DIR } = require('../constants');

// ── Chrome discovery ──

const CHROME_PATHS = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
];

function findChrome() {
  const found = CHROME_PATHS.find(p => fs.existsSync(p));
  if (!found) throw new Error('Chrome not found — install Google Chrome for export features');
  return found;
}

/**
 * Launch a headless Puppeteer browser and load an HTML artifact.
 * Caller is responsible for closing the browser.
 */
async function openArtifact(htmlWebPath) {
  const puppeteer = require('puppeteer-core');

  const htmlFile = resolveArtifactPath(htmlWebPath);
  if (!fs.existsSync(htmlFile)) {
    throw new Error(`Artifact not found: ${htmlFile}`);
  }

  const browser = await puppeteer.launch({
    headless: true,
    executablePath: findChrome(),
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();
  await page.goto(`file://${htmlFile}`, { waitUntil: 'networkidle0', timeout: 30000 });

  return { browser, page };
}

/**
 * Resolve a web-relative artifact path to an absolute filesystem path.
 */
function resolveArtifactPath(webPath) {
  return path.join(ARTIFACTS_DIR, webPath.replace(/^\/artifacts\//, ''));
}

/**
 * Generate a slug-based output filename.
 */
function makeOutputName(title, ext) {
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 60);
  const timestamp = Date.now().toString(36);
  return `${slug}-${timestamp}.${ext}`;
}

// ── PDF Export ──

/**
 * Convert an HTML artifact to PDF using headless Chrome.
 * @param {string} htmlWebPath - Web-relative path like /artifacts/report-abc.html
 * @param {string} title - Title for the PDF filename
 * @returns {Promise<{webPath: string, filePath: string, filename: string}>}
 */
async function exportPdf(htmlWebPath, title) {
  const pdfFilename = makeOutputName(title || 'export', 'pdf');
  const pdfPath = path.join(ARTIFACTS_DIR, pdfFilename);

  const { browser, page } = await openArtifact(htmlWebPath);
  try {
    await page.pdf({
      path: pdfPath,
      format: 'Letter',
      margin: { top: '0.5in', bottom: '0.5in', left: '0.5in', right: '0.5in' },
      printBackground: true,
    });
    console.log(`[export] PDF: ${pdfFilename} (${(fs.statSync(pdfPath).size / 1024).toFixed(0)} KB)`);
    return { webPath: `/artifacts/${pdfFilename}`, filePath: pdfPath, filename: pdfFilename };
  } finally {
    await browser.close();
  }
}

// ── PNG Screenshot Export ──

/**
 * Convert an HTML artifact to a high-res PNG screenshot.
 * @param {string} htmlWebPath - Web-relative path like /artifacts/report-abc.html
 * @param {string} title - Title for the PNG filename
 * @returns {Promise<{webPath: string, filePath: string, filename: string}>}
 */
async function exportPng(htmlWebPath, title) {
  const pngFilename = makeOutputName(title || 'export', 'png');
  const pngPath = path.join(ARTIFACTS_DIR, pngFilename);

  const { browser, page } = await openArtifact(htmlWebPath);
  try {
    // Set a wide viewport for high-res capture
    await page.setViewport({ width: 1200, height: 800, deviceScaleFactor: 2 });
    // Re-wait after viewport change for any responsive layout shifts
    await page.waitForNetworkIdle({ timeout: 5000 }).catch(() => {});

    await page.screenshot({
      path: pngPath,
      fullPage: true,
      type: 'png',
    });
    console.log(`[export] PNG: ${pngFilename} (${(fs.statSync(pngPath).size / 1024).toFixed(0)} KB)`);
    return { webPath: `/artifacts/${pngFilename}`, filePath: pngPath, filename: pngFilename };
  } finally {
    await browser.close();
  }
}

// ── Self-Contained HTML Export ──

/**
 * Create a self-contained HTML file with all external resources inlined.
 * - CSS <link> tags → inlined <style> blocks
 * - <img src="..."> → base64 data URIs
 * - <script src="..."> → inlined <script> blocks
 * @param {string} htmlWebPath - Web-relative path like /artifacts/report-abc.html
 * @param {string} title - Title for the output filename
 * @returns {Promise<{webPath: string, filePath: string, filename: string}>}
 */
async function exportSelfContainedHtml(htmlWebPath, title) {
  const htmlFile = resolveArtifactPath(htmlWebPath);
  if (!fs.existsSync(htmlFile)) {
    throw new Error(`Artifact not found: ${htmlFile}`);
  }

  let html = fs.readFileSync(htmlFile, 'utf-8');
  const baseDir = path.dirname(htmlFile);

  // Inline CSS <link rel="stylesheet" href="...">
  html = html.replace(/<link\s+[^>]*rel=["']stylesheet["'][^>]*href=["']([^"']+)["'][^>]*\/?>/gi, (match, href) => {
    const cssContent = tryReadResource(href, baseDir);
    if (cssContent !== null) {
      return `<style>/* Inlined from ${href} */\n${cssContent}</style>`;
    }
    return match; // Keep original if can't resolve
  });

  // Also match <link href="..." rel="stylesheet">
  html = html.replace(/<link\s+[^>]*href=["']([^"']+)["'][^>]*rel=["']stylesheet["'][^>]*\/?>/gi, (match, href) => {
    const cssContent = tryReadResource(href, baseDir);
    if (cssContent !== null) {
      return `<style>/* Inlined from ${href} */\n${cssContent}</style>`;
    }
    return match;
  });

  // Inline images → base64 data URIs
  html = html.replace(/<img\s+[^>]*src=["']([^"']+)["']/gi, (match, src) => {
    // Skip already-inlined data URIs and external URLs
    if (src.startsWith('data:') || src.startsWith('http://') || src.startsWith('https://')) {
      return match;
    }
    const imgData = tryReadResourceAsBase64(src, baseDir);
    if (imgData) {
      return match.replace(src, imgData);
    }
    return match;
  });

  // Inline <script src="...">
  html = html.replace(/<script\s+[^>]*src=["']([^"']+)["'][^>]*><\/script>/gi, (match, src) => {
    if (src.startsWith('http://') || src.startsWith('https://')) {
      return match; // Keep external CDN scripts
    }
    const jsContent = tryReadResource(src, baseDir);
    if (jsContent !== null) {
      return `<script>/* Inlined from ${src} */\n${jsContent}</script>`;
    }
    return match;
  });

  const outFilename = makeOutputName(title || 'export-standalone', 'html');
  const outPath = path.join(ARTIFACTS_DIR, outFilename);
  fs.writeFileSync(outPath, html);

  console.log(`[export] HTML (self-contained): ${outFilename} (${(fs.statSync(outPath).size / 1024).toFixed(0)} KB)`);
  return { webPath: `/artifacts/${outFilename}`, filePath: outPath, filename: outFilename };
}

/**
 * Try to read a local resource file as text.
 * Resolves relative paths against baseDir, and /artifacts/ paths against ARTIFACTS_DIR.
 */
function tryReadResource(href, baseDir) {
  try {
    let resolved;
    if (href.startsWith('/artifacts/')) {
      resolved = path.join(ARTIFACTS_DIR, href.replace(/^\/artifacts\//, ''));
    } else if (href.startsWith('/')) {
      // Absolute path from web root — try artifacts dir
      resolved = path.join(ARTIFACTS_DIR, href.replace(/^\//, ''));
    } else {
      resolved = path.join(baseDir, href);
    }
    if (fs.existsSync(resolved)) {
      return fs.readFileSync(resolved, 'utf-8');
    }
  } catch {}
  return null;
}

/**
 * Try to read a local resource file as a base64 data URI.
 */
function tryReadResourceAsBase64(src, baseDir) {
  try {
    let resolved;
    if (src.startsWith('/artifacts/')) {
      resolved = path.join(ARTIFACTS_DIR, src.replace(/^\/artifacts\//, ''));
    } else if (src.startsWith('/')) {
      resolved = path.join(ARTIFACTS_DIR, src.replace(/^\//, ''));
    } else {
      resolved = path.join(baseDir, src);
    }
    if (fs.existsSync(resolved)) {
      const buf = fs.readFileSync(resolved);
      const ext = path.extname(resolved).toLowerCase();
      const mimeMap = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.svg': 'image/svg+xml', '.webp': 'image/webp' };
      const mime = mimeMap[ext] || 'application/octet-stream';
      return `data:${mime};base64,${buf.toString('base64')}`;
    }
  } catch {}
  return null;
}

module.exports = { exportPdf, exportPng, exportSelfContainedHtml, findChrome, resolveArtifactPath };
