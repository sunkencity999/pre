// PRE Web GUI — Browser agent (computer control via screenshots + vision)
// Uses Puppeteer-core with existing Chrome installation.
// The model sees screenshots and issues navigation/click/type commands.

const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Find Chrome on macOS
const CHROME_PATHS = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  path.join(os.homedir(), 'Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
];

let browser = null;
let page = null;

function findChrome() {
  for (const p of CHROME_PATHS) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

/**
 * Launch or reuse the browser instance
 */
async function ensureBrowser() {
  if (browser && browser.connected) return;

  const chromePath = findChrome();
  if (!chromePath) {
    throw new Error('Chrome not found. Install Google Chrome to use browser tools.');
  }

  browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--window-size=1280,800',
      '--disable-dev-shm-usage',
    ],
    defaultViewport: { width: 1280, height: 800 },
  });

  // Clean up on exit
  browser.on('disconnected', () => { browser = null; page = null; });
}

/**
 * Get or create the active page
 */
async function ensurePage() {
  await ensureBrowser();
  if (page && !page.isClosed()) return page;

  const pages = await browser.pages();
  page = pages.length > 0 ? pages[0] : await browser.newPage();

  // Set reasonable defaults
  await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');

  return page;
}

/**
 * Take a screenshot and return as base64
 */
async function takeScreenshot(fullPage = false) {
  const p = await ensurePage();
  const buffer = await p.screenshot({
    type: 'png',
    fullPage,
    encoding: 'binary',
  });
  return buffer.toString('base64');
}

/**
 * Main browser tool dispatcher
 * Actions: navigate, screenshot, click, type, scroll, read, evaluate, back, forward, wait, close
 */
async function browserAction(args) {
  const { action } = args;
  if (!action) return 'Error: action is required';

  switch (action) {
    case 'navigate': {
      const { url } = args;
      if (!url) return 'Error: url is required for navigate';
      const p = await ensurePage();
      await p.goto(url, { waitUntil: 'domcontentloaded', timeout: 15000 });
      const title = await p.title();
      const screenshot = await takeScreenshot();
      // Extract page text so the result is useful even without vision (e.g. sub-agents)
      const text = await p.evaluate(() => {
        const main = document.querySelector('main, article, [role="main"], .content, #content');
        const target = main || document.body;
        return target.innerText.slice(0, 8000);
      });
      return JSON.stringify({
        action: 'navigate',
        url: p.url(),
        title,
        text,
        screenshot,
        message: `Navigated to ${title} (${p.url()})`,
      });
    }

    case 'screenshot': {
      const screenshot = await takeScreenshot(args.full_page === true);
      const p = await ensurePage();
      const title = await p.title();
      return JSON.stringify({
        action: 'screenshot',
        url: p.url(),
        title,
        screenshot,
        message: `Screenshot of ${title}`,
      });
    }

    case 'click': {
      const { selector, x, y, text } = args;
      const p = await ensurePage();

      if (text) {
        // Click element containing specific text
        const clicked = await p.evaluate((searchText) => {
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
          while (walker.nextNode()) {
            if (walker.currentNode.textContent.trim().includes(searchText)) {
              const el = walker.currentNode.parentElement;
              if (el) { el.click(); return el.tagName + ': ' + el.textContent.slice(0, 50); }
            }
          }
          return null;
        }, text);
        if (!clicked) return `Error: No element found containing text "${text}"`;
        const screenshot = await takeScreenshot();
        return JSON.stringify({ action: 'click', clicked, screenshot, message: `Clicked element containing "${text}"` });
      }

      if (selector) {
        await p.click(selector);
        await p.waitForNetworkIdle({ timeout: 3000 }).catch(() => {});
        const screenshot = await takeScreenshot();
        return JSON.stringify({ action: 'click', selector, screenshot, message: `Clicked ${selector}` });
      }

      if (x !== undefined && y !== undefined) {
        await p.mouse.click(parseInt(x), parseInt(y));
        await p.waitForNetworkIdle({ timeout: 3000 }).catch(() => {});
        const screenshot = await takeScreenshot();
        return JSON.stringify({ action: 'click', x, y, screenshot, message: `Clicked at (${x}, ${y})` });
      }

      return 'Error: click requires selector, text, or x/y coordinates';
    }

    case 'type': {
      const { selector, text, clear } = args;
      if (!text) return 'Error: text is required for type';
      const p = await ensurePage();

      if (selector) {
        if (clear) {
          await p.click(selector, { clickCount: 3 }); // Select all
          await p.keyboard.press('Backspace');
        }
        await p.type(selector, text, { delay: 30 });
      } else {
        // Type into currently focused element
        await p.keyboard.type(text, { delay: 30 });
      }

      const screenshot = await takeScreenshot();
      return JSON.stringify({ action: 'type', text: text.slice(0, 50), screenshot, message: `Typed "${text.slice(0, 50)}"` });
    }

    case 'press': {
      const { key } = args;
      if (!key) return 'Error: key is required for press (e.g. Enter, Tab, Escape)';
      const p = await ensurePage();
      await p.keyboard.press(key);
      await new Promise(r => setTimeout(r, 500));
      const screenshot = await takeScreenshot();
      return JSON.stringify({ action: 'press', key, screenshot, message: `Pressed ${key}` });
    }

    case 'scroll': {
      const { direction, amount } = args;
      const p = await ensurePage();
      const pixels = parseInt(amount) || 400;
      const dy = direction === 'up' ? -pixels : pixels;
      await p.mouse.wheel({ deltaY: dy });
      await new Promise(r => setTimeout(r, 300));
      const screenshot = await takeScreenshot();
      return JSON.stringify({ action: 'scroll', direction: direction || 'down', pixels, screenshot, message: `Scrolled ${direction || 'down'} ${pixels}px` });
    }

    case 'read': {
      // Extract page text content (no screenshot needed for this)
      const p = await ensurePage();
      const text = await p.evaluate(() => {
        // Get visible text, prioritizing main content
        const main = document.querySelector('main, article, [role="main"], .content, #content');
        const target = main || document.body;
        return target.innerText.slice(0, 8000);
      });
      const title = await p.title();
      return JSON.stringify({
        action: 'read',
        title,
        url: p.url(),
        text,
        message: `Read ${text.length} chars from ${title}`,
      });
    }

    case 'evaluate': {
      // Run arbitrary JS in page context
      const { script } = args;
      if (!script) return 'Error: script is required for evaluate';
      const p = await ensurePage();
      try {
        const result = await p.evaluate(script);
        return JSON.stringify({ action: 'evaluate', result: String(result).slice(0, 4000), message: 'Script executed' });
      } catch (err) {
        return `Error evaluating script: ${err.message}`;
      }
    }

    case 'select': {
      // Describe clickable elements on the page (for the model to choose from)
      const p = await ensurePage();
      const elements = await p.evaluate(() => {
        const clickable = document.querySelectorAll('a, button, input, select, textarea, [onclick], [role="button"], [role="link"], [role="tab"]');
        const results = [];
        for (const el of clickable) {
          const rect = el.getBoundingClientRect();
          if (rect.width === 0 || rect.height === 0) continue;
          if (rect.top > window.innerHeight) continue; // Off screen
          results.push({
            tag: el.tagName.toLowerCase(),
            text: (el.textContent || el.value || el.placeholder || '').trim().slice(0, 60),
            type: el.type || '',
            href: el.href || '',
            x: Math.round(rect.x + rect.width / 2),
            y: Math.round(rect.y + rect.height / 2),
          });
          if (results.length >= 50) break;
        }
        return results;
      });
      const screenshot = await takeScreenshot();
      return JSON.stringify({
        action: 'select',
        elements,
        screenshot,
        message: `Found ${elements.length} interactive elements`,
      });
    }

    case 'back': {
      const p = await ensurePage();
      await p.goBack({ waitUntil: 'domcontentloaded', timeout: 10000 }).catch(() => {});
      const screenshot = await takeScreenshot();
      const title = await p.title();
      return JSON.stringify({ action: 'back', title, url: p.url(), screenshot, message: `Went back to ${title}` });
    }

    case 'forward': {
      const p = await ensurePage();
      await p.goForward({ waitUntil: 'domcontentloaded', timeout: 10000 }).catch(() => {});
      const screenshot = await takeScreenshot();
      const title = await p.title();
      return JSON.stringify({ action: 'forward', title, url: p.url(), screenshot, message: `Went forward to ${title}` });
    }

    case 'wait': {
      const { selector: sel, timeout: to } = args;
      const p = await ensurePage();
      const timeout = parseInt(to) || 5000;
      if (sel) {
        await p.waitForSelector(sel, { timeout });
      } else {
        await new Promise(r => setTimeout(r, timeout));
      }
      const screenshot = await takeScreenshot();
      return JSON.stringify({ action: 'wait', screenshot, message: sel ? `Waited for ${sel}` : `Waited ${timeout}ms` });
    }

    case 'close': {
      if (browser) {
        await browser.close().catch(() => {});
        browser = null;
        page = null;
      }
      return 'Browser closed.';
    }

    default:
      return `Error: unknown browser action '${action}'. Available: navigate, screenshot, click, type, press, scroll, read, evaluate, select, back, forward, wait, close`;
  }
}

/**
 * Check if the browser is available
 */
function isAvailable() {
  return !!findChrome();
}

module.exports = { browserAction, isAvailable };
