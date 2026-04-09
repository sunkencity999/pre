// PRE Web GUI — Lightweight Markdown-to-HTML parser
// Handles: headings, bold, italic, code, links, lists, blockquotes, paragraphs

const Markdown = (() => {
  function escapeHtml(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function parseInline(text) {
    return text
      // Code spans (must be first to avoid processing inside them)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      // Bold + italic
      .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
      // Bold
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // Italic
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      // Links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
      // Bare URLs
      .replace(/(^|[^"(])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
  }

  function render(text) {
    if (!text) return '';

    const lines = text.split('\n');
    const html = [];
    let inCodeBlock = false;
    let codeBlockContent = '';
    let codeBlockLang = '';
    let inList = false;
    let listType = '';

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Code blocks
      if (line.startsWith('```')) {
        if (inCodeBlock) {
          const langLabel = codeBlockLang && codeBlockLang !== 'text' ? codeBlockLang : '';
          const codeId = 'code-' + Math.random().toString(36).slice(2, 8);
          html.push(
            `<div class="code-block-wrapper">` +
            `<div class="code-block-header">` +
            (langLabel ? `<span class="code-block-lang">${escapeHtml(langLabel)}</span>` : '<span></span>') +
            `<button class="code-block-copy" onclick="copyCodeBlock(this, '${codeId}')" title="Copy code">` +
            `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>` +
            `</button></div>` +
            `<pre><code id="${codeId}" class="lang-${escapeHtml(codeBlockLang)}">${escapeHtml(codeBlockContent.trimEnd())}</code></pre>` +
            `</div>`
          );
          inCodeBlock = false;
          codeBlockContent = '';
          codeBlockLang = '';
        } else {
          if (inList) { html.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
          inCodeBlock = true;
          codeBlockLang = line.slice(3).trim() || 'text';
        }
        continue;
      }
      if (inCodeBlock) {
        codeBlockContent += line + '\n';
        continue;
      }

      // Blank line
      if (line.trim() === '') {
        if (inList) { html.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
        continue;
      }

      // Headings
      const headingMatch = line.match(/^(#{1,6})\s+(.+)/);
      if (headingMatch) {
        if (inList) { html.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
        const level = headingMatch[1].length;
        html.push(`<h${level}>${parseInline(escapeHtml(headingMatch[2]))}</h${level}>`);
        continue;
      }

      // Blockquote
      if (line.startsWith('> ')) {
        if (inList) { html.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
        html.push(`<blockquote>${parseInline(escapeHtml(line.slice(2)))}</blockquote>`);
        continue;
      }

      // Unordered list
      if (/^[\s]*[-*+]\s+/.test(line)) {
        const content = line.replace(/^[\s]*[-*+]\s+/, '');
        if (!inList || listType !== 'ul') {
          if (inList) html.push(listType === 'ul' ? '</ul>' : '</ol>');
          html.push('<ul>');
          inList = true;
          listType = 'ul';
        }
        html.push(`<li>${parseInline(escapeHtml(content))}</li>`);
        continue;
      }

      // Ordered list
      if (/^[\s]*\d+\.\s+/.test(line)) {
        const content = line.replace(/^[\s]*\d+\.\s+/, '');
        if (!inList || listType !== 'ol') {
          if (inList) html.push(listType === 'ul' ? '</ul>' : '</ol>');
          html.push('<ol>');
          inList = true;
          listType = 'ol';
        }
        html.push(`<li>${parseInline(escapeHtml(content))}</li>`);
        continue;
      }

      // Horizontal rule
      if (/^[-*_]{3,}$/.test(line.trim())) {
        if (inList) { html.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
        html.push('<hr>');
        continue;
      }

      // Paragraph
      if (inList) { html.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
      html.push(`<p>${parseInline(escapeHtml(line))}</p>`);
    }

    // Close any open blocks (streaming — code block not yet closed)
    if (inCodeBlock) {
      const langLabel = codeBlockLang && codeBlockLang !== 'text' ? codeBlockLang : '';
      html.push(
        `<div class="code-block-wrapper">` +
        `<div class="code-block-header">` +
        (langLabel ? `<span class="code-block-lang">${escapeHtml(langLabel)}</span>` : '<span></span>') +
        `<span class="code-block-lang" style="opacity:0.5">...</span>` +
        `</div>` +
        `<pre><code class="lang-${escapeHtml(codeBlockLang)}">${escapeHtml(codeBlockContent.trimEnd())}</code></pre>` +
        `</div>`
      );
    }
    if (inList) {
      html.push(listType === 'ul' ? '</ul>' : '</ol>');
    }

    return html.join('\n');
  }

  return { render, escapeHtml };
})();

// Global handler for code block copy buttons
function copyCodeBlock(btn, codeId) {
  const codeEl = document.getElementById(codeId);
  if (!codeEl) return;
  navigator.clipboard.writeText(codeEl.textContent).then(() => {
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
      btn.classList.remove('copied');
    }, 2000);
  }).catch(() => {});
}
