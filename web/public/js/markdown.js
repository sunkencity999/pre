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
          html.push(`<pre><code class="lang-${escapeHtml(codeBlockLang)}">${escapeHtml(codeBlockContent.trimEnd())}</code></pre>`);
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

    // Close any open blocks
    if (inCodeBlock) {
      html.push(`<pre><code>${escapeHtml(codeBlockContent.trimEnd())}</code></pre>`);
    }
    if (inList) {
      html.push(listType === 'ul' ? '</ul>' : '</ol>');
    }

    return html.join('\n');
  }

  return { render, escapeHtml };
})();
