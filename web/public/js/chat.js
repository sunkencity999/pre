// PRE Web GUI — Chat UI, message rendering, streaming

const Chat = (() => {
  const messagesEl = () => document.getElementById('messages');
  let currentStreamEl = null;
  let currentThinkingEl = null;
  let streamContent = '';
  let thinkContent = '';
  let isStreaming = false;

  /**
   * Add a message to the chat
   */
  function addMessage(role, content, extra = {}) {
    const container = messagesEl();

    // Remove welcome screen on first message
    const welcome = container.querySelector('.welcome');
    if (welcome) welcome.remove();

    const msgEl = document.createElement('div');
    msgEl.className = `message ${role}`;

    const header = document.createElement('div');
    header.className = 'message-header';
    const roleSpan = document.createElement('span');
    roleSpan.className = 'message-role';
    roleSpan.textContent = role === 'user' ? 'You' : 'PRE';
    header.appendChild(roleSpan);
    msgEl.appendChild(header);

    // Thinking block
    if (extra.thinking) {
      const details = document.createElement('details');
      details.className = 'thinking-block';
      const summary = document.createElement('summary');
      summary.textContent = 'Reasoning';
      details.appendChild(summary);
      const thinkBody = document.createElement('div');
      thinkBody.innerHTML = Markdown.render(extra.thinking);
      details.appendChild(thinkBody);
      msgEl.appendChild(details);
    }

    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';
    if (role === 'user') {
      contentEl.textContent = content;
    } else {
      contentEl.innerHTML = Markdown.render(content);
    }
    msgEl.appendChild(contentEl);

    // Stats
    if (extra.stats) {
      const statsEl = document.createElement('div');
      statsEl.className = 'stats-line';
      const parts = [];
      if (extra.stats.tok_s) parts.push(`${extra.stats.tok_s} tok/s`);
      if (extra.stats.eval_count) parts.push(`${extra.stats.eval_count} tokens`);
      if (extra.stats.ttft_ms) parts.push(`TTFT ${extra.stats.ttft_ms}ms`);
      statsEl.textContent = parts.join(' · ');
      msgEl.appendChild(statsEl);
    }

    // Tool calls
    if (extra.toolCalls) {
      for (const tc of extra.toolCalls) {
        msgEl.appendChild(createToolCard(tc));
      }
    }

    container.appendChild(msgEl);
    scrollToBottom();
    return msgEl;
  }

  /**
   * Start streaming a new assistant message
   */
  function startStream() {
    const container = messagesEl();
    const welcome = container.querySelector('.welcome');
    if (welcome) welcome.remove();

    isStreaming = true;
    streamContent = '';
    thinkContent = '';

    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant';
    msgEl.id = 'streaming-message';

    const header = document.createElement('div');
    header.className = 'message-header';
    const roleSpan = document.createElement('span');
    roleSpan.className = 'message-role';
    roleSpan.textContent = 'PRE';
    header.appendChild(roleSpan);
    msgEl.appendChild(header);

    // Thinking block (collapsible, starts open during stream)
    const thinkDetails = document.createElement('details');
    thinkDetails.className = 'thinking-block';
    thinkDetails.id = 'stream-thinking';
    thinkDetails.style.display = 'none';
    thinkDetails.open = true;
    const summary = document.createElement('summary');
    summary.textContent = 'Reasoning';
    thinkDetails.appendChild(summary);
    currentThinkingEl = document.createElement('div');
    thinkDetails.appendChild(currentThinkingEl);
    msgEl.appendChild(thinkDetails);

    currentStreamEl = document.createElement('div');
    currentStreamEl.className = 'message-content streaming-cursor';
    msgEl.appendChild(currentStreamEl);

    container.appendChild(msgEl);
    scrollToBottom();

    // Show typing indicator
    setTypingIndicator('Thinking...');
  }

  /**
   * Append a token to the streaming message
   */
  function appendToken(content) {
    if (!currentStreamEl) return;
    streamContent += content;
    currentStreamEl.innerHTML = Markdown.render(streamContent);
    scrollToBottom();
    setTypingIndicator('Generating...');
  }

  /**
   * Append thinking content
   */
  function appendThinking(content) {
    if (!currentThinkingEl) return;
    thinkContent += content;
    const thinkBlock = document.getElementById('stream-thinking');
    if (thinkBlock) thinkBlock.style.display = '';
    currentThinkingEl.innerHTML = Markdown.render(thinkContent);
    scrollToBottom();
  }

  /**
   * End streaming — finalize the message
   */
  function endStream(stats, context) {
    if (!currentStreamEl) return;
    isStreaming = false;

    // Remove streaming cursor
    currentStreamEl.classList.remove('streaming-cursor');

    // Close thinking details
    const thinkBlock = document.getElementById('stream-thinking');
    if (thinkBlock && thinkContent) {
      thinkBlock.open = false;
    }

    // Add stats line
    if (stats) {
      const msgEl = document.getElementById('streaming-message');
      if (msgEl) {
        const statsEl = document.createElement('div');
        statsEl.className = 'stats-line';
        const parts = [];
        if (stats.tok_s) parts.push(`${stats.tok_s} tok/s`);
        if (stats.eval_count) parts.push(`${stats.eval_count} tokens`);
        if (stats.ttft_ms) parts.push(`TTFT ${stats.ttft_ms}ms`);
        statsEl.textContent = parts.join(' · ');
        msgEl.appendChild(statsEl);
      }
    }

    // Update context bar
    if (context) {
      updateContextBar(context);
    }

    // Clean up streaming state
    const msgEl = document.getElementById('streaming-message');
    if (msgEl) msgEl.removeAttribute('id');
    currentStreamEl = null;
    currentThinkingEl = null;

    setTypingIndicator('');
    scrollToBottom();
  }

  /**
   * Show a tool call card in the current message
   */
  function addToolCall(toolCall) {
    const msgEl = document.getElementById('streaming-message') ||
                  messagesEl().lastElementChild;
    if (msgEl) {
      msgEl.appendChild(createToolCard(toolCall));
      scrollToBottom();
    }
  }

  /**
   * Update an existing tool card with result
   */
  function updateToolCard(toolId, output, status) {
    const card = document.querySelector(`[data-tool-id="${toolId}"]`);
    if (!card) return;

    const statusEl = card.querySelector('.tool-card-status');
    if (statusEl) {
      statusEl.textContent = status || 'done';
      statusEl.className = 'tool-card-status ' + (status === 'done' ? 'status-done' : status === 'skipped' ? 'status-skipped' : '');
    }

    if (output) {
      let body = card.querySelector('.tool-card-body');
      if (!body) {
        body = document.createElement('div');
        body.className = 'tool-card-body';
        card.appendChild(body);
      }
      const preview = output.length > 500 ? output.slice(0, 500) + '...' : output;
      body.innerHTML = `<pre><code>${Markdown.escapeHtml(preview)}</code></pre>`;
    }
  }

  function createToolCard(tc) {
    const card = document.createElement('div');
    card.className = 'tool-card';
    if (tc.id) card.setAttribute('data-tool-id', tc.id);

    const header = document.createElement('div');
    header.className = 'tool-card-header';
    header.innerHTML = `<span>Tool</span> <span class="tool-card-name">${Markdown.escapeHtml(tc.name || tc.function?.name || 'unknown')}</span>`;

    const status = document.createElement('span');
    status.className = 'tool-card-status';
    status.textContent = tc.status || 'pending';
    header.appendChild(status);

    card.appendChild(header);

    // Show args preview
    const args = tc.args || tc.function?.arguments;
    if (args) {
      const body = document.createElement('div');
      body.className = 'tool-card-body';
      const preview = typeof args === 'string' ? args : JSON.stringify(args, null, 2);
      body.innerHTML = `<pre><code>${Markdown.escapeHtml(preview.slice(0, 500))}</code></pre>`;
      card.appendChild(body);
    }

    return card;
  }

  function addError(message) {
    const container = messagesEl();
    const errEl = document.createElement('div');
    errEl.className = 'message assistant';
    errEl.innerHTML = `<div class="message-content" style="color:var(--danger)">Error: ${Markdown.escapeHtml(message)}</div>`;
    container.appendChild(errEl);
    scrollToBottom();
  }

  /**
   * Load session history into chat
   */
  function loadHistory(messages) {
    const container = messagesEl();
    container.innerHTML = '';

    if (!messages || messages.length === 0) {
      container.innerHTML = `
        <div class="welcome">
          <div class="welcome-logo">PRE</div>
          <p class="welcome-text">Personal Reasoning Engine</p>
          <p class="welcome-sub">Local AI assistant running on Apple Silicon. All data stays on this machine.</p>
        </div>`;
      return;
    }

    for (const msg of messages) {
      if (msg.role === 'system') continue;
      if (msg.role === 'tool') {
        // Show tool results as collapsed cards
        const container2 = messagesEl();
        const card = document.createElement('div');
        card.className = 'tool-card';
        card.style.maxWidth = '800px';
        card.style.margin = '0 auto 8px';
        const preview = msg.content.slice(0, 300);
        card.innerHTML = `<div class="tool-card-header"><span>Tool Result</span></div>
          <div class="tool-card-body"><pre><code>${Markdown.escapeHtml(preview)}${msg.content.length > 300 ? '...' : ''}</code></pre></div>`;
        container2.appendChild(card);
        continue;
      }
      addMessage(msg.role, msg.content, {
        toolCalls: msg.tool_calls,
      });
    }
    scrollToBottom();
  }

  function updateContextBar(ctx) {
    const fill = document.getElementById('context-fill');
    const label = document.getElementById('context-label');
    if (!fill || !label) return;

    fill.style.width = `${ctx.pct}%`;
    fill.className = 'context-fill' +
      (ctx.pct > 75 ? ' danger' : ctx.pct > 50 ? ' warning' : '');
    label.textContent = `${ctx.pct}% of ${Math.round(ctx.max / 1024)}K`;
  }

  function scrollToBottom() {
    const el = messagesEl();
    requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
  }

  function setTypingIndicator(text) {
    const el = document.getElementById('typing-indicator');
    if (!el) return;
    if (text) {
      el.textContent = text;
      el.classList.remove('hidden');
    } else {
      el.classList.add('hidden');
    }
  }

  return {
    addMessage,
    startStream,
    appendToken,
    appendThinking,
    endStream,
    addToolCall,
    updateToolCard,
    addError,
    loadHistory,
    updateContextBar,
    get isStreaming() { return isStreaming; },
  };
})();
