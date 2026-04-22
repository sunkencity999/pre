// PRE Web GUI — Chat UI, message rendering, streaming

const Chat = (() => {
  const messagesEl = () => document.getElementById('messages');
  let currentStreamEl = null;
  let currentThinkingEl = null;
  let streamContent = '';
  let thinkContent = '';
  let isStreaming = false;
  let currentDelegateStreamEl = null;
  let delegateStreamContent = '';

  // Agent tool names that get compact agent cards instead of verbose tool cards
  const AGENT_TOOL_NAMES = new Set(['spawn_agent', 'spawn_multi']);
  // Map toolId → { cardEl, toolLog, toolCount, agentId }
  const activeAgentCards = new Map();

  const DELEGATE_META = {
    claude: { name: 'Claude', color: '#cc785c' },
    codex:  { name: 'Codex',  color: '#10a37f' },
    gemini: { name: 'Gemini', color: '#4285f4' },
  };

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
    if (extra.delegate && role === 'assistant') {
      header.appendChild(createDelegateBadge(extra.delegate));
    } else {
      const roleSpan = document.createElement('span');
      roleSpan.className = 'message-role';
      roleSpan.textContent = role === 'user' ? 'You' : 'PRE';
      header.appendChild(roleSpan);
      if (extra.delegate && role === 'user') {
        const meta = DELEGATE_META[extra.delegate];
        if (meta) {
          const tag = document.createElement('span');
          tag.className = 'delegate-tag';
          tag.textContent = `→ ${meta.name}`;
          tag.style.color = meta.color;
          header.appendChild(tag);
        }
      }
    }
    if (role === 'assistant') {
      header.appendChild(createCopyBtn(() => content));
    }
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

    // Attachments (shown above content in user messages)
    if (extra.attachments && extra.attachments.length > 0) {
      const attBar = document.createElement('div');
      attBar.className = 'message-attachments';
      for (const att of extra.attachments) {
        if (att.isImage && att.dataUrl) {
          const thumb = document.createElement('div');
          thumb.className = 'msg-attachment-image';
          thumb.innerHTML = `<img src="${Markdown.escapeHtml(att.dataUrl)}" alt="${Markdown.escapeHtml(att.name)}">`;
          attBar.appendChild(thumb);
        } else {
          const chip = document.createElement('div');
          chip.className = 'msg-attachment-file';
          chip.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`
            + `<span>${Markdown.escapeHtml(att.name)}</span>`;
          attBar.appendChild(chip);
        }
      }
      msgEl.appendChild(attBar);
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
    hideThinkingIndicator();
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
    header.appendChild(createCopyBtn(() => streamContent));
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
   * Strip <tool_call> blocks from display text.
   * The server parses and executes these — they shouldn't be shown to the user.
   */
  function stripToolCallDisplay(text) {
    // Remove complete <tool_call>...</tool_call> blocks
    let cleaned = text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '');
    // Hide unclosed <tool_call> blocks (still streaming)
    const unclosedIdx = cleaned.indexOf('<tool_call>');
    if (unclosedIdx !== -1) {
      cleaned = cleaned.slice(0, unclosedIdx);
    }
    return cleaned.trim();
  }

  /**
   * Append a token to the streaming message
   */
  function appendToken(content) {
    if (!currentStreamEl) return;
    streamContent += content;
    const displayContent = stripToolCallDisplay(streamContent);
    currentStreamEl.innerHTML = Markdown.render(displayContent);
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
    isStreaming = false;
    hideThinkingIndicator();
    if (!currentStreamEl) return;

    // Remove streaming cursor
    currentStreamEl.classList.remove('streaming-cursor');

    // Final cleanup: strip any <tool_call> blocks from rendered content
    if (streamContent.includes('<tool_call>')) {
      const cleaned = stripToolCallDisplay(streamContent);
      currentStreamEl.innerHTML = Markdown.render(cleaned);
    }

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
   * Show a tool call card in the current message.
   * Agent tools (spawn_agent, spawn_multi) get compact agent cards.
   */
  function addToolCall(toolCall) {
    const toolName = toolCall.name || toolCall.function?.name || '';
    if (AGENT_TOOL_NAMES.has(toolName)) {
      addAgentCard(toolCall);
      return;
    }
    const msgEl = document.getElementById('streaming-message') ||
                  messagesEl().lastElementChild;
    if (msgEl) {
      msgEl.appendChild(createToolCard(toolCall));
      scrollToBottom();
    }
  }

  /**
   * Update an existing tool card with result.
   * Agent tool cards get routed to the compact card finalizer.
   */
  function updateToolCard(toolId, output, status) {
    // Check if this is an agent card
    if (activeAgentCards.has(toolId)) {
      finalizeAgentCard(toolId, output, status);
      return;
    }
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

  /**
   * Route agent status events to the matching compact agent card.
   * Falls back to the agent-status-bar if no card is tracking this agent.
   */
  function updateAgentStatus(event) {
    // Find the card tracking this agent by agentId
    let cardState = null;
    for (const [, st] of activeAgentCards) {
      if (st.agentId === event.id) { cardState = st; break; }
    }

    // For agent_started, match against pending cards (not yet assigned an agentId)
    if (!cardState && event.type === 'agent_started') {
      for (const [, st] of activeAgentCards) {
        if (st.agentId.startsWith('pending_')) { cardState = st; break; }
      }
    }

    if (cardState) {
      const card = cardState.cardEl;
      const icon = card.querySelector('.agent-card-icon');

      if (event.type === 'agent_started') {
        const oldId = cardState.agentId;
        cardState.agentId = event.id;
        if (event.sessionId) cardState.sessionId = event.sessionId;
        if (icon) { icon.textContent = '◉'; icon.classList.add('spinning'); }
        // Update feed entry with real agent ID and session
        if (window.AgentFeed) {
          window.AgentFeed.updateId(oldId, event.id);
          if (event.sessionId) window.AgentFeed.setSessionId(event.id, event.sessionId);
        }
      } else if (event.type === 'agent_tool') {
        cardState.toolCount++;
        const countEl = card.querySelector('.agent-card-tools-count');
        if (countEl) countEl.textContent = `${cardState.toolCount} tool${cardState.toolCount !== 1 ? 's' : ''}`;
        // Add to tool log
        const li = document.createElement('li');
        li.textContent = event.tool || '';
        cardState.toolLog.appendChild(li);
        // Push to agent feed
        if (window.AgentFeed) window.AgentFeed.addTool(cardState.agentId, event.tool);
      } else if (event.type === 'agent_progress') {
        const taskEl = card.querySelector('.agent-card-task');
        if (taskEl) taskEl.textContent = `[${event.current}/${event.total}] ${event.task || ''}`;
      } else if (event.type === 'agent_task_done') {
        // multi-agent sub-task done
        if (window.AgentFeed) window.AgentFeed.addTool(cardState.agentId, `Task ${event.current}/${event.total} complete`);
      } else if (event.type === 'agent_completed') {
        card.classList.add('agent-done');
        if (icon) { icon.textContent = '✓'; icon.classList.remove('spinning'); }
        const dur = event.duration ? (event.duration / 1000).toFixed(1) + 's' : '';
        const durEl = card.querySelector('.agent-card-duration');
        if (durEl) durEl.textContent = dur;
        if (window.AgentFeed) window.AgentFeed.complete(cardState.agentId, dur);
      } else if (event.type === 'agent_failed') {
        card.classList.add('agent-failed');
        if (icon) { icon.textContent = '✗'; icon.classList.remove('spinning'); }
        if (window.AgentFeed) window.AgentFeed.fail(cardState.agentId, event.error);
      }
      scrollToBottom();
      return;
    }

    // No matching card — fall back to legacy status bar (shouldn't normally happen)
    let container = document.getElementById('agent-status-bar');
    if (!container) {
      container = document.createElement('div');
      container.id = 'agent-status-bar';
      container.className = 'agent-status-bar';
      const msgEl = document.getElementById('streaming-message') ||
                    messagesEl().lastElementChild;
      if (msgEl) msgEl.appendChild(container);
    }
    if (event.type === 'agent_completed') {
      const secs = event.duration ? (event.duration / 1000).toFixed(1) + 's' : '';
      container.innerHTML = `<span class="agent-status-icon done">◉</span> Agent finished ${secs}`;
      setTimeout(() => container.remove(), 5000);
    } else if (event.type === 'agent_failed') {
      container.innerHTML = `<span class="agent-status-icon error">◉</span> Agent failed: ${Markdown.escapeHtml(event.error || '')}`;
      setTimeout(() => container.remove(), 8000);
    } else if (event.type === 'agent_tool') {
      container.innerHTML = `<span class="agent-status-icon spinning">◉</span> Agent using <strong>${Markdown.escapeHtml(event.tool || '')}</strong>...`;
    }
    scrollToBottom();
  }

  /**
   * Create a compact agent card for spawn_agent / spawn_multi tool calls.
   */
  function addAgentCard(toolCall) {
    const toolId = toolCall.id;
    const args = toolCall.args || toolCall.function?.arguments || {};
    const task = args.task || args.tasks
      ? (typeof args.tasks === 'string' ? args.tasks : JSON.stringify(args.tasks || args.task)).slice(0, 120)
      : 'Agent task';
    const isMulti = (toolCall.name || '').includes('multi');

    const card = document.createElement('div');
    card.className = 'agent-card';
    if (toolId) card.setAttribute('data-tool-id', toolId);

    const toolLog = document.createElement('ul');
    toolLog.className = 'agent-card-tool-log';

    card.innerHTML = `
      <div class="agent-card-header">
        <span class="agent-card-icon spinning">◉</span>
        <span class="agent-card-task">${Markdown.escapeHtml(task)}</span>
        <span class="agent-card-meta">
          <span class="agent-card-tools-count">0 tools</span>
          <span class="agent-card-duration"></span>
        </span>
        <svg class="agent-card-chevron" width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M6.5 3l5 5-5 5V3z"/></svg>
      </div>
      <div class="agent-card-details">
        <div class="agent-card-result-container"></div>
      </div>
    `;

    // Insert tool log into details
    card.querySelector('.agent-card-details').prepend(toolLog);

    // Toggle expand/collapse
    card.querySelector('.agent-card-header').addEventListener('click', () => {
      card.classList.toggle('expanded');
    });

    const msgEl = document.getElementById('streaming-message') ||
                  messagesEl().lastElementChild;
    if (msgEl) {
      msgEl.appendChild(card);
      scrollToBottom();
    }

    // Register a placeholder agentId — will be updated by agent_started event
    const agentId = `pending_${toolId}`;
    activeAgentCards.set(toolId, {
      cardEl: card,
      toolLog,
      toolCount: 0,
      agentId,
      isMulti,
    });

    // Register in agent feed
    if (window.AgentFeed) window.AgentFeed.add(agentId, task);
  }

  /**
   * Finalize a compact agent card when the tool_result arrives.
   */
  function finalizeAgentCard(toolId, output, status) {
    const state = activeAgentCards.get(toolId);
    if (!state) return;

    const card = state.cardEl;
    const icon = card.querySelector('.agent-card-icon');

    if (status === 'done' || status === 'skipped') {
      card.classList.add(status === 'skipped' ? 'agent-failed' : 'agent-done');
      if (icon) { icon.textContent = status === 'skipped' ? '⊘' : '✓'; icon.classList.remove('spinning'); }
    }

    // Add result to the details section (collapsed by default)
    if (output) {
      const container = card.querySelector('.agent-card-result-container');
      if (container) {
        const resultEl = document.createElement('div');
        resultEl.className = 'agent-card-result';
        const preview = output.length > 1500 ? output.slice(0, 1500) + '\n...' : output;
        resultEl.innerHTML = `<pre><code>${Markdown.escapeHtml(preview)}</code></pre>`;
        container.appendChild(resultEl);
      }
      // Session link (if agent has a dedicated session) or feed link
      const details = card.querySelector('.agent-card-details');
      if (details) {
        if (state.sessionId) {
          const sessionLink = document.createElement('button');
          sessionLink.className = 'agent-card-feed-link';
          sessionLink.innerHTML = '↗ Open Agent Session';
          sessionLink.addEventListener('click', (e) => {
            e.stopPropagation();
            if (window.AgentFeed) window.AgentFeed.openSession(state.sessionId);
          });
          details.appendChild(sessionLink);
        }
        const feedLink = document.createElement('button');
        feedLink.className = 'agent-card-feed-link';
        feedLink.innerHTML = '↗ View in Agent Feed';
        feedLink.addEventListener('click', (e) => {
          e.stopPropagation();
          if (window.AgentFeed) window.AgentFeed.open(state.agentId);
        });
        details.appendChild(feedLink);
      }

      // Update feed with result
      if (window.AgentFeed) window.AgentFeed.setResult(state.agentId, output);
    }

    activeAgentCards.delete(toolId);
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

  function addDocumentCard(title, docPath, format) {
    const container = messagesEl();
    const card = document.createElement('div');
    card.className = 'document-card';
    const icon = getDocIcon(format);
    const safePath = Markdown.escapeHtml(docPath);
    const viewable = ['txt', 'xml', 'html', 'pdf'].includes((format || '').toLowerCase());
    card.innerHTML = `
      <div class="document-card-header">
        <span class="document-card-icon">${icon}</span>
        <div class="document-card-info">
          <span class="document-card-title">${Markdown.escapeHtml(title)}</span>
          <span class="document-card-format">${Markdown.escapeHtml((format || 'txt').toUpperCase())}</span>
        </div>
      </div>
      <div class="document-card-actions">
        <a href="${safePath}" download class="btn btn-primary btn-sm">Download</a>
        ${viewable ? `<a href="${safePath}" target="_blank" class="btn btn-ghost btn-sm">Open</a>` : ''}
        <button class="btn btn-ghost btn-sm" onclick="window._revealInFinder('${safePath}')">Show in Finder</button>
      </div>
    `;
    container.appendChild(card);
    scrollToBottom();
  }

  function getDocIcon(format) {
    switch ((format || '').toLowerCase()) {
      case 'pdf': return '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>';
      case 'docx': case 'doc': return '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>';
      case 'xlsx': case 'xls': return '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>';
      default: return '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>';
    }
  }

  function addArtifactCard(title, artPath, type) {
    const container = messagesEl();
    const card = document.createElement('div');
    card.className = 'artifact-card';
    const safePath = Markdown.escapeHtml(artPath);
    const safeTitle = Markdown.escapeHtml(title);
    const safeType = Markdown.escapeHtml(type || 'html');
    const isHtml = (type || 'html').toLowerCase() === 'html';
    card.innerHTML = `
      <div class="artifact-card-header">
        <span class="artifact-card-icon">&#9672;</span>
        <span class="artifact-card-title">${safeTitle}</span>
        <span class="artifact-card-type">${safeType}</span>
      </div>
      <div class="artifact-card-actions">
        <a href="${safePath}" target="_blank" class="btn btn-primary btn-sm">Open</a>
        <a href="${safePath}" download class="btn btn-ghost btn-sm">Download</a>
        <button class="btn btn-ghost btn-sm" onclick="window._revealInFinder('${safePath}')">Show in Finder</button>
      </div>
      ${isHtml ? `<div class="artifact-card-share">
        <span class="artifact-share-label">Share as</span>
        <button class="btn btn-share btn-sm" onclick="window._exportArtifact(this, '${safePath}', 'pdf', '${safeTitle}')">PDF</button>
        <button class="btn btn-share btn-sm" onclick="window._exportArtifact(this, '${safePath}', 'png', '${safeTitle}')">PNG</button>
        <button class="btn btn-share btn-sm" onclick="window._exportArtifact(this, '${safePath}', 'html', '${safeTitle}')">Standalone HTML</button>
      </div>` : ''}
    `;
    container.appendChild(card);
    scrollToBottom();
  }

  function addImageCard(prompt, imgPath) {
    const container = messagesEl();
    const card = document.createElement('div');
    card.className = 'image-card';
    const safePath = Markdown.escapeHtml(imgPath);
    card.innerHTML = `
      <div class="image-card-preview">
        <img src="${safePath}" alt="${Markdown.escapeHtml(prompt)}" loading="lazy">
      </div>
      <div class="image-card-footer">
        <span class="image-card-prompt">${Markdown.escapeHtml(prompt.length > 120 ? prompt.slice(0, 120) + '...' : prompt)}</span>
        <div class="image-card-actions">
          <a href="${safePath}" target="_blank" class="btn btn-ghost btn-sm">Full size</a>
          <a href="${safePath}" download class="btn btn-primary btn-sm">Download</a>
          <button class="btn btn-ghost btn-sm" onclick="window._revealInFinder('${safePath}')">Show in Finder</button>
        </div>
      </div>
    `;
    container.appendChild(card);
    scrollToBottom();
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
      updateContextBar({ used: 0, max: 65536, pct: 0 });
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

      // Render persisted display records (artifacts, documents, images)
      if (msg.role === 'display') {
        if (msg.display === 'image') {
          addImageCard(msg.prompt || '', msg.path);
        } else if (msg.display === 'artifact') {
          addArtifactCard(msg.title || 'Artifact', msg.path, msg.artifactType || 'html');
        } else if (msg.display === 'document') {
          addDocumentCard(msg.title || 'Document', msg.path, msg.artifactType || 'txt');
        }
        continue;
      }

      if (msg.role === 'tool') {
        // Parse <tool_response> tags for rich rendering of images/artifacts/documents
        const responses = msg.content.match(/<tool_response\s+name="([^"]+)">([\s\S]*?)<\/tool_response>/g) || [];
        let renderedRich = false;

        for (const tr of responses) {
          const nameMatch = tr.match(/<tool_response\s+name="([^"]+)">/);
          const bodyMatch = tr.match(/<tool_response[^>]*>([\s\S]*?)<\/tool_response>/);
          const toolName = nameMatch ? nameMatch[1] : '';
          const body = bodyMatch ? bodyMatch[1].trim() : '';

          // Image generation → image card
          if (toolName === 'image_generate') {
            const pathMatch = body.match(/View at: (\/artifacts\/[^\s]+)/);
            if (pathMatch) {
              const promptMatch = body.match(/^Image generated:.*?([^\/]+)\.\w+$/m);
              addImageCard(promptMatch ? promptMatch[1].replace(/_/g, ' ') : '', pathMatch[1]);
              renderedRich = true;
              continue;
            }
          }

          // Artifact → artifact card
          if (toolName === 'artifact') {
            const pathMatch = body.match(/View at: (\/artifacts\/[^\s]+)/);
            if (pathMatch) {
              const titleMatch = body.match(/Artifact saved: .+\/([^\/]+)\.\w+/);
              addArtifactCard(titleMatch ? decodeURIComponent(titleMatch[1]).replace(/_/g, ' ') : 'Artifact', pathMatch[1], 'html');
              renderedRich = true;
              continue;
            }
          }

          // Document → document card
          if (toolName === 'document') {
            const pathMatch = body.match(/Download: (\/artifacts\/[^\s]+)/);
            if (pathMatch) {
              const ext = pathMatch[1].split('.').pop() || 'txt';
              const titleMatch = body.match(/Document created: ([^(]+)/);
              addDocumentCard(titleMatch ? titleMatch[1].trim() : 'Document', pathMatch[1], ext);
              renderedRich = true;
              continue;
            }
          }
        }

        // Fallback: show as generic tool result card if no rich cards were rendered
        if (!renderedRich) {
          const container2 = messagesEl();
          const card = document.createElement('div');
          card.className = 'tool-card';
          card.style.maxWidth = '800px';
          card.style.margin = '0 auto 8px';
          const preview = msg.content.slice(0, 300);
          card.innerHTML = `<div class="tool-card-header"><span>Tool Result</span></div>
            <div class="tool-card-body"><pre><code>${Markdown.escapeHtml(preview)}${msg.content.length > 300 ? '...' : ''}</code></pre></div>`;
          container2.appendChild(card);
        }
        continue;
      }
      addMessage(msg.role, msg.content, {
        toolCalls: msg.tool_calls,
        delegate: msg.delegate,
      });
    }

    // Estimate context usage from session history
    // ~1 token per 4 chars is a rough approximation
    // Include tool_calls arguments size (can be large for artifacts)
    const totalChars = messages.reduce((sum, m) => {
      if (m.role === 'display') return sum; // Display records aren't sent to the model
      let chars = (m.content || '').length;
      if (m.tool_calls) {
        try { chars += JSON.stringify(m.tool_calls).length; } catch {}
      }
      return sum + chars;
    }, 0);
    const estimatedTokens = Math.round(totalChars / 4);
    const max = 65536;
    updateContextBar({
      used: estimatedTokens,
      max,
      pct: Math.min(99, Math.round(estimatedTokens * 100 / max)),
    });

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

  function createCopyBtn(getContent) {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.title = 'Copy response';
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(getContent());
        btn.classList.add('copied');
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
        setTimeout(() => {
          btn.classList.remove('copied');
          btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
        }, 2000);
      } catch {}
    });
    return btn;
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

  /** Show a thinking placeholder in the chat area while waiting for TTFT */
  function showThinkingIndicator() {
    hideThinkingIndicator();
    const container = messagesEl();
    const welcome = container.querySelector('.welcome');
    if (welcome) welcome.remove();

    const el = document.createElement('div');
    el.className = 'message assistant';
    el.id = 'thinking-indicator-msg';
    el.innerHTML = `
      <div class="message-header">
        <span class="message-role">PRE</span>
      </div>
      <div class="thinking-indicator-content">
        <div class="thinking-indicator-dots">
          <span></span><span></span><span></span>
        </div>
        <span class="thinking-indicator-label">Thinking</span>
      </div>
    `;
    container.appendChild(el);
    scrollToBottom();
  }

  function hideThinkingIndicator() {
    const el = document.getElementById('thinking-indicator-msg');
    if (el) el.remove();
  }

  // ── Delegate (frontier AI) streaming ──

  /**
   * Start streaming a delegated response from a frontier AI
   */
  function startDelegateStream(target) {
    hideThinkingIndicator();
    const container = messagesEl();
    const welcome = container.querySelector('.welcome');
    if (welcome) welcome.remove();

    isStreaming = true;
    delegateStreamContent = '';

    const meta = DELEGATE_META[target] || { name: target, color: 'var(--primary)' };

    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant';
    msgEl.id = 'streaming-message';

    const header = document.createElement('div');
    header.className = 'message-header';
    const badge = document.createElement('span');
    badge.className = `message-role delegate-badge delegate-badge-${target}`;
    badge.textContent = meta.name;
    badge.style.setProperty('--delegate-color', meta.color);
    header.appendChild(badge);
    header.appendChild(createCopyBtn(() => delegateStreamContent));
    msgEl.appendChild(header);

    currentDelegateStreamEl = document.createElement('div');
    currentDelegateStreamEl.className = 'message-content streaming-cursor';
    msgEl.appendChild(currentDelegateStreamEl);

    container.appendChild(msgEl);
    scrollToBottom();
    setTypingIndicator(`Waiting for ${meta.name}...`);
  }

  /**
   * Append a token to the delegate stream
   */
  function appendDelegateToken(content) {
    if (!currentDelegateStreamEl) return;
    delegateStreamContent += content;
    currentDelegateStreamEl.innerHTML = Markdown.render(delegateStreamContent);
    scrollToBottom();
    setTypingIndicator('Receiving...');
  }

  /**
   * End the delegate stream
   */
  function endDelegateStream(target, duration) {
    isStreaming = false;
    hideThinkingIndicator();
    if (!currentDelegateStreamEl) return;

    currentDelegateStreamEl.classList.remove('streaming-cursor');

    const msgEl = document.getElementById('streaming-message');
    if (msgEl && duration) {
      const statsEl = document.createElement('div');
      statsEl.className = 'stats-line';
      const secs = (duration / 1000).toFixed(1);
      const meta = DELEGATE_META[target] || { name: target };
      statsEl.textContent = `${meta.name} · ${secs}s`;
      msgEl.appendChild(statsEl);
    }

    if (msgEl) msgEl.removeAttribute('id');
    currentDelegateStreamEl = null;
    setTypingIndicator('');
    scrollToBottom();
  }

  /**
   * Create a delegate badge element for history rendering
   */
  function createDelegateBadge(target) {
    const meta = DELEGATE_META[target] || { name: target, color: 'var(--primary)' };
    const badge = document.createElement('span');
    badge.className = `message-role delegate-badge delegate-badge-${target}`;
    badge.textContent = meta.name;
    badge.style.setProperty('--delegate-color', meta.color);
    return badge;
  }

  return {
    addMessage,
    startStream,
    appendToken,
    appendThinking,
    endStream,
    startDelegateStream,
    appendDelegateToken,
    endDelegateStream,
    addToolCall,
    updateToolCard,
    updateAgentStatus,
    addDocumentCard,
    addArtifactCard,
    addImageCard,
    addError,
    loadHistory,
    updateContextBar,
    showThinkingIndicator,
    get isStreaming() { return isStreaming; },
  };
})();
