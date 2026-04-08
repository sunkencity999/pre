// PRE Web GUI — Bootstrap, routing, state

(() => {
  // Init themes
  Themes.init();

  // Connect WebSocket
  WS.connect();

  // Track current session
  let currentSession = 'web:general';

  // ── WebSocket message handler ──
  WS.on('message', (msg) => {
    switch (msg.type) {
      case 'thinking':
        if (!Chat.isStreaming) Chat.startStream();
        Chat.appendThinking(msg.content);
        break;

      case 'token':
        if (!Chat.isStreaming) Chat.startStream();
        Chat.appendToken(msg.content);
        break;

      case 'tool_calls':
        if (msg.calls) {
          for (const tc of msg.calls) {
            Chat.addToolCall({
              name: tc.function?.name,
              args: tc.function?.arguments,
              status: 'pending',
            });
          }
        }
        break;

      case 'tool_call':
        Chat.addToolCall(msg);
        break;

      case 'tool_result':
        Chat.updateToolCard(msg.id, msg.output, msg.status);
        break;

      case 'done_partial':
        // Tool loop intermediate: end current stream before tool execution
        Chat.endStream(msg.stats);
        break;

      case 'done':
        Chat.endStream(msg.stats, msg.context);
        break;

      case 'confirm_request':
        showConfirmDialog(msg.id, msg.tool, msg.args);
        break;

      case 'error':
        Chat.endStream();
        Chat.addError(msg.message);
        break;

      case 'session_history':
        Chat.loadHistory(msg.messages);
        currentSession = msg.sessionId || currentSession;
        updateSessionName();
        break;
    }
  });

  WS.on('open', () => {
    // Load current session on connect
    WS.send({ type: 'switch_session', sessionId: currentSession });
    loadSessionList();
  });

  // ── Chat form ──
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage();
  });

  // Textarea auto-resize and keyboard shortcuts
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  function sendMessage() {
    const content = input.value.trim();
    if (!content || Chat.isStreaming) return;

    // Add user message to UI
    Chat.addMessage('user', content);

    // Send to server
    WS.send({ type: 'message', content });

    // Clear input
    input.value = '';
    input.style.height = 'auto';
  }

  // ── Sidebar ──
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');

  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });

  // Close sidebar on mobile when clicking outside
  document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768 &&
        sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        e.target !== sidebarToggle) {
      sidebar.classList.remove('open');
    }
  });

  // New session button
  document.getElementById('new-session-btn').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/sessions/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project: 'web', channel: 'general' }),
      });
      const data = await res.json();
      currentSession = data.id;
      WS.send({ type: 'switch_session', sessionId: currentSession });
      loadSessionList();
    } catch (err) {
      console.error('Failed to create session:', err);
    }
  });

  async function loadSessionList() {
    try {
      const res = await fetch('/api/sessions');
      const sessions = await res.json();
      const list = document.getElementById('session-list');
      list.innerHTML = '';

      for (const session of sessions.slice(0, 20)) {
        const item = document.createElement('div');
        item.className = 'session-item' + (session.id === currentSession ? ' active' : '');

        const info = document.createElement('button');
        info.className = 'session-item-info';
        info.innerHTML = `
          <div class="session-item-name">${escapeHtml(session.channel || 'general')}</div>
          <div class="session-item-preview">${escapeHtml(session.preview || 'New session')}</div>
          <div class="session-item-time">${formatTime(session.modified)}</div>
        `;
        info.addEventListener('click', () => {
          currentSession = session.id;
          WS.send({ type: 'switch_session', sessionId: currentSession });
          loadSessionList();
          if (window.innerWidth <= 768) sidebar.classList.remove('open');
        });

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'session-delete-btn';
        deleteBtn.title = 'Delete session';
        deleteBtn.innerHTML = '&times;';
        deleteBtn.addEventListener('click', async (e) => {
          e.stopPropagation();
          if (!confirm(`Delete session "${session.channel || session.id}"?`)) return;
          try {
            await fetch(`/api/sessions/${encodeURIComponent(session.id)}`, { method: 'DELETE' });
            if (session.id === currentSession) {
              currentSession = 'web:general';
              await fetch('/api/sessions/new', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ project: 'web', channel: 'general' }),
              });
              WS.send({ type: 'switch_session', sessionId: currentSession });
            }
            loadSessionList();
          } catch {}
        });

        item.appendChild(info);
        item.appendChild(deleteBtn);
        list.appendChild(item);
      }
    } catch {}
  }

  function updateSessionName() {
    const el = document.getElementById('session-name');
    if (el) {
      const parts = currentSession.split(':');
      el.textContent = parts[1] || parts[0] || 'general';
    }
  }

  function formatTime(isoString) {
    if (!isoString) return '';
    const d = new Date(isoString);
    const now = new Date();
    const diff = now - d;
    if (diff < 60000) return 'just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return d.toLocaleDateString();
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // ── Tool confirmation dialog ──
  function showConfirmDialog(id, toolName, args) {
    // Create overlay
    const overlay = document.createElement('div');
    overlay.className = 'confirm-overlay';
    overlay.innerHTML = `
      <div class="confirm-dialog">
        <div class="confirm-title">Confirm Tool Execution</div>
        <div class="confirm-body">
          <div class="confirm-tool">${escapeHtml(toolName)}</div>
          <pre class="confirm-args"><code>${escapeHtml(JSON.stringify(args, null, 2))}</code></pre>
        </div>
        <div class="confirm-actions">
          <button class="btn btn-ghost" id="confirm-deny">Deny</button>
          <button class="btn btn-primary" id="confirm-approve">Approve</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    overlay.querySelector('#confirm-approve').addEventListener('click', () => {
      WS.send({ type: 'confirm', id, approved: true });
      overlay.remove();
    });
    overlay.querySelector('#confirm-deny').addEventListener('click', () => {
      WS.send({ type: 'confirm', id, approved: false });
      overlay.remove();
    });
  }

  // ── Right panel close ──
  const panelClose = document.getElementById('panel-close');
  if (panelClose) {
    panelClose.addEventListener('click', () => {
      document.getElementById('right-panel').classList.add('hidden');
    });
  }
})();
