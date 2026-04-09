// PRE Web GUI — Bootstrap, routing, state

(() => {
  // Init themes
  Themes.init();

  // Connect WebSocket
  WS.connect();

  // Track current session — restore across refreshes, default to most recent
  let currentSession = sessionStorage.getItem('pre-session') || null;
  function setCurrentSession(id) {
    currentSession = id;
    if (id) sessionStorage.setItem('pre-session', id);
  }

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

      case 'artifact':
        Chat.addArtifactCard(msg.title, msg.path, msg.artifactType);
        break;

      case 'document':
        Chat.addDocumentCard(msg.title, msg.path, msg.artifactType);
        break;

      case 'session_renamed':
        if (msg.sessionId === currentSession) {
          currentDisplayName = msg.name;
          updateSessionName();
        }
        loadSessionList();
        break;

      case 'error':
        Chat.endStream();
        Chat.addError(msg.message);
        break;

      case 'session_history':
        Chat.loadHistory(msg.messages);
        setCurrentSession(msg.sessionId || currentSession);
        updateSessionName();
        break;
    }
  });

  WS.on('open', async () => {
    // Load session list first so we can pick the most recent if needed
    await loadSessionList();
    if (!currentSession) {
      // No saved session — default to most recent
      try {
        const res = await fetch('/api/sessions');
        const sessions = await res.json();
        if (sessions.length > 0) {
          setCurrentSession(sessions[0].id); // already sorted by most recent
        }
      } catch {}
    }
    if (!currentSession) setCurrentSession('web:general');
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

  // Track which drag is active
  let dragSessionId = null;

  // New session button (ungrouped)
  document.getElementById('new-session-btn').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/sessions/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project: 'web', channel: 'general' }),
      });
      const data = await res.json();
      setCurrentSession(data.id);
      WS.send({ type: 'switch_session', sessionId: currentSession });
      loadSessionList();
    } catch (err) {
      console.error('Failed to create session:', err);
    }
  });

  // New project button
  document.getElementById('new-project-btn').addEventListener('click', () => {
    const list = document.getElementById('session-list');
    // Insert inline input at top of list
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'project-rename-input';
    input.placeholder = 'Project name...';
    list.prepend(input);
    input.focus();

    let saved = false;
    const save = async () => {
      if (saved) return;
      saved = true;
      const name = input.value.trim();
      input.remove();
      if (!name) return;
      try {
        await fetch('/api/projects', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
        loadSessionList();
      } catch {}
    };
    input.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') { ev.preventDefault(); save(); }
      if (ev.key === 'Escape') { input.remove(); }
    });
    input.addEventListener('blur', save);
  });

  let currentDisplayName = null;

  // ── Build a session item element ──
  function buildSessionItem(session) {
    const item = document.createElement('div');
    item.className = 'session-item' + (session.id === currentSession ? ' active' : '');
    item.draggable = true;
    item.dataset.sessionId = session.id;

    if (session.id === currentSession) {
      currentDisplayName = session.displayName || null;
      updateSessionName();
    }

    const displayLabel = session.displayName || session.channel || 'general';

    const info = document.createElement('button');
    info.className = 'session-item-info';
    info.innerHTML = `
      <div class="session-item-name">${escapeHtml(displayLabel)}</div>
      <div class="session-item-preview">${escapeHtml(session.preview || 'New session')}</div>
      <div class="session-item-time">${formatTime(session.modified)}</div>
    `;
    info.addEventListener('click', () => {
      if (session.id === currentSession) return;
      setCurrentSession(session.id);
      WS.send({ type: 'switch_session', sessionId: currentSession });
      loadSessionList();
      if (window.innerWidth <= 768) sidebar.classList.remove('open');
    });
    info.addEventListener('dblclick', (e) => {
      e.preventDefault();
      e.stopPropagation();
      startInlineRename(info, session);
    });

    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'session-delete-btn';
    deleteBtn.title = 'Delete session';
    deleteBtn.innerHTML = '&times;';
    deleteBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete session "${displayLabel}"?`)) return;
      try {
        await fetch(`/api/sessions/${encodeURIComponent(session.id)}`, { method: 'DELETE' });
        if (session.id === currentSession) {
          setCurrentSession('web:general');
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

    // Drag events
    item.addEventListener('dragstart', (e) => {
      dragSessionId = session.id;
      item.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', session.id);
      // Show ungrouped drop zone
      const dropZone = document.querySelector('.ungrouped-drop');
      if (dropZone) dropZone.classList.add('visible');
    });
    item.addEventListener('dragend', () => {
      dragSessionId = null;
      item.classList.remove('dragging');
      const dropZone = document.querySelector('.ungrouped-drop');
      if (dropZone) dropZone.classList.remove('visible');
      document.querySelectorAll('.drag-over').forEach(el => el.classList.remove('drag-over'));
    });

    item.appendChild(info);
    item.appendChild(deleteBtn);
    return item;
  }

  // ── Collapsed state persistence ──
  function getExpandedProjects() {
    try { return JSON.parse(sessionStorage.getItem('pre-expanded-projects') || '{}'); } catch { return {}; }
  }
  function setProjectExpanded(slug, expanded) {
    const state = getExpandedProjects();
    state[slug] = expanded;
    sessionStorage.setItem('pre-expanded-projects', JSON.stringify(state));
  }

  async function loadSessionList() {
    try {
      const [sessionsRes, projectsRes] = await Promise.all([
        fetch('/api/sessions'),
        fetch('/api/projects'),
      ]);
      const sessions = await sessionsRes.json();
      const projects = await projectsRes.json();

      const list = document.getElementById('session-list');
      list.innerHTML = '';

      const expandedState = getExpandedProjects();

      // Group sessions by projectSlug
      const grouped = {};
      const ungrouped = [];
      for (const s of sessions) {
        if (s.projectSlug) {
          if (!grouped[s.projectSlug]) grouped[s.projectSlug] = [];
          grouped[s.projectSlug].push(s);
        } else {
          ungrouped.push(s);
        }
      }

      // Render ungrouped drop zone (hidden until drag starts)
      const ungroupedDrop = document.createElement('div');
      ungroupedDrop.className = 'ungrouped-drop';
      ungroupedDrop.textContent = 'Drop here to ungroup';
      ungroupedDrop.addEventListener('dragover', (e) => { e.preventDefault(); ungroupedDrop.classList.add('drag-over'); });
      ungroupedDrop.addEventListener('dragleave', () => { ungroupedDrop.classList.remove('drag-over'); });
      ungroupedDrop.addEventListener('drop', async (e) => {
        e.preventDefault();
        ungroupedDrop.classList.remove('drag-over');
        ungroupedDrop.classList.remove('visible');
        const sid = e.dataTransfer.getData('text/plain');
        if (sid) {
          await fetch(`/api/sessions/${encodeURIComponent(sid)}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ projectSlug: null }),
          });
          loadSessionList();
        }
      });
      list.appendChild(ungroupedDrop);

      // Render projects
      for (const proj of projects) {
        const group = document.createElement('div');
        group.className = 'project-group';
        const isExpanded = expandedState[proj.slug] !== false; // default expanded
        if (isExpanded) group.classList.add('expanded');

        const header = document.createElement('div');
        header.className = 'project-header';
        header.innerHTML = `
          <svg class="project-chevron" viewBox="0 0 16 16" fill="currentColor"><path d="M6.5 3l5 5-5 5V3z"/></svg>
          <svg class="project-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M1 3.5A1.5 1.5 0 0 1 2.5 2h3.879a1.5 1.5 0 0 1 1.06.44l1.122 1.12A1.5 1.5 0 0 0 9.62 4H13.5A1.5 1.5 0 0 1 15 5.5v7a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 1 12.5v-9z"/></svg>
          <span class="project-name">${escapeHtml(proj.name)}</span>
          <span class="project-count">${proj.sessionCount}</span>
          <div class="project-actions">
            <button class="project-add-btn" title="New session in project">+</button>
            <button class="project-delete-btn" title="Delete project">&times;</button>
          </div>
        `;

        // Toggle collapse
        header.addEventListener('click', (e) => {
          if (e.target.closest('.project-actions')) return;
          group.classList.toggle('expanded');
          setProjectExpanded(proj.slug, group.classList.contains('expanded'));
        });

        // Double-click to rename project
        header.addEventListener('dblclick', (e) => {
          e.preventDefault();
          e.stopPropagation();
          const nameEl = header.querySelector('.project-name');
          if (!nameEl) return;
          const currentName = nameEl.textContent;
          const input = document.createElement('input');
          input.type = 'text';
          input.className = 'project-rename-input';
          input.value = currentName;
          nameEl.replaceWith(input);
          input.focus();
          input.select();
          let saved = false;
          const doSave = async () => {
            if (saved) return;
            saved = true;
            const newName = input.value.trim();
            if (newName && newName !== currentName) {
              await fetch(`/api/projects/${encodeURIComponent(proj.slug)}/rename`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: newName }),
              });
            }
            loadSessionList();
          };
          input.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter') { ev.preventDefault(); doSave(); }
            if (ev.key === 'Escape') { loadSessionList(); }
          });
          input.addEventListener('blur', doSave);
        });

        // Add session to project
        header.querySelector('.project-add-btn').addEventListener('click', async (e) => {
          e.stopPropagation();
          try {
            const res = await fetch('/api/sessions/new', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ project: 'web', channel: 'general', projectSlug: proj.slug }),
            });
            const data = await res.json();
            setCurrentSession(data.id);
            setProjectExpanded(proj.slug, true);
            WS.send({ type: 'switch_session', sessionId: currentSession });
            loadSessionList();
          } catch {}
        });

        // Delete project
        header.querySelector('.project-delete-btn').addEventListener('click', async (e) => {
          e.stopPropagation();
          if (!confirm(`Delete project "${proj.name}"? Sessions will be ungrouped, not deleted.`)) return;
          await fetch(`/api/projects/${encodeURIComponent(proj.slug)}`, { method: 'DELETE' });
          loadSessionList();
        });

        // Drop target for drag-and-drop
        header.addEventListener('dragover', (e) => { e.preventDefault(); header.classList.add('drag-over'); });
        header.addEventListener('dragleave', () => { header.classList.remove('drag-over'); });
        header.addEventListener('drop', async (e) => {
          e.preventDefault();
          header.classList.remove('drag-over');
          const sid = e.dataTransfer.getData('text/plain');
          if (sid) {
            await fetch(`/api/sessions/${encodeURIComponent(sid)}/move`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ projectSlug: proj.slug }),
            });
            setProjectExpanded(proj.slug, true);
            loadSessionList();
          }
        });

        const sessionsContainer = document.createElement('div');
        sessionsContainer.className = 'project-sessions';

        const projectSessions = grouped[proj.slug] || [];
        for (const session of projectSessions) {
          sessionsContainer.appendChild(buildSessionItem(session));
        }

        group.appendChild(header);
        group.appendChild(sessionsContainer);
        list.appendChild(group);
      }

      // Render ungrouped sessions
      if (ungrouped.length > 0 && projects.length > 0) {
        const label = document.createElement('div');
        label.className = 'sidebar-label';
        label.style.marginTop = '8px';
        label.textContent = 'Recent';
        list.appendChild(label);
      }
      for (const session of ungrouped.slice(0, 20)) {
        list.appendChild(buildSessionItem(session));
      }
    } catch {}
  }

  // Inline rename helper for sidebar items
  function startInlineRename(infoEl, session) {
    const nameEl = infoEl.querySelector('.session-item-name');
    if (!nameEl) return;
    const currentName = nameEl.textContent;
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'session-rename-input';
    input.value = currentName;
    nameEl.replaceWith(input);
    input.focus();
    input.select();

    let saved = false;
    const save = async () => {
      if (saved) return;
      saved = true;
      const newName = input.value.trim();
      try {
        await fetch(`/api/sessions/${encodeURIComponent(session.id)}/rename`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: newName || null }),
        });
      } catch {}
      loadSessionList();
    };
    input.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') { ev.preventDefault(); save(); }
      if (ev.key === 'Escape') { loadSessionList(); }
    });
    input.addEventListener('blur', save);
  }

  // Topbar session name — click to rename
  function setupTopbarRename() {
    const el = document.getElementById('session-name');
    if (!el) return;
    el.style.cursor = 'pointer';
    el.title = 'Click to rename session';
    el.addEventListener('click', () => {
      const currentName = el.textContent;
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'session-rename-input topbar-rename';
      input.value = currentName;
      el.replaceWith(input);
      input.focus();
      input.select();

      let saved = false;
      const save = async () => {
        if (saved) return;
        saved = true;
        const newName = input.value.trim();
        try {
          await fetch(`/api/sessions/${encodeURIComponent(currentSession)}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName || null }),
          });
        } catch {}
        // Restore the span
        const newEl = document.createElement('span');
        newEl.id = 'session-name';
        input.replaceWith(newEl);
        loadSessionList();
        setupTopbarRename();
      };
      input.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') { ev.preventDefault(); save(); }
        if (ev.key === 'Escape') {
          const newEl = document.createElement('span');
          newEl.id = 'session-name';
          newEl.textContent = currentName;
          input.replaceWith(newEl);
          setupTopbarRename();
        }
      });
      input.addEventListener('blur', save);
    });
  }
  setupTopbarRename();

  function updateSessionName() {
    const el = document.getElementById('session-name');
    if (el) {
      if (currentDisplayName) {
        el.textContent = currentDisplayName;
      } else {
        const parts = currentSession.split(':');
        el.textContent = parts[1] || parts[0] || 'general';
      }
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
