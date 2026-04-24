// PRE Web GUI — Bootstrap, routing, state

(() => {
  // Init themes
  Themes.init();

  // Connect WebSocket
  WS.connect();

  // Track current session — restore across refreshes, default to most recent
  let currentSession = sessionStorage.getItem('pre-session') || null;

  // ── Delegate (frontier AI) state ──
  let currentDelegate = 'local'; // 'local', 'claude', 'codex', 'gemini'
  let delegateAvailability = {}; // populated on connect

  // ── Deep Research mode ──
  let researchMode = false;
  let researchFrontier = false; // false = local multi-pass, or 'claude'/'codex'/'gemini'
  let researchProgressEl = null; // tracks the current progress card in the chat
  function setCurrentSession(id) {
    currentSession = id;
    if (id) sessionStorage.setItem('pre-session', id);
    // Clear Argus reactions when switching sessions — they're session-specific
    if (window.Argus) window.Argus.clear();
  }

  // ── Background job tracking (cron/trigger events) ──
  const backgroundJobs = new Map(); // cronSessionId → { tokens, toolCount }

  function handleBackgroundEvent(msg) {
    const jobId = msg.cronJobId || msg.cronSessionId;
    const sessionId = msg.cronSessionId;
    if (!backgroundJobs.has(sessionId)) {
      backgroundJobs.set(sessionId, { tokens: [], toolCount: 0, description: '' });
      // Create a feed entry for this background job
      const desc = msg.cronDescription || jobId || 'Background job';
      backgroundJobs.get(sessionId).description = desc;
      if (window.AgentFeed) window.AgentFeed.add(sessionId, desc, sessionId);
    }
    const job = backgroundJobs.get(sessionId);

    if (msg.type === 'token' && msg.content) {
      job.tokens.push(msg.content);
    } else if (msg.type === 'tool_call') {
      job.toolCount++;
      if (window.AgentFeed) window.AgentFeed.addTool(sessionId, msg.name || 'tool');
    } else if (msg.type === 'done') {
      const result = job.tokens.join('');
      if (window.AgentFeed) {
        window.AgentFeed.setResult(sessionId, result);
        window.AgentFeed.complete(sessionId);
      }
      backgroundJobs.delete(sessionId);
    }
    // agent_status events from cron sub-agents (type is 'agent_status', sub-type in event fields)
    if (msg.type === 'agent_status' && msg.tool && window.AgentFeed) {
      window.AgentFeed.addTool(sessionId, msg.tool);
    }
  }

  // ── WebSocket message handler ──
  WS.on('message', (msg) => {
    // Route background job events (cron/triggers) to Agent Feed instead of current chat
    if (msg.cronSessionId) {
      handleBackgroundEvent(msg);
      // Still allow cron_complete through for the toast notification
      if (msg.type !== 'cron_complete') return;
    }

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
        hideStopButton();
        break;

      case 'agent_status':
        Chat.updateAgentStatus(msg);
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

      case 'image_generated':
        Chat.addImageCard(msg.prompt, msg.path);
        break;

      case 'memory_saved':
        if (msg.memories && msg.memories.length > 0) {
          const names = msg.memories.map(m => m.name).join(', ');
          console.log(`[memory] Auto-saved: ${names}`);
        }
        break;

      case 'session_renamed':
        if (msg.sessionId === currentSession) {
          currentDisplayName = msg.name;
          updateSessionName();
        }
        loadSessionList();
        break;

      case 'argus_reaction':
        if (window.Argus) window.Argus.addReaction(msg);
        break;

      case 'cron_complete':
        // A cron job finished running server-side — show toast with link to its session
        showCronToast(msg.description, msg.sessionId, msg.preview);
        loadSessionList(); // refresh sidebar to show new cron session
        // Refresh cron panel if it's currently open so stats are up to date
        if (!document.getElementById('right-panel').classList.contains('hidden')
            && document.getElementById('panel-title').textContent === 'Scheduled Jobs') {
          openCronPanel();
        }
        break;

      // ── Delegate (frontier AI) events ──
      case 'delegate_start':
        Chat.startDelegateStream(msg.target);
        break;

      case 'delegate_token':
        Chat.appendDelegateToken(msg.content);
        break;

      case 'delegate_done':
        Chat.endDelegateStream(msg.target, msg.duration);
        hideStopButton();
        break;

      case 'delegate_error': {
        Chat.endDelegateStream(msg.target);
        const name = { claude: 'Claude', codex: 'Codex', gemini: 'Gemini' }[msg.target] || msg.target;
        Chat.addError(`${name}: ${msg.message}`);
        hideStopButton();
        break;
      }

      // ── Deep Research events ──
      case 'research_status':
        handleResearchStatus(msg);
        break;

      case 'error':
        Chat.endStream();
        Chat.addError(msg.message);
        hideStopButton();
        break;

      case 'session_history':
        Chat.loadHistory(msg.messages);
        setCurrentSession(msg.sessionId || currentSession);
        updateSessionName();
        break;
    }
  });

  WS.on('open', async () => {
    // Load delegate availability
    try {
      const dRes = await fetch('/api/delegates');
      delegateAvailability = await dRes.json();
      initDelegateSelector();
      initResearchOptions();
    } catch {}

    // Load session list first so we can pick the most recent if needed
    await loadSessionList();

    // Check URL params — ?session=ID deep-links to a specific session (used by cron notifications)
    const urlParams = new URLSearchParams(window.location.search);
    const urlSession = urlParams.get('session');
    if (urlSession) {
      setCurrentSession(urlSession);
      // Clean URL without reloading
      window.history.replaceState({}, '', window.location.pathname);
    }

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
  const fileInput = document.getElementById('file-input');
  const attachBtn = document.getElementById('attach-btn');
  const attachPreview = document.getElementById('attachment-preview');

  // Pending attachments waiting to be sent
  let pendingAttachments = [];

  const IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/svg+xml']);
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

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

  // Attach button opens file picker
  attachBtn.addEventListener('click', () => fileInput.click());

  // Handle file selection
  fileInput.addEventListener('change', () => {
    for (const file of fileInput.files) {
      if (file.size > MAX_FILE_SIZE) {
        Chat.addError(`File "${file.name}" is too large (max 10MB)`);
        continue;
      }
      readAttachment(file);
    }
    fileInput.value = ''; // reset so same file can be re-selected
  });

  // Drag-and-drop on the input area
  const inputArea = document.querySelector('.input-area');
  inputArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    inputArea.classList.add('drag-over');
  });
  inputArea.addEventListener('dragleave', () => {
    inputArea.classList.remove('drag-over');
  });
  inputArea.addEventListener('drop', (e) => {
    e.preventDefault();
    inputArea.classList.remove('drag-over');
    for (const file of e.dataTransfer.files) {
      if (file.size > MAX_FILE_SIZE) {
        Chat.addError(`File "${file.name}" is too large (max 10MB)`);
        continue;
      }
      readAttachment(file);
    }
  });

  // Paste images from clipboard
  input.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.kind === 'file') {
        e.preventDefault();
        const file = item.getAsFile();
        if (file) readAttachment(file);
      }
    }
  });

  function readAttachment(file) {
    const isImage = IMAGE_TYPES.has(file.type);
    const reader = new FileReader();
    reader.onload = () => {
      const attachment = {
        name: file.name,
        type: file.type,
        size: file.size,
        isImage,
      };
      if (isImage) {
        // Store as base64 data URL for preview and base64 raw for Ollama
        attachment.dataUrl = reader.result;
        attachment.base64 = reader.result.split(',')[1]; // strip data:...;base64, prefix
      } else {
        attachment.text = reader.result;
      }
      pendingAttachments.push(attachment);
      renderAttachmentPreview();
    };
    if (isImage) {
      reader.readAsDataURL(file);
    } else {
      reader.readAsText(file);
    }
  }

  function renderAttachmentPreview() {
    if (pendingAttachments.length === 0) {
      attachPreview.classList.add('hidden');
      attachPreview.innerHTML = '';
      return;
    }
    attachPreview.classList.remove('hidden');
    attachPreview.innerHTML = pendingAttachments.map((att, i) => {
      const sizeStr = att.size < 1024 ? `${att.size}B`
        : att.size < 1048576 ? `${(att.size / 1024).toFixed(0)}KB`
        : `${(att.size / 1048576).toFixed(1)}MB`;
      if (att.isImage) {
        return `<div class="attachment-chip image-chip">
          <img src="${att.dataUrl}" alt="${Markdown.escapeHtml(att.name)}">
          <span class="attachment-chip-name">${Markdown.escapeHtml(att.name)}</span>
          <button type="button" class="attachment-remove" onclick="window._removeAttachment(${i})">&times;</button>
        </div>`;
      }
      return `<div class="attachment-chip file-chip">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span class="attachment-chip-name">${Markdown.escapeHtml(att.name)}</span>
        <span class="attachment-chip-size">${sizeStr}</span>
        <button type="button" class="attachment-remove" onclick="window._removeAttachment(${i})">&times;</button>
      </div>`;
    }).join('');
  }

  window._removeAttachment = (index) => {
    pendingAttachments.splice(index, 1);
    renderAttachmentPreview();
  };

  // Reveal a file in Finder (macOS)
  window._revealInFinder = (filePath) => {
    fetch('/api/artifacts/reveal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filePath }),
    }).catch(() => {});
  };

  // Export an artifact to PDF, PNG, or self-contained HTML
  window._exportArtifact = async (btn, artPath, format, title) => {
    const origText = btn.textContent;
    btn.textContent = 'Exporting...';
    btn.disabled = true;
    btn.classList.add('btn-share-loading');

    try {
      const res = await fetch('/api/artifacts/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: artPath, format, title }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Export failed');

      // Trigger download of the exported file
      const a = document.createElement('a');
      a.href = data.path;
      a.download = data.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      btn.textContent = 'Done!';
      setTimeout(() => { btn.textContent = origText; btn.disabled = false; btn.classList.remove('btn-share-loading'); }, 2000);
    } catch (err) {
      btn.textContent = 'Failed';
      setTimeout(() => { btn.textContent = origText; btn.disabled = false; btn.classList.remove('btn-share-loading'); }, 2000);
      console.error('[export]', err);
    }
  };

  // ── Stop button ──
  const sendBtn = document.getElementById('send-btn');
  const stopBtn = document.getElementById('stop-btn');

  function showStopButton() {
    sendBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
  }

  function hideStopButton() {
    stopBtn.classList.add('hidden');
    sendBtn.classList.remove('hidden');
  }

  stopBtn.addEventListener('click', () => {
    WS.send({ type: 'cancel' });
    hideStopButton();
  });

  function sendMessage() {
    const content = input.value.trim();
    if ((!content && pendingAttachments.length === 0) || Chat.isStreaming) return;

    // Build attachment summaries for display
    const attachments = pendingAttachments.map(a => ({
      name: a.name, isImage: a.isImage, dataUrl: a.dataUrl || null,
      size: a.size, textPreview: a.text ? a.text.slice(0, 200) : null,
    }));

    // Add user message to UI (with attachment info)
    Chat.addMessage('user', content || '(attached files)', { attachments });

    // Show thinking indicator immediately
    Chat.showThinkingIndicator();

    // Build WS message
    const wsMsg = { type: 'message', content: content || '' };

    // Collect images (base64) for Ollama multimodal
    const images = pendingAttachments.filter(a => a.isImage).map(a => a.base64);
    if (images.length > 0) wsMsg.images = images;

    // Collect text files — prepend to content
    const textFiles = pendingAttachments.filter(a => !a.isImage && a.text);
    if (textFiles.length > 0) {
      const fileBlocks = textFiles.map(f => {
        const ext = f.name.split('.').pop() || '';
        return `<file name="${f.name}">\n\`\`\`${ext}\n${f.text}\n\`\`\`\n</file>`;
      }).join('\n\n');
      wsMsg.content = fileBlocks + (wsMsg.content ? '\n\n' + wsMsg.content : '');
    }

    // Attach research mode flags if active
    if (researchMode) {
      wsMsg.researchMode = true;
      if (researchFrontier) wsMsg.useFrontier = researchFrontier;
    }

    // Send to server — delegate or local
    if (currentDelegate !== 'local') {
      WS.send({ type: 'delegate', target: currentDelegate, content: wsMsg.content });
    } else {
      WS.send(wsMsg);
    }

    // Show stop button while generating
    showStopButton();

    // Clear input and attachments
    input.value = '';
    input.style.height = 'auto';
    pendingAttachments = [];
    renderAttachmentPreview();
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

    const displayLabel = session.displayName || (session.turnCount === 0 ? 'New Session' : session.channel || 'general');

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
          // Show welcome screen with a fresh session
          showWelcomeScreen();
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

  // ── Session search ──
  let allSessions = [];
  let allProjects = [];
  const searchInput = document.getElementById('session-search');

  searchInput.addEventListener('input', () => {
    renderSessionList(searchInput.value.trim());
  });

  // Escape clears the search
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      searchInput.value = '';
      renderSessionList('');
      searchInput.blur();
    }
  });

  // ── Cron toast notification ──
  // ── Delegate selector UI ──
  const delegateBtn = document.getElementById('delegate-btn');
  const delegateDropdown = document.getElementById('delegate-dropdown');
  const delegateLabel = delegateBtn.querySelector('.delegate-label');

  const DELEGATE_COLORS = { local: null, claude: '#cc785c', codex: '#10a37f', gemini: '#4285f4' };
  const DELEGATE_NAMES = { local: 'PRE', claude: 'Claude', codex: 'Codex', gemini: 'Gemini' };

  function initDelegateSelector() {
    // Hide unavailable delegates, show available ones with version
    for (const [key, info] of Object.entries(delegateAvailability)) {
      const option = delegateDropdown.querySelector(`.delegate-option[data-target="${key}"]`);
      if (!option) continue;
      if (info.available) {
        option.style.display = '';
        const badge = option.querySelector('.delegate-status');
        if (badge) {
          badge.textContent = info.version || 'installed';
          badge.classList.remove('unavailable');
        }
      } else {
        option.style.display = 'none';
      }
    }
    // If no delegates available at all, hide the entire delegate button
    const anyAvailable = Object.values(delegateAvailability).some(d => d.available);
    delegateBtn.style.display = anyAvailable ? '' : 'none';
  }

  delegateBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    delegateDropdown.classList.toggle('hidden');
  });

  // Close dropdown on outside click
  document.addEventListener('click', () => delegateDropdown.classList.add('hidden'));

  // Handle delegate selection
  delegateDropdown.addEventListener('click', (e) => {
    const option = e.target.closest('.delegate-option');
    if (!option) return;
    const target = option.dataset.target;

    // Check availability for non-local delegates
    if (target !== 'local' && delegateAvailability[target] && !delegateAvailability[target].available) {
      const hint = delegateAvailability[target].installHint || '';
      Chat.addError(`${DELEGATE_NAMES[target]} CLI is not installed.${hint ? ' Install with: ' + hint : ''}`);
      return;
    }

    currentDelegate = target;
    delegateDropdown.classList.add('hidden');

    // Update button appearance
    delegateLabel.textContent = DELEGATE_NAMES[target] || 'PRE';
    if (target === 'local') {
      delegateBtn.classList.remove('delegate-active');
      delegateBtn.style.removeProperty('--delegate-color');
      input.placeholder = 'Message PRE...';
    } else {
      delegateBtn.classList.add('delegate-active');
      delegateBtn.style.setProperty('--delegate-color', DELEGATE_COLORS[target]);
      input.placeholder = `Message ${DELEGATE_NAMES[target]}...`;
    }

    // Update active state in dropdown
    delegateDropdown.querySelectorAll('.delegate-option').forEach(o => {
      o.classList.toggle('active', o.dataset.target === target);
    });
  });

  // ── Research mode toggle + dropdown ──
  const researchBtn = document.getElementById('research-btn');
  const researchDropdown = document.getElementById('research-dropdown');

  function updateResearchPlaceholder() {
    if (researchMode) {
      const label = researchFrontier
        ? `Deep Research (${researchFrontier}): ask a research question...`
        : 'Deep Research (local multi-pass): ask a research question...';
      input.placeholder = label;
    } else {
      if (currentDelegate !== 'local') {
        input.placeholder = `Message ${DELEGATE_NAMES[currentDelegate]}...`;
      } else {
        input.placeholder = 'Message PRE...';
      }
    }
  }

  // Click the flask: toggle research on/off
  researchBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (researchMode) {
      // Turn off — hide dropdown too
      researchMode = false;
      researchBtn.classList.remove('research-active');
      researchDropdown.classList.add('hidden');
      updateResearchPlaceholder();
    } else {
      // Turn on — show dropdown for settings
      researchMode = true;
      researchBtn.classList.add('research-active');
      researchDropdown.classList.remove('hidden');
      updateResearchPlaceholder();
    }
  });

  // Right-click the flask: open settings without toggling
  researchBtn.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (researchMode) {
      researchDropdown.classList.toggle('hidden');
    }
  });

  // Close dropdown on outside click
  document.addEventListener('click', (e) => {
    if (!researchDropdown.contains(e.target) && e.target !== researchBtn) {
      researchDropdown.classList.add('hidden');
    }
  });

  // Handle radio selection in dropdown
  researchDropdown.addEventListener('change', (e) => {
    if (e.target.name === 'research-synth') {
      const val = e.target.value;
      researchFrontier = val === 'local' ? false : val;
      updateResearchPlaceholder();
    }
  });

  // On WS connect, hide unavailable frontier options
  function initResearchOptions() {
    const frontierModels = ['claude', 'codex', 'gemini'];
    for (const model of frontierModels) {
      const radio = researchDropdown.querySelector(`input[value="${model}"]`);
      if (!radio) continue;
      const label = radio.closest('.research-option');
      if (!label) continue;
      const available = delegateAvailability[model]?.available;
      if (!available) {
        label.style.display = 'none';
        radio.disabled = true;
      } else {
        label.style.display = '';
        radio.disabled = false;
      }
    }
  }

  // ── Research progress handler ──
  function handleResearchStatus(msg) {
    const messagesContainer = document.getElementById('messages');

    switch (msg.phase) {
      case 'starting':
        // Create the research progress card
        researchProgressEl = document.createElement('div');
        researchProgressEl.className = 'research-progress';
        researchProgressEl.innerHTML = `
          <div class="research-progress-header">
            <div class="research-progress-spinner"></div>
            <span>Deep Research</span>
          </div>
          <div class="research-progress-bar"><div class="research-progress-fill" style="width:5%"></div></div>
          <div class="research-progress-status">${msg.message}</div>
          <ul class="research-progress-sections"></ul>
        `;
        messagesContainer.appendChild(researchProgressEl);
        Chat.scrollToBottom();
        break;

      case 'outline_done': {
        if (!researchProgressEl) break;
        const fill = researchProgressEl.querySelector('.research-progress-fill');
        if (fill) fill.style.width = '15%';
        const status = researchProgressEl.querySelector('.research-progress-status');
        if (status) status.textContent = msg.message;
        // Populate section list
        const list = researchProgressEl.querySelector('.research-progress-sections');
        if (list && msg.sections) {
          list.innerHTML = msg.sections.map(s =>
            `<li><span class="research-section-check">-</span> ${s}</li>`
          ).join('');
        }
        Chat.scrollToBottom();
        break;
      }

      case 'gather_section': {
        if (!researchProgressEl) break;
        const total = msg.total || 1;
        const pct = 15 + Math.round((msg.current - 1) / total * 60);
        const fill = researchProgressEl.querySelector('.research-progress-fill');
        if (fill) fill.style.width = `${pct}%`;
        const status = researchProgressEl.querySelector('.research-progress-status');
        if (status) status.textContent = msg.message;
        // Mark current section as active
        const items = researchProgressEl.querySelectorAll('.research-progress-sections li');
        items.forEach((li, i) => {
          li.className = i < msg.current - 1 ? 'done' : i === msg.current - 1 ? 'active' : '';
          const check = li.querySelector('.research-section-check');
          if (check) check.textContent = i < msg.current - 1 ? '\u2713' : i === msg.current - 1 ? '\u25B6' : '-';
        });
        Chat.scrollToBottom();
        break;
      }

      case 'gather_section_done': {
        if (!researchProgressEl) break;
        const total = msg.total || 1;
        const pct = 15 + Math.round(msg.current / total * 60);
        const fill = researchProgressEl.querySelector('.research-progress-fill');
        if (fill) fill.style.width = `${pct}%`;
        // Mark section done
        const items = researchProgressEl.querySelectorAll('.research-progress-sections li');
        if (items[msg.current - 1]) {
          items[msg.current - 1].className = 'done';
          const check = items[msg.current - 1].querySelector('.research-section-check');
          if (check) check.textContent = '\u2713';
        }
        break;
      }

      case 'synthesize': {
        if (!researchProgressEl) break;
        const fill = researchProgressEl.querySelector('.research-progress-fill');
        if (fill) fill.style.width = '80%';
        const status = researchProgressEl.querySelector('.research-progress-status');
        if (status) status.textContent = msg.message;
        // Mark all sections done
        researchProgressEl.querySelectorAll('.research-progress-sections li').forEach(li => {
          li.className = 'done';
          const check = li.querySelector('.research-section-check');
          if (check) check.textContent = '\u2713';
        });
        Chat.scrollToBottom();
        break;
      }

      case 'synthesize_progress': {
        if (!researchProgressEl) break;
        const status = researchProgressEl.querySelector('.research-progress-status');
        if (status) status.textContent = msg.message;
        break;
      }

      case 'assemble': {
        if (!researchProgressEl) break;
        const fill = researchProgressEl.querySelector('.research-progress-fill');
        if (fill) fill.style.width = '95%';
        const status = researchProgressEl.querySelector('.research-progress-status');
        if (status) status.textContent = msg.message;
        Chat.scrollToBottom();
        break;
      }

      case 'complete': {
        if (!researchProgressEl) break;
        const fill = researchProgressEl.querySelector('.research-progress-fill');
        if (fill) fill.style.width = '100%';
        const spinner = researchProgressEl.querySelector('.research-progress-spinner');
        if (spinner) spinner.style.display = 'none';
        const status = researchProgressEl.querySelector('.research-progress-status');
        if (status) {
          status.textContent = msg.message;
          status.style.color = '#8b5cf6';
        }
        researchProgressEl = null;
        Chat.scrollToBottom();
        break;
      }

      default: {
        // Generic status update
        if (researchProgressEl) {
          const status = researchProgressEl.querySelector('.research-progress-status');
          if (status) status.textContent = msg.message || '';
        }
      }
    }
  }

  function showCronToast(description, sessionId, preview) {
    const toast = document.createElement('div');
    toast.className = 'cron-toast';
    const shortPreview = preview && preview.length > 150 ? preview.slice(0, 150) + '...' : (preview || '');
    toast.innerHTML = `
      <div class="cron-toast-header">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 3.5a.5.5 0 0 0-1 0V8a.5.5 0 0 0 .252.434l3.5 2a.5.5 0 0 0 .496-.868L8 7.71V3.5z"/><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm7-8A7 7 0 1 1 1 8a7 7 0 0 1 14 0z"/></svg>
        <strong>Cron: ${description}</strong>
        <button class="cron-toast-close" onclick="this.closest('.cron-toast').remove()">&times;</button>
      </div>
      <div class="cron-toast-preview">${shortPreview.replace(/</g, '&lt;')}</div>
      <button class="btn btn-sm btn-primary cron-toast-open" data-session="${sessionId}">View Result</button>
    `;
    document.body.appendChild(toast);
    // Click to switch to the cron session
    toast.querySelector('.cron-toast-open').addEventListener('click', async () => {
      setCurrentSession(sessionId);
      updateSessionName();
      loadSessionList();
      toast.remove();
      // Load via REST (reliable even when WS is disconnected)
      try {
        const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
        const messages = await res.json();
        Chat.loadHistory(messages);
      } catch {}
      WS.send({ type: 'switch_session', sessionId });
    });
    // Auto-dismiss after 30 seconds
    setTimeout(() => { if (toast.parentNode) toast.remove(); }, 30000);
  }

  async function loadSessionList() {
    try {
      const [sessionsRes, projectsRes] = await Promise.all([
        fetch('/api/sessions'),
        fetch('/api/projects'),
      ]);
      allSessions = await sessionsRes.json();
      allProjects = await projectsRes.json();
      renderSessionList(searchInput.value.trim());
    } catch {}
  }

  function renderSessionList(searchQuery) {
    const query = (searchQuery || '').toLowerCase();

    // Filter out agent sub-sessions from the sidebar (still accessible via search or Agent Feed)
    let sessions = allSessions.filter(s => {
      const name = (s.displayName || s.channel || '').toLowerCase();
      const id = (s.id || '').toLowerCase();
      // Hide sessions that look like agent sub-sessions
      if (name.startsWith('agent-') || id.includes(':agent-') || id.includes(':agent_')) return false;
      return true;
    });

    // Apply search query (search restores hidden agent sessions so they're still findable)
    if (query) {
      sessions = allSessions.filter(s => {
        const name = (s.displayName || s.channel || '').toLowerCase();
        const preview = (s.preview || '').toLowerCase();
        const id = (s.id || '').toLowerCase();
        return name.includes(query) || preview.includes(query) || id.includes(query);
      });
    }

    const projects = allProjects;
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

        // During search, skip empty projects and force-expand those with matches
        if (query && projectSessions.length === 0) continue;
        if (query && projectSessions.length > 0) {
          group.classList.add('expanded');
        }

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
        label.textContent = query ? 'Matches' : 'Recent';
        list.appendChild(label);
      }
      const maxShown = query ? 50 : 20;
      for (const session of ungrouped.slice(0, maxShown)) {
        list.appendChild(buildSessionItem(session));
      }

      // No results message
      if (query && sessions.length === 0) {
        const noResults = document.createElement('div');
        noResults.style.cssText = 'color:var(--text-muted);font-size:0.82rem;padding:16px 0;text-align:center';
        noResults.textContent = `No sessions matching "${searchQuery}"`;
        list.appendChild(noResults);
      }
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
        el.textContent = 'New Session';
      }
    }
  }

  // Show the welcome screen by creating a fresh empty session
  async function showWelcomeScreen() {
    try {
      const res = await fetch('/api/sessions/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project: 'web', channel: 'general' }),
      });
      const data = await res.json();
      setCurrentSession(data.id);
      currentDisplayName = null;
      updateSessionName();
      WS.send({ type: 'switch_session', sessionId: currentSession });
      loadSessionList();
    } catch (err) {
      console.error('Failed to show welcome screen:', err);
    }
  }

  // Logo click → welcome screen
  document.querySelector('.logo').addEventListener('click', () => {
    showWelcomeScreen();
  });

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

  // ── Settings panel ──
  const SERVICE_ICONS = {
    brave_search: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L3 7v10l9 5 9-5V7L12 2zm0 2.18l6.66 3.7L12 11.56 5.34 7.88 12 4.18z"/></svg>',
    github: '<svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>',
    google: '<svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>',
    wolfram: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L2 7v10l10 5 10-5V7L12 2z"/></svg>',
    telegram: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.479.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/></svg>',
    jira: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M11.571 11.513H0a5.218 5.218 0 0 0 5.232 5.215h2.13v2.057A5.215 5.215 0 0 0 12.575 24V12.518a1.005 1.005 0 0 0-1.005-1.005zm5.723-5.756H5.736a5.215 5.215 0 0 0 5.215 5.214h2.129v2.058a5.218 5.218 0 0 0 5.215 5.214V6.758a1.001 1.001 0 0 0-1.001-1.001zM23.013 0H11.455a5.215 5.215 0 0 0 5.215 5.215h2.129v2.057A5.215 5.215 0 0 0 24.013 12.5V1.005A1.005 1.005 0 0 0 23.013 0z"/></svg>',
    confluence: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M.87 18.257c-.248.382-.53.875-.763 1.245a.764.764 0 0 0 .255 1.04l4.965 3.054a.764.764 0 0 0 1.058-.26c.199-.332.494-.833.812-1.39 1.398-2.45 2.818-2.16 5.348-.875l4.947 2.51a.764.764 0 0 0 1.013-.382l2.227-5.093a.764.764 0 0 0-.357-1.006c-1.24-.63-3.45-1.76-4.986-2.537-5.14-2.602-9.01-2.269-14.519 3.694zm22.26-12.514c.249-.383.531-.876.764-1.246a.764.764 0 0 0-.256-1.04L18.674.403a.764.764 0 0 0-1.058.26c-.2.332-.494.833-.813 1.39-1.397 2.45-2.817 2.16-5.347.875L6.509.418A.764.764 0 0 0 5.496.8L3.27 5.893a.764.764 0 0 0 .357 1.006c1.24.63 3.45 1.76 4.985 2.537 5.14 2.602 9.01 2.27 14.52-3.693z"/></svg>',
    smartsheet: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M22.157 0H1.843A1.843 1.843 0 0 0 0 1.843v20.314A1.843 1.843 0 0 0 1.843 24h20.314A1.843 1.843 0 0 0 24 22.157V1.843A1.843 1.843 0 0 0 22.157 0zM5.5 7h13a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-13a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5zm0 4h13a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-13a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5zm0 4h13a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-13a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5z"/></svg>',
    slack: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zm1.271 0a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zm0 1.271a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312zM18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zm-1.27 0a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.163 0a2.528 2.528 0 0 1 2.523 2.522v6.312zM15.163 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.163 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zm0-1.27a2.527 2.527 0 0 1-2.52-2.523 2.527 2.527 0 0 1 2.52-2.52h6.315A2.528 2.528 0 0 1 24 15.163a2.528 2.528 0 0 1-2.522 2.523h-6.315z"/></svg>',
    microsoft: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M0 0h11.377v11.372H0zm12.623 0H24v11.372H12.623zM0 12.623h11.377V24H0zm12.623 0H24V24H12.623z"/></svg>',
    linear: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.652 11.422a.727.727 0 0 1 0-1.03L11.422 1.62a.727.727 0 0 1 .515-.213c.193 0 .378.077.515.213l8.926 8.926a.727.727 0 0 1 0 1.03l-8.926 8.926a.727.727 0 0 1-1.03 0l-8.77-8.08z"/></svg>',
    zoom: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M4.585 6.836A3.252 3.252 0 0 0 1.34 10.09v3.82a3.252 3.252 0 0 0 3.245 3.254h8.348v-3.82H6.53a1.95 1.95 0 0 1-1.945-1.954V8.79h11.608a1.95 1.95 0 0 1 1.945 1.6l3.358-2.26A1.3 1.3 0 0 0 22.66 7V5.75a.92.92 0 0 0-.917-.914H4.585v2zm17.07 3.234l-3.357 2.26v4.834a.92.92 0 0 1-.917.914h-4.453v2h5.37a3.25 3.25 0 0 0 3.245-3.254v-3.82a3.25 3.25 0 0 0-.888-2.934z"/></svg>',
    figma: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M8 24c2.208 0 4-1.792 4-4v-4H8c-2.208 0-4 1.792-4 4s1.792 4 4 4zm0-20c-2.208 0-4 1.792-4 4s1.792 4 4 4h4V4c0-2.208-1.792-4-4-4zm0 8c-2.208 0-4 1.792-4 4s1.792 4 4 4h4v-8H8zm8-8h-4v8h4c2.208 0 4-1.792 4-4s-1.792-4-4-4zm0 12a4 4 0 1 0 0 8 4 4 0 0 0 0-8z"/></svg>',
    asana: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="6" r="5.5"/><circle cx="5.5" cy="17.5" r="5.5"/><circle cx="18.5" cy="17.5" r="5.5"/></svg>',
    dynamics365: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M4 4l8-4v10.6L4 14V4zm0 12l8-3.4V24l-8-4V16zm10-5.4L22 8v8l-8 4V10.6z"/></svg>',
  };

  document.getElementById('settings-btn').addEventListener('click', () => {
    openSettingsPanel();
  });

  // ── Cron jobs panel ──
  document.getElementById('cron-btn').addEventListener('click', () => {
    openCronPanel();
  });

  async function openCronPanel() {
    const panel = document.getElementById('right-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');
    title.textContent = 'Scheduled Jobs';
    panel.classList.remove('hidden');
    content.innerHTML = '<div class="spinner" style="margin:20px auto"></div>';

    try {
      const res = await fetch('/api/cron');
      const jobs = await res.json();
      renderCronPanel(content, jobs);
    } catch {
      content.innerHTML = '<p style="color:var(--danger)">Failed to load cron jobs</p>';
    }
  }

  function renderCronPanel(container, jobs) {
    let html = '<div class="settings-section">';
    html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">';
    html += '<div class="settings-section-title" style="margin:0">Jobs</div>';
    html += '<button class="btn btn-primary btn-sm" onclick="window.Cron.showAdd()">+ New Job</button>';
    html += '</div>';

    if (jobs.length === 0) {
      html += '<p style="color:var(--text-muted);font-size:0.85rem">No scheduled jobs yet. Create one to run prompts on a recurring schedule.</p>';
    } else {
      for (const job of jobs) {
        const statusDot = job.enabled
          ? '<span style="color:var(--success)">&#9679;</span>'
          : '<span style="color:var(--text-muted)">&#9679;</span>';
        const lastRun = job.last_run_at
          ? new Date(job.last_run_at).toLocaleString()
          : 'Never';
        const nextRun = job.enabled ? nextCronRun(job.schedule) : null;
        const nextRunStr = nextRun
          ? nextRun.toLocaleString(undefined, { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
          : 'N/A';
        html += `<div class="connection-card" style="margin-bottom:10px">
          <div class="connection-card-header" style="padding:12px 14px">
            <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:0">
              ${statusDot}
              <div style="flex:1;min-width:0">
                <div style="font-weight:600;font-size:0.9rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${Markdown.escapeHtml(job.description)}</div>
                <div style="font-size:0.75rem;color:var(--text-muted);margin-top:2px">${Markdown.escapeHtml(describeCron(job.schedule))} &middot; <code style="font-family:'SF Mono',monospace;font-size:0.7rem">${Markdown.escapeHtml(job.schedule)}</code></div>
                <div style="font-size:0.75rem;color:var(--text-muted);margin-top:1px">Next: <strong style="color:var(--text-secondary)">${nextRunStr}</strong> &middot; Runs: ${job.run_count || 0} &middot; Last: ${lastRun}</div>
              </div>
            </div>
            <div style="display:flex;gap:4px;flex-shrink:0">
              ${job.last_session_id ? `<button class="btn btn-ghost btn-sm" onclick="window.Cron.viewResult('${job.last_session_id}')" title="View last result">Result</button>` : ''}
              <button class="btn btn-ghost btn-sm" onclick="window.Cron.toggle('${job.id}', ${!job.enabled})" title="${job.enabled ? 'Disable' : 'Enable'}">${job.enabled ? 'Disable' : 'Enable'}</button>
              <button class="btn btn-ghost btn-sm" onclick="window.Cron.run('${job.id}')" title="Run now">Run</button>
              <button class="btn btn-ghost btn-sm" onclick="window.Cron.del('${job.id}')" style="color:var(--danger)" title="Delete">&times;</button>
            </div>
          </div>
          <div style="padding:8px 14px;font-size:0.8rem;color:var(--text-secondary);border-top:1px solid var(--border)">${Markdown.escapeHtml(job.prompt.length > 200 ? job.prompt.slice(0, 200) + '...' : job.prompt)}</div>
        </div>`;
      }
    }

    // Add job form (hidden by default)
    html += `<div id="cron-add-form" style="display:none;margin-top:16px">
      <div class="settings-section-title">New Scheduled Job</div>
      <div style="display:flex;flex-direction:column;gap:10px">
        <div>
          <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">When should it run?</label>
          <input id="cron-schedule" type="text" placeholder="every weekday at 9am" oninput="window.Cron.parseSchedule()" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem">
          <div id="cron-parse-preview" style="font-size:0.75rem;margin-top:4px;min-height:1.2em"></div>
          <div style="font-size:0.7rem;color:var(--text-muted);margin-top:2px">Examples: "every day at 9am" &middot; "weekdays at 8:30am" &middot; "every 15 minutes" &middot; "monday and friday at 3pm" &middot; or raw cron: "0 9 * * 1-5"</div>
        </div>
        <div>
          <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">Description</label>
          <input id="cron-description" type="text" placeholder="Morning briefing" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem">
        </div>
        <div>
          <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">Prompt (sent to the model when triggered)</label>
          <textarea id="cron-prompt" rows="3" placeholder="Give me a summary of today's calendar and top 3 priorities" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem;font-family:inherit;resize:vertical"></textarea>
        </div>
        <div style="display:flex;gap:8px">
          <button class="btn btn-primary btn-sm" onclick="window.Cron.save()">Create Job</button>
          <button class="btn btn-ghost btn-sm" onclick="window.Cron.hideAdd()">Cancel</button>
        </div>
      </div>
    </div>`;

    // Notification delivery info
    html += '<div class="settings-section" style="margin-top:16px">';
    html += '<div class="settings-section-title">Result Delivery</div>';
    html += '<p style="font-size:0.82rem;color:var(--text-muted);line-height:1.5;margin:0 0 8px 0">Cron jobs run server-side — no browser needed. Each run creates its own chat session. Results are delivered via:</p>';
    html += '<ul style="font-size:0.82rem;color:var(--text-secondary);line-height:1.8;margin:0;padding-left:20px">';
    html += '<li><strong>macOS notification</strong> — always (click to open result)</li>';
    html += '<li><strong>Telegram</strong> — configure bot token in Settings to enable</li>';
    html += '<li><strong>GUI toast</strong> — shown if browser is open</li>';
    html += '</ul></div>';

    html += '</div>';
    container.innerHTML = html;
  }

  // Natural language → cron expression parser
  function nlToCron(input) {
    if (!input) return null;
    const s = input.toLowerCase().trim();

    // Already a valid cron expression (5 fields of cron-like chars)
    if (/^[\d*\/,\-]+(\s+[\d*\/,\-]+){4}$/.test(s)) {
      return { cron: s, description: describeCron(s) };
    }

    // Parse time from string: "9am", "9:30pm", "14:00", "3 pm", "noon", "midnight"
    // Prefers am/pm-qualified times over bare numbers (e.g. "1st at 10am" → 10am, not 1)
    function parseTime(str) {
      if (/\bnoon\b/.test(str)) return { h: 12, m: 0 };
      if (/\bmidnight\b/.test(str)) return { h: 0, m: 0 };
      // Try am/pm-qualified match first
      let match = str.match(/(\d{1,2})(?::(\d{2}))?\s*(am|pm)/i);
      // Fall back to colon-separated time (e.g. "14:00")
      if (!match) match = str.match(/(\d{1,2}):(\d{2})/);
      if (!match) return null;
      let h = parseInt(match[1]);
      const m = match[2] ? parseInt(match[2]) : 0;
      const ampm = match[3]?.toLowerCase();
      if (ampm === 'pm' && h < 12) h += 12;
      if (ampm === 'am' && h === 12) h = 0;
      if (h > 23 || m > 59) return null;
      return { h, m };
    }

    // Infer time from time-of-day words when no explicit time given
    function inferTimeOfDay(str) {
      if (/\bmorning\b/.test(str)) return { h: 9, m: 0 };
      if (/\bafternoon\b/.test(str)) return { h: 14, m: 0 };
      if (/\bevening\b/.test(str)) return { h: 18, m: 0 };
      if (/\bnight\b/.test(str)) return { h: 21, m: 0 };
      if (/\bmidday\b/.test(str)) return { h: 12, m: 0 };
      return null;
    }

    // Day name → cron day number
    const dayMap = { sun: 0, sunday: 0, mon: 1, monday: 1, tue: 2, tuesday: 2, wed: 3, wednesday: 3, thu: 4, thursday: 4, fri: 5, friday: 5, sat: 6, saturday: 6 };

    // Parse day names from string (word-boundary match to avoid "mon" in "monthly"; handles plurals like "fridays")
    function parseDays(str) {
      const found = [];
      for (const [name, num] of Object.entries(dayMap)) {
        if (new RegExp('\\b' + name + 's?\\b').test(str)) found.push(num);
      }
      return [...new Set(found)].sort();
    }

    // Resolve time: explicit > time-of-day word > default 9am
    const time = parseTime(s) || inferTimeOfDay(s);
    const min = time ? time.m : 0;
    const hour = time ? time.h : 9;

    // "every N minutes" / "every N mins"
    let match = s.match(/every\s+(\d+)\s*min/);
    if (match) return { cron: `*/${match[1]} * * * *`, description: `Every ${match[1]} minutes` };

    // "every minute"
    if (/every\s+minute/.test(s)) return { cron: '* * * * *', description: 'Every minute' };

    // "every half hour" / "every 30 minutes" (covered above but also handle phrasing)
    if (/every\s+half\s*hour|half[\s-]hourly/.test(s)) {
      return { cron: '*/30 * * * *', description: 'Every 30 minutes' };
    }

    // "every N hours"
    match = s.match(/every\s+(\d+)\s*hour/);
    if (match) return { cron: `0 */${match[1]} * * *`, description: `Every ${match[1]} hours` };

    // "every hour" / "hourly"
    if (/\bevery\s+hour\b|\bhourly\b/.test(s)) {
      return { cron: `${min} * * * *`, description: `Every hour${min ? ` at :${String(min).padStart(2,'0')}` : ''}` };
    }

    // "weekdays at TIME" / "every weekday" / "monday through friday"
    if (/weekday|week\s*day|mon(day)?\s*(through|thru|-|to)\s*fri(day)?/.test(s)) {
      return { cron: `${min} ${hour} * * 1-5`, description: `Weekdays at ${fmtTime(hour, min)}` };
    }

    // "weekends at TIME"
    if (/weekend/.test(s)) {
      return { cron: `${min} ${hour} * * 0,6`, description: `Weekends at ${fmtTime(hour, min)}` };
    }

    // "twice a day" / "2x daily" — must come before the "daily" catch-all
    if (/twice\s+a\s+day|2x?\s+daily|two\s+times?\s+a\s+day/.test(s)) {
      return { cron: `0 9,17 * * *`, description: 'Twice daily at 9:00 AM and 5:00 PM' };
    }

    // "three times a day" / "3x daily"
    if (/three\s+times?\s+a\s+day|3x?\s+daily/.test(s)) {
      return { cron: `0 8,13,18 * * *`, description: 'Three times daily at 8 AM, 1 PM, 6 PM' };
    }

    // "every day" / "daily" / "every morning" / "every afternoon" / "every evening" / "every night"
    if (/every\s*day|daily|every\s+morning|every\s+afternoon|every\s+evening|every\s+night|every\s+midday|each\s+(day|morning|afternoon|evening|night)/.test(s)) {
      return { cron: `${min} ${hour} * * *`, description: `Daily at ${fmtTime(hour, min)}` };
    }

    // Specific days: "monday and wednesday at 3pm", "every tuesday at 10am", "on fridays at noon"
    const days = parseDays(s);
    if (days.length > 0) {
      const dow = days.join(',');
      const dayNames = days.map(d => Object.keys(dayMap).find(k => dayMap[k] === d && k.length > 3) || d);
      const label = dayNames.map(d => typeof d === 'string' ? d.charAt(0).toUpperCase() + d.slice(1) : d).join(', ');
      return { cron: `${min} ${hour} * * ${dow}`, description: `${label} at ${fmtTime(hour, min)}` };
    }

    // "every month on the Nth" / "monthly on the Nth" / "on the 1st of every month"
    match = s.match(/(?:every\s+month|monthly|of\s+every\s+month)\s*(?:on\s+)?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?/)
         || s.match(/(?:on\s+)?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?:every|each)\s+month/);
    if (match) {
      const dom = parseInt(match[1]);
      return { cron: `${min} ${hour} ${dom} * *`, description: `Monthly on the ${dom}${ordSuffix(dom)} at ${fmtTime(hour, min)}` };
    }

    // "every week" / "weekly"
    if (/every\s*week\b|weekly/.test(s)) {
      return { cron: `${min} ${hour} * * 1`, description: `Weekly on Monday at ${fmtTime(hour, min)}` };
    }

    // "at TIME" with no other context → daily
    if (/^(?:at\s+)/.test(s) && time) {
      return { cron: `${min} ${hour} * * *`, description: `Daily at ${fmtTime(hour, min)}` };
    }

    // Just a bare time like "9am" or "3:30pm" → daily
    if (time && s.replace(/\d{1,2}(:\d{2})?\s*(am|pm)?/i, '').trim().length < 5) {
      return { cron: `${min} ${hour} * * *`, description: `Daily at ${fmtTime(hour, min)}` };
    }

    // Last resort: if we have any time-of-day word, treat as daily
    if (inferTimeOfDay(s)) {
      return { cron: `${min} ${hour} * * *`, description: `Daily at ${fmtTime(hour, min)}` };
    }

    return null;
  }

  function fmtTime(h, m) {
    const ampm = h >= 12 ? 'PM' : 'AM';
    const h12 = h === 0 ? 12 : h > 12 ? h - 12 : h;
    return `${h12}:${String(m).padStart(2, '0')} ${ampm}`;
  }

  function ordSuffix(n) {
    if (n > 3 && n < 21) return 'th';
    switch (n % 10) { case 1: return 'st'; case 2: return 'nd'; case 3: return 'rd'; default: return 'th'; }
  }

  const DOW_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const DOW_FULL = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  function describeCron(expr) {
    const [min, hour, dom, mon, dow] = expr.split(/\s+/);
    if (min.startsWith('*/')) return `Every ${min.slice(2)} minutes`;
    if (hour.startsWith('*/')) return `Every ${hour.slice(2)} hours` + (min !== '0' ? ` at :${min.padStart(2, '0')}` : '');
    const h = parseInt(hour), m = parseInt(min);
    const timeStr = !isNaN(h) ? fmtTime(h, isNaN(m) ? 0 : m) : `${hour}:${min}`;
    if (dow === '1-5') return `Weekdays at ${timeStr}`;
    if (dow === '0,6') return `Weekends at ${timeStr}`;
    if (dow !== '*') {
      const dayStr = describeDow(dow);
      return `${dayStr} at ${timeStr}`;
    }
    if (dom !== '*') return `Day ${dom} of month at ${timeStr}`;
    return `Daily at ${timeStr}`;
  }

  function describeDow(dow) {
    // Expand ranges and lists: "1,3,5" → "Mon, Wed, Fri"; "1-3" → "Mon–Wed"; "3" → "Every Wednesday"
    const parts = dow.split(',');
    const names = [];
    for (const part of parts) {
      if (part.includes('-')) {
        const [a, b] = part.split('-').map(Number);
        if (!isNaN(a) && !isNaN(b) && DOW_NAMES[a] && DOW_NAMES[b]) {
          names.push(`${DOW_NAMES[a]}–${DOW_NAMES[b]}`);
        } else {
          names.push(part);
        }
      } else {
        const n = parseInt(part);
        if (!isNaN(n) && DOW_FULL[n]) {
          names.push(parts.length === 1 ? `Every ${DOW_FULL[n]}` : DOW_NAMES[n]);
        } else {
          names.push(part);
        }
      }
    }
    return names.join(', ');
  }

  /**
   * Compute the next fire time for a 5-field cron expression.
   * Returns a Date or null if unparseable.
   */
  function nextCronRun(expr) {
    const [minF, hourF, domF, monF, dowF] = expr.split(/\s+/);
    const now = new Date();
    // Brute-force: scan forward up to 400 days in 1-minute increments
    // (optimised: jump by hour/day when possible)
    const candidate = new Date(now.getFullYear(), now.getMonth(), now.getDate(), now.getHours(), now.getMinutes() + 1, 0, 0);

    function matchesField(val, field, rangeMax) {
      if (field === '*') return true;
      for (const part of field.split(',')) {
        if (part.includes('/')) {
          const [base, step] = part.split('/');
          const s = parseInt(step);
          const b = base === '*' ? 0 : parseInt(base);
          if (!isNaN(s) && (val - b) % s === 0 && val >= b) return true;
        } else if (part.includes('-')) {
          const [a, b] = part.split('-').map(Number);
          if (val >= a && val <= b) return true;
        } else {
          if (parseInt(part) === val) return true;
        }
      }
      return false;
    }

    for (let i = 0; i < 525600; i++) { // up to 365 days of minutes
      const d = new Date(candidate.getTime() + i * 60000);
      if (!matchesField(d.getMonth() + 1, monF)) continue;
      // DOM and DOW: if both are restricted (not *), either can match (OR logic per cron spec)
      const domMatch = matchesField(d.getDate(), domF);
      const dowMatch = matchesField(d.getDay(), dowF);
      if (domF !== '*' && dowF !== '*') {
        if (!domMatch && !dowMatch) continue;
      } else {
        if (!domMatch || !dowMatch) continue;
      }
      if (!matchesField(d.getHours(), hourF)) {
        // Skip ahead to next hour
        i += 59 - d.getMinutes();
        continue;
      }
      if (!matchesField(d.getMinutes(), minF)) continue;
      return d;
    }
    return null;
  }

  window.Cron = {
    showAdd() {
      const form = document.getElementById('cron-add-form');
      if (form) form.style.display = 'block';
    },
    hideAdd() {
      const form = document.getElementById('cron-add-form');
      if (form) form.style.display = 'none';
    },
    parseSchedule() {
      const input = document.getElementById('cron-schedule')?.value || '';
      const preview = document.getElementById('cron-parse-preview');
      if (!preview) return;
      if (!input.trim()) { preview.innerHTML = ''; return; }
      const result = nlToCron(input);
      if (result) {
        preview.innerHTML = `<span style="color:var(--success)">&#10003;</span> <code style="font-family:'SF Mono',monospace">${Markdown.escapeHtml(result.cron)}</code> &mdash; ${Markdown.escapeHtml(result.description)}`;
      } else {
        preview.innerHTML = `<span style="color:var(--warning)">Could not parse. Try "every day at 9am" or "weekdays at 8:30am"</span>`;
      }
    },
    async save() {
      const rawSchedule = document.getElementById('cron-schedule')?.value.trim();
      const prompt = document.getElementById('cron-prompt')?.value.trim();
      const description = document.getElementById('cron-description')?.value.trim();
      if (!rawSchedule || !prompt) return alert('Schedule and prompt are required.');
      // Parse natural language to cron
      const parsed = nlToCron(rawSchedule);
      if (!parsed) return alert('Could not understand the schedule. Try something like "every day at 9am" or "weekdays at 8:30am".');
      const schedule = parsed.cron;
      try {
        const res = await fetch('/api/cron', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ schedule, prompt, description: description || parsed.description }),
        });
        if (!res.ok) {
          const err = await res.json();
          return alert(err.error || 'Failed to create job');
        }
        openCronPanel();
      } catch (e) { alert('Error: ' + e.message); }
    },
    async toggle(id, enabled) {
      await fetch(`/api/cron/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      openCronPanel();
    },
    async run(id) {
      await fetch(`/api/cron/${id}/run`, { method: 'POST' });
    },
    async del(id) {
      if (!confirm('Delete this scheduled job?')) return;
      await fetch(`/api/cron/${id}`, { method: 'DELETE' });
      openCronPanel();
    },
    async viewResult(sessionId) {
      // Close the cron panel and switch to the cron result session
      document.getElementById('right-panel').classList.add('hidden');
      setCurrentSession(sessionId);
      updateSessionName();
      loadSessionList();
      // Load session history via REST (reliable even when WS is disconnected)
      try {
        const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
        const messages = await res.json();
        Chat.loadHistory(messages);
      } catch {}
      // Notify server of the active session for future messages
      WS.send({ type: 'switch_session', sessionId });
    },
  };

  // ── Memory browser panel ──
  document.getElementById('memory-btn').addEventListener('click', () => {
    openMemoryPanel();
  });

  async function openMemoryPanel() {
    const panel = document.getElementById('right-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');
    title.textContent = 'Memory';
    panel.classList.remove('hidden');
    content.innerHTML = '<div class="spinner" style="margin:20px auto"></div>';

    try {
      const res = await fetch('/api/memory');
      const memories = await res.json();
      renderMemoryPanel(content, memories);
    } catch {
      content.innerHTML = '<p style="color:var(--danger)">Failed to load memories</p>';
    }
  }

  function renderMemoryPanel(container, memories) {
    const typeColors = {
      user: 'var(--primary)',
      feedback: '#f59e0b',
      project: '#10b981',
      reference: '#8b5cf6',
    };

    let html = '<div class="settings-section">';
    html += '<div class="settings-section-title" style="display:flex;justify-content:space-between;align-items:center">';
    html += `<span>Memories (${memories.length})</span>`;
    html += '<button class="btn btn-ghost btn-sm" onclick="Memory.create()">+ New</button>';
    html += '</div>';

    if (memories.length === 0) {
      html += '<p style="color:var(--text-muted);font-size:0.85rem;padding:12px 0">No memories saved yet. Memories are automatically extracted from conversations, or you can create them manually.</p>';
    }

    // Group by type
    const groups = { feedback: [], user: [], project: [], reference: [] };
    for (const m of memories) {
      (groups[m.type] || groups.project).push(m);
    }

    for (const [type, mems] of Object.entries(groups)) {
      if (mems.length === 0) continue;
      const color = typeColors[type] || 'var(--text-muted)';
      html += `<div style="margin-top:12px">`;
      html += `<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;color:${color};margin-bottom:6px">${type} (${mems.length})</div>`;

      for (const m of mems) {
        const age = m.age ? `<span style="font-size:0.7rem;color:var(--text-muted)"> ${escapeHtml(m.age)}</span>` : '';
        html += `<div class="connection-card" style="margin-bottom:8px;cursor:pointer" onclick="Memory.view('${escapeHtml(m.filename)}')">`;
        html += `<div style="padding:8px 12px">`;
        html += `<div style="font-size:0.85rem;font-weight:500">${escapeHtml(m.name)}${age}</div>`;
        html += `<div style="font-size:0.78rem;color:var(--text-muted);margin-top:2px">${escapeHtml(m.description)}</div>`;
        html += `<div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px">${escapeHtml(m.modified)} &middot; ${escapeHtml(m.scope)}</div>`;
        html += `</div></div>`;
      }
      html += '</div>';
    }

    html += '</div>';
    container.innerHTML = html;
  }

  window.Memory = {
    async view(filename) {
      const panel = document.getElementById('panel-content');
      panel.innerHTML = '<div class="spinner" style="margin:20px auto"></div>';
      try {
        const res = await fetch(`/api/memory/${encodeURIComponent(filename)}`);
        const m = await res.json();
        const typeColors = { user: 'var(--primary)', feedback: '#f59e0b', project: '#10b981', reference: '#8b5cf6' };
        const color = typeColors[m.type] || 'var(--text-muted)';
        let html = '<div class="settings-section">';
        html += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">`;
        html += `<button class="btn btn-ghost btn-sm" onclick="Memory.back()">&larr; Back</button>`;
        html += `<button class="btn btn-ghost btn-sm" style="color:var(--danger)" onclick="Memory.del('${escapeHtml(filename)}')">Delete</button>`;
        html += `</div>`;
        html += `<h3 style="margin:0 0 8px 0;font-size:1.05rem">${escapeHtml(m.name)}</h3>`;
        html += `<div style="display:flex;gap:8px;margin-bottom:12px;font-size:0.78rem">`;
        html += `<span style="color:${color};text-transform:uppercase;font-weight:600">${m.type}</span>`;
        html += `<span style="color:var(--text-muted)">${escapeHtml(m.scope)}</span>`;
        html += `<span style="color:var(--text-muted)">${escapeHtml(m.modified)}</span>`;
        if (m.age) html += `<span style="color:var(--text-muted)">${escapeHtml(m.age)}</span>`;
        html += `</div>`;
        html += `<div style="font-size:0.82rem;color:var(--text-muted);margin-bottom:12px">${escapeHtml(m.description)}</div>`;
        html += `<div style="font-size:0.85rem;white-space:pre-wrap;line-height:1.5;padding:12px;background:var(--surface-2, var(--surface));border-radius:6px;border:1px solid var(--border)">${escapeHtml(m.body)}</div>`;
        html += '</div>';
        panel.innerHTML = html;
      } catch {
        panel.innerHTML = '<p style="color:var(--danger)">Failed to load memory</p>';
      }
    },

    async del(filename) {
      if (!confirm('Delete this memory?')) return;
      try {
        await fetch(`/api/memory/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        openMemoryPanel();
      } catch {}
    },

    create() {
      const panel = document.getElementById('panel-content');
      let html = '<div class="settings-section">';
      html += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">`;
      html += `<button class="btn btn-ghost btn-sm" onclick="Memory.back()">&larr; Back</button>`;
      html += `</div>`;
      html += `<div class="connection-input-group">`;
      html += `<input type="text" id="mem-name" placeholder="Memory name" autocomplete="off">`;
      html += `<select id="mem-type" style="padding:6px 8px;border-radius:4px;border:1px solid var(--border);background:var(--surface);color:var(--text)">
        <option value="user">user</option>
        <option value="feedback">feedback</option>
        <option value="project" selected>project</option>
        <option value="reference">reference</option>
      </select>`;
      html += `<input type="text" id="mem-desc" placeholder="One-line description" autocomplete="off">`;
      html += `<textarea id="mem-content" rows="6" placeholder="Memory content..." style="width:100%;padding:8px;border-radius:4px;border:1px solid var(--border);background:var(--surface);color:var(--text);resize:vertical;font-family:inherit;font-size:0.85rem"></textarea>`;
      html += `<div class="connection-input-row">`;
      html += `<button class="btn btn-primary btn-sm" onclick="Memory.save()">Save</button>`;
      html += `<button class="btn btn-ghost btn-sm" onclick="Memory.back()">Cancel</button>`;
      html += `</div></div></div>`;
      panel.innerHTML = html;
      document.getElementById('mem-name').focus();
    },

    async save() {
      const name = document.getElementById('mem-name')?.value.trim();
      const type = document.getElementById('mem-type')?.value;
      const description = document.getElementById('mem-desc')?.value.trim();
      const content = document.getElementById('mem-content')?.value.trim();
      if (!name || !content) return;
      try {
        await fetch('/api/memory', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, type, description, content }),
        });
        openMemoryPanel();
      } catch {}
    },

    back() {
      openMemoryPanel();
    },
  };

  // ── Triggers panel ──
  document.getElementById('triggers-btn').addEventListener('click', () => {
    openTriggersPanel();
  });

  async function openTriggersPanel() {
    const panel = document.getElementById('right-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');
    title.textContent = 'Triggers';
    panel.classList.remove('hidden');
    content.innerHTML = '<div class="spinner" style="margin:20px auto"></div>';

    try {
      const res = await fetch('/api/triggers');
      const triggers = await res.json();
      renderTriggersPanel(content, triggers);
    } catch {
      content.innerHTML = '<p style="color:var(--danger)">Failed to load triggers</p>';
    }
  }

  function renderTriggersPanel(container, triggers) {
    let html = '<div class="settings-section">';
    html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">';
    html += '<div class="settings-section-title" style="margin:0">Triggers</div>';
    html += '<button class="btn btn-primary btn-sm" onclick="window.Triggers.showAdd()">+ New Trigger</button>';
    html += '</div>';

    if (triggers.length === 0) {
      html += '<p style="color:var(--text-muted);font-size:0.85rem">No triggers yet. Create a file watcher or webhook to fire prompts automatically.</p>';
    } else {
      for (const t of triggers) {
        const statusDot = t.enabled
          ? '<span style="color:var(--success)">&#9679;</span>'
          : '<span style="color:var(--text-muted)">&#9679;</span>';
        const typeLabel = t.type === 'file_watch' ? 'File Watch' : 'Webhook';
        const typeBadge = `<span style="font-size:0.7rem;padding:1px 6px;border-radius:4px;background:var(--surface-hover);color:var(--text-secondary)">${typeLabel}</span>`;
        const lastFired = t.last_fired_at ? new Date(t.last_fired_at).toLocaleString() : 'Never';
        const detail = t.type === 'file_watch' && t.config?.path
          ? `<div style="font-size:0.75rem;color:var(--text-muted);margin-top:2px">Path: <code style="font-family:'SF Mono',monospace;font-size:0.7rem">${Markdown.escapeHtml(t.config.path)}</code>${t.config.glob ? ` &middot; Glob: ${Markdown.escapeHtml(t.config.glob)}` : ''}</div>`
          : t.type === 'webhook'
            ? `<div style="font-size:0.75rem;color:var(--text-muted);margin-top:2px">Endpoint: <code style="font-family:'SF Mono',monospace;font-size:0.7rem">/api/triggers/webhook/${Markdown.escapeHtml(t.id)}</code></div>`
            : '';

        html += `<div class="connection-card" style="margin-bottom:10px">
          <div class="connection-card-header" style="padding:12px 14px">
            <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:0">
              ${statusDot}
              <div style="flex:1;min-width:0">
                <div style="font-weight:600;font-size:0.9rem;display:flex;align-items:center;gap:8px">${Markdown.escapeHtml(t.name)} ${typeBadge}</div>
                ${detail}
                <div style="font-size:0.73rem;color:var(--text-muted);margin-top:2px">Fires: ${t.fire_count || 0} &middot; Last: ${lastFired}</div>
              </div>
            </div>
            <div style="display:flex;gap:4px;flex-shrink:0">
              <button class="btn btn-ghost btn-sm" onclick="window.Triggers.toggle('${t.id}', ${!t.enabled})">${t.enabled ? 'Disable' : 'Enable'}</button>
              <button class="btn btn-ghost btn-sm" onclick="window.Triggers.del('${t.id}')" style="color:var(--danger)" title="Delete">&times;</button>
            </div>
          </div>
          <div style="padding:8px 14px;font-size:0.8rem;color:var(--text-secondary);border-top:1px solid var(--border)">${Markdown.escapeHtml(t.prompt.length > 200 ? t.prompt.slice(0, 200) + '...' : t.prompt)}</div>
        </div>`;
      }
    }

    // Add trigger form
    html += `<div id="trigger-add-form" style="display:none;margin-top:16px">
      <div class="settings-section-title">New Trigger</div>
      <div style="display:flex;flex-direction:column;gap:10px">
        <div>
          <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">Type</label>
          <select id="trigger-type" onchange="window.Triggers.typeChanged()" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem">
            <option value="file_watch">File Watcher</option>
            <option value="webhook">Webhook</option>
          </select>
        </div>
        <div>
          <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">Name (optional)</label>
          <input id="trigger-name" type="text" placeholder="My trigger" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem">
        </div>
        <div id="trigger-path-group">
          <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">Watch path</label>
          <input id="trigger-path" type="text" placeholder="/Users/you/Documents" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem">
        </div>
        <div id="trigger-glob-group">
          <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">Glob filter (optional)</label>
          <input id="trigger-glob" type="text" placeholder="*.pdf" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem">
        </div>
        <div id="trigger-secret-group" style="display:none">
          <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">Webhook secret (optional)</label>
          <input id="trigger-secret" type="text" placeholder="shared-secret" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem">
        </div>
        <div>
          <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">Prompt (executed when triggered)</label>
          <textarea id="trigger-prompt" rows="3" placeholder="Summarize the new files in {path}" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem;font-family:inherit;resize:vertical"></textarea>
          <div style="font-size:0.7rem;color:var(--text-muted);margin-top:2px">Variables: <code>{file}</code>, <code>{event}</code>, <code>{path}</code> (file watch) &middot; <code>{payload}</code>, <code>{headers}</code> (webhook)</div>
        </div>
        <div style="display:flex;gap:8px">
          <button class="btn btn-primary btn-sm" onclick="window.Triggers.save()">Create Trigger</button>
          <button class="btn btn-ghost btn-sm" onclick="window.Triggers.hideAdd()">Cancel</button>
        </div>
      </div>
    </div>`;

    html += '</div>';
    container.innerHTML = html;
  }

  window.Triggers = {
    showAdd() {
      const form = document.getElementById('trigger-add-form');
      if (form) form.style.display = 'block';
    },
    hideAdd() {
      const form = document.getElementById('trigger-add-form');
      if (form) form.style.display = 'none';
    },
    typeChanged() {
      const type = document.getElementById('trigger-type')?.value;
      const pathGroup = document.getElementById('trigger-path-group');
      const globGroup = document.getElementById('trigger-glob-group');
      const secretGroup = document.getElementById('trigger-secret-group');
      if (type === 'webhook') {
        if (pathGroup) pathGroup.style.display = 'none';
        if (globGroup) globGroup.style.display = 'none';
        if (secretGroup) secretGroup.style.display = 'block';
      } else {
        if (pathGroup) pathGroup.style.display = 'block';
        if (globGroup) globGroup.style.display = 'block';
        if (secretGroup) secretGroup.style.display = 'none';
      }
    },
    async save() {
      const type = document.getElementById('trigger-type')?.value;
      const name = document.getElementById('trigger-name')?.value.trim();
      const prompt = document.getElementById('trigger-prompt')?.value.trim();
      if (!prompt) return alert('Prompt is required.');
      const body = { type, prompt };
      if (name) body.name = name;
      if (type === 'file_watch') {
        const path = document.getElementById('trigger-path')?.value.trim();
        if (!path) return alert('Watch path is required for file watchers.');
        body.path = path;
        const glob = document.getElementById('trigger-glob')?.value.trim();
        if (glob) body.glob = glob;
      }
      if (type === 'webhook') {
        const secret = document.getElementById('trigger-secret')?.value.trim();
        if (secret) body.secret = secret;
      }
      try {
        const res = await fetch('/api/triggers', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const err = await res.json();
          return alert(err.error || 'Failed to create trigger');
        }
        openTriggersPanel();
      } catch (e) { alert('Error: ' + e.message); }
    },
    async toggle(id, enabled) {
      await fetch(`/api/triggers/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      openTriggersPanel();
    },
    async del(id) {
      if (!confirm('Delete this trigger?')) return;
      await fetch(`/api/triggers/${id}`, { method: 'DELETE' });
      openTriggersPanel();
    },
  };

  // ── RAG panel ──
  document.getElementById('rag-btn').addEventListener('click', () => {
    openRagPanel();
  });

  async function openRagPanel() {
    const panel = document.getElementById('right-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');
    title.textContent = 'RAG Indexes';
    panel.classList.remove('hidden');
    content.innerHTML = '<div class="spinner" style="margin:20px auto"></div>';

    try {
      const res = await fetch('/api/rag');
      const data = await res.json();
      renderRagPanel(content, data.text);
    } catch {
      content.innerHTML = '<p style="color:var(--danger)">Failed to load RAG indexes</p>';
    }
  }

  function renderRagPanel(container, ragText) {
    let html = '<div class="settings-section">';
    html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">';
    html += '<div class="settings-section-title" style="margin:0">Document Indexes</div>';
    html += '<button class="btn btn-primary btn-sm" onclick="window.RAG.showIndex()">+ Index Directory</button>';
    html += '</div>';

    // Parse the text output to display indexes
    if (ragText.includes('No indexes')) {
      html += '<p style="color:var(--text-muted);font-size:0.85rem">No RAG indexes yet. Index a directory to enable semantic search over your documents.</p>';
    } else {
      // Display the raw formatted text from the tool
      html += `<div style="font-size:0.85rem;white-space:pre-wrap;line-height:1.6;padding:12px;background:var(--surface-2, var(--surface));border-radius:6px;border:1px solid var(--border)">${Markdown.escapeHtml(ragText)}</div>`;
    }

    // Search form
    html += `<div style="margin-top:16px">
      <div class="settings-section-title">Semantic Search</div>
      <div style="display:flex;gap:8px">
        <input id="rag-search-query" type="text" placeholder="Search your indexed documents..." style="flex:1;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem" onkeydown="if(event.key==='Enter')window.RAG.search()">
        <input id="rag-search-index" type="text" placeholder="Index name (optional)" style="width:160px;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem">
        <button class="btn btn-primary btn-sm" onclick="window.RAG.search()">Search</button>
      </div>
      <div id="rag-search-results" style="margin-top:10px"></div>
    </div>`;

    // Index form
    html += `<div id="rag-index-form" style="display:none;margin-top:16px">
      <div class="settings-section-title">Index a Directory</div>
      <div style="display:flex;flex-direction:column;gap:10px">
        <div>
          <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">Directory path</label>
          <input id="rag-index-path" type="text" placeholder="/Users/you/project" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem">
        </div>
        <div>
          <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">Index name (optional — defaults to directory name)</label>
          <input id="rag-index-name" type="text" placeholder="my-project" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem">
        </div>
        <div style="display:flex;gap:8px">
          <button class="btn btn-primary btn-sm" onclick="window.RAG.doIndex()">Index</button>
          <button class="btn btn-ghost btn-sm" onclick="window.RAG.hideIndex()">Cancel</button>
        </div>
        <div id="rag-index-status" style="font-size:0.82rem"></div>
      </div>
    </div>`;

    // Delete section
    html += `<div style="margin-top:16px">
      <div class="settings-section-title">Manage</div>
      <div style="display:flex;gap:8px;align-items:center">
        <input id="rag-delete-name" type="text" placeholder="Index name to delete" style="flex:1;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem">
        <button class="btn btn-ghost btn-sm" onclick="window.RAG.del()" style="color:var(--danger)">Delete</button>
      </div>
    </div>`;

    html += '</div>';
    container.innerHTML = html;
  }

  window.RAG = {
    showIndex() {
      const form = document.getElementById('rag-index-form');
      if (form) form.style.display = 'block';
    },
    hideIndex() {
      const form = document.getElementById('rag-index-form');
      if (form) form.style.display = 'none';
    },
    async doIndex() {
      const dirPath = document.getElementById('rag-index-path')?.value.trim();
      const indexName = document.getElementById('rag-index-name')?.value.trim();
      const statusEl = document.getElementById('rag-index-status');
      if (!dirPath) return alert('Directory path is required.');
      if (statusEl) statusEl.innerHTML = '<span style="color:var(--primary)">Indexing... this may take a moment.</span>';
      try {
        const body = { path: dirPath };
        if (indexName) body.index_name = indexName;
        const res = await fetch('/api/rag/index', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (statusEl) statusEl.innerHTML = `<div style="white-space:pre-wrap;padding:8px;background:var(--surface-2, var(--surface));border-radius:6px;border:1px solid var(--border);margin-top:8px">${Markdown.escapeHtml(data.text)}</div>`;
        // Refresh the index list after a short delay
        setTimeout(() => openRagPanel(), 2000);
      } catch (e) {
        if (statusEl) statusEl.innerHTML = `<span style="color:var(--danger)">Error: ${e.message}</span>`;
      }
    },
    async search() {
      const query = document.getElementById('rag-search-query')?.value.trim();
      const indexName = document.getElementById('rag-search-index')?.value.trim();
      const resultsEl = document.getElementById('rag-search-results');
      if (!query) return;
      if (resultsEl) resultsEl.innerHTML = '<span style="color:var(--text-muted);font-size:0.82rem">Searching...</span>';
      try {
        const body = { query };
        if (indexName) body.index_name = indexName;
        const res = await fetch('/api/rag/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (resultsEl) resultsEl.innerHTML = `<div style="white-space:pre-wrap;font-size:0.82rem;padding:10px;background:var(--surface-2, var(--surface));border-radius:6px;border:1px solid var(--border);max-height:400px;overflow-y:auto">${Markdown.escapeHtml(data.text)}</div>`;
      } catch (e) {
        if (resultsEl) resultsEl.innerHTML = `<span style="color:var(--danger);font-size:0.82rem">Error: ${e.message}</span>`;
      }
    },
    async del() {
      const name = document.getElementById('rag-delete-name')?.value.trim();
      if (!name) return alert('Enter an index name to delete.');
      if (!confirm(`Delete RAG index "${name}"? This cannot be undone.`)) return;
      try {
        await fetch(`/api/rag/${encodeURIComponent(name)}`, { method: 'DELETE' });
        openRagPanel();
      } catch {}
    },
  };

  // ── Workflow panel ──
  document.getElementById('workflow-btn').addEventListener('click', () => {
    openWorkflowPanel();
  });

  async function openWorkflowPanel() {
    const panel = document.getElementById('right-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');
    title.textContent = 'Workflows';
    panel.classList.remove('hidden');
    content.innerHTML = '<div class="spinner" style="margin:20px auto"></div>';

    try {
      const res = await fetch('/api/workflows');
      const workflows = await res.json();
      renderWorkflowPanel(content, workflows);
    } catch {
      content.innerHTML = '<p style="color:var(--danger)">Failed to load workflows</p>';
    }
  }

  function renderWorkflowPanel(container, workflows) {
    let html = '<div class="settings-section">';
    html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">';
    html += '<div class="settings-section-title" style="margin:0">Saved Workflows</div>';
    html += '</div>';

    if (workflows.length === 0) {
      html += '<p style="color:var(--text-muted);font-size:0.85rem">No saved workflows yet. Record a workflow by asking PRE to start recording, then performing Computer Use actions.</p>';
    } else {
      for (const wf of workflows) {
        const dur = wf.duration ? `${(wf.duration / 1000).toFixed(0)}s` : '?';
        const created = wf.created ? new Date(wf.created).toLocaleDateString() : '?';
        html += `<div class="connection-card" style="margin-bottom:10px">
          <div class="connection-card-header" style="padding:12px 14px">
            <div style="flex:1;min-width:0">
              <div style="font-weight:600;font-size:0.9rem">${Markdown.escapeHtml(wf.name)}</div>
              ${wf.description ? `<div style="font-size:0.78rem;color:var(--text-muted);margin-top:2px">${Markdown.escapeHtml(wf.description)}</div>` : ''}
              <div style="font-size:0.73rem;color:var(--text-muted);margin-top:2px">${wf.stepCount} steps &middot; ${dur} duration &middot; ${created}</div>
            </div>
            <div style="display:flex;gap:4px;flex-shrink:0">
              <button class="btn btn-ghost btn-sm" onclick="window.Workflows.view('${Markdown.escapeHtml(wf.name)}')" title="View steps">View</button>
              <button class="btn btn-ghost btn-sm" onclick="window.Workflows.replay('${Markdown.escapeHtml(wf.name)}')" title="Replay workflow">Replay</button>
              <button class="btn btn-ghost btn-sm" onclick="window.Workflows.del('${Markdown.escapeHtml(wf.name)}')" style="color:var(--danger)" title="Delete">&times;</button>
            </div>
          </div>
        </div>`;
      }
    }

    html += '<div id="workflow-detail" style="margin-top:12px"></div>';
    html += '</div>';
    container.innerHTML = html;
  }

  window.Workflows = {
    async view(name) {
      const detailEl = document.getElementById('workflow-detail');
      if (!detailEl) return;
      detailEl.innerHTML = '<span style="color:var(--text-muted);font-size:0.82rem">Loading...</span>';
      try {
        const res = await fetch(`/api/workflows/${encodeURIComponent(name)}`);
        const data = await res.json();
        detailEl.innerHTML = `<div style="white-space:pre-wrap;font-size:0.82rem;padding:10px;background:var(--surface-2, var(--surface));border-radius:6px;border:1px solid var(--border);max-height:400px;overflow-y:auto">${Markdown.escapeHtml(data.text)}</div>`;
      } catch {
        detailEl.innerHTML = '<span style="color:var(--danger);font-size:0.82rem">Failed to load workflow</span>';
      }
    },
    async replay(name) {
      if (!confirm(`Replay workflow "${name}"? This will execute desktop actions.`)) return;
      const detailEl = document.getElementById('workflow-detail');
      if (detailEl) detailEl.innerHTML = '<span style="color:var(--primary);font-size:0.82rem">Replaying...</span>';
      try {
        const res = await fetch(`/api/workflows/${encodeURIComponent(name)}/replay`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ speed: 1.0 }),
        });
        const data = await res.json();
        if (detailEl) detailEl.innerHTML = `<div style="white-space:pre-wrap;font-size:0.82rem;padding:10px;background:var(--surface-2, var(--surface));border-radius:6px;border:1px solid var(--border)">${Markdown.escapeHtml(data.text)}</div>`;
      } catch (e) {
        if (detailEl) detailEl.innerHTML = `<span style="color:var(--danger);font-size:0.82rem">Replay failed: ${e.message}</span>`;
      }
    },
    async del(name) {
      if (!confirm(`Delete workflow "${name}"?`)) return;
      await fetch(`/api/workflows/${encodeURIComponent(name)}`, { method: 'DELETE' });
      openWorkflowPanel();
    },
  };

  // ── Agent Feed panel ──
  document.getElementById('agent-feed-btn').addEventListener('click', () => {
    openAgentFeedPanel();
  });

  // In-memory agent activity store (accumulates from WS events)
  const agentFeedEntries = [];

  let agentFeedHistoryLoaded = false;

  async function openAgentFeedPanel() {
    const panel = document.getElementById('right-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');
    title.textContent = 'Agent Feed';
    panel.classList.remove('hidden');

    // On first open, seed the feed with historical cron/agent sessions
    if (!agentFeedHistoryLoaded) {
      agentFeedHistoryLoaded = true;
      content.innerHTML = '<div class="agent-feed-empty">Loading agent history...</div>';
      try {
        const res = await fetch('/api/sessions');
        const sessions = await res.json();
        // Find sessions in the scheduled-jobs project that aren't already in the feed
        const existingIds = new Set(agentFeedEntries.map(e => e.id));
        const cronSessions = sessions
          .filter(s => s.projectSlug === 'scheduled-jobs' && !existingIds.has(s.id))
          .slice(0, 50); // cap to avoid flooding

        for (const s of cronSessions) {
          agentFeedEntries.push({
            id: s.id,
            task: s.displayName || s.preview || s.channel || 'Scheduled job',
            sessionId: s.id,
            status: 'completed',
            tools: [],
            result: s.preview || null,
            error: null,
            duration: null,
            startedAt: new Date(s.modified).getTime(),
          });
        }
      } catch (err) {
        console.warn('[agent-feed] Failed to load history:', err);
      }
    }

    renderAgentFeedPanel(content);
  }

  function renderAgentFeedPanel(container) {
    if (agentFeedEntries.length === 0) {
      container.innerHTML = '<div class="agent-feed-empty">No agent activity yet.<br>Agents will appear here when the model spawns sub-agents or background jobs run.</div>';
      return;
    }

    let html = '<div class="settings-section">';
    html += '<div class="settings-section-title" style="margin-bottom:12px">Agent Activity</div>';

    // Render newest first (sort by startedAt descending)
    const sorted = [...agentFeedEntries].sort((a, b) => (b.startedAt || 0) - (a.startedAt || 0));
    for (const entry of sorted) {
      const statusClass = entry.status === 'running' ? 'running' : entry.status === 'failed' ? 'failed' : 'completed';
      const dur = entry.duration || '';
      const toolCount = entry.tools.length;
      let time = '';
      if (entry.startedAt) {
        const d = new Date(entry.startedAt);
        const today = new Date();
        const isToday = d.toDateString() === today.toDateString();
        time = isToday
          ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          : d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      }
      const sessionBtn = entry.sessionId
        ? `<button class="btn btn-ghost btn-sm agent-feed-session-btn" onclick="window.AgentFeed.openSession('${escapeHtml(entry.sessionId)}')">Open Session</button>`
        : '';

      html += `<div class="agent-feed-entry ${statusClass}" data-agent-feed-id="${escapeHtml(entry.id)}">
        <div class="agent-feed-header" onclick="this.parentElement.classList.toggle('expanded')">
          <span class="agent-feed-status"></span>
          <div class="agent-feed-info">
            <div class="agent-feed-task">${escapeHtml(entry.task)}</div>
            <div class="agent-feed-meta">
              <span>${toolCount} tool${toolCount !== 1 ? 's' : ''}</span>
              ${dur ? `<span>${dur}</span>` : ''}
              <span>${time}</span>
            </div>
          </div>
          <svg class="agent-feed-chevron" width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M6.5 3l5 5-5 5V3z"/></svg>
        </div>
        <div class="agent-feed-body">
          ${toolCount > 0 ? '<ul class="agent-feed-tools">' + entry.tools.map(t => `<li>${escapeHtml(t)}</li>`).join('') + '</ul>' : ''}
          ${entry.result ? `<div class="agent-feed-result"><pre><code>${escapeHtml(entry.result.length > 2000 ? entry.result.slice(0, 2000) + '\n...' : entry.result)}</code></pre></div>` : ''}
          ${entry.error ? `<div style="color:var(--danger);font-size:0.82rem">Error: ${escapeHtml(entry.error)}</div>` : ''}
          ${sessionBtn}
        </div>
      </div>`;
    }

    html += '</div>';
    container.innerHTML = html;
  }

  // Global AgentFeed API — called from Chat agent cards, WS handler, and background job tracker
  window.AgentFeed = {
    add(agentId, task, sessionId) {
      agentFeedEntries.push({
        id: agentId,
        task,
        sessionId: sessionId || null,
        status: 'running',
        tools: [],
        result: null,
        error: null,
        duration: null,
        startedAt: Date.now(),
      });
      refreshFeedPanelIfOpen();
    },
    updateId(oldId, newId) {
      const entry = agentFeedEntries.find(e => e.id === oldId);
      if (entry) entry.id = newId;
    },
    setSessionId(agentId, sessionId) {
      const entry = agentFeedEntries.find(e => e.id === agentId);
      if (entry) entry.sessionId = sessionId;
    },
    addTool(agentId, toolName) {
      const entry = agentFeedEntries.find(e => e.id === agentId);
      if (entry) entry.tools.push(toolName);
      refreshFeedPanelIfOpen();
    },
    setResult(agentId, result) {
      const entry = agentFeedEntries.find(e => e.id === agentId);
      if (entry) entry.result = result;
      refreshFeedPanelIfOpen();
    },
    complete(agentId, duration) {
      const entry = agentFeedEntries.find(e => e.id === agentId);
      if (entry) {
        entry.status = 'completed';
        entry.duration = duration;
      }
      refreshFeedPanelIfOpen();
    },
    fail(agentId, error) {
      const entry = agentFeedEntries.find(e => e.id === agentId);
      if (entry) {
        entry.status = 'failed';
        entry.error = error;
      }
      refreshFeedPanelIfOpen();
    },
    open(agentId) {
      openAgentFeedPanel();
      // Expand the entry after render
      setTimeout(() => {
        const el = document.querySelector(`[data-agent-feed-id="${agentId}"]`);
        if (el) {
          el.classList.add('expanded');
          el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 50);
    },
    async openSession(sessionId) {
      // Close the feed panel and switch to the agent/cron session
      document.getElementById('right-panel').classList.add('hidden');
      setCurrentSession(sessionId);
      updateSessionName();
      loadSessionList();
      try {
        const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
        const messages = await res.json();
        Chat.loadHistory(messages);
      } catch {}
      WS.send({ type: 'switch_session', sessionId });
    },
    get entries() { return agentFeedEntries; },
  };

  function refreshFeedPanelIfOpen() {
    const panel = document.getElementById('right-panel');
    const title = document.getElementById('panel-title');
    if (!panel.classList.contains('hidden') && title.textContent === 'Agent Feed') {
      renderAgentFeedPanel(document.getElementById('panel-content'));
    }
  }

  // ── Tutorial system ──
  const TUTORIAL_DATA = [
    {
      icon: '💬', title: 'Getting Started', category: 'basics',
      tip: 'PRE auto-titles sessions and remembers context within a conversation.',
      prompts: [
        "What can you help me with? Give me a quick overview of your capabilities.",
        "What's running on my Mac right now? Show me the top 10 processes by CPU usage.",
        "Summarize what's in my Downloads folder.",
        "What day of the week was July 4, 1776?",
      ]
    },
    {
      icon: '📁', title: 'File Operations', category: 'files',
      tip: 'PRE can read, write, search, and edit any file on your system.',
      prompts: [
        "Find all Python files in my home directory that import pandas.",
        "Read my ~/.zshrc and suggest improvements for performance.",
        "Search my Documents folder for any file mentioning \"quarterly review\".",
        "Find every TODO comment in this project and create a summary.",
      ]
    },
    {
      icon: '🍎', title: 'Native macOS', category: 'macos',
      tip: 'Works with any email/calendar provider configured on your Mac — no API keys needed.',
      prompts: [
        "What's on my calendar today? Include meeting links if available.",
        "Check my email for anything from my boss in the last 3 days.",
        "Remind me to submit the expense report by Friday at 5pm.",
        "Search my notes for anything about the API migration.",
        "Find all PDF documents on my Mac that contain \"budget proposal\".",
      ]
    },
    {
      icon: '🖥️', title: 'Desktop Automation', category: 'computer',
      tip: 'PRE sees your screen and operates any app via mouse and keyboard.',
      prompts: [
        "Take a screenshot and describe what's on my screen.",
        "Open System Settings and navigate to the Wi-Fi section.",
        "Press Cmd+Space to open Spotlight, type \"Activity Monitor\", and press Enter.",
        "Start recording a workflow called \"morning-setup\".",
      ]
    },
    {
      icon: '🌐', title: 'Browser & Web', category: 'browser',
      tip: 'Built-in headless Chrome for scraping, browsing, and form filling.',
      prompts: [
        "Search the web for the latest news about Apple Silicon.",
        "Open the browser, go to news.ycombinator.com, and tell me the top 5 stories.",
        "Navigate to Wikipedia and find today's featured article.",
      ]
    },
    {
      icon: '🧠', title: 'Memory & RAG', category: 'memory',
      tip: 'Memories persist across sessions. RAG searches documents by meaning.',
      prompts: [
        "Remember that our team standup is at 9:15 AM Pacific every weekday.",
        "Search my memories for anything about deployment procedures.",
        "Index my ~/Documents/notes folder and call the index \"my-notes\".",
        "Search the \"my-notes\" index for anything about project deadlines.",
      ]
    },
    {
      icon: '⏰', title: 'Scheduling & Triggers', category: 'automation',
      tip: 'Cron jobs run in the background, even when you\'re away.',
      prompts: [
        "Schedule a daily morning briefing at 8am that checks my calendar and unread emails. Run it Monday through Friday.",
        "Create a trigger that watches ~/Downloads for new PDFs and summarizes them.",
        "List all my scheduled jobs and their next run times.",
      ]
    },
    {
      icon: '🤖', title: 'Sub-Agents', category: 'agents',
      tip: 'Agents work autonomously, each with their own session and tools.',
      prompts: [
        "Spawn agents to research PostgreSQL vs MySQL for a high-write workload, then compare their findings.",
        "Spawn an agent to read all README files in my ~/projects directory and summarize each project.",
      ]
    },
    {
      icon: '☁️', title: 'Cloud Integrations', category: 'cloud',
      tip: 'Configure integrations in Settings (gear icon). 15 services available.',
      prompts: [
        "Show me all Jira tickets assigned to me that are In Progress.",
        "Search Slack for messages about the production deployment in the last 24 hours.",
        "List my open pull requests on GitHub.",
        "What Zoom meetings do I have scheduled this week?",
      ]
    },
    {
      icon: '🎨', title: 'Artifacts & Exports', category: 'artifacts',
      tip: 'PRE creates interactive HTML documents, reports, and visualizations.',
      prompts: [
        "Create an interactive HTML dashboard showing a sample project timeline with milestones and progress bars.",
        "Build a Pomodoro timer as an HTML artifact with start/pause/reset buttons.",
        "Create a Word document summarizing today's meeting notes with action items.",
      ]
    },
    {
      icon: '🎙️', title: 'Voice Interface', category: 'voice',
      tip: 'All audio is processed locally via Whisper — nothing leaves your Mac.',
      prompts: [
        "Read this aloud: \"Good morning! Here's your daily briefing.\"",
        "What voices are available for text-to-speech?",
      ]
    },
    {
      icon: '🔥', title: 'Power Workflows', category: 'power',
      tip: 'Combine multiple features into real-world multi-step workflows.',
      prompts: [
        "Check my calendar for today, summarize important unread emails, list my top Jira tickets, and give me a morning briefing.",
        "Index this repository with RAG, then search for the authentication flow and database schema. Give me a developer onboarding summary.",
        "Check disk usage, list top 20 processes by memory, and verify network connectivity. Format as a system health report.",
      ]
    },
  ];

  // Send a tutorial prompt to the chat
  function sendTutorialPrompt(text) {
    input.value = text;
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
    input.focus();
    // Scroll input into view
    input.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  // Render welcome screen tutorial cards (compact, 6 random categories)
  function renderWelcomeTutorial() {
    const container = document.getElementById('welcome-tutorial');
    if (!container) return;

    // Pick 6 diverse categories for the welcome screen
    const showcaseOrder = ['basics', 'macos', 'computer', 'memory', 'automation', 'power'];
    const showcase = showcaseOrder
      .map(cat => TUTORIAL_DATA.find(t => t.category === cat))
      .filter(Boolean);

    container.innerHTML = showcase.map(t => {
      const prompt = t.prompts[Math.floor(Math.random() * t.prompts.length)];
      return `<div class="tutorial-card" onclick="window.Tutorial.send(this.dataset.prompt)" data-prompt="${escapeHtml(prompt)}">
        <span class="tutorial-card-icon">${t.icon}</span>
        <div class="tutorial-card-title">${escapeHtml(t.title)}</div>
        <div class="tutorial-card-prompt">${escapeHtml(prompt)}</div>
      </div>`;
    }).join('');
  }

  // Full tutorial panel (opened from sidebar or welcome)
  function openTutorialPanel() {
    const panel = document.getElementById('right-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');
    title.textContent = 'Tutorial';
    panel.classList.remove('hidden');

    let html = '<div style="padding:0 4px">';
    html += '<p style="font-size:0.78rem;color:var(--text-muted);margin-bottom:16px;line-height:1.5">Click any prompt below to load it into the chat input. These examples showcase every major feature of PRE.</p>';

    for (const cat of TUTORIAL_DATA) {
      html += `<div class="tutorial-category">`;
      html += `<div class="tutorial-category-title">${cat.icon} ${escapeHtml(cat.title)}</div>`;
      for (const prompt of cat.prompts) {
        html += `<div class="tutorial-prompt-item" onclick="window.Tutorial.send(this.dataset.prompt)" data-prompt="${escapeHtml(prompt)}">${escapeHtml(prompt)}</div>`;
      }
      html += `<div class="tutorial-tip">${escapeHtml(cat.tip)}</div>`;
      html += '</div>';
    }

    html += '</div>';
    content.innerHTML = html;
  }

  // Global Tutorial API
  window.Tutorial = {
    send(prompt) {
      if (prompt) sendTutorialPrompt(prompt);
    },
    open() {
      openTutorialPanel();
    },
    _renderWelcome() {
      renderWelcomeTutorial();
    },
  };

  // Render welcome tutorial on load
  renderWelcomeTutorial();

  // ── Voice input ──
  (async function initVoice() {
    const voiceBtn = document.getElementById('voice-btn');
    if (!voiceBtn) return;

    // Check if voice capabilities are available
    try {
      const res = await fetch('/api/voice/status');
      const status = await res.json();
      if (status.whisper) {
        voiceBtn.classList.remove('hidden');
      }
    } catch {
      // Voice not available — leave button hidden
      return;
    }

    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    voiceBtn.addEventListener('mousedown', startRecording);
    voiceBtn.addEventListener('mouseup', stopRecording);
    voiceBtn.addEventListener('mouseleave', stopRecording);
    // Touch support
    voiceBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
    voiceBtn.addEventListener('touchend', (e) => { e.preventDefault(); stopRecording(); });

    async function startRecording() {
      if (isRecording) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = async () => {
          stream.getTracks().forEach(t => t.stop());
          if (audioChunks.length === 0) return;
          const blob = new Blob(audioChunks, { type: 'audio/webm' });
          const reader = new FileReader();
          reader.onloadend = async () => {
            const base64 = reader.result.split(',')[1];
            voiceBtn.title = 'Transcribing...';
            try {
              const res = await fetch('/api/voice/transcribe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ audio: base64, mime_type: 'audio/webm' }),
              });
              const data = await res.json();
              if (data.text) {
                const input = document.getElementById('chat-input');
                if (input) {
                  input.value = (input.value ? input.value + ' ' : '') + data.text;
                  input.focus();
                  input.dispatchEvent(new Event('input'));
                }
              }
            } catch {}
            voiceBtn.title = 'Voice input (hold to record)';
          };
          reader.readAsDataURL(blob);
        };
        mediaRecorder.start();
        isRecording = true;
        voiceBtn.classList.add('recording');
        voiceBtn.title = 'Recording... release to transcribe';
      } catch {
        // Microphone permission denied or not available
      }
    }

    function stopRecording() {
      if (!isRecording || !mediaRecorder) return;
      isRecording = false;
      voiceBtn.classList.remove('recording');
      if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
    }
  })();

  async function openSettingsPanel() {
    const panel = document.getElementById('right-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');
    title.textContent = 'Settings';
    panel.classList.remove('hidden');

    content.innerHTML = '<div class="spinner" style="margin:20px auto"></div>';

    try {
      const [connRes, mcpRes, sysRes] = await Promise.all([
        fetch('/api/connections'),
        fetch('/api/mcp'),
        fetch('/api/system/autostart'),
      ]);
      const connections = await connRes.json();
      const mcpStatus = await mcpRes.json();
      const autostart = await sysRes.json();
      renderConnectionsPanel(content, connections, mcpStatus, autostart);
    } catch {
      content.innerHTML = '<p style="color:var(--danger)">Failed to load settings</p>';
    }
  }

  function renderConnectionsPanel(container, connections, mcpStatus, autostart) {
    let html = '';

    // Tutorial banner at top
    html += '<div style="margin-bottom:20px;background:linear-gradient(135deg,#1e293b,#0f172a);border-radius:10px;padding:16px 20px;display:flex;align-items:center;gap:14px;cursor:pointer;transition:transform 0.15s,box-shadow 0.15s" onclick="window.open(\'/tutorial.html\',\'_blank\')" onmouseenter="this.style.transform=\'translateY(-1px)\';this.style.boxShadow=\'0 4px 12px rgba(0,0,0,0.15)\'" onmouseleave="this.style.transform=\'\';this.style.boxShadow=\'\'">';
    html += '<div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#2563eb,#7c3aed);display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0">\uD83C\uDF93</div>';
    html += '<div style="flex:1;min-width:0">';
    html += '<div style="font-family:Georgia,serif;font-weight:700;font-size:0.95rem;color:#fff">PRE Academy</div>';
    html += '<div style="font-size:0.78rem;color:#94a3b8">Interactive tutorial — learn every feature step by step</div>';
    html += '</div>';
    html += '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>';
    html += '</div>';

    html += '<div class="settings-section">';
    html += '<div class="settings-section-title">Connections</div>';

    for (const conn of connections) {
      const icon = SERVICE_ICONS[conn.service] || '';
      const statusClass = conn.active ? 'connected' : '';
      const statusText = conn.active ? 'Connected' : 'Not configured';

      html += `<div class="connection-card ${conn.active ? 'active' : ''}" data-service="${conn.service}">`;
      html += `<div class="connection-card-header">`;
      html += `<div class="connection-card-icon">${icon}</div>`;
      html += `<div class="connection-card-info">`;
      html += `<div class="connection-card-name">${escapeHtml(conn.label)}</div>`;
      html += `<div class="connection-card-status ${statusClass}">`;
      if (conn.active && conn.masked) {
        html += `${statusText} &middot; ${escapeHtml(conn.masked)}`;
      } else if (conn.active && conn.type === 'oauth') {
        html += `${statusText}`;
      } else {
        html += statusText;
      }
      html += `</div></div></div>`;

      // Action area
      html += `<div class="connection-card-actions" id="conn-actions-${conn.service}">`;
      if (conn.type === 'api_key') {
        if (conn.active) {
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.editKey('${conn.service}')">Edit</button>`;
          html += `<button class="btn btn-ghost btn-sm" style="color:var(--danger)" onclick="Settings.removeConn('${conn.service}')">Remove</button>`;
        } else {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.addKey('${conn.service}')">Add API Key</button>`;
        }
      } else if (conn.type === 'oauth' && conn.service === 'google') {
        if (conn.active) {
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.reconnectGoogle()">Reconnect</button>`;
          html += `<button class="btn btn-ghost btn-sm" style="color:var(--danger)" onclick="Settings.removeConn('${conn.service}')">Remove</button>`;
        } else if (conn.hasCredentials) {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.authorizeGoogle()">Authorize</button>`;
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.editGoogleCreds()">Edit Credentials</button>`;
        } else {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.setupGoogle()">Setup</button>`;
        }
      } else if (conn.type === 'oauth' && conn.service === 'microsoft') {
        if (conn.active) {
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.reconnectMicrosoft()">Reconnect</button>`;
          html += `<button class="btn btn-ghost btn-sm" style="color:var(--danger)" onclick="Settings.removeConn('${conn.service}')">Remove</button>`;
        } else if (conn.hasCredentials) {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.authorizeMicrosoft()">Authorize</button>`;
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.editMicrosoftCreds()">Edit Credentials</button>`;
        } else {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.setupMicrosoft()">Setup</button>`;
        }
      } else if (conn.type === 'jira') {
        if (conn.active) {
          html += `<div style="font-size:0.8rem;margin-bottom:6px;color:var(--text-muted)">${escapeHtml(conn.url)}</div>`;
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.setupJira()">Edit</button>`;
          html += `<button class="btn btn-ghost btn-sm" style="color:var(--danger)" onclick="Settings.removeConn('${conn.service}')">Remove</button>`;
        } else {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.setupJira()">Setup</button>`;
        }
      } else if (conn.type === 'confluence') {
        if (conn.active) {
          html += `<div style="font-size:0.8rem;margin-bottom:6px;color:var(--text-muted)">${escapeHtml(conn.url)}</div>`;
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.setupConfluence()">Edit</button>`;
          html += `<button class="btn btn-ghost btn-sm" style="color:var(--danger)" onclick="Settings.removeConn('${conn.service}')">Remove</button>`;
        } else {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.setupConfluence()">Setup</button>`;
        }
      } else if (conn.type === 'slack') {
        if (conn.active) {
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.setupSlack()">Edit</button>`;
          html += `<button class="btn btn-ghost btn-sm" style="color:var(--danger)" onclick="Settings.removeConn('${conn.service}')">Remove</button>`;
        } else {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.setupSlack()">Setup</button>`;
        }
      } else if (conn.type === 'zoom') {
        if (conn.active) {
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.setupZoom()">Edit</button>`;
          html += `<button class="btn btn-ghost btn-sm" style="color:var(--danger)" onclick="Settings.removeConn('${conn.service}')">Remove</button>`;
        } else {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.setupZoom()">Setup</button>`;
        }
      } else if (conn.type === 'dynamics365') {
        if (conn.active) {
          const mode = conn.authMode === 'delegated' ? 'Delegated (user)' : 'Client credentials (app)';
          html += `<div style="font-size:0.8rem;margin-bottom:6px;color:var(--text-muted)">${escapeHtml(conn.url)} &middot; ${mode}</div>`;
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.reconnectD365()">Reconnect</button>`;
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.setupD365()">Edit Credentials</button>`;
          html += `<button class="btn btn-ghost btn-sm" style="color:var(--danger)" onclick="Settings.removeConn('${conn.service}')">Remove</button>`;
        } else if (conn.hasCredentials) {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.authorizeD365()">Authorize</button>`;
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.setupD365()">Edit Credentials</button>`;
        } else {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.setupD365()">Setup</button>`;
        }
      } else if (conn.type === 'telegram') {
        if (conn.active) {
          const chatStatus = conn.chatId
            ? `<span style="color:var(--success)">Chat ID: ${escapeHtml(conn.chatId)}</span>`
            : `<span style="color:var(--warning)">No chat ID — send a message to the bot or set it below</span>`;
          html += `<div style="font-size:0.8rem;margin-bottom:6px">${chatStatus}</div>`;
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.setupTelegram()">Configure</button>`;
          html += `<button class="btn btn-ghost btn-sm" style="color:var(--danger)" onclick="Settings.removeConn('${conn.service}')">Remove</button>`;
        } else {
          html += `<button class="btn btn-primary btn-sm" onclick="Settings.setupTelegram()">Setup</button>`;
        }
      }
      html += `</div></div>`;
    }

    html += '</div>';

    // ── MCP Servers section ──
    html += '<div class="settings-section">';
    html += '<div class="settings-section-title">MCP Servers</div>';

    const mcpNames = Object.keys(mcpStatus || {});
    if (mcpNames.length === 0) {
      html += '<p style="font-size:0.8rem;color:var(--text-muted);margin:0 0 8px">No MCP servers configured. Add a server to extend PRE with external tools.</p>';
    }

    for (const name of mcpNames) {
      const srv = mcpStatus[name];
      const active = srv.connected && !srv.disabled;
      const statusClass = active ? 'connected' : '';
      let statusText = srv.disabled ? 'Disabled' : (srv.connected ? `Connected (${srv.tools} tool${srv.tools !== 1 ? 's' : ''})` : 'Disconnected');
      const typeLabel = srv.type === 'http' ? srv.url : srv.command;

      html += `<div class="connection-card ${active ? 'active' : ''}" data-mcp="${escapeHtml(name)}">`;
      html += `<div class="connection-card-header">`;
      html += `<div class="connection-card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg></div>`;
      html += `<div class="connection-card-info">`;
      html += `<div class="connection-card-name">${escapeHtml(name)}</div>`;
      html += `<div class="connection-card-status ${statusClass}">`;
      html += statusText;
      if (typeLabel) html += ` &middot; <span style="opacity:0.7">${escapeHtml(typeLabel)}</span>`;
      html += `</div></div></div>`;

      html += `<div class="connection-card-actions" id="mcp-actions-${CSS.escape(name)}">`;
      if (srv.connected) {
        if (srv.tools > 0) {
          html += `<button class="btn btn-ghost btn-sm" onclick="Settings.mcpShowTools('${escapeHtml(name)}')">Tools</button>`;
        }
        html += `<button class="btn btn-ghost btn-sm" onclick="Settings.mcpDisconnect('${escapeHtml(name)}')">Disconnect</button>`;
      } else if (!srv.disabled) {
        html += `<button class="btn btn-primary btn-sm" onclick="Settings.mcpConnect('${escapeHtml(name)}')">Connect</button>`;
      }
      html += `<button class="btn btn-ghost btn-sm" style="color:var(--danger)" onclick="Settings.mcpRemove('${escapeHtml(name)}')">Remove</button>`;
      html += `</div></div>`;
    }

    // Add server button
    html += `<div id="mcp-add-area">`;
    html += `<button class="btn btn-primary btn-sm" onclick="Settings.mcpShowAdd()">Add MCP Server</button>`;
    html += `</div>`;
    html += '</div>';

    // ── System section ──
    if (autostart) {
      html += '<div class="settings-section">';
      html += '<div class="settings-section-title">System</div>';

      const autostartOn = autostart.installed;
      html += `<div class="setting-toggle ${autostartOn ? 'active' : ''}">`;
      html += `<div class="setting-toggle-info">`;
      html += `<div class="setting-toggle-name">Start at login</div>`;
      html += `<div class="setting-toggle-desc">`;
      html += autostartOn
        ? 'PRE server starts automatically when you log in'
        : 'Launch PRE manually with <code>pre-launch</code>';
      html += `</div></div>`;
      html += `<label class="toggle-switch">`;
      html += `<input type="checkbox" ${autostartOn ? 'checked' : ''} onchange="Settings.toggleAutostart(this.checked)">`;
      html += `<span class="slider"></span>`;
      html += `</label>`;
      html += `</div>`;

      html += '</div>';
    }

    container.innerHTML = html;
  }

  // Global Settings object for onclick handlers
  window.Settings = {
    async toggleAutostart(enabled) {
      try {
        await fetch('/api/system/autostart', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled }),
        });
      } catch { /* ignore */ }
      this.refresh();
    },

    async addKey(service) {
      const actionsEl = document.getElementById(`conn-actions-${service}`);
      if (!actionsEl) return;
      actionsEl.innerHTML = `
        <div class="connection-input-group">
          <div class="connection-input-row">
            <input type="password" id="key-input-${service}" placeholder="Paste API key..." autocomplete="off">
            <button class="btn btn-primary btn-sm" onclick="Settings.saveKey('${service}')">Save</button>
            <button class="btn btn-ghost btn-sm" onclick="Settings.refresh()">Cancel</button>
          </div>
        </div>
      `;
      document.getElementById(`key-input-${service}`).focus();
    },

    editKey(service) {
      this.addKey(service);
    },

    async saveKey(service) {
      const input = document.getElementById(`key-input-${service}`);
      const key = input?.value.trim();
      if (!key) return;
      try {
        await fetch(`/api/connections/${service}/key`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key }),
        });
        this.refresh();
      } catch {}
    },

    async removeConn(service) {
      if (!confirm(`Remove ${service} connection?`)) return;
      await fetch(`/api/connections/${service}`, { method: 'DELETE' });
      this.refresh();
    },

    setupGoogle() {
      const actionsEl = document.getElementById('conn-actions-google');
      if (!actionsEl) return;
      actionsEl.innerHTML = `
        <div class="connection-input-group">
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0">
            Create OAuth credentials at
            <a href="https://console.cloud.google.com/apis/credentials" target="_blank" style="color:var(--primary)">Google Cloud Console</a>.
            Enable Gmail, Drive, and Docs APIs. Set redirect URI to
            <code style="font-size:0.75rem">http://localhost:7749/oauth/callback</code>
          </p>
          <input type="text" id="google-client-id" placeholder="Client ID" autocomplete="off">
          <input type="password" id="google-client-secret" placeholder="Client Secret" autocomplete="off">
          <div class="connection-input-row">
            <button class="btn btn-primary btn-sm" onclick="Settings.saveGoogleCreds()">Save & Authorize</button>
            <button class="btn btn-ghost btn-sm" onclick="Settings.refresh()">Cancel</button>
          </div>
        </div>
      `;
      document.getElementById('google-client-id').focus();
    },

    editGoogleCreds() {
      this.setupGoogle();
    },

    async saveGoogleCreds() {
      const clientId = document.getElementById('google-client-id')?.value.trim();
      const clientSecret = document.getElementById('google-client-secret')?.value.trim();
      if (!clientId || !clientSecret) return;
      try {
        await fetch('/api/connections/google/credentials', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ clientId, clientSecret }),
        });
        this.authorizeGoogle();
      } catch {}
    },

    async authorizeGoogle() {
      try {
        const res = await fetch('/api/connections/google/auth-url');
        const data = await res.json();
        if (data.url) {
          window.open(data.url, '_blank');
          // Show a waiting message
          const actionsEl = document.getElementById('conn-actions-google');
          if (actionsEl) {
            actionsEl.innerHTML = `
              <div style="font-size:0.8rem;color:var(--text-muted);display:flex;align-items:center;gap:8px">
                <div class="spinner" style="width:14px;height:14px;border-width:2px"></div>
                Waiting for authorization... Complete sign-in in the opened tab.
              </div>
              <button class="btn btn-ghost btn-sm" style="margin-top:8px" onclick="Settings.refresh()">Done</button>
            `;
          }
        }
      } catch {}
    },

    reconnectGoogle() {
      this.authorizeGoogle();
    },

    setupMicrosoft() {
      const actionsEl = document.getElementById('conn-actions-microsoft');
      if (!actionsEl) return;
      actionsEl.innerHTML = `
        <div class="connection-input-group">
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0">
            Register an app at
            <a href="https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade" target="_blank" style="color:var(--primary)">Azure App Registrations</a>.
            Add API permissions: <code style="font-size:0.75rem">Sites.Read.All</code>, <code style="font-size:0.75rem">Files.ReadWrite.All</code>, <code style="font-size:0.75rem">User.Read</code>.
            Set redirect URI to
            <code style="font-size:0.75rem">http://localhost:7749/oauth/microsoft/callback</code> (Web platform).
          </p>
          <input type="text" id="ms-tenant-id" placeholder="Tenant ID (Directory ID)" autocomplete="off">
          <input type="text" id="ms-client-id" placeholder="Application (Client) ID" autocomplete="off">
          <input type="password" id="ms-client-secret" placeholder="Client Secret (Value)" autocomplete="off">
          <div class="connection-input-row">
            <button class="btn btn-primary btn-sm" onclick="Settings.saveMicrosoftCreds()">Save & Authorize</button>
            <button class="btn btn-ghost btn-sm" onclick="Settings.refresh()">Cancel</button>
          </div>
        </div>
      `;
      document.getElementById('ms-tenant-id').focus();
    },

    editMicrosoftCreds() {
      this.setupMicrosoft();
    },

    async saveMicrosoftCreds() {
      const tenantId = document.getElementById('ms-tenant-id')?.value.trim();
      const clientId = document.getElementById('ms-client-id')?.value.trim();
      const clientSecret = document.getElementById('ms-client-secret')?.value.trim();
      if (!tenantId || !clientId || !clientSecret) return;
      try {
        await fetch('/api/connections/microsoft/credentials', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tenantId, clientId, clientSecret }),
        });
        this.authorizeMicrosoft();
      } catch {}
    },

    async authorizeMicrosoft() {
      try {
        const res = await fetch('/api/connections/microsoft/auth-url');
        const data = await res.json();
        if (data.url) {
          window.open(data.url, '_blank');
          const actionsEl = document.getElementById('conn-actions-microsoft');
          if (actionsEl) {
            actionsEl.innerHTML = `
              <div style="font-size:0.8rem;color:var(--text-muted);display:flex;align-items:center;gap:8px">
                <div class="spinner" style="width:14px;height:14px;border-width:2px"></div>
                Waiting for Microsoft authorization... Complete sign-in in the opened tab.
              </div>
              <button class="btn btn-ghost btn-sm" style="margin-top:8px" onclick="Settings.refresh()">Done</button>
            `;
          }
        }
      } catch {}
    },

    reconnectMicrosoft() {
      this.authorizeMicrosoft();
    },

    setupJira() {
      const actionsEl = document.getElementById('conn-actions-jira');
      if (!actionsEl) return;
      actionsEl.innerHTML = `
        <div class="connection-input-group">
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0">
            Enter your Jira Server URL and a
            <a href="https://confluence.atlassian.com/enterprise/using-personal-access-tokens-1026032365.html" target="_blank" style="color:var(--primary)">Personal Access Token</a>.
          </p>
          <input type="text" id="jira-url-input" placeholder="Jira URL (e.g. https://jira.company.com)" autocomplete="off">
          <input type="password" id="jira-token-input" placeholder="Personal Access Token" autocomplete="off">
          <div class="connection-input-row">
            <button class="btn btn-primary btn-sm" onclick="Settings.saveJira()">Save</button>
            <button class="btn btn-ghost btn-sm" onclick="Settings.refresh()">Cancel</button>
          </div>
        </div>
      `;
      document.getElementById('jira-url-input').focus();
    },

    async saveJira() {
      const url = document.getElementById('jira-url-input')?.value.trim();
      const token = document.getElementById('jira-token-input')?.value.trim();
      if (!url || !token) return;
      try {
        await fetch('/api/connections/jira/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, token }),
        });
        this.refresh();
      } catch {}
    },

    setupConfluence() {
      const actionsEl = document.getElementById('conn-actions-confluence');
      if (!actionsEl) return;
      actionsEl.innerHTML = `
        <div class="connection-input-group">
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0">
            Enter your Confluence Server URL and a
            <a href="https://confluence.atlassian.com/enterprise/using-personal-access-tokens-1026032365.html" target="_blank" style="color:var(--primary)">Personal Access Token</a>.
          </p>
          <input type="text" id="confluence-url-input" placeholder="Confluence URL (e.g. https://confluence.company.com)" autocomplete="off">
          <input type="password" id="confluence-token-input" placeholder="Personal Access Token" autocomplete="off">
          <div class="connection-input-row">
            <button class="btn btn-primary btn-sm" onclick="Settings.saveConfluence()">Save</button>
            <button class="btn btn-ghost btn-sm" onclick="Settings.refresh()">Cancel</button>
          </div>
        </div>
      `;
      document.getElementById('confluence-url-input').focus();
    },

    async saveConfluence() {
      const url = document.getElementById('confluence-url-input')?.value.trim();
      const token = document.getElementById('confluence-token-input')?.value.trim();
      if (!url || !token) return;
      try {
        await fetch('/api/connections/confluence/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, token }),
        });
        this.refresh();
      } catch {}
    },

    setupSlack() {
      const actionsEl = document.getElementById('conn-actions-slack');
      if (!actionsEl) return;
      actionsEl.innerHTML = `
        <div class="connection-input-group">
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0">
            1. Create a Slack App at <a href="https://api.slack.com/apps" target="_blank" style="color:var(--primary)">api.slack.com/apps</a>.<br>
            2. Under <strong>OAuth & Permissions</strong>, add bot scopes: <code style="font-size:0.75rem">channels:read</code>, <code style="font-size:0.75rem">channels:history</code>, <code style="font-size:0.75rem">chat:write</code>, <code style="font-size:0.75rem">reactions:write</code>, <code style="font-size:0.75rem">users:read</code>, <code style="font-size:0.75rem">search:read</code>.<br>
            3. Install to your workspace and copy the <strong>Bot User OAuth Token</strong>.
          </p>
          <div class="connection-input-row">
            <input type="password" id="slack-token-input" placeholder="Bot User OAuth Token (xoxb-...)" autocomplete="off">
          </div>
          <div class="connection-input-row">
            <button class="btn btn-primary btn-sm" onclick="Settings.saveSlack()">Save</button>
            <button class="btn btn-ghost btn-sm" onclick="Settings.refresh()">Cancel</button>
          </div>
        </div>
      `;
      document.getElementById('slack-token-input').focus();
    },

    async saveSlack() {
      const token = document.getElementById('slack-token-input')?.value.trim();
      if (!token) return;
      try {
        await fetch('/api/connections/slack/key', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key: token }),
        });
        this.refresh();
      } catch {}
    },

    setupZoom() {
      const actionsEl = document.getElementById('conn-actions-zoom');
      if (!actionsEl) return;
      actionsEl.innerHTML = `
        <div class="connection-input-group">
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0">
            1. Create a Server-to-Server OAuth app at <a href="https://marketplace.zoom.us/develop/create" target="_blank" style="color:var(--primary)">Zoom Marketplace</a>.<br>
            2. Add scopes: <code style="font-size:0.75rem">meeting:read</code>, <code style="font-size:0.75rem">meeting:write</code>, <code style="font-size:0.75rem">user:read</code>, <code style="font-size:0.75rem">recording:read</code>.<br>
            3. Copy the Account ID, Client ID, and Client Secret.
          </p>
          <input type="text" id="zoom-account-id" placeholder="Account ID" autocomplete="off">
          <input type="text" id="zoom-client-id" placeholder="Client ID" autocomplete="off">
          <input type="password" id="zoom-client-secret" placeholder="Client Secret" autocomplete="off">
          <div class="connection-input-row">
            <button class="btn btn-primary btn-sm" onclick="Settings.saveZoom()">Save</button>
            <button class="btn btn-ghost btn-sm" onclick="Settings.refresh()">Cancel</button>
          </div>
        </div>
      `;
      document.getElementById('zoom-account-id').focus();
    },

    async saveZoom() {
      const accountId = document.getElementById('zoom-account-id')?.value.trim();
      const clientId = document.getElementById('zoom-client-id')?.value.trim();
      const clientSecret = document.getElementById('zoom-client-secret')?.value.trim();
      if (!accountId || !clientId || !clientSecret) return;
      try {
        await fetch('/api/connections/zoom/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ accountId, clientId, clientSecret }),
        });
        this.refresh();
      } catch {}
    },

    setupD365() {
      const actionsEl = document.getElementById('conn-actions-dynamics365');
      if (!actionsEl) return;
      actionsEl.innerHTML = `
        <div class="connection-input-group">
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0">
            1. Register an app in <a href="https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade" target="_blank" style="color:var(--primary)">Azure AD (Entra ID)</a>.<br>
            2. Add API permission: <strong>Dynamics CRM &gt; user_impersonation</strong>.<br>
            3. Create a client secret under Certificates & Secrets.<br>
            4. Set redirect URI to <code style="font-size:0.75rem">http://localhost:7749/oauth/dynamics365/callback</code> (Web platform).
          </p>
          <input type="text" id="d365-url-input" placeholder="Environment URL (e.g. https://org.crm.dynamics.com)" autocomplete="off">
          <input type="text" id="d365-tenant-input" placeholder="Azure Tenant ID" autocomplete="off">
          <input type="text" id="d365-client-id-input" placeholder="Client ID (Application ID)" autocomplete="off">
          <input type="password" id="d365-client-secret-input" placeholder="Client Secret" autocomplete="off">
          <div class="connection-input-row">
            <button class="btn btn-primary btn-sm" onclick="Settings.saveD365Creds()">Save & Authorize</button>
            <button class="btn btn-ghost btn-sm" onclick="Settings.saveD365Direct()">Save (Client Credentials Only)</button>
            <button class="btn btn-ghost btn-sm" onclick="Settings.refresh()">Cancel</button>
          </div>
        </div>
      `;
      document.getElementById('d365-url-input').focus();
    },

    async saveD365Creds() {
      const url = document.getElementById('d365-url-input')?.value.trim();
      const tenantId = document.getElementById('d365-tenant-input')?.value.trim();
      const clientId = document.getElementById('d365-client-id-input')?.value.trim();
      const clientSecret = document.getElementById('d365-client-secret-input')?.value.trim();
      if (!url || !tenantId || !clientId || !clientSecret) return;
      try {
        await fetch('/api/connections/dynamics365/credentials', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, tenantId, clientId, clientSecret }),
        });
        this.authorizeD365();
      } catch {}
    },

    async saveD365Direct() {
      const url = document.getElementById('d365-url-input')?.value.trim();
      const tenantId = document.getElementById('d365-tenant-input')?.value.trim();
      const clientId = document.getElementById('d365-client-id-input')?.value.trim();
      const clientSecret = document.getElementById('d365-client-secret-input')?.value.trim();
      if (!url || !tenantId || !clientId || !clientSecret) return;
      try {
        await fetch('/api/connections/dynamics365/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, tenantId, clientId, clientSecret }),
        });
        this.refresh();
      } catch {}
    },

    async authorizeD365() {
      try {
        const res = await fetch('/api/connections/dynamics365/auth-url');
        const data = await res.json();
        if (data.url) {
          window.open(data.url, '_blank');
          const actionsEl = document.getElementById('conn-actions-dynamics365');
          if (actionsEl) {
            actionsEl.innerHTML = `
              <div style="font-size:0.8rem;color:var(--text-muted);display:flex;align-items:center;gap:8px">
                <div class="spinner" style="width:14px;height:14px;border-width:2px"></div>
                Waiting for D365 authorization... Complete sign-in in the opened tab.
              </div>
              <button class="btn btn-ghost btn-sm" style="margin-top:8px" onclick="Settings.refresh()">Done</button>
            `;
          }
        }
      } catch {}
    },

    reconnectD365() {
      this.authorizeD365();
    },

    setupTelegram() {
      const actionsEl = document.getElementById('conn-actions-telegram');
      if (!actionsEl) return;
      actionsEl.innerHTML = `
        <div class="connection-input-group">
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0">
            1. Create a bot with <a href="https://t.me/BotFather" target="_blank" style="color:var(--primary)">@BotFather</a> on Telegram.
            Copy the bot token below.
          </p>
          <div class="connection-input-row">
            <input type="password" id="tg-token-input" placeholder="Bot token (e.g. 123456:ABC-DEF...)" autocomplete="off">
            <button class="btn btn-ghost btn-sm" onclick="Settings.testTelegram()">Test</button>
          </div>
          <div id="tg-test-result" style="font-size:0.8rem;min-height:1.2em"></div>
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0">
            2. Your Chat ID — send any message to the bot, then click "Detect", or enter it manually.
            Find your ID via <a href="https://t.me/userinfobot" target="_blank" style="color:var(--primary)">@userinfobot</a>.
          </p>
          <div class="connection-input-row">
            <input type="text" id="tg-chatid-input" placeholder="Chat ID (e.g. 6902857843)" autocomplete="off">
            <button class="btn btn-ghost btn-sm" onclick="Settings.detectTelegramChat()">Detect</button>
          </div>
          <div id="tg-chat-result" style="font-size:0.8rem;min-height:1.2em"></div>
          <div class="connection-input-row">
            <button class="btn btn-primary btn-sm" onclick="Settings.saveTelegram()">Save</button>
            <button class="btn btn-ghost btn-sm" onclick="Settings.refresh()">Cancel</button>
          </div>
        </div>
      `;
      document.getElementById('tg-token-input').focus();
    },

    async testTelegram() {
      const token = document.getElementById('tg-token-input')?.value.trim();
      const resultEl = document.getElementById('tg-test-result');
      if (!token) { resultEl.innerHTML = '<span style="color:var(--danger)">Enter a token first</span>'; return; }
      resultEl.innerHTML = '<span style="color:var(--text-muted)">Testing...</span>';
      try {
        const res = await fetch('/api/connections/telegram/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token }),
        });
        const data = await res.json();
        if (data.success) {
          resultEl.innerHTML = `<span style="color:var(--success)">Bot: @${escapeHtml(data.bot.username)} (${escapeHtml(data.bot.first_name)})</span>`;
        } else {
          resultEl.innerHTML = `<span style="color:var(--danger)">${escapeHtml(data.error)}</span>`;
        }
      } catch {
        resultEl.innerHTML = '<span style="color:var(--danger)">Test failed</span>';
      }
    },

    async detectTelegramChat() {
      const token = document.getElementById('tg-token-input')?.value.trim();
      const resultEl = document.getElementById('tg-chat-result');
      const chatInput = document.getElementById('tg-chatid-input');
      if (!token) { resultEl.innerHTML = '<span style="color:var(--danger)">Enter bot token first</span>'; return; }
      resultEl.innerHTML = '<span style="color:var(--text-muted)">Checking for messages...</span>';
      try {
        // Use the bot token to call getUpdates directly
        const res = await fetch(`https://api.telegram.org/bot${token}/getUpdates?limit=5`);
        const data = await res.json();
        if (data.ok && data.result && data.result.length > 0) {
          const lastMsg = data.result.reverse().find(u => u.message);
          if (lastMsg) {
            const chatId = lastMsg.message.chat.id;
            const from = lastMsg.message.from?.first_name || 'Unknown';
            chatInput.value = chatId;
            resultEl.innerHTML = `<span style="color:var(--success)">Found: ${escapeHtml(from)} (${chatId})</span>`;
          } else {
            resultEl.innerHTML = '<span style="color:var(--warning)">No messages found. Send a message to the bot first.</span>';
          }
        } else {
          resultEl.innerHTML = '<span style="color:var(--warning)">No messages yet. Send a message to the bot first.</span>';
        }
      } catch {
        resultEl.innerHTML = '<span style="color:var(--danger)">Detection failed</span>';
      }
    },

    async saveTelegram() {
      const token = document.getElementById('tg-token-input')?.value.trim();
      const chatId = document.getElementById('tg-chatid-input')?.value.trim();
      if (!token) return;
      try {
        // Save the bot token
        await fetch('/api/connections/telegram/key', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key: token }),
        });
        // Save chat_id if provided
        if (chatId) {
          await fetch('/api/connections/telegram/chat-id', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chatId }),
          });
        }
        this.refresh();
      } catch {}
    },

    // ── MCP Server management ──

    mcpShowAdd() {
      const area = document.getElementById('mcp-add-area');
      if (!area) return;
      area.innerHTML = `
        <div class="connection-input-group">
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0">
            Add a <strong>stdio</strong> server (local command) or an <strong>HTTP</strong> server (remote URL).
          </p>
          <div class="connection-input-row" style="margin-bottom:4px">
            <label style="font-size:0.8rem;display:flex;align-items:center;gap:4px;cursor:pointer">
              <input type="radio" name="mcp-type" value="stdio" checked onchange="Settings.mcpToggleType()"> Command (stdio)
            </label>
            <label style="font-size:0.8rem;display:flex;align-items:center;gap:4px;cursor:pointer">
              <input type="radio" name="mcp-type" value="http" onchange="Settings.mcpToggleType()"> HTTP URL
            </label>
          </div>
          <input type="text" id="mcp-add-name" placeholder="Server name (e.g. my-server)" autocomplete="off">
          <div id="mcp-type-fields">
            <input type="text" id="mcp-add-command" placeholder="Command (e.g. npx -y @modelcontextprotocol/server-filesystem)" autocomplete="off">
            <input type="text" id="mcp-add-args" placeholder="Arguments, comma-separated (optional)" autocomplete="off">
            <input type="text" id="mcp-add-env" placeholder="Environment: KEY=VALUE, KEY2=VALUE2 (optional)" autocomplete="off">
          </div>
          <div class="connection-input-row">
            <button class="btn btn-primary btn-sm" onclick="Settings.mcpSaveAdd()">Add & Connect</button>
            <button class="btn btn-ghost btn-sm" onclick="Settings.refresh()">Cancel</button>
          </div>
          <div id="mcp-add-status" style="font-size:0.8rem;min-height:1.2em"></div>
        </div>
      `;
      document.getElementById('mcp-add-name').focus();
    },

    mcpToggleType() {
      const isHttp = document.querySelector('input[name="mcp-type"]:checked')?.value === 'http';
      const fields = document.getElementById('mcp-type-fields');
      if (!fields) return;
      if (isHttp) {
        fields.innerHTML = `<input type="text" id="mcp-add-url" placeholder="Server URL (e.g. https://api.example.com/mcp)" autocomplete="off">`;
      } else {
        fields.innerHTML = `
          <input type="text" id="mcp-add-command" placeholder="Command (e.g. npx -y @modelcontextprotocol/server-filesystem)" autocomplete="off">
          <input type="text" id="mcp-add-args" placeholder="Arguments, comma-separated (optional)" autocomplete="off">
          <input type="text" id="mcp-add-env" placeholder="Environment: KEY=VALUE, KEY2=VALUE2 (optional)" autocomplete="off">
        `;
      }
    },

    async mcpSaveAdd() {
      const name = document.getElementById('mcp-add-name')?.value.trim();
      const statusEl = document.getElementById('mcp-add-status');
      if (!name) { statusEl.innerHTML = '<span style="color:var(--danger)">Name is required</span>'; return; }

      const isHttp = document.querySelector('input[name="mcp-type"]:checked')?.value === 'http';
      let body;

      if (isHttp) {
        const url = document.getElementById('mcp-add-url')?.value.trim();
        if (!url) { statusEl.innerHTML = '<span style="color:var(--danger)">URL is required</span>'; return; }
        body = { name, url };
      } else {
        const command = document.getElementById('mcp-add-command')?.value.trim();
        if (!command) { statusEl.innerHTML = '<span style="color:var(--danger)">Command is required</span>'; return; }
        const argsStr = document.getElementById('mcp-add-args')?.value.trim();
        const envStr = document.getElementById('mcp-add-env')?.value.trim();
        const args = argsStr ? argsStr.split(',').map(s => s.trim()).filter(Boolean) : [];
        const env = {};
        if (envStr) {
          for (const pair of envStr.split(',')) {
            const [k, ...v] = pair.split('=');
            if (k?.trim()) env[k.trim()] = v.join('=').trim();
          }
        }
        body = { name, command, args, env };
      }

      statusEl.innerHTML = '<span style="color:var(--text-muted)">Connecting...</span>';

      try {
        const res = await fetch('/api/mcp/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data.error && !data.added) {
          statusEl.innerHTML = `<span style="color:var(--danger)">${escapeHtml(data.error)}</span>`;
        } else if (data.connected) {
          statusEl.innerHTML = `<span style="color:var(--success)">Connected (${data.tools} tools)</span>`;
          setTimeout(() => this.refresh(), 1000);
        } else {
          statusEl.innerHTML = `<span style="color:var(--warning)">Added but failed to connect: ${escapeHtml(data.error || 'unknown error')}</span>`;
          setTimeout(() => this.refresh(), 2000);
        }
      } catch (err) {
        statusEl.innerHTML = `<span style="color:var(--danger)">Failed: ${escapeHtml(err.message)}</span>`;
      }
    },

    async mcpConnect(name) {
      const actionsEl = document.getElementById(`mcp-actions-${CSS.escape(name)}`);
      if (actionsEl) actionsEl.innerHTML = '<span style="font-size:0.8rem;color:var(--text-muted)">Connecting...</span>';
      try {
        await fetch('/api/mcp/connect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
      } catch {}
      this.refresh();
    },

    async mcpDisconnect(name) {
      try {
        await fetch('/api/mcp/disconnect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
      } catch {}
      this.refresh();
    },

    async mcpRemove(name) {
      if (!confirm(`Remove MCP server "${name}"?`)) return;
      try {
        await fetch(`/api/mcp/${encodeURIComponent(name)}`, { method: 'DELETE' });
      } catch {}
      this.refresh();
    },

    mcpShowTools(name) {
      const actionsEl = document.getElementById(`mcp-actions-${CSS.escape(name)}`);
      if (!actionsEl) return;
      // Fetch current status to get tool names
      fetch('/api/mcp').then(r => r.json()).then(status => {
        const srv = status[name];
        if (!srv || !srv.toolNames?.length) {
          actionsEl.innerHTML = '<span style="font-size:0.8rem;color:var(--text-muted)">No tools available</span>';
          return;
        }
        let html = '<div style="font-size:0.75rem;color:var(--text-muted);max-height:150px;overflow-y:auto;margin-bottom:6px">';
        for (const tool of srv.toolNames) {
          html += `<div style="padding:2px 0;font-family:monospace">${escapeHtml(tool)}</div>`;
        }
        html += '</div>';
        html += `<button class="btn btn-ghost btn-sm" onclick="Settings.refresh()">Close</button>`;
        actionsEl.innerHTML = html;
      }).catch(() => {});
    },

    async refresh() {
      openSettingsPanel();
    },
  };

  // ── Argus companion widget ──
  const argusReactions = [];
  const MAX_ARGUS_REACTIONS = 30;
  let argusExpanded = false;
  let argusEnabled = true;
  let argusName = 'Argus';

  function initArgus() {
    // Create widget DOM
    const widget = document.createElement('div');
    widget.id = 'argus-widget';
    widget.className = 'argus-widget';
    widget.innerHTML = `
      <div class="argus-panel" id="argus-panel">
        <div class="argus-panel-header">
          <div class="argus-panel-header-left">
            <span id="argus-panel-name">${escapeHtml(argusName)}</span>
          </div>
          <div class="argus-panel-toggle active" id="argus-toggle" title="Toggle Argus"></div>
        </div>
        <div class="argus-panel-body" id="argus-panel-body">
          <div class="argus-empty">Argus is watching and will share<br>thoughts as you work.</div>
        </div>
      </div>
      <div class="argus-fab" id="argus-fab" title="Argus companion">
        <span>✦</span>
        <div class="argus-badge"></div>
      </div>
    `;
    document.body.appendChild(widget);

    // Load config
    fetch('/api/argus').then(r => r.json()).then(cfg => {
      argusEnabled = cfg.enabled;
      argusName = cfg.name || 'Argus';
      const nameEl = document.getElementById('argus-panel-name');
      if (nameEl) nameEl.textContent = argusName;
      const toggle = document.getElementById('argus-toggle');
      if (toggle) toggle.classList.toggle('active', argusEnabled);
      if (!argusEnabled) widget.style.opacity = '0.5';
    }).catch(() => {});

    // FAB click → toggle panel
    document.getElementById('argus-fab').addEventListener('click', () => {
      argusExpanded = !argusExpanded;
      const panel = document.getElementById('argus-panel');
      panel.classList.toggle('argus-panel--open', argusExpanded);
      // Clear badge
      document.getElementById('argus-fab').classList.remove('argus-fab--active');
      if (argusExpanded) {
        const body = document.getElementById('argus-panel-body');
        body.scrollTop = body.scrollHeight;
      }
    });

    // Toggle switch
    document.getElementById('argus-toggle').addEventListener('click', (e) => {
      e.stopPropagation();
      argusEnabled = !argusEnabled;
      e.currentTarget.classList.toggle('active', argusEnabled);
      widget.style.opacity = argusEnabled ? '1' : '0.5';
      fetch('/api/argus', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: argusEnabled }),
      }).catch(() => {});
    });
  }

  function renderArgusReactions() {
    const body = document.getElementById('argus-panel-body');
    if (!body) return;
    if (argusReactions.length === 0) {
      body.innerHTML = '<div class="argus-empty">Argus is watching and will share<br>thoughts as you work.</div>';
      return;
    }
    body.innerHTML = argusReactions.map(r => {
      const time = new Date(r.timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
      const trigger = r.trigger || '';
      const tool = r.tool ? ` · ${r.tool}` : '';
      return `<div class="argus-reaction">
        <div class="argus-reaction-content">${escapeHtml(r.content)}</div>
        <div class="argus-reaction-meta">
          <span>${time}</span>
          <span class="argus-reaction-trigger">${escapeHtml(trigger)}${escapeHtml(tool)}</span>
        </div>
      </div>`;
    }).join('');
    body.scrollTop = body.scrollHeight;
  }

  window.Argus = {
    addReaction(msg) {
      argusReactions.push(msg);
      if (argusReactions.length > MAX_ARGUS_REACTIONS) argusReactions.shift();
      renderArgusReactions();
      // Pulse the FAB if panel is closed
      if (!argusExpanded) {
        document.getElementById('argus-fab')?.classList.add('argus-fab--active');
      }
    },
    clear() {
      argusReactions.length = 0;
      renderArgusReactions();
      document.getElementById('argus-fab')?.classList.remove('argus-fab--active');
    },
    toggle() {
      document.getElementById('argus-fab')?.click();
    },
    get reactions() { return argusReactions; },
  };

  // Init Argus on first load (DOM ready, don't need WS for setup)
  initArgus();
})();
