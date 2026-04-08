// PRE Web GUI — WebSocket client with reconnect

const WS = (() => {
  let ws = null;
  let reconnectTimer = null;
  let reconnectDelay = 1000;
  let handlers = {};

  function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}`);

    ws.onopen = () => {
      reconnectDelay = 1000;
      updateStatus('connected');
      if (handlers.open) handlers.open();
    };

    ws.onmessage = (event) => {
      let msg;
      try { msg = JSON.parse(event.data); } catch { return; }
      if (handlers.message) handlers.message(msg);
    };

    ws.onclose = () => {
      updateStatus('disconnected');
      scheduleReconnect();
    };

    ws.onerror = () => {
      updateStatus('error');
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 10000);
  }

  function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
      return true;
    }
    return false;
  }

  function on(event, handler) {
    handlers[event] = handler;
  }

  function updateStatus(state) {
    const dot = document.querySelector('.status-dot');
    const text = document.getElementById('status-text');
    if (!dot || !text) return;

    dot.className = 'status-dot';
    switch (state) {
      case 'connected':
        dot.classList.add('connected');
        text.textContent = 'Connected';
        break;
      case 'disconnected':
        text.textContent = 'Reconnecting...';
        break;
      case 'error':
        dot.classList.add('error');
        text.textContent = 'Connection error';
        break;
    }
  }

  return { connect, send, on };
})();
