(() => {
  const connDot = document.getElementById('conn-dot');
  const projName = document.getElementById('proj-name');
  const phaseBadge = document.getElementById('phase-badge');
  const transcriptPane = document.getElementById('transcript-pane');
  const cardStream = document.getElementById('card-stream');
  const queryInput = document.getElementById('query-input');
  const querySend = document.getElementById('query-send');

  let ws = null;
  let backoffMs = 1000;
  const BACKOFF_MIN = 1000;
  const BACKOFF_MAX = 10000;

  function hueFor(userId) {
    let h = 0;
    for (let i = 0; i < userId.length; i++) {
      h = ((h * 31) + userId.charCodeAt(i)) | 0;
    }
    return ((h % 360) + 360) % 360;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => {
      switch (c) {
        case '&': return '&amp;';
        case '<': return '&lt;';
        case '>': return '&gt;';
        case '"': return '&quot;';
        case "'": return '&#39;';
        default: return c;
      }
    });
  }

  function renderInline(escaped) {
    let s = escaped.replace(/`([^`\n]+)`/g, (m, code) => '<code>' + code + '</code>');
    s = s.replace(/\*\*([^*\n]+)\*\*/g, (m, txt) => '<strong>' + txt + '</strong>');
    return s;
  }

  function isSeparatorRow(row) {
    const t = row.trim();
    if (!/\|/.test(t)) return false;
    const body = t.replace(/^\|/, '').replace(/\|$/, '');
    const cells = body.split('|');
    if (cells.length === 0) return false;
    return cells.every((c) => /^[\s\-:]+$/.test(c) && /-/.test(c));
  }

  function splitCells(row) {
    const t = row.trim().replace(/^\|/, '').replace(/\|$/, '');
    return t.split('|').map((c) => c.trim());
  }

  function renderMarkdown(md) {
    const escaped = escapeHtml(md == null ? '' : md);
    const lines = escaped.split('\n');
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];

      if (/^\s*\|.*\|\s*$/.test(line)) {
        const block = [];
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
          block.push(lines[i]);
          i++;
        }
        if (block.length >= 2) {
          const headerCells = splitCells(block[0]);
          let bodyStart = 1;
          if (block.length >= 3 && isSeparatorRow(block[1])) {
            bodyStart = 2;
          }
          const bodyRows = block.slice(bodyStart);
          const thead = '<thead><tr>' + headerCells.map((c) => '<th>' + renderInline(c) + '</th>').join('') + '</tr></thead>';
          const tbody = '<tbody>' + bodyRows.map((r) => {
            const cells = splitCells(r);
            return '<tr>' + cells.map((c) => '<td>' + renderInline(c) + '</td>').join('') + '</tr>';
          }).join('') + '</tbody>';
          out.push('<table>' + thead + tbody + '</table>');
        } else {
          for (const r of block) out.push(renderInline(r));
        }
        continue;
      }

      if (/^###\s+/.test(line)) {
        out.push('<h3>' + renderInline(line.replace(/^###\s+/, '')) + '</h3>');
        i++;
        continue;
      }
      if (/^##\s+/.test(line)) {
        out.push('<h2>' + renderInline(line.replace(/^##\s+/, '')) + '</h2>');
        i++;
        continue;
      }
      if (/^#\s+/.test(line)) {
        out.push('<h1>' + renderInline(line.replace(/^#\s+/, '')) + '</h1>');
        i++;
        continue;
      }

      if (/^-\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^-\s+/.test(lines[i])) {
          items.push('<li>' + renderInline(lines[i].replace(/^-\s+/, '')) + '</li>');
          i++;
        }
        out.push('<ul>' + items.join('') + '</ul>');
        continue;
      }

      out.push(renderInline(line));
      i++;
    }
    return out.join('\n');
  }

  function addTranscript(userId, text) {
    const div = document.createElement('div');
    div.setAttribute('data-testid', 'transcript-line');
    div.className = 'transcript-line';

    const dot = document.createElement('span');
    dot.className = 'user-dot';
    const hue = hueFor(userId || '');
    dot.style.backgroundColor = 'hsl(' + hue + ', 70%, 60%)';
    div.appendChild(dot);

    const who = document.createElement('span');
    who.className = 'user-id';
    who.textContent = userId || '';
    div.appendChild(who);

    const txt = document.createElement('span');
    txt.className = 'transcript-text';
    txt.textContent = text || '';
    div.appendChild(txt);

    transcriptPane.appendChild(div);
    transcriptPane.scrollTop = transcriptPane.scrollHeight;
  }

  function addCard(card) {
    const article = document.createElement('article');
    article.setAttribute('data-testid', 'card');
    const kind = (card && card.kind) ? String(card.kind) : 'info';
    article.setAttribute('data-kind', kind);
    article.className = 'card kind-' + kind;

    const title = document.createElement('h2');
    title.className = 'card-title';
    title.textContent = (card && card.title) ? String(card.title) : '';
    article.appendChild(title);

    const body = document.createElement('div');
    body.className = 'card-body';
    const md = (card && card.body_md) ? String(card.body_md) : '';
    body.innerHTML = renderMarkdown(md);
    article.appendChild(body);

    cardStream.prepend(article);
  }

  function setPhase(text) {
    phaseBadge.textContent = text == null ? '' : String(text);
  }

  function setConnected(on) {
    if (on) {
      connDot.classList.add('on');
      connDot.classList.remove('off');
    } else {
      connDot.classList.remove('on');
      connDot.classList.add('off');
    }
  }

  function handleEvent(msg) {
    if (!msg || typeof msg !== 'object') return;
    switch (msg.type) {
      case 'transcript':
        addTranscript(msg.user_id || '', msg.text || '');
        break;
      case 'card':
        if (msg.card) addCard(msg.card);
        break;
      case 'status':
        setPhase(msg.state || msg.detail || '');
        if (msg.project && typeof msg.project === 'string') {
          projName.textContent = msg.project;
        }
        break;
      case 'init_progress':
        setPhase(msg.stage || '');
        break;
      case 'project':
        if (typeof msg.name === 'string') projName.textContent = msg.name;
        break;
      default:
        break;
    }
  }

  function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = proto + '//' + location.host + '/ws';
    try {
      ws = new WebSocket(url);
    } catch (e) {
      scheduleReconnect();
      return;
    }
    setConnected(false);
    ws.onopen = () => {
      setConnected(true);
      backoffMs = BACKOFF_MIN;
    };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      handleEvent(msg);
    };
    ws.onerror = () => {};
    ws.onclose = () => {
      setConnected(false);
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    const delay = backoffMs;
    backoffMs = Math.min(backoffMs * 2, BACKOFF_MAX);
    setTimeout(connect, delay);
  }

  async function sendQuery() {
    const text = queryInput.value;
    if (!text || !text.trim()) return;
    querySend.disabled = true;
    queryInput.disabled = true;
    const payload = text;
    queryInput.value = '';
    try {
      const resp = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: payload })
      });
      let data = null;
      try { data = await resp.json(); } catch (e) { data = null; }
      if (!resp.ok || (data && data.ok === false)) {
        setPhase((data && data.detail) ? 'query error: ' + data.detail : 'query error');
      }
    } catch (e) {
      setPhase('query error');
    } finally {
      querySend.disabled = false;
      queryInput.disabled = false;
      queryInput.focus();
    }
  }

  querySend.addEventListener('click', sendQuery);
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendQuery();
    }
  });

  fetch('/api/status')
    .then((r) => r.ok ? r.json() : null)
    .then((s) => {
      if (s && typeof s.project === 'string' && s.project) {
        projName.textContent = s.project;
      }
    })
    .catch(() => {});

  connect();

  const captureToggle = document.getElementById('capture-toggle');
  const captureStatus = document.getElementById('capture-status');
  let capturing = false;
  let audioCtx = null;
  let displayStream = null;
  let micStream = null;
  let wsAudio = null;
  let processor = null;

  function setCaptureUI(on, detail) {
    capturing = on;
    captureToggle.textContent = on ? '■ Stop' : '● Capture';
    captureToggle.classList.toggle('on', on);
    if (captureStatus) captureStatus.textContent = detail || (on ? 'capturing' : '');
  }

  function downsampleTo16k(float32, inRate) {
    if (inRate === 16000) {
      const out = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      return out.buffer;
    }
    const ratio = inRate / 16000;
    const outLen = Math.round(float32.length / ratio);
    const out = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const idx = i * ratio;
      const i0 = Math.floor(idx);
      const i1 = Math.min(i0 + 1, float32.length - 1);
      const frac = idx - i0;
      const s = float32[i0] * (1 - frac) + float32[i1] * frac;
      const clamped = Math.max(-1, Math.min(1, s));
      out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF;
    }
    return out.buffer;
  }

  async function startCapture() {
    if (!window.isSecureContext) {
      setCaptureUI(false, 'system audio requires https or http://localhost — use ssh -L 8760:localhost:8760, or enable server.https in config, or chrome://flags → Insecure origins treated as secure');
    }
    let sysAudioTracks = [];
    let displayStreamFailed = false;
    try {
      displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      sysAudioTracks = displayStream.getAudioTracks();
      if (sysAudioTracks.length === 0) {
        displayStreamFailed = true;
        setCaptureUI(false, 'no system audio — check "Share system audio" or use https/localhost; falling back to mic only');
        displayStream.getTracks().forEach((t) => t.stop());
        displayStream = null;
      }
    } catch (e) {
      const name = (e && e.name) || '';
      if (!window.isSecureContext || name === 'NotAllowedError') {
        setCaptureUI(false, 'system audio denied — requires https or http://localhost. Try ssh -L 8760:localhost:8760, or set server.https in config.yaml');
      } else {
        setCaptureUI(false, 'system audio denied');
      }
      displayStreamFailed = true;
      if (e && e.name === 'NotAllowedError' && !window.isSecureContext) {
        // allow mic-only fallback below
      } else if (e && e.name !== 'NotAllowedError') {
        return;
      }
      if (displayStream) { try { displayStream.getTracks().forEach((t) => t.stop()); } catch (e2) {} displayStream = null; }
      if (!confirm('System audio unavailable — continue with mic only?')) return;
    }
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      if (displayStream) displayStream.getTracks().forEach((t) => t.stop());
      displayStream = null;
      setCaptureUI(false, 'mic denied');
      return;
    }

    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioCtx();
    const inRate = audioCtx.sampleRate;

    const proto2 = location.protocol === 'https:' ? 'wss:' : 'ws:';
    wsAudio = new WebSocket(proto2 + '//' + location.host + '/ws/audio');
    wsAudio.binaryType = 'arraybuffer';
    await new Promise((resolve, reject) => {
      wsAudio.onopen = resolve;
      wsAudio.onerror = () => reject(new Error('audio ws failed'));
      setTimeout(() => reject(new Error('audio ws timeout')), 5000);
    }).catch((e) => {
      setCaptureUI(false, 'audio ws failed');
      throw e;
    });

    const sysSource = sysAudioTracks.length
      ? audioCtx.createMediaStreamSource(new MediaStream(sysAudioTracks))
      : null;
    const micSource = audioCtx.createMediaStreamSource(micStream);
    const mixGain = audioCtx.createGain();
    mixGain.gain.value = 1.0;
    if (sysSource) sysSource.connect(mixGain);
    micSource.connect(mixGain);

    const bufferSize = 4096;
    processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
    mixGain.connect(processor);
    processor.connect(audioCtx.destination);

    let pending = [];
    processor.onaudioprocess = (ev) => {
      const input = ev.inputBuffer.getChannelData(0);
      const chunk = new Float32Array(input.length);
      chunk.set(input);
      pending.push(chunk);
      const totalLen = pending.reduce((a, c) => a + c.length, 0);
      if (totalLen >= inRate * 0.1) {
        const merged = new Float32Array(totalLen);
        let off = 0;
        for (const c of pending) { merged.set(c, off); off += c.length; }
        pending = [];
        const pcmBuf = downsampleTo16k(merged, inRate);
        if (wsAudio && wsAudio.readyState === 1) wsAudio.send(pcmBuf);
      }
    };

    const stopOnEnded = () => { if (capturing) stopCapture(); };
    if (sysAudioTracks[0]) sysAudioTracks[0].onended = stopOnEnded;

    setCaptureUI(true, 'capturing — system + mic → 16kHz');
  }

  function stopCapture() {
    try { if (processor) { processor.disconnect(); processor.onaudioprocess = null; } } catch (e) {}
    processor = null;
    try { if (audioCtx) audioCtx.close(); } catch (e) {}
    audioCtx = null;
    try { if (displayStream) displayStream.getTracks().forEach((t) => t.stop()); } catch (e) {}
    displayStream = null;
    try { if (micStream) micStream.getTracks().forEach((t) => t.stop()); } catch (e) {}
    micStream = null;
    try { if (wsAudio) wsAudio.close(); } catch (e) {}
    wsAudio = null;
    setCaptureUI(false, '');
  }

  if (captureToggle) {
    captureToggle.addEventListener('click', async () => {
      if (capturing) stopCapture();
      else {
        captureToggle.disabled = true;
        try { await startCapture(); } catch (e) { stopCapture(); }
        captureToggle.disabled = false;
      }
    });
  }

  const dmInput = document.getElementById('dm-input');
  const dmDropdown = document.getElementById('dm-dropdown');
  let allMembers = [];
  let selectedDmId = null;

  async function fetchMembers() {
    try {
      const r = await fetch('/api/guild/members');
      const data = await r.json();
      if (data && data.ok && Array.isArray(data.members)) allMembers = data.members;
    } catch (e) {}
  }

  async function fetchCurrentDm() {
    try {
      const r = await fetch('/api/config/dm');
      const data = await r.json();
      if (data && data.dm_user_id) {
        selectedDmId = data.dm_user_id;
        const m = allMembers.find((x) => x.id === selectedDmId);
        if (m) dmInput.value = m.display_name || m.username;
        else dmInput.value = selectedDmId;
      }
    } catch (e) {}
  }

  function renderDropdown(filter) {
    if (!dmDropdown) return;
    const q = (filter || '').toLowerCase();
    const matches = q
      ? allMembers.filter((m) => (m.username + ' ' + m.display_name).toLowerCase().includes(q)).slice(0, 8)
      : allMembers.slice(0, 8);
    dmDropdown.innerHTML = '';
    if (matches.length === 0 || !q) { dmDropdown.classList.remove('open'); return; }
    for (const m of matches) {
      const div = document.createElement('div');
      div.textContent = (m.display_name || m.username) + ' ';
      const span = document.createElement('span');
      span.className = 'dm-username';
      span.textContent = '@' + m.username;
      div.appendChild(span);
      div.addEventListener('click', async () => {
        dmInput.value = m.display_name || m.username;
        dmDropdown.classList.remove('open');
        selectedDmId = m.id;
        await fetch('/api/config/dm', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dm_user_id: m.id }),
        });
      });
      dmDropdown.appendChild(div);
    }
    dmDropdown.classList.add('open');
  }

  if (dmInput) {
    fetchMembers().then(fetchCurrentDm);
    dmInput.addEventListener('input', () => renderDropdown(dmInput.value));
    dmInput.addEventListener('focus', () => renderDropdown(dmInput.value));
    dmInput.addEventListener('blur', () => setTimeout(() => dmDropdown && dmDropdown.classList.remove('open'), 200));
    dmInput.addEventListener('keydown', async (e) => {
      if (e.key === 'Enter') {
        const val = dmInput.value.trim();
        if (!val) {
          await fetch('/api/config/dm', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dm_user_id: null }) });
          selectedDmId = null;
          dmDropdown.classList.remove('open');
        } else if (!selectedDmId || val !== (allMembers.find((m) => m.id === selectedDmId) || {}).display_name) {
          await fetch('/api/config/dm', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dm_user_id: val }) });
        }
      }
    });
  }
})();
