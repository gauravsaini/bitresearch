// Dashboard UI — connects to coordinator and renders real-time training metrics

interface DashboardState {
  connected: boolean;
  step: number;
  loss: number;
  bestLoss: number;
  numWorkers: number;
  tokensPerSec: number;
  totalTokens: number;
  elapsed: number;
  workers: any[];
  losses: number[];
  smoothLosses: number[];
  config: any;
}

const state: DashboardState = {
  connected: false,
  step: 0,
  loss: 0,
  bestLoss: Infinity,
  numWorkers: 0,
  tokensPerSec: 0,
  totalTokens: 0,
  elapsed: 0,
  workers: [],
  losses: [],
  smoothLosses: [],
  config: null,
};

// ── WebSocket Connection ──────────────────────────────────────

let ws: WebSocket | null = null;

function connect(): void {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${window.location.host}/ws?dashboard=true`;

  ws = new WebSocket(url);

  ws.onopen = () => {
    state.connected = true;
    updateConnectionStatus(true);
    addLog('Connected to coordinator', 'success');

    // Also fetch initial state via REST
    fetch('/api/status')
      .then(r => r.json())
      .then(data => {
        state.config = data.config;
        renderConfig();
      })
      .catch(() => {});
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    } catch (e) {
      console.error('Failed to parse message:', e);
    }
  };

  ws.onclose = () => {
    state.connected = false;
    updateConnectionStatus(false);
    addLog('Disconnected from coordinator. Reconnecting...', 'warning');
    setTimeout(connect, 3000);
  };

  ws.onerror = () => {
    addLog('WebSocket error', 'error');
  };
}

function handleMessage(msg: any): void {
  switch (msg.type) {
    case 0x20: // STATUS_UPDATE
      state.step = msg.step;
      state.loss = msg.loss;
      state.bestLoss = msg.bestLoss || state.bestLoss;
      state.numWorkers = msg.numWorkers;
      state.tokensPerSec = msg.tokensPerSec;
      state.totalTokens = msg.totalTokens || 0;
      state.elapsed = msg.elapsedSeconds;
      state.workers = msg.workers || [];
      if (msg.losses) state.losses = msg.losses;
      updateUI();
      break;

    case 0x21: // LOSS_UPDATE
      state.step = msg.step;
      state.loss = msg.smoothLoss || msg.loss;
      state.bestLoss = msg.bestLoss || state.bestLoss;
      state.tokensPerSec = msg.tokensPerSec;
      state.totalTokens = msg.totalTokens || state.totalTokens;
      state.elapsed = msg.elapsed;
      state.losses.push(msg.loss);
      state.smoothLosses.push(msg.smoothLoss || msg.loss);

      // Trim to last 500
      if (state.losses.length > 500) {
        state.losses = state.losses.slice(-500);
        state.smoothLosses = state.smoothLosses.slice(-500);
      }

      updateUI();
      drawLossChart();
      addLog(`Step ${msg.step} | loss=${msg.loss.toFixed(6)} | ${msg.tokensPerSec?.toFixed(0)} tok/s`, 'info');
      break;
  }
}

// ── UI Updates ────────────────────────────────────────────────

function updateUI(): void {
  // Stats
  setText('step-counter', state.step.toString());
  setText('worker-count', state.numWorkers.toString());
  setText('elapsed-time', formatElapsed(state.elapsed));
  setText('current-loss', state.loss > 0 ? state.loss.toFixed(6) : '—');
  setText('best-loss', state.bestLoss < Infinity ? state.bestLoss.toFixed(6) : '—');
  setText('tokens-per-sec', state.tokensPerSec > 0 ? formatNumber(Math.floor(state.tokensPerSec)) : '—');
  setText('total-tokens', formatNumber(state.totalTokens));

  // Loss trend
  if (state.losses.length > 1) {
    const recent = state.losses.slice(-10);
    const prev = state.losses.slice(-20, -10);
    if (prev.length > 0) {
      const avgRecent = recent.reduce((a, b) => a + b, 0) / recent.length;
      const avgPrev = prev.reduce((a, b) => a + b, 0) / prev.length;
      const diff = avgRecent - avgPrev;
      const trend = diff < 0 ? '↓ improving' : diff > 0 ? '↑ increasing' : '→ stable';
      const trendColor = diff < 0 ? 'color: var(--accent-green)' : diff > 0 ? 'color: var(--accent-red)' : '';
      const el = document.getElementById('loss-trend');
      if (el) {
        el.innerHTML = `<span style="${trendColor}">${trend}</span>`;
      }
    }
  }

  // Workers grid
  renderWorkers();
  drawWorkerChart();
}

function renderWorkers(): void {
  const grid = document.getElementById('workers-grid');
  if (!grid) return;

  if (state.workers.length === 0) {
    grid.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🖥️</div>
        <p>No workers connected yet</p>
        <p class="muted">Open <a href="/worker.html" target="_blank">worker.html</a> in a new tab to join</p>
      </div>
    `;
    return;
  }

  grid.innerHTML = state.workers.map(w => `
    <div class="worker-card-item">
      <div class="worker-state-dot ${w.state}"></div>
      <div class="worker-id">${w.id.slice(-8)}</div>
      <div class="worker-device">${w.device || 'Unknown GPU'}</div>
      <div class="worker-stats">
        <div class="worker-stat-mini">
          <span class="label">State</span>
          <span class="value">${w.state}</span>
        </div>
        <div class="worker-stat-mini">
          <span class="label">Steps</span>
          <span class="value">${w.stepsCompleted || 0}</span>
        </div>
        <div class="worker-stat-mini">
          <span class="label">Avg Time</span>
          <span class="value">${(w.avgComputeTimeMs || 0).toFixed(0)}ms</span>
        </div>
        <div class="worker-stat-mini">
          <span class="label">Joined</span>
          <span class="value">${formatTimeAgo(w.joinedAt)}</span>
        </div>
      </div>
    </div>
  `).join('');
}

function renderConfig(): void {
  const grid = document.getElementById('config-grid');
  if (!grid || !state.config) return;

  const items = [
    { label: 'Layers', value: state.config.nLayer },
    { label: 'Embed Dim', value: state.config.nEmbd },
    { label: 'Heads', value: state.config.nHead },
    { label: 'KV Heads', value: state.config.nKvHead },
    { label: 'Seq Length', value: state.config.sequenceLen },
    { label: 'Vocab Size', value: formatNumber(state.config.vocabSize) },
    { label: 'Window', value: state.config.windowPattern },
  ];

  grid.innerHTML = items.map(i => `
    <div class="config-item">
      <div class="config-label">${i.label}</div>
      <div class="config-value">${i.value}</div>
    </div>
  `).join('');
}

function updateConnectionStatus(connected: boolean): void {
  const el = document.getElementById('connection-status');
  if (!el) return;
  el.className = `status-badge ${connected ? 'connected' : ''}`;
  el.innerHTML = `
    <span class="status-dot"></span>
    <span>${connected ? 'Connected' : 'Connecting...'}</span>
  `;
}

// ── Charts ────────────────────────────────────────────────────

function drawLossChart(): void {
  const canvas = document.getElementById('loss-chart') as HTMLCanvasElement;
  if (!canvas || state.losses.length < 2) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const padding = { top: 10, right: 10, bottom: 30, left: 60 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;

  ctx.clearRect(0, 0, w, h);

  const losses = state.losses;
  const smoothed = state.smoothLosses.length > 0 ? state.smoothLosses : losses;
  const allVals = [...losses, ...smoothed].filter(v => v > 0 && isFinite(v));
  if (allVals.length === 0) return;

  const minVal = Math.min(...allVals) * 0.98;
  const maxVal = Math.max(...allVals) * 1.02;
  const range = maxVal - minVal || 1;

  const toX = (i: number) => padding.left + (i / (losses.length - 1)) * plotW;
  const toY = (v: number) => padding.top + (1 - (v - minVal) / range) * plotH;

  // Grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();

    // Y-axis labels
    const val = maxVal - (i / 4) * range;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.font = '10px "JetBrains Mono"';
    ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(3), padding.left - 8, y + 3);
  }

  // X-axis labels
  ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.textAlign = 'center';
  ctx.fillText('0', padding.left, h - 5);
  ctx.fillText(losses.length.toString(), w - padding.right, h - 5);

  // Raw loss line
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(79, 140, 255, 0.3)';
  ctx.lineWidth = 1;
  for (let i = 0; i < losses.length; i++) {
    const x = toX(i);
    const y = toY(losses[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Smoothed loss line
  if (smoothed.length > 0) {
    ctx.beginPath();
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 2;
    for (let i = 0; i < smoothed.length; i++) {
      const x = toX(i);
      const y = toY(smoothed[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Gradient fill under smoothed line
    const gradient = ctx.createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, 'rgba(52, 211, 153, 0.08)');
    gradient.addColorStop(1, 'rgba(52, 211, 153, 0)');
    ctx.lineTo(toX(smoothed.length - 1), h - padding.bottom);
    ctx.lineTo(toX(0), h - padding.bottom);
    ctx.fillStyle = gradient;
    ctx.fill();
  }
}

function drawWorkerChart(): void {
  const canvas = document.getElementById('worker-chart') as HTMLCanvasElement;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  if (state.workers.length === 0) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.font = '13px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('No workers', w / 2, h / 2);
    return;
  }

  // Bar chart of workers by state
  const stateColors: Record<string, string> = {
    idle: '#4f8cff',
    training: '#34d399',
    syncing: '#a78bfa',
    disconnected: '#565a6e',
  };

  const barHeight = Math.min(40, (h - 40) / state.workers.length);
  const gap = 6;
  const startY = 20;

  state.workers.forEach((worker, i) => {
    const y = startY + i * (barHeight + gap);
    const barWidth = w - 100;

    // Background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.beginPath();
    ctx.roundRect(80, y, barWidth, barHeight, 4);
    ctx.fill();

    // Fill based on steps
    const maxSteps = Math.max(...state.workers.map((w: any) => w.stepsCompleted || 1));
    const fillRatio = (worker.stepsCompleted || 0) / maxSteps;

    const gradient = ctx.createLinearGradient(80, 0, 80 + barWidth * fillRatio, 0);
    gradient.addColorStop(0, stateColors[worker.state] || '#4f8cff');
    gradient.addColorStop(1, stateColors[worker.state] + '80' || '#4f8cff80');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.roundRect(80, y, barWidth * fillRatio, barHeight, 4);
    ctx.fill();

    // Label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '10px "JetBrains Mono"';
    ctx.textAlign = 'right';
    ctx.fillText(worker.id.slice(-6), 72, y + barHeight / 2 + 3);

    // Steps count
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.textAlign = 'left';
    ctx.fillText(`${worker.stepsCompleted || 0}`, 85 + barWidth * fillRatio + 4, y + barHeight / 2 + 3);
  });
}

// ── Logging ───────────────────────────────────────────────────

function addLog(msg: string, type: string = ''): void {
  const container = document.getElementById('log-container');
  if (!container) return;

  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  const time = new Date().toLocaleTimeString();
  entry.textContent = `[${time}] ${msg}`;

  container.appendChild(entry);
  container.scrollTop = container.scrollHeight;

  // Trim to 200 entries
  while (container.children.length > 200) {
    container.removeChild(container.firstChild!);
  }
}

// ── Utilities ────────────────────────────────────────────────

function setText(id: string, text: string): void {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatNumber(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

function formatTimeAgo(ts: number): string {
  if (!ts) return '—';
  const seconds = Math.floor((Date.now() - ts) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

// ── Events ────────────────────────────────────────────────────

document.getElementById('btn-add-worker')?.addEventListener('click', () => {
  window.open('/worker.html', '_blank');
});

document.getElementById('btn-clear-log')?.addEventListener('click', () => {
  const container = document.getElementById('log-container');
  if (container) container.innerHTML = '';
});

// Handle canvas resize
window.addEventListener('resize', () => {
  drawLossChart();
  drawWorkerChart();
});

// ── Init ──────────────────────────────────────────────────────

addLog('Initializing dashboard...', 'info');
connect();

// Periodic UI refresh
setInterval(() => {
  if (state.connected) {
    state.elapsed += 1;
    setText('elapsed-time', formatElapsed(state.elapsed));
  }
}, 1000);
