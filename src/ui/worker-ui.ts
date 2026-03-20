// Worker UI — initializes WebGPU worker and connects to coordinator
import { WorkerClient } from '../distributed/worker-client';

const worker = new WorkerClient();
const activityHistory: number[] = [];

// ── UI References ────────────────────────────────────────────

const statusBadge = document.querySelector('#worker-status .status-badge') as HTMLElement;
const gpuInfo = document.getElementById('gpu-info') as HTMLElement;
const stepsCompleted = document.getElementById('steps-completed') as HTMLElement;
const lastLoss = document.getElementById('last-loss') as HTMLElement;
const avgSpeed = document.getElementById('avg-speed') as HTMLElement;
const logContainer = document.getElementById('worker-log') as HTMLElement;
const activityBars = document.getElementById('activity-bars') as HTMLElement;
const pulseDot = document.querySelector('.pulse-dot') as HTMLElement;

// ── State Change Handler ─────────────────────────────────────

worker.onStateChange = (state) => {
  if (!statusBadge) return;
  statusBadge.className = `status-badge ${state}`;

  const labels: Record<string, string> = {
    connecting: '🔄 Connecting...',
    idle: '💤 Idle',
    training: '🔥 Training',
    syncing: '📡 Syncing',
    error: '❌ Error',
    disconnected: '🔌 Disconnected',
  };

  statusBadge.textContent = labels[state] || state;

  // Update pulse dot
  if (pulseDot) {
    if (state === 'training') {
      pulseDot.style.background = '#34d399';
      pulseDot.style.boxShadow = '0 0 12px rgba(52, 211, 153, 0.4)';
    } else if (state === 'error' || state === 'disconnected') {
      pulseDot.style.background = '#f87171';
      pulseDot.style.boxShadow = '0 0 12px rgba(248, 113, 113, 0.4)';
    } else {
      pulseDot.style.background = '#4f8cff';
      pulseDot.style.boxShadow = '0 0 12px rgba(79, 140, 255, 0.4)';
    }
  }
};

// ── Metrics Update Handler ───────────────────────────────────

worker.onMetricsUpdate = (metrics) => {
  if (gpuInfo) gpuInfo.textContent = metrics.gpuInfo;
  if (stepsCompleted) stepsCompleted.textContent = metrics.stepsCompleted.toString();
  if (lastLoss) lastLoss.textContent = metrics.lastLoss > 0 ? metrics.lastLoss.toFixed(6) : '—';
  if (avgSpeed) avgSpeed.textContent = metrics.avgComputeTimeMs > 0
    ? `${metrics.avgComputeTimeMs.toFixed(0)} ms/step`
    : '— ms/step';

  // Track activity for visualization
  activityHistory.push(metrics.state === 'training' ? 100 : metrics.state === 'idle' ? 20 : 0);
  if (activityHistory.length > 30) activityHistory.shift();
  renderActivityBars();
};

// ── Log Handler ──────────────────────────────────────────────

worker.onLog = (msg) => {
  if (!logContainer) return;

  const entry = document.createElement('div');
  entry.className = 'log-entry';
  const time = new Date().toLocaleTimeString();
  entry.textContent = `[${time}] ${msg}`;

  logContainer.appendChild(entry);
  logContainer.scrollTop = logContainer.scrollHeight;

  // Trim
  while (logContainer.children.length > 100) {
    logContainer.removeChild(logContainer.firstChild!);
  }
};

// ── Activity Bars ────────────────────────────────────────────

function renderActivityBars(): void {
  if (!activityBars) return;

  activityBars.innerHTML = activityHistory.map((val, i) => `
    <div class="activity-bar" style="opacity: ${0.4 + (i / activityHistory.length) * 0.6}">
      <div class="fill" style="width: ${val}%"></div>
      <span class="label">${val > 50 ? '🔥' : val > 0 ? '💤' : '⏸️'} ${val}%</span>
    </div>
  `).join('');
}

// Initialize with empty bars
for (let i = 0; i < 10; i++) activityHistory.push(0);
renderActivityBars();

// ── Start Worker ─────────────────────────────────────────────

const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${protocol}//${window.location.host}/ws`;

worker.start(wsUrl);

// ── Cleanup ──────────────────────────────────────────────────

window.addEventListener('beforeunload', () => {
  worker.stop();
});
