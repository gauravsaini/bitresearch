// Coordinator Server — Parameter Server + Gradient Aggregator + Worker Manager
import express from 'express';
import { WebSocketServer, WebSocket } from 'ws';
import cors from 'cors';
import http from 'http';
import path from 'path';

// ── Types ─────────────────────────────────────────────────────────────

interface WorkerInfo {
  id: string;
  ws: WebSocket;
  state: 'idle' | 'training' | 'syncing' | 'disconnected';
  deviceInfo: any;
  lastSeen: number;
  stepsCompleted: number;
  totalComputeMs: number;
  joinedAt: number;
}

interface TrainingState {
  step: number;
  totalTokens: number;
  startTime: number;
  losses: number[];
  smoothLoss: number;
  bestLoss: number;
}

interface ModelConfig {
  sequenceLen: number;
  vocabSize: number;
  nLayer: number;
  nHead: number;
  nKvHead: number;
  nEmbd: number;
  windowPattern: string;
}

// ── Configuration ─────────────────────────────────────────────────────

const PORT = parseInt(process.env.PORT || '8787');
const BATCH_SIZE = parseInt(process.env.BATCH_SIZE || '4');

// Smaller model for WebGPU
const MODEL_CONFIG: ModelConfig = {
  sequenceLen: 512,
  vocabSize: 8192,
  nLayer: 4,
  nHead: 4,
  nKvHead: 4,
  nEmbd: 256,
  windowPattern: 'L',
};

// ── Server Setup ──────────────────────────────────────────────────────

const app = express();
app.use(cors());
app.use(express.json());
app.use('/data', express.static(path.resolve('public', 'data')));

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

// ── State ─────────────────────────────────────────────────────────────

const workers = new Map<string, WorkerInfo>();
const dashboardClients = new Set<WebSocket>();

const trainingState: TrainingState = {
  step: 0,
  totalTokens: 0,
  startTime: Date.now(),
  losses: [],
  smoothLoss: 0,
  bestLoss: Infinity,
};

// Pending gradients for current step
const pendingGradients = new Map<string, { loss: number; tokensProcessed: number; computeTimeMs: number }>();

// ── Data Generation (synthetic for now) ───────────────────────────────

function generateSyntheticBatch(B: number, T: number, vocabSize: number): { inputIds: number[]; targets: number[] } {
  const inputIds: number[] = [];
  const targets: number[] = [];

  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const tokenId = Math.floor(Math.random() * vocabSize);
      inputIds.push(tokenId);
      // Target is next token (shifted by 1) — for synthetic data, just use random
      targets.push(Math.floor(Math.random() * vocabSize));
    }
  }

  return { inputIds, targets };
}

// ── WebSocket Handlers ────────────────────────────────────────────────

wss.on('connection', (ws: WebSocket, req) => {
  const isDashboard = req.url?.includes('dashboard');

  if (isDashboard) {
    dashboardClients.add(ws);
    console.log(`📊 Dashboard connected (${dashboardClients.size} active)`);

    // Send current state
    sendDashboardUpdate();

    ws.on('close', () => {
      dashboardClients.delete(ws);
      console.log(`📊 Dashboard disconnected (${dashboardClients.size} active)`);
    });
    return;
  }

  // Worker connection
  console.log(`🔌 New connection from ${req.socket.remoteAddress}`);

  ws.on('message', (data: Buffer) => {
    try {
      const msg = JSON.parse(data.toString());
      handleWorkerMessage(ws, msg);
    } catch (e) {
      console.error('Failed to parse worker message:', e);
    }
  });

  ws.on('close', () => {
    // Find and remove worker
    for (const [id, worker] of workers) {
      if (worker.ws === ws) {
        worker.state = 'disconnected';
        console.log(`👋 Worker ${id.slice(-6)} disconnected (${workers.size - 1} remaining)`);
        workers.delete(id);
        sendDashboardUpdate();
        break;
      }
    }
  });

  ws.on('error', (e) => {
    console.error('Worker WebSocket error:', e);
  });
});

function handleWorkerMessage(ws: WebSocket, msg: any): void {
  switch (msg.type) {
    case 0x01: // WORKER_JOIN
      handleWorkerJoin(ws, msg);
      break;
    case 0x02: // WORKER_HEARTBEAT
      handleHeartbeat(msg);
      break;
    case 0x03: // GRADIENT_PUSH
      handleGradientPush(msg);
      break;
    case 0x04: // WORKER_LEAVE
      handleWorkerLeave(msg);
      break;
  }
}

function handleWorkerJoin(ws: WebSocket, msg: any): void {
  const worker: WorkerInfo = {
    id: msg.workerId,
    ws,
    state: 'idle',
    deviceInfo: msg.deviceInfo,
    lastSeen: Date.now(),
    stepsCompleted: 0,
    totalComputeMs: 0,
    joinedAt: Date.now(),
  };

  workers.set(msg.workerId, worker);
  console.log(`✅ Worker ${msg.workerId.slice(-6)} joined | GPU: ${msg.deviceInfo?.vendor || 'unknown'} ${msg.deviceInfo?.architecture || ''} | Total workers: ${workers.size}`);

  // Send welcome with model config
  sendToWorker(ws, {
    type: 0x10, // WELCOME
    workerId: msg.workerId,
    config: MODEL_CONFIG,
    batchSize: BATCH_SIZE,
    totalWorkers: workers.size,
  });

  sendDashboardUpdate();

  // If we have enough workers, start training
  if (workers.size >= 1) {
    setTimeout(() => startTrainingRound(), 2000);
  }
}

function handleHeartbeat(msg: any): void {
  const worker = workers.get(msg.workerId);
  if (worker) {
    worker.lastSeen = Date.now();
  }
}

function handleGradientPush(msg: any): void {
  const worker = workers.get(msg.workerId);
  if (!worker) return;

  worker.stepsCompleted++;
  worker.totalComputeMs += msg.computeTimeMs || 0;
  worker.state = 'idle';

  // Store gradient info
  pendingGradients.set(msg.workerId, {
    loss: msg.loss,
    tokensProcessed: msg.tokensProcessed || 0,
    computeTimeMs: msg.computeTimeMs || 0,
  });

  console.log(`📥 Gradient from ${msg.workerId.slice(-6)} | step=${msg.stepId} loss=${msg.loss?.toFixed(6)} | ${msg.computeTimeMs?.toFixed(0)}ms`);

  // Check if all active workers have reported
  const activeWorkers = Array.from(workers.values()).filter(w => w.state !== 'disconnected');
  if (pendingGradients.size >= activeWorkers.length) {
    completeStep();
  }
}

function handleWorkerLeave(msg: any): void {
  workers.delete(msg.workerId);
  console.log(`👋 Worker ${msg.workerId.slice(-6)} left gracefully`);
  sendDashboardUpdate();
}

// ── Training Orchestration ────────────────────────────────────────────

function startTrainingRound(): void {
  const activeWorkers = Array.from(workers.values()).filter(w => w.state === 'idle');
  if (activeWorkers.length === 0) {
    console.log('⏳ No idle workers, waiting...');
    return;
  }

  trainingState.step++;
  pendingGradients.clear();

  console.log(`\n🚀 Step ${trainingState.step} | ${activeWorkers.length} workers`);

  // Generate and distribute batches
  for (const worker of activeWorkers) {
    const batch = generateSyntheticBatch(BATCH_SIZE, MODEL_CONFIG.sequenceLen, MODEL_CONFIG.vocabSize);

    worker.state = 'training';

    sendToWorker(worker.ws, {
      type: 0x12, // BATCH_ASSIGN
      stepId: trainingState.step,
      inputIds: batch.inputIds,
      targets: batch.targets,
    });
  }

  sendDashboardUpdate();

  // Timeout: if workers don't respond in 30s, complete step anyway
  setTimeout(() => {
    if (pendingGradients.size < activeWorkers.length) {
      console.log(`⚠️ Timeout: ${pendingGradients.size}/${activeWorkers.length} workers responded`);
      completeStep();
    }
  }, 30000);
}

function completeStep(): void {
  // Average losses
  let totalLoss = 0;
  let totalTokens = 0;
  let totalComputeMs = 0;

  for (const grad of pendingGradients.values()) {
    totalLoss += grad.loss;
    totalTokens += grad.tokensProcessed;
    totalComputeMs += grad.computeTimeMs;
  }

  const avgLoss = totalLoss / pendingGradients.size;
  const avgComputeMs = totalComputeMs / pendingGradients.size;

  // EMA smoothing
  const beta = 0.9;
  trainingState.smoothLoss = trainingState.smoothLoss === 0
    ? avgLoss
    : beta * trainingState.smoothLoss + (1 - beta) * avgLoss;

  // Debias
  const debiased = trainingState.smoothLoss / (1 - Math.pow(beta, trainingState.step));

  trainingState.totalTokens += totalTokens;
  trainingState.losses.push(avgLoss);

  if (avgLoss < trainingState.bestLoss) {
    trainingState.bestLoss = avgLoss;
  }

  const elapsed = (Date.now() - trainingState.startTime) / 1000;
  const tokPerSec = totalTokens > 0 ? totalTokens / (avgComputeMs / 1000) : 0;

  console.log(`✅ Step ${trainingState.step} complete | loss=${debiased.toFixed(6)} | ${tokPerSec.toFixed(0)} tok/s | ${elapsed.toFixed(0)}s elapsed`);

  // Notify dashboard
  broadcastToDashboard({
    type: 0x21, // LOSS_UPDATE
    step: trainingState.step,
    loss: avgLoss,
    smoothLoss: debiased,
    bestLoss: trainingState.bestLoss,
    numWorkers: pendingGradients.size,
    tokensPerSec: tokPerSec,
    totalTokens: trainingState.totalTokens,
    elapsed,
  });

  sendDashboardUpdate();
  pendingGradients.clear();

  // Start next round after a brief delay
  setTimeout(() => startTrainingRound(), 100);
}

// ── Communication Helpers ─────────────────────────────────────────────

function sendToWorker(ws: WebSocket, msg: any): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

function broadcastToDashboard(msg: any): void {
  const data = JSON.stringify(msg);
  for (const client of dashboardClients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(data);
    }
  }
}

function sendDashboardUpdate(): void {
  const workerList = Array.from(workers.values()).map(w => ({
    id: w.id,
    state: w.state,
    device: `${w.deviceInfo?.vendor || '?'} ${w.deviceInfo?.architecture || ''}`,
    lastSeen: w.lastSeen,
    stepsCompleted: w.stepsCompleted,
    avgComputeTimeMs: w.stepsCompleted > 0 ? w.totalComputeMs / w.stepsCompleted : 0,
    joinedAt: w.joinedAt,
  }));

  const elapsed = (Date.now() - trainingState.startTime) / 1000;

  broadcastToDashboard({
    type: 0x20, // STATUS_UPDATE
    step: trainingState.step,
    loss: trainingState.smoothLoss,
    bestLoss: trainingState.bestLoss,
    numWorkers: workers.size,
    tokensPerSec: trainingState.totalTokens / Math.max(elapsed, 1),
    elapsedSeconds: elapsed,
    totalTokens: trainingState.totalTokens,
    workers: workerList,
    losses: trainingState.losses.slice(-100), // last 100 losses
  });
}

// ── REST API ──────────────────────────────────────────────────────────

app.get('/api/status', (_req, res) => {
  const elapsed = (Date.now() - trainingState.startTime) / 1000;
  res.json({
    step: trainingState.step,
    loss: trainingState.smoothLoss,
    bestLoss: trainingState.bestLoss,
    numWorkers: workers.size,
    tokensPerSec: trainingState.totalTokens / Math.max(elapsed, 1),
    totalTokens: trainingState.totalTokens,
    elapsedSeconds: elapsed,
    workers: Array.from(workers.values()).map(w => ({
      id: w.id,
      state: w.state,
      device: `${w.deviceInfo?.vendor || '?'} ${w.deviceInfo?.architecture || ''}`,
      stepsCompleted: w.stepsCompleted,
    })),
    config: MODEL_CONFIG,
  });
});

app.get('/api/losses', (_req, res) => {
  res.json({ losses: trainingState.losses });
});

// ── Start ─────────────────────────────────────────────────────────────

server.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║       ⚡ WebGPU Distributed Training Coordinator ⚡         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Dashboard:  http://localhost:${PORT}                          ║
║  Worker WS:  ws://localhost:${PORT}/ws                         ║
║  API:        http://localhost:${PORT}/api/status                ║
║  Data:       http://localhost:${PORT}/data/tokens.bin          ║
║  Parity:     http://localhost:${PORT}/data/parity/manifest.json ║
║                                                              ║
║  Model: ${MODEL_CONFIG.nLayer}L / ${MODEL_CONFIG.nEmbd}D / ${MODEL_CONFIG.nHead}H              ║
║  Vocab: ${MODEL_CONFIG.vocabSize}  Seq: ${MODEL_CONFIG.sequenceLen}  Batch: ${BATCH_SIZE}                  ║
║                                                              ║
║  Waiting for workers to connect...                           ║
╚══════════════════════════════════════════════════════════════╝
  `);
});

// ── Cleanup ───────────────────────────────────────────────────────────

process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down...');
  for (const worker of workers.values()) {
    sendToWorker(worker.ws, { type: 0x16 }); // KICK
  }
  server.close();
  process.exit(0);
});
