// Distributed Trainer — runs the training loop on each WebRTC worker
// Coordinates with peers via WebRTC ring all-reduce for gradient sync.
// Now with MuonAdamW optimizer and Karpathy's exact schedules.

import * as tf from '@tensorflow/tfjs';
import { GPTModel } from '../model/gpt';
import { GPTConfig, DEFAULT_CONFIG } from '../model/config';
import { WebRTCMesh } from './webrtc-mesh';
import { RingAllReduce } from './ring-allreduce';
import { TokenDataLoader, evaluateBPB } from '../data/dataloader';
import { MuonAdamWOptimizer, createMuonAdamW } from '../train/muon';

export interface TrainerConfig {
  modelConfig: GPTConfig;
  batchSize: number;
  // MuonAdamW hyperparameters
  unembedding_lr: number;
  embedding_lr: number;
  matrix_lr: number;
  scalar_lr: number;
  weight_decay: number;
  adam_betas: [number, number];
  // Schedule
  warmupRatio: number;       // fraction of time budget for LR warmup
  warmdownRatio: number;     // fraction of time budget for LR warmdown
  finalLrFrac: number;       // final LR as fraction of initial
  // Training
  maxSteps: number;
  timeBudget: number;        // seconds (default 300 = 5 min)
  gradAccumSteps: number;    // total batch = batchSize * seqLen * gradAccumSteps
  // Distributed
  signalingUrl: string;
  minRingSize: number;
  checkpointInterval: number;
  dataUrl: string;
  // Eval
  evalTokens: number;        // tokens for BPB evaluation
}

export interface TrainerMetrics {
  step: number;
  loss: number;
  smoothLoss: number;
  gradNorm: number;
  tokensPerSec: number;
  totalTokens: number;
  peersConnected: number;
  allReduceTimeMs: number;
  computeTimeMs: number;
  elapsedSeconds: number;
  valBpb: number;
  mfuPercent: number;
  numParamsM: number;
}

const DEFAULT_TRAINER_CONFIG: TrainerConfig = {
  modelConfig: DEFAULT_CONFIG,
  batchSize: 4,
  unembedding_lr: 0.004,
  embedding_lr: 0.6,
  matrix_lr: 0.04,
  scalar_lr: 0.5,
  weight_decay: 0.2,
  adam_betas: [0.8, 0.95],
  warmupRatio: 0.0,
  warmdownRatio: 0.5,
  finalLrFrac: 0.0,
  maxSteps: Infinity,
  timeBudget: 300, // 5 minutes
  gradAccumSteps: 1,
  signalingUrl: 'ws://localhost:8788',
  minRingSize: 1,
  checkpointInterval: 0,
  dataUrl: '/data/tokens.bin',
  evalTokens: 40 * 524288, // ~20M tokens for eval (matches Karpathy)
};

// H100 BF16 peak FLOPs (for MFU calculation)
const H100_BF16_PEAK_FLOPS = 989.5e12;

export class DistributedTrainer {
  private config: TrainerConfig;
  private model: GPTModel | null = null;
  private optimizer: MuonAdamWOptimizer | null = null;
  private mesh: WebRTCMesh;
  private allReduce: RingAllReduce | null = null;
  private dataLoader: TokenDataLoader | null = null;
  private running = false;
  private startTime = 0;
  private totalTrainingTime = 0;
  private numFlopsPerToken = 0;

  metrics: TrainerMetrics = {
    step: 0,
    loss: 0,
    smoothLoss: 0,
    gradNorm: 0,
    tokensPerSec: 0,
    totalTokens: 0,
    peersConnected: 0,
    allReduceTimeMs: 0,
    computeTimeMs: 0,
    elapsedSeconds: 0,
    valBpb: 0,
    mfuPercent: 0,
    numParamsM: 0,
  };

  onMetricsUpdate: ((metrics: TrainerMetrics) => void) | null = null;
  onLog: ((msg: string) => void) | null = null;

  constructor(config: Partial<TrainerConfig> = {}) {
    this.config = { ...DEFAULT_TRAINER_CONFIG, ...config };
    this.mesh = new WebRTCMesh(`trainer-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
  }

  private lossScale = 1024.0;
  private consecutiveGoodGradients = 0;

  private log(msg: string): void {
    console.log(`[Trainer] ${msg}`);
    this.onLog?.(msg);
  }

  // ---------------------------------------------------------------------------
  // Karpathy's exact schedules
  // ---------------------------------------------------------------------------

  /**
   * LR multiplier: warmup → steady → warmdown
   * progress = totalTrainingTime / timeBudget
   */
  private getLrMultiplier(progress: number): number {
    const { warmupRatio, warmdownRatio, finalLrFrac } = this.config;
    if (progress < warmupRatio) {
      return warmupRatio > 0 ? progress / warmupRatio : 1.0;
    } else if (progress < 1.0 - warmdownRatio) {
      return 1.0;
    } else {
      const cooldown = (1.0 - progress) / warmdownRatio;
      return cooldown * 1.0 + (1 - cooldown) * finalLrFrac;
    }
  }

  /**
   * Muon momentum: ramps from 0.85 to 0.95 over 300 steps
   */
  private getMuonMomentum(step: number): number {
    const frac = Math.min(step / 300, 1);
    return (1 - frac) * 0.85 + frac * 0.95;
  }

  /**
   * Weight decay: decays to 0 over training
   */
  private getWeightDecay(progress: number): number {
    return this.config.weight_decay * (1 - progress);
  }

  // ---------------------------------------------------------------------------
  // Initialization
  // ---------------------------------------------------------------------------

  async initialize(): Promise<void> {
    this.log('🔧 Initializing TensorFlow.js WebGPU backend...');
    this.model = new GPTModel(this.config.modelConfig);
    await this.model.init();
    this.log(`🎮 Backend: ${tf.getBackend()}`);

    const vars = this.model.getTrainableVariables();
    let numParams = 0;
    vars.forEach(v => numParams += v.size);
    this.metrics.numParamsM = numParams / 1e6;
    this.log(`📊 Model: ${this.metrics.numParamsM.toFixed(1)}M trainable params`);

    // Compute FLOPs per token for MFU tracking
    this.numFlopsPerToken = this.model.estimateFlopsPerToken();
    this.log(`📊 Estimated FLOPs/token: ${this.numFlopsPerToken.toExponential(2)}`);

    // Load real tokenized data
    if (this.config.dataUrl) {
      this.log(`📦 Loading training data from ${this.config.dataUrl}...`);
      try {
        this.dataLoader = new TokenDataLoader(this.config.dataUrl, this.config.modelConfig.vocabSize);
        await this.dataLoader.load();
        this.log(`📦 Loaded ${(this.dataLoader.numTokens / 1e3).toFixed(0)}K tokens`);
      } catch (e) {
        this.log(`⚠️ Failed to load data, falling back to synthetic: ${e}`);
        this.dataLoader = null;
      }
    }

    // Initialize MuonAdamW optimizer
    this.log('🔧 Initializing MuonAdamW optimizer...');
    this.optimizer = createMuonAdamW(this.model, {
      unembedding_lr: this.config.unembedding_lr,
      embedding_lr: this.config.embedding_lr,
      matrix_lr: this.config.matrix_lr,
      scalar_lr: this.config.scalar_lr,
      weight_decay: this.config.weight_decay,
      adam_betas: this.config.adam_betas,
      model_dim: this.config.modelConfig.nEmbd,
    });

    // Connect to WebRTC signaling
    this.log('🌐 Connecting to signaling server...');
    this.mesh.onLog = (msg) => this.log(`[P2P] ${msg}`);
    this.mesh.onPeerStateChange = (peerId, state) => {
      this.metrics.peersConnected = this.mesh.connectedPeerCount;
      this.log(`Peer ${peerId.slice(-6)}: ${state} | Total peers: ${this.metrics.peersConnected}`);
    };

    this.mesh.onRingFormed = (ring) => {
      this.log(`🔗 Ring formed! Position ${ring.self}/${ring.totalPeers}`);
      const dynamicTimeout = this.mesh.dynamicTimeoutMs;
      this.log(`⏱️ Dynamic all-reduce timeout: ${dynamicTimeout.toFixed(0)}ms`);
      this.allReduce = new RingAllReduce(this.mesh, {
        timeoutMs: dynamicTimeout,
        onProgress: (phase, step, total) => {
          this.log(`  AllReduce ${phase}: ${step}/${total}`);
        },
        onAbort: (reason, phase, step) => {
          this.log(`🚨 All-reduce aborted at ${phase} step ${step}: ${reason}`);
        },
      });
      this.allReduce.onLog = (msg) => this.log(`[AR] ${msg}`);
    };

    this.mesh.onPeerLatencyUpdate = (_peerId, _rtt, smoothed) => {
      if (this.allReduce) {
        const newTimeout = this.mesh.dynamicTimeoutMs;
        this.allReduce.onLog?.(`Dynamic timeout updated: ${newTimeout.toFixed(0)}ms (peer latency: ${smoothed.toFixed(1)}ms)`);
      }
    };

    this.mesh.connectSignaling(this.config.signalingUrl);
    this.log('✅ Initialization complete. Waiting for peers...');
  }

  // ---------------------------------------------------------------------------
  // Batch Generation
  // ---------------------------------------------------------------------------

  private generateBatch(): { inputIds: Int32Array; targets: Int32Array } {
    const B = this.config.batchSize;
    const T = this.config.modelConfig.sequenceLen;

    if (this.dataLoader) {
      return this.dataLoader.nextBatch(B, T);
    }

    // Fallback: synthetic random data
    const V = this.config.modelConfig.vocabSize;
    const inputIds = new Int32Array(B * T);
    const targets = new Int32Array(B * T);
    for (let i = 0; i < B * T; i++) {
      inputIds[i] = Math.floor(Math.random() * V);
      targets[i] = Math.floor(Math.random() * V);
    }
    return { inputIds, targets };
  }

  // ---------------------------------------------------------------------------
  // Training Step
  // ---------------------------------------------------------------------------

  private residualBuffer: Float32Array | null = null;
  private sparsificationRatio = 0.10;

  private applyTopKSparsification(grad: Float32Array): void {
    const N = grad.length;
    if (!this.residualBuffer || this.residualBuffer.length !== N) {
      this.residualBuffer = new Float32Array(N);
    }

    for (let i = 0; i < N; i++) {
      grad[i] += this.residualBuffer[i];
    }

    const sampleSize = Math.min(N, 10000);
    const samples = new Float32Array(sampleSize);
    for (let i = 0; i < sampleSize; i++) {
      samples[i] = Math.abs(grad[Math.floor(Math.random() * N)]);
    }
    samples.sort();
    const threshold = samples[Math.floor(sampleSize * (1.0 - this.sparsificationRatio))];

    for (let i = 0; i < N; i++) {
      if (Math.abs(grad[i]) < threshold) {
        this.residualBuffer[i] = grad[i];
        grad[i] = 0.0;
      } else {
        this.residualBuffer[i] = 0.0;
      }
    }
  }

  async trainStep(): Promise<number> {
    if (!this.model || !this.optimizer) throw new Error('Not initialized');

    const B = this.config.batchSize;
    const T = this.config.modelConfig.sequenceLen;
    const step = this.metrics.step;
    const progress = Math.min(this.totalTrainingTime / this.config.timeBudget, 1.0);

    // 1. Generate Batch
    const { inputIds, targets } = this.generateBatch();
    const inputTensor = tf.tensor2d(inputIds, [B, T], 'int32');
    const targetTensor = tf.tensor2d(targets, [B, T], 'int32');

    const computeStart = performance.now();
    const vars = this.model.getTrainableVariables();

    // 2. Compute Scaled Loss and Gradients
    const { value: scaledLossTensor, grads: scaledGrads } = tf.variableGrads(() => {
      const fwdLoss = this.model!.forward(inputTensor, targetTensor).loss;
      return fwdLoss.mul(tf.scalar(this.lossScale));
    }, vars);

    const scaledLossArray = await scaledLossTensor.array() as number;
    const loss = scaledLossArray / this.lossScale;
    const computeTimeMs = performance.now() - computeStart;

    // Fast fail: abort if loss is exploding or NaN
    if (isNaN(loss) || loss > 100) {
      this.log(`❌ NaN/overflow in loss: ${loss}. Aborting step.`);
      scaledLossTensor.dispose();
      inputTensor.dispose();
      targetTensor.dispose();
      for (const g of Object.values(scaledGrads)) g.dispose();
      return loss;
    }

    // Check for gradient overflow
    let overflow = false;
    const gradTensors = Object.values(scaledGrads);
    if (gradTensors.length > 0) {
      const maxArr = gradTensors.map(g => g.abs().max());
      const globalMaxT = tf.max(tf.stack(maxArr));
      const globalMax = (await globalMaxT.data())[0];
      tf.dispose(maxArr);
      globalMaxT.dispose();
      if (globalMax > 65000 || isNaN(globalMax) || !isFinite(globalMax)) {
        overflow = true;
      }
    }

    if (overflow) {
      this.log(`⚠️ Gradient overflow! Halving loss scale to ${this.lossScale / 2}`);
      this.lossScale /= 2;
      this.consecutiveGoodGradients = 0;
      for (const g of gradTensors) g.dispose();
      scaledLossTensor.dispose();
      inputTensor.dispose();
      targetTensor.dispose();
      return loss;
    }

    this.consecutiveGoodGradients++;
    if (this.consecutiveGoodGradients >= 2000) {
      this.lossScale *= 2;
      this.consecutiveGoodGradients = 0;
      this.log(`📈 Safe for 2000 steps. Doubling loss scale to ${this.lossScale}`);
    }

    // 3. Serialize Gradients into flat Float32Array for WebRTC
    let totalGradSize = 0;
    vars.forEach(v => totalGradSize += v.size);
    const flatGrads = new Float32Array(totalGradSize);

    let offset = 0;
    for (const v of vars) {
      const g = scaledGrads[v.name];
      if (g) {
        const gData = await g.data();
        flatGrads.set(gData as Float32Array, offset);
        g.dispose();
      }
      offset += v.size;
    }

    // Apply Top-K sparsification
    this.applyTopKSparsification(flatGrads);

    // 4. All-Reduce Gradients
    let averagedGradients: Float32Array | null = null;
    let allReduceTimeMs = 0;

    if (this.allReduce && this.mesh.connectedPeerCount > 0) {
      const arStart = performance.now();
      averagedGradients = await this.allReduce.allReduce(flatGrads, step);
      allReduceTimeMs = performance.now() - arStart;

      if (averagedGradients === null) {
        this.log(`⚠️ All-reduce aborted at step ${step}, skipping`);
        scaledLossTensor.dispose();
        inputTensor.dispose();
        targetTensor.dispose();
        return loss;
      }
    } else {
      averagedGradients = flatGrads;
    }

    // 5. Reconstruct gradients and unscale
    const newGrads: Record<string, tf.Tensor> = {};
    offset = 0;
    for (const v of vars) {
      const gData = averagedGradients.subarray(offset, offset + v.size);
      newGrads[v.name] = tf.tensor(gData, v.shape).div(tf.scalar(this.lossScale));
      offset += v.size;
    }

    // 6. Apply Karpathy's schedules
    const lrm = this.getLrMultiplier(progress);
    const muonMomentum = this.getMuonMomentum(step);
    const muonWd = this.getWeightDecay(progress);

    this.optimizer.lrMultiplier = lrm;
    this.optimizer.muonMomentum = muonMomentum;
    this.optimizer.muonWeightDecay = muonWd;

    // 7. Optimizer step
    this.optimizer.step(newGrads);

    // Cleanup
    for (const v of vars) { newGrads[v.name].dispose(); }
    scaledLossTensor.dispose();
    inputTensor.dispose();
    targetTensor.dispose();

    // Update metrics
    const tokensProcessed = B * T;
    this.metrics.step = step + 1;
    this.metrics.loss = loss;
    this.metrics.smoothLoss = this.metrics.smoothLoss === 0
      ? loss
      : 0.9 * this.metrics.smoothLoss + 0.1 * loss;
    this.metrics.totalTokens += tokensProcessed;
    this.metrics.computeTimeMs = computeTimeMs;
    this.metrics.allReduceTimeMs = allReduceTimeMs;
    this.metrics.peersConnected = this.mesh.connectedPeerCount;
    this.metrics.tokensPerSec = tokensProcessed / (computeTimeMs / 1000);
    this.metrics.elapsedSeconds = (performance.now() - this.startTime) / 1000;

    // MFU (Model FLOPs Utilization)
    const steadyStateSteps = Math.max(1, step - 10);
    if (this.totalTrainingTime > 0) {
      this.metrics.mfuPercent = 100 * this.numFlopsPerToken * tokensProcessed * steadyStateSteps
        / this.totalTrainingTime / H100_BF16_PEAK_FLOPS;
    }

    this.onMetricsUpdate?.(this.metrics);

    return loss;
  }

  // ---------------------------------------------------------------------------
  // Training Loop
  // ---------------------------------------------------------------------------

  async startTraining(): Promise<void> {
    if (!this.model) await this.initialize();
    this.running = true;
    this.startTime = performance.now();
    this.log('🚀 Training started!');

    while (this.running && this.metrics.step < this.config.maxSteps) {
      // Time budget check (skip first 10 steps for compilation warmup)
      if (this.metrics.step > 10 && this.totalTrainingTime >= this.config.timeBudget) {
        this.log('⏰ Time budget exhausted!');
        break;
      }

      try {
        if (this.config.minRingSize > 1) {
          const connectedPeers = this.mesh.connectedPeerCount;
          if (connectedPeers < this.config.minRingSize - 1) {
            this.log(`⏸️ Ring too small (${connectedPeers + 1}/${this.config.minRingSize}), waiting...`);
            await new Promise(r => setTimeout(r, 2000));
            continue;
          }
        }

        const t0 = performance.now();
        const loss = await this.trainStep();
        const dt = (performance.now() - t0) / 1000;

        if (this.metrics.step > 10) {
          this.totalTrainingTime += dt;
        }

        const step = this.metrics.step;
        if (step % 5 === 0) {
          const lrm = this.getLrMultiplier(Math.min(this.totalTrainingTime / this.config.timeBudget, 1));
          this.log(
            `Step ${step} | loss=${loss.toFixed(6)} | smooth=${this.metrics.smoothLoss.toFixed(6)}` +
            ` | grad=${this.metrics.gradNorm.toFixed(4)} | lrm=${lrm.toFixed(2)}` +
            ` | ${this.metrics.tokensPerSec.toFixed(0)} tok/s | peers=${this.metrics.peersConnected}` +
            ` | ${this.metrics.elapsedSeconds.toFixed(0)}s`
          );
        }

        // Memory profiling
        if (step > 0 && step % 50 === 0) {
          const mem = tf.memory();
          this.log(`📊 Memory: ${mem.numTensors} tensors | ${(mem.numBytes / 1024 / 1024).toFixed(1)} MB`);
        }

        // Checkpoint saving
        if (this.config.checkpointInterval > 0 && step > 0 && step % this.config.checkpointInterval === 0) {
          await this.saveCheckpoint();
        }
      } catch (e) {
        this.log(`❌ Training error: ${e}`);
        await new Promise(r => setTimeout(r, 1000));
      }
    }

    // Final BPB evaluation (Karpathy's metric)
    if (this.model && this.dataLoader) {
      this.log('📊 Running final BPB evaluation...');
      try {
        const valBpb = await evaluateBPB(
          { forward: (x, y, r) => this.model!.forward(x, y, r) },
          this.dataLoader,
          this.config.batchSize,
          this.config.modelConfig.sequenceLen,
          this.config.evalTokens
        );
        this.metrics.valBpb = valBpb;
        this.log(`📊 val_bpb: ${valBpb.toFixed(6)}`);
      } catch (e) {
        this.log(`⚠️ BPB evaluation failed: ${e}`);
      }
    }

    // Print final summary (Karpathy format)
    const tEnd = performance.now();
    const totalSeconds = (tEnd - this.startTime) / 1000;
    const peakVramMb = tf.memory().numBytes / 1024 / 1024;

    this.log('---');
    this.log(`val_bpb:          ${this.metrics.valBpb.toFixed(6)}`);
    this.log(`training_seconds: ${this.totalTrainingTime.toFixed(1)}`);
    this.log(`total_seconds:    ${totalSeconds.toFixed(1)}`);
    this.log(`peak_vram_mb:     ${peakVramMb.toFixed(1)}`);
    this.log(`mfu_percent:      ${this.metrics.mfuPercent.toFixed(2)}`);
    this.log(`total_tokens_M:   ${(this.metrics.totalTokens / 1e6).toFixed(1)}`);
    this.log(`num_steps:        ${this.metrics.step}`);
    this.log(`num_params_M:     ${this.metrics.numParamsM.toFixed(1)}`);
    this.log(`depth:            ${this.config.modelConfig.nLayer}`);

    this.log('🏁 Training complete!');
  }

  // ---------------------------------------------------------------------------
  // Checkpoint (same as before, simplified)
  // ---------------------------------------------------------------------------

  async saveCheckpoint(): Promise<void> {
    if (!this.model) return;

    const step = this.metrics.step;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `checkpoint_step${step.toString().padStart(6, '0')}_${timestamp}.safetensors`;

    this.log(`💾 Saving checkpoint: ${filename}`);

    const vars = this.model.getTrainableVariables();
    const tensors: Record<string, { shape: number[]; dtype: string; offset: number; data: ArrayBuffer }> = {};
    let currentOffset = 0;

    for (const v of vars) {
      const data = await v.data();
      const floatData = new Float32Array(data);
      const buf = new ArrayBuffer(floatData.byteLength);
      new Float32Array(buf).set(floatData);
      tensors[v.name] = { shape: v.shape, dtype: 'F32', offset: currentOffset, data: buf };
      currentOffset += buf.byteLength;
    }

    const headerEntries: Record<string, any> = {};
    for (const [name, info] of Object.entries(tensors)) {
      headerEntries[name] = {
        dtype: info.dtype,
        shape: info.shape,
        data_offsets: [info.offset, info.offset + info.data.byteLength],
      };
    }
    headerEntries.__metadata__ = {
      step,
      lossScale: this.lossScale,
      totalTokens: this.metrics.totalTokens,
      config: JSON.stringify(this.config.modelConfig),
    };

    const headerJson = JSON.stringify(headerEntries);
    const headerBytes = new TextEncoder().encode(headerJson);
    const headerSize = BigInt(headerBytes.length);

    const totalSize = 8 + headerBytes.length + currentOffset;
    const output = new ArrayBuffer(totalSize);
    const view = new DataView(output);
    const uint8View = new Uint8Array(output);

    view.setBigUint64(0, headerSize, true);
    uint8View.set(headerBytes, 8);
    const dataBase = 8 + headerBytes.length;
    for (const info of Object.values(tensors)) {
      uint8View.set(new Uint8Array(info.data), dataBase + info.offset);
    }

    const blob = new Blob([output], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.log(`✅ Checkpoint saved: ${filename} (${(totalSize / 1024 / 1024).toFixed(2)} MB)`);
  }

  /**
   * Save checkpoint as ArrayBuffer (for testing).
   */
  async saveCheckpointAsBuffer(): Promise<ArrayBuffer> {
    if (!this.model) throw new Error('Model not initialized');

    const step = this.metrics.step;
    const vars = this.model.getTrainableVariables();
    const tensors: Record<string, { shape: number[]; dtype: string; offset: number; data: ArrayBuffer }> = {};
    let currentOffset = 0;

    for (const v of vars) {
      const data = await v.data();
      const floatData = new Float32Array(data);
      const buf = new ArrayBuffer(floatData.byteLength);
      new Float32Array(buf).set(floatData);
      tensors[v.name] = { shape: v.shape, dtype: 'F32', offset: currentOffset, data: buf };
      currentOffset += buf.byteLength;
    }

    const headerEntries: Record<string, any> = {};
    for (const [name, info] of Object.entries(tensors)) {
      headerEntries[name] = {
        dtype: info.dtype,
        shape: info.shape,
        data_offsets: [info.offset, info.offset + info.data.byteLength],
      };
    }
    headerEntries.__metadata__ = {
      step,
      lossScale: this.lossScale,
      totalTokens: this.metrics.totalTokens,
      config: JSON.stringify(this.config.modelConfig),
    };

    const headerJson = JSON.stringify(headerEntries);
    const headerBytes = new TextEncoder().encode(headerJson);
    const headerSize = BigInt(headerBytes.length);

    const totalSize = 8 + headerBytes.length + currentOffset;
    const output = new ArrayBuffer(totalSize);
    const view = new DataView(output);
    const uint8View = new Uint8Array(output);

    view.setBigUint64(0, headerSize, true);
    uint8View.set(headerBytes, 8);
    const dataBase = 8 + headerBytes.length;
    for (const info of Object.values(tensors)) {
      uint8View.set(new Uint8Array(info.data), dataBase + info.offset);
    }

    return output;
  }

  /**
   * Load checkpoint from a .safetensors File.
   */
  async loadCheckpoint(file: File): Promise<void> {
    if (!this.model) {
      throw new Error('Cannot load checkpoint: model not initialized. Call initialize() first.');
    }

    this.log(`📂 Loading checkpoint: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
    const buf = await file.arrayBuffer();
    const view = new DataView(buf);

    const headerSize = Number(view.getBigUint64(0, true));
    const headerBytes = new Uint8Array(buf, 8, headerSize);
    const headerJson = new TextDecoder().decode(headerBytes);
    const header: Record<string, any> = JSON.parse(headerJson);

    const dataBase = 8 + headerSize;
    const varsByName = this.model.getTrainableVariablesByName();
    let restoredWeights = 0;

    for (const [varName, varObj] of Object.entries(varsByName)) {
      const entry = header[varName];
      if (!entry) {
        this.log(`⚠️ No checkpoint data for weight: ${varName}`);
        continue;
      }

      const [start, end] = entry.data_offsets;
      const floatData = new Float32Array(buf, dataBase + start, (end - start) / 4);
      varObj.assign(tf.tensor(floatData, entry.shape));
      restoredWeights++;
    }

    const meta = header.__metadata__;
    if (meta) {
      this.metrics.step = meta.step ?? 0;
      this.lossScale = meta.lossScale ?? 1024.0;
      this.metrics.totalTokens = meta.totalTokens ?? 0;
      this.log(`📊 Resumed from step ${this.metrics.step} | lossScale=${this.lossScale}`);
    }

    this.log(`✅ Checkpoint loaded: ${restoredWeights} weights restored`);
  }

  /** Prompt user to select a checkpoint file and load it */
  async promptLoadCheckpoint(): Promise<void> {
    return new Promise((resolve, reject) => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.safetensors';
      input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) {
          reject(new Error('No file selected'));
          return;
        }
        try {
          await this.loadCheckpoint(file);
          resolve();
        } catch (e) {
          reject(e);
        }
      };
      input.click();
    });
  }

  stop(): void {
    this.running = false;
    this.log('⏹️ Stop requested');
  }

  destroy(): void {
    this.stop();
    this.mesh.disconnect();
    this.optimizer?.dispose();
  }
}
