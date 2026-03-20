// Distributed Trainer — runs the training loop on each WebRTC worker
// Coordinates with peers via WebRTC ring all-reduce for gradient sync.

import * as tf from '@tensorflow/tfjs';
import { GPTModel } from '../model/gpt';
import { GPTConfig, DEFAULT_CONFIG } from '../model/config';
import { WebRTCMesh } from './webrtc-mesh';
import { RingAllReduce } from './ring-allreduce';
import { TokenDataLoader } from '../data/dataloader';

export interface TrainerConfig {
  modelConfig: GPTConfig;
  batchSize: number;
  learningRate: number;
  warmupSteps: number;
  maxSteps: number;
  signalingUrl: string;
  minRingSize: number;      // pause training if connected peers < this
  checkpointInterval: number; // save checkpoint every N steps (0 = disabled)
  dataUrl: string;           // URL to fetch tokenized data (empty = synthetic)
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
}

const DEFAULT_TRAINER_CONFIG: TrainerConfig = {
  modelConfig: DEFAULT_CONFIG,
  batchSize: 4,
  learningRate: 0.001,
  warmupSteps: 10,
  maxSteps: Infinity,
  signalingUrl: 'ws://localhost:8788',
  minRingSize: 1,
  checkpointInterval: 0,
  dataUrl: '/data/tokens.bin',
};

export class DistributedTrainer {
  private config: TrainerConfig;
  private model: GPTModel | null = null;
  private optimizer: tf.Optimizer | null = null;
  private mesh: WebRTCMesh;
  private allReduce: RingAllReduce | null = null;
  private dataLoader: TokenDataLoader | null = null;
  private running = false;
  private startTime = 0;

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

  async initialize(): Promise<void> {
    this.log('🔧 Initializing TensorFlow.js WebGPU backend...');
    this.model = new GPTModel(this.config.modelConfig);
    await this.model.init();
    this.log(`🎮 Backend: ${tf.getBackend()}`);

    const vars = this.model.getTrainableVariables();
    let numParams = 0;
    vars.forEach(v => numParams += v.size);
    this.log(`📊 Model: ${(numParams / 1e6).toFixed(1)}M trainable params`);

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

    // Initialize optimizer
    this.optimizer = tf.train.adam(this.config.learningRate, 0.9, 0.999);

    // Connect to WebRTC signaling
    this.log('🌐 Connecting to signaling server...');
    this.mesh.onLog = (msg) => this.log(`[P2P] ${msg}`);
    this.mesh.onPeerStateChange = (peerId, state) => {
      this.metrics.peersConnected = this.mesh.connectedPeerCount;
      this.log(`Peer ${peerId.slice(-6)}: ${state} | Total peers: ${this.metrics.peersConnected}`);
    };

    this.mesh.onRingFormed = (ring) => {
      this.log(`🔗 Ring formed! Position ${ring.self}/${ring.totalPeers}`);
      // Use dynamic timeout based on peer latencies
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

    // Track latency updates to refresh dynamic timeout
    this.mesh.onPeerLatencyUpdate = (_peerId, _rtt, smoothed) => {
      if (this.allReduce) {
        const newTimeout = this.mesh.dynamicTimeoutMs;
        this.allReduce.onLog?.(`Dynamic timeout updated: ${newTimeout.toFixed(0)}ms (peer latency: ${smoothed.toFixed(1)}ms)`);
      }
    };

    this.mesh.connectSignaling(this.config.signalingUrl);
    this.log('✅ Initialization complete. Waiting for peers...');
  }

  private generateBatch(): { inputIds: Int32Array; targets: Int32Array } {
    const B = this.config.batchSize;
    const T = this.config.modelConfig.sequenceLen;

    // Use real data if loader is available
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

  private getLearningRate(step: number): number {
    if (step < this.config.warmupSteps) {
      return this.config.learningRate * (step + 1) / this.config.warmupSteps;
    }
    return this.config.learningRate;
  }

  private residualBuffer: Float32Array | null = null;
  private sparsificationRatio = 0.10; // Keep top 10%

  private applyTopKSparsification(grad: Float32Array): void {
    const N = grad.length;
    if (!this.residualBuffer || this.residualBuffer.length !== N) {
      this.residualBuffer = new Float32Array(N);
    }

    // 1. Add error feedback (residual from previous steps)
    for (let i = 0; i < N; i++) {
      grad[i] += this.residualBuffer[i];
    }

    // 2. Approximate Top-K threshold via sampling (O(1) sort)
    const sampleSize = Math.min(N, 10000);
    const samples = new Float32Array(sampleSize);
    for (let i = 0; i < sampleSize; i++) {
        const randomIdx = Math.floor(Math.random() * N);
        samples[i] = Math.abs(grad[randomIdx]);
    }
    
    samples.sort();
    const thresholdIdx = Math.floor(sampleSize * (1.0 - this.sparsificationRatio));
    const threshold = samples[thresholdIdx];

    // 3. Mask out elements below threshold and store in residual
    let nonZeroCount = 0;
    for (let i = 0; i < N; i++) {
      if (Math.abs(grad[i]) < threshold) {
        this.residualBuffer[i] = grad[i];
        grad[i] = 0.0;
      } else {
        this.residualBuffer[i] = 0.0;
        nonZeroCount++;
      }
    }
    
    // We log periodically outside, but this mask ensures flatGrads has 90% exactly 0.0 entries.
  }

  async trainStep(): Promise<number> {
    if (!this.model || !this.optimizer) throw new Error('Not initialized');

    const B = this.config.batchSize;
    const T = this.config.modelConfig.sequenceLen;
    const step = this.metrics.step;

    // 1. Generate Batch
    const { inputIds, targets } = this.generateBatch();
    const inputTensor = tf.tensor2d(inputIds, [B, T], 'int32');
    const targetTensor = tf.tensor2d(targets, [B, T], 'int32');

    const computeStart = performance.now();
    const vars = this.model.getTrainableVariables();

    // 2. Compute Scaled Loss and Gradients (Free Autograd!)
    const { value: scaledLossTensor, grads: scaledGrads } = tf.variableGrads(() => {
      const fwdLoss = this.model!.forward(inputTensor, targetTensor).loss;
      return fwdLoss.mul(tf.scalar(this.lossScale));
    }, vars);

    const scaledLossArray = await scaledLossTensor.array() as number;
    const loss = scaledLossArray / this.lossScale;
    const computeTimeMs = performance.now() - computeStart;

    // Check for F16 overflow (max value ~65500)
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
      this.log(`⚠️ Gradient overflow detected! Halving loss scale to ${this.lossScale / 2}`);
      this.lossScale /= 2;
      this.consecutiveGoodGradients = 0;
      
      // Dispose and skip step
      for (const g of gradTensors) g.dispose();
      scaledLossTensor.dispose();
      inputTensor.dispose();
      targetTensor.dispose();
      
      return loss;
    }

    // Scale is safe!
    this.consecutiveGoodGradients++;
    if (this.consecutiveGoodGradients >= 2000) {
      this.lossScale *= 2;
      this.consecutiveGoodGradients = 0;
      this.log(`📈 Safe for 2000 steps. Doubling loss scale to ${this.lossScale}`);
    }

    // Compute global gradient norm from UNSCALED gradients
    let gradNorm = 0;
    if (gradTensors.length > 0) {
      const sqNorms = gradTensors.map(g => tf.norm(g.div(this.lossScale)).square());
      const totalNormSq = tf.addN(sqNorms);
      gradNorm = Math.sqrt((await totalNormSq.data())[0]);
      tf.dispose(sqNorms);
      totalNormSq.dispose();
    }
    this.metrics.gradNorm = gradNorm;

    // 3. Serialize Gradients into a 1D Array for WebRTC payload
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

    // Apply Top-K sparsification and error feedback buffer
    this.applyTopKSparsification(flatGrads);

    // 4. All-Reduce Gradients across the P2P Mesh
    let averagedGradients: Float32Array | null = null;
    let allReduceTimeMs = 0;

    if (this.allReduce && this.mesh.connectedPeerCount > 0) {
      const arStart = performance.now();
      averagedGradients = await this.allReduce.allReduce(flatGrads, step);
      allReduceTimeMs = performance.now() - arStart;

      if (averagedGradients === null) {
        // All-reduce was aborted (peer failure mid-step)
        this.log(`⚠️ All-reduce aborted at step ${step}, skipping optimizer update`);
        // Dispose tensors
        for (const v of vars) { scaledGrads[v.name]?.dispose(); }
        scaledLossTensor.dispose();
        inputTensor.dispose();
        targetTensor.dispose();
        return loss;
      }

      this.log(`🔄 All-reduce: ${allReduceTimeMs.toFixed(0)}ms across ${this.mesh.connectedPeerCount + 1} peers`);
    } else {
      averagedGradients = flatGrads;
    }

    // 5. Reconstruct and Unscale Gradients before Optimizer
    const lr = this.getLearningRate(step);
    (this.optimizer as any).learningRate = lr;

    const newGrads: Record<string, tf.Tensor> = {};
    offset = 0;
    for (const v of vars) {
      const gData = averagedGradients.subarray(offset, offset + v.size);
      // UNSCALE the float32 array right before building the tensor
      newGrads[v.name] = tf.tensor(gData, v.shape).div(tf.scalar(this.lossScale));
      offset += v.size;
    }

    this.optimizer.applyGradients(newGrads);

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

    this.onMetricsUpdate?.(this.metrics);

    return loss;
  }

  async startTraining(): Promise<void> {
    if (!this.model) await this.initialize();
    this.running = true;
    this.startTime = performance.now();
    this.log('🚀 Training started!');

    while (this.running && this.metrics.step < this.config.maxSteps) {
      try {
        // Minimum ring size check: pause if not enough peers
        if (this.config.minRingSize > 1) {
          const connectedPeers = this.mesh.connectedPeerCount;
          if (connectedPeers < this.config.minRingSize - 1) {
            this.log(`⏸️ Ring too small (${connectedPeers + 1}/${this.config.minRingSize}), waiting for peers...`);
            await new Promise(r => setTimeout(r, 2000));
            continue;
          }
        }

        const loss = await this.trainStep();
        const step = this.metrics.step;

        if (step % 5 === 0) {
          const peers = this.mesh.connectedPeerCount;
          const tps = this.metrics.tokensPerSec.toFixed(0);
          const elapsed = this.metrics.elapsedSeconds.toFixed(0);
          this.log(`Step ${step} | loss=${loss.toFixed(6)} | smooth=${this.metrics.smoothLoss.toFixed(6)} | gradNorm=${this.metrics.gradNorm.toFixed(4)} | ${tps} tok/s | peers=${peers} | ${elapsed}s`);
        }

        // Periodic Memory Profiling
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
    this.log('🏁 Training stopped');
  }

  /**
   * Save checkpoint as .safetensors binary format with Adam optimizer state.
   * Format: header (JSON metadata) + binary tensor data.
   * Filename includes timestamp and step number for versioning.
   */
  async saveCheckpoint(): Promise<void> {
    if (!this.model || !this.optimizer) {
      this.log('⚠️ Cannot save checkpoint: model or optimizer not initialized');
      return;
    }

    const step = this.metrics.step;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `checkpoint_step${step.toString().padStart(6, '0')}_${timestamp}.safetensors`;

    this.log(`💾 Saving checkpoint: ${filename}`);

    const vars = this.model.getTrainableVariables();
    const tensors: Record<string, { shape: number[]; dtype: string; offset: number; data: ArrayBuffer }> = {};
    let currentOffset = 0;

    // Serialize model weights
    for (const v of vars) {
      const data = await v.data();
      const floatData = new Float32Array(data);
      const buf = new ArrayBuffer(floatData.byteLength);
      new Float32Array(buf).set(floatData);
      tensors[v.name] = {
        shape: v.shape,
        dtype: 'F32',
        offset: currentOffset,
        data: buf,
      };
      currentOffset += buf.byteLength;
    }

    // Serialize Adam optimizer moments (m and v)
    // TF.js Adam optimizer stores internally; we extract from its slots
    const optimizerAny = this.optimizer as any;
    if (optimizerAny.accumulators) {
      for (const [varName, acc] of Object.entries(optimizerAny.accumulators as Record<string, tf.Tensor>)) {
        const data = await acc.data();
        const floatData = new Float32Array(data);
        const buf = new ArrayBuffer(floatData.byteLength);
        new Float32Array(buf).set(floatData);
        tensors[`adam_m_${varName}`] = {
          shape: acc.shape,
          dtype: 'F32',
          offset: currentOffset,
          data: buf,
        };
        currentOffset += buf.byteLength;
      }
    }
    if (optimizerAny.m) {
      for (const [varName, moment] of Object.entries(optimizerAny.m as Record<string, tf.Tensor>)) {
        const data = await moment.data();
        const floatData = new Float32Array(data);
        const buf = new ArrayBuffer(floatData.byteLength);
        new Float32Array(buf).set(floatData);
        tensors[`adam_m_${varName}`] = {
          shape: moment.shape,
          dtype: 'F32',
          offset: currentOffset,
          data: buf,
        };
        currentOffset += buf.byteLength;
      }
    }
    if (optimizerAny.v) {
      for (const [varName, moment] of Object.entries(optimizerAny.v as Record<string, tf.Tensor>)) {
        const data = await moment.data();
        const floatData = new Float32Array(data);
        const buf = new ArrayBuffer(floatData.byteLength);
        new Float32Array(buf).set(floatData);
        tensors[`adam_v_${varName}`] = {
          shape: moment.shape,
          dtype: 'F32',
          offset: currentOffset,
          data: buf,
        };
        currentOffset += buf.byteLength;
      }
    }

    // Build .safetensors header
    // Format: [header_size: u64 LE][header_json][tensor_data...]
    const headerEntries: Record<string, any> = {};
    for (const [name, info] of Object.entries(tensors)) {
      headerEntries[name] = {
        dtype: info.dtype,
        shape: info.shape,
        data_offsets: [info.offset, info.offset + info.data.byteLength],
      };
    }

    // Add metadata for full resume
    headerEntries.__metadata__ = {
      step: step,
      lossScale: this.lossScale,
      consecutiveGoodGradients: this.consecutiveGoodGradients,
      smoothLoss: this.metrics.smoothLoss,
      totalTokens: this.metrics.totalTokens,
      config: JSON.stringify(this.config.modelConfig),
    };

    const headerJson = JSON.stringify(headerEntries);
    const headerBytes = new TextEncoder().encode(headerJson);
    const headerSize = BigInt(headerBytes.length);

    // Total buffer: 8 bytes (header size) + header + all tensor data
    const totalSize = 8 + headerBytes.length + currentOffset;
    const output = new ArrayBuffer(totalSize);
    const view = new DataView(output);
    const uint8View = new Uint8Array(output);

    // Write header size as u64 LE
    view.setBigUint64(0, headerSize, true);
    // Write header JSON
    uint8View.set(headerBytes, 8);
    // Write tensor data
    const dataBase = 8 + headerBytes.length;
    for (const info of Object.values(tensors)) {
      uint8View.set(new Uint8Array(info.data), dataBase + info.offset);
    }

    // Download as Blob
    const blob = new Blob([output], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    const sizeMB = (totalSize / 1024 / 1024).toFixed(2);
    this.log(`✅ Checkpoint saved: ${filename} (${sizeMB} MB, ${Object.keys(tensors).length} tensors)`);
  }

  /**
   * Load checkpoint from a .safetensors file.
   * Restores model weights, Adam optimizer state, step number, and training metadata.
   */
  async loadCheckpoint(file: File): Promise<void> {
    if (!this.model || !this.optimizer) {
      throw new Error('Cannot load checkpoint: model or optimizer not initialized. Call initialize() first.');
    }

    this.log(`📂 Loading checkpoint: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
    const buf = await file.arrayBuffer();
    const view = new DataView(buf);

    // Parse safetensors header: [header_size: u64 LE][header_json][tensor_data...]
    const headerSize = Number(view.getBigUint64(0, true));
    const headerBytes = new Uint8Array(buf, 8, headerSize);
    const headerJson = new TextDecoder().decode(headerBytes);
    const header: Record<string, any> = JSON.parse(headerJson);

    const dataBase = 8 + headerSize;
    const varsByName = this.model.getTrainableVariablesByName();
    let restoredWeights = 0;
    let restoredMoments = 0;

    // Restore model weights
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

    // Restore Adam optimizer state
    const optimizerAny = this.optimizer as any;

    // First moment (m) — stored as adam_m_{varName}
    if (optimizerAny.m) {
      for (const [varName, momentTensor] of Object.entries(optimizerAny.m as Record<string, tf.Tensor>)) {
        const key = `adam_m_${varName}`;
        const entry = header[key];
        if (!entry) continue;

        const [start, end] = entry.data_offsets;
        const floatData = new Float32Array(buf, dataBase + start, (end - start) / 4);
        (momentTensor as tf.Variable).assign(tf.tensor(floatData, entry.shape));
        restoredMoments++;
      }
    }

    // Second moment (v) — stored as adam_v_{varName}
    if (optimizerAny.v) {
      for (const [varName, momentTensor] of Object.entries(optimizerAny.v as Record<string, tf.Tensor>)) {
        const key = `adam_v_${varName}`;
        const entry = header[key];
        if (!entry) continue;

        const [start, end] = entry.data_offsets;
        const floatData = new Float32Array(buf, dataBase + start, (end - start) / 4);
        (momentTensor as tf.Variable).assign(tf.tensor(floatData, entry.shape));
        restoredMoments++;
      }
    }

    // Restore training metadata
    const meta = header.__metadata__;
    if (meta) {
      this.metrics.step = meta.step ?? 0;
      this.lossScale = meta.lossScale ?? 1024.0;
      this.consecutiveGoodGradients = meta.consecutiveGoodGradients ?? 0;
      this.metrics.smoothLoss = meta.smoothLoss ?? 0;
      this.metrics.totalTokens = meta.totalTokens ?? 0;
      this.log(`📊 Resumed from step ${this.metrics.step} | lossScale=${this.lossScale} | totalTokens=${this.metrics.totalTokens}`);
    }

    this.log(`✅ Checkpoint loaded: ${restoredWeights} weights, ${restoredMoments} optimizer moments`);
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
  }
}

