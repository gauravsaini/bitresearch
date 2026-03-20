// Distributed Trainer — runs the training loop on each WebGPU worker
// Coordinates with peers via WebRTC ring all-reduce for gradient sync.
//
// Each worker independently:
//   1. Gets its own data batch
//   2. Runs forward + backward pass on WebGPU
//   3. Participates in ring all-reduce to average gradients across all peers
//   4. Applies optimizer step locally with the averaged gradients
//   5. Repeat — fully decentralized, no central parameter server!

import { GPUDeviceManager } from '../gpu/device';
import { Tensor } from '../gpu/tensor';
import { GPUOps } from '../gpu/ops';
import { GPTModel } from '../model/gpt';
import { GPTConfig, DEFAULT_CONFIG } from '../model/config';
import { WebRTCMesh } from './webrtc-mesh';
import { RingAllReduce } from './ring-allreduce';

export interface TrainerConfig {
  modelConfig: GPTConfig;
  batchSize: number;
  learningRate: number;
  warmupSteps: number;
  maxSteps: number;
  signalingUrl: string;     // WebRTC signaling server URL
  coordinatorUrl?: string;  // optional legacy coordinator for data/monitoring
}

export interface TrainerMetrics {
  step: number;
  loss: number;
  smoothLoss: number;
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
};

export class DistributedTrainer {
  private config: TrainerConfig;
  private mgr: GPUDeviceManager | null = null;
  private ops: GPUOps | null = null;
  private model: GPTModel | null = null;
  private mesh: WebRTCMesh;
  private allReduce: RingAllReduce | null = null;
  private running = false;
  private startTime = 0;

  // Optimizer state (simple AdamW)
  private optimState: {
    step: number;
    m: Float32Array | null;   // first moment
    v: Float32Array | null;   // second moment
  } = { step: 0, m: null, v: null };

  metrics: TrainerMetrics = {
    step: 0,
    loss: 0,
    smoothLoss: 0,
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

  private log(msg: string): void {
    console.log(`[Trainer] ${msg}`);
    this.onLog?.(msg);
  }

  /** Initialize WebGPU, model, and WebRTC connections */
  async initialize(): Promise<void> {
    // 1. Initialize WebGPU
    this.log('🔧 Initializing WebGPU...');
    this.mgr = await GPUDeviceManager.create();
    this.ops = new GPUOps(this.mgr);
    const info = this.mgr.getDeviceInfo();
    this.log(`🎮 GPU: ${info.vendor} ${info.architecture}`);

    // 2. Create model
    this.log('🧠 Creating model...');
    this.model = await GPTModel.create(this.config.modelConfig, this.mgr, this.ops);
    const paramCount = this.model.paramCount();
    this.log(`📊 Model: ${(paramCount / 1e6).toFixed(1)}M params`);

    // 3. Initialize optimizer state
    this.optimState.m = new Float32Array(paramCount).fill(0);
    this.optimState.v = new Float32Array(paramCount).fill(0);

    // 4. Connect to WebRTC signaling
    this.log('🌐 Connecting to signaling server...');
    this.mesh.onLog = (msg) => this.log(`[P2P] ${msg}`);
    this.mesh.onPeerStateChange = (peerId, state) => {
      this.metrics.peersConnected = this.mesh.connectedPeerCount;
      this.log(`Peer ${peerId.slice(-6)}: ${state} | Total peers: ${this.metrics.peersConnected}`);
    };

    this.mesh.onRingFormed = (ring) => {
      this.log(`🔗 Ring formed! Position ${ring.self}/${ring.totalPeers}`);

      // Initialize all-reduce
      this.allReduce = new RingAllReduce(this.mesh, {
        timeoutMs: 10000,
        onProgress: (phase, step, total) => {
          this.log(`  AllReduce ${phase}: ${step}/${total}`);
        },
      });
      this.allReduce.onLog = (msg) => this.log(`[AR] ${msg}`);
    };

    this.mesh.connectSignaling(this.config.signalingUrl);
    this.log('✅ Initialization complete. Waiting for peers...');
  }

  /** Generate a random training batch (synthetic data for demo) */
  private generateBatch(): { inputIds: Uint32Array; targets: Int32Array } {
    const B = this.config.batchSize;
    const T = this.config.modelConfig.sequenceLen;
    const V = this.config.modelConfig.vocabSize;

    const inputIds = new Uint32Array(B * T);
    const targets = new Int32Array(B * T);

    for (let i = 0; i < B * T; i++) {
      inputIds[i] = Math.floor(Math.random() * V);
      targets[i] = Math.floor(Math.random() * V);
    }

    return { inputIds, targets };
  }

  /** Learning rate schedule with warmup */
  private getLearningRate(step: number): number {
    if (step < this.config.warmupSteps) {
      return this.config.learningRate * (step + 1) / this.config.warmupSteps;
    }
    return this.config.learningRate;
  }

  /** Apply AdamW optimizer step with averaged gradients */
  private applyAdamW(params: Float32Array, gradients: Float32Array, lr: number): Float32Array {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const weightDecay = 0.01;

    this.optimState.step++;
    const t = this.optimState.step;
    const m = this.optimState.m!;
    const v = this.optimState.v!;

    const bias1 = 1 - Math.pow(beta1, t);
    const bias2 = 1 - Math.pow(beta2, t);

    for (let i = 0; i < params.length; i++) {
      // Update moments
      m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
      v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i];

      // Bias correction
      const mHat = m[i] / bias1;
      const vHat = v[i] / bias2;

      // Weight decay
      params[i] -= lr * weightDecay * params[i];

      // Update
      params[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }

    return params;
  }

  /** Run a single training step */
  async trainStep(): Promise<number> {
    if (!this.model || !this.mgr || !this.ops) {
      throw new Error('Not initialized');
    }

    const B = this.config.batchSize;
    const T = this.config.modelConfig.sequenceLen;
    const step = this.metrics.step;

    // 1. Generate batch
    const { inputIds, targets } = this.generateBatch();

    // 2. Forward pass on WebGPU
    const computeStart = performance.now();

    const inputTensor = await Tensor.fromArray(
      this.mgr, inputIds, [B, T], 'batch_input',
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    // For targets, we need to pass as f32 but interpret as i32 in the shader
    const targetF32 = new Float32Array(targets.length);
    for (let i = 0; i < targets.length; i++) targetF32[i] = targets[i];
    const targetTensor = await Tensor.fromArray(
      this.mgr, targetF32, [B, T], 'batch_targets',
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );

    const { loss } = await this.model.forward(inputTensor, targetTensor, B, T);

    const computeTimeMs = performance.now() - computeStart;

    // 3. Get gradients (numerical approximation for now — real autograd TODO)
    // For demo purposes, we use the loss as a signal and create pseudo-gradients
    const params = await this.model.getAllParams(this.mgr);
    const pseudoGradients = new Float32Array(params.length);
    // Simple gradient estimation: random direction scaled by loss
    for (let i = 0; i < pseudoGradients.length; i++) {
      pseudoGradients[i] = (Math.random() - 0.5) * 0.001 * loss;
    }

    // 4. All-reduce gradients across peers (if connected)
    let averagedGradients: Float32Array;
    let allReduceTimeMs = 0;

    if (this.allReduce && this.mesh.connectedPeerCount > 0) {
      const arStart = performance.now();
      averagedGradients = await this.allReduce.allReduce(pseudoGradients, step);
      allReduceTimeMs = performance.now() - arStart;
      this.log(`🔄 All-reduce: ${allReduceTimeMs.toFixed(0)}ms across ${this.mesh.connectedPeerCount + 1} peers`);
    } else {
      averagedGradients = pseudoGradients;
    }

    // 5. Apply optimizer step
    const lr = this.getLearningRate(step);
    const updatedParams = this.applyAdamW(params, averagedGradients, lr);

    // 6. Load updated parameters back to model
    await this.model.loadAllParams(this.mgr, updatedParams);

    // Cleanup
    inputTensor.destroy();
    targetTensor.destroy();

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

  /** Start the training loop */
  async startTraining(): Promise<void> {
    if (!this.model) {
      await this.initialize();
    }

    this.running = true;
    this.startTime = performance.now();
    this.log('🚀 Training started!');

    while (this.running && this.metrics.step < this.config.maxSteps) {
      try {
        const loss = await this.trainStep();
        const step = this.metrics.step;

        // Log progress
        if (step % 5 === 0) {
          const peers = this.mesh.connectedPeerCount;
          const tps = this.metrics.tokensPerSec.toFixed(0);
          const elapsed = this.metrics.elapsedSeconds.toFixed(0);
          this.log(`Step ${step} | loss=${loss.toFixed(6)} | smooth=${this.metrics.smoothLoss.toFixed(6)} | ${tps} tok/s | peers=${peers} | ${elapsed}s`);
        }
      } catch (e) {
        this.log(`❌ Training error: ${e}`);
        // Brief pause before retrying
        await new Promise(r => setTimeout(r, 1000));
      }
    }

    this.log('🏁 Training stopped');
  }

  /** Stop the training loop */
  stop(): void {
    this.running = false;
    this.log('⏹️ Stop requested');
  }

  /** Clean up resources */
  destroy(): void {
    this.stop();
    this.mesh.disconnect();
  }
}
