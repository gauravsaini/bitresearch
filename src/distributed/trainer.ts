// Distributed Trainer — runs the training loop on each WebRTC worker
// Coordinates with peers via WebRTC ring all-reduce for gradient sync.

import * as tf from '@tensorflow/tfjs';
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
  signalingUrl: string;
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
  private model: GPTModel | null = null;
  private optimizer: tf.Optimizer | null = null;
  private mesh: WebRTCMesh;
  private allReduce: RingAllReduce | null = null;
  private running = false;
  private startTime = 0;

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

  async initialize(): Promise<void> {
    this.log('🔧 Initializing TensorFlow.js WebGPU backend...');
    this.model = new GPTModel(this.config.modelConfig);
    await this.model.init();
    this.log(`🎮 Backend: ${tf.getBackend()}`);

    const vars = this.model.getTrainableVariables();
    let numParams = 0;
    vars.forEach(v => numParams += v.size);
    this.log(`📊 Model: ${(numParams / 1e6).toFixed(1)}M trainable params`);

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

  private generateBatch(): { inputIds: Int32Array; targets: Int32Array } {
    const B = this.config.batchSize;
    const T = this.config.modelConfig.sequenceLen;
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

    // 2. Compute Loss and Gradients (Free Autograd!)
    const { value: lossTensor, grads } = tf.variableGrads(() => {
      return this.model!.forward(inputTensor, targetTensor).loss;
    }, vars);

    const loss = (await lossTensor.array()) as number;
    const computeTimeMs = performance.now() - computeStart;

    // 3. Serialize Gradients into a 1D Array for WebRTC payload
    let totalGradSize = 0;
    vars.forEach(v => totalGradSize += v.size);
    const flatGrads = new Float32Array(totalGradSize);
    
    let offset = 0;
    for (const v of vars) {
      const g = grads[v.name];
      if (g) {
        // TFJS grads might be sparse or non-existent if not actively contributing to loss
        const gData = await g.data();
        flatGrads.set(gData as Float32Array, offset);
        g.dispose();
      }
      offset += v.size;
    }

    // 4. All-Reduce Gradients across the P2P Mesh
    let averagedGradients: Float32Array;
    let allReduceTimeMs = 0;

    if (this.allReduce && this.mesh.connectedPeerCount > 0) {
      const arStart = performance.now();
      averagedGradients = await this.allReduce.allReduce(flatGrads, step);
      allReduceTimeMs = performance.now() - arStart;
      this.log(`🔄 All-reduce: ${allReduceTimeMs.toFixed(0)}ms across ${this.mesh.connectedPeerCount + 1} peers`);
    } else {
      averagedGradients = flatGrads;
    }

    // 5. Reconstruct and Apply Gradients
    const lr = this.getLearningRate(step);
    // There is no setLearningRate, but we can access optimizer.learningRate in some versions
    // If not, we just recreate the optimizer if LR changes significantly, or hack it.
    // For simplicity, we just use the global LR object.
    (this.optimizer as any).learningRate = lr;

    const newGrads: Record<string, tf.Tensor> = {};
    offset = 0;
    for (const v of vars) {
      const gData = averagedGradients.subarray(offset, offset + v.size);
      newGrads[v.name] = tf.tensor(gData, v.shape);
      offset += v.size;
    }

    this.optimizer.applyGradients(newGrads);

    // Cleanup
    for (const v of vars) { newGrads[v.name].dispose(); }
    lossTensor.dispose();
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
        const loss = await this.trainStep();
        const step = this.metrics.step;

        if (step % 5 === 0) {
          const peers = this.mesh.connectedPeerCount;
          const tps = this.metrics.tokensPerSec.toFixed(0);
          const elapsed = this.metrics.elapsedSeconds.toFixed(0);
          this.log(`Step ${step} | loss=${loss.toFixed(6)} | smooth=${this.metrics.smoothLoss.toFixed(6)} | ${tps} tok/s | peers=${peers} | ${elapsed}s`);
        }
      } catch (e) {
        this.log(`❌ Training error: ${e}`);
        await new Promise(r => setTimeout(r, 1000));
      }
    }
    this.log('🏁 Training stopped');
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

