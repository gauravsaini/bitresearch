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
import { GradientSparsifier, type SparseGradient, type SparsificationConfig } from "./gradient-sparsifier";

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
  minRingSize: number;      // pause training if connected peers < this
  checkpointInterval: number; // save checkpoint every N steps (0 = disabled)
  dataUrl: string;           // URL to fetch tokenized data (empty = synthetic)
  artifactBaseUrl?: string;  // stable bundle root for parity-prepared artifacts
  manifestUrl?: string;      // optional manifest URL for parity-prepared artifacts
  documentsUrl?: string;     // optional explicit document payload URL
  dataFormat?: 'auto' | 'tokens' | 'documents';
  documentBufferSize?: number;
  documentsAreBOSPrefixed?: boolean;
  bosTokenId?: number;
  tokenBytesUrl?: string;    // optional URL for token byte lengths (enables exact BPB)
  // Eval
  evalTokens: number;        // tokens for BPB evaluation
  // Gradient sparsification
  sparsification?: Partial<SparsificationConfig>;
  enableSparsification?: boolean;
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
  stepTimeMs: number;
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
  artifactBaseUrl: '',
  manifestUrl: '',
  documentsUrl: '',
  dataFormat: 'auto',
  documentBufferSize: 1000,
  documentsAreBOSPrefixed: false,
  bosTokenId: DEFAULT_CONFIG.vocabSize,
  tokenBytesUrl: '',
  evalTokens: 40 * 524288, // ~20M tokens for eval (matches Karpathy)
  enableSparsification: false,
  sparsification: { topKRatio: 0.10, warmupSteps: 100, sampleSize: 10000, momentumFactor: 0.9 },
};

// H100 BF16 peak FLOPs (for MFU calculation)
const H100_BF16_PEAK_FLOPS = 989.5e12;

interface ParityArtifactManifest {
  parityUrl?: string;
  binUrl?: string;
  tokenizerUrl?: string;
  files?: {
    bin?: string[];
    tokenizer?: string[];
  };
  source?: {
    binDir?: string;
    tokenizerDir?: string;
  };
}

export class DistributedTrainer {
  config: TrainerConfig;
  private model: GPTModel | null = null;
  private optimizer: MuonAdamWOptimizer | null = null;
  private mesh: WebRTCMesh;
  private allReduce: RingAllReduce | null = null;
  private dataLoader: TokenDataLoader | null = null;
  private valTokens: Int32Array | null = null;
  private tokenBytesTensor: tf.Tensor1D | null = null;
  private parityManifest: ParityArtifactManifest | null = null;
  private running = false;
  private startTime = 0;
  valBpb: number = 0;
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
    stepTimeMs: 0,
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

  private hasValue(value?: string): boolean {
    return Boolean(value && value.trim());
  }

  private normalizeUrl(value: string): string {
    return value.trim();
  }

  private makeUrlCandidates(baseUrl: string, names: string[]): string[] {
    const candidates: string[] = [];
    for (const name of names) {
      try {
        candidates.push(new URL(name, baseUrl).toString());
      } catch {
        // Ignore malformed URLs and keep the candidate list best-effort.
      }
    }
    return candidates;
  }

  private getArtifactBaseCandidates(): string[] {
    const candidates = new Set<string>();
    const explicitBase = this.config.artifactBaseUrl?.trim();
    if (explicitBase) {
      candidates.add(explicitBase);
    }
    return Array.from(candidates);
  }

  private getManifestUrlCandidates(): string[] {
    const candidates = new Set<string>();

    const explicitManifest = this.config.manifestUrl?.trim();
    if (explicitManifest) {
      candidates.add(this.normalizeUrl(explicitManifest));
    }

    for (const baseUrl of this.getArtifactBaseCandidates()) {
      for (const candidate of this.makeUrlCandidates(baseUrl, ['manifest.json'])) {
        candidates.add(candidate);
      }
    }

    return Array.from(candidates);
  }

  private resolveUrl(baseUrl: string, relativePath: string): string {
    return new URL(relativePath, baseUrl).toString();
  }

  private getManifestBinCandidates(manifest: ParityArtifactManifest): string[] {
    const candidates = new Set<string>();
    const baseUrl = manifest.binUrl?.trim()
      || (manifest.parityUrl?.trim() ? this.resolveUrl(manifest.parityUrl.trim(), 'bin/') : null)
      || (this.config.artifactBaseUrl?.trim() ? this.resolveUrl(this.config.artifactBaseUrl.trim(), 'bin/') : null);

    if (!baseUrl) {
      return [];
    }

    for (const fileName of manifest.files?.bin ?? []) {
      if (!fileName.endsWith('.bin')) continue;
      try {
        candidates.add(this.resolveUrl(baseUrl, fileName));
      } catch {
        // Ignore malformed manifest paths and continue with the rest.
      }
    }

    return Array.from(candidates);
  }

  private getManifestTokenBytesCandidates(manifest: ParityArtifactManifest): string[] {
    const candidates = new Set<string>();
    const tokenizerBase = manifest.tokenizerUrl?.trim()
      || (manifest.parityUrl?.trim() ? this.resolveUrl(manifest.parityUrl.trim(), 'tokenizer/') : null)
      || (this.config.artifactBaseUrl?.trim() ? this.resolveUrl(this.config.artifactBaseUrl.trim(), 'tokenizer/') : null);
    const binBase = manifest.binUrl?.trim()
      || (manifest.parityUrl?.trim() ? this.resolveUrl(manifest.parityUrl.trim(), 'bin/') : null)
      || (this.config.artifactBaseUrl?.trim() ? this.resolveUrl(this.config.artifactBaseUrl.trim(), 'bin/') : null);

    for (const baseUrl of [tokenizerBase, binBase]) {
      if (!baseUrl) continue;
      for (const candidate of this.makeUrlCandidates(baseUrl, ['token_bytes.bin', 'token-bytes.bin', 'token_bytes.json'])) {
        candidates.add(candidate);
      }
    }

    return Array.from(candidates);
  }

  private async loadParityManifest(): Promise<ParityArtifactManifest | null> {
    if (this.parityManifest) {
      return this.parityManifest;
    }

    const candidates = this.getManifestUrlCandidates();
    if (candidates.length === 0) {
      return null;
    }

    for (const url of candidates) {
      try {
        const res = await fetch(url);
        if (!res.ok) {
          continue;
        }
        const manifest = await res.json() as ParityArtifactManifest;
        this.log(`🗺️ Loaded parity manifest from ${url}`);
        this.parityManifest = manifest;
        return manifest;
      } catch {
        // Try the next manifest candidate.
      }
    }

    return null;
  }

  private getDataUrlCandidates(): string[] {
    const candidates = new Set<string>();
    const dataFormat = this.config.dataFormat ?? 'auto';
    const docsFirst = dataFormat !== 'tokens';
    const rootNames = docsFirst
      ? ['documents.json', 'docs.json', 'tokens.json', 'tokens.bin']
      : ['tokens.bin', 'tokens.json', 'documents.json', 'docs.json'];
    const binNames = docsFirst
      ? ['documents.json', 'docs.json', 'tokens.json', 'tokens.bin', 'train.bin', 'train_tokens.bin']
      : ['tokens.bin', 'tokens.json', 'documents.json', 'docs.json', 'train.bin', 'train_tokens.bin'];

    const explicitDocumentsUrl = this.config.documentsUrl?.trim();
    if (explicitDocumentsUrl) {
      candidates.add(this.normalizeUrl(explicitDocumentsUrl));
    }

    for (const baseUrl of this.getArtifactBaseCandidates()) {
      for (const candidate of this.makeUrlCandidates(baseUrl, rootNames)) {
        candidates.add(candidate);
      }
      for (const candidate of this.makeUrlCandidates(this.resolveUrl(baseUrl, 'bin/'), binNames)) {
        candidates.add(candidate);
      }
    }

    const explicitDataUrl = this.config.dataUrl?.trim();
    if (explicitDataUrl) {
      candidates.add(this.normalizeUrl(explicitDataUrl));
    }

    return Array.from(candidates);
  }

  private getTokenBytesUrlCandidates(): string[] {
    const candidates = new Set<string>();

    const explicit = this.config.tokenBytesUrl?.trim();
    if (explicit) {
      candidates.add(this.normalizeUrl(explicit));
    }

    const sidecarNames = ['token_bytes.bin', 'token-bytes.bin', 'token_bytes.json'];
    for (const baseUrl of this.getArtifactBaseCandidates()) {
      for (const candidate of this.makeUrlCandidates(this.resolveUrl(baseUrl, 'tokenizer/'), sidecarNames)) {
        candidates.add(candidate);
      }
      for (const candidate of this.makeUrlCandidates(baseUrl, sidecarNames)) {
        candidates.add(candidate);
      }
      for (const candidate of this.makeUrlCandidates(this.resolveUrl(baseUrl, 'bin/'), sidecarNames)) {
        candidates.add(candidate);
      }
    }

    if (this.config.documentsUrl?.trim()) {
      try {
        const baseHref = typeof window !== 'undefined' ? window.location.href : 'http://localhost/';
        const documentsUrl = new URL(this.config.documentsUrl.trim(), baseHref);
        for (const candidate of this.makeUrlCandidates(documentsUrl.toString(), sidecarNames)) {
          candidates.add(candidate);
        }
      } catch {
        // Ignore malformed URLs and fall back to other discovery paths.
      }
    }

    if (this.config.dataUrl?.trim()) {
      try {
        const baseHref = typeof window !== 'undefined' ? window.location.href : 'http://localhost/';
        const dataUrl = new URL(this.config.dataUrl.trim(), baseHref);
        for (const candidate of this.makeUrlCandidates(dataUrl.toString(), sidecarNames)) {
          candidates.add(candidate);
        }
      } catch {
        // Ignore malformed URLs and fall back to explicit configuration only.
      }
    }

    return Array.from(candidates);
  }

  private flattenDocuments(documents: Int32Array[]): Int32Array {
    let total = 0;
    for (const doc of documents) {
      total += doc.length;
    }

    const flattened = new Int32Array(total);
    let offset = 0;
    for (const doc of documents) {
      flattened.set(doc, offset);
      offset += doc.length;
    }
    return flattened;
  }

  private normalizeTokenByteLengths(lengths: ArrayLike<number>): Int32Array {
    const vocabSize = this.config.modelConfig.vocabSize;
    const normalized = new Int32Array(vocabSize);
    const limit = Math.min(vocabSize, lengths.length);
    for (let i = 0; i < limit; i++) {
      normalized[i] = Math.max(0, Math.trunc(lengths[i]));
    }
    if (lengths.length !== vocabSize) {
      this.log(`⚠️ token byte table length ${lengths.length} does not match vocab size ${vocabSize}; padded/truncated for BPB.`);
    }
    return normalized;
  }

  private async loadTokenBytesTable(): Promise<void> {
    const manifest = await this.loadParityManifest();
    const orderedCandidates: string[] = [];
    const pushUnique = (urls: string[]) => {
      for (const url of urls) {
        if (!orderedCandidates.includes(url)) {
          orderedCandidates.push(url);
        }
      }
    };

    pushUnique(this.config.tokenBytesUrl?.trim() ? [this.normalizeUrl(this.config.tokenBytesUrl.trim())] : []);
    if (manifest) {
      pushUnique(this.getManifestTokenBytesCandidates(manifest));
    }
    pushUnique(this.getTokenBytesUrlCandidates());

    if (orderedCandidates.length === 0) {
      return;
    }

    const shouldLogFailures = Boolean(this.config.tokenBytesUrl?.trim());

    for (const url of orderedCandidates) {
      try {
        const res = await fetch(url);
        if (!res.ok) {
          if (shouldLogFailures) {
            this.log(`⚠️ Could not load token byte lengths from ${url}: HTTP ${res.status}`);
          }
          continue;
        }

        let lengths: Int32Array | null = null;
        const contentType = res.headers.get('content-type') || '';
        if (url.endsWith('.json') || contentType.includes('json')) {
          const json = await res.json();
          const arr = Array.isArray(json) ? json : Array.isArray((json as any)?.data) ? (json as any).data : null;
          if (!arr) {
            throw new Error(`Unsupported token bytes JSON format at ${url}`);
          }
          lengths = this.normalizeTokenByteLengths(arr as ArrayLike<number>);
        } else {
          const buf = await res.arrayBuffer();
          if (buf.byteLength === 0) {
            throw new Error(`Empty token byte table at ${url}`);
          }
          if (buf.byteLength % 4 !== 0) {
            throw new Error(`Token byte table at ${url} is not 32-bit aligned (${buf.byteLength} bytes)`);
          }
          lengths = this.normalizeTokenByteLengths(new Int32Array(buf));
        }

        if (this.tokenBytesTensor) {
          this.tokenBytesTensor.dispose();
        }
        this.tokenBytesTensor = tf.tensor1d(lengths, 'int32');
        this.log(`🧮 Loaded token byte lengths from ${url} for exact BPB evaluation`);
        return;
      } catch (e) {
        if (shouldLogFailures) {
          this.log(`⚠️ Could not load token byte lengths from ${url}: ${e}`);
        }
      }
    }
  }

  private async loadTokenStreamFromUrls(urls: string[]): Promise<Int32Array> {
    const chunks: Int32Array[] = [];
    let total = 0;

    for (const url of urls) {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Failed to fetch parity shard ${url}: HTTP ${res.status}`);
      }

      const buf = await res.arrayBuffer();
      if (buf.byteLength === 0) {
        continue;
      }
      if (buf.byteLength % 4 !== 0) {
        throw new Error(`Parity shard ${url} is not 32-bit aligned (${buf.byteLength} bytes)`);
      }

      const chunk = new Int32Array(buf);
      chunks.push(chunk);
      total += chunk.length;
    }

    const merged = new Int32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  }

  private async loadTrainingDataFromManifest(manifest: ParityArtifactManifest): Promise<boolean> {
    const binCandidates = this.getManifestBinCandidates(manifest);
    if (binCandidates.length === 0) {
      return false;
    }

    const vocabSize = this.config.modelConfig.vocabSize;
    const loaderOptions = {
      bosTokenId: this.config.bosTokenId ?? vocabSize,
      documentBufferSize: this.config.documentBufferSize,
      documentsAreBOSPrefixed: this.config.documentsAreBOSPrefixed,
    };

    const stream = await this.loadTokenStreamFromUrls(binCandidates);
    const loader = new TokenDataLoader('', vocabSize, loaderOptions);
    loader.setTokenStream(stream);
    this.dataLoader = loader;
    this.valTokens = loader.valSplit(0.1);
    this.log(
      `📦 Loaded parity shard stream from manifest ${this.config.manifestUrl || this.config.artifactBaseUrl || manifest.parityUrl || '/data/parity/manifest.json'} — ${binCandidates.length} shard files, ${(loader.numTokens / 1e3).toFixed(0)}K train, ${(this.valTokens.length / 1e3).toFixed(0)}K val`,
    );
    return true;
  }

  private async loadTrainingDataFromCandidates(): Promise<void> {
    const candidates = this.getDataUrlCandidates();
    const hasAnyConfiguredSource =
      this.hasValue(this.config.documentsUrl) ||
      this.hasValue(this.config.artifactBaseUrl) ||
      this.hasValue(this.config.dataUrl);
    if (candidates.length === 0 || !hasAnyConfiguredSource) {
      return;
    }

    const vocabSize = this.config.modelConfig.vocabSize;
    const loaderOptions = {
      bosTokenId: this.config.bosTokenId ?? vocabSize,
      documentBufferSize: this.config.documentBufferSize,
      documentsAreBOSPrefixed: this.config.documentsAreBOSPrefixed,
    };

    for (const url of candidates) {
      try {
        const loader = new TokenDataLoader(url, vocabSize, loaderOptions);
        await loader.load({ format: 'auto' });

        try {
          const { trainDocuments, valDocuments } = loader.splitDocuments(0.1);
          const effectiveTrainDocuments = trainDocuments.length > 0 ? trainDocuments : valDocuments;
          const trainLoader = new TokenDataLoader('', vocabSize, {
            ...loaderOptions,
            documentsAreBOSPrefixed: true,
          });
          trainLoader.setDocuments(effectiveTrainDocuments);
          this.dataLoader = trainLoader;
          this.valTokens = valDocuments.length > 0 ? this.flattenDocuments(valDocuments) : new Int32Array(0);
          if (trainDocuments.length === 0 && valDocuments.length > 0) {
            this.log(`⚠️ Document split from ${url} left no train docs; using the validation side for training.`);
          }
          this.log(`📦 Loaded document-aware training data from ${url} — ${effectiveTrainDocuments.length} training docs used, ${valDocuments.length} val docs`);
          return;
        } catch {
          this.dataLoader = loader;
          this.valTokens = loader.valSplit(0.1);
          this.log(`📦 Loaded token stream from ${url} — ${(loader.numTokens / 1e3).toFixed(0)}K train, ${(this.valTokens.length / 1e3).toFixed(0)}K val`);
          return;
        }
      } catch {
        // Fall through to the next candidate.
      }
    }

    this.log('⚠️ No configured data artifact could be loaded; falling back to synthetic batches');
  }

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

    // Load parity-sharded data first, then fall back to legacy single-file sources.
    const manifest = await this.loadParityManifest();
    if (manifest && await this.loadTrainingDataFromManifest(manifest)) {
      // Manifest-driven parity bundle loaded.
    } else {
      await this.loadTrainingDataFromCandidates();
    }

    await this.loadTokenBytesTable();

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

  private gradientSparsifier: GradientSparsifier | null = null;
  

  private applyTopKSparsification(grad: Float32Array): void {
    if (!this.gradientSparsifier) {
      this.gradientSparsifier = new GradientSparsifier(this.config.sparsification);
    }
    // Sparsify in-place (zeroes non-top-K values, stores residuals internally)
    const sparse = this.gradientSparsifier.sparsify(grad);
    const ratio = GradientSparsifier.compressionRatio(sparse);
    if (this.metrics.step % 50 === 0) {
      this.log(`📦 Gradient sparsification: ${sparse.indices.length}/${sparse.length} values kept (${(ratio * 100).toFixed(1)}% of dense)`);
    }
  }

  async trainStep(): Promise<number> {
    if (!this.model || !this.optimizer) throw new Error('Not initialized');

    const B = this.config.batchSize;
    const T = this.config.modelConfig.sequenceLen;
    const step = this.metrics.step;
    const stepStart = performance.now();
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
    const gradMap = scaledGrads as Record<string, tf.Tensor>;
    const gradTensors = Object.values(gradMap);
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
      const g = gradMap[v.name];
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
        // All-reduce was aborted (peer failure mid-step)
        this.log(`⚠️ All-reduce aborted at step ${step}, skipping optimizer update`);
        // Dispose tensors
        for (const v of vars) { gradMap[v.name]?.dispose(); }
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
    const stepTimeMs = performance.now() - stepStart;
    this.metrics.stepTimeMs = stepTimeMs;
    this.metrics.tokensPerSec = tokensProcessed / (stepTimeMs / 1000);
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

  /**
   * Evaluate validation bits per byte (BPB).
   * Runs forward-only passes on held-out validation tokens and computes byte-weighted
   * cross-entropy. When a token byte-length table is available, this mirrors Karpathy's
   * evaluate_bpb() semantics exactly: sum nats, sum bytes, then convert nats/byte to bits/byte.
   * Otherwise it falls back to the legacy avgBytesPerToken approximation.
   */
  async evaluateBpb(batchSize?: number, numBatches = 10, avgBytesPerToken = 1.0): Promise<number> {
    if (!this.model || !this.valTokens || this.valTokens.length === 0) {
      return 0;
    }

    const B = batchSize ?? this.config.batchSize;
    const T = this.config.modelConfig.sequenceLen;
    const tokensPerBatch = B * T;
    const totalTokens = this.valTokens.length;
    if (tokensPerBatch <= 0 || totalTokens <= 1) return 0;

    const hasExactTokenBytes = this.tokenBytesTensor !== null;
    const tokenBytesTensor = this.tokenBytesTensor;
    let totalNats = 0;
    let totalBytes = 0;
    let offset = 0;

    for (let b = 0; b < numBatches; b++) {
      const { inputIds, targets } = TokenDataLoader.batchFromArray(this.valTokens, B, T, offset);
      offset = (offset + tokensPerBatch) % totalTokens;

      const inputTensor = tf.tensor2d(inputIds, [B, T], 'int32');
      const targetTensor = tf.tensor2d(targets, [B, T], 'int32');

      const batchStats = tf.tidy(() => {
        const forward = this.model!.forward(inputTensor, targetTensor, true);
        const logits = forward.logits as tf.Tensor2D;
        const flatTargets = targetTensor.reshape([-1]);
        const validMask = flatTargets.notEqual(-1);
        const safeTargets = tf.maximum(flatTargets, 0).toInt();
        const tokenIndices = tf.range(0, safeTargets.size, 1, 'int32');
        const gatherIndices = tf.stack([tokenIndices, safeTargets], 1);
        const logProbs = tf.logSoftmax(logits, -1);
        const perTokenNats = tf.neg(tf.gatherND(logProbs, gatherIndices));
        const validMaskF = tf.cast(validMask, 'float32');

        if (hasExactTokenBytes && tokenBytesTensor) {
          const byteLengths = tf.gather(tokenBytesTensor, safeTargets).toFloat();
          const byteMask = tf.logicalAnd(validMask, byteLengths.greater(0));
          const byteMaskF = tf.cast(byteMask, 'float32');
          return {
            totalNatsTensor: perTokenNats.mul(byteMaskF).sum(),
            totalBytesTensor: byteLengths.mul(byteMaskF).sum(),
            exact: true,
          };
        }

        return {
          totalNatsTensor: perTokenNats.mul(validMaskF).sum(),
          totalBytesTensor: tf.scalar(avgBytesPerToken, 'float32').mul(validMaskF.sum()),
          exact: false,
        };
      });

      const [batchNats] = await batchStats.totalNatsTensor.data();
      const [batchBytes] = await batchStats.totalBytesTensor.data();
      totalNats += batchNats;
      totalBytes += batchBytes;
      batchStats.totalNatsTensor.dispose();
      batchStats.totalBytesTensor.dispose();
      inputTensor.dispose();
      targetTensor.dispose();
    }

    if (totalBytes <= 0) {
      return 0;
    }

    this.valBpb = totalNats / (Math.log(2) * totalBytes);
    if (hasExactTokenBytes) {
      this.log(
        `📊 val_bpb: ${this.valBpb.toFixed(6)} (exact bytes, total_nats=${totalNats.toFixed(6)}, total_bytes=${totalBytes.toFixed(0)}, ${numBatches} batches)`,
      );
    } else {
      this.log(
        `📊 val_bpb: ${this.valBpb.toFixed(6)} (fallback avgBytesPerToken=${avgBytesPerToken}, total_nats=${totalNats.toFixed(6)}, total_bytes=${totalBytes.toFixed(0)}, ${numBatches} batches)`,
      );
    }
    return this.valBpb;
  }

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


    // ── Auto-log results to results.tsv ──
    if (typeof (globalThis as any).process !== 'undefined') {
      try {
        const { ResultsLogger } = await import('../train/results-logger');
        const logger = new ResultsLogger();
        logger.log({
          valBpb: this.metrics.valBpb,
          peakMemoryMb: peakVramMb,
          status: this.metrics.valBpb > 0 ? 'keep' : 'crash',
          description: `step ${this.metrics.step} | ${this.metrics.numParamsM.toFixed(1)}M params | depth=${this.config.modelConfig.nLayer}`,
          trainingSeconds: this.totalTrainingTime,
          totalSeconds: totalSeconds,
          mfuPercent: this.metrics.mfuPercent,
          totalTokensM: this.metrics.totalTokens / 1e6,
          numSteps: this.metrics.step,
          numParamsM: this.metrics.numParamsM,
          depth: this.config.modelConfig.nLayer,
        });
        this.log('📝 Results logged to results.tsv');
      } catch (e) {
        this.log(`⚠️ Could not log results to results.tsv: ${e}`);
      }
    }

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

    // Restore model weights
    for (const [varName, varObj] of Object.entries(varsByName) as Array<[string, tf.Variable]>) {
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
    if (this.tokenBytesTensor) {
      this.tokenBytesTensor.dispose();
      this.tokenBytesTensor = null;
    }
    this.mesh.disconnect();
    this.optimizer?.dispose();
  }
}
