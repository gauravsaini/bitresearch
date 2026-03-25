/**
 * Top-K Gradient Sparsification with Error Feedback
 *
 * Reduces WebRTC gradient bandwidth by transmitting only the top-K% of
 * gradient magnitudes. Residuals (zeroed-out gradients) are accumulated
 * in an error feedback buffer and added to the next step's gradients,
 * guaranteeing eventual convergence.
 *
 * Reference: "Deep Gradient Compression" (Lin et al., 2018)
 *
 * Features:
 *   - Configurable sparsity ratio (default 90% sparse = 10% transmitted)
 *   - Error feedback buffer for lossless convergence
 *   - Momentum correction (optional)
 *   - Warm-up period with dense gradients for stable early training
 *   - Efficient approximate top-k via random sampling (O(N) instead of O(N log N))
 *   - Sparse encoding/decoding for wire format (index + value pairs)
 */

export interface SparsificationConfig {
  /** Fraction of gradients to KEEP (0.0 to 1.0). Default: 0.10 (keep top 10%) */
  topKRatio: number;
  /** Number of warm-up steps with dense gradients. Default: 100 */
  warmupSteps: number;
  /** Sample size for approximate threshold finding. Default: 10000 */
  sampleSize: number;
  /** Momentum factor for error feedback. 0 = no momentum. Default: 0.9 */
  momentumFactor: number;
}

const DEFAULT_CONFIG: SparsificationConfig = {
  topKRatio: 0.10,
  warmupSteps: 100,
  sampleSize: 10000,
  momentumFactor: 0.9,
};

/** Sparse gradient: indices + values (much smaller than dense) */
export interface SparseGradient {
  /** Original dense length */
  length: number;
  /** Indices of non-zero elements */
  indices: Uint32Array;
  /** Values at those indices */
  values: Float32Array;
}

export class GradientSparsifier {
  private config: SparsificationConfig;
  private errorBuffer: Float32Array | null = null;
  private velocityBuffer: Float32Array | null = null;
  private step = 0;

  constructor(config: Partial<SparsificationConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    if (this.config.topKRatio <= 0 || this.config.topKRatio > 1) {
      throw new Error(`topKRatio must be in (0, 1], got ${this.config.topKRatio}`);
    }
  }

  /** Reset internal state */
  reset(): void {
    this.errorBuffer = null;
    this.velocityBuffer = null;
    this.step = 0;
  }

  /**
   * Whether this step should use dense (unsparsified) gradients.
   * During warmup, all gradients are sent dense for training stability.
   */
  get isDenseStep(): boolean {
    return this.step < this.config.warmupSteps;
  }

  /**
   * Sparsify gradients in-place using top-K with error feedback.
   *
   * @param gradients Dense gradient array (modified in-place: non-top-K values zeroed)
   * @returns SparseGradient with only the top-K entries
   */
  sparsify(gradients: Float32Array): SparseGradient {
    const N = gradients.length;
    this.step++;

    // During warmup, send everything
    if (this.isDenseStep) {
      const indices = new Uint32Array(N);
      for (let i = 0; i < N; i++) indices[i] = i;
      return { length: N, indices, values: new Float32Array(gradients) };
    }

    // Initialize buffers on first sparse step
    if (!this.errorBuffer || this.errorBuffer.length !== N) {
      this.errorBuffer = new Float32Array(N);
    }
    if (this.config.momentumFactor > 0 && (!this.velocityBuffer || this.velocityBuffer.length !== N)) {
      this.velocityBuffer = new Float32Array(N);
    }

    // Step 1: Add error feedback from previous step
    const errorBuf = this.errorBuffer;
    for (let i = 0; i < N; i++) {
      gradients[i] += errorBuf[i];
    }

    // Step 2: Optional momentum correction
    if (this.velocityBuffer && this.config.momentumFactor > 0) {
      const mu = this.config.momentumFactor;
      const vel = this.velocityBuffer;
      for (let i = 0; i < N; i++) {
        vel[i] = mu * vel[i] + gradients[i];
        gradients[i] = vel[i];
      }
    }

    // Step 3: Find approximate top-K threshold via random sampling
    const threshold = this.findThreshold(gradients, N);

    // Step 4: Select top-K entries and build sparse representation
    const kExpected = Math.max(1, Math.ceil(N * this.config.topKRatio));
    const selectedIndices: number[] = [];
    const selectedValues: number[] = [];

    for (let i = 0; i < N; i++) {
      if (Math.abs(gradients[i]) >= threshold) {
        selectedIndices.push(i);
        selectedValues.push(gradients[i]);
        // Clear error for transmitted values
        errorBuf[i] = 0;
      } else {
        // Accumulate error for next step
        errorBuf[i] = gradients[i];
        // Zero out in the dense array
        gradients[i] = 0;
      }
    }

    return {
      length: N,
      indices: Uint32Array.from(selectedIndices),
      values: Float32Array.from(selectedValues),
    };
  }

  /**
   * Densify a sparse gradient back to a full array.
   */
  static densify(sparse: SparseGradient): Float32Array {
    const dense = new Float32Array(sparse.length);
    for (let i = 0; i < sparse.indices.length; i++) {
      dense[sparse.indices[i]] = sparse.values[i];
    }
    return dense;
  }

  /**
   * Encode a sparse gradient for wire transfer.
   * Format: [length(u32) | numEntries(u32) | indices(u32[]) | values(f32[])]
   */
  static encode(sparse: SparseGradient): ArrayBuffer {
    const numEntries = sparse.indices.length;
    const headerSize = 8; // 2 x u32
    const dataSize = numEntries * (4 + 4); // u32 index + f32 value per entry
    const buf = new ArrayBuffer(headerSize + dataSize);
    const view = new DataView(buf);

    view.setUint32(0, sparse.length, true);
    view.setUint32(4, numEntries, true);

    const indicesOffset = headerSize;
    const valuesOffset = headerSize + numEntries * 4;

    new Uint32Array(buf, indicesOffset, numEntries).set(sparse.indices);
    new Float32Array(buf, valuesOffset, numEntries).set(sparse.values);

    return buf;
  }

  /**
   * Decode a wire-format sparse gradient.
   */
  static decode(buf: ArrayBuffer): SparseGradient {
    const view = new DataView(buf);
    const length = view.getUint32(0, true);
    const numEntries = view.getUint32(4, true);

    const headerSize = 8;
    const indices = new Uint32Array(buf, headerSize, numEntries);
    const values = new Float32Array(buf, headerSize + numEntries * 4, numEntries);

    return { length, indices: new Uint32Array(indices), values: new Float32Array(values) };
  }

  /**
   * Compression ratio for the last sparsification.
   * Returns ratio of sparse size vs dense size (lower = more compression).
   */
  static compressionRatio(sparse: SparseGradient): number {
    const denseBytes = sparse.length * 4; // f32
    const sparseBytes = 8 + sparse.indices.length * 8; // header + (u32 + f32) per entry
    return sparseBytes / denseBytes;
  }

  // ── Internal ──

  /**
   * Find approximate threshold for top-K using random sampling.
   * O(sampleSize * log(sampleSize)) instead of O(N * log(N)).
   */
  private findThreshold(gradients: Float32Array, N: number): number {
    const sampleSize = Math.min(N, this.config.sampleSize);
    const samples = new Float32Array(sampleSize);

    // Random sample of absolute values
    for (let i = 0; i < sampleSize; i++) {
      const idx = Math.floor(Math.random() * N);
      samples[i] = Math.abs(gradients[idx]);
    }

    // Sort ascending
    samples.sort();

    // Threshold = value at (1 - topKRatio) percentile
    const cutoffIndex = Math.floor(sampleSize * (1.0 - this.config.topKRatio));
    return samples[Math.min(cutoffIndex, sampleSize - 1)];
  }
}
