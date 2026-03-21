import * as tf from '@tensorflow/tfjs';

// ---------------------------------------------------------------------------
// TokenDataLoader — streaming token loader from binary file
// ---------------------------------------------------------------------------

export class TokenDataLoader {
  private url: string;
  private vocabSize: number;
  private tokens: Int32Array = new Int32Array(0);
  private pos: number = 0;
  private epochs: number = 0;

  constructor(url: string, vocabSize: number) {
    this.url = url;
    this.vocabSize = vocabSize;
  }

  async load(): Promise<void> {
    const res = await fetch(this.url);
    if (!res.ok) throw new Error(`Failed to fetch ${this.url}: ${res.status}`);
    const buf = await res.arrayBuffer();
    this.tokens = new Int32Array(buf);
    this.pos = 0;
    this.epochs = 0;
  }

  get numTokens(): number {
    return this.tokens.length;
  }

  get currentEpoch(): number {
    return this.epochs;
  }

  nextBatch(batchSize: number, seqLen: number): { inputIds: Int32Array; targets: Int32Array; epoch: number } {
    const needed = batchSize * seqLen;
    const inputIds = new Int32Array(needed);
    const targets = new Int32Array(needed);

    for (let i = 0; i < needed; i++) {
      if (this.pos >= this.tokens.length) {
        this.pos = 0;
        this.epochs++;
      }
      inputIds[i] = this.tokens[this.pos];
      this.pos++;
      targets[i] = this.pos >= this.tokens.length ? this.tokens[0] : this.tokens[this.pos];
    }

    return { inputIds, targets, epoch: this.epochs };
  }

  /**
   * Get a fixed batch for validation (non-streaming, no epoch wrap).
   * Used by evaluate_bpb to get deterministic eval data.
   */
  getFixedBatch(batchSize: number, seqLen: number, startOffset: number): { inputIds: Int32Array; targets: Int32Array } {
    const needed = batchSize * seqLen;
    const inputIds = new Int32Array(needed);
    const targets = new Int32Array(needed);

    for (let i = 0; i < needed; i++) {
      const idx = (startOffset + i) % this.tokens.length;
      inputIds[i] = this.tokens[idx];
      targets[i] = this.tokens[(idx + 1) % this.tokens.length];
    }

    return { inputIds, targets };
  }
}

// ---------------------------------------------------------------------------
// Best-Fit Packing Dataloader (Karpathy's make_dataloader port)
// ---------------------------------------------------------------------------
// Karpathy's best-fit packing: every row starts with BOS, documents packed
// using best-fit to minimize cropping. When no doc fits, crops shortest doc.
// 100% utilization (no padding).

export interface Document {
  tokens: Int32Array;
}

export class BestFitPackingLoader {
  private documents: Document[];
  private docBuffer: Document[] = [];
  private bosToken: number;
  private rowCapacity: number;
  private bufferSize: number;
  private pos: number = 0;
  private epochs: number = 0;

  constructor(
    tokens: Int32Array,
    bosToken: number,
    seqLen: number,
    bufferSize: number = 1000
  ) {
    this.bosToken = bosToken;
    this.rowCapacity = seqLen + 1; // T+1 because we need T inputs + 1 target
    this.bufferSize = bufferSize;

    // Split tokens into documents at BOS boundaries
    this.documents = this.splitDocuments(tokens);
    this.refillBuffer();
  }

  private splitDocuments(tokens: Int32Array): Document[] {
    const docs: Document[] = [];
    let start = 0;
    for (let i = 1; i < tokens.length; i++) {
      if (tokens[i] === this.bosToken) {
        if (i > start) {
          docs.push({ tokens: tokens.slice(start, i) });
        }
        start = i;
      }
    }
    if (start < tokens.length) {
      docs.push({ tokens: tokens.slice(start) });
    }
    return docs;
  }

  private refillBuffer(): void {
    while (this.docBuffer.length < this.bufferSize) {
      if (this.pos >= this.documents.length) {
        this.pos = 0;
        this.epochs++;
      }
      this.docBuffer.push(this.documents[this.pos]);
      this.pos++;
    }
  }

  nextBatch(B: number, T: number): { inputIds: tf.Tensor2D; targets: tf.Tensor2D; epoch: number } {
    const rowCapacity = T + 1;
    const rowData = new Int32Array(B * rowCapacity);

    for (let rowIdx = 0; rowIdx < B; rowIdx++) {
      let pos = 0;
      const rowOffset = rowIdx * rowCapacity;

      while (pos < rowCapacity) {
        this.refillBuffer();
        const remaining = rowCapacity - pos;

        // Find largest doc that fits entirely
        let bestIdx = -1;
        let bestLen = 0;
        for (let i = 0; i < this.docBuffer.length; i++) {
          const docLen = this.docBuffer[i].tokens.length;
          if (docLen <= remaining && docLen > bestLen) {
            bestIdx = i;
            bestLen = docLen;
          }
        }

        if (bestIdx >= 0) {
          const doc = this.docBuffer.splice(bestIdx, 1)[0];
          rowData.set(doc.tokens, rowOffset + pos);
          pos += doc.tokens.length;
        } else {
          // No doc fits — crop shortest to fill remaining
          let shortestIdx = 0;
          let shortestLen = Infinity;
          for (let i = 0; i < this.docBuffer.length; i++) {
            if (this.docBuffer[i].tokens.length < shortestLen) {
              shortestIdx = i;
              shortestLen = this.docBuffer[i].tokens.length;
            }
          }
          const doc = this.docBuffer.splice(shortestIdx, 1)[0];
          rowData.set(doc.tokens.slice(0, remaining), rowOffset + pos);
          pos += remaining;
        }
      }
    }

    // Split into inputs (all but last) and targets (all but first)
    const inputsData = new Int32Array(B * T);
    const targetsData = new Int32Array(B * T);
    for (let b = 0; b < B; b++) {
      const rowOffset = b * rowCapacity;
      const inputOffset = b * T;
      for (let t = 0; t < T; t++) {
        inputsData[inputOffset + t] = rowData[rowOffset + t];
        targetsData[inputOffset + t] = rowData[rowOffset + t + 1];
      }
    }

    const inputIds = tf.tensor2d(inputsData, [B, T], 'int32') as tf.Tensor2D;
    const targets = tf.tensor2d(targetsData, [B, T], 'int32') as tf.Tensor2D;

    return { inputIds, targets, epoch: this.epochs };
  }
}

// ---------------------------------------------------------------------------
// Token Bytes Lookup — for BPB evaluation
// ---------------------------------------------------------------------------

let _tokenBytesCache: Int32Array | null = null;

/**
 * Load token byte lengths from server.
 * Format: JSON array of integers, one per token.
 * Special tokens have byte length 0.
 */
export async function loadTokenBytes(url: string = '/data/token_bytes.json'): Promise<Int32Array> {
  if (_tokenBytesCache) return _tokenBytesCache;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
    const arr = await res.json();
    _tokenBytesCache = new Int32Array(arr);
    return _tokenBytesCache;
  } catch (e) {
    console.warn(`Failed to load token_bytes from ${url}, falling back to estimated lengths`);
    // Fallback: assume all tokens have byte length 1
    _tokenBytesCache = new Int32Array(8192).fill(1);
    return _tokenBytesCache;
  }
}

// ---------------------------------------------------------------------------
// evaluate_bpb — Karpathy's vocab-size-independent metric
// ---------------------------------------------------------------------------
// BPB = bits per byte
// Sums per-token cross-entropy (in nats), sums target byte lengths,
// then converts nats/byte to bits/byte.
// Special tokens (byte length 0) are excluded from both sums.

export interface BPBModel {
  forward(inputIds: tf.Tensor, targets: tf.Tensor, reduction: 'none'): tf.Tensor;
}

export async function evaluateBPB(
  model: BPBModel,
  valLoader: TokenDataLoader,
  batchSize: number,
  seqLen: number,
  evalTokens: number
): Promise<number> {
  const tokenBytes = await loadTokenBytes();
  const steps = Math.floor(evalTokens / (batchSize * seqLen));

  let totalNats = 0.0;
  let totalBytes = 0;

  const valStartOffset = Math.floor(valLoader.numTokens * 0.9); // last 10% as val

  for (let step = 0; step < steps; step++) {
    const { inputIds, targets } = valLoader.getFixedBatch(batchSize, seqLen, valStartOffset + step * batchSize * seqLen);

    const inputTensor = tf.tensor2d(inputIds, [batchSize, seqLen], 'int32');
    const targetTensor = tf.tensor2d(targets, [batchSize, seqLen], 'int32');

    // Forward with reduction='none' to get per-token loss
    const lossTensor = model.forward(inputTensor, targetTensor, 'none') as tf.Tensor2D;
    const lossFlat = lossTensor.reshape([-1]);
    const yFlat = targetTensor.reshape([-1]);

    // Look up byte lengths
    const yData = await yFlat.data() as Int32Array;
    const lossData = await lossFlat.data() as Float32Array;

    for (let i = 0; i < yData.length; i++) {
      const nbytes = tokenBytes[yData[i]] || 0;
      if (nbytes > 0) {
        totalNats += lossData[i];
        totalBytes += nbytes;
      }
    }

    // Cleanup
    inputTensor.dispose();
    targetTensor.dispose();
    lossTensor.dispose();
    lossFlat.dispose();
    yFlat.dispose();
  }

  // Convert nats/byte to bits/byte
  return totalNats / (Math.log(2) * totalBytes);
}
