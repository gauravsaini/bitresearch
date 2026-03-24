import * as tf from '@tensorflow/tfjs';
import {
  BestFitDocumentBatcher,
  type DocumentInput,
  type PackedBatch,
  normalizeDocuments,
  prependToken,
} from './bestFitPacking';

export { BestFitDocumentBatcher, normalizeDocuments, prependToken } from './bestFitPacking';
export type { DocumentInput, PackedBatch } from './bestFitPacking';

export interface TokenBatch {
  inputIds: Int32Array;
  targets: Int32Array;
  epoch: number;
}

export interface TokenDataLoaderOptions {
  bosTokenId?: number;
  documentBufferSize?: number;
  documentsAreBOSPrefixed?: boolean;
}

export interface LoadOptions {
  format?: 'auto' | 'tokens' | 'documents';
}

export interface DocumentSplit {
  trainDocuments: Int32Array[];
  valDocuments: Int32Array[];
}

function cloneTokens(tokens: ArrayLike<number>): Int32Array {
  return tokens instanceof Int32Array ? new Int32Array(tokens) : Int32Array.from(tokens);
}

function inferJsonDocuments(payload: unknown): Int32Array[] {
  if (Array.isArray(payload)) {
    if (payload.length === 0) {
      return [];
    }

    const first = payload[0];
    if (typeof first === 'number') {
      throw new Error('Expected an array of documents, not a flat token array.');
    }

    if (Array.isArray(first) || ArrayBuffer.isView(first as ArrayBufferView)) {
      return normalizeDocuments(payload as DocumentInput);
    }
  }

  if (payload && typeof payload === 'object') {
    const record = payload as Record<string, unknown>;
    if (Array.isArray(record.documents)) {
      return inferJsonDocuments(record.documents);
    }
  }

  throw new Error('Unsupported documents payload. Expected an array of token arrays.');
}

function inferJsonTokens(payload: unknown): Int32Array {
  if (Array.isArray(payload)) {
    if (payload.length === 0) {
      return new Int32Array(0);
    }

    const first = payload[0];
    if (typeof first === 'number') {
      return Int32Array.from(payload as number[]);
    }

    if (Array.isArray(first) || ArrayBuffer.isView(first as ArrayBufferView)) {
      const docs = normalizeDocuments(payload as DocumentInput);
      let total = 0;
      for (const doc of docs) {
        total += doc.length;
      }

      const tokens = new Int32Array(total);
      let offset = 0;
      for (const doc of docs) {
        tokens.set(doc, offset);
        offset += doc.length;
      }
      return tokens;
    }
  }

  if (payload && typeof payload === 'object') {
    const record = payload as Record<string, unknown>;
    if (Array.isArray(record.tokens)) {
      return inferJsonTokens(record.tokens);
    }
  }

  throw new Error('Unsupported token payload. Expected a flat token array.');
}

async function readJsonPayload(url: string): Promise<unknown> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.json();
}

export class TokenDataLoader {
  private url: string;
  private vocabSize: number;
  private bosTokenId: number;
  private documentBufferSize: number;
  private documentsAreBOSPrefixed: boolean;
  private tokens: Int32Array = new Int32Array(0);
  private documents: Int32Array[] | null = null;
  private docBatcher: BestFitDocumentBatcher | null = null;
  private pos: number = 0;
  private epochs: number = 0;

  constructor(url: string, vocabSize: number, options: TokenDataLoaderOptions = {}) {
    this.url = url;
    this.vocabSize = vocabSize;
    this.bosTokenId = options.bosTokenId ?? vocabSize;
    this.documentBufferSize = Math.max(1, options.documentBufferSize ?? 1000);
    this.documentsAreBOSPrefixed = options.documentsAreBOSPrefixed ?? false;
  }

  async load(options: LoadOptions = {}): Promise<void> {
    const format = options.format ?? 'tokens';
    if (format === 'documents') {
      await this.loadDocuments();
      return;
    }
    if (format === 'tokens') {
      await this.loadTokens();
      return;
    }

    const res = await fetch(this.url);
    if (!res.ok) throw new Error(`Failed to fetch ${this.url}: ${res.status}`);
    const contentType = res.headers.get('content-type')?.toLowerCase() ?? '';
    if (contentType.includes('json') || this.url.endsWith('.json')) {
      const payload = await res.json();
      if (Array.isArray(payload)) {
        if (payload.length === 0) {
          this.setTokenStream(new Int32Array(0));
          return;
        }
        if (typeof payload[0] === 'number') {
          this.setTokenStream(inferJsonTokens(payload));
          return;
        }
        this.setDocuments(inferJsonDocuments(payload));
        return;
      }

      if (payload && typeof payload === 'object') {
        const record = payload as Record<string, unknown>;
        if (Array.isArray(record.documents)) {
          this.setDocuments(inferJsonDocuments(record.documents));
          return;
        }
        if (Array.isArray(record.tokens)) {
          this.setTokenStream(inferJsonTokens(record.tokens));
          return;
        }
      }

      throw new Error('Unsupported JSON data shape. Expected tokens or documents.');
      return;
    }
    this.setTokenStream(new Int32Array(await res.arrayBuffer()));
  }

  async loadTokens(url: string = this.url): Promise<void> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
    this.setTokenStream(new Int32Array(await res.arrayBuffer()));
  }

  async loadDocuments(url: string = this.url): Promise<void> {
    const payload = await readJsonPayload(url);
    this.setDocuments(inferJsonDocuments(payload));
  }

  setTokenStream(tokens: ArrayLike<number>): void {
    this.tokens = cloneTokens(tokens);
    this.documents = null;
    this.docBatcher = null;
    this.pos = 0;
    this.epochs = 0;
  }

  setDocuments(documents: DocumentInput): void {
    const docs = normalizeDocuments(documents, {
      bosTokenId: this.bosTokenId,
      prependBos: !this.documentsAreBOSPrefixed,
    });
    this.documents = docs;
    this.docBatcher = new BestFitDocumentBatcher(docs, {
      bosTokenId: this.bosTokenId,
      bufferSize: this.documentBufferSize,
      documentsAreBOSPrefixed: true,
    });
    this.tokens = new Int32Array(0);
    this.pos = 0;
    this.epochs = 0;
  }

  get numTokens(): number {
    if (this.documents) {
      let total = 0;
      for (const doc of this.documents) {
        total += doc.length;
      }
      return total;
    }
    return this.tokens.length;
  }

  get currentEpoch(): number {
    return this.docBatcher?.currentEpoch ?? this.epochs;
  }

  nextBatch(batchSize: number, seqLen: number): TokenBatch {
    if (this.docBatcher) {
      return this.docBatcher.nextBatch(batchSize, seqLen);
    }

    const needed = batchSize * seqLen;
    const inputIds = new Int32Array(needed);
    const targets = new Int32Array(needed);

    if (this.tokens.length === 0) {
      throw new Error('TokenDataLoader has no token stream loaded.');
    }

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

  /** Split flat tokens into train and val streams. Document mode uses splitDocuments(). */
  valSplit(ratio = 0.1): Int32Array {
    if (this.documents) {
      throw new Error('valSplit() only supports flat token streams. Use splitDocuments() for document data.');
    }

    if (this.tokens.length === 0) {
      return new Int32Array(0);
    }

    const splitIdx = Math.floor(this.tokens.length * (1 - ratio));
    const valTokens = this.tokens.slice(splitIdx);
    // Trim training cursor to train-only range
    this.tokens = this.tokens.slice(0, splitIdx);
    this.pos = 0;
    return valTokens;
  }

  splitDocuments(ratio = 0.1): DocumentSplit {
    if (!this.documents) {
      throw new Error('splitDocuments() requires document mode. Call setDocuments() or loadDocuments().');
    }

    const splitIdx = Math.floor(this.documents.length * (1 - ratio));
    const trainDocuments = this.documents.slice(0, splitIdx);
    const valDocuments = this.documents.slice(splitIdx);
    return { trainDocuments, valDocuments };
  }

  /** Generate a batch from a fixed token array (for validation). */
  static batchFromArray(tokens: Int32Array, batchSize: number, seqLen: number, offset: number): {
    inputIds: Int32Array;
    targets: Int32Array;
  } {
    if (tokens.length === 0) {
      return {
        inputIds: new Int32Array(batchSize * seqLen),
        targets: new Int32Array(batchSize * seqLen),
      };
    }

    const needed = batchSize * seqLen;
    const inputIds = new Int32Array(needed);
    const targets = new Int32Array(needed);
    const base = ((offset % tokens.length) + tokens.length) % tokens.length;

    for (let i = 0; i < needed; i++) {
      const idx = (base + i) % tokens.length;
      inputIds[i] = tokens[idx];
      const nextIdx = (idx + 1) % tokens.length;
      targets[i] = tokens[nextIdx];
    }

    return { inputIds, targets };
  }

  /**
   * Get a fixed batch for validation (non-streaming, no epoch wrap).
   * Used by evaluateBPB to get deterministic eval data.
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

export function createBestFitPackedBatcher(
  documents: DocumentInput,
  options: TokenDataLoaderOptions,
): BestFitDocumentBatcher {
  return new BestFitDocumentBatcher(documents, {
    bosTokenId: options.bosTokenId ?? 0,
    bufferSize: Math.max(1, options.documentBufferSize ?? 1000),
    documentsAreBOSPrefixed: options.documentsAreBOSPrefixed ?? false,
  });
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
