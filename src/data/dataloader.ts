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
    this.tokens = this.tokens.subarray(0, splitIdx);
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
