export interface PackedBatch {
  inputIds: Int32Array;
  targets: Int32Array;
  epoch: number;
}

export interface BestFitDocumentBatcherOptions {
  bosTokenId: number;
  bufferSize?: number;
  documentsAreBOSPrefixed?: boolean;
}

export type DocumentInput = ArrayLike<ArrayLike<number>> | Iterable<ArrayLike<number>>;

export function normalizeDocuments(
  documents: DocumentInput,
  options: { bosTokenId?: number; prependBos?: boolean } = {},
): Int32Array[] {
  const { bosTokenId = 0, prependBos = false } = options;
  return Array.from(documents, (doc) => {
    const ids = Int32Array.from(doc);
    if (!prependBos) {
      return ids;
    }
    const packed = new Int32Array(ids.length + 1);
    packed[0] = bosTokenId;
    packed.set(ids, 1);
    return packed;
  });
}

export function prependToken(doc: ArrayLike<number>, tokenId: number): Int32Array {
  const ids = Int32Array.from(doc);
  const packed = new Int32Array(ids.length + 1);
  packed[0] = tokenId;
  packed.set(ids, 1);
  return packed;
}

export class BestFitDocumentBatcher {
  private readonly bosTokenId: number;
  private readonly bufferSize: number;
  private readonly documentsAreBOSPrefixed: boolean;
  private sourceDocuments: Int32Array[] = [];
  private sourceIndex = 0;
  private bufferedDocuments: Int32Array[] = [];
  private epoch = 1;

  constructor(
    documents: DocumentInput = [],
    options: BestFitDocumentBatcherOptions,
  ) {
    this.bosTokenId = options.bosTokenId;
    this.bufferSize = Math.max(1, options.bufferSize ?? 1000);
    this.documentsAreBOSPrefixed = options.documentsAreBOSPrefixed ?? false;
    this.setDocuments(documents);
  }

  get currentEpoch(): number {
    return this.epoch;
  }

  get documentCount(): number {
    return this.sourceDocuments.length;
  }

  get bufferedDocumentCount(): number {
    return this.bufferedDocuments.length;
  }

  reset(): void {
    this.sourceIndex = 0;
    this.bufferedDocuments = [];
    this.epoch = 1;
  }

  setDocuments(documents: DocumentInput): void {
    this.sourceDocuments = normalizeDocuments(documents, {
      bosTokenId: this.bosTokenId,
      prependBos: !this.documentsAreBOSPrefixed,
    });
    this.reset();
  }

  private takeNextDocument(): Int32Array {
    if (this.sourceDocuments.length === 0) {
      throw new Error('BestFitDocumentBatcher requires at least one document.');
    }
    if (this.sourceIndex >= this.sourceDocuments.length) {
      this.sourceIndex = 0;
      this.epoch += 1;
    }
    return this.sourceDocuments[this.sourceIndex++];
  }

  private refillBuffer(): void {
    while (this.bufferedDocuments.length < this.bufferSize) {
      this.bufferedDocuments.push(this.takeNextDocument());
    }
  }

  private findBestFitIndex(remaining: number): number {
    let bestIndex = -1;
    let bestLength = 0;
    for (let i = 0; i < this.bufferedDocuments.length; i++) {
      const docLength = this.bufferedDocuments[i].length;
      if (docLength <= remaining && docLength > bestLength) {
        bestIndex = i;
        bestLength = docLength;
      }
    }
    return bestIndex;
  }

  private findShortestIndex(): number {
    if (this.bufferedDocuments.length === 0) {
      throw new Error('BestFitDocumentBatcher buffer is empty.');
    }
    let shortestIndex = 0;
    let shortestLength = this.bufferedDocuments[0].length;
    for (let i = 1; i < this.bufferedDocuments.length; i++) {
      const docLength = this.bufferedDocuments[i].length;
      if (docLength < shortestLength) {
        shortestIndex = i;
        shortestLength = docLength;
      }
    }
    return shortestIndex;
  }

  packRows(batchSize: number, seqLen: number): Int32Array {
    const rowCapacity = seqLen + 1;
    const rows = new Int32Array(batchSize * rowCapacity);

    for (let rowIdx = 0; rowIdx < batchSize; rowIdx++) {
      let pos = 0;
      const rowStart = rowIdx * rowCapacity;

      while (pos < rowCapacity) {
        this.refillBuffer();

        const remaining = rowCapacity - pos;
        const bestIndex = this.findBestFitIndex(remaining);

        if (bestIndex >= 0) {
          const doc = this.bufferedDocuments.splice(bestIndex, 1)[0];
          rows.set(doc, rowStart + pos);
          pos += doc.length;
          continue;
        }

        const shortestIndex = this.findShortestIndex();
        const doc = this.bufferedDocuments.splice(shortestIndex, 1)[0];
        rows.set(doc.subarray(0, remaining), rowStart + pos);
        pos += remaining;
      }
    }

    return rows;
  }

  nextBatch(batchSize: number, seqLen: number): PackedBatch {
    const rowCapacity = seqLen + 1;
    const rows = this.packRows(batchSize, seqLen);
    const inputIds = new Int32Array(batchSize * seqLen);
    const targets = new Int32Array(batchSize * seqLen);

    for (let rowIdx = 0; rowIdx < batchSize; rowIdx++) {
      const rowStart = rowIdx * rowCapacity;
      const flatStart = rowIdx * seqLen;
      inputIds.set(rows.subarray(rowStart, rowStart + seqLen), flatStart);
      targets.set(rows.subarray(rowStart + 1, rowStart + rowCapacity), flatStart);
    }

    return { inputIds, targets, epoch: this.currentEpoch };
  }
}
