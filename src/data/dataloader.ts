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

  /** Split tokens into train (first 90%) and val (last 10%). Returns val tokens. */
  valSplit(ratio = 0.1): Int32Array {
    const splitIdx = Math.floor(this.tokens.length * (1 - ratio));
    const valTokens = this.tokens.slice(splitIdx);
    // Trim training cursor to train-only range
    this.tokens = this.tokens.subarray(0, splitIdx);
    this.pos = 0;
    return valTokens;
  }

  /** Generate a batch from a fixed token array (for validation). */
  static batchFromArray(tokens: Int32Array, batchSize: number, seqLen: number, offset: number): { inputIds: Int32Array; targets: Int32Array } {
    const needed = batchSize * seqLen;
    const inputIds = new Int32Array(needed);
    const targets = new Int32Array(needed);
    for (let i = 0; i < needed; i++) {
      const idx = (offset + i) % tokens.length;
      inputIds[i] = tokens[idx];
      const nextIdx = (offset + i + 1) % tokens.length;
      targets[i] = tokens[nextIdx];
    }
    return { inputIds, targets };
  }
}
