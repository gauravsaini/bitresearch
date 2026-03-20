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
}
