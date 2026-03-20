// GPT Model Configuration — mirrors the PyTorch GPTConfig

export interface GPTConfig {
  sequenceLen: number;   // context window (default 512 for WebGPU)
  vocabSize: number;     // vocabulary size
  nLayer: number;        // number of transformer layers
  nHead: number;         // number of attention heads
  nKvHead: number;       // number of KV heads (for GQA/MQA)
  nEmbd: number;         // embedding dimension
  windowPattern: string; // sliding window pattern 'SSSL'
}

// Smaller defaults for WebGPU (consumer GPUs have less memory/compute)
export const DEFAULT_CONFIG: GPTConfig = {
  sequenceLen: 512,
  vocabSize: 8192,
  nLayer: 4,
  nHead: 4,
  nKvHead: 4,
  nEmbd: 256,
  windowPattern: 'L',
};

export function hasValueEmbedding(layerIdx: number, nLayer: number): boolean {
  return layerIdx % 2 === (nLayer - 1) % 2;
}

// Compute window sizes from pattern
export function computeWindowSizes(config: GPTConfig): [number, number][] {
  const pattern = config.windowPattern.toUpperCase();
  if (pattern.length === 0) {
    throw new Error('windowPattern must not be empty');
  }
  const longWindow = config.sequenceLen;
  const shortWindow = Math.floor(longWindow / 2);
  const charToWindow: Record<string, [number, number]> = {
    L: [longWindow, 0],
    S: [shortWindow, 0],
  };

  const windowSizes: [number, number][] = [];
  for (let i = 0; i < config.nLayer; i++) {
    const char = pattern[i % pattern.length];
    windowSizes.push(charToWindow[char]);
  }
  // Last layer always full attention
  if (windowSizes.length > 0) {
    windowSizes[windowSizes.length - 1] = [longWindow, 0];
  }
  return windowSizes;
}

// Estimate parameter count
export function estimateParams(config: GPTConfig): number {
  const { nLayer, nEmbd, nHead, nKvHead, vocabSize } = config;
  const headDim = nEmbd / nHead;
  const kvDim = nKvHead * headDim;
  const veGateChannels = 32;

  // Embedding + LM head
  let params = vocabSize * nEmbd * 2;

  // Per layer
  for (let i = 0; i < nLayer; i++) {
    // Attention: Q, K, V projections + output projection
    params += nEmbd * (nHead * headDim);  // Q
    params += nEmbd * kvDim;               // K
    params += nEmbd * kvDim;               // V
    params += nEmbd * nEmbd;               // output projection

    // MLP: fc + proj
    params += nEmbd * (4 * nEmbd);  // fc
    params += (4 * nEmbd) * nEmbd;  // proj

    // Value embeddings (alternating layers)
    if (hasValueEmbedding(i, nLayer)) {
      params += vocabSize * kvDim;
      params += veGateChannels * nKvHead;
    }
  }

  // Per-layer scalars
  params += nLayer * 2;

  return params;
}
