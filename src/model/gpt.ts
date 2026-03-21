import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { GPTConfig, computeWindowSizes } from './config';

export interface LayerWeights {
  qWeight: tf.Variable;
  kWeight: tf.Variable;
  vWeight: tf.Variable;
  projWeight: tf.Variable;
  fcWeight: tf.Variable;
  mlpProjWeight: tf.Variable;
  // Value Embedding gate (only on alternating layers)
  veGate: tf.Variable | null;
}

export interface ValueEmbedding {
  weight: tf.Variable;
}

function hasVe(layerIdx: number, nLayer: number): boolean {
  return layerIdx % 2 === (nLayer - 1) % 2;
}

export class GPTModel {
  config: GPTConfig;
  wte!: tf.Variable;
  lmHead!: tf.Variable;
  residLambdas!: tf.Variable;
  x0Lambdas!: tf.Variable;
  valueEmbeds: Map<number, ValueEmbedding> = new Map();
  layers: LayerWeights[];
  windowSizes: [number, number][];

  // Precomputed RoPE — [1, T, 1, headDim] shaped for broadcast
  cosTable!: tf.Tensor;
  sinTable!: tf.Tensor;

  constructor(config: GPTConfig) {
    this.config = config;
    this.windowSizes = computeWindowSizes(config);
    this.layers = [];
  }

  async init() {
    try {
      await tf.setBackend('webgpu');
    } catch (e) {
      console.warn('WebGPU not supported or failed to initialize. Falling back to WebGL.');
      await tf.setBackend('webgl');
    }
    await tf.ready();

    const { vocabSize, nLayer, nHead, nKvHead, nEmbd, sequenceLen } = this.config;
    const headDim = nEmbd / nHead;
    const kvDim = nKvHead * headDim;
    const veGateChannels = 32;

    // Karpathy's exact weight init:
    // wte: normal std=1.0
    // lm_head: normal std=0.001
    // Q, K, V, fc: uniform [-s, s] where s = 3^0.5 * n_embd^-0.5
    // proj, mlp_proj: zeros
    // resid_lambdas: ones, x0_lambdas: 0.1
    // value_embeds: uniform [-s, s]
    // ve_gate: zeros
    const s = Math.sqrt(3.0) * Math.pow(nEmbd, -0.5);

    tf.tidy(() => {
      // Token embedding: normal std=1.0 (Karpathy style)
      this.wte = tf.variable(tf.randomStandardNormal([vocabSize, nEmbd]), true, 'wte');

      // LM head: normal std=0.001
      this.lmHead = tf.variable(tf.randomStandardNormal([vocabSize, nEmbd]).mul(0.001), true, 'lmHead');

      // Residual lambdas: all ones
      this.residLambdas = tf.variable(tf.ones([nLayer]), true, 'residLambdas');
      // X0 lambdas: all 0.1
      this.x0Lambdas = tf.variable(tf.fill([nLayer], 0.1), true, 'x0Lambdas');

      for (let i = 0; i < nLayer; i++) {
        // Q, K, V, fc: uniform [-s, s]
        const qInit = tf.randomUniform([nEmbd, nHead * headDim], -s, s);
        const kInit = tf.randomUniform([nEmbd, kvDim], -s, s);
        const vInit = tf.randomUniform([nEmbd, kvDim], -s, s);
        const fcInit = tf.randomUniform([nEmbd, 4 * nEmbd], -s, s);

        // proj, mlp_proj: zeros
        const projInit = tf.zeros([nEmbd, nEmbd]);
        const mlpProjInit = tf.zeros([4 * nEmbd, nEmbd]);

        // Value embedding gate (alternating layers)
        const useVe = hasVe(i, nLayer);
        const veGateInit = useVe ? tf.zeros([veGateChannels, nKvHead]) : null;

        this.layers.push({
          qWeight: tf.variable(qInit, true, `layer${i}_q`),
          kWeight: tf.variable(kInit, true, `layer${i}_k`),
          vWeight: tf.variable(vInit, true, `layer${i}_v`),
          projWeight: tf.variable(projInit, true, `layer${i}_proj`),
          fcWeight: tf.variable(fcInit, true, `layer${i}_fc`),
          mlpProjWeight: tf.variable(mlpProjInit, true, `layer${i}_mlp_proj`),
          veGate: useVe ? tf.variable(veGateInit!, true, `layer${i}_ve_gate`) : null,
        });

        // Value embeddings for alternating layers
        if (useVe) {
          const veInit = tf.randomUniform([vocabSize, kvDim], -s, s);
          this.valueEmbeds.set(i, {
            weight: tf.variable(veInit, true, `ve_${i}`),
          });
        }
      }

      // Precompute RoPE: [1, T, 1, headDim]
      const invFreq = tf.div(1.0, tf.pow(10000.0, tf.div(tf.range(0, headDim, 2), headDim))) as tf.Tensor1D;
      const t = tf.range(0, sequenceLen);
      const freqs = tf.outerProduct(t, invFreq);
      const emb = tf.concat([freqs, freqs], -1);

      this.cosTable = tf.keep(tf.cos(emb).reshape([1, sequenceLen, 1, headDim]));
      this.sinTable = tf.keep(tf.sin(emb).reshape([1, sequenceLen, 1, headDim]));
    });
  }

  /**
   * RMS Normalization — Karpathy style: F.rms_norm(x, (x.size(-1),))
   */
  private rmsNorm(x: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const variance = tf.mean(tf.square(x), -1, true);
      const invRms = tf.rsqrt(variance.add(1e-5));
      return x.mul(invRms);
    });
  }

  /**
   * Applies Rotary Positional Embeddings — Karpathy's exact formula:
   * y1 = x1 * cos + x2 * sin
   * y2 = x1 * (-sin) + x2 * cos
   */
  private applyRotary(x: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const [B, T, numHeads, headDim] = x.shape;
      const d = headDim / 2;

      const x1 = x.slice([0, 0, 0, 0], [B, T, numHeads, d]);
      const x2 = x.slice([0, 0, 0, d], [B, T, numHeads, d]);

      const cos = this.cosTable.slice([0, 0, 0, 0], [1, T, 1, d]);
      const sin = this.sinTable.slice([0, 0, 0, 0], [1, T, 1, d]);

      // y1 = x1 * cos + x2 * sin
      const y1 = x1.mul(cos).add(x2.mul(sin));
      // y2 = x1 * (-sin) + x2 * cos
      const y2 = x1.mul(tf.neg(sin)).add(x2.mul(cos));

      return tf.concat([y1, y2], -1);
    });
  }

  /**
   * Forward pass — exact Karpathy autoresearch port.
   * @param reduction 'mean' returns scalar loss, 'none' returns per-token loss [B,T]
   */
  forward(inputIds: tf.Tensor, targets: tf.Tensor, returnLogits?: boolean): { loss: tf.Scalar, logits?: tf.Tensor };
  forward(inputIds: tf.Tensor, targets: tf.Tensor, reduction: 'none'): tf.Tensor2D;
  forward(inputIds: tf.Tensor, targets: tf.Tensor, returnLogitsOrReduction: boolean | 'none' = false): any {
    return tf.tidy(() => {
      const [B, T] = inputIds.shape;
      const { nEmbd, nLayer, nHead, nKvHead, vocabSize } = this.config;
      const headDim = nEmbd / nHead;

      // 1. Token embedding + RMSNorm
      let x = tf.gather(this.wte, inputIds);
      x = this.rmsNorm(x);
      const x0 = x; // save for residual ladder

      let current = x;

      // Precompute causal mask once: [T, T], lower triangular = 0, upper = -1e9
      const mask = tf.linalg.bandPart(tf.ones([T, T]), -1, 0).sub(1).mul(1e9);

      // 2. Transformer blocks
      for (let i = 0; i < nLayer; i++) {
        const layer = this.layers[i];
        const windowSize = this.windowSizes[i];

        // Residual ladder: x = resid_lambdas[i] * x + x0_lambdas[i] * x0
        const lambdaResid = this.residLambdas.slice([i], [1]).reshape([1, 1, 1]);
        const lambdaX0 = this.x0Lambdas.slice([i], [1]).reshape([1, 1, 1]);
        current = current.mul(lambdaResid).add(x0.mul(lambdaX0));

        // Pre-norm
        let preNorm = this.rmsNorm(current);

        // Flatten for matmul: [B*T, nEmbd]
        const flat = preNorm.reshape([B * T, nEmbd]);

        // Q, K, V projections
        let q = flat.matMul(layer.qWeight).reshape([B, T, nHead, headDim]);
        let k = flat.matMul(layer.kWeight).reshape([B, T, nKvHead, headDim]);
        let v = flat.matMul(layer.vWeight).reshape([B, T, nKvHead, headDim]);

        // Value Embedding (ResFormer): mix in value embedding with input-dependent gate
        if (this.valueEmbeds.has(i) && layer.veGate) {
          const ve = this.valueEmbeds.get(i)!;
          let veVal = tf.gather(ve.weight, inputIds).reshape([B, T, nKvHead, headDim]);

          // Gate: 2 * sigmoid(ve_gate(x[..., :32]))
          const xGated = current.slice([0, 0, 0], [B, T, 32]);
          const flatGated = xGated.reshape([B * T, 32]);
          const gateRaw = flatGated.matMul(layer.veGate).reshape([B, T, nKvHead, 1]);
          const gate = tf.sigmoid(gateRaw).mul(2);

          v = v.add(veVal.mul(gate));
        }

        // Apply RoPE
        q = this.applyRotary(q);
        k = this.applyRotary(k);

        // RMSNorm on Q and K (per Karpathy)
        q = this.rmsNorm(q);
        k = this.rmsNorm(k);

        // Transpose: [B, T, nHead, headDim] -> [B, nHead, T, headDim]
        q = q.transpose([0, 2, 1, 3]);
        k = k.transpose([0, 2, 1, 3]);
        v = v.transpose([0, 2, 1, 3]);

        // Causal attention with sliding window
        // scores = Q @ K^T / sqrt(headDim)
        const scale = Math.sqrt(headDim);
        const scores = tf.matMul(q, k, false, true).div(scale);
        const maskedScores = scores.add(mask.reshape([1, 1, T, T]));

        // Apply sliding window mask if needed
        let windowedScores = maskedScores;
        if (windowSize[0] > 0 && windowSize[0] < T) {
          const ws = windowSize[0];
          // Build window mask: positions where (j < i - ws + 1) are masked
          const rowIdx = tf.range(0, T).reshape([T, 1]);
          const colIdx = tf.range(0, T).reshape([1, T]);
          const windowMask = tf.cast(colIdx.less(rowIdx.sub(ws - 1).maximum(0)), 'float32').mul(1e9);
          windowedScores = maskedScores.sub(windowMask.reshape([1, 1, T, T]));
        }

        const probs = tf.softmax(windowedScores, -1);
        let attnOut = tf.matMul(probs, v);

        // Transpose back: [B, nHead, T, headDim] -> [B, T, nHead, headDim] -> [B*T, nEmbd]
        attnOut = attnOut.transpose([0, 2, 1, 3]).reshape([B * T, nEmbd]);

        // Output projection (initialized to zero, so starts as identity-like)
        let projected = attnOut.matMul(layer.projWeight).reshape([B, T, nEmbd]);

        // First residual
        current = current.add(projected);

        // MLP block: fc -> ReLU^2 -> proj
        let mlpNorm = this.rmsNorm(current).reshape([B * T, nEmbd]);
        let hidden = mlpNorm.matMul(layer.fcWeight);
        hidden = tf.relu(hidden).square(); // Squared ReLU activation
        let mlpOut = hidden.matMul(layer.mlpProjWeight).reshape([B, T, nEmbd]);

        // Second residual
        current = current.add(mlpOut);
      }

      // 3. Final norm
      current = this.rmsNorm(current).reshape([B * T, nEmbd]);

      // 4. LM Head (logits) — lmHead is [V, D], transpose for matmul
      let logits = current.matMul(this.lmHead, false, true);

      // 5. Softcap logits (Karpathy: 15 * tanh(logits / 15))
      const softcap = 15.0;
      logits = tf.tanh(logits.div(softcap)).mul(softcap);

      // 6. Cross-Entropy Loss
      const flatTargets = targets.reshape([-1]);
      const validMask = flatTargets.notEqual(-1);
      const oneHotTargets = tf.oneHot(tf.cast(tf.relu(flatTargets), 'int32'), vocabSize);
      const softmaxProbs = tf.softmax(logits, -1);
      const logProbs = tf.log(softmaxProbs.add(1e-10));
      const unreducedLoss = tf.neg(tf.sum(oneHotTargets.mul(logProbs), -1));
      const maskedLoss = unreducedLoss.mul(tf.cast(validMask, 'float32'));
      const cce = tf.mean(maskedLoss);

      if (returnLogitsOrReduction === 'none') {
        // Return per-token loss [B*T] reshaped to [B, T]
        return unreducedLoss.reshape([B, T]) as tf.Tensor2D;
      }

      if (returnLogitsOrReduction === true) {
        return { loss: cce as tf.Scalar, logits };
      }
      return { loss: cce as tf.Scalar };
    });
  }

  /**
   * Get all trainable variables (flat list)
   */
  getTrainableVariables(): tf.Variable[] {
    const vars: tf.Variable[] = [
      this.wte, this.lmHead, this.residLambdas, this.x0Lambdas
    ];
    for (const layer of this.layers) {
      vars.push(layer.qWeight, layer.kWeight, layer.vWeight, layer.projWeight, layer.fcWeight, layer.mlpProjWeight);
      if (layer.veGate) vars.push(layer.veGate);
    }
    for (const ve of this.valueEmbeds.values()) {
      vars.push(ve.weight);
    }
    return vars;
  }

  /**
   * Get variables grouped by type for optimizer parameter groups.
   * Returns:
   * - lmHead: unembedding (AdamW, low LR)
   * - wte: token embedding (AdamW, high LR)
   * - valueEmbeds: value embeddings (AdamW, high LR)
   * - scalars: residLambdas, x0Lambdas (AdamW, very high LR)
   * - matrices: all Q/K/V/proj/fc/mlp weights (Muon)
   */
  getTrainableVariablesByName(): Record<string, tf.Variable> {
    const map: Record<string, tf.Variable> = {
      wte: this.wte,
      lmHead: this.lmHead,
      residLambdas: this.residLambdas,
      x0Lambdas: this.x0Lambdas,
    };
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      map[`layer${i}_q`] = layer.qWeight;
      map[`layer${i}_k`] = layer.kWeight;
      map[`layer${i}_v`] = layer.vWeight;
      map[`layer${i}_proj`] = layer.projWeight;
      map[`layer${i}_fc`] = layer.fcWeight;
      map[`layer${i}_mlp_proj`] = layer.mlpProjWeight;
      if (layer.veGate) map[`layer${i}_ve_gate`] = layer.veGate;
    }
    for (const [idx, ve] of this.valueEmbeds) {
      map[`ve_${idx}`] = ve.weight;
    }
    return map;
  }

  getParamGroups(): {
    lmHead: tf.Variable[];
    wte: tf.Variable[];
    valueEmbeds: tf.Variable[];
    scalars: tf.Variable[];
    matrices: tf.Variable[];
  } {
    const matrices: tf.Variable[] = [];
    for (const layer of this.layers) {
      matrices.push(layer.qWeight, layer.kWeight, layer.vWeight, layer.projWeight, layer.fcWeight, layer.mlpProjWeight);
      if (layer.veGate) matrices.push(layer.veGate);
    }
    const valueEmbedsVars: tf.Variable[] = [];
    for (const ve of this.valueEmbeds.values()) {
      valueEmbedsVars.push(ve.weight);
    }
    return {
      lmHead: [this.lmHead],
      wte: [this.wte],
      valueEmbeds: valueEmbedsVars,
      scalars: [this.residLambdas, this.x0Lambdas],
      matrices,
    };
  }

  /**
   * Karpathy's exact FLOPs estimation (for MFU tracking)
   */
  estimateFlopsPerToken(): number {
    const { nLayer, nEmbd, nHead, nKvHead, sequenceLen } = this.config;
    const headDim = nEmbd / nHead;
    const kvDim = nKvHead * headDim;

    // Count matrix params (exclude embeddings, lm_head, scalars)
    let matrixParams = 0;
    for (const layer of this.layers) {
      matrixParams += nEmbd * nHead * headDim;  // Q
      matrixParams += nEmbd * kvDim;             // K
      matrixParams += nEmbd * kvDim;             // V
      matrixParams += nEmbd * nEmbd;             // proj
      matrixParams += nEmbd * 4 * nEmbd;         // fc
      matrixParams += 4 * nEmbd * nEmbd;         // mlp_proj
    }

    // Attention FLOPs
    let attnFlops = 0;
    for (const windowSize of this.windowSizes) {
      const effectiveSeq = windowSize[0] > 0 ? Math.min(windowSize[0], sequenceLen) : sequenceLen;
      attnFlops += 12 * nHead * headDim * effectiveSeq;
    }

    return 6 * matrixParams + attnFlops;
  }
}
