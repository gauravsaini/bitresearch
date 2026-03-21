import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { GPTConfig, computeWindowSizes, hasValueEmbedding } from './config';

export interface LayerWeights {
  qWeight: tf.Variable;
  kWeight: tf.Variable;
  vWeight: tf.Variable;
  projWeight: tf.Variable;
  fcWeight: tf.Variable;
  mlpProjWeight: tf.Variable;
  valueEmbedWeight?: tf.Variable;
  veGateWeight?: tf.Variable;
}

export class GPTModel {
  private static readonly VE_GATE_CHANNELS = 32;

  config: GPTConfig;
  wte!: tf.Variable;
  lmHead!: tf.Variable;
  residLambdas!: tf.Variable;
  x0Lambdas!: tf.Variable;
  layers: LayerWeights[];
  windowSizes: [number, number][];

  // Precomputed RoPE
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
      try {
        await tf.setBackend('webgl');
      } catch {
        console.warn('WebGL backend not available. Falling back to CPU.');
        await tf.setBackend('cpu');
      }
    }
    await tf.ready();

    const { vocabSize, nLayer, nHead, nKvHead, nEmbd, sequenceLen } = this.config;
    if (nEmbd % nHead !== 0) {
      throw new Error(`nEmbd (${nEmbd}) must be divisible by nHead (${nHead})`);
    }
    if (nKvHead <= 0 || nHead % nKvHead !== 0) {
      throw new Error(`nKvHead (${nKvHead}) must be a positive divisor of nHead (${nHead})`);
    }
    const headDim = nEmbd / nHead;
    if (headDim % 2 !== 0) {
      throw new Error(`head dimension (${headDim}) must be even for RoPE`);
    }
    if (nEmbd < GPTModel.VE_GATE_CHANNELS) {
      throw new Error(`nEmbd (${nEmbd}) must be at least ${GPTModel.VE_GATE_CHANNELS} for ve_gate`);
    }
    const kvDim = nKvHead * headDim;

    tf.tidy(() => {
      this.wte = tf.variable(tf.randomStandardNormal([vocabSize, nEmbd]).mul(0.02), true, 'wte');
      this.lmHead = tf.variable(tf.randomStandardNormal([vocabSize, nEmbd]).mul(0.02), true, 'lmHead');

      this.residLambdas = tf.variable(tf.ones([nLayer]), true, 'residLambdas');
      this.x0Lambdas = tf.variable(tf.fill([nLayer], 0.1), true, 'x0Lambdas');

      for (let i = 0; i < nLayer; i++) {
        const layer: LayerWeights = {
          qWeight: tf.variable(tf.randomStandardNormal([nEmbd, nHead * headDim]).mul(0.02), true, `layer${i}_q`),
          kWeight: tf.variable(tf.randomStandardNormal([nEmbd, kvDim]).mul(0.02), true, `layer${i}_k`),
          vWeight: tf.variable(tf.randomStandardNormal([nEmbd, kvDim]).mul(0.02), true, `layer${i}_v`),
          projWeight: tf.variable(tf.zeros([nEmbd, nEmbd]), true, `layer${i}_proj`),
          fcWeight: tf.variable(tf.randomStandardNormal([nEmbd, 4 * nEmbd]).mul(0.02), true, `layer${i}_fc`),
          mlpProjWeight: tf.variable(tf.zeros([4 * nEmbd, nEmbd]), true, `layer${i}_mlp_proj`),
        };
        if (hasValueEmbedding(i, nLayer)) {
          layer.valueEmbedWeight = tf.variable(
            tf.randomStandardNormal([vocabSize, kvDim]).mul(0.02),
            true,
            `layer${i}_value_embed`,
          );
          layer.veGateWeight = tf.variable(
            tf.zeros([GPTModel.VE_GATE_CHANNELS, nKvHead]),
            true,
            `layer${i}_ve_gate`,
          );
        }
        this.layers.push(layer);
      }

      // Precompute RoPE (simplified rotary embedding)
      const invFreq = tf.div(1.0, tf.pow(10000.0, tf.div(tf.range(0, headDim, 2), headDim))) as tf.Tensor1D;
      const t = tf.range(0, sequenceLen);
      const freqs = tf.outerProduct(t, invFreq);

      // We keep these as tensors, not variables, since they don't train
      this.cosTable = tf.keep(tf.cos(freqs));
      this.sinTable = tf.keep(tf.sin(freqs));
    });
  }

  /**
   * Helper for RMS Normalization
   */
  private rmsNorm(x: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const epsilon = 1e-5;
      const variance = tf.mean(tf.square(x), -1, true);
      const invRms = tf.rsqrt(variance.add(epsilon));
      return x.mul(invRms);
    });
  }

  /**
   * Applies Rotary Positional Embeddings
   * x: [B, T, num_heads, head_dim]
   */
  private applyRotary(x: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const [B, T, numHeads, headDim] = x.shape;
      const halfDim = headDim / 2;
      
      const x1 = x.slice([0, 0, 0, 0], [B, T, numHeads, halfDim]);
      const x2 = x.slice([0, 0, 0, halfDim], [B, T, numHeads, halfDim]);
      const cos = this.cosTable.slice([0, 0], [T, -1]).reshape([1, T, 1, halfDim]);
      const sin = this.sinTable.slice([0, 0], [T, -1]).reshape([1, T, 1, halfDim]);
      const y1 = x1.mul(cos).add(x2.mul(sin));
      const y2 = x1.mul(sin.neg()).add(x2.mul(cos));
      return tf.concat([y1, y2], -1);
    });
  }

  private buildAttentionMask(seqLen: number, windowSize: [number, number]): tf.Tensor {
    return tf.tidy(() => {
      const [left, right] = windowSize;
      const allowed = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), left, right);
      return allowed.sub(1).mul(1e9);
    });
  }

  /**
   * Forward pass returning loss and optional logits
   */
  forward(inputIds: tf.Tensor, targets: tf.Tensor, returnLogits = false): { loss: tf.Scalar, logits?: tf.Tensor } {
    return tf.tidy(() => {
      const [B, T] = inputIds.shape;
      const { nEmbd, nLayer, nHead, nKvHead } = this.config;
      const headDim = nEmbd / nHead;
      const kvGroupSize = nHead / nKvHead;

      // 1. Token embeddings: [B, T] -> [B, T, D]
      let x = tf.gather(this.wte, inputIds);
      let xNorm = this.rmsNorm(x);
      
      const x0 = xNorm;
      let current = xNorm;

      // 2. Transformer blocks
      for (let i = 0; i < nLayer; i++) {
        const layer = this.layers[i];
        
        // Residual ladder
        const lambdaResid = this.residLambdas.slice([i], [1]).reshape([1, 1, 1]);
        const lambdaX0 = this.x0Lambdas.slice([i], [1]).reshape([1, 1, 1]);
        
        let scaled = current.mul(lambdaResid).add(x0.mul(lambdaX0));
        let preNorm = this.rmsNorm(scaled);
        
        // Flatten B and T for matmul: [B*T, D]
        const flatNorm = preNorm.reshape([B * T, nEmbd]);
        
        // QKV Projections
        let q = flatNorm.matMul(layer.qWeight).reshape([B, T, nHead, headDim]);
        let k = flatNorm.matMul(layer.kWeight).reshape([B, T, nKvHead, headDim]);
        let v = flatNorm.matMul(layer.vWeight).reshape([B, T, nKvHead, headDim]);

        if (layer.valueEmbedWeight && layer.veGateWeight) {
          const ve = tf.gather(layer.valueEmbedWeight, inputIds).reshape([B, T, nKvHead, headDim]);
          const gateInput = preNorm.slice([0, 0, 0], [B, T, GPTModel.VE_GATE_CHANNELS]);
          const gate = tf.sigmoid(
            gateInput.reshape([B * T, GPTModel.VE_GATE_CHANNELS]).matMul(layer.veGateWeight)
          ).mul(2).reshape([B, T, nKvHead, 1]);
          v = v.add(ve.mul(gate));
        }
        
        // Apply RoPE
        q = this.applyRotary(q);
        k = this.applyRotary(k);
        
        // RMSNorm on Q and K (head-wise)
        q = this.rmsNorm(q);
        k = this.rmsNorm(k);

        if (kvGroupSize > 1) {
          k = k.reshape([B, T, nKvHead, 1, headDim]).tile([1, 1, 1, kvGroupSize, 1]).reshape([B, T, nHead, headDim]);
          v = v.reshape([B, T, nKvHead, 1, headDim]).tile([1, 1, 1, kvGroupSize, 1]).reshape([B, T, nHead, headDim]);
        }
        
        // Transpose for attention -> [B, nHead, T, headDim]
        q = q.transpose([0, 2, 1, 3]);
        k = k.transpose([0, 2, 1, 3]);
        v = v.transpose([0, 2, 1, 3]);

        // Causal/local Attention: softmax(Q*K^T / sqrt(d) + mask) * V
        const scores = tf.matMul(q, k, false, true).div(Math.sqrt(headDim));
        const maskedScores = scores.add(this.buildAttentionMask(T, this.windowSizes[i]).reshape([1, 1, T, T]));
        const probs = tf.softmax(maskedScores, -1);
        let attnOut = tf.matMul(probs, v);
        
        // Transpose back and flatten: [B, T, nHead, headDim] -> [B*T, D]
        attnOut = attnOut.transpose([0, 2, 1, 3]).reshape([B * T, nEmbd]);
        
        // Output projection
        let projected = attnOut.matMul(layer.projWeight).reshape([B, T, nEmbd]);
        
        // First residual sum
        scaled = scaled.add(projected);
        
        // MLP block
        let mlpNorm = this.rmsNorm(scaled).reshape([B * T, nEmbd]);
        let hidden = mlpNorm.matMul(layer.fcWeight);
        hidden = tf.relu(hidden).square(); // Squared ReLU
        let mlpOut = hidden.matMul(layer.mlpProjWeight).reshape([B, T, nEmbd]);
        
        // Second residual sum
        current = scaled.add(mlpOut);
      }

      // 3. Final norm
      current = this.rmsNorm(current).reshape([B * T, nEmbd]);
      
      // 4. LM Head (logits) -> [B*T, V]
      // lmHead is [V, D]. We transpose it to [D, V] for matmul
      let logits = current.matMul(this.lmHead, false, true);
      
      // 5. Softcap logits
      const softcap = 15.0;
      logits = tf.tanh(logits.div(softcap)).mul(softcap);
      
      // 6. Cross Entropy Loss
      // targets is [B, T], flatten to [B*T]
      const flatTargets = targets.reshape([-1]);
      
      const validMask = flatTargets.notEqual(-1);
      
      // Explicit mathematically-exact CCE to match PyTorch reduction semantics
      const oneHotTargets = tf.oneHot(tf.cast(tf.relu(flatTargets), 'int32'), this.config.vocabSize);
      
      const probs = tf.softmax(logits, -1);
      const logProbs = tf.log(probs.add(1e-10));
      const unreducedLoss = tf.neg(tf.sum(oneHotTargets.mul(logProbs), -1));
      
      const maskedLoss = unreducedLoss.mul(tf.cast(validMask, 'float32'));
      const cce = tf.mean(maskedLoss);
      
      if (returnLogits) {
        return { loss: cce as tf.Scalar, logits };
      }
      return { loss: cce as tf.Scalar };
    });
  }

  /**
   * Get all trainable variables
   */
  getTrainableVariables(): tf.Variable[] {
    const vars: tf.Variable[] = [
      this.wte, this.lmHead, this.residLambdas, this.x0Lambdas
    ];
    for (const layer of this.layers) {
      vars.push(layer.qWeight, layer.kWeight, layer.vWeight, layer.projWeight, layer.fcWeight, layer.mlpProjWeight);
      if (layer.valueEmbedWeight) {
        vars.push(layer.valueEmbedWeight);
      }
      if (layer.veGateWeight) {
        vars.push(layer.veGateWeight);
      }
    }
    return vars;
  }

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
      if (layer.valueEmbedWeight) {
        map[`layer${i}_value_embed`] = layer.valueEmbedWeight;
      }
      if (layer.veGateWeight) {
        map[`layer${i}_ve_gate`] = layer.veGateWeight;
      }
    }
    return map;
  }
}
