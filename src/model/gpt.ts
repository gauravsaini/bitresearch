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
}

export class GPTModel {
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
      await tf.setBackend('webgl');
    }
    await tf.ready();

    const { vocabSize, nLayer, nHead, nKvHead, nEmbd, sequenceLen } = this.config;
    const headDim = nEmbd / nHead;
    const kvDim = nKvHead * headDim;
    
    // PyTorch GPT-2 standard initialization parity
    const std = 0.02;
    const projStd = 0.02 / Math.sqrt(2 * nLayer);

    tf.tidy(() => {
      this.wte = tf.variable(tf.randomStandardNormal([vocabSize, nEmbd]).mul(std), true, 'wte');
      this.lmHead = tf.variable(tf.randomStandardNormal([vocabSize, nEmbd]).mul(std), true, 'lmHead');
      
      this.residLambdas = tf.variable(tf.ones([nLayer]), true, 'residLambdas');
      this.x0Lambdas = tf.variable(tf.fill([nLayer], 0.1), true, 'x0Lambdas');

      for (let i = 0; i < nLayer; i++) {
        this.layers.push({
          qWeight: tf.variable(tf.randomStandardNormal([nEmbd, nHead * headDim]).mul(std), true, `layer${i}_q`),
          kWeight: tf.variable(tf.randomStandardNormal([nEmbd, kvDim]).mul(std), true, `layer${i}_k`),
          vWeight: tf.variable(tf.randomStandardNormal([nEmbd, kvDim]).mul(std), true, `layer${i}_v`),
          projWeight: tf.variable(tf.randomStandardNormal([nEmbd, nEmbd]).mul(projStd), true, `layer${i}_proj`),
          fcWeight: tf.variable(tf.randomStandardNormal([nEmbd, 4 * nEmbd]).mul(std), true, `layer${i}_fc`),
          mlpProjWeight: tf.variable(tf.randomStandardNormal([4 * nEmbd, nEmbd]).mul(projStd), true, `layer${i}_mlp_proj`),
        });
      }

      // Precompute RoPE (simplified rotary embedding)
      const invFreq = tf.div(1.0, tf.pow(10000.0, tf.div(tf.range(0, headDim, 2), headDim))) as tf.Tensor1D;
      const t = tf.range(0, sequenceLen);
      const freqs = tf.outerProduct(t, invFreq);
      const emb = tf.concat([freqs, freqs], -1);
      
      // We keep these as tensors, not variables, since they don't train
      this.cosTable = tf.keep(tf.cos(emb));
      this.sinTable = tf.keep(tf.sin(emb));
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
  private applyRotary(x: tf.Tensor, isK: boolean = false): tf.Tensor {
    return tf.tidy(() => {
      const [B, T, numHeads, headDim] = x.shape;
      
      // Split into half (x1, x2) for rotary
      const x1 = x.slice([0, 0, 0, 0], [B, T, numHeads, headDim / 2]);
      const x2 = x.slice([0, 0, 0, headDim / 2], [B, T, numHeads, headDim / 2]);
      
      // rotated x: [-x2, x1]
      const halfRotated = tf.concat([tf.neg(x2), x1], -1);
      
      // Broadcast cos and sin to [B, T, numHeads, headDim]
      // Current cosTable is [T, headDim]. Needs reshape to [1, T, 1, headDim] for broadcast
      const sliceLen = T;
      const cos = this.cosTable.slice([0, 0], [sliceLen, -1]).reshape([1, T, 1, headDim]);
      const sin = this.sinTable.slice([0, 0], [sliceLen, -1]).reshape([1, T, 1, headDim]);
      
      return x.mul(cos).add(halfRotated.mul(sin));
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

      // 1. Token embeddings: [B, T] -> [B, T, D]
      let x = tf.gather(this.wte, inputIds);
      let xNorm = this.rmsNorm(x);
      
      const x0 = xNorm;
      let current = xNorm;

      // Causal mask: [T, T], where upper triangle is -1e9
      const mask = tf.linalg.bandPart(tf.ones([T, T]), -1, 0).sub(1).mul(1e9);

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
        
        // Apply RoPE
        q = this.applyRotary(q);
        k = this.applyRotary(k, true);
        
        // RMSNorm on Q and K (head-wise)
        q = this.rmsNorm(q);
        k = this.rmsNorm(k);
        
        // Transpose for attention -> [B, nHead, T, headDim]
        q = q.transpose([0, 2, 1, 3]);
        k = k.transpose([0, 2, 1, 3]);
        v = v.transpose([0, 2, 1, 3]);
        
        // Causal Attention: softmax(Q*K^T / sqrt(d) + mask) * V
        const scores = tf.matMul(q, k, false, true).div(Math.sqrt(headDim));
        const maskedScores = scores.add(mask.reshape([1, 1, T, T]));
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
    }
    return map;
  }
}
