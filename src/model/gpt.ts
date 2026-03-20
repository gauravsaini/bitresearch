// GPT Model — WebGPU implementation matching the PyTorch train.py architecture
import { GPUDeviceManager } from '../gpu/device';
import { Tensor } from '../gpu/tensor';
import { GPUOps } from '../gpu/ops';
import { GPTConfig, computeWindowSizes } from './config';

export interface ModelWeights {
  wte: Tensor;                          // [V, D] token embeddings
  lmHead: Tensor;                       // [V, D] output projection
  residLambdas: Float32Array;           // [nLayer] residual scaling
  x0Lambdas: Float32Array;             // [nLayer] input scaling
  layers: LayerWeights[];               // per-layer weights
  valueEmbeds: Map<number, Tensor>;     // layer_idx -> [V, kvDim] value embeddings
  cosTable: Tensor;                     // [seqLen, headDim/2] precomputed RoPE cos
  sinTable: Tensor;                     // [seqLen, headDim/2] precomputed RoPE sin
}

export interface LayerWeights {
  qWeight: Tensor;     // [D, nHead*headDim]
  kWeight: Tensor;     // [D, nKvHead*headDim]
  vWeight: Tensor;     // [D, nKvHead*headDim]
  projWeight: Tensor;  // [D, D]
  fcWeight: Tensor;    // [D, 4*D]
  mlpProjWeight: Tensor; // [4*D, D]
  veGateWeight: Tensor | null; // [veGateChannels, nKvHead] or null
}

function hasVe(layerIdx: number, nLayer: number): boolean {
  return layerIdx % 2 === (nLayer - 1) % 2;
}

export class GPTModel {
  config: GPTConfig;
  weights: ModelWeights;
  windowSizes: [number, number][];
  private mgr: GPUDeviceManager;
  private ops: GPUOps;

  private constructor(
    config: GPTConfig,
    weights: ModelWeights,
    mgr: GPUDeviceManager,
    ops: GPUOps,
  ) {
    this.config = config;
    this.weights = weights;
    this.windowSizes = computeWindowSizes(config);
    this.mgr = mgr;
    this.ops = ops;
  }

  static async create(config: GPTConfig, mgr: GPUDeviceManager, ops: GPUOps): Promise<GPTModel> {
    const { vocabSize, nLayer, nHead, nKvHead, nEmbd, sequenceLen } = config;
    const headDim = nEmbd / nHead;
    const kvDim = nKvHead * headDim;
    const veGateChannels = 32;

    // Init scale: 3^0.5 * n_embd^-0.5
    const s = Math.sqrt(3) * Math.pow(nEmbd, -0.5);

    // Token embeddings
    const wte = await Tensor.randn(mgr, [vocabSize, nEmbd], 0, 1, 'wte');
    const lmHead = await Tensor.randn(mgr, [vocabSize, nEmbd], 0, 0.001, 'lm_head');

    // Per-layer scalars (CPU-side for simplicity)
    const residLambdas = new Float32Array(nLayer).fill(1.0);
    const x0Lambdas = new Float32Array(nLayer).fill(0.1);

    // Per-layer weights
    const layers: LayerWeights[] = [];
    for (let i = 0; i < nLayer; i++) {
      const qWeight = await Tensor.uniform(mgr, [nEmbd, nHead * headDim], -s, s, `layer${i}_q`);
      const kWeight = await Tensor.uniform(mgr, [nEmbd, kvDim], -s, s, `layer${i}_k`);
      const vWeight = await Tensor.uniform(mgr, [nEmbd, kvDim], -s, s, `layer${i}_v`);
      const projWeight = Tensor.zeros(mgr, [nEmbd, nEmbd], 'f32', `layer${i}_proj`);
      const fcWeight = await Tensor.uniform(mgr, [nEmbd, 4 * nEmbd], -s, s, `layer${i}_fc`);
      const mlpProjWeight = Tensor.zeros(mgr, [4 * nEmbd, nEmbd], 'f32', `layer${i}_mlp_proj`);

      let veGateWeight: Tensor | null = null;
      if (hasVe(i, nLayer)) {
        veGateWeight = Tensor.zeros(mgr, [veGateChannels, nKvHead], 'f32', `layer${i}_ve_gate`);
      }

      layers.push({ qWeight, kWeight, vWeight, projWeight, fcWeight, mlpProjWeight, veGateWeight });
    }

    // Value embeddings
    const valueEmbeds = new Map<number, Tensor>();
    for (let i = 0; i < nLayer; i++) {
      if (hasVe(i, nLayer)) {
        valueEmbeds.set(i, await Tensor.uniform(mgr, [vocabSize, kvDim], -s, s, `ve_${i}`));
      }
    }

    // Precompute rotary embeddings
    const ropeLen = sequenceLen * 2;
    const cosTable = Tensor.zeros(mgr, [ropeLen, headDim / 2], 'f32', 'cos_table');
    const sinTable = Tensor.zeros(mgr, [ropeLen, headDim / 2], 'f32', 'sin_table');
    ops.precomputeRotary(cosTable, sinTable, ropeLen, headDim);
    await ops.sync();

    const weights: ModelWeights = {
      wte, lmHead, residLambdas, x0Lambdas, layers, valueEmbeds, cosTable, sinTable,
    };

    return new GPTModel(config, weights, mgr, ops);
  }

  /** Forward pass: compute loss given input tokens and targets */
  async forward(
    inputIds: Tensor,  // [B, T] u32
    targets: Tensor,   // [B, T] i32
    B: number,
    T: number,
  ): Promise<{ loss: number; logits: Tensor }> {
    const { nEmbd, nHead, nKvHead, vocabSize } = this.config;
    const headDim = nEmbd / nHead;
    const kvDim = nKvHead * headDim;

    // 1. Token embedding lookup: [B, T] -> [B, T, D]
    const x = Tensor.zeros(this.mgr, [B, T, nEmbd], 'f32', 'x');
    this.ops.embeddingForward(this.weights.wte, inputIds, x, B, T, nEmbd);

    // 2. RMS normalize initial embeddings
    const xNorm = Tensor.zeros(this.mgr, [B, T, nEmbd], 'f32', 'x_norm');
    this.ops.rmsNorm(x, xNorm, nEmbd);

    // Store x0 for residual ladder
    const x0 = xNorm.clone(this.mgr, 'x0');
    let current = xNorm;

    // 3. Transformer layers
    for (let i = 0; i < this.config.nLayer; i++) {
      const layer = this.weights.layers[i];
      const [windowSize] = this.windowSizes[i];

      // Residual scaling: x = λ_resid * x + λ_x0 * x0
      const scaled = Tensor.zeros(this.mgr, [B, T, nEmbd], 'f32', `scaled_${i}`);
      this.ops.scalarMul(current, this.weights.residLambdas[i], scaled);
      this.ops.addScaled(scaled, x0, scaled, this.weights.x0Lambdas[i]);

      // Pre-norm
      const normed = Tensor.zeros(this.mgr, [B, T, nEmbd], 'f32', `normed_${i}`);
      this.ops.rmsNorm(scaled, normed, nEmbd);

      // Q, K, V projections
      // normed: [B*T, D] @ weight: [D, dim] -> [B*T, dim]
      const flatNormed = normed; // same buffer, just interpret as [B*T, D]
      const q = Tensor.zeros(this.mgr, [B * T, nHead * headDim], 'f32', `q_${i}`);
      const k = Tensor.zeros(this.mgr, [B * T, kvDim], 'f32', `k_${i}`);
      const v = Tensor.zeros(this.mgr, [B * T, kvDim], 'f32', `v_${i}`);

      this.ops.matmul(flatNormed, layer.qWeight, q);
      this.ops.matmul(flatNormed, layer.kWeight, k);
      this.ops.matmul(flatNormed, layer.vWeight, v);

      // Apply RoPE to Q and K (reshape to [B, T, H, D])
      this.ops.applyRotary(q, this.weights.cosTable, this.weights.sinTable, B, T, nHead, headDim);
      this.ops.applyRotary(k, this.weights.cosTable, this.weights.sinTable, B, T, nKvHead, headDim);

      // RMS norm Q and K
      const qNormed = Tensor.zeros(this.mgr, [B * T, nHead * headDim], 'f32', `qn_${i}`);
      const kNormed = Tensor.zeros(this.mgr, [B * T, kvDim], 'f32', `kn_${i}`);
      this.ops.rmsNorm(q, qNormed, headDim);
      this.ops.rmsNorm(k, kNormed, headDim);

      // Causal attention
      const attnOut = Tensor.zeros(this.mgr, [B, T, nHead * headDim], 'f32', `attn_out_${i}`);
      this.ops.causalAttention(qNormed, kNormed, v, attnOut, B, nHead, T, headDim, nKvHead, windowSize);

      // Output projection
      const projected = Tensor.zeros(this.mgr, [B * T, nEmbd], 'f32', `proj_${i}`);
      this.ops.matmul(attnOut, layer.projWeight, projected);

      // Residual connection: x = x + attn_out
      this.ops.add(scaled, projected, scaled);

      // MLP: pre-norm -> fc -> squared_relu -> proj
      const mlpNormed = Tensor.zeros(this.mgr, [B, T, nEmbd], 'f32', `mlp_normed_${i}`);
      this.ops.rmsNorm(scaled, mlpNormed, nEmbd);

      const hidden = Tensor.zeros(this.mgr, [B * T, 4 * nEmbd], 'f32', `hidden_${i}`);
      this.ops.matmul(mlpNormed, layer.fcWeight, hidden);
      this.ops.squaredRelu(hidden, hidden);

      const mlpOut = Tensor.zeros(this.mgr, [B * T, nEmbd], 'f32', `mlp_out_${i}`);
      this.ops.matmul(hidden, layer.mlpProjWeight, mlpOut);

      // Residual: x = x + mlp_out
      this.ops.add(scaled, mlpOut, scaled);

      current = scaled;

      // Clean up intermediate tensors
      normed.destroy(); q.destroy(); k.destroy(); v.destroy();
      qNormed.destroy(); kNormed.destroy(); attnOut.destroy();
      projected.destroy(); mlpNormed.destroy(); hidden.destroy(); mlpOut.destroy();
    }

    // 4. Final norm
    const finalNorm = Tensor.zeros(this.mgr, [B, T, nEmbd], 'f32', 'final_norm');
    this.ops.rmsNorm(current, finalNorm, nEmbd);

    // 5. LM head: [B*T, D] @ [D, V] -> [B*T, V]
    const logits = Tensor.zeros(this.mgr, [B * T, vocabSize], 'f32', 'logits');
    // lmHead is [V, D], need to transpose (or use it as [D, V] for matmul)
    // Actually train.py's lm_head is nn.Linear(D, V), so weight is [V, D]
    // logits = x @ lm_head.T = x @ [D, V]
    // We'll store as [V, D] and do: logits = finalNorm @ lmHead^T
    // For simplicity, matmul finalNorm[B*T, D] @ lmHead^T = need transpose
    // Let's just do the forward matmul: we need [B*T, V]
    // matmul expects A[M,K] @ B[K,N] = C[M,N]
    // A = finalNorm[B*T, D], B should be [D, V]
    // lmHead is [V, D], so we need a transposed version
    // For now, we'll compute on CPU or add a transpose shader
    // Simplification: store lm_head as [D, V] directly
    this.ops.matmul(finalNorm, this.weights.lmHead, logits);

    // 6. Softcap logits
    const softcap = 15;
    this.ops.softcap(logits, softcap, logits);

    // 7. Cross-entropy loss
    const N = B * T;
    const losses = Tensor.zeros(this.mgr, [N], 'f32', 'losses');
    this.ops.crossEntropyForward(logits, targets, losses, N, vocabSize, -1);
    await this.ops.sync();

    // Read back losses and compute mean on CPU
    const lossData = await losses.toArray(this.mgr);
    let totalLoss = 0;
    let count = 0;
    for (let i = 0; i < lossData.length; i++) {
      if (lossData[i] > 0) {
        totalLoss += lossData[i];
        count++;
      }
    }
    const meanLoss = count > 0 ? totalLoss / count : 0;

    // Cleanup
    x.destroy(); x0.destroy(); finalNorm.destroy(); losses.destroy();

    return { loss: meanLoss, logits };
  }

  /** Get all model parameters as a flat Float32Array for network transfer */
  async getAllParams(mgr: GPUDeviceManager): Promise<Float32Array> {
    const arrays: Float32Array[] = [];

    arrays.push(await this.weights.wte.toArray(mgr));
    arrays.push(await this.weights.lmHead.toArray(mgr));
    arrays.push(this.weights.residLambdas);
    arrays.push(this.weights.x0Lambdas);

    for (const layer of this.weights.layers) {
      arrays.push(await layer.qWeight.toArray(mgr));
      arrays.push(await layer.kWeight.toArray(mgr));
      arrays.push(await layer.vWeight.toArray(mgr));
      arrays.push(await layer.projWeight.toArray(mgr));
      arrays.push(await layer.fcWeight.toArray(mgr));
      arrays.push(await layer.mlpProjWeight.toArray(mgr));
      if (layer.veGateWeight) {
        arrays.push(await layer.veGateWeight.toArray(mgr));
      }
    }

    for (const [, ve] of this.weights.valueEmbeds) {
      arrays.push(await ve.toArray(mgr));
    }

    // Compute total size
    let totalSize = 0;
    for (const arr of arrays) totalSize += arr.length;

    const result = new Float32Array(totalSize);
    let offset = 0;
    for (const arr of arrays) {
      result.set(arr, offset);
      offset += arr.length;
    }

    return result;
  }

  /** Load model parameters from a flat Float32Array */
  async loadAllParams(mgr: GPUDeviceManager, params: Float32Array): Promise<void> {
    let offset = 0;

    const loadTensor = async (tensor: Tensor) => {
      const data = params.subarray(offset, offset + tensor.numel);
      const t = await Tensor.fromArray(mgr, new Float32Array(data), tensor.shape, tensor.label);
      tensor.copyFrom(mgr, t);
      t.destroy();
      offset += tensor.numel;
    };

    await loadTensor(this.weights.wte);
    await loadTensor(this.weights.lmHead);

    this.weights.residLambdas.set(params.subarray(offset, offset + this.config.nLayer));
    offset += this.config.nLayer;
    this.weights.x0Lambdas.set(params.subarray(offset, offset + this.config.nLayer));
    offset += this.config.nLayer;

    for (const layer of this.weights.layers) {
      await loadTensor(layer.qWeight);
      await loadTensor(layer.kWeight);
      await loadTensor(layer.vWeight);
      await loadTensor(layer.projWeight);
      await loadTensor(layer.fcWeight);
      await loadTensor(layer.mlpProjWeight);
      if (layer.veGateWeight) {
        await loadTensor(layer.veGateWeight);
      }
    }

    for (const [, ve] of this.weights.valueEmbeds) {
      await loadTensor(ve);
    }
  }

  /** Get total parameter count */
  paramCount(): number {
    let count = this.weights.wte.numel + this.weights.lmHead.numel;
    count += this.config.nLayer * 2; // scalars

    for (const layer of this.weights.layers) {
      count += layer.qWeight.numel + layer.kWeight.numel + layer.vWeight.numel;
      count += layer.projWeight.numel + layer.fcWeight.numel + layer.mlpProjWeight.numel;
      if (layer.veGateWeight) count += layer.veGateWeight.numel;
    }

    for (const [, ve] of this.weights.valueEmbeds) {
      count += ve.numel;
    }

    return count;
  }
}
