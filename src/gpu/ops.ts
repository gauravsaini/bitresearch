// GPU Operations — high-level API wrapping WGSL compute shaders
import { GPUDeviceManager } from './device';
import { Tensor } from './tensor';

// Import shaders as raw strings (Vite handles ?raw imports)
import matmulShader from './shaders/matmul.wgsl?raw';
import elementwiseShader from './shaders/elementwise.wgsl?raw';
import attentionShader from './shaders/attention.wgsl?raw';
import normalizationShader from './shaders/normalization.wgsl?raw';
import embeddingShader from './shaders/embedding.wgsl?raw';
import crossEntropyShader from './shaders/cross_entropy.wgsl?raw';
import rotaryShader from './shaders/rotary.wgsl?raw';

export class GPUOps {
  private mgr: GPUDeviceManager;
  private modules: Map<string, GPUShaderModule> = new Map();

  constructor(mgr: GPUDeviceManager) {
    this.mgr = mgr;
    this.initModules();
  }

  private initModules(): void {
    this.modules.set('matmul', this.mgr.createShaderModule(matmulShader, 'matmul'));
    this.modules.set('elementwise', this.mgr.createShaderModule(elementwiseShader, 'elementwise'));
    this.modules.set('attention', this.mgr.createShaderModule(attentionShader, 'attention'));
    this.modules.set('normalization', this.mgr.createShaderModule(normalizationShader, 'normalization'));
    this.modules.set('embedding', this.mgr.createShaderModule(embeddingShader, 'embedding'));
    this.modules.set('crossEntropy', this.mgr.createShaderModule(crossEntropyShader, 'crossEntropy'));
    this.modules.set('rotary', this.mgr.createShaderModule(rotaryShader, 'rotary'));
  }

  private createUniformBuffer(data: ArrayBuffer): GPUBuffer {
    // Pad to 16-byte alignment
    const paddedSize = Math.ceil(data.byteLength / 16) * 16;
    const padded = new ArrayBuffer(paddedSize);
    new Uint8Array(padded).set(new Uint8Array(data));

    const buffer = this.mgr.device.createBuffer({
      size: paddedSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(padded));
    buffer.unmap();
    return buffer;
  }

  // ── Matrix Multiplication ───────────────────────────────────────────
  /** C = alpha * A @ B + beta * C. A:[M,K], B:[K,N], C:[M,N] */
  matmul(A: Tensor, B: Tensor, C: Tensor, alpha = 1.0, beta = 0.0): void {
    const M = A.shape[A.ndim - 2];
    const K = A.shape[A.ndim - 1];
    const N = B.shape[B.ndim - 1];

    const paramsData = new ArrayBuffer(32);
    const view = new DataView(paramsData);
    view.setUint32(0, M, true);
    view.setUint32(4, K, true);
    view.setUint32(8, N, true);
    view.setFloat32(12, alpha, true);
    view.setFloat32(16, beta, true);
    const paramsBuffer = this.createUniformBuffer(paramsData);

    const bindGroupLayout = this.mgr.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.mgr.getOrCreatePipeline(
      'matmul', this.modules.get('matmul')!, 'matmul', bindGroupLayout
    );

    const bindGroup = this.mgr.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: A.buffer } },
        { binding: 1, resource: { buffer: B.buffer } },
        { binding: 2, resource: { buffer: C.buffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const encoder = this.mgr.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(N / 16),
      Math.ceil(M / 16),
    );
    pass.end();
    this.mgr.device.queue.submit([encoder.finish()]);
    paramsBuffer.destroy();
  }

  // ── Element-wise Operations ─────────────────────────────────────────
  private dispatchElementwise(
    entryPoint: string,
    inputs: Tensor[],
    output: Tensor,
    scalar: number = 0,
  ): void {
    const size = output.numel;
    const paramsData = new ArrayBuffer(16);
    const view = new DataView(paramsData);
    view.setUint32(0, size, true);
    view.setFloat32(4, scalar, true);
    const paramsBuffer = this.createUniformBuffer(paramsData);

    const layoutEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ];
    const bindEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: inputs[0].buffer } },
      { binding: 1, resource: { buffer: output.buffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ];

    if (inputs.length > 1) {
      layoutEntries.push({
        binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' },
      });
      bindEntries.push({
        binding: 3, resource: { buffer: inputs[1].buffer },
      });
    }

    const bindGroupLayout = this.mgr.device.createBindGroupLayout({ entries: layoutEntries });
    const pipeline = this.mgr.getOrCreatePipeline(
      `elementwise_${entryPoint}`, this.modules.get('elementwise')!, entryPoint, bindGroupLayout
    );
    const bindGroup = this.mgr.device.createBindGroup({
      layout: bindGroupLayout, entries: bindEntries,
    });

    const encoder = this.mgr.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(size / 256));
    pass.end();
    this.mgr.device.queue.submit([encoder.finish()]);
    paramsBuffer.destroy();
  }

  add(a: Tensor, b: Tensor, out: Tensor): void {
    this.dispatchElementwise('add', [a, b], out);
  }

  mul(a: Tensor, b: Tensor, out: Tensor): void {
    this.dispatchElementwise('mul', [a, b], out);
  }

  addScaled(a: Tensor, b: Tensor, out: Tensor, scale: number): void {
    this.dispatchElementwise('add_scaled', [a, b], out, scale);
  }

  scalarMul(a: Tensor, scalar: number, out: Tensor): void {
    this.dispatchElementwise('mul_scalar', [a], out, scalar);
  }

  relu(a: Tensor, out: Tensor): void {
    this.dispatchElementwise('relu', [a], out);
  }

  squaredRelu(a: Tensor, out: Tensor): void {
    this.dispatchElementwise('squared_relu', [a], out);
  }

  softcap(a: Tensor, cap: number, out: Tensor): void {
    this.dispatchElementwise('softcap', [a], out, cap);
  }

  sigmoid(a: Tensor, out: Tensor): void {
    this.dispatchElementwise('sigmoid', [a], out);
  }

  // ── RMS Normalization ───────────────────────────────────────────────
  rmsNorm(input: Tensor, output: Tensor, D: number, eps: number = 1e-6): void {
    const size = input.numel;
    const paramsData = new ArrayBuffer(16);
    const view = new DataView(paramsData);
    view.setUint32(0, size, true);
    view.setUint32(4, D, true);
    view.setFloat32(8, eps, true);
    view.setFloat32(12, 0, true); // padding
    const paramsBuffer = this.createUniformBuffer(paramsData);

    const bindGroupLayout = this.mgr.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.mgr.getOrCreatePipeline(
      'rms_norm', this.modules.get('normalization')!, 'rms_norm', bindGroupLayout
    );
    const bindGroup = this.mgr.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: output.buffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const numRows = size / D;
    const encoder = this.mgr.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(numRows);
    pass.end();
    this.mgr.device.queue.submit([encoder.finish()]);
    paramsBuffer.destroy();
  }

  // ── Embedding Lookup ────────────────────────────────────────────────
  embeddingForward(table: Tensor, indices: Tensor, output: Tensor, B: number, T: number, D: number): void {
    const V = table.shape[0];
    const paramsData = new ArrayBuffer(16);
    const view = new DataView(paramsData);
    view.setUint32(0, B, true);
    view.setUint32(4, T, true);
    view.setUint32(8, D, true);
    view.setUint32(12, V, true);
    const paramsBuffer = this.createUniformBuffer(paramsData);

    const bindGroupLayout = this.mgr.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.mgr.getOrCreatePipeline(
      'embedding_forward', this.modules.get('embedding')!, 'embedding_forward', bindGroupLayout
    );
    const bindGroup = this.mgr.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: table.buffer } },
        { binding: 1, resource: { buffer: indices.buffer } },
        { binding: 2, resource: { buffer: output.buffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const total = B * T * D;
    const encoder = this.mgr.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(total / 256));
    pass.end();
    this.mgr.device.queue.submit([encoder.finish()]);
    paramsBuffer.destroy();
  }

  // ── Causal Self-Attention ───────────────────────────────────────────
  causalAttention(
    Q: Tensor, K: Tensor, V: Tensor, O: Tensor,
    B: number, H: number, T: number, D: number, kvH: number, window: number,
  ): void {
    const paramsData = new ArrayBuffer(32);
    const view = new DataView(paramsData);
    view.setUint32(0, B, true);
    view.setUint32(4, H, true);
    view.setUint32(8, T, true);
    view.setUint32(12, D, true);
    view.setUint32(16, kvH, true);
    view.setUint32(20, window, true);
    const paramsBuffer = this.createUniformBuffer(paramsData);

    const bindGroupLayout = this.mgr.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.mgr.getOrCreatePipeline(
      'causal_attention', this.modules.get('attention')!, 'causal_attention', bindGroupLayout
    );
    const bindGroup = this.mgr.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: Q.buffer } },
        { binding: 1, resource: { buffer: K.buffer } },
        { binding: 2, resource: { buffer: V.buffer } },
        { binding: 3, resource: { buffer: O.buffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });

    const totalWorkItems = B * H * T;
    const encoder = this.mgr.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(totalWorkItems);
    pass.end();
    this.mgr.device.queue.submit([encoder.finish()]);
    paramsBuffer.destroy();
  }

  // ── Rotary Embeddings ───────────────────────────────────────────────
  precomputeRotary(cosOut: Tensor, sinOut: Tensor, seqLen: number, headDim: number, base = 10000): void {
    const paramsData = new ArrayBuffer(16);
    const view = new DataView(paramsData);
    view.setUint32(0, seqLen, true);
    view.setUint32(4, headDim, true);
    view.setFloat32(8, base, true);
    view.setFloat32(12, 0, true);
    const paramsBuffer = this.createUniformBuffer(paramsData);

    const bindGroupLayout = this.mgr.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.mgr.getOrCreatePipeline(
      'precompute_rotary', this.modules.get('rotary')!, 'precompute_rotary', bindGroupLayout
    );
    const bindGroup = this.mgr.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: cosOut.buffer } },
        { binding: 1, resource: { buffer: sinOut.buffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const total = seqLen * (headDim / 2);
    const encoder = this.mgr.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(total / 256));
    pass.end();
    this.mgr.device.queue.submit([encoder.finish()]);
    paramsBuffer.destroy();
  }

  applyRotary(x: Tensor, cosTable: Tensor, sinTable: Tensor, B: number, T: number, H: number, D: number): void {
    const paramsData = new ArrayBuffer(16);
    const view = new DataView(paramsData);
    view.setUint32(0, B, true);
    view.setUint32(4, T, true);
    view.setUint32(8, H, true);
    view.setUint32(12, D, true);
    const paramsBuffer = this.createUniformBuffer(paramsData);

    const bindGroupLayout = this.mgr.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.mgr.getOrCreatePipeline(
      'apply_rotary', this.modules.get('rotary')!, 'apply_rotary', bindGroupLayout
    );
    const bindGroup = this.mgr.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: x.buffer } },
        { binding: 1, resource: { buffer: cosTable.buffer } },
        { binding: 2, resource: { buffer: sinTable.buffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const totalPairs = B * T * H * (D / 2);
    const encoder = this.mgr.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(totalPairs / 256));
    pass.end();
    this.mgr.device.queue.submit([encoder.finish()]);
    paramsBuffer.destroy();
  }

  // ── Cross-Entropy Loss ──────────────────────────────────────────────
  crossEntropyForward(logits: Tensor, targets: Tensor, losses: Tensor, N: number, V: number, ignoreIdx = -1): void {
    const paramsData = new ArrayBuffer(16);
    const view = new DataView(paramsData);
    view.setUint32(0, N, true);
    view.setUint32(4, V, true);
    view.setInt32(8, ignoreIdx, true);
    view.setUint32(12, 0, true);
    const paramsBuffer = this.createUniformBuffer(paramsData);

    const bindGroupLayout = this.mgr.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.mgr.getOrCreatePipeline(
      'cross_entropy_forward', this.modules.get('crossEntropy')!, 'cross_entropy_forward', bindGroupLayout
    );
    const bindGroup = this.mgr.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: logits.buffer } },
        { binding: 1, resource: { buffer: targets.buffer } },
        { binding: 2, resource: { buffer: losses.buffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const encoder = this.mgr.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(N);
    pass.end();
    this.mgr.device.queue.submit([encoder.finish()]);
    paramsBuffer.destroy();
  }

  // ── Utility: await GPU completion ───────────────────────────────────
  async sync(): Promise<void> {
    await this.mgr.device.queue.onSubmittedWorkDone();
  }
}
