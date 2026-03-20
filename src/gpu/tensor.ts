// GPU Tensor — thin wrapper around GPUBuffer with shape metadata
import { GPUDeviceManager } from './device';

export type TensorShape = number[];
export type TensorDType = 'f32' | 'u32' | 'i32';

const DTYPE_BYTES: Record<TensorDType, number> = { f32: 4, u32: 4, i32: 4 };

export class Tensor {
  buffer: GPUBuffer;
  shape: TensorShape;
  dtype: TensorDType;
  strides: number[];
  numel: number;
  byteSize: number;
  label: string;

  // Autograd fields
  grad: Tensor | null = null;
  requiresGrad: boolean = false;
  gradFn: (() => void) | null = null;
  private children: Tensor[] = [];

  constructor(
    buffer: GPUBuffer,
    shape: TensorShape,
    dtype: TensorDType = 'f32',
    label: string = '',
  ) {
    this.buffer = buffer;
    this.shape = [...shape];
    this.dtype = dtype;
    this.label = label;
    this.numel = shape.reduce((a, b) => a * b, 1);
    this.byteSize = this.numel * DTYPE_BYTES[dtype];
    this.strides = Tensor.computeStrides(shape);
  }

  static computeStrides(shape: TensorShape): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  get ndim(): number {
    return this.shape.length;
  }

  get rows(): number {
    return this.shape.length >= 2 ? this.shape[this.shape.length - 2] : 1;
  }

  get cols(): number {
    return this.shape[this.shape.length - 1];
  }

  /** Create a tensor from CPU data */
  static async fromArray(
    mgr: GPUDeviceManager,
    data: Float32Array | Uint32Array | Int32Array,
    shape: TensorShape,
    label: string = '',
    usage?: number,
  ): Promise<Tensor> {
    const dtype: TensorDType =
      data instanceof Float32Array ? 'f32' : data instanceof Uint32Array ? 'u32' : 'i32';

    const bufferUsage =
      usage ?? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    const buffer = mgr.device.createBuffer({
      label,
      size: data.byteLength,
      usage: bufferUsage,
      mappedAtCreation: true,
    });

    const mapped = buffer.getMappedRange();
    if (data instanceof Float32Array) {
      new Float32Array(mapped).set(data);
    } else if (data instanceof Uint32Array) {
      new Uint32Array(mapped).set(data);
    } else {
      new Int32Array(mapped).set(data);
    }
    buffer.unmap();

    return new Tensor(buffer, shape, dtype, label);
  }

  /** Create a zero-filled tensor */
  static zeros(
    mgr: GPUDeviceManager,
    shape: TensorShape,
    dtype: TensorDType = 'f32',
    label: string = '',
  ): Tensor {
    const numel = shape.reduce((a, b) => a * b, 1);
    const byteSize = numel * DTYPE_BYTES[dtype];
    const buffer = mgr.device.createBuffer({
      label,
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    return new Tensor(buffer, shape, dtype, label);
  }

  /** Create tensor with random normal values */
  static async randn(
    mgr: GPUDeviceManager,
    shape: TensorShape,
    mean: number = 0,
    std: number = 1,
    label: string = '',
  ): Promise<Tensor> {
    const numel = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(numel);
    // Box-Muller transform
    for (let i = 0; i < numel; i += 2) {
      const u1 = Math.random();
      const u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1));
      const theta = 2 * Math.PI * u2;
      data[i] = mean + std * r * Math.cos(theta);
      if (i + 1 < numel) {
        data[i + 1] = mean + std * r * Math.sin(theta);
      }
    }
    return Tensor.fromArray(mgr, data, shape, label);
  }

  /** Create tensor with uniform random values */
  static async uniform(
    mgr: GPUDeviceManager,
    shape: TensorShape,
    low: number = -1,
    high: number = 1,
    label: string = '',
  ): Promise<Tensor> {
    const numel = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(numel);
    const range = high - low;
    for (let i = 0; i < numel; i++) {
      data[i] = low + Math.random() * range;
    }
    return Tensor.fromArray(mgr, data, shape, label);
  }

  /** Read tensor data back to CPU */
  async toArray(mgr: GPUDeviceManager): Promise<Float32Array> {
    const readBuffer = mgr.device.createBuffer({
      size: this.byteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = mgr.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffer, 0, readBuffer, 0, this.byteSize);
    mgr.device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();

    return result;
  }

  /** Copy this tensor to a new tensor */
  clone(mgr: GPUDeviceManager, label?: string): Tensor {
    const newBuffer = mgr.device.createBuffer({
      label: label || `${this.label}_clone`,
      size: this.byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const encoder = mgr.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffer, 0, newBuffer, 0, this.byteSize);
    mgr.device.queue.submit([encoder.finish()]);

    return new Tensor(newBuffer, this.shape, this.dtype, label || `${this.label}_clone`);
  }

  /** Copy data from another tensor into this tensor */
  copyFrom(mgr: GPUDeviceManager, src: Tensor): void {
    const encoder = mgr.device.createCommandEncoder();
    encoder.copyBufferToBuffer(src.buffer, 0, this.buffer, 0, Math.min(this.byteSize, src.byteSize));
    mgr.device.queue.submit([encoder.finish()]);
  }

  /** Serialize tensor to ArrayBuffer for network transfer */
  async serialize(mgr: GPUDeviceManager): Promise<ArrayBuffer> {
    const data = await this.toArray(mgr);
    // Header: ndim(u32) + shape(u32 * ndim) + data(f32 * numel)
    const headerSize = 4 + this.shape.length * 4;
    const totalSize = headerSize + data.byteLength;
    const buf = new ArrayBuffer(totalSize);
    const view = new DataView(buf);
    let offset = 0;

    // ndim
    view.setUint32(offset, this.shape.length, true);
    offset += 4;
    // shape
    for (const s of this.shape) {
      view.setUint32(offset, s, true);
      offset += 4;
    }
    // data
    new Float32Array(buf, offset).set(data);

    return buf;
  }

  /** Deserialize tensor from ArrayBuffer */
  static async deserialize(
    mgr: GPUDeviceManager,
    buf: ArrayBuffer,
    label: string = '',
  ): Promise<Tensor> {
    const view = new DataView(buf);
    let offset = 0;

    const ndim = view.getUint32(offset, true);
    offset += 4;

    const shape: number[] = [];
    for (let i = 0; i < ndim; i++) {
      shape.push(view.getUint32(offset, true));
      offset += 4;
    }

    const data = new Float32Array(buf, offset);
    return Tensor.fromArray(mgr, data, shape, label);
  }

  /** Release the GPU buffer */
  destroy(): void {
    this.buffer.destroy();
  }

  /** Set autograd tracking */
  setRequiresGrad(val: boolean): Tensor {
    this.requiresGrad = val;
    return this;
  }

  /** Register child tensors for autograd graph */
  addChildren(...children: Tensor[]): void {
    this.children = children;
  }

  /** Backward pass for autograd */
  backward(): void {
    // Topological sort
    const order: Tensor[] = [];
    const visited = new Set<Tensor>();

    const topo = (t: Tensor) => {
      if (visited.has(t)) return;
      visited.add(t);
      for (const child of t.children) {
        topo(child);
      }
      order.push(t);
    };
    topo(this);
    order.reverse();

    for (const t of order) {
      if (t.gradFn) {
        t.gradFn();
      }
    }
  }
}
