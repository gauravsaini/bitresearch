// WebGPU Device Manager — singleton initialization and pipeline caching

export class GPUDeviceManager {
  private static instance: GPUDeviceManager | null = null;
  device!: GPUDevice;
  private pipelineCache = new Map<string, GPUComputePipeline>();
  private shaderCache = new Map<string, GPUShaderModule>();
  adapterInfo: GPUAdapterInfo | null = null;
  maxWorkgroupSize: number = 256;
  maxBufferSize: number = 256 * 1024 * 1024; // 256MB default

  static async create(): Promise<GPUDeviceManager> {
    if (GPUDeviceManager.instance) return GPUDeviceManager.instance;

    const mgr = new GPUDeviceManager();
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser. Use Chrome 113+ or Edge 113+.');
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) {
      throw new Error('No WebGPU adapter found. Ensure GPU drivers are up to date.');
    }

    // requestAdapterInfo may be a method or property depending on browser
    try {
      const a = adapter as any;
      mgr.adapterInfo = typeof a.requestAdapterInfo === 'function'
        ? await a.requestAdapterInfo()
        : a.info || null;
    } catch {
      mgr.adapterInfo = null;
    }
    const limits = adapter.limits;
    mgr.maxWorkgroupSize = limits.maxComputeWorkgroupSizeX;
    mgr.maxBufferSize = limits.maxBufferSize;

    mgr.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(limits.maxStorageBufferBindingSize, 1024 * 1024 * 1024),
        maxBufferSize: Math.min(limits.maxBufferSize, 1024 * 1024 * 1024),
        maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ,
        maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
        maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
      },
    });

    mgr.device.lost.then((info) => {
      console.error(`WebGPU device lost: ${info.message}`);
      GPUDeviceManager.instance = null;
    });

    GPUDeviceManager.instance = mgr;
    return mgr;
  }

  createShaderModule(code: string, label?: string): GPUShaderModule {
    const key = label || code;
    let module = this.shaderCache.get(key);
    if (!module) {
      module = this.device.createShaderModule({ code, label });
      this.shaderCache.set(key, module);
    }
    return module;
  }

  getOrCreatePipeline(
    label: string,
    shaderModule: GPUShaderModule,
    entryPoint: string,
    bindGroupLayout: GPUBindGroupLayout,
  ): GPUComputePipeline {
    const key = `${label}:${entryPoint}`;
    let pipeline = this.pipelineCache.get(key);
    if (!pipeline) {
      pipeline = this.device.createComputePipeline({
        label,
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout],
        }),
        compute: { module: shaderModule, entryPoint },
      });
      this.pipelineCache.set(key, pipeline);
    }
    return pipeline;
  }

  getDeviceInfo(): { vendor: string; architecture: string; description: string } {
    return {
      vendor: this.adapterInfo?.vendor || 'unknown',
      architecture: this.adapterInfo?.architecture || 'unknown',
      description: this.adapterInfo?.description || 'unknown',
    };
  }
}
