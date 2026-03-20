// Worker client — connects to coordinator, receives params, trains, sends gradients
import { GPUDeviceManager } from '../gpu/device';
import { Tensor } from '../gpu/tensor';
import { GPUOps } from '../gpu/ops';
import { GPTModel } from '../model/gpt';
import { GPTConfig } from '../model/config';
import { MessageType, encodeMessage, decodeMessage } from './protocol';

export type WorkerState = 'connecting' | 'idle' | 'training' | 'syncing' | 'error' | 'disconnected';

export interface WorkerMetrics {
  state: WorkerState;
  currentStep: number;
  lastLoss: number;
  avgComputeTimeMs: number;
  tokensProcessed: number;
  stepsCompleted: number;
  gpuInfo: string;
}

export class WorkerClient {
  private ws: WebSocket | null = null;
  private mgr: GPUDeviceManager | null = null;
  private ops: GPUOps | null = null;
  private model: GPTModel | null = null;
  private config: GPTConfig | null = null;
  private workerId: string;
  private batchSize: number = 4;

  state: WorkerState = 'connecting';
  metrics: WorkerMetrics;
  onStateChange: ((state: WorkerState) => void) | null = null;
  onMetricsUpdate: ((metrics: WorkerMetrics) => void) | null = null;
  onLog: ((msg: string) => void) | null = null;

  constructor() {
    this.workerId = `worker-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
    this.metrics = {
      state: 'connecting',
      currentStep: 0,
      lastLoss: 0,
      avgComputeTimeMs: 0,
      tokensProcessed: 0,
      stepsCompleted: 0,
      gpuInfo: 'Initializing...',
    };
  }

  private log(msg: string): void {
    console.log(`[Worker ${this.workerId.slice(-6)}] ${msg}`);
    this.onLog?.(msg);
  }

  private setState(state: WorkerState): void {
    this.state = state;
    this.metrics.state = state;
    this.onStateChange?.(state);
    this.onMetricsUpdate?.(this.metrics);
  }

  /** Initialize WebGPU and connect to coordinator */
  async start(serverUrl: string): Promise<void> {
    try {
      // Initialize WebGPU
      this.log('Initializing WebGPU...');
      this.mgr = await GPUDeviceManager.create();
      this.ops = new GPUOps(this.mgr);

      const deviceInfo = this.mgr.getDeviceInfo();
      this.metrics.gpuInfo = `${deviceInfo.vendor} ${deviceInfo.architecture}`;
      this.log(`GPU: ${this.metrics.gpuInfo}`);

      // Connect to coordinator
      this.log(`Connecting to ${serverUrl}...`);
      this.ws = new WebSocket(serverUrl);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        this.log('Connected! Sending join request...');
        this.sendMessage({
          type: MessageType.WORKER_JOIN,
          workerId: this.workerId,
          deviceInfo: {
            ...deviceInfo,
            maxBufferSize: this.mgr!.maxBufferSize,
          },
          timestamp: Date.now(),
        });
      };

      this.ws.onmessage = async (event) => {
        try {
          const msg = decodeMessage(event.data as ArrayBuffer);
          await this.handleMessage(msg);
        } catch (e) {
          // Try JSON parsing for simple messages
          try {
            const msg = JSON.parse(event.data);
            await this.handleMessage(msg);
          } catch {
            this.log(`Failed to parse message: ${e}`);
          }
        }
      };

      this.ws.onclose = () => {
        this.log('Disconnected from coordinator');
        this.setState('disconnected');
      };

      this.ws.onerror = (e) => {
        this.log(`WebSocket error: ${e}`);
        this.setState('error');
      };

      // Heartbeat
      setInterval(() => {
        if (this.ws?.readyState === WebSocket.OPEN) {
          this.sendMessage({
            type: MessageType.WORKER_HEARTBEAT,
            workerId: this.workerId,
            timestamp: Date.now(),
            memoryUsage: 0, // WebGPU doesn't expose this
          });
        }
      }, 5000);
    } catch (e) {
      this.log(`Initialization failed: ${e}`);
      this.setState('error');
    }
  }

  private sendMessage(msg: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const encoded = typeof msg === 'string' ? msg : JSON.stringify(msg);
      this.ws.send(encoded);
    }
  }

  private async handleMessage(msg: any): Promise<void> {
    switch (msg.type) {
      case MessageType.WELCOME:
        this.log(`Welcome! Config: ${msg.config.nLayer}L/${msg.config.nEmbd}D, batch=${msg.batchSize}`);
        this.config = msg.config;
        this.batchSize = msg.batchSize;

        // Create model
        this.log('Creating model...');
        this.model = await GPTModel.create(this.config!, this.mgr!, this.ops!);
        this.log(`Model created: ${(this.model.paramCount() / 1e6).toFixed(1)}M params`);
        this.setState('idle');
        break;

      case MessageType.PARAMS_BROADCAST:
        this.log(`Received params for step ${msg.stepId}`);
        if (this.model && msg.params) {
          const params = msg.params instanceof Float32Array ? msg.params : new Float32Array(msg.params);
          await this.model.loadAllParams(this.mgr!, params);
        }
        this.setState('idle');
        break;

      case MessageType.BATCH_ASSIGN:
        await this.trainOnBatch(msg);
        break;

      case MessageType.TRAIN_STEP:
        this.log(`Starting training step ${msg.stepId}...`);
        this.metrics.currentStep = msg.stepId;
        break;

      case MessageType.KICK:
        this.log('Kicked by coordinator');
        this.ws?.close();
        break;
    }
  }

  private async trainOnBatch(msg: any): Promise<void> {
    if (!this.model || !this.mgr || !this.ops) {
      this.log('Model not ready, skipping batch');
      return;
    }

    this.setState('training');
    const startTime = performance.now();

    try {
      const B = this.batchSize;
      const T = this.config!.sequenceLen;

      // Create input tensors from received data
      const inputData = msg.inputIds instanceof Uint32Array ? msg.inputIds : new Uint32Array(msg.inputIds);
      const targetData = msg.targets instanceof Int32Array ? msg.targets : new Int32Array(msg.targets);

      const inputIds = await Tensor.fromArray(this.mgr, inputData, [B, T], 'input_ids');
      const targets = await Tensor.fromArray(
        this.mgr,
        new Float32Array(targetData), // Cross-entropy expects f32 but we'll cast
        [B, T],
        'targets',
      );

      // Forward pass
      const { loss } = await this.model.forward(inputIds, targets, B, T);

      const computeTimeMs = performance.now() - startTime;
      const tokensProcessed = B * T;

      // Update metrics
      this.metrics.lastLoss = loss;
      this.metrics.stepsCompleted++;
      this.metrics.tokensProcessed += tokensProcessed;
      this.metrics.avgComputeTimeMs =
        (this.metrics.avgComputeTimeMs * (this.metrics.stepsCompleted - 1) + computeTimeMs) /
        this.metrics.stepsCompleted;

      this.log(`Step ${msg.stepId} | loss=${loss.toFixed(6)} | ${computeTimeMs.toFixed(0)}ms | ${(tokensProcessed / (computeTimeMs / 1000)).toFixed(0)} tok/s`);

      // Send gradient/loss back (simplified: send loss for now, full gradients later)
      this.sendMessage({
        type: MessageType.GRADIENT_PUSH,
        workerId: this.workerId,
        stepId: msg.stepId,
        loss,
        tokensProcessed,
        computeTimeMs,
      });

      // Cleanup
      inputIds.destroy();
      targets.destroy();

      this.onMetricsUpdate?.(this.metrics);
    } catch (e) {
      this.log(`Training error: ${e}`);
      this.setState('error');
    }

    this.setState('idle');
  }

  stop(): void {
    this.ws?.close();
    this.setState('disconnected');
  }
}
