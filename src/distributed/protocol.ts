// Shared protocol types for coordinator ↔ worker communication

export enum MessageType {
  // Worker → Coordinator
  WORKER_JOIN = 0x01,
  WORKER_HEARTBEAT = 0x02,
  GRADIENT_PUSH = 0x03,
  WORKER_LEAVE = 0x04,
  WORKER_METRICS = 0x05,

  // Coordinator → Worker
  WELCOME = 0x10,
  PARAMS_BROADCAST = 0x11,
  BATCH_ASSIGN = 0x12,
  TRAIN_STEP = 0x13,
  STEP_COMPLETE = 0x14,
  CONFIG_UPDATE = 0x15,
  KICK = 0x16,

  // Dashboard
  STATUS_UPDATE = 0x20,
  LOSS_UPDATE = 0x21,
}

export interface WorkerJoinMessage {
  type: MessageType.WORKER_JOIN;
  workerId: string;
  deviceInfo: {
    vendor: string;
    architecture: string;
    description: string;
    maxBufferSize: number;
  };
  timestamp: number;
}

export interface WorkerHeartbeat {
  type: MessageType.WORKER_HEARTBEAT;
  workerId: string;
  timestamp: number;
  memoryUsage: number;
}

export interface GradientPushMessage {
  type: MessageType.GRADIENT_PUSH;
  workerId: string;
  stepId: number;
  gradients: Float32Array;  // flat parameter gradients
  loss: number;
  tokensProcessed: number;
  computeTimeMs: number;
}

export interface WelcomeMessage {
  type: MessageType.WELCOME;
  workerId: string;
  config: {
    sequenceLen: number;
    vocabSize: number;
    nLayer: number;
    nHead: number;
    nKvHead: number;
    nEmbd: number;
    windowPattern: string;
  };
  batchSize: number;
  totalWorkers: number;
}

export interface ParamsBroadcast {
  type: MessageType.PARAMS_BROADCAST;
  stepId: number;
  params: Float32Array;  // flat model parameters
}

export interface BatchAssignment {
  type: MessageType.BATCH_ASSIGN;
  stepId: number;
  inputIds: Uint32Array;  // [B, T] token IDs
  targets: Int32Array;     // [B, T] target IDs
}

export interface TrainStepCommand {
  type: MessageType.TRAIN_STEP;
  stepId: number;
  learningRate: number;
}

export interface StepCompleteMessage {
  type: MessageType.STEP_COMPLETE;
  stepId: number;
  avgLoss: number;
  numWorkers: number;
  tokensPerSec: number;
}

export interface StatusUpdate {
  type: MessageType.STATUS_UPDATE;
  step: number;
  loss: number;
  numWorkers: number;
  tokensPerSec: number;
  elapsedSeconds: number;
  workers: WorkerStatus[];
}

export interface WorkerStatus {
  id: string;
  state: 'idle' | 'training' | 'syncing' | 'disconnected';
  device: string;
  lastSeen: number;
  stepsCompleted: number;
  avgComputeTimeMs: number;
}

// Binary protocol helpers for efficient WebSocket communication
export function encodeMessage(msg: any): ArrayBuffer {
  const json = JSON.stringify(msg, (_, v) => {
    if (v instanceof Float32Array || v instanceof Uint32Array || v instanceof Int32Array) {
      return { __typed_array: true, type: v.constructor.name, data: Array.from(v) };
    }
    return v;
  });
  return new TextEncoder().encode(json).buffer;
}

export function decodeMessage(buf: ArrayBuffer): any {
  const text = new TextDecoder().decode(buf);
  return JSON.parse(text, (_, v) => {
    if (v && v.__typed_array) {
      const ArrayType = v.type === 'Float32Array' ? Float32Array :
                        v.type === 'Uint32Array' ? Uint32Array : Int32Array;
      return new ArrayType(v.data);
    }
    return v;
  });
}

// More efficient binary encoding for large tensors
export function encodeTensorMessage(type: MessageType, stepId: number, data: Float32Array): ArrayBuffer {
  // Header: type(u8) + stepId(u32) + dataLen(u32) = 9 bytes
  const headerSize = 9;
  const buf = new ArrayBuffer(headerSize + data.byteLength);
  const view = new DataView(buf);

  view.setUint8(0, type);
  view.setUint32(1, stepId, true);
  view.setUint32(5, data.length, true);

  new Float32Array(buf, headerSize + (headerSize % 4 === 0 ? 0 : 4 - (headerSize % 4))).set(data);

  return buf;
}

export function decodeTensorMessage(buf: ArrayBuffer): { type: MessageType; stepId: number; data: Float32Array } {
  const view = new DataView(buf);
  const type = view.getUint8(0) as MessageType;
  const stepId = view.getUint32(1, true);
  const dataLen = view.getUint32(5, true);

  const headerSize = 9;
  const dataOffset = headerSize + (headerSize % 4 === 0 ? 0 : 4 - (headerSize % 4));
  const data = new Float32Array(buf, dataOffset, dataLen);

  return { type, stepId, data };
}
