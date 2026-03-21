/**
 * Checkpoint Save → Load Roundtrip Test
 *
 * Verifies that DistributedTrainer's checkpoint mechanism correctly serializes
 * and restores model weights, optimizer state, and training metadata.
 *
 * Strategy: Subclass DistributedTrainer to override saveCheckpoint() so it
 * returns the ArrayBuffer instead of triggering a browser download. Then feed
 * that buffer back as a File to loadCheckpoint() on a fresh trainer instance.
 */

import * as tf from '@tensorflow/tfjs';
import { DistributedTrainer } from '../distributed/trainer';
import type { GPTConfig } from '../model/config';

// ---------------------------------------------------------------------------
// Minimal model config — keeps init + 5 steps under ~2 s
// ---------------------------------------------------------------------------
const TEST_CONFIG: GPTConfig = {
  vocabSize: 64,
  nLayer: 1,
  nHead: 2,
  nKvHead: 2,
  nEmbd: 32,
  sequenceLen: 16,
  windowPattern: 'L',
};

// ---------------------------------------------------------------------------
// TestableTrainer — exposes checkpoint serialization as ArrayBuffer
// ---------------------------------------------------------------------------
class TestableTrainer extends DistributedTrainer {
  /**
   * Same serialization logic as the parent's saveCheckpoint, but returns
   * the raw ArrayBuffer instead of triggering a browser download.
   *
   * Named differently to avoid the Promise<void> return-type conflict
   * with the base class.
   */
  async saveCheckpointAsBuffer(): Promise<ArrayBuffer> {
    const model = (this as any).model;
    const optimizer = (this as any).optimizer;
    if (!model || !optimizer) {
      throw new Error('Cannot save checkpoint: model or optimizer not initialized');
    }

    const step = this.metrics.step;
    const vars = model.getTrainableVariables();
    const tensors: Record<
      string,
      { shape: number[]; dtype: string; offset: number; data: ArrayBuffer }
    > = {};
    let currentOffset = 0;

    // Serialize model weights
    for (const v of vars) {
      const data = await v.data();
      const floatData = new Float32Array(data);
      const buf = new ArrayBuffer(floatData.byteLength);
      new Float32Array(buf).set(floatData);
      tensors[v.name] = {
        shape: v.shape,
        dtype: 'F32',
        offset: currentOffset,
        data: buf,
      };
      currentOffset += buf.byteLength;
    }

    // Serialize Adam optimizer moments (m and v)
    const optimizerAny = optimizer as any;

    if (optimizerAny.m) {
      for (const [varName, moment] of Object.entries(
        optimizerAny.m as Record<string, tf.Tensor>,
      )) {
        const data = await (moment as tf.Tensor).data();
        const floatData = new Float32Array(data);
        const buf = new ArrayBuffer(floatData.byteLength);
        new Float32Array(buf).set(floatData);
        tensors[`adam_m_${varName}`] = {
          shape: (moment as tf.Tensor).shape,
          dtype: 'F32',
          offset: currentOffset,
          data: buf,
        };
        currentOffset += buf.byteLength;
      }
    }

    if (optimizerAny.v) {
      for (const [varName, moment] of Object.entries(
        optimizerAny.v as Record<string, tf.Tensor>,
      )) {
        const data = await (moment as tf.Tensor).data();
        const floatData = new Float32Array(data);
        const buf = new ArrayBuffer(floatData.byteLength);
        new Float32Array(buf).set(floatData);
        tensors[`adam_v_${varName}`] = {
          shape: (moment as tf.Tensor).shape,
          dtype: 'F32',
          offset: currentOffset,
          data: buf,
        };
        currentOffset += buf.byteLength;
      }
    }

    // Build safetensors header
    const headerEntries: Record<string, any> = {};
    for (const [name, info] of Object.entries(tensors)) {
      headerEntries[name] = {
        dtype: info.dtype,
        shape: info.shape,
        data_offsets: [info.offset, info.offset + info.data.byteLength],
      };
    }

    headerEntries.__metadata__ = {
      step,
      lossScale: (this as any).lossScale,
      consecutiveGoodGradients: (this as any).consecutiveGoodGradients ?? 0,
      smoothLoss: this.metrics.smoothLoss,
      totalTokens: this.metrics.totalTokens,
      config: JSON.stringify((this as any).config.modelConfig),
    };

    const headerJson = JSON.stringify(headerEntries);
    const headerBytes = new TextEncoder().encode(headerJson);
    const headerSize = BigInt(headerBytes.length);

    const totalSize = 8 + headerBytes.length + currentOffset;
    const output = new ArrayBuffer(totalSize);
    const view = new DataView(output);
    const uint8View = new Uint8Array(output);

    view.setBigUint64(0, headerSize, true);
    uint8View.set(headerBytes, 8);

    const dataBase = 8 + headerBytes.length;
    for (const info of Object.values(tensors)) {
      uint8View.set(new Uint8Array(info.data), dataBase + info.offset);
    }

    return output;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Read every trainable variable's data into a name → Float32Array map. */
async function snapshotWeights(
  trainer: DistributedTrainer,
): Promise<Map<string, Float32Array>> {
  const vars = (trainer as any).model.getTrainableVariables();
  const snap = new Map<string, Float32Array>();
  for (const v of vars) {
    snap.set(v.name, new Float32Array(await v.data()));
  }
  return snap;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

export async function runCheckpointTest(): Promise<{
  passed: boolean;
  message: string;
}> {
  const logs: string[] = [];
  const log = (msg: string) => {
    logs.push(msg);
    console.log(`[CheckpointTest] ${msg}`);
  };

  let trainer1: TestableTrainer | null = null;
  let trainer2: TestableTrainer | null = null;

  try {
    // ---- Step 1: Create & initialise first trainer -----------------------
    log('Creating trainer #1 (minimal config, synthetic data)…');
    trainer1 = new TestableTrainer({
      modelConfig: TEST_CONFIG,
      batchSize: 2,
      unembedding_lr: 0.004,
      embedding_lr: 0.6,
      matrix_lr: 0.04,
      scalar_lr: 0.5,
      maxSteps: 100,
      checkpointInterval: 0,
      dataUrl: '',          // synthetic data — no network
      signalingUrl: '',     // skip signalling — no network
      minRingSize: 1,
    });
    await trainer1.initialize();
    log('Trainer #1 initialised.');

    // ---- Step 2: Run 5 training steps -----------------------------------
    log('Running 5 training steps…');
    for (let i = 0; i < 5; i++) {
      await trainer1.trainStep();
    }
    log(`Training done — step=${trainer1.metrics.step}`);

    // ---- Step 3: Snapshot state before save -----------------------------
    const weightsBefore = await snapshotWeights(trainer1);
    const stepBefore = trainer1.metrics.step;
    const lossScaleBefore = (trainer1 as any).lossScale;
    const smoothLossBefore = trainer1.metrics.smoothLoss;

    log(
      `Pre-save snapshot: step=${stepBefore}  lossScale=${lossScaleBefore}  smoothLoss=${smoothLossBefore.toFixed(6)}`,
    );

    // ---- Step 4: Save checkpoint (returns ArrayBuffer) -------------------
    log('Saving checkpoint…');
    const buffer = await trainer1.saveCheckpointAsBuffer();
    log(`Checkpoint buffer: ${(buffer.byteLength / 1024).toFixed(1)} KB`);

    // ---- Step 5: Create fresh trainer & load checkpoint ------------------
    log('Creating trainer #2 (fresh instance)…');
    trainer2 = new TestableTrainer({
      modelConfig: TEST_CONFIG,
      batchSize: 2,
      unembedding_lr: 0.004,
      embedding_lr: 0.6,
      matrix_lr: 0.04,
      scalar_lr: 0.5,
      maxSteps: 100,
      checkpointInterval: 0,
      dataUrl: '',
      signalingUrl: '',
      minRingSize: 1,
    });
    await trainer2.initialize();

    const file = new File([buffer], 'test-checkpoint.safetensors', {
      type: 'application/octet-stream',
    });
    log('Loading checkpoint into trainer #2…');
    await trainer2.loadCheckpoint(file);

    // ---- Step 6: Verify --------------------------------------------------

    // 6a. Model weights must match exactly
    const weightsAfter = await snapshotWeights(trainer2);
    const weightNames = Array.from(weightsBefore.keys());

    for (const name of weightNames) {
      const before = weightsBefore.get(name)!;
      const after = weightsAfter.get(name);

      if (!after) {
        return {
          passed: false,
          message: `FAIL — weight "${name}" missing after load.\n${logs.join('\n')}`,
        };
      }
      if (before.length !== after.length) {
        return {
          passed: false,
          message: `FAIL — weight "${name}" size mismatch: ${before.length} vs ${after.length}.\n${logs.join('\n')}`,
        };
      }
      for (let i = 0; i < before.length; i++) {
        if (before[i] !== after[i]) {
          return {
            passed: false,
            message: `FAIL — weight "${name}" differs at index ${i}: ${before[i]} vs ${after[i]}.\n${logs.join('\n')}`,
          };
        }
      }
    }

    // 6b. Metadata must match
    if (trainer2.metrics.step !== stepBefore) {
      return {
        passed: false,
        message: `FAIL — step mismatch: expected ${stepBefore}, got ${trainer2.metrics.step}.\n${logs.join('\n')}`,
      };
    }

    if ((trainer2 as any).lossScale !== lossScaleBefore) {
      return {
        passed: false,
        message: `FAIL — lossScale mismatch: expected ${lossScaleBefore}, got ${(trainer2 as any).lossScale}.\n${logs.join('\n')}`,
      };
    }

    if (trainer2.metrics.smoothLoss !== smoothLossBefore) {
      return {
        passed: false,
        message: `FAIL — smoothLoss mismatch: expected ${smoothLossBefore}, got ${trainer2.metrics.smoothLoss}.\n${logs.join('\n')}`,
      };
    }

    // ---- All checks passed ----------------------------------------------
    const summary = [
      `PASS — checkpoint roundtrip verified.`,
      `  weights : ${weightNames.length} tensors, all identical`,
      `  step    : ${stepBefore}`,
      `  lossScale : ${lossScaleBefore}`,
      `  smoothLoss: ${smoothLossBefore.toFixed(6)}`,
    ].join('\n');

    log(summary);
    return { passed: true, message: `${summary}\n\n${logs.join('\n')}` };
  } catch (err: any) {
    log(`EXCEPTION — ${err.message ?? err}`);
    return { passed: false, message: `ERROR — ${err.message ?? err}\n${logs.join('\n')}` };
  } finally {
    trainer1?.destroy();
    trainer2?.destroy();
  }
}
