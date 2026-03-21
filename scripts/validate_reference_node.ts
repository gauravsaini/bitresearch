import * as fs from 'fs/promises';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs';
import { GPTModel } from '../src/model/gpt';
import {
  VALIDATION_CONFIG,
  formatCoverageSummary,
  summarizeReferenceCoverage,
  type TensorMeta,
} from '../src/model/reference';

type BatchData = {
  inputs: number[][];
  targets: number[][];
};

type LossData = {
  loss: number;
};

async function readJson<T>(filePath: string): Promise<T> {
  const raw = await fs.readFile(filePath, 'utf8');
  return JSON.parse(raw) as T;
}

async function readBin(filePath: string): Promise<ArrayBuffer> {
  const raw = await fs.readFile(filePath);
  return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
}

function shapesEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function isBatchData(value: unknown): value is BatchData {
  if (!value || typeof value !== 'object') return false;
  const maybe = value as BatchData;
  return Array.isArray(maybe.inputs) && Array.isArray(maybe.targets);
}

function hasExpectedBatchShape(batch: BatchData, sequenceLen: number): boolean {
  return (
    batch.inputs.length > 0 &&
    batch.targets.length === batch.inputs.length &&
    batch.inputs.every(row => row.length === sequenceLen) &&
    batch.targets.every(row => row.length === sequenceLen)
  );
}

function readReferenceTensor(
  bin: ArrayBuffer,
  meta: TensorMeta,
  expectedShape: readonly number[],
  name: string,
): tf.Tensor {
  const expectedBytes = expectedShape.reduce((total, dim) => total * dim, 1) * 4;
  const outOfBounds = meta.offset < 0 || meta.offset + meta.byteLength > bin.byteLength;

  if (!shapesEqual(meta.shape, expectedShape)) {
    throw new Error(`${name}: shape mismatch expected [${expectedShape.join(', ')}] found [${meta.shape.join(', ')}]`);
  }
  if (meta.byteLength !== expectedBytes) {
    throw new Error(`${name}: byte-length mismatch expected ${expectedBytes} found ${meta.byteLength}`);
  }
  if (meta.byteLength % 4 !== 0) {
    throw new Error(`${name}: byte-length ${meta.byteLength} is not float32 aligned`);
  }
  if (outOfBounds) {
    throw new Error(`${name}: reference slice is out of bounds`);
  }

  const f32 = new Float32Array(bin, meta.offset, meta.byteLength / 4);
  return tf.tensor(f32, expectedShape as number[]);
}

function maxAbsDiff(a: tf.Tensor, b: tf.Tensor): number {
  return a.sub(b).abs().max().dataSync()[0];
}

async function main(): Promise<void> {
  const referenceDir = path.resolve(process.cwd(), 'public/reference');
  const [weightsMeta, weightsBin, gradsMeta, gradsBin, batch, lossData, logitsBin] = await Promise.all([
    readJson<Record<string, TensorMeta>>(path.join(referenceDir, 'weights_meta.json')),
    readBin(path.join(referenceDir, 'weights.bin')),
    readJson<Record<string, TensorMeta>>(path.join(referenceDir, 'grads_meta.json')),
    readBin(path.join(referenceDir, 'grads.bin')),
    readJson<unknown>(path.join(referenceDir, 'batch.json')),
    readJson<LossData>(path.join(referenceDir, 'loss.json')),
    readBin(path.join(referenceDir, 'logits.bin')),
  ]);

  const model = new GPTModel(VALIDATION_CONFIG);
  await model.init();
  console.log(`TFJS backend: ${tf.getBackend()}`);

  const modelVars = model.getTrainableVariablesByName();
  const modelSpecs = Object.entries(modelVars).map(([name, variable]) => ({ name, shape: variable.shape }));
  const weightsCoverage = summarizeReferenceCoverage(modelSpecs, weightsMeta, weightsBin.byteLength);
  const gradsCoverage = summarizeReferenceCoverage(modelSpecs, gradsMeta, gradsBin.byteLength);

  for (const line of formatCoverageSummary('Weights', weightsCoverage)) {
    console.log(line);
  }
  for (const line of formatCoverageSummary('Gradients', gradsCoverage)) {
    console.log(line);
  }

  if (!weightsCoverage.complete || !gradsCoverage.complete) {
    throw new Error('reference coverage is incomplete; run the exporter before validating');
  }
  if (!isBatchData(batch) || !hasExpectedBatchShape(batch, VALIDATION_CONFIG.sequenceLen)) {
    throw new Error('batch.json is not usable for validation');
  }
  if (!Number.isFinite(lossData.loss)) {
    throw new Error('loss.json does not contain a finite scalar loss');
  }

  for (const [name, variable] of Object.entries(modelVars)) {
    const meta = weightsMeta[name];
    const tensor = readReferenceTensor(weightsBin, meta, variable.shape, name);
    variable.assign(tensor);
    tensor.dispose();
  }

  const inputs = tf.tensor2d(batch.inputs, [batch.inputs.length, VALIDATION_CONFIG.sequenceLen], 'int32');
  const targets = tf.tensor2d(batch.targets, [batch.targets.length, VALIDATION_CONFIG.sequenceLen], 'int32');

  const forward = model.forward(inputs, targets, true);
  const forwardLoss = (await forward.loss.data())[0];
  const tfLogits = forward.logits!;
  const referenceLogits = tf.tensor(new Float32Array(logitsBin), tfLogits.shape);
  const logitsDiff = maxAbsDiff(tfLogits, referenceLogits);

  console.log(`PyTorch loss: ${lossData.loss.toFixed(8)}`);
  console.log(`TFJS loss:    ${forwardLoss.toFixed(8)}`);
  console.log(`Loss delta:   ${Math.abs(forwardLoss - lossData.loss).toExponential(6)}`);
  console.log(`Logits diff:  ${logitsDiff.toExponential(6)}`);

  if (Math.abs(forwardLoss - lossData.loss) > 1e-5) {
    throw new Error(`forward loss mismatch exceeds tolerance: ${Math.abs(forwardLoss - lossData.loss)}`);
  }
  if (logitsDiff > 1e-4) {
    throw new Error(`logits max diff exceeds tolerance: ${logitsDiff}`);
  }

  const { value, grads } = tf.variableGrads(() => model.forward(inputs, targets).loss);
  await value.data();

  let worstGrad = 0;
  let worstName = '';
  for (const [name, variable] of Object.entries(modelVars)) {
    const tfGrad = grads[variable.name];
    if (!tfGrad) {
      throw new Error(`missing TFJS gradient for ${name}`);
    }
    const refTensor = readReferenceTensor(gradsBin, gradsMeta[name], variable.shape, name);
    const diff = maxAbsDiff(tfGrad, refTensor);
    if (diff > worstGrad) {
      worstGrad = diff;
      worstName = name;
    }
    tfGrad.dispose();
    refTensor.dispose();
  }

  console.log(`Worst gradient diff: ${worstGrad.toExponential(6)}${worstName ? ` (${worstName})` : ''}`);
  if (worstGrad > 1e-4) {
    throw new Error(`gradient max diff exceeds tolerance: ${worstGrad} (${worstName})`);
  }

  tf.dispose([inputs, targets, forward.loss, tfLogits, referenceLogits, value]);
  console.log('Reference validation passed.');
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
