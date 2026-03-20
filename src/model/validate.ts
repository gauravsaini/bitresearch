import * as tf from '@tensorflow/tfjs';
import { GPTModel } from './gpt';
import {
  VALIDATION_CONFIG,
  formatCoverageSummary,
  summarizeReferenceCoverage,
  type TensorMeta,
} from './reference';

type BatchData = {
  inputs: number[][];
  targets: number[][];
};

type LossData = {
  loss: number;
};

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${url}: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function fetchBin(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${url}: ${res.status} ${res.statusText}`);
  }
  return await res.arrayBuffer();
}

function isBatchData(value: unknown): value is BatchData {
  if (!value || typeof value !== 'object') {
    return false;
  }

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

function buildModelSpecs(model: GPTModel) {
  return Object.entries(model.getTrainableVariablesByName()).map(([name, variable]) => ({
    name,
    shape: variable.shape,
  }));
}

function readReferenceTensor(
  bin: ArrayBuffer,
  meta: TensorMeta,
  expectedShape: readonly number[],
  name: string,
  log: (msg: string) => void,
): tf.Tensor | null {
  const expectedBytes = expectedShape.reduce((total, dim) => total * dim, 1) * 4;
  const actualShape = meta.shape;
  const actualBytes = meta.byteLength;
  const outOfBounds = meta.offset < 0 || meta.offset + meta.byteLength > bin.byteLength;

  if (!shapesEqual(actualShape, expectedShape)) {
    log(
      `${name}: shape mismatch, expected [${expectedShape.join(', ')}] but reference has [${actualShape.join(', ')}]`,
    );
    return null;
  }

  if (actualBytes !== expectedBytes) {
    log(`${name}: byte-length mismatch, expected ${expectedBytes} but reference has ${actualBytes}`);
    return null;
  }

  if (actualBytes % 4 !== 0) {
    log(`${name}: reference byte-length ${actualBytes} is not aligned to float32 values`);
    return null;
  }

  if (outOfBounds) {
    log(`${name}: reference slice is out of bounds for the loaded binary asset`);
    return null;
  }

  const f32 = new Float32Array(bin, meta.offset, meta.byteLength / 4);
  return tf.tensor(f32, expectedShape as number[]);
}

function shapesEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) {
    return false;
  }

  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }

  return true;
}

function validateBatch(batch: BatchData, sequenceLen: number, vocabSize: number, log: (msg: string) => void): boolean {
  if (!hasExpectedBatchShape(batch, sequenceLen)) {
    log(
      `Reference batch shape mismatch: inputs=${batch.inputs.length}x${batch.inputs[0]?.length ?? 0}, targets=${batch.targets.length}x${batch.targets[0]?.length ?? 0}, expected Nx${sequenceLen}`,
    );
    return false;
  }

  if (!batch.inputs.flat().every(Number.isInteger) || !batch.targets.flat().every(Number.isInteger)) {
    log('Reference batch contains non-integer token ids.');
    return false;
  }

  if (
    batch.inputs.flat().some(id => id < 0 || id >= vocabSize) ||
    batch.targets.flat().some(id => id < 0 || id >= vocabSize)
  ) {
    log('Reference batch contains token ids outside the model vocabulary.');
    return false;
  }

  return true;
}

function setStatus(statusEl: HTMLElement, html: string) {
  statusEl.innerHTML = html;
}

async function runValidation() {
  const statusEl = document.getElementById('status')!;
  const logsEl = document.getElementById('logs')!;

  const log = (msg: string) => {
    console.log(msg);
    const row = document.createElement('div');
    row.textContent = msg;
    logsEl.appendChild(row);
  };

  const logHtml = (html: string) => {
    console.log(html);
    const row = document.createElement('div');
    row.innerHTML = html;
    logsEl.appendChild(row);
  };

  const config = VALIDATION_CONFIG;

  setStatus(statusEl, 'Loading reference metadata...');
  const [weightsMeta, weightsBin, gradsMeta, gradsBin] = await Promise.all([
    fetchJson<Record<string, TensorMeta>>('/reference/weights_meta.json'),
    fetchBin('/reference/weights.bin'),
    fetchJson<Record<string, TensorMeta>>('/reference/grads_meta.json'),
    fetchBin('/reference/grads.bin'),
  ]);

  statusEl.innerText = 'Initializing WebGPU Model...';
  const model = new GPTModel(config);
  await model.init();

  const modelSpecs = buildModelSpecs(model);
  const weightsCoverage = summarizeReferenceCoverage(modelSpecs, weightsMeta, weightsBin.byteLength);
  const gradsCoverage = summarizeReferenceCoverage(modelSpecs, gradsMeta, gradsBin.byteLength);

  for (const line of formatCoverageSummary('Weights', weightsCoverage)) {
    log(line);
  }
  for (const line of formatCoverageSummary('Gradients', gradsCoverage)) {
    log(line);
  }

  const hasWeights = weightsCoverage.complete;
  const hasGradients = gradsCoverage.complete;

  if (!hasWeights) {
    setStatus(
      statusEl,
      '<span style="color: #f59e0b;">⚠ Reference bundle is stale or incomplete. Validation skipped until weights/meta match the current model.</span>',
    );
    log('Parity was skipped because the reference weights do not match the current model variable set.');
    log('Run `pnpm exec tsx scripts/check_reference.ts` to see the exact refresh gap.');
    return;
  }

  statusEl.innerText = 'Loading PyTorch reference weights...';
  const modelVars = model.getTrainableVariablesByName();
  for (const [name, variable] of Object.entries(modelVars)) {
    const meta = weightsMeta[name];
    if (!meta) {
      log(`Missing weight metadata for ${name}`);
      continue;
    }

    const tensor = readReferenceTensor(weightsBin, meta, variable.shape, name, log);
    if (!tensor) {
      continue;
    }

    variable.assign(tensor);
    tensor.dispose();
  }

  statusEl.innerText = 'Checking forward parity...';
  const batch = await fetchJson<unknown>('/reference/batch.json');
  const refLossData = await fetchJson<LossData>('/reference/loss.json');
  const logitsBin = await fetchBin('/reference/logits.bin');

  if (!isBatchData(batch) || !validateBatch(batch, config.sequenceLen, config.vocabSize, log)) {
    setStatus(statusEl, '<span style="color: #f59e0b;">⚠ Reference batch is not usable for parity comparison.</span>');
    return;
  }
  if (!Number.isFinite(refLossData.loss)) {
    setStatus(statusEl, '<span style="color: #f59e0b;">⚠ Reference loss is invalid, so parity comparison was skipped.</span>');
    log('loss.json does not contain a finite scalar loss.');
    return;
  }

  const inputs = tf.tensor2d(batch.inputs, [batch.inputs.length, config.sequenceLen], 'int32');
  const targets = tf.tensor2d(batch.targets, [batch.targets.length, config.sequenceLen], 'int32');

  let forwardLoss = Number.NaN;
  let logitsDiff = Number.NaN;
  let forwardReady = true;
  tf.tidy(() => {
    const fwd = model.forward(inputs, targets, true);
    forwardLoss = fwd.loss.dataSync()[0];
    const tfLogits = fwd.logits!;
    const expectedLogitsBytes = tfLogits.size * 4;
    if (logitsBin.byteLength !== expectedLogitsBytes) {
      log(
        `Reference logits are stale: expected ${expectedLogitsBytes} bytes for shape [${tfLogits.shape.join(', ')}], found ${logitsBin.byteLength}`,
      );
      forwardReady = false;
      return;
    }

    const ptLogitsF32 = new Float32Array(logitsBin);
    const ptLogits = tf.tensor(ptLogitsF32, tfLogits.shape);
    logitsDiff = tfLogits.sub(ptLogits).abs().max().dataSync()[0];
    ptLogits.dispose();
  });

  if (!forwardReady || !Number.isFinite(forwardLoss) || !Number.isFinite(logitsDiff)) {
    setStatus(
      statusEl,
      '<span style="color: #f59e0b;">⚠ Forward reference assets are stale or incomplete, so parity comparison was skipped.</span>',
    );
    tf.dispose([inputs, targets]);
    return;
  }

  logHtml(`<strong>PyTorch Loss:</strong> ${refLossData.loss.toFixed(6)}`);
  logHtml(`<strong>TF.js Loss:   </strong> ${forwardLoss.toFixed(6)}`);
  logHtml(`<strong>Loss Delta:   </strong> ${Math.abs(forwardLoss - refLossData.loss).toFixed(6)}`);
  logHtml(`<strong>Logits Max Diff:</strong> ${logitsDiff.toFixed(6)}`);

  if (!hasGradients) {
    setStatus(
      statusEl,
      '<span style="color: #f59e0b;">⚠ Forward parity passed for the current reference bundle, but gradient metadata is stale or incomplete so backward parity was skipped.</span>',
    );
    log('Gradient parity was skipped because grads_meta.json does not match the current model variable set.');
    return;
  }

  statusEl.innerText = 'Running TFJS backward parity check...';
  const { value, grads } = tf.variableGrads(() => model.forward(inputs, targets).loss);
  await value.data();

  let passed = true;
  for (const [name, variable] of Object.entries(modelVars)) {
    const tfGrad = grads[variable.name];
    if (!tfGrad) {
      log(`Missing TF.js gradient for ${name}`);
      passed = false;
      continue;
    }

    const meta = gradsMeta[name];
    if (!meta) {
      log(`Missing gradient metadata for ${name}`);
      passed = false;
      tfGrad.dispose();
      continue;
    }

    const ptGrad = readReferenceTensor(gradsBin, meta, variable.shape, name, log);
    if (!ptGrad) {
      passed = false;
      tfGrad.dispose();
      continue;
    }

    const diff = tfGrad.sub(ptGrad).abs().max();
    const l2A = tf.norm(tfGrad);
    const l2B = tf.norm(ptGrad);

    const maxDiff = (await diff.data())[0];
    const nA = (await l2A.data())[0];
    const nB = (await l2B.data())[0];

    const color = maxDiff > 1e-4 ? 'red' : 'green';
    logHtml(
      `<span style="color: ${color}">${name} | maxDiff: ${maxDiff.toFixed(6)} | tf-l2: ${nA.toFixed(6)} | pt-l2: ${nB.toFixed(6)}</span>`,
    );

    if (maxDiff > 1e-4) {
      passed = false;
    }

    tfGrad.dispose();
    ptGrad.dispose();
  }

  setStatus(
    statusEl,
    passed
      ? '<span style="color: #4ade80;">✅ Validation passed for the current reference bundle.</span>'
      : '<span style="color: #f87171;">❌ Validation failed. One or more gradient comparisons exceeded the tolerance.</span>',
  );

  tf.dispose([inputs, targets, value]);
}

runValidation().catch(e => {
  console.error(e);
  document.getElementById('status')!.innerText = `Error: ${e.message}`;
});
