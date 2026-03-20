import * as fs from 'fs/promises';
import * as path from 'path';
import {
  VALIDATION_CONFIG,
  buildValidationSpecs,
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

function summarizeForwardAssets(batch: BatchData, logitsBin: ArrayBuffer, loss: LossData) {
  const expectedLogitsBytes = batch.inputs.length * VALIDATION_CONFIG.sequenceLen * VALIDATION_CONFIG.vocabSize * 4;
  const issues: string[] = [];

  if (!hasExpectedBatchShape(batch, VALIDATION_CONFIG.sequenceLen)) {
    issues.push(
      `batch.json shape mismatch: inputs=${batch.inputs.length}x${batch.inputs[0]?.length ?? 0}, targets=${batch.targets.length}x${batch.targets[0]?.length ?? 0}`,
    );
  }

  if (!Number.isFinite(loss.loss)) {
    issues.push('loss.json does not contain a finite loss value');
  }

  if (logitsBin.byteLength !== expectedLogitsBytes) {
    issues.push(`logits.bin byte-length mismatch: expected ${expectedLogitsBytes}, found ${logitsBin.byteLength}`);
  }

  return issues;
}

async function main() {
  const referenceDir = path.resolve(process.cwd(), 'public/reference');
  const specs = buildValidationSpecs(VALIDATION_CONFIG);

  const [weightsMeta, weightsBin, gradsMeta, gradsBin, batch, loss, logitsBin] = await Promise.all([
    readJson<Record<string, TensorMeta>>(path.join(referenceDir, 'weights_meta.json')),
    readBin(path.join(referenceDir, 'weights.bin')),
    readJson<Record<string, TensorMeta>>(path.join(referenceDir, 'grads_meta.json')),
    readBin(path.join(referenceDir, 'grads.bin')),
    readJson<unknown>(path.join(referenceDir, 'batch.json')),
    readJson<LossData>(path.join(referenceDir, 'loss.json')),
    readBin(path.join(referenceDir, 'logits.bin')),
  ]);

  const weightsCoverage = summarizeReferenceCoverage(specs, weightsMeta, weightsBin.byteLength);
  const gradsCoverage = summarizeReferenceCoverage(specs, gradsMeta, gradsBin.byteLength);

  console.log(`Reference directory: ${referenceDir}`);
  for (const line of formatCoverageSummary('Weights', weightsCoverage)) {
    console.log(line);
  }
  for (const line of formatCoverageSummary('Gradients', gradsCoverage)) {
    console.log(line);
  }

  const batchIsValid = isBatchData(batch);
  const forwardIssues = batchIsValid ? summarizeForwardAssets(batch, logitsBin, loss) : ['batch.json is not a valid batch payload.'];

  if (batchIsValid) {
    if (forwardIssues.length === 0) {
      console.log('Forward assets: batch.json, logits.bin, and loss.json are internally consistent.');
    } else {
      console.log('Forward assets need refresh:');
      for (const issue of forwardIssues) {
        console.log(`- ${issue}`);
      }
    }
  } else {
    console.log('batch.json is not a valid batch payload.');
  }

  const complete = weightsCoverage.complete && gradsCoverage.complete && batchIsValid && forwardIssues.length === 0;
  if (!complete) {
    process.exitCode = 1;
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
