import { hasValueEmbedding, type GPTConfig } from './config';

export interface TensorSpec {
  name: string;
  shape: readonly number[];
}

export interface TensorMeta {
  shape: number[];
  offset: number;
  byteLength: number;
}

export interface CoverageIssue {
  name: string;
  expectedShape: readonly number[];
  actualShape?: readonly number[];
  expectedByteLength: number;
  actualByteLength?: number;
  reason: 'missing' | 'shape-mismatch' | 'byte-length-mismatch' | 'out-of-bounds';
}

export interface CoverageSummary {
  expectedCount: number;
  matchedCount: number;
  missing: CoverageIssue[];
  mismatches: CoverageIssue[];
  extra: string[];
  complete: boolean;
}

export const VALIDATION_CONFIG = {
  vocabSize: 8192,
  nLayer: 4,
  nHead: 4,
  nKvHead: 4,
  nEmbd: 256,
  sequenceLen: 256,
  windowPattern: 'SSSL',
} satisfies GPTConfig;

export function expectedByteLength(shape: readonly number[]): number {
  return shape.reduce((total, dim) => total * dim, 1) * 4;
}

export function buildValidationSpecs(config: GPTConfig = VALIDATION_CONFIG): TensorSpec[] {
  const headDim = config.nEmbd / config.nHead;
  const kvDim = config.nKvHead * headDim;

  const specs: TensorSpec[] = [
    { name: 'wte', shape: [config.vocabSize, config.nEmbd] },
    { name: 'lmHead', shape: [config.vocabSize, config.nEmbd] },
    { name: 'residLambdas', shape: [config.nLayer] },
    { name: 'x0Lambdas', shape: [config.nLayer] },
  ];

  for (let i = 0; i < config.nLayer; i++) {
    specs.push(
      { name: `layer${i}_q`, shape: [config.nEmbd, config.nEmbd] },
      { name: `layer${i}_k`, shape: [config.nEmbd, kvDim] },
      { name: `layer${i}_v`, shape: [config.nEmbd, kvDim] },
      { name: `layer${i}_proj`, shape: [config.nEmbd, config.nEmbd] },
      { name: `layer${i}_fc`, shape: [config.nEmbd, 4 * config.nEmbd] },
      { name: `layer${i}_mlp_proj`, shape: [4 * config.nEmbd, config.nEmbd] },
    );

    if (hasValueEmbedding(i, config.nLayer)) {
      specs.push(
        { name: `layer${i}_value_embed`, shape: [config.vocabSize, kvDim] },
        { name: `layer${i}_ve_gate`, shape: [32, config.nKvHead] },
      );
    }
  }

  return specs;
}

export function summarizeReferenceCoverage(
  specs: readonly TensorSpec[],
  meta: Record<string, TensorMeta>,
  binByteLength?: number,
): CoverageSummary {
  const missing: CoverageIssue[] = [];
  const mismatches: CoverageIssue[] = [];
  const expectedNames = new Set<string>();
  let matchedCount = 0;

  for (const spec of specs) {
    expectedNames.add(spec.name);
    const entry = meta[spec.name];
    if (!entry) {
      missing.push({
        name: spec.name,
        expectedShape: spec.shape,
        expectedByteLength: expectedByteLength(spec.shape),
        reason: 'missing',
      });
      continue;
    }

    const expectedBytes = expectedByteLength(spec.shape);
    const actualBytes = entry.byteLength;
    const shapeMatches = shapesEqual(entry.shape, spec.shape);
    const byteMatches = actualBytes === expectedBytes;
    const inBounds = binByteLength === undefined || (entry.offset >= 0 && entry.offset + entry.byteLength <= binByteLength);

    if (shapeMatches && byteMatches && inBounds) {
      matchedCount += 1;
      continue;
    }

    mismatches.push({
      name: spec.name,
      expectedShape: spec.shape,
      actualShape: entry.shape,
      expectedByteLength: expectedBytes,
      actualByteLength: actualBytes,
      reason: !shapeMatches ? 'shape-mismatch' : !byteMatches ? 'byte-length-mismatch' : 'out-of-bounds',
    });
  }

  const extra = Object.keys(meta).filter(name => !expectedNames.has(name));

  return {
    expectedCount: specs.length,
    matchedCount,
    missing,
    mismatches,
    extra,
    complete: missing.length === 0 && mismatches.length === 0,
  };
}

export function formatCoverageSummary(label: string, summary: CoverageSummary): string[] {
  const lines = [`${label}: ${summary.matchedCount}/${summary.expectedCount} tensors matched`];

  if (summary.missing.length > 0) {
    lines.push(
      `${label}: missing ${summary.missing.length} tensor(s): ${summary.missing.map(issue => issue.name).join(', ')}`,
    );
  }

  if (summary.mismatches.length > 0) {
    for (const issue of summary.mismatches) {
      const actualShape = issue.actualShape ? `[${issue.actualShape.join(', ')}]` : 'unknown';
      lines.push(
        `${label}: ${issue.name} ${issue.reason} expected [${issue.expectedShape.join(', ')}] (${issue.expectedByteLength} bytes) but found ${actualShape}${issue.actualByteLength !== undefined ? ` (${issue.actualByteLength} bytes)` : ''}`,
      );
    }
  }

  if (summary.extra.length > 0) {
    lines.push(`${label}: extra reference tensor(s): ${summary.extra.join(', ')}`);
  }

  return lines;
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
