/**
 * results.tsv Auto-Logger — Karpathy-compatible experiment logging.
 *
 * Automatically appends experiment results to results.tsv after each training run.
 * Format: commit\tval_bpb\tmemory_gb\tstatus\tdescription
 *
 * This module runs in Node.js context only (server-side / CLI).
 * Browser trainer uses dynamic import() to conditionally load this.
 */

/* eslint-disable @typescript-eslint/no-var-requires */
declare const require: (id: string) => any;

export interface ExperimentResult {
  valBpb: number;
  peakMemoryMb: number;
  status: 'keep' | 'discard' | 'crash';
  description: string;
  commitHash?: string;
  trainingSeconds?: number;
  totalSeconds?: number;
  mfuPercent?: number;
  totalTokensM?: number;
  numSteps?: number;
  numParamsM?: number;
  depth?: number;
}

const TSV_HEADER = 'commit\tval_bpb\tmemory_gb\tstatus\tdescription';

function getGitCommitShort(): string {
  try {
    const cp = require('child_process');
    return cp.execSync('git rev-parse --short=7 HEAD', { encoding: 'utf-8' }).trim();
  } catch {
    return 'unknown';
  }
}

export class ResultsLogger {
  private filePath: string;
  private fs: any;
  private path: any;

  constructor(filePath = 'results.tsv') {
    this.fs = require('fs');
    this.path = require('path');
    this.filePath = this.path.resolve(filePath);
    this.ensureFile();
  }

  private ensureFile(): void {
    if (!this.fs.existsSync(this.filePath)) {
      this.fs.writeFileSync(this.filePath, TSV_HEADER + '\n', 'utf-8');
    } else {
      const content = this.fs.readFileSync(this.filePath, 'utf-8');
      if (!content.startsWith('commit\t')) {
        this.fs.writeFileSync(this.filePath, TSV_HEADER + '\n' + content, 'utf-8');
      }
    }
  }

  log(result: ExperimentResult): void {
    const commit = result.commitHash || getGitCommitShort();
    const valBpb = result.status === 'crash' ? '0.000000' : result.valBpb.toFixed(6);
    const memoryGb = result.status === 'crash' ? '0.0' : (result.peakMemoryMb / 1024).toFixed(1);
    const status = result.status;
    const description = result.description.replace(/[\t\n\r]/g, ' ').trim();
    const line = `${commit}\t${valBpb}\t${memoryGb}\t${status}\t${description}`;
    this.fs.appendFileSync(this.filePath, line + '\n', 'utf-8');
  }

  readAll(): ExperimentResult[] {
    if (!this.fs.existsSync(this.filePath)) return [];
    const lines = (this.fs.readFileSync(this.filePath, 'utf-8') as string).trim().split('\n');
    return lines.slice(1).map((line: string) => {
      const [commit, valBpb, memoryGb, status, ...descParts] = line.split('\t');
      return {
        commitHash: commit,
        valBpb: parseFloat(valBpb) || 0,
        peakMemoryMb: (parseFloat(memoryGb) || 0) * 1024,
        status: (status as 'keep' | 'discard' | 'crash') || 'crash',
        description: descParts.join('\t'),
      };
    });
  }

  getBestBpb(): number {
    const results = this.readAll().filter((r: ExperimentResult) => r.status === 'keep');
    if (results.length === 0) return Infinity;
    return Math.min(...results.map((r: ExperimentResult) => r.valBpb));
  }

  getCount(): number {
    return this.readAll().length;
  }
}
