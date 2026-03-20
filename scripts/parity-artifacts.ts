import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { fileURLToPath } from 'url';

export interface ParityArtifactSyncOptions {
  cacheDir: string;
  publicDataDir?: string;
  binDir?: string;
  tokenizerDir?: string;
}

export interface ParityArtifactSyncResult {
  publicDataDir: string;
  parityDir: string;
  binDir: string;
  tokenizerDir: string;
  binFiles: string[];
  tokenizerFiles: string[];
  manifestPath: string;
  defaultDataPath: string;
  defaultTokenBytesPath: string;
}

const DEFAULT_CACHE_DIR = path.join(os.homedir(), '.cache', 'autoresearch');
const DEFAULT_PUBLIC_DATA_DIR = path.resolve(process.cwd(), 'public', 'data');

async function ensureCleanDir(dir: string): Promise<void> {
  await fs.rm(dir, { recursive: true, force: true });
  await fs.mkdir(dir, { recursive: true });
}

async function copyRequiredFile(sourcePath: string, destinationPath: string): Promise<void> {
  await fs.copyFile(sourcePath, destinationPath);
}

async function readJsonFile<T>(filePath: string): Promise<T> {
  const raw = await fs.readFile(filePath, 'utf8');
  return JSON.parse(raw) as T;
}

function pickDefaultBinFile(binFiles: string[]): string {
  const preferred = binFiles.find(fileName => !fileName.includes('06542'));
  return preferred ?? binFiles[0];
}

export async function syncParityArtifacts(options: ParityArtifactSyncOptions): Promise<ParityArtifactSyncResult> {
  const cacheDir = options.cacheDir;
  const binDir = options.binDir ?? path.join(cacheDir, 'bin');
  const tokenizerDir = options.tokenizerDir ?? path.join(cacheDir, 'tokenizer');
  const publicDataDir = options.publicDataDir ?? DEFAULT_PUBLIC_DATA_DIR;
  const parityDir = path.join(publicDataDir, 'parity');
  const parityBinDir = path.join(parityDir, 'bin');
  const parityTokenizerDir = path.join(parityDir, 'tokenizer');

  const binEntries = (await fs.readdir(binDir, { withFileTypes: true })).filter(entry => entry.isFile() && entry.name.endsWith('.bin'));
  const binFiles = binEntries.map(entry => entry.name).sort();
  if (binFiles.length === 0) {
    throw new Error(`No .bin artifacts found in ${binDir}`);
  }

  const tokenizerFiles = ['token_bytes.bin', 'tokenizer.json'];
  for (const fileName of tokenizerFiles) {
    await fs.access(path.join(tokenizerDir, fileName));
  }
  const tokenizerMeta = await readJsonFile<Record<string, unknown>>(path.join(tokenizerDir, 'tokenizer.json'));

  await ensureCleanDir(parityBinDir);
  await ensureCleanDir(parityTokenizerDir);

  for (const fileName of binFiles) {
    await copyRequiredFile(path.join(binDir, fileName), path.join(parityBinDir, fileName));
  }

  for (const fileName of tokenizerFiles) {
    const sourcePath = path.join(tokenizerDir, fileName);
    await copyRequiredFile(sourcePath, path.join(parityBinDir, fileName));
    await copyRequiredFile(sourcePath, path.join(parityTokenizerDir, fileName));
  }

  const manifest = {
    cacheDir,
    publicDataDir,
    parityUrl: '/data/parity',
    binUrl: '/data/parity/bin',
    tokenizerUrl: '/data/parity/tokenizer',
    source: {
      binDir,
      tokenizerDir,
    },
    files: {
      bin: binFiles,
      tokenizer: tokenizerFiles,
    },
  };

  const manifestPath = path.join(parityDir, 'manifest.json');
  await fs.mkdir(parityDir, { recursive: true });
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

  // Keep the legacy root data entrypoints pointing at parity outputs so the
  // existing browser runtime defaults consume prepared data without extra config.
  const defaultBinFile = pickDefaultBinFile(binFiles);
  const defaultBinSource = path.join(binDir, defaultBinFile);
  const defaultDataPath = path.join(publicDataDir, 'tokens.bin');
  const defaultTokenBytesPath = path.join(publicDataDir, 'token_bytes.bin');
  const defaultTokenizerPath = path.join(publicDataDir, 'tokenizer.json');
  const defaultDataStat = await fs.stat(defaultBinSource);
  const defaultMetaPath = path.join(publicDataDir, 'tokens_meta.json');
  const defaultMeta = {
    source: defaultBinFile,
    numTokens: Math.floor(defaultDataStat.size / 4),
    vocabSize: typeof tokenizerMeta.fullVocabSize === 'number'
      ? tokenizerMeta.fullVocabSize
      : typeof tokenizerMeta.vocabSize === 'number'
        ? tokenizerMeta.vocabSize
        : null,
    parityManifest: '/data/parity/manifest.json',
  };

  await copyRequiredFile(defaultBinSource, defaultDataPath);
  await copyRequiredFile(path.join(tokenizerDir, 'token_bytes.bin'), defaultTokenBytesPath);
  await copyRequiredFile(path.join(tokenizerDir, 'tokenizer.json'), defaultTokenizerPath);
  await fs.writeFile(defaultMetaPath, `${JSON.stringify(defaultMeta, null, 2)}\n`);

  return {
    publicDataDir,
    parityDir,
    binDir,
    tokenizerDir,
    binFiles,
    tokenizerFiles,
    manifestPath,
    defaultDataPath,
    defaultTokenBytesPath,
  };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const valueFor = (flag: string, fallback: string): string => {
    const index = args.indexOf(flag);
    if (index >= 0 && index + 1 < args.length) {
      return args[index + 1];
    }
    return fallback;
  };

  const cacheDir = valueFor('--cache-dir', DEFAULT_CACHE_DIR);
  const publicDataDir = valueFor('--public-data-dir', DEFAULT_PUBLIC_DATA_DIR);
  const binDir = valueFor('--bin-dir', path.join(cacheDir, 'bin'));
  const tokenizerDir = valueFor('--tokenizer-dir', path.join(cacheDir, 'tokenizer'));

  const result = await syncParityArtifacts({
    cacheDir,
    publicDataDir,
    binDir,
    tokenizerDir,
  });

  console.log(`Synced ${result.binFiles.length} parity bin file(s) into ${result.parityDir}`);
  console.log(`Parity manifest: ${result.manifestPath}`);
  console.log(`Default data file: ${result.defaultDataPath}`);
  console.log(`Default token-bytes file: ${result.defaultTokenBytesPath}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
