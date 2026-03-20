import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import * as readline from 'readline';
import { Tokenizer } from 'rustbpe-wasm';

// --- Constants ---
const CACHE_DIR = path.join(os.homedir(), '.cache', 'autoresearch');
const TEXT_DIR = path.join(CACHE_DIR, 'text');
const BIN_DIR = process.env.PREPARE_WASM_OUTPUT_DIR ?? path.join(CACHE_DIR, 'bin');
const VOCAB_SIZE = 8192;
const SPECIAL_TOKENS = ['<|reserved_0|>', '<|reserved_1|>', '<|reserved_2|>', '<|reserved_3|>'];
const SPLIT_PATTERN = `'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,2}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+`;
const TRAINING_SHARD_LIMIT = Number(process.env.PREPARE_WASM_TRAIN_SHARDS ?? '1');
const TOKENIZE_BATCH_SIZE = Number(process.env.PREPARE_WASM_TOKENIZE_BATCH_SIZE ?? '2048');
const TOKENIZE_SHARD_LIMIT = Number(process.env.PREPARE_WASM_TOKENIZE_SHARDS ?? '0');

function encodeBatch(tokenizer: Tokenizer, docs: string[]): Uint32Array[] {
    try {
        return tokenizer.batchEncode(docs);
    } catch (error) {
        if (error instanceof RangeError) {
            console.warn(`batchEncode overflowed on ${docs.length} docs; falling back to sequential encode() for this batch.`);
            return docs.map(doc => tokenizer.encode(doc));
        }
        throw error;
    }
}

async function readNonEmptyLines(filePath: string): Promise<string[]> {
    const docs: string[] = [];
    const reader = readline.createInterface({
        input: fs.createReadStream(filePath, { encoding: 'utf-8' }),
        crlfDelay: Infinity,
    });

    for await (const line of reader) {
        if (line.length > 0) {
            docs.push(line);
        }
    }

    return docs;
}

async function tokenizeShard(
    tokenizer: Tokenizer,
    inputPath: string,
    outputPath: string,
    bosId: number,
): Promise<{ docs: number; tokens: number }> {
    const tempPath = `${outputPath}.tmp`;
    const writer = await fsp.open(tempPath, 'w');
    const reader = readline.createInterface({
        input: fs.createReadStream(inputPath, { encoding: 'utf-8' }),
        crlfDelay: Infinity,
    });

    const docs: string[] = [];
    let totalDocs = 0;
    let totalTokens = 0;
    let flushedBatches = 0;

    const flushBatch = async () => {
        if (docs.length === 0) {
            return;
        }

        const encodedDocs = encodeBatch(tokenizer, docs);
        let chunkLen = 0;
        for (const ids of encodedDocs) {
            chunkLen += 1 + ids.length;
        }

        const packed = new Int32Array(chunkLen);
        let offset = 0;
        for (const ids of encodedDocs) {
            packed[offset++] = bosId;
            packed.set(ids, offset);
            offset += ids.length;
        }

        await writer.write(Buffer.from(packed.buffer, packed.byteOffset, packed.byteLength));
        totalDocs += docs.length;
        totalTokens += chunkLen;
        flushedBatches += 1;

        if (flushedBatches % 50 === 0) {
            console.log(`  processed ${totalDocs.toLocaleString()} docs...`);
        }

        docs.length = 0;
    };

    try {
        for await (const line of reader) {
            if (line.length === 0) {
                continue;
            }

            docs.push(line);
            if (docs.length >= TOKENIZE_BATCH_SIZE) {
                await flushBatch();
            }
        }

        await flushBatch();
        await writer.close();
        await fsp.rename(tempPath, outputPath);
        return { docs: totalDocs, tokens: totalTokens };
    } catch (error) {
        await writer.close().catch(() => undefined);
        await fsp.rm(tempPath, { force: true }).catch(() => undefined);
        throw error;
    }
}

async function main() {
    console.log('--- rustbpe-wasm paritiser ---');

    if (!(await fsp.stat(TEXT_DIR).catch(() => null))) {
        console.error(`Text directory ${TEXT_DIR} not found. Add extracted shard_*.txt files there first.`);
        process.exit(1);
    }

    const files = (await fsp.readdir(TEXT_DIR)).filter(f => f.endsWith('.txt')).sort();
    const trainFiles = files.filter(f => !f.includes('06542'));
    const valFiles = files.filter(f => f.includes('06542'));

    console.log(`Found ${trainFiles.length} training shards and ${valFiles.length} validation shards.`);

    const tokenizer = new Tokenizer();
    const vocabSizeNoSpecial = VOCAB_SIZE - SPECIAL_TOKENS.length;

    // Train on the first shard by default to match the current lightweight parity flow.
    console.log('Loading text for training...');
    const trainingTexts: string[] = [];
    for (const f of trainFiles.slice(0, TRAINING_SHARD_LIMIT)) {
        const filePath = path.join(TEXT_DIR, f);
        const lines = await readNonEmptyLines(filePath);
        console.log(`Adding ${lines.length} lines from ${f}...`);
        for (const line of lines) {
            trainingTexts.push(line);
        }
    }

    console.log(`Training on ${trainingTexts.length} document snippets...`);
    const t0 = Date.now();
    tokenizer.train(trainingTexts, vocabSizeNoSpecial, SPLIT_PATTERN);
    const t1 = Date.now();
    console.log(`Tokenizer trained in ${((t1 - t0) / 1000).toFixed(1)}s (vocab=${tokenizer.vocabSize()})`);

    // --- Save parity check ---
    console.log('Verifying parity...');
    const testText = "Hello world! Numbers: 123. Unicode: 你好";
    const ids = tokenizer.encode(testText);
    const decoded = tokenizer.decode(ids);
    console.log(`Input:   ${testText}`);
    console.log(`Encoded: ${ids}`);
    console.log(`Decoded: ${decoded}`);

    // --- Produce .bin shards ---
    await fsp.mkdir(BIN_DIR, { recursive: true });

    const filesToTokenize = TOKENIZE_SHARD_LIMIT > 0 ? files.slice(0, TOKENIZE_SHARD_LIMIT) : files;
    const bosId = tokenizer.vocabSize();

    for (const f of filesToTokenize) {
        console.log(`Tokenizing ${f}...`);
        const inputPath = path.join(TEXT_DIR, f);
        const binName = f.replace('.txt', '.bin');
        const outputPath = path.join(BIN_DIR, binName);
        const { docs, tokens } = await tokenizeShard(tokenizer, inputPath, outputPath, bosId);
        console.log(`  wrote ${docs.toLocaleString()} docs / ${tokens.toLocaleString()} tokens -> ${binName}`);
    }

    console.log(`Binaries ready in ${BIN_DIR}`);
}

main().catch(console.error);
