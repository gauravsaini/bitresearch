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
const TOKENIZER_DIR = process.env.PREPARE_WASM_TOKENIZER_DIR ?? path.join(CACHE_DIR, 'tokenizer');
const VOCAB_SIZE = 8192;
const SPECIAL_TOKENS = ['<|reserved_0|>', '<|reserved_1|>', '<|reserved_2|>', '<|reserved_3|>'];
const SPLIT_PATTERN = `'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,2}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+`;
const TRAINING_SHARD_LIMIT = parseEnvInt('PREPARE_WASM_TRAIN_SHARDS', 0, 0);
const TRAINING_MAX_CHARS = parseEnvInt('PREPARE_WASM_TRAIN_MAX_CHARS', 1_000_000_000, 1);
const TRAINING_DOC_CAP = parseEnvInt('PREPARE_WASM_TRAIN_DOC_CAP', 10_000, 1);
const TOKENIZE_BATCH_SIZE = parseEnvInt('PREPARE_WASM_TOKENIZE_BATCH_SIZE', 2048, 1);
const TOKENIZE_SHARD_LIMIT = parseEnvInt('PREPARE_WASM_TOKENIZE_SHARDS', 0, 0);

function parseEnvInt(name: string, defaultValue: number, minValue: number): number {
    const raw = process.env[name];
    if (raw === undefined || raw === '') {
        return defaultValue;
    }

    const value = Number.parseInt(raw, 10);
    if (!Number.isFinite(value) || Number.isNaN(value)) {
        throw new Error(`Invalid integer for ${name}: ${raw}`);
    }
    if (value < minValue) {
        throw new Error(`Invalid integer for ${name}: must be >= ${minValue}`);
    }

    return value;
}

function bytesToBase64(bytes: Uint8Array): string {
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
}

async function writeAtomicFile(filePath: string, data: string | Buffer | Uint8Array): Promise<void> {
    const tempPath = `${filePath}.tmp`;
    await fsp.writeFile(tempPath, data);
    await fsp.rename(tempPath, filePath);
}

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

async function collectTrainingCorpus(
    textDir: string,
    fileNames: string[],
    maxChars: number,
    docCap: number,
): Promise<{ docs: string[]; docCount: number; charCount: number; filesUsed: string[] }> {
    const docs: string[] = [];
    const filesUsed: string[] = [];
    let docCount = 0;
    let charCount = 0;

    for (const fileName of fileNames) {
        const filePath = path.join(textDir, fileName);
        const reader = readline.createInterface({
            input: fs.createReadStream(filePath, { encoding: 'utf-8' }),
            crlfDelay: Infinity,
        });

        let fileDocs = 0;
        let fileChars = 0;

        try {
            for await (const line of reader) {
                if (line.length === 0) {
                    continue;
                }

                const doc = line.length > docCap ? line.slice(0, docCap) : line;
                docs.push(doc);
                docCount += 1;
                fileDocs += 1;
                charCount += doc.length;
                fileChars += doc.length;

                if (charCount >= maxChars) {
                    filesUsed.push(fileName);
                    console.log(`  added ${fileDocs.toLocaleString()} docs / ${fileChars.toLocaleString()} chars from ${fileName}`);
                    console.log(`Reached training char cap at ${charCount.toLocaleString()} chars; stopping corpus scan.`);
                    return { docs, docCount, charCount, filesUsed };
                }
            }
        } finally {
            reader.close();
        }

        filesUsed.push(fileName);
        console.log(`  added ${fileDocs.toLocaleString()} docs / ${fileChars.toLocaleString()} chars from ${fileName}`);
    }

    return { docs, docCount, charCount, filesUsed };
}

function buildTokenizerArtifacts(tokenizer: Tokenizer): {
    metadata: Record<string, unknown>;
    tokenBytes: Int32Array;
} {
    const baseVocabSize = tokenizer.vocabSize();
    const fullVocabSize = baseVocabSize + SPECIAL_TOKENS.length;
    const tokenBytes = new Int32Array(fullVocabSize);
    const mergeableRanks = tokenizer.getMergeableRanks();

    for (const [bytes, rank] of mergeableRanks) {
        tokenBytes[rank] = bytes.byteLength;
    }

    const specialTokenIds: Record<string, number> = Object.fromEntries(
        SPECIAL_TOKENS.map((token, index) => [token, baseVocabSize + index]),
    ) as Record<string, number>;
    if (mergeableRanks.length !== baseVocabSize) {
        throw new Error(`Unexpected merge table size: ${mergeableRanks.length} != ${baseVocabSize}`);
    }

    const metadata = {
        artifactVersion: 1,
        name: 'rustbpe-wasm',
        pattern: tokenizer.getPattern(),
        vocabSize: baseVocabSize,
        fullVocabSize,
        numMerges: tokenizer.numMerges(),
        specialTokens: SPECIAL_TOKENS,
        specialTokenIds,
        bosToken: SPECIAL_TOKENS[0],
        bosTokenId: baseVocabSize,
        tokenBytesFile: 'token_bytes.bin',
        mergeableRanksEncoding: 'base64',
        mergeableRanks: mergeableRanks.map(([bytes, rank]) => [bytesToBase64(bytes), rank] as const),
    };

    for (const tokenId of Object.values(specialTokenIds)) {
        tokenBytes[tokenId] = 0;
    }

    return { metadata, tokenBytes };
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
    console.log(`Training corpus config: shards=${TRAINING_SHARD_LIMIT > 0 ? TRAINING_SHARD_LIMIT : 'all'}, maxChars=${TRAINING_MAX_CHARS.toLocaleString()}, docCap=${TRAINING_DOC_CAP.toLocaleString()}`);

    const tokenizer = new Tokenizer();
    const vocabSizeNoSpecial = VOCAB_SIZE - SPECIAL_TOKENS.length;

    console.log('Loading text for training...');
    const trainFilesToUse = TRAINING_SHARD_LIMIT > 0 ? trainFiles.slice(0, TRAINING_SHARD_LIMIT) : trainFiles;
    const { docs: trainingTexts, docCount, charCount, filesUsed } = await collectTrainingCorpus(
        TEXT_DIR,
        trainFilesToUse,
        TRAINING_MAX_CHARS,
        TRAINING_DOC_CAP,
    );

    if (trainingTexts.length === 0) {
        throw new Error('No training documents were collected. Check PREPARE_WASM_TRAIN_SHARDS / TEXT_DIR.');
    }

    console.log(`Training on ${docCount.toLocaleString()} document snippets from ${filesUsed.length.toLocaleString()} files (${charCount.toLocaleString()} chars)...`);
    const t0 = Date.now();
    tokenizer.train(trainingTexts, vocabSizeNoSpecial, SPLIT_PATTERN);
    const t1 = Date.now();
    console.log(`Tokenizer trained in ${((t1 - t0) / 1000).toFixed(1)}s (vocab=${tokenizer.vocabSize()}, merges=${tokenizer.numMerges()})`);

    // --- Save parity check ---
    console.log('Verifying parity...');
    const testText = "Hello world! Numbers: 123. Unicode: 你好";
    const ids = tokenizer.encode(testText);
    const decoded = tokenizer.decode(ids);
    console.log(`Input:   ${testText}`);
    console.log(`Encoded: ${ids}`);
    console.log(`Decoded: ${decoded}`);

    // --- Persist tokenizer artifacts for downstream parity ---
    await fsp.mkdir(TOKENIZER_DIR, { recursive: true });
    const { metadata, tokenBytes } = buildTokenizerArtifacts(tokenizer);
    const tokenizerJson = `${JSON.stringify(metadata, null, 2)}\n`;
    const tokenBytesPath = path.join(TOKENIZER_DIR, 'token_bytes.bin');
    const tokenizerPath = path.join(TOKENIZER_DIR, 'tokenizer.json');
    await writeAtomicFile(tokenizerPath, tokenizerJson);
    await writeAtomicFile(tokenBytesPath, Buffer.from(tokenBytes.buffer, tokenBytes.byteOffset, tokenBytes.byteLength));
    console.log(`Tokenizer bundle saved to ${TOKENIZER_DIR}`);

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
    console.log(`Tokenizer artifacts ready in ${TOKENIZER_DIR}`);
}

main().catch(console.error);
