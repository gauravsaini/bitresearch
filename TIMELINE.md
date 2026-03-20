# Project Timeline

## UPDATED ON : 2026-03-20

### Feature — Dynamic Ring Reformation, TURN Support, Checkpoint Saving

1. **Dynamic Heartbeat (P5)**: Implemented periodic ping/pong in `webrtc-mesh.ts` with EMA-smoothed latency tracking. Dynamic all-reduce timeout computed as `maxPeerLatency * 3 + 5000ms`. Added `onPeerLatencyUpdate` callback for runtime monitoring.

2. **Abort-and-Recompute (P5)**: `ring-allreduce.ts` now returns `null` on abort instead of hanging. Added `abort(reason)` method triggered on send failure or chunk timeout mid-reduce-scatter. Trainer skips optimizer update and retries next step.

3. **Min Ring Size (P5)**: Added `minRingSize` config to `TrainerConfig`. Training loop pauses with 2s poll when connected peers < threshold. Prevents corrupted gradients from incomplete rings.

4. **TURN Support (P6)**: `server/signaling.ts` generates Coturn HMAC-SHA1 credentials via `crypto.createHmac`. Credentials embedded in `ring-topology` message. REST endpoint `/turn-credentials` for managed TURN (Cloudflare/Metered.ca). `webrtc-mesh.ts` accepts `TurnServerConfig[]` and injects into `RTCPeerConnection`.

5. **Room Routing (P6)**: Signaling server now supports room-based isolation. Peers register with `room` field. Rings form independently per room. Empty rooms auto-cleaned. Global `peers` map replaced with `Map<string, Map<string, RegisteredPeer>>`.

6. **Checkpoint Saving (P7)**: `trainer.ts` `saveCheckpoint()` serializes to `.safetensors` binary format: `[u64 header_size][JSON header][tensor_data]`. Extracts Adam optimizer moments (m, v) from TF.js internals. Timestamped filename: `checkpoint_step{N}_{ISO}.safetensors`. Auto-download via Blob URL.

**Files changed**: `src/distributed/webrtc-mesh.ts`, `src/distributed/ring-allreduce.ts`, `src/distributed/trainer.ts`, `server/signaling.ts`

### Feature — Checkpoint Loading / Resume

1. **saveCheckpoint metadata**: Added `__metadata__` entry to safetensors header containing `step`, `lossScale`, `consecutiveGoodGradients`, `smoothLoss`, `totalTokens`, and `modelConfig` for full training state capture.

2. **loadCheckpoint()**: Parses `.safetensors` binary format (header size u64 LE → JSON header → tensor data). Restores model weights via `Variable.assign()`. Restores Adam optimizer `m` (1st moment) and `v` (2nd moment) from stored tensors. Restores training metadata (step, loss scale, smooth loss, total tokens).

3. **promptLoadCheckpoint()**: File picker UI via hidden `<input type="file" accept=".safetensors">`. Enables browser-based resume without server involvement.

**Files changed**: `src/distributed/trainer.ts`

### Feature — Real Data Training, Checkpoint Resume, Smoke Test

1. **Token DataLoader (`src/data/dataloader.ts`)**: Browser-compatible `TokenDataLoader` class. Fetches binary Int32 token file, provides `nextBatch(B, T)` with automatic epoch wrapping. Flat row-major format matching TF.js tensor2d expectations.

2. **Token Export (`scripts/export_tokens.py`)**: Python script to export tokenized data as flat Int32 binary. Uses trained BPE tokenizer if available, falls back to hash-based tokenization. Outputs `public/data/tokens.bin` + `tokens_meta.json`.

3. **Trainer Data Integration**: `TrainerConfig.dataUrl` added. `generateBatch()` now uses `TokenDataLoader` when available, falls back to synthetic random data on load failure. Data loaded during `initialize()`.

4. **Checkpoint Resume**: `saveCheckpoint()` now includes `__metadata__` (step, lossScale, smoothLoss, totalTokens, modelConfig). `loadCheckpoint(file)` parses safetensors binary, restores weights via `Variable.assign()`, restores Adam optimizer m/v moments, restores all training metadata.

5. **Checkpoint Test (`src/tests/checkpoint-resume.test.ts`)**: Browser test verifying save→load roundtrip. Subclasses trainer to intercept download, compares all weights element-wise after reload. Minimal model config (vocab=64, layers=1, embed=32) for speed.

6. **Smoke Test (`scripts/smoke-test.mjs`)**: Node.js script verifying server boot + HTTP endpoints. Polling-based readiness detection. Tests `/status` (peers, rooms, ringVersion) and `/turn-credentials` (HMAC username, credential, urls, expiry). Playwright browser tests if available. Exit code 0/1.

**Files changed**: `src/distributed/trainer.ts`, `src/data/dataloader.ts`, `scripts/export_tokens.py`, `scripts/smoke-test.mjs`, `src/tests/checkpoint-resume.test.ts`, `checkpoint-test.html`, `vite.config.ts`

## UPDATED ON : 2026-03-21

### Feature — Karpathy Parity Push

1. **Gap assessment completed**: Compared current `bitresearch` against a fresh local clone of `karpathy/autoresearch` and identified the highest-value missing pieces: BOS-packed dataloading, exact BPB, sliding-window attention semantics, value embeddings / gated value residuals, GQA correctness, and richer tokenizer artifacts.

2. **Data pipeline parity**: Added BOS-aligned best-fit document packing in `src/data/bestFitPacking.ts` and extended `src/data/dataloader.ts` to support both legacy flat token streams and document-oriented inputs for Karpathy-style packing.

3. **Model parity**: `src/model/gpt.ts` and `src/model/config.ts` now support alternating value embeddings with `ve_gate`, per-layer windowed causal masking from `windowPattern`, and grouped-query semantics for `nKvHead < nHead`.

4. **Trainer parity**: `src/distributed/trainer.ts` now supports token-byte-aware BPB evaluation when a byte-length sidecar is available, Karpathy-style warmup/warmdown LR scheduling, and step-level throughput accounting.

5. **Prepare/tokenizer parity**: `scripts/prepare_wasm.ts` now trains on a parity-friendly corpus by default, persists browser-friendly tokenizer artifacts (`tokenizer.json`, `token_bytes.bin`) under `~/.cache/autoresearch/tokenizer`, and preserves the large-shard WASM stack-overflow fix.

6. **Coordination policy**: Main thread owns `PLAN.md` and `TIMELINE.md` updates plus final integration/testing to keep progress logging centralized and avoid collisions.

**Files changed**: `scripts/prepare_wasm.ts`, `src/data/bestFitPacking.ts`, `src/data/dataloader.ts`, `src/model/config.ts`, `src/model/gpt.ts`, `src/distributed/trainer.ts`, `PLAN.md`

### Feature — CLI Auto-Start, UI Optional, Headless Mode

1. **Auto-start URL params**: `p2p.html` now reads `autoStart`, `hideUI`, `maxSteps`, `maxSeconds` from URL query string. `autoStart=1` triggers training after 2s delay (signaling connect time). `hideUI=1` hides metrics grid, topology, chart — only shows log.

2. **CLI Launcher (`scripts/launch.ts`)**: Single-command training launch. Starts signaling server + Vite, waits for both ready, opens N browser tabs with auto-start. Supports `--peers N`, `--no-ui`, `--timeout S`, `--max-steps N`. Platform-aware browser opening (macOS/Linux/Windows).

3. **Headless mode**: `--no-ui` hides all dashboard elements, outputs logs to console for capture. Combined with `--timeout` for timed runs matching Karpathy's 5-minute experiment budget.

4. **Summary output**: Trainer prints structured summary on stop (val_loss, tokens/sec, steps, peers, all_reduce_ms) for easy `grep` extraction like Karpathy's `run.log` workflow.

5. **npm scripts**: `pnpm run train` (3 peers), `pnpm run train:5` (5 peers), `pnpm run train:no-ui` (headless), `pnpm run train:headless` (headless + 5min timeout).

**Files changed**: `p2p.html`, `scripts/launch.ts`, `package.json`, `README.md`
