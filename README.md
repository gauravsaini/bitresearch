# bitresearch

> WebGPU port of Karpathy's autoresearch — autonomous GPT training research across a swarm of browser tabs via WebRTC.

## Karpathy's autoresearch vs bitresearch

| | Karpathy's autoresearch | bitresearch |
|---|---|---|
| **Runtime** | Python + PyTorch + CUDA | TypeScript + TF.js + WebGPU |
| **Hardware** | Single NVIDIA GPU | Any device with a browser (Chrome/Safari/Edge) |
| **Scaling** | Single GPU | Distributed across browser tabs via WebRTC ring all-reduce |
| **Run command** | `uv run train.py` | `pnpm run train:headless` |
| **Data** | `uv run prepare.py --dataset <dataset>` | `pnpm run prepare:data` |
| **Editable files** | `train.py` | `src/distributed/trainer.ts`, `src/model/gpt.ts` |
| **Validation** | `evaluate_bpb()` in `prepare.py` | `evaluateBpb()` in `trainer.ts` |
| **Metric** | `val_bpb` (bits per byte) | `val_bpb` (bits per byte) |
| **Autonomous loop** | `program.md` instructs the agent | Same `program.md` protocol |
| **Setup** | `uv run prepare.py --dataset <dataset> && git checkout -b autoresearch/<tag>` | `pnpm install && pnpm run prepare:data && git checkout -b bitresearch/<tag>` |
| **No Python** | — | ✅ Zero Python at runtime |

## Overview

This project implements a fully decentralized GPT training system that runs entirely in the browser using:
1. **TensorFlow.js (WebGPU)**: For highly-optimized, hardware-accelerated tensor operations, and free automatic differentiation (autograd).
2. **WebRTC**: For peer-to-peer, decentralized distributed training using a bandwidth-optimal ring all-reduce algorithm.

Instead of a single powerful GPU, this system allows you to train a model by opening multiple browser tabs across different machines.

## Architecture & Modes

The project supports two modes of operation:

### 1. P2P Distributed Mode (WebRTC)
Each browser tab is an autonomous training node. There is no central parameter server.
- Gradients are averaged via **WebRTC data channels** directly between browsers.
- Uses a **ring all-reduce** algorithm for bandwidth-optimal synchronization.
- A lightweight Node.js signaling server is used *only* for peer discovery.

### 2. Centralized Mode (WebSocket)
A traditional parameter server architecture.
- A Node.js coordinator manages workers, distributes batches, and aggregates gradients.
- Workers connect via WebSocket.
- Includes a premium real-time monitoring dashboard with loss curves and worker stats.

## Next Steps to Production

The core infrastructure and WebRTC ring topology are implemented. To make this production-ready, the following roadmap is planned:

1. **Port GPT Model to TensorFlow.js** (Critical Architectural Win)
   - Replace complex, manual WGSL compute shaders with robust TFJS ops for a free backward pass and optimized AdamW.
2. **Gradient compression** (Bandwidth Blocker)
   - Implement `f16` quantization and top-k sparsification to reduce WebRTC payload sizes.
3. **TURN server support & Cross-network training** (Networking Blocker)
   - Add STUN/TURN support to allow peers behind strict NATs to connect over the internet.
4. **Dynamic ring reformation** (Fault Tolerance)
   - Handle worker churn (tabs opening/closing) by dynamically rebuilding the WebRTC ring mid-training without crashing.
5. **Checkpoint saving** (Usability)
   - Export trained model weights as `.safetensors` for download directly from the browser.

*(See `PLAN.md` for detailed technical TODOs).*

## How to Run

### Requirements
- Node.js (for signaling/dev server)
- `pnpm`
- A browser with WebGPU enabled (Chrome 113+, Edge 113+, Safari 18+)

### Quick Start (CLI — Karpathy Style)

One command. No manual clicking. Auto-starts everything:

```bash
pnpm install
pnpm run train
```

This launches the signaling server, Vite dev server, opens 3 browser tabs, and auto-starts training in each. Like Karpathy's autoresearch — just run and go.

### Data Preparation Parity

Karpathy's data prep entrypoint is:

```bash
uv run prepare.py --dataset <dataset>
```

The bitresearch parity path is:

```bash
pnpm run prepare:data
```

This script uses `rustbpe-wasm` instead of the Python `rustbpe` binding, keeps the same split pattern and BOS packing behavior, and writes parity `.bin` shards to `PREPARE_WASM_OUTPUT_DIR` (default `~/.cache/autoresearch/bin/`) plus tokenizer sidecar artifacts to `PREPARE_WASM_TOKENIZER_DIR` (default `~/.cache/autoresearch/tokenizer/`).

After tokenization it also mirrors a browser-ready bundle into `public/data/`:

- `public/data/parity/manifest.json` plus shard files under `public/data/parity/bin/`
- `public/data/parity/tokenizer/tokenizer.json`
- `public/data/parity/tokenizer/token_bytes.bin`
- legacy aliases at `public/data/tokens.bin`, `public/data/token_bytes.bin`, and `public/data/tokenizer.json`

The default P2P runtime now prefers `/data/parity/manifest.json`, loads the shard stream listed there, and auto-discovers token-byte sidecars for exact BPB. If you need a custom location, override it with `artifactBaseUrl`, `manifestUrl`, `dataUrl`, `documentsUrl`, or `tokenBytesUrl` query params in `p2p.html`.

Before running it, place extracted `shard_*.txt` files under `~/.cache/autoresearch/text/`. If you only want a quick parity check on a subset, you can limit work with:

```bash
PREPARE_WASM_TOKENIZE_SHARDS=1 pnpm run prepare:data
```

**More options:**

```bash
# 5 peers instead of 3
pnpm run train:5

# Headless mode (no dashboard, minimal logging, auto-stop after 5min)
pnpm run train:headless

# Custom: 8 peers, stop after 600 seconds
tsx scripts/launch.ts --peers 8 --timeout 600

# Headless with custom step limit
tsx scripts/launch.ts --no-ui --max-steps 500
```

### Manual Mode (Dev)

```bash
# Start signaling + Vite
pnpm run dev:p2p

# Then manually open tabs to http://localhost:5173/p2p.html
# Or with auto-start: http://localhost:5173/p2p.html?autoStart=1
```

## Tech Stack
- **GPU Compute**: TensorFlow.js (`@tensorflow/tfjs-backend-webgpu`)
- **Frontend**: TypeScript, Vite, Vanilla CSS
- **P2P Networking**: WebRTC (DataChannels)
- **Signaling Server**: Node.js, `ws`

## Codebase Map (Karpathy's autoresearch → this project)

| Karpathy's autoresearch | This project | Purpose |
|---|---|---|
| `prepare.py` (data + tokenizer) | `scripts/prepare_wasm.ts` | Rust/WASM parity prep path for Karpathy-style shard tokenization, tokenizer sidecars, and browser bundle mirroring. |
| `train.py` (training loop) | `src/distributed/trainer.ts` | Forward/backward, optimizer, all-reduce, loss scaling. 100% browser-based. |
| `train.py` (GPT model) | `src/model/gpt.ts` | TFJS transformer: RoPE, RMSNorm, sliding window attention. |
| `make_dataloader()` | `src/data/dataloader.ts` | `TokenDataLoader` consumes manifest-backed shard streams or legacy `tokens.bin` inputs. |
| `evaluate_bpb()` | `trainer.evaluateBpb()` | Held-out val split, forward-only, convert nats → bpb with token-byte sidecar support when available. |
| Hyperparams (top of `train.py`) | `src/model/config.ts` + trainer config in `p2p.html` | Model size, LR, batch size, time budget. |
| `run.log` / `grep "^val_bpb:"` | Console output + `printSummary()` on stop | Structured summary for result extraction. |
| `results.tsv` | `results.tsv` (untracked) | Same 5-column TSV format. |

**The entire training pipeline — data loading, model forward/backward, optimization, gradient sync — runs in the browser via TFJS + WebRTC. No Python at runtime.**

## Training Goal

The default goal is: **minimize `val_bpb` (validation bits per byte) within a fixed time budget.**

This matches Karpathy's autoresearch protocol. Configure via:

- **Time budget**: `--timeout 300` (5 min default in headless mode) or `--max-steps N`
- **Model size**: Edit `src/model/config.ts` (`nLayer`, `nHead`, `nEmbd`, `vocabSize`)
- **Hyperparams**: Edit trainer config in `p2p.html` or pass via URL params
- **Data**: By default the browser runtime reads `/data/parity/manifest.json` and falls back to `/data/tokens.bin`. BPB uses a token-byte sidecar when one is available.

For custom goals (e.g., reach a target loss, maximize throughput), edit the `onMetricsUpdate` callback in `p2p.html` or the `printSummary()` function.
