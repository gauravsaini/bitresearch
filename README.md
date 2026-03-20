# WebGPU Distributed Training

> Porting Karpathy's autoresearch GPT training to WebGPU and enabling distributed data-parallel training across a swarm of browser-based workers using WebRTC.

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
| `prepare.py` | `scripts/export_tokens.py` + `src/data/dataloader.ts` | Data prep: downloads data, trains BPE tokenizer, exports tokenized binary. Browser-side `TokenDataLoader` fetches and streams batches. |
| `train.py` | `src/distributed/trainer.ts` + `src/model/gpt.ts` | Training loop + GPT model. Distributed trainer handles forward/backward, optimizer, all-reduce, loss scaling, sparsification. Model is TFJS-based transformer with RoPE, RMSNorm, sliding window attention. |
| `make_dataloader()` | `src/data/dataloader.ts` → `TokenDataLoader.nextBatch()` | Batch generation. Python uses BOS-aligned best-fit packing; browser version is simpler sequential streaming. |
| `evaluate_bpb()` | `src/model/validate.ts` | Validation. Python uses bits-per-byte on pinned val shard; browser uses smooth loss on training stream. |
| Hyperparameters (top of `train.py`) | `src/model/config.ts` + trainer config in `p2p.html` | Model size, LR, batch size, time budget. Edit `config.ts` for model architecture, trainer config in `p2p.html` for hyperparams. |
| `run.log` / `grep "^val_loss:"` | Console output + `printSummary()` on stop | Structured summary: `val_loss`, `tokens_per_sec`, `num_steps`, `peers`, `all_reduce_ms`. Capture with browser console or headless mode. |
| `results.tsv` | Manual (or pipe console output) | Log experiment results. Headless mode prints machine-readable summary to stdout. |

### Additional scripts (gradient validation)
| Script | Purpose |
|---|---|
| `scripts/export_reference.py` | Exports PyTorch weights + forward pass outputs for verifying TFJS gradients match PyTorch exactly. |
| `scripts/train_reference.py` | Full PyTorch reference training loop (not used at runtime — just for gradient parity testing). |

## Training Goal

The default goal is: **minimize `val_loss` (cross-entropy) within a fixed time budget.**

This matches Karpathy's autoresearch protocol. Configure via:

- **Time budget**: `--timeout 300` (5 min default in headless mode) or `--max-steps N`
- **Model size**: Edit `src/model/config.ts` (`nLayer`, `nHead`, `nEmbd`, `vocabSize`)
- **Hyperparams**: Edit trainer config in `p2p.html` or pass via URL params
- **Data**: Run `python scripts/export_tokens.py` to prepare `public/data/tokens.bin`

For custom goals (e.g., reach a target loss, maximize throughput), edit the `onMetricsUpdate` callback in `p2p.html` or the `printSummary()` function.
