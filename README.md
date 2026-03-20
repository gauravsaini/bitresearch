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
| `prepare.py` (data + tokenizer) | `public/data/tokens.bin` (pre-tokenized, shipped) | Data prep is offline. Tokens are already exported — no Python needed at runtime. |
| `train.py` (training loop) | `src/distributed/trainer.ts` | Training loop: forward/backward, optimizer, all-reduce, loss scaling, sparsification. 100% browser-based. |
| `train.py` (GPT model) | `src/model/gpt.ts` | TFJS transformer: RoPE, RMSNorm, sliding window attention, residual ladder, softcap logits. |
| `make_dataloader()` | `src/data/dataloader.ts` | `TokenDataLoader` fetches `tokens.bin` and streams batches via `nextBatch(B, T)`. |
| `evaluate_bpb()` | `src/model/validate.ts` | Validation. |
| Hyperparams (top of `train.py`) | `src/model/config.ts` + trainer config in `p2p.html` | Model size, LR, batch size, time budget. |
| `run.log` / `grep "^val_loss:"` | Console output + `printSummary()` on stop | Structured summary for `grep`-based result extraction. |
| `results.tsv` | Manual (or pipe console output) | Log experiment results. |

### Offline scripts (NOT part of runtime training)
| Script | Purpose |
|---|---|
| `scripts/export_tokens.py` | One-time data prep: tokenizes text → `public/data/tokens.bin`. Already done, no need to re-run. |
| `scripts/export_reference.py` | Exports PyTorch weights for gradient parity testing against TFJS. |
| `scripts/train_reference.py` | PyTorch reference training loop for gradient validation. |

**None of the Python scripts run during training.** The entire training pipeline — data loading, model forward/backward, optimization, gradient sync — runs in the browser via TFJS + WebRTC.

## Training Goal

The default goal is: **minimize `val_loss` (cross-entropy) within a fixed time budget.**

This matches Karpathy's autoresearch protocol. Configure via:

- **Time budget**: `--timeout 300` (5 min default in headless mode) or `--max-steps N`
- **Model size**: Edit `src/model/config.ts` (`nLayer`, `nHead`, `nEmbd`, `vocabSize`)
- **Hyperparams**: Edit trainer config in `p2p.html` or pass via URL params
- **Data**: Tokens are pre-exported at `public/data/tokens.bin`. To re-export, run `python scripts/export_tokens.py` (offline, one-time).

For custom goals (e.g., reach a target loss, maximize throughput), edit the `onMetricsUpdate` callback in `p2p.html` or the `printSummary()` function.
