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

### Starting the Training Swarm

```bash
# Install dependencies
pnpm install

# Start the signaling server + Vite dev server
pnpm run dev:p2p
```

Then open multiple tabs to `http://localhost:5173/p2p.html`. The tabs will automatically discover each other via the signaling server, negotiate WebRTC connections, form a ring topology, and begin distributed training.

## Tech Stack
- **GPU Compute**: TensorFlow.js (`@tensorflow/tfjs-backend-webgpu`)
- **Frontend**: TypeScript, Vite, Vanilla CSS
- **P2P Networking**: WebRTC (DataChannels)
- **Signaling Server**: Node.js, `ws`
