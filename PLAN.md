# WebGPU Distributed Training — Implementation Plan

> Port Karpathy's autoresearch GPT training to WebGPU and enable distributed data-parallel training across a swarm of browser-based workers.

## Architecture Overview

```mermaid
graph TB
    subgraph Coordinator["Coordinator Server (Node.js)"]
        PS[Parameter Server]
        GA[Gradient Aggregator]
        WM[Worker Manager]
        DL[Data Distributor]
        DASH[Dashboard API]
    end

    subgraph Worker1["Browser Worker 1"]
        W1GPU[WebGPU Device]
        W1Model[GPT Model]
        W1Train[Forward + Backward]
    end

    subgraph Worker2["Browser Worker 2"]
        W2GPU[WebGPU Device]
        W2Model[GPT Model]
        W2Train[Forward + Backward]
    end

    subgraph WorkerN["Browser Worker N"]
        WNGPU[WebGPU Device]
        WNModel[GPT Model]
        WNTrain[Forward + Backward]
    end

    subgraph Dashboard["Dashboard UI"]
        LC[Loss Curves]
        WS_STATUS[Worker Status]
        METRICS[Training Metrics]
    end

    PS <-->|WebSocket| Worker1
    PS <-->|WebSocket| Worker2
    PS <-->|WebSocket| WorkerN
    DASH --> Dashboard
```

## Data-Parallel Training Protocol

```
1. Coordinator broadcasts current parameters to all workers
2. Each worker receives a unique data batch
3. Workers compute forward + backward pass on WebGPU
4. Workers send gradients to coordinator
5. Coordinator averages gradients (all-reduce)
6. Coordinator applies optimizer step
7. Goto 1
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| GPU Compute | **TensorFlow.js** (`@tensorflow/tfjs-backend-webgpu`) |
| Frontend | TypeScript + Vite |
| Server | Node.js + Express + ws |
| Communication | WebRTC (DataChannels P2P) + WebSocket signaling |
| Styling | Vanilla CSS (premium dark theme) |
| Charting | Canvas-based (custom) |

## Production Roadmap & Next Steps

Ranked by the highest value for taking this distributed training to production.

### Priority 1: Port GPT Model to TensorFlow.js
**Value:** Massive architectural win. Replaces thousands of lines of fragile WGSL shaders with robust, optimized TFJS ops offering free automatic differentiation.
- [ ] Install `@tensorflow/tfjs` and `@tensorflow/tfjs-backend-webgpu`.
- [ ] Delete the custom `src/gpu/` module and manual WGSL shaders.
- [ ] Rewrite `GPTModel` using `tf.tidy`, `tf.matMul`, `tf.softmax`, and `tf.variable` for weights.
- [ ] Extract gradients efficiently using `tf.valueAndGrads` during the local forward pass.
- [ ] Apply averaged gradients across the ring using TFJS's `tf.train.adamw` optimizer.

### Priority 2: Gradient compression
**Value:** High. Raw parameter gradients are too large for efficient WebRTC transfer. Compression unlocks fast iteration speeds.
- [ ] Implement `f32` to `f16` quantization and dequantization in JavaScript/WGSL.
- [ ] Create a Top-K sparsification compute shader to identify and extract only the most significant gradient values.
- [ ] Update the WebRTC data channel payload serialization to pack and unpack the compressed and sparsified tensors.

### Priority 3: TURN server support & Cross-network training
**Value:** High. Browsers behind symmetric NATs (corporate networks, strict firewalls) will fail to connect P2P without TURN.
- [ ] Deploy a Coturn server or provision a cloud TURN provider (e.g., Twilio, Metered).
- [ ] Update the `RTCPeerConnection` configuration in `webrtc-mesh.ts` to include TURN credentials.
- [ ] Deploy the Node.js signaling server (`server/signaling.ts`) to a public endpoint (e.g., Render, Fly.io) to allow workers from different networks to discover each other.

### Priority 4: Dynamic ring reformation
**Value:** Medium-High. Essential for swarm stability. If a single tab closes, the current ring hangs. 
- [ ] Add strict timeout detection during `Reduce-Scatter` and `All-Gather` phases.
- [ ] Implement a lightweight heartbeat mechanism between peers and the signaling server.
- [ ] Automatically trigger ring rebuilding when a peer drops, discard the failed step's gradients, and seamlessly resume the training loop.

### Priority 5: Checkpoint saving
**Value:** Medium. Needed to extract the trained model and actually use the computation results.
- [ ] Implement weight serialization into a standard format (e.g., `.safetensors` or raw binary).
- [ ] Add an "Export Checkpoint" button in the Dashboard/Worker UI.
- [ ] Allow downloading the serialized Blob directly from the browser memory.
