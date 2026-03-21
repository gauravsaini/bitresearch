# WebGPU Autoresearch — Feature Parity Plan

> Port Karpathy's autoresearch training loop to a browser-native WebGPU/WebRTC stack without losing the high-value training semantics.

## Current Status (2026-03-21)

### Completed Parity Workstreams

- [x] **MuonAdamW optimizer** — Polar Express orthogonalization, NorMuon variance reduction, cautious weight decay, Nesterov momentum, per-group LR with dmodel scaling
- [x] **Model parity**
  - [x] Value embeddings and `ve_gate` (ResFormer, alternating layers)
  - [x] Exact Karpathy weight init: uniform(`3^0.5 * n_embd^-0.5`), proj zeros, wte std=1.0, lm_head std=0.001
  - [x] Rotary embeddings with exact cos/sin formula
  - [x] Sliding-window attention driven by `windowPattern`
  - [x] Grouped-query attention semantics for `nKvHead < nHead`
  - [x] MLP ReLU² activation, softcap logits (15*tanh)
  - [x] Residual lambdas with x0 skip connections
- [x] **Trainer parity**
  - [x] MuonAdamW optimizer integration
  - [x] LR schedule: warmup → steady → warmdown
  - [x] Momentum schedule: 0.85 → 0.95 over 300 steps
  - [x] Weight decay schedule
  - [x] MFU (Model FLOPs Utilization) tracking
  - [x] Fast fail on NaN/overflow
  - [x] Exact token-byte-aware BPB evaluation
- [x] **Data pipeline parity**
  - [x] BOS-aligned best-fit document packing
  - [x] Document-aware loader paths alongside legacy flat token streams
  - [x] Validation-compatible held-out splits
  - [x] Bitstamp BTC/USD OHLCV data preparation
- [x] **Prepare/tokenizer parity**
  - [x] Rust/WASM parity prep path using `rustbpe-wasm`
  - [x] Persisted tokenizer metadata and `token_bytes.bin`
  - [x] Stable large-shard tokenization without the JS/WASM stack overflow
  - [x] PyTorch reference script for gradient validation
- [x] **End-to-end browser artifact wiring**
  - [x] Mirror parity outputs from `~/.cache/autoresearch/` into `public/data/parity/`
  - [x] Emit `/data/parity/manifest.json` plus legacy root aliases under `public/data/`
  - [x] Load manifest-driven shard streams in the browser trainer by default
  - [x] Auto-discover token-byte sidecars from the parity bundle
- [x] **Distributed training robustness**
  - [x] Dynamic ring reformation
  - [x] Minimum ring size gating
  - [x] Abort-and-recompute all-reduce recovery
  - [x] Baseline f16 transport with dynamic loss scaling

### Next High-Value Gaps

- [ ] Decide whether browser training should stay stream-based on concatenated shard bins or grow a manifest-driven document mode for exact document-boundary splits at runtime.
- [ ] Top-k sparsification with error feedback for gradient compression
- [ ] Add profile-guided memory checks for larger parity corpora in real browser sessions
- [ ] Expose artifact/reference health directly in the UI
- [ ] Add `results.tsv` experiment logging

### Constraints

- CUDA-specific wins such as FlashAttention, `torch.compile`, and bf16 autocast need browser-native equivalents rather than literal ports.
- We prefer thin compatibility layers over duplicating Karpathy's code structure when the browser runtime already has a cleaner abstraction.

## Feature Parity Matrix

### Legend
- ✅ **Ported** — Already implemented in bitresearch
- 🟢 **Direct port** — TFJS has 1:1 equivalent ops
- 🟡 **Adapted** — Needs shader work or API difference, but math is identical
- 🔴 **Blocked** — Requires custom WGSL compute shader
- ⬜ **N/A** — Not applicable (WebRTC distributed vs single-GPU)

---

## 1. Model Architecture

| Feature | Karpathy (PyTorch) | bitresearch Status | TFJS Feasibility |
|---------|-------------------|-------------------|-----------------|
| **Token Embedding (`wte`)** | `nn.Embedding(vocab, n_embd)` | ✅ Ported | 🟢 `tf.gather` |
| **RMSNorm** | `F.rms_norm(x, (x.size(-1),))` | ✅ Ported | 🟢 `tf.rsqrt(tf.mean(tf.square(x))) * x` |
| **Rotary Embeddings** | Precomputed cos/sin, `apply_rotary_emb` | ✅ Ported | 🟢 Precompute tensors, element-wise ops |
| **GQA (n_kv_head)** | Separate Q/K/V projections with `n_kv_head` | ✅ Ported | 🟢 `tf.tile` K,V heads for attention |
| **Sliding Window Attention** | `window_pattern = "SSSL"`, banded causal mask | ✅ Ported | 🟡 `tf.linalg.bandPart` or mask tensor |
| **Value Embeddings (ResFormer)** | Alternating layers, input-dependent gate via `ve_gate` | ✅ Ported | 🟢 `tf.gather` + `tf.sigmoid` gate + element-wise add |
| **Residual Lambdas** | `resid_lambdas[i] * x + x0_lambdas[i] * x0` | ✅ Ported | 🟢 `tf.scalar` params, element-wise ops |
| **MLP (ReLU²)** | `relu(x).square()` | ✅ Ported | 🟢 `tf.relu(x).square()` |
| **Softcap logits** | `15 * tanh(logits / 15)` | ✅ Ported | 🟢 `tf.tanh` |
| **Weight init** | Custom: uniform `3^0.5 * n_embd^-0.5`, proj zeros, lm_head std=0.001 | ✅ Ported | 🟢 `tf.randomUniform`, `tf.zeros` |

## 2. Optimizer (MuonAdamW)

| Feature | Karpathy (PyTorch) | bitresearch Status | TFJS Feasibility |
|---------|-------------------|-------------------|-----------------|
| **AdamW (scalars/embeddings)** | Custom fused `adamw_step_fused` with `torch.compile` | ✅ Ported | 🟢 Custom impl with per-group LR/WD |
| **Muon (matrix params)** | Polar Express + NorMuon + cautious WD | ✅ Ported | 🟢 All ops are matMul + element-wise |
| **Polar Express** | 5 Newton-Schulz iterations | ✅ Ported | 🟢 `tf.matMul`, `tf.add`, `tf.mul` |
| **NorMuon variance reduction** | `v_norm / v_norm_new` scaling | ✅ Ported | 🟢 `tf.mean`, `tf.square`, `tf.sqrt` |
| **Cautious weight decay** | `mask = (grad * param) >= 0` | ✅ Ported | 🟢 `tf.greaterEqual`, `tf.where` |
| **Nesterov momentum** | Momentum buffer with lerp | ✅ Ported | 🟢 Manual implementation |
| **Per-group LR scaling** | Separate groups: lm_head, wte, value_embeds, scalars, matrices | ✅ Ported | 🟢 Custom optimizer param groups |
| **`dmodel_lr_scale`** | `(model_dim / 768) ** -0.5` | ✅ Ported | 🟢 Scalar multiply |
| **LR schedule** | Warmup → steady → warmdown | ✅ Ported | 🟢 JS math |
| **Momentum schedule** | 0.85 → 0.95 over 300 steps | ✅ Ported | 🟢 JS math |
| **Weight decay schedule** | `WEIGHT_DECAY * (1 - progress)` | ✅ Ported | 🟢 JS math |

## 3. Training Loop

| Feature | Karpathy (PyTorch) | bitresearch Status | TFJS Feasibility |
|---------|-------------------|-------------------|-----------------|
| **Fixed time budget** | 5 min wall clock (after warmup) | ✅ Ported | 🟢 JS `Date.now()` |
| **Gradient accumulation** | `TOTAL_BATCH_SIZE / (B * T)` micro-steps | ✅ Implemented | 🟢 Loop + accumulate |
| **`torch.compile`** | Both model and adamw_step are compiled | ⬜ N/A | 🟡 TFJS WebGPU compiles shaders at graph build |
| **GC management** | `gc.collect()` → `gc.freeze()` → `gc.disable()` | 🟡 Partial | 🟡 `tf.tidy()` aggressively |
| **Fast fail on NaN/overflow** | `if loss > 100 or isnan: exit(1)` | ✅ Ported | 🟢 `tf.isNaN`, `.dataSync()` |
| **MFU tracking** | `flops_per_token * batch / dt / peak_flops` | ✅ Ported | 🟡 Estimated from H100 specs |
| **VRAM tracking** | `torch.cuda.max_memory_allocated()` | ✅ Ported | 🟡 `tf.memory().numBytes` |
| **Smoothed loss logging** | EMA with `beta=0.9`, debiased | ✅ Implemented | 🟢 JS math |

## 4. Data & Evaluation

| Feature | Karpathy (PyTorch) | bitresearch Status | TFJS Feasibility |
|---------|-------------------|-------------------|-----------------|
| **BPE tokenizer** | `rustbpe` + `tiktoken` pickle | ✅ Via WASM | 🟢 `rustbpe-wasm` |
| **Best-fit packing** | Documents packed into rows, BOS-aligned | ✅ Ported | 🟢 Array manipulation |
| **`evaluate_bpb`** | Bits per byte: vocab-independent metric | ✅ Ported | 🟢 Sum cross-entropy, divide by byte lengths |
| **Token byte lookup** | `token_bytes.pt` tensor | ✅ Ported | 🟢 Decode each token, measure UTF-8 length |
| **`make_dataloader`** | Infinite iterator, pin_memory, GPU prefetch | ✅ Implemented | 🟢 `Int32Array` from fetch |

## 5. Distributed Training

| Feature | Karpathy (autoresearch) | bitresearch Status | TFJS Feasibility |
|---------|------------------------|-------------------|-----------------|
| **Single GPU** | One H100, no distributed | ✅ Multi-tab WebRTC | ⬜ N/A |
| **Ring All-Reduce** | N/A | ✅ Implemented | 🟢 Already done |
| **WebRTC DataChannels** | N/A | ✅ Implemented | 🟢 Already done |
| **f16 gradient compression** | N/A | ✅ Implemented | 🟢 Already done |
| **Dynamic ring reformation** | N/A | ✅ Implemented | 🟢 WebRTC signaling |

## 6. Scripts & Tooling

| Feature | Karpathy (PyTorch) | bitresearch Status | TFJS Feasibility |
|---------|-------------------|-------------------|-----------------|
| **`prepare.py`** | Downloads data, trains BPE, saves shards | ✅ Ported | 🟢 `scripts/prepare.py` |
| **`train.py`** | Single-file training | ✅ Split across files | 🟢 Already structured |
| **`program.md`** | Agent instructions | ✅ Implemented | 🟢 Already done |
| **`results.tsv`** | Experiment logging | ❌ Missing | 🟢 File I/O |
| **PyTorch reference** | `train_reference.py` | ✅ Ported | 🟢 `scripts/train_reference.py` |

---

## Production Roadmap

### Priority 1: Gradient Correctness ✅
- [x] Browser-side validation harness
- [x] Reference coverage checker
- [x] PyTorch reference script for gradient validation

### Priority 2: Distributed Training Robustness ✅
- [x] Dynamic ring reformation
- [x] Minimum ring size gating
- [x] Abort-and-recompute all-reduce recovery

### Priority 3: Compression and Efficiency
- [x] Baseline f16 transport
- [x] Dynamic loss scaling
- [ ] Top-k sparsification with error feedback
- [ ] Better browser memory instrumentation

### Priority 4: Usability
- [x] Headless launcher
- [x] Checkpoint save/load
- [ ] Expose artifact/reference health in UI
- [ ] `results.tsv` experiment logging
