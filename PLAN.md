# WebGPU Autoresearch — Feature Parity Plan

> Goal: Exact feature parity with Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) using TypeScript + TensorFlow.js on WebGPU.

## Feature Parity Matrix

### Legend
- ✅ **Ported** — Already implemented in bitresearch
- 🟢 **Direct port** — TFJS has 1:1 equivalent ops
- 🟡 **Adapted** — Needs shader work or API difference, but math is identical
- 🔴 **Blocked** — Requires custom WGSL compute shader
- ⬜ **N/A** — Not applicable (WebRTC distributed vs single-GPU)

---

## 1. Model Architecture

| Feature | Karpathy (PyTorch) | bitresearch Current | TFJS Feasibility | Plan |
|---------|-------------------|---------------------|-----------------|------|
| **Token Embedding (`wte`)** | `nn.Embedding(vocab, n_embd)` | ✅ Implemented | 🟢 `tf.gather` | — |
| **RMSNorm** | `F.rms_norm(x, (x.size(-1),))` | ✅ Implemented | 🟢 `tf.rsqrt(tf.mean(tf.square(x))) * x` | — |
| **Rotary Embeddings** | Precomputed cos/sin, `apply_rotary_emb` | ❌ Missing | 🟢 Precompute tensors, element-wise ops | Add `precomputeRotaryEmbeddings()`, `applyRotaryEmb()` |
| **GQA (n_kv_head)** | Separate Q/K/V projections with `n_kv_head` | ❌ Missing (uses MHA) | 🟢 `tf.tile` K,V heads for attention | Change `CausalSelfAttention` to use `n_kv_head`, add head repeat |
| **Sliding Window Attention** | `window_pattern = "SSSL"`, banded causal mask | ✅ Implemented (basic) | 🟡 `tf.linalg.bandPart` or mask tensor | Verify window masking matches Karpathy's FA3 window_size semantics |
| **Value Embeddings (ResFormer)** | Alternating layers, input-dependent gate via `ve_gate` | ❌ Missing | 🟢 `tf.gather` + `tf.sigmoid` gate + element-wise add | Add `valueEmbeds` ModuleDict, `veGate` linear layer, alternating logic |
| **Residual Lambdas** | `resid_lambdas[i] * x + x0_lambdas[i] * x0` | ❌ Missing | 🟢 `tf.scalar` params, element-wise ops | Add `residLambdas` and `x0Lambdas` as trainable scalars |
| **MLP (SwiGLU→ReLU²)** | `relu(x).square()` | ❌ Missing (uses GELU) | 🟢 `tf.relu(x).square()` | Change activation from GELU to ReLU² |
| **Softcap logits** | `15 * tanh(logits / 15)` | ❌ Missing | 🟢 `tf.tanh` | Add softcap after lm_head |
| **BF16 activations** | `torch.amp.autocast(dtype=bfloat16)` | ❌ Missing | 🟡 TFJS WebGPU supports f16; bf16 needs custom cast | Use `tf.cast(x, 'float32')` internally; f16 for storage |
| **Weight init** | Custom: uniform `3^0.5 * n_embd^-0.5`, proj zeros, lm_head std=0.001 | ❌ Missing (random normal) | 🟢 `tf.randomUniform`, `tf.zeros` | Implement `initWeights()` matching PyTorch exact |

## 2. Optimizer (MuonAdamW)

| Feature | Karpathy (PyTorch) | bitresearch Current | TFJS Feasibility | Plan |
|---------|-------------------|---------------------|-----------------|------|
| **AdamW (scalars/embeddings)** | Custom fused `adamw_step_fused` with `torch.compile` | ✅ `tf.train.adamw` | 🟢 TFJS built-in | Need custom impl for per-group LR/WD |
| **Muon (matrix params)** | Polar Express + NorMuon + cautious WD | ❌ Missing (AdamW only) | 🟢 All ops are matMul + element-wise | **Critical**: Implement `MuonOptimizer` class |
| **Polar Express** | 5 Newton-Schulz iterations: `A=X^T@X, B=b*A+c*A*A, X=a*X+X@B` | ❌ Missing | 🟢 `tf.matMul`, `tf.add`, `tf.mul` | Port coefficients and iteration logic |
| **NorMuon variance reduction** | `v_norm / v_norm_new` scaling | ❌ Missing | 🟢 `tf.mean`, `tf.square`, `tf.sqrt` | Port variance tracking |
| **Cautious weight decay** | `mask = (grad * param) >= 0`, apply WD only where sign matches | ❌ Missing | 🟢 `tf.greaterEqual`, `tf.where` | Port masking logic |
| **Nesterov momentum** | `momentum_buffer.lerp_(grad, 1-momentum); g = lerp(grad, buf, momentum)` | ❌ Missing | 🟢 `tf.addScaled` or manual | Add momentum buffer state |
| **Per-group LR scaling** | Separate groups: lm_head, wte, value_embeds, scalars, matrices | ❌ Single LR | 🟢 TFJS custom optimizer | Implement param groups in optimizer |
| **`dmodel_lr_scale`** | `(model_dim / 768) ** -0.5` | ❌ Missing | 🟢 Scalar multiply | Add to optimizer setup |
| **LR schedule** | Warmup → steady → warmdown | ✅ Basic decay | 🟢 JS math | Match Karpathy's `get_lr_multiplier(progress)` |
| **Momentum schedule** | 0.85 → 0.95 over 300 steps | ❌ Missing | 🟢 JS math | Add `get_muon_momentum(step)` |
| **Weight decay schedule** | `WEIGHT_DECAY * (1 - progress)` | ❌ Missing | 🟢 JS math | Add `get_weight_decay(progress)` |

## 3. Training Loop

| Feature | Karpathy (PyTorch) | bitresearch Current | TFJS Feasibility | Plan |
|---------|-------------------|---------------------|-----------------|------|
| **Fixed time budget** | 5 min wall clock (after warmup) | ✅ 5 min | 🟢 JS `Date.now()` | — |
| **Gradient accumulation** | `TOTAL_BATCH_SIZE / (B * T)` micro-steps | ✅ Implemented | 🟢 Loop + accumulate | — |
| **`torch.compile`** | Both model and adamw_step are compiled | ❌ N/A | 🟡 TFJS WebGPU compiles shaders at graph build | Pre-warm with `tf.ready()` + dummy forward pass |
| **GC management** | `gc.collect()` → `gc.freeze()` → `gc.disable()` | ❌ Missing | 🟡 JS GC can't be frozen, but can minimize allocs | Use `tf.tidy()` aggressively, pre-allocate tensors |
| **Fast fail on NaN/overflow** | `if loss > 100 or isnan: exit(1)` | ❌ Missing | 🟢 `tf.isNaN`, `.dataSync()` | Add NaN/overflow check after each step |
| **MFU tracking** | `flops_per_token * batch / dt / peak_flops` | ❌ Missing | 🟡 No peak FLOP API in browser | Can estimate based on GPU model or skip |
| **VRAM tracking** | `torch.cuda.max_memory_allocated()` | ❌ Missing | 🟡 `tf.memory().numBytes` (approximate) | Use `tf.memory()` after each step |
| **Smoothed loss logging** | EMA with `beta=0.9`, debiased | ✅ Implemented | 🟢 JS math | — |
| **`torch.set_float32_matmul_precision("high")`** | Use TF32 on Ampere+ | ⬜ N/A | ⬜ WebGPU handles precision | — |

## 4. Data & Evaluation

| Feature | Karpathy (PyTorch) | bitresearch Current | TFJS Feasibility | Plan |
|---------|-------------------|---------------------|-----------------|------|
| **BPE tokenizer** | `rustbpe` + `tiktoken` pickle | ❌ Missing (raw tokens) | 🟡 Can use pre-tokenized data OR port BPE | For now: use pre-tokenized binary. Later: JS BPE |
| **Best-fit packing** | Documents packed into rows, BOS-aligned | ❌ Missing | 🟢 Array manipulation | Port packing algorithm |
| **`evaluate_bpb`** | Bits per byte: vocab-independent metric | ❌ Missing (uses `val_loss`) | 🟢 Sum cross-entropy, divide by byte lengths | **Critical**: Implement BPB for fair comparison |
| **Token byte lookup** | `token_bytes.pt` tensor | ❌ Missing | 🟢 Decode each token, measure UTF-8 length | Generate and save `token_bytes.json` |
| **`make_dataloader`** | Infinite iterator, pin_memory, GPU prefetch | ✅ Basic streaming | 🟡 `Int32Array` from fetch, no pin_memory | Already works, optimize prefetch |

## 5. Distributed Training

| Feature | Karpathy (autoresearch) | bitresearch Current | TFJS Feasibility | Plan |
|---------|------------------------|---------------------|-----------------|------|
| **Single GPU** | One H100, no distributed | ✅ Multi-tab WebRTC | ⬜ N/A | bitresearch has MORE than Karpathy here |
| **Ring All-Reduce** | N/A | ✅ Implemented | 🟢 Already done | — |
| **WebRTC DataChannels** | N/A | ✅ Implemented | 🟢 Already done | — |
| **f16 gradient compression** | N/A | ✅ Implemented | 🟢 Already done | — |
| **Dynamic ring reformation** | N/A | ❌ Partial | 🟢 WebRTC signaling | Complete heartbeat + reformation |

## 6. Scripts & Tooling

| Feature | Karpathy (PyTorch) | bitresearch Current | TFJS Feasibility | Plan |
|---------|-------------------|---------------------|-----------------|------|
| **`prepare.py`** | Downloads data, trains BPE, saves shards | ❌ Different format | 🟢 Port to Node.js script | Create `scripts/prepare.ts` |
| **`train.py`** | Single-file training | ✅ Split across files | 🟢 Already structured | — |
| **`program.md`** | Agent instructions | ✅ Implemented | 🟢 Already done | — |
| **`results.tsv`** | Experiment logging | ❌ Missing | 🟢 File I/O | Add results logging |
| **PyTorch reference script** | `train_reference.py` | ❌ Missing | 🟢 Keep Python for validation | Create reference for gradient checking |

---

## Implementation Priority

### Phase 1: Model Parity (Highest Impact)
1. **Rotary Embeddings** — `src/model/gpt.ts:precomputeRotaryEmbeddings()`
2. **GQA (n_kv_head)** — `src/model/gpt.ts:CausalSelfAttention` refactor
3. **Value Embeddings (ResFormer)** — `src/model/gpt.ts` add `valueEmbeds`, `veGate`
4. **Residual Lambdas** — `src/model/gpt.ts` add `residLambdas`, `x0Lambdas`
5. **MLP ReLU²** — Change activation in `src/model/gpt.ts:MLP`
6. **Softcap logits** — After `lm_head` in forward pass
7. **Weight init** — `src/model/gpt.ts:initWeights()` exact PyTorch match

### Phase 2: Optimizer Parity (Critical for Convergence)
8. **MuonAdamW optimizer** — New file `src/train/muon.ts`
   - Polar Express orthogonalization (5 Newton-Schulz iterations)
   - NorMuon variance reduction
   - Cautious weight decay
   - Nesterov momentum buffer
   - Per-group parameter groups
9. **LR/Momentum/WD schedules** — `src/train/trainer.ts`
10. **`dmodel_lr_scale`** — In optimizer setup

### Phase 3: Evaluation Parity (Critical for Fair Comparison)
11. **`evaluate_bpb`** — New function in `src/data/dataloader.ts`
12. **Token byte lookup** — Generate `token_bytes.json` in prepare step
13. **Best-fit packing** — Refactor `src/data/dataloader.ts`

### Phase 4: Training Loop Polish
14. **GC management** — `tf.tidy()` usage review
15. **Fast fail** — NaN/overflow detection
16. **VRAM tracking** — `tf.memory()` instrumentation
17. **MFU estimation** — Best-effort based on known GPU specs

### Phase 5: Tooling
18. **`scripts/prepare.ts`** — Node.js data prep (BPE + tokenization)
19. **`results.tsv` logging** — Experiment tracking
20. **PyTorch reference** — `scripts/train_reference.py` for gradient validation

---

## Key Porting Notes

### MuonAdamW — The Biggest Missing Piece

Karpathy's `MuonAdamW` is the single most impactful missing feature. The math is entirely matrix operations that TFJS supports:

```typescript
// Polar Express orthogonalization (5 iterations)
// Each iteration: A = X^T @ X, B = b*A + c*A@A, X = a*X + X@B
for (const [a, b, c] of POLAR_EXPRESS_COEFFS.slice(0, nsSteps)) {
  const A = tf.matMul(X, X, false, true);  // X^T @ X
  const B = tf.add(tf.mul(b, A), tf.mul(c, tf.matMul(A, A)));
  X = tf.add(tf.mul(a, X), tf.matMul(X, B));
}
```

The `torch.compile` on the fused AdamW step is a CUDA optimization that doesn't apply to TFJS (WebGPU has its own shader compilation). The algorithmic logic ports 1:1.

### Why This Works on WebGPU

All Karpathy's innovations are **algorithmic**, not CUDA-specific:
- Polar Express = matrix multiplications → `tf.matMul`
- NorMuon = variance tracking → `tf.mean`, `tf.square`, `tf.sqrt`
- Cautious WD = sign masking → `tf.greaterEqual`, `tf.where`
- Value Embeddings = gather + sigmoid + add → `tf.gather`, `tf.sigmoid`, `tf.add`
- Rotary = precomputed cos/sin tables → `tf.mul`, element-wise

The only thing that doesn't port is `torch.compile` (CUDA graph fusion), but TFJS WebGPU compiles to WGSL shaders at graph build time, which is approximately equivalent.
