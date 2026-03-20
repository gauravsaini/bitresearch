# WebGPU Distributed Training — Implementation Plan

> Port Karpathy's autoresearch training loop to a browser-native WebGPU/WebRTC stack without losing the high-value training semantics.

## Current Focus — Karpathy Parity (2026-03-21)

Primary goal: keep closing the implementation gap against `karpathy/autoresearch` while preserving the browser-first architecture.

### Completed Parity Workstreams

- [x] Data pipeline parity
  - [x] BOS-aligned best-fit document packing
  - [x] Document-aware loader paths alongside legacy flat token streams
  - [x] Validation-compatible held-out splits

- [x] Model parity
  - [x] Value embeddings and `ve_gate`
  - [x] Sliding-window attention driven by `windowPattern`
  - [x] Grouped-query attention semantics for `nKvHead < nHead`

- [x] Trainer parity
  - [x] Exact token-byte-aware BPB when sidecars are present
  - [x] Warmup / warmdown LR scheduling
  - [x] Throughput accounting that matches the new loader/model surfaces

- [x] Prepare/tokenizer parity
  - [x] Rust/WASM parity prep path using `rustbpe-wasm`
  - [x] Persisted tokenizer metadata and `token_bytes.bin`
  - [x] Stable large-shard tokenization without the JS/WASM stack overflow

- [x] End-to-end browser artifact wiring
  - [x] Mirror parity outputs from `~/.cache/autoresearch/` into `public/data/parity/`
  - [x] Emit `/data/parity/manifest.json` plus legacy root aliases under `public/data/`
  - [x] Load manifest-driven shard streams in the browser trainer by default
  - [x] Auto-discover token-byte sidecars from the parity bundle

- [x] Validation hygiene
  - [x] Detect stale reference bundles before pretending parity still holds
  - [x] Add a local checker for `public/reference/` completeness

### Next High-Value Gaps

- [ ] Refresh `public/reference/` assets for the new model shape so forward/backward parity can be re-established, not just safely skipped.
- [ ] Decide whether browser training should stay stream-based on concatenated shard bins or grow a manifest-driven document mode for exact document-boundary splits at runtime.
- [ ] Revisit optimizer parity beyond Adam scheduling if we want a closer analogue to Karpathy’s `MuonAdamW`.
- [ ] Add profile-guided memory checks for larger parity corpora in real browser sessions.

### Constraints

- CUDA-specific wins such as FlashAttention, `torch.compile`, and bf16 autocast need browser-native equivalents rather than literal ports.
- `PLAN.md` and `TIMELINE.md` stay owned by the main thread so parallel agents do not trample project tracking.
- We prefer thin compatibility layers over duplicating Karpathy’s code structure when the browser runtime already has a cleaner abstraction.

## Near-Term Execution Order

1. Refresh and verify reference assets against the value-embedding/windowed model.
2. Run a real browser training smoke test against the parity manifest bundle.
3. Evaluate whether the runtime should preserve per-document boundaries instead of concatenating shard streams.
4. Continue optimizer and memory-profile parity work only after the data/reference path is stable.

## Production Roadmap

### Priority 1: Gradient Correctness

- [x] Browser-side validation harness exists.
- [x] Reference coverage checker exists.
- [ ] Regenerate reference tensors for the current model.
- [ ] Add a one-command refresh workflow for reference assets.

### Priority 2: Distributed Training Robustness

- [x] Dynamic ring reformation
- [x] Minimum ring size gating
- [x] Abort-and-recompute all-reduce recovery
- [ ] Profile multi-peer runs with parity-scale data

### Priority 3: Compression and Efficiency

- [x] Baseline f16 transport
- [x] Dynamic loss scaling
- [ ] Top-k sparsification with error feedback
- [ ] Better browser memory instrumentation around shard loading and evaluation

### Priority 4: Usability

- [x] Headless launcher
- [x] Checkpoint save/load
- [x] Default parity data mirroring into browser-served assets
- [ ] Expose artifact/reference health directly in the UI
