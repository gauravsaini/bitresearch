# bitresearch — BTC/USD Trading Signal Generation

This is an experiment to train a GPT model to generate buy/sell signals from Bitstamp BTC/USD 1-minute OHLCV data.

## Setup

To set up a new experiment:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar21`). The branch `bitresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b bitresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `src/data/dataloader.ts` — data loading, best-fit packing, BPB evaluation. Do not modify.
   - `src/distributed/trainer.ts` — training loop, optimizer, gradient compression, all-reduce. You modify this.
   - `src/model/gpt.ts` — model architecture, forward pass, loss. You modify this for architecture changes.
   - `src/model/config.ts` — model hyperparameters (nLayer, nHead, nKvHead, nEmbd, vocabSize, sequenceLen, windowPattern).
   - `src/train/muon.ts` — MuonAdamW optimizer. You can tune coefficients here.
4. **Prepare data**: Run `python scripts/prepare_bitstamp.py` to process the bitstamp OHLCV data into tokenized format. This creates `public/data/tokens.bin` and `public/data/tokens_meta.json`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Start the servers**: Run `pnpm run dev:p2p` to launch the signaling server and Vite dev server.
7. **Open browser tabs**: Navigate to `http://localhost:5173/p2p.html` in multiple tabs. Each tab is an autonomous training node connected via WebRTC.
8. **Confirm and go**: Confirm setup looks good.

## Data Preparation

The bitstamp data is prepared using `scripts/prepare_bitstamp.py`:

```bash
# Full dataset (~6.8M records)
python scripts/prepare_bitstamp.py

# Sample for testing (500K rows)
python scripts/prepare_bitstamp.py --sample 500000
```

This script:
1. Loads Bitstamp BTC/USD 1-minute OHLCV data from `/home/gsai/bitstamp-btcusd-minute-data/data/historical/`
2. Computes technical indicators (RSI, MACD, Bollinger Bands, ATR, volume ratios)
3. Discretizes features into token vocabulary (hash-based, vocab_size=8192)
4. Generates buy/sell/hold signals based on future returns (0.2% threshold, 5-min lookahead)
5. Saves interleaved `[token, signal, token, signal, ...]` as binary Int32Array

## What You CAN Modify
- `src/distributed/trainer.ts` — training loop, optimizer config, schedules, batch size, gradient compression
- `src/model/gpt.ts` — model architecture, activation functions, normalization
- `src/model/config.ts` — hyperparameters (nLayer, nHead, nKvHead, nEmbd, sequenceLen, windowPattern)
- `src/train/muon.ts` — MuonAdamW coefficients (Polar Express, momentum, LR)

## What You CANNOT Modify
- `src/data/dataloader.ts` — fixed data loading and evaluation. Read-only.
- Install new packages. Only use what's in `package.json`.
- Break the WebRTC protocol format (gradient chunk header: `[type(1) | phase(1) | chunkIndex(u32) | stepId(u32) | data(f16[])]`).
- Remove f16 gradient compression — raw f32 payloads are too large for WebRTC data channels.
- Modify the validation evaluation. The `evaluateBpb()` method in `trainer.ts` is the ground truth metric for keep/discard decisions.

**The goal is simple: get the lowest val_bpb (validation bits per byte — lower is better).** val_loss is also logged for reference but is NOT the optimization target. Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

## Model Architecture (Karpathy's autoresearch parity)

The model has **exact feature parity** with Karpathy's autoresearch:

- **Rotary Embeddings**: Precomputed cos/sin tables, exact formula: `y1 = x1*cos + x2*sin`, `y2 = x1*(-sin) + x2*cos`
- **GQA (Grouped Query Attention)**: Separate n_kv_head for K/V sharing
- **Value Embeddings (ResFormer)**: Alternating layers get value embeddings with input-dependent sigmoid gating
- **Residual Lambdas**: `x = resid_lambdas[i] * x + x0_lambdas[i] * x0`
- **MLP ReLU²**: `relu(x).square()` activation (Karpathy's choice)
- **Softcap Logits**: `15 * tanh(logits / 15)` to prevent logit explosion
- **RMSNorm**: Pre-norm architecture
- **Sliding Window Attention**: Window pattern (e.g. "SSSL") with per-layer window sizes
- **Exact Weight Init**: Karpathy's uniform `3^0.5 * n_embd^-0.5`, zero projections, std=0.001 lm_head

## Optimizer (MuonAdamW parity)

The optimizer has **exact feature parity** with Karpathy's MuonAdamW:

- **Muon for matrix params**: Polar Express Newton-Schulz orthogonalization (5 iterations), NorMuon variance reduction, cautious weight decay (only where `grad*param >= 0`)
- **AdamW for scalars/embeddings**: Separate LR groups for lm_head, wte, value_embeddings, residLambdas, x0Lambdas
- **`dmodel_lr_scale`**: LRs auto-scaled by `(model_dim / 768)^-0.5` for dimension-independent tuning
- **Nesterov momentum**: Ramps from 0.85 → 0.95 over 300 steps
- **Weight decay schedule**: Decays to 0 over training duration
- **LR schedule**: Warmup → steady → warmdown (configurable ratios)

## Evaluation

The metric is **val_bpb** (validation bits per byte) — vocab-size-independent, comparable across architecture changes.

At training end, `evaluate_bpb` runs automatically:
- Sums per-token cross-entropy (in nats)
- Sums target byte lengths
- Converts nats/byte → bits/byte
- Special tokens (byte length 0) excluded

## The Experiment Loop

Each experiment runs on a dedicated branch (e.g. `bitresearch/mar21`).

**LOOP FOREVER:**

1. Look at git state: current branch/commit
2. Tune `src/distributed/trainer.ts` and/or `src/model/gpt.ts` with an experimental idea
3. `git add src/distributed/trainer.ts src/model/gpt.ts src/model/config.ts src/train/muon.ts && git commit -m "experiment: <description>"` (never `git add -A`)
4. Run experiment: `pnpm run dev:p2p`, open browser tabs, click "Start Training", wait 5 minutes
5. Read results: `grep "^val_bpb:" run.log`
6. If empty → crash. Run `tail -n 50 run.log` for stack trace.
7. Log results to `results.tsv` (tab-separated, don't commit)
8. If val_bpb improved (lower) → advance branch (keep commit)
9. If val_bpb equal/worse → `git reset --hard <previous kept commit>` to revert cleanly

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Output Format

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     4506.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

**Optimization target**: `val_bpb` only (lower = better).
Do NOT attempt to optimize: `peers`, `all_reduce_ms`, `tokens_per_sec`, `grad_norm`, or any WebRTC field.
These are infrastructure metrics. Ignore them when deciding keep/discard.

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 0.997900) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 4.4 — divide browser peak bytes by 1073741824) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	4.4	keep	baseline
b2c3d4e	0.993200	4.4	keep	increase LR to 0.04
c3d4e5f	1.005000	4.4	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## Ideas to Try

Based on Karpathy's autoresearch findings and the trading signal domain:

1. **Model depth**: Start with DEPTH=4 (baseline), try DEPTH=6, 8
2. **Window pattern**: Try "SSSL" (alternating short/long) vs "L" (full attention)
3. **Sequence length**: Longer context may capture multi-hour trends
4. **Vocab size**: Smaller vocab (4096) may generalize better on structured data
5. **Learning rates**: Karpathy found embedding_lr=0.6 works well; try scaling
6. **Weight decay**: 0.2 is Karpathy's default; try 0.0 (no decay) for comparison
7. **Head dimension**: Karpathy uses 128; try 64 for smaller models
8. **Signal generation**: Modify `prepare_bitstamp.py` to use different thresholds/lookahead
9. **Feature engineering**: Add more indicators (OBV, Ichimoku, VWAP) to prepare step
10. **Window sizes**: Wider windows may capture longer market cycles
