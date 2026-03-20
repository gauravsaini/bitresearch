# bitresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `bitresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b bitresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `src/data/dataloader.ts` — fixed data loading, token streaming, batch generation. Do not modify.
   - `src/distributed/trainer.ts` — the file you modify. Training loop, optimizer, gradient compression, all-reduce.
   - `src/model/gpt.ts` — model architecture, forward pass, loss. You modify this for architecture changes.
   - `src/model/config.ts` — model hyperparameters (nLayer, nHead, nEmbd, vocabSize, sequenceLen).
4. **Verify data exists**: Check that the configured parity bundle or shard location contains tokenized data. By default that means `/data/parity/manifest.json` or the legacy `/data/tokens.bin` alias. If exact BPB is expected, confirm the token-byte sidecar lives next to the data source. If not, tell the human to run the data preparation script and mirror the output into the browser-served bundle.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Establish baseline**: Run the experiment once as-is (no code changes) to record YOUR baseline on this hardware. Do NOT use baseline numbers from other platforms:
   ```
   pnpm run train:headless > run.log 2>&1
   ```
   Then `grep "^val_bpb:" run.log` and log it to `results.tsv` as the baseline.

Once the baseline is recorded, kick off the experimentation.

## Experimentation

Each experiment is launched by running:
```
pnpm run train:headless > run.log 2>&1
```
Do NOT use tee. Do NOT open tabs manually. Do NOT pipe to stdout. All output goes to run.log. The training loop runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation).

**What you CAN do:**
- Modify `src/distributed/trainer.ts` and `src/model/gpt.ts` — these are the only files you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, gradient compression, loss scaling, sparsification.

**What you CANNOT do:**
- Modify `src/data/dataloader.ts`. It is read-only. It contains the fixed data loading, batch generation, and train/val split.
- Install new packages or add dependencies. You can only use what's already in `package.json`.
- Break the WebRTC protocol format (gradient chunk header: `[type(1) | phase(1) | chunkIndex(u32) | stepId(u32) | data(f16[])]`).
- Remove the f16 gradient compression — raw f32 payloads are too large for WebRTC data channels.
- Modify the validation evaluation. The `evaluateBpb()` method in `trainer.ts` is the ground truth metric for keep/discard decisions.

**The goal is simple: get the lowest val_bpb (validation bits per byte — lower is better).** val_loss is also logged for reference but is NOT the optimization target. Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          1.234567
val_loss:         3.214567
training_seconds: 300.1
total_seconds:    325.9
mfu_percent:      12.40
tokens_per_sec:   1523
total_tokens_M:   45.6
num_steps:        1204
num_params_M:     4.2
depth:            4
peers:            3
all_reduce_ms:    127.3
grad_norm:        1.2345
```

**Optimization target**: `val_bpb` only (lower = better).
Do NOT attempt to optimize: `peers`, `all_reduce_ms`, `tokens_per_sec`, `grad_norm`, or any WebRTC field.
These are infrastructure metrics. Ignore them when deciding keep/discard.

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform and number of browser tabs the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:\|^mfu_percent:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 0.4 — divide browser peak bytes by 1073741824) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234567	0.4	keep	baseline
b2c3d4e	1.229234	0.4	keep	increase LR to 0.003
c3d4e5f	1.301000	0.4	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `bitresearch/mar5` or `bitresearch/mar5-gpu0`).

> **CRITICAL**: Always run as `pnpm run train:headless > run.log 2>&1`.
> NEVER use `tee`. NEVER pipe to stdout. NEVER open a browser tab manually.
> The headless script handles everything. Violating this floods your context window.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `src/distributed/trainer.ts` and/or `src/model/gpt.ts` with an experimental idea by directly hacking the code.
3. `git add src/distributed/trainer.ts src/model/gpt.ts && git commit -m "experiment: <description>"` (never `git add -A`)
4. Run the experiment:
   ```
   pnpm run train:headless > run.log 2>&1
   ```
   This launches headless browsers, runs training for 5 minutes, and captures all output to run.log.
   Do NOT use tee. Do NOT open any tabs. Wait for the command to exit.
5. Read out the results: `grep "^val_bpb:\|^mfu_percent:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, record the discard commit hash, then `git reset --hard <previous kept commit>` to revert cleanly

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
