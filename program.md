# autoresearch — Parameter Golf

This is an autonomous research agent for OpenAI's **Parameter Golf** challenge. The goal: train the best language model that fits in a **16MB artifact** (code + compressed model) and trains in **under 10 minutes on 1xH100**, evaluated by compression on the FineWeb validation set (tokenizer-agnostic, **bits per byte**).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the training script**: Read `train_gpt.py` in full — it contains the model architecture, optimizer, training loop, data loading, evaluation, and quantization. This is the only file you modify.
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains training shards and `./data/tokenizers/` contains the tokenizer. If not, tell the human to run `uv run python data/cached_challenge_fineweb.py --variant sp1024`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on **1xH100 GPU** in an interactive session. The training script has a built-in **10-minute wallclock cap** (`MAX_WALLCLOCK_SECONDS=600`). You launch it as:

```bash
uv run python train_gpt.py > run.log 2>&1
```

**What you CAN modify:**
- `train_gpt.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, quantization scheme, evaluation strategy, etc.

**What you CANNOT do:**
- Modify data files in `data/` — the dataset and tokenizer files are pre-downloaded and fixed.
- Install new packages beyond what's in `pyproject.toml`. Use only what's available.
- Access validation data during training or cheat the evaluation.

**The goal is simple: get the lowest `val_bpb` after int8+zlib quantization, while keeping the total artifact under 16MB.**

The artifact size = code bytes (`train_gpt.py`) + compressed model bytes (`final_model.int8.ptz`). The cap is 16,000,000 bytes. If your changes cause the artifact to exceed 16MB, the run is invalid — treat it as a failure and discard.

**VRAM** is a soft constraint (~80GB on H100). Some increase is acceptable for meaningful val_bpb gains, but OOM is a crash.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline — run `train_gpt.py` as-is.

## Output format

The script prints training logs during the run, then at the end produces the key metrics. The lines you care about are:

```
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX eval_time:XXXms
Total submission size int8+zlib: XXXXX bytes
peak memory allocated: XXXX MiB
```

The **primary metric** is `val_bpb` from the `final_int8_zlib_roundtrip` line — this is the post-quantization score, which is what the challenge actually evaluates.

Extract key metrics from the log:

```bash
grep "final_int8_zlib_roundtrip val_loss\|Total submission size int8+zlib\|peak memory allocated" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	val_bpb	artifact_mb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (from `final_int8_zlib_roundtrip` line) — use 0.000000 for crashes
3. total artifact size in MB, round to .2f (divide `Total submission size int8+zlib` by 1_000_000) — use 0.00 for crashes. **Must be < 16.00**
4. peak memory in GB, round to .1f (divide `peak memory allocated` MiB by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_bpb	artifact_mb	memory_gb	status	description
a1b2c3d	1.2244	14.52	44.0	keep	baseline
b2c3d4e	1.2100	14.80	44.2	keep	increase num_layers to 10
c3d4e5f	1.2300	14.50	44.0	discard	switch to GeLU (worse)
d4e5f6g	0.0000	0.00	0.0	crash	double model width (OOM)
e5f6g7h	1.1900	16.20	45.0	discard	int6 quantization (artifact too large)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar22`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train_gpt.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run python train_gpt.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "final_int8_zlib_roundtrip val_loss\|Total submission size int8+zlib\|peak memory allocated" run.log`
6. If the grep output is empty or missing the `final_int8_zlib_roundtrip` line, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on this idea.
7. Check the artifact size — if `Total submission size int8+zlib` exceeds 16,000,000 bytes, the run is **invalid** even if val_bpb improved. Treat it as a discard.
8. Record the results in the tsv (NOTE: do not commit results.tsv, leave it untracked by git)
9. If val_bpb improved (lower) AND artifact is under 16MB, you "advance" the branch, keeping the git commit
10. If val_bpb is equal or worse or artifact is over 16MB, you `git reset --hard` back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~10 minutes (+ a few minutes for compilation warmup and eval overhead). If a run exceeds 20 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: if it's a typo or easy fix, fix and re-run. If the idea is fundamentally broken, log "crash", revert, and move on.

**NEVER STOP**: Once the experiment loop has begun (after initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read `train_gpt.py` for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example: if each experiment takes ~12 minutes end-to-end, you can run ~5/hour, ~40 overnight. The user wakes up to a full results.tsv of experiments.
