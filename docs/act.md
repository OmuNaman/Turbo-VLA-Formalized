# ACT-Intent Guide

This guide covers the experimental chunked-action policy path in `act_intent_policy`.

It reuses the same task-conditioned dataset as `intent_cnn_policy`, but predicts a short future chunk of normalized actions instead of a single action.

## What This Mode Is

ACT-Intent is a good fit when:

- you want to keep the same classroom-friendly task recording flow
- you want a stronger research baseline than the tiny Intent-CNN
- you want chunked action prediction without the full SmolVLA stack

Important behavior:

- this is still a closed-set task-label model, not free-text language understanding
- it is experimental and less battle-tested than `intent_cnn_policy`
- it uses continuous normalized `[vx, vy, omega]` supervision, not binary button labels

## Recording Flow

ACT-Intent does **not** add a new recording mode.

Use the same task-conditioned dataset path as Intent-CNN:

```text
python -m client.cli --robot-ip <ROBOT_IP>
  -> CNN-based
  -> intent-conditioned dataset (recommended)
```

Dataset root:

```text
data/turbopi_intent_cnn/
```

That means:

- built-in tasks and `Custom task...` work exactly the same way
- recorded `task` and `task_index` fields are reused directly
- existing task-conditioned sessions can train either Intent-CNN or ACT-Intent

## What the Model Sees

Inputs:

- 3 recent RGB frames
- resize to `160x120` by default
- one saved task id from the session vocabulary

Outputs:

- a chunk of future normalized actions
- default chunk size: `8`
- each action is `[vx, vy, omega]`

Training uses:

- Smooth L1 / Huber reconstruction loss over valid chunk positions
- KL regularization with warmup for the CVAE latent
- AdamW with warmup + cosine-style decay

## Training

Install the same extras used for Intent-CNN:

```bash
pip install -r requirements-cnn.txt
```

Train:

```bash
python -m act_intent_policy.train \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --run-dir runs/act_intent_v1
```

Common useful knobs:

```bash
python -m act_intent_policy.train \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --run-dir runs/act_intent_v1 \
  --chunk-size 8 \
  --batch-size 16 \
  --epochs 30 \
  --d-model 128 \
  --latent-dim 32
```

## Fast Cache Workflow

The recommended ACT path on strong GPUs is:

1. build the ACT cache once
2. train from that cache repeatedly

Why:

- raw ACT training still decodes MP4s and does CPU image work
- the cache path stores prebuilt frame-history stacks and chunk targets
- training then spends its time moving packed arrays to GPU instead of rebuilding samples in Python

Build the cache:

```bash
python -m act_intent_policy.cache \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --cache-dir data/turbopi_intent_cnn/act_cache_w160_h120_hist3_chunk8 \
  --workers 8
```

Train from cache:

```bash
python -m act_intent_policy.train \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --cache-dir data/turbopi_intent_cnn/act_cache_w160_h120_hist3_chunk8 \
  --cache-mode require \
  --run-dir runs/act_intent_v1 \
  --device cuda \
  --batch-size 128 \
  --num-workers 8
```

Or build/reuse the cache automatically from the train command:

```bash
python -m act_intent_policy.train \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --cache-mode build \
  --run-dir runs/act_intent_v1 \
  --device cuda \
  --batch-size 128 \
  --num-workers 8
```

Useful fast-path flags:

- `--cache-mode build`: create or reuse a compatible cache automatically
- `--cache-mode require`: fail fast unless a compatible cache already exists
- `--cache-dir <DIR>`: choose where the reusable cache lives
- `--cache-workers <N>` / `cache.py --workers <N>`: parallelize cache build across episodes
- `--amp` / `--no-amp`: mixed precision is enabled by default on CUDA
- `--prefetch-factor <N>`: adjust worker prefetching for cached training
- `--compile`: opt into `torch.compile` on CUDA when supported

The train logs now report:

- data source: raw vs cache
- AMP on/off
- samples/sec and batches/sec
- average step/data/compute time
- peak GPU allocated/reserved memory

If `data_ms` is larger than `compute_ms`, the run is still data-bound.

## Evaluation

```bash
python -m act_intent_policy.eval \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --checkpoint <RUN_DIR>/checkpoints/best.pt
```

The evaluation output reports:

- overall validation loss
- chunk MAE across valid predicted steps
- first-step MAE for `vx`, `vy`, and `omega`

That first-step metric matters most for online control, because the driver usually replans every frame.

## Driving the Robot

```bash
python -m act_intent_policy.drive \
  --robot-ip <ROBOT_IP> \
  --checkpoint <RUN_DIR>/checkpoints/best.pt \
  --task "go left"
```

Important runtime behavior:

- by default the driver replans from the newest camera frame each step
- this is usually better than consuming a long cached chunk open-loop
- if you want to test open-loop chunk execution anyway, pass `--reuse-action-queue`

Useful runtime tuning:

```bash
python -m act_intent_policy.drive \
  --robot-ip <ROBOT_IP> \
  --checkpoint <RUN_DIR>/checkpoints/best.pt \
  --task "go right" \
  --vx-cap 60 \
  --omega-cap 35 \
  --min-vx 18 \
  --min-omega 10 \
  --smoothing 0.25
```

## When To Use This Instead of Intent-CNN

Prefer `intent_cnn_policy` when:

- you want the simplest and most reliable student baseline
- you want faster local training
- you want the most battle-tested lightweight path in this repo

Try `act_intent_policy` when:

- the single-step CNN feels too limited
- you want chunk prediction experiments
- you want a stronger intermediate baseline before SmolVLA

## Limitations

- It still only supports task labels seen during training.
- It is more complex and slower than the Intent-CNN baseline.
- Chunk prediction can become sluggish if you force the driver to reuse the whole predicted queue open-loop.
