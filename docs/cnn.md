# Intent-CNN Guide

This guide covers the official lightweight model path for the repo: `intent_cnn_policy`.

It uses:

- 3 recent RGB frames
- a task label chosen during recording
- a small CNN that regresses normalized `[vx, vy, omega]`

## What This Mode Is

Intent-CNN is a good fit for student projects where:

- you want a fast local training loop
- you want to condition behavior on a small set of task prompts such as `go left` or `go right`
- you want to stay lighter-weight than a VLA

Important behavior:

- the model is task-conditioned, but still closed-set
- it does **not** do open-vocabulary language understanding
- it can only drive using task strings that were present in the training data for that checkpoint

## Launcher Flow

```text
python -m client.cli --robot-ip <ROBOT_IP>
  -> CNN-based
  -> intent-conditioned (recommended)
```

During recording, students can either:

- pick one of the built-in tasks from `tasks.py::DEFAULT_INTENT_CNN_TASKS`
- or choose `Custom task...` and type a new task string

That typed task is saved like any other task and becomes part of the session vocabulary.

## Data Layout

Intent-CNN data should live in its own dataset root:

```text
data/turbopi_intent_cnn/
|-- raw/
|   `-- session_YYYYMMDD_HHMMSS/
|       |-- session_info.json
|       |-- telemetry.jsonl
|       `-- video.mp4
`-- episodes/
    `-- session_YYYYMMDD_HHMMSS/
        |-- session_info.json
        |-- tasks.json
        `-- episode_000000/
            |-- data.parquet
            |-- episode_info.json
            `-- video.mp4
```

Each accepted episode stores:

- `task`
- `task_index`
- `observation.state`
- `action`

## What the Model Sees

The baseline uses:

- 3 recent RGB frames
- resize to `160x120`
- stacked input tensor shaped `9 x 120 x 160`
- one task id from the saved task vocabulary

The network predicts:

```text
[vx, vy, omega]
```

Those values stay normalized and get converted back to robot command units only at inference time.

## Architecture

The model is intentionally small:

```text
Input: 9 x 120 x 160
Conv 9 -> 32, k=5, s=2, p=2 + BN + ReLU
Conv 32 -> 64, k=3, s=2, p=1 + BN + ReLU
Conv 64 -> 128, k=3, s=2, p=1 + BN + ReLU
Conv 128 -> 128, k=3, s=2, p=1 + BN + ReLU
Global Average Pool
Task embedding
MLP head
Linear 32 -> 3 + Tanh
```

Why this shape:

- it trains quickly on a laptop
- it is easy to debug
- it is strong enough for task-conditioned path following without jumping straight to a heavyweight model

## Install the CNN Extras

```bash
pip install -r requirements-cnn.txt
```

## Training Command

```bash
python -m intent_cnn_policy.train \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --run-dir runs/intent_cnn_v1
```

The trainer creates a timestamped child run such as `runs/intent_cnn_v1/run_YYYYMMDD_HHMMSS`.

Training defaults:

- Huber loss on normalized actions
- session-level train/val split
- light brightness, contrast, hue, blur, and geometry augmentation
- one timestamped run folder per launch

## Evaluation

```bash
python -m intent_cnn_policy.eval \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --checkpoint <RUN_DIR>/checkpoints/best.pt
```

## Inference

```bash
python -m intent_cnn_policy.drive \
  --robot-ip <ROBOT_IP> \
  --checkpoint <RUN_DIR>/checkpoints/best.pt \
  --task "go left"
```

You can also use `--task-index`, but `--task` is usually easier for students.

## Upload One Session To Hugging Face

```bash
python scripts/upload_hf_session.py
```

The uploader scans the local `data/` tree by default, so it can find sessions under `data/turbopi_intent_cnn/` without extra flags.

If you prefer the terminal:

```bash
python scripts/upload_hf_session.py --no-gui
```

Dry run only:

```bash
python scripts/upload_hf_session.py --dry-run --no-gui
```

## Legacy No-Language Loop CNN

The repo still ships the older no-language loop-CNN workflow for taped loops and lap-based collection.

That path is now secondary:

```text
python -m client.cli --robot-ip <ROBOT_IP>
  -> CNN-based
  -> no-language loop mode (legacy)
```

The legacy training/inference package is `cnn_policy/`. Its underlying implementation still lives in `loop_cnn/`, but `cnn_policy/` is the public compatibility wrapper.

## Common Issues

- If the model trains but a custom task does not work at inference, confirm that exact task string appeared in the training data.
- If the robot drifts away from the path, collect more clean recovery examples.
- If motion feels jittery, increase runtime smoothing before making the model larger.
- If the path is hard to see, improve lighting before adding more data.
