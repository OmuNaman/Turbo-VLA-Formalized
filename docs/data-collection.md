# Data Collection Guide

This repo records teleoperated episodes from a laptop while the TurboPi runs the robot server.

The official student-facing recording paths are:

- `VLA-based`: task-conditioned recording for LeRobot export and SmolVLA
- `CNN-based -> intent-conditioned dataset (recommended)`: task-conditioned recording for `intent_cnn_policy` and `act_intent_policy`

The older no-language loop CNN path still exists, but it is secondary.

## Start a Recording Session

Teleop-only smoke test:

```bash
python -m client.teleop --robot-ip <ROBOT_IP>
```

Full launcher:

```bash
python -m client.cli --robot-ip <ROBOT_IP>
```

AP-mode quick test:

```bash
python -m client
```

After launch:

- choose `VLA-based` for the LeRobot / SmolVLA data path
- choose `CNN-based -> intent-conditioned dataset (recommended)` for the Intent-CNN or ACT-Intent path

## Controls

- `W`, `A`, `S`, `D`: drive forward, left, backward, right
- `Q`, `E`: rotate
- `+`, `-`: increase or decrease teleop speed
- right arrow: start recording after positioning, or accept the current episode
- left arrow: discard the current episode
- `Esc`: stop the full session

## Built-In Tasks Plus Custom Tasks

The VLA and intent-conditioned CNN recorders both show:

- the built-in task list from `tasks.py`
  VLA uses `DEFAULT_TASKS`
  Intent-CNN uses `DEFAULT_INTENT_CNN_TASKS`
- one extra option: `Custom task...`

If a student selects `Custom task...`:

1. the client prompts for a task string
2. that string is assigned a session-local `task_index`
3. the task is appended to the current session list
4. later episodes in the same session can reuse it from the menu directly

This keeps the default classroom task list simple while still allowing one-off prompts without editing code.

## What Gets Saved

The client creates a fresh timestamped session folder by default, or resumes a named session if you pass `--session-name`.

Example layout:

```text
data/turbopi_nav/
|-- raw/
|   `-- session_20260401_101500/
|       |-- session_info.json
|       |-- telemetry.jsonl
|       `-- video.mp4
`-- episodes/
    `-- session_20260401_101500/
        |-- session_info.json
        |-- tasks.json
        `-- episode_000000/
            |-- data.parquet
            |-- episode_info.json
            `-- video.mp4
```

Why there are two outputs:

- `raw/` keeps a continuous backup of the full session
- `episodes/` keeps only accepted episodes in a cleaner training/export structure

Important saved fields:

- `task`: exact task text selected for that episode
- `task_index`: session-local integer id for that task
- `observation.state`: previous normalized action
- `action`: current normalized teleop command

The raw telemetry backup also records the active task text for each saved frame.

## Why the Saved State Matters

For training, `observation.state` should not be the same thing as the current target action.

This recorder saves:

- `observation.state`: previous normalized action
- `action`: current normalized teleop command

That avoids leaking the answer into the model input.

## New Sessions vs Resume

By default each run creates a new folder named like `session_YYYYMMDD_HHMMSS`.

That means:

- old recordings stay untouched
- you can compare sessions later
- students do not lose data by starting a new run

If you pass `--session-name`, the client resumes that session instead:

- accepted episodes keep their saved `task_index` ordering
- raw telemetry appends to the existing backup
- new custom tasks are appended after the already-saved task list

## Export to LeRobot

Accepted VLA episodes can be converted into a LeRobot-compatible dataset:

```bash
pip install -r requirements-export.txt

python scripts/export_lerobot.py \
  --episodes-dir data/turbopi_nav/episodes \
  --output-dir data/turbopi_nav/lerobot \
  --repo-id <HF_DATASET_REPO>
```

Important exporter behavior:

- it scans every accepted `episode_*` folder under `episodes/`
- it can also scan one specific `session_YYYYMMDD_HHMMSS/` folder
- it verifies that `data.parquet` rows match decoded video frames
- it rebuilds `observation.state` from the previous action by default
- it writes a standard LeRobot dataset with `observation.images.front`, `action`, `task`, and optionally `observation.state`
- custom task strings are exported exactly as saved

If you want different state behavior:

- `--state-source shifted_action`: recommended default
- `--state-source recorded`: trust the saved `observation.state`
- `--state-source zeros`: use a zero vector for every frame
- `--state-source none`: export without `observation.state`

## Train the Intent-CNN or ACT-Intent Baseline

The task-conditioned CNN dataset is intentionally separate from VLA data:

```text
data/turbopi_intent_cnn/
```

Install the CNN extras:

```bash
pip install -r requirements-cnn.txt
```

Train:

```bash
python -m intent_cnn_policy.train \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --run-dir runs/intent_cnn_v1
```

Evaluate:

```bash
python -m intent_cnn_policy.eval \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --checkpoint <RUN_DIR>/checkpoints/best.pt
```

Drive:

```bash
python -m intent_cnn_policy.drive \
  --robot-ip <ROBOT_IP> \
  --checkpoint <RUN_DIR>/checkpoints/best.pt \
  --task "go right"
```

If you trained on a custom task, that exact string becomes a valid inference label for that checkpoint.

Train ACT-Intent from the same dataset:

```bash
python -m act_intent_policy.train \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --run-dir runs/act_intent_v1
```

For repeated experiments on a strong GPU, build the ACT cache once and train from it:

```bash
python -m act_intent_policy.cache \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --cache-dir data/turbopi_intent_cnn/act_cache_w160_h120_hist3_chunk8 \
  --workers 8

python -m act_intent_policy.train \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --cache-dir data/turbopi_intent_cnn/act_cache_w160_h120_hist3_chunk8 \
  --cache-mode require \
  --run-dir runs/act_intent_v1 \
  --device cuda \
  --batch-size 128 \
  --num-workers 8
```

Evaluate:

```bash
python -m act_intent_policy.eval \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --checkpoint <RUN_DIR>/checkpoints/best.pt
```

Drive:

```bash
python -m act_intent_policy.drive \
  --robot-ip <ROBOT_IP> \
  --checkpoint <RUN_DIR>/checkpoints/best.pt \
  --task "go right"
```

## Inspect Recorded Actions

If you want to verify what was captured:

```bash
python scripts/inspect_episode.py --episodes-dir data/turbopi_nav/episodes
```

Useful reports include:

- `action_vx`, `action_vy`, `action_omega`
- `state_vx`, `state_vy`, `state_omega`
- counts for left, right, rotate-left, rotate-right, and stop frames
- whether `state` looks like a shifted version of `action`

Optional CSV:

```bash
python scripts/inspect_episode.py \
  --episodes-dir data/turbopi_nav/episodes \
  --csv data/turbopi_nav/inspection.csv
```

## Practical Tips

- Start with a lower teleop speed until the controls feel natural.
- Charge the battery before long recording sessions.
- Run the client on the local laptop session, not inside a remote SSH shell, because `pynput` listens for local keyboard events.
- If a Wi-Fi hiccup causes motor commands to fail, the recorder skips those frames instead of silently saving bad labels.
