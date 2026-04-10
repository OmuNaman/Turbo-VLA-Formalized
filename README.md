# TurboPi VLA Formalized

TurboPi VLA Formalized is a student-friendly recording, training, export, and inference stack for the TurboPi Advanced Kit.

The repo is built around two main student workflows plus one experimental chunked-policy path:

- `Intent-CNN`: a lightweight task-conditioned CNN that learns from camera frames plus a saved task label
- `SmolVLA`: a LeRobot-compatible workflow that exports recorded episodes and drives the robot with a fine-tuned SmolVLA checkpoint
- `ACT-Intent` (experimental): a chunked action-transformer style policy that reuses the same task-conditioned recording dataset as Intent-CNN

The recording client keeps the familiar built-in task list, but now also includes a `Custom task...` option so students can add one-off task prompts during collection without changing code first.

## What This Repo Includes

- `robot_server/`: HTTP server for camera, velocity, and health endpoints on the TurboPi
- `client/`: laptop-side launcher, teleop loop, and recording sessions
- `intent_cnn_policy/`: train, evaluate, and drive the task-conditioned CNN
- `act_intent_policy/`: experimental ACT-style chunked task-conditioned policy
- `smolvla_policy/`: TurboPi inference adapter for fine-tuned SmolVLA checkpoints
- `storage/`: raw backup writer, accepted-episode writer, and LeRobot exporter
- `scripts/export_lerobot.py`: export accepted episodes into a LeRobot dataset
- `scripts/upload_hf_session.py`: upload one recorded session to Hugging Face
- `docs/`: student-facing setup and workflow guides
- `requirements-smolvla.txt`: local SmolVLA inference/runtime dependencies

Legacy note:

- `cnn_policy/` remains available for the older no-language loop-CNN workflow, but the official student path is now `intent_cnn_policy/`

## Official Student Workflow

### 1. Set up the laptop

Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-laptop.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-laptop.txt
```

### 2. Connect the robot and laptop to the same Wi-Fi

Use the robot hotspot only for first access. After that, move the robot onto the same Wi-Fi as your laptop so you can SSH normally and keep internet access on the laptop.

Hotspot first access:

```bash
ssh pi@192.168.149.1
```

Then on the robot:

```bash
nmcli dev wifi list
sudo nmcli device wifi connect "<SSID>" password "<PASSWORD>"
```

Reconnect to the robot on the shared network:

```bash
ssh pi@<ROBOT_IP>
```

### 3. Install robot-side dependencies

If the repo already exists on the robot:

```bash
python3 -m pip install -r requirements-robot.txt
python3 robot_server/server.py --port 8080
```

If the repo only exists on your laptop, use the helper script from Git Bash or WSL:

```bash
bash scripts/deploy_server.sh deps
bash scripts/deploy_server.sh start
```

### 4. Start the laptop client

Quick teleop smoke test:

```bash
python -m client.teleop --robot-ip <ROBOT_IP>
```

Start the launcher:

```bash
python -m client.cli --robot-ip <ROBOT_IP>
```

Hotspot-only quick test:

```bash
python -m client
```

The launcher offers:

- `CNN-based`
- `VLA-based`

Inside `CNN-based`, the recommended branch is:

- `intent-conditioned dataset (recommended)`

The older `no-language loop mode (legacy)` path still exists, but it is no longer the primary student workflow.

## Recording With Built-In Tasks Plus Custom Tasks

Both the VLA recorder and the intent-conditioned CNN recorder now behave like this:

- built-in tasks still appear as numbered options
- the last option is `Custom task...`
- if selected, the client prompts for a task string and adds it to the current session
- that typed task is saved just like any built-in task
- future episodes in the same session can reuse it directly from the numbered list
- VLA defaults come from `tasks.py::DEFAULT_TASKS` and Intent-CNN defaults come from `tasks.py::DEFAULT_INTENT_CNN_TASKS`

Saved session structure:

```text
data/<dataset_name>/
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

Important saved fields:

- `task`: exact task text chosen for the episode
- `task_index`: session-local index for that task
- `observation.state`: previous normalized action
- `action`: current normalized teleop command

## Intent-CNN Workflow

The official lightweight baseline is the task-conditioned CNN.

Recommended recording dataset root:

```text
data/turbopi_intent_cnn/
```

Record data:

```bash
python -m client.cli --robot-ip <ROBOT_IP>
```

Then choose:

```text
CNN-based
-> intent-conditioned dataset (recommended)
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
  --task "go left"
```

Important limitation:

- the intent-conditioned CNN is still a closed-set task-label model
- it can drive only with task strings that were present in the training data for that checkpoint

## ACT-Intent Workflow

`ACT-Intent` is an experimental chunked-action policy for students who want something more expressive than the tiny Intent-CNN baseline without jumping all the way to SmolVLA.

It reuses the exact same task-conditioned recording dataset:

```text
data/turbopi_intent_cnn/
```

Record data the same way:

```bash
python -m client.cli --robot-ip <ROBOT_IP>
```

Then choose:

```text
CNN-based
-> intent-conditioned dataset (recommended)
```

Train:

```bash
python -m act_intent_policy.train \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --run-dir runs/act_intent_v1
```

Recommended fast path for cloud GPUs:

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
  --task "go left"
```

Important notes:

- ACT-Intent is still closed-set with respect to task labels
- by default the driver replans from the latest frame every step
- `--reuse-action-queue` is available, but is usually worse for reactive path following
- raw ACT training works, but the cache-backed path is much faster on cloud GPUs because it avoids repeated MP4 decode and per-sample PIL transforms during training

## SmolVLA Workflow

The repo supports SmolVLA through:

- LeRobot export from recorded TurboPi episodes
- a TurboPi-specific `smolvla_policy` drive adapter

Recommended VLA recording dataset root:

```text
data/turbopi_nav/
```

Record task-conditioned episodes:

```bash
python -m client.cli --robot-ip <ROBOT_IP>
```

Then choose:

```text
VLA-based
```

Export to LeRobot:

```bash
pip install -r requirements-export.txt

python scripts/export_lerobot.py \
  --episodes-dir data/turbopi_nav/episodes \
  --output-dir data/turbopi_nav/lerobot \
  --state-source shifted_action \
  --overwrite
```

Fine-tuning itself runs through LeRobot, for example:

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/<YOUR_EXPORTED_REPO_ID> \
  --dataset.root=<YOUR_EXPORT_DIR> \
  --batch_size=32 \
  --steps=5000 \
  --output_dir=<RUN_DIR> \
  --job_name=turbopi_smolvla \
  --policy.device=cuda
```

Drive the robot from a fine-tuned checkpoint:

```bash
pip install -r requirements-smolvla.txt

python -m smolvla_policy.drive \
  --checkpoint <PRETRAINED_MODEL_DIR> \
  --task "go left" \
  --robot-ip <ROBOT_IP> \
  --device cuda
```

The SmolVLA driver replans from the latest camera frame each loop by default, which is better for line-following and reactive path correction than consuming a long cached action chunk open-loop.

If you trained with a non-default export state contract, keep runtime aligned:

- export `--state-source shifted_action` -> drive with default `--state-mode auto`
- export `--state-source zeros` -> drive with `--state-mode zeros`
- export `--state-source none` -> use a checkpoint that does not expect `observation.state`

## Utilities

Upload one recorded session to Hugging Face:

```bash
python scripts/upload_hf_session.py
```

By default the uploader scans the whole local `data/` tree, so it can find sessions under both `data/turbopi_intent_cnn/` and `data/turbopi_nav/`.

Inspect recorded episode contents:

```bash
python scripts/inspect_episode.py --episodes-dir data/turbopi_nav/episodes
```

## Docs

- [Getting Started](docs/getting-started.md)
- [Wi-Fi and SSH Guide](docs/wifi-and-ssh.md)
- [Data Collection Guide](docs/data-collection.md)
- [Intent-CNN Guide](docs/cnn.md)
- [ACT-Intent Guide](docs/act.md)
- [Troubleshooting](docs/troubleshooting.md)

## Repo Layout

```text
.
|-- client/
|-- intent_cnn_policy/
|-- act_intent_policy/
|-- smolvla_policy/
|-- cnn_policy/              # legacy no-language loop CNN
|-- loop_cnn/               # internal legacy implementation behind cnn_policy
|-- robot_server/
|-- scripts/
|-- storage/
|-- docs/
|-- requirements-laptop.txt
|-- requirements-robot.txt
|-- requirements-cnn.txt
|-- requirements-smolvla.txt
`-- requirements-export.txt
```

## Notes for Open-Source Users

- Runtime data is not tracked in this repo. Recording runs create timestamped folders under `data/<dataset_name>/`.
- Pass `--session-name <NAME>` if you want to resume/appended into an existing session folder instead of creating a fresh timestamped one.
- Accepted episodes are stored as one folder per episode with `video.mp4`, `data.parquet`, and `episode_info.json`.
- `observation.state` is saved as the previous normalized action to avoid target leakage during training.
- `ros_robot_controller_sdk` is TurboPi-specific and comes from the robot image, not from `pip`.
- Generated artifacts such as local checkpoints, exported datasets, copied pretrained models, and temp folders are intentionally ignored.

## References

- [Hiwonder TurboPi network setup](https://docs.hiwonder.com/projects/TurboPi/en/advanced/docs/7.network_configuration.html)
- [Hiwonder TurboPi getting ready](https://docs.hiwonder.com/projects/TurboPi/en/latest/docs/1.getting_ready.html)
- [Hugging Face LeRobot dataset docs](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3)
