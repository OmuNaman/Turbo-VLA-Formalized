# Getting Started

This guide gets a student from fresh clone to either:

- a trained `intent_cnn_policy` checkpoint
- or an exported LeRobot dataset ready for SmolVLA fine-tuning

## What You Need

- A TurboPi Advanced Kit with the vendor image installed
- A Windows, macOS, or Linux laptop with Python 3.10 or newer
- SSH access to the robot
- A Wi-Fi network that both the robot and the laptop can join

## Install Laptop Dependencies

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

## Install Robot Dependencies

On the TurboPi:

```bash
python3 -m pip install -r requirements-robot.txt
```

Important:

- `ros_robot_controller_sdk` is not installed from `pip`
- `cv2` is usually already present on the TurboPi image
- this repo assumes the normal TurboPi software image, not a clean Raspberry Pi install

From the laptop instead:

```bash
bash scripts/deploy_server.sh deps
```

## First Run Order

1. Follow [Wi-Fi and SSH Guide](wifi-and-ssh.md) to move the robot onto the same Wi-Fi as your laptop.
2. Install the robot-side Python packages if needed.
3. Start the robot server.
4. Optionally test teleop-only mode.
5. Start the laptop launcher.
6. Choose either the intent-conditioned CNN workflow or the VLA recording workflow.
7. Record accepted episodes.
8. Train `intent_cnn_policy` or export VLA episodes to LeRobot.

## Start the Robot Server

If the repo is already on the robot:

```bash
python3 robot_server/server.py --port 8080
```

If the repo only exists on your laptop:

```bash
bash scripts/deploy_server.sh start
```

Windows users can run that from Git Bash or WSL.

## Start the Laptop Client

Quick teleop-only test:

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

`python -m client` uses the default AP-mode robot IP `192.168.149.1`. Once you move to shared Wi-Fi, pass `--robot-ip <ROBOT_IP>` explicitly.

The launcher then offers:

- `CNN-based`
- `VLA-based`

Inside `CNN-based`, choose:

- `intent-conditioned (recommended)`

## Recording Tasks

The VLA recorder and the intent-conditioned CNN recorder both support:

- the built-in task list
- one extra menu item: `Custom task...`

If a student chooses `Custom task...`, the typed task:

- is appended to the current session task list
- gets a session-local `task_index`
- is saved into `tasks.json`, `session_info.json`, `episode_info.json`, `data.parquet`, and raw telemetry

That means students can mix built-in tasks and one-off tasks without changing `tasks.py` first.

## Where Recordings Are Saved

The client writes new timestamped folders under:

```text
data/<dataset_name>/
|-- raw/
`-- episodes/
```

Example:

```text
data/turbopi_intent_cnn/
|-- raw/session_20260401_101500/
`-- episodes/session_20260401_101500/
```

Each run gets its own `session_YYYYMMDD_HHMMSS` folder, so old recordings stay untouched.

## Train the Intent-CNN Policy

Install the CNN training extras on the laptop:

```bash
pip install -r requirements-cnn.txt
```

Train from the intent-conditioned episodes root:

```bash
python -m intent_cnn_policy.train \
  --episodes-dir data/turbopi_intent_cnn/episodes \
  --run-dir runs/intent_cnn_v1
```

The `--run-dir` value is a base directory. Each training launch creates a fresh timestamped child run.

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

Important:

- the checkpoint only knows tasks that appeared in its training data
- if you record custom tasks, those custom strings become valid inference labels for that checkpoint

## Export to LeRobot

Install the export extras:

```bash
pip install -r requirements-export.txt
```

Then run:

```bash
python scripts/export_lerobot.py \
  --episodes-dir data/turbopi_nav/episodes \
  --output-dir data/turbopi_nav/lerobot \
  --repo-id <HF_DATASET_REPO>
```

The default `--state-source shifted_action` reconstructs `observation.state` from the previous action and is the safest choice for older recordings.

Helpful notes:

- `--episodes-dir` can be the full episodes root or one specific `session_YYYYMMDD_HHMMSS` folder
- the exporter includes every accepted `episode_*` folder under that path
- if one episode is clearly bad, delete that episode folder before exporting
- LeRobot often stores exported frames as chunked dataset videos rather than one MP4 per episode

## Useful Flags

Shorter collection run:

```bash
python -m client.cli --robot-ip <ROBOT_IP> --episodes 10 --episode-time 20
```

Different VLA dataset name:

```bash
python -m client.cli --robot-ip <ROBOT_IP> --dataset classroom_nav
```

Different intent-CNN dataset name:

```bash
python -m client.cli --robot-ip <ROBOT_IP> --intent-cnn-dataset classroom_intent_cnn
```

Show export options:

```bash
python scripts/export_lerobot.py --help
```

## Change the Built-In Tasks

The built-in task list lives in `tasks.py`. Edit `DEFAULT_TASKS` if you want a different classroom default, but students can now also create one-off tasks during recording without modifying the code first.
