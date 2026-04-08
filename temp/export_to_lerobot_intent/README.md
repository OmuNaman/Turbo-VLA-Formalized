# Export To LeRobot: Intent-CNN Session

This temp note captures the current experiment for turning the language-conditioned TurboPi dataset into a LeRobot dataset so we can try a VLA-style policy such as SmolVLA.

## What We Checked

- Source session: `data/turbopi_intent_cnn/episodes/session_20260408_105311`
- Accepted episodes: `50`
- Total frames: `6668`
- Task mix:
  - `go left`: `25` episodes, `3285` frames
  - `go right`: `25` episodes, `3383` frames

## Important Result

The source dataset already contains language task labels, so **no copy or patch of the original dataset was needed**.

We confirmed:

- `tasks.json` contains the task vocabulary:
  - `go left`
  - `go right`
  - `go forward`
  - `go backward`
- Each frame row in `data.parquet` already contains:
  - `task_index`
  - `task`
  - `observation.state`
  - `action`

So the session is already suitable for exporting to LeRobot format.

## Temp Export We Ran

We exported the session locally to:

- `temp/intent_cnn_lerobot_export`

Command used:

```powershell
python scripts\export_lerobot.py --episodes-dir data\turbopi_intent_cnn\episodes\session_20260408_105311 --output-dir temp\intent_cnn_lerobot_export --state-source shifted_action --overwrite
```

Export summary:

- repo_id: `local/episodes_lerobot`
- episodes: `50`
- frames: `6668`
- image key: `observation.images.front`
- state source: `shifted_action`

## Local Reload Check

We successfully reloaded the exported dataset with LeRobot and verified:

- `num_episodes = 50`
- `num_frames = 6668`
- sample keys include:
  - `observation.images.front`
  - `observation.state`
  - `action`
  - `task`
  - `task_index`

That means the exported LeRobot dataset really does preserve the language instruction field.

## Why This Is Promising

SmolVLA expects:

- camera image input
- robot state input
- natural language instruction

Our exported dataset now has:

- one camera stream: `observation.images.front`
- 3D state: `observation.state`
- natural language task: `task`
- 3D action target: `action`

So this is a reasonable first attempt at a lightweight VLA-style fine-tune.

## RunPod Plan

On RunPod:

1. Clone this repo.
2. Download the original uploaded session from HF.
3. Re-run the export there or upload `temp/intent_cnn_lerobot_export` later if you want the fully converted dataset on the Hub.
4. Install LeRobot with SmolVLA support.
5. Fine-tune `lerobot/smolvla_base`.

## Candidate RunPod Commands

These are the commands we should try next on RunPod.

Clone repo:

```bash
git clone https://github.com/OmuNaman/Turbo-VLA-Formalized.git
cd Turbo-VLA-Formalized
```

Install repo deps:

```bash
pip install -r requirements-cnn.txt
```

Download the original session from the Hub:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="omunaman/session_20260408_105311",
    repo_type="dataset",
    local_dir="data/hf_session_20260408_105311",
)
print(path)
PY
```

Export to LeRobot on RunPod:

```bash
python scripts/export_lerobot.py \
  --episodes-dir data/hf_session_20260408_105311/episodes \
  --output-dir temp/intent_cnn_lerobot_export \
  --state-source shifted_action \
  --overwrite
```

Install LeRobot with SmolVLA support:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
cd ..
```

Candidate fine-tune command:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/episodes_lerobot \
  --dataset.root=./temp/intent_cnn_lerobot_export \
  --batch_size=32 \
  --steps=20000 \
  --output_dir=outputs/train/turbopi_smolvla_intent \
  --job_name=turbopi_smolvla_intent \
  --policy.device=cuda
```

## Notes

- The main SmolVLA docs show the standard training flow with `lerobot-train` and `--dataset.repo_id` using a LeRobot dataset.
- The community dataset examples show a local dataset workflow using `--dataset.root`.
- I am inferring that this TurboPi mobile-robot dataset is worth trying with SmolVLA because the LeRobot export reloads correctly and preserves `task`.
- This is still more experimental than the custom `intent_cnn_policy` we already proved on the real robot.

## Sources

- SmolVLA docs: https://huggingface.co/docs/lerobot/en/smolvla
- LeRobot dataset v3 docs: https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
- Community dataset example showing `--dataset.root`: https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v1
