#!/usr/bin/env python3
"""Export accepted TurboPi episodes to a LeRobot-compatible dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import ExportConfig
from storage.lerobot_exporter import (
    STATE_SOURCE_CHOICES,
    export_lerobot_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    defaults = ExportConfig()

    parser = argparse.ArgumentParser(description="Export TurboPi episodes to LeRobot format")
    parser.add_argument("--episodes-dir", default=str(defaults.episodes_dir))
    parser.add_argument("--output-dir", default=str(defaults.output_dir))
    parser.add_argument("--repo-id", default=defaults.repo_id)
    parser.add_argument("--robot-type", default=defaults.robot_type)
    parser.add_argument("--fps", type=int, default=defaults.fps)
    parser.add_argument("--image-key", default=defaults.image_key)
    parser.add_argument(
        "--state-source",
        choices=sorted(STATE_SOURCE_CHOICES),
        default=defaults.state_source,
        help=(
            "How to populate observation.state in the exported dataset. "
            "'shifted_action' is the safest default for older sessions; "
            "'recorded' now rejects obviously leaky same-step state labels."
        ),
    )
    parser.add_argument("--vcodec", default=defaults.vcodec)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser


def derive_repo_id(episodes_dir: Path, output_dir: Path, requested_repo_id: str) -> str:
    """Choose a reasonable local repo id when the placeholder was not replaced."""
    if requested_repo_id != "<HF_DATASET_REPO>":
        return requested_repo_id

    if episodes_dir.name.startswith("session_"):
        dataset_name = episodes_dir.parent.parent.name or output_dir.parent.name or "turbopi"
        return f"local/{dataset_name}_{episodes_dir.name}_lerobot"

    dataset_name = episodes_dir.parent.name or output_dir.parent.name or "turbopi"
    return f"local/{dataset_name}_lerobot"


def main() -> None:
    """Run the export."""
    parser = build_parser()
    args = parser.parse_args()

    episodes_dir = Path(args.episodes_dir)
    output_dir = Path(args.output_dir)
    repo_id = derive_repo_id(episodes_dir, output_dir, args.repo_id)
    if args.push_to_hub and repo_id.startswith("local/"):
        raise ValueError(
            "--push-to-hub requires a real Hugging Face dataset repo id, for example "
            "'your-name/your-dataset'. Replace the placeholder --repo-id before pushing."
        )

    summary = export_lerobot_dataset(
        episodes_dir=episodes_dir,
        output_dir=output_dir,
        repo_id=repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        image_key=args.image_key,
        state_source=args.state_source,
        vcodec=args.vcodec,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
    )

    print()
    print("LeRobot export complete")
    print(f"  repo_id:       {summary.repo_id}")
    print(f"  output_dir:    {summary.output_dir}")
    print(f"  episodes:      {summary.num_episodes}")
    print(f"  frames:        {summary.num_frames}")
    print(f"  image_key:     {summary.image_key}")
    print(f"  state_source:  {summary.state_source}")
    print()


if __name__ == "__main__":
    main()
