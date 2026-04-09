"""Merge multiple recorded sessions into one normalized intent-conditioned session."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge recorded sessions into one normalized intent-conditioned session")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Session directories containing episode_* folders",
    )
    parser.add_argument(
        "--output-root",
        default="data/turbopi_intent_cnn/episodes",
        help="Parent directory where the merged session folder will be created",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Optional explicit merged session folder name; defaults to session_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--task-map",
        default=None,
        help='JSON mapping of source task strings to normalized output task strings, for example \'{"go to the left of the box":"go left"}\'',
    )
    return parser


def load_task_map(raw: str | None) -> dict[str, str]:
    if raw is None:
        return {}
    candidate = Path(raw)
    if candidate.exists():
        mapping = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("--task-map file must decode to a JSON object")
        return {str(key): str(value) for key, value in mapping.items()}
    try:
        mapping = json.loads(raw)
        if not isinstance(mapping, dict):
            raise ValueError("--task-map must decode to a JSON object")
        return {str(key): str(value) for key, value in mapping.items()}
    except json.JSONDecodeError:
        result: dict[str, str] = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    "--task-map must be JSON, a JSON file path, or comma-separated src=dst pairs"
                )
            src, dst = item.split("=", 1)
            result[src.strip()] = dst.strip()
        return result


def default_session_name() -> str:
    return "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def discover_episode_dirs(session_dir: Path) -> list[Path]:
    return [path for path in sorted(session_dir.glob("episode_*")) if path.is_dir()]


def build_session_info(
    *,
    session_name: str,
    tasks: list[str],
    sources: list[Path],
    fps: int,
    episode_time_s: float,
    max_duty: float,
    teleop_speed: float,
    robot_url: str,
    vcodec: str,
) -> dict[str, object]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "last_updated_at": datetime.now().isoformat(timespec="seconds"),
        "session_name": session_name,
        "dataset_name": "turbopi_intent_cnn",
        "robot_url": robot_url,
        "robot_type": "turbopi",
        "fps": fps,
        "episode_time_s": episode_time_s,
        "max_duty": max_duty,
        "teleop_speed": teleop_speed,
        "vcodec": vcodec,
        "tasks": tasks,
        "mode_family": "cnn",
        "intent_mode": "language",
        "task_type": "instruction_conditioned_path_following",
        "episode_definition": "one_instruction_one_demo_manual_accept",
        "collection_style": "clean_demo",
        "observation_state_semantics": "previous_action_normalized",
        "action_semantics": "current_action_normalized",
        "accepted_episode_timestamps": "episode_relative_seconds",
        "raw_backup_timestamps": "robot_monotonic_seconds",
        "source_sessions": [str(path) for path in sources],
    }


def main() -> None:
    args = build_parser().parse_args()
    source_dirs = [Path(path) for path in args.sources]
    for source in source_dirs:
        if not source.exists():
            raise FileNotFoundError(f"Source session does not exist: {source}")

    task_map = load_task_map(args.task_map)
    output_root = Path(args.output_root)
    session_name = args.session_name or default_session_name()
    output_dir = output_root / session_name
    if output_dir.exists():
        raise FileExistsError(f"Output session already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    normalized_tasks: list[str] = []
    task_to_index: dict[str, int] = {}

    episode_specs: list[tuple[Path, str]] = []
    for source_dir in source_dirs:
        for episode_dir in discover_episode_dirs(source_dir):
            parquet_path = episode_dir / "data.parquet"
            video_path = episode_dir / "video.mp4"
            if not parquet_path.exists() or not video_path.exists():
                continue
            df = pd.read_parquet(parquet_path, columns=["task"])
            if df.empty:
                continue
            raw_task = str(df["task"].iloc[0])
            normalized_task = task_map.get(raw_task, raw_task)
            if normalized_task not in task_to_index:
                task_to_index[normalized_task] = len(normalized_tasks)
                normalized_tasks.append(normalized_task)
            episode_specs.append((episode_dir, normalized_task))

    if not episode_specs:
        raise RuntimeError("No valid episodes were found in the requested source sessions.")

    first_session_info = {}
    for source_dir in source_dirs:
        session_info_path = source_dir / "session_info.json"
        if session_info_path.exists():
            first_session_info = json.loads(session_info_path.read_text(encoding="utf-8"))
            break

    fps = int(first_session_info.get("fps", 10))
    episode_time_s = float(first_session_info.get("episode_time_s", 60.0))
    max_duty = float(first_session_info.get("max_duty", 80.0))
    teleop_speed = float(first_session_info.get("teleop_speed", 50.0))
    robot_url = str(first_session_info.get("robot_url", "http://192.168.149.1:8080"))
    vcodec = str(first_session_info.get("vcodec", "h264"))

    for merged_index, (source_episode_dir, normalized_task) in enumerate(episode_specs):
        target_episode_dir = output_dir / f"episode_{merged_index:06d}"
        target_episode_dir.mkdir(parents=True, exist_ok=False)

        df = pd.read_parquet(source_episode_dir / "data.parquet")
        df["episode_index"] = merged_index
        df["task"] = normalized_task
        df["task_index"] = task_to_index[normalized_task]
        df.to_parquet(target_episode_dir / "data.parquet", index=False)

        shutil.copy2(source_episode_dir / "video.mp4", target_episode_dir / "video.mp4")

        duration_s = float(len(df) / fps) if fps > 0 else 0.0
        episode_info = {
            "episode_index": merged_index,
            "mode_family": "cnn",
            "intent_mode": "language",
            "task_type": "instruction_conditioned_path_following",
            "episode_definition": "one_instruction_one_demo_manual_accept",
            "collection_style": "clean_demo",
            "task_name": normalized_task,
            "task_index": task_to_index[normalized_task],
            "num_frames": int(len(df)),
            "duration_s": duration_s,
            "source_episode_dir": str(source_episode_dir),
        }
        (target_episode_dir / "episode_info.json").write_text(
            json.dumps(episode_info, indent=2),
            encoding="utf-8",
        )

    tasks_json = {str(index): task for task, index in task_to_index.items()}
    (output_dir / "tasks.json").write_text(json.dumps(tasks_json, indent=2), encoding="utf-8")
    session_info = build_session_info(
        session_name=session_name,
        tasks=normalized_tasks,
        sources=source_dirs,
        fps=fps,
        episode_time_s=episode_time_s,
        max_duty=max_duty,
        teleop_speed=teleop_speed,
        robot_url=robot_url,
        vcodec=vcodec,
    )
    (output_dir / "session_info.json").write_text(json.dumps(session_info, indent=2), encoding="utf-8")

    counts = Counter(task for _, task in episode_specs)
    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "episodes": len(episode_specs),
            "task_names": normalized_tasks,
            "task_episode_counts": counts,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
