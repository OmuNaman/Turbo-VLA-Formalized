"""Helpers for resuming recording into an existing session folder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SessionResumeState:
    """Summary of accepted episodes already saved for a session."""

    next_episode_index: int = 0
    accepted_count: int = 0
    total_frames: int = 0


def _inspect_raw_max_episode_index(raw_dir: Path | None) -> int:
    """Return the highest episode_idx seen in raw telemetry, or -1 if unavailable."""
    if raw_dir is None:
        return -1
    telemetry_path = Path(raw_dir) / "telemetry.jsonl"
    if not telemetry_path.exists():
        return -1

    max_episode_index = -1
    with telemetry_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            episode_idx = payload.get("episode_idx")
            if episode_idx is None:
                continue
            try:
                max_episode_index = max(max_episode_index, int(episode_idx))
            except (TypeError, ValueError):
                continue
    return max_episode_index


def inspect_saved_session(episodes_dir: Path, raw_dir: Path | None = None) -> SessionResumeState:
    """Inspect an accepted-episodes folder and derive resume counters."""
    episode_dirs = sorted(path for path in episodes_dir.glob("episode_*") if path.is_dir())
    total_frames = 0
    max_episode_index = -1
    accepted_count = 0

    for episode_dir in episode_dirs:
        try:
            episode_index = int(episode_dir.name.split("_")[-1])
        except ValueError:
            continue

        max_episode_index = max(max_episode_index, episode_index)
        parquet_path = episode_dir / "data.parquet"
        video_path = episode_dir / "video.mp4"
        info_path = episode_dir / "episode_info.json"
        if not parquet_path.exists() or not video_path.exists() or not info_path.exists():
            continue

        try:
            total_frames += len(pd.read_parquet(parquet_path, columns=["frame_index"]))
            accepted_count += 1
        except Exception:
            pass

    max_episode_index = max(max_episode_index, _inspect_raw_max_episode_index(raw_dir))

    return SessionResumeState(
        next_episode_index=max_episode_index + 1,
        accepted_count=accepted_count,
        total_frames=total_frames,
    )
