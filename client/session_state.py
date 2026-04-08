"""Helpers for resuming recording into an existing session folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SessionResumeState:
    """Summary of accepted episodes already saved for a session."""

    next_episode_index: int = 0
    accepted_count: int = 0
    total_frames: int = 0


def inspect_saved_session(episodes_dir: Path) -> SessionResumeState:
    """Inspect an accepted-episodes folder and derive resume counters."""
    episode_dirs = sorted(path for path in episodes_dir.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        return SessionResumeState()

    total_frames = 0
    max_episode_index = -1

    for episode_dir in episode_dirs:
        try:
            episode_index = int(episode_dir.name.split("_")[-1])
        except ValueError:
            continue

        max_episode_index = max(max_episode_index, episode_index)
        parquet_path = episode_dir / "data.parquet"
        if not parquet_path.exists():
            continue

        try:
            total_frames += len(pd.read_parquet(parquet_path, columns=["frame_index"]))
        except Exception:
            pass

    return SessionResumeState(
        next_episode_index=max_episode_index + 1,
        accepted_count=len(episode_dirs),
        total_frames=total_frames,
    )
