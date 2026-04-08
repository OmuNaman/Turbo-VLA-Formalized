"""Episode buffer management: start, record frames, accept, discard."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeFrame:
    """Single frame of recording data."""

    timestamp: float
    image: np.ndarray
    state: np.ndarray
    action: np.ndarray


@dataclass
class EpisodeBuffer:
    """In-memory buffer for one episode."""

    episode_index: int
    task: str
    task_index: int
    frames: list[EpisodeFrame] = field(default_factory=list)
    start_time: float = 0.0
    status: str = "recording"


class EpisodeManager:
    """Manages episode lifecycle: start -> record frames -> accept/discard."""

    def __init__(
        self,
        start_episode_index: int = 0,
        accepted_count: int = 0,
        total_frames: int = 0,
    ):
        self.current: EpisodeBuffer | None = None
        self.episode_count: int = start_episode_index
        self.accepted_count: int = accepted_count
        self.total_frames: int = total_frames

    def start_episode(self, task: str, task_index: int) -> None:
        """Begin a new episode."""
        self.current = EpisodeBuffer(
            episode_index=self.episode_count,
            task=task,
            task_index=task_index,
            start_time=time.monotonic(),
        )

    def add_frame(
        self,
        image: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        timestamp: float,
    ) -> None:
        """Add a frame to the current episode buffer."""
        if self.current is None:
            raise RuntimeError("No active episode. Call start_episode() first.")

        self.current.frames.append(
            EpisodeFrame(
                timestamp=timestamp,
                image=image,
                state=state.copy(),
                action=action.copy(),
            )
        )

    def accept_episode(self) -> EpisodeBuffer:
        """Accept the current episode and return it for saving."""
        if self.current is None:
            raise RuntimeError("No active episode to accept.")

        self.current.status = "accepted"
        episode = self.current
        self.current = None
        self.episode_count += 1
        self.accepted_count += 1
        self.total_frames += len(episode.frames)
        return episode

    def discard_episode(self) -> None:
        """Discard the current episode, freeing memory."""
        if self.current is None:
            return

        self.current.status = "discarded"
        self.current = None
        self.episode_count += 1

    @property
    def is_recording(self) -> bool:
        return self.current is not None and self.current.status == "recording"

    @property
    def current_frame_count(self) -> int:
        if self.current is None:
            return 0
        return len(self.current.frames)

    @property
    def current_duration(self) -> float:
        if self.current is None:
            return 0.0
        return time.monotonic() - self.current.start_time
