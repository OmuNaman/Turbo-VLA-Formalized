"""Chunked dataset utilities for the ACT-style task-conditioned policy."""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    import av
except ImportError as exc:  # pragma: no cover - environment specific
    raise RuntimeError("PyAV is required for ACT dataset loading. Install with `pip install av`.") from exc

from intent_cnn_policy.dataset import (
    EpisodeRecord,
    discover_intent_episodes,
    discover_task_names,
    split_sessions,
)


@dataclass(frozen=True)
class SampleIndex:
    """Address one chunk start within one episode."""

    episode_idx: int
    frame_idx: int


def build_action_chunk(
    actions: np.ndarray,
    *,
    start_index: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed-length future action chunk plus validity mask."""
    if actions.ndim != 2:
        raise ValueError(f"Expected action array shaped [T,A], got {actions.shape}")
    action_dim = actions.shape[1]
    chunk = np.zeros((chunk_size, action_dim), dtype=np.float32)
    mask = np.zeros(chunk_size, dtype=np.float32)
    if len(actions) == 0:
        return chunk, mask

    end_index = min(len(actions), start_index + chunk_size)
    available = max(0, end_index - start_index)
    if available > 0:
        chunk[:available] = actions[start_index:end_index]
        mask[:available] = 1.0
    if available < chunk_size:
        chunk[available:] = actions[end_index - 1 if available > 0 else -1]
    return chunk, mask


class _EpisodeCache:
    """Small LRU cache for decoded and resized ACT episodes."""

    def __init__(self, image_size: tuple[int, int], max_items: int = 4):
        self.image_size = image_size
        self.max_items = max_items
        self._frames: OrderedDict[Path, list[np.ndarray]] = OrderedDict()
        self._actions: OrderedDict[Path, np.ndarray] = OrderedDict()

    def get(self, record: EpisodeRecord) -> tuple[list[np.ndarray], np.ndarray]:
        key = record.episode_dir
        if key in self._frames and key in self._actions:
            self._frames.move_to_end(key)
            self._actions.move_to_end(key)
            return self._frames[key], self._actions[key]

        frames = self._load_frames(record.episode_dir / "video.mp4")
        actions = self._load_actions(record.episode_dir / "data.parquet")
        if len(frames) != len(actions):
            raise ValueError(
                f"Episode {record.episode_dir} has {len(frames)} decoded frames but {len(actions)} action rows."
            )

        self._frames[key] = frames
        self._actions[key] = actions
        if len(self._frames) > self.max_items:
            self._frames.popitem(last=False)
            self._actions.popitem(last=False)
        return frames, actions

    def _load_frames(self, video_path: Path) -> list[np.ndarray]:
        width, height = self.image_size
        decoded: list[np.ndarray] = []
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                image = image.resize((width, height), Image.Resampling.BILINEAR)
                decoded.append(np.asarray(image, dtype=np.uint8))
        return decoded

    def _load_actions(self, parquet_path: Path) -> np.ndarray:
        df = pd.read_parquet(parquet_path, columns=["action"])
        return np.asarray(df["action"].tolist(), dtype=np.float32)


class ActEpisodeDataset(Dataset):
    """Stack recent frames and predict a chunk of future normalized actions."""

    def __init__(
        self,
        records: list[EpisodeRecord],
        task_names: list[str],
        *,
        split: str,
        image_size: tuple[int, int],
        history: int,
        chunk_size: int,
        augment: bool = False,
        cache_size: int = 4,
    ):
        self.records = list(records)
        self.task_names = list(task_names)
        self.split = split
        self.image_size = image_size
        self.history = history
        self.chunk_size = chunk_size
        self.augment = augment
        self.task_to_index = {task: index for index, task in enumerate(self.task_names)}
        if not self.task_to_index:
            raise ValueError("task_names must not be empty")

        self.samples: list[SampleIndex] = []
        self.sample_weights: list[float] = []
        self.task_counts: dict[str, int] = {task: 0 for task in self.task_names}
        self._episode_actions: list[np.ndarray] = []

        for record in self.records:
            if record.task not in self.task_to_index:
                raise ValueError(
                    f"Task '{record.task}' from {record.episode_dir} is not present in the provided task vocabulary."
                )
            actions = np.asarray(
                pd.read_parquet(record.episode_dir / "data.parquet", columns=["action"])["action"].tolist(),
                dtype=np.float32,
            )
            self._episode_actions.append(actions)
            self.task_counts[record.task] = self.task_counts.get(record.task, 0) + len(actions)

        for episode_idx, (record, actions) in enumerate(zip(self.records, self._episode_actions, strict=False)):
            for frame_idx, action in enumerate(actions):
                self.samples.append(SampleIndex(episode_idx=episode_idx, frame_idx=frame_idx))
                self.sample_weights.append(self._compute_sample_weight(action, record.task))

        effective_cache_size = cache_size
        if self.records and len(self.records) <= 64:
            effective_cache_size = max(cache_size, len(self.records))
        self.cache = _EpisodeCache(image_size=image_size, max_items=effective_cache_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        record = self.records[sample.episode_idx]
        frames, actions = self.cache.get(record)

        frame_indices = [max(0, sample.frame_idx - offset) for offset in reversed(range(self.history))]
        selected = [frames[frame_idx] for frame_idx in frame_indices]
        stacked = self._apply_transforms(selected)
        chunk, mask = build_action_chunk(actions, start_index=sample.frame_idx, chunk_size=self.chunk_size)

        task_index = torch.tensor(self.task_to_index[record.task], dtype=torch.long)
        action_chunk = torch.as_tensor(chunk, dtype=torch.float32)
        action_mask = torch.as_tensor(mask, dtype=torch.float32)
        first_action = torch.as_tensor(chunk[0], dtype=torch.float32)

        return {
            "image": stacked,
            "task_index": task_index,
            "action_chunk": action_chunk,
            "action_mask": action_mask,
            "first_action": first_action,
            "task": record.task,
            "session_name": record.session_name,
        }

    def _apply_transforms(self, frames: list[np.ndarray]) -> torch.Tensor:
        pil_frames = [Image.fromarray(frame) for frame in frames]
        if self.augment:
            pil_frames = self._augment_frames(pil_frames)
        tensors = [TF.to_tensor(frame) for frame in pil_frames]
        return torch.cat(tensors, dim=0)

    def _augment_frames(self, frames: list[Image.Image]) -> list[Image.Image]:
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        saturation = random.uniform(0.9, 1.1)
        hue = random.uniform(-0.03, 0.03)
        angle = random.uniform(-5.0, 5.0)
        translate_x = int(round(random.uniform(-0.05, 0.05) * self.image_size[0]))
        translate_y = int(round(random.uniform(-0.05, 0.05) * self.image_size[1]))
        do_blur = random.random() < 0.2
        blur_sigma = random.uniform(0.1, 1.0)

        augmented: list[Image.Image] = []
        for frame in frames:
            frame = TF.adjust_brightness(frame, brightness)
            frame = TF.adjust_contrast(frame, contrast)
            frame = TF.adjust_saturation(frame, saturation)
            frame = TF.adjust_hue(frame, hue)
            frame = TF.affine(
                frame,
                angle=angle,
                translate=(translate_x, translate_y),
                scale=1.0,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            if do_blur:
                frame = TF.gaussian_blur(frame, kernel_size=3, sigma=blur_sigma)
            augmented.append(frame)
        return augmented

    def _compute_sample_weight(self, action: np.ndarray, task: str) -> float:
        vx, vy, omega = np.asarray(action, dtype=np.float32).tolist()
        magnitude = max(abs(vx), abs(vy), abs(omega))
        if magnitude < 0.05:
            base = 0.3
        elif abs(vy) > 0.05 or abs(omega) > 0.05:
            base = 1.3
        else:
            base = 0.9

        task_count = max(1, self.task_counts.get(task, 1))
        task_balance = len(self.task_names) / task_count
        return base * task_balance

    @property
    def total_frames(self) -> int:
        return sum(record.num_frames for record in self.records)

    @property
    def estimated_cache_bytes(self) -> int:
        width, height = self.image_size
        return self.total_frames * width * height * 3

    def preload_all(
        self,
        *,
        show_progress: bool = True,
        desc: str | None = None,
    ) -> None:
        records = self.records
        if show_progress:
            records = tqdm(
                self.records,
                total=len(self.records),
                desc=desc or f"Preload {self.split}",
                unit="episode",
                dynamic_ncols=True,
                leave=False,
            )
        for record in records:
            self.cache.get(record)


def build_datasets(
    episodes_dir: Path | str,
    *,
    image_size: tuple[int, int] = (160, 120),
    history: int = 3,
    chunk_size: int = 8,
    val_ratio: float = 0.2,
    seed: int | None = None,
    task_names: list[str] | None = None,
    cache_size: int = 4,
) -> tuple[ActEpisodeDataset, ActEpisodeDataset, list[str]]:
    """Create train/val ACT-style datasets with a shared task vocabulary."""
    episodes_dir = Path(episodes_dir)
    all_records = discover_intent_episodes(episodes_dir)
    if task_names is None:
        task_names = discover_task_names(episodes_dir, all_records)
    else:
        task_names = list(task_names)

    train_records = split_sessions(all_records, split="train", val_ratio=val_ratio, seed=seed)
    val_records = split_sessions(all_records, split="val", val_ratio=val_ratio, seed=seed)

    train_dataset = ActEpisodeDataset(
        train_records,
        task_names,
        split="train",
        image_size=image_size,
        history=history,
        chunk_size=chunk_size,
        augment=True,
        cache_size=cache_size,
    )
    val_dataset = ActEpisodeDataset(
        val_records,
        task_names,
        split="val",
        image_size=image_size,
        history=history,
        chunk_size=chunk_size,
        augment=False,
        cache_size=cache_size,
    )
    return train_dataset, val_dataset, task_names
