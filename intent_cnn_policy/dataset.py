"""Dataset utilities for task-conditioned CNN training."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    import av
except ImportError as exc:  # pragma: no cover - environment specific
    raise RuntimeError("PyAV is required for CNN dataset loading. Install with `pip install av`.") from exc


@dataclass(frozen=True)
class EpisodeRecord:
    """Metadata about one accepted intent-conditioned episode."""

    episode_dir: Path
    session_name: str
    num_frames: int
    task: str
    task_index_hint: int | None = None


@dataclass(frozen=True)
class SampleIndex:
    """Address one target frame within one episode."""

    episode_idx: int
    frame_idx: int


def discover_session_dirs(episodes_root: Path | str) -> list[Path]:
    """Discover session directories beneath the episodes root."""
    root = Path(episodes_root)
    if not root.exists():
        return []
    session_dirs: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if any(child.is_dir() and child.name.startswith("episode_") for child in path.iterdir()):
            session_dirs.append(path)
    return session_dirs


def _read_session_task_names(session_dir: Path) -> list[str]:
    """Read the task ordering saved beside a recorded session."""
    tasks_path = session_dir / "tasks.json"
    if not tasks_path.exists():
        return []

    try:
        data = json.loads(tasks_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        items: list[tuple[int, str]] = []
        for key, value in data.items():
            try:
                index = int(key)
            except Exception:
                continue
            items.append((index, str(value)))
        return [value for _, value in sorted(items, key=lambda item: item[0])]
    return []


def discover_intent_episodes(episodes_dir: Path) -> list[EpisodeRecord]:
    """Discover saved intent-conditioned episodes and their tasks."""
    records: list[EpisodeRecord] = []
    for episode_dir in sorted(Path(episodes_dir).glob("**/episode_*")):
        if not episode_dir.is_dir():
            continue
        parquet_path = episode_dir / "data.parquet"
        video_path = episode_dir / "video.mp4"
        info_path = episode_dir / "episode_info.json"
        if not parquet_path.exists() or not video_path.exists() or not info_path.exists():
            continue

        df = pd.read_parquet(parquet_path, columns=["task"])
        if df.empty:
            continue

        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            info = {}

        task = str(df["task"].iloc[0])
        task_index_hint = info.get("task_index")
        if task_index_hint is None and "task_index" in df.columns:
            try:
                task_index_hint = int(df["task_index"].iloc[0])
            except Exception:
                task_index_hint = None
        try:
            task_index_hint = None if task_index_hint is None else int(task_index_hint)
        except Exception:
            task_index_hint = None

        records.append(
            EpisodeRecord(
                episode_dir=episode_dir,
                session_name=episode_dir.parent.name,
                num_frames=len(df),
                task=task,
                task_index_hint=task_index_hint,
            )
        )
    return records


def discover_task_names(episodes_dir: Path, records: list[EpisodeRecord]) -> list[str]:
    """Build a stable task vocabulary from session task files plus episode labels."""
    ordered: list[str] = []
    seen: set[str] = set()

    for session_dir in discover_session_dirs(episodes_dir):
        for task in _read_session_task_names(session_dir):
            if task not in seen:
                seen.add(task)
                ordered.append(task)

    remaining = sorted({record.task for record in records if record.task not in seen})
    ordered.extend(remaining)
    return ordered


def split_sessions(
    records: list[EpisodeRecord],
    split: str,
    val_ratio: float = 0.2,
    seed: int | None = None,
) -> list[EpisodeRecord]:
    """Split at the session level to avoid frame leakage."""
    if split not in {"train", "val", "all"}:
        raise ValueError(f"Unsupported split: {split}")
    if split == "all":
        return list(records)

    sessions = sorted({record.session_name for record in records})
    if len(sessions) <= 1:
        return list(records) if split == "train" else []

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(sessions)

    val_count = max(1, int(round(len(sessions) * val_ratio)))
    val_sessions = set(sessions[-val_count:])
    if split == "train":
        return [record for record in records if record.session_name not in val_sessions]
    return [record for record in records if record.session_name in val_sessions]


class _EpisodeCache:
    """LRU cache for decoded and resized episode frames."""

    def __init__(self, image_size: tuple[int, int], max_items: int = 4):
        self.image_size = image_size
        self.max_items = max_items
        self._frames: OrderedDict[Path, list[np.ndarray]] = OrderedDict()
        self._actions: OrderedDict[Path, np.ndarray] = OrderedDict()

    def get(self, record: EpisodeRecord) -> tuple[list[np.ndarray], np.ndarray]:
        """Return resized RGB frames and action array for one episode."""
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
        df = pd.read_parquet(parquet_path)
        return np.asarray(df["action"].tolist(), dtype=np.float32)


class IntentEpisodeDataset(Dataset):
    """Stack 3 recent frames and predict the current normalized action with a task id."""

    def __init__(
        self,
        records: list[EpisodeRecord],
        task_names: list[str],
        *,
        split: str = "train",
        image_size: tuple[int, int] = (160, 120),
        history: int = 3,
        augment: bool = False,
        cache_size: int = 4,
    ):
        self.records = list(records)
        self.split = split
        self.image_size = image_size
        self.history = history
        self.augment = augment
        self.task_names = list(task_names)
        self.task_to_index = {task: index for index, task in enumerate(self.task_names)}
        if not self.task_to_index:
            raise ValueError("task_names must not be empty")

        self.samples: list[SampleIndex] = []
        self.sample_weights: list[float] = []
        self.task_counts: dict[str, int] = {task: 0 for task in self.task_names}

        for episode_idx, record in enumerate(self.records):
            if record.task not in self.task_to_index:
                raise ValueError(
                    f"Task '{record.task}' from {record.episode_dir} is not present in the provided task vocabulary."
                )
            df = pd.read_parquet(record.episode_dir / "data.parquet", columns=["action"])
            actions = np.asarray(df["action"].tolist(), dtype=np.float32)
            self.task_counts[record.task] = self.task_counts.get(record.task, 0) + len(actions)
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
        action = torch.as_tensor(actions[sample.frame_idx], dtype=torch.float32)
        task_index = torch.tensor(self.task_to_index[record.task], dtype=torch.long)

        return {
            "image": stacked,
            "action": action,
            "task_index": task_index,
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
        """Favor turning and task-balanced samples over idle/straight frames."""
        vx, vy, omega = np.asarray(action, dtype=np.float32).tolist()
        magnitude = max(abs(vx), abs(vy), abs(omega))
        if magnitude < 0.05:
            base = 0.35
        elif abs(vy) > 0.05 or abs(omega) > 0.05:
            base = 1.2
        else:
            base = 0.8

        task_count = max(1, self.task_counts.get(task, 1))
        task_balance = len(self.task_names) / task_count
        return base * task_balance

    @property
    def total_frames(self) -> int:
        """Total frame count represented by this dataset split."""
        return sum(record.num_frames for record in self.records)

    @property
    def estimated_cache_bytes(self) -> int:
        """Approximate bytes needed to cache the resized RGB frames."""
        width, height = self.image_size
        return self.total_frames * width * height * 3

    def preload_all(self) -> None:
        """Decode and cache every episode once up front."""
        for record in self.records:
            self.cache.get(record)


def build_datasets(
    episodes_dir: Path | str,
    *,
    image_size: tuple[int, int] = (160, 120),
    history: int = 3,
    val_ratio: float = 0.2,
    seed: int | None = None,
    task_names: list[str] | None = None,
) -> tuple[IntentEpisodeDataset, IntentEpisodeDataset, list[str]]:
    """Create train/val datasets with shared hyperparameters and task vocabulary."""
    episodes_dir = Path(episodes_dir)
    all_records = discover_intent_episodes(episodes_dir)
    if task_names is None:
        task_names = discover_task_names(episodes_dir, all_records)
    else:
        task_names = list(task_names)

    train_records = split_sessions(all_records, split="train", val_ratio=val_ratio, seed=seed)
    val_records = split_sessions(all_records, split="val", val_ratio=val_ratio, seed=seed)

    train_dataset = IntentEpisodeDataset(
        train_records,
        task_names,
        split="train",
        image_size=image_size,
        history=history,
        augment=True,
    )
    val_dataset = IntentEpisodeDataset(
        val_records,
        task_names,
        split="val",
        image_size=image_size,
        history=history,
        augment=False,
    )
    return train_dataset, val_dataset, task_names
