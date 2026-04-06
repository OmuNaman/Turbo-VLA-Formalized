"""Dataset utilities for CNN path-following training."""

from __future__ import annotations

import hashlib
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

from . import DEFAULT_FRAME_HISTORY, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH

try:
    import av
except ImportError as exc:  # pragma: no cover - environment specific
    raise RuntimeError("PyAV is required for CNN dataset loading. Install with `pip install av`.") from exc


@dataclass(frozen=True)
class EpisodeRecord:
    """Metadata about one accepted CNN episode."""

    episode_dir: Path
    session_name: str
    num_frames: int
    task: str
    direction: str


@dataclass(frozen=True)
class SampleIndex:
    """Address one target frame within one episode."""

    episode_idx: int
    frame_idx: int


def discover_cnn_episodes(episodes_dir: Path) -> list[EpisodeRecord]:
    """Discover saved CNN episodes and their directions."""
    records: list[EpisodeRecord] = []
    for episode_dir in sorted(Path(episodes_dir).glob("**/episode_*")):
        if not episode_dir.is_dir():
            continue
        parquet_path = episode_dir / "data.parquet"
        video_path = episode_dir / "video.mp4"
        info_path = episode_dir / "episode_info.json"
        if not parquet_path.exists() or not video_path.exists() or not info_path.exists():
            continue

        df = pd.read_parquet(parquet_path)
        if df.empty:
            continue

        info = pd.read_json(info_path, typ="series")
        records.append(
            EpisodeRecord(
                episode_dir=episode_dir,
                session_name=episode_dir.parent.name,
                num_frames=len(df),
                task=str(df["task"].iloc[0]),
                direction=str(info.get("direction", "unknown")),
            )
        )
    return records


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


def split_session_dirs(
    session_dirs: list[Path],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Split session directories into train/val subsets."""
    if len(session_dirs) <= 1:
        return list(session_dirs), []

    rng = random.Random(seed)
    shuffled = list(session_dirs)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_set = set(shuffled[:val_count])
    train_sessions = sorted(path for path in shuffled if path not in val_set)
    val_sessions = sorted(val_set)
    return train_sessions, val_sessions


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


class LoopEpisodeDataset(Dataset):
    """Stack 3 recent frames and predict the current normalized action."""

    def __init__(
        self,
        episodes_dir: Path | str,
        split: str = "train",
        image_size: tuple[int, int] = (160, 120),
        history: int = 3,
        augment: bool = False,
        val_ratio: float = 0.2,
        seed: int | None = None,
        cache_size: int = 4,
    ):
        self.episodes_dir = Path(episodes_dir)
        self.image_size = image_size
        self.history = history
        self.augment = augment
        self.records = split_sessions(
            discover_cnn_episodes(self.episodes_dir),
            split=split,
            val_ratio=val_ratio,
            seed=seed,
        )
        self.samples: list[SampleIndex] = []
        self.sample_weights: list[float] = []
        for episode_idx, record in enumerate(self.records):
            df = pd.read_parquet(record.episode_dir / "data.parquet", columns=["action"])
            actions = np.asarray(df["action"].tolist(), dtype=np.float32)
            for frame_idx, action in enumerate(actions):
                self.samples.append(SampleIndex(episode_idx=episode_idx, frame_idx=frame_idx))
                self.sample_weights.append(self._compute_sample_weight(action))
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

        return {
            "image": stacked,
            "action": action,
            "direction": record.direction,
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

    def _compute_sample_weight(self, action: np.ndarray) -> float:
        """Favor turning and lateral corrections over idle/straight samples."""
        vx, vy, omega = np.asarray(action, dtype=np.float32).tolist()
        magnitude = max(abs(vx), abs(vy), abs(omega))
        if magnitude < 0.05:
            return 0.35
        if abs(vy) > 0.05 or abs(omega) > 0.05:
            return 1.2
        return 0.8

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
    image_size: tuple[int, int] = (160, 120),
    history: int = 3,
    val_ratio: float = 0.2,
    seed: int | None = None,
) -> tuple[LoopEpisodeDataset, LoopEpisodeDataset]:
    """Create train/val datasets with shared hyperparameters."""
    train_dataset = LoopEpisodeDataset(
        episodes_dir=episodes_dir,
        split="train",
        image_size=image_size,
        history=history,
        augment=True,
        val_ratio=val_ratio,
        seed=seed,
    )
    val_dataset = LoopEpisodeDataset(
        episodes_dir=episodes_dir,
        split="val",
        image_size=image_size,
        history=history,
        augment=False,
        val_ratio=val_ratio,
        seed=seed,
    )
    return train_dataset, val_dataset


def stable_worker_seed(worker_id: int) -> int:
    """Return a deterministic seed for dataloader workers."""
    base = int.from_bytes(hashlib.sha256(f"loop_cnn_worker_{worker_id}".encode("utf-8")).digest()[:4], "little")
    return base


def frame_to_tensor(image_rgb: np.ndarray, *, image_width: int, image_height: int) -> torch.Tensor:
    """Resize one RGB frame and convert it to CHW float32 tensor in [0, 1]."""
    image = Image.fromarray(image_rgb)
    image = image.resize((image_width, image_height), Image.Resampling.BILINEAR)
    return TF.to_tensor(image)


class LoopPolicyDataset(LoopEpisodeDataset):
    """Compatibility wrapper used by the train/eval/drive entrypoints."""

    def __init__(
        self,
        *,
        episodes_root: Path | str,
        session_dirs: list[Path],
        frame_history: int = DEFAULT_FRAME_HISTORY,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        augment: bool = False,
    ):
        self.episodes_root = Path(episodes_root)
        self.session_dirs = list(session_dirs)
        self.image_size = (image_width, image_height)
        self.history = frame_history
        self.augment = augment

        allowed = {session_dir.name for session_dir in self.session_dirs}
        self.records = [
            record
            for record in discover_cnn_episodes(self.episodes_root)
            if record.session_name in allowed
        ]
        self.samples: list[SampleIndex] = []
        self.sample_weights: list[float] = []
        for episode_idx, record in enumerate(self.records):
            df = pd.read_parquet(record.episode_dir / "data.parquet", columns=["action"])
            actions = np.asarray(df["action"].tolist(), dtype=np.float32)
            for frame_idx, action in enumerate(actions):
                self.samples.append(SampleIndex(episode_idx=episode_idx, frame_idx=frame_idx))
                self.sample_weights.append(self._compute_sample_weight(action))
        self.cache = _EpisodeCache(image_size=self.image_size, max_items=4)
