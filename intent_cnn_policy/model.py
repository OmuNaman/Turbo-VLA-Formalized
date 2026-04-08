"""Task-conditioned CNN policy for image-based path following."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from . import (
    DEFAULT_FRAME_HISTORY,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_TASK_EMBEDDING_DIM,
)


@dataclass(frozen=True)
class IntentCNNConfig:
    """Architecture and input-shape settings for the task-conditioned policy."""

    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT
    frame_history: int = DEFAULT_FRAME_HISTORY
    task_embedding_dim: int = DEFAULT_TASK_EMBEDDING_DIM
    task_vocab_size: int = 1
    action_dim: int = 3
    hidden_dim: int = 64
    dropout: float = 0.1

    @property
    def input_channels(self) -> int:
        return self.frame_history * 3


class ConvBlock(nn.Module):
    """Conv-BN-ReLU helper block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IntentCNNPolicy(nn.Module):
    """Tiny CNN that regresses normalized [vx, vy, omega] conditioned on a task id."""

    def __init__(self, config: IntentCNNConfig | None = None):
        super().__init__()
        self.config = config or IntentCNNConfig()
        if self.config.task_vocab_size < 1:
            raise ValueError("task_vocab_size must be at least 1")

        self.encoder = nn.Sequential(
            ConvBlock(self.config.input_channels, 32, kernel_size=5, stride=2, padding=2),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.task_embedding = nn.Embedding(self.config.task_vocab_size, self.config.task_embedding_dim)
        fused_dim = 128 + self.config.task_embedding_dim
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.config.action_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [B,C,H,W], got shape {tuple(x.shape)}")
        if x.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected {self.config.input_channels} input channels, got {x.shape[1]}. "
                f"Frame history={self.config.frame_history}"
            )
        if task_ids.ndim == 0:
            task_ids = task_ids.unsqueeze(0)
        if task_ids.ndim != 1:
            raise ValueError(f"Expected task ids shaped [B], got {tuple(task_ids.shape)}")

        vision_feat = self.encoder(x).flatten(1)
        task_feat = self.task_embedding(task_ids.long())
        fused = torch.cat([vision_feat, task_feat], dim=1)
        return self.head(fused)


TaskIntentCNNModel = IntentCNNPolicy


def build_model(config: IntentCNNConfig | None = None) -> IntentCNNPolicy:
    """Create a fresh task-conditioned CNN policy model."""
    return IntentCNNPolicy(config=config)


def save_checkpoint(
    path: Path,
    model: IntentCNNPolicy,
    *,
    epoch: int,
    metrics: dict[str, float],
    extra: dict[str, object] | None = None,
) -> None:
    """Persist a checkpoint with model and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device | None = None) -> tuple[IntentCNNPolicy, dict[str, object]]:
    """Load a checkpoint and return the instantiated model plus raw payload."""
    payload = torch.load(Path(path), map_location=map_location)
    config = IntentCNNConfig(**payload.get("model_config", {}))
    model = IntentCNNPolicy(config=config)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
