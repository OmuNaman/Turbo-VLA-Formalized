"""Task-conditioned CNN policy package for TurboPi."""

from __future__ import annotations

DEFAULT_IMAGE_WIDTH = 160
DEFAULT_IMAGE_HEIGHT = 120
DEFAULT_FRAME_HISTORY = 3
DEFAULT_TASK_EMBEDDING_DIM = 32
DEFAULT_DATA_ROOT = "data/turbopi_intent_cnn/episodes"

__all__ = [
    "DEFAULT_IMAGE_WIDTH",
    "DEFAULT_IMAGE_HEIGHT",
    "DEFAULT_FRAME_HISTORY",
    "DEFAULT_TASK_EMBEDDING_DIM",
    "DEFAULT_DATA_ROOT",
]
