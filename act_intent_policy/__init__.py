"""ACT-style task-conditioned policy package for TurboPi."""

from __future__ import annotations

DEFAULT_IMAGE_WIDTH = 160
DEFAULT_IMAGE_HEIGHT = 120
DEFAULT_FRAME_HISTORY = 3
DEFAULT_DATA_ROOT = "data/turbopi_intent_cnn/episodes"
DEFAULT_CHUNK_SIZE = 8
DEFAULT_D_MODEL = 128
DEFAULT_LATENT_DIM = 32

__all__ = [
    "DEFAULT_IMAGE_WIDTH",
    "DEFAULT_IMAGE_HEIGHT",
    "DEFAULT_FRAME_HISTORY",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_D_MODEL",
    "DEFAULT_LATENT_DIM",
]
