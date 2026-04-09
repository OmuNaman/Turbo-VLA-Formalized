"""ACT-style task-conditioned policy model for TurboPi."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn

from . import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_D_MODEL,
    DEFAULT_FRAME_HISTORY,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_LATENT_DIM,
)


def _downsample_dim(size: int, *, kernel_size: int, stride: int, padding: int) -> int:
    return ((size + 2 * padding - kernel_size) // stride) + 1


def _spatial_shape(width: int, height: int) -> tuple[int, int]:
    convs = (
        (5, 2, 2),
        (3, 2, 1),
        (3, 2, 1),
        (3, 2, 1),
    )
    out_w = width
    out_h = height
    for kernel_size, stride, padding in convs:
        out_w = _downsample_dim(out_w, kernel_size=kernel_size, stride=stride, padding=padding)
        out_h = _downsample_dim(out_h, kernel_size=kernel_size, stride=stride, padding=padding)
    return out_w, out_h


@dataclass(frozen=True)
class ActIntentConfig:
    """Architecture and input-shape settings for the ACT-style intent policy."""

    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT
    frame_history: int = DEFAULT_FRAME_HISTORY
    task_vocab_size: int = 1
    action_dim: int = 3
    chunk_size: int = DEFAULT_CHUNK_SIZE
    d_model: int = DEFAULT_D_MODEL
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    ffn_mult: int = 4
    latent_dim: int = DEFAULT_LATENT_DIM
    dropout: float = 0.1

    @property
    def input_channels(self) -> int:
        return self.frame_history * 3

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return _spatial_shape(self.image_width, self.image_height)

    @property
    def num_spatial_tokens(self) -> int:
        width, height = self.spatial_shape
        return width * height


class ConvBlock(nn.Module):
    """Small Conv-BN-ReLU helper."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActIntentPolicy(nn.Module):
    """Compact ACT-style CVAE with task conditioning and chunked action prediction."""

    def __init__(self, config: ActIntentConfig | None = None):
        super().__init__()
        self.config = config or ActIntentConfig()
        if self.config.task_vocab_size < 1:
            raise ValueError("task_vocab_size must be at least 1")

        self.conv_stem = nn.Sequential(
            nn.Conv2d(self.config.input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ConvBlock(32, 64),
            ConvBlock(64, 96),
            ConvBlock(96, 128),
        )

        self.spatial_proj = nn.Linear(128, self.config.d_model)
        self.spatial_pos = nn.Parameter(
            torch.randn(1, self.config.num_spatial_tokens, self.config.d_model) * 0.02
        )
        self.task_embed = nn.Embedding(self.config.task_vocab_size, self.config.d_model)

        posterior_in_dim = self.config.chunk_size * (self.config.action_dim + 1)
        self.cvae_encoder = nn.Sequential(
            nn.Linear(posterior_in_dim, self.config.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.ReLU(inplace=True),
        )
        self.cvae_mu = nn.Linear(self.config.d_model, self.config.latent_dim)
        self.cvae_logvar = nn.Linear(self.config.d_model, self.config.latent_dim)
        self.z_proj = nn.Linear(self.config.latent_dim, self.config.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_model * self.config.ffn_mult,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.config.n_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_model * self.config.ffn_mult,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=self.config.n_decoder_layers)

        self.action_queries = nn.Parameter(torch.randn(self.config.chunk_size, self.config.d_model) * 0.02)
        self.action_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.d_model, self.config.action_dim),
            nn.Tanh(),
        )

    def encode_z(
        self,
        action_chunk: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action_mask is None:
            action_mask = torch.ones(
                action_chunk.shape[:2],
                dtype=action_chunk.dtype,
                device=action_chunk.device,
            )
        masked_actions = action_chunk * action_mask.unsqueeze(-1)
        posterior_input = torch.cat(
            [masked_actions.flatten(1), action_mask.to(action_chunk.dtype)],
            dim=1,
        )
        hidden = self.cvae_encoder(posterior_input)
        return self.cvae_mu(hidden), self.cvae_logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        image: torch.Tensor,
        task_ids: torch.Tensor,
        action_chunk: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor [B,C,H,W], got {tuple(image.shape)}")
        if task_ids.ndim == 0:
            task_ids = task_ids.unsqueeze(0)
        if task_ids.ndim != 1:
            raise ValueError(f"Expected task ids shaped [B], got {tuple(task_ids.shape)}")

        batch_size = image.shape[0]
        x = self.conv_stem(image)
        x = x.flatten(2).transpose(1, 2)
        x = self.spatial_proj(x) + self.spatial_pos[:, : x.shape[1], :]

        task_token = self.task_embed(task_ids.long())

        mu = logvar = None
        if action_chunk is not None:
            mu, logvar = self.encode_z(action_chunk, action_mask)
            z = self.reparameterize(mu, logvar)
        else:
            z = torch.zeros(batch_size, self.config.latent_dim, dtype=x.dtype, device=x.device)

        z_token = self.z_proj(z).unsqueeze(1)
        encoder_input = torch.cat([x, task_token.unsqueeze(1), z_token], dim=1)
        memory = self.transformer_encoder(encoder_input)

        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + task_token.unsqueeze(1)
        decoded = self.transformer_decoder(queries, memory)
        pred = self.action_head(decoded)

        if action_chunk is not None:
            return pred, mu, logvar
        return pred


def build_model(config: ActIntentConfig | None = None) -> ActIntentPolicy:
    """Create a fresh ACT-style task-conditioned policy."""
    return ActIntentPolicy(config=config)


def save_checkpoint(
    path: Path,
    model: ActIntentPolicy,
    *,
    epoch: int,
    metrics: dict[str, float],
    extra: dict[str, object] | None = None,
) -> None:
    """Persist a checkpoint with model weights and metadata."""
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


def load_checkpoint(
    path: Path,
    map_location: str | torch.device | None = None,
) -> tuple[ActIntentPolicy, dict[str, object]]:
    """Load a saved ACT-style checkpoint and ignore unknown config keys."""
    payload = torch.load(Path(path), map_location=map_location)
    raw_config = payload.get("model_config", {})
    known_keys = {field.name for field in fields(ActIntentConfig)}
    filtered_config = {key: value for key, value in raw_config.items() if key in known_keys}
    config = ActIntentConfig(**filtered_config)
    model = ActIntentPolicy(config=config)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
