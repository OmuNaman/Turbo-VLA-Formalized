"""Train the ACT-style task-conditioned TurboPi policy."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from . import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_D_MODEL,
    DEFAULT_DATA_ROOT,
    DEFAULT_FRAME_HISTORY,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_LATENT_DIM,
)
from .cache import build_cache, cache_is_compatible, default_cache_dir
from .dataset import build_cached_datasets, build_datasets
from .model import ActIntentConfig, build_model, save_checkpoint


def resolve_run_dir(base_dir: Path) -> Path:
    """Create a unique timestamped run directory beneath the requested base path."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = base_dir / timestamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base_dir / f"{timestamp}_{suffix:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_enabled(device: torch.device, requested: bool | None) -> bool:
    """Enable AMP by default on CUDA, unless the caller explicitly disables it."""
    if device.type != "cuda":
        return False
    if requested is None:
        return True
    return bool(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the TurboPi ACT-style task-conditioned policy")
    parser.add_argument("--episodes-dir", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--run-dir",
        default="runs/act_intent_v1",
        help="Base directory for ACT-Intent runs; each launch creates a timestamped child run",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-history", type=int, default=DEFAULT_FRAME_HISTORY)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-encoder-layers", type=int, default=2)
    parser.add_argument("--n-decoder-layers", type=int, default=2)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--huber-delta", type=float, default=0.25)
    parser.add_argument(
        "--first-action-weight",
        type=float,
        default=0.25,
        help="Extra weight on the first action in each predicted chunk to better match drive-time usage",
    )
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cache-size", type=int, default=4, help="Per-worker decoded episode cache size")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for the reusable ACT sample cache; defaults beside the dataset root",
    )
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=None,
        help="Worker processes for ACT cache building; defaults to an automatic CPU-based value",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("off", "build", "require"),
        default="off",
        help="Use the ACT cache for fast training (`build` creates/reuses, `require` errors if missing)",
    )
    parser.add_argument(
        "--preload-all",
        action="store_true",
        help="Force preloading all resized frames into RAM before training in raw mode",
    )
    parser.add_argument(
        "--preload-threshold-records",
        type=int,
        default=64,
        help="Auto-preload when the train split has at most this many episodes in raw mode",
    )
    parser.add_argument(
        "--preload-threshold-frames",
        type=int,
        default=25000,
        help="Auto-preload when the train split has at most this many frames in raw mode",
    )
    parser.add_argument("--log-interval", type=int, default=25, help="Update timing/utilization logs every N steps")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision")
    parser.set_defaults(amp=None)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return parser


def _loader_kwargs(
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    prefetch_factor: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = max(2, prefetch_factor)
    return kwargs


def build_loaders(
    episodes_dir: Path,
    *,
    val_ratio: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    frame_history: int,
    image_width: int,
    image_height: int,
    chunk_size: int,
    cache_size: int,
    cache_dir: Path | None,
    cache_workers: int | None = None,
    cache_mode: str,
    preload_all: bool,
    preload_threshold_records: int,
    preload_threshold_frames: int,
    prefetch_factor: int,
    device: torch.device,
    show_progress: bool,
) -> tuple[DataLoader, DataLoader | None, list[str], list[str], list[str], str, Path | None]:
    image_size = (image_width, image_height)
    resolved_cache_dir = None
    data_source = "raw"

    if cache_mode != "off":
        resolved_cache_dir = cache_dir or default_cache_dir(
            episodes_dir,
            image_width=image_width,
            image_height=image_height,
            frame_history=frame_history,
            chunk_size=chunk_size,
        )
        if cache_mode == "build":
            if not cache_is_compatible(
                resolved_cache_dir,
                image_width=image_width,
                image_height=image_height,
                frame_history=frame_history,
                chunk_size=chunk_size,
            ):
                print(f"[train] Building ACT cache at {resolved_cache_dir}")
                build_cache(
                    episodes_dir,
                    resolved_cache_dir,
                    image_width=image_width,
                    image_height=image_height,
                    frame_history=frame_history,
                    chunk_size=chunk_size,
                    workers=cache_workers,
                    overwrite=resolved_cache_dir.exists(),
                    show_progress=show_progress,
                )
            else:
                print(f"[train] Reusing ACT cache at {resolved_cache_dir}")
        else:
            if not cache_is_compatible(
                resolved_cache_dir,
                image_width=image_width,
                image_height=image_height,
                frame_history=frame_history,
                chunk_size=chunk_size,
            ):
                raise RuntimeError(
                    f"ACT cache is missing or incompatible: {resolved_cache_dir}. "
                    "Use --cache-mode build to create it."
                )

        train_dataset, val_dataset, task_names = build_cached_datasets(
            resolved_cache_dir,
            val_ratio=val_ratio,
            seed=seed,
        )
        data_source = "cache"
    else:
        train_dataset, val_dataset, task_names = build_datasets(
            episodes_dir=episodes_dir,
            image_size=image_size,
            history=frame_history,
            chunk_size=chunk_size,
            val_ratio=val_ratio,
            seed=seed,
            cache_size=cache_size,
        )
        if len(train_dataset) == 0:
            raise RuntimeError(f"No intent-conditioned episodes found under {episodes_dir}")
        if (
            train_dataset.records
            and (
                preload_all
                or (
                    len(train_dataset.records) <= preload_threshold_records
                    and train_dataset.total_frames <= preload_threshold_frames
                )
            )
        ):
            estimated_gb = train_dataset.estimated_cache_bytes / (1024 ** 3)
            print(
                f"[train] Preloading {len(train_dataset.records)} ACT train episodes into RAM "
                f"(~{estimated_gb:.2f} GB resized frames) to avoid repeated video decode."
            )
            train_dataset.preload_all(show_progress=show_progress, desc="Decode train episodes")
            if len(val_dataset.records) > 0:
                val_dataset.preload_all(show_progress=show_progress, desc="Decode val episodes")

    if len(train_dataset) == 0:
        raise RuntimeError(f"No ACT training samples available under {episodes_dir}")

    loader_kwargs = _loader_kwargs(
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        prefetch_factor=prefetch_factor,
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=WeightedRandomSampler(
            weights=torch.as_tensor(train_dataset.sample_weights, dtype=torch.double),
            num_samples=len(train_dataset.sample_weights),
            replacement=True,
        ),
        shuffle=False,
        **loader_kwargs,
    )

    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )

    train_sessions = sorted({record.session_name for record in train_dataset.records})
    val_sessions = sorted({record.session_name for record in val_dataset.records})
    return train_loader, val_loader, train_sessions, val_sessions, task_names, data_source, resolved_cache_dir


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    """Huber loss over valid chunk positions only."""
    criterion = nn.HuberLoss(delta=delta, reduction="none")
    elementwise = criterion(pred, target)
    weighted = elementwise * mask.unsqueeze(-1)
    denom = torch.clamp(mask.sum() * pred.shape[-1], min=1.0)
    return weighted.sum() / denom


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL divergence between posterior and standard normal prior."""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def compute_first_step_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute error of the first predicted action in each chunk."""
    return torch.mean(torch.abs(pred[:, 0, :] - target[:, 0, :]), dim=0)


def first_action_huber_loss(
    pred: torch.Tensor,
    first_action: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    """Huber loss on the first predicted action only."""
    criterion = nn.HuberLoss(delta=delta, reduction="mean")
    return criterion(pred[:, 0, :], first_action)


def cosine_lr(base_lr: float, *, epoch: int, epochs: int, warmup_epochs: int = 3) -> float:
    """Warmup plus cosine decay learning-rate schedule."""
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def move_images_to_device(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move image batches to device and normalize uint8 caches on GPU."""
    images = images.to(device, non_blocking=device.type == "cuda")
    if images.dtype == torch.uint8:
        images = images.to(dtype=torch.float32).div_(255.0)
    else:
        images = images.to(dtype=torch.float32)
    if images.ndim == 4 and device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def move_batch_to_device(
    batch: dict[str, torch.Tensor | str],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move the tensor parts of one ACT batch to the requested device."""
    images = move_images_to_device(batch["image"], device)
    task_ids = batch["task_index"].to(device, non_blocking=device.type == "cuda")
    target = batch["action_chunk"].to(device, non_blocking=device.type == "cuda", dtype=torch.float32)
    mask = batch["action_mask"].to(device, non_blocking=device.type == "cuda", dtype=torch.float32)
    first_action = batch["first_action"].to(device, non_blocking=device.type == "cuda", dtype=torch.float32)
    return images, task_ids, target, mask, first_action


def epoch_timing_summary(
    *,
    total_examples: int,
    steps: int,
    epoch_seconds: float,
    data_seconds: float,
    device: torch.device,
) -> dict[str, float]:
    """Convert coarse wall-clock timings into user-friendly throughput metrics."""
    compute_seconds = max(epoch_seconds - data_seconds, 0.0)
    summary = {
        "samples_per_sec": float(total_examples / max(epoch_seconds, 1e-6)),
        "batches_per_sec": float(steps / max(epoch_seconds, 1e-6)),
        "avg_step_time_ms": float((epoch_seconds / max(steps, 1)) * 1000.0),
        "avg_data_time_ms": float((data_seconds / max(steps, 1)) * 1000.0),
        "avg_compute_time_ms": float((compute_seconds / max(steps, 1)) * 1000.0),
        "epoch_seconds": float(epoch_seconds),
    }
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        summary["gpu_peak_allocated_gb"] = float(allocated)
        summary["gpu_peak_reserved_gb"] = float(reserved)
    else:
        summary["gpu_peak_allocated_gb"] = 0.0
        summary["gpu_peak_reserved_gb"] = 0.0
    return summary


def write_training_summary(
    path: Path,
    *,
    device: torch.device,
    args,
    model_config: ActIntentConfig,
    train_sessions: list[str],
    val_sessions: list[str],
    task_names: list[str],
    history: list[dict[str, float]],
    best_epoch: int,
    best_metric: float,
    interrupted: bool,
) -> None:
    summary = {
        "device": str(device),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "task_names": task_names,
        "model_config": asdict(model_config),
        "history": history,
        "interrupted": interrupted,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader | None,
    *,
    device: torch.device,
    huber_delta: float,
    first_action_weight: float,
    amp_enabled: bool,
    show_progress: bool,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    if loader is None:
        return {
            "loss": math.nan,
            "objective_loss": math.nan,
            "first_action_loss": math.nan,
            "chunk_mae": math.nan,
            "first_mae_vx": math.nan,
            "first_mae_vy": math.nan,
            "first_mae_omega": math.nan,
        }

    model.eval()
    total_loss = 0.0
    total_objective = 0.0
    total_first_action_loss = 0.0
    total_examples = 0
    total_valid_steps = 0.0
    total_chunk_abs = 0.0
    first_abs = torch.zeros(3, dtype=torch.float64)

    progress = None
    if show_progress:
        progress = tqdm(
            total=len(loader.dataset),
            desc=f"Epoch {epoch:03d}/{epochs:03d} val",
            unit="sample",
            dynamic_ncols=True,
            leave=False,
        )

    autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32
    try:
        with torch.no_grad():
            for batch in loader:
                images, task_ids, target, mask, first_action = move_batch_to_device(batch, device=device)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                    pred = model(images, task_ids)
                    loss = masked_huber_loss(pred, target, mask, delta=huber_delta)
                    first_loss = first_action_huber_loss(pred, first_action, delta=huber_delta)
                    objective_loss = loss + first_action_weight * first_loss

                batch_size = images.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_objective += float(objective_loss.item()) * batch_size
                total_first_action_loss += float(first_loss.item()) * batch_size
                total_examples += batch_size
                total_valid_steps += float(mask.sum().item())
                total_chunk_abs += float((torch.abs(pred - target) * mask.unsqueeze(-1)).sum().item())
                first_abs += compute_first_step_mae(pred, target).double().cpu() * batch_size

                if progress is not None:
                    progress.update(batch_size)
                    progress.set_postfix(avg_loss=f"{total_objective / max(1, total_examples):.4f}")
    finally:
        if progress is not None:
            progress.close()

    if total_examples == 0:
        return {
            "loss": math.nan,
            "objective_loss": math.nan,
            "first_action_loss": math.nan,
            "chunk_mae": math.nan,
            "first_mae_vx": math.nan,
            "first_mae_vy": math.nan,
            "first_mae_omega": math.nan,
        }

    denom = max(total_valid_steps * target.shape[-1], 1.0)
    return {
        "loss": total_loss / total_examples,
        "objective_loss": total_objective / total_examples,
        "first_action_loss": total_first_action_loss / total_examples,
        "chunk_mae": total_chunk_abs / denom,
        "first_mae_vx": float(first_abs[0].item() / total_examples),
        "first_mae_vy": float(first_abs[1].item() / total_examples),
        "first_mae_omega": float(first_abs[2].item() / total_examples),
    }


def save_epoch_artifacts(
    checkpoint_dir: Path,
    model: nn.Module,
    epoch: int,
    metrics: dict[str, float],
    *,
    extra: dict[str, object],
    is_best: bool,
) -> None:
    epoch_dir = checkpoint_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(epoch_dir / "last.pt", model, epoch=epoch, metrics=metrics, extra=extra)
    with (epoch_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    if is_best:
        save_checkpoint(epoch_dir / "best.pt", model, epoch=epoch, metrics=metrics, extra=extra)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    huber_delta: float,
    first_action_weight: float,
    kl_scale: float,
    grad_clip: float,
    amp_enabled: bool,
    log_interval: int,
    show_progress: bool,
    epoch: int,
    epochs: int,
    lr: float,
) -> dict[str, float]:
    model.train()
    total_examples = 0
    total_objective = 0.0
    total_recon = 0.0
    total_first_action_loss = 0.0
    total_kl = 0.0
    total_chunk_abs = 0.0
    total_valid_steps = 0.0
    total_data_seconds = 0.0
    first_abs = torch.zeros(3, dtype=torch.float64)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = None
    if show_progress:
        progress = tqdm(
            total=len(loader.dataset),
            desc=f"Epoch {epoch:03d}/{epochs:03d} train",
            unit="sample",
            dynamic_ncols=True,
            leave=False,
        )

    autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32
    epoch_start = time.perf_counter()
    last_end = time.perf_counter()
    steps = 0
    try:
        for steps, batch in enumerate(loader, start=1):
            total_data_seconds += time.perf_counter() - last_end
            images, task_ids, target, mask, first_action = move_batch_to_device(batch, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                pred, mu, logvar = model(images, task_ids, target, mask)
                recon_loss = masked_huber_loss(pred, target, mask, delta=huber_delta)
                first_loss = first_action_huber_loss(pred, first_action, delta=huber_delta)
                kl_loss = kl_divergence(mu, logvar)
                loss = recon_loss + first_action_weight * first_loss + kl_scale * kl_loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch_size = images.shape[0]
            total_examples += batch_size
            total_objective += float(loss.item()) * batch_size
            total_recon += float(recon_loss.item()) * batch_size
            total_first_action_loss += float(first_loss.item()) * batch_size
            total_kl += float(kl_loss.item()) * batch_size
            total_valid_steps += float(mask.sum().item())
            total_chunk_abs += float((torch.abs(pred.detach() - target) * mask.unsqueeze(-1)).sum().item())
            first_abs += compute_first_step_mae(pred.detach(), target).double().cpu() * batch_size

            if progress is not None:
                progress.update(batch_size)
                if steps == 1 or steps % max(1, log_interval) == 0 or total_examples >= len(loader.dataset):
                    elapsed = max(time.perf_counter() - epoch_start, 1e-6)
                    avg_data_ms = (total_data_seconds / steps) * 1000.0
                    avg_compute_ms = max((elapsed - total_data_seconds) / steps, 0.0) * 1000.0
                    postfix = {
                        "obj": f"{total_objective / max(1, total_examples):.4f}",
                        "recon": f"{total_recon / max(1, total_examples):.4f}",
                        "first": f"{total_first_action_loss / max(1, total_examples):.4f}",
                        "kl": f"{total_kl / max(1, total_examples):.4f}",
                        "lr": f"{lr:.2e}",
                        "sample_s": f"{total_examples / elapsed:.1f}",
                        "data_ms": f"{avg_data_ms:.1f}",
                        "compute_ms": f"{avg_compute_ms:.1f}",
                    }
                    if device.type == "cuda":
                        postfix["gpu_gb"] = f"{torch.cuda.memory_allocated(device) / (1024 ** 3):.2f}"
                    progress.set_postfix(postfix)
            last_end = time.perf_counter()
    finally:
        if progress is not None:
            progress.close()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    epoch_seconds = max(time.perf_counter() - epoch_start, 1e-6)
    timing = epoch_timing_summary(
        total_examples=total_examples,
        steps=max(steps, 1),
        epoch_seconds=epoch_seconds,
        data_seconds=total_data_seconds,
        device=device,
    )
    denom = max(total_valid_steps * target.shape[-1], 1.0)
    return {
        "objective_loss": total_objective / max(1, total_examples),
        "recon_loss": total_recon / max(1, total_examples),
        "first_action_loss": total_first_action_loss / max(1, total_examples),
        "kl_loss": total_kl / max(1, total_examples),
        "chunk_mae": total_chunk_abs / denom,
        "first_mae_vx": float(first_abs[0].item() / max(1, total_examples)),
        "first_mae_vy": float(first_abs[1].item() / max(1, total_examples)),
        "first_mae_omega": float(first_abs[2].item() / max(1, total_examples)),
        **timing,
    }


def maybe_compile_model(model: nn.Module, *, enabled: bool) -> tuple[nn.Module, bool]:
    """Optionally wrap the model with torch.compile when available."""
    if not enabled:
        return model, False
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:  # pragma: no cover - version dependent
        print("[train] WARNING: torch.compile is unavailable in this PyTorch build; continuing without it.")
        return model, False
    try:
        return compile_fn(model, mode="reduce-overhead"), True
    except Exception as exc:  # pragma: no cover - backend dependent
        print(f"[train] WARNING: torch.compile failed ({exc}); continuing without it.")
        return model, False


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = resolve_amp_enabled(device, args.amp)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    episodes_dir = Path(args.episodes_dir)
    run_base_dir = Path(args.run_dir)
    run_dir = resolve_run_dir(run_base_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    requested_cache_dir = Path(args.cache_dir) if args.cache_dir else None
    (
        train_loader,
        val_loader,
        train_sessions,
        val_sessions,
        task_names,
        data_source,
        resolved_cache_dir,
    ) = build_loaders(
        episodes_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        frame_history=args.frame_history,
        image_width=args.image_width,
        image_height=args.image_height,
        chunk_size=args.chunk_size,
        cache_size=args.cache_size,
        cache_dir=requested_cache_dir,
        cache_workers=args.cache_workers,
        cache_mode=args.cache_mode,
        preload_all=args.preload_all,
        preload_threshold_records=args.preload_threshold_records,
        preload_threshold_frames=args.preload_threshold_frames,
        prefetch_factor=args.prefetch_factor,
        device=device,
        show_progress=not args.no_progress,
    )

    train_dataset = train_loader.dataset
    val_dataset = None if val_loader is None else val_loader.dataset

    if not val_sessions:
        print("[train] WARNING: Only one session available. Validation will be skipped.")

    print(f"[train] Sessions: {', '.join(train_sessions) if train_sessions else '(none)'}")
    print(f"[train] Device: {device}")
    print(f"[train] Data source: {data_source}")
    print(f"[train] AMP: {'on' if amp_enabled else 'off'}")
    print(f"[train] Batch size: {args.batch_size}")
    print(f"[train] First-action weight: {args.first_action_weight:.3f}")
    if resolved_cache_dir is not None:
        print(f"[train] Cache dir: {resolved_cache_dir}")
    print(f"[train] Run base dir: {run_base_dir}")
    print(f"[train] Run dir: {run_dir}")
    print(f"[train] Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"[train] Val samples: {len(val_dataset)}")
    print(f"[train] Task vocabulary: {len(task_names)} tasks")

    model_config = ActIntentConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        frame_history=args.frame_history,
        task_vocab_size=max(1, len(task_names)),
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        ffn_mult=args.ffn_mult,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    )
    model = build_model(model_config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model, compiled = maybe_compile_model(model, enabled=args.compile and device.type == "cuda")
    if compiled:
        print("[train] torch.compile: on")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.amp.GradScaler("cpu", enabled=False)

    history: list[dict[str, float]] = []
    best_metric = math.inf
    best_epoch = 0
    interrupted = False
    show_progress = not args.no_progress

    extra = {
        "task_names": task_names,
        "task_to_index": train_dataset.task_to_index,
        "dataset_root": str(episodes_dir),
        "run_dir": str(run_dir),
        "huber_delta": float(args.huber_delta),
        "first_action_weight": float(args.first_action_weight),
        "kl_weight": float(args.kl_weight),
        "kl_warmup_epochs": int(args.kl_warmup_epochs),
        "data_source": data_source,
        "cache_dir": None if resolved_cache_dir is None else str(resolved_cache_dir),
        "amp_enabled": bool(amp_enabled),
        "compiled": bool(compiled),
    }

    try:
        for epoch in range(1, args.epochs + 1):
            lr = cosine_lr(args.lr, epoch=epoch, epochs=args.epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr
            kl_scale = args.kl_weight * min(1.0, epoch / max(1, args.kl_warmup_epochs))

            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler=scaler,
                device=device,
                huber_delta=args.huber_delta,
                first_action_weight=args.first_action_weight,
                kl_scale=kl_scale,
                grad_clip=args.grad_clip,
                amp_enabled=amp_enabled,
                log_interval=args.log_interval,
                show_progress=show_progress,
                epoch=epoch,
                epochs=args.epochs,
                lr=lr,
            )
            val_metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                huber_delta=args.huber_delta,
                first_action_weight=args.first_action_weight,
                amp_enabled=amp_enabled,
                show_progress=show_progress,
                epoch=epoch,
                epochs=args.epochs,
            )

            metric = val_metrics["objective_loss"]
            if math.isnan(metric):
                metric = train_metrics["objective_loss"]

            epoch_metrics = {
                "epoch": float(epoch),
                "train_objective_loss": float(train_metrics["objective_loss"]),
                "train_recon_loss": float(train_metrics["recon_loss"]),
                "train_first_action_loss": float(train_metrics["first_action_loss"]),
                "train_kl_loss": float(train_metrics["kl_loss"]),
                "train_chunk_mae": float(train_metrics["chunk_mae"]),
                "train_first_mae_vx": float(train_metrics["first_mae_vx"]),
                "train_first_mae_vy": float(train_metrics["first_mae_vy"]),
                "train_first_mae_omega": float(train_metrics["first_mae_omega"]),
                "train_samples_per_sec": float(train_metrics["samples_per_sec"]),
                "train_batches_per_sec": float(train_metrics["batches_per_sec"]),
                "train_avg_step_time_ms": float(train_metrics["avg_step_time_ms"]),
                "train_avg_data_time_ms": float(train_metrics["avg_data_time_ms"]),
                "train_avg_compute_time_ms": float(train_metrics["avg_compute_time_ms"]),
                "train_gpu_peak_allocated_gb": float(train_metrics["gpu_peak_allocated_gb"]),
                "train_gpu_peak_reserved_gb": float(train_metrics["gpu_peak_reserved_gb"]),
                "val_loss": float(val_metrics["loss"]),
                "val_objective_loss": float(val_metrics["objective_loss"]),
                "val_first_action_loss": float(val_metrics["first_action_loss"]),
                "val_chunk_mae": float(val_metrics["chunk_mae"]),
                "val_first_mae_vx": float(val_metrics["first_mae_vx"]),
                "val_first_mae_vy": float(val_metrics["first_mae_vy"]),
                "val_first_mae_omega": float(val_metrics["first_mae_omega"]),
                "lr": float(lr),
                "kl_scale": float(kl_scale),
            }
            history.append(epoch_metrics)

            is_best = metric < best_metric
            if is_best:
                best_metric = metric
                best_epoch = epoch

            save_epoch_artifacts(
                checkpoint_dir,
                model,
                epoch,
                epoch_metrics,
                extra=extra,
                is_best=is_best,
            )
            save_checkpoint(checkpoint_dir / "last.pt", model, epoch=epoch, metrics=epoch_metrics, extra=extra)
            if is_best:
                save_checkpoint(checkpoint_dir / "best.pt", model, epoch=epoch, metrics=epoch_metrics, extra=extra)

            write_training_summary(
                run_dir / "training_summary.json",
                device=device,
                args=args,
                model_config=model_config,
                train_sessions=train_sessions,
                val_sessions=val_sessions,
                task_names=task_names,
                history=history,
                best_epoch=best_epoch,
                best_metric=float(best_metric),
                interrupted=False,
            )

            bound = (
                "data-bound"
                if train_metrics["avg_data_time_ms"] > train_metrics["avg_compute_time_ms"]
                else "compute-bound"
            )
            print(
                f"[train] epoch {epoch:03d} "
                f"obj={train_metrics['objective_loss']:.4f} "
                f"recon={train_metrics['recon_loss']:.4f} "
                f"first={train_metrics['first_action_loss']:.4f} "
                f"kl={train_metrics['kl_loss']:.4f} "
                f"val_obj={val_metrics['objective_loss']:.4f} "
                f"sample/s={train_metrics['samples_per_sec']:.1f} "
                f"step_ms={train_metrics['avg_step_time_ms']:.1f} "
                f"data_ms={train_metrics['avg_data_time_ms']:.1f} "
                f"compute_ms={train_metrics['avg_compute_time_ms']:.1f} "
                f"gpu_gb={train_metrics['gpu_peak_allocated_gb']:.2f}/{train_metrics['gpu_peak_reserved_gb']:.2f} "
                f"{bound}"
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[train] Interrupted by user.")
    finally:
        write_training_summary(
            run_dir / "training_summary.json",
            device=device,
            args=args,
            model_config=model_config,
            train_sessions=train_sessions,
            val_sessions=val_sessions,
            task_names=task_names,
            history=history,
            best_epoch=best_epoch,
            best_metric=float(best_metric),
            interrupted=interrupted,
        )
        print(f"[train] Saved checkpoints to {checkpoint_dir}")
        print(f"[train] Best epoch: {best_epoch}")


if __name__ == "__main__":
    main()
