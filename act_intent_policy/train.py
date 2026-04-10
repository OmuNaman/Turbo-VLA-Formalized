"""Train the ACT-style task-conditioned TurboPi policy."""

from __future__ import annotations

import argparse
import json
import math
import random
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
from .dataset import build_datasets
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
    parser.add_argument("--batch-size", type=int, default=16)
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
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cache-size", type=int, default=4, help="Per-worker decoded episode cache size")
    parser.add_argument(
        "--preload-all",
        action="store_true",
        help="Force preloading all resized frames into RAM before training",
    )
    parser.add_argument(
        "--preload-threshold-records",
        type=int,
        default=64,
        help="Auto-preload when the train split has at most this many episodes",
    )
    parser.add_argument(
        "--preload-threshold-frames",
        type=int,
        default=25000,
        help="Auto-preload when the train split has at most this many frames",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return parser


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
    preload_all: bool,
    preload_threshold_records: int,
    preload_threshold_frames: int,
    show_progress: bool,
) -> tuple[DataLoader, DataLoader | None, list[str], list[str], list[str]]:
    train_dataset, val_dataset, task_names = build_datasets(
        episodes_dir=episodes_dir,
        image_size=(image_width, image_height),
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(
            weights=torch.as_tensor(train_dataset.sample_weights, dtype=torch.double),
            num_samples=len(train_dataset.sample_weights),
            replacement=True,
        ),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    train_sessions = sorted({record.session_name for record in train_dataset.records})
    val_sessions = sorted({record.session_name for record in val_dataset.records})
    return train_loader, val_loader, train_sessions, val_sessions, task_names


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


def cosine_lr(base_lr: float, *, epoch: int, epochs: int, warmup_epochs: int = 3) -> float:
    """Warmup plus cosine decay learning-rate schedule."""
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


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
    show_progress: bool,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    if loader is None:
        return {
            "loss": math.nan,
            "chunk_mae": math.nan,
            "first_mae_vx": math.nan,
            "first_mae_vy": math.nan,
            "first_mae_omega": math.nan,
        }

    model.eval()
    total_loss = 0.0
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

    try:
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                task_ids = batch["task_index"].to(device)
                target = batch["action_chunk"].to(device)
                mask = batch["action_mask"].to(device)

                pred = model(images, task_ids)
                loss = masked_huber_loss(pred, target, mask, delta=huber_delta)

                batch_size = images.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size
                total_valid_steps += float(mask.sum().item())
                total_chunk_abs += float((torch.abs(pred - target) * mask.unsqueeze(-1)).sum().item())
                first_abs += compute_first_step_mae(pred, target).double().cpu() * batch_size

                if progress is not None:
                    progress.update(batch_size)
                    progress.set_postfix(avg_loss=f"{total_loss / max(1, total_examples):.4f}")
    finally:
        if progress is not None:
            progress.close()

    if total_examples == 0:
        return {
            "loss": math.nan,
            "chunk_mae": math.nan,
            "first_mae_vx": math.nan,
            "first_mae_vy": math.nan,
            "first_mae_omega": math.nan,
        }

    denom = max(total_valid_steps * target.shape[-1], 1.0)
    return {
        "loss": total_loss / total_examples,
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
    device: torch.device,
    huber_delta: float,
    kl_scale: float,
    grad_clip: float,
    show_progress: bool,
    epoch: int,
    epochs: int,
    lr: float,
) -> dict[str, float]:
    model.train()
    total_examples = 0
    total_recon = 0.0
    total_kl = 0.0
    total_chunk_abs = 0.0
    total_valid_steps = 0.0
    first_abs = torch.zeros(3, dtype=torch.float64)

    progress = None
    if show_progress:
        progress = tqdm(
            total=len(loader.dataset),
            desc=f"Epoch {epoch:03d}/{epochs:03d} train",
            unit="sample",
            dynamic_ncols=True,
            leave=False,
        )

    try:
        for batch in loader:
            images = batch["image"].to(device)
            task_ids = batch["task_index"].to(device)
            target = batch["action_chunk"].to(device)
            mask = batch["action_mask"].to(device)

            pred, mu, logvar = model(images, task_ids, target, mask)
            recon_loss = masked_huber_loss(pred, target, mask, delta=huber_delta)
            kl_loss = kl_divergence(mu, logvar)
            loss = recon_loss + kl_scale * kl_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            batch_size = images.shape[0]
            total_examples += batch_size
            total_recon += float(recon_loss.item()) * batch_size
            total_kl += float(kl_loss.item()) * batch_size
            total_valid_steps += float(mask.sum().item())
            total_chunk_abs += float((torch.abs(pred.detach() - target) * mask.unsqueeze(-1)).sum().item())
            first_abs += compute_first_step_mae(pred.detach(), target).double().cpu() * batch_size

            if progress is not None:
                progress.update(batch_size)
                progress.set_postfix(
                    recon=f"{total_recon / max(1, total_examples):.4f}",
                    kl=f"{total_kl / max(1, total_examples):.4f}",
                    lr=f"{lr:.2e}",
                )
    finally:
        if progress is not None:
            progress.close()

    denom = max(total_valid_steps * target.shape[-1], 1.0)
    return {
        "recon_loss": total_recon / max(1, total_examples),
        "kl_loss": total_kl / max(1, total_examples),
        "chunk_mae": total_chunk_abs / denom,
        "first_mae_vx": float(first_abs[0].item() / max(1, total_examples)),
        "first_mae_vy": float(first_abs[1].item() / max(1, total_examples)),
        "first_mae_omega": float(first_abs[2].item() / max(1, total_examples)),
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    episodes_dir = Path(args.episodes_dir)
    run_base_dir = Path(args.run_dir)
    run_dir = resolve_run_dir(run_base_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_sessions, val_sessions, task_names = build_loaders(
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
        preload_all=args.preload_all,
        preload_threshold_records=args.preload_threshold_records,
        preload_threshold_frames=args.preload_threshold_frames,
        show_progress=not args.no_progress,
    )

    train_dataset = train_loader.dataset
    val_dataset = None if val_loader is None else val_loader.dataset

    if not val_sessions:
        print("[train] WARNING: Only one session available. Validation will be skipped.")

    print(f"[train] Sessions: {', '.join(train_sessions) if train_sessions else '(none)'}")
    print(f"[train] Device: {device}")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
        "kl_weight": float(args.kl_weight),
        "kl_warmup_epochs": int(args.kl_warmup_epochs),
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
                device=device,
                huber_delta=args.huber_delta,
                kl_scale=kl_scale,
                grad_clip=args.grad_clip,
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
                show_progress=show_progress,
                epoch=epoch,
                epochs=args.epochs,
            )

            metric = val_metrics["loss"]
            if math.isnan(metric):
                metric = train_metrics["recon_loss"]

            epoch_metrics = {
                "epoch": float(epoch),
                "train_recon_loss": float(train_metrics["recon_loss"]),
                "train_kl_loss": float(train_metrics["kl_loss"]),
                "train_chunk_mae": float(train_metrics["chunk_mae"]),
                "train_first_mae_vx": float(train_metrics["first_mae_vx"]),
                "train_first_mae_vy": float(train_metrics["first_mae_vy"]),
                "train_first_mae_omega": float(train_metrics["first_mae_omega"]),
                "val_loss": float(val_metrics["loss"]),
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

            print(
                f"[train] epoch {epoch:03d} "
                f"recon={train_metrics['recon_loss']:.4f} "
                f"kl={train_metrics['kl_loss']:.4f} "
                f"val={val_metrics['loss']:.4f} "
                f"val_first_mae=[{val_metrics['first_mae_vx']:.4f}, {val_metrics['first_mae_vy']:.4f}, {val_metrics['first_mae_omega']:.4f}]"
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
