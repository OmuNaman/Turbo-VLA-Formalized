"""Train the task-conditioned TurboPi CNN policy."""

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
    DEFAULT_DATA_ROOT,
    DEFAULT_FRAME_HISTORY,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
)
from .dataset import build_datasets
from .model import IntentCNNConfig, build_model, save_checkpoint


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


def write_training_summary(
    path: Path,
    *,
    device: torch.device,
    args,
    model_config: IntentCNNConfig,
    train_sessions: list[str],
    val_sessions: list[str],
    task_names: list[str],
    history: list[dict[str, float]],
    best_epoch: int,
    best_metric: float,
    interrupted: bool,
) -> None:
    """Persist the current training state so interrupted runs still keep history."""
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the TurboPi task-conditioned CNN policy")
    parser.add_argument("--episodes-dir", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--run-dir",
        default="runs/intent_cnn_v1",
        help="Base directory for training runs; each launch creates a timestamped child run",
    )
    parser.add_argument("--epochs", type=int, default=20)
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
    parser.add_argument("--task-embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return parser


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
) -> tuple[DataLoader, DataLoader | None, list[str], list[str], list[str]]:
    train_dataset, val_dataset, task_names = build_datasets(
        episodes_dir=episodes_dir,
        image_size=(image_width, image_height),
        history=frame_history,
        val_ratio=val_ratio,
        seed=seed,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(f"No intent-conditioned CNN episodes found under {episodes_dir}")

    preload_threshold_frames = 25000
    preload_threshold_records = 64
    if train_dataset.records and len(train_dataset.records) <= preload_threshold_records and train_dataset.total_frames <= preload_threshold_frames:
        estimated_gb = train_dataset.estimated_cache_bytes / (1024 ** 3)
        print(
            f"[train] Preloading {len(train_dataset.records)} train episodes into RAM "
            f"(~{estimated_gb:.2f} GB resized frames) to avoid repeated video decode."
        )
        train_dataset.preload_all()
        if len(val_dataset.records) > 0:
            val_dataset.preload_all()

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


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader | None, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    if loader is None:
        return {"loss": math.nan, "mae_vx": math.nan, "mae_vy": math.nan, "mae_omega": math.nan}

    model.eval()
    total_loss = 0.0
    total_examples = 0
    abs_error = torch.zeros(3, dtype=torch.float64)

    for batch in loader:
        images = batch["image"].to(device)
        task_ids = batch["task_index"].to(device)
        targets = batch["action"].to(device)
        preds = model(images, task_ids)
        loss = criterion(preds, targets)
        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        abs_error += torch.abs(preds - targets).sum(dim=0).double().cpu()

    if total_examples == 0:
        return {"loss": math.nan, "mae_vx": math.nan, "mae_vy": math.nan, "mae_omega": math.nan}

    return {
        "loss": total_loss / total_examples,
        "mae_vx": float(abs_error[0].item() / total_examples),
        "mae_vy": float(abs_error[1].item() / total_examples),
        "mae_omega": float(abs_error[2].item() / total_examples),
    }


@torch.no_grad()
def evaluate_model_with_progress(
    model: nn.Module,
    loader: DataLoader | None,
    criterion: nn.Module,
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
    show_progress: bool,
) -> dict[str, float]:
    if loader is None:
        return {"loss": math.nan, "mae_vx": math.nan, "mae_vy": math.nan, "mae_omega": math.nan}

    if not show_progress:
        return evaluate_model(model, loader, criterion, device)

    model.eval()
    total_loss = 0.0
    total_examples = 0
    abs_error = torch.zeros(3, dtype=torch.float64)
    start_time = time.perf_counter()

    bar = tqdm(
        total=len(loader.dataset),
        desc=f"Epoch {epoch:03d}/{epochs:03d} val",
        unit="sample",
        dynamic_ncols=True,
        leave=False,
    )

    try:
        for batch in loader:
            images = batch["image"].to(device)
            task_ids = batch["task_index"].to(device)
            targets = batch["action"].to(device)
            preds = model(images, task_ids)
            loss = criterion(preds, targets)

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            abs_error += torch.abs(preds - targets).sum(dim=0).double().cpu()

            elapsed = max(time.perf_counter() - start_time, 1e-6)
            bar.update(batch_size)
            bar.set_postfix(
                batch_loss=f"{float(loss.item()):.4f}",
                avg_loss=f"{total_loss / total_examples:.4f}",
                samples_per_s=f"{total_examples / elapsed:.1f}",
            )
    finally:
        bar.close()

    if total_examples == 0:
        return {"loss": math.nan, "mae_vx": math.nan, "mae_vy": math.nan, "mae_omega": math.nan}

    return {
        "loss": total_loss / total_examples,
        "mae_vx": float(abs_error[0].item() / total_examples),
        "mae_vy": float(abs_error[1].item() / total_examples),
        "mae_omega": float(abs_error[2].item() / total_examples),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
    lr: float,
    show_progress: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    abs_error = torch.zeros(3, dtype=torch.float64)
    start_time = time.perf_counter()

    bar = None
    if show_progress:
        bar = tqdm(
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
            targets = batch["action"].to(device)
            preds = model(images, task_ids)
            loss = criterion(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            abs_error += torch.abs(preds.detach() - targets).sum(dim=0).double().cpu()

            if bar is not None:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                bar.update(batch_size)
                bar.set_postfix(
                    batch_loss=f"{float(loss.item()):.4f}",
                    avg_loss=f"{total_loss / total_examples:.4f}",
                    lr=f"{lr:.2e}",
                    samples_per_s=f"{total_examples / elapsed:.1f}",
                )
    finally:
        if bar is not None:
            bar.close()

    denom = max(1, total_examples)
    return {
        "loss": total_loss / denom,
        "mae_vx": float(abs_error[0].item() / denom),
        "mae_vy": float(abs_error[1].item() / denom),
        "mae_omega": float(abs_error[2].item() / denom),
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

    model_config = IntentCNNConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        frame_history=args.frame_history,
        task_embedding_dim=args.task_embedding_dim,
        task_vocab_size=max(1, len(task_names)),
        hidden_dim=args.hidden_dim,
    )
    model = build_model(model_config).to(device)
    criterion = nn.HuberLoss(delta=args.huber_delta)
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
    }

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch=epoch,
                epochs=args.epochs,
                lr=args.lr,
                show_progress=show_progress,
            )
            val_metrics = evaluate_model_with_progress(
                model,
                val_loader,
                criterion,
                device,
                epoch=epoch,
                epochs=args.epochs,
                show_progress=show_progress,
            )

            metric = val_metrics["loss"]
            if math.isnan(metric):
                metric = train_metrics["loss"]

            epoch_metrics = {
                "epoch": float(epoch),
                "train_loss": float(train_metrics["loss"]),
                "train_mae_vx": float(train_metrics["mae_vx"]),
                "train_mae_vy": float(train_metrics["mae_vy"]),
                "train_mae_omega": float(train_metrics["mae_omega"]),
                "val_loss": float(val_metrics["loss"]),
                "val_mae_vx": float(val_metrics["mae_vx"]),
                "val_mae_vy": float(val_metrics["mae_vy"]),
                "val_mae_omega": float(val_metrics["mae_omega"]),
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
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_mae=[{val_metrics['mae_vx']:.4f}, {val_metrics['mae_vy']:.4f}, {val_metrics['mae_omega']:.4f}]"
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
