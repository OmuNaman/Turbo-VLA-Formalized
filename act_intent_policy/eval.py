"""Evaluate a trained ACT-style task-conditioned TurboPi policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from intent_cnn_policy.dataset import discover_intent_episodes, discover_task_names, split_sessions

from . import DEFAULT_DATA_ROOT
from .dataset import ActEpisodeDataset
from .model import load_checkpoint
from .train import evaluate_model, resolve_amp_enabled, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the TurboPi ACT-style task-conditioned policy")
    parser.add_argument("--episodes-dir", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def build_loader(
    episodes_dir: Path,
    *,
    split: str,
    val_ratio: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    model,
    task_names: list[str],
) -> DataLoader | None:
    records = discover_intent_episodes(episodes_dir)
    dataset = ActEpisodeDataset(
        split_sessions(records, split=split, val_ratio=val_ratio, seed=seed),
        task_names,
        split=split,
        image_size=(model.config.image_width, model.config.image_height),
        history=model.config.frame_history,
        chunk_size=model.config.chunk_size,
        augment=False,
    )
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model, payload = load_checkpoint(Path(args.checkpoint), map_location=device)
    model = model.to(device)

    episodes_dir = Path(args.episodes_dir)
    task_names = list(payload.get("extra", {}).get("task_names") or [])
    if not task_names:
        task_names = discover_task_names(episodes_dir, discover_intent_episodes(episodes_dir))

    loader = build_loader(
        episodes_dir,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=model,
        task_names=task_names,
    )
    if loader is None:
        raise RuntimeError(f"No sessions available for split '{args.split}'.")

    huber_delta = float(payload.get("extra", {}).get("huber_delta", 0.25))
    metrics = evaluate_model(
        model,
        loader,
        device=device,
        huber_delta=huber_delta,
        amp_enabled=resolve_amp_enabled(device, None),
        show_progress=False,
        epoch=1,
        epochs=1,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "device": str(device),
        "task_names": task_names,
        "huber_delta": huber_delta,
        "metrics": metrics,
        "checkpoint_epoch": payload.get("epoch"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
