"""Build a reusable ACT sample cache for fast GPU training."""

from __future__ import annotations

import argparse
import os
import gc
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm

from intent_cnn_policy.dataset import discover_intent_episodes, discover_task_names

from . import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_FRAME_HISTORY,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
)
from .dataset import (
    ACT_CACHE_ACTION_CHUNK,
    ACT_CACHE_ACTION_MASK,
    ACT_CACHE_FIRST_ACTION,
    ACT_CACHE_IMAGES,
    ACT_CACHE_MANIFEST,
    ACT_CACHE_SAMPLE_EPISODE_INDEX,
    ACT_CACHE_SAMPLE_WEIGHT,
    ACT_CACHE_TASK_INDEX,
    ACT_CACHE_VERSION,
    build_action_chunk,
    compute_sample_weight,
    load_act_cache_manifest,
    load_episode_actions,
    load_episode_frames,
    validate_act_cache_manifest,
)


def _default_cache_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


def _prepare_cached_episode(
    *,
    episode_index: int,
    episode_dir: str,
    session_name: str,
    num_frames: int,
    task: str,
    task_index: int,
    image_width: int,
    image_height: int,
    frame_history: int,
    chunk_size: int,
    task_names: list[str],
    task_counts: dict[str, int],
) -> dict[str, object]:
    """Decode and precompute all ACT samples for one episode."""
    episode_path = Path(episode_dir)
    frames = load_episode_frames(episode_path / "video.mp4", image_size=(image_width, image_height))
    actions = load_episode_actions(episode_path / "data.parquet")
    if len(frames) != len(actions):
        raise ValueError(f"Episode {episode_path} has {len(frames)} frames but {len(actions)} actions.")

    frame_chw = np.stack([frame.transpose(2, 0, 1) for frame in frames], axis=0)
    sample_count = len(actions)
    action_dim = actions.shape[1] if actions.ndim == 2 and len(actions) > 0 else 3

    images = np.empty((sample_count, frame_history * 3, image_height, image_width), dtype=np.uint8)
    action_chunk = np.empty((sample_count, chunk_size, action_dim), dtype=np.float32)
    action_mask = np.empty((sample_count, chunk_size), dtype=np.float32)
    first_action = np.empty((sample_count, action_dim), dtype=np.float32)
    sample_weight = np.empty((sample_count,), dtype=np.float32)

    for frame_idx, action in enumerate(actions):
        frame_indices = [max(0, frame_idx - offset) for offset in reversed(range(frame_history))]
        images[frame_idx] = frame_chw[frame_indices].reshape(frame_history * 3, image_height, image_width)
        chunk, mask = build_action_chunk(actions, start_index=frame_idx, chunk_size=chunk_size)
        action_chunk[frame_idx] = chunk
        action_mask[frame_idx] = mask
        first_action[frame_idx] = chunk[0]
        sample_weight[frame_idx] = compute_sample_weight(
            action,
            task=task,
            task_names=task_names,
            task_counts=task_counts,
        )

    return {
        "episode_index": episode_index,
        "episode_dir": episode_dir,
        "session_name": session_name,
        "num_frames": num_frames,
        "task": task,
        "task_index": task_index,
        "images": images,
        "action_chunk": action_chunk,
        "action_mask": action_mask,
        "first_action": first_action,
        "sample_weight": sample_weight,
    }


def default_cache_dir(
    episodes_dir: Path,
    *,
    image_width: int,
    image_height: int,
    frame_history: int,
    chunk_size: int,
) -> Path:
    """Place the cache beside the dataset root by default."""
    base_dir = episodes_dir.parent if episodes_dir.name == "episodes" else episodes_dir
    suffix = f"act_cache_w{image_width}_h{image_height}_hist{frame_history}_chunk{chunk_size}"
    return base_dir / suffix


def cache_is_compatible(
    cache_dir: Path,
    *,
    image_width: int,
    image_height: int,
    frame_history: int,
    chunk_size: int,
) -> bool:
    """Return whether an on-disk ACT cache matches the requested config."""
    try:
        manifest = load_act_cache_manifest(cache_dir)
        validate_act_cache_manifest(
            manifest,
            image_size=(image_width, image_height),
            history=frame_history,
            chunk_size=chunk_size,
        )
        return True
    except Exception:
        return False


def build_cache(
    episodes_dir: Path | str,
    cache_dir: Path | str,
    *,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    frame_history: int = DEFAULT_FRAME_HISTORY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    workers: int | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """Materialize ACT training tensors into reusable memory-mapped arrays."""
    episodes_dir = Path(episodes_dir)
    cache_dir = Path(cache_dir)
    records = discover_intent_episodes(episodes_dir)
    if not records:
        raise RuntimeError(f"No intent-conditioned episodes found under {episodes_dir}")
    task_names = discover_task_names(episodes_dir, records)
    if not task_names:
        raise RuntimeError(f"No task vocabulary discovered under {episodes_dir}")
    workers = _default_cache_workers() if workers is None else max(1, int(workers))

    if cache_dir.exists():
        if not overwrite and cache_is_compatible(
            cache_dir,
            image_width=image_width,
            image_height=image_height,
            frame_history=frame_history,
            chunk_size=chunk_size,
        ):
            manifest = load_act_cache_manifest(cache_dir)
            print(
                f"[cache] Reusing existing ACT cache at {cache_dir} "
                f"({manifest.get('num_samples', 0)} samples)."
            )
            return manifest
        if overwrite:
            shutil.rmtree(cache_dir, ignore_errors=True)
        else:
            raise RuntimeError(
                f"Cache dir already exists and is incompatible: {cache_dir}. "
                "Pass --overwrite to rebuild it."
            )

    build_dir = cache_dir.parent / f"{cache_dir.name}.tmp_build"
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=False)

    task_to_index = {task: index for index, task in enumerate(task_names)}
    total_samples = sum(record.num_frames for record in records)
    action_dim = 3
    task_counts = {task: 0 for task in task_names}
    for record in records:
        task_counts[record.task] = task_counts.get(record.task, 0) + int(record.num_frames)

    images = open_memmap(
        build_dir / ACT_CACHE_IMAGES,
        mode="w+",
        dtype=np.uint8,
        shape=(total_samples, frame_history * 3, image_height, image_width),
    )
    task_index = open_memmap(build_dir / ACT_CACHE_TASK_INDEX, mode="w+", dtype=np.int64, shape=(total_samples,))
    action_chunk = open_memmap(
        build_dir / ACT_CACHE_ACTION_CHUNK,
        mode="w+",
        dtype=np.float32,
        shape=(total_samples, chunk_size, action_dim),
    )
    action_mask = open_memmap(
        build_dir / ACT_CACHE_ACTION_MASK,
        mode="w+",
        dtype=np.float32,
        shape=(total_samples, chunk_size),
    )
    first_action = open_memmap(
        build_dir / ACT_CACHE_FIRST_ACTION,
        mode="w+",
        dtype=np.float32,
        shape=(total_samples, action_dim),
    )
    sample_weight = open_memmap(
        build_dir / ACT_CACHE_SAMPLE_WEIGHT,
        mode="w+",
        dtype=np.float32,
        shape=(total_samples,),
    )
    sample_episode_index = open_memmap(
        build_dir / ACT_CACHE_SAMPLE_EPISODE_INDEX,
        mode="w+",
        dtype=np.int32,
        shape=(total_samples,),
    )

    episode_manifest: list[dict[str, object]] = []
    episode_offsets: list[int] = []
    cursor = 0
    for record in records:
        episode_offsets.append(cursor)
        cursor += int(record.num_frames)

    started_at = time.perf_counter()
    progress = None
    if show_progress:
        progress = tqdm(total=len(records), desc="Build ACT cache", unit="episode", dynamic_ncols=True)

    worker_payloads = [
        {
            "episode_index": episode_index,
            "episode_dir": str(record.episode_dir),
            "session_name": record.session_name,
            "num_frames": int(record.num_frames),
            "task": record.task,
            "task_index": int(task_to_index[record.task]),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "frame_history": int(frame_history),
            "chunk_size": int(chunk_size),
            "task_names": task_names,
            "task_counts": task_counts,
        }
        for episode_index, record in enumerate(records)
    ]

    def write_prepared_episode(prepared: dict[str, object]) -> None:
        episode_index = int(prepared["episode_index"])
        sample_start = episode_offsets[episode_index]
        sample_count = int(prepared["num_frames"])
        sample_end = sample_start + sample_count

        images[sample_start:sample_end] = prepared["images"]
        task_index[sample_start:sample_end] = int(prepared["task_index"])
        action_chunk[sample_start:sample_end] = prepared["action_chunk"]
        action_mask[sample_start:sample_end] = prepared["action_mask"]
        first_action[sample_start:sample_end] = prepared["first_action"]
        sample_weight[sample_start:sample_end] = prepared["sample_weight"]
        sample_episode_index[sample_start:sample_end] = episode_index

        episode_manifest.append(
            {
                "episode_index": episode_index,
                "episode_dir": str(prepared["episode_dir"]),
                "session_name": str(prepared["session_name"]),
                "num_frames": sample_count,
                "task": str(prepared["task"]),
                "task_index": int(prepared["task_index"]),
                "sample_start": sample_start,
                "sample_count": sample_count,
            }
        )

    try:
        if workers <= 1 or len(worker_payloads) <= 1:
            for payload in worker_payloads:
                write_prepared_episode(_prepare_cached_episode(**payload))
                if progress is not None:
                    progress.update(1)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_prepare_cached_episode, **payload) for payload in worker_payloads]
                for future in as_completed(futures):
                    write_prepared_episode(future.result())
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    episode_manifest.sort(key=lambda item: int(item["episode_index"]))

    for array in (images, task_index, action_chunk, action_mask, first_action, sample_weight, sample_episode_index):
        array.flush()
        mmap_obj = getattr(array, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
    del images
    del task_index
    del action_chunk
    del action_mask
    del first_action
    del sample_weight
    del sample_episode_index
    gc.collect()

    cache_bytes = sum(path.stat().st_size for path in build_dir.glob("*.npy"))
    elapsed = max(time.perf_counter() - started_at, 1e-6)
    manifest = {
        "version": ACT_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(episodes_dir),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "frame_history": int(frame_history),
        "chunk_size": int(chunk_size),
        "action_dim": int(action_dim),
        "num_samples": int(total_samples),
        "num_episodes": int(len(records)),
        "build_workers": int(workers),
        "task_names": task_names,
        "episodes": episode_manifest,
        "total_cache_bytes": int(cache_bytes),
        "build_seconds": float(elapsed),
        "samples_per_second": float(total_samples / elapsed),
    }
    (build_dir / ACT_CACHE_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    build_dir.replace(cache_dir)

    print(
        f"[cache] Wrote {total_samples} samples from {len(records)} episodes to {cache_dir} "
        f"({cache_bytes / (1024 ** 3):.2f} GB, {total_samples / elapsed:.1f} samples/s, {elapsed:.1f}s, "
        f"{workers} worker{'s' if workers != 1 else ''})."
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a reusable ACT-Intent training cache")
    parser.add_argument("--episodes-dir", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--frame-history", type=int, default=DEFAULT_FRAME_HISTORY)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--workers", type=int, default=_default_cache_workers())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episodes_dir = Path(args.episodes_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir(
        episodes_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        frame_history=args.frame_history,
        chunk_size=args.chunk_size,
    )
    manifest = build_cache(
        episodes_dir,
        cache_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        frame_history=args.frame_history,
        chunk_size=args.chunk_size,
        workers=args.workers,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(
        f"[cache] Summary: samples={manifest['num_samples']} episodes={manifest['num_episodes']} "
        f"size={manifest['total_cache_bytes'] / (1024 ** 3):.2f}GB"
    )


if __name__ == "__main__":
    main()
