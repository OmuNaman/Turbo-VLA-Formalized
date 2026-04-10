from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

from act_intent_policy.cache import build_cache
from act_intent_policy.dataset import ActEpisodeDataset, build_cached_datasets
from act_intent_policy.model import ActIntentConfig, build_model
from act_intent_policy.train import build_loaders, train_epoch
from intent_cnn_policy.dataset import EpisodeRecord, discover_intent_episodes


def _write_video(video_path: Path, frames: list[np.ndarray]) -> None:
    container = av.open(str(video_path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _write_episode(
    session_dir: Path,
    episode_name: str,
    *,
    task: str,
    task_index: int,
    frames: list[np.ndarray],
    actions: np.ndarray,
) -> None:
    episode_dir = session_dir / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)
    _write_video(episode_dir / "video.mp4", frames)
    df = pd.DataFrame(
        {
            "frame_index": list(range(len(actions))),
            "timestamp": [float(index) * 0.1 for index in range(len(actions))],
            "episode_index": [0] * len(actions),
            "task_index": [task_index] * len(actions),
            "task": [task] * len(actions),
            "observation.state": actions.tolist(),
            "action": actions.tolist(),
        }
    )
    df.to_parquet(episode_dir / "data.parquet", index=False)
    (episode_dir / "episode_info.json").write_text(
        json.dumps(
            {
                "task": task,
                "task_index": task_index,
                "frame_count": len(actions),
                "intent_mode": "language",
                "mode_family": "cnn",
                "task_type": "instruction_conditioned_path_following",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _create_intent_root(root: Path) -> Path:
    episodes_root = root / "episodes"
    session_dir = episodes_root / "session_20260410_000000"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "tasks.json").write_text(json.dumps(["go left", "go right"], indent=2), encoding="utf-8")
    (session_dir / "session_info.json").write_text(
        json.dumps(
            {
                "intent_mode": "language",
                "mode_family": "cnn",
                "task_type": "instruction_conditioned_path_following",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    frames_left = [
        np.full((24, 32, 3), 16, dtype=np.uint8),
        np.full((24, 32, 3), 64, dtype=np.uint8),
        np.full((24, 32, 3), 128, dtype=np.uint8),
    ]
    actions_left = np.asarray(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.1],
            [0.3, 0.0, 0.2],
        ],
        dtype=np.float32,
    )
    _write_episode(
        session_dir,
        "episode_000000",
        task="go left",
        task_index=0,
        frames=frames_left,
        actions=actions_left,
    )

    frames_right = [
        np.full((24, 32, 3), 32, dtype=np.uint8),
        np.full((24, 32, 3), 96, dtype=np.uint8),
        np.full((24, 32, 3), 192, dtype=np.uint8),
    ]
    actions_right = np.asarray(
        [
            [0.0, 0.0, -0.1],
            [0.1, 0.0, -0.2],
            [0.2, 0.0, -0.3],
        ],
        dtype=np.float32,
    )
    _write_episode(
        session_dir,
        "episode_000001",
        task="go right",
        task_index=1,
        frames=frames_right,
        actions=actions_right,
    )
    return episodes_root


class ActCacheTests(unittest.TestCase):
    def test_cache_build_matches_raw_dataset_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes_root = _create_intent_root(root)
            cache_dir = root / "act_cache"
            build_cache(
                episodes_root,
                cache_dir,
                image_width=32,
                image_height=24,
                frame_history=3,
                chunk_size=4,
                show_progress=False,
            )

            records = discover_intent_episodes(episodes_root)
            raw_dataset = ActEpisodeDataset(
                records,
                ["go left", "go right"],
                split="train",
                image_size=(32, 24),
                history=3,
                chunk_size=4,
                augment=False,
            )
            cached_dataset, cached_val_dataset, task_names = build_cached_datasets(cache_dir, val_ratio=0.2, seed=42)

            self.assertEqual(task_names, ["go left", "go right"])
            self.assertEqual(len(raw_dataset), len(cached_dataset))

            raw_item = raw_dataset[1]
            cached_item = cached_dataset[1]

            cached_image = cached_item["image"].to(dtype=torch.float32) / 255.0
            self.assertEqual(tuple(cached_item["image"].shape), tuple(raw_item["image"].shape))
            np.testing.assert_allclose(raw_item["image"].numpy(), cached_image.numpy(), atol=1.0 / 255.0)
            self.assertEqual(int(raw_item["task_index"].item()), int(cached_item["task_index"].item()))
            np.testing.assert_allclose(raw_item["action_chunk"].numpy(), cached_item["action_chunk"].numpy(), atol=1e-6)
            np.testing.assert_allclose(raw_item["action_mask"].numpy(), cached_item["action_mask"].numpy(), atol=1e-6)
            np.testing.assert_allclose(raw_item["first_action"].numpy(), cached_item["first_action"].numpy(), atol=1e-6)
            del cached_image
            del cached_item
            cached_dataset.close()
            cached_val_dataset.close()
            del cached_dataset
            del cached_val_dataset
            gc.collect()

    def test_build_loaders_uses_cache_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes_root = _create_intent_root(root)
            cache_dir = root / "act_cache"
            build_cache(
                episodes_root,
                cache_dir,
                image_width=32,
                image_height=24,
                frame_history=3,
                chunk_size=4,
                show_progress=False,
            )

            train_loader, val_loader, _, _, task_names, data_source, resolved_cache_dir = build_loaders(
                episodes_root,
                val_ratio=0.2,
                seed=42,
                batch_size=2,
                num_workers=0,
                frame_history=3,
                image_width=32,
                image_height=24,
                chunk_size=4,
                cache_size=1,
                cache_dir=cache_dir,
                cache_mode="require",
                preload_all=False,
                preload_threshold_records=64,
                preload_threshold_frames=25000,
                prefetch_factor=2,
                device=torch.device("cpu"),
                show_progress=False,
            )

            self.assertEqual(data_source, "cache")
            self.assertEqual(resolved_cache_dir, cache_dir)
            self.assertEqual(task_names, ["go left", "go right"])
            self.assertEqual(train_loader.dataset.data_source, "cache")
            self.assertIsNone(val_loader)
            train_loader.dataset.close()
            del train_loader
            gc.collect()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for ACT cache smoke test")
    def test_cached_train_epoch_runs_on_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes_root = _create_intent_root(root)
            cache_dir = root / "act_cache"
            build_cache(
                episodes_root,
                cache_dir,
                image_width=32,
                image_height=24,
                frame_history=3,
                chunk_size=4,
                show_progress=False,
            )

            train_loader, _, _, _, task_names, _, _ = build_loaders(
                episodes_root,
                val_ratio=0.2,
                seed=42,
                batch_size=2,
                num_workers=0,
                frame_history=3,
                image_width=32,
                image_height=24,
                chunk_size=4,
                cache_size=1,
                cache_dir=cache_dir,
                cache_mode="require",
                preload_all=False,
                preload_threshold_records=64,
                preload_threshold_frames=25000,
                prefetch_factor=2,
                device=torch.device("cuda"),
                show_progress=False,
            )

            model = build_model(
                ActIntentConfig(
                    image_width=32,
                    image_height=24,
                    frame_history=3,
                    task_vocab_size=len(task_names),
                    chunk_size=4,
                    d_model=32,
                    latent_dim=8,
                )
            ).to("cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda", enabled=True)

            metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler=scaler,
                device=torch.device("cuda"),
                huber_delta=0.25,
                kl_scale=1e-3,
                grad_clip=1.0,
                amp_enabled=True,
                log_interval=1,
                show_progress=False,
                epoch=1,
                epochs=1,
                lr=1e-3,
            )

            self.assertGreater(metrics["samples_per_sec"], 0.0)
            self.assertGreater(metrics["gpu_peak_allocated_gb"], 0.0)
            self.assertEqual(next(model.parameters()).device.type, "cuda")
            train_loader.dataset.close()
            del train_loader
            gc.collect()


if __name__ == "__main__":
    unittest.main()
