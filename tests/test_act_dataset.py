from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from act_intent_policy.dataset import ActEpisodeDataset, build_action_chunk
from intent_cnn_policy.dataset import EpisodeRecord


class ActDatasetTests(unittest.TestCase):
    def test_build_action_chunk_pads_tail_with_mask(self) -> None:
        actions = np.asarray(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.1],
                [0.3, 0.0, 0.2],
            ],
            dtype=np.float32,
        )

        chunk, mask = build_action_chunk(actions, start_index=2, chunk_size=4)

        self.assertEqual(chunk.shape, (4, 3))
        self.assertEqual(mask.tolist(), [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(chunk[0], actions[2])
        np.testing.assert_allclose(chunk[1], actions[2])

    def test_dataset_returns_chunked_targets_and_first_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "episode_000000"
            episode_dir.mkdir(parents=True, exist_ok=True)
            actions = np.asarray(
                [
                    [0.1, 0.0, 0.0],
                    [0.2, 0.0, 0.1],
                    [0.3, 0.0, 0.2],
                ],
                dtype=np.float32,
            )
            pd.DataFrame({"action": actions.tolist()}).to_parquet(episode_dir / "data.parquet", index=False)

            record = EpisodeRecord(
                episode_dir=episode_dir,
                session_name="session_1",
                num_frames=3,
                task="go left",
                task_index_hint=0,
            )
            dataset = ActEpisodeDataset(
                [record],
                ["go left"],
                split="train",
                image_size=(32, 24),
                history=3,
                chunk_size=4,
                augment=False,
            )

            frames = [
                np.zeros((24, 32, 3), dtype=np.uint8),
                np.full((24, 32, 3), 64, dtype=np.uint8),
                np.full((24, 32, 3), 128, dtype=np.uint8),
            ]
            dataset.cache.get = Mock(return_value=(frames, actions))

            item = dataset[1]

            self.assertEqual(tuple(item["image"].shape), (9, 24, 32))
            self.assertEqual(int(item["task_index"].item()), 0)
            self.assertEqual(tuple(item["action_chunk"].shape), (4, 3))
            self.assertEqual(item["action_mask"].tolist(), [1.0, 1.0, 0.0, 0.0])
            np.testing.assert_allclose(item["first_action"].numpy(), actions[1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
