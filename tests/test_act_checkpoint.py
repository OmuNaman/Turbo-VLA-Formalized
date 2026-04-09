from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from act_intent_policy.model import ActIntentConfig, build_model, load_checkpoint


class ActCheckpointTests(unittest.TestCase):
    def test_load_checkpoint_ignores_unknown_config_keys_and_runs_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            model = build_model(
                ActIntentConfig(
                    image_width=64,
                    image_height=48,
                    task_vocab_size=2,
                    chunk_size=4,
                    d_model=32,
                    latent_dim=8,
                    n_heads=4,
                )
            )
            torch.save(
                {
                    "epoch": 1,
                    "metrics": {"loss": 0.1},
                    "model_config": {
                        "image_width": 64,
                        "image_height": 48,
                        "task_vocab_size": 2,
                        "chunk_size": 4,
                        "d_model": 32,
                        "latent_dim": 8,
                        "future_config_key": "ignored",
                    },
                    "model_state_dict": model.state_dict(),
                },
                path,
            )

            loaded, payload = load_checkpoint(path)
            images = torch.randn(
                2,
                loaded.config.input_channels,
                loaded.config.image_height,
                loaded.config.image_width,
            )
            task_ids = torch.tensor([0, 1], dtype=torch.long)
            pred = loaded(images, task_ids)

            self.assertEqual(loaded.config.task_vocab_size, 2)
            self.assertEqual(loaded.config.chunk_size, 4)
            self.assertIn("future_config_key", payload["model_config"])
            self.assertFalse(hasattr(loaded.config, "future_config_key"))
            self.assertEqual(tuple(pred.shape), (2, 4, 3))


if __name__ == "__main__":
    unittest.main()
