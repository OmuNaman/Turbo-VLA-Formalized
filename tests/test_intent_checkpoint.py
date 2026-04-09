from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from intent_cnn_policy.model import IntentCNNConfig, build_model, load_checkpoint


class IntentCheckpointTests(unittest.TestCase):
    def test_load_checkpoint_ignores_unknown_config_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            model = build_model(IntentCNNConfig(task_vocab_size=2))
            torch.save(
                {
                    "epoch": 1,
                    "metrics": {"loss": 0.1},
                    "model_config": {
                        "task_vocab_size": 2,
                        "hidden_dim": 64,
                        "future_config_key": "ignored",
                    },
                    "model_state_dict": model.state_dict(),
                },
                path,
            )

            loaded, payload = load_checkpoint(path)

            self.assertEqual(loaded.config.task_vocab_size, 2)
            self.assertEqual(loaded.config.hidden_dim, 64)
            self.assertIn("future_config_key", payload["model_config"])
            self.assertFalse(hasattr(loaded.config, "future_config_key"))


if __name__ == "__main__":
    unittest.main()
