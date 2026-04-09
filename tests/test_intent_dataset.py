from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from intent_cnn_policy.dataset import discover_intent_episodes


def _write_episode(episode_dir: Path, *, task: str, task_index: int) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "task": [task],
            "task_index": [task_index],
            "action": [[0.0, 0.0, 0.0]],
        }
    ).to_parquet(episode_dir / "data.parquet", index=False)
    (episode_dir / "video.mp4").write_bytes(b"stub")


class IntentDatasetDiscoveryTests(unittest.TestCase):
    def test_discover_intent_episodes_skips_no_language_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            language_session = root / "session_language"
            language_session.mkdir()
            (language_session / "session_info.json").write_text(
                json.dumps({"mode_family": "cnn", "intent_mode": "language"}),
                encoding="utf-8",
            )
            _write_episode(language_session / "episode_000000", task="go left", task_index=0)
            (language_session / "episode_000000" / "episode_info.json").write_text(
                json.dumps({"mode_family": "cnn", "intent_mode": "language", "task_name": "go left"}),
                encoding="utf-8",
            )

            loop_session = root / "session_loop"
            loop_session.mkdir()
            (loop_session / "session_info.json").write_text(
                json.dumps({"mode_family": "cnn", "intent_mode": "no_language"}),
                encoding="utf-8",
            )
            _write_episode(loop_session / "episode_000000", task="clockwise", task_index=0)
            (loop_session / "episode_000000" / "episode_info.json").write_text(
                json.dumps({"mode_family": "cnn", "intent_mode": "no_language", "direction": "clockwise"}),
                encoding="utf-8",
            )

            records = discover_intent_episodes(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].task, "go left")


if __name__ == "__main__":
    unittest.main()
