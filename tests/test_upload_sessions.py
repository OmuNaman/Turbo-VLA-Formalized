from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.upload_hf_session import discover_sessions


class UploadSessionDiscoveryTests(unittest.TestCase):
    def test_discover_sessions_scans_data_root_and_preserves_task_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            session_dir = data_root / "turbopi_intent_cnn" / "episodes" / "session_20260409_130000"
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "session_info.json").write_text(
                json.dumps({"fps": 5, "intent_mode": "language"}),
                encoding="utf-8",
            )

            episode_dir = session_dir / "episode_000000"
            episode_dir.mkdir()
            pd.DataFrame({"frame_index": [0, 1], "task": ["go left", "go left"]}).to_parquet(
                episode_dir / "data.parquet",
                index=False,
            )
            (episode_dir / "episode_info.json").write_text(
                json.dumps({"task_name": "go left"}),
                encoding="utf-8",
            )

            sessions = discover_sessions(data_root)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].dataset_name, "turbopi_intent_cnn")
            self.assertEqual(sessions[0].labels, ("go left",))
            self.assertAlmostEqual(sessions[0].duration_s, 0.4)


if __name__ == "__main__":
    unittest.main()
