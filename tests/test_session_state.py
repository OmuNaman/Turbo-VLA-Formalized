from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from client.session_state import inspect_saved_session


class SessionStateTests(unittest.TestCase):
    def test_resume_state_ignores_incomplete_episodes_and_uses_raw_attempt_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes_dir = root / "episodes" / "session_20260409_120000"
            raw_dir = root / "raw" / "session_20260409_120000"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)

            complete = episodes_dir / "episode_000000"
            complete.mkdir()
            pd.DataFrame({"frame_index": [0, 1, 2]}).to_parquet(complete / "data.parquet", index=False)
            (complete / "video.mp4").write_bytes(b"ok")
            (complete / "episode_info.json").write_text("{}", encoding="utf-8")

            incomplete = episodes_dir / "episode_000001"
            incomplete.mkdir()
            (incomplete / "video.mp4").write_bytes(b"incomplete")

            (raw_dir / "telemetry.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"episode_idx": 0}),
                        json.dumps({"episode_idx": 2}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            state = inspect_saved_session(episodes_dir, raw_dir)

            self.assertEqual(state.accepted_count, 1)
            self.assertEqual(state.total_frames, 3)
            self.assertEqual(state.next_episode_index, 3)


if __name__ == "__main__":
    unittest.main()
