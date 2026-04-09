from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tasks import build_task_manager


class TaskManagerTests(unittest.TestCase):
    def test_build_task_manager_preserves_saved_session_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            (session_dir / "tasks.json").write_text(
                json.dumps({"0": "saved left", "1": "saved right"}, indent=2),
                encoding="utf-8",
            )

            manager = build_task_manager(
                session_dir,
                ["go forward", "saved right", "go backward"],
            )

            self.assertEqual(
                manager.tasks,
                ["saved left", "saved right", "go forward", "go backward"],
            )


if __name__ == "__main__":
    unittest.main()
