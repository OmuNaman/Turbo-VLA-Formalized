"""Task definitions and management for task-conditioned recording flows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# Default tasks for the TurboPi navigation demo.
# Edit this list to match your setup and objects.
DEFAULT_TASKS = [
    "go to the left of the box",
    "go to the right of the box",
    "go forward to the box",
    "go behind the box",
]

CUSTOM_TASK_LABEL = "Custom task..."
TASK_MAPPING_FILENAME = "tasks.json"


def load_saved_tasks(path: Path) -> list[str]:
    """Load a saved task mapping from a session folder or tasks.json file."""
    mapping_path = Path(path)
    if mapping_path.is_dir():
        mapping_path = mapping_path / TASK_MAPPING_FILENAME
    if not mapping_path.exists():
        return []

    try:
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, list):
        return [str(item).strip() for item in data if str(item).strip()]
    if isinstance(data, dict):
        ordered: list[tuple[int, str]] = []
        for key, value in data.items():
            try:
                index = int(key)
            except Exception:
                continue
            task = str(value).strip()
            if task:
                ordered.append((index, task))
        return [task for _, task in sorted(ordered, key=lambda item: item[0])]
    return []


class TaskManager:
    """Manages task descriptions and their integer indices."""

    def __init__(self, tasks: list[str] | None = None):
        self.tasks: list[str] = []
        self._task_to_index: dict[str, int] = {}
        self.merge_tasks(DEFAULT_TASKS if tasks is None else tasks)

    def _normalize_task(self, task: str) -> str:
        normalized = str(task).strip()
        if not normalized:
            raise ValueError("Task text cannot be empty.")
        return normalized

    def get_task(self, index: int) -> str:
        return self.tasks[index]

    def has_task(self, task: str) -> bool:
        """Whether the given task string already exists in the current vocabulary."""
        return self._normalize_task(task) in self._task_to_index

    def get_index(self, task: str) -> int:
        normalized = self._normalize_task(task)
        if normalized not in self._task_to_index:
            self._task_to_index[normalized] = len(self.tasks)
            self.tasks.append(normalized)
        return self._task_to_index[normalized]

    def merge_tasks(self, tasks: list[str]) -> None:
        """Append unseen task strings while preserving existing order."""
        for task in tasks:
            if str(task).strip():
                self.get_index(str(task))

    def list_tasks(self) -> list[tuple[int, str]]:
        return list(enumerate(self.tasks))

    def print_tasks(self, *, include_custom_option: bool = False) -> None:
        print("\n  Available tasks:")
        for i, task in enumerate(self.tasks):
            print(f"    [{i}] {task}")
        if include_custom_option:
            print(f"    [{len(self.tasks)}] {CUSTOM_TASK_LABEL}")
        print()

    def to_parquet(self, path: Path) -> None:
        """Write meta/tasks.parquet in LeRobot v3.0 format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "task_index": range(len(self.tasks)),
            "task": self.tasks,
        })
        df.to_parquet(path, index=False)

    def __len__(self) -> int:
        return len(self.tasks)
