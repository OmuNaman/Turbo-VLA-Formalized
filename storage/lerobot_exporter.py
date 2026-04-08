"""Utilities for exporting TurboPi episodes into LeRobot dataset format."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import av
except ImportError as exc:  # pragma: no cover - exercised on machines without PyAV
    raise RuntimeError("PyAV is required for LeRobot export. Install it with `pip install av`.") from exc

from lerobot.datasets.lerobot_dataset import LeRobotDataset


TASK_COLUMN = "task"
ACTION_COLUMN = "action"
STATE_COLUMN = "observation.state"
DEFAULT_IMAGE_KEY = "observation.images.front"
ACTION_NAMES = ["vx", "vy", "omega"]
STATE_SOURCE_CHOICES = {"shifted_action", "recorded", "zeros", "none"}


@dataclass
class ExportSummary:
    """Summary returned after a successful export."""

    repo_id: str
    output_dir: Path
    num_episodes: int
    num_frames: int
    image_key: str
    state_source: str


def export_lerobot_dataset(
    episodes_dir: Path,
    output_dir: Path,
    repo_id: str,
    robot_type: str,
    fps: int,
    image_key: str = DEFAULT_IMAGE_KEY,
    state_source: str = "shifted_action",
    vcodec: str = "h264",
    overwrite: bool = False,
    push_to_hub: bool = False,
) -> ExportSummary:
    """Convert accepted TurboPi episodes into a LeRobot dataset."""
    episodes_dir = Path(episodes_dir)
    output_dir = Path(output_dir)

    if state_source not in STATE_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported state source '{state_source}'. Choose from: {sorted(STATE_SOURCE_CHOICES)}"
        )
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")

    episode_dirs = discover_episode_dirs(episodes_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode folders were found under: {episodes_dir}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use overwrite=True to replace it."
            )
        shutil.rmtree(output_dir)

    first_episode_frames = decode_video_frames(episode_dirs[0] / "video.mp4")
    if not first_episode_frames:
        raise ValueError(f"No frames could be decoded from: {episode_dirs[0] / 'video.mp4'}")

    image_shape = first_episode_frames[0].shape
    include_state = state_source != "none"
    features = build_features(image_shape=image_shape, image_key=image_key, include_state=include_state)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        vcodec=vcodec,
    )

    total_frames = 0
    try:
        for episode_dir in episode_dirs:
            rows = load_episode_rows(episode_dir)
            if state_source == "recorded":
                validate_recorded_state_rows(rows, episode_dir)
            frames = first_episode_frames if episode_dir == episode_dirs[0] else decode_video_frames(
                episode_dir / "video.mp4"
            )

            if len(rows) != len(frames):
                raise ValueError(
                    f"Frame count mismatch in {episode_dir}: parquet has {len(rows)} rows, "
                    f"video has {len(frames)} frames."
                )

            previous_action = np.zeros(3, dtype=np.float32)
            for frame_index, row in enumerate(rows):
                action = as_float32_vector(row[ACTION_COLUMN], ACTION_COLUMN)

                frame = {
                    image_key: frames[frame_index],
                    "action": action,
                    "task": str(row[TASK_COLUMN]),
                }

                state = build_state_vector(
                    row=row,
                    state_source=state_source,
                    previous_action=previous_action,
                )
                if state is not None:
                    frame[STATE_COLUMN] = state

                dataset.add_frame(frame)
                previous_action = action.copy()

            dataset.save_episode(parallel_encoding=False)
            total_frames += len(rows)

        dataset.finalize()

        if push_to_hub:
            dataset.push_to_hub()

        reloaded = LeRobotDataset(repo_id=repo_id, root=output_dir)
        if reloaded.num_episodes != len(episode_dirs):
            raise RuntimeError(
                f"Reloaded LeRobot dataset has {reloaded.num_episodes} episodes, expected {len(episode_dirs)}."
            )
        if reloaded.num_frames != total_frames:
            raise RuntimeError(
                f"Reloaded LeRobot dataset has {reloaded.num_frames} frames, expected {total_frames}."
            )

        return ExportSummary(
            repo_id=repo_id,
            output_dir=output_dir,
            num_episodes=reloaded.num_episodes,
            num_frames=reloaded.num_frames,
            image_key=image_key,
            state_source=state_source,
        )
    except Exception:
        dataset.finalize()
        raise


def discover_episode_dirs(episodes_dir: Path) -> list[Path]:
    """Find accepted episode directories below a root episodes path."""
    episode_dirs = []
    for path in sorted(episodes_dir.glob("**/episode_*")):
        if not path.is_dir():
            continue
        if (path / "data.parquet").exists() and (path / "video.mp4").exists():
            episode_dirs.append(path)
    return episode_dirs


def build_features(image_shape: tuple[int, int, int], image_key: str, include_state: bool) -> dict[str, dict]:
    """Build a LeRobot feature dictionary for single-camera TurboPi data."""
    features: dict[str, dict] = {
        image_key: {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ACTION_NAMES,
        },
    }
    if include_state:
        features[STATE_COLUMN] = {
            "dtype": "float32",
            "shape": (3,),
            "names": ACTION_NAMES,
        }
    return features


def load_episode_rows(episode_dir: Path) -> list[dict]:
    """Load episode rows from Parquet and validate required columns."""
    df = pd.read_parquet(episode_dir / "data.parquet")
    required_columns = {TASK_COLUMN, ACTION_COLUMN}
    if missing := sorted(required_columns - set(df.columns)):
        raise ValueError(f"{episode_dir / 'data.parquet'} is missing required columns: {missing}")
    return df.to_dict(orient="records")


def decode_video_frames(video_path: Path) -> list[np.ndarray]:
    """Decode an MP4 file into a list of RGB uint8 numpy arrays."""
    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def as_float32_vector(value, column_name: str) -> np.ndarray:
    """Parse a saved vector column as a float32 numpy array."""
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"Expected a 3D vector in column '{column_name}', got shape {array.shape}")
    return array


def validate_recorded_state_rows(
    rows: list[dict],
    episode_dir: Path,
    *,
    tolerance: float = 1e-5,
    min_action_changes: int = 3,
) -> None:
    """Refuse obviously leaky same-step state labels when recorded state is requested.

    Modern sessions store observation.state as the previous normalized action. Older buggy
    sessions may instead store the current action label, which leaks the answer into the
    model input. This check intentionally only trips on strong evidence of that old bug.
    """
    if not rows:
        return
    if STATE_COLUMN not in rows[0]:
        raise ValueError(
            f"State source 'recorded' was requested, but '{STATE_COLUMN}' is missing from {episode_dir / 'data.parquet'}."
        )
    if len(rows) <= 1:
        return

    actions = np.asarray([as_float32_vector(row[ACTION_COLUMN], ACTION_COLUMN) for row in rows], dtype=np.float32)
    states = np.asarray([as_float32_vector(row[STATE_COLUMN], STATE_COLUMN) for row in rows], dtype=np.float32)

    # If the action almost never changes, both same-step and shifted views can look identical.
    action_change = np.abs(actions[1:] - actions[:-1]).max(axis=1)
    if int((action_change > tolerance).sum()) < min_action_changes:
        return

    same_step_ratio = float((np.abs(states - actions).max(axis=1) <= tolerance).mean())
    shifted_ratio = float((np.abs(states[1:] - actions[:-1]).max(axis=1) <= tolerance).mean())

    if same_step_ratio >= 0.98 and same_step_ratio > shifted_ratio + 0.1:
        raise ValueError(
            "Recorded observation.state appears to match the same-step action labels in "
            f"{episode_dir}, which would leak the target during training "
            f"(same_step_match_ratio={same_step_ratio:.3f}, shifted_match_ratio={shifted_ratio:.3f}). "
            "Export again with --state-source shifted_action."
        )


def build_state_vector(row: dict, state_source: str, previous_action: np.ndarray) -> np.ndarray | None:
    """Build the observation.state vector for a frame."""
    if state_source == "none":
        return None
    if state_source == "zeros":
        return np.zeros(3, dtype=np.float32)
    if state_source == "recorded":
        if STATE_COLUMN not in row:
            raise ValueError(
                f"State source 'recorded' was requested, but '{STATE_COLUMN}' is missing from the episode data."
            )
        return as_float32_vector(row[STATE_COLUMN], STATE_COLUMN)
    if state_source == "shifted_action":
        return previous_action.copy()
    raise AssertionError(f"Unhandled state source: {state_source}")
