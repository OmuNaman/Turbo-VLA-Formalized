"""Episode writer that saves accepted episodes as MP4 plus Parquet."""

import json
import shutil
from pathlib import Path

import pandas as pd

try:
    import av
except ImportError:  # pragma: no cover - exercised on machines without PyAV
    av = None


class EpisodeWriter:
    """Writes accepted episodes to disk."""

    def __init__(self, episodes_dir: Path, fps: int = 10, vcodec: str = "h264"):
        self.episodes_dir = Path(episodes_dir)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.vcodec = vcodec

    @property
    def video_available(self) -> bool:
        """Whether PyAV is available for MP4 writing."""
        return av is not None

    def save_episode(self, episode) -> Path:
        """Save an accepted EpisodeBuffer to disk."""
        ep_dir = self.episodes_dir / f"episode_{episode.episode_index:06d}"
        tmp_dir = self.episodes_dir / f".episode_{episode.episode_index:06d}.tmp"
        if ep_dir.exists():
            raise FileExistsError(f"Episode directory already exists: {ep_dir}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=False)

        video_path = tmp_dir / "video.mp4"
        parquet_path = tmp_dir / "data.parquet"

        try:
            self._save_video(episode.frames, video_path)
            self._save_parquet(episode, parquet_path)
            tmp_dir.replace(ep_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        print(
            f"  [EpisodeWriter] Saved episode {episode.episode_index} "
            f"({len(episode.frames)} frames, {len(episode.frames) / self.fps:.1f}s) -> {ep_dir}"
        )
        return ep_dir

    def _save_video(self, frames: list, video_path: Path) -> None:
        """Encode a list of RGB frames to MP4."""
        if not frames:
            return
        if av is None:
            raise RuntimeError(
                "PyAV is not installed, so accepted episodes cannot be saved as MP4. "
                "Install it with `pip install av`."
            )

        first_img = frames[0].image
        height, width = first_img.shape[:2]

        container = av.open(str(video_path), "w")
        failed = False
        try:
            stream = self._add_stream_with_fallback(container)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"

            for frame_index, frame_data in enumerate(frames):
                video_frame = av.VideoFrame.from_ndarray(frame_data.image, format="rgb24")
                video_frame.pts = frame_index
                for packet in stream.encode(video_frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
        except Exception:
            failed = True
            raise
        finally:
            container.close()
            if failed:
                video_path.unlink(missing_ok=True)

    def _add_stream_with_fallback(self, container):
        """Create a video stream, falling back to a broadly supported codec."""
        attempted_codecs: list[str] = []
        last_error: Exception | None = None

        for codec in (self.vcodec, "mpeg4"):
            if codec in attempted_codecs:
                continue
            attempted_codecs.append(codec)
            try:
                return container.add_stream(codec, rate=self.fps)
            except Exception as exc:  # pragma: no cover - codec availability is machine-specific
                last_error = exc

        raise RuntimeError(
            f"Failed to open a video encoder. Tried codecs: {', '.join(attempted_codecs)}"
        ) from last_error

    def _save_parquet(self, episode, parquet_path: Path) -> None:
        """Save per-frame data as Parquet."""
        num_frames = len(episode.frames)
        if num_frames == 0:
            return

        data = {
            "frame_index": list(range(num_frames)),
            "timestamp": [frame.timestamp for frame in episode.frames],
            "episode_index": [episode.episode_index] * num_frames,
            "task_index": [episode.task_index] * num_frames,
            "task": [episode.task] * num_frames,
            "observation.state": [frame.state.tolist() for frame in episode.frames],
            "action": [frame.action.tolist() for frame in episode.frames],
        }

        pd.DataFrame(data).to_parquet(parquet_path, index=False)

    def save_task_mapping(self, tasks: list[str]) -> None:
        """Save task-index to task-string mapping in the session folder."""
        path = self.episodes_dir / "tasks.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump({i: task for i, task in enumerate(tasks)}, handle, indent=2)

    def get_episode_count(self) -> int:
        """Count saved episodes on disk."""
        return len(list(self.episodes_dir.glob("episode_*")))
