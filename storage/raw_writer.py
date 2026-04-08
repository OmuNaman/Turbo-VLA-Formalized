"""Raw session writer for continuous MP4 plus JSONL telemetry backup."""

import json
import threading
from pathlib import Path

import numpy as np

try:
    import av
except ImportError:  # pragma: no cover - exercised on machines without PyAV
    av = None


class RawWriter:
    """Writes continuous session recording for backup."""

    def __init__(self, session_dir: Path, fps: int = 10, vcodec: str = "h264"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.vcodec = vcodec

        self._video_path = self.session_dir / "video.mp4"
        self._telemetry_path = self.session_dir / "telemetry.jsonl"

        self._container = None
        self._stream = None
        self._telemetry_file = None
        self._frame_count = 0
        self._session_frames_before_start = 0
        self._lock = threading.Lock()
        self._flush_interval = 10
        self._video_enabled = av is not None
        self._warned_video_disable = False

    @property
    def video_available(self) -> bool:
        """Whether MP4 backup writing is available."""
        return self._video_enabled and av is not None

    def start(self) -> None:
        """Open telemetry output for writing."""
        if av is None:
            print("[RawWriter] WARNING: PyAV not installed, skipping raw session video backup")
            print("  Install with: pip install av")

        self._prepare_resume_outputs()
        mode = "a" if self._telemetry_path.exists() else "w"
        self._telemetry_file = open(self._telemetry_path, mode, encoding="utf-8")

    def _prepare_resume_outputs(self) -> None:
        """Reuse the same raw folder without overwriting prior files."""
        telemetry_path = self.session_dir / "telemetry.jsonl"
        self._telemetry_path = telemetry_path
        if telemetry_path.exists():
            self._session_frames_before_start = self._count_existing_telemetry_frames(telemetry_path)
            self._frame_count = self._session_frames_before_start
        else:
            self._session_frames_before_start = 0
            self._frame_count = 0

        base_video_path = self.session_dir / "video.mp4"
        if not base_video_path.exists():
            self._video_path = base_video_path
            return

        part_index = 2
        while True:
            candidate = self.session_dir / f"video_part{part_index:03d}.mp4"
            if not candidate.exists():
                self._video_path = candidate
                return
            part_index += 1

    def _count_existing_telemetry_frames(self, telemetry_path: Path) -> int:
        """Count existing telemetry rows so resumed frame indices stay monotonic."""
        count = 0
        with telemetry_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    def write_frame(
        self,
        image: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        timestamp: float,
        task: str,
        episode_index: int | None = None,
    ) -> None:
        """Write one frame of video and telemetry."""
        with self._lock:
            if self._video_enabled:
                try:
                    self._write_video_frame(image)
                except Exception as exc:  # pragma: no cover - codec availability is machine-specific
                    self._video_enabled = False
                    if not self._warned_video_disable:
                        print(f"\n[RawWriter] WARNING: disabling raw video backup: {exc}")
                        self._warned_video_disable = True

            if self._telemetry_file is not None:
                entry = {
                    "t": round(timestamp, 6),
                    "frame_idx": self._frame_count,
                    "state": state.tolist(),
                    "action": action.tolist(),
                    "task": task,
                    "episode_idx": episode_index,
                }
                self._telemetry_file.write(json.dumps(entry) + "\n")

                if (self._frame_count + 1) % self._flush_interval == 0:
                    self._telemetry_file.flush()

            self._frame_count += 1

    def _write_video_frame(self, image: np.ndarray) -> None:
        """Encode and write a single RGB video frame."""
        if av is None:
            return

        if self._container is None:
            height, width = image.shape[:2]
            self._container = av.open(str(self._video_path), "w")
            self._stream = self._add_stream_with_fallback(self._container)
            self._stream.width = width
            self._stream.height = height
            self._stream.pix_fmt = "yuv420p"

        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = self._frame_count
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def _add_stream_with_fallback(self, container):
        """Create a video stream with a compatibility fallback codec."""
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

    def close(self) -> None:
        """Finalize and close all files."""
        with self._lock:
            if self._container is not None:
                for packet in self._stream.encode():
                    self._container.mux(packet)
                self._container.close()
                self._container = None
                self._stream = None

            if self._telemetry_file is not None:
                self._telemetry_file.flush()
                self._telemetry_file.close()
                self._telemetry_file = None

        if self._frame_count > 0:
            new_frames = self._frame_count - self._session_frames_before_start
            if self._session_frames_before_start > 0:
                print(
                    f"  [RawWriter] Saved {new_frames} new frames "
                    f"({self._frame_count} total) to {self.session_dir}"
                )
            else:
                print(f"  [RawWriter] Saved {self._frame_count} frames to {self.session_dir}")

    @property
    def frame_count(self) -> int:
        """Return the number of raw frames written so far."""
        return self._frame_count
