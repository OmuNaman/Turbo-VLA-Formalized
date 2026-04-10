"""Helpers for recording per-step policy drive traces for offline inspection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _jsonify(value: Any) -> Any:
    """Convert numpy-heavy payloads into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


class DriveTraceRecorder:
    """Persist step-by-step policy outputs and optional annotated frames."""

    def __init__(
        self,
        trace_dir: Path,
        *,
        save_frames: bool = False,
        frame_every: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.trace_dir / "frames"
        self.save_frames = bool(save_frames)
        self.frame_every = max(int(frame_every), 1)
        self.trace_path = self.trace_dir / "trace.jsonl"
        self.metadata_path = self.trace_dir / "metadata.json"
        self._font = ImageFont.load_default()

        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        payload = dict(metadata or {})
        payload.setdefault("created_at_unix", time.time())
        payload.setdefault("trace_path", self.trace_path.name)
        if self.save_frames:
            payload.setdefault("frames_dir", self.frames_dir.name)
            payload.setdefault("frame_every", self.frame_every)
        self.metadata_path.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")
        self._handle = self.trace_path.open("w", encoding="utf-8")

    def _save_annotated_frame(self, frame: np.ndarray, *, rel_path: str, overlay_lines: list[str]) -> None:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image, "RGBA")

        if overlay_lines:
            line_height = 14
            padding = 6
            box_height = padding * 2 + line_height * len(overlay_lines)
            box_width = min(image.width - 12, 540)
            draw.rectangle((6, 6, 6 + box_width, 6 + box_height), fill=(0, 0, 0, 170))
            y = 6 + padding
            for line in overlay_lines:
                draw.text((12, y), line, fill=(255, 255, 255, 255), font=self._font)
                y += line_height

        image.save(self.trace_dir / rel_path)

    def record(
        self,
        *,
        step_idx: int,
        frame: np.ndarray | None,
        payload: dict[str, Any],
        overlay_lines: list[str] | None = None,
    ) -> None:
        record = dict(payload)
        record.setdefault("step_idx", int(step_idx))
        record.setdefault("recorded_at_unix", time.time())

        if self.save_frames and frame is not None and step_idx % self.frame_every == 0:
            rel_path = f"frames/frame_{step_idx:06d}.png"
            self._save_annotated_frame(frame, rel_path=rel_path, overlay_lines=list(overlay_lines or []))
            record["frame_path"] = rel_path

        self._handle.write(json.dumps(_jsonify(record), ensure_ascii=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

