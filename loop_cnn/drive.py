"""Autonomous laptop-first driver for the TurboPi CNN policy."""

from __future__ import annotations

import argparse
import collections
import signal
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from client.robot_client import RobotClient

from .model import load_checkpoint
from .train import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drive TurboPi using a trained CNN policy")
    parser.add_argument("--robot-ip", default="192.168.149.1")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--smoothing", type=float, default=0.65, help="EMA factor for previous action")
    parser.add_argument("--vx-cap", type=float, default=35.0)
    parser.add_argument("--vy-cap", type=float, default=35.0)
    parser.add_argument("--omega-cap", type=float, default=25.0)
    parser.add_argument("--min-vx", type=float, default=0.0,
                        help="Minimum absolute vx command to send when vx is nonzero")
    parser.add_argument("--min-vy", type=float, default=0.0,
                        help="Minimum absolute vy command to send when vy is nonzero")
    parser.add_argument("--min-omega", type=float, default=0.0,
                        help="Minimum absolute omega command to send when omega is nonzero")
    parser.add_argument("--device", default="auto")
    return parser


def denormalize_action(action: np.ndarray, vx_cap: float, vy_cap: float, omega_cap: float) -> np.ndarray:
    """Convert normalized [-1, 1] actions into robot command units."""
    clipped = np.clip(action, -1.0, 1.0)
    caps = np.asarray([vx_cap, vy_cap, omega_cap], dtype=np.float32)
    return clipped * caps


def apply_minimum_command_floor(
    action: np.ndarray,
    *,
    min_vx: float,
    min_vy: float,
    min_omega: float,
    zero_eps: float = 1e-4,
) -> np.ndarray:
    """Lift small nonzero commands above the motor deadzone."""
    result = np.asarray(action, dtype=np.float32).copy()
    minimums = np.asarray([min_vx, min_vy, min_omega], dtype=np.float32)
    for index, floor in enumerate(minimums):
        if floor <= 0:
            continue
        value = float(result[index])
        if abs(value) <= zero_eps:
            result[index] = 0.0
            continue
        if abs(value) < floor:
            result[index] = np.sign(value) * floor
    return result


def frame_to_tensor(frame: np.ndarray, *, image_width: int, image_height: int) -> torch.Tensor:
    image = Image.fromarray(frame).convert("RGB")
    image = image.resize((image_width, image_height), Image.Resampling.BILINEAR)
    return TF.to_tensor(image)


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model, payload = load_checkpoint(Path(args.checkpoint), map_location=device)
    model = model.to(device)
    model.eval()

    image_width = int(model.config.image_width)
    image_height = int(model.config.image_height)
    frame_history = int(model.config.frame_history)
    robot_url = f"http://{args.robot_ip}:{args.robot_port}"
    period_s = 1.0 / max(args.loop_hz, 1.0)
    smooth_alpha = float(np.clip(args.smoothing, 0.0, 0.99))

    client = RobotClient(robot_url=robot_url, timeout=1.0, max_retries=2)
    if not client.is_connected():
        raise RuntimeError(f"Cannot reach robot at {robot_url}")

    try:
        health = client.get_health()
    except Exception:
        health = {}

    print()
    print("=" * 50)
    print("  TurboPi CNN Drive")
    print("=" * 50)
    print(f"  Robot: {robot_url}")
    print(f"  Device: {device}")
    print(f"  Checkpoint epoch: {payload.get('epoch')}")
    print(f"  Battery: {health.get('battery_mv', '?')}mV")
    print(f"  Camera: {'OK' if health.get('camera_ok') else 'FAIL'}")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    buffer: collections.deque[np.ndarray] = collections.deque(maxlen=frame_history)
    previous_action = np.zeros(3, dtype=np.float32)

    def safe_stop(*_args):
        try:
            client.stop()
        except Exception:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGINT, safe_stop)
    signal.signal(signal.SIGTERM, safe_stop)

    try:
        first_frame, _, _ = client.get_frame_rgb()
        for _ in range(frame_history):
            buffer.append(first_frame.copy())

        while True:
            loop_start = time.monotonic()
            frame, _, _ = client.get_frame_rgb()
            buffer.append(frame)
            if len(buffer) < frame_history:
                time.sleep(period_s)
                continue

            stacked = torch.stack(
                [frame_to_tensor(img, image_width=image_width, image_height=image_height) for img in list(buffer)],
                dim=0,
            ).reshape(1, frame_history * 3, image_height, image_width)

            with torch.no_grad():
                pred = model(stacked.to(device)).squeeze(0).detach().cpu().numpy().astype(np.float32)

            smoothed = smooth_alpha * previous_action + (1.0 - smooth_alpha) * np.clip(pred, -1.0, 1.0)
            raw_action = denormalize_action(smoothed, args.vx_cap, args.vy_cap, args.omega_cap)
            safe_action = apply_minimum_command_floor(
                raw_action,
                min_vx=args.min_vx,
                min_vy=args.min_vy,
                min_omega=args.min_omega,
            )
            previous_action = smoothed

            try:
                client.send_velocity(float(safe_action[0]), float(safe_action[1]), float(safe_action[2]))
            except Exception as exc:
                print(f"\n  [WARN] Failed to send velocity command: {exc}")
                client.stop()
                break

            print(
                f"\r  pred=[{pred[0]:5.2f},{pred[1]:5.2f},{pred[2]:5.2f}] "
                f"cmd=[{safe_action[0]:6.2f},{safe_action[1]:6.2f},{safe_action[2]:6.2f}]   ",
                end="",
                flush=True,
            )

            elapsed = time.monotonic() - loop_start
            sleep_for = period_s - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        try:
            client.stop()
        except Exception:
            pass
        print("  Loop drive stopped.")


if __name__ == "__main__":
    raise SystemExit(
        "The direct `python -m loop_cnn.drive` entrypoint is deprecated. "
        "Use `python -m cnn_policy.drive` instead."
    )
