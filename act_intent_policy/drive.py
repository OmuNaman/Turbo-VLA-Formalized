"""Autonomous driver for the ACT-style TurboPi task-conditioned policy."""

from __future__ import annotations

import argparse
import collections
import signal
import time
from pathlib import Path

import numpy as np
import torch

from client.robot_client import RobotClient
from intent_cnn_policy.drive import (
    apply_minimum_command_floor,
    denormalize_action,
    frame_to_tensor,
    resolve_task_selection,
)

from .model import load_checkpoint
from .train import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drive TurboPi using a trained ACT-style task-conditioned policy")
    parser.add_argument("--robot-ip", default="192.168.149.1")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument(
        "--task",
        default=None,
        help="Exact task string from the checkpoint vocabulary, for example 'go left'",
    )
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument(
        "--reuse-action-queue",
        action="store_true",
        help="Consume the predicted chunk over multiple steps instead of replanning every frame",
    )
    parser.add_argument("--smoothing", type=float, default=0.35, help="EMA factor for previous normalized action")
    parser.add_argument("--vx-cap", type=float, default=35.0)
    parser.add_argument("--vy-cap", type=float, default=35.0)
    parser.add_argument("--omega-cap", type=float, default=25.0)
    parser.add_argument("--min-vx", type=float, default=0.0)
    parser.add_argument("--min-vy", type=float, default=0.0)
    parser.add_argument("--min-omega", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model, payload = load_checkpoint(Path(args.checkpoint), map_location=device)
    model = model.to(device)
    model.eval()

    task_names = list(payload.get("extra", {}).get("task_names") or [])
    task_index, task_name = resolve_task_selection(
        task_names,
        args.task,
        args.task_index,
        task_vocab_size=int(model.config.task_vocab_size),
    )

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
    print("  TurboPi ACT-Intent Drive")
    print("=" * 50)
    print(f"  Robot: {robot_url}")
    print(f"  Device: {device}")
    print(f"  Checkpoint epoch: {payload.get('epoch')}")
    print(f"  Task: {task_name} [{task_index}]")
    print(f"  Chunk size: {int(model.config.chunk_size)}")
    print(f"  Mode: {'reuse predicted chunk' if args.reuse_action_queue else 'replan every frame'}")
    print(f"  Battery: {health.get('battery_mv', '?')}mV")
    print(f"  Camera: {'OK' if health.get('camera_ok') else 'FAIL'}")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    buffer: collections.deque[np.ndarray] = collections.deque(maxlen=frame_history)
    queued_actions: collections.deque[np.ndarray] = collections.deque()
    previous_action = np.zeros(3, dtype=np.float32)
    task_tensor = torch.tensor([task_index], dtype=torch.long, device=device)

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

            source = "queue"
            pred_chunk = None
            if args.reuse_action_queue and queued_actions:
                pred = np.asarray(queued_actions.popleft(), dtype=np.float32)
            else:
                stacked = torch.stack(
                    [frame_to_tensor(img, image_width=image_width, image_height=image_height) for img in list(buffer)],
                    dim=0,
                ).reshape(1, frame_history * 3, image_height, image_width)

                with torch.no_grad():
                    pred_chunk = (
                        model(stacked.to(device), task_tensor)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                pred = pred_chunk[0]
                source = "fresh"
                if args.reuse_action_queue:
                    queued_actions.clear()
                    for step_action in pred_chunk[1:]:
                        queued_actions.append(np.asarray(step_action, dtype=np.float32))

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
                f"\r  task={task_index:02d} src={source:5s} pred=[{pred[0]:5.2f},{pred[1]:5.2f},{pred[2]:5.2f}] "
                f"cmd=[{safe_action[0]:6.2f},{safe_action[1]:6.2f},{safe_action[2]:6.2f}] "
                f"queue={len(queued_actions):02d}   ",
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
        print("  ACT-Intent drive stopped.")


if __name__ == "__main__":
    main()
