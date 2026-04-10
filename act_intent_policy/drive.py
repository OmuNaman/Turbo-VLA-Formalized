"""Autonomous driver for the ACT-style TurboPi task-conditioned policy."""

from __future__ import annotations

import argparse
import collections
import signal
import time
from pathlib import Path

import numpy as np
import torch

from client.drive_trace import DriveTraceRecorder
from client.robot_client import RobotClient
from intent_cnn_policy.drive import (
    apply_minimum_command_floor,
    denormalize_action,
    frame_to_tensor,
    resolve_task_selection,
)

from .model import load_checkpoint
from .train import resolve_device


def prune_chunk_history(
    history: collections.deque[tuple[np.ndarray, int]],
    *,
    current_step: int,
    chunk_size: int,
    window: int,
) -> None:
    """Keep only overlapping recent chunks relevant to the current timestep."""
    max_items = max(int(window), 1)
    while history and (current_step - history[0][1] >= chunk_size or len(history) > max_items):
        history.popleft()


def temporal_ensemble_action(
    history: collections.deque[tuple[np.ndarray, int]],
    *,
    current_step: int,
    chunk_size: int,
    decay: float,
) -> tuple[np.ndarray | None, list[dict[str, float]]]:
    """Average overlapping chunk predictions for the current timestep."""
    weighted = np.zeros(3, dtype=np.float32)
    total_weight = 0.0
    contributions: list[dict[str, float]] = []

    for chunk, origin_step in history:
        offset = current_step - origin_step
        if offset < 0 or offset >= min(int(chunk.shape[0]), chunk_size):
            continue
        weight = float(decay ** offset) if decay > 0.0 else (1.0 if offset == 0 else 0.0)
        if weight <= 0.0:
            continue
        action = np.asarray(chunk[offset], dtype=np.float32)
        weighted += weight * action
        total_weight += weight
        contributions.append(
            {
                "origin_step": float(origin_step),
                "offset": float(offset),
                "weight": weight,
            }
        )

    if total_weight <= 0.0:
        return None, []
    return weighted / total_weight, contributions


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
    parser.add_argument(
        "--no-temporal-ensemble",
        action="store_true",
        help="When replanning every frame, use only the newest first action instead of blending overlapping chunks",
    )
    parser.add_argument(
        "--temporal-ensemble-window",
        type=int,
        default=8,
        help="How many recent predicted chunks to blend when replanning every frame",
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.7,
        help="Exponential decay for older chunk predictions during temporal ensembling",
    )
    parser.add_argument("--smoothing", type=float, default=0.35, help="EMA factor for previous normalized action")
    parser.add_argument("--vx-cap", type=float, default=35.0)
    parser.add_argument("--vy-cap", type=float, default=35.0)
    parser.add_argument("--omega-cap", type=float, default=25.0)
    parser.add_argument("--vx-scale", type=float, default=1.0, help="Multiplier on normalized forward prediction")
    parser.add_argument("--vy-scale", type=float, default=1.0, help="Multiplier on normalized lateral prediction")
    parser.add_argument("--omega-scale", type=float, default=1.0, help="Multiplier on normalized turn prediction")
    parser.add_argument("--min-vx", type=float, default=0.0)
    parser.add_argument("--min-vy", type=float, default=0.0)
    parser.add_argument("--min-omega", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trace-dir", default=None, help="Write a per-step drive trace to this directory")
    parser.add_argument(
        "--trace-save-frames",
        action="store_true",
        help="Save annotated camera frames alongside trace.jsonl",
    )
    parser.add_argument(
        "--trace-frame-every",
        type=int,
        default=1,
        help="When tracing frames, save every Nth control step",
    )
    parser.add_argument(
        "--trace-max-steps",
        type=int,
        default=None,
        help="Automatically stop after this many control steps",
    )
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
    chunk_size = int(model.config.chunk_size)
    period_s = 1.0 / max(args.loop_hz, 1.0)
    smooth_alpha = float(np.clip(args.smoothing, 0.0, 0.99))
    use_temporal_ensemble = bool(not args.reuse_action_queue and not args.no_temporal_ensemble)
    axis_scale = np.asarray(
        [float(args.vx_scale), float(args.vy_scale), float(args.omega_scale)],
        dtype=np.float32,
    )

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
    print(f"  Chunk size: {chunk_size}")
    if args.reuse_action_queue:
        mode = "reuse predicted chunk"
    elif use_temporal_ensemble:
        mode = (
            f"replan + temporal ensemble "
            f"(window={max(int(args.temporal_ensemble_window), 1)}, decay={float(args.temporal_ensemble_decay):.2f})"
        )
    else:
        mode = "replan every frame (first action only)"
    print(f"  Mode: {mode}")
    print(f"  Battery: {health.get('battery_mv', '?')}mV")
    print(f"  Camera: {'OK' if health.get('camera_ok') else 'FAIL'}")
    if args.trace_dir:
        print(f"  Trace: {args.trace_dir}")
        if args.trace_save_frames:
            print(f"  Trace frames: every {max(int(args.trace_frame_every), 1)} step(s)")
        if args.trace_max_steps is not None:
            print(f"  Trace max steps: {args.trace_max_steps}")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    buffer: collections.deque[np.ndarray] = collections.deque(maxlen=frame_history)
    queued_actions: collections.deque[np.ndarray] = collections.deque()
    recent_chunks: collections.deque[tuple[np.ndarray, int]] = collections.deque()
    previous_action = np.zeros(3, dtype=np.float32)
    task_tensor = torch.tensor([task_index], dtype=torch.long, device=device)
    trace = None
    if args.trace_dir:
        trace = DriveTraceRecorder(
            Path(args.trace_dir),
            save_frames=args.trace_save_frames,
            frame_every=args.trace_frame_every,
            metadata={
                "policy": "act_intent",
                "robot_url": robot_url,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_epoch": payload.get("epoch"),
                "task_index": task_index,
                "task_name": task_name,
                "frame_history": frame_history,
                "image_width": image_width,
                "image_height": image_height,
                "chunk_size": int(model.config.chunk_size),
                "reuse_action_queue": bool(args.reuse_action_queue),
                "temporal_ensemble_enabled": bool(use_temporal_ensemble),
                "temporal_ensemble_window": int(args.temporal_ensemble_window),
                "temporal_ensemble_decay": float(args.temporal_ensemble_decay),
                "loop_hz": float(args.loop_hz),
                "smoothing": float(args.smoothing),
                "vx_cap": float(args.vx_cap),
                "vy_cap": float(args.vy_cap),
                "omega_cap": float(args.omega_cap),
                "vx_scale": float(args.vx_scale),
                "vy_scale": float(args.vy_scale),
                "omega_scale": float(args.omega_scale),
                "min_vx": float(args.min_vx),
                "min_vy": float(args.min_vy),
                "min_omega": float(args.min_omega),
                "device": str(device),
            },
        )

    def safe_stop(*_args):
        try:
            client.stop()
        except Exception:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGINT, safe_stop)
    signal.signal(signal.SIGTERM, safe_stop)

    try:
        step_idx = 0
        first_frame, _, _ = client.get_frame_rgb()
        for _ in range(frame_history):
            buffer.append(first_frame.copy())

        while True:
            loop_start = time.monotonic()
            frame, frame_timestamp, frame_index = client.get_frame_rgb()
            buffer.append(frame)
            if len(buffer) < frame_history:
                time.sleep(period_s)
                continue

            source = "queue"
            pred_chunk = None
            ensemble_contributions: list[dict[str, float]] = []
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
                if args.reuse_action_queue:
                    pred = pred_chunk[0]
                    source = "fresh"
                    queued_actions.clear()
                    for step_action in pred_chunk[1:]:
                        queued_actions.append(np.asarray(step_action, dtype=np.float32))
                elif use_temporal_ensemble:
                    recent_chunks.append((pred_chunk.copy(), step_idx))
                    prune_chunk_history(
                        recent_chunks,
                        current_step=step_idx,
                        chunk_size=chunk_size,
                        window=args.temporal_ensemble_window,
                    )
                    pred, ensemble_contributions = temporal_ensemble_action(
                        recent_chunks,
                        current_step=step_idx,
                        chunk_size=chunk_size,
                        decay=float(args.temporal_ensemble_decay),
                    )
                    if pred is None:
                        pred = pred_chunk[0]
                        source = "fresh"
                    else:
                        pred = np.asarray(pred, dtype=np.float32)
                        source = "ens"
                else:
                    pred = pred_chunk[0]
                    source = "fresh"

            smoothed = smooth_alpha * previous_action + (1.0 - smooth_alpha) * np.clip(pred, -1.0, 1.0)
            scaled = np.clip(smoothed * axis_scale, -1.0, 1.0)
            raw_action = denormalize_action(scaled, args.vx_cap, args.vy_cap, args.omega_cap)
            safe_action = apply_minimum_command_floor(
                raw_action,
                min_vx=args.min_vx,
                min_vy=args.min_vy,
                min_omega=args.min_omega,
            )
            previous_action = smoothed

            trace_payload = {
                "step_idx": step_idx,
                "policy": "act_intent",
                "task_index": task_index,
                "task_name": task_name,
                "frame_index": int(frame_index),
                "robot_timestamp": float(frame_timestamp),
                "source": source,
                "queue_depth": len(queued_actions),
                "pred_norm": pred,
                "smoothed_norm": smoothed,
                "scaled_norm": scaled,
                "raw_action": raw_action,
                "safe_action": safe_action,
                "axis_scale": axis_scale,
                "loop_period_s": float(period_s),
            }
            if pred_chunk is not None:
                trace_payload["pred_chunk_norm"] = pred_chunk
            if ensemble_contributions:
                trace_payload["ensemble_count"] = len(ensemble_contributions)
                trace_payload["ensemble_contributions"] = ensemble_contributions
            try:
                client.send_velocity(float(safe_action[0]), float(safe_action[1]), float(safe_action[2]))
                trace_payload["send_ok"] = True
            except Exception as exc:
                trace_payload["send_ok"] = False
                trace_payload["send_error"] = str(exc)
                if trace is not None:
                    elapsed = time.monotonic() - loop_start
                    trace_payload["loop_elapsed_ms"] = elapsed * 1000.0
                    trace.record(
                        step_idx=step_idx,
                        frame=frame,
                        payload=trace_payload,
                        overlay_lines=[
                            f"step={step_idx} task={task_name} src={source}",
                            f"pred={pred[0]:+.2f}, {pred[1]:+.2f}, {pred[2]:+.2f}",
                            f"scaled={scaled[0]:+.2f}, {scaled[1]:+.2f}, {scaled[2]:+.2f}",
                            f"cmd={safe_action[0]:+.2f}, {safe_action[1]:+.2f}, {safe_action[2]:+.2f}",
                            f"frame={frame_index} queue={len(queued_actions)} ens={len(ensemble_contributions)} send=error",
                        ],
                    )
                print(f"\n  [WARN] Failed to send velocity command: {exc}")
                client.stop()
                break

            elapsed = time.monotonic() - loop_start
            trace_payload["loop_elapsed_ms"] = elapsed * 1000.0
            if trace is not None:
                trace.record(
                    step_idx=step_idx,
                    frame=frame,
                    payload=trace_payload,
                    overlay_lines=[
                        f"step={step_idx} task={task_name} src={source}",
                        f"pred={pred[0]:+.2f}, {pred[1]:+.2f}, {pred[2]:+.2f}",
                        f"scaled={scaled[0]:+.2f}, {scaled[1]:+.2f}, {scaled[2]:+.2f}",
                        f"cmd={safe_action[0]:+.2f}, {safe_action[1]:+.2f}, {safe_action[2]:+.2f}",
                        f"frame={frame_index} queue={len(queued_actions)} ens={len(ensemble_contributions)} send=ok",
                    ],
                )

            print(
                f"\r  task={task_index:02d} src={source:5s} pred=[{pred[0]:5.2f},{pred[1]:5.2f},{pred[2]:5.2f}] "
                f"cmd=[{safe_action[0]:6.2f},{safe_action[1]:6.2f},{safe_action[2]:6.2f}] "
                f"queue={len(queued_actions):02d} ens={len(ensemble_contributions):02d}   ",
                end="",
                flush=True,
            )

            sleep_for = period_s - elapsed
            step_idx += 1
            if args.trace_max_steps is not None and step_idx >= args.trace_max_steps:
                print(f"\n  Reached trace max steps ({args.trace_max_steps}).")
                break
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
        if trace is not None:
            trace.close()
        print("  ACT-Intent drive stopped.")


if __name__ == "__main__":
    main()
