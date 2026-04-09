"""Drive TurboPi using a fine-tuned SmolVLA checkpoint."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import SafetensorError

from client.robot_client import RobotClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drive TurboPi using a fine-tuned SmolVLA checkpoint")
    parser.add_argument("--robot-ip", default="192.168.149.1")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to either the downloaded outer folder or the inner pretrained_model folder")
    parser.add_argument("--task", default=None,
                        help="Language task prompt, for example 'go left' or 'go right'")
    parser.add_argument("--vlm-assets", default=None,
                        help="Optional local path or HF model id for SmolVLM tokenizer/config assets")
    parser.add_argument("--loop-hz", type=float, default=5.0)
    parser.add_argument("--smoothing", type=float, default=0.65,
                        help="EMA factor applied in normalized action space")
    parser.add_argument(
        "--reuse-action-queue",
        action="store_true",
        help=(
            "Use SmolVLA's cached action chunk across steps. Disabled by default because line-following "
            "needs fresh replanning from the latest camera frame."
        ),
    )
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


def resolve_device(requested: str) -> torch.device:
    """Resolve `auto` to the best available local torch device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover
        return torch.device("mps")
    return torch.device("cpu")


def denormalize_action(action: np.ndarray, vx_cap: float, vy_cap: float, omega_cap: float) -> np.ndarray:
    """Convert normalized dataset actions into robot command units."""
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


def resolve_task(task: str | None) -> str:
    """Prompt for the language task if it was not provided on the CLI."""
    if task:
        return task.strip()

    while True:
        value = input("  Enter task prompt (for example 'go left'): ").strip()
        if value:
            return value
        print("  Task prompt cannot be empty.")


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Accept either the outer downloaded folder or the inner pretrained_model folder."""
    root = Path(path).expanduser().resolve()
    candidates = [root, root / "pretrained_model"]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "model.safetensors").exists() and (candidate / "train_config.json").exists():
            return candidate
    raise FileNotFoundError(
        "Checkpoint folder must contain model.safetensors and train_config.json, or contain a "
        "'pretrained_model' subdirectory with those files."
    )


def validate_checkpoint_files(checkpoint_dir: Path) -> None:
    """Fail fast with a useful message when the downloaded checkpoint is incomplete."""
    model_path = checkpoint_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint weights: {model_path}")

    size_bytes = model_path.stat().st_size
    try:
        from safetensors import safe_open

        with safe_open(str(model_path), framework="pt", device="cpu") as handle:
            handle.keys()
    except SafetensorError as exc:
        size_mb = size_bytes / (1024 * 1024)
        raise RuntimeError(
            "The SmolVLA checkpoint weights are incomplete or corrupted.\n"
            f"File: {model_path}\n"
            f"Size: {size_bytes} bytes ({size_mb:.1f} MB)\n"
            "This fine-tuned SmolVLA model should be much larger than that, so this usually means the "
            "download or copy stopped early. Re-copy the whole `pretrained_model` folder from RunPod and try again."
        ) from exc


def clear_lerobot_modules() -> None:
    """Drop partially imported LeRobot modules before trying another import source."""
    for name in list(sys.modules):
        if name == "lerobot" or name.startswith("lerobot."):
            sys.modules.pop(name, None)


def import_lerobot_runtime(repo_root: Path) -> tuple[Any, Any, Any]:
    """Import the LeRobot classes needed for local SmolVLA inference."""
    attempts: list[tuple[str, Exception]] = []
    source_candidates = [None, repo_root / ".tmp_lerobot_src"]

    for source_root in source_candidates:
        clear_lerobot_modules()
        if source_root is not None and source_root.exists():
            sys.path.insert(0, str(source_root))

        try:
            from lerobot.configs.train import TrainPipelineConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.processor.pipeline import PolicyProcessorPipeline

            return TrainPipelineConfig, PolicyProcessorPipeline, SmolVLAPolicy
        except Exception as exc:  # pragma: no cover - depends on external env
            label = "installed lerobot" if source_root is None else f"local source {source_root}"
            attempts.append((label, exc))
        finally:
            if source_root is not None and source_root.exists():
                try:
                    sys.path.remove(str(source_root))
                except ValueError:
                    pass

    details = "\n".join(f"  - {label}: {type(exc).__name__}: {exc}" for label, exc in attempts)
    raise RuntimeError(
        "Could not import a usable LeRobot SmolVLA runtime.\n"
        "Install `lerobot[smolvla]` in the environment you use for driving, or keep a compatible "
        "LeRobot source checkout under `.tmp_lerobot_src`.\n"
        f"Import attempts:\n{details}"
    ) from attempts[-1][1]


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert an RGB uint8 frame into CHW float32 in the [0, 1] range."""
    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(torch.float32)
    return tensor / 255.0


def state_to_tensor(state: np.ndarray) -> torch.Tensor:
    """Convert the previous normalized action/state into a float32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(state)).to(torch.float32)


def action_to_numpy(action: Any) -> np.ndarray:
    """Convert a postprocessed action object into a flat float32 vector."""
    if isinstance(action, torch.Tensor):
        array = action.detach().cpu().numpy()
    else:
        array = np.asarray(action)
    return np.asarray(array, dtype=np.float32).reshape(-1)


def prepare_policy_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Ensure all tensor inputs are batched and live on the same device as the policy."""
    prepared: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor = value
            if key == "observation.state" and tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            prepared[key] = tensor.to(device=device)
        else:
            prepared[key] = value
    return prepared


def load_policy_runtime(
    checkpoint_dir: Path,
    *,
    device: torch.device,
    vlm_assets: str | None,
) -> tuple[Any, Any, Any, Any]:
    """Load the SmolVLA policy and its saved pre/postprocessors from disk."""
    repo_root = Path(__file__).resolve().parent.parent
    TrainPipelineConfig, PolicyProcessorPipeline, SmolVLAPolicy = import_lerobot_runtime(repo_root)
    validate_checkpoint_files(checkpoint_dir)

    train_cfg = TrainPipelineConfig.from_pretrained(checkpoint_dir)
    if train_cfg.policy is None:
        raise RuntimeError(f"No policy config found in {checkpoint_dir / 'train_config.json'}")

    policy_cfg = train_cfg.policy
    policy_cfg.device = str(device)
    policy_cfg.load_vlm_weights = False
    if vlm_assets:
        policy_cfg.vlm_model_name = vlm_assets

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": str(device)}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
    )
    policy = SmolVLAPolicy.from_pretrained(checkpoint_dir, config=policy_cfg)
    policy.reset()
    return policy, preprocessor, postprocessor, train_cfg


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    task = resolve_task(args.task)
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    robot_url = f"http://{args.robot_ip}:{args.robot_port}"
    period_s = 1.0 / max(args.loop_hz, 1.0)
    smooth_alpha = float(np.clip(args.smoothing, 0.0, 0.99))

    policy, preprocessor, postprocessor, train_cfg = load_policy_runtime(
        checkpoint_dir,
        device=device,
        vlm_assets=args.vlm_assets,
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
    print("  TurboPi SmolVLA Drive")
    print("=" * 50)
    print(f"  Robot: {robot_url}")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Task: {task}")
    print(f"  Steps trained: {train_cfg.steps}")
    print(f"  VLM assets: {args.vlm_assets or train_cfg.policy.vlm_model_name}")
    print(f"  Battery: {health.get('battery_mv', '?')}mV")
    print(f"  Camera: {'OK' if health.get('camera_ok') else 'FAIL'}")
    print()
    print("  Press Ctrl+C to stop.")
    print()

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
        while True:
            loop_start = time.monotonic()
            frame, _, _ = client.get_frame_rgb()

            batch = {
                "observation.images.front": frame_to_tensor(frame),
                "observation.state": state_to_tensor(previous_action.copy()),
                "task": task,
            }
            processed = prepare_policy_batch(preprocessor(batch), device)

            with torch.no_grad():
                if not args.reuse_action_queue:
                    # SmolVLA is an action-chunking policy. For line following we want to replan from
                    # the newest image each loop instead of blindly consuming a 50-step open-loop chunk.
                    policy.reset()
                pred = policy.select_action(processed)

            postprocessed = postprocessor({"action": pred})
            model_action = action_to_numpy(postprocessed["action"])[:3]

            smoothed = smooth_alpha * previous_action + (1.0 - smooth_alpha) * np.clip(model_action, -1.0, 1.0)
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
                f"\r  task='{task}' pred=[{model_action[0]:6.3f},{model_action[1]:6.3f},{model_action[2]:6.3f}] "
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
        print("  SmolVLA drive stopped.")


if __name__ == "__main__":
    main()
