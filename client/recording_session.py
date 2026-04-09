"""Main recording session orchestrator for the 10 Hz teleop loop."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from config import RecordingConfig
from storage.episode_writer import EpisodeWriter
from storage.raw_writer import RawWriter
from tasks import CUSTOM_TASK_LABEL, TaskManager, load_saved_tasks
from timing import FPSRegulator

from .episode_manager import EpisodeManager
from .robot_client import RobotClient
from .session_state import inspect_saved_session
from .teleop_controller import TeleopController


def _flush_stdin() -> None:
    """Flush buffered stdin so input() does not consume stale keypresses."""
    try:
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass


class RecordingSession:
    """Orchestrates the full recording session."""

    def __init__(self, config: RecordingConfig, tasks: TaskManager | None = None):
        self.config = config
        self.tasks = tasks or TaskManager()

        self.config.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.client = RobotClient(
            robot_url=config.robot_url,
            timeout=0.5,
            max_retries=1,
        )
        self.teleop = TeleopController(
            speed=config.teleop_speed,
            max_speed=config.max_duty,
        )
        self.session_name = config.session_name or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.tasks.merge_tasks(load_saved_tasks(config.episodes_dir / self.session_name))
        self.resume_state = inspect_saved_session(config.episodes_dir / self.session_name)
        self.episodes = EpisodeManager(
            start_episode_index=self.resume_state.next_episode_index,
            accepted_count=self.resume_state.accepted_count,
            total_frames=self.resume_state.total_frames,
        )
        self.fps_reg = FPSRegulator(target_fps=config.fps)

        self.episode_writer = EpisodeWriter(
            episodes_dir=config.episodes_dir / self.session_name,
            fps=config.fps,
            vcodec=config.vcodec,
        )
        self.raw_writer = RawWriter(
            session_dir=config.raw_dir / self.session_name,
            fps=config.fps,
            vcodec=config.vcodec,
        )

        self._running = False
        self._last_health_check = 0.0
        self._health: dict = {}

        self._write_session_info(increment_resume=True)

    def run(self) -> None:
        """Main recording loop that records episodes until stopped."""
        print()
        print("=" * 50)
        print("  TurboPi VLA Recording Session")
        print("=" * 50)

        print("\n  Connecting to robot...")
        if not self.client.is_connected():
            print(f"  ERROR: Cannot reach robot at {self.config.robot_url}")
            print("  Make sure the robot server is running.")
            return

        health = self.client.get_health()
        print(
            f"  Connected! Battery: {health.get('battery_mv', '?')}mV, "
            f"Camera: {'OK' if health.get('camera_ok') else 'FAIL'}"
        )
        if self.resume_state.accepted_count > 0:
            print(
                f"  Resuming session {self.session_name}: "
                f"{self.resume_state.accepted_count} accepted episodes, "
                f"{self.resume_state.total_frames} frames already saved"
            )

        if not self.episode_writer.video_available:
            print("  ERROR: PyAV is required to save accepted episodes as MP4.")
            print("  Install it on the laptop with: pip install av")
            return

        self.teleop.start()
        self.raw_writer.start()
        self.episode_writer.save_task_mapping(self.tasks.tasks)
        self._running = True

        print("\n  Controls:")
        print("    WASD+QE  = drive robot")
        print("    +/-      = speed up/down")
        print("    right    = accept episode / start recording")
        print("    left     = discard episode")
        print("    ESC      = stop session")

        try:
            episode_num = 0
            while self._running and episode_num < self.config.num_episodes:
                task, task_index = self._select_task()
                if not self._running:
                    break

                print(
                    "\n  Drive the robot into position, then press right arrow to start recording."
                )
                self.teleop.clear_events()
                self._drive_until_ready()
                if self.teleop.events["stop_session"]:
                    break

                self.teleop.clear_events()
                accepted = self._record_episode(task, task_index)

                if accepted:
                    episode_num += 1
                    print(
                        f"  Total accepted: {self.episodes.accepted_count} episodes, "
                        f"{self.episodes.total_frames} frames"
                    )

                if self.teleop.events["stop_session"]:
                    break

        except KeyboardInterrupt:
            print("\n\n  Ctrl+C - stopping...")
        finally:
            self._shutdown()

    def _write_session_info(self, *, increment_resume: bool) -> None:
        """Persist session-level metadata beside raw and accepted outputs."""
        now = datetime.now().isoformat(timespec="seconds")
        session_info = {
            "created_at": now,
            "last_updated_at": now,
            "session_name": self.session_name,
            "dataset_name": self.config.dataset_name,
            "robot_url": self.config.robot_url,
            "robot_type": self.config.robot_type,
            "fps": self.config.fps,
            "episode_time_s": self.config.episode_time_s,
            "max_duty": self.config.max_duty,
            "teleop_speed": self.config.teleop_speed,
            "vcodec": self.config.vcodec,
            "tasks": self.tasks.tasks,
            "mode_family": "vla",
            "task_selection_mode": "listed_plus_custom",
            "custom_task_enabled": True,
            "observation_state_semantics": "previous_action_normalized",
            "action_semantics": "current_action_normalized",
            "accepted_episode_timestamps": "episode_relative_seconds",
            "raw_backup_timestamps": "robot_monotonic_seconds",
        }

        for base_dir in (self.config.raw_dir, self.config.episodes_dir):
            session_dir = base_dir / self.session_name
            session_dir.mkdir(parents=True, exist_ok=True)
            path = session_dir / "session_info.json"
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        existing = json.load(handle)
                    session_info["created_at"] = existing.get("created_at", session_info["created_at"])
                    existing_resume_count = int(existing.get("resume_count", 0))
                    session_info["resume_count"] = (
                        existing_resume_count + 1 if increment_resume else existing_resume_count
                    )
                except Exception:
                    session_info["resume_count"] = 1 if increment_resume else 0
            else:
                session_info["resume_count"] = 0
            with path.open("w", encoding="utf-8") as handle:
                json.dump(session_info, handle, indent=2)

    def _persist_task_catalog(self) -> None:
        """Rewrite the session task mapping and session metadata after task changes."""
        self.episode_writer.save_task_mapping(self.tasks.tasks)
        self._write_session_info(increment_resume=False)

    def _prompt_for_custom_task(self) -> str | None:
        """Prompt for a custom task string and persist it if it is new."""
        while True:
            try:
                _flush_stdin()
                custom_task = input("  Enter custom task text: ").strip()
            except EOFError:
                self._running = False
                return None

            if not custom_task:
                print("  Task text cannot be empty.")
                continue

            is_new = not self.tasks.has_task(custom_task)
            task_index = self.tasks.get_index(custom_task)
            if is_new:
                self._persist_task_catalog()
                print(f'  -> Added custom task [{task_index}]: "{custom_task}"')
            else:
                print(f'  -> Reusing existing task [{task_index}]: "{custom_task}"')
            return custom_task

    def _select_task(self) -> tuple[str, int]:
        """Prompt the user to select a task."""
        self.tasks.print_tasks(include_custom_option=True)
        time.sleep(0.3)
        _flush_stdin()

        while True:
            try:
                _flush_stdin()
                choice = input("  Select task number: ").strip()
                idx = int(choice)
                if 0 <= idx < len(self.tasks):
                    task = self.tasks.get_task(idx)
                    print(f'  -> Task: "{task}"')
                    return task, idx
                if idx == len(self.tasks):
                    custom_task = self._prompt_for_custom_task()
                    if custom_task is None:
                        return "", 0
                    return custom_task, self.tasks.get_index(custom_task)
                print(f"  Invalid. Choose 0-{len(self.tasks)}")
            except ValueError:
                print(f"  Enter a number or choose {CUSTOM_TASK_LABEL}.")
            except EOFError:
                self._running = False
                return "", 0

    def _drive_until_ready(self) -> None:
        """Let the user drive until right arrow is pressed to start recording."""
        print("  Driving mode - press right arrow when ready to record...\n")
        self.teleop.clear_events()

        while not self.teleop.events["accept_episode"] and not self.teleop.events["stop_session"]:
            vx, vy, omega = self.teleop.get_action()
            try:
                sent = self.client.send_velocity(vx, vy, omega)
                error = ""
            except Exception as exc:
                sent = False
                error = str(exc)

            if vx > 0:
                status = "FWD"
            elif vx < 0:
                status = "BWD"
            elif vy > 0:
                status = "LEFT"
            elif vy < 0:
                status = "RIGHT"
            elif omega > 0:
                status = "ROT_L"
            elif omega < 0:
                status = "ROT_R"
            else:
                status = "STOP"

            line = f"\r  [{status:<6}] speed={self.teleop.speed:.0f}%"
            if not sent:
                line += "  WARN: command not delivered"
                if error:
                    line += f": {error}"
            print(f"{line}  ", end="", flush=True)
            time.sleep(0.1)

        self.client.stop()
        self.teleop.clear_events()
        print("\r" + " " * 60 + "\r", end="")

    def _record_episode(self, task: str, task_index: int) -> bool:
        """Record a single episode at target FPS."""
        self.episodes.start_episode(task, task_index)
        self.fps_reg.reset()
        self.teleop.clear_events()

        ep_idx = self.episodes.current.episode_index
        print(f"\n  RECORDING Episode {ep_idx}  [{task}]")
        print(f"    Max duration: {self.config.episode_time_s:.0f}s")
        print("    right = accept, left = discard, ESC = stop\n")

        start_time = time.monotonic()
        frame_count = 0
        moving_frames = 0
        previous_action = np.zeros(3, dtype=np.float32)

        while True:
            self.fps_reg.tick()
            self._check_health()

            elapsed = time.monotonic() - start_time
            if elapsed >= self.config.episode_time_s:
                print(f"\r  Time limit reached ({self.config.episode_time_s:.0f}s)")
                break
            if self.teleop.events["accept_episode"]:
                break
            if self.teleop.events["discard_episode"]:
                break
            if self.teleop.events["stop_session"]:
                break

            try:
                image_rgb, robot_ts, _ = self.client.get_frame_rgb()
            except Exception as exc:
                print(f"\r  [WARN] Frame grab failed: {exc}   ", end="", flush=True)
                continue

            vx, vy, omega = self.teleop.get_action()
            action = np.array([vx, vy, omega], dtype=np.float32) / self.config.max_duty
            state = previous_action.copy()

            try:
                sent = self.client.send_velocity(vx, vy, omega)
            except Exception as exc:
                print(f"\r  [WARN] Motor command failed: {exc}   ", end="", flush=True)
                continue

            if not sent:
                print("\r  [WARN] Robot rejected the velocity command.   ", end="", flush=True)
                continue

            episode_ts = frame_count / self.config.fps
            self.episodes.add_frame(image_rgb, state, action, episode_ts)
            self.raw_writer.write_frame(
                image=image_rgb,
                state=state,
                action=action,
                timestamp=robot_ts,
                task=task,
                task_index=task_index,
                episode_index=ep_idx,
            )

            previous_action = action.copy()
            if not np.allclose(action, 0.0, atol=1e-6):
                moving_frames += 1
            frame_count += 1

            fps_str = f"{self.fps_reg.actual_fps:.1f}" if frame_count > 2 else "..."
            print(
                f"\r  REC {elapsed:5.1f}s  frames={frame_count}  "
                f"fps={fps_str}  speed={self.teleop.speed:.0f}%   ",
                end="",
                flush=True,
            )

        self.client.stop()
        print()

        if self.teleop.events["discard_episode"]:
            self.episodes.discard_episode()
            print("  x Episode discarded.\n")
            return False
        if frame_count < 5:
            self.episodes.discard_episode()
            print(f"  x Episode too short ({frame_count} frames), discarded.\n")
            return False
        if moving_frames < 3:
            self.episodes.discard_episode()
            print(
                f"  x Episode had too little movement ({moving_frames} moving frames), "
                "discarded.\n"
            )
            return False

        episode = self.episodes.accept_episode()
        episode_dir = self.episode_writer.save_episode(episode)
        self._write_episode_info(episode_dir=episode_dir, task=task, task_index=task_index, episode=episode)
        print(
            f"  ok Episode {ep_idx} accepted "
            f"({frame_count} frames, {frame_count / self.config.fps:.1f}s)\n"
        )
        return True

    def _write_episode_info(self, episode_dir: Path, task: str, task_index: int, episode) -> None:
        """Save task-conditioned VLA episode metadata beside saved media."""
        info = {
            "episode_index": episode.episode_index,
            "mode_family": "vla",
            "task_name": task,
            "task_index": task_index,
            "num_frames": len(episode.frames),
            "duration_s": len(episode.frames) / self.config.fps,
            "observation_state_semantics": "previous_action_normalized",
            "action_semantics": "current_action_normalized",
        }
        with (episode_dir / "episode_info.json").open("w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)

    def _check_health(self) -> None:
        """Run a periodic robot health check without blocking the control loop."""
        now = time.monotonic()
        if now - self._last_health_check < 30:
            return
        self._last_health_check = now

        try:
            self._health = self.client.get_health()
        except Exception:
            return

        battery_mv = self._health.get("battery_mv", 0)
        if battery_mv and battery_mv < 7200:
            print(f"\n  [WARN] Battery is low ({battery_mv}mV). Consider charging soon.")

        if not self._health.get("camera_ok", True):
            print("\n  [WARN] Robot reports camera problems.")

    def _shutdown(self) -> None:
        """Clean shutdown of all components."""
        print("\n  Shutting down...")

        try:
            self.client.stop()
        except Exception:
            pass

        if self.episodes.is_recording:
            self.episodes.discard_episode()
            print("  Discarded in-progress episode.")

        self.raw_writer.close()
        self.teleop.stop()
        _flush_stdin()

        print(f"\n  Session complete: {self.session_name}")
        print(f"    Accepted episodes: {self.episodes.accepted_count}")
        print(f"    Total frames: {self.episodes.total_frames}")
        print(f"    Episodes: {self.config.episodes_dir / self.session_name}")
        print(f"    Raw data: {self.config.raw_dir / self.session_name}")
        print()
