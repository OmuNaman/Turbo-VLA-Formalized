"""VLA recording entry point preserved behind the launcher."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def run_from_args(args: Namespace) -> None:
    """Run the existing VLA recorder using parsed launcher args."""
    from config import RecordingConfig
    from tasks import DEFAULT_TASKS, TaskManager

    from .recording_session import RecordingSession

    config = RecordingConfig(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        dataset_name=args.dataset,
        repo_id=args.repo_id,
        fps=args.fps,
        num_episodes=args.episodes,
        episode_time_s=args.episode_time,
        teleop_speed=args.speed,
        data_dir=Path(args.data_dir),
        session_name=args.session_name,
    )

    tasks = TaskManager(args.tasks if args.tasks else DEFAULT_TASKS)
    session = RecordingSession(config, tasks)
    session.run()
