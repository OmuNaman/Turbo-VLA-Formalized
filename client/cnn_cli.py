"""CNN launcher branch and dataset recorder entry points."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from tasks import DEFAULT_INTENT_CNN_TASKS, TaskManager


def _run_cnn_dataset_recording(args: Namespace) -> None:
    """Run the no-language CNN dataset recorder."""
    from config import RecordingConfig

    from .cnn_loop_session import CNNLoopSession

    config = RecordingConfig(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        dataset_name=args.cnn_dataset,
        repo_id=args.repo_id,
        fps=args.fps,
        num_episodes=args.episodes,
        episode_time_s=args.episode_time,
        teleop_speed=args.speed,
        data_dir=Path(args.data_dir),
        session_name=args.session_name,
    )
    session = CNNLoopSession(config)
    session.run()


def _run_cnn_language_recording(args: Namespace) -> None:
    """Run the task-conditioned recorder shared by Intent-CNN and ACT-Intent."""
    from config import RecordingConfig

    from .cnn_language_session import CNNLanguageSession

    dataset_name = getattr(args, "intent_cnn_dataset", None) or args.cnn_dataset
    config = RecordingConfig(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        dataset_name=dataset_name,
        repo_id=args.repo_id,
        fps=args.fps,
        num_episodes=args.episodes,
        episode_time_s=args.episode_time,
        teleop_speed=args.speed,
        data_dir=Path(args.data_dir),
        session_name=args.session_name,
    )
    tasks = TaskManager(args.tasks if args.tasks else DEFAULT_INTENT_CNN_TASKS)
    session = CNNLanguageSession(config, tasks)
    session.run()


def run_from_args(args: Namespace, prompt_menu) -> None:
    """Handle the CNN launcher subtree."""
    if args.cnn_intent is None:
        selection = prompt_menu(
            "CNN Intent Options",
            ["intent-conditioned dataset (recommended)", "no-language loop mode (legacy)"],
        )
        if selection is None:
            return
        args.cnn_intent = "language" if selection == 0 else "no-language"

    if args.cnn_intent == "language":
        _run_cnn_language_recording(args)
        return

    if args.cnn_task is None:
        selection = prompt_menu(
            "CNN No-Language Options",
            ["dataset recording"],
        )
        if selection is None:
            return
        args.cnn_task = "dataset-recording"

    if args.cnn_task == "dataset-recording":
        _run_cnn_dataset_recording(args)
        return

    raise ValueError(f"Unsupported legacy CNN task selector: {args.cnn_task}")
