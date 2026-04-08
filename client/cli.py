"""Top-level client launcher for VLA and CNN recording modes."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the launcher."""
    parser = argparse.ArgumentParser(description="TurboPi Client Launcher")
    parser.add_argument("--robot-ip", default="192.168.149.1")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--dataset", default="turbopi_nav",
                        help="Dataset name for the VLA recorder")
    parser.add_argument("--cnn-dataset", default="turbopi_cnn",
                        help="Dataset name for the CNN recorder")
    parser.add_argument("--intent-cnn-dataset", default="turbopi_intent_cnn",
                        help="Dataset name for the language-intent CNN recorder")
    parser.add_argument("--repo-id", default="<HF_DATASET_REPO>")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode-time", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=50.0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--session-name", default=None,
                        help="Resume or force a specific session folder name under the chosen dataset")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Custom VLA task list (overrides defaults)")
    parser.add_argument("--mode", choices=["launcher", "cnn", "vla"], default="launcher")
    parser.add_argument("--cnn-intent", choices=["language", "no-language"], default=None)
    parser.add_argument("--cnn-task", default=None,
                        help="Internal CNN task selector; public mode uses dataset-recording")
    return parser


def _prompt_menu(title: str, options: list[str]) -> int | None:
    """Prompt the user to select one option from a numbered list."""
    print(f"\n  {title}:")
    for index, label in enumerate(options):
        print(f"    [{index}] {label}")
    print()

    while True:
        try:
            choice = input("  Select number: ").strip()
        except EOFError:
            return None

        if choice == "":
            print("  Enter a number.")
            continue

        try:
            index = int(choice)
        except ValueError:
            print("  Enter a number.")
            continue

        if 0 <= index < len(options):
            return index

        print(f"  Invalid. Choose 0-{len(options) - 1}")

def main() -> None:
    """Run the top-level launcher."""
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "vla":
        from .vla_cli import run_from_args as run_vla

        run_vla(args)
        return

    if args.mode == "cnn":
        from .cnn_cli import run_from_args as run_cnn

        run_cnn(args, _prompt_menu)
        return

    print()
    print("=" * 50)
    print("  TurboPi Client Launcher")
    print("=" * 50)

    selection = _prompt_menu("Recording Modes", ["CNN-based", "VLA-based"])
    if selection is None:
        return

    if selection == 0:
        from .cnn_cli import run_from_args as run_cnn

        run_cnn(args, _prompt_menu)
        return

    from .vla_cli import run_from_args as run_vla

    run_vla(args)
    return


if __name__ == "__main__":
    main()
