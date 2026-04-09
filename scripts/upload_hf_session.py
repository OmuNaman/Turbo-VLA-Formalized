#!/usr/bin/env python3
"""Interactive uploader for recorded TurboPi sessions on Hugging Face."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
try:
    from huggingface_hub import HfApi, login
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:  # pragma: no cover - optional dependency
    HfApi = None
    HfHubHTTPError = Exception

    def login(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("huggingface_hub is not installed")

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
except ImportError:  # pragma: no cover - platform dependent
    tk = None
    filedialog = None
    messagebox = None
    simpledialog = None
    ttk = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES_ROOT = REPO_ROOT / "data"


@dataclass(frozen=True)
class SessionSummary:
    """Summary information about one saved recording session."""

    dataset_name: str
    session_name: str
    session_dir: Path
    raw_dir: Path | None
    episode_count: int
    frame_count: int
    duration_s: float
    session_fps: float
    labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Pick one recorded session and upload it to a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--episodes-root",
        default=str(DEFAULT_EPISODES_ROOT),
        help="Path to a data directory, dataset directory, or episodes directory to scan",
    )
    parser.add_argument("--session", default=None, help="Optional session name to preselect in CLI mode")
    parser.add_argument("--namespace", default=None, help="Hugging Face namespace/user or org")
    parser.add_argument("--repo-name", default=None, help="Dataset repo name. Defaults to the selected session name")
    parser.add_argument("--token", default=None, help="Hugging Face token. If omitted, cached login is used")
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private")
    parser.add_argument("--include-raw", action="store_true", help="Upload the matching raw session backup too")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without pushing anything")
    parser.add_argument("--no-gui", action="store_true", help="Use the terminal prompt instead of the Tk session picker")
    return parser


def ensure_hf_hub() -> None:
    """Fail early with a clear install hint when Hugging Face helpers are unavailable."""
    if HfApi is None:
        raise RuntimeError(
            "huggingface_hub is required for session upload. Install it with "
            "`pip install huggingface_hub` or `pip install -r requirements-laptop.txt`."
        )


def discover_episode_roots(path: Path) -> list[Path]:
    """Expand a data dir, dataset dir, or episodes dir into concrete episodes roots."""
    path = Path(path)
    if not path.exists():
        return []

    if path.name == "episodes":
        return [path]
    if (path / "episodes").exists():
        return [path / "episodes"]

    roots: list[Path] = []
    for child in sorted(path.iterdir()):
        candidate = child / "episodes"
        if candidate.exists() and candidate.is_dir():
            roots.append(candidate)
    return roots


def sanitize_repo_name(value: str) -> str:
    """Convert a folder-like name into a valid-ish Hugging Face repo name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-.")
    return cleaned or "turbopi-session"


def count_parquet_rows(path: Path) -> int:
    """Read the row count from Parquet metadata only."""
    return int(pq.ParquetFile(str(path)).metadata.num_rows)


def discover_sessions(episodes_root: Path) -> list[SessionSummary]:
    """Discover recorded sessions beneath one episodes root."""
    sessions: list[SessionSummary] = []

    for concrete_root in discover_episode_roots(Path(episodes_root)):
        dataset_name = concrete_root.parent.name
        raw_root = concrete_root.parent / "raw"

        for session_dir in sorted(path for path in concrete_root.iterdir() if path.is_dir()):
            episode_dirs = sorted(
                path
                for path in session_dir.iterdir()
                if path.is_dir() and path.name.startswith("episode_")
            )
            if not episode_dirs:
                continue

            session_info_path = session_dir / "session_info.json"
            try:
                session_info = json.loads(session_info_path.read_text(encoding="utf-8"))
            except Exception:
                session_info = {}
            session_fps = float(session_info.get("fps", 10.0) or 10.0)

            frame_count = 0
            valid_episodes = 0
            labels: list[str] = []
            for episode_dir in episode_dirs:
                parquet_path = episode_dir / "data.parquet"
                if not parquet_path.exists():
                    continue
                frame_count += count_parquet_rows(parquet_path)
                valid_episodes += 1

                info_path = episode_dir / "episode_info.json"
                if info_path.exists():
                    try:
                        info = json.loads(info_path.read_text(encoding="utf-8"))
                        label = str(
                            info.get("task_name")
                            or info.get("task")
                            or info.get("direction")
                            or ""
                        ).strip()
                        if label:
                            labels.append(label)
                    except json.JSONDecodeError:
                        pass

            if valid_episodes == 0:
                continue

            raw_dir = raw_root / session_dir.name
            sessions.append(
                SessionSummary(
                    dataset_name=dataset_name,
                    session_name=session_dir.name,
                    session_dir=session_dir,
                    raw_dir=raw_dir if raw_dir.exists() else None,
                    episode_count=valid_episodes,
                    frame_count=frame_count,
                    duration_s=frame_count / max(session_fps, 1e-6),
                    session_fps=session_fps,
                    labels=tuple(sorted(set(labels)) or ["unknown"]),
                )
            )
    return sessions


def format_labels(labels: Iterable[str]) -> str:
    """Render the unique task labels in one short string."""
    return ", ".join(labels)


def describe_session(summary: SessionSummary) -> str:
    """Human-readable description of one session."""
    return (
        f"{summary.session_name} | episodes={summary.episode_count} | "
        f"frames={summary.frame_count} | duration={summary.duration_s:.1f}s | "
        f"tasks={format_labels(summary.labels)}"
    )


def repo_card_text(summary: SessionSummary, repo_id: str, include_raw: bool) -> str:
    """Create a small dataset card for the uploaded repo."""
    raw_note = "included" if include_raw and summary.raw_dir else "not included"
    return textwrap.dedent(
        f"""\
        ---
        language:
        - en
        license: mit
        task_categories:
        - robotics
        pretty_name: {summary.session_name}
        size_categories:
        - n<1K
        ---

        # {summary.session_name}

        This dataset repo was uploaded from the TurboPi recording stack.

        ## Summary

        - Repo id: `{repo_id}`
        - Dataset root: `{summary.dataset_name}`
        - Session: `{summary.session_name}`
        - Accepted episodes: `{summary.episode_count}`
        - Accepted frames: `{summary.frame_count}`
        - Approx duration at `{summary.session_fps:g} Hz`: `{summary.duration_s:.1f}s`
        - Tasks seen: `{format_labels(summary.labels)}`
        - Raw backup: `{raw_note}`

        ## Layout

        - `episodes/`: accepted episode folders for this selected session
        - `raw/`: optional raw backup for the same session
        - `upload_manifest.json`: machine-readable upload metadata

        ## Notes

        This repo contains one selected TurboPi session only. The repo name intentionally matches the session folder name.
        """
    ).strip() + "\n"


def build_manifest(summary: SessionSummary, repo_id: str, include_raw: bool) -> dict[str, object]:
    """Create machine-readable upload metadata."""
    return {
        "repo_id": repo_id,
        "dataset_name": summary.dataset_name,
        "session_name": summary.session_name,
        "episodes_dir": str(summary.session_dir),
        "raw_dir": str(summary.raw_dir) if include_raw and summary.raw_dir else None,
        "episode_count": summary.episode_count,
        "frame_count": summary.frame_count,
        "duration_s": summary.duration_s,
        "labels": list(summary.labels),
        "session_fps": summary.session_fps,
        "include_raw": bool(include_raw and summary.raw_dir),
    }


def stage_upload_folder(summary: SessionSummary, *, repo_id: str, include_raw: bool) -> Path:
    """Create a temporary upload folder that contains only the selected session."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="turbopi_hf_upload_"))
    episodes_target = tmp_dir / "episodes"
    shutil.copytree(summary.session_dir, episodes_target)

    if include_raw and summary.raw_dir is not None:
        shutil.copytree(summary.raw_dir, tmp_dir / "raw")

    (tmp_dir / "README.md").write_text(repo_card_text(summary, repo_id, include_raw), encoding="utf-8")
    (tmp_dir / "upload_manifest.json").write_text(
        json.dumps(build_manifest(summary, repo_id, include_raw), indent=2),
        encoding="utf-8",
    )
    return tmp_dir


def resolve_auth(api: HfApi, *, explicit_token: str | None, namespace: str | None) -> tuple[str | None, str]:
    """Resolve a usable token and namespace from explicit or cached login."""
    token = explicit_token.strip() if explicit_token else None
    try:
        whoami = api.whoami(token=token)
    except Exception as exc:
        if token:
            raise RuntimeError(f"Could not authenticate with the provided token: {exc}") from exc
        if namespace:
            return None, namespace.strip()
        raise RuntimeError(
            "No Hugging Face login found. Set HF_TOKEN, pass --token, or use the GUI login button."
        ) from exc

    detected_namespace = (
        namespace.strip()
        if namespace and namespace.strip()
        else str(whoami.get("name") or whoami.get("user") or "").strip()
    )
    if not detected_namespace:
        raise RuntimeError("Authenticated successfully, but could not determine the Hugging Face namespace.")
    return token, detected_namespace


def upload_selected_session(
    summary: SessionSummary,
    *,
    namespace: str,
    repo_name: str,
    token: str | None,
    private: bool,
    include_raw: bool,
    dry_run: bool,
) -> tuple[str, Path | None]:
    """Upload one selected session to a dataset repo."""
    repo_name = sanitize_repo_name(repo_name or summary.session_name)
    repo_id = f"{namespace}/{repo_name}"

    staged_dir = stage_upload_folder(summary, repo_id=repo_id, include_raw=include_raw)
    if dry_run:
        return repo_id, staged_dir

    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=staged_dir,
            repo_type="dataset",
            token=token,
            commit_message=f"Upload TurboPi session {summary.session_name}",
            commit_description=(
                f"Dataset={summary.dataset_name}, episodes={summary.episode_count}, frames={summary.frame_count}"
            ),
        )
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)

    return repo_id, None


def find_session(sessions: list[SessionSummary], requested_name: str | None) -> SessionSummary | None:
    """Find one session by exact folder name."""
    if not requested_name:
        return None
    for summary in sessions:
        if summary.session_name == requested_name:
            return summary
    return None


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for a yes/no answer in CLI mode."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Enter y or n.")


def run_cli(args: argparse.Namespace) -> int:
    """Run the uploader in terminal mode."""
    ensure_hf_hub()
    episodes_root = Path(args.episodes_root)
    sessions = discover_sessions(episodes_root)
    if not sessions:
        print(f"No sessions found under {episodes_root}")
        return 1

    summary = find_session(sessions, args.session)
    if summary is None:
        print()
        print("Available sessions:")
        for index, session in enumerate(sessions):
            print(f"  [{index}] {describe_session(session)}")
        print()
        while True:
            choice = input("Select session number: ").strip()
            try:
                idx = int(choice)
                summary = sessions[idx]
                break
            except (ValueError, IndexError):
                print(f"Choose a number between 0 and {len(sessions) - 1}.")

    assert summary is not None
    print()
    print("Selected session:")
    print(f"  {describe_session(summary)}")

    include_raw = args.include_raw or prompt_yes_no("Include raw backup if available?", default=False)
    private = args.private or prompt_yes_no("Create the Hugging Face repo as private?", default=True)

    api = HfApi()
    token = args.token.strip() if args.token else None
    namespace = args.namespace.strip() if args.namespace else None
    if not args.dry_run:
        if not token:
            token = input("Hugging Face token (leave blank to use cached login): ").strip() or None
            if token:
                login(token=token, add_to_git_credential=False)
        token, namespace = resolve_auth(api, explicit_token=token, namespace=namespace)
    else:
        namespace = namespace or "dry-run"

    repo_name = args.repo_name or summary.session_name
    repo_id, staged_dir = upload_selected_session(
        summary,
        namespace=namespace,
        repo_name=repo_name,
        token=token,
        private=private,
        include_raw=include_raw,
        dry_run=args.dry_run,
    )

    print()
    if args.dry_run:
        print("Dry run complete")
        print(f"  repo_id:     {repo_id}")
        print(f"  staged_dir:  {staged_dir}")
        print(f"  upload_from: {summary.session_dir}")
        return 0

    print("Upload complete")
    print(f"  repo_id:  {repo_id}")
    print(f"  url:      https://huggingface.co/datasets/{repo_id}")
    return 0


class SessionUploaderApp:
    """Small Tk app for selecting and uploading one recorded session."""

    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        ensure_hf_hub()
        self.api = HfApi()
        self.sessions: list[SessionSummary] = []
        self.auth_token = args.token.strip() if args.token else None

        self.episodes_root_var = tk.StringVar(value=str(Path(args.episodes_root)))
        self.namespace_var = tk.StringVar(value=args.namespace or "")
        self.repo_name_var = tk.StringVar(value=args.repo_name or "")
        self.private_var = tk.BooleanVar(value=bool(args.private))
        self.include_raw_var = tk.BooleanVar(value=bool(args.include_raw))
        self.dry_run_var = tk.BooleanVar(value=bool(args.dry_run))
        self.status_var = tk.StringVar(value="Loading sessions...")
        self.auth_var = tk.StringVar(value="Checking Hugging Face login...")

        self._build()
        self._load_auth()
        self.refresh_sessions()

    def _build(self) -> None:
        self.root.title("TurboPi Hugging Face Session Uploader")
        self.root.geometry("980x620")
        self.root.minsize(900, 560)

        container = ttk.Frame(self.root, padding=14)
        container.pack(fill="both", expand=True)

        root_row = ttk.LabelFrame(container, text="Episodes Root", padding=10)
        root_row.pack(fill="x")
        ttk.Entry(root_row, textvariable=self.episodes_root_var).pack(side="left", fill="x", expand=True)
        ttk.Button(root_row, text="Browse", command=self.browse_episodes_root).pack(side="left", padx=(8, 0))
        ttk.Button(root_row, text="Refresh", command=self.refresh_sessions).pack(side="left", padx=(8, 0))

        auth_row = ttk.LabelFrame(container, text="Hugging Face", padding=10)
        auth_row.pack(fill="x", pady=(12, 0))
        ttk.Label(auth_row, textvariable=self.auth_var).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(auth_row, text="Namespace").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(auth_row, textvariable=self.namespace_var, width=28).grid(row=1, column=1, sticky="we", padx=(8, 16), pady=(8, 0))
        ttk.Label(auth_row, text="Repo Name").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(auth_row, textvariable=self.repo_name_var, width=30).grid(row=1, column=3, sticky="we", padx=(8, 0), pady=(8, 0))
        ttk.Button(auth_row, text="Set Token", command=self.prompt_login).grid(row=1, column=4, padx=(12, 0), pady=(8, 0))
        auth_row.columnconfigure(1, weight=1)
        auth_row.columnconfigure(3, weight=1)

        options_row = ttk.Frame(container)
        options_row.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(options_row, text="Private repo", variable=self.private_var).pack(side="left")
        ttk.Checkbutton(options_row, text="Include raw backup", variable=self.include_raw_var).pack(side="left", padx=(16, 0))
        ttk.Checkbutton(options_row, text="Dry run only", variable=self.dry_run_var).pack(side="left", padx=(16, 0))

        table_frame = ttk.LabelFrame(container, text="Recorded Sessions", padding=10)
        table_frame.pack(fill="both", expand=True, pady=(12, 0))

        columns = ("dataset", "session", "episodes", "frames", "duration", "tasks")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=16)
        headings = {
            "dataset": "Dataset",
            "session": "Session",
            "episodes": "Episodes",
            "frames": "Frames",
            "duration": "Duration (s)",
            "tasks": "Tasks",
        }
        widths = {
            "dataset": 130,
            "session": 220,
            "episodes": 80,
            "frames": 90,
            "duration": 100,
            "tasks": 240,
        }
        for key in columns:
            self.tree.heading(key, text=headings[key])
            self.tree.column(key, width=widths[key], anchor="center" if key != "session" else "w")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_session)

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scroll.set)

        footer = ttk.Frame(container)
        footer.pack(fill="x", pady=(12, 0))
        ttk.Label(footer, textvariable=self.status_var).pack(side="left")
        ttk.Button(footer, text="Upload Selected Session", command=self.upload_selected).pack(side="right")

    def _load_auth(self) -> None:
        """Load cached or explicit authentication."""
        try:
            if self.auth_token:
                login(token=self.auth_token, add_to_git_credential=False)
            _, namespace = resolve_auth(self.api, explicit_token=self.auth_token, namespace=self.namespace_var.get() or None)
        except Exception as exc:
            self.auth_var.set(f"Not logged in yet: {exc}")
            return

        if not self.namespace_var.get().strip():
            self.namespace_var.set(namespace)
        self.auth_var.set(f"Authenticated as {namespace}")

    def browse_episodes_root(self) -> None:
        """Pick an episodes root folder from the filesystem."""
        if filedialog is None:
            self.status_var.set("Tk file dialog is unavailable in this environment.")
            return
        selected = filedialog.askdirectory(
            title="Select an episodes root",
            initialdir=self.episodes_root_var.get() or str(REPO_ROOT / "data"),
        )
        if selected:
            self.episodes_root_var.set(selected)
            self.refresh_sessions()

    def refresh_sessions(self) -> None:
        """Reload the session table from the selected episodes root."""
        episodes_root = Path(self.episodes_root_var.get())
        self.sessions = discover_sessions(episodes_root)

        for item in self.tree.get_children():
            self.tree.delete(item)

        for index, session in enumerate(self.sessions):
            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    session.dataset_name,
                    session.session_name,
                    session.episode_count,
                    session.frame_count,
                    f"{session.duration_s:.1f}",
                    format_labels(session.labels),
                ),
            )

        if self.sessions:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self.on_select_session()
            self.status_var.set(f"Loaded {len(self.sessions)} sessions from {episodes_root}")
        else:
            self.repo_name_var.set("")
            self.status_var.set(f"No sessions found under {episodes_root}")

    def selected_session(self) -> SessionSummary | None:
        """Return the currently selected session or None."""
        selection = self.tree.selection()
        if not selection:
            return None
        try:
            return self.sessions[int(selection[0])]
        except (ValueError, IndexError):
            return None

    def on_select_session(self, _event=None) -> None:
        """Update derived fields after the selected row changes."""
        session = self.selected_session()
        if session is None:
            return
        if not self.repo_name_var.get().strip() or self.repo_name_var.get().strip().startswith("session_"):
            self.repo_name_var.set(session.session_name)
        self.status_var.set(describe_session(session))

    def prompt_login(self) -> None:
        """Prompt for a Hugging Face token and cache it."""
        if simpledialog is None:
            self.status_var.set("Tk dialogs are unavailable in this environment.")
            return
        token = simpledialog.askstring(
            "Hugging Face Token",
            "Paste your Hugging Face token:",
            show="*",
            parent=self.root,
        )
        if not token:
            return

        try:
            login(token=token.strip(), add_to_git_credential=False)
            self.auth_token = token.strip()
            _, namespace = resolve_auth(self.api, explicit_token=self.auth_token, namespace=self.namespace_var.get() or None)
        except Exception as exc:
            messagebox.showerror("Login failed", str(exc), parent=self.root)
            return

        if not self.namespace_var.get().strip():
            self.namespace_var.set(namespace)
        self.auth_var.set(f"Authenticated as {namespace}")
        self.status_var.set("Hugging Face token saved for this session.")

    def upload_selected(self) -> None:
        """Upload the selected session."""
        session = self.selected_session()
        if session is None:
            messagebox.showwarning("No session selected", "Pick one session first.", parent=self.root)
            return

        namespace = self.namespace_var.get().strip()
        repo_name = self.repo_name_var.get().strip() or session.session_name
        include_raw = self.include_raw_var.get()
        dry_run = self.dry_run_var.get()

        try:
            token = self.auth_token
            if not dry_run:
                token, namespace = resolve_auth(self.api, explicit_token=token, namespace=namespace or None)
            else:
                namespace = namespace or "dry-run"

            self.root.config(cursor="watch")
            self.status_var.set(f"Uploading {session.session_name}...")
            self.root.update_idletasks()

            repo_id, staged_dir = upload_selected_session(
                session,
                namespace=namespace,
                repo_name=repo_name,
                token=token,
                private=self.private_var.get(),
                include_raw=include_raw,
                dry_run=dry_run,
            )
        except RuntimeError as exc:
            messagebox.showerror("Upload failed", str(exc), parent=self.root)
            self.status_var.set(str(exc))
            return
        except HfHubHTTPError as exc:
            messagebox.showerror("Hugging Face API error", str(exc), parent=self.root)
            self.status_var.set(f"API error: {exc}")
            return
        except Exception as exc:
            messagebox.showerror("Upload failed", str(exc), parent=self.root)
            self.status_var.set(f"Upload failed: {exc}")
            return
        finally:
            self.root.config(cursor="")

        if dry_run:
            messagebox.showinfo(
                "Dry run complete",
                f"Repo id: {repo_id}\nStaged folder: {staged_dir}",
                parent=self.root,
            )
            self.status_var.set(f"Dry run complete for {repo_id}")
            return

        url = f"https://huggingface.co/datasets/{repo_id}"
        messagebox.showinfo("Upload complete", f"Uploaded successfully.\n\n{url}", parent=self.root)
        self.status_var.set(f"Upload complete: {url}")


def run_gui(args: argparse.Namespace) -> int:
    """Run the Tk session picker."""
    if tk is None:
        raise RuntimeError("Tkinter is not available, so GUI mode cannot be used.")

    root = tk.Tk()
    app = SessionUploaderApp(root, args)
    if args.session:
        for index, session in enumerate(app.sessions):
            if session.session_name == args.session:
                app.tree.selection_set(str(index))
                app.tree.focus(str(index))
                app.on_select_session()
                break
    root.mainloop()
    return 0


def main() -> int:
    """Program entrypoint."""
    args = build_parser().parse_args()
    if args.no_gui or tk is None:
        return run_cli(args)
    try:
        return run_gui(args)
    except Exception as exc:
        print(f"GUI mode failed: {exc}")
        print("Falling back to terminal mode...")
        return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
