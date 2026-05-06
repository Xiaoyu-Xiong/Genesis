from __future__ import annotations

import contextlib
import fcntl
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from code_agent.utils.local_execution import LocalRunConfig, run_local


_GENESIS_EXECUTION_THREAD_LOCK = threading.Lock()


@dataclass(slots=True)
class ExecutionReport:
    command: list[str]
    returncode: int
    duration_sec: float
    stdout_path: str
    stderr_path: str
    artifacts: dict[str, str]
    lock_wait_sec: float = 0.0
    lock_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "duration_sec": self.duration_sec,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "artifacts": self.artifacts,
            "lock_wait_sec": self.lock_wait_sec,
            "lock_path": self.lock_path,
            "ok": self.ok,
        }


def run_generated_simulation(
    *,
    main_py: Path,
    run_dir: Path,
    backend: str,
    timeout_sec: float,
    steps: int,
    render_fps: int,
    render: bool = True,
    duration_sec: float | None = None,
    target_video_frames: int | None = None,
) -> ExecutionReport:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_dir_abs = run_dir.resolve()
    main_path = main_py.resolve()
    try:
        main_file = str(main_path.relative_to(run_dir_abs))
    except ValueError:
        main_file = str(main_path)
    render_arg = "--render" if render else "--no-render"
    extra_args = [
        "--backend",
        backend,
        "--out-dir",
        "artifacts",
        "--steps",
        str(int(steps)),
        "--fps",
        str(int(render_fps)),
    ]
    if duration_sec is not None:
        extra_args.extend(("--duration-sec", str(float(duration_sec))))
    if target_video_frames is not None:
        extra_args.extend(("--target-video-frames", str(int(target_video_frames))))
    deformable_config_path = run_dir / "contracts" / "deformable_config.json"
    if deformable_config_path.exists():
        extra_args.extend(("--deformable-config", str(deformable_config_path.relative_to(run_dir))))
    extra_args.append(render_arg)
    with _exclusive_genesis_execution_lock() as lock_info:
        raw_report = run_local(
            LocalRunConfig(
                workspace_dir=run_dir,
                main_file=main_file,
                output_dir=reports_dir,
                timeout_sec=timeout_sec,
                python_executable="uv run python",
                extra_args=tuple(extra_args),
                extra_artifact_paths=("artifacts",),
                env={"GENESIS_BACKEND": backend},
            )
        )
    artifact_paths = raw_report.get("artifact_paths", [])
    artifacts = {Path(path).stem: path for path in artifact_paths if isinstance(path, str)}
    report = ExecutionReport(
        command=list(raw_report["command"]),
        returncode=int(raw_report["exit_code"]),
        duration_sec=float(raw_report["duration_sec"]),
        stdout_path=str(raw_report["stdout_path"]),
        stderr_path=str(raw_report["stderr_path"]),
        artifacts=artifacts,
        lock_wait_sec=float(lock_info["wait_sec"]),
        lock_path=str(lock_info["path"]),
    )
    return report


@contextlib.contextmanager
def _exclusive_genesis_execution_lock():
    """Serialize local Genesis simulation subprocesses across parallel suite cases."""

    lock_path = Path(tempfile.gettempdir()) / f"genesis_code_agent_{os.getuid()}_execution.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with _GENESIS_EXECUTION_THREAD_LOCK:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            wait_sec = time.time() - started
            try:
                yield {"path": lock_path, "wait_sec": wait_sec}
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
