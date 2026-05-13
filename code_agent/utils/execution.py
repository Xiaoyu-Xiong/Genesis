from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from code_agent.utils.local_execution import LocalRunConfig, build_local_execution_env, run_local


_GENESIS_EXECUTION_THREAD_LOCK = threading.Lock()


@dataclass(slots=True)
class ExecutionReport:
    command: list[str]
    returncode: int
    duration_sec: float
    stdout_path: str
    stderr_path: str
    artifacts: dict[str, str]
    diagnostics: dict[str, object] | None = None
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
            "diagnostics": self.diagnostics or {},
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
        diagnostic_report = _maybe_render_initial_without_ipc(
            raw_report=raw_report,
            run_dir=run_dir,
            backend=backend,
            sim_dt=_sim_dt_from_timing(run_dir),
            sim_substeps=_sim_substeps_from_timing(run_dir),
            render_fps=render_fps,
        )
        if diagnostic_report is not None:
            _attach_initial_render_diagnostic(raw_report, diagnostic_report)
    artifact_paths = raw_report.get("artifact_paths", [])
    artifacts = {Path(path).stem: path for path in artifact_paths if isinstance(path, str)}
    diagnostics = raw_report.get("diagnostics") if isinstance(raw_report.get("diagnostics"), dict) else {}
    initial = diagnostics.get("initial_no_ipc_render") if isinstance(diagnostics, dict) else None
    if isinstance(initial, dict) and isinstance(initial.get("image_path"), str):
        artifacts["initial_no_ipc_render"] = initial["image_path"]
    report = ExecutionReport(
        command=list(raw_report["command"]),
        returncode=int(raw_report["exit_code"]),
        duration_sec=float(raw_report["duration_sec"]),
        stdout_path=str(raw_report["stdout_path"]),
        stderr_path=str(raw_report["stderr_path"]),
        artifacts=artifacts,
        diagnostics=diagnostics,
        lock_wait_sec=float(lock_info["wait_sec"]),
        lock_path=str(lock_info["path"]),
    )
    return report


def _maybe_render_initial_without_ipc(
    *,
    raw_report: dict[str, object],
    run_dir: Path,
    backend: str,
    sim_dt: float,
    sim_substeps: int,
    render_fps: int,
) -> dict[str, object] | None:
    if int(raw_report.get("exit_code") or 0) == 0:
        return None
    output_text = _execution_output_text(raw_report, run_dir)
    if not _looks_like_ipc_build_failure(output_text):
        return None

    reports_dir = run_dir / "reports"
    diagnostic_stdout = reports_dir / "initial_no_ipc_render.stdout.txt"
    diagnostic_stderr = reports_dir / "initial_no_ipc_render.stderr.txt"
    out_dir = run_dir / "artifacts" / "initial_no_ipc_render"
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "code_agent.utils.initial_render",
        "--run-dir",
        str(run_dir.resolve()),
        "--backend",
        backend,
        "--out-dir",
        str(out_dir),
        "--sim-dt",
        str(float(sim_dt)),
        "--sim-substeps",
        str(int(sim_substeps)),
        "--fps",
        str(int(render_fps)),
    ]
    deformable_config_path = run_dir / "contracts" / "deformable_config.json"
    if deformable_config_path.exists():
        command.extend(("--deformable-config", str(deformable_config_path.resolve())))

    started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=build_local_execution_env({"GENESIS_BACKEND": backend}),
            capture_output=True,
            text=True,
            timeout=240.0,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_timeout_stream(exc.stdout)
        stderr = _decode_timeout_stream(exc.stderr)
        stderr = (stderr + "\n" if stderr else "") + "Timed out after 240.000 seconds."
        returncode = 124

    diagnostic_stdout.write_text(stdout, encoding="utf-8")
    diagnostic_stderr.write_text(stderr, encoding="utf-8")
    report_path = out_dir / "initial_no_ipc_render_report.json"
    report = _read_json(report_path) or {
        "ok": False,
        "diagnostic": "initial_no_ipc_render",
        "out_dir": str(out_dir),
    }
    report.update(
        {
            "trigger": "ipc_build_failure",
            "command": command,
            "returncode": returncode,
            "duration_sec_total": time.time() - started,
            "stdout_path": str(diagnostic_stdout),
            "stderr_path": str(diagnostic_stderr),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _attach_initial_render_diagnostic(raw_report: dict[str, object], diagnostic_report: dict[str, object]) -> None:
    diagnostics = raw_report.setdefault("diagnostics", {})
    if isinstance(diagnostics, dict):
        diagnostics["initial_no_ipc_render"] = diagnostic_report

    artifact_paths = raw_report.setdefault("artifact_paths", [])
    if not isinstance(artifact_paths, list):
        artifact_paths = []
        raw_report["artifact_paths"] = artifact_paths
    for key in ("image_path", "video_path", "render_stats_path", "report_path"):
        path_value = diagnostic_report.get(key)
        if isinstance(path_value, str) and Path(path_value).is_file() and path_value not in artifact_paths:
            artifact_paths.append(path_value)

    artifacts = raw_report.setdefault("artifacts", {})
    if isinstance(artifacts, dict):
        image_path = diagnostic_report.get("image_path")
        if isinstance(image_path, str):
            artifacts["initial_no_ipc_render"] = image_path

    report_path = raw_report.get("execution_report_path")
    if isinstance(report_path, str):
        Path(report_path).write_text(json.dumps(raw_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _execution_output_text(raw_report: dict[str, object], run_dir: Path) -> str:
    chunks: list[str] = []
    for key in ("stdout_path", "stderr_path"):
        value = raw_report.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = run_dir / path
        try:
            if path.is_file():
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return "\n".join(chunks)


def _looks_like_ipc_build_failure(text: str) -> bool:
    lowered = text.lower()
    build_markers = (
        "simplicialsurfaceintersectioncheck",
        "simplicialsurfacedistancecheck",
        "world is not valid",
        "skipping init",
        "sanity_check",
        "initial penetration",
    )
    has_build_marker = any(marker in lowered for marker in build_markers)
    if not has_build_marker:
        return False
    if "invalid accelerations causing 'nan'" in lowered:
        return False
    ipc_markers = ("ipc", "uipc", "libuipc", "world is not valid")
    return any(marker in lowered for marker in ipc_markers)


def _sim_dt_from_timing(run_dir: Path) -> float:
    timing = _read_json(run_dir / "contracts" / "timing.json")
    if isinstance(timing, dict):
        try:
            return float(timing.get("sim_dt", 0.01))
        except (TypeError, ValueError):
            pass
    return 0.01


def _sim_substeps_from_timing(run_dir: Path) -> int:
    timing = _read_json(run_dir / "contracts" / "timing.json")
    if isinstance(timing, dict):
        try:
            return int(timing.get("sim_substeps", 1))
        except (TypeError, ValueError):
            pass
    return 1


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _decode_timeout_stream(stream: bytes | str | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


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
