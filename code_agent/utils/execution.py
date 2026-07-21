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

from code_agent.configs import CONFIGS
from code_agent.io_utils import decode_process_stream, load_json_object
from code_agent.utils.local_execution import LocalRunConfig, build_local_execution_env, run_local


_GENESIS_EXECUTION_THREAD_LOCK = threading.Lock()
GENESIS_EXECUTION_LOCK_PATH_ENV = "GENESIS_EXECUTION_LOCK_PATH"


@dataclass(slots=True)
class ExecutionReport:
    command: list[str]
    returncode: int
    duration_sec: float
    stdout_path: str
    stderr_path: str
    artifacts: dict[str, str]
    diagnostics: dict[str, object] | None = None
    failure_class: str | None = None
    failure_reason: str | None = None
    rework_required: bool = False
    progress_watchdog: dict[str, object] | None = None
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
            "failure_class": self.failure_class,
            "failure_reason": self.failure_reason,
            "rework_required": self.rework_required,
            "progress_watchdog": self.progress_watchdog or {},
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
    sim_dt: float,
    sim_substeps: int,
    render_every_n_steps: int,
    render_res: tuple[int, int],
    render: bool = True,
    duration_sec: float | None = None,
    target_video_frames: int | None = None,
    render_profile: str = "debug_raster",
    save_state_cache: bool = True,
    require_state_cache: bool = True,
    replay_cache: Path | None = None,
    render_only: bool = False,
    execution_lock_path: Path | None = None,
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
        "--sim-dt",
        str(float(sim_dt)),
        "--sim-substeps",
        str(int(sim_substeps)),
        "--render-every-n-steps",
        str(int(render_every_n_steps)),
        "--render-res",
        str(int(render_res[0])),
        str(int(render_res[1])),
    ]
    if duration_sec is not None:
        extra_args.extend(("--duration-sec", str(float(duration_sec))))
    if target_video_frames is not None:
        extra_args.extend(("--target-video-frames", str(int(target_video_frames))))
    extra_args.extend(("--render-profile", str(render_profile)))
    effective_save_state_cache = bool(save_state_cache) and not render_only
    effective_require_state_cache = bool(require_state_cache) and not render_only
    if effective_save_state_cache:
        extra_args.append("--save-state-cache")
    if effective_require_state_cache:
        extra_args.append("--require-state-cache")
    if replay_cache is not None:
        extra_args.extend(("--replay-cache", str(replay_cache)))
    if render_only:
        extra_args.append("--render-only")
    deformable_config_path = run_dir / "contracts" / "deformable_config.json"
    if deformable_config_path.exists():
        extra_args.extend(("--deformable-config", str(deformable_config_path.relative_to(run_dir))))
    extra_args.append(render_arg)
    with _exclusive_genesis_execution_lock(execution_lock_path) as lock_info:
        env_overrides = _execution_env_overrides(
            backend=backend, render_profile=render_profile, replay_cache=replay_cache
        )
        raw_report = run_local(
            LocalRunConfig(
                workspace_dir=run_dir,
                main_file=main_file,
                output_dir=reports_dir,
                timeout_sec=timeout_sec,
                python_executable="uv run --no-sync python",
                extra_args=tuple(extra_args),
                extra_artifact_paths=("artifacts",),
                env=env_overrides,
                progress_frames_dir=run_dir / "artifacts" / "frames" if render and target_video_frames else None,
                progress_target_frames=target_video_frames if render else None,
                progress_checkpoints=(
                    CONFIGS.harness.frame_progress_checkpoints
                    if render and target_video_frames and CONFIGS.harness.frame_progress_watchdog_enabled
                    else ()
                ),
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
    return ExecutionReport(
        command=list(raw_report["command"]),
        returncode=int(raw_report["exit_code"]),
        duration_sec=float(raw_report["duration_sec"]),
        stdout_path=str(raw_report["stdout_path"]),
        stderr_path=str(raw_report["stderr_path"]),
        artifacts=artifacts,
        diagnostics=diagnostics,
        failure_class=_optional_report_str(raw_report.get("failure_class")),
        failure_reason=_optional_report_str(raw_report.get("failure_reason")),
        rework_required=bool(raw_report.get("rework_required", False)),
        progress_watchdog=(
            raw_report.get("progress_watchdog") if isinstance(raw_report.get("progress_watchdog"), dict) else None
        ),
        lock_wait_sec=float(lock_info["wait_sec"]),
        lock_path=str(lock_info["path"]),
    )


def _optional_report_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _execution_env_overrides(
    *,
    backend: str,
    render_profile: str,
    replay_cache: Path | None,
) -> dict[str, str]:
    env: dict[str, str] = {
        "GENESIS_BACKEND": backend,
        "GENESIS_RENDER_PROFILE": render_profile,
    }
    if render_profile == "final_path_traced" or replay_cache is not None:
        env.update(_path_tracing_env_overrides())
    return env


def _path_tracing_env_overrides() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    cuda_home = repo_root / ".venv" / "cuda-12.8"
    ld_candidates = (
        "/opt/nvidia-optix-595/lib",
        str(repo_root / "genesis" / "ext" / "LuisaRender" / "build" / "bin"),
        str(cuda_home / "lib"),
        "/usr/lib/wsl/lib",
    )
    path_candidates = (
        str(cuda_home / "bin"),
        str(repo_root / ".venv" / "bin"),
        str(Path.home() / ".local" / "bin"),
    )
    return {
        "LD_LIBRARY_PATH": _join_existing_paths(ld_candidates, os.environ.get("LD_LIBRARY_PATH", "")),
        "PATH": _join_existing_paths(path_candidates, os.environ.get("PATH", "")),
        "GENESIS_PATH_TRACING_OPTIX_DIR": "/opt/nvidia-optix-595/lib",
    }


def _join_existing_paths(candidates: tuple[str, ...], current: str) -> str:
    existing = [path for path in candidates if Path(path).exists()]
    current_parts = [path for path in current.split(os.pathsep) if path]
    return os.pathsep.join([*existing, *[path for path in current_parts if path not in existing]])


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
        "--no-sync",
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
        stdout = decode_process_stream(exc.stdout)
        stderr = decode_process_stream(exc.stderr)
        stderr = (stderr + "\n" if stderr else "") + "Timed out after 240.000 seconds."
        returncode = 124

    diagnostic_stdout.write_text(stdout, encoding="utf-8")
    diagnostic_stderr.write_text(stderr, encoding="utf-8")
    report_path = out_dir / "initial_no_ipc_render_report.json"
    report = load_json_object(report_path) or {
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
    timing = load_json_object(run_dir / "contracts" / "timing.json")
    if isinstance(timing, dict):
        try:
            return float(timing.get("sim_dt", 0.01))
        except (TypeError, ValueError):
            pass
    return 0.01


def _sim_substeps_from_timing(run_dir: Path) -> int:
    timing = load_json_object(run_dir / "contracts" / "timing.json")
    if isinstance(timing, dict):
        try:
            return int(timing.get("sim_substeps", 1))
        except (TypeError, ValueError):
            pass
    return 1


@contextlib.contextmanager
def _exclusive_genesis_execution_lock(lock_path: Path | None = None):
    """Serialize local Genesis simulation subprocesses across parallel suite cases."""

    lock_path = _resolve_genesis_execution_lock_path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with _GENESIS_EXECUTION_THREAD_LOCK, lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        wait_sec = time.time() - started
        try:
            yield {"path": lock_path, "wait_sec": wait_sec}
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _resolve_genesis_execution_lock_path(lock_path: Path | None = None) -> Path:
    if lock_path is not None:
        return lock_path.expanduser().resolve()
    env_path = os.environ.get(GENESIS_EXECUTION_LOCK_PATH_ENV)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(tempfile.gettempdir()) / f"genesis_code_agent_{os.getuid()}_execution.lock"
