from __future__ import annotations

import argparse
import contextlib
import json
import os
import runpy
import signal
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS
from code_agent.io_utils import decode_process_stream


DEFAULT_ARTIFACT_DIR_NAMES = ("artifacts", "outputs", "renders", "frames")
DEFAULT_ARTIFACT_FILE_NAMES = (
    "summary.json",
    "metrics.json",
    "events.json",
    "event_log.json",
    "run_result.json",
    "render_stats.json",
    "render.mp4",
    "video.mp4",
)
PROCESS_TREE_TERMINATION_GRACE_SEC = 5.0
STALE_ARTIFACT_SAMPLE_LIMIT = 20
ARTIFACT_MTIME_EPSILON_SEC = 1.0e-6


@dataclass(slots=True, frozen=True)
class LocalRunConfig:
    """Configuration for one generated-code run in the repository uv environment."""

    workspace_dir: Path
    main_file: str = "main.py"
    output_dir: Path | None = None
    timeout_sec: float = CONFIGS.harness.execution_timeout_sec
    python_executable: str = "python"
    extra_args: tuple[str, ...] = ()
    artifact_dir_names: tuple[str, ...] = DEFAULT_ARTIFACT_DIR_NAMES
    artifact_file_names: tuple[str, ...] = DEFAULT_ARTIFACT_FILE_NAMES
    extra_artifact_paths: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class _CommandRunResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    timeout_process_tree: dict[str, Any] | None = None


def run_local(config: LocalRunConfig) -> dict[str, Any]:
    """Run generated ``main.py`` directly and write an execution report."""

    workspace_dir = config.workspace_dir.resolve()
    output_dir = (config.output_dir or workspace_dir).resolve()
    main_path = (workspace_dir / config.main_file).resolve()
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    report_path = output_dir / "execution_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    command = _build_command(config)
    run_env = _build_env(config.env)

    if not main_path.is_file():
        duration_sec = time.time() - started_at
        message = f"Generated entry point not found: {main_path}"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(message + "\n", encoding="utf-8")
        report = _base_report(
            config, command, workspace_dir, main_path, output_dir, started_at, duration_sec, run_env=run_env
        )
        artifact_paths, stale_artifact_paths = _collect_artifact_paths(
            config, workspace_dir, output_dir, min_mtime=started_at
        )
        report.update(
            {
                "status": "failed",
                "exit_code": 127,
                "timed_out": False,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "execution_report_path": str(report_path),
                **_artifact_report_fields(artifact_paths, stale_artifact_paths),
            }
        )
        _write_json(report_path, report)
        return report

    completed = _run_command_with_process_group(
        command,
        cwd=workspace_dir,
        env=run_env,
        timeout_sec=config.timeout_sec,
    )

    duration_sec = time.time() - started_at
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    report = _base_report(
        config, command, workspace_dir, main_path, output_dir, started_at, duration_sec, run_env=run_env
    )
    artifact_paths, stale_artifact_paths = _collect_artifact_paths(
        config, workspace_dir, output_dir, min_mtime=started_at
    )
    report.update(
        {
            "status": _status(completed.exit_code, completed.timed_out),
            "exit_code": completed.exit_code,
            "timed_out": completed.timed_out,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "execution_report_path": str(report_path),
            **_artifact_report_fields(artifact_paths, stale_artifact_paths),
        }
    )
    if completed.timeout_process_tree is not None:
        report["timeout_process_tree"] = completed.timeout_process_tree
    _write_json(report_path, report)
    return report


def run_local_in_process(config: LocalRunConfig) -> dict[str, Any]:
    """Run generated ``main.py`` inside the current Python interpreter.

    This is used by the Opt runner's long-lived worker process to reuse the
    fixed CUDA/Genesis/libuipc context across multiple trials. Timeouts are
    enforced by the parent process that owns the worker.
    """

    workspace_dir = config.workspace_dir.resolve()
    output_dir = (config.output_dir or workspace_dir).resolve()
    main_path = (workspace_dir / config.main_file).resolve()
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    report_path = output_dir / "execution_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    command = _build_command(config)
    run_env = _build_env(config.env)

    if not main_path.is_file():
        duration_sec = time.time() - started_at
        message = f"Generated entry point not found: {main_path}"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(message + "\n", encoding="utf-8")
        report = _base_report(
            config, command, workspace_dir, main_path, output_dir, started_at, duration_sec, run_env=run_env
        )
        artifact_paths, stale_artifact_paths = _collect_artifact_paths(
            config, workspace_dir, output_dir, min_mtime=started_at
        )
        report.update(
            {
                "runner": "local_in_process",
                "status": "failed",
                "exit_code": 127,
                "timed_out": False,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "execution_report_path": str(report_path),
                **_artifact_report_fields(artifact_paths, stale_artifact_paths),
            }
        )
        _write_json(report_path, report)
        return report

    stdout_text = ""
    stderr_text = ""
    exit_code = 0
    with _temporary_process_context(
        cwd=workspace_dir,
        argv=[str(main_path), *config.extra_args],
        env=run_env,
        import_paths=(main_path.parent, workspace_dir),
    ):
        _evict_workspace_modules(workspace_dir)
        stdout_buffer = _TextBuffer()
        stderr_buffer = _TextBuffer()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                runpy.run_path(str(main_path), run_name="__main__")
            except SystemExit as exc:
                exit_code = _system_exit_code(exc)
            except BaseException:
                exit_code = 1
                traceback.print_exc()
        stdout_text = stdout_buffer.text
        stderr_text = stderr_buffer.text

    duration_sec = time.time() - started_at
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")

    report = _base_report(
        config, command, workspace_dir, main_path, output_dir, started_at, duration_sec, run_env=run_env
    )
    artifact_paths, stale_artifact_paths = _collect_artifact_paths(
        config, workspace_dir, output_dir, min_mtime=started_at
    )
    report.update(
        {
            "runner": "local_in_process",
            "status": _status(exit_code, False),
            "exit_code": exit_code,
            "timed_out": False,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "execution_report_path": str(report_path),
            **_artifact_report_fields(artifact_paths, stale_artifact_paths),
        }
    )
    _write_json(report_path, report)
    return report


def _build_command(config: LocalRunConfig) -> list[str]:
    python_command = shlex.split(config.python_executable)
    return [
        *python_command,
        config.main_file,
        *config.extra_args,
    ]


def _build_env(overrides: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    cuda_home = repo_root / ".venv" / "cuda-12.8"
    env["LD_LIBRARY_PATH"] = _prepend_existing_paths(
        env.get("LD_LIBRARY_PATH", ""),
        (str(cuda_home / "lib"), "/usr/lib/wsl/lib"),
    )
    env["PATH"] = _prepend_existing_paths(
        env.get("PATH", ""),
        (
            str(repo_root / ".venv" / "bin"),
            str(Path.home() / ".local" / "bin"),
            str(cuda_home / "bin"),
            "/usr/lib/wsl/lib",
        ),
    )
    if cuda_home.is_dir():
        env.setdefault("CUDA_HOME", str(cuda_home))
    cache_root = Path(os.environ.get("CODE_AGENT_CACHE_ROOT", "/tmp/code-agent-cache"))
    _setdefault_cache_dir(env, "UV_CACHE_DIR", cache_root / "uv")
    _setdefault_cache_dir(env, "NUMBA_CACHE_DIR", cache_root / "numba")
    _setdefault_cache_dir(env, "MPLCONFIGDIR", cache_root / "matplotlib")
    _setdefault_cache_dir(env, "XDG_CACHE_HOME", cache_root / "xdg")
    _setdefault_cache_dir(env, "QD_OFFLINE_CACHE_FILE_PATH", cache_root / "quadrants")
    env.update(
        {
            "GENESIS_BACKEND": CONFIGS.harness.default_backend,
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(overrides)
    return env


def _run_command_with_process_group(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_sec: float,
) -> _CommandRunResult:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    process_group_id = process.pid
    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired as timeout_exc:
        kill_report: dict[str, Any] = {
            "process_group": True,
            "pid": process.pid,
            "pgid": process_group_id,
            "grace_sec": PROCESS_TREE_TERMINATION_GRACE_SEC,
            "signals": [],
        }
        _send_process_group_signal(process, process_group_id, signal.SIGTERM, kill_report)
        try:
            stdout, stderr = process.communicate(timeout=PROCESS_TREE_TERMINATION_GRACE_SEC)
            kill_report["terminated_after_sigterm"] = True
        except subprocess.TimeoutExpired as sigterm_exc:
            kill_report["terminated_after_sigterm"] = False
            kill_report["escalated_to_sigkill"] = True
            _send_process_group_signal(process, process_group_id, signal.SIGKILL, kill_report)
            try:
                stdout, stderr = process.communicate(timeout=PROCESS_TREE_TERMINATION_GRACE_SEC)
                kill_report["terminated_after_sigkill"] = True
            except subprocess.TimeoutExpired as sigkill_exc:
                kill_report["terminated_after_sigkill"] = False
                stdout = _latest_timeout_stream(sigkill_exc.stdout, sigterm_exc.stdout, timeout_exc.stdout)
                stderr = _latest_timeout_stream(sigkill_exc.stderr, sigterm_exc.stderr, timeout_exc.stderr)
        kill_report["returncode_after_kill"] = process.returncode
        stderr = _append_stderr_line(stderr, f"Timed out after {timeout_sec:.3f} seconds.")
        return _CommandRunResult(
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            timeout_process_tree=kill_report,
        )
    return _CommandRunResult(
        exit_code=int(process.returncode or 0),
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
    )


def _send_process_group_signal(
    process: subprocess.Popen[str],
    process_group_id: int,
    sig: signal.Signals,
    report: dict[str, Any],
) -> None:
    signal_entry: dict[str, Any] = {
        "signal": sig.name,
        "sent": False,
        "target": "process_group",
        "pgid": process_group_id,
    }
    if hasattr(os, "killpg"):
        try:
            os.killpg(process_group_id, sig)
            signal_entry["sent"] = True
        except ProcessLookupError:
            signal_entry["error"] = "process_group_not_found"
        except OSError as exc:
            signal_entry["error"] = f"{type(exc).__name__}: {exc}"
    else:
        signal_entry["target"] = "process"
        try:
            process.send_signal(sig)
            signal_entry["sent"] = True
        except ProcessLookupError:
            signal_entry["error"] = "process_not_found"
        except OSError as exc:
            signal_entry["error"] = f"{type(exc).__name__}: {exc}"
    report["signals"].append(signal_entry)


def _latest_timeout_stream(*streams: bytes | str | None) -> str:
    for stream in streams:
        if stream:
            return decode_process_stream(stream)
    return ""


def _append_stderr_line(stderr: str, line: str) -> str:
    return (stderr + "\n" if stderr else "") + line


class _TextBuffer:
    def __init__(self) -> None:
        self._parts: list[str] = []

    def write(self, text: str) -> int:
        self._parts.append(str(text))
        return len(text)

    def flush(self) -> None:
        return None

    @property
    def text(self) -> str:
        return "".join(self._parts)


@contextlib.contextmanager
def _temporary_process_context(
    *,
    cwd: Path,
    argv: list[str],
    env: dict[str, str],
    import_paths: tuple[Path, ...],
):
    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    old_env = os.environ.copy()
    old_path = sys.path[:]
    os.chdir(cwd)
    sys.argv = argv
    for path in reversed(tuple(str(path) for path in import_paths)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        sys.argv = old_argv
        sys.path[:] = old_path
        os.chdir(old_cwd)


def _system_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _evict_workspace_modules(workspace_dir: Path) -> None:
    workspace_dir = workspace_dir.resolve()
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            path = Path(module_file).resolve()
        except OSError:
            continue
        if _is_relative_to(path, workspace_dir):
            sys.modules.pop(name, None)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def build_local_execution_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Return the environment used for generated local Genesis runs."""

    return _build_env({} if overrides is None else dict(overrides))


def _prepend_existing_paths(current: str, candidates: tuple[str, ...]) -> str:
    existing = [path for path in candidates if Path(path).exists()]
    parts = [path for path in current.split(os.pathsep) if path]
    return os.pathsep.join([*existing, *[path for path in parts if path not in existing]])


def _setdefault_cache_dir(env: dict[str, str], key: str, path: Path) -> None:
    env.setdefault(key, str(path))
    with contextlib.suppress(OSError):
        Path(env[key]).mkdir(parents=True, exist_ok=True)


def _base_report(
    config: LocalRunConfig,
    command: list[str],
    workspace_dir: Path,
    main_path: Path,
    output_dir: Path,
    started_at: float,
    duration_sec: float,
    *,
    run_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    report = {
        "runner": "local",
        "schema_version": 1,
        "workspace_dir": str(workspace_dir),
        "main_path": str(main_path),
        "output_dir": str(output_dir),
        "command": command,
        "backend": config.env.get("GENESIS_BACKEND", CONFIGS.harness.default_backend),
        "timeout_sec": config.timeout_sec,
        "started_at_unix": started_at,
        "duration_sec": duration_sec,
    }
    if run_env is not None:
        report["environment"] = _environment_report(run_env)
    return report


def _environment_report(env: dict[str, str]) -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi", path=env.get("PATH"))
    report: dict[str, Any] = {
        "GENESIS_BACKEND": env.get("GENESIS_BACKEND"),
        "CUDA_HOME": env.get("CUDA_HOME"),
        "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH", ""),
        "PATH": env.get("PATH", ""),
        "PATH_head": _path_head(env.get("PATH", "")),
        "python_path": shutil.which("python", path=env.get("PATH")),
        "uv_path": shutil.which("uv", path=env.get("PATH")),
        "nvidia_smi_path": nvidia_smi,
    }
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["nvidia_smi"] = {"available": False, "error": str(exc)}
        else:
            report["nvidia_smi"] = {
                "available": completed.returncode == 0,
                "exit_code": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
    return report


def _path_head(value: str, *, count: int = 8) -> list[str]:
    return [part for part in value.split(os.pathsep) if part][:count]


def _collect_artifact_paths(
    config: LocalRunConfig,
    workspace_dir: Path,
    output_dir: Path,
    *,
    min_mtime: float | None = None,
) -> tuple[list[str], list[str]]:
    candidates: set[Path] = set()

    for name in config.artifact_file_names:
        for root in (workspace_dir, output_dir):
            path = (root / name).resolve()
            if path.is_file():
                candidates.add(path)

    for name in config.artifact_dir_names:
        for root in (workspace_dir, output_dir):
            artifact_dir = (root / name).resolve()
            if artifact_dir.is_dir():
                candidates.update(path.resolve() for path in artifact_dir.rglob("*") if path.is_file())

    for path_text in config.extra_artifact_paths:
        path = Path(path_text)
        if not path.is_absolute():
            path = workspace_dir / path
        path = path.resolve()
        if path.is_file():
            candidates.add(path)
        elif path.is_dir():
            candidates.update(child.resolve() for child in path.rglob("*") if child.is_file())

    excluded = {
        (output_dir / "stdout.txt").resolve(),
        (output_dir / "stderr.txt").resolve(),
        (output_dir / "execution_report.json").resolve(),
    }
    fresh_paths: list[str] = []
    stale_paths: list[str] = []
    for path in sorted(candidates - excluded):
        if min_mtime is not None and _is_stale_artifact(path, min_mtime):
            stale_paths.append(str(path))
        else:
            fresh_paths.append(str(path))
    return fresh_paths, stale_paths


def _is_stale_artifact(path: Path, min_mtime: float) -> bool:
    try:
        return path.stat().st_mtime < min_mtime - ARTIFACT_MTIME_EPSILON_SEC
    except OSError:
        return False


def _artifact_report_fields(artifact_paths: list[str], stale_artifact_paths: list[str]) -> dict[str, Any]:
    return {
        "artifact_paths": artifact_paths,
        "artifacts": _artifact_map(artifact_paths),
        "stale_artifact_count": len(stale_artifact_paths),
        "stale_artifact_paths_sample": stale_artifact_paths[:STALE_ARTIFACT_SAMPLE_LIMIT],
    }


def _artifact_map(paths: list[str]) -> dict[str, str | None]:
    by_name = {Path(path).name: path for path in paths}
    frames_dir = next((str(Path(path).parent) for path in paths if Path(path).parent.name == "frames"), None)
    return {
        "run_result": by_name.get("run_result.json"),
        "event_log": by_name.get("event_log.json") or by_name.get("events.json"),
        "metrics": by_name.get("metrics.json"),
        "video": by_name.get("render.mp4") or by_name.get("video.mp4"),
        "frames_dir": frames_dir,
    }


def _status(exit_code: int, timed_out: bool) -> str:
    if timed_out:
        return "timed_out"
    return "passed" if exit_code == 0 else "failed"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generated Genesis main.py directly.")
    parser.add_argument("workspace_dir", type=Path)
    parser.add_argument("--main-file", default="main.py")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-sec", type=float, default=CONFIGS.harness.execution_timeout_sec)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default=CONFIGS.harness.default_backend)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    extra_args = tuple(arg for arg in args.extra_args if arg != "--")
    report = run_local(
        LocalRunConfig(
            workspace_dir=args.workspace_dir,
            main_file=args.main_file,
            output_dir=args.output_dir,
            timeout_sec=args.timeout_sec,
            python_executable=args.python_executable,
            extra_args=extra_args,
            extra_artifact_paths=tuple(args.artifact),
            env={"GENESIS_BACKEND": args.backend},
        )
    )
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
