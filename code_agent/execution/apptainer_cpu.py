from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_APPTAINER_IMAGE = "/ocean/projects/cis250078p/xxiong1/containers/genesis.sif"
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


@dataclass(slots=True, frozen=True)
class ApptainerCpuRunConfig:
    """Configuration for one generated-code CPU smoke run inside Apptainer."""

    workspace_dir: Path
    main_file: str = "main.py"
    output_dir: Path | None = None
    apptainer_image: str = DEFAULT_APPTAINER_IMAGE
    timeout_sec: float = 1000.0
    python_executable: str = "python"
    extra_args: tuple[str, ...] = ()
    artifact_dir_names: tuple[str, ...] = DEFAULT_ARTIFACT_DIR_NAMES
    artifact_file_names: tuple[str, ...] = DEFAULT_ARTIFACT_FILE_NAMES
    extra_artifact_paths: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


def run_apptainer_cpu(config: ApptainerCpuRunConfig) -> dict[str, Any]:
    """Run generated ``main.py`` in the standard Apptainer image and write an execution report.

    The runner itself must be invoked from an allowed context. It never runs generated Python on the host; the
    generated script is executed only through ``apptainer exec``.
    """

    workspace_dir = config.workspace_dir.resolve()
    output_dir = (config.output_dir or workspace_dir).resolve()
    main_path = (workspace_dir / config.main_file).resolve()
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    report_path = output_dir / "execution_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    command = _build_command(config, workspace_dir)
    run_env = _build_env(config.env)

    if not main_path.is_file():
        duration_sec = time.time() - started_at
        message = f"Generated entry point not found: {main_path}"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(message + "\n", encoding="utf-8")
        report = _base_report(config, command, workspace_dir, main_path, output_dir, started_at, duration_sec)
        artifact_paths = _collect_artifact_paths(config, workspace_dir, output_dir)
        report.update(
            {
                "status": "failed",
                "exit_code": 127,
                "timed_out": False,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "artifact_paths": artifact_paths,
                "artifacts": _artifact_map(artifact_paths),
                "execution_report_path": str(report_path),
            }
        )
        _write_json(report_path, report)
        return report

    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=str(workspace_dir),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
            check=False,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = _decode_timeout_stream(exc.stdout)
        stderr = _decode_timeout_stream(exc.stderr)
        stderr = (stderr + "\n" if stderr else "") + f"Timed out after {config.timeout_sec:.3f} seconds."

    duration_sec = time.time() - started_at
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    report = _base_report(config, command, workspace_dir, main_path, output_dir, started_at, duration_sec)
    artifact_paths = _collect_artifact_paths(config, workspace_dir, output_dir)
    report.update(
        {
            "status": _status(exit_code, timed_out),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "artifact_paths": artifact_paths,
            "artifacts": _artifact_map(artifact_paths),
            "execution_report_path": str(report_path),
        }
    )
    _write_json(report_path, report)
    return report


def _build_command(config: ApptainerCpuRunConfig, workspace_dir: Path) -> list[str]:
    python_command = shlex.split(config.python_executable)
    if _inside_apptainer():
        return [
            *python_command,
            config.main_file,
            *config.extra_args,
        ]
    return [
        "apptainer",
        "exec",
        "--cleanenv",
        "--pwd",
        str(workspace_dir),
        config.apptainer_image,
        *python_command,
        config.main_file,
        *config.extra_args,
    ]


def _build_env(overrides: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "APPTAINERENV_CUDA_VISIBLE_DEVICES": "",
            "APPTAINERENV_GENESIS_BACKEND": "cpu",
            "APPTAINERENV_PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "GENESIS_BACKEND": "cpu",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(overrides)
    return env


def _base_report(
    config: ApptainerCpuRunConfig,
    command: list[str],
    workspace_dir: Path,
    main_path: Path,
    output_dir: Path,
    started_at: float,
    duration_sec: float,
) -> dict[str, Any]:
    return {
        "runner": "apptainer_cpu",
        "schema_version": 1,
        "workspace_dir": str(workspace_dir),
        "main_path": str(main_path),
        "output_dir": str(output_dir),
        "apptainer_image": config.apptainer_image,
        "inside_apptainer": _inside_apptainer(),
        "command": command,
        "backend": config.env.get("GENESIS_BACKEND", "cpu"),
        "timeout_sec": config.timeout_sec,
        "started_at_unix": started_at,
        "duration_sec": duration_sec,
    }


def _collect_artifact_paths(config: ApptainerCpuRunConfig, workspace_dir: Path, output_dir: Path) -> list[str]:
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
    return [str(path) for path in sorted(candidates - excluded)]


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


def _inside_apptainer() -> bool:
    return bool(os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"))


def _decode_timeout_stream(stream: bytes | str | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generated Genesis main.py inside Apptainer on CPU.")
    parser.add_argument("workspace_dir", type=Path)
    parser.add_argument("--main-file", default="main.py")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image", default=DEFAULT_APPTAINER_IMAGE)
    parser.add_argument("--timeout-sec", type=float, default=1000.0)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    extra_args = tuple(arg for arg in args.extra_args if arg != "--")
    report = run_apptainer_cpu(
        ApptainerCpuRunConfig(
            workspace_dir=args.workspace_dir,
            main_file=args.main_file,
            output_dir=args.output_dir,
            apptainer_image=args.image,
            timeout_sec=args.timeout_sec,
            python_executable=args.python_executable,
            extra_args=extra_args,
            extra_artifact_paths=tuple(args.artifact),
        )
    )
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
