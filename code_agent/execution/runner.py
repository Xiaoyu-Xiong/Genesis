from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .apptainer_cpu import ApptainerCpuRunConfig, run_apptainer_cpu


@dataclass(slots=True)
class ExecutionReport:
    command: list[str]
    returncode: int
    duration_sec: float
    stdout_path: str
    stderr_path: str
    artifacts: dict[str, str]

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
            "ok": self.ok,
        }


def run_generated_simulation(
    *,
    main_py: Path,
    run_dir: Path,
    backend: str,
    timeout_sec: float,
    render: bool = True,
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
    raw_report = run_apptainer_cpu(
        ApptainerCpuRunConfig(
            workspace_dir=run_dir,
            main_file=main_file,
            output_dir=reports_dir,
            timeout_sec=timeout_sec,
            python_executable="uv run python",
            extra_args=("--backend", backend, "--out-dir", "artifacts", "--steps", "40", render_arg),
            extra_artifact_paths=("artifacts",),
            env={"GENESIS_BACKEND": backend, "APPTAINERENV_GENESIS_BACKEND": backend},
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
    )
    (reports_dir / "legacy_execution_report.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return report
