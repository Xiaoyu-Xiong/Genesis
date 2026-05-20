from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code_agent.io_utils import load_json_object
from code_agent.opt.contracts import OptContracts, write_params_payload
from code_agent.opt.objective import ObjectiveScore, evaluate_objective
from code_agent.utils.execution import _exclusive_genesis_execution_lock
from code_agent.utils.local_execution import LocalRunConfig, run_local


@dataclass(slots=True)
class RunOptOptions:
    backend: str
    max_trials: int
    population_size: int
    seed: int | None
    timeout_sec: float
    steps: int | None
    duration_sec: float | None
    render_fps: int | None
    target_video_frames: int | None
    render_best: bool
    baseline_trials: int
    trial_root: Path
    best_out_dir: Path
    current_params_path: Path


@dataclass(slots=True)
class TrialResult:
    entry: dict[str, Any]
    score: ObjectiveScore
    params_payload: dict[str, Any]


class TrialExecutor:
    def __init__(self, *, case_dir: Path, contracts_dir: Path, reports_dir: Path, main_file: str) -> None:
        self.case_dir = case_dir
        self.contracts_dir = contracts_dir
        self.reports_dir = reports_dir
        self.main_file = main_file

    def run_trial(
        self,
        *,
        trial_index: int,
        params_payload: dict[str, Any],
        options: RunOptOptions,
        contracts: OptContracts,
    ) -> TrialResult:
        trial_name = f"trial_{trial_index:03d}"
        artifacts_dir = options.trial_root / trial_name
        report_dir = self.reports_dir / "opt_trials" / trial_name
        self._clean_dir(artifacts_dir)
        self._clean_dir(report_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        params_path = artifacts_dir / "opt_params.json"
        write_params_payload(params_path, params_payload)
        current_payload = payload_for_trial(
            params_payload,
            source="current",
            trial_index=trial_index,
            metadata={"trial_params_path": self.rel(params_path)},
        )
        write_params_payload(options.current_params_path, current_payload)
        raw_report = self._run_main(
            artifacts_dir=artifacts_dir,
            report_dir=report_dir,
            options=options,
            render=False,
        )
        metrics_path = self._metrics_path(raw_report, artifacts_dir)
        metrics = load_json_object(metrics_path) if metrics_path is not None else None
        execution_failed = int(raw_report.get("exit_code", 1)) != 0 or metrics is None
        failure_reason = None
        if int(raw_report.get("exit_code", 1)) != 0:
            failure_reason = f"exit_code={raw_report.get('exit_code')}"
        elif metrics is None:
            failure_reason = "missing_metrics"
        score = evaluate_objective(
            target_spec=contracts.target_spec,
            metrics=metrics,
            execution_failed=execution_failed,
            failure_reason=failure_reason,
        )
        status = _trace_status(raw_report, metrics)
        entry = {
            "schema_version": 1,
            "trial_index": trial_index,
            "status": status,
            "params_path": self.rel(params_path),
            "artifacts_dir": self.rel(artifacts_dir),
            "metrics_path": None if metrics_path is None else self.rel(metrics_path),
            "execution_report_path": self.rel(Path(str(raw_report.get("execution_report_path")))),
            "score": score.score,
            "duration_sec": float(raw_report.get("duration_sec", 0.0)),
            "exit_code": int(raw_report.get("exit_code", 1)),
            "objective": score.to_report(),
            "errors": [] if status == "completed" else [score.failure_reason or status],
            "warnings": score.warnings,
        }
        return TrialResult(entry=entry, score=score, params_payload=params_payload)

    def run_best_render(self, params_payload: dict[str, Any], options: RunOptOptions) -> None:
        self._clean_dir(options.best_out_dir)
        best_report_dir = self.reports_dir / "opt_best"
        self._clean_dir(best_report_dir)
        options.best_out_dir.mkdir(parents=True, exist_ok=True)
        write_params_payload(options.best_out_dir / "opt_params.json", params_payload)
        write_params_payload(options.current_params_path, params_payload)
        self._run_main(
            artifacts_dir=options.best_out_dir,
            report_dir=best_report_dir,
            options=options,
            render=True,
        )

    def rel(self, path: Path) -> str:
        path = path.resolve()
        try:
            return str(path.relative_to(self.case_dir))
        except ValueError:
            return str(path)

    def _run_main(
        self,
        *,
        artifacts_dir: Path,
        report_dir: Path,
        options: RunOptOptions,
        render: bool,
    ) -> dict[str, Any]:
        extra_args = self._main_args(artifacts_dir=artifacts_dir, options=options, render=render)
        with _exclusive_genesis_execution_lock() as lock_info:
            raw_report = run_local(
                LocalRunConfig(
                    workspace_dir=self.case_dir,
                    main_file=self.main_file,
                    output_dir=report_dir,
                    timeout_sec=options.timeout_sec,
                    python_executable="uv run --no-sync python",
                    extra_args=tuple(extra_args),
                    artifact_dir_names=(),
                    artifact_file_names=(),
                    extra_artifact_paths=(self.rel(artifacts_dir),),
                    env={"GENESIS_BACKEND": options.backend},
                )
            )
        raw_report["lock_wait_sec"] = float(lock_info["wait_sec"])
        raw_report["lock_path"] = str(lock_info["path"])
        return raw_report

    def _main_args(self, *, artifacts_dir: Path, options: RunOptOptions, render: bool) -> list[str]:
        args = ["--backend", options.backend, "--out-dir", self.rel(artifacts_dir)]
        if options.steps is not None:
            args.extend(("--steps", str(int(options.steps))))
        if options.render_fps is not None:
            args.extend(("--fps", str(int(options.render_fps))))
        if options.duration_sec is not None:
            args.extend(("--duration-sec", str(float(options.duration_sec))))
        if render and options.target_video_frames is not None:
            args.extend(("--target-video-frames", str(int(options.target_video_frames))))
        deformable_config_path = self.contracts_dir / "deformable_config.json"
        if deformable_config_path.is_file():
            args.extend(("--deformable-config", self.rel(deformable_config_path)))
        args.append("--render" if render else "--no-render")
        return args

    def _metrics_path(self, raw_report: dict[str, Any], artifacts_dir: Path) -> Path | None:
        direct = artifacts_dir / "metrics.json"
        if direct.is_file():
            return direct
        artifacts = raw_report.get("artifacts")
        if isinstance(artifacts, dict):
            metrics = artifacts.get("metrics")
            if isinstance(metrics, str) and Path(metrics).is_file():
                return Path(metrics)
        return None

    def _clean_dir(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)


def payload_for_trial(
    payload: dict[str, Any],
    *,
    source: str,
    trial_index: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    copied = copy.deepcopy(payload)
    copied["schema_version"] = 1
    copied["source"] = source
    copied["trial_index"] = trial_index
    copied["metadata"] = {**copied.get("metadata", {}), **metadata}
    return copied


def _trace_status(raw_report: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    if bool(raw_report.get("timed_out")):
        return "timed_out"
    if int(raw_report.get("exit_code", 1)) != 0:
        return "failed"
    if metrics is None:
        return "failed"
    return "completed"
