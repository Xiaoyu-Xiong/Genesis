from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from code_agent.io_utils import load_json_object
from code_agent.opt.contracts import OptContracts, get_dotted, write_params_payload
from code_agent.opt.objective import ObjectiveScore, evaluate_objective
from code_agent.opt.parallel_policy import OptParallelPolicy, TrialExecutionPlan, plan_trial_execution
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
    sim_dt: float | None
    sim_substeps: int | None
    render_every_n_steps: int | None
    render_res: tuple[int, int] | None
    target_video_frames: int | None
    render_best: bool
    baseline_trials: int
    best_repeat_trials: int
    trial_root: Path
    best_out_dir: Path
    current_params_path: Path
    parallel_policy: OptParallelPolicy = field(default_factory=OptParallelPolicy)


@dataclass(slots=True)
class TrialResult:
    entry: dict[str, Any]
    score: ObjectiveScore
    params_payload: dict[str, Any]


@dataclass(slots=True, frozen=True)
class TrialRequest:
    trial_index: int
    params_payload: dict[str, Any]
    variable_names: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class _PreparedTrial:
    trial_index: int
    trial_name: str
    artifacts_dir: Path
    report_dir: Path
    params_path: Path
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
        prepared = self._prepare_trial(
            trial_index=trial_index,
            params_payload=params_payload,
            options=options,
        )
        raw_report = self._run_main(
            artifacts_dir=prepared.artifacts_dir,
            report_dir=prepared.report_dir,
            options=options,
            render=False,
        )
        return self._result_from_report(prepared, raw_report, contracts)

    def run_trials(
        self,
        requests: list[TrialRequest],
        *,
        options: RunOptOptions,
        contracts: OptContracts,
    ) -> list[TrialResult]:
        """Run a generation-sized batch of trial requests.

        This compatibility path deliberately preserves the old serial
        subprocess behavior. Specialized backends can take over here without
        changing the optimizer's ask/tell loop.
        """

        plan = plan_trial_execution(
            policy=options.parallel_policy,
            contracts=contracts,
            contracts_dir=self.contracts_dir,
            case_dir=self.case_dir,
            request_variable_names=_request_variable_names(requests),
            request_count=len(requests),
        )
        return self._run_trials_with_plan(requests, options=options, contracts=contracts, plan=plan)

    def _run_trials_with_plan(
        self,
        requests: list[TrialRequest],
        *,
        options: RunOptOptions,
        contracts: OptContracts,
        plan: TrialExecutionPlan,
    ) -> list[TrialResult]:
        if len(requests) > plan.batch_size:
            results: list[TrialResult] = []
            for offset in range(0, len(requests), plan.batch_size):
                results.extend(
                    self._run_trials_with_plan(
                        requests[offset : offset + plan.batch_size],
                        options=options,
                        contracts=contracts,
                        plan=plan,
                    )
                )
            return results

        if plan.backend == "subprocess_parallel":
            from code_agent.opt.execution_backends import SubprocessParallelTrialBackend

            return SubprocessParallelTrialBackend(self, plan).run_trials(
                requests,
                options=options,
                contracts=contracts,
            )
        return self._run_trials_subprocess_serial(
            requests,
            options=options,
            contracts=contracts,
            execution_backend=plan.backend,
            execution_plan=plan,
        )

    def _run_trials_subprocess_serial(
        self,
        requests: list[TrialRequest],
        *,
        options: RunOptOptions,
        contracts: OptContracts,
        execution_backend: str = "subprocess_serial",
        execution_plan: TrialExecutionPlan | None = None,
    ) -> list[TrialResult]:
        results: list[TrialResult] = []
        for request in requests:
            result = self.run_trial(
                trial_index=request.trial_index,
                params_payload=request.params_payload,
                options=options,
                contracts=contracts,
            )
            result.entry["execution_backend"] = execution_backend
            if execution_plan is not None:
                result.entry["execution_plan"] = _plan_report(execution_plan)
            results.append(result)
        return results

    def _prepare_trial(
        self,
        *,
        trial_index: int,
        params_payload: dict[str, Any],
        options: RunOptOptions,
    ) -> _PreparedTrial:
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
        return _PreparedTrial(
            trial_index=trial_index,
            trial_name=trial_name,
            artifacts_dir=artifacts_dir,
            report_dir=report_dir,
            params_path=params_path,
            params_payload=params_payload,
        )

    def _result_from_report(
        self,
        prepared: _PreparedTrial,
        raw_report: dict[str, Any],
        contracts: OptContracts,
    ) -> TrialResult:
        metrics_path = self._metrics_path(raw_report, prepared.artifacts_dir)
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
        runtime_warnings = _runtime_opt_warnings(
            metrics=metrics,
            params_payload=prepared.params_payload,
            contracts=contracts,
        )
        plan_warnings = _execution_plan_warnings(raw_report.get("execution_plan"))
        status = _trace_status(raw_report, metrics)
        entry = {
            "schema_version": 1,
            "trial_index": prepared.trial_index,
            "status": status,
            "params_path": self.rel(prepared.params_path),
            "artifacts_dir": self.rel(prepared.artifacts_dir),
            "metrics_path": None if metrics_path is None else self.rel(metrics_path),
            "execution_report_path": self.rel(Path(str(raw_report.get("execution_report_path")))),
            "score": score.score,
            "duration_sec": float(raw_report.get("duration_sec", 0.0)),
            "exit_code": int(raw_report.get("exit_code", 1)),
            "execution_backend": str(raw_report.get("execution_backend") or "subprocess_serial"),
            "objective": score.to_report(),
            "errors": [] if status == "completed" else [score.failure_reason or status],
            "warnings": [*score.warnings, *runtime_warnings, *plan_warnings],
        }
        execution_plan = raw_report.get("execution_plan")
        if isinstance(execution_plan, dict):
            entry["execution_plan"] = execution_plan
        return TrialResult(entry=entry, score=score, params_payload=prepared.params_payload)

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
        if options.sim_dt is not None:
            args.extend(("--sim-dt", str(float(options.sim_dt))))
        if options.sim_substeps is not None:
            args.extend(("--sim-substeps", str(int(options.sim_substeps))))
        if options.render_every_n_steps is not None:
            args.extend(("--render-every-n-steps", str(int(options.render_every_n_steps))))
        if options.render_res is not None:
            args.extend(("--render-res", str(int(options.render_res[0])), str(int(options.render_res[1]))))
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


def _runtime_opt_warnings(
    *,
    metrics: dict[str, Any] | None,
    params_payload: dict[str, Any],
    contracts: OptContracts,
) -> list[str]:
    if not isinstance(metrics, dict):
        return []
    active_names = {variable.name for variable in contracts.active_variables}
    warnings: list[str] = []
    covered_names: set[str] = set()

    opt_params = metrics.get("opt_params") if isinstance(metrics.get("opt_params"), dict) else {}
    if isinstance(opt_params.get("loaded"), dict):
        loaded_params = opt_params["loaded"]
    elif isinstance(metrics.get("opt_params_loaded"), dict) and isinstance(metrics["opt_params_loaded"].get("params"), dict):
        loaded_params = metrics["opt_params_loaded"]["params"]
    elif isinstance(metrics.get("opt_param_overrides"), dict):
        loaded_params = metrics["opt_param_overrides"]
    elif isinstance(metrics.get("profile"), dict):
        loaded_params = metrics["profile"]
    else:
        loaded_params = opt_params
    loaded_mismatches = _loaded_param_mismatches(
        active_names=active_names,
        loaded_params=loaded_params,
        params_payload=params_payload,
        covered_names=covered_names,
    )
    if loaded_mismatches:
        warnings.append(
            "Generated metrics echoed opt params that differ from the requested trial params: "
            + ", ".join(loaded_mismatches[:8])
            + ("." if len(loaded_mismatches) <= 8 else f", ... ({len(loaded_mismatches)} total).")
        )

    action_diag = (
        opt_params.get("action_param_diagnostics")
        if isinstance(opt_params.get("action_param_diagnostics"), dict)
        else {}
    )
    ignored_keys = _string_items(action_diag.get("sign_sensitive_opt_keys_ignored"))
    ignored_active = sorted(f"action.{key}" for key in ignored_keys if f"action.{key}" in active_names)
    if ignored_active:
        warnings.append(
            "Generated action opt hook ignored active sign-sensitive variables: "
            + ", ".join(ignored_active)
            + ". Add the required schedule/version guard to default/current opt params or remove these variables."
        )

    control_params = metrics.get("control_params") if isinstance(metrics.get("control_params"), dict) else None
    if control_params is not None:
        changed = _changed_action_params(
            params_payload=params_payload,
            control_params=control_params,
            active_names=active_names,
            covered_names=covered_names,
        )
        if changed:
            warnings.append(
                "Generated action opt hook changed or clamped active variables before simulation: "
                + ", ".join(changed[:8])
                + ("." if len(changed) <= 8 else f", ... ({len(changed)} total).")
            )
    _cover_effective_params(
        opt_params=opt_params,
        params_payload=params_payload,
        active_names=active_names,
        covered_names=covered_names,
        warnings=warnings,
    )
    missing = sorted(active_names - covered_names)
    if missing:
        warnings.append(
            "Generated metrics do not echo requested or effective values for active opt variables: "
            + ", ".join(missing[:8])
            + (
                ". Record loaded/effective opt params in metrics so Opt can detect ignored or clamped variables."
                if len(missing) <= 8
                else f", ... ({len(missing)} total). Record loaded/effective opt params in metrics."
            )
        )
    return warnings


def _execution_plan_warnings(execution_plan: Any) -> list[str]:
    if not isinstance(execution_plan, dict):
        return []
    return []


def _changed_action_params(
    *,
    params_payload: dict[str, Any],
    control_params: dict[str, Any],
    active_names: set[str],
    covered_names: set[str],
) -> list[str]:
    params = params_payload.get("params", {})
    changed: list[str] = []
    for name in sorted(active_names):
        if not name.startswith("action."):
            continue
        key = name.split(".", maxsplit=1)[1]
        requested = get_dotted(params, name)
        effective = control_params.get(key)
        if isinstance(effective, int | float) and not isinstance(effective, bool):
            covered_names.add(name)
        if not _different_numbers(requested, effective):
            continue
        changed.append(f"{name} requested={float(requested):.6g} effective={float(effective):.6g}")
    return changed


def _loaded_param_mismatches(
    *,
    active_names: set[str],
    loaded_params: dict[str, Any],
    params_payload: dict[str, Any],
    covered_names: set[str],
) -> list[str]:
    params = params_payload.get("params", {})
    mismatches: list[str] = []
    for name in sorted(active_names):
        requested = get_dotted(params, name)
        loaded = _loaded_value(loaded_params, name)
        if loaded is None:
            continue
        covered_names.add(name)
        if _different_numbers(requested, loaded):
            mismatches.append(f"{name} requested={float(requested):.6g} loaded={float(loaded):.6g}")
    return mismatches


def _cover_effective_params(
    *,
    opt_params: dict[str, Any],
    params_payload: dict[str, Any],
    active_names: set[str],
    covered_names: set[str],
    warnings: list[str],
) -> None:
    params = params_payload.get("params", {})
    for container_name in ("body_params", "scene_params", "material_params", "effective_params"):
        container = opt_params.get(container_name)
        if not isinstance(container, dict):
            continue
        mismatches: list[str] = []
        for name in sorted(active_names - covered_names):
            key = name.split(".", maxsplit=1)[1] if "." in name else name
            effective = container.get(key)
            requested = get_dotted(params, name)
            if effective is None:
                continue
            covered_names.add(name)
            if _different_numbers(requested, effective):
                mismatches.append(f"{name} requested={float(requested):.6g} effective={float(effective):.6g}")
        if mismatches:
            warnings.append(
                f"Generated {container_name} changed or clamped active variables before simulation: "
                + ", ".join(mismatches[:8])
                + ("." if len(mismatches) <= 8 else f", ... ({len(mismatches)} total).")
            )


def _loaded_value(loaded_params: dict[str, Any], name: str) -> Any:
    value = get_dotted(loaded_params, name)
    if value is not None:
        return value
    if name in loaded_params:
        return loaded_params[name]
    if "." in name:
        _, key = name.split(".", maxsplit=1)
        if key in loaded_params:
            return loaded_params[key]
    return None


def _different_numbers(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, int | float) or not isinstance(right, int | float):
        return False
    return abs(float(left) - float(right)) > max(1e-9, 1e-6 * max(1.0, abs(float(left)), abs(float(right))))


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _request_variable_names(requests: list[TrialRequest]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for request in requests:
        for name in request.variable_names:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return tuple(names)


def _plan_report(plan: TrialExecutionPlan) -> dict[str, Any]:
    return {
        "backend": plan.backend,
        "workers": plan.workers,
        "batch_size": plan.batch_size,
        "reason": plan.reason,
        "variables": list(plan.variable_profile.variable_names),
        "variable_parallel_reasons": list(plan.variable_profile.reasons),
        "requires_scene_rebuild": plan.variable_profile.requires_scene_rebuild,
        "has_topology_changing": plan.variable_profile.has_topology_changing,
        "memory": {
            "usable_gpu_memory_gb": plan.memory_profile.usable_gpu_memory_gb,
            "reserve_gb": plan.memory_profile.reserve_gb,
            "subprocess_increment_gb": plan.memory_profile.subprocess_increment_gb,
            "subprocess_capacity": plan.memory_profile.subprocess_capacity,
            "source": plan.memory_profile.source,
        },
    }
