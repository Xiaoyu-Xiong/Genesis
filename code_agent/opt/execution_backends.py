from __future__ import annotations

import concurrent.futures
import shutil
import tempfile
from pathlib import Path
from typing import Any

from code_agent.opt.contracts import OptContracts, write_params_payload
from code_agent.opt.parallel_policy import TrialExecutionPlan
from code_agent.opt.trials import RunOptOptions, TrialRequest, payload_for_trial
from code_agent.utils.local_execution import LocalRunConfig, _write_json, run_local


class SubprocessParallelTrialBackend:
    """Run trials concurrently in isolated workspace copies."""

    def __init__(self, executor, plan: TrialExecutionPlan) -> None:
        self.executor = executor
        self.plan = plan

    def run_trials(
        self,
        requests: list[TrialRequest],
        *,
        options: RunOptOptions,
        contracts: OptContracts,
    ):
        if not requests:
            return []
        prepared_trials = [
            self.executor._prepare_trial(
                trial_index=request.trial_index,
                params_payload=request.params_payload,
                options=options,
            )
            for request in requests
        ]
        results_by_trial: dict[int, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, self.plan.workers)) as pool:
            futures = [
                pool.submit(
                    _run_isolated_subprocess_trial,
                    executor=self.executor,
                    prepared=prepared,
                    options=options,
                    plan=self.plan,
                )
                for prepared in prepared_trials
            ]
            for future in concurrent.futures.as_completed(futures):
                prepared, raw_report = future.result()
                result = self.executor._result_from_report(prepared, raw_report, contracts)
                result.entry["execution_backend"] = "subprocess_parallel"
                result.entry["execution_plan"] = _plan_report(self.plan)
                results_by_trial[prepared.trial_index] = result
        return [results_by_trial[prepared.trial_index] for prepared in prepared_trials]


def _run_isolated_subprocess_trial(
    *,
    executor,
    prepared,
    options: RunOptOptions,
    plan: TrialExecutionPlan,
):
    with tempfile.TemporaryDirectory(prefix="code_agent_opt_trial_") as temp_root:
        temp_case_dir = Path(temp_root) / executor.case_dir.name
        shutil.copytree(executor.case_dir, temp_case_dir, ignore=_workspace_copy_ignore)
        current_params_path = _workspace_relative_path(
            original_path=options.current_params_path,
            original_root=executor.case_dir,
            copied_root=temp_case_dir,
        )
        current_payload = payload_for_trial(
            prepared.params_payload,
            source="current",
            trial_index=prepared.trial_index,
            metadata={"trial_params_path": executor.rel(prepared.params_path)},
        )
        current_params_path.parent.mkdir(parents=True, exist_ok=True)
        write_params_payload(current_params_path, current_payload)
        raw_report = run_local(
            LocalRunConfig(
                workspace_dir=temp_case_dir,
                main_file=executor.main_file,
                output_dir=prepared.report_dir,
                timeout_sec=options.timeout_sec,
                python_executable="uv run --no-sync python",
                extra_args=tuple(_isolated_main_args(temp_case_dir, prepared.artifacts_dir, options)),
                artifact_dir_names=(),
                artifact_file_names=(),
                extra_artifact_paths=(str(prepared.artifacts_dir),),
                env={"GENESIS_BACKEND": options.backend},
            )
        )
    raw_report["execution_backend"] = "subprocess_parallel"
    raw_report["execution_plan"] = _plan_report(plan)
    raw_report["isolated_workspace"] = True
    _write_json(prepared.report_dir / "execution_report.json", raw_report)
    return prepared, raw_report


def _isolated_main_args(temp_case_dir: Path, artifacts_dir: Path, options: RunOptOptions) -> list[str]:
    args = ["--backend", options.backend, "--out-dir", str(artifacts_dir)]
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
    deformable_config_path = temp_case_dir / "contracts" / "deformable_config.json"
    if deformable_config_path.is_file():
        args.extend(("--deformable-config", "contracts/deformable_config.json"))
    args.append("--no-render")
    return args


def _workspace_relative_path(*, original_path: Path, original_root: Path, copied_root: Path) -> Path:
    try:
        rel_path = original_path.resolve().relative_to(original_root.resolve())
    except ValueError:
        rel_path = Path("contracts") / original_path.name
    return copied_root / rel_path


def _workspace_copy_ignore(directory: str, names: list[str]) -> set[str]:
    del directory
    ignored = {".git", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "artifacts", "logs", "reports"}
    return set(names) & ignored


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
