from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_agent.opt.contracts import OptContracts, OptVariable, load_opt_contracts, params_payload_from_variables
from code_agent.opt.objective import ObjectiveScore
from code_agent.opt.parallel_policy import plan_trial_execution, resolve_parallel_policy
from code_agent.opt.runner import RunOptConfig, run_optimization
from code_agent.opt.search import CMAESStrategyRunner
from code_agent.opt.strategy import resolve_strategy
from code_agent.opt.trials import RunOptOptions, TrialExecutor, TrialRequest, TrialResult
from code_agent.utils.local_execution import build_local_execution_env


def test_trial_executor_runs_explicit_subprocess_serial_batch(tmp_path: Path):
    case_dir = tmp_path / "case"
    src_dir = case_dir / "src"
    contracts_dir = case_dir / "contracts"
    src_dir.mkdir(parents=True)
    contracts_dir.mkdir()
    (src_dir / "helper.py").write_text(
        """
import json
from pathlib import Path

PAYLOAD = json.loads(Path("contracts/current_opt_params.json").read_text(encoding="utf-8"))
FORCE = float(PAYLOAD["params"]["action"]["force"])
""".lstrip(),
        encoding="utf-8",
    )
    (src_dir / "main.py").write_text(
        """
from __future__ import annotations

import argparse
import json
from pathlib import Path

from helper import FORCE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deformable-config", type=Path, default=None)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps({"score": FORCE}), encoding="utf-8")


if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )
    target_spec = {
        "schema_version": 1,
        "task_family": "unit_opt_runtime",
        "objective": {
            "type": "weighted_terms",
            "direction": "maximize",
            "terms": [{"name": "score", "metric_path": "score", "weight": 1.0, "transform": "identity"}],
        },
    }
    opt_space = {
        "schema_version": 1,
        "optimizer": "cma_es",
        "variables": [
            {
                "name": "action.force",
                "type": "float",
                "default": 1.0,
                "bounds": [0.0, 3.0],
                "scale": "linear",
                "owner": "action",
                "group": "control",
                "description": "Unit-test force scalar.",
            }
        ],
        "budget": {"max_trials": 2, "population_size": 3},
        "execution": {"parallel_policy": {"mode": "subprocess_serial"}},
    }
    default_params = {"schema_version": 1, "source": "default", "params": {"action": {"force": 1.0}}}
    (contracts_dir / "target_spec.json").write_text(json.dumps(target_spec), encoding="utf-8")
    (contracts_dir / "opt_space.json").write_text(json.dumps(opt_space), encoding="utf-8")
    (contracts_dir / "default_opt_params.json").write_text(json.dumps(default_params), encoding="utf-8")

    contracts = load_opt_contracts(case_dir=case_dir)
    options = RunOptOptions(
        backend="cpu",
        max_trials=2,
        population_size=3,
        seed=0,
        timeout_sec=20.0,
        steps=None,
        duration_sec=None,
        render_fps=None,
        target_video_frames=None,
        render_best=False,
        baseline_trials=1,
        best_repeat_trials=1,
        trial_root=case_dir / "artifacts" / "opt_trials",
        best_out_dir=case_dir / "artifacts" / "opt_best",
        current_params_path=contracts_dir / "current_opt_params.json",
        parallel_policy=resolve_parallel_policy(opt_space),
    )
    executor = TrialExecutor(
        case_dir=case_dir,
        contracts_dir=contracts_dir,
        reports_dir=case_dir / "reports",
        main_file="src/main.py",
    )

    results = executor.run_trials(
        [
            TrialRequest(0, {"schema_version": 1, "source": "trial", "params": {"action": {"force": 1.5}}}),
            TrialRequest(1, {"schema_version": 1, "source": "trial", "params": {"action": {"force": 2.25}}}),
        ],
        options=options,
        contracts=contracts,
    )

    assert [result.score.score for result in results] == [1.5, 2.25]
    assert [result.entry["execution_backend"] for result in results] == ["subprocess_serial", "subprocess_serial"]
    report = json.loads((case_dir / "reports" / "opt_trials" / "trial_001" / "execution_report.json").read_text())
    assert report["runner"] == "local"
    _assert_execution_environment_report(report, backend="cpu")


def test_cma_es_runner_submits_generation_batches(tmp_path: Path):
    variable = OptVariable(
        name="action.force",
        default=1.0,
        lower=0.0,
        upper=2.0,
        scale="linear",
        owner="action",
        group="control",
        description="Unit-test control scalar.",
    )
    opt_space: dict[str, Any] = {
        "schema_version": 1,
        "optimizer": "cma_es",
        "variables": [],
        "budget": {"max_trials": 4, "population_size": 3},
        "strategy": {"early_stop": {"enabled": False}},
    }
    contracts = OptContracts(
        case_dir=tmp_path,
        target_spec={
            "schema_version": 1,
            "task_family": "unit_search",
            "objective": {
                "type": "weighted_terms",
                "direction": "maximize",
                "terms": [{"name": "score", "metric_path": "score", "weight": 1.0, "transform": "identity"}],
            },
        },
        opt_space=opt_space,
        variables=(variable,),
        default_params=params_payload_from_variables([variable], source="default"),
        target_spec_path=tmp_path / "target_spec.json",
        opt_space_path=tmp_path / "opt_space.json",
        default_params_path=tmp_path / "default_opt_params.json",
    )
    options = RunOptOptions(
        backend="cpu",
        max_trials=4,
        population_size=3,
        seed=7,
        timeout_sec=10.0,
        steps=None,
        duration_sec=None,
        render_fps=None,
        target_video_frames=None,
        render_best=False,
        baseline_trials=1,
        best_repeat_trials=1,
        trial_root=tmp_path / "trials",
        best_out_dir=tmp_path / "best",
        current_params_path=tmp_path / "current.json",
        parallel_policy=resolve_parallel_policy(opt_space),
    )
    fake_trials = _BatchRecordingTrials()
    traces: list[dict[str, Any]] = []

    result = CMAESStrategyRunner(
        trials=fake_trials, trace_callback=traces.append, warning_callback=lambda warning: None
    ).run(
        contracts=contracts,
        options=options,
        strategy=resolve_strategy(opt_space),
        trial_index=0,
        best_result=None,
    )

    assert fake_trials.batch_sizes == [3, 1]
    assert [entry["trial_index"] for entry in traces] == [0, 1, 2, 3]
    assert result.trials_used == 4


def test_trial_executor_warns_when_generated_hook_ignores_or_clamps_params(tmp_path: Path):
    case_dir = tmp_path / "case"
    src_dir = case_dir / "src"
    contracts_dir = case_dir / "contracts"
    src_dir.mkdir(parents=True)
    contracts_dir.mkdir()
    (src_dir / "main.py").write_text(
        """
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deformable-config", type=Path, default=None)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "score": 1.0,
        "opt_params": {
            "action_param_diagnostics": {
                "sign_sensitive_opt_keys_ignored": ["travel"],
            }
        },
        "control_params": {
            "travel": 0.0,
            "gain": 10.0,
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )
    target_spec = {
        "schema_version": 1,
        "task_family": "unit_opt_runtime",
        "objective": {
            "type": "weighted_terms",
            "direction": "maximize",
            "terms": [{"name": "score", "metric_path": "score", "weight": 1.0, "transform": "identity"}],
        },
    }
    opt_space = {
        "schema_version": 1,
        "optimizer": "cma_es",
        "variables": [
            {
                "name": "action.travel",
                "type": "float",
                "default": 0.5,
                "bounds": [0.0, 1.0],
                "scale": "linear",
                "owner": "action",
                "group": "control",
                "description": "Unit-test travel.",
            },
            {
                "name": "action.gain",
                "type": "float",
                "default": 5.0,
                "bounds": [1.0, 20.0],
                "scale": "linear",
                "owner": "action",
                "group": "actuator",
                "description": "Unit-test gain.",
            },
        ],
        "budget": {"max_trials": 1},
    }
    default_params = {
        "schema_version": 1,
        "source": "default",
        "params": {"action": {"travel": 0.5, "gain": 5.0}},
    }
    (contracts_dir / "target_spec.json").write_text(json.dumps(target_spec), encoding="utf-8")
    (contracts_dir / "opt_space.json").write_text(json.dumps(opt_space), encoding="utf-8")
    (contracts_dir / "default_opt_params.json").write_text(json.dumps(default_params), encoding="utf-8")
    contracts = load_opt_contracts(case_dir=case_dir)
    options = RunOptOptions(
        backend="cpu",
        max_trials=1,
        population_size=3,
        seed=0,
        timeout_sec=20.0,
        steps=None,
        duration_sec=None,
        render_fps=None,
        target_video_frames=None,
        render_best=False,
        baseline_trials=1,
        best_repeat_trials=1,
        trial_root=case_dir / "artifacts" / "opt_trials",
        best_out_dir=case_dir / "artifacts" / "opt_best",
        current_params_path=contracts_dir / "current_opt_params.json",
        parallel_policy=resolve_parallel_policy(opt_space),
    )
    executor = TrialExecutor(
        case_dir=case_dir,
        contracts_dir=contracts_dir,
        reports_dir=case_dir / "reports",
        main_file="src/main.py",
    )

    result = executor.run_trial(
        trial_index=0,
        params_payload=contracts.default_params,
        options=options,
        contracts=contracts,
    )

    warnings = "\n".join(result.entry["warnings"])
    assert "ignored active sign-sensitive variables: action.travel" in warnings
    assert "action.gain requested=5 effective=10" in warnings
    assert "action.travel requested=0.5 effective=0" in warnings


def test_trial_executor_warns_when_metrics_do_not_echo_active_params(tmp_path: Path):
    case_dir, contracts, options = _write_fake_opt_case(
        tmp_path,
        variable={
            "name": "body.friction",
            "owner": "body",
            "group": "contact",
            "default": 1.0,
            "bounds": [0.1, 3.0],
        },
        parallel_policy={"mode": "subprocess_serial"},
    )
    executor = TrialExecutor(
        case_dir=case_dir,
        contracts_dir=case_dir / "contracts",
        reports_dir=case_dir / "reports",
        main_file="src/main.py",
    )

    result = executor.run_trial(
        trial_index=0,
        params_payload=contracts.default_params,
        options=options,
        contracts=contracts,
    )

    warnings = "\n".join(result.entry["warnings"])
    assert "do not echo requested or effective values for active opt variables: body.friction" in warnings


def test_parallel_policy_auto_ignores_disabled_deformable_defaults(tmp_path: Path):
    variable = OptVariable(
        name="action.force",
        default=1.0,
        lower=0.0,
        upper=2.0,
        scale="linear",
        owner="action",
        group="control",
        description="Unit-test control scalar.",
    )
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    (contracts_dir / "deformable_config.json").write_text(
        json.dumps({"enabled": False, "ipc_enabled": False, "ipc_contact_d_hat": 0.01}),
        encoding="utf-8",
    )
    contracts = OptContracts(
        case_dir=tmp_path,
        target_spec={
            "schema_version": 1,
            "task_family": "unit_policy",
            "objective": {
                "type": "weighted_terms",
                "direction": "maximize",
                "terms": [{"name": "score", "metric_path": "score", "weight": 1.0, "transform": "identity"}],
            },
        },
        opt_space={
            "schema_version": 1,
            "optimizer": "cma_es",
            "variables": [],
            "budget": {"max_trials": 1},
            "execution": {
                "parallel_policy": {
                    "gpu_memory_limit_gb": 14.0,
                    "gpu_memory_reserve_gb": 0.0,
                    "subprocess_gpu_increment_gb": 2.5,
                }
            },
        },
        variables=(variable,),
        default_params=params_payload_from_variables([variable], source="default"),
        target_spec_path=contracts_dir / "target_spec.json",
        opt_space_path=contracts_dir / "opt_space.json",
        default_params_path=contracts_dir / "default_opt_params.json",
    )

    plan = plan_trial_execution(
        policy=resolve_parallel_policy(contracts.opt_space),
        contracts=contracts,
        contracts_dir=contracts_dir,
        case_dir=tmp_path,
        request_variable_names=("action.force",),
        request_count=2,
    )
    assert plan.backend == "subprocess_parallel"

    (contracts_dir / "deformable_config.json").write_text(
        json.dumps({"enabled": True, "ipc_enabled": True}),
        encoding="utf-8",
    )
    contracts.opt_space["execution"]["parallel_policy"]["subprocess_gpu_increment_gb"] = 20.0
    plan = plan_trial_execution(
        policy=resolve_parallel_policy(contracts.opt_space),
        contracts=contracts,
        contracts_dir=contracts_dir,
        case_dir=tmp_path,
        request_variable_names=("action.force",),
        request_count=2,
    )
    assert plan.backend == "subprocess_serial"
    assert plan.reason == "auto_subprocess_capacity_serial_capacity"


def test_parallel_policy_ignores_unknown_legacy_mode_and_uses_subprocess_capacity(tmp_path: Path):
    case_dir, contracts, _options = _write_fake_opt_case(
        tmp_path,
        variable={
            "name": "action.force",
            "owner": "action",
            "group": "control",
            "default": 1.0,
            "bounds": [0.0, 3.0],
        },
        parallel_policy={
            "mode": "legacy_batch_mode",
            "gpu_memory_limit_gb": 10.0,
            "gpu_memory_reserve_gb": 0.0,
            "subprocess_gpu_increment_gb": 3.0,
        },
    )
    (case_dir / "contracts" / "deformable_config.json").write_text(
        json.dumps({"enabled": True, "ipc_enabled": True}),
        encoding="utf-8",
    )

    plan = plan_trial_execution(
        policy=resolve_parallel_policy(contracts.opt_space),
        contracts=contracts,
        contracts_dir=case_dir / "contracts",
        case_dir=case_dir,
        request_variable_names=("action.force",),
        request_count=6,
    )

    assert plan.backend == "subprocess_parallel"
    assert plan.batch_size == 3
    assert plan.reason == "auto_subprocess_capacity"
    assert plan.memory_profile.subprocess_capacity == 3


def test_parallel_policy_auto_uses_subprocess_for_env_local_high_memory_params(tmp_path: Path):
    case_dir, contracts, _options = _write_fake_opt_case(
        tmp_path,
        variable={
            "name": "action.force",
            "owner": "action",
            "group": "control",
            "default": 1.0,
            "bounds": [0.0, 3.0],
        },
        parallel_policy={
            "mode": "auto",
            "gpu_memory_limit_gb": 18.0,
            "gpu_memory_reserve_gb": 0.0,
            "subprocess_gpu_increment_gb": 8.0,
        },
    )
    (case_dir / "contracts" / "deformable_config.json").write_text(
        json.dumps({"enabled": True, "ipc_enabled": True}),
        encoding="utf-8",
    )

    plan = plan_trial_execution(
        policy=resolve_parallel_policy(contracts.opt_space),
        contracts=contracts,
        contracts_dir=case_dir / "contracts",
        case_dir=case_dir,
        request_variable_names=("action.force",),
        request_count=4,
    )

    assert plan.backend == "subprocess_parallel"
    assert plan.reason == "auto_subprocess_capacity"


def test_parallel_policy_groups_high_memory_subprocess_capacity(tmp_path: Path):
    variable = OptVariable(
        name="body.fem_youngs_modulus",
        default=100000.0,
        lower=10000.0,
        upper=200000.0,
        scale="linear",
        owner="body",
        group="material",
        description="Unit-test FEM material scalar.",
    )
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    (contracts_dir / "deformable_config.json").write_text(
        json.dumps({"enabled": True, "ipc_enabled": True}),
        encoding="utf-8",
    )
    contracts = OptContracts(
        case_dir=tmp_path,
        target_spec={
            "schema_version": 1,
            "task_family": "unit_policy",
            "objective": {
                "type": "weighted_terms",
                "direction": "maximize",
                "terms": [{"name": "score", "metric_path": "score", "weight": 1.0, "transform": "identity"}],
            },
        },
        opt_space={
            "schema_version": 1,
            "optimizer": "cma_es",
            "variables": [],
            "budget": {"max_trials": 4},
            "execution": {
                "parallel_policy": {
                    "mode": "auto",
                    "gpu_memory_limit_gb": 18.0,
                    "gpu_memory_reserve_gb": 0.0,
                    "subprocess_gpu_increment_gb": 8.0,
                }
            },
        },
        variables=(variable,),
        default_params=params_payload_from_variables([variable], source="default"),
        target_spec_path=contracts_dir / "target_spec.json",
        opt_space_path=contracts_dir / "opt_space.json",
        default_params_path=contracts_dir / "default_opt_params.json",
    )

    plan = plan_trial_execution(
        policy=resolve_parallel_policy(contracts.opt_space),
        contracts=contracts,
        contracts_dir=contracts_dir,
        case_dir=tmp_path,
        request_variable_names=("body.fem_youngs_modulus",),
        request_count=4,
    )

    assert plan.backend == "subprocess_parallel"
    assert plan.workers == 2
    assert plan.batch_size == 2
    assert plan.reason == "auto_subprocess_capacity"
    assert plan.memory_profile.subprocess_capacity == 2


def test_trial_executor_runs_isolated_subprocess_parallel_batch(tmp_path: Path):
    case_dir, contracts, options = _write_fake_opt_case(
        tmp_path,
        variable={
            "name": "action.force",
            "owner": "action",
            "group": "control",
            "default": 1.0,
            "bounds": [0.0, 3.0],
        },
        parallel_policy={"mode": "subprocess_parallel", "subprocess_workers": 2},
    )
    executor = TrialExecutor(
        case_dir=case_dir,
        contracts_dir=case_dir / "contracts",
        reports_dir=case_dir / "reports",
        main_file="src/main.py",
    )

    results = executor.run_trials(
        [
            TrialRequest(0, {"schema_version": 1, "source": "trial", "params": {"action": {"force": 0.5}}}),
            TrialRequest(1, {"schema_version": 1, "source": "trial", "params": {"action": {"force": 1.25}}}),
            TrialRequest(2, {"schema_version": 1, "source": "trial", "params": {"action": {"force": 2.0}}}),
        ],
        options=options,
        contracts=contracts,
    )

    assert [result.score.score for result in results] == [0.5, 1.25, 2.0]
    assert [result.entry["execution_backend"] for result in results] == [
        "subprocess_parallel",
        "subprocess_parallel",
        "subprocess_parallel",
    ]
    assert results[0].entry["execution_plan"]["workers"] == 2
    report = json.loads((case_dir / "reports" / "opt_trials" / "trial_000" / "execution_report.json").read_text())
    assert report["execution_backend"] == "subprocess_parallel"
    assert report["isolated_workspace"] is True
    assert report["execution_plan"]["workers"] == 2
    _assert_execution_environment_report(report, backend="cpu")


def test_run_optimization_cleans_stale_opt_outputs(tmp_path: Path):
    case_dir, _contracts, _options = _write_fake_opt_case(
        tmp_path,
        variable={
            "name": "action.force",
            "owner": "action",
            "group": "control",
            "default": 1.0,
            "bounds": [0.0, 3.0],
        },
        parallel_policy={"mode": "subprocess_serial"},
    )
    stale_report = case_dir / "reports" / "opt_trials" / "trial_999" / "execution_report.json"
    stale_metric = case_dir / "artifacts" / "opt_trials" / "trial_999" / "metrics.json"
    stale_report.parent.mkdir(parents=True)
    stale_metric.parent.mkdir(parents=True)
    stale_report.write_text("{}", encoding="utf-8")
    stale_metric.write_text("{}", encoding="utf-8")

    report = run_optimization(
        RunOptConfig(
            case_dir=case_dir,
            backend="cpu",
            max_trials=1,
            population_size=3,
            timeout_sec=20.0,
            render_best=False,
            main_file="src/main.py",
        )
    )

    assert report["status"] == "completed"
    assert report["num_trials"] >= 2
    assert not stale_report.exists()
    assert not stale_metric.exists()


class _BatchRecordingTrials:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def run_trials(
        self,
        requests: list[TrialRequest],
        *,
        options: RunOptOptions,
        contracts: OptContracts,
    ) -> list[TrialResult]:
        self.batch_sizes.append(len(requests))
        results: list[TrialResult] = []
        for request in requests:
            score = float(request.params_payload["params"]["action"]["force"])
            results.append(
                TrialResult(
                    entry={"trial_index": request.trial_index, "status": "completed"},
                    score=ObjectiveScore(score=score, success=False, terms={}, measured={}),
                    params_payload=request.params_payload,
                )
            )
        return results

    def run_trial(self, **kwargs):
        raise AssertionError("CMAESStrategyRunner should use run_trials for search candidates")


def _assert_execution_environment_report(report: dict[str, Any], *, backend: str) -> None:
    environment = report["environment"]
    expected_env = build_local_execution_env({"GENESIS_BACKEND": backend})
    assert environment["GENESIS_BACKEND"] == backend
    assert environment["LD_LIBRARY_PATH"] == expected_env["LD_LIBRARY_PATH"]
    assert environment["uv_path"] is not None
    cuda_lib = Path(__file__).resolve().parents[1] / ".venv" / "cuda-12.8" / "lib"
    if cuda_lib.exists():
        assert str(cuda_lib) in environment["LD_LIBRARY_PATH"].split(":")


def _write_fake_opt_case(
    tmp_path: Path,
    *,
    variable: dict[str, Any],
    parallel_policy: dict[str, Any],
) -> tuple[Path, OptContracts, RunOptOptions]:
    case_dir = tmp_path / "case"
    src_dir = case_dir / "src"
    contracts_dir = case_dir / "contracts"
    src_dir.mkdir(parents=True)
    contracts_dir.mkdir()
    variable_name = str(variable["name"])
    first_key, second_key = variable_name.split(".", maxsplit=1)
    (src_dir / "helper.py").write_text(
        f"""
import json
from pathlib import Path

PAYLOAD = json.loads(Path("contracts/current_opt_params.json").read_text(encoding="utf-8"))
VALUE = float(PAYLOAD["params"]["{first_key}"]["{second_key}"])
""".lstrip(),
        encoding="utf-8",
    )
    (src_dir / "main.py").write_text(
        """
from __future__ import annotations

import argparse
import json
from pathlib import Path

from helper import VALUE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deformable-config", type=Path, default=None)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps({"score": VALUE}), encoding="utf-8")


if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )
    target_spec = {
        "schema_version": 1,
        "task_family": "unit_opt_runtime",
        "objective": {
            "type": "weighted_terms",
            "direction": "maximize",
            "terms": [{"name": "score", "metric_path": "score", "weight": 1.0, "transform": "identity"}],
        },
    }
    opt_space = {
        "schema_version": 1,
        "optimizer": "cma_es",
        "variables": [
            {
                "name": variable_name,
                "type": "float",
                "default": float(variable["default"]),
                "bounds": variable["bounds"],
                "scale": "linear",
                "owner": variable["owner"],
                "group": variable["group"],
                "description": "Unit-test scalar.",
            }
        ],
        "budget": {"max_trials": 3, "population_size": 3},
        "execution": {"parallel_policy": parallel_policy},
    }
    default_params = {
        "schema_version": 1,
        "source": "default",
        "params": {first_key: {second_key: float(variable["default"])}},
    }
    (contracts_dir / "target_spec.json").write_text(json.dumps(target_spec), encoding="utf-8")
    (contracts_dir / "opt_space.json").write_text(json.dumps(opt_space), encoding="utf-8")
    (contracts_dir / "default_opt_params.json").write_text(json.dumps(default_params), encoding="utf-8")

    contracts = load_opt_contracts(case_dir=case_dir)
    options = RunOptOptions(
        backend="cpu",
        max_trials=3,
        population_size=3,
        seed=0,
        timeout_sec=20.0,
        steps=None,
        duration_sec=None,
        render_fps=None,
        target_video_frames=None,
        render_best=False,
        baseline_trials=1,
        best_repeat_trials=1,
        trial_root=case_dir / "artifacts" / "opt_trials",
        best_out_dir=case_dir / "artifacts" / "opt_best",
        current_params_path=contracts_dir / "current_opt_params.json",
        parallel_policy=resolve_parallel_policy(opt_space),
    )
    return case_dir, contracts, options
