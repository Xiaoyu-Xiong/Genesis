from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from code_agent.assets.mesh import episode as mesh_episode
from code_agent.assets.mesh.manifest import _scale_to_bbox, _uniform_scale_factor
from code_agent.assets.mesh.models import MeshGenesisFEMImportResult
from code_agent.assets.mesh.request_adapter import request_size
from code_agent.assets.builtin_guard import builtin_asset_violations
from code_agent.assets.xml.validation import _validate_mesh_asset_paths, validate_xml_asset
from code_agent.configs import deformable_config_dict
from code_agent.opt import agent as opt_agent
from code_agent.opt.contracts import OptContractError, load_opt_contracts
from code_agent.opt.objective import evaluate_objective
from code_agent.opt.optimizers.cma_es import CMAESOptimizer
from code_agent.opt.types import OptAgentRequest
from code_agent.planner.session import PlannerSession, PlannerSessionConfig
from code_agent.utils import codex
from code_agent.utils.integrator import write_main


def test_run_codex_exec_resolves_relative_io_paths_against_repo_root(tmp_path, monkeypatch):
    fake_repo = tmp_path / "repo"
    polluted_cwd = tmp_path / "polluted"
    fake_repo.mkdir()
    polluted_cwd.mkdir()
    monkeypatch.setattr(codex, "DEFAULT_REPO_ROOT", fake_repo)
    monkeypatch.chdir(polluted_cwd)

    result = codex.run_codex_exec(
        codex.CodexExecRequest(
            role="test",
            prompt="",
            cwd=Path("."),
            output_jsonl_path=Path("logs/codex_test.jsonl"),
            final_message_path=Path("logs/codex_test.final.txt"),
            codex_bin="definitely_missing_codex_binary_for_test",
        )
    )

    expected_jsonl = fake_repo / "logs" / "codex_test.jsonl"
    expected_final = fake_repo / "logs" / "codex_test.final.txt"
    assert result.output_jsonl_path == str(expected_jsonl.resolve())
    assert result.final_message_path == str(expected_final.resolve())
    assert result.cwd == str(fake_repo.resolve())
    assert expected_jsonl.is_file()
    assert expected_final.is_file()
    assert not (polluted_cwd / "logs" / "codex_test.jsonl").exists()


def test_codex_command_hides_genesis_builtin_assets(tmp_path):
    final_path = tmp_path / "logs" / "final.json"
    jsonl_path = tmp_path / "logs" / "events.jsonl"
    command = codex.build_codex_exec_command(
        codex.CodexExecRequest(
            role="test",
            prompt="test",
            cwd=Path("."),
            output_jsonl_path=jsonl_path,
            final_message_path=final_path,
            writable_roots=(tmp_path,),
        ),
        resolved_codex="/usr/bin/codex",
    )

    if command[0].endswith("bwrap"):
        assert "--tmpfs" in command
        assert str((codex.DEFAULT_REPO_ROOT / "genesis" / "assets").resolve()) in command
        assert str(tmp_path.resolve()) in command


def test_builtin_asset_guard_flags_genesis_assets_references():
    violations = builtin_asset_violations(
        {
            "scene_plan": {
                "simulation_strategy": (
                    "Use genesis/assets/meshes/bunny.obj, gs.utils.get_assets_dir(), and Path(gs.__file__).parent."
                )
            }
        },
        label="planner_output",
    )

    assert any("genesis/assets" in item for item in violations)
    assert any("get_assets_dir" in item for item in violations)
    assert any("gs.__file__" in item for item in violations)


def test_run_opt_agent_uses_parseable_payload_even_if_codex_times_out(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "artifacts" / "opt_best").mkdir(parents=True)
    (case_dir / "artifacts" / "opt_best" / "render.mp4").write_bytes(b"fake")
    payload = {
        "status": "success",
        "case_type": "test_case",
        "edited_files": ["src/action.py"],
        "optimized_variables": ["action.gain"],
        "baseline": {"success": False},
        "best": {"success": True, "video_path": "artifacts/opt_best/render.mp4"},
        "diagnosis": "parameters were limiting behavior",
        "recommendation": "accept best params",
        "evidence": ["video_checked=sampled artifacts/opt_best/render.mp4 and the target behavior is visible"],
        "opt_report_path": "reports/opt_report.json",
        "failures": [],
    }

    def fake_run_codex_exec(request):
        request.output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        request.final_message_path.write_text(json.dumps(payload), encoding="utf-8")
        request.output_jsonl_path.write_text("", encoding="utf-8")
        stderr_path = request.output_jsonl_path.with_suffix(request.output_jsonl_path.suffix + ".stderr")
        stderr_path.write_text("", encoding="utf-8")
        return codex.CodexExecResult(
            role=request.role,
            success=False,
            exit_code=-9,
            duration_sec=1.0,
            command=["codex"],
            cwd=str(request.cwd),
            sandbox=request.sandbox,
            output_jsonl_path=str(request.output_jsonl_path),
            final_message_path=str(request.final_message_path),
            output_schema_path=str(request.output_schema_path),
            codex_version="codex-test",
            error_type="timeout",
            error_message="timed out after writing final payload",
            stderr_path=str(stderr_path),
            timed_out=True,
            started_at_unix=time.time(),
            ended_at_unix=time.time() + 1.0,
        )

    monkeypatch.setattr(opt_agent, "run_codex_exec", fake_run_codex_exec)

    result = opt_agent.run_opt_agent(OptAgentRequest(case_dir=case_dir, original_prompt="test"))

    assert result.status == "success"
    assert result.best == {"success": True}
    report = json.loads((case_dir / "reports" / "opt_subagent_report.json").read_text(encoding="utf-8"))
    assert report["result"]["status"] == "success"


def test_run_opt_agent_downgrades_success_without_video_evidence(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    payload = {
        "status": "success",
        "case_type": "test_case",
        "edited_files": ["src/action.py"],
        "optimized_variables": ["action.gain"],
        "baseline": {"success": False},
        "best": {"success": True, "score": 5.0},
        "diagnosis": "numeric metrics passed",
        "recommendation": "accept best params",
        "evidence": ["best_score=5.0"],
        "opt_report_path": "reports/opt_report.json",
        "failures": [],
    }

    def fake_run_codex_exec(request):
        request.output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        request.final_message_path.write_text(json.dumps(payload), encoding="utf-8")
        request.output_jsonl_path.write_text("", encoding="utf-8")
        stderr_path = request.output_jsonl_path.with_suffix(request.output_jsonl_path.suffix + ".stderr")
        stderr_path.write_text("", encoding="utf-8")
        return codex.CodexExecResult(
            role=request.role,
            success=True,
            exit_code=0,
            duration_sec=1.0,
            command=["codex"],
            cwd=str(request.cwd),
            sandbox=request.sandbox,
            output_jsonl_path=str(request.output_jsonl_path),
            final_message_path=str(request.final_message_path),
            output_schema_path=str(request.output_schema_path),
            codex_version="codex-test",
            error_type=None,
            error_message=None,
            stderr_path=str(stderr_path),
            timed_out=False,
            started_at_unix=time.time(),
            ended_at_unix=time.time() + 1.0,
        )

    monkeypatch.setattr(opt_agent, "run_codex_exec", fake_run_codex_exec)

    result = opt_agent.run_opt_agent(OptAgentRequest(case_dir=case_dir, original_prompt="test"))

    assert result.status == "needs_more_optimization"
    assert "missing_explicit_video_evidence" in result.failures


def test_run_opt_agent_downgrades_unvalidated_xml_patch(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "artifacts" / "opt_best").mkdir(parents=True)
    (case_dir / "artifacts" / "opt_best" / "render.mp4").write_bytes(b"fake")
    payload = {
        "status": "success",
        "case_type": "test_case",
        "edited_files": ["assets/xml/robot/robot.xml"],
        "optimized_variables": ["xml.actuator.wrist_kp"],
        "baseline": {"success": False},
        "best": {"success": True, "video_path": "artifacts/opt_best/render.mp4"},
        "diagnosis": "xml kp improved behavior",
        "recommendation": "accept best params",
        "evidence": ["video_checked=sampled artifacts/opt_best/render.mp4 and behavior is visible"],
        "opt_report_path": "reports/opt_report.json",
        "failures": [],
    }

    def fake_run_codex_exec(request):
        request.output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        request.final_message_path.write_text(json.dumps(payload), encoding="utf-8")
        request.output_jsonl_path.write_text("", encoding="utf-8")
        stderr_path = request.output_jsonl_path.with_suffix(request.output_jsonl_path.suffix + ".stderr")
        stderr_path.write_text("", encoding="utf-8")
        return codex.CodexExecResult(
            role=request.role,
            success=True,
            exit_code=0,
            duration_sec=1.0,
            command=["codex"],
            cwd=str(request.cwd),
            sandbox=request.sandbox,
            output_jsonl_path=str(request.output_jsonl_path),
            final_message_path=str(request.final_message_path),
            output_schema_path=str(request.output_schema_path),
            codex_version="codex-test",
            error_type=None,
            error_message=None,
            stderr_path=str(stderr_path),
            timed_out=False,
            started_at_unix=time.time(),
            ended_at_unix=time.time() + 1.0,
        )

    monkeypatch.setattr(opt_agent, "run_codex_exec", fake_run_codex_exec)

    result = opt_agent.run_opt_agent(OptAgentRequest(case_dir=case_dir, original_prompt="test"))

    assert result.status == "needs_more_optimization"
    assert "missing_xml_scalar_patch_validation" in result.failures


def test_run_opt_agent_recovers_fresh_opt_report_after_timeout(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    (case_dir / "logs").mkdir(parents=True)
    (case_dir / "reports").mkdir()
    (case_dir / "contracts").mkdir()
    (case_dir / "artifacts" / "opt_trials" / "trial_000").mkdir(parents=True)
    (case_dir / "artifacts" / "opt_trials" / "trial_001").mkdir(parents=True)
    (case_dir / "artifacts" / "opt_best").mkdir(parents=True)
    (case_dir / "contracts" / "target_spec.json").write_text(
        json.dumps({"schema_version": 1, "task_family": "unit_opt", "objective": {"direction": "maximize"}}),
        encoding="utf-8",
    )
    (case_dir / "contracts" / "opt_space.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variables": [
                    {
                        "name": "action.force",
                        "owner": "action",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "artifacts" / "opt_best" / "render.mp4").write_bytes(b"fake")

    def fake_run_codex_exec(request):
        started = time.time()
        request.output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        request.final_message_path.write_text("timeout: Codex invocation timed out", encoding="utf-8")
        request.output_jsonl_path.write_text("", encoding="utf-8")
        stderr_path = request.output_jsonl_path.with_suffix(request.output_jsonl_path.suffix + ".stderr")
        stderr_path.write_text("", encoding="utf-8")
        (case_dir / "reports" / "opt_trace.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "trial_index": 0,
                            "metrics_path": "artifacts/opt_trials/trial_000/metrics.json",
                            "params_path": "artifacts/opt_trials/trial_000/opt_params.json",
                            "objective": {"success": False},
                        }
                    ),
                    json.dumps(
                        {
                            "trial_index": 1,
                            "metrics_path": "artifacts/opt_trials/trial_001/metrics.json",
                            "params_path": "artifacts/opt_trials/trial_001/opt_params.json",
                            "objective": {"success": False},
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )
        (case_dir / "reports" / "verification_report.json").write_text(
            json.dumps({"schema_version": 1, "success": False, "score": 2.0, "target": {}, "measured": {}, "terms": {}}),
            encoding="utf-8",
        )
        (case_dir / "reports" / "opt_report.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "completed",
                    "optimizer": "cma_es",
                    "num_trials": 2,
                    "baseline_score": 1.0,
                    "best_trial": 1,
                    "best_score": 2.0,
                    "best_params_path": "contracts/best_opt_params.json",
                    "best_render_dir": "artifacts/opt_best",
                    "trace_path": "reports/opt_trace.jsonl",
                    "verification_report_path": "reports/verification_report.json",
                    "budget": {},
                    "summary": "Optimization completed; best score improved over baseline.",
                    "failures": [],
                    "warnings": [],
                }
            ),
            encoding="utf-8",
        )
        return codex.CodexExecResult(
            role=request.role,
            success=False,
            exit_code=-9,
            duration_sec=1.0,
            command=["codex"],
            cwd=str(request.cwd),
            sandbox=request.sandbox,
            output_jsonl_path=str(request.output_jsonl_path),
            final_message_path=str(request.final_message_path),
            output_schema_path=str(request.output_schema_path),
            codex_version="codex-test",
            error_type="timeout",
            error_message="timed out after writing opt report",
            stderr_path=str(stderr_path),
            timed_out=True,
            started_at_unix=started,
            ended_at_unix=time.time() + 1.0,
        )

    monkeypatch.setattr(opt_agent, "run_codex_exec", fake_run_codex_exec)

    result = opt_agent.run_opt_agent(OptAgentRequest(case_dir=case_dir, original_prompt="test"))

    assert result.status == "needs_more_optimization"
    assert result.opt_report_path == "reports/opt_report.json"
    assert result.best["score"] == 2.0
    assert result.optimized_variables == ["action.force"]
    report = json.loads((case_dir / "reports" / "opt_subagent_report.json").read_text(encoding="utf-8"))
    assert report["result"]["status"] == "needs_more_optimization"


def test_planner_summary_reconciles_late_opt_success_report(tmp_path):
    case_dir = tmp_path / "case"
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="case",
            task="task",
            case_dir=case_dir,
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
            opt_enabled=True,
        )
    )
    session._ensure_dirs()
    session.state["opt"]["attempts"] = 1
    session.state["opt"]["status"] = "failed"
    session.state["opt"]["latest_result"] = {"status": "failed", "diagnosis": "timeout"}
    session.state["critic"] = {"verdict": "fail"}
    (case_dir / "contracts" / "best_opt_params.json").write_text(
        json.dumps({"params": {"action": {"gain": 2.0}}}),
        encoding="utf-8",
    )
    success_payload = {
        "status": "success",
        "case_type": "test_case",
        "edited_files": ["src/action.py"],
        "optimized_variables": ["action.gain"],
        "baseline": {"success": False},
        "best": {"success": True},
        "diagnosis": "late success",
        "recommendation": "rerun",
        "evidence": [],
        "opt_report_path": "reports/opt_report.json",
        "failures": [],
    }
    (case_dir / "logs" / "codex_opt_subagent.final.json").write_text(
        json.dumps(success_payload),
        encoding="utf-8",
    )

    summary = session.build_summary()

    assert summary["opt"]["status"] == "success"
    assert summary["opt"]["latest_result"]["best"] == {"success": True}
    assert session.state["control"]["needs_execution"] is True
    current_params = json.loads((case_dir / "contracts" / "current_opt_params.json").read_text(encoding="utf-8"))
    assert current_params["metadata"]["selected_by"] == "planner.opt_disk_reconcile"


def test_opt_contract_rejects_rendering_owner(tmp_path):
    case_dir = tmp_path / "case"
    contracts_dir = case_dir / "contracts"
    contracts_dir.mkdir(parents=True)
    (contracts_dir / "target_spec.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_family": "unit_test",
                "objective": {
                    "type": "weighted_terms",
                    "direction": "maximize",
                    "terms": [
                        {
                            "name": "score",
                            "metric_path": "score",
                            "weight": 1.0,
                            "transform": "identity",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (contracts_dir / "opt_space.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "optimizer": "cma_es",
                "variables": [
                    {
                        "name": "camera.distance",
                        "type": "float",
                        "default": 1.0,
                        "bounds": [0.5, 2.0],
                        "scale": "linear",
                        "owner": "rendering",
                        "description": "Rendering should not be an optimization owner.",
                    }
                ],
                "budget": {"max_trials": 1},
            }
        ),
        encoding="utf-8",
    )
    (contracts_dir / "default_opt_params.json").write_text(
        json.dumps({"schema_version": 1, "source": "default", "params": {"camera": {"distance": 1.0}}}),
        encoding="utf-8",
    )

    try:
        load_opt_contracts(case_dir=case_dir)
    except OptContractError as exc:
        assert "unsupported owner" in str(exc)
    else:
        raise AssertionError("rendering owner should be rejected by opt contracts")


def test_opt_contract_accepts_xml_scalar_owner(tmp_path):
    case_dir = tmp_path / "case"
    contracts_dir = case_dir / "contracts"
    contracts_dir.mkdir(parents=True)
    (contracts_dir / "target_spec.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_family": "unit_xml_opt",
                "objective": {
                    "type": "weighted_terms",
                    "direction": "minimize",
                    "terms": [
                        {
                            "name": "angle_error",
                            "metric_path": "angle_error",
                            "weight": 1.0,
                            "transform": "identity",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (contracts_dir / "opt_space.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "optimizer": "cma_es",
                "variables": [
                    {
                        "name": "xml.actuator.wrist_kp",
                        "type": "float",
                        "default": 120.0,
                        "bounds": [30.0, 500.0],
                        "scale": "log",
                        "owner": "xml",
                        "group": "actuator",
                        "description": "Existing XML actuator kp scalar.",
                    },
                    {
                        "name": "body.initial.card_lean",
                        "type": "float",
                        "default": 0.08,
                        "bounds": [-0.15, 0.15],
                        "scale": "linear",
                        "owner": "body",
                        "group": "initial",
                        "description": "Initial lean angle for a balance setup.",
                    },
                ],
                "budget": {"max_trials": 1},
            }
        ),
        encoding="utf-8",
    )
    (contracts_dir / "default_opt_params.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "default",
                "params": {
                    "xml": {"actuator": {"wrist_kp": 120.0}},
                    "body": {"initial": {"card_lean": 0.08}},
                },
            }
        ),
        encoding="utf-8",
    )

    contracts = load_opt_contracts(case_dir=case_dir)

    assert [variable.owner for variable in contracts.active_variables] == ["xml", "body"]
    assert [variable.group for variable in contracts.active_variables] == ["actuator", "initial"]


def test_opt_contract_accepts_generic_strategy_knobs(tmp_path):
    case_dir = tmp_path / "case"
    contracts_dir = case_dir / "contracts"
    contracts_dir.mkdir(parents=True)
    (contracts_dir / "target_spec.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_family": "unit_test",
                "objective": {
                    "type": "weighted_terms",
                    "direction": "maximize",
                    "terms": [
                        {
                            "name": "distance_progress",
                            "metric_path": "progress",
                            "weight": 1.0,
                            "transform": "identity",
                        }
                    ],
                },
                "success_criteria": [{"name": "done", "metric_path": "done", "op": "==", "threshold": True}],
            }
        ),
        encoding="utf-8",
    )
    (contracts_dir / "opt_space.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "optimizer": "cma_es",
                "variables": [
                    {
                        "name": "action.force",
                        "type": "float",
                        "default": 1.0,
                        "bounds": [0.1, 3.0],
                        "scale": "linear",
                        "owner": "action",
                        "group": "control",
                        "description": "Test control variable.",
                    },
                    {
                        "name": "body.friction",
                        "type": "float",
                        "default": 0.5,
                        "bounds": [0.1, 1.0],
                        "scale": "linear",
                        "owner": "body",
                        "group": "contact",
                        "description": "Test contact variable.",
                    },
                ],
                "budget": {"max_trials": 4, "population_size": None},
                "strategy": {
                    "early_stop": {"enabled": True, "patience_generations": 2, "min_delta": 0.01},
                    "restarts": [{"name": "wide", "sigma_scale": 1.5, "max_trials": 2}],
                    "phases": [
                        {"name": "control_first", "groups": ["control"], "max_trials": 2},
                        {"name": "contact_next", "groups": ["contact"], "max_trials": 2, "start_from_best": True},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    contracts = load_opt_contracts(case_dir=case_dir)

    assert [variable.name for variable in contracts.active_variables] == ["action.force", "body.friction"]
    assert contracts.opt_space["strategy"]["phases"][0]["groups"] == ["control"]


def test_opt_contract_rejects_custom_objective_transform(tmp_path):
    case_dir = tmp_path / "case"
    contracts_dir = case_dir / "contracts"
    contracts_dir.mkdir(parents=True)
    (contracts_dir / "target_spec.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_family": "unit_test",
                "objective": {
                    "type": "weighted_terms",
                    "direction": "maximize",
                    "terms": [
                        {
                            "name": "custom_score",
                            "metric_path": "score",
                            "weight": 1.0,
                            "transform": "custom",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (contracts_dir / "opt_space.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "optimizer": "cma_es",
                "variables": [
                    {
                        "name": "action.force",
                        "type": "float",
                        "default": 1.0,
                        "bounds": [0.0, 2.0],
                        "scale": "linear",
                        "owner": "action",
                        "description": "Test variable.",
                    }
                ],
                "budget": {"max_trials": 1},
            }
        ),
        encoding="utf-8",
    )

    try:
        load_opt_contracts(case_dir=case_dir)
    except OptContractError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("custom objective transforms should be rejected")


def test_objective_without_success_criteria_is_not_successful():
    score = evaluate_objective(
        target_spec={
            "schema_version": 1,
            "task_family": "unit_test",
            "objective": {
                "type": "weighted_terms",
                "direction": "maximize",
                "terms": [
                    {"name": "score", "metric_path": "score", "weight": 1.0, "transform": "identity"}
                ],
            },
        },
        metrics={"score": 10.0},
    )

    assert score.score == 10.0
    assert score.success is False
    assert any("success_criteria" in warning for warning in score.warnings)


def test_cma_es_keeps_candidates_in_unit_interval():
    optimizer = CMAESOptimizer(dim=2, mean=[0.98, 0.02], initial_sigmas=[0.9, 0.9], seed=0, population_size=12)
    samples = optimizer.ask(count=32)

    assert all(0.0 <= value <= 1.0 for sample in samples for value in sample)
    assert any(0.0 < value < 1.0 for sample in samples for value in sample)


def test_cma_es_handles_tail_batches_without_pycma_update():
    optimizer = CMAESOptimizer(dim=1, mean=[0.5], seed=0, population_size=3)
    for count in (1, 2, 3):
        samples = optimizer.ask(count=count)
        assert len(samples) == count
        assert all(0.0 <= sample[0] <= 1.0 for sample in samples)
        optimizer.tell(samples, [-(sample[0] - 0.7) ** 2 for sample in samples], maximize=True)

    state = optimizer.state()
    assert state.iteration == 1
    assert state.best_score is not None


def test_xml_validator_accepts_explicit_passive_freejoint_projectile(tmp_path):
    xml_path = tmp_path / "passive_ring.xml"
    xml_path.write_text(
        """
<mujoco model="passive_ring">
  <worldbody>
    <body name="ring" pos="0 0 0">
      <freejoint name="ring_freejoint"/>
      <geom name="ring_top" type="capsule" fromto="-0.1 0 0 0.1 0 0" size="0.02" rgba="1 0 0 1"/>
      <geom name="ring_bottom" type="capsule" fromto="-0.1 0.1 0 0.1 0.1 0" size="0.02" rgba="1 0 0 1"/>
      <geom name="ring_left" type="capsule" fromto="-0.1 0 0 -0.1 0.1 0" size="0.02" rgba="1 0 0 1"/>
      <geom name="ring_right" type="capsule" fromto="0.1 0 0 0.1 0.1 0" size="0.02" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )

    strict_report = validate_xml_asset(xml_path)
    passive_report = validate_xml_asset(xml_path, allow_passive_freejoint=True)

    assert strict_report["ok"] is False
    assert "XML asset must include at least one actuator" in " ".join(strict_report["errors"])
    assert passive_report["ok"] is True
    assert passive_report["passive_freejoint_ok"] is True


def test_xml_mesh_paths_allow_case_generated_meshes_and_reject_builtin_assets(tmp_path):
    xml_path = tmp_path / "asset.xml"
    mesh_path = tmp_path / "meshes" / "generated_part.obj"
    mesh_path.parent.mkdir()
    mesh_path.write_text("# generated case-local mesh\n", encoding="utf-8")

    local_root = ET.fromstring(
        '<mujoco><asset><mesh name="generated_part" file="meshes/generated_part.obj"/></asset></mujoco>'
    )
    errors: list[str] = []
    warnings: list[str] = []
    _validate_mesh_asset_paths(local_root, xml_path=xml_path, allowed_asset_roots=(), errors=errors, warnings=warnings)
    assert errors == []

    builtin_mesh = codex.DEFAULT_REPO_ROOT / "genesis" / "assets" / "meshes" / "bunny.obj"
    builtin_root = ET.fromstring(f'<mujoco><asset><mesh name="builtin" file="{builtin_mesh}"/></asset></mujoco>')
    errors = []
    _validate_mesh_asset_paths(
        builtin_root,
        xml_path=xml_path,
        allowed_asset_roots=(),
        errors=errors,
        warnings=[],
    )
    assert any("forbidden Genesis built-in assets" in item for item in errors)


def test_planner_schema_lookup_is_repo_relative_when_cwd_is_polluted(tmp_path, monkeypatch):
    polluted_cwd = tmp_path / "polluted"
    polluted_cwd.mkdir()
    monkeypatch.chdir(polluted_cwd)
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="case",
            task="task",
            case_dir=tmp_path / "case",
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )

    schema = session.load_json(Path("code_agent/specs/planner_action.schema.json"))
    errors = session.validate_json_schema({}, Path("code_agent/specs/planner_action.schema.json"))

    assert schema is not None
    assert isinstance(errors, list)
    assert not any("schema missing" in error for error in errors)


def test_deformable_config_does_not_override_fem_friction_mu():
    cfg = deformable_config_dict(deformable_enabled=True, ipc_enabled=True)

    assert "fem_friction_mu" not in cfg
    assert "friction" in cfg


def test_generated_main_adds_repo_root_before_code_agent_import(tmp_path):
    run_dir = tmp_path / "case"
    main_path = write_main(
        run_dir=run_dir,
        task="task",
        default_steps=1,
        default_render_fps=1,
        default_duration_sec=1.0,
        default_target_video_frames=1,
        deformable_cfg={"ipc_contact_d_hat_adaptive": True},
    )

    source = main_path.read_text(encoding="utf-8")
    sys_path_pos = source.index("sys.path.insert(0, str(REPO_ROOT))")
    adaptive_import_pos = source.index("from code_agent.utils.adaptive_ipc import")

    assert sys_path_pos < adaptive_import_pos


def test_generated_mesh_manifest_uses_uniform_bbox_fit_scale():
    scale = _scale_to_bbox(mesh_bbox=[1.0, 2.0, 4.0], request_bbox=[1.0, 1.0, 1.0])

    assert scale == 0.25


def test_mesh_asset_request_scale_is_uniform_factor_not_size():
    request = {"scale": 0.25, "bbox": [1.0, 2.0, 3.0]}

    assert request_size(request) == [1.0, 2.0, 3.0]
    assert _uniform_scale_factor(request["scale"]) == 0.25


def test_planner_update_mesh_asset_metadata_action_reuses_ready_geometry(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    mesh_path = case_dir / "assets" / "asset.obj"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text(
        "v 0 0 0\nv 2 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    manifest_path = case_dir / "assets" / "asset_manifest.json"
    original_request = {
        "name": "soft_asset",
        "asset_type": "generated_mesh",
        "purpose": "one closed soft asset",
        "scale": 1.0,
        "bbox": [2.0, 1.0, 1.0],
        "texture_needs": None,
        "simulation_role": "test role",
    }
    manifest_path.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "logical_name": "soft_asset",
                        "source_type": "generated_mesh",
                        "runtime_path": str(mesh_path),
                        "visual_path": None,
                        "scale": 1.0,
                        "bbox": [2.0, 1.0, 0.0],
                        "file_meshes_are_zup": True,
                        "texture_path": None,
                        "validation": {"manifold": {"ok": True}},
                        "asset_request": original_request,
                        "simulation_role": "test role",
                        "status": "ready",
                        "notes": [],
                    }
                ],
                "assumptions": [],
                "unresolved_risks": [],
            }
        ),
        encoding="utf-8",
    )

    def fake_validation(entry):
        return MeshGenesisFEMImportResult(
            ok=True,
            runtime_path=Path(entry["runtime_path"]),
            visual_path=None,
            scale=(0.5, 0.5, 0.5),
            file_meshes_are_zup=True,
            tet_resolution=2,
        )

    monkeypatch.setattr(mesh_episode, "run_genesis_fem_import_validation", fake_validation)
    updated_request = {**original_request, "scale": 0.5, "bbox": [1.0, 0.5, 0.5]}
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="case",
            task="task",
            case_dir=case_dir,
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )
    action = _planner_action(
        action="update_mesh_asset_metadata",
        planner_output=_planner_output_with_asset(updated_request),
        asset_names=["soft_asset"],
    )

    result = session.actions.execute(action, turn=0)

    assert result["ok"] is True
    assert result["status"] == "mesh_asset_metadata_updated"
    assert result["metadata_updated_asset_names"] == ["soft_asset"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["assets"][0]
    assert entry["runtime_path"] == str(mesh_path)
    assert entry["scale"] == 0.5
    assert entry["status"] == "ready"
    assert entry["asset_request"] == updated_request


def _planner_output_with_asset(asset_request: dict[str, object]) -> dict[str, object]:
    return {
        "scene_brief": {
            "user_intent": "test",
            "required_entities": ["soft_asset"],
            "interaction_goal": "test",
            "success_criteria": ["test"],
            "failure_criteria": ["fail"],
            "assumptions": [],
        },
        "scene_plan": {
            "simulation_strategy": "test",
            "physics_risks": [],
            "resource_level": "low",
            "rendering_needs": [],
        },
        "asset_requests": [asset_request],
        "module_contracts": [
            {
                "owner_role": role,
                "target_files": [f"src/{role}.py"],
                "allowed_write_paths": [f"src/{role}.py"],
                "required_exports": [],
                "input_dependencies": [],
                "asset_dependencies": ["soft_asset"] if role == "body" else [],
                "forbidden_edits": [],
                "validation_expectation": None,
                "final_report_schema": "none",
            }
            for role in ("scene", "body", "action", "rendering")
        ],
        "dispatch_graph": {
            "nodes": ["scene", "body", "action", "rendering"],
            "edges": [],
            "parallel_groups": [["scene", "body", "action", "rendering"]],
            "wait_for_asset_manifest": True,
        },
        "execution_plan": {
            "mode": "local_gpu",
            "backend": "gpu",
            "duration_sec": 1.0,
            "step_budget": 1,
            "render_fps": 1,
            "render_budget": 1,
            "notes": [],
        },
        "risk_register": [],
    }


def _planner_action(
    *,
    action: str,
    planner_output: dict[str, object] | None = None,
    asset_names: list[str] | None = None,
) -> dict[str, object]:
    return {
        "action": action,
        "rationale": "test",
        "planner_output": planner_output,
        "asset_names": asset_names,
        "roles": None,
        "owner": None,
        "repair_brief": None,
        "backend": None,
        "render": None,
        "timeout_sec": None,
        "cwd": None,
        "python_args": None,
        "pytest_args": None,
        "verdict": None,
        "summary": None,
        "notes": [],
    }


def _load_generated_main(tmp_path: Path, monkeypatch, *, body_source: str | None = None):
    run_dir = tmp_path / "case"
    src_dir = run_dir / "src"
    write_main(
        run_dir=run_dir,
        task="task",
        default_steps=1,
        default_render_fps=1,
        default_duration_sec=1.0,
        default_target_video_frames=1,
        deformable_cfg={"ipc_contact_d_hat_adaptive": True},
    )
    (src_dir / "action.py").write_text("def run_actions(*args, **kwargs):\n    return None\n", encoding="utf-8")
    (src_dir / "body.py").write_text(
        body_source or "def create_bodies(*args, **kwargs):\n    return {}\n",
        encoding="utf-8",
    )
    (src_dir / "rendering.py").write_text(
        "def setup_rendering(*args, **kwargs):\n    return None\n"
        "def finalize_rendering(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (src_dir / "scene.py").write_text("def create_scene(*args, **kwargs):\n    return None\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(src_dir))
    spec = importlib.util.spec_from_file_location("generated_main_for_test", src_dir / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return run_dir, module


def test_adaptive_d_hat_uses_genesis_frame_before_scale(tmp_path, monkeypatch):
    run_dir, module = _load_generated_main(tmp_path, monkeypatch)
    mesh_path = run_dir / "assets" / "triangle.obj"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 2 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    manifest = {
        "assets": [
            {
                "logical_name": "yup_triangle",
                "source_type": "input_mesh",
                "runtime_path": str(mesh_path),
                "scale": [1.0, 2.0, 3.0],
                "bbox": None,
                "file_meshes_are_zup": False,
                "simulation_role": "test mesh",
                "status": "ready",
            }
        ]
    }
    manifest_path = run_dir / "assets" / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = module._adaptive_contact_d_hat_report()

    assert report["selected_asset"] == "yup_triangle"
    assert report["selected_source_kind"] == "mesh_edges"
    assert math.isclose(report["bbox_diag"], math.sqrt(37.0), rel_tol=1e-9)
    assert math.isclose(report["median_feature_length"], 6.0, rel_tol=1e-9)
    assert math.isclose(report["global_bbox_diag"], math.sqrt(37.0), rel_tol=1e-9)
    assert math.isclose(report["selected_bbox_diag"], math.sqrt(37.0), rel_tol=1e-9)
    assert math.isclose(report["ipc_contact_d_hat"], 2e-3 * math.sqrt(37.0), rel_tol=1e-9)


def test_adaptive_d_hat_includes_bbox_only_primitive_asset(tmp_path, monkeypatch):
    run_dir, module = _load_generated_main(tmp_path, monkeypatch)
    assets_dir = run_dir / "assets"
    assets_dir.mkdir(parents=True)
    manifest = {
        "assets": [
            {
                "logical_name": "small_box_primitive",
                "source_type": "primitive",
                "runtime_path": None,
                "scale": None,
                "bbox": [0.2, 0.1, 0.1],
                "file_meshes_are_zup": None,
                "simulation_role": "bbox-only primitive collision body",
                "status": "ready",
            }
        ]
    }
    (assets_dir / "asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = module._adaptive_contact_d_hat_report()

    expected_diag = math.sqrt(0.2**2 + 0.1**2 + 0.1**2)
    assert report["selected_asset"] == "small_box_primitive"
    assert report["selected_source_kind"] == "bbox_fallback"
    assert report["edge_count"] == 0
    assert math.isclose(report["median_feature_length"], 0.1, rel_tol=1e-9)
    assert math.isclose(report["global_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["selected_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["ipc_contact_d_hat"], 2e-3 * expected_diag, rel_tol=1e-9)


def test_adaptive_d_hat_uses_planner_asset_request_bbox_fallback(tmp_path, monkeypatch):
    run_dir, module = _load_generated_main(tmp_path, monkeypatch)
    contracts_dir = run_dir / "contracts"
    contracts_dir.mkdir(parents=True)
    planner_output = {
        "asset_requests": [
            {
                "name": "planned_primitive",
                "asset_type": "primitive_box",
                "scale": None,
                "bbox": [0.3, 0.2, 0.1],
                "simulation_role": "planned primitive with no mesh file",
            }
        ]
    }
    (contracts_dir / "planner_output.json").write_text(json.dumps(planner_output), encoding="utf-8")

    report = module._adaptive_contact_d_hat_report()

    expected_diag = math.sqrt(0.3**2 + 0.2**2 + 0.1**2)
    assert report["source"] == "contracts/planner_output.json asset_requests"
    assert report["selected_asset"] == "planned_primitive"
    assert report["selected_source_kind"] == "bbox_fallback"
    assert math.isclose(report["global_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["ipc_contact_d_hat"], 2e-3 * expected_diag, rel_tol=1e-9)


def test_adaptive_d_hat_includes_direct_body_primitives(tmp_path, monkeypatch):
    body_source = """
TUBE_LENGTH = 2.4
TUBE_RADIUS = 0.22


def create_bodies(scene, task, *, deformable_cfg):
    scene.add_entity(
        morph=gs.morphs.Cylinder(height=TUBE_LENGTH, radius=TUBE_RADIUS, tet_resolution=2),
        material=None,
    )
    return {}
"""
    _run_dir, module = _load_generated_main(tmp_path, monkeypatch, body_source=body_source)

    report = module._adaptive_contact_d_hat_report()

    expected_diag = math.sqrt(0.44**2 + 0.44**2 + 2.4**2)
    assert report["selected_asset"] == "src/body.py:Cylinder"
    assert report["selected_source_kind"] == "direct_primitive_morph"
    assert math.isclose(report["median_feature_length"], 0.44 / 5.0, rel_tol=1e-9)
    assert math.isclose(report["global_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["selected_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["ipc_contact_d_hat"], 2e-3 * expected_diag, rel_tol=1e-9)


def test_adaptive_d_hat_includes_mjcf_collision_primitive_geoms(tmp_path, monkeypatch):
    run_dir, module = _load_generated_main(tmp_path, monkeypatch)
    assets_dir = run_dir / "assets"
    assets_dir.mkdir(parents=True)
    xml_path = assets_dir / "collar.xml"
    xml_path.write_text(
        """
<mujoco>
  <worldbody>
    <body name="root">
      <geom name="visual_skip" type="box" size="1 1 1" contype="0" conaffinity="0"/>
      <geom name="mount_dummy_collision" type="box" size="0.01 0.01 0.01" contype="1" conaffinity="1"/>
      <geom name="jaw_collision" type="box" size="0.085 0.025 0.060" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    manifest = {
        "assets": [
            {
                "logical_name": "collar_asset",
                "source_type": "mjcf",
                "runtime_path": str(xml_path),
                "scale": None,
                "bbox": [2.0, 1.0, 1.0],
                "file_meshes_are_zup": None,
                "simulation_role": "test collar",
                "status": "ready",
            }
        ]
    }
    (assets_dir / "asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = module._adaptive_contact_d_hat_report()

    expected_geom_diag = math.sqrt(0.17**2 + 0.05**2 + 0.12**2)
    expected_global_diag = math.sqrt(2.0**2 + 1.0**2 + 1.0**2)
    assert report["selected_asset"] == "collar_asset/geom:jaw_collision"
    assert report["selected_source_kind"] == "mjcf_primitive_geom"
    assert math.isclose(report["median_feature_length"], 0.05, rel_tol=1e-9)
    assert math.isclose(report["selected_bbox_diag"], expected_geom_diag, rel_tol=1e-9)
    assert math.isclose(report["global_bbox_diag"], expected_global_diag, rel_tol=1e-9)
    assert math.isclose(report["ipc_contact_d_hat"], 2e-3 * expected_global_diag, rel_tol=1e-9)
