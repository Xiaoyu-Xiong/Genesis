from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code_agent.utils.execution import GENESIS_EXECUTION_LOCK_PATH_ENV, _resolve_genesis_execution_lock_path
from code_agent.utils.suite import Case

import baselines.end_to_end_codex.runner as runner
from baselines.end_to_end_codex.case_tools import apply_adaptive_ipc_d_hat, prepare_contracts
from baselines.end_to_end_codex.configs import deformable_config_from_planner_output
from baselines.end_to_end_codex.prompt import build_end_to_end_prompt
from baselines.end_to_end_codex.runner import resolve_case_parallelism, suite_summary, validate_planner_output


def test_baseline_does_not_directly_import_main_configs_or_prompts() -> None:
    baseline_root = ROOT / "baselines" / "end_to_end_codex"
    forbidden = ("code_agent.configs", "code_agent.prompts", "prompts_legacy", "code_agent.utils.timing")
    offenders = []
    for path in sorted(baseline_root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.name}: {token}")

    assert offenders == []


def test_end_to_end_prompt_exposes_single_agent_and_meshy_contract(tmp_path: Path) -> None:
    prompt = build_end_to_end_prompt(
        case_id="demo",
        task="Create a red ball rolling into a blue block.",
        case_dir=tmp_path / "demo",
        backend="gpu",
        render=True,
        steps=32,
        duration_sec=None,
        render_fps=25,
        genesis_context="context path",
    )

    assert "only Codex code-generation agent" in prompt
    assert "src/main.py" in prompt
    assert "contracts/planner_output.json" in prompt
    assert 'asset_type: "generated_mesh"' in prompt
    assert "assets/asset_manifest.json" in prompt
    assert "case_tools run-simulation" in prompt
    assert "shared Genesis execution lock" in prompt
    assert "latest" in prompt
    assert "locked `run-simulation` report has `ok=true`" in prompt
    assert "A failed, timed-out, skipped, or artifact-missing simulation is not an acceptable final state" in prompt
    assert "rigid ABD state retrieval/accessor errors" in prompt
    assert "Preserve the intended IPC contact/coupling model" in prompt
    assert "clean no-penetration repro" in prompt
    assert "Do not run direct `python src/main.py`" in prompt


def _minimal_planner_output() -> dict:
    return {
        "scene_brief": {
            "user_intent": "Roll a red ball into a blue block.",
            "required_entities": ["red ball", "blue block", "floor"],
            "interaction_goal": "The ball collides with the block.",
            "success_criteria": ["ball contacts block"],
            "failure_criteria": ["no visible collision"],
            "assumptions": [],
        },
        "physics_plan": {
            "mode": "rigid",
            "deformable_enabled": False,
            "deformable_kind": "none",
            "ipc_enabled": False,
            "rationale": "Rigid primitives are sufficient.",
        },
        "scene_plan": {
            "simulation_strategy": "Use primitive rigid bodies on a plane.",
            "physics_risks": [],
            "resource_level": "low",
            "rendering_needs": ["camera sees the contact"],
        },
        "asset_requests": [],
        "module_contracts": [
            {
                "owner_role": "end_to_end_codex",
                "target_files": ["src/main.py"],
                "allowed_write_paths": ["src/main.py"],
                "required_exports": ["main"],
                "input_dependencies": ["inputs/user_prompt.md"],
                "asset_dependencies": [],
                "forbidden_edits": ["repository source"],
                "validation_expectation": "Harness executes src/main.py.",
                "final_report_schema": "code_agent/specs/worker_report.schema.json",
            }
        ],
        "dispatch_graph": {
            "nodes": ["end_to_end_codex"],
            "edges": [],
            "parallel_groups": [["end_to_end_codex"]],
            "wait_for_asset_manifest": False,
        },
        "execution_plan": {
            "mode": "local_gpu",
            "backend": "gpu",
            "duration_sec": 1.0,
            "step_budget": 100,
            "sim_dt": 0.01,
            "sim_substeps": 10,
            "render_every_n_steps": 4,
            "render_fps": 25,
            "render_budget": 25,
            "render_res": [640, 480],
            "notes": [],
        },
        "risk_register": [],
    }


def test_validate_planner_output_accepts_minimal_baseline_plan() -> None:
    planner_output = _minimal_planner_output()

    assert validate_planner_output(planner_output) == []


def test_deformable_config_keeps_cloth_as_agent_decision() -> None:
    planner_output = _minimal_planner_output()
    planner_output["physics_plan"]["mode"] = "fem_ipc"
    planner_output["physics_plan"]["deformable_kind"] = "cloth"

    deformable_config = deformable_config_from_planner_output(planner_output)

    assert deformable_config["enabled"] is True
    assert deformable_config["ipc_enabled"] is True
    assert deformable_config["deformable_kind"] == "cloth"
    assert "fem_cloth_enabled" not in deformable_config


def test_adaptive_ipc_updates_d_hat_from_planner_asset_bbox(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    contracts_dir = case_dir / "contracts"
    contracts_dir.mkdir(parents=True)
    planner_output = _minimal_planner_output()
    planner_output["physics_plan"]["mode"] = "fem_ipc"
    planner_output["physics_plan"]["deformable_enabled"] = True
    planner_output["physics_plan"]["deformable_kind"] = "cloth"
    planner_output["physics_plan"]["ipc_enabled"] = True
    planner_output["asset_requests"] = [
        {
            "name": "cloth_panel",
            "asset_type": "generated_mesh",
            "purpose": "Thin deformable cloth panel.",
            "scale": None,
            "bbox": [0.2, 0.1, 0.005],
            "cloth_target_edge_length": 0.01,
            "texture_needs": None,
            "simulation_role": "deformable",
        }
    ]
    (contracts_dir / "planner_output.json").write_text(json.dumps(planner_output), encoding="utf-8")

    prepare_report = prepare_contracts(case_dir=case_dir)
    adaptive_report = apply_adaptive_ipc_d_hat(case_dir=case_dir)

    deformable_config = json.loads((contracts_dir / "deformable_config.json").read_text(encoding="utf-8"))
    assert prepare_report["ok"] is True
    assert adaptive_report["ok"] is True
    assert adaptive_report["status"] == "adaptive_applied"
    assert deformable_config["ipc_contact_d_hat"] < 0.01
    assert adaptive_report["adaptive_report"]["source"] == "contracts/planner_output.json asset_requests"
    assert adaptive_report["adaptive_report"]["selected_asset"] == "cloth_panel"


def test_execution_lock_path_can_come_from_environment(tmp_path: Path, monkeypatch) -> None:
    lock_path = tmp_path / "suite" / ".locks" / "genesis_execution.lock"
    monkeypatch.setenv(GENESIS_EXECUTION_LOCK_PATH_ENV, str(lock_path))

    assert _resolve_genesis_execution_lock_path() == lock_path.resolve()


def test_baseline_codex_request_uses_shared_execution_lock(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_codex_exec(request):
        captured["request"] = request
        return runner.CodexExecResult(
            role=request.role,
            success=True,
            exit_code=0,
            duration_sec=0.0,
            command=[],
            cwd=str(request.cwd),
            sandbox=request.sandbox,
            output_jsonl_path=str(request.output_jsonl_path),
            final_message_path=str(request.final_message_path),
            output_schema_path=str(request.output_schema_path) if request.output_schema_path else None,
            codex_version=None,
        )

    monkeypatch.setattr(runner, "run_codex_exec", fake_run_codex_exec)
    suite_dir = tmp_path / "suite"
    case_dir = suite_dir / "case_a"
    (case_dir / "logs").mkdir(parents=True)
    lock_path = suite_dir / ".locks" / "genesis_execution.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.touch()

    runner._run_end_to_end_codex(
        case=Case(case_id="case_a", task="A ball hits a block."),
        case_dir=case_dir,
        suite_config=runner.EndToEndBaselineConfig(tasks_file=suite_dir / "tasks.txt", out_dir=suite_dir),
        execution_lock_path=lock_path,
    )

    request = captured["request"]
    assert (GENESIS_EXECUTION_LOCK_PATH_ENV, str(lock_path)) in request.env_overrides
    assert lock_path in request.writable_roots


def test_prepare_contracts_writes_timing_and_deformable_config(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    contracts_dir = case_dir / "contracts"
    contracts_dir.mkdir(parents=True)
    (contracts_dir / "planner_output.json").write_text(
        json.dumps(_minimal_planner_output()),
        encoding="utf-8",
    )

    report = prepare_contracts(case_dir=case_dir)

    assert report["ok"] is True
    assert report["status"] == "contracts_ready"
    assert (contracts_dir / "timing.json").is_file()
    assert (contracts_dir / "deformable_config.json").is_file()
    deformable_config = json.loads((contracts_dir / "deformable_config.json").read_text(encoding="utf-8"))
    assert deformable_config["deformable_kind"] == "none"
    assert "fem_cloth_enabled" not in deformable_config
    assert (case_dir / "reports" / "baseline_agent_tool_history.jsonl").is_file()


def test_partial_worker_report_does_not_block_final_execution() -> None:
    assert runner._ready_for_execution(
        codex_result=runner.CodexExecResult(
            role="end_to_end_codex",
            success=True,
            exit_code=0,
            duration_sec=0.0,
            command=[],
            cwd=str(ROOT),
            sandbox="workspace-write",
            output_jsonl_path="logs/codex.jsonl",
            final_message_path="logs/codex.final.json",
            output_schema_path="code_agent/specs/worker_report.schema.json",
            codex_version=None,
        ),
        worker_report={"status": "partial"},
        worker_error=None,
        timing=runner.BaselineTimingPlan(
            steps=10,
            duration_sec=0.1,
            render_fps=10,
            sim_dt=0.01,
            sim_substeps=1,
            render_every_n_steps=1,
            render_res=(64, 64),
            target_video_frames=1,
            source="test",
        ),
        planner_errors=[],
        asset_errors=[],
        adaptive_ipc_errors=[],
        source_violations=[],
    )


def test_baseline_parallelism_and_summary_counts() -> None:
    assert resolve_case_parallelism(num_cases=5, max_parallel_cases=None) == 5
    assert resolve_case_parallelism(num_cases=5, max_parallel_cases=2) == 2
    assert resolve_case_parallelism(num_cases=0, max_parallel_cases=2) == 1

    summary = suite_summary(
        {"baseline": "end_to_end_codex"},
        results=[
            {"case_id": "a", "verdict": "pass"},
            {"case_id": "b", "verdict": "fail", "retry_recommended": True},
        ],
        num_cases_total=3,
    )

    assert summary["num_cases"] == 3
    assert summary["num_completed"] == 2
    assert summary["num_passed"] == 1
    assert summary["num_failed"] == 1
    assert summary["retry_candidates"] == ["b"]
