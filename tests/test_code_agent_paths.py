from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh

from code_agent.assets.mesh import episode as mesh_episode
from code_agent.assets.mesh.manifest import _scale_to_bbox, _uniform_scale_factor
from code_agent.assets.mesh.models import (
    MeshGenesisClothImportResult,
    MeshGenesisFEMImportResult,
    MeshManifoldCheckResult,
    MeshRepairResult,
    MeshyGenerationResult,
    MeshyRequestError,
    TextToMeshBundle,
)
from code_agent.assets.mesh.request_adapter import request_size
from code_agent.assets.builtin_guard import builtin_asset_violations
from code_agent.assets.xml.validation import _validate_mesh_asset_paths, validate_xml_asset
from code_agent.configs import CONFIGS, deformable_config_dict, runtime_defaults_dict
from code_agent.opt import agent as opt_agent
from code_agent.opt.contracts import OptContractError, load_opt_contracts
from code_agent.opt.objective import evaluate_objective
from code_agent.opt.optimizers.cma_es import CMAESOptimizer
from code_agent.opt.types import OptAgentRequest
from code_agent.planner.session import PlannerSession, PlannerSessionConfig
from code_agent.utils import codex
from code_agent.utils.integrator import write_main
from code_agent.utils.timing import resolve_timing


def _codex_test_result(
    request: codex.CodexExecRequest,
    *,
    success: bool,
    error_type: str | None = None,
    error_message: str | None = None,
) -> codex.CodexExecResult:
    now = time.time()
    return codex.CodexExecResult(
        role=request.role,
        success=success,
        exit_code=0 if success else 1,
        duration_sec=0.01,
        command=["codex"],
        cwd=str(request.cwd),
        sandbox=request.sandbox,
        output_jsonl_path=str(request.output_jsonl_path),
        final_message_path=str(request.final_message_path),
        output_schema_path=str(request.output_schema_path) if request.output_schema_path else None,
        codex_version="codex-test",
        error_type=error_type,
        error_message=error_message,
        stderr_path=str(request.output_jsonl_path) + ".stderr",
        codex_account_name=request.codex_account_name,
        started_at_unix=now,
        ended_at_unix=now + 0.01,
    )


def _write_codex_auth_for_test(codex_home: Path, *, age_days: float) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "last_refresh": time.time() - age_days * 24 * 60 * 60,
                "tokens": {
                    "access_token": "test",
                    "refresh_token": "test",
                },
            }
        ),
        encoding="utf-8",
    )


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


def test_run_codex_exec_rotates_accounts_on_usage_limit(tmp_path, monkeypatch):
    codex._reset_codex_account_state_for_tests()
    monkeypatch.setenv(
        "CODE_AGENT_CODEX_ACCOUNTS",
        f"first={tmp_path / 'codex-first'};second={tmp_path / 'codex-second'}",
    )
    calls: list[str | None] = []

    def fake_run_once(request: codex.CodexExecRequest) -> codex.CodexExecResult:
        calls.append(request.codex_account_name)
        if request.codex_account_name == "first":
            return _codex_test_result(
                request,
                success=False,
                error_type="codex_usage_limit",
                error_message="usage limit; try again later",
            )
        return _codex_test_result(request, success=True)

    monkeypatch.setattr(codex, "_run_codex_exec_request_once", fake_run_once)

    result = codex.run_codex_exec(
        codex.CodexExecRequest(
            role="planner",
            prompt="test",
            cwd=tmp_path,
            output_jsonl_path=tmp_path / "logs" / "planner.jsonl",
            final_message_path=tmp_path / "logs" / "planner.final.txt",
        )
    )

    assert result.success is True
    assert result.codex_account_name == "second"
    assert calls == ["first", "second"]
    quota_log = tmp_path / "logs" / "planner.jsonl.quota.jsonl"
    assert "account_usage_limited" in quota_log.read_text(encoding="utf-8")


def test_run_codex_exec_retries_transient_model_capacity(tmp_path, monkeypatch):
    codex._reset_codex_account_state_for_tests()
    monkeypatch.setenv("CODE_AGENT_CODEX_CAPACITY_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("CODE_AGENT_CODEX_CAPACITY_RETRY_DELAY_SEC", "0")
    calls = 0

    def fake_run_once(request: codex.CodexExecRequest) -> codex.CodexExecResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _codex_test_result(
                request,
                success=False,
                error_type="codex_capacity",
                error_message="Selected model is at capacity. Please try a different model.",
            )
        return _codex_test_result(request, success=True)

    monkeypatch.setattr(codex, "_run_codex_exec_request_once", fake_run_once)

    result = codex.run_codex_exec(
        codex.CodexExecRequest(
            role="planner",
            prompt="test",
            cwd=tmp_path,
            output_jsonl_path=tmp_path / "logs" / "planner.jsonl",
            final_message_path=tmp_path / "logs" / "planner.final.txt",
        )
    )

    assert result.success is True
    assert calls == 2
    retry_log = tmp_path / "logs" / "planner.jsonl.quota.jsonl"
    assert "capacity_retry_scheduled" in retry_log.read_text(encoding="utf-8")


def test_codex_failure_classification_detects_model_capacity(tmp_path):
    jsonl_path = tmp_path / "codex.jsonl"
    stderr_path = tmp_path / "codex.jsonl.stderr"
    jsonl_path.write_text(
        json.dumps({"type": "error", "message": "Selected model is at capacity. Please try a different model."}) + "\n",
        encoding="utf-8",
    )
    stderr_path.write_text("", encoding="utf-8")

    error_type, message = codex._classify_codex_failure(jsonl_path=jsonl_path, stderr_path=stderr_path)

    assert error_type == "codex_capacity"
    assert message is not None
    assert "capacity" in message


def test_codex_failure_classification_detects_nested_bwrap_sandbox_failure(tmp_path):
    jsonl_path = tmp_path / "codex.jsonl"
    stderr_path = tmp_path / "codex.jsonl.stderr"
    jsonl_path.write_text(json.dumps({"type": "turn.completed"}) + "\n", encoding="utf-8")
    stderr_path.write_text(
        "apply_patch verification failed: fs sandbox helper failed: "
        "bwrap: Can't mkdir /repo/.agents: Read-only file system\n",
        encoding="utf-8",
    )

    error_type, message = codex._classify_codex_failure(jsonl_path=jsonl_path, stderr_path=stderr_path)

    assert error_type == "codex_sandbox_failed"
    assert message is not None
    assert ".agents" in message


def test_run_codex_exec_treats_exit_zero_sandbox_failure_as_error(tmp_path):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    fake_codex = tmp_path / "fake-codex"
    fake_codex.write_text(
        "#!/bin/sh\n"
        'echo "fs sandbox helper failed: bwrap: Can\'t mkdir $PWD/.agents: Read-only file system" >&2\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    codex_home = tmp_path / "codex-home"
    _write_codex_auth_for_test(codex_home, age_days=0)

    result = codex._run_codex_exec_request_once(
        codex.CodexExecRequest(
            role="worker",
            prompt="test",
            cwd=fake_repo,
            output_jsonl_path=fake_repo / "events.jsonl",
            final_message_path=fake_repo / "final.txt",
            codex_bin=str(fake_codex),
            codex_home=codex_home,
            hide_builtin_assets=False,
        )
    )

    assert result.exit_code == 0
    assert result.success is False
    assert result.error_type == "codex_sandbox_failed"
    assert "codex_sandbox_failed" in (fake_repo / "final.txt").read_text(encoding="utf-8")


def test_run_codex_exec_waits_until_any_account_quota_recovers(tmp_path, monkeypatch):
    codex._reset_codex_account_state_for_tests()
    monkeypatch.setenv(
        "CODE_AGENT_CODEX_ACCOUNTS",
        f"first={tmp_path / 'codex-first'};second={tmp_path / 'codex-second'}",
    )
    monkeypatch.setenv("CODE_AGENT_CODEX_QUOTA_PROBE_INITIAL_DELAY_SEC", "0")
    monkeypatch.setenv("CODE_AGENT_CODEX_QUOTA_PROBE_INTERVAL_SEC", "0")
    recovered_accounts: set[str] = set()
    calls: list[tuple[str, str | None]] = []

    def fake_run_once(request: codex.CodexExecRequest) -> codex.CodexExecResult:
        calls.append((request.role, request.codex_account_name))
        account = str(request.codex_account_name)
        if request.role.startswith("quota_probe_"):
            if account == "second":
                recovered_accounts.add(account)
                return _codex_test_result(request, success=True)
            return _codex_test_result(
                request,
                success=False,
                error_type="codex_usage_limit",
                error_message="usage limit; try again later",
            )
        if account in recovered_accounts:
            return _codex_test_result(request, success=True)
        return _codex_test_result(
            request,
            success=False,
            error_type="codex_usage_limit",
            error_message="usage limit; try again later",
        )

    monkeypatch.setattr(codex, "_run_codex_exec_request_once", fake_run_once)

    result = codex.run_codex_exec(
        codex.CodexExecRequest(
            role="worker",
            prompt="test",
            cwd=tmp_path,
            output_jsonl_path=tmp_path / "logs" / "worker.jsonl",
            final_message_path=tmp_path / "logs" / "worker.final.txt",
        )
    )

    assert result.success is True
    assert result.codex_account_name == "second"
    assert ("quota_probe_first", "first") in calls
    assert ("quota_probe_second", "second") in calls
    assert calls[-1] == ("worker", "second")
    quota_log = tmp_path / "logs" / "worker.jsonl.quota.jsonl"
    text = quota_log.read_text(encoding="utf-8")
    assert "quota_pause_started" in text
    assert "quota_recovered" in text


def test_codex_failure_classification_uses_diagnostic_jsonl_message(tmp_path):
    jsonl_path = tmp_path / "codex.jsonl"
    stderr_path = tmp_path / "codex.jsonl.stderr"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread"}),
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "error",
                        "message": (
                            "Your access token could not be refreshed because your refresh token was revoked. "
                            "Please log out and sign in again."
                        ),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stderr_path.write_text(
        "failed to refresh token: 401 Unauthorized: refresh_token_invalidated\n",
        encoding="utf-8",
    )

    error_type, message = codex._classify_codex_failure(jsonl_path=jsonl_path, stderr_path=stderr_path)

    assert error_type == "codex_auth_failed"
    assert message is not None
    assert "refresh token was revoked" in message
    assert "thread.started" not in message


def test_run_codex_exec_aborts_quota_wait_when_probes_are_auth_failed(tmp_path, monkeypatch):
    codex._reset_codex_account_state_for_tests()
    monkeypatch.setenv(
        "CODE_AGENT_CODEX_ACCOUNTS",
        f"first={tmp_path / 'codex-first'};second={tmp_path / 'codex-second'}",
    )
    monkeypatch.setenv("CODE_AGENT_CODEX_QUOTA_PROBE_INITIAL_DELAY_SEC", "0")
    monkeypatch.setenv("CODE_AGENT_CODEX_QUOTA_PROBE_INTERVAL_SEC", "0")
    calls: list[tuple[str, str | None]] = []

    def fake_run_once(request: codex.CodexExecRequest) -> codex.CodexExecResult:
        calls.append((request.role, request.codex_account_name))
        if request.role.startswith("quota_probe_"):
            return _codex_test_result(
                request,
                success=False,
                error_type="codex_auth_failed",
                error_message="refresh token was revoked",
            )
        if calls.count((request.role, request.codex_account_name)) == 1:
            return _codex_test_result(
                request,
                success=False,
                error_type="codex_usage_limit",
                error_message="usage limit; try again later",
            )
        return _codex_test_result(
            request,
            success=False,
            error_type="codex_auth_failed",
            error_message="refresh token was revoked",
        )

    monkeypatch.setattr(codex, "_run_codex_exec_request_once", fake_run_once)

    result = codex.run_codex_exec(
        codex.CodexExecRequest(
            role="planner",
            prompt="test",
            cwd=tmp_path,
            output_jsonl_path=tmp_path / "logs" / "planner.jsonl",
            final_message_path=tmp_path / "logs" / "planner.final.txt",
        )
    )

    assert result.success is False
    assert result.error_type == "codex_auth_failed"
    assert len(calls) < 8
    quota_log = tmp_path / "logs" / "planner.jsonl.quota.jsonl"
    assert "quota_pause_aborted" in quota_log.read_text(encoding="utf-8")


def test_run_codex_exec_fails_before_subprocess_when_login_is_stale(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-stale"
    _write_codex_auth_for_test(codex_home, age_days=8)
    monkeypatch.setenv("CODE_AGENT_CODEX_AUTH_MAX_AGE_DAYS", "7")

    result = codex._run_codex_exec_request_once(
        codex.CodexExecRequest(
            role="planner",
            prompt="test",
            cwd=tmp_path,
            output_jsonl_path=tmp_path / "logs" / "planner.jsonl",
            final_message_path=tmp_path / "logs" / "planner.final.txt",
            codex_bin="/bin/true",
            codex_home=codex_home,
            codex_account_name="stale",
            hide_builtin_assets=False,
        )
    )

    assert result.success is False
    assert result.error_type == "codex_auth_failed"
    assert result.error_message is not None
    assert "older than 7 days" in result.error_message
    assert f"CODEX_HOME={codex_home.resolve()} codex login" in result.error_message
    assert "account_auth_stale" in (tmp_path / "logs" / "planner.jsonl.quota.jsonl").read_text(encoding="utf-8")


def test_ensure_configured_codex_accounts_fresh_rejects_stale_accounts(tmp_path, monkeypatch):
    stale_home = tmp_path / "codex-stale"
    fresh_home = tmp_path / "codex-fresh"
    _write_codex_auth_for_test(stale_home, age_days=8)
    _write_codex_auth_for_test(fresh_home, age_days=1)
    monkeypatch.setenv("CODE_AGENT_CODEX_AUTH_MAX_AGE_DAYS", "7")
    monkeypatch.setenv("CODE_AGENT_CODEX_ACCOUNTS", f"stale={stale_home};fresh={fresh_home}")

    try:
        codex.ensure_configured_codex_accounts_fresh()
    except codex.CodexAuthFreshnessError as exc:
        message = str(exc)
    else:
        raise AssertionError("stale Codex auth should block task startup")

    assert "stale" in message
    assert "fresh" not in message
    assert f"CODEX_HOME={stale_home.resolve()} codex login" in message


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


def test_codex_asset_sandbox_prepares_project_agents_mountpoint(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "nested"
    (repo / ".git").mkdir(parents=True)
    nested.mkdir()

    skills_root = codex._ensure_codex_project_sandbox_mountpoints(nested)

    assert skills_root == repo / ".agents" / "skills"
    assert skills_root.is_dir()


def test_codex_command_places_top_level_args_before_exec(tmp_path):
    command = codex.build_codex_exec_command(
        codex.CodexExecRequest(
            role="test",
            prompt="test",
            cwd=Path("."),
            output_jsonl_path=tmp_path / "events.jsonl",
            final_message_path=tmp_path / "final.json",
            codex_top_level_args=("--search",),
            hide_builtin_assets=False,
        ),
        resolved_codex="/usr/bin/codex",
    )

    assert command[:3] == ["/usr/bin/codex", "--search", "exec"]


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

    policy_only = builtin_asset_violations(
        {
            "forbidden_edits": [
                "Do not inspect, copy, import, or reference prepackaged assets under genesis/assets.",
                "Never use gs.utils.get_assets_dir() for generated task geometry.",
            ],
            "failure_criteria": [
                "The robot is implemented using packaged Genesis assets, external downloads, or genesis/assets paths."
            ],
            "success_criteria": [
                "Generated code does not inspect, import, copy, or reference assets under genesis/assets."
            ],
        },
        label="planner_output",
    )
    assert policy_only == []


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
    assert result.best == payload["best"]
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
            json.dumps(
                {"schema_version": 1, "success": False, "score": 2.0, "target": {}, "measured": {}, "terms": {}}
            ),
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


def test_planner_summary_marks_worker_sandbox_failure_as_retryable_infra(tmp_path):
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="case",
            task="task",
            case_dir=tmp_path / "case",
            backend="gpu",
            timeout_sec=1.0,
            render=True,
            repair_rounds=0,
        )
    )
    session.state["status"] = "inconclusive"
    session.state["workers"]["body"] = {
        "status": "failed",
        "ok": False,
        "codex": {"error_type": "codex_sandbox_failed", "error_message": "bwrap failed"},
    }

    summary = session.build_summary()

    assert summary["outcome_class"] == "infra_blocked"
    assert summary["infra_blocked_reason"] == "body_worker:codex_sandbox_failed"
    assert summary["retry_recommended"] is True


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
                "terms": [{"name": "score", "metric_path": "score", "weight": 1.0, "transform": "identity"}],
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
        optimizer.tell(samples, [-((sample[0] - 0.7) ** 2) for sample in samples], maximize=True)

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


def test_planner_schemas_accept_cloth_target_edge_length(tmp_path):
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
    planner_output = _planner_output_with_asset(
        {
            "name": "cloth_sheet",
            "asset_type": "cloth_mesh_square",
            "purpose": "square FEM cloth sheet",
            "scale": 1.0,
            "bbox": [0.4, 0.4, 0.001],
            "cloth_target_edge_length": 0.04,
            "texture_needs": None,
            "simulation_role": "dynamic FEM cloth sheet",
        }
    )
    action = _planner_action(
        action="start_mesh_assets",
        planner_output=planner_output,
        asset_names=["cloth_sheet"],
    )

    assert not session.validate_json_schema(planner_output, Path("code_agent/specs/planner_output.schema.json"))
    assert not session.validate_json_schema(action, Path("code_agent/specs/planner_action.schema.json"))


def test_codex_output_schemas_avoid_unsupported_composition_keywords():
    schema_paths = [
        Path("code_agent/specs/planner_action.schema.json"),
        Path("code_agent/specs/worker_report.schema.json"),
        Path("code_agent/specs/xml_worker_report.schema.json"),
        Path("code_agent/specs/critic_report.schema.json"),
        Path("code_agent/specs/opt_schema/opt_subagent_report.schema.json"),
        Path("code_agent/scores/physical/sbar_report.schema.json"),
        *sorted(Path("code_agent/dataset/schemas").glob("*.schema.json")),
    ]
    forbidden = {"allOf", "oneOf", "not", "if", "then", "else"}

    for schema_path in schema_paths:
        pending = [json.loads(schema_path.read_text(encoding="utf-8"))]
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                assert not forbidden.intersection(node), schema_path
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)


def test_planner_output_physics_plan_updates_deformable_contract(tmp_path):
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="case",
            task="soft cloth drapes over a rigid bar",
            case_dir=tmp_path / "case",
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )
    session._ensure_dirs()
    planner_output = _planner_output_with_asset_requests([])
    planner_output["physics_plan"] = {
        "mode": "fem_ipc",
        "deformable_enabled": True,
        "deformable_kind": "cloth",
        "ipc_enabled": False,
        "rationale": "cloth requires FEM+IPC",
    }
    planner_output["execution_plan"]["sim_dt"] = 0.005
    planner_output["execution_plan"]["render_fps"] = 20

    accepted = session.accept_planner_output(planner_output)

    assert accepted["ok"] is True
    assert session.deformable_config["enabled"] is True
    assert session.deformable_config["ipc_enabled"] is True
    written_cfg = json.loads(session.deformable_config_path.read_text(encoding="utf-8"))
    written_plan = json.loads((session.contracts_dir / "planner_output.json").read_text(encoding="utf-8"))
    assert written_cfg["enabled"] is True
    assert written_cfg["ipc_enabled"] is True
    assert written_plan["physics_plan"]["ipc_enabled"] is True


def test_resolve_timing_uses_planner_runtime_fields_and_mode_defaults():
    planner_output = _planner_output_with_asset_requests([])
    planner_output["physics_plan"] = {
        "mode": "rigid_ipc",
        "deformable_enabled": False,
        "deformable_kind": "none",
        "ipc_enabled": True,
        "rationale": "dense rigid contact",
    }
    planner_output["execution_plan"].update(
        {
            "duration_sec": 2.0,
            "sim_dt": 0.0025,
            "sim_substeps": 2,
            "render_every_n_steps": 3,
            "render_fps": 30,
            "render_res": [320, 240],
        }
    )

    timing = resolve_timing(planner_output=planner_output)

    assert timing.sim_dt == 0.0025
    assert timing.steps == 800
    assert timing.sim_substeps == 2
    assert timing.render_every_n_steps == 3
    assert timing.render_fps == 30
    assert timing.render_res == (320, 240)
    assert timing.target_video_frames == 60


def test_resolve_timing_uses_ipc_defaults_when_planner_omits_optional_runtime_values():
    planner_output = _planner_output_with_asset_requests([])
    planner_output["physics_plan"] = {
        "mode": "rigid_ipc",
        "deformable_enabled": False,
        "deformable_kind": "none",
        "ipc_enabled": True,
        "rationale": "dense rigid contact",
    }
    for key in ("sim_dt", "sim_substeps", "render_every_n_steps", "render_res", "render_fps"):
        planner_output["execution_plan"].pop(key)

    timing = resolve_timing(planner_output=planner_output)
    ipc_defaults = runtime_defaults_dict(ipc_enabled=True)

    assert timing.sim_dt == ipc_defaults["sim_dt"]
    assert timing.sim_substeps == ipc_defaults["sim_substeps"]
    assert timing.render_every_n_steps == ipc_defaults["render_every_n_steps"]
    assert timing.render_fps == ipc_defaults["render_fps"]
    assert timing.render_res == ipc_defaults["render_res"]


def test_deformable_config_does_not_override_fem_friction_mu():
    cfg = deformable_config_dict(physics_mode="fem_ipc")

    assert not hasattr(CONFIGS.deformable, "enabled")
    assert not hasattr(CONFIGS.ipc, "enabled")
    assert "fem_friction_mu" not in cfg
    assert "friction" in cfg
    assert cfg["fem_cloth_enabled"] is True
    assert cfg["cloth_thickness_default"] > 0.0
    assert cfg["cloth_grid_resolution_default"] > 0


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
        "cloth_target_edge_length": None,
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


def test_generate_procedural_cloth_mesh_assets(tmp_path):
    case_dir = tmp_path / "case"
    requests = [
        {
            "name": "cloth_square",
            "asset_type": "cloth_mesh_square",
            "purpose": "square FEM cloth sheet",
            "scale": 1.0,
            "bbox": [0.5, 0.5, 0.001],
            "cloth_target_edge_length": 0.05,
            "texture_needs": None,
            "simulation_role": "dynamic FEM cloth square",
        },
        {
            "name": "cloth_rectangle",
            "asset_type": "cloth_mesh_rectangle",
            "purpose": "rectangular FEM cloth ribbon",
            "scale": 1.0,
            "bbox": [0.8, 0.3, 0.001],
            "texture_needs": None,
            "simulation_role": "dynamic FEM cloth rectangle",
        },
        {
            "name": "cloth_disk",
            "asset_type": "cloth_mesh_disk",
            "purpose": "circular FEM cloth target membrane for a panda projectile",
            "scale": 1.0,
            "bbox": [0.6, 0.6, 0.001],
            "cloth_target_edge_length": 0.05,
            "texture_needs": None,
            "simulation_role": "dynamic open circular FEM cloth membrane",
        },
        {
            "name": "cloth_cylinder",
            "asset_type": "cloth_mesh_cylinder",
            "purpose": "cylindrical FEM cloth shell",
            "scale": 1.0,
            "bbox": [0.4, 0.4, 0.8],
            "texture_needs": None,
            "simulation_role": "dynamic FEM cloth cylinder shell",
        },
        {
            "name": "cloth_sphere",
            "asset_type": "cloth_mesh_sphere",
            "purpose": "spherical FEM cloth shell",
            "scale": 1.0,
            "bbox": [0.5, 0.5, 0.5],
            "texture_needs": None,
            "simulation_role": "dynamic FEM cloth sphere shell",
        },
    ]

    report = mesh_episode.generate_mesh_assets_for_episode(
        case_dir=case_dir,
        task="Create a FEM.Cloth smoke asset set.",
        planner_output=_planner_output_with_asset_requests(requests),
    )

    assert report["ok"] is True
    assert report["status"] == "mesh_assets_generated"
    manifest_path = case_dir / "assets" / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    assert not session.validate_json_schema(manifest, Path("code_agent/specs/asset_manifest.schema.json"))
    entries = {entry["logical_name"]: entry for entry in manifest["assets"]}
    assert set(entries) == {request["name"] for request in requests}
    for request in requests:
        entry = entries[request["name"]]
        runtime_path = Path(entry["runtime_path"])
        assert entry["source_type"] == "cloth_mesh"
        assert entry["status"] == "ready"
        assert entry["file_meshes_are_zup"] is True
        assert runtime_path.is_file()
        text = runtime_path.read_text(encoding="utf-8")
        assert "\nv " in "\n" + text
        assert "\nf " in "\n" + text
        validation = entry["validation"]["cloth_mesh"]
        assert validation["shape"] in {"square", "rectangle", "disk", "cylinder", "sphere"}
        assert validation["face_count"] <= deformable_config_dict()["cloth_max_faces"]
        if request.get("cloth_target_edge_length") is not None:
            assert validation["target_edge_length"] == request["cloth_target_edge_length"]
            assert validation["target_edge_length_source"] == "asset_request"
            if validation["shape"] == "square":
                assert validation["face_count"] == 200
            elif validation["shape"] == "disk":
                assert validation["vertex_count"] - validation["edge_count"] + validation["face_count"] == 1
                assert validation["bbox_size"] == [0.6, 0.6, 0.0]
                assert validation["face_count"] == 288


def test_procedural_cloth_mesh_rejects_complex_open_silhouette(tmp_path):
    case_dir = tmp_path / "case"
    request = {
        "name": "blue_armadillo_patch",
        "asset_type": "cloth_mesh",
        "purpose": "armadillo-shaped open FEM cloth patch silhouette",
        "scale": 1.0,
        "bbox": [0.6, 0.3, 0.001],
        "cloth_target_edge_length": 0.04,
        "texture_needs": "blue knitted fabric",
        "simulation_role": "dynamic FEM.Cloth animal-shaped thin open cloth patch",
    }

    report = mesh_episode.generate_mesh_assets_for_episode(
        case_dir=case_dir,
        task="Create an armadillo-shaped open FEM.Cloth patch.",
        planner_output=_planner_output_with_asset(request),
    )

    assert report["ok"] is False
    assert report["status"] == "cloth_mesh_unsupported_shape"
    assert report["recommended_owner"] == "planner"
    assert report["failure_classes"] == ["cloth_mesh.unsupported_shape"]
    assert "unsupported arbitrary open cloth silhouette" in report["repair_summary"]
    manifest = json.loads((case_dir / "assets" / "asset_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["assets"][0]
    assert entry["logical_name"] == "blue_armadillo_patch"
    assert entry["source_type"] == "cloth_mesh"
    assert entry["status"] == "failed"
    assert entry["runtime_path"] == "unavailable"
    assert "Unsupported procedural cloth_mesh shape" in entry["validation"]["cloth_mesh"]["error"]


def test_meshy_api_download_retries_transient_dns_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_AGENT_MESHY_API_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("CODE_AGENT_MESHY_API_RETRY_DELAY_SEC", "0")
    calls = 0

    class FakeDownloadedMeshyAsset:
        def to_dict(self):
            return {"provider": "fake"}

    def fake_download_meshy_mesh_from_text(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MeshyRequestError("Meshy submit request failed: [Errno -3] Temporary failure in name resolution")
        return FakeDownloadedMeshyAsset()

    monkeypatch.setattr(mesh_episode, "download_meshy_mesh_from_text", fake_download_meshy_mesh_from_text)

    result = mesh_episode._download_one_mesh_asset(
        index=0,
        request={
            "name": "soft_bunny",
            "asset_type": "generated_mesh",
            "purpose": "soft bunny mesh",
            "scale": None,
            "bbox": [0.2, 0.2, 0.2],
            "cloth_target_edge_length": None,
            "texture_needs": None,
            "simulation_role": "dynamic soft body",
        },
        task="Create a soft bunny.",
        output_root=tmp_path,
        api_config=mesh_episode.MeshyApiConfig(api_key="test"),
    )

    assert result["ok"] is True
    assert calls == 2


def test_meshy_generated_cloth_request_writes_cloth_manifest(tmp_path, monkeypatch):
    request = {
        "name": "ruffled_closed_cloth_shell",
        "asset_type": "generated_mesh",
        "purpose": "complex ruffled FEM.Cloth closed manifold shell surface",
        "scale": 1.0,
        "bbox": [0.5, 0.3, 0.4],
        "cloth_target_edge_length": None,
        "texture_needs": None,
        "simulation_role": "dynamic FEM.Cloth closed manifold shell",
    }
    mesh_path = tmp_path / "repaired.obj"
    trimesh.creation.icosphere(subdivisions=1, radius=0.5).export(mesh_path)
    bundle = _text_to_mesh_bundle_for_test(mesh_path)

    def fake_process_downloaded_meshy_mesh(*, downloaded, repair_config):
        _ = downloaded, repair_config
        return bundle

    def fake_cloth_validation(entry):
        return MeshGenesisClothImportResult(
            ok=True,
            runtime_path=Path(entry["runtime_path"]),
            visual_path=Path(entry["visual_path"]) if entry.get("visual_path") else None,
            scale=(1.0, 1.0, 1.0),
            file_meshes_are_zup=False,
            vertex_count=42,
            element_count=80,
            surface_vertex_count=42,
            surface_face_count=80,
        )

    def fail_fem_validation(_entry):
        raise AssertionError("Meshy-generated FEM.Cloth shells must not use volumetric FEM import validation.")

    monkeypatch.setattr(mesh_episode, "process_downloaded_meshy_mesh", fake_process_downloaded_meshy_mesh)
    monkeypatch.setattr(mesh_episode, "run_genesis_cloth_import_validation", fake_cloth_validation)
    monkeypatch.setattr(mesh_episode, "run_genesis_fem_import_validation", fail_fem_validation)

    result = mesh_episode._process_one_mesh_asset(
        {
            "ok": True,
            "request": request,
            "mesh_prompt": "Create one FEM.Cloth closed manifold shell.",
            "downloaded": object(),
        }
    )

    assert result["ok"] is True
    entry = result["manifest_entry"]
    assert entry["source_type"] == "cloth_mesh"
    assert entry["file_meshes_are_zup"] is False
    assert entry["status"] == "ready"
    cloth_validation = entry["validation"]["cloth_mesh"]
    assert cloth_validation["generation"] == "meshy"
    assert cloth_validation["manifold"]["ok"] is True
    assert cloth_validation["genesis_cloth_import"]["ok"] is True


def test_meshy_generated_cloth_prompt_uses_cloth_shell_language():
    request = {
        "name": "fabric_creature_skin",
        "asset_type": "generated_mesh",
        "purpose": "complex FEM.Cloth closed manifold shell with an organic silhouette",
        "scale": 1.0,
        "bbox": [0.4, 0.3, 0.2],
        "cloth_target_edge_length": None,
        "texture_needs": "woven red fabric",
        "simulation_role": "dynamic cloth shell",
    }

    prompt = mesh_episode.mesh_prompt_from_request(request, "task")

    assert "FEM.Cloth closed manifold surface mesh" in prompt
    assert "not as a volumetric FEM soft body" in prompt
    assert "closed watertight manifold surface" in prompt


def test_planner_start_mesh_assets_action_generates_cloth_mesh(tmp_path):
    case_dir = tmp_path / "case"
    request = {
        "name": "cloth_sheet",
        "asset_type": "cloth_mesh_square",
        "purpose": "square FEM cloth sheet",
        "scale": 1.0,
        "bbox": [0.4, 0.4, 0.001],
        "cloth_target_edge_length": 0.04,
        "texture_needs": None,
        "simulation_role": "dynamic FEM cloth sheet",
    }
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="case",
            task="Create one procedural FEM.Cloth sheet.",
            case_dir=case_dir,
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )
    start = session.actions.execute(
        _planner_action(
            action="start_mesh_assets",
            planner_output=_planner_output_with_asset(request),
            asset_names=["cloth_sheet"],
        ),
        turn=0,
    )

    result = session.actions.execute(_planner_action(action="wait_mesh_assets"), turn=1)

    assert start["ok"] is True
    assert start["status"] == "mesh_assets_started"
    assert result["ok"] is True
    assert result["status"] == "mesh_assets_generated"
    assert session.state["assets"]["jobs"]["mesh"]["status"] == "ready"
    manifest = json.loads((case_dir / "assets" / "asset_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["assets"][0]
    assert entry["logical_name"] == "cloth_sheet"
    assert entry["source_type"] == "cloth_mesh"
    assert entry["validation"]["cloth_mesh"]["target_edge_length"] == 0.04
    assert Path(entry["runtime_path"]).is_file()


def _text_to_mesh_bundle_for_test(mesh_path: Path) -> TextToMeshBundle:
    output_dir = mesh_path.parent
    generation = MeshyGenerationResult(
        provider="meshy",
        prompt="Create one FEM.Cloth closed manifold shell.",
        output_dir=output_dir,
        mesh_path=mesh_path,
        prompt_path=output_dir / "prompt.txt",
        submit_response_path=output_dir / "meshy_submit_response.json",
        final_response_path=output_dir / "meshy_final_response.json",
        metadata_path=output_dir / "metadata.json",
        preview_task_id="preview-task",
        final_status="SUCCEEDED",
        submit_response={},
        final_response={},
    )
    repair = MeshRepairResult(
        ok=True,
        input_mesh_path=mesh_path,
        output_mesh_path=mesh_path,
        attempt_index=1,
        strategy_name="test",
        operations=("test_repair",),
        vertex_count_before=42,
        face_count_before=80,
        component_count_before=1,
        vertex_count_after=42,
        face_count_after=80,
        component_count_after=1,
        bbox_min=(-0.5, -0.5, -0.5),
        bbox_max=(0.5, 0.5, 0.5),
        bbox_size=(1.0, 1.0, 1.0),
        centroid_at_origin=True,
    )
    manifold = MeshManifoldCheckResult(
        ok=True,
        mesh_path=mesh_path,
        vertex_count=42,
        face_count=80,
        component_count=1,
        is_watertight=True,
        is_winding_consistent=True,
        volume=1.0,
        tetgen_ready=True,
    )
    return TextToMeshBundle(
        generation=generation,
        repair=repair,
        raw_manifold=manifold,
        manifold=manifold,
        profile_sec={},
    )


def _planner_output_with_asset(asset_request: dict[str, object]) -> dict[str, object]:
    return _planner_output_with_asset_requests([asset_request])


def _planner_output_with_asset_requests(asset_requests: list[dict[str, object]]) -> dict[str, object]:
    normalized_asset_requests = [
        {"cloth_target_edge_length": None, **asset_request} for asset_request in asset_requests
    ]
    return {
        "scene_brief": {
            "user_intent": "test",
            "required_entities": ["soft_asset"],
            "interaction_goal": "test",
            "success_criteria": ["test"],
            "failure_criteria": ["fail"],
            "assumptions": [],
        },
        "physics_plan": {
            "mode": "rigid",
            "deformable_enabled": False,
            "deformable_kind": "none",
            "ipc_enabled": False,
            "rationale": "test",
        },
        "scene_plan": {
            "simulation_strategy": "test",
            "physics_risks": [],
            "resource_level": "low",
            "rendering_needs": [],
        },
        "asset_requests": normalized_asset_requests,
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
            "sim_dt": 0.01,
            "sim_substeps": 1,
            "render_every_n_steps": 1,
            "render_fps": 1,
            "render_budget": 1,
            "render_res": [640, 480],
            "notes": [],
        },
        "risk_register": [],
    }


def _planner_action(
    *,
    action: str,
    planner_output: dict[str, object] | None = None,
    asset_names: list[str] | None = None,
    target_face_count: int | None = None,
    target_edge_length: float | None = None,
    target_face_tolerance: float | None = None,
) -> dict[str, object]:
    return {
        "action": action,
        "rationale": "test",
        "planner_output": planner_output,
        "asset_names": asset_names,
        "target_face_count": target_face_count,
        "target_edge_length": target_edge_length,
        "target_face_tolerance": target_face_tolerance,
        "roles": None,
        "owner": None,
        "repair_brief": None,
        "backend": None,
        "render": None,
        "render_profile": None,
        "save_state_cache": None,
        "require_state_cache": None,
        "replay_cache": None,
        "render_only": None,
        "timeout_sec": None,
        "cwd": None,
        "python_args": None,
        "pytest_args": None,
        "verdict": None,
        "summary": None,
        "notes": [],
        "simdebug_cards": None,
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
                "cloth_target_edge_length": None,
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
    assert math.isclose(report["median_feature_length"], 0.44 / 3.0, rel_tol=1e-9)
    assert math.isclose(report["global_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["selected_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["ipc_contact_d_hat"], 2e-3 * expected_diag, rel_tol=1e-9)


def test_adaptive_d_hat_includes_direct_main_primitives_with_local_constants(tmp_path):
    from code_agent.utils.adaptive_ipc import adaptive_contact_d_hat_report

    case_root = tmp_path / "case"
    src_dir = case_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "main.py").write_text(
        """
import genesis as gs


def main():
    rod_radius = 0.042
    rod_height = 1.36
    tet_resolution = 2
    scene.add_entity(
        morph=gs.morphs.Cylinder(
            radius=rod_radius,
            height=rod_height,
            tet_resolution=tet_resolution,
        ),
        material=None,
    )
""",
        encoding="utf-8",
    )

    report = adaptive_contact_d_hat_report(
        case_root=case_root,
        default_deformable_cfg={"ipc_contact_d_hat_adaptive": True, "tet_resolution": 2},
        repo_root=tmp_path,
    )

    expected_diag = math.sqrt(0.084**2 + 0.084**2 + 1.36**2)
    assert report is not None
    assert report["source"] == "generated source primitive morphs"
    assert report["selected_asset"] == "src/main.py:Cylinder"
    assert report["selected_source_kind"] == "direct_primitive_morph"
    assert math.isclose(report["median_feature_length"], 0.084 / 3.0, rel_tol=1e-9)
    assert math.isclose(report["global_bbox_diag"], expected_diag, rel_tol=1e-9)
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
