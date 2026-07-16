from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import run_codex_critic
from .deterministic import DeterministicEvaluationConfig, evaluate_artifacts
from .visual import evaluate_visual_artifacts


def evaluate_generated_run(
    *,
    run_dir: Path,
    task: str,
    execution_ok: bool,
    require_render: bool = True,
    use_codex_critic: bool = True,
    simdebug_card_context: str = "",
) -> dict[str, Any]:
    """Evaluate one generated simulation run and write the merged critic report."""

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    artifact_report = evaluate_artifacts(
        DeterministicEvaluationConfig(
            run_dir=run_dir,
            execution_report_path=reports_dir / "execution_report.json",
            output_path=reports_dir / "artifact_evaluation.json",
            require_successful_exit=True,
            require_render=require_render,
        )
    )
    visual_report = evaluate_visual_artifacts(run_dir=run_dir, output_path=reports_dir / "visual_evaluation.json")
    artifact_report["visual_report"] = visual_report
    codex_report = (
        run_codex_critic(
            run_dir=run_dir,
            task=task,
            artifact_report=artifact_report,
            simdebug_card_context=simdebug_card_context,
        )
        if use_codex_critic
        else None
    )
    codex_passed = codex_report is None or codex_report.get("verdict") == "pass"
    critic_infra_status = _critic_infra_status(codex_report)
    critic_infra_blocked = critic_infra_status not in {"ok", "not_used"}
    if execution_ok and artifact_report["passed"] and critic_infra_blocked:
        verdict = "inconclusive"
    else:
        verdict = "pass" if execution_ok and artifact_report["passed"] and codex_passed else "fail"
    missing = [
        check["name"]
        for check in artifact_report["checks"]
        if check["status"] == "fail" and str(check.get("reason", "")).endswith(".missing")
    ]
    failure_classes = artifact_report["failure_classes"]
    render_ok = "render.missing" not in failure_classes and "render.empty" not in failure_classes
    deterministic_owner = str(artifact_report.get("recommended_owner") or "none")
    deterministic_repair_summary = artifact_report.get("repair_summary")
    strong_deterministic_owner = deterministic_owner != "none" and "ipc.initial_penetration" in failure_classes
    if strong_deterministic_owner:
        recommended_owner = deterministic_owner
    elif codex_report is not None:
        recommended_owner = str(codex_report.get("recommended_owner", "none"))
    elif deterministic_owner != "none":
        recommended_owner = deterministic_owner
    elif not execution_ok:
        recommended_owner = "execution"
    else:
        recommended_owner = "none"

    report: dict[str, Any] = {
        "verdict": verdict,
        "confidence": 0.75 if verdict == "pass" else 0.0 if verdict == "inconclusive" else 0.35,
        "task": task,
        "execution_ok": execution_ok,
        "metric_ok": "metrics.missing" not in failure_classes,
        "render_ok": render_ok,
        "event_ok": True,
        "missing_artifacts": missing,
        "recommended_owner": recommended_owner,
        "deterministic_recommended_owner": deterministic_owner,
        "deterministic_repair_summary": deterministic_repair_summary,
        "physical_plausibility_score": 0.6 if verdict == "pass" else 0.2,
        "task_completion_score": 0.6 if verdict == "pass" else 0.2,
        "visual_clarity_score": 0.6 if render_ok else 0.0,
        "summary": deterministic_repair_summary or "Combined artifact checks and Codex critic.",
        "critic_infra_status": critic_infra_status,
        "critic_attempts": _critic_attempt_count(codex_report),
        "artifact_report": artifact_report,
        "codex_critic_report": codex_report,
        "cache_source_consistency": (
            codex_report.get("cache_source_consistency") if isinstance(codex_report, dict) else None
        ),
        "state_cache_source_report": (
            codex_report.get("state_cache_source_report") if isinstance(codex_report, dict) else None
        ),
    }
    (reports_dir / "critic_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _critic_infra_status(codex_report: dict[str, Any] | None) -> str:
    if codex_report is None:
        return "not_used"
    status = codex_report.get("critic_infra_status")
    if isinstance(status, str) and status:
        return status
    codex_result = codex_report.get("codex_result")
    if not isinstance(codex_result, dict):
        return "ok"
    error_type = codex_result.get("error_type")
    error_message = str(codex_result.get("error_message") or "").lower()
    if error_type == "codex_usage_limit" or "usage limit" in error_message:
        return "quota_blocked"
    if error_type == "codex_auth_failed" or "401 unauthorized" in error_message:
        return "auth_failed"
    if error_type == "codex_input_too_large" or "input exceeds the maximum length" in error_message:
        return "critic_prompt_too_large"
    if error_type == "timeout":
        return "critic_timeout"
    if error_type:
        return "critic_exec_failed"
    return "ok"


def _critic_attempt_count(codex_report: dict[str, Any] | None) -> int:
    if not isinstance(codex_report, dict):
        return 0
    attempts = codex_report.get("critic_attempts")
    return len(attempts) if isinstance(attempts, list) else 1
