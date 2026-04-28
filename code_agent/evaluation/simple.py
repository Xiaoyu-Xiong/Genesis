from __future__ import annotations

import json
from pathlib import Path

from .codex_critic import run_codex_critic
from .deterministic import DeterministicEvaluationConfig, evaluate_run as evaluate_deterministic_run


def load_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def evaluate_run(
    *,
    run_dir: Path,
    task: str,
    execution_ok: bool,
    require_render: bool = True,
    use_codex_critic: bool = True,
) -> dict[str, object]:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    execution_report_path = reports_dir / "execution_report.json"
    raw_report = evaluate_deterministic_run(
        DeterministicEvaluationConfig(
            run_dir=run_dir,
            execution_report_path=execution_report_path,
            output_path=reports_dir / "critic_report.json",
            require_successful_exit=True,
            require_render=require_render,
        )
    )
    codex_report = (
        run_codex_critic(run_dir=run_dir, task=task, deterministic_report=raw_report) if use_codex_critic else None
    )
    codex_passed = codex_report is None or codex_report.get("verdict") == "pass"
    verdict = "pass" if execution_ok and raw_report["passed"] and codex_passed else "fail"
    missing = [
        check["name"]
        for check in raw_report["checks"]
        if check["status"] == "fail" and str(check.get("reason", "")).endswith(".missing")
    ]
    render_ok = "render.missing" not in raw_report["failure_classes"] and "render.empty" not in raw_report["failure_classes"]
    recommended_owner = "none"
    if codex_report is not None:
        recommended_owner = str(codex_report.get("recommended_owner", "none"))
    elif not execution_ok:
        recommended_owner = "execution"
    report = {
        "verdict": verdict,
        "confidence": 0.75 if verdict == "pass" else 0.35,
        "task": task,
        "execution_ok": execution_ok,
        "metric_ok": "metrics.missing" not in raw_report["failure_classes"],
        "render_ok": render_ok,
        "event_ok": True,
        "missing_artifacts": missing,
        "recommended_owner": recommended_owner,
        "physical_plausibility_score": 0.6 if verdict == "pass" else 0.2,
        "task_completion_score": 0.6 if verdict == "pass" else 0.2,
        "visual_clarity_score": 0.6 if render_ok else 0.0,
        "summary": "Combined deterministic checks and single-pass Codex critic.",
        "deterministic_report": raw_report,
        "codex_critic_report": codex_report,
    }
    (reports_dir / "critic_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
