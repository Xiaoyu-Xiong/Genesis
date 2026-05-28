"""Programmatic harness for the Opt agent.

This module is the Planner-facing execution wrapper. It builds the Opt prompt,
invokes `codex exec`, loads the subagent's structured JSON, normalizes it into
`OptAgentResult`, and writes `reports/opt_subagent_report.json`. CLI argument
parsing belongs in `opt/cli.py`; case-specific optimization policy belongs in
the Codex subagent's reasoning and generated workspace edits, not here.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from code_agent.assets.builtin_guard import case_source_builtin_asset_violations
from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json, load_json_object
from code_agent.opt.types import OptAgentRequest, OptAgentResult
from code_agent.prompts.opt import build_opt_prompt
from code_agent.utils.codex import DEFAULT_REPO_ROOT, CodexExecRequest, run_codex_exec


OPT_SUBAGENT_SCHEMA = Path("code_agent/specs/opt_schema/opt_subagent_report.schema.json")


def run_opt_agent(request: OptAgentRequest) -> OptAgentResult:
    """Invoke the Codex Opt agent on one generated case workspace."""

    case_dir = request.case_dir.resolve()
    reports_dir = case_dir / "reports"
    logs_dir = case_dir / "logs"

    if not case_dir.is_dir():
        reports_dir.mkdir(parents=True, exist_ok=True)
        result = OptAgentResult(
            status="failed",
            case_type=None,
            diagnosis=f"Case directory does not exist: {case_dir}",
            recommendation="Planner should regenerate or provide a valid case workspace.",
            failures=["missing case_dir"],
        )
        _write_agent_report(case_dir, request, result, codex_result=None)
        return result

    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    codex_sandbox = _codex_sandbox_for_request(request)
    codex_result = run_codex_exec(
        CodexExecRequest(
            role="opt_subagent",
            prompt=build_opt_prompt(request),
            cwd=DEFAULT_REPO_ROOT,
            sandbox=codex_sandbox,
            model=CONFIGS.codex.opt_model,
            output_schema_path=OPT_SUBAGENT_SCHEMA,
            output_jsonl_path=logs_dir / "codex_opt_subagent.jsonl",
            final_message_path=logs_dir / "codex_opt_subagent.final.json",
            timeout_sec=CONFIGS.codex.opt_timeout_sec,
            hide_builtin_assets=codex_sandbox != "danger-full-access",
            writable_roots=(case_dir,),
        )
    )

    payload = _load_opt_result_payload(
        case_dir,
        Path(codex_result.final_message_path),
        min_mtime=codex_result.started_at_unix,
    )
    if payload is not None:
        result = _result_from_payload(payload)
        if not codex_result.success:
            result.evidence.append(
                "Codex invocation returned a non-zero status after writing a parseable Opt result; "
                "using the structured Opt payload as the source of truth."
            )
    elif not codex_result.success:
        recovered = _recover_result_from_opt_report(
            case_dir,
            min_mtime=codex_result.started_at_unix,
            codex_error=codex_result.error_message or codex_result.error_type or "codex_exec_failed",
        )
        if recovered is not None:
            result = recovered
        else:
            result = OptAgentResult(
                status="failed",
                case_type=None,
                diagnosis=f"Codex Opt subagent failed: {codex_result.error_message or codex_result.error_type}",
                recommendation="Planner should retry Opt after resolving the Codex invocation failure.",
                evidence=[f"codex_jsonl={_rel(case_dir, Path(codex_result.output_jsonl_path))}"],
                failures=[codex_result.error_type or "codex_exec_failed"],
            )
    else:
        result = OptAgentResult(
            status="failed",
            case_type=None,
            diagnosis="Codex Opt subagent did not return parseable JSON matching the opt report schema.",
            recommendation="Planner should retry Opt or route the case to structural repair.",
            evidence=[f"codex_final={_rel(case_dir, Path(codex_result.final_message_path))}"],
            failures=["invalid opt subagent final JSON"],
        )

    _enforce_video_evidence(case_dir, request, result)
    _enforce_xml_patch_evidence(result)
    _enforce_builtin_asset_policy(case_dir, result)
    _write_agent_report(case_dir, request, result, codex_result=codex_result.to_dict())
    return result


def _load_opt_result_payload(case_dir: Path, final_message_path: Path, *, min_mtime: float) -> dict[str, Any] | None:
    payload = load_json_object(final_message_path)
    if payload is not None:
        return payload
    report_path = case_dir / "reports" / "opt_subagent_report.json"
    if report_path.exists() and report_path.stat().st_mtime + _FRESH_MTIME_TOLERANCE_SEC < min_mtime:
        return None
    report = load_json_object(report_path)
    result = report.get("result") if isinstance(report, dict) else None
    return result if isinstance(result, dict) else None


def _codex_sandbox_for_request(request: OptAgentRequest) -> str:
    if request.backend == "gpu":
        return "danger-full-access"
    return CONFIGS.codex.opt_sandbox


def _result_from_payload(payload: dict[str, Any]) -> OptAgentResult:
    return OptAgentResult(
        status=str(payload.get("status", "failed")),
        case_type=_optional_str(payload.get("case_type")),
        edited_files=_str_list(payload.get("edited_files")),
        optimized_variables=_str_list(payload.get("optimized_variables")),
        baseline=_dict(payload.get("baseline")),
        best=_dict(payload.get("best")),
        diagnosis=_optional_str(payload.get("diagnosis")),
        recommendation=_optional_str(payload.get("recommendation")),
        evidence=_str_list(payload.get("evidence")),
        opt_report_path=_optional_str(payload.get("opt_report_path")),
        failures=_str_list(payload.get("failures")),
    )


def _recover_result_from_opt_report(case_dir: Path, *, min_mtime: float, codex_error: str) -> OptAgentResult | None:
    opt_report_path = case_dir / "reports" / "opt_report.json"
    if not _is_fresh_file(opt_report_path, min_mtime=min_mtime):
        return None
    opt_report = load_json_object(opt_report_path)
    if not isinstance(opt_report, dict):
        return None

    verification_path = _resolve_case_path(case_dir, opt_report.get("verification_report_path"))
    verification = load_json_object(verification_path) if verification_path and verification_path.is_file() else {}
    opt_space = load_json_object(case_dir / "contracts" / "opt_space.json") or {}
    target_spec = load_json_object(case_dir / "contracts" / "target_spec.json") or {}
    trace_entries = _load_trace_entries(case_dir, opt_report.get("trace_path"))

    best_trial = opt_report.get("best_trial")
    best_entry = _trace_entry_by_trial(trace_entries, best_trial)
    baseline_entry = trace_entries[0] if trace_entries else None
    best_render_dir = _resolve_case_path(case_dir, opt_report.get("best_render_dir"))
    best_success = _optional_bool(verification.get("success")) if isinstance(verification, dict) else None
    if best_success is None and isinstance(best_entry, dict):
        objective = best_entry.get("objective")
        best_success = _optional_bool(objective.get("success")) if isinstance(objective, dict) else None

    status = _recovered_status(opt_report, verification, target_spec)
    evidence = [
        "Codex Opt subagent did not return a structured final payload, but a fresh lower-level opt_report.json was "
        "written during this invocation; recovering that evidence instead of discarding it.",
        f"codex_error={codex_error}",
        f"opt_report={_rel(case_dir, opt_report_path)}",
        f"opt_report.status={opt_report.get('status')}",
        f"num_trials={opt_report.get('num_trials')}",
        f"baseline_score={opt_report.get('baseline_score')}",
        f"best_score={opt_report.get('best_score')}",
    ]
    if verification_path is not None and verification_path.is_file():
        evidence.append(f"verification_report={_rel(case_dir, verification_path)} success={verification.get('success')}")

    failures = [codex_error]
    failures.extend(str(item) for item in opt_report.get("failures", []) if item)
    return OptAgentResult(
        status=status,
        case_type=_optional_str(target_spec.get("task_family")) if isinstance(target_spec, dict) else None,
        edited_files=_recovered_edited_files(opt_space),
        optimized_variables=_optimized_variables(opt_space),
        baseline=_outcome_from_entry(
            case_dir=case_dir,
            entry=baseline_entry,
            score=opt_report.get("baseline_score"),
            params_path="contracts/default_opt_params.json",
            video_path="artifacts/opt_agent_baseline/render.mp4",
            summary="Recovered baseline evidence from the lower-level optimization report.",
        ),
        best=_outcome_from_entry(
            case_dir=case_dir,
            entry=best_entry,
            score=opt_report.get("best_score"),
            params_path=opt_report.get("best_params_path"),
            video_path=None if best_render_dir is None else str(Path(_rel(case_dir, best_render_dir)) / "render.mp4"),
            summary="Recovered best-trial evidence from the lower-level optimization report.",
            success=best_success,
        ),
        diagnosis=(
            "The Codex Opt subagent invocation failed or timed out after preparing/running the lower-level optimizer. "
            "A fresh opt_report.json was recovered, so Planner should use the numerical evidence rather than treating "
            "the optimization attempt as empty."
        ),
        recommendation=_recovered_recommendation(status),
        evidence=evidence,
        opt_report_path=_rel(case_dir, opt_report_path),
        failures=failures,
    )


def _recovered_status(opt_report: dict[str, Any], verification: dict[str, Any] | None, target_spec: dict[str, Any]) -> str:
    if isinstance(verification, dict) and verification.get("success") is True:
        return "success"
    if opt_report.get("status") == "completed" and opt_report.get("best_score") is not None:
        return "needs_more_optimization" if _score_improved(opt_report, target_spec) else "failed"
    return "failed"


def _recovered_recommendation(status: str) -> str:
    if status == "success":
        return "Planner should rerun execution/critic with the recovered best parameters."
    if status == "needs_more_optimization":
        return (
            "Planner should inspect the recovered best render/metrics and either run another Opt pass with a narrower "
            "budget or route structural repair if the remaining miss is not parameter-level."
        )
    return "Planner should inspect the recovered opt report and route retry or structural repair."


def _score_improved(opt_report: dict[str, Any], target_spec: dict[str, Any]) -> bool:
    baseline = opt_report.get("baseline_score")
    best = opt_report.get("best_score")
    if not isinstance(baseline, int | float) or not isinstance(best, int | float):
        return "improved" in str(opt_report.get("summary", "")).lower()
    direction = None
    objective = target_spec.get("objective") if isinstance(target_spec, dict) else None
    if isinstance(objective, dict):
        direction = objective.get("direction")
    if direction == "maximize":
        return float(best) > float(baseline)
    if direction == "minimize":
        return float(best) < float(baseline)
    return "improved" in str(opt_report.get("summary", "")).lower()


def _outcome_from_entry(
    *,
    case_dir: Path,
    entry: dict[str, Any] | None,
    score: Any,
    params_path: Any,
    video_path: str | None,
    summary: str,
    success: bool | None = None,
) -> dict[str, Any]:
    objective = entry.get("objective") if isinstance(entry, dict) else None
    if success is None and isinstance(objective, dict):
        success = _optional_bool(objective.get("success"))
    metrics_path = entry.get("metrics_path") if isinstance(entry, dict) else None
    if video_path is not None and not (case_dir / video_path).is_file():
        video_path = None
    return {
        "success": success,
        "score": float(score) if isinstance(score, int | float) else None,
        "metrics_path": str(metrics_path) if isinstance(metrics_path, str) else None,
        "video_path": video_path,
        "params_path": str(params_path) if isinstance(params_path, str) else None,
        "summary": summary,
    }


def _optimized_variables(opt_space: dict[str, Any]) -> list[str]:
    variables = opt_space.get("variables") if isinstance(opt_space, dict) else None
    if not isinstance(variables, list):
        return []
    return [str(variable.get("name")) for variable in variables if isinstance(variable, dict) and variable.get("name")]


def _recovered_edited_files(opt_space: dict[str, Any]) -> list[str]:
    files = {"contracts/target_spec.json", "contracts/opt_space.json", "contracts/default_opt_params.json"}
    variables = opt_space.get("variables") if isinstance(opt_space, dict) else None
    if isinstance(variables, list):
        owners = {str(variable.get("owner")) for variable in variables if isinstance(variable, dict)}
        for owner in owners:
            if owner in {"scene", "body", "action"}:
                files.add(f"src/{owner}.py")
            elif owner == "xml":
                files.add("assets/xml/**/*.xml")
    files.update({"reports/opt_report.json", "reports/verification_report.json"})
    return sorted(files)


def _load_trace_entries(case_dir: Path, trace_path_value: Any) -> list[dict[str, Any]]:
    trace_path = _resolve_case_path(case_dir, trace_path_value)
    if trace_path is None or not trace_path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = load_json_object_from_text(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def load_json_object_from_text(raw: str) -> dict[str, Any] | None:
    import json

    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else None


def _trace_entry_by_trial(entries: list[dict[str, Any]], trial_index: Any) -> dict[str, Any] | None:
    if not isinstance(trial_index, int):
        return None
    for entry in entries:
        if entry.get("trial_index") == trial_index:
            return entry
    return None


def _resolve_case_path(case_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = case_dir / path
    try:
        path.resolve().relative_to(case_dir.resolve())
    except ValueError:
        return None
    return path


_FRESH_MTIME_TOLERANCE_SEC = 5.0


def _is_fresh_file(path: Path, *, min_mtime: float) -> bool:
    return path.is_file() and path.stat().st_mtime + _FRESH_MTIME_TOLERANCE_SEC >= min_mtime


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


_VIDEO_EVIDENCE_MARKERS = (
    "video_checked",
    "visual evidence",
    "inspected video",
    "reviewed video",
    "sampled frame",
    "frame sample",
    "watched video",
)


def _enforce_video_evidence(case_dir: Path, request: OptAgentRequest, result: OptAgentResult) -> None:
    if not request.render_best or result.status not in {"success", "partial_success"}:
        return
    best_video = result.best.get("video_path") if isinstance(result.best, dict) else None
    video_path = _resolve_case_path(case_dir, best_video)
    has_video = video_path is not None and video_path.is_file()
    evidence_text = " ".join(
        str(item)
        for item in [
            *(result.evidence or []),
            result.diagnosis or "",
            result.recommendation or "",
            result.best.get("summary") if isinstance(result.best, dict) else "",
        ]
    ).lower()
    has_explicit_review = any(marker in evidence_text for marker in _VIDEO_EVIDENCE_MARKERS)
    if has_video and has_explicit_review:
        return

    previous_status = result.status
    result.status = "needs_more_optimization"
    result.failures.append("missing_explicit_video_evidence")
    if not has_video:
        result.evidence.append(
            "Opt success was downgraded because no best render video file was available for visual verification."
        )
    else:
        result.evidence.append(
            "Opt success was downgraded because the final report did not explicitly state that best-video/frame "
            "evidence was inspected."
        )
    result.diagnosis = (
        f"Opt returned {previous_status}, but Planner requires visual evidence in addition to numeric metrics. "
        f"{result.diagnosis or ''}"
    ).strip()
    result.recommendation = (
        "Planner should ask Opt to inspect the best render/video evidence and only accept success if the video supports "
        "the metric result."
    )


_XML_PATCH_EVIDENCE_MARKERS = (
    "xml_scalar_patch_validated",
    "xml scalar patch validated",
)


def _enforce_xml_patch_evidence(result: OptAgentResult) -> None:
    if result.status not in {"success", "partial_success", "needs_more_optimization"}:
        return
    touched_xml = any(str(path).startswith("assets/xml/") for path in result.edited_files)
    touched_xml = touched_xml or any(str(name).startswith("xml.") for name in result.optimized_variables)
    if not touched_xml:
        return
    evidence_text = " ".join(str(item) for item in [*(result.evidence or []), result.diagnosis or ""]).lower()
    if any(marker in evidence_text for marker in _XML_PATCH_EVIDENCE_MARKERS):
        return
    result.status = "needs_more_optimization"
    result.failures.append("missing_xml_scalar_patch_validation")
    result.evidence.append(
        "Opt result was downgraded because XML asset parameters were edited or optimized without explicit "
        "xml_scalar_patch_validated evidence that only whitelisted numeric attributes changed."
    )
    result.recommendation = (
        "Planner should ask Opt to validate XML scalar changes or route an XML asset rewrite if topology must change."
    )


def _enforce_builtin_asset_policy(case_dir: Path, result: OptAgentResult) -> None:
    violations = case_source_builtin_asset_violations(case_dir)
    if not violations:
        return
    result.status = "failed"
    result.failures.extend(violations)
    result.evidence.append("Opt result rejected because generated source references forbidden Genesis built-in assets.")
    result.diagnosis = (
        "Generated source references forbidden Genesis built-in assets after the Opt pass. "
        f"{result.diagnosis or ''}"
    ).strip()
    result.recommendation = (
        "Planner should route source repair so the case uses primitives, case-generated assets, or explicit "
        "case-workspace assets instead of Genesis built-in assets."
    )


def _write_agent_report(
    case_dir: Path,
    request: OptAgentRequest,
    result: OptAgentResult,
    *,
    codex_result: dict[str, Any] | None,
) -> Path:
    report_path = case_dir / "reports" / "opt_subagent_report.json"
    result.subagent_report_path = _rel(case_dir, report_path)
    dump_json(
        _json_safe(
            {
                "schema_version": 1,
                "request": asdict(request),
                "result": asdict(result),
                "codex_result": codex_result,
            }
        ),
        report_path,
    )
    return report_path


def _rel(case_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(case_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
