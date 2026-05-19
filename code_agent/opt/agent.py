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

from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json, load_json_object
from code_agent.prompts.opt import build_opt_prompt
from code_agent.opt.types import OptAgentRequest, OptAgentResult
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

    codex_result = run_codex_exec(
        CodexExecRequest(
            role="opt_subagent",
            prompt=build_opt_prompt(request),
            cwd=DEFAULT_REPO_ROOT,
            sandbox=CONFIGS.codex.opt_sandbox,
            model=CONFIGS.codex.opt_model,
            output_schema_path=OPT_SUBAGENT_SCHEMA,
            output_jsonl_path=logs_dir / "codex_opt_subagent.jsonl",
            final_message_path=logs_dir / "codex_opt_subagent.final.json",
            timeout_sec=CONFIGS.codex.opt_timeout_sec,
        )
    )

    payload = _load_opt_result_payload(case_dir, Path(codex_result.final_message_path))
    if not codex_result.success:
        result = OptAgentResult(
            status="failed",
            case_type=None,
            diagnosis=f"Codex Opt subagent failed: {codex_result.error_message or codex_result.error_type}",
            recommendation="Planner should retry Opt after resolving the Codex invocation failure.",
            evidence=[f"codex_jsonl={_rel(case_dir, Path(codex_result.output_jsonl_path))}"],
            failures=[codex_result.error_type or "codex_exec_failed"],
        )
    elif payload is None:
        result = OptAgentResult(
            status="failed",
            case_type=None,
            diagnosis="Codex Opt subagent did not return parseable JSON matching the opt report schema.",
            recommendation="Planner should retry Opt or route the case to structural repair.",
            evidence=[f"codex_final={_rel(case_dir, Path(codex_result.final_message_path))}"],
            failures=["invalid opt subagent final JSON"],
        )
    else:
        result = _result_from_payload(payload)

    _write_agent_report(case_dir, request, result, codex_result=codex_result.to_dict())
    return result


def _load_opt_result_payload(case_dir: Path, final_message_path: Path) -> dict[str, Any] | None:
    payload = load_json_object(final_message_path)
    if payload is not None:
        return payload
    report = load_json_object(case_dir / "reports" / "opt_subagent_report.json")
    result = report.get("result") if isinstance(report, dict) else None
    return result if isinstance(result, dict) else None


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
