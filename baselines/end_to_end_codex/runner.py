from __future__ import annotations

import json
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - the uv environment normally has jsonschema.
    Draft202012Validator = None  # type: ignore[assignment]

from code_agent.assets.builtin_guard import source_file_builtin_asset_violations
from code_agent.assets.layout_reuse import prepare_layout_reusable_assets
from code_agent.assets.mesh.episode import generate_mesh_assets_for_episode
from code_agent.assets.mesh.request_adapter import select_mesh_requests
from code_agent.context.genesis import GenesisContextPack, build_genesis_context_pack, install_genesis_context_pack
from code_agent.io_utils import dump_json, load_json_object
from code_agent.utils.codex import DEFAULT_REPO_ROOT, CodexExecRequest, CodexExecResult, run_codex_exec
from code_agent.utils.execution import (
    GENESIS_EXECUTION_LOCK_PATH_ENV,
    ExecutionReport,
    _resolve_genesis_execution_lock_path,
    run_generated_simulation,
)
from code_agent.utils.suite import Case, load_cases

from baselines.end_to_end_codex.configs import (
    DEFAULT_BACKEND,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    DEFAULT_CODEX_SANDBOX,
    DEFAULT_CODEX_SERVICE_TIER,
    DEFAULT_CODEX_TIMEOUT_SEC,
    DEFAULT_EXECUTION_TIMEOUT_SEC,
    DEFAULT_MAX_PARALLEL_CASES,
    deformable_config_from_planner_output,
)
from baselines.end_to_end_codex.case_tools import apply_adaptive_ipc_d_hat
from baselines.end_to_end_codex.prompt import (
    build_end_to_end_prompt,
    load_genesis_context_summary,
    load_layout_context,
)
from baselines.end_to_end_codex.timing import BaselineTimingPlan, resolve_baseline_timing


@dataclass(slots=True, frozen=True)
class EndToEndBaselineConfig:
    tasks_file: Path
    out_dir: Path
    backend: str = DEFAULT_BACKEND
    max_cases: int | None = None
    max_parallel_cases: int | None = DEFAULT_MAX_PARALLEL_CASES
    execution_timeout_sec: float = DEFAULT_EXECUTION_TIMEOUT_SEC
    codex_timeout_sec: float = DEFAULT_CODEX_TIMEOUT_SEC
    render: bool = True
    steps: int | None = None
    duration_sec: float | None = None
    render_fps: int | None = None
    model: str = DEFAULT_CODEX_MODEL
    reasoning_effort: str | None = DEFAULT_CODEX_REASONING_EFFORT
    service_tier: Literal["fast", "standard"] | None = DEFAULT_CODEX_SERVICE_TIER


def run_end_to_end_suite(config: EndToEndBaselineConfig) -> dict[str, Any]:
    tasks_file = config.tasks_file.resolve()
    out_dir = config.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    execution_lock_path = _resolve_genesis_execution_lock_path()
    execution_lock_path.parent.mkdir(parents=True, exist_ok=True)
    execution_lock_path.touch(exist_ok=True)
    cases = load_cases(tasks_file)
    if config.max_cases is not None:
        cases = cases[: config.max_cases]

    genesis_context = build_genesis_context_pack(out_dir)
    max_workers = resolve_case_parallelism(num_cases=len(cases), max_parallel_cases=config.max_parallel_cases)
    started_at = time.time()
    results_by_index: list[dict[str, Any] | None] = [None] * len(cases)
    summary_base = {
        "baseline": "end_to_end_codex",
        "tasks_file": str(tasks_file),
        "out_dir": str(out_dir),
        "backend": config.backend,
        "render": config.render,
        "steps": config.steps,
        "duration_sec": config.duration_sec,
        "render_fps": config.render_fps,
        "model": config.model,
        "max_parallel_cases": max_workers,
        "codex_parallel": max_workers > 1,
        "mesh_agent": "code_agent.assets.mesh.episode.generate_mesh_assets_for_episode",
        "genesis_execution_lock": {
            "scope": "all baseline agent-launched and harness-launched run_generated_simulation calls for this suite",
            "path": str(execution_lock_path),
            "allows_parallel_codex_generation": True,
            "allows_agent_in_loop_execution": True,
            "serializes_local_genesis_execution": True,
        },
        "genesis_context_path": str(genesis_context.markdown_path),
        "genesis_context_json_path": str(genesis_context.json_path),
        "genesis_docs_dir": str(genesis_context.docs_dir),
    }

    if not cases:
        summary = suite_summary(summary_base, results=[])
        dump_json(summary, out_dir / "summary.json")
        return summary

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="baseline-e2e-case") as executor:
        futures = {
            executor.submit(
                _run_case,
                case=case,
                suite_config=config,
                out_dir=out_dir,
                genesis_context=genesis_context,
                execution_lock_path=execution_lock_path,
            ): index
            for index, case in enumerate(cases)
        }
        for future in as_completed(futures):
            index = futures[future]
            case = cases[index]
            case_dir = out_dir / case.case_id
            try:
                result = future.result()
            except Exception as exc:
                result = case_exception_summary(case=case, case_dir=case_dir, exc=exc)
            results_by_index[index] = result
            partial = suite_summary(
                summary_base,
                results=[item for item in results_by_index if item is not None],
                num_cases_total=len(cases),
                started_at_unix=started_at,
                completed_at_unix=None,
            )
            dump_json(partial, out_dir / "summary.json")

    results = [
        item if item is not None else missing_case_summary(cases[index], out_dir / cases[index].case_id)
        for index, item in enumerate(results_by_index)
    ]
    summary = suite_summary(
        summary_base,
        results=results,
        num_cases_total=len(cases),
        started_at_unix=started_at,
        completed_at_unix=time.time(),
    )
    dump_json(summary, out_dir / "summary.json")
    return summary


def resolve_case_parallelism(*, num_cases: int, max_parallel_cases: int | None) -> int:
    if num_cases <= 0:
        return 1
    if max_parallel_cases is None:
        return num_cases
    return max(1, min(int(max_parallel_cases), num_cases))


def _run_case(
    *,
    case: Case,
    suite_config: EndToEndBaselineConfig,
    out_dir: Path,
    genesis_context: GenesisContextPack,
    execution_lock_path: Path,
) -> dict[str, Any]:
    case_dir = out_dir / case.case_id
    _prepare_case_workspace(case_dir=case_dir, case=case)
    install_genesis_context_pack(case_dir, genesis_context)

    codex_result = _run_end_to_end_codex(
        case=case,
        case_dir=case_dir,
        suite_config=suite_config,
        execution_lock_path=execution_lock_path,
    )
    worker_report, worker_error = parse_worker_report(Path(codex_result.final_message_path))
    planner_output = load_json_object(case_dir / "contracts" / "planner_output.json") or {}
    planner_errors = validate_planner_output(planner_output) if planner_output else ["missing planner_output.json"]
    deformable_config = deformable_config_from_planner_output(planner_output)
    dump_json(deformable_config, case_dir / "contracts" / "deformable_config.json")

    timing: BaselineTimingPlan | None = None
    timing_error: str | None = None
    try:
        timing = resolve_baseline_timing(
            planner_output=planner_output,
            steps=suite_config.steps,
            duration_sec=suite_config.duration_sec,
            render_fps=suite_config.render_fps,
        )
        dump_json(timing.to_dict(), case_dir / "contracts" / "timing.json")
    except Exception as exc:
        timing_error = f"{type(exc).__name__}: {exc}"

    asset_report = _maybe_generate_mesh_assets(case_dir=case_dir, task=case.task, planner_output=planner_output)
    asset_errors = _asset_errors(asset_report)
    adaptive_ipc_report = apply_adaptive_ipc_d_hat(case_dir=case_dir)
    adaptive_ipc_errors = _adaptive_ipc_errors(adaptive_ipc_report)
    source_violations = _source_violations(case_dir / "src" / "main.py", case_dir=case_dir)
    execution = None
    if _ready_for_execution(
        codex_result=codex_result,
        worker_report=worker_report,
        worker_error=worker_error,
        timing=timing,
        planner_errors=planner_errors,
        asset_errors=asset_errors,
        adaptive_ipc_errors=adaptive_ipc_errors,
        source_violations=source_violations,
    ):
        execution = run_generated_simulation(
            main_py=case_dir / "src" / "main.py",
            run_dir=case_dir,
            backend=suite_config.backend,
            timeout_sec=suite_config.execution_timeout_sec,
            steps=timing.steps,
            render_fps=timing.render_fps,
            sim_dt=timing.sim_dt,
            sim_substeps=timing.sim_substeps,
            render_every_n_steps=timing.render_every_n_steps,
            render_res=timing.render_res,
            render=suite_config.render,
            duration_sec=timing.duration_sec,
            target_video_frames=timing.target_video_frames,
            execution_lock_path=execution_lock_path,
        )

    summary = case_summary(
        case=case,
        case_dir=case_dir,
        codex_result=codex_result,
        worker_report=worker_report,
        worker_error=worker_error,
        planner_errors=planner_errors,
        timing=timing,
        timing_error=timing_error,
        asset_report=asset_report,
        asset_errors=asset_errors,
        adaptive_ipc_report=adaptive_ipc_report,
        adaptive_ipc_errors=adaptive_ipc_errors,
        source_violations=source_violations,
        execution=execution,
    )
    dump_json(summary, case_dir / "summary.json")
    return summary


def _prepare_case_workspace(*, case_dir: Path, case: Case) -> None:
    for path in (
        case_dir / "inputs",
        case_dir / "contracts",
        case_dir / "logs",
        case_dir / "reports",
        case_dir / "src",
        case_dir / "artifacts",
    ):
        path.mkdir(parents=True, exist_ok=True)
    (case_dir / "inputs" / "user_prompt.md").write_text(case.task + "\n", encoding="utf-8")
    (case_dir / "inputs" / "capabilities.md").write_text(
        "\n".join(
            [
                "End-to-end single Codex agent baseline.",
                "The agent writes, runs, inspects, and repairs the generated simulation in one Codex invocation.",
                "The agent may generate declared Meshy assets through baselines.end_to_end_codex.case_tools.",
                "Agent-launched and harness-launched Genesis executions are serialized through the shared lock.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if case.layout_path is not None:
        layout_text = case.layout_path.read_text(encoding="utf-8", errors="replace")
        layout_copy = case_dir / "inputs" / f"layout{case.layout_path.suffix or '.txt'}"
        layout_copy.write_text(layout_text, encoding="utf-8")
        layout_asset_report = prepare_layout_reusable_assets(case_dir=case_dir, layout_path=case.layout_path)
        layout_lines = []
        if layout_asset_report is not None:
            layout_lines = [
                "",
                "Reusable layout assets:",
                f"- Status: `{layout_asset_report.get('status')}`",
                f"- Partial manifest: `{layout_asset_report.get('asset_manifest_path')}`",
                f"- Report: `{layout_asset_report.get('asset_generation_report_path')}`",
            ]
        language = case.layout_path.suffix.lstrip(".") or "text"
        (case_dir / "inputs" / "layout_context.md").write_text(
            "\n".join(
                [
                    "# User-Provided Layout",
                    "",
                    f"- Original layout path: `{case.layout_path}`",
                    f"- Workspace copy: `{layout_copy}`",
                    *layout_lines,
                    "",
                    f"```{language}",
                    layout_text.rstrip(),
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def _run_end_to_end_codex(
    *,
    case: Case,
    case_dir: Path,
    suite_config: EndToEndBaselineConfig,
    execution_lock_path: Path,
) -> CodexExecResult:
    logs_dir = case_dir / "logs"
    prompt = build_end_to_end_prompt(
        case_id=case.case_id,
        task=case.task,
        case_dir=case_dir,
        backend=suite_config.backend,
        render=suite_config.render,
        steps=suite_config.steps,
        duration_sec=suite_config.duration_sec,
        render_fps=suite_config.render_fps,
        genesis_context=load_genesis_context_summary(case_dir),
        layout_context=load_layout_context(case_dir),
    )
    (logs_dir / "end_to_end_prompt.md").write_text(prompt + "\n", encoding="utf-8")
    return run_codex_exec(
        CodexExecRequest(
            role="end_to_end_codex",
            prompt=prompt,
            cwd=DEFAULT_REPO_ROOT,
            sandbox=DEFAULT_CODEX_SANDBOX,
            model=suite_config.model,
            output_schema_path=Path("code_agent/specs/worker_report.schema.json"),
            output_jsonl_path=logs_dir / "codex_end_to_end.jsonl",
            final_message_path=logs_dir / "codex_end_to_end.final.json",
            reasoning_effort=suite_config.reasoning_effort,
            service_tier=suite_config.service_tier,
            timeout_sec=suite_config.codex_timeout_sec,
            writable_roots=(case_dir, execution_lock_path),
            env_overrides=((GENESIS_EXECUTION_LOCK_PATH_ENV, str(execution_lock_path)),),
        )
    )


def parse_worker_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing final message: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid final JSON: {exc}"
    if not isinstance(data, dict):
        return None, "final message is not a JSON object"
    return data, None


def validate_planner_output(planner_output: dict[str, Any]) -> list[str]:
    schema_path = DEFAULT_REPO_ROOT / "code_agent" / "specs" / "planner_output.schema.json"
    if Draft202012Validator is None:
        return []
    schema = load_json_object(schema_path)
    if not schema:
        return [f"missing planner_output schema: {schema_path}"]
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(planner_output), key=lambda item: list(item.path))
    return [f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}" for error in errors]


def _physics_mode_from_planner_output(planner_output: dict[str, Any]) -> str:
    physics_plan = planner_output.get("physics_plan")
    if isinstance(physics_plan, dict):
        mode = str(physics_plan.get("mode") or "")
        if mode in {"rigid", "rigid_ipc", "fem_ipc"}:
            return mode
    return "rigid"


def _maybe_generate_mesh_assets(*, case_dir: Path, task: str, planner_output: dict[str, Any]) -> dict[str, Any] | None:
    selected, skipped = select_mesh_requests(planner_output, asset_names=None)
    if not selected and not skipped:
        return None
    return generate_mesh_assets_for_episode(case_dir=case_dir, task=task, planner_output=planner_output)


def _asset_errors(asset_report: dict[str, Any] | None) -> list[str]:
    if asset_report is None or bool(asset_report.get("ok")):
        return []
    errors = [str(asset_report.get("status") or "mesh_asset_generation_failed")]
    message = asset_report.get("message")
    if message:
        errors.append(str(message))
    failure_classes = asset_report.get("failure_classes")
    if isinstance(failure_classes, list):
        errors.extend(str(item) for item in failure_classes)
    return errors


def _adaptive_ipc_errors(adaptive_ipc_report: dict[str, Any] | None) -> list[str]:
    if adaptive_ipc_report is None or bool(adaptive_ipc_report.get("ok")):
        return []
    return [str(adaptive_ipc_report.get("status") or "adaptive_ipc_failed")]


def _source_violations(main_py: Path, *, case_dir: Path) -> list[str]:
    if not main_py.exists():
        return [f"missing generated entrypoint: {main_py}"]
    return source_file_builtin_asset_violations(main_py, case_dir=case_dir)


def _ready_for_execution(
    *,
    codex_result: CodexExecResult,
    worker_report: dict[str, Any] | None,
    worker_error: str | None,
    timing: BaselineTimingPlan | None,
    planner_errors: list[str],
    asset_errors: list[str],
    adaptive_ipc_errors: list[str],
    source_violations: list[str],
) -> bool:
    if not codex_result.success or worker_report is None or worker_error:
        return False
    return timing is not None and not planner_errors and not asset_errors and not adaptive_ipc_errors and not source_violations


def case_summary(
    *,
    case: Case,
    case_dir: Path,
    codex_result: CodexExecResult,
    worker_report: dict[str, Any] | None,
    worker_error: str | None,
    planner_errors: list[str],
    timing: BaselineTimingPlan | None,
    timing_error: str | None,
    asset_report: dict[str, Any] | None,
    asset_errors: list[str],
    adaptive_ipc_report: dict[str, Any] | None,
    adaptive_ipc_errors: list[str],
    source_violations: list[str],
    execution: ExecutionReport | None,
) -> dict[str, Any]:
    execution_ok = bool(execution and execution.ok)
    worker_status = worker_report.get("status") if isinstance(worker_report, dict) else None
    worker_status_warning = (
        None
        if worker_status in (None, "completed")
        else f"worker_report_status_not_completed: {worker_status}"
    )
    preflight_errors = [
        *([] if worker_error is None else [worker_error]),
        *planner_errors,
        *([] if timing_error is None else [timing_error]),
        *asset_errors,
        *adaptive_ipc_errors,
        *source_violations,
    ]
    verdict = "pass" if execution_ok else "fail"
    stop_reason = "execution_passed" if execution_ok else _stop_reason(codex_result, preflight_errors, execution)
    return {
        "case_id": case.case_id,
        "verdict": verdict,
        "outcome_class": "baseline_execution_pass" if execution_ok else "baseline_execution_fail",
        "execution_ok": execution_ok,
        "retry_recommended": False,
        "recommended_owner": "none",
        "repair_attempts": 0,
        "case_dir": str(case_dir),
        "timing": None if timing is None else timing.to_dict(),
        "asset_manifest_path": str(case_dir / "assets" / "asset_manifest.json")
        if (case_dir / "assets" / "asset_manifest.json").exists()
        else None,
        "asset_generation_report_path": None if asset_report is None else asset_report.get("asset_generation_report_path"),
        "agent_tool_history_path": str(case_dir / "reports" / "baseline_agent_tool_history.jsonl")
        if (case_dir / "reports" / "baseline_agent_tool_history.jsonl").exists()
        else None,
        "episode_state_path": None,
        "planner_actions_path": None,
        "dispatch_history_path": None,
        "codex": {
            "ok": codex_result.success,
            "returncode": codex_result.exit_code,
            "duration_sec": codex_result.duration_sec,
            "final_message_path": codex_result.final_message_path,
            "stderr_path": codex_result.stderr_path,
            "error_type": codex_result.error_type,
            "error_message": codex_result.error_message,
            "account": codex_result.codex_account_name,
        },
        "worker_status": worker_status,
        "worker_status_warning": worker_status_warning,
        "worker_report": worker_report,
        "worker_error": worker_error,
        "planner_errors": planner_errors,
        "timing_error": timing_error,
        "asset_errors": asset_errors,
        "adaptive_ipc_report": adaptive_ipc_report,
        "adaptive_ipc_errors": adaptive_ipc_errors,
        "source_violations": source_violations,
        "asset_report": asset_report,
        "execution": None if execution is None else execution.to_dict(),
        "stop_reason": stop_reason,
    }


def _stop_reason(
    codex_result: CodexExecResult,
    preflight_errors: list[str],
    execution: ExecutionReport | None,
) -> str:
    if not codex_result.success:
        return f"codex_failed: {codex_result.error_type or codex_result.exit_code}"
    if preflight_errors:
        return "preflight_failed: " + "; ".join(preflight_errors[:5])
    if execution is None:
        return "execution_not_run"
    return f"execution_failed: returncode={execution.returncode}"


def suite_summary(
    base: dict[str, Any],
    *,
    results: list[dict[str, Any]],
    num_cases_total: int | None = None,
    started_at_unix: float | None = None,
    completed_at_unix: float | None = None,
) -> dict[str, Any]:
    summary = dict(base)
    if started_at_unix is not None:
        summary["started_at_unix"] = started_at_unix
    if completed_at_unix is not None:
        summary["completed_at_unix"] = completed_at_unix
        if started_at_unix is not None:
            summary["suite_duration_sec"] = completed_at_unix - started_at_unix
    summary["num_cases"] = len(results) if num_cases_total is None else num_cases_total
    summary["num_completed"] = len(results)
    summary["num_passed"] = sum(1 for item in results if item.get("verdict") == "pass")
    summary["num_failed"] = sum(1 for item in results if item.get("verdict") == "fail")
    summary["num_inconclusive"] = sum(1 for item in results if item.get("verdict") == "inconclusive")
    summary["num_infra_blocked"] = sum(1 for item in results if item.get("outcome_class") == "infra_blocked")
    summary["num_semantic_inconclusive"] = sum(
        1 for item in results if item.get("outcome_class") == "semantic_inconclusive"
    )
    summary["retry_candidates"] = [
        item.get("case_id")
        for item in results
        if item.get("retry_recommended") and isinstance(item.get("case_id"), str)
    ]
    summary["results"] = results
    return summary


def case_exception_summary(*, case: Case, case_dir: Path, exc: Exception) -> dict[str, Any]:
    reports_dir = case_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    traceback_path = reports_dir / "case_exception.txt"
    traceback_path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")
    summary = {
        "case_id": case.case_id,
        "verdict": "fail",
        "outcome_class": "fail",
        "execution_ok": False,
        "retry_recommended": False,
        "recommended_owner": "none",
        "repair_attempts": 0,
        "case_dir": str(case_dir),
        "timing": None,
        "asset_manifest_path": None,
        "episode_state_path": None,
        "planner_actions_path": None,
        "dispatch_history_path": None,
        "stop_reason": f"case_exception: {type(exc).__name__}: {exc}",
        "exception_path": str(traceback_path),
    }
    dump_json(summary, case_dir / "summary.json")
    return summary


def missing_case_summary(case: Case, case_dir: Path) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "verdict": "fail",
        "outcome_class": "fail",
        "execution_ok": False,
        "retry_recommended": False,
        "recommended_owner": "none",
        "repair_attempts": 0,
        "case_dir": str(case_dir),
        "timing": None,
        "asset_manifest_path": None,
        "episode_state_path": None,
        "planner_actions_path": None,
        "dispatch_history_path": None,
        "stop_reason": "case did not produce a result",
    }
