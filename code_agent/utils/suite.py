from __future__ import annotations

import re
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS
from code_agent.assets.layout_reuse import prepare_layout_reusable_assets
from code_agent.context.genesis import GenesisContextPack, build_genesis_context_pack, install_genesis_context_pack
from code_agent.io_utils import dump_json
from code_agent.planner.session import PlannerSession, PlannerSessionConfig


@dataclass(slots=True)
class Case:
    case_id: str
    task: str
    layout_path: Path | None = None


_LAYOUT_DIRECTIVE_RE = re.compile(r"@layout\s+(?P<path>\S+)")


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(f"Invalid case line, expected case_id|prompt: {line}")
        case_id, task = line.split("|", 1)
        task, layout_path = _extract_layout_directive(task.strip(), base_dir=path.parent)
        cases.append(Case(case_id.strip(), task, layout_path=layout_path))
    return cases


def _extract_layout_directive(task: str, *, base_dir: Path) -> tuple[str, Path | None]:
    match = _LAYOUT_DIRECTIVE_RE.search(task)
    if match is None:
        return task, None

    raw_path = match.group("path")
    layout_path = Path(raw_path)
    if not layout_path.is_absolute():
        layout_path = base_dir / layout_path
    layout_path = layout_path.resolve()
    if not layout_path.is_file():
        raise ValueError(f"Layout file referenced by @layout does not exist: {layout_path}")

    cleaned_task = (task[: match.start()] + task[match.end() :]).strip()
    if not cleaned_task:
        raise ValueError(f"Case prompt became empty after removing @layout directive for {layout_path}")
    return cleaned_task, layout_path


def _write_case_inputs(
    case_dir: Path,
    case: Case,
) -> None:
    inputs = case_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "user_prompt.md").write_text(case.task + "\n", encoding="utf-8")
    if case.layout_path is not None:
        layout_text = case.layout_path.read_text(encoding="utf-8", errors="replace")
        layout_copy = inputs / f"layout{case.layout_path.suffix or '.txt'}"
        layout_copy.write_text(layout_text, encoding="utf-8")
        layout_asset_report = prepare_layout_reusable_assets(case_dir=case_dir, layout_path=case.layout_path)
        layout_asset_lines: list[str] = []
        if layout_asset_report is not None:
            layout_asset_lines = [
                "",
                "Reusable layout assets:",
                f"- Status: `{layout_asset_report.get('status')}`",
                f"- Partial manifest: `{layout_asset_report.get('asset_manifest_path')}`",
                f"- Report: `{layout_asset_report.get('asset_generation_report_path')}`",
                "- Planner and workers should reuse ready `repo_asset` manifest entries instead of regenerating the "
                "same mesh.",
            ]
        language = case.layout_path.suffix.lstrip(".") or "text"
        (inputs / "layout_context.md").write_text(
            "\n".join(
                [
                    "# User-Provided Layout",
                    "",
                    f"- Original layout path: `{case.layout_path}`",
                    f"- Workspace copy: `{layout_copy}`",
                    "",
                    "Planner and workers should treat this layout as explicit scene-structure input. Preserve the",
                    "layout's source-derived dimensions, initial placements, axes, timing, and stated approximations",
                    "unless they are impossible in Genesis; when adapting coordinate systems or asset scales, document",
                    "the mapping and keep the original contact/interlock intent.",
                    *layout_asset_lines,
                    "",
                    f"```{language}",
                    layout_text.rstrip(),
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    capability_line = (
        "Planner must choose the physics mode for this case. Use ordinary rigid mode with IPC off for normal rigid "
        "examples; use rigid IPC only for extreme/delicate/interlocking rigid contact; use FEM+IPC for any "
        "soft-body, deformable body, or FEM.Cloth cloth behavior. Deformable always forces IPC on."
    )
    (inputs / "capabilities.md").write_text(
        "Genesis local GPU code generation for rigid, articulated, mesh, texture, rendering, critic, and repair "
        f"workflows. {capability_line} Planner controls worker dispatch, execution, critic, and repair actions.\n",
        encoding="utf-8",
    )


def run_suite(
    *,
    tasks_file: Path,
    out_dir: Path,
    backend: str,
    max_cases: int | None,
    timeout_sec: float,
    render: bool,
    repair_rounds: int,
    max_parallel_cases: int | None = CONFIGS.harness.max_parallel_cases,
    steps: int | None = None,
    duration_sec: float | None = None,
    render_fps: int | None = None,
    opt_enabled: bool | None = None,
    summary_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, object]:
    tasks_file = tasks_file.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(tasks_file)
    if max_cases is not None:
        cases = cases[:max_cases]

    effective_opt_enabled = CONFIGS.opt.enabled if opt_enabled is None else bool(opt_enabled)
    genesis_context = build_genesis_context_pack(out_dir)
    max_workers = _resolve_max_parallel_cases(num_cases=len(cases), max_parallel_cases=max_parallel_cases)
    started_at = time.time()
    results_by_index: list[dict[str, Any] | None] = [None] * len(cases)

    summary_base = {
        "tasks_file": str(tasks_file),
        "out_dir": str(out_dir),
        "backend": backend,
        "render": render,
        "steps": steps,
        "duration_sec": duration_sec,
        "render_fps": render_fps,
        "physics_selection": "planner_decides",
        "opt_enabled": effective_opt_enabled,
        "max_parallel_cases": max_workers,
        "genesis_execution_lock": {
            "scope": "all run_generated_simulation calls in this Python process, plus a per-user /tmp file lock",
            "allows_parallel_case_planning": True,
            "serializes_local_genesis_execution": True,
        },
        "genesis_context_path": str(genesis_context.markdown_path),
        "genesis_context_json_path": str(genesis_context.json_path),
        "genesis_docs_dir": str(genesis_context.docs_dir),
    }

    if not cases:
        summary = _suite_summary(summary_base, results=[])
        dump_json(summary, out_dir / "summary.json")
        if summary_callback is not None:
            summary_callback(summary)
        return summary

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="code-agent-case") as executor:
        futures = {
            executor.submit(
                _run_case,
                case=case,
                out_dir=out_dir,
                genesis_context=genesis_context,
                backend=backend,
                timeout_sec=timeout_sec,
                render=render,
                repair_rounds=repair_rounds,
                steps=steps,
                duration_sec=duration_sec,
                render_fps=render_fps,
                opt_enabled=effective_opt_enabled,
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
                result = _case_exception_summary(case=case, case_dir=case_dir, exc=exc)
            results_by_index[index] = result
            partial_results = [item for item in results_by_index if item is not None]
            partial_summary = _suite_summary(
                summary_base,
                results=partial_results,
                num_cases_total=len(cases),
                completed_at_unix=None,
                started_at_unix=started_at,
            )
            dump_json(partial_summary, out_dir / "summary.json")
            if summary_callback is not None:
                summary_callback(partial_summary)

    results = [
        item if item is not None else _missing_case_summary(cases[index], out_dir / cases[index].case_id)
        for index, item in enumerate(results_by_index)
    ]
    summary = _suite_summary(
        summary_base,
        results=results,
        num_cases_total=len(cases),
        completed_at_unix=time.time(),
        started_at_unix=started_at,
    )
    dump_json(summary, out_dir / "summary.json")
    if summary_callback is not None:
        summary_callback(summary)
    return summary


def _run_case(
    *,
    case: Case,
    out_dir: Path,
    genesis_context: GenesisContextPack,
    backend: str,
    timeout_sec: float,
    render: bool,
    repair_rounds: int,
    steps: int | None,
    duration_sec: float | None,
    render_fps: int | None,
    opt_enabled: bool,
) -> dict[str, Any]:
    case_dir = out_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_case_inputs(case_dir, case)
    install_genesis_context_pack(case_dir, genesis_context)
    session = PlannerSession(
        PlannerSessionConfig(
            case_id=case.case_id,
            task=case.task,
            case_dir=case_dir,
            backend=backend,
            timeout_sec=timeout_sec,
            render=render,
            repair_rounds=repair_rounds,
            steps=steps,
            duration_sec=duration_sec,
            render_fps=render_fps,
            opt_enabled=opt_enabled,
        )
    )
    return session.run()


def _resolve_max_parallel_cases(*, num_cases: int, max_parallel_cases: int | None) -> int:
    if num_cases <= 0:
        return 1
    if max_parallel_cases is None:
        return num_cases
    return max(1, min(int(max_parallel_cases), num_cases))


def _suite_summary(
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


def _case_exception_summary(*, case: Case, case_dir: Path, exc: Exception) -> dict[str, Any]:
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
        "episode_state_path": str(reports_dir / "episode_state.json"),
        "planner_actions_path": str(reports_dir / "planner_actions.jsonl"),
        "dispatch_history_path": str(reports_dir / "dispatch_history.jsonl"),
        "stop_reason": f"case_exception: {type(exc).__name__}: {exc}",
        "exception_path": str(traceback_path),
    }
    dump_json(summary, case_dir / "summary.json")
    return summary


def _missing_case_summary(case: Case, case_dir: Path) -> dict[str, Any]:
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
        "episode_state_path": str(case_dir / "reports" / "episode_state.json"),
        "planner_actions_path": str(case_dir / "reports" / "planner_actions.jsonl"),
        "dispatch_history_path": str(case_dir / "reports" / "dispatch_history.jsonl"),
        "stop_reason": "case did not produce a result",
    }
