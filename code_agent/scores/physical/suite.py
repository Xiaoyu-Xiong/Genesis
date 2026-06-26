from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code_agent.io_utils import dump_json, load_json_object
from code_agent.scores.physical.agent import PhysicalScoreRequest, run_physical_score
from code_agent.utils.suite import load_cases


@dataclass(slots=True, frozen=True)
class PhysicalSuiteCase:
    case_id: str
    run_dir: Path
    prompt: str | None = None
    code_root: Path | None = None


def score_physical_suite(
    *,
    suite_dir: Path,
    tasks_file: Path | None = None,
    output_dir: Path | None = None,
    max_workers: int = 2,
    max_cases: int | None = None,
    model: str | None = None,
    timeout_sec: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Score every case under a suite directory with SBAR-v1 in parallel."""

    suite_dir = suite_dir.resolve()
    output_dir = (output_dir or suite_dir / "physical_scores").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    cases = _discover_suite_cases(suite_dir=suite_dir, tasks_file=tasks_file)
    if max_cases is not None:
        cases = cases[:max_cases]

    started_at = time.time()
    max_workers = max(1, min(int(max_workers), max(1, len(cases))))
    results_by_index: list[dict[str, Any] | None] = [None] * len(cases)
    summary_base = {
        "metric": "SBAR-v1",
        "suite_dir": str(suite_dir),
        "tasks_file": str(tasks_file.resolve()) if tasks_file else None,
        "output_dir": str(output_dir),
        "max_workers": max_workers,
        "force": force,
    }
    if not cases:
        summary = _summary(
            summary_base,
            results=[],
            num_cases_total=0,
            started_at_unix=started_at,
            completed_at_unix=time.time(),
        )
        dump_json(summary, summary_path)
        return summary

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="physical-score") as executor:
        futures = {
            executor.submit(
                _score_case,
                case=case,
                output_dir=output_dir,
                model=model,
                timeout_sec=timeout_sec,
                force=force,
            ): index
            for index, case in enumerate(cases)
        }
        for future in as_completed(futures):
            index = futures[future]
            case = cases[index]
            try:
                result = future.result()
            except Exception as exc:
                result = _case_exception(case, output_dir=output_dir, exc=exc)
            results_by_index[index] = result
            partial_results = [item for item in results_by_index if item is not None]
            partial_summary = _summary(
                summary_base,
                results=partial_results,
                num_cases_total=len(cases),
                started_at_unix=started_at,
                completed_at_unix=None,
            )
            dump_json(partial_summary, summary_path)

    results = [
        item if item is not None else _missing_case(cases[index], output_dir=output_dir)
        for index, item in enumerate(results_by_index)
    ]
    summary = _summary(
        summary_base,
        results=results,
        num_cases_total=len(cases),
        started_at_unix=started_at,
        completed_at_unix=time.time(),
    )
    dump_json(summary, summary_path)
    return summary


def _discover_suite_cases(*, suite_dir: Path, tasks_file: Path | None) -> list[PhysicalSuiteCase]:
    if tasks_file is not None:
        cases = []
        for case in load_cases(tasks_file.resolve()):
            cases.append(
                PhysicalSuiteCase(
                    case_id=case.case_id,
                    run_dir=suite_dir / case.case_id,
                    prompt=case.task,
                )
            )
        return cases

    summary = load_json_object(suite_dir / "summary.json")
    if isinstance(summary, dict) and isinstance(summary.get("results"), list):
        cases: list[PhysicalSuiteCase] = []
        for item in summary["results"]:
            if not isinstance(item, dict):
                continue
            case_id = item.get("case_id")
            case_dir = item.get("case_dir")
            if not isinstance(case_id, str):
                continue
            run_dir = Path(case_dir).resolve() if isinstance(case_dir, str) else (suite_dir / case_id).resolve()
            prompt = item.get("task") if isinstance(item.get("task"), str) else None
            cases.append(PhysicalSuiteCase(case_id=case_id, run_dir=run_dir, prompt=prompt))
        if cases:
            return cases

    ignored = {"context", "physical_scores", "logs", "reports", "artifacts", "assets", "src"}
    cases = []
    for path in sorted(item for item in suite_dir.iterdir() if item.is_dir() and item.name not in ignored):
        if _looks_like_case_dir(path):
            cases.append(PhysicalSuiteCase(case_id=path.name, run_dir=path))
    return cases


def _looks_like_case_dir(path: Path) -> bool:
    return any(
        candidate.exists()
        for candidate in (
            path / "inputs" / "user_prompt.md",
            path / "summary.json",
            path / "src",
            path / "artifacts",
        )
    )


def _score_case(
    *,
    case: PhysicalSuiteCase,
    output_dir: Path,
    model: str | None,
    timeout_sec: float | None,
    force: bool,
) -> dict[str, Any]:
    report_path = case.run_dir / "reports" / "physical_score_report.json"
    report = run_physical_score(
        PhysicalScoreRequest(
            run_dir=case.run_dir,
            prompt=case.prompt,
            code_root=case.code_root,
            case_id=case.case_id,
            output_path=report_path,
            model=model,
            timeout_sec=timeout_sec,
            force=force,
        )
    )
    result = _result_row(case=case, report=report, report_path=report_path)
    dump_json(result, output_dir / f"{case.case_id}.json")
    return result


def _result_row(*, case: PhysicalSuiteCase, report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "run_dir": str(case.run_dir),
        "report_path": str(report_path),
        "scorer_status": report.get("scorer_status", "unknown"),
        "overall_score": _number(report.get("overall_score")),
        "scene_score": _number(report.get("scene_score")),
        "body_score": _number(report.get("body_score")),
        "action_score": _number(report.get("action_score")),
        "render_score": _number(report.get("render_score")),
        "render_faithfulness": _number(report.get("render_faithfulness")),
        "render_aesthetic_quality": _number(report.get("render_aesthetic_quality")),
        "violation_penalty": _number(report.get("violation_penalty")),
        "caps_applied": report.get("caps_applied") if isinstance(report.get("caps_applied"), list) else [],
        "fatal_violations": report.get("fatal_violations") if isinstance(report.get("fatal_violations"), list) else [],
        "confidence": _number(report.get("confidence")),
        "summary": report.get("summary"),
    }


def _summary(
    base: dict[str, Any],
    *,
    results: list[dict[str, Any]],
    num_cases_total: int,
    started_at_unix: float,
    completed_at_unix: float | None,
) -> dict[str, Any]:
    summary = dict(base)
    summary["started_at_unix"] = started_at_unix
    if completed_at_unix is not None:
        summary["completed_at_unix"] = completed_at_unix
        summary["suite_duration_sec"] = completed_at_unix - started_at_unix
    summary["num_cases"] = num_cases_total
    summary["num_completed"] = len(results)
    summary["num_succeeded"] = sum(1 for item in results if item.get("scorer_status") in {"completed", "cached"})
    summary["num_failed"] = sum(1 for item in results if item.get("scorer_status") not in {"completed", "cached"})
    summary["averages"] = _averages(results)
    summary["results"] = results
    return summary


def _averages(results: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = (
        "overall_score",
        "scene_score",
        "body_score",
        "action_score",
        "render_score",
        "render_faithfulness",
        "render_aesthetic_quality",
        "violation_penalty",
        "confidence",
    )
    averages: dict[str, float | None] = {}
    for key in keys:
        values = [float(item[key]) for item in results if isinstance(item.get(key), int | float)]
        averages[key] = round(sum(values) / len(values), 4) if values else None
    return averages


def _case_exception(case: PhysicalSuiteCase, *, output_dir: Path, exc: Exception) -> dict[str, Any]:
    path = output_dir / f"{case.case_id}.exception.txt"
    path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")
    result = {
        "case_id": case.case_id,
        "run_dir": str(case.run_dir),
        "scorer_status": "failed",
        "overall_score": 0.0,
        "scene_score": 0.0,
        "body_score": 0.0,
        "action_score": 0.0,
        "render_score": 0.0,
        "error": f"{type(exc).__name__}: {exc}",
        "exception_path": str(path),
    }
    dump_json(result, output_dir / f"{case.case_id}.json")
    return result


def _missing_case(case: PhysicalSuiteCase, *, output_dir: Path) -> dict[str, Any]:
    result = {
        "case_id": case.case_id,
        "run_dir": str(case.run_dir),
        "scorer_status": "missing",
        "overall_score": 0.0,
        "scene_score": 0.0,
        "body_score": 0.0,
        "action_score": 0.0,
        "render_score": 0.0,
        "error": "case did not produce a scorer result",
    }
    dump_json(result, output_dir / f"{case.case_id}.json")
    return result


def _number(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
