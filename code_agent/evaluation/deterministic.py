from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RENDER_SUFFIXES = (".mp4", ".mov", ".gif", ".png", ".jpg", ".jpeg", ".webp")
IPC_INITIAL_PENETRATION_REASON = "ipc.initial_penetration"
IPC_INITIAL_PENETRATION_OWNER = "body"
_IPC_ERROR_TERMS = ("ipc", "uipc", "libuipc", "sanity_check", "sanity check")
_PENETRATION_TERMS = (
    "initial penetration",
    "interpenetr",
    "penetrat",
    "intersection",
    "intersect",
    "thickness",
    "distance",
    "d_hat",
    "barrier",
)


@dataclass(slots=True, frozen=True)
class DeterministicEvaluationConfig:
    """Inputs for deterministic artifact checks after execution."""

    run_dir: Path
    execution_report_path: Path | None = None
    summary_path: Path | None = None
    metrics_path: Path | None = None
    render_path: Path | None = None
    output_path: Path | None = None
    require_successful_exit: bool = True
    require_render: bool = False


def evaluate_artifacts(config: DeterministicEvaluationConfig) -> dict[str, Any]:
    """Validate execution artifacts and write an artifact evaluation report.

    This deterministic evaluator parses JSON files, checks expected artifact presence, and emits repair-ready issue
    records without using an LLM or running Genesis.
    """

    run_dir = config.run_dir.resolve()
    output_path = (config.output_path or (run_dir / "artifact_evaluation.json")).resolve()
    execution_report_path = (config.execution_report_path or (run_dir / "execution_report.json")).resolve()

    checks: list[dict[str, Any]] = []
    execution_report = _load_json_check("execution_report", execution_report_path, checks)
    artifact_paths = _artifact_paths(execution_report)

    if config.require_successful_exit:
        _check_exit_code(execution_report, checks)
    repair_hint = _classify_repair_hint(execution_report, run_dir, checks)

    summary_path = _resolve_artifact_path(config.summary_path, run_dir, artifact_paths, "summary.json")
    metrics_path = _resolve_artifact_path(config.metrics_path, run_dir, artifact_paths, "metrics.json")
    render_path = _resolve_render_path(config.render_path, run_dir, artifact_paths)

    summary = _load_json_check("summary", summary_path, checks)
    metrics = _load_json_check("metrics", metrics_path, checks)
    _check_json_object("summary", summary, checks)
    _check_json_object("metrics", metrics, checks)
    _check_metrics_values(metrics, checks)
    _check_render(render_path, checks, required=config.require_render or config.render_path is not None)

    passed = all(check["status"] in {"pass", "skip"} for check in checks)
    report = {
        "evaluator": "artifact_checks",
        "schema_version": 1,
        "run_dir": str(run_dir),
        "execution_report_path": str(execution_report_path),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "metrics_path": str(metrics_path) if metrics_path is not None else None,
        "render_path": str(render_path) if render_path is not None else None,
        "passed": passed,
        "checks": checks,
        "failure_classes": _failure_classes(checks),
        "recommended_owner": repair_hint["recommended_owner"],
        "repair_summary": repair_hint["repair_summary"],
        "created_at_unix": time.time(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _load_json_check(label: str, path: Path | None, checks: list[dict[str, Any]]) -> Any:
    if path is None:
        checks.append(_check(label, "fail", f"{label}.missing", f"{label} path was not found."))
        return None
    if not path.is_file():
        checks.append(_check(label, "fail", f"{label}.missing", f"{path} does not exist."))
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        checks.append(_check(label, "fail", f"{label}.invalid_json", f"{path} is not valid JSON: {exc}"))
        return None
    checks.append(_check(label, "pass", None, f"{path} parsed successfully."))
    return payload


def _check_exit_code(execution_report: Any, checks: list[dict[str, Any]]) -> None:
    if not isinstance(execution_report, dict):
        checks.append(_check("exit_code", "fail", "execution_report.unavailable", "Execution report is unavailable."))
        return
    exit_code = execution_report.get("exit_code")
    timed_out = execution_report.get("timed_out", False)
    if exit_code == 0 and timed_out is False:
        checks.append(_check("exit_code", "pass", None, "Execution exited successfully."))
        return
    reason = "execution.timeout" if timed_out else "execution.nonzero_exit"
    checks.append(_check("exit_code", "fail", reason, f"exit_code={exit_code}, timed_out={timed_out}."))


def _check_json_object(label: str, payload: Any, checks: list[dict[str, Any]]) -> None:
    if payload is None:
        return
    if isinstance(payload, dict):
        checks.append(_check(f"{label}_schema", "pass", None, f"{label} is a JSON object."))
    else:
        checks.append(_check(f"{label}_schema", "fail", f"{label}.not_object", f"{label} must be a JSON object."))


def _check_metrics_values(metrics: Any, checks: list[dict[str, Any]]) -> None:
    if not isinstance(metrics, dict):
        return
    invalid_keys = [
        key
        for key, value in metrics.items()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    ]
    if invalid_keys:
        checks.append(
            _check(
                "metrics_values",
                "fail",
                "metrics.non_finite",
                f"Metrics contain non-finite float values: {', '.join(sorted(invalid_keys))}.",
            )
        )
    else:
        checks.append(_check("metrics_values", "pass", None, "Metrics contain no non-finite float values."))


def _check_render(path: Path | None, checks: list[dict[str, Any]], required: bool) -> None:
    if path is None:
        status = "fail" if required else "skip"
        reason = "render.missing" if required else None
        checks.append(_check("render", status, reason, "No render artifact was provided or discovered."))
        return
    if not path.is_file():
        checks.append(_check("render", "fail", "render.missing", f"{path} does not exist."))
        return
    if path.stat().st_size <= 0:
        checks.append(_check("render", "fail", "render.empty", f"{path} is empty."))
        return
    checks.append(_check("render", "pass", None, f"{path} exists and is non-empty."))


def _classify_repair_hint(execution_report: Any, run_dir: Path, checks: list[dict[str, Any]]) -> dict[str, str | None]:
    if not isinstance(execution_report, dict):
        return {"recommended_owner": "none", "repair_summary": None}
    exit_code = execution_report.get("exit_code")
    if exit_code == 0 and not execution_report.get("timed_out", False):
        return {"recommended_owner": "none", "repair_summary": None}

    output_text = "\n".join(
        text
        for text in (
            _read_execution_text(execution_report.get("stderr_path"), run_dir),
            _read_execution_text(execution_report.get("stdout_path"), run_dir),
            str(execution_report.get("error", "")),
            str(execution_report.get("exception", "")),
        )
        if text
    ).lower()
    if not output_text:
        return {"recommended_owner": "none", "repair_summary": None}

    ipc_seen = any(term in output_text for term in _IPC_ERROR_TERMS)
    penetration_seen = any(term in output_text for term in _PENETRATION_TERMS)
    if not (ipc_seen and penetration_seen):
        return {"recommended_owner": "none", "repair_summary": None}

    summary = (
        "libuipc/IPC reported an initial penetration, intersection, thickness, distance, or sanity-check failure. "
        "Repair body.py by adjusting initial poses, scales, spacing, and container dimensions for FEM/generated-mesh/"
        "rigid IPC bodies so collision surfaces do not overlap and have positive clearance before gravity or actions "
        "produce compression. If the same log later reports 'IPC rigid state accessor feature is unavailable', treat "
        "that accessor message as a secondary consequence of the invalid IPC world unless it is reproduced without "
        "initial-geometry or 'World is not valid' diagnostics."
    )
    checks.append(_check("ipc_initial_penetration", "fail", IPC_INITIAL_PENETRATION_REASON, summary))
    return {"recommended_owner": IPC_INITIAL_PENETRATION_OWNER, "repair_summary": summary}


def _read_execution_text(path_value: Any, run_dir: Path) -> str:
    paths: list[Path] = []
    if isinstance(path_value, str) and path_value:
        path = Path(path_value)
        paths.append(path if path.is_absolute() else (run_dir / path))
    paths.extend([run_dir / "reports" / "stderr.txt", run_dir / "reports" / "stdout.txt"])
    for path in paths:
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return ""


def _artifact_paths(execution_report: Any) -> list[Path]:
    if not isinstance(execution_report, dict):
        return []
    paths = execution_report.get("artifact_paths", [])
    if not isinstance(paths, list):
        return []
    return [Path(path) for path in paths if isinstance(path, str)]


def _resolve_artifact_path(
    explicit_path: Path | None,
    run_dir: Path,
    artifact_paths: list[Path],
    file_name: str,
) -> Path | None:
    if explicit_path is not None:
        return explicit_path.resolve() if explicit_path.is_absolute() else (run_dir / explicit_path).resolve()
    direct = (run_dir / file_name).resolve()
    if direct.is_file():
        return direct
    for path in artifact_paths:
        if path.name == file_name:
            return path.resolve()
    return direct


def _resolve_render_path(explicit_path: Path | None, run_dir: Path, artifact_paths: list[Path]) -> Path | None:
    if explicit_path is not None:
        return explicit_path.resolve() if explicit_path.is_absolute() else (run_dir / explicit_path).resolve()
    for path in artifact_paths:
        if path.suffix.lower() in DEFAULT_RENDER_SUFFIXES:
            return path.resolve()
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in DEFAULT_RENDER_SUFFIXES:
            return path.resolve()
    return None


def _failure_classes(checks: list[dict[str, Any]]) -> list[str]:
    return sorted({check["reason"] for check in checks if check["status"] == "fail" and check.get("reason")})


def _check(name: str, status: str, reason: str | None, message: str) -> dict[str, Any]:
    return {"name": name, "status": status, "reason": reason, "message": message}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic checks for generated Genesis artifacts.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--execution-report", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-nonzero-exit", action="store_true")
    parser.add_argument("--require-render", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = evaluate_artifacts(
        DeterministicEvaluationConfig(
            run_dir=args.run_dir,
            execution_report_path=args.execution_report,
            summary_path=args.summary,
            metrics_path=args.metrics,
            render_path=args.render,
            output_path=args.output,
            require_successful_exit=not args.allow_nonzero_exit,
            require_render=args.require_render,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
