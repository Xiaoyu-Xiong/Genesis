from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_agent.io_utils import dump_json
from code_agent.opt.contracts import OptContracts, normalized_initial_sigma
from code_agent.opt.trials import RunOptOptions, TrialResult


class OptReporter:
    def __init__(self, case_dir: Path) -> None:
        self.case_dir = case_dir
        self.contracts_dir = case_dir / "contracts"
        self.reports_dir = case_dir / "reports"
        self.trace_path = self.reports_dir / "opt_trace.jsonl"
        self.opt_report_path = self.reports_dir / "opt_report.json"
        self.verification_report_path = self.reports_dir / "verification_report.json"
        self.best_params_path = self.contracts_dir / "best_opt_params.json"
        self.trace_entries: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def prepare(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.contracts_dir.mkdir(parents=True, exist_ok=True)
        self.trace_entries.clear()
        self.trace_path.write_text("", encoding="utf-8")

    def write_failed_report(self, failure: str) -> dict[str, Any]:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "status": "failed",
            "optimizer": "cma_es",
            "num_trials": 0,
            "baseline_score": None,
            "best_trial": None,
            "best_score": None,
            "best_params_path": None,
            "best_render_dir": None,
            "trace_path": self.rel(self.trace_path),
            "verification_report_path": None,
            "budget": {},
            "summary": "Optimization contract validation failed before any rollout.",
            "failures": [failure],
            "warnings": [],
        }
        dump_json(report, self.opt_report_path)
        return report

    def append_trace(self, entry: dict[str, Any]) -> None:
        self.trace_entries.append(entry)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def write_verification(
        self,
        best_result: TrialResult | None,
        best_render_dir: Path | None,
        target_spec: dict[str, Any],
    ) -> dict[str, Any] | None:
        if best_result is None:
            return None
        verification = {
            "schema_version": 1,
            "success": bool(best_result.score.success),
            "score": best_result.score.score,
            "target": self._target_summary(best_result, target_spec),
            "measured": best_result.score.measured,
            "terms": best_result.score.terms,
            "best_trial": int(best_result.entry["trial_index"]),
            "best_params_path": self.rel(self.best_params_path),
            "best_render_dir": None if best_render_dir is None else self.rel(best_render_dir),
            "failure_reason": best_result.score.failure_reason,
            "warnings": best_result.score.warnings,
        }
        dump_json(verification, self.verification_report_path)
        return verification

    def write_report(
        self,
        *,
        contracts: OptContracts,
        options: RunOptOptions,
        baseline_scores: list[float],
        best_result: TrialResult | None,
        best_render_dir: Path | None,
        verification_path: Path | None,
        default_initial_sigma: float,
    ) -> dict[str, Any]:
        completed_count = sum(1 for entry in self.trace_entries if entry["status"] == "completed")
        status = "completed" if completed_count > 0 else "inconclusive"
        baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else None
        best_trial = None if best_result is None else int(best_result.entry["trial_index"])
        best_score = None if best_result is None else best_result.score.score
        direction = contracts.target_spec["objective"]["direction"]
        summary = self._summary(status, baseline_score, best_score, direction)
        report = {
            "schema_version": 1,
            "status": status,
            "optimizer": "cma_es",
            "num_trials": len(self.trace_entries),
            "baseline_score": baseline_score,
            "best_trial": best_trial,
            "best_score": best_score,
            "best_params_path": None if best_result is None else self.rel(self.best_params_path),
            "best_render_dir": None if best_render_dir is None else self.rel(best_render_dir),
            "trace_path": self.rel(self.trace_path),
            "verification_report_path": None if verification_path is None else self.rel(verification_path),
            "budget": {
                "max_trials": options.max_trials,
                "baseline_trials": options.baseline_trials,
                "population_size": options.population_size,
                "seed": options.seed,
                "default_initial_sigma": default_initial_sigma,
                "initial_sigmas": {
                    variable.name: normalized_initial_sigma(variable) or default_initial_sigma
                    for variable in contracts.active_variables
                },
                "backend": options.backend,
                "timeout_sec": options.timeout_sec,
                "render_best": options.render_best,
            },
            "summary": summary,
            "failures": self.failures,
            "warnings": self.warnings,
        }
        dump_json(report, self.opt_report_path)
        return report

    def rel(self, path: Path) -> str:
        path = path.resolve()
        try:
            return str(path.relative_to(self.case_dir))
        except ValueError:
            return str(path)

    def _target_summary(self, best_result: TrialResult, target_spec: dict[str, Any]) -> dict[str, Any]:
        goal = target_spec.get("goal")
        if isinstance(goal, dict):
            return goal
        target_values: dict[str, Any] = {}
        for term_name, term in best_result.score.terms.items():
            if isinstance(term, dict) and term.get("target") is not None:
                target_values[str(term_name)] = term["target"]
        return target_values

    def _summary(self, status: str, baseline_score: float | None, best_score: float | None, direction: str) -> str:
        if best_score is None:
            return "Optimization produced no usable score."
        if baseline_score is None:
            return f"Optimization {status}; best score is {best_score:.6g}."
        improved = best_score > baseline_score if direction == "maximize" else best_score < baseline_score
        label = "improved" if improved else "did not improve"
        return f"Optimization {status}; best score {label} over baseline ({best_score:.6g} vs {baseline_score:.6g})."
