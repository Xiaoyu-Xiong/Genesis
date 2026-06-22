from __future__ import annotations

import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS
from code_agent.io_utils import load_json_object
from code_agent.opt.objective import ObjectiveScore
from code_agent.opt.contracts import (
    OptContractError,
    OptContracts,
    load_opt_contracts,
    write_params_payload,
)
from code_agent.opt.optimizers.cma_es import default_population_size
from code_agent.opt.parallel_policy import resolve_parallel_policy
from code_agent.opt.reports import OptReporter
from code_agent.opt.search import CMAESStrategyRunner
from code_agent.opt.strategy import choose_best, maximize, resolve_strategy, strategy_report
from code_agent.opt.trials import RunOptOptions, TrialExecutor, TrialResult, payload_for_trial


@dataclass(slots=True, frozen=True)
class RunOptConfig:
    case_dir: Path
    target_spec_path: Path | None = None
    opt_space_path: Path | None = None
    default_params_path: Path | None = None
    backend: str | None = None
    max_trials: int | None = None
    population_size: int | None = None
    seed: int | None = None
    timeout_sec: float | None = None
    steps: int | None = None
    duration_sec: float | None = None
    render_fps: int | None = None
    render_best: bool | None = None
    main_file: str = CONFIGS.opt.runner_main_file


def run_optimization(config: RunOptConfig) -> dict[str, Any]:
    """Run the low-level numerical optimizer for one generated case workspace."""

    runner = _OptRunner(config)
    try:
        return runner.run()
    except OptContractError as exc:
        return runner.write_failed_report(str(exc))


class _OptRunner:
    def __init__(self, config: RunOptConfig) -> None:
        self.config = config
        self.case_dir = config.case_dir.resolve()
        self.contracts_dir = self.case_dir / "contracts"
        self.reporter = OptReporter(self.case_dir)
        self.trials = TrialExecutor(
            case_dir=self.case_dir,
            contracts_dir=self.contracts_dir,
            reports_dir=self.reporter.reports_dir,
            main_file=config.main_file,
        )

    def run(self) -> dict[str, Any]:
        contracts = load_opt_contracts(
            case_dir=self.case_dir,
            target_spec_path=self.config.target_spec_path,
            opt_space_path=self.config.opt_space_path,
            default_params_path=self.config.default_params_path,
        )
        options = self._resolve_options(contracts)
        strategy = resolve_strategy(contracts.opt_space)
        self.reporter.prepare()
        self._clean_previous_run_artifacts(options)

        best_result, baseline_scores, next_trial_index = self._run_baselines(contracts, options)
        search = CMAESStrategyRunner(
            trials=self.trials,
            trace_callback=self.reporter.append_trace,
            warning_callback=self.reporter.warnings.append,
        ).run(
            contracts=contracts,
            options=options,
            strategy=strategy,
            trial_index=next_trial_index,
            best_result=best_result,
        )
        best_result = search.best_result

        best_result, _ = self._confirm_best_result(
            contracts,
            options,
            best_result,
            search.next_trial_index,
        )
        best_render_dir = self._write_and_render_best(best_result, options)
        verification = self.reporter.write_verification(best_result, best_render_dir, contracts.target_spec)
        return self.reporter.write_report(
            contracts=contracts,
            options=options,
            baseline_scores=baseline_scores,
            best_result=best_result,
            best_render_dir=best_render_dir,
            verification_path=self.reporter.verification_report_path if verification else None,
            default_initial_sigma=CONFIGS.opt.runner_default_initial_sigma,
            strategy_report=strategy_report(strategy, search.phase_reports, search.stop_reason),
        )

    def write_failed_report(self, failure: str) -> dict[str, Any]:
        return self.reporter.write_failed_report(failure)

    def _clean_previous_run_artifacts(self, options: RunOptOptions) -> None:
        for path in (options.trial_root, options.best_out_dir):
            if path.resolve() == self.case_dir:
                continue
            shutil.rmtree(path, ignore_errors=True)

    def _run_baselines(
        self,
        contracts: OptContracts,
        options: RunOptOptions,
    ) -> tuple[TrialResult | None, list[float], int]:
        best_result: TrialResult | None = None
        baseline_scores: list[float] = []
        trial_index = 0
        for _ in range(options.baseline_trials):
            baseline_payload = payload_for_trial(
                contracts.default_params,
                source="trial",
                trial_index=trial_index,
                metadata={"kind": "baseline"},
            )
            result = self.trials.run_trial(
                trial_index=trial_index,
                params_payload=baseline_payload,
                options=options,
                contracts=contracts,
            )
            self.reporter.append_trace(result.entry)
            baseline_scores.append(_score_for_report(result, maximize=maximize(contracts)))
            best_result = choose_best(best_result, result, maximize=maximize(contracts))
            trial_index += 1
        return best_result, baseline_scores, trial_index

    def _confirm_best_result(
        self,
        contracts: OptContracts,
        options: RunOptOptions,
        best_result: TrialResult | None,
        trial_index: int,
    ) -> tuple[TrialResult | None, int]:
        if best_result is None or options.best_repeat_trials <= 1:
            return best_result, trial_index

        results = [best_result]
        source_trial = int(best_result.entry["trial_index"])
        for repeat_index in range(options.best_repeat_trials - 1):
            payload = payload_for_trial(
                best_result.params_payload,
                source="trial",
                trial_index=trial_index,
                metadata={
                    "kind": "best_confirmation",
                    "source_trial": source_trial,
                    "repeat_index": repeat_index,
                },
            )
            result = self.trials.run_trial(
                trial_index=trial_index,
                params_payload=payload,
                options=options,
                contracts=contracts,
            )
            self.reporter.append_trace(result.entry)
            results.append(result)
            trial_index += 1

        return self._aggregate_confirmed_best(results), trial_index

    def _aggregate_confirmed_best(self, results: list[TrialResult]) -> TrialResult:
        scored = [
            result
            for result in results
            if isinstance(result.score.score, int | float) and math.isfinite(float(result.score.score))
        ]
        if not scored:
            return results[0]
        median_score = float(statistics.median(float(result.score.score) for result in scored))
        representative = min(scored, key=lambda result: abs(float(result.score.score) - median_score))
        success_count = sum(1 for result in results if result.score.success)
        success = success_count * 2 > len(results)
        warnings = [
            *representative.score.warnings,
            (
                f"best params confirmed with {len(results)} rollout(s); median_score={median_score:.6g}; "
                f"success_count={success_count}/{len(results)}"
            ),
        ]
        score = ObjectiveScore(
            score=median_score,
            success=success,
            terms=representative.score.terms,
            measured=representative.score.measured,
            failure_penalty=representative.score.failure_penalty,
            failure_reason=representative.score.failure_reason,
            warnings=warnings,
        )
        entry = dict(representative.entry)
        entry["score"] = score.score
        entry["objective"] = score.to_report()
        entry["warnings"] = warnings
        return TrialResult(entry=entry, score=score, params_payload=representative.params_payload)

    def _write_and_render_best(self, best_result: TrialResult | None, options: RunOptOptions) -> Path | None:
        if best_result is None:
            return None
        best_payload = payload_for_trial(
            best_result.params_payload,
            source="best",
            trial_index=int(best_result.entry["trial_index"]),
            metadata={"selected_by": "code_agent.opt", "score": best_result.score.score},
        )
        write_params_payload(self.reporter.best_params_path, best_payload)
        write_params_payload(options.current_params_path, best_payload)
        if not options.render_best:
            return None
        self.trials.run_best_render(best_payload, options)
        return options.best_out_dir

    def _resolve_options(self, contracts: OptContracts) -> RunOptOptions:
        budget = contracts.opt_space.get("budget", {})
        execution = contracts.opt_space.get("execution", {}) or {}
        timing = load_json_object(self.contracts_dir / "timing.json") or {}
        backend = self.config.backend or budget.get("backend") or CONFIGS.harness.default_backend
        max_trials = int(self.config.max_trials if self.config.max_trials is not None else budget["max_trials"])
        population_size = (
            self.config.population_size if self.config.population_size is not None else budget.get("population_size")
        )
        if population_size is None:
            population_size = default_population_size(len(contracts.active_variables))
        seed = self.config.seed if self.config.seed is not None else budget.get("seed")
        timeout_sec = float(
            self.config.timeout_sec
            if self.config.timeout_sec is not None
            else budget.get("timeout_sec") or CONFIGS.opt.runner_timeout_sec
        )
        steps = _first_int(self.config.steps, execution.get("steps"), timing.get("steps"))
        duration_sec = _first_float(self.config.duration_sec, execution.get("duration_sec"), timing.get("duration_sec"))
        render_fps = _first_int(self.config.render_fps, execution.get("render_fps"), timing.get("render_fps"))
        sim_dt = _first_float(execution.get("sim_dt"), timing.get("sim_dt"))
        sim_substeps = _first_int(execution.get("sim_substeps"), timing.get("sim_substeps"))
        render_every_n_steps = _first_int(
            execution.get("render_every_n_steps"),
            timing.get("render_every_n_steps"),
        )
        render_res = _first_render_res(execution.get("render_res"), timing.get("render_res"))
        target_video_frames = _first_int(None, timing.get("target_video_frames"))
        render_best = (
            self.config.render_best
            if self.config.render_best is not None
            else bool(budget.get("render_best", CONFIGS.opt.runner_render_best))
        )
        baseline_trials = max(1, int(budget.get("baseline_trials", CONFIGS.opt.runner_baseline_trials)))
        best_repeat_trials = max(1, int(budget.get("best_repeat_trials", CONFIGS.opt.runner_best_repeat_trials)))
        trial_root = self._workspace_path(
            execution.get("trial_root", CONFIGS.opt.runner_trial_root),
            "execution.trial_root",
        )
        best_out_dir = self._workspace_path(
            execution.get("best_out_dir", CONFIGS.opt.runner_best_out_dir),
            "execution.best_out_dir",
        )
        current_params_path = self._workspace_path(
            execution.get("params_path", CONFIGS.opt.runner_current_params_path),
            "execution.params_path",
        )
        return RunOptOptions(
            backend=str(backend),
            max_trials=max_trials,
            population_size=population_size,
            seed=seed,
            timeout_sec=timeout_sec,
            steps=steps,
            duration_sec=duration_sec,
            render_fps=render_fps,
            sim_dt=sim_dt,
            sim_substeps=sim_substeps,
            render_every_n_steps=render_every_n_steps,
            render_res=render_res,
            target_video_frames=target_video_frames,
            render_best=render_best,
            baseline_trials=baseline_trials,
            best_repeat_trials=best_repeat_trials,
            trial_root=trial_root,
            best_out_dir=best_out_dir,
            current_params_path=current_params_path,
            parallel_policy=resolve_parallel_policy(contracts.opt_space),
        )

    def _workspace_path(self, value: Any, field_name: str) -> Path:
        text = str(value)
        path = Path(text)
        if path.is_absolute() or ".." in path.parts:
            raise OptContractError(f"{field_name} must be a relative path inside the case workspace.")
        return self.case_dir / path


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        return int(value)
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        return float(value)
    return None


def _first_render_res(*values: Any) -> tuple[int, int] | None:
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        return (int(value[0]), int(value[1]))
    return None


def _score_for_report(result: TrialResult, *, maximize: bool) -> float:
    score = result.score.score
    if isinstance(score, int | float) and math.isfinite(float(score)):
        return float(score)
    return -1.0e12 if maximize else 1.0e12
