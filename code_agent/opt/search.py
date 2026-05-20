from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from code_agent.configs import CONFIGS
from code_agent.opt.contracts import (
    OptContracts,
    OptVariable,
    payload_from_vector,
    vector_from_payload,
)
from code_agent.opt.optimizers.cma_es import CMAESOptimizer
from code_agent.opt.strategy import (
    EarlyStopState,
    PhaseConfig,
    RestartConfig,
    StrategyConfig,
    choose_best,
    has_success_criteria,
    initial_sigmas,
    maximize,
)
from code_agent.opt.trials import RunOptOptions, TrialExecutor, TrialResult


@dataclass(slots=True)
class SearchResult:
    best_result: TrialResult | None
    next_trial_index: int
    trials_used: int
    phase_reports: list[dict[str, Any]]
    stop_reason: str | None


@dataclass(slots=True)
class _PhaseRunResult:
    best_result: TrialResult | None
    next_trial_index: int
    trials_used: int
    stop_reason: str | None
    report: dict[str, Any]


class CMAESStrategyRunner:
    def __init__(self, *, trials: TrialExecutor, trace_callback, warning_callback) -> None:
        self.trials = trials
        self.trace_callback = trace_callback
        self.warning_callback = warning_callback

    def run(
        self,
        *,
        contracts: OptContracts,
        options: RunOptOptions,
        strategy: StrategyConfig,
        trial_index: int,
        best_result: TrialResult | None,
    ) -> SearchResult:
        remaining = options.max_trials
        phase_reports: list[dict[str, Any]] = []
        stop_reason: str | None = None
        for phase_index, phase in enumerate(strategy.phases):
            if remaining <= 0:
                break
            phase_variables = self._phase_variables(contracts.active_variables, phase)
            if not phase_variables:
                self.warning_callback(f"Optimization phase {phase.name!r} matched no active variables.")
                continue
            phase_budget = min(remaining, phase.max_trials or remaining)
            phase_base_payload = (
                best_result.params_payload
                if phase.start_from_best and best_result is not None
                else contracts.default_params
            )
            phase_result = self._run_phase(
                contracts=contracts,
                options=options,
                strategy=strategy,
                phase=phase,
                phase_index=phase_index,
                phase_variables=phase_variables,
                phase_budget=phase_budget,
                phase_base_payload=phase_base_payload,
                trial_index=trial_index,
                best_result=best_result,
            )
            best_result = phase_result.best_result
            trial_index = phase_result.next_trial_index
            remaining -= phase_result.trials_used
            phase_reports.append(phase_result.report)
            if phase_result.stop_reason is not None:
                stop_reason = phase_result.stop_reason
                if phase_result.stop_reason.startswith("success"):
                    break
        return SearchResult(
            best_result=best_result,
            next_trial_index=trial_index,
            trials_used=options.max_trials - remaining,
            phase_reports=phase_reports,
            stop_reason=stop_reason,
        )

    def _run_phase(
        self,
        *,
        contracts: OptContracts,
        options: RunOptOptions,
        strategy: StrategyConfig,
        phase: PhaseConfig,
        phase_index: int,
        phase_variables: tuple[OptVariable, ...],
        phase_budget: int,
        phase_base_payload: dict[str, Any],
        trial_index: int,
        best_result: TrialResult | None,
    ) -> _PhaseRunResult:
        phase_remaining = phase_budget
        trials_used = 0
        stop_reason: str | None = None
        restart_reports: list[dict[str, Any]] = []
        restarts = phase.restarts or strategy.restarts
        for restart_index, restart in enumerate(restarts):
            if phase_remaining <= 0:
                break
            restart_budget = self._restart_budget(restarts, restart, restart_index, phase_remaining)
            restart_base_payload = (
                best_result.params_payload
                if restart.start_from_best and best_result is not None
                else phase_base_payload
            )
            restart_result = self._run_restart(
                contracts=contracts,
                options=options,
                strategy=strategy,
                phase=phase,
                phase_index=phase_index,
                restart=restart,
                restart_index=restart_index,
                variables=phase_variables,
                restart_budget=restart_budget,
                base_payload=restart_base_payload,
                trial_index=trial_index,
                best_result=best_result,
            )
            best_result = restart_result.best_result
            trial_index = restart_result.next_trial_index
            phase_remaining -= restart_result.trials_used
            trials_used += restart_result.trials_used
            restart_reports.append(restart_result.report)
            if restart_result.stop_reason is not None:
                stop_reason = restart_result.stop_reason
                if restart_result.stop_reason.startswith("success"):
                    break
        return _PhaseRunResult(
            best_result=best_result,
            next_trial_index=trial_index,
            trials_used=trials_used,
            stop_reason=stop_reason,
            report={
                "name": phase.name,
                "variables": [variable.name for variable in phase_variables],
                "max_trials": phase_budget,
                "trials_used": trials_used,
                "stop_reason": stop_reason,
                "restarts": restart_reports,
            },
        )

    def _run_restart(
        self,
        *,
        contracts: OptContracts,
        options: RunOptOptions,
        strategy: StrategyConfig,
        phase: PhaseConfig,
        phase_index: int,
        restart: RestartConfig,
        restart_index: int,
        variables: tuple[OptVariable, ...],
        restart_budget: int,
        base_payload: dict[str, Any],
        trial_index: int,
        best_result: TrialResult | None,
    ) -> _PhaseRunResult:
        restart_initial_sigmas = initial_sigmas(variables, phase, restart)
        optimizer = CMAESOptimizer(
            dim=len(variables),
            mean=vector_from_payload(variables, base_payload),
            initial_sigmas=restart_initial_sigmas,
            population_size=restart.population_size or phase.population_size or options.population_size,
            seed=self._restart_seed(options.seed, restart.seed, phase_index, restart_index),
        )
        early_stop = EarlyStopState(strategy.early_stop, maximize(contracts), has_success_criteria(contracts))
        remaining = restart_budget
        trials_used = 0
        stop_reason: str | None = None
        while remaining > 0:
            count = min(optimizer.population_size, remaining)
            candidates = optimizer.ask(count=count)
            candidate_scores: list[float] = []
            generation_best = best_result
            generation_success = False
            for candidate in candidates:
                result = self._run_candidate(
                    contracts=contracts,
                    options=options,
                    variables=variables,
                    candidate=candidate,
                    base_payload=base_payload,
                    phase=phase,
                    restart=restart,
                    optimizer=optimizer,
                    trial_index=trial_index,
                )
                candidate_scores.append(float(result.score.score) if result.score.score is not None else 0.0)
                best_result = choose_best(best_result, result, maximize=maximize(contracts))
                generation_best = choose_best(generation_best, result, maximize=maximize(contracts))
                generation_success = generation_success or bool(result.score.success)
                trial_index += 1
                remaining -= 1
                trials_used += 1
            optimizer.tell(candidates[: len(candidate_scores)], candidate_scores, maximize=maximize(contracts))
            stop_reason = early_stop.update(generation_best, generation_success)
            if stop_reason is not None:
                self.warning_callback(f"Early stop in phase {phase.name!r}, restart {restart.name!r}: {stop_reason}.")
                break
        return _PhaseRunResult(
            best_result=best_result,
            next_trial_index=trial_index,
            trials_used=trials_used,
            stop_reason=stop_reason,
            report={
                "name": restart.name,
                "max_trials": restart_budget,
                "trials_used": trials_used,
                "stop_reason": stop_reason,
                "population_size": optimizer.population_size,
                "initial_sigmas": dict(
                    zip((variable.name for variable in variables), restart_initial_sigmas, strict=True)
                ),
            },
        )

    def _run_candidate(
        self,
        *,
        contracts: OptContracts,
        options: RunOptOptions,
        variables: tuple[OptVariable, ...],
        candidate: list[float],
        base_payload: dict[str, Any],
        phase: PhaseConfig,
        restart: RestartConfig,
        optimizer: CMAESOptimizer,
        trial_index: int,
    ) -> TrialResult:
        candidate_payload = payload_from_vector(
            variables,
            candidate,
            base_payload=base_payload,
            source="trial",
            trial_index=trial_index,
            metadata={
                "kind": "cma_es",
                "phase": phase.name,
                "restart": restart.name,
                "optimizer_state": asdict(optimizer.state()),
            },
        )
        result = self.trials.run_trial(
            trial_index=trial_index,
            params_payload=candidate_payload,
            options=options,
            contracts=contracts,
        )
        self.trace_callback(result.entry)
        return result

    def _phase_variables(
        self,
        active_variables: tuple[OptVariable, ...],
        phase: PhaseConfig,
    ) -> tuple[OptVariable, ...]:
        if phase.groups is None and phase.variables is None:
            return active_variables
        names = set(phase.variables or ())
        groups = set(phase.groups or ())
        return tuple(
            variable
            for variable in active_variables
            if variable.name in names or (variable.group is not None and variable.group in groups)
        )

    def _restart_budget(
        self,
        restarts: tuple[RestartConfig, ...],
        restart: RestartConfig,
        restart_index: int,
        phase_remaining: int,
    ) -> int:
        remaining_restarts = max(1, len(restarts) - restart_index)
        default_restart_budget = max(1, phase_remaining // remaining_restarts)
        return min(phase_remaining, restart.max_trials or default_restart_budget)

    def _restart_seed(
        self,
        base_seed: int | None,
        explicit_seed: int | None,
        phase_index: int,
        restart_index: int,
    ) -> int | None:
        if explicit_seed is not None:
            return explicit_seed
        if base_seed is None:
            return None
        offset = CONFIGS.opt.runner_restart_seed_stride
        return int(base_seed) + offset * (phase_index + 1) + restart_index

