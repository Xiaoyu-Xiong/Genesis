from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from code_agent.configs import CONFIGS
from code_agent.opt.contracts import OptContracts, OptVariable, normalized_initial_sigma
from code_agent.opt.trials import TrialResult


@dataclass(slots=True, frozen=True)
class EarlyStopConfig:
    enabled: bool
    patience_generations: int
    min_delta: float
    stop_on_success: bool


@dataclass(slots=True, frozen=True)
class RestartConfig:
    name: str
    max_trials: int | None
    sigma_scale: float
    seed: int | None
    population_size: int | None
    start_from_best: bool


@dataclass(slots=True, frozen=True)
class PhaseConfig:
    name: str
    groups: tuple[str, ...] | None
    variables: tuple[str, ...] | None
    max_trials: int | None
    sigma_scale: float
    population_size: int | None
    start_from_best: bool
    restarts: tuple[RestartConfig, ...] | None


@dataclass(slots=True, frozen=True)
class StrategyConfig:
    early_stop: EarlyStopConfig
    phases: tuple[PhaseConfig, ...]
    restarts: tuple[RestartConfig, ...]


def resolve_strategy(opt_space: dict[str, Any]) -> StrategyConfig:
    raw = opt_space.get("strategy")
    strategy = raw if isinstance(raw, dict) else {}
    early_stop = _early_stop_config(strategy.get("early_stop"))
    restarts = _restart_configs(strategy.get("restarts"), fallback_name="default")
    phases = _phase_configs(strategy.get("phases"), default_restarts=restarts)
    if not phases:
        phases = (
            PhaseConfig(
                name="all_variables",
                groups=None,
                variables=None,
                max_trials=None,
                sigma_scale=1.0,
                population_size=None,
                start_from_best=False,
                restarts=None,
            ),
        )
    return StrategyConfig(early_stop=early_stop, phases=phases, restarts=restarts)


def strategy_report(
    strategy: StrategyConfig,
    phase_reports: list[dict[str, Any]],
    stop_reason: str | None,
) -> dict[str, Any]:
    return {
        "early_stop": asdict(strategy.early_stop),
        "phases": phase_reports,
        "top_level_restarts": [asdict(restart) for restart in strategy.restarts],
        "stop_reason": stop_reason,
    }


def choose_best(current_best: TrialResult | None, candidate: TrialResult, *, maximize: bool) -> TrialResult:
    if current_best is None:
        return candidate
    if candidate.score.success and not current_best.score.success:
        return candidate
    if current_best.score.success and not candidate.score.success:
        return current_best
    candidate_score = candidate.score.score
    best_score = current_best.score.score
    if candidate_score is None:
        return current_best
    if best_score is None:
        return candidate
    if maximize:
        return candidate if candidate_score > best_score else current_best
    return candidate if candidate_score < best_score else current_best


class EarlyStopState:
    def __init__(self, config: EarlyStopConfig, maximize: bool, has_success_criteria: bool) -> None:
        self.config = config
        self.maximize = maximize
        self.has_success_criteria = has_success_criteria
        self.best_score: float | None = None
        self.stale_generations = 0

    def update(self, generation_best: TrialResult | None, generation_success: bool) -> str | None:
        if not self.config.enabled:
            return None
        if self.config.stop_on_success and self.has_success_criteria and generation_success:
            return "success_criteria_satisfied"
        score = generation_best.score.score if generation_best is not None else None
        if score is None:
            self.stale_generations += 1
        elif self.best_score is None or self._improved(float(score), self.best_score):
            self.best_score = float(score)
            self.stale_generations = 0
        else:
            self.stale_generations += 1
        if self.stale_generations >= self.config.patience_generations:
            return f"no_score_improvement_for_{self.stale_generations}_generations"
        return None

    def _improved(self, score: float, best_score: float) -> bool:
        delta = self.config.min_delta
        if self.maximize:
            return score > best_score + delta
        return score < best_score - delta


def initial_sigmas(
    variables: tuple[OptVariable, ...],
    phase: PhaseConfig,
    restart: RestartConfig,
) -> list[float]:
    return [
        min(
            1.0,
            max(
                1e-4,
                (normalized_initial_sigma(variable) or CONFIGS.opt.runner_default_initial_sigma)
                * phase.sigma_scale
                * restart.sigma_scale,
            ),
        )
        for variable in variables
    ]


def maximize(contracts: OptContracts) -> bool:
    return contracts.target_spec["objective"]["direction"] == "maximize"


def has_success_criteria(contracts: OptContracts) -> bool:
    return bool(contracts.target_spec.get("success_criteria"))


def _early_stop_config(raw: Any) -> EarlyStopConfig:
    payload = raw if isinstance(raw, dict) else {}
    return EarlyStopConfig(
        enabled=bool(payload.get("enabled", CONFIGS.opt.runner_early_stop_enabled)),
        patience_generations=max(
            1,
            int(payload.get("patience_generations", CONFIGS.opt.runner_early_stop_patience_generations)),
        ),
        min_delta=max(0.0, float(payload.get("min_delta", CONFIGS.opt.runner_early_stop_min_delta))),
        stop_on_success=bool(payload.get("stop_on_success", CONFIGS.opt.runner_stop_on_success)),
    )


def _restart_configs(raw: Any, *, fallback_name: str) -> tuple[RestartConfig, ...]:
    items = raw if isinstance(raw, list) and raw else [{"name": fallback_name}]
    restarts: list[RestartConfig] = []
    for index, item in enumerate(items):
        payload = item if isinstance(item, dict) else {}
        restarts.append(
            RestartConfig(
                name=str(payload.get("name") or f"{fallback_name}_{index}"),
                max_trials=_optional_positive_int(payload.get("max_trials")),
                sigma_scale=max(1e-4, float(payload.get("sigma_scale", 1.0))),
                seed=_optional_int(payload.get("seed")),
                population_size=_optional_population(payload.get("population_size")),
                start_from_best=bool(payload.get("start_from_best", False)),
            )
        )
    return tuple(restarts)


def _phase_configs(
    raw: Any,
    *,
    default_restarts: tuple[RestartConfig, ...],
) -> tuple[PhaseConfig, ...]:
    if not isinstance(raw, list):
        return ()
    phases: list[PhaseConfig] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        phases.append(
            PhaseConfig(
                name=str(item.get("name") or f"phase_{index}"),
                groups=_optional_str_tuple(item.get("groups")),
                variables=_optional_str_tuple(item.get("variables")),
                max_trials=_optional_positive_int(item.get("max_trials")),
                sigma_scale=max(1e-4, float(item.get("sigma_scale", 1.0))),
                population_size=_optional_population(item.get("population_size")),
                start_from_best=bool(item.get("start_from_best", index > 0)),
                restarts=(
                    _restart_configs(item.get("restarts"), fallback_name=f"phase_{index}_restart")
                    if isinstance(item.get("restarts"), list)
                    else None
                ),
            )
        )
    return tuple(phases) or (
        PhaseConfig(
            name="all_variables",
            groups=None,
            variables=None,
            max_trials=None,
            sigma_scale=1.0,
            population_size=None,
            start_from_best=False,
            restarts=default_restarts,
        ),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    return max(1, int(value))


def _optional_population(value: Any) -> int | None:
    if value is None:
        return None
    return max(2, int(value))


def _optional_str_tuple(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    items = tuple(str(item) for item in value if isinstance(item, str) and item)
    return items or None
