from __future__ import annotations

import math
from dataclasses import dataclass

import cma
import numpy as np

from code_agent.configs import CONFIGS


@dataclass(slots=True, frozen=True)
class CMAESState:
    iteration: int
    mean: list[float]
    sigma: float
    axis_sigmas: list[float]
    best_score: float | None
    best_vector: list[float] | None


def default_population_size(dim: int) -> int:
    """Return the default CMA-ES population size for a normalized search space."""

    if dim < 1:
        raise ValueError("CMA-ES requires dim >= 1")
    opt_cfg = CONFIGS.opt
    canonical = opt_cfg.cma_es_population_base + math.floor(opt_cfg.cma_es_population_log_multiplier * math.log(dim))
    if dim <= opt_cfg.cma_es_low_dim_threshold:
        return max(opt_cfg.cma_es_low_dim_min_population, min(opt_cfg.cma_es_low_dim_max_population, canonical))
    return int(canonical)


class CMAESOptimizer:
    """Pycma-backed bounded CMA-ES over normalized vectors in [0, 1]."""

    def __init__(
        self,
        *,
        dim: int,
        mean: list[float] | tuple[float, ...],
        sigma: float | None = None,
        initial_sigmas: list[float] | tuple[float, ...] | None = None,
        population_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        if dim < 1:
            raise ValueError("CMA-ES requires dim >= 1")
        self.dim = int(dim)
        self.population_size = int(
            population_size if population_size is not None else default_population_size(self.dim)
        )
        if self.population_size < 3:
            raise ValueError("population_size must be >= 3 for pycma")

        start_mean = _unit_vector(mean, self.dim)
        if initial_sigmas is None:
            sigma0 = CONFIGS.opt.runner_default_initial_sigma if sigma is None else float(sigma)
            self._initial_axis_sigmas = np.full(self.dim, _clamp_sigma(sigma0), dtype=np.float64)
            cma_stds = None
        else:
            self._initial_axis_sigmas = _normalized_sigmas(initial_sigmas, self.dim)
            sigma0 = float(np.median(self._initial_axis_sigmas))
            cma_stds = self._initial_axis_sigmas / max(sigma0, 1e-12)
        sigma0 = _clamp_sigma(sigma0)

        options: dict[str, object] = {
            "bounds": [0.0, 1.0],
            "popsize": self.population_size,
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
        }
        if seed is not None:
            options["seed"] = int(seed)
        if cma_stds is not None:
            options["CMA_stds"] = cma_stds.astype(float).tolist()

        self._es = cma.CMAEvolutionStrategy(start_mean.astype(float).tolist(), sigma0, options)
        self._rng = np.random.default_rng(seed)
        self.best_score: float | None = None
        self.best_vector: np.ndarray | None = None

    def ask(self, count: int | None = None) -> list[list[float]]:
        count = int(count or self.population_size)
        if count < 1:
            return []
        if count < 3:
            return [self._tail_sample().astype(float).tolist() for _ in range(count)]
        return [_unit_vector(candidate, self.dim).astype(float).tolist() for candidate in self._es.ask(number=count)]

    def tell(self, candidates: list[list[float]], scores: list[float], *, maximize: bool) -> None:
        if not candidates:
            return
        if len(candidates) != len(scores):
            raise ValueError("candidates and scores must have equal length")
        x = np.asarray(candidates, dtype=np.float64).reshape(len(candidates), self.dim)
        s = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(s)):
            raise ValueError("scores must be finite")

        order = np.argsort(s)
        if maximize:
            order = order[::-1]
        best_index = int(order[0])
        best_score = float(s[best_index])
        if self.best_score is None or (best_score > self.best_score if maximize else best_score < self.best_score):
            self.best_score = best_score
            self.best_vector = x[best_index].copy()

        # Pycma rejects tiny population updates. This only happens at the tail of a rollout budget;
        # the candidate still contributes to best-trial tracking above, but not to distribution adaptation.
        if len(candidates) < 3:
            return
        objective_values = (-s if maximize else s).astype(float).tolist()
        self._es.tell(_unit_matrix(x, self.dim).astype(float).tolist(), objective_values)

    def state(self) -> CMAESState:
        return CMAESState(
            iteration=int(getattr(self._es, "countiter", 0)),
            mean=_unit_vector(getattr(self._es, "mean", np.zeros(self.dim)), self.dim).astype(float).tolist(),
            sigma=float(getattr(self._es, "sigma", 0.0)),
            axis_sigmas=self._axis_sigmas().astype(float).tolist(),
            best_score=self.best_score,
            best_vector=None if self.best_vector is None else self.best_vector.astype(float).tolist(),
        )

    def _axis_sigmas(self) -> np.ndarray:
        stds = getattr(self._es, "stds", None)
        if stds is not None:
            values = np.asarray(stds, dtype=np.float64).reshape(-1)
            if values.size == self.dim and np.all(np.isfinite(values)):
                return np.clip(values, 1e-4, 1.0)
        sigma = float(getattr(self._es, "sigma", 0.0))
        variances = np.asarray(getattr(self._es.sm, "variances", np.ones(self.dim)), dtype=np.float64).reshape(-1)
        if variances.size == self.dim and np.all(np.isfinite(variances)):
            return np.clip(sigma * np.sqrt(np.clip(variances, 1e-12, None)), 1e-4, 1.0)
        return self._initial_axis_sigmas.copy()

    def _tail_sample(self) -> np.ndarray:
        mean = _unit_vector(getattr(self._es, "mean", np.zeros(self.dim)), self.dim)
        return np.clip(mean + self._axis_sigmas() * self._rng.standard_normal(self.dim), 0.0, 1.0)


def _unit_vector(values: object, dim: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(dim)
    if not np.all(np.isfinite(vector)):
        raise ValueError("CMA-ES vectors must be finite")
    return np.clip(vector, 0.0, 1.0)


def _unit_matrix(values: object, dim: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64).reshape(-1, dim)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("CMA-ES candidates must be finite")
    return np.clip(matrix, 0.0, 1.0)


def _normalized_sigmas(sigmas: list[float] | tuple[float, ...], dim: int) -> np.ndarray:
    values = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    if values.size != dim:
        raise ValueError(f"initial_sigmas has {values.size} values for dim={dim}")
    if not np.all(np.isfinite(values)):
        raise ValueError("initial_sigmas must be finite")
    if np.any(values <= 0.0):
        raise ValueError("initial_sigmas must be positive")
    return np.clip(values, 1e-4, 1.0)


def _clamp_sigma(value: float) -> float:
    return float(np.clip(value, 1e-4, 1.0))
