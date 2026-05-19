from __future__ import annotations

import math
from dataclasses import dataclass

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
    """Small bounded CMA-ES implementation over normalized vectors in [0, 1]."""

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
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        self.mu = max(1, self.population_size // 2)
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1, dtype=np.float64))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = float(1.0 / np.sum(self.weights**2))

        self.mean = np.clip(np.asarray(mean, dtype=np.float64), 0.0, 1.0).reshape(self.dim)
        if initial_sigmas is None:
            base_sigma = CONFIGS.opt.runner_default_initial_sigma if sigma is None else sigma
            self.sigma = float(np.clip(base_sigma, 1e-4, 1.0))
            self.cov = np.eye(self.dim, dtype=np.float64)
        else:
            axis_sigmas = _normalized_sigmas(initial_sigmas, self.dim)
            self.sigma = float(np.clip(np.median(axis_sigmas), 1e-4, 1.0))
            self.cov = np.diag((axis_sigmas / self.sigma) ** 2)
        self.pc = np.zeros(self.dim, dtype=np.float64)
        self.ps = np.zeros(self.dim, dtype=np.float64)
        self.basis = np.eye(self.dim, dtype=np.float64)
        self.diag = np.ones(self.dim, dtype=np.float64)
        self.invsqrt_cov = np.eye(self.dim, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

        self.cc = (4.0 + self.mu_eff / self.dim) / (self.dim + 4.0 + 2.0 * self.mu_eff / self.dim)
        self.cs = (self.mu_eff + 2.0) / (self.dim + self.mu_eff + 5.0)
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((self.dim + 2.0) ** 2 + self.mu_eff),
        )
        self.damps = 1.0 + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (self.dim + 1.0)) - 1.0) + self.cs
        self.chi_n = math.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim**2))
        self.iteration = 0
        self.best_score: float | None = None
        self.best_vector: np.ndarray | None = None
        self._update_eigensystem()

    def ask(self, count: int | None = None) -> list[list[float]]:
        count = int(count or self.population_size)
        if count < 1:
            return []
        samples = []
        transform = self.basis @ np.diag(self.diag)
        for _ in range(count):
            z = self.rng.standard_normal(self.dim)
            y = transform @ z
            candidate = np.clip(self.mean + self.sigma * y, 0.0, 1.0)
            samples.append(candidate.astype(float).tolist())
        return samples

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
        selected = x[order[: self.mu]]
        if len(selected) < self.mu:
            weights = self.weights[: len(selected)]
            weights = weights / np.sum(weights)
        else:
            weights = self.weights

        best_index = order[0]
        best_score = float(s[best_index])
        if self.best_score is None or (best_score > self.best_score if maximize else best_score < self.best_score):
            self.best_score = best_score
            self.best_vector = x[best_index].copy()

        old_mean = self.mean.copy()
        y_k = (selected - old_mean) / self.sigma
        y_w = np.sum(y_k * weights[:, None], axis=0)
        self.mean = np.clip(old_mean + self.sigma * y_w, 0.0, 1.0)

        self.ps = (1.0 - self.cs) * self.ps + math.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff) * (
            self.invsqrt_cov @ y_w
        )
        norm_ps = float(np.linalg.norm(self.ps))
        correction = math.sqrt(max(1e-12, 1.0 - (1.0 - self.cs) ** (2.0 * (self.iteration + 1))))
        hsig = norm_ps / correction / self.chi_n < (1.4 + 2.0 / (self.dim + 1.0))
        hsig_float = 1.0 if hsig else 0.0
        self.pc = (1.0 - self.cc) * self.pc + hsig_float * math.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * y_w

        rank_mu = np.zeros_like(self.cov)
        for weight, y in zip(weights, y_k):
            rank_mu += float(weight) * np.outer(y, y)
        delta_hsig = (1.0 - hsig_float) * self.cc * (2.0 - self.cc)
        self.cov = (
            (1.0 - self.c1 - self.cmu + self.c1 * delta_hsig) * self.cov
            + self.c1 * np.outer(self.pc, self.pc)
            + self.cmu * rank_mu
        )
        self.cov = (self.cov + self.cov.T) * 0.5
        self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chi_n - 1.0))
        self.sigma = float(np.clip(self.sigma, 1e-4, 1.0))
        self.iteration += 1
        self._update_eigensystem()

    def state(self) -> CMAESState:
        return CMAESState(
            iteration=self.iteration,
            mean=self.mean.astype(float).tolist(),
            sigma=float(self.sigma),
            axis_sigmas=(self.sigma * np.sqrt(np.clip(np.diag(self.cov), 1e-12, None))).astype(float).tolist(),
            best_score=self.best_score,
            best_vector=None if self.best_vector is None else self.best_vector.astype(float).tolist(),
        )

    def _update_eigensystem(self) -> None:
        if not np.all(np.isfinite(self.cov)):
            self.cov = np.eye(self.dim, dtype=np.float64)
        self.cov += np.eye(self.dim, dtype=np.float64) * 1e-12
        diag_squared, basis = np.linalg.eigh(self.cov)
        diag_squared = np.clip(diag_squared, 1e-12, 1e12)
        self.diag = np.sqrt(diag_squared)
        self.basis = basis
        self.invsqrt_cov = basis @ np.diag(1.0 / self.diag) @ basis.T


def _normalized_sigmas(sigmas: list[float] | tuple[float, ...], dim: int) -> np.ndarray:
    values = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    if values.size != dim:
        raise ValueError(f"initial_sigmas has {values.size} values for dim={dim}")
    if not np.all(np.isfinite(values)):
        raise ValueError("initial_sigmas must be finite")
    if np.any(values <= 0.0):
        raise ValueError("initial_sigmas must be positive")
    return np.clip(values, 1e-4, 1.0)
