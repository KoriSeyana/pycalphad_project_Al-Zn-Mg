from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed


@dataclass
class ACOResult:
    """ACO 优化结果。"""

    best_score: float
    best_position: np.ndarray
    convergence_curve: np.ndarray


class ACOMatlabOptimizer:
    """按 MATLAB Newyiqun 核心逻辑移植的 ACO 优化器。"""

    def __init__(
        self,
        ant_count: int,
        max_iter: int,
        lb: np.ndarray,
        ub: np.ndarray,
        fobj: Callable[[np.ndarray], float],
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q: float = 1.0,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.ant_count = int(ant_count)
        self.max_iter = int(max_iter)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = len(self.lb)
        self.fobj = fobj
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.q = float(q)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.rng = np.random.default_rng(self.random_state)

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        if self.n_jobs == 1:
            return np.array([float(self.fobj(p)) for p in population], dtype=float)
        values = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self.fobj)(p.copy()) for p in population
        )
        return np.array(values, dtype=float)

    def optimize(self) -> ACOResult:
        tau = np.ones(self.dim, dtype=float)
        best_position: np.ndarray | None = None
        best_score = float("inf")
        convergence = np.zeros(self.max_iter, dtype=float)

        for it in range(self.max_iter):
            prob = (tau**self.alpha) * (1.0**self.beta)
            prob = prob / prob.sum()
            max_prob = float(np.max(prob))

            rand_weights = self.lb + (self.ub - self.lb) * self.rng.random((self.ant_count, self.dim))
            ants = rand_weights * (prob / max_prob)
            ants = np.clip(ants, self.lb, self.ub)
            errors = self._evaluate_population(ants)

            idx = int(np.argmin(errors))
            if errors[idx] < best_score:
                best_score = float(errors[idx])
                best_position = ants[idx].copy()

            delta_tau = np.zeros(self.dim, dtype=float)
            for ant_err in errors:
                if ant_err > 0:
                    delta_tau += (self.q / ant_err) * np.ones(self.dim, dtype=float)
            tau = (1 - self.rho) * tau + delta_tau
            convergence[it] = best_score

        if best_position is None:
            best_position = self.lb.copy()

        return ACOResult(best_score=best_score, best_position=best_position, convergence_curve=convergence)
