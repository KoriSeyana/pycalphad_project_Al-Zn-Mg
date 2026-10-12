from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed


@dataclass
class ABCResult:
    """ABC 优化结果。"""

    best_score: float
    best_position: np.ndarray
    convergence_curve: np.ndarray


class ABCMatlabOptimizer:
    """按 MATLAB Newfengqun 核心逻辑移植的 ABC 优化器。"""

    def __init__(
        self,
        colony_size: int,
        max_iter: int,
        lb: np.ndarray,
        ub: np.ndarray,
        fobj: Callable[[np.ndarray], float],
        limit: int = 20,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.colony_size = int(colony_size)
        self.max_iter = int(max_iter)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = len(self.lb)
        self.fobj = fobj
        self.limit = int(limit)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.rng = np.random.default_rng(self.random_state)

    def _evaluate_population(self, points: np.ndarray) -> np.ndarray:
        if self.n_jobs == 1:
            return np.array([float(self.fobj(p)) for p in points], dtype=float)
        values = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self.fobj)(p.copy()) for p in points
        )
        return np.array(values, dtype=float)

    def _roulette(self, prob: np.ndarray) -> int:
        r = float(self.rng.random())
        csum = np.cumsum(prob)
        idx = int(np.searchsorted(csum, r, side="left"))
        return min(idx, len(prob) - 1)

    def optimize(self) -> ABCResult:
        employed_count = int(round(self.colony_size / 2))
        onlooker_count = self.colony_size - employed_count

        foods = self.lb + (self.ub - self.lb) * self.rng.random((self.colony_size, self.dim))
        fitness = self._evaluate_population(foods)
        trial = np.zeros(self.colony_size, dtype=int)

        best_idx = int(np.argmin(fitness))
        best_score = float(fitness[best_idx])
        best_position = foods[best_idx].copy()
        convergence_curve = np.zeros(self.max_iter, dtype=float)

        for it in range(self.max_iter):
            employed_candidates = foods[:employed_count].copy()
            for i in range(employed_count):
                k = i
                while k == i:
                    k = int(self.rng.integers(0, self.colony_size))
                j = int(self.rng.integers(0, self.dim))
                phi = -1.0 + 2.0 * self.rng.random()
                employed_candidates[i, j] = foods[i, j] + phi * (foods[i, j] - foods[k, j])
            employed_candidates = np.clip(employed_candidates, self.lb, self.ub)
            employed_fit = self._evaluate_population(employed_candidates)

            improved_emp = employed_fit < fitness[:employed_count]
            foods[:employed_count][improved_emp] = employed_candidates[improved_emp]
            fitness[:employed_count][improved_emp] = employed_fit[improved_emp]
            trial[:employed_count][improved_emp] = 0
            trial[:employed_count][~improved_emp] += 1

            scaled = fitness - np.min(fitness) + np.finfo(float).eps
            inv = 1.0 / scaled
            prob = inv / inv.sum()

            selected_indices = np.array([self._roulette(prob) for _ in range(onlooker_count)], dtype=int)
            onlooker_candidates = np.zeros((onlooker_count, self.dim), dtype=float)
            for i in range(onlooker_count):
                selected = int(selected_indices[i])
                k = selected
                while k == selected:
                    k = int(self.rng.integers(0, self.colony_size))
                j = int(self.rng.integers(0, self.dim))
                phi = -1.0 + 2.0 * self.rng.random()
                candidate = foods[selected].copy()
                candidate[j] = foods[selected, j] + phi * (foods[selected, j] - foods[k, j])
                onlooker_candidates[i] = candidate
            onlooker_candidates = np.clip(onlooker_candidates, self.lb, self.ub)
            onlooker_fit = self._evaluate_population(onlooker_candidates)

            for i in range(onlooker_count):
                selected = int(selected_indices[i])
                if onlooker_fit[i] < fitness[selected]:
                    foods[selected] = onlooker_candidates[i]
                    fitness[selected] = onlooker_fit[i]
                    trial[selected] = 0
                else:
                    trial[selected] += 1

            scout_idx = np.where(trial > self.limit)[0]
            if len(scout_idx) > 0:
                scout_points = self.lb + (self.ub - self.lb) * self.rng.random((len(scout_idx), self.dim))
                scout_fit = self._evaluate_population(scout_points)
                foods[scout_idx] = scout_points
                fitness[scout_idx] = scout_fit
                trial[scout_idx] = 0

            idx = int(np.argmin(fitness))
            if fitness[idx] < best_score:
                best_score = float(fitness[idx])
                best_position = foods[idx].copy()
            convergence_curve[it] = best_score

        return ABCResult(
            best_score=best_score,
            best_position=best_position,
            convergence_curve=convergence_curve,
        )
