from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed


@dataclass
class WOAResult:
    """WOA 优化结果。"""

    best_score: float
    best_position: np.ndarray
    convergence_curve: np.ndarray


class WOAMatlabOptimizer:
    """按 MATLAB Newyuanjingyu 核心逻辑移植的 WOA 优化器。"""

    def __init__(
        self,
        n_agents: int,
        max_iter: int,
        lb: np.ndarray,
        ub: np.ndarray,
        fobj: Callable[[np.ndarray], float],
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.n_agents = int(n_agents)
        self.max_iter = int(max_iter)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = len(self.lb)
        self.fobj = fobj
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.rng = np.random.default_rng(self.random_state)

    def _initialization(self) -> np.ndarray:
        return self.rng.random((self.n_agents, self.dim)) * (self.ub - self.lb) + self.lb

    def _evaluate_population(self, positions: np.ndarray) -> np.ndarray:
        if self.n_jobs == 1:
            return np.array([float(self.fobj(p)) for p in positions], dtype=float)
        values = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self.fobj)(p.copy()) for p in positions
        )
        return np.array(values, dtype=float)

    def optimize(self) -> WOAResult:
        leader_pos = np.zeros(self.dim, dtype=float)
        leader_score = float("inf")
        positions = self._initialization()
        convergence_curve = np.zeros(self.max_iter, dtype=float)

        for t in range(self.max_iter):
            positions = np.clip(positions, self.lb, self.ub)
            fitness = self._evaluate_population(positions)
            idx = int(np.argmin(fitness))
            if fitness[idx] < leader_score:
                leader_score = float(fitness[idx])
                leader_pos = positions[idx].copy()

            a = 2 - t * (2 / self.max_iter)
            a2 = -1 + t * (-1 / self.max_iter)

            for i in range(self.n_agents):
                r1 = self.rng.random()
                r2 = self.rng.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = (a2 - 1) * self.rng.random() + 1
                p = self.rng.random()

                if p < 0.5:
                    if abs(A) >= 1:
                        rand_idx = int(self.rng.integers(0, self.n_agents))
                        x_rand = positions[rand_idx]
                        d_x_rand = np.abs(C * x_rand - positions[i])
                        positions[i] = x_rand - A * d_x_rand
                    else:
                        d_leader = np.abs(C * leader_pos - positions[i])
                        positions[i] = leader_pos - A * d_leader
                else:
                    distance = np.abs(leader_pos - positions[i])
                    positions[i] = distance * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader_pos

            convergence_curve[t] = leader_score

        return WOAResult(
            best_score=leader_score,
            best_position=leader_pos,
            convergence_curve=convergence_curve,
        )
