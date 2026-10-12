from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed


@dataclass
class XWOAResult:
    """改进 WOA 优化结果。"""

    best_score: float
    best_position: np.ndarray
    convergence_curve: np.ndarray


class XWOAMatlabOptimizer:
    """按 MATLAB Newxinjingyu 核心逻辑移植的改进 WOA。"""

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

    def _evaluate_population(self, positions: np.ndarray) -> np.ndarray:
        if self.n_jobs == 1:
            return np.array([float(self.fobj(p)) for p in positions], dtype=float)
        values = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self.fobj)(p.copy()) for p in positions
        )
        return np.array(values, dtype=float)

    def optimize(self) -> XWOAResult:
        positions = self.lb + (self.ub - self.lb) * self.rng.random((self.n_agents, self.dim))
        fitness = self._evaluate_population(positions)
        best_idx = int(np.argmin(fitness))
        leader_pos = positions[best_idx].copy()
        leader_score = float(fitness[best_idx])
        curve = np.zeros(self.max_iter, dtype=float)

        for t in range(1, self.max_iter + 1):
            a = 2 - t * (2 / self.max_iter)
            a2 = -1 + t * (-1 / self.max_iter)

            next_positions = positions.copy()
            for i in range(self.n_agents):
                r1 = self.rng.random()
                r2 = self.rng.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1.0
                l = (a2 - 1) * self.rng.random() + 1
                p = self.rng.random()

                current = positions[i]
                if p < 0.5:
                    if abs(A) >= 1:
                        x_rand = self.lb + (self.ub - self.lb) * self.rng.random(self.dim)
                        d_x_rand = np.abs(C * x_rand - current)
                        new_pos = x_rand - A * d_x_rand
                    else:
                        d_leader = np.abs(C * leader_pos - current)
                        new_pos = leader_pos - A * d_leader
                else:
                    distance = np.abs(leader_pos - current)
                    spiral_type = int(self.rng.integers(1, 4))
                    if spiral_type == 1:
                        new_pos = distance * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader_pos
                    elif spiral_type == 2:
                        theta = 2 * np.pi * l
                        r_spiral = b * theta
                        indices = np.arange(self.dim)
                        angle_d = theta + indices * np.pi / self.dim
                        new_pos = distance * (r_spiral / (2 * np.pi * b)) * np.cos(angle_d) + leader_pos
                    else:
                        theta = 2 * np.pi * l
                        r_hyper = b / (1 + abs(theta))
                        indices = np.arange(self.dim)
                        angle_d = theta + indices * np.pi / self.dim
                        new_pos = distance * r_hyper * np.cos(angle_d) + leader_pos

                next_positions[i] = np.clip(new_pos, self.lb, self.ub)

            positions = next_positions
            fitness = self._evaluate_population(positions)
            idx = int(np.argmin(fitness))
            if fitness[idx] < leader_score:
                leader_score = float(fitness[idx])
                leader_pos = positions[idx].copy()
            curve[t - 1] = leader_score

        return XWOAResult(best_score=leader_score, best_position=leader_pos, convergence_curve=curve)
