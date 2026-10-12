from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed


@dataclass
class PSOResult:
    """PSO 优化结果。"""

    best_score: float
    best_position: np.ndarray
    convergence_curve: np.ndarray


class PSOMatlabOptimizer:
    """按 MATLAB NewPSO 核心并行段移植的 PSO 优化器。"""

    def __init__(
        self,
        n_particles: int,
        max_iter: int,
        lb: np.ndarray,
        ub: np.ndarray,
        fobj: Callable[[np.ndarray], float],
        w: float = 0.9,
        wdamp: float = 0.99,
        c1: float = 2.0,
        c2: float = 2.0,
        vel_min: float = -0.2,
        vel_max: float = 0.2,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.n_particles = int(n_particles)
        self.max_iter = int(max_iter)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = len(self.lb)
        self.fobj = fobj
        self.w = float(w)
        self.wdamp = float(wdamp)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.vel_min = float(vel_min)
        self.vel_max = float(vel_max)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.rng = np.random.default_rng(self.random_state)

    def _evaluate_population(self, positions: np.ndarray) -> np.ndarray:
        if self.n_jobs == 1:
            return np.array([float(self.fobj(pos)) for pos in positions], dtype=float)

        values = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self.fobj)(pos.copy()) for pos in positions
        )
        return np.array(values, dtype=float)

    def optimize(self) -> PSOResult:
        positions = self.rng.random((self.n_particles, self.dim)) * (self.ub - self.lb) + self.lb
        velocities = np.zeros((self.n_particles, self.dim), dtype=float)

        costs = self._evaluate_population(positions)
        pbest_positions = positions.copy()
        pbest_costs = costs.copy()

        gbest_idx = int(np.argmin(pbest_costs))
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_cost = float(pbest_costs[gbest_idx])
        convergence_curve = np.zeros(self.max_iter, dtype=float)

        inertia = self.w
        for it in range(self.max_iter):
            next_positions = positions.copy()
            next_velocities = velocities.copy()

            for i in range(self.n_particles):
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)

                new_vel = (
                    inertia * velocities[i]
                    + self.c1 * r1 * (pbest_positions[i] - positions[i])
                    + self.c2 * r2 * (gbest_position - positions[i])
                )
                new_vel = np.clip(new_vel, self.vel_min, self.vel_max)
                new_pos = np.clip(positions[i] + new_vel, self.lb, self.ub)

                next_velocities[i] = new_vel
                next_positions[i] = new_pos

            positions = next_positions
            velocities = next_velocities
            costs = self._evaluate_population(positions)

            improved = costs < pbest_costs
            pbest_costs[improved] = costs[improved]
            pbest_positions[improved] = positions[improved]

            gbest_idx = int(np.argmin(pbest_costs))
            if pbest_costs[gbest_idx] < gbest_cost:
                gbest_cost = float(pbest_costs[gbest_idx])
                gbest_position = pbest_positions[gbest_idx].copy()

            convergence_curve[it] = gbest_cost
            inertia *= self.wdamp

        return PSOResult(
            best_score=gbest_cost,
            best_position=gbest_position,
            convergence_curve=convergence_curve,
        )
