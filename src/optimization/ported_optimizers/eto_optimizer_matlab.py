from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed


@dataclass
class ETOResult:
    """ETO 优化结果。"""

    best_score: float
    best_position: np.ndarray
    convergence_curve: np.ndarray


class ETOMatlabOptimizer:
    """按 MATLAB 原始公式移植的 ETO 优化器。

    说明：
    1. 保留 ETO.m 的核心更新公式与两阶段搜索结构。
    2. 仅做 Python 化改写，并补充最小并行评估能力。
    3. 目标函数为最小化问题，返回标量值。
    """

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

    def _initialization(self, n_agents: int, dim: int, ub: np.ndarray, lb: np.ndarray) -> np.ndarray:
        """对应 MATLAB 的 initialization.m。"""
        if ub.size == 1:
            return self.rng.random((n_agents, dim)) * (ub.item() - lb.item()) + lb.item()

        x = np.zeros((n_agents, dim), dtype=float)
        for i in range(dim):
            x[:, i] = self.rng.random(n_agents) * (ub[i] - lb[i]) + lb[i]
        return x

    def _evaluate_population(self, positions: np.ndarray) -> np.ndarray:
        """并行计算一组候选解的适应度。"""
        if self.n_jobs == 1:
            return np.array([float(self.fobj(pos)) for pos in positions], dtype=float)

        values = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self.fobj)(pos.copy()) for pos in positions
        )
        return np.array(values, dtype=float)

    def optimize(self) -> ETOResult:
        """执行 ETO 优化。"""
        destination_position = np.zeros(self.dim, dtype=float)
        destination_fitness = float("inf")
        destination_position_second = np.zeros(self.dim, dtype=float)
        convergence_curve = np.zeros(self.max_iter, dtype=float)
        position_sort = np.zeros((self.n_agents, self.dim), dtype=float)

        b = 1.55
        ce = int(np.floor(1 + (self.max_iter / b)))
        t_switch = int(np.floor(1.2 + self.max_iter / 2.25))
        cei = 0
        cei_temp = 0
        ub_2 = self.ub.copy()
        lb_2 = self.lb.copy()

        x = self._initialization(self.n_agents, self.dim, self.ub, self.lb)
        objective_values = self._evaluate_population(x)

        best_idx = int(np.argmin(objective_values))
        destination_position = x[best_idx].copy()
        destination_fitness = float(objective_values[best_idx])
        if self.max_iter >= 1:
            convergence_curve[0] = destination_fitness

        t = 2
        while t <= self.max_iter:
            for i in range(self.n_agents):
                for j in range(self.dim):
                    d1 = 0.1 * np.exp(-0.01 * t) * np.cos(0.5 * self.max_iter * (1 - t / self.max_iter))
                    d2 = -0.1 * np.exp(-0.01 * t) * np.cos(0.5 * self.max_iter * (1 - t / self.max_iter))

                    cm = (np.sqrt(t / self.max_iter) ** np.tan(d1 / d2)) * self.rng.random() * 0.01

                    if t == cei:
                        ub_2[j] = (
                            destination_position[j]
                            + (1 - t / self.max_iter)
                            * abs(self.rng.random() * destination_position[j] - destination_position_second[j])
                            * self.rng.random()
                        )
                        lb_2[j] = (
                            destination_position[j]
                            - (1 - t / self.max_iter)
                            * abs(self.rng.random() * destination_position[j] - destination_position_second[j])
                            * self.rng.random()
                        )

                        if ub_2[j] > self.ub[j]:
                            ub_2[j] = self.ub[j]
                        if lb_2[j] < self.lb[j]:
                            lb_2[j] = self.lb[j]

                        x = self._initialization(self.n_agents, self.dim, ub_2, lb_2)
                        cei_temp = cei
                        cei = 0

                    if t <= t_switch:
                        q1 = self.rng.random()
                        q3 = self.rng.random()
                        q4 = self.rng.random()

                        if cm > 1:
                            d1 = 0.1 * np.exp(-0.01 * t) * np.cos(0.5 * self.max_iter * q1)
                            d2 = -0.1 * np.exp(-0.01 * t) * np.cos(0.5 * self.max_iter * q1)
                            alpha_1 = (
                                self.rng.random()
                                * 3
                                * (t / self.max_iter - 0.85)
                                * np.exp(abs(d1 / d2) - 1)
                            )
                            if q1 <= 0.5:
                                x[i, j] = (
                                    destination_position[j]
                                    + self.rng.random() * alpha_1 * abs(destination_position[j] - x[i, j])
                                )
                            else:
                                x[i, j] = (
                                    destination_position[j]
                                    - self.rng.random() * alpha_1 * abs(destination_position[j] - x[i, j])
                                )
                        else:
                            d1 = 0.1 * np.exp(-0.01 * t) * np.cos(0.5 * self.max_iter * q3)
                            d2 = -0.1 * np.exp(-0.01 * t) * np.cos(0.5 * self.max_iter * q3)
                            alpha_3 = (
                                self.rng.random()
                                * 3
                                * (t / self.max_iter - 0.85)
                                * np.exp(abs(d1 / d2) - 1.3)
                            )
                            if q3 <= 0.5:
                                x[i, j] = (
                                    destination_position[j]
                                    + q4
                                    * alpha_3
                                    * abs(self.rng.random() * destination_position[j] - x[i, j])
                                )
                            else:
                                x[i, j] = (
                                    destination_position[j]
                                    - q4
                                    * alpha_3
                                    * abs(self.rng.random() * destination_position[j] - x[i, j])
                                )
                    else:
                        q2 = self.rng.random()
                        alpha_2 = self.rng.random() * np.exp(
                            np.tanh(1.5 * (-t / self.max_iter - 0.75) - self.rng.random())
                        )
                        if cm < 1:
                            d1 = 0.1 * np.exp(-0.01 * t) * np.cos(0.5 * self.max_iter * q2)
                            d2 = -0.1 * np.exp(-0.01 * t) * np.cos(0.5 * self.max_iter * q2)
                            x[i, j] = x[i, j] + np.exp(
                                np.tan(abs(d1 / d2))
                                * abs(self.rng.random() * alpha_2 * destination_position[j] - x[i, j])
                            )
                        else:
                            if q2 <= 0.5:
                                x[i, j] = x[i, j] + 3 * abs(
                                    self.rng.random() * alpha_2 * destination_position[j] - x[i, j]
                                )
                            else:
                                x[i, j] = x[i, j] - 3 * abs(
                                    self.rng.random() * alpha_2 * destination_position[j] - x[i, j]
                                )

                cei = cei_temp

            flag_ub = x > ub_2
            flag_lb = x < lb_2
            x = x * (~(flag_ub | flag_lb)) + ((ub_2 + lb_2) / 2) * flag_ub + lb_2 * flag_lb
            objective_values = self._evaluate_population(x)

            best_idx = int(np.argmin(objective_values))
            if objective_values[best_idx] < destination_fitness:
                destination_position = x[best_idx].copy()
                destination_fitness = float(objective_values[best_idx])

            if t == ce:
                cei = ce + 1
                denominator = self.max_iter - ce * 4.6
                if denominator != 0:
                    ce = ce + int(np.floor(2 - t * 2 / denominator))

                temp = np.zeros(self.n_agents, dtype=float)
                temp2 = np.zeros((self.n_agents, self.dim), dtype=float)

                for i in range(self.n_agents - 1):
                    for j in range(self.n_agents - 1 - i):
                        if objective_values[j] > objective_values[j + 1]:
                            temp[j] = objective_values[j]
                            objective_values[j] = objective_values[j + 1]
                            objective_values[j + 1] = temp[j]

                            temp2[j, :] = position_sort[j, :]
                            position_sort[j, :] = position_sort[j + 1, :]
                            position_sort[j + 1, :] = temp2[j, :]

                destination_position_second = position_sort[1, :].copy()

            convergence_curve[t - 1] = destination_fitness
            t += 1

        return ETOResult(
            best_score=destination_fitness,
            best_position=destination_position,
            convergence_curve=convergence_curve,
        )
