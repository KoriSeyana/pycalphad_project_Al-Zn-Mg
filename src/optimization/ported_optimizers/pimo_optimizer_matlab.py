from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed


@dataclass
class PIMOResult:
    """PIMO 优化结果。"""

    best_score: float
    best_position: np.ndarray
    convergence_curve: np.ndarray


class PIMOMatlabOptimizer:
    """按 MATLAB 原始公式移植的 PIMO 优化器。

    说明：
    1. 保留原始 PIMO.m 的核心搜索逻辑、Levy 飞行和雅可比近似。
    2. 仅做 Python 化改写和并行适配，不改变算法根本更新公式。
    3. 目标函数必须返回标量，适用于超参数优化等最小化问题。
    """

    def __init__(
        self,
        n_agents: int,
        max_iter: int,
        lb: np.ndarray,
        ub: np.ndarray,
        fobj: Callable[[np.ndarray], float],
        epsilon: float = 1e-6,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.n_agents = int(n_agents)
        self.max_iter = int(max_iter)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = len(self.lb)
        self.fobj = fobj
        self.epsilon = float(epsilon)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.rng = np.random.default_rng(self.random_state)

    def _initialization(self) -> np.ndarray:
        """对应 MATLAB 的 initialization.m。"""
        return self.rng.random((self.n_agents, self.dim)) * (self.ub - self.lb) + self.lb

    def _levy(self, n: int, m: int, beta: float) -> np.ndarray:
        """对应 MATLAB 中的 levy 函数。"""
        num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma_u = (num / den) ** (1 / beta)
        u = self.rng.normal(0, sigma_u, size=(n, m))
        v = self.rng.normal(0, 1, size=(n, m))
        return u / (np.abs(v) ** (1 / beta))

    def _get_jacobian(self, x: np.ndarray) -> np.ndarray:
        """按有限差分近似雅可比矩阵。

        MATLAB 原版支持向量函数，这里项目目标函数为标量，
        因此返回形状为 (1, dim) 的雅可比矩阵。
        """
        fx = float(self.fobj(x))
        jacobian = np.zeros((1, self.dim), dtype=float)

        perturbed_points = []
        for i in range(self.dim):
            x_perturbed = x.copy()
            x_perturbed[i] += self.epsilon
            perturbed_points.append(x_perturbed)

        if self.n_jobs == 1:
            perturbed_values = [float(self.fobj(point)) for point in perturbed_points]
        else:
            perturbed_values = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(self.fobj)(point.copy()) for point in perturbed_points
            )

        for i, f_perturbed in enumerate(perturbed_values):
            jacobian[0, i] = (float(f_perturbed) - fx) / self.epsilon
        return jacobian

    def _evaluate_population(self, positions: np.ndarray) -> np.ndarray:
        """并行计算种群适应度。"""
        if self.n_jobs == 1:
            return np.array([float(self.fobj(pos)) for pos in positions], dtype=float)
        values = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self.fobj)(pos.copy()) for pos in positions
        )
        return np.array(values, dtype=float)

    def optimize(self) -> PIMOResult:
        """执行优化。"""
        position = self._initialization()
        fitness = self._evaluate_population(position)
        ind = np.argsort(fitness)
        best_score = float(fitness[ind[0]])
        best_pos = position[ind[0]].copy()
        cg_curve = np.zeros(self.max_iter, dtype=float)
        alphabet = np.arange(self.n_agents)

        for it in range(1, self.max_iter + 1):
            delta = np.sin(np.pi / 2 * (1 - (2 * it / self.max_iter)) ** 5)
            a = (1 - it / self.max_iter) * self.rng.random(self.dim)

            for i in range(self.n_agents):
                rr1 = self.rng.random()
                rr2 = self.rng.random()

                if rr1 >= rr2:
                    probabilities = fitness ** 2
                    probabilities[~np.isfinite(probabilities)] = 0
                    total_prob_sum = probabilities.sum()
                    if total_prob_sum == 0:
                        probabilities = np.ones(self.n_agents, dtype=float) / self.n_agents
                    else:
                        probabilities = probabilities / total_prob_sum
                    pair = self.rng.choice(alphabet, size=2, replace=True, p=probabilities)
                    p1, p2 = int(pair[0]), int(pair[1])
                else:
                    pair = ind[:2]
                    p1, p2 = int(pair[0]), int(pair[1])

                r = self.rng.random()
                jacobian = self._get_jacobian(position[p1])

                if 3 * self.rng.random() >= 2 * self.rng.random():
                    grad = (r * (position[p1] - best_pos) + (1 - r) * (position[p2] - best_pos)) / 2
                    pos_n1 = position[i] - delta * grad
                else:
                    grad = (r * (position[p2] - best_pos) + (1 - r) * (position[p1] - best_pos)) / 2
                    pos_n1 = position[i] + delta * (jacobian @ grad.reshape(-1, 1)).ravel()
                newpos1 = np.clip(pos_n1, self.lb, self.ub)
                fitt = float(self.fobj(newpos1))
                if fitt < fitness[i]:
                    fitness[i] = fitt
                    position[i] = newpos1

                if self.rng.random() > self.rng.random():
                    pair = self.rng.permutation(self.n_agents)[:2]
                    p1, p2 = int(pair[0]), int(pair[1])
                    if 3 * self.rng.random() >= 2 * self.rng.random():
                        grad = (r * (position[p1] - best_pos) + (1 - r) * (position[p2] - best_pos)) / 2
                        pos_n2 = position[i] - delta * grad
                    else:
                        grad = (r * (position[p2] - best_pos) + (1 - r) * (position[p1] - best_pos)) / 2
                        pos_n2 = position[i] + delta * (jacobian @ grad.reshape(-1, 1)).ravel()
                    newpos2 = np.clip(pos_n2, self.lb, self.ub)
                    fitt = float(self.fobj(newpos2))
                    if fitt < fitness[i]:
                        fitness[i] = fitt
                        position[i] = newpos2

                for j in range(self.dim):
                    r1 = 1 + self.rng.random()
                    r2 = 1 + self.rng.random()
                    pho_1 = r1 * position[i] + (1 - r1) * best_pos + r2 * (position[i] - best_pos)
                    pho_2 = position[i] + a * (newpos1 - best_pos)
                    pos_n3 = position[i].copy()
                    if self.rng.random() / (j + 1) > self.rng.random():
                        pos_n3[j] = pho_1[j]
                    else:
                        pos_n3[j] = pho_2[j]
                    newpos3 = np.clip(pos_n3, self.lb, self.ub)
                    fitt = float(self.fobj(newpos3))
                    if fitt < fitness[i]:
                        fitness[i] = fitt
                        position[i] = newpos3

            threshold = 0.5 * (np.tanh(9 * it / self.max_iter - 5) + 1)
            if self.rng.random() <= threshold:
                d = self.rng.random() * (1 - it / self.max_iter) ** 2
                step_length = self._levy(self.n_agents, self.dim, 1.5)
                elite = np.repeat(best_pos.reshape(1, -1), self.n_agents, axis=0)
                r = self.rng.random()
                for i in range(self.n_agents):
                    pos_n4 = np.zeros(self.dim, dtype=float)
                    for j in range(self.dim):
                        pos_n4[j] = (
                            r * elite[i, j]
                            + (1 - r)
                            * step_length[i, j]
                            * d
                            * (elite[i, j] - position[i, j] * (2 * it / self.max_iter))
                        )
                    newpos4 = np.clip(pos_n4, self.lb, self.ub)
                    fitt = float(self.fobj(newpos4))
                    if fitt < fitness[i]:
                        fitness[i] = fitt
                        position[i] = newpos4

            ind = np.argsort(fitness)
            if fitness[ind[0]] < best_score:
                best_score = float(fitness[ind[0]])
                best_pos = position[ind[0]].copy()
            cg_curve[it - 1] = best_score

        return PIMOResult(
            best_score=best_score,
            best_position=best_pos,
            convergence_curve=cg_curve,
        )
