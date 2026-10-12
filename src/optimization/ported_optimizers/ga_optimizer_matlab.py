from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed


@dataclass
class GAResult:
    """GA 优化结果。"""

    best_score: float
    best_position: np.ndarray
    convergence_curve: np.ndarray


class GAMatlabOptimizer:
    """按 MATLAB NewGA 核心遗传策略移植的优化器。"""

    def __init__(
        self,
        pop_size: int,
        max_gen: int,
        lb: np.ndarray,
        ub: np.ndarray,
        fobj: Callable[[np.ndarray], float],
        pc: float = 0.8,
        pm: float = 0.1,
        elite_ratio: float = 0.1,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.pop_size = int(pop_size)
        self.max_gen = int(max_gen)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = len(self.lb)
        self.fobj = fobj
        self.pc = float(pc)
        self.pm = float(pm)
        self.elite_ratio = float(elite_ratio)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.rng = np.random.default_rng(self.random_state)

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        if self.n_jobs == 1:
            return np.array([float(self.fobj(ind)) for ind in population], dtype=float)

        values = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self.fobj)(ind.copy()) for ind in population
        )
        return np.array(values, dtype=float)

    def _initialize_population(self) -> np.ndarray:
        return self.lb + (self.ub - self.lb) * self.rng.random((self.pop_size, self.dim))

    def _selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        fitness_scaled = 1.0 / (1.0 + np.maximum(fitness, 0))
        total = fitness_scaled.sum()
        if total <= 0 or not np.isfinite(total):
            prob = np.full(self.pop_size, 1.0 / self.pop_size, dtype=float)
        else:
            prob = fitness_scaled / total
        idx = self.rng.choice(self.pop_size, size=self.pop_size, replace=True, p=prob)
        return population[idx].copy()

    def _crossover(self, population: np.ndarray) -> np.ndarray:
        offspring = population.copy()
        for i in range(0, self.pop_size - 1, 2):
            if self.rng.random() < self.pc:
                alpha = float(self.rng.random())
                p1 = population[i]
                p2 = population[i + 1]
                offspring[i] = alpha * p1 + (1 - alpha) * p2
                offspring[i + 1] = (1 - alpha) * p1 + alpha * p2
                offspring[i] = np.clip(offspring[i], self.lb, self.ub)
                offspring[i + 1] = np.clip(offspring[i + 1], self.lb, self.ub)
        return offspring

    def _mutation(self, population: np.ndarray) -> np.ndarray:
        mutated = population.copy()
        sigma = 0.1 * (self.ub - self.lb)
        for i in range(self.pop_size):
            mask = self.rng.random(self.dim) < self.pm
            if np.any(mask):
                mutated[i, mask] = mutated[i, mask] + sigma[mask] * self.rng.normal(size=mask.sum())
                mutated[i] = np.clip(mutated[i], self.lb, self.ub)
        return mutated

    def _elitism(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        offspring: np.ndarray,
        offspring_fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        combined_pop = np.vstack([population, offspring])
        combined_fit = np.concatenate([fitness, offspring_fitness])
        order = np.argsort(combined_fit)
        new_pop = combined_pop[order[: self.pop_size]].copy()
        new_fit = combined_fit[order[: self.pop_size]].copy()
        return new_pop, new_fit

    def optimize(self) -> GAResult:
        population = self._initialize_population()
        fitness = self._evaluate_population(population)

        best_idx = int(np.argmin(fitness))
        best_fitness = float(fitness[best_idx])
        best_individual = population[best_idx].copy()
        convergence_curve = np.zeros(self.max_gen, dtype=float)

        for gen in range(self.max_gen):
            selected = self._selection(population, fitness)
            offspring = self._crossover(selected)
            offspring = self._mutation(offspring)
            offspring_fitness = self._evaluate_population(offspring)

            population, fitness = self._elitism(population, fitness, offspring, offspring_fitness)

            current_best_idx = int(np.argmin(fitness))
            current_best = float(fitness[current_best_idx])
            if current_best < best_fitness:
                best_fitness = current_best
                best_individual = population[current_best_idx].copy()

            convergence_curve[gen] = best_fitness

        return GAResult(
            best_score=best_fitness,
            best_position=best_individual,
            convergence_curve=convergence_curve,
        )
