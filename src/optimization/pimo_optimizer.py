"""
PIMO 优化器模块
支持并行计算和早停机制的 PIMO 算法实现
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
from joblib import Parallel, delayed


class PIMOOptimizer:
    """
    PIMO (Particle-Inspired Metaheuristic Optimizer) 优化器

    基于粒子群和梯度信息的混合优化算法
    支持并行计算和早停机制
    """

    def __init__(self,
                 n_agents: int,
                 max_iter: int,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 dim: int,
                 fobj: Callable,
                 n_jobs: int = -1,
                 epsilon: float = 1e-6,
                 patience: int = 50):
        """
        初始化 PIMO 优化器

        Args:
            n_agents: 种群大小
            max_iter: 最大迭代次数
            lb: 下界数组
            ub: 上界数组
            dim: 维度
            fobj: 目标函数
            n_jobs: 并行核心数，-1 表示使用所有核心，1 表示串行
            epsilon: Jacobian 计算的扰动量
            patience: 早停耐心值，连续多少次迭代无改善则停止
        """
        self.N = n_agents
        self.Max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.fobj = fobj
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.patience = patience

        # 优化历史
        self.convergence_curve = None
        self.best_score = None
        self.best_pos = None

    def initialization(self) -> np.ndarray:
        """
        初始化种群位置

        Returns:
            np.ndarray: 初始位置矩阵 (n_agents, dim)
        """
        return np.random.uniform(0, 1, (self.N, self.dim)) * (self.ub - self.lb) + self.lb

    def get_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        计算 Jacobian 向量（数值梯度）

        Args:
            x: 当前位置

        Returns:
            np.ndarray: Jacobian 向量
        """
        fx = self.fobj(x)
        j = np.zeros(self.dim)

        for i in range(self.dim):
            x_perturbed = x.copy()
            x_perturbed[i] += self.epsilon
            f_perturbed = self.fobj(x_perturbed)
            j[i] = (f_perturbed - fx) / self.epsilon

        return j

    def _evaluate_single_agent(self, pos: np.ndarray) -> float:
        """
        评估单个智能体的适应度

        Args:
            pos: 位置

        Returns:
            float: 适应度值
        """
        return self.fobj(pos)

    def _update_single_agent(self,
                            current_pos: np.ndarray,
                            current_fit: float,
                            best_pos: np.ndarray,
                            delta: float) -> Tuple[np.ndarray, float]:
        """
        更新单个智能体的位置

        Args:
            current_pos: 当前位置
            current_fit: 当前适应度
            best_pos: 全局最优位置
            delta: 当前迭代的衰减因子

        Returns:
            Tuple: (新位置, 新适应度)
        """
        rr1, rr2 = np.random.rand(), np.random.rand()

        if rr1 >= rr2:
            # 使用 Jacobian 信息进行搜索
            j_matrix = self.get_jacobian(current_pos)
            new_pos = current_pos - delta * j_matrix * np.random.rand()
        else:
            # 向全局最优学习
            r = np.random.rand()
            new_pos = best_pos + delta * (best_pos - current_pos) * r

        # 边界约束
        new_pos = np.clip(new_pos, self.lb, self.ub)

        # 计算新适应度
        new_fit = self.fobj(new_pos)

        # 如果新位置更好，返回新位置和新适应度，否则返回旧的
        if new_fit < current_fit:
            return new_pos, new_fit
        return current_pos, current_fit

    def optimize(self, verbose: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        执行优化

        Args:
            verbose: 是否打印进度信息

        Returns:
            Tuple: (best_score, best_pos, convergence_curve)
        """
        # 初始化种群
        position = self.initialization()

        if verbose:
            mode = "并行" if self.n_jobs != 1 else "串行"
            print(f"正在{mode}初始化 {self.N} 个个体的适应度 (n_jobs={self.n_jobs})...")

        # 并行计算初始种群的适应度
        if self.n_jobs == 1:
            # 串行模式
            fitness = np.array([self.fobj(position[i, :]) for i in range(self.N)])
        else:
            # 并行模式
            fitness = np.array(Parallel(n_jobs=self.n_jobs)(
                delayed(self.fobj)(position[i, :]) for i in range(self.N)
            ))

        # 记录全局最优
        best_idx = np.argmin(fitness)
        best_score = fitness[best_idx]
        best_pos = position[best_idx, :].copy()
        convergence_curve = []

        # 早停计数器
        no_improve_count = 0
        prev_best_score = best_score

        # 迭代优化
        for it in range(self.Max_iter):
            # 计算当前迭代的衰减因子
            delta = np.sin((np.pi / 2) * (1 - (2 * (it + 1) / self.Max_iter)) ** 5)

            # 并行更新所有智能体
            if self.n_jobs == 1:
                # 串行模式
                for i in range(self.N):
                    new_pos, new_fit = self._update_single_agent(
                        position[i, :], fitness[i], best_pos, delta
                    )
                    position[i, :] = new_pos
                    fitness[i] = new_fit
            else:
                # 并行模式
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._update_single_agent)(
                        position[i, :], fitness[i], best_pos, delta
                    )
                    for i in range(self.N)
                )

                # 写回结果
                for i, (new_pos, new_fit) in enumerate(results):
                    position[i, :] = new_pos
                    fitness[i] = new_fit

            # 更新全局最优
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_score:
                best_score = fitness[current_best_idx]
                best_pos = position[current_best_idx, :].copy()

            convergence_curve.append(best_score)

            # 早停检查
            if best_score < prev_best_score:
                no_improve_count = 0
                prev_best_score = best_score
            else:
                no_improve_count += 1

            if verbose:
                print(f"迭代 [{it + 1}/{self.Max_iter}], 最佳误差: {best_score:.6f}, 早停计数: {no_improve_count}/{self.patience}")

            # 早停判断
            if no_improve_count >= self.patience:
                if verbose:
                    print(f"早停触发: 连续 {self.patience} 次迭代无改善，在第 {it + 1} 代停止")
                break

        # 保存结果
        self.best_score = best_score
        self.best_pos = best_pos
        self.convergence_curve = np.array(convergence_curve)

        return best_score, best_pos, self.convergence_curve

    def get_results(self) -> dict:
        """
        获取优化结果

        Returns:
            dict: 包含 best_score, best_pos, convergence_curve 的字典
        """
        return {
            'best_score': self.best_score,
            'best_pos': self.best_pos,
            'convergence_curve': self.convergence_curve,
        }


def create_pimo_optimizer(config: dict,
                         lb: List[float],
                         ub: List[float],
                         fobj: Callable) -> PIMOOptimizer:
    """
    创建 PIMO 优化器实例

    Args:
        config: 配置字典
        lb: 下界列表
        ub: 上界列表
        fobj: 目标函数

    Returns:
        PIMOOptimizer: PIMO 优化器实例
    """
    optimizer_config = config.get('optimizer', {})
    pimo_config = config.get('pimo', {})

    n_agents = optimizer_config.get('n_agents', 8)
    total_evaluations = optimizer_config.get('total_evaluations', 800)
    max_iter = total_evaluations // n_agents

    return PIMOOptimizer(
        n_agents=n_agents,
        max_iter=max_iter,
        lb=np.array(lb),
        ub=np.array(ub),
        dim=len(lb),
        fobj=fobj,
        n_jobs=optimizer_config.get('n_jobs', -1),
        epsilon=pimo_config.get('epsilon', 1e-6),
        patience=optimizer_config.get('patience', 50),
    )
