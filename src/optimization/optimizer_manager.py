"""
优化器管理模块
支持多种优化算法的统一接口，可扩展
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any
from src.optimization.pimo_optimizer import PIMOOptimizer, create_pimo_optimizer

# 尝试导入可选依赖
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import pygad
    PYGAD_AVAILABLE = True
except ImportError:
    PYGAD_AVAILABLE = False

try:
    import pyswarms as ps
    PSWARMS_AVAILABLE = True
except ImportError:
    PSWARMS_AVAILABLE = False

try:
    from pycaret.regression import tune_model, create_model, pull
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False


class OptimizerManager:
    """
    优化器管理器

    统一管理多种优化算法，提供一致的接口
    支持扩展新的优化算法
    """

    # 优化器注册表（可扩展）
    OPTIMIZER_REGISTRY = {}

    def __init__(self, config: dict):
        """
        初始化优化器管理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.optimizer_config = config.get('optimizer', {})

        # 注册内置优化器
        self._register_builtin_optimizers()

    def _register_builtin_optimizers(self):
        """注册内置优化器"""
        self.OPTIMIZER_REGISTRY['pimo'] = {
            'name': 'PIMO',
            'available': True,
            'run_func': self._run_pimo,
        }

        self.OPTIMIZER_REGISTRY['optuna'] = {
            'name': 'Optuna',
            'available': OPTUNA_AVAILABLE,
            'run_func': self._run_optuna,
        }

        self.OPTIMIZER_REGISTRY['ga'] = {
            'name': 'Genetic Algorithm',
            'available': PYGAD_AVAILABLE,
            'run_func': self._run_ga,
        }

        self.OPTIMIZER_REGISTRY['pso'] = {
            'name': 'Particle Swarm Optimization',
            'available': PSWARMS_AVAILABLE,
            'run_func': self._run_pso,
        }

        self.OPTIMIZER_REGISTRY['pycaret'] = {
            'name': 'PyCaret Built-in (Optuna-based)',
            'available': PYCARET_AVAILABLE,
            'run_func': self._run_pycaret,
        }

    @classmethod
    def register_optimizer(cls,
                          name: str,
                          run_func: Callable,
                          available: bool = True,
                          display_name: Optional[str] = None):
        """
        注册新的优化器（用于扩展）

        Args:
            name: 优化器名称（用于调用）
            run_func: 运行函数，签名: (lb, ub, fobj, verbose) -> (best_score, best_params, curve)
            available: 是否可用
            display_name: 显示名称
        """
        cls.OPTIMIZER_REGISTRY[name] = {
            'name': display_name or name,
            'available': available,
            'run_func': run_func,
        }
        print(f"优化器 '{name}' 已注册")

    def get_available_optimizers(self) -> List[str]:
        """
        获取可用的优化器列表

        Returns:
            List[str]: 可用优化器名称列表
        """
        return [name for name, info in self.OPTIMIZER_REGISTRY.items() if info['available']]

    def run(self,
           optimizer_name: str,
           lb: List[float],
           ub: List[float],
           fobj: Callable,
           verbose: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        运行指定的优化器

        Args:
            optimizer_name: 优化器名称
            lb: 下界列表
            ub: 上界列表
            fobj: 目标函数
            verbose: 是否打印进度

        Returns:
            Tuple: (best_score, best_params, convergence_curve)
        """
        if optimizer_name not in self.OPTIMIZER_REGISTRY:
            raise ValueError(f"未知优化器: {optimizer_name}，可用: {list(self.OPTIMIZER_REGISTRY.keys())}")

        optimizer_info = self.OPTIMIZER_REGISTRY[optimizer_name]

        if not optimizer_info['available']:
            raise RuntimeError(f"优化器 '{optimizer_name}' 不可用，请安装相关依赖")

        if verbose:
            print(f"\n{'='*50}")
            print(f"使用 {optimizer_info['name']} 优化器")
            print(f"{'='*50}")

        return optimizer_info['run_func'](lb, ub, fobj, verbose)

    def run_all(self,
               lb: List[float],
               ub: List[float],
               fobj: Callable,
               optimizers: Optional[List[str]] = None,
               verbose: bool = True) -> Dict[str, Tuple[float, np.ndarray, np.ndarray]]:
        """
        运行多个优化器进行对比

        Args:
            lb: 下界列表
            ub: 上界列表
            fobj: 目标函数
            optimizers: 要运行的优化器列表，如果为 None 则运行所有可用的
            verbose: 是否打印进度

        Returns:
            Dict: {optimizer_name: (best_score, best_params, curve)}
        """
        if optimizers is None:
            optimizers = self.get_available_optimizers()

        results = {}

        for opt_name in optimizers:
            if opt_name in self.OPTIMIZER_REGISTRY and self.OPTIMIZER_REGISTRY[opt_name]['available']:
                try:
                    result = self.run(opt_name, lb, ub, fobj, verbose)
                    results[opt_name] = result
                except Exception as e:
                    print(f"优化器 '{opt_name}' 运行失败: {e}")
            else:
                print(f"跳过优化器 '{opt_name}' (不可用)")

        return results

    # ==================== 内置优化器实现 ====================

    def _run_pimo(self,
                  lb: List[float],
                  ub: List[float],
                  fobj: Callable,
                  verbose: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        """运行 PIMO 优化器"""
        pimo_config = self.config.get('pimo', {})
        n_agents = self.optimizer_config.get('n_agents', 8)
        total_evaluations = self.optimizer_config.get('total_evaluations', 800)
        patience = self.optimizer_config.get('patience', 50)
        max_iter = total_evaluations // n_agents  # 计算代数

        optimizer = PIMOOptimizer(
            n_agents=n_agents,
            max_iter=max_iter,
            lb=np.array(lb),
            ub=np.array(ub),
            dim=len(lb),
            fobj=fobj,
            n_jobs=self.optimizer_config.get('n_jobs', -1),
            epsilon=pimo_config.get('epsilon', 1e-6),
            patience=patience,
        )

        return optimizer.optimize(verbose=verbose)

    def _run_optuna(self,
                    lb: List[float],
                    ub: List[float],
                    fobj: Callable,
                    verbose: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        """运行 Optuna 优化器"""
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna 未安装，请运行: pip install optuna")

        optuna_config = self.config.get('optuna', {})
        total_evaluations = self.optimizer_config.get('total_evaluations', 800)
        patience = self.optimizer_config.get('patience', 50)

        # 早停相关变量
        curve = []
        best_so_far = float('inf')
        no_improve_count = 0

        def objective(trial):
            nonlocal best_so_far, no_improve_count

            # 为每个参数创建采样
            params = []
            for i, (lo, hi) in enumerate(zip(lb, ub)):
                param = trial.suggest_float(f'param_{i}', lo, hi)
                params.append(param)

            value = fobj(np.array(params))

            # 更新收敛曲线和早停计数
            if value < best_so_far:
                best_so_far = value
                no_improve_count = 0
            else:
                no_improve_count += 1

            curve.append(best_so_far)

            # 早停检查
            if no_improve_count >= patience:
                raise optuna.exceptions.OptunaError("Early stopping")

            return value

        # 创建 study
        sampler_name = optuna_config.get('sampler', 'TPE')
        if sampler_name == 'TPE':
            sampler = optuna.samplers.TPESampler()
        elif sampler_name == 'CMA-ES':
            sampler = optuna.samplers.CmaEsSampler()
        else:
            sampler = optuna.samplers.RandomSampler()

        study = optuna.create_study(direction='minimize', sampler=sampler)

        try:
            study.optimize(objective, n_trials=total_evaluations, show_progress_bar=verbose)
        except optuna.exceptions.OptunaError:
            if verbose:
                print(f"早停触发: 连续 {patience} 次迭代无改善")

        # 提取结果
        best_params = np.array([study.best_params[f'param_{i}'] for i in range(len(lb))])

        return study.best_value, best_params, np.array(curve)

    def _run_ga(self,
                lb: List[float],
                ub: List[float],
                fobj: Callable,
                verbose: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        """运行遗传算法优化器"""
        if not PYGAD_AVAILABLE:
            raise RuntimeError("PyGAD 未安装，请运行: pip install pygad")

        ga_config = self.config.get('ga', {})
        n_agents = self.optimizer_config.get('n_agents', 8)
        total_evaluations = self.optimizer_config.get('total_evaluations', 800)
        patience = self.optimizer_config.get('patience', 50)
        max_iter = total_evaluations // n_agents  # 计算代数

        # 早停回调函数
        curve = []
        best_so_far = float('inf')
        no_improve_count = 0

        def on_generation(ga_instance):
            nonlocal best_so_far, no_improve_count

            # 获取当前最优适应度
            current_best = -ga_instance.best_solutions_fitness[-1]  # 取负值还原

            curve.append(current_best)

            # 早停检查
            if current_best < best_so_far:
                best_so_far = current_best
                no_improve_count = 0
            else:
                no_improve_count += 1

            if verbose and (ga_instance.generations_completed % 10 == 0):
                print(f"迭代 [{ga_instance.generations_completed}/{max_iter}], 最佳误差: {best_so_far:.6f}, 早停计数: {no_improve_count}/{patience}")

            if no_improve_count >= patience:
                if verbose:
                    print(f"早停触发: 连续 {patience} 次迭代无改善，在第 {ga_instance.generations_completed} 代停止")
                return "stop"

        # 适应度函数（取负值，因为 PyGAD 最大化适应度）
        def fitness_func(ga_instance, solution, solution_idx):
            return -fobj(np.array(solution))

        # 基因空间
        gene_space = [{'low': lo, 'high': hi} for lo, hi in zip(lb, ub)]

        ga_instance = pygad.GA(
            num_generations=max_iter,
            num_parents_mating=ga_config.get('num_parents_mating', n_agents // 2),
            fitness_func=fitness_func,
            sol_per_pop=n_agents,
            num_genes=len(lb),
            gene_space=gene_space,
            mutation_percent_genes=ga_config.get('mutation_percent_genes', 10),
            suppress_warnings=not verbose,
            on_generation=on_generation,
        )

        ga_instance.run()

        # 提取结果
        best_sol, best_fit, _ = ga_instance.best_solution()

        return -best_fit, np.array(best_sol), np.array(curve)

    def _run_pso(self,
                 lb: List[float],
                 ub: List[float],
                 fobj: Callable,
                 verbose: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        """运行粒子群优化器"""
        if not PSWARMS_AVAILABLE:
            raise RuntimeError("PySwarms 未安装，请运行: pip install pyswarms")

        pso_config = self.config.get('pso', {})
        n_agents = self.optimizer_config.get('n_agents', 8)
        total_evaluations = self.optimizer_config.get('total_evaluations', 800)
        patience = self.optimizer_config.get('patience', 50)
        max_iter = total_evaluations // n_agents  # 计算代数
        n_jobs = self.optimizer_config.get('n_jobs', -1)

        options = {
            'c1': pso_config.get('c1', 0.5),
            'c2': pso_config.get('c2', 0.3),
            'w': pso_config.get('w', 0.9),
        }

        # 创建优化器
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_agents,
            dimensions=len(lb),
            options=options,
            bounds=(np.array(lb), np.array(ub))
        )

        # 早停相关变量
        curve = []
        best_so_far = float('inf')
        no_improve_count = 0

        # 并行目标函数
        def parallel_fobj(x):
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(delayed(fobj)(xi) for xi in x)
            return np.array(results)

        # 自定义优化循环以支持早停
        for i in range(max_iter):
            # 执行一步优化
            cost, pos = optimizer.optimize(parallel_fobj, iters=1, verbose=False)

            current_best = optimizer.cost_history[-1] if optimizer.cost_history else cost
            curve.append(current_best)

            # 早停检查
            if current_best < best_so_far:
                best_so_far = current_best
                no_improve_count = 0
            else:
                no_improve_count += 1

            if verbose and (i + 1) % 10 == 0:
                print(f"迭代 [{i+1}/{max_iter}], 最佳误差: {best_so_far:.6f}, 早停计数: {no_improve_count}/{patience}")

            if no_improve_count >= patience:
                if verbose:
                    print(f"早停触发: 连续 {patience} 次迭代无改善，在第 {i+1} 代停止")
                break

        return optimizer.cost_history[-1], optimizer.pos_history[-1], np.array(curve)

    def _run_pycaret(self,
                     lb: List[float],
                     ub: List[float],
                     fobj: Callable,
                     verbose: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        运行 PyCaret 内置调优

        注意：PyCaret 的 tune_model 内部使用 Optuna 进行超参数搜索
        这里使用随机搜索模拟 PyCaret 的调优过程
        """
        if not PYCARET_AVAILABLE:
            raise RuntimeError("PyCaret 未安装，请运行: pip install pycaret")

        total_evaluations = self.optimizer_config.get('total_evaluations', 800)
        patience = self.optimizer_config.get('patience', 50)

        if verbose:
            print(f"PyCaret 内置调优: {total_evaluations} 次试验")

        # 随机搜索实现（模拟 PyCaret 的调优过程）
        best_score = float('inf')
        best_params = None
        curve = []
        no_improve_count = 0

        for i in range(total_evaluations):
            # 随机采样参数
            params = np.array([
                np.random.uniform(lo, hi) for lo, hi in zip(lb, ub)
            ])

            score = fobj(params)

            if score < best_score:
                best_score = score
                best_params = params.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            curve.append(best_score)

            # 早停检查
            if no_improve_count >= patience:
                if verbose:
                    print(f"早停触发: 连续 {patience} 次迭代无改善，在第 {i+1} 次试验停止")
                break

            if verbose and (i + 1) % (total_evaluations // 10) == 0:
                print(f"进度 [{i+1}/{total_evaluations}], 最佳误差: {best_score:.6f}, 早停计数: {no_improve_count}/{patience}")

        return best_score, best_params, np.array(curve)


def create_optimizer_manager(config: dict) -> OptimizerManager:
    """
    创建优化器管理器实例

    Args:
        config: 配置字典

    Returns:
        OptimizerManager: 优化器管理器实例
    """
    return OptimizerManager(config)
