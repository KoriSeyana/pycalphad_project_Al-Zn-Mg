"""
多目标优化 - 流水线目标函数设计
基于pymoo的自定义Problem类

架构：
NSGA-II → 决策变量 → 特征补齐(微代理) → 完整特征 → 主模型预测 → 返回目标值
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score


# ==================== 1. 配置类 ====================

@dataclass
class OptimizationConfig:
    """优化配置"""

    # 文件路径
    main_model_path: str = 'results/best_model/best_model.pkl'
    enhanced_data_path: str = 'data/enhanced_dataset_full_utf8sig.csv'
    surrogate_model_path: str = 'results/surrogate/phase_predictor.pkl'

    # NSGA-II参数
    pop_size: int = 100
    n_gen: int = 300

    # 决策变量定义 (名称, 下界, 上界)
    decision_variables: List[Tuple[str, float, float]] = None

    def __post_init__(self):
        if self.decision_variables is None:
            self.decision_variables = [
                # 元素成分 (%)
                ('Zn', 4.0, 12.0),
                ('Mg', 1.0, 4.0),
                ('Cu', 0.0, 3.0),
                ('Cr', 0.0, 0.35),
                ('Mn', 0.0, 0.5),
                ('Zr', 0.0, 0.25),
                ('Ti', 0.0, 0.15),
                ('Fe', 0.0, 0.5),
                ('Si', 0.0, 0.5),
                # 固溶参数
                ('solution_temp', 440, 500),
                ('solution_time', 30, 240),
                # 时效参数
                ('aging_temp', 100, 200),
                ('aging_time', 120, 1440),
            ]


# ==================== 2. 相分数预测器（微代理模型） ====================

class PhaseFractionPredictor:
    """
    相分数预测器
    输入：成分 + 温度
    输出：各相分数

    三种实现方案：
    A. 使用已训练的代理模型（推荐，R²=0.9877）
    B. 基于轻量回归器（需要训练）
    C. 基于插值（适合低维）
    """

    def __init__(self, method='surrogate', surrogate_model_path=None):
        """
        Args:
            method: 'surrogate', 'lightgbm', 或 'interpolation'
            surrogate_model_path: 代理模型路径（method='surrogate'时使用）
        """
        self.method = method
        self.model = None
        self.feature_cols = None
        self.target_cols = None
        self.is_fitted = False

        # 如果使用代理模型，直接加载
        if method == 'surrogate':
            from src.optimization.surrogate_model import load_surrogate_model
            self.model = load_surrogate_model(surrogate_model_path)
            self.feature_cols = self.model.input_features
            self.target_cols = self.model.output_features
            self.is_fitted = True
            print(f"已加载代理模型: {self.model.model_path}")
            print(f"  输入特征: {len(self.feature_cols)}")
            print(f"  输出特征: {len(self.target_cols)}")

    def fit(self, df: pd.DataFrame,
            feature_cols: List[str],
            target_cols: List[str]):
        """
        训练相分数预测器

        Args:
            df: 包含CALPHAD计算结果的DataFrame
            feature_cols: 输入特征列（成分+温度）
            target_cols: 目标列（各相分数）
        """
        # 如果已经加载了代理模型，跳过训练
        if self.method == 'surrogate' and self.is_fitted:
            print("代理模型已加载，跳过训练")
            return

        self.feature_cols = feature_cols
        self.target_cols = target_cols

        X = df[feature_cols].values
        y = df[target_cols].values

        if self.method == 'lightgbm':
            # 使用LightGBM作为微代理模型
            base_model = LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                verbose=-1,
                n_jobs=-1
            )
            self.model = MultiOutputRegressor(base_model)
            self.model.fit(X, y)

            # 评估模型质量
            print("相分数预测器训练完成")
            print(f"  特征数: {len(feature_cols)}")
            print(f"  目标数: {len(target_cols)}")
            print(f"  样本数: {len(df)}")

        elif self.method == 'interpolation':
            # 基于KDTree的插值（仅适合低维）
            from scipy.interpolate import LinearNDInterpolator
            from scipy.spatial import Delaunay

            # 检查维度
            if len(feature_cols) > 4:
                print("警告：维度>4，插值效果可能不佳")

            self.model = LinearNDInterpolator(X, y, fill_value=0)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测相分数

        Args:
            X: 输入特征 (n_samples, n_features)

        Returns:
            np.ndarray: 预测的相分数 (n_samples, n_targets)
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()")

        return self.model.predict(X)

    def save(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'PhaseFractionPredictor':
        """加载模型"""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ==================== 3. 特征拼接器 ====================

class FeatureAssembler:
    """
    特征拼接器
    将决策变量 + 相分数 → 完整156维特征向量
    """

    def __init__(self,
                 main_model_features: List[str],
                 decision_var_names: List[str],
                 phase_predictor: PhaseFractionPredictor):
        """
        Args:
            main_model_features: 主模型需要的特征列名列表
            decision_var_names: 决策变量名称列表
            phase_predictor: 相分数预测器
        """
        self.main_model_features = main_model_features
        self.decision_var_names = decision_var_names
        self.phase_predictor = phase_predictor

        # 建立决策变量到特征的映射
        self.decision_to_feature = self._build_mapping()

        # 确定哪些特征需要由预测器生成
        self.predicted_features = [
            f for f in main_model_features
            if f not in decision_var_names and '_fraction' in f
        ]

    def _build_mapping(self) -> Dict[str, str]:
        """建立决策变量名到特征名的映射"""
        mapping = {
            'Zn': 'Zn/％',
            'Mg': 'Mg/％',
            'Cu': 'Cu/％',
            'Cr': 'Cr/％',
            'Mn': 'Mn/％',
            'Zr': 'Zr/％',
            'Ti': 'Ti/％',
            'Fe': 'Fe/％',
            'Si': 'Si/％',
            'solution_temp': 'Primary solution temperature (℃)',
            'solution_time': 'Primary solution time (min)',
            'aging_temp': 'Primary aging temperature (℃)',
            'aging_time': 'Primary aging time (min)',
        }
        return mapping

    def assemble(self, decision_vars: Dict[str, float]) -> np.ndarray:
        """
        组装完整特征向量

        Args:
            decision_vars: 决策变量字典 {变量名: 值}

        Returns:
            np.ndarray: 完整特征向量
        """
        # 1. 初始化特征向量
        feature_dict = {f: 0.0 for f in self.main_model_features}

        # 2. 填充决策变量对应的特征
        for var_name, value in decision_vars.items():
            if var_name in self.decision_to_feature:
                feature_name = self.decision_to_feature[var_name]
                if feature_name in feature_dict:
                    feature_dict[feature_name] = value

        # 3. 使用微代理预测相分数
        # 构建预测器输入
        predictor_input = np.array([[
            decision_vars.get('Zn', 0),
            decision_vars.get('Mg', 0),
            decision_vars.get('Cu', 0),
            decision_vars.get('Cr', 0),
            decision_vars.get('Mn', 0),
            decision_vars.get('Zr', 0),
            decision_vars.get('Ti', 0),
            decision_vars.get('Fe', 0),
            decision_vars.get('Si', 0),
            decision_vars.get('solution_temp', 0),
            decision_vars.get('aging_temp', 0),
        ]])

        # 预测相分数
        phase_fractions = self.phase_predictor.predict(predictor_input)[0]

        # 填充相分数特征
        for i, feat_name in enumerate(self.predicted_features):
            if feat_name in feature_dict and i < len(phase_fractions):
                feature_dict[feat_name] = phase_fractions[i]

        # 4. 转换为数组（保持与主模型一致的顺序）
        feature_array = np.array([feature_dict[f] for f in self.main_model_features])

        return feature_array


# ==================== 4. 多目标优化问题类 ====================

class AlloyOptimizationPipeline(Problem):
    """
    合金多目标优化问题 - 流水线实现

    流水线：
    1. NSGA-II生成决策变量
    2. 微代理补齐相分数
    3. 拼接完整特征
    4. 主模型预测目标值
    """

    def __init__(self,
                 config: OptimizationConfig,
                 main_model,
                 feature_assembler: FeatureAssembler):
        """
        Args:
            config: 优化配置
            main_model: 主预测模型（TrAdaBoost）
            feature_assembler: 特征拼接器
        """
        self.config = config
        self.main_model = main_model
        self.assembler = feature_assembler

        # 提取边界
        var_names = [v[0] for v in config.decision_variables]
        lb = np.array([v[1] for v in config.decision_variables])
        ub = np.array([v[2] for v in config.decision_variables])

        n_var = len(var_names)
        n_obj = 2  # UTS, Elongation

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=lb,
            xu=ub
        )

        self.var_names = var_names

        print(f"优化问题定义完成:")
        print(f"  决策变量数: {n_var}")
        print(f"  目标数: {n_obj}")
        print(f"  变量: {var_names}")

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        """
        评估函数 - 实现4步流水线

        Args:
            X: 决策变量矩阵 (pop_size, n_var)
            out: 输出字典，设置 out["F"]
        """
        n_pop = X.shape[0]
        F = np.zeros((n_pop, 2))

        for i in range(n_pop):
            # ============================================
            # Step 1: 提取决策变量
            # ============================================
            decision_vars = {
                self.var_names[j]: X[i, j]
                for j in range(self.n_var)
            }

            # ============================================
            # Step 2: 特征补齐（微代理预测相分数）
            # ============================================
            # 在FeatureAssembler.assemble()中完成

            # ============================================
            # Step 3: 拼接完整156维特征
            # ============================================
            full_features = self.assembler.assemble(decision_vars)

            # ============================================
            # Step 4: 主模型预测
            # ============================================
            try:
                pred = self.main_model.predict(full_features.reshape(1, -1))[0]
                uts = pred[0]
                elong = pred[1]
            except Exception as e:
                # 预测失败时返回惩罚值
                uts = 0
                elong = 0

            # pymoo默认最小化，取负
            F[i, 0] = -uts
            F[i, 1] = -elong

        out["F"] = F


# ==================== 5. 训练相分数预测器 ====================

def train_phase_predictor(data_path: str,
                          save_path: str,
                          method: str = 'lightgbm') -> PhaseFractionPredictor:
    """
    训练相分数预测器

    Args:
        data_path: 增强数据集路径
        save_path: 模型保存路径
        method: 预测方法

    Returns:
        训练好的预测器
    """
    print("训练相分数预测器...")

    # 加载数据
    df = pd.read_csv(data_path)

    # 填充NaN为0
    df.fillna(0, inplace=True)

    # 定义输入特征（成分+温度）
    input_features = [
        'Zn/％', 'Mg/％', 'Cu/％', 'Cr/％', 'Mn/％', 'Zr/％', 'Ti/％', 'Fe/％', 'Si/％',
        'Primary solution temperature (℃)',
        'Primary aging temperature (℃)',
    ]

    # 定义目标（相分数）
    target_features = [c for c in df.columns if '_fraction' in c]

    # 过滤掉零方差的目标
    target_features = [
        t for t in target_features
        if df[t].std() > 1e-6
    ]

    print(f"  输入特征: {len(input_features)}")
    print(f"  目标特征: {len(target_features)}")

    # 训练预测器
    predictor = PhaseFractionPredictor(method=method)
    predictor.fit(df, input_features, target_features)

    # 保存
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    predictor.save(save_path)
    print(f"  模型已保存: {save_path}")

    return predictor


# ==================== 6. 主优化流程 ====================

def run_optimization(config: OptimizationConfig = None):
    """
    运行多目标优化

    Args:
        config: 优化配置
    """
    if config is None:
        config = OptimizationConfig()

    print("="*70)
    print("多目标优化 - 流水线架构")
    print("="*70)

    # 1. 加载主模型
    print("\n[1/4] 加载主预测模型...")
    with open(config.main_model_path, 'rb') as f:
        main_model = pickle.load(f)
    print("  主模型加载完成")

    # 2. 训练或加载相分数预测器
    print("\n[2/4] 准备相分数预测器...")
    import os
    if os.path.exists(config.surrogate_model_path):
        phase_predictor = PhaseFractionPredictor.load(config.surrogate_model_path)
        print("  从文件加载预测器")
    else:
        phase_predictor = train_phase_predictor(
            config.enhanced_data_path,
            config.surrogate_model_path
        )

    # 3. 构建特征拼接器
    print("\n[3/4] 构建特征拼接器...")

    # 获取主模型的特征列名
    with open('results/best_model/feature_columns.txt', 'r') as f:
        main_model_features = [line.strip() for line in f.readlines()]

    var_names = [v[0] for v in config.decision_variables]

    assembler = FeatureAssembler(
        main_model_features=main_model_features,
        decision_var_names=var_names,
        phase_predictor=phase_predictor
    )
    print("  特征拼接器构建完成")

    # 4. 定义优化问题
    print("\n[4/4] 定义优化问题...")
    problem = AlloyOptimizationPipeline(
        config=config,
        main_model=main_model,
        feature_assembler=assembler
    )

    # 5. 配置算法
    algorithm = NSGA2(
        pop_size=config.pop_size,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
    )

    # 6. 运行优化
    print("\n开始优化...")
    print(f"  种群大小: {config.pop_size}")
    print(f"  迭代代数: {config.n_gen}")

    result = minimize(
        problem,
        algorithm,
        ('n_gen', config.n_gen),
        seed=42,
        verbose=True
    )

    print(f"\n优化完成! 找到 {len(result.F)} 个Pareto最优解")

    return result, problem


# ==================== 7. 测试入口 ====================

if __name__ == "__main__":
    # 快速测试
    config = OptimizationConfig()
    config.pop_size = 20
    config.n_gen = 10

    result, problem = run_optimization(config)

    # 输出结果
    if result is not None:
        F = result.F
        UTS_values = -F[:, 0]
        Elong_values = -F[:, 1]

        print("\n" + "="*70)
        print("Pareto前沿")
        print("="*70)
        print(f"UTS范围: {UTS_values.min():.2f} - {UTS_values.max():.2f} MPa")
        print(f"Elongation范围: {Elong_values.min():.2f} - {Elong_values.max():.2f} %")
