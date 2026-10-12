"""
模型训练模块
使用 PyCaret 进行模型管理和训练
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional, List, Tuple

# 导入 PyCaret
try:
    from pycaret.regression import (
        setup, compare_models, create_model, tune_model,
        finalize_model, predict_model, pull, get_config
    )
    PYCARET_AVAILABLE = True
    PYCARET_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    PYCARET_AVAILABLE = False
    PYCARET_IMPORT_ERROR = exc
    print(f"警告: PyCaret 不可用: {exc}")

# 导入 sklearn 模型（用于自定义优化器）
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import pickle


class ModelTrainer:
    """
    模型训练器

    使用 PyCaret 进行模型管理和训练
    支持多种回归模型和多输出回归
    """

    # PyCaret 模型名称映射
    PYCARET_MODEL_MAP = {
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'rf': 'rf',
        'et': 'et',           # Extra Trees
        'gbr': 'gbr',         # Gradient Boosting
        'ridge': 'ridge',
        'lasso': 'lasso',
        'en': 'en',           # Elastic Net
        'svm': 'svm',         # Support Vector Regression
        'knn': 'knn',         # K Neighbors Regressor
        'dt': 'dt',           # Decision Tree
        'mlp': 'mlp',         # MLP Regressor
        'ada': 'ada',         # AdaBoost Regressor
        'lr': 'lr',           # Linear Regression
    }

    # sklearn 模型类映射（用于自定义优化器）
    SKLEARN_MODEL_MAP = {
        # 集成学习
        'xgboost': XGBRegressor,
        'lightgbm': LGBMRegressor,
        'catboost': CatBoostRegressor,
        'rf': RandomForestRegressor,
        'et': ExtraTreesRegressor,
        'gbr': GradientBoostingRegressor,
        'ada': AdaBoostRegressor,
        # 线性模型
        'ridge': Ridge,
        'lasso': Lasso,
        'en': ElasticNet,
        'br': BayesianRidge,
        # 支持向量机和KNN
        'svr': SVR,
        'knn': KNeighborsRegressor,
        # 神经网络
        'mlp': MLPRegressor,
        # 决策树
        'dt': DecisionTreeRegressor,
        # 鲁棒回归
        'huber': HuberRegressor,
    }

    # 模型超参数边界（用于自定义优化器）
    HYPERPARAMETER_BOUNDS = {
        # 集成学习
        'xgboost': {
            'n_estimators': (50, 500),
            'max_depth': (3, 12),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.5, 1.0),
        },
        'lightgbm': {
            'n_estimators': (50, 500),
            'num_leaves': (20, 100),
            'learning_rate': (0.01, 0.3),
        },
        'catboost': {
            'iterations': (50, 500),
            'depth': (3, 10),
            'learning_rate': (0.01, 0.3),
        },
        'rf': {
            'n_estimators': (50, 500),
            'max_depth': (3, 15),
        },
        'et': {
            'n_estimators': (50, 500),
            'max_depth': (3, 15),
        },
        'gbr': {
            'n_estimators': (50, 500),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
        },
        'ada': {
            'n_estimators': (50, 300),
            'learning_rate': (0.01, 1.0),
        },
        # 线性模型
        'ridge': {
            'alpha': (0.01, 100),
        },
        'lasso': {
            'alpha': (0.0001, 1.0),
        },
        'en': {
            'alpha': (0.0001, 1.0),
            'l1_ratio': (0.1, 0.9),
        },
        'br': {
            'alpha_1': (1e-7, 1e-5),
            'lambda_1': (1e-7, 1e-5),
        },
        # 支持向量机和KNN
        'svr': {
            'C': (0.1, 100),
            'epsilon': (0.01, 0.5),
        },
        'knn': {
            'n_neighbors': (3, 20),
            'weights': [0, 1],  # 0: uniform, 1: distance
        },
        # 神经网络
        'mlp': {
            'alpha': (0.0001, 0.01),
        },
        # 决策树
        'dt': {
            'max_depth': (3, 20),
            'min_samples_split': (2, 20),
        },
        # 鲁棒回归
        'huber': {
            'epsilon': (1.1, 2.0),
            'alpha': (0.0001, 0.01),
        },
    }

    # 默认模型参数
    DEFAULT_PARAMS = {
        # 集成学习
        'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'n_jobs': 1},
        'lightgbm': {'n_estimators': 100, 'num_leaves': 31, 'learning_rate': 0.1, 'random_state': 42, 'verbose': -1},
        'catboost': {'iterations': 100, 'depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'verbose': 0},
        'rf': {'n_estimators': 100, 'random_state': 42, 'n_jobs': 1},
        'et': {'n_estimators': 100, 'random_state': 42, 'n_jobs': 1},
        'gbr': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
        'ada': {'n_estimators': 50, 'learning_rate': 1.0, 'random_state': 42},
        # 线性模型
        'ridge': {'alpha': 1.0},
        'lasso': {'alpha': 0.1},
        'en': {'alpha': 0.1, 'l1_ratio': 0.5},
        'br': {},
        # 支持向量机和KNN
        'svr': {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'},
        'knn': {'n_neighbors': 5, 'weights': 'uniform'},
        # 神经网络
        'mlp': {'hidden_layer_sizes': (100, 50), 'max_iter': 1000, 'random_state': 42},
        # 决策树
        'dt': {'max_depth': 10, 'random_state': 42},
        # 鲁棒回归
        'huber': {'epsilon': 1.35},
    }

    def __init__(self, config: dict):
        """
        初始化模型训练器

        Args:
            config: 模型配置字典
        """
        self.config = config
        self.model_name = config.get('default', 'xgboost')
        self.pycaret_model = None
        self.sklearn_model = None
        self.is_multioutput = True

    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        获取可用模型列表

        Returns:
            List[str]: 模型名称列表
        """
        return list(cls.PYCARET_MODEL_MAP.keys())

    @classmethod
    def get_hyperparameter_bounds(cls, model_name: str) -> Dict[str, tuple]:
        """
        获取模型的超参数边界

        Args:
            model_name: 模型名称

        Returns:
            Dict: 超参数边界字典
        """
        # 标准化模型名称
        pycaret_name = cls.PYCARET_MODEL_MAP.get(model_name, model_name)

        if pycaret_name in cls.HYPERPARAMETER_BOUNDS:
            return cls.HYPERPARAMETER_BOUNDS[pycaret_name].copy()

        # 返回默认边界
        return {
            'n_estimators': (50, 500),
            'max_depth': (3, 10),
        }

    # ==================== PyCaret 方法 ====================

    def create_pycaret_model(self,
                            model_name: Optional[str] = None,
                            **kwargs):
        """
        使用 PyCaret 创建模型

        Args:
            model_name: 模型名称
            **kwargs: 模型参数

        Returns:
            PyCaret 模型对象
        """
        if not PYCARET_AVAILABLE:
            raise RuntimeError("PyCaret 未安装，请运行: pip install pycaret")

        if model_name is None:
            model_name = self.model_name

        # 获取 PyCaret 模型名称
        pycaret_name = self.PYCARET_MODEL_MAP.get(model_name, model_name)

        # 创建模型
        model = create_model(pycaret_name, **kwargs)
        self.pycaret_model = model

        return model

    def compare_pycaret_models(self,
                              n_select: int = 3,
                              sort: str = 'MSE',
                              verbose: bool = True) -> List:
        """
        使用 PyCaret 比较多个模型

        Args:
            n_select: 选择的模型数量
            sort: 排序指标
            verbose: 是否显示详细输出

        Returns:
            List: 最佳模型列表
        """
        if not PYCARET_AVAILABLE:
            raise RuntimeError("PyCaret 未安装，请运行: pip install pycaret")

        best_models = compare_models(
            n_select=n_select,
            sort=sort,
            verbose=verbose
        )

        return best_models

    def tune_pycaret_model(self,
                          model,
                          n_iter: int = 10,
                          optimize: str = 'MSE',
                          verbose: bool = True):
        """
        使用 PyCaret 调优模型

        Args:
            model: PyCaret 模型对象
            n_iter: 迭代次数
            optimize: 优化指标
            verbose: 是否显示详细输出

        Returns:
            调优后的模型
        """
        if not PYCARET_AVAILABLE:
            raise RuntimeError("PyCaret 未安装，请运行: pip install pycaret")

        tuned_model = tune_model(
            model,
            n_iter=n_iter,
            optimize=optimize,
            verbose=verbose
        )

        self.pycaret_model = tuned_model

        return tuned_model

    def finalize_pycaret_model(self, model):
        """
        使用 PyCaret 完成模型训练（在全量数据上重新训练）

        Args:
            model: PyCaret 模型对象

        Returns:
            最终模型
        """
        if not PYCARET_AVAILABLE:
            raise RuntimeError("PyCaret 未安装，请运行: pip install pycaret")

        final_model = finalize_model(model)
        self.pycaret_model = final_model

        return final_model

    def get_pycaret_leaderboard(self) -> pd.DataFrame:
        """
        获取 PyCaret 模型比较排行榜

        Returns:
            pd.DataFrame: 模型排行榜
        """
        if not PYCARET_AVAILABLE:
            raise RuntimeError("PyCaret 未安装，请运行: pip install pycaret")

        return pull()

    # ==================== sklearn 方法（用于自定义优化器）====================

    def create_sklearn_model(self,
                            model_name: Optional[str] = None,
                            params: Optional[dict] = None):
        """
        创建 sklearn 兼容的模型（用于自定义优化器）

        Args:
            model_name: 模型名称
            params: 模型参数

        Returns:
            sklearn 模型对象
        """
        if model_name is None:
            model_name = self.model_name

        if model_name not in self.SKLEARN_MODEL_MAP:
            raise ValueError(f"未知模型: {model_name}，可用: {list(self.SKLEARN_MODEL_MAP.keys())}")

        model_class = self.SKLEARN_MODEL_MAP[model_name]

        # 合并参数
        merged_params = self.DEFAULT_PARAMS.get(model_name, {}).copy()
        if params:
            merged_params.update(params)

        model = model_class(**merged_params)

        if self.is_multioutput:
            model = MultiOutputRegressor(model)

        self.sklearn_model = model

        return model

    def build_model(self, params: np.ndarray, model_name: Optional[str] = None) -> Callable:
        """
        根据参数数组构建模型（用于优化器）

        Args:
            params: 参数数组
            model_name: 模型名称

        Returns:
            Callable: 模型构建函数
        """
        if model_name is None:
            model_name = self.model_name

        bounds = self.get_hyperparameter_bounds(model_name)
        param_names = list(bounds.keys())

        def builder(p):
            pd = {}
            for j, n in enumerate(param_names):
                if j < len(p):
                    v = p[j]
                    # 处理整数类型参数
                    if n in ['n_estimators', 'max_depth', 'num_leaves', 'iterations', 'depth']:
                        v = int(round(v))
                    pd[n] = v

            # 创建基础模型
            model_class = self.SKLEARN_MODEL_MAP.get(model_name)
            if model_class is None:
                raise ValueError(f"未知模型: {model_name}")

            base_model = model_class(**pd)

            if self.is_multioutput:
                return MultiOutputRegressor(base_model)
            return base_model

        return builder

    def get_param_bounds(self, model_name: Optional[str] = None) -> Tuple[List[float], List[float], List[str]]:
        """
        获取参数边界（用于优化器）

        Args:
            model_name: 模型名称

        Returns:
            Tuple: (lower_bounds, upper_bounds, param_names)
        """
        if model_name is None:
            model_name = self.model_name

        bounds = self.get_hyperparameter_bounds(model_name)

        lb = []
        ub = []
        param_names = []

        for name, (lo, hi) in bounds.items():
            lb.append(lo)
            ub.append(hi)
            param_names.append(name)

        return lb, ub, param_names

    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             model: Optional[Any] = None,
             model_name: Optional[str] = None,
             params: Optional[dict] = None) -> Any:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            model: 模型实例

        Returns:
            训练好的模型
        """
        if model is None and (model_name is not None or params is not None):
            model = self.create_sklearn_model(model_name=model_name, params=params)

        if model is None:
            model = self.sklearn_model

        if model is None:
            raise ValueError("请先创建模型")

        model.fit(X_train, y_train)
        self.sklearn_model = model
        return model

    def evaluate(self,
                 model: Any,
                 X_test: np.ndarray,
                 y_test: np.ndarray) -> float:
        """评估模型并返回统一平均 R2。"""
        y_pred = model.predict(X_test)
        return float(r2_score(y_test, y_pred, multioutput='uniform_average'))

    def save_model(self, model: Any, path: str) -> None:
        """保存模型到 pickle 文件。"""
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, path: str) -> Any:
        """从 pickle 文件加载模型。"""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self.sklearn_model = model
        return model

    def predict(self,
               X: np.ndarray,
               model: Optional[Any] = None) -> np.ndarray:
        """
        使用模型进行预测

        Args:
            X: 特征数据
            model: 模型实例

        Returns:
            np.ndarray: 预测结果
        """
        if model is None:
            model = self.sklearn_model or self.pycaret_model

        if model is None:
            raise ValueError("请先训练模型或提供模型实例")

        return model.predict(X)


def create_model_trainer(config: dict) -> ModelTrainer:
    """
    创建模型训练器实例

    Args:
        config: 模型配置字典

    Returns:
        ModelTrainer: 模型训练器实例
    """
    return ModelTrainer(config)


def get_model_builder(model_name: str, config: dict) -> Tuple[Callable, List[float], List[float], List[str]]:
    """
    获取模型构建器和参数边界

    Args:
        model_name: 模型名称
        config: 配置字典

    Returns:
        Tuple: (model_builder, lb, ub, param_names)
    """
    trainer = ModelTrainer(config)
    trainer.model_name = model_name

    lb, ub, param_names = trainer.get_param_bounds(model_name)
    builder = trainer.build_model(np.array([(l + u) / 2 for l, u in zip(lb, ub)]))

    return builder, lb, ub, param_names
