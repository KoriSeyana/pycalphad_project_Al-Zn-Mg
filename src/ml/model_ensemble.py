"""
模型集成模块
通过 Stacking、Blending、加权平均等方法提高预测精度
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import BaseEstimator, RegressorMixin
from typing import List, Dict, Any, Optional


class StackingRegressor(BaseEstimator, RegressorMixin):
    """
    Stacking 集成回归器

    使用多个基础模型的预测结果作为特征，
    训练一个元模型进行最终预测
    """

    def __init__(self,
                 base_models: Dict[str, Any],
                 meta_model: Any = None,
                 n_folds: int = 5,
                 use_features_in_meta: bool = False):
        """
        Args:
            base_models: 基础模型字典 {name: model}
            meta_model: 元模型
            n_folds: 交叉验证折数
            use_features_in_meta: 元模型是否使用原始特征
        """
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.n_folds = n_folds
        self.use_features_in_meta = use_features_in_meta
        self.fitted_base_models = {}
        self.fitted_meta_model = None

    def _get_base_predictions(self, X: np.ndarray, y: np.ndarray = None, fit: bool = True) -> np.ndarray:
        """
        获取基础模型的预测结果

        Args:
            X: 特征
            y: 目标（训练时使用）
            fit: 是否训练模型

        Returns:
            np.ndarray: 基础模型的预测结果
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        if fit:
            # 训练模式：使用交叉验证生成预测
            n_outputs = y.shape[1] if y.ndim > 1 else 1
            predictions = np.zeros((n_samples, n_models * n_outputs))
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

            # 存储每个折训练的模型
            fold_models = {name: [] for name in self.base_models.keys()}

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                for model_idx, (name, model) in enumerate(self.base_models.items()):
                    # 克隆模型
                    from sklearn.base import clone
                    fold_model = clone(model)

                    # 包装为多输出
                    if not hasattr(fold_model, 'estimators_'):
                        fold_model = MultiOutputRegressor(fold_model)

                    fold_model.fit(X_train, y_train)
                    pred = fold_model.predict(X_val)

                    predictions[val_idx, model_idx*n_outputs:(model_idx+1)*n_outputs] = pred
                    fold_models[name].append(fold_model)

            # 使用全部数据训练最终的基础模型
            self.fitted_base_models = {}
            self.n_outputs = n_outputs
            for name, model in self.base_models.items():
                from sklearn.base import clone
                final_model = clone(model)
                if not hasattr(final_model, 'estimators_'):
                    final_model = MultiOutputRegressor(final_model)
                final_model.fit(X, y)
                self.fitted_base_models[name] = final_model

            return predictions
        else:
            # 预测模式：使用训练好的模型
            n_outputs = getattr(self, 'n_outputs', 2)
            predictions = np.zeros((n_samples, n_models * n_outputs))

            for model_idx, (name, model) in enumerate(self.fitted_base_models.items()):
                pred = model.predict(X)
                predictions[:, model_idx*n_outputs:(model_idx+1)*n_outputs] = pred

            return predictions

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型

        Args:
            X: 特征
            y: 目标

        Returns:
            self
        """
        # 获取基础模型的预测
        base_predictions = self._get_base_predictions(X, y, fit=True)

        # 准备元模型的输入
        if self.use_features_in_meta:
            meta_X = np.hstack([X, base_predictions])
        else:
            meta_X = base_predictions

        # 训练元模型
        self.fitted_meta_model = MultiOutputRegressor(self.meta_model)
        self.fitted_meta_model.fit(meta_X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 特征

        Returns:
            np.ndarray: 预测结果
        """
        # 获取基础模型的预测
        base_predictions = self._get_base_predictions(X, fit=False)

        # 准备元模型的输入
        if self.use_features_in_meta:
            meta_X = np.hstack([X, base_predictions])
        else:
            meta_X = base_predictions

        return self.fitted_meta_model.predict(meta_X)


class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    加权集成回归器

    根据各模型的交叉验证性能自动分配权重
    """

    def __init__(self,
                 models: Dict[str, Any],
                 n_folds: int = 5):
        """
        Args:
            models: 模型字典 {name: model}
            n_folds: 交叉验证折数
        """
        self.models = models
        self.n_folds = n_folds
        self.weights = None
        self.fitted_models = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型并计算权重

        Args:
            X: 特征
            y: 目标

        Returns:
            self
        """
        from sklearn.model_selection import cross_val_score

        scores = {}

        for name, model in self.models.items():
            from sklearn.base import clone
            wrapped_model = clone(model)
            if not hasattr(wrapped_model, 'estimators_'):
                wrapped_model = MultiOutputRegressor(wrapped_model)

            # 交叉验证
            cv_scores = cross_val_score(
                wrapped_model, X, y,
                cv=self.n_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            scores[name] = -cv_scores.mean()

        # 计算权重（MSE 越小权重越大）
        total_inverse_mse = sum(1/s for s in scores.values())
        self.weights = {name: (1/score)/total_inverse_mse for name, score in scores.items()}

        print("模型权重:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f} (MSE: {scores[name]:.4f})")

        # 使用全部数据训练所有模型
        self.fitted_models = {}
        for name, model in self.models.items():
            from sklearn.base import clone
            final_model = clone(model)
            if not hasattr(final_model, 'estimators_'):
                final_model = MultiOutputRegressor(final_model)
            final_model.fit(X, y)
            self.fitted_models[name] = final_model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        加权预测

        Args:
            X: 特征

        Returns:
            np.ndarray: 预测结果
        """
        predictions = []

        for name, model in self.fitted_models.items():
            pred = model.predict(X)
            predictions.append(pred * self.weights[name])

        return np.sum(predictions, axis=0)


def create_stacking_model(model_configs: Dict[str, Dict]) -> StackingRegressor:
    """
    创建 Stacking 模型

    Args:
        model_configs: 模型配置字典

    Returns:
        StackingRegressor: Stacking 模型
    """
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    model_classes = {
        'xgboost': XGBRegressor,
        'lightgbm': LGBMRegressor,
        'catboost': CatBoostRegressor,
        'rf': RandomForestRegressor,
        'gbr': GradientBoostingRegressor,
    }

    base_models = {}
    for name, config in model_configs.items():
        if name in model_classes:
            base_models[name] = model_classes[name](**config)

    return StackingRegressor(
        base_models=base_models,
        meta_model=Ridge(alpha=1.0),
        n_folds=5
    )


def create_weighted_ensemble(model_configs: Dict[str, Dict]) -> WeightedEnsembleRegressor:
    """
    创建加权集成模型

    Args:
        model_configs: 模型配置字典

    Returns:
        WeightedEnsembleRegressor: 加权集成模型
    """
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    model_classes = {
        'xgboost': XGBRegressor,
        'lightgbm': LGBMRegressor,
        'catboost': CatBoostRegressor,
    }

    models = {}
    for name, config in model_configs.items():
        if name in model_classes:
            models[name] = model_classes[name](**config)

    return WeightedEnsembleRegressor(models=models, n_folds=5)


if __name__ == "__main__":
    print("模型集成模块")
    print("="*50)
    print("可用功能:")
    print("1. StackingRegressor - Stacking 集成")
    print("2. WeightedEnsembleRegressor - 加权集成")
    print("3. create_stacking_model - 创建 Stacking 模型")
    print("4. create_weighted_ensemble - 创建加权集成模型")
