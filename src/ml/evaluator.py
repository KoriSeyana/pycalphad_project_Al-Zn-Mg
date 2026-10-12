"""
模型评估模块
计算和展示模型性能指标
"""

import os
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


class ModelEvaluator:
    """
    模型评估器

    计算多种回归评估指标
    """

    def __init__(self, target_names: Optional[List[str]] = None):
        """
        初始化评估器

        Args:
            target_names: 目标变量名称列表
        """
        self.target_names = target_names if target_names else ['Target_1', 'Target_2']

    def calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray) -> pd.DataFrame:
        """
        计算各项评估指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            pd.DataFrame: 包含各项指标的 DataFrame
        """
        # 确保输入是 numpy 数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics_results = {}

        # 针对每个输出目标单独计算指标
        for i, name in enumerate(self.target_names):
            y_t = y_true[:, i] if y_true.ndim > 1 else y_true
            y_p = y_pred[:, i] if y_pred.ndim > 1 else y_pred

            mse = mean_squared_error(y_t, y_p)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_t, y_p)
            r2 = r2_score(y_t, y_p)
            mape = mean_absolute_percentage_error(y_t, y_p) * 100  # 转为百分比

            metrics_results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE(%)': mape,
            }

        return pd.DataFrame(metrics_results).T

    def print_table(self, metrics_df: pd.DataFrame):
        """
        打印评估结果表格

        Args:
            metrics_df: 评估指标 DataFrame
        """
        print("\n" + "=" * 60)
        print("模型预测性能评估结果")
        print("=" * 60)
        print(metrics_df.round(4))
        print("=" * 60)

    def save_to_csv(self,
                   metrics_df: pd.DataFrame,
                   filename: str = 'model_performance_metrics.csv',
                   output_dir: str = 'results'):
        """
        保存评估结果到 CSV 文件

        Args:
            metrics_df: 评估指标 DataFrame
            filename: 文件名
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)
        metrics_df.to_csv(filepath)
        print(f"评估结果已保存至: {filepath}")

    def compare_optimizers(self,
                          results: Dict[str, tuple],
                          output_dir: str = 'results') -> pd.DataFrame:
        """
        对比不同优化器的结果

        Args:
            results: 优化器结果字典 {name: (best_score, best_params, curve)}
            output_dir: 输出目录

        Returns:
            pd.DataFrame: 对比结果
        """
        comparison = []

        for opt_name, (best_score, best_params, curve) in results.items():
            comparison.append({
                'Optimizer': opt_name,
                'Best_MSE': best_score,
                'Best_Params': str(best_params),
                'Final_Error': curve[-1] if len(curve) > 0 else best_score,
            })

        df = pd.DataFrame(comparison)
        df = df.sort_values('Best_MSE')

        # 保存对比结果
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'optimizer_comparison.csv')
        df.to_csv(filepath, index=False)
        print(f"\n优化器对比结果已保存至: {filepath}")

        return df

    def print_comparison(self, comparison_df: pd.DataFrame):
        """
        打印优化器对比结果

        Args:
            comparison_df: 对比结果 DataFrame
        """
        print("\n" + "=" * 60)
        print("优化器对比结果 (按最佳 MSE 排序)")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        print("=" * 60)


def create_evaluator(config: dict) -> ModelEvaluator:
    """
    创建评估器实例

    Args:
        config: 配置字典

    Returns:
        ModelEvaluator: 评估器实例
    """
    eval_config = config.get('eval', {})
    target_names = eval_config.get('target_names', ['Target_1', 'Target_2'])

    return ModelEvaluator(target_names=target_names)
