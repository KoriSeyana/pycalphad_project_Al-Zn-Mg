"""
代理模型接口
封装已训练的 LightGBM 代理模型，用于多目标优化中的相分数预测
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional


class SurrogateModel:
    """
    代理模型接口

    加载已训练的 LightGBM 模型，提供相分数预测功能
    输入: 成分 + 固溶温度 + 时效温度 (11 维)
    输出: 固溶态和时效态的相分数 (16 维)
    """

    def __init__(self, model_path: Union[str, Path]):
        """
        初始化代理模型

        Args:
            model_path: 模型文件路径 (.pkl)
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()

        # 输入特征 (11 维)
        self.input_features = [
            'ZN', 'MG', 'CU', 'CR', 'MN', 'ZR', 'TI', 'FE', 'SI',
            'solution_temp', 'aging_temp'
        ]

        # 输出特征 (16 维)
        self.output_features = [
            'sol_FCC_A1_fraction', 'sol_ETA_fraction', 'sol_T_PHASE_fraction',
            'sol_S_PHASE_fraction', 'sol_AL3ZR_fraction', 'sol_MG2SI_fraction',
            'sol_AL_FE_SI_fraction', 'sol_HCP_A3_fraction',
            'age_FCC_A1_fraction', 'age_ETA_fraction', 'age_T_PHASE_fraction',
            'age_S_PHASE_fraction', 'age_AL3ZR_fraction', 'age_MG2SI_fraction',
            'age_AL_FE_SI_fraction', 'age_HCP_A3_fraction'
        ]

        info_path = self.model_path.with_name('model_info.json')
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding='utf-8'))
            self.input_features = info.get('input_features', self.input_features)
            self.output_features = info.get('output_features', self.output_features)

        self.is_fitted = True

    def _load_model(self):
        """加载模型文件"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)

        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        批量预测相分数

        Args:
            X: 输入特征矩阵 (n_samples, 11)
               列顺序: [ZN, MG, CU, CR, MN, ZR, TI, FE, SI, solution_temp, aging_temp]

        Returns:
            y: 预测的相分数矩阵 (n_samples, 16)
               列顺序: [sol_FCC_A1, sol_ETA, ..., age_FCC_A1, age_ETA, ...]
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(self.input_features):
            raise ValueError(
                f"输入特征维度错误: 期望 {len(self.input_features)}, "
                f"实际 {X.shape[1]}"
            )

        return self.model.predict(X)

    def predict_single(
        self,
        composition: List[float],
        solution_temp: float,
        aging_temp: float
    ) -> np.ndarray:
        """
        单样本预测

        Args:
            composition: 成分列表 [Zn, Mg, Cu, Cr, Mn, Zr, Ti, Fe, Si] (wt%)
            solution_temp: 固溶温度 (°C)
            aging_temp: 时效温度 (°C)

        Returns:
            相分数数组 (16,)
        """
        if len(composition) != 9:
            raise ValueError(f"成分列表长度应为 9, 实际为 {len(composition)}")

        X = np.array([composition + [solution_temp, aging_temp]])
        return self.predict(X)[0]

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从 DataFrame 预测

        Args:
            df: 包含输入特征的 DataFrame

        Returns:
            包含预测结果的 DataFrame
        """
        # 检查必需的列
        missing_cols = set(self.input_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"缺少输入特征列: {missing_cols}")

        X = df[self.input_features].values
        y_pred = self.predict(X)

        result_df = pd.DataFrame(y_pred, columns=self.output_features, index=df.index)
        return result_df

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'model_path': str(self.model_path),
            'input_features': self.input_features,
            'output_features': self.output_features,
            'n_input': len(self.input_features),
            'n_output': len(self.output_features),
            'is_fitted': self.is_fitted,
        }


def load_surrogate_model(model_path: Optional[str] = None) -> SurrogateModel:
    """
    便捷函数: 加载代理模型

    Args:
        model_path: 模型路径,如果为 None 则使用默认路径

    Returns:
        SurrogateModel 实例
    """
    if model_path is None:
        from config import get_config
        config = get_config()
        model_path = config['surrogate']['model_path']

    return SurrogateModel(model_path)
