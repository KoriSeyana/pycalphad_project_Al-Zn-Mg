"""数据加载模块，用于读取原始数据和提取成分范围"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器，用于读取data.xlsx和提取成分信息"""

    # data.xlsx中的标准列名
    COMPOSITION_COLS = [
        "Zn/％",
        "Mg/％",
        "Cu/％",
        "Cr/％",
        "Mn/％",
        "Zr/％",
        "Ti/％",
        "Fe/％",
        "Si/％",
    ]

    PROCESS_COLS = [
        "Primary solution temperature (℃)",
        "Primary solution time (min)",
        "Secondary solution temperature (℃)",
        "Secondary solution time (min)",
        "Primary aging temperature (℃)",
        "Primary aging time (min)",
        "Secondary aging temperature (℃)",
        "Secondary aging time (min)",
    ]

    TARGET_COLS = [
        "Ultimate tensile strength (MPa)",
        "Elongation (%)",
    ]

    def __init__(self, data_path: str | Path):
        """初始化数据加载器

        Args:
            data_path: data.xlsx文件路径
        """
        self.data_path = Path(data_path)
        self._df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame:
        """加载并缓存数据框"""
        if self._df is None:
            self._df = self._load_data()
        return self._df

    def _load_data(self) -> pd.DataFrame:
        """加载Excel数据"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        df = pd.read_excel(self.data_path)
        logger.info(f"加载数据: {df.shape[0]} 行 x {df.shape[1]} 列")
        return df

    def get_composition_range(self) -> dict[str, tuple[float, float]]:
        """提取各元素的成分范围

        Returns:
            dict: {元素列名: (最小值, 最大值)}
        """
        ranges = {}
        for col in self.COMPOSITION_COLS:
            if col in self.df.columns:
                min_val = float(self.df[col].min())
                max_val = float(self.df[col].max())
                ranges[col] = (min_val, max_val)
        return ranges

    def get_process_range(self) -> dict[str, tuple[float, float]]:
        """提取工艺参数范围

        Returns:
            dict: {工艺参数列名: (最小值, 最大值)}
        """
        ranges = {}
        for col in self.PROCESS_COLS:
            if col in self.df.columns:
                min_val = float(self.df[col].min())
                max_val = float(self.df[col].max())
                ranges[col] = (min_val, max_val)
        return ranges

    def get_statistics(self) -> dict[str, Any]:
        """获取数据统计信息

        Returns:
            dict: 统计信息字典
        """
        stats = {
            "total_samples": len(self.df),
            "composition_range": self.get_composition_range(),
            "process_range": self.get_process_range(),
            "target_stats": {},
        }

        for col in self.TARGET_COLS:
            if col in self.df.columns:
                stats["target_stats"][col] = {
                    "mean": float(self.df[col].mean()),
                    "std": float(self.df[col].std()),
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                }

        return stats

    def generate_param_space_from_data(self) -> dict[str, Any]:
        """从原始数据生成参数空间配置

        Returns:
            dict: 参数空间配置
        """
        comp_range = self.get_composition_range()
        proc_range = self.get_process_range()

        # 提取列名中的元素符号
        element_map = {
            "Zn/％": "ZN",
            "Mg/％": "MG",
            "Cu/％": "CU",
            "Cr/％": "CR",
            "Mn/％": "MN",
            "Zr/％": "ZR",
            "Ti/％": "TI",
            "Fe/％": "FE",
            "Si/％": "SI",
        }

        composition = {}
        for col_name, elem in element_map.items():
            if col_name in comp_range:
                composition[elem] = comp_range[col_name]

        # 提取温度范围
        temperature = {}
        if "Primary solution temperature (℃)" in proc_range:
            temperature["solution_temp"] = proc_range["Primary solution temperature (℃)"]
        if "Primary aging temperature (℃)" in proc_range:
            temperature["aging_temp"] = proc_range["Primary aging temperature (℃)"]

        return {
            "composition": composition,
            "temperature": temperature,
        }
