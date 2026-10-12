"""Load an input workbook and extract composition and process ranges."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Read a workbook and extract alloy composition and process information."""

    # Standard workbook column names.
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

    PROCESS_KEYWORDS = {
        "solution_temp": "primary solution temperature",
        "aging_temp": "primary aging temperature",
    }

    TARGET_COLS = [
        "Ultimate tensile strength (MPa)",
        "Elongation (%)",
    ]

    def __init__(self, data_path: str | Path):
        """初始化数据加载器

        Args:
            data_path: Path to an input workbook.
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

    def _find_column(self, target_name: str) -> str | None:
        """按精确名或宽松规则查找列名。"""
        if target_name in self.df.columns:
            return target_name

        def _normalize_name(name: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", name.lower())

        target_lower = target_name.lower()
        target_norm = _normalize_name(target_name)
        for col in self.df.columns:
            col_lower = col.lower()
            if target_lower in col_lower:
                return col
            if target_norm and target_norm == _normalize_name(col):
                return col

        if "/" in target_name:
            prefix = target_name.split("/", 1)[0].lower()
            for col in self.df.columns:
                if col.lower().startswith(f"{prefix}/"):
                    return col

        return None

    def _get_numeric_series(self, col_name: str) -> pd.Series:
        """读取并转换数值列，自动忽略非法值。"""
        return pd.to_numeric(self.df[col_name], errors="coerce").dropna()

    def get_composition_range(self) -> dict[str, tuple[float, float]]:
        """提取各元素的成分范围

        Returns:
            dict: {元素列名: (最小值, 最大值)}
        """
        ranges = {}
        for col in self.COMPOSITION_COLS:
            actual_col = self._find_column(col)
            if actual_col is None:
                continue
            series = self._get_numeric_series(actual_col)
            if series.empty:
                continue
            min_val = float(series.min())
            max_val = float(series.max())
            ranges[col] = (min_val, max_val)
        return ranges

    def get_process_range(self) -> dict[str, tuple[float, float]]:
        """提取工艺参数范围

        Returns:
            dict: {工艺参数列名: (最小值, 最大值)}
        """
        ranges = {}
        for col in self.PROCESS_COLS:
            actual_col = self._find_column(col)
            if actual_col is None:
                continue
            series = self._get_numeric_series(actual_col)
            if series.empty:
                continue
            min_val = float(series.min())
            max_val = float(series.max())
            ranges[col] = (min_val, max_val)
        return ranges

    def get_top_temperature_levels(
        self,
        top_k: int = 5,
        min_count: int = 2,
    ) -> dict[str, list[tuple[float, int]]]:
        """提取温度高频档位。

        Returns:
            dict: {"solution_temp": [(温度, 频次), ...], "aging_temp": [...]}。
        """
        levels: dict[str, list[tuple[float, int]]] = {}

        for key, keyword in self.PROCESS_KEYWORDS.items():
            col = next(
                (c for c in self.df.columns if keyword in c.lower()),
                None,
            )
            if col is None:
                continue

            series = self._get_numeric_series(col).round(3)
            if series.empty:
                continue

            value_counts = series.value_counts()
            selected = value_counts[value_counts >= min_count].head(top_k)
            if selected.empty:
                selected = value_counts.head(top_k)

            levels[key] = [
                (float(temp), int(count)) for temp, count in selected.items()
            ]

        return levels

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

    def generate_param_space_from_data(
        self, top_k_temp_levels: int = 5
    ) -> dict[str, Any]:
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
            temperature["solution_temp"] = proc_range[
                "Primary solution temperature (℃)"
            ]
        if "Primary aging temperature (℃)" in proc_range:
            temperature["aging_temp"] = proc_range["Primary aging temperature (℃)"]

        temperature_levels = self.get_top_temperature_levels(top_k=top_k_temp_levels)

        return {
            "composition": composition,
            "temperature": temperature,
            "temperature_levels": temperature_levels,
        }
