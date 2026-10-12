from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats.qmc import LatinHypercube as ScipyLatinHypercube
except Exception:
    ScipyLatinHypercube = None

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """参数空间定义"""

    composition_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "ZN": (4.0, 12.0),
            "MG": (1.0, 4.0),
            "CU": (0.0, 3.0),
            "CR": (0.0, 0.35),
            "MN": (0.0, 0.5),
            "ZR": (0.0, 0.25),
            "TI": (0.0, 0.15),
            "FE": (0.0, 0.5),
            "SI": (0.0, 0.5),
        }
    )

    temperature_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "solution_temp": (440, 500),
            "aging_temp": (100, 200),
        }
    )
    temperature_levels: dict[str, list[float]] = field(default_factory=dict)
    temperature_level_weights: dict[str, list[float]] = field(default_factory=dict)


class DataGenerator:
    """虚拟数据生成器"""

    TARGET_PHASES = [
        "FCC_A1",
        "ETA",
        "T_PHASE",
        "S_PHASE",
        "AL3ZR",
        "MG2SI",
        "AL_FE_SI",
        "HCP_A3",
    ]

    OUTPUT_COLUMNS = [
        "idx",
        "Zn",
        "Mg",
        "Cu",
        "Cr",
        "Mn",
        "Zr",
        "Ti",
        "Fe",
        "Si",
        "solution_temp",
        "aging_temp",
        "sol_FCC_A1_fraction",
        "sol_ETA_fraction",
        "sol_T_PHASE_fraction",
        "sol_S_PHASE_fraction",
        "sol_AL3ZR_fraction",
        "sol_MG2SI_fraction",
        "sol_AL_FE_SI_fraction",
        "sol_HCP_A3_fraction",
        "age_FCC_A1_fraction",
        "age_ETA_fraction",
        "age_T_PHASE_fraction",
        "age_S_PHASE_fraction",
        "age_AL3ZR_fraction",
        "age_MG2SI_fraction",
        "age_AL_FE_SI_fraction",
        "age_HCP_A3_fraction",
        "sol_total_strengthening",
        "age_total_strengthening",
    ]

    def __init__(self, param_space: ParameterSpace | None = None):
        """初始化数据生成器

        Args:
            param_space: 参数空间定义，使用默认值则传 None
        """
        self.param_space = param_space or ParameterSpace()

    def generate_lhs_samples(
        self,
        n_samples: int,
        seed: int = 20260402,
    ) -> list[dict[str, float]]:
        """使用 LHS 生成参数样本

        Args:
            n_samples: 样本数量
            seed: 随机种子

        Returns:
            list: 参数点列表
        """
        bounds_order = [
            "ZN",
            "MG",
            "CU",
            "CR",
            "MN",
            "ZR",
            "TI",
            "FE",
            "SI",
            "solution_temp",
            "aging_temp",
        ]
        bounds = []
        for name in bounds_order:
            if name in self.param_space.composition_ranges:
                bounds.append(self.param_space.composition_ranges[name])
            else:
                bounds.append(self.param_space.temperature_ranges[name])

        samples_unit = self._lhs_unit(n_samples, len(bounds_order), seed)

        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)
        samples_scaled = lb + samples_unit * (ub - lb)
        rng = np.random.default_rng(seed + 1)

        points: list[dict[str, float]] = []
        for idx, row in enumerate(samples_scaled):
            point = {
                name: float(row[col_idx]) for col_idx, name in enumerate(bounds_order)
            }

            for temp_key in ("solution_temp", "aging_temp"):
                levels = self.param_space.temperature_levels.get(temp_key, [])
                if not levels:
                    continue

                weights = self.param_space.temperature_level_weights.get(temp_key, [])
                if weights and len(weights) == len(levels):
                    point[temp_key] = float(rng.choice(levels, p=weights))
                else:
                    point[temp_key] = float(rng.choice(levels))

            point["idx"] = idx
            points.append(point)

        logger.info(f"LHS 采样完成: {len(points)} 个点, seed={seed}")
        return points

    @staticmethod
    def _lhs_unit(n_samples: int, n_dims: int, seed: int) -> np.ndarray:
        """生成 [0,1] 区间的 LHS 样本，scipy 不可用时回退到 numpy 实现。"""
        if ScipyLatinHypercube is not None:
            sampler = ScipyLatinHypercube(d=n_dims, seed=seed)
            return sampler.random(n=n_samples)

        rng = np.random.default_rng(seed)
        samples = np.zeros((n_samples, n_dims), dtype=float)
        for dim in range(n_dims):
            bins = (
                np.arange(n_samples, dtype=float) + rng.random(n_samples)
            ) / n_samples
            rng.shuffle(bins)
            samples[:, dim] = bins
        return samples

    def format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化输出数据

        Args:
            df: 原始计算结果

        Returns:
            pd.DataFrame: 格式化后的数据
        """
        # 映射内部字段名到文档标准列名
        rename_map = {
            "ZN": "Zn",
            "MG": "Mg",
            "CU": "Cu",
            "CR": "Cr",
            "MN": "Mn",
            "ZR": "Zr",
            "TI": "Ti",
            "FE": "Fe",
            "SI": "Si",
        }

        df_out = df.rename(columns=rename_map)

        for col in self.OUTPUT_COLUMNS:
            if col not in df_out.columns:
                df_out[col] = 0.0

        return df_out[self.OUTPUT_COLUMNS]

    def validate_dataset(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """执行基础质量校验。"""
        errors: list[str] = []

        nan_count = int(df.isna().sum().sum())
        if nan_count > 0:
            errors.append(f"存在 {nan_count} 个 NaN")

        phase_cols = [c for c in df.columns if c.endswith("_fraction")]
        for col in phase_cols:
            if float(df[col].min()) < 0:
                errors.append(f"{col} 存在负值")

        sol_cols = [f"sol_{p}_fraction" for p in self.TARGET_PHASES]
        age_cols = [f"age_{p}_fraction" for p in self.TARGET_PHASES]
        if set(sol_cols).issubset(df.columns):
            sol_sum = df[sol_cols].sum(axis=1)
            if ((sol_sum < 0.99) | (sol_sum > 1.01)).any():
                errors.append("固溶态相分数之和偏离 1.0")
        if set(age_cols).issubset(df.columns):
            age_sum = df[age_cols].sum(axis=1)
            if ((age_sum < 0.99) | (age_sum > 1.01)).any():
                errors.append("时效态相分数之和偏离 1.0")

        if (
            "sol_ETA_fraction" in df.columns
            and float(df["sol_ETA_fraction"].max()) > 0.05
        ):
            errors.append("固溶态 ETA 分数过高")

        if {
            "sol_ETA_fraction",
            "age_ETA_fraction",
        }.issubset(df.columns) and (
            df["age_ETA_fraction"] < df["sol_ETA_fraction"]
        ).any():
            errors.append("存在时效态 ETA 小于固溶态 ETA 的样本")

        return len(errors) == 0, errors

    def build_validation_report(self, df: pd.DataFrame) -> str:
        """生成校验报告文本。"""
        is_valid, errors = self.validate_dataset(df)
        lines = [
            "虚拟数据质量校验报告",
            "=" * 40,
            f"样本数: {len(df)}",
            f"校验结果: {'通过' if is_valid else '未通过'}",
            "",
        ]
        if errors:
            lines.append("问题列表:")
            lines.extend([f"- {msg}" for msg in errors])
        else:
            lines.append("未发现问题。")
        return "\n".join(lines)

    def save(self, df: pd.DataFrame, output_path: str | Path) -> None:
        """保存数据到 CSV

        Args:
            df: 数据框
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"数据已保存: {output_path}, {len(df)} 条记录")
