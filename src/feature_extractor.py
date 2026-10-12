from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class FeatureExtractor:
    """从 CALPHAD 计算结果中提取物理特征"""

    # 输出特征列 -> 可由 CALPHAD 直接算出的相名（允许多相合并到一个工程特征）
    PRECIPITATION_PHASE_MAP: dict[str, list[str]] = {
        # 7xxx 强化相（η 系）
        "ETA_fraction": ["ETA", "ALCU_ETA", "MGZN2", "MG2ZN11", "MGZN", "MG2ZN3"],
        # 7xxx 中常见 T 相家族（数据库命名存在差异）
        "T_PHASE_fraction": ["T_PHASE", "TAU", "QPHASE", "VPHASE", "ALMG_GAMMA"],
        # Al-Cu-Mg 相关 S 相
        "S_PHASE_fraction": ["S_PHASE", "SPHASE"],
        # 弥散强化相
        "AL3ZR_fraction": ["AL3ZR"],
        # 常见杂质/脆性相关析出相
        "MG2SI_fraction": ["MG2SI"],
        "AL_FE_SI_fraction": [
            "AL13FE4",
            "AL2FE",
            "AL5FE2",
            "ALFESI_ALPHA",
            "ALFESI_BETA",
            "ALFESI_GAMMA",
            "ALFESI_DELTA",
        ],
        "SI2TI_fraction": ["SI2TI", "SITI", "SI3TI5", "SITI3"],
        "SIZR_fraction": ["SIZR", "SI2ZR", "SI2ZR3", "SI3ZR5", "SI4ZR5"],
        "AL_MN_intermetallic_fraction": [
            "AL6MN",
            "AL11MN4",
            "AL12MN",
            "AL4MN",
            "AL8MN5_D810",
            "ALMNSI_ALPHA",
            "ALMNSI_BETA",
            "ALMNSI_DELTA",
        ],
    }

    MATRIX_PHASES = {
        "FCC_A1": "Al 基固溶体",
        "HCP_A3": "HCP 结构相",
    }

    ALLOY_ELEMENTS = ["ZN", "MG", "CU", "CR", "MN", "ZR", "TI"]
    IMPURITY_ELEMENTS = ["FE", "SI"]

    def extract_phase_features(
        self, phase_fractions: dict[str, float]
    ) -> dict[str, float]:
        """提取相分数特征

        Args:
            phase_fractions: 各相的摩尔分数字典

        Returns:
            dict: 相分数特征字典
        """
        features = {}

        for output_col, phase_aliases in self.PRECIPITATION_PHASE_MAP.items():
            features[output_col] = sum(phase_fractions.get(p, 0.0) for p in phase_aliases)

        for phase in self.MATRIX_PHASES:
            features[f"{phase}_fraction"] = phase_fractions.get(phase, 0.0)

        features["total_strengthening_fraction"] = (
            features.get("ETA_fraction", 0.0)
            + features.get("T_PHASE_fraction", 0.0)
            + features.get("S_PHASE_fraction", 0.0)
            + features.get("AL3ZR_fraction", 0.0)
        )

        features["total_detrimental_fraction"] = (
            features.get("MG2SI_fraction", 0.0)
            + features.get("AL_FE_SI_fraction", 0.0)
            + features.get("SI2TI_fraction", 0.0)
            + features.get("SIZR_fraction", 0.0)
            + features.get("AL_MN_intermetallic_fraction", 0.0)
        )

        return features

    def extract_composition_features(
        self, composition: dict[str, float]
    ) -> dict[str, float]:
        """提取成分衍生特征

        Args:
            composition: 元素质量分数字典

        Returns:
            dict: 成分特征字典
        """
        features = {}

        zn = composition.get("ZN", 0)
        mg = composition.get("MG", 0)
        features["Zn/Mg_ratio"] = zn / (mg + 1e-6)

        features["total_alloy_content"] = sum(
            composition.get(e, 0) for e in self.ALLOY_ELEMENTS
        )

        features["total_impurity"] = sum(
            composition.get(e, 0) for e in self.IMPURITY_ELEMENTS
        )

        return features

    def extract_all_features(
        self,
        composition: dict[str, float],
        phase_fractions: dict[str, float],
        process_params: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """提取所有特征

        Args:
            composition: 元素质量分数
            phase_fractions: 各相摩尔分数
            process_params: 工艺参数（固溶温度、时效温度等）

        Returns:
            dict: 完整特征字典
        """
        features = {}

        features.update(self.extract_composition_features(composition))
        features.update(self.extract_phase_features(phase_fractions))

        if process_params:
            features.update(process_params)

        return features

    @classmethod
    def calculate_derived_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """从数据框中计算衍生特征

        Args:
            df: 包含原始特征的数据框

        Returns:
            pd.DataFrame: 添加衍生特征后的数据框
        """
        df = df.copy()

        # 处理 data.xlsx 格式的列名 (Zn/％, Mg/％等)
        zn_col = "Zn/％" if "Zn/％" in df.columns else "ZN"
        mg_col = "Mg/％" if "Mg/％" in df.columns else "MG"

        if zn_col in df.columns and mg_col in df.columns:
            df["Zn/Mg_ratio"] = df[zn_col] / (df[mg_col] + 1e-6)

        # 合金元素列名（兼容两种格式）
        alloy_cols = []
        for elem in ["ZN", "MG", "CU", "CR", "MN", "ZR"]:
            col_std = elem
            col_cn = f"{elem.title()[:2]}/％" if elem != "ZR" else f"{elem.title()}/％"
            if col_std in df.columns:
                alloy_cols.append(col_std)
            elif col_cn in df.columns:
                alloy_cols.append(col_cn)

        if alloy_cols:
            df["total_alloy_content"] = df[alloy_cols].sum(axis=1)

        # 杂质元素
        impurity_cols = []
        for elem in ["FE", "SI"]:
            col_std = elem
            col_cn = f"{elem.title()}/％"
            if col_std in df.columns:
                impurity_cols.append(col_std)
            elif col_cn in df.columns:
                impurity_cols.append(col_cn)

        if impurity_cols:
            df["total_impurity"] = df[impurity_cols].sum(axis=1)

        # 从 CALPHAD 原始相列汇总到工程特征列
        for output_col, phase_aliases in cls.PRECIPITATION_PHASE_MAP.items():
            existing_cols = [c for c in phase_aliases if c in df.columns]
            if existing_cols:
                df[output_col] = df[existing_cols].fillna(0.0).sum(axis=1)
            elif output_col not in df.columns:
                df[output_col] = 0.0

        # 基体相
        if "FCC_A1" in df.columns:
            df["FCC_A1_fraction"] = df["FCC_A1"].fillna(0.0)
        elif "FCC_A1_fraction" not in df.columns:
            df["FCC_A1_fraction"] = 0.0

        # 汇总特征
        df["total_strengthening_fraction"] = (
            df.get("ETA_fraction", 0.0)
            + df.get("T_PHASE_fraction", 0.0)
            + df.get("S_PHASE_fraction", 0.0)
            + df.get("AL3ZR_fraction", 0.0)
        )

        df["total_detrimental_fraction"] = (
            df.get("MG2SI_fraction", 0.0)
            + df.get("AL_FE_SI_fraction", 0.0)
            + df.get("SI2TI_fraction", 0.0)
            + df.get("SIZR_fraction", 0.0)
            + df.get("AL_MN_intermetallic_fraction", 0.0)
        )

        return df
