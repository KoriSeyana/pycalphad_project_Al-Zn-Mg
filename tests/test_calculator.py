from __future__ import annotations

from pathlib import Path

import pytest

from src.calculator import CALPHADCalculator
from src.data_generator import DataGenerator
from src.feature_extractor import FeatureExtractor

DB_PATH = Path("data/COST507.pycalphad.tdb")


@pytest.fixture
def calculator():
    """创建计算器实例"""
    if not DB_PATH.exists():
        pytest.skip(f"数据库文件不存在: {DB_PATH}")
    return CALPHADCalculator(DB_PATH)


@pytest.fixture
def feature_extractor():
    """创建特征提取器实例"""
    return FeatureExtractor()


@pytest.fixture
def data_generator():
    """创建数据生成器实例"""
    return DataGenerator()


class TestCALPHADCalculator:
    """测试 CALPHAD 计算器"""

    def test_init(self, calculator):
        """测试初始化"""
        assert calculator.db is not None
        elements = calculator.get_elements()
        assert "AL" in elements
        assert "ZN" in elements
        assert "MG" in elements

    def test_get_phases(self, calculator):
        """测试获取相列表"""
        phases = calculator.get_phases()
        assert len(phases) > 0
        assert "FCC_A1" in phases

    def test_calculate_equilibrium(self, calculator):
        """测试单点平衡计算"""
        composition = {"ZN": 0.08, "MG": 0.025}
        temp_k = 470 + 273.15

        result = calculator.calculate_equilibrium(composition, temp_k)

        assert isinstance(result, dict)
        assert "FCC_A1" in result
        assert result["FCC_A1"] > 0.0

    def test_invalid_composition(self, calculator):
        """测试无效成分"""
        with pytest.raises(ValueError):
            calculator.calculate_equilibrium({"ZN": 2.0}, 700)


class TestFeatureExtractor:
    """测试特征提取器"""

    def test_extract_phase_features(self, feature_extractor):
        """测试相分数特征提取"""
        phase_fractions = {"FCC_A1": 0.95, "ETA": 0.03, "T_PHASE": 0.02}

        features = feature_extractor.extract_phase_features(phase_fractions)

        assert "ETA_fraction" in features
        assert "total_strengthening_fraction" in features
        assert features["total_strengthening_fraction"] == 0.05

    def test_extract_composition_features(self, feature_extractor):
        """测试成分特征提取"""
        composition = {"ZN": 8.0, "MG": 2.5, "CU": 1.5}

        features = feature_extractor.extract_composition_features(composition)

        assert "Zn/Mg_ratio" in features
        assert features["Zn/Mg_ratio"] > 0
        assert "total_alloy_content" in features
        assert features["total_alloy_content"] == 12.0


class TestDataGenerator:
    """测试数据生成器"""

    def test_generate_lhs_small(self, data_generator):
        """测试小规模 LHS 生成"""
        points = data_generator.generate_lhs_samples(n_samples=10, seed=20260402)

        assert len(points) == 10
        assert "ZN" in points[0]
        assert "MG" in points[0]
        assert "idx" in points[0]

    def test_format_output(self, data_generator):
        """测试输出格式化"""
        import pandas as pd

        df = pd.DataFrame(
            {
                "idx": [0],
                "ZN": [8.0],
                "MG": [2.5],
                "CU": [1.0],
                "CR": [0.1],
                "MN": [0.1],
                "ZR": [0.1],
                "TI": [0.05],
                "FE": [0.1],
                "SI": [0.1],
                "solution_temp": [460],
                "aging_temp": [150],
                "sol_FCC_A1_fraction": [0.95],
                "age_FCC_A1_fraction": [0.90],
            }
        )

        formatted = data_generator.format_output(df)

        assert "Zn" in formatted.columns
        assert "sol_FCC_A1_fraction" in formatted.columns
