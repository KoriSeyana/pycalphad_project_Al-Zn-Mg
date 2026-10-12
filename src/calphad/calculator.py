from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory


_PHASE_RECORD_FACTORY_PATCHED = False
ATOMIC_WEIGHTS = {
    "AL": 26.9815385,
    "ZN": 65.38,
    "MG": 24.305,
    "CU": 63.546,
    "CR": 51.9961,
    "MN": 54.938044,
    "ZR": 91.224,
    "TI": 47.867,
    "FE": 55.845,
    "SI": 28.085,
}


def _patch_phase_record_factory() -> None:
    """兼容 pycalphad 在部分环境下传入 numpy.str_ 相名的问题。"""
    global _PHASE_RECORD_FACTORY_PATCHED
    if _PHASE_RECORD_FACTORY_PATCHED:
        return

    original_get = PhaseRecordFactory.get

    def patched_get(self, phase_name):
        return original_get(self, str(phase_name))

    PhaseRecordFactory.get = patched_get
    _PHASE_RECORD_FACTORY_PATCHED = True


class CALPHADCalculator:
    """CALPHAD 热力学计算器核心类"""

    def __init__(self, db_path: str | Path):
        """初始化计算器

        Args:
            db_path: TDB 数据库文件路径
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")
        _patch_phase_record_factory()
        self.db = Database(self.db_path)
        self._validate_database()

    def _validate_database(self) -> None:
        """验证数据库是否包含 7000 系铝合金所需元素"""
        required_elements = {"AL", "ZN", "MG"}
        available = set(self.db.elements) - {"/-"}
        missing = required_elements - available
        if missing:
            raise ValueError(f"数据库缺少必要元素: {missing}")

    def get_elements(self) -> list[str]:
        """获取数据库包含的元素列表"""
        return [e for e in self.db.elements if e != "/-"]

    def get_phases(self) -> list[str]:
        """获取数据库包含的相列表"""
        return [str(p) for p in self.db.phases.keys()]

    @staticmethod
    def _mass_to_mole_fraction(composition: dict[str, float]) -> dict[str, float]:
        """将输入的质量分数转换为摩尔分数（以 AL 为余量元素）。"""
        cleaned = {
            str(elem).upper(): float(value)
            for elem, value in composition.items()
            if float(value) > 0.0
        }
        if not cleaned:
            return {}

        unknown_elements = [elem for elem in cleaned if elem not in ATOMIC_WEIGHTS]
        if unknown_elements:
            raise ValueError(f"缺少原子量定义: {unknown_elements}")

        solute_mass = float(sum(cleaned.values()))
        if solute_mass >= 1.0:
            raise ValueError(f"溶质质量分数之和必须小于1，当前为 {solute_mass:.6f}")

        al_mass = 1.0 - solute_mass
        total_moles = al_mass / ATOMIC_WEIGHTS["AL"]
        mole_values: dict[str, float] = {}
        for elem, mass_frac in cleaned.items():
            mole_val = mass_frac / ATOMIC_WEIGHTS[elem]
            mole_values[elem] = mole_val
            total_moles += mole_val

        if total_moles <= 0.0:
            raise ValueError("总物质的量为0，无法转换摩尔分数")

        return {elem: val / total_moles for elem, val in mole_values.items()}

    def calculate_equilibrium(
        self,
        composition: dict[str, float],
        temperature_k: float,
        pressure: float = 101325,
        phases: list[str] | None = None,
    ) -> dict[str, Any]:
        """计算给定成分和温度下的平衡状态

        Args:
            composition: 元素质量分数，如 {"ZN": 0.08, "MG": 0.025}
            temperature_k: 温度（开尔文）
            pressure: 压力（帕斯卡），默认 101325 Pa
            phases: 要计算的相列表，默认为所有相

        Returns:
            dict: 包含各相摩尔分数和化学成分的字典
        """
        mole_composition = self._mass_to_mole_fraction(composition)
        elements = ["AL", "VA"] + list(mole_composition.keys())
        conditions: dict[Any, float] = {v.T: temperature_k, v.P: pressure, v.N: 1}
        for elem, comp in mole_composition.items():
            conditions[v.X(elem)] = comp

        if phases is None:
            phases = [str(p) for p in self.db.phases.keys()]
        else:
            phases = [str(p) for p in phases]

        try:
            result = equilibrium(self.db, elements, phases, conditions)
        except Exception as e:
            raise RuntimeError(f"计算失败: {e}") from e

        phase_fractions = self._extract_phase_fractions(result, phases)
        return phase_fractions

    def _extract_phase_fractions(
        self, result: Any, phases: list[str]
    ) -> dict[str, float]:
        """从计算结果中提取相分数"""
        fractions = {}
        for phase_name in phases:
            try:
                # equilibrium 结果中按 Phase 标签筛选并累加该相的 NP
                phase_mask = result.Phase == phase_name
                fraction = float(result.NP.where(phase_mask).sum().values)
                if fraction > 1e-6:
                    fractions[phase_name] = fraction
            except (KeyError, IndexError, ValueError, TypeError):
                continue
        return fractions

    def calculate_equilibrium_multi_temp(
        self,
        composition: dict[str, float],
        temperatures: list[float] | np.ndarray,
        pressure: float = 101325,
    ) -> list[dict[str, float]]:
        """计算多个温度下的平衡状态

        Args:
            composition: 元素质量分数
            temperatures: 温度列表（开尔文）
            pressure: 压力（帕斯卡）

        Returns:
            list: 每个温度对应的相分数字典
        """
        results = []
        for temp in temperatures:
            try:
                phase_fractions = self.calculate_equilibrium(
                    composition, temp, pressure
                )
                phase_fractions["temperature_k"] = temp
                results.append(phase_fractions)
            except RuntimeError:
                results.append({"temperature_k": temp, "error": True})
        return results
