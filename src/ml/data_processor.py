"""
数据处理模块
支持数据分离、指示变量、物理化学参数、零值特殊归一化等功能
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, List, Optional, Dict, Any

# 合金元素的物理化学参数
ELEMENT_PARAMS = {
    'Zn': {
        'atomic_radius': 1.34,      # 埃
        'electronegativity': 1.65,  # Pauling
        'melting_point': 419.5,     # 摄氏度
        'density': 7.13,            # g/cm³
        'valence_electrons': 2,
        'atomic_number': 30,
    },
    'Mg': {
        'atomic_radius': 1.60,
        'electronegativity': 1.31,
        'melting_point': 650,
        'density': 1.74,
        'valence_electrons': 2,
        'atomic_number': 12,
    },
    'Cu': {
        'atomic_radius': 1.28,
        'electronegativity': 1.90,
        'melting_point': 1085,
        'density': 8.96,
        'valence_electrons': 1,
        'atomic_number': 29,
    },
    'Cr': {
        'atomic_radius': 1.28,
        'electronegativity': 1.66,
        'melting_point': 1907,
        'density': 7.19,
        'valence_electrons': 6,
        'atomic_number': 24,
    },
    'Mn': {
        'atomic_radius': 1.27,
        'electronegativity': 1.55,
        'melting_point': 1246,
        'density': 7.44,
        'valence_electrons': 7,
        'atomic_number': 25,
    },
    'Zr': {
        'atomic_radius': 1.60,
        'electronegativity': 1.33,
        'melting_point': 1855,
        'density': 6.51,
        'valence_electrons': 4,
        'atomic_number': 40,
    },
    'Ti': {
        'atomic_radius': 1.47,
        'electronegativity': 1.54,
        'melting_point': 1668,
        'density': 4.51,
        'valence_electrons': 4,
        'atomic_number': 22,
    },
    'Fe': {
        'atomic_radius': 1.26,
        'electronegativity': 1.83,
        'melting_point': 1538,
        'density': 7.87,
        'valence_electrons': 8,
        'atomic_number': 26,
    },
    'Si': {
        'atomic_radius': 1.17,
        'electronegativity': 1.90,
        'melting_point': 1414,
        'density': 2.33,
        'valence_electrons': 4,
        'atomic_number': 14,
    },
}


class ZeroAwareScaler:
    """
    零值感知的归一化器
    对非零值进行标准化，零值保持为特殊值
    """

    def __init__(self, zero_value: float = -1.0, method: str = 'standard'):
        """
        Args:
            zero_value: 零值映射的目标值
            method: 归一化方法 ('standard', 'minmax', 'robust')
        """
        self.zero_value = zero_value
        self.method = method
        self.scalers = {}
        self.zero_masks = {}

    def fit(self, X: np.ndarray) -> 'ZeroAwareScaler':
        """
        拟合归一化器

        Args:
            X: 输入数据 (n_samples, n_features)

        Returns:
            self
        """
        X = np.array(X)
        n_features = X.shape[1]

        for i in range(n_features):
            col = X[:, i]
            zero_mask = (col == 0)
            self.zero_masks[i] = zero_mask

            # 只对非零值拟合
            non_zero_values = col[~zero_mask].reshape(-1, 1)

            if len(non_zero_values) > 0:
                if self.method == 'standard':
                    scaler = StandardScaler()
                elif self.method == 'minmax':
                    scaler = MinMaxScaler()
                elif self.method == 'robust':
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"未知的归一化方法: {self.method}")

                scaler.fit(non_zero_values)
                self.scalers[i] = scaler

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换数据

        Args:
            X: 输入数据

        Returns:
            np.ndarray: 转换后的数据
        """
        X = np.array(X).copy()
        result = np.zeros_like(X, dtype=float)

        for i in range(X.shape[1]):
            col = X[:, i]
            zero_mask = (col == 0)

            # 零值设为特殊值
            result[zero_mask, i] = self.zero_value

            # 非零值进行归一化
            if i in self.scalers:
                non_zero_values = col[~zero_mask].reshape(-1, 1)
                if len(non_zero_values) > 0:
                    result[~zero_mask, i] = self.scalers[i].transform(non_zero_values).flatten()

        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        逆转换

        Args:
            X: 归一化后的数据

        Returns:
            np.ndarray: 原始尺度的数据
        """
        X = np.array(X).copy()
        result = np.zeros_like(X, dtype=float)

        for i in range(X.shape[1]):
            col = X[:, i]

            # 特殊值映射回零
            zero_mask = (col == self.zero_value)
            result[zero_mask, i] = 0

            # 非零值逆转换
            if i in self.scalers:
                non_zero_values = col[~zero_mask].reshape(-1, 1)
                result[~zero_mask, i] = self.scalers[i].inverse_transform(non_zero_values).flatten()

        return result


class DataProcessor:
    """
    数据处理器

    支持以下功能：
    1. 数据分离策略（按热处理阶段）
    2. 添加指示变量
    3. 添加物理化学参数
    4. 零值特殊归一化
    """

    def __init__(self, config: dict):
        """
        初始化数据处理器

        Args:
            config: 数据配置字典
        """
        self.config = config
        self.data = None
        self.feature_cols = None
        self.target_cols = config.get('target_cols', [])

        # 归一化器
        normalize_method = config.get('normalize_method', 'zero_aware')
        if normalize_method == 'zero_aware':
            self.scaler_X = ZeroAwareScaler(zero_value=-1.0, method='standard')
            self.scaler_y = ZeroAwareScaler(zero_value=-1.0, method='standard')
        elif normalize_method == 'robust':
            self.scaler_X = RobustScaler()
            self.scaler_y = RobustScaler()
        elif normalize_method == 'minmax':
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        else:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()

        self._is_fitted = False

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载数据

        Args:
            file_path: 数据文件路径

        Returns:
            pd.DataFrame: 加载的数据
        """
        if file_path is None:
            file_path = self.config.get('data_path', 'data/data.xlsx')

        # 根据文件后缀自动选择读取方式
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.data = pd.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

        # 应用特征工程
        if self.config.get('add_indicator_vars', False):
            self._add_indicator_variables()

        if self.config.get('add_element_params', False):
            self._add_element_parameters()

        if self.config.get('add_interaction_features', False):
            self._add_interaction_features()

        # 获取排除列列表
        exclude_cols = self.config.get('exclude_cols', [])

        # 自动填充 NaN 为 0（可选热处理阶段的默认值）
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        nan_count = self.data[numeric_cols].isnull().sum().sum()
        if nan_count > 0:
            self.data[numeric_cols] = self.data[numeric_cols].fillna(0)
            print(f"  NaN 填充: {nan_count} 个缺失值已填充为 0")

        # 自动识别特征列（排除目标列和排除列，以及非数值列）
        self.feature_cols = []
        for col in self.data.columns:
            # 跳过目标列
            if col in self.target_cols:
                continue
            # 跳过手动排除的列
            if col in exclude_cols:
                print(f"  排除列: {col} (手动排除)")
                continue
            # 跳过非数值列
            if not np.issubdtype(self.data[col].dtype, np.number):
                print(f"  排除列: {col} (非数值类型: {self.data[col].dtype})")
                continue
            self.feature_cols.append(col)

        print(f"数据加载成功: {self.data.shape[0]} 样本, {self.data.shape[1]} 列")
        print(f"特征列: {len(self.feature_cols)} 个")
        print(f"目标列: {self.target_cols}")

        return self.data

    def _add_indicator_variables(self):
        """添加指示变量（标记是否进行某热处理步骤）"""
        print("添加指示变量...")

        # 热处理温度列（零值代表未进行该步骤）
        temp_cols = [
            'Secondary solution temperature',
            'Tertiary solution temperature',
            'Secondary aging temperature',
            'Tertiary aging temperature',
        ]

        for col_prefix in temp_cols:
            # 查找匹配的列
            matching_cols = [c for c in self.data.columns if col_prefix in c]
            for col in matching_cols:
                indicator_name = f'has_{col_prefix.replace(" ", "_").lower()}'
                if indicator_name not in self.data.columns:
                    self.data[indicator_name] = (self.data[col] > 0).astype(int)
                    print(f"  添加: {indicator_name}")

    def _add_element_parameters(self):
        """添加元素的物理化学参数（完整高熵合金/多主元合金特征）"""
        print("添加元素物理化学参数...")

        # 元素列名映射
        element_col_map = {
            'Zn/': 'Zn',
            'Mg/': 'Mg',
            'Cu/': 'Cu',
            'Cr/': 'Cr',
            'Mn/': 'Mn',
            'Zr/': 'Zr',
            'Ti/': 'Ti',
            'Fe/': 'Fe',
            'Si/': 'Si',
        }

        # 获取每个样本的元素摩尔分数
        n_samples = len(self.data)
        element_names = list(element_col_map.values())
        mole_fractions = {}

        # 原子量 (g/mol)
        atomic_weights = {
            'Zn': 65.38, 'Mg': 24.305, 'Cu': 63.546, 'Cr': 51.996,
            'Mn': 54.938, 'Zr': 91.224, 'Ti': 47.867, 'Fe': 55.845, 'Si': 28.085
        }

        # 计算摩尔分数
        total_moles = np.zeros(n_samples)
        for elem in element_names:
            col_prefix = [k for k, v in element_col_map.items() if v == elem][0]
            matching_cols = [c for c in self.data.columns if c.startswith(col_prefix)]
            if matching_cols:
                col = matching_cols[0]
                wt_pct = self.data[col].values
                moles = wt_pct / atomic_weights[elem]
                mole_fractions[elem] = moles
                total_moles += moles

        # 归一化为摩尔分数
        for elem in element_names:
            if elem in mole_fractions:
                mole_fractions[elem] = np.where(total_moles > 0, mole_fractions[elem] / total_moles, 0)

        # 1. 加权平均参数（基于质量分数）
        param_names = ['atomic_radius', 'electronegativity', 'melting_point', 'density', 'valence_electrons']

        for param in param_names:
            weighted_sum = np.zeros(n_samples)
            total_weight = np.zeros(n_samples)

            for col_prefix, element in element_col_map.items():
                matching_cols = [c for c in self.data.columns if c.startswith(col_prefix)]
                if matching_cols and element in ELEMENT_PARAMS:
                    col = matching_cols[0]
                    weight = self.data[col].values
                    value = ELEMENT_PARAMS[element][param]
                    weighted_sum += weight * value
                    total_weight += weight

            with np.errstate(divide='ignore', invalid='ignore'):
                weighted_avg = np.where(total_weight > 0, weighted_sum / total_weight, 0)

            self.data[f'weighted_avg_{param}'] = weighted_avg
            print(f"  添加: weighted_avg_{param}")

        # 2. VEC (Valence Electron Concentration) - 价电子浓度
        vec = np.zeros(n_samples)
        for elem in element_names:
            if elem in mole_fractions and elem in ELEMENT_PARAMS:
                vec += mole_fractions[elem] * ELEMENT_PARAMS[elem]['valence_electrons']
        self.data['VEC'] = vec
        print(f"  添加: VEC")

        # 3. δ (Atomic size difference parameter)
        # δ = sqrt(sum(ci * (1 - ri/r_avg)^2)) * 100%
        r_avg = np.zeros(n_samples)
        for elem in element_names:
            if elem in mole_fractions and elem in ELEMENT_PARAMS:
                r_avg += mole_fractions[elem] * ELEMENT_PARAMS[elem]['atomic_radius']

        delta = np.zeros(n_samples)
        for elem in element_names:
            if elem in mole_fractions and elem in ELEMENT_PARAMS:
                ri = ELEMENT_PARAMS[elem]['atomic_radius']
                delta += mole_fractions[elem] * (1 - ri / np.where(r_avg > 0, r_avg, 1)) ** 2
        delta = np.sqrt(delta) * 100
        self.data['atomic_size_diff_delta'] = delta
        print(f"  添加: atomic_size_diff_delta")

        # 4. Δχ (Electronegativity difference)
        # Δχ = sqrt(sum(ci * (χi - χ_avg)^2))
        chi_avg = np.zeros(n_samples)
        for elem in element_names:
            if elem in mole_fractions and elem in ELEMENT_PARAMS:
                chi_avg += mole_fractions[elem] * ELEMENT_PARAMS[elem]['electronegativity']

        delta_chi = np.zeros(n_samples)
        for elem in element_names:
            if elem in mole_fractions and elem in ELEMENT_PARAMS:
                chi_i = ELEMENT_PARAMS[elem]['electronegativity']
                delta_chi += mole_fractions[elem] * (chi_i - chi_avg) ** 2
        delta_chi = np.sqrt(delta_chi)
        self.data['electronegativity_diff_delta_chi'] = delta_chi
        print(f"  添加: electronegativity_diff_delta_chi")

        # 5. ΔH_mix (Mixing enthalpy) - 简化版，使用二元相互作用近似
        # ΔH_mix = 4 * sum(ci * cj * ΔH_AB)
        # 这里使用经验值表 (kJ/mol)
        mixing_enthalpy = {
            ('Al', 'Zn'): 0, ('Al', 'Mg'): 0, ('Al', 'Cu'): -1, ('Al', 'Cr'): -10,
            ('Al', 'Mn'): -2, ('Al', 'Zr'): -41, ('Al', 'Ti'): -30, ('Al', 'Fe'): -11,
            ('Al', 'Si'): -11, ('Zn', 'Mg'): 0, ('Zn', 'Cu'): 0, ('Zn', 'Cr'): 0,
            ('Zn', 'Mn'): 0, ('Zn', 'Zr'): 0, ('Zn', 'Ti'): 0, ('Zn', 'Fe'): 0,
            ('Zn', 'Si'): 0, ('Mg', 'Cu'): -4, ('Mg', 'Cr'): 0, ('Mg', 'Mn'): 0,
            ('Mg', 'Zr'): 0, ('Mg', 'Ti'): 0, ('Mg', 'Fe'): 0, ('Mg', 'Si'): -8,
            ('Cu', 'Cr'): 5, ('Cu', 'Mn'): 0, ('Cu', 'Zr'): -23, ('Cu', 'Ti'): -9,
            ('Cu', 'Fe'): -1, ('Cu', 'Si'): -4, ('Cr', 'Mn'): 0, ('Cr', 'Zr'): -15,
            ('Cr', 'Ti'): -5, ('Cr', 'Fe'): -1, ('Cr', 'Si'): -28, ('Mn', 'Zr'): -17,
            ('Mn', 'Ti'): -5, ('Mn', 'Fe'): -2, ('Mn', 'Si'): -18, ('Zr', 'Ti'): -2,
            ('Zr', 'Fe'): -25, ('Zr', 'Si'): -46, ('Ti', 'Fe'): -17, ('Ti', 'Si'): -38,
            ('Fe', 'Si'): -20,
        }

        # Al 是基体，计算 Al 与其他元素的摩尔分数
        al_mole = 1.0 - sum(mole_fractions.get(e, np.zeros(n_samples)) for e in element_names)

        delta_h_mix = np.zeros(n_samples)
        # Al 与其他元素
        for elem in element_names:
            if elem in mole_fractions:
                pair = tuple(sorted(['Al', elem]))
                h_ab = mixing_enthalpy.get(pair, 0)
                delta_h_mix += 4 * al_mole * mole_fractions[elem] * h_ab

        # 其他元素之间
        for i, e1 in enumerate(element_names):
            for e2 in element_names[i+1:]:
                if e1 in mole_fractions and e2 in mole_fractions:
                    pair = tuple(sorted([e1, e2]))
                    h_ab = mixing_enthalpy.get(pair, 0)
                    delta_h_mix += 4 * mole_fractions[e1] * mole_fractions[e2] * h_ab

        self.data['mixing_enthalpy_delta_H_mix'] = delta_h_mix
        print(f"  添加: mixing_enthalpy_delta_H_mix")

        # 6. ΔS_mix (Mixing entropy)
        # ΔS_mix = -R * sum(ci * ln(ci))
        R = 8.314  # J/(mol·K)
        delta_s_mix = np.zeros(n_samples)
        # Al
        al_mask = al_mole > 1e-10
        delta_s_mix[al_mask] -= R * al_mole[al_mask] * np.log(al_mole[al_mask])
        for elem in element_names:
            if elem in mole_fractions:
                mask = mole_fractions[elem] > 1e-10
                delta_s_mix[mask] -= R * mole_fractions[elem][mask] * np.log(mole_fractions[elem][mask])
        self.data['mixing_entropy_delta_S_mix'] = delta_s_mix
        print(f"  添加: mixing_entropy_delta_S_mix")

        # 7. Ω parameter
        # Ω = Tm * ΔS_mix / |ΔH_mix|
        # Tm = sum(ci * Tm,i)
        Tm = np.zeros(n_samples)
        # Al 熔点
        Tm += al_mole * 933.47  # Al melting point in K
        for elem in element_names:
            if elem in mole_fractions and elem in ELEMENT_PARAMS:
                Tm += mole_fractions[elem] * (ELEMENT_PARAMS[elem]['melting_point'] + 273.15)

        omega = np.where(np.abs(delta_h_mix) > 1e-10, Tm * delta_s_mix / np.abs(delta_h_mix), 0)
        self.data['omega_parameter'] = omega
        print(f"  添加: omega_parameter")

        # 8. 原子半径极差 (max - min)
        radii = np.array([ELEMENT_PARAMS[e]['atomic_radius'] for e in element_names])
        self.data['atomic_radius_range'] = np.max(radii) - np.min(radii)
        print(f"  添加: atomic_radius_range")

        # 9. 电负性极差
        chi = np.array([ELEMENT_PARAMS[e]['electronegativity'] for e in element_names])
        self.data['electronegativity_range'] = np.max(chi) - np.min(chi)
        print(f"  添加: electronegativity_range")

        # 10. 价电子浓度 (总)
        self.data['total_valence_electrons'] = vec * np.sum([mole_fractions.get(e, np.zeros(n_samples)) for e in element_names], axis=0)
        print(f"  添加: total_valence_electrons")

        # 11. 原子尺寸差异的另一种表示 (最大原子半径差)
        max_radius_diff = np.zeros(n_samples)
        for i, e1 in enumerate(element_names):
            for e2 in element_names[i+1:]:
                if e1 in mole_fractions and e2 in mole_fractions:
                    if ELEMENT_PARAMS[e1]['atomic_radius'] != ELEMENT_PARAMS[e2]['atomic_radius']:
                        diff = abs(ELEMENT_PARAMS[e1]['atomic_radius'] - ELEMENT_PARAMS[e2]['atomic_radius'])
                        max_radius_diff = np.maximum(max_radius_diff, diff * np.where((mole_fractions[e1] > 0) & (mole_fractions[e2] > 0), 1, 0))
        self.data['max_atomic_radius_diff'] = max_radius_diff
        print(f"  添加: max_atomic_radius_diff")

        # 12. 电负性差异的另一种表示
        max_chi_diff = np.zeros(n_samples)
        for i, e1 in enumerate(element_names):
            for e2 in element_names[i+1:]:
                if e1 in mole_fractions and e2 in mole_fractions:
                    if ELEMENT_PARAMS[e1]['electronegativity'] != ELEMENT_PARAMS[e2]['electronegativity']:
                        diff = abs(ELEMENT_PARAMS[e1]['electronegativity'] - ELEMENT_PARAMS[e2]['electronegativity'])
                        max_chi_diff = np.maximum(max_chi_diff, diff * np.where((mole_fractions[e1] > 0) & (mole_fractions[e2] > 0), 1, 0))
        self.data['max_electronegativity_diff'] = max_chi_diff
        print(f"  添加: max_electronegativity_diff")

    def _add_interaction_features(self):
        """添加交互特征"""
        print("添加交互特征...")

        # 查找元素列
        element_cols = {}
        for col in self.data.columns:
            for elem in ['Zn', 'Mg', 'Cu', 'Cr', 'Mn', 'Zr', 'Ti', 'Fe', 'Si']:
                if col.startswith(f'{elem}/'):
                    element_cols[elem] = col
                    break

        # 添加比值特征
        if 'Zn' in element_cols and 'Mg' in element_cols:
            self.data['Zn_Mg_ratio'] = self.data[element_cols['Zn']] / (self.data[element_cols['Mg']] + 1e-6)
            print("  添加: Zn_Mg_ratio")

        if 'Cu' in element_cols and 'Zn' in element_cols and 'Mg' in element_cols:
            self.data['Cu_total_ratio'] = self.data[element_cols['Cu']] / (
                self.data[element_cols['Zn']] + self.data[element_cols['Mg']] + 1e-6
            )
            print("  添加: Cu_total_ratio")

        # 添加总合金含量
        alloy_sum = np.zeros(len(self.data))
        for elem, col in element_cols.items():
            alloy_sum += self.data[col].values
        self.data['total_alloy_content'] = alloy_sum
        print("  添加: total_alloy_content")

        # 添加热处理强度指标
        solution_temp_cols = [c for c in self.data.columns if 'solution temperature' in c.lower()]
        solution_time_cols = [c for c in self.data.columns if 'solution time' in c.lower()]

        for temp_col, time_col in zip(solution_temp_cols, solution_time_cols):
            if temp_col in self.data.columns and time_col in self.data.columns:
                intensity_name = f'{temp_col.split()[0].lower()}_solution_intensity'
                self.data[intensity_name] = self.data[temp_col] * self.data[time_col] / 1000
                print(f"  添加: {intensity_name}")

    def split_by_process_stage(self) -> Dict[str, pd.DataFrame]:
        """
        按热处理阶段分离数据

        Returns:
            Dict[str, pd.DataFrame]: 分离后的数据集
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")

        datasets = {}

        # 检查是否存在二级/三级热处理
        secondary_cols = [c for c in self.data.columns if 'secondary' in c.lower() and 'temp' in c.lower()]
        tertiary_cols = [c for c in self.data.columns if 'tertiary' in c.lower() and 'temp' in c.lower()]

        if secondary_cols:
            secondary_col = secondary_cols[0]
            # 仅一级热处理
            mask_primary_only = (self.data[secondary_col] == 0)
            datasets['primary_only'] = self.data[mask_primary_only].copy()
            print(f"仅一级热处理: {mask_primary_only.sum()} 样本")

        if secondary_cols and tertiary_cols:
            tertiary_col = tertiary_cols[0]
            # 二级热处理（有二级但无三级）
            mask_secondary = (self.data[secondary_col] > 0) & (self.data[tertiary_col] == 0)
            datasets['with_secondary'] = self.data[mask_secondary].copy()
            print(f"二级热处理: {mask_secondary.sum()} 样本")

            # 三级热处理
            mask_tertiary = (self.data[tertiary_col] > 0)
            datasets['with_tertiary'] = self.data[mask_tertiary].copy()
            print(f"三级热处理: {mask_tertiary.sum()} 样本")

        # 全部数据
        datasets['all'] = self.data.copy()

        return datasets

    def get_feature_target(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取特征和目标变量

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y)
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")

        X = self.data[self.feature_cols].values
        y = self.data[self.target_cols].values

        return X, y

    def split_data(self,
                   test_size: Optional[float] = None,
                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        分割数据集

        Args:
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = self.config.get('test_size', 0.2)
        if random_state is None:
            random_state = self.config.get('random_state', 42)

        X, y = self.get_feature_target()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        print(f"数据分割完成:")
        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  测试集: {X_test.shape[0]} 样本")

        return X_train, X_test, y_train, y_test

    def normalize(self,
                  X_train: np.ndarray,
                  X_test: np.ndarray,
                  y_train: np.ndarray,
                  y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        数据标准化

        Args:
            X_train, X_test, y_train, y_test: 数据集

        Returns:
            Tuple: 标准化后的数据集
        """
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)

        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)

        self._is_fitted = True

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        反向转换目标变量

        Args:
            y_scaled: 标准化的目标变量

        Returns:
            np.ndarray: 原始空间的目标变量
        """
        if not self._is_fitted:
            raise ValueError("请先调用 normalize() 进行标准化")

        return self.scaler_y.inverse_transform(y_scaled)

    def get_train_test_data(self,
                           normalize: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        一站式获取训练和测试数据

        Args:
            normalize: 是否标准化

        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        if normalize is None:
            normalize = self.config.get('normalize', True)

        X_train, X_test, y_train, y_test = self.split_data()

        if normalize:
            X_train, X_test, y_train, y_test = self.normalize(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test

    def get_feature_bounds(self) -> Tuple[List[float], List[float]]:
        """
        获取特征变量的边界

        Returns:
            Tuple: (lower_bounds, upper_bounds)
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")

        X = self.data[self.feature_cols]
        lb = X.min().tolist()
        ub = X.max().tolist()

        return lb, ub

    def get_feature_names(self) -> List[str]:
        """
        获取特征列名称

        Returns:
            List[str]: 特征列名称列表
        """
        return self.feature_cols.copy() if self.feature_cols else []

    def get_target_names(self) -> List[str]:
        """
        获取目标列名称

        Returns:
            List[str]: 目标列名称列表
        """
        return self.target_cols.copy()


def create_data_processor(config: dict) -> DataProcessor:
    """
    创建数据处理器实例

    Args:
        config: 数据配置字典

    Returns:
        DataProcessor: 数据处理器实例
    """
    return DataProcessor(config)
