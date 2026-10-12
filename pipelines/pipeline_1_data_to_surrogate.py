"""
端到端流程 1: 虚拟数据生成 + 特征增强 + 代理模型训练

完整流程:
1. 生成虚拟数据 (CALPHAD 计算)
2. 特征增强 (添加相分数特征)
3. 训练代理模型 (LightGBM)

使用场景: 从零开始构建完整的数据集和代理模型
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config
from src.calphad.data_generator import VirtualDataGenerator
from src.calphad.feature_enhancer import FeatureEnhancer
import pandas as pd
import pickle
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def main():
    """端到端流程: 虚拟数据生成 → 特征增强 → 代理模型训练"""

    # 加载配置
    config = get_config(mode='workstation')

    print("=" * 60)
    print("端到端流程 1: 虚拟数据生成 + 特征增强 + 代理模型训练")
    print("=" * 60)

    # ==================== 步骤 1: 生成虚拟数据 ====================
    print("\n[步骤 1/3] 生成虚拟数据...")

    generator = VirtualDataGenerator(
        db_path=config['data']['tdb_path'],
        output_dir='output',
        workers=config['calphad']['workers']
    )

    virtual_data_path = generator.generate(
        n_samples=config['calphad']['max_points'],
        batch_size=config['calphad']['batch_size']
    )

    print(f"✓ 虚拟数据已生成: {virtual_data_path}")

    # ==================== 步骤 2: 特征增强 ====================
    print("\n[步骤 2/3] 特征增强...")

    enhancer = FeatureEnhancer(
        db_path=config['data']['tdb_path'],
        workers=config['calphad']['workers']
    )

    df_virtual = pd.read_csv(virtual_data_path)
    df_enhanced = enhancer.enhance(df_virtual)

    enhanced_path = 'output/enhanced_virtual_data.csv'
    df_enhanced.to_csv(enhanced_path, index=False)
    print(f"✓ 特征增强完成: {enhanced_path}")

    # ==================== 步骤 3: 训练代理模型 ====================
    print("\n[步骤 3/3] 训练代理模型...")

    # 准备训练数据
    input_features = config['surrogate']['input_features']
    output_features = config['surrogate']['output_features']

    X = df_enhanced[input_features].values
    y = df_enhanced[output_features].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config['data']['random_state']
    )

    # 训练模型
    base_model = LGBMRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        verbose=-1,
        n_jobs=-1
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='uniform_average'))

    print(f"✓ 代理模型训练完成")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    # 保存模型
    model_path = config['surrogate']['model_path']
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✓ 模型已保存: {model_path}")

    print("\n" + "=" * 60)
    print("流程完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
