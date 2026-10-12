"""
端到端流程 2: 数据准备 + 模型训练 + 超参数优化

完整流程:
1. 加载增强数据集
2. 数据预处理
3. 模型训练
4. 超参数优化 (PIMO/Optuna/GA/PSO)

使用场景: 已有增强数据集,需要训练和优化机器学习模型
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config
from src.ml.data_processor import DataProcessor
from src.ml.model_trainer import ModelTrainer
from src.optimization.pimo_optimizer import PIMOOptimizer
import pandas as pd


def main():
    """端到端流程: 数据准备 → 模型训练 → 超参数优化"""

    # 加载配置
    config = get_config(mode='workstation')

    print("=" * 60)
    print("端到端流程 2: 数据准备 + 模型训练 + 超参数优化")
    print("=" * 60)

    # ==================== 步骤 1: 数据准备 ====================
    print("\n[步骤 1/3] 数据准备...")

    processor_config = dict(config['data'])
    processor_config['normalize'] = False
    processor = DataProcessor(processor_config)
    processor.load_data(config['data']['data_path'])
    X_train, X_test, y_train, y_test = processor.get_train_test_data(normalize=False)
    feature_columns = processor.get_feature_names()

    print(f"✓ 数据加载完成")
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")

    # ==================== 步骤 2: 模型训练 ====================
    print("\n[步骤 2/3] 模型训练...")

    trainer = ModelTrainer(config)
    model = trainer.train(
        X_train, y_train,
        model_name=config['model']['default']
    )

    # 评估模型
    score = trainer.evaluate(model, X_test, y_test)
    print(f"✓ 模型训练完成")
    print(f"  R² Score: {score:.4f}")

    # ==================== 步骤 3: 超参数优化 ====================
    print("\n[步骤 3/3] 超参数优化...")

    optimizer = PIMOOptimizer(config)
    best_params, best_score = optimizer.optimize(
        X_train, y_train,
        model_name=config['model']['default']
    )

    print(f"✓ 优化完成")
    print(f"  最佳 R² Score: {best_score:.4f}")
    print(f"  最佳参数: {best_params}")

    # 使用最佳参数重新训练
    final_model = trainer.train(
        X_train, y_train,
        model_name=config['model']['default'],
        params=best_params
    )

    # 保存最终模型
    model_path = 'results/best_model/optimized_model.pkl'
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_model, model_path)

    feature_columns_path = Path('results/best_model/feature_columns.txt')
    feature_columns_path.write_text('\n'.join(feature_columns), encoding='utf-8')

    print(f"✓ 最终模型已保存: {model_path}")
    print(f"✓ 特征列已保存: {feature_columns_path}")

    print("\n" + "=" * 60)
    print("流程完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
