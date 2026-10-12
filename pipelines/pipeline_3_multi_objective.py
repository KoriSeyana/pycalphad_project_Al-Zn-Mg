"""
端到端流程 3: 多目标优化 (集成代理模型)

完整流程:
1. 加载主模型和代理模型
2. 配置多目标优化问题
3. 运行 NSGA-II 优化
4. 输出 Pareto 前沿

使用场景: 已有训练好的主模型和代理模型,进行成分和工艺参数的多目标优化
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config
from src.optimization.multi_objective import OptimizationConfig, AlloyOptimizationPipeline, FeatureAssembler, PhaseFractionPredictor
from src.optimization.surrogate_model import load_surrogate_model
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import pandas as pd
import pickle


def main():
    """端到端流程: 多目标优化 (集成代理模型)"""

    # 加载配置
    config = get_config(mode='workstation')

    print("=" * 60)
    print("端到端流程 3: 多目标优化 (集成代理模型)")
    print("=" * 60)

    # ==================== 步骤 1: 加载模型 ====================
    print("\n[步骤 1/3] 加载模型...")

    # 加载主模型
    main_model_path = 'results/best_model/optimized_model.pkl'
    with open(main_model_path, 'rb') as f:
        main_model = pickle.load(f)
    print(f"✓ 主模型已加载: {main_model_path}")

    # 加载代理模型
    surrogate_model = load_surrogate_model(config['surrogate']['model_path'])
    print(f"✓ 代理模型已加载: {config['surrogate']['model_path']}")

    # ==================== 步骤 2: 配置优化问题 ====================
    print("\n[步骤 2/3] 配置优化问题...")

    opt_config = OptimizationConfig(
        main_model_path=main_model_path,
        surrogate_model_path=config['surrogate']['model_path'],
        pop_size=config['optimizer']['n_agents'],
        n_gen=config['optimizer']['total_evaluations'] // config['optimizer']['n_agents']
    )

    # 构建特征拼接器
    phase_predictor = PhaseFractionPredictor(
        method='surrogate',
        surrogate_model_path=config['surrogate']['model_path']
    )

    # 获取主模型的特征列名
    feature_columns_path = 'results/best_model/feature_columns.txt'
    with open(feature_columns_path, 'r') as f:
        main_model_features = [line.strip() for line in f.readlines()]

    var_names = [v[0] for v in opt_config.decision_variables]

    assembler = FeatureAssembler(
        main_model_features=main_model_features,
        decision_var_names=var_names,
        phase_predictor=phase_predictor
    )

    # 创建优化问题
    problem = AlloyOptimizationPipeline(
        config=opt_config,
        main_model=main_model,
        feature_assembler=assembler
    )

    print(f"✓ 优化问题已配置")
    print(f"  决策变量数: {problem.n_var}")
    print(f"  目标数: {problem.n_obj}")
    print(f"  种群大小: {opt_config.pop_size}")
    print(f"  迭代代数: {opt_config.n_gen}")

    # ==================== 步骤 3: 运行优化 ====================
    print("\n[步骤 3/3] 运行 NSGA-II 优化...")

    algorithm = NSGA2(
        pop_size=opt_config.pop_size,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    result = minimize(
        problem,
        algorithm,
        ('n_gen', opt_config.n_gen),
        verbose=True,
        seed=config['data']['random_state']
    )

    print(f"\n✓ 优化完成")
    print(f"  Pareto 解数量: {len(result.F)}")

    # 保存结果
    output_dir = Path('results/multi_objective')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 Pareto 前沿
    pareto_df = pd.DataFrame(
        result.X,
        columns=[var[0] for var in opt_config.decision_variables]
    )
    pareto_df['UTS'] = -result.F[:, 0]  # 转回最大化
    pareto_df['Elongation'] = -result.F[:, 1]

    pareto_path = output_dir / 'pareto_front.csv'
    pareto_df.to_csv(pareto_path, index=False)
    print(f"✓ Pareto 前沿已保存: {pareto_path}")

    # 输出最优解示例
    print("\n最优解示例 (前 5 个):")
    print(pareto_df.head())

    print("\n" + "=" * 60)
    print("流程完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
