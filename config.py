"""
项目配置文件 - 整合版
集中管理 CALPHAD 计算、机器学习、优化算法的所有配置参数
"""

# ==================== 数据配置 ====================
DATA_CONFIG = {
    # 数据路径
    'data_path': 'output/enhanced_dataset.csv',  # 用户本地准备的增强数据集
    'tdb_path': 'data/COST507.pycalphad.tdb',  # CALPHAD 热力学数据库

    # 目标列
    'target_cols': ['Ultimate tensile strength (MPa)', 'Elongation (%)'],
    'exclude_cols': ['Alloys'],  # 排除的列

    # 数据划分
    'test_size': 0.2,
    'random_state': 2024256,

    # 特征工程 - PIMO 物理化学参数
    'normalize': True,
    'normalize_method': 'zero_aware',  # 零值感知归一化
    'add_indicator_vars': True,  # 添加热处理指示变量
    'add_element_params': True,  # 添加元素物理化学参数
    'add_interaction_features': True,  # 添加交互特征

    # 特征工程 - CALPHAD 相分数特征
    'add_calphad_features': False,  # 是否添加 CALPHAD 特征(按需启用)
}

# ==================== CALPHAD 配置 ====================
CALPHAD_CONFIG = {
    'workers': 16,  # 并行进程数
    'batch_size': 400,  # 每批计算点数
    'checkpoint_dir': 'output/checkpoints',  # 检查点目录
    'max_points': 10000,  # 最大计算点数
}

# ==================== 代理模型配置 ====================
SURROGATE_CONFIG = {
    'model_path': 'output/surrogate_models/lightgbm_surrogate_model.pkl',
    'use_surrogate': False,  # 多目标优化时启用
    'input_features': ['ZN', 'MG', 'CU', 'CR', 'MN', 'ZR', 'TI', 'FE', 'SI',
                      'solution_temp', 'aging_temp'],
    'output_features': [
        'sol_FCC_A1_fraction', 'sol_ETA_fraction', 'sol_T_PHASE_fraction',
        'sol_S_PHASE_fraction', 'sol_AL3ZR_fraction', 'sol_MG2SI_fraction',
        'sol_AL_FE_SI_fraction', 'sol_HCP_A3_fraction',
        'age_FCC_A1_fraction', 'age_ETA_fraction', 'age_T_PHASE_fraction',
        'age_S_PHASE_fraction', 'age_AL3ZR_fraction', 'age_MG2SI_fraction',
        'age_AL_FE_SI_fraction', 'age_HCP_A3_fraction',
    ],
}

# ==================== 模型配置 ====================
AVAILABLE_MODELS = [
    # 集成学习模型
    'xgboost', 'lightgbm', 'catboost', 'rf', 'et', 'gbr', 'ada',
    # 线性模型
    'ridge', 'lasso', 'en', 'br',
    # 其他
    'svr', 'knn', 'mlp', 'dt', 'huber',
]

MODEL_NAME_MAP = {
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost',
    'rf': 'rf',
    'et': 'et',
    'gbr': 'gbr',
    'ada': 'ada',
    'ridge': 'ridge',
    'lasso': 'lasso',
    'en': 'en',
    'br': 'br',
    'svr': 'svm',
    'knn': 'knn',
    'mlp': 'mlp',
    'dt': 'dt',
    'huber': 'huber',
}

DEFAULT_MODEL = 'xgboost'

# ==================== 优化器配置 ====================
AVAILABLE_OPTIMIZERS = [
    'pimo',     # PIMO 算法
    'optuna',   # Optuna 贝叶斯优化
    'ga',       # 遗传算法
    'pso',      # 粒子群优化
    'pycaret',  # PyCaret 内置调优
]

OPTIMIZER_CONFIG = {
    'n_agents': 8,
    'total_evaluations': 800,
    'patience': 70,
    'n_jobs': -1,
    'cv_folds': 5,
}

PIMO_CONFIG = {
    'epsilon': 1e-6,
}

OPTUNA_CONFIG = {
    'sampler': 'TPE',
}

GA_CONFIG = {
    'num_parents_mating': None,
    'mutation_percent_genes': 10,
}

PSO_CONFIG = {
    'c1': 0.5,
    'c2': 0.3,
    'w': 0.9,
}

# ==================== 评估配置 ====================
EVAL_CONFIG = {
    'target_names': ['UTS (MPa)', 'Elongation (%)'],
    'metrics': ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE'],
}

# ==================== 输出配置 ====================
OUTPUT_CONFIG = {
    'results_dir': 'results',
    'save_model': True,
    'save_predictions': True,
    'save_convergence': True,
}

# ==================== 工作站配置 ====================
WORKSTATION_CONFIG = {
    'n_agents': 64,
    'total_evaluations': 6400,
    'n_jobs': -1,
    'calphad_workers': 16,
}

def get_config(mode='local'):
    """
    获取配置

    Args:
        mode: 'local' 本地测试配置, 'workstation' 工作站配置

    Returns:
        dict: 合并后的配置
    """
    config = {
        'data': DATA_CONFIG.copy(),
        'calphad': CALPHAD_CONFIG.copy(),
        'surrogate': SURROGATE_CONFIG.copy(),
        'model': {
            'default': DEFAULT_MODEL,
            'available': AVAILABLE_MODELS,
            'name_map': MODEL_NAME_MAP,
        },
        'optimizer': OPTIMIZER_CONFIG.copy(),
        'pimo': PIMO_CONFIG.copy(),
        'optuna': OPTUNA_CONFIG.copy(),
        'ga': GA_CONFIG.copy(),
        'pso': PSO_CONFIG.copy(),
        'eval': EVAL_CONFIG.copy(),
        'output': OUTPUT_CONFIG.copy(),
    }

    if mode == 'workstation':
        config['optimizer'].update(WORKSTATION_CONFIG)
        config['calphad']['workers'] = WORKSTATION_CONFIG['calphad_workers']

    return config

def print_config(config):
    """打印当前配置"""
    print("=" * 50)
    print("当前配置:")
    print("=" * 50)
    for section, values in config.items():
        print(f"\n[{section}]")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {values}")
    print("=" * 50)
