"""
模型评估与推广 - 配置文件
Model Evaluation and Promotion Configuration
"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, '2026_MCM_Problem_C_Data.csv')

# 各问题路径
Q1_PATH = os.path.join(BASE_DIR, 'yanxiao')
Q2_PATH = os.path.join(BASE_DIR, 'question2')
Q3_PATH = os.path.join(BASE_DIR, 'question3')
Q4_PATH = os.path.join(BASE_DIR, 'question4')

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'model_evaluation', 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

# 创建输出目录
for dir_path in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 评估配置
EVALUATION_CONFIG = {
    # 交叉验证设置
    'cv_folds': 5,
    'test_ratio': 0.2,
    'random_state': 42,
    
    # 敏感性分析设置
    'sensitivity_perturbation_range': [0.9, 0.95, 1.0, 1.05, 1.1],  # ±5%, ±10%
    'sensitivity_n_iterations': 100,
    
    # Bootstrap设置
    'bootstrap_n_iterations': 1000,
    'bootstrap_confidence_level': 0.95,
    
    # 模型泛化测试
    'holdout_seasons': [30, 31, 32, 33, 34],  # 最近5个赛季作为测试集
}

# 可视化配置
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 150,
    'style': 'seaborn-v0_8-whitegrid',
    'colors': {
        'Q1': '#2ecc71',  # 绿色
        'Q2': '#3498db',  # 蓝色
        'Q3': '#e74c3c',  # 红色
        'Q4': '#f39c12',  # 橙色
        'baseline': '#95a5a6',  # 灰色
    },
    'font_size': 12,
}

# 模型信息
MODEL_INFO = {
    'Q1': {
        'name': 'Vote Estimation Model',
        'chinese_name': '观众投票估计模型',
        'type': 'Bayesian Hierarchical + Constrained Optimization',
        'target': 'vote_share',
        'metrics': ['accuracy', 'consistency', 'uncertainty'],
    },
    'Q2': {
        'name': 'Voting Method Comparison',
        'chinese_name': '投票方法对比分析',
        'type': 'Comparative Analysis + Simulation',
        'target': 'elimination_prediction',
        'metrics': ['method_agreement', 'controversy_detection'],
    },
    'Q3': {
        'name': 'Factor Impact Analysis',
        'chinese_name': '影响因素分析模型',
        'type': 'Random Forest + OLS Regression',
        'target': ['judge_score', 'fan_vote'],
        'metrics': ['r2', 'rmse', 'feature_importance'],
    },
    'Q4': {
        'name': 'Dynamic Voting System',
        'chinese_name': '动态投票系统',
        'type': 'Dramatic Arc System',
        'target': 'composite_score',
        'metrics': ['fairness', 'excitement', 'consistency', 'innovation'],
    },
}
