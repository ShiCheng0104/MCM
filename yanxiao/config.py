"""
配置参数模块
"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '2026_MCM_Problem_C_Data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据配置
MAX_WEEKS = 11
MAX_JUDGES = 4
SEASONS_RANK_METHOD = list(range(1, 3)) + list(range(28, 35))  # 排名法赛季
SEASONS_PERCENT_METHOD = list(range(3, 28))  # 百分比法赛季

# 模型配置
class ModelConfig:
    # 约束优化配置
    OPTIMIZATION_METHOD = 'SLSQP'
    OPTIMIZATION_MAX_ITER = 1000
    OPTIMIZATION_TOL = 1e-6
    
    # 贝叶斯模型配置
    MCMC_SAMPLES = 2000
    MCMC_TUNE = 1000
    MCMC_CHAINS = 2
    
    # 基线模型配置
    BASELINE_ALPHA = 0.5  # 评委得分对投票的影响系数
    
    # 随机种子
    RANDOM_SEED = 42

# 评估配置
class EvalConfig:
    # 一致性检验阈值
    CONSISTENCY_THRESHOLD = 0.95
    
    # 置信区间水平
    CONFIDENCE_LEVEL = 0.95
    
    # 不确定性分类阈值
    CV_HIGH_THRESHOLD = 0.1
    CV_MEDIUM_THRESHOLD = 0.3

# 可视化配置
class PlotConfig:
    FIGURE_DPI = 150
    FIGURE_SIZE = (12, 8)
    COLOR_PALETTE = 'husl'
    FONT_SIZE = 12
