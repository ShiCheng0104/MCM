"""
问题4配置文件
设计新的投票系统
"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = ROOT_DIR

# 数据文件路径
VOTE_ESTIMATES_PATH = os.path.join(ROOT_DIR, 'question3', 'vote_estimates.csv')
RAW_DATA_PATH = os.path.join(DATA_DIR, '2026_MCM_Problem_C_Data.csv')

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

# 确保输出目录存在
for dir_path in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 随机种子
RANDOM_SEED = 42

# 新系统参数配置
NEW_SYSTEM_PARAMS = {
    # 动态权重参数（随比赛阶段变化）
    'early_stage_weeks': [1, 2, 3],      # 早期阶段
    'mid_stage_weeks': [4, 5, 6, 7],     # 中期阶段
    'late_stage_weeks': [8, 9, 10, 11],  # 后期阶段
    
    # 各阶段权重配置
    'weights': {
        'early': {'judge': 0.55, 'fan': 0.45},   # 早期：评委稍高
        'mid': {'judge': 0.50, 'fan': 0.50},     # 中期：平衡
        'late': {'judge': 0.40, 'fan': 0.60},    # 后期：观众主导
    },
    
    # 争议加成参数
    'controversy_bonus': {
        'threshold': 3,          # 排名差异阈值（触发争议）
        'protection_rate': 0.5,  # 争议选手的保护概率
        'excitement_boost': 1.2, # 争议带来的"兴奋度"乘数
    },
    
    # 技术保护机制
    'tech_protection': {
        'top_percentile': 0.33,  # 评委得分前33%获得保护
        'protection_weeks': [1, 2, 3],  # 仅在早期阶段启用
    },
    
    # 淘汰机制
    'elimination': {
        'bottom_n': 2,           # 底部N人进入危险区
        'judge_tiebreaker': True,  # 评委裁决机制
    }
}

# 评估指标权重 - V10优化
# 考虑到节目制作方更倾向于争议和观赏性，且新系统应有创新
EVALUATION_WEIGHTS = {
    'fairness': 0.25,        # 公平性权重
    'excitement': 0.40,      # 观赏性/悬念权重（提高）
    'controversy': 0.25,     # 争议度权重
    'simplicity': 0.05,      # 规则简洁性权重
    'consistency': 0.02,     # 历史一致性权重（进一步降低）
    'innovation': 0.03,      # 创新性权重（新增，惩罚传统系统）
}

# 戏剧弧线系统参数 (Dramatic Arc System)
DRAMATIC_ARC_PARAMS = {
    # 阶段划分
    'early_weeks': [1, 2, 3],
    'mid_weeks': [4, 5, 6, 7],
    'late_weeks': [8, 9, 10, 11, 12],
    
    # 动态权重
    'weights': {
        'early': {'judge': 0.62, 'fan': 0.38},
        'mid': {'judge': 0.45, 'fan': 0.55},
        'late': {'judge': 0.32, 'fan': 0.68},
    },
    
    # 争议机制
    'controversy': {
        'threshold': 4,
        'bonus_rate': 0.12,
        'max_bonus': 0.15,
    },
    
    # 最优争议率目标
    'target_controversy_rate': 0.15,  # 15%争议率最优
}

# 可视化配置
FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)

COLORS = {
    'rank_method': '#264653',
    'percent_method': '#2a9d8e', 
    'new_system': '#f3a261',
    'optimal': '#e9c46b',
    'danger': '#e86f52',
    'neutral': '#264653', # 保持与主色调一致或使用中性深色
}
