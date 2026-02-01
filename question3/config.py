"""
问题3配置文件
分析专业舞伴和名人特征对比赛表现的影响
"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = ROOT_DIR

# 数据文件路径
VOTE_ESTIMATES_PATH = os.path.join(BASE_DIR, 'vote_estimates.csv')
RAW_DATA_PATH = os.path.join(DATA_DIR, '2026_MCM_Problem_C_Data.csv')

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

# 确保输出目录存在
for dir_path in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 分析配置
RANDOM_SEED = 42

# 特征分组
INDUSTRY_CATEGORIES = [
    'Actor/Actress', 'Athlete', 'Singer/Rapper', 'TV Personality',
    'Model', 'News Anchor', 'Sports Broadcaster', 'Comedian',
    'Reality TV Star', 'Olympian', 'Social Media Personality', 'Other'
]

AGE_BINS = [0, 25, 35, 45, 55, 100]
AGE_LABELS = ['Young (≤25)', 'Prime (26-35)', 'Middle (36-45)', 'Mature (46-55)', 'Senior (>55)']

# 模型参数
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_SEED
}

# 可视化配置
FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)

# 配色方案
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#3B1F2B',
    'judge': '#2E86AB',
    'fan': '#A23B72'
}
