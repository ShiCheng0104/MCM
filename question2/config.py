# -*- coding: utf-8 -*-
"""
Question 2: 配置文件
比较和对比两种投票组合方法（排名法和百分比法）
"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')

# 数据文件路径
VOTE_ESTIMATES_FILE = os.path.join(BASE_DIR, 'vote_estimates.csv')
ORIGINAL_DATA_FILE = os.path.join(ROOT_DIR, '2026_MCM_Problem_C_Data.csv')

# 赛季与投票方法的对应关系
# 排名法(rank): 第1-2季, 第28-34季
# 百分比法(percent): 第3-27季
RANK_METHOD_SEASONS = list(range(1, 3)) + list(range(28, 35))  # [1, 2, 28, 29, 30, 31, 32, 33, 34]
PERCENT_METHOD_SEASONS = list(range(3, 28))  # [3, 4, ..., 27]

# 争议案例定义
CONTROVERSY_CASES = {
    'Jerry Rice': {
        'season': 2,
        'placement': 2,
        'description': '尽管有5周评委得分最低仍获得亚军',
        'lowest_judge_weeks': 5
    },
    'Billy Ray Cyrus': {
        'season': 4,
        'placement': 5,
        'description': '尽管有6周评委得分垫底仍获得第5名',
        'lowest_judge_weeks': 6
    },
    'Bristol Palin': {
        'season': 11,
        'placement': 3,
        'description': '12次获得最低评委得分仍获得第3名',
        'lowest_judge_weeks': 12
    },
    'Bobby Bones': {
        'season': 27,
        'placement': 1,
        'description': '尽管评委得分持续偏低仍赢得冠军',
        'lowest_judge_weeks': 'consistently_low'
    }
}

# 评委裁决规则（从第28季开始）
JUDGE_TIEBREAKER_START_SEASON = 28

# 可视化配置
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
