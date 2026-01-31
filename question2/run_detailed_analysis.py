# -*- coding: utf-8 -*-
"""
运行详细分析
生成：
1. 周次级别详细对比表
2. 评委裁决模拟
3. 可视化轨迹图
"""

import os
import sys

# 确保能导入analysis包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.detailed_analysis import run_detailed_analysis

if __name__ == "__main__":
    # 配置路径
    vote_estimates_path = "vote_estimates.csv"
    original_data_path = "../2026_MCM_Problem_C_Data.csv"
    output_dir = "outputs"
    
    # 运行详细分析
    run_detailed_analysis(vote_estimates_path, original_data_path, output_dir)
