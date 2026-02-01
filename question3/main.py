"""
Question 3: 影响因素分析主程序
分析专业舞伴和名人特征对比赛表现（评委得分和观众投票）的影响

主要分析内容:
1. 舞伴效应 - 不同专业舞伴对选手表现的影响
2. 年龄效应 - 选手年龄对表现的影响
3. 行业效应 - 选手职业背景对表现的影响
4. 国籍效应 - 国内/国际选手的表现差异
5. 评委得分 vs 观众投票的效应差异对比
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import Question3Analyzer


def main():
    """主函数"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     MCM 2026 Problem C - Question 3                         ║
    ║     Impact Factor Analysis                                   ║
    ║                                                              ║
    ║     分析专业舞伴和名人特征对比赛表现的影响                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建分析器并运行
    analyzer = Question3Analyzer()
    analyzer.run_full_analysis()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     分析完成!                                                ║
    ║                                                              ║
    ║     输出文件:                                                ║
    ║     - outputs/figures/ : 可视化图表                         ║
    ║     - outputs/tables/  : 数据表格                           ║
    ║     - outputs/reports/ : 分析报告                           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
