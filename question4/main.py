"""
Question 4: 新投票系统设计
主程序入口
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import Question4Analyzer


def main():
    """主函数"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     MCM 2026 Problem C - Question 4                             ║
    ║     New Voting System Design                                     ║
    ║                                                                  ║
    ║     设计新的投票系统，平衡公平性与观赏性                           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    analyzer = Question4Analyzer()
    analyzer.run_full_analysis()
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     分析完成!                                                    ║
    ║                                                                  ║
    ║     输出文件:                                                    ║
    ║     - outputs/figures/ : 可视化图表                             ║
    ║     - outputs/tables/  : 数据表格                               ║
    ║     - outputs/reports/ : 分析报告                               ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
