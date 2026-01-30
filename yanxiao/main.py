"""
主程序入口
问题1：观众投票估计模型
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, OUTPUT_DIR, ModelConfig, EvalConfig
from src.data_preprocessing import DataPreprocessor, load_and_preprocess
from src.vote_estimator import VoteEstimator
from src.consistency_check import ConsistencyChecker, run_consistency_check
from src.uncertainty_measure import UncertaintyAnalyzer, run_uncertainty_analysis
from visualization.plots import VotePlotter


def print_header():
    """打印程序头部信息"""
    print("="*70)
    print("  2026 MCM Problem C: Dancing with the Stars")
    print("  问题1: 观众投票估计模型")
    print("="*70)
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def main():
    """主函数"""
    print_header()
    
    # ========================================
    # 1. 数据加载与预处理
    # ========================================
    print("[1/5] 加载并预处理数据...")
    print("-" * 50)
    
    preprocessor, data = load_and_preprocess(DATA_DIR)
    
    cleaned_data = data['cleaned_data']
    weekly_data = data['weekly_data']
    season_week_data = data['season_week_data']
    elimination_info = data['elimination_info']
    season_info = data['season_info']
    
    print(f"  - 选手记录数: {len(cleaned_data)}")
    print(f"  - 周级别记录数: {len(weekly_data)}")
    print(f"  - 赛季数: {len(season_info)}")
    print(f"  - 淘汰记录数: {len(elimination_info)}")
    print()
    
    # ========================================
    # 2. 模型拟合
    # ========================================
    print("[2/5] 拟合投票估计模型...")
    print("-" * 50)
    
    # 使用约束优化模型作为主要模型
    estimator = VoteEstimator(
        model_type='constrained',
        config={
            'opt_method': ModelConfig.OPTIMIZATION_METHOD,
            'max_iter': ModelConfig.OPTIMIZATION_MAX_ITER,
            'prior_weight': 0.5,
            'random_seed': ModelConfig.RANDOM_SEED
        }
    )
    
    # 拟合模型
    estimator.fit(weekly_data, season_week_data, elimination_info)
    print()
    
    # ========================================
    # 3. 估计观众投票
    # ========================================
    print("[3/5] 估计所有周次的观众投票...")
    print("-" * 50)
    
    estimates = estimator.estimate(
        season_week_data,
        elimination_info,
        total_votes=1e6
    )
    
    # 导出估计结果
    estimates_df = estimator.get_all_estimates_df()
    estimates_df.to_csv(os.path.join(OUTPUT_DIR, 'vote_estimates.csv'), index=False)
    print(f"  投票估计结果已保存到: {OUTPUT_DIR}/vote_estimates.csv")
    print()
    
    # ========================================
    # 4. 一致性检验
    # ========================================
    print("[4/5] 进行一致性检验...")
    print("-" * 50)
    
    checker, consistency_summary = run_consistency_check(
        estimates,
        elimination_info,
        cleaned_data,
        verbose=True
    )
    
    # 保存一致性检验结果
    checker.results.to_csv(os.path.join(OUTPUT_DIR, 'consistency_results.csv'), index=False)
    
    # 保存错误预测案例
    incorrect = checker.get_incorrect_predictions()
    incorrect.to_csv(os.path.join(OUTPUT_DIR, 'incorrect_predictions.csv'), index=False)
    print(f"  错误预测案例数: {len(incorrect)}")
    print()
    
    # ========================================
    # 5. 不确定性分析
    # ========================================
    print("[5/5] 进行不确定性分析...")
    print("-" * 50)
    
    # 生成投票样本用于不确定性分析
    samples_dict = {}
    if hasattr(estimator, 'bayesian') and hasattr(estimator.bayesian, 'samples'):
        samples_dict = estimator.bayesian.samples
    else:
        # 使用约束优化模型生成样本
        print("  生成投票样本...")
        for (season, week), contestants in list(season_week_data.items())[:50]:  # 限制数量
            if len(contestants) > 1:
                scores = contestants['total_score'].values
                eliminated_idx = np.argmin(scores)  # 简化假设
                samples = estimator.constrained.generate_vote_samples(
                    scores, eliminated_idx, season, n_samples=100
                )
                samples_dict[(season, week)] = samples
    
    analyzer, uncertainty_summary = run_uncertainty_analysis(
        estimates,
        samples_dict,
        weekly_data,
        verbose=True
    )
    
    # ========================================
    # 6. 可视化
    # ========================================
    print("\n[6/6] 生成可视化图表...")
    print("-" * 50)
    
    plotter = VotePlotter(output_dir=os.path.join(OUTPUT_DIR, 'figures'))
    
    # 选择一些示例周次
    sample_weeks = [
        (2, 8),   # Jerry Rice争议赛季
        (11, 10), # Bristol Palin争议赛季
        (27, 9),  # Bobby Bones争议赛季
        (1, 4),   # 早期赛季示例
        (20, 5),  # 中期赛季示例
    ]
    
    # 生成并保存图表
    plotter.save_all_figures(
        estimates=estimates,
        consistency_results=checker.results,
        uncertainty_stats=analyzer.uncertainty_stats,
        summary=consistency_summary,
        sample_weeks=sample_weeks
    )
    
    # ========================================
    # 结果汇总
    # ========================================
    print("\n" + "="*70)
    print("  分析完成！结果汇总")
    print("="*70)
    print(f"  淘汰预测准确率: {consistency_summary['elimination_accuracy']:.2%}")
    print(f"  底2预测准确率: {consistency_summary['bottom_two_accuracy']:.2%}")
    print(f"  平均变异系数(CV): {uncertainty_summary['mean_cv']:.4f}")
    print(f"  高确定性估计占比: {uncertainty_summary['high_certainty_pct']:.2%}")
    print()
    print(f"  所有输出文件保存在: {OUTPUT_DIR}")
    print("="*70 + "\n")
    
    return {
        'estimates': estimates,
        'consistency_summary': consistency_summary,
        'uncertainty_summary': uncertainty_summary,
        'checker': checker,
        'analyzer': analyzer
    }


if __name__ == '__main__':
    results = main()
