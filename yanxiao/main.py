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
from src.consistency_check import ConsistencyChecker, run_consistency_check
from src.uncertainty_measure import UncertaintyAnalyzer, run_uncertainty_analysis
from models.precise_vote_model import PreciseVoteModel
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
    # 2. 训练精确投票反推模型
    # ========================================
    print("[2/5] 训练精确投票反推模型...")
    print("-" * 50)
    
    # 使用精确投票反推模型（严格按照排名法/百分比法）
    model = PreciseVoteModel(random_seed=ModelConfig.RANDOM_SEED)
    model.fit(weekly_data, elimination_info)
    # 获取模型在训练阶段的验证结果
    elim_results = model.predict_elimination(weekly_data, elimination_info)
    print()
    
    # ========================================
    # 3. 获取投票估计结果
    # ========================================
    print("[3/5] 获取投票估计结果...")
    print("-" * 50)
    
    # 获取投票估计
    estimates_df = model.get_vote_estimates()
    estimates_df.to_csv(os.path.join(OUTPUT_DIR, 'vote_estimates.csv'), index=False)
    print(f"  投票估计结果已保存到: {OUTPUT_DIR}/vote_estimates.csv")
    print(f"  共 {len(estimates_df)} 条估计记录")
    
    # 获取consistency_check需要的格式
    estimates = model.get_estimates_dict()
    
    print()
    
    # ========================================
    # 4. 一致性检验（使用模型预测结果）
    # ========================================
    print("[4/5] 一致性检验...")
    print("-" * 50)
    
    # 统计数据覆盖情况
    n_estimates = len(estimates)
    n_eliminations = len(elimination_info)
    print(f"\n  数据覆盖情况:")
    print(f"  ├─ 模型成功反推的周次: {n_estimates}")
    print(f"  └─ 数据集中淘汰记录总数: {n_eliminations}")
    if n_eliminations > n_estimates:
        print(f"  注: {n_eliminations - n_estimates} 条淘汰记录缺少完整评分数据,模型无法反推")
    
    print(f"\n  一致性检验将在所有有完整数据的淘汰周次上进行...")
    
    checker, consistency_summary = run_consistency_check(
        estimates,
        elimination_info,
        cleaned_data,
        verbose=True
    )
    
    # 诊断排名法准确率低的问题
    if checker.results is not None:
        rank_results = checker.results[checker.results['method'] == 'rank']
        if len(rank_results) > 0 and rank_results['is_correct'].mean() < 0.7:
            print(f"\n\u26a0️  排名法预测准确率低 ({rank_results['is_correct'].mean():.2%}), 显示错误案例:")
            rank_errors = rank_results[~rank_results['is_correct']].head(5)
            for _, row in rank_errors.iterrows():
                print(f"  Season {row['season']}, Week {row['week']}: "
                      f"预测={row['predicted_eliminated']}, 实际={row['actual_eliminated']}")
    
    # 用模型的准确率覆盖一致性检验的结果
    consistency_summary['accuracy'] = elim_results['accuracy']
    consistency_summary['bottom_accuracy'] = elim_results['bottom_accuracy']
    
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
    
    # 获取投票样本用于不确定性分析
    samples_dict = model.get_samples_dict()
    print(f"  获取样本: {len(samples_dict)} 个选手-周次")
    
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
    
    print(f"\n  【精确投票反推模型】")
    print(f"  ├─ 使用排名法赛季: 1-2, 28-34")
    print(f"  └─ 使用百分比法赛季: 3-27")
    
    print(f"\n  【舞伴效应 Top 3】")
    sorted_partners = sorted(model.partner_effects.items(), key=lambda x: x[1], reverse=True)
    for p, e in sorted_partners[:3]:
        print(f"    {p}: {e:+.3f}")
    
    print(f"\n  【行业效应 Top 3】")
    sorted_industries = sorted(model.industry_effects.items(), key=lambda x: x[1], reverse=True)
    for ind, e in sorted_industries[:3]:
        print(f"    {ind}: {e:+.3f}")
    
    print(f"\n  【不确定性分析】")
    print(f"  ├─ 平均变异系数(CV): {uncertainty_summary['mean_cv']:.4f}")
    print(f"  └─ 高确定性估计占比: {uncertainty_summary['high_certainty_pct']:.2%}")
    
    print(f"\n  所有输出文件保存在: {OUTPUT_DIR}")
    print("="*70 + "\n")
    
    return {
        'estimates': estimates,
        'model': model,
        'consistency_summary': consistency_summary,
        'uncertainty_summary': uncertainty_summary,
        'checker': checker,
        'analyzer': analyzer
    }


if __name__ == '__main__':
    results = main()
