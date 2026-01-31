# -*- coding: utf-8 -*-
"""
Question 2: 投票方法对比分析主程序

分析内容：
1. 比较排名法和百分比法在各季产生的结果差异
2. 分析争议案例（Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones）
3. 评估评委裁决机制的影响
4. 给出方法推荐建议
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from config import (
    VOTE_ESTIMATES_FILE, ORIGINAL_DATA_FILE, OUTPUT_DIR, FIGURE_DIR,
    RANK_METHOD_SEASONS, PERCENT_METHOD_SEASONS, CONTROVERSY_CASES
)
from analysis import (
    VotingMethodComparator, ControversyAnalyzer, MethodComparisonAnalyzer
)


def load_data():
    """加载数据"""
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)
    
    # 加载观众投票估计数据
    vote_estimates = pd.read_csv(VOTE_ESTIMATES_FILE)
    print(f"观众投票估计数据: {len(vote_estimates)} 条记录")
    print(f"  赛季范围: {vote_estimates['season'].min()} - {vote_estimates['season'].max()}")
    print(f"  列: {list(vote_estimates.columns)}")
    
    # 加载原始比赛数据
    original_data = pd.read_csv(ORIGINAL_DATA_FILE)
    print(f"\n原始比赛数据: {len(original_data)} 条记录")
    print(f"  赛季范围: {original_data['season'].min()} - {original_data['season'].max()}")
    
    return vote_estimates, original_data


def analyze_method_comparison(vote_estimates: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
    """
    分析两种投票方法的对比
    """
    print("\n" + "=" * 60)
    print("任务2.1: 两种投票方法对比分析")
    print("=" * 60)
    
    comparator = VotingMethodComparator(vote_estimates, original_data)
    
    # 对所有赛季应用两种方法
    comparison_df = comparator.compare_all_seasons()
    
    # 保存比较结果
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'method_comparison.csv'), index=False)
    print(f"\n方法比较结果已保存到: {os.path.join(OUTPUT_DIR, 'method_comparison.csv')}")
    
    # 统计分析
    total_weeks = len(comparison_df)
    agree_count = comparison_df['methods_agree'].sum()
    disagree_count = total_weeks - agree_count
    
    print(f"\n【结果汇总】")
    print(f"  总分析周次: {total_weeks}")
    print(f"  两种方法结果一致: {agree_count} 周 ({agree_count/total_weeks*100:.1f}%)")
    print(f"  两种方法结果不一致: {disagree_count} 周 ({disagree_count/total_weeks*100:.1f}%)")
    
    # 按赛季统计
    print(f"\n【按赛季统计不一致周次】")
    disagree_by_season = comparison_df[~comparison_df['methods_agree']].groupby('season').size()
    for season, count in disagree_by_season.items():
        season_total = len(comparison_df[comparison_df['season'] == season])
        print(f"  第{season}季: {count} 周不一致 (共{season_total}周)")
    
    # 分析偏向性
    print(f"\n【观众投票偏向性分析】")
    bias_analysis = comparator.analyze_fan_vote_bias(comparison_df)
    print(f"  不一致率: {bias_analysis['disagreement_rate']*100:.1f}%")
    
    return comparison_df


def analyze_controversy_cases(vote_estimates: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
    """
    分析争议案例
    """
    print("\n" + "=" * 60)
    print("任务2.2: 争议案例深度分析")
    print("=" * 60)
    
    analyzer = ControversyAnalyzer(vote_estimates, original_data)
    
    # 分析已知争议案例
    controversy_summary = analyzer.compare_controversy_cases()
    
    print("\n【已知争议案例分析】")
    for _, row in controversy_summary.iterrows():
        print(f"\n{row['celebrity']} (第{row['season']}季)")
        print(f"  实际名次: 第{row['actual_placement']}名")
        print(f"  参赛周数: {row['weeks_competed']}周")
        print(f"  评委最低分周数: {row['num_lowest_judge_weeks']}周")
        print(f"  排名法淘汰周次: {row['would_be_eliminated_rank_weeks']}周")
        print(f"  百分比法淘汰周次: {row['would_be_eliminated_percent_weeks']}周")
        print(f"  进入底部两人周数(排名法): {row['weeks_in_bottom_two_rank']}周")
        print(f"  进入底部两人周数(百分比法): {row['weeks_in_bottom_two_percent']}周")
    
    # 保存争议案例分析结果
    controversy_summary.to_csv(os.path.join(OUTPUT_DIR, 'controversy_analysis.csv'), index=False)
    
    # 详细分析每个案例
    detailed_analyses = {}
    for name, info in CONTROVERSY_CASES.items():
        analysis = analyzer.analyze_controversy_case(name, info['season'])
        detailed_analyses[name] = analysis
        
        if 'error' not in analysis:
            print(f"\n【{name} 详细轨迹分析】")
            print(f"  {info['description']}")
            
            method_comp = analysis['method_comparison']
            if method_comp['week_details']:
                diff_weeks = [d for d in method_comp['week_details'] if d['methods_differ']]
                if diff_weeks:
                    print(f"  两种方法结果不同的周次:")
                    for week_info in diff_weeks[:5]:  # 只显示前5个
                        print(f"    第{week_info['week']}周: 排名法淘汰{week_info['rank_eliminated']}, 百分比法淘汰{week_info['percent_eliminated']}")
    
    # 识别其他潜在争议案例
    print("\n【其他潜在争议案例】")
    other_controversies = analyzer.identify_additional_controversies(min_lowest_weeks=3)
    
    # 过滤掉已知案例
    known_names = list(CONTROVERSY_CASES.keys())
    other_controversies = other_controversies[
        ~other_controversies['celebrity'].apply(
            lambda x: any(kn.lower() in x.lower() for kn in known_names)
        )
    ]
    
    if len(other_controversies) > 0:
        print("\n最具争议的其他选手（评委最低分≥3周且进入前3名）:")
        top_controversies = other_controversies[other_controversies['is_finalist'] == True].head(10)
        for _, row in top_controversies.iterrows():
            print(f"  {row['celebrity']} (第{row['season']}季): "
                  f"评委最低{row['lowest_judge_weeks']}周, 最终第{row['final_placement']}名")
    
    other_controversies.to_csv(os.path.join(OUTPUT_DIR, 'additional_controversies.csv'), index=False)
    
    return controversy_summary


def analyze_judge_tiebreaker(comparison_df: pd.DataFrame, controversy_df: pd.DataFrame, 
                             vote_estimates: pd.DataFrame) -> None:
    """
    分析评委裁决机制的影响
    """
    print("\n" + "=" * 60)
    print("任务2.2(续): 评委裁决机制影响分析")
    print("=" * 60)
    
    print("\n【评委裁决机制说明】")
    print("从第28季开始，首先通过综合得分确定排名最后的两对选手，")
    print("然后由评委投票决定淘汰其中哪一对。")
    
    # 分析在底部两人机制下，争议选手的命运
    print("\n【争议选手在评委裁决机制下的分析】")
    print("如果应用评委裁决机制（假设评委倾向保留评委分更高者）:")
    
    for _, row in controversy_df.iterrows():
        rank_bottom = row.get('weeks_in_bottom_two_rank', 0)
        percent_bottom = row.get('weeks_in_bottom_two_percent', 0)
        
        print(f"\n  {row['celebrity']} (第{row['season']}季):")
        print(f"    排名法下进入底部两人: {rank_bottom} 周")
        print(f"    百分比法下进入底部两人: {percent_bottom} 周")
        
        if rank_bottom > 0 or percent_bottom > 0:
            print(f"    评委裁决可能在这些周次改变结果")


def generate_recommendation(comparison_df: pd.DataFrame, controversy_df: pd.DataFrame,
                           vote_estimates: pd.DataFrame) -> None:
    """
    生成最终推荐建议
    """
    print("\n" + "=" * 60)
    print("任务2.3: 方法推荐与建议")
    print("=" * 60)
    
    analyzer = MethodComparisonAnalyzer(comparison_df, controversy_df, vote_estimates)
    
    # 生成推荐
    recommendation = analyzer.generate_recommendation()
    
    # 生成完整报告
    report = analyzer.create_summary_report()
    
    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, 'recommendation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n推荐报告已保存到: {report_path}")
    
    # 打印推荐
    print(recommendation['final_recommendation'])
    
    # 详细分析
    print("\n【优缺点对比】")
    for method, pros_cons in recommendation['pros_cons'].items():
        method_name = {
            'rank_method': '排名法',
            'percent_method': '百分比法',
            'judge_tiebreaker': '评委裁决机制'
        }.get(method, method)
        
        print(f"\n{method_name}:")
        print("  优点:")
        for pro in pros_cons['pros']:
            print(f"    + {pro}")
        print("  缺点:")
        for con in pros_cons['cons']:
            print(f"    - {con}")


def create_visualizations(comparison_df: pd.DataFrame, controversy_df: pd.DataFrame,
                          vote_estimates: pd.DataFrame) -> None:
    """
    创建可视化图表
    """
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)
    
    # 图1: 两种方法一致率按赛季分布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1.1 按赛季的一致率
    ax1 = axes[0, 0]
    season_stats = comparison_df.groupby('season').agg({
        'methods_agree': ['sum', 'count']
    })
    season_stats.columns = ['agree', 'total']
    season_stats['rate'] = season_stats['agree'] / season_stats['total']
    
    colors = ['#2ecc71' if s in RANK_METHOD_SEASONS else '#3498db' 
              for s in season_stats.index]
    ax1.bar(season_stats.index, season_stats['rate'], color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=season_stats['rate'].mean(), color='red', linestyle='--', label=f'平均值: {season_stats["rate"].mean():.1%}')
    ax1.set_xlabel('赛季', fontsize=12)
    ax1.set_ylabel('方法一致率', fontsize=12)
    ax1.set_title('两种投票方法结果一致率（按赛季）', fontsize=14)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # 添加图例说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='排名法赛季'),
        Patch(facecolor='#3498db', alpha=0.7, label='百分比法赛季')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # 1.2 不一致周次分布
    ax2 = axes[0, 1]
    disagree_weeks = comparison_df[~comparison_df['methods_agree']]
    if len(disagree_weeks) > 0:
        disagree_counts = disagree_weeks.groupby('season').size()
        ax2.bar(disagree_counts.index, disagree_counts.values, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('赛季', fontsize=12)
        ax2.set_ylabel('不一致周次数', fontsize=12)
        ax2.set_title('两种方法结果不一致的周次数（按赛季）', fontsize=14)
    
    # 1.3 争议案例对比
    ax3 = axes[1, 0]
    if len(controversy_df) > 0 and 'would_be_eliminated_rank_weeks' in controversy_df.columns:
        x = range(len(controversy_df))
        width = 0.35
        
        ax3.bar([i - width/2 for i in x], controversy_df['would_be_eliminated_rank_weeks'], 
                width, label='排名法淘汰周次', color='#e74c3c', alpha=0.7)
        ax3.bar([i + width/2 for i in x], controversy_df['would_be_eliminated_percent_weeks'], 
                width, label='百分比法淘汰周次', color='#3498db', alpha=0.7)
        
        ax3.set_xlabel('争议选手', fontsize=12)
        ax3.set_ylabel('可能被淘汰的周次', fontsize=12)
        ax3.set_title('争议案例：两种方法淘汰情况对比', fontsize=14)
        ax3.set_xticks(x)
        ax3.set_xticklabels(controversy_df['celebrity'], rotation=45, ha='right')
        ax3.legend()
    
    # 1.4 争议选手评委最低分周数与最终名次
    ax4 = axes[1, 1]
    if len(controversy_df) > 0:
        scatter = ax4.scatter(controversy_df['num_lowest_judge_weeks'], 
                             controversy_df['actual_placement'],
                             s=200, c=controversy_df['season'], cmap='viridis',
                             edgecolor='black', linewidth=2)
        
        for i, row in controversy_df.iterrows():
            ax4.annotate(row['celebrity'], 
                        (row['num_lowest_judge_weeks'], row['actual_placement']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('评委最低分周数', fontsize=12)
        ax4.set_ylabel('最终名次', fontsize=12)
        ax4.set_title('争议案例：评委表现 vs 最终结果', fontsize=14)
        ax4.invert_yaxis()  # 名次越小越好
        plt.colorbar(scatter, ax=ax4, label='赛季')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'method_comparison_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {os.path.join(FIGURE_DIR, 'method_comparison_overview.png')}")
    
    # 图2: 详细争议案例轨迹图
    create_controversy_trajectory_plot(vote_estimates)


def create_controversy_trajectory_plot(vote_estimates: pd.DataFrame) -> None:
    """
    创建争议案例的详细轨迹图
    """
    from analysis.voting_methods import RankMethod, PercentMethod
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    controversy_cases = [
        ('Jerry Rice', 2),
        ('Billy Ray Cyrus', 4),
        ('Bristol Palin', 11),
        ('Bobby Bones', 27)
    ]
    
    for idx, (name, season) in enumerate(controversy_cases):
        ax = axes[idx // 2, idx % 2]
        
        # 获取该选手所在赛季的数据
        season_data = vote_estimates[vote_estimates['season'] == season].copy()
        
        # 获取选手的周次数据
        celebrity_data = season_data[
            season_data['celebrity'].str.contains(name.split()[0], case=False, na=False)
        ].sort_values('week')
        
        if len(celebrity_data) == 0:
            ax.text(0.5, 0.5, f'未找到 {name} 的数据', ha='center', va='center', transform=ax.transAxes)
            continue
        
        weeks = celebrity_data['week'].values
        
        # 计算每周的排名
        judge_ranks = []
        fan_ranks = []
        combined_ranks_rank = []
        combined_ranks_percent = []
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week]
            contestants = week_data['celebrity'].tolist()
            judge_scores = week_data['total_score'].values
            fan_votes = week_data['estimated_votes'].values
            
            # 找到该选手在本周的位置
            celeb_idx = None
            for i, c in enumerate(contestants):
                if name.split()[0].lower() in c.lower():
                    celeb_idx = i
                    break
            
            if celeb_idx is not None:
                # 排名法
                j_rank, f_rank, c_rank = RankMethod.calculate_combined_scores(judge_scores, fan_votes)
                judge_ranks.append(j_rank[celeb_idx])
                fan_ranks.append(f_rank[celeb_idx])
                combined_ranks_rank.append(c_rank[celeb_idx])
                
                # 百分比法
                j_pct, f_pct, c_pct = PercentMethod.calculate_combined_scores(judge_scores, fan_votes)
                # 将百分比转换为排名（用于可视化）
                pct_rank = len(contestants) - np.argsort(np.argsort(c_pct))[celeb_idx]
                combined_ranks_percent.append(pct_rank)
        
        # 绘制轨迹
        ax.plot(weeks, judge_ranks, 'o-', label='评委排名', color='#e74c3c', linewidth=2, markersize=8)
        ax.plot(weeks, fan_ranks, 's-', label='观众排名', color='#2ecc71', linewidth=2, markersize=8)
        ax.plot(weeks, combined_ranks_rank, '^-', label='综合排名(排名法)', color='#3498db', linewidth=2, markersize=8)
        
        ax.set_xlabel('周次', fontsize=11)
        ax.set_ylabel('排名 (1=最佳)', fontsize=11)
        ax.set_title(f'{name} (第{season}季) - 排名轨迹', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.invert_yaxis()  # 排名1在上面
        ax.grid(True, alpha=0.3)
        ax.set_xticks(weeks)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'controversy_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {os.path.join(FIGURE_DIR, 'controversy_trajectories.png')}")


def main():
    """主程序"""
    print("\n" + "=" * 60)
    print("问题二：投票方法对比分析")
    print("=" * 60)
    
    # 1. 加载数据
    vote_estimates, original_data = load_data()
    
    # 2. 两种方法对比分析
    comparison_df = analyze_method_comparison(vote_estimates, original_data)
    
    # 3. 争议案例分析
    controversy_df = analyze_controversy_cases(vote_estimates, original_data)
    
    # 4. 评委裁决机制分析
    analyze_judge_tiebreaker(comparison_df, controversy_df, vote_estimates)
    
    # 5. 生成推荐建议
    generate_recommendation(comparison_df, controversy_df, vote_estimates)
    
    # 6. 创建可视化
    create_visualizations(comparison_df, controversy_df, vote_estimates)
    
    print("\n" + "=" * 60)
    print("分析完成！所有结果已保存到 outputs 目录")
    print("=" * 60)


if __name__ == '__main__':
    main()
