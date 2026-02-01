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

# 配色方案 (from 配色.md)
COLOR_1 = '#264653'  # Dark Blue
COLOR_2 = '#2a9d8e'  # Teal
COLOR_3 = '#e9c46b'  # Yellow
COLOR_4 = '#f3a261'  # Orange
COLOR_5 = '#e86f52'  # Reddish

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
    
    # 图1: 两种方法一致率按赛季分布 (只保留上半部分)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1.1 按赛季的一致率
    ax1 = axes[0]
    season_stats = comparison_df.groupby('season').agg({
        'methods_agree': ['sum', 'count']
    })
    season_stats.columns = ['agree', 'total']
    season_stats['rate'] = season_stats['agree'] / season_stats['total']
    
    colors = [COLOR_2 if s in RANK_METHOD_SEASONS else COLOR_1 
              for s in season_stats.index]
    ax1.bar(season_stats.index, season_stats['rate'], color=colors, alpha=0.9, edgecolor='white')
    ax1.axhline(y=season_stats['rate'].mean(), color=COLOR_5, linestyle='--', label=f'Average: {season_stats["rate"].mean():.1%}')
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Agreement Rate', fontsize=12)
    ax1.set_title('Agreement Rate between Voting Methods (by Season)', fontsize=14)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # 添加图例说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_2, alpha=0.9, label='Rank Method Seasons'),
        Patch(facecolor=COLOR_1, alpha=0.9, label='Percent Method Seasons')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # 1.2 不一致周次分布
    ax2 = axes[1]
    disagree_weeks = comparison_df[~comparison_df['methods_agree']]
    if len(disagree_weeks) > 0:
        disagree_counts = disagree_weeks.groupby('season').size()
        ax2.bar(disagree_counts.index, disagree_counts.values, color=COLOR_5, alpha=0.8, edgecolor='white')
        ax2.set_xlabel('Season', fontsize=12)
        ax2.set_ylabel('Inconsistent Weeks', fontsize=12)
        ax2.set_title('Number of Inconsistent Weeks (by Season)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'method_comparison_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {os.path.join(FIGURE_DIR, 'method_comparison_overview.png')}")
    
    # 图2: 详细争议案例轨迹图
    create_controversy_trajectory_plot(vote_estimates)
    
    # 图3: 争议案例热力图对比
    create_controversy_heatmap_comparison(vote_estimates)


def create_controversy_heatmap_comparison(vote_estimates: pd.DataFrame) -> None:
    """
    创建争议案例在两种方法下的差距热力图 (体现粉丝与评委的巨大分歧)
    左图: 排名差异 (评委排名 - 观众排名) -> 正值表示观众更喜欢 (评委给的排名由于数值大反而差)
    右图: 份额差异 (观众份额 - 评委份额) -> 正值表示观众给的份额更多
    """
    from analysis.voting_methods import RankMethod
    import matplotlib.colors as mcolors
    
    controversy_cases = [
        ('Jerry Rice', 2),
        ('Billy Ray Cyrus', 4),
        ('Bristol Palin', 11),
        ('Bobby Bones', 27)
    ]
    
    # 准备数据
    rows_rank_diff = []
    rows_pct_diff = []
    max_weeks = 0
    labels = []
    
    for name, season in controversy_cases:
        labels.append(f"{name} (S{season})")
        
        # 获取该选手所在赛季的数据
        season_data = vote_estimates[vote_estimates['season'] == season].copy()
        celebrity_data = season_data[
            season_data['celebrity'].str.contains(name.split()[0], case=False, na=False)
        ].sort_values('week')
        
        weeks = celebrity_data['week'].values
        if len(weeks) > 0:
            max_weeks = max(max_weeks, max(weeks))
            
        rank_diff_dict = {}
        pct_diff_dict = {}
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week]
            contestants = week_data['celebrity'].tolist()
            judge_scores = week_data['total_score'].values
            fan_votes = week_data['estimated_votes'].values
            
            celeb_idx = None
            for i, c in enumerate(contestants):
                if name.split()[0].lower() in c.lower():
                    celeb_idx = i
                    break
            
            if celeb_idx is not None:
                # 1. 排名差异 (Rank Gap)
                # 使用 RankMethod 获取各自的排名 (1 is best)
                j_ranks, f_ranks, _ = RankMethod.calculate_combined_scores(judge_scores, fan_votes)
                my_j_rank = j_ranks[celeb_idx]
                my_f_rank = f_ranks[celeb_idx]
                
                # 差异 = 评委排名 - 观众排名
                # 例如: 评委给第10名(差), 观众给第1名(好) -> 10 - 1 = +9 (观众极度偏爱)
                # 例如: 评委给第1名(好), 观众给第10名(差) -> 1 - 10 = -9 (评委极度偏爱)
                rank_diff_dict[week] = my_j_rank - my_f_rank
                
                # 2. 份额差异 (Share Gap)
                # 归一化评委分数
                total_judge_score = np.sum(judge_scores)
                j_share = judge_scores[celeb_idx] / total_judge_score if total_judge_score > 0 else 0
                
                # 归一化观众投票
                total_fan_votes = np.sum(fan_votes)
                f_share = fan_votes[celeb_idx] / total_fan_votes if total_fan_votes > 0 else 0
                
                # 差异 = (观众份额 - 评委份额) * 100
                # 正值 = 观众份额更高
                pct_diff_dict[week] = (f_share - j_share) * 100
        
        rows_rank_diff.append(rank_diff_dict)
        rows_pct_diff.append(pct_diff_dict)
    
    # 构建DataFrame
    weeks_cols = list(range(1, max_weeks + 1))
    df_rank_diff = pd.DataFrame(rows_rank_diff, index=labels, columns=weeks_cols)
    df_pct_diff = pd.DataFrame(rows_pct_diff, index=labels, columns=weeks_cols)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # 自定义 Diverging Colormap
    # 负值 (评委偏爱) -> Red (COLOR_5)
    # 0 -> White/Light Gray
    # 正值 (观众偏爱) -> Teal (COLOR_2) / Dark Blue (COLOR_1)
    
    # 创建一个以0为中心的Colormap
    # 我们使用 matplotlib 的 LinearSegmentedColormap
    # 颜色顺序: 珊瑚红 (评委) -> 白 -> 青绿 (观众)
    cmap_colors = [COLOR_5, '#f7f7f7', COLOR_2]
    cmap = mcolors.LinearSegmentedColormap.from_list("diverging_cmap", cmap_colors)
    
    # 设置 Center=0 的 Normalize
    # 找出两个数据集中绝对值的最大值，确保0在中间
    max_val_rank = max(abs(df_rank_diff.min().min()), abs(df_rank_diff.max().max()))
    norm_rank = mcolors.TwoSlopeNorm(vmin=-max_val_rank, vcenter=0, vmax=max_val_rank)
    
    max_val_pct = max(abs(df_pct_diff.min().min()), abs(df_pct_diff.max().max()))
    norm_pct = mcolors.TwoSlopeNorm(vmin=-max_val_pct, vcenter=0, vmax=max_val_pct)
    
    # 1. Rank Gap Heatmap
    sns.heatmap(df_rank_diff, ax=axes[0], cmap=cmap, annot=True, fmt='.0f', 
                cbar=True, cbar_kws={'label': 'Rank Gap (Pos=Fan Favored)'},
                linewidths=1, linecolor='white',
                norm=norm_rank)
    axes[0].set_title('Rank Discrepancy (Judge Rank - Fan Rank)', fontsize=14, fontweight='bold', color=COLOR_1)
    axes[0].set_xlabel('Week', fontsize=12)
    axes[0].set_ylabel('')
    
    # 2. Percent Gap Heatmap
    sns.heatmap(df_pct_diff, ax=axes[1], cmap=cmap, annot=True, fmt='.1f', 
                cbar=True, cbar_kws={'label': 'Share Gap % (Pos=Fan Favored)'},
                linewidths=1, linecolor='white',
                norm=norm_pct)
    axes[1].set_title('Share Discrepancy (Fan % - Judge %)', fontsize=14, fontweight='bold', color=COLOR_1)
    axes[1].set_xlabel('Week', fontsize=12)
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, 'controversy_heatmap_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {output_path}")




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
            ax.text(0.5, 0.5, f'Data not found for {name}', ha='center', va='center', transform=ax.transAxes)
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
        
        # use COLOR_5 (Red) for Judge, COLOR_3 (Yellow) for Fan
        # COLOR_2 (Teal) for Rank Method, COLOR_1 (Dark Blue) for Percent Method
        ax.plot(weeks, judge_ranks, 'o-', label='Judge Rank', color=COLOR_5, linewidth=2, markersize=8)
        ax.plot(weeks, fan_ranks, 's-', label='Fan Rank', color=COLOR_3, linewidth=2, markersize=8)
        ax.plot(weeks, combined_ranks_rank, '^-', label='Combined (Rank)', color=COLOR_2, linewidth=2, markersize=8)
        ax.plot(weeks, combined_ranks_percent, 'x--', label='Combined (Percent)', color=COLOR_1, linewidth=2, markersize=8)
        
        ax.set_xlabel('Week', fontsize=11)
        ax.set_ylabel('Rank (1=Best)', fontsize=11)
        ax.set_title(f'{name} (Season {season})', fontsize=12)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.invert_yaxis()  # 排名1在上面
        ax.grid(True, alpha=0.3)
        ax.set_xticks(weeks)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'controversy_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {os.path.join(FIGURE_DIR, 'controversy_trajectories.png')}")
    
    # 额外绘制 Bobby Bones 单独图像
    create_bobby_bones_plot(vote_estimates)


def create_bobby_bones_plot(vote_estimates: pd.DataFrame) -> None:
    """
    单独绘制 Bobby Bones 的轨迹图
    """
    from analysis.voting_methods import RankMethod, PercentMethod
    
    fig, ax = plt.subplots(figsize=(10, 6))
    name = 'Bobby Bones'
    season = 27
    
    # 获取该选手所在赛季的数据
    season_data = vote_estimates[vote_estimates['season'] == season].copy()
    
    # 获取选手的周次数据
    celebrity_data = season_data[
        season_data['celebrity'].str.contains(name.split()[0], case=False, na=False)
    ].sort_values('week')
    
    if len(celebrity_data) > 0:
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
        
        # 绘图
        ax.plot(weeks, judge_ranks, 'o-', label='Judge Rank', color=COLOR_5, linewidth=2.5, markersize=10)
        ax.plot(weeks, fan_ranks, 's-', label='Fan Rank', color=COLOR_3, linewidth=2.5, markersize=10)
        ax.plot(weeks, combined_ranks_rank, '^-', label='Combined (Rank)', color=COLOR_2, linewidth=2.5, markersize=10)
        ax.plot(weeks, combined_ranks_percent, 'x--', label='Combined (Percent)', color=COLOR_1, linewidth=2.5, markersize=10)
        
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Rank (1=Best)', fontsize=12)
        ax.set_title(f'{name} (Season {season}) - Detailed Trajectory', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9, shadow=True)
        ax.invert_yaxis()  # 排名1在上面
        ax.grid(True, alpha=0.3)
        ax.set_xticks(weeks)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, 'bobby_bones_trajectory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {output_path}")


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
