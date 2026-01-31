# -*- coding: utf-8 -*-
"""
详细分析模块
1. 周次级别对比表
2. 评委裁决模拟
3. 可视化轨迹图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple
from .voting_methods import RankMethod, PercentMethod

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class DetailedWeeklyAnalysis:
    """
    周次级别详细对比分析
    """
    
    def __init__(self, vote_estimates_df: pd.DataFrame, original_data_df: pd.DataFrame):
        self.vote_estimates = vote_estimates_df
        self.original_data = original_data_df
    
    def get_weekly_comparison_table(self, celebrity_name: str, season: int) -> pd.DataFrame:
        """
        生成某选手在特定赛季的每周详细对比表
        
        展示：
        - 每周的评委得分和排名
        - 预测的观众投票和排名
        - 两种方法的综合得分
        - 两种方法预测的淘汰人选
        - 实际淘汰人选
        - 该选手是否在底部二人中
        """
        season_data = self.vote_estimates[self.vote_estimates['season'] == season].copy()
        
        results = []
        
        for week in sorted(season_data['week'].unique()):
            week_data = season_data[season_data['week'] == week].copy()
            
            if len(week_data) == 0:
                continue
            
            contestants = week_data['celebrity'].tolist()
            judge_scores = week_data['total_score'].values
            fan_votes = week_data['estimated_votes'].values
            
            # 检查目标选手是否在本周
            target_idx = None
            for i, c in enumerate(contestants):
                if celebrity_name.lower() in c.lower():
                    target_idx = i
                    break
            
            if target_idx is None:
                continue  # 选手已被淘汰
            
            # 计算排名法结果
            judge_ranks_r, fan_ranks_r, combined_ranks = RankMethod.calculate_combined_scores(
                judge_scores, fan_votes
            )
            rank_eliminated, _ = RankMethod.determine_elimination(judge_scores, fan_votes, contestants)
            rank_bottom_two, _ = RankMethod.get_bottom_two(judge_scores, fan_votes, contestants)
            
            # 计算百分比法结果
            judge_pcts, fan_pcts, combined_pcts = PercentMethod.calculate_combined_scores(
                judge_scores, fan_votes
            )
            pct_eliminated, _ = PercentMethod.determine_elimination(judge_scores, fan_votes, contestants)
            pct_bottom_two, _ = PercentMethod.get_bottom_two(judge_scores, fan_votes, contestants)
            
            # 获取实际淘汰
            actual_eliminated_row = week_data[week_data['is_eliminated'] == True]
            actual_eliminated = actual_eliminated_row['celebrity'].values[0] if len(actual_eliminated_row) > 0 else None
            
            # 该选手的详细数据
            target_judge_score = judge_scores[target_idx]
            target_fan_votes = fan_votes[target_idx]
            target_judge_rank = judge_ranks_r[target_idx]
            target_fan_rank = fan_ranks_r[target_idx]
            target_combined_rank = combined_ranks[target_idx]
            target_judge_pct = judge_pcts[target_idx]
            target_fan_pct = fan_pcts[target_idx]
            target_combined_pct = combined_pcts[target_idx]
            
            # 判断该选手是否会被各方法淘汰
            would_be_eliminated_rank = celebrity_name.lower() in rank_eliminated.lower()
            would_be_eliminated_pct = celebrity_name.lower() in pct_eliminated.lower()
            
            # 判断该选手是否在底部二人中
            in_bottom_two_rank = any(celebrity_name.lower() in b.lower() for b in rank_bottom_two)
            in_bottom_two_pct = any(celebrity_name.lower() in b.lower() for b in pct_bottom_two)
            
            results.append({
                'week': week,
                'num_contestants': len(contestants),
                'judge_score': target_judge_score,
                'judge_rank': int(target_judge_rank),
                'fan_votes': int(target_fan_votes),
                'fan_rank': int(target_fan_rank),
                'combined_rank_score': target_combined_rank,
                'rank_position': int(np.sum(combined_ranks <= target_combined_rank)),  # 综合排名位置
                'judge_pct': round(target_judge_pct * 100, 2),
                'fan_pct': round(target_fan_pct * 100, 2),
                'combined_pct': round(target_combined_pct * 100, 2),
                'pct_position': int(np.sum(combined_pcts >= target_combined_pct)),  # 百分比排名位置
                'rank_method_eliminates': rank_eliminated,
                'pct_method_eliminates': pct_eliminated,
                'actual_eliminated': actual_eliminated if actual_eliminated else '(无淘汰)',
                'would_be_eliminated_rank': would_be_eliminated_rank,
                'would_be_eliminated_pct': would_be_eliminated_pct,
                'in_bottom_two_rank': in_bottom_two_rank,
                'in_bottom_two_pct': in_bottom_two_pct,
                'rank_bottom_two': ', '.join(rank_bottom_two),
                'pct_bottom_two': ', '.join(pct_bottom_two),
                'methods_differ': rank_eliminated != pct_eliminated
            })
        
        return pd.DataFrame(results)


class JudgeTiebreakerSimulator:
    """
    评委裁决模拟器
    
    模拟从第28季开始的规则：底部两名由评委投票决定淘汰谁
    假设：评委倾向于保留技术更好（评委得分更高）的选手
    """
    
    def __init__(self, vote_estimates_df: pd.DataFrame, original_data_df: pd.DataFrame):
        self.vote_estimates = vote_estimates_df
        self.original_data = original_data_df
    
    def simulate_judge_decision(self, bottom_two: List[str], 
                                 judge_scores: np.ndarray,
                                 contestants: List[str],
                                 bias_strength: float = 0.9) -> Tuple[str, str, float]:
        """
        模拟评委从底部两人中选择淘汰对象
        
        Parameters:
        -----------
        bottom_two : List[str]
            底部两名选手
        judge_scores : np.ndarray
            所有选手的评委得分
        contestants : List[str]
            所有选手名单
        bias_strength : float
            评委偏向技术的强度 (0-1)，1表示完全按技术淘汰低分者
            
        Returns:
        --------
        eliminated : str
            被淘汰的选手
        saved : str
            被保留的选手
        confidence : float
            评委决策的置信度
        """
        # 获取底部两人的评委得分
        idx1 = None
        idx2 = None
        for i, c in enumerate(contestants):
            if bottom_two[0].lower() in c.lower():
                idx1 = i
            if bottom_two[1].lower() in c.lower():
                idx2 = i
        
        if idx1 is None or idx2 is None:
            return bottom_two[0], bottom_two[1], 0.5
        
        score1 = judge_scores[idx1]
        score2 = judge_scores[idx2]
        
        # 评委倾向于保留得分更高的选手
        if score1 > score2:
            # 选手1得分更高，更可能被保留
            prob_eliminate_2 = 0.5 + (score1 - score2) / (score1 + score2) * bias_strength
            if np.random.random() < prob_eliminate_2:
                return bottom_two[1], bottom_two[0], prob_eliminate_2
            else:
                return bottom_two[0], bottom_two[1], 1 - prob_eliminate_2
        elif score2 > score1:
            # 选手2得分更高，更可能被保留
            prob_eliminate_1 = 0.5 + (score2 - score1) / (score1 + score2) * bias_strength
            if np.random.random() < prob_eliminate_1:
                return bottom_two[0], bottom_two[1], prob_eliminate_1
            else:
                return bottom_two[1], bottom_two[0], 1 - prob_eliminate_1
        else:
            # 得分相同，随机选择
            if np.random.random() < 0.5:
                return bottom_two[0], bottom_two[1], 0.5
            else:
                return bottom_two[1], bottom_two[0], 0.5
    
    def simulate_season_with_tiebreaker(self, celebrity_name: str, season: int,
                                         method: str = 'rank',
                                         num_simulations: int = 1000) -> Dict:
        """
        模拟整个赛季的评委裁决机制对特定选手的影响
        
        Parameters:
        -----------
        celebrity_name : str
            目标选手
        season : int
            赛季
        method : str
            使用的综合方法 ('rank' 或 'percent')
        num_simulations : int
            模拟次数
        """
        season_data = self.vote_estimates[self.vote_estimates['season'] == season].copy()
        
        # 记录每周该选手被淘汰的概率
        weekly_elimination_probs = []
        
        for week in sorted(season_data['week'].unique()):
            week_data = season_data[season_data['week'] == week].copy()
            
            contestants = week_data['celebrity'].tolist()
            judge_scores = week_data['total_score'].values
            fan_votes = week_data['estimated_votes'].values
            
            # 检查目标选手是否在本周
            target_in_week = any(celebrity_name.lower() in c.lower() for c in contestants)
            if not target_in_week:
                continue
            
            # 获取底部两人
            if method == 'rank':
                bottom_two, _ = RankMethod.get_bottom_two(judge_scores, fan_votes, contestants)
            else:
                bottom_two, _ = PercentMethod.get_bottom_two(judge_scores, fan_votes, contestants)
            
            # 检查目标选手是否在底部两人中
            target_in_bottom = any(celebrity_name.lower() in b.lower() for b in bottom_two)
            
            if target_in_bottom:
                # 模拟评委裁决
                elimination_count = 0
                for _ in range(num_simulations):
                    eliminated, saved, _ = self.simulate_judge_decision(
                        bottom_two, judge_scores, contestants
                    )
                    if celebrity_name.lower() in eliminated.lower():
                        elimination_count += 1
                
                elimination_prob = elimination_count / num_simulations
            else:
                elimination_prob = 0.0
            
            # 获取实际淘汰
            actual_eliminated_row = week_data[week_data['is_eliminated'] == True]
            actual_eliminated = actual_eliminated_row['celebrity'].values[0] if len(actual_eliminated_row) > 0 else None
            
            weekly_elimination_probs.append({
                'week': week,
                'in_bottom_two': target_in_bottom,
                'bottom_two': bottom_two,
                'elimination_probability': elimination_prob,
                'would_survive_probability': 1 - elimination_prob,
                'actual_eliminated': actual_eliminated
            })
        
        # 计算累计生存概率
        survival_prob = 1.0
        for record in weekly_elimination_probs:
            if record['in_bottom_two']:
                survival_prob *= record['would_survive_probability']
        
        return {
            'celebrity': celebrity_name,
            'season': season,
            'method': method,
            'weekly_details': weekly_elimination_probs,
            'weeks_in_bottom_two': sum(1 for r in weekly_elimination_probs if r['in_bottom_two']),
            'cumulative_survival_probability': survival_prob,
            'expected_to_survive': survival_prob > 0.5
        }


class ControversyTrajectoryVisualizer:
    """
    争议选手轨迹可视化
    """
    
    def __init__(self, vote_estimates_df: pd.DataFrame, original_data_df: pd.DataFrame):
        self.vote_estimates = vote_estimates_df
        self.original_data = original_data_df
        self.detailed_analyzer = DetailedWeeklyAnalysis(vote_estimates_df, original_data_df)
    
    def plot_single_trajectory(self, celebrity_name: str, season: int, 
                                save_path: str = None) -> plt.Figure:
        """
        绘制单个选手的详细轨迹图
        
        包含：
        - 评委得分排名变化
        - 观众投票排名变化
        - 两种方法下的综合排名变化
        - 标记进入底部二人的周次
        - 标记两种方法预测淘汰的周次
        """
        weekly_data = self.detailed_analyzer.get_weekly_comparison_table(celebrity_name, season)
        
        if len(weekly_data) == 0:
            print(f"未找到 {celebrity_name} 在第{season}季的数据")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{celebrity_name} - Season {season} Trajectory Analysis', fontsize=14, fontweight='bold')
        
        weeks = weekly_data['week'].values
        
        # 图1: 评委得分和观众投票排名
        ax1 = axes[0, 0]
        ax1.plot(weeks, weekly_data['judge_rank'], 'b-o', label='Judge Rank', linewidth=2, markersize=8)
        ax1.plot(weeks, weekly_data['fan_rank'], 'r-s', label='Fan Vote Rank', linewidth=2, markersize=8)
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Rank (1=Best)')
        ax1.set_title('Judge Score Rank vs Fan Vote Rank')
        ax1.legend()
        ax1.invert_yaxis()  # 排名1在上面
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(weeks)
        
        # 图2: 两种方法下的综合位置
        ax2 = axes[0, 1]
        ax2.plot(weeks, weekly_data['rank_position'], 'g-o', label='Rank Method Position', linewidth=2, markersize=8)
        ax2.plot(weeks, weekly_data['pct_position'], 'm-s', label='Percent Method Position', linewidth=2, markersize=8)
        
        # 标记进入底部二人的周次
        bottom_weeks_rank = weekly_data[weekly_data['in_bottom_two_rank'] == True]['week'].values
        bottom_weeks_pct = weekly_data[weekly_data['in_bottom_two_pct'] == True]['week'].values
        
        for w in bottom_weeks_rank:
            row = weekly_data[weekly_data['week'] == w].iloc[0]
            ax2.scatter([w], [row['rank_position']], c='green', s=200, marker='*', zorder=5)
        for w in bottom_weeks_pct:
            row = weekly_data[weekly_data['week'] == w].iloc[0]
            ax2.scatter([w], [row['pct_position']], c='purple', s=200, marker='*', zorder=5)
        
        ax2.set_xlabel('Week')
        ax2.set_ylabel('Position (Higher = Worse)')
        ax2.set_title('Combined Score Position (★ = In Bottom Two)')
        ax2.legend()
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(weeks)
        
        # 图3: 评委百分比 vs 观众百分比
        ax3 = axes[1, 0]
        ax3.bar(weeks - 0.15, weekly_data['judge_pct'], 0.3, label='Judge %', color='blue', alpha=0.7)
        ax3.bar(weeks + 0.15, weekly_data['fan_pct'], 0.3, label='Fan %', color='red', alpha=0.7)
        ax3.set_xlabel('Week')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Judge Score % vs Fan Vote %')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xticks(weeks)
        
        # 图4: 是否会被淘汰的标记
        ax4 = axes[1, 1]
        
        # 创建标记数据
        would_elim_rank = weekly_data['would_be_eliminated_rank'].astype(int).values
        would_elim_pct = weekly_data['would_be_eliminated_pct'].astype(int).values
        in_bottom_rank = weekly_data['in_bottom_two_rank'].astype(int).values
        in_bottom_pct = weekly_data['in_bottom_two_pct'].astype(int).values
        
        width = 0.2
        x = np.arange(len(weeks))
        
        ax4.bar(x - 1.5*width, would_elim_rank, width, label='Would be eliminated (Rank)', color='darkgreen', alpha=0.8)
        ax4.bar(x - 0.5*width, would_elim_pct, width, label='Would be eliminated (Percent)', color='darkmagenta', alpha=0.8)
        ax4.bar(x + 0.5*width, in_bottom_rank, width, label='In Bottom 2 (Rank)', color='lightgreen', alpha=0.8)
        ax4.bar(x + 1.5*width, in_bottom_pct, width, label='In Bottom 2 (Percent)', color='plum', alpha=0.8)
        
        ax4.set_xlabel('Week')
        ax4.set_ylabel('Yes (1) / No (0)')
        ax4.set_title('Risk Assessment by Week')
        ax4.set_xticks(x)
        ax4.set_xticklabels(weeks)
        ax4.legend(loc='upper right', fontsize=8)
        ax4.set_ylim(0, 1.5)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        return fig
    
    def plot_all_controversies(self, output_dir: str):
        """
        绘制所有争议案例的轨迹图
        """
        cases = [
            ('Jerry Rice', 2),
            ('Billy Ray Cyrus', 4),
            ('Bristol Palin', 11),
            ('Bobby Bones', 27)
        ]
        
        for name, season in cases:
            filename = f"{name.replace(' ', '_')}_S{season}_trajectory.png"
            save_path = f"{output_dir}/{filename}"
            self.plot_single_trajectory(name, season, save_path)
    
    def plot_comparison_summary(self, save_path: str = None) -> plt.Figure:
        """
        绘制所有争议案例的汇总对比图
        """
        cases = [
            ('Jerry Rice', 2, 2),      # (name, season, actual_placement)
            ('Billy Ray Cyrus', 4, 5),
            ('Bristol Palin', 11, 3),
            ('Bobby Bones', 27, 1)
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Controversy Cases: Method Comparison Summary', fontsize=14, fontweight='bold')
        
        for idx, (name, season, placement) in enumerate(cases):
            ax = axes[idx // 2, idx % 2]
            
            weekly_data = self.detailed_analyzer.get_weekly_comparison_table(name, season)
            
            if len(weekly_data) == 0:
                ax.text(0.5, 0.5, f'No data for {name}', ha='center', va='center')
                continue
            
            weeks = weekly_data['week'].values
            
            # 绘制两种方法的综合位置
            ax.plot(weeks, weekly_data['rank_position'], 'g-o', label='Rank Method', linewidth=2, markersize=6)
            ax.plot(weeks, weekly_data['pct_position'], 'm-s', label='Percent Method', linewidth=2, markersize=6)
            
            # 标记会被淘汰的周次
            for w in weekly_data[weekly_data['would_be_eliminated_rank'] == True]['week'].values:
                row = weekly_data[weekly_data['week'] == w].iloc[0]
                ax.scatter([w], [row['rank_position']], c='red', s=150, marker='X', zorder=5, edgecolors='black')
            for w in weekly_data[weekly_data['would_be_eliminated_pct'] == True]['week'].values:
                row = weekly_data[weekly_data['week'] == w].iloc[0]
                ax.scatter([w], [row['pct_position']], c='orange', s=150, marker='X', zorder=5, edgecolors='black')
            
            # 添加底线（表示被淘汰位置）
            max_contestants = weekly_data['num_contestants'].max()
            ax.axhline(y=max_contestants, color='red', linestyle='--', alpha=0.5, label='Elimination Zone')
            
            placement_text = {1: 'Winner', 2: 'Runner-up', 3: '3rd Place', 5: '5th Place'}
            ax.set_title(f'{name} (S{season}) - Actual: {placement_text.get(placement, f"{placement}th")}')
            ax.set_xlabel('Week')
            ax.set_ylabel('Position')
            ax.legend(loc='upper left', fontsize=8)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(weeks)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"汇总图表已保存: {save_path}")
        
        return fig


def run_detailed_analysis(vote_estimates_path: str, original_data_path: str, output_dir: str):
    """
    运行完整的详细分析
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    
    # 加载数据
    vote_estimates = pd.read_csv(vote_estimates_path)
    original_data = pd.read_csv(original_data_path)
    
    print("=" * 70)
    print("开始详细分析...")
    print("=" * 70)
    
    # 1. 生成周次级别对比表
    print("\n1. 生成周次级别详细对比表...")
    detailed_analyzer = DetailedWeeklyAnalysis(vote_estimates, original_data)
    
    cases = [
        ('Jerry Rice', 2),
        ('Billy Ray Cyrus', 4),
        ('Bristol Palin', 11),
        ('Bobby Bones', 27)
    ]
    
    all_weekly_tables = []
    for name, season in cases:
        table = detailed_analyzer.get_weekly_comparison_table(name, season)
        table['celebrity'] = name
        table['season'] = season
        all_weekly_tables.append(table)
        
        print(f"\n【{name} - 第{season}季 详细周次对比】")
        print(table[['week', 'judge_rank', 'fan_rank', 'rank_position', 'pct_position', 
                     'would_be_eliminated_rank', 'would_be_eliminated_pct', 
                     'in_bottom_two_rank', 'in_bottom_two_pct']].to_string(index=False))
    
    # 保存周次对比表
    combined_weekly = pd.concat(all_weekly_tables, ignore_index=True)
    combined_weekly.to_csv(f"{output_dir}/detailed_weekly_comparison.csv", index=False)
    print(f"\n周次对比表已保存: {output_dir}/detailed_weekly_comparison.csv")
    
    # 2. 评委裁决模拟
    print("\n" + "=" * 70)
    print("2. 评委裁决模拟分析...")
    print("=" * 70)
    
    simulator = JudgeTiebreakerSimulator(vote_estimates, original_data)
    
    tiebreaker_results = []
    for name, season in cases:
        for method in ['rank', 'percent']:
            result = simulator.simulate_season_with_tiebreaker(name, season, method, num_simulations=1000)
            
            print(f"\n【{name} - S{season} - {method.upper()}法 评委裁决模拟】")
            print(f"  进入底部二人的周数: {result['weeks_in_bottom_two']}")
            print(f"  累计生存概率: {result['cumulative_survival_probability']:.1%}")
            print(f"  预期能存活: {'是' if result['expected_to_survive'] else '否'}")
            
            for detail in result['weekly_details']:
                if detail['in_bottom_two']:
                    print(f"    第{detail['week']}周: 底部二人={detail['bottom_two']}, "
                          f"被淘汰概率={detail['elimination_probability']:.1%}")
            
            tiebreaker_results.append({
                'celebrity': name,
                'season': season,
                'method': method,
                'weeks_in_bottom_two': result['weeks_in_bottom_two'],
                'cumulative_survival_prob': result['cumulative_survival_probability'],
                'expected_to_survive': result['expected_to_survive']
            })
    
    # 保存评委裁决模拟结果
    tiebreaker_df = pd.DataFrame(tiebreaker_results)
    tiebreaker_df.to_csv(f"{output_dir}/judge_tiebreaker_simulation.csv", index=False)
    print(f"\n评委裁决模拟结果已保存: {output_dir}/judge_tiebreaker_simulation.csv")
    
    # 3. 生成可视化轨迹图
    print("\n" + "=" * 70)
    print("3. 生成可视化轨迹图...")
    print("=" * 70)
    
    visualizer = ControversyTrajectoryVisualizer(vote_estimates, original_data)
    
    # 单独的轨迹图
    visualizer.plot_all_controversies(f"{output_dir}/figures")
    
    # 汇总对比图
    visualizer.plot_comparison_summary(f"{output_dir}/figures/controversy_summary.png")
    
    print("\n" + "=" * 70)
    print("详细分析完成！")
    print("=" * 70)
    
    return combined_weekly, tiebreaker_df


if __name__ == "__main__":
    run_detailed_analysis(
        vote_estimates_path="vote_estimates.csv",
        original_data_path="../2026_MCM_Problem_C_Data.csv",
        output_dir="outputs"
    )
