"""
可视化函数模块
配色方案来自配色.md:
    #264653 (深蓝绿)
    #2a9d8e (青绿)  
    #e9c46b (金黄)
    #f3a261 (橙色)
    #e86f52 (珊瑚红)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# 定义配色方案 (来自配色.md)
COLORS = {
    'primary': '#264653',      # 深蓝绿
    'secondary': '#2a9d8e',    # 青绿
    'accent1': '#e9c46b',      # 金黄
    'accent2': '#f3a261',      # 橙色
    'accent3': '#e86f52',      # 珊瑚红
}
COLOR_PALETTE = ['#264653', '#2a9d8e', '#e9c46b', '#f3a261', '#e86f52']


class VotePlotter:
    """投票估计可视化类"""
    
    def __init__(self, 
                 output_dir: str = 'outputs/figures',
                 style: str = 'seaborn-v0_8-whitegrid',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 150):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
            style: matplotlib样式
            figsize: 默认图形大小
            dpi: 图形DPI
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_vote_estimates_bar(self,
                                estimates: Dict,
                                season: int,
                                week: int,
                                save: bool = True) -> plt.Figure:
        """
        绘制某周投票估计柱状图
        
        Args:
            estimates: 估计结果字典
            season: 赛季
            week: 周次
            save: 是否保存图片
        
        Returns:
            matplotlib Figure
        """
        key = (season, week)
        if key not in estimates:
            print(f"未找到 Season {season} Week {week} 的数据")
            return None
        
        est = estimates[key]
        names = est['names']
        votes = np.array(est['votes'])
        scores = np.array(est['scores'])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 创建柱状图
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, votes/1000, width, label='Estimated Votes (K)', 
                       color=COLORS['primary'])
        
        # 在第二个y轴显示评委得分
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, scores, width, label='Judge Scores', 
                        color=COLORS['accent2'], alpha=0.7)
        
        ax.set_xlabel('Contestant')
        ax.set_ylabel('Votes (Thousands)')
        ax2.set_ylabel('Judge Scores')
        ax.set_title(f'Season {season} Week {week} - Vote Estimates vs Judge Scores')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'votes_s{season}_w{week}.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"图片已保存: {filepath}")
        
        return fig
    
    def plot_vote_share_radar(self, 
                               estimates: Dict,
                               season: int,
                               week: int,
                               save: bool = True) -> plt.Figure:
        """
        绘制投票份额雷达图 - 对比投票份额、评委得分份额和Google Trends份额
        
        Args:
            estimates: 估计结果字典
            season: 赛季
            week: 周次
            save: 是否保存
        
        Returns:
            matplotlib Figure
        """
        key = (season, week)
        if key not in estimates:
            print(f"未找到 Season {season} Week {week} 的数据")
            return None
        
        est = estimates[key]
        names = est['names']
        
        # 获取各项数据
        votes = np.array(est['votes'])
        scores = np.array(est['scores'])
        # 尝试获取popularity，如果可以的话
        popularities = np.array(est.get('popularities', [0]*len(names)))
        
        # 计算份额 (Normalization)
        # 投票本身已经是估算的票数或份额
        vote_shares = votes / np.sum(votes) if np.sum(votes) > 0 else np.zeros_like(votes)
        
        # 评委得分份额
        score_shares = scores / np.sum(scores) if np.sum(scores) > 0 else np.zeros_like(scores)
        
        # Google Trends份额
        pop_sum = np.sum(popularities)
        if pop_sum > 0:
            pop_shares = popularities / pop_sum
        else:
            pop_shares = np.zeros_like(popularities)
            
        # 转换为百分比用于显示
        # 但为了雷达图的可读性，保持比例即可，这里用0-1比例
        
        n = len(names)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 闭合数据点
        vote_shares_closed = np.append(vote_shares, vote_shares[0])
        score_shares_closed = np.append(score_shares, score_shares[0])
        pop_shares_closed = np.append(pop_shares, pop_shares[0])
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # 增加背景装饰环
        ax.set_facecolor('#fafafa')
        
        # 绘制评委得分 (Secondary Color)
        ax.plot(angles, score_shares_closed, 'o-', linewidth=2, 
                color=COLORS['secondary'], label='Judge Scores', markersize=6)
        ax.fill(angles, score_shares_closed, alpha=0.1, color=COLORS['secondary'])
        
        # 绘制Google Trends (Accent Color)
        if pop_sum > 0:
            ax.plot(angles, pop_shares_closed, 's-', linewidth=2, 
                    color=COLORS['accent1'], label='Google Trends', markersize=6)
            ax.fill(angles, pop_shares_closed, alpha=0.1, color=COLORS['accent1'])
            
        # 绘制预测投票 (Primary Color) - 最后绘制以突显
        ax.plot(angles, vote_shares_closed, 'D-', linewidth=3, 
                color=COLORS['primary'], label='Predicted Votes', markersize=8)
        ax.fill(angles, vote_shares_closed, alpha=0.25, color=COLORS['primary']) # 增加透明度让颜色更丰富
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(names, size=14, fontweight='bold')
        
        # 设置Y轴标签格式
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        
        ax.set_title(f'Season {season} Week {week} - Comparison of Indicators', 
                     size=16, pad=20)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=14)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'radar_s{season}_w{week}.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"图片已保存: {filepath}")
        
        return fig
        
        return fig
    
    def plot_vote_estimates_with_ci(self,
                                    estimates: Dict,
                                    uncertainty_stats: Dict,
                                    season: int,
                                    week: int,
                                    save: bool = True) -> plt.Figure:
        """
        绘制带置信区间的投票估计图
        
        Args:
            estimates: 估计结果
            uncertainty_stats: 不确定性统计
            season: 赛季
            week: 周次
            save: 是否保存
        
        Returns:
            matplotlib Figure
        """
        key = (season, week)
        if key not in estimates or key not in uncertainty_stats:
            print(f"未找到 Season {season} Week {week} 的数据")
            return None
        
        est = estimates[key]
        stats = uncertainty_stats[key]
        
        names = est['names']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(names))
        
        # 绘制均值和置信区间
        means = stats['mean'].values / 1000  # 转换为千
        lowers = stats['ci_lower'].values / 1000
        uppers = stats['ci_upper'].values / 1000
        
        yerr_lower = means - lowers
        yerr_upper = uppers - means
        
        # 根据确定性等级着色（使用配色方案）
        colors = []
        for level in stats['certainty_level']:
            if level == 'High':
                colors.append(COLORS['secondary'])  # 青绿
            elif level == 'Medium':
                colors.append(COLORS['accent1'])    # 金黄
            else:
                colors.append(COLORS['accent3'])    # 珊瑚红
        
        ax.bar(x, means, yerr=[yerr_lower, yerr_upper], capsize=5, 
               color=colors, alpha=0.8, edgecolor=COLORS['primary'])
        
        ax.set_xlabel('Contestant')
        ax.set_ylabel('Estimated Votes (Thousands)')
        ax.set_title(f'Season {season} Week {week} - Vote Estimates with 95% CI')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # 添加图例
        legend_elements = [
            Patch(facecolor=COLORS['secondary'], alpha=0.8, label='High Certainty'),
            Patch(facecolor=COLORS['accent1'], alpha=0.8, label='Medium Certainty'),
            Patch(facecolor=COLORS['accent3'], alpha=0.8, label='Low Certainty')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'votes_ci_s{season}_w{week}.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"图片已保存: {filepath}")
        
        return fig
    
    def plot_consistency_heatmap(self,
                                 consistency_results: pd.DataFrame,
                                 save: bool = True) -> plt.Figure:
        """
        绘制一致性检验热力图
        
        Args:
            consistency_results: 一致性检验结果
            save: 是否保存
        
        Returns:
            matplotlib Figure
        """
        # 创建赛季-周次的正确预测矩阵
        pivot = consistency_results.pivot_table(
            index='season', 
            columns='week', 
            values='is_correct', 
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 二分对错配色: 0 (错误) -> accent3 (红), 1 (正确) -> secondary (绿)
        from matplotlib.colors import ListedColormap
        binary_cmap = ListedColormap([COLORS['accent3'], COLORS['secondary']])
        
        # 将数据二值化 (以防只要不是1就算错的情况，不过原数据应该是bool转的float)
        # pivot 中的数据是 'is_correct' mean。如果每个单元格唯一，即0或1。
        # 如果aggfunc='mean'且有重复（不太可能），可能是小数。这里假设唯一。
        
        # Annotation 显示自定义字符而不是百分比
        # 构造 annotation 矩阵
        annot_matrix = pivot.applymap(lambda x: '✓' if x >= 0.99 else ('✗' if pd.notnull(x) else ''))
        
        sns.heatmap(pivot, annot=annot_matrix, fmt='', cmap=binary_cmap, 
                   ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Prediction (Green=Correct, Red=Incorrect)', 'ticks': [0.25, 0.75]},
                   linewidths=0.5, linecolor='white')
        
        # 修改 colorbar 的 ticks 标签
        cbar = ax.collections[0].colorbar
        cbar.ax.set_yticklabels(['Incorrect', 'Correct'])
        
        ax.set_title('Elimination Prediction Accuracy by Season and Week (Binary)')
        ax.set_xlabel('Week')
        ax.set_ylabel('Season')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'consistency_heatmap.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"图片已保存: {filepath}")
        
        return fig
    
    def plot_uncertainty_distribution(self,
                                      uncertainty_stats: Dict,
                                      save: bool = True) -> plt.Figure:
        """
        绘制不确定性分布图
        
        Args:
            uncertainty_stats: 不确定性统计字典
            save: 是否保存
        
        Returns:
            matplotlib Figure
        """
        all_stats = pd.concat(uncertainty_stats.values(), ignore_index=True)

        # 确保 cv 字段为数值并过滤掉非有限值（NaN, inf）以避免直方图报错
        all_stats['cv'] = pd.to_numeric(all_stats.get('cv', pd.Series()), errors='coerce')
        finite_mask = np.isfinite(all_stats['cv'].values)
        all_stats_cv = all_stats[finite_mask].copy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. CV分布直方图
        ax1 = axes[0]
        if not all_stats_cv.empty:
            ax1.hist(all_stats_cv['cv'].values, bins=30, edgecolor=COLORS['primary'], 
                     alpha=0.7, color=COLORS['secondary'])
            ax1.axvline(0.1, color=COLORS['accent1'], linestyle='--', linewidth=2, label='High threshold')
            ax1.axvline(0.3, color=COLORS['accent3'], linestyle='--', linewidth=2, label='Low threshold')
        else:
            ax1.text(0.5, 0.5, 'No finite CV data', transform=ax1.transAxes, ha='center')
        ax1.set_xlabel('Coefficient of Variation (CV)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Uncertainty (CV)')
        ax1.legend()

        # 2. 确定性等级饼图
        ax2 = axes[1]
        certainty_counts = all_stats['certainty_level'].dropna().value_counts()
        pie_colors = {'High': COLORS['secondary'], 'Medium': COLORS['accent1'], 'Low': COLORS['accent3']}
        if not certainty_counts.empty:
            ax2.pie(certainty_counts.values, labels=certainty_counts.index, 
                   autopct='%1.1f%%', colors=[pie_colors.get(c, 'gray') for c in certainty_counts.index],
                   wedgeprops={'edgecolor': COLORS['primary'], 'linewidth': 1.5})
        else:
            ax2.text(0.5, 0.5, 'No certainty data', transform=ax2.transAxes, ha='center')
        ax2.set_title('Certainty Level Distribution')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'uncertainty_distribution.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"图片已保存: {filepath}")
        
        return fig
    
    def plot_model_comparison(self,
                              comparison_df: pd.DataFrame,
                              save: bool = True) -> plt.Figure:
        """
        绘制模型比较图
        
        Args:
            comparison_df: 模型比较DataFrame
            save: 是否保存
        
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 三种模型对的相关性分布
        correlations = [
            ('corr_baseline_constrained', 'Baseline vs Constrained'),
            ('corr_baseline_bayesian', 'Baseline vs Bayesian'),
            ('corr_constrained_bayesian', 'Constrained vs Bayesian')
        ]
        
        for i, (col, title) in enumerate(correlations):
            ax = axes[i]
            ax.hist(comparison_df[col].dropna(), bins=20, edgecolor=COLORS['primary'], 
                    alpha=0.7, color=COLOR_PALETTE[i])
            ax.axvline(comparison_df[col].mean(), color=COLORS['accent3'], linestyle='--', 
                      linewidth=2, label=f'Mean: {comparison_df[col].mean():.3f}')
            ax.set_xlabel('Correlation')
            ax.set_ylabel('Frequency')
            ax.set_title(title)
            ax.legend()
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'model_comparison.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"图片已保存: {filepath}")
        
        return fig
    
    def plot_accuracy_summary(self,
                              summary: Dict,
                              consistency_results: pd.DataFrame = None,
                              save: bool = True) -> plt.Figure:
        """
        绘制准确率汇总图
        
        Args:
            summary: 汇总统计字典
            consistency_results: 一致性检验详细结果DataFrame (如果提供，将计算不同方法的准确率)
            save: 是否保存
        
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 总体 vs 方法准确率
        ax1 = axes[0]
        
        metrics = ['Overall Accuracy']
        values = [summary['elimination_accuracy']]
        
        # 如果有一致性结果，计算不同方法的准确率
        if consistency_results is not None and not consistency_results.empty:
            rank_res = consistency_results[consistency_results['method'] == 'rank']
            pct_res = consistency_results[consistency_results['method'] == 'percent']
            
            if not rank_res.empty:
                metrics.append('Rank Method')
                values.append(rank_res['is_correct'].mean())
            else:
                metrics.append('Rank Method')
                values.append(0)
                
            if not pct_res.empty:
                metrics.append('Percent Method')
                values.append(pct_res['is_correct'].mean())
            else:
                metrics.append('Percent Method')
                values.append(0)
        else:
            # Fallback if no detailed results
            metrics.append('Bottom-2 Accuracy')
            values.append(summary['bottom_two_accuracy'])
            
        bar_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent1']]
        
        bars = ax1.bar(metrics, values, color=bar_colors[:len(metrics)], 
                       edgecolor=COLORS['primary'], linewidth=1.5)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Prediction Accuracy by Method')
        ax1.set_ylim(0, 1.05)
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', fontsize=12, fontweight='bold',
                    color=COLORS['primary'])
        
        # 2. 按赛季准确率
        ax2 = axes[1]
        season_stats = summary['season_stats']
        ax2.plot(season_stats.index, season_stats['is_correct'], marker='o', 
                 color=COLORS['primary'], linewidth=2, markersize=6, label='Elimination')
        ax2.plot(season_stats.index, season_stats['in_bottom_two'], marker='s', 
                 color=COLORS['accent2'], linewidth=2, markersize=6, label='Bottom-2')
        
        # 使用填充色突出趋势
        ax2.fill_between(season_stats.index, season_stats['is_correct'], alpha=0.15, color=COLORS['primary'])

        # 标注排名法和百分比法的区间
        # 排名法赛季: 1-2, 28-34
        # 百分比法赛季: 3-27
        all_seasons = season_stats.index.tolist()
        min_season = min(all_seasons) if all_seasons else 1
        max_season = max(all_seasons) if all_seasons else 34
        
        # 定义区间 (为了美观，稍微向两边扩展0.5)
        # 区间1: [1, 2.5]
        ax2.axvspan(min_season - 0.5, 2.5, color=COLORS['secondary'], alpha=0.1, label='Rank Method')
        # 区间2: [2.5, 27.5]
        ax2.axvspan(2.5, 27.5, color=COLORS['accent1'], alpha=0.1, label='Percent Method')
        # 区间3: [27.5, max]
        if max_season >= 28:
            ax2.axvspan(27.5, max_season + 0.5, color=COLORS['secondary'], alpha=0.1)  # 不加label以免重复
            
        ax2.set_xlabel('Season')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Prediction Accuracy by Season')
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        ax2.set_xlim(min_season - 0.5, max_season + 0.5)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'accuracy_summary.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"图片已保存: {filepath}")
        
        return fig
    
    def plot_elimination_confusion_matrix(self, 
                                          incorrect_predictions: pd.DataFrame, 
                                          save: bool = True) -> plt.Figure:
        """
        绘制淘汰预测的错误分析混淆矩阵风格图
        展示: 实际淘汰者 vs 预测淘汰者 的排名差异
        """
        if incorrect_predictions.empty:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 散点图: x=实际淘汰者综合排名, y=预测淘汰者综合排名
        actual_ranks = incorrect_predictions['actual_combined_rank']
        pred_ranks = incorrect_predictions['pred_combined_rank']
        
        # 添加抖动以防重叠
        jitter_x = np.random.uniform(-0.2, 0.2, size=len(actual_ranks))
        jitter_y = np.random.uniform(-0.2, 0.2, size=len(pred_ranks))
        
        scatter = ax.scatter(actual_ranks + jitter_x, pred_ranks + jitter_y, 
                   alpha=0.6, c=COLORS['accent3'], s=80, edgecolors='white', linewidth=1)
        
        # 绘制对角线 (理想情况)
        max_rank = max(actual_ranks.max(), pred_ranks.max())
        ax.plot([0, max_rank], [0, max_rank], '--', color=COLORS['secondary'], label='Ideal Prediction')
        
        ax.set_xlabel('Actual Eliminated Contestant Combined Rank')
        ax.set_ylabel('Predicted Eliminated Contestant Combined Rank')
        ax.set_title('Prediction Error Analysis: Rank Comparison')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 添加注释
        ax.text(0.05, 0.95, 'Points above line:\nActual eliminated person had better rank\nthan predicted person', 
               transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'error_analysis_ranks.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"图片已保存: {filepath}")
            
        return fig

    def plot_season_vote_trajectory(self, 
                                    estimates: Dict, 
                                    season: int, 
                                    save: bool = True) -> plt.Figure:
        """
        绘制赛季投票份额轨迹图 - 展示选手在赛季中的投票趋势
        """
        # 收集该赛季的数据
        season_data = []
        weeks = []
        
        # 找出该赛季的所有周次
        season_keys = sorted([k for k in estimates.keys() if k[0] == season], key=lambda x: x[1])
        if not season_keys:
            print(f"未找到 Season {season} 的数据")
            return None
            
        # 整理数据
        contestant_shares = {} # {name: {week: share}}
        
        for s, w in season_keys:
            est = estimates[(s, w)]
            names = est['names']
            votes = np.array(est['votes'])
            shares = votes / np.sum(votes)
            
            weeks.append(w)
            
            for name, share in zip(names, shares):
                if name not in contestant_shares:
                    contestant_shares[name] = {}
                contestant_shares[name][w] = share

        if not contestant_shares:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制每个选手的轨迹
        sorted_contestants = sorted(contestant_shares.keys())
        
        # 使用配色方案轮循
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent1'], 
                  COLORS['accent2'], COLORS['accent3']]
        
        # 仅标注存活较久的选手以避免图例拥挤
        long_survivors = []
        for name, shares in contestant_shares.items():
            if len(shares) >= 3: # 至少存活3周
                long_survivors.append(name)
        
        for idx, name in enumerate(long_survivors):
            data = contestant_shares[name]
            x = sorted(data.keys())
            y = [data[k] for k in x]
            
            color = colors[idx % len(colors)]
            linestyle = '-' if idx < 5 else '--'
            marker = 'o' if idx < 5 else 's'
            
            ax.plot(x, y, marker=marker, linestyle=linestyle, linewidth=2, 
                    color=color, label=name, alpha=0.8)
            
            # 标注被淘汰点 (假设最后一周数据点为淘汰，大概率)
            # 这里简单处理，不做淘汰标记以免逻辑复杂
        
        ax.set_xlabel('Week')
        ax.set_ylabel('Vote Share')
        ax.set_title(f'Season {season} - Vote Share Trajectory (Top Survivors)')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        
        # 放置图例
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'season_{season}_trajectory.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"图片已保存: {filepath}")
            
        return fig

    def save_all_figures(self,
                        estimates: Dict,
                        consistency_results: pd.DataFrame,
                        uncertainty_stats: Dict,
                        summary: Dict,
                        sample_weeks: List[Tuple[int, int]] = None):
        """
        保存所有可视化图片
        
        Args:
            estimates: 估计结果
            consistency_results: 一致性检验结果
            uncertainty_stats: 不确定性统计
            summary: 汇总统计
            sample_weeks: 要详细展示的示例周次列表
        """
        print("\n正在生成可视化图表...")
        
        # 示例周次的详细图
        if sample_weeks:
            # 收集涉及的赛季
            sample_seasons = set()
            
            for season, week in sample_weeks:
                # 仅保留雷达图
                # self.plot_vote_estimates_bar(estimates, season, week)
                self.plot_vote_share_radar(estimates, season, week)
                # if (season, week) in uncertainty_stats:
                #    self.plot_vote_estimates_with_ci(estimates, uncertainty_stats, season, week)
            
            # 轨迹图也暂时移除，只保留特定的雷达图
            # for season in sample_seasons:
            #    self.plot_season_vote_trajectory(estimates, season)
        
        # 一致性热力图
        self.plot_consistency_heatmap(consistency_results)
        
        # 不确定性分布
        self.plot_uncertainty_distribution(uncertainty_stats)
        
        # 准确率汇总 (现在使用consistency_results来展示不同方法的对比)
        self.plot_accuracy_summary(summary, consistency_results=consistency_results)
        
        # 错误分析图 (如果有错误数据)
        incorrect_df = consistency_results[~consistency_results['is_correct']]
        if not incorrect_df.empty:
            self.plot_elimination_confusion_matrix(incorrect_df)
        
        print(f"\n所有图表已保存到: {self.output_dir}")
