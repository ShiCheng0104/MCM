"""
可视化函数模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os


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
        
        bars1 = ax.bar(x - width/2, votes/1000, width, label='Estimated Votes (K)', color='steelblue')
        
        # 在第二个y轴显示评委得分
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, scores, width, label='Judge Scores', color='coral', alpha=0.7)
        
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
        
        # 根据确定性等级着色
        colors = []
        for level in stats['certainty_level']:
            if level == 'High':
                colors.append('green')
            elif level == 'Medium':
                colors.append('orange')
            else:
                colors.append('red')
        
        ax.bar(x, means, yerr=[yerr_lower, yerr_upper], capsize=5, 
               color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Contestant')
        ax.set_ylabel('Estimated Votes (Thousands)')
        ax.set_title(f'Season {season} Week {week} - Vote Estimates with 95% CI')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='High Certainty'),
            Patch(facecolor='orange', alpha=0.7, label='Medium Certainty'),
            Patch(facecolor='red', alpha=0.7, label='Low Certainty')
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
        
        sns.heatmap(pivot, annot=True, fmt='.0%', cmap='RdYlGn', 
                   ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
        
        ax.set_title('Elimination Prediction Accuracy by Season and Week')
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
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. CV分布直方图
        ax1 = axes[0, 0]
        ax1.hist(all_stats['cv'], bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(0.1, color='green', linestyle='--', label='High threshold')
        ax1.axvline(0.3, color='red', linestyle='--', label='Low threshold')
        ax1.set_xlabel('Coefficient of Variation (CV)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Uncertainty (CV)')
        ax1.legend()
        
        # 2. 确定性等级饼图
        ax2 = axes[0, 1]
        certainty_counts = all_stats['certainty_level'].value_counts()
        colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
        ax2.pie(certainty_counts.values, labels=certainty_counts.index, 
               autopct='%1.1f%%', colors=[colors[c] for c in certainty_counts.index])
        ax2.set_title('Certainty Level Distribution')
        
        # 3. CV按赛季变化
        ax3 = axes[1, 0]
        season_cv = all_stats.groupby('season')['cv'].mean()
        ax3.plot(season_cv.index, season_cv.values, marker='o')
        ax3.set_xlabel('Season')
        ax3.set_ylabel('Mean CV')
        ax3.set_title('Uncertainty Trend by Season')
        
        # 4. CV按周次变化
        ax4 = axes[1, 1]
        week_cv = all_stats.groupby('week')['cv'].mean()
        ax4.bar(week_cv.index, week_cv.values, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Week')
        ax4.set_ylabel('Mean CV')
        ax4.set_title('Uncertainty by Week')
        
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
            ax.hist(comparison_df[col].dropna(), bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(comparison_df[col].mean(), color='red', linestyle='--', 
                      label=f'Mean: {comparison_df[col].mean():.3f}')
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
                              save: bool = True) -> plt.Figure:
        """
        绘制准确率汇总图
        
        Args:
            summary: 汇总统计字典
            save: 是否保存
        
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 总体准确率
        ax1 = axes[0]
        metrics = ['Elimination\nAccuracy', 'Bottom-2\nAccuracy']
        values = [summary['elimination_accuracy'], summary['bottom_two_accuracy']]
        colors = ['steelblue', 'coral']
        
        bars = ax1.bar(metrics, values, color=colors, edgecolor='black')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Overall Prediction Accuracy')
        ax1.set_ylim(0, 1)
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', fontsize=12, fontweight='bold')
        
        # 2. 按赛季准确率
        ax2 = axes[1]
        season_stats = summary['season_stats']
        ax2.plot(season_stats.index, season_stats['is_correct'], marker='o', label='Elimination')
        ax2.plot(season_stats.index, season_stats['in_bottom_two'], marker='s', label='Bottom-2')
        ax2.set_xlabel('Season')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Prediction Accuracy by Season')
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'accuracy_summary.png')
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
            for season, week in sample_weeks:
                self.plot_vote_estimates_bar(estimates, season, week)
                if (season, week) in uncertainty_stats:
                    self.plot_vote_estimates_with_ci(estimates, uncertainty_stats, season, week)
        
        # 一致性热力图
        self.plot_consistency_heatmap(consistency_results)
        
        # 不确定性分布
        self.plot_uncertainty_distribution(uncertainty_stats)
        
        # 准确率汇总
        self.plot_accuracy_summary(summary)
        
        print(f"\n所有图表已保存到: {self.output_dir}")
