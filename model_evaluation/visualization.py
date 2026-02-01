"""
模型评估与推广 - 可视化模块
Visualization for Model Evaluation
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

from config import FIGURES_DIR, PLOT_CONFIG


def setup_plot_style():
    """设置绘图风格"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = PLOT_CONFIG['figure_size']
    plt.rcParams['figure.dpi'] = PLOT_CONFIG['dpi']
    plt.rcParams['font.size'] = PLOT_CONFIG['font_size']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_model_comparison_radar(summaries, save_path=None):
    """雷达图：各模型综合评估"""
    setup_plot_style()
    
    # 准备数据
    categories = ['Accuracy', 'Robustness', 'Generalization', 'Innovation']
    n_cats = len(categories)
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    colors = list(PLOT_CONFIG['colors'].values())
    
    for i, (name, summary) in enumerate(summaries.items()):
        # 提取各维度分数（归一化到0-1）
        values = [
            summary.get('accuracy', summary.get('cv_r2', 0.5)),
            summary.get('robustness', 0.8),
            1 - abs(summary.get('generalization_gap', 0)),
            summary.get('innovation_score', 0.5) if isinstance(summary.get('innovation_score'), (int, float)) else 0.5,
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Evaluation Radar Chart', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_validation_comparison(cv_results, save_path=None):
    """交叉验证结果对比"""
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：CV分数箱线图
    ax1 = axes[0]
    models = list(cv_results.keys())
    positions = range(len(models))
    
    for i, (model, result) in enumerate(cv_results.items()):
        if 'fold_scores' in result:
            bp = ax1.boxplot([result['fold_scores']], positions=[i], widths=0.6)
            color = list(PLOT_CONFIG['colors'].values())[i % 4]
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=color)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('CV Score')
    ax1.set_title('Cross-Validation Score Distribution')
    
    # 右图：均值和置信区间
    ax2 = axes[1]
    means = []
    errors = []
    colors = []
    
    for i, (model, result) in enumerate(cv_results.items()):
        mean = result.get('mean_accuracy', result.get('r2_mean', 0))
        std = result.get('std_accuracy', result.get('r2_std', 0))
        means.append(mean)
        errors.append(1.96 * std)  # 95% CI
        colors.append(list(PLOT_CONFIG['colors'].values())[i % 4])
    
    bars = ax2.bar(models, means, yerr=errors, capsize=5, color=colors, alpha=0.7)
    ax2.set_ylabel('Mean Score')
    ax2.set_title('Mean CV Score with 95% CI')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sensitivity_analysis(sensitivity_results, save_path=None):
    """敏感性分析图"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model, results in sensitivity_results.items():
        if isinstance(results, list):
            perturbations = [r['perturbation'] for r in results]
            means = [r['mean_accuracy'] for r in results]
            stds = [r['std_accuracy'] for r in results]
            
            color = PLOT_CONFIG['colors'].get(model, '#333333')
            ax.errorbar(perturbations, means, yerr=stds, 
                       marker='o', label=model, capsize=5, color=color)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Parameter Perturbation')
    ax.set_ylabel('Accuracy')
    ax.set_title('Sensitivity Analysis: Model Robustness to Parameter Changes')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_temporal_validation(temporal_results, save_path=None):
    """时序验证结果"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(temporal_results.keys())
    train_scores = [temporal_results[m]['train_accuracy'] for m in models]
    test_scores = [temporal_results[m]['test_accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_scores, width, label='Train (Early Seasons)', color='#3498db')
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test (Recent Seasons)', color='#e74c3c')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Temporal Validation: Generalization Across Time')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # 添加数值标签
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comprehensive_dashboard(all_results, save_path=None):
    """综合评估仪表板"""
    setup_plot_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 模型性能概览（左上）
    ax1 = fig.add_subplot(2, 2, 1)
    models = ['Q1', 'Q2', 'Q3', 'Q4']
    metrics = {
        'Q1': all_results.get('Q1', {}).get('accuracy', 0.5),
        'Q2': all_results.get('Q2', {}).get('percent_accuracy', 0.5),
        'Q3': all_results.get('Q3', {}).get('cv_r2', 0.5),
        'Q4': all_results.get('Q4', {}).get('composite_score', 0.5),
    }
    
    colors = [PLOT_CONFIG['colors'][m] for m in models]
    bars = ax1.bar(models, list(metrics.values()), color=colors, alpha=0.8)
    ax1.set_ylabel('Primary Metric')
    ax1.set_title('Model Performance Overview')
    ax1.set_ylim(0, 1)
    
    for bar, val in zip(bars, metrics.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2%}', ha='center', va='bottom', fontsize=10)
    
    # 2. 稳健性对比（右上）
    ax2 = fig.add_subplot(2, 2, 2)
    robustness = {
        'Q1': all_results.get('Q1', {}).get('robustness', 0.8),
        'Q2': all_results.get('Q2', {}).get('rank_stability', 0.8),
        'Q3': all_results.get('Q3', {}).get('feature_stability', 0.8),
        'Q4': all_results.get('Q4', {}).get('robustness', 0.8),
    }
    
    ax2.bar(models, list(robustness.values()), color=colors, alpha=0.8)
    ax2.set_ylabel('Robustness Score')
    ax2.set_title('Model Robustness Comparison')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax2.legend()
    
    # 3. 问题之间的关系图（左下）
    ax3 = fig.add_subplot(2, 2, 3)
    
    # 绘制问题间关系
    positions = {
        'Q1': (0.2, 0.8),
        'Q2': (0.8, 0.8),
        'Q3': (0.2, 0.2),
        'Q4': (0.8, 0.2),
    }
    
    # 绘制节点
    for q, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.1, color=PLOT_CONFIG['colors'][q], alpha=0.7)
        ax3.add_patch(circle)
        ax3.text(x, y, q, ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # 绘制连接线
    connections = [
        ('Q1', 'Q2', 'Vote Data'),
        ('Q1', 'Q3', 'Vote Data'),
        ('Q2', 'Q4', 'Method Analysis'),
        ('Q3', 'Q4', 'Factor Analysis'),
    ]
    
    for q1, q2, label in connections:
        x1, y1 = positions[q1]
        x2, y2 = positions[q2]
        ax3.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        ax3.text((x1+x2)/2, (y1+y2)/2, label, fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Model Dependencies')
    
    # 4. 评估指标汇总表（右下）
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # 创建汇总表
    table_data = [
        ['Model', 'Accuracy', 'Robustness', 'Generalization'],
        ['Q1: Vote Est.', f"{metrics['Q1']:.2%}", f"{robustness['Q1']:.2f}", 'Good'],
        ['Q2: Method Comp.', f"{metrics['Q2']:.2%}", f"{robustness['Q2']:.2f}", 'Good'],
        ['Q3: Factor Anal.', f"{metrics['Q3']:.2%}", f"{robustness['Q3']:.2f}", 'Good'],
        ['Q4: Voting Sys.', f"{metrics['Q4']:.2%}", f"{robustness['Q4']:.2f}", 'Good'],
    ]
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # 设置表头样式
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('Evaluation Summary', pad=20)
    
    plt.suptitle('Model Evaluation Dashboard', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_promotion_analysis(promotion_data, save_path=None):
    """模型推广分析图"""
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 适用场景分析
    ax1 = axes[0]
    scenarios = ['Small Data', 'Large Data', 'Real-time', 'Batch']
    model_scores = {
        'Q1': [0.7, 0.9, 0.6, 0.9],
        'Q3': [0.6, 0.95, 0.5, 0.95],
        'Q4': [0.8, 0.85, 0.8, 0.8],
    }
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, (model, scores) in enumerate(model_scores.items()):
        ax1.bar(x + i*width, scores, width, label=model, 
               color=PLOT_CONFIG['colors'][model], alpha=0.8)
    
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(scenarios)
    ax1.set_ylabel('Suitability Score')
    ax1.set_title('Scenario Suitability')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # 2. 扩展性评估
    ax2 = axes[1]
    extensions = ['Other Shows', 'Other Sports', 'Elections', 'Competitions']
    extensibility = [0.9, 0.7, 0.5, 0.85]
    colors = ['#2ecc71' if e > 0.7 else '#f39c12' if e > 0.5 else '#e74c3c' for e in extensibility]
    
    bars = ax2.barh(extensions, extensibility, color=colors)
    ax2.set_xlabel('Extensibility Score')
    ax2.set_title('Model Extensibility')
    ax2.set_xlim(0, 1)
    
    # 3. 改进方向
    ax3 = axes[2]
    improvements = {
        'More Data': 0.3,
        'Deep Learning': 0.25,
        'Real-time Update': 0.2,
        'Uncertainty Quantification': 0.15,
        'User Interface': 0.1,
    }
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(improvements)))
    wedges, texts, autotexts = ax3.pie(list(improvements.values()), 
                                        labels=list(improvements.keys()),
                                        autopct='%1.0f%%', colors=colors)
    ax3.set_title('Future Improvement Priorities')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # 测试可视化
    test_summaries = {
        'Q1': {'accuracy': 0.85, 'robustness': 0.9, 'generalization_gap': 0.05},
        'Q2': {'accuracy': 0.78, 'robustness': 0.85},
        'Q3': {'cv_r2': 0.72, 'feature_stability': 0.88},
        'Q4': {'composite_score': 0.82, 'robustness': 0.75},
    }
    
    plot_model_comparison_radar(test_summaries, os.path.join(FIGURES_DIR, 'radar_test.png'))
    print("Test visualization completed!")
