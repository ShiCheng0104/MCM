"""
可视化模块
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from config import FIGURES_DIR, FIGURE_DPI, FIGURE_SIZE, COLORS
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def save_figure(fig, filename):
    """保存图片"""
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"图片已保存: {filepath}")


def plot_partner_effect(partner_effects, top_n=15):
    """绘制舞伴效应对比图"""
    # 筛选有足够数据的舞伴
    df = partner_effects[partner_effects['num_celebrities'] >= 2].head(top_n)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # 评委得分效应
    ax1 = axes[0]
    colors = [COLORS['judge'] if x >= 0 else COLORS['secondary'] for x in df['score_effect']]
    bars1 = ax1.barh(df['partner'], df['score_effect'], color=colors, alpha=0.8)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Score Effect (vs Overall Mean)', fontsize=11)
    ax1.set_title('Partner Effect on Judge Scores', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    
    # 观众投票效应（使用百分比）
    ax2 = axes[1]
    colors = [COLORS['fan'] if x >= 0 else COLORS['neutral'] for x in df['vote_effect_pct']]
    bars2 = ax2.barh(df['partner'], df['vote_effect_pct'], color=colors, alpha=0.8)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Vote Effect (% vs Overall Mean)', fontsize=11)
    ax2.set_title('Partner Effect on Fan Votes', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    save_figure(fig, 'partner_effect_comparison.png')
    return fig


def plot_age_effect(age_effects):
    """绘制年龄效应图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    x = np.arange(len(age_effects))
    width = 0.35
    
    # 评委得分 vs 观众投票
    ax1 = axes[0]
    ax1.bar(x - width/2, age_effects['avg_score'], width, label='Judge Score', color=COLORS['judge'], alpha=0.8)
    ax1.bar(x + width/2, age_effects['avg_votes']/10000, width, label='Votes (×10k)', color=COLORS['fan'], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(age_effects['age_group'], rotation=45, ha='right')
    ax1.set_ylabel('Score / Votes (×10k)')
    ax1.set_title('Age Group: Judge Score vs Fan Votes', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 效应对比
    ax2 = axes[1]
    ax2.bar(x - width/2, age_effects['score_effect'], width, label='Score Effect', color=COLORS['judge'], alpha=0.8)
    ax2.bar(x + width/2, age_effects['vote_effect']/1000, width, label='Vote Effect (×1k)', color=COLORS['fan'], alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(age_effects['age_group'], rotation=45, ha='right')
    ax2.set_ylabel('Effect')
    ax2.set_title('Age Group Effect (vs Overall Mean)', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    save_figure(fig, 'age_effect.png')
    return fig


def plot_industry_effect(industry_effects):
    """绘制行业效应图"""
    df = industry_effects.sort_values('avg_placement')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 平均名次
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
    bars = ax1.barh(df['industry'], df['avg_placement'], color=colors)
    ax1.set_xlabel('Average Placement (Lower is Better)')
    ax1.set_title('Average Placement by Industry', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # 平均评委得分
    ax2 = axes[0, 1]
    ax2.barh(df['industry'], df['avg_score'], color=COLORS['judge'], alpha=0.8)
    ax2.set_xlabel('Average Judge Score')
    ax2.set_title('Average Judge Score by Industry', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    
    # 平均观众投票
    ax3 = axes[1, 0]
    ax3.barh(df['industry'], df['avg_votes']/1000, color=COLORS['fan'], alpha=0.8)
    ax3.set_xlabel('Average Votes (×1000)')
    ax3.set_title('Average Fan Votes by Industry', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    
    # 效应对比散点图
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['score_effect'], df['vote_effect']/1000, 
                         s=df['num_celebrities']*20, alpha=0.7, c=df['avg_placement'],
                         cmap='RdYlGn_r')
    
    for i, row in df.iterrows():
        ax4.annotate(row['industry'], (row['score_effect'], row['vote_effect']/1000),
                    fontsize=8, ha='center', va='bottom')
    
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax4.set_xlabel('Score Effect')
    ax4.set_ylabel('Vote Effect (×1000)')
    ax4.set_title('Score vs Vote Effect by Industry', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Avg Placement')
    
    plt.tight_layout()
    save_figure(fig, 'industry_effect.png')
    return fig


def plot_feature_importance(importance_df, target_name, top_n=15):
    """绘制特征重要性图"""
    df = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))[::-1]
    bars = ax.barh(df['feature'], df['importance'], color=colors)
    
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(f'Feature Importance for {target_name}', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    
    # 添加数值标签
    for bar, val in zip(bars, df['importance']):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, f'feature_importance_{target_name}.png')
    return fig


def plot_importance_comparison(judge_importance, vote_importance, top_n=12):
    """对比评委得分和观众投票的特征重要性"""
    # 合并特征
    all_features = set(judge_importance['feature'].head(top_n).tolist() + 
                      vote_importance['feature'].head(top_n).tolist())
    
    judge_dict = judge_importance.set_index('feature')['importance'].to_dict()
    vote_dict = vote_importance.set_index('feature')['importance'].to_dict()
    
    comparison = []
    for feat in all_features:
        comparison.append({
            'feature': feat,
            'judge_importance': judge_dict.get(feat, 0),
            'vote_importance': vote_dict.get(feat, 0)
        })
    
    df = pd.DataFrame(comparison).sort_values('judge_importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y = np.arange(len(df))
    height = 0.35
    
    bars1 = ax.barh(y - height/2, df['judge_importance'], height, 
                   label='Judge Score Model', color=COLORS['judge'], alpha=0.8)
    bars2 = ax.barh(y + height/2, df['vote_importance'], height,
                   label='Fan Vote Model', color=COLORS['fan'], alpha=0.8)
    
    ax.set_yticks(y)
    ax.set_yticklabels(df['feature'])
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Feature Importance: Judge Score vs Fan Votes', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    save_figure(fig, 'feature_importance_comparison.png')
    return fig


def plot_effect_heatmap(effect_comparison):
    """绘制效应对比热力图"""
    # 重塑数据
    pivot_score = effect_comparison.pivot_table(
        values='score_effect', index='factor', columns='category', aggfunc='first'
    )
    pivot_vote = effect_comparison.pivot_table(
        values='vote_effect', index='factor', columns='category', aggfunc='first'
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 评委得分效应热力图
    sns.heatmap(pivot_score, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
               ax=axes[0], cbar_kws={'label': 'Score Effect'})
    axes[0].set_title('Judge Score Effect by Factor', fontsize=12, fontweight='bold')
    
    # 观众投票效应热力图（标准化）
    pivot_vote_norm = pivot_vote / 1000  # 转换为千
    sns.heatmap(pivot_vote_norm, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
               ax=axes[1], cbar_kws={'label': 'Vote Effect (×1000)'})
    axes[1].set_title('Fan Vote Effect by Factor', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'effect_heatmap.png')
    return fig


def plot_regression_coefficients(coef_dict, pvalue_dict, target_name):
    """绘制回归系数图"""
    # 过滤显著系数
    sig_coefs = {k: v for k, v in coef_dict.items() 
                if k != 'Intercept' and pvalue_dict.get(k, 1) < 0.05}
    
    if not sig_coefs:
        print(f"没有显著系数用于 {target_name}")
        return None
    
    df = pd.DataFrame({
        'variable': list(sig_coefs.keys()),
        'coefficient': list(sig_coefs.values()),
        'pvalue': [pvalue_dict[k] for k in sig_coefs.keys()]
    }).sort_values('coefficient')
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(df)*0.4)))
    
    colors = [COLORS['primary'] if c > 0 else COLORS['secondary'] for c in df['coefficient']]
    bars = ax.barh(df['variable'], df['coefficient'], color=colors, alpha=0.8)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Coefficient', fontsize=11)
    ax.set_title(f'Significant Regression Coefficients ({target_name})', 
                fontsize=13, fontweight='bold')
    
    # 添加显著性标记
    for bar, pval in zip(bars, df['pvalue']):
        if pval < 0.001:
            sig = '***'
        elif pval < 0.01:
            sig = '**'
        else:
            sig = '*'
        x_pos = bar.get_width() + (0.02 if bar.get_width() > 0 else -0.02)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, sig, 
               va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, f'regression_coef_{target_name}.png')
    return fig


def plot_summary_dashboard(effects_dict, importance_dict):
    """绘制综合仪表板"""
    fig = plt.figure(figsize=(16, 12))
    
    # 创建子图布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 行业效应对比
    ax1 = fig.add_subplot(gs[0, :2])
    industry = effects_dict['industry'].sort_values('avg_placement')
    x = np.arange(len(industry))
    width = 0.35
    ax1.bar(x - width/2, industry['score_effect'], width, label='Score Effect', 
           color=COLORS['judge'], alpha=0.8)
    ax1.bar(x + width/2, industry['vote_effect']/1000, width, label='Vote Effect (×1k)',
           color=COLORS['fan'], alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(industry['industry'], rotation=45, ha='right', fontsize=8)
    ax1.set_title('Industry Effect Comparison', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    
    # 2. 年龄效应
    ax2 = fig.add_subplot(gs[0, 2])
    age = effects_dict['age']
    ax2.plot(age['age_group'], age['score_effect'], 'o-', color=COLORS['judge'], 
            label='Score', linewidth=2, markersize=8)
    ax2.plot(age['age_group'], age['vote_effect']/1000, 's-', color=COLORS['fan'],
            label='Vote (×1k)', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_title('Age Effect', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # 3. Top 10 舞伴效应
    ax3 = fig.add_subplot(gs[1, :])
    partner = effects_dict['partner'].head(10)
    x = np.arange(len(partner))
    width = 0.35
    ax3.bar(x - width/2, partner['score_effect'], width, label='Score Effect',
           color=COLORS['judge'], alpha=0.8)
    ax3.bar(x + width/2, partner['vote_effect_pct'], width, label='Vote Effect (%)',
           color=COLORS['fan'], alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(partner['partner'], rotation=45, ha='right', fontsize=9)
    ax3.set_title('Top 10 Partners: Effect Comparison', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    
    # 4. 特征重要性对比
    if 'judge_score' in importance_dict and 'fan_vote' in importance_dict:
        ax4 = fig.add_subplot(gs[2, :2])
        judge_imp = importance_dict['judge_score'].head(8)
        vote_imp = importance_dict['fan_vote'].head(8)
        
        # 合并特征
        features = list(set(judge_imp['feature'].tolist() + vote_imp['feature'].tolist()))[:10]
        judge_dict = judge_imp.set_index('feature')['importance'].to_dict()
        vote_dict = vote_imp.set_index('feature')['importance'].to_dict()
        
        x = np.arange(len(features))
        width = 0.35
        ax4.bar(x - width/2, [judge_dict.get(f, 0) for f in features], width,
               label='Judge Score', color=COLORS['judge'], alpha=0.8)
        ax4.bar(x + width/2, [vote_dict.get(f, 0) for f in features], width,
               label='Fan Vote', color=COLORS['fan'], alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
        ax4.set_title('Feature Importance Comparison', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
    
    # 5. 关键发现
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    findings = [
        "Key Findings:",
        "",
        "1. Partner effect differs:",
        "   - Score: Technical skills",
        "   - Votes: Popularity boost",
        "",
        "2. Industry matters:",
        "   - Athletes: High votes, low scores",
        "   - Actors: Balanced performance",
        "",
        "3. Age effect:",
        "   - Younger: Higher fan votes",
        "   - Prime age: Best judge scores"
    ]
    ax5.text(0.1, 0.9, '\n'.join(findings), transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Question 3: Impact Analysis Dashboard', fontsize=14, fontweight='bold', y=1.02)
    save_figure(fig, 'analysis_dashboard.png')
    return fig


if __name__ == '__main__':
    print("可视化模块测试...")
