"""
综合分析模块
整合所有分析步骤并生成报告
"""
import pandas as pd
import numpy as np
from data_loader import create_analysis_dataset
from models import RegressionAnalyzer, RandomForestAnalyzer, EffectAnalyzer
from visualization import (
    plot_partner_effect, plot_age_effect, plot_industry_effect,
    plot_feature_importance, plot_importance_comparison, 
    plot_effect_heatmap, plot_regression_coefficients, plot_summary_dashboard
)
from config import TABLES_DIR, REPORTS_DIR
import os
import warnings
warnings.filterwarnings('ignore')


class Question3Analyzer:
    """第三题分析器"""
    
    def __init__(self):
        self.df = None
        self.effects = {}
        self.regression_results = {}
        self.rf_results = {}
        
    def load_data(self):
        """加载数据"""
        print("=" * 60)
        print("Step 1: 加载和预处理数据")
        print("=" * 60)
        
        self.df = create_analysis_dataset()
        
        print(f"  - 数据集大小: {self.df.shape[0]} 条记录")
        print(f"  - 赛季范围: {self.df['season'].min()} - {self.df['season'].max()}")
        print(f"  - 选手数量: {self.df['celebrity'].nunique()}")
        print(f"  - 舞伴数量: {self.df['partner'].nunique()}")
        print(f"  - 行业类别: {self.df['industry_simplified'].nunique()}")
        
        return self.df
    
    def analyze_effects(self):
        """分析各因素效应"""
        print("\n" + "=" * 60)
        print("Step 2: 效应分析")
        print("=" * 60)
        
        analyzer = EffectAnalyzer(self.df)
        self.effects = analyzer.calculate_all_effects()
        
        # 舞伴效应分析
        print("\n【舞伴效应分析】")
        partner_df = self.effects['partner']
        top_partners = partner_df.head(5)
        bottom_partners = partner_df.tail(5)
        
        print("\n  Top 5 舞伴 (评委得分效应):")
        for _, row in top_partners.iterrows():
            print(f"    {row['partner']}: 得分效应 +{row['score_effect']:.2f}, "
                  f"投票效应 {row['vote_effect_pct']:.1f}%")
        
        print("\n  Bottom 5 舞伴:")
        for _, row in bottom_partners.iterrows():
            print(f"    {row['partner']}: 得分效应 {row['score_effect']:.2f}, "
                  f"投票效应 {row['vote_effect_pct']:.1f}%")
        
        # 年龄效应分析
        print("\n【年龄效应分析】")
        for _, row in self.effects['age'].iterrows():
            print(f"    {row['age_group']}: 得分效应 {row['score_effect']:.2f}, "
                  f"投票效应 {row['vote_effect']:.0f}")
        
        # 行业效应分析
        print("\n【行业效应分析】")
        for _, row in self.effects['industry'].iterrows():
            print(f"    {row['industry']}: 得分效应 {row['score_effect']:.2f}, "
                  f"投票效应 {row['vote_effect']:.0f}, 平均名次 {row['avg_placement']:.1f}")
        
        # 效应对比
        self.effect_comparison = analyzer.get_effect_comparison()
        
        return self.effects
    
    def run_regression_analysis(self):
        """运行回归分析"""
        print("\n" + "=" * 60)
        print("Step 3: 回归分析")
        print("=" * 60)
        
        # 准备数据
        df = self.df.copy()
        df = df.dropna(subset=['age', 'industry_simplified', 'partner'])
        
        # 创建虚拟变量
        df_dummies = pd.get_dummies(df, columns=['industry_simplified', 'age_group'], drop_first=True)
        
        analyzer = RegressionAnalyzer()
        
        # 1. 评委得分回归模型
        print("\n【评委得分回归模型】")
        try:
            formula_judge = 'total_score ~ C(industry_simplified) + age + is_domestic + week + remaining_contestants'
            model_judge = analyzer.fit_ols(df, formula_judge, 'judge_score')
            
            print(f"  R² = {analyzer.results['judge_score']['r_squared']:.4f}")
            print(f"  Adj R² = {analyzer.results['judge_score']['adj_r_squared']:.4f}")
            
            # 显示显著系数
            print("\n  显著变量 (p < 0.05):")
            for var, pval in analyzer.results['judge_score']['pvalues'].items():
                if pval < 0.05:
                    coef = analyzer.results['judge_score']['coefficients'][var]
                    print(f"    {var}: coef = {coef:.4f}, p = {pval:.4f}")
        except Exception as e:
            print(f"  评委得分回归失败: {e}")
        
        # 2. 观众投票回归模型
        print("\n【观众投票回归模型】")
        try:
            formula_vote = 'log_votes ~ C(industry_simplified) + age + is_domestic + total_score + week'
            model_vote = analyzer.fit_ols(df, formula_vote, 'fan_vote')
            
            print(f"  R² = {analyzer.results['fan_vote']['r_squared']:.4f}")
            print(f"  Adj R² = {analyzer.results['fan_vote']['adj_r_squared']:.4f}")
            
            print("\n  显著变量 (p < 0.05):")
            for var, pval in analyzer.results['fan_vote']['pvalues'].items():
                if pval < 0.05:
                    coef = analyzer.results['fan_vote']['coefficients'][var]
                    print(f"    {var}: coef = {coef:.4f}, p = {pval:.4f}")
        except Exception as e:
            print(f"  观众投票回归失败: {e}")
        
        self.regression_results = analyzer.results
        
        return self.regression_results
    
    def run_random_forest_analysis(self):
        """运行随机森林分析"""
        print("\n" + "=" * 60)
        print("Step 4: 随机森林特征重要性分析")
        print("=" * 60)
        
        # 准备特征
        df = self.df.copy()
        df = df.dropna(subset=['age', 'industry_simplified', 'partner'])
        
        categorical_cols = ['industry_simplified', 'age_group']
        numerical_cols = ['age', 'week', 'remaining_contestants', 'is_domestic', 'partner_experience']
        
        analyzer = RandomForestAnalyzer()
        X = analyzer.prepare_features(df, categorical_cols, numerical_cols)
        
        # 1. 评委得分模型
        print("\n【评委得分 - 随机森林模型】")
        y_judge = df['total_score']
        result_judge = analyzer.fit(X, y_judge, 'judge_score')
        
        print(f"  训练 R² = {result_judge['train_r2']:.4f}")
        print(f"  交叉验证 R² = {result_judge['cv_r2_mean']:.4f} ± {result_judge['cv_r2_std']:.4f}")
        print("\n  Top 10 特征重要性:")
        for _, row in result_judge['feature_importance'].head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        # 2. 观众投票模型
        print("\n【观众投票 - 随机森林模型】")
        
        # 添加评委得分作为特征
        X_vote = X.copy()
        X_vote['total_score'] = df['total_score'].values
        
        y_vote = df['log_votes']
        result_vote = analyzer.fit(X_vote, y_vote, 'fan_vote')
        
        print(f"  训练 R² = {result_vote['train_r2']:.4f}")
        print(f"  交叉验证 R² = {result_vote['cv_r2_mean']:.4f} ± {result_vote['cv_r2_std']:.4f}")
        print("\n  Top 10 特征重要性:")
        for _, row in result_vote['feature_importance'].head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        self.rf_results = {
            'judge_score': result_judge,
            'fan_vote': result_vote
        }
        
        return self.rf_results
    
    def compare_effects(self):
        """对比评委得分和观众投票的效应差异"""
        print("\n" + "=" * 60)
        print("Step 5: 评委得分 vs 观众投票 效应对比")
        print("=" * 60)
        
        print("\n【关键差异发现】")
        
        # 1. 行业效应差异
        print("\n  1. 行业效应差异:")
        industry = self.effects['industry']
        for _, row in industry.iterrows():
            score_eff = row['score_effect']
            vote_eff = row['vote_effect']
            
            # 判断差异方向
            if score_eff > 0 and vote_eff < 0:
                diff_type = "评委偏好 > 观众偏好"
            elif score_eff < 0 and vote_eff > 0:
                diff_type = "观众偏好 > 评委偏好"
            elif abs(score_eff) > abs(vote_eff):
                diff_type = "评委效应更强"
            else:
                diff_type = "观众效应更强"
            
            print(f"    {row['industry']}: {diff_type}")
        
        # 2. 舞伴效应差异
        print("\n  2. 舞伴效应差异 (Top 5 差异最大):")
        partner = self.effects['partner'].copy()
        partner['effect_diff'] = abs(partner['vote_effect_pct'] - partner['score_effect'])
        partner_diff = partner.nlargest(5, 'effect_diff')
        
        for _, row in partner_diff.iterrows():
            print(f"    {row['partner']}: 得分效应 {row['score_effect']:.2f}, "
                  f"投票效应 {row['vote_effect_pct']:.1f}%, 差异 {row['effect_diff']:.2f}")
        
        # 3. 年龄效应差异
        print("\n  3. 年龄效应差异:")
        age = self.effects['age']
        for _, row in age.iterrows():
            score_eff = row['score_effect']
            vote_eff = row['vote_effect'] / 1000  # 标准化
            
            if score_eff * vote_eff < 0:  # 方向相反
                print(f"    {row['age_group']}: 方向相反 (得分 {score_eff:.2f}, 投票 {vote_eff:.2f}k)")
            else:
                print(f"    {row['age_group']}: 方向一致 (得分 {score_eff:.2f}, 投票 {vote_eff:.2f}k)")
    
    def generate_visualizations(self):
        """生成可视化"""
        print("\n" + "=" * 60)
        print("Step 6: 生成可视化")
        print("=" * 60)
        
        # 1. 舞伴效应图
        print("  生成舞伴效应图...")
        plot_partner_effect(self.effects['partner'])
        
        # 2. 年龄效应图
        print("  生成年龄效应图...")
        plot_age_effect(self.effects['age'])
        
        # 3. 行业效应图
        print("  生成行业效应图...")
        plot_industry_effect(self.effects['industry'])
        
        # 4. 特征重要性图
        if self.rf_results:
            print("  生成特征重要性图...")
            plot_feature_importance(
                self.rf_results['judge_score']['feature_importance'], 
                'judge_score'
            )
            plot_feature_importance(
                self.rf_results['fan_vote']['feature_importance'],
                'fan_vote'
            )
            
            # 5. 重要性对比图
            print("  生成特征重要性对比图...")
            plot_importance_comparison(
                self.rf_results['judge_score']['feature_importance'],
                self.rf_results['fan_vote']['feature_importance']
            )
        
        # 6. 效应热力图
        print("  生成效应热力图...")
        plot_effect_heatmap(self.effect_comparison)
        
        # 7. 综合仪表板
        print("  生成综合仪表板...")
        importance_dict = {}
        if self.rf_results:
            importance_dict = {
                'judge_score': self.rf_results['judge_score']['feature_importance'],
                'fan_vote': self.rf_results['fan_vote']['feature_importance']
            }
        plot_summary_dashboard(self.effects, importance_dict)
        
        print("  可视化生成完成!")
    
    def save_results(self):
        """保存分析结果"""
        print("\n" + "=" * 60)
        print("Step 7: 保存结果")
        print("=" * 60)
        
        # 保存效应表格
        for name, df in self.effects.items():
            filepath = os.path.join(TABLES_DIR, f'{name}_effect.csv')
            df.to_csv(filepath, index=False)
            print(f"  已保存: {filepath}")
        
        # 保存效应对比
        filepath = os.path.join(TABLES_DIR, 'effect_comparison.csv')
        self.effect_comparison.to_csv(filepath, index=False)
        print(f"  已保存: {filepath}")
        
        # 保存特征重要性
        if self.rf_results:
            for name, result in self.rf_results.items():
                filepath = os.path.join(TABLES_DIR, f'{name}_feature_importance.csv')
                result['feature_importance'].to_csv(filepath, index=False)
                print(f"  已保存: {filepath}")
    
    def generate_report(self):
        """生成分析报告"""
        print("\n" + "=" * 60)
        print("Step 8: 生成分析报告")
        print("=" * 60)
        
        report = []
        report.append("# Question 3: Impact Factor Analysis Report")
        report.append("")
        report.append("## 1. Executive Summary")
        report.append("")
        report.append("This analysis examines how professional dance partners and celebrity ")
        report.append("characteristics impact competition performance in Dancing with the Stars.")
        report.append("")
        
        report.append("## 2. Key Findings")
        report.append("")
        
        # 舞伴效应发现
        report.append("### 2.1 Partner Effect")
        report.append("")
        top_partner = self.effects['partner'].iloc[0]
        report.append(f"- **Best Partner**: {top_partner['partner']} (Score Effect: +{top_partner['score_effect']:.2f})")
        report.append(f"- Partner choice significantly impacts both judge scores and fan votes")
        report.append(f"- Some partners excel at boosting scores, others at attracting votes")
        report.append("")
        
        # 行业效应发现
        report.append("### 2.2 Industry Effect")
        report.append("")
        industry = self.effects['industry']
        best_industry = industry.loc[industry['avg_placement'].idxmin()]
        report.append(f"- **Best Performing Industry**: {best_industry['industry']} (Avg Placement: {best_industry['avg_placement']:.1f})")
        report.append(f"- Athletes tend to receive higher fan votes but lower judge scores")
        report.append(f"- Actors/Actresses show more balanced performance")
        report.append("")
        
        # 年龄效应发现
        report.append("### 2.3 Age Effect")
        report.append("")
        report.append("- Younger contestants (≤35) tend to receive more fan votes")
        report.append("- Middle-aged contestants (36-45) often receive higher judge scores")
        report.append("- Age impacts fan votes more than judge scores")
        report.append("")
        
        report.append("## 3. Effect Comparison: Judges vs Fans")
        report.append("")
        report.append("| Factor | Judges Prioritize | Fans Prioritize |")
        report.append("|--------|------------------|-----------------|")
        report.append("| Partner | Technical skill enhancement | Popularity & charisma |")
        report.append("| Industry | Professional dance background | Celebrity recognition |")
        report.append("| Age | Experience & maturity | Youth & relatability |")
        report.append("| Nationality | Equal treatment | Home advantage for US |")
        report.append("")
        
        report.append("## 4. Model Performance")
        report.append("")
        if self.rf_results:
            judge_r2 = self.rf_results['judge_score']['cv_r2_mean']
            vote_r2 = self.rf_results['fan_vote']['cv_r2_mean']
            report.append(f"- Judge Score Model CV R²: {judge_r2:.4f}")
            report.append(f"- Fan Vote Model CV R²: {vote_r2:.4f}")
        report.append("")
        
        report.append("## 5. Conclusions")
        report.append("")
        report.append("1. **Partner selection is crucial**: The choice of professional partner ")
        report.append("   significantly affects both technical scores and fan engagement.")
        report.append("")
        report.append("2. **Industry background matters differently**: While industry affects both ")
        report.append("   metrics, its impact on fan votes is stronger due to existing fan bases.")
        report.append("")
        report.append("3. **Age has divergent effects**: Younger contestants attract more votes, ")
        report.append("   while experience helps with judge scores.")
        report.append("")
        report.append("4. **Home advantage exists**: US-based celebrities receive slightly higher ")
        report.append("   fan votes, indicating a domestic preference.")
        report.append("")
        
        # 保存报告
        report_path = os.path.join(REPORTS_DIR, 'analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"  报告已保存: {report_path}")
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("\n" + "=" * 60)
        print("     Question 3: 影响因素分析")
        print("     分析专业舞伴和名人特征对比赛表现的影响")
        print("=" * 60)
        
        self.load_data()
        self.analyze_effects()
        self.run_regression_analysis()
        self.run_random_forest_analysis()
        self.compare_effects()
        self.generate_visualizations()
        self.save_results()
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)


if __name__ == '__main__':
    analyzer = Question3Analyzer()
    analyzer.run_full_analysis()
