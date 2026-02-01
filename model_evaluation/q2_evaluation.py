"""
模型评估与推广 - Q2模型评估
Voting Method Comparison Evaluation
"""
import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_evaluation.config import Q2_PATH, EVALUATION_CONFIG, DATA_PATH


class Q2ModelEvaluator:
    """Q2: 投票方法对比分析评估"""
    
    def __init__(self):
        self.load_data()
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        # 加载方法对比结果
        comparison_path = os.path.join(Q2_PATH, 'outputs', 'method_comparison.csv')
        self.method_comparison = pd.read_csv(comparison_path)
        
        # 加载争议分析
        controversy_path = os.path.join(Q2_PATH, 'outputs', 'controversy_analysis.csv')
        if os.path.exists(controversy_path):
            self.controversy_analysis = pd.read_csv(controversy_path)
        else:
            self.controversy_analysis = None
            
    def evaluate_method_agreement(self):
        """评估两种方法的一致性"""
        df = self.method_comparison.copy()
        
        # 总体一致率
        overall_agreement = df['methods_agree'].mean()
        
        # 分赛季一致率
        season_agreement = df.groupby('season')['methods_agree'].mean()
        
        # 按选手数量分组
        df['contestant_group'] = pd.cut(df['num_contestants'], 
                                        bins=[0, 5, 8, 15], 
                                        labels=['Final (2-5)', 'Mid (6-8)', 'Early (9+)'])
        group_agreement = df.groupby('contestant_group')['methods_agree'].mean()
        
        self.results['method_agreement'] = {
            'overall': overall_agreement,
            'by_season': season_agreement.to_dict(),
            'by_stage': group_agreement.to_dict(),
            'disagreement_count': (~df['methods_agree']).sum(),
            'total_weeks': len(df),
        }
        
        return self.results['method_agreement']
    
    def evaluate_method_accuracy(self):
        """评估各方法的历史准确率"""
        df = self.method_comparison.copy()
        
        # 只看有实际淘汰记录的周次
        df_with_actual = df[df['actual_eliminated'].notna()]
        
        # 排名法准确率
        rank_accuracy = df_with_actual['rank_matches_actual'].mean()
        
        # 百分比法准确率  
        percent_accuracy = df_with_actual['percent_matches_actual'].mean()
        
        # 分赛季准确率
        rank_by_season = df_with_actual.groupby('season')['rank_matches_actual'].mean()
        percent_by_season = df_with_actual.groupby('season')['percent_matches_actual'].mean()
        
        self.results['method_accuracy'] = {
            'rank_method': {
                'overall': rank_accuracy,
                'by_season': rank_by_season.to_dict(),
            },
            'percent_method': {
                'overall': percent_accuracy,
                'by_season': percent_by_season.to_dict(),
            },
            'accuracy_difference': percent_accuracy - rank_accuracy,
        }
        
        return self.results['method_accuracy']
    
    def controversy_detection_rate(self):
        """争议案例检测率"""
        df = self.method_comparison.copy()
        
        # 方法不一致 = 潜在争议
        disagreements = df[~df['methods_agree']]
        
        # 不一致率
        controversy_rate = len(disagreements) / len(df)
        
        # 分析不一致的模式
        patterns = {}
        if len(disagreements) > 0:
            # 哪些赛季争议最多
            season_controversy = disagreements.groupby('season').size()
            patterns['most_controversial_season'] = season_controversy.idxmax()
            patterns['controversy_by_season'] = season_controversy.to_dict()
        
        self.results['controversy_detection'] = {
            'controversy_rate': controversy_rate,
            'controversy_count': len(disagreements),
            'patterns': patterns,
        }
        
        return self.results['controversy_detection']
    
    def statistical_significance_test(self):
        """统计显著性检验：两种方法是否有显著差异"""
        df = self.method_comparison.copy()
        df_with_actual = df[df['actual_eliminated'].notna()]
        
        # 配对McNemar检验
        rank_correct = df_with_actual['rank_matches_actual'].values
        percent_correct = df_with_actual['percent_matches_actual'].values
        
        # 构建2x2列联表
        a = ((rank_correct == 1) & (percent_correct == 1)).sum()  # 两者都对
        b = ((rank_correct == 1) & (percent_correct == 0)).sum()  # rank对，percent错
        c = ((rank_correct == 0) & (percent_correct == 1)).sum()  # rank错，percent对
        d = ((rank_correct == 0) & (percent_correct == 0)).sum()  # 两者都错
        
        # McNemar统计量
        if (b + c) > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2, 1)
        else:
            chi2 = 0
            p_value = 1.0
        
        self.results['statistical_test'] = {
            'test': 'McNemar',
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'contingency_table': {
                'both_correct': int(a),
                'rank_only_correct': int(b),
                'percent_only_correct': int(c),
                'both_wrong': int(d),
            },
        }
        
        return self.results['statistical_test']
    
    def robustness_analysis(self):
        """稳健性分析：不同条件下的方法表现"""
        df = self.method_comparison.copy()
        df_with_actual = df[df['actual_eliminated'].notna()]
        
        # 早期赛季 vs 后期赛季
        early_seasons = df_with_actual[df_with_actual['season'] <= 17]
        late_seasons = df_with_actual[df_with_actual['season'] > 17]
        
        robustness = {
            'early_seasons': {
                'rank_accuracy': early_seasons['rank_matches_actual'].mean() if len(early_seasons) > 0 else 0,
                'percent_accuracy': early_seasons['percent_matches_actual'].mean() if len(early_seasons) > 0 else 0,
            },
            'late_seasons': {
                'rank_accuracy': late_seasons['rank_matches_actual'].mean() if len(late_seasons) > 0 else 0,
                'percent_accuracy': late_seasons['percent_matches_actual'].mean() if len(late_seasons) > 0 else 0,
            },
        }
        
        # 计算稳健性指标（跨时期一致性）
        rank_diff = abs(robustness['early_seasons']['rank_accuracy'] - 
                       robustness['late_seasons']['rank_accuracy'])
        percent_diff = abs(robustness['early_seasons']['percent_accuracy'] - 
                          robustness['late_seasons']['percent_accuracy'])
        
        robustness['rank_stability'] = 1 - rank_diff
        robustness['percent_stability'] = 1 - percent_diff
        
        self.results['robustness'] = robustness
        return self.results['robustness']
    
    def run_full_evaluation(self):
        """运行完整评估"""
        print("=" * 60)
        print("Q2: Voting Method Comparison Evaluation")
        print("=" * 60)
        
        print("\n1. Method Agreement...")
        agree = self.evaluate_method_agreement()
        print(f"   Overall Agreement: {agree['overall']:.2%}")
        print(f"   Disagreements: {agree['disagreement_count']}/{agree['total_weeks']}")
        
        print("\n2. Method Accuracy...")
        acc = self.evaluate_method_accuracy()
        print(f"   Rank Method: {acc['rank_method']['overall']:.2%}")
        print(f"   Percent Method: {acc['percent_method']['overall']:.2%}")
        
        print("\n3. Controversy Detection...")
        cont = self.controversy_detection_rate()
        print(f"   Controversy Rate: {cont['controversy_rate']:.2%}")
        
        print("\n4. Statistical Significance...")
        stat = self.statistical_significance_test()
        print(f"   McNemar p-value: {stat['p_value']:.4f}")
        print(f"   Significant Difference: {stat['significant']}")
        
        print("\n5. Robustness Analysis...")
        rob = self.robustness_analysis()
        print(f"   Rank Stability: {rob['rank_stability']:.2%}")
        print(f"   Percent Stability: {rob['percent_stability']:.2%}")
        
        return self.results
    
    def get_summary_metrics(self):
        """获取摘要指标"""
        if not self.results:
            self.run_full_evaluation()
        
        return {
            'model': 'Q2: Method Comparison',
            'method_agreement': self.results['method_agreement']['overall'],
            'rank_accuracy': self.results['method_accuracy']['rank_method']['overall'],
            'percent_accuracy': self.results['method_accuracy']['percent_method']['overall'],
            'controversy_rate': self.results['controversy_detection']['controversy_rate'],
            'significance_p': self.results['statistical_test']['p_value'],
            'rank_stability': self.results['robustness']['rank_stability'],
            'percent_stability': self.results['robustness']['percent_stability'],
        }


if __name__ == '__main__':
    evaluator = Q2ModelEvaluator()
    results = evaluator.run_full_evaluation()
    print("\n" + "=" * 60)
    print("Summary:")
    print(evaluator.get_summary_metrics())
