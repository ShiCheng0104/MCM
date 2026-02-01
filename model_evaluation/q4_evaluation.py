"""
模型评估与推广 - Q4模型评估
Dynamic Voting System Evaluation
"""
import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_evaluation.config import Q4_PATH, EVALUATION_CONFIG


class Q4ModelEvaluator:
    """Q4: 动态投票系统评估"""
    
    def __init__(self):
        self.load_data()
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        # 加载系统对比结果
        comparison_path = os.path.join(Q4_PATH, 'outputs', 'tables', 'system_comparison.csv')
        self.system_comparison = pd.read_csv(comparison_path)
        
        # 加载详细指标
        detailed_path = os.path.join(Q4_PATH, 'outputs', 'tables', 'detailed_metrics.csv')
        if os.path.exists(detailed_path):
            self.detailed_metrics = pd.read_csv(detailed_path)
        else:
            self.detailed_metrics = None
    
    def evaluate_composite_performance(self):
        """评估综合性能"""
        df = self.system_comparison.copy()
        
        # 找到我们的系统和基准系统
        our_system = df[df['System'] == 'dramatic_arc'].iloc[0] if 'dramatic_arc' in df['System'].values else None
        baseline = df[df['System'] == 'percent'].iloc[0] if 'percent' in df['System'].values else None
        
        if our_system is not None and baseline is not None:
            performance = {
                'our_composite': our_system['Composite Score'],
                'baseline_composite': baseline['Composite Score'],
                'improvement': our_system['Composite Score'] - baseline['Composite Score'],
                'relative_improvement': (our_system['Composite Score'] - baseline['Composite Score']) / baseline['Composite Score'] * 100,
            }
            
            # 各维度对比
            dimensions = ['Fairness', 'Excitement', 'Consistency', 'Simplicity']
            for dim in dimensions:
                if dim in our_system.index:
                    performance[f'{dim.lower()}_ours'] = our_system[dim]
                    performance[f'{dim.lower()}_baseline'] = baseline[dim]
                    performance[f'{dim.lower()}_diff'] = our_system[dim] - baseline[dim]
        else:
            performance = {'error': 'System not found'}
        
        self.results['composite_performance'] = performance
        return self.results['composite_performance']
    
    def evaluate_multi_objective_tradeoff(self):
        """多目标权衡分析"""
        df = self.system_comparison.copy()
        
        # 计算Pareto效率
        pareto_efficient = []
        
        for idx, row in df.iterrows():
            is_dominated = False
            for idx2, row2 in df.iterrows():
                if idx == idx2:
                    continue
                # 检查是否被支配（在所有维度上都更差）
                dims = ['Fairness', 'Excitement']
                if all(row2[d] >= row[d] for d in dims if d in row.index) and \
                   any(row2[d] > row[d] for d in dims if d in row.index):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_efficient.append(row['System'])
        
        self.results['pareto_analysis'] = {
            'pareto_efficient_systems': pareto_efficient,
            'our_system_pareto': 'dramatic_arc' in pareto_efficient,
        }
        
        return self.results['pareto_analysis']
    
    def controversy_optimality(self):
        """争议率最优性分析"""
        df = self.system_comparison.copy()
        
        # 最优争议率范围
        optimal_range = (0.12, 0.18)
        
        controversy_analysis = []
        for _, row in df.iterrows():
            rate = row['Controversy Rate']
            in_range = optimal_range[0] <= rate <= optimal_range[1]
            distance = 0 if in_range else min(abs(rate - optimal_range[0]), 
                                               abs(rate - optimal_range[1]))
            controversy_analysis.append({
                'system': row['System'],
                'rate': rate,
                'in_optimal_range': in_range,
                'distance_from_optimal': distance,
            })
        
        df_analysis = pd.DataFrame(controversy_analysis)
        
        self.results['controversy_optimality'] = {
            'analysis': controversy_analysis,
            'our_system_in_range': df_analysis[df_analysis['system'] == 'dramatic_arc']['in_optimal_range'].values[0] if 'dramatic_arc' in df_analysis['system'].values else False,
            'systems_in_range': df_analysis[df_analysis['in_optimal_range']]['system'].tolist(),
        }
        
        return self.results['controversy_optimality']
    
    def robustness_across_metrics(self):
        """跨指标稳健性"""
        df = self.system_comparison.copy()
        
        metrics = ['Fairness', 'Excitement', 'Consistency', 'Simplicity']
        available_metrics = [m for m in metrics if m in df.columns]
        
        robustness_scores = {}
        for _, row in df.iterrows():
            scores = [row[m] for m in available_metrics]
            robustness_scores[row['System']] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'cv': np.std(scores) / np.mean(scores),  # 变异系数
                'robustness': 1 - np.std(scores) / np.mean(scores),
            }
        
        self.results['robustness'] = robustness_scores
        return self.results['robustness']
    
    def ranking_stability(self):
        """排名稳定性分析"""
        df = self.system_comparison.copy()
        
        # 各指标排名
        rankings = {}
        for col in ['Fairness', 'Excitement', 'Consistency', 'Simplicity', 'Composite Score']:
            if col in df.columns:
                rankings[col] = df.sort_values(col, ascending=False)['System'].tolist()
        
        # 计算每个系统的排名变异
        system_rank_variance = {}
        for system in df['System'].unique():
            ranks = []
            for col, ranking in rankings.items():
                if system in ranking:
                    ranks.append(ranking.index(system) + 1)
            if ranks:
                system_rank_variance[system] = {
                    'mean_rank': np.mean(ranks),
                    'rank_std': np.std(ranks),
                    'best_rank': min(ranks),
                    'worst_rank': max(ranks),
                }
        
        self.results['ranking_stability'] = system_rank_variance
        return self.results['ranking_stability']
    
    def innovation_value(self):
        """创新价值评估"""
        df = self.system_comparison.copy()
        
        # 计算相对于baseline的改进
        if 'percent' in df['System'].values and 'dramatic_arc' in df['System'].values:
            baseline = df[df['System'] == 'percent'].iloc[0]
            ours = df[df['System'] == 'dramatic_arc'].iloc[0]
            
            innovations = {}
            for col in ['Fairness', 'Excitement', 'Innovation']:
                if col in baseline.index:
                    innovations[col] = {
                        'baseline': baseline[col],
                        'ours': ours[col],
                        'improvement': ours[col] - baseline[col],
                    }
            
            # 综合创新分数
            improvements = [v['improvement'] for v in innovations.values() if 'improvement' in v]
            innovations['total_innovation_score'] = sum(max(0, imp) for imp in improvements)
            
            self.results['innovation'] = innovations
        else:
            self.results['innovation'] = {'error': 'Systems not found'}
        
        return self.results['innovation']
    
    def run_full_evaluation(self):
        """运行完整评估"""
        print("=" * 60)
        print("Q4: Dynamic Voting System Evaluation")
        print("=" * 60)
        
        print("\n1. Composite Performance...")
        cp = self.evaluate_composite_performance()
        if 'improvement' in cp:
            print(f"   Our Composite: {cp['our_composite']:.3f}")
            print(f"   Baseline: {cp['baseline_composite']:.3f}")
            print(f"   Improvement: {cp['improvement']:+.3f} ({cp['relative_improvement']:+.1f}%)")
        
        print("\n2. Pareto Analysis...")
        pa = self.evaluate_multi_objective_tradeoff()
        print(f"   Pareto Efficient Systems: {pa['pareto_efficient_systems']}")
        print(f"   Our System Pareto: {pa['our_system_pareto']}")
        
        print("\n3. Controversy Optimality...")
        co = self.controversy_optimality()
        print(f"   Systems in Optimal Range: {co['systems_in_range']}")
        
        print("\n4. Robustness Analysis...")
        rob = self.robustness_across_metrics()
        if 'dramatic_arc' in rob:
            print(f"   Our Robustness: {rob['dramatic_arc']['robustness']:.3f}")
        
        print("\n5. Ranking Stability...")
        rs = self.ranking_stability()
        if 'dramatic_arc' in rs:
            print(f"   Mean Rank: {rs['dramatic_arc']['mean_rank']:.1f}")
            print(f"   Rank Range: {rs['dramatic_arc']['best_rank']}-{rs['dramatic_arc']['worst_rank']}")
        
        print("\n6. Innovation Value...")
        iv = self.innovation_value()
        if 'total_innovation_score' in iv:
            print(f"   Innovation Score: {iv['total_innovation_score']:.3f}")
        
        return self.results
    
    def get_summary_metrics(self):
        """获取摘要指标"""
        if not self.results:
            self.run_full_evaluation()
        
        summary = {'model': 'Q4: Voting System'}
        
        if 'composite_performance' in self.results:
            cp = self.results['composite_performance']
            if 'our_composite' in cp:
                summary['composite_score'] = cp['our_composite']
                summary['vs_baseline'] = cp.get('improvement', 0)
        
        if 'pareto_analysis' in self.results:
            summary['pareto_efficient'] = self.results['pareto_analysis']['our_system_pareto']
        
        if 'robustness' in self.results and 'dramatic_arc' in self.results['robustness']:
            summary['robustness'] = self.results['robustness']['dramatic_arc']['robustness']
        
        if 'innovation' in self.results and 'total_innovation_score' in self.results['innovation']:
            summary['innovation_score'] = self.results['innovation']['total_innovation_score']
        
        return summary


if __name__ == '__main__':
    evaluator = Q4ModelEvaluator()
    results = evaluator.run_full_evaluation()
    print("\n" + "=" * 60)
    print("Summary:")
    print(evaluator.get_summary_metrics())
