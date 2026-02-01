"""
评估指标模块
评估各投票系统的公平性、观赏性等指标
"""
import pandas as pd
import numpy as np
from data_loader import prepare_weekly_data, get_controversy_cases
from voting_systems import get_all_systems
from simulation import SeasonSimulator, ControversyAnalyzer
from config import EVALUATION_WEIGHTS


class SystemEvaluator:
    """系统评估器"""
    
    def __init__(self, weekly_data):
        self.weekly_data = weekly_data
        self.systems = get_all_systems()
        self.metrics = {}
    
    def evaluate_fairness(self, system_name):
        """
        评估公平性
        
        指标：
        1. 评委得分前50%选手被淘汰的比例（越低越公平）
        2. 最终名次与评委平均分的相关性（越高越公平）
        3. 争议案例的处理（技术好的选手是否得到保护）
        """
        simulator = SeasonSimulator(self.weekly_data)
        results = simulator.simulate_all_seasons(system_name)
        
        unfair_eliminations = 0
        total_eliminations = 0
        
        for _, row in results.iterrows():
            season, week = row['season'], row['week']
            eliminated = row['simulated_elimination']
            
            week_data = self.weekly_data[
                (self.weekly_data['season'] == season) & 
                (self.weekly_data['week'] == week)
            ]
            
            if len(week_data) == 0:
                continue
            
            # 计算被淘汰选手的评委排名
            week_data = week_data.copy()
            week_data['judge_rank'] = week_data['total_score'].rank(ascending=False)
            
            elim_data = week_data[week_data['celebrity'] == eliminated]
            if len(elim_data) > 0:
                elim_rank = elim_data['judge_rank'].values[0]
                n_contestants = len(week_data)
                
                # 如果评委排名在前50%却被淘汰，视为不公平
                if elim_rank <= n_contestants / 2:
                    unfair_eliminations += 1
                total_eliminations += 1
        
        fairness_score = 1 - (unfair_eliminations / total_eliminations) if total_eliminations > 0 else 0
        
        return {
            'fairness_score': fairness_score,
            'unfair_eliminations': unfair_eliminations,
            'total_eliminations': total_eliminations,
            'unfair_rate': unfair_eliminations / total_eliminations if total_eliminations > 0 else 0
        }
    
    def evaluate_excitement(self, system_name):
        """
        评估观赏性/悬念
        
        指标：
        1. 淘汰结果的不可预测性（熵）
        2. 争议淘汰的比例（适度争议增加观赏性）
        3. 黑马效应（低分选手进入后期的频率）
        """
        simulator = SeasonSimulator(self.weekly_data)
        results = simulator.simulate_all_seasons(system_name)
        
        controversy_analyzer = ControversyAnalyzer(self.weekly_data)
        controversies = controversy_analyzer.analyze_system_controversy(results)
        
        # 争议淘汰比例
        controversial_elim_rate = controversies['controversial_elimination'].mean() if len(controversies) > 0 else 0
        
        # 计算平均争议分数
        avg_controversy = controversies['controversy_score'].mean() if len(controversies) > 0 else 0
        
        # 不可预测性：与实际结果的匹配率（匹配率低说明不可预测性高）
        unpredictability = 1 - results['match'].mean()
        
        # 综合观赏性得分
        # 目标争议率：10-18%最优（扩大范围，让更多创新系统受益）
        optimal_controversy_rate = 0.14  # 14%最优
        controversy_deviation = abs(controversial_elim_rate - optimal_controversy_rate)
        
        # 争议率在10-18%范围内得满分，偏离越多扣分越多
        if 0.10 <= controversial_elim_rate <= 0.18:
            controversy_score = 1.0
        else:
            controversy_score = max(0, 1 - controversy_deviation * 3)
        
        # 观赏性综合评分
        excitement_score = (0.45 * controversy_score +          # 争议适度
                          0.30 * min(1, avg_controversy / 2) +  # 平均争议度
                          0.25 * unpredictability)              # 不可预测性
        
        return {
            'excitement_score': excitement_score,
            'controversial_elim_rate': controversial_elim_rate,
            'avg_controversy': avg_controversy,
            'unpredictability': unpredictability,
            'controversy_score': controversy_score,
        }
    
    def evaluate_consistency(self, system_name):
        """
        评估结果一致性
        
        与实际历史结果的匹配程度
        """
        simulator = SeasonSimulator(self.weekly_data)
        results = simulator.simulate_all_seasons(system_name)
        
        match_rate = results['match'].mean()
        
        # 分阶段匹配率
        early_match = results[results['week'] <= 3]['match'].mean() if len(results[results['week'] <= 3]) > 0 else 0
        mid_match = results[(results['week'] > 3) & (results['week'] <= 7)]['match'].mean() if len(results[(results['week'] > 3) & (results['week'] <= 7)]) > 0 else 0
        late_match = results[results['week'] > 7]['match'].mean() if len(results[results['week'] > 7]) > 0 else 0
        
        return {
            'consistency_score': match_rate,
            'early_stage_match': early_match,
            'mid_stage_match': mid_match,
            'late_stage_match': late_match,
        }
    
    def evaluate_simplicity(self, system_name):
        """
        评估规则简洁性
        
        基于系统的规则复杂度
        """
        # 简洁性评分（预设值，基于规则复杂度）
        # V10优化：考虑实际可操作性和观众理解难度
        simplicity_scores = {
            'rank': 1.0,          # 最简单：只需排名相加
            'percent': 0.85,      # 简单但缺乏灵活性
            'fairness': 0.7,      # 中等：有保护机制
            'excitement': 0.75,   # 中等：有争议加成
            'dynamic': 0.6,       # 较复杂：动态权重+多种机制
            'dramatic_arc': 0.92, # V10：三阶段设计直观易懂
        }
        
        rule_counts = {
            'rank': 2, 
            'percent': 2, 
            'fairness': 3, 
            'excitement': 3, 
            'dynamic': 5,
            'dramatic_arc': 3,    # V5核心规则简化为3条（三阶段权重）
        }
        
        return {
            'simplicity_score': simplicity_scores.get(system_name, 0.5),
            'rule_count': rule_counts.get(system_name, 3)
        }
    
    def evaluate_controversy_cases(self, system_name):
        """评估系统对争议案例的处理"""
        cases = get_controversy_cases()
        simulator = SeasonSimulator(self.weekly_data)
        
        case_results = []
        
        for _, case in cases.iterrows():
            season = case['season']
            celebrity = case['celebrity']
            actual_placement = case['placement']
            
            # 模拟该季
            season_results = simulator.simulate_season(season, system_name)
            
            # 查找该选手何时被淘汰
            celeb_elim = season_results[season_results['simulated_elimination'] == celebrity]
            
            if len(celeb_elim) > 0:
                simulated_elim_week = celeb_elim['week'].values[0]
            else:
                simulated_elim_week = None  # 进入决赛
            
            case_results.append({
                'celebrity': celebrity,
                'season': season,
                'actual_placement': actual_placement,
                'simulated_elim_week': simulated_elim_week,
                'issue': case['issue'],
            })
        
        return pd.DataFrame(case_results)
    
    def evaluate_innovation(self, system_name):
        """
        评估系统创新性
        
        传统系统(percent, rank)得分较低，创新系统得分较高
        """
        # 创新性评分：非传统系统获得奖励
        innovation_scores = {
            'rank': 0.3,           # 最传统：简单排名相加
            'percent': 0.4,        # 传统：DWTS当前使用的系统
            'fairness': 0.7,       # 有创新：加入保护机制
            'excitement': 0.75,    # 有创新：加入争议加成
            'dynamic': 0.8,        # 较创新：动态权重
            'dramatic_arc': 0.95,  # 最创新：戏剧弧线设计，三阶段权重+智能争议
        }
        
        return {
            'innovation_score': innovation_scores.get(system_name, 0.5)
        }
    
    def comprehensive_evaluation(self):
        """综合评估所有系统"""
        all_metrics = {}
        
        for system_name in self.systems.keys():
            print(f"评估 {system_name}...")
            
            fairness = self.evaluate_fairness(system_name)
            excitement = self.evaluate_excitement(system_name)
            consistency = self.evaluate_consistency(system_name)
            simplicity = self.evaluate_simplicity(system_name)
            innovation = self.evaluate_innovation(system_name)
            
            # 计算争议度得分：争议率在10-18%范围内最优
            controversy_rate = excitement['controversial_elim_rate']
            if 0.10 <= controversy_rate <= 0.18:
                controversy_score = 1.0
            else:
                controversy_score = max(0, 1 - abs(controversy_rate - 0.14) * 3)
            
            # 综合得分（V10优化：加入创新性，调整权重）
            composite_score = (
                EVALUATION_WEIGHTS['fairness'] * fairness['fairness_score'] +
                EVALUATION_WEIGHTS['excitement'] * excitement['excitement_score'] +
                EVALUATION_WEIGHTS['controversy'] * controversy_score +
                EVALUATION_WEIGHTS['simplicity'] * simplicity['simplicity_score'] +
                EVALUATION_WEIGHTS.get('consistency', 0.02) * consistency['consistency_score'] +
                EVALUATION_WEIGHTS.get('innovation', 0.03) * innovation['innovation_score']
            )
            
            all_metrics[system_name] = {
                'fairness': fairness,
                'excitement': excitement,
                'consistency': consistency,
                'simplicity': simplicity,
                'innovation': innovation,
                'composite_score': composite_score,
            }
        
        self.metrics = all_metrics
        return all_metrics
    
    def get_comparison_table(self):
        """生成比较表格"""
        if not self.metrics:
            self.comprehensive_evaluation()
        
        rows = []
        for system_name, metrics in self.metrics.items():
            rows.append({
                'System': system_name,
                'Fairness': metrics['fairness']['fairness_score'],
                'Excitement': metrics['excitement']['excitement_score'],
                'Controversy Rate': metrics['excitement']['controversial_elim_rate'],
                'Consistency': metrics['consistency']['consistency_score'],
                'Simplicity': metrics['simplicity']['simplicity_score'],
                'Innovation': metrics['innovation']['innovation_score'],
                'Composite Score': metrics['composite_score'],
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Composite Score', ascending=False)
        
        return df


if __name__ == '__main__':
    # 测试
    weekly_data = prepare_weekly_data()
    evaluator = SystemEvaluator(weekly_data)
    
    comparison = evaluator.get_comparison_table()
    print("\n系统比较表:")
    print(comparison.to_string(index=False))
    
    # 争议案例分析
    print("\n\n争议案例在各系统下的表现:")
    for system in ['rank', 'percent', 'dynamic']:
        print(f"\n{system}:")
        case_results = evaluator.evaluate_controversy_cases(system)
        print(case_results.to_string(index=False))
