"""
一致性检验模块
验证估计的投票是否与实际淘汰结果一致
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from .utils import (
    compute_rank_combined_score,
    compute_percent_combined_score,
    get_eliminated_index_rank,
    get_eliminated_index_percent
)


class ConsistencyChecker:
    """
    一致性检验类
    检验估计的观众投票是否能正确预测淘汰结果
    """
    
    def __init__(self):
        """初始化一致性检验器"""
        self.results = None
        self.summary = None
        
    def check_elimination_consistency(self,
                                      estimates: Dict,
                                      elimination_info: pd.DataFrame,
                                      method: str = 'auto') -> pd.DataFrame:
        """
        检验淘汰一致性
        
        Args:
            estimates: 投票估计结果字典
            elimination_info: 淘汰信息DataFrame
            method: 投票计算方法 ('rank', 'percent', 'auto')
        
        Returns:
            一致性检验结果DataFrame
        """
        results = []
        
        for (season, week), est in estimates.items():
            # 获取实际淘汰者（可能有多人）
            actual_elim = elimination_info[
                (elimination_info['season'] == season) &
                (elimination_info['week'] == week)
            ]
            
            if len(actual_elim) == 0:
                # 无淘汰的周次
                continue
            
            # 获取所有实际被淘汰者的名字列表
            actual_names = actual_elim['eliminated_name'].tolist()
            n_eliminated = len(actual_names)
            
            names = est['names']
            scores = est['scores']
            votes = est['votes']
            
            # 确定使用的方法
            if method == 'auto':
                use_method = 'rank' if (season in [1, 2] or season >= 28) else 'percent'
            else:
                use_method = method
            
            # 计算综合得分并预测淘汰者
            if use_method == 'rank':
                combined = compute_rank_combined_score(scores, votes)
                # 预测底N名（N=实际淘汰人数）
                pred_indices = np.argsort(-combined)[:n_eliminated]  # 综合排名最高（最差）的N人
            else:
                combined = compute_percent_combined_score(scores, votes)
                # 预测底N名
                pred_indices = np.argsort(combined)[:n_eliminated]  # 综合百分比最低的N人
            
            pred_names = [names[i] for i in pred_indices]
            
            # 获取实际淘汰者的索引列表
            actual_indices = []
            for actual_name in actual_names:
                for i, name in enumerate(names):
                    if name == actual_name:
                        actual_indices.append(i)
                        break
            
            # 计算准确性：预测的淘汰者是否都在实际淘汰者中
            # 使用集合交集来判断
            pred_set = set(pred_names)
            actual_set = set(actual_names)
            
            # 完全正确：预测集合与实际集合完全相同
            is_correct = (pred_set == actual_set)
            
            # 部分正确：预测集合与实际集合有交集
            correct_count = len(pred_set & actual_set)
            
            # 计算底N+1准确性（实际淘汰者是否都在预测的底N+1中）
            if use_method == 'rank':
                bottom_n_plus_1_idx = set(np.argsort(-combined)[:n_eliminated + 1])
            else:
                bottom_n_plus_1_idx = set(np.argsort(combined)[:n_eliminated + 1])
            
            in_bottom_n_plus_1 = all(idx in bottom_n_plus_1_idx for idx in actual_indices if idx is not None)
            
            # 计算排名
            if use_method == 'rank':
                pred_rank = combined[pred_indices[0]] if len(pred_indices) > 0 else None
                actual_rank = combined[actual_indices[0]] if len(actual_indices) > 0 else None
            else:
                sorted_indices = np.argsort(combined)
                rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
                pred_rank = rank_map.get(pred_indices[0]) if len(pred_indices) > 0 else None
                actual_rank = rank_map.get(actual_indices[0]) if len(actual_indices) > 0 else None
            
            results.append({
                'season': season,
                'week': week,
                'method': use_method,
                'n_contestants': len(names),
                'n_eliminated': n_eliminated,
                'actual_eliminated': ', '.join(actual_names),
                'predicted_eliminated': ', '.join(pred_names),
                'is_correct': is_correct,
                'correct_count': correct_count,
                'in_bottom_two': in_bottom_n_plus_1,  # 保持字段名兼容，但实际是底N+1
                'actual_idx': actual_indices[0] if actual_indices else None,
                'pred_idx': pred_indices[0] if len(pred_indices) > 0 else None,
                'actual_combined_rank': actual_rank,
                'pred_combined_rank': pred_rank
            })
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def compute_summary_statistics(self) -> Dict:
        """
        计算汇总统计量
        
        Returns:
            汇总统计字典
        """
        if self.results is None:
            raise ValueError("请先运行 check_elimination_consistency")
        
        total = len(self.results)
        correct = self.results['is_correct'].sum()
        in_bottom_two = self.results['in_bottom_two'].sum()
        
        # 按方法分组统计
        method_stats = self.results.groupby('method').agg({
            'is_correct': ['sum', 'count', 'mean'],
            'in_bottom_two': 'mean'
        }).round(4)
        
        # 按赛季分组统计
        season_stats = self.results.groupby('season').agg({
            'is_correct': 'mean',
            'in_bottom_two': 'mean'
        }).round(4)
        
        self.summary = {
            'total_weeks': total,
            'correct_predictions': correct,
            'elimination_accuracy': correct / total if total > 0 else 0,
            'bottom_two_accuracy': in_bottom_two / total if total > 0 else 0,
            'method_stats': method_stats,
            'season_stats': season_stats
        }
        
        return self.summary
    
    def print_summary(self):
        """打印汇总结果"""
        if self.summary is None:
            self.compute_summary_statistics()
        
        print("\n" + "="*60)
        print("一致性检验汇总")
        print("="*60)
        print(f"总周次数: {self.summary['total_weeks']}")
        print(f"正确预测数: {self.summary['correct_predictions']}")
        print(f"淘汰预测准确率: {self.summary['elimination_accuracy']:.2%}")
        print(f"底2准确率: {self.summary['bottom_two_accuracy']:.2%}")
        print("\n按方法分组统计:")
        print(self.summary['method_stats'])
        print("="*60 + "\n")
    
    def get_incorrect_predictions(self) -> pd.DataFrame:
        """
        获取预测错误的案例
        
        Returns:
            错误预测DataFrame
        """
        if self.results is None:
            raise ValueError("请先运行 check_elimination_consistency")
        
        return self.results[~self.results['is_correct']].copy()
    
    def compute_rank_correlation(self,
                                estimates: Dict,
                                elimination_info: pd.DataFrame) -> pd.DataFrame:
        """
        计算排名相关性（Kendall's τ 和 Spearman's ρ）
        
        Args:
            estimates: 投票估计
            elimination_info: 淘汰信息
        
        Returns:
            相关性结果DataFrame
        """
        correlations = []
        
        for (season, week), est in estimates.items():
            scores = np.array(est['scores'])
            votes = np.array(est['votes'])
            
            # 计算评委排名和投票排名的相关性
            score_ranks = np.argsort(np.argsort(-scores)) + 1
            vote_ranks = np.argsort(np.argsort(-votes)) + 1
            
            if len(scores) >= 3:
                kendall_tau, kendall_p = stats.kendalltau(score_ranks, vote_ranks)
                spearman_rho, spearman_p = stats.spearmanr(score_ranks, vote_ranks)
            else:
                kendall_tau, kendall_p = np.nan, np.nan
                spearman_rho, spearman_p = np.nan, np.nan
            
            correlations.append({
                'season': season,
                'week': week,
                'n_contestants': len(scores),
                'kendall_tau': kendall_tau,
                'kendall_p': kendall_p,
                'spearman_rho': spearman_rho,
                'spearman_p': spearman_p
            })
        
        return pd.DataFrame(correlations)
    
    def analyze_controversy_cases(self,
                                 estimates: Dict,
                                 cleaned_data: pd.DataFrame,
                                 threshold: float = 3.0) -> pd.DataFrame:
        """
        分析争议案例（评委排名与综合排名差异大的选手）
        
        Args:
            estimates: 投票估计
            cleaned_data: 清洗后的数据
            threshold: 排名差异阈值
        
        Returns:
            争议案例DataFrame
        """
        controversy_cases = []
        
        for (season, week), est in estimates.items():
            names = est['names']
            scores = np.array(est['scores'])
            votes = np.array(est['votes'])
            
            # 计算各种排名
            score_ranks = np.argsort(np.argsort(-scores)) + 1
            vote_ranks = np.argsort(np.argsort(-votes)) + 1
            
            # 确定方法并计算综合排名
            if season in [1, 2] or season >= 28:
                combined = compute_rank_combined_score(scores, votes)
                combined_ranks = np.argsort(np.argsort(combined)) + 1
            else:
                combined = compute_percent_combined_score(scores, votes)
                combined_ranks = np.argsort(np.argsort(-combined)) + 1
            
            for i, name in enumerate(names):
                rank_diff = abs(score_ranks[i] - combined_ranks[i])
                
                if rank_diff >= threshold:
                    # 获取选手信息
                    player_info = cleaned_data[cleaned_data['celebrity_name'] == name]
                    
                    controversy_cases.append({
                        'season': season,
                        'week': week,
                        'celebrity_name': name,
                        'judge_score': scores[i],
                        'judge_rank': score_ranks[i],
                        'vote_rank': vote_ranks[i],
                        'combined_rank': combined_ranks[i],
                        'rank_difference': rank_diff,
                        'final_placement': player_info['placement'].iloc[0] if len(player_info) > 0 else None
                    })
        
        return pd.DataFrame(controversy_cases)


def run_consistency_check(estimates: Dict,
                         elimination_info: pd.DataFrame,
                         cleaned_data: pd.DataFrame,
                         verbose: bool = True) -> Tuple[ConsistencyChecker, Dict]:
    """
    运行完整的一致性检验
    
    Args:
        estimates: 投票估计结果
        elimination_info: 淘汰信息
        cleaned_data: 清洗后的数据
        verbose: 是否打印结果
    
    Returns:
        (检验器对象, 汇总结果)
    """
    checker = ConsistencyChecker()
    
    # 淘汰一致性检验
    checker.check_elimination_consistency(estimates, elimination_info)
    summary = checker.compute_summary_statistics()
    
    if verbose:
        checker.print_summary()
    
    # 排名相关性分析
    correlations = checker.compute_rank_correlation(estimates, elimination_info)
    summary['avg_kendall_tau'] = correlations['kendall_tau'].mean()
    summary['avg_spearman_rho'] = correlations['spearman_rho'].mean()
    
    if verbose:
        print(f"平均Kendall τ: {summary['avg_kendall_tau']:.3f}")
        print(f"平均Spearman ρ: {summary['avg_spearman_rho']:.3f}")
    
    # 争议案例分析
    controversy = checker.analyze_controversy_cases(estimates, cleaned_data)
    summary['n_controversy_cases'] = len(controversy)
    
    if verbose:
        print(f"争议案例数: {summary['n_controversy_cases']}")
    
    return checker, summary
