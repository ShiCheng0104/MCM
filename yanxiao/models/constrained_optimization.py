"""
约束优化模型
基于淘汰结果约束反推观众投票
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from typing import Dict, Tuple, List, Optional
from src.utils import (
    compute_rank_combined_score,
    compute_percent_combined_score,
    get_eliminated_index_rank,
    get_eliminated_index_percent,
    normalize_votes,
    rank_scores
)


class ConstrainedOptimizationModel:
    """
    约束优化投票估计模型
    
    核心思想：
    1. 观众投票必须使得综合排名最低的选手被淘汰
    2. 在满足约束的条件下，最小化与先验估计的偏差
    """
    
    def __init__(self, 
                 method: str = 'SLSQP',
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 prior_weight: float = 0.5):
        """
        初始化约束优化模型
        
        Args:
            method: 优化方法
            max_iter: 最大迭代次数
            tol: 收敛容差
            prior_weight: 先验权重（评委得分对先验投票的影响）
        """
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.prior_weight = prior_weight
        self.estimates = {}
        self.optimization_results = {}
        
    def compute_prior_votes(self, 
                           judge_scores: np.ndarray,
                           total_votes: float = 1e6) -> np.ndarray:
        """
        计算先验投票估计（基于评委得分）
        
        Args:
            judge_scores: 评委得分
            total_votes: 总投票数
        
        Returns:
            先验投票估计
        """
        # 使用softmax风格的先验
        exp_scores = np.exp(self.prior_weight * judge_scores / judge_scores.max())
        prior = exp_scores / exp_scores.sum() * total_votes
        return prior
    
    def estimate_votes_rank_method(self,
                                   judge_scores: np.ndarray,
                                   eliminated_idx: int,
                                   total_votes: float = 1e6) -> Tuple[np.ndarray, Dict]:
        """
        排名法下的约束优化估计
        
        约束：被淘汰选手的综合排名得分最高
        
        Args:
            judge_scores: 评委得分数组
            eliminated_idx: 被淘汰选手的索引
            total_votes: 总投票数
        
        Returns:
            (估计投票, 优化结果信息)
        """
        n = len(judge_scores)
        prior = self.compute_prior_votes(judge_scores, total_votes)
        
        # 目标函数：最小化与先验的偏差
        def objective(votes):
            # 使用对数空间减少数值问题
            log_votes = np.log(votes + 1)
            log_prior = np.log(prior + 1)
            return np.sum((log_votes - log_prior) ** 2)
        
        # 约束：被淘汰者的综合排名得分 >= 其他所有人
        def elimination_constraint(votes):
            combined = compute_rank_combined_score(judge_scores, votes)
            eliminated_score = combined[eliminated_idx]
            other_scores = np.delete(combined, eliminated_idx)
            # 被淘汰者得分应该 >= max(其他人得分)
            # 返回 eliminated_score - max(others) >= 0
            return eliminated_score - np.max(other_scores)
        
        # 初始值
        x0 = prior.copy()
        
        # 边界：投票数非负
        bounds = [(100, total_votes * 0.8) for _ in range(n)]
        
        # 约束条件
        constraints = [
            {'type': 'ineq', 'fun': elimination_constraint},
            {'type': 'eq', 'fun': lambda v: np.sum(v) - total_votes}  # 总票数约束
        ]
        
        # 优化
        result = minimize(
            objective,
            x0,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        votes = result.x
        votes = normalize_votes(votes, total_votes)
        
        return votes, {
            'success': result.success,
            'message': result.message,
            'fun': result.fun,
            'nit': result.nit
        }
    
    def estimate_votes_percent_method(self,
                                      judge_scores: np.ndarray,
                                      eliminated_idx: int,
                                      total_votes: float = 1e6) -> Tuple[np.ndarray, Dict]:
        """
        百分比法下的约束优化估计
        
        约束：被淘汰选手的综合百分比得分最低
        
        Args:
            judge_scores: 评委得分数组
            eliminated_idx: 被淘汰选手的索引
            total_votes: 总投票数
        
        Returns:
            (估计投票, 优化结果信息)
        """
        n = len(judge_scores)
        prior = self.compute_prior_votes(judge_scores, total_votes)
        
        # 目标函数
        def objective(votes):
            log_votes = np.log(votes + 1)
            log_prior = np.log(prior + 1)
            return np.sum((log_votes - log_prior) ** 2)
        
        # 约束：被淘汰者的综合百分比 <= 其他所有人
        def elimination_constraint(votes):
            combined = compute_percent_combined_score(judge_scores, votes)
            eliminated_score = combined[eliminated_idx]
            other_scores = np.delete(combined, eliminated_idx)
            # 被淘汰者得分应该 <= min(其他人得分)
            # 返回 min(others) - eliminated_score >= 0
            return np.min(other_scores) - eliminated_score
        
        x0 = prior.copy()
        bounds = [(100, total_votes * 0.8) for _ in range(n)]
        
        constraints = [
            {'type': 'ineq', 'fun': elimination_constraint},
            {'type': 'eq', 'fun': lambda v: np.sum(v) - total_votes}
        ]
        
        result = minimize(
            objective,
            x0,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        votes = result.x
        votes = normalize_votes(votes, total_votes)
        
        return votes, {
            'success': result.success,
            'message': result.message,
            'fun': result.fun,
            'nit': result.nit
        }
    
    def estimate_votes_for_week(self,
                                contestants: pd.DataFrame,
                                eliminated_name: str,
                                season: int,
                                method: str = 'auto',
                                total_votes: float = 1e6) -> Tuple[np.ndarray, Dict]:
        """
        估计某周的观众投票
        
        Args:
            contestants: 该周参赛选手DataFrame
            eliminated_name: 被淘汰选手姓名
            season: 赛季编号
            method: 方法类型 ('rank', 'percent', 'auto')
            total_votes: 总投票数
        
        Returns:
            (估计投票数组, 优化结果)
        """
        scores = contestants['total_score'].values
        names = contestants['celebrity_name'].values
        
        # 找到被淘汰者的索引
        eliminated_idx = None
        for i, name in enumerate(names):
            if name == eliminated_name:
                eliminated_idx = i
                break
        
        if eliminated_idx is None:
            # 如果找不到淘汰者，使用先验估计
            prior = self.compute_prior_votes(scores, total_votes)
            return prior, {'success': False, 'message': 'Eliminated contestant not found'}
        
        # 自动选择方法
        if method == 'auto':
            if season in [1, 2] or season >= 28:
                method = 'rank'
            else:
                method = 'percent'
        
        # 执行优化
        if method == 'rank':
            return self.estimate_votes_rank_method(scores, eliminated_idx, total_votes)
        else:
            return self.estimate_votes_percent_method(scores, eliminated_idx, total_votes)
    
    def estimate_all_weeks(self,
                          season_week_data: Dict,
                          elimination_info: pd.DataFrame,
                          total_votes: float = 1e6,
                          verbose: bool = True) -> Dict:
        """
        估计所有赛季所有周次的投票
        
        Args:
            season_week_data: 赛季-周次数据
            elimination_info: 淘汰信息
            total_votes: 每周总投票数
            verbose: 是否打印进度
        
        Returns:
            估计结果字典
        """
        self.estimates = {}
        self.optimization_results = {}
        
        success_count = 0
        total_count = 0
        
        for (season, week), contestants in season_week_data.items():
            if len(contestants) <= 1:
                continue
            
            # 获取该周的淘汰者
            elim_record = elimination_info[
                (elimination_info['season'] == season) &
                (elimination_info['week'] == week)
            ]
            
            if len(elim_record) == 0:
                # 无淘汰的周次，使用先验估计
                scores = contestants['total_score'].values
                votes = self.compute_prior_votes(scores, total_votes)
                opt_result = {'success': True, 'message': 'No elimination this week'}
            else:
                eliminated_name = elim_record.iloc[0]['eliminated_name']
                votes, opt_result = self.estimate_votes_for_week(
                    contestants, eliminated_name, season, 'auto', total_votes
                )
            
            names = contestants['celebrity_name'].values
            scores = contestants['total_score'].values
            
            self.estimates[(season, week)] = {
                'names': names,
                'scores': scores,
                'votes': votes
            }
            self.optimization_results[(season, week)] = opt_result
            
            if opt_result['success']:
                success_count += 1
            total_count += 1
        
        if verbose:
            print(f"优化完成: {success_count}/{total_count} 成功")
        
        return self.estimates
    
    def generate_vote_samples(self,
                              judge_scores: np.ndarray,
                              eliminated_idx: int,
                              season: int,
                              n_samples: int = 100,
                              total_votes: float = 1e6) -> np.ndarray:
        """
        生成满足约束的投票样本（用于不确定性量化）
        
        Args:
            judge_scores: 评委得分
            eliminated_idx: 被淘汰者索引
            season: 赛季编号
            n_samples: 样本数量
            total_votes: 总投票数
        
        Returns:
            样本数组 (n_samples, n_contestants)
        """
        n = len(judge_scores)
        samples = []
        
        # 确定方法
        if season in [1, 2] or season >= 28:
            method = 'rank'
        else:
            method = 'percent'
        
        for _ in range(n_samples):
            # 随机扰动先验权重
            perturbed_weight = self.prior_weight * np.random.uniform(0.8, 1.2)
            original_weight = self.prior_weight
            self.prior_weight = perturbed_weight
            
            try:
                if method == 'rank':
                    votes, _ = self.estimate_votes_rank_method(
                        judge_scores, eliminated_idx, total_votes
                    )
                else:
                    votes, _ = self.estimate_votes_percent_method(
                        judge_scores, eliminated_idx, total_votes
                    )
                samples.append(votes)
            except:
                pass
            
            self.prior_weight = original_weight
        
        if len(samples) == 0:
            # 如果采样失败，返回基于先验的样本
            prior = self.compute_prior_votes(judge_scores, total_votes)
            for _ in range(n_samples):
                noise = np.random.normal(1, 0.1, n)
                samples.append(normalize_votes(prior * noise, total_votes))
        
        return np.array(samples)
