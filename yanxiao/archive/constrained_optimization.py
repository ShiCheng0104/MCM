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
    3. 先验可以融入贝叶斯模型估计的效应（舞伴、行业等）
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
        
        # 贝叶斯模型参数（可选）
        self.bayesian_params = None
        
    def set_bayesian_params(self, bayesian_model):
        """
        从贝叶斯模型中提取参数用于改进先验
        
        Args:
            bayesian_model: 已拟合的贝叶斯模型
        """
        if bayesian_model.is_fitted:
            self.bayesian_params = {
                'beta_0': bayesian_model.beta_0_mean,
                'beta_score': bayesian_model.beta_score_mean,
                'beta_age': bayesian_model.beta_age_mean,
                'partner_effects': bayesian_model.partner_effects,
                'season_effects': bayesian_model.season_effects,
                'industry_effects': bayesian_model.industry_effects
            }
            print("  已加载贝叶斯模型参数到约束优化模型")
        
    def compute_prior_votes(self, 
                           judge_scores: np.ndarray,
                           total_votes: float = 1e6,
                           contestants_info: pd.DataFrame = None) -> np.ndarray:
        """
        计算先验投票估计（基于评委得分和贝叶斯效应）
        
        Args:
            judge_scores: 评委得分
            total_votes: 总投票数
            contestants_info: 选手信息（用于获取舞伴、行业等）
        
        Returns:
            先验投票估计
        """
        n = len(judge_scores)
        
        # 基础先验：基于评分
        exp_scores = np.exp(self.prior_weight * judge_scores / judge_scores.max())
        base_prior = exp_scores / exp_scores.sum()
        
        # 如果有贝叶斯参数和选手信息，添加效应调整
        if self.bayesian_params is not None and contestants_info is not None:
            adjustments = np.zeros(n)
            
            for i, (_, row) in enumerate(contestants_info.iterrows()):
                # 舞伴效应
                partner = row.get('pro_name', None)
                if partner and partner in self.bayesian_params['partner_effects']:
                    adjustments[i] += self.bayesian_params['partner_effects'][partner]
                
                # 行业效应
                industry = row.get('industry', None)
                if industry and industry in self.bayesian_params['industry_effects']:
                    adjustments[i] += self.bayesian_params['industry_effects'][industry]
            
            # 将效应转换为投票调整（使用softmax）
            effect_weights = np.exp(adjustments)
            effect_weights = effect_weights / effect_weights.sum()
            
            # 混合基础先验和效应调整（70%评分 + 30%贝叶斯效应）
            prior = 0.7 * base_prior + 0.3 * effect_weights
        else:
            prior = base_prior
        
        return prior * total_votes
    
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
    
    def estimate_votes_for_week_multi(self,
                                      contestants: pd.DataFrame,
                                      eliminated_names: list,
                                      season: int,
                                      method: str = 'auto',
                                      total_votes: float = 1e6) -> Tuple[np.ndarray, Dict]:
        """
        估计某周的观众投票（支持多人淘汰）
        
        Args:
            contestants: 该周参赛选手DataFrame
            eliminated_names: 被淘汰选手姓名列表
            season: 赛季编号
            method: 方法类型 ('rank', 'percent', 'auto')
            total_votes: 总投票数
        
        Returns:
            (估计投票数组, 优化结果)
        """
        scores = contestants['total_score'].values
        names = contestants['celebrity_name'].values
        n = len(scores)
        
        # 找到所有被淘汰者的索引
        eliminated_indices = []
        for elim_name in eliminated_names:
            for i, name in enumerate(names):
                if name == elim_name:
                    eliminated_indices.append(i)
                    break
        
        if len(eliminated_indices) == 0:
            # 如果找不到淘汰者，使用先验估计
            prior = self.compute_prior_votes(scores, total_votes)
            return prior, {'success': False, 'message': 'Eliminated contestants not found'}
        
        # 自动选择方法
        if method == 'auto':
            if season in [1, 2] or season >= 28:
                method = 'rank'
            else:
                method = 'percent'
        
        prior = self.compute_prior_votes(scores, total_votes)
        
        # 目标函数
        def objective(votes):
            log_votes = np.log(votes + 1)
            log_prior = np.log(prior + 1)
            return np.sum((log_votes - log_prior) ** 2)
        
        if method == 'rank':
            # 排名法约束：所有被淘汰者的综合排名分 >= 其他所有人
            def elimination_constraint(votes):
                combined = compute_rank_combined_score(scores, votes)
                # 被淘汰者中最小的综合分数应该 >= 非淘汰者中最大的综合分数
                elim_scores = [combined[i] for i in eliminated_indices]
                other_indices = [i for i in range(n) if i not in eliminated_indices]
                if len(other_indices) == 0:
                    return 0
                other_scores = [combined[i] for i in other_indices]
                return min(elim_scores) - max(other_scores)
        else:
            # 百分比法约束：所有被淘汰者的综合百分比 <= 其他所有人
            def elimination_constraint(votes):
                combined = compute_percent_combined_score(scores, votes)
                # 被淘汰者中最大的综合分数应该 <= 非淘汰者中最小的综合分数
                elim_scores = [combined[i] for i in eliminated_indices]
                other_indices = [i for i in range(n) if i not in eliminated_indices]
                if len(other_indices) == 0:
                    return 0
                other_scores = [combined[i] for i in other_indices]
                return min(other_scores) - max(elim_scores)
        
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
            'nit': result.nit,
            'n_eliminated': len(eliminated_indices)
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
        self.samples = {}  # 添加样本存储
        
        success_count = 0
        total_count = 0
        
        for (season, week), contestants in season_week_data.items():
            if len(contestants) <= 1:
                continue
            
            # 获取该周的淘汰者（可能有多人）
            elim_record = elimination_info[
                (elimination_info['season'] == season) &
                (elimination_info['week'] == week)
            ]
            
            if len(elim_record) == 0:
                # 无淘汰的周次，使用先验估计（传入选手信息以使用贝叶斯效应）
                scores = contestants['total_score'].values
                votes = self.compute_prior_votes(scores, total_votes, contestants)
                opt_result = {'success': True, 'message': 'No elimination this week'}
            else:
                # 获取所有被淘汰者的名字
                eliminated_names = elim_record['eliminated_name'].tolist()
                votes, opt_result = self.estimate_votes_for_week_multi(
                    contestants, eliminated_names, season, 'auto', total_votes
                )
            
            names = contestants['celebrity_name'].values
            scores = contestants['total_score'].values
            
            self.estimates[(season, week)] = {
                'names': names,
                'scores': scores,
                'votes': votes
            }
            self.optimization_results[(season, week)] = opt_result
            
            # 存储元数据用于延迟样本生成（不在这里生成样本，太慢）
            if len(elim_record) > 0:
                eliminated_names = elim_record['eliminated_name'].tolist()
                eliminated_indices = []
                for en in eliminated_names:
                    idx = np.where(names == en)[0]
                    if len(idx) > 0:
                        eliminated_indices.append(idx[0])
                if eliminated_indices:
                    # 只存储元数据，不立即生成样本
                    self.samples[(season, week)] = {
                        'scores': scores,
                        'eliminated_indices': eliminated_indices,
                        'season': season,
                        'votes': votes  # 存储点估计作为中心
                    }
            
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

    def generate_vote_samples_multi(self,
                                    judge_scores: np.ndarray,
                                    eliminated_indices: List[int],
                                    season: int,
                                    n_samples: int = 100,
                                    total_votes: float = 1e6) -> np.ndarray:
        """
        生成满足多淘汰约束的投票样本（用于不确定性量化）
        
        使用参数扰动和噪声注入来生成具有真实变异性的样本
        
        Args:
            judge_scores: 评委得分
            eliminated_indices: 被淘汰者索引列表
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
        
        # 先获取基准估计
        if len(eliminated_indices) == 1:
            base_votes, _ = self.estimate_votes_rank_method(
                judge_scores, eliminated_indices[0], total_votes
            ) if method == 'rank' else self.estimate_votes_percent_method(
                judge_scores, eliminated_indices[0], total_votes
            )
        else:
            # 多淘汰：使用约束优化
            prior = self.compute_prior_votes(judge_scores, total_votes)
            base_votes = prior.copy()
        
        for i in range(n_samples):
            # 方法1：参数扰动
            perturbed_weight = self.prior_weight * np.random.uniform(0.7, 1.3)
            original_weight = self.prior_weight
            self.prior_weight = perturbed_weight
            
            try:
                if len(eliminated_indices) == 1:
                    if method == 'rank':
                        votes, _ = self.estimate_votes_rank_method(
                            judge_scores, eliminated_indices[0], total_votes
                        )
                    else:
                        votes, _ = self.estimate_votes_percent_method(
                            judge_scores, eliminated_indices[0], total_votes
                        )
                else:
                    # 多淘汰情况：在基准上添加噪声
                    noise_scale = 0.1 + 0.1 * np.random.random()  # 10%-20%噪声
                    noise = np.random.normal(1, noise_scale, n)
                    votes = normalize_votes(base_votes * noise, total_votes)
                    
                    # 确保淘汰者票数最低
                    non_elim_min = np.min([votes[j] for j in range(n) if j not in eliminated_indices])
                    for idx in eliminated_indices:
                        if votes[idx] >= non_elim_min:
                            votes[idx] = non_elim_min * np.random.uniform(0.7, 0.95)
                    votes = normalize_votes(votes, total_votes)
                
                samples.append(votes)
            except:
                # 失败时使用噪声扰动
                noise = np.random.normal(1, 0.15, n)
                noisy_votes = normalize_votes(base_votes * noise, total_votes)
                samples.append(noisy_votes)
            
            self.prior_weight = original_weight
        
        return np.array(samples)
