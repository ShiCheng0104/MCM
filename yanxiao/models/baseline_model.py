"""
基线模型
基于评委得分的简单比例估计观众投票
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from src.utils import (
    compute_rank_combined_score,
    compute_percent_combined_score,
    get_eliminated_index_rank,
    get_eliminated_index_percent,
    normalize_votes
)


class BaselineModel:
    """
    基线投票估计模型
    
    假设：观众投票与评委得分成正比
    V_i ∝ S_i^α，其中α为可调参数
    """
    
    def __init__(self, alpha: float = 1.0, noise_level: float = 0.1):
        """
        初始化基线模型
        
        Args:
            alpha: 评委得分对投票的幂次影响
            noise_level: 噪声水平（相对标准差）
        """
        self.alpha = alpha
        self.noise_level = noise_level
        self.estimates = {}
        
    def estimate_votes(self, 
                      judge_scores: np.ndarray,
                      total_votes: float = 1e6) -> np.ndarray:
        """
        基于评委得分估计观众投票
        
        Args:
            judge_scores: 评委得分数组
            total_votes: 假设的总投票数
        
        Returns:
            估计的投票数数组
        """
        # 基础估计：得分的α次幂
        base_votes = np.power(judge_scores, self.alpha)
        
        # 归一化到总投票数
        votes = normalize_votes(base_votes, total_votes)
        
        return votes
    
    def estimate_votes_with_noise(self,
                                  judge_scores: np.ndarray,
                                  total_votes: float = 1e6,
                                  n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带噪声的投票估计（用于不确定性量化）
        
        Args:
            judge_scores: 评委得分数组
            total_votes: 假设的总投票数
            n_samples: 采样数量
        
        Returns:
            (均值估计, 下界, 上界)
        """
        n_contestants = len(judge_scores)
        samples = np.zeros((n_samples, n_contestants))
        
        for i in range(n_samples):
            # 添加随机噪声
            noise = np.random.normal(1, self.noise_level, n_contestants)
            noise = np.clip(noise, 0.5, 1.5)  # 限制噪声范围
            
            base_votes = np.power(judge_scores, self.alpha) * noise
            samples[i] = normalize_votes(base_votes, total_votes)
        
        mean_votes = samples.mean(axis=0)
        lower = np.percentile(samples, 2.5, axis=0)
        upper = np.percentile(samples, 97.5, axis=0)
        
        return mean_votes, lower, upper
    
    def fit_alpha(self,
                  season_week_data: Dict,
                  elimination_info: pd.DataFrame,
                  method: str = 'rank') -> float:
        """
        通过网格搜索拟合最优alpha参数
        
        Args:
            season_week_data: 赛季-周次数据字典
            elimination_info: 淘汰信息DataFrame
            method: 投票计算方法 ('rank' 或 'percent')
        
        Returns:
            最优alpha值
        """
        alphas = np.arange(0.5, 2.1, 0.1)
        best_alpha = 1.0
        best_accuracy = 0.0
        
        for alpha in alphas:
            self.alpha = alpha
            accuracy = self._compute_elimination_accuracy(
                season_week_data, elimination_info, method
            )
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_alpha = alpha
        
        self.alpha = best_alpha
        print(f"最优alpha: {best_alpha:.2f}, 准确率: {best_accuracy:.2%}")
        return best_alpha
    
    def _compute_elimination_accuracy(self,
                                       season_week_data: Dict,
                                       elimination_info: pd.DataFrame,
                                       method: str) -> float:
        """
        计算淘汰预测准确率
        
        Args:
            season_week_data: 赛季-周次数据
            elimination_info: 淘汰信息
            method: 方法类型
        
        Returns:
            准确率
        """
        correct = 0
        total = 0
        
        for (season, week), contestants in season_week_data.items():
            if len(contestants) <= 1:
                continue
            
            # 获取实际淘汰者
            actual_elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(actual_elim) == 0:
                continue
            
            actual_name = actual_elim.iloc[0]['eliminated_name']
            
            # 预测淘汰者
            scores = contestants['total_score'].values
            votes = self.estimate_votes(scores)
            
            if method == 'rank':
                combined = compute_rank_combined_score(scores, votes)
                pred_idx = get_eliminated_index_rank(combined)
            else:
                combined = compute_percent_combined_score(scores, votes)
                pred_idx = get_eliminated_index_percent(combined)
            
            pred_name = contestants.iloc[pred_idx]['celebrity_name']
            
            if pred_name == actual_name:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def estimate_all_weeks(self,
                          season_week_data: Dict,
                          total_votes: float = 1e6) -> Dict:
        """
        估计所有赛季所有周次的投票
        
        Args:
            season_week_data: 赛季-周次数据
            total_votes: 每周总投票数
        
        Returns:
            估计结果字典
        """
        self.estimates = {}
        
        for (season, week), contestants in season_week_data.items():
            scores = contestants['total_score'].values
            names = contestants['celebrity_name'].values
            
            votes = self.estimate_votes(scores, total_votes)
            
            self.estimates[(season, week)] = {
                'names': names,
                'scores': scores,
                'votes': votes
            }
        
        return self.estimates
