"""
纯预测投票模型
不使用淘汰结果作为约束，真正预测观众投票

核心思想：
1. 基于评分 + 选手特征（舞伴、行业、年龄）预测投票
2. 使用贝叶斯模型估计的效应
3. 根据预测投票计算综合得分，预测淘汰
4. 与实际淘汰比较验证模型
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


class PredictiveVoteModel:
    """
    纯预测投票模型
    
    不同于约束优化模型，这个模型：
    - 不使用淘汰结果作为约束
    - 纯粹基于评分和选手特征预测投票
    - 可以真正验证预测准确性
    """
    
    def __init__(self, vote_scale: float = 1e6):
        """
        初始化预测模型
        
        Args:
            vote_scale: 投票规模（总投票数）
        """
        self.vote_scale = vote_scale
        self.is_fitted = False
        
        # 模型参数
        self.beta_0 = 10.0  # 基础log投票
        self.beta_score = 0.5  # 评分效应（正值：高分→高票）
        self.beta_age = 0.0  # 年龄效应
        self.partner_effects = {}
        self.season_effects = {}
        self.industry_effects = {}
        self.sigma = 1.0
        
        # 存储预测结果
        self.predictions = {}
        
    def set_params_from_bayesian(self, bayesian_model):
        """
        从贝叶斯模型加载参数
        
        Args:
            bayesian_model: 已拟合的贝叶斯模型
        """
        if bayesian_model.is_fitted:
            self.beta_0 = bayesian_model.beta_0_mean
            self.beta_score = bayesian_model.beta_score_mean
            self.beta_age = bayesian_model.beta_age_mean
            self.partner_effects = bayesian_model.partner_effects.copy()
            self.season_effects = bayesian_model.season_effects.copy()
            self.industry_effects = bayesian_model.industry_effects.copy()
            if hasattr(bayesian_model, 'sigma_mean'):
                self.sigma = bayesian_model.sigma_mean
            self.is_fitted = True
            print("  已从贝叶斯模型加载参数")
        else:
            print("  警告：贝叶斯模型未拟合，使用默认参数")
    
    def predict_votes(self, 
                     week_data: pd.DataFrame,
                     season: int,
                     week: int) -> np.ndarray:
        """
        预测一周内所有选手的投票
        
        Args:
            week_data: 该周的选手数据
            season: 赛季
            week: 周次
            
        Returns:
            预测的投票数组（归一化）
        """
        n = len(week_data)
        log_votes = np.zeros(n)
        
        for i, (_, row) in enumerate(week_data.iterrows()):
            # 基础效应
            mu = self.beta_0
            
            # 评分效应（使用标准化得分）
            if 'score_normalized' in row:
                mu += self.beta_score * row['score_normalized']
            elif 'total_score' in row:
                # 如果没有标准化得分，使用原始得分的简单标准化
                mu += self.beta_score * (row['total_score'] / 30.0 - 1)
            
            # 年龄效应
            if 'age_normalized' in row and pd.notna(row['age_normalized']):
                mu += self.beta_age * row['age_normalized']
            
            # 舞伴效应
            partner = row.get('pro', row.get('partner', None))
            if partner and partner in self.partner_effects:
                mu += self.partner_effects[partner]
            
            # 赛季效应
            if season in self.season_effects:
                mu += self.season_effects[season]
            
            # 行业效应
            industry = row.get('industry', None)
            if industry and industry in self.industry_effects:
                mu += self.industry_effects[industry]
            
            log_votes[i] = mu
        
        # 转换为投票并归一化
        votes = np.exp(log_votes)
        votes = votes / votes.sum()  # 归一化为投票份额
        
        return votes
    
    def predict_elimination(self,
                           week_data: pd.DataFrame,
                           season: int,
                           week: int,
                           method: str = 'percent') -> Dict:
        """
        预测该周谁会被淘汰
        
        Args:
            week_data: 该周的选手数据
            season: 赛季
            week: 周次
            method: 投票方法 ('percent' 或 'rank')
            
        Returns:
            预测结果字典
        """
        # 预测投票
        predicted_votes = self.predict_votes(week_data, season, week)
        
        # 获取评委得分
        if 'total_score' in week_data.columns:
            judge_scores = week_data['total_score'].values
        else:
            judge_scores = np.ones(len(week_data)) * 25  # 默认得分
        
        # 计算综合得分
        if method == 'rank':
            combined_scores = compute_rank_combined_score(judge_scores, predicted_votes)
            predicted_elim_idx = get_eliminated_index_rank(combined_scores)
        else:
            combined_scores = compute_percent_combined_score(judge_scores, predicted_votes)
            predicted_elim_idx = get_eliminated_index_percent(combined_scores)
        
        # 获取选手名字
        if 'celebrity' in week_data.columns:
            names = week_data['celebrity'].values
        else:
            names = week_data.index.values
        
        # 按综合得分排序（低分先淘汰）
        sorted_indices = np.argsort(combined_scores)
        
        return {
            'season': season,
            'week': week,
            'method': method,
            'predicted_votes': predicted_votes,
            'combined_scores': combined_scores,
            'predicted_elimination_idx': predicted_elim_idx,
            'predicted_elimination_name': names[predicted_elim_idx] if predicted_elim_idx < len(names) else None,
            'ranking': [(names[i], combined_scores[i]) for i in sorted_indices],
            'bottom_2_idx': sorted_indices[:2].tolist(),
            'bottom_2_names': [names[i] for i in sorted_indices[:2]]
        }
    
    def predict_all_weeks(self,
                         weekly_data: pd.DataFrame,
                         elimination_info: pd.DataFrame) -> pd.DataFrame:
        """
        预测所有周次的淘汰结果
        
        Args:
            weekly_data: 所有周级别数据
            elimination_info: 淘汰信息
            
        Returns:
            预测结果DataFrame
        """
        results = []
        
        # 按赛季-周次分组
        grouped = weekly_data.groupby(['season', 'week'])
        
        for (season, week), week_data in grouped:
            # 确定投票方法
            if season in [1, 2] or season >= 28:
                method = 'rank'
            else:
                method = 'percent'
            
            # 预测
            pred = self.predict_elimination(week_data, season, week, method)
            
            # 查找实际淘汰者
            elim_mask = (elimination_info['season'] == season) & \
                       (elimination_info['week'] == week)
            actual_elim = elimination_info[elim_mask]
            
            if len(actual_elim) > 0:
                actual_names = actual_elim['eliminated_name'].tolist()
                # 检查预测是否正确
                is_correct = pred['predicted_elimination_name'] in actual_names
                # 检查是否在底2
                in_bottom_2 = any(name in pred['bottom_2_names'] for name in actual_names)
            else:
                actual_names = []
                is_correct = None
                in_bottom_2 = None
            
            results.append({
                'season': season,
                'week': week,
                'method': method,
                'predicted_elim': pred['predicted_elimination_name'],
                'actual_elim': actual_names[0] if actual_names else None,
                'is_correct': is_correct,
                'in_bottom_2': in_bottom_2,
                'n_contestants': len(week_data)
            })
        
        return pd.DataFrame(results)
    
    def evaluate_accuracy(self, predictions_df: pd.DataFrame) -> Dict:
        """
        评估预测准确率
        
        Args:
            predictions_df: 预测结果DataFrame
            
        Returns:
            准确率统计
        """
        # 过滤有效预测（有实际淘汰记录的）
        valid = predictions_df[predictions_df['is_correct'].notna()]
        
        if len(valid) == 0:
            return {'elimination_accuracy': 0, 'bottom2_accuracy': 0, 'n_predictions': 0}
        
        elim_correct = valid['is_correct'].sum()
        bottom2_correct = valid['in_bottom_2'].sum()
        n = len(valid)
        
        # 按方法分组
        by_method = valid.groupby('method').agg({
            'is_correct': ['sum', 'count', 'mean'],
            'in_bottom_2': 'mean'
        })
        
        return {
            'elimination_accuracy': elim_correct / n,
            'bottom2_accuracy': bottom2_correct / n,
            'n_predictions': n,
            'by_method': by_method
        }
    
    def generate_vote_samples(self,
                             week_data: pd.DataFrame,
                             season: int,
                             week: int,
                             n_samples: int = 100) -> np.ndarray:
        """
        生成投票的不确定性样本
        
        Args:
            week_data: 该周的选手数据
            season: 赛季
            week: 周次
            n_samples: 样本数量
            
        Returns:
            投票样本数组 (n_samples, n_contestants)
        """
        n = len(week_data)
        samples = np.zeros((n_samples, n))
        
        # 获取点估计
        mean_votes = self.predict_votes(week_data, season, week)
        
        # 添加不确定性
        for s in range(n_samples):
            # 在log空间添加噪声
            log_votes = np.log(mean_votes + 1e-10)
            log_votes_noisy = log_votes + np.random.normal(0, self.sigma * 0.3, n)
            votes_noisy = np.exp(log_votes_noisy)
            samples[s] = votes_noisy / votes_noisy.sum()
        
        return samples
