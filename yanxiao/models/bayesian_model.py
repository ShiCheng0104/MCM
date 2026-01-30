"""
贝叶斯层次模型
使用层次结构建模观众投票
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy import stats
from src.utils import normalize_votes


class BayesianVoteModel:
    """
    贝叶斯层次投票估计模型
    
    模型结构：
    log(V_{i,w}) ~ Normal(μ_{i,w}, σ²)
    μ_{i,w} = β₀ + β₁·S_{i,w} + β₂·Age_i + α_partner + γ_season
    
    由于完整的贝叶斯推断需要PyMC等库，这里使用简化的经验贝叶斯方法
    """
    
    def __init__(self, 
                 n_samples: int = 1000,
                 random_seed: int = 42):
        """
        初始化贝叶斯模型
        
        Args:
            n_samples: 采样数量
            random_seed: 随机种子
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 模型参数
        self.beta_0 = 10.0  # 截距（对数尺度）
        self.beta_score = 0.1  # 评委得分系数
        self.beta_age = -0.01  # 年龄系数
        self.sigma = 0.5  # 残差标准差
        
        # 随机效应
        self.partner_effects = {}
        self.season_effects = {}
        self.industry_effects = {}
        
        # 存储结果
        self.estimates = {}
        self.samples = {}
        
    def initialize_random_effects(self, data: pd.DataFrame):
        """
        初始化随机效应
        
        Args:
            data: 周级别数据
        """
        # 舞伴效应
        for partner in data['ballroom_partner'].unique():
            self.partner_effects[partner] = np.random.normal(0, 0.3)
        
        # 赛季效应
        for season in data['season'].unique():
            self.season_effects[season] = np.random.normal(0, 0.2)
        
        # 行业效应
        for industry in data['celebrity_industry'].dropna().unique():
            self.industry_effects[industry] = np.random.normal(0, 0.3)
    
    def compute_expected_log_votes(self,
                                   score: float,
                                   age: float,
                                   partner: str,
                                   season: int,
                                   industry: Optional[str] = None) -> float:
        """
        计算期望的对数投票数
        
        Args:
            score: 评委总分
            age: 选手年龄
            partner: 舞伴姓名
            season: 赛季编号
            industry: 行业类别
        
        Returns:
            期望对数投票
        """
        mu = self.beta_0
        mu += self.beta_score * score
        mu += self.beta_age * age
        
        # 添加随机效应
        if partner in self.partner_effects:
            mu += self.partner_effects[partner]
        
        if season in self.season_effects:
            mu += self.season_effects[season]
        
        if industry and industry in self.industry_effects:
            mu += self.industry_effects[industry]
        
        return mu
    
    def sample_votes(self,
                    contestants: pd.DataFrame,
                    n_samples: Optional[int] = None) -> np.ndarray:
        """
        从后验分布采样投票
        
        Args:
            contestants: 选手数据
            n_samples: 采样数量
        
        Returns:
            样本数组 (n_samples, n_contestants)
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        n_contestants = len(contestants)
        samples = np.zeros((n_samples, n_contestants))
        
        for i, (_, row) in enumerate(contestants.iterrows()):
            mu = self.compute_expected_log_votes(
                score=row['total_score'],
                age=row.get('celebrity_age', 35),
                partner=row['ballroom_partner'],
                season=row['season'],
                industry=row.get('celebrity_industry', None)
            )
            
            # 采样对数投票
            log_votes = np.random.normal(mu, self.sigma, n_samples)
            samples[:, i] = np.exp(log_votes)
        
        # 归一化每个样本
        for j in range(n_samples):
            samples[j] = normalize_votes(samples[j], total_votes=1e6)
        
        return samples
    
    def estimate_votes(self,
                      contestants: pd.DataFrame,
                      total_votes: float = 1e6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        估计投票及置信区间
        
        Args:
            contestants: 选手数据
            total_votes: 总投票数
        
        Returns:
            (均值, 下界, 上界)
        """
        samples = self.sample_votes(contestants)
        
        # 重新归一化
        for i in range(len(samples)):
            samples[i] = normalize_votes(samples[i], total_votes)
        
        mean_votes = samples.mean(axis=0)
        lower = np.percentile(samples, 2.5, axis=0)
        upper = np.percentile(samples, 97.5, axis=0)
        
        return mean_votes, lower, upper
    
    def fit(self, weekly_data: pd.DataFrame, elimination_info: pd.DataFrame):
        """
        拟合模型参数（简化的经验贝叶斯）
        
        Args:
            weekly_data: 周级别数据
            elimination_info: 淘汰信息
        """
        # 初始化随机效应
        self.initialize_random_effects(weekly_data)
        
        # 使用网格搜索优化参数
        best_accuracy = 0.0
        best_params = (self.beta_score, self.beta_age, self.sigma)
        
        for beta_score in np.arange(0.05, 0.2, 0.02):
            for beta_age in np.arange(-0.02, 0.01, 0.005):
                for sigma in np.arange(0.3, 0.8, 0.1):
                    self.beta_score = beta_score
                    self.beta_age = beta_age
                    self.sigma = sigma
                    
                    accuracy = self._evaluate_accuracy(weekly_data, elimination_info)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (beta_score, beta_age, sigma)
        
        self.beta_score, self.beta_age, self.sigma = best_params
        print(f"拟合完成: beta_score={self.beta_score:.3f}, "
              f"beta_age={self.beta_age:.3f}, sigma={self.sigma:.2f}")
        print(f"最佳准确率: {best_accuracy:.2%}")
    
    def _evaluate_accuracy(self, 
                          weekly_data: pd.DataFrame,
                          elimination_info: pd.DataFrame) -> float:
        """评估淘汰预测准确率"""
        correct = 0
        total = 0
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            if len(group) <= 1:
                continue
            
            elim = elimination_info[
                (elimination_info['season'] == season) &
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            actual_name = elim.iloc[0]['eliminated_name']
            
            # 预测
            mean_votes, _, _ = self.estimate_votes(group)
            scores = group['total_score'].values
            
            # 使用自动选择的方法
            if season in [1, 2] or season >= 28:
                # 排名法
                from src.utils import compute_rank_combined_score, get_eliminated_index_rank
                combined = compute_rank_combined_score(scores, mean_votes)
                pred_idx = get_eliminated_index_rank(combined)
            else:
                # 百分比法
                from src.utils import compute_percent_combined_score, get_eliminated_index_percent
                combined = compute_percent_combined_score(scores, mean_votes)
                pred_idx = get_eliminated_index_percent(combined)
            
            pred_name = group.iloc[pred_idx]['celebrity_name']
            
            if pred_name == actual_name:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def estimate_all_weeks(self,
                          season_week_data: Dict,
                          total_votes: float = 1e6) -> Dict:
        """
        估计所有周次的投票
        
        Args:
            season_week_data: 赛季-周次数据
            total_votes: 每周总投票数
        
        Returns:
            估计结果字典
        """
        self.estimates = {}
        self.samples = {}
        
        for (season, week), contestants in season_week_data.items():
            if len(contestants) <= 1:
                continue
            
            samples = self.sample_votes(contestants)
            mean_votes, lower, upper = self.estimate_votes(contestants, total_votes)
            
            self.estimates[(season, week)] = {
                'names': contestants['celebrity_name'].values,
                'scores': contestants['total_score'].values,
                'votes': mean_votes,
                'lower': lower,
                'upper': upper
            }
            self.samples[(season, week)] = samples
        
        return self.estimates
    
    def get_uncertainty_stats(self, season: int, week: int) -> Dict:
        """
        获取某周的不确定性统计
        
        Args:
            season: 赛季
            week: 周次
        
        Returns:
            不确定性统计字典
        """
        key = (season, week)
        if key not in self.estimates:
            return {}
        
        est = self.estimates[key]
        samples = self.samples.get(key, None)
        
        result = {
            'names': est['names'],
            'mean': est['votes'],
            'lower': est['lower'],
            'upper': est['upper'],
            'ci_width': est['upper'] - est['lower']
        }
        
        if samples is not None:
            result['std'] = samples.std(axis=0)
            result['cv'] = result['std'] / result['mean']
        
        return result
