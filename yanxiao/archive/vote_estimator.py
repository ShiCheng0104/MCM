"""
投票估计器
整合多种模型，提供统一接口
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


class VoteEstimator:
    """
    投票估计器主类
    整合基线模型、约束优化模型和贝叶斯模型
    """
    
    def __init__(self, 
                 model_type: str = 'constrained',
                 config: Dict = None):
        """
        初始化投票估计器
        
        Args:
            model_type: 模型类型 ('baseline', 'constrained', 'bayesian', 'ensemble')
            config: 模型配置参数
        """
        # 延迟导入以避免循环导入
        from models.baseline_model import BaselineModel
        from models.constrained_optimization import ConstrainedOptimizationModel
        from models.bayesian_model import BayesianVoteModel
        
        self.model_type = model_type
        self.config = config or {}
        
        # 初始化各模型
        self.baseline = BaselineModel(
            alpha=self.config.get('baseline_alpha', 1.0),
            noise_level=self.config.get('noise_level', 0.1)
        )
        
        self.constrained = ConstrainedOptimizationModel(
            method=self.config.get('opt_method', 'SLSQP'),
            max_iter=self.config.get('max_iter', 1000),
            prior_weight=self.config.get('prior_weight', 0.5)
        )
        
        self.bayesian = BayesianVoteModel(
            n_samples=self.config.get('n_samples', 1000),
            random_seed=self.config.get('random_seed', 42)
        )
        
        # 存储结果
        self.estimates = {}
        self.samples = {}
        self.model_weights = {'baseline': 0.2, 'constrained': 0.5, 'bayesian': 0.3}
        
    def fit(self, 
            weekly_data: pd.DataFrame,
            season_week_data: Dict,
            elimination_info: pd.DataFrame):
        """
        拟合模型
        
        Args:
            weekly_data: 周级别数据
            season_week_data: 赛季-周次数据
            elimination_info: 淘汰信息
        """
        print("正在拟合模型...")
        
        # 拟合基线模型
        print("  - 拟合基线模型...")
        self.baseline.fit_alpha(season_week_data, elimination_info)
        
        # 拟合贝叶斯模型
        print("  - 拟合贝叶斯模型...")
        self.bayesian.fit(weekly_data, elimination_info)
        
        print("模型拟合完成!")
    
    def estimate(self,
                season_week_data: Dict,
                elimination_info: pd.DataFrame,
                total_votes: float = 1e6) -> Dict:
        """
        估计所有周次的观众投票
        
        Args:
            season_week_data: 赛季-周次数据
            elimination_info: 淘汰信息
            total_votes: 每周总投票数
        
        Returns:
            估计结果字典
        """
        print(f"使用 {self.model_type} 模型进行估计...")
        
        if self.model_type == 'baseline':
            self.estimates = self.baseline.estimate_all_weeks(season_week_data, total_votes)
            
        elif self.model_type == 'constrained':
            # 将贝叶斯参数传递给约束优化模型以改进先验
            if self.bayesian.is_fitted:
                self.constrained.set_bayesian_params(self.bayesian)
            
            self.estimates = self.constrained.estimate_all_weeks(
                season_week_data, elimination_info, total_votes
            )
            # 获取约束优化模型生成的样本
            self.samples = getattr(self.constrained, 'samples', {})
            
        elif self.model_type == 'bayesian':
            self.estimates = self.bayesian.estimate_all_weeks(season_week_data, total_votes)
            self.samples = self.bayesian.samples
            
        elif self.model_type == 'ensemble':
            self.estimates = self._ensemble_estimate(
                season_week_data, elimination_info, total_votes
            )
        
        print(f"完成估计: {len(self.estimates)} 个周次")
        return self.estimates
    
    def _ensemble_estimate(self,
                          season_week_data: Dict,
                          elimination_info: pd.DataFrame,
                          total_votes: float = 1e6) -> Dict:
        """
        集成多个模型的估计结果
        
        Args:
            season_week_data: 赛季-周次数据
            elimination_info: 淘汰信息
            total_votes: 总投票数
        
        Returns:
            集成估计结果
        """
        # 获取各模型估计
        baseline_est = self.baseline.estimate_all_weeks(season_week_data, total_votes)
        constrained_est = self.constrained.estimate_all_weeks(
            season_week_data, elimination_info, total_votes
        )
        bayesian_est = self.bayesian.estimate_all_weeks(season_week_data, total_votes)
        
        # 加权集成
        ensemble_est = {}
        
        for key in season_week_data.keys():
            if key not in baseline_est or key not in constrained_est:
                continue
            
            names = baseline_est[key]['names']
            scores = baseline_est[key]['scores']
            
            v1 = baseline_est[key]['votes']
            v2 = constrained_est[key]['votes']
            v3 = bayesian_est[key]['votes'] if key in bayesian_est else v1
            
            # 加权平均
            ensemble_votes = (
                self.model_weights['baseline'] * v1 +
                self.model_weights['constrained'] * v2 +
                self.model_weights['bayesian'] * v3
            )
            
            # 归一化
            ensemble_votes = ensemble_votes / ensemble_votes.sum() * total_votes
            
            ensemble_est[key] = {
                'names': names,
                'scores': scores,
                'votes': ensemble_votes,
                'baseline_votes': v1,
                'constrained_votes': v2,
                'bayesian_votes': v3
            }
        
        return ensemble_est
    
    def get_estimate(self, season: int, week: int) -> Optional[Dict]:
        """
        获取特定周次的估计结果
        
        Args:
            season: 赛季
            week: 周次
        
        Returns:
            估计结果字典
        """
        key = (season, week)
        return self.estimates.get(key, None)
    
    def get_all_estimates_df(self) -> pd.DataFrame:
        """
        将所有估计结果转换为DataFrame
        
        Returns:
            估计结果DataFrame
        """
        records = []
        
        for (season, week), est in self.estimates.items():
            names = est['names']
            scores = est['scores']
            votes = est['votes']
            
            for i in range(len(names)):
                records.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': names[i],
                    'judge_score': scores[i],
                    'estimated_votes': votes[i]
                })
        
        return pd.DataFrame(records)
    
    def compare_models(self,
                      season_week_data: Dict,
                      elimination_info: pd.DataFrame,
                      total_votes: float = 1e6) -> pd.DataFrame:
        """
        比较不同模型的估计结果
        
        Args:
            season_week_data: 赛季-周次数据
            elimination_info: 淘汰信息
            total_votes: 总投票数
        
        Returns:
            模型比较结果DataFrame
        """
        # 获取各模型估计
        baseline_est = self.baseline.estimate_all_weeks(season_week_data, total_votes)
        constrained_est = self.constrained.estimate_all_weeks(
            season_week_data, elimination_info, total_votes
        )
        bayesian_est = self.bayesian.estimate_all_weeks(season_week_data, total_votes)
        
        # 计算相关性和差异
        results = []
        
        for key in season_week_data.keys():
            if key not in baseline_est or key not in constrained_est:
                continue
            
            season, week = key
            
            v1 = baseline_est[key]['votes']
            v2 = constrained_est[key]['votes']
            v3 = bayesian_est[key]['votes'] if key in bayesian_est else v1
            
            # 计算模型间相关性
            corr_12 = np.corrcoef(v1, v2)[0, 1]
            corr_13 = np.corrcoef(v1, v3)[0, 1]
            corr_23 = np.corrcoef(v2, v3)[0, 1]
            
            results.append({
                'season': season,
                'week': week,
                'n_contestants': len(v1),
                'corr_baseline_constrained': corr_12,
                'corr_baseline_bayesian': corr_13,
                'corr_constrained_bayesian': corr_23
            })
        
        return pd.DataFrame(results)
    
    def generate_samples(self,
                        season: int,
                        week: int,
                        n_samples: int = 1000) -> Optional[np.ndarray]:
        """
        生成投票样本用于不确定性分析
        
        Args:
            season: 赛季
            week: 周次
            n_samples: 样本数量
        
        Returns:
            样本数组 (n_samples, n_contestants)
        """
        key = (season, week)
        
        if self.model_type == 'bayesian' and key in self.samples:
            return self.samples[key]
        
        # 否则使用约束优化模型生成样本
        if key in self.constrained.estimates:
            est = self.constrained.estimates[key]
            scores = est['scores']
            
            # 找到被淘汰者索引（需要从外部获取）
            # 这里简化处理，使用最低得分者
            eliminated_idx = np.argmin(scores)
            
            return self.constrained.generate_vote_samples(
                scores, eliminated_idx, season, n_samples
            )
        
        return None
