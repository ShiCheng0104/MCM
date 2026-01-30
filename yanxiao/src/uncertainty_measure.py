"""
不确定性度量模块
量化投票估计的确定性
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from .utils import compute_confidence_interval, compute_cv


class UncertaintyAnalyzer:
    """
    不确定性分析类
    度量投票估计的确定性和可信度
    """
    
    # 不确定性分类阈值
    CV_HIGH = 0.1  # CV < 0.1 -> 高确定性
    CV_MEDIUM = 0.3  # 0.1 <= CV < 0.3 -> 中等确定性
    # CV >= 0.3 -> 低确定性
    
    def __init__(self):
        """初始化不确定性分析器"""
        self.uncertainty_stats = {}
        self.summary = None
        
    def compute_uncertainty_from_samples(self,
                                         samples: np.ndarray,
                                         names: np.ndarray,
                                         confidence: float = 0.95) -> pd.DataFrame:
        """
        从样本计算不确定性统计量
        
        Args:
            samples: 样本数组 (n_samples, n_contestants)
            names: 选手姓名数组
            confidence: 置信水平
        
        Returns:
            不确定性统计DataFrame
        """
        n_contestants = samples.shape[1]
        results = []
        
        for i in range(n_contestants):
            contestant_samples = samples[:, i]
            
            mean = np.mean(contestant_samples)
            std = np.std(contestant_samples)
            cv = std / mean if mean > 0 else float('inf')
            
            lower, upper = compute_confidence_interval(contestant_samples, confidence)
            ci_width = upper - lower
            ci_relative_width = ci_width / mean if mean > 0 else float('inf')
            
            # 分类确定性等级
            if cv < self.CV_HIGH:
                certainty_level = 'High'
            elif cv < self.CV_MEDIUM:
                certainty_level = 'Medium'
            else:
                certainty_level = 'Low'
            
            results.append({
                'celebrity_name': names[i],
                'mean': mean,
                'std': std,
                'cv': cv,
                'ci_lower': lower,
                'ci_upper': upper,
                'ci_width': ci_width,
                'ci_relative_width': ci_relative_width,
                'certainty_level': certainty_level
            })
        
        return pd.DataFrame(results)
    
    def analyze_all_weeks(self,
                         estimates: Dict,
                         samples_dict: Dict,
                         confidence: float = 0.95) -> Dict:
        """
        分析所有周次的不确定性
        
        Args:
            estimates: 估计结果字典
            samples_dict: 样本字典
            confidence: 置信水平
        
        Returns:
            不确定性统计字典
        """
        self.uncertainty_stats = {}
        
        for (season, week), est in estimates.items():
            key = (season, week)
            
            if key in samples_dict:
                samples = samples_dict[key]
                names = est['names']
                
                stats_df = self.compute_uncertainty_from_samples(samples, names, confidence)
                stats_df['season'] = season
                stats_df['week'] = week
                
                self.uncertainty_stats[key] = stats_df
            else:
                # 如果没有样本，使用估计值的简单统计
                names = est['names']
                votes = est['votes']
                
                # 假设10%的相对不确定性
                std_assumed = votes * 0.1
                
                records = []
                for i in range(len(names)):
                    cv = 0.1  # 假设值
                    records.append({
                        'celebrity_name': names[i],
                        'mean': votes[i],
                        'std': std_assumed[i],
                        'cv': cv,
                        'ci_lower': votes[i] * 0.8,
                        'ci_upper': votes[i] * 1.2,
                        'ci_width': votes[i] * 0.4,
                        'ci_relative_width': 0.4,
                        'certainty_level': 'Medium',
                        'season': season,
                        'week': week
                    })
                
                self.uncertainty_stats[key] = pd.DataFrame(records)
        
        return self.uncertainty_stats
    
    def compute_summary(self) -> Dict:
        """
        计算不确定性汇总统计
        
        Returns:
            汇总统计字典
        """
        if not self.uncertainty_stats:
            raise ValueError("请先运行 analyze_all_weeks")
        
        # 合并所有结果
        all_stats = pd.concat(self.uncertainty_stats.values(), ignore_index=True)
        
        # 计算汇总统计
        self.summary = {
            'total_estimates': len(all_stats),
            'mean_cv': all_stats['cv'].mean(),
            'median_cv': all_stats['cv'].median(),
            'mean_ci_relative_width': all_stats['ci_relative_width'].mean(),
            'certainty_distribution': all_stats['certainty_level'].value_counts(normalize=True).to_dict(),
            'high_certainty_pct': (all_stats['certainty_level'] == 'High').mean(),
            'medium_certainty_pct': (all_stats['certainty_level'] == 'Medium').mean(),
            'low_certainty_pct': (all_stats['certainty_level'] == 'Low').mean()
        }
        
        # 按赛季统计
        season_stats = all_stats.groupby('season').agg({
            'cv': 'mean',
            'ci_relative_width': 'mean'
        }).round(4)
        self.summary['season_uncertainty'] = season_stats
        
        return self.summary
    
    def print_summary(self):
        """打印不确定性汇总"""
        if self.summary is None:
            self.compute_summary()
        
        print("\n" + "="*60)
        print("不确定性分析汇总")
        print("="*60)
        print(f"总估计数: {self.summary['total_estimates']}")
        print(f"平均变异系数 (CV): {self.summary['mean_cv']:.4f}")
        print(f"中位变异系数: {self.summary['median_cv']:.4f}")
        print(f"平均相对CI宽度: {self.summary['mean_ci_relative_width']:.4f}")
        print(f"\n确定性等级分布:")
        print(f"  高确定性: {self.summary['high_certainty_pct']:.2%}")
        print(f"  中等确定性: {self.summary['medium_certainty_pct']:.2%}")
        print(f"  低确定性: {self.summary['low_certainty_pct']:.2%}")
        print("="*60 + "\n")
    
    def get_low_certainty_cases(self, 
                                cv_threshold: float = None) -> pd.DataFrame:
        """
        获取低确定性案例
        
        Args:
            cv_threshold: CV阈值，默认使用类属性
        
        Returns:
            低确定性案例DataFrame
        """
        if not self.uncertainty_stats:
            raise ValueError("请先运行 analyze_all_weeks")
        
        threshold = cv_threshold or self.CV_MEDIUM
        all_stats = pd.concat(self.uncertainty_stats.values(), ignore_index=True)
        
        return all_stats[all_stats['cv'] >= threshold].sort_values('cv', ascending=False)
    
    def analyze_uncertainty_by_factors(self,
                                       weekly_data: pd.DataFrame) -> pd.DataFrame:
        """
        分析不确定性与各因素的关系
        
        Args:
            weekly_data: 周级别数据（包含选手特征）
        
        Returns:
            因素分析结果DataFrame
        """
        if not self.uncertainty_stats:
            raise ValueError("请先运行 analyze_all_weeks")
        
        all_stats = pd.concat(self.uncertainty_stats.values(), ignore_index=True)
        
        # 合并选手特征
        merged = all_stats.merge(
            weekly_data[['celebrity_name', 'season', 'week', 
                        'celebrity_industry', 'celebrity_age', 'total_score']],
            on=['celebrity_name', 'season', 'week'],
            how='left'
        )
        
        # 按行业分析
        industry_uncertainty = merged.groupby('celebrity_industry').agg({
            'cv': ['mean', 'std'],
            'ci_relative_width': 'mean'
        }).round(4)
        
        # 按得分区间分析
        merged['score_bin'] = pd.cut(merged['total_score'], 
                                     bins=[0, 20, 25, 30, 100],
                                     labels=['Low', 'Medium', 'High', 'Very High'])
        
        score_uncertainty = merged.groupby('score_bin', observed=False).agg({
            'cv': ['mean', 'std'],
            'ci_relative_width': 'mean'
        }).round(4)
        
        # 按周次分析
        week_uncertainty = merged.groupby('week').agg({
            'cv': 'mean',
            'ci_relative_width': 'mean'
        }).round(4)
        
        return {
            'by_industry': industry_uncertainty,
            'by_score': score_uncertainty,
            'by_week': week_uncertainty
        }
    
    def compute_sensitivity_index(self,
                                  samples: np.ndarray,
                                  scores: np.ndarray,
                                  method: str = 'rank') -> np.ndarray:
        """
        计算边际敏感度指数
        即投票变化对淘汰结果的敏感程度
        
        Args:
            samples: 投票样本 (n_samples, n_contestants)
            scores: 评委得分
            method: 方法类型
        
        Returns:
            敏感度数组
        """
        from .utils import compute_rank_combined_score, compute_percent_combined_score
        
        n_samples, n_contestants = samples.shape
        elimination_counts = np.zeros(n_contestants)
        
        for i in range(n_samples):
            votes = samples[i]
            
            if method == 'rank':
                combined = compute_rank_combined_score(scores, votes)
                eliminated = np.argmax(combined)
            else:
                combined = compute_percent_combined_score(scores, votes)
                eliminated = np.argmin(combined)
            
            elimination_counts[eliminated] += 1
        
        # 被淘汰概率
        elimination_prob = elimination_counts / n_samples
        
        # 敏感度：与最高概率的差距
        max_prob = np.max(elimination_prob)
        sensitivity = max_prob - elimination_prob
        
        return sensitivity


def run_uncertainty_analysis(estimates: Dict,
                            samples_dict: Dict,
                            weekly_data: pd.DataFrame,
                            verbose: bool = True) -> Tuple[UncertaintyAnalyzer, Dict]:
    """
    运行完整的不确定性分析
    
    Args:
        estimates: 投票估计结果
        samples_dict: 投票样本字典
        weekly_data: 周级别数据
        verbose: 是否打印结果
    
    Returns:
        (分析器对象, 汇总结果)
    """
    analyzer = UncertaintyAnalyzer()
    
    # 分析所有周次
    analyzer.analyze_all_weeks(estimates, samples_dict)
    summary = analyzer.compute_summary()
    
    if verbose:
        analyzer.print_summary()
    
    # 因素分析
    factor_analysis = analyzer.analyze_uncertainty_by_factors(weekly_data)
    summary['factor_analysis'] = factor_analysis
    
    # 低确定性案例
    low_certainty = analyzer.get_low_certainty_cases()
    summary['n_low_certainty'] = len(low_certainty)
    
    if verbose:
        print(f"低确定性案例数: {summary['n_low_certainty']}")
    
    return analyzer, summary
