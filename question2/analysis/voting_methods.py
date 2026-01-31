# -*- coding: utf-8 -*-
"""
两种投票组合方法的实现
- 排名法 (Rank Method): 用于第1-2季和第28-34季
- 百分比法 (Percent Method): 用于第3-27季
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.stats import rankdata


class RankMethod:
    """
    排名法投票组合方法
    
    综合排名 = 评委得分排名 + 观众投票排名
    淘汰规则：综合排名最高者（排名数字越大表示越差）
    
    注意：排名从1开始，1表示最好
    """
    
    @staticmethod
    def calculate_combined_scores(judge_scores: np.ndarray, 
                                   fan_votes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算排名法下的综合得分
        
        Parameters:
        -----------
        judge_scores : np.ndarray
            评委总分数组
        fan_votes : np.ndarray
            观众投票数数组
            
        Returns:
        --------
        judge_ranks : np.ndarray
            评委得分排名（1=最高分）
        fan_ranks : np.ndarray
            观众投票排名（1=最多票）
        combined_ranks : np.ndarray
            综合排名分数（越低越好）
        """
        # 评委得分排名（得分越高排名越靠前，即数字越小）
        # 使用'max'方法处理平局：相同得分取较大排名
        judge_ranks = rankdata(-judge_scores, method='average')
        
        # 观众投票排名（票数越多排名越靠前）
        fan_ranks = rankdata(-fan_votes, method='average')
        
        # 综合排名 = 评委排名 + 观众排名（越小越好）
        combined_ranks = judge_ranks + fan_ranks
        
        return judge_ranks, fan_ranks, combined_ranks
    
    @staticmethod
    def determine_elimination(judge_scores: np.ndarray,
                               fan_votes: np.ndarray,
                               contestant_names: List[str]) -> Tuple[str, pd.DataFrame]:
        """
        确定排名法下应该淘汰的选手
        
        Returns:
        --------
        eliminated : str
            被淘汰选手的名字
        rankings_df : pd.DataFrame
            包含所有排名信息的DataFrame
        """
        judge_ranks, fan_ranks, combined_ranks = RankMethod.calculate_combined_scores(
            judge_scores, fan_votes
        )
        
        # 创建排名DataFrame
        rankings_df = pd.DataFrame({
            'contestant': contestant_names,
            'judge_score': judge_scores,
            'judge_rank': judge_ranks,
            'fan_votes': fan_votes,
            'fan_rank': fan_ranks,
            'combined_rank': combined_ranks
        })
        
        # 综合排名最高（数字最大）的被淘汰
        eliminated_idx = np.argmax(combined_ranks)
        eliminated = contestant_names[eliminated_idx]
        
        return eliminated, rankings_df
    
    @staticmethod
    def get_bottom_two(judge_scores: np.ndarray,
                        fan_votes: np.ndarray,
                        contestant_names: List[str]) -> Tuple[List[str], pd.DataFrame]:
        """
        获取排名最后的两位选手（用于评委裁决机制）
        """
        _, _, combined_ranks = RankMethod.calculate_combined_scores(judge_scores, fan_votes)
        
        # 获取综合排名最高的两位
        sorted_indices = np.argsort(-combined_ranks)  # 降序排列
        bottom_two_indices = sorted_indices[:2]
        bottom_two = [contestant_names[i] for i in bottom_two_indices]
        
        return bottom_two, combined_ranks


class PercentMethod:
    """
    百分比法投票组合方法
    
    综合百分比 = 评委得分百分比 + 观众投票百分比
    淘汰规则：综合百分比最低者
    """
    
    @staticmethod
    def calculate_combined_scores(judge_scores: np.ndarray,
                                   fan_votes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算百分比法下的综合得分
        
        Parameters:
        -----------
        judge_scores : np.ndarray
            评委总分数组
        fan_votes : np.ndarray
            观众投票数数组
            
        Returns:
        --------
        judge_percents : np.ndarray
            评委得分百分比
        fan_percents : np.ndarray
            观众投票百分比
        combined_percents : np.ndarray
            综合百分比（越高越好）
        """
        # 评委得分百分比
        total_judge = np.sum(judge_scores)
        judge_percents = judge_scores / total_judge if total_judge > 0 else np.zeros_like(judge_scores)
        
        # 观众投票百分比
        total_votes = np.sum(fan_votes)
        fan_percents = fan_votes / total_votes if total_votes > 0 else np.zeros_like(fan_votes)
        
        # 综合百分比
        combined_percents = judge_percents + fan_percents
        
        return judge_percents, fan_percents, combined_percents
    
    @staticmethod
    def determine_elimination(judge_scores: np.ndarray,
                               fan_votes: np.ndarray,
                               contestant_names: List[str]) -> Tuple[str, pd.DataFrame]:
        """
        确定百分比法下应该淘汰的选手
        
        Returns:
        --------
        eliminated : str
            被淘汰选手的名字
        rankings_df : pd.DataFrame
            包含所有百分比信息的DataFrame
        """
        judge_percents, fan_percents, combined_percents = PercentMethod.calculate_combined_scores(
            judge_scores, fan_votes
        )
        
        # 创建排名DataFrame
        rankings_df = pd.DataFrame({
            'contestant': contestant_names,
            'judge_score': judge_scores,
            'judge_percent': judge_percents,
            'fan_votes': fan_votes,
            'fan_percent': fan_percents,
            'combined_percent': combined_percents
        })
        
        # 综合百分比最低的被淘汰
        eliminated_idx = np.argmin(combined_percents)
        eliminated = contestant_names[eliminated_idx]
        
        return eliminated, rankings_df
    
    @staticmethod
    def get_bottom_two(judge_scores: np.ndarray,
                        fan_votes: np.ndarray,
                        contestant_names: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        获取综合百分比最低的两位选手（用于评委裁决机制）
        """
        _, _, combined_percents = PercentMethod.calculate_combined_scores(judge_scores, fan_votes)
        
        # 获取综合百分比最低的两位
        sorted_indices = np.argsort(combined_percents)  # 升序排列
        bottom_two_indices = sorted_indices[:2]
        bottom_two = [contestant_names[i] for i in bottom_two_indices]
        
        return bottom_two, combined_percents


class VotingMethodComparator:
    """
    两种投票方法的比较器
    """
    
    def __init__(self, vote_estimates_df: pd.DataFrame, original_data_df: pd.DataFrame):
        """
        Parameters:
        -----------
        vote_estimates_df : pd.DataFrame
            观众投票估计数据
        original_data_df : pd.DataFrame
            原始比赛数据
        """
        self.vote_estimates = vote_estimates_df
        self.original_data = original_data_df
        
    def compare_methods_for_week(self, season: int, week: int) -> Dict:
        """
        对比两种方法在特定赛季某周的结果
        
        Returns:
        --------
        comparison : dict
            包含两种方法结果的对比
        """
        # 获取该周的数据
        week_data = self.vote_estimates[
            (self.vote_estimates['season'] == season) & 
            (self.vote_estimates['week'] == week) &
            (self.vote_estimates['is_eliminated'] != True) |
            ((self.vote_estimates['season'] == season) & 
             (self.vote_estimates['week'] == week) &
             (self.vote_estimates['is_eliminated'] == True))
        ].copy()
        
        # 过滤掉已被淘汰的选手（之前周次淘汰的）
        week_data = self.vote_estimates[
            (self.vote_estimates['season'] == season) & 
            (self.vote_estimates['week'] == week)
        ].copy()
        
        if len(week_data) == 0:
            return None
            
        contestants = week_data['celebrity'].tolist()
        judge_scores = week_data['total_score'].values
        fan_votes = week_data['estimated_votes'].values
        
        # 排名法结果
        rank_eliminated, rank_df = RankMethod.determine_elimination(
            judge_scores, fan_votes, contestants
        )
        rank_bottom_two, _ = RankMethod.get_bottom_two(judge_scores, fan_votes, contestants)
        
        # 百分比法结果
        percent_eliminated, percent_df = PercentMethod.determine_elimination(
            judge_scores, fan_votes, contestants
        )
        percent_bottom_two, _ = PercentMethod.get_bottom_two(judge_scores, fan_votes, contestants)
        
        # 实际淘汰结果
        actual_eliminated = week_data[week_data['is_eliminated'] == True]['celebrity'].tolist()
        actual_eliminated = actual_eliminated[0] if actual_eliminated else None
        
        return {
            'season': season,
            'week': week,
            'contestants': contestants,
            'rank_method': {
                'eliminated': rank_eliminated,
                'bottom_two': rank_bottom_two,
                'details': rank_df
            },
            'percent_method': {
                'eliminated': percent_eliminated,
                'bottom_two': percent_bottom_two,
                'details': percent_df
            },
            'actual_eliminated': actual_eliminated,
            'methods_agree': rank_eliminated == percent_eliminated,
            'rank_matches_actual': rank_eliminated == actual_eliminated,
            'percent_matches_actual': percent_eliminated == actual_eliminated
        }
    
    def compare_all_seasons(self) -> pd.DataFrame:
        """
        对所有赛季应用两种方法并比较结果
        
        Returns:
        --------
        comparison_df : pd.DataFrame
            所有周次的方法比较结果
        """
        results = []
        
        seasons = self.vote_estimates['season'].unique()
        
        for season in sorted(seasons):
            season_data = self.vote_estimates[self.vote_estimates['season'] == season]
            weeks = season_data['week'].unique()
            
            for week in sorted(weeks):
                comparison = self.compare_methods_for_week(season, week)
                if comparison:
                    results.append({
                        'season': season,
                        'week': week,
                        'num_contestants': len(comparison['contestants']),
                        'rank_eliminated': comparison['rank_method']['eliminated'],
                        'percent_eliminated': comparison['percent_method']['eliminated'],
                        'actual_eliminated': comparison['actual_eliminated'],
                        'methods_agree': comparison['methods_agree'],
                        'rank_matches_actual': comparison['rank_matches_actual'],
                        'percent_matches_actual': comparison['percent_matches_actual']
                    })
        
        return pd.DataFrame(results)
    
    def analyze_fan_vote_bias(self, comparison_df: pd.DataFrame) -> Dict:
        """
        分析哪种方法更偏向观众投票
        
        通过分析低评委分选手在两种方法下的淘汰/晋级情况
        """
        # 统计两种方法的差异
        disagreement_df = comparison_df[comparison_df['methods_agree'] == False]
        
        # 分析差异情况
        bias_analysis = {
            'total_weeks': len(comparison_df),
            'disagreement_count': len(disagreement_df),
            'disagreement_rate': len(disagreement_df) / len(comparison_df) if len(comparison_df) > 0 else 0,
            'disagreement_by_season': disagreement_df.groupby('season').size().to_dict() if len(disagreement_df) > 0 else {}
        }
        
        return bias_analysis
