# -*- coding: utf-8 -*-
"""
争议案例分析
分析评委与观众意见存在分歧的特定选手
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .voting_methods import RankMethod, PercentMethod


class ControversyAnalyzer:
    """
    争议案例分析器
    
    分析：
    1. Jerry Rice (S2) - 5周最低评委分仍获亚军
    2. Billy Ray Cyrus (S4) - 6周最低评委分仍获第5名
    3. Bristol Palin (S11) - 12次最低评委分仍获季军
    4. Bobby Bones (S27) - 持续低分仍获冠军
    """
    
    def __init__(self, vote_estimates_df: pd.DataFrame, original_data_df: pd.DataFrame):
        self.vote_estimates = vote_estimates_df
        self.original_data = original_data_df
        
    def get_celebrity_trajectory(self, celebrity_name: str, season: int) -> pd.DataFrame:
        """
        获取选手在特定赛季的完整比赛轨迹
        """
        trajectory = self.vote_estimates[
            (self.vote_estimates['celebrity'].str.contains(celebrity_name, case=False, na=False)) &
            (self.vote_estimates['season'] == season)
        ].copy()
        
        return trajectory.sort_values('week')
    
    def analyze_judge_rank_by_week(self, season: int) -> pd.DataFrame:
        """
        分析某赛季每周的评委得分排名
        
        Returns:
        --------
        DataFrame with weekly judge score rankings for all contestants
        """
        season_data = self.vote_estimates[self.vote_estimates['season'] == season].copy()
        
        results = []
        for week in sorted(season_data['week'].unique()):
            week_data = season_data[season_data['week'] == week].copy()
            
            # 计算该周评委得分排名（1=最高）
            week_data['judge_rank'] = week_data['total_score'].rank(ascending=False, method='min')
            
            for _, row in week_data.iterrows():
                results.append({
                    'season': season,
                    'week': week,
                    'celebrity': row['celebrity'],
                    'judge_score': row['total_score'],
                    'judge_rank': row['judge_rank'],
                    'num_contestants': len(week_data),
                    'is_lowest': row['judge_rank'] == len(week_data),
                    'estimated_votes': row['estimated_votes'],
                    'is_eliminated': row['is_eliminated']
                })
        
        return pd.DataFrame(results)
    
    def analyze_controversy_case(self, celebrity_name: str, season: int) -> Dict:
        """
        深度分析单个争议案例
        """
        # 获取选手轨迹
        trajectory = self.get_celebrity_trajectory(celebrity_name, season)
        
        if len(trajectory) == 0:
            return {'error': f'未找到 {celebrity_name} 在第{season}季的数据'}
        
        # 获取该赛季的评委排名数据
        season_rankings = self.analyze_judge_rank_by_week(season)
        
        # 筛选该选手的排名记录
        celebrity_rankings = season_rankings[
            season_rankings['celebrity'].str.contains(celebrity_name, case=False, na=False)
        ]
        
        # 统计最低评委分周次
        lowest_score_weeks = celebrity_rankings[celebrity_rankings['is_lowest'] == True]['week'].tolist()
        
        # 模拟两种方法的结果
        method_comparison = self._simulate_both_methods(celebrity_name, season)
        
        # 获取实际最终名次
        original_record = self.original_data[
            (self.original_data['celebrity_name'].str.contains(celebrity_name, case=False, na=False)) &
            (self.original_data['season'] == season)
        ]
        actual_placement = original_record['placement'].values[0] if len(original_record) > 0 else None
        
        return {
            'celebrity': celebrity_name,
            'season': season,
            'actual_placement': actual_placement,
            'weeks_competed': len(trajectory),
            'lowest_judge_score_weeks': lowest_score_weeks,
            'num_lowest_weeks': len(lowest_score_weeks),
            'trajectory': trajectory,
            'weekly_rankings': celebrity_rankings,
            'method_comparison': method_comparison,
            'judge_tiebreaker_analysis': self._analyze_judge_tiebreaker_impact(celebrity_name, season)
        }
    
    def _simulate_both_methods(self, celebrity_name: str, season: int) -> Dict:
        """
        模拟两种方法对该选手的影响
        """
        season_data = self.vote_estimates[self.vote_estimates['season'] == season].copy()
        
        results = {
            'rank_method_would_eliminate': [],
            'percent_method_would_eliminate': [],
            'week_details': []
        }
        
        for week in sorted(season_data['week'].unique()):
            week_data = season_data[season_data['week'] == week]
            
            contestants = week_data['celebrity'].tolist()
            judge_scores = week_data['total_score'].values
            fan_votes = week_data['estimated_votes'].values
            
            # 排名法
            rank_eliminated, rank_df = RankMethod.determine_elimination(
                judge_scores, fan_votes, contestants
            )
            
            # 百分比法
            percent_eliminated, percent_df = PercentMethod.determine_elimination(
                judge_scores, fan_votes, contestants
            )
            
            # 检查该选手是否会被淘汰
            celebrity_in_week = any(celebrity_name.lower() in c.lower() for c in contestants)
            
            if celebrity_in_week:
                would_be_eliminated_rank = celebrity_name.lower() in rank_eliminated.lower()
                would_be_eliminated_percent = celebrity_name.lower() in percent_eliminated.lower()
                
                results['week_details'].append({
                    'week': week,
                    'rank_eliminated': rank_eliminated,
                    'percent_eliminated': percent_eliminated,
                    'celebrity_eliminated_by_rank': would_be_eliminated_rank,
                    'celebrity_eliminated_by_percent': would_be_eliminated_percent,
                    'methods_differ': rank_eliminated != percent_eliminated
                })
                
                if would_be_eliminated_rank:
                    results['rank_method_would_eliminate'].append(week)
                if would_be_eliminated_percent:
                    results['percent_method_would_eliminate'].append(week)
        
        return results
    
    def _analyze_judge_tiebreaker_impact(self, celebrity_name: str, season: int) -> Dict:
        """
        分析评委裁决机制（底部二人选择）对该选手的影响
        """
        season_data = self.vote_estimates[self.vote_estimates['season'] == season].copy()
        
        impact_analysis = {
            'weeks_in_bottom_two_rank': [],
            'weeks_in_bottom_two_percent': [],
            'potential_saves': [],
            'potential_eliminations': []
        }
        
        for week in sorted(season_data['week'].unique()):
            week_data = season_data[season_data['week'] == week]
            
            contestants = week_data['celebrity'].tolist()
            judge_scores = week_data['total_score'].values
            fan_votes = week_data['estimated_votes'].values
            
            # 获取底部两人
            rank_bottom_two, _ = RankMethod.get_bottom_two(judge_scores, fan_votes, contestants)
            percent_bottom_two, _ = PercentMethod.get_bottom_two(judge_scores, fan_votes, contestants)
            
            # 检查选手是否在底部两人中
            in_rank_bottom = any(celebrity_name.lower() in c.lower() for c in rank_bottom_two)
            in_percent_bottom = any(celebrity_name.lower() in c.lower() for c in percent_bottom_two)
            
            if in_rank_bottom:
                impact_analysis['weeks_in_bottom_two_rank'].append({
                    'week': week,
                    'bottom_two': rank_bottom_two
                })
            
            if in_percent_bottom:
                impact_analysis['weeks_in_bottom_two_percent'].append({
                    'week': week,
                    'bottom_two': percent_bottom_two
                })
        
        return impact_analysis
    
    def compare_controversy_cases(self) -> pd.DataFrame:
        """
        比较所有争议案例
        """
        cases = [
            ('Jerry Rice', 2),
            ('Billy Ray Cyrus', 4),
            ('Bristol Palin', 11),
            ('Bobby Bones', 27)
        ]
        
        results = []
        for name, season in cases:
            analysis = self.analyze_controversy_case(name, season)
            
            if 'error' not in analysis:
                method_comp = analysis['method_comparison']
                results.append({
                    'celebrity': name,
                    'season': season,
                    'actual_placement': analysis['actual_placement'],
                    'weeks_competed': analysis['weeks_competed'],
                    'num_lowest_judge_weeks': analysis['num_lowest_weeks'],
                    'would_be_eliminated_rank_weeks': len(method_comp['rank_method_would_eliminate']),
                    'would_be_eliminated_percent_weeks': len(method_comp['percent_method_would_eliminate']),
                    'rank_eliminates_earlier': len(method_comp['rank_method_would_eliminate']) > 0,
                    'percent_eliminates_earlier': len(method_comp['percent_method_would_eliminate']) > 0,
                    'weeks_in_bottom_two_rank': len(analysis['judge_tiebreaker_analysis']['weeks_in_bottom_two_rank']),
                    'weeks_in_bottom_two_percent': len(analysis['judge_tiebreaker_analysis']['weeks_in_bottom_two_percent'])
                })
        
        return pd.DataFrame(results)
    
    def identify_additional_controversies(self, min_lowest_weeks: int = 3) -> pd.DataFrame:
        """
        识别数据中的其他潜在争议案例
        
        Parameters:
        -----------
        min_lowest_weeks : int
            至少有多少周评委得分最低才算争议
        """
        all_controversies = []
        
        for season in self.vote_estimates['season'].unique():
            season_rankings = self.analyze_judge_rank_by_week(season)
            
            # 统计每位选手获得最低评委分的次数
            lowest_counts = season_rankings[season_rankings['is_lowest'] == True].groupby('celebrity').size()
            
            for celebrity, count in lowest_counts.items():
                if count >= min_lowest_weeks:
                    # 获取该选手的最终名次
                    orig = self.original_data[
                        (self.original_data['celebrity_name'].str.contains(celebrity.split()[0], case=False, na=False)) &
                        (self.original_data['season'] == season)
                    ]
                    
                    placement = orig['placement'].values[0] if len(orig) > 0 else None
                    
                    all_controversies.append({
                        'celebrity': celebrity,
                        'season': season,
                        'lowest_judge_weeks': count,
                        'final_placement': placement,
                        'is_finalist': placement is not None and placement <= 3
                    })
        
        df = pd.DataFrame(all_controversies)
        
        # 按照争议程度排序（最低周次多且名次好的更具争议）
        if len(df) > 0:
            df['controversy_score'] = df['lowest_judge_weeks'] * (1 / df['final_placement'].fillna(10))
            df = df.sort_values('controversy_score', ascending=False)
        
        return df
