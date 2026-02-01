"""
历史数据回测模块
对各投票系统进行回测，比较结果
"""
import pandas as pd
import numpy as np
from data_loader import prepare_weekly_data, load_vote_estimates
from voting_systems import get_all_systems
from config import RANDOM_SEED

np.random.seed(RANDOM_SEED)


class SeasonSimulator:
    """赛季模拟器"""
    
    def __init__(self, weekly_data):
        self.weekly_data = weekly_data
        self.systems = get_all_systems()
    
    def simulate_season(self, season, system_name):
        """模拟一个赛季"""
        system = self.systems[system_name]
        season_data = self.weekly_data[self.weekly_data['season'] == season].copy()
        
        weeks = sorted(season_data['week'].unique())
        
        results = []
        eliminated_celebrities = set()
        
        for week in weeks:
            # 获取该周数据（排除已淘汰选手）
            week_data = season_data[
                (season_data['week'] == week) & 
                (~season_data['celebrity'].isin(eliminated_celebrities))
            ].copy()
            
            if len(week_data) <= 1:
                continue
            
            # 使用系统确定淘汰
            eliminated, scored_data = system.determine_elimination(
                week_data, 
                week_num=week, 
                total_weeks=max(weeks)
            )
            
            # 记录实际淘汰的选手
            actual_eliminated = week_data[week_data['is_eliminated'] == True]
            actual_elim_name = actual_eliminated['celebrity'].values[0] if len(actual_eliminated) > 0 else None
            
            results.append({
                'season': season,
                'week': week,
                'system': system_name,
                'simulated_elimination': eliminated,
                'actual_elimination': actual_elim_name,
                'match': eliminated == actual_elim_name,
                'n_contestants': len(week_data),
            })
            
            # 更新淘汰名单
            eliminated_celebrities.add(eliminated)
        
        return pd.DataFrame(results)
    
    def simulate_all_seasons(self, system_name):
        """模拟所有赛季"""
        seasons = self.weekly_data['season'].unique()
        all_results = []
        
        for season in seasons:
            season_results = self.simulate_season(season, system_name)
            all_results.append(season_results)
        
        return pd.concat(all_results, ignore_index=True)
    
    def compare_systems(self):
        """比较所有系统"""
        comparison = {}
        
        for system_name in self.systems.keys():
            results = self.simulate_all_seasons(system_name)
            
            # 计算匹配率
            match_rate = results['match'].mean()
            
            # 分季节统计
            season_match = results.groupby('season')['match'].mean()
            
            comparison[system_name] = {
                'results': results,
                'match_rate': match_rate,
                'season_match': season_match,
                'total_weeks': len(results),
                'total_matches': results['match'].sum(),
            }
        
        return comparison


class WeeklyComparator:
    """周级别比较器"""
    
    def __init__(self, weekly_data):
        self.weekly_data = weekly_data
        self.systems = get_all_systems()
    
    def compare_week(self, season, week):
        """比较单周在不同系统下的结果"""
        week_data = self.weekly_data[
            (self.weekly_data['season'] == season) & 
            (self.weekly_data['week'] == week)
        ].copy()
        
        if len(week_data) == 0:
            return None
        
        results = {'season': season, 'week': week}
        
        for system_name, system in self.systems.items():
            eliminated, scored_data = system.determine_elimination(
                week_data, week_num=week, total_weeks=11
            )
            results[f'{system_name}_eliminated'] = eliminated
            
            # 获取综合得分排名
            scored_data = scored_data.sort_values('combined_score', ascending=False)
            results[f'{system_name}_ranking'] = scored_data['celebrity'].tolist()
        
        # 实际淘汰
        actual_elim = week_data[week_data['is_eliminated'] == True]
        results['actual_eliminated'] = actual_elim['celebrity'].values[0] if len(actual_elim) > 0 else None
        
        return results
    
    def find_disagreements(self):
        """找出不同系统产生不同结果的周次"""
        disagreements = []
        
        for season in self.weekly_data['season'].unique():
            season_data = self.weekly_data[self.weekly_data['season'] == season]
            for week in season_data['week'].unique():
                result = self.compare_week(season, week)
                if result is None:
                    continue
                
                eliminations = [result.get(f'{s}_eliminated') for s in self.systems.keys()]
                
                # 如果有不同的淘汰结果
                if len(set(eliminations)) > 1:
                    result['has_disagreement'] = True
                    disagreements.append(result)
        
        return pd.DataFrame(disagreements)


class ControversyAnalyzer:
    """争议分析器"""
    
    def __init__(self, weekly_data):
        self.weekly_data = weekly_data
    
    def calculate_controversy_score(self, week_data):
        """计算争议分数"""
        # 排名差异的标准差作为争议度
        rank_diff = abs(week_data['judge_rank'] - week_data['vote_rank'])
        return rank_diff.std()
    
    def analyze_system_controversy(self, system_results):
        """分析某系统产生的争议"""
        controversies = []
        
        for _, row in system_results.iterrows():
            season, week = row['season'], row['week']
            week_data = self.weekly_data[
                (self.weekly_data['season'] == season) & 
                (self.weekly_data['week'] == week)
            ]
            
            if len(week_data) > 0:
                controversy_score = self.calculate_controversy_score(week_data)
                
                # 检查淘汰是否是争议性的
                eliminated = row['simulated_elimination']
                elim_data = week_data[week_data['celebrity'] == eliminated]
                
                if len(elim_data) > 0:
                    elim_rank_diff = abs(
                        elim_data['judge_rank'].values[0] - 
                        elim_data['vote_rank'].values[0]
                    )
                    is_controversial_elim = elim_rank_diff >= 3
                else:
                    is_controversial_elim = False
                
                controversies.append({
                    'season': season,
                    'week': week,
                    'controversy_score': controversy_score,
                    'controversial_elimination': is_controversial_elim,
                })
        
        return pd.DataFrame(controversies)


if __name__ == '__main__':
    # 测试
    weekly_data = prepare_weekly_data()
    
    simulator = SeasonSimulator(weekly_data)
    comparison = simulator.compare_systems()
    
    print("系统比较结果:")
    for system_name, result in comparison.items():
        print(f"\n{system_name}:")
        print(f"  匹配率: {result['match_rate']:.2%}")
        print(f"  总周次: {result['total_weeks']}")
        print(f"  匹配周次: {result['total_matches']}")
