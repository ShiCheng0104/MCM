"""
数据预处理模块
负责加载、清洗、转换原始数据
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .utils import (
    extract_elimination_week,
    compute_weekly_total_score,
    compute_weekly_avg_score,
    is_contestant_active
)


class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, data_path: str):
        """
        初始化数据预处理器
        
        Args:
            data_path: 原始数据CSV文件路径
        """
        self.data_path = data_path
        self.raw_data = None
        self.cleaned_data = None
        self.weekly_data = None
        self.season_week_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        加载原始数据
        
        Returns:
            原始数据DataFrame
        """
        self.raw_data = pd.read_csv(self.data_path)
        print(f"成功加载数据: {len(self.raw_data)} 条记录")
        return self.raw_data
    
    def clean_data(self) -> pd.DataFrame:
        """
        清洗数据
        - 处理N/A值
        - 转换数据类型
        - 添加派生特征
        
        Returns:
            清洗后的DataFrame
        """
        if self.raw_data is None:
            self.load_data()
        
        df = self.raw_data.copy()
        
        # 处理评委得分列，将N/A转换为NaN
        score_columns = [col for col in df.columns if 'judge' in col and 'score' in col]
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 添加淘汰周次
        df['elimination_week'] = df['results'].apply(extract_elimination_week)
        
        # 添加是否决赛选手标志
        df['is_finalist'] = df['results'].str.contains('Place', na=False)
        
        # 添加是否退赛标志
        df['is_withdrew'] = df['results'].str.contains('Withdrew', na=False)
        
        # 添加是否美国选手标志
        df['is_domestic'] = df['celebrity_homecountry/region'] == 'United States'
        
        # 计算每周的总分和平均分
        for week in range(1, 12):
            df[f'week{week}_total'] = df.apply(
                lambda row: compute_weekly_total_score(row, week), axis=1
            )
            df[f'week{week}_avg'] = df.apply(
                lambda row: compute_weekly_avg_score(row, week), axis=1
            )
        
        # 计算选手参赛周数
        df['weeks_competed'] = df.apply(self._count_weeks_competed, axis=1)
        
        self.cleaned_data = df
        print(f"数据清洗完成: {len(df)} 条记录")
        return df
    
    def _count_weeks_competed(self, row: pd.Series) -> int:
        """计算选手参赛周数"""
        count = 0
        for week in range(1, 12):
            if compute_weekly_total_score(row, week) > 0:
                count += 1
        return count
    
    def create_weekly_long_format(self) -> pd.DataFrame:
        """
        创建周级别的长格式数据
        每行代表一个选手在一周的数据
        
        Returns:
            周级别长格式DataFrame
        """
        if self.cleaned_data is None:
            self.clean_data()
        
        records = []
        
        for _, row in self.cleaned_data.iterrows():
            for week in range(1, 12):
                total_score = row[f'week{week}_total']
                
                # 只保留有有效得分的周次
                if total_score > 0:
                    records.append({
                        'celebrity_name': row['celebrity_name'],
                        'ballroom_partner': row['ballroom_partner'],
                        'celebrity_industry': row['celebrity_industry'],
                        'celebrity_age': row['celebrity_age_during_season'],
                        'is_domestic': row['is_domestic'],
                        'season': row['season'],
                        'week': week,
                        'total_score': total_score,
                        'avg_score': row[f'week{week}_avg'],
                        'final_placement': row['placement'],
                        'elimination_week': row['elimination_week'],
                        'is_finalist': row['is_finalist'],
                        'is_withdrew': row['is_withdrew'],
                        'results': row['results']
                    })
        
        self.weekly_data = pd.DataFrame(records)
        print(f"创建周级别数据: {len(self.weekly_data)} 条记录")
        return self.weekly_data
    
    def get_season_week_contestants(self) -> Dict[Tuple[int, int], pd.DataFrame]:
        """
        获取每个赛季每周的参赛选手数据
        
        Returns:
            字典，键为(season, week)，值为该周参赛选手的DataFrame
        """
        if self.weekly_data is None:
            self.create_weekly_long_format()
        
        self.season_week_data = {}
        
        for (season, week), group in self.weekly_data.groupby(['season', 'week']):
            self.season_week_data[(season, week)] = group.copy()
        
        print(f"共有 {len(self.season_week_data)} 个赛季-周次组合")
        return self.season_week_data
    
    def get_elimination_info(self) -> pd.DataFrame:
        """
        获取每周淘汰信息
        
        Returns:
            淘汰信息DataFrame
        """
        if self.cleaned_data is None:
            self.clean_data()
        
        elimination_records = []
        
        for _, row in self.cleaned_data.iterrows():
            elim_week = row['elimination_week']
            if elim_week is not None:
                elimination_records.append({
                    'season': row['season'],
                    'week': elim_week,
                    'eliminated_name': row['celebrity_name'],
                    'eliminated_placement': row['placement'],
                    'final_score': row[f'week{elim_week}_total']
                })
        
        elim_df = pd.DataFrame(elimination_records)
        return elim_df
    
    def get_active_contestants_for_week(self, season: int, week: int) -> pd.DataFrame:
        """
        获取某赛季某周的活跃选手
        
        Args:
            season: 赛季编号
            week: 周次
        
        Returns:
            活跃选手DataFrame
        """
        if self.season_week_data is None:
            self.get_season_week_contestants()
        
        key = (season, week)
        if key in self.season_week_data:
            return self.season_week_data[key]
        return pd.DataFrame()
    
    def get_season_info(self) -> pd.DataFrame:
        """
        获取赛季信息汇总
        
        Returns:
            赛季信息DataFrame
        """
        if self.cleaned_data is None:
            self.clean_data()
        
        season_info = []
        
        for season in self.cleaned_data['season'].unique():
            season_data = self.cleaned_data[self.cleaned_data['season'] == season]
            
            # 计算该赛季的周数
            max_week = 0
            for week in range(1, 12):
                col = f'week{week}_total'
                if season_data[col].sum() > 0:
                    max_week = week
            
            season_info.append({
                'season': season,
                'num_contestants': len(season_data),
                'num_weeks': max_week,
                'num_finalists': season_data['is_finalist'].sum(),
                'num_withdrew': season_data['is_withdrew'].sum()
            })
        
        return pd.DataFrame(season_info).sort_values('season')
    
    def prepare_model_data(self) -> Dict:
        """
        准备模型所需的完整数据结构
        
        Returns:
            包含所有预处理数据的字典
        """
        if self.cleaned_data is None:
            self.clean_data()
        if self.weekly_data is None:
            self.create_weekly_long_format()
        if self.season_week_data is None:
            self.get_season_week_contestants()
        
        elimination_info = self.get_elimination_info()
        season_info = self.get_season_info()
        
        return {
            'cleaned_data': self.cleaned_data,
            'weekly_data': self.weekly_data,
            'season_week_data': self.season_week_data,
            'elimination_info': elimination_info,
            'season_info': season_info
        }


def load_and_preprocess(data_path: str) -> Tuple[DataPreprocessor, Dict]:
    """
    便捷函数：加载并预处理数据
    
    Args:
        data_path: 数据文件路径
    
    Returns:
        (预处理器对象, 预处理后的数据字典)
    """
    preprocessor = DataPreprocessor(data_path)
    data = preprocessor.prepare_model_data()
    return preprocessor, data
