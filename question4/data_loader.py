"""
数据加载模块
"""
import pandas as pd
import numpy as np
from config import VOTE_ESTIMATES_PATH, RAW_DATA_PATH


def load_vote_estimates():
    """加载预测的观众投票数据"""
    df = pd.read_csv(VOTE_ESTIMATES_PATH)
    return df


def load_raw_data():
    """加载原始比赛数据"""
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.rename(columns={
        'celebrity_name': 'celebrity',
        'ballroom_partner': 'partner',
        'celebrity_industry': 'industry',
        'celebrity_homestate': 'home_state',
        'celebrity_homecountry/region': 'home_country',
        'celebrity_age_during_season': 'age'
    })
    return df


def prepare_weekly_data():
    """准备周级别数据，用于投票系统模拟"""
    vote_df = load_vote_estimates()
    raw_df = load_raw_data()
    
    # 合并数据
    weekly_data = []
    
    for season in vote_df['season'].unique():
        season_votes = vote_df[vote_df['season'] == season]
        season_raw = raw_df[raw_df['season'] == season]
        
        for week in season_votes['week'].unique():
            week_data = season_votes[season_votes['week'] == week].copy()
            
            if len(week_data) == 0:
                continue
            
            # 添加原始数据信息
            for idx, row in week_data.iterrows():
                celeb_raw = season_raw[season_raw['celebrity'] == row['celebrity']]
                if len(celeb_raw) > 0:
                    week_data.loc[idx, 'placement'] = celeb_raw['placement'].values[0]
                    week_data.loc[idx, 'industry'] = celeb_raw['industry'].values[0]
                    week_data.loc[idx, 'age'] = celeb_raw['age'].values[0]
            
            # 计算排名
            week_data['judge_rank'] = week_data['total_score'].rank(ascending=False)
            week_data['vote_rank'] = week_data['estimated_votes'].rank(ascending=False)
            
            # 计算百分比
            total_score_sum = week_data['total_score'].sum()
            total_votes_sum = week_data['estimated_votes'].sum()
            
            week_data['judge_percent'] = week_data['total_score'] / total_score_sum
            week_data['vote_percent'] = week_data['estimated_votes'] / total_votes_sum
            
            # 计算排名差异（评委排名 - 观众排名）
            week_data['rank_diff'] = week_data['judge_rank'] - week_data['vote_rank']
            
            weekly_data.append(week_data)
    
    return pd.concat(weekly_data, ignore_index=True)


def get_elimination_history():
    """获取历史淘汰记录"""
    vote_df = load_vote_estimates()
    
    eliminations = vote_df[vote_df['is_eliminated'] == True].copy()
    eliminations = eliminations[['season', 'week', 'celebrity', 'total_score', 
                                  'estimated_votes', 'vote_share']].copy()
    
    return eliminations


def get_controversy_cases():
    """获取争议案例（评委排名与最终名次差异大的选手）"""
    raw_df = load_raw_data()
    vote_df = load_vote_estimates()
    
    controversies = []
    
    # 已知争议案例
    known_cases = [
        {'season': 2, 'celebrity': 'Jerry Rice', 'issue': '5周最低评委分仍获亚军'},
        {'season': 4, 'celebrity': 'Billy Ray Cyrus', 'issue': '6周最低评委分仍获第5'},
        {'season': 11, 'celebrity': 'Bristol Palin', 'issue': '12次最低评委分仍获季军'},
        {'season': 27, 'celebrity': 'Bobby Bones', 'issue': '持续低分仍获冠军'},
    ]
    
    for case in known_cases:
        season_data = vote_df[vote_df['season'] == case['season']]
        celeb_data = season_data[season_data['celebrity'] == case['celebrity']]
        raw_celeb = raw_df[(raw_df['season'] == case['season']) & 
                           (raw_df['celebrity'] == case['celebrity'])]
        
        if len(celeb_data) > 0 and len(raw_celeb) > 0:
            case['placement'] = raw_celeb['placement'].values[0]
            case['avg_score'] = celeb_data['total_score'].mean()
            case['avg_votes'] = celeb_data['estimated_votes'].mean()
            case['weeks_participated'] = len(celeb_data)
            
            # 计算每周的评委排名
            ranks = []
            for week in celeb_data['week'].unique():
                week_all = season_data[season_data['week'] == week]
                week_celeb = celeb_data[celeb_data['week'] == week]
                if len(week_celeb) > 0:
                    rank = (week_all['total_score'] > week_celeb['total_score'].values[0]).sum() + 1
                    ranks.append(rank)
            
            case['avg_judge_rank'] = np.mean(ranks) if ranks else None
            case['lowest_rank_weeks'] = sum(1 for r in ranks if r == max(ranks))
            
            controversies.append(case)
    
    return pd.DataFrame(controversies)


if __name__ == '__main__':
    # 测试
    df = prepare_weekly_data()
    print(f"数据集大小: {df.shape}")
    print(f"\n列名: {df.columns.tolist()}")
    
    controversies = get_controversy_cases()
    print(f"\n争议案例:\n{controversies}")
