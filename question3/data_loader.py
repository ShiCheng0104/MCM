"""
数据加载和预处理模块
"""
import pandas as pd
import numpy as np
from config import VOTE_ESTIMATES_PATH, RAW_DATA_PATH, AGE_BINS, AGE_LABELS


def load_raw_data():
    """加载原始比赛数据"""
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 重命名列以便处理
    df = df.rename(columns={
        'celebrity_name': 'celebrity',
        'ballroom_partner': 'partner',
        'celebrity_industry': 'industry',
        'celebrity_homestate': 'home_state',
        'celebrity_homecountry/region': 'home_country',
        'celebrity_age_during_season': 'age'
    })
    
    return df


def load_vote_estimates():
    """加载预测的观众投票数据"""
    df = pd.read_csv(VOTE_ESTIMATES_PATH)
    return df


def extract_weekly_judge_scores(raw_df):
    """提取每周评委得分，转换为长格式"""
    records = []
    
    for _, row in raw_df.iterrows():
        for week in range(1, 12):
            judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
            scores = []
            
            for col in judge_cols:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val) and val != 'N/A' and val != 0:
                        try:
                            scores.append(float(val))
                        except (ValueError, TypeError):
                            pass
            
            if scores and sum(scores) > 0:  # 只记录有效分数
                records.append({
                    'celebrity': row['celebrity'],
                    'partner': row['partner'],
                    'industry': row['industry'],
                    'home_state': row['home_state'],
                    'home_country': row['home_country'],
                    'age': row['age'],
                    'season': row['season'],
                    'placement': row['placement'],
                    'results': row['results'],
                    'week': week,
                    'judge_total_score': sum(scores),
                    'judge_avg_score': np.mean(scores),
                    'num_judges': len(scores)
                })
    
    return pd.DataFrame(records)


def merge_data():
    """合并投票估计和原始数据"""
    # 加载数据
    vote_df = load_vote_estimates()
    raw_df = load_raw_data()
    weekly_df = extract_weekly_judge_scores(raw_df)
    
    # 合并数据
    merged = pd.merge(
        vote_df,
        weekly_df,
        on=['season', 'week', 'celebrity'],
        how='left',
        suffixes=('_vote', '_raw')
    )
    
    # 填充缺失的特征
    for col in ['partner', 'industry', 'home_state', 'home_country', 'age', 'placement']:
        if f'{col}_vote' in merged.columns and f'{col}_raw' in merged.columns:
            merged[col] = merged[f'{col}_raw'].fillna(merged[f'{col}_vote'])
        elif f'{col}_raw' in merged.columns:
            merged[col] = merged[f'{col}_raw']
    
    return merged


def create_analysis_dataset():
    """创建用于分析的完整数据集"""
    merged = merge_data()
    
    # 过滤掉无效数据
    df = merged[merged['estimated_votes'] > 0].copy()
    
    # 特征工程
    # 1. 年龄分组
    df['age_group'] = pd.cut(
        df['age'], 
        bins=AGE_BINS, 
        labels=AGE_LABELS,
        include_lowest=True
    )
    
    # 2. 是否美国人
    df['is_domestic'] = (df['home_country'] == 'United States').astype(int)
    
    # 3. 行业简化分类
    df['industry_simplified'] = df['industry'].apply(simplify_industry)
    
    # 4. 计算舞伴经验（历史参赛次数）
    partner_exp = df.groupby('partner')['season'].nunique().reset_index()
    partner_exp.columns = ['partner', 'partner_experience']
    df = pd.merge(df, partner_exp, on='partner', how='left')
    
    # 5. 计算选手在该周的排名
    df['score_rank'] = df.groupby(['season', 'week'])['total_score'].rank(ascending=False)
    df['vote_rank'] = df.groupby(['season', 'week'])['estimated_votes'].rank(ascending=False)
    
    # 6. 剩余选手数
    df['remaining_contestants'] = df.groupby(['season', 'week'])['celebrity'].transform('count')
    
    # 7. 对数投票数（用于回归）
    df['log_votes'] = np.log1p(df['estimated_votes'])
    
    # 8. 标准化得分
    df['score_normalized'] = df.groupby(['season', 'week'])['total_score'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )
    
    # 9. 标准化投票
    df['vote_normalized'] = df.groupby(['season', 'week'])['estimated_votes'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )
    
    return df


def simplify_industry(industry):
    """简化行业分类"""
    if pd.isna(industry):
        return 'Other'
    
    industry = str(industry).lower()
    
    if 'actor' in industry or 'actress' in industry:
        return 'Actor/Actress'
    elif 'athlete' in industry or 'olympian' in industry or 'sports' in industry:
        return 'Athlete'
    elif 'singer' in industry or 'rapper' in industry or 'musician' in industry:
        return 'Singer/Musician'
    elif 'tv' in industry or 'television' in industry or 'reality' in industry:
        return 'TV Personality'
    elif 'model' in industry:
        return 'Model'
    elif 'news' in industry or 'journalist' in industry:
        return 'News/Journalist'
    elif 'comedian' in industry:
        return 'Comedian'
    elif 'social media' in industry or 'influencer' in industry or 'youtuber' in industry:
        return 'Social Media'
    else:
        return 'Other'


def get_partner_summary(df):
    """获取舞伴统计摘要"""
    partner_stats = df.groupby('partner').agg({
        'celebrity': 'nunique',
        'season': 'nunique',
        'total_score': 'mean',
        'estimated_votes': 'mean',
        'placement': 'mean',
        'judge_avg_score': 'mean'
    }).reset_index()
    
    partner_stats.columns = [
        'partner', 'num_celebrities', 'num_seasons',
        'avg_total_score', 'avg_votes', 'avg_placement', 'avg_judge_score'
    ]
    
    # 计算冠军率和前三率
    finalist_df = df.groupby('partner').apply(
        lambda x: pd.Series({
            'win_rate': (x['placement'] == 1).sum() / x['celebrity'].nunique(),
            'top3_rate': (x['placement'] <= 3).sum() / x['celebrity'].nunique()
        })
    ).reset_index()
    
    partner_stats = pd.merge(partner_stats, finalist_df, on='partner')
    
    return partner_stats.sort_values('avg_placement')


if __name__ == '__main__':
    # 测试数据加载
    df = create_analysis_dataset()
    print(f"数据集大小: {df.shape}")
    print(f"\n列名: {df.columns.tolist()}")
    print(f"\n行业分布:\n{df['industry_simplified'].value_counts()}")
    print(f"\n年龄分布:\n{df['age_group'].value_counts()}")
