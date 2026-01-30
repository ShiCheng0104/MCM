"""
工具函数模块
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import re


def extract_elimination_week(result_str: str) -> Optional[int]:
    """
    从结果字符串中提取淘汰周次
    
    Args:
        result_str: 结果字符串，如 "Eliminated Week 3", "1st Place"
    
    Returns:
        淘汰周次，若为决赛选手则返回None
    """
    if pd.isna(result_str):
        return None
    
    result_str = str(result_str)
    
    # 决赛选手
    if 'Place' in result_str:
        return None
    
    # 退赛选手
    if 'Withdrew' in result_str:
        return None
    
    # 提取周次
    match = re.search(r'Week (\d+)', result_str)
    if match:
        return int(match.group(1))
    
    return None


def get_placement_from_result(result_str: str, placement: int) -> int:
    """
    获取最终名次
    
    Args:
        result_str: 结果字符串
        placement: 原始placement值
    
    Returns:
        最终名次
    """
    return placement


def compute_weekly_total_score(row: pd.Series, week: int) -> float:
    """
    计算某周的评委总分
    
    Args:
        row: 数据行
        week: 周次
    
    Returns:
        评委总分
    """
    total = 0
    count = 0
    
    for judge in range(1, 5):
        col = f'week{week}_judge{judge}_score'
        if col in row.index:
            val = row[col]
            if pd.notna(val) and val != 'N/A':
                try:
                    score = float(val)
                    if score > 0:  # 排除已淘汰的0分
                        total += score
                        count += 1
                except (ValueError, TypeError):
                    pass
    
    return total if count > 0 else 0


def compute_weekly_avg_score(row: pd.Series, week: int) -> float:
    """
    计算某周的评委平均分
    
    Args:
        row: 数据行
        week: 周次
    
    Returns:
        评委平均分
    """
    scores = []
    
    for judge in range(1, 5):
        col = f'week{week}_judge{judge}_score'
        if col in row.index:
            val = row[col]
            if pd.notna(val) and val != 'N/A':
                try:
                    score = float(val)
                    if score > 0:
                        scores.append(score)
                except (ValueError, TypeError):
                    pass
    
    return np.mean(scores) if scores else 0


def rank_scores(scores: np.ndarray, ascending: bool = False) -> np.ndarray:
    """
    计算排名（1为最好）
    
    Args:
        scores: 得分数组
        ascending: 是否升序排名（True表示分数越低排名越好）
    
    Returns:
        排名数组
    """
    scores = np.asarray(scores, dtype=float)
    if ascending:
        return np.argsort(np.argsort(scores)) + 1
    else:
        return np.argsort(np.argsort(-scores)) + 1


def compute_rank_combined_score(judge_scores: np.ndarray, 
                                 fan_votes: np.ndarray) -> np.ndarray:
    """
    计算排名法下的综合得分（得分越高排名越差）
    
    Args:
        judge_scores: 评委得分数组
        fan_votes: 观众投票数组
    
    Returns:
        综合排名得分（越高越差）
    """
    judge_scores = np.asarray(judge_scores, dtype=float)
    fan_votes = np.asarray(fan_votes, dtype=float)
    judge_ranks = rank_scores(judge_scores, ascending=False)
    fan_ranks = rank_scores(fan_votes, ascending=False)
    return judge_ranks + fan_ranks


def compute_percent_combined_score(judge_scores: np.ndarray, 
                                    fan_votes: np.ndarray) -> np.ndarray:
    """
    计算百分比法下的综合得分（得分越高排名越好）
    
    Args:
        judge_scores: 评委得分数组
        fan_votes: 观众投票数组
    
    Returns:
        综合百分比得分（越高越好）
    """
    judge_scores = np.asarray(judge_scores, dtype=float)
    fan_votes = np.asarray(fan_votes, dtype=float)
    judge_pct = judge_scores / judge_scores.sum() if judge_scores.sum() > 0 else np.zeros_like(judge_scores)
    fan_pct = fan_votes / fan_votes.sum() if fan_votes.sum() > 0 else np.zeros_like(fan_votes)
    return judge_pct + fan_pct


def get_eliminated_index_rank(combined_ranks: np.ndarray) -> int:
    """
    排名法下获取被淘汰选手的索引（综合排名最高/最差者）
    
    Args:
        combined_ranks: 综合排名得分
    
    Returns:
        被淘汰选手索引
    """
    return int(np.argmax(combined_ranks))


def get_eliminated_index_percent(combined_pcts: np.ndarray) -> int:
    """
    百分比法下获取被淘汰选手的索引（综合百分比最低者）
    
    Args:
        combined_pcts: 综合百分比得分
    
    Returns:
        被淘汰选手索引
    """
    return int(np.argmin(combined_pcts))


def is_contestant_active(row: pd.Series, week: int) -> bool:
    """
    判断选手在某周是否仍在比赛
    
    Args:
        row: 选手数据行
        week: 周次
    
    Returns:
        是否仍在比赛
    """
    elimination_week = extract_elimination_week(row['results'])
    
    # 决赛选手或退赛选手需要特殊处理
    if elimination_week is None:
        # 检查该周是否有得分
        total = compute_weekly_total_score(row, week)
        return total > 0
    
    # 淘汰周之前的周次都是活跃的
    return week < elimination_week


def normalize_votes(votes: np.ndarray, 
                   total_votes: float = 1e6) -> np.ndarray:
    """
    归一化投票数到指定总量
    
    Args:
        votes: 原始投票估计
        total_votes: 目标总投票数
    
    Returns:
        归一化后的投票数
    """
    if votes.sum() > 0:
        return votes / votes.sum() * total_votes
    return votes


def compute_confidence_interval(samples: np.ndarray, 
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
    计算置信区间
    
    Args:
        samples: 样本数组
        confidence: 置信水平
    
    Returns:
        (下界, 上界)
    """
    alpha = 1 - confidence
    lower = np.percentile(samples, alpha / 2 * 100)
    upper = np.percentile(samples, (1 - alpha / 2) * 100)
    return (lower, upper)


def compute_cv(values: np.ndarray) -> float:
    """
    计算变异系数 (Coefficient of Variation)
    
    Args:
        values: 数值数组
    
    Returns:
        变异系数
    """
    mean = np.mean(values)
    std = np.std(values)
    if mean > 0:
        return std / mean
    return float('inf')
