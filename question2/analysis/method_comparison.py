# -*- coding: utf-8 -*-
"""
方法对比分析与推荐
综合分析两种投票方法的优劣，给出推荐建议
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


class MethodComparisonAnalyzer:
    """
    方法对比分析器
    
    分析维度：
    1. 两种方法的淘汰结果一致性
    2. 哪种方法更偏向观众投票
    3. 哪种方法更偏向评委得分
    4. 争议案例的处理方式
    5. 评委裁决机制的影响
    """
    
    def __init__(self, comparison_df: pd.DataFrame, 
                 controversy_df: pd.DataFrame,
                 vote_estimates_df: pd.DataFrame):
        self.comparison_df = comparison_df
        self.controversy_df = controversy_df
        self.vote_estimates = vote_estimates_df
        
    def calculate_agreement_rate(self) -> Dict:
        """
        计算两种方法的结果一致率
        """
        total = len(self.comparison_df)
        agree = self.comparison_df['methods_agree'].sum()
        
        # 按赛季分组
        by_season = self.comparison_df.groupby('season').agg({
            'methods_agree': ['sum', 'count']
        })
        by_season.columns = ['agree_count', 'total']
        by_season['agreement_rate'] = by_season['agree_count'] / by_season['total']
        
        return {
            'overall_agreement_rate': agree / total if total > 0 else 0,
            'total_weeks_analyzed': total,
            'weeks_agree': agree,
            'weeks_disagree': total - agree,
            'by_season': by_season.to_dict()
        }
    
    def analyze_fan_vote_bias(self) -> Dict:
        """
        分析哪种方法更偏向观众投票
        
        通过分析低评委分选手在两种方法下的生存情况来判断
        """
        # 获取两种方法不一致的周次
        disagree_weeks = self.comparison_df[self.comparison_df['methods_agree'] == False].copy()
        
        if len(disagree_weeks) == 0:
            return {'message': '两种方法在所有周次结果一致，无法判断偏向性'}
        
        fan_bias_analysis = {
            'num_disagreements': len(disagree_weeks),
            'disagreement_details': [],
            'rank_method_favors_fans': 0,
            'percent_method_favors_fans': 0
        }
        
        for _, row in disagree_weeks.iterrows():
            season = row['season']
            week = row['week']
            
            # 获取该周的详细数据
            week_data = self.vote_estimates[
                (self.vote_estimates['season'] == season) &
                (self.vote_estimates['week'] == week)
            ]
            
            rank_eliminated = row['rank_eliminated']
            percent_eliminated = row['percent_eliminated']
            
            # 找到两种方法淘汰的不同选手的评委分数
            rank_elim_score = week_data[
                week_data['celebrity'].str.contains(rank_eliminated.split()[0], case=False, na=False)
            ]['total_score'].values
            
            percent_elim_score = week_data[
                week_data['celebrity'].str.contains(percent_eliminated.split()[0], case=False, na=False)
            ]['total_score'].values
            
            rank_elim_score = rank_elim_score[0] if len(rank_elim_score) > 0 else None
            percent_elim_score = percent_elim_score[0] if len(percent_elim_score) > 0 else None
            
            if rank_elim_score and percent_elim_score:
                # 如果排名法淘汰的人评委分更高，说明排名法更偏向观众
                if rank_elim_score > percent_elim_score:
                    fan_bias_analysis['rank_method_favors_fans'] += 1
                elif percent_elim_score > rank_elim_score:
                    fan_bias_analysis['percent_method_favors_fans'] += 1
                
                fan_bias_analysis['disagreement_details'].append({
                    'season': season,
                    'week': week,
                    'rank_eliminated': rank_eliminated,
                    'rank_elim_judge_score': rank_elim_score,
                    'percent_eliminated': percent_eliminated,
                    'percent_elim_judge_score': percent_elim_score
                })
        
        # 判断整体偏向
        if fan_bias_analysis['rank_method_favors_fans'] > fan_bias_analysis['percent_method_favors_fans']:
            fan_bias_analysis['conclusion'] = '排名法更偏向观众投票（更容易淘汰高评委分选手）'
        elif fan_bias_analysis['percent_method_favors_fans'] > fan_bias_analysis['rank_method_favors_fans']:
            fan_bias_analysis['conclusion'] = '百分比法更偏向观众投票（更容易淘汰高评委分选手）'
        else:
            fan_bias_analysis['conclusion'] = '两种方法对观众投票的偏向程度相似'
        
        return fan_bias_analysis
    
    def analyze_judge_tiebreaker_impact(self) -> Dict:
        """
        分析评委裁决机制（底部二人选择）的影响
        """
        impact_analysis = {
            'description': '从第28季开始，底部两名由评委投票决定淘汰谁',
            'potential_reversals': [],
            'reversal_rate': 0
        }
        
        # 分析在使用评委裁决的赛季中，有多少次评委可能会改变结果
        tiebreaker_seasons = self.comparison_df[self.comparison_df['season'] >= 28]
        
        if len(tiebreaker_seasons) == 0:
            impact_analysis['message'] = '评委裁决机制从第28季开始，当前数据可能不包含这些赛季'
            return impact_analysis
        
        # 模拟评委裁决
        # 假设评委倾向于保留评委分更高的选手
        for _, row in tiebreaker_seasons.iterrows():
            # 这里需要更详细的底部二人信息
            # 简化分析：如果两种方法不一致，评委裁决可能改变结果
            if not row['methods_agree']:
                impact_analysis['potential_reversals'].append({
                    'season': row['season'],
                    'week': row['week'],
                    'rank_eliminated': row['rank_eliminated'],
                    'percent_eliminated': row['percent_eliminated']
                })
        
        impact_analysis['reversal_rate'] = (
            len(impact_analysis['potential_reversals']) / len(tiebreaker_seasons) 
            if len(tiebreaker_seasons) > 0 else 0
        )
        
        return impact_analysis
    
    def generate_recommendation(self) -> Dict:
        """
        基于分析结果生成方法推荐
        """
        agreement = self.calculate_agreement_rate()
        fan_bias = self.analyze_fan_vote_bias()
        tiebreaker = self.analyze_judge_tiebreaker_impact()
        
        recommendation = {
            'summary': {},
            'pros_cons': {
                'rank_method': {'pros': [], 'cons': []},
                'percent_method': {'pros': [], 'cons': []},
                'judge_tiebreaker': {'pros': [], 'cons': []}
            },
            'final_recommendation': '',
            'reasoning': []
        }
        
        # 分析排名法
        recommendation['pros_cons']['rank_method']['pros'] = [
            '规则简单直观，易于理解',
            '每个维度（评委、观众）权重相等',
            '对极端评委分差异有平滑作用'
        ]
        recommendation['pros_cons']['rank_method']['cons'] = [
            '可能导致评委分差异被掩盖（如评委分接近的选手排名差异不大）',
            '在选手数量较少时，排名的分辨度降低'
        ]
        
        # 分析百分比法
        recommendation['pros_cons']['percent_method']['pros'] = [
            '精确反映评委分数和观众投票的实际比例',
            '能够体现得分/投票的数值差异'
        ]
        recommendation['pros_cons']['percent_method']['cons'] = [
            '可能导致评委分对结果影响过大（当评委分差异大时）',
            '计算相对复杂，不易向观众解释'
        ]
        
        # 分析评委裁决机制
        recommendation['pros_cons']['judge_tiebreaker']['pros'] = [
            '增加评委的最终决定权，体现专业判断',
            '可以纠正纯数字计算可能带来的"不公平"结果',
            '增加节目悬念和观赏性'
        ]
        recommendation['pros_cons']['judge_tiebreaker']['cons'] = [
            '可能削弱观众参与感',
            '增加主观因素，可能引发新的争议',
            '评委可能受到压力做出非专业判断'
        ]
        
        # 基于争议案例分析
        if len(self.controversy_df) > 0:
            # 如果争议案例在某种方法下能更早淘汰
            rank_eliminates = self.controversy_df['would_be_eliminated_rank_weeks'].sum() if 'would_be_eliminated_rank_weeks' in self.controversy_df.columns else 0
            percent_eliminates = self.controversy_df['would_be_eliminated_percent_weeks'].sum() if 'would_be_eliminated_percent_weeks' in self.controversy_df.columns else 0
            
            recommendation['summary']['controversy_analysis'] = {
                'rank_method_early_eliminations': rank_eliminates,
                'percent_method_early_eliminations': percent_eliminates
            }
        
        # 生成最终推荐
        recommendation['reasoning'] = [
            f"两种方法的结果一致率为 {agreement['overall_agreement_rate']:.1%}",
            f"在 {agreement['weeks_disagree']} 个周次中两种方法产生了不同的淘汰结果",
        ]
        
        if 'conclusion' in fan_bias:
            recommendation['reasoning'].append(fan_bias['conclusion'])
        
        # 最终推荐
        recommendation['final_recommendation'] = """
基于以上分析，我们推荐：

1. **主要方法：百分比法**
   - 能够更精确地反映评委评分和观众投票的实际差异
   - 对于技术水平差距明显的选手，能够更好地体现这种差距

2. **辅助机制：保留评委裁决**
   - 建议保留"评委从底部两人中选择淘汰对象"的机制
   - 这可以作为防止极端争议情况的安全阀
   - 同时增加节目的悬念和可看性

3. **权重调整建议**
   - 考虑在比赛不同阶段调整评委与观众的权重
   - 初期可以更偏向评委（筛选技术），后期更偏向观众（尊重民意）
"""
        
        return recommendation
    
    def create_summary_report(self) -> str:
        """
        生成完整的分析摘要报告
        """
        agreement = self.calculate_agreement_rate()
        fan_bias = self.analyze_fan_vote_bias()
        tiebreaker = self.analyze_judge_tiebreaker_impact()
        recommendation = self.generate_recommendation()
        
        report = f"""
# 问题二：投票方法对比分析报告

## 1. 两种方法概述

### 排名法 (Rank Method)
- 使用赛季：第1-2季, 第28-34季
- 计算方式：综合排名 = 评委得分排名 + 观众投票排名
- 淘汰规则：综合排名最高者（数字最大）

### 百分比法 (Percent Method)
- 使用赛季：第3-27季
- 计算方式：综合百分比 = 评委得分占比 + 观众投票占比
- 淘汰规则：综合百分比最低者

## 2. 方法一致性分析

- **总分析周次**: {agreement['total_weeks_analyzed']}
- **结果一致周次**: {agreement['weeks_agree']}
- **结果不一致周次**: {agreement['weeks_disagree']}
- **整体一致率**: {agreement['overall_agreement_rate']:.1%}

## 3. 观众投票偏向性分析

{fan_bias.get('conclusion', '分析中...')}

- 不一致案例中，排名法更偏向观众的次数: {fan_bias.get('rank_method_favors_fans', 'N/A')}
- 不一致案例中，百分比法更偏向观众的次数: {fan_bias.get('percent_method_favors_fans', 'N/A')}

## 4. 评委裁决机制分析

{tiebreaker.get('description', '')}

- 潜在结果改变率: {tiebreaker.get('reversal_rate', 0):.1%}

## 5. 推荐建议

{recommendation['final_recommendation']}

---
*报告生成时间: 自动生成*
"""
        return report
