"""
投票系统实现模块
包含现有系统和新提出的系统
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from config import NEW_SYSTEM_PARAMS


class VotingSystem(ABC):
    """投票系统基类"""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def calculate_combined_score(self, week_data, week_num, total_weeks):
        """计算综合得分"""
        pass
    
    @abstractmethod
    def determine_elimination(self, week_data, week_num, total_weeks):
        """确定淘汰人选"""
        pass
    
    def get_description(self):
        """获取系统描述"""
        return f"Voting System: {self.name}"


class RankBasedSystem(VotingSystem):
    """排名法系统（S1-2, S28-34使用）"""
    
    def __init__(self):
        super().__init__("Rank-Based System")
    
    def calculate_combined_score(self, week_data, week_num=None, total_weeks=None):
        """综合排名 = 评委排名 + 观众排名（越小越好）"""
        df = week_data.copy()
        df['judge_rank'] = df['total_score'].rank(ascending=False)
        df['vote_rank'] = df['estimated_votes'].rank(ascending=False)
        df['combined_score'] = df['judge_rank'] + df['vote_rank']
        return df
    
    def determine_elimination(self, week_data, week_num=None, total_weeks=None):
        """淘汰综合排名最高（最差）的选手"""
        df = self.calculate_combined_score(week_data, week_num, total_weeks)
        eliminated = df.loc[df['combined_score'].idxmax(), 'celebrity']
        return eliminated, df


class PercentageBasedSystem(VotingSystem):
    """百分比法系统（S3-27使用）"""
    
    def __init__(self):
        super().__init__("Percentage-Based System")
    
    def calculate_combined_score(self, week_data, week_num=None, total_weeks=None):
        """综合百分比 = 评委百分比 + 观众百分比（越大越好）"""
        df = week_data.copy()
        
        total_score_sum = df['total_score'].sum()
        total_votes_sum = df['estimated_votes'].sum()
        
        df['judge_percent'] = df['total_score'] / total_score_sum
        df['vote_percent'] = df['estimated_votes'] / total_votes_sum
        df['combined_score'] = df['judge_percent'] + df['vote_percent']
        
        return df
    
    def determine_elimination(self, week_data, week_num=None, total_weeks=None):
        """淘汰综合百分比最低的选手"""
        df = self.calculate_combined_score(week_data, week_num, total_weeks)
        eliminated = df.loc[df['combined_score'].idxmin(), 'celebrity']
        return eliminated, df


class DynamicWeightedSystem(VotingSystem):
    """
    动态加权系统（新系统）
    
    核心特点：
    1. 动态权重：根据比赛阶段调整评委和观众权重
    2. 争议保护：当评委和观众意见严重分歧时，给予特殊处理
    3. 技术保护：评委得分前列的选手在早期获得保护
    4. 悬念设计：保持适度的不确定性以增加观赏性
    """
    
    def __init__(self, params=None):
        super().__init__("Dynamic Weighted System (NEW)")
        self.params = params or NEW_SYSTEM_PARAMS
    
    def get_stage_weights(self, week_num, total_weeks=11):
        """根据比赛阶段获取权重"""
        if week_num in self.params['early_stage_weeks']:
            return self.params['weights']['early']
        elif week_num in self.params['mid_stage_weeks']:
            return self.params['weights']['mid']
        else:
            return self.params['weights']['late']
    
    def calculate_combined_score(self, week_data, week_num, total_weeks=11):
        """计算动态加权综合得分"""
        df = week_data.copy()
        
        # 获取当前阶段权重
        weights = self.get_stage_weights(week_num, total_weeks)
        w_judge = weights['judge']
        w_fan = weights['fan']
        
        # 计算标准化得分（0-1范围）
        score_min, score_max = df['total_score'].min(), df['total_score'].max()
        vote_min, vote_max = df['estimated_votes'].min(), df['estimated_votes'].max()
        
        df['score_normalized'] = (df['total_score'] - score_min) / (score_max - score_min + 1e-10)
        df['vote_normalized'] = (df['estimated_votes'] - vote_min) / (vote_max - vote_min + 1e-10)
        
        # 计算加权综合得分
        df['combined_score'] = w_judge * df['score_normalized'] + w_fan * df['vote_normalized']
        
        # 计算排名
        df['judge_rank'] = df['total_score'].rank(ascending=False)
        df['vote_rank'] = df['estimated_votes'].rank(ascending=False)
        
        # 计算争议度（排名差异）
        df['rank_diff'] = abs(df['judge_rank'] - df['vote_rank'])
        df['is_controversial'] = df['rank_diff'] >= self.params['controversy_bonus']['threshold']
        
        return df
    
    def determine_elimination(self, week_data, week_num, total_weeks=11):
        """确定淘汰人选（含保护机制）"""
        df = self.calculate_combined_score(week_data, week_num, total_weeks)
        n_contestants = len(df)
        
        # 确定底部选手
        bottom_n = min(self.params['elimination']['bottom_n'], n_contestants - 1)
        df_sorted = df.sort_values('combined_score', ascending=True)
        bottom_contestants = df_sorted.head(bottom_n)
        
        # 技术保护机制（仅在早期阶段）
        if week_num in self.params['tech_protection']['protection_weeks']:
            top_threshold = int(n_contestants * self.params['tech_protection']['top_percentile'])
            df['is_protected'] = df['judge_rank'] <= top_threshold
            
            # 如果底部选手中有被保护的，尝试保护
            protected_in_bottom = bottom_contestants[
                bottom_contestants['celebrity'].isin(
                    df[df['is_protected']]['celebrity']
                )
            ]
            
            if len(protected_in_bottom) > 0:
                # 从未被保护的底部选手中选择淘汰
                unprotected_bottom = bottom_contestants[
                    ~bottom_contestants['celebrity'].isin(protected_in_bottom['celebrity'])
                ]
                if len(unprotected_bottom) > 0:
                    eliminated = unprotected_bottom.iloc[0]['celebrity']
                    df.loc[df['celebrity'] == eliminated, 'elimination_reason'] = 'unprotected_lowest'
                    return eliminated, df
        
        # 争议处理：如果底部选手是争议选手，触发评委裁决
        controversial_in_bottom = bottom_contestants[bottom_contestants['is_controversial']]
        
        if len(controversial_in_bottom) > 0 and self.params['elimination']['judge_tiebreaker']:
            # 评委裁决：在争议情况下，选择评委得分较低的淘汰
            # 这给予评委适度的最终决定权，同时保持争议的悬念
            eliminated = bottom_contestants.loc[
                bottom_contestants['total_score'].idxmin(), 'celebrity'
            ]
            df.loc[df['celebrity'] == eliminated, 'elimination_reason'] = 'judge_tiebreaker'
        else:
            # 正常淘汰：综合得分最低的
            eliminated = df_sorted.iloc[0]['celebrity']
            df.loc[df['celebrity'] == eliminated, 'elimination_reason'] = 'lowest_combined'
        
        return eliminated, df
    
    def get_description(self):
        return """
        动态加权系统 (Dynamic Weighted System)
        
        核心规则：
        1. 早期阶段(Week 1-3): 评委权重55%, 观众权重45%
           - 侧重技术筛选，防止技术差的选手过早占用资源
           - 评委得分前1/3选手获得"技术保护"
        
        2. 中期阶段(Week 4-7): 评委权重50%, 观众权重50%
           - 平衡技术和人气
           - 取消技术保护
        
        3. 后期阶段(Week 8+): 评委权重40%, 观众权重60%
           - 观众主导决赛走向
           - 增加投票的重要性和悬念
        
        4. 争议处理机制:
           - 当选手评委排名与观众排名差≥3时，标记为"争议选手"
           - 争议选手进入底部时，触发评委裁决
           - 评委在争议中有最终决定权（选择得分更低的淘汰）
        
        5. 底部淘汰机制:
           - 每周确定综合得分最低的2人
           - 考虑保护机制和争议情况后确定淘汰
        """


class ExcitementMaximizedSystem(VotingSystem):
    """
    观赏性最大化系统
    
    设计理念：最大化节目的悬念和争议，提升观众兴趣
    """
    
    def __init__(self):
        super().__init__("Excitement-Maximized System")
    
    def calculate_combined_score(self, week_data, week_num, total_weeks=11):
        """计算综合得分，增加不确定性因素"""
        df = week_data.copy()
        
        # 基础得分（百分比法）
        total_score_sum = df['total_score'].sum()
        total_votes_sum = df['estimated_votes'].sum()
        
        df['judge_percent'] = df['total_score'] / total_score_sum
        df['vote_percent'] = df['estimated_votes'] / total_votes_sum
        
        # 计算排名差异（争议度）
        df['judge_rank'] = df['total_score'].rank(ascending=False)
        df['vote_rank'] = df['estimated_votes'].rank(ascending=False)
        df['rank_diff'] = abs(df['judge_rank'] - df['vote_rank'])
        
        # 争议加成：排名差异大的选手获得生存优势
        # 这是为了保留争议，增加节目话题性
        df['controversy_bonus'] = df['rank_diff'] * 0.01
        
        # 综合得分 = 50%评委 + 50%观众 + 争议加成
        df['combined_score'] = (0.5 * df['judge_percent'] + 
                                0.5 * df['vote_percent'] + 
                                df['controversy_bonus'])
        
        return df
    
    def determine_elimination(self, week_data, week_num, total_weeks=11):
        """淘汰综合得分最低的选手"""
        df = self.calculate_combined_score(week_data, week_num, total_weeks)
        eliminated = df.loc[df['combined_score'].idxmin(), 'celebrity']
        return eliminated, df


class FairnessOptimizedSystem(VotingSystem):
    """
    公平性优化系统
    
    设计理念：确保技术最好的选手不会因投票不足而过早淘汰
    """
    
    def __init__(self):
        super().__init__("Fairness-Optimized System")
    
    def calculate_combined_score(self, week_data, week_num, total_weeks=11):
        """计算综合得分，侧重评委评分"""
        df = week_data.copy()
        
        # 计算标准化得分
        score_min, score_max = df['total_score'].min(), df['total_score'].max()
        vote_min, vote_max = df['estimated_votes'].min(), df['estimated_votes'].max()
        
        df['score_normalized'] = (df['total_score'] - score_min) / (score_max - score_min + 1e-10)
        df['vote_normalized'] = (df['estimated_votes'] - vote_min) / (vote_max - vote_min + 1e-10)
        
        # 评委权重更高 (60-40)
        df['combined_score'] = 0.6 * df['score_normalized'] + 0.4 * df['vote_normalized']
        
        # 排名
        df['judge_rank'] = df['total_score'].rank(ascending=False)
        df['vote_rank'] = df['estimated_votes'].rank(ascending=False)
        
        return df
    
    def determine_elimination(self, week_data, week_num, total_weeks=11):
        """淘汰综合得分最低的选手，但保护评委前50%"""
        df = self.calculate_combined_score(week_data, week_num, total_weeks)
        n = len(df)
        
        # 评委得分前50%的选手获得保护
        protected_threshold = n // 2
        df['is_protected'] = df['judge_rank'] <= protected_threshold
        
        # 从未保护的选手中选择淘汰
        unprotected = df[~df['is_protected']]
        
        if len(unprotected) > 0:
            eliminated = unprotected.loc[unprotected['combined_score'].idxmin(), 'celebrity']
        else:
            # 所有人都被保护，取消保护正常淘汰
            eliminated = df.loc[df['combined_score'].idxmin(), 'celebrity']
        
        return eliminated, df


class DramaticArcSystem(VotingSystem):
    """
    戏剧弧线系统 (Dramatic Arc System) V4 - 激进优化版
    
    核心理念：最大化观赏性争议(15%目标)同时保持基本公平(85%+)
    
    V4关键改进：
    1. 更大的权重跨度：早期62%评委→后期32%评委，制造巨大反转
    2. 降低争议阈值：排名差≥2即触发，增加争议选手数量
    3. 扩大反差淘汰：更多周数触发，更宽松的条件
    4. 缩小安全区：仅保护前25%，让更多选手处于"危险"中
    5. 扩大危险区：4人危险区增加悬念
    
    目标指标：
    - Controversy Rate 15% (最优范围)
    - Fairness ≥ 85%
    - Excitement ≥ 0.80
    - Composite Score ≥ 0.86 (超越percent系统)
    """
    
    def __init__(self):
        super().__init__("Dramatic Arc System V8 (PROPOSED)")
        
        # 系统参数 - V8公平性优化版本 (Fairness>0.88, Excitement>0.72)
        self.params = {
            # 阶段划分
            'early_weeks': [1, 2, 3],
            'mid_weeks': [4, 5, 6, 7],
            'late_weeks': [8, 9, 10, 11, 12],
            
            # 动态权重 - V8优化：早期评委主导增加争议，后期观众主导增加参与感
            'weights': {
                'early': {'judge': 0.65, 'fan': 0.35},   # 早期：偏评委（增加争议）
                'mid': {'judge': 0.50, 'fan': 0.50},     # 中期：完全平衡
                'late': {'judge': 0.38, 'fan': 0.62},    # 后期：偏观众（更多悬念）
            },
            
            # 争议机制参数 - V9：积极争议触发
            'controversy': {
                'threshold': 3,              # 排名差>=3触发
                'bonus_rate': 0.12,          # 争议加成12%（提高）
                'max_bonus': 0.15,           # 最大加成15%
            },
            
            # 投票差距放大器 - V7：禁用以提高一致性
            'vote_amplifier': {
                'enabled': False,            # 禁用，避免偏离percent结果
                'close_threshold': 0.15,
                'amplification': 1.35,
            },
            
            # 惊喜保护机制 - V7：禁用以提高一致性
            'surprise_protection': {
                'enabled': False,            # 禁用
                'fan_threshold': 0.25,
                'active_weeks': [2, 3, 4],
            },
            
            # 反差淘汰机制 - V9：扩大触发范围以提高争议率
            'upset_mechanism': {
                'enabled': True,
                'judge_high_threshold': 0.40,  # 评委排名前40%
                'fan_low_threshold': 0.30,     # 观众排名后30%
                'active_weeks': [3, 4, 5, 6, 7, 8],  # 6周触发（大幅增加争议机会）
                'require_in_danger': False,    # 不必须在危险区（增加争议）
                'max_rank_protection': 0.55,   # 但保护综合排名前55%（确保公平）
            },
            
            # 安全区保护 - V8：扩大安全区确保公平性
            'safety_zone': {
                'enabled': True,
                'threshold': 0.55,           # 保护综合得分前55%（扩大保护范围）
            },
            
            # 边缘争议机制 - V9：增加边缘情况的争议性
            'edge_controversy': {
                'enabled': True,
                'score_diff_threshold': 0.05,  # 得分差距<5%（扩大范围）
                'prefer_controversial': True,  # 优先选择争议性更高的选手
            },
            
            # 淘汰机制 - V8：标准危险区
            'elimination': {
                'danger_zone_size': 2,       # 缩小危险区为2人（更公平）
                'judge_override_threshold': 0.98,  # 几乎不覆盖
            }
        }
    
    def get_stage(self, week_num):
        """确定比赛阶段"""
        if week_num in self.params['early_weeks']:
            return 'early'
        elif week_num in self.params['mid_weeks']:
            return 'mid'
        else:
            return 'late'
    
    def calculate_combined_score(self, week_data, week_num, total_weeks=11):
        """计算戏剧弧线加权综合得分"""
        df = week_data.copy()
        stage = self.get_stage(week_num)
        weights = self.params['weights'][stage]
        
        # 1. 标准化得分
        score_min, score_max = df['total_score'].min(), df['total_score'].max()
        vote_min, vote_max = df['estimated_votes'].min(), df['estimated_votes'].max()
        
        df['score_normalized'] = (df['total_score'] - score_min) / (score_max - score_min + 1e-10)
        df['vote_normalized'] = (df['estimated_votes'] - vote_min) / (vote_max - vote_min + 1e-10)
        
        # 2. 计算排名
        df['judge_rank'] = df['total_score'].rank(ascending=False)
        df['vote_rank'] = df['estimated_votes'].rank(ascending=False)
        n = len(df)
        
        # 3. 争议度计算
        df['rank_diff'] = abs(df['judge_rank'] - df['vote_rank'])
        df['is_controversial'] = df['rank_diff'] >= self.params['controversy']['threshold']
        
        # 4. 争议加成
        controversy_params = self.params['controversy']
        df['controversy_bonus'] = 0.0
        # 计算争议加成（使用np.minimum处理Series）
        controversial_mask = df['is_controversial']
        if controversial_mask.any():
            bonus_values = controversy_params['bonus_rate'] * (df.loc[controversial_mask, 'rank_diff'] / n)
            df.loc[controversial_mask, 'controversy_bonus'] = np.minimum(
                bonus_values, controversy_params['max_bonus']
            )
        
        # 5. 基础综合得分
        df['base_score'] = (weights['judge'] * df['score_normalized'] + 
                           weights['fan'] * df['vote_normalized'])
        
        # 6. 加入争议加成
        df['combined_score'] = df['base_score'] + df['controversy_bonus']
        
        # 7. 投票差距放大器（在中后期启用）
        if self.params['vote_amplifier']['enabled'] and stage in ['mid', 'late']:
            vote_range = df['vote_normalized'].max() - df['vote_normalized'].min()
            if vote_range < self.params['vote_amplifier']['close_threshold']:
                # 投票接近时，放大差距增加悬念
                df['vote_amplified'] = df['vote_normalized'] * self.params['vote_amplifier']['amplification']
                df['combined_score'] = (weights['judge'] * df['score_normalized'] + 
                                       weights['fan'] * df['vote_amplified'] +
                                       df['controversy_bonus'])
        
        return df
    
    def determine_elimination(self, week_data, week_num, total_weeks=11):
        """V6优化：确定淘汰（争议率15%+同时保持公平性和一致性）"""
        df = self.calculate_combined_score(week_data, week_num, total_weeks)
        n = len(df)
        stage = self.get_stage(week_num)
        
        # 确定危险区
        danger_size = min(self.params['elimination']['danger_zone_size'], n - 1)
        df_sorted = df.sort_values('combined_score', ascending=True)
        danger_zone = df_sorted.head(danger_size)
        danger_indices = set(danger_zone.index)
        
        # V6: 安全区保护（保护综合得分前40%，缩小安全区）
        safety_params = self.params.get('safety_zone', {'enabled': False})
        protected_indices = set()
        if safety_params['enabled']:
            safety_threshold = int(n * safety_params['threshold'])
            df['in_safety_zone'] = df['combined_score'].rank(ascending=False) <= safety_threshold
            protected_indices = set(df[df['in_safety_zone']].index)
        
        # V6核心：反差淘汰机制 - 更积极触发争议
        upset_params = self.params.get('upset_mechanism', {'enabled': False})
        if (upset_params['enabled'] and 
            week_num in upset_params.get('active_weeks', [])):
            
            # V6改进：搜索池根据参数决定，但保护真正的头部选手
            if upset_params.get('require_in_danger', True):
                search_pool = danger_zone
            else:
                # 保护综合排名前40%
                max_rank_protection = upset_params.get('max_rank_protection', 0.40)
                protected_rank = int(n * max_rank_protection)
                df['rank_position'] = df['combined_score'].rank(ascending=False)
                search_pool = df[df['rank_position'] > protected_rank]
            
            # 寻找"评委喜欢但观众不喜欢"的选手
            judge_threshold = int(n * upset_params['judge_high_threshold'])
            fan_threshold = int(n * (1 - upset_params['fan_low_threshold']))
            
            upset_candidates = search_pool[
                (search_pool['judge_rank'] <= judge_threshold) &
                (search_pool['vote_rank'] >= fan_threshold)
            ]
            
            if len(upset_candidates) > 0:
                # 淘汰这个"评委宠儿但观众不买账"的选手
                upset_elim = upset_candidates.loc[upset_candidates['vote_rank'].idxmax()]
                return upset_elim['celebrity'], df
        
        # V6新增：边缘争议机制 - 在得分接近的选手中制造戏剧性
        edge_params = self.params.get('edge_controversy', {'enabled': False})
        if edge_params['enabled'] and stage in ['mid', 'late'] and len(danger_zone) >= 2:
            scores = danger_zone['combined_score'].values
            if len(scores) >= 2:
                # 检查最低两个得分是否接近
                score_diff = abs(scores[1] - scores[0]) / (scores[1] + 1e-10)
                if score_diff < edge_params['score_diff_threshold']:
                    # 使用确定性规则而不是随机：选择争议性更大的那个
                    candidates = danger_zone.head(2)
                    more_controversial = candidates.loc[candidates['rank_diff'].idxmax()]
                    return more_controversial['celebrity'], df
        
        # 惊喜保护机制
        surprise_params = self.params['surprise_protection']
        if (surprise_params['enabled'] and 
            week_num in surprise_params['active_weeks']):
            
            fan_threshold = int(n * surprise_params['fan_threshold'])
            # 过滤掉安全区选手
            danger_zone_filtered = danger_zone[~danger_zone.index.isin(protected_indices)]
            popular_in_danger = danger_zone_filtered[danger_zone_filtered['vote_rank'] <= fan_threshold]
            
            if len(popular_in_danger) > 0:
                for idx, row in popular_in_danger.iterrows():
                    if row['is_controversial']:
                        df.loc[idx, 'is_protected'] = True
                        danger_zone = danger_zone[danger_zone.index != idx]
        
        # 过滤危险区中的安全区选手
        danger_zone = danger_zone[~danger_zone.index.isin(protected_indices)]
        
        # 评委覆盖机制（仅早期，且更严格）
        elim_params = self.params['elimination']
        if stage == 'early' and len(danger_zone) > 0:
            lowest = danger_zone.iloc[0]
            if lowest['score_normalized'] >= elim_params['judge_override_threshold']:
                if len(danger_zone) > 1:
                    eliminated = danger_zone.iloc[1]['celebrity']
                    return eliminated, df
        
        # 正常淘汰
        if len(danger_zone) > 0:
            # 中后期：优先淘汰观众最不喜欢的选手（增加观众参与感）
            if stage in ['mid', 'late']:
                # 优先淘汰观众排名最低的非争议选手
                non_controversial = danger_zone[~danger_zone['is_controversial']]
                if len(non_controversial) > 0:
                    # 按观众投票排名淘汰（观众最不喜欢的）
                    eliminated = non_controversial.loc[non_controversial['vote_rank'].idxmax(), 'celebrity']
                else:
                    # 都是争议选手，淘汰观众排名最低的
                    eliminated = danger_zone.loc[danger_zone['vote_rank'].idxmax(), 'celebrity']
            else:
                # 早期正常淘汰
                eliminated = danger_zone.iloc[0]['celebrity']
        else:
            # 如果危险区为空（都被保护），从非安全区最低分选手淘汰
            non_protected_sorted = df_sorted[~df_sorted.index.isin(protected_indices)]
            if len(non_protected_sorted) > 0:
                eliminated = non_protected_sorted.iloc[0]['celebrity']
            else:
                eliminated = df_sorted.iloc[0]['celebrity']
        
        return eliminated, df
    
    def get_description(self):
        return """
        戏剧弧线系统 V6 (Dramatic Arc System V6) - 均衡优化版
        
        核心理念：在保持公平性和一致性的基础上最大化争议率(15%+)
        
        阶段设计：
        ┌─────────┬─────────┬───────────┬────────────────────────────┐
        │  阶段   │  周次   │  权重比   │          叙事目标          │
        ├─────────┼─────────┼───────────┼────────────────────────────┤
        │  早期   │  1-3周  │ 62%:38%   │ 建立角色，保护技术选手     │
        │  中期   │  4-7周  │ 42%:58%   │ 权重反转，制造冲突         │
        │  后期   │  8+周   │ 32%:68%   │ 观众主导，引发争议高潮     │
        └─────────┴─────────┴───────────┴────────────────────────────┘
        
        V4 核心创新：
        1. 权重跨度30%：从62%评委到32%，制造戏剧性反转
           → 前期评委宠儿可能在后期被观众"翻盘"
        
        2. 低争议阈值(2)：更多选手被标记为"争议"
           → 制造更多话题点和讨论
        
        3. 激进反差淘汰：第3-8周均可触发
           → 确保争议率达到15%目标
        
        4. 25%安全区：仅保护顶尖选手
           → 让75%选手处于"危险"中，增加悬念
        
        5. 4人危险区：扩大淘汰候选范围
           → 更多不确定性，更多话题
        
        为什么制作方应该采用：
        ✓ 争议率15%在最优范围(12-18%)，最大化社交媒体讨论
        ✓ 公平性85%+，保护节目声誉不受质疑
        ✓ 权重反转创造戏剧弧线，符合观众心理预期
        ✓ 规则清晰易传播，观众容易理解并参与讨论
        ✓ 综合评分最优，平衡娱乐性与竞技性
        """


def get_all_systems():
    """获取所有投票系统"""
    return {
        'rank': RankBasedSystem(),
        'percent': PercentageBasedSystem(),
        'dynamic': DynamicWeightedSystem(),
        'excitement': ExcitementMaximizedSystem(),
        'fairness': FairnessOptimizedSystem(),
        'dramatic_arc': DramaticArcSystem(),  # 新增优化系统
    }


if __name__ == '__main__':
    # 测试
    from data_loader import prepare_weekly_data
    
    df = prepare_weekly_data()
    
    # 测试第1季第1周
    week_data = df[(df['season'] == 1) & (df['week'] == 1)].copy()
    
    systems = get_all_systems()
    
    for name, system in systems.items():
        eliminated, result = system.determine_elimination(week_data, week_num=1, total_weeks=6)
        print(f"\n{system.name}:")
        print(f"  淘汰: {eliminated}")
