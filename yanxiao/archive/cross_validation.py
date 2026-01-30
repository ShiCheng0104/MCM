"""
交叉验证模块
实现真正的预测验证，而不是事后解释
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.utils import (
    compute_rank_combined_score,
    compute_percent_combined_score,
    normalize_votes
)


class PredictiveValidator:
    """
    预测验证器
    
    使用历史数据学习投票模式，然后预测新周次的淘汰结果
    """
    
    def __init__(self):
        self.learned_patterns = {}
        self.results = []
        
    def learn_vote_patterns(self, 
                           season_week_data: Dict,
                           elimination_info: pd.DataFrame) -> Dict:
        """
        从历史数据中学习投票模式
        
        学习的模式：
        1. 评分差距与投票差距的关系
        2. 舞伴效应
        3. 行业效应
        4. 赛季效应
        
        Returns:
            学习到的模式参数
        """
        patterns = {
            'score_to_vote_ratio': [],  # 评分差距 -> 投票差距
            'upset_rate_by_score_gap': {},  # 评分差距 -> 冷门概率
            'partner_bonus': {},  # 舞伴加成
            'industry_bonus': {},  # 行业加成
        }
        
        for (season, week), contestants in season_week_data.items():
            if len(contestants) <= 1:
                continue
                
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            elim_name = elim['eliminated_name'].values[0]
            scores = contestants['total_score'].values
            names = contestants['celebrity_name'].values
            
            # 找到被淘汰者
            elim_idx = None
            for i, name in enumerate(names):
                if name == elim_name:
                    elim_idx = i
                    break
            
            if elim_idx is None:
                continue
            
            # 计算评分排名
            score_rank = np.argsort(np.argsort(-scores))  # 0=最高分
            elim_score_rank = score_rank[elim_idx]
            
            # 记录：被淘汰者不是最低分的情况（冷门）
            is_upset = elim_score_rank < len(scores) - 1
            score_gap = scores[elim_idx] - np.min(scores)  # 与最低分的差距
            
            gap_bucket = int(score_gap / 5) * 5  # 每5分一个桶
            if gap_bucket not in patterns['upset_rate_by_score_gap']:
                patterns['upset_rate_by_score_gap'][gap_bucket] = {'upset': 0, 'total': 0}
            patterns['upset_rate_by_score_gap'][gap_bucket]['total'] += 1
            if is_upset:
                patterns['upset_rate_by_score_gap'][gap_bucket]['upset'] += 1
            
            # 学习舞伴效应
            if 'pro_name' in contestants.columns:
                for _, row in contestants.iterrows():
                    partner = row['pro_name']
                    is_elim = row['celebrity_name'] == elim_name
                    if partner not in patterns['partner_bonus']:
                        patterns['partner_bonus'][partner] = {'survived': 0, 'eliminated': 0}
                    if is_elim:
                        patterns['partner_bonus'][partner]['eliminated'] += 1
                    else:
                        patterns['partner_bonus'][partner]['survived'] += 1
            
            # 学习行业效应
            if 'industry' in contestants.columns:
                for _, row in contestants.iterrows():
                    industry = row['industry']
                    is_elim = row['celebrity_name'] == elim_name
                    if industry not in patterns['industry_bonus']:
                        patterns['industry_bonus'][industry] = {'survived': 0, 'eliminated': 0}
                    if is_elim:
                        patterns['industry_bonus'][industry]['eliminated'] += 1
                    else:
                        patterns['industry_bonus'][industry]['survived'] += 1
        
        # 计算舞伴存活率
        for partner, counts in patterns['partner_bonus'].items():
            total = counts['survived'] + counts['eliminated']
            if total > 0:
                patterns['partner_bonus'][partner]['survival_rate'] = counts['survived'] / total
        
        # 计算行业存活率
        for industry, counts in patterns['industry_bonus'].items():
            total = counts['survived'] + counts['eliminated']
            if total > 0:
                patterns['industry_bonus'][industry]['survival_rate'] = counts['survived'] / total
        
        self.learned_patterns = patterns
        return patterns
    
    def predict_elimination(self,
                           contestants: pd.DataFrame,
                           season: int,
                           use_patterns: bool = True) -> Tuple[str, np.ndarray]:
        """
        预测谁会被淘汰
        
        Args:
            contestants: 选手数据
            season: 赛季
            use_patterns: 是否使用学习的模式
        
        Returns:
            (预测被淘汰者名字, 淘汰概率数组)
        """
        scores = contestants['total_score'].values
        names = contestants['celebrity_name'].values
        n = len(contestants)
        
        # 基础：评分越低越容易淘汰
        # 使用softmax将评分转换为存活概率
        score_probs = np.exp(0.1 * scores)
        score_probs = score_probs / score_probs.sum()
        
        # 淘汰概率 = 1 - 存活概率的归一化
        elim_probs = 1 - score_probs
        elim_probs = elim_probs / elim_probs.sum()
        
        if use_patterns and self.learned_patterns:
            adjustments = np.ones(n)
            
            # 舞伴效应调整
            if 'pro_name' in contestants.columns:
                for i, (_, row) in enumerate(contestants.iterrows()):
                    partner = row['pro_name']
                    if partner in self.learned_patterns['partner_bonus']:
                        survival_rate = self.learned_patterns['partner_bonus'][partner].get('survival_rate', 0.5)
                        # 存活率高 -> 淘汰概率低
                        adjustments[i] *= (1 - survival_rate + 0.1)  # 加0.1避免为0
            
            # 行业效应调整
            if 'industry' in contestants.columns:
                for i, (_, row) in enumerate(contestants.iterrows()):
                    industry = row['industry']
                    if industry in self.learned_patterns['industry_bonus']:
                        survival_rate = self.learned_patterns['industry_bonus'][industry].get('survival_rate', 0.5)
                        adjustments[i] *= (1 - survival_rate + 0.1)
            
            # 应用调整
            elim_probs = elim_probs * adjustments
            elim_probs = elim_probs / elim_probs.sum()
        
        # 预测淘汰概率最高的人
        pred_idx = np.argmax(elim_probs)
        pred_name = names[pred_idx]
        
        return pred_name, elim_probs
    
    def cross_validate(self,
                      season_week_data: Dict,
                      elimination_info: pd.DataFrame,
                      verbose: bool = True) -> pd.DataFrame:
        """
        留一法交叉验证
        
        对每个周次：用其他所有周次学习，然后预测该周次
        
        Returns:
            验证结果DataFrame
        """
        results = []
        
        # 获取所有有淘汰的周次
        valid_weeks = []
        for (season, week), contestants in season_week_data.items():
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            if len(elim) > 0 and len(contestants) > 1:
                valid_weeks.append((season, week))
        
        total = len(valid_weeks)
        correct = 0
        correct_with_patterns = 0
        in_bottom_2 = 0
        in_bottom_2_patterns = 0
        
        for i, (test_season, test_week) in enumerate(valid_weeks):
            # 获取测试数据
            test_contestants = season_week_data[(test_season, test_week)]
            test_elim = elimination_info[
                (elimination_info['season'] == test_season) & 
                (elimination_info['week'] == test_week)
            ]
            actual_name = test_elim['eliminated_name'].values[0]
            
            # 用除了当前周次的所有数据学习
            train_data = {k: v for k, v in season_week_data.items() if k != (test_season, test_week)}
            self.learn_vote_patterns(train_data, elimination_info)
            
            # 预测（不用模式）
            pred_name_baseline, probs_baseline = self.predict_elimination(
                test_contestants, test_season, use_patterns=False
            )
            
            # 预测（用模式）
            pred_name_patterns, probs_patterns = self.predict_elimination(
                test_contestants, test_season, use_patterns=True
            )
            
            # 检查准确性
            is_correct_baseline = (pred_name_baseline == actual_name)
            is_correct_patterns = (pred_name_patterns == actual_name)
            
            # 检查是否在底2
            names = test_contestants['celebrity_name'].values
            top2_baseline = names[np.argsort(-probs_baseline)[:2]]
            top2_patterns = names[np.argsort(-probs_patterns)[:2]]
            in_bottom_2_baseline = actual_name in top2_baseline
            in_bottom_2_patterns_flag = actual_name in top2_patterns
            
            if is_correct_baseline:
                correct += 1
            if is_correct_patterns:
                correct_with_patterns += 1
            if in_bottom_2_baseline:
                in_bottom_2 += 1
            if in_bottom_2_patterns_flag:
                in_bottom_2_patterns += 1
            
            results.append({
                'season': test_season,
                'week': test_week,
                'actual': actual_name,
                'pred_baseline': pred_name_baseline,
                'pred_patterns': pred_name_patterns,
                'correct_baseline': is_correct_baseline,
                'correct_patterns': is_correct_patterns,
                'in_bottom_2_baseline': in_bottom_2_baseline,
                'in_bottom_2_patterns': in_bottom_2_patterns_flag
            })
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{total}")
        
        if verbose:
            print(f"\n交叉验证结果:")
            print(f"  总周次: {total}")
            print(f"  基线准确率: {correct/total:.2%}")
            print(f"  模式准确率: {correct_with_patterns/total:.2%}")
            print(f"  基线底2率: {in_bottom_2/total:.2%}")
            print(f"  模式底2率: {in_bottom_2_patterns/total:.2%}")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def simple_baseline_test(self,
                            season_week_data: Dict,
                            elimination_info: pd.DataFrame) -> Dict:
        """
        简单基线测试：仅用评分预测淘汰
        
        预测规则：评分最低者被淘汰
        
        Returns:
            准确率统计
        """
        correct = 0
        total = 0
        in_bottom_2 = 0
        
        for (season, week), contestants in season_week_data.items():
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0 or len(contestants) <= 1:
                continue
            
            actual_name = elim['eliminated_name'].values[0]
            scores = contestants['total_score'].values
            names = contestants['celebrity_name'].values
            
            # 预测：评分最低者被淘汰
            pred_idx = np.argmin(scores)
            pred_name = names[pred_idx]
            
            # 底2
            bottom_2_idx = np.argsort(scores)[:2]
            bottom_2_names = names[bottom_2_idx]
            
            total += 1
            if pred_name == actual_name:
                correct += 1
            if actual_name in bottom_2_names:
                in_bottom_2 += 1
        
        result = {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0,
            'in_bottom_2': in_bottom_2,
            'bottom_2_rate': in_bottom_2 / total if total > 0 else 0
        }
        
        print(f"\n简单基线（仅评分）:")
        print(f"  总周次: {total}")
        print(f"  准确率: {result['accuracy']:.2%}")
        print(f"  底2率: {result['bottom_2_rate']:.2%}")
        
        return result
