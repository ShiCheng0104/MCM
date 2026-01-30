"""
精确投票反推模型

严格按照题目规则：
1. 排名法 (赛季1-2, 28-34): 评委排名 + 观众排名 = 综合排名，最高者淘汰
2. 百分比法 (赛季3-27): 评委百分比 + 观众百分比 = 综合百分比，最低者淘汰

核心思路：
- 已知：评委得分、谁被淘汰
- 求解：观众投票（使被淘汰者的综合得分最低/排名最高）
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution, LinearConstraint
from scipy.stats import rankdata
import warnings

# 赛季分类
SEASONS_RANK_METHOD = list(range(1, 3)) + list(range(28, 35))  # 排名法赛季
SEASONS_PERCENT_METHOD = list(range(3, 28))  # 百分比法赛季

# 导入基线模型用于无淘汰周次的预测
try:
    from .baseline_model import BaselineModel
    BASELINE_AVAILABLE = True
except ImportError:
    BASELINE_AVAILABLE = False
    print("Warning: baseline_model not available, will use simple estimation for non-elimination weeks")


class PreciseVoteModel:
    """
    精确投票反推模型
    
    关键：根据淘汰结果反推出满足约束的观众投票
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 基线模型用于无淘汰周次预测
        if BASELINE_AVAILABLE:
            self.baseline_model = BaselineModel(alpha=1.2, noise_level=0.15)
        else:
            self.baseline_model = None
        
        # 学习到的投票偏好因子
        self.partner_effects = {}
        self.industry_effects = {}
        
        # 全局参数
        self.base_vote_share = 0.5  # 基础投票份额
        self.score_influence = 0.3  # 评分对投票的影响
        
        # 标准化参数
        self.age_mean = 0
        self.age_std = 1
        
        # 结果存储
        self.results_df = None
        self.week_results = {}
        self.is_fitted = False
        
    def fit(self, weekly_data: pd.DataFrame, elimination_info: pd.DataFrame):
        """训练模型"""
        print("正在训练精确投票反推模型...")
        
        # 1. 学习投票偏好（舞伴/行业效应）
        self._learn_vote_preferences(weekly_data, elimination_info)
        
        # 2. 对每个周次反推投票
        self._solve_all_weeks(weekly_data, elimination_info)
        
        # 注: 准确率验证在 predict_elimination() 中单独进行
        
        self.is_fitted = True
        print("模型训练完成!")
        
    def _learn_vote_preferences(self, weekly_data: pd.DataFrame, 
                                elimination_info: pd.DataFrame):
        """
        学习投票偏好：哪些舞伴/行业能获得更多投票支持
        
        方法：统计各舞伴/行业的"超预期存活率"
        """
        print("  学习投票偏好...")
        
        # 标准化年龄
        ages = weekly_data['celebrity_age'].dropna()
        self.age_mean = ages.mean()
        self.age_std = ages.std() if ages.std() > 0 else 1
        
        partner_stats = {}  # {partner: [survive_boost, ...]}
        industry_stats = {}
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            eliminated_names = set(elim['eliminated_name'].tolist())
            n = len(group)
            
            # 计算评分排名
            scores = group['total_score'].values
            score_ranks = n - rankdata(scores, method='ordinal') + 1  # 1=最高分
            
            for idx, (_, row) in enumerate(group.iterrows()):
                name = row['celebrity_name']
                partner = row.get('ballroom_partner', 'Unknown')
                industry = row.get('celebrity_industry', 'Unknown')
                score_rank = score_ranks[idx]
                
                # 超预期存活：评分低但没被淘汰
                # 负向超预期：评分高但被淘汰
                expected_elim_rank = n  # 预期最低分被淘汰
                
                if name in eliminated_names:
                    # 被淘汰：计算"提前淘汰程度"
                    boost = -(n - score_rank) / n  # 评分越高，被淘汰越意外
                else:
                    # 存活：计算"超预期存活程度"
                    boost = (score_rank - 1) / n  # 评分越低，存活越意外
                
                # 记录
                if partner not in partner_stats:
                    partner_stats[partner] = []
                partner_stats[partner].append(boost)
                
                if industry not in industry_stats:
                    industry_stats[industry] = []
                industry_stats[industry].append(boost)
        
        # 计算平均效应
        overall_mean = 0
        for partner, boosts in partner_stats.items():
            self.partner_effects[partner] = np.mean(boosts) - overall_mean
        
        for industry, boosts in industry_stats.items():
            self.industry_effects[industry] = np.mean(boosts) - overall_mean
        
        # 打印效应
        sorted_partners = sorted(self.partner_effects.items(), key=lambda x: x[1], reverse=True)
        print(f"    舞伴效应 (Top 5, 正值=更多投票支持):")
        for p, e in sorted_partners[:5]:
            print(f"      {p}: {e:+.3f}")
        
        sorted_industries = sorted(self.industry_effects.items(), key=lambda x: x[1], reverse=True)
        print(f"    行业效应 (Top 5):")
        for ind, e in sorted_industries[:5]:
            print(f"      {ind}: {e:+.3f}")
    
    def _solve_all_weeks(self, weekly_data: pd.DataFrame, 
                         elimination_info: pd.DataFrame):
        """对每个周次反推投票"""
        print("  反推各周次投票...")
        
        results = []
        success_count = 0
        total_count = 0        
        failed_weeks = []  # 记录失败的周次        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                # 无淘汰周次：使用基线模型 + 先验偏好预测投票
                method = 'rank' if season in SEASONS_RANK_METHOD else 'percent'
                
                # 准备选手数据
                contestants_no_elim = []
                for _, row in group.iterrows():
                    partner = row.get('ballroom_partner', 'Unknown')
                    industry = row.get('celebrity_industry', 'Unknown')
                    prior_boost = (
                        self.partner_effects.get(partner, 0) +
                        self.industry_effects.get(industry, 0)
                    )
                    contestants_no_elim.append({
                        'name': row['celebrity_name'],
                        'score': row['total_score'],
                        'prior_boost': prior_boost
                    })
                
                scores = np.array([c['score'] for c in contestants_no_elim])
                prior_boosts = np.array([c['prior_boost'] for c in contestants_no_elim])
                
                # 方法1：基于评分的幂次关系（alpha=1.2）
                if self.baseline_model is not None:
                    base_votes = self.baseline_model.estimate_votes(scores, total_votes=1_000_000)
                    base_shares = base_votes / 1_000_000
                else:
                    base_votes_raw = np.power(scores, 1.2)
                    base_shares = base_votes_raw / np.sum(base_votes_raw)
                
                # 方法2：结合先验偏好
                score_based = scores / np.sum(scores)
                prior_adjusted = score_based * (1 + prior_boosts * 0.3)
                prior_adjusted = prior_adjusted / np.sum(prior_adjusted)
                
                # 混合：基线70% + 先验30%
                vote_shares_est = 0.7 * base_shares + 0.3 * prior_adjusted
                vote_shares_est = vote_shares_est / np.sum(vote_shares_est)
                
                total_votes = 1_000_000
                for i, c in enumerate(contestants_no_elim):
                    results.append({
                        'season': season,
                        'week': week,
                        'celebrity': c['name'],
                        'total_score': c['score'],
                        'estimated_votes': vote_shares_est[i] * total_votes,
                        'vote_share': vote_shares_est[i],
                        'method': method,
                        'is_eliminated': False
                    })
                continue
            
            eliminated_names = set(elim['eliminated_name'].tolist())
            method = 'rank' if season in SEASONS_RANK_METHOD else 'percent'
            
            # 准备数据
            contestants = []
            for _, row in group.iterrows():
                partner = row.get('ballroom_partner', 'Unknown')
                industry = row.get('celebrity_industry', 'Unknown')
                
                # 先验投票偏好
                prior_boost = (
                    self.partner_effects.get(partner, 0) +
                    self.industry_effects.get(industry, 0)
                )
                
                contestants.append({
                    'name': row['celebrity_name'],
                    'score': row['total_score'],
                    'is_eliminated': row['celebrity_name'] in eliminated_names,
                    'prior_boost': prior_boost,
                    'partner': partner,
                    'industry': industry
                })
            
            # 检查淘汰者是否在选手中
            valid_eliminated = [c for c in contestants if c['is_eliminated']]
            if not valid_eliminated:
                # 无淘汰周次：使用多种方法预测投票
                scores = np.array([c['score'] for c in contestants])
                prior_boosts = np.array([c['prior_boost'] for c in contestants])
                
                # 方法1：基于评分的幂次关系（alpha=1.2模拟评分对投票的非线性影响）
                if self.baseline_model is not None:
                    base_votes = self.baseline_model.estimate_votes(scores, total_votes=1_000_000)
                    base_shares = base_votes / 1_000_000
                else:
                    # 降级方案：评分的1.2次幂
                    base_votes_raw = np.power(scores, 1.2)
                    base_shares = base_votes_raw / np.sum(base_votes_raw)
                
                # 方法2：结合先验偏好调整
                score_based = scores / np.sum(scores)
                prior_adjusted = score_based * (1 + prior_boosts * 0.3)
                prior_adjusted = prior_adjusted / np.sum(prior_adjusted)
                
                # 混合两种方法：基线模型70% + 先验调整30%
                vote_shares_est = 0.7 * base_shares + 0.3 * prior_adjusted
                vote_shares_est = vote_shares_est / np.sum(vote_shares_est)
                
                total_votes = 1_000_000
                for i, c in enumerate(contestants):
                    results.append({
                        'season': season,
                        'week': week,
                        'celebrity': c['name'],
                        'total_score': c['score'],
                        'estimated_votes': vote_shares_est[i] * total_votes,
                        'vote_share': vote_shares_est[i],
                        'method': method,
                        'is_eliminated': False  # 无淘汰
                    })
                continue
            
            # 反推投票
            vote_shares, success = self._solve_votes_for_week(contestants, method)
            
            if success:
                success_count += 1
            else:
                failed_weeks.append((season, week))
            total_count += 1
            
            # 假设总投票100万
            total_votes = 1_000_000
            
            for i, c in enumerate(contestants):
                votes = vote_shares[i] * total_votes if vote_shares is not None else np.nan
                results.append({
                    'season': season,
                    'week': week,
                    'celebrity': c['name'],
                    'total_score': c['score'],
                    'estimated_votes': votes,
                    'vote_share': vote_shares[i] if vote_shares is not None else np.nan,
                    'method': method,
                    'is_eliminated': c['is_eliminated']
                })
            
            # 存储周结果
            self.week_results[(season, week)] = {
                'contestants': contestants,
                'vote_shares': vote_shares,
                'method': method,
                'success': success
            }
        
        print(f"    反推成功: {success_count}/{total_count} 周次")
        if failed_weeks:
            print(f"    优化失败的周次 ({len(failed_weeks)}): {failed_weeks[:5]}{'...' if len(failed_weeks) > 5 else ''}")
        
        self.results_df = pd.DataFrame(results)
        
        # 统计有效估计数
        valid_estimates = self.results_df.groupby(['season', 'week']).first()
        has_valid_votes = valid_estimates['estimated_votes'].notna().sum()
        print(f"    有效投票估计: {has_valid_votes} 个周次（包括优化失败但给出估计的）")
    
    def _solve_votes_for_week(self, contestants: List[Dict], method: str) -> Tuple[np.ndarray, bool]:
        """
        为单周反推投票份额
        
        约束：被淘汰者的综合得分必须是最低的（排名法）或最低的（百分比法）
        """
        n = len(contestants)
        scores = np.array([c['score'] for c in contestants])
        prior_boosts = np.array([c['prior_boost'] for c in contestants])
        is_eliminated = np.array([c['is_eliminated'] for c in contestants])
        
        # 被淘汰者索引
        elim_indices = np.where(is_eliminated)[0]
        surv_indices = np.where(~is_eliminated)[0]
        
        if len(elim_indices) == 0:
            return None, False
        
        def objective(vote_shares):
            """
            目标：使投票份额接近先验，同时满足淘汰约束
            """
            # 正则化：投票份额应该接近评分归一化后的值（加上先验偏好）
            score_based = scores / np.sum(scores)
            prior_votes = score_based * (1 + prior_boosts)
            prior_votes = prior_votes / np.sum(prior_votes)
            
            # L2正则化损失
            reg_loss = np.sum((vote_shares - prior_votes) ** 2)
            
            # 淘汰约束违反惩罚
            if method == 'rank':
                combined = self._compute_combined_rank(scores, vote_shares, n)
                # 排名法：被淘汰者的combined_rank应该最大（排名最差）
                elim_combined = combined[elim_indices]
                surv_combined = combined[surv_indices]
                
                # 惩罚：确保所有被淘汰者的排名都大于所有存活者
                # 排名数值越大 = 排名越差
                constraint_violation = 0
                elim_min = np.min(elim_combined)  # 被淘汰者中最好的排名
                surv_max = np.max(surv_combined) if len(surv_combined) > 0 else 0  # 存活者中最差的排名
                
                if elim_min <= surv_max:  # 如果被淘汰者中有人排名不是最差
                    # 需要拉大差距
                    constraint_violation = (surv_max - elim_min + 1.0) ** 2
            else:
                combined = self._compute_combined_percent(scores, vote_shares)
                # 百分比法：被淘汰者的combined_percent应该最小
                elim_combined = combined[elim_indices]
                surv_combined = combined[surv_indices]
                
                # 惩罚：如果有存活者的combined_percent <= 被淘汰者
                constraint_violation = 0
                for ec in elim_combined:
                    for sc in surv_combined:
                        if sc <= ec:  # 存活者百分比更低是错误的
                            constraint_violation += (ec - sc + 0.01) ** 2
            
            return reg_loss + 1000 * constraint_violation
        
        # 初始值：基于评分和先验偏好
        score_based = scores / np.sum(scores)
        x0 = score_based * (1 + prior_boosts * 0.5)
        x0 = x0 / np.sum(x0)

        # 边界：每个份额在[lb, ub]之间，使用小的epsilon避免触及边界
        eps = 1e-6
        lb = 1e-3
        ub = 1.0 - 1e-3
        bounds = [(lb, ub)] * n

        # 确保初始值严格在边界内部，避免优化器在初始步骤就裁剪导致警告
        x0 = np.clip(x0, lb + eps, ub - eps)
        
        # 约束：份额之和为1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-8}
        )
        
        vote_shares = result.x
        vote_shares = vote_shares / np.sum(vote_shares)  # 确保和为1
        
        # 验证约束是否满足
        success = self._verify_elimination(scores, vote_shares, is_eliminated, method)
        
        return vote_shares, success
    
    def _compute_combined_rank(self, scores: np.ndarray, vote_shares: np.ndarray, n: int) -> np.ndarray:
        """
        计算排名法的综合排名
        
        规则：综合排名 = 评委排名 + 观众排名
        
        排名解释：
        - 排名1 = 第一名（最好）
        - 排名越大 = 名次越差
        - 综合排名数值最大者被淘汰（排名最低/最差）
        
        示例：
        选手A: 评分排名3 + 投票排名2 = 综合排名5
        选手B: 评分排名1 + 投票排名5 = 综合排名6 ← 被淘汰（数值最大）
        选手C: 评分排名2 + 投票排名1 = 综合排名3
        """
        # 评委排名：分数越高排名越好（数值越小）
        score_ranks = rankdata(-scores, method='ordinal')  # 最高分=1, 最低分=n
        
        # 观众排名：投票越多排名越好（数值越小）
        vote_ranks = rankdata(-vote_shares, method='ordinal')  # 最高票=1, 最低票=n
        
        # 综合排名（数值越大 = 排名越差 = 越可能被淘汰）
        combined_rank = score_ranks + vote_ranks
        
        return combined_rank
    
    def _compute_combined_percent(self, scores: np.ndarray, vote_shares: np.ndarray) -> np.ndarray:
        """
        计算百分比法的综合百分比
        
        综合百分比 = 评委百分比 + 观众百分比（越小越可能被淘汰）
        """
        # 评委百分比
        score_percent = scores / np.sum(scores)
        
        # 观众百分比 = vote_shares（已经是百分比）
        
        # 综合百分比
        combined_percent = score_percent + vote_shares
        
        return combined_percent
    
    def _verify_elimination(self, scores: np.ndarray, vote_shares: np.ndarray, 
                           is_eliminated: np.ndarray, method: str) -> bool:
        """验证淘汰约束是否满足"""
        n = len(scores)
        
        if method == 'rank':
            combined = self._compute_combined_rank(scores, vote_shares, n)
            # 被淘汰者的综合排名应该是最大的（数值最大 = 排名最差）
            # 处理多人淘汰：确保所有被淘汰者的排名都大于所有存活者
            elim_min = np.min(combined[is_eliminated])  # 被淘汰者中最好的
            surv_max = np.max(combined[~is_eliminated]) if np.any(~is_eliminated) else 0  # 存活者中最差的
            return elim_min > surv_max  # 所有被淘汰者都比所有存活者差
        else:
            combined = self._compute_combined_percent(scores, vote_shares)
            # 被淘汰者的综合百分比应该是最小的
            elim_min = np.min(combined[is_eliminated])
            surv_min = np.min(combined[~is_eliminated]) if np.any(~is_eliminated) else float('inf')
            return elim_min < surv_min
    
    def _validate_predictions(self, weekly_data: pd.DataFrame, 
                             elimination_info: pd.DataFrame):
        """验证预测准确率"""
        correct = 0
        bottom_n = 0
        total = 0
        
        for (season, week), week_result in self.week_results.items():
            if not week_result['success']:
                continue
            
            contestants = week_result['contestants']
            vote_shares = week_result['vote_shares']
            method = week_result['method']
            
            scores = np.array([c['score'] for c in contestants])
            is_eliminated = np.array([c['is_eliminated'] for c in contestants])
            n = len(contestants)
            n_eliminated = np.sum(is_eliminated)
            
            # 计算综合得分
            if method == 'rank':
                combined = self._compute_combined_rank(scores, vote_shares, n)
                # 排名法：预浌综合排名数值最大的n_eliminated个人（排名最差）
                # np.argsort返回从小到大的索引，取最后 n_eliminated 个
                pred_indices = np.argsort(combined)[-n_eliminated:]
            else:
                combined = self._compute_combined_percent(scores, vote_shares)
                # 预测综合百分比最低的为淘汰者
                pred_indices = np.argsort(combined)[:n_eliminated]
            
            pred_eliminated = set(contestants[i]['name'] for i in pred_indices)
            actual_eliminated = set(c['name'] for c in contestants if c['is_eliminated'])
            
            if pred_eliminated == actual_eliminated:
                correct += 1
            
            # 检查是否在底部
            if method == 'rank':
                bottom_indices = np.argsort(combined)[-max(2, n_eliminated):]
            else:
                bottom_indices = np.argsort(combined)[:max(2, n_eliminated)]
            
            bottom_names = set(contestants[i]['name'] for i in bottom_indices)
            if actual_eliminated.issubset(bottom_names):
                bottom_n += 1
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        bottom_accuracy = bottom_n / total if total > 0 else 0
        print(f"\n============================================================")
        print(f"投票反推拟合验证（训练集）")
        print(f"============================================================")
        print(f"检验周次数: {total}  （仅优化成功的周次）")
        print(f"正确反推数: {correct}")
        print(f"反推准确率: {accuracy:.2%}")
        print(f"底N准确率: {bottom_accuracy:.2%}")
        print(f"")
        print(f"注: 此准确率衡量模型对已知淘汰结果的拟合能力")
        print(f"    (仅统计约束优化成功的周次)")
        print(f"============================================================")

        # 返回结果供外部使用
        return {
            'accuracy': accuracy,
            'bottom_accuracy': bottom_accuracy,
            'total': total,
            'correct': correct,
            'bottom_correct': bottom_n
        }
    
    def predict_elimination(self, weekly_data: pd.DataFrame, 
                           elimination_info: pd.DataFrame) -> Dict:
        """预测淘汰结果（用于验证）"""
        return self._validate_predictions(weekly_data, elimination_info)
    
    def get_vote_estimates(self) -> pd.DataFrame:
        """返回投票估计结果"""
        return self.results_df
    
    def get_estimates_dict(self) -> Dict:
        """返回字典格式的估计结果"""
        estimates = {}
        
        for (season, week), group in self.results_df.groupby(['season', 'week']):
            names = group['celebrity'].tolist()
            scores = group['total_score'].tolist()
            votes = group['estimated_votes'].tolist()
            
            estimates[(season, week)] = {
                'names': names,
                'scores': scores,
                'votes': votes
            }
        
        return estimates
    
    def get_samples_dict(self) -> Dict:
        """返回样本字典"""
        samples_dict = {}
        
        for _, row in self.results_df.iterrows():
            if pd.isna(row['estimated_votes']):
                continue
            key = (row['season'], row['week'], row['celebrity'])
            votes = row['estimated_votes']
            cv = 0.1 + 0.1 * (1 - row.get('vote_share', 0.5))
            samples = np.random.normal(votes, votes * cv, 100)
            samples = np.maximum(samples, 0)
            samples_dict[key] = samples
        
        return samples_dict
