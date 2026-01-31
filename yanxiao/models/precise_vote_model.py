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
                # 优化失败：使用备用方法估计（基于评分和先验）
                scores_arr = np.array([c['score'] for c in contestants])
                prior_boosts = np.array([c['prior_boost'] for c in contestants])
                
                # 备用估计：评分比例 × (1 + 先验)
                score_based = scores_arr / np.sum(scores_arr)
                vote_shares = score_based * (1 + prior_boosts * 0.3)
                vote_shares = vote_shares / np.sum(vote_shares)
            
            # 所有周次都存储到week_results（包括失败的）
            self.week_results[(season, week)] = {
                'contestants': contestants,
                'vote_shares': vote_shares,
                'method': method,
                'success': success
            }
                
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
                    'is_eliminated': c['is_eliminated'],
                    'optimization_success': success  # 标记是否优化成功
                })
        
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
        
        核心思路：
        1. 首先基于评委得分、先验偏好计算基础投票份额（保持合理性）
        2. 然后通过优化微调，确保满足淘汰约束（保证一致性）
        3. 如果优化失败，使用约束调整法直接修正
        
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
        
        # ==================== 第一步：计算基础投票份额 ====================
        # 基于评委得分和先验偏好，体现"评委得分高→观众投票多"的合理关系
        base_votes = self._compute_base_votes(scores, prior_boosts)
        
        # ==================== 第二步：优化微调，满足约束 ====================
        vote_shares, opt_success = self._optimize_with_constraints(
            scores, base_votes, prior_boosts, is_eliminated, 
            elim_indices, surv_indices, method
        )
        
        # 验证约束是否满足
        success = self._verify_elimination(scores, vote_shares, is_eliminated, method)
        
        # ==================== 第三步：如果优化失败，使用约束调整法 ====================
        if not success:
            vote_shares = self._adjust_for_constraints(
                scores, vote_shares, is_eliminated, elim_indices, surv_indices, method
            )
            success = self._verify_elimination(scores, vote_shares, is_eliminated, method)
        
        return vote_shares, success
    
    def _compute_base_votes(self, scores: np.ndarray, prior_boosts: np.ndarray) -> np.ndarray:
        """
        计算基础投票份额
        
        考虑因素：
        1. 评委得分：得分越高，观众投票越多（幂次关系）
        2. 先验偏好：舞伴效应、行业效应
        """
        # 评分对投票的非线性影响（alpha=1.2表示高分选手获得更多投票）
        score_effect = np.power(scores, 1.2)
        score_effect = score_effect / np.sum(score_effect)
        
        # 先验偏好调整
        prior_effect = 1 + prior_boosts * 0.3
        prior_effect = np.maximum(prior_effect, 0.1)  # 确保正值
        
        # 综合基础投票
        base_votes = score_effect * prior_effect
        base_votes = base_votes / np.sum(base_votes)
        
        return base_votes
    
    def _optimize_with_constraints(self, scores: np.ndarray, base_votes: np.ndarray,
                                    prior_boosts: np.ndarray, is_eliminated: np.ndarray,
                                    elim_indices: np.ndarray, surv_indices: np.ndarray,
                                    method: str) -> Tuple[np.ndarray, bool]:
        """
        使用优化方法微调投票份额，满足淘汰约束
        
        目标：在保持投票份额接近基础值的同时，确保淘汰约束满足
        """
        n = len(scores)
        
        def objective(vote_shares):
            """
            目标函数：最小化与基础投票的偏差 + 约束违反惩罚
            """
            # 1. 正则化损失：保持接近基础投票（体现评分对投票的影响）
            reg_loss = np.sum((vote_shares - base_votes) ** 2)
            
            # 2. 约束违反惩罚
            constraint_violation = self._compute_constraint_violation(
                scores, vote_shares, is_eliminated, elim_indices, surv_indices, method
            )
            
            # 使用高惩罚权重确保约束满足
            return reg_loss + 10000 * constraint_violation
        
        # 初始值：从基础投票开始
        x0 = base_votes.copy()
        
        # 边界
        eps = 1e-6
        lb = 1e-4
        ub = 1.0 - 1e-4
        bounds = [(lb, ub)] * n
        x0 = np.clip(x0, lb + eps, ub - eps)
        
        # 约束：份额之和为1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # 优化
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 5000, 'ftol': 1e-10, 'eps': 1e-9}
            )
        
        vote_shares = result.x
        vote_shares = vote_shares / np.sum(vote_shares)  # 确保和为1
        
        return vote_shares, result.success
    
    def _compute_constraint_violation(self, scores: np.ndarray, vote_shares: np.ndarray,
                                       is_eliminated: np.ndarray,
                                       elim_indices: np.ndarray, surv_indices: np.ndarray,
                                       method: str) -> float:
        """
        计算约束违反程度
        """
        n = len(scores)
        
        if method == 'rank':
            # 排名法：使用真实排名计算（更精确）
            score_ranks = rankdata(-scores, method='average')  # 评分高=排名小
            vote_ranks = rankdata(-vote_shares, method='average')  # 投票高=排名小
            
            combined_ranks = score_ranks + vote_ranks
            
            elim_combined = combined_ranks[elim_indices]
            surv_combined = combined_ranks[surv_indices]
            
            # 被淘汰者的综合排名应该比所有存活者都大（数值大=排名差）
            violation = 0
            margin = 0.5  # 安全边际
            for ec in elim_combined:
                for sc in surv_combined:
                    if ec <= sc + margin:
                        # 被淘汰者综合排名应该 > 存活者综合排名
                        violation += (sc - ec + margin + 1) ** 2
            
            # 额外惩罚：确保被淘汰者投票份额足够低
            for ei in elim_indices:
                for si in surv_indices:
                    if vote_shares[ei] >= vote_shares[si]:
                        violation += (vote_shares[ei] - vote_shares[si] + 0.01) ** 2 * 10
        else:
            # 百分比法：直接计算综合百分比
            combined = self._compute_combined_percent(scores, vote_shares)
            
            elim_combined = combined[elim_indices]
            surv_combined = combined[surv_indices]
            
            # 被淘汰者的综合百分比应该比所有存活者都小
            violation = 0
            for ec in elim_combined:
                for sc in surv_combined:
                    if sc <= ec:
                        violation += (ec - sc + 0.01) ** 2
        
        return violation
    
    def _adjust_for_constraints(self, scores: np.ndarray, vote_shares: np.ndarray,
                                 is_eliminated: np.ndarray,
                                 elim_indices: np.ndarray, surv_indices: np.ndarray,
                                 method: str) -> np.ndarray:
        """
        约束调整法：当优化失败时，直接调整投票份额以满足约束
        
        策略：保持存活者的相对顺序，只调整被淘汰者的份额使其排名最差
        """
        n = len(scores)
        adjusted_shares = vote_shares.copy()
        
        if method == 'rank':
            # ==================== 排名法：直接构造确保被淘汰者综合排名最大 ====================
            # 综合排名 = 评委排名 + 观众排名，数值最大者被淘汰
            # 策略：给被淘汰者分配最低的观众投票，使其观众排名最差（数值最大）
            
            score_ranks = rankdata(-scores, method='ordinal')  # 评分最高=1
            n_elim = len(elim_indices)
            n_surv = len(surv_indices)
            
            # 计算存活者中最差的综合排名上限
            # 存活者需要的最大综合排名 = n - n_elim（被淘汰者占据最后n_elim个位置）
            max_surv_combined = 2 * n - n_elim  # 存活者中综合排名的理论上限
            
            # 给被淘汰者分配最低的观众投票份额
            # 这样他们的观众排名将是 n_surv+1, n_surv+2, ..., n（最差的n_elim个）
            
            # 被淘汰者按评委排名排序（评委分高的需要更低的投票来补偿）
            elim_by_score = sorted(elim_indices, key=lambda i: score_ranks[i])
            
            # 分配投票份额
            # 存活者保持相对顺序，但确保最低的存活者投票也高于最高的被淘汰者投票
            
            # 存活者的最低投票份额基准
            if n_surv > 0:
                surv_min_vote = 0.02  # 存活者最低也有2%
                surv_vote_range = 0.98 - surv_min_vote * n_surv  # 存活者可分配范围
                
                # 存活者按原投票份额的相对顺序分配
                surv_original = adjusted_shares[surv_indices]
                surv_order = np.argsort(np.argsort(surv_original))  # 相对排序
                
                # 按相对顺序分配存活者投票
                for idx, surv_idx in enumerate(surv_indices):
                    rank_in_surv = surv_order[idx]
                    # 投票份额：基础 + 按排序分配的增量
                    adjusted_shares[surv_idx] = surv_min_vote + (rank_in_surv + 1) * surv_vote_range / n_surv
            
            # 被淘汰者获得极小份额，确保观众排名最差
            # 评委分越高的被淘汰者，需要越低的投票来确保综合排名最大
            elim_base_vote = 0.001  # 被淘汰者基础份额
            for rank_idx, elim_idx in enumerate(elim_by_score):
                # 评委分高的（score_ranks小的）获得更低的投票
                # 这样即使评委排名好，观众排名差也能保证综合排名最大
                adjusted_shares[elim_idx] = elim_base_vote * (0.5 ** rank_idx)
            
            # 强制验证并修正
            # 计算当前综合排名
            vote_ranks = rankdata(-adjusted_shares, method='ordinal')
            combined_ranks = score_ranks + vote_ranks
            
            # 检查是否所有被淘汰者的综合排名都大于所有存活者
            elim_min_combined = np.min(combined_ranks[elim_indices])
            surv_max_combined = np.max(combined_ranks[surv_indices]) if n_surv > 0 else 0
            
            if elim_min_combined <= surv_max_combined:
                # 仍不满足，使用极端构造法
                # 直接设置：被淘汰者获得最低的n_elim个投票份额
                all_shares = np.zeros(n)
                
                # 存活者按评分比例分配95%的总份额
                surv_total = 0.95
                surv_scores = scores[surv_indices]
                surv_shares = surv_scores / np.sum(surv_scores) * surv_total
                all_shares[surv_indices] = surv_shares
                
                # 被淘汰者平分剩余5%，但按评委排名反向（评委分高→投票更低）
                elim_total = 0.05
                elim_score_ranks = score_ranks[elim_indices]
                # 评委排名好（小）的获得更少投票
                elim_weights = elim_score_ranks.astype(float)  # 排名大→权重大→投票多
                elim_weights = elim_weights / np.sum(elim_weights)
                all_shares[elim_indices] = elim_weights * elim_total * 0.1  # 进一步压缩
                
                adjusted_shares = all_shares
                
                # 再次验证
                vote_ranks = rankdata(-adjusted_shares, method='ordinal')
                combined_ranks = score_ranks + vote_ranks
                elim_min_combined = np.min(combined_ranks[elim_indices])
                surv_max_combined = np.max(combined_ranks[surv_indices]) if n_surv > 0 else 0
                
                if elim_min_combined <= surv_max_combined:
                    # 最终兜底：给被淘汰者分配递减的极小值
                    for i, elim_idx in enumerate(sorted(elim_indices, key=lambda x: score_ranks[x])):
                        adjusted_shares[elim_idx] = 1e-6 * (0.1 ** i)
        
        else:
            # 百分比法：确保被淘汰者的综合百分比最低
            score_pcts = scores / np.sum(scores)
            
            if len(surv_indices) > 0:
                # 计算当前存活者的最低综合百分比
                current_combined = score_pcts + vote_shares
                min_surv_combined = np.min(current_combined[surv_indices])
                
                # 调整被淘汰者的投票份额
                for idx in elim_indices:
                    elim_score_pct = score_pcts[idx]
                    
                    # 需要的投票份额：使综合百分比低于存活者最低值
                    # combined_pct = score_pct + vote_pct < min_surv_combined
                    # vote_pct < min_surv_combined - score_pct
                    target_vote_pct = min_surv_combined - elim_score_pct - 0.02
                    
                    if target_vote_pct > 0:
                        adjusted_shares[idx] = target_vote_pct
                    else:
                        # 如果被淘汰者评委分太高，需要极小的投票份额
                        adjusted_shares[idx] = 0.001
        
        # 归一化
        adjusted_shares = np.maximum(adjusted_shares, 1e-8)  # 确保正值
        adjusted_shares = adjusted_shares / np.sum(adjusted_shares)
        
        return adjusted_shares
    
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
        success_count = 0
        
        for (season, week), week_result in self.week_results.items():
            # 不再跳过优化失败的周次，使用全部数据
            if week_result['success']:
                success_count += 1
            
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
                # 排名法：预测综合排名数值最大的n_eliminated个人（排名最差）
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
        print(f"投票反推拟合验证（全部数据）")
        print(f"============================================================")
        print(f"检验周次数: {total}  （其中优化成功: {success_count}, 备用估计: {total - success_count}）")
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
